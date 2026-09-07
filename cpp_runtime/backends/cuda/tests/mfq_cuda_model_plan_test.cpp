#include "mfq_cuda_model_plan.h"

#include <iostream>
#include <stdexcept>

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

mfq::MfqModelGraph graph_with(
        std::string backbone,
        std::string vision = {},
        std::string predictor = {}) {
    mfq::MfqModelGraph graph;
    graph.backbone = std::move(backbone);
    graph.components.push_back({
        "text", "model", graph.backbone, "decoder", {}, {}, {}});
    if (!vision.empty()) {
        graph.components.push_back({
            "vision", "vision", std::move(vision), "optional", {}, {}, {}});
    }
    if (!predictor.empty()) {
        graph.components.push_back({
            "predictor", "predictor", std::move(predictor), "optional",
            {}, {}, {}});
    }
    return graph;
}

} // namespace

int main() {
    try {
        using namespace mfq::cuda;

        const auto qwen = graph_with(
            "qwen3_5", "grid_vit", "next_token_prediction");
        const auto qwen_plan = mfq_cuda_model_plan(qwen);
        require(
            qwen_plan.backbone == MfqCudaBackbone::generic_qwen,
            "Qwen semantic backbone did not resolve");
        const auto qwen_state = mfq_cuda_component_state(
            qwen, qwen_plan, true, true);
        require(
            qwen_state.vision_declared && qwen_state.mtp_declared,
            "graph components were not preserved");
        require(
            !qwen_state.vision_supported && qwen_state.mtp_supported &&
                !qwen_state.vision_available && qwen_state.mtp_available,
            "Qwen predictor/vision availability did not match execution adapters");
        const auto qwen_unloaded = mfq_cuda_component_state(qwen, qwen_plan, false, false);
        require(qwen_unloaded.mtp_supported && !qwen_unloaded.mtp_available && !qwen_unloaded.mtp_enabled,
            "unloaded Qwen predictor was reported available");
        const auto unknown_predictor = graph_with("qwen3_5", {}, "experimental_draft");
        require(mfq_cuda_model_plan(unknown_predictor).predictor == MfqCudaPredictorAdapter::none,
            "unknown predictor implementation was inferred from backbone name");

        auto minicpm = graph_with("minicpmo45", "minicpmo45_vision");
        minicpm.components.push_back({
            "audio_input", "audio", "minicpmo45_audio", "optional",
            {}, {}, {}});
        minicpm.components.push_back({
            "audio_output", "tts", "minicpmo45_tts", "optional",
            {}, {}, {}});
        const auto minicpm_plan = mfq_cuda_model_plan(minicpm);
        require(
            minicpm_plan.vision == MfqCudaVisionAdapter::minicpmo45,
            "MiniCPM semantic Vision adapter did not resolve");
        const auto defaults = mfq_cuda_component_state(
            minicpm, minicpm_plan, true, false);
        require(
            defaults.vision_supported && defaults.vision_available &&
                defaults.vision_enabled,
            "loaded MiniCPM Vision did not default on");
        const auto disabled = mfq_cuda_component_state(
            minicpm, minicpm_plan, true, false, false, true);
        require(
            disabled.vision_available && !disabled.vision_enabled,
            "manual switch changed component availability");

        const auto partial_minicpm = graph_with(
            "minicpmo45", "minicpmo45_vision");
        require(
            mfq_cuda_model_plan(partial_minicpm).vision ==
                MfqCudaVisionAdapter::none,
            "partial MiniCPM graph selected a composite CUDA adapter");

        const auto unsupported = graph_with("unknown_experimental", "grid_vit");
        require(
            mfq_cuda_model_plan(unsupported).backbone ==
                MfqCudaBackbone::unsupported,
            "unimplemented CUDA backbone was accepted");
        const auto qwen4 = graph_with("qwen4_exp", "grid_vit", "next_token_prediction");
        const auto qwen4_plan = mfq_cuda_model_plan(qwen4);
        require(qwen4_plan.backbone == MfqCudaBackbone::qwen4_exp &&
            qwen4_plan.vision == MfqCudaVisionAdapter::none && qwen4_plan.predictor == MfqCudaPredictorAdapter::none,
            "Qwen4 text/optional-component selection mismatch");

        const auto glm_next = graph_with("glm5_next", "grid_vit", "next_token_prediction");
        const auto glm_plan = mfq_cuda_model_plan(glm_next);
        require(glm_plan.backbone == MfqCudaBackbone::glm5_next,
            "GLM Flash-Next text adapter was not selected");
        require(glm_plan.vision == MfqCudaVisionAdapter::none &&
            glm_plan.predictor == MfqCudaPredictorAdapter::none,
            "text support incorrectly advertised unimplemented optional components");

        std::cout << "MFQ CUDA model plan tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MFQ CUDA model plan test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
