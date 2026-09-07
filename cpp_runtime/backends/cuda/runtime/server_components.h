#pragma once

// Backend-specific optional components are selected only from model_graph.json.
// The CLI/server consumes this single holder, mirroring Metal's component layer.
struct CudaRuntimeComponents {
    mfq::MfqModelGraph graph;
    mfq::cuda::MfqCudaModelPlan plan;
    std::optional<MiniCPMO45Runtime> minicpmo;
    std::unique_ptr<CudaMtpModule> mtp;
    bool vision_available = false;
    bool mtp_available = false;

    Model& language(Model& fallback) {
        return minicpmo ? minicpmo->language : fallback;
    }

    mfq::cuda::MfqCudaComponentState state() const noexcept {
        return mfq::cuda::mfq_cuda_component_state(
            graph, plan, vision_available, mtp_available);
    }
};

static CudaRuntimeComponents load_cuda_runtime_components(
        Model& model,
        const std::string& mfq_path,
        bool load_optional_components,
        const std::string& config_path = {}) {
    CudaRuntimeComponents result;
    result.graph = model.c.model_graph;
    result.plan = model.c.runtime_plan;
    if (!load_optional_components) return result;

    switch (result.plan.vision) {
        case mfq::cuda::MfqCudaVisionAdapter::none:
            break;
        case mfq::cuda::MfqCudaVisionAdapter::minicpmo45:
            result.minicpmo.emplace(
                MiniCPMO45Runtime::load_with_language(
                    std::move(model), mfq_path));
            result.vision_available = true;
            break;
    }
    if (result.plan.predictor == mfq::cuda::MfqCudaPredictorAdapter::qwen35) {
        const bool single_device = !g_tensor_parallel.enabled() && !g_layer_placement.enabled() &&
            g_dense_cpu_layer_count == 0 && g_dsv4_cpu_offload_layers.empty() && !g_moe_expert_cache;
        if (single_device && model.c.num_experts == 0 && model.supports_qwen_speculation()) {
            MfqFile predictor_file(mfq_path);
            (void)load_config(predictor_file, config_path);
            auto predictor = CudaQwen35Mtp::load_if_present(predictor_file, model.c);
            if (predictor) {
                result.mtp = std::make_unique<CudaQwen35Mtp>(std::move(*predictor));
            }
            result.mtp_available = static_cast<bool>(result.mtp);
        } else {
            std::cerr << "qwen_mtp unavailable: initial CUDA adapter requires dense single-GPU Qwen blocks\n";
        }
    }
    if (result.plan.predictor == mfq::cuda::MfqCudaPredictorAdapter::flash_next) {
        MFQ_RUNTIME_CHECK(model.c.is_flash_next() && model.supports_qwen_speculation(),
            "invalid Flash-Next predictor backbone");
        MfqFile predictor_file(mfq_path);
        (void)load_config(predictor_file, config_path);
        auto predictor = CudaFlashNextMtp::load_if_present(predictor_file, model.c);
        if (predictor) {
            result.mtp = std::make_unique<CudaFlashNextMtp>(std::move(*predictor));
        }
        result.mtp_available = static_cast<bool>(result.mtp);
    }
    return result;
}
