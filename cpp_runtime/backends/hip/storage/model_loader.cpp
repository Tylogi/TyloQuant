#include "models/nemotron_h/mtp.h"
#include "hip/hip_runtime.h"
#include "storage/weight_loader.h"
#include "storage/model_loader.h"
#include "models/qwen4_exp/mtp.h"
#include "models/glm5_next/mtp.h"
#include "models/qwen35/mtp.h"
#include "models/deepseek_v41/dspark.h"

#include "hip_execution.h"
#include "storage/moe_expert_cache.h"

#include <algorithm>
#include <iostream>
#include <utility>

namespace {

// Allocation, packing, placement and device lifetime are native. The shared
// causal model supplies weight relationships and layer traversal.
template <class Model> struct CudaWeightLoader {
    Model &model;
    const mfq::ModelSource &source;
    auto embedding(const std::string &name) {
        return load_quant_linear(*model.execution, source, name);
    }
    auto output(const std::string &name) {
        return load_quant_linear(*model.execution, source, name, TensorParallelAxis::Output);
    }
    auto tied_output(const QuantLinear &embedding, const std::string &name) {
        return model.execution->tensor_parallel.enabled() ? output(name) : embedding;
    }
    bool has_weight(const std::string &name) { return has_tensor(source, name); }
    void final_state(mfq_tensor_backend::Tensor &norm) {
        model.adapter_load_final_state(source, norm);
    }
    void prepare_blocks() { model.adapter_prepare_blocks(source); }
    auto block(int layer, const std::string &type) {
        auto &execution = *model.execution;
        const int device = execution.layer_placement.device_for_layer(layer);
        const bool cpu_offloaded = layer < execution.dense_cpu_layer_count;
        execution.loading_cpu_layer = cpu_offloaded;
        execution.layer_placement.load_device = device;
        MfqCudaGuard layer_guard(device);
        std::cerr << "loading layer " << layer << ' ' << type << ' '
                  << (cpu_offloaded ? "CPU" : "CUDA") << std::endl;
        auto block = model.adapter_load_block(source, layer, device, type);
        block->cuda_device = device;
        block->cpu_offloaded = cpu_offloaded;
        return block;
    }
};

} // namespace

template <typename Model>
Model mfq::cuda::load_causal_lm(CudaExecutionContext &execution, const std::string &model_path,
                                const std::string &config_path, int64_t context_size_override,
                                bool load_blocks, bool defer_moe_cache_finalize,
                                std::shared_ptr<const mfq::ModelSource> model_source) {
    Model model;
    model.execution = &execution;
    model.source = model_source ? std::move(model_source) : mfq::open_model_source(model_path);
    const auto &source = *model.source;
    validate_model_source(source);
    model.graph = source.resolved_model_graph();
    model.plan = cuda_model_plan(model.graph);
    model.load_definition(load_model_config_json(source, config_path), model.graph, source,
                          context_size_override);

    model.adapter_validate_load_options();
    if (execution.expert_parallel.enabled() && model.num_experts() <= 0) {
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    }

    execution.layer_placement.prepare(model.num_hidden_layers());
    execution.dense_cpu_layer_count = 0;
    if (execution.n_gpu_layers >= 0) {
        execution.dense_cpu_layer_count = static_cast<int>(
            std::max<int64_t>(model.num_hidden_layers() - execution.n_gpu_layers, 0));
        if (execution.dense_cpu_layer_count > 0) {
            if (model_parallel_enabled(execution) || execution.layer_placement.enabled()) {
                throw std::runtime_error(
                    "--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
            }
            bool supported = model.adapter_supports_dense_cpu_offload() && model.num_experts() <= 0;
            for (int layer = 0; supported && layer < model.num_hidden_layers(); ++layer) {
                const auto type = model.layer_type(layer);
                supported = type == "full_attention" || type == "linear_attention";
            }
            if (!supported) {
                throw std::runtime_error(
                    "--n-gpu-layers currently supports dense Qwen-style blocks only");
            }
            std::cerr << "dense_layer_placement cpu=0-" << (execution.dense_cpu_layer_count - 1)
                      << " gpu=" << execution.dense_cpu_layer_count << '-'
                      << (model.num_hidden_layers() - 1) << " cpu_threads=" << mfq_get_num_threads()
                      << std::endl;
        }
    }
    execution.layer_placement.load_device = execution.layer_placement.primary_device();
    MfqCudaGuard model_guard(execution.layer_placement.primary_device());
    if (model.plan.vision == CudaVisionAdapter::grid_vit && execution.dense_cpu_layer_count > 0) {
        throw std::runtime_error("CUDA grid-Vision currently requires GPU-resident text layers");
    }

    if (model.adapter_uses_common_rope() && !model.metadata.no_rope) {
        auto make_rope = [&](mfq_tensor_backend::Device device) {
            RopeCache rope(model.max_position_embeddings(), model.rotary_dim(), model.rope_base(),
                           0, -1, device, model.metadata.rope_interleaved);
            model.adapter_configure_rope(rope, device);
            return rope;
        };
        const auto primary = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA,
                                                        execution.layer_placement.primary_device());
        model.rope = make_rope(primary);
        if (execution.dense_cpu_layer_count > 0) {
            model.cpu_rope = make_rope(mfq_tensor_backend::Device(mfq_tensor_backend::kCPU));
        }
        if (execution.layer_placement.enabled()) {
            for (int device : execution.layer_placement.devices) {
                MfqCudaGuard rope_guard(device);
                model.device_ropes.emplace(device, make_rope(mfq_tensor_backend::Device(
                                                       mfq_tensor_backend::kCUDA, device)));
            }
        }
    }

    CudaWeightLoader<Model> weights{model, source};
    model.load_weights(weights, load_blocks);

    execution.loading_cpu_layer = false;
    execution.layer_placement.load_device = execution.layer_placement.primary_device();
    if (moe_expert_cache_has_sources(execution.moe_expert_cache) &&
        !moe_expert_cache_finalized(execution.moe_expert_cache) && !defer_moe_cache_finalize) {
        finalize_moe_expert_cache(execution.moe_expert_cache);
    }
    return model;
}

#define MFQ_INSTANTIATE_CAUSAL_LM(TYPE)                                                            \
    template TYPE mfq::cuda::load_causal_lm<TYPE>(CudaExecutionContext &, const std::string &,     \
                                                  const std::string &, int64_t, bool, bool,        \
                                                  std::shared_ptr<const mfq::ModelSource>)

MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::Qwen35CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::MiniCPMO45CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::MiniCPMOTtsCausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::Glm5CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::Qwen4CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::DeepseekV4CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::DeepseekV41CausalLm);
MFQ_INSTANTIATE_CAUSAL_LM(mfq::cuda::NemotronHCausalLm);

#undef MFQ_INSTANTIATE_CAUSAL_LM

template <typename Model>
RuntimeComponents<Model> load_runtime_components(
        Model& model,
        bool load_optional_components) {
    RuntimeComponents<Model> result{model.graph, model.plan};
    if (load_optional_components &&
            (result.plan.vision != mfq::cuda::CudaVisionAdapter::none ||
             result.plan.predictor != mfq::cuda::CudaPredictorAdapter::none)) {
        throw std::runtime_error(
            "CUDA component adapter is unsupported for this model");
    }
    return result;
}

template <>
RuntimeComponents<mfq::cuda::Qwen35CausalLm> load_runtime_components(
        mfq::cuda::Qwen35CausalLm& model,
        bool load_optional_components) {
    RuntimeComponents<mfq::cuda::Qwen35CausalLm> result{model.graph, model.plan};
    if (!load_optional_components) return result;

    if (result.plan.vision == mfq::cuda::CudaVisionAdapter::grid_vit) {
        const auto* component = result.graph.component("vision");
        const auto& config = model.config;
        if (component == nullptr || !config.grid_vision ||
                !config.image_token_id || !config.video_token_id) {
            throw std::runtime_error(
                "CUDA grid-Vision configuration is incomplete");
        }
        result.grid_vision.emplace(
            mfq::cuda::grid_vision_runtime::CudaGridVisionPromptComponent::load(
                *model.execution, *model.source, *config.grid_vision,
                *config.image_token_id, *config.video_token_id,
                component->input_contract, component->position_policy));
        result.vision_available = true;
    } else if (result.plan.vision != mfq::cuda::CudaVisionAdapter::none) {
        throw std::runtime_error(
            "CUDA vision adapter is unsupported for Qwen");
    }

    if (result.plan.predictor == mfq::cuda::CudaPredictorAdapter::qwen35) {
        const auto& model_execution = *model.execution;
        const bool supported_placement =
            !model_execution.layer_placement.enabled() &&
            model_execution.dense_cpu_layer_count == 0 &&
            model_execution.dsv4_cpu_offload_layers.empty() &&
            !model_execution.moe_expert_cache;
        if (supported_placement && model.num_experts() == 0 &&
                model.supports_speculation()) {
            auto predictor = Qwen35Mtp::load_if_present(
                *model.source, model.config, *model.execution);
            if (predictor) {
                result.mtp = std::make_unique<Qwen35Mtp>(
                    std::move(*predictor));
            }
            result.mtp_available = static_cast<bool>(result.mtp);
        } else {
            std::cerr << "qwen_mtp unavailable: CUDA adapter requires dense GPU-resident Qwen blocks\n";
        }
    } else if (result.plan.predictor !=
            mfq::cuda::CudaPredictorAdapter::none) {
        throw std::runtime_error(
            "CUDA predictor adapter is unsupported for Qwen");
    }
    return result;
}

template <>
RuntimeComponents<mfq::cuda::Qwen4CausalLm> load_runtime_components(
        mfq::cuda::Qwen4CausalLm& model,
        bool load_optional_components) {
    RuntimeComponents<mfq::cuda::Qwen4CausalLm> result{model.graph, model.plan};
    if (!load_optional_components) return result;
    if (result.plan.vision == mfq::cuda::CudaVisionAdapter::grid_vit) {
        const auto document = nlohmann::json::parse(model.source->model_config_json());
        const auto& vision = document.at("vision_config");
        if (vision.value("hidden_act", std::string("gelu_pytorch_tanh")) != "gelu_pytorch_tanh" ||
                !vision.value("deepstack_visual_indexes", std::vector<int>{}).empty())
            throw std::runtime_error("unsupported Qwen4 grid-ViT activation or deepstack");
        mfq::GridVisionConfig config;
        config.hidden_size = vision.at("hidden_size").get<int64_t>();
        config.intermediate_size = vision.at("intermediate_size").get<int64_t>();
        config.depth = vision.at("depth").get<int64_t>();
        config.num_heads = vision.at("num_heads").get<int64_t>();
        config.in_channels = vision.value("in_channels", int64_t{3});
        config.patch_size = vision.at("patch_size").get<int64_t>();
        config.temporal_patch_size = vision.at("temporal_patch_size").get<int64_t>();
        config.spatial_merge_size = vision.value("spatial_merge_size", int64_t{2});
        config.out_hidden_size = vision.at("out_hidden_size").get<int64_t>();
        config.num_position_embeddings = vision.at("num_position_embeddings").get<int64_t>();
        config.rope_theta = vision.value("rope_theta", 10000.0);
        if (vision.contains("rope_parameters"))
            config.rope_theta = vision.at("rope_parameters").value("rope_theta", config.rope_theta);
        config.layer_norm_eps = vision.value("layer_norm_eps", 1e-6);
        config.validate();
        const auto image = document.at("image_token_id").get<int64_t>();
        const auto video = document.at("video_token_id").get<int64_t>();
        if (config.out_hidden_size != model.config.hidden || model.config.sections.size() != 3 ||
                image < 0 || video < 0 || image >= model.config.vocab || video >= model.config.vocab || image == video)
            throw std::runtime_error("invalid Qwen4 grid-ViT text geometry or placeholder IDs");
        const auto* component = result.graph.component("vision");
        if(!component)throw std::runtime_error("Qwen4 grid-ViT component is missing");
        result.grid_vision.emplace(
            mfq::cuda::grid_vision_runtime::CudaGridVisionPromptComponent::load(
                *model.execution, *model.source, config, image, video,
                component->input_contract, component->position_policy));
        result.vision_available = true;
    } else if (result.plan.vision != mfq::cuda::CudaVisionAdapter::none) {
        throw std::runtime_error("unsupported Qwen4 CUDA vision adapter");
    }
    if (result.plan.predictor != mfq::cuda::CudaPredictorAdapter::none &&
             result.plan.predictor !=
                 mfq::cuda::CudaPredictorAdapter::flash_next) {
        throw std::runtime_error(
            "unsupported Qwen4 CUDA component adapter");
    }
    if (result.plan.predictor ==
            mfq::cuda::CudaPredictorAdapter::none) return result;
    MFQ_RUNTIME_CHECK(
        model.supports_speculation(),
        "invalid Flash-Next predictor model");
    auto predictor = mfq::cuda::qwen4_exp::Qwen4ExpMtp::load_if_present(
        *model.execution, *model.source, model.config);
    if (predictor) {
        result.mtp =
            std::make_unique<mfq::cuda::qwen4_exp::Qwen4ExpMtp>(
                std::move(*predictor));
    }
    result.mtp_available = static_cast<bool>(result.mtp);
    return result;
}

template <>
RuntimeComponents<mfq::cuda::Glm5CausalLm> load_runtime_components(
        mfq::cuda::Glm5CausalLm& model,
        bool load_optional_components) {
    RuntimeComponents<mfq::cuda::Glm5CausalLm> result{model.graph, model.plan};
    if (!load_optional_components) return result;
    if (result.plan.vision != mfq::cuda::CudaVisionAdapter::none ||
            (result.plan.predictor != mfq::cuda::CudaPredictorAdapter::none &&
             result.plan.predictor !=
                 mfq::cuda::CudaPredictorAdapter::flash_next)) {
        throw std::runtime_error(
            "unsupported GLM5 CUDA component adapter");
    }
    if (result.plan.predictor ==
            mfq::cuda::CudaPredictorAdapter::none) return result;
    MFQ_RUNTIME_CHECK(
        model.supports_speculation(),
        "invalid Flash-Next predictor model");
    auto predictor = mfq::cuda::glm5_next::Glm5NextMtp::load_if_present(
        *model.execution, *model.source, model.config);
    if (predictor) {
        result.mtp =
            std::make_unique<mfq::cuda::glm5_next::Glm5NextMtp>(
                std::move(*predictor));
    }
    result.mtp_available = static_cast<bool>(result.mtp);
    return result;
}

template <>
RuntimeComponents<mfq::cuda::DeepseekV41CausalLm> load_runtime_components(
        mfq::cuda::DeepseekV41CausalLm& model,
        bool load_optional_components) {
    RuntimeComponents<mfq::cuda::DeepseekV41CausalLm> result{model.graph, model.plan};
    if (!load_optional_components) return result;
    if ((result.plan.vision != mfq::cuda::CudaVisionAdapter::none &&
         result.plan.vision != mfq::cuda::CudaVisionAdapter::deepseek_v41) ||
            (result.plan.predictor != mfq::cuda::CudaPredictorAdapter::none &&
             result.plan.predictor !=
                 mfq::cuda::CudaPredictorAdapter::deepseek_v41_dspark)) {
        throw std::runtime_error(
            "unsupported DeepSeek-V4.1 CUDA component adapter");
    }
    if(result.plan.vision==mfq::cuda::CudaVisionAdapter::deepseek_v41) {
        result.deepseek_vision.emplace(mfq::cuda::deepseek_v41_runtime::Vision::load(model.source,model.shared->config));
        result.vision_available=true;
    }
    if (result.plan.predictor ==
            mfq::cuda::CudaPredictorAdapter::none) return result;
    MFQ_RUNTIME_CHECK(
        model.supports_suffix_speculation() && model.shared,
        "invalid DeepSeek-V4.1 DSpark model");
    result.mtp =
        mfq::cuda::deepseek_v41_runtime::load_dspark_if_present(
            *model.execution, *model.source, model.shared->config);
    result.mtp_available = static_cast<bool>(result.mtp);
    return result;
}

template <>
RuntimeComponents<mfq::cuda::MiniCPMO45CausalLm>
load_runtime_components(
        mfq::cuda::MiniCPMO45CausalLm& model,
        bool load_optional_components) {
    RuntimeComponents<mfq::cuda::MiniCPMO45CausalLm> result{model.graph, model.plan};
    if (!load_optional_components ||
            result.plan.vision == mfq::cuda::CudaVisionAdapter::none) {
        return result;
    }
    if (result.plan.vision !=
            mfq::cuda::CudaVisionAdapter::minicpmo45) {
        throw std::runtime_error(
            "unsupported MiniCPM-o CUDA vision adapter");
    }

    result.composite = std::make_unique<mfq::cuda::minicpmo45::Components>(std::move(model));
    result.language_override = &result.composite->language();
    result.vision_available = true;
    return result;
}

template <>
RuntimeComponents<mfq::cuda::MiniCPMOTtsCausalLm>
load_runtime_components(
        mfq::cuda::MiniCPMOTtsCausalLm& model,
        bool) {
    RuntimeComponents<mfq::cuda::MiniCPMOTtsCausalLm> result{model.graph, model.plan};
    return result;
}

#define MFQ_INSTANTIATE_COMPONENTS(TYPE)                                  \
    template RuntimeComponents<TYPE> load_runtime_components(TYPE&, bool)

MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_COMPONENTS(mfq::cuda::DeepseekV4CausalLm);


#undef MFQ_INSTANTIATE_COMPONENTS

template <>
RuntimeComponents<mfq::cuda::NemotronHCausalLm> load_runtime_components(
        mfq::cuda::NemotronHCausalLm& model,
        bool load_optional_components) {
    RuntimeComponents<mfq::cuda::NemotronHCausalLm> result{model.graph, model.plan};
    if (!load_optional_components) return result;
    if ((result.plan.vision != mfq::cuda::CudaVisionAdapter::none &&
         result.plan.vision != mfq::cuda::CudaVisionAdapter::radio) ||
            (result.plan.predictor != mfq::cuda::CudaPredictorAdapter::none &&
             result.plan.predictor != mfq::cuda::CudaPredictorAdapter::nemotron_h)) {
        throw std::runtime_error(
            "unsupported Nemotron-H CUDA component adapter");
    }
    if (result.plan.vision == mfq::cuda::CudaVisionAdapter::radio) {
        const auto* component = result.graph.component("vision");
        MFQ_RUNTIME_CHECK(component != nullptr,
            "Nemotron-H graph declares a RADIO adapter without a vision component");
        const auto vision = mfq::models::nemotron_h::RadioVisionConfig::from_json(
            model.source->model_config_json());
        result.radio_vision.emplace(
            mfq::cuda::nemotron_h::CudaRadioPromptComponent::load(
                *model.execution, *model.source, vision, component->input_contract,
                component->position_policy));
        result.vision_available = true;
    }
    if (result.plan.predictor ==
            mfq::cuda::CudaPredictorAdapter::none) return result;
    MFQ_RUNTIME_CHECK(model.supports_speculation(),
        "invalid Nemotron-H predictor model");
    auto predictor = mfq::cuda::nemotron_h::NemotronHMtp::load_if_present(
        *model.execution, *model.source, model.config);
    // The graph declares the component, so missing weights are a broken
    // container rather than a text-only model.
    MFQ_RUNTIME_CHECK(predictor,
        "Nemotron-H graph declares the predictor component but the source has no MTP weights");
    result.mtp = std::make_unique<mfq::cuda::nemotron_h::NemotronHMtp>(
        std::move(*predictor));
    result.mtp_available = true;
    return result;
}
