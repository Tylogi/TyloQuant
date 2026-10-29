#include "mfq_weight_memory.h"
#include "mfq_hip_kv_quantization.h"
#include "hip/hip_runtime.h"
#include "hip_runtime_config.h"
#include "hip_execution.h"
#include "generation.h"
#include "storage/model_loader.h"
#include "storage/weight_loader.h"
#include "hip_batching.h"
#include "storage/text_session_cache.h"
#include "mtp_metrics.h"
#include "request_executor.h"
#include "generation_flow.h"

#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <iostream>
#include <variant>

namespace mfq::cuda {
using namespace mfq::engine;
using namespace mfq::cuda::internal;

namespace {
template <typename Model>
struct CudaEngineState {
    std::shared_ptr<WeightMemoryLedger> weight_memory;
    std::shared_ptr<CudaExecutionContext> execution;
    CudaRuntimeConfig config;
    Model model;
    RuntimeComponents<Model> components;
    Model& language;
    DecodeGraphCache graph;
    TextSessionCache cache;
    std::unique_ptr<ContinuousBatch<QwenBatchOperations>> batching;
    bool duplex_active = false;

    CudaEngineState(std::shared_ptr<CudaExecutionContext> owner, Model loaded,
            RuntimeComponents<Model> optional, CudaRuntimeConfig runtime_config,std::shared_ptr<WeightMemoryLedger> weights)
        : weight_memory(std::move(weights)), execution(std::move(owner)), config(std::move(runtime_config)),
          model(std::move(loaded)), components(std::move(optional)),
          language(components.language(model)), graph(language.max_position_embeddings()),
          cache(config.session_cache, config.prefix_cache,
              make_cuda_paged_prefix_cache(*language.source, language.max_position_embeddings(),
                  language.supports_paged_text_session_state(), config.prefix_cache),
                language.supports_text_session_state(), language.supports_text_session_state() ? 0 : 1) {
        if (config.continuous_batch.max_sequences) {
            if constexpr (std::is_same_v<Model, Qwen35CausalLm>)
                batching = std::make_unique<ContinuousBatch<QwenBatchOperations>>(
                    config.generation.prefill_chunk_size, config.continuous_batch.prefill_token_budget,
                    language, *execution, config.continuous_batch);
            else
                // Capacity is advisory on HIP too: keep the single-request path
                // and report capacity one rather than failing the load.
                std::cerr<<"continuous_batching unavailable model="<<language.model_type()
                    <<" requested_capacity="<<config.continuous_batch.max_sequences<<'\n';
        }
    }
    bool can_batch(const EngineRequest& request) const {
        return batching && batch_compatible(request, cache.persistent_prefix_enabled(), bool(components.mtp));
    }
    std::size_t batch_capacity() const {
        return batching ? std::max<std::size_t>(1, config.continuous_batch.max_sequences) : 1;
    }
    bool exclusive() const { return duplex_active; }
    bool mtp_available() const { return bool(components.mtp); }
    void execute(const std::vector<RequestId>& eligible) {
        if (batching) batching->step(eligible);
    }
    using Prepared = CudaPreparedPrompt;
    mfq::StepSequence<Prepared> prepare(InferenceRequest& input) {
        if constexpr(std::is_same_v<Model,mfq::cuda::DeepseekV41CausalLm>)
            if(components.deepseek_vision)return components.deepseek_vision->prepare(language,input.prompt,*input.vision);
        if (components.composite) return components.composite->prepare(input.prompt, *input.vision);
        if (components.radio_vision) return components.radio_vision->prepare(language,input.prompt,*input.vision);
        if (components.grid_vision) return components.grid_vision->prepare(language, input.prompt, *input.vision);
        throw std::invalid_argument("model has no multimodal component");
    }
    auto advance_preparation(mfq::StepSequence<Prepared>& preparation) {
        PrefillCudaTimer timer;
        auto step = preparation.next();
        MFQ_CUDA_CHECK(hipEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
        return std::pair{std::move(step), timer.elapsed_ms()};
    }
    Generation generate_text(InferenceRequest& input, InferenceOutput& output, std::optional<Prepared> prepared,
                             const RequestId& id, bool batched) {
        return internal::generate(language, graph, cache, config,
                                  input, output, batched ? nullptr : components.mtp.get(), std::move(prepared),
                                  batched ? batching.get() : nullptr, id);
    }
    Generation generate(const RequestId& id, ExecutionRequest& request) {
        if (request.batched) graph.invalidate();
        return generate_prepared(*this, request.input, request.output, id, request.batched);
    }
    template <class T> ControlResult control(T value) {
        if constexpr (std::is_same_v<T, RuntimeMetrics>) {
            size_t free = 0, total = 0;
            MFQ_CUDA_CHECK(hipMemGetInfo(&free, &total));
            auto memory = mfq_hip_memory_stats(mfq_current_cuda_device());
            const auto capabilities = components.state();
            Metrics metrics{{"device_free_bytes", double(free)}, {"device_total_bytes", double(total)},
                {"continuous_batching_mtp_supported", 0.0},
                {"continuous_batching_mtp_stochastic_supported", 0.0},
                {"continuous_batching_mtp_graph_supported", 0.0},
                {"cuda_allocated_bytes", double(memory.allocated_bytes)}, {"cuda_reserved_bytes", double(memory.reserved_bytes)},
                {"vision_declared", double(capabilities.vision_declared)}, {"vision_supported", double(capabilities.vision_supported)},
                {"vision_available", double(capabilities.vision_available)}, {"mtp_declared", double(capabilities.mtp_declared)},
                {"mtp_supported", double(capabilities.mtp_supported)}, {"mtp_available", double(capabilities.mtp_available)},
                {"kv_quantization_supported",(std::is_same_v<Model,Qwen35CausalLm> || std::is_same_v<Model,Qwen4CausalLm>)?1.0:0.0},
                {"kv_quantization_bits",(std::is_same_v<Model,Qwen35CausalLm> || std::is_same_v<Model,Qwen4CausalLm>)?cuda_kv_quantization().bits:0.0}};
            const auto ple_bytes=weight_memory->ple_bytes();
            const bool complete_weights=!moe_expert_cache_has_sources(execution->moe_expert_cache);
            metrics.insert(metrics.end(),{{"resident_observed_weight_bytes",double(weight_memory->resident_bytes())},
                {"resident_weight_bytes_complete",complete_weights?1.0:0.0},
                {"ssd_ple_enabled",ple_bytes>0?1.0:0.0},{"ssd_ple_payload_bytes",double(ple_bytes)}});
            if(complete_weights)metrics.emplace_back("resident_weight_bytes",double(weight_memory->resident_bytes()));
            if (components.mtp) mtp::append_generation_metrics(metrics, components.mtp->last_stats);
            if (batching) {
                auto batch = batching->metrics(); metrics.insert(metrics.end(), batch.begin(), batch.end());
            }
            return metrics;
        } else {
            if (!components.composite) throw std::invalid_argument("model has no duplex component");
            if constexpr (std::is_same_v<T, MfqDuplexSessionParams>) {
                components.composite->start(value); duplex_active = true;
            } else if constexpr (std::is_same_v<T, MfqDuplexStepInput>) {
                return components.composite->step(value);
            } else { components.composite->stop(); duplex_active = false; }
            return std::monostate{};
        }
    }

};
using State = std::variant<
    std::unique_ptr<CudaEngineState<Qwen35CausalLm>>,
    std::unique_ptr<CudaEngineState<MiniCPMO45CausalLm>>,
    std::unique_ptr<CudaEngineState<MiniCPMOTtsCausalLm>>,
    std::unique_ptr<CudaEngineState<Gemma4CausalLm>>,
    std::unique_ptr<CudaEngineState<GlmDsaCausalLm>>,
    std::unique_ptr<CudaEngineState<Glm5CausalLm>>,
    std::unique_ptr<CudaEngineState<Qwen4CausalLm>>,
    std::unique_ptr<CudaEngineState<DeepseekV4CausalLm>>,
    std::unique_ptr<CudaEngineState<DeepseekV41CausalLm>>,
    std::unique_ptr<CudaEngineState<NemotronHCausalLm>>>;

struct CudaBackend {
    using Options = CudaEngineOptions;
    Options options;
    State state;
    template <class F> decltype(auto) visit(F&& run) {
        return std::visit([&](auto& state) -> decltype(auto) {
            if (!state) throw std::runtime_error("CUDA engine is unloaded");
            MfqCudaGuard guard(state->execution->layer_placement.primary_device());
            mfq_tensor_backend::NoGradGuard no_grad;
            return run(*state);
        }, state);
    }
    std::pair<EngineInfo, std::unique_ptr<TextProcessor>> load() {
        if (!options.context_size) options.context_size = 32768;
        auto execution = std::make_shared<CudaExecutionContext>();
        setup_cuda_load(options, *execution);
        execution->profiler.enabled = false;
        mfq_tensor_backend::NoGradGuard no_grad;
        auto config = resolve_cuda_runtime_config(options);
        auto weight_memory=std::make_shared<WeightMemoryLedger>();
        const WeightMemoryLoadScope memory_load(*weight_memory);
        state = with_loaded_cuda_model(*execution, options, true,
            [&](auto& model, auto& components, auto, auto) -> State {
                using Model = std::decay_t<decltype(model)>;
                if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                    const auto* reuse=std::getenv("MFQ_RUNTIME_REUSE_DECODE_GRAPHS");
                    model.reuse_decode_graphs=!reuse || reuse[0]!='0';
                }
                return std::make_unique<CudaEngineState<Model>>(execution, std::move(model), std::move(components), config,weight_memory);
            });
        return visit([&](auto& state) {
            EngineInfo info;
            info.batch_capacity = state.batch_capacity();
            info.max_requests = info.batch_capacity;
            info.max_context = state.language.max_position_embeddings();
            info.vocab_size = static_cast<int32_t>(state.language.vocab_size());
            info.model_type = state.language.model_type();
            info.multimodal = state.components.grid_vision.has_value() || state.components.deepseek_vision.has_value() || state.components.radio_vision.has_value() || bool(state.components.composite);
            info.reload = true;
            const auto caps = cuda_runtime_capabilities(state.components.graph, state.components.plan,
                state.components.state(), info.multimodal, bool(state.components.composite));
            info.duplex = caps.full_duplex;
            info.capabilities = {state.components.graph.architecture, caps.text,
                caps.image_input, caps.video_input, caps.audio_input, caps.audio_output,
                caps.full_duplex, caps.mtp, "model-graph+cuda-adapters"};
            auto text = TextProcessor::load(*state.language.source, options.tokenizer_model,
                info.vocab_size, info.model_type);
            info.stop_token_ids = model_stop_token_ids(
                load_model_config_json(*state.language.source, options.config_path), info.vocab_size);
            return std::pair{std::move(info), std::move(text)};
        });
    }
    bool exclusive() const {
        return std::visit([](const auto& state) { return state && state->exclusive(); }, state);
    }
    void unload() {
        state = std::unique_ptr<CudaEngineState<Qwen35CausalLm>>{};
    }
};
} // namespace

std::unique_ptr<Engine> load_cuda_engine(CudaEngineOptions options) {
    return std::make_unique<EngineInstance<CudaBackend>>(std::move(options));
}
} // namespace mfq::cuda
