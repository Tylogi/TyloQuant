#include "models/nemotron_h/ops.h"
#include "generation.h"
#include "generation_flow.h"
#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/qwen35/ops.h"
#include "models/qwen4_exp/ops.h"
#include "core/full_block.h"
#include "hip_sampling.h"
#include "storage/text_session_cache.h"
#include "mfq_hip_sampling_ops.h"
#include "mfq_hip_context.h"

#include <algorithm>
#include <iostream>

namespace mfq::cuda::internal {
using mfq::engine::Generation;
namespace {
class EagerDecodePool {
    std::vector<std::pair<std::shared_ptr<Context>, hipStream_t>> pools_;
public:
    EagerDecodePool() {
        int devices = 0;
        MFQ_CUDA_CHECK(hipGetDeviceCount(&devices));
        pools_.reserve(devices);
        try {
            for (int device = 0; device < devices; ++device) {
                auto context = default_context(device);
                const auto stream = mfq_get_current_cuda_stream(device).stream();
                context->begin_graph_pool(stream);
                pools_.emplace_back(std::move(context), stream);
                pools_.back().first->begin_graph_warmup(stream);
            }
        } catch (...) {
            clear();
            throw;
        }
    }
    void clear() noexcept {
        for (auto it = pools_.rbegin(); it != pools_.rend(); ++it) {
            it->first->end_graph_warmup(it->second);
            it->first->end_graph_pool(it->second);
        }
        pools_.clear();
    }
    ~EagerDecodePool() { clear(); }
};

template <typename Model>
static mfq_tensor_backend::Tensor sample_token(Model &model, mfq_tensor_backend::Tensor ids,
    mfq::cuda::Sampler &sampler, mfq_tensor_backend::Tensor counts,
    const MfqTokenConstraintPtr &token_constraint) {
    if (sampler.greedy() && !sampler.has_penalties() && !token_constraint) {
        return model.next_token(ids);
    }

    auto logits = model.last_logits(ids).contiguous().view({1, -1});
    return mfq::cuda::sample_logits(sampler, std::move(logits), counts, token_constraint);
}

template <typename Model> struct CudaGenerationOps {
    Model &model;
    DecodeGraphCache &graph;
    const MfqTokenConstraintPtr &constraint;
    mfq_tensor_backend::Tensor pending, counts;
    mfq::cuda::Sampler sampler;
    bool has_penalties;
    const CudaPreparedPrompt *prepared;
    const CudaDecodeGraphConfig &graph_config;
    mfq_tensor_backend::Tensor full_ids;
    std::int32_t generation_limit = 0;
    bool graph_active = false;
    bool graph_prepared = false;
    int64_t prefill_compute_chunk = 0;
    mfq::engine::InferenceOutput* output = nullptr;
    std::unique_ptr<EagerDecodePool> eager_decode_pool;

    CudaGenerationOps(Model &model, DecodeGraphCache &graph,
        const mfq::engine::InferenceRequest &request, const CudaPreparedPrompt *prepared,
        const CudaDecodeGraphConfig &graph_config)
        : model(model), graph(graph), constraint(request.token_constraint),
          sampler(request.sampling),
          has_penalties(sampler.has_penalties()), prepared(prepared), graph_config(graph_config) {
        full_ids = mfq_tensor_backend::tensor(request.prompt,
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA))
                       .reshape({1, -1})
                       .contiguous();
        graph.ensure_storage(model.vocab_size());
        if (has_penalties) {
            counts = mfq::cuda::SamplingOps::token_counts(full_ids, model.vocab_size(), graph.counts);
        }
        generation_limit = request.sampling.max_tokens;
        graph_active = graph_eligible();
    }

    double prefill(const mfq::engine::PrefillChunk &chunk) {
        using Tensor = mfq_tensor_backend::Tensor;
        PrefillCudaTimer timer;
        auto ids = full_ids.narrow(1, chunk.offset, chunk.count).contiguous();
        Tensor hidden;
        if constexpr(std::is_same_v<Model,DeepseekV41CausalLm>) {
            if(!prepared || !prepared->transformed())
                hidden=deepseek_v41_prefill_hidden(model,ids,prefill_compute_chunk,
                    [&] {return output && output->stopped();});
        }
        if(output && output->stopped()) {
            MFQ_CUDA_CHECK(hipEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
            return timer.elapsed_ms();
        }
        if(hidden.defined()) {
        } else if (prepared && prepared->transformed()) {
            struct ImageMaskScope {
                Model& model;
                ImageMaskScope(Model& m,const Tensor& mask):model(m) {
                    if constexpr(std::is_same_v<Model,DeepseekV41CausalLm>)model.adapter_set_image_mask(mask);
                }
                ~ImageMaskScope() {
                    if constexpr(std::is_same_v<Model,DeepseekV41CausalLm>)model.adapter_set_image_mask({});
                }
            } participation(model,prepared->image_mask.defined()?
                prepared->image_mask.narrow(1,chunk.offset,chunk.count).contiguous():Tensor{});
            hidden = model.hidden_forward_inputs(ids,
                prepared->embeddings.narrow(1, chunk.offset, chunk.count).contiguous(),
                prepared->positions.defined()?MfqOptional<Tensor>(
                    prepared->positions.narrow(-1, chunk.offset, chunk.count).contiguous()):mfq_nullopt,
                mfq_nullopt,
                nullptr,
                mfq_nullopt,
                true);
            model.decode_position_delta = prepared->decode_position_delta;
        } else {
            MfqOptional<Tensor> length = mfq_nullopt;
            if (model.adapter_uses_decode_sequence_length() && chunk.offset > 0 && chunk.count == 1)
                length = mfq_tensor_backend::full({1}, chunk.offset + 1, full_ids.options());
            hidden = model.hidden_forward(ids, mfq_nullopt, length);
        }
        if (chunk.offset + chunk.count == full_ids.size(1)) {
            auto logits = model
                              .logits_from_hidden(hidden.index({Slice(), -1, Slice()})
                                      .to(mfq_tensor_backend::kFloat16)
                                      .contiguous())
                              .reshape({1, -1});
            MFQ_CUDA_CHECK(hipEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
            pending = mfq::cuda::sample_logits(sampler, std::move(logits), counts, constraint);
        } else
            MFQ_CUDA_CHECK(hipEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
        return timer.elapsed_ms();
    }
    std::int64_t first_token() { return pending.template item<int64_t>(); }

    bool graph_eligible() const {
        // Engram hashes and demand-paged rows are published by the CPU for
        // each token; they are outside the generic whole-model graph contract.
        if constexpr(std::is_same_v<Model,DeepseekV41CausalLm>)return false;
        const auto &execution = *model.execution;
        return graph_config.enabled && (!prepared || !prepared->transformed()) && !constraint &&
               !model.metadata.flash_next && mfq_hip_graph_capture_supported() &&
               execution.dsv4_cpu_offload_layers.empty() && execution.dense_cpu_layer_count == 0 &&
               !execution.moe_expert_cache && model_parallel_cuda_graph_enabled(execution) &&
               generation_limit >= graph_config.minimum_generation_tokens &&
               generation_limit <= graph.generated_capacity;
    }
    void prepare_graph() {
        const auto &execution = *model.execution;
        graph.ensure_compute_streams(execution.tensor_parallel.enabled()
                                         ? execution.tensor_parallel
                                         : execution.expert_parallel);
        MfqCudaGuard graph_device_guard(graph.stream.device_index());
        auto graph_stream_guards = activate_cuda_graph_compute_streams(graph.compute_streams);
        hipStream_t graph_stream = graph.stream.stream();
        if (has_penalties) {
            sample_token_counts_add_cuda(counts, pending.contiguous());
        }

        std::int64_t position = model.cache_pos;
        std::int64_t length = position + 1;
        std::int64_t step = 1;
        MFQ_CUDA_CHECK(hipMemcpyAsync(graph.static_input.template data_ptr<int64_t>(),
            pending.template data_ptr<int64_t>(),
            sizeof(int64_t),
            hipMemcpyDeviceToDevice,
            graph_stream));
        MFQ_CUDA_CHECK(hipMemcpyAsync(graph.generated.template data_ptr<int64_t>(),
            pending.template data_ptr<int64_t>(),
            sizeof(int64_t),
            hipMemcpyDeviceToDevice,
            graph_stream));
        MFQ_CUDA_CHECK(hipMemcpyAsync(graph.static_pos.template data_ptr<int64_t>(),
            &position,
            sizeof(int64_t),
            hipMemcpyHostToDevice,
            graph_stream));
        MFQ_CUDA_CHECK(hipMemcpyAsync(graph.static_len.template data_ptr<int64_t>(),
            &length,
            sizeof(int64_t),
            hipMemcpyHostToDevice,
            graph_stream));
        MFQ_CUDA_CHECK(hipMemcpyAsync(graph.static_step.template data_ptr<int64_t>(),
            &step,
            sizeof(int64_t),
            hipMemcpyHostToDevice,
            graph_stream));
        *sampler.ops().random_host().template data_ptr<float>() = 0.5f;
        MFQ_CUDA_CHECK(hipMemcpyAsync(graph.random.template data_ptr<float>(),
            sampler.ops().random_host().template data_ptr<float>(),
            sizeof(float),
            hipMemcpyHostToDevice,
            graph_stream));
        MFQ_CUDA_CHECK(hipStreamSynchronize(graph_stream));

        const std::int64_t requested = model.cache_pos + generation_limit;
        const std::int64_t planned =
            decode_graph_bucket(requested, model.max_position_embeddings());
        const std::int64_t parts =
            decode_graph_attention_parts(planned, FullBlock::kDecodeAttentionMaxParts);
        const bool greedy = sampler.greedy();
        const auto sample_static = [&]() {
            if (greedy && !has_penalties) {
                return model.next_token_static(
                    graph.static_input, graph.static_pos, graph.static_len, {planned, parts});
            }
            auto logits =
                model
                    .last_logits_static(
                        graph.static_input, graph.static_pos, graph.static_len, {planned, parts})
                    .contiguous()
                    .view({1, -1});
            if (has_penalties) {
                logits = sampler.apply_penalties(std::move(logits), counts);
            }
            if (greedy) {
                return sampler.ops().sample_greedy(std::move(logits));
            }
            return sampler.ops().sample_stochastic(
                std::move(logits), graph.random, sampler.params());
        };
        const bool hit = graph.ensure_captured(model, planned, sampler.params(), greedy, [&]() {
            return sample_static();
        }, [&](const mfq_tensor_backend::Tensor &next) {
            if (has_penalties) {
                sample_token_counts_add_cuda(counts, next.contiguous());
            }
            decode_graph_commit_cuda(next,
                graph.generated,
                graph.static_step,
                graph.static_input,
                graph.static_pos,
                graph.static_len);
        });
        if (!hit)
            report_cuda_memory(execution.config, "runtime_graph_capture");
        if (graph_config.trace) {
            std::cerr << "runtime_cuda_graph action=" << (hit ? "reuse" : "capture")
                      << " requested_len=" << requested << " planned_len=" << planned
                      << " captures=" << graph.captures << " reuses=" << graph.reuses << std::endl;
        }
    }
    std::int64_t advance() {
        if (!graph_active) {
            if constexpr (std::is_same_v<Model, DeepseekV41CausalLm>) {
                static const bool enabled = [] {
                    const char* value = std::getenv("MFQ_HIP_EAGER_DECODE_POOL");
                    return !value || value[0] != '0';
                }();
                if (enabled && !eager_decode_pool)
                    eager_decode_pool = std::make_unique<EagerDecodePool>();
            }
            pending = sample_token(model, pending.reshape({1, 1}), sampler, counts, constraint);
            return pending.template item<int64_t>();
        }

        if (!graph_prepared) {
            prepare_graph();
            graph_prepared = true;
        }
        MfqCudaGuard graph_device_guard(graph.stream.device_index());
        auto graph_stream_guards = activate_cuda_graph_compute_streams(graph.compute_streams);
        hipStream_t graph_stream = graph.stream.stream();
        if (!sampler.greedy()) {
            *sampler.ops().random_host().template data_ptr<float>() = sampler.next_uniform_float();
            MFQ_CUDA_CHECK(hipMemcpyAsync(graph.random.template data_ptr<float>(),
                sampler.ops().random_host().template data_ptr<float>(),
                sizeof(float),
                hipMemcpyHostToDevice,
                graph_stream));
        }
        graph.graph->replay();
        const auto token = graph.static_next.template item<int64_t>();
        ++model.cache_pos;
        return token;
    }
    void accept(std::int64_t) {
        if (has_penalties && !graph_active) {
            sample_token_counts_add_cuda(counts, pending.contiguous());
        }
    }
};

} // namespace

template <class Model> struct CudaGenerationContext {
    using Hidden = mfq_tensor_backend::Tensor;
    using Restore = TextSessionRestore;
    Model &model;
    DecodeGraphCache &graph;
    TextSessionCache &cache;
    const CudaRuntimeConfig &config;
    MtpModule *mtp;
    std::optional<CudaPreparedPrompt> prepared;
    mfq::engine::ContinuousBatch<QwenBatchOperations>* batching;
    std::string request_id;

    bool batched() const { return batching != nullptr; }
    void invalidate_plan() { graph.invalidate(); }
    bool has_hidden(const Hidden &value) const { return value.defined(); }
    void reset() {
        if (batching) return;
        model.reset(1);
        if (mtp)
            mtp->reset(1);
    }
    void release() {
        if (batching) return; // Each row owns its slot cleanup.
        try { reset(); }
        catch (...) { graph.invalidate(); throw; }
        if (std::uncaught_exceptions()) graph.invalidate();
    }
    Generation plain(mfq::engine::InferenceRequest& input, mfq::engine::InferenceOutput& output,
                     size_t reused, size_t stable) {
        if (batching) return batching->generate(request_id, input, output);
        auto operations=CudaGenerationOps<Model>(model,graph,input,prepared?&*prepared:nullptr,config.decode_graph);
        operations.prefill_compute_chunk=config.generation.prefill_chunk_size;operations.output=&output;
        auto logical_chunk=config.generation.prefill_chunk_size;
        if constexpr(std::is_same_v<Model,DeepseekV41CausalLm>)
            if(!prepared || !prepared->transformed())logical_chunk=input.prompt.size();
        return mfq::engine::generate_sequence(std::move(operations),output,input.prompt.size(),reused,stable,logical_chunk);
    }
    Generation speculate(mfq::engine::InferenceRequest &input, mfq::engine::InferenceOutput &output,
        size_t reused, Hidden restored, Hidden *committed) {
        return run_mtp_generation(model,
            *mtp,
            input,
            output,
            config.generation.prefill_chunk_size,
            prepared ? &*prepared : nullptr,
            reused,
            std::move(restored),
            committed);
    }
};

template <typename Model>
Generation generate(Model &model, DecodeGraphCache &graph, TextSessionCache &cache,
    const CudaRuntimeConfig &config, mfq::engine::InferenceRequest &request,
    mfq::engine::InferenceOutput &output, MtpModule *mtp,
    std::optional<CudaPreparedPrompt> prepared,
    mfq::engine::ContinuousBatch<QwenBatchOperations>* batching, std::string request_id) {
    return mfq::engine::generate_request(
        CudaGenerationContext<Model>{model, graph, cache, config, mtp, std::move(prepared), batching,
                                     std::move(request_id)},
        request,
        output);
}

#define MFQ_INSTANTIATE_FLOW(MODEL)                                                                \
    template Generation generate(MODEL &,                                                          \
        DecodeGraphCache &,                                                                        \
        TextSessionCache &,                                                                        \
        const CudaRuntimeConfig &,                                                                 \
        mfq::engine::InferenceRequest &,                                                           \
        mfq::engine::InferenceOutput &,                                                            \
        MtpModule *,                                                                               \
        std::optional<CudaPreparedPrompt>,                                                         \
        mfq::engine::ContinuousBatch<QwenBatchOperations>*, std::string);
MFQ_INSTANTIATE_FLOW(mfq::cuda::Qwen35CausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::MiniCPMO45CausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::MiniCPMOTtsCausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::Gemma4CausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::GlmDsaCausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::Glm5CausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::Qwen4CausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::DeepseekV4CausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::DeepseekV41CausalLm)
MFQ_INSTANTIATE_FLOW(mfq::cuda::NemotronHCausalLm)

#undef MFQ_INSTANTIATE_FLOW
} // namespace mfq::cuda::internal
