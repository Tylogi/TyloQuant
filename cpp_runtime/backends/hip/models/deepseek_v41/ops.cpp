#include "storage/transformer_loader.h"
#include "storage/weight_loader.h"
#include "ops.h"
#include "storage/session_codec.h"

namespace mfq::cuda::deepseek_v41_runtime {

FFN load_moe_at(CudaExecutionContext &execution, const mfq::ModelSource &model,
                const CommonConfig &config, const std::string &prefix, std::int64_t layer,
                std::int64_t top_k) {
    auto result = load_moe_weights(execution, model, prefix,
                                   {.layer = static_cast<int>(layer),
                                    .router_bias_required = true,
                                    .shared_gate_up_compatible_prefix = 0});
    result.moe_top_k = static_cast<int>(top_k);
    result.moe_use_sqrt_softplus = true;
    result.moe_normalize = config.norm_topk_prob;
    result.moe_delayed_softmax = false;
    result.moe_shared_ungated = true;
    result.moe_router_scale = config.routed_scaling;
    result.swiglu_limit = config.swiglu_limit;
    result.shared->swiglu_limit = config.swiglu_limit;
    return result;
}

std::unique_ptr<::Block> load_block(CudaExecutionContext &execution, const mfq::ModelSource &model,
                                    std::int64_t layer,
                                    const std::shared_ptr<SharedState> &shared) {
    MFQ_RUNTIME_CHECK(shared, "missing DeepSeek-V4.1 CUDA state");
    const auto &config = shared->config;
    const auto prefix = "model.block." + std::to_string(layer) + ".";
    auto result = std::make_unique<Block>();
    result->config = config;
    result->layer = layer;
    result->ratio = config.compress_ratios.at(static_cast<std::size_t>(layer));
    result->max_context = config.max_position_embeddings;
    result->shared = shared;
    if (config.has_engram(layer)) {
        result->engram = Engram::load(execution, model, config, static_cast<int>(layer));
        shared->engram_prefetch.push_back(result->engram.get());
    }
    result->attention_norm = load_dense_gpu(execution, model, prefix + "attention.norm.weight")
                                 .to(mfq_tensor_backend::kFloat32).contiguous();
    result->mlp_norm = load_dense_gpu(execution, model, prefix + "mlp.norm.weight")
                           .to(mfq_tensor_backend::kFloat32).contiguous();
    result->query_a_norm =
        load_dense_gpu(execution, model, prefix + "attention.query_a_norm.weight");
    result->key_value_norm =
        load_dense_gpu(execution, model, prefix + "attention.key_value_norm.weight");
    result->sinks = load_dense_gpu(execution, model, prefix + "attention.sink")
                        .to(mfq_tensor_backend::kFloat32)
                        .contiguous();
    result->attention_mhc_function =
        load_dense_gpu(execution, model, prefix + "attention.mhc.pre.function")
            .to(mfq_tensor_backend::kFloat32)
            .contiguous();
    result->attention_mhc_scale =
        load_dense_gpu(execution, model, prefix + "attention.mhc.pre.scale")
            .to(mfq_tensor_backend::kFloat32)
            .contiguous();
    result->attention_mhc_base = load_dense_gpu(execution, model, prefix + "attention.mhc.pre.base")
                                     .to(mfq_tensor_backend::kFloat32)
                                     .contiguous();
    result->mlp_mhc_function = load_dense_gpu(execution, model, prefix + "mlp.mhc.pre.function")
                                   .to(mfq_tensor_backend::kFloat32)
                                   .contiguous();
    result->mlp_mhc_scale = load_dense_gpu(execution, model, prefix + "mlp.mhc.pre.scale")
                                .to(mfq_tensor_backend::kFloat32)
                                .contiguous();
    result->mlp_mhc_base = load_dense_gpu(execution, model, prefix + "mlp.mhc.pre.base")
                               .to(mfq_tensor_backend::kFloat32)
                               .contiguous();
    result->query_a = load_quant_linear(execution, model, prefix + "attention.query_a.weight");
    result->query_b = load_quant_linear(execution, model, prefix + "attention.query_b.weight");
    result->key_value = load_quant_linear(execution, model, prefix + "attention.key_value.weight");
    result->output_a = load_quant_linear(execution, model, prefix + "attention.output_a.weight");
    result->output_b = load_quant_linear(execution, model, prefix + "attention.output_b.weight");
    if (result->kv_source()) {
        result->compressor_key_value =
            load_quant_linear(execution, model, prefix + "attention.compressor.key_value.weight");
        result->compressor_norm =
            load_dense_gpu(execution, model, prefix + "attention.compressor.norm.weight");
        result->index_key =
            load_quant_linear(execution, model, prefix + "attention.indexer.key.weight");
        result->index_key_norm =
            load_dense_gpu(execution, model, prefix + "attention.indexer.key_norm.weight");
        if (result->ratio > 1) {
            result->compressor_gate =
                load_quant_linear(execution, model, prefix + "attention.compressor.gate.weight");
        }
    }
    if (result->index_source()) {
        result->index_query =
            load_quant_linear(execution, model, prefix + "attention.indexer.query.weight");
        result->index_score =
            load_quant_linear(execution, model, prefix + "attention.indexer.score.weight");
    }
    result->mlp = load_moe_at(execution, model, config, prefix + "mlp.", layer, config.top_k);
    if(has_tensor(model,prefix+"mlp.router.vision_bias")) {
        result->vision_router_bias=load_dense_gpu(execution,model,prefix+"mlp.router.vision_bias")
            .to(mfq_tensor_backend::kFloat32).contiguous();
        MFQ_RUNTIME_CHECK(result->vision_router_bias.dim()==1 && result->vision_router_bias.size(0)==config.n_experts,
                          "DeepSeek-V4.1 vision router bias shape mismatch");
    }
    result->rope = Dsv4RopeTable(
        config.max_position_embeddings, config.rope_theta, result->ratio,
        config.compress_rope_theta, config.rope_scaling.original_max_position_embeddings,
        config.rope_scaling.factor, config.rope_scaling.beta_fast, config.rope_scaling.beta_slow);

    const auto function_width = config.hc_mult * config.hidden;
    const auto valid_mhc = [&](const Tensor &function, const Tensor &scale, const Tensor &base,
                               const Tensor &norm) {
        return function.dim() == 2 && function.size(0) == 24 &&
               function.size(1) == function_width && scale.numel() == 3 && base.numel() == 24 &&
               norm.numel() == config.hidden;
    };
    const bool base_shapes =
        valid_mhc(result->attention_mhc_function, result->attention_mhc_scale,
                  result->attention_mhc_base, result->attention_norm) &&
        valid_mhc(result->mlp_mhc_function, result->mlp_mhc_scale, result->mlp_mhc_base,
                  result->mlp_norm) &&
        result->query_a_norm.numel() == config.q_lora_rank &&
        result->key_value_norm.numel() == config.head_dim &&
        result->sinks.numel() == config.n_heads && result->query_a.neuron_len() == config.hidden &&
        result->query_a.out() == config.q_lora_rank &&
        result->query_b.neuron_len() == config.q_lora_rank &&
        result->query_b.out() == config.n_heads * config.head_dim &&
        result->key_value.neuron_len() == config.hidden &&
        result->key_value.out() == config.head_dim &&
        result->output_a.neuron_len() == config.n_heads * config.head_dim / config.o_groups &&
        result->output_a.out() == config.o_groups * config.o_lora_rank &&
        result->output_b.neuron_len() == config.o_groups * config.o_lora_rank &&
        result->output_b.out() == config.hidden;
    const bool source_shapes =
        !result->kv_source() ||
        (result->compressor_key_value.neuron_len() == config.hidden &&
         result->compressor_key_value.out() == config.head_dim &&
         result->compressor_norm.numel() == config.head_dim &&
         result->index_key.neuron_len() == config.head_dim &&
         result->index_key.out() == config.index_head_dim &&
         result->index_key_norm.numel() == config.index_head_dim &&
         (result->ratio == 1 || (result->compressor_gate.neuron_len() == config.hidden &&
                                 result->compressor_gate.out() == config.head_dim)));
    const bool index_shapes =
        !result->index_source() ||
        (result->index_query.neuron_len() == config.q_lora_rank &&
         result->index_query.out() == config.index_n_heads * config.index_head_dim &&
         result->index_score.neuron_len() == config.hidden &&
         result->index_score.out() == config.index_n_heads);
    MFQ_RUNTIME_CHECK(config.hc_mult == 4 && config.n_heads == 64 && config.head_dim == 512 &&
                          config.rope_head_dim == 64 && config.index_n_heads == 32 &&
                          config.index_head_dim == 128 && base_shapes && source_shapes &&
                          index_shapes,
                      "DeepSeek-V4.1 CUDA tensor geometry disagrees at layer ", layer);
    return result;
}

void validate_load_options(const CudaExecutionContext &execution) {
    if (execution.tensor_parallel.enabled() || execution.layer_placement.enabled() ||
        execution.n_gpu_layers >= 0) {
        throw std::runtime_error("DeepSeek-V4.1 native CUDA currently supports single-device dense "
                                 "placement or expert parallelism");
    }
}

std::shared_ptr<SharedState> load_shared_state(const mfq::ModelSource &source,
                                               const CommonConfig &config) {
    auto state = std::make_shared<SharedState>();
    state->config = config;
    state->engram_hash = EngramHashState::load(source, config);
    return state;
}

Tensor finalize_hidden(const std::shared_ptr<SharedState> &state, const Tensor &hidden,
                       CudaProfiler &profiler) {
    MFQ_RUNTIME_CHECK(state, "DeepSeek-V4.1 final state is unavailable");
    return profiler.measure("model.deepseek_v41.final_collapse", [&]() {
        return state->final_collapse(hidden, state->config.n_layers);
    });
}

} // namespace mfq::cuda::deepseek_v41_runtime

namespace mfq::cuda {

void DeepseekV41Model::adapter_validate_load_options() const {
    deepseek_v41_runtime::validate_load_options(*execution);
}

void DeepseekV41Model::adapter_load_final_state(const mfq::ModelSource &source,
                                                mfq_tensor_backend::Tensor &output_norm) {
    CausalResources::adapter_load_final_state(source, output_norm);
}

void DeepseekV41Model::adapter_prepare_blocks(const mfq::ModelSource &source) {
    shared = deepseek_v41_runtime::load_shared_state(source, config);
}

bool DeepseekV41Model::adapter_prefetch_token_rows(DeepseekV41CausalLm& model,
        const mfq_tensor_backend::Tensor& ids) {
    if(!shared || shared->batch!=1 || !shared->engram_hash || shared->engram_prefetch.empty())return false;
    if(!ids.is_cpu() || ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<=0 ||
            ids.size(1)>std::numeric_limits<int>::max() || ids.scalar_type()!=mfq_tensor_backend::kInt64 ||
            !ids.is_contiguous())throw std::invalid_argument("Engram prefetch needs contiguous host token IDs");
    const auto hashes=shared->engram_hash->preview(ids.data_ptr<int64_t>(),1,int(ids.size(1)),model.cache_pos);
    for(const auto* memory:shared->engram_prefetch)memory->prefetch(hashes);
    return true;
}

std::unique_ptr<Block> DeepseekV41Model::adapter_load_block(const mfq::ModelSource &source,
                                                            int layer, int,
                                                            const std::string &type) {
    MFQ_RUNTIME_CHECK(type == "deepseek_v41" && shared, "invalid DeepSeek-V4.1 block loader state");
    return deepseek_v41_runtime::load_block(*execution, source, layer, shared);
}

void DeepseekV41Model::adapter_begin_forward(bool capture_raw_hidden) {
    MFQ_RUNTIME_CHECK(shared, "DeepSeek-V4.1 target capture state is unavailable");
    shared->begin_forward(capture_raw_hidden, shared->config.dspark_target_layer_ids.size());
}

mfq_tensor_backend::Tensor DeepseekV41Model::collapse_hidden(mfq_tensor_backend::Tensor hidden,
                                                             int64_t batch, int64_t tokens) const {
    return deepseek_v41_runtime::finalize_hidden(shared, hidden, execution->profiler);
}

mfq_tensor_backend::Tensor
DeepseekV41Model::normalize_hidden(mfq_tensor_backend::Tensor hidden,
                                   const mfq_tensor_backend::Tensor &output_norm, int64_t batch,
                                   int64_t tokens) const {
    return execution->profiler.measure("model.output_norm", [&]() {
        return qwen_rms_norm(hidden.reshape({batch * tokens, metadata.hidden_size})
                                 .to(mfq_tensor_backend::kFloat32),
                             output_norm, metadata.rms_norm_eps, metadata.norm_weight_offset)
            .reshape({batch, tokens, metadata.hidden_size});
    });
}

mfq_tensor_backend::Tensor
DeepseekV41Model::adapter_raw_hidden(const mfq_tensor_backend::Tensor &,
                                     const mfq_tensor_backend::Tensor &) const {
    return shared->dspark_target_hidden();
}

void DeepseekV41Model::adapter_begin_speculative() {
    MFQ_RUNTIME_CHECK(shared, "DeepSeek-V4.1 speculative state is unavailable");
    shared->begin_speculative();
}

void DeepseekV41Model::adapter_commit_speculative() {
    MFQ_RUNTIME_CHECK(shared, "DeepSeek-V4.1 speculative state is unavailable");
    shared->commit_speculative();
}

void DeepseekV41Model::adapter_rollback_speculative(int64_t position) {
    MFQ_RUNTIME_CHECK(shared, "DeepSeek-V4.1 speculative state is unavailable");
    shared->rollback_speculative(position);
}

} // namespace mfq::cuda

namespace mfq::models {
template struct deepseek_v41::CausalLm<cuda::CudaCausalOps<cuda::DeepseekV41Model>>;
} // namespace mfq::models
