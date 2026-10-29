#include "storage/transformer_loader.h"
#include "storage/weight_loader.h"
#include "ops.h"
#include "storage/session_codec.h"

namespace mfq::cuda::deepseek_v4 {

mfq_tensor_backend::Tensor output_projection(CudaExecutionContext &execution,
                                             mfq_tensor_backend::Tensor attention,
                                             const QuantLinear &output_a,
                                             const QuantLinear &output_b, std::int64_t groups,
                                             bool groupwise, bool profile) {
    auto &profiler = execution.profiler;
    const auto batch = attention.size(0);
    const auto tokens = attention.size(1);
    const auto rows = batch * tokens;
    const auto rank = output_a.out() / groups;
    auto grouped = attention.contiguous()
                       .reshape({rows, groups, output_a.neuron_len()})
                       .to(mfq_tensor_backend::kFloat16);
    const auto measure = [&](const char *name, auto &&operation) {
        return profile ? profiler.measure(name, operation) : operation();
    };
    if(output_a.is_tq()) {
        auto low_rank=measure("dsv4.output_a",[&] {return output_a.tq.forward_groupwise(grouped,groups);});
        return measure("dsv4.output_b",[&] {
            return output_b.forward(execution,low_rank).reshape({batch,tokens,output_b.out()});
        });
    }
    if (groupwise && output_a.is_nint() && output_a.nint.bits == 8 && output_a.nint.gs == 48 &&
        output_a.nint.out == groups * rank) {
        auto low_rank = measure("dsv4.output_a", [&]() {
            return nint_matmul_groupwise_u8(profiler, output_a.nint, grouped, groups);
        });
        return measure("dsv4.output_b", [&]() {
            return output_b.forward(execution, low_rank).reshape({batch, tokens, output_b.out()});
        });
    }
    if (groupwise && output_a.is_mxfp8() && output_a.out() == groups * rank) {
        auto low_rank = measure("dsv4.output_a", [&]() {
            return output_a.forward_mxfp8_groupwise(execution, grouped, groups);
        });
        return measure("dsv4.output_b", [&]() {
            return output_b.forward(execution, low_rank).reshape({batch, tokens, output_b.out()});
        });
    }
    auto expanded = measure("dsv4.output_a", [&]() {
        return output_a.forward(execution, grouped.reshape({rows * groups, grouped.size(-1)}))
            .reshape({rows, groups, groups, rank});
    });
    std::vector<mfq_tensor_backend::Tensor> diagonal;
    diagonal.reserve(static_cast<std::size_t>(groups));
    for (std::int64_t group = 0; group < groups; ++group) {
        diagonal.push_back(expanded.index({Slice(), group, group, Slice()}));
    }
    auto low_rank = mfq_tensor_backend::stack(diagonal, 1)
                        .reshape({rows, output_a.out()})
                        .to(mfq_tensor_backend::kFloat16)
                        .contiguous();
    return measure("dsv4.output_b", [&]() {
        return output_b.forward(execution, low_rank).reshape({batch, tokens, output_b.out()});
    });
}

std::unique_ptr<::Block> load_block(CudaExecutionContext &execution, const mfq::ModelSource &mfq,
                                    const Config &c, int i, const std::string &type,
                                    const std::shared_ptr<::Dsv4SharedState> &state) {
    const auto &config = c;
    if (type != "deepseek_v4" || !state) {
        throw std::runtime_error("invalid DeepSeek V4 block loader state");
    }
    const std::string p = "model.block." + std::to_string(i) + ".";
    auto b = std::make_unique<Dsv4Block>();
    b->layer = i;
    b->max_positions = c.max_position_embeddings;
    b->compress_ratio = config.compress_ratios.at(static_cast<size_t>(i));
    b->hidden_size = c.hidden_size;
    b->heads = c.num_attention_heads;
    b->head_dim = c.head_dim;
    b->groups = c.o_groups;
    b->hc_mult = c.hc_mult;
    b->hc_iterations = c.hc_sinkhorn_iters;
    b->eps = c.rms_norm_eps;
    b->hc_eps = c.hc_eps;
    b->shared_state = state;

    b->attn_norm = load_dense_gpu(execution, mfq, p + "attention.norm.weight");
    b->ffn_norm = load_dense_gpu(execution, mfq, p + "mlp.norm.weight");
    b->q_a_norm = load_dense_gpu(execution, mfq, p + "attention.query_a_norm.weight");
    b->kv_norm = load_dense_gpu(execution, mfq, p + "attention.key_value_a_norm.weight");
    b->sinks = load_dense_gpu(execution, mfq, p + "attention.sink")
                   .to(mfq_tensor_backend::kFloat32)
                   .contiguous();
    b->hc_attn_fn = load_dense_gpu(execution, mfq, p + "attention.mhc.pre.function")
                        .to(mfq_tensor_backend::kFloat32)
                        .contiguous();
    b->hc_attn_scale = load_dense_gpu(execution, mfq, p + "attention.mhc.pre.scale")
                           .to(mfq_tensor_backend::kFloat32)
                           .contiguous();
    b->hc_attn_base = load_dense_gpu(execution, mfq, p + "attention.mhc.pre.base")
                          .to(mfq_tensor_backend::kFloat32)
                          .contiguous();
    b->hc_ffn_fn = load_dense_gpu(execution, mfq, p + "mlp.mhc.pre.function")
                       .to(mfq_tensor_backend::kFloat32)
                       .contiguous();
    b->hc_ffn_scale = load_dense_gpu(execution, mfq, p + "mlp.mhc.pre.scale")
                          .to(mfq_tensor_backend::kFloat32)
                          .contiguous();
    b->hc_ffn_base = load_dense_gpu(execution, mfq, p + "mlp.mhc.pre.base")
                         .to(mfq_tensor_backend::kFloat32)
                         .contiguous();
    b->q_a = load_quant_linear(execution, mfq, p + "attention.query_a.weight");
    b->q_b = load_quant_linear(execution, mfq, p + "attention.query_b.weight");
    b->kv = load_quant_linear(execution, mfq, p + "attention.key_value_a.weight");
    b->output_a = load_quant_linear(execution, mfq, p + "attention.output_a.weight");
    b->output_b = load_quant_linear(execution, mfq, p + "attention.output_b.weight");
    b->attention_rope =
        Dsv4RopeTable(c.max_position_embeddings, c.rope_base, b->compress_ratio,
                      config.compress_rope_base, config.rope_original_positions, config.rope_factor,
                      config.rope_beta_fast, config.rope_beta_slow);

    if (b->compress_ratio > 0) {
        b->compressor.ratio = b->compress_ratio;
        b->compressor.head_dim = c.head_dim;
        b->compressor.overlap = b->compress_ratio == 4;
        b->compressor.cache_quant_mode = 1;
        b->compressor.projection = make_fp32_quant_group(
            execution, load_quant_group(execution, mfq,
                                        {p + "attention.compressor.key_value.weight",
                                         p + "attention.compressor.gate.weight"}));
        b->compressor.ape = load_dense_gpu(execution, mfq, p + "attention.compressor.position")
                                .to(mfq_tensor_backend::kFloat32)
                                .contiguous();
        b->compressor.norm = load_dense_gpu(execution, mfq, p + "attention.compressor.norm.weight")
                                 .to(mfq_tensor_backend::kFloat32)
                                 .contiguous();
    }
    if (b->compress_ratio == 4) {
        b->indexer_compressor.ratio = 4;
        b->indexer_compressor.head_dim = c.index_head_dim;
        b->indexer_compressor.overlap = true;
        b->indexer_compressor.cache_quant_mode = 2;
        b->indexer_compressor.projection = make_fp32_quant_group(
            execution, load_quant_group(execution, mfq,
                                        {p + "attention.indexer.compressor.key_value.weight",
                                         p + "attention.indexer.compressor.gate.weight"}));
        b->indexer_compressor.ape =
            load_dense_gpu(execution, mfq, p + "attention.indexer.compressor.position")
                .to(mfq_tensor_backend::kFloat32)
                .contiguous();
        b->indexer_compressor.norm =
            load_dense_gpu(execution, mfq, p + "attention.indexer.compressor.norm.weight")
                .to(mfq_tensor_backend::kFloat32)
                .contiguous();
        b->indexer_q = load_quant_linear(execution, mfq, p + "attention.indexer.query.weight");
        b->indexer_weight = load_dense_gpu(execution, mfq, p + "attention.indexer.score.weight")
                                .to(mfq_tensor_backend::kFloat32)
                                .contiguous();
    }

    const bool cpu_offload = execution.dsv4_cpu_offload_layers.count(i) != 0;
    b->ffn = load_moe_weights(
        execution, mfq, p + "mlp.",
        {.layer = i, .cpu_offloaded = cpu_offload, .shared_gate_up_compatible_prefix = 0});
    if (cpu_offload) {
        const int64_t gate_up_bytes =
            b->ffn.moe_split_gate_up
                ? b->ffn.moe_gate.mixed_weight_bytes + b->ffn.moe_up.mixed_weight_bytes
                : b->ffn.moe_gate_up.mixed_weight_bytes;
        const int64_t down_bytes = b->ffn.moe_down.mixed_weight_bytes;
        execution.dsv4_cpu_offload_host_bytes += gate_up_bytes + down_bytes;
        std::cerr << "cpu_offload layer=" << i << " gate_up_bytes=" << gate_up_bytes
                  << " down_bytes=" << down_bytes
                  << " total_host_bytes=" << execution.dsv4_cpu_offload_host_bytes << std::endl;
    }
    if (i < config.hash_layer_count) {
        b->ffn.moe_hash_ids = load_dense_gpu(execution, mfq, p + "mlp.router.token_to_expert")
                                  .to(mfq_tensor_backend::kInt32)
                                  .contiguous();
    }
    b->ffn.moe_top_k = static_cast<int>(c.num_experts_per_tok);
    b->ffn.moe_use_sqrt_softplus = true;
    b->ffn.moe_normalize = c.norm_topk_prob;
    b->ffn.moe_delayed_softmax = false;
    b->ffn.moe_shared_ungated = true;
    b->ffn.moe_router_scale = c.routed_scaling_factor;
    b->ffn.swiglu_limit = c.swiglu_limit;
    b->ffn.shared->swiglu_limit = c.swiglu_limit;

    const bool base_shapes =
        b->attn_norm.numel() == c.hidden_size && b->ffn_norm.numel() == c.hidden_size &&
        b->q_a_norm.numel() == c.q_lora_rank && b->kv_norm.numel() == c.head_dim &&
        b->sinks.numel() == c.num_attention_heads && b->q_a.neuron_len() == c.hidden_size &&
        b->q_a.out() == c.q_lora_rank && b->q_b.neuron_len() == c.q_lora_rank &&
        b->q_b.out() == c.num_attention_heads * c.head_dim && b->kv.neuron_len() == c.hidden_size &&
        b->kv.out() == c.head_dim &&
        b->output_a.neuron_len() == c.num_attention_heads * c.head_dim / c.o_groups &&
        b->output_a.out() == c.o_groups * c.o_lora_rank &&
        b->output_b.neuron_len() == c.o_groups * c.o_lora_rank &&
        b->output_b.out() == c.hidden_size;
    const bool gate_up_shapes =
        b->ffn.moe_split_gate_up
            ? b->ffn.moe_gate.n_experts == c.num_experts &&
                  b->ffn.moe_up.n_experts == c.num_experts &&
                  b->ffn.moe_gate.neuron_len == c.hidden_size &&
                  b->ffn.moe_up.neuron_len == c.hidden_size &&
                  b->ffn.moe_gate.out_per_expert == c.moe_intermediate_size &&
                  b->ffn.moe_up.out_per_expert == c.moe_intermediate_size
            : b->ffn.moe_gate_up.n_experts == c.num_experts &&
                  b->ffn.moe_gate_up.neuron_len == c.hidden_size &&
                  b->ffn.moe_gate_up.out_per_expert == 2 * c.moe_intermediate_size;
    const bool moe_shapes = gate_up_shapes && b->ffn.moe_down.n_experts == c.num_experts &&
                            b->ffn.moe_down.neuron_len == c.moe_intermediate_size &&
                            b->ffn.moe_down.out_per_expert == c.hidden_size &&
                            b->ffn.moe_router.size(0) == c.num_experts &&
                            b->ffn.moe_router.size(1) == c.hidden_size;
    if (!base_shapes || !moe_shapes) {
        throw std::runtime_error("DeepSeek V4 tensor shapes disagree with config at layer " +
                                 std::to_string(i));
    }
    return b;
}

void validate_load_options(const Config &config, CudaExecutionContext &execution) {
    if (config.hidden_size != 4096 || config.num_attention_heads != 64 || config.head_dim != 512 ||
        config.q_lora_rank != 1024 || config.qk_rope_head_dim != 64 ||
        config.index_head_dim != 128 || config.index_n_heads != 64 || config.index_topk != 512 ||
        config.o_groups != 8 || config.o_lora_rank != 1024 || config.hc_mult != 4 ||
        config.hc_sinkhorn_iters != 20 || config.num_experts != 256 ||
        config.num_experts_per_tok != 6 || config.moe_intermediate_size != 2048 ||
        config.shared_expert_count != 1 || config.scoring_func != "sqrtsoftplus") {
        throw std::runtime_error("unsupported DeepSeek V4 CUDA configuration");
    }
    if (execution.dsv4_cpu_offload_layers.empty())
        return;
    for (int layer : execution.dsv4_cpu_offload_layers) {
        if (layer < 0 || layer >= config.num_hidden_layers) {
            throw std::runtime_error("CPU-offload layer is outside the model: " +
                                     std::to_string(layer));
        }
    }
    execution.dsv4_cpu_offload_host_bytes = 0;
    execution.drop_file_cache = true;
}

OutputHeadWeights load_output_head(CudaExecutionContext &execution,
                                   const mfq::ModelSource &source) {
    return {
        load_dense_gpu(execution, source, "model.mhc.output.function")
            .to(mfq_tensor_backend::kFloat32)
            .contiguous(),
        load_dense_gpu(execution, source, "model.mhc.output.scale")
            .to(mfq_tensor_backend::kFloat32)
            .contiguous(),
        load_dense_gpu(execution, source, "model.mhc.output.base")
            .to(mfq_tensor_backend::kFloat32)
            .contiguous(),
    };
}

mfq_tensor_backend::Tensor finalize_hidden(mfq_tensor_backend::Tensor hidden,
                                           const OutputHeadWeights &output_head,
                                           const Config &config, CudaProfiler &profiler,
                                           int64_t batch, int64_t tokens) {
    return profiler.measure("model.dsv4_hc_head", [&]() {
        auto flat = hidden.flatten(2).to(mfq_tensor_backend::kFloat32);
        auto inverse_rms =
            mfq_tensor_backend::rsqrt(flat.square().mean(-1, true) + config.rms_norm_eps);
        auto mixes =
            mfq_tensor_backend::matmul(flat, output_head.function.transpose(0, 1)) * inverse_rms;
        auto pre = mfq_tensor_backend::sigmoid(mixes * output_head.scale + output_head.base) +
                   config.hc_eps;
        return (pre.unsqueeze(-1) *
                flat.reshape({batch, tokens, config.hc_mult, config.hidden_size}))
            .sum(2)
            .to(mfq_tensor_backend::kFloat16)
            .contiguous();
    });
}

} // namespace mfq::cuda::deepseek_v4

namespace mfq::cuda {

void DeepseekV4Model::adapter_validate_load_options() const {
    deepseek_v4::validate_load_options(config, *execution);
}

void DeepseekV4Model::adapter_load_final_state(const mfq::ModelSource &source,
                                               mfq_tensor_backend::Tensor &output_norm) {
    CausalResources::adapter_load_final_state(source, output_norm);
    output_head = deepseek_v4::load_output_head(*execution, source);
}

std::unique_ptr<Block> DeepseekV4Model::adapter_load_block(const mfq::ModelSource &source,
                                                           int layer, int device,
                                                           const std::string &type) {
    auto &state = block_states[device];
    if (!state)
        state = std::make_shared<Dsv4SharedState>();
    return deepseek_v4::load_block(*execution, source, config, layer, type, state);
}

TextSessionStateKind CudaSessionCodec<DeepseekV4Model>::kind(const Model &model) {
    return !model.blocks.empty() &&
                   std::all_of(model.blocks.begin(), model.blocks.end(),
                               [](const std::unique_ptr<::Block> &block) {
                                   return dynamic_cast<const Dsv4Block *>(block.get()) != nullptr;
                               })
               ? TextSessionStateKind::DeepseekV4
               : TextSessionStateKind::Unsupported;
}

bool CudaSessionCodec<DeepseekV4Model>::supports_paged(const Model &) { return false; }

TextSessionState CudaSessionCodec<DeepseekV4Model>::capture(const Model &model,
                                                            const std::vector<int64_t> &tokens) {
    if (kind(model) != TextSessionStateKind::DeepseekV4 || model.cache_pos <= 0 ||
        static_cast<size_t>(model.cache_pos) != tokens.size()) {
        throw std::runtime_error("DeepSeek-V4 text session state is unavailable");
    }
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = model.cache_pos;
    state.payload = std::vector<Dsv4BlockSessionState>{};
    auto &layers = std::get<std::vector<Dsv4BlockSessionState>>(state.payload);
    layers.reserve(model.blocks.size());
    for (const auto &block : model.blocks) {
        MfqCudaGuard guard(block->cuda_device);
        const auto *dsv4 = dynamic_cast<const Dsv4Block *>(block.get());
        if (dsv4 == nullptr || !dsv4->local_cache.defined()) {
            throw std::runtime_error("DeepSeek V4 local session cache is unavailable");
        }
        Dsv4BlockSessionState saved;
        saved.local_cache = dsv4->local_cache.clone();
        state.bytes += session_tensor_bytes(saved.local_cache);
        saved.compressor =
            capture_dsv4_pool_session_state(dsv4->compressor, model.cache_pos, state.bytes);
        saved.indexer_compressor =
            capture_dsv4_pool_session_state(dsv4->indexer_compressor, model.cache_pos, state.bytes);
        layers.push_back(std::move(saved));
    }
    return state;
}

void CudaSessionCodec<DeepseekV4Model>::restore(Model &model, const TextSessionState &state) {
    const auto *layers = std::get_if<std::vector<Dsv4BlockSessionState>>(&state.payload);
    if (kind(model) != TextSessionStateKind::DeepseekV4 ||
        state.kind() != TextSessionStateKind::DeepseekV4 || state.cache_pos <= 0 ||
        static_cast<size_t>(state.cache_pos) != state.tokens.size() || layers == nullptr ||
        layers->size() != model.blocks.size()) {
        throw CudaSessionStateError("DeepSeek V4 text session state is incompatible");
    }
    for (size_t index = 0; index < model.blocks.size(); ++index) {
        auto &block = model.blocks[index];
        MfqCudaGuard guard(block->cuda_device);
        auto *dsv4 = dynamic_cast<Dsv4Block *>(block.get());
        const auto &saved = (*layers)[index];
        if (dsv4 == nullptr || !saved.local_cache.defined() || saved.local_cache.dim() != 3 ||
            saved.local_cache.size(0) != 1 || saved.local_cache.size(1) != 128) {
            throw CudaSessionStateError("DeepSeek V4 saved local cache is invalid");
        }
        restore_session_tensor(dsv4->local_cache, saved.local_cache);
        restore_dsv4_pool_session_state(dsv4->compressor, saved.compressor);
        restore_dsv4_pool_session_state(dsv4->indexer_compressor, saved.indexer_compressor);
        dsv4->shared_state->ensure();
    }
    model.cache_pos = state.cache_pos;
}

mfq_tensor_backend::Tensor DeepseekV4Model::collapse_hidden(mfq_tensor_backend::Tensor hidden,
                                                            int64_t batch, int64_t tokens) const {
    return deepseek_v4::finalize_hidden(std::move(hidden), output_head, config, execution->profiler,
                                        batch, tokens);
}

mfq_tensor_backend::Tensor
DeepseekV4Model::normalize_hidden(mfq_tensor_backend::Tensor hidden,
                                  const mfq_tensor_backend::Tensor &output_norm, int64_t batch,
                                  int64_t tokens) const {
    return execution->profiler.measure("model.output_norm", [&]() {
        return qwen_rms_norm(hidden.reshape({batch * tokens, metadata.hidden_size})
                                 .to(mfq_tensor_backend::kFloat32),
                             output_norm, metadata.rms_norm_eps, metadata.norm_weight_offset)
            .reshape({batch, tokens, metadata.hidden_size});
    });
}

} // namespace mfq::cuda

Dsv4PoolSessionState capture_dsv4_pool_session_state(const Dsv4PoolState &source, int64_t cache_pos,
                                                     size_t &bytes) {
    Dsv4PoolSessionState state;
    state.ratio = source.ratio;
    state.head_dim = source.head_dim;
    state.cache_quant_mode = source.cache_quant_mode;
    state.capacity = source.capacity;
    state.overlap = source.overlap;
    if (source.ratio <= 0)
        return state;
    if (source.capacity <= 0 || !source.state_kv.defined() || !source.state_gate.defined() ||
        !source.pool.defined() || source.pool.dim() != 3 || source.pool.size(0) != 1 ||
        source.pool.size(1) != source.capacity ||
        (source.overlap && (!source.previous_kv.defined() || !source.previous_gate.defined()))) {
        throw std::runtime_error("DeepSeek V4 session compressor state is unavailable");
    }
    const int64_t visible = std::min<int64_t>(cache_pos / source.ratio, source.capacity);
    state.state_kv = source.state_kv.clone();
    state.state_gate = source.state_gate.clone();
    if (source.overlap) {
        state.previous_kv = source.previous_kv.clone();
        state.previous_gate = source.previous_gate.clone();
    }
    state.pool = source.pool.narrow(1, 0, visible).clone();
    bytes += session_tensor_bytes(state.state_kv);
    bytes += session_tensor_bytes(state.state_gate);
    bytes += session_tensor_bytes(state.previous_kv);
    bytes += session_tensor_bytes(state.previous_gate);
    bytes += session_tensor_bytes(state.pool);
    return state;
}

void restore_dsv4_pool_session_state(Dsv4PoolState &target, const Dsv4PoolSessionState &state) {
    if (target.ratio != state.ratio || target.head_dim != state.head_dim ||
        target.cache_quant_mode != state.cache_quant_mode || target.overlap != state.overlap) {
        throw CudaSessionStateError("DeepSeek V4 session compressor configuration changed");
    }
    if (state.ratio <= 0)
        return;
    if (state.capacity <= 0 || !state.state_kv.defined() || !state.state_gate.defined() ||
        !state.pool.defined() || state.pool.dim() != 3 || state.pool.size(0) != 1 ||
        state.pool.size(1) > state.capacity ||
        (state.overlap && (!state.previous_kv.defined() || !state.previous_gate.defined()))) {
        throw CudaSessionStateError("DeepSeek V4 saved compressor state is invalid");
    }
    target.capacity = state.capacity;
    restore_session_tensor(target.state_kv, state.state_kv);
    restore_session_tensor(target.state_gate, state.state_gate);
    if (state.overlap) {
        restore_session_tensor(target.previous_kv, state.previous_kv);
        restore_session_tensor(target.previous_gate, state.previous_gate);
    } else {
        target.previous_kv = mfq_tensor_backend::Tensor();
        target.previous_gate = mfq_tensor_backend::Tensor();
    }
    restore_session_prefix_tensor(target.pool, state.pool, 1, state.capacity);
}

namespace mfq::cuda {} // namespace mfq::cuda

namespace mfq::models {
template struct deepseek_v4::CausalLm<cuda::CudaCausalOps<cuda::DeepseekV4Model>>;
} // namespace mfq::models
