#include "storage/weight_loader.h"
#include "dspark.h"

#include "core/causal_model.h"
#include "core/mtp.h"
#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace mfq::cuda::deepseek_v41_runtime {

static std::pair<Tensor, Tensor>
dspark_full_attention_plan(std::int64_t batch, std::int64_t queries, std::int64_t keys,
                           const mfq_tensor_backend::Device &device) {
    MFQ_RUNTIME_CHECK(batch > 0 && queries > 0 && keys > 0,
                      "invalid DeepSeek-V4.1 DSpark attention plan geometry");
    const auto selected = ((keys + 31) / 32) * 32;
    auto integer =
        mfq_tensor_backend::TensorOptions().device(device).dtype(mfq_tensor_backend::kInt64);
    auto positions = mfq_tensor_backend::arange(selected, integer);
    auto valid = positions < static_cast<double>(keys);
    auto indices = mfq_tensor_backend::where(valid, positions, 0.0)
                       .to(mfq_tensor_backend::kInt32)
                       .reshape({1, 1, selected})
                       .expand({batch, queries, selected})
                       .contiguous();
    auto mask =
        mfq_tensor_backend::where(
            valid,
            mfq_tensor_backend::zeros({selected}, integer.dtype(mfq_tensor_backend::kFloat16)),
            mfq_tensor_backend::full({selected}, -std::numeric_limits<float>::infinity(),
                                     integer.dtype(mfq_tensor_backend::kFloat16)))
            .reshape({1, 1, selected})
            .expand({batch, queries, selected})
            .contiguous();
    return {std::move(indices), std::move(mask)};
}

struct DeepseekV41Dspark final : MtpModule {
    CudaExecutionContext *execution = nullptr;
    CommonConfig config;
    QuantLinear main_projection;
    Tensor main_norm;
    Tensor output_norm;
    QuantLinear markov_embedding;
    QuantLinear markov_output;
    QuantLinear confidence_projection;
    std::vector<std::unique_ptr<Block>> stages;
    std::vector<Tensor> rings;
    Tensor attention_meta;
    std::int64_t maximum_context = 0;
    std::int64_t batch = 0;
    std::int64_t position = 0;

    static std::unique_ptr<Block> load_stage(CudaExecutionContext &execution,
                                             const mfq::ModelSource &model,
                                             const CommonConfig &config, std::int64_t stage) {
        const auto prefix = "predictor.stage." + std::to_string(stage) + ".";
        auto result = std::make_unique<Block>();
        result->config = config;
        result->layer = config.n_layers + stage;
        result->ratio = 0;
        result->max_context = config.max_position_embeddings;
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
        result->attention_mhc_base =
            load_dense_gpu(execution, model, prefix + "attention.mhc.pre.base")
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
        result->key_value =
            load_quant_linear(execution, model, prefix + "attention.key_value.weight");
        result->output_a =
            load_quant_linear(execution, model, prefix + "attention.output_a.weight");
        result->output_b =
            load_quant_linear(execution, model, prefix + "attention.output_b.weight");
        // Predictor bundles must not extend the backbone's final-layer marker:
        // residency updates also run when ordinary decoding skips DSpark.
        result->mlp = load_moe_at(execution, model, config, prefix + "mlp.",
                                  -1 - stage, config.dspark_top_k);
        result->rope = Dsv4RopeTable(config.max_position_embeddings, config.rope_theta, 0);
        result->cuda_device = execution.layer_placement.primary_device();

        const auto function_width = config.hc_mult * config.hidden;
        const auto valid_mhc = [&](const Tensor &function, const Tensor &scale, const Tensor &base,
                                   const Tensor &norm) {
            return function.dim() == 2 && function.size(0) == 24 &&
                   function.size(1) == function_width && scale.numel() == 3 && base.numel() == 24 &&
                   norm.numel() == config.hidden;
        };
        const auto routed_experts = result->mlp.moe_tq
                                        ? result->mlp.moe_tq->gate->experts
                                        : result->mlp.moe_split_gate_up
                                        ? result->mlp.moe_gate.n_experts
                                        : result->mlp.moe_gate_up.n_experts;
        MFQ_RUNTIME_CHECK(valid_mhc(result->attention_mhc_function, result->attention_mhc_scale,
                                    result->attention_mhc_base, result->attention_norm) &&
                              valid_mhc(result->mlp_mhc_function, result->mlp_mhc_scale,
                                        result->mlp_mhc_base, result->mlp_norm) &&
                              result->query_a.neuron_len() == config.hidden &&
                              result->query_a.out() == config.q_lora_rank &&
                              result->query_b.neuron_len() == config.q_lora_rank &&
                              result->query_b.out() == config.n_heads * config.head_dim &&
                              result->key_value.neuron_len() == config.hidden &&
                              result->key_value.out() == config.head_dim &&
                              result->output_a.neuron_len() ==
                                  config.n_heads * config.head_dim / config.o_groups &&
                              result->output_a.out() == config.o_groups * config.o_lora_rank &&
                              result->output_b.neuron_len() ==
                                  config.o_groups * config.o_lora_rank &&
                              result->output_b.out() == config.hidden &&
                              result->query_a_norm.numel() == config.q_lora_rank &&
                              result->key_value_norm.numel() == config.head_dim &&
                              result->sinks.numel() == config.n_heads &&
                              routed_experts == config.dspark_n_experts &&
                              (result->mlp.moe_tq?result->mlp.moe_tq->down->experts:
                                  result->mlp.moe_down.n_experts) == config.dspark_n_experts,
                          "DeepSeek-V4.1 DSpark stage tensor geometry disagrees at stage ", stage);
        return result;
    }

    static std::optional<DeepseekV41Dspark> load_if_present(CudaExecutionContext &execution,
                                                            const mfq::ModelSource &model,
                                                            const CommonConfig &config) {
        const bool root = has_tensor(model, "predictor.stage.0.main_projection.weight");
        const bool any = root || has_tensor(model, "predictor.stage.0.attention.query_a.weight") ||
                         has_tensor(model, "predictor.stage.0.mlp.router.weight") ||
                         has_tensor(model, "predictor.stage.0.output_norm.weight");
        if (!root) {
            MFQ_RUNTIME_CHECK(!any,
                              "DeepSeek-V4.1 model source contains an incomplete DSpark head");
            return std::nullopt;
        }
        MFQ_RUNTIME_CHECK(config.has_dspark(),
                          "DeepSeek-V4.1 model source has DSpark tensors without configuration");
        DeepseekV41Dspark result;
        result.execution = &execution;
        result.config = config;
        result.maximum_context = config.max_position_embeddings;
        const auto first = std::string("predictor.stage.0.");
        const auto last = "predictor.stage." + std::to_string(result.config.n_mtp_layers - 1) + ".";
        result.main_projection =
            load_quant_linear(execution, model, first + "main_projection.weight");
        result.main_norm = load_dense_gpu(execution, model, first + "main_norm.weight");
        result.output_norm = load_dense_gpu(execution, model, last + "output_norm.weight");
        result.markov_embedding =
            load_quant_linear(execution, model, last + "markov.embedding.weight");
        result.markov_output = load_quant_linear(execution, model, last + "markov.output.weight");
        result.confidence_projection =
            load_quant_linear(execution, model, last + "confidence.projection.weight");
        result.stages.reserve(static_cast<std::size_t>(result.config.n_mtp_layers));
        for (std::int64_t stage = 0; stage < result.config.n_mtp_layers; ++stage) {
            result.stages.push_back(load_stage(execution, model, result.config, stage));
        }
        const auto target_width =
            result.config.hidden *
            static_cast<std::int64_t>(result.config.dspark_target_layer_ids.size());
        MFQ_RUNTIME_CHECK(
            result.config.hc_mult == 4 && result.config.n_heads == 64 &&
                result.config.head_dim == 512 && result.config.rope_head_dim == 64 &&
                result.main_projection.neuron_len() == target_width &&
                result.main_projection.out() == result.config.hidden &&
                result.main_norm.numel() == result.config.hidden &&
                result.output_norm.numel() == result.config.hidden &&
                result.markov_embedding.out() == result.config.vocab &&
                result.markov_embedding.neuron_len() == result.config.dspark_markov_rank &&
                result.markov_output.neuron_len() == result.config.dspark_markov_rank &&
                result.markov_output.out() == result.config.vocab &&
                result.confidence_projection.neuron_len() ==
                    result.config.hidden + result.config.dspark_markov_rank &&
                result.confidence_projection.out() == 1,
            "DeepSeek-V4.1 DSpark top-level tensor geometry disagrees");
        return result;
    }

    void reset(std::int64_t next_batch = 1) override {
        MFQ_RUNTIME_CHECK(next_batch > 0, "DeepSeek-V4.1 DSpark batch must be positive");
        auto half = mfq_tensor_backend::TensorOptions()
                        .device(mfq_tensor_backend::kCUDA)
                        .dtype(mfq_tensor_backend::kFloat16);
        rings.clear();
        rings.reserve(stages.size());
        for (std::size_t stage = 0; stage < stages.size(); ++stage) {
            rings.push_back(mfq_tensor_backend::zeros(
                {next_batch, config.sliding_window, config.head_dim}, half));
        }
        attention_meta =
            mfq_tensor_backend::empty({8 * 1024 * 1024}, half.dtype(mfq_tensor_backend::kFloat32));
        batch = next_batch;
        position = 0;
    }

    Tensor forward(const MtpTarget &, Tensor, Tensor) override {
        throw std::runtime_error("DeepSeek-V4.1 DSpark uses blockwise drafting");
    }

    std::int64_t cache_position() const noexcept override { return position; }

    bool supports_session_state() const noexcept override {return true;}

    MtpSessionState capture_session_state(int64_t cache_pos,const Tensor& hidden) const override {
        const auto width=config.hidden*int64_t(config.dspark_target_layer_ids.size());
        MFQ_RUNTIME_CHECK(batch==1 && cache_pos==position && position>0 && rings.size()==stages.size() &&
            hidden.defined() && hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)>0 && hidden.size(2)==width,
            "DeepSeek-V4.1 DSpark session position or hidden state disagrees");
        MtpSessionState state;state.cache_pos=position;state.last_target_hidden=hidden.clone();
        state.bytes=session_tensor_bytes(state.last_target_hidden);
        for(const auto& ring:rings) {
            MFQ_RUNTIME_CHECK(ring.dim()==3 && ring.size(0)==1 && ring.size(1)==config.sliding_window &&
                ring.size(2)==config.head_dim,"DeepSeek-V4.1 DSpark session ring disagrees");
            state.deepseek_v41_rings.push_back(ring.clone());
            state.bytes+=session_tensor_bytes(state.deepseek_v41_rings.back());
        }
        return state;
    }

    void restore_session_state(const MtpSessionState& state) override {
        const auto width=config.hidden*int64_t(config.dspark_target_layer_ids.size());
        const auto& hidden=state.last_target_hidden;
        if(state.cache_pos<=0 || state.cache_pos>maximum_context ||
                state.deepseek_v41_rings.size()!=stages.size() || !state.blocks.empty() ||
                !state.qwen4_blocks.empty() || !hidden.defined() || !hidden.is_cuda() ||
                hidden.dim()!=3 || hidden.size(0)!=1 || hidden.size(1)<=0 || hidden.size(2)!=width)
            throw CudaSessionStateError("DeepSeek-V4.1 DSpark session state is incompatible");
        for(const auto& ring:state.deepseek_v41_rings)
            if(!ring.defined() || !ring.is_cuda() || ring.scalar_type()!=mfq_tensor_backend::kFloat16 ||
                    ring.dim()!=3 || ring.size(0)!=1 || ring.size(1)!=config.sliding_window ||
                    ring.size(2)!=config.head_dim)
                throw CudaSessionStateError("DeepSeek-V4.1 DSpark saved ring geometry is incompatible");
        if(batch!=1 || rings.size()!=stages.size() || !attention_meta.defined())reset(1);
        for(std::size_t i=0;i<rings.size();++i)restore_session_tensor(rings[i],state.deepseek_v41_rings[i]);
        position=state.cache_pos;
    }

    void trim_cache_to(std::int64_t selected_position) override {
        MFQ_RUNTIME_CHECK(selected_position == position,
                          "DeepSeek-V4.1 DSpark only stores committed context");
    }

    bool teacher_forced_prompt_prime() const noexcept override { return false; }

    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return true; }
    bool blockwise_drafting() const noexcept override { return true; }
    bool dspark_policy() const noexcept override { return true; }
    bool retains_partial_target_prefix() const noexcept override { return true; }

    int maximum_draft_depth() const noexcept override {
        return static_cast<int>(config.dspark_block_size);
    }

    void seed_target_context(Tensor hidden, std::int64_t tokens) override {
        MFQ_RUNTIME_CHECK(position == 0 && hidden.dim() == 3 && hidden.size(1) > 0 &&
            tokens >= hidden.size(1) && tokens <= maximum_context &&
            hidden.size(1) >= std::min(tokens, config.sliding_window),
            "invalid DeepSeek-V4.1 DSpark prompt tail");
        position = tokens - hidden.size(1);
        append_target_context(std::move(hidden), position);
    }

    void append_target_context(Tensor target_hidden, std::int64_t start_position) override {
        const auto target_width =
            config.hidden * static_cast<std::int64_t>(config.dspark_target_layer_ids.size());
        MFQ_RUNTIME_CHECK(target_hidden.dim() == 3 && target_hidden.size(0) == batch &&
                              target_hidden.size(1) > 0 && target_hidden.size(2) == target_width &&
                              start_position >= position && position >= 0 &&
                              (start_position==position || target_hidden.size(1)>=config.sliding_window) &&
                              start_position + target_hidden.size(1) <= maximum_context &&
                              rings.size() == stages.size(),
                          "invalid DeepSeek-V4.1 DSpark context append");
        const auto tokens = target_hidden.size(1);
        const auto retained = std::min(tokens, config.sliding_window);
        target_hidden = target_hidden.narrow(1, tokens - retained, retained).contiguous();
        auto positions =
            mfq_tensor_backend::arange(start_position + tokens - retained, start_position + tokens,
                                       mfq_tensor_backend::TensorOptions()
                                           .device(mfq_tensor_backend::kCUDA)
                                           .dtype(mfq_tensor_backend::kInt64));
        auto slots = positions.remainder(static_cast<double>(config.sliding_window))
                         .to(mfq_tensor_backend::kInt64)
                         .contiguous();
        auto next_position=start_position;
        mfq::models::deepseek_v41::dspark_context(
            next_position, tokens, stages.size(),
            [&] {
                auto floating = target_hidden.to(mfq_tensor_backend::kFloat32);
                auto scale = (floating.abs().amax(-1, true) / 64.0).clamp_min(1.0);
                return weighted_rms(
                    main_projection.forward(
                        *execution,
                        (floating / scale).to(mfq_tensor_backend::kFloat16).contiguous()),
                    main_norm, config.rms_eps);
            },
            [&](size_t index, const Tensor &main_x) {
                auto key_value = weighted_rms(stages[index]->key_value.forward(*execution, main_x),
                                              stages[index]->key_value_norm, config.rms_eps);
                key_value = rotate_token_major_tail(
                    key_value,
                    mfq_tensor_backend::arange(start_position + tokens - retained, start_position + tokens,
                                               positions.options()),
                    stages[index]->rope);
                key_value = deepseek_v41_mxfp8_e4m3_sim_cuda(key_value.contiguous());
                return key_value;
            },
            [&](size_t index, Tensor key_value) { rings[index].index_copy_(1, slots, key_value); });
        position=next_position;
    }

    Tensor attention(const Tensor &input, Block &stage, const Tensor &ring) {
        const auto tokens = input.size(1);
        auto positions = mfq_tensor_backend::arange(position, position + tokens,
                                                    mfq_tensor_backend::TensorOptions()
                                                        .device(mfq_tensor_backend::kCUDA)
                                                        .dtype(mfq_tensor_backend::kInt64));
        const auto active = std::min(position, config.sliding_window);
        return mfq::models::deepseek_v41::dspark_attention(
            active,
            [&] {
                auto q_rank = weighted_rms(stage.query_a.forward(*execution, input),
                                           stage.query_a_norm, config.rms_eps);
                auto query = stage.query_b.forward(*execution, q_rank)
                                 .reshape({batch, tokens, config.n_heads, config.head_dim})
                                 .transpose(1, 2)
                                 .contiguous()
                                 .to(mfq_tensor_backend::kFloat32);
                query = dsv4_rotate_rope_tail(query, positions, stage.rope, false);
                return query;
            },
            [&] {
                auto current_keys = weighted_rms(stage.key_value.forward(*execution, input),
                                                 stage.key_value_norm, config.rms_eps);
                current_keys = rotate_token_major_tail(current_keys, positions, stage.rope);
                current_keys = deepseek_v41_mxfp8_e4m3_sim_cuda(current_keys.contiguous());
                return current_keys;
            },
            [&](Tensor current_keys) {
                return mfq_tensor_backend::cat({ring.narrow(1, 0, active), current_keys}, 1)
                    .contiguous();
            },
            [&](const Tensor &keys) {
                return dspark_full_attention_plan(batch, tokens, keys.size(1), input.device());
            },
            [&](Tensor query, Tensor keys, const auto &plan) {
                return attention_dsv4_sparse_cuda(
                    query, keys, plan.first, plan.second, stage.sinks, attention_meta,
                    1.0 / std::sqrt(static_cast<double>(config.head_dim)));
            },
            [&](Tensor attended) {
                return dsv4_rotate_rope_tail(attended.transpose(1, 2).contiguous(), positions,
                                             stage.rope, true)
                    .transpose(1, 2)
                    .contiguous();
            },
            [&](Tensor attended) { return stage.output_projection(*execution, attended); });
    }

    Tensor draft_block(const MtpTarget &target, Tensor anchor_ids, int requested) override {
        const auto physical_width =
            std::min<std::int64_t>(config.dspark_block_size, maximum_context - position);
        MFQ_RUNTIME_CHECK(
            batch == 1 && anchor_ids.dim() == 2 && anchor_ids.size(0) == batch &&
                anchor_ids.size(1) == 1 && requested > 0 && requested <= config.dspark_block_size &&
                requested <= physical_width && position > 0 && rings.size() == stages.size(),
            "invalid DeepSeek-V4.1 DSpark draft input");
        anchor_ids =
            anchor_ids.to(mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64).contiguous();
        auto draft_ids = anchor_ids;
        if (physical_width > 1) {
            draft_ids = mfq_tensor_backend::cat(
                            {anchor_ids, mfq_tensor_backend::full({batch, physical_width - 1},
                                                                  config.dspark_noise_token_id,
                                                                  anchor_ids.options())},
                            1)
                            .contiguous();
        }
        return mfq::models::deepseek_v41::dspark_draft(
            physical_width, requested, stages.size(),
            [&](int64_t) { return target.embed(draft_ids).to(mfq_tensor_backend::kFloat16); },
            [&](Tensor embedded) {
                return embedded.unsqueeze(2)
                    .expand({batch, physical_width, config.hc_mult, config.hidden})
                    .contiguous();
            },
            [&](Tensor embedded) {
                auto previous_pre = mfq_tensor_backend::zeros(
                    {batch, physical_width, config.hc_mult},
                    embedded.options().dtype(mfq_tensor_backend::kFloat32));
                previous_pre.narrow(2, 0, 1).fill_(1.0);
                return previous_pre;
            },
            [&](size_t index, Tensor hidden, Tensor &previous_pre, int64_t output_rows) {
                auto &stage = *stages[index];
                return mfq::models::deepseek_v41::mega_layer(
                    std::move(hidden), previous_pre,
                    [&](Tensor value, const Tensor &pre, int branch) {
                        // The last attention still sees the complete block.
                        // Its following FFN is independent for each token.
                        const auto rows = branch == 0 ? physical_width : output_rows;
                        return stage.collapse(
                            execution->profiler, value.narrow(1, 0, rows), pre.narrow(1, 0, rows),
                            branch == 0 ? stage.attention_mhc_function : stage.mlp_mhc_function,
                            branch == 0 ? stage.attention_mhc_scale : stage.mlp_mhc_scale,
                            branch == 0 ? stage.attention_mhc_base : stage.mlp_mhc_base,
                            branch == 0 ? stage.attention_norm : stage.mlp_norm,
                            branch == 0 ? "deepseek_v41.dspark.mhc.attention.collapse"
                                        : "deepseek_v41.dspark.mhc.mlp.collapse");
                    },
                    [&](Tensor value) { return attention(value, stage, rings[index]); },
                    [&](Tensor value, Tensor residual, const auto &mix, int branch) {
                        return stage.expand(execution->profiler, value,
                                            residual.narrow(1, 0, value.size(1)), mix,
                                            branch == 0 ? "deepseek_v41.dspark.mhc.attention.expand"
                                                        : "deepseek_v41.dspark.mhc.mlp.expand");
                    },
                    [&](Tensor value) {
                        const auto rows = value.size(1);
                        return stage.mlp
                            .forward(*execution, value.reshape({-1, config.hidden}),
                                     draft_ids.narrow(1, 0, rows))
                            .reshape({batch, rows, config.hidden});
                    });
            },
            [&](Tensor hidden, const Tensor &previous_pre) {
                return (previous_pre.to(mfq_tensor_backend::kFloat32).unsqueeze(-1) *
                        hidden.to(mfq_tensor_backend::kFloat32))
                    .sum(2)
                    .to(mfq_tensor_backend::kFloat16)
                    .contiguous();
            },
            [&](Tensor head_hidden) {
                return target.logits(weighted_rms(head_hidden, output_norm, config.rms_eps))
                    .to(mfq_tensor_backend::kFloat32)
                    .contiguous();
            },
            [](Tensor logits, int64_t count) { return logits.narrow(1, 0, count); });
    }

    Tensor draft_next(Tensor logits, Tensor previous) override {
        auto markov = quant_embedding_lookup(markov_embedding, previous)
                          .to(mfq_tensor_backend::kFloat16)
                          .contiguous();
        return logits + markov_output.forward(*execution, markov).to(mfq_tensor_backend::kFloat32);
    }
};

std::unique_ptr<::MtpModule> load_dspark_if_present(CudaExecutionContext &execution,
                                                    const mfq::ModelSource &source,
                                                    const CommonConfig &config) {
    auto predictor = DeepseekV41Dspark::load_if_present(execution, source, config);
    return predictor ? std::make_unique<DeepseekV41Dspark>(std::move(*predictor)) : nullptr;
}

void run_dspark_self_check() {
    // Seeding only the final window must address the same absolute RoPE and
    // ring slots as a complete target history.
    CudaExecutionContext execution;
    DeepseekV41Dspark predictor;
    predictor.execution = &execution;
    predictor.config.hidden = 32;
    predictor.config.head_dim = 512;
    predictor.config.sliding_window = 128;
    predictor.config.dspark_target_layer_ids = {2, 3};
    predictor.maximum_context = 1024;
    const auto options = mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
                            .dtype(mfq_tensor_backend::kFloat32);
    const auto projection = [&](int rows, int columns) {
        QuantLinear weight;
        weight.kind = QuantLinearKind::Dense;
        weight.dense = (mfq_tensor_backend::randn({rows, columns}, options) / std::sqrt(double(columns)))
                           .to(mfq_tensor_backend::kFloat16).contiguous();
        return weight;
    };
    predictor.main_projection = projection(32, 64);
    predictor.main_norm = mfq_tensor_backend::ones({32}, options);
    auto stage = std::make_unique<Block>();
    stage->key_value = projection(512, 32);
    stage->key_value_norm = mfq_tensor_backend::ones({512}, options);
    stage->rope = Dsv4RopeTable(1024, 10000., 0);
    predictor.stages.push_back(std::move(stage));
    auto history = mfq_tensor_backend::randn({1, 257, 64}, options).to(mfq_tensor_backend::kFloat16);
    predictor.reset();
    predictor.append_target_context(history, 0);
    auto expected_ring = predictor.rings.front().clone();
    predictor.reset();
    predictor.seed_target_context(history.narrow(1, 129, 128).contiguous(), 257);
    MFQ_RUNTIME_CHECK(predictor.cache_position() == 257 && predictor.rings.front().equal(expected_ring),
                      "DeepSeek-V4.1 DSpark bounded prompt seed changed context");
    auto saved=predictor.capture_session_state(257,history.narrow(1,256,1).contiguous());
    predictor.append_target_context(history.narrow(1, 0, 1).contiguous(), 257);
    MFQ_RUNTIME_CHECK(predictor.cache_position() == 258,
                      "DeepSeek-V4.1 DSpark append after bounded seed lost position");
    auto advanced=predictor.rings.front().clone();
    predictor.restore_session_state(saved);
    MFQ_RUNTIME_CHECK(predictor.cache_position()==257 && predictor.rings.front().equal(expected_ring),
                      "DeepSeek-V4.1 DSpark session restore lost its committed ring");
    predictor.append_target_context(history.narrow(1,0,1).contiguous(),257);
    MFQ_RUNTIME_CHECK(predictor.rings.front().equal(advanced) &&
                      saved.deepseek_v41_rings.front().equal(expected_ring),
                      "DeepSeek-V4.1 DSpark session changed continuation or mutated its snapshot");
    auto broken=saved;broken.deepseek_v41_rings.front()=broken.deepseek_v41_rings.front().narrow(1,0,127);
    bool rejected=false;try{predictor.restore_session_state(broken);}catch(const CudaSessionStateError&){rejected=true;}
    MFQ_RUNTIME_CHECK(rejected && predictor.cache_position()==258 && predictor.rings.front().equal(advanced),
                      "DeepSeek-V4.1 DSpark accepted an incompatible session ring");
    auto tail=history.narrow(1,129,128).contiguous();
    predictor.append_target_context(tail,400);
    auto skipped=predictor.rings.front().clone();
    predictor.reset();predictor.seed_target_context(tail,528);
    MFQ_RUNTIME_CHECK(predictor.cache_position()==528 && predictor.rings.front().equal(skipped),
                      "DeepSeek-V4.1 DSpark restored suffix tail changed absolute ring positions");
    rejected=false;
    try{predictor.append_target_context(history.narrow(1,0,1),600);}
    catch(const std::runtime_error&){rejected=true;}
    MFQ_RUNTIME_CHECK(rejected && predictor.cache_position()==528,
                      "DeepSeek-V4.1 DSpark skipped live history without a full replacement window");

    auto plan =
        dspark_full_attention_plan(1, 2, 33, mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA));
    auto indices = plan.first.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt32).contiguous();
    auto mask = plan.second.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32).contiguous();
    MFQ_RUNTIME_CHECK(indices.sizes() == mfq_tensor_backend::IntArrayRef({1, 2, 64}) &&
                          indices.index({0, 0, 0}).item<std::int32_t>() == 0 &&
                          indices.index({0, 0, 32}).item<std::int32_t>() == 32 &&
                          indices.index({0, 0, 33}).item<std::int32_t>() == 0 &&
                          mask.index({0, 1, 32}).item<float>() == 0.0f &&
                          std::isinf(mask.index({0, 1, 33}).item<float>()) &&
                          mask.index({0, 1, 33}).item<float>() < 0.0f,
                      "DeepSeek-V4.1 DSpark attention plan contract failed");

    CommonConfig capture_config;
    capture_config.hidden = 2;
    capture_config.hc_mult = 4;
    capture_config.dspark_target_layer_ids = {3, 1};
    SharedState capture;
    capture.begin_forward(true, 2);
    auto first = mfq_tensor_backend::arange(8, mfq_tensor_backend::TensorOptions()
                                                   .device(mfq_tensor_backend::kCUDA)
                                                   .dtype(mfq_tensor_backend::kFloat32))
                     .reshape({1, 1, 4, 2});
    auto second = first + 8.0;
    capture.capture_dspark_target(1, capture_config, second);
    capture.capture_dspark_target(3, capture_config, first);
    auto captured = capture.dspark_target_hidden()
                        .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
                        .contiguous();
    MFQ_RUNTIME_CHECK(captured.sizes() == mfq_tensor_backend::IntArrayRef({1, 1, 4}) &&
                          captured.index({0, 0, 0}).item<float>() == 3.0f &&
                          captured.index({0, 0, 2}).item<float>() == 11.0f,
                      "DeepSeek-V4.1 DSpark target ordering contract failed");

    Block checkpoint;
    checkpoint.config.sliding_window = 4;
    checkpoint.state.local_kv =
        mfq_tensor_backend::arange(8, mfq_tensor_backend::TensorOptions()
                                          .device(mfq_tensor_backend::kCUDA)
                                          .dtype(mfq_tensor_backend::kFloat16))
            .reshape({1, 4, 2});
    checkpoint.state.position = 4;
    checkpoint.begin_speculative(2);
    auto overwritten =
        mfq_tensor_backend::full({1, 2, 2}, 99.0, checkpoint.state.local_kv.options());
    checkpoint.state.local_kv.narrow(1, 0, 2).copy_(overwritten);
    checkpoint.state.position = 6;
    checkpoint.rollback_speculative(4);
    auto restored =
        checkpoint.state.local_kv.to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
            .contiguous();
    MFQ_RUNTIME_CHECK(checkpoint.state.position == 4 &&
                          restored.index({0, 0, 0}).item<float>() == 0.0f &&
                          restored.index({0, 1, 1}).item<float>() == 3.0f,
                      "DeepSeek-V4.1 DSpark ring rollback contract failed");
}

} // namespace mfq::cuda::deepseek_v41_runtime
