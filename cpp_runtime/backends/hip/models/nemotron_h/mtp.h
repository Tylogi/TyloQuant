#pragma once

#include "storage/weight_loader.h"

#include "core/mtp.h"
#include "core/full_block.h"
#include "models/nemotron_h/causal_lm.h"
#include "ops.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

namespace mfq::cuda::nemotron_h {

// Nemotron-H's predictor is one multi-token-prediction head whose sub-blocks
// are listed in mtp_layers_block_type; both blocks run inside the single stage
// of the shared projected_predictor graph. The attention sub-block owns the
// predictor history; the MoE sub-block is stateless.
// The semantics follow llama.cpp (src/models/nemotron-h-moe.cpp), vLLM and
// SGLang rather than DeepSeek-V3: the backbone hands over its post-output_norm
// hidden (see NemotronHModel::adapter_raw_hidden) and each further draft step
// reuses this module's final_layernorm output. The final norm stays RMSNorm as
// in vLLM/SGLang (llama.cpp uses LayerNorm; the real model will A/B the two).
struct NemotronHMtp final : MtpModule {
    using Tensor = mfq_tensor_backend::Tensor;

    mfq::cuda::nemotron_h::Config config;
    QuantLinear fusion;
    Tensor embedding_norm, hidden_norm, output_norm;
    std::vector<std::unique_ptr<Block>> blocks;
    std::vector<int64_t> lengths;
    CudaExecutionContext *execution = nullptr;
    int64_t batch = 0;

    // The predictor history shares the backbone context limit, which the
    // loader may have overridden with --ctx-size.
    int64_t predictor_capacity() const noexcept { return config.max_position_embeddings; }

    static std::optional<NemotronHMtp> load_if_present(CudaExecutionContext &execution,
                                                       const mfq::ModelSource &file,
                                                       const mfq::models::nemotron_h::Config &main) {
        const bool fusion_present = has_tensor(file, "predictor.fusion.weight");
        if (!fusion_present) {
            MFQ_RUNTIME_CHECK(!has_tensor_prefix(file, "predictor."),
                              "Nemotron-H source contains an incomplete MTP head");
            return std::nullopt;
        }
        MFQ_RUNTIME_CHECK(main.num_nextn_predict_layers == 1 && !main.mtp_mixers.empty(),
                          "Nemotron-H MTP requires one declared predictor stage");
        NemotronHMtp result;
        result.config = main;
        result.execution = &execution;
        result.embedding_norm = load_dense_gpu(execution, file, "predictor.embedding_norm.weight");
        result.hidden_norm = load_dense_gpu(execution, file, "predictor.hidden_norm.weight");
        result.output_norm = load_dense_gpu(execution, file, "predictor.output_norm.weight");
        result.fusion = load_quant_linear(execution, file, "predictor.fusion.weight");
        MFQ_RUNTIME_CHECK(result.fusion.neuron_len() == 2 * main.hidden_size &&
                              result.fusion.out() == main.hidden_size &&
                              result.embedding_norm.numel() == main.hidden_size &&
                              result.hidden_norm.numel() == main.hidden_size &&
                              result.output_norm.numel() == main.hidden_size,
                          "Nemotron-H MTP fusion/norm dimensions disagree with the backbone");
        for (std::size_t index = 0; index < main.mtp_mixers.size(); ++index) {
            const auto type =
                std::string(mfq::models::nemotron_h::Config::mixer_name(main.mtp_mixers[index]));
            auto block = mfq::cuda::nemotron_h::load_block(execution, file, main,
                                                          static_cast<int>(index), type,
                                                          "predictor");
            block->cuda_device = execution.layer_placement.primary_device();
            result.blocks.push_back(std::move(block));
        }
        result.lengths.assign(1, 0);
        return result;
    }

    void reset(int64_t next_batch = 1) override {
        MFQ_RUNTIME_CHECK(next_batch > 0, "Nemotron-H MTP batch must be positive");
        for (auto &block : blocks)
            block->reset(next_batch);
        std::fill(lengths.begin(), lengths.end(), 0);
        batch = next_batch;
    }

    std::pair<Tensor, Tensor> evaluate(const MtpTarget &target, const Tensor &hidden,
                                       const Tensor &ids, int64_t depth = 0,
                                       const Tensor &supplied_positions = {}) {
        struct Ops {
            using Tensor = mfq_tensor_backend::Tensor;
            NemotronHMtp &model;
            const MtpTarget &target;
            static int rank(const Tensor &x) { return x.dim(); }
            static int64_t size(const Tensor &x, int axis) { return x.size(axis); }
            static int64_t elements(const Tensor &x) { return x.numel(); }
            static bool defined(const Tensor &x) { return x.defined(); }
            Tensor embed(Tensor ids, const Tensor &like) {
                return target.embed(std::move(ids)).to(like.scalar_type());
            }
            Tensor normalize(const Tensor &x, mfq::models::nemotron_h::PredictorNorm role) {
                using Norm = mfq::models::nemotron_h::PredictorNorm;
                const auto &weight = role == Norm::embedding ? model.embedding_norm
                                     : role == Norm::hidden  ? model.hidden_norm
                                                             : model.output_norm;
                return qwen_rms_norm(x.reshape({-1, model.config.hidden_size})
                                         .to(mfq_tensor_backend::kFloat32),
                                     weight, model.config.rms_norm_eps, 0.0)
                    .reshape(x.sizes());
            }
            Tensor positions(Tensor explicit_positions, int64_t start, int64_t tokens) {
                if (explicit_positions.defined())
                    return explicit_positions.to(mfq_tensor_backend::kInt64).contiguous();
                // NoPE: positions only address the predictor's KV history.
                return mfq_tensor_backend::arange(start, start + tokens,
                                                  mfq_tensor_backend::TensorOptions()
                                                      .device(mfq_tensor_backend::kCUDA)
                                                      .dtype(mfq_tensor_backend::kInt64));
            }
            Tensor fuse(Tensor e, Tensor h) {
                return model.fusion.forward(*model.execution, mfq_tensor_backend::cat({e, h}, -1));
            }
            static Tensor lengths(int64_t batch, int64_t end, const Tensor &positions) {
                return mfq_tensor_backend::full({batch}, end, positions.options());
            }
            // One predictor stage: every declared sub-block adds its own
            // pre-normed mixer branch to the same residual stream.
            Tensor mixer_blocks(Tensor x, int64_t stage, const Tensor &pos) {
                const int64_t tokens = x.size(1);
                std::optional<Tensor> sequence_lengths;
                if (tokens == 1 && model.lengths[stage] > 0)
                    sequence_lengths = lengths(x.size(0), model.lengths[stage] + 1, pos);
                for (auto &block : model.blocks)
                    x = block->forward(*model.execution, std::move(x), pos, model.lengths[stage],
                                       sequence_lengths, *target.rope);
                return x;
            }
        } ops{*this, target};
        return mfq::models::nemotron_h::mtp_predictor(ops, hidden, ids, supplied_positions, depth,
                                                     true);
    }

    Tensor forward(const MtpTarget &target, Tensor hidden, Tensor ids) override {
        return evaluate(target, hidden, ids).first;
    }

    // Depth chaining reuses the MTP's own final_layernorm output as the next
    // step's hidden state (llama.cpp/vLLM/SGLang), so the chain value is the
    // same normalized tensor the lm_head consumes. The backbone's output_norm
    // is not applied again.
    MtpStep step(const MtpTarget &target, Tensor hidden, Tensor ids) override {
        auto result = evaluate(target, hidden, ids);
        return {result.first, result.first};
    }

    MtpStep step_positioned(const MtpTarget &target, Tensor hidden, Tensor ids,
                            Tensor positions) override {
        auto result = evaluate(target, hidden, ids, 0, positions);
        return {result.first, result.first};
    }

    int64_t cache_position() const noexcept override {
        return lengths.empty() ? 0 : lengths.front();
    }

    void trim_cache_to(int64_t position) override {
        MFQ_RUNTIME_CHECK(position >= 0 && position <= cache_position(),
                          "Nemotron-H MTP cache trim position is invalid");
        // The attention sub-block's KV storage is position addressed. Lowering
        // the logical boundary overwrites the rejected draft suffix without
        // copying history-sized tensors.
        lengths.front() = position;
    }

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return true; }
    bool retains_partial_target_prefix() const noexcept override { return true; }

    bool supports_session_state() const noexcept override { return true; }

    MtpSessionState
    capture_session_state(int64_t target_cache_position,
                          const mfq_tensor_backend::Tensor &last_target_hidden) const override {
        const int64_t predictor_position = target_cache_position - 1;
        if (predictor_position <= 0 || predictor_position > cache_position() ||
            !last_target_hidden.defined() || last_target_hidden.dim() != 3 ||
            last_target_hidden.size(0) != 1 || last_target_hidden.size(1) != 1 ||
            last_target_hidden.size(2) != config.hidden_size) {
            throw std::runtime_error("Nemotron-H MTP session boundary is unavailable");
        }
        MtpSessionState state;
        state.cache_pos = predictor_position;
        for (const auto &block : blocks) {
            const auto *full = dynamic_cast<const FullBlock *>(block.get());
            if (full == nullptr)
                continue;
            MfqCudaGuard guard(block->cuda_device);
            state.blocks.push_back(
                capture_full_attention_session_state(*full, predictor_position, state.bytes));
        }
        state.last_target_hidden = last_target_hidden.clone();
        state.bytes += session_tensor_bytes(state.last_target_hidden);
        return state;
    }

    void restore_session_state(const MtpSessionState &state) override {
        std::vector<FullBlock *> saved;
        for (auto &block : blocks)
            if (auto *full = dynamic_cast<FullBlock *>(block.get()))
                saved.push_back(full);
        if (state.cache_pos <= 0 || state.blocks.size() != saved.size() ||
            !state.last_target_hidden.defined() || state.last_target_hidden.dim() != 3 ||
            state.last_target_hidden.size(0) != 1 || state.last_target_hidden.size(1) != 1 ||
            state.last_target_hidden.size(2) != config.hidden_size) {
            throw std::runtime_error("Nemotron-H MTP session state is incompatible");
        }
        for (std::size_t index = 0; index < saved.size(); ++index) {
            MfqCudaGuard guard(saved[index]->cuda_device);
            restore_full_attention_session_state(*saved[index], state.blocks[index]);
        }
        lengths.front() = state.cache_pos;
    }
};

} // namespace mfq::cuda::nemotron_h
