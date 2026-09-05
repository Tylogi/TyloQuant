#pragma once

#include "deepseek_v4_model.h"
#include "mlx_deepseek_v4_moe.h"
#include "mlx_hf_tensor.h"
#include "mlx_tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// Committed-only DSpark context.  The released drafter never commits draft
// K/V: every stage owns one physical sliding-window ring containing target
// model hidden states that survived verification.
class MlxDeepseekV4DSparkState {
public:
    static MlxDeepseekV4DSparkState allocate(
        int stages,
        int batch,
        int window,
        int head_dim,
        mlx::core::Dtype dtype = mlx::core::float16);

    int position() const noexcept { return position_; }
    int batch() const noexcept;
    int window() const noexcept;
    std::size_t stages() const noexcept { return rings_.size(); }
    const mlx::core::array& ring(std::size_t stage) const;

    MlxDeepseekV4DSparkState snapshot() const;
    void restore_snapshot(MlxDeepseekV4DSparkState snapshot);

private:
    MlxDeepseekV4DSparkState(
        std::vector<mlx::core::array> rings,
        int position);

    std::vector<mlx::core::array> rings_;
    int position_ = 0;

    friend class MlxDeepseekV4DSpark;
};

struct MlxDeepseekV4DSparkAttentionComponents {
    MlxLinear q_a;
    MlxLinear q_b;
    MlxLinear kv;
    MlxLinear wo_a;
    MlxLinear wo_b;
    mlx::core::array q_norm;
    mlx::core::array kv_norm;
    mlx::core::array sinks;
};

struct MlxDeepseekV4DSparkStageComponents {
    MlxDeepseekV4DSparkAttentionComponents attention;
    MlxDeepseekV4Moe moe;
    mlx::core::array attention_norm;
    mlx::core::array ffn_norm;
    MlxLinear hc_attention_fn;
    mlx::core::array hc_attention_base;
    mlx::core::array hc_attention_scale;
    MlxLinear hc_ffn_fn;
    mlx::core::array hc_ffn_base;
    mlx::core::array hc_ffn_scale;
};

struct MlxDeepseekV4DSparkHeadComponents {
    mlx::core::array norm;
    MlxLinear hc_head_fn;
    mlx::core::array hc_head_base;
    mlx::core::array hc_head_scale;
    MlxEmbedding markov_embedding;
    MlxLinear markov_output;
    MlxLinear confidence;
};

struct MlxDeepseekV4DSparkDraft {
    // [batch,width] greedy block tokens (the anchor is not repeated).
    mlx::core::array tokens;
    // [batch,width,vocab] after the sequential Markov bias.
    mlx::core::array logits;
    // [batch,width] checkpoint confidence scores. Verification remains the
    // source of correctness; callers may use these only as a scheduling hint.
    mlx::core::array confidence;
};

// Native block-parallel DSpark operator.  It is deliberately separate from
// the dense Qwen MTP module: each stage is a complete DeepSeek HC + attention
// + routed/shared MoE block, and all stages share the target embedding/head.
class MlxDeepseekV4DSpark {
public:
    static std::optional<MlxDeepseekV4DSpark> load_if_present(
        const MfqContainer& model,
        const DeepseekV4Config& config,
        const MlxEmbedding& embedding,
        const MlxLinear& output,
        int max_context,
        std::shared_ptr<MlxNintMoeOffloadCache> expert_offload =
            nullptr,
        std::size_t expert_layer_base = 0);

    static MlxDeepseekV4DSpark load_hf(
        const MlxHfTensorStore& model,
        const DeepseekV4Config& config,
        const MlxEmbedding& embedding,
        const MlxLinear& output,
        int max_context,
        std::shared_ptr<MlxDeepseekV4SsdExpertCache> expert_cache,
        std::size_t expert_layer_base);

    MlxDeepseekV4DSpark(
        DeepseekV4Config config,
        MlxEmbedding embedding,
        MlxLinear output,
        MlxLinear main_projection,
        mlx::core::array main_norm,
        std::vector<MlxDeepseekV4DSparkStageComponents> stages,
        MlxDeepseekV4DSparkHeadComponents head,
        int max_context,
        std::pair<mlx::core::array, mlx::core::array> rope);

    MlxDeepseekV4DSparkState make_state(
        int batch = 1,
        mlx::core::Dtype dtype = mlx::core::float16) const;

    // Append concatenated target taps [B,T,target_layers*hidden] at the exact
    // next committed position.  This performs main_proj/main_norm once and
    // writes every stage's main K/V ring.
    void append_context(
        const mlx::core::array& main_hidden,
        MlxDeepseekV4DSparkState& state,
        int start_position) const;

    // Produce 1..dspark_block_size tokens from anchor_ids [B,1].  Draft block
    // attention is non-causal by checkpoint design; Markov bias is then
    // applied left-to-right before each greedy decision.
    MlxDeepseekV4DSparkDraft draft_greedy(
        const mlx::core::array& anchor_ids,
        MlxDeepseekV4DSparkState& state,
        int width = 0) const;

    int block_size() const noexcept;
    std::size_t stage_count() const noexcept;
    int context_position(
        const MlxDeepseekV4DSparkState& state) const noexcept {
        return state.position();
    }

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace mfq::metal
