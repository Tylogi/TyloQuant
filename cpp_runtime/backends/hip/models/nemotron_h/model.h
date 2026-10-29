#pragma once

#include "core/block.h"
#include "core/ffn.h"
#include "core/full_block.h"
#include "quant_linear.h"
#include "core/recurrent_mixer_state.h"

#include <array>
#include <cstdint>
#include <vector>

struct CudaExecutionContext;

namespace mfq::cuda::nemotron_h {

// One pre-normed Mamba-2 mixer plus its residual. The depthwise convolution
// and the selective scan each own float32 state; both are checkpointed for a
// speculative verification window and replayed on rollback.
struct Mamba2MixerBlock final : Block {
    int layer = -1;
    int64_t mamba_heads = 0;
    int64_t mamba_head_dim = 0;
    int64_t ssm_state_size = 0;
    int64_t mamba_groups = 0;
    int64_t conv_kernel = 0;
    double rms_norm_eps = 1e-6;
    double time_step_limit_lo = 0.0;
    double time_step_limit_hi = 0.0;
    mfq_tensor_backend::Tensor mixer_norm;
    QuantLinear in_proj;
    mfq_tensor_backend::Tensor conv_weight;
    mfq_tensor_backend::Tensor conv_bias;
    mfq_tensor_backend::Tensor a_log;
    mfq_tensor_backend::Tensor d;
    mfq_tensor_backend::Tensor dt_bias;
    mfq_tensor_backend::Tensor scan_norm;
    QuantLinear out_proj;
    mfq::cuda::RecurrentMixerState mixer;
    // Projections of the pending verification window, replayed on rollback.
    mfq_tensor_backend::Tensor speculative_xbc;
    mfq_tensor_backend::Tensor speculative_dt;
    int64_t speculative_start = -1;
    int64_t speculative_confirmed = 0;
    int64_t speculative_tokens = 0;

    bool supports_speculation() const noexcept override { return true; }
    std::vector<mfq_tensor_backend::Tensor*> graph_warmup_state() override {
        MFQ_RUNTIME_CHECK(!mixer.speculative_pending && mixer.defined(),
                          "decode warmup requires confirmed mamba state");
        return {&mixer.convolution, &mixer.recurrent};
    }
    void reset(int64_t batch) override {
        clear_speculative();
        mixer.reset(batch);
    }
    mfq_tensor_backend::Tensor forward(
        CudaExecutionContext& execution, mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor positions, int64_t cache_position,
        const MfqOptional<mfq_tensor_backend::Tensor>& sequence_lengths, const RopeCache& rope,
        const MfqOptional<mfq_tensor_backend::Tensor>& cache_positions = mfq_nullopt,
        const MfqOptional<mfq_tensor_backend::Tensor>& attention_mask = mfq_nullopt) override;
    mfq_tensor_backend::Tensor forward_context(CudaExecutionContext& execution,
                                               mfq_tensor_backend::Tensor x,
                                               const Block::Context& context,
                                               const RopeCache& rope) override;
    void commit_speculative() noexcept override { clear_speculative(); }
    void rollback_speculative(int64_t keep_position) override;

    void clear_speculative() noexcept;
    // Re-run convolution and scan over the retained rows of the window.
    void replay_recurrent(int64_t retained_tokens);
};

// One pre-normed latent MoE (or dense relu-squared MLP) plus its residual.
// The mixer is stateless, so a speculative window leaves nothing behind.
struct MoeMixerBlock final : Block {
    int layer = -1;
    double rms_norm_eps = 1e-6;
    mfq_tensor_backend::Tensor ffn_norm;
    FFN ffn;

    bool supports_speculation() const noexcept override { return true; }
    void reset(int64_t) override {}
    mfq_tensor_backend::Tensor forward(
        CudaExecutionContext& execution, mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor positions, int64_t cache_position,
        const MfqOptional<mfq_tensor_backend::Tensor>& sequence_lengths, const RopeCache& rope,
        const MfqOptional<mfq_tensor_backend::Tensor>& cache_positions = mfq_nullopt,
        const MfqOptional<mfq_tensor_backend::Tensor>& attention_mask = mfq_nullopt) override;
};

// NoPE attention mixer: the shared full-attention block without its FFN tail.
struct AttentionMixerBlock final : FullBlock {
    mfq_tensor_backend::Tensor forward_ffn(CudaExecutionContext& execution,
                                           mfq_tensor_backend::Tensor residual,
                                           mfq_tensor_backend::Tensor attention_output,
                                           int64_t batch, int64_t tokens,
                                           int64_t hidden) override;
};

} // namespace mfq::cuda::nemotron_h
