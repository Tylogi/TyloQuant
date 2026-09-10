#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>

struct CudaMtpStep {
    mfq_tensor_backend::Tensor sample_hidden;
    mfq_tensor_backend::Tensor chain_hidden;
};

using CudaMtpTokenSelector = std::function<std::int32_t(
    mfq_tensor_backend::Tensor)>;

struct CudaMtpBlockDraft {
    mfq_tensor_backend::Tensor tokens;
    mfq_tensor_backend::Tensor logits;
    mfq_tensor_backend::Tensor confidence;
};

// Generation sees one predictor contract. Model-specific modules own their
// equations and cache layout; the server does not branch on architecture.
struct CudaMtpModule {
    virtual ~CudaMtpModule() = default;
    virtual void reset(int64_t batch = 1) = 0;
    virtual mfq_tensor_backend::Tensor forward(
        Model& target,
        mfq_tensor_backend::Tensor previous_hidden,
        mfq_tensor_backend::Tensor next_ids) = 0;
    virtual CudaMtpStep step(
        Model& target,
        mfq_tensor_backend::Tensor previous_hidden,
        mfq_tensor_backend::Tensor next_ids) {
        auto hidden = forward(
            target, std::move(previous_hidden), std::move(next_ids));
        return {hidden, hidden};
    }
    virtual int64_t cache_position() const noexcept = 0;
    virtual void trim_cache_to(int64_t position) = 0;
    virtual bool teacher_forced_prompt_prime() const noexcept = 0;
    virtual bool target_bootstrap_decode() const noexcept = 0;
    virtual bool preserve_output_dtype() const noexcept = 0;
    virtual int maximum_draft_depth() const noexcept {
        return mfq::cuda::mtp::kMaximumDraftDepth;
    }
    virtual bool blockwise_drafting() const noexcept { return false; }
    virtual bool split_target_verification() const noexcept { return false; }
    virtual void append_target_context(
        mfq_tensor_backend::Tensor,
        std::int64_t) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no target-context adapter");
    }
    virtual CudaMtpBlockDraft draft_block(
        Model&,
        mfq_tensor_backend::Tensor,
        const CudaMtpTokenSelector&,
        int) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no block-draft adapter");
    }

    mfq::cuda::mtp::GenerationStats last_stats;
    uint64_t last_cycles = 0;
    uint64_t last_accepted = 0;
    uint64_t last_rejected = 0;
};
