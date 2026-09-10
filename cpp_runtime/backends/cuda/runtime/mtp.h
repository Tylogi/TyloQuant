#pragma once

struct CudaMtpStep {
    mfq_tensor_backend::Tensor sample_hidden;
    mfq_tensor_backend::Tensor chain_hidden;
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

    mfq::cuda::mtp::GenerationStats last_stats;
    uint64_t last_cycles = 0;
    uint64_t last_accepted = 0;
    uint64_t last_rejected = 0;
};
