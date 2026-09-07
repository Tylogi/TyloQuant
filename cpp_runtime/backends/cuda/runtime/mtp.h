#pragma once

// Generation sees one predictor contract. Model-specific modules own their
// equations and cache layout; the server does not branch on architecture.
struct CudaMtpModule {
    virtual ~CudaMtpModule() = default;
    virtual void reset(int64_t batch = 1) = 0;
    virtual mfq_tensor_backend::Tensor forward(
        Model& target,
        mfq_tensor_backend::Tensor previous_hidden,
        mfq_tensor_backend::Tensor next_ids) = 0;
    virtual bool teacher_forced_prompt_prime() const noexcept = 0;
    virtual bool preserve_output_dtype() const noexcept = 0;

    uint64_t last_cycles = 0;
    uint64_t last_accepted = 0;
    uint64_t last_rejected = 0;
};
