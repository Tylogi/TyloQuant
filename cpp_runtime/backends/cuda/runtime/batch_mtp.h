#pragma once

#include "core/mtp.h"
#include "cuda_sampling.h"
#include "continuous_batch.h"

#include <functional>

namespace mfq::cuda {

// Phase-one capability: greedy chains, including penalties and grammar. Other
// sampling modes retain the exclusive single-request MTP implementation.
bool batch_mtp_enabled();
bool batch_mtp_supported(const MtpModule*, const MfqSamplingParams&);
// Recurrent-state copies batched verification may hold at once. The budget is
// fixed per runtime: MFQ_BATCH_MTP_CHECKPOINTS, else the predictor's draft depth.
int64_t batch_mtp_checkpoints(const MtpModule&);

struct BatchMtpRow {
    using Tensor = mfq_tensor_backend::Tensor;
    std::unique_ptr<MtpBatchState> predictor;
    std::unique_ptr<mfq::engine::mtp::GenerationPolicy<mfq::engine::mtp::DsparkDepthController>> depth;
    Tensor history_hidden;
    std::vector<int64_t> history_ids;
    int64_t first_token = 0;
    std::vector<int64_t> candidates;
    int accepted = 0, drafted = 0;
    std::function<void(int)> resolve;
    mfq::engine::mtp::GenerationStats stats;
    void discard();
    ~BatchMtpRow();
};

struct BatchMtpInput {
    mfq::engine::BatchRequest* request;
    BatchMtpRow* mtp;
    Sampler* sampler;
    mfq_tensor_backend::Tensor counts;
};

std::unique_ptr<BatchMtpRow> make_batch_mtp(MtpModule*, const MfqSamplingParams&, bool gradual_warmup = false);

struct BatchMtpVerification {
    mfq_tensor_backend::Tensor logits, raw;
    // Keep counts include the already-published pending token.
    std::vector<std::function<void(int)>> resolve;
};

using BatchMtpVerify = std::function<BatchMtpVerification(
    const mfq_tensor_backend::Tensor&, const mfq_tensor_backend::Tensor&, const std::vector<int>&)>;

void prime_batch_mtp(MtpModule&, const MtpTarget&, BatchMtpRow&,
    mfq_tensor_backend::Tensor raw, const mfq_tensor_backend::Tensor& ids, int64_t offset, int64_t prompt);
// All eligible requests share one target window. Draft work is batched by
// history width and then by active chain position; parked rows draft nothing.
// A non-negative `draft_tokens` bounds the drafts of the whole window, so its
// verification state stays fixed as rows are added.
std::vector<int64_t> run_batch_mtp(MtpModule&, const MtpTarget&, const std::vector<BatchMtpInput>&,
    int64_t max_context, const BatchMtpVerify&, int64_t draft_tokens = -1);
mfq::engine::TokenOutput publish_batch_mtp(BatchMtpInput, mfq::engine::InferenceOutput&);
void accept_batch_mtp_tokens(BatchMtpInput, const mfq::engine::TokenOutput&);

} // namespace mfq::cuda
