#pragma once

#include "continuous_batch.h"
#include "cuda_execution.h"
#include "cuda_runtime_config.h"
#include "cuda_sampling.h"
#include "runtime/batch_sampling.h"
#include "runtime/batch_mtp.h"
#include "runtime/kv_admission.h"
#include "models/qwen35/batch_state.h"
#include "models/qwen4_exp/ops.h"
#include <type_traits>

namespace mfq::cuda {

bool qwen_continuous_batch_cuda_graph_enabled(const Qwen35CausalLm &,
                                              const CudaContinuousBatchConfig &);
class BatchDecodeGraphCache;

struct QwenBatchRequest : mfq::engine::BatchRequest {
    using BatchRequest::BatchRequest;
    std::optional<Sampler> sampler;
    mfq_tensor_backend::Tensor counts;
    mfq_tensor_backend::Tensor prefill_ids;
    std::vector<std::unique_ptr<internal::PrefillCudaTimer>> prefill_timers;
    qwen35::QwenBatchRequestState cache;
    std::unique_ptr<BatchMtpRow> mtp;
    int64_t decode_row = -1;
};

// Native operations consumed directly by engine::ContinuousBatch.
class QwenBatchOperations {
  public:
    using Tensor = mfq_tensor_backend::Tensor;
    using Request = QwenBatchRequest;
    using State = mfq::engine::BatchState<Request>;
    struct Sample {
        Tensor next;
        int64_t token = 0;
        MfqPrefillTiming timing;
    };
    struct Decoded {
        Tensor logits, greedy_tokens, sampled_next, sampled_tokens;
    };

    QwenBatchOperations(Qwen35CausalLm &, CudaExecutionContext &, CudaContinuousBatchConfig,
                        MtpModule* = nullptr, BatchMemoryProvider = {});
    ~QwenBatchOperations();
    QwenBatchOperations(const QwenBatchOperations &) = delete;
    QwenBatchOperations &operator=(const QwenBatchOperations &) = delete;
    int64_t vocab_size() const;
    int64_t max_context() const;
    mfq::engine::Metrics metrics() const;
    mfq::engine::KvAdmissionConfig admission_config() const { return config_.admission; }
    mfq::engine::KvAdmissionBytes admission_available() const;
    mfq::engine::KvAdmissionBytes admission_cost(const std::shared_ptr<Request>&, int64_t expected) const;
    void reclaim_prefill_memory(const std::shared_ptr<Request>&);
    void suspend_decode();
    void release_idle_buffers();
    mfq::engine::PrefillChunk prefill_chunk(const std::shared_ptr<Request>&, mfq::engine::PrefillChunk);
    Sample prefill(const std::shared_ptr<Request> &, mfq::engine::PrefillChunk);
    void activate(const std::shared_ptr<Request> &, const Sample &);
    void resume_decode(int64_t cache_position);
    void discard_prefill(const std::shared_ptr<Request> &);
    void retire(const std::vector<std::shared_ptr<Request>> &, int64_t cache_position);
    Decoded decode(State &);
    std::optional<mfq::engine::Generation> generate_decode(const std::shared_ptr<Request>&,
        mfq::engine::InferenceOutput&, std::function<void()>, std::function<bool()>, std::function<void()>);
    int64_t cache_length(const std::shared_ptr<Request>& request) const;
    Sample sample(const std::shared_ptr<Request> &, const Decoded &);
    void accept(const std::shared_ptr<Request> &, const Sample &);
    void recover(State &);

  private:
    Decoded decode_mtp(State&);
    void initialize_sampling(Request &, const Tensor &prompt_ids);
    void invalidate_decode_graph();

    Qwen35CausalLm &model_;
    qwen35::Qwen35BatchStateAdapter state_adapter_;
    CudaExecutionContext &execution_;
    const CudaContinuousBatchConfig config_;
    BatchMemoryProvider memory_;
    int32_t max_sequences_ = 0;
    std::vector<std::weak_ptr<Request>> requests_;
    bool admission_storage_ready_ = false;
    bool moe_enabled_ = false;
    bool cached_moe_enabled_ = false;
    int64_t decode_batches_{0};
    int64_t decode_tokens_{0};
    int64_t batched_greedy_batches_{0};
    int64_t packed_metadata_batches_{0};
    BatchSampling sampling_;
    std::unique_ptr<BatchDecodeGraphCache> decode_graphs_;
    MtpModule* predictor_ = nullptr;
    // Rows whose pre-window recurrent state batched verification may keep.
    int64_t mtp_checkpoints_ = 0;
    int64_t mtp_batches_ = 0, mtp_drafted_ = 0, mtp_accepted_ = 0;
};

struct FlashNextBatchRequest : mfq::engine::BatchRequest {
    using BatchRequest::BatchRequest;
    std::optional<Sampler> sampler;
    mfq_tensor_backend::Tensor counts, prefill_ids;
    std::vector<std::unique_ptr<internal::PrefillCudaTimer>> prefill_timers;
    std::unique_ptr<Qwen4BatchRequestState> cache;
    std::unique_ptr<BatchMtpRow> mtp;
    int64_t decode_row = -1;
    double packed_prefill_ms = 0;
};
class FlashNextBatchOperations {
  public:
    using Tensor = mfq_tensor_backend::Tensor;
    using Request = FlashNextBatchRequest;
    using State = mfq::engine::BatchState<Request>;
    using Sample = QwenBatchOperations::Sample;
    struct Decoded { Tensor logits, greedy_tokens, sampled_next, sampled_tokens; };
    FlashNextBatchOperations(Qwen4CausalLm&, CudaExecutionContext&, CudaContinuousBatchConfig,
                             MtpModule* = nullptr, BatchMemoryProvider = {});
    ~FlashNextBatchOperations();
    int64_t vocab_size() const;
    int64_t max_context() const;
    mfq::engine::Metrics metrics() const;
    mfq::engine::KvAdmissionConfig admission_config() const { return config_.admission; }
    mfq::engine::KvAdmissionBytes admission_available() const;
    mfq::engine::KvAdmissionBytes admission_cost(const std::shared_ptr<Request>&, int64_t expected) const;
    void reclaim_prefill_memory(const std::shared_ptr<Request>&);
    void suspend_decode();
    void release_idle_buffers();
    mfq::engine::PrefillChunk prefill_chunk(const std::shared_ptr<Request>&, mfq::engine::PrefillChunk);
    Sample prefill(const std::shared_ptr<Request>&, mfq::engine::PrefillChunk);
    std::size_t prefill_batch_limit() const;
    int64_t prefill_batch_token_budget() const;
    std::vector<mfq::engine::PrefillBatchResult<Sample>> prefill_batch(
        const std::vector<std::pair<std::shared_ptr<Request>,mfq::engine::PrefillChunk>>&);
    void activate(const std::shared_ptr<Request>&, const Sample&);
    void resume_decode(int64_t);
    void discard_prefill(const std::shared_ptr<Request>&);
    void retire(const std::vector<std::shared_ptr<Request>>&, int64_t);
    Decoded decode(State&);
    std::optional<mfq::engine::Generation> generate_decode(const std::shared_ptr<Request>&,
        mfq::engine::InferenceOutput&, std::function<void()>, std::function<bool()>, std::function<void()>);
    int64_t cache_length(const std::shared_ptr<Request>& request) const;
    Sample sample(const std::shared_ptr<Request>&, const Decoded&);
    void accept(const std::shared_ptr<Request>&, const Sample&);
    void recover(State&);
    bool decode_has_capacity(const State&) const;
    void retract(const std::shared_ptr<Request>&);
    void replay_prefill(const std::shared_ptr<Request>&, mfq::engine::PrefillChunk);
    static constexpr bool isolate_prefill_failures = true;
    static constexpr bool isolate_context_growth_failures = true;
  private:
    Decoded decode_mtp(State&);
    std::size_t managed_kv_bytes() const;
    std::size_t admission_base_kv_bytes_ = 0;
    void initialize_prefill(const std::shared_ptr<Request>&,mfq::engine::PrefillChunk);
    Sample finish_prefill(const std::shared_ptr<Request>&,Tensor next);
    Qwen4CausalLm& model_;
    CudaExecutionContext& execution_;
    CudaContinuousBatchConfig config_;
    BatchSampling sampling_;
    BatchMemoryProvider memory_;
    std::vector<std::weak_ptr<Request>> requests_;
    std::unique_ptr<Qwen4BatchDecodeGraph> graph_;
    int64_t decode_batches_ = 0, decode_tokens_ = 0, greedy_batches_ = 0;
    int64_t packed_prefill_batches_ = 0, packed_prefill_requests_ = 0, packed_prefill_tokens_ = 0;
    MtpModule* predictor_ = nullptr;
    // Fixed verification checkpoints shared by all batched MTP rows.
    std::shared_ptr<Qwen4BatchCheckpointPool> mtp_checkpoints_;
    int64_t mtp_batches_ = 0, mtp_drafted_ = 0, mtp_accepted_ = 0;
};
template<class Model> using CudaBatchOperations = std::conditional_t<
    std::is_same_v<Model,Qwen4CausalLm>, FlashNextBatchOperations, QwenBatchOperations>;
} // namespace mfq::cuda
