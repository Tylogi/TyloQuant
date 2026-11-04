#include "mfq_cuda_sampling_ops.h"
#include "cuda_batching.h"

#include "cuda_execution.h"
#include "core/decode_graph.h"
#include "runtime/batch_decode_graph.h"
#include "core/full_block.h"
#include "models/qwen35/ops.h"
#include "storage/moe_expert_cache.h"
#include "runtime/execution_options.h"
#include "prefill_budget.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda {

using internal::PrefillCudaTimer;
using Tensor = mfq_tensor_backend::Tensor;
using qwen35::Qwen35BatchStateAdapter;
using qwen35::QwenBatchRequestState;

template<class Model>
static mfq::engine::PrefillChunk budget_batch_prefill(Model& model,
        CudaExecutionContext& execution,mfq::engine::PrefillChunk chunk,std::size_t growth=0) {
    const auto context=default_context(execution.layer_placement.primary_device());
    const auto transfer=moe_prefill_workspace_budget(execution.moe_expert_cache,false).transfer_bytes;
    const auto fixed=mfq::engine::prefill_budget_add((std::size_t(1)<<30)+growth,transfer);
    const auto available=context->available_memory()+context->workspace_available();
    const auto row_bytes=std::size_t(model.hidden_size())*model.hc_mult()*sizeof(float)*8;
    const auto selected=mfq::engine::select_prefill_chunk(chunk.count,chunk.count,available,fixed,row_bytes);
    if(!selected)throw mfq::engine::InferenceInputError(
        mfq::engine::InferenceInputErrorCode::ContextLength,
        "insufficient CUDA workspace for batched prefill; reduce concurrent requests");
    chunk.count=selected;
    return chunk;
}

mfq::engine::PrefillChunk QwenBatchOperations::prefill_chunk(
        const std::shared_ptr<Request>& request,mfq::engine::PrefillChunk chunk) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    reclaim_prefill_memory(request);
    return budget_batch_prefill(model_,execution_,chunk);
}

mfq::engine::PrefillChunk FlashNextBatchOperations::prefill_chunk(
        const std::shared_ptr<Request>& request,mfq::engine::PrefillChunk chunk) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    reclaim_prefill_memory(request);
    initialize_prefill(request,chunk);
    return budget_batch_prefill(model_,execution_,chunk,request->cache->prefill_growth_bytes(chunk.count));
}

bool qwen_continuous_batch_cuda_graph_enabled(const Qwen35CausalLm &model,
                                              const CudaContinuousBatchConfig &config) {
    const auto &execution = *model.execution;
    return config.cuda_graph && !Qwen35BatchStateAdapter::has_cached_moe(model) &&
           mfq_cuda_graph_capture_supported() && model_parallel_cuda_graph_enabled(execution);
}

QwenBatchOperations::QwenBatchOperations(Qwen35CausalLm &model, CudaExecutionContext &execution,
                                         CudaContinuousBatchConfig config, MtpModule* predictor,
                                         BatchMemoryProvider memory)
    : model_(model), state_adapter_(model, config.max_sequences, config.paged_kv),
      execution_(execution), config_(std::move(config)), memory_(std::move(memory)),
      max_sequences_(static_cast<int32_t>(config_.max_sequences)),
      moe_enabled_(state_adapter_.has_moe()),
      cached_moe_enabled_(state_adapter_.has_cached_moe()),
      decode_graphs_(std::make_unique<BatchDecodeGraphCache>()), predictor_(predictor) {
    MFQ_RUNTIME_CHECK(config_.max_sequences > 0 && config_.max_sequences <= config_.sequence_limit &&
                      config_.sequence_limit <= mfq::engine::ContinuousBatchConfig::validated_max_sequences,
                      "continuous batching capacity exceeds the configured or validated sequence limit");
    // Each slot holds one row's recurrent state from before a speculative window.
    if (batch_mtp_enabled() && predictor_ && predictor_->supports_batch())
        mtp_checkpoints_ = batch_mtp_checkpoints(*predictor_);
}

QwenBatchOperations::~QwenBatchOperations() = default;

int64_t QwenBatchOperations::vocab_size() const { return model_.vocab_size(); }

int64_t QwenBatchOperations::max_context() const { return model_.max_position_embeddings(); }

std::vector<std::pair<std::string, double>> QwenBatchOperations::metrics() const {
    const auto paged = state_adapter_.paged_kv_stats();
    return {
        {"continuous_batching_mtp_supported", double(batch_mtp_enabled() && predictor_ && predictor_->supports_batch())},
        {"continuous_batching_mtp_stochastic_supported", 0.0},
        {"continuous_batching_mtp_graph_supported", 0.0},
        {"continuous_batching_mtp_checkpoints", double(mtp_checkpoints_)},
        {"continuous_batching_mtp_batches", double(mtp_batches_)},
        {"continuous_batching_mtp_drafted_tokens", double(mtp_drafted_)},
        {"continuous_batching_mtp_accepted_tokens", double(mtp_accepted_)},
        {"continuous_batching_decode_batches", static_cast<double>(decode_batches_)},
        {"continuous_batching_decode_tokens", static_cast<double>(decode_tokens_)},

        {"continuous_batching_compactions", 0.0},
        {"continuous_batching_stable_slot_releases",
         static_cast<double>(state_adapter_.slot_releases())},
        {"continuous_batching_batched_greedy_batches",
         static_cast<double>(batched_greedy_batches_)},
        {"continuous_batching_packed_metadata_batches",
         static_cast<double>(packed_metadata_batches_)},
        {"continuous_batching_cuda_graph_captures", static_cast<double>(decode_graphs_->captures)},
        {"continuous_batching_cuda_graph_replays", static_cast<double>(decode_graphs_->replays)},
        {"continuous_batching_cuda_graph_membership_captures", double(decode_graphs_->membership_captures)},
        {"continuous_batching_cuda_graph_bucket_reuses", double(decode_graphs_->bucket_reuses)},
        {"continuous_batching_cuda_graph_capture_ms", decode_graphs_->capture_ms},
        {"continuous_batching_moe", moe_enabled_ ? 1.0 : 0.0},
        {"continuous_batching_moe_cached_row_serial", cached_moe_enabled_ ? 1.0 : 0.0},
        {"continuous_batching_paged_kv", state_adapter_.paged_kv_enabled() ? 1.0 : 0.0},
        {"paged_kv_page_size", static_cast<double>(paged.page_size)},
        {"paged_kv_live_pages", static_cast<double>(paged.live_pages)},
        {"paged_kv_peak_live_pages", static_cast<double>(paged.peak_live_pages)},
        {"paged_kv_capacity_pages", static_cast<double>(paged.capacity_pages)},
        {"paged_kv_reserved_bytes", static_cast<double>(paged.reserved_bytes)},
        {"paged_kv_page_allocations", static_cast<double>(paged.allocations)},
        {"paged_kv_page_reuses", static_cast<double>(paged.reuses)},
        {"paged_kv_page_releases", static_cast<double>(paged.releases)},
        {"paged_kv_table_updates", static_cast<double>(paged.table_updates)},
    };
}

void QwenBatchOperations::initialize_sampling(Request &request, const Tensor &prompt_ids) {
    request.sampler.emplace(request.sampling);
    request.mtp = make_batch_mtp(predictor_, request.sampling);
    if (request.sampler->has_penalties())
        request.counts = SamplingOps::token_counts(prompt_ids, model_.vocab_size());
}

void QwenBatchOperations::suspend_decode() {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    state_adapter_.suspend_decode();
}

void QwenBatchOperations::release_idle_buffers() {
    invalidate_decode_graph();
    suspend_decode();
}

QwenBatchOperations::Sample QwenBatchOperations::prefill(
    const std::shared_ptr<Request> &request, mfq::engine::PrefillChunk chunk) {
    const int primary = execution_.layer_placement.primary_device();
    MfqCudaGuard guard(primary);
    state_adapter_.prepare_prefill(request->cache, chunk.offset, request->prompt.size());
    if (chunk.offset == 0) {
        std::erase_if(requests_, [](const auto& weak) { return weak.expired(); });
        requests_.push_back(request);
        request->prefill_ids =
            mfq_tensor_backend::tensor(
                request->prompt,
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, primary)))
                .reshape({1, -1})
                .contiguous();
        initialize_sampling(*request, request->prefill_ids);
    }
    Tensor hidden, raw;
    std::unique_ptr<PrefillCudaTimer> final_timer;
    auto chunk_timer = std::make_unique<PrefillCudaTimer>();
    MfqOptional<Tensor> length = mfq_nullopt;
    if (chunk.offset > 0 && chunk.count == 1)
        length =
            mfq_tensor_backend::full({1}, chunk.offset + 1, request->prefill_ids.options());
    hidden = model_.hidden_forward(
        request->prefill_ids.narrow(1, chunk.offset, chunk.count).contiguous(), mfq_nullopt,
        length, nullptr, mfq_nullopt, request->mtp ? &raw : nullptr);
    if (request->mtp)
        prime_batch_mtp(*predictor_, MtpTarget(model_), *request->mtp, std::move(raw),
            request->prefill_ids, chunk.offset, request->prompt.size());
    const auto end = chunk.offset + chunk.count;
    if (end < request->prefill_ids.size(1)) {
        MFQ_CUDA_CHECK(
            cudaEventRecord(chunk_timer->finished_event(), mfq_get_current_cuda_stream()));
        request->prefill_timers.push_back(std::move(chunk_timer));
    } else {
        final_timer = std::move(chunk_timer);
    }
    if (end < static_cast<int64_t>(request->prompt.size())) {
        state_adapter_.pause_prefill(request->cache);
        return {};
    }
    auto logits = state_adapter_.logits_from_last_hidden(std::move(hidden));
    MFQ_RUNTIME_CHECK(final_timer != nullptr, "continuous batching lost the prefill timer");
    MFQ_CUDA_CHECK(
        cudaEventRecord(final_timer->finished_event(), mfq_get_current_cuda_stream()));
    request->prefill_timers.push_back(std::move(final_timer));
    auto cursor = request->mtp && request->token_constraint
        ? request->token_constraint->clone() : request->token_constraint;
    MFQ_RUNTIME_CHECK(!request->token_constraint || cursor, "batch prefill grammar clone is incomplete");
    auto next = mfq::cuda::sample_logits(*request->sampler, std::move(logits), request->counts, cursor);
    const int64_t token = next.item<int64_t>();
    double prefill_ms = 0.0;
    for (const auto &timer : request->prefill_timers) {
        prefill_ms += timer->elapsed_ms();
    }
    request->prefill_timers.clear();
    request->prefill_ids = Tensor();
    return {std::move(next), token, {request->prompt.size(), prefill_ms, 0.0, prefill_ms}};
}

void QwenBatchOperations::activate(const std::shared_ptr<Request> &request, const Sample &sample) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    if (request->counts.defined() && !request->mtp)
        sample_token_counts_add_cuda(request->counts, sample.next.contiguous());
    state_adapter_.activate(request->cache);
    if (request->mtp) {
        request->mtp->first_token = sample.token;
        request->mtp->history_ids = {sample.token};
    }
    admission_storage_ready_ = true;
}

void QwenBatchOperations::resume_decode(int64_t cache_position) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    state_adapter_.resume_decode(cache_position);
}

void QwenBatchOperations::discard_prefill(const std::shared_ptr<Request> &request) {
    if (request->mtp) request->mtp->discard();
    request->mtp.reset();
    state_adapter_.discard_prefill(request->cache);
    request->prefill_timers.clear();
    request->prefill_ids = Tensor();
}

void QwenBatchOperations::retire(const std::vector<std::shared_ptr<Request>> &retired,
                                 int64_t cache_position) {
    for (const auto &request : retired) {
        if (request->mtp) request->mtp->discard();
        request->mtp.reset();
        state_adapter_.release(request->cache);
    }
    state_adapter_.finish_retire(cache_position);
    if (cache_position == 0) admission_storage_ready_ = false;
}

void QwenBatchOperations::invalidate_decode_graph() {
    decode_graphs_->clear();
}

QwenBatchOperations::Sample QwenBatchOperations::sample(
    const std::shared_ptr<Request> &request, const Decoded &decoded) {
    const auto slot = request->decode_row;
    MFQ_RUNTIME_CHECK(slot >= 0, "continuous batching sampled an inactive row");
    if (decoded.greedy_tokens.defined())
        return {{}, decoded.greedy_tokens.data_ptr<int64_t>()[slot], {}};
    return {decoded.sampled_next.narrow(0, slot, 1), decoded.sampled_tokens.data_ptr<int64_t>()[slot], {}};
}

void QwenBatchOperations::accept(const std::shared_ptr<Request> &request, const Sample &sample) {
    if (request->counts.defined()) {
        MFQ_RUNTIME_CHECK(sample.next.defined(),
                          "batched greedy sampling cannot update token penalties");
        sample_token_counts_add_cuda(request->counts, sample.next.contiguous());
    }
}

void QwenBatchOperations::recover(State &state) {
    std::vector<QwenBatchRequestState *> caches;
    for (const auto &request : state.prefilling) {
        if (request->mtp) request->mtp->discard();
        request->mtp.reset();
        caches.push_back(&request->cache);
    }
    for (const auto &request : state.active) {
        if (request->mtp) request->mtp->discard();
        request->mtp.reset();
        caches.push_back(&request->cache);
    }
    state_adapter_.recover(caches);
    admission_storage_ready_ = false;
    invalidate_decode_graph();
}

FlashNextBatchOperations::FlashNextBatchOperations(Qwen4CausalLm& model,
        CudaExecutionContext& execution, CudaContinuousBatchConfig config, MtpModule* predictor,
        BatchMemoryProvider memory)
    : model_(model), execution_(execution), config_(std::move(config)), memory_(std::move(memory)),
      predictor_(predictor) {
    MFQ_RUNTIME_CHECK(config_.max_sequences > 0 && config_.max_sequences <= config_.sequence_limit &&
                      config_.sequence_limit <= mfq::engine::ContinuousBatchConfig::validated_max_sequences,
                      "continuous batching capacity exceeds the configured or validated sequence limit");
    model_.reset(1);
    admission_base_kv_bytes_ = managed_kv_bytes();
    if (config_.cuda_graph) graph_ = std::make_unique<Qwen4BatchDecodeGraph>(model_);
    // Each slot holds the recurrent state after one drafted token.
    if (batch_mtp_enabled() && predictor_ && predictor_->supports_batch())
        mtp_checkpoints_ = std::make_shared<Qwen4BatchCheckpointPool>(batch_mtp_checkpoints(*predictor_));
}
FlashNextBatchOperations::~FlashNextBatchOperations() = default;
int64_t FlashNextBatchOperations::vocab_size() const { return model_.vocab_size(); }
int64_t FlashNextBatchOperations::max_context() const { return model_.max_position_embeddings(); }
mfq::engine::Metrics FlashNextBatchOperations::metrics() const {
    std::size_t bytes = 0, contexts = 0;
    for (const auto& weak : requests_) if (const auto request = weak.lock(); request && request->cache) {
        bytes += request->cache->resident_bytes();
        contexts += request->cache->position() > 0;
    }
    return {{"continuous_batching_mtp_supported", double(batch_mtp_enabled() && predictor_ && predictor_->supports_batch())},
        {"continuous_batching_mtp_stochastic_supported", 0.0},
        {"continuous_batching_mtp_graph_supported", 0.0},
        {"continuous_batching_mtp_checkpoints", double(mtp_checkpoints_ ? mtp_checkpoints_->tokens() : 0)},
        {"continuous_batching_mtp_batches", double(mtp_batches_)},
        {"continuous_batching_mtp_drafted_tokens", double(mtp_drafted_)},
        {"continuous_batching_mtp_accepted_tokens", double(mtp_accepted_)},
        {"continuous_batching_decode_batches",double(decode_batches_)},
        {"continuous_batching_decode_tokens",double(decode_tokens_)},
        {"continuous_batching_batched_greedy_batches",double(greedy_batches_)},
        {"continuous_batching_packed_prefill_batches",double(packed_prefill_batches_)},
        {"continuous_batching_packed_prefill_requests",double(packed_prefill_requests_)},
        {"continuous_batching_packed_prefill_tokens",double(packed_prefill_tokens_)},
        {"continuous_batching_moe_cached_row_serial",0.0},
        {"continuous_batching_cuda_graph_captures",double(graph_ ? graph_->captures() : 0)},
        {"continuous_batching_cuda_graph_replays",double(graph_ ? graph_->replays() : 0)},
        {"continuous_batching_cuda_graph_membership_captures",double(graph_ ? graph_->membership_captures() : 0)},
        {"continuous_batching_cuda_graph_bucket_reuses",double(graph_ ? graph_->bucket_reuses() : 0)},
        {"continuous_batching_cuda_graph_capture_ms",graph_ ? graph_->capture_ms() : 0},
        {"continuous_batching_request_kv_bytes",double(bytes)},
        {"continuous_batching_request_contexts",double(contexts)}};
}
void FlashNextBatchOperations::suspend_decode() {}
void FlashNextBatchOperations::release_idle_buffers() {
    if (graph_) graph_->clear();
}
void FlashNextBatchOperations::resume_decode(int64_t) {}
static std::exception_ptr flash_next_prefill_failure(std::exception_ptr failure) {
    try {std::rethrow_exception(failure);}
    catch(const std::exception& error) {
        const std::string message=error.what();
        if(dynamic_cast<const MfqResourceExhausted*>(&error) ||
           message.starts_with("CUDA process VRAM limit exceeded:") ||
           message.starts_with("CUDA VRAM limit exceeded:") ||
           message.starts_with("CUDA pinned allocation failed:"))
            return std::make_exception_ptr(mfq::engine::InferenceInputError(
                mfq::engine::InferenceInputErrorCode::ContextLength,
                "insufficient GPU or pinned RAM for prefill workspace; reduce concurrent requests or the packed prefill token budget"));
    } catch(...) {}
    return failure;
}
void FlashNextBatchOperations::initialize_prefill(
        const std::shared_ptr<Request>& request, mfq::engine::PrefillChunk chunk) {
    if (!request->cache) {
        MFQ_RUNTIME_CHECK(chunk.offset == 0,"Flash-Next batch lost request state");
        std::erase_if(requests_,[](const auto& weak) { return weak.expired(); });
        const bool active=std::any_of(requests_.begin(),requests_.end(),[](const auto& weak) {
            const auto row=weak.lock();return row && row->cache;
        });
        if(!active) {
            // The preceding exclusive request's global KV/recurrent state is
            // unused while independent batch rows own their own context.
            qwen4_release_decode_graphs(model_,nullptr);
            model_.reset(1);
            if (!graph_ || !graph_->has_graphs())
                release_moe_decode_buffers(execution_.moe_expert_cache);
            default_context(execution_.layer_placement.primary_device())->reclaim_idle_memory();
        }
        requests_.push_back(request);
        request->cache = std::make_unique<Qwen4BatchRequestState>(model_);
        request->prefill_ids = mfq_tensor_backend::tensor(request->prompt,
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA,
                    execution_.layer_placement.primary_device()))).reshape({1,-1});
        request->sampler.emplace(request->sampling);
        request->mtp = make_batch_mtp(predictor_, request->sampling, true);
        if (request->sampler->has_penalties())
            request->counts = SamplingOps::token_counts(request->prefill_ids,model_.vocab_size());
    }
    MFQ_RUNTIME_CHECK(request->cache->position() == chunk.offset,"Flash-Next batch prefill position mismatch");
}
FlashNextBatchOperations::Sample FlashNextBatchOperations::finish_prefill(
        const std::shared_ptr<Request>& request,Tensor next) {
    if (!next.defined()) return {};
    const auto token=next.item<int64_t>();
    double milliseconds=request->packed_prefill_ms;
    for (const auto& measured:request->prefill_timers) milliseconds+=measured->elapsed_ms();
    request->prefill_timers.clear(); request->prefill_ids={}; request->packed_prefill_ms=0;
    return {std::move(next),token,{request->prompt.size(),milliseconds,0.0,milliseconds}};
}
FlashNextBatchOperations::Sample FlashNextBatchOperations::prefill(
        const std::shared_ptr<Request>& request, mfq::engine::PrefillChunk chunk) {
    try {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    initialize_prefill(request,chunk);
    auto timer = std::make_unique<PrefillCudaTimer>();
    Tensor raw;
    auto hidden = request->cache->prefill(model_,request->prefill_ids.narrow(1,chunk.offset,chunk.count).contiguous(),
        request->mtp ? &raw : nullptr);
    if (request->mtp)
        prime_batch_mtp(*predictor_, MtpTarget(model_), *request->mtp, std::move(raw),
            request->prefill_ids, chunk.offset, request->prompt.size());
    const auto finished = chunk.offset + chunk.count == static_cast<int64_t>(request->prompt.size());
    Tensor next;
    if (finished) {
        auto cursor = request->mtp && request->token_constraint
            ? request->token_constraint->clone() : request->token_constraint;
        MFQ_RUNTIME_CHECK(!request->token_constraint || cursor, "batch prefill grammar clone is incomplete");
        next = sample_logits(*request->sampler,
            model_.logits_from_hidden(model_.last_hidden(std::move(hidden))).reshape({1,-1}),
            request->counts,cursor);
    }
    MFQ_CUDA_CHECK(cudaEventRecord(timer->finished_event(),mfq_get_current_cuda_stream()));
    request->prefill_timers.push_back(std::move(timer));
    return finish_prefill(request,std::move(next));
    } catch(...) {std::rethrow_exception(flash_next_prefill_failure(std::current_exception()));}
}
std::size_t FlashNextBatchOperations::prefill_batch_limit() const {
    return runtime_options::batch_packed_prefill()?config_.max_sequences:1;
}
int64_t FlashNextBatchOperations::prefill_batch_token_budget() const {
    return runtime_options::batch_packed_prefill_token_budget();
}
std::vector<mfq::engine::PrefillBatchResult<FlashNextBatchOperations::Sample>>
FlashNextBatchOperations::prefill_batch(
        const std::vector<std::pair<std::shared_ptr<Request>,mfq::engine::PrefillChunk>>& items) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    qwen4_release_decode_graphs(model_,nullptr);
    model_.adapter_begin_forward(false);
    if (!graph_ || !graph_->has_graphs())
        release_moe_decode_buffers(execution_.moe_expert_cache);
    std::vector<mfq::engine::PrefillBatchResult<Sample>> result(items.size());
    if (std::any_of(items.begin(), items.end(), [&](const auto& item) {
            return item.first->sampling.enable_mtp && batch_mtp_supported(predictor_, item.first->sampling);
        })) {
        for (std::size_t row = 0; row < items.size(); ++row) {
            try { result[row].sample = prefill(items[row].first, items[row].second); }
            catch (...) { result[row].failure = std::current_exception(); }
        }
        return result;
    }
    std::vector<std::size_t> ready;
    for (std::size_t row=0;row<items.size();++row) {
        try { initialize_prefill(items[row].first,items[row].second); ready.push_back(row); }
        catch (...) { result[row].failure=flash_next_prefill_failure(std::current_exception()); }
    }
    const auto execute=[&](auto&& self,const std::vector<std::size_t>& rows)->void {
        if (rows.empty()) return;
        std::size_t growth=std::size_t(256)<<20;
        for (const auto row:rows) growth+=items[row].first->cache->prefill_growth_bytes(items[row].second.count);
        try { if(model_.reserve_kv_workspace)model_.reserve_kv_workspace(growth); }
        catch (const mfq::engine::InferenceInputError& error) {
            if(error.code!=mfq::engine::InferenceInputErrorCode::ContextLength)throw;
            if(rows.size()==1) {result[rows.front()].failure=std::current_exception();return;}
            const auto middle=rows.begin()+rows.size()/2;
            self(self,std::vector<std::size_t>(rows.begin(),middle));
            self(self,std::vector<std::size_t>(middle,rows.end()));
            return;
        }
        try {
            if(rows.size()==1) {
                const auto row=rows.front();
                result[row].sample=prefill(items[row].first,items[row].second);
                return;
            }
            std::vector<Qwen4BatchRequestState*> caches;
            std::vector<Tensor> chunks;
            int64_t tokens=0;
            for (const auto row:rows) {
                const auto& [request,chunk]=items[row];
                caches.push_back(request->cache.get());
                chunks.push_back(request->prefill_ids.narrow(1,chunk.offset,chunk.count).contiguous());
                tokens+=chunk.count;
            }
            PrefillCudaTimer timer;
            auto hidden=qwen4_batch_prefill(model_,caches,chunks);
            std::vector<Tensor> last;
            std::vector<std::size_t> finished;
            for(std::size_t i=0;i<rows.size();++i) {
                const auto& [request,chunk]=items[rows[i]];
                if(chunk.offset+chunk.count==static_cast<int64_t>(request->prompt.size())) {
                    last.push_back(hidden[i]);finished.push_back(rows[i]);
                }
            }
            std::vector<Tensor> next(items.size());
            if(!last.empty()) {
                auto logits=model_.logits_from_hidden(mfq_tensor_backend::cat(last,0)).reshape({int64_t(last.size()),-1});
                for(std::size_t i=0;i<finished.size();++i) {
                    const auto row=finished[i];const auto& request=items[row].first;
                    next[row]=sample_logits(*request->sampler,logits.narrow(0,i,1),request->counts,request->token_constraint);
                }
            }
            MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
            const auto milliseconds=timer.elapsed_ms();
            for(const auto row:rows) {
                items[row].first->packed_prefill_ms+=milliseconds*items[row].second.count/tokens;
                result[row].sample=finish_prefill(items[row].first,std::move(next[row]));
            }
            ++packed_prefill_batches_;packed_prefill_requests_+=rows.size();packed_prefill_tokens_+=tokens;
        } catch (...) {
            // Kernels may have advanced attention state. Fail this group;
            // splitting/retrying is allowed only before numerical execution.
            const auto error=flash_next_prefill_failure(std::current_exception());
            for(const auto row:rows)result[row].failure=error;
        }
    };
    execute(execute,ready);
    return result;
}
void FlashNextBatchOperations::activate(const std::shared_ptr<Request>& request, const Sample& sampled) {
    if (!request->mtp) accept(request,sampled);
    if (request->mtp) {
        request->mtp->first_token = sampled.token;
        request->mtp->history_ids = {sampled.token};
    }
}
void FlashNextBatchOperations::discard_prefill(const std::shared_ptr<Request>& request) {
    finish_moe_expert_exchanges(execution_.moe_expert_cache);
    if (request->mtp) request->mtp->discard();
    request->mtp.reset();
    request->cache.reset(); request->prefill_ids = {}; request->prefill_timers.clear(); request->packed_prefill_ms=0;
}
void FlashNextBatchOperations::retire(const std::vector<std::shared_ptr<Request>>& retired,int64_t) {
    for (const auto& request : retired) discard_prefill(request);
    const bool active = std::any_of(requests_.begin(),requests_.end(),[](const auto& weak) {
        const auto request = weak.lock();
        return request && request->cache;
    });
    if (!active && !model_.graph_registry && (!graph_ || !graph_->has_graphs())) {
        // Eager decoding has no bucket windows retaining these transfer buffers.
        release_moe_decode_buffers(execution_.moe_expert_cache);
        default_context(execution_.layer_placement.primary_device())->reclaim_idle_memory();
    }
}
FlashNextBatchOperations::Decoded FlashNextBatchOperations::decode(State& state) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    if (std::any_of(state.active.begin(), state.active.end(), [](const auto& request) {
            return request->eligible && bool(request->mtp);
        })) return decode_mtp(state);
    std::vector<Qwen4BatchRequestState*> caches;
    std::vector<int64_t> tokens;
    for (const auto& request : state.active) {
        request->decode_row = -1;
        if (!request->eligible) continue;
        request->decode_row = static_cast<int64_t>(tokens.size());
        MFQ_RUNTIME_CHECK(request->cache && request->cache->position() == request->cache_length,
                          "Flash-Next batch decode position mismatch");
        caches.push_back(request->cache.get()); tokens.push_back(request->pending_token);
    }
    auto host_ids = mfq_tensor_backend::tensor(tokens,
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64)).reshape({-1,1});
    auto ids = host_ids.to(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA,
                                                   execution_.layer_placement.primary_device()));
    MoeContinuousBatchCacheScope moe_cache_scope(execution_,tokens.size() > 8);
    const bool greedy = config_.greedy && std::all_of(state.active.begin(),state.active.end(),[](const auto& request) {
        return !request->eligible || (request->sampler->greedy() && !request->sampler->has_penalties() && !request->token_constraint);
    });
    std::optional<Tensor> captured;
    try { if(graph_)captured=graph_->run(caches,ids,host_ids,greedy,config_.max_sequences); }
    catch(const std::exception& error) {
        const std::string message=error.what();
        if(!dynamic_cast<const MfqResourceExhausted*>(&error) &&
           !message.starts_with("CUDA process VRAM limit exceeded:") &&
           !message.starts_with("CUDA VRAM limit exceeded:") &&
           !message.starts_with("CUDA pinned allocation failed:"))throw;
        throw mfq::engine::InferenceInputError(mfq::engine::InferenceInputErrorCode::ContextLength,
            "insufficient GPU or pinned RAM for the batch decode workspace; reduce concurrent requests");
    }
    auto logits = captured ? std::move(*captured) : qwen4_batch_decode(model_,caches,std::move(ids),host_ids);
    Tensor greedy_tokens;
    if (greedy) {
        greedy_tokens = captured ? logits.to(mfq_tensor_backend::kCPU).contiguous()
            : sample_greedy_cuda(logits.contiguous()).to(mfq_tensor_backend::kCPU).contiguous();
        ++greedy_batches_;
    }
    ++decode_batches_; decode_tokens_ += tokens.size();
    BatchSamplingResult sampled;
    if (!greedy_tokens.defined()) {
        std::vector<BatchSamplingInput> inputs(tokens.size());
        for (const auto& request : state.active) {
            if (!request->eligible) continue;
            inputs[request->decode_row] = {&*request->sampler, request->counts, request->token_constraint};
        }
        sampling_.reserve(int(config_.max_sequences), int(vocab_size()), logits.device());
        sampled = sampling_.sample(logits.contiguous().view({int64_t(tokens.size()), -1}), inputs);
    }
    return {std::move(logits), std::move(greedy_tokens), std::move(sampled.next), std::move(sampled.host)};
}
FlashNextBatchOperations::Sample FlashNextBatchOperations::sample(
        const std::shared_ptr<Request>& request,const Decoded& decoded) {
    MFQ_RUNTIME_CHECK(request->decode_row >= 0,"Flash-Next sampled an inactive row");
    if(decoded.greedy_tokens.defined())return {{},decoded.greedy_tokens.data_ptr<int64_t>()[request->decode_row],{}};
    return {decoded.sampled_next.narrow(0, request->decode_row, 1),
        decoded.sampled_tokens.data_ptr<int64_t>()[request->decode_row], {}};
}
void FlashNextBatchOperations::accept(const std::shared_ptr<Request>& request,const Sample& sampled) {
    if (request->counts.defined()) sample_token_counts_add_cuda(request->counts,sampled.next.contiguous());
}
void FlashNextBatchOperations::recover(State& state) {
    if (graph_) graph_->clear();
    for (const auto& request : state.active) discard_prefill(request);
    for (const auto& request : state.prefilling) discard_prefill(request);
}
} // namespace mfq::cuda
