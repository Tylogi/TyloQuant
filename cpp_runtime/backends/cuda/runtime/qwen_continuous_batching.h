#pragma once

// Included by mfq_decode.cpp after the CUDA Qwen model and sampler are defined.
// The scheduler owns request concurrency; the adapter below owns the hybrid
// full-attention/recurrent state carried between decode iterations.

namespace mfq::cuda::continuous {

using Tensor = mfq_tensor_backend::Tensor;

struct QwenBatchLayerState {
    enum class Kind { FullAttention, Recurrent };
    Kind kind = Kind::FullAttention;
    Tensor first;
    Tensor second;
    bool ring = false;
};

struct QwenBatchState {
    int64_t batch = 0;
    std::vector<QwenBatchLayerState> layers;
};

static void clear_full_attention_decode_workspaces(FullBlock & block) {
    block.decode_partial_o = Tensor();
    block.decode_partial_m = Tensor();
    block.decode_partial_l = Tensor();
    block.decode_mma_mask = Tensor();
    block.decode_mma_kv_max = Tensor();
    block.decode_mma_meta = Tensor();
}

static std::string qwen_continuous_batching_incompatibility(
        const Model & model) {
    if (model.c.runtime_plan.backbone !=
            mfq::cuda::MfqCudaBackbone::generic_qwen ||
            model.c.is_minicpmo45() || model.c.is_qwen4() ||
            model.c.is_flash_next()) {
        return "continuous batching currently requires the generic Qwen runtime";
    }
    if (model.blocks.empty()) {
        return "continuous batching requires at least one model block";
    }
    if (g_dense_cpu_layer_count != 0 || !g_dsv4_cpu_offload_layers.empty()) {
        return "continuous batching requires GPU-resident model blocks";
    }
    if (g_moe_expert_cache) {
        return "continuous batching cannot use the expert cache";
    }
    for (const auto & block : model.blocks) {
        if (block->cpu_offloaded) {
            return "continuous batching cannot use CPU-offloaded blocks";
        }
        if (const auto * full = dynamic_cast<const FullBlock *>(block.get())) {
            if (full->sliding || full->ffn.is_moe) {
                return "continuous batching requires dense non-sliding Qwen blocks";
            }
            continue;
        }
        if (const auto * linear = dynamic_cast<const LinearBlock *>(block.get())) {
            if (linear->ffn.is_moe) {
                return "continuous batching requires dense Qwen blocks";
            }
            continue;
        }
        return "continuous batching encountered an unsupported Qwen block";
    }
    return {};
}

static QwenBatchState take_qwen_batch_state(Model & model, int64_t batch) {
    MFQ_RUNTIME_CHECK(model.speculative_start < 0,
        "continuous batching cannot detach speculative state");
    QwenBatchState state;
    state.batch = batch;
    state.layers.reserve(model.blocks.size());
    for (auto & block : model.blocks) {
        MfqCudaGuard guard(block->cuda_device);
        QwenBatchLayerState layer;
        if (auto * full = dynamic_cast<FullBlock *>(block.get())) {
            MFQ_RUNTIME_CHECK(full->cache.k.defined() && full->cache.v.defined() &&
                full->cache.k.dim() == 4 &&
                full->cache.k.size(0) == batch &&
                full->cache.v.sizes() == full->cache.k.sizes(),
                "continuous batching full-attention state is unavailable");
            layer.kind = QwenBatchLayerState::Kind::FullAttention;
            layer.first = full->cache.k;
            layer.second = full->cache.v;
            layer.ring = full->cache.ring;
            full->cache = KVCache();
            clear_full_attention_decode_workspaces(*full);
        } else if (auto * linear = dynamic_cast<LinearBlock *>(block.get())) {
            MFQ_RUNTIME_CHECK(linear->conv_state.defined() &&
                linear->gdn_state.defined() &&
                linear->conv_state.size(0) == batch &&
                linear->gdn_state.size(0) == batch &&
                !linear->speculative_pending,
                "continuous batching recurrent state is unavailable");
            layer.kind = QwenBatchLayerState::Kind::Recurrent;
            layer.first = linear->conv_state;
            layer.second = linear->gdn_state;
            linear->conv_state = Tensor();
            linear->gdn_state = Tensor();
            linear->speculative_conv = Tensor();
            linear->speculative_gdn = Tensor();
            linear->speculative_pending = false;
        } else {
            throw std::runtime_error(
                "continuous batching encountered an unsupported block state");
        }
        state.layers.push_back(std::move(layer));
    }
    model.cache_pos = 0;
    return state;
}

static Tensor merge_batch_tensors(
        const std::vector<QwenBatchState> & states, size_t layer,
        bool second) {
    std::vector<Tensor> values;
    values.reserve(states.size());
    for (const auto & state : states) {
        values.push_back(second
            ? state.layers[layer].second
            : state.layers[layer].first);
    }
    return values.size() == 1 ? values.front() :
        mfq_tensor_backend::cat(values, 0).contiguous();
}

static void restore_qwen_batch_states(
        Model & model, const std::vector<QwenBatchState> & states,
        int64_t cache_position) {
    MFQ_RUNTIME_CHECK(!states.empty(),
        "continuous batching cannot restore an empty state list");
    int64_t batch = 0;
    for (const auto & state : states) {
        MFQ_RUNTIME_CHECK(state.batch > 0 &&
            state.layers.size() == model.blocks.size(),
            "continuous batching state layout changed");
        batch += state.batch;
    }
    for (size_t layer_index = 0;
            layer_index < model.blocks.size(); ++layer_index) {
        auto & block = model.blocks[layer_index];
        MfqCudaGuard guard(block->cuda_device);
        const auto kind = states.front().layers[layer_index].kind;
        for (const auto & state : states) {
            MFQ_RUNTIME_CHECK(state.layers[layer_index].kind == kind,
                "continuous batching mixed incompatible layer states");
        }
        if (auto * full = dynamic_cast<FullBlock *>(block.get())) {
            MFQ_RUNTIME_CHECK(kind ==
                QwenBatchLayerState::Kind::FullAttention,
                "continuous batching full-attention state kind changed");
            const auto shape = states.front().layers[layer_index].first.sizes();
            for (const auto & state : states) {
                const auto & saved = state.layers[layer_index];
                MFQ_RUNTIME_CHECK(!saved.ring &&
                    saved.first.dim() == 4 &&
                    saved.second.sizes() == saved.first.sizes() &&
                    saved.first.size(1) == shape[1] &&
                    saved.first.size(2) == shape[2] &&
                    saved.first.size(3) == shape[3],
                    "continuous batching KV cache geometry changed");
            }
            full->cache.k = merge_batch_tensors(
                states, layer_index, false);
            full->cache.v = merge_batch_tensors(
                states, layer_index, true);
            full->cache.ring = false;
            MFQ_RUNTIME_CHECK(full->cache.k.size(0) == batch,
                "continuous batching KV merge produced the wrong batch");
            clear_full_attention_decode_workspaces(*full);
        } else if (auto * linear = dynamic_cast<LinearBlock *>(block.get())) {
            MFQ_RUNTIME_CHECK(kind == QwenBatchLayerState::Kind::Recurrent,
                "continuous batching recurrent state kind changed");
            linear->conv_state = merge_batch_tensors(
                states, layer_index, false);
            linear->gdn_state = merge_batch_tensors(
                states, layer_index, true);
            linear->speculative_conv = Tensor();
            linear->speculative_gdn = Tensor();
            linear->speculative_pending = false;
            MFQ_RUNTIME_CHECK(linear->conv_state.size(0) == batch &&
                linear->gdn_state.size(0) == batch,
                "continuous batching recurrent merge produced the wrong batch");
        } else {
            throw std::runtime_error(
                "continuous batching restore encountered an unsupported block");
        }
    }
    model.cache_pos = cache_position;
    model.speculative_start = -1;
    model.speculative_confirmed = 0;
    model.qwen4_positions = Tensor();
    model.qwen4_batch = batch;
}

static void compact_qwen_batch_state(
        Model & model, const std::vector<int64_t> & rows,
        int64_t cache_position) {
    MFQ_RUNTIME_CHECK(!rows.empty(),
        "continuous batching cannot compact to an empty batch");
    for (auto & block : model.blocks) {
        MfqCudaGuard guard(block->cuda_device);
        auto indices = mfq_tensor_backend::tensor(
            rows, mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA, block->cuda_device)));
        if (auto * full = dynamic_cast<FullBlock *>(block.get())) {
            full->cache.k = full->cache.k
                .index_select(0, indices).contiguous();
            full->cache.v = full->cache.v
                .index_select(0, indices).contiguous();
            clear_full_attention_decode_workspaces(*full);
        } else if (auto * linear = dynamic_cast<LinearBlock *>(block.get())) {
            linear->conv_state = linear->conv_state
                .index_select(0, indices).contiguous();
            linear->gdn_state = linear->gdn_state
                .index_select(0, indices).contiguous();
            linear->speculative_conv = Tensor();
            linear->speculative_gdn = Tensor();
            linear->speculative_pending = false;
        }
    }
    model.cache_pos = cache_position;
}

static Tensor qwen_logits_from_last_hidden(Model & model, Tensor hidden) {
    auto last = hidden.index({Slice(), -1, Slice()})
        .to(mfq_tensor_backend::kFloat16).contiguous();
    auto logits = model.lm_head.forward(last);
    if (model.c.final_logit_softcapping > 0.0) {
        logits = mfq_tensor_backend::tanh(
            logits / model.c.final_logit_softcapping) *
            model.c.final_logit_softcapping;
    }
    return logits;
}

class CudaContinuousBatcher {
public:
    CudaContinuousBatcher(
            Model & model, std::mutex & model_mutex,
            int32_t max_sequences,
            std::chrono::microseconds initial_batch_wait =
                std::chrono::microseconds(1000))
        : model_(model), model_mutex_(model_mutex),
          max_sequences_(max_sequences),
          initial_batch_wait_(initial_batch_wait) {
        if (max_sequences_ < 1) {
            throw std::invalid_argument(
                "continuous batching max sequences must be positive");
        }
        const auto incompatibility =
            qwen_continuous_batching_incompatibility(model_);
        if (!incompatibility.empty()) {
            throw std::runtime_error(incompatibility);
        }
        worker_ = std::thread([this] { worker_main(); });
    }

    ~CudaContinuousBatcher() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stopping_ = true;
        }
        queue_ready_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    CudaContinuousBatcher(const CudaContinuousBatcher &) = delete;
    CudaContinuousBatcher & operator=(
        const CudaContinuousBatcher &) = delete;

    int32_t submit(
            const std::vector<int64_t> & prompt,
            const MfqSamplingParams & sampling,
            const MfqTokenCallback & on_token,
            const MfqPrefillCallback & on_prefill,
            const MfqPromptCachePlan & cache_plan,
            const MfqTokenConstraintPtr & token_constraint) {
        if (prompt.empty() ||
                prompt.size() > static_cast<size_t>(
                    model_.c.max_position_embeddings)) {
            throw std::invalid_argument(
                "continuous batching prompt length is invalid");
        }
        if (sampling.max_tokens < 0) {
            throw std::invalid_argument(
                "continuous batching max_tokens cannot be negative");
        }
        for (const auto token : prompt) {
            if (token < 0 || token >= model_.c.vocab_size) {
                throw std::invalid_argument(
                    "continuous batching prompt token is outside the vocabulary");
            }
        }
        if (sampling.max_tokens == 0 ||
                prompt.size() == static_cast<size_t>(
                    model_.c.max_position_embeddings)) {
            return 0;
        }
        auto request = std::make_shared<Request>(
            prompt, sampling, token_constraint);
        request->generation_limit = static_cast<int32_t>(
            std::min<int64_t>(sampling.max_tokens,
                model_.c.max_position_embeddings -
                    static_cast<int64_t>(prompt.size())));
        if (!cache_plan.session_id.empty() ||
                cache_plan.stable_prefix_tokens != 0) {
            ++prefix_cache_bypasses_;
        }
        if (sampling.enable_mtp) ++mtp_bypasses_;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stopping_) {
                throw std::runtime_error(
                    "continuous batching scheduler is stopping");
            }
            pending_.push_back(request);
            queued_.fetch_add(1, std::memory_order_relaxed);
        }
        queue_ready_.notify_one();
        int32_t delivered = 0;
        bool callbacks_enabled = true;
        std::exception_ptr callback_error;
        std::exception_ptr producer_error;
        for (;;) {
            std::optional<MfqPrefillTiming> prefill;
            std::optional<int64_t> token;
            bool producer_done = false;
            {
                std::unique_lock<std::mutex> lock(request->mutex);
                request->output_ready.wait(lock, [&] {
                    return request->prefill_timing.has_value() ||
                        !request->output_tokens.empty() || request->done;
                });
                if (request->prefill_timing.has_value()) {
                    prefill = std::move(request->prefill_timing);
                    request->prefill_timing.reset();
                } else if (!request->output_tokens.empty()) {
                    token = request->output_tokens.front();
                    request->output_tokens.pop_front();
                } else {
                    producer_done = request->done;
                    producer_error = request->error;
                }
            }
            if (producer_done) break;
            if (!callbacks_enabled) continue;
            try {
                if (prefill.has_value()) {
                    if (on_prefill) on_prefill(*prefill);
                } else if (token.has_value()) {
                    ++delivered;
                    if (on_token && !on_token(*token)) {
                        callbacks_enabled = false;
                        request->cancel_requested.store(
                            true, std::memory_order_release);
                        queue_ready_.notify_one();
                    }
                }
            } catch (...) {
                callback_error = std::current_exception();
                callbacks_enabled = false;
                request->cancel_requested.store(
                    true, std::memory_order_release);
                queue_ready_.notify_one();
            }
        }
        if (callback_error) std::rethrow_exception(callback_error);
        if (producer_error) std::rethrow_exception(producer_error);
        return delivered;
    }

    std::vector<std::pair<std::string, double>> metrics() const {
        return {
            {"continuous_batching_max_sequences",
                static_cast<double>(max_sequences_)},
            {"continuous_batching_active",
                static_cast<double>(active_count_.load())},
            {"continuous_batching_queued",
                static_cast<double>(queued_.load())},
            {"continuous_batching_requests",
                static_cast<double>(requests_.load())},
            {"continuous_batching_decode_batches",
                static_cast<double>(decode_batches_.load())},
            {"continuous_batching_decode_tokens",
                static_cast<double>(decode_tokens_.load())},
            {"continuous_batching_max_batch",
                static_cast<double>(max_batch_seen_.load())},
            {"continuous_batching_admissions",
                static_cast<double>(admissions_.load())},
            {"continuous_batching_compactions",
                static_cast<double>(compactions_.load())},
            {"continuous_batching_mtp_target_only_requests",
                static_cast<double>(mtp_bypasses_.load())},
            {"continuous_batching_prefix_cache_bypasses",
                static_cast<double>(prefix_cache_bypasses_.load())},
        };
    }

    int64_t queued_requests() const {
        return queued_.load(std::memory_order_relaxed);
    }

private:
    struct Request {
        Request(
                const std::vector<int64_t> & input_prompt,
                const MfqSamplingParams & input_sampling,
                const MfqTokenConstraintPtr & input_constraint)
            : prompt(input_prompt), sampling(input_sampling),
              token_constraint(input_constraint), rng(input_sampling.seed) {}

        std::vector<int64_t> prompt;
        MfqSamplingParams sampling;
        MfqTokenConstraintPtr token_constraint;
        std::mt19937_64 rng;
        Tensor counts;
        Tensor random_host;
        Tensor random_cuda;
        int32_t generation_limit = 0;
        int32_t produced = 0;
        int64_t pending_token = 0;
        int64_t cache_length = 0;
        std::mutex mutex;
        std::condition_variable output_ready;
        std::optional<MfqPrefillTiming> prefill_timing;
        std::deque<int64_t> output_tokens;
        std::atomic<bool> cancel_requested{false};
        bool done = false;
        std::exception_ptr error;
    };

    static void complete_request(
            const std::shared_ptr<Request> & request,
            std::exception_ptr error = {}) {
        {
            std::lock_guard<std::mutex> lock(request->mutex);
            if (request->done) return;
            request->error = error;
            request->done = true;
        }
        request->output_ready.notify_one();
    }

    static void publish_prefill(
            const std::shared_ptr<Request> & request,
            const MfqPrefillTiming & timing) {
        {
            std::lock_guard<std::mutex> lock(request->mutex);
            request->prefill_timing = timing;
        }
        request->output_ready.notify_one();
    }

    static void publish_token(
            const std::shared_ptr<Request> & request, int64_t token) {
        {
            std::lock_guard<std::mutex> lock(request->mutex);
            request->output_tokens.push_back(token);
        }
        request->output_ready.notify_one();
    }

    void fail_requests(
            const std::vector<std::shared_ptr<Request>> & requests,
            std::exception_ptr error) {
        for (const auto & request : requests) {
            complete_request(request, error);
        }
    }

    std::vector<std::shared_ptr<Request>> take_pending() {
        std::vector<std::shared_ptr<Request>> requests;
        std::lock_guard<std::mutex> lock(queue_mutex_);
        const size_t available = max_sequences_ >
                static_cast<int32_t>(active_.size())
            ? static_cast<size_t>(max_sequences_) - active_.size() : 0;
        const size_t count = std::min(available, pending_.size());
        requests.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            requests.push_back(std::move(pending_.front()));
            pending_.pop_front();
            queued_.fetch_sub(1, std::memory_order_relaxed);
        }
        return requests;
    }

    void initialize_sampling(Request & request, const Tensor & prompt_ids) {
        const int primary = g_layer_placement.primary_device();
        const auto cuda_options = mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA, primary));
        if (sampling_has_penalties(request.sampling)) {
            request.counts = mfq_tensor_backend::zeros(
                {model_.c.vocab_size},
                cuda_options.dtype(mfq_tensor_backend::kInt32));
            sample_token_counts_add_cuda(request.counts, prompt_ids);
        }
        request.random_host = mfq_tensor_backend::empty(
            {1}, mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::kCPU)
                .dtype(mfq_tensor_backend::kFloat32)
                .pinned_memory(true));
        request.random_cuda = mfq_tensor_backend::empty(
            {1}, cuda_options.dtype(mfq_tensor_backend::kFloat32));
    }

    void admit_requests(
            const std::vector<std::shared_ptr<Request>> & incoming) {
        if (incoming.empty()) return;
        std::lock_guard<std::mutex> model_lock(model_mutex_);
        const int primary = g_layer_placement.primary_device();
        MfqCudaGuard primary_guard(primary);
        std::vector<QwenBatchState> states;
        states.reserve(1 + incoming.size());
        if (!active_.empty()) {
            states.push_back(take_qwen_batch_state(
                model_, static_cast<int64_t>(active_.size())));
        }
        std::vector<std::shared_ptr<Request>> admitted;
        admitted.reserve(incoming.size());
        for (const auto & request : incoming) {
            try {
                model_.reset(1);
                auto ids = mfq_tensor_backend::tensor(
                    request->prompt,
                    mfq_tensor_backend::TensorOptions()
                        .dtype(mfq_tensor_backend::kInt64)
                        .device(mfq_tensor_backend::Device(
                            mfq_tensor_backend::kCUDA, primary)))
                    .reshape({1, -1}).contiguous();
                initialize_sampling(*request, ids);
                ServerPrefillCudaTimer timer;
                auto hidden = model_.hidden_forward(ids);
                auto logits = qwen_logits_from_last_hidden(
                    model_, std::move(hidden));
                MFQ_CUDA_CHECK(cudaEventRecord(
                    timer.finished_event(), mfq_get_current_cuda_stream()));
                auto next = sample_server_logits(
                    std::move(logits), request->sampling,
                    request->counts, request->random_host,
                    request->random_cuda, request->rng,
                    request->token_constraint);
                const int64_t token = next.item<int64_t>();
                const double prefill_ms = timer.elapsed_ms();
                publish_prefill(request, MfqPrefillTiming{
                    request->prompt.size(), prefill_ms, 0.0, prefill_ms});
                request->pending_token = token;
                request->cache_length =
                    static_cast<int64_t>(request->prompt.size());
                request->produced = 1;
                publish_token(request, token);
                if (request->cancel_requested.load(
                            std::memory_order_acquire) ||
                        request->produced >= request->generation_limit) {
                    complete_request(request);
                    continue;
                }
                if (request->counts.defined()) {
                    sample_token_counts_add_cuda(
                        request->counts, next.contiguous());
                }
                states.push_back(take_qwen_batch_state(model_, 1));
                admitted.push_back(request);
                ++admissions_;
                ++requests_;
            } catch (...) {
                try { model_.reset(1); } catch (...) {}
                complete_request(request, std::current_exception());
            }
        }
        active_.insert(active_.end(), admitted.begin(), admitted.end());
        if (!states.empty()) {
            int64_t max_cache_position = 0;
            for (const auto & request : active_) {
                max_cache_position = std::max(
                    max_cache_position, request->cache_length);
            }
            restore_qwen_batch_states(
                model_, states, max_cache_position);
        } else {
            model_.reset(1);
        }
        active_count_.store(
            static_cast<int64_t>(active_.size()),
            std::memory_order_relaxed);
        int64_t previous = max_batch_seen_.load();
        while (previous < static_cast<int64_t>(active_.size()) &&
                !max_batch_seen_.compare_exchange_weak(
                    previous, static_cast<int64_t>(active_.size()))) {}
    }

    void retire_cancelled_requests() {
        std::vector<std::shared_ptr<Request>> survivors;
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<int64_t> survivor_rows;
        int64_t survivor_max_position = 0;
        survivors.reserve(active_.size());
        survivor_rows.reserve(active_.size());
        for (size_t row = 0; row < active_.size(); ++row) {
            const auto & request = active_[row];
            if (request->cancel_requested.load(
                    std::memory_order_acquire)) {
                cancelled.push_back(request);
                continue;
            }
            survivors.push_back(request);
            survivor_rows.push_back(static_cast<int64_t>(row));
            survivor_max_position = std::max(
                survivor_max_position, request->cache_length);
        }
        if (survivors.size() == active_.size()) return;
        if (survivors.empty()) {
            model_.reset(1);
        } else {
            compact_qwen_batch_state(
                model_, survivor_rows, survivor_max_position);
            ++compactions_;
        }
        active_ = std::move(survivors);
        active_count_.store(
            static_cast<int64_t>(active_.size()),
            std::memory_order_relaxed);
        for (const auto & request : cancelled) {
            complete_request(request);
        }
    }

    void decode_active() {
        if (active_.empty()) return;
        std::lock_guard<std::mutex> model_lock(model_mutex_);
        const int primary = g_layer_placement.primary_device();
        MfqCudaGuard primary_guard(primary);
        retire_cancelled_requests();
        if (active_.empty()) return;
        const int64_t batch = static_cast<int64_t>(active_.size());
        std::vector<int64_t> input_tokens;
        std::vector<int64_t> positions;
        std::vector<int64_t> sequence_lengths;
        input_tokens.reserve(active_.size());
        positions.reserve(active_.size());
        sequence_lengths.reserve(active_.size());
        int64_t max_position = 0;
        int64_t max_sequence_length = 0;
        for (const auto & request : active_) {
            input_tokens.push_back(request->pending_token);
            positions.push_back(request->cache_length);
            sequence_lengths.push_back(request->cache_length + 1);
            max_position = std::max(max_position, request->cache_length);
            max_sequence_length = std::max(
                max_sequence_length, request->cache_length + 1);
        }
        const auto options = mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kInt64)
            .device(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA, primary));
        auto ids = mfq_tensor_backend::tensor(input_tokens, options)
            .reshape({batch, 1}).contiguous();
        auto pos = mfq_tensor_backend::tensor(positions, options)
            .reshape({batch, 1}).contiguous();
        auto lengths = mfq_tensor_backend::tensor(
            sequence_lengths, options).contiguous();
        Tensor logits;
        try {
            model_.cache_pos = max_position;
            auto hidden = model_.hidden_forward(
                ids, pos, lengths, nullptr, pos);
            model_.cache_pos = max_sequence_length;
            logits = qwen_logits_from_last_hidden(
                model_, std::move(hidden));
        } catch (...) {
            auto error = std::current_exception();
            try { model_.reset(1); } catch (...) {}
            fail_requests(active_, error);
            active_.clear();
            active_count_.store(0);
            return;
        }
        ++decode_batches_;
        decode_tokens_.fetch_add(batch);
        std::vector<std::shared_ptr<Request>> survivors;
        std::vector<std::pair<std::shared_ptr<Request>,
            std::exception_ptr>> completions;
        std::vector<int64_t> survivor_rows;
        survivors.reserve(active_.size());
        survivor_rows.reserve(active_.size());
        int64_t survivor_max_position = 0;
        for (size_t row = 0; row < active_.size(); ++row) {
            const auto & request = active_[row];
            request->cache_length += 1;
            if (request->cancel_requested.load(
                    std::memory_order_acquire)) {
                completions.emplace_back(request, std::exception_ptr{});
                continue;
            }
            try {
                auto next = sample_server_logits(
                    logits.narrow(0, static_cast<int64_t>(row), 1),
                    request->sampling, request->counts,
                    request->random_host, request->random_cuda,
                    request->rng, request->token_constraint);
                const int64_t token = next.item<int64_t>();
                request->pending_token = token;
                ++request->produced;
                publish_token(request, token);
                if (request->cancel_requested.load(
                            std::memory_order_acquire) ||
                        request->produced >= request->generation_limit) {
                    completions.emplace_back(
                        request, std::exception_ptr{});
                    continue;
                }
                if (request->counts.defined()) {
                    sample_token_counts_add_cuda(
                        request->counts, next.contiguous());
                }
                survivors.push_back(request);
                survivor_rows.push_back(static_cast<int64_t>(row));
                survivor_max_position = std::max(
                    survivor_max_position, request->cache_length);
            } catch (...) {
                completions.emplace_back(
                    request, std::current_exception());
            }
        }
        if (survivors.empty()) {
            model_.reset(1);
        } else if (survivors.size() != active_.size()) {
            compact_qwen_batch_state(
                model_, survivor_rows, survivor_max_position);
            ++compactions_;
        } else {
            model_.cache_pos = survivor_max_position;
        }
        active_ = std::move(survivors);
        active_count_.store(
            static_cast<int64_t>(active_.size()),
            std::memory_order_relaxed);
        for (const auto & completion : completions) {
            complete_request(completion.first, completion.second);
        }
    }

    void worker_main() noexcept {
        for (;;) {
            try {
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    queue_ready_.wait(lock, [&] {
                        return stopping_ || !pending_.empty() ||
                            !active_.empty();
                    });
                    if (stopping_) {
                        auto error = std::make_exception_ptr(
                            std::runtime_error(
                                "continuous batching scheduler stopped"));
                        std::vector<std::shared_ptr<Request>> pending;
                        while (!pending_.empty()) {
                            pending.push_back(std::move(pending_.front()));
                            pending_.pop_front();
                        }
                        queued_.store(0);
                        lock.unlock();
                        fail_requests(pending, error);
                        fail_requests(active_, error);
                        active_.clear();
                        active_count_.store(0);
                        try {
                            std::lock_guard<std::mutex> model_lock(model_mutex_);
                            model_.reset(1);
                        } catch (...) {}
                        return;
                    }
                    if (active_.empty() &&
                            pending_.size() <
                                static_cast<size_t>(max_sequences_)) {
                        queue_ready_.wait_for(
                            lock, initial_batch_wait_, [&] {
                                return stopping_ ||
                                    pending_.size() >=
                                        static_cast<size_t>(max_sequences_);
                            });
                        if (stopping_) continue;
                    }
                }
                auto incoming = take_pending();
                admit_requests(incoming);
                decode_active();
            } catch (...) {
                auto error = std::current_exception();
                fail_requests(active_, error);
                active_.clear();
                active_count_.store(0);
                try {
                    std::lock_guard<std::mutex> model_lock(model_mutex_);
                    model_.reset(1);
                } catch (...) {}
            }
        }
    }

    Model & model_;
    std::mutex & model_mutex_;
    int32_t max_sequences_ = 0;
    std::chrono::microseconds initial_batch_wait_;
    std::thread worker_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_ready_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::vector<std::shared_ptr<Request>> active_;
    bool stopping_ = false;
    std::atomic<int64_t> queued_{0};
    std::atomic<int64_t> active_count_{0};
    std::atomic<int64_t> requests_{0};
    std::atomic<int64_t> decode_batches_{0};
    std::atomic<int64_t> decode_tokens_{0};
    std::atomic<int64_t> max_batch_seen_{0};
    std::atomic<int64_t> admissions_{0};
    std::atomic<int64_t> compactions_{0};
    std::atomic<int64_t> mtp_bypasses_{0};
    std::atomic<int64_t> prefix_cache_bypasses_{0};
};

static int run_qwen_continuous_batching_check(Model & model) {
    const auto incompatibility =
        qwen_continuous_batching_incompatibility(model);
    MFQ_RUNTIME_CHECK(incompatibility.empty(), incompatibility);
    MFQ_RUNTIME_CHECK(model.c.vocab_size > 1024 &&
        model.c.max_position_embeddings >= 208,
        "continuous batching check requires vocab>1024 and context>=208");

    MfqSamplingParams first_params;
    first_params.max_tokens = 12;
    first_params.temperature = 0.0;
    first_params.top_k = 1;
    first_params.top_p = 1.0;
    first_params.enable_mtp = false;
    first_params.seed = 20260907;
    auto second_params = first_params;
    second_params.max_tokens = 4;
    second_params.seed += 1;
    std::vector<int64_t> first_prompt(193);
    std::vector<int64_t> second_prompt(17);
    for (size_t index = 0; index < first_prompt.size(); ++index) {
        first_prompt[index] = 101 +
            static_cast<int64_t>((index * 37) % 900);
    }
    for (size_t index = 0; index < second_prompt.size(); ++index) {
        second_prompt[index] = 113 +
            static_cast<int64_t>((index * 53) % 880);
    }

    auto serial = [&](const std::vector<int64_t> & prompt,
                      const MfqSamplingParams & params) {
        std::vector<int64_t> output;
        std::mutex mutex;
        ServerDecodeGraphCache graph_cache(
            model.c.max_position_embeddings);
        ServerTextSessionCache session_cache;
        const int32_t produced = generate_server_tokens(
            model, mutex, graph_cache, session_cache, prompt, params,
            [&](int64_t token) {
                output.push_back(token);
                return true;
            }, {}, {}, {}, nullptr);
        MFQ_RUNTIME_CHECK(
            produced == params.max_tokens &&
                output.size() == static_cast<size_t>(produced),
            "continuous batching serial oracle length mismatch");
        return output;
    };
    const auto first_reference = serial(first_prompt, first_params);
    const auto second_reference = serial(second_prompt, second_params);
    model.reset(1);

    std::mutex model_mutex;
    CudaContinuousBatcher batcher(
        model, model_mutex, 4, std::chrono::milliseconds(100));
    std::mutex gate_mutex;
    std::condition_variable gate_ready;
    bool first_prefilled = false;
    bool second_delivered = false;
    bool release_first = false;
    std::vector<int64_t> first_output;
    std::vector<int64_t> second_output;
    std::exception_ptr first_error;
    std::exception_ptr second_error;
    int32_t first_produced = 0;
    int32_t second_produced = 0;

    std::thread first_thread([&] {
        try {
            first_produced = batcher.submit(
                first_prompt, first_params,
                [&](int64_t token) {
                    first_output.push_back(token);
                    if (first_output.size() == 1) {
                        std::unique_lock<std::mutex> lock(gate_mutex);
                        first_prefilled = true;
                        gate_ready.notify_one();
                        gate_ready.wait(lock, [&] { return release_first; });
                    }
                    return true;
                }, {}, {}, {});
        } catch (...) {
            first_error = std::current_exception();
        }
    });
    bool first_queued = false;
    for (int attempt = 0; attempt < 50; ++attempt) {
        if (batcher.queued_requests() > 0) {
            first_queued = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::thread second_thread([&] {
        try {
            second_produced = batcher.submit(
                second_prompt, second_params,
                [&](int64_t token) {
                    second_output.push_back(token);
                    if (second_output.size() == 1) {
                        std::lock_guard<std::mutex> lock(gate_mutex);
                        second_delivered = true;
                        gate_ready.notify_one();
                    }
                    return true;
                }, {}, {}, {});
        } catch (...) {
            second_error = std::current_exception();
        }
    });
    bool first_callback_started = false;
    bool callback_isolated = false;
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        first_callback_started = gate_ready.wait_for(
            lock, std::chrono::seconds(10), [&] {
                return first_prefilled;
            });
        callback_isolated = first_callback_started && gate_ready.wait_for(
            lock, std::chrono::seconds(10), [&] {
                return second_delivered;
            });
        release_first = true;
    }
    gate_ready.notify_one();
    first_thread.join();
    second_thread.join();
    if (first_error) std::rethrow_exception(first_error);
    if (second_error) std::rethrow_exception(second_error);
    MFQ_RUNTIME_CHECK(first_queued && first_callback_started &&
        callback_isolated,
        "a blocked response callback stalled the scheduler");
    MFQ_RUNTIME_CHECK(first_produced == first_params.max_tokens &&
        second_produced == second_params.max_tokens,
        "continuous batching generated token count mismatch");
    MFQ_RUNTIME_CHECK(first_output == first_reference,
        "continuous batching first request differs from serial greedy oracle");
    MFQ_RUNTIME_CHECK(second_output == second_reference,
        "continuous batching second request differs from serial greedy oracle");
    auto cancel_params = second_params;
    cancel_params.max_tokens = 12;
    int32_t cancellation_callbacks = 0;
    const int32_t cancellation_produced = batcher.submit(
        second_prompt, cancel_params,
        [&](int64_t) {
            ++cancellation_callbacks;
            return false;
        }, {}, {}, {});
    MFQ_RUNTIME_CHECK(cancellation_produced == 1 &&
        cancellation_callbacks == 1,
        "continuous batching callback cancellation did not stop at one token");
    const auto values = batcher.metrics();
    auto metric = [&](const std::string & name) {
        const auto found = std::find_if(
            values.begin(), values.end(), [&](const auto & item) {
                return item.first == name;
            });
        return found == values.end() ? 0.0 : found->second;
    };
    std::cout << "continuous_batching_check metrics max_batch="
              << metric("continuous_batching_max_batch")
              << " compactions="
              << metric("continuous_batching_compactions")
              << " active="
              << metric("continuous_batching_active")
              << " queued="
              << metric("continuous_batching_queued") << '\n';
    MFQ_RUNTIME_CHECK(metric("continuous_batching_max_batch") >= 2.0 &&
        metric("continuous_batching_compactions") >= 1.0 &&
        metric("continuous_batching_active") == 0.0 &&
        metric("continuous_batching_queued") == 0.0,
        "continuous batching check did not exercise join and retire");
    std::cout << "continuous_batching_check PASS concurrent_requests=2"
              << " cancellation_tokens=1 max_batch="
              << metric("continuous_batching_max_batch")
              << " prompt_lengths=193,17 split_k=1"
              << " decode_batches="
              << metric("continuous_batching_decode_batches")
              << " compactions="
              << metric("continuous_batching_compactions") << '\n';
    return 0;
}

} // namespace mfq::cuda::continuous
