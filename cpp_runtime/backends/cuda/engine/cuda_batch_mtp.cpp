#include "cuda_batching.h"

#include "batched_speculative_sequence.h"
#include "models/qwen35/linear_attention.h"
#include "models/qwen35/ops.h"
#include "runtime/batch_decode_graph.h"

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;
using Tensor = tb::Tensor;

template<class Request>
static BatchMtpInput mtp_input(const std::shared_ptr<Request>& request) {
    return {request.get(), request->mtp.get(), &*request->sampler, request->counts};
}

template<class Request>
static std::optional<mfq::engine::Generation> generate_batch_decode(const std::shared_ptr<Request>& request,
        mfq::engine::InferenceOutput& output, std::function<void()> schedule,
        std::function<bool()> ready, std::function<void()> check) {
    if (!request->mtp) return {};
    output.metrics.mtp = request->mtp->stats;
    return mfq::engine::batched_speculative_sequence(*request, output, request->mtp->first_token,
        std::move(schedule), std::move(ready), std::move(check),
        [request](auto& destination) { return publish_batch_mtp(mtp_input(request), destination); },
        [request](const auto& delta) { accept_batch_mtp_tokens(mtp_input(request), delta); });
}

std::optional<mfq::engine::Generation> QwenBatchOperations::generate_decode(
        const std::shared_ptr<Request>& request, mfq::engine::InferenceOutput& output,
        std::function<void()> schedule, std::function<bool()> ready, std::function<void()> check) {
    return generate_batch_decode(request, output, std::move(schedule), std::move(ready), std::move(check));
}

std::optional<mfq::engine::Generation> FlashNextBatchOperations::generate_decode(
        const std::shared_ptr<Request>& request, mfq::engine::InferenceOutput& output,
        std::function<void()> schedule, std::function<bool()> ready, std::function<void()> check) {
    return generate_batch_decode(request, output, std::move(schedule), std::move(ready), std::move(check));
}

int64_t QwenBatchOperations::cache_length(const std::shared_ptr<Request>& request) const {
    return request->cache_length + !request->mtp;
}

int64_t FlashNextBatchOperations::cache_length(const std::shared_ptr<Request>& request) const {
    return request->mtp ? request->cache_length : request->cache->position();
}

static Tensor verification_mask(const std::vector<int>& widths, int64_t tokens, const Tensor& ids) {
    std::vector<int64_t> enabled;
    for (const auto width : widths)
        for (int64_t token = 0; token < tokens; ++token) enabled.push_back(token < width);
    return tb::tensor(enabled, tb::TensorOptions().dtype(tb::kInt64)).to(ids.device());
}

template<class State>
static std::vector<BatchMtpInput> eligible_mtp_rows(State& state) {
    std::vector<BatchMtpInput> inputs;
    for (const auto& request : state.active) {
        request->decode_row = -1;
        if (!request->eligible) continue;
        request->decode_row = inputs.size();
        inputs.push_back(mtp_input(request));
    }
    return inputs;
}

struct RecurrentPrefix {
    qwen35::LinearAttentionBlock* block;
    qwen35::LinearRecurrentInputs inputs;
    Tensor initial_convolution, initial_recurrent, convolution, recurrent;
    void retain(int count) {
        MfqCudaGuard guard(convolution.get_device());
        auto saved = block->mixer;
        block->mixer.convolution = convolution;
        block->mixer.recurrent = recurrent;
        try {
            convolution.copy_(initial_convolution);
            recurrent.copy_(initial_recurrent);
            block->replay_recurrent_cuda(inputs, count);
            convolution.copy_(block->mixer.convolution);
            recurrent.copy_(block->mixer.recurrent);
            block->mixer = std::move(saved);
        } catch (...) { block->mixer = std::move(saved); throw; }
    }
};

QwenBatchOperations::Decoded QwenBatchOperations::decode_mtp(State& state) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    auto inputs = eligible_mtp_rows(state);
    std::vector<int32_t> slots;
    for (const auto& request : state.active)
        if (request->eligible) slots.push_back(request->cache.slot());
    const auto verify = [&](const Tensor& host, const Tensor& ids, const std::vector<int>& widths) {
        const auto batch = ids.size(0), tokens = ids.size(1);
        BatchDecodeInputs metadata(batch, 2 * model_.blocks.size(), ids.get_device());
        std::vector<BatchDecodeRow> rows;
        int64_t maximum = 0;
        for (std::size_t row = 0; row < inputs.size(); ++row) {
            const auto position = inputs[row].request->cache_length;
            state_adapter_.ensure_decode_tokens(static_cast<Request*>(inputs[row].request)->cache, position + tokens);
            rows.push_back({host.data_ptr<int64_t>()[row * tokens], position, slots[row]});
            maximum = std::max(maximum, position);
        }
        state_adapter_.prepare_decode(maximum);
        metadata.update_rows(rows);
        auto destinations = state_adapter_.recurrent_slot_tensors(slots);
        state_adapter_.bind_decode_rows(metadata, slots, false);
        BatchMtpVerification result;
        std::vector<std::vector<RecurrentPrefix>> prefixes(batch);
        try {
            state_adapter_.load_decode_rows(metadata);
            auto positions = metadata.positions().reshape({batch, 1}) + tb::arange(tokens, ids.options());
            auto lengths = metadata.positions() + tokens;
            BatchDecodeMaskScope mask(execution_, verification_mask(widths, tokens, ids));
            // The scalar causal entry point deliberately requires B==1 for a
            // speculative transaction. Here the row table owns the independent
            // transactions, while each layer still sees the complete window.
            auto hidden = model_.adapter_prepare_hidden(model_.embed_forward(ids), batch, tokens);
            model_.adapter_begin_forward(true);
            for (const auto& block : model_.blocks) {
                MfqCudaGuard layer_guard(block->cuda_device);
                Block::Context context;
                context.token_ids = tensor_to_cuda_device(execution_.model_parallel_collectives,
                    ids, block->cuda_device);
                context.host_token_ids = host;
                context.positions = tensor_to_cuda_device(execution_.model_parallel_collectives,
                    positions, block->cuda_device);
                context.full_positions = context.positions;
                context.cache_positions = context.positions;
                context.sequence_lengths = tensor_to_cuda_device(execution_.model_parallel_collectives,
                    lengths, block->cuda_device);
                context.cache_position = maximum;
                context.confirmed_prefix = tokens > 1 ? 1 : 0;
                block->set_token_ids(context.token_ids);
                hidden = tensor_to_cuda_device(execution_.model_parallel_collectives, hidden, block->cuda_device);
                const auto& rope = model_.device_ropes.empty() ? model_.rope
                    : model_.device_ropes.at(block->cuda_device);
                hidden = block->forward_context(execution_, std::move(hidden), context, rope);
            }
            hidden = tensor_to_cuda_device(execution_.model_parallel_collectives, hidden,
                execution_.layer_placement.primary_device());
            auto finalized = model_.finalize_hidden(hidden, batch, tokens);
            result.raw = model_.adapter_raw_hidden(hidden, finalized);
            hidden = std::move(finalized);
            // Ordinary Qwen decoding rounds the normalized head input to FP16.
            result.logits = model_.logits_from_hidden(hidden.to(tb::kFloat16).contiguous())
                .reshape({batch, tokens, -1});
            std::size_t destination = 0;
            if (tokens > 1) {
                for (const auto& base : model_.blocks) {
                    auto* block = dynamic_cast<qwen35::LinearAttentionBlock*>(base.get());
                    if (!block) continue;
                    const auto& captured = block->speculative_inputs;
                    for (int64_t row = 0; row < batch; ++row) {
                        auto projected = captured;
                        for (auto* tensor : {&projected.qk, &projected.value, &projected.qkv,
                                &projected.gate, &projected.beta})
                            if (tensor->defined()) *tensor = tensor->narrow(0, row, 1);
                        projected.batch = 1;
                        auto convolution = destinations[destination++];
                        auto recurrent = destinations[destination++];
                        // Views keep the window's single snapshot alive until
                        // the row resolves; the next window snapshots afresh.
                        prefixes[row].push_back({block, std::move(projected),
                            block->mixer.speculative_convolution.narrow(0, row, 1),
                            block->mixer.speculative_recurrent.narrow(0, row, 1),
                            std::move(convolution), std::move(recurrent)});
                    }
                }
            }
            state_adapter_.store_decode_rows(metadata);
            if (tokens > 1)
                for (const auto& block : model_.blocks) block->commit_speculative();
            state_adapter_.unbind_decode_rows();
            state_adapter_.finish_decode(maximum + tokens);
        } catch (...) {
            state_adapter_.unbind_decode_rows();
            throw;
        }
        for (int64_t row = 0; row < batch; ++row) {
            result.resolve.push_back([saved = std::move(prefixes[row]), tokens](int retained) mutable {
                MFQ_RUNTIME_CHECK(retained > 0 && retained <= tokens, "invalid batched recurrent prefix");
                if (retained < tokens)
                    for (auto& layer : saved) layer.retain(retained);
            });
        }
        return result;
    };
    // A speculative window snapshots every row of the batch. Rows beyond the
    // fixed checkpoint budget decode one token instead of growing that snapshot.
    const auto draft_tokens = int64_t(inputs.size()) <= mtp_checkpoints_ ? -1 : 0;
    auto tokens = run_batch_mtp(*predictor_, MtpTarget(model_), inputs, max_context(), verify, draft_tokens);
    ++mtp_batches_;
    ++decode_batches_;
    decode_tokens_ += inputs.size();
    for (const auto& input : inputs) if (input.mtp) {
        mtp_drafted_ += input.mtp->drafted;
        mtp_accepted_ += input.mtp->accepted;
    }
    auto host = tb::tensor(tokens, tb::TensorOptions().dtype(tb::kInt64));
    return {{}, {}, host.to(tb::Device(tb::kCUDA, execution_.layer_placement.primary_device())), host};
}

FlashNextBatchOperations::Decoded FlashNextBatchOperations::decode_mtp(State& state) {
    auto inputs = eligible_mtp_rows(state);
    std::vector<Qwen4BatchRequestState*> caches;
    for (const auto& request : state.active) if (request->eligible) caches.push_back(request->cache.get());
    const auto verify = [&](const Tensor& host, const Tensor& ids, const std::vector<int>& widths) {
        BatchDecodeMaskScope mask(execution_, verification_mask(widths, ids.size(1), ids));
        BatchMtpVerification result;
        result.logits = qwen4_batch_decode(model_, caches, ids, host, &result.raw, widths, true, mtp_checkpoints_)
            .reshape({ids.size(0), ids.size(1), -1});
        for (auto* cache : caches)
            result.resolve.push_back([cache](int retained) { cache->resolve_verified(retained); });
        return result;
    };
    auto tokens = run_batch_mtp(*predictor_, MtpTarget(model_), inputs, max_context(), verify,
                                mtp_checkpoints_ ? mtp_checkpoints_->available() : -1);
    ++mtp_batches_;
    ++decode_batches_;
    decode_tokens_ += inputs.size();
    for (const auto& input : inputs) if (input.mtp) {
        mtp_drafted_ += input.mtp->drafted;
        mtp_accepted_ += input.mtp->accepted;
    }
    auto host = tb::tensor(tokens, tb::TensorOptions().dtype(tb::kInt64));
    return {{}, {}, host.to(tb::Device(tb::kCUDA, execution_.layer_placement.primary_device())), host};
}

} // namespace mfq::cuda
