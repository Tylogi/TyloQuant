#include "cuda_batching.h"

#include "core/full_block.h"
#include "models/qwen35/linear_attention.h"
#include "models/qwen35/ops.h"
#include "models/qwen4_exp/model.h"
#include "runtime/batch_decode_graph.h"
#include "runtime/kv_vmm.h"
#include "storage/moe_expert_cache.h"

namespace mfq::cuda {
namespace {
using mfq::engine::KvAdmissionBytes;
using mfq::engine::kv_admission_multiply;
using mfq::engine::prefill_budget_add;
template<class Model> std::size_t workspace_headroom(const Model& model, int device) {
    const auto rows = mfq::engine::prefill_workspace_row_bytes(
        model.hidden_size(), model.hc_mult(), model.num_attention_heads(), model.head_dim());
    return mfq::engine::prefill_workspace_headroom(default_context(device)->memory_stats().limit, rows);
}

template<class Model>
KvAdmissionBytes available_bytes(const KvAdmissionBytes& geometry, const BatchMemoryProvider& memory,
                                  std::size_t transfer_bytes, const Model& model) {
    const auto evictable = memory.evictable ? memory.evictable() : KvAdmissionBytes{};
    KvAdmissionBytes result;
    for (const auto& [device, width] : geometry) {
        MfqCudaGuard guard(device);
        const auto context = default_context(device);
        auto bytes = context->available_memory();
        if (const auto it = evictable.find(device); it != evictable.end())
            bytes = prefill_budget_add(bytes, it->second);
        const auto headroom = prefill_budget_add(workspace_headroom(model, device), transfer_bytes);
        result[device] = bytes > headroom ? bytes - headroom : 0;
    }
    return result;
}

KvAdmissionBytes token_geometry(const Qwen35CausalLm& model) {
    KvAdmissionBytes result;
    for (const auto& base : model.blocks) {
        result.try_emplace(base->cuda_device, 0);
        const auto* block = dynamic_cast<const FullBlock*>(base.get());
        if (!block) continue;
        const auto heads = block->kv_heads > 0 ? block->kv_heads : model.num_key_value_heads();
        const auto width = block->attention_head_dim > 0 ? block->attention_head_dim : model.head_dim();
        const auto quantization = block->kv_quantization;
        const auto row = quantization.enabled()
            ? (cuda_kv_packed_width(width, quantization.key_bits()) +
               cuda_kv_packed_width(width, quantization.value_bits())) * sizeof(int32_t)
            : 2 * width * sizeof(uint16_t);
        result[base->cuda_device] = prefill_budget_add(result[base->cuda_device], heads * row);
    }
    return result;
}

KvAdmissionBytes recurrent_geometry(const Qwen35CausalLm& model) {
    KvAdmissionBytes result;
    for (const auto& base : model.blocks) {
        const auto* block = dynamic_cast<const qwen35::LinearAttentionBlock*>(base.get());
        if (!block) continue;
        const auto& c = block->qwen_config;
        const auto elements = std::size_t((c.linear_conv_kernel_dim - 1) * (2 * c.linear_k_size() +
            c.linear_v_size()) + c.linear_num_value_heads * c.linear_value_head_dim * c.linear_value_head_dim);
        result[base->cuda_device] = prefill_budget_add(result[base->cuda_device], elements * sizeof(float));
    }
    return result;
}

std::size_t request_fixed_bytes(const mfq::models::qwen4_exp::Config& c) {
    const auto linear = c.layers - c.layers / c.interval;
    const auto convolution = (c.kernel - 1) * (2 * c.key_heads + c.value_heads) * c.linear_width;
    const auto recurrence = c.value_heads * c.linear_width * c.linear_width;
    const auto counters = c.value_heads * (c.linear_width + 1);
    const auto ple = c.ple_layers.size() * (c.ple_kernel - 1) * c.hidden * sizeof(float);
    return (convolution + recurrence + counters) * linear * sizeof(float) + ple;
}

std::size_t token_geometry(const Qwen4CausalLm& model) {
    const auto& c = model.config;
    const auto quantization = cuda_kv_quantization();
    const auto row = quantization.enabled()
        ? (cuda_kv_packed_width(c.width, quantization.key_bits()) +
           cuda_kv_packed_width(c.width, quantization.value_bits())) * sizeof(int32_t)
        : 2 * c.width * sizeof(uint16_t);
    // Include the FP16 index and pooled index, plus logical position storage.
    return (c.layers / c.interval) * (c.kv_heads * row + 2 * c.index_width * sizeof(uint16_t)) + sizeof(int64_t);
}
}

mfq::engine::KvAdmissionBytes QwenBatchOperations::admission_available() const {
    const auto geometry = token_geometry(model_);
    const auto transfer = moe_prefill_workspace_budget(execution_.moe_expert_cache, false).transfer_bytes;
    auto available = available_bytes(geometry, memory_, transfer, model_);
    const auto paged = state_adapter_.paged_kv_stats();
    const auto recurrent = recurrent_geometry(model_);
    for (const auto& [device, width] : geometry) {
        const auto reusable = kv_admission_multiply((paged.capacity_pages - paged.live_pages) * paged.page_size, width);
        available[device] = prefill_budget_add(available[device], reusable);
        const auto it = recurrent.find(device);
        const auto fixed = it == recurrent.end() ? 0 : it->second;
        auto slots = kv_admission_multiply(fixed, config_.max_sequences);
        if (!state_adapter_.paged_kv_enabled())
            slots = prefill_budget_add(slots, kv_admission_multiply(
                kv_admission_multiply(model_.max_position_embeddings(), width), config_.max_sequences));
        const auto pending_slots = admission_storage_ready_ ? 0 : slots;
        auto pending = pending_slots;
        if (state_adapter_.paged_kv_enabled())
            pending = prefill_budget_add(pending, kv_admission_multiply(1024 - 16, width));
        available[device] = available[device] > pending ? available[device] - pending : 0;
        if (config_.kv_budget_bytes) {
            auto used = kv_admission_multiply(paged.live_pages * paged.page_size, width);
            used = prefill_budget_add(used, slots);
            for (const auto& weak : requests_) {
                const auto request = weak.lock();
                if (!request || request->cache.slot() >= 0 || request->prefill_offset <= 0) continue;
                used = prefill_budget_add(used, fixed);
                if (!state_adapter_.paged_kv_enabled())
                    used = prefill_budget_add(used, kv_admission_multiply(model_.max_position_embeddings(), width));
            }
            available[device] = std::min(available[device],
                config_.kv_budget_bytes > used ? config_.kv_budget_bytes - used : 0);
        }
    }
    return available;
}

mfq::engine::KvAdmissionBytes QwenBatchOperations::admission_cost(
        const std::shared_ptr<Request>& request, int64_t expected) const {
    auto result = token_geometry(model_);
    const auto prefill = request ? std::max<int64_t>(0, int64_t(request->prompt.size()) - request->prefill_offset) : 0;
    const auto tokens = std::min<int64_t>(prefill + expected, model_.max_position_embeddings());
    const auto page = state_adapter_.paged_kv_enabled() ? int64_t(16) : int64_t(1);
    const auto rounded = mfq::engine::kv_admission_round_up(tokens, page);
    for (auto& [device, bytes] : result) {
        const auto allocation = state_adapter_.paged_kv_enabled() ? prefill_budget_add(rounded, page)
            : std::size_t(!request || request->prefill_offset == 0 ? model_.max_position_embeddings() : 0);
        bytes = kv_admission_multiply(allocation, bytes);
    }
    if (request && request->prefill_offset == 0) {
        for (const auto& [device, bytes] : recurrent_geometry(model_))
            result[device] = prefill_budget_add(result[device], bytes);
    }
    return result;
}

void QwenBatchOperations::reclaim_prefill_memory(const std::shared_ptr<Request>& request) {
    if (!memory_.reclaim) return;
    auto required = admission_cost(request, 0);
    const auto recurrent = recurrent_geometry(model_);
    const auto transfer = moe_prefill_workspace_budget(execution_.moe_expert_cache, false).transfer_bytes;
    for (const auto& [device, width] : token_geometry(model_)) {
        auto& bytes = required[device];
        if (!admission_storage_ready_) {
            const auto it = recurrent.find(device);
            const auto fixed = it == recurrent.end() ? 0 : it->second;
            bytes = prefill_budget_add(bytes, kv_admission_multiply(fixed, config_.max_sequences));
            if (!state_adapter_.paged_kv_enabled())
                bytes = prefill_budget_add(bytes, kv_admission_multiply(
                    kv_admission_multiply(model_.max_position_embeddings(), width), config_.max_sequences));
        }
        if (state_adapter_.paged_kv_enabled())
            bytes = prefill_budget_add(bytes, kv_admission_multiply(1024 - 16, width));
        bytes = prefill_budget_add(bytes, prefill_budget_add(workspace_headroom(model_, device), transfer));
    }
    memory_.reclaim(required);
}

mfq::engine::KvAdmissionBytes FlashNextBatchOperations::admission_available() const {
    const auto device = execution_.layer_placement.primary_device();
    const auto transfer = moe_prefill_workspace_budget(execution_.moe_expert_cache, false).transfer_bytes;
    auto available = available_bytes({{device, token_geometry(model_)}}, memory_, transfer, model_);
    if (config_.kv_budget_bytes) {
        std::size_t used = 0;
        for (const auto& weak : requests_) {
            const auto request = weak.lock();
            if (request && request->cache) used = prefill_budget_add(used, request->cache->resident_bytes());
        }
        const auto managed = managed_kv_bytes();
        if (managed > admission_base_kv_bytes_)
            used = prefill_budget_add(used, managed - admission_base_kv_bytes_);
        available[device] = std::min(available[device],
            config_.kv_budget_bytes > used ? config_.kv_budget_bytes - used : 0);
    }
    return available;
}

std::size_t FlashNextBatchOperations::managed_kv_bytes() const {
    if (execution_.kv_memory)
        for (const auto& [name, value] : execution_.kv_memory->metrics())
            if (name == "kv_memory_gpu_bytes") return std::size_t(value);
    return 0;
}

mfq::engine::KvAdmissionBytes FlashNextBatchOperations::admission_cost(
        const std::shared_ptr<Request>& request, int64_t expected) const {
    const auto device = execution_.layer_placement.primary_device();
    const auto offset = request->replay_offset >= 0 ? request->replay_offset : request->prefill_offset;
    const auto length = request->replay_offset >= 0 ? request->replay_length() : int64_t(request->prompt.size());
    const auto position = request->cache ? request->cache->position() : 0;
    const auto tokens = std::min(std::max<int64_t>(0, length - offset) + expected,
                                 model_.max_position_embeddings() - position);
    if (tokens <= 0) return {{device, 0}};
    if (request->cache)
        return {{device, prefill_budget_add(request->cache->prefill_growth_bytes(tokens),
                                            kv_admission_multiply(tokens, sizeof(int64_t)))}};
    if (execution_.kv_memory && !execution_.qsa_kv_store) {
        const auto& c = model_.config;
        const auto layers = c.layers / c.interval;
        const auto quantization = cuda_kv_quantization();
        const auto key = c.kv_heads * (quantization.enabled()
            ? cuda_kv_packed_width(c.width, quantization.key_bits()) * sizeof(int32_t)
            : c.width * sizeof(uint16_t));
        const auto value = c.kv_heads * (quantization.enabled()
            ? cuda_kv_packed_width(c.width, quantization.value_bits()) * sizeof(int32_t)
            : c.width * sizeof(uint16_t));
        std::size_t raw = 0;
        if (execution_.kv_memory->dynamic()) {
            MfqCudaGuard guard(device);
            const auto quantum = std::max<std::size_t>(1, kv_vmm::vmm_granularity());
            raw = prefill_budget_add(mfq::engine::kv_admission_mapping_bytes(
                c.kv_heads, c.maximum, tokens, key / c.kv_heads, quantum),
                mfq::engine::kv_admission_mapping_bytes(
                    c.kv_heads, c.maximum, tokens, value / c.kv_heads, quantum));
        } else {
            // Respect explicit preallocation: the allocator needs the whole
            // physical cache up front even when expected decode is shorter.
            raw = kv_admission_multiply(c.maximum, key + value);
        }
        const auto index = std::size_t(c.maximum + (c.maximum + c.pool - 1) / c.pool) * c.index_width * 2;
        auto layer_bytes = prefill_budget_add(raw, index);
        if (const auto placed = execution_.kv_memory->additional_preallocated_gpu_bytes(
                c.maximum, c.kv_heads, key / c.kv_heads, value / c.kv_heads, c.index_width, c.pool))
            layer_bytes = *placed;
        const auto cache = kv_admission_multiply(layers, layer_bytes);
        return {{device, prefill_budget_add(cache, prefill_budget_add(request_fixed_bytes(c),
            kv_admission_multiply(tokens, sizeof(int64_t))))}};
    }
    auto capacity = std::min<int64_t>(16, model_.max_position_embeddings());
    while (capacity < tokens)
        capacity = capacity > model_.max_position_embeddings() / 2 ? model_.max_position_embeddings() : capacity * 2;
    return {{device, prefill_budget_add(kv_admission_multiply(capacity, token_geometry(model_)),
                                        request_fixed_bytes(model_.config))}};
}

void FlashNextBatchOperations::reclaim_prefill_memory(const std::shared_ptr<Request>& request) {
    if (!memory_.reclaim) return;
    auto required = admission_cost(request, 0);
    const auto transfer = moe_prefill_workspace_budget(execution_.moe_expert_cache, false).transfer_bytes;
    for (auto& [device, bytes] : required)
        bytes = prefill_budget_add(bytes, prefill_budget_add(workspace_headroom(model_, device), transfer));
    memory_.reclaim(required);
}

bool FlashNextBatchOperations::decode_has_capacity(const State& state) const {
    const auto device = execution_.layer_placement.primary_device();
    std::size_t needed = 0;
    int64_t required = 0;
    int64_t window = 1, headroom = max_context();
    // Verification pads every participating row to the largest selected draft
    // width. Charge that transient width before running the target model.
    for (const auto& request : state.active) {
        if (!request->eligible) continue;
        headroom = std::min(headroom, max_context() - request->cache_length);
        if (request->mtp)
            window = std::max(window, 1 + std::max<int64_t>(0, std::min<int64_t>(
                request->mtp->depth->depth(), request->generation_limit - request->produced - 1)));
    }
    window = std::max<int64_t>(1, std::min(window, headroom));
    for (const auto& request : state.active) {
        if (!request->eligible) continue;
        needed = prefill_budget_add(needed, request->cache->prefill_growth_bytes(window));
        required = std::max(required, request->cache->position() + window);
    }
    if (graph_ && execution_.kv_memory)
        needed = std::max(needed, execution_.kv_memory->growth_bytes(required));
    if (!needed) return true;
    mfq::engine::KvAdmissionBudget budget(admission_available());
    return budget.reserve({{device, needed}});
}

void FlashNextBatchOperations::retract(const std::shared_ptr<Request>& request) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    // Graphs retain scratch and state bindings; retire them before freeing KV.
    if (graph_) graph_->clear();
    mfq_cuda_synchronize();
    auto mtp = std::move(request->mtp);
    if (mtp) {
        mtp->discard();
        mtp->predictor.reset();
        mtp->history_hidden = {};
        mtp->history_ids.clear();
        mtp->candidates.clear();
    }
    discard_prefill(request);
    request->mtp = std::move(mtp);
    default_context(execution_.layer_placement.primary_device())->reclaim_idle_memory();
}

void FlashNextBatchOperations::replay_prefill(
        const std::shared_ptr<Request>& request, mfq::engine::PrefillChunk chunk) {
    MfqCudaGuard guard(execution_.layer_placement.primary_device());
    reclaim_prefill_memory(request);
    if (!request->cache) {
        request->cache = std::make_unique<Qwen4BatchRequestState>(model_);
        // requests_ already tracks this request. Keep sampler/RNG and counts.
        if (request->mtp) {
            auto replay = request->prompt;
            const auto generated = request->replay_length() - int64_t(replay.size());
            MFQ_RUNTIME_CHECK(generated >= 0 && generated <= int64_t(request->accepted_tokens.size()),
                              "MTP replay is missing accepted tokens");
            replay.insert(replay.end(), request->accepted_tokens.begin(), request->accepted_tokens.begin() + generated);
            request->prefill_ids = mfq_tensor_backend::tensor(replay, mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA,
                    execution_.layer_placement.primary_device()))).reshape({1, -1});
        }
    }
    std::vector<int64_t> tokens;
    tokens.reserve(chunk.count);
    for (auto index = chunk.offset; index < chunk.offset + chunk.count; ++index)
        tokens.push_back(index < int64_t(request->prompt.size()) ? request->prompt[index]
            : request->accepted_tokens[index - request->prompt.size()]);
    auto ids = mfq_tensor_backend::tensor(tokens, mfq_tensor_backend::TensorOptions()
        .dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA,
            execution_.layer_placement.primary_device()))).reshape({1, -1});
    Tensor raw;
    (void)request->cache->prefill(model_, std::move(ids), request->mtp ? &raw : nullptr);
    if (request->mtp) {
        prime_batch_mtp(*predictor_, MtpTarget(model_), *request->mtp, std::move(raw),
            request->prefill_ids, chunk.offset, request->replay_length());
        if (chunk.offset + chunk.count == request->replay_length()) {
            request->mtp->first_token = request->pending_token;
            request->mtp->history_ids = {request->pending_token};
            request->prefill_ids = {};
        }
    }
}
} // namespace mfq::cuda
