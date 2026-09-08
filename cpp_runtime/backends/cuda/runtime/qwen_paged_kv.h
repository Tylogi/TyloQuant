#pragma once

#include "paged_kv_allocator.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::cuda::continuous {

struct QwenPagedKvSequence {
    std::vector<std::int32_t> physical_pages;
};

class QwenPagedKvArena {
public:
    static constexpr int64_t kPageSize = 16;
    static constexpr int64_t kPagesPerChunk = 64;

    QwenPagedKvArena(Model & model, int32_t maximum_sequences)
        : maximum_sequences_(maximum_sequences),
          logical_pages_per_sequence_(
              (model.c.max_position_embeddings + kPageSize - 1) /
              kPageSize),
          maximum_physical_pages_(checked_maximum_pages(
              maximum_sequences_, logical_pages_per_sequence_)),
          allocator_(maximum_physical_pages_),
          host_page_table_(
              static_cast<size_t>(maximum_sequences_) *
                  static_cast<size_t>(logical_pages_per_sequence_),
              -1) {
        MFQ_RUNTIME_CHECK(maximum_sequences_ > 0 &&
            logical_pages_per_sequence_ > 0,
            "Paged KV requires positive sequence and context limits");
        const int64_t maximum_chunks =
            (maximum_physical_pages_ + kPagesPerChunk - 1) /
            kPagesPerChunk;
        for (auto & block_value : model.blocks) {
            auto * block = dynamic_cast<FullBlock *>(block_value.get());
            if (block == nullptr) continue;
            const int64_t heads = block->kv_heads > 0
                ? block->kv_heads : model.c.num_key_value_heads;
            const int64_t head_dim = block->attention_head_dim > 0
                ? block->attention_head_dim : model.c.head_dim;
            MFQ_RUNTIME_CHECK(heads > 0 && head_dim > 0,
                "Paged KV encountered invalid full-attention geometry");
            const int device = block->cuda_device;
            MfqCudaGuard guard(device);
            auto table = device_page_tables_.find(device);
            if (table == device_page_tables_.end()) {
                auto page_table = mfq_tensor_backend::empty(
                    {maximum_sequences_, logical_pages_per_sequence_},
                    mfq_tensor_backend::TensorOptions()
                        .device(mfq_tensor_backend::Device(
                            mfq_tensor_backend::kCUDA, device))
                        .dtype(mfq_tensor_backend::kInt32));
                page_table.fill_(-1);
                table = device_page_tables_.emplace(
                    device, std::move(page_table)).first;
            }
            auto pointer_options = mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA, device))
                .dtype(mfq_tensor_backend::kInt64);
            LayerPool pool;
            pool.block = block;
            pool.device = device;
            pool.heads = heads;
            pool.head_dim = head_dim;
            pool.dtype = mfq_tensor_backend::kFloat16;
            pool.k_chunk_ptrs = mfq_tensor_backend::zeros(
                {maximum_chunks}, pointer_options);
            pool.v_chunk_ptrs = mfq_tensor_backend::zeros(
                {maximum_chunks}, pointer_options);
            layers_.push_back(std::move(pool));
        }
        MFQ_RUNTIME_CHECK(!layers_.empty(),
            "Paged KV requires at least one full-attention layer");
    }

    bool ensure_tokens(
            QwenPagedKvSequence & sequence, int64_t token_count) {
        MFQ_RUNTIME_CHECK(token_count >= 0 &&
            token_count <= logical_pages_per_sequence_ * kPageSize,
            "Paged KV sequence length exceeds the configured context");
        const size_t required_pages = static_cast<size_t>(
            (token_count + kPageSize - 1) / kPageSize);
        if (required_pages <= sequence.physical_pages.size()) return false;
        const size_t previous_size = sequence.physical_pages.size();
        auto pages = allocator_.allocate(required_pages - previous_size);
        sequence.physical_pages.insert(
            sequence.physical_pages.end(), pages.begin(), pages.end());
        try {
            ensure_storage(allocator_.high_watermark());
        } catch (...) {
            sequence.physical_pages.resize(previous_size);
            allocator_.release(pages);
            publish_allocator_metrics();
            throw;
        }
        publish_allocator_metrics();
        return true;
    }

    void release(QwenPagedKvSequence & sequence) {
        if (sequence.physical_pages.empty()) return;
        allocator_.release(sequence.physical_pages);
        sequence.physical_pages.clear();
        publish_allocator_metrics();
    }

    void bind(const std::vector<const QwenPagedKvSequence *> & sequences) {
        MFQ_RUNTIME_CHECK(!sequences.empty() &&
            sequences.size() <= static_cast<size_t>(maximum_sequences_),
            "Paged KV cannot bind the requested batch size");
        const size_t rows = sequences.size();
        const size_t row_width =
            static_cast<size_t>(logical_pages_per_sequence_);
        std::fill(
            host_page_table_.begin(),
            host_page_table_.begin() + rows * row_width, -1);
        for (size_t row = 0; row < rows; ++row) {
            const auto * sequence = sequences[row];
            MFQ_RUNTIME_CHECK(sequence != nullptr &&
                sequence->physical_pages.size() <= row_width,
                "Paged KV received an invalid sequence page list");
            for (size_t logical_page = 0;
                    logical_page < sequence->physical_pages.size();
                    ++logical_page) {
                const auto physical =
                    sequence->physical_pages[logical_page];
                MFQ_RUNTIME_CHECK(allocator_.owns(physical),
                    "Paged KV page table references an unowned page");
                host_page_table_[row * row_width + logical_page] = physical;
            }
        }
        const size_t bytes = rows * row_width * sizeof(std::int32_t);
        for (auto & [device, table] : device_page_tables_) {
            MfqCudaGuard guard(device);
            MFQ_CUDA_CHECK(cudaMemcpy(
                table.data_ptr<std::int32_t>(), host_page_table_.data(),
                bytes, cudaMemcpyHostToDevice));
        }
        const int64_t batch = static_cast<int64_t>(rows);
        for (auto & layer : layers_) {
            layer.block->cache = KVCache::paged_view(
                layer.k_chunk_ptrs, layer.v_chunk_ptrs,
                device_page_tables_.at(layer.device), batch,
                layer.heads, layer.head_dim, kPageSize,
                kPagesPerChunk, layer.dtype);
        }
        page_table_updates_metric_.fetch_add(1, std::memory_order_relaxed);
    }

    void detach() {
        for (auto & layer : layers_) layer.block->cache = KVCache();
    }

    int64_t page_size() const noexcept { return kPageSize; }
    size_t live_pages() const noexcept {
        return live_pages_metric_.load(std::memory_order_relaxed);
    }
    size_t peak_live_pages() const noexcept {
        return peak_live_pages_metric_.load(std::memory_order_relaxed);
    }
    size_t capacity_pages() const noexcept {
        return capacity_pages_metric_.load(std::memory_order_relaxed);
    }
    size_t allocation_count() const noexcept {
        return allocation_count_metric_.load(std::memory_order_relaxed);
    }
    size_t reuse_count() const noexcept {
        return reuse_count_metric_.load(std::memory_order_relaxed);
    }
    size_t release_count() const noexcept {
        return release_count_metric_.load(std::memory_order_relaxed);
    }
    size_t page_table_updates() const noexcept {
        return page_table_updates_metric_.load(std::memory_order_relaxed);
    }
    size_t reserved_bytes() const noexcept {
        return reserved_bytes_metric_.load(std::memory_order_relaxed);
    }

private:
    struct LayerPool {
        FullBlock * block = nullptr;
        int device = 0;
        int64_t heads = 0;
        int64_t head_dim = 0;
        mfq_tensor_backend::ScalarType dtype = mfq_tensor_backend::kFloat16;
        mfq_tensor_backend::Tensor k_chunk_ptrs;
        mfq_tensor_backend::Tensor v_chunk_ptrs;
        std::vector<mfq_tensor_backend::Tensor> k_chunks;
        std::vector<mfq_tensor_backend::Tensor> v_chunks;
    };

    struct PendingChunk {
        mfq_tensor_backend::Tensor k;
        mfq_tensor_backend::Tensor v;
    };

    static size_t checked_maximum_pages(
            int32_t maximum_sequences, int64_t pages_per_sequence) {
        if (maximum_sequences <= 0 || pages_per_sequence <= 0 ||
                static_cast<uint64_t>(maximum_sequences) >
                    static_cast<uint64_t>(
                        std::numeric_limits<int32_t>::max()) /
                    static_cast<uint64_t>(pages_per_sequence)) {
            throw std::invalid_argument(
                "Paged KV maximum physical page count exceeds int32");
        }
        return static_cast<size_t>(maximum_sequences) *
            static_cast<size_t>(pages_per_sequence);
    }

    void publish_allocator_metrics() noexcept {
        live_pages_metric_.store(
            allocator_.live_pages(), std::memory_order_relaxed);
        peak_live_pages_metric_.store(
            allocator_.peak_live_pages(), std::memory_order_relaxed);
        allocation_count_metric_.store(
            allocator_.allocation_count(), std::memory_order_relaxed);
        reuse_count_metric_.store(
            allocator_.reuse_count(), std::memory_order_relaxed);
        release_count_metric_.store(
            allocator_.release_count(), std::memory_order_relaxed);
    }

    void ensure_storage(size_t required_pages) {
        while (storage_pages_ < required_pages) {
            const size_t chunk_pages = std::min<size_t>(
                kPagesPerChunk, maximum_physical_pages_ - storage_pages_);
            MFQ_RUNTIME_CHECK(chunk_pages > 0,
                "Paged KV physical storage exhausted");
            std::vector<PendingChunk> pending;
            pending.reserve(layers_.size());
            for (const auto & layer : layers_) {
                MfqCudaGuard guard(layer.device);
                auto options = mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::Device(
                        mfq_tensor_backend::kCUDA, layer.device))
                    .dtype(layer.dtype);
                PendingChunk chunk;
                chunk.k = mfq_tensor_backend::empty(
                    {static_cast<int64_t>(chunk_pages), layer.heads,
                     kPageSize, layer.head_dim}, options);
                chunk.v = mfq_tensor_backend::empty(
                    {static_cast<int64_t>(chunk_pages), layer.heads,
                     kPageSize, layer.head_dim}, options);
                pending.push_back(std::move(chunk));
            }
            const size_t chunk_index = storage_pages_ / kPagesPerChunk;
            for (size_t index = 0; index < layers_.size(); ++index) {
                auto & layer = layers_[index];
                auto & chunk = pending[index];
                const int64_t k_pointer = reinterpret_cast<int64_t>(
                    chunk.k.data_ptr());
                const int64_t v_pointer = reinterpret_cast<int64_t>(
                    chunk.v.data_ptr());
                MfqCudaGuard guard(layer.device);
                MFQ_CUDA_CHECK(cudaMemcpy(
                    layer.k_chunk_ptrs.data_ptr<int64_t>() + chunk_index,
                    &k_pointer, sizeof(k_pointer), cudaMemcpyHostToDevice));
                MFQ_CUDA_CHECK(cudaMemcpy(
                    layer.v_chunk_ptrs.data_ptr<int64_t>() + chunk_index,
                    &v_pointer, sizeof(v_pointer), cudaMemcpyHostToDevice));
            }
            for (size_t index = 0; index < layers_.size(); ++index) {
                reserved_bytes_ += pending[index].k.nbytes() +
                    pending[index].v.nbytes();
                layers_[index].k_chunks.push_back(
                    std::move(pending[index].k));
                layers_[index].v_chunks.push_back(
                    std::move(pending[index].v));
            }
            storage_pages_ += chunk_pages;
            capacity_pages_metric_.store(
                storage_pages_, std::memory_order_relaxed);
            reserved_bytes_metric_.store(
                reserved_bytes_, std::memory_order_relaxed);
        }
    }

    int32_t maximum_sequences_ = 0;
    int64_t logical_pages_per_sequence_ = 0;
    size_t maximum_physical_pages_ = 0;
    PagedKvPageAllocator allocator_;
    std::vector<std::int32_t> host_page_table_;
    std::map<int, mfq_tensor_backend::Tensor> device_page_tables_;
    std::vector<LayerPool> layers_;
    size_t storage_pages_ = 0;
    size_t reserved_bytes_ = 0;
    std::atomic<size_t> live_pages_metric_{0};
    std::atomic<size_t> peak_live_pages_metric_{0};
    std::atomic<size_t> capacity_pages_metric_{0};
    std::atomic<size_t> reserved_bytes_metric_{0};
    std::atomic<size_t> allocation_count_metric_{0};
    std::atomic<size_t> reuse_count_metric_{0};
    std::atomic<size_t> release_count_metric_{0};
    std::atomic<size_t> page_table_updates_metric_{0};
};

} // namespace mfq::cuda::continuous
