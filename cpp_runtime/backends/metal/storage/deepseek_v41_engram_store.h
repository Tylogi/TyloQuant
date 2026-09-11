#pragma once

#include "hf_safetensors_store.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace mfq::metal {

struct DeepseekV41EngramHashLayer {
    std::int32_t layer_id = 0;
    std::uint64_t table_rows = 0;
    std::vector<std::uint64_t> multipliers;
    std::vector<std::uint32_t> primes;
    std::vector<std::uint32_t> offsets;
};

// Stateful, CPU-side implementation of the released V4.1 n-gram hash.  The
// tokenizer-normalization map and NumPy-generated multipliers are loaded from
// the deterministic D41T asset produced by the server.  Keeping this state on
// CPU avoids copying the 129k-entry token map through Metal for every decode
// token and gives the SSD row loader concrete indices before graph execution.
class DeepseekV41EngramHashState {
public:
    static DeepseekV41EngramHashState load(
        const std::filesystem::path& asset,
        std::uint32_t expected_vocab_size,
        std::uint32_t expected_compressed_vocab_size,
        std::uint32_t expected_max_ngram_size,
        std::uint32_t expected_n_heads,
        std::int32_t pad_token_id,
        std::span<const std::int64_t> expected_layer_ids,
        std::span<const std::int64_t> expected_table_rows);

    // input_ids and token_mask are row-major [batch,tokens]. A zero mask or
    // an out-of-vocabulary sentinel marks an image/dead token. Returned rows
    // are [engram_layer][batch*tokens*n_hash_columns].
    std::vector<std::vector<std::uint64_t>> update(
        std::span<const std::int32_t> input_ids,
        int batch,
        int tokens,
        int start_position,
        std::span<const std::uint8_t> token_mask = {});

    void reset() noexcept;
    void truncate(int position);
    int batch() const noexcept;
    int position() const noexcept;
    std::size_t n_hash_columns() const noexcept;
    const std::vector<DeepseekV41EngramHashLayer>& layers() const noexcept;

private:
    DeepseekV41EngramHashState(
        std::vector<std::uint32_t> token_map,
        std::uint32_t compressed_vocab_size,
        std::uint32_t max_ngram_size,
        std::uint32_t n_heads,
        std::uint32_t pad_id,
        std::vector<DeepseekV41EngramHashLayer> layers);

    std::vector<std::uint32_t> token_map_;
    std::uint32_t compressed_vocab_size_ = 0;
    std::uint32_t max_ngram_size_ = 0;
    std::uint32_t n_heads_ = 0;
    std::uint32_t pad_id_ = 0;
    std::vector<DeepseekV41EngramHashLayer> layers_;
    std::vector<std::vector<std::int64_t>> history_;
    int batch_ = 0;
    int position_ = 0;
};

struct DeepseekV41EngramSsdStats {
    std::uint64_t row_requests = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
    std::uint64_t rows_loaded = 0;
    std::uint64_t bytes_read = 0;
    std::uint64_t read_calls = 0;
    double io_seconds = 0.0;
    std::size_t resident_rows = 0;
    std::size_t resident_payload_bytes = 0;
    std::size_t cache_limit_bytes = 0;

    double hit_rate() const noexcept {
        return row_requests == 0
            ? 0.0
            : static_cast<double>(cache_hits) /
                static_cast<double>(row_requests);
    }
};

// Bounded sparse-row cache over the official Engram FP8 Safetensors. The two
// ~100 GB tables remain on SSD. A lookup issues exact pread() calls only for
// cold rows, caches their native 256-byte FP8 + 8-byte scale payloads, and
// returns dequantized BF16 bits in request order.
class DeepseekV41EngramSsdStore {
public:
    DeepseekV41EngramSsdStore(
        std::shared_ptr<HfSafetensorStore> checkpoint,
        std::span<const std::int64_t> layer_ids,
        std::span<const std::int64_t> table_rows,
        std::size_t head_dim,
        std::size_t cache_bytes,
        std::size_t io_workers = 8);
    ~DeepseekV41EngramSsdStore();

    DeepseekV41EngramSsdStore(
        const DeepseekV41EngramSsdStore&) = delete;
    DeepseekV41EngramSsdStore& operator=(
        const DeepseekV41EngramSsdStore&) = delete;

    std::vector<std::uint16_t> lookup_bf16(
        std::size_t table_index,
        std::span<const std::uint64_t> row_ids);

    std::size_t table_count() const noexcept;
    std::size_t head_dim() const noexcept;
    std::size_t cache_limit_bytes() const noexcept;
    DeepseekV41EngramSsdStats stats() const;
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::metal
