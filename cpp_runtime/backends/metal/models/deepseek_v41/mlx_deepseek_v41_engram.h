#pragma once

#include "deepseek_v41_model.h"
#include "mlx_tensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// Hashes are stored layer-major so each Engram layer receives one contiguous
// [batch,tokens,hash-columns] view without copying.
struct DeepseekV41EngramHashBatch {
    int batch = 0;
    int tokens = 0;
    int layers = 0;
    int hash_columns = 0;
    std::vector<std::int64_t> values;

    std::span<const std::int64_t> layer(int index) const;
};

struct DeepseekV41EngramHashSnapshot {
    int batch = 0;
    int position = 0;
    std::vector<std::int64_t> context;
};

// Token normalization is performed once during conversion. The native
// runtime only applies the embedded compressed-token map and exact released
// integer hash contract.
class MlxDeepseekV41EngramHashState {
public:
    static MlxDeepseekV41EngramHashState load(
        const MfqContainer& model,
        const DeepseekV41Config& config);

    void reset(int batch);
    void clear() noexcept;

    DeepseekV41EngramHashBatch forward(
        const mlx::core::array& token_ids,
        int pos0,
        const std::optional<mlx::core::array>& participation_mask,
        bool use_cache);

    DeepseekV41EngramHashSnapshot snapshot() const;
    void restore(DeepseekV41EngramHashSnapshot snapshot);

    int position() const noexcept { return position_; }
    int layer_count() const noexcept {
        return static_cast<int>(layer_ids_.size());
    }
    int hash_columns() const noexcept {
        return (max_ngram_size_ - 1) * heads_;
    }

private:
    MlxDeepseekV41EngramHashState(
        int vocabulary,
        int compressed_vocabulary,
        int max_ngram_size,
        int heads,
        std::int64_t pad_id,
        std::vector<std::int64_t> layer_ids,
        std::vector<std::int64_t> table_rows,
        std::vector<std::int64_t> primes,
        std::vector<std::int64_t> offsets,
        std::vector<std::int64_t> multipliers,
        std::vector<std::int32_t> token_map);

    int vocabulary_;
    int compressed_vocabulary_;
    int max_ngram_size_;
    int heads_;
    std::int64_t pad_id_;
    std::vector<std::int64_t> layer_ids_;
    std::vector<std::int64_t> table_rows_;
    std::vector<std::int64_t> primes_;
    std::vector<std::int64_t> offsets_;
    std::vector<std::int64_t> multipliers_;
    std::vector<std::int32_t> token_map_;
    int batch_ = 0;
    int position_ = 0;
    std::vector<std::int64_t> context_;
};

// V4.1's table remains source-exact E4M3+E8M0 on disk. Only the 24 selected
// rows per token are faulted in and decoded; a bounded decoded-row LRU avoids
// repeated storage reads without making the table resident.
class MlxDeepseekV41Engram {
public:
    static MlxDeepseekV41Engram load(
        const MfqContainer& model,
        const DeepseekV41Config& config,
        int layer);

    mlx::core::array forward(
        const mlx::core::array& hidden_streams,
        const DeepseekV41EngramHashBatch& hashes,
        const std::optional<mlx::core::array>& participation_mask) const;

    int layer() const noexcept { return layer_; }
    int hash_layer_index() const noexcept { return hash_layer_index_; }
    std::size_t cached_rows() const noexcept;

private:
    class Table;

    MlxDeepseekV41Engram(
        DeepseekV41Config config,
        int layer,
        int hash_layer_index,
        std::shared_ptr<Table> table,
        MlxLinear projection,
        mlx::core::array query_key_weight);

    DeepseekV41Config config_;
    int layer_;
    int hash_layer_index_;
    std::shared_ptr<Table> table_;
    MlxLinear projection_;
    mlx::core::array query_key_weight_;
};

} // namespace mfq::metal
