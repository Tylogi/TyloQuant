#pragma once

#include "deepseek_v41_engram_store.h"
#include "deepseek_v4_model.h"
#include "mlx_hf_tensor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV41HfEngramBatch {
    int batch = 0;
    int tokens = 0;
    int start_position = 0;
    std::vector<std::uint8_t> token_mask;
    std::vector<std::vector<std::uint64_t>> row_ids;
    std::vector<
        std::shared_future<std::vector<std::uint16_t>>>
        prefetched_rows;
};

// Model-wide V4.1 Engram adapter. Hashing happens once per input chunk; each
// Engram layer then pulls only its selected rows through the bounded SSD cache
// immediately before that layer executes.
class MlxDeepseekV41HfEngram {
public:
    static MlxDeepseekV41HfEngram load_hf(
        const MlxHfTensorStore& model,
        const DeepseekV4Config& config,
        const std::filesystem::path& hash_asset,
        std::size_t cache_bytes,
        std::size_t io_workers = 8);

    MlxDeepseekV41HfEngramBatch prepare(
        const mlx::core::array& token_ids,
        int start_position,
        bool prefetch_rows = false);

    bool has_layer(std::size_t layer) const noexcept;
    mlx::core::array apply(
        std::size_t layer,
        const mlx::core::array& hidden,
        const MlxDeepseekV41HfEngramBatch& batch) const;

    void reset_hash() noexcept;
    void truncate_hash(int position);
    // Rebuild only the tiny rolling hash history when restoring a text KV
    // cache. No Engram table rows are read by this operation.
    void restore_text_hash(std::span<const std::int64_t> token_ids);
    DeepseekV41EngramSsdStats stats() const;
    void clear_rows();

private:
    struct Impl;
    explicit MlxDeepseekV41HfEngram(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_;
};

} // namespace mfq::metal
