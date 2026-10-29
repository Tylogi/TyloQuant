#pragma once

#include "mfq_container.h"
#include "mlx_nint_rows.h"
#include "mlx_tq_rows.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxQuantizedRowBatch {
    int width = 0;
    std::vector<mlx::core::float16_t> values;
    MlxNintRowBatch packed;
    MlxTqRowBatch tq;
    mlx::core::array decode() const;
};

struct MlxQuantizedRowStoreStats {
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

// Architecture-neutral bounded row cache for native MXFP8 and NINT tables.
// Cold rows are read through MfqContainer, so the same implementation serves
// regular MFQ artifacts and canonical virtual records over untouched HF
// checkpoints. Persistent workers expose SSD queue depth without holding the
// LRU lock across I/O. NINT keeps only selectors, rank checkpoints and selected
// packed rows; the common row kernel decodes the requested batch on the GPU.
class MlxQuantizedRowStore {
public:
    MlxQuantizedRowStore(
        const MfqContainer& model,
        std::string record,
        std::int64_t expected_rows,
        int expected_width,
        std::size_t cache_rows,
        std::size_t io_workers = 8);
    ~MlxQuantizedRowStore();

    MlxQuantizedRowStore(const MlxQuantizedRowStore&) = delete;
    MlxQuantizedRowStore& operator=(const MlxQuantizedRowStore&) = delete;

    // Returns decoded F16 rows in request order with shape [N,width].
    mlx::core::array gather(std::span<const std::int64_t> row_ids) const;
    // Reader threads prepare CPU storage; decode belongs to the inference stream.
    MlxQuantizedRowBatch prepare_rows(std::span<const std::int64_t> row_ids) const;

    std::int64_t rows() const noexcept;
    int width() const noexcept;
    std::size_t cached_rows() const noexcept;
    MlxQuantizedRowStoreStats stats() const;
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::metal
