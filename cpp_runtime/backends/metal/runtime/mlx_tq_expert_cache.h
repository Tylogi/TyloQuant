#pragma once

#include "mlx_tq.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mfq::metal {

struct MlxTqExpertWeights {
    MlxTqWeight gate, up, down;
    void prepare(int top_k, float limit = 0.f, bool gelu = false) const;
};

struct MlxTqExpertCacheStats {
    std::uint64_t requests = 0, hits = 0, misses = 0, evictions = 0;
    std::uint64_t bytes_read = 0, read_calls = 0;
    std::size_t cached_experts = 0, cached_bytes = 0, allocated_bytes = 0;
    double io_seconds = 0, route_sync_seconds = 0, prepare_seconds = 0, graph_seconds = 0;
    std::size_t metadata_bytes = 0;
    std::uint64_t overlapped_read_batches = 0;
};

class MlxTqExpertCache {
public:
    MlxTqExpertCache(const MfqContainer& model, const std::vector<std::string>& prefixes,
        std::size_t budget, int io_workers = 8);
    mlx::core::array apply(std::size_t layer, const mlx::core::array& ids,
        const std::function<mlx::core::array(const MlxTqExpertWeights&)>& operation);
    mlx::core::array apply(std::size_t layer, const mlx::core::array& ids,
        const std::function<mlx::core::array(const MlxTqExpertWeights&,
            const mlx::core::array*)>& operation);
    mlx::core::array apply(std::size_t layer, const mlx::core::array& ids,
        const std::function<mlx::core::array(const MlxTqExpertWeights&,
            const mlx::core::array*)>& gate_up,
        const std::function<mlx::core::array(const MlxTqExpertWeights&,
            const mlx::core::array&, const mlx::core::array*)>& down,
        const mlx::core::array* parallel = nullptr);
    MlxTqExpertCacheStats stats() const;
    std::size_t limit_bytes() const noexcept;
    std::size_t backing_bytes() const noexcept;
    std::size_t resident_bytes() const noexcept;
    std::size_t set_limit(std::size_t bytes);
    std::size_t trim(std::size_t bytes);
    void validate_layer(std::size_t layer, int experts, int input, int intermediate) const;
    void prepare_layer(std::size_t layer, int top_k, float limit = 0.f, bool gelu = false) const;
    void clear();

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace mfq::metal
