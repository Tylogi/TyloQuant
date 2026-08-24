#pragma once

#include "mlx_ssd_expert_arena.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace mfq::metal {

struct MlxDeepseekV4SsdCacheStats {
    std::uint64_t requests = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t coalesced = 0;
    std::uint64_t evictions = 0;
    std::uint64_t loads = 0;
    std::uint64_t failures = 0;
    std::uint64_t bytes_read = 0;
    std::uint64_t read_calls = 0;
    std::uint64_t prefill_layers = 0;
    std::uint64_t prefill_cache_hits = 0;
    std::uint64_t prefill_expert_reads = 0;
    std::uint64_t prefill_bytes_read = 0;
    double io_seconds = 0.0;
    double wait_seconds = 0.0;
    double prefill_wait_seconds = 0.0;
    double route_sync_seconds = 0.0;
    double route_sync_cpu_seconds = 0.0;
    double route_host_seconds = 0.0;
    double prepare_seconds = 0.0;
    double view_seconds = 0.0;
    std::size_t resident_experts = 0;
    std::size_t resident_bytes = 0;
    std::size_t cache_slots = 0;

    double hit_rate() const noexcept {
        return requests == 0
            ? 0.0
            : static_cast<double>(hits) /
                static_cast<double>(requests);
    }
};

class MlxDeepseekV4SsdPrefetchedLayer {
public:
    MlxDeepseekV4SsdPrefetchedLayer(
        MlxDeepseekV4SsdPrefetchedLayer&&) noexcept;
    MlxDeepseekV4SsdPrefetchedLayer& operator=(
        MlxDeepseekV4SsdPrefetchedLayer&&) noexcept;
    ~MlxDeepseekV4SsdPrefetchedLayer();

    MlxDeepseekV4SsdPrefetchedLayer(
        const MlxDeepseekV4SsdPrefetchedLayer&) = delete;
    MlxDeepseekV4SsdPrefetchedLayer& operator=(
        const MlxDeepseekV4SsdPrefetchedLayer&) = delete;

    const MlxDeepseekV4SsdExpertWeights& wait();
    std::size_t layer() const noexcept;

private:
    struct Impl;
    explicit MlxDeepseekV4SsdPrefetchedLayer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend class MlxDeepseekV4SsdExpertCache;
};

class MlxDeepseekV4SsdPreparedExperts {
public:
    MlxDeepseekV4SsdPreparedExperts(
        MlxDeepseekV4SsdPreparedExperts&&) noexcept;
    MlxDeepseekV4SsdPreparedExperts& operator=(
        MlxDeepseekV4SsdPreparedExperts&&) noexcept;
    ~MlxDeepseekV4SsdPreparedExperts();

    MlxDeepseekV4SsdPreparedExperts(
        const MlxDeepseekV4SsdPreparedExperts&) = delete;
    MlxDeepseekV4SsdPreparedExperts& operator=(
        const MlxDeepseekV4SsdPreparedExperts&) = delete;

    const MlxDeepseekV4SsdExpertWeights& weights() const noexcept;
    std::span<const std::int32_t> slot_for_expert() const noexcept;

private:
    MlxDeepseekV4SsdPreparedExperts(
        MlxDeepseekV4SsdExpertWeights weights,
        std::vector<std::int32_t> slot_for_expert,
        std::function<void()> release);

    std::unique_ptr<MlxDeepseekV4SsdExpertWeights> weights_;
    std::vector<std::int32_t> slot_for_expert_;
    std::function<void()> release_;

    friend class MlxDeepseekV4SsdExpertCache;
};

// Concurrent exact-expert LRU over the shared MLX UMA arena. Safetensors on
// SSD are the complete expert pool and the arena is the only physical cache:
// CPU IO workers fill its pages and Metal consumes those same pages in place,
// with no host-pool or host-to-device copy. A prepare() call issues all cold
// reads before waiting, so the configured IO workers expose the SSD queue
// depth. The returned object pins its slots until destruction; callers must
// materialize the lazy Metal graph before releasing it.
class MlxDeepseekV4SsdExpertCache {
public:
    MlxDeepseekV4SsdExpertCache(
        std::filesystem::path model_root,
        std::size_t cache_bytes,
        std::size_t io_workers = 8,
        bool prefill_overlap = false);
    ~MlxDeepseekV4SsdExpertCache();

    MlxDeepseekV4SsdExpertCache(
        const MlxDeepseekV4SsdExpertCache&) = delete;
    MlxDeepseekV4SsdExpertCache& operator=(
        const MlxDeepseekV4SsdExpertCache&) = delete;

    MlxDeepseekV4SsdPreparedExperts prepare(
        std::size_t layer,
        std::span<const std::int32_t> active_experts,
        std::function<void(
            const MlxDeepseekV4SsdExpertWeights&,
            std::span<const std::int32_t> ready_experts,
            std::span<const std::int32_t> slot_for_expert)> overlap = {},
        std::function<void(
            const MlxDeepseekV4SsdExpertWeights&,
            std::span<const std::int32_t> pending_experts,
            std::span<const std::int32_t> slot_for_expert)> gate_up_ready = {});

    // Start loading a complete routed-expert layer into one of two alternating
    // buffers. Resident LRU rows are pinned and reused in place; only misses
    // consume SSD bandwidth. Call wait() before submitting the layer's Metal
    // graph and keep the returned object alive until that graph is evaluated.
    MlxDeepseekV4SsdPrefetchedLayer prefetch_layer(std::size_t layer);

    std::size_t cache_limit_bytes() const noexcept;
    std::size_t cache_slots() const noexcept;
    bool prefill_overlap_enabled() const noexcept;
    MlxDeepseekV4SsdCacheStats stats() const;

    void record_route_timing(
        double sync_seconds,
        double sync_cpu_seconds,
        double host_seconds) noexcept;
    void reset_stats();
    void prewarm_metal();
    // Release slots retained for a lazy Metal graph after that graph has
    // reached an evaluation boundary.
    void release_deferred();
    void clear();

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;

    friend class MlxDeepseekV4SsdPrefetchedLayer;
};

} // namespace mfq::metal
