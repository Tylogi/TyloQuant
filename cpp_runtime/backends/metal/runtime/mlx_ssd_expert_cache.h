#pragma once

#include "mlx_ssd_expert_arena.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mfq::metal {

struct MlxSsdExpertCacheStats {
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
    std::uint64_t device_route_layers = 0;
    std::uint64_t device_route_hits = 0;
    std::uint64_t device_route_misses = 0;
    double io_seconds = 0.0;
    double wait_seconds = 0.0;
    double prefill_wait_seconds = 0.0;
    double route_sync_seconds = 0.0;
    double route_sync_cpu_seconds = 0.0;
    double route_host_seconds = 0.0;
    double prepare_seconds = 0.0;
    double view_seconds = 0.0;
    double device_route_eval_seconds = 0.0;
    double device_route_host_seconds = 0.0;
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

struct MlxSsdExpertRouteSelection {
    std::size_t layer = 0;
    std::vector<std::int32_t> experts;
    bool all_hit = false;
};

struct MlxSsdExpertRouteTransactionResult {
    std::vector<MlxSsdExpertRouteSelection> routes;
    bool all_hit = true;
};

class MlxSsdPrefetchedExpertLayer {
public:
    MlxSsdPrefetchedExpertLayer(
        MlxSsdPrefetchedExpertLayer&&) noexcept;
    MlxSsdPrefetchedExpertLayer& operator=(
        MlxSsdPrefetchedExpertLayer&&) noexcept;
    ~MlxSsdPrefetchedExpertLayer();

    MlxSsdPrefetchedExpertLayer(
        const MlxSsdPrefetchedExpertLayer&) = delete;
    MlxSsdPrefetchedExpertLayer& operator=(
        const MlxSsdPrefetchedExpertLayer&) = delete;

    const MlxSsdExpertWeights& wait();
    std::size_t layer() const noexcept;

private:
    struct Impl;
    explicit MlxSsdPrefetchedExpertLayer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend class MlxMoeSsdExpertCache;
};

class MlxSsdPreparedExperts {
public:
    MlxSsdPreparedExperts(MlxSsdPreparedExperts&&) noexcept;
    MlxSsdPreparedExperts& operator=(MlxSsdPreparedExperts&&) noexcept;
    ~MlxSsdPreparedExperts();

    MlxSsdPreparedExperts(const MlxSsdPreparedExperts&) = delete;
    MlxSsdPreparedExperts& operator=(const MlxSsdPreparedExperts&) = delete;

    const MlxSsdExpertWeights& weights() const noexcept;
    std::span<const std::int32_t> slot_for_expert() const noexcept;

private:
    MlxSsdPreparedExperts(
        MlxSsdExpertWeights weights,
        std::vector<std::int32_t> slot_for_expert,
        std::function<void()> release);

    std::unique_ptr<MlxSsdExpertWeights> weights_;
    std::vector<std::int32_t> slot_for_expert_;
    std::function<void()> release_;

    friend class MlxMoeSsdExpertCache;
};

// Immutable resident page-table view for one routed-expert layer. While this
// handle is alive, cache slots present in the table cannot be evicted or
// reused. Readiness is represented explicitly and a missing expert maps to
// slot -1, which routed Metal kernels treat as a zero-output route. Call
// finish() only after all arrays using the page table have been evaluated.
class MlxSsdExpertPageTableSnapshot {
public:
    MlxSsdExpertPageTableSnapshot(MlxSsdExpertPageTableSnapshot&&) noexcept;
    MlxSsdExpertPageTableSnapshot& operator=(
        MlxSsdExpertPageTableSnapshot&&) noexcept;
    ~MlxSsdExpertPageTableSnapshot();

    MlxSsdExpertPageTableSnapshot(
        const MlxSsdExpertPageTableSnapshot&) = delete;
    MlxSsdExpertPageTableSnapshot& operator=(
        const MlxSsdExpertPageTableSnapshot&) = delete;

    const MlxSsdExpertWeights& weights() const noexcept;
    const mlx::core::array& slot_ids() const noexcept;
    const mlx::core::array& generations() const noexcept;
    const mlx::core::array& readiness() const noexcept;
    std::span<const std::int32_t> slot_for_expert() const noexcept;

    void finish(
        std::span<const std::int32_t> active_experts,
        bool all_hit,
        double eval_seconds,
        double host_seconds);
    void defer_finish(
        std::span<const std::int32_t> active_experts,
        double eval_seconds,
        double host_seconds);
    void defer_transaction(const mlx::core::array& packed_expert_ids);

private:
    struct Impl;
    explicit MlxSsdExpertPageTableSnapshot(
        std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend class MlxMoeSsdExpertCache;
};

// Concurrent exact-expert LRU over the shared MLX UMA arena. Safetensors on
// SSD are the complete expert pool and the arena is the only physical cache:
// CPU IO workers fill its pages and Metal consumes those same pages in place,
// with no host-pool or host-to-device copy. A prepare() call issues all cold
// reads before waiting, so the configured IO workers expose the SSD queue
// depth. The returned object pins its slots until destruction; callers must
// materialize the lazy Metal graph before releasing it.
class MfqContainer;

class MlxMoeSsdExpertCache {
public:
    MlxMoeSsdExpertCache(
        const MfqContainer& model,
        std::vector<std::string> layer_prefixes,
        std::size_t hidden_size,
        std::size_t intermediate_size,
        std::vector<std::size_t> experts_per_layer,
        std::size_t cache_bytes,
        std::size_t io_workers = 8,
        bool prefill_overlap = false);
    MlxMoeSsdExpertCache(
        const MfqContainer& model,
        std::vector<std::string> layer_prefixes,
        std::size_t hidden_size,
        std::size_t intermediate_size,
        std::size_t num_experts,
        std::size_t cache_bytes,
        std::size_t io_workers = 8,
        bool prefill_overlap = false);
    ~MlxMoeSsdExpertCache();

    MlxMoeSsdExpertCache(const MlxMoeSsdExpertCache&) = delete;
    MlxMoeSsdExpertCache& operator=(const MlxMoeSsdExpertCache&) = delete;

    MlxSsdPreparedExperts prepare(
        std::size_t layer,
        std::span<const std::int32_t> active_experts,
        std::function<void(
            const MlxSsdExpertWeights&,
            std::span<const std::int32_t> ready_experts,
            std::span<const std::int32_t> slot_for_expert)> overlap = {},
        std::function<void(
            const MlxSsdExpertWeights&,
            std::span<const std::int32_t> pending_experts,
            std::span<const std::int32_t> slot_for_expert)> gate_up_ready = {});

    // Freeze the current resident mapping for a layer and expose it as device
    // arrays. This is the decode fast path: routing IDs can be remapped to
    // arena slots without a host synchronization before routed MoE execution.
    MlxSsdExpertPageTableSnapshot snapshot_page_table(
        std::size_t layer);

    // A decode route transaction keeps all page-table generations frozen
    // while one complete lazy token graph runs. Packed route IDs are checked
    // only at the final model evaluation boundary. A miss therefore never
    // escapes as output: the model can pin the returned routes and replay the
    // token from its lightweight cache checkpoint.
    void begin_route_transaction();
    bool route_transaction_active() const noexcept;
    bool route_layer_likely_hit(std::size_t layer) const noexcept;
    MlxSsdExpertRouteTransactionResult resolve_route_transaction();
    void cancel_route_transaction() noexcept;

    // Start loading a complete routed-expert layer into one of two alternating
    // buffers. Resident LRU rows are pinned and reused in place; only misses
    // consume SSD bandwidth. Call wait() before submitting the layer's Metal
    // graph and keep the returned object alive until that graph is evaluated.
    MlxSsdPrefetchedExpertLayer prefetch_layer(std::size_t layer);

    std::size_t cache_limit_bytes() const noexcept;
    std::size_t cache_slots() const noexcept;
    bool prefill_overlap_enabled() const noexcept;
    MlxSsdExpertCacheStats stats() const;

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

    friend class MlxSsdPrefetchedExpertLayer;
    friend class MlxSsdExpertPageTableSnapshot;
};

} // namespace mfq::metal
