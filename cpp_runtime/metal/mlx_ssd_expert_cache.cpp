#include "mlx_ssd_expert_cache.h"

#include "hf_safetensors_store.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

std::uint64_t expert_key(std::size_t layer, std::int32_t expert) {
    return (static_cast<std::uint64_t>(layer) << 32u) |
        static_cast<std::uint32_t>(expert);
}

} // namespace

struct MlxDeepseekV4SsdExpertCache::Impl {
    enum class State {
        empty,
        loading,
        ready,
        failed,
    };

    struct Slot {
        State state = State::empty;
        std::uint64_t key = 0;
        std::uint64_t generation = 0;
        std::uint64_t last_use = 0;
        std::size_t pins = 0;
        std::shared_future<void> ready;
    };

    struct Task {
        std::size_t arena_slot = 0;
        std::size_t cache_slot = 0;
        std::uint64_t generation = 0;
        std::size_t layer = 0;
        std::int32_t expert = 0;
        bool cache_admission = true;
        std::shared_ptr<std::promise<void>> completion;
    };

    struct Acquisition {
        std::size_t slot = 0;
        std::uint64_t generation = 0;
        std::shared_future<void> ready;
    };

    Impl(
        std::filesystem::path root,
        std::size_t bytes,
        std::size_t worker_count,
        bool overlap)
        : store(std::move(root), 43, 256),
          slot_bytes(store.slot_bytes()),
          prefill_slots(overlap ? 2 * store.num_experts() : 0),
          total_slots(bytes / slot_bytes),
          slot_count(total_slots > prefill_slots
              ? total_slots - prefill_slots
              : 0),
          limit_bytes(total_slots * slot_bytes),
          arena(total_slots),
          slots(slot_count) {
        if (slot_count < 6) {
            throw std::invalid_argument(
                "SSD expert cache must hold at least six routed experts");
        }
        if (worker_count == 0) {
            throw std::invalid_argument("SSD expert IO worker count must be positive");
        }
        workers.reserve(worker_count);
        for (std::size_t index = 0; index < worker_count; ++index) {
            workers.emplace_back([this] { worker(); });
        }
    }

    ~Impl() {
        {
            std::scoped_lock lock(mutex);
            stopping = true;
        }
        condition.notify_all();
        for (auto& worker_thread : workers) {
            worker_thread.join();
        }
    }

    Acquisition acquire(std::size_t layer, std::int32_t expert) {
        if (layer >= store.num_layers() || expert < 0 ||
            static_cast<std::size_t>(expert) >= store.num_experts()) {
            throw std::out_of_range("SSD expert cache key out of range");
        }
        std::unique_lock lock(mutex);
        ++counters.requests;
        const auto key = expert_key(layer, expert);
        auto found = resident.find(key);
        if (found != resident.end()) {
            auto& slot = slots[found->second];
            if (slot.state == State::ready || slot.state == State::loading) {
                ++counters.hits;
                if (slot.state == State::loading) {
                    ++counters.coalesced;
                }
                ++slot.pins;
                slot.last_use = ++clock;
                return {
                    .slot = found->second,
                    .generation = slot.generation,
                    .ready = slot.ready,
                };
            }
            resident.erase(found);
        }

        std::size_t victim = slots.size();
        std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t index = 0; index < slots.size(); ++index) {
            const auto& slot = slots[index];
            if (slot.state == State::empty) {
                victim = index;
                break;
            }
            if (slot.state != State::loading && slot.pins == 0 &&
                slot.last_use < oldest) {
                oldest = slot.last_use;
                victim = index;
            }
        }
        if (victim == slots.size()) {
            throw std::runtime_error(
                "SSD expert cache has no unpinned eviction slot");
        }
        auto& slot = slots[victim];
        if (slot.state != State::empty) {
            resident.erase(slot.key);
            ++counters.evictions;
        }
        auto completion = std::make_shared<std::promise<void>>();
        slot.state = State::loading;
        slot.key = key;
        ++slot.generation;
        slot.last_use = ++clock;
        slot.pins = 1;
        slot.ready = completion->get_future().share();
        resident.emplace(key, victim);
        ++counters.misses;
        tasks.push_back({
            .arena_slot = prefill_slots + victim,
            .cache_slot = victim,
            .generation = slot.generation,
            .layer = layer,
            .expert = expert,
            .cache_admission = true,
            .completion = completion,
        });
        const auto result = Acquisition{
            .slot = victim,
            .generation = slot.generation,
            .ready = slot.ready,
        };
        lock.unlock();
        condition.notify_one();
        return result;
    }

    std::optional<Acquisition> pin_if_resident(
        std::size_t layer,
        std::int32_t expert) {
        std::scoped_lock lock(mutex);
        const auto found = resident.find(expert_key(layer, expert));
        if (found == resident.end()) {
            return std::nullopt;
        }
        auto& slot = slots[found->second];
        if (slot.state != State::ready && slot.state != State::loading) {
            return std::nullopt;
        }
        ++slot.pins;
        return Acquisition{
            .slot = found->second,
            .generation = slot.generation,
            .ready = slot.ready,
        };
    }

    std::shared_future<void> schedule_direct(
        std::size_t layer,
        std::int32_t expert,
        std::size_t arena_slot) {
        auto completion = std::make_shared<std::promise<void>>();
        auto ready = completion->get_future().share();
        {
            std::scoped_lock lock(mutex);
            tasks.push_back({
                .arena_slot = arena_slot,
                .cache_slot = 0,
                .generation = 0,
                .layer = layer,
                .expert = expert,
                .cache_admission = false,
                .completion = std::move(completion),
            });
        }
        condition.notify_one();
        return ready;
    }

    void claim_prefill_buffer(std::size_t buffer) {
        std::scoped_lock lock(mutex);
        if (prefill_slots == 0 || buffer >= prefill_busy.size()) {
            throw std::runtime_error("SSD expert prefill overlap is disabled");
        }
        if (prefill_busy[buffer]) {
            throw std::runtime_error(
                "SSD expert prefill buffer reused before release");
        }
        prefill_busy[buffer] = true;
        ++counters.prefill_layers;
    }

    void release_prefill_buffer(
        std::size_t buffer,
        const std::vector<Acquisition>& acquisitions) noexcept {
        std::scoped_lock lock(mutex);
        for (const auto& acquired : acquisitions) {
            if (acquired.slot >= slots.size()) {
                continue;
            }
            auto& slot = slots[acquired.slot];
            if (slot.generation == acquired.generation && slot.pins != 0) {
                --slot.pins;
            }
        }
        if (buffer < prefill_busy.size()) {
            prefill_busy[buffer] = false;
        }
    }

    void release(const std::vector<Acquisition>& acquisitions) noexcept {
        std::scoped_lock lock(mutex);
        for (const auto& acquired : acquisitions) {
            if (acquired.slot >= slots.size()) {
                continue;
            }
            auto& slot = slots[acquired.slot];
            if (slot.generation == acquired.generation && slot.pins != 0) {
                --slot.pins;
            }
        }
    }

    void defer_release(std::vector<Acquisition> acquisitions) noexcept {
        std::scoped_lock lock(mutex);
        deferred.insert(
            deferred.end(),
            std::make_move_iterator(acquisitions.begin()),
            std::make_move_iterator(acquisitions.end()));
    }

    void release_deferred() noexcept {
        std::scoped_lock lock(mutex);
        for (const auto& acquired : deferred) {
            if (acquired.slot >= slots.size()) {
                continue;
            }
            auto& slot = slots[acquired.slot];
            if (slot.generation == acquired.generation && slot.pins != 0) {
                --slot.pins;
            }
        }
        deferred.clear();
    }

    void worker() {
        while (true) {
            Task task;
            {
                std::unique_lock lock(mutex);
                condition.wait(lock, [&] {
                    return stopping || !tasks.empty();
                });
                if (stopping && tasks.empty()) {
                    return;
                }
                task = std::move(tasks.front());
                tasks.pop_front();
            }
            const auto begin = std::chrono::steady_clock::now();
            try {
                const auto result = store.load_scatter(
                    task.layer,
                    static_cast<std::size_t>(task.expert),
                    arena.destination(task.arena_slot));
                const auto seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - begin).count();
                {
                    std::scoped_lock lock(mutex);
                    if (task.cache_admission) {
                        auto& slot = slots[task.cache_slot];
                        if (slot.generation != task.generation ||
                            slot.state != State::loading) {
                            throw std::logic_error(
                                "SSD expert slot changed while loading");
                        }
                        slot.state = State::ready;
                        ++counters.loads;
                        counters.bytes_read += result.bytes;
                        counters.read_calls += result.read_calls;
                    } else {
                        ++counters.prefill_expert_reads;
                        counters.prefill_bytes_read += result.bytes;
                    }
                    counters.io_seconds += seconds;
                }
                task.completion->set_value();
            } catch (...) {
                {
                    std::scoped_lock lock(mutex);
                    if (task.cache_admission) {
                        auto& slot = slots[task.cache_slot];
                        if (slot.generation == task.generation) {
                            slot.state = State::failed;
                        }
                    }
                    ++counters.failures;
                }
                try {
                    task.completion->set_exception(std::current_exception());
                } catch (...) {
                }
            }
        }
    }

    MlxDeepseekV4SsdCacheStats stats() const {
        std::scoped_lock lock(mutex);
        auto result = counters;
        result.cache_slots = slot_count;
        result.resident_experts = static_cast<std::size_t>(std::count_if(
            slots.begin(),
            slots.end(),
            [](const Slot& slot) { return slot.state == State::ready; }));
        result.resident_bytes = result.resident_experts * slot_bytes;
        return result;
    }

    DeepseekV4NativeExpertStore store;
    const std::size_t slot_bytes;
    const std::size_t prefill_slots;
    const std::size_t total_slots;
    const std::size_t slot_count;
    const std::size_t limit_bytes;
    MlxDeepseekV4SsdExpertArena arena;
    mutable std::mutex mutex;
    std::condition_variable condition;
    bool stopping = false;
    std::uint64_t clock = 0;
    std::vector<Slot> slots;
    std::unordered_map<std::uint64_t, std::size_t> resident;
    std::deque<Task> tasks;
    std::vector<Acquisition> deferred;
    std::vector<std::thread> workers;
    std::array<bool, 2> prefill_busy{false, false};
    MlxDeepseekV4SsdCacheStats counters;
};

struct MlxDeepseekV4SsdPrefetchedLayer::Impl {
    std::shared_ptr<MlxDeepseekV4SsdExpertCache::Impl> cache;
    std::size_t layer = 0;
    std::size_t buffer = 0;
    std::vector<MlxDeepseekV4SsdExpertCache::Impl::Acquisition> cache_pins;
    std::vector<std::shared_future<void>> reads;
    std::vector<std::int32_t> slot_map;
    std::unique_ptr<MlxDeepseekV4SsdExpertWeights> weights;
    bool released = false;

    const MlxDeepseekV4SsdExpertWeights& wait() {
        if (weights) {
            return *weights;
        }
        const auto begin = std::chrono::steady_clock::now();
        for (const auto& pin : cache_pins) {
            pin.ready.get();
        }
        for (const auto& read : reads) {
            read.get();
        }
        const auto seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        const auto expert_count = cache->store.num_experts();
        std::vector<std::int32_t> active(expert_count);
        for (std::size_t expert = 0; expert < expert_count; ++expert) {
            active[expert] = static_cast<std::int32_t>(expert);
        }
        weights = std::make_unique<MlxDeepseekV4SsdExpertWeights>(
            cache->arena.routed_weights(slot_map, active));
        {
            std::scoped_lock lock(cache->mutex);
            cache->counters.prefill_wait_seconds += seconds;
        }
        return *weights;
    }

    void release() noexcept {
        if (released || !cache) {
            return;
        }
        // A buffer cannot be handed to the next layer while an IO worker is
        // still writing it, even when the caller abandons a prefetch early.
        for (const auto& pin : cache_pins) {
            try {
                pin.ready.get();
            } catch (...) {
            }
        }
        for (const auto& read : reads) {
            try {
                read.get();
            } catch (...) {
            }
        }
        cache->release_prefill_buffer(buffer, cache_pins);
        released = true;
    }

    ~Impl() {
        release();
    }
};

MlxDeepseekV4SsdPreparedExperts::MlxDeepseekV4SsdPreparedExperts(
    MlxDeepseekV4SsdExpertWeights weights,
    std::function<void()> release)
    : weights_(std::make_unique<MlxDeepseekV4SsdExpertWeights>(
          std::move(weights))),
      release_(std::move(release)) {}

MlxDeepseekV4SsdPreparedExperts::MlxDeepseekV4SsdPreparedExperts(
    MlxDeepseekV4SsdPreparedExperts&& other) noexcept
    : weights_(std::move(other.weights_)),
      release_(std::move(other.release_)) {
    other.release_ = {};
}

MlxDeepseekV4SsdPreparedExperts&
MlxDeepseekV4SsdPreparedExperts::operator=(
    MlxDeepseekV4SsdPreparedExperts&& other) noexcept {
    if (this != &other) {
        if (release_) {
            release_();
        }
        weights_ = std::move(other.weights_);
        release_ = std::move(other.release_);
        other.release_ = {};
    }
    return *this;
}

MlxDeepseekV4SsdPreparedExperts::~MlxDeepseekV4SsdPreparedExperts() {
    if (release_) {
        release_();
    }
}

const MlxDeepseekV4SsdExpertWeights&
MlxDeepseekV4SsdPreparedExperts::weights() const noexcept {
    return *weights_;
}

MlxDeepseekV4SsdPrefetchedLayer::MlxDeepseekV4SsdPrefetchedLayer(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MlxDeepseekV4SsdPrefetchedLayer::MlxDeepseekV4SsdPrefetchedLayer(
    MlxDeepseekV4SsdPrefetchedLayer&&) noexcept = default;

MlxDeepseekV4SsdPrefetchedLayer&
MlxDeepseekV4SsdPrefetchedLayer::operator=(
    MlxDeepseekV4SsdPrefetchedLayer&&) noexcept = default;

MlxDeepseekV4SsdPrefetchedLayer::~MlxDeepseekV4SsdPrefetchedLayer() = default;

const MlxDeepseekV4SsdExpertWeights&
MlxDeepseekV4SsdPrefetchedLayer::wait() {
    if (!impl_) {
        throw std::runtime_error("SSD expert prefetch handle is empty");
    }
    return impl_->wait();
}

std::size_t MlxDeepseekV4SsdPrefetchedLayer::layer() const noexcept {
    return impl_ ? impl_->layer : 0;
}

MlxDeepseekV4SsdExpertCache::MlxDeepseekV4SsdExpertCache(
    std::filesystem::path model_root,
    std::size_t cache_bytes,
    std::size_t io_workers,
    bool prefill_overlap)
    : impl_(std::make_shared<Impl>(
          std::move(model_root),
          cache_bytes,
          io_workers,
          prefill_overlap)) {}

MlxDeepseekV4SsdExpertCache::~MlxDeepseekV4SsdExpertCache() = default;

MlxDeepseekV4SsdPreparedExperts MlxDeepseekV4SsdExpertCache::prepare(
    std::size_t layer,
    std::span<const std::int32_t> active_experts,
    std::function<void()> overlap) {
    const auto prepare_begin = std::chrono::steady_clock::now();
    // The caller has evaluated the current layer's routing IDs before this
    // point, which also completes the previous layer graph. Its arena rows can
    // now safely return to the LRU before acquiring this layer's experts.
    impl_->release_deferred();
    std::vector<std::int32_t> unique(active_experts.begin(), active_experts.end());
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    if (unique.empty()) {
        throw std::invalid_argument("SSD expert prepare set is empty");
    }

    std::vector<Impl::Acquisition> acquisitions;
    acquisitions.reserve(unique.size());
    try {
        for (const auto expert : unique) {
            acquisitions.push_back(impl_->acquire(layer, expert));
        }
        const bool pending = std::any_of(
            acquisitions.begin(),
            acquisitions.end(),
            [](const Impl::Acquisition& acquired) {
                return acquired.ready.wait_for(
                           std::chrono::seconds(0)) !=
                    std::future_status::ready;
            });
        if (pending && overlap) {
            overlap();
        }
        const auto wait_begin = std::chrono::steady_clock::now();
        for (const auto& acquired : acquisitions) {
            acquired.ready.get();
        }
        const auto wait_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wait_begin).count();
        std::vector<std::int32_t> slot_map(impl_->store.num_experts(), -1);
        for (std::size_t index = 0; index < unique.size(); ++index) {
            slot_map[static_cast<std::size_t>(unique[index])] =
                static_cast<std::int32_t>(
                    impl_->prefill_slots + acquisitions[index].slot);
        }
        {
            std::scoped_lock lock(impl_->mutex);
            impl_->counters.wait_seconds += wait_seconds;
        }
        const auto view_begin = std::chrono::steady_clock::now();
        auto weights = impl_->arena.routed_weights(slot_map, unique);
        const auto view_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - view_begin).count();
        const auto prepare_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - prepare_begin).count();
        {
            std::scoped_lock lock(impl_->mutex);
            impl_->counters.view_seconds += view_seconds;
            impl_->counters.prepare_seconds += prepare_seconds;
        }
        auto impl = impl_;
        return MlxDeepseekV4SsdPreparedExperts(
            std::move(weights),
            [impl, acquisitions = std::move(acquisitions)]() mutable noexcept {
                impl->defer_release(std::move(acquisitions));
            });
    } catch (...) {
        impl_->release(acquisitions);
        throw;
    }
}

void MlxDeepseekV4SsdExpertCache::record_route_sync(
    double seconds) noexcept {
    std::scoped_lock lock(impl_->mutex);
    impl_->counters.route_sync_seconds += seconds;
}

MlxDeepseekV4SsdPrefetchedLayer
MlxDeepseekV4SsdExpertCache::prefetch_layer(std::size_t layer) {
    if (layer >= impl_->store.num_layers()) {
        throw std::out_of_range("SSD expert prefetch layer out of range");
    }
    const auto buffer = layer % 2;
    impl_->claim_prefill_buffer(buffer);
    auto state = std::make_unique<MlxDeepseekV4SsdPrefetchedLayer::Impl>();
    state->cache = impl_;
    state->layer = layer;
    state->buffer = buffer;
    state->slot_map.assign(impl_->store.num_experts(), -1);
    state->cache_pins.reserve(impl_->store.num_experts());
    state->reads.reserve(impl_->store.num_experts());
    try {
        for (std::size_t expert = 0;
             expert < impl_->store.num_experts();
             ++expert) {
            const auto id = static_cast<std::int32_t>(expert);
            auto resident = impl_->pin_if_resident(layer, id);
            if (resident.has_value()) {
                state->slot_map[expert] = static_cast<std::int32_t>(
                    impl_->prefill_slots + resident->slot);
                state->cache_pins.push_back(std::move(*resident));
                std::scoped_lock lock(impl_->mutex);
                ++impl_->counters.prefill_cache_hits;
            } else {
                const auto arena_slot = buffer * impl_->store.num_experts() + expert;
                state->slot_map[expert] = static_cast<std::int32_t>(arena_slot);
                state->reads.push_back(
                    impl_->schedule_direct(layer, id, arena_slot));
            }
        }
        return MlxDeepseekV4SsdPrefetchedLayer(std::move(state));
    } catch (...) {
        state->release();
        throw;
    }
}

std::size_t MlxDeepseekV4SsdExpertCache::cache_limit_bytes() const noexcept {
    return impl_->limit_bytes;
}

std::size_t MlxDeepseekV4SsdExpertCache::cache_slots() const noexcept {
    return impl_->slot_count;
}

bool MlxDeepseekV4SsdExpertCache::prefill_overlap_enabled() const noexcept {
    return impl_->prefill_slots != 0;
}

MlxDeepseekV4SsdCacheStats MlxDeepseekV4SsdExpertCache::stats() const {
    return impl_->stats();
}

void MlxDeepseekV4SsdExpertCache::reset_stats() {
    std::scoped_lock lock(impl_->mutex);
    impl_->counters = {};
}

void MlxDeepseekV4SsdExpertCache::prewarm_metal() {
    impl_->arena.prewarm_metal();
}

void MlxDeepseekV4SsdExpertCache::release_deferred() {
    impl_->release_deferred();
}

void MlxDeepseekV4SsdExpertCache::clear() {
    impl_->release_deferred();
    std::scoped_lock lock(impl_->mutex);
    for (auto& slot : impl_->slots) {
        if (slot.state == Impl::State::loading || slot.pins != 0) {
            throw std::runtime_error(
                "cannot clear SSD expert cache while experts are in use");
        }
    }
    impl_->resident.clear();
    for (auto& slot : impl_->slots) {
        slot = {};
    }
}

} // namespace mfq::metal
