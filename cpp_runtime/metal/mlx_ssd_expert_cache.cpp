#include "mlx_ssd_expert_cache.h"

#include "hf_safetensors_store.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
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

std::vector<std::string> main_layer_prefixes(std::size_t count) {
    std::vector<std::string> result;
    result.reserve(count);
    for (std::size_t layer = 0; layer < count; ++layer) {
        result.push_back("layers." + std::to_string(layer));
    }
    return result;
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
        std::shared_future<void> gate_up_ready;
        std::shared_future<void> ready;
    };

    enum class TaskPart {
        full,
        scales,
        gate,
        up,
        down,
    };

    struct ParallelLoad {
        std::size_t arena_slot = 0;
        std::size_t cache_slot = 0;
        std::uint64_t generation = 0;
        std::size_t layer = 0;
        std::int32_t expert = 0;
        std::shared_ptr<std::promise<void>> gate_up_completion;
        std::shared_ptr<std::promise<void>> completion;
        std::atomic<int> gate_parts{3};
        std::atomic<std::uint64_t> bytes{0};
        std::atomic<std::uint64_t> read_calls{0};
        std::atomic<std::uint64_t> io_nanoseconds{0};
        std::mutex error_mutex;
        std::exception_ptr error;
        std::once_flag failure_once;
        std::atomic<bool> gate_published{false};
    };

    struct Task {
        std::size_t arena_slot = 0;
        std::size_t layer = 0;
        std::int32_t expert = 0;
        std::shared_ptr<std::promise<void>> completion;
        TaskPart part = TaskPart::full;
        std::shared_ptr<ParallelLoad> parallel;
    };

    struct Acquisition {
        std::size_t slot = 0;
        std::uint64_t generation = 0;
        std::shared_future<void> gate_up_ready;
        std::shared_future<void> ready;
        bool cache_hit = false;
    };

    struct PageTable {
        std::uint64_t version = 1;
        std::uint64_t built_version = 0;
        std::shared_ptr<const std::vector<std::int32_t>> slot_map;
        std::shared_ptr<const std::vector<std::uint64_t>> generation_map;
        std::shared_ptr<const mlx::core::array> slot_ids;
        std::shared_ptr<const mlx::core::array> generations;
        std::shared_ptr<const mlx::core::array> readiness;
    };

    struct TransactionRoute {
        std::size_t layer = 0;
        mlx::core::array packed_expert_ids;
        std::shared_ptr<const std::vector<std::int32_t>> slot_map;
        std::shared_ptr<const std::vector<std::uint64_t>> generation_map;
    };

    Impl(
        std::filesystem::path root,
        std::vector<std::string> layer_prefixes,
        std::size_t num_experts,
        std::size_t bytes,
        std::size_t worker_count,
        bool overlap)
        : store(
              std::move(root),
              std::move(layer_prefixes),
              num_experts),
          slot_bytes(store.slot_bytes()),
          prefill_slots(overlap ? 2 * store.num_experts() : 0),
          total_slots(bytes / slot_bytes),
          slot_count(total_slots > prefill_slots
              ? total_slots - prefill_slots
              : 0),
          limit_bytes(total_slots * slot_bytes),
          arena(total_slots),
          slots(slot_count),
          page_tables(store.num_layers()),
          route_confidence(store.num_layers(), 0) {
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
        while (true) {
            auto found = resident.find(key);
            if (found != resident.end()) {
                auto& slot = slots[found->second];
                if (slot.state == State::ready ||
                    slot.state == State::loading) {
                    ++counters.hits;
                    if (slot.state == State::loading) {
                        ++counters.coalesced;
                    }
                    ++slot.pins;
                    slot.last_use = ++clock;
                    return {
                        .slot = found->second,
                        .generation = slot.generation,
                        .gate_up_ready = slot.gate_up_ready,
                        .ready = slot.ready,
                        .cache_hit = true,
                    };
                }
                resident.erase(found);
            }
            if (active_page_table_snapshots == 0) {
                break;
            }
            condition.wait(lock, [&] {
                return stopping || active_page_table_snapshots == 0;
            });
            if (stopping) {
                throw std::runtime_error("SSD expert cache is stopping");
            }
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
            invalidate_page_table(slot.key);
            resident.erase(slot.key);
            ++counters.evictions;
        }
        auto gate_up_completion =
            std::make_shared<std::promise<void>>();
        auto completion = std::make_shared<std::promise<void>>();
        slot.state = State::loading;
        slot.key = key;
        ++slot.generation;
        slot.last_use = ++clock;
        slot.pins = 1;
        slot.gate_up_ready =
            gate_up_completion->get_future().share();
        slot.ready = completion->get_future().share();
        resident.emplace(key, victim);
        ++counters.misses;
        auto load = std::make_shared<ParallelLoad>();
        load->arena_slot = prefill_slots + victim;
        load->cache_slot = victim;
        load->generation = slot.generation;
        load->layer = layer;
        load->expert = expert;
        load->gate_up_completion = gate_up_completion;
        load->completion = completion;
        for (const auto part : {
                 TaskPart::scales,
                 TaskPart::gate,
                 TaskPart::up,
             }) {
            tasks.push_back({
                .part = part,
                .parallel = load,
            });
        }
        const auto result = Acquisition{
            .slot = victim,
            .generation = slot.generation,
            .gate_up_ready = slot.gate_up_ready,
            .ready = slot.ready,
            .cache_hit = false,
        };
        lock.unlock();
        condition.notify_all();
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
            .gate_up_ready = slot.gate_up_ready,
            .ready = slot.ready,
            .cache_hit = true,
        };
    }

    void observe_route(std::size_t layer, bool all_hit) noexcept {
        if (layer >= route_confidence.size()) {
            return;
        }
        auto& confidence = route_confidence[layer];
        confidence = all_hit
            ? static_cast<std::uint8_t>(std::min<int>(15, confidence + 1))
            : 0;
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
                .layer = layer,
                .expert = expert,
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
        bool notify = false;
        {
            std::scoped_lock lock(mutex);
            for (const auto& acquired : deferred) {
                if (acquired.slot >= slots.size()) {
                    continue;
                }
                auto& slot = slots[acquired.slot];
                if (slot.generation == acquired.generation &&
                    slot.pins != 0) {
                    --slot.pins;
                }
            }
            deferred.clear();
            if (deferred_page_table_snapshots != 0) {
                active_page_table_snapshots -=
                    deferred_page_table_snapshots;
                deferred_page_table_snapshots = 0;
                notify = true;
            }
        }
        if (notify) {
            condition.notify_all();
        }
    }

    void begin_route_transaction() {
        release_deferred();
        std::scoped_lock lock(mutex);
        if (route_transaction_active || !transaction_routes.empty()) {
            throw std::logic_error(
                "SSD expert route transaction is already active");
        }
        route_transaction_active = true;
    }

    void defer_transaction_route(
        std::size_t layer,
        const mlx::core::array& packed_expert_ids,
        std::shared_ptr<const std::vector<std::int32_t>> slot_map,
        std::shared_ptr<const std::vector<std::uint64_t>> generation_map) {
        std::scoped_lock lock(mutex);
        if (!route_transaction_active) {
            throw std::logic_error(
                "SSD expert route transaction is not active");
        }
        transaction_routes.push_back({
            .layer = layer,
            .packed_expert_ids = packed_expert_ids,
            .slot_map = std::move(slot_map),
            .generation_map = std::move(generation_map),
        });
    }

    MlxDeepseekV4SsdRouteTransactionResult resolve_route_transaction() {
        std::vector<TransactionRoute> routes;
        {
            std::scoped_lock lock(mutex);
            if (!route_transaction_active) {
                throw std::logic_error(
                    "SSD expert route transaction is not active");
            }
            routes = std::move(transaction_routes);
            transaction_routes.clear();
            route_transaction_active = false;
        }

        MlxDeepseekV4SsdRouteTransactionResult result;
        result.routes.reserve(routes.size());
        for (const auto& route : routes) {
            std::vector<std::int32_t> experts;
            experts.reserve(route.packed_expert_ids.size());
            bool encoded_all_hit = true;
            const auto* encoded =
                route.packed_expert_ids.data<std::int32_t>();
            for (std::size_t index = 0;
                 index < route.packed_expert_ids.size();
                 ++index) {
                const auto expert = encoded[index] & 0xff;
                experts.push_back(expert);
                if (expert < 0 ||
                    static_cast<std::size_t>(expert) >=
                        route.slot_map->size() ||
                    (encoded[index] >> 8) - 1 != (*route.slot_map)[
                        static_cast<std::size_t>(expert)]) {
                    encoded_all_hit = false;
                }
            }
            std::sort(experts.begin(), experts.end());
            experts.erase(
                std::unique(experts.begin(), experts.end()),
                experts.end());
            result.routes.push_back({
                .layer = route.layer,
                .experts = std::move(experts),
                .all_hit = encoded_all_hit,
            });
        }

        {
            std::scoped_lock lock(mutex);
            for (std::size_t route_index = 0;
                 route_index < routes.size();
                 ++route_index) {
                auto& selection = result.routes[route_index];
                const auto& route = routes[route_index];
                for (const auto expert : selection.experts) {
                    if (!selection.all_hit) {
                        break;
                    }
                    if (expert < 0 ||
                        static_cast<std::size_t>(expert) >=
                            route.slot_map->size()) {
                        selection.all_hit = false;
                        break;
                    }
                    const auto arena_slot = (*route.slot_map)[
                        static_cast<std::size_t>(expert)];
                    if (arena_slot <
                        static_cast<std::int32_t>(prefill_slots)) {
                        selection.all_hit = false;
                        break;
                    }
                    const auto cache_slot = static_cast<std::size_t>(
                        arena_slot -
                        static_cast<std::int32_t>(prefill_slots));
                    const auto found = resident.find(
                        expert_key(route.layer, expert));
                    if (cache_slot >= slots.size() ||
                        found == resident.end() ||
                        found->second != cache_slot) {
                        selection.all_hit = false;
                        break;
                    }
                    const auto& slot = slots[cache_slot];
                    if (slot.state != State::ready ||
                        slot.key != expert_key(route.layer, expert) ||
                        slot.generation != (*route.generation_map)[
                            static_cast<std::size_t>(expert)]) {
                        selection.all_hit = false;
                        break;
                    }
                }
                ++counters.device_route_layers;
                counters.requests += selection.experts.size();
                if (selection.all_hit) {
                    ++counters.device_route_hits;
                    counters.hits += selection.experts.size();
                    for (const auto expert : selection.experts) {
                        const auto arena_slot = (*route.slot_map)[
                            static_cast<std::size_t>(expert)];
                        const auto cache_slot = static_cast<std::size_t>(
                            arena_slot -
                            static_cast<std::int32_t>(prefill_slots));
                        slots[cache_slot].last_use = ++clock;
                    }
                } else {
                    ++counters.device_route_misses;
                    result.all_hit = false;
                }
                observe_route(route.layer, selection.all_hit);
            }
            if (active_page_table_snapshots < routes.size()) {
                throw std::logic_error(
                    "SSD expert route transaction freeze underflow");
            }
            active_page_table_snapshots -= routes.size();
        }
        condition.notify_all();
        return result;
    }

    void cancel_route_transaction() noexcept {
        std::size_t route_count = 0;
        {
            std::scoped_lock lock(mutex);
            route_count = transaction_routes.size();
            transaction_routes.clear();
            route_transaction_active = false;
            if (active_page_table_snapshots >= route_count) {
                active_page_table_snapshots -= route_count;
            } else {
                active_page_table_snapshots = 0;
            }
        }
        condition.notify_all();
    }

    void fail_parallel(const std::shared_ptr<ParallelLoad>& load) noexcept {
        std::call_once(load->failure_once, [&] {
            std::exception_ptr error;
            {
                std::scoped_lock lock(load->error_mutex);
                error = load->error;
            }
            if (!error) {
                error = std::make_exception_ptr(std::runtime_error(
                    "parallel SSD expert read failed"));
            }
            {
                std::scoped_lock lock(mutex);
                auto& slot = slots[load->cache_slot];
                if (slot.generation == load->generation) {
                    slot.state = State::failed;
                }
                ++counters.failures;
            }
            if (!load->gate_published.exchange(true)) {
                try {
                    load->gate_up_completion->set_exception(error);
                } catch (...) {
                }
            }
            try {
                load->completion->set_exception(error);
            } catch (...) {
            }
        });
    }

    void finish_gate_part(const std::shared_ptr<ParallelLoad>& load) {
        if (load->gate_parts.fetch_sub(1) != 1) {
            return;
        }
        bool failed = false;
        {
            std::scoped_lock lock(load->error_mutex);
            failed = static_cast<bool>(load->error);
        }
        if (failed) {
            fail_parallel(load);
            return;
        }
        load->gate_published.store(true);
        load->gate_up_completion->set_value();
        {
            std::scoped_lock lock(mutex);
            tasks.push_back({
                .part = TaskPart::down,
                .parallel = load,
            });
        }
        condition.notify_one();
    }

    void run_parallel_part(
        TaskPart part,
        const std::shared_ptr<ParallelLoad>& load) {
        const auto begin = std::chrono::steady_clock::now();
        try {
            const auto destination = arena.destination(load->arena_slot);
            DeepseekV4NativeExpertLoadStats result;
            switch (part) {
            case TaskPart::scales:
                result = store.load_scales_scatter(
                    load->layer,
                    static_cast<std::size_t>(load->expert),
                    destination);
                break;
            case TaskPart::gate:
                result = store.load_gate_scatter(
                    load->layer,
                    static_cast<std::size_t>(load->expert),
                    destination);
                break;
            case TaskPart::up:
                result = store.load_up_scatter(
                    load->layer,
                    static_cast<std::size_t>(load->expert),
                    destination);
                break;
            case TaskPart::down:
                result = store.load_down_scatter(
                    load->layer,
                    static_cast<std::size_t>(load->expert),
                    destination);
                break;
            case TaskPart::full:
                throw std::logic_error("invalid parallel SSD task part");
            }
            const auto nanoseconds = std::chrono::duration_cast<
                std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - begin).count();
            load->bytes.fetch_add(result.bytes);
            load->read_calls.fetch_add(result.read_calls);
            load->io_nanoseconds.fetch_add(
                static_cast<std::uint64_t>(nanoseconds));
            if (part == TaskPart::down) {
                {
                    std::scoped_lock lock(mutex);
                    auto& slot = slots[load->cache_slot];
                    if (slot.generation != load->generation ||
                        slot.state != State::loading) {
                        throw std::logic_error(
                            "SSD expert slot changed while loading");
                    }
                    slot.state = State::ready;
                    invalidate_page_table(slot.key);
                    ++counters.loads;
                    counters.bytes_read += load->bytes.load();
                    counters.read_calls += load->read_calls.load();
                    counters.io_seconds += static_cast<double>(
                        load->io_nanoseconds.load()) / 1.0e9;
                }
                load->completion->set_value();
            }
        } catch (...) {
            {
                std::scoped_lock lock(load->error_mutex);
                if (!load->error) {
                    load->error = std::current_exception();
                }
            }
            if (part == TaskPart::down) {
                fail_parallel(load);
            }
        }
        if (part != TaskPart::down) {
            finish_gate_part(load);
        }
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
            if (task.parallel) {
                run_parallel_part(task.part, task.parallel);
                continue;
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
                    ++counters.prefill_expert_reads;
                    counters.prefill_bytes_read += result.bytes;
                    counters.io_seconds += seconds;
                }
                task.completion->set_value();
            } catch (...) {
                const auto error = std::current_exception();
                {
                    std::scoped_lock lock(mutex);
                    ++counters.failures;
                }
                try {
                    task.completion->set_exception(error);
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

    void invalidate_page_table(std::uint64_t key) noexcept {
        const auto layer = static_cast<std::size_t>(key >> 32u);
        if (layer < page_tables.size()) {
            ++page_tables[layer].version;
        }
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
    std::size_t active_page_table_snapshots = 0;
    std::size_t deferred_page_table_snapshots = 0;
    bool route_transaction_active = false;
    std::vector<TransactionRoute> transaction_routes;
    std::vector<PageTable> page_tables;
    std::vector<std::uint8_t> route_confidence;
    MlxDeepseekV4SsdCacheStats counters;
};

struct MlxDeepseekV4SsdPageTableSnapshot::Impl {
    std::shared_ptr<MlxDeepseekV4SsdExpertCache::Impl> cache;
    std::size_t layer = 0;
    std::shared_ptr<const std::vector<std::int32_t>> slot_map;
    std::shared_ptr<const std::vector<std::uint64_t>> generation_map;
    std::shared_ptr<const mlx::core::array> slot_ids;
    std::shared_ptr<const mlx::core::array> generations;
    std::shared_ptr<const mlx::core::array> readiness;
    bool released = false;

    void release_freeze() noexcept {
        if (released || !cache) {
            return;
        }
        {
            std::scoped_lock lock(cache->mutex);
            if (cache->active_page_table_snapshots != 0) {
                --cache->active_page_table_snapshots;
            }
        }
        cache->condition.notify_all();
        released = true;
    }

    void finish(
        std::span<const std::int32_t> active_experts,
        bool all_hit,
        double eval_seconds,
        double host_seconds,
        bool defer_current = false) {
        if (released || !cache) {
            throw std::logic_error(
                "SSD expert page-table snapshot is already released");
        }
        std::vector<std::int32_t> unique(
            active_experts.begin(), active_experts.end());
        unique.erase(
            std::remove_if(
                unique.begin(),
                unique.end(),
                [](std::int32_t expert) { return expert < 0; }),
            unique.end());
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());

        bool valid = true;
        {
            std::scoped_lock lock(cache->mutex);
            ++cache->counters.device_route_layers;
            cache->counters.device_route_eval_seconds += eval_seconds;
            cache->counters.device_route_host_seconds += host_seconds;
            if (all_hit) {
                for (const auto expert : unique) {
                    if (expert < 0 ||
                        static_cast<std::size_t>(expert) >= slot_map->size()) {
                        valid = false;
                        break;
                    }
                    const auto arena_slot =
                        (*slot_map)[static_cast<std::size_t>(expert)];
                    if (arena_slot <
                        static_cast<std::int32_t>(cache->prefill_slots)) {
                        valid = false;
                        break;
                    }
                    const auto cache_slot = static_cast<std::size_t>(
                        arena_slot -
                        static_cast<std::int32_t>(cache->prefill_slots));
                    const auto found = cache->resident.find(
                        expert_key(layer, expert));
                    if (cache_slot >= cache->slots.size() ||
                        found == cache->resident.end() ||
                        found->second != cache_slot) {
                        valid = false;
                        break;
                    }
                    const auto& slot = cache->slots[cache_slot];
                    if (slot.state !=
                            MlxDeepseekV4SsdExpertCache::Impl::State::ready ||
                        slot.key != expert_key(layer, expert) ||
                        slot.generation != (*generation_map)[
                            static_cast<std::size_t>(expert)]) {
                        valid = false;
                        break;
                    }
                }
            }
            if (all_hit && valid) {
                ++cache->counters.device_route_hits;
                cache->counters.requests += unique.size();
                cache->counters.hits += unique.size();
                for (const auto expert : unique) {
                    const auto arena_slot =
                        (*slot_map)[static_cast<std::size_t>(expert)];
                    const auto cache_slot = static_cast<std::size_t>(
                        arena_slot -
                        static_cast<std::int32_t>(cache->prefill_slots));
                    cache->slots[cache_slot].last_use = ++cache->clock;
                }
            } else {
                ++cache->counters.device_route_misses;
            }
            cache->observe_route(layer, all_hit && valid);

            // The snapshot graph has reached its evaluation boundary. Pins
            // deferred by the preceding layer can now return to the LRU.
            for (const auto& acquired : cache->deferred) {
                if (acquired.slot >= cache->slots.size()) {
                    continue;
                }
                auto& slot = cache->slots[acquired.slot];
                if (slot.generation == acquired.generation &&
                    slot.pins != 0) {
                    --slot.pins;
                }
            }
            cache->deferred.clear();
            if (cache->deferred_page_table_snapshots != 0) {
                cache->active_page_table_snapshots -=
                    cache->deferred_page_table_snapshots;
                cache->deferred_page_table_snapshots = 0;
            }
            if (defer_current && all_hit && valid) {
                ++cache->deferred_page_table_snapshots;
            } else if (cache->active_page_table_snapshots != 0) {
                --cache->active_page_table_snapshots;
            }
            released = true;
        }
        cache->condition.notify_all();
        if (all_hit && !valid) {
            throw std::runtime_error(
                "SSD expert page table changed before graph completion");
        }
    }

    void defer_transaction(const mlx::core::array& packed_expert_ids) {
        if (released || !cache) {
            throw std::logic_error(
                "SSD expert page-table snapshot is already released");
        }
        cache->defer_transaction_route(
            layer,
            packed_expert_ids,
            slot_map,
            generation_map);
        released = true;
    }

    ~Impl() {
        release_freeze();
    }
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
    std::vector<std::int32_t> slot_for_expert,
    std::function<void()> release)
    : weights_(std::make_unique<MlxDeepseekV4SsdExpertWeights>(
          std::move(weights))),
      slot_for_expert_(std::move(slot_for_expert)),
      release_(std::move(release)) {}

MlxDeepseekV4SsdPreparedExperts::MlxDeepseekV4SsdPreparedExperts(
    MlxDeepseekV4SsdPreparedExperts&& other) noexcept
    : weights_(std::move(other.weights_)),
      slot_for_expert_(std::move(other.slot_for_expert_)),
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
        slot_for_expert_ = std::move(other.slot_for_expert_);
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

std::span<const std::int32_t>
MlxDeepseekV4SsdPreparedExperts::slot_for_expert() const noexcept {
    return slot_for_expert_;
}

MlxDeepseekV4SsdPageTableSnapshot::MlxDeepseekV4SsdPageTableSnapshot(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MlxDeepseekV4SsdPageTableSnapshot::MlxDeepseekV4SsdPageTableSnapshot(
    MlxDeepseekV4SsdPageTableSnapshot&&) noexcept = default;

MlxDeepseekV4SsdPageTableSnapshot&
MlxDeepseekV4SsdPageTableSnapshot::operator=(
    MlxDeepseekV4SsdPageTableSnapshot&&) noexcept = default;

MlxDeepseekV4SsdPageTableSnapshot::~MlxDeepseekV4SsdPageTableSnapshot() =
    default;

const MlxDeepseekV4SsdExpertWeights&
MlxDeepseekV4SsdPageTableSnapshot::weights() const noexcept {
    return impl_->cache->arena.slot_weights();
}

const mlx::core::array&
MlxDeepseekV4SsdPageTableSnapshot::slot_ids() const noexcept {
    return *impl_->slot_ids;
}

const mlx::core::array&
MlxDeepseekV4SsdPageTableSnapshot::generations() const noexcept {
    return *impl_->generations;
}

const mlx::core::array&
MlxDeepseekV4SsdPageTableSnapshot::readiness() const noexcept {
    return *impl_->readiness;
}

std::span<const std::int32_t>
MlxDeepseekV4SsdPageTableSnapshot::slot_for_expert() const noexcept {
    return *impl_->slot_map;
}

void MlxDeepseekV4SsdPageTableSnapshot::finish(
    std::span<const std::int32_t> active_experts,
    bool all_hit,
    double eval_seconds,
    double host_seconds) {
    if (!impl_) {
        throw std::logic_error(
            "SSD expert page-table snapshot is empty");
    }
    impl_->finish(
        active_experts,
        all_hit,
        eval_seconds,
        host_seconds);
}

void MlxDeepseekV4SsdPageTableSnapshot::defer_finish(
    std::span<const std::int32_t> active_experts,
    double eval_seconds,
    double host_seconds) {
    if (!impl_) {
        throw std::logic_error(
            "SSD expert page-table snapshot is empty");
    }
    impl_->finish(
        active_experts,
        true,
        eval_seconds,
        host_seconds,
        true);
}

void MlxDeepseekV4SsdPageTableSnapshot::defer_transaction(
    const mlx::core::array& packed_expert_ids) {
    if (!impl_) {
        throw std::logic_error(
            "SSD expert page-table snapshot is empty");
    }
    impl_->defer_transaction(packed_expert_ids);
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
          main_layer_prefixes(43),
          256,
          cache_bytes,
          io_workers,
          prefill_overlap)) {}

MlxDeepseekV4SsdExpertCache::MlxDeepseekV4SsdExpertCache(
    std::filesystem::path model_root,
    std::vector<std::string> layer_prefixes,
    std::size_t cache_bytes,
    std::size_t io_workers,
    bool prefill_overlap,
    std::size_t num_experts)
    : impl_(std::make_shared<Impl>(
          std::move(model_root),
          std::move(layer_prefixes),
          num_experts,
          cache_bytes,
          io_workers,
          prefill_overlap)) {}

MlxDeepseekV4SsdExpertCache::~MlxDeepseekV4SsdExpertCache() = default;

MlxDeepseekV4SsdPreparedExperts MlxDeepseekV4SsdExpertCache::prepare(
    std::size_t layer,
    std::span<const std::int32_t> active_experts,
    std::function<void(
        const MlxDeepseekV4SsdExpertWeights&,
        std::span<const std::int32_t>,
        std::span<const std::int32_t>)> overlap,
    std::function<void(
        const MlxDeepseekV4SsdExpertWeights&,
        std::span<const std::int32_t>,
        std::span<const std::int32_t>)> gate_up_ready) {
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
        std::vector<std::int32_t> ready_experts;
        std::vector<std::int32_t> pending_experts;
        ready_experts.reserve(unique.size());
        pending_experts.reserve(unique.size());
        bool pending = false;
        for (std::size_t index = 0; index < acquisitions.size(); ++index) {
            if (acquisitions[index].ready.wait_for(
                    std::chrono::seconds(0)) ==
                std::future_status::ready) {
                ready_experts.push_back(unique[index]);
            } else {
                pending = true;
                pending_experts.push_back(unique[index]);
            }
        }
        std::vector<std::int32_t> slot_map(impl_->store.num_experts(), -1);
        for (std::size_t index = 0; index < unique.size(); ++index) {
            slot_map[static_cast<std::size_t>(unique[index])] =
                static_cast<std::int32_t>(
                    impl_->prefill_slots + acquisitions[index].slot);
        }
        const auto view_begin = std::chrono::steady_clock::now();
        const auto& weights = impl_->arena.slot_weights();
        const auto view_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - view_begin).count();
        if (pending && overlap) {
            overlap(weights, ready_experts, slot_map);
        }
        const auto gate_wait_begin = std::chrono::steady_clock::now();
        for (const auto& acquired : acquisitions) {
            acquired.gate_up_ready.get();
        }
        const auto gate_wait_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - gate_wait_begin).count();
        if (pending && gate_up_ready) {
            gate_up_ready(weights, pending_experts, slot_map);
        }
        const auto wait_begin = std::chrono::steady_clock::now();
        for (const auto& acquired : acquisitions) {
            acquired.ready.get();
        }
        const auto wait_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wait_begin).count();
        {
            std::scoped_lock lock(impl_->mutex);
            impl_->observe_route(
                layer,
                std::all_of(
                    acquisitions.begin(),
                    acquisitions.end(),
                    [](const Impl::Acquisition& acquisition) {
                        return acquisition.cache_hit;
                    }));
            impl_->counters.wait_seconds +=
                gate_wait_seconds + wait_seconds;
        }
        const auto prepare_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - prepare_begin).count();
        {
            std::scoped_lock lock(impl_->mutex);
            impl_->counters.view_seconds += view_seconds;
            impl_->counters.prepare_seconds += prepare_seconds;
        }
        auto impl = impl_;
        return MlxDeepseekV4SsdPreparedExperts(
            weights,
            std::move(slot_map),
            [impl, acquisitions = std::move(acquisitions)]() mutable noexcept {
                impl->defer_release(std::move(acquisitions));
            });
    } catch (...) {
        impl_->release(acquisitions);
        throw;
    }
}

MlxDeepseekV4SsdPageTableSnapshot
MlxDeepseekV4SsdExpertCache::snapshot_page_table(std::size_t layer) {
    if (layer >= impl_->store.num_layers()) {
        throw std::out_of_range(
            "SSD expert page-table layer out of range");
    }
    const auto expert_count = impl_->store.num_experts();
    auto state = std::make_unique<MlxDeepseekV4SsdPageTableSnapshot::Impl>();
    state->cache = impl_;
    state->layer = layer;
    try {
        std::scoped_lock lock(impl_->mutex);
        ++impl_->active_page_table_snapshots;
        auto& page_table = impl_->page_tables[layer];
        if (page_table.built_version != page_table.version) {
            auto slot_map = std::make_shared<std::vector<std::int32_t>>(
                expert_count,
                -1);
            auto generation_map =
                std::make_shared<std::vector<std::uint64_t>>(
                    expert_count,
                    0);
            std::vector<std::uint32_t> device_generations(
                expert_count,
                0);
            std::vector<std::int32_t> device_readiness(
                expert_count,
                0);
            for (std::size_t expert = 0; expert < expert_count; ++expert) {
                const auto found = impl_->resident.find(expert_key(
                    layer,
                    static_cast<std::int32_t>(expert)));
                if (found == impl_->resident.end()) {
                    continue;
                }
                const auto& slot = impl_->slots[found->second];
                if (slot.state != Impl::State::ready) {
                    continue;
                }
                (*slot_map)[expert] = static_cast<std::int32_t>(
                    impl_->prefill_slots + found->second);
                (*generation_map)[expert] = slot.generation;
                device_generations[expert] =
                    static_cast<std::uint32_t>(slot.generation);
                device_readiness[expert] = 1;
            }
            page_table.slot_ids = std::make_shared<mlx::core::array>(
                slot_map->begin(),
                mlx::core::Shape{static_cast<int>(expert_count)});
            page_table.generations = std::make_shared<mlx::core::array>(
                device_generations.begin(),
                mlx::core::Shape{static_cast<int>(expert_count)});
            page_table.readiness = std::make_shared<mlx::core::array>(
                device_readiness.begin(),
                mlx::core::Shape{static_cast<int>(expert_count)});
            page_table.slot_map = std::move(slot_map);
            page_table.generation_map = std::move(generation_map);
            page_table.built_version = page_table.version;
        }
        state->slot_map = page_table.slot_map;
        state->generation_map = page_table.generation_map;
        state->slot_ids = page_table.slot_ids;
        state->generations = page_table.generations;
        state->readiness = page_table.readiness;
        return MlxDeepseekV4SsdPageTableSnapshot(std::move(state));
    } catch (...) {
        state->release_freeze();
        throw;
    }
}

void MlxDeepseekV4SsdExpertCache::begin_route_transaction() {
    impl_->begin_route_transaction();
}

bool MlxDeepseekV4SsdExpertCache::route_transaction_active() const noexcept {
    std::scoped_lock lock(impl_->mutex);
    return impl_->route_transaction_active;
}

bool MlxDeepseekV4SsdExpertCache::route_layer_likely_hit(
    std::size_t layer) const noexcept {
    std::scoped_lock lock(impl_->mutex);
    return layer < impl_->route_confidence.size() &&
        impl_->route_confidence[layer] >= 8;
}

MlxDeepseekV4SsdRouteTransactionResult
MlxDeepseekV4SsdExpertCache::resolve_route_transaction() {
    return impl_->resolve_route_transaction();
}

void MlxDeepseekV4SsdExpertCache::cancel_route_transaction() noexcept {
    impl_->cancel_route_transaction();
}

void MlxDeepseekV4SsdExpertCache::record_route_timing(
    double sync_seconds,
    double sync_cpu_seconds,
    double host_seconds) noexcept {
    std::scoped_lock lock(impl_->mutex);
    impl_->counters.route_sync_seconds += sync_seconds;
    impl_->counters.route_sync_cpu_seconds += sync_cpu_seconds;
    impl_->counters.route_host_seconds += host_seconds;
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
    if (impl_->active_page_table_snapshots != 0) {
        throw std::runtime_error(
            "cannot clear SSD expert cache while a page table is active");
    }
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
    for (auto& page_table : impl_->page_tables) {
        ++page_table.version;
    }
    std::fill(
        impl_->route_confidence.begin(),
        impl_->route_confidence.end(),
        0);
}

} // namespace mfq::metal
