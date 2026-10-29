#include "mlx_tq_expert_cache.h"
#include "mlx_tq_rows.h"
#include "mfq_container.h"
#include "mlx_weight_residency.h"
#include "mlx_resource_telemetry.h"
#include "mlx_kernel_prepare.h"
#include "mfq/host_parallel.h"

#include <mlx/allocator.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <sys/mman.h>
#include <unistd.h>

namespace mfq::metal {
namespace {
using namespace mlx::core;
using Clock = std::chrono::steady_clock;

std::size_t aligned(std::size_t size) {
    if (size > SIZE_MAX - 255) throw std::overflow_error("TQ cache allocation is too large");
    return (size + 255) / 256 * 256;
}

struct Range { std::size_t offset = 0, bytes = 0; };

std::size_t pool_page() {
    static const auto page = std::max(std::size_t{4096}, std::size_t(getpagesize()));
    return page;
}

struct PoolStorage {
    void* data = MAP_FAILED;
    std::size_t bytes = 0;
    ~PoolStorage() {
        if (data != MAP_FAILED) munmap(data, bytes);
    }
};

struct PoolBuffer {
    std::shared_ptr<PoolStorage> storage;
    allocator::Buffer buffer{nullptr};
    ~PoolBuffer() { if (buffer.ptr()) allocator::release(buffer); }
};

struct Pool {
    using Sizes = std::multimap<std::size_t, std::size_t>;
    struct FreeRange { std::size_t bytes; Sizes::iterator size_entry; };
    array words = array(0, uint32);
    std::byte* data = nullptr;
    std::size_t capacity = 0;
    std::shared_ptr<PoolStorage> storage;
    std::map<std::size_t, FreeRange> free;
    Sizes sizes;

    void insert_free(std::size_t offset, std::size_t bytes) {
        const auto position = sizes.emplace(bytes, offset);
        try { free.emplace(offset, FreeRange{bytes, position}); }
        catch (...) { sizes.erase(position); throw; }
    }

    void erase_free(std::map<std::size_t, FreeRange>::iterator range) {
        sizes.erase(range->second.size_entry);
        free.erase(range);
    }

    void reset_free(std::size_t offset) {
        free.clear(); sizes.clear();
        if (offset < capacity) insert_free(offset, capacity - offset);
    }

    void allocate_storage(std::size_t bytes) {
        capacity = bytes / pool_page() * pool_page();
        if (!capacity || capacity / 4096 > std::size_t(INT_MAX))
            throw std::invalid_argument("invalid TQ expert pool capacity");
        if (capacity > mlx::core::metal::device(Device::gpu).mtl_device()->maxBufferLength())
            throw std::invalid_argument("TQ expert pool exceeds the Metal buffer size limit");
        storage = std::make_shared<PoolStorage>();
        storage->bytes = capacity;
        storage->data = mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (storage->data == MAP_FAILED) throw std::bad_alloc();
        bind_storage(capacity);
        reset_free(0);
    }

    void bind_storage(std::size_t bytes) {
        if (!storage || !bytes || bytes > storage->bytes || bytes % pool_page())
            throw std::invalid_argument("invalid TQ expert pool view");
        if (bytes > capacity)
            madvise(static_cast<std::byte*>(storage->data) + capacity, bytes - capacity, MADV_FREE_REUSE);
        auto view = std::make_shared<PoolBuffer>();
        view->storage = storage;
        view->buffer = allocator::make_buffer(storage->data, bytes);
        if (!view->buffer.ptr()) throw std::runtime_error("cannot register TQ expert pool with Metal");
        words = array(view->buffer, Shape{int(bytes / 4096), 1024}, uint32,
            [view](allocator::Buffer) {});
        capacity = bytes;
        data = reinterpret_cast<std::byte*>(words.data<std::uint32_t>());
    }

    bool fits(std::size_t bytes, std::size_t limit = SIZE_MAX) const {
        auto it = sizes.lower_bound(bytes);
        if (limit == SIZE_MAX) return it != sizes.end();
        for (; it != sizes.end(); ++it)
            if (it->second <= limit && bytes <= limit - it->second) return true;
        return false;
    }

    Range take(std::size_t bytes, std::size_t limit = SIZE_MAX) {
        auto it = sizes.lower_bound(bytes);
        if (limit != SIZE_MAX) while (it != sizes.end()
            && (it->second > limit || bytes > limit - it->second)) ++it;
        if (it == sizes.end()) throw std::runtime_error("TQ expert pool has no contiguous range");
        const Range result{it->second, bytes};
        const auto remaining = it->first - bytes;
        auto size_node = sizes.extract(it);
        auto range_node = free.extract(result.offset);
        if (remaining) {
            size_node.key() = remaining; size_node.mapped() = result.offset + bytes;
            const auto position = sizes.insert(std::move(size_node));
            range_node.key() = result.offset + bytes;
            range_node.mapped() = FreeRange{remaining, position};
            free.insert(std::move(range_node));
        }
        return result;
    }

    void release(Range range) {
        auto next = free.lower_bound(range.offset);
        if (next != free.begin()) {
            const auto previous = std::prev(next);
            if (previous->first + previous->second.bytes == range.offset) {
                range.offset = previous->first;
                range.bytes += previous->second.bytes;
                erase_free(previous);
            }
        }
        if (next != free.end() && range.offset + range.bytes == next->first) {
            range.bytes += next->second.bytes;
            erase_free(next);
        }
        insert_free(range.offset, range.bytes);
    }

    std::optional<Range> relocate(Range range, std::size_t limit) {
        if (fits(range.bytes, limit)) {
            const auto target = take(range.bytes, limit);
            std::memmove(data + target.offset, data + range.offset, range.bytes);
            release(range);
            return target;
        }
        const auto next = free.lower_bound(range.offset);
        if (next == free.begin()) return std::nullopt;
        const auto previous = std::prev(next);
        if (previous->first + previous->second.bytes != range.offset
            || previous->first > limit || range.bytes > limit - previous->first)
            return std::nullopt;
        const Range target{previous->first, range.bytes};
        std::memmove(data + target.offset, data + range.offset, range.bytes);
        erase_free(previous);
        release({target.offset + target.bytes, range.offset - target.offset});
        return target;
    }
};
}

void MlxTqExpertWeights::prepare(int top_k, float limit, bool gelu) const {
    auto* preparation = MlxKernelPreparation::current();
    if (!preparation) return;
    if (top_k < 1 || top_k > gate.experts())
        throw std::invalid_argument("invalid TQ preparation route count");
    auto rows_to_prepare = preparation->row_buckets();
    const auto maximum = *std::max_element(rows_to_prepare.begin(), rows_to_prepare.end());
    for (const int factor : {3, 9, 32, 64, 128}) {
        const auto boundary = (std::int64_t(gate.experts()) * factor + top_k - 1) / top_k;
        for (const auto requested : {boundary, std::max<std::int64_t>(256, boundary)}) {
            const int rows = int(std::min<std::int64_t>(maximum, requested));
            if (std::find(rows_to_prepare.begin(), rows_to_prepare.end(), rows) == rows_to_prepare.end())
                rows_to_prepare.push_back(rows);
        }
    }
    preparation->collect([&] {
        for (const int rows : rows_to_prepare) {
            const auto ids = zeros({rows, top_k}, int32);
            const auto input = zeros({rows, gate.input_size()}, float16);
            const auto hidden = gate.swiglu(up, input, ids, limit, gelu);
            preparation->add(hidden);
            preparation->add(down.routed_matmul(hidden, ids));
            preparation->add(down.routed_matmul_reduce(hidden, ids,
                full({rows, top_k}, 1.f / top_k, float32)));
            if (rows >= 2 && rows <= 16) {
                std::vector<int> routes(rows * top_k);
                for (std::size_t row = 0; row < routes.size(); ++row) routes[row] = int(row) % top_k;
                const auto plan = make_tq_reuse_plan(routes, rows, top_k, gate.experts());
                if (plan) {
                    const auto reused_ids = array(routes.begin(), {rows, top_k}, int32);
                    const auto reused = gate.swiglu(up, input, reused_ids, limit, gelu, &*plan);
                    preparation->add(reused);
                    preparation->add(down.routed_matmul(reused, reused_ids, &*plan));
                    preparation->add(down.routed_matmul_reduce(reused, reused_ids,
                        full({rows, top_k}, 1.f / top_k, float32), &*plan));
                }
            }
        }
    });
}

struct MlxTqExpertCache::Impl {
    struct Bank {
        std::string name;
        MfqMappedBytes mapping;
        std::unique_ptr<MlxMappedTqRows> rows;
        std::vector<MlxMappedTqRows::PackedRows> ranges;
        std::optional<MlxTqWeight> weight;
    };
    struct Layer {
        std::array<Bank, 3> banks;
        std::unique_ptr<MlxTqExpertWeights> weights;
        int experts = 0, cached = 0;
    };
    struct Entry {
        std::size_t layer = 0;
        int expert = 0;
        std::array<Range, 3> allocations;
        std::list<std::uint64_t>::iterator order;
    };
    struct Read {
        std::string record;
        std::size_t offset;
        std::vector<std::span<std::byte>> destinations;
        std::size_t bytes = 0;
    };
    struct HotRoute { std::size_t layer; array ids; };

    Impl(const MfqContainer& source, const std::vector<std::string>& prefixes,
            std::size_t budget, int io_workers) : source(source), io_workers(io_workers) {
        if (prefixes.empty() || prefixes.size() > std::size_t(UINT32_MAX) || io_workers < 1)
            throw std::invalid_argument("invalid TQ expert cache configuration");
        const auto meta = source.header().extra_json.find("tq_shapes");
        const auto shapes = meta == source.header().extra_json.end() ? nlohmann::json::object()
            : nlohmann::json::parse(meta->second);
        std::array<std::size_t, 3> largest{};
        layers.resize(prefixes.size());
        for (std::size_t index = 0; index < layers.size(); ++index) {
            auto& layer = layers[index];
            for (int projection = 0; projection < 3; ++projection) {
                auto& bank = layer.banks[projection];
                bank.name = prefixes[index] + ".mlp.experts."
                    + std::array<const char*, 3>{"gate", "up", "down"}[projection] + ".weight";
                if (source.record(bank.name).dtype != "TQ")
                    throw std::invalid_argument("TQ expert cache requires three TQ banks per layer");
                bank.mapping = source.map_record(bank.name, true);
                std::vector<int> shape;
                if (shapes.contains(bank.name)) shape = shapes.at(bank.name).get<std::vector<int>>();
                else {
                    bank.rows = std::make_unique<MlxMappedTqRows>(bank.mapping.view());
                    const auto& rows = *bank.rows;
                    if (!rows.expert_container()) throw std::invalid_argument("TQ expert bank lacks logical shape");
                    shape = {int(rows.expert_count()), int(rows.rows() / rows.expert_count()), rows.width()};
                }
                if (shape.size() != 3 || std::any_of(shape.begin(), shape.end(), [](int x) { return x < 1; })
                    || (projection && layer.experts != shape[0]))
                    throw std::invalid_argument("invalid TQ expert bank geometry");
                layer.experts = shape[0];
                if (!bank.rows) bank.rows = std::make_unique<MlxMappedTqRows>(bank.mapping.view(), shape[2]);
                if (bank.rows->rows() != std::int64_t(shape[0]) * shape[1]
                    || (bank.rows->expert_container() && bank.rows->expert_count() != std::size_t(shape[0])))
                    throw std::invalid_argument("TQ expert bank row count differs");
                for (int expert = 0; expert < layer.experts; ++expert) {
                    const auto span = bank.rows->packed_rows(std::int64_t(expert) * shape[1], shape[1]);
                    bank.ranges.push_back(span);
                    bank_capacity[projection] += aligned(span.bytes);
                    largest[projection] = std::max(largest[projection], aligned(span.bytes));
                }
                backing += source.record(bank.name).nbytes;
            }
        }
        metadata = 0;
        for (int projection = 0; projection < 3; ++projection) {
            const auto words = (largest[projection] + 3) / 4;
            auto temporary = array(allocator::malloc(words * 4), Shape{int(words)}, uint32);
            for (auto& layer : layers) {
                auto& bank = layer.banks[projection];
                const std::vector<std::uint64_t> offsets(layer.experts, 0);
                bank.weight.emplace(bank.rows->expert_container()
                    ? MlxTqWeight::from_expert_rows(*bank.rows, layer.experts, &temporary, offsets)
                    : MlxTqWeight::from_pool(bank.mapping.view(), temporary, offsets, bank.rows->width()));
                metadata += bank.weight->descriptors_.nbytes() + bank.weight->scales_.nbytes();
            }
        }
        bank_largest = largest;
        for (auto& capacity : bank_capacity) capacity = (capacity + pool_page() - 1) / pool_page() * pool_page();
        const auto capacities = pool_capacities(budget);
        allocated = metadata;
        for (int projection = 0; projection < 3; ++projection) {
            pools[projection].allocate_storage(capacities[projection]);
            allocated += pools[projection].capacity;
            for (auto& layer : layers) layer.banks[projection].weight->words_ = pools[projection].words;
        }
        for (auto& layer : layers) {
            layer.weights = std::make_unique<MlxTqExpertWeights>(MlxTqExpertWeights{
                *layer.banks[0].weight, *layer.banks[1].weight, *layer.banks[2].weight});
            const auto& weights = *layer.weights;
            if (weights.gate.input_size() != weights.up.input_size()
                || weights.gate.output_size() != weights.up.output_size()
                || weights.down.input_size() != weights.gate.output_size()
                || weights.down.output_size() != weights.gate.input_size())
                throw std::invalid_argument("TQ expert Gate/Up/Down dimensions differ");
        }
        pool_residency = MlxWeightResidency::retain_arrays(
            {pools[0].words, pools[1].words, pools[2].words}, allocated);
        requested_limit = budget;
        resources.set({0, 0, allocated.load(), backing, 0});
        preload_complete();
    }

    std::array<std::size_t, 3> pool_capacities(std::size_t budget) const {
        std::array<std::size_t, 3> smallest{}, capacity{};
        const auto page = pool_page();
        const auto device_limit = std::size_t(mlx::core::metal::device(Device::gpu).mtl_device()->maxBufferLength())
            / page * page;
        for (int p = 0; p < 3; ++p) {
            smallest[p] = (bank_largest[p] + page - 1) / page * page;
            capacity[p] = std::min({bank_capacity[p], device_limit,
                std::size_t(INT_MAX) * 4096 / page * page});
            if (smallest[p] > capacity[p])
                throw std::invalid_argument("TQ expert pool exceeds the Metal buffer size limit");
        }
        const auto minimum = std::accumulate(smallest.begin(), smallest.end(), std::size_t{0});
        if (budget < metadata || budget - metadata < minimum)
            throw std::invalid_argument("TQ expert cache budget cannot hold its metadata and one expert");
        const auto maximum = std::accumulate(capacity.begin(), capacity.end(), std::size_t{0});
        if (budget - metadata >= maximum) return capacity;
        const auto extra = budget - metadata - minimum, room = maximum - minimum;
        std::array<std::size_t, 3> result;
        for (int p = 0; p < 3; ++p) {
            result[p] = smallest[p] + std::size_t(static_cast<long double>(extra)
                * (capacity[p] - smallest[p]) / room) / page * page;
        }
        return result;
    }

    void bind_pools() {
        for (auto& layer : layers) {
            for (int p = 0; p < 3; ++p) layer.banks[p].weight->words_ = pools[p].words;
            layer.weights = std::make_unique<MlxTqExpertWeights>(MlxTqExpertWeights{
                *layer.banks[0].weight, *layer.banks[1].weight, *layer.banks[2].weight});
        }
    }

    void release_pools() {
        pool_residency.reset();
        for (auto& pool : pools) {
            pool.words = array(0, uint32); pool.data = nullptr; pool.capacity = 0;
            pool.storage.reset();
            pool.free.clear(); pool.sizes.clear();
        }
        bind_pools();
        mlx::core::clear_cache();
        allocated = metadata;
        resources.set({0, 0, metadata, backing, 0});
    }

    void allocate_pools(const std::array<std::size_t, 3>& capacities) {
        std::size_t bytes = metadata;
        for (int p = 0; p < 3; ++p) {
            pools[p].allocate_storage(capacities[p]);
            bytes += pools[p].capacity;
        }
        bind_pools();
        pool_residency = MlxWeightResidency::retain_arrays(
            {pools[0].words, pools[1].words, pools[2].words}, bytes);
        allocated = bytes;
        resources.set({0, 0, bytes, backing, 0});
        preload_complete();
    }

    std::size_t set_limit(std::size_t budget) {
        std::lock_guard lock(execution);
        return resize(budget, false);
    }

    std::size_t trim(std::size_t budget) {
        std::lock_guard lock(execution);
        if (budget >= allocated) return 0;
        auto minimum = metadata;
        for (auto bytes : bank_largest) minimum += (bytes + pool_page() - 1) / pool_page() * pool_page();
        return resize(std::max(minimum, budget), true);
    }

    std::size_t resize(std::size_t budget, bool preserve_ceiling) {
        const auto capacities = pool_capacities(budget);
        std::array<std::size_t, 3> previous;
        for (int p = 0; p < 3; ++p) previous[p] = pools[p].capacity;
        if (capacities == previous) {
            if (!preserve_ceiling) requested_limit = budget;
            return 0;
        }
        finish_pending();
        bool reusable = true;
        for (int p = 0; p < 3; ++p)
            reusable &= pools[p].storage && capacities[p] <= pools[p].storage->bytes;
        if (reusable) {
            const auto before = allocated.load();
            const std::vector<std::uint64_t> priority(order.begin(), order.end());
            for (const auto key : priority) {
                auto entry = entries.find(key);
                if (entry == entries.end()) continue;
                for (int p = 0; p < 3; ++p) {
                    const auto range = entry->second.allocations[p];
                    if (range.offset <= capacities[p] && range.bytes <= capacities[p] - range.offset) continue;
                    auto target = pools[p].relocate(range, capacities[p]);
                    while (!target) {
                        const auto victim = entries.find(order.back());
                        const bool current = victim == entry;
                        if (current) {
                            std::size_t occupied = 0;
                            for (const auto& [cached_key, cached_entry] : entries)
                                occupied += cached_entry.allocations[p].bytes;
                            if (occupied <= capacities[p]) {
                                compact(p);
                                target = entry->second.allocations[p];
                                break;
                            }
                        }
                        erase(victim);
                        evictions.fetch_add(1, std::memory_order_relaxed);
                        if (current) { entry = entries.end(); break; }
                        target = pools[p].relocate(range, capacities[p]);
                    }
                    if (entry == entries.end()) break;
                    entry->second.allocations[p] = *target;
                    layers[entry->second.layer].banks[p].weight->set_expert_word_offset(
                        entry->second.expert, target->offset / 4);
                }
            }
            pool_residency.reset();
            for (auto& layer : layers) {
                layer.weights.reset();
                for (auto& bank : layer.banks) bank.weight->words_ = array(0, uint32);
            }
            for (auto& pool : pools) pool.words = array(0, uint32);
            try {
                for (int p = 0; p < 3; ++p) pools[p].bind_storage(capacities[p]);
                bind_pools();
                pool_residency = MlxWeightResidency::retain_arrays(
                    {pools[0].words, pools[1].words, pools[2].words}, budget);
            } catch (...) {
                const auto failure = std::current_exception();
                for (int p = 0; p < 3; ++p) pools[p].bind_storage(previous[p]);
                bind_pools();
                pool_residency = MlxWeightResidency::retain_arrays(
                    {pools[0].words, pools[1].words, pools[2].words}, before);
                rebuild_free_ranges();
                std::rethrow_exception(failure);
            }
            rebuild_free_ranges();
            for (int p = 0; p < 3; ++p) if (capacities[p] < previous[p])
                madvise(pools[p].data + capacities[p], previous[p] - capacities[p], MADV_FREE_REUSABLE);
            allocated = metadata + std::accumulate(capacities.begin(), capacities.end(), std::size_t{0});
            resources.set({0, 0, allocated.load(), backing, 0});
            if (!preserve_ceiling) requested_limit = budget;
            preload_complete();
            return before > allocated ? before - allocated : 0;
        }
        while (!entries.empty()) erase(entries.begin());
        const auto before = allocated.load();
        release_pools();
        try { allocate_pools(capacities); }
        catch (...) {
            const auto failure = std::current_exception();
            release_pools();
            allocate_pools(previous);
            std::rethrow_exception(failure);
        }
        if (!preserve_ceiling) requested_limit = budget;
        return before > allocated ? before - allocated : 0;
    }

    void rebuild_free_ranges() {
        for (int p = 0; p < 3; ++p) {
            std::vector<Range> live;
            for (const auto& [key, entry] : entries) live.push_back(entry.allocations[p]);
            std::sort(live.begin(), live.end(), [](const auto a, const auto b) { return a.offset < b.offset; });
            auto& pool = pools[p];
            pool.free.clear(); pool.sizes.clear();
            std::size_t offset = 0;
            for (const auto range : live) {
                if (range.offset > offset) pool.insert_free(offset, range.offset - offset);
                offset = range.offset + range.bytes;
            }
            if (offset < pool.capacity) pool.insert_free(offset, pool.capacity - offset);
        }
    }

    void erase(std::unordered_map<std::uint64_t, Entry>::iterator entry) {
        --layers[entry->second.layer].cached;
        for (int p = 0; p < 3; ++p) {
            pools[p].release(entry->second.allocations[p]);
            used.fetch_sub(entry->second.allocations[p].bytes, std::memory_order_relaxed);
        }
        order.erase(entry->second.order);
        entries.erase(entry);
        cached.store(entries.size(), std::memory_order_relaxed);
    }

    void record_hot_route(const HotRoute& route) {
        auto& layer = layers[route.layer];
        std::array<bool, 1024> selected{};
        const auto* ids = route.ids.data<int>();
        for (std::size_t row = 0; row < route.ids.size(); ++row)
            if (ids[row] >= 0 && ids[row] < layer.experts) selected[ids[row]] = true;
        std::uint64_t count = 0;
        for (int expert = 0; expert < layer.experts; ++expert) if (selected[expert]) {
            ++count;
            const auto key = (std::uint64_t(route.layer) << 32) | std::uint32_t(expert);
            const auto found = entries.find(key);
            if (found == entries.end()) throw std::logic_error("TQ hot route was evicted before completion");
            order.splice(order.begin(), order, found->second.order);
        }
        requests.fetch_add(count, std::memory_order_relaxed);
        hits.fetch_add(count, std::memory_order_relaxed);
    }

    void drain_completed() {
        std::erase_if(hot_routes, [&](const auto& route) {
            if (!route.ids.is_available()) return false;
            record_hot_route(route);
            return true;
        });
        std::erase_if(hot_outputs, [](const auto& output) { return output.is_available(); });
    }

    void finish_pending(const array* extra = nullptr) {
        auto pending = hot_outputs;
        if (pending_output) pending.push_back(*pending_output);
        if (extra) pending.push_back(*extra);
        for (const auto& route : hot_routes) pending.push_back(route.ids);
        try { if (!pending.empty()) mlx::core::eval(pending); }
        catch (...) {
            const auto failure = std::current_exception();
            try { synchronize(default_stream(Device::gpu)); } catch (...) {}
            pending_output.reset(); hot_outputs.clear(); hot_routes.clear();
            std::rethrow_exception(failure);
        }
        for (const auto& route : hot_routes) record_hot_route(route);
        pending_output.reset(); hot_outputs.clear(); hot_routes.clear();
    }

    void preload_complete() {
        for (int p = 0; p < 3; ++p) if (pools[p].capacity < bank_capacity[p]) return;
        try {
            for (std::size_t index = 0; index < layers.size(); ++index) {
                std::vector<int> ids(layers[index].experts);
                std::iota(ids.begin(), ids.end(), 0);
                apply_locked(index, array(ids.begin(), Shape{1, int(ids.size())}),
                    [](const auto&, const array*) { return array(0); }, true);
            }
            finish_pending();
        } catch (...) {
            while (!entries.empty()) erase(entries.begin());
            throw;
        }
    }

    void compact(int projection) {
        std::vector<Entry*> sorted;
        for (auto& [key, entry] : entries) sorted.push_back(&entry);
        std::sort(sorted.begin(), sorted.end(), [&](auto* a, auto* b) {
            return a->allocations[projection].offset < b->allocations[projection].offset;
        });
        std::size_t next = 0;
        auto& pool = pools[projection];
        for (auto* entry : sorted) {
            auto& span = entry->allocations[projection];
            if (span.offset != next) std::memmove(pool.data + next, pool.data + span.offset, span.bytes);
            span.offset = next;
            layers[entry->layer].banks[projection].weight->set_expert_word_offset(entry->expert, next / 4);
            next += span.bytes;
        }
        pool.reset_free(next);
    }

    array apply(std::size_t index, const array& route_ids,
            const std::function<array(const MlxTqExpertWeights&, const array*)>& operation,
            const std::function<void(const MlxTqExpertWeights&, const array*)>* stage = nullptr,
            const array* parallel = nullptr) {
        std::lock_guard lock(execution);
        return apply_locked(index, route_ids, operation, false, stage, parallel);
    }

    array apply_locked(std::size_t index, const array& route_ids,
            const std::function<array(const MlxTqExpertWeights&, const array*)>& operation, bool preload,
            const std::function<void(const MlxTqExpertWeights&, const array*)>* stage = nullptr,
            const array* parallel = nullptr) {
        if (index >= layers.size() || route_ids.ndim() != 2 || route_ids.size() == 0
            || (route_ids.dtype() != int32 && route_ids.dtype() != uint32))
            throw std::invalid_argument("invalid TQ expert cache route window");
        auto& layer = layers[index];
        const bool scan = !preload && route_ids.shape(0) > 16;
        drain_completed();
        if (!preload && layer.cached == layer.experts && layer.experts <= 1024
            && route_ids.dtype() == int32 && route_ids.flags().row_contiguous
            && hot_outputs.size() < 256 && hot_routes.size() < 256) {
            try {
                const auto started = Clock::now();
                auto output = operation(*layer.weights, nullptr);
                hot_outputs.push_back(output);
                hot_routes.push_back({index, route_ids});
                graph_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-started).count());
                return output;
            } catch (...) {
                const auto failure = std::current_exception();
                try { finish_pending(); } catch (...) {}
                std::rethrow_exception(failure);
            }
        }
        const auto sync_start = Clock::now();
        auto host = contiguous(astype(route_ids, int32));
        finish_pending(&host);
        route_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-sync_start).count());
        const auto prepared_from = Clock::now();
        std::vector<int> active;
        for (std::size_t row = 0; row < host.size(); ++row) {
            const int expert = host.data<int>()[row];
            if (expert >= 0 && expert < layer.experts) active.push_back(expert);
        }
        std::sort(active.begin(), active.end());
        active.erase(std::unique(active.begin(), active.end()), active.end());
        std::unordered_set<std::uint64_t> protected_keys;
        std::array<std::size_t, 3> required{};
        for (const int expert : active) {
            protected_keys.insert((std::uint64_t(index) << 32) | std::uint32_t(expert));
            for (int p = 0; p < 3; ++p) required[p] += aligned(layer.banks[p].ranges[expert].bytes);
        }
        for (int p = 0; p < 3; ++p) if (required[p] > pools[p].capacity)
            throw std::runtime_error("TQ expert cache budget cannot hold this routed window");
        std::vector<std::uint64_t> added;
        added.reserve(active.size());
        std::array<std::vector<Read>, 3> pending;
        std::optional<array> reuse;
        bool stage_started = false;
        try {
            for (const int expert : active) {
                if (!preload) requests.fetch_add(1, std::memory_order_relaxed);
                const auto key = (std::uint64_t(index) << 32) | std::uint32_t(expert);
                auto found = entries.find(key);
                if (found != entries.end()) {
                    if (!preload) hits.fetch_add(1, std::memory_order_relaxed);
                    order.splice(order.begin(), order, found->second.order);
                    continue;
                }
                if (!preload) misses.fetch_add(1, std::memory_order_relaxed);
                std::array<std::size_t, 3> sizes;
                for (int p = 0; p < 3; ++p) sizes[p] = aligned(layer.banks[p].ranges[expert].bytes);
                auto fits = [&] {
                    for (int p = 0; p < 3; ++p) if (!pools[p].fits(sizes[p])) return false;
                    return true;
                };
                while (!fits()) {
                    auto victim = std::find_if(order.rbegin(), order.rend(), [&](auto key) { return !protected_keys.contains(key); });
                    if (victim == order.rend()) {
                        for (int p = 0; p < 3; ++p) compact(p);
                        if (!fits()) throw std::runtime_error("TQ expert cache working set is fragmented");
                        break;
                    }
                    erase(entries.find(*victim));
                    evictions.fetch_add(1, std::memory_order_relaxed);
                }
                const auto position = order.insert(scan ? order.end() : order.begin(), key);
                Entry entry{index, expert, {}, position};
                int taken = 0;
                try {
                    for (; taken < 3; ++taken) {
                        entry.allocations[taken] = pools[taken].take(sizes[taken]);
                        used.fetch_add(sizes[taken], std::memory_order_relaxed);
                    }
                    entries.emplace(key, entry);
                    ++layer.cached;
                } catch (...) {
                    for (int p = 0; p < taken; ++p) {
                        pools[p].release(entry.allocations[p]);
                        used.fetch_sub(sizes[p], std::memory_order_relaxed);
                    }
                    order.erase(position);
                    throw;
                }
                added.push_back(key);
            }
            for (const auto key : added) {
                const auto& entry = entries.at(key);
                for (int p = 0; p < 3; ++p) {
                    const auto range = layer.banks[p].ranges[entry.expert];
                    pending[p].push_back({layer.banks[p].name, range.offset,
                        {{pools[p].data + entry.allocations[p].offset, range.bytes}}, range.bytes});
                }
            }
            std::vector<Read> reads;
            std::size_t gate_up_reads = 0;
            for (auto& bank : pending) for (auto& read : bank) {
                if (!reads.empty() && reads.back().record == read.record
                    && reads.back().offset + reads.back().bytes == read.offset) {
                    reads.back().destinations.push_back(read.destinations.front());
                    reads.back().bytes += read.bytes;
                } else reads.push_back(std::move(read));
            }
            for (const auto& read : reads) if (read.record != layer.banks[2].name) ++gate_up_reads;
            if (!preload) reuse = make_tq_reuse_plan(
                {host.data<int>(), host.size()}, route_ids.shape(0), route_ids.shape(1), layer.experts);
            const auto started = Clock::now();
            prepare_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(started-prepared_from).count());
            const auto read_range = [&](std::size_t begin, std::size_t end) {
                const auto read_started = Clock::now();
                workers.run(begin, end, 1, io_workers, [&](auto first, auto last) {
                    for (auto r = first; r < last; ++r) {
                        const auto& read = reads[r];
                        source.readv_range(read.record, read.offset, read.destinations);
                        bytes_read.fetch_add(read.bytes, std::memory_order_relaxed);
                        read_calls.fetch_add(1, std::memory_order_relaxed);
                    }
                });
                io_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-read_started).count());
            };
            const auto publish_offsets = [&](int first, int last) {
                for (const auto key : added) {
                    const auto& entry = entries.at(key);
                    for (int p = first; p < last; ++p)
                        layer.banks[p].weight->set_expert_word_offset(entry.expert, entry.allocations[p].offset / 4);
                }
            };
            if (stage && !preload && !scan && gate_up_reads < reads.size()) {
                if (parallel && !parallel->is_available()) {
                    stage_started = true;
                    const auto graph_started = Clock::now();
                    async_eval({*parallel});
                    graph_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-graph_started).count());
                }
                read_range(0, gate_up_reads);
                publish_offsets(0, 2);
                stage_started = true;
                const auto graph_started = Clock::now();
                (*stage)(*layer.weights, reuse ? &*reuse : nullptr);
                graph_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-graph_started).count());
                overlapped_read_batches.fetch_add(1, std::memory_order_relaxed);
                read_range(gate_up_reads, reads.size());
                publish_offsets(2, 3);
            } else {
                read_range(0, reads.size());
                publish_offsets(0, 3);
            }
            const auto read_done = Clock::now();
            cached.store(entries.size(), std::memory_order_relaxed);
            prepare_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-read_done).count());
        } catch (...) {
            if (stage_started) {
                const auto failure = std::current_exception();
                try { synchronize(default_stream(Device::gpu)); } catch (...) {}
                for (auto key : added) if (auto found = entries.find(key); found != entries.end()) erase(found);
                std::rethrow_exception(failure);
            }
            for (auto key : added) if (auto found = entries.find(key); found != entries.end()) erase(found);
            throw;
        }
        try {
            const auto started = Clock::now();
            auto output = operation(*layer.weights, reuse ? &*reuse : nullptr);
            graph_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-started).count());
            pending_output = output;
            return output;
        } catch (...) {
            const auto failure = std::current_exception();
            try { synchronize(default_stream(Device::gpu)); } catch (...) {}
            std::rethrow_exception(failure);
        }
    }

    MfqContainer source;
    int io_workers;
    mfq::HostParallelPool workers{{}, false};
    std::mutex execution;
    std::vector<Layer> layers;
    std::array<Pool, 3> pools;
    std::list<std::uint64_t> order;
    std::unordered_map<std::uint64_t, Entry> entries;
    std::optional<array> pending_output;
    std::vector<array> hot_outputs;
    std::vector<HotRoute> hot_routes;
    std::size_t metadata = 0, backing = 0;
    std::array<std::size_t, 3> bank_largest{}, bank_capacity{};
    std::unique_ptr<MlxWeightResidency> pool_residency;
    std::atomic<std::size_t> allocated{0}, requested_limit{0};
    std::atomic<std::uint64_t> requests{0}, hits{0}, misses{0}, evictions{0}, bytes_read{0}, read_calls{0};
    std::atomic<std::uint64_t> overlapped_read_batches{0};
    std::atomic<std::uint64_t> io_ns{0}, route_ns{0}, prepare_ns{0}, graph_ns{0};
    std::atomic<std::size_t> used{0}, cached{0};
    MlxResourceTelemetry resources;
};

MlxTqExpertCache::MlxTqExpertCache(const MfqContainer& source, const std::vector<std::string>& prefixes,
        std::size_t budget, int io_workers) : impl_(std::make_shared<Impl>(source, prefixes, budget, io_workers)) {}

array MlxTqExpertCache::apply(std::size_t layer, const array& ids,
        const std::function<array(const MlxTqExpertWeights&)>& operation) {
    return impl_->apply(layer, ids, [&](const auto& weights, const array*) { return operation(weights); });
}

array MlxTqExpertCache::apply(std::size_t layer, const array& ids,
        const std::function<array(const MlxTqExpertWeights&, const array*)>& operation) {
    return impl_->apply(layer, ids, operation);
}

array MlxTqExpertCache::apply(std::size_t layer, const array& ids,
        const std::function<array(const MlxTqExpertWeights&, const array*)>& gate_up,
        const std::function<array(const MlxTqExpertWeights&, const array&, const array*)>& down,
        const array* parallel) {
    std::optional<array> intermediate;
    const std::function<void(const MlxTqExpertWeights&, const array*)> stage = [&](const auto& weights, const array* reuse) {
        intermediate = gate_up(weights, reuse);
        async_eval({*intermediate});
    };
    return impl_->apply(layer, ids, [&](const auto& weights, const array* reuse) {
        auto hidden = intermediate ? *intermediate : gate_up(weights, reuse);
        return down(weights, hidden, reuse);
    }, &stage, parallel);
}

MlxTqExpertCacheStats MlxTqExpertCache::stats() const {
    std::lock_guard lock(impl_->execution);
    impl_->drain_completed();
    return {impl_->requests.load(), impl_->hits.load(), impl_->misses.load(), impl_->evictions.load(),
        impl_->bytes_read.load(), impl_->read_calls.load(), impl_->cached.load(), impl_->used.load(),
        impl_->allocated.load(), impl_->io_ns.load() * 1e-9, impl_->route_ns.load() * 1e-9,
        impl_->prepare_ns.load() * 1e-9, impl_->graph_ns.load() * 1e-9, impl_->metadata,
        impl_->overlapped_read_batches.load()};
}
std::size_t MlxTqExpertCache::limit_bytes() const noexcept { return impl_->requested_limit.load(); }
std::size_t MlxTqExpertCache::backing_bytes() const noexcept { return impl_->backing; }
std::size_t MlxTqExpertCache::resident_bytes() const noexcept { return impl_->allocated; }
std::size_t MlxTqExpertCache::set_limit(std::size_t bytes) { return impl_->set_limit(bytes); }
std::size_t MlxTqExpertCache::trim(std::size_t bytes) { return impl_->trim(bytes); }
void MlxTqExpertCache::validate_layer(std::size_t layer, int experts, int input, int intermediate) const {
    if (layer >= impl_->layers.size()) throw std::invalid_argument("invalid TQ cache layer");
    const auto& weights = *impl_->layers[layer].weights;
    if (weights.gate.experts() != experts || weights.gate.input_size() != input
        || weights.gate.output_size() != intermediate)
        throw std::invalid_argument("TQ cache layer geometry differs");
}
void MlxTqExpertCache::prepare_layer(std::size_t layer, int top_k, float limit, bool gelu) const {
    std::lock_guard lock(impl_->execution);
    if (layer >= impl_->layers.size()) throw std::invalid_argument("invalid TQ cache layer");
    impl_->layers[layer].weights->prepare(top_k, limit, gelu);
}
void MlxTqExpertCache::clear() {
    std::lock_guard lock(impl_->execution);
    impl_->finish_pending();
    while (!impl_->entries.empty()) impl_->erase(impl_->entries.begin());
}

} // namespace mfq::metal
