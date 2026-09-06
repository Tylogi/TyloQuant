#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <list>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mfq {

struct MoeCacheKey {
    int source = -1;
    int cohort = -1;
    int expert = -1;

    bool operator==(const MoeCacheKey & other) const noexcept {
        return source == other.source &&
            cohort == other.cohort &&
            expert == other.expert;
    }
};

struct MoeCacheKeyHash {
    size_t operator()(const MoeCacheKey & key) const noexcept {
        size_t value = static_cast<size_t>(static_cast<uint32_t>(key.source));
        value = value * 1315423911u +
            static_cast<size_t>(static_cast<uint32_t>(key.cohort));
        value = value * 1315423911u +
            static_cast<size_t>(static_cast<uint32_t>(key.expert));
        return value;
    }
};

struct MoeCacheSlotLease {
    int slot = -1;
    uint64_t generation = 0;
    bool hit = false;
    std::optional<MoeCacheKey> replaced;
};

class MoeCacheSlotBook {
public:
    explicit MoeCacheSlotBook(int capacity)
        : slots_(static_cast<size_t>(capacity)) {
        if (capacity <= 0) {
            throw std::invalid_argument("MoE cache capacity must be positive");
        }
        free_.reserve(static_cast<size_t>(capacity));
        for (int slot = capacity - 1; slot >= 0; --slot) {
            free_.push_back(slot);
        }
    }

    int capacity() const noexcept {
        return static_cast<int>(slots_.size());
    }

    int size() const noexcept {
        return static_cast<int>(by_key_.size());
    }

    int slot_for(const MoeCacheKey & key) const noexcept {
        const auto found = by_key_.find(key);
        return found == by_key_.end() ? -1 : found->second;
    }

    const std::optional<MoeCacheKey> & key_for(int slot) const {
        return checked(slot).key;
    }

    MoeCacheSlotLease acquire(const MoeCacheKey & key) {
        const auto found = by_key_.find(key);
        if (found != by_key_.end()) {
            Slot & state = checked(found->second);
            touch(found->second);
            return {found->second, state.generation, true, std::nullopt};
        }

        int slot = -1;
        std::optional<MoeCacheKey> replaced;
        if (!free_.empty()) {
            slot = free_.back();
            free_.pop_back();
        } else {
            for (const int candidate : lru_) {
                if (!slots_[static_cast<size_t>(candidate)].inflight) {
                    slot = candidate;
                    break;
                }
            }
            if (slot < 0) {
                throw std::runtime_error(
                    "all MoE cache slots are in-flight");
            }
            Slot & state = checked(slot);
            replaced = state.key;
            if (state.key.has_value()) {
                by_key_.erase(*state.key);
            }
            erase_lru(slot);
        }

        Slot & state = checked(slot);
        ++state.generation;
        state.key = key;
        state.inflight = false;
        by_key_.emplace(key, slot);
        append_lru(slot);
        return {slot, state.generation, false, replaced};
    }

    void touch(int slot) {
        Slot & state = checked(slot);
        if (!state.key.has_value()) {
            throw std::runtime_error("cannot touch an empty MoE cache slot");
        }
        erase_lru(slot);
        append_lru(slot);
    }

    void mark_inflight(int slot) {
        Slot & state = checked(slot);
        if (!state.key.has_value()) {
            throw std::runtime_error(
                "cannot mark an empty MoE cache slot in-flight");
        }
        state.inflight = true;
    }

    void clear_inflight(int slot) {
        checked(slot).inflight = false;
    }

    bool inflight(int slot) const {
        return checked(slot).inflight;
    }

private:
    struct Slot {
        std::optional<MoeCacheKey> key;
        uint64_t generation = 0;
        bool inflight = false;
        bool listed = false;
        std::list<int>::iterator position;
    };

    Slot & checked(int slot) {
        if (slot < 0 || slot >= capacity()) {
            throw std::out_of_range("MoE cache slot is out of range");
        }
        return slots_[static_cast<size_t>(slot)];
    }

    const Slot & checked(int slot) const {
        if (slot < 0 || slot >= capacity()) {
            throw std::out_of_range("MoE cache slot is out of range");
        }
        return slots_[static_cast<size_t>(slot)];
    }

    void erase_lru(int slot) {
        Slot & state = checked(slot);
        if (!state.listed) return;
        lru_.erase(state.position);
        state.listed = false;
    }

    void append_lru(int slot) {
        lru_.push_back(slot);
        Slot & state = checked(slot);
        state.position = std::prev(lru_.end());
        state.listed = true;
    }

    std::vector<Slot> slots_;
    std::vector<int> free_;
    std::list<int> lru_;
    std::unordered_map<MoeCacheKey, int, MoeCacheKeyHash> by_key_;
};

struct MoeArenaDemand {
    std::string signature;
    int64_t slot_bytes = 0;
    int minimum_slots = 0;
    int registered_experts = 0;
};

inline std::unordered_map<std::string, int> plan_moe_arena_slots(
        int64_t budget_bytes,
        const std::vector<MoeArenaDemand> & demands) {
    if (budget_bytes <= 0) {
        throw std::invalid_argument(
            "MoE cache budget must be positive");
    }
    if (demands.empty()) {
        throw std::invalid_argument(
            "MoE cache planner requires at least one arena");
    }

    std::unordered_set<std::string> signatures;
    std::unordered_map<std::string, int> result;
    int64_t minimum_bytes = 0;
    long double total_registered_bytes = 0.0L;
    for (const auto & demand : demands) {
        if (demand.signature.empty() ||
            !signatures.insert(demand.signature).second) {
            throw std::invalid_argument(
                "MoE cache arena signatures must be unique and non-empty");
        }
        if (demand.slot_bytes <= 0 ||
            demand.minimum_slots <= 0 ||
            demand.registered_experts < demand.minimum_slots) {
            throw std::invalid_argument(
                "invalid MoE cache arena demand");
        }
        if (demand.slot_bytes >
            std::numeric_limits<int64_t>::max() / demand.minimum_slots) {
            throw std::overflow_error(
                "MoE cache minimum byte count overflows int64");
        }
        minimum_bytes +=
            demand.slot_bytes * static_cast<int64_t>(demand.minimum_slots);
        total_registered_bytes +=
            static_cast<long double>(demand.slot_bytes) *
            static_cast<long double>(demand.registered_experts);
        result.emplace(demand.signature, demand.minimum_slots);
    }
    if (minimum_bytes > budget_bytes) {
        throw std::runtime_error(
            "MoE cache budget is below the minimum decode working set");
    }

    int64_t used_bytes = minimum_bytes;
    for (const auto & demand : demands) {
        const long double target_bytes =
            static_cast<long double>(budget_bytes) *
            (static_cast<long double>(demand.slot_bytes) *
             static_cast<long double>(demand.registered_experts)) /
            total_registered_bytes;
        const int target_slots = std::min(
            demand.registered_experts,
            std::max(
                demand.minimum_slots,
                static_cast<int>(target_bytes / demand.slot_bytes)));
        const int current = result.at(demand.signature);
        const int64_t additional_bytes =
            static_cast<int64_t>(target_slots - current) *
            demand.slot_bytes;
        if (additional_bytes <= budget_bytes - used_bytes) {
            result[demand.signature] = target_slots;
            used_bytes += additional_bytes;
        }
    }

    while (true) {
        const MoeArenaDemand * best = nullptr;
        long double best_deficit = -1.0L;
        for (const auto & demand : demands) {
            const int current = result.at(demand.signature);
            if (current >= demand.registered_experts ||
                demand.slot_bytes > budget_bytes - used_bytes) {
                continue;
            }
            const long double deficit =
                static_cast<long double>(
                    demand.registered_experts - current) /
                static_cast<long double>(demand.registered_experts);
            if (deficit > best_deficit) {
                best = &demand;
                best_deficit = deficit;
            }
        }
        if (best == nullptr) break;
        ++result.at(best->signature);
        used_bytes += best->slot_bytes;
    }

    return result;
}

}  // namespace mfq
