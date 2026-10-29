#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include "mfq/runtime.h"
#include <utility>

namespace mfq::metal {

using MlxResidentBudgetExceeded = MfqResourceExhausted;

template <typename Runtime>
std::size_t mlx_trim_expert_residency(Runtime& runtime, std::size_t bytes) {
    if (runtime.reclaimable_expert_bytes() <= bytes) return 0;
    if constexpr (requires { runtime.trim_preallocated_expert_cache(bytes); })
        if (const auto released = runtime.trim_preallocated_expert_cache(bytes)) return *released;
    const auto full = runtime.resident_full_expert_bytes();
    const auto cache = runtime.expert_cache_limit_bytes();
    if (cache > std::numeric_limits<std::size_t>::max() - full)
        throw MlxResidentBudgetExceeded("expert residency ceiling overflow");
    const auto released = runtime.set_expert_cache_limit(bytes);
    runtime.set_expert_cache_limit(full + cache);
    return released;
}

class MlxResidentBudgetScope {
public:
    explicit MlxResidentBudgetScope(
        std::function<void(std::size_t)> reserve,
        std::function<bool(std::size_t)> permits_optional = {})
        : previous_(std::move(handler())), previous_optional_(std::move(optional_handler())) {
        handler() = std::move(reserve);
        optional_handler() = std::move(permits_optional);
    }
    ~MlxResidentBudgetScope() {
        handler() = std::move(previous_);
        optional_handler() = std::move(previous_optional_);
    }
    MlxResidentBudgetScope(const MlxResidentBudgetScope&) = delete;
    MlxResidentBudgetScope& operator=(const MlxResidentBudgetScope&) = delete;
    static void reserve(std::size_t bytes) { if (handler()) handler()(bytes); }
    static bool enabled() { return static_cast<bool>(handler()); }
    static bool permits_optional(std::size_t bytes) {
        return !optional_handler() || optional_handler()(bytes);
    }
    static std::size_t& temporary_bytes() { static thread_local std::size_t value = 0; return value; }
private:
    static std::function<void(std::size_t)>& handler() {
        static thread_local std::function<void(std::size_t)> value;
        return value;
    }
    static std::function<bool(std::size_t)>& optional_handler() {
        static thread_local std::function<bool(std::size_t)> value;
        return value;
    }
    std::function<void(std::size_t)> previous_;
    std::function<bool(std::size_t)> previous_optional_;
};

class MlxResidentBudgetHold {
public:
    explicit MlxResidentBudgetHold(std::size_t bytes) : bytes_(bytes) {
        MlxResidentBudgetScope::reserve(bytes);
        MlxResidentBudgetScope::temporary_bytes() += bytes;
    }
    ~MlxResidentBudgetHold() { MlxResidentBudgetScope::temporary_bytes() -= bytes_; }
    MlxResidentBudgetHold(const MlxResidentBudgetHold&) = delete;
    MlxResidentBudgetHold& operator=(const MlxResidentBudgetHold&) = delete;
private:
    std::size_t bytes_;
};

}
