#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal::detail {

// MLX graph construction is lazy and can be substantial for a full model.
// This scope deliberately accumulates only time spent inside explicit MLX
// evaluations.  CPU graph construction, tokenizer work, request queuing, and
// first-token sampling remain part of TTFT, but not prefill throughput.
inline thread_local double* active_mlx_evaluation_ms = nullptr;

struct ComponentTiming {
    double elapsed_ms = 0.0;
    std::size_t evaluations = 0;
};

class ComponentProfile {
public:
    void record(
        std::string_view component,
        double elapsed_ms) {
        auto& timing = timings_[std::string(component)];
        timing.elapsed_ms += elapsed_ms;
        ++timing.evaluations;
    }

    const std::map<std::string, ComponentTiming>&
    timings() const noexcept {
        return timings_;
    }

    double evaluated_ms() const noexcept {
        double total = 0.0;
        for (const auto& [_, timing] : timings_) {
            total += timing.elapsed_ms;
        }
        return total;
    }

private:
    std::map<std::string, ComponentTiming> timings_;
};

inline thread_local ComponentProfile* active_component_profile = nullptr;

inline bool component_profile_requested() noexcept {
    static const bool requested = [] {
        const char* value =
            std::getenv("MFQ_METAL_PROFILE_COMPONENTS");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return requested;
}

inline int component_profile_steps() noexcept {
    static const int steps = [] {
        const char* value =
            std::getenv("MFQ_METAL_PROFILE_STEPS");
        if (value == nullptr) {
            return 1;
        }
        return std::max(1, std::atoi(value));
    }();
    return steps;
}

inline int component_profile_skip_steps() noexcept {
    static const int steps = [] {
        const char* value =
            std::getenv("MFQ_METAL_PROFILE_SKIP_STEPS");
        return value == nullptr
            ? 0
            : std::max(0, std::atoi(value));
    }();
    return steps;
}

inline bool component_profile_active() noexcept {
    return active_component_profile != nullptr;
}

class ScopedComponentProfile {
public:
    explicit ScopedComponentProfile(
        ComponentProfile* profile) noexcept
        : previous_(active_component_profile) {
        active_component_profile = profile;
    }

    ScopedComponentProfile(
        const ScopedComponentProfile&) = delete;
    ScopedComponentProfile& operator=(
        const ScopedComponentProfile&) = delete;

    ~ScopedComponentProfile() {
        active_component_profile = previous_;
    }

private:
    ComponentProfile* previous_ = nullptr;
};

class ScopedMlxEvaluationTiming {
public:
    explicit ScopedMlxEvaluationTiming(
        double* elapsed_ms) noexcept
        : previous_(active_mlx_evaluation_ms) {
        active_mlx_evaluation_ms = elapsed_ms;
    }

    ScopedMlxEvaluationTiming(
        const ScopedMlxEvaluationTiming&) = delete;
    ScopedMlxEvaluationTiming& operator=(
        const ScopedMlxEvaluationTiming&) = delete;

    ~ScopedMlxEvaluationTiming() {
        active_mlx_evaluation_ms = previous_;
    }

private:
    double* previous_ = nullptr;
};

template <typename Function>
void measure_evaluation(Function&& function) {
    if (active_mlx_evaluation_ms == nullptr) {
        std::forward<Function>(function)();
        return;
    }
    const auto started =
        std::chrono::steady_clock::now();
    std::forward<Function>(function)();
    *active_mlx_evaluation_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started)
            .count();
}

inline void eval_with_timing(mlx::core::array& value) {
    measure_evaluation([&value]() { value.eval(); });
}

inline void eval_with_timing(
    std::vector<mlx::core::array> values) {
    measure_evaluation(
        [values = std::move(values)]() mutable {
            mlx::core::eval(std::move(values));
        });
}

template <typename Function>
void profile_evaluation(
    std::string_view component,
    Function&& function) {
    if (active_component_profile == nullptr) {
        return;
    }
    const auto started =
        std::chrono::steady_clock::now();
    std::forward<Function>(function)();
    active_component_profile->record(
        component,
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started)
            .count());
}

inline void profile_eval(
    std::string_view component,
    mlx::core::array& value) {
    profile_evaluation(
        component,
        [&value]() { value.eval(); });
}

inline void profile_eval(
    std::string_view component,
    std::vector<mlx::core::array> values) {
    profile_evaluation(
        component,
        [values = std::move(values)]() mutable {
            mlx::core::eval(std::move(values));
        });
}

}  // namespace mfq::metal::detail
