#pragma once

#include <chrono>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal::detail {

// MLX graph construction is lazy and can be substantial for a full model.
// This scope deliberately accumulates only time spent inside explicit MLX
// evaluations.  CPU graph construction, tokenizer work, request queuing, and
// first-token sampling remain part of TTFT, but not prefill throughput.
inline thread_local double* active_mlx_evaluation_ms = nullptr;

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

}  // namespace mfq::metal::detail
