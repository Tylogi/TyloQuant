#include "mlx_deepseek_v4_causal_lm.h"

#include <mlx/mlx.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    using mfq::metal::MlxDeepseekV4CausalLm;
    using mlx::core::Shape;
    using mlx::core::array;

    if (argc < 2 || argc > 5) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL_DIR [CACHE_MIB] [TOKEN_ID] [STEPS]\n";
        return 2;
    }
    try {
        const std::filesystem::path root(argv[1]);
        const std::size_t cache_mib = argc >= 3
            ? static_cast<std::size_t>(std::stoull(argv[2]))
            : 256;
        const std::int32_t token = argc >= 4
            ? static_cast<std::int32_t>(std::stol(argv[3]))
            : 0;
        const int steps = argc >= 5 ? std::stoi(argv[4]) : 1;
        if (steps <= 0) {
            throw std::invalid_argument("STEPS must be positive");
        }
        const auto load_begin = std::chrono::steady_clock::now();
        auto model = MlxDeepseekV4CausalLm::load_hf(
            root,
            128,
            cache_mib * 1024ull * 1024ull,
            8,
            false);
        const auto load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - load_begin).count();
        std::int32_t current = token;
        std::cout << std::fixed << std::setprecision(3)
                  << "load_seconds=" << load_seconds << '\n';
        for (int step = 0; step < steps; ++step) {
            const std::array<std::int32_t, 1> ids{current};
            const auto forward_begin = std::chrono::steady_clock::now();
            auto logits = step == 0
                ? model.forward(array(ids.data(), Shape{1, 1}), false)
                : model.decode(array(ids.data(), Shape{1, 1}));
            mlx::core::eval(logits);
            const auto seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - forward_begin).count();
            if (logits.dtype() != mlx::core::float32 ||
                logits.shape() != Shape{1, 1, 129280}) {
                throw std::runtime_error("unexpected HF model logits shape/dtype");
            }
            const auto* values = logits.data<float>();
            const auto* best = std::max_element(values, values + logits.size());
            double checksum = 0.0;
            double absolute_checksum = 0.0;
            if (!std::all_of(values, values + logits.size(), [](float value) {
                    return std::isfinite(value);
                })) {
                throw std::runtime_error("HF model produced non-finite logits");
            }
            current = static_cast<std::int32_t>(best - values);
            for (std::size_t index = 0; index < logits.size(); ++index) {
                checksum += values[index];
                absolute_checksum += std::fabs(values[index]);
            }
            std::cout << "step=" << step
                      << " seconds=" << seconds
                      << " token=" << current
                      << " logit=" << *best
                      << " checksum=" << checksum
                      << " abs_checksum=" << absolute_checksum << '\n';
            const auto step_stats = model.ssd_expert_cache_stats();
            if (step_stats) {
                std::cout << "expert_requests=" << step_stats->requests
                          << " expert_hits=" << step_stats->hits
                          << " expert_misses=" << step_stats->misses
                          << " expert_loads=" << step_stats->loads
                          << " expert_bytes_read=" << step_stats->bytes_read
                          << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "HF model smoke failed: " << error.what() << '\n';
        return 1;
    }
}
