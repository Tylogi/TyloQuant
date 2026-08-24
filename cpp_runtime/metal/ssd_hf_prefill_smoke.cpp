#include "mlx_deepseek_v4_causal_lm.h"
#include "mlx_eval_timing.h"

#include <mlx/mlx.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    using mfq::metal::MlxDeepseekV4CausalLm;
    using mlx::core::Shape;
    using mlx::core::array;

    if (argc != 5 && argc != 6) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL_DIR CACHE_MIB TOKENS PREFILL_BUFFER [LOGITS_F32]\n";
        return 2;
    }
    try {
        const std::filesystem::path root(argv[1]);
        const auto cache_mib = static_cast<std::size_t>(
            std::stoull(argv[2]));
        const int tokens = std::stoi(argv[3]);
        const bool prefill_buffer = std::stoi(argv[4]) != 0;
        if (tokens < 2 || tokens > 8192) {
            throw std::invalid_argument("TOKENS must be in [2, 8192]");
        }
        auto model = MlxDeepseekV4CausalLm::load_hf(
            root,
            tokens + 8,
            cache_mib * 1024ull * 1024ull,
            8,
            prefill_buffer);
        std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens));
        for (int index = 0; index < tokens; ++index) {
            ids[static_cast<std::size_t>(index)] =
                static_cast<std::int32_t>((index * 17 + 3) % 129280);
        }
        const bool profile_components =
            mfq::metal::detail::component_profile_requested();
        mfq::metal::detail::ComponentProfile component_profile;
        mfq::metal::detail::ScopedComponentProfile component_scope(
            profile_components ? &component_profile : nullptr);
        const auto started = std::chrono::steady_clock::now();
        auto logits = model.prefill(
            array(ids.data(), Shape{1, tokens}),
            tokens,
            false);
        mlx::core::eval(logits);
        const auto seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        if (logits.shape() != Shape{1, 129280} ||
            logits.dtype() != mlx::core::float32) {
            throw std::runtime_error("unexpected prefill logits geometry");
        }
        const auto* values = logits.data<float>();
        if (argc == 6) {
            std::ofstream output(argv[5], std::ios::binary);
            output.write(
                reinterpret_cast<const char*>(values),
                static_cast<std::streamsize>(
                    logits.size() * sizeof(float)));
            if (!output) {
                throw std::runtime_error("cannot write prefill logits");
            }
        }
        double checksum = 0.0;
        double absolute = 0.0;
        for (std::size_t index = 0; index < logits.size(); ++index) {
            if (!std::isfinite(values[index])) {
                throw std::runtime_error("prefill produced non-finite logits");
            }
            checksum += values[index];
            absolute += std::fabs(values[index]);
        }
        const auto best = std::max_element(values, values + logits.size());
        const auto stats = model.ssd_expert_cache_stats();
        std::cout << std::fixed << std::setprecision(6)
                  << "tokens=" << tokens
                  << " prefill_buffer=" << static_cast<int>(prefill_buffer)
                  << " seconds=" << seconds
                  << " token=" << (best - values)
                  << " logit=" << *best
                  << " checksum=" << checksum
                  << " abs_checksum=" << absolute;
        if (stats.has_value()) {
            std::cout << " requests=" << stats->requests
                      << " hits=" << stats->hits
                      << " bytes_read=" << stats->bytes_read
                      << " prefill_layers=" << stats->prefill_layers
                      << " prefill_cache_hits="
                      << stats->prefill_cache_hits
                      << " prefill_expert_reads="
                      << stats->prefill_expert_reads
                      << " prefill_bytes_read="
                      << stats->prefill_bytes_read
                      << " prefill_wait_seconds="
                      << stats->prefill_wait_seconds
                      << " io_worker_seconds=" << stats->io_seconds;
        }
        std::cout << '\n';
        if (profile_components) {
            const auto evaluated_ms = component_profile.evaluated_ms();
            for (const auto& [name, timing] : component_profile.timings()) {
                std::cout << "component_cost phase=prefill"
                          << " name=" << name
                          << " ms=" << timing.elapsed_ms
                          << " calls=" << timing.evaluations
                          << " pct_evaluated="
                          << (evaluated_ms > 0.0
                                  ? 100.0 * timing.elapsed_ms / evaluated_ms
                                  : 0.0)
                          << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "HF prefill smoke failed: " << error.what() << '\n';
        return 1;
    }
}
