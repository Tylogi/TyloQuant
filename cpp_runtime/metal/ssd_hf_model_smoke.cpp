#include "mlx_deepseek_v4_causal_lm.h"
#include "mlx_eval_timing.h"

#include <mlx/mlx.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
            std::max(128, steps + 1),
            cache_mib * 1024ull * 1024ull,
            8,
            false);
        const auto load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - load_begin).count();
        std::int32_t current = token;
        std::vector<std::int32_t> replay_inputs;
        std::vector<std::uint64_t> replay_hashes;
        replay_inputs.reserve(static_cast<std::size_t>(steps));
        replay_hashes.reserve(static_cast<std::size_t>(steps));
        std::cout << std::fixed << std::setprecision(3)
                  << "load_seconds=" << load_seconds << '\n';
        for (int step = 0; step < steps; ++step) {
            replay_inputs.push_back(current);
            const std::array<std::int32_t, 1> ids{current};
            const auto before_stats = model.ssd_expert_cache_stats();
            const int profile_skip =
                mfq::metal::detail::component_profile_skip_steps();
            const bool profile_components =
                mfq::metal::detail::component_profile_requested() &&
                step >= profile_skip &&
                step - profile_skip <
                    mfq::metal::detail::component_profile_steps();
            mfq::metal::detail::ComponentProfile component_profile;
            mfq::metal::detail::ScopedComponentProfile profile_scope(
                profile_components ? &component_profile : nullptr);
            const auto forward_begin = std::chrono::steady_clock::now();
            auto logits = step == 0
                ? model.forward(array(ids.data(), Shape{1, 1}), false)
                : model.decode(array(ids.data(), Shape{1, 1}));
            if (profile_components) {
                mfq::metal::detail::profile_eval("model.output", logits);
            } else {
                mlx::core::eval(logits);
            }
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
            std::uint64_t logit_hash = 1469598103934665603ull;
            const auto* logit_bytes =
                reinterpret_cast<const unsigned char*>(values);
            for (std::size_t index = 0;
                 index < logits.nbytes();
                 ++index) {
                logit_hash ^= logit_bytes[index];
                logit_hash *= 1099511628211ull;
            }
            replay_hashes.push_back(logit_hash);
            std::cout << "step_logit_hash=" << logit_hash << '\n';
            std::cout << "step=" << step
                      << " seconds=" << seconds
                      << " token=" << current
                      << " logit=" << *best
                      << " checksum=" << checksum
                      << " abs_checksum=" << absolute_checksum << '\n';
            const auto step_stats = model.ssd_expert_cache_stats();
            if (step_stats) {
                const auto& before = *before_stats;
                const auto delta_requests =
                    step_stats->requests - before.requests;
                const auto delta_hits = step_stats->hits - before.hits;
                const auto delta_misses = step_stats->misses - before.misses;
                const auto delta_loads = step_stats->loads - before.loads;
                const auto delta_bytes =
                    step_stats->bytes_read - before.bytes_read;
                const auto delta_reads =
                    step_stats->read_calls - before.read_calls;
                const double delta_wait =
                    step_stats->wait_seconds - before.wait_seconds;
                const double delta_io =
                    step_stats->io_seconds - before.io_seconds;
                const double delta_route_sync =
                    step_stats->route_sync_seconds -
                    before.route_sync_seconds;
                const double delta_prepare =
                    step_stats->prepare_seconds - before.prepare_seconds;
                const double delta_view =
                    step_stats->view_seconds - before.view_seconds;
                std::cout << "expert_requests=" << step_stats->requests
                          << " expert_hits=" << step_stats->hits
                          << " expert_misses=" << step_stats->misses
                          << " expert_loads=" << step_stats->loads
                          << " expert_bytes_read=" << step_stats->bytes_read
                          << '\n';
                std::cout << "expert_step_requests=" << delta_requests
                          << " expert_step_hits=" << delta_hits
                          << " expert_step_misses=" << delta_misses
                          << " expert_step_loads=" << delta_loads
                          << " expert_step_bytes_read=" << delta_bytes
                          << " expert_step_read_calls=" << delta_reads
                          << " expert_step_wait_seconds=" << delta_wait
                          << " expert_step_io_worker_seconds=" << delta_io
                          << " expert_step_route_sync_seconds="
                          << delta_route_sync
                          << " expert_step_prepare_seconds="
                          << delta_prepare
                          << " expert_step_view_seconds=" << delta_view
                          << " expert_step_wait_gbps="
                          << (delta_wait > 0.0
                                  ? static_cast<double>(delta_bytes) /
                                        delta_wait / 1.0e9
                                  : 0.0)
                          << " expert_resident="
                          << step_stats->resident_experts
                          << " expert_slots=" << step_stats->cache_slots
                          << '\n';
            }
            if (profile_components) {
                double evaluated_ms = 0.0;
                for (const auto& [_, timing] :
                     component_profile.timings()) {
                    evaluated_ms += timing.elapsed_ms;
                }
                std::cout << "component_profile step=" << step
                          << " evaluated_ms=" << evaluated_ms
                          << " wall_ms=" << seconds * 1000.0
                          << " unscoped_ms="
                          << std::max(
                                 0.0,
                                 seconds * 1000.0 - evaluated_ms)
                          << '\n';
                for (const auto& [name, timing] :
                     component_profile.timings()) {
                    std::cout << "component_cost step=" << step
                              << " name=" << name
                              << " ms=" << timing.elapsed_ms
                              << " calls=" << timing.evaluations
                              << '\n';
                }
            }
        }
        std::uint64_t sequence_hash = 1469598103934665603ull;
        for (const auto hash : replay_hashes) {
            for (unsigned shift = 0; shift < 64; shift += 8) {
                sequence_hash ^=
                    static_cast<unsigned char>(hash >> shift);
                sequence_hash *= 1099511628211ull;
            }
        }
        std::cout << "sequence_logit_hash=" << sequence_hash << '\n';
        const char* replay_value = std::getenv("MFQ_SSD_REPLAY_PROFILE");
        if (replay_value != nullptr && std::atoi(replay_value) != 0) {
            const auto before = *model.ssd_expert_cache_stats();
            std::array<double, 4> window_seconds{};
            std::size_t replay_mismatches = 0;
            for (int step = 0; step < steps; ++step) {
                const std::array<std::int32_t, 1> ids{
                    replay_inputs[static_cast<std::size_t>(step)]};
                const int profile_skip =
                    mfq::metal::detail::component_profile_skip_steps();
                const bool profile_components =
                    mfq::metal::detail::component_profile_requested() &&
                    step >= profile_skip &&
                    step - profile_skip <
                        mfq::metal::detail::component_profile_steps();
                mfq::metal::detail::ComponentProfile component_profile;
                mfq::metal::detail::ScopedComponentProfile profile_scope(
                    profile_components ? &component_profile : nullptr);
                const auto started = std::chrono::steady_clock::now();
                auto logits = step == 0
                    ? model.forward(array(ids.data(), Shape{1, 1}), false)
                    : model.decode(array(ids.data(), Shape{1, 1}));
                if (profile_components) {
                    mfq::metal::detail::profile_eval(
                        "model.output",
                        logits);
                } else {
                    mlx::core::eval(logits);
                }
                const auto elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                window_seconds[std::min(3, step / 64)] += elapsed;
                const auto* values = logits.data<float>();
                std::uint64_t hash = 1469598103934665603ull;
                const auto* bytes =
                    reinterpret_cast<const unsigned char*>(values);
                for (std::size_t index = 0;
                     index < logits.nbytes();
                     ++index) {
                    hash ^= bytes[index];
                    hash *= 1099511628211ull;
                }
                replay_mismatches += hash !=
                    replay_hashes[static_cast<std::size_t>(step)];
                if (profile_components) {
                    std::cout << "replay_component_profile step=" << step
                              << " evaluated_ms="
                              << component_profile.evaluated_ms()
                              << " wall_ms=" << elapsed * 1000.0
                              << '\n';
                    for (const auto& [name, timing] :
                         component_profile.timings()) {
                        std::cout << "replay_component_cost step=" << step
                                  << " name=" << name
                                  << " ms=" << timing.elapsed_ms
                                  << " calls=" << timing.evaluations
                                  << '\n';
                    }
                }
            }
            const auto after = *model.ssd_expert_cache_stats();
            for (int window = 0;
                 window < (steps + 63) / 64;
                 ++window) {
                const int count = std::min(64, steps - window * 64);
                std::cout << "replay_window=" << window * 64
                          << '-' << window * 64 + count - 1
                          << " seconds=" << window_seconds[window]
                          << " tok_s="
                          << static_cast<double>(count) /
                                 window_seconds[window]
                          << '\n';
            }
            std::cout << "replay_hits=" << after.hits - before.hits
                      << " replay_misses=" << after.misses - before.misses
                      << " replay_bytes_read="
                      << after.bytes_read - before.bytes_read
                      << " replay_wait_seconds="
                      << after.wait_seconds - before.wait_seconds
                      << " replay_route_sync_seconds="
                      << after.route_sync_seconds -
                             before.route_sync_seconds
                      << " replay_prepare_seconds="
                      << after.prepare_seconds - before.prepare_seconds
                      << " replay_view_seconds="
                      << after.view_seconds - before.view_seconds
                      << " replay_logit_mismatches="
                      << replay_mismatches << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "HF model smoke failed: " << error.what() << '\n';
        return 1;
    }
}
