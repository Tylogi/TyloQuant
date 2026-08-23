#include "hf_safetensors_store.h"
#include "mlx_ssd_expert_arena.h"
#include "mlx_ssd_expert_cache.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mlx::core::Shape;
using mlx::core::array;

array input(int columns) {
    std::vector<float> values(static_cast<std::size_t>(columns));
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = static_cast<float>(
            static_cast<int>((index * 13) % 37) - 18) / 32.0f;
    }
    return mlx::core::astype(
        array(values.begin(), Shape{1, columns}),
        mlx::core::float16);
}

float maximum_error(const array& left, const array& right) {
    auto lhs = mlx::core::contiguous(
        mlx::core::astype(left, mlx::core::float32));
    auto rhs = mlx::core::contiguous(
        mlx::core::astype(right, mlx::core::float32));
    mlx::core::eval({lhs, rhs});
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error("validation shape mismatch");
    }
    float result = 0.0f;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        result = std::max(
            result,
            std::abs(
                lhs.data<float>()[index] - rhs.data<float>()[index]));
    }
    return result;
}

int integer(const char* text, const char* name) {
    try {
        std::size_t consumed = 0;
        const auto value = std::stoi(text, &consumed);
        if (consumed != std::string(text).size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 4) {
            throw std::runtime_error(
                "usage: mfq-metal-ssd-expert-real-smoke MODEL [LAYER] [EXPERT]");
        }
        const int layer = argc >= 3 ? integer(argv[2], "layer") : 0;
        const int expert = argc >= 4 ? integer(argv[3], "expert") : 0;
        if (layer < 0 || layer >= 43 || expert < 0 || expert >= 256) {
            throw std::runtime_error("layer/expert is out of range");
        }
        mfq::metal::DeepseekV4NativeExpertStore store(argv[1], 43, 256);
        mfq::metal::MlxDeepseekV4SsdExpertArena arena(1);
        store.checkpoint().drop_file_cache();
        const auto begin = std::chrono::steady_clock::now();
        const auto stats = store.load_scatter(
            static_cast<std::size_t>(layer),
            static_cast<std::size_t>(expert),
            arena.destination(0));
        const auto load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();

        std::vector<std::int32_t> map(256, -1);
        map[static_cast<std::size_t>(expert)] = 0;
        const std::vector<std::int32_t> active{expert};
        auto weights = arena.routed_weights(map, active);
        const array ids(
            std::vector<std::int32_t>{expert}.begin(),
            Shape{1, 1});

        auto hidden = input(4096);
        const auto gate_error = maximum_error(
            weights.gate.forward(hidden, ids),
            mlx::core::expand_dims(
                arena.expert_weight(0, '1').matmul(hidden),
                1));
        const auto up_error = maximum_error(
            weights.up.forward(hidden, ids),
            mlx::core::expand_dims(
                arena.expert_weight(0, '3').matmul(hidden),
                1));
        auto intermediate = input(2048);
        const auto down_error = maximum_error(
            weights.down.forward(
                mlx::core::expand_dims(intermediate, 1),
                ids),
            mlx::core::expand_dims(
                arena.expert_weight(0, '2').matmul(intermediate),
                1));

        std::cout << "layer=" << layer
                  << " expert=" << expert
                  << " bytes=" << stats.bytes
                  << " read_calls=" << stats.read_calls
                  << " load_ms=" << std::fixed << std::setprecision(3)
                  << load_seconds * 1000.0
                  << " gate_max_error=" << gate_error
                  << " up_max_error=" << up_error
                  << " down_max_error=" << down_error << '\n';
        if (gate_error != 0.0f || up_error != 0.0f || down_error != 0.0f) {
            throw std::runtime_error("official MXFP4 routed validation failed");
        }

        mfq::metal::MlxDeepseekV4SsdExpertCache cache(
            argv[1],
            8 * store.slot_bytes(),
            6);
        std::vector<std::int32_t> cache_active;
        for (int index = 0; index < 6; ++index) {
            cache_active.push_back((expert + index) % 256);
        }
        const array cache_ids(cache_active.begin(), Shape{1, 6});
        array first = mlx::core::array(0.0f);
        {
            auto prepared = cache.prepare(
                static_cast<std::size_t>(layer),
                cache_active);
            first = mlx::core::contiguous(
                prepared.weights().gate.forward(hidden, cache_ids));
            first.eval();
        }
        array second = mlx::core::array(0.0f);
        {
            auto prepared = cache.prepare(
                static_cast<std::size_t>(layer),
                cache_active);
            second = mlx::core::contiguous(
                prepared.weights().gate.forward(hidden, cache_ids));
            second.eval();
        }
        const auto cache_error = maximum_error(first, second);
        const auto cache_stats = cache.stats();
        std::cout << "cache_slots=" << cache_stats.cache_slots
                  << " cache_requests=" << cache_stats.requests
                  << " cache_hits=" << cache_stats.hits
                  << " cache_misses=" << cache_stats.misses
                  << " cache_loads=" << cache_stats.loads
                  << " cache_hit_rate=" << cache_stats.hit_rate()
                  << " cache_repeat_max_error=" << cache_error << '\n';
        if (cache_error != 0.0f || cache_stats.misses != 6 ||
            cache_stats.hits != 6 || cache_stats.loads != 6) {
            throw std::runtime_error("SSD expert cache validation failed");
        }

        mfq::metal::MlxDeepseekV4SsdExpertCache prefill_cache(
            argv[1],
            (512 + 8) * store.slot_bytes(),
            8,
            true);
        auto current = prefill_cache.prefetch_layer(
            static_cast<std::size_t>(layer));
        auto next = prefill_cache.prefetch_layer(
            static_cast<std::size_t>((layer + 1) % 43));
        const auto prefill_begin = std::chrono::steady_clock::now();
        const auto& current_weights = current.wait();
        const auto current_wait_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - prefill_begin).count();
        const auto prefill_error = maximum_error(
            current_weights.gate.forward(hidden, ids),
            mlx::core::expand_dims(
                arena.expert_weight(0, '1').matmul(hidden),
                1));
        const auto next_begin = std::chrono::steady_clock::now();
        static_cast<void>(next.wait());
        const auto next_wait_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - next_begin).count();
        const auto prefill_stats = prefill_cache.stats();
        std::cout << "prefill_current_wait_ms="
                  << current_wait_seconds * 1000.0
                  << " prefill_next_wait_ms=" << next_wait_seconds * 1000.0
                  << " prefill_layers=" << prefill_stats.prefill_layers
                  << " prefill_reads=" << prefill_stats.prefill_expert_reads
                  << " prefill_bytes=" << prefill_stats.prefill_bytes_read
                  << " prefill_gate_max_error=" << prefill_error << '\n';
        if (prefill_error != 0.0f || prefill_stats.prefill_layers != 2 ||
            prefill_stats.prefill_expert_reads != 512) {
            throw std::runtime_error("SSD expert double-buffer validation failed");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mfq-metal-ssd-expert-real-smoke: "
                  << error.what() << '\n';
        return 1;
    }
}
