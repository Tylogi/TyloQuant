#include "mlx_qwen35_causal_lm.h"
#include "qwen35_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/memory.h>
#include <mlx/mlx.h>

namespace {

using Clock = std::chrono::steady_clock;
using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MfqContainer;
using mfq::metal::MlxQwen35CausalLm;
using mfq::metal::Qwen35Config;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(
        Clock::now() - start).count();
}

double mib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void print_memory(const std::string& label) {
    std::cout
        << label
        << ": active=" << std::fixed << std::setprecision(1)
        << mib(mlx::core::get_active_memory()) << " MiB"
        << ", cache=" << mib(mlx::core::get_cache_memory()) << " MiB"
        << ", peak=" << mib(mlx::core::get_peak_memory()) << " MiB\n";
}

array last_token(const array& logits, int vocab) {
    require(
        logits.ndim() == 3 &&
            logits.shape(0) == 1 &&
            logits.shape(1) > 0 &&
            logits.shape(2) == vocab,
        "real model logits shape mismatch");
    return mlx::core::reshape(
        mlx::core::slice(
            logits,
            Shape{0, logits.shape(1) - 1, 0},
            Shape{1, logits.shape(1), vocab}),
        Shape{vocab});
}

std::int32_t validate_logits(
    array logits,
    int vocab,
    const std::string& label) {
    logits = mlx::core::astype(
        last_token(logits, vocab),
        mlx::core::float32);
    logits.eval();
    const auto* values = logits.data<float>();
    float maximum_absolute = 0.0f;
    float maximum = -std::numeric_limits<float>::infinity();
    std::int32_t maximum_index = -1;
    double checksum = 0.0;
    for (int index = 0; index < vocab; ++index) {
        const float value = values[index];
        require(
            std::isfinite(value),
            label + " produced a non-finite logit");
        maximum_absolute =
            std::max(maximum_absolute, std::fabs(value));
        checksum += static_cast<double>(value);
        if (value > maximum) {
            maximum = value;
            maximum_index = index;
        }
    }
    require(
        maximum_absolute > 1e-8f,
        label + " produced all-zero logits");
    require(
        maximum_index >= 0 && maximum_index < vocab,
        label + " argmax token is invalid");
    std::cout
        << label
        << ": argmax=" << maximum_index
        << ", max=" << maximum
        << ", checksum=" << checksum << "\n";
    return maximum_index;
}

void validate_real_config(const Qwen35Config& config) {
    require(
        config.hidden_size == 5120,
        "real smoke expects hidden_size=5120");
    require(
        config.vocab_size == 248320,
        "real smoke expects vocab_size=248320");
    require(
        config.num_hidden_layers == 64,
        "real smoke expects 64 transformer layers");
    require(
        config.layer_types.size() == 64,
        "real smoke layer type count mismatch");
    require(
        std::count(
            config.layer_types.begin(),
            config.layer_types.end(),
            "linear_attention") == 48,
        "real smoke expects 48 linear-attention layers");
    require(
        std::count(
            config.layer_types.begin(),
            config.layer_types.end(),
            "full_attention") == 16,
        "real smoke expects 16 full-attention layers");
}

std::vector<float> evaluated_logits(
    array logits,
    int tokens,
    int vocab) {
    require(
        logits.shape() == Shape{1, tokens, vocab},
        "cache-equivalence logits shape mismatch");
    logits = mlx::core::astype(
        logits,
        mlx::core::float32);
    logits.eval();
    return std::vector<float>(
        logits.data<float>(),
        logits.data<float>() + logits.size());
}

int row_argmax(
    const std::vector<float>& values,
    int row,
    int vocab) {
    return static_cast<int>(
        std::max_element(
            values.begin() +
                static_cast<std::ptrdiff_t>(row) * vocab,
            values.begin() +
                static_cast<std::ptrdiff_t>(row + 1) * vocab) -
        (values.begin() +
         static_cast<std::ptrdiff_t>(row) * vocab));
}

void validate_cache_equivalence(
    MlxQwen35CausalLm& model,
    int vocab) {
    constexpr int token_count = 4;
    const std::int32_t token_values[token_count] = {
        1, 2, 3, 4,
    };
    const array sequence(
        token_values,
        Shape{1, token_count},
        mlx::core::int32);
    const auto baseline = evaluated_logits(
        model.forward(sequence, false),
        token_count,
        vocab);

    model.reset_cache(1);
    float maximum_difference = 0.0f;
    double total_difference = 0.0;
    std::size_t compared = 0;
    for (int token = 0; token < token_count; ++token) {
        const array one(
            {token_values[token]},
            Shape{1, 1},
            mlx::core::int32);
        const auto cached = evaluated_logits(
            model.forward(one, true),
            1,
            vocab);
        const int expected_top =
            row_argmax(baseline, token, vocab);
        const int actual_top =
            row_argmax(cached, 0, vocab);
        require(
            actual_top == expected_top,
            "cached decode top-1 differs from full causal prefill "
            "at token " +
                std::to_string(token));
        for (int item = 0; item < vocab; ++item) {
            const auto difference = std::fabs(
                cached[static_cast<std::size_t>(item)] -
                baseline[
                    static_cast<std::size_t>(token) * vocab +
                    item]);
            maximum_difference =
                std::max(maximum_difference, difference);
            total_difference += difference;
            ++compared;
        }
    }
    const auto mean_difference =
        total_difference /
        static_cast<double>(compared);
    std::cout
        << "real cache equivalence: max_abs="
        << maximum_difference
        << ", mean_abs=" << mean_difference << "\n";
    require(
        maximum_difference < 0.5f &&
            mean_difference < 0.02,
        "cached decode logits diverge from full causal prefill");
    model.clear_cache();
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(
            argc == 2,
            "usage: qwen35_real_model_smoke MODEL.mfq");
#ifdef MFQ_MLX_METALLIB_DEFAULT
        mlx::core::metal::set_metallib_path(
            MFQ_MLX_METALLIB_DEFAULT);
#endif
        require(
            mlx::core::is_available(mlx::core::Device::gpu),
            "no Apple GPU is available");
        mlx::core::set_default_device(mlx::core::Device::gpu);
        mlx::core::reset_peak_memory();

        const auto container_start = Clock::now();
        const MfqContainer container(argv[1]);
        const auto config = Qwen35Config::from_mfq(container);
        validate_real_config(config);
        std::cout
            << "container indexed in "
            << seconds_since(container_start)
            << " s; records=" << container.records().size()
            << "\n";

        const auto load_start = Clock::now();
        auto model = MlxQwen35CausalLm::load(container);
        require(
            model.layer_count() == 64,
            "loaded real model layer count mismatch");
        std::cout
            << "native C++ model loaded in "
            << seconds_since(load_start) << " s\n";
        print_memory("after load");

        validate_cache_equivalence(
            model,
            static_cast<int>(config.vocab_size));

        const array prompt(
            {1, 2, 3},
            Shape{1, 3},
            mlx::core::int32);
        const auto prefill_start = Clock::now();
        auto prefill = model.forward(prompt, true);
        const auto next_token = validate_logits(
            std::move(prefill),
            static_cast<int>(config.vocab_size),
            "prefill logits");
        require(
            model.cache_position() == 3,
            "real model prefill cache position mismatch");
        std::cout
            << "prefill evaluated in "
            << std::setprecision(6)
            << seconds_since(prefill_start) << " s\n";
        print_memory("after prefill");

        const array decode_token(
            {next_token},
            Shape{1, 1},
            mlx::core::int32);
        const auto decode_start = Clock::now();
        auto decode = model.forward(decode_token, true);
        (void)validate_logits(
            std::move(decode),
            static_cast<int>(config.vocab_size),
            "decode logits");
        require(
            model.cache_position() == 4,
            "real model decode cache position mismatch");
        std::cout
            << "one-token decode evaluated in "
            << std::setprecision(6)
            << seconds_since(decode_start) << " s\n";
        print_memory("after decode");

        constexpr std::int32_t benchmark_tokens = 64;
        mfq::metal::MlxSamplingParams greedy;
        const auto benchmark_start = Clock::now();
        const auto generated = model.generate(
            {1},
            greedy,
            benchmark_tokens);
        const auto benchmark_seconds =
            seconds_since(benchmark_start);
        require(
            generated == benchmark_tokens,
            "real model decode benchmark ended early");
        std::cout
            << "steady greedy decode: "
            << std::setprecision(6)
            << benchmark_seconds << " s / "
            << benchmark_tokens << " tokens = "
            << static_cast<double>(benchmark_tokens) /
                   benchmark_seconds
            << " tok/s\n";

        model.clear_cache();
        mlx::core::clear_cache();
        require(
            model.cache_position() == 0 &&
                model.cache_batch() == 0,
            "real model cache did not clear");
        print_memory("after cache clear");
        std::cout
            << "Qwen3.5 real 64-layer C++/Metal model smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
