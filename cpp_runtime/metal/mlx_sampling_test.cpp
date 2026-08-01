#include "mlx_sampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using mlx::core::Shape;
using mlx::core::array;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

array floats(
    const std::vector<float>& values,
    Shape shape,
    mlx::core::Dtype dtype = mlx::core::float32) {
    array result(values.begin(), std::move(shape), mlx::core::float32);
    if (dtype != mlx::core::float32) {
        result = mlx::core::astype(result, dtype);
    }
    return result;
}

std::vector<std::int32_t> evaluated_ids(array values) {
    values.eval();
    return std::vector<std::int32_t>(
        values.data<std::int32_t>(),
        values.data<std::int32_t>() + values.size());
}

std::vector<float> evaluated_floats(array values) {
    values = mlx::core::astype(values, mlx::core::float32);
    values.eval();
    return std::vector<float>(
        values.data<float>(),
        values.data<float>() + values.size());
}

float sanitized(float value) {
    return std::isnan(value)
        ? -std::numeric_limits<float>::max()
        : value;
}

std::int32_t cpu_greedy(
    const float* logits,
    int vocab) {
    float best = -std::numeric_limits<float>::max();
    int best_index = 0;
    for (int token = 0; token < vocab; ++token) {
        const float value = sanitized(logits[token]);
        if (value > best ||
            (value == best && token < best_index)) {
            best = value;
            best_index = token;
        }
    }
    return best_index;
}

std::int32_t cpu_sample(
    const float* logits,
    int vocab,
    float uniform,
    float temperature,
    int top_k,
    float top_p) {
    std::vector<int> order(static_cast<std::size_t>(vocab));
    for (int token = 0; token < vocab; ++token) {
        order[static_cast<std::size_t>(token)] = token;
    }
    if (top_k > 0 || top_p < 1.0f) {
        std::sort(
            order.begin(),
            order.end(),
            [&](int left, int right) {
                const float left_value =
                    sanitized(logits[left]) / temperature;
                const float right_value =
                    sanitized(logits[right]) / temperature;
                if (left_value != right_value) {
                    return left_value > right_value;
                }
                return left < right;
            });
    }
    const int count =
        top_k <= 0 ? vocab : std::min(top_k, vocab);
    order.resize(static_cast<std::size_t>(count));

    std::vector<float> probabilities(
        static_cast<std::size_t>(count));
    float maximum = -std::numeric_limits<float>::max();
    for (const int token : order) {
        maximum = std::max(
            maximum,
            sanitized(logits[token]) / temperature);
    }
    float total = 0.0f;
    for (int rank = 0; rank < count; ++rank) {
        const float probability = std::exp(
            sanitized(logits[order[static_cast<std::size_t>(rank)]]) /
                temperature -
            maximum);
        probabilities[static_cast<std::size_t>(rank)] = probability;
        total += probability;
    }

    int keep = count;
    float keep_sum = total;
    if (top_p > 0.0f && top_p < 1.0f) {
        const float cutoff = top_p * total;
        float cumulative = 0.0f;
        for (int rank = 0; rank < count; ++rank) {
            cumulative +=
                probabilities[static_cast<std::size_t>(rank)];
            if (cumulative >= cutoff) {
                keep = rank + 1;
                keep_sum = cumulative;
                break;
            }
        }
    }

    uniform = std::clamp(uniform, 0.0f, 0.99999994f);
    const float target = uniform * keep_sum;
    float cumulative = 0.0f;
    int chosen = order[static_cast<std::size_t>(keep - 1)];
    for (int rank = 0; rank < keep; ++rank) {
        cumulative += probabilities[static_cast<std::size_t>(rank)];
        if (cumulative >= target) {
            chosen = order[static_cast<std::size_t>(rank)];
            break;
        }
    }
    return chosen;
}

void require_ids(
    array actual,
    const std::vector<std::int32_t>& expected,
    const std::string& name) {
    const auto values = evaluated_ids(std::move(actual));
    require(
        values.size() == expected.size(),
        name + " output size mismatch");
    for (std::size_t index = 0; index < values.size(); ++index) {
        require(
            values[index] == expected[index],
            name + " mismatch at " + std::to_string(index) +
                ": actual=" + std::to_string(values[index]) +
                " expected=" + std::to_string(expected[index]));
    }
}

void test_greedy() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> logits{
        1.0f, 4.0f, 4.0f, nan,
        nan, -2.0f, -1.0f, -1.0f,
        -5.0f, -5.0f, -5.0f, -5.0f,
        0.0f, 3.0f, 2.0f, 1.0f,
    };
    const auto input = floats(logits, Shape{2, 2, 4});
    std::vector<std::int32_t> expected;
    for (int row = 0; row < 4; ++row) {
        expected.push_back(cpu_greedy(
            logits.data() + row * 4,
            4));
    }
    auto output = mfq::metal::sample_greedy(input);
    require(
        output.shape() == Shape({2, 2}),
        "greedy did not preserve the logits prefix shape");
    require_ids(
        std::move(output),
        expected,
        "greedy f32");

    require_ids(
        mfq::metal::sample_greedy(
            mlx::core::astype(input, mlx::core::float16)),
        expected,
        "greedy f16");
}

void test_softmax() {
    const std::vector<float> logits{
        -1.0f, 0.0f, 0.5f, 2.0f, -0.5f,
        1.5f, -2.0f, 0.25f, 0.0f, 1.0f,
    };
    const std::vector<float> uniforms{0.12f, 0.83f};
    std::vector<std::int32_t> expected;
    for (int row = 0; row < 2; ++row) {
        expected.push_back(cpu_sample(
            logits.data() + row * 5,
            5,
            uniforms[static_cast<std::size_t>(row)],
            0.75f,
            0,
            1.0f));
    }
    const auto random = floats(uniforms, Shape{2});
    require_ids(
        mfq::metal::sample_softmax(
            floats(logits, Shape{2, 5}),
            random,
            0.75),
        expected,
        "softmax f32");
    require_ids(
        mfq::metal::sample_softmax(
            floats(logits, Shape{2, 5}, mlx::core::float16),
            random,
            0.75),
        expected,
        "softmax f16");
}

void test_direct_top_k_top_p() {
    const std::vector<float> logits{
        0.0f, 3.0f, 1.0f, 2.0f, -2.0f, 2.0f, 0.5f, -1.0f,
        1.0f, 0.0f, 4.0f, -2.0f, 3.0f, 1.5f, 2.0f, -1.0f,
    };
    const std::vector<float> uniforms{0.3f, 0.91f};
    std::vector<std::int32_t> expected;
    for (int row = 0; row < 2; ++row) {
        expected.push_back(cpu_sample(
            logits.data() + row * 8,
            8,
            uniforms[static_cast<std::size_t>(row)],
            1.25f,
            5,
            0.7f));
    }
    require_ids(
        mfq::metal::sample_top_k_top_p(
            floats(logits, Shape{2, 8}),
            floats(uniforms, Shape{2}),
            1.25,
            5,
            0.7),
        expected,
        "direct top-k/top-p");
}

void test_direct_top_k_large_vocab_and_stable_ties() {
    constexpr int rows = 4;
    constexpr int vocab = 513;
    std::vector<float> tied_logits(
        static_cast<std::size_t>(rows * vocab),
        0.0f);
    const std::vector<float> uniforms{
        0.0f,
        0.1f,
        0.5f,
        0.999999f,
    };
    std::vector<std::int32_t> expected;
    for (int row = 0; row < rows; ++row) {
        expected.push_back(cpu_sample(
            tied_logits.data() + row * vocab,
            vocab,
            uniforms[static_cast<std::size_t>(row)],
            1.0f,
            64,
            1.0f));
    }
    for (const auto dtype :
         {mlx::core::float32, mlx::core::float16}) {
        require_ids(
            mfq::metal::sample_top_k_top_p(
                floats(
                    tied_logits,
                    Shape{rows, vocab},
                    dtype),
                floats(uniforms, Shape{rows}),
                1.0,
                64,
                1.0),
            expected,
            "direct top-k stable ties");
    }

    std::vector<float> logits(
        static_cast<std::size_t>(rows * vocab));
    for (int row = 0; row < rows; ++row) {
        for (int token = 0; token < vocab; ++token) {
            logits[static_cast<std::size_t>(
                row * vocab + token)] =
                std::sin(
                    static_cast<float>(token) * 0.03125f +
                    static_cast<float>(row) * 0.17f) *
                    3.0f +
                static_cast<float>(token % 11) * 0.0625f;
        }
    }
    // Exercise equal values both within a lane's 256-token stripe and across
    // SIMD groups.  The lower token id must always win the tie.
    for (int row = 0; row < rows; ++row) {
        const std::size_t base =
            static_cast<std::size_t>(row * vocab);
        logits[base + 3] = 8.0f;
        logits[base + 67] = 8.0f;
        logits[base + 259] = 8.0f;
        logits[base + 323] = 8.0f;
    }
    logits[17] = std::numeric_limits<float>::quiet_NaN();
    logits[vocab + 29] =
        -std::numeric_limits<float>::infinity();

    for (const int top_k : {1, 2, 20, 32, 63, 64}) {
        expected.clear();
        for (int row = 0; row < rows; ++row) {
            expected.push_back(cpu_sample(
                logits.data() + row * vocab,
                vocab,
                uniforms[static_cast<std::size_t>(row)],
                0.85f,
                top_k,
                0.82f));
        }
        require_ids(
            mfq::metal::sample_top_k_top_p(
                floats(logits, Shape{rows, vocab}),
                floats(uniforms, Shape{rows}),
                0.85,
                top_k,
                0.82),
            expected,
            "direct top-k large-vocab merge k=" +
                std::to_string(top_k));
    }

    std::vector<float> exceptional(
        static_cast<std::size_t>(vocab),
        -std::numeric_limits<float>::infinity());
    exceptional[275] =
        std::numeric_limits<float>::quiet_NaN();
    exceptional[17] =
        std::numeric_limits<float>::quiet_NaN();
    require_ids(
        mfq::metal::sample_top_k_top_p(
            floats(exceptional, Shape{1, vocab}),
            floats({0.0f}, Shape{1}),
            1.0,
            64,
            1.0),
        {17},
        "direct top-k NaN and negative-infinity ordering");
}

void test_sorted_top_k_and_global_top_p() {
    constexpr int vocab = 96;
    std::vector<float> logits(static_cast<std::size_t>(vocab));
    for (int token = 0; token < vocab; ++token) {
        logits[static_cast<std::size_t>(token)] =
            std::sin(static_cast<float>(token) * 0.37f) * 2.0f +
            static_cast<float>(token % 7) * 0.11f;
    }
    const auto input = floats(logits, Shape{1, vocab});

    require_ids(
        mfq::metal::sample_top_k_top_p(
            input,
            floats({0.61f}, Shape{1}),
            0.9,
            70,
            0.82),
        {
            cpu_sample(
                logits.data(),
                vocab,
                0.61f,
                0.9f,
                70,
                0.82f),
        },
        "sorted top-k/top-p");

    mfq::metal::MlxSamplingParams params;
    params.temperature = 0.9;
    params.top_k = 0;
    params.top_p = 0.72;
    require_ids(
        mfq::metal::sample(
            input,
            params,
            floats({0.44f}, Shape{1})),
        {
            cpu_sample(
                logits.data(),
                vocab,
                0.44f,
                0.9f,
                0,
                0.72f),
        },
        "global top-p");
}

void benchmark_direct_top_k() {
    constexpr int vocab = 248320;
    constexpr int iterations = 20;
    std::vector<float> logits(static_cast<std::size_t>(vocab));
    for (int token = 0; token < vocab; ++token) {
        logits[static_cast<std::size_t>(token)] =
            std::sin(static_cast<float>(token) * 0.001953125f) *
                4.0f +
            static_cast<float>(token % 31) * 0.015625f;
    }
    auto input = floats(
        logits,
        Shape{1, vocab},
        mlx::core::float16);
    input.eval();
    const auto random = floats({0.37f}, Shape{1});

    // Warm both the MLX graph and the TOP_K=20 Metal specialization before
    // measuring synchronous single-row decode sampling.
    for (int iteration = 0; iteration < 3; ++iteration) {
        mfq::metal::sample_top_k_top_p(
            input,
            random,
            0.7,
            20,
            0.8).eval();
    }
    const auto start = std::chrono::steady_clock::now();
    std::int64_t checksum = 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        auto sampled = mfq::metal::sample_top_k_top_p(
            input,
            random,
            0.7,
            20,
            0.8);
        sampled.eval();
        checksum += sampled.data<std::int32_t>()[0];
    }
    const auto elapsed = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - start);
    std::cout
        << "MFQ direct top-k microbenchmark: vocab="
        << vocab
        << " top_k=20 f16 rows=1 average_us="
        << elapsed.count() / static_cast<double>(iterations)
        << " checksum="
        << checksum
        << '\n';
}

void test_seeded_sampler() {
    constexpr int vocab = 257;
    const std::vector<float> logits(
        static_cast<std::size_t>(vocab),
        0.0f);
    const auto input = floats(logits, Shape{1, vocab});

    mfq::metal::MlxSamplingParams params;
    params.temperature = 1.0;
    params.seed = 0x123456789abcdef0ULL;
    mfq::metal::MlxSampler first(params);
    mfq::metal::MlxSampler second(params);
    std::mt19937_64 reference_rng(params.seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    std::int32_t initial = -1;
    for (int step = 0; step < 4; ++step) {
        const auto expected = cpu_sample(
            logits.data(),
            vocab,
            uniform(reference_rng),
            1.0f,
            0,
            1.0f);
        const auto first_id = evaluated_ids(first.sample(input)).front();
        const auto second_id = evaluated_ids(second.sample(input)).front();
        require(
            first_id == expected && second_id == expected,
            "seeded sampler diverged from CUDA runtime RNG semantics");
        if (step == 0) {
            initial = first_id;
        }
    }

    first.reset_seed(params.seed);
    require_ids(
        first.sample(input),
        {initial},
        "sampler reset_seed");

    require_ids(
        mfq::metal::sample(input, params),
        {initial},
        "stateless seeded sample");
}

void test_counts_and_penalties() {
    const array counts(
        {0, 1, 0, 2, 5},
        mlx::core::int32);
    const array tokens(
        {1, 2, 2, 4, -1, 8},
        mlx::core::int32);
    auto updated = mfq::metal::sample_token_counts_add(
        counts,
        tokens);
    require_ids(
        updated,
        {0, 2, 2, 2, 6},
        "token counts");

    const std::vector<float> logits{
        2.0f, -1.0f, 0.5f, -0.25f, 4.0f,
        -3.0f, 1.5f, -2.0f, 0.75f, -0.5f,
    };
    const std::vector<std::int32_t> count_values{
        0, 2, 2, 2, 6,
    };
    std::vector<float> expected = logits;
    for (int row = 0; row < 2; ++row) {
        for (int token = 0; token < 5; ++token) {
            const int count =
                count_values[static_cast<std::size_t>(token)];
            if (count == 0) {
                continue;
            }
            auto& value =
                expected[static_cast<std::size_t>(row * 5 + token)];
            value = value < 0.0f ? value * 2.0f : value / 2.0f;
            value -= 0.25f + 0.1f * static_cast<float>(count);
        }
    }

    for (const auto dtype :
         {mlx::core::float32, mlx::core::float16}) {
        const auto actual = evaluated_floats(
            mfq::metal::sample_apply_penalties(
                floats(logits, Shape{2, 5}, dtype),
                updated,
                0.25,
                0.1,
                2.0));
        const float tolerance =
            dtype == mlx::core::float16 ? 3.0e-3f : 1.0e-6f;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            require(
                std::fabs(actual[index] - expected[index]) <= tolerance,
                "penalty mismatch at " + std::to_string(index));
        }
    }

    mfq::metal::MlxSamplingParams params;
    params.temperature = 0.0;
    params.presence_penalty = 0.5;
    mfq::metal::MlxSampler sampler(params);
    require_ids(
        sampler.sample(
            floats({2.0f, 1.9f}, Shape{1, 2}),
            array({1, 0}, mlx::core::int32)),
        {1},
        "sampler configured penalties");
}

void test_validation_and_greedy_precedence() {
    const auto logits = floats({0.0f, 1.0f, 2.0f}, Shape{1, 3});
    const auto random = floats({0.5f}, Shape{1});

    require_throws(
        [&] {
            (void)mfq::metal::sample_greedy(array(1.0f));
        },
        "scalar logits were accepted");
    require_throws(
        [&] {
            (void)mfq::metal::sample_softmax(logits, random, 0.0);
        },
        "zero temperature was accepted by softmax");
    require_throws(
        [&] {
            (void)mfq::metal::sample_top_k_top_p(
                logits,
                random,
                1.0,
                4,
                1.0);
        },
        "top_k above vocab was accepted");
    require_throws(
        [&] {
            (void)mfq::metal::sample_top_k_top_p(
                logits,
                random,
                1.0,
                2,
                0.0);
        },
        "zero top_p was accepted");
    require_throws(
        [&] {
            mfq::metal::MlxSamplingParams params;
            params.top_k = -1;
            (void)mfq::metal::sample(logits, params);
        },
        "negative top_k was accepted");
    require_throws(
        [&] {
            (void)mfq::metal::sample_apply_penalties(
                logits,
                array({0, 0}, mlx::core::int32));
        },
        "mismatched penalty counts were accepted");
    require_throws(
        [&] {
            (void)mfq::metal::sample_apply_penalties(
                logits,
                array({0, 0, 0}, mlx::core::int32),
                0.0,
                0.0,
                0.0);
        },
        "non-positive repetition penalty was accepted");

    mfq::metal::MlxSamplingParams greedy;
    greedy.temperature = 0.0;
    greedy.top_k = 9999;
    greedy.top_p = 0.0;
    require_ids(
        mfq::metal::sample(logits, greedy),
        {2},
        "temperature-zero greedy precedence");

    greedy.temperature =
        std::numeric_limits<double>::quiet_NaN();
    greedy.top_k = 1;
    require_ids(
        mfq::metal::sample(logits, greedy),
        {2},
        "top-k-one greedy precedence");
}

} // namespace

int main() {
    try {
        test_greedy();
        test_softmax();
        test_direct_top_k_top_p();
        test_direct_top_k_large_vocab_and_stable_ties();
        test_sorted_top_k_and_global_top_p();
        test_seeded_sampler();
        test_counts_and_penalties();
        test_validation_and_greedy_precedence();
        benchmark_direct_top_k();
        std::cout
            << "MFQ C++ sampling Apple GPU/CPU numerical tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "MFQ C++ sampling test failed: "
            << error.what()
            << '\n';
        return 1;
    }
}
