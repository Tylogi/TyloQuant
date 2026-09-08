#include "qwen4_ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

using mlx::core::Shape;
using mlx::core::array;

array patterned_bfloat(
    std::size_t count,
    const Shape& shape,
    int multiplier,
    float scale) {
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const int centered =
            static_cast<int>((index * multiplier) % 257) - 128;
        values[index] = static_cast<float>(centered) * scale;
    }
    return mlx::core::astype(
        array(values.begin(), shape, mlx::core::float32),
        mlx::core::bfloat16);
}

void require_bit_exact(
    array actual,
    array expected,
    const char* name) {
    if (actual.shape() != expected.shape()) {
        throw std::runtime_error(
            std::string(name) + " shape mismatch");
    }
    if (actual.dtype() != mlx::core::bfloat16 ||
        expected.dtype() != mlx::core::bfloat16) {
        throw std::runtime_error(
            std::string(name) + " dtype mismatch: actual=" +
            std::to_string(static_cast<int>(actual.dtype().val())) +
            " expected=" +
            std::to_string(static_cast<int>(expected.dtype().val())));
    }
    mlx::core::eval(actual, expected);
    const auto* actual_bits = actual.data<std::uint16_t>();
    const auto* expected_bits = expected.data<std::uint16_t>();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual_bits[index] != expected_bits[index]) {
            auto actual_float = mlx::core::astype(
                actual, mlx::core::float32);
            auto expected_float = mlx::core::astype(
                expected, mlx::core::float32);
            mlx::core::eval(actual_float, expected_float);
            throw std::runtime_error(
                std::string(name) + " changed element " +
                std::to_string(index) + ": actual=" +
                std::to_string(actual_float.data<float>()[index]) +
                " expected=" +
                std::to_string(expected_float.data<float>()[index]));
        }
    }
}

array reference_grouped_rms_norm(
    const array& value,
    const array& weight,
    int group_size,
    float eps) {
    auto shape = value.shape();
    const int width = shape.back();
    shape.pop_back();
    shape.push_back(width / group_size);
    shape.push_back(group_size);
    auto normalized = mlx::core::fast::rms_norm(
        mlx::core::reshape(
            mlx::core::astype(value, mlx::core::float32), shape),
        std::nullopt,
        eps);
    normalized = mlx::core::reshape(normalized, value.shape()) *
        (array(1.0f) +
         mlx::core::astype(weight, mlx::core::float32));
    return mlx::core::astype(normalized, value.dtype());
}

void test_decode_fast_path_matches_reference() {
    // Exercise the real Flash-Next/Qwen decode geometry so the combined
    // 10240->320 and 10240->4 projection kernel is covered as well.
    constexpr int hidden = 2560;
    constexpr int streams = 4;
    constexpr int low_rank = 320;
    constexpr int width = hidden * streams;
    constexpr float eps = 1e-6f;

    const auto input = patterned_bfloat(
        width, Shape{1, 1, width}, 37, 1.0f / 53.0f);
    const auto norm = patterned_bfloat(
        width, Shape{width}, 17, 1.0f / 1024.0f);
    const auto down = patterned_bfloat(
        static_cast<std::size_t>(low_rank) * width,
        Shape{low_rank, width}, 29, 1.0f / 4096.0f);
    const auto up = patterned_bfloat(
        static_cast<std::size_t>(width) * low_rank,
        Shape{width, low_rank}, 43, 1.0f / 4096.0f);
    const auto injection = patterned_bfloat(
        static_cast<std::size_t>(streams) * width,
        Shape{streams, width}, 61, 1.0f / 4096.0f);

    auto normalized = reference_grouped_rms_norm(
        input, norm, hidden, eps);
    const array connection_count(
        static_cast<float>(streams), normalized.dtype());
    auto low = mlx::core::matmul(
        normalized, mlx::core::transpose(down)) /
        connection_count;
    low = low * mlx::core::sigmoid(low);
    auto mixing = mlx::core::sigmoid(mlx::core::matmul(
        low, mlx::core::transpose(up)));
    auto reference_branch = mlx::core::mean(
        mlx::core::reshape(
            mixing, Shape{1, 1, streams, hidden}) *
            mlx::core::reshape(
                normalized, Shape{1, 1, streams, hidden}),
        -2);
    auto reference_injection =
        array(2.0f, normalized.dtype()) * mlx::core::sigmoid(
        mlx::core::matmul(
            normalized, mlx::core::transpose(injection)) /
        connection_count);

    auto actual = mfq::metal::qwen4_gated_residual_pre(
        input,
        norm,
        down,
        up,
        std::optional<array>(injection),
        hidden,
        streams,
        eps);
    require_bit_exact(
        mfq::metal::qwen4_grouped_rms_norm(
            input, norm, hidden, eps),
        normalized,
        "Qwen grouped RMSNorm");
    if (!actual.injection) {
        throw std::runtime_error(
            "Qwen gated-HC decode path omitted injection");
    }
    require_bit_exact(
        std::move(actual.branch),
        std::move(reference_branch),
        "Qwen gated-HC branch");
    require_bit_exact(
        std::move(*actual.injection),
        std::move(reference_injection),
        "Qwen gated-HC injection");
}

void benchmark_decode_path() {
    constexpr int hidden = 2560;
    constexpr int streams = 4;
    constexpr int low_rank = 320;
    constexpr int width = hidden * streams;
    constexpr int warmups = 24;
    constexpr int samples = 7;
    constexpr int iterations = 200;
    constexpr float eps = 1e-6f;

    const auto input = patterned_bfloat(
        width, Shape{1, 1, width}, 37, 1.0f / 53.0f);
    const auto norm = patterned_bfloat(
        width, Shape{width}, 17, 1.0f / 1024.0f);
    const auto down = patterned_bfloat(
        static_cast<std::size_t>(low_rank) * width,
        Shape{low_rank, width}, 29, 1.0f / 4096.0f);
    const auto up = patterned_bfloat(
        static_cast<std::size_t>(width) * low_rank,
        Shape{width, low_rank}, 43, 1.0f / 4096.0f);
    const auto injection = patterned_bfloat(
        static_cast<std::size_t>(streams) * width,
        Shape{streams, width}, 61, 1.0f / 4096.0f);
    mlx::core::eval({input, norm, down, up, injection});

    auto run = [&] {
        auto result = mfq::metal::qwen4_gated_residual_pre(
            input,
            norm,
            down,
            up,
            std::optional<array>(injection),
            hidden,
            streams,
            eps);
        mlx::core::eval({result.branch, *result.injection});
    };
    for (int iteration = 0; iteration < warmups; ++iteration)
        run();
    std::vector<double> timings;
    timings.reserve(samples);
    for (int sample = 0; sample < samples; ++sample) {
        const auto started = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration)
            run();
        timings.push_back(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started)
                .count() /
            iterations);
    }
    std::sort(timings.begin(), timings.end());
    const char* setting = std::getenv("MFQ_METAL_QWEN_GATED_HC_FAST");
    std::cout
        << "Qwen gated-HC decode microbenchmark: path="
        << ((setting == nullptr || std::atoi(setting) != 0)
                ? "fast" : "reference")
        << " median_ms=" << timings[samples / 2]
        << " min_ms=" << timings.front()
        << " max_ms=" << timings.back() << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        test_decode_fast_path_matches_reference();
        std::cout << "Qwen gated-HC fast path test passed\n";
        if (argc == 2 && std::string(argv[1]) == "--benchmark")
            benchmark_decode_path();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
