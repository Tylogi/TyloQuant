#include "mfq_container.h"
#include "mlx_mx.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/stream.h>

namespace {

using Clock = std::chrono::steady_clock;
using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MfqContainer;
using mfq::metal::MlxMxWeight;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

array make_input(int width) {
    std::vector<float> values(static_cast<std::size_t>(width));
    for (int index = 0; index < width; ++index) {
        values[static_cast<std::size_t>(index)] =
            static_cast<float>((index * 17 + 11) % 127 - 63) / 256.0f;
    }
    return mlx::core::astype(
        array(values.begin(), Shape{1, width}),
        mlx::core::float16);
}

array make_grouped_input(int rows, int groups, int width) {
    auto row = make_input(width);
    return mlx::core::contiguous(
        mlx::core::broadcast_to(
            mlx::core::reshape(row, Shape{1, 1, 1, width}),
            Shape{1, rows, groups, width}));
}

double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
}

void benchmark(
    const MfqContainer& model,
    const std::string& name,
    int repetitions) {
    const auto& record = model.record(name);
    require(
        mfq::metal::is_mx_dtype(record.dtype),
        "benchmark tensor is not MXFP4/MXFP8: " + name);
    const auto weight = MlxMxWeight::from_blob(
        record.dtype,
        model.read(name));
    const auto source = make_input(weight.input_size());

    auto result = weight.matmul(source);
    mlx::core::eval(result);
    for (int index = 0; index < 5; ++index) {
        result = weight.matmul(source);
        mlx::core::eval(result);
    }
    mlx::core::synchronize();

    const auto started = Clock::now();
    for (int index = 0; index < repetitions; ++index) {
        result = weight.matmul(source);
        mlx::core::eval(result);
    }
    mlx::core::synchronize();
    const double total_ms = milliseconds_since(started);
    const double mean_ms = total_ms / static_cast<double>(repetitions);
    const double decimal_gbps =
        static_cast<double>(weight.packed_nbytes()) / (mean_ms * 1.0e6);

    auto checked = mlx::core::astype(result, mlx::core::float32);
    mlx::core::eval(checked);
    const auto* values = checked.data<float>();
    double checksum = 0.0;
    float maximum = 0.0f;
    for (int index = 0; index < weight.output_size(); ++index) {
        const float value = values[index];
        require(
            std::isfinite(value),
            "benchmark output contains non-finite values");
        checksum += static_cast<double>(value);
        maximum = std::max(maximum, std::fabs(value));
    }

    std::cout
        << record.dtype << '\t'
        << name << '\t'
        << weight.input_size() << 'x' << weight.output_size() << '\t'
        << weight.packed_nbytes() << '\t'
        << std::fixed << std::setprecision(3) << mean_ms << '\t'
        << std::setprecision(1) << decimal_gbps << '\t'
        << std::setprecision(6) << checksum << '\t'
        << maximum << '\n';
}

array fallback_grouped(
    const MlxMxWeight& weight,
    const array& input,
    int groups) {
    auto complete = weight.matmul(input);
    const int output_per_group = weight.output_size() / groups;
    std::vector<array> pieces;
    pieces.reserve(static_cast<std::size_t>(groups));
    for (int group = 0; group < groups; ++group) {
        auto selected = mlx::core::take(
            complete,
            group,
            complete.ndim() - 2);
        Shape starts(selected.ndim(), 0);
        Shape stops = selected.shape();
        starts.back() = group * output_per_group;
        stops.back() = (group + 1) * output_per_group;
        pieces.push_back(
            mlx::core::slice(selected, starts, stops));
    }
    return mlx::core::stack(pieces, input.ndim() - 2);
}

void benchmark_grouped(
    const MfqContainer& model,
    const std::string& name,
    int rows,
    int groups,
    int repetitions) {
    const auto& record = model.record(name);
    require(record.dtype == "MXFP8", "grouped benchmark requires MXFP8");
    const auto weight = MlxMxWeight::from_blob(
        record.dtype,
        model.read(name));
    require(
        weight.output_size() % groups == 0,
        "grouped benchmark output is not divisible by groups");
    const auto input = make_grouped_input(
        rows,
        groups,
        weight.input_size());
    const auto measure = [&](const char* label, const auto& operation) {
        auto output = operation();
        mlx::core::eval(output);
        mlx::core::synchronize();
        const auto started = Clock::now();
        for (int index = 0; index < repetitions; ++index) {
            output = operation();
            mlx::core::eval(output);
        }
        mlx::core::synchronize();
        std::cout << label << '\t' << std::fixed << std::setprecision(3)
                  << milliseconds_since(started) / repetitions << " ms\n";
    };
    measure("fallback", [&] {
        return fallback_grouped(weight, input, groups);
    });
    measure("grouped", [&] {
        return weight.grouped_row_matmul(input, groups);
    });
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::string(argv[1]) == "--grouped") {
            require(
                argc == 7,
                "usage: mfq-metal-mx-benchmark --grouped MODEL.mfq "
                "TENSOR ROWS GROUPS REPETITIONS");
            const MfqContainer model(argv[2]);
            benchmark_grouped(
                model,
                argv[3],
                std::stoi(argv[4]),
                std::stoi(argv[5]),
                std::stoi(argv[6]));
            return 0;
        }
        require(
            argc >= 2,
            "usage: mfq-metal-mx-benchmark MODEL.mfq [REPETITIONS] [TENSOR ...]");
        const int repetitions = argc >= 3 ? std::stoi(argv[2]) : 50;
        require(repetitions > 0, "repetitions must be positive");
        const MfqContainer model(argv[1]);
        std::vector<std::string> names;
        for (int index = 3; index < argc; ++index) {
            names.emplace_back(argv[index]);
        }
        if (names.empty()) {
            names = {
                "blk.3.attn_q_a.weight",
                "blk.3.attn_kv.weight",
                "blk.3.attn_q_b.weight",
                "blk.3.attn_output_a.weight",
                "blk.3.attn_output_b.weight",
                "blk.22.indexer.attn_q_b.weight",
            };
        }
        std::cout
            << "dtype\ttensor\tshape\tpacked_bytes\tms\tGB/s\tchecksum\tmax_abs\n";
        for (const auto& name : names) {
            benchmark(model, name, repetitions);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MFQ MX Metal benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }
}
