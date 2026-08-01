#include "mfq_container.h"
#include "mlx_nint.h"
#include "mlx_nint8_zero.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/stream.h>

namespace {

using Clock = std::chrono::steady_clock;
using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MfqContainer;
using mfq::metal::MlxNint8ZeroWeight;
using mfq::metal::MlxNintWeight;

using PackedWeight = std::variant<MlxNintWeight, MlxNint8ZeroWeight>;

struct BenchmarkCase {
    std::string name;
    std::string dtype;
    PackedWeight weight;
};

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

int input_size(const PackedWeight& weight) {
    return std::visit(
        [](const auto& value) {
            return value.input_size();
        },
        weight);
}

int output_size(const PackedWeight& weight) {
    return std::visit(
        [](const auto& value) {
            return value.output_size();
        },
        weight);
}

std::size_t packed_nbytes(const PackedWeight& weight) {
    return std::visit(
        [](const auto& value) {
            return value.packed_nbytes();
        },
        weight);
}

array matmul(const PackedWeight& weight, const array& input) {
    return std::visit(
        [&](const auto& value) {
            return value.matmul(input);
        },
        weight);
}

BenchmarkCase load_case(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    auto blob = model.read(name);
    if (mfq::metal::is_nint_dtype(record.dtype)) {
        return {
            name,
            record.dtype,
            MlxNintWeight::from_blob(blob),
        };
    }
    if (mfq::metal::is_nint8_zero_dtype(record.dtype)) {
        return {
            name,
            record.dtype,
            MlxNint8ZeroWeight::from_blob(blob),
        };
    }
    throw std::runtime_error(
        "benchmark tensor is not NINT/NINT8-0: " + name);
}

double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
}

void benchmark(
    const MfqContainer& model,
    const std::string& name,
    int warmup,
    int repetitions) {
    auto benchmark_case = load_case(model, name);
    const int input = input_size(benchmark_case.weight);
    const int output = output_size(benchmark_case.weight);
    const auto bytes = packed_nbytes(benchmark_case.weight);
    const auto source = make_input(input);

    array result = matmul(benchmark_case.weight, source);
    mlx::core::eval(result);
    for (int index = 0; index < warmup; ++index) {
        result = matmul(benchmark_case.weight, source);
        mlx::core::eval(result);
    }
    mlx::core::synchronize();

    const auto started = Clock::now();
    for (int index = 0; index < repetitions; ++index) {
        result = matmul(benchmark_case.weight, source);
        mlx::core::eval(result);
    }
    mlx::core::synchronize();
    const double total_ms = milliseconds_since(started);
    const double mean_ms = total_ms / static_cast<double>(repetitions);
    const double decimal_gbps =
        static_cast<double>(bytes) / (mean_ms * 1.0e6);

    auto checked = mlx::core::astype(result, mlx::core::float32);
    mlx::core::eval(checked);
    const auto* values = checked.data<float>();
    double checksum = 0.0;
    float maximum = 0.0f;
    for (int index = 0; index < output; ++index) {
        const float value = values[index];
        require(
            std::isfinite(value),
            "benchmark output contains non-finite values");
        checksum += static_cast<double>(value);
        maximum = std::max(maximum, std::fabs(value));
    }

    std::cout
        << benchmark_case.dtype << '\t'
        << name << '\t'
        << input << 'x' << output << '\t'
        << bytes << '\t'
        << std::fixed << std::setprecision(3) << mean_ms << '\t'
        << std::setprecision(1) << decimal_gbps << '\t'
        << "1\t"
        << std::setprecision(6) << checksum << '\t'
        << maximum << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(
            argc >= 2,
            "usage: mfq-metal-nint-benchmark MODEL.mfq [REPETITIONS]");
        const int repetitions =
            argc >= 3 ? std::stoi(argv[2]) : 20;
        require(repetitions > 0, "repetitions must be positive");

        const MfqContainer model(argv[1]);
        std::cout
            << "dtype\ttensor\tshape\tpacked_bytes\tms\tGB/s\t"
               "launches\tchecksum\tmax_abs\n";
        for (const auto& name : {
                 "blk.0.ffn_gate.weight",
                 "blk.0.attn_gate.weight",
                 "blk.0.ffn_down.weight",
                 "blk.0.ssm_out.weight",
                 "blk.3.attn_output.weight",
                 "blk.3.attn_v.weight",
                 "output.weight",
             }) {
            benchmark(model, name, 3, repetitions);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MFQ NINT Metal benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }
}
