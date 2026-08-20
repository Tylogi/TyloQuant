#include "mfq_container.h"
#include "mlx_vq.h"

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
using mfq::metal::MlxVqWeight;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

array make_input(int rows, int width) {
    std::vector<float> values(
        static_cast<std::size_t>(rows) * width);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < width; ++column) {
            const auto index =
                static_cast<std::size_t>(row) * width + column;
            values[index] = static_cast<float>(
                (column * 17 + row * 29 + 11) % 127 - 63) / 256.0f;
        }
    }
    return mlx::core::astype(
        array(values.begin(), Shape{rows, width}),
        mlx::core::float16);
}

MlxVqWeight load_weight(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    require(
        mfq::metal::is_vq_dtype(record.dtype),
        "benchmark tensor is not NVQ/NPQ/NEPQ: " + name);
    const auto mapped = model.map_record(name);
    return MlxVqWeight::from_blob(record.dtype, mapped.view());
}

double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
}

void benchmark(
    const MfqContainer& model,
    const std::string& name,
    int rows,
    int warmup,
    int repetitions) {
    const auto& record = model.record(name);
    auto weight = load_weight(model, name);
    const auto source = make_input(rows, weight.input_size());

    auto result = weight.matmul(source);
    mlx::core::eval(result);
    for (int index = 0; index < warmup; ++index) {
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
    const double mean_ms = total_ms / repetitions;
    const double decimal_gbps =
        static_cast<double>(weight.packed_nbytes()) /
        (mean_ms * 1.0e6);

    auto checked = mlx::core::astype(result, mlx::core::float32);
    mlx::core::eval(checked);
    const auto* values = checked.data<float>();
    const auto count = static_cast<std::size_t>(rows) * weight.output_size();
    double checksum = 0.0;
    float maximum = 0.0f;
    for (std::size_t index = 0; index < count; ++index) {
        require(
            std::isfinite(values[index]),
            "benchmark output contains non-finite values");
        checksum += values[index];
        maximum = std::max(maximum, std::fabs(values[index]));
    }

    std::cout
        << record.dtype << '\t'
        << name << '\t'
        << rows << 'x' << weight.input_size() << 'x'
        << weight.output_size() << '\t'
        << weight.packed_nbytes() << '\t'
        << std::fixed << std::setprecision(3) << mean_ms << '\t'
        << std::setprecision(1) << decimal_gbps << '\t'
        << std::setprecision(6) << checksum << '\t'
        << maximum << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(
            argc >= 5,
            "usage: mfq-metal-vq-benchmark MODEL.mfq REPETITIONS ROWS "
            "TENSOR [TENSOR ...]");
        const int repetitions = std::stoi(argv[2]);
        const int rows = std::stoi(argv[3]);
        require(repetitions > 0, "repetitions must be positive");
        require(rows >= 1 && rows <= 16, "rows must be in [1, 16]");

        const MfqContainer model(argv[1]);
        std::cout
            << "dtype\ttensor\tshape\tpacked_bytes\tms\tGB/s\t"
               "checksum\tmax_abs\n";
        for (int index = 4; index < argc; ++index) {
            benchmark(model, argv[index], rows, 3, repetitions);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MFQ VQ Metal benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }
}
