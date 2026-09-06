#include "mfq_container.h"
#include "mlx_vq.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

std::uint64_t output_hash(array output) {
    constexpr std::uint64_t offset = 1469598103934665603ull;
    constexpr std::uint64_t prime = 1099511628211ull;
    output = mlx::core::contiguous(std::move(output));
    output.eval();
    const auto* bytes = output.data<std::uint8_t>();
    std::uint64_t result = offset;
    for (std::size_t index = 0; index < output.nbytes(); ++index) {
        result ^= bytes[index];
        result *= prime;
    }
    return result;
}

void report_group64_delta(array reference, array candidate) {
    reference = mlx::core::contiguous(
        mlx::core::astype(std::move(reference), mlx::core::float32));
    candidate = mlx::core::contiguous(
        mlx::core::astype(std::move(candidate), mlx::core::float32));
    mlx::core::eval(reference, candidate);
    require(reference.size() == candidate.size(), "comparison shape differs");
    const auto* expected = reference.data<float>();
    const auto* actual = candidate.data<float>();
    std::size_t different = 0;
    double total_absolute = 0.0;
    float maximum_absolute = 0.0f;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const float absolute = std::fabs(expected[index] - actual[index]);
        different += absolute != 0.0f;
        total_absolute += absolute;
        maximum_absolute = std::max(maximum_absolute, absolute);
    }
    std::cerr
        << "group64_tile_delta\telements=" << reference.size()
        << "\tdifferent=" << different
        << "\tmean_abs=" << std::setprecision(9)
        << total_absolute / reference.size()
        << "\tmax_abs=" << maximum_absolute << '\n';
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
    int eval_batch = 1;
    if (const auto* value = std::getenv("MFQ_METAL_VQ_EVAL_BATCH")) {
        eval_batch = std::stoi(value);
        require(eval_batch > 0, "eval batch must be positive");
    }
    if (const auto* tile = std::getenv("MFQ_METAL_VQ_COMPARE_GROUP64_TILE")) {
        const auto* previous = std::getenv(
            "MFQ_METAL_VQ_GROUP64_OUTPUT_TILE");
        const std::string saved = previous == nullptr ? "" : previous;
        const bool had_previous = previous != nullptr;
        setenv("MFQ_METAL_VQ_GROUP64_OUTPUT_TILE", "legacy", 1);
        auto reference = weight.matmul(source);
        mlx::core::eval(reference);
        setenv("MFQ_METAL_VQ_GROUP64_OUTPUT_TILE", tile, 1);
        auto candidate = weight.matmul(source);
        mlx::core::eval(candidate);
        mlx::core::synchronize();
        report_group64_delta(std::move(reference), std::move(candidate));
        if (had_previous) {
            setenv(
                "MFQ_METAL_VQ_GROUP64_OUTPUT_TILE",
                saved.c_str(),
                1);
        } else {
            unsetenv("MFQ_METAL_VQ_GROUP64_OUTPUT_TILE");
        }
    }

    auto result = weight.matmul(source);
    mlx::core::eval(result);
    for (int index = 0; index < warmup; ++index) {
        result = weight.matmul(source);
        mlx::core::eval(result);
    }
    mlx::core::synchronize();

    const auto started = Clock::now();
    for (int index = 0; index < repetitions; index += eval_batch) {
        std::vector<array> pending;
        const int count = std::min(eval_batch, repetitions - index);
        pending.reserve(static_cast<std::size_t>(count));
        for (int item = 0; item < count; ++item) {
            result = weight.matmul(source);
            pending.push_back(result);
        }
        mlx::core::eval(std::move(pending));
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
    const auto hash = output_hash(result);

    std::cout
        << record.dtype << '\t'
        << name << '\t'
        << rows << 'x' << weight.input_size() << 'x'
        << weight.output_size() << '\t'
        << weight.group_size() << 'x' << weight.vector_size()
        << 'x' << weight.index_bits() << 'x'
        << weight.state_bits() << '\t'
        << weight.packed_nbytes() << '\t'
        << std::fixed << std::setprecision(3) << mean_ms << '\t'
        << std::setprecision(1) << decimal_gbps << '\t'
        << std::setprecision(6) << checksum << '\t'
        << maximum << '\t' << std::hex << hash << std::dec << '\n';
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
            << "dtype\ttensor\tshape\tlayout\tpacked_bytes\tms\tGB/s\t"
               "checksum\tmax_abs\thash\n";
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
