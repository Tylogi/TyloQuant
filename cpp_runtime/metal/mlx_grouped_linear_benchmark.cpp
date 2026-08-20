#include "mfq_container.h"
#include "mlx_grouped_linear.h"
#include "mlx_tensor.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
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
using mfq::metal::MlxGroupedLinear;
using mfq::metal::MlxGroupedLinearWeightRef;
using mfq::metal::MlxLinear;

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
            values[static_cast<std::size_t>(row) * width + column] =
                static_cast<float>(
                    (column * 17 + row * 29 + 11) % 127 - 63) / 256.0f;
        }
    }
    return mlx::core::astype(
        array(values.begin(), Shape{rows, width}),
        mlx::core::float16);
}

std::size_t packed_bytes(const MlxLinear& linear) {
    const auto ref = linear.grouped_weight_ref();
    require(ref.has_value(), "benchmark tensor cannot use grouped linear");
    return std::visit(
        [](const auto* weight) {
            return weight->packed_nbytes();
        },
        *ref);
}

std::uint64_t output_hash(std::vector<array> outputs) {
    constexpr std::uint64_t offset = 1469598103934665603ull;
    constexpr std::uint64_t prime = 1099511628211ull;
    std::uint64_t result = offset;
    for (auto& output : outputs) {
        output = mlx::core::contiguous(std::move(output));
        output.eval();
        const auto* bytes = output.data<std::uint8_t>();
        for (std::size_t index = 0; index < output.nbytes(); ++index) {
            result ^= bytes[index];
            result *= prime;
        }
    }
    return result;
}

double elapsed_ms(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(
            argc >= 6,
            "usage: mfq-metal-grouped-linear-benchmark MODEL.mfq "
            "REPETITIONS ROWS LABEL TENSOR [TENSOR ...]");
        const int repetitions = std::stoi(argv[2]);
        const int rows = std::stoi(argv[3]);
        require(repetitions > 0, "repetitions must be positive");
        require(rows >= 1 && rows <= 16, "rows must be in [1, 16]");

        const MfqContainer model(argv[1]);
        std::vector<MlxLinear> linears;
        linears.reserve(static_cast<std::size_t>(argc - 5));
        for (int index = 5; index < argc; ++index) {
            linears.push_back(MlxLinear::load(model, argv[index]));
        }
        const int input_width = linears.front().input_size();
        std::size_t bytes = 0;
        std::vector<MlxGroupedLinearWeightRef> refs;
        refs.reserve(linears.size());
        for (const auto& linear : linears) {
            require(
                linear.input_size() == input_width,
                "grouped tensor input widths differ");
            bytes += packed_bytes(linear);
            refs.push_back(*linear.grouped_weight_ref());
        }
        MlxGroupedLinear grouped(std::move(refs));
        const auto source = make_input(rows, input_width);

        auto outputs = grouped(source);
        mlx::core::eval(outputs);
        for (int index = 0; index < 4; ++index) {
            outputs = grouped(source);
            mlx::core::eval(outputs);
        }
        mlx::core::synchronize();

        const auto started = Clock::now();
        for (int index = 0; index < repetitions; ++index) {
            outputs = grouped(source);
            mlx::core::eval(outputs);
        }
        mlx::core::synchronize();
        const double mean_ms = elapsed_ms(started) / repetitions;
        const double gbps = static_cast<double>(bytes) /
            (mean_ms * 1.0e6);
        const auto hash = output_hash(std::move(outputs));

        std::cout
            << argv[4] << '\t' << rows << '\t' << linears.size() << '\t'
            << bytes << '\t' << std::fixed << std::setprecision(4)
            << mean_ms << '\t' << std::setprecision(1) << gbps << '\t'
            << std::hex << hash << std::dec << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MFQ grouped Metal benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }
}
