#include "mfq_container.h"
#include "mlx_grouped_linear.h"
#include "mlx_tensor.h"

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

struct OutputDelta {
    std::size_t elements = 0;
    std::size_t different = 0;
    double mean_absolute = 0.0;
    float maximum_absolute = 0.0f;
};

OutputDelta compare_outputs(
    std::vector<array> reference,
    std::vector<array> candidate) {
    require(
        reference.size() == candidate.size(),
        "comparison output count differs");
    OutputDelta delta;
    double total_absolute = 0.0;
    for (std::size_t projection = 0;
         projection < reference.size();
         ++projection) {
        auto reference_f32 = mlx::core::contiguous(
            mlx::core::astype(reference[projection], mlx::core::float32));
        auto candidate_f32 = mlx::core::contiguous(
            mlx::core::astype(candidate[projection], mlx::core::float32));
        mlx::core::eval(reference_f32, candidate_f32);
        require(
            reference_f32.size() == candidate_f32.size(),
            "comparison output shape differs");
        const auto* reference_values = reference_f32.data<float>();
        const auto* candidate_values = candidate_f32.data<float>();
        for (std::size_t index = 0;
             index < reference_f32.size();
             ++index) {
            const float absolute = std::fabs(
                reference_values[index] - candidate_values[index]);
            delta.different += absolute != 0.0f;
            total_absolute += absolute;
            delta.maximum_absolute = std::max(
                delta.maximum_absolute,
                absolute);
        }
        delta.elements += reference_f32.size();
    }
    if (delta.elements != 0) {
        delta.mean_absolute = total_absolute / delta.elements;
    }
    return delta;
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
        int eval_batch = 1;
        if (const auto* value =
                std::getenv("MFQ_METAL_GROUPED_EVAL_BATCH")) {
            eval_batch = std::stoi(value);
            require(eval_batch > 0, "eval batch must be positive");
        }
        if (eval_batch != 1) {
            std::cerr << "eval_batch\t" << eval_batch << '\n';
        }

        if (std::getenv("MFQ_METAL_GROUPED_COMPARE_LAYOUTS") != nullptr &&
            rows >= 2 && rows <= 4) {
            const auto* previous =
                std::getenv("MFQ_METAL_GROUPED_SMALL_M_LAYOUT");
            const std::string saved = previous == nullptr ? "" : previous;
            const bool had_previous = previous != nullptr;
            setenv(
                "MFQ_METAL_GROUPED_SMALL_M_LAYOUT",
                "scalar",
                1);
            auto scalar = grouped(source);
            mlx::core::eval(scalar);
            setenv(
                "MFQ_METAL_GROUPED_SMALL_M_LAYOUT",
                "blockwise",
                1);
            auto blockwise = grouped(source);
            mlx::core::eval(blockwise);
            mlx::core::synchronize();
            const auto delta = compare_outputs(
                std::move(scalar),
                std::move(blockwise));
            if (had_previous) {
                setenv(
                    "MFQ_METAL_GROUPED_SMALL_M_LAYOUT",
                    saved.c_str(),
                    1);
            } else {
                unsetenv("MFQ_METAL_GROUPED_SMALL_M_LAYOUT");
            }
            std::cerr
                << "layout_delta\telements=" << delta.elements
                << "\tdifferent=" << delta.different
                << "\tmean_abs=" << std::setprecision(9)
                << delta.mean_absolute
                << "\tmax_abs=" << delta.maximum_absolute << '\n';
        }

        auto outputs = grouped(source);
        mlx::core::eval(outputs);
        for (int index = 0; index < 4; ++index) {
            outputs = grouped(source);
            mlx::core::eval(outputs);
        }
        mlx::core::synchronize();

        const auto started = Clock::now();
        for (int index = 0; index < repetitions; index += eval_batch) {
            std::vector<array> pending;
            const int count = std::min(eval_batch, repetitions - index);
            pending.reserve(
                static_cast<std::size_t>(count) * linears.size());
            for (int item = 0; item < count; ++item) {
                outputs = grouped(source);
                pending.insert(
                    pending.end(),
                    outputs.begin(),
                    outputs.end());
            }
            mlx::core::eval(std::move(pending));
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
