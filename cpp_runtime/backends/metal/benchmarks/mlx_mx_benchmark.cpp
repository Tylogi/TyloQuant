#include "mfq_container.h"
#include "mlx_grouped_linear.h"
#include "mlx_mx.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
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
using mfq::metal::MlxMxWeight;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void append(std::vector<std::uint8_t>& target, T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    target.insert(target.end(), bytes, bytes + sizeof(T));
}

std::vector<std::uint8_t> make_synthetic_mx_blob(
    int bits,
    int output_size,
    int input_size) {
    require(bits == 4 || bits == 8, "synthetic MX bits must be 4 or 8");
    require(
        output_size > 0 && input_size > 0 &&
            ((bits == 4 && input_size % 32 == 0) ||
             (bits == 8 && input_size % 128 == 0)),
        "invalid synthetic MX geometry");
    std::vector<std::uint8_t> blob{'M', 'X', 'T', '1'};
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, static_cast<std::uint8_t>(bits));
    append<std::uint16_t>(blob, 0);
    append<std::uint64_t>(blob, output_size);
    append<std::uint64_t>(blob, input_size);
    append<std::uint64_t>(blob, output_size);
    append<std::uint64_t>(blob, bits == 4 ? input_size / 2 : input_size);
    append<std::uint64_t>(
        blob,
        bits == 4 ? output_size : (output_size + 127) / 128);
    append<std::uint64_t>(blob, bits == 4 ? input_size / 32 : input_size / 128);
    const auto value_bytes = static_cast<std::size_t>(output_size) *
        static_cast<std::size_t>(bits == 4 ? input_size / 2 : input_size);
    for (std::size_t index = 0; index < value_bytes; ++index) {
        blob.push_back(bits == 4
            ? static_cast<std::uint8_t>(0x21u + ((index * 17u) & 0x44u))
            : static_cast<std::uint8_t>(0x38u + (index & 3u)));
    }
    const auto scale_bytes = static_cast<std::size_t>(
        bits == 4 ? output_size : (output_size + 127) / 128) *
        static_cast<std::size_t>(bits == 4 ? input_size / 32 : input_size / 128);
    blob.insert(blob.end(), scale_bytes, 127u);
    return blob;
}

array make_input(int width, int rows = 1) {
    std::vector<float> values(static_cast<std::size_t>(rows) * width);
    for (int row = 0; row < rows; ++row) {
        for (int index = 0; index < width; ++index) {
            values[static_cast<std::size_t>(row) * width + index] =
                static_cast<float>(
                    (index * 17 + row * 29 + 11) % 127 - 63) / 256.0f;
        }
    }
    return mlx::core::astype(
        array(values.begin(), Shape{rows, width}),
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

void benchmark_weight(
    std::string_view dtype,
    const std::string& name,
    MlxMxWeight weight,
    int rows,
    int repetitions) {
    const auto source = make_input(weight.input_size(), rows);

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
    for (int index = 0; index < rows * weight.output_size(); ++index) {
        const float value = values[index];
        require(
            std::isfinite(value),
            "benchmark output contains non-finite values");
        checksum += static_cast<double>(value);
        maximum = std::max(maximum, std::fabs(value));
    }

    std::cout
        << dtype << '\t'
        << name << '\t'
        << rows << 'x' << weight.input_size() << 'x'
        << weight.output_size() << '\t'
        << weight.packed_nbytes() << '\t'
        << std::fixed << std::setprecision(3) << mean_ms << '\t'
        << std::setprecision(1) << decimal_gbps << '\t'
        << std::setprecision(6) << checksum << '\t'
        << maximum << '\n';
}

void benchmark(
    const MfqContainer& model,
    const std::string& name,
    int repetitions) {
    const auto& record = model.record(name);
    require(
        mfq::metal::is_mx_dtype(record.dtype),
        "benchmark tensor is not MXFP4/MXFP8: " + name);
    benchmark_weight(
        record.dtype,
        name,
        MlxMxWeight::from_blob(record.dtype, model.read(name)),
        1,
        repetitions);
}

void benchmark_grouped_synthetic(
    int bits,
    int rows,
    int output_size,
    int input_size,
    int projections,
    int repetitions) {
    require(
        projections >= 2 && projections <= 3,
        "synthetic grouped projection count must be 2 or 3");
    const std::string dtype = bits == 4 ? "MXFP4" : "MXFP8";
    std::vector<MlxMxWeight> weights;
    weights.reserve(static_cast<std::size_t>(projections));
    for (int projection = 0; projection < projections; ++projection) {
        weights.push_back(MlxMxWeight::from_blob(
            dtype,
            make_synthetic_mx_blob(bits, output_size, input_size)));
    }
    std::vector<MlxGroupedLinearWeightRef> refs;
    refs.reserve(weights.size());
    std::size_t bytes = 0;
    for (const auto& weight : weights) {
        refs.emplace_back(&weight);
        bytes += weight.packed_nbytes();
    }
    const MlxGroupedLinear grouped(std::move(refs));
    const auto source = make_input(input_size, rows);
    auto execute = [&] {
        auto result = grouped(source);
        mlx::core::eval(result);
        return result;
    };
    auto result = execute();
    for (int index = 0; index < 3; ++index) {
        result = execute();
    }
    mlx::core::synchronize();
    const auto started = Clock::now();
    for (int index = 0; index < repetitions; ++index) {
        result = execute();
    }
    mlx::core::synchronize();
    const double mean_ms = milliseconds_since(started) / repetitions;
    double checksum = 0.0;
    float maximum = 0.0f;
    for (auto& projection : result) {
        auto checked = mlx::core::contiguous(
            mlx::core::astype(projection, mlx::core::float32));
        mlx::core::eval(checked);
        for (std::size_t index = 0; index < checked.size(); ++index) {
            const float value = checked.data<float>()[index];
            require(
                std::isfinite(value),
                "grouped MX benchmark produced a non-finite value");
            checksum += value;
            maximum = std::max(maximum, std::fabs(value));
        }
    }
    std::cout
        << dtype << "\tsynthetic_mx_x" << projections << '\t'
        << rows << 'x' << input_size << 'x'
        << output_size * projections << '\t'
        << bytes << '\t'
        << std::fixed << std::setprecision(3) << mean_ms << '\t'
        << std::setprecision(1)
        << static_cast<double>(bytes) / (mean_ms * 1.0e6) << '\t'
        << std::setprecision(6) << checksum << '\t' << maximum << '\n';
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
        if (argc >= 2 && std::string(argv[1]) == "--synthetic-grouped") {
            require(
                argc == 8,
                "usage: mfq-metal-mx-benchmark --synthetic-grouped "
                "BITS REPETITIONS ROWS OUTPUT INPUT PROJECTIONS");
            const int bits = std::stoi(argv[2]);
            const int repetitions = std::stoi(argv[3]);
            const int rows = std::stoi(argv[4]);
            const int output = std::stoi(argv[5]);
            const int input = std::stoi(argv[6]);
            const int projections = std::stoi(argv[7]);
            require(repetitions > 0, "repetitions must be positive");
            require(rows >= 1 && rows <= 16, "rows must be in [1, 16]");
            std::cout
                << "dtype\ttensor\tshape\tpacked_bytes\tms\tGB/s\t"
                   "checksum\tmax_abs\n";
            benchmark_grouped_synthetic(
                bits,
                rows,
                output,
                input,
                projections,
                repetitions);
            return 0;
        }
        if (argc >= 2 && std::string(argv[1]) == "--synthetic") {
            require(
                argc == 7,
                "usage: mfq-metal-mx-benchmark --synthetic "
                "BITS REPETITIONS ROWS OUTPUT INPUT");
            const int bits = std::stoi(argv[2]);
            const int repetitions = std::stoi(argv[3]);
            const int rows = std::stoi(argv[4]);
            const int output = std::stoi(argv[5]);
            const int input = std::stoi(argv[6]);
            require(repetitions > 0, "repetitions must be positive");
            require(rows >= 1 && rows <= 16, "rows must be in [1, 16]");
            const std::string dtype = bits == 4 ? "MXFP4" : "MXFP8";
            std::cout
                << "dtype\ttensor\tshape\tpacked_bytes\tms\tGB/s\t"
                   "checksum\tmax_abs\n";
            benchmark_weight(
                dtype,
                "synthetic_mx",
                MlxMxWeight::from_blob(
                    dtype,
                    make_synthetic_mx_blob(bits, output, input)),
                rows,
                repetitions);
            return 0;
        }
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
