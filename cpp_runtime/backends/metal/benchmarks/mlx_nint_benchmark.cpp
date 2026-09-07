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
#include <string_view>
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

template <typename T>
void append(std::vector<std::uint8_t>& target, T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    target.insert(target.end(), bytes, bytes + sizeof(T));
}

std::vector<std::uint8_t> make_synthetic_q8_blob(
    int output_size,
    int input_size) {
    if (output_size <= 0 || input_size <= 0 || input_size % 32 != 0) {
        throw std::runtime_error("invalid synthetic NINT8-0 geometry");
    }
    const int groups = input_size / 32;
    std::vector<std::uint8_t> blob{'N', 'I', '8', '0'};
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input_size);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, input_size);
    append<std::uint32_t>(blob, output_size);
    append<std::uint32_t>(blob, groups);
    for (int output = 0; output < output_size; ++output) {
        for (int group = 0; group < groups; ++group) {
            append<std::uint16_t>(blob, 0x2800u);
            for (int element = 0; element < 32; ++element) {
                const int value =
                    (output * 13 + group * 7 + element * 5) % 31 - 15;
                append<std::int8_t>(
                    blob,
                    static_cast<std::int8_t>(value));
            }
        }
    }
    return blob;
}

std::vector<std::uint8_t> make_synthetic_nint_blob(
    int bits,
    int group_size,
    int output_size,
    int input_size) {
    constexpr int sub_bits = 1;
    if (bits < 1 || bits > 8 ||
        group_size <= 0 || output_size <= 0 || input_size <= 0) {
        throw std::runtime_error("invalid synthetic NINT geometry");
    }
    const int groups = (input_size + group_size - 1) / group_size;
    const auto metadata_count =
        static_cast<std::size_t>(output_size) * groups;
    const auto value_count = metadata_count * group_size;

    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(blob, bits);
    append<std::uint8_t>(blob, sub_bits);
    append<std::int32_t>(blob, group_size);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input_size);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, input_size);
    append<std::uint32_t>(blob, output_size);
    append<std::uint32_t>(blob, groups);
    for (int output = 0; output < output_size; ++output) {
        append<std::uint16_t>(blob, 0x3000u);
    }
    for (int output = 0; output < output_size; ++output) {
        append<std::uint16_t>(blob, 0u);
    }

    blob.insert(blob.end(), (metadata_count + 7) / 8, 0xffu);
    blob.insert(blob.end(), (metadata_count + 7) / 8, 0u);
    std::vector<std::uint8_t> packed((value_count * bits + 7) / 8, 0u);
    std::uint64_t bit_buffer = 0;
    int buffered_bits = 0;
    std::size_t packed_index = 0;
    for (std::size_t index = 0; index < value_count; ++index) {
        const auto value = static_cast<std::uint8_t>(
            (index * 13u + 7u) & ((1u << bits) - 1u));
        bit_buffer |= static_cast<std::uint64_t>(value) << buffered_bits;
        buffered_bits += bits;
        while (buffered_bits >= 8) {
            packed[packed_index++] =
                static_cast<std::uint8_t>(bit_buffer);
            bit_buffer >>= 8;
            buffered_bits -= 8;
        }
    }
    if (buffered_bits != 0) {
        packed[packed_index] = static_cast<std::uint8_t>(bit_buffer);
    }
    blob.insert(blob.end(), packed.begin(), packed.end());
    return blob;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

array make_input(int rows, int width) {
    std::vector<float> values(
        static_cast<std::size_t>(rows) * width);
    for (int row = 0; row < rows; ++row) {
        for (int index = 0; index < width; ++index) {
            values[
                static_cast<std::size_t>(row) * width + index
            ] = static_cast<float>(
                (index * 17 + row * 29 + 11) % 127 - 63) / 256.0f;
        }
    }
    return mlx::core::astype(
        array(values.begin(), Shape{rows, width}),
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

void benchmark_case(
    BenchmarkCase benchmark_case,
    int rows,
    int warmup,
    int repetitions) {
    const int input = input_size(benchmark_case.weight);
    const int output = output_size(benchmark_case.weight);
    const auto bytes = packed_nbytes(benchmark_case.weight);
    const auto source = make_input(rows, input);

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
    for (int index = 0; index < rows * output; ++index) {
        const float value = values[index];
        require(
            std::isfinite(value),
            "benchmark output contains non-finite values");
        checksum += static_cast<double>(value);
        maximum = std::max(maximum, std::fabs(value));
    }
    const auto hash = output_hash(result);

    std::cout
        << benchmark_case.dtype << '\t'
        << benchmark_case.name << '\t'
        << rows << 'x' << input << 'x' << output << '\t'
        << bytes << '\t'
        << std::fixed << std::setprecision(3) << mean_ms << '\t'
        << std::setprecision(1) << decimal_gbps << '\t'
        << rows << '\t'
        << std::setprecision(6) << checksum << '\t'
        << maximum << '\t' << std::hex << hash << std::dec << '\n';
}

void benchmark(
    const MfqContainer& model,
    const std::string& name,
    int rows,
    int warmup,
    int repetitions) {
    benchmark_case(
        load_case(model, name),
        rows,
        warmup,
        repetitions);
}

std::vector<BenchmarkCase> load_layer_stack(
    const MfqContainer& model,
    const std::string& pattern) {
    constexpr std::string_view marker = "{layer}";
    const auto marker_at = pattern.find(marker);
    require(
        marker_at != std::string::npos,
        "layer-stack pattern is missing {layer}");
    std::vector<BenchmarkCase> result;
    for (int layer = 0;; ++layer) {
        auto name = pattern;
        name.replace(
            marker_at,
            marker.size(),
            std::to_string(layer));
        if (!model.contains(name)) break;
        result.push_back(load_case(model, name));
    }
    require(!result.empty(), "layer-stack pattern matched no tensors");
    return result;
}

void benchmark_layer_stack(
    const MfqContainer& model,
    const std::string& pattern,
    int rows,
    int warmup,
    int repetitions) {
    auto cases = load_layer_stack(model, pattern);
    const int input = input_size(cases.front().weight);
    const int output = output_size(cases.front().weight);
    std::size_t bytes = 0;
    for (const auto& item : cases) {
        require(
            input_size(item.weight) == input &&
                output_size(item.weight) == output,
            "layer-stack tensor shapes disagree");
        bytes += packed_nbytes(item.weight);
    }
    const auto source = make_input(rows, input);
    const auto execute = [&] {
        std::vector<array> outputs;
        outputs.reserve(cases.size());
        for (const auto& item : cases) {
            outputs.push_back(matmul(item.weight, source));
        }
        mlx::core::eval(outputs);
        return outputs;
    };

    auto outputs = execute();
    for (int index = 0; index < warmup; ++index) {
        outputs = execute();
    }
    mlx::core::synchronize();

    const auto started = Clock::now();
    for (int index = 0; index < repetitions; ++index) {
        outputs = execute();
    }
    mlx::core::synchronize();
    const double total_ms = milliseconds_since(started);
    const auto launches = cases.size();
    const double mean_dispatch_ms = total_ms /
        static_cast<double>(repetitions * launches);
    const double decimal_gbps =
        static_cast<double>(bytes) * repetitions /
        (total_ms * 1.0e6);

    auto checked = mlx::core::astype(outputs.back(), mlx::core::float32);
    mlx::core::eval(checked);
    const auto* values = checked.data<float>();
    double checksum = 0.0;
    float maximum = 0.0f;
    for (int index = 0; index < rows * output; ++index) {
        const float value = values[index];
        require(
            std::isfinite(value),
            "layer-stack output contains non-finite values");
        checksum += static_cast<double>(value);
        maximum = std::max(maximum, std::fabs(value));
    }

    std::cout
        << cases.front().dtype << '\t'
        << pattern << '\t'
        << rows << 'x' << input << 'x' << output << 'x' << launches << '\t'
        << bytes << '\t'
        << std::fixed << std::setprecision(3)
        << mean_dispatch_ms << '\t'
        << std::setprecision(1) << decimal_gbps << '\t'
        << launches << '\t'
        << std::setprecision(6) << checksum << '\t'
        << maximum << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(
            argc >= 2,
            "usage: mfq-metal-nint-benchmark MODEL.mfq [REPETITIONS] "
            "[--rows ROWS] [TENSOR ...] | --synthetic-q8|"
            "--synthetic-nint5 REPETITIONS ROWS OUTPUT INPUT | "
            "--synthetic-nint BITS GROUP_SIZE REPETITIONS ROWS "
            "OUTPUT INPUT");
        if (std::string_view(argv[1]) == "--synthetic-q8") {
            require(
                argc == 6,
                "usage: mfq-metal-nint-benchmark --synthetic-q8 "
                "REPETITIONS ROWS OUTPUT INPUT");
            const int repetitions = std::stoi(argv[2]);
            const int rows = std::stoi(argv[3]);
            const int output = std::stoi(argv[4]);
            const int input = std::stoi(argv[5]);
            require(repetitions > 0, "repetitions must be positive");
            require(rows >= 0 && rows <= 16, "rows must be in [0, 16]");
            std::cout
                << "dtype\ttensor\tshape\tpacked_bytes\tms\tGB/s\t"
                   "launches\tchecksum\tmax_abs\thash\n";
            BenchmarkCase benchmark{
                "synthetic_q8",
                "NINT8-0",
                MlxNint8ZeroWeight::from_blob(
                    make_synthetic_q8_blob(output, input)),
            };
            if (rows == 0) {
                for (int small_m = 2; small_m <= 6; ++small_m) {
                    benchmark_case(
                        benchmark, small_m, 3, repetitions);
                }
            } else {
                benchmark_case(benchmark, rows, 3, repetitions);
            }
            return 0;
        }
        if (std::string_view(argv[1]) == "--synthetic-nint5") {
            require(
                argc == 6,
                "usage: mfq-metal-nint-benchmark --synthetic-nint5 "
                "REPETITIONS ROWS OUTPUT INPUT");
            const int repetitions = std::stoi(argv[2]);
            const int rows = std::stoi(argv[3]);
            const int output = std::stoi(argv[4]);
            const int input = std::stoi(argv[5]);
            require(repetitions > 0, "repetitions must be positive");
            require(rows >= 0 && rows <= 16, "rows must be in [0, 16]");
            std::cout
                << "dtype\ttensor\tshape\tpacked_bytes\tms\tGB/s\t"
                   "launches\tchecksum\tmax_abs\thash\n";
            BenchmarkCase benchmark{
                "synthetic_nint5",
                "NINT5",
                MlxNintWeight::from_blob(
                    make_synthetic_nint_blob(5, 28, output, input)),
            };
            if (rows == 0) {
                for (int small_m = 2; small_m <= 6; ++small_m) {
                    benchmark_case(
                        benchmark, small_m, 3, repetitions);
                }
            } else {
                benchmark_case(benchmark, rows, 3, repetitions);
            }
            return 0;
        }
        if (std::string_view(argv[1]) == "--synthetic-nint") {
            require(
                argc == 8,
                "usage: mfq-metal-nint-benchmark --synthetic-nint "
                "BITS GROUP_SIZE REPETITIONS ROWS OUTPUT INPUT");
            const int bits = std::stoi(argv[2]);
            const int group_size = std::stoi(argv[3]);
            const int repetitions = std::stoi(argv[4]);
            const int rows = std::stoi(argv[5]);
            const int output = std::stoi(argv[6]);
            const int input = std::stoi(argv[7]);
            require(repetitions > 0, "repetitions must be positive");
            require(rows >= 0 && rows <= 16, "rows must be in [0, 16]");
            std::cout
                << "dtype\ttensor\tshape\tpacked_bytes\tms\tGB/s\t"
                   "launches\tchecksum\tmax_abs\thash\n";
            BenchmarkCase benchmark{
                "synthetic_nint" + std::to_string(bits) + "_gs" +
                    std::to_string(group_size),
                "NINT" + std::to_string(bits),
                MlxNintWeight::from_blob(make_synthetic_nint_blob(
                    bits, group_size, output, input)),
            };
            if (rows == 0) {
                for (int small_m = 2; small_m <= 6; ++small_m) {
                    benchmark_case(
                        benchmark, small_m, 3, repetitions);
                }
            } else {
                benchmark_case(benchmark, rows, 3, repetitions);
            }
            return 0;
        }
        const int repetitions =
            argc >= 3 ? std::stoi(argv[2]) : 20;
        int rows = 1;
        int first_tensor = 3;
        if (argc >= 5 && std::string_view(argv[3]) == "--rows") {
            rows = std::stoi(argv[4]);
            first_tensor = 5;
        }
        require(repetitions > 0, "repetitions must be positive");
        require(rows >= 1 && rows <= 16, "rows must be in [1, 16]");

        const MfqContainer model(argv[1]);
        std::cout
            << "dtype\ttensor\tshape\tpacked_bytes\tms\tGB/s\t"
               "launches\tchecksum\tmax_abs\thash\n";
        std::vector<std::string> names;
        if (argc > first_tensor) {
            names.assign(argv + first_tensor, argv + argc);
        } else {
            names = {
                "blk.0.ffn_gate.weight",
                "blk.0.attn_gate.weight",
                "blk.0.ffn_down.weight",
                "blk.0.ssm_out.weight",
                "blk.3.attn_output.weight",
                "blk.3.attn_v.weight",
                "output.weight",
            };
        }
        for (const auto& name : names) {
            if (name.find("{layer}") != std::string::npos) {
                benchmark_layer_stack(model, name, rows, 3, repetitions);
            } else {
                benchmark(model, name, rows, 3, repetitions);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MFQ NINT Metal benchmark failed: "
                  << error.what() << '\n';
        return 1;
    }
}
