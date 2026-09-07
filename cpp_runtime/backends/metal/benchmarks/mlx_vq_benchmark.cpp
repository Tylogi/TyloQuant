#include "mfq_container.h"
#include "mlx_grouped_linear.h"
#include "mlx_vq.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
using mfq::metal::MlxVqWeight;

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

std::vector<std::uint8_t> make_synthetic_group64_blob(
    int output_size,
    int input_size) {
    require(
        output_size > 0 && input_size > 0 && input_size % 8 == 0,
        "invalid synthetic group64 geometry");
    constexpr int group_size = 24;
    constexpr int vector_size = 8;
    constexpr int index_bits = 12;
    constexpr int states = 16;
    constexpr int code_banks = 2;
    constexpr int entries = 1 << index_bits;
    constexpr std::uint8_t jsc_profile = 0x20u | 5u;
    const int groups = (input_size + group_size - 1) / group_size;
    const int vectors = (input_size + vector_size - 1) / vector_size;

    std::vector<std::uint8_t> blob{'N', 'V', 'Q', '1'};
    append<std::uint8_t>(blob, jsc_profile);
    append<std::uint8_t>(blob, 4);
    append<std::uint16_t>(blob, group_size);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input_size);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, input_size);
    append<std::uint32_t>(blob, output_size);

    append<std::uint8_t>(blob, 2);
    append<std::uint8_t>(blob, code_banks);
    append<std::uint8_t>(blob, states);
    append<std::uint8_t>(blob, 0);
    for (int state = 0; state < states; ++state) {
        append<std::uint16_t>(blob, 0x3c00u);
    }
    for (int state = 0; state < states; ++state) {
        append<std::uint8_t>(
            blob,
            static_cast<std::uint8_t>(state & 1));
    }
    append<std::uint8_t>(blob, 1);
    blob.insert(blob.end(), 11, 0);

    for (int bank = 0; bank < code_banks; ++bank) {
        for (int entry = 0; entry < entries; ++entry) {
            for (int component = 0; component < vector_size; ++component) {
                const int value =
                    ((entry * 5 + component * 3 + bank * 7) % 15) - 7;
                append<std::int8_t>(
                    blob,
                    static_cast<std::int8_t>(value));
            }
        }
    }
    for (int output = 0; output < output_size; ++output) {
        append<std::uint16_t>(blob, 0x3000u);
    }
    for (int output = 0; output < output_size; ++output) {
        for (int group = 0; group < groups; ++group) {
            std::uint64_t record = 0;
            for (int vector = 0; vector < 3; ++vector) {
                if (group * 3 + vector >= vectors) {
                    continue;
                }
                const std::uint64_t index = static_cast<std::uint64_t>(
                    (output * 17 + group * 13 + vector * 101) & 4095);
                const auto sign7 = static_cast<std::uint8_t>(
                    (output * 11 + group * 7 + vector * 29) & 127);
                const std::uint64_t sign8 = static_cast<std::uint64_t>(
                    sign7 | ((std::popcount(sign7) & 1u) << 7));
                record |= (index | (sign8 << 12)) << (vector * 20);
            }
            record |= static_cast<std::uint64_t>(
                (output + group) & 15) << 60;
            append<std::uint64_t>(blob, record);
        }
    }
    return blob;
}

template <typename Generator>
void append_packed_values(
    std::vector<std::uint8_t>& target,
    std::size_t count,
    int bits,
    Generator&& generator) {
    const auto begin = target.size();
    target.resize(begin + (count * static_cast<std::size_t>(bits) + 7) / 8);
    for (std::size_t index = 0; index < count; ++index) {
        const auto value = static_cast<std::uint32_t>(generator(index));
        const auto bit_offset = index * static_cast<std::size_t>(bits);
        for (int bit = 0; bit < bits; ++bit) {
            if (((value >> bit) & 1u) != 0u) {
                const auto destination = bit_offset + static_cast<std::size_t>(bit);
                target[begin + destination / 8] |= static_cast<std::uint8_t>(
                    1u << (destination & 7));
            }
        }
    }
}

std::vector<std::uint8_t> make_synthetic_streams_blob(
    int output_size,
    int input_size) {
    require(
        output_size > 0 && input_size > 0 && input_size % 8 == 0,
        "invalid synthetic streams geometry");
    constexpr int group_size = 24;
    constexpr int vector_size = 8;
    constexpr int index_bits = 12;
    constexpr int states = 16;
    constexpr int code_banks = 2;
    constexpr int entries = 1 << index_bits;
    constexpr std::uint8_t jsc_profile = 0x20u | 5u;
    const int groups = (input_size + group_size - 1) / group_size;
    const int vectors = (input_size + vector_size - 1) / vector_size;

    std::vector<std::uint8_t> blob{'N', 'V', 'Q', '1'};
    append<std::uint8_t>(blob, jsc_profile);
    append<std::uint8_t>(blob, 4);
    append<std::uint16_t>(blob, group_size);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input_size);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, input_size);
    append<std::uint32_t>(blob, output_size);

    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, code_banks);
    append<std::uint8_t>(blob, states);
    append<std::uint8_t>(blob, 0);
    for (int state = 0; state < states; ++state) {
        append<std::uint16_t>(blob, 0x3c00u);
    }
    for (int state = 0; state < states; ++state) {
        append<std::uint8_t>(
            blob,
            static_cast<std::uint8_t>(state & 1));
    }
    append<std::uint8_t>(blob, 0);
    blob.insert(blob.end(), 11, 0);

    for (int bank = 0; bank < code_banks; ++bank) {
        for (int entry = 0; entry < entries; ++entry) {
            for (int component = 0; component < vector_size; ++component) {
                const int value =
                    ((entry * 5 + component * 3 + bank * 7) % 15) - 7;
                append<std::int8_t>(
                    blob,
                    static_cast<std::int8_t>(value));
            }
        }
    }
    for (int output = 0; output < output_size; ++output) {
        append<std::uint16_t>(blob, 0x3000u);
    }
    const auto outputs = static_cast<std::size_t>(output_size);
    append_packed_values(
        blob,
        outputs * static_cast<std::size_t>(groups),
        4,
        [](std::size_t index) { return index * 7u + 3u; });
    append_packed_values(
        blob,
        outputs * static_cast<std::size_t>(vectors),
        index_bits,
        [](std::size_t index) { return index * 13u + 17u; });
    append_packed_values(
        blob,
        outputs * static_cast<std::size_t>(vectors),
        7,
        [](std::size_t index) { return index * 11u + 5u; });
    return blob;
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

void benchmark_weight(
    std::string_view dtype,
    const std::string& name,
    MlxVqWeight weight,
    int rows,
    int warmup,
    int repetitions) {
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
        << dtype << '\t'
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

void benchmark(
    const MfqContainer& model,
    const std::string& name,
    int rows,
    int warmup,
    int repetitions) {
    const auto& record = model.record(name);
    benchmark_weight(
        record.dtype,
        name,
        load_weight(model, name),
        rows,
        warmup,
        repetitions);
}

void benchmark_grouped_synthetic(
    int rows,
    int output_size,
    int input_size,
    int projections,
    int warmup,
    int repetitions) {
    require(
        projections >= 2 && projections <= 3,
        "synthetic grouped projection count must be 2 or 3");
    std::vector<MlxVqWeight> weights;
    weights.reserve(static_cast<std::size_t>(projections));
    for (int projection = 0; projection < projections; ++projection) {
        weights.push_back(MlxVqWeight::from_blob(
            "NVQ2J-XL",
            make_synthetic_group64_blob(output_size, input_size)));
    }
    std::vector<MlxGroupedLinearWeightRef> refs;
    refs.reserve(weights.size());
    std::size_t bytes = 0;
    for (const auto& weight : weights) {
        refs.emplace_back(&weight);
        bytes += weight.packed_nbytes();
    }
    const MlxGroupedLinear grouped(std::move(refs));
    const auto source = make_input(rows, input_size);
    auto execute = [&] {
        auto result = grouped(source);
        mlx::core::eval(result);
        return result;
    };
    auto result = execute();
    for (int index = 0; index < warmup; ++index) {
        result = execute();
    }
    mlx::core::synchronize();
    const auto started = Clock::now();
    for (int index = 0; index < repetitions; ++index) {
        result = execute();
    }
    mlx::core::synchronize();
    const double mean_ms =
        milliseconds_since(started) / repetitions;
    double checksum = 0.0;
    float maximum = 0.0f;
    std::uint64_t hash = 1469598103934665603ull;
    for (auto& projection : result) {
        auto checked = mlx::core::contiguous(
            mlx::core::astype(projection, mlx::core::float32));
        mlx::core::eval(checked);
        for (std::size_t index = 0; index < checked.size(); ++index) {
            const float value = checked.data<float>()[index];
            require(
                std::isfinite(value),
                "grouped benchmark output contains non-finite values");
            checksum += value;
            maximum = std::max(maximum, std::fabs(value));
        }
        hash ^= output_hash(std::move(projection));
        hash *= 1099511628211ull;
    }
    std::cout
        << "NVQ2J-XL\tsynthetic_group64_x" << projections << '\t'
        << rows << 'x' << input_size << 'x'
        << (output_size * projections) << "\t24x8x12x4\t"
        << bytes << '\t'
        << std::fixed << std::setprecision(3) << mean_ms << '\t'
        << std::setprecision(1)
        << static_cast<double>(bytes) / (mean_ms * 1.0e6) << '\t'
        << std::setprecision(6) << checksum << '\t'
        << maximum << '\t' << std::hex << hash << std::dec << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(
            argc >= 5,
            "usage: mfq-metal-vq-benchmark MODEL.mfq REPETITIONS ROWS "
            "TENSOR [TENSOR ...] | --synthetic-group64 REPETITIONS "
            "ROWS OUTPUT INPUT | --synthetic-group64-grouped REPETITIONS "
            "ROWS OUTPUT INPUT PROJECTIONS | --synthetic-streams "
            "REPETITIONS ROWS OUTPUT INPUT");
        const int repetitions = std::stoi(argv[2]);
        const int rows = std::stoi(argv[3]);
        require(repetitions > 0, "repetitions must be positive");
        require(rows >= 1 && rows <= 16, "rows must be in [1, 16]");

        std::cout
            << "dtype\ttensor\tshape\tlayout\tpacked_bytes\tms\tGB/s\t"
               "checksum\tmax_abs\thash\n";
        if (std::string_view(argv[1]) == "--synthetic-group64") {
            require(
                argc == 6,
                "usage: mfq-metal-vq-benchmark --synthetic-group64 "
                "REPETITIONS ROWS OUTPUT INPUT");
            const int output = std::stoi(argv[4]);
            const int input = std::stoi(argv[5]);
            benchmark_weight(
                "NVQ2J-XL",
                "synthetic_group64",
                MlxVqWeight::from_blob(
                    "NVQ2J-XL",
                    make_synthetic_group64_blob(output, input)),
                rows,
                3,
                repetitions);
            return 0;
        }
        if (std::string_view(argv[1]) ==
            "--synthetic-group64-grouped") {
            require(
                argc == 7,
                "usage: mfq-metal-vq-benchmark "
                "--synthetic-group64-grouped REPETITIONS ROWS "
                "OUTPUT INPUT PROJECTIONS");
            benchmark_grouped_synthetic(
                rows,
                std::stoi(argv[4]),
                std::stoi(argv[5]),
                std::stoi(argv[6]),
                3,
                repetitions);
            return 0;
        }
        if (std::string_view(argv[1]) == "--synthetic-streams") {
            require(
                argc == 6,
                "usage: mfq-metal-vq-benchmark --synthetic-streams "
                "REPETITIONS ROWS OUTPUT INPUT");
            const int output = std::stoi(argv[4]);
            const int input = std::stoi(argv[5]);
            benchmark_weight(
                "NVQ2J-XL",
                "synthetic_streams",
                MlxVqWeight::from_blob(
                    "NVQ2J-XL",
                    make_synthetic_streams_blob(output, input)),
                rows,
                3,
                repetitions);
            return 0;
        }
        const MfqContainer model(argv[1]);
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
