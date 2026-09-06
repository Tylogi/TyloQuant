#include "mlx_moe_ops.h"
#include "mlx_ssd_expert_arena.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/stream.h>

namespace {

using Clock = std::chrono::steady_clock;
using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MlxDeepseekV4SsdExpertArena;
using mfq::metal::MlxDeepseekV4SsdExpertWeights;

constexpr int kHidden = 4096;
constexpr int kRouted = 2048;
constexpr int kExperts = 256;
constexpr int kRoutes = 6;
constexpr int kMaximumRows = 6;
constexpr int kSlots = kMaximumRows * kRoutes;
constexpr std::size_t kGateUpBytesPerExpert =
    2ull * 2048ull * 2048ull + 2ull * 2048ull * 128ull;
constexpr std::size_t kDownBytesPerExpert =
    2048ull * 2048ull + 2048ull * 128ull;

struct Options {
    int rows = 5;
    int repetitions = 20;
    int trials = 5;
    int eval_batch = 1;
    std::string pattern = "unique";
};

enum class Path {
    sorted,
    unsorted,
    omlx,
    nax,
    adaptive,
    row_replay,
};

enum class Stage {
    gate_up,
    down,
    full,
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int parse_positive(const char* text, const char* option) {
    try {
        std::size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        if (consumed != std::string(text).size() || value <= 0) {
            throw std::invalid_argument("not positive");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error(
            std::string("invalid value for ") + option + ": " + text);
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&](const char* option) -> const char* {
            if (index + 1 >= argc) {
                throw std::runtime_error(
                    std::string("missing value for ") + option);
            }
            return argv[++index];
        };
        if (argument == "--rows") {
            options.rows = parse_positive(value("--rows"), "--rows");
        } else if (argument == "--repetitions") {
            options.repetitions = parse_positive(
                value("--repetitions"), "--repetitions");
        } else if (argument == "--trials") {
            options.trials = parse_positive(value("--trials"), "--trials");
        } else if (argument == "--eval-batch") {
            options.eval_batch = parse_positive(
                value("--eval-batch"), "--eval-batch");
        } else if (argument == "--pattern") {
            options.pattern = value("--pattern");
        } else if (argument == "--help") {
            std::cout
                << "Usage: mfq-metal-deepseek-v4-moe-smallm-benchmark "
                   "[--rows 2..6] [--pattern unique|repeat|mixed] "
                   "[--repetitions N] [--trials N] [--eval-batch N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    require(options.rows >= 2 && options.rows <= kMaximumRows,
            "--rows must be in [2, 6]");
    require(options.pattern == "unique" || options.pattern == "repeat" ||
                options.pattern == "mixed",
            "--pattern must be unique, repeat, or mixed");
    return options;
}

void fill_slot(
    MlxDeepseekV4SsdExpertArena& arena,
    std::size_t slot) {
    auto destination = arena.destination(slot);
    for (const auto bytes : {
             destination.w1_scale,
             destination.w2_scale,
             destination.w3_scale,
         }) {
        std::fill(bytes.begin(), bytes.end(), std::byte{127});
    }
    for (const auto bytes : {
             destination.w1_weight,
             destination.w2_weight,
             destination.w3_weight,
         }) {
        // 0x22 is two +1 E2M1 values.  Keeping every slot equal makes the
        // route-order comparison bit-for-bit checkable while still forcing
        // the production kernel to consume the complete packed matrices.
        std::fill(bytes.begin(), bytes.end(), std::byte{0x22});
    }
}

array make_input(int rows, int columns, int salt) {
    std::vector<float> values(
        static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            values[static_cast<std::size_t>(row) * columns + column] =
                static_cast<float>(
                    (column * 17 + row * 29 + salt) % 127 - 63) /
                1024.0f;
        }
    }
    return mlx::core::astype(
        array(values.begin(), Shape{rows, columns}),
        mlx::core::float16);
}

array make_ids(const Options& options) {
    std::vector<std::int32_t> ids(
        static_cast<std::size_t>(options.rows) * kRoutes);
    for (int row = 0; row < options.rows; ++row) {
        for (int route = 0; route < kRoutes; ++route) {
            const int linear = row * kRoutes + route;
            // A fixed full-period permutation avoids giving the unsorted
            // path an unrealistically sequential expert-bank traversal.
            int expert = (linear * 17 + 5) % kSlots;
            if (options.pattern == "repeat") {
                expert = (route * 5 + row) % kRoutes;
            } else if (options.pattern == "mixed" && route < 3) {
                expert = route;
            } else if (options.pattern == "mixed") {
                const int unique = row * 3 + route - 3;
                expert = 3 + (unique * 7 + 5) % (kSlots - 3);
            }
            ids[static_cast<std::size_t>(row) * kRoutes + route] = expert;
        }
    }
    return array(ids.begin(), Shape{options.rows, kRoutes});
}

array make_route_weights(int rows) {
    std::vector<float> values(
        static_cast<std::size_t>(rows) * kRoutes,
        1.0f / static_cast<float>(kRoutes));
    return array(values.begin(), Shape{rows, kRoutes});
}

const char* path_name(Path path) {
    switch (path) {
        case Path::sorted:
            return "sorted";
        case Path::unsorted:
            return "unsorted";
        case Path::omlx:
            return "omlx";
        case Path::nax:
            return "nax";
        case Path::adaptive:
            return "adaptive";
        case Path::row_replay:
            return "row_replay";
    }
    return "unknown";
}

const char* stage_name(Stage stage) {
    switch (stage) {
        case Stage::gate_up:
            return "gate_up_swiglu";
        case Stage::down:
            return "down_reduce";
        case Stage::full:
            return "full_ffn";
    }
    return "unknown";
}

array row_slice(const array& input, int row) {
    Shape starts(input.ndim(), 0);
    Shape stops = input.shape();
    starts[0] = row;
    stops[0] = row + 1;
    return mlx::core::slice(input, starts, stops);
}

class Benchmark {
public:
    explicit Benchmark(const Options& options)
        : options_(options),
          arena_(kSlots),
          input_(make_input(options.rows, kHidden, 11)),
          down_input_(mlx::core::broadcast_to(
              mlx::core::expand_dims(
                  make_input(options.rows, kRouted, 37),
                  1),
              Shape{options.rows, kRoutes, kRouted})),
          ids_(make_ids(options)),
          route_weights_(make_route_weights(options.rows)),
          routed_(make_routed_weights()),
          omlx_up_weight_(mlx::core::full(
              Shape{kSlots, kRouted, kHidden / 8},
              static_cast<std::uint32_t>(0x22222222u),
              mlx::core::uint32)),
          omlx_gate_weight_(mlx::core::full(
              Shape{kSlots, kRouted, kHidden / 8},
              static_cast<std::uint32_t>(0x22222222u),
              mlx::core::uint32)),
          omlx_down_weight_(mlx::core::full(
              Shape{kSlots, kHidden, kRouted / 8},
              static_cast<std::uint32_t>(0x22222222u),
              mlx::core::uint32)),
          omlx_up_scale_(mlx::core::full(
              Shape{kSlots, kRouted, kHidden / 32},
              static_cast<std::uint8_t>(127),
              mlx::core::uint8)),
          omlx_gate_scale_(mlx::core::full(
              Shape{kSlots, kRouted, kHidden / 32},
              static_cast<std::uint8_t>(127),
              mlx::core::uint8)),
          omlx_down_scale_(mlx::core::full(
              Shape{kSlots, kHidden, kRouted / 32},
              static_cast<std::uint8_t>(127),
              mlx::core::uint8)) {
        mlx::core::eval(
            input_,
            down_input_,
            ids_,
            route_weights_,
            omlx_up_weight_,
            omlx_gate_weight_,
            omlx_down_weight_,
            omlx_up_scale_,
            omlx_gate_scale_,
            omlx_down_scale_);
        mlx::core::synchronize();
    }

    array operation(Path path, Stage stage) const {
        setenv(
            "MFQ_METAL_NINTM_SORT_ROUTES",
            path == Path::sorted || path == Path::adaptive ? "1" : "0",
            1);
        setenv(
            "MFQ_METAL_NINTM_PREFILL_NAX",
            "0",
            1);
        setenv(
            "MFQ_METAL_NINTM_SMALLM_NAX",
            path == Path::adaptive ? "auto" : "0",
            1);
        if (path == Path::omlx) {
            return omlx(stage);
        }
        if (path == Path::nax) {
            return nax(stage);
        }
        if (path == Path::adaptive) {
            return adaptive(stage);
        }
        if (path == Path::row_replay) {
            return row_replay(stage);
        }
        return batched(stage, input_, down_input_, ids_, route_weights_);
    }

    std::size_t logical_bytes(Stage stage) const {
        std::size_t bytes = 0;
        if (stage == Stage::gate_up || stage == Stage::full) {
            bytes += kGateUpBytesPerExpert;
        }
        if (stage == Stage::down || stage == Stage::full) {
            bytes += kDownBytesPerExpert;
        }
        return bytes * static_cast<std::size_t>(options_.rows) * kRoutes;
    }

private:
    MlxDeepseekV4SsdExpertWeights make_routed_weights() {
        for (int slot = 0; slot < kSlots; ++slot) {
            fill_slot(arena_, static_cast<std::size_t>(slot));
        }
        std::vector<std::int32_t> slot_for_expert(kExperts, 0);
        std::vector<std::int32_t> active(kSlots);
        std::iota(active.begin(), active.end(), 0);
        for (int slot = 0; slot < kSlots; ++slot) {
            slot_for_expert[static_cast<std::size_t>(slot)] = slot;
        }
        return arena_.routed_weights(slot_for_expert, active);
    }

    array batched(
        Stage stage,
        const array& input,
        const array& down_input,
        const array& ids,
        const array& route_weights) const {
        if (stage == Stage::gate_up) {
            return routed_.gate_up.swiglu(input, ids, 0.0f);
        }
        if (stage == Stage::down) {
            return mfq::metal::moe_weighted_reduce(
                routed_.down.forward(down_input, ids),
                route_weights);
        }
        auto hidden = routed_.gate_up.swiglu(input, ids, 0.0f);
        return mfq::metal::moe_weighted_reduce(
            routed_.down.forward(hidden, ids),
            route_weights);
    }

    array row_replay(Stage stage) const {
        std::vector<array> outputs;
        outputs.reserve(static_cast<std::size_t>(options_.rows));
        for (int row = 0; row < options_.rows; ++row) {
            outputs.push_back(batched(
                stage,
                row_slice(input_, row),
                row_slice(down_input_, row),
                row_slice(ids_, row),
                row_slice(route_weights_, row)));
        }
        return mlx::core::concatenate(std::move(outputs), 0);
    }

    array nax(Stage stage) const {
        const int route_count = options_.rows * kRoutes;
        auto route_order = mlx::core::contiguous(
            mlx::core::astype(
                mlx::core::argsort(
                    mlx::core::reshape(ids_, Shape{route_count})),
                mlx::core::int32));
        auto restore = [&](array sorted, int width) {
            return mlx::core::reshape(
                mlx::core::take(
                    std::move(sorted),
                    mlx::core::argsort(route_order),
                    0),
                Shape{options_.rows, kRoutes, width});
        };
        if (stage == Stage::gate_up) {
            return restore(
                routed_.gate_up.swiglu_sorted(
                    input_, ids_, route_order, 0.0f, true),
                kRouted);
        }
        auto sorted_down_input = mlx::core::take(
            mlx::core::reshape(
                down_input_,
                Shape{route_count, kRouted}),
            route_order,
            0);
        if (stage == Stage::down) {
            return mfq::metal::moe_weighted_reduce(
                restore(
                    routed_.down.forward_sorted(
                        sorted_down_input,
                        ids_,
                        route_order,
                        true,
                        nullptr,
                        true),
                    kHidden),
                route_weights_);
        }
        auto hidden = routed_.gate_up.swiglu_sorted(
            input_, ids_, route_order, 0.0f, true);
        return mfq::metal::moe_weighted_reduce(
            restore(
                routed_.down.forward_sorted(
                    hidden,
                    ids_,
                    route_order,
                    true,
                    nullptr,
                    true),
                kHidden),
            route_weights_);
    }

    array adaptive(Stage stage) const {
        if (
            !routed_.gate_up.prefers_mxfp4_smallm_nax(ids_)
            || !routed_.down.prefers_mxfp4_smallm_nax(ids_)
        ) {
            return batched(
                stage,
                input_,
                down_input_,
                ids_,
                route_weights_);
        }
        return nax(stage);
    }

    array omlx_qmm(
        const array& input,
        const array& weight,
        const array& scale) const {
        return mlx::core::gather_qmm(
            input,
            weight,
            scale,
            std::nullopt,
            std::nullopt,
            ids_,
            true,
            32,
            4,
            "mxfp4",
            false);
    }

    array omlx(Stage stage) const {
        if (stage == Stage::down) {
            auto projected = omlx_qmm(
                mlx::core::expand_dims(down_input_, 2),
                omlx_down_weight_,
                omlx_down_scale_);
            return mfq::metal::moe_weighted_reduce(
                mlx::core::squeeze(std::move(projected), 2),
                route_weights_);
        }
        auto source = mlx::core::reshape(
            input_,
            Shape{options_.rows, 1, 1, kHidden});
        auto up = omlx_qmm(
            source,
            omlx_up_weight_,
            omlx_up_scale_);
        auto gate = omlx_qmm(
            source,
            omlx_gate_weight_,
            omlx_gate_scale_);
        auto hidden = up * gate * mlx::core::sigmoid(gate);
        if (stage == Stage::gate_up) {
            return mlx::core::squeeze(std::move(hidden), 2);
        }
        auto projected = omlx_qmm(
            hidden,
            omlx_down_weight_,
            omlx_down_scale_);
        return mfq::metal::moe_weighted_reduce(
            mlx::core::squeeze(std::move(projected), 2),
            route_weights_);
    }

    const Options& options_;
    MlxDeepseekV4SsdExpertArena arena_;
    array input_;
    array down_input_;
    array ids_;
    array route_weights_;
    MlxDeepseekV4SsdExpertWeights routed_;
    array omlx_up_weight_;
    array omlx_gate_weight_;
    array omlx_down_weight_;
    array omlx_up_scale_;
    array omlx_gate_scale_;
    array omlx_down_scale_;
};

double elapsed_ms(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - begin).count();
}

double measure(
    const Benchmark& benchmark,
    Path path,
    Stage stage,
    int repetitions,
    int eval_batch) {
    const auto begin = Clock::now();
    for (int index = 0; index < repetitions; index += eval_batch) {
        const int count = std::min(eval_batch, repetitions - index);
        std::vector<array> pending;
        pending.reserve(static_cast<std::size_t>(count));
        for (int item = 0; item < count; ++item) {
            pending.push_back(benchmark.operation(path, stage));
        }
        mlx::core::eval(std::move(pending));
    }
    mlx::core::synchronize();
    return elapsed_ms(begin) / repetitions;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    if ((values.size() & 1u) != 0u) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
}

struct Delta {
    std::size_t elements = 0;
    std::size_t different = 0;
    double mean_absolute = 0.0;
    float maximum_absolute = 0.0f;
};

Delta compare(array reference, array candidate) {
    reference = mlx::core::contiguous(
        mlx::core::astype(std::move(reference), mlx::core::float32));
    candidate = mlx::core::contiguous(
        mlx::core::astype(std::move(candidate), mlx::core::float32));
    mlx::core::eval(reference, candidate);
    require(reference.shape() == candidate.shape(), "output shape differs");
    Delta result{.elements = reference.size()};
    double total_absolute = 0.0;
    const auto* expected = reference.data<float>();
    const auto* actual = candidate.data<float>();
    for (std::size_t index = 0; index < reference.size(); ++index) {
        require(
            std::isfinite(expected[index]) && std::isfinite(actual[index]),
            "benchmark output contains a non-finite value");
        const float absolute = std::fabs(expected[index] - actual[index]);
        result.different += absolute != 0.0f;
        result.maximum_absolute = std::max(result.maximum_absolute, absolute);
        total_absolute += absolute;
    }
    result.mean_absolute = total_absolute / result.elements;
    return result;
}

void report_delta(
    const char* candidate,
    const Delta& delta) {
    std::cout
        << "DELTA\tcandidate=" << candidate
        << "\telements=" << delta.elements
        << "\tdifferent=" << delta.different
        << "\tmean_abs=" << std::setprecision(9) << delta.mean_absolute
        << "\tmax_abs=" << delta.maximum_absolute << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        std::cout
            << "CONFIG\trows=" << options.rows
            << "\troutes=" << kRoutes
            << "\texperts=" << kExperts
            << "\thidden=" << kHidden
            << "\trouted=" << kRouted
            << "\tpattern=" << options.pattern
            << "\trepetitions=" << options.repetitions
            << "\ttrials=" << options.trials
            << "\teval_batch=" << options.eval_batch << '\n';

        Benchmark benchmark(options);
        for (const auto path : {
                 Path::sorted,
                 Path::unsorted,
                 Path::omlx,
                 Path::nax,
                 Path::adaptive,
                 Path::row_replay,
             }) {
            for (const auto stage : {
                     Stage::gate_up,
                     Stage::down,
                     Stage::full,
                 }) {
                for (int warmup = 0; warmup < 3; ++warmup) {
                    auto output = benchmark.operation(path, stage);
                    mlx::core::eval(output);
                }
            }
        }
        mlx::core::synchronize();

        report_delta(
            "unsorted",
            compare(
                benchmark.operation(Path::sorted, Stage::full),
                benchmark.operation(Path::unsorted, Stage::full)));
        report_delta(
            "omlx",
            compare(
                benchmark.operation(Path::sorted, Stage::full),
                benchmark.operation(Path::omlx, Stage::full)));
        report_delta(
            "nax",
            compare(
                benchmark.operation(Path::sorted, Stage::full),
                benchmark.operation(Path::nax, Stage::full)));
        report_delta(
            "adaptive",
            compare(
                benchmark.operation(Path::sorted, Stage::full),
                benchmark.operation(Path::adaptive, Stage::full)));
        report_delta(
            "row_replay",
            compare(
                benchmark.operation(Path::sorted, Stage::full),
                benchmark.operation(Path::row_replay, Stage::full)));

        constexpr std::array paths{
            Path::sorted,
            Path::unsorted,
            Path::omlx,
            Path::nax,
            Path::adaptive,
            Path::row_replay,
        };
        for (const auto stage : {
                 Stage::gate_up,
                 Stage::down,
                 Stage::full,
             }) {
            std::array<std::vector<double>, paths.size()> samples;
            for (int trial = 0; trial < options.trials; ++trial) {
                for (std::size_t offset = 0; offset < paths.size(); ++offset) {
                    const auto path_index =
                        (static_cast<std::size_t>(trial) + offset) %
                        paths.size();
                    samples[path_index].push_back(measure(
                        benchmark,
                        paths[path_index],
                        stage,
                        options.repetitions,
                        options.eval_batch));
                }
            }
            for (std::size_t index = 0; index < paths.size(); ++index) {
                const double middle = median(samples[index]);
                const double logical_gbs =
                    static_cast<double>(benchmark.logical_bytes(stage)) /
                    (middle * 1.0e6);
                std::cout
                    << "RESULT\tstage=" << stage_name(stage)
                    << "\tpath=" << path_name(paths[index])
                    << "\tmedian_ms=" << std::fixed << std::setprecision(4)
                    << middle
                    << "\tlogical_GB/s=" << std::setprecision(1)
                    << logical_gbs
                    << "\tsamples_ms=";
                for (std::size_t sample = 0;
                     sample < samples[index].size();
                     ++sample) {
                    if (sample != 0) {
                        std::cout << ',';
                    }
                    std::cout << std::setprecision(4)
                              << samples[index][sample];
                }
                std::cout << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "DeepSeek-V4 small-M MoE benchmark failed: "
            << error.what() << '\n';
        return 1;
    }
}
