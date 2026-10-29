#include "mlx_tq.h"
#include "mlx_tensor.h"
#include "mfq_container.h"
#include "mlx_deepseek_v41_moe.h"
#include "mlx_moe.h"
#include "mlx_moe_ops.h"
#include "mlx_tq_rows.h"
#include "mlx_quantized_row_store.h"
#include "mlx_tq_expert_cache.h"
#include "mlx_resource_telemetry.h"
#include "mlx_kernel_prepare.h"
#include "mlx_eval_timing.h"
#include "mlx_platform.h"
#include "mfq_tq_embedded.h"
#include "qwen4_ops.h"

#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <memory>
#include <numeric>
#include <functional>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <array>
#include <bit>
#include <source_location>
#include <tuple>
#include <stdexcept>

using namespace mlx::core;
using namespace mfq::metal;

static void check(const array& actual, const array& expected, float tolerance,
        const std::source_location location = std::source_location::current()) {
    auto a = astype(actual, float32), b = astype(expected, float32);
    float error = max(abs(a - b)).item<float>();
    if (!std::isfinite(error) || error > tolerance)
        throw std::runtime_error("TQ numerical mismatch at line " + std::to_string(location.line())
            + ": " + std::to_string(error) + " > " + std::to_string(tolerance));
}

static bool column_projection_roundoff(const array& actual, const array& expected) {
    if (actual.dtype() != float16) return false;
    const auto a = astype(actual, float32), b = astype(expected, float32);
    const auto difference = a - b;
    const float rms = sqrt(sum(square(difference)) / maximum(sum(square(b)), array(1e-20f))).item<float>();
    const float error = max(abs(difference)).item<float>();
    const float scale = max(abs(b)).item<float>();
    return std::isfinite(rms) && std::isfinite(error) && std::isfinite(scale)
        && rms <= .0015f && error <= .003f * scale + .000002f;
}

static std::vector<std::uint8_t> rotate_rows(const std::vector<std::uint8_t>& blob) {
    auto output = blob;
    std::uint32_t rows, width, columns, gs;
    std::memcpy(&rows, blob.data() + 12, 4); std::memcpy(&width, blob.data() + 16, 4);
    std::memcpy(&columns, blob.data() + 20, 4); std::memcpy(&gs, blob.data() + 24, 4);
    const auto groups = columns ? 0 : (width + gs - 1) / gs;
    const std::size_t scales = columns ? width * 2 : (rows * groups + 1) / 2;
    const auto packed_start = 64 + rows * 8 + scales;
    std::uint32_t offset = 0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto source = (r + 7) % rows;
        std::uint64_t desc;
        std::memcpy(&desc, blob.data() + 64 + source * 8, 8);
        const auto words = width * (((desc >> 32) & 7) + 1) / 32;
        std::memcpy(output.data() + packed_start + offset * 4,
            blob.data() + packed_start + std::uint32_t(desc) * 4, words * 4);
        desc = (desc & 0xffffffff00000000ull) | offset;
        std::memcpy(output.data() + 64 + r * 8, &desc, 8);
        if (!columns) for (std::uint32_t g = 0; g < groups; ++g) {
            const auto src = source * groups + g, dst = r * groups + g;
            const auto code = (blob[64 + rows * 8 + src / 2] >> ((src & 1) * 4)) & 15;
            auto& target = output[64 + rows * 8 + dst / 2];
            target = (target & ~(15 << ((dst & 1) * 4))) | (code << ((dst & 1) * 4));
        }
        offset += words;
    }
    return output;
}

template <typename Weight>
static void chained(const Weight& weight, const array& input, const std::vector<array>& schedule,
        int width, const std::string& name, const char* implementation, int repeats = 3, int first_repeat = 0) {
    auto gain = array(.001f, input.dtype());
    gain.eval();
    for (int repeat = first_repeat; repeat < first_repeat + repeats; ++repeat) {
        const auto begin = std::chrono::steady_clock::now();
        auto source = input;
        for (const auto& ids : schedule) {
            auto output = weight.routed_matmul(source, ids);
            const bool shared = input.ndim() == 2 && input.shape(0) == ids.shape(0);
            auto feedback = shared
                ? reshape(slice(reshape(output, {-1}), {0}, {int(input.size())}), input.shape())
                : reshape(slice(output, {0, 0, 0}, {ids.shape(0), ids.shape(1), width}), input.shape());
            source = add(input, multiply(feedback, gain));
        }
        const auto built = std::chrono::steady_clock::now();
        source.eval();
        const auto end = std::chrono::steady_clock::now();
        std::cout << nlohmann::json{{"name", name}, {"implementation", implementation},
            {"chained", true}, {"repeat", repeat},
            {"build_ms", std::chrono::duration<double, std::milli>(built-begin).count()/schedule.size()},
            {"eval_ms", std::chrono::duration<double, std::milli>(end-built).count()/schedule.size()},
            {"boundary", "dependent projections plus identical input-feedback operations; one final wait"}} << std::endl;
    }
}

struct RoutedBenchmarkCall {
    std::function<array(const array&, const array&)> call;
    array routed_matmul(const array& x, const array& ids) const { return call(x, ids); }
};

static std::vector<std::uint8_t> random_blob(int rows, int logical_width, std::uint32_t seed,
        int group_size = 512, bool columns = false) {
    const int width = (logical_width + 255) / 256 * 256, groups = (width + group_size - 1) / group_size;
    std::mt19937 rng(seed);
    std::vector<std::uint64_t> descriptors(rows);
    std::uint32_t words = 0;
    for (auto& descriptor : descriptors) {
        const std::uint64_t bits = 1 + rng() % 8;
        descriptor = words | ((bits - 1) << 32)
            | (std::uint64_t(std::bit_cast<std::uint16_t>(float16_t(.0005f))) << 40);
        words += width * bits / 32;
    }
    const auto scale_bytes = columns ? std::size_t(width) * 2 : (std::size_t(rows) * groups + 1) / 2;
    const std::size_t packed_begin = 64 + rows * 8 + scale_bytes;
    std::vector<std::uint8_t> blob(packed_begin + words * 4, 0);
    std::memcpy(blob.data(), "RTRELL05", 8);
    const std::uint32_t header[]{5, std::uint32_t(rows), std::uint32_t(width), std::uint32_t(columns),
        columns ? 0u : std::uint32_t(group_size), words * 4};
    std::memcpy(blob.data() + 8, header, sizeof(header));
    std::memcpy(blob.data() + 64, descriptors.data(), rows * 8);
    for (auto offset = std::size_t(64 + rows * 8); offset < blob.size(); ++offset)
        blob[offset] = std::uint8_t(rng());
    if (columns) for (int col = 0; col < width; ++col) {
        const auto value = float16_t(.5f + .01f * (col % 31));
        std::memcpy(blob.data() + 64 + rows * 8 + col * 2, &value, 2);
    }
    return blob;
}

static MlxTqWeight random_weight(int rows, int logical_width, std::uint32_t seed) {
    return MlxTqWeight::from_blob(random_blob(rows, logical_width, seed), logical_width);
}

static void check_expansion_vectors() {
    for (int width : {513, 2304, 4097}) for (int layout : {0, 1, 2}) {
        const int outputs = width * 2 + 3;
        const auto op = MlxTqWeight::from_blob(random_blob(outputs, width, 3941, layout == 0 ? 16 : 512, layout == 2), width);
        const auto dense = astype(op.embedding(arange(outputs, int32)), float32);
        for (auto dtype : {float16, bfloat16, float32}) for (int tokens : {1, 2, 6}) {
            const auto input = reshape(astype(sin(arange(tokens * width, float32)) * .1f, dtype), {tokens, width});
            const auto expected = matmul(astype(input, float32), transpose(dense));
            const auto routed = reshape(op.routed_matmul(input, zeros({tokens, 1}, int32)), {tokens, outputs});
            const auto source = contiguous(astype(input, float32));
            eval({source, dense});
            int path = 0;
            for (const auto& actual : {op.matmul(input), routed}) {
                const auto values = astype(actual, float32);
                const auto error = reshape(abs(values - expected), {-1});
                const auto worst = argmax(error).item<std::uint32_t>();
                const int token = worst / outputs, row = worst % outputs;
                double oracle = 0.;
                for (int col = 0; col < width; ++col)
                    oracle += double(source.data<float>()[token * width + col]) * dense.data<float>()[row * width + col];
                const double tolerance = (dtype == bfloat16 ? .008 : dtype == float16 ? .003 : .00001) * std::abs(oracle) + .000001;
                if (!allclose(actual, expected, dtype == bfloat16 ? .008 : .003, .00001).item<bool>()
                    || std::abs(double(values.data<float>()[worst]) - oracle) > tolerance)
                    throw std::runtime_error("TQ expanding vector projection differs from decoded reference: K="
                        + std::to_string(width) + ", M=" + std::to_string(tokens) + ", layout=" + std::to_string(layout)
                        + ", dtype=" + (dtype == float16 ? "f16" : dtype == bfloat16 ? "bf16" : "f32") + ", path=" + std::to_string(path)
                        + ", error=" + std::to_string(max(error).item<float>())
                        + ", oracle_error=" + std::to_string(std::abs(double(values.data<float>()[worst]) - oracle)));
                ++path;
            }
        }
    }
    std::cout << "TQ expanding vector projection checks passed\n";
}

static void check_prefill_reduction() {
    constexpr int experts = 7, outputs = 257, width = 640, routes = 4;
    for (bool columns : {false, true}) {
        const auto weight = MlxTqWeight::from_blob(random_blob(experts * outputs, width, 3821, 16, columns), width, experts);
        for (auto dtype : {float16, bfloat16, float32}) for (int tokens : {1, 6, 7, 16, 33, 257}) {
            auto ids = reshape(remainder(arange(tokens * routes, int32), array(experts)), {tokens, routes});
            auto scales = full({tokens, routes}, 1.f / routes, float32);
            auto input = reshape(astype(sin(arange(tokens * routes * width, float32)) * .1f, dtype), {tokens, routes, width});
            auto pairs = weight.routed_matmul(input, ids);
            if (dtype == bfloat16) pairs = astype(pairs, float32);
            check(weight.routed_matmul_reduce(input, ids, scales), astype(moe_weighted_reduce(pairs, scales), dtype), 0.f);
        }
    }
    std::cout << "TQ narrow-input prefill reduction checks passed\n";
}

static MlxTqWeight pooled_weight(const std::vector<std::uint8_t>& blob, int width, int experts);

static void check_projection_cache() {
    for (int width : {321, 1025, 2560, 4097}) {
        std::vector<MlxTqWeight> weights;
        std::vector<array> dense;
        for (const auto [rows, layout] : {std::pair{17, 0}, std::pair{65, 1}, std::pair{129, 2}}) {
            const auto blob = random_blob(rows, width, 3823 + rows, layout == 0 ? 16 : 512, layout == 2);
            weights.push_back(layout == 1 ? pooled_weight(blob, width, 1) : MlxTqWeight::from_blob(blob, width));
            dense.push_back(astype(weights.back().embedding(arange(rows, int32)), float32));
        }
        for (int tokens : {7, 8, 12, 16, 17, 32}) {
            const auto input = reshape(astype(sin(arange(tokens * width, float32)) * .1f, float16), {tokens, width});
            const auto result = MlxTqWeight::matmul_many(weights, input);
            for (std::size_t i = 0; i < weights.size(); ++i)
                if (!allclose(result[i], matmul(astype(input, float32), transpose(dense[i])), .003, .000001).item<bool>())
                    throw std::runtime_error("TQ grouped projection cache differs from decoded reference");
        }
    }
    std::cout << "TQ mixed grouped projection cache checks passed\n";
}

static std::vector<std::uint8_t> expert_blob(const std::vector<std::vector<std::uint8_t>>& experts,
        int rows, int width) {
    std::size_t size = 64 + experts.size() * 16;
    for (const auto& expert : experts) size += expert.size();
    std::vector<std::uint8_t> blob(size, 0);
    std::memcpy(blob.data(), "RTQEXP01", 8);
    const std::uint32_t header[]{1, std::uint32_t(experts.size()), std::uint32_t(rows), std::uint32_t(width), 16, 0};
    std::memcpy(blob.data() + 8, header, sizeof(header));
    std::uint64_t offset = 64 + experts.size() * 16;
    for (std::size_t index = 0; index < experts.size(); ++index) {
        const std::uint64_t bytes = experts[index].size();
        std::memcpy(blob.data() + 64 + index * 16, &offset, 8);
        std::memcpy(blob.data() + 72 + index * 16, &bytes, 8);
        std::memcpy(blob.data() + offset, experts[index].data(), bytes);
        offset += bytes;
    }
    return blob;
}

static MlxTqWeight pooled_weight(const std::vector<std::uint8_t>& blob, int width, int experts);

static void check_column_inputs() {
    for (const auto geometry : {std::array{4097, 1, 8193}, std::array{4097, 4, 2049}, std::array{513, 64, 1025}}) {
        const int width = geometry[0], experts = geometry[1], outputs = geometry[2];
        const int stored = (width + 255) / 256 * 256;
        std::vector<std::vector<std::uint8_t>> children, up_children;
        for (int expert = 0; expert < experts; ++expert) {
            auto blob = random_blob(outputs, width, 3981 + expert, 512, true);
            auto up_blob = random_blob(outputs, width, 4081 + expert, 512, true);
            for (auto* image : {&blob, &up_blob}) for (int row = 0; row < outputs; ++row) {
                std::uint64_t descriptor;
                std::memcpy(&descriptor, image->data() + 64 + row * 8, 8);
                descriptor = (descriptor & ~(std::uint64_t{0xffff} << 40))
                    | (std::uint64_t(std::bit_cast<std::uint16_t>(float16_t(.01f))) << 40);
                std::memcpy(image->data() + 64 + row * 8, &descriptor, 8);
            }
            for (int col = 0; col < stored; ++col) {
                const auto scale = float16_t(.25f + expert * .125f + (col % 17) * .03125f);
                const auto up_scale = float16_t(.375f + expert * .0625f + (col % 13) * .015625f);
                std::memcpy(blob.data() + 64 + outputs * 8 + col * 2, &scale, 2);
                std::memcpy(up_blob.data() + 64 + outputs * 8 + col * 2, &up_scale, 2);
            }
            children.push_back(std::move(blob));
            up_children.push_back(std::move(up_blob));
        }
        const auto blob = experts == 1 ? children[0] : expert_blob(children, outputs, width);
        const auto up_blob = experts == 1 ? up_children[0] : expert_blob(up_children, outputs, width);
        auto weight = MlxTqWeight::from_blob(blob, width, experts);
        auto up = MlxTqWeight::from_blob(up_blob, width, experts);
        const auto pool = pooled_weight(blob, width, experts), up_pool = pooled_weight(up_blob, width, experts);
        auto dense = astype(weight.embedding(arange(experts * outputs, int32)), float32);
        auto up_dense = astype(up.embedding(arange(experts * outputs, int32)), float32);
        const auto counts = experts == 64 ? std::vector{64, 128} : std::vector{1, 2, 4, 5, 6, 7, 8, 12, 16};
        for (bool shared : {false, true}) for (int tokens : counts) {
            const int routes = experts == 64 ? 1 : experts;
            auto ids = broadcast_to(reshape(arange(routes, int32), {1, routes}), {tokens, routes});
            std::optional<array> reuse;
            if (experts > 1 && tokens >= 2 && tokens <= 16) {
                std::vector<int> selected(tokens * routes);
                for (int row = 0; row < int(selected.size()); ++row) selected[row] = row % routes;
                reuse = make_tq_reuse_plan(selected, tokens, routes, experts);
                if (!reuse) throw std::runtime_error("TQ column reuse fixture has no repeated routes");
            }
            auto input = reshape(astype(sin(arange(tokens * (shared ? 1 : routes) * width, float32)) * .1f, float16),
                shared ? Shape{tokens, width} : Shape{tokens, routes, width});
            eval({input, ids, dense, up_dense});
            if (experts == 1 && !shared && tokens == 1) {
                MlxKernelPreparation preparation(32, 32);
                preparation.collect([&] { preparation.add(weight.routed_matmul(input, ids)); });
                if (preparation.size() != 2) throw std::runtime_error("TQ column input kernels were not selected");
                preparation.finish();
                preparation.collect([&] { preparation.add(weight.swiglu(up, input, ids)); });
                if (preparation.size() != 2) throw std::runtime_error("TQ Gate/Up input preparation was not fused");
                preparation.finish();
            }
            std::vector<array> expected, expected_up;
            for (int expert = 0; expert < routes; ++expert) {
                auto source = shared ? input : reshape(slice(input, {0, expert, 0}, {tokens, expert + 1, width}), {tokens, width});
                auto selected = slice(dense, {expert * outputs, 0}, {(expert + 1) * outputs, width});
                expected.push_back(matmul(astype(source, float32), transpose(selected)));
                auto selected_up = slice(up_dense, {expert * outputs, 0}, {(expert + 1) * outputs, width});
                expected_up.push_back(matmul(astype(source, float32), transpose(selected_up)));
            }
            auto reference = stack(expected, 1);
            auto actual = astype(weight.routed_matmul(input, ids), float32);
            check(astype(pool.routed_matmul(input, ids), float32), actual, 0.f);
            const float relative = sqrt(sum(square(actual - reference)) / maximum(sum(square(reference)), array(1e-20f))).item<float>();
            if (!std::isfinite(relative) || relative > .0015f)
                throw std::runtime_error("TQ column input factorization exceeds numerical tolerance");
            if (reuse) {
                const auto reused = astype(weight.routed_matmul(input, ids, &*reuse), float32);
                check(astype(pool.routed_matmul(input, ids, &*reuse), float32), reused, 0.f);
                const float error = sqrt(sum(square(reused - reference))
                    / maximum(sum(square(reference)), array(1e-20f))).item<float>();
                if (!std::isfinite(error) || error > .0015f)
                    throw std::runtime_error("TQ compact column reuse exceeds numerical tolerance");
            }
            if (experts == 1 && tokens == 1)
                check(reshape(weight.matmul(input), actual.shape()), actual, 0.f);
            for (bool gelu : {false, true}) for (float limit : {0.f, .05f}) {
                auto g = astype(astype(reference, float16), float32);
                auto u = astype(astype(stack(expected_up, 1), float16), float32);
                if (limit > 0.f) { g = minimum(g, array(limit)); u = clip(u, array(-limit), array(limit)); }
                auto activated = gelu ? array(.5f) * g * (array(1.f) + tanh(array(.7978845608028654f)
                    * g * (array(1.f) + array(.044715f) * g * g))) : g * sigmoid(g);
                auto expected_pair = astype(astype(activated * u, float16), float32);
                auto actual_pair = astype(weight.swiglu(up, input, ids, limit, gelu), float32);
                check(astype(pool.swiglu(up_pool, input, ids, limit, gelu), float32), actual_pair, 0.f);
                const float pair_error = sqrt(sum(square(actual_pair - expected_pair))
                    / maximum(sum(square(expected_pair)), array(1e-20f))).item<float>();
                if (!std::isfinite(pair_error) || pair_error > .0025f)
                    throw std::runtime_error("TQ column Gate/Up factorization exceeds numerical tolerance");
                if (reuse) {
                    const auto reused = astype(weight.swiglu(up, input, ids, limit, gelu, &*reuse), float32);
                    check(astype(pool.swiglu(up_pool, input, ids, limit, gelu, &*reuse), float32), reused, 0.f);
                    const float error = sqrt(sum(square(reused - expected_pair))
                        / maximum(sum(square(expected_pair)), array(1e-20f))).item<float>();
                    if (!std::isfinite(error) || error > .0025f)
                        throw std::runtime_error("TQ compact column Gate/Up reuse exceeds numerical tolerance");
                }
            }
        }
        const auto invalid = array({-1, experts, experts, -1}, {2, 2}, int32);
        const auto input = ones({2, width}, float16);
        check(weight.routed_matmul(input, invalid), zeros({2, 2, outputs}, float16), 0.f);
        check(pool.swiglu(up_pool, input, invalid), zeros({2, 2, outputs}, float16), 0.f);
    }
    std::cout << "TQ column input factorization checks passed\n";
}

static MlxTqWeight pooled_weight(const std::vector<std::uint8_t>& blob, int width, int experts) {
    const MlxMappedTqRows table(blob, width);
    const int rows = table.rows() / experts;
    std::size_t bytes = 0;
    for (int expert = 0; expert < experts; ++expert) bytes += table.packed_rows(expert * rows, rows).bytes;
    std::vector<std::uint32_t> words(bytes / 4 + experts * 17 + 64, 0);
    std::vector<std::uint64_t> offsets(experts);
    std::size_t cursor = 31;
    for (int expert = experts - 1; expert >= 0; --expert) {
        const auto range = table.packed_rows(expert * rows, rows);
        offsets[expert] = cursor;
        std::memcpy(words.data() + cursor, blob.data() + range.offset, range.bytes);
        cursor += range.bytes / 4 + 17;
    }
    return MlxTqWeight::from_pool(blob, array(words.begin(), {int(words.size())}, uint32), offsets, width);
}

static void check_expert_containers() {
    constexpr int experts = 3;
    {
        std::vector<std::vector<std::uint8_t>> children;
        for (int e = 0; e < experts; ++e) children.push_back(random_blob(17, 513, 3001 + e, e == 0 ? 16 : 512, e == 2));
        const auto blob = expert_blob(children, 17, 513);
        const MlxMappedTqRows table(blob);
        MlxTqRowBatch selected;
        std::vector<array> expected;
        for (int row : {0, 16, 17, 50, 0, 33}) {
            table.append_row(row, selected);
            expected.push_back(MlxTqWeight::from_blob(children[row / 17], 513).embedding(array({row % 17}, int32)));
        }
        check(selected.decode(), concatenate(expected, 0), 0.f);
        bool rejected = false;
        try { table.packed_rows(16, 2); } catch (const std::out_of_range&) { rejected = true; }
        if (!rejected) throw std::runtime_error("TQ expert packed span crosses records");
        for (int offset : {8, 12, 16, 20, 24, 28, 32, 64, 72}) {
            auto malformed = blob;
            malformed[offset] ^= 127;
            rejected = false;
            try { MlxMappedTqRows invalid(malformed, 513); } catch (const std::runtime_error&) { rejected = true; }
            if (!rejected) throw std::runtime_error("TQ expert container accepts malformed metadata");
        }
    }
    for (const int width : {513, 1025}) for (const bool columns : {false, true}) {
        const int outputs = width == 513 ? 17 : 2065;
        std::vector<std::vector<std::uint8_t>> children, ups;
        std::vector<array> values;
        for (int e = 0; e < experts; ++e) {
            children.push_back(random_blob(outputs, width, 3011 + e, e == 0 ? 16 : 512, columns));
            ups.push_back(random_blob(outputs, width, 3021 + e, 512, columns));
            if (columns) for (auto* child : {&children.back(), &ups.back()}) {
                const int stored = (width + 255) / 256 * 256;
                for (int col = 0; col < stored; ++col) {
                    const float16_t scale(.25f + e * .5f + .01f * (col % 31));
                    std::memcpy(child->data() + 64 + outputs * 8 + col * 2, &scale, 2);
                }
            }
            values.push_back(MlxTqWeight::from_blob(children.back(), width).embedding(arange(outputs, int32), float32));
        }
        const auto blob = expert_blob(children, outputs, width), up_blob = expert_blob(ups, outputs, width);
        const auto weight = MlxTqWeight::from_blob(blob), pool = pooled_weight(blob, width, experts);
        const auto up = MlxTqWeight::from_blob(up_blob), up_pool = pooled_weight(up_blob, width, experts);
        const auto dense = transpose(reshape(concatenate(values, 0), {experts, outputs, width}), {0, 2, 1});
        check(weight.embedding(arange(experts * outputs, int32), float32), concatenate(values, 0), 0.f);
        check(pool.embedding(arange(experts * outputs, int32), float32), concatenate(values, 0), 0.f);
        for (int e = 0; e < experts; ++e)
            check(weight.expert_view(e).embedding(arange(outputs, int32), float32), values[e], 0.f);
        for (const auto dtype : {float16, bfloat16, float32}) for (int tokens : {1, 2, 3, 4, 5, 6, 7, 12, 16, 17, 257}) {
            const auto input = astype(reshape(sin(arange(tokens * width, float32)), {tokens, width}), dtype);
            std::vector<int> ids_host(tokens * 2);
            for (int row = 0; row < int(ids_host.size()); ++row) ids_host[row] = row % 2 * 2;
            const auto ids = array(ids_host.begin(), {tokens, 2}, int32);
            const auto expected = astype(reshape(gather_mm(reshape(astype(input, float32), {tokens, 1, width}),
                dense, floor_divide(arange(tokens * 2, int32), array(2)), reshape(ids, {-1})),
                {tokens, 2, outputs}), dtype);
            const auto actual = weight.routed_matmul(input, ids);
            if (!allclose(astype(actual, float32), astype(expected, float32), dtype == bfloat16 ? .01 : .005, .000002).item<bool>()
                && !(columns && column_projection_roundoff(actual, expected))) {
                const auto difference = astype(actual, float32) - astype(expected, float32);
                const float rms = sqrt(sum(square(difference)) / maximum(sum(square(astype(expected, float32))), array(1e-20f))).item<float>();
                throw std::runtime_error("TQ expert container projection differs from decoded GEMM: width="
                    + std::to_string(width) + " columns=" + std::to_string(columns) + " tokens=" + std::to_string(tokens)
                    + " dtype=" + (dtype == float16 ? "F16" : dtype == bfloat16 ? "BF16" : "F32")
                    + " max_abs=" + std::to_string(max(abs(difference)).item<float>()) + " relative_rms=" + std::to_string(rms));
            }
            check(pool.routed_matmul(input, ids), actual, 0.f);
            const auto hidden = weight.swiglu(up, input, ids);
            check(pool.swiglu(up_pool, input, ids), hidden, 0.f);
            const auto factors = full({tokens, 2}, .5f, float32);
            check(pool.routed_matmul_reduce(input, ids, factors), weight.routed_matmul_reduce(input, ids, factors), 0.f);
            if (const auto reuse = make_tq_reuse_plan(ids_host, tokens, 2, experts)) {
                const auto reused = weight.routed_matmul(input, ids, &*reuse);
                const auto paired = pool.swiglu(up_pool, input, ids, 0.f, false, &*reuse);
                const float tolerance = dtype == bfloat16 ? .008f : dtype == float16 ? .003f : .00001f;
                if (!allclose(astype(reused, float32), astype(expected, float32), tolerance, .000002).item<bool>()
                    || !allclose(astype(paired, float32), astype(hidden, float32), tolerance, .000002).item<bool>())
                    throw std::runtime_error("TQ expert reuse differs from projection reference");
            }
            for (int e = 0; e < experts; ++e) {
                const auto view = pool.expert_view(e), single = MlxTqWeight::from_blob(children[e], width);
                check(view.matmul(input), single.matmul(input), 0.f);
            }
        }
    }
}

static void check_gated_residual() {
    auto down = random_weight(320, 10240, 1), up = random_weight(10240, 320, 2);
    auto injection = random_weight(4, 10240, 3);
    auto dense_down = astype(down.embedding(arange(320, int32)), float32);
    auto dense_up = astype(up.embedding(arange(10240, int32)), float32);
    auto dense_injection = astype(injection.embedding(arange(4, int32)), float32);
    const auto norm = zeros({10240}, float32);
    for (int tokens : {1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 17, 32, 64, 65}) {
        auto input = reshape(sin(arange(tokens * 10240, float32)), {1, tokens, 10240});
        auto actual = qwen4_gated_residual_pre(input, norm, MlxLinear(down), MlxLinear(up),
            MlxLinear(injection), 2560, 4, 1e-6f);
        auto expected = qwen4_gated_residual_pre(input, norm, dense_down, dense_up,
            dense_injection, 2560, 4, 1e-6f);
        check(actual.branch, expected.branch, .001f);
        check(*actual.injection, *expected.injection, .001f);
        auto after = qwen4_gated_residual_pre_after(actual.branch, actual.residual, *actual.injection,
            norm, MlxLinear(down), MlxLinear(up), MlxLinear(injection), 2560, 4, 1e-6f);
        auto after_ref = qwen4_gated_residual_pre_after(actual.branch, actual.residual, *actual.injection,
            norm, dense_down, dense_up, dense_injection, 2560, 4, 1e-6f);
        check(after.branch, after_ref.branch, .001f);
        check(after.residual, after_ref.residual, 0.f);
        check(*after.injection, *after_ref.injection, .001f);
        for (auto dtype : {float16, bfloat16}) {
            auto typed = astype(input, dtype);
            auto normalized = qwen4_grouped_rms_norm(typed, norm, 2560, 1e-6f);
            auto parts = MlxTqWeight::matmul_many({down, injection}, normalized);
            auto low = parts[0] / array(4.f, dtype);
            low = low * sigmoid(low);
            auto projected = up.matmul(low);
            auto branch = mean(reshape(sigmoid(projected), {1, tokens, 4, 2560})
                * reshape(normalized, {1, tokens, 4, 2560}), -2);
            auto injection_reference = array(2.f, dtype) * sigmoid(parts[1] / array(4.f, dtype));
            auto result = qwen4_gated_residual_pre(typed, norm, MlxLinear(down), MlxLinear(up),
                MlxLinear(injection), 2560, 4, 1e-6f);
            check(result.branch, branch, dtype == bfloat16 ? .008f : .001f);
            check(*result.injection, injection_reference, 0.f);
        }
    }
}

static void check_gated_mix_dispatch() {
    for (int outputs : {1028, 1029, 1040}) for (int layout : {0, 1, 2}) {
        const auto blob = random_blob(outputs, 321, 1437, layout == 0 ? 16 : 512, layout == 2);
        const auto weight = MlxTqWeight::from_blob(blob, 321), pooled = pooled_weight(blob, 321, 1);
        for (int tokens : {7, 8, 9, 12, 15, 16, 17, 32, 64, 65}) {
            const auto input = astype(reshape(sin(arange(tokens * 321, float32)), {tokens, 321}), float16);
            const auto streams = astype(reshape(cos(arange(tokens * outputs, float32)), {tokens, outputs}), float16);
            const auto gated = sigmoid(weight.matmul(input)) * streams;
            for (int groups : {1, 3, 4, 16}) if (outputs % groups == 0) {
                const auto actual = weight.gated_mix(input, streams, groups);
                check(actual, mean(reshape(gated, {tokens, groups, outputs / groups}), 1), .001f);
                check(pooled.gated_mix(input, streams, groups), actual, 0.f);
            }
        }
    }
}

static void check_shared_routes() {
    auto first = random_weight(33, 1025, 18), second = random_weight(33, 1025, 19);
    auto gate = MlxTqWeight::concatenate_experts({first, second});
    auto up = MlxTqWeight::concatenate_experts({second, first});
    std::vector<std::int32_t> routes;
    for (int token = 0; token < 257; ++token) {
        routes.push_back(0);
        routes.push_back(token % 3 ? 1 : token % 2 ? -1 : 2);
    }
    const auto ids = array(routes.data(), {257, 2});
    for (bool shared : {true, false}) {
        auto input = astype(reshape(sin(arange((shared ? 257 : 514) * 1025, float32)),
            {shared ? 257 : 514, 1025}), float16);
        check(gate.routed_matmul(input, ids),
            astype(gate.routed_matmul(astype(input, float32), ids), float16), .001f);
        check(gate.swiglu(up, input, ids),
            astype(gate.swiglu(up, astype(input, float32), ids), float16), .001f);
    }
}

static void check_wide_routes() {
    constexpr int rows = 257, width = 1281, outputs = 1025;
    auto bank = MlxTqWeight::from_blob(random_blob(3 * outputs, width, 813), width, 3);
    auto ids = broadcast_to(array({0, 1, 2, -1, 3}, {1, 5}, int32), {rows, 5});
    for (bool shared : {false, true}) {
        const int input_rows = rows * (shared ? 1 : 5);
        auto x = reshape(astype(sin(arange(input_rows * width, float32)), float16), {input_rows, width});
        check(bank.routed_matmul(x, ids),
            astype(bank.routed_matmul(astype(x, float32), ids), float16), .003f);
    }
}

static void check_small_outputs() {
    auto weight = random_weight(24, 20481, 41);
    auto reference = astype(weight.embedding(arange(24, int32)), float32);
    for (auto dtype : {float16, bfloat16, float32}) for (int tokens : {1, 2, 6, 7, 8, 12, 16, 17, 32, 128, 257, 1025}) {
        auto input = astype(reshape(sin(arange(tokens * 20481, float32)), {tokens, 20481}), dtype);
        auto expected = matmul(astype(input, float32), transpose(reference));
        check(weight.matmul(input), expected, dtype == bfloat16 ? .004f : .001f);
        const auto outputs = MlxTqWeight::matmul_many({weight, weight}, input);
        check(outputs[0], outputs[1], 0.f);
        check(outputs[0], expected, dtype == bfloat16 ? .004f : .001f);
    }
}

static void check_dense_dispatch() {
    for (const auto [width, outputs] : {std::pair{320, 1025}, std::pair{2305, 389}, std::pair{4097, 767}, std::pair{6145, 2049},
        std::pair{6145, 2561}, std::pair{8193, 257}, std::pair{10241, 321}, std::pair{16385, 1025}}) {
    for (int layout : {0, 1, 2}) {
        const auto blob = random_blob(outputs, width, 927, layout == 0 ? 16 : 512, layout == 2);
        const auto op = MlxTqWeight::from_blob(blob, width);
        const auto pooled = pooled_weight(blob, width, 1);
        const auto other = random_weight(47, width, 928);
        auto reference = astype(op.embedding(arange(outputs, int32)), float32);
        auto other_reference = astype(other.embedding(arange(47, int32)), float32);
        for (auto dtype : {float16, bfloat16, float32}) for (int tokens : {1, 2, 3, 4, 5, 6, 7, 16, 25, 32, 33, 64, 65, 128}) {
            auto input = astype(reshape(sin(arange(tokens * width, float32)), {tokens, width}), dtype);
            auto expected = matmul(astype(input, float32), transpose(reference));
            check(op.matmul(input), expected, dtype == bfloat16 ? .004f : .001f);
            if (dtype == float16 && tokens == 6 && outputs == 2561
                && !mlx_metal_nax_disabled() && mlx_apple_chip_starts_with("Apple M5")) {
                const auto expanded = concatenate({input, zeros({1, width}, dtype)}, 0);
                check(op.matmul(input), slice(op.matmul(expanded), {0, 0}, {tokens, outputs}), .001f);
            }
            const auto paired = MlxTqWeight::matmul_many({op, other}, input);
            check(paired[0], expected, dtype == bfloat16 ? .004f : .001f);
            check(paired[1], matmul(astype(input, float32), transpose(other_reference)),
                dtype == bfloat16 ? .004f : .001f);
            if (dtype == float16 && tokens <= 6 && width >= 2048) {
                check(pooled.matmul(input), op.matmul(input), 0.f);
                const auto pooled_pair = MlxTqWeight::matmul_many({pooled, other}, input);
                check(pooled_pair[0], paired[0], 0.f);
                check(pooled_pair[1], paired[1], 0.f);
            }
        }
    }
    }
}

static void check_dense_pair_dispatch() {
    for (const auto [width, outputs] : {std::pair{2049, 389}, std::pair{6145, 2049}})
    for (int layout : {0, 1, 2}) for (bool mixed : {false, true}) {
        const int other_layout = mixed ? (layout + 1) % 3 : layout;
        const auto gate_blob = random_blob(outputs, width, 1211, layout == 0 ? 16 : 512, layout == 2);
        const auto up_blob = random_blob(outputs, width, 1212, other_layout == 0 ? 16 : 512, other_layout == 2);
        const auto gate = MlxTqWeight::from_blob(gate_blob, width), up = MlxTqWeight::from_blob(up_blob, width);
        const auto pooled_gate = pooled_weight(gate_blob, width, 1), pooled_up = pooled_weight(up_blob, width, 1);
        const auto dense_gate = astype(gate.embedding(arange(outputs, int32)), float32),
            dense_up = astype(up.embedding(arange(outputs, int32)), float32);
        const auto packed = mixed || layout == 2 ? std::optional<MlxTqWeight>{}
            : std::optional(MlxTqWeight::concatenate_projections({gate, up}));
        for (auto dtype : {float16, bfloat16, float32}) for (int tokens : {2, 3, 4, 5, 6}) {
            const auto input = astype(reshape(sin(arange(tokens * width, float32)), {tokens, width}), dtype);
            const auto ids = zeros({tokens, 1}, int32);
            for (float limit : {0.f, .025f}) for (bool gelu : {false, true}) {
                auto g = astype(astype(matmul(astype(input, float32), transpose(dense_gate)), dtype), float32),
                    u = astype(astype(matmul(astype(input, float32), transpose(dense_up)), dtype), float32);
                if (limit > 0.f) { g = minimum(g, array(limit)); u = clip(u, array(-limit), array(limit)); }
                auto expected = gelu ? array(.5f) * g * (array(1.f) + tanh(array(.7978845608028654f)
                    * g * (array(1.f) + array(.044715f) * g * g))) : g * sigmoid(g);
                expected = reshape(astype(expected * u, dtype), {tokens, 1, outputs});
                const auto actual = gate.swiglu(up, input, ids, limit, gelu);
                check(actual, expected, dtype == bfloat16 ? .004f : .001f);
                check(pooled_gate.swiglu(pooled_up, input, ids, limit, gelu), actual, 0.f);
                check(pooled_gate.swiglu(up, input, ids, limit, gelu), actual, 0.f);
                check(gate.swiglu(pooled_up, input, ids, limit, gelu), actual, 0.f);
                if (packed) check(packed->swiglu_split(input, ids, limit, gelu), actual, 0.f);
                const auto mask = reshape(arange(tokens, int32) % array(3, int32) - array(1, int32), {tokens, 1});
                check(gate.swiglu(up, input, mask, limit, gelu),
                    where(expand_dims(equal(mask, array(0, int32)), -1), actual, zeros_like(actual)), 0.f);
            }
        }
    }
}

static void check_dispatch_preparation() {
    for (const auto [rows, width] : {std::pair{24, 20481}, std::pair{49, 8192},
        std::pair{105, 10240}, std::pair{320, 10240}, std::pair{512, 2560}, std::pair{1025, 320},
        std::pair{2048, 2560}, std::pair{2048, 6145}}) {
        const auto op = random_weight(rows, width, 1041);
        for (auto dtype : {float16, bfloat16, float32}) {
            MlxKernelPreparation preparation(512, 512);
            const auto add = [&](int tokens) {
                auto input = zeros({tokens, width}, dtype);
                preparation.add(op.matmul(input));
                preparation.add(op.rms_matmul(input, .000001f));
                for (const auto& output : MlxTqWeight::matmul_many({op, op}, input)) preparation.add(output);
                preparation.add(op.swiglu(op, input, zeros({tokens, 1}, int32)));
                for (int groups : {1, 3, 4}) if (rows % groups == 0)
                    preparation.add(op.gated_mix(input, zeros({tokens, rows}, dtype), groups));
                for (int groups : {3, 7, 8}) if (rows % groups == 0)
                    preparation.add(op.grouped_row_matmul(zeros({tokens, groups, width}, dtype), groups));
            };
            preparation.collect([&] { for (const auto tokens : preparation.row_buckets()) add(tokens); });
            preparation.finish();
            for (int tokens = 1; tokens <= 128; ++tokens) {
                preparation.collect([&] { add(tokens); });
                if (preparation.size()) throw std::runtime_error("TQ loading misses a small-batch compilation path: M="
                    + std::to_string(tokens) + ", N=" + std::to_string(rows) + ", K=" + std::to_string(width)
                    + ", dtype=" + (dtype == float16 ? "float16" : dtype == bfloat16 ? "bfloat16" : "float32"));
            }
        }
    }
}

static void check_repeated_routes() {
    constexpr int experts = 6, width = 1537, hidden = 129;
    const auto down = MlxTqWeight::from_blob(random_blob(experts * 513, hidden, 2083), hidden, experts);
    const std::vector<int> distinct{0, 1, 2, 3};
    if (make_tq_reuse_plan(distinct, 2, 2, experts))
        throw std::runtime_error("TQ reuses routes without repeated experts");
    const std::vector<int> pair_ids{0, 1, 2, 0, 1, 2}, triple_ids{0, 1, 0, 1, 0, 2};
    const auto pair_plan = make_tq_reuse_plan(pair_ids, 2, 3, experts);
    const auto triple_plan = make_tq_reuse_plan(triple_ids, 3, 2, experts);
    if (!pair_plan || pair_plan->shape(1) != 3 || !triple_plan || triple_plan->shape(1) != 5)
        throw std::runtime_error("TQ reuse plan does not preserve its maximum row count");
    for (const auto [tokens, columns] : {std::pair{4, 5}, std::pair{5, 7}, std::pair{6, 7},
        std::pair{8, 5}, std::pair{12, 7}, std::pair{16, 7}}) {
        std::vector<int> ids(tokens * 3);
        for (int row = 0; row < int(ids.size()); ++row) ids[row] = row % 3;
        const auto plan = make_tq_reuse_plan(ids, tokens, 3, experts);
        if (!plan || plan->shape(1) != columns) throw std::runtime_error("TQ reuse plan did not select the cheaper grouping");
        std::vector<bool> seen(ids.size(), false);
        const auto* blocks = plan->data<int>();
        for (int group = 0; group < plan->shape(0); ++group) for (int slot = 1; slot < columns; ++slot) {
            const int index = blocks[group * columns + slot];
            if (index < 0) continue;
            if (index >= int(ids.size()) || seen[index] || ids[index] != blocks[group * columns])
                throw std::runtime_error("TQ reuse plan duplicated or mixed expert routes");
            seen[index] = true;
        }
        if (!std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }))
            throw std::runtime_error("TQ reuse plan lost routes");
    }
    for (int layout : {0, 1, 2, 3}) {
        const int gs = layout == 1 ? 512 : 16;
        const auto gate_blob = random_blob(experts * hidden, width, 2081, gs, layout == 2);
        const auto up_blob = random_blob(experts * hidden, width, 2082, gs, layout >= 2);
        const auto gate = MlxTqWeight::from_blob(gate_blob, width, experts);
        const auto up = MlxTqWeight::from_blob(up_blob, width, experts);
        const auto pooled_gate = pooled_weight(gate_blob, width, experts);
        const auto pooled_up = pooled_weight(up_blob, width, experts);
        for (auto dtype : {float16, bfloat16, float32}) {
            const auto ids = array(pair_ids.begin(), {2, 3}, int32);
            const auto input = astype(reshape(sin(arange(2 * width, float32)), {2, width}), dtype);
            const auto expected = gate.swiglu(up, input, ids);
            check(gate.swiglu(up, input, ids, 0.f, false, &*pair_plan), expected, .000001f);
            check(pooled_gate.swiglu(pooled_up, input, ids, 0.f, false, &*pair_plan), expected, .000001f);
            check(gate.routed_matmul(input, ids, &*pair_plan), gate.routed_matmul(input, ids), .0001f);
        }
        for (int tokens : {2, 3, 5, 6, 8, 12, 16}) {
            std::vector<int> selected(tokens * 3);
            for (int row = 0; row < tokens; ++row) {
                selected[row * 3] = 1;
                selected[row * 3 + 1] = row % 2 ? 2 : -1;
                selected[row * 3 + 2] = experts;
            }
            const auto plan = make_tq_reuse_plan(selected, tokens, 3, experts);
            if (!plan) throw std::runtime_error("TQ repeated routes have no reuse plan");
            const auto ids = array(selected.begin(), {tokens, 3}, int32);
            const auto weights = reshape(sin(arange(tokens * 3, float32)), {tokens, 3});
            for (auto dtype : {float16, bfloat16, float32}) for (bool gelu : {false, true})
                for (float limit : {0.f, .01f}) {
                const auto input = astype(reshape(sin(arange(tokens * width, float32)), {tokens, width}), dtype);
                const auto expected = gate.swiglu(up, input, ids, limit, gelu);
                const auto actual = gate.swiglu(up, input, ids, limit, gelu, &*plan);
                check(actual, expected, .000001f);
                if (tokens == 6 && limit == 0.f) {
                    check(pooled_gate.swiglu(pooled_up, input, ids, limit, gelu), expected, .000001f);
                    check(pooled_gate.swiglu(up, input, ids, limit, gelu), expected, .000001f);
                    check(gate.swiglu(pooled_up, input, ids, limit, gelu), expected, .000001f);
                    check(pooled_gate.swiglu(pooled_up, input, ids, limit, gelu, &*plan), expected, .000001f);
                    check(pooled_gate.swiglu(up, input, ids, limit, gelu, &*plan), expected, .000001f);
                    check(gate.swiglu(pooled_up, input, ids, limit, gelu, &*plan), expected, .000001f);
                }
                if (gate.can_concatenate_rows(up)) {
                    const auto combined = MlxTqWeight::concatenate_projections({gate, up});
                    check(combined.swiglu_split(input, ids, limit, gelu), expected, .000001f);
                }
                const auto individual = astype(reshape(cos(arange(tokens * 3 * width, float32)), {tokens * 3, width}), dtype);
                check(gate.swiglu(up, individual, ids, limit, gelu),
                    gate.swiglu(up, individual, ids, limit, gelu, &*plan), .000001f);
                check(gate.routed_matmul(input, ids, &*plan), gate.routed_matmul(input, ids), .0001f);
                check(down.routed_matmul(actual, ids, &*plan), down.routed_matmul(expected, ids), .000001f);
                check(down.routed_matmul_reduce(actual, ids, weights, &*plan),
                    down.routed_matmul_reduce(expected, ids, weights), .000001f);
            }
        }
    }
}

static void check_device_reuse_plans() {
    for (int tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17}) for (int routes : {1, 3, 16})
        for (int pattern : {0, 1, 2}) {
            constexpr int experts = 1024;
            std::vector<int> selected(tokens * routes);
            for (int index = 0; index < int(selected.size()); ++index)
                selected[index] = pattern == 0 ? index : pattern == 1 ? 0
                    : index % 7 == 0 ? -1 : index % 11 == 0 ? experts : index % 5;
            const auto ids = add(array(selected.begin(), {tokens, routes}, int32), array(0));
            const auto plan = make_tq_reuse_plan(ids, experts);
            if (ids.is_available()) throw std::runtime_error("TQ device reuse planning evaluates routes on the host");
            if (tokens < 3 || tokens > 16) {
                if (plan) throw std::runtime_error("TQ device planning changes single/pair or long-token dispatch");
                continue;
            }
            if (!plan || plan->shape() != Shape{tokens * routes, 18})
                throw std::runtime_error("TQ device reuse plan geometry differs");
            eval({*plan});
            std::vector<bool> seen(selected.size(), false);
            const auto* values = plan->data<int>();
            for (int row = 0; row < plan->shape(0); ++row) {
                const int count = values[row * 18 + 1];
                if (count < 0 || count > 16) throw std::runtime_error("TQ device group exceeds its capacity");
                for (int member = 0; member < count; ++member) {
                    const int index = values[row * 18 + member + 2];
                    if (index < 0 || index >= int(selected.size()) || seen[index]
                        || selected[index] != values[row * 18])
                        throw std::runtime_error("TQ device plan duplicates or mixes expert routes");
                    seen[index] = true;
                }
                for (int member = count; member < 16; ++member)
                    if (values[row * 18 + member + 2] != -1)
                        throw std::runtime_error("TQ device plan leaves live padding");
            }
            if (!std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }))
                throw std::runtime_error("TQ device plan lost expert routes");
        }
    constexpr int experts = 20, width = 2305, outputs = 1027, routes = 2;
    for (int layout : {0, 1, 2}) {
        const auto blob = random_blob(experts * outputs, width, 5471, layout == 0 ? 16 : 512, layout == 2);
        const auto up_blob = random_blob(experts * outputs, width, 5472, layout == 0 ? 16 : 512, layout == 2);
        const auto weight = MlxTqWeight::from_blob(blob, width, experts), up = MlxTqWeight::from_blob(up_blob, width, experts);
        const auto pooled = pooled_weight(blob, width, experts);
        for (int tokens : {3, 6, 8, 12, 16}) {
            std::vector<int> selected(tokens * routes);
            for (int token = 0; token < tokens; ++token) {
                selected[token * routes] = 0;
                selected[token * routes + 1] = token % 4 == 0 ? -1 : token + 1;
            }
            const auto ids = array(selected.begin(), {tokens, routes}, int32);
            const auto plan = make_tq_reuse_plan(ids, experts);
            for (auto dtype : {float16, bfloat16, float32}) for (bool shared : {true, false}) {
                const auto input = reshape(astype(sin(arange(tokens * (shared ? 1 : routes) * width, float32)) * .1f, dtype),
                    shared ? Shape{tokens, width} : Shape{tokens, routes, width});
                const auto expected = weight.routed_matmul(input, ids);
                const auto actual = weight.routed_matmul(input, ids, &*plan);
                if (!allclose(actual, expected, .003, .0001).item<bool>())
                    throw std::runtime_error("TQ device repeated projection differs from ordinary projection");
                check(pooled.routed_matmul(input, ids, &*plan), actual, 0.f);
                for (bool gelu : {false, true}) {
                    const auto result = weight.swiglu(up, input, ids, .01f, gelu, &*plan);
                    if (!allclose(result, weight.swiglu(up, input, ids, .01f, gelu), .003, .0001).item<bool>())
                        throw std::runtime_error("TQ device repeated Gate/Up differs from ordinary projection");
                }
            }
        }
    }
}

static void check_compact_repeated_routes() {
    constexpr int experts = 3, width = 1537, routes = 2;
    for (int outputs : {129, 1025}) for (int layout : {0, 1, 2}) {
        const int gs = layout == 0 ? 16 : 512;
        const auto blob = random_blob(experts * outputs, width, 5291, gs, layout == 2);
        const auto up_blob = random_blob(experts * outputs, width, 5292, gs, layout == 2);
        const auto weight = MlxTqWeight::from_blob(blob, width, experts);
        const auto up = MlxTqWeight::from_blob(up_blob, width, experts);
        const auto pool = pooled_weight(blob, width, experts), up_pool = pooled_weight(up_blob, width, experts);
        for (int tokens : {4, 5, 6}) for (auto dtype : {float16, bfloat16, float32}) {
            std::vector<int> selected(tokens * routes);
            for (int token = 0; token < tokens; ++token) {
                selected[token * routes] = 0;
                selected[token * routes + 1] = token % 2 ? 1 : -1;
            }
            const auto ids = array(selected.begin(), {tokens, routes}, int32);
            const auto plan = make_tq_reuse_plan(selected, tokens, routes, experts);
            if (!plan) throw std::runtime_error("TQ compact reuse fixture has no plan");
            for (bool shared : {true, false}) {
                const auto input = reshape(astype(sin(arange(tokens * (shared ? 1 : routes) * width, float32)) * .1f, dtype),
                    shared ? Shape{tokens, width} : Shape{tokens, routes, width});
                const auto expected = weight.routed_matmul(input, ids);
                const auto actual = weight.routed_matmul(input, ids, &*plan);
                if (!allclose(actual, expected, .003, .0001).item<bool>())
                    throw std::runtime_error("TQ compact repeated projection differs from ordinary projection");
                check(pool.routed_matmul(input, ids, &*plan), actual, 0.f);
                for (bool gelu : {false, true}) for (float limit : {0.f, .01f}) {
                    const auto reference = weight.swiglu(up, input, ids, limit, gelu);
                    const auto result = weight.swiglu(up, input, ids, limit, gelu, &*plan);
                    if (!allclose(result, reference, .003, .0001).item<bool>())
                        throw std::runtime_error("TQ compact repeated Gate/Up differs from ordinary projection");
                    check(pool.swiglu(up_pool, input, ids, limit, gelu, &*plan), result, 0.f);
                }
            }
        }
    }
    const bool hardware_nax = !mlx_metal_nax_disabled() && mlx_apple_chip_starts_with("Apple M5");
    for (const auto geometry : {std::array{1024, 512, 2, 16}, std::array{1024, 1024, 7, 16},
            std::array{1024, 1024, 8, 8}, std::array{1024, 2048, 1, 8}}) {
        const int k = geometry[0], n = geometry[1], selected_routes = geometry[2];
        const auto gate = MlxTqWeight::from_blob(random_blob(selected_routes * n, k, 5381), k, selected_routes);
        const auto up = MlxTqWeight::from_blob(random_blob(selected_routes * n, k, 5382), k, selected_routes);
        for (int tokens : {8, 16}) {
            std::vector<int> selected(tokens * selected_routes);
            for (int row = 0; row < int(selected.size()); ++row) selected[row] = row % selected_routes;
            const auto ids = array(selected.begin(), {tokens, selected_routes}, int32);
            const auto plan = make_tq_reuse_plan(selected, tokens, selected_routes, selected_routes);
            if (!plan) throw std::runtime_error("TQ NAX dispatch fixture has no reuse plan");
            detail::ComponentProfile profile;
            {
                detail::ScopedComponentProfile scope(&profile);
                (void)gate.swiglu(up, zeros({tokens, k}, float16), ids, 0.f, false, &*plan);
            }
            const std::string marker = hardware_nax && tokens >= geometry[3]
                ? "tq.dispatch.routed_reuse_nax." : "tq.dispatch.routed_pair_reuse.";
            if (!std::any_of(profile.timings().begin(), profile.timings().end(),
                    [&](const auto& entry) { return entry.first.starts_with(marker); }))
                throw std::runtime_error("TQ repeated NAX dispatch ignores projection parallelism: K="
                    + std::to_string(k) + ", N=" + std::to_string(n) + ", routes="
                    + std::to_string(selected_routes) + ", M=" + std::to_string(tokens));
        }
    }
}

static void check_tiled_repeated_routes() {
    constexpr int experts = 4, width = 1027, hidden = 513, routes = 2;
    for (int layout : {0, 1, 2}) {
        const int gs = layout == 1 ? 512 : 16;
        const auto gate_blob = random_blob(experts * hidden, width, 3131, gs, layout == 2);
        const auto up_blob = random_blob(experts * hidden, width, 3132, gs, layout == 2);
        const auto down_blob = random_blob(experts * width, hidden, 3133, gs, layout == 2);
        const auto gate = MlxTqWeight::from_blob(gate_blob, width, experts);
        const auto up = MlxTqWeight::from_blob(up_blob, width, experts);
        const auto down = MlxTqWeight::from_blob(down_blob, hidden, experts);
        const auto pooled_gate = pooled_weight(gate_blob, width, experts);
        const auto pooled_up = pooled_weight(up_blob, width, experts);
        const auto pooled_down = pooled_weight(down_blob, hidden, experts);
        for (int tokens : {8, 9, 12, 15, 16}) for (int pattern : {0, 1, 2, 3}) {
            std::vector<int> selected(tokens * routes);
            for (int token = 0; token < tokens; ++token) {
                selected[token * routes] = pattern == 3 ? -1 : 0;
                selected[token * routes + 1] = pattern == 1 ? 1 + token % 3 : pattern == 2 ? 0 : 1;
            }
            const auto plan = make_tq_reuse_plan(selected, tokens, routes, experts);
            if (!plan) throw std::runtime_error("TQ tiled reuse has no plan");
            const auto ids = array(selected.begin(), {tokens, routes}, int32);
            const auto input = astype(reshape(sin(arange(tokens * width, float32)) * array(.1f),
                {tokens, width}), float16);
            const auto individual = astype(reshape(cos(arange(tokens * routes * width, float32)) * array(.1f),
                {tokens, routes, width}), float16);
            const auto down_input = astype(reshape(cos(arange(tokens * routes * hidden, float32)) * array(.1f),
                {tokens, routes, hidden}), float16);
            auto check_close = [&](const array& actual, const array& expected) {
                if (!all(isfinite(actual)).item<bool>() || !allclose(actual, expected, .003, .0001).item<bool>())
                    throw std::runtime_error("TQ tiled repeated routes differ from single-token projections");
            };
            for (bool gelu : {false, true}) for (float limit : {0.f, .01f}) for (bool shared : {true, false}) {
                const auto& source = shared ? input : individual;
                std::vector<array> expected;
                for (int token = 0; token < tokens; ++token)
                    expected.push_back(gate.swiglu(up, take(source, array({token}), 0),
                        take(ids, array({token}), 0), limit, gelu));
                const auto reference = mlx::core::concatenate(expected, 0);
                check_close(gate.swiglu(up, source, ids, limit, gelu, &*plan), reference);
                check_close(pooled_gate.swiglu(pooled_up, source, ids, limit, gelu, &*plan), reference);
                check_close(pooled_gate.swiglu(up, source, ids, limit, gelu, &*plan), reference);
                check_close(gate.swiglu(pooled_up, source, ids, limit, gelu, &*plan), reference);
            }
            std::vector<array> expected;
            for (int token = 0; token < tokens; ++token)
                expected.push_back(down.routed_matmul(take(down_input, array({token}), 0), take(ids, array({token}), 0)));
            const auto reference = mlx::core::concatenate(expected, 0);
            check_close(down.routed_matmul(down_input, ids, &*plan), reference);
            check_close(pooled_down.routed_matmul(down_input, ids, &*plan), reference);
            if (pattern == 0 && (tokens == 8 || tokens == 9)) {
                const auto padded = mlx::core::concatenate({*plan,
                    full({3, plan->shape(1)}, -1, int32)}, 0);
                check(down.routed_matmul(down_input, ids, &padded),
                    down.routed_matmul(down_input, ids, &*plan), 0.f);
                check(gate.swiglu(up, input, ids, 0.f, false, &padded),
                    gate.swiglu(up, input, ids, 0.f, false, &*plan), 0.f);
            }
        }
    }
}

static void check_partial_route_tiles() {
    constexpr int experts = 3, width = 1025, outputs = 257, routes = 3;
    for (int layout : {0, 1, 2, 3}) {
        auto blob = random_blob(experts * outputs, width, 3201, layout == 0 ? 16 : 512, layout >= 2);
        if (layout == 3) {
            std::vector<std::vector<std::uint8_t>> children;
            for (int expert = 0; expert < experts; ++expert)
                children.push_back(random_blob(outputs, width, 3202 + expert, 512, true));
            blob = expert_blob(children, outputs, width);
        }
        const auto weight = MlxTqWeight::from_blob(blob, width, experts);
        const auto pooled = pooled_weight(blob, width, experts);
        for (int tokens : {32, 33, 63, 64, 65, 96, 127, 128, 129}) {
            std::vector<int> selected(tokens * routes), order(tokens * routes), blocks;
            for (int token = 0; token < tokens; ++token) {
                selected[token * routes] = 0;
                selected[token * routes + 1] = 1;
                selected[token * routes + 2] = -1;
            }
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return selected[a] < selected[b]; });
            for (int first = 0; first < int(order.size());) {
                int last = first + 1;
                while (last < int(order.size()) && selected[order[last]] == selected[order[first]]) ++last;
                for (int row = first; row < last; row += 128)
                    blocks.insert(blocks.end(), {row, selected[order[first]], std::min(128, last - row)});
                first = last;
            }
            const int count = blocks.size() / 3;
            const auto ids = array(selected.begin(), {tokens, routes}, int32);
            const auto permutation = array(order.begin(), {tokens * routes}, int32);
            const MlxTqRoutePlan plan{array(blocks.begin(), {count, 3}, int32), array({count}, int32),
                count, 128, tokens * routes, experts};
            for (bool shared : {false, true}) {
                const int rows = shared ? tokens : tokens * routes;
                const auto input = astype(reshape(sin(arange(rows * width, float32)) * array(.1f),
                    shared ? Shape{tokens, width} : Shape{tokens, routes, width}), float16);
                const auto expected = weight.routed_matmul(input, ids);
                const auto actual = weight.routed_matmul_sorted(input, ids, permutation, plan);
                if (!all(isfinite(actual)).item<bool>() || !allclose(actual, expected, .003, .0001).item<bool>())
                    throw std::runtime_error("TQ partial routed tile differs from projection reference");
                check(pooled.routed_matmul_sorted(input, ids, permutation, plan), actual, 0.f);
            }
        }
    }
}

static void check_large_row_offsets() {
    const auto kernel = fast::metal_kernel("mfq_cpp_tq_test_large_offsets", {"rows"}, {"offsets"}, R"METAL(
uint index = thread_position_in_grid.x;
if (index < uint(rows_shape[0])) offsets[index] = tq_row_offset<K>(rows[index], K - 1u);
)METAL", kTqDecode, true, false);
    const std::vector<std::uint32_t> rows{0, 1, 838860, 884735, 0x7fffffffu, 0xffffffffu};
    for (int width : {513, 5120, 20480}) {
        const auto actual = kernel({array(rows.begin(), {int(rows.size())}, uint32)}, {Shape{int(rows.size())}},
            {uint64}, {int(rows.size()), 1, 1}, {32, 1, 1}, {{"K", width}}, std::nullopt, false, {}).front();
        eval({actual});
        for (std::size_t index = 0; index < rows.size(); ++index)
            if (actual.data<std::uint64_t>()[index] != std::uint64_t(rows[index]) * width + width - 1)
                throw std::runtime_error("TQ output row address overflows");
    }
}

static void check_unaligned_inputs() {
    for (int width : {129, 130, 131, 513, 1025}) {
        const auto op = random_weight(17, width, 2091);
        const auto weight = astype(op.embedding(arange(17, int32)), float32);
        for (auto dtype : {float16, bfloat16, float32}) for (int tokens : {2, 6, 7, 17, 65, 128}) {
            const auto input = astype(reshape(sin(arange(tokens * width, float32)), {tokens, width}), dtype);
            const auto source = astype(input, float32);
            eval({source, weight});
            std::vector<float> values(tokens * 17);
            for (int token = 0; token < tokens; ++token) for (int row = 0; row < 17; ++row) {
                double value = 0;
                for (int col = 0; col < width; ++col)
                    value += double(source.data<float>()[token * width + col]) * weight.data<float>()[row * width + col];
                values[token * 17 + row] = float(value);
            }
            const auto expected = astype(array(values.begin(), {tokens, 17}, float32), dtype);
            const auto actual = op.matmul(input);
            const float relative = dtype == bfloat16 ? .008f : dtype == float16 ? .001f : .00001f;
            if (!allclose(actual, expected, relative, .000001f).item<bool>()) {
                throw std::runtime_error("TQ unaligned projection differs from decoded GEMM: K="
                    + std::to_string(width) + ", M=" + std::to_string(tokens));
            }
        }
    }
}

static void check_grouped_projections() {
    for (const auto [groups, outputs, width] : {std::tuple{3, 17, 513}, std::tuple{7, 31, 1025}, std::tuple{8, 128, 512}, std::tuple{8, 129, 320}, std::tuple{8, 1024, 4096}, std::tuple{3, 1025, 4097}})
        for (int layout : {0, 1, 2}) {
            const auto blob = random_blob(groups * outputs, width, 2101, layout == 0 ? 16 : 512, layout == 2);
            const auto weight = MlxTqWeight::from_blob(blob, width);
            const auto bank = MlxTqWeight::from_blob(blob, width, groups);
            std::optional<MlxTqWeight> pooled;
            if (outputs >= 1024) {
                MlxMappedTqRows rows(blob, width);
                const auto range = rows.packed_rows(0, groups * outputs);
                std::vector<std::uint32_t> words(17 + range.bytes / 4);
                std::memcpy(words.data() + 17, blob.data() + range.offset, range.bytes);
                const std::array<std::uint64_t, 1> offsets{17};
                pooled = MlxTqWeight::from_pool(blob, array(words.data(), Shape{int(words.size())}), offsets, width);
            }
            for (auto dtype : {float16, bfloat16, float32})
                for (int tokens : outputs >= 1024 ? std::vector{1, 2, 3, 4, 5, 6, 7}
                        : std::vector{1, 2, 3, 6, 7, 16, 17, 33, 65, 128, 257}) {
                    auto input = astype(reshape(sin(arange(tokens * groups * width, float32)), {tokens, groups, width}), dtype);
                    if (tokens == 6) input = transpose(reshape(input, {tokens, width, groups}), {0, 2, 1});
                    std::vector<array> expected;
                    for (int group = 0; group < groups; ++group)
                        expected.push_back(bank.expert_view(group).matmul(reshape(
                            slice(input, {0, group, 0}, {tokens, group + 1, width}), {tokens, width})));
                    const auto reference = stack(expected, 1);
                    const auto actual = weight.grouped_row_matmul(reshape(input, {1, tokens, groups, width}), groups);
                    if (pooled) check(pooled->grouped_row_matmul(reshape(input, {1, tokens, groups, width}), groups), actual, 0.f);
                    const double absolute = dtype == float32 && tokens >= 256 ? .0001 : .00001;
                    if (!allclose(actual, reshape(reference, actual.shape()), dtype == bfloat16 ? .008 : .003, absolute).item<bool>()) {
                        auto error = reshape(abs(astype(actual, float32) - astype(reshape(reference, actual.shape()), float32)), {-1});
                        const auto index = argmax(error).item<std::uint32_t>();
                        throw std::runtime_error("TQ static grouped projection differs: M=" + std::to_string(tokens)
                            + ", G=" + std::to_string(groups) + ", K=" + std::to_string(width)
                            + ", layout=" + std::to_string(layout) + ", dtype="
                            + (dtype == float16 ? "float16" : dtype == bfloat16 ? "bfloat16" : "float32")
                            + ", max_error=" + std::to_string(max(error).item<float>())
                            + ", index=" + std::to_string(index));
                    }
                    if (dtype == float32 && tokens == 257 && groups == 7) {
                        const auto dense = astype(weight.embedding(arange(groups * outputs, int32)), float32);
                        const auto source = contiguous(input);
                        eval({source, dense});
                        std::vector<float> oracle(tokens * groups * outputs);
                        for (int token = 0; token < tokens; ++token)
                            for (int row = 0; row < groups * outputs; ++row) {
                                double value = 0;
                                for (int col = 0; col < width; ++col)
                                    value += double(source.data<float>()[(token * groups + row / outputs) * width + col])
                                        * dense.data<float>()[row * width + col];
                                oracle[token * groups * outputs + row] = float(value);
                            }
                        check(actual, array(oracle.begin(), actual.shape(), float32), .0001f);
                    }
                }
        }
}

static void check_normalized_projections() {
    for (const auto [rows, width] : {std::pair{16, 8192}, std::pair{24, 20481}, std::pair{64, 8192},
            std::pair{15, 8192}, std::pair{65, 8192}, std::pair{24, 4097}})
        for (int layout : {0, 1, 2}) {
            const auto blob = random_blob(rows, width, 2111, layout == 0 ? 16 : 512, layout == 2);
            const auto weight = MlxTqWeight::from_blob(blob, width);
            const auto pooled = pooled_weight(blob, width, 1);
            for (auto dtype : {float16, bfloat16, float32})
                for (int tokens : {1, 2, 3, 5, 6, 7, 17}) for (float eps : {.000001f, .25f}) {
                    auto input = astype(reshape(sin(arange(tokens * width, float32)), {tokens, width}), dtype);
                    if (tokens == 6) input = transpose(reshape(input, {width, tokens}), {1, 0});
                    const auto source = astype(input, float32);
                    const auto expected = weight.matmul(source * rsqrt(mean(source * source, -1, true) + array(eps)));
                    const auto actual = weight.rms_matmul(reshape(input, {1, tokens, width}), eps);
                    if (actual.dtype() != float32) throw std::runtime_error("TQ normalized projection changed output precision");
                    check(actual, reshape(expected, actual.shape()), .00001f);
                    if (tokens == 2) check(pooled.rms_matmul(input, eps), expected, .00001f);
                }
            check(weight.rms_matmul(zeros({2, width}, float16), .000001f), zeros({2, rows}, float32), 0.f);
            for (float eps : {0.f, -1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
                bool rejected = false;
                try { (void)weight.rms_matmul(zeros({1, width}, float16), eps); }
                catch (const std::invalid_argument&) { rejected = true; }
                if (!rejected) throw std::runtime_error("TQ normalized projection accepts invalid epsilon");
            }
        }
}

static void check_normalized_prefill() {
    for (const auto [rows, width] : {std::pair{16, 8193}, std::pair{24, 20481}, std::pair{64, 8192}})
    for (int layout : {0, 1, 2}) {
        const auto blob = random_blob(rows, width, 2511, layout == 0 ? 16 : 512, layout == 2);
        const auto weight = MlxTqWeight::from_blob(blob, width), pooled = pooled_weight(blob, width, 1);
        const auto decoded = astype(weight.embedding(arange(rows, int32)), float32);
        MlxKernelPreparation preparation(4096, 4096);
        preparation.collect([&] {
            for (int tokens : {32, 128, 256, 512, 1024, 2048, 4096})
                preparation.add(weight.rms_matmul(zeros({tokens, width}, float16), .000001f));
        });
        preparation.finish();
        for (int tokens : {33, 63, 129, 257, 513, 1025, 2049}) {
            preparation.collect([&] { preparation.add(weight.rms_matmul(zeros({tokens, width}, float16), .000001f)); });
            if (preparation.size()) throw std::runtime_error("TQ loading misses normalized prefill compilation");
        }
        for (int tokens : {32, 33, 64, 129, 257, 513, 1025}) for (float eps : {.000001f, .25f}) {
            auto input = astype(reshape(sin(arange(tokens * width, float32)), {tokens, width}), float16);
            if (tokens == 33) input = transpose(reshape(input, {width, tokens}), {1, 0});
            const auto source = astype(input, float32);
            const auto actual = reshape(weight.rms_matmul(reshape(input, {1, tokens, width}), eps), {tokens, rows});
            if (actual.dtype() != float32) throw std::runtime_error("TQ normalized prefill changed output precision");
            check(pooled.rms_matmul(input, eps), actual, 0.f);
            eval({source, decoded, actual});
            const auto contiguous_source = contiguous(source);
            eval({contiguous_source});
            const auto* xs = contiguous_source.data<float>();
            const auto* ws = decoded.data<float>();
            for (int row : {0, tokens / 2, tokens - 1}) {
                double square = 0.;
                for (int k = 0; k < width; ++k) square += double(xs[row * width + k]) * xs[row * width + k];
                const double inverse = 1. / std::sqrt(square / width + double(eps));
                for (int out = 0; out < rows; ++out) {
                    double value = 0.;
                    for (int k = 0; k < width; ++k) value += double(xs[row * width + k]) * ws[out * width + k];
                    const double error = std::abs(double(actual.data<float>()[row * rows + out]) - value * inverse);
                    if (!std::isfinite(error) || error > .00001)
                        throw std::runtime_error("TQ normalized prefill differs from CPU FP64 reference");
                }
            }
        }
        check(weight.rms_matmul(zeros({33, width}, float16), .000001f), zeros({33, rows}, float32), 0.f);
    }
}

static void check_short_expanding_prefill() {
    constexpr int experts = 64, routes = 4, width = 640, outputs = 1297;
    const auto weight = MlxTqWeight::from_blob(random_blob(experts * outputs, width, 2141), width, experts);
    const auto decoded = weight.embedding(arange(experts * outputs, int32), float32);
    const auto dense = transpose(reshape(decoded, {experts, outputs, width}), {0, 2, 1});
    for (const int tokens : {255, 256, 257, 511}) {
        std::vector<int> selected(tokens * routes);
        for (int i = 0; i < int(selected.size()); ++i) selected[i] = (i * 7 + i / routes) % experts;
        const auto ids = array(selected.begin(), {tokens, routes}, int32);
        const auto input = astype(reshape(sin(arange(tokens * routes * width, float32)),
            {tokens, routes, width}), float16);
        const auto expected = astype(reshape(gather_mm(reshape(astype(input, float32), {-1, 1, width}), dense,
            arange(tokens * routes, int32), reshape(ids, {-1})), {tokens, routes, outputs}), float16);
        const auto actual = weight.routed_matmul(input, ids);
        if (!allclose(actual, expected, .003, .000001).item<bool>())
            throw std::runtime_error("TQ short expanding prefill differs from decoded gather GEMM");
        const auto source = astype(input, float32);
        const auto output = astype(actual, float32);
        const auto maximum_error = argmax(reshape(abs(output - astype(expected, float32)), {-1})).item<std::uint32_t>();
        eval({source, output, decoded});
        for (const auto index : {std::uint32_t{0}, std::uint32_t(actual.size() - 1), maximum_error}) {
            const auto row = index / outputs, col = index % outputs;
            double sum = 0;
            for (int k = 0; k < width; ++k)
                sum += double(source.data<float>()[row * width + k])
                    * decoded.data<float>()[(selected[row] * outputs + col) * width + k];
            const auto oracle = float(float16_t(float(sum)));
            if (std::abs(output.data<float>()[index] - oracle) > .002f * std::abs(oracle) + .000001f)
                throw std::runtime_error("TQ short expanding prefill differs from the CPU dot oracle");
        }
    }
}

static void check_small_sorted_routes() {
    constexpr int tokens = 7, experts = 2, width = 513, outputs = 37;
    for (const bool columns : {false, true}) {
        const auto weight = MlxTqWeight::from_blob(random_blob(experts * outputs, width, 2145, 16, columns), width, experts);
        const std::vector<int> selected{0, -1, 1, 0, 2, -1, 1};
        std::vector<int> permutation(tokens); std::iota(permutation.begin(), permutation.end(), 0);
        std::stable_sort(permutation.begin(), permutation.end(), [&](int a, int b) { return selected[a] < selected[b]; });
        std::vector<int> blocks;
        for (int first = 0; first < tokens;) {
            const int expert = selected[permutation[first]];
            int last = first;
            while (last < tokens && selected[permutation[last]] == expert) ++last;
            blocks.insert(blocks.end(), {first, expert, last - first});
            first = last;
        }
        const int count = blocks.size() / 3;
        const MlxTqRoutePlan plan{array(blocks.begin(), {count, 3}, int32), array({count}, int32), count, 16, tokens, experts};
        const auto ids = array(selected.begin(), {tokens, 1}, int32), order = array(permutation.begin(), {tokens}, int32);
        const auto input = astype(reshape(sin(arange(tokens * width, float32)), {tokens, width}), float16);
        const auto decoded = weight.embedding(arange(experts * outputs, int32), float32), source = astype(input, float32);
        eval({decoded, source});
        std::vector<float> expected(tokens * outputs, 0.f);
        for (int token = 0; token < tokens; ++token) if (selected[token] >= 0 && selected[token] < experts)
            for (int row = 0; row < outputs; ++row) {
                double sum = 0;
                for (int k = 0; k < width; ++k) sum += double(source.data<float>()[token * width + k])
                    * decoded.data<float>()[(selected[token] * outputs + row) * width + k];
                expected[token * outputs + row] = float(float16_t(float(sum)));
            }
        const auto actual = weight.routed_matmul_sorted(input, ids, order, plan);
        if (!allclose(astype(actual, float32), array(expected.begin(), {tokens, 1, outputs}), .003, .000001).item<bool>())
            throw std::runtime_error("TQ small sorted routes differ from the CPU oracle");
    }
}

static void check_shared_expansion_tiles() {
    constexpr int experts = 5, routes = 3, tokens = 97;
    for (const int width : {513, 640, 1025}) for (const int gs : {16, 32, 512})
        for (const bool columns : {false, true}) {
            if (columns && gs != 16) continue;
            const int outputs = width * 2 + 17;
            const auto blob = random_blob(experts * outputs, width, 2151, gs, columns);
            const auto resident = MlxTqWeight::from_blob(blob, width, experts);
            const auto pooled = pooled_weight(blob, width, experts);
            const auto decoded = resident.embedding(arange(experts * outputs, int32), float32);
            const auto dense = transpose(reshape(decoded, {experts, outputs, width}), {0, 2, 1});
            std::vector<int> selected(tokens * routes), permutation(tokens * routes);
            for (int i = 0; i < int(selected.size()); ++i)
                selected[i] = i % 13 == 0 ? -1 : i % 17 == 0 ? experts : (i / 37 + i % 3) % experts;
            std::iota(permutation.begin(), permutation.end(), 0);
            std::stable_sort(permutation.begin(), permutation.end(), [&](int a, int b) { return selected[a] < selected[b]; });
            std::vector<MlxTqRoutePlan> plans;
            for (const int block_rows : {1, 16, 17, 32, 64}) {
                std::vector<int> blocks;
                for (int first = 0; first < int(selected.size());) {
                    const int expert = selected[permutation[first]];
                    int last = first;
                    while (last < int(selected.size()) && selected[permutation[last]] == expert) ++last;
                    for (int row = first; row < last; row += block_rows)
                        blocks.insert(blocks.end(), {row, expert, std::min(block_rows, last - row)});
                    first = last;
                }
                const int count = blocks.size() / 3;
                plans.push_back({array(blocks.begin(), {count, 3}, int32), array({count}, int32),
                    count, 64, tokens * routes, experts});
            }
            const auto ids = array(selected.begin(), {tokens, routes}, int32);
            const auto order = array(permutation.begin(), {tokens * routes}, int32);
            const auto safe_ids = reshape(clip(ids, array(0), array(experts - 1)), {-1});
            const auto valid = expand_dims(logical_and(ids >= array(0), ids < array(experts)), -1);
            for (const bool shared : {false, true}) {
                const auto input = astype(reshape(sin(arange(tokens * (shared ? 1 : routes) * width, float32)),
                    shared ? Shape{tokens, width} : Shape{tokens, routes, width}), float16);
                auto source_ids = arange(tokens * routes, int32);
                if (shared) source_ids = floor_divide(source_ids, array(routes));
                const auto expected = where(valid, astype(reshape(gather_mm(
                    reshape(astype(input, float32), {-1, 1, width}), dense, source_ids, safe_ids),
                    {tokens, routes, outputs}), float16), zeros({tokens, routes, outputs}, float16));
                for (const auto& weight : {resident, pooled}) for (const auto& plan : plans) {
                    const auto actual = weight.routed_matmul_sorted(input, ids, order, plan);
                    if (!allclose(actual, expected, .004, .000001).item<bool>()
                        && !(columns && column_projection_roundoff(actual, expected)))
                        throw std::runtime_error("TQ shared expansion tile differs from decoded gather GEMM");
                }
            }
        }
}

static void check_routed_preparation() {
    for (int experts : {6, 512}) {
        const int routes = experts == 6 ? 3 : 10, width = 1025, hidden = 17;
        MlxTqExpertWeights weights{
            MlxTqWeight::from_blob(random_blob(experts * hidden, width, 2121), width, experts),
            MlxTqWeight::from_blob(random_blob(experts * hidden, width, 2122), width, experts),
            MlxTqWeight::from_blob(random_blob(experts * 8, hidden, 2123), hidden, experts)};
        MlxKernelPreparation preparation(4096, 4096);
        weights.prepare(routes);
        preparation.finish();
        for (int tokens : {2, 3, 6}) {
            std::vector<int> selected(tokens * routes);
            for (int row = 0; row < int(selected.size()); ++row) selected[row] = row % routes;
            const auto ids = array(selected.begin(), {tokens, routes}, int32);
            const auto plan = make_tq_reuse_plan(selected, tokens, routes, experts);
            if (!plan) throw std::runtime_error("TQ preparation has no repeated route plan");
            preparation.collect([&] {
                const auto value = weights.gate.swiglu(weights.up, zeros({tokens, width}, float16),
                    ids, 0.f, false, &*plan);
                preparation.add(value);
                preparation.add(weights.down.routed_matmul(value, ids, &*plan));
                preparation.add(weights.down.routed_matmul_reduce(value, ids,
                    full({tokens, routes}, 1.f / routes, float32), &*plan));
            });
            if (preparation.size()) throw std::runtime_error("TQ loading misses a repeated-route compilation path: M="
                + std::to_string(tokens) + ", E=" + std::to_string(experts));
        }
        for (int tokens : {154, 192, 257, 258, 461, 513, 576, 1025, 2049, 3277}) {
            preparation.collect([&] {
                const auto ids = zeros({tokens, routes}, int32);
                const auto value = weights.gate.swiglu(weights.up, zeros({tokens, width}, float16), ids);
                preparation.add(value);
                preparation.add(weights.down.routed_matmul(value, ids));
                preparation.add(weights.down.routed_matmul_reduce(value, ids, full({tokens, routes}, 1.f / routes, float32)));
            });
            if (preparation.size()) throw std::runtime_error("TQ loading misses a routed prefill compilation path: M="
                + std::to_string(tokens) + ", E=" + std::to_string(experts));
        }
    }
}

static void check_topk(bool benchmark = false) {
    for (auto dtype : {float16, bfloat16, float32}) for (int rows : {1, 2, 4, 6, 8, 16}) {
        auto base = astype(reshape(sin(arange(rows * 512, float32) * array(.7919f)), {rows, 512}), dtype);
        const auto run = [&](const array& input, bool groups) {
            return moe_topk(input, 10, false, false, true, false,
                std::nullopt, std::nullopt, 1e-20f, 1.f, groups);
        };
        auto grouped = run(base, true), plain = run(base, false);
        check(grouped.ids, plain.ids, 0.f);
        check(grouped.weights, plain.weights, 0.f);
        if (plain.groups || bool(grouped.groups) != (rows > 1))
            throw std::runtime_error("TQ ungrouped router metadata differs");
        if (!benchmark || dtype != float16) continue;
        auto chain = [&](bool groups) {
            auto value = base;
            for (int i = 0; i < 64; ++i) {
                auto result = run(value, groups);
                auto feedback = reshape(slice(reshape(result.weights, {-1}), {0}, {1}), {});
                value = astype(base + feedback * array(.001f), dtype);
            }
            return value;
        };
        chain(false).eval(); chain(true).eval();
        for (int repeat = 0; repeat < 3; ++repeat) for (int phase = 0; phase < 2; ++phase) {
            const bool groups = bool(phase ^ (repeat & 1));
            auto output = chain(groups);
            const auto started = std::chrono::steady_clock::now();
            output.eval();
            std::cout << nlohmann::json{{"kind", "router_metadata_screen"}, {"rows", rows}, {"route_groups", groups},
                {"repeat", repeat}, {"chained_us", std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now()-started).count()/64},
                {"boundary", "64 dependent top-k operations plus identical scalar feedback; one final wait"}} << std::endl;
        }
    }
}

static void check_expert_cache(const MfqContainer& model) {
    const auto resources_before = MlxResourceTelemetry::snapshot();
    MlxTqExpertCache cache(model, {"unit"}, 256 << 10, 4);
    const auto resources = MlxResourceTelemetry::snapshot();
    if (resources.expert_payload_bytes != resources_before.expert_payload_bytes + cache.backing_bytes()
        || resources.dynamic_weight_bytes != resources_before.dynamic_weight_bytes + cache.stats().allocated_bytes)
        throw std::runtime_error("TQ cache resource telemetry differs");
    const auto gate = MlxTqWeight::load(model, "unit.mlp.experts.gate.weight");
    const auto up = MlxTqWeight::load(model, "unit.mlp.experts.up.weight");
    const auto down = MlxTqWeight::load(model, "unit.mlp.experts.down.weight");
    for (int tokens : {1, 2, 6}) for (int first : {0, 0, 2, 4, 6, 0}) {
        auto x = astype(reshape(sin(arange(tokens * 256, float32)), {tokens, 256}), float16);
        auto ids = broadcast_to(array({first, first + 1}, {1, 2}, int32), {tokens, 2});
        auto weights = full({tokens, 2}, .5f, float32);
        auto expected = down.routed_matmul_reduce(gate.swiglu(up, x, ids), ids, weights);
        auto actual = cache.apply(0, ids, [&](const auto& bank, const array* reuse) {
            if (tokens > 1 && !reuse) throw std::runtime_error("TQ streamed reuse plan was not passed to projections");
            return bank.down.routed_matmul_reduce(bank.gate.swiglu(bank.up, x, ids, 0.f, false, reuse),
                ids, weights, reuse);
        });
        check(actual, expected, 0.f);
    }
    cache.clear();
    for (int tokens : {1, 2, 6, 8, 16, 17}) for (int first : {0, 0, 2, 4, 6, 0}) {
        auto x = astype(reshape(sin(arange(tokens * 256, float32)), {tokens, 256}), float16);
        auto ids = broadcast_to(array({first, first + 1}, {1, 2}, int32), {tokens, 2});
        auto weights = full({tokens, 2}, .5f, float32);
        auto expected = down.routed_matmul_reduce(gate.swiglu(up, x, ids), ids, weights);
        int gate_calls = 0, down_calls = 0;
        auto parallel = cos(x);
        auto actual = cache.apply(0, ids, [&](const auto& bank, const array* reuse) {
            ++gate_calls;
            return bank.gate.swiglu(bank.up, x, ids, 0.f, false, reuse);
        }, [&](const auto& bank, const array& hidden, const array* reuse) {
            ++down_calls;
            return bank.down.routed_matmul_reduce(hidden, ids, weights, reuse);
        }, &parallel);
        check(actual, expected, 0.f);
        check(parallel, cos(x), 0.f);
        if (gate_calls != 1 || down_calls != 1)
            throw std::runtime_error("TQ streamed pipeline repeats a projection");
    }
    cache.clear();
    bool stage_failed = false;
    try {
        cache.apply(0, array({0, 1}, {1, 2}, int32), [](const auto&, const array*) -> array {
            throw std::runtime_error("intentional TQ Gate/Up stage failure");
        }, [](const auto&, const array&, const array*) { return array(0); });
    } catch (const std::runtime_error&) { stage_failed = true; }
    if (!stage_failed || cache.stats().cached_experts || cache.stats().cached_bytes)
        throw std::runtime_error("TQ failed streamed stage leaves incomplete experts cached");
    cache.apply(0, array({0, 1}, {1, 2}, int32), [](const auto&) { return array(0); }).eval();
    const auto stats = cache.stats();
    if (!stats.hits || !stats.misses || !stats.evictions || !stats.bytes_read || stats.cached_experts == 0
        || stats.allocated_bytes > (256 << 10) || stats.cached_bytes + stats.metadata_bytes > stats.allocated_bytes
        || stats.read_calls > stats.misses * 3 || !stats.overlapped_read_batches)
        throw std::runtime_error("TQ expert cache accounting differs");
    cache.clear();
    if (cache.stats().cached_experts || cache.stats().cached_bytes)
        throw std::runtime_error("TQ expert cache did not clear");
    bool failed = false;
    try {
        cache.apply(0, array({0, 1}, {1, 2}, int32), [](const auto&) -> array {
            throw std::runtime_error("intentional TQ callback failure");
        });
    } catch (const std::runtime_error&) { failed = true; }
    if (!failed) throw std::runtime_error("TQ cache callback failure was lost");
    {
        auto x = ones({1, 256}, float16);
        auto first_ids = array({0, 1}, {1, 2}, int32);
        auto last_ids = array({6, 7}, {1, 2}, int32);
        const auto operation = [&](const MlxTqExpertWeights& bank, const array& ids) {
            return bank.down.routed_matmul_reduce(bank.gate.swiglu(bank.up, x, ids), ids,
                full({1, 2}, .5f, float32));
        };
        auto first = cache.apply(0, first_ids, [&](const auto& bank) { return operation(bank, first_ids); });
        auto last = cache.apply(0, last_ids, [&](const auto& bank) { return operation(bank, last_ids); });
        cache.clear();
        const MlxTqExpertWeights resident{gate, up, down};
        check(first, operation(resident, first_ids), 0.f);
        check(last, operation(resident, last_ids), 0.f);
    }
    auto ids = array({-1, 8}, {1, 2}, int32);
    auto x = ones({1, 256}, float16);
    {
        auto selected = array({0, 1}, {1, 2}, int32);
        const auto operation = [&](const auto& bank) {
            return bank.down.routed_matmul_reduce(bank.gate.swiglu(bank.up, x, selected), selected,
                full({1, 2}, .5f, float32));
        };
        const MlxTqExpertWeights resident{gate, up, down};
        const auto expected = operation(resident);
        for (auto budget : {2 << 20, 256 << 10, 512 << 10, 256 << 10}) {
            auto pending = cache.apply(0, selected, operation);
            cache.set_limit(budget);
            if (cache.limit_bytes() != std::size_t(budget) || cache.stats().allocated_bytes > std::size_t(budget))
                throw std::runtime_error("TQ resized cache budget differs");
            if (MlxResourceTelemetry::snapshot().dynamic_weight_bytes
                != resources_before.dynamic_weight_bytes + cache.stats().allocated_bytes)
                throw std::runtime_error("TQ resized cache resource telemetry differs");
            check(pending, expected, 0.f);
            check(cache.apply(0, selected, operation), expected, 0.f);
            const auto before = cache.stats();
            if (cache.set_limit(budget) || before.cached_experts != cache.stats().cached_experts)
                throw std::runtime_error("TQ unchanged budget evicts experts");
        }
        const auto before = cache.stats();
        for (auto budget : {std::size_t{1}}) {
            bool rejected = false;
            try { cache.set_limit(budget); } catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected || before.cached_experts != cache.stats().cached_experts
                || before.allocated_bytes != cache.stats().allocated_bytes)
                throw std::runtime_error("TQ invalid resize changes the cache");
        }
        check(cache.apply(0, selected, operation), expected, 0.f);
        cache.set_limit(std::numeric_limits<std::size_t>::max());
        if (cache.stats().allocated_bytes > cache.backing_bytes() * 2)
            throw std::runtime_error("TQ cache allocates beyond the complete model payload");
        check(cache.apply(0, selected, operation), expected, 0.f);
        cache.set_limit(256 << 10);
    }
    check(cache.apply(0, ids, [&](const auto& bank) {
        return bank.down.routed_matmul_reduce(bank.gate.swiglu(bank.up, x, ids), ids,
            full({1, 2}, .5f, float32));
    }), zeros({1, 256}, float16), 0.f);
    check(cache.apply(0, ids, [&](const auto& bank, const array* reuse) {
        return bank.gate.swiglu(bank.up, x, ids, 0.f, false, reuse);
    }, [&](const auto& bank, const array& hidden, const array* reuse) {
        return bank.down.routed_matmul_reduce(hidden, ids, full({1, 2}, .5f, float32), reuse);
    }), zeros({1, 256}, float16), 0.f);
    MlxTqExpertCache prefill(model, {"unit"}, 2 << 20, 4);
    auto prompt = astype(reshape(sin(arange(257 * 256, float32)), {257, 256}), float16);
    auto all_ids = broadcast_to(arange(8, int32), {257, 8});
    check(prefill.apply(0, all_ids, [&](const auto& bank) {
        return bank.down.routed_matmul(bank.gate.swiglu(bank.up, prompt, all_ids), all_ids);
    }), down.routed_matmul(gate.swiglu(up, prompt, all_ids), all_ids), 0.f);
    const auto full_stats = prefill.stats();
    if (full_stats.cached_experts != 8) throw std::runtime_error("TQ full-cache fixture is incomplete");
    {
        MlxKernelPreparation preparation(257, 257);
        prefill.prepare_layer(0, 2);
        if (!preparation.size()) throw std::runtime_error("TQ expert preparation collected no kernels");
        preparation.finish();
        const auto after = prefill.stats();
        if (after.requests != full_stats.requests || after.bytes_read != full_stats.bytes_read
            || after.cached_experts != full_stats.cached_experts)
            throw std::runtime_error("TQ pure preparation executes expert-cache work");
    }
    auto lazy_ids = reshape(add(arange(2, int32), array(2)), {1, 2});
    if (lazy_ids.is_available()) throw std::runtime_error("TQ lazy route fixture was evaluated early");
    const auto resident_projection = [&](const array& selected) {
        return down.routed_matmul_reduce(gate.swiglu(up, x, selected), selected,
            full(selected.shape(), 1.f / selected.shape(1), float32));
    };
    const auto cached_projection = [&](const array& selected) {
        return prefill.apply(0, selected, [&](const auto& bank) {
            return bank.down.routed_matmul_reduce(bank.gate.swiglu(bank.up, x, selected), selected,
                full(selected.shape(), 1.f / selected.shape(1), float32));
        });
    };
    auto lazy_output = cached_projection(lazy_ids);
    if (lazy_ids.is_available()) throw std::runtime_error("TQ fully cached routes still force a CPU wait");
    auto duplicate_ids = array({0, 0, 1, -1, 8}, {1, 5}, int32);
    auto duplicate_output = cached_projection(duplicate_ids);
    const auto ceiling = prefill.limit_bytes();
    if (!prefill.trim(256 << 10) || prefill.limit_bytes() != ceiling
        || prefill.resident_bytes() > (256 << 10))
        throw std::runtime_error("TQ trim restores allocations or changes its configured ceiling");
    check(lazy_output, resident_projection(lazy_ids), 0.f);
    check(duplicate_output, resident_projection(duplicate_ids), 0.f);
    const auto hot_stats = prefill.stats();
    if (hot_stats.requests != full_stats.requests + 4 || hot_stats.hits != full_stats.hits + 4
        || hot_stats.misses != full_stats.misses || !hot_stats.cached_experts)
        throw std::runtime_error("TQ deferred hot-route accounting or resize differs");
    const auto retained_ids = array({0, 1}, {1, 2}, int32);
    check(cached_projection(retained_ids), resident_projection(retained_ids), 0.f);
    if (prefill.stats().bytes_read != hot_stats.bytes_read)
        throw std::runtime_error("TQ budget trim discarded experts inside the retained pool");
    {
        MlxTqExpertCache relocated(model, {"unit"}, 2 << 20, 4);
        const auto hot = array({6, 7}, {1, 2}, int32);
        const auto project = [&](const auto& bank) {
            return bank.down.routed_matmul_reduce(bank.gate.swiglu(bank.up, x, hot), hot,
                full({1, 2}, .5f, float32));
        };
        auto pending = relocated.apply(0, hot, project);
        const auto before = relocated.stats();
        const auto ceiling = relocated.limit_bytes();
        if (!relocated.trim(256 << 10) || relocated.limit_bytes() != ceiling)
            throw std::runtime_error("TQ hot-relocation trim did not reduce capacity");
        check(pending, resident_projection(hot), 0.f);
        check(relocated.apply(0, hot, project), resident_projection(hot), 0.f);
        if (relocated.stats().bytes_read != before.bytes_read || relocated.stats().allocated_bytes > (256 << 10))
            throw std::runtime_error("TQ shrink evicted a hot expert because of its physical address");
        for (const auto selected : {array({0, 1}, {1, 2}, int32), hot}) {
            check(relocated.apply(0, selected, [&](const auto& bank) {
                return bank.down.routed_matmul_reduce(bank.gate.swiglu(bank.up, x, selected), selected,
                    full({1, 2}, .5f, float32));
            }), resident_projection(selected), 0.f);
        }
        for (int cycle = 0; cycle < 8; ++cycle) {
            relocated.set_limit(cycle % 2 ? 2 << 20 : 512 << 10);
            const auto selected = array({cycle % 8, (cycle + 3) % 8}, {1, 2}, int32);
            const auto operation = [&](const auto& bank) {
                return bank.down.routed_matmul_reduce(bank.gate.swiglu(bank.up, x, selected), selected,
                    full({1, 2}, .5f, float32));
            };
            auto output = relocated.apply(0, selected, operation);
            const auto reads = relocated.stats().bytes_read;
            relocated.trim(256 << 10);
            check(output, resident_projection(selected), 0.f);
            check(relocated.apply(0, selected, operation), resident_projection(selected), 0.f);
            if (relocated.stats().bytes_read != reads)
                throw std::runtime_error("TQ repeated budget trims lost the most recent experts at cycle "
                    + std::to_string(cycle) + ": " + std::to_string(reads) + " -> "
                    + std::to_string(relocated.stats().bytes_read));
        }
    }
    std::size_t scan_budget = 0;
    for (const auto* projection : {"gate", "up", "down"})
        scan_budget += 2 * model.record(std::string("unit.mlp.experts.") + projection + ".weight").nbytes;
    MlxTqExpertCache scanned(model, {"unit", "unit", "unit", "unit"}, scan_budget, 4);
    const auto scan_input = full({17, 256}, .1f, float16);
    const auto scan_ids = broadcast_to(arange(8, int32), {17, 8});
    const auto scan_reference = down.routed_matmul(
        gate.swiglu(up, scan_input, scan_ids), scan_ids);
    std::array<std::uint64_t, 2> scan_reads{};
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto before = scanned.stats().bytes_read;
        for (std::size_t layer = 0; layer < 4; ++layer)
            check(scanned.apply(layer, scan_ids, [&](const auto& bank) {
                return bank.down.routed_matmul(bank.gate.swiglu(bank.up, scan_input, scan_ids), scan_ids);
            }), scan_reference, 0.f);
        scan_reads[repeat] = scanned.stats().bytes_read - before;
    }
    if (!scan_reads[1] || scan_reads[1] >= scan_reads[0] || !scanned.stats().evictions
        || scanned.stats().allocated_bytes > scan_budget)
        throw std::runtime_error("TQ prefill scan discarded the entire reusable working set");
    MlxTqExpertCache promoted(model, {"unit", "unit", "unit", "unit"},
        std::numeric_limits<std::size_t>::max(), 4);
    promoted.set_limit(promoted.stats().allocated_bytes * 3 / 4);
    promoted.clear();
    const auto all_experts = reshape(arange(8, int32), {1, 8});
    for (std::size_t layer : {0, 1})
        promoted.apply(layer, all_experts, [](const auto&) { return array(0); }).eval();
    for (std::size_t layer : {0, 3})
        check(promoted.apply(layer, scan_ids, [&](const auto& bank) {
            return bank.down.routed_matmul(bank.gate.swiglu(bank.up, scan_input, scan_ids), scan_ids);
        }), scan_reference, 0.f);
    const auto promoted_stats = promoted.stats();
    check(promoted.apply(0, all_experts, [&](const auto& bank) {
        return bank.gate.routed_matmul(x, all_experts);
    }), gate.routed_matmul(x, all_experts), 0.f);
    if (promoted.stats().bytes_read != promoted_stats.bytes_read)
        throw std::runtime_error("TQ prefill hits did not refresh expert recency");
}

static void check_expert_cache_containers() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() / ("mfq-tq-experts-" + std::to_string(nonce) + ".tq");
    try {
        std::vector<std::vector<std::uint8_t>> blobs;
        for (int projection = 0; projection < 3; ++projection) {
            std::vector<std::vector<std::uint8_t>> children;
            const int outputs = projection == 2 ? 256 : 128, width = projection == 2 ? 128 : 256;
            for (int expert = 0; expert < 8; ++expert) {
                auto child = random_blob(outputs, width, 4001 + projection * 8 + expert, 512, true);
                for (int col = 0; col < 256; ++col) {
                    const float16_t scale(.25f + .125f * expert + .01f * (col % 31));
                    std::memcpy(child.data() + 64 + outputs * 8 + col * 2, &scale, 2);
                }
                children.push_back(std::move(child));
            }
            blobs.push_back(expert_blob(children, outputs, width));
        }
        {
            std::ofstream stream(path, std::ios::binary);
            const auto scalar = [&](auto value) { stream.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
            const auto text = [&](const std::string& value) { scalar(std::uint32_t(value.size())); stream.write(value.data(), value.size()); };
            stream.write("MFQ1", 4); scalar(std::uint32_t{2}); text("test-tq");
            scalar(std::uint32_t{0}); scalar(std::uint32_t{3});
            for (int p = 0; p < 3; ++p) {
                text(std::string("unit.mlp.experts.") + std::array{"gate", "up", "down"}[p] + ".weight");
                text("TQ"); scalar(std::uint64_t(blobs[p].size()));
            }
            for (const auto& blob : blobs) stream.write(reinterpret_cast<const char*>(blob.data()), blob.size());
            if (!stream) throw std::runtime_error("cannot write TQ expert test fixture");
        }
        const MfqContainer fixture(path);
        check_expert_cache(fixture);
    } catch (...) { std::filesystem::remove(path); throw; }
    std::filesystem::remove(path);
}

static void paired(const RoutedBenchmarkCall& weight, const array& input,
        const std::vector<array>& schedule, int width, const std::string& name, const char* implementation) {
    for (int i = 0; i < 8; ++i) weight.routed_matmul(input, schedule[i]).eval();
    for (int repeat = 0; repeat < 3; ++repeat) {
        const auto begin = std::chrono::steady_clock::now();
        for (const auto& ids : schedule) weight.routed_matmul(input, ids).eval();
        std::cout << nlohmann::json{{"name", name}, {"implementation", implementation},
            {"repeat", repeat}, {"ms", std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now()-begin).count()/schedule.size()},
            {"boundary", "native graph+dispatch+GPU completion"}} << std::endl;
    }
    chained(weight, input, schedule, width, name, implementation);
    for (int repeat = 0; repeat < 3; ++repeat) {
        std::vector<array> outputs;
        for (const auto& ids : schedule) outputs.push_back(weight.routed_matmul(input, ids));
        const auto begin = std::chrono::steady_clock::now();
        eval(outputs);
        std::cout << nlohmann::json{{"name", name}, {"implementation", implementation},
            {"queued", true}, {"repeat", repeat}, {"eval_ms", std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now()-begin).count()/schedule.size()}} << std::endl;
    }
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--column-inputs") {
            check_column_inputs(); return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--compact-reuse") {
            check_compact_repeated_routes(); check_device_reuse_plans(); return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--expansion-vectors") {
            check_expansion_vectors(); return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--projection-cache") {
            check_projection_cache(); return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--prefill-reduction") {
            check_prefill_reduction(); return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--streamed-experts") {
            check_expert_cache_containers();
            std::cout << "TQ streamed expert checks passed\n"; return 0;
        }
        if (argc == 3 && std::string(argv[2]) == "--expert-cache") {
            check_expert_cache(MfqContainer(argv[1]));
            std::cout << "TQ expert cache checks passed\n"; return 0;
        }
        if (argc == 3 && std::string(argv[2]) == "--topk") { check_topk(true); return 0; }
        if (argc == 3 && std::string(argv[2]) == "--expert-containers") {
            check_expert_containers(); check_expert_cache_containers(); return 0;
        }
        if (argc == 3 && std::string(argv[2]) == "--repeated-routes") {
            check_repeated_routes(); check_compact_repeated_routes(); check_tiled_repeated_routes();
            check_routed_preparation(); return 0;
        }
        if (argc == 3 && std::string(argv[2]) == "--routed-tiles") {
            check_partial_route_tiles(); check_small_sorted_routes(); check_shared_expansion_tiles(); return 0;
        }
        const bool matrix_bench = argc >= 3 && std::string(argv[argc - 1]) == "--matrix";
        if (matrix_bench) --argc;
        if (argc < 2 || argc > 4) throw std::runtime_error("usage: mfq-metal-tq-test fixture.mfq [qwen-reference.mfq [deepseek-reference.mfq]]");
        MfqContainer model(argv[1]);
        check_expert_containers();
        const bool checks_only = argc == 3 && std::string(argv[2]) == "--checks-only";
        std::unique_ptr<MfqContainer> reference;
        if (argc >= 3 && !checks_only) reference = std::make_unique<MfqContainer>(argv[2]);
        std::unique_ptr<MfqContainer> deepseek_reference;
        if (argc == 4) deepseek_reference = std::make_unique<MfqContainer>(argv[3]);
        check_expert_cache(model);
        check_expert_cache_containers();
        check_topk();
        for (int bit : {35, 36, 37, 38, 39, 56, 63}) {
            auto malformed = model.read("check0");
            std::uint64_t descriptor;
            std::memcpy(&descriptor, malformed.data() + 64, sizeof(descriptor));
            descriptor |= std::uint64_t{1} << bit;
            std::memcpy(malformed.data() + 64, &descriptor, sizeof(descriptor));
            bool projection_rejected = false, row_rejected = false;
            try { MlxTqWeight::from_blob(malformed, 513); }
            catch (const std::runtime_error&) { projection_rejected = true; }
            try {
                MlxMappedTqRows table(malformed, 513);
                MlxTqRowBatch batch;
                table.append_row(0, batch);
            } catch (const std::runtime_error&) { row_rejected = true; }
            if (!projection_rejected || !row_rejected)
                throw std::runtime_error("TQ accepts reserved row descriptor bits");
        }
        {
            std::vector<std::vector<std::uint8_t>> blobs;
            std::vector<array> expected;
            MlxTqRowBatch batch, combined;
            for (const auto name : {"check0", "check1", "check2", "check1"}) blobs.push_back(model.read(name));
            auto different_columns = model.read("check1");
            for (int col = 0; col < 768; ++col) {
                const float16_t scale(.25f + float(col % 11) * .125f);
                std::memcpy(different_columns.data() + 64 + 27 * 8 + col * 2, &scale, 2);
            }
            blobs.push_back(std::move(different_columns));
            blobs.push_back(random_blob(27, 512, 20261022));
            for (const auto& blob : blobs) {
                MlxMappedTqRows table(blob, 512);
                MlxTqRowBatch selected;
                for (int row : {26, 0, 9}) {
                    table.append_row(row, batch);
                    table.append_row(row, selected);
                }
                combined.append_batch(selected);
                expected.push_back(MlxTqWeight::from_blob(blob, 512).embedding(array({26, 0, 9}, int32)));
                if (selected.packed_nbytes() > selected.rows() * table.selected_row_nbytes_bound())
                    throw std::runtime_error("TQ row-store resident bound differs");
            }
            check(batch.decode(), concatenate(expected, 0), 0.f);
            check(combined.decode(), concatenate(expected, 0), 0.f);
        }
        for (const auto name : {"check0", "check1", "check2"}) {
            auto op = MlxTqWeight::load(model, name);
            auto ref = load_dense_array("F16", model.read(std::string(name) + ".reference"));
            check(op.embedding(arange(27, int32)), ref, 0.f);
            auto table_blob = model.read(name);
            MlxMappedTqRows table(table_blob, 513);
            {
                std::vector<std::uint64_t> offsets(3);
                const auto packed_bytes = table.packed_rows(0, 27).bytes;
                const auto pool_words = (packed_bytes / 4 + 256 + 1) / 2 * 2;
                std::vector<std::uint32_t> pool(pool_words, 0);
                std::size_t cursor = 31;
                for (int expert : {2, 0, 1}) {
                    const auto range = table.packed_rows(expert * 9, 9);
                    offsets[expert] = cursor;
                    model.read_range_into(name, range.offset, std::as_writable_bytes(
                        std::span(pool.data() + cursor, range.bytes / 4)));
                    cursor += range.bytes / 4 + 17;
                }
                auto pooled = MlxTqWeight::from_pool(table_blob,
                    array(pool.data(), Shape{2, int(pool_words / 2)}), offsets, 513);
                auto resident = MlxTqWeight::from_blob(table_blob, 513, 3);
                check(pooled.embedding(arange(27, int32)), ref, 0.f);
                for (auto dtype : {float16, bfloat16, float32}) for (int tokens : {1, 2, 6, 17, 257}) {
                    auto x = astype(reshape(sin(arange(tokens * 513, float32)), {tokens, 513}), dtype);
                    auto ids = broadcast_to(array({0, 2, -1, 3}, {1, 4}, int32), {tokens, 4});
                    check(pooled.routed_matmul(x, ids), resident.routed_matmul(x, ids), 0.f);
                    check(pooled.swiglu(pooled, x, ids), resident.swiglu(resident, x, ids), 0.f);
                    check(pooled.swiglu(resident, x, ids), resident.swiglu(resident, x, ids), 0.f);
                    auto weights = ones({tokens, 4}, float32) * array(.25f);
                    check(pooled.routed_matmul_reduce(x, ids, weights), resident.routed_matmul_reduce(x, ids, weights), 0.f);
                    auto view = pooled.expert_view(1), resident_view = resident.expert_view(1);
                    check(view.matmul(x), resident_view.matmul(x), 0.f);
                    const auto grouped = MlxTqWeight::matmul_many({view, resident_view}, x);
                    check(grouped[0], grouped[1], 0.f);
                    auto streams = astype(reshape(cos(arange(tokens * 9, float32)), {tokens, 9}), dtype);
                    check(view.gated_mix(x, streams, 3), resident_view.gated_mix(x, streams, 3), 0.f);
                    auto split = reshape(broadcast_to(expand_dims(x, 1), {tokens, 3, 513}), {tokens, 3, 513});
                    check(view.grouped_row_matmul(split, 3), resident_view.grouped_row_matmul(split, 3), 0.f);
                }
                bool invalid_pool = false;
                offsets[0] = pool_words;
                try { MlxTqWeight::from_pool(table_blob, array(pool.data(), Shape{int(pool_words)}), offsets, 513); }
                catch (const std::out_of_range&) { invalid_pool = true; }
                if (!invalid_pool) throw std::runtime_error("TQ pool bounds were not checked");
            }
            const auto row_span = table.packed_rows(3, 8);
            const auto read_span = model.read_range(name, row_span.offset, row_span.bytes);
            if (!std::equal(read_span.begin(), read_span.end(), table_blob.begin() + row_span.offset))
                throw std::runtime_error("TQ packed row span differs from direct source read");
            std::vector<std::uint64_t> rebased(8);
            table.write_descriptors(3, 17, rebased);
            std::uint32_t next_word = 17;
            for (const auto descriptor : rebased) {
                if (std::uint32_t(descriptor) != next_word)
                    throw std::runtime_error("TQ packed row arena offsets differ");
                next_word += 768 * (((descriptor >> 32) & 7) + 1) / 32;
            }
            if ((next_word - 17) * 4 != row_span.bytes)
                throw std::runtime_error("TQ packed row arena span differs");
            MlxTqRowBatch selected, first, last;
            for (int row : {26, 0, 9, 9, 1}) table.append_row(row, selected);
            table.append_row(26, first);
            for (int row : {0, 9, 9, 1}) table.append_row(row, last);
            first.append_batch(last);
            auto selected_ref = take(ref, array({26, 0, 9, 9, 1}, int32), 0);
            check(selected.decode(), selected_ref, 0.f);
            check(first.decode(), selected_ref, 0.f);
            for (std::size_t capacity : {0, 2, 32}) {
                MlxQuantizedRowStore rows(model, name, 27, 513, capacity, 4);
                const std::vector<std::int64_t> ids{26, 0, 9, 9, 1};
                check(rows.gather(ids), selected_ref, 0.f);
                check(rows.gather(ids), selected_ref, 0.f);
                if (rows.cached_rows() > capacity || rows.stats().rows_loaded == 0)
                    throw std::runtime_error("TQ row-store cache accounting differs");
                rows.clear();
                if (rows.cached_rows() != 0) throw std::runtime_error("TQ row-store clear failed");
            }
            bool invalid_row_rejected = false;
            try { table.append_row(27, selected); }
            catch (const std::out_of_range&) { invalid_row_rejected = true; }
            if (!invalid_row_rejected) throw std::runtime_error("TQ row bounds were not checked");
            auto wide = MlxTqWeight::concatenate_projections(std::vector<MlxTqWeight>(48, op));
            auto wide_ref = concatenate(std::vector<array>(48, ref), 0);
            for (auto dtype : {float16, bfloat16, float32}) for (int tokens : {1, 2, 3, 4, 5, 6, 257}) {
                auto x = astype(reshape(sin(arange(tokens * 513, float32)), {tokens, 513}), dtype);
                auto expected = matmul(astype(x, float32), transpose(astype(wide_ref, float32)));
                check(wide.matmul(x), expected, dtype == bfloat16 ? .02f : .003f);
                const auto grouped = MlxTqWeight::matmul_many({wide, op}, x);
                check(grouped[0], expected, dtype == bfloat16 ? .02f : .003f);
                check(grouped[1], matmul(astype(x, float32), transpose(astype(ref, float32))),
                    dtype == bfloat16 ? .02f : .003f);
                if (tokens <= 6) {
                    auto streams = astype(reshape(cos(arange(tokens * 1296, float32)), {tokens, 1296}), dtype);
                    auto gated = sigmoid(wide.matmul(x)) * streams;
                    for (int groups : {1, 3, 4})
                        check(wide.gated_mix(x, streams, groups),
                            mean(reshape(gated, {tokens, groups, 1296 / groups}), 1),
                            dtype == bfloat16 ? .008f : .001f);
                }
                for (const float limit : {0.f, .25f}) for (const bool gelu : {false, true}) {
                    auto gate = astype(wide.matmul(x), float32), up = gate;
                    if (limit > 0.f) { gate = minimum(gate, array(limit)); up = clip(up, array(-limit), array(limit)); }
                    auto activated = gelu ? array(.5f) * gate * (array(1.f) + tanh(array(.7978845608028654f)
                        * gate * (array(1.f) + array(.044715f) * gate * gate))) : gate * sigmoid(gate);
                    check(reshape(wide.swiglu(wide, x, zeros({tokens, 1}, int32), limit, gelu), {tokens, 1296}),
                        astype(activated * up, dtype), dtype == bfloat16 ? .01f : .005f);
                }
                {
                    auto ids = reshape(arange(tokens, int32) % array(3, int32) - array(1, int32), {tokens, 1});
                    auto projected = wide.swiglu(wide, x, zeros({tokens, 1}, int32));
                    check(wide.swiglu(wide, x, ids),
                        where(expand_dims(equal(ids, array(0, int32)), -1), projected, zeros_like(projected)), 0.f);
                }
            }
            auto rotated = MlxTqWeight::from_blob(rotate_rows(model.read(name)), 513);
            auto rotated_ref = take(ref, remainder(arange(7, 34, int32), array(27)), 0);
            auto combined = MlxTqWeight::concatenate_projections({op, rotated});
            auto experts = MlxTqWeight::concatenate_experts({op, rotated, op});
            auto experts_up = MlxTqWeight::concatenate_experts({rotated, op, rotated});
            auto experts_pair = MlxTqWeight::concatenate_projections({experts, experts_up});
            {
                auto expanded = MlxTqWeight::concatenate_experts({wide, wide, wide});
                auto x = astype(reshape(sin(arange(257 * 513, float32)), {257, 513}), float16);
                auto ids = broadcast_to(array({0, 2, -1, 3}, {1, 4}, int32), {257, 4});
                auto expected = wide.matmul(x);
                check(expanded.routed_matmul(x, ids),
                    stack({expected, expected, zeros_like(expected), zeros_like(expected)}, 1), .003f);
                auto independent = reshape(broadcast_to(expand_dims(x, 1), {257, 4, 513}), {257 * 4, 513});
                check(expanded.routed_matmul(independent, ids),
                    stack({expected, expected, zeros_like(expected), zeros_like(expected)}, 1), .003f);
            }
            check(experts.expert_view(1).embedding(arange(27, int32)), rotated_ref, 0.f);
            auto routed = array({0, 1, 2, -1, 3}, {1, 5}, int32);
            check(combined.embedding(arange(54, int32)), concatenate({ref, rotated_ref}, 0), 0.f);
            for (int tokens : {7, 16, 17, 32, 65}) {
                auto x = astype(reshape(sin(arange(tokens * 513, float32)), {tokens, 513}), float16);
                check(op.matmul(x), matmul(astype(x, float32), transpose(astype(ref, float32))), .003f);
            }
            for (int tokens : {7, 17, 65}) {
                auto x = astype(reshape(sin(arange(tokens * 513, float32)), {tokens, 513}), float16);
                auto ids = broadcast_to(routed, {tokens, 5});
                check(experts.routed_matmul(x, ids), astype(experts.routed_matmul(astype(x, float32), ids), float16), .003f);
                auto individual = reshape(broadcast_to(expand_dims(x, 1), {tokens, 5, 513}), {tokens * 5, 513});
                check(experts.routed_matmul(individual, ids), astype(experts.routed_matmul(astype(individual, float32), ids), float16), .003f);
                for (bool gelu : {false, true}) for (float limit : {0.f, .2f})
                    check(experts.swiglu(experts_up, x, ids, limit, gelu),
                        experts_pair.swiglu_split(x, ids, limit, gelu), .001f);
            }
            for (auto dtype : {float16, bfloat16, float32}) {
                auto x = astype(reshape(sin(arange(513, float32)), {1, 513}), dtype);
                auto expected = matmul(astype(x, float32), transpose(astype(ref, float32)));
                check(op.matmul(x), expected, dtype == bfloat16 ? .02f : .002f);
                auto up_expected = matmul(astype(x, float32), transpose(astype(rotated_ref, float32)));
                check(combined.matmul(x), concatenate({expected, up_expected}, -1), dtype == bfloat16 ? .02f : .002f);
                check(experts.routed_matmul(x, routed), stack({astype(expected, dtype), astype(up_expected, dtype),
                    astype(expected, dtype), zeros_like(expected), zeros_like(expected)}, 1), .002f);
                auto weights = array({.2f, .1f, .3f, .1f, .3f}, {1, 5});
                const auto reduced_reference = [&](const array& values) {
                    auto pairs = experts.routed_matmul(values, routed);
                    if (dtype == bfloat16) pairs = astype(pairs, float32);
                    return astype(moe_weighted_reduce(pairs, weights), dtype);
                };
                for (int route = 0; route < 5; ++route) {
                    auto one_hot = astype(equal(arange(5, int32), array(route)), float32);
                    try {
                        check(experts.routed_matmul_reduce(x, routed, reshape(one_hot, {1, 5})),
                            reshape(slice(experts.routed_matmul(x, routed), {0, route, 0}, {1, route + 1, 27}), {1, 27}), 0.f);
                    } catch (const std::exception& error) {
                        throw std::runtime_error(std::string(name) + " route=" + std::to_string(route)
                            + (dtype == float16 ? " F16: " : dtype == bfloat16 ? " BF16: " : " F32: ") + error.what());
                    }
                }
                try {
                    check(experts.routed_matmul_reduce(x, routed, weights),
                        reduced_reference(x), 0.f);
                } catch (const std::exception& error) {
                    throw std::runtime_error(std::string(name) + " shared " + (dtype == float16 ? "F16" : dtype == bfloat16 ? "BF16" : "F32") + ": " + error.what());
                }
                auto routed_x = multiply(broadcast_to(reshape(x, {1, 1, 513}), {1, 5, 513}),
                    astype(reshape(arange(1, 6, float32), {1, 5, 1}), dtype));
                try {
                    check(experts.routed_matmul_reduce(routed_x, routed, weights),
                        reduced_reference(routed_x), 0.f);
                } catch (const std::exception& error) {
                    throw std::runtime_error(std::string(name) + " routed " + (dtype == float16 ? "F16" : dtype == bfloat16 ? "BF16" : "F32") + ": " + error.what());
                }
                for (bool gelu : {false, true}) {
                    auto ids = array({0, -1}, {1, 2}, int32);
                    check(combined.swiglu_split(x, ids, .2f, gelu), op.swiglu(rotated, x, ids, .2f, gelu), 0.f);
                }
            }
            auto bad = model.read(name); bad[32] = 1;
            bool rejected = false;
            try { MlxTqWeight::from_blob(bad); } catch (const std::runtime_error&) { rejected = true; }
            if (!rejected) throw std::runtime_error("TQ invalid header was accepted");
        }
        auto a = MlxTqWeight::load(model, "check0"), b = MlxTqWeight::load(model, "check1");
        auto c = MlxTqWeight::load(model, "check2");
        auto wide = MlxTqWeight::concatenate_projections({c, c});
        if (a.can_concatenate_rows(b) || a.can_concatenate_rows(c))
            throw std::runtime_error("TQ incompatible scale layouts were accepted");
        std::vector<MlxLinear> linears{MlxLinear(a), MlxLinear(b), MlxLinear(wide)};
        auto batch = MlxProjectionBatch({&linears[0], &linears[1], &linears[2]});
        if (batch.grouped_projection_count() != 3 || !batch.projections_share_group(0, 3))
            throw std::runtime_error("TQ independent projections were not grouped");
        for (auto dtype : {float16, bfloat16, float32}) for (int tokens : {1, 2, 6, 7, 17, 65}) {
            auto x = astype(reshape(sin(arange(tokens * 513, float32)), {1, tokens, 513}), dtype);
            auto actual = batch(x);
            for (std::size_t p = 0; p < linears.size(); ++p) check(actual[p], linears[p](x), 0.f);
        }
        check_gated_residual();
        check_gated_mix_dispatch();
        check_shared_routes();
        check_wide_routes();
        check_small_outputs();
        check_dense_dispatch();
        check_dense_pair_dispatch();
        check_dispatch_preparation();
        check_repeated_routes();
        check_compact_repeated_routes();
        check_tiled_repeated_routes();
        check_partial_route_tiles();
        check_large_row_offsets();
        check_unaligned_inputs();
        check_grouped_projections();
        check_normalized_projections();
        check_normalized_prefill();
        check_short_expanding_prefill();
        check_small_sorted_routes();
        check_shared_expansion_tiles();
        check_routed_preparation();
        std::cout << "TQ native numerical checks passed\n";
        DeepseekV41Config config;
        config.hidden = 256; config.moe_inter = 128; config.n_experts = 8;
        config.top_k = 6; config.norm_topk_prob = true; config.routed_scaling = 1.5;
        config.swiglu_limit = 10;
        auto moe = MlxDeepseekV41Moe::load(model, config, "unit.mlp");
        auto mixed = moe.forward(reshape(astype(sin(arange(256, float32)), float16), {1, 1, 256}));
        if (!all(isfinite(mixed.routed)).item<bool>() || !all(isfinite(mixed.shared)).item<bool>())
            throw std::runtime_error("TQ native MoE returned nonfinite output");
        auto cache = std::make_shared<MlxTqExpertCache>(model, std::vector<std::string>{"unit"}, 2 << 20, 4);
        auto streamed = MlxDeepseekV41Moe::load(model, config, "unit.mlp", false, nullptr, nullptr, 0, cache);
        for (auto dtype : {float16, bfloat16, float32}) for (int tokens : {1, 2, 6, 17, 257}) {
            auto x = reshape(astype(sin(arange(tokens * 256, float32)), dtype), {1, tokens, 256});
            auto expected = moe.forward(x);
            auto actual = streamed.forward(x);
            check(actual.routed, expected.routed, 0.f);
            check(actual.shared, expected.shared, 0.f);
            cache->clear();
        }
        auto conversion_input = reshape(astype(sin(arange(256, float32)), float16), {1, 1, 256});
        auto before_conversion = moe.forward(conversion_input);
        eval({before_conversion.routed, before_conversion.shared});
        moe.use_tq_offload(cache);
        if (moe.resident_routed_bytes()) throw std::runtime_error("TQ resident experts were not released");
        auto converted = moe.forward(conversion_input);
        check(converted.routed, before_conversion.routed, 0.f);
        check(converted.shared, before_conversion.shared, 0.f);
        std::cout << "TQ native DeepSeek MoE smoke passed\n";
        if (checks_only) return 0;
        if (matrix_bench) {
            for (const auto architecture : {"qwen", "deepseek"}) {
                const bool qwen = std::string(architecture) == "qwen";
                const auto* baseline = qwen ? reference.get() : deepseek_reference.get();
                if (!baseline) continue;
                const int topk = qwen ? 10 : 6;
                for (const auto projection : {"gate", "down", "gate_up"}) {
                    const bool pair = std::string(projection) == "gate_up";
                    const bool down = std::string(projection) == "down";
                    const auto key = std::string(architecture) + "." + (pair ? "gate" : projection);
                    auto op = MlxTqWeight::load(model, key);
                    auto up = pair ? std::optional<MlxTqWeight>(MlxTqWeight::load(model, std::string(architecture) + ".up")) : std::nullopt;
                    std::vector<std::int32_t> experts(op.experts());
                    std::iota(experts.begin(), experts.end(), 0);
                    auto old = MlxMfeWeight::from_blob(baseline->read("model.block.0.mlp.experts."
                        + std::string(pair ? "gate" : projection) + ".weight")).select_experts(experts);
                    if (pair) {
                        auto old_up = MlxMfeWeight::from_blob(baseline->read("model.block.0.mlp.experts.up.weight")).select_experts(experts);
                        old = MlxMfeWeight::concatenate_projections({old, old_up}).materialize_packed_projections();
                    }
                    RoutedBenchmarkCall current{[&](const array& x, const array& ids) {
                        return pair ? op.swiglu(*up, x, ids) : op.routed_matmul(x, ids);
                    }};
                    RoutedBenchmarkCall legacy{[&](const array& x, const array& ids) {
                        return pair ? old.routed_swiglu(x, ids) : old.routed_matmul(x, ids);
                    }};
                    for (int tokens : {1, 2, 4, 6, 8, 16, 32, 256, 1024}) {
                        const int steps = tokens <= 16 ? 64 : tokens <= 256 ? 8 : 2;
                        std::mt19937 rng(20261022);
                        std::vector<array> schedule;
                        for (int step = 0; step < steps; ++step) {
                            std::vector<int> ids;
                            for (int token = 0; token < tokens; ++token) {
                                std::vector<int> selected;
                                while (int(selected.size()) < topk) {
                                    int expert = rng() % op.experts();
                                    if (std::find(selected.begin(), selected.end(), expert) == selected.end()) selected.push_back(expert);
                                }
                                ids.insert(ids.end(), selected.begin(), selected.end());
                            }
                            schedule.emplace_back(ids.begin(), Shape{tokens, topk}, int32);
                            schedule.back().eval();
                        }
                        auto shape = down ? Shape{tokens, topk, op.input_size()} : Shape{tokens, op.input_size()};
                        auto x = reshape(astype(sin(arange(tokens * (down ? topk : 1) * op.input_size(), float32)), float16), shape);
                        x.eval();
                        for (const auto& ids : schedule) {
                            legacy.routed_matmul(x, ids).eval();
                            current.routed_matmul(x, ids).eval();
                        }
                        auto name = std::string(architecture) + "." + projection + ".M" + std::to_string(tokens);
                        for (int repeat = 0; repeat < 3; ++repeat) {
                            if (repeat % 2 == 0) {
                                chained(legacy, x, schedule, op.input_size(), name, "legacy", 1, repeat);
                                chained(current, x, schedule, op.input_size(), name, "TQ", 1, repeat);
                            } else {
                                chained(current, x, schedule, op.input_size(), name, "TQ", 1, repeat);
                                chained(legacy, x, schedule, op.input_size(), name, "legacy", 1, repeat);
                            }
                        }
                    }
                }
            }
            return 0;
        }
        auto byte_counts = nlohmann::json::parse(model.header().extra_json.at("benchmark_expert_bytes"));
        for (const auto architecture : {"qwen", "deepseek"}) {
            for (const auto projection : {"gate", "up", "down"}) {
                std::string name = std::string(architecture) + "." + projection;
                auto op = MlxTqWeight::load(model, name);
                int topk = std::string(architecture) == "qwen" ? 10 : 6;
                int inputs = std::string(projection) == "down" ? topk : 1;
                auto x = reshape(astype(sin(arange(inputs * op.input_size(), float32)), float16),
                    {inputs, op.input_size()});
                x.eval();
                std::vector<array> schedule;
                double selected_bytes = 0;
                std::mt19937 rng(20261022);
                for (int step = 0; step < 64; ++step) {
                    std::vector<int> ids;
                    while (int(ids.size()) < topk) {
                        int id = rng() % op.experts();
                        if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
                    }
                    schedule.emplace_back(ids.begin(), Shape{1, topk}, int32);
                    for (int id : ids) selected_bytes += byte_counts.at(name).at(id).get<double>();
                    schedule.back().eval();
                }
                for (int i = 0; i < 8; ++i) op.routed_matmul(x, schedule[i]).eval();
                const auto* baseline = std::string(architecture) == "qwen" ? reference.get() : deepseek_reference.get();
                const char* baseline_name = std::string(architecture) == "qwen" ? "V3-XXS legacy" : "Q3 legacy";
                if (baseline) {
                    std::vector<std::int32_t> selected(op.experts());
                    std::iota(selected.begin(), selected.end(), 0);
                    auto old = MlxMfeWeight::from_blob(baseline->read(
                        "model.block.0.mlp.experts." + std::string(projection) + ".weight")).select_experts(selected);
                    auto old_x = inputs == topk ? reshape(x, {1, topk, op.input_size()}) : x;
                    for (int i = 0; i < 8; ++i) old.routed_matmul(old_x, schedule[i]).eval();
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        const auto start = std::chrono::steady_clock::now();
                        for (auto& ids : schedule) old.routed_matmul(old_x, ids).eval();
                        std::cout << nlohmann::json{{"name", name}, {"baseline", baseline_name},
                            {"repeat", repeat}, {"ms", std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start).count() / schedule.size()},
                            {"boundary", "native graph+dispatch+GPU completion"}} << std::endl;
                    }
                    chained(old, old_x, schedule, op.input_size(), name, baseline_name);
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        std::vector<array> results;
                        for (auto& ids : schedule) results.push_back(old.routed_matmul(old_x, ids));
                        const auto start = std::chrono::steady_clock::now();
                        eval(results);
                        std::cout << nlohmann::json{{"name", name}, {"baseline", baseline_name},
                            {"queued", true}, {"repeat", repeat}, {"eval_ms", std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start).count() / schedule.size()}} << std::endl;
                    }
                    if (std::string(projection) == "gate") {
                        auto old_up = MlxMfeWeight::from_blob(baseline->read(
                            "model.block.0.mlp.experts.up.weight")).select_experts(selected);
                        auto pair = MlxMfeWeight::concatenate_projections({old, old_up}).materialize_packed_projections();
                        paired(RoutedBenchmarkCall{[&](const array& values, const array& ids) {
                            return pair.routed_swiglu(values, ids);
                        }}, x, schedule, op.input_size(), std::string(architecture)+".gate_up", baseline_name);
                    }
                }
                for (int repeat = 0; repeat < 3; ++repeat) {
                    const auto start = std::chrono::steady_clock::now();
                    double build_ms = 0;
                    for (auto& ids : schedule) {
                        const auto begin = std::chrono::steady_clock::now();
                        auto result = op.routed_matmul(x, ids);
                        build_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
                        result.eval();
                    }
                    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / schedule.size();
                    const double bytes = selected_bytes / schedule.size();
                    std::cout << nlohmann::json{{"name", name}, {"ms", ms}, {"gb_s", bytes / ms / 1e6},
                        {"repeat", repeat}, {"bytes", bytes}, {"mtp", false}, {"build_ms", build_ms / schedule.size()},
                        {"boundary", "native graph+dispatch+GPU completion"},
                        {"bytes_definition", "unique packed words+descriptors+scales touched; not hardware traffic"}} << '\n';
                }
                chained(op, x, schedule, op.input_size(), name, "TQ");
                if (std::string(projection) == "down") {
                    auto weights = full({1, topk}, 1.f / topk, float32); weights.eval();
                    for (const auto& ids : schedule)
                        check(op.routed_matmul_reduce(x, ids, weights),
                            moe_weighted_reduce(op.routed_matmul(x, ids), weights), 0.f);
                    for (bool fused : {false, true}) {
                        RoutedBenchmarkCall call{[&](const array& values, const array& ids) {
                            auto output = fused ? op.routed_matmul_reduce(values, ids, weights)
                                : moe_weighted_reduce(op.routed_matmul(values, ids), weights);
                            return broadcast_to(reshape(output, {1, 1, op.output_size()}), {1, topk, op.output_size()});
                        }};
                        paired(call, x, schedule, op.input_size(), std::string(architecture)+".down_reduce",
                            fused ? "TQ selected reduction" : "TQ separate reduction");
                    }
                }
                for (int repeat = 0; repeat < 3; ++repeat) {
                    const auto begin = std::chrono::steady_clock::now();
                    std::vector<array> queued;
                    for (auto& ids : schedule) queued.push_back(op.routed_matmul(x, ids));
                    const auto built = std::chrono::steady_clock::now();
                    eval(queued);
                    const auto end = std::chrono::steady_clock::now();
                    std::cout << nlohmann::json{{"name", name}, {"queued", true}, {"repeat", repeat},
                        {"build_ms", std::chrono::duration<double, std::milli>(built - begin).count() / schedule.size()},
                        {"eval_ms", std::chrono::duration<double, std::milli>(end - built).count() / schedule.size()}} << '\n';
                }
                if (std::string(projection) == "gate") {
                    auto up = MlxTqWeight::load(model, std::string(architecture)+".up");
                    paired(RoutedBenchmarkCall{[&](const array& values, const array& ids) {
                        return op.swiglu(up, values, ids);
                    }}, x, schedule, op.input_size(), std::string(architecture)+".gate_up", "TQ separate banks");
                    auto combined = MlxTqWeight::concatenate_projections({op, up});
                    for (const auto& ids : schedule)
                        check(combined.swiglu_split(x, ids), op.swiglu(up, x, ids), 0.f);
                    paired(RoutedBenchmarkCall{[&](const array& values, const array& ids) {
                        return combined.swiglu_split(values, ids);
                    }}, x, schedule, op.input_size(), std::string(architecture)+".gate_up", "TQ unified bank");
                    std::vector<MlxLinear> projections{MlxLinear(op.expert_view(0)), MlxLinear(up.expert_view(1)),
                        MlxLinear(combined.expert_view(2))};
                    MlxProjectionBatch batch({&projections[0], &projections[1], &projections[2]});
                    auto batched = batch(x);
                    for (std::size_t p = 0; p < projections.size(); ++p) check(batched[p], projections[p](x), 0.f);
                    for (bool fused : {false, true}) {
                        RoutedBenchmarkCall call{[&](const array& values, const array&) {
                            auto results = fused ? batch(values) : std::vector<array>{
                                projections[0](values), projections[1](values), projections[2](values)};
                            return concatenate(results, -1);
                        }};
                        auto ids = zeros({1, 1}, int32); ids.eval();
                        std::vector<array> dense_schedule(64, ids);
                        paired(call, x, dense_schedule, op.input_size(), std::string(architecture)+".projection_batch",
                            fused ? "TQ unified rows" : "TQ independent launches");
                    }
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
