#include "mlx_moe_ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <numbers>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/fast_primitives.h>

namespace {

using mlx::core::Shape;
using mlx::core::array;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    float actual,
    float expected,
    float tolerance = 3e-5f) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "MoE value mismatch: actual=" +
            std::to_string(actual) +
            " expected=" +
            std::to_string(expected));
    }
}

std::vector<float> floats(array value) {
    value = mlx::core::astype(value, mlx::core::float32);
    value.eval();
    return {
        value.data<float>(),
        value.data<float>() + value.size(),
    };
}

std::vector<std::int32_t> integers(array value) {
    value = mlx::core::astype(value, mlx::core::int32);
    value.eval();
    return {
        value.data<std::int32_t>(),
        value.data<std::int32_t>() + value.size(),
    };
}

void test_runtime_row_kernel_reuse() {
    std::map<std::string, std::pair<std::string, std::string>> signatures;
    const auto check = [&](const std::string& label, array value) {
        const auto& primitive = value.primitive();
        require(std::string(primitive.name()) == "CustomKernel",
            label + " must use a custom kernel");
        const auto state = static_cast<const mlx::core::fast::CustomKernel&>(
            primitive).state();
        const auto signature = std::pair{std::get<0>(state), std::get<1>(state)};
        const auto [entry, inserted] = signatures.emplace(label, signature);
        if (!inserted && entry->second != signature) {
            const auto mismatch = std::mismatch(entry->second.second.begin(),
                entry->second.second.end(), signature.second.begin(), signature.second.end());
            const auto offset = static_cast<std::size_t>(
                mismatch.first - entry->second.second.begin());
            throw std::runtime_error(label + " recompiles for a different row count: "
                + entry->second.first + " vs " + signature.first + " source: "
                + entry->second.second.substr(offset, 100) + " vs "
                + signature.second.substr(std::min(offset, signature.second.size()), 100));
        }
        value.eval();
    };
    constexpr int experts = 32;
    constexpr int routes = 6;
    constexpr int width = 64;
    for (const int tokens : {7, 25, 30, 31, 45, 53, 54, 63, 64, 65, 97, 129}) {
        const auto logits = mlx::core::zeros(Shape{tokens, experts});
        const auto selected = mfq::metal::moe_topk(logits, routes);
        check("topk ids", selected.ids);
        const auto weights = mfq::metal::moe_sqrtsoftplus_weights(
            logits, selected.ids);
        check("sqrt-softplus", weights);
        const auto available = mlx::core::ones(Shape{experts}, mlx::core::bool_);
        check("repair ids", mfq::metal::moe_repair_hash_ids(
            selected.ids, selected.ids, available));
        const auto pairs = mlx::core::full(Shape{tokens, routes, width}, 0.25f);
        const auto shared = mlx::core::full(Shape{tokens, width}, 0.5f);
        const auto gates = mlx::core::zeros(Shape{tokens});
        const std::string gate_binding = tokens <= 16 ? " inline gates" : "";
        const auto reduced = mfq::metal::moe_weighted_reduce(pairs, weights);
        check("reduce", reduced);
        check("shared gate" + gate_binding, mfq::metal::moe_add_shared_gate(
            reduced, shared, gates));
        const auto fused = mfq::metal::moe_weighted_reduce_shared_gate(
            pairs, weights, shared, gates);
        check("reduce shared gate" + gate_binding, fused);
        for (const float value : floats(fused)) {
            require_close(value, 0.5f);
        }
        const auto order = mlx::core::arange(tokens * routes, mlx::core::int32);
        const auto inverse = mfq::metal::moe_inverse_permutation(order);
        check("inverse permutation", inverse);
        const auto sorted_pairs = mlx::core::reshape(
            pairs, Shape{tokens * routes, width});
        auto restored = mfq::metal::moe_restore_routed_rows(sorted_pairs,order,selected.ids);
        check("restore routed rows", restored.inputs().front());
        check("sorted reduce", mfq::metal::moe_weighted_reduce_sorted(
            sorted_pairs, inverse, weights));
        check("sorted reduce shared gate" + gate_binding,
            mfq::metal::moe_weighted_reduce_shared_gate_sorted(
                sorted_pairs, inverse, weights, shared, gates));
        const auto gate_up = mlx::core::zeros(Shape{tokens, 2, 2 * width});
        const auto split = mfq::metal::moe_swiglu_split(gate_up);
        check("split swiglu", split);
        require(split.shape() == Shape{tokens, 2, width},
            "runtime GLU must preserve leading dimensions");
        check("split geglu", mfq::metal::moe_geglu_split(gate_up));
        const auto gate = mlx::core::zeros(Shape{tokens, 2, width});
        check("paired swiglu", mfq::metal::moe_swiglu_pair(gate, gate));
        check("expert scale", mfq::metal::moe_apply_expert_scale(
            weights, selected.ids, mlx::core::ones(Shape{experts})));
    }
}

std::vector<int> stable_top_k(
    const std::vector<float>& scores,
    int top_k,
    const std::vector<bool>& available = {}) {
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(
        order.begin(),
        order.end(),
        [&](int left, int right) {
            const auto left_value =
                !available.empty() && !available[left]
                ? -std::numeric_limits<float>::infinity()
                : scores[left];
            const auto right_value =
                !available.empty() && !available[right]
                ? -std::numeric_limits<float>::infinity()
                : scores[right];
            if (left_value != right_value) {
                return left_value > right_value;
            }
            return left < right;
        });
    order.resize(static_cast<std::size_t>(top_k));
    return order;
}

void test_router_modes() {
    constexpr int rows = 2;
    constexpr int experts = 8;
    constexpr int top_k = 3;
    const std::vector<float> logits{
        0.5f, -1.0f, 2.0f, 0.25f, 1.5f, -0.4f, 0.9f, 0.1f,
        -0.2f, 0.7f, 0.3f, 1.2f, -1.5f, 2.1f, 0.0f, 0.8f,
    };
    auto result = mfq::metal::moe_topk(
        array(logits.begin(), Shape{rows, experts}),
        top_k);
    const auto ids = integers(result.ids);
    const auto weights = floats(result.weights);

    for (int row = 0; row < rows; ++row) {
        const auto begin =
            logits.begin() + row * experts;
        std::vector<float> probabilities(begin, begin + experts);
        const float maximum =
            *std::max_element(
                probabilities.begin(),
                probabilities.end());
        float total = 0.0f;
        for (auto& value : probabilities) {
            value = std::exp(value - maximum);
            total += value;
        }
        for (auto& value : probabilities) {
            value /= total;
        }
        const auto expected =
            stable_top_k(probabilities, top_k);
        for (int rank = 0; rank < top_k; ++rank) {
            const int index = row * top_k + rank;
            require(
                ids[index] == expected[rank],
                "softmax router selected wrong expert");
            require_close(
                weights[index],
                probabilities[expected[rank]]);
        }
    }

    const std::vector<float> bias{
        0.0f, 0.01f, -0.03f, 0.02f,
        0.04f, -0.02f, 0.01f, 0.0f,
    };
    const std::vector<std::uint8_t> available{
        1, 1, 0, 1, 1, 1, 1, 1,
    };
    auto sigmoid = mfq::metal::moe_topk(
        array(logits.begin(), Shape{rows, experts}),
        top_k,
        true,
        false,
        true,
        false,
        array(bias.begin(), Shape{experts}),
        array(
            available.data(),
            Shape{experts},
            mlx::core::uint8),
        1e-20f,
        1.5f);
    const auto sigmoid_ids = integers(sigmoid.ids);
    const auto sigmoid_weights = floats(sigmoid.weights);
    for (int row = 0; row < rows; ++row) {
        std::vector<float> transformed(experts);
        std::vector<float> scores(experts);
        std::vector<bool> mask(experts);
        for (int expert = 0; expert < experts; ++expert) {
            transformed[expert] =
                1.0f /
                (1.0f +
                 std::exp(-logits[row * experts + expert]));
            scores[expert] = transformed[expert] + bias[expert];
            mask[expert] = available[expert] != 0;
        }
        const auto expected =
            stable_top_k(scores, top_k, mask);
        float selected_total = 0.0f;
        for (const auto expert : expected) {
            selected_total += transformed[expert];
        }
        for (int rank = 0; rank < top_k; ++rank) {
            const int index = row * top_k + rank;
            require(
                sigmoid_ids[index] == expected[rank],
                "sigmoid router selected wrong expert");
            require_close(
                sigmoid_weights[index],
                transformed[expected[rank]] /
                    selected_total * 1.5f,
                5e-5f);
        }
    }

    auto delayed = mfq::metal::moe_topk(
        array(logits.begin(), Shape{rows, experts}),
        top_k,
        false,
        false,
        false,
        true);
    const auto delayed_ids = integers(delayed.ids);
    const auto delayed_weights = floats(delayed.weights);
    for (int row = 0; row < rows; ++row) {
        std::vector<float> values(
            logits.begin() + row * experts,
            logits.begin() + (row + 1) * experts);
        const auto expected = stable_top_k(values, top_k);
        const float maximum = values[expected.front()];
        float total = 0.0f;
        for (const auto expert : expected) {
            total += std::exp(values[expert] - maximum);
        }
        for (int rank = 0; rank < top_k; ++rank) {
            const int index = row * top_k + rank;
            require(
                delayed_ids[index] == expected[rank],
                "delayed softmax selected wrong expert");
            require_close(
                delayed_weights[index],
                std::exp(values[expected[rank]] - maximum) /
                    total);
        }
    }
}

void test_dense_router_logits() {
    constexpr int experts = 256;
    constexpr int width = 512;
    std::vector<float> input_values(width);
    std::vector<float> weight_values(
        static_cast<std::size_t>(experts) * width);
    for (int column = 0; column < width; ++column) {
        input_values[column] = static_cast<float>(
            (column * 7) % 31 - 15) / 128.0f;
    }
    for (std::size_t index = 0; index < weight_values.size(); ++index) {
        weight_values[index] = static_cast<float>(
            static_cast<int>((index * 11 + index / width) % 37) - 18)
            / 256.0f;
    }

    for (const auto dtype : {
             mlx::core::float16,
             mlx::core::bfloat16,
         }) {
        auto input = mlx::core::contiguous(mlx::core::astype(
            array(input_values.begin(), Shape{1, width}), dtype));
        auto weight = mlx::core::contiguous(mlx::core::astype(
            array(weight_values.begin(), Shape{experts, width}), dtype));
        require(
            mfq::metal::moe_dense_router_logits_supported(input, weight),
            "valid dense router GEMV shape was rejected");
        const auto actual = floats(
            mfq::metal::moe_dense_router_logits(input, weight));
        const auto expected = floats(mlx::core::matmul(
            input, mlx::core::transpose(weight)));
        require(actual.size() == expected.size(),
                "dense router GEMV output size mismatch");
        for (std::size_t index = 0; index < actual.size(); ++index) {
            require_close(actual[index], expected[index], 2e-2f);
        }
    }

    const std::vector<float> two_row_values(2 * width, 0.0f);
    auto two_rows = mlx::core::astype(
        array(
            two_row_values.data(),
            Shape{2, width}),
        mlx::core::float16);
    auto weight = mlx::core::astype(
        array(weight_values.begin(), Shape{experts, width}),
        mlx::core::float16);
    require(
        !mfq::metal::moe_dense_router_logits_supported(two_rows, weight),
        "multi-row input unexpectedly accepted dense router GEMV");
}

void test_single_row_softmax_topk() {
    constexpr int experts = 512;
    constexpr int top_k = 10;
    std::vector<float> logits(experts);
    for (int expert = 0; expert < experts; ++expert) {
        logits[expert] = static_cast<float>(
            (expert * 97) % 509 - 254) / 64.0f;
    }
    auto result = mfq::metal::moe_topk(
        mlx::core::astype(
            array(logits.begin(), Shape{1, experts}),
            mlx::core::float16),
        top_k,
        false,
        false,
        true);
    const auto ids = integers(result.ids);
    const auto weights = floats(result.weights);
    const auto expected = stable_top_k(logits, top_k);
    float denominator = 0.0f;
    for (const int expert : expected) {
        denominator += std::exp(logits[expert] - logits[expected.front()]);
    }
    for (int rank = 0; rank < top_k; ++rank) {
        require(ids[rank] == expected[rank],
                "single-row softmax Top-K selected wrong expert");
        require_close(
            weights[rank],
            std::exp(logits[expected[rank]] - logits[expected.front()]) /
                denominator,
            2e-3f);
    }

    const std::vector<float> ties(experts, 0.0f);
    auto tied = mfq::metal::moe_topk(
        mlx::core::astype(
            array(ties.begin(), Shape{1, experts}),
            mlx::core::bfloat16),
        top_k,
        false,
        false,
        true);
    const auto tied_ids = integers(tied.ids);
    const auto tied_weights = floats(tied.weights);
    for (int rank = 0; rank < top_k; ++rank) {
        require(tied_ids[rank] == rank,
                "single-row softmax Top-K tie order changed");
        require_close(tied_weights[rank], 0.1f, 1e-6f);
    }
}

void test_small_m_softmax_topk() {
    for (const auto dtype : {
             mlx::core::float16, mlx::core::bfloat16, mlx::core::float32}) {
        for (const int experts : {32, 512, 4096}) {
            for (const int rows : {2, 3, 4, 5, 6, 7, 8, 12, 16}) {
                constexpr int top_k = 10;
                std::vector<float> logits(rows * experts);
                for (int row = 0; row < rows; ++row) {
                    for (int expert = 0; expert < experts; ++expert) {
                        logits[row * experts + expert] = row == rows - 1
                            ? 0.0f
                            : static_cast<float>((expert * 13 + (row % 2 ? row * 7 : 0)) % 61 - 30) / 16.0f;
                    }
                }
                auto source = mlx::core::astype(
                    array(logits.begin(), Shape{rows, experts}), dtype);
                auto result = mfq::metal::moe_topk(
                    source, top_k, false, false, true);
                const auto ids = integers(result.ids);
                const auto weights = floats(result.weights);
                require(result.groups.has_value(), "small-M Top-K did not produce route groups");
                require(result.groups->shape() == Shape{rows, top_k, 8},
                        "small-M route group shape mismatch");
                const auto groups = integers(*result.groups);
                for (int index = 0; index < rows * top_k; ++index) {
                    std::vector<int> matched;
                    for (int route = 0; route < rows * top_k; ++route)
                        if (ids[route] == ids[index]) matched.push_back(route);
                    const auto position = std::find(matched.begin(), matched.end(), index) - matched.begin();
                    const auto first = position / 6 * 6;
                    const auto count = std::min<std::size_t>(6, matched.size() - first);
                    require(groups[index * 8 + 1] == static_cast<int>(count) &&
                            groups[index * 8] == ids[index], "route group count or expert mismatch");
                    for (int slot = 0; slot < 6; ++slot)
                        require(groups[index * 8 + slot + 2] ==
                            (slot < static_cast<int>(count) ? matched[first + slot] : -1),
                            "route group members or padding mismatch");
                }
                for (int row = 0; row < rows; ++row) {
                    const std::vector<float> values(
                        logits.begin() + row * experts,
                        logits.begin() + (row + 1) * experts);
                    const auto expected = stable_top_k(values, top_k);
                    float denominator = 0.0f;
                    for (const int expert : expected) {
                        denominator += std::exp(values[expert] - values[expected.front()]);
                    }
                    auto single = mfq::metal::moe_topk(
                        mlx::core::slice(source, Shape{row, 0}, Shape{row + 1, experts}),
                        top_k, false, false, true);
                    const auto single_ids = integers(single.ids);
                    const auto single_weights = floats(single.weights);
                    for (int rank = 0; rank < top_k; ++rank) {
                        const int index = row * top_k + rank;
                        require(ids[index] == expected[rank],
                                "small-M Top-K selected wrong expert or tie order");
                        require_close(weights[index],
                            std::exp(values[expected[rank]] - values[expected.front()]) / denominator);
                        require(ids[index] == single_ids[rank] && weights[index] == single_weights[rank],
                                "small-M Top-K differs from single-row execution");
                    }
                }
            }
        }
    }
}

void test_small_m_route_group_boundaries() {
    constexpr int experts = 32;
    for (int rows : {2, 6, 7, 16}) for (int top_k : {1, 15, 16}) {
        auto logits = mlx::core::reshape(mlx::core::sin(mlx::core::arange(rows * experts,
            mlx::core::float32) * array(.7919f)), Shape{rows, experts});
        const auto grouped = mfq::metal::moe_topk(logits, top_k, false, false, true);
        const auto plain = mfq::metal::moe_topk(logits, top_k, false, false, true,
            false, std::nullopt, std::nullopt, 1e-20f, 1.f, false);
        const auto ids = integers(grouped.ids);
        require(ids == integers(plain.ids) && floats(grouped.weights) == floats(plain.weights),
            "route grouping changes TopK IDs or weights");
        const auto expected = mfq::metal::moe_route_groups_host(ids, rows, top_k);
        require(grouped.groups && expected && integers(*grouped.groups) == integers(*expected),
            "route groups differ from the host oracle at the TopK boundary");
    }
}

void test_softmax_topk_normalization_parameters() {
    constexpr int experts = 32;
    std::vector<float> logits(experts);
    for (int expert = 0; expert < experts; ++expert) {
        logits[expert] = static_cast<float>(expert - experts + 1) / 16.0f;
    }
    float total = 0.0f;
    for (const float value : logits) total += std::exp(value);
    for (const int rows : {1, 6}) {
        auto source = mlx::core::broadcast_to(
            array(logits.begin(), Shape{1, experts}), Shape{rows, experts});
        for (const int top_k : {1, 16}) {
            auto result = mfq::metal::moe_topk(
                source, top_k, false, false, true, false,
                std::nullopt, std::nullopt, 2.0f, 1.5f);
            const auto ids = integers(result.ids);
            const auto weights = floats(result.weights);
            for (int row = 0; row < rows; ++row) {
                for (int rank = 0; rank < top_k; ++rank) {
                    const int index = row * top_k + rank;
                    require(ids[index] == experts - rank - 1,
                            "parameterized Top-K selected wrong expert");
                    require_close(weights[index],
                        std::exp(logits[ids[index]]) / total / 2.0f * 1.5f);
                }
            }
        }
    }
}

void test_prefill_softmax_topk_shared_reuse() {
    constexpr int experts = 512;
    constexpr int top_k = 10;
    for (const auto dtype : {
             mlx::core::float16, mlx::core::bfloat16, mlx::core::float32}) {
        for (const int rows : {4096, 8192}) {
            std::vector<float> logits(rows * experts);
            std::uint32_t random = 123456789;
            for (auto& value : logits) {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                value = static_cast<float>(int(random & 255u) - 128);
            }
            auto source = mlx::core::astype(
                array(logits.begin(), Shape{rows, experts}), dtype);
            std::vector<int> expected(rows * top_k);
            std::vector<float> expected_weights(rows * top_k);
            for (int row = 0; row < rows; ++row) {
                const std::vector<float> values(
                    logits.begin() + row * experts,
                    logits.begin() + (row + 1) * experts);
                const auto selected = stable_top_k(values, top_k);
                float denominator = 0.0f;
                for (const int expert : selected) {
                    denominator += std::exp(
                        values[expert] - values[selected.front()]);
                }
                for (int rank = 0; rank < top_k; ++rank) {
                    expected[row * top_k + rank] = selected[rank];
                    expected_weights[row * top_k + rank] = std::exp(
                        values[selected[rank]] - values[selected.front()]) / denominator;
                }
            }
            for (int repeat = 0; repeat < 6; ++repeat) {
                auto result = mfq::metal::moe_topk(
                    source, top_k, false, false, true);
                const auto ids = integers(result.ids);
                const auto weights = floats(result.weights);
                for (std::size_t index = 0; index < expected.size(); ++index) {
                    require(ids[index] == expected[index],
                            "prefill Top-K shared reduction reuse changed expert selection");
                    require_close(weights[index], expected_weights[index], 1e-6f);
                }
            }
        }
    }
}

void test_fused_dense_router_topk() {
    constexpr int experts = 256;
    constexpr int width = 4096;
    constexpr int routes = 6;
    std::vector<float> input_values(width);
    for (int column = 0; column < width; ++column) {
        input_values[column] =
            static_cast<float>((column * 7) % 19 - 9) / 32.0f;
    }
    std::vector<float> weight_values(
        static_cast<std::size_t>(experts) * width);
    for (int expert = 0; expert < experts; ++expert) {
        for (int column = 0; column < width; ++column) {
            weight_values[
                static_cast<std::size_t>(expert) * width + column
            ] =
                static_cast<float>(expert - 127) / 512.0f
                + static_cast<float>((column * 3 + expert) % 11 - 5)
                    / 1024.0f;
        }
    }
    std::vector<float> bias(experts);
    for (int expert = 0; expert < experts; ++expert) {
        bias[expert] =
            static_cast<float>((expert * 5) % 13 - 6) / 128.0f;
    }
    std::vector<std::uint8_t> available(experts, 1);
    available[255] = 0;
    available[251] = 0;

    auto input = mlx::core::contiguous(
        mlx::core::astype(
            array(input_values.begin(), Shape{1, width}),
            mlx::core::float16));
    auto weight = mlx::core::contiguous(
        mlx::core::astype(
            array(weight_values.begin(), Shape{experts, width}),
            mlx::core::float16));
    auto bias_array = array(bias.begin(), Shape{experts});
    auto available_array = mlx::core::astype(
        array(available.begin(), Shape{experts}),
        mlx::core::bool_);
    require(
        mfq::metal::moe_dense_router_topk_supported(
            input,
            weight),
        "valid fused dense router shape was rejected");
    require(
        !mfq::metal::moe_dense_router_topk_supported(
            mlx::core::astype(input, mlx::core::float32),
            weight),
        "float32 input unexpectedly accepted fused dense router");

    auto logits = mlx::core::matmul(
        mlx::core::astype(input, mlx::core::float32),
        mlx::core::transpose(
            mlx::core::astype(weight, mlx::core::float32)));
    auto reference = mfq::metal::moe_topk(
        logits,
        routes,
        false,
        true,
        true,
        false,
        bias_array,
        available_array,
        1e-20f,
        1.5f);
    auto fused = mfq::metal::moe_dense_router_topk(
        input,
        weight,
        bias_array,
        available_array,
        1e-20f,
        1.5f);
    const auto reference_ids = integers(reference.ids);
    const auto fused_ids = integers(fused.ids);
    const auto reference_weights = floats(reference.weights);
    const auto fused_weights = floats(fused.weights);
    require(
        fused_ids == reference_ids,
        "fused dense router selected different experts");
    for (int route = 0; route < routes; ++route) {
        require_close(
            fused_weights[route],
            reference_weights[route],
            8e-4f);
    }

    auto bfloat_input = mlx::core::contiguous(
        mlx::core::astype(
            array(input_values.begin(), Shape{1, width}),
            mlx::core::bfloat16));
    auto bfloat_weight = mlx::core::contiguous(
        mlx::core::astype(
            array(weight_values.begin(), Shape{experts, width}),
            mlx::core::bfloat16));
    require(
        mfq::metal::moe_dense_router_topk_supported(
            bfloat_input,
            bfloat_weight),
        "valid BF16 fused dense router shape was rejected");
    auto bfloat_logits = mlx::core::matmul(
        mlx::core::astype(
            bfloat_input,
            mlx::core::float32),
        mlx::core::transpose(
            mlx::core::astype(
                bfloat_weight,
                mlx::core::float32)));
    auto bfloat_reference = mfq::metal::moe_topk(
        bfloat_logits,
        routes,
        false,
        true,
        true,
        false,
        bias_array,
        available_array,
        1e-20f,
        1.5f);
    auto bfloat_fused = mfq::metal::moe_dense_router_topk(
        bfloat_input,
        bfloat_weight,
        bias_array,
        available_array,
        1e-20f,
        1.5f);
    require(
        integers(bfloat_fused.ids) ==
            integers(bfloat_reference.ids),
        "BF16 fused dense router selected different experts");
    const auto bfloat_reference_weights =
        floats(bfloat_reference.weights);
    const auto bfloat_fused_weights =
        floats(bfloat_fused.weights);
    for (int route = 0; route < routes; ++route) {
        require_close(
            bfloat_fused_weights[route],
            bfloat_reference_weights[route],
            2e-3f);
    }

    constexpr int verifier_rows = 6;
    std::vector<float> verifier_input_values(
        verifier_rows * width);
    for (int row = 0; row < verifier_rows; ++row) {
        for (int column = 0; column < width; ++column) {
            verifier_input_values[row * width + column] =
                input_values[column]
                + static_cast<float>((row * 3 + column) % 7 - 3)
                    / 256.0f;
        }
    }
    const auto exercise_verifier = [
        &verifier_input_values,
        &weight_values,
        &bias_array,
        &available_array
    ](mlx::core::Dtype dtype, float tolerance) {
        auto verifier_input = mlx::core::contiguous(
            mlx::core::astype(
                array(
                    verifier_input_values.begin(),
                    Shape{verifier_rows, width}),
                dtype));
        auto verifier_weight = mlx::core::contiguous(
            mlx::core::astype(
                array(
                    weight_values.begin(),
                    Shape{experts, width}),
                dtype));
        require(
            mfq::metal::moe_dense_router_topk_supported(
                verifier_input,
                verifier_weight),
            "M=6 fused dense router shape was rejected");
        auto verifier_logits = mlx::core::matmul(
            mlx::core::astype(verifier_input, mlx::core::float32),
            mlx::core::transpose(
                mlx::core::astype(
                    verifier_weight,
                    mlx::core::float32)));
        auto verifier_reference = mfq::metal::moe_topk(
            verifier_logits,
            routes,
            false,
            true,
            true,
            false,
            bias_array,
            available_array,
            1e-20f,
            1.5f);
        auto verifier_fused = mfq::metal::moe_dense_router_topk(
            verifier_input,
            verifier_weight,
            bias_array,
            available_array,
            1e-20f,
            1.5f);
        require(
            integers(verifier_fused.ids) ==
                integers(verifier_reference.ids),
            "M=6 fused dense router selected different experts");
        const auto reference_values =
            floats(verifier_reference.weights);
        const auto fused_values = floats(verifier_fused.weights);
        for (std::size_t index = 0;
             index < fused_values.size();
             ++index) {
            require_close(
                fused_values[index],
                reference_values[index],
                tolerance);
        }
    };
    exercise_verifier(mlx::core::float16, 8e-4f);
    exercise_verifier(mlx::core::bfloat16, 2e-3f);

    // DeepSeek-V4.1 widens the router from 256 to 384 experts. Exercise the
    // final reduction groups explicitly; the highest-scoring live experts
    // all sit above ID 255 in this fixture.
    constexpr int v41_experts = 384;
    constexpr int v41_width = 5120;
    std::vector<float> v41_input_values(v41_width);
    for (int column = 0; column < v41_width; ++column) {
        v41_input_values[column] =
            static_cast<float>((column % 17) + 1) / 64.0f;
    }
    std::vector<float> v41_weight_values(
        static_cast<std::size_t>(v41_experts) * v41_width);
    for (int expert = 0; expert < v41_experts; ++expert) {
        for (int column = 0; column < v41_width; ++column) {
            v41_weight_values[
                static_cast<std::size_t>(expert) * v41_width + column
            ] = static_cast<float>(expert) / 8192.0f
                + static_cast<float>((column + expert) % 7 - 3)
                    / 32768.0f;
        }
    }
    std::vector<float> v41_bias(v41_experts, 0.0f);
    std::vector<std::uint8_t> v41_available(v41_experts, 1);
    v41_available[383] = 0;
    v41_available[379] = 0;
    auto v41_input = mlx::core::contiguous(
        mlx::core::astype(
            array(v41_input_values.begin(), Shape{1, v41_width}),
            mlx::core::float16));
    auto v41_weight = mlx::core::contiguous(
        mlx::core::astype(
            array(
                v41_weight_values.begin(),
                Shape{v41_experts, v41_width}),
            mlx::core::float16));
    auto v41_bias_array = array(v41_bias.begin(), Shape{v41_experts});
    auto v41_available_array = mlx::core::astype(
        array(v41_available.begin(), Shape{v41_experts}),
        mlx::core::bool_);
    require(
        mfq::metal::moe_dense_router_topk_supported(
            v41_input,
            v41_weight),
        "valid 384-expert fused dense router shape was rejected");
    auto v41_logits = mlx::core::matmul(
        mlx::core::astype(v41_input, mlx::core::float32),
        mlx::core::transpose(
            mlx::core::astype(v41_weight, mlx::core::float32)));
    auto v41_reference = mfq::metal::moe_topk(
        v41_logits,
        routes,
        false,
        true,
        true,
        false,
        v41_bias_array,
        v41_available_array,
        1e-20f,
        1.5f);
    auto v41_fused = mfq::metal::moe_dense_router_topk(
        v41_input,
        v41_weight,
        v41_bias_array,
        v41_available_array,
        1e-20f,
        1.5f);
    const auto v41_reference_ids = integers(v41_reference.ids);
    const auto v41_fused_ids = integers(v41_fused.ids);
    require(
        v41_fused_ids == v41_reference_ids,
        "384-expert fused dense router selected different experts");
    require(
        std::all_of(
            v41_fused_ids.begin(),
            v41_fused_ids.end(),
            [](std::int32_t expert) { return expert > 255; }),
        "384-expert fused dense router did not exercise widened IDs");
    const auto v41_reference_weights = floats(v41_reference.weights);
    const auto v41_fused_weights = floats(v41_fused.weights);
    for (int route = 0; route < routes; ++route) {
        require_close(
            v41_fused_weights[route],
            v41_reference_weights[route],
            2e-3f);
    }
    const std::vector<std::int32_t> v41_expert_map(v41_experts, 0);
    bool rejected_wide_packed_ids = false;
    try {
        (void)mfq::metal::moe_dense_router_topk_packed(
            v41_input,
            v41_weight,
            array(
                v41_expert_map.begin(),
                Shape{v41_experts}),
            v41_bias_array,
            v41_available_array,
            1e-20f,
            1.5f);
    } catch (const std::invalid_argument&) {
        rejected_wide_packed_ids = true;
    }
    require(
        rejected_wide_packed_ids,
        "packed fused router accepted expert IDs wider than its encoding");

    if (std::getenv("MFQ_METAL_MOE_ROUTER_BENCH") != nullptr) {
        constexpr int warmup = 20;
        constexpr int iterations = 200;
        const auto benchmark = [&](const char* name, const auto& operation) {
            for (int iteration = 0; iteration < warmup; ++iteration) {
                auto output = operation();
                mlx::core::eval({output.ids, output.weights});
            }
            const auto begin = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < iterations; ++iteration) {
                auto output = operation();
                mlx::core::eval({output.ids, output.weights});
            }
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
            std::cout << "router_bench name=" << name
                      << " ms_per_call=" << elapsed_ms / iterations
                      << "\n";
        };
        benchmark("generic", [&]() {
            auto values = mlx::core::matmul(
                mlx::core::astype(v41_input, mlx::core::float32),
                mlx::core::transpose(
                    mlx::core::astype(v41_weight, mlx::core::float32)));
            return mfq::metal::moe_topk(
                values,
                routes,
                false,
                true,
                true,
                false,
                v41_bias_array,
                v41_available_array,
                1e-20f,
                1.5f);
        });
        benchmark("fused", [&]() {
            return mfq::metal::moe_dense_router_topk(
                v41_input,
                v41_weight,
                v41_bias_array,
                v41_available_array,
                1e-20f,
                1.5f);
        });
    }
}

void test_sqrtsoftplus_weights() {
    constexpr int rows = 2;
    constexpr int experts = 8;
    constexpr int routes = 3;
    const std::vector<float> logits{
        -2.0f, -0.5f, 0.0f, 0.4f, 1.0f, 2.0f, 3.0f, 5.0f,
        0.2f, 0.7f, -1.0f, 1.4f, 2.2f, -0.3f, 4.0f, 0.0f,
    };
    const std::vector<std::int32_t> ids{
        1, 4, 7,
        0, 3, 6,
    };
    const auto actual = floats(
        mfq::metal::moe_sqrtsoftplus_weights(
            mlx::core::astype(
                array(logits.begin(), Shape{rows, experts}),
                mlx::core::float16),
            array(ids.begin(), Shape{rows, routes}),
            1e-20f,
            1.5f));
    for (int row = 0; row < rows; ++row) {
        float denominator = 0.0f;
        float selected[routes];
        for (int route = 0; route < routes; ++route) {
            const float raw =
                static_cast<float>(
                    static_cast<mlx::core::float16_t>(
                        logits[row * experts +
                               ids[row * routes + route]]));
            selected[route] =
                std::sqrt(std::log1p(std::exp(raw)));
            denominator += selected[route];
        }
        for (int route = 0; route < routes; ++route) {
            require_close(
                actual[row * routes + route],
                selected[route] / denominator * 1.5f,
                7e-5f);
        }
    }

    const auto unnormalized = floats(
        mfq::metal::moe_selected_sqrtsoftplus_weights(
            array(logits.begin(), Shape{rows, experts}),
            array(ids.begin(), Shape{rows, routes}),
            false,
            1e-20f,
            0.75f));
    for (int row = 0; row < rows; ++row) {
        for (int route = 0; route < routes; ++route) {
            const float raw =
                logits[row * experts +
                       ids[row * routes + route]];
            require_close(
                unnormalized[row * routes + route],
                std::sqrt(std::log1p(std::exp(raw))) * 0.75f,
                7e-5f);
        }
    }
}


void test_fused_dense_hash_router() {
    constexpr int experts = 256;
    constexpr int width = 128;
    constexpr int vocab = 17;
    constexpr int routes = 6;
    std::vector<float> weight_values(
        static_cast<std::size_t>(experts) * width);
    for (int expert = 0; expert < experts; ++expert) {
        for (int column = 0; column < width; ++column) {
            weight_values[
                static_cast<std::size_t>(expert) * width + column
            ] = static_cast<float>(
                    ((expert * 11 + column * 7) % 37) - 18)
                / 512.0f;
        }
    }
    std::vector<std::int32_t> table_values(vocab * routes);
    for (int token = 0; token < vocab; ++token) {
        for (int route = 0; route < routes; ++route) {
            table_values[token * routes + route] =
                (token * 29 + route * 31 + 7) % experts;
        }
    }
    const array table(
        table_values.begin(),
        Shape{vocab, routes});

    const auto exercise = [&](mlx::core::Dtype dtype, float tolerance) {
        auto weight = mlx::core::contiguous(
            mlx::core::astype(
                array(
                    weight_values.begin(),
                    Shape{experts, width}),
                dtype));
        for (int rows = 1; rows <= 6; ++rows) {
            std::vector<float> input_values(rows * width);
            std::vector<std::int32_t> token_values(rows);
            std::vector<std::int32_t> selected_values(rows * routes);
            for (int row = 0; row < rows; ++row) {
                token_values[row] = (row * 5 + rows) % vocab;
                for (int column = 0; column < width; ++column) {
                    input_values[row * width + column] =
                        static_cast<float>(
                            ((row * 13 + column * 3) % 29) - 14)
                        / 64.0f;
                }
                for (int route = 0; route < routes; ++route) {
                    selected_values[row * routes + route] =
                        table_values[
                            token_values[row] * routes + route];
                }
            }
            auto input = mlx::core::contiguous(
                mlx::core::astype(
                    array(
                        input_values.begin(),
                        Shape{rows, width}),
                    dtype));
            const array token_ids(
                token_values.begin(),
                Shape{rows});
            require(
                mfq::metal::moe_dense_hash_router_supported(
                    input,
                    weight,
                    token_ids,
                    table),
                "valid fused dense hash router shape was rejected");
            auto fused = mfq::metal::moe_dense_hash_router(
                input,
                weight,
                token_ids,
                table,
                1e-20f,
                1.5f);
            require(
                integers(fused.ids) == selected_values,
                "fused dense hash router selected different experts");
            auto logits = mlx::core::matmul(
                mlx::core::astype(input, mlx::core::float32),
                mlx::core::transpose(
                    mlx::core::astype(weight, mlx::core::float32)));
            const auto reference_weights = floats(
                mfq::metal::moe_selected_sqrtsoftplus_weights(
                    logits,
                    array(
                        selected_values.begin(),
                        Shape{rows, routes}),
                    true,
                    1e-20f,
                    1.5f));
            const auto fused_weights = floats(fused.weights);
            for (std::size_t index = 0;
                 index < fused_weights.size();
                 ++index) {
                require_close(
                    fused_weights[index],
                    reference_weights[index],
                    tolerance);
            }
        }
    };
    exercise(mlx::core::float16, 1e-3f);
    exercise(mlx::core::bfloat16, 3e-3f);

    std::vector<float> input_values(width, 0.125f);
    const std::vector<std::int32_t> token_values{3};
    std::vector<std::int32_t> expert_map_values(experts);
    for (int expert = 0; expert < experts; ++expert) {
        expert_map_values[expert] = (expert * 3) % 41;
    }
    auto input = mlx::core::contiguous(
        mlx::core::astype(
            array(input_values.begin(), Shape{1, width}),
            mlx::core::float16));
    auto weight = mlx::core::contiguous(
        mlx::core::astype(
            array(weight_values.begin(), Shape{experts, width}),
            mlx::core::float16));
    auto packed = mfq::metal::moe_dense_hash_router_packed(
        input,
        weight,
        array(token_values.begin(), Shape{1}),
        table,
        array(expert_map_values.begin(), Shape{experts}),
        1e-20f,
        1.5f);
    const auto packed_ids = integers(packed.ids);
    for (int route = 0; route < routes; ++route) {
        const int expert = table_values[3 * routes + route];
        const int expected =
            expert | ((expert_map_values[expert] + 1) << 8);
        require(
            packed_ids[route] == expected,
            "fused dense hash router packed a wrong SSD slot");
    }
}

void test_hash_id_repair() {
    const std::vector<std::int32_t> static_ids{
        0, 1, 2,
        5, 3, 1,
    };
    const std::vector<std::int32_t> candidates{
        4, 2, 0, 5, 3, 1,
        2, 0, 4, 3, 1, 5,
    };
    const std::vector<bool> available_host{
        true,
        false,
        true,
        true,
        true,
        false,
    };
    std::vector<std::uint8_t> available_bytes(
        available_host.size());
    for (std::size_t index = 0;
         index < available_host.size();
         ++index) {
        available_bytes[index] =
            static_cast<std::uint8_t>(available_host[index]);
    }
    auto repaired = mfq::metal::moe_repair_hash_ids(
        array(static_ids.begin(), Shape{2, 3}),
        array(candidates.begin(), Shape{2, 6}),
        mlx::core::astype(
            array(
                available_bytes.begin(),
                Shape{6}),
            mlx::core::bool_));
    repaired.eval();
    const auto* values = repaired.data<std::int32_t>();
    const std::array<std::int32_t, 6> expected{
        0, 4, 2,
        2, 3, 0,
    };
    for (std::size_t index = 0;
         index < expected.size();
         ++index) {
        require(
            values[index] == expected[index],
            "hash expert repair mismatch");
    }
}

void test_reduce_and_shared_gate() {
    constexpr int tokens = 2;
    constexpr int routes = 3;
    constexpr int width = 4;
    const std::vector<float> pairs{
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.5f, 1.5f, 2.0f,
        0.25f, 0.75f, -0.5f, 1.0f,
        2.0f, -1.0f, 0.0f, 0.5f,
        1.0f, 1.5f, 2.0f, 2.5f,
        -0.5f, 0.25f, 0.75f, 1.25f,
    };
    const std::vector<float> route_weights{
        0.5f, 0.3f, 0.2f,
        0.2f, 0.6f, 0.2f,
    };
    auto pair_array = mlx::core::astype(
        array(pairs.begin(), Shape{tokens, routes, width}),
        mlx::core::float16);
    const array weight_array(
        route_weights.begin(),
        Shape{tokens, routes});
    auto reduced =
        mfq::metal::moe_weighted_reduce(
            pair_array,
            weight_array);
    require(
        reduced.dtype() == mlx::core::float16,
        "weighted reduce did not preserve activation dtype");
    const auto reduced_values = floats(reduced);
    std::vector<float> expected(tokens * width, 0.0f);
    for (int token = 0; token < tokens; ++token) {
        for (int column = 0; column < width; ++column) {
            float value = 0.0f;
            for (int route = 0; route < routes; ++route) {
                const auto pair_index =
                    (token * routes + route) * width + column;
                const float half_value =
                    static_cast<float>(
                        static_cast<mlx::core::float16_t>(
                            pairs[pair_index]));
                value += half_value *
                    route_weights[token * routes + route];
            }
            expected[token * width + column] =
                static_cast<float>(
                    static_cast<mlx::core::float16_t>(value));
            require_close(
                reduced_values[token * width + column],
                expected[token * width + column],
                1e-6f);
        }
    }

    const std::vector<float> shared{
        0.1f, 0.2f, -0.3f, 0.4f,
        0.5f, -0.4f, 0.3f, 0.2f,
    };
    const std::vector<float> gates{-0.5f, 1.0f};
    const auto shared_array = mlx::core::astype(
        array(shared.begin(), Shape{tokens, width}),
        mlx::core::float16);
    const array gate_array(gates.begin(), Shape{tokens, 1});
    const auto separate = floats(
        mfq::metal::moe_add_shared_gate(
            reduced,
            shared_array,
            gate_array));
    const auto fused = floats(
        mfq::metal::moe_weighted_reduce_shared_gate(
            pair_array,
            weight_array,
            shared_array,
            gate_array));
    for (std::size_t index = 0; index < fused.size(); ++index) {
        require_close(fused[index], separate[index], 1e-6f);
    }

    const std::vector<std::int32_t> route_order{4, 0, 5, 2, 1, 3};
    std::vector<std::int32_t> inverse_order(route_order.size());
    std::vector<float> sorted_pairs(pairs.size());
    for (std::size_t sorted = 0; sorted < route_order.size(); ++sorted) {
        const auto original = static_cast<std::size_t>(route_order[sorted]);
        inverse_order[original] = static_cast<std::int32_t>(sorted);
        std::copy_n(
            pairs.begin() + static_cast<std::ptrdiff_t>(original * width),
            width,
            sorted_pairs.begin() + static_cast<std::ptrdiff_t>(sorted * width));
    }
    const auto gpu_inverse_order = integers(
        mfq::metal::moe_inverse_permutation(
            array(route_order.begin(), Shape{tokens * routes})));
    require(
        gpu_inverse_order == inverse_order,
        "MoE inverse permutation mismatch");
    const auto sorted_reduced = floats(
        mfq::metal::moe_weighted_reduce_sorted(
            mlx::core::astype(
                array(
                    sorted_pairs.begin(),
                    Shape{tokens * routes, width}),
                mlx::core::float16),
            mfq::metal::moe_inverse_permutation(
                array(route_order.begin(), Shape{tokens * routes})),
            weight_array));
    for (std::size_t index = 0; index < reduced_values.size(); ++index) {
        require_close(sorted_reduced[index], reduced_values[index], 1e-6f);
    }
    const auto sorted_fused = floats(
        mfq::metal::moe_weighted_reduce_shared_gate_sorted(
            mlx::core::astype(
                array(
                    sorted_pairs.begin(),
                    Shape{tokens * routes, width}),
                mlx::core::float16),
            mfq::metal::moe_inverse_permutation(
                array(route_order.begin(), Shape{tokens * routes})),
            weight_array,
            shared_array,
            gate_array));
    for (std::size_t index = 0; index < fused.size(); ++index) {
        require_close(sorted_fused[index], fused[index], 1e-6f);
    }
}

void test_glu_and_expert_scale() {
    const std::vector<float> values{
        -1.0f, 0.5f, 2.0f,
        0.25f, -0.75f, 1.5f,
        0.2f, -1.3f, 0.7f,
        1.0f, 0.4f, -0.5f,
    };
    auto input = mlx::core::astype(
        array(values.begin(), Shape{2, 6}),
        mlx::core::float16);
    auto swiglu = mfq::metal::moe_swiglu_split(input);
    auto paired_swiglu = mfq::metal::moe_swiglu_pair(
        mlx::core::slice(input, {0, 0}, {2, 3}),
        mlx::core::slice(input, {0, 3}, {2, 6}));
    constexpr float swiglu_limit = 0.6f;
    auto limited_swiglu =
        mfq::metal::moe_limited_swiglu_split(
            input,
            swiglu_limit);
    const std::vector<float> gate_values{
        -1.0f, 0.5f, 2.0f,
        0.2f, -1.3f, 0.7f,
    };
    const std::vector<float> up_values{
        0.25f, -0.75f, 1.5f,
        1.0f, 0.4f, -0.5f,
    };
    auto paired_limited_swiglu = mfq::metal::moe_limited_swiglu_pair(
        mlx::core::astype(
            array(gate_values.begin(), Shape{2, 3}),
            mlx::core::float16),
        mlx::core::astype(
            array(up_values.begin(), Shape{2, 3}),
            mlx::core::float16),
        swiglu_limit);
    auto geglu = mfq::metal::moe_geglu_split(input);
    require(
        swiglu.shape() == Shape{2, 3} &&
            limited_swiglu.shape() == Shape{2, 3} &&
            geglu.shape() == Shape{2, 3} &&
            swiglu.dtype() == mlx::core::float16 &&
            limited_swiglu.dtype() == mlx::core::float16 &&
            paired_limited_swiglu.dtype() == mlx::core::float16 &&
            geglu.dtype() == mlx::core::float16,
        "GLU split shape or dtype mismatch");
    const auto swiglu_values = floats(swiglu);
    const auto paired_swiglu_values = floats(paired_swiglu);
    for (std::size_t index = 0; index < swiglu_values.size(); ++index) {
        require_close(paired_swiglu_values[index], swiglu_values[index], 1e-6f);
    }
    const auto limited_swiglu_values =
        floats(limited_swiglu);
    const auto paired_limited_values =
        floats(paired_limited_swiglu);
    const auto geglu_values = floats(geglu);
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 3; ++column) {
            const float gate = static_cast<float>(
                static_cast<mlx::core::float16_t>(
                    values[row * 6 + column]));
            const float up = static_cast<float>(
                static_cast<mlx::core::float16_t>(
                    values[row * 6 + 3 + column]));
            const float silu =
                gate / (1.0f + std::exp(-gate));
            const float limited_gate =
                std::min(gate, swiglu_limit);
            const float limited_up = std::clamp(
                up,
                -swiglu_limit,
                swiglu_limit);
            const float limited_silu =
                limited_gate /
                (1.0f + std::exp(-limited_gate));
            const float inner =
                std::sqrt(
                    2.0f / std::numbers::pi_v<float>) *
                (gate + 0.044715f * gate * gate * gate);
            const float gelu =
                0.5f * gate * (1.0f + std::tanh(inner));
            require_close(
                swiglu_values[row * 3 + column],
                static_cast<float>(
                    static_cast<mlx::core::float16_t>(
                        silu * up)),
                1e-6f);
            require_close(
                limited_swiglu_values[
                    row * 3 + column],
                static_cast<float>(
                    static_cast<mlx::core::float16_t>(
                        limited_silu * limited_up)),
                1e-6f);
            require_close(
                paired_limited_values[row * 3 + column],
                limited_swiglu_values[row * 3 + column],
                1e-6f);
            require_close(
                geglu_values[row * 3 + column],
                static_cast<float>(
                    static_cast<mlx::core::float16_t>(
                        gelu * up)),
                2e-3f);
        }
    }

    const std::vector<float> weights{
        0.2f, 0.4f, 0.6f,
        0.8f, 1.0f, 1.2f,
    };
    const std::vector<std::int32_t> ids{
        0, 2, 1,
        3, 0, 2,
    };
    const std::vector<float> scales{
        0.5f, 1.0f, 1.5f, 2.0f,
    };
    const auto scaled = floats(
        mfq::metal::moe_apply_expert_scale(
            array(weights.begin(), Shape{2, 3}),
            array(ids.begin(), Shape{2, 3}),
            array(scales.begin(), Shape{4})));
    for (std::size_t index = 0; index < scaled.size(); ++index) {
        require_close(
            scaled[index],
            weights[index] * scales[ids[index]],
            1e-6f);
    }
}

void test_restore_routed_rows() {
    using namespace mlx::core;
    for (auto dtype : {float16,bfloat16,float32}) for (int rows : {1,7,257}) for (int width : {1,13,64,511}) {
        auto values = contiguous(astype(reshape(sin(arange(rows*width,float32)),{rows,width}),dtype));
        std::vector<std::int32_t> order_values(rows),ids_values(rows);
        for (int row=0;row<rows;++row) {order_values[row]=(row*13+5)%rows;ids_values[row]=row%7==0?-1:row%5;}
        auto order = array(order_values.begin(),Shape{rows});
        auto ids = array(ids_values.begin(),Shape{rows});
        auto expected = where(expand_dims(ids>=array(0,int32),-1),take(values,argsort(order),0),zeros({rows,width},dtype));
        auto actual = mfq::metal::moe_restore_routed_rows(values,order,ids);
        auto expected_bytes = view(contiguous(expected),uint8),actual_bytes=view(contiguous(actual),uint8);
        eval(expected_bytes,actual_bytes);
        require(expected.shape()==actual.shape()&&expected.dtype()==actual.dtype(),"restored route shape/dtype mismatch");
        require(std::equal(expected_bytes.data<std::uint8_t>(),expected_bytes.data<std::uint8_t>()+expected_bytes.size(),
            actual_bytes.data<std::uint8_t>()),"restored routes are not bit-exact");
    }
}

} // namespace

int main() {
    try {
#ifdef MFQ_MLX_METALLIB_DEFAULT
        mlx::core::metal::set_metallib_path(
            MFQ_MLX_METALLIB_DEFAULT);
        mlx::core::set_default_device(
            mlx::core::Device::gpu);
#endif
        test_router_modes();
        test_dense_router_logits();
        test_single_row_softmax_topk();
        test_small_m_softmax_topk();
        test_small_m_route_group_boundaries();
        test_softmax_topk_normalization_parameters();
        test_prefill_softmax_topk_shared_reuse();
        test_fused_dense_router_topk();
        test_fused_dense_hash_router();
        test_sqrtsoftplus_weights();
        test_hash_id_repair();
        test_reduce_and_shared_gate();
        test_glu_and_expert_scale();
        test_runtime_row_kernel_reuse();
        test_restore_routed_rows();
        std::cout
            << "MFQ C++ routed MoE Metal primitive tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
