#include "mlx_moe_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

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
        input,
        mlx::core::transpose(weight));
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
    constexpr float swiglu_limit = 0.6f;
    auto limited_swiglu =
        mfq::metal::moe_limited_swiglu_split(
            input,
            swiglu_limit);
    auto geglu = mfq::metal::moe_geglu_split(input);
    require(
        swiglu.shape() == Shape{2, 3} &&
            limited_swiglu.shape() == Shape{2, 3} &&
            geglu.shape() == Shape{2, 3} &&
            swiglu.dtype() == mlx::core::float16 &&
            limited_swiglu.dtype() == mlx::core::float16 &&
            geglu.dtype() == mlx::core::float16,
        "GLU split shape or dtype mismatch");
    const auto swiglu_values = floats(swiglu);
    const auto limited_swiglu_values =
        floats(limited_swiglu);
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
        test_fused_dense_router_topk();
        test_sqrtsoftplus_weights();
        test_hash_id_repair();
        test_reduce_and_shared_gate();
        test_glu_and_expert_scale();
        std::cout
            << "MFQ C++ routed MoE Metal primitive tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
