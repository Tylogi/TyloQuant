#include "mlx_linear_attention.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

void require_close(
    float actual,
    float expected,
    float tolerance = 2e-4f) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "linear-attention mismatch: actual=" +
            std::to_string(actual) +
            " expected=" +
            std::to_string(expected));
    }
}

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

mlx::core::array patterned_bfloat(
    std::size_t count,
    const mlx::core::Shape& shape,
    int multiplier,
    float scale) {
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const int centered =
            static_cast<int>((index * multiplier) % 257) - 128;
        values[index] = static_cast<float>(centered) * scale;
    }
    return mlx::core::astype(
        mlx::core::array(
            values.begin(), shape, mlx::core::float32),
        mlx::core::bfloat16);
}

void require_bit_exact(
    mlx::core::array actual,
    mlx::core::array expected,
    const char* name) {
    if (actual.shape() != expected.shape() ||
        actual.dtype() != mlx::core::bfloat16 ||
        expected.dtype() != mlx::core::bfloat16) {
        throw std::runtime_error(
            std::string(name) + " shape/dtype mismatch");
    }
    mlx::core::eval(actual, expected);
    const auto* actual_bits = actual.data<std::uint16_t>();
    const auto* expected_bits = expected.data<std::uint16_t>();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual_bits[index] != expected_bits[index]) {
            throw std::runtime_error(
                std::string(name) + " changed element " +
                std::to_string(index));
        }
    }
}

void require_vector_close(
    const float* actual,
    const std::vector<float>& expected,
    float tolerance,
    const char* name) {
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!std::isfinite(actual[index]) ||
            std::fabs(actual[index] - expected[index]) > tolerance) {
            throw std::runtime_error(
                std::string(name) + " mismatch at " +
                std::to_string(index) + ": actual=" +
                std::to_string(actual[index]) + " expected=" +
                std::to_string(expected[index]));
        }
    }
}

void test_blocked_gdn_prefill(bool tiled_heads) {
    using namespace mlx::core;
    constexpr int batch = 1;
    constexpr int query_heads = 2;
    constexpr int value_heads = 4;
    constexpr int tokens = 67;
    constexpr int dimension = 128;
    const std::size_t query_size =
        batch * query_heads * tokens * dimension;
    const std::size_t value_size =
        batch * value_heads * tokens * dimension;
    const std::size_t gate_size = batch * value_heads * tokens;
    const std::size_t state_size =
        batch * value_heads * dimension * dimension;

    std::vector<float> query(query_size);
    std::vector<float> key(query_size);
    std::vector<float> value(value_size);
    std::vector<float> gate(gate_size);
    std::vector<float> beta(gate_size);
    std::vector<float> initial_state(state_size);
    for (std::size_t index = 0; index < query_size; ++index) {
        query[index] =
            static_cast<float>(static_cast<int>((index * 13) % 29) - 14) *
            0.0015f;
        key[index] =
            static_cast<float>(static_cast<int>((index * 17) % 31) - 15) *
            0.0012f;
    }
    for (std::size_t index = 0; index < value_size; ++index) {
        value[index] =
            static_cast<float>(static_cast<int>((index * 19) % 37) - 18) *
            0.002f;
    }
    for (std::size_t index = 0; index < gate_size; ++index) {
        gate[index] = -0.015f - static_cast<float>(index % 7) * 0.004f;
        beta[index] = 0.35f + static_cast<float>(index % 5) * 0.07f;
    }
    for (std::size_t index = 0; index < state_size; ++index) {
        initial_state[index] =
            static_cast<float>(static_cast<int>((index * 23) % 41) - 20) *
            0.00003f;
    }

    std::vector<float> expected_output(value_size, 0.0f);
    auto expected_state = initial_state;
    const float output_scale = 1.0f / std::sqrt(float(dimension));
    for (int value_head = 0; value_head < value_heads; ++value_head) {
        const int query_head = tiled_heads
            ? value_head % query_heads
            : value_head / (value_heads / query_heads);
        float* head_state = expected_state.data() +
            value_head * dimension * dimension;
        for (int token = 0; token < tokens; ++token) {
            const float* query_row = query.data() +
                (query_head * tokens + token) * dimension;
            const float* key_row = key.data() +
                (query_head * tokens + token) * dimension;
            const float decay = std::exp(
                gate[(value_head * tokens) + token]);
            const float beta_value =
                beta[(value_head * tokens) + token];
            for (int value_dimension = 0;
                 value_dimension < dimension;
                 ++value_dimension) {
                float projected_key = 0.0f;
                for (int key_dimension = 0;
                     key_dimension < dimension;
                     ++key_dimension) {
                    projected_key +=
                        head_state[key_dimension * dimension +
                                   value_dimension] *
                        key_row[key_dimension];
                }
                const std::size_t output_index =
                    (value_head * tokens + token) * dimension +
                    value_dimension;
                const float delta =
                    (value[output_index] - decay * projected_key) *
                    beta_value;
                float output = 0.0f;
                for (int key_dimension = 0;
                     key_dimension < dimension;
                     ++key_dimension) {
                    float& state_value =
                        head_state[key_dimension * dimension +
                                   value_dimension];
                    state_value = decay * state_value +
                        key_row[key_dimension] * delta;
                    output += state_value * query_row[key_dimension];
                }
                expected_output[output_index] = output * output_scale;
            }
        }
    }

    auto result = mfq::metal::gated_delta_net(
        array(query.begin(), Shape{batch, query_heads, tokens, dimension}),
        array(key.begin(), Shape{batch, query_heads, tokens, dimension}),
        array(value.begin(), Shape{batch, value_heads, tokens, dimension}),
        array(gate.begin(), Shape{batch, value_heads, tokens}),
        array(beta.begin(), Shape{batch, value_heads, tokens}),
        array(
            initial_state.begin(),
            Shape{batch, value_heads, dimension, dimension}),
        false,
        tiled_heads);
    eval(result.output, result.state);
    require_vector_close(
        result.output.data<float>(),
        expected_output,
        3e-5f,
        tiled_heads ? "blocked tiled GDN output" : "blocked GDN output");
    require_vector_close(
        result.state.data<float>(),
        expected_state,
        3e-5f,
        tiled_heads ? "blocked tiled GDN state" : "blocked GDN state");
}

void test_cached_depthwise_dilated_decode() {
    using namespace mlx::core;
    constexpr int batch = 1;
    constexpr int channels = 10240;
    constexpr int kernel = 4;
    constexpr int dilation = 3;
    constexpr int state_length = (kernel - 1) * dilation;
    const auto input = patterned_bfloat(
        channels,
        Shape{batch, 1, channels},
        37,
        1.0f / 53.0f);
    const auto weight = patterned_bfloat(
        static_cast<std::size_t>(channels) * kernel,
        Shape{channels, kernel},
        29,
        1.0f / 4096.0f);
    const auto state = patterned_bfloat(
        static_cast<std::size_t>(batch) * state_length * channels,
        Shape{batch, state_length, channels},
        43,
        1.0f / 61.0f);

    auto combined = concatenate({state, input}, 1);
    auto reference_output = zeros(
        Shape{batch, 1, channels}, float32);
    auto weight_float = astype(weight, float32);
    for (int tap = 0; tap < kernel; ++tap) {
        auto source = slice(
            combined,
            Shape{0, tap * dilation, 0},
            Shape{batch, tap * dilation + 1, channels});
        auto coefficient = reshape(
            slice(
                weight_float,
                Shape{0, tap},
                Shape{channels, tap + 1}),
            Shape{1, 1, channels});
        reference_output = reference_output +
            astype(source, float32) * coefficient;
    }
    reference_output = astype(
        reference_output * sigmoid(reference_output),
        bfloat16);
    auto reference_state = contiguous(slice(
        combined,
        Shape{0, 1, 0},
        Shape{batch, state_length + 1, channels}));

    auto actual = mfq::metal::cached_depthwise_conv_silu(
        input,
        weight,
        std::optional<array>(state),
        dilation);
    require_bit_exact(
        std::move(actual.output),
        std::move(reference_output),
        "cached depthwise convolution output");
    require_bit_exact(
        std::move(actual.state),
        std::move(reference_state),
        "cached depthwise convolution state");
}

} // namespace

int main() {
    try {
        using namespace mlx::core;

        test_cached_depthwise_dilated_decode();
        test_blocked_gdn_prefill(false);
        test_blocked_gdn_prefill(true);

        const array conv_input(
            {
                1.0f, 2.0f,
                3.0f, 4.0f,
                5.0f, 6.0f,
            },
            Shape{1, 3, 2});
        const array conv_weight(
            {
                1.0f, 2.0f,
                -1.0f, 0.5f,
            },
            Shape{2, 2});
        const array conv_bias(
            {0.5f, -0.5f},
            Shape{2});
        auto convolved = mfq::metal::ssm_conv_silu(
            conv_input,
            conv_weight,
            2,
            conv_bias);
        convolved.eval();
        const auto* convolved_values = convolved.data<float>();
        require_close(convolved_values[0], silu(7.5f));
        require_close(convolved_values[1], silu(-0.5f));
        require_close(convolved_values[2], silu(13.5f));
        require_close(convolved_values[3], silu(-1.5f));

        constexpr int dimension = 32;
        constexpr int qk_width = 2 * dimension;
        constexpr int value_width = dimension;
        constexpr int channels = qk_width + value_width;
        std::vector<float> qk_data(2 * qk_width, 1.0f);
        std::vector<float> value_data(2 * value_width, 1.0f);
        std::vector<float> weight_data(channels * 2, 0.0f);
        std::vector<float> state_data(channels, 0.0f);
        std::vector<float> gate_data(2, 0.0f);
        std::vector<float> beta_data(2, 1.0f);
        std::vector<float> recurrent_value_data(
            2 * dimension,
            1.0f);
        for (int channel = 0; channel < channels; ++channel) {
            weight_data[channel * 2 + 1] = 1.0f;
        }
        const array state(
            state_data.begin(),
            Shape{1, 1, channels});
        const array qk_values(qk_data.begin(), Shape{1, 2, qk_width});
        const array value_values(
            value_data.begin(),
            Shape{1, 2, value_width});
        const array weights(
            weight_data.begin(),
            Shape{channels, 2});

        auto projected = mfq::metal::linear_conv_qkv(
            state,
            qk_values,
            value_values,
            weights,
            1,
            1,
            dimension,
            dimension);
        projected.query.eval();
        projected.key.eval();
        projected.value.eval();
        projected.state.eval();
        const float normalized =
            1.0f / std::sqrt(static_cast<float>(dimension));
        for (int index = 0; index < 2 * dimension; ++index) {
            require_close(
                projected.query.data<float>()[index],
                normalized);
            require_close(
                projected.key.data<float>()[index],
                normalized);
            require_close(
                projected.value.data<float>()[index],
                silu(1.0f));
        }
        const auto* next_state = projected.state.data<float>();
        for (int index = 0; index < channels; ++index) {
            require_close(next_state[index], 1.0f);
        }

        const array gate(
            gate_data.begin(),
            Shape{1, 1, 2});
        const array beta(
            beta_data.begin(),
            Shape{1, 1, 2});
        const array recurrent_value(
            recurrent_value_data.begin(),
            Shape{1, 1, 2, dimension});
        auto recurrent = mfq::metal::gated_delta_net(
            projected.query,
            projected.key,
            recurrent_value,
            gate,
            beta);
        recurrent.output.eval();
        recurrent.state.eval();
        for (int index = 0; index < 2 * dimension; ++index) {
            require_close(
                recurrent.output.data<float>()[index],
                normalized);
        }
        for (std::size_t index = 0;
             index < recurrent.state.size();
             ++index) {
            require_close(
                recurrent.state.data<float>()[index],
                normalized);
        }

        std::cout
            << "MFQ C++ Gated DeltaNet Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
