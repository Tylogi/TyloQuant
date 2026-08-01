#include "mlx_linear_attention.h"

#include <cmath>
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

} // namespace

int main() {
    try {
        using namespace mlx::core;

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
