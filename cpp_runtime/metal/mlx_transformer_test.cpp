#include "mlx_transformer.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

void require_close(float actual, float expected, float tolerance = 1e-4f) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "transformer primitive mismatch: actual=" +
            std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

void require_array_close(
    const mlx::core::array& actual,
    const mlx::core::array& expected,
    float tolerance = 5e-4f) {
    using namespace mlx::core;
    if (actual.shape() != expected.shape()) {
        throw std::runtime_error(
            "transformer array shape mismatch");
    }
    auto actual_f32 = astype(actual, float32);
    auto expected_f32 = astype(expected, float32);
    actual_f32.eval();
    expected_f32.eval();
    const auto* actual_values = actual_f32.data<float>();
    const auto* expected_values = expected_f32.data<float>();
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require_close(
            actual_values[index],
            expected_values[index],
            tolerance);
    }
}

int reference_mrope_axis(
    int pair,
    const std::vector<std::int64_t>& sections,
    bool interleaved) {
    if (sections.empty()) {
        return 0;
    }
    if (!interleaved) {
        if (pair < sections[0]) {
            return 0;
        }
        return pair < sections[0] + sections[1] ? 1 : 2;
    }
    const int residue = pair % 3;
    if (residue == 1 && pair < sections[1] * 3) {
        return 1;
    }
    if (residue == 2 && pair < sections[2] * 3) {
        return 2;
    }
    return 0;
}

std::vector<float> reference_mrope(
    const std::vector<float>& input,
    int tokens,
    int dimension,
    const std::vector<std::int32_t>& positions,
    int position_axes,
    int rotary_dimension,
    float base,
    const std::vector<std::int64_t>& sections,
    bool interleaved) {
    const int rows =
        static_cast<int>(input.size()) / dimension;
    const int half = rotary_dimension / 2;
    std::vector<float> output = input;
    for (int row = 0; row < rows; ++row) {
        const int token = row % tokens;
        for (int pair = 0; pair < half; ++pair) {
            const int axis = position_axes == 1
                ? 0
                : reference_mrope_axis(
                      pair,
                      sections,
                      interleaved);
            const int position =
                positions[axis * tokens + token];
            const float frequency = std::pow(
                base,
                -2.0f * static_cast<float>(pair) /
                    static_cast<float>(rotary_dimension));
            const float angle =
                static_cast<float>(position) * frequency;
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const int first_index =
                row * dimension + pair;
            const int second_index =
                first_index + half;
            const float first = input[first_index];
            const float second = input[second_index];
            output[first_index] =
                first * cosine - second * sine;
            output[second_index] =
                second * cosine + first * sine;
        }
    }
    return output;
}

} // namespace

int main() {
    try {
        using namespace mlx::core;

        const mfq::metal::MlxRmsNorm norm(
            array({0.0f, 0.0f}, Shape{2}),
            1e-6f,
            1.0f);
        auto normalized = norm(array({3.0f, 4.0f}, Shape{1, 2}));
        normalized.eval();
        const auto* normalized_values = normalized.data<float>();
        const float inverse = 1.0f / std::sqrt(12.5f + 1e-6f);
        require_close(normalized_values[0], 3.0f * inverse);
        require_close(normalized_values[1], 4.0f * inverse);

        auto half_normalized = norm(astype(
            array({3.0f, 4.0f}, Shape{1, 2}),
            float16));
        if (half_normalized.dtype() != float16) {
            throw std::runtime_error(
                "RMSNorm promoted FP16 activations to FP32");
        }
        auto half_normalized_f32 =
            astype(half_normalized, float32);
        half_normalized_f32.eval();
        const auto* half_normalized_values =
            half_normalized_f32.data<float>();
        require_close(
            half_normalized_values[0],
            3.0f * inverse,
            2e-3f);
        require_close(
            half_normalized_values[1],
            4.0f * inverse,
            2e-3f);

        std::vector<float> wide_input_values(4096);
        std::vector<float> wide_weight_values(4096);
        for (int index = 0; index < 4096; ++index) {
            wide_input_values[static_cast<std::size_t>(index)] =
                static_cast<float>((index * 37) % 257 - 128) / 31.0f;
            wide_weight_values[static_cast<std::size_t>(index)] =
                0.75f + static_cast<float>((index * 13) % 41) / 97.0f;
        }
        const mfq::metal::MlxRmsNorm wide_norm(
            array(
                wide_weight_values.begin(),
                Shape{4096},
                float32),
            1e-6f);
        const auto wide_input = astype(
            array(
                wide_input_values.begin(),
                Shape{1, 4096},
                float32),
            float16);
        auto fused_wide = wide_norm(wide_input);
        auto reference_wide = astype(
            mlx::core::fast::rms_norm(
                wide_input,
                std::optional<array>(wide_norm.weight()),
                1e-6f),
            float16);
        mlx::core::eval(fused_wide, reference_wide);
        const auto* fused_wide_bits =
            fused_wide.data<std::uint16_t>();
        const auto* reference_wide_bits =
            reference_wide.data<std::uint16_t>();
        for (int index = 0; index < 4096; ++index) {
            if (fused_wide_bits[index] != reference_wide_bits[index]) {
                throw std::runtime_error(
                    "fused FP16 RMSNorm changed element " +
                    std::to_string(index));
            }
        }

        const array rope_input(
            {1.0f, 2.0f, 3.0f, 4.0f},
            Shape{1, 1, 1, 4});
        auto rotated = mfq::metal::apply_rope(
            rope_input,
            4,
            10000.0f,
            0);
        rotated.eval();
        const auto* rotated_values = rotated.data<float>();
        for (int index = 0; index < 4; ++index) {
            require_close(rotated_values[index], index + 1.0f);
        }

        constexpr int mrope_tokens = 3;
        constexpr int mrope_dimension = 72;
        constexpr int mrope_rotary = 64;
        constexpr float mrope_base = 10000000.0f;
        const std::vector<std::int64_t> mrope_sections{
            11,
            11,
            10,
        };
        std::vector<float> mrope_input_values(
            mrope_tokens * mrope_dimension);
        for (std::size_t index = 0;
             index < mrope_input_values.size();
             ++index) {
            mrope_input_values[index] =
                static_cast<float>(
                    static_cast<int>(index % 13) - 6) *
                0.125f;
        }
        const array mrope_input(
            mrope_input_values.begin(),
            Shape{1, 1, mrope_tokens, mrope_dimension});
        const std::vector<std::int32_t> text_positions{
            5,
            6,
            7,
        };
        const std::vector<std::int32_t> text_positions_3d{
            5,
            6,
            7,
            5,
            6,
            7,
            5,
            6,
            7,
        };
        const array text_position_array(
            text_positions.begin(),
            Shape{mrope_tokens});
        const array text_position_array_3d(
            text_positions_3d.begin(),
            Shape{3, mrope_tokens});
        auto text_fast = mfq::metal::apply_rope(
            mrope_input,
            mrope_rotary,
            mrope_base,
            5);
        auto text_explicit = mfq::metal::apply_rope(
            mrope_input,
            text_position_array,
            mrope_rotary,
            mrope_base,
            mrope_sections,
            false);
        auto text_mrope_interleaved =
            mfq::metal::apply_rope(
                mrope_input,
                text_position_array_3d,
                mrope_rotary,
                mrope_base,
                mrope_sections,
                true);
        require_array_close(text_explicit, text_fast, 7e-4f);
        require_array_close(
            text_mrope_interleaved,
            text_fast,
            7e-4f);

        const std::vector<std::int32_t> multimodal_positions{
            2,
            7,
            11,
            13,
            19,
            23,
        };
        const array multimodal_position_array(
            multimodal_positions.begin(),
            Shape{3, 2});
        std::vector<float> multimodal_input_values(
            2 * mrope_dimension);
        for (std::size_t index = 0;
             index < multimodal_input_values.size();
             ++index) {
            multimodal_input_values[index] =
                static_cast<float>(
                    static_cast<int>(index % 11) - 5) *
                0.2f;
        }
        const array multimodal_input(
            multimodal_input_values.begin(),
            Shape{1, 1, 2, mrope_dimension});
        auto contiguous_mrope = mfq::metal::apply_rope(
            multimodal_input,
            multimodal_position_array,
            mrope_rotary,
            mrope_base,
            mrope_sections,
            false);
        auto interleaved_mrope = mfq::metal::apply_rope(
            multimodal_input,
            multimodal_position_array,
            mrope_rotary,
            mrope_base,
            mrope_sections,
            true);
        const auto contiguous_expected_values =
            reference_mrope(
                multimodal_input_values,
                2,
                mrope_dimension,
                multimodal_positions,
                3,
                mrope_rotary,
                mrope_base,
                mrope_sections,
                false);
        const auto interleaved_expected_values =
            reference_mrope(
                multimodal_input_values,
                2,
                mrope_dimension,
                multimodal_positions,
                3,
                mrope_rotary,
                mrope_base,
                mrope_sections,
                true);
        require_array_close(
            contiguous_mrope,
            array(
                contiguous_expected_values.begin(),
                Shape{1, 1, 2, mrope_dimension}),
            7e-4f);
        require_array_close(
            interleaved_mrope,
            array(
                interleaved_expected_values.begin(),
                Shape{1, 1, 2, mrope_dimension}),
            7e-4f);
        bool layouts_differ = false;
        for (std::size_t index = 0;
             index < contiguous_expected_values.size();
             ++index) {
            layouts_differ =
                layouts_differ ||
                std::fabs(
                    contiguous_expected_values[index] -
                    interleaved_expected_values[index]) >
                    1e-3f;
        }
        if (!layouts_differ) {
            throw std::runtime_error(
                "interleaved MRoPE did not differ from "
                "contiguous sections");
        }
        auto half_mrope = mfq::metal::apply_rope(
            astype(multimodal_input, float16),
            multimodal_position_array,
            mrope_rotary,
            mrope_base,
            mrope_sections,
            true);
        if (half_mrope.dtype() != float16) {
            throw std::runtime_error(
                "MRoPE did not preserve FP16 activation dtype");
        }
        require_array_close(
            half_mrope,
            array(
                interleaved_expected_values.begin(),
                Shape{1, 1, 2, mrope_dimension}),
            3e-3f);

        mfq::metal::MlxKvCache cache(
            1,
            1,
            8,
            2,
            1,
            float32);
        cache.append(
            array({1.0f, 2.0f}, Shape{1, 1, 1, 2}),
            array({3.0f, 4.0f}, Shape{1, 1, 1, 2}));
        auto cached = cache.append(
            array(
                {5.0f, 6.0f, 7.0f, 8.0f},
                Shape{1, 1, 2, 2}),
            array(
                {9.0f, 10.0f, 11.0f, 12.0f},
                Shape{1, 1, 2, 2}));
        cached.first.eval();
        cached.second.eval();
        if (cache.position() != 3 || cache.capacity() < 3) {
            throw std::runtime_error("KV cache position/capacity mismatch");
        }
        const auto* cached_keys = cached.first.data<float>();
        const auto* cached_values = cached.second.data<float>();
        const float expected_cached_keys[] = {
            1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        };
        for (int index = 0; index < 6; ++index) {
            require_close(
                cached_keys[index],
                expected_cached_keys[index]);
        }
        const float expected_cached_values[] = {
            3.0f, 4.0f, 9.0f, 10.0f, 11.0f, 12.0f,
        };
        for (int index = 0; index < 6; ++index) {
            require_close(
                cached_values[index],
                expected_cached_values[index]);
        }
        cache.trim(1);
        auto rewritten = cache.append(
            array({13.0f, 14.0f}, Shape{1, 1, 1, 2}),
            array({15.0f, 16.0f}, Shape{1, 1, 1, 2}));
        rewritten.first.eval();
        rewritten.second.eval();
        if (cache.position() != 3) {
            throw std::runtime_error("KV cache trim did not rewind position");
        }
        require_close(rewritten.first.data<float>()[4], 13.0f);
        require_close(rewritten.first.data<float>()[5], 14.0f);
        require_close(rewritten.second.data<float>()[4], 15.0f);
        require_close(rewritten.second.data<float>()[5], 16.0f);

        const array query(
            {0.0f, 0.0f, 0.0f, 0.0f},
            Shape{1, 1, 2, 2});
        const array keys(
            {0.0f, 0.0f, 0.0f, 0.0f},
            Shape{1, 1, 2, 2});
        const array values(
            {2.0f, 4.0f, 6.0f, 8.0f},
            Shape{1, 1, 2, 2});
        auto attended = mfq::metal::scaled_dot_product_attention(
            query,
            keys,
            values,
            true);
        attended.eval();
        const auto* attention_values = attended.data<float>();
        require_close(attention_values[0], 2.0f);
        require_close(attention_values[1], 4.0f);
        require_close(attention_values[2], 4.0f);
        require_close(attention_values[3], 6.0f);

        std::cout
            << "MFQ C++ Transformer primitives Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
