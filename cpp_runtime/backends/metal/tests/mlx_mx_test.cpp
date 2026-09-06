#include "mlx_mx.h"
#include "mlx_grouped_linear.h"
#include "mlx_nint8_zero.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

template <typename T>
void append(std::vector<std::uint8_t>& target, T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    target.insert(target.end(), bytes, bytes + sizeof(T));
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<std::uint8_t> make_blob(int bits, int outputs, int inputs) {
    std::vector<std::uint8_t> blob{'M', 'X', 'T', '1'};
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, static_cast<std::uint8_t>(bits));
    append<std::uint16_t>(blob, 0);
    append<std::uint64_t>(blob, outputs);
    append<std::uint64_t>(blob, inputs);
    append<std::uint64_t>(blob, outputs);
    append<std::uint64_t>(blob, bits == 4 ? inputs / 2 : inputs);
    append<std::uint64_t>(blob, bits == 4 ? outputs : (outputs + 127) / 128);
    append<std::uint64_t>(blob, bits == 4 ? inputs / 32 : inputs / 128);
    const auto values = static_cast<std::size_t>(outputs) *
        static_cast<std::size_t>(bits == 4 ? inputs / 2 : inputs);
    blob.insert(blob.end(), values, bits == 4 ? 0x22 : 0x38);
    const auto scales = static_cast<std::size_t>(
        bits == 4 ? outputs : (outputs + 127) / 128) *
        static_cast<std::size_t>(bits == 4 ? inputs / 32 : inputs / 128);
    blob.insert(blob.end(), scales, 127);
    return blob;
}

std::vector<std::uint8_t> make_block_scaled_mxfp8_blob() {
    constexpr int outputs = 129;
    constexpr int inputs = 256;
    auto blob = make_blob(8, outputs, inputs);
    constexpr std::size_t header_bytes = 4 + 1 + 1 + 2 + 8 * 6;
    const std::size_t scale_offset =
        header_bytes + static_cast<std::size_t>(outputs) * inputs;
    blob[scale_offset + 0] = 126;
    blob[scale_offset + 1] = 128;
    blob[scale_offset + 2] = 127;
    blob[scale_offset + 3] = 129;
    return blob;
}

std::vector<std::uint8_t> make_q8_blob(int outputs, int inputs) {
    const int groups = inputs / 32;
    std::vector<std::uint8_t> blob{'N', 'I', '8', '0'};
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, inputs);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, outputs);
    append<std::int64_t>(blob, inputs);
    append<std::uint32_t>(blob, outputs);
    append<std::uint32_t>(blob, groups);
    for (int group = 0; group < outputs * groups; ++group) {
        append<std::uint16_t>(blob, 0x3c00);
        blob.insert(blob.end(), 32, 1);
    }
    return blob;
}

void test_matmul(
    const std::string& dtype,
    int inputs,
    int rows,
    bool use_half = false) {
    using namespace mlx::core;
    constexpr int outputs = 5;
    const int bits = dtype == "MXFP4" ? 4 : 8;
    const auto weight = mfq::metal::MlxMxWeight::from_blob(
        dtype, make_blob(bits, outputs, inputs));
    std::vector<float> source_values(
        static_cast<std::size_t>(rows) * inputs);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < inputs; ++column) {
            source_values[static_cast<std::size_t>(row) * inputs + column] =
                static_cast<float>((column % 11) - 5) / 32.0f;
        }
    }
    auto source = array(source_values.begin(), Shape{rows, inputs});
    if (use_half) {
        source = astype(source, float16);
    }
    auto output = astype(
        weight.matmul(source),
        float32);
    eval(output);
    for (int row = 0; row < rows; ++row) {
        float expected = 0.0f;
        for (int column = 0; column < inputs; ++column) {
            expected += source_values[
                static_cast<std::size_t>(row) * inputs + column];
        }
        for (int out = 0; out < outputs; ++out) {
            const auto actual = output.data<float>()[row * outputs + out];
            require(
                std::isfinite(actual) && std::fabs(actual - expected) < 2e-4f,
                dtype + " Metal matmul mismatch");
        }
    }
}

void test_embedding(const std::string& dtype, int inputs) {
    using namespace mlx::core;
    constexpr int outputs = 5;
    const int bits = dtype == "MXFP4" ? 4 : 8;
    const auto weight = mfq::metal::MlxMxWeight::from_blob(
        dtype, make_blob(bits, outputs, inputs));
    const std::vector<std::int32_t> indices{4, 1};
    auto output = astype(
        weight.embedding(
            array(indices.begin(), Shape{static_cast<int>(indices.size())}),
            float16),
        float32);
    eval(output);
    require(
        output.shape() == Shape{2, inputs},
        dtype + " Metal embedding shape mismatch");
    for (std::size_t index = 0; index < output.size(); ++index) {
        require(
            output.data<float>()[index] == 1.0f,
            dtype + " Metal embedding value mismatch");
    }
}

void test_native_mxfp8_scale_expansion() {
    using namespace mlx::core;
    constexpr int outputs = 129;
    constexpr int inputs = 256;
    const auto weight = mfq::metal::MlxMxWeight::from_blob(
        "MXFP8", make_block_scaled_mxfp8_blob());
    std::vector<float> values(inputs, 1.0f);
    auto output = astype(
        weight.matmul(astype(
            array(values.begin(), Shape{1, inputs}),
            float16)),
        float32);
    eval(output);
    require(
        output.shape() == Shape{1, outputs},
        "native MXFP8 scale expansion shape mismatch");
    require(
        output.data<float>()[0] == 320.0f &&
            output.data<float>()[127] == 320.0f &&
            output.data<float>()[128] == 640.0f,
        "native MXFP8 scale expansion value mismatch");
}

void test_grouped_mxfp8() {
    using namespace mlx::core;
    constexpr int inputs = 128;
    const auto first = mfq::metal::MlxMxWeight::from_blob(
        "MXFP8", make_blob(8, 17, inputs));
    const auto second = mfq::metal::MlxMxWeight::from_blob(
        "MXFP8", make_blob(8, 9, inputs));
    const mfq::metal::MlxGroupedLinear grouped({&first, &second});
    require(
        grouped.has_single_row_mxfp8_fast_path(),
        "grouped MXFP8 fast path was not selected");
    std::vector<float> values(inputs);
    for (int index = 0; index < inputs; ++index) {
        values[static_cast<std::size_t>(index)] =
            static_cast<float>((index % 13) - 6) / 32.0f;
    }
    auto input = astype(
        array(values.begin(), Shape{1, inputs}),
        float16);
    const auto expected_first = astype(first.matmul(input), float32);
    const auto expected_second = astype(second.matmul(input), float32);
    auto actual = grouped(input);
    actual[0] = astype(actual[0], float32);
    actual[1] = astype(actual[1], float32);
    eval(expected_first, expected_second, actual[0], actual[1]);
    require(
        actual[0].shape() == Shape{1, 17} &&
        actual[1].shape() == Shape{1, 9},
        "grouped MXFP8 output shape mismatch");
    for (std::size_t index = 0; index < actual[0].size(); ++index) {
        require(
            std::fabs(
                actual[0].data<float>()[index] -
                expected_first.data<float>()[index]) < 2e-4f,
            "grouped MXFP8 first projection mismatch");
    }
    for (std::size_t index = 0; index < actual[1].size(); ++index) {
        require(
            std::fabs(
                actual[1].data<float>()[index] -
                expected_second.data<float>()[index]) < 2e-4f,
            "grouped MXFP8 second projection mismatch");
    }

}

void test_grouped_mxfp8_swiglu() {
    using namespace mlx::core;
    constexpr int inputs = 128;
    constexpr int outputs = 17;
    const auto gate = mfq::metal::MlxMxWeight::from_blob(
        "MXFP8", make_blob(8, outputs, inputs));
    const auto up = mfq::metal::MlxMxWeight::from_blob(
        "MXFP8", make_blob(8, outputs, inputs));
    const mfq::metal::MlxGroupedLinear grouped({&gate, &up});
    std::vector<float> values(inputs);
    for (int index = 0; index < inputs; ++index) {
        values[static_cast<std::size_t>(index)] =
            static_cast<float>((index % 13) - 6) / 32.0f;
    }
    auto input = astype(
        array(values.begin(), Shape{1, inputs}),
        float16);
    require(
        grouped.supports_single_row_swiglu(input),
        "grouped MXFP8 SwiGLU fast path was not selected");
    auto gate_expected = astype(gate.matmul(input), float32);
    auto up_expected = astype(up.matmul(input), float32);
    auto actual = astype(
        grouped.single_row_swiglu(input, 0.0f),
        float32);
    eval(gate_expected, up_expected, actual);
    require(
        actual.shape() == Shape{1, outputs},
        "grouped MXFP8 SwiGLU output shape mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float gate_value = gate_expected.data<float>()[index];
        const float up_value = up_expected.data<float>()[index];
        const float expected =
            gate_value / (1.0f + std::exp(-gate_value)) * up_value;
        require(
            std::fabs(actual.data<float>()[index] - expected) < 2e-3f,
            "grouped MXFP8 SwiGLU value mismatch");
    }

    auto bfloat_input = astype(
        array(values.begin(), Shape{1, inputs}),
        bfloat16);
    require(
        grouped.supports_single_row_swiglu(bfloat_input),
        "grouped MXFP8 BF16 SwiGLU fast path was not selected");
    auto bfloat_gate_expected = astype(
        gate.matmul(bfloat_input),
        float32);
    auto bfloat_up_expected = astype(
        up.matmul(bfloat_input),
        float32);
    auto bfloat_actual = astype(
        grouped.single_row_swiglu(bfloat_input, 0.0f),
        float32);
    eval(
        bfloat_gate_expected,
        bfloat_up_expected,
        bfloat_actual);
    for (std::size_t index = 0;
         index < bfloat_actual.size();
         ++index) {
        const float gate_value =
            bfloat_gate_expected.data<float>()[index];
        const float up_value =
            bfloat_up_expected.data<float>()[index];
        const float expected =
            gate_value / (1.0f + std::exp(-gate_value)) * up_value;
        require(
            std::fabs(
                bfloat_actual.data<float>()[index] - expected)
                < 2e-2f,
            "grouped MXFP8 BF16 SwiGLU value mismatch");
    }
}

void test_grouped_mxfp8_q8() {
    using namespace mlx::core;
    constexpr int inputs = 128;
    const auto first = mfq::metal::MlxMxWeight::from_blob(
        "MXFP8", make_blob(8, 19, inputs));
    const auto second = mfq::metal::MlxNint8ZeroWeight::from_blob(
        make_q8_blob(11, inputs));
    const auto third = mfq::metal::MlxMxWeight::from_blob(
        "MXFP8", make_blob(8, 7, inputs));
    const mfq::metal::MlxGroupedLinear grouped(
        {&first, &second, &third});
    require(
        grouped.has_single_row_mxfp8_fast_path(),
        "mixed grouped MXFP8 fast path was not selected");
    std::vector<float> values(inputs);
    for (int index = 0; index < inputs; ++index) {
        values[static_cast<std::size_t>(index)] =
            static_cast<float>((index % 9) - 4) / 16.0f;
    }
    auto input = astype(
        array(values.begin(), Shape{1, inputs}),
        float16);
    std::vector<array> expected{
        astype(first.matmul(input), float32),
        astype(second.matmul(input), float32),
        astype(third.matmul(input), float32),
    };
    auto actual = grouped(input);
    for (auto& value : actual) {
        value = astype(value, float32);
    }
    eval(expected);
    eval(actual);
    require(actual.size() == expected.size(), "mixed grouped output count");
    for (std::size_t projection = 0; projection < actual.size(); ++projection) {
        require(
            actual[projection].shape() == expected[projection].shape(),
            "mixed grouped output shape mismatch");
        for (std::size_t index = 0; index < actual[projection].size(); ++index) {
            require(
                std::fabs(
                    actual[projection].data<float>()[index] -
                    expected[projection].data<float>()[index]) < 2e-4f,
                "mixed grouped MXFP8/NINT8-0 projection mismatch");
        }
    }
}

void test_grouped_mxfp8_inverse_rope() {
    using namespace mlx::core;
    constexpr int inputs = 128;
    constexpr int outputs = 10;
    const auto weight = mfq::metal::MlxMxWeight::from_blob(
        "MXFP8", make_blob(8, outputs, inputs));
    std::vector<float> values(2 * inputs, 0.5f);
    std::fill(values.begin() + inputs, values.end(), -0.25f);
    auto input = astype(
        array(values.begin(), Shape{1, 1, 2, inputs}),
        float16);
    const std::vector<float> cosine(32, 1.0f);
    const std::vector<float> sine(32, 0.0f);
    auto output = astype(
        weight.grouped_row_matmul_inverse_rope(
            input,
            2,
            array(cosine.begin(), Shape{1, 32}),
            array(sine.begin(), Shape{1, 32}),
            128,
            64),
        float32);
    eval(output);
    require(
        output.shape() == Shape{1, 1, 2, 5},
        "MXFP8 inverse-RoPE grouped-row shape mismatch");
    for (int index = 0; index < outputs; ++index) {
        const float expected = index < 5 ? 64.0f : -32.0f;
        require(
            std::fabs(output.data<float>()[index] - expected) < 2e-3f,
            "MXFP8 inverse-RoPE grouped-row value mismatch");
    }
}

void test_grouped_row_mxfp8_prefill() {
    using namespace mlx::core;
    constexpr int groups = 2;
    constexpr int outputs_per_group = 3;
    constexpr int inputs = 128;
    constexpr int tokens = 3;
    const auto weight = mfq::metal::MlxMxWeight::from_blob(
        "MXFP8",
        make_blob(
            8,
            groups * outputs_per_group,
            inputs));
    std::vector<float> values(
        static_cast<std::size_t>(tokens) * groups * inputs);
    for (int token = 0; token < tokens; ++token) {
        for (int group = 0; group < groups; ++group) {
            const float value =
                static_cast<float>(1 + token * groups + group) /
                128.0f;
            for (int column = 0; column < inputs; ++column) {
                values[
                    (static_cast<std::size_t>(token) * groups + group) *
                        inputs + column] = value;
            }
        }
    }
    auto input = astype(
        array(values.begin(), Shape{1, tokens, groups, inputs}),
        float16);
    auto output = contiguous(astype(
        weight.grouped_row_matmul(input, groups),
        float32));
    eval(output);
    require(
        output.shape() ==
            Shape{1, tokens, groups, outputs_per_group},
        "MXFP8 prefill grouped-row shape mismatch");
    for (int token = 0; token < tokens; ++token) {
        for (int group = 0; group < groups; ++group) {
            const float expected =
                static_cast<float>(1 + token * groups + group);
            for (int output_index = 0;
                 output_index < outputs_per_group;
                 ++output_index) {
                const std::size_t index =
                    ((static_cast<std::size_t>(token) * groups + group) *
                        outputs_per_group) + output_index;
                require(
                    std::fabs(output.data<float>()[index] - expected) <
                        2e-3f,
                    "MXFP8 prefill grouped-row value mismatch");
            }
        }
    }
}

} // namespace

int main() {
    try {
        test_matmul("MXFP4", 96, 1);
        test_matmul("MXFP4", 96, 7);
        test_matmul("MXFP4", 96, 64);
        test_matmul("MXFP8", 128, 1);
        test_matmul("MXFP8", 128, 1, true);
        test_matmul("MXFP8", 128, 7);
        test_matmul("MXFP8", 128, 64);
        test_native_mxfp8_scale_expansion();
        test_embedding("MXFP4", 96);
        test_embedding("MXFP8", 128);
        test_grouped_mxfp8();
        test_grouped_mxfp8_swiglu();
        test_grouped_mxfp8_q8();
        test_grouped_mxfp8_inverse_rope();
        test_grouped_row_mxfp8_prefill();
        std::cout << "MFQ MXFP4/MXFP8 Metal GEMV/MMQ/GEMM/grouped/embedding passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
