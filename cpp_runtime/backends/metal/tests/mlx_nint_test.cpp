#include "mlx_nint.h"

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <mlx/mlx.h>

namespace {

template <typename T>
void append(std::vector<std::uint8_t>& blob, T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    blob.insert(blob.end(), bytes, bytes + sizeof(value));
}

std::vector<std::uint8_t> pack_values(
    const std::vector<std::uint8_t>& values,
    int bits) {
    std::vector<std::uint8_t> result(
        (values.size() * static_cast<std::size_t>(bits) + 7) / 8,
        0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        for (int bit = 0; bit < bits; ++bit) {
            if ((values[index] >> bit) & 1u) {
                const auto target =
                    index * static_cast<std::size_t>(bits) +
                    static_cast<std::size_t>(bit);
                result[target / 8] |=
                    static_cast<std::uint8_t>(1u << (target & 7));
            }
        }
    }
    return result;
}

struct Fixture {
    std::vector<std::uint8_t> blob;
    std::vector<std::uint8_t> quantized;
};

struct ScaledFixture {
    std::vector<std::uint8_t> blob;
    std::vector<std::uint8_t> quantized;
    std::vector<std::uint8_t> sub_scales;
    std::vector<std::uint8_t> sub_mins;
};

Fixture make_nint_blob(int bits) {
    constexpr std::int32_t output_size = 2;
    constexpr std::int32_t group_size = 5;
    constexpr std::int32_t groups = 2;
    constexpr std::int32_t input_size = 9;
    const auto maximum = (1u << bits) - 1u;
    std::vector<std::uint8_t> quantized(
        output_size * groups * group_size);
    for (std::size_t index = 0; index < quantized.size(); ++index) {
        quantized[index] = static_cast<std::uint8_t>(
            (index * 3 + 1) % (maximum + 1));
    }

    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(blob, static_cast<std::uint8_t>(bits));
    append<std::uint8_t>(blob, 1);
    append<std::int32_t>(blob, group_size);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input_size);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, input_size);
    append<std::uint32_t>(blob, output_size);
    append<std::uint32_t>(blob, groups);
    append<std::uint16_t>(blob, 0x3c00);
    append<std::uint16_t>(blob, 0x3c00);
    append<std::uint16_t>(blob, 0);
    append<std::uint16_t>(blob, 0);
    append<std::uint8_t>(blob, 0x0f);
    append<std::uint8_t>(blob, 0x00);
    const auto packed = pack_values(quantized, bits);
    blob.insert(blob.end(), packed.begin(), packed.end());
    return {std::move(blob), std::move(quantized)};
}

Fixture make_nint_blob(
    int bits,
    std::int32_t output_size,
    std::int32_t group_size,
    std::int32_t groups,
    std::int32_t input_size,
    int seed) {
    if (output_size <= 0 ||
        group_size <= 0 ||
        groups <= 0 ||
        input_size <= 0 ||
        input_size > group_size * groups) {
        throw std::runtime_error("invalid shaped NINT test fixture");
    }
    const auto maximum = (1u << bits) - 1u;
    const auto metadata_count =
        static_cast<std::size_t>(output_size) *
        static_cast<std::size_t>(groups);
    std::vector<std::uint8_t> quantized(
        metadata_count * static_cast<std::size_t>(group_size));
    for (std::size_t index = 0; index < quantized.size(); ++index) {
        quantized[index] = static_cast<std::uint8_t>(
            (index * 5 + static_cast<std::size_t>(seed)) %
            (maximum + 1));
    }

    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(blob, static_cast<std::uint8_t>(bits));
    append<std::uint8_t>(blob, 1);
    append<std::int32_t>(blob, group_size);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input_size);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, input_size);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(output_size));
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(groups));
    for (int index = 0; index < output_size; ++index) {
        append<std::uint16_t>(blob, 0x3c00);
    }
    for (int index = 0; index < output_size; ++index) {
        append<std::uint16_t>(blob, 0);
    }
    const std::vector<std::uint8_t> sub_scales(
        metadata_count,
        1);
    const std::vector<std::uint8_t> sub_mins(
        metadata_count,
        0);
    const auto packed_scales = pack_values(sub_scales, 1);
    const auto packed_mins = pack_values(sub_mins, 1);
    const auto packed_q = pack_values(quantized, bits);
    blob.insert(blob.end(), packed_scales.begin(), packed_scales.end());
    blob.insert(blob.end(), packed_mins.begin(), packed_mins.end());
    blob.insert(blob.end(), packed_q.begin(), packed_q.end());
    return {std::move(blob), std::move(quantized)};
}

ScaledFixture make_nint5_gs28_scaled_blob() {
    constexpr std::int32_t output_size = 37;
    constexpr std::int32_t group_size = 28;
    constexpr std::int32_t groups = 4;
    constexpr std::int32_t input_size = 111;
    constexpr int bits = 5;
    constexpr int sub_bits = 7;
    const auto metadata_count =
        static_cast<std::size_t>(output_size) * groups;
    std::vector<std::uint8_t> quantized(
        metadata_count * static_cast<std::size_t>(group_size));
    std::vector<std::uint8_t> sub_scales(metadata_count);
    std::vector<std::uint8_t> sub_mins(metadata_count);
    for (std::size_t index = 0; index < metadata_count; ++index) {
        sub_scales[index] =
            static_cast<std::uint8_t>(1 + (index * 3 + 2) % 7);
        sub_mins[index] =
            static_cast<std::uint8_t>((index * 5 + 1) % 6);
    }
    for (std::size_t index = 0; index < quantized.size(); ++index) {
        quantized[index] =
            static_cast<std::uint8_t>((index * 13 + 7) & 31u);
    }

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
        // 0x3000 = FP16 0.125.
        append<std::uint16_t>(blob, 0x3000);
    }
    for (int output = 0; output < output_size; ++output) {
        // 0x2c00 = FP16 0.0625.
        append<std::uint16_t>(blob, 0x2c00);
    }
    const auto packed_scales = pack_values(sub_scales, sub_bits);
    const auto packed_mins = pack_values(sub_mins, sub_bits);
    const auto packed_q = pack_values(quantized, bits);
    blob.insert(
        blob.end(),
        packed_scales.begin(),
        packed_scales.end());
    blob.insert(
        blob.end(),
        packed_mins.begin(),
        packed_mins.end());
    blob.insert(blob.end(), packed_q.begin(), packed_q.end());
    return {
        std::move(blob),
        std::move(quantized),
        std::move(sub_scales),
        std::move(sub_mins),
    };
}

ScaledFixture make_nint_gs24_scaled_blob(
    int bits,
    std::int32_t output_size,
    std::int32_t input_size) {
    constexpr std::int32_t group_size = 24;
    constexpr int sub_bits = 7;
    if ((bits != 4 && bits != 6) ||
        output_size <= 0 ||
        input_size <= 0) {
        throw std::runtime_error("invalid NINT GS24 test fixture");
    }
    const auto maximum = (1u << bits) - 1u;
    const std::int32_t groups =
        (input_size + group_size - 1) / group_size;
    const auto metadata_count =
        static_cast<std::size_t>(output_size) *
        static_cast<std::size_t>(groups);
    std::vector<std::uint8_t> quantized(
        metadata_count * static_cast<std::size_t>(group_size));
    std::vector<std::uint8_t> sub_scales(metadata_count);
    std::vector<std::uint8_t> sub_mins(metadata_count);
    for (std::size_t index = 0; index < metadata_count; ++index) {
        sub_scales[index] =
            static_cast<std::uint8_t>(1 + (index * 5 + 3) % 8);
        sub_mins[index] =
            static_cast<std::uint8_t>(1 + (index * 7 + 2) % 6);
    }
    for (std::size_t index = 0; index < quantized.size(); ++index) {
        quantized[index] =
            static_cast<std::uint8_t>(
                (index * 17 + 11) & maximum);
    }

    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(blob, bits);
    append<std::uint8_t>(blob, sub_bits);
    append<std::int32_t>(blob, group_size);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input_size);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, input_size);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(output_size));
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(groups));
    for (int output = 0; output < output_size; ++output) {
        // 0x3000 = FP16 0.125.
        append<std::uint16_t>(blob, 0x3000);
    }
    for (int output = 0; output < output_size; ++output) {
        // 0x2c00 = FP16 0.0625.
        append<std::uint16_t>(blob, 0x2c00);
    }
    const auto packed_scales = pack_values(sub_scales, sub_bits);
    const auto packed_mins = pack_values(sub_mins, sub_bits);
    const auto packed_q = pack_values(quantized, bits);
    blob.insert(
        blob.end(),
        packed_scales.begin(),
        packed_scales.end());
    blob.insert(
        blob.end(),
        packed_mins.begin(),
        packed_mins.end());
    blob.insert(blob.end(), packed_q.begin(), packed_q.end());
    return {
        std::move(blob),
        std::move(quantized),
        std::move(sub_scales),
        std::move(sub_mins),
    };
}

void require_close(
    float actual,
    float expected,
    float tolerance = 1e-4f) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "NINT Metal result mismatch: actual=" +
            std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

void test_nint5_gs28_decode() {
    using namespace mlx::core;
    constexpr int output_size = 37;
    constexpr int group_size = 28;
    constexpr int groups = 4;
    constexpr int input_size = 111;
    const auto fixture = make_nint5_gs28_scaled_blob();
    const auto weight =
        mfq::metal::MlxNintWeight::from_blob(fixture.blob);

    std::vector<float> input_values(2 * input_size);
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] =
            static_cast<float>(
                static_cast<int>((index * 11 + 3) % 31) - 15) /
            512.0f;
    }

    for (const auto dtype : {float16, float32}) {
        const int rows = dtype == float16 ? 1 : 2;
        const auto input = astype(
            array(
                input_values.begin(),
                Shape{rows, input_size}),
            dtype);
        auto output = astype(weight.matmul(input), float32);
        eval(output);
        const auto* values = output.data<float>();
        for (int input_row = 0; input_row < rows; ++input_row) {
            for (int output_row = 0;
                 output_row < output_size;
                 ++output_row) {
                float expected = 0.0f;
                for (int column = 0; column < input_size; ++column) {
                    const int group = column / group_size;
                    const auto metadata_index =
                        static_cast<std::size_t>(output_row) * groups +
                        static_cast<std::size_t>(group);
                    const auto quantized_index =
                        metadata_index * group_size +
                        static_cast<std::size_t>(column % group_size);
                    const float decoded =
                        0.125f *
                            fixture.sub_scales[metadata_index] *
                            fixture.quantized[quantized_index] -
                        0.0625f *
                            fixture.sub_mins[metadata_index];
                    expected +=
                        input_values[
                            static_cast<std::size_t>(input_row) *
                                input_size +
                            static_cast<std::size_t>(column)] *
                        decoded;
                }
                const float tolerance =
                    dtype == float16
                    ? 0.005f + 0.001f * std::fabs(expected)
                    : 0.0002f + 0.00002f * std::fabs(expected);
                require_close(
                    values[input_row * output_size + output_row],
                    expected,
                    tolerance);
            }
        }
    }
}

void verify_nint_gs24_decode(
    int bits,
    int output_size,
    int input_size,
    bool exercise_fallbacks) {
    using namespace mlx::core;
    constexpr int group_size = 24;
    const int groups = (input_size + group_size - 1) / group_size;
    const auto fixture =
        make_nint_gs24_scaled_blob(
            bits,
            output_size,
            input_size);
    const auto weight =
        mfq::metal::MlxNintWeight::from_blob(fixture.blob);

    std::vector<float> input_values(
        2 * static_cast<std::size_t>(input_size));
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] =
            static_cast<float>(
                static_cast<int>((index * 13 + 5) % 37) - 18) /
            512.0f;
    }

    const int passes = exercise_fallbacks ? 3 : 1;
    for (int pass = 0; pass < passes; ++pass) {
        const int rows = pass == 2 ? 2 : 1;
        const auto dtype = pass == 1 ? float32 : float16;
        const auto input = astype(
            array(
                input_values.begin(),
                Shape{rows, input_size}),
            dtype);
        auto output = astype(weight.matmul(input), float32);
        eval(output);
        const auto* values = output.data<float>();
        for (int input_row = 0; input_row < rows; ++input_row) {
            for (int output_row = 0;
                 output_row < output_size;
                 ++output_row) {
                float expected = 0.0f;
                for (int column = 0; column < input_size; ++column) {
                    const int group = column / group_size;
                    const auto metadata_index =
                        static_cast<std::size_t>(output_row) * groups +
                        static_cast<std::size_t>(group);
                    const auto quantized_index =
                        metadata_index * group_size +
                        static_cast<std::size_t>(
                            column % group_size);
                    const float decoded =
                        0.125f *
                            fixture.sub_scales[metadata_index] *
                            fixture.quantized[quantized_index] -
                        0.0625f *
                            fixture.sub_mins[metadata_index];
                    expected +=
                        input_values[
                            static_cast<std::size_t>(input_row) *
                                input_size +
                            static_cast<std::size_t>(column)] *
                        decoded;
                }
                const float tolerance =
                    dtype == float16
                    ? 0.006f + 0.001f * std::fabs(expected)
                    : 0.0003f +
                        0.00003f * std::fabs(expected);
                require_close(
                    values[
                        input_row * output_size + output_row],
                    expected,
                    tolerance);
            }
        }

        if (bits == 6 && rows == 1 && dtype == float16) {
            auto fused = weight.greedy_argmax(input);
            if (!fused) {
                throw std::runtime_error(
                    "NINT6/GS24 fused greedy path was unavailable");
            }
            fused->eval();
            int expected_index = 0;
            for (int output_row = 1;
                 output_row < output_size;
                 ++output_row) {
                if (values[output_row] > values[expected_index]) {
                    expected_index = output_row;
                }
            }
            if (fused->data<std::int32_t>()[0] != expected_index) {
                throw std::runtime_error(
                    "NINT6/GS24 fused greedy token mismatch");
            }
        }

        if (bits == 4) {
            std::vector<float> residual_values(
                static_cast<std::size_t>(rows) * output_size);
            for (std::size_t index = 0;
                 index < residual_values.size();
                 ++index) {
                residual_values[index] =
                    static_cast<float>(
                        static_cast<int>(index % 19) - 9) /
                    64.0f;
            }
            const auto residual = astype(
                array(
                    residual_values.begin(),
                    Shape{rows, output_size}),
                dtype);
            auto reference = astype(
                weight.matmul(input) + residual,
                float32);
            auto fused = astype(
                weight.matmul_add(input, residual),
                float32);
            eval(reference, fused);
            const auto* expected = reference.data<float>();
            const auto* actual = fused.data<float>();
            for (std::size_t index = 0;
                 index < residual_values.size();
                 ++index) {
                require_close(
                    actual[index],
                    expected[index],
                    dtype == float16 ? 0.001f : 0.0001f);
            }
        }
    }
}

void test_nint4_gs24_decode() {
    // Both dimensions have tails: K is not a multiple of GS24 and OUT is not
    // a multiple of the kernel's 16-row threadgroup tile. FP32 and M2 also
    // exercise the retained NINT4 fallbacks.
    verify_nint_gs24_decode(4, 37, 111, true);

    // Exercise direct byte addressing and a large non-aligned output grid.
    verify_nint_gs24_decode(4, 65'539, 25, false);
}

void test_nint6_gs24_decode() {
    // Both dimensions have tails: K is not a multiple of GS24 and OUT is not
    // a multiple of the kernel's 16-row threadgroup tile.
    verify_nint_gs24_decode(6, 37, 111, true);

    // Exercise a large, non-16-aligned output grid without allocating a dense
    // Qwen-size matrix. Metadata and q-values remain non-zero throughout.
    verify_nint_gs24_decode(6, 65'539, 25, false);
}

void test_nint4_swiglu() {
    using namespace mlx::core;
    constexpr int output_size = 11;
    constexpr int group_size = 6;
    constexpr int groups = 3;
    constexpr int input_size = 17;
    const auto gate_fixture = make_nint_blob(
        4,
        output_size,
        group_size,
        groups,
        input_size,
        1);
    const auto up_fixture = make_nint_blob(
        4,
        output_size,
        group_size,
        groups,
        input_size,
        7);
    const auto gate =
        mfq::metal::MlxNintWeight::from_blob(gate_fixture.blob);
    const auto up =
        mfq::metal::MlxNintWeight::from_blob(up_fixture.blob);
    if (!gate.can_fuse_swiglu(up)) {
        throw std::runtime_error(
            "compatible NINT4 SwiGLU weights were rejected");
    }

    std::vector<float> input_values(input_size);
    for (int index = 0; index < input_size; ++index) {
        input_values[static_cast<std::size_t>(index)] =
            static_cast<float>((index * 11) % 29 - 14) / 128.0f;
    }
    for (const auto dtype : {float16, float32}) {
        const auto input = astype(
            array(
                input_values.begin(),
                Shape{1, 1, input_size}),
            dtype);
        const auto gate_value = gate.matmul(input);
        const auto up_value = up.matmul(input);
        auto reference = astype(
            gate_value * sigmoid(gate_value) * up_value,
            float32);
        auto fused = astype(gate.swiglu(up, input), float32);
        eval(reference, fused);
        const auto* expected = reference.data<float>();
        const auto* actual = fused.data<float>();
        for (int output = 0; output < output_size; ++output) {
            const float tolerance =
                dtype == float16
                ? 0.08f +
                    0.01f * std::fabs(expected[output])
                : 0.002f +
                    0.0002f * std::fabs(expected[output]);
            require_close(
                actual[output],
                expected[output],
                tolerance);
        }
    }

    // The production MiniCPM decode path uses GS24. Verify the fused
    // group-affine formulation directly against the established separate
    // GS24 GEMVs, including both input and output tails.
    {
        constexpr int gs24_output = 37;
        constexpr int gs24_group_size = 24;
        constexpr int gs24_groups = 5;
        constexpr int gs24_input = 111;
        const auto gs24_gate = mfq::metal::MlxNintWeight::from_blob(
            make_nint_blob(
                4,
                gs24_output,
                gs24_group_size,
                gs24_groups,
                gs24_input,
                11).blob);
        const auto gs24_up = mfq::metal::MlxNintWeight::from_blob(
            make_nint_blob(
                4,
                gs24_output,
                gs24_group_size,
                gs24_groups,
                gs24_input,
                17).blob);
        std::vector<float> gs24_input_values(gs24_input);
        for (int index = 0; index < gs24_input; ++index) {
            gs24_input_values[static_cast<std::size_t>(index)] =
                static_cast<float>((index * 17) % 41 - 20) / 192.0f;
        }
        const auto gs24_source = astype(
            array(
                gs24_input_values.begin(),
                Shape{1, 1, gs24_input}),
            float16);
        const auto gs24_gate_value = gs24_gate.matmul(gs24_source);
        const auto gs24_up_value = gs24_up.matmul(gs24_source);
        auto gs24_reference = astype(
            gs24_gate_value * sigmoid(gs24_gate_value) * gs24_up_value,
            float32);
        auto gs24_fused = astype(
            gs24_gate.swiglu(gs24_up, gs24_source),
            float32);
        auto maximum = max(abs(gs24_fused - gs24_reference));
        maximum.eval();
        if (!std::isfinite(maximum.item<float>()) ||
            maximum.item<float>() > 0.0f) {
            throw std::runtime_error(
                "GS24 fused NINT4 SwiGLU changed FP16 values: " +
                std::to_string(maximum.item<float>()));
        }
    }

    const auto odd_group = mfq::metal::MlxNintWeight::from_blob(
        make_nint_blob(4).blob);
    if (gate.can_fuse_swiglu(odd_group)) {
        throw std::runtime_error(
            "incompatible odd-GS NINT4 SwiGLU weights were accepted");
    }
    const auto nint6 = mfq::metal::MlxNintWeight::from_blob(
        make_nint_blob(
            6,
            output_size,
            group_size,
            groups,
            input_size,
            3).blob);
    if (gate.can_fuse_swiglu(nint6)) {
        throw std::runtime_error(
            "mixed NINT4/NINT6 SwiGLU weights were accepted");
    }

    bool rejected_multirow = false;
    try {
        const std::vector<float> multirow_values(
            2 * static_cast<std::size_t>(input_size),
            0.25f);
        const array multirow(
            multirow_values.begin(),
            Shape{2, input_size});
        auto invalid = gate.swiglu(up, multirow);
        invalid.eval();
    } catch (const std::runtime_error&) {
        rejected_multirow = true;
    }
    if (!rejected_multirow) {
        throw std::runtime_error(
            "multi-row fused NINT SwiGLU input was accepted");
    }
}

} // namespace

int main() {
    try {
        using namespace mlx::core;
        if (!mfq::metal::is_nint_dtype("NINT")) {
            throw std::runtime_error("legacy NINT dtype was rejected");
        }
        for (int bits = 1; bits <= 8; ++bits) {
            if (!mfq::metal::is_nint_dtype(
                    "NINT" + std::to_string(bits))) {
                throw std::runtime_error(
                    "NINT dtype was rejected for bit width " +
                    std::to_string(bits));
            }
        }
        for (const char* invalid : {
                 "", "NIN", "NINT0", "NINT9", "NINTM",
                 "NINT10", "NINT4-24", "nint4",
             }) {
            if (mfq::metal::is_nint_dtype(invalid)) {
                throw std::runtime_error(
                    std::string("invalid NINT dtype was accepted: ") +
                    invalid);
            }
        }

        constexpr int input_size = 9;
        constexpr int packed_row_size = 10;
        const std::vector<float> input_values{
            1.0f, -1.0f, 2.0f, -2.0f, 3.0f,
            -3.0f, 4.0f, -4.0f, 5.0f,
            -1.0f, 2.0f, -3.0f, 4.0f, -5.0f,
            6.0f, -7.0f, 8.0f, -9.0f,
        };
        const array input(
            input_values.begin(),
            Shape{2, input_size});
        for (int bits = 1; bits <= 8; ++bits) {
            const auto fixture = make_nint_blob(bits);
            const auto weight =
                mfq::metal::MlxNintWeight::from_blob(fixture.blob);
            auto output = weight.matmul(input);
            output.eval();
            const auto* values = output.data<float>();
            for (int input_row = 0; input_row < 2; ++input_row) {
                for (int output_row = 0; output_row < 2; ++output_row) {
                    float expected = 0.0f;
                    for (int index = 0; index < input_size; ++index) {
                        expected +=
                            input_values[input_row * input_size + index] *
                            fixture.quantized[
                                output_row * packed_row_size + index];
                    }
                    require_close(
                        values[input_row * 2 + output_row],
                        expected);
                }
            }

            std::vector<float> large_input_values(64 * input_size);
            for (std::size_t index = 0;
                 index < large_input_values.size();
                 ++index) {
                large_input_values[index] =
                    static_cast<float>(
                        static_cast<int>(index % input_size) - 4) /
                    256.0f;
            }
            const array large_input(
                large_input_values.begin(),
                Shape{64, input_size});
            auto large_output = astype(
                weight.matmul(astype(large_input, float16)),
                float32);
            large_output.eval();
            const auto* large_values = large_output.data<float>();
            for (int input_row = 0; input_row < 64; ++input_row) {
                for (int output_row = 0; output_row < 2; ++output_row) {
                    float expected = 0.0f;
                    for (int index = 0; index < input_size; ++index) {
                        expected +=
                            large_input_values[
                                input_row * input_size + index] *
                            fixture.quantized[
                                output_row * packed_row_size + index];
                    }
                    require_close(
                        large_values[input_row * 2 + output_row],
                        expected,
                        0.02f);
                }
            }

            const array token_ids({1, 0}, Shape{2}, int32);
            auto embeddings = weight.embedding(token_ids, float32);
            embeddings.eval();
            const auto* embedded = embeddings.data<float>();
            for (int token = 0; token < 2; ++token) {
                const int source_row = token == 0 ? 1 : 0;
                for (int index = 0; index < input_size; ++index) {
                    require_close(
                        embedded[token * input_size + index],
                        fixture.quantized[
                            source_row * packed_row_size + index]);
                }
            }
        }
        test_nint5_gs28_decode();
        test_nint4_gs24_decode();
        test_nint6_gs24_decode();
        test_nint4_swiglu();
        std::cout
            << "MFQ C++ NINT1-NINT8 matmul/embedding and "
               "NINT4/NINT6 GS24 and NINT5 GS28 decode and "
               "NINT4 SwiGLU "
               "Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
