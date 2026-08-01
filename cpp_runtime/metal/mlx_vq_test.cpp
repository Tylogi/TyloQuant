#include "mlx_vq.h"
#include "mlx_tensor.h"

#include "../nvq_codebooks.generated.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace {

constexpr int kMatrixOutput = 3;
constexpr int kInputSize = 24;

template <typename T>
void append(
    std::vector<std::uint8_t>& blob,
    T value) {
    const auto* bytes =
        reinterpret_cast<const std::uint8_t*>(&value);
    blob.insert(
        blob.end(),
        bytes,
        bytes + sizeof(value));
}

void append_magic(
    std::vector<std::uint8_t>& blob,
    std::string_view magic) {
    if (magic.size() != 4) {
        throw std::runtime_error("fixture magic must have four bytes");
    }
    blob.insert(
        blob.end(),
        magic.begin(),
        magic.end());
}

void append_bytes(
    std::vector<std::uint8_t>& blob,
    const std::vector<std::uint8_t>& bytes) {
    blob.insert(
        blob.end(),
        bytes.begin(),
        bytes.end());
}

std::vector<std::uint8_t> pack_values(
    const std::vector<std::uint16_t>& values,
    int bits) {
    if (bits == 0) {
        if (!values.empty()) {
            throw std::runtime_error(
                "zero-bit fixture stream must be empty");
        }
        return {};
    }
    std::vector<std::uint8_t> result(
        (
            values.size() * static_cast<std::size_t>(bits)
            + 7
        ) / 8,
        0);
    for (std::size_t index = 0;
         index < values.size();
         ++index) {
        if (values[index] >= (1u << bits)) {
            throw std::runtime_error(
                "fixture value does not fit its bit stream");
        }
        for (int bit = 0; bit < bits; ++bit) {
            if (((values[index] >> bit) & 1u) == 0) {
                continue;
            }
            const auto destination =
                index * static_cast<std::size_t>(bits)
                + static_cast<std::size_t>(bit);
            result[destination / 8] |=
                static_cast<std::uint8_t>(
                    1u << (destination & 7));
        }
    }
    return result;
}

std::vector<std::uint16_t> filled_values(
    std::size_t count,
    std::uint16_t value) {
    return std::vector<std::uint16_t>(count, value);
}

void append_matrix_header(
    std::vector<std::uint8_t>& blob,
    std::string_view magic,
    std::uint8_t profile,
    std::uint8_t state_bits,
    std::uint16_t group_size,
    int output_size,
    int input_size) {
    append_magic(blob, magic);
    append<std::uint8_t>(blob, profile);
    append<std::uint8_t>(blob, state_bits);
    append<std::uint16_t>(blob, group_size);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input_size);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, input_size);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(output_size));
}

void append_anchors(
    std::vector<std::uint8_t>& blob,
    int count) {
    for (int index = 0; index < count; ++index) {
        append<std::uint16_t>(blob, 0x3c00);
    }
}

void append_matrix_streams(
    std::vector<std::uint8_t>& blob,
    int output_size,
    int input_size,
    int group_size,
    int vector_size,
    int state_bits,
    int index_bits,
    int auxiliary_bits,
    bool signs) {
    append_anchors(blob, output_size);
    const int groups =
        (input_size + group_size - 1) / group_size;
    const int vectors =
        (input_size + vector_size - 1) / vector_size;
    append_bytes(
        blob,
        pack_values(
            filled_values(
                static_cast<std::size_t>(output_size)
                    * groups,
                1),
            state_bits));
    append_bytes(
        blob,
        pack_values(
            filled_values(
                static_cast<std::size_t>(output_size)
                    * vectors,
                0),
            index_bits));
    if (signs) {
        const int sign_groups = (input_size + 7) / 8;
        append_bytes(
            blob,
            pack_values(
                filled_values(
                    static_cast<std::size_t>(output_size)
                        * sign_groups,
                    0),
                7));
    } else if (auxiliary_bits != 0) {
        append_bytes(
            blob,
            pack_values(
                filled_values(
                    static_cast<std::size_t>(output_size)
                        * groups,
                    0),
                auxiliary_bits));
    }
}

struct Fixture {
    std::string dtype;
    std::vector<std::uint8_t> blob;
    std::vector<std::uint8_t> runtime_payload;
    std::vector<float> dense;
    std::vector<int> output_shape;
    std::vector<std::int8_t> rotation_signs;
    int rotation_block = 0;
    int output_size = 0;
    int input_size = 0;
};

std::vector<float> repeated_dense(
    int output_size,
    int input_size,
    const std::vector<float>& vector) {
    if (vector.empty() ||
        input_size % static_cast<int>(vector.size()) != 0) {
        throw std::runtime_error(
            "invalid fixture vector repetition");
    }
    std::vector<float> result(
        static_cast<std::size_t>(output_size)
            * input_size);
    for (int output = 0;
         output < output_size;
         ++output) {
        for (int column = 0;
             column < input_size;
             ++column) {
            result[
                static_cast<std::size_t>(output)
                    * input_size
                + column
            ] = vector[
                static_cast<std::size_t>(column)
                % vector.size()
            ];
        }
    }
    return result;
}

Fixture make_plain_nvq(
    std::string dtype,
    int codebook_id,
    int vector_size) {
    constexpr std::uint8_t kCustomCodebook = 0x40;
    constexpr int state_bits = 4;
    constexpr int group_size = 24;
    constexpr int index_bits = 8;
    constexpr int entries = 256;
    std::vector<std::uint8_t> blob;
    append_matrix_header(
        blob,
        "NVQ1",
        static_cast<std::uint8_t>(
            codebook_id | kCustomCodebook),
        state_bits,
        group_size,
        kMatrixOutput,
        kInputSize);
    for (int entry = 0; entry < entries; ++entry) {
        // A unique, valid packed odd-lattice entry.  Index zero is all +1.
        append<std::uint16_t>(
            blob,
            static_cast<std::uint16_t>(entry));
    }
    append_matrix_streams(
        blob,
        kMatrixOutput,
        kInputSize,
        group_size,
        vector_size,
        state_bits,
        index_bits,
        7,
        true);
    return {
        std::move(dtype),
        std::move(blob),
        {},
        repeated_dense(
            kMatrixOutput,
            kInputSize,
            std::vector<float>(
                static_cast<std::size_t>(vector_size),
                1.0f)),
        {kMatrixOutput},
        {},
        0,
        kMatrixOutput,
        kInputSize,
    };
}

Fixture make_jsc(
    std::string dtype,
    int codebook_id,
    int vector_size,
    int index_bits) {
    constexpr std::uint8_t kJsc = 0x20;
    constexpr int entries_base = 1;
    const int entries = entries_base << index_bits;
    std::vector<std::uint8_t> blob;
    append_matrix_header(
        blob,
        "NVQ1",
        static_cast<std::uint8_t>(codebook_id | kJsc),
        4,
        24,
        kMatrixOutput,
        kInputSize);
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, 2);
    append<std::uint8_t>(blob, 16);
    append<std::uint8_t>(blob, 0);
    for (int state = 0; state < 16; ++state) {
        append<std::uint16_t>(blob, 0x3c00);
    }
    for (int state = 0; state < 16; ++state) {
        append<std::uint8_t>(
            blob,
            static_cast<std::uint8_t>(state & 1));
    }
    blob.insert(blob.end(), 12, 0);
    for (int bank = 0; bank < 2; ++bank) {
        for (int entry = 0; entry < entries; ++entry) {
            for (int component = 0;
                 component < vector_size;
                 ++component) {
                append<std::int8_t>(
                    blob,
                    static_cast<std::int8_t>(bank + 1));
            }
        }
    }
    append_matrix_streams(
        blob,
        kMatrixOutput,
        kInputSize,
        24,
        vector_size,
        4,
        index_bits,
        7,
        true);
    return {
        std::move(dtype),
        std::move(blob),
        {},
        repeated_dense(
            kMatrixOutput,
            kInputSize,
            std::vector<float>(
                static_cast<std::size_t>(vector_size),
                2.0f)),
        {kMatrixOutput},
        {},
        0,
        kMatrixOutput,
        kInputSize,
    };
}

std::vector<float> decode_ternary_word(
    std::uint16_t word) {
    std::vector<float> result(8);
    for (int component = 0; component < 8; ++component) {
        const auto digit =
            (word >> (2 * component)) & 3u;
        if (digit > 2) {
            throw std::runtime_error(
                "invalid generated ternary fixture word");
        }
        result[component] =
            static_cast<float>(
                static_cast<int>(digit) - 1);
    }
    return result;
}

Fixture make_nvq1_l() {
    std::vector<std::uint8_t> blob;
    append_matrix_header(
        blob,
        "NQ1L",
        1,
        3,
        24,
        kMatrixOutput,
        kInputSize);
    append_matrix_streams(
        blob,
        kMatrixOutput,
        kInputSize,
        24,
        8,
        3,
        11,
        1,
        false);
    auto vector = decode_ternary_word(
        mfq::nvq_codebooks::kNvq1LCodebookPacked[0]);
    for (auto& value : vector) {
        value += 0.125f;
    }
    return {
        "NVQ1-L",
        std::move(blob),
        {},
        repeated_dense(
            kMatrixOutput,
            kInputSize,
            vector),
        {kMatrixOutput},
        {},
        0,
        kMatrixOutput,
        kInputSize,
    };
}

void append_nvq1_s_table(
    std::vector<std::uint8_t>& blob,
    bool reverse = false) {
    for (int bank = 0; bank < 2; ++bank) {
        for (int entry = 0; entry < 512; ++entry) {
            const int source = reverse
                ? (511 - entry) * 4
                : entry * 4;
            append<std::uint16_t>(
                blob,
                mfq::nvq_codebooks::
                    kNvq1LCodebookPacked[source]);
        }
    }
}

Fixture make_nvq1_s() {
    std::vector<std::uint8_t> blob;
    append_matrix_header(
        blob,
        "NQ1S",
        1,
        4,
        24,
        kMatrixOutput,
        kInputSize);
    append_nvq1_s_table(blob);
    append_matrix_streams(
        blob,
        kMatrixOutput,
        kInputSize,
        24,
        8,
        4,
        9,
        1,
        false);
    auto vector = decode_ternary_word(
        mfq::nvq_codebooks::kNvq1LCodebookPacked[0]);
    for (auto& value : vector) {
        value += 0.15625f;
    }
    return {
        "NVQ1-S",
        std::move(blob),
        {},
        repeated_dense(
            kMatrixOutput,
            kInputSize,
            vector),
        {kMatrixOutput},
        {},
        0,
        kMatrixOutput,
        kInputSize,
    };
}

std::vector<std::uint8_t> npq_table(
    bool short_profile,
    int bank) {
    const int states = short_profile ? 4 : 8;
    const int first_entries = 8;
    const int second_entries = short_profile ? 8 : 16;
    const std::size_t bytes =
        short_profile ? 320 : 832;
    std::vector<std::uint8_t> table(bytes, 0);
    table[0] = static_cast<std::uint8_t>(
        short_profile ? 2 : 1);
    table[1] = static_cast<std::uint8_t>(states);
    table[2] = 3;
    table[3] = static_cast<std::uint8_t>(
        short_profile ? 3 : 4);
    table[4] = 24;
    table[5] = 8;
    for (int state = 0; state < states; ++state) {
        const std::uint16_t one = 0x3c00;
        std::memcpy(
            table.data() + 8 + state * 2,
            &one,
            sizeof(one));
    }
    const auto first_count =
        states * first_entries * 4;
    const auto second_count =
        states * second_entries * 4;
    std::fill_n(
        table.begin() + 64,
        first_count,
        static_cast<std::uint8_t>(1 + bank * 2));
    std::fill_n(
        table.begin() + 64 + first_count,
        second_count,
        static_cast<std::uint8_t>(2 + bank * 2));
    return table;
}

Fixture make_npq(bool short_profile) {
    std::vector<std::uint8_t> blob;
    append_matrix_header(
        blob,
        short_profile ? "NPQS" : "NPQL",
        static_cast<std::uint8_t>(
            short_profile ? 2 : 1),
        static_cast<std::uint8_t>(
            short_profile ? 2 : 3),
        24,
        kMatrixOutput,
        kInputSize);
    append_bytes(
        blob,
        npq_table(short_profile, 0));
    append_matrix_streams(
        blob,
        kMatrixOutput,
        kInputSize,
        24,
        8,
        short_profile ? 2 : 3,
        short_profile ? 6 : 7,
        0,
        false);
    return {
        short_profile ? "NPQ0-S" : "NPQ0-L",
        std::move(blob),
        {},
        repeated_dense(
            kMatrixOutput,
            kInputSize,
            {1.0f, 1.0f, 1.0f, 1.0f,
             2.0f, 2.0f, 2.0f, 2.0f}),
        {kMatrixOutput},
        {},
        0,
        kMatrixOutput,
        kInputSize,
    };
}

void append_nepq_table(
    std::vector<std::uint8_t>& blob,
    int profile,
    int bank) {
    if (profile == 0 || profile == 1) {
        append_bytes(
            blob,
            npq_table(profile == 0, bank));
        return;
    }
    if (profile == 2) {
        append_nvq1_s_table(blob, bank != 0);
        return;
    }
    for (int entry = 0; entry < 2048; ++entry) {
        const int source = bank == 0
            ? entry
            : 2047 - entry;
        append<std::uint16_t>(
            blob,
            mfq::nvq_codebooks::
                kNvq1LCodebookPacked[source]);
    }
}

std::vector<float> nepq_vector(
    int profile,
    int bank) {
    if (profile == 0 || profile == 1) {
        return {
            static_cast<float>(1 + bank * 2),
            static_cast<float>(1 + bank * 2),
            static_cast<float>(1 + bank * 2),
            static_cast<float>(1 + bank * 2),
            static_cast<float>(2 + bank * 2),
            static_cast<float>(2 + bank * 2),
            static_cast<float>(2 + bank * 2),
            static_cast<float>(2 + bank * 2),
        };
    }
    const int source = profile == 2
        ? (bank == 0 ? 0 : 511 * 4)
        : (bank == 0 ? 0 : 2047);
    auto result = decode_ternary_word(
        mfq::nvq_codebooks::
            kNvq1LCodebookPacked[source]);
    const float delta =
        profile == 2 ? 0.15625f : 0.125f;
    for (auto& value : result) {
        value += delta;
    }
    return result;
}

Fixture make_nepq(
    int profile,
    bool rotated = false) {
    constexpr int n_experts = 2;
    constexpr int out_per_expert = 2;
    constexpr int output_size =
        n_experts * out_per_expert;
    constexpr int table_banks = 2;
    constexpr std::array<const char*, 4> labels{
        "NEPQ0-S",
        "NEPQ0-L",
        "NEPQ1-S",
        "NEPQ1-L",
    };
    constexpr std::array<int, 4> state_bits{
        2, 3, 4, 3,
    };
    constexpr std::array<int, 4> index_bits{
        6, 7, 9, 11,
    };
    constexpr std::array<int, 4> auxiliary_bits{
        0, 0, 1, 1,
    };
    const int rotation_block = rotated ? 8 : 0;
    const std::uint64_t rotation_seed =
        rotated ? 0x123456789abcdef0ull : 0;

    std::vector<std::uint8_t> blob;
    append_magic(blob, "NEP1");
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(
        blob,
        static_cast<std::uint8_t>(profile));
    append<std::uint8_t>(blob, 4);
    append<std::uint8_t>(
        blob,
        static_cast<std::uint8_t>(rotated ? 1 : 0));
    append<std::uint32_t>(blob, n_experts);
    append<std::uint32_t>(blob, out_per_expert);
    append<std::uint32_t>(blob, kInputSize);
    append<std::uint32_t>(blob, table_banks);
    append<std::uint32_t>(blob, rotation_block);
    append<std::uint64_t>(blob, rotation_seed);
    for (int bank = 0; bank < table_banks; ++bank) {
        append_nepq_table(
            blob,
            profile,
            bank);
    }
    append_anchors(blob, output_size);
    append_bytes(
        blob,
        pack_values(
            filled_values(output_size, 1),
            state_bits[profile]));
    append_bytes(
        blob,
        pack_values(
            filled_values(
                static_cast<std::size_t>(output_size)
                    * (kInputSize / 8),
                0),
            index_bits[profile]));
    if (auxiliary_bits[profile] != 0) {
        append_bytes(
            blob,
            pack_values(
                filled_values(output_size, 0),
                auxiliary_bits[profile]));
    }
    for (int output = 0;
         output < output_size;
         ++output) {
        append<std::uint8_t>(
            blob,
            static_cast<std::uint8_t>(output & 1));
    }

    std::vector<float> dense(
        static_cast<std::size_t>(output_size)
            * kInputSize);
    for (int output = 0;
         output < output_size;
         ++output) {
        const auto vector =
            nepq_vector(profile, output & 1);
        for (int column = 0;
             column < kInputSize;
             ++column) {
            dense[
                static_cast<std::size_t>(output)
                    * kInputSize
                + column
            ] = vector[
                static_cast<std::size_t>(column)
                % vector.size()
            ];
        }
    }

    std::vector<std::uint8_t> runtime;
    std::vector<std::int8_t> signs;
    if (rotated) {
        append_magic(runtime, "HSG1");
        append<std::uint32_t>(runtime, kInputSize);
        append<std::uint32_t>(runtime, rotation_block);
        append<std::uint64_t>(runtime, rotation_seed);
        signs.resize(kInputSize);
        for (int column = 0;
             column < kInputSize;
             ++column) {
            signs[column] = static_cast<std::int8_t>(
                (column % 3) == 0 ? -1 : 1);
            append<std::int8_t>(
                runtime,
                signs[column]);
        }
    }
    return {
        labels[profile],
        std::move(blob),
        std::move(runtime),
        std::move(dense),
        {n_experts, out_per_expert},
        std::move(signs),
        rotation_block,
        output_size,
        kInputSize,
    };
}

std::vector<Fixture> fixtures() {
    std::vector<Fixture> result;
    result.push_back(make_plain_nvq("NVQ2", 1, 8));
    result.push_back(make_jsc("NVQ2J", 1, 8, 8));
    result.push_back(make_jsc("NVQ2J-L", 4, 8, 10));
    result.push_back(make_jsc("NVQ2J-XL", 5, 8, 12));
    result.push_back(make_plain_nvq("NVQ3", 2, 4));
    result.push_back(make_jsc("NVQ3J", 2, 4, 8));
    result.push_back(make_jsc("NVQ3J-512", 3, 4, 9));
    result.push_back(make_jsc("NVQ3J-L", 6, 4, 10));
    result.push_back(make_nvq1_l());
    result.push_back(make_nvq1_s());
    result.push_back(make_npq(false));
    result.push_back(make_npq(true));
    result.push_back(make_nepq(1));
    result.push_back(make_nepq(0));
    result.push_back(make_nepq(3));
    result.push_back(make_nepq(2));
    return result;
}

void require_close(
    float actual,
    float expected,
    float tolerance,
    std::string_view context) {
    if (!std::isfinite(actual) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            std::string(context)
            + " mismatch: actual="
            + std::to_string(actual)
            + " expected="
            + std::to_string(expected));
    }
}

void require_shape(
    const mlx::core::array& value,
    const std::vector<int>& expected,
    std::string_view context) {
    const auto& actual = value.shape();
    if (actual.size() != expected.size() ||
        !std::equal(
            actual.begin(),
            actual.end(),
            expected.begin())) {
        throw std::runtime_error(
            std::string(context)
            + " has the wrong output shape");
    }
}

std::vector<float> evaluated_float(
    mlx::core::array value) {
    using namespace mlx::core;
    if (value.dtype() != float32) {
        value = astype(value, float32);
    }
    value.eval();
    const auto* data = value.data<float>();
    return std::vector<float>(
        data,
        data + static_cast<std::ptrdiff_t>(value.size()));
}

std::vector<float> input_values(
    int rows,
    int input_size) {
    std::vector<float> result(
        static_cast<std::size_t>(rows)
            * input_size);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0;
             column < input_size;
             ++column) {
            result[
                static_cast<std::size_t>(row)
                    * input_size
                + column
            ] = static_cast<float>(
                ((row * 3 + column * 5) % 9)
                - 4) / 8.0f;
        }
    }
    return result;
}

void signed_hadamard_cpu(
    std::vector<float>& values,
    int rows,
    int width,
    const std::vector<std::int8_t>& signs,
    int block) {
    if (block == 0) {
        return;
    }
    for (int row = 0; row < rows; ++row) {
        for (int start = 0; start < width; start += block) {
            for (int index = 0; index < block; ++index) {
                values[
                    static_cast<std::size_t>(row)
                        * width
                    + start + index
                ] *= signs[
                    static_cast<std::size_t>(start + index)
                ];
            }
            for (int stride = 1;
                 stride < block;
                 stride <<= 1) {
                for (int base = 0;
                     base < block;
                     base += stride * 2) {
                    for (int offset = 0;
                         offset < stride;
                         ++offset) {
                        const auto first_index =
                            static_cast<std::size_t>(row)
                                * width
                            + start + base + offset;
                        const auto second_index =
                            first_index + stride;
                        const float first = values[first_index];
                        const float second = values[second_index];
                        values[first_index] = first + second;
                        values[second_index] = first - second;
                    }
                }
            }
            const float inverse =
                1.0f / std::sqrt(static_cast<float>(block));
            for (int index = 0; index < block; ++index) {
                values[
                    static_cast<std::size_t>(row)
                        * width
                    + start + index
                ] *= inverse;
            }
        }
    }
}

std::vector<float> expected_matmul(
    const Fixture& fixture,
    std::vector<float> input,
    int rows) {
    signed_hadamard_cpu(
        input,
        rows,
        fixture.input_size,
        fixture.rotation_signs,
        fixture.rotation_block);
    std::vector<float> result(
        static_cast<std::size_t>(rows)
            * fixture.output_size,
        0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int output = 0;
             output < fixture.output_size;
             ++output) {
            float total = 0.0f;
            for (int column = 0;
                 column < fixture.input_size;
                 ++column) {
                total +=
                    input[
                        static_cast<std::size_t>(row)
                            * fixture.input_size
                        + column
                    ]
                    * fixture.dense[
                        static_cast<std::size_t>(output)
                            * fixture.input_size
                        + column
                    ];
            }
            result[
                static_cast<std::size_t>(row)
                    * fixture.output_size
                + output
            ] = total;
        }
    }
    return result;
}

std::vector<int> matmul_shape(
    int rows,
    const Fixture& fixture,
    bool vector_input) {
    std::vector<int> result;
    if (!vector_input) {
        result.push_back(rows);
    }
    result.insert(
        result.end(),
        fixture.output_shape.begin(),
        fixture.output_shape.end());
    return result;
}

template <typename Function>
void require_throws(
    Function&& function,
    std::string_view context) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(
        std::string(context)
        + " did not reject malformed input");
}

void check_values(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance,
    std::string_view context) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(
            std::string(context)
            + " produced the wrong element count");
    }
    for (std::size_t index = 0;
         index < actual.size();
         ++index) {
        require_close(
            actual[index],
            expected[index],
            tolerance,
            context);
    }
}

void test_fixture(const Fixture& fixture) {
    using namespace mlx::core;
    const auto metadata =
        mfq::metal::inspect_vq_blob(
            fixture.dtype,
            fixture.blob,
            fixture.runtime_payload);
    if (metadata.format_label != fixture.dtype ||
        metadata.input_size != fixture.input_size ||
        metadata.output_size != fixture.output_size ||
        metadata.output_shape != fixture.output_shape ||
        metadata.rotation_block != fixture.rotation_block) {
        throw std::runtime_error(
            fixture.dtype
            + " header-only metadata mismatch");
    }
    const auto weight =
        mfq::metal::MlxVqWeight::from_blob(
            fixture.dtype,
            fixture.blob,
            fixture.runtime_payload);
    if (weight.format_label() != fixture.dtype ||
        weight.input_size() != fixture.input_size ||
        weight.output_size() != fixture.output_size ||
        weight.output_shape() != fixture.output_shape ||
        weight.packed_nbytes() == 0) {
        throw std::runtime_error(
            fixture.dtype
            + " native metadata mismatch");
    }

    auto dense = weight.dequantize(float32);
    require_shape(
        dense,
        {fixture.output_size, fixture.input_size},
        fixture.dtype + " dequantize");
    check_values(
        evaluated_float(std::move(dense)),
        fixture.dense,
        1e-5f,
        fixture.dtype + " dequantize");

    constexpr std::array<int, 3> packed_rows{
        1, 4, 17,
    };
    for (const int rows : packed_rows) {
        auto input = input_values(
            rows,
            fixture.input_size);
        const auto expected = expected_matmul(
            fixture,
            input,
            rows);
        array input_array(
            input.begin(),
            rows == 1
                ? Shape{fixture.input_size}
                : Shape{rows, fixture.input_size});
        array output = rows == 1
            ? weight.gemv(input_array)
            : (
                rows <= 16
                ? weight.mmq(input_array)
                : weight.gemm(input_array)
            );
        require_shape(
            output,
            matmul_shape(
                rows,
                fixture,
                rows == 1),
            fixture.dtype + " packed matmul");
        check_values(
            evaluated_float(std::move(output)),
            expected,
            3e-4f,
            fixture.dtype + " packed matmul");
    }

    constexpr int large_rows = 64;
    auto large_input = input_values(
        large_rows,
        fixture.input_size);
    const auto large_expected = expected_matmul(
        fixture,
        large_input,
        large_rows);
    const array large_source(
        large_input.begin(),
        Shape{large_rows, fixture.input_size});
    auto large_output = weight.matmul(
        astype(large_source, float16));
    require_shape(
        large_output,
        matmul_shape(
            large_rows,
            fixture,
            false),
        fixture.dtype + " dense large-M");
    check_values(
        evaluated_float(std::move(large_output)),
        large_expected,
        0.08f,
        fixture.dtype + " dense large-M");

    if (fixture.output_shape.size() == 1 &&
        fixture.rotation_block == 0) {
        const mfq::metal::MlxLinear linear(
            mfq::metal::MlxVqWeight::from_blob(
                fixture.dtype,
                fixture.blob,
                fixture.runtime_payload));
        const auto integration_input = input_values(
            2,
            fixture.input_size);
        auto projected = linear(
            array(
                integration_input.begin(),
                Shape{2, fixture.input_size}));
        check_values(
            evaluated_float(std::move(projected)),
            expected_matmul(
                fixture,
                integration_input,
                2),
            3e-4f,
            fixture.dtype + " MlxLinear integration");

        const mfq::metal::MlxEmbedding embedding(
            mfq::metal::MlxVqWeight::from_blob(
                fixture.dtype,
                fixture.blob,
                fixture.runtime_payload));
        const array ids(
            {fixture.output_size - 1, 0},
            Shape{2},
            int32);
        auto embedded = embedding(ids, float32);
        require_shape(
            embedded,
            {2, fixture.input_size},
            fixture.dtype + " embedding");
        auto tied = embedding.project(
            array(
                integration_input.begin(),
                Shape{2, fixture.input_size}));
        check_values(
            evaluated_float(std::move(tied)),
            expected_matmul(
                fixture,
                integration_input,
                2),
            3e-4f,
            fixture.dtype + " tied embedding projection");
        std::vector<float> expected;
        for (const int output : {
                 fixture.output_size - 1,
                 0,
             }) {
            expected.insert(
                expected.end(),
                fixture.dense.begin()
                    + static_cast<std::ptrdiff_t>(
                        output * fixture.input_size),
                fixture.dense.begin()
                    + static_cast<std::ptrdiff_t>(
                        (output + 1) * fixture.input_size));
        }
        check_values(
            evaluated_float(std::move(embedded)),
            expected,
            1e-5f,
            fixture.dtype + " embedding");
    } else {
        require_throws(
            [&]() {
                (void)weight.embedding(
                    array({0}, Shape{1}, int32),
                    float32);
            },
            fixture.dtype + " embedding");
    }
}

} // namespace

int main() {
    try {
        constexpr std::array<std::string_view, 16> public_dtypes{
            "NVQ2",
            "NVQ2J",
            "NVQ2J-L",
            "NVQ2J-XL",
            "NVQ3",
            "NVQ3J",
            "NVQ3J-512",
            "NVQ3J-L",
            "NVQ1-L",
            "NVQ1-S",
            "NPQ0-L",
            "NPQ0-S",
            "NEPQ0-L",
            "NEPQ0-S",
            "NEPQ1-L",
            "NEPQ1-S",
        };
        for (const auto dtype : public_dtypes) {
            if (!mfq::metal::is_vq_dtype(dtype)) {
                throw std::runtime_error(
                    "public native VQ dtype was rejected: "
                    + std::string(dtype));
            }
        }
        for (const auto invalid : {
                 "",
                 "NVQ",
                 "NVQ4",
                 "NPQ",
                 "NEPQ2-S",
                 "nvq2",
             }) {
            if (mfq::metal::is_vq_dtype(invalid)) {
                throw std::runtime_error(
                    "invalid native VQ dtype was accepted: "
                    + std::string(invalid));
            }
        }

        auto all = fixtures();
        if (all.size() != public_dtypes.size()) {
            throw std::runtime_error(
                "native VQ test matrix is incomplete");
        }
        for (const auto& fixture : all) {
            test_fixture(fixture);
        }

        auto rotated = make_nepq(0, true);
        test_fixture(rotated);
        require_throws(
            [&]() {
                (void)mfq::metal::inspect_vq_blob(
                    rotated.dtype,
                    rotated.blob);
            },
            "rotated NEPQ metadata missing HSG1");
        require_throws(
            [&]() {
                (void)mfq::metal::MlxVqWeight::from_blob(
                    rotated.dtype,
                    rotated.blob);
            },
            "rotated NEPQ missing HSG1");
        auto corrupt_runtime = rotated.runtime_payload;
        corrupt_runtime.back() = 0;
        require_throws(
            [&]() {
                (void)mfq::metal::MlxVqWeight::from_blob(
                    rotated.dtype,
                    rotated.blob,
                    corrupt_runtime);
            },
            "rotated NEPQ corrupt sign");

        auto wrong_dtype = all.front();
        require_throws(
            [&]() {
                (void)mfq::metal::inspect_vq_blob(
                    "NVQ3",
                    wrong_dtype.blob);
            },
            "VQ metadata dtype/blob mismatch");
        require_throws(
            [&]() {
                (void)mfq::metal::MlxVqWeight::from_blob(
                    "NVQ3",
                    wrong_dtype.blob);
            },
            "VQ dtype/blob mismatch");
        auto trailing = all.front();
        trailing.blob.push_back(0);
        require_throws(
            [&]() {
                (void)mfq::metal::MlxVqWeight::from_blob(
                    trailing.dtype,
                    trailing.blob);
            },
            "VQ trailing bytes");

        std::cout
            << "MFQ native C++/MLX VQ 16-dtype "
               "dequant/embedding/GEMV/MMQ/GEMM tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
