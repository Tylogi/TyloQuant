#include "mlx_moe.h"
#include "mfq_container.h"

#include "../nvq_codebooks.generated.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace {

constexpr int kGroupSize = 16;

template <typename T>
void append(
    std::vector<std::uint8_t>& blob,
    T value) {
    const auto* bytes =
        reinterpret_cast<const std::uint8_t*>(
            &value);
    blob.insert(
        blob.end(),
        bytes,
        bytes + sizeof(value));
}

std::vector<std::uint8_t> pack_values(
    const std::vector<std::uint8_t>& values,
    int bits) {
    std::vector<std::uint8_t> packed(
        (
            values.size()
                * static_cast<std::size_t>(bits)
            + 7
        ) / 8,
        0);
    for (
        std::size_t index = 0;
        index < values.size();
        ++index
    ) {
        for (int bit = 0; bit < bits; ++bit) {
            if (((values[index] >> bit) & 1u) == 0u) {
                continue;
            }
            const auto target =
                index * static_cast<std::size_t>(bits)
                + static_cast<std::size_t>(bit);
            packed[target / 8] |=
                static_cast<std::uint8_t>(
                    1u << (target & 7));
        }
    }
    return packed;
}

void append_magic(
    std::vector<std::uint8_t>& blob,
    std::string_view magic) {
    if (magic.size() != 4) {
        throw std::runtime_error(
            "fixture magic must have four bytes");
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

std::vector<std::uint8_t> pack_vq_values(
    const std::vector<std::uint16_t>& values,
    int bits) {
    if (bits == 0) {
        if (!values.empty()) {
            throw std::runtime_error(
                "zero-bit VQ stream must be empty");
        }
        return {};
    }
    std::vector<std::uint8_t> packed(
        (
            values.size()
                * static_cast<std::size_t>(bits)
            + 7
        ) / 8,
        0);
    for (
        std::size_t index = 0;
        index < values.size();
        ++index
    ) {
        if (values[index] >= (1u << bits)) {
            throw std::runtime_error(
                "VQ fixture value exceeds bit width");
        }
        for (int bit = 0; bit < bits; ++bit) {
            if (((values[index] >> bit) & 1u) == 0u) {
                continue;
            }
            const auto destination =
                index * static_cast<std::size_t>(bits)
                + static_cast<std::size_t>(bit);
            packed[destination / 8] |=
                static_cast<std::uint8_t>(
                    1u << (destination & 7));
        }
    }
    return packed;
}

std::vector<std::uint16_t> filled_vq_values(
    std::size_t count,
    std::uint16_t value) {
    return std::vector<std::uint16_t>(
        count,
        value);
}

void append_vq_matrix_header(
    std::vector<std::uint8_t>& blob,
    std::string_view magic,
    std::uint8_t profile,
    std::uint8_t state_bits,
    std::uint16_t group_size,
    int output,
    int input) {
    append_magic(blob, magic);
    append<std::uint8_t>(blob, profile);
    append<std::uint8_t>(blob, state_bits);
    append<std::uint16_t>(blob, group_size);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output);
    append<std::int64_t>(blob, input);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(output));
}

void append_vq_anchors(
    std::vector<std::uint8_t>& blob,
    int count) {
    for (int index = 0; index < count; ++index) {
        append<std::uint16_t>(blob, 0x3c00);
    }
}

void append_vq_matrix_streams(
    std::vector<std::uint8_t>& blob,
    int output,
    int input,
    int group_size,
    int vector_size,
    int state_bits,
    int index_bits,
    int auxiliary_bits,
    bool signs) {
    append_vq_anchors(blob, output);
    const int groups =
        (input + group_size - 1) / group_size;
    const int vectors =
        (input + vector_size - 1) / vector_size;
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(
                static_cast<std::size_t>(output)
                    * groups,
                1),
            state_bits));
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(
                static_cast<std::size_t>(output)
                    * vectors,
                0),
            index_bits));
    if (signs) {
        const int sign_groups = (input + 7) / 8;
        append_bytes(
            blob,
            pack_vq_values(
                filled_vq_values(
                    static_cast<std::size_t>(output)
                        * sign_groups,
                    0),
                7));
    } else if (auxiliary_bits != 0) {
        append_bytes(
            blob,
            pack_vq_values(
                filled_vq_values(
                    static_cast<std::size_t>(output)
                        * groups,
                    0),
                auxiliary_bits));
    }
}

std::vector<float> repeated_vq_dense(
    int output,
    int input,
    const std::vector<float>& vector) {
    std::vector<float> dense(
        static_cast<std::size_t>(output) * input);
    for (int row = 0; row < output; ++row) {
        for (int column = 0; column < input; ++column) {
            dense[
                static_cast<std::size_t>(row)
                    * input
                + column
            ] = vector[
                static_cast<std::size_t>(column)
                    % vector.size()];
        }
    }
    return dense;
}

struct VqFixture {
    std::string dtype;
    std::vector<std::uint8_t> blob;
    std::vector<std::uint8_t> runtime;
    std::vector<float> dense;
    std::vector<std::int8_t> signs;
    int rotation_block = 0;
    std::uint64_t rotation_seed = 0;
    int output = 0;
    int input = 0;
};

VqFixture make_plain_nvq(
    int output,
    int input,
    bool index_parity = false) {
    constexpr std::uint8_t custom_codebook = 0x40;
    constexpr std::uint8_t parity = 0x80;
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        "NVQ1",
        static_cast<std::uint8_t>(
            1
            | custom_codebook
            | (index_parity ? parity : 0)),
        4,
        24,
        output,
        input);
    for (int entry = 0; entry < 256; ++entry) {
        append<std::uint16_t>(
            blob,
            static_cast<std::uint16_t>(entry));
    }
    append_vq_matrix_streams(
        blob,
        output,
        input,
        24,
        8,
        4,
        8,
        7,
        true);
    return {
        "NVQ2",
        std::move(blob),
        {},
        repeated_vq_dense(
            output,
            input,
            std::vector<float>(8, 1.0f)),
        {},
        0,
        0,
        output,
        input,
    };
}

VqFixture make_jsc_nvq(
    int output,
    int input,
    std::string dtype = "NVQ2J",
    int codebook_id = 1,
    int vector_size = 8,
    int index_bits = 8) {
    constexpr std::uint8_t jsc = 0x20;
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        "NVQ1",
        static_cast<std::uint8_t>(
            codebook_id | jsc),
        4,
        24,
        output,
        input);
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
            static_cast<std::uint8_t>(
                state & 1));
    }
    blob.insert(blob.end(), 12, 0);
    for (int bank = 0; bank < 2; ++bank) {
        for (
            int entry = 0;
            entry < (1 << index_bits);
            ++entry
        ) {
            for (int component = 0;
                 component < vector_size;
                 ++component) {
                append<std::int8_t>(
                    blob,
                    static_cast<std::int8_t>(
                        bank + 1));
            }
        }
    }
    append_vq_matrix_streams(
        blob,
        output,
        input,
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
        repeated_vq_dense(
            output,
            input,
            std::vector<float>(
                static_cast<std::size_t>(
                    vector_size),
                2.0f)),
        {},
        0,
        0,
        output,
        input,
    };
}

std::vector<float> decode_ternary_word(
    std::uint16_t word) {
    std::vector<float> result(8);
    for (int component = 0;
         component < 8;
         ++component) {
        const auto digit =
            (word >> (2 * component)) & 3u;
        if (digit > 2) {
            throw std::runtime_error(
                "invalid ternary fixture word");
        }
        result[component] =
            static_cast<float>(
                static_cast<int>(digit) - 1);
    }
    return result;
}

void append_nvq1_s_table(
    std::vector<std::uint8_t>& blob,
    bool reverse = false) {
    for (int bank = 0; bank < 2; ++bank) {
        for (int entry = 0; entry < 512; ++entry) {
            const int source =
                reverse ? (511 - entry) * 4 : entry * 4;
            append<std::uint16_t>(
                blob,
                mfq::nvq_codebooks::
                    kNvq1LCodebookPacked[source]);
        }
    }
}

VqFixture make_nvq1_s(
    int output,
    int input) {
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        "NQ1S",
        1,
        4,
        24,
        output,
        input);
    append_nvq1_s_table(blob);
    append_vq_matrix_streams(
        blob,
        output,
        input,
        24,
        8,
        4,
        9,
        1,
        false);
    auto vector = decode_ternary_word(
        mfq::nvq_codebooks::
            kNvq1LCodebookPacked[0]);
    for (auto& value : vector) {
        value += 0.15625f;
    }
    return {
        "NVQ1-S",
        std::move(blob),
        {},
        repeated_vq_dense(output, input, vector),
        {},
        0,
        0,
        output,
        input,
    };
}

std::vector<std::uint8_t> npq_table(
    bool short_profile,
    int bank) {
    const int states = short_profile ? 4 : 8;
    const int first_entries = 8;
    const int second_entries =
        short_profile ? 8 : 16;
    const std::size_t bytes =
        short_profile ? 320 : 832;
    std::vector<std::uint8_t> table(bytes, 0);
    table[0] =
        static_cast<std::uint8_t>(
            short_profile ? 2 : 1);
    table[1] =
        static_cast<std::uint8_t>(states);
    table[2] = 3;
    table[3] =
        static_cast<std::uint8_t>(
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
    const int first_count =
        states * first_entries * 4;
    const int second_count =
        states * second_entries * 4;
    std::fill_n(
        table.begin() + 64,
        first_count,
        static_cast<std::uint8_t>(
            1 + bank * 2));
    std::fill_n(
        table.begin() + 64 + first_count,
        second_count,
        static_cast<std::uint8_t>(
            2 + bank * 2));
    return table;
}

VqFixture make_npq(
    int output,
    int input) {
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        "NPQL",
        1,
        3,
        24,
        output,
        input);
    append_bytes(blob, npq_table(false, 0));
    append_vq_matrix_streams(
        blob,
        output,
        input,
        24,
        8,
        3,
        7,
        0,
        false);
    return {
        "NPQ0-L",
        std::move(blob),
        {},
        repeated_vq_dense(
            output,
            input,
            {
                1.0f, 1.0f, 1.0f, 1.0f,
                2.0f, 2.0f, 2.0f, 2.0f,
            }),
        {},
        0,
        0,
        output,
        input,
    };
}

void append_nepq1_s_table(
    std::vector<std::uint8_t>& blob,
    int bank) {
    append_nvq1_s_table(blob, bank != 0);
}

std::vector<float> nepq1_s_vector(int bank) {
    auto vector = decode_ternary_word(
        mfq::nvq_codebooks::
            kNvq1LCodebookPacked[
                bank == 0 ? 0 : 511 * 4]);
    for (auto& value : vector) {
        value += 0.15625f;
    }
    return vector;
}

VqFixture make_rotated_nepq1_s(
    int output,
    int input,
    std::uint64_t seed =
        0x123456789abcdef0ull,
    int sign_shift = 0,
    int cohort_experts = 1) {
    constexpr int table_banks = 2;
    constexpr int rotation_block = 8;
    const int rows = cohort_experts * output;
    std::vector<std::uint8_t> blob;
    append_magic(blob, "NEP1");
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, 2);
    append<std::uint8_t>(blob, 4);
    append<std::uint8_t>(blob, 1);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(
            cohort_experts));
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, table_banks);
    append<std::uint32_t>(blob, rotation_block);
    append<std::uint64_t>(blob, seed);
    for (int bank = 0; bank < table_banks; ++bank) {
        append_nepq1_s_table(blob, bank);
    }
    append_vq_anchors(blob, rows);
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(rows, 1),
            4));
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(
                static_cast<std::size_t>(rows)
                    * (input / 8),
                0),
            9));
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(rows, 0),
            1));
    for (int row = 0; row < rows; ++row) {
        append<std::uint8_t>(
            blob,
            static_cast<std::uint8_t>(row & 1));
    }

    std::vector<float> dense(
        static_cast<std::size_t>(rows) * input);
    for (int row = 0; row < rows; ++row) {
        const auto vector =
            nepq1_s_vector(row & 1);
        for (int column = 0; column < input; ++column) {
            dense[
                static_cast<std::size_t>(row)
                    * input
                + column
            ] = vector[
                static_cast<std::size_t>(column)
                    % vector.size()];
        }
    }

    std::vector<std::uint8_t> runtime;
    append_magic(runtime, "HSG1");
    append<std::uint32_t>(runtime, input);
    append<std::uint32_t>(
        runtime,
        rotation_block);
    append<std::uint64_t>(runtime, seed);
    std::vector<std::int8_t> signs(input);
    for (int column = 0; column < input; ++column) {
        signs[column] = static_cast<std::int8_t>(
            ((column + sign_shift) % 3) == 0
                ? -1
                : 1);
        append<std::int8_t>(
            runtime,
            signs[column]);
    }
    return {
        "NEPQ1-S",
        std::move(blob),
        std::move(runtime),
        std::move(dense),
        std::move(signs),
        rotation_block,
        seed,
        rows,
        input,
    };
}

struct TensorFixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    int rows = 0;
    int columns = 0;
};

TensorFixture make_nint_tensor(
    int bits,
    int rows,
    int columns,
    int salt) {
    if (columns % kGroupSize != 0) {
        throw std::runtime_error(
            "test NINT width must be a multiple of 16");
    }
    const int groups = columns / kGroupSize;
    const auto metadata_count =
        static_cast<std::size_t>(rows) * groups;
    const auto value_count =
        metadata_count * kGroupSize;
    const auto maximum = (1u << bits) - 1u;

    std::vector<std::uint8_t> sub_scale(
        metadata_count);
    std::vector<std::uint8_t> sub_min(
        metadata_count);
    for (
        std::size_t index = 0;
        index < metadata_count;
        ++index
    ) {
        sub_scale[index] =
            static_cast<std::uint8_t>(
                1 + (index + salt) % 3);
        sub_min[index] =
            static_cast<std::uint8_t>(
                (index + salt) & 1u);
    }
    std::vector<std::uint8_t> quantized(
        value_count);
    for (
        std::size_t index = 0;
        index < value_count;
        ++index
    ) {
        quantized[index] =
            static_cast<std::uint8_t>(
                (
                    index
                        * static_cast<std::size_t>(
                            bits + 3)
                    + static_cast<std::size_t>(
                        salt * 5 + 1)
                ) % (maximum + 1));
    }

    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(
        blob,
        static_cast<std::uint8_t>(bits));
    append<std::uint8_t>(blob, 2);
    append<std::int32_t>(
        blob,
        kGroupSize);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, columns);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, rows);
    append<std::int64_t>(blob, columns);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(rows));
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(groups));

    // Exact FP16 powers of two: anchor scale=1/64, minimum=1/128.
    for (int row = 0; row < rows; ++row) {
        append<std::uint16_t>(blob, 0x2400);
    }
    for (int row = 0; row < rows; ++row) {
        append<std::uint16_t>(blob, 0x2000);
    }
    for (
        const auto* metadata :
        {&sub_scale, &sub_min}
    ) {
        const auto packed =
            pack_values(*metadata, 2);
        blob.insert(
            blob.end(),
            packed.begin(),
            packed.end());
    }
    const auto packed =
        pack_values(quantized, bits);
    blob.insert(
        blob.end(),
        packed.begin(),
        packed.end());

    std::vector<float> dense(
        static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        for (
            int column = 0;
            column < columns;
            ++column
        ) {
            const int group =
                column / kGroupSize;
            const auto metadata =
                static_cast<std::size_t>(row)
                    * groups
                + group;
            const auto value =
                metadata * kGroupSize
                + column % kGroupSize;
            dense[
                static_cast<std::size_t>(row)
                    * columns
                + column
            ] =
                static_cast<float>(
                    sub_scale[metadata])
                    * (1.0f / 64.0f)
                    * static_cast<float>(
                        quantized[value])
                - static_cast<float>(
                    sub_min[metadata])
                    * (1.0f / 128.0f);
        }
    }
    return {
        std::move(blob),
        std::move(dense),
        rows,
        columns,
    };
}

TensorFixture make_q8_tensor(
    int rows,
    int columns,
    int salt) {
    if (columns % 32 != 0) {
        throw std::runtime_error(
            "test Q8 width must be a multiple of 32");
    }
    constexpr std::uint16_t scale_bits[] = {
        0x2000,
        0x2400,
        0x2800,
        0x2c00,
    };
    constexpr float scale_values[] = {
        1.0f / 128.0f,
        1.0f / 64.0f,
        1.0f / 32.0f,
        1.0f / 16.0f,
    };
    const int groups = columns / 32;
    std::vector<std::int8_t> quantized(
        static_cast<std::size_t>(rows) * columns);
    for (
        std::size_t index = 0;
        index < quantized.size();
        ++index
    ) {
        quantized[index] =
            static_cast<std::int8_t>(
                static_cast<int>(
                    (
                        index * 11
                        + static_cast<std::size_t>(
                            salt * 7)
                    ) % 25)
                - 12);
    }

    std::vector<std::uint8_t> blob{
        'N', 'I', '8', '0',
    };
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, columns);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, rows);
    append<std::int64_t>(blob, columns);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(rows));
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(groups));
    for (int row = 0; row < rows; ++row) {
        for (
            int group = 0;
            group < groups;
            ++group
        ) {
            const int scale =
                (row + group + salt) & 3;
            append<std::uint16_t>(
                blob,
                scale_bits[scale]);
            const auto offset =
                static_cast<std::size_t>(row)
                    * columns
                + group * 32;
            const auto* bytes =
                reinterpret_cast<
                    const std::uint8_t*>(
                    quantized.data() + offset);
            blob.insert(
                blob.end(),
                bytes,
                bytes + 32);
        }
    }

    std::vector<float> dense(
        static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        for (
            int column = 0;
            column < columns;
            ++column
        ) {
            const int group = column / 32;
            dense[
                static_cast<std::size_t>(row)
                    * columns
                + column
            ] =
                scale_values[
                    (row + group + salt) & 3]
                * static_cast<float>(
                    quantized[
                        static_cast<std::size_t>(row)
                            * columns
                        + column]);
        }
    }
    return {
        std::move(blob),
        std::move(dense),
        rows,
        columns,
    };
}

struct PoolFixture {
    std::vector<std::int32_t> expert_ids;
    std::string dtype;
    TensorFixture tensor;
    std::vector<std::uint8_t> runtime;
};

struct MoeFixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    int experts = 0;
    int output = 0;
    int input = 0;
};

MoeFixture make_moe_fixture(
    const std::vector<std::string>& profiles,
    int output,
    int input,
    int salt) {
    const int experts =
        static_cast<int>(profiles.size());
    std::vector<PoolFixture> pools;
    pools.reserve(profiles.size());

    // Deliberately reverse the pool order.  Global expert order must not be
    // confused with cohort-local row order.
    for (
        int expert = experts - 1;
        expert >= 0;
        --expert
    ) {
        const auto& profile = profiles[expert];
        TensorFixture tensor =
            profile == "NINT8-0"
            ? make_q8_tensor(
                  output,
                  input,
                  salt + expert)
            : make_nint_tensor(
                  std::stoi(profile.substr(4)),
                  output,
                  input,
                  salt + expert);
        pools.push_back({
            {expert},
            profile,
            std::move(tensor),
            {},
        });
    }

    std::vector<std::uint8_t> blob{
        'N', 'I', 'M', '2',
    };
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(experts));
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(output));
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(input));
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(
            pools.size()));
    std::vector<float> dense(
        static_cast<std::size_t>(experts)
            * output * input);
    for (const auto& pool : pools) {
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(
                pool.expert_ids.size()));
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(
                pool.dtype.size()));
        append<std::uint64_t>(
            blob,
            static_cast<std::uint64_t>(
                pool.tensor.blob.size()));
        append<std::uint64_t>(
            blob,
            static_cast<std::uint64_t>(
                pool.runtime.size()));
        for (const auto expert : pool.expert_ids) {
            append<std::int32_t>(blob, expert);
        }
        blob.insert(
            blob.end(),
            pool.dtype.begin(),
            pool.dtype.end());
        blob.insert(
            blob.end(),
            pool.runtime.begin(),
            pool.runtime.end());
        blob.insert(
            blob.end(),
            pool.tensor.blob.begin(),
            pool.tensor.blob.end());

        for (
            std::size_t local = 0;
            local < pool.expert_ids.size();
            ++local
        ) {
            const int expert =
                pool.expert_ids[local];
            const auto source =
                local
                * static_cast<std::size_t>(output)
                * input;
            const auto target =
                static_cast<std::size_t>(expert)
                * output * input;
            std::copy_n(
                pool.tensor.dense.begin()
                    + static_cast<std::ptrdiff_t>(
                        source),
                static_cast<std::size_t>(output)
                    * input,
                dense.begin()
                    + static_cast<std::ptrdiff_t>(
                        target));
        }
    }
    return {
        std::move(blob),
        std::move(dense),
        experts,
        output,
        input,
    };
}

std::vector<std::uint8_t> make_raw_nim2(
    int experts,
    int output,
    int input,
    const std::vector<PoolFixture>& pools) {
    std::vector<std::uint8_t> blob{
        'N', 'I', 'M', '2',
    };
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(
            pools.size()));
    for (const auto& pool : pools) {
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(
                pool.expert_ids.size()));
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(
                pool.dtype.size()));
        append<std::uint64_t>(
            blob,
            pool.tensor.blob.size());
        append<std::uint64_t>(
            blob,
            pool.runtime.size());
        for (const auto expert : pool.expert_ids) {
            append<std::int32_t>(blob, expert);
        }
        blob.insert(
            blob.end(),
            pool.dtype.begin(),
            pool.dtype.end());
        blob.insert(
            blob.end(),
            pool.runtime.begin(),
            pool.runtime.end());
        blob.insert(
            blob.end(),
            pool.tensor.blob.begin(),
            pool.tensor.blob.end());
    }
    return blob;
}

std::vector<std::uint8_t> make_nim1(
    int experts,
    int output,
    int input,
    const TensorFixture& tensor) {
    std::vector<std::uint8_t> blob{
        'N', 'I', 'M', '1',
    };
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, experts);
    append<std::uint64_t>(
        blob,
        tensor.blob.size());
    for (int expert = 0; expert < experts; ++expert) {
        append<std::int32_t>(blob, expert);
    }
    blob.insert(
        blob.end(),
        tensor.blob.begin(),
        tensor.blob.end());
    return blob;
}

struct RoutedVqFixture {
    std::vector<std::uint8_t> blob;
    std::vector<VqFixture> expert_weights;
    int experts = 0;
    int output = 0;
    int input = 0;
};

RoutedVqFixture make_vq_moe_fixture(
    std::vector<VqFixture> weights) {
    if (weights.empty()) {
        throw std::runtime_error(
            "VQ MoE fixture cannot be empty");
    }
    const int output = weights.front().output;
    const int input = weights.front().input;
    for (const auto& weight : weights) {
        if (
            weight.output != output
            || weight.input != input
        ) {
            throw std::runtime_error(
                "VQ MoE fixture shape mismatch");
        }
    }
    const int experts =
        static_cast<int>(weights.size());
    std::vector<std::uint8_t> blob{
        'N', 'I', 'M', '2',
    };
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, experts);
    for (int expert = experts - 1;
         expert >= 0;
         --expert) {
        const auto& weight = weights[expert];
        append<std::uint32_t>(blob, 1);
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(
                weight.dtype.size()));
        append<std::uint64_t>(
            blob,
            weight.blob.size());
        append<std::uint64_t>(
            blob,
            weight.runtime.size());
        append<std::int32_t>(blob, expert);
        blob.insert(
            blob.end(),
            weight.dtype.begin(),
            weight.dtype.end());
        blob.insert(
            blob.end(),
            weight.runtime.begin(),
            weight.runtime.end());
        blob.insert(
            blob.end(),
            weight.blob.begin(),
            weight.blob.end());
    }
    return {
        std::move(blob),
        std::move(weights),
        experts,
        output,
        input,
    };
}

std::vector<float> rotate_vq_input(
    const std::vector<float>& source,
    const VqFixture& weight) {
    auto values = source;
    if (weight.rotation_block == 0) {
        return values;
    }
    for (int start = 0;
         start < weight.input;
         start += weight.rotation_block) {
        for (int index = 0;
             index < weight.rotation_block;
             ++index) {
            values[start + index] *=
                weight.signs[start + index];
        }
        for (int stride = 1;
             stride < weight.rotation_block;
             stride <<= 1) {
            for (int base = 0;
                 base < weight.rotation_block;
                 base += stride * 2) {
                for (int offset = 0;
                     offset < stride;
                     ++offset) {
                    const int first =
                        start + base + offset;
                    const int second = first + stride;
                    const float first_value =
                        values[first];
                    const float second_value =
                        values[second];
                    values[first] =
                        first_value + second_value;
                    values[second] =
                        first_value - second_value;
                }
            }
        }
        const float inverse =
            1.0f / std::sqrt(
                static_cast<float>(
                    weight.rotation_block));
        for (int index = 0;
             index < weight.rotation_block;
             ++index) {
            values[start + index] *= inverse;
        }
    }
    return values;
}

float routed_vq_dot(
    const std::vector<float>& source,
    const RoutedVqFixture& fixture,
    int expert,
    int output) {
    const auto rotated = rotate_vq_input(
        source,
        fixture.expert_weights[expert]);
    float result = 0.0f;
    const auto& weight =
        fixture.expert_weights[expert];
    for (int column = 0;
         column < fixture.input;
         ++column) {
        result +=
            rotated[column]
            * weight.dense[
                static_cast<std::size_t>(output)
                    * fixture.input
                + column];
    }
    return result;
}

void require(
    bool condition,
    const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    float actual,
    float expected,
    float tolerance = 8e-4f) {
    if (
        !std::isfinite(actual)
        || std::fabs(actual - expected) > tolerance
    ) {
        throw std::runtime_error(
            "NINTM Metal result mismatch: actual="
            + std::to_string(actual)
            + " expected="
            + std::to_string(expected));
    }
}

float dot(
    const std::vector<float>& input,
    std::size_t input_offset,
    const MoeFixture& weight,
    int expert,
    int output) {
    float result = 0.0f;
    const auto weight_offset = (
        static_cast<std::size_t>(expert)
            * weight.output
        + output
    ) * weight.input;
    for (
        int column = 0;
        column < weight.input;
        ++column
    ) {
        result +=
            input[input_offset + column]
            * weight.dense[
                weight_offset + column];
    }
    return result;
}

std::vector<float> evaluated_floats(
    mlx::core::array value) {
    value = mlx::core::astype(
        value,
        mlx::core::float32);
    value.eval();
    return {
        value.data<float>(),
        value.data<float>() + value.size(),
    };
}

struct StreamCccpProfile {
    std::string dtype;
    int tier = 0;
    int vector_size = 0;
    int entries = 0;
    int bits = 0;
};

struct StreamCccpPoolFixture {
    StreamCccpProfile profile;
    std::vector<std::int32_t> expert_ids;
    std::vector<std::uint8_t> payload;
    std::vector<std::uint8_t> runtime;
    std::vector<float> dense;
};

struct StreamCccpFixture {
    std::vector<std::uint8_t> record;
    std::vector<float> dense;
    std::size_t shared_codebook_nbytes = 0;
    int experts = 0;
    int output = 0;
    int input = 0;
};

StreamCccpPoolFixture make_stream_cccp_pool(
    StreamCccpProfile profile,
    std::vector<std::int32_t> expert_ids,
    int output,
    int input,
    int salt,
    int invalid_local_expert = -1) {
    require(
        !expert_ids.empty()
            && output > 0
            && input > 0
            && input % profile.vector_size == 0,
        "invalid streamed CCCP fixture shape");
    const int blocks =
        input / profile.vector_size;
    std::vector<float> codebook(
        static_cast<std::size_t>(
            profile.entries)
            * profile.vector_size);
    for (
        int entry = 0;
        entry < profile.entries;
        ++entry
    ) {
        for (
            int component = 0;
            component < profile.vector_size;
            ++component
        ) {
            codebook[
                static_cast<std::size_t>(entry)
                    * profile.vector_size
                + component
            ] =
                static_cast<float>(
                    (
                        entry * 3
                        + component * 5
                        + salt
                    ) % 31
                    - 15)
                / 64.0f;
        }
    }
    const auto rows =
        static_cast<int>(
            expert_ids.size())
        * output;
    std::vector<std::uint16_t> indices(
        static_cast<std::size_t>(rows)
            * blocks);
    for (
        std::size_t index = 0;
        index < indices.size();
        ++index
    ) {
        indices[index] =
            static_cast<std::uint16_t>(
                (
                    index * 97
                    + static_cast<std::size_t>(
                        salt * 11)
                    + profile.entries - 1
                ) % profile.entries);
    }
    if (!indices.empty()) {
        indices.front() =
            static_cast<std::uint16_t>(
                profile.entries - 1);
    }
    if (invalid_local_expert >= 0) {
        require(
            profile.entries
                < (1 << profile.bits),
            "invalid-index fixture needs spare codes");
        require(
            invalid_local_expert
                < static_cast<int>(
                    expert_ids.size()),
            "invalid-index local expert is out "
            "of range");
        indices[
            static_cast<std::size_t>(
                invalid_local_expert)
                * output * blocks] =
            static_cast<std::uint16_t>(
                profile.entries + 3);
    }

    std::vector<std::uint8_t> payload;
    append_magic(payload, "CPQ1");
    append<std::uint8_t>(payload, 1);
    append<std::uint8_t>(
        payload,
        static_cast<std::uint8_t>(
            profile.tier));
    append<std::uint8_t>(
        payload,
        static_cast<std::uint8_t>(
            profile.vector_size));
    append<std::uint8_t>(
        payload,
        static_cast<std::uint8_t>(
            profile.bits));
    append<std::int32_t>(payload, 0);
    append<std::int32_t>(payload, input);
    append<std::uint32_t>(payload, 2);
    append<std::uint32_t>(
        payload,
        static_cast<std::uint32_t>(
            profile.entries));
    append<std::int64_t>(payload, rows);
    append<std::int64_t>(payload, input);
    append<std::uint32_t>(payload, rows);
    for (const auto value : codebook) {
        append<float>(payload, value);
    }
    append_bytes(
        payload,
        pack_vq_values(
            indices,
            profile.bits));

    std::vector<float> dense(
        static_cast<std::size_t>(rows)
            * input);
    for (int row = 0; row < rows; ++row) {
        for (
            int block = 0;
            block < blocks;
            ++block
        ) {
            const auto code =
                indices[
                    static_cast<std::size_t>(
                        row)
                        * blocks
                    + block];
            if (
                code
                >= static_cast<std::uint16_t>(
                    profile.entries)
            ) {
                continue;
            }
            for (
                int component = 0;
                component < profile.vector_size;
                ++component
            ) {
                dense[
                    static_cast<std::size_t>(
                        row)
                        * input
                    + block
                        * profile.vector_size
                    + component
                ] =
                    codebook[
                        static_cast<std::size_t>(
                            code)
                            * profile.vector_size
                        + component
                    ];
            }
        }
    }
    return {
        std::move(profile),
        std::move(expert_ids),
        std::move(payload),
        {},
        std::move(dense),
    };
}

std::vector<std::uint8_t>
assemble_stream_nim2(
    int experts,
    int output,
    int input,
    const std::vector<
        StreamCccpPoolFixture>& pools) {
    std::vector<std::uint8_t> result;
    append_magic(result, "NIM2");
    append<std::uint32_t>(result, experts);
    append<std::uint32_t>(result, output);
    append<std::uint32_t>(result, input);
    append<std::uint32_t>(
        result,
        static_cast<std::uint32_t>(
            pools.size()));
    for (const auto& pool : pools) {
        append<std::uint32_t>(
            result,
            static_cast<std::uint32_t>(
                pool.expert_ids.size()));
        append<std::uint32_t>(
            result,
            static_cast<std::uint32_t>(
                pool.profile.dtype.size()));
        append<std::uint64_t>(
            result,
            pool.payload.size());
        append<std::uint64_t>(
            result,
            pool.runtime.size());
        for (const auto expert : pool.expert_ids) {
            append<std::int32_t>(
                result,
                expert);
        }
        result.insert(
            result.end(),
            pool.profile.dtype.begin(),
            pool.profile.dtype.end());
        append_bytes(result, pool.runtime);
        append_bytes(result, pool.payload);
    }
    return result;
}

StreamCccpFixture
make_stream_cccp_fixture() {
    constexpr int experts = 14;
    constexpr int output = 1;
    constexpr int input = 24;
    const std::vector<StreamCccpProfile>
        profiles{
            {"CCCP-X", 1, 8, 256, 8},
            {"CCCP-X", 1, 8, 256, 12},
            {"CCCP-X", 1, 8, 256, 14},
            {"CCCP-W", 2, 8, 4096, 12},
            {"CCCP-W", 2, 8, 4096, 14},
            {"CCCP-W", 2, 8, 4096, 16},
            {"CCCP-V", 3, 4, 256, 8},
            {"CCCP-V", 3, 4, 256, 12},
            {"CCCP-V", 3, 4, 256, 14},
            {"CCCP-VV", 4, 4, 4096, 12},
            {"CCCP-VV", 4, 4, 4096, 14},
            {"CCCP-VV", 4, 4, 4096, 16},
        };
    const std::vector<
        std::vector<std::int32_t>> ids{
            {8},
            {1},
            {12, 0},
            {10},
            {2},
            {9},
            {3},
            {7},
            {4},
            {6},
            {5},
            {11},
        };
    std::vector<StreamCccpPoolFixture> pools;
    pools.reserve(profiles.size());
    std::vector<float> dense(
        static_cast<std::size_t>(
            experts)
            * output * input,
        0.0f);
    std::size_t shared_codebook_nbytes = 0;
    for (
        std::size_t index = 0;
        index < profiles.size();
        ++index
    ) {
        auto pool = make_stream_cccp_pool(
            profiles[index],
            ids[index],
            output,
            input,
            static_cast<int>(index + 1));
        for (
            std::size_t local = 0;
            local < pool.expert_ids.size();
            ++local
        ) {
            const auto source =
                local
                * static_cast<std::size_t>(
                    output)
                * input;
            const auto target =
                static_cast<std::size_t>(
                    pool.expert_ids[local])
                * output * input;
            std::copy_n(
                pool.dense.begin()
                    + static_cast<
                        std::ptrdiff_t>(
                        source),
                static_cast<std::size_t>(
                    output)
                    * input,
                dense.begin()
                    + static_cast<
                        std::ptrdiff_t>(
                        target));
        }
        shared_codebook_nbytes +=
            static_cast<std::size_t>(
                pool.profile.entries)
            * pool.profile.vector_size
            * sizeof(std::uint16_t);
        pools.push_back(std::move(pool));
    }
    return {
        assemble_stream_nim2(
            experts,
            output,
            input,
            pools),
        std::move(dense),
        shared_codebook_nbytes,
        experts,
        output,
        input,
    };
}

struct MappedRecordFixture {
    std::string name;
    std::string dtype;
    std::vector<std::uint8_t> payload;
};

void append_mfq_string(
    std::vector<std::uint8_t>& output,
    std::string_view value) {
    append<std::uint32_t>(
        output,
        static_cast<std::uint32_t>(
            value.size()));
    output.insert(
        output.end(),
        value.begin(),
        value.end());
}

class TemporaryMfq {
public:
    explicit TemporaryMfq(
        const std::vector<MappedRecordFixture>&
            records)
        : path_(
              std::filesystem::
                  temp_directory_path()
              / "mfq-metal-streamed-cccp-test.mfq") {
        std::vector<std::uint8_t> file;
        append_magic(file, "MFQ1");
        append<std::uint32_t>(file, 1);
        append_mfq_string(
            file,
            "streamed-cccp-test");
        append<std::uint32_t>(
            file,
            static_cast<std::uint32_t>(
                records.size()));
        for (const auto& record : records) {
            append_mfq_string(
                file,
                record.name);
            append_mfq_string(
                file,
                record.dtype);
            append<std::uint64_t>(
                file,
                record.payload.size());
        }
        for (const auto& record : records) {
            append_bytes(
                file,
                record.payload);
        }
        std::ofstream stream(
            path_,
            std::ios::binary
                | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error(
                "cannot create streamed CCCP "
                "test container");
        }
        stream.write(
            reinterpret_cast<const char*>(
                file.data()),
            static_cast<std::streamsize>(
                file.size()));
        if (!stream) {
            throw std::runtime_error(
                "cannot write streamed CCCP "
                "test container");
        }
    }

    ~TemporaryMfq() {
        std::error_code ignored;
        std::filesystem::remove(
            path_,
            ignored);
    }

    const std::filesystem::path& path() const {
        return path_;
    }

private:
    std::filesystem::path path_;
};

template <typename Function>
void require_stream_rejected(
    Function&& function,
    const std::string& context) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(
        rejected,
        "streamed CCCP malformed fixture was "
        "accepted: " + context);
}

void test_streamed_cccp_residency() {
    auto fixture =
        make_stream_cccp_fixture();
    constexpr int experts = 14;
    constexpr int output = 1;
    constexpr int input = 24;

    auto bad_tier_pool =
        make_stream_cccp_pool(
            {"CCCP-X", 1, 8, 256, 8},
            {0},
            output,
            input,
            21);
    bad_tier_pool.payload[5] = 4;
    auto bad_tier = assemble_stream_nim2(
        experts,
        output,
        input,
        {bad_tier_pool});

    auto bad_padding_pool =
        make_stream_cccp_pool(
            {"CCCP-X", 1, 8, 256, 14},
            {0},
            output,
            input,
            22);
    bad_padding_pool.payload.back() |= 0x80u;
    auto bad_padding = assemble_stream_nim2(
        experts,
        output,
        input,
        {bad_padding_pool});

    auto bad_index_pool =
        make_stream_cccp_pool(
            {"CCCP-W", 2, 8, 4096, 14},
            {0},
            output,
            input,
            23,
            0);
    auto bad_index = assemble_stream_nim2(
        experts,
        output,
        input,
        {bad_index_pool});

    auto partial_small =
        make_stream_cccp_pool(
            {"CCCP-X", 1, 8, 256, 8},
            {0},
            output,
            input,
            31);
    auto partial_large =
        make_stream_cccp_pool(
            {"CCCP-W", 2, 8, 4096, 14},
            {1},
            output,
            input,
            32);
    auto partial_late_failure =
        make_stream_cccp_pool(
            {"CCCP-X", 1, 8, 256, 12},
            {3, 2},
            output,
            input,
            33,
            1);
    auto partial_failure =
        assemble_stream_nim2(
            experts,
            output,
            input,
            {
                partial_small,
                partial_large,
                partial_late_failure,
            });

    auto p12_unaligned_pool =
        make_stream_cccp_pool(
            {"CCCP-W", 2, 8, 4096, 12},
            {2, 5, 7},
            output,
            input,
            34);
    auto p12_unaligned =
        assemble_stream_nim2(
            experts,
            output,
            input,
            {p12_unaligned_pool});

    constexpr int shifted_output = 3;
    auto p14_shifted_pool =
        make_stream_cccp_pool(
            {"CCCP-W", 2, 8, 4096, 14},
            {1, 3, 5, 7},
            shifted_output,
            input,
            35);
    auto p14_shifted =
        assemble_stream_nim2(
            experts,
            shifted_output,
            input,
            {p14_shifted_pool});

    auto duplicate_first =
        make_stream_cccp_pool(
            {"CCCP-X", 1, 8, 256, 8},
            {0},
            output,
            input,
            24);
    auto duplicate_second =
        make_stream_cccp_pool(
            {"CCCP-V", 3, 4, 256, 8},
            {0},
            output,
            input,
            25);
    auto duplicate = assemble_stream_nim2(
        experts,
        output,
        input,
        {
            duplicate_first,
            duplicate_second,
        });

    auto trailing = fixture.record;
    trailing.push_back(0);

    StreamCccpPoolFixture fallback_pool;
    fallback_pool.profile.dtype = "NINT4";
    fallback_pool.expert_ids.resize(experts);
    for (int expert = 0; expert < experts; ++expert) {
        fallback_pool.expert_ids[
            static_cast<std::size_t>(
                expert)] = expert;
    }
    fallback_pool.payload.resize(64, 0);
    auto fallback = assemble_stream_nim2(
        experts,
        output,
        input,
        {fallback_pool});

    const TemporaryMfq file({
        {"good", "NINTM", fixture.record},
        {"bad_tier", "NINTM", bad_tier},
        {"bad_padding", "NINTM", bad_padding},
        {"bad_index", "NINTM", bad_index},
        {
            "partial_failure",
            "NINTM",
            partial_failure,
        },
        {
            "p12_unaligned",
            "NINTM",
            p12_unaligned,
        },
        {
            "p14_shifted",
            "NINTM",
            p14_shifted,
        },
        {"duplicate", "NINTM", duplicate},
        {"trailing", "NINTM", trailing},
        {"fallback", "NINTM", fallback},
    });
    const mfq::metal::MfqContainer model(
        file.path());
    std::unique_ptr<
        mfq::metal::MlxCccpExpertResidency>
        detached_residency;
    {
        const mfq::metal::MfqContainer
            short_lived_model(file.path());
        detached_residency =
            std::make_unique<
                mfq::metal::
                    MlxCccpExpertResidency>(
                short_lived_model,
                64,
                experts);
        require(
            detached_residency->can_stream(
                "good"),
            "short-lived container CCCP parse failed");
    }
    const auto detached_weight =
        detached_residency->grouped(
            "good",
            {0});
    std::vector<float> detached_source(
        input,
        1.0f / 32.0f);
    const std::vector<std::int32_t>
        detached_ids{0};
    const auto detached_output =
        evaluated_floats(
            detached_weight.routed_matmul(
                mlx::core::array(
                    detached_source.begin(),
                    mlx::core::Shape{
                        1,
                        input,
                    }),
                mlx::core::array(
                    detached_ids.begin(),
                    mlx::core::Shape{1, 1})));
    float detached_expected = 0.0f;
    for (int column = 0;
         column < input;
         ++column) {
        detached_expected +=
            detached_source[
                static_cast<std::size_t>(
                    column)]
            * fixture.dense[
                static_cast<std::size_t>(
                    column)];
    }
    require_close(
        detached_output.front(),
        detached_expected,
        1.5e-3f);
    detached_residency->clear();
    const auto after_clear_output =
        evaluated_floats(
            detached_weight.routed_matmul(
                mlx::core::array(
                    detached_source.begin(),
                    mlx::core::Shape{
                        1,
                        input,
                    }),
                mlx::core::array(
                    detached_ids.begin(),
                    mlx::core::Shape{1, 1})));
    require_close(
        after_clear_output.front(),
        detached_expected,
        1.5e-3f);
    detached_residency.reset();
    const auto after_destroy_output =
        evaluated_floats(
            detached_weight.routed_matmul(
                mlx::core::array(
                    detached_source.begin(),
                    mlx::core::Shape{
                        1,
                        input,
                    }),
                mlx::core::array(
                    detached_ids.begin(),
                    mlx::core::Shape{1, 1})));
    require_close(
        after_destroy_output.front(),
        detached_expected,
        1.5e-3f);

    mfq::metal::MlxCccpExpertResidency
        residency(model, 10, experts);
    require(
        residency.can_stream("good"),
        "valid CCCP NIM2 record was not streamable");
    require(
        !residency.can_stream("fallback"),
        "non-CCCP NIM2 record was streamable");
    const auto info =
        residency.projection_info("good");
    require(
        info.experts == experts
            && info.out_per_expert == output
            && info.neuron_len == input
            && info.available_experts.size() == 13
            && info.shared_codebook_nbytes
                == fixture.shared_codebook_nbytes,
        "streamed CCCP projection metadata mismatch");
    const auto available =
        residency.availability("good");
    require(
        available.size() == experts
            && available[0] == 1
            && available[12] == 1
            && available[13] == 0,
        "streamed CCCP global availability mismatch");

    require_stream_rejected(
        [&] {
            (void)residency.can_stream(
                "bad_tier");
        },
        "tier mismatch");
    require_stream_rejected(
        [&] {
            (void)residency.can_stream(
                "bad_padding");
        },
        "packed padding");
    require_stream_rejected(
        [&] {
            (void)residency.can_stream(
                "duplicate");
        },
        "duplicate global ID");
    require_stream_rejected(
        [&] {
            (void)residency.can_stream(
                "trailing");
        },
        "record tail");
    require(
        residency.can_stream("bad_index"),
        "bad-index metadata did not parse");
    require_stream_rejected(
        [&] {
            (void)residency.grouped(
                "bad_index",
                {0});
        },
        "missing codeword");

    mfq::metal::MlxCccpExpertResidency
        transactional(model, 11, experts);
    require(
        transactional.can_stream(
            "partial_failure"),
        "partial-failure CCCP metadata did not "
        "parse");
    (void)transactional.grouped(
        "partial_failure",
        {0});
    (void)transactional.grouped(
        "partial_failure",
        {1});
    require(
        transactional.cached_expert_count() == 2
            && transactional
                    .resident_packed_bytes()
                == 9,
        "transactional CCCP cache seed mismatch");
    const auto require_transaction_unchanged =
        [&](
            const auto& operation,
            const std::string& context) {
            require_stream_rejected(
                operation,
                context);
            require(
                transactional
                        .cached_expert_count()
                    == 2
                    && transactional
                            .resident_packed_bytes()
                        == 9,
                "failed CCCP grouped transaction "
                "changed cache accounting: "
                    + context);
        };
    require_transaction_unchanged(
        [&] {
            (void)transactional.grouped(
                "partial_failure",
                {0, experts});
        },
        "late out-of-range expert");
    require_transaction_unchanged(
        [&] {
            (void)transactional.grouped(
                "partial_failure",
                {0, 13});
        },
        "late unavailable expert");
    require_transaction_unchanged(
        [&] {
            // Expert 3 is fully read and allocated before
            // expert 2 exposes its malformed codeword.
            (void)transactional.grouped(
                "partial_failure",
                {0, 3, 2});
        },
        "late malformed expert");
    (void)transactional.grouped(
        "partial_failure",
        {3});
    require(
        transactional.cached_expert_count() == 2
            && transactional
                    .resident_packed_bytes()
                == 11,
        "failed CCCP transaction changed LRU "
        "ordering");

    mfq::metal::MlxCccpExpertResidency
        alignment(model, 128, experts);
    const auto verify_stream_pool =
        [&](
            const std::string& name,
            const StreamCccpPoolFixture& pool,
            int pool_output) {
            const auto routed =
                alignment.grouped(
                    name,
                    pool.expert_ids);
            std::vector<float> source_row(input);
            for (
                std::size_t column = 0;
                column < source_row.size();
                ++column
            ) {
                source_row[column] =
                    static_cast<float>(
                        static_cast<int>(
                            (column * 13 + 5) % 23)
                        - 11)
                    / 64.0f;
            }
            const auto actual_pool =
                evaluated_floats(
                    routed.routed_matmul(
                        mlx::core::array(
                            source_row.begin(),
                            mlx::core::Shape{
                                1,
                                input,
                            }),
                        mlx::core::array(
                            pool.expert_ids.begin(),
                            mlx::core::Shape{
                                1,
                                static_cast<int>(
                                    pool.expert_ids
                                        .size()),
                            })));
            require(
                actual_pool.size()
                    == pool.expert_ids.size()
                        * static_cast<
                            std::size_t>(
                            pool_output),
                "unaligned CCCP output shape "
                "mismatch: " + name);
            for (
                std::size_t local = 0;
                local < pool.expert_ids.size();
                ++local
            ) {
                for (
                    int row = 0;
                    row < pool_output;
                    ++row
                ) {
                    float expected = 0.0f;
                    for (
                        int column = 0;
                        column < input;
                        ++column
                    ) {
                        expected +=
                            source_row[
                                static_cast<
                                    std::size_t>(
                                    column)]
                            * pool.dense[
                                (
                                    local
                                        * pool_output
                                    + static_cast<
                                        std::size_t>(
                                        row)
                                ) * input
                                + column];
                    }
                    require_close(
                        actual_pool[
                            local * pool_output
                            + static_cast<
                                std::size_t>(
                                row)],
                        expected,
                        1.5e-3f);
                }
            }
        };
    require(
        (
            output * (input / 8) * 12
        ) % 8 == 4,
        "p12 multi-local fixture is byte "
        "aligned");
    verify_stream_pool(
        "p12_unaligned",
        p12_unaligned_pool,
        output);
    const auto p14_expert_bits =
        shifted_output * (input / 8) * 14;
    require(
        p14_expert_bits % 8 == 6
            && (2 * p14_expert_bits) % 8 == 4
            && (3 * p14_expert_bits) % 8 == 2,
        "p14 fixture does not cover shifts "
        "6/4/2");
    verify_stream_pool(
        "p14_shifted",
        p14_shifted_pool,
        shifted_output);

    (void)residency.grouped(
        "good",
        {12, 0});
    require(
        residency.cached_expert_count() == 2
            && residency.resident_packed_bytes()
                == 12,
        "active streamed CCCP experts were "
        "incorrectly evicted");
    (void)residency.grouped(
        "good",
        {1});
    require(
        residency.cached_expert_count() == 1
            && residency.resident_packed_bytes()
                == 5,
        "streamed CCCP LRU byte eviction mismatch");

    std::vector<std::int32_t> active(13);
    for (
        int expert = 0;
        expert < 13;
        ++expert
    ) {
        active[static_cast<std::size_t>(
            expert)] = expert;
    }
    const auto weight =
        residency.grouped(
            "good",
            active);
    require(
        weight.experts() == experts
            && weight.out_per_expert() == output
            && weight.neuron_len() == input
            && weight.shared_codebook_nbytes()
                == fixture.shared_codebook_nbytes,
        "streamed CCCP routed weight metadata mismatch");

    constexpr int tokens = 5;
    constexpr int routes = 3;
    std::vector<float> source(
        tokens * input);
    for (
        std::size_t index = 0;
        index < source.size();
        ++index
    ) {
        source[index] =
            static_cast<float>(
                static_cast<int>(
                    (index * 17 + 3) % 29)
                - 14)
            / 128.0f;
    }
    const std::vector<std::int32_t> ids{
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11,
        12, 13, -1,
    };
    const mlx::core::array input_array(
        source.begin(),
        mlx::core::Shape{
            tokens,
            input,
        });
    const mlx::core::array id_array(
        ids.begin(),
        mlx::core::Shape{
            tokens,
            routes,
        });
    const auto actual =
        evaluated_floats(
            weight.routed_matmul(
                input_array,
                id_array));
    require(
        actual.size()
            == static_cast<std::size_t>(
                tokens * routes * output),
        "streamed CCCP routed output size mismatch");
    for (
        int token = 0;
        token < tokens;
        ++token
    ) {
        for (
            int route = 0;
            route < routes;
            ++route
        ) {
            const auto expert =
                ids[
                    static_cast<std::size_t>(
                        token)
                        * routes
                    + route];
            float expected = 0.0f;
            if (expert >= 0 && expert < 13) {
                for (
                    int column = 0;
                    column < input;
                    ++column
                ) {
                    expected +=
                        source[
                            static_cast<
                                std::size_t>(
                                token)
                                * input
                            + column]
                        * fixture.dense[
                            static_cast<
                                std::size_t>(
                                expert)
                                * input
                            + column];
                }
            }
            require_close(
                actual[
                    (
                        static_cast<std::size_t>(
                            token)
                            * routes
                        + route
                    ) * output],
                expected,
                1.5e-3f);
        }
    }
    static_assert(noexcept(
        residency.discard_record(
            std::declval<const std::string&>())));
    residency.discard_record("good");
    require(
        residency.cached_expert_count() == 0
            && residency.resident_packed_bytes()
                == 0,
        "streamed CCCP record discard mismatch");
    const auto after_discard =
        evaluated_floats(
            weight.routed_matmul(
                input_array,
                id_array));
    require(
        after_discard.size() == actual.size(),
        "discarded-record routed output size "
        "mismatch");
    for (
        std::size_t index = 0;
        index < actual.size();
        ++index
    ) {
        require_close(
            after_discard[index],
            actual[index],
            1.5e-3f);
    }
    residency.discard_record("not-present");
    require(
        residency.can_stream("good"),
        "discarded CCCP projection could not be "
        "parsed again");
    residency.clear();
    require(
        residency.cached_expert_count() == 0
            && residency.resident_packed_bytes()
                == 0,
        "streamed CCCP residency clear mismatch");
}

void test_all_families_and_projections() {
    constexpr int tokens = 3;
    constexpr int routes = 4;
    constexpr int input_width = 64;
    constexpr int output_width = 5;
    const std::vector<std::string> profiles{
        "NINT2",
        "NINT7",
        "NINT4",
        "NINT6",
        "NINT8-0",
        "NINT3",
        "NINT8",
        "NINT1",
        "NINT5",
    };
    const auto first = make_moe_fixture(
        profiles,
        output_width,
        input_width,
        1);
    const auto second = make_moe_fixture(
        profiles,
        output_width,
        input_width,
        17);
    const auto first_weight =
        mfq::metal::MlxMoeWeight::from_blob(
            first.blob);
    const auto second_weight =
        mfq::metal::MlxMoeWeight::from_blob(
            second.blob);
    require(
        first_weight.experts()
                == static_cast<int>(
                    profiles.size())
            && first_weight.out_per_expert()
                == output_width
            && first_weight.neuron_len()
                == input_width
            && first_weight.projections() == 1
            && first_weight.packed_nbytes() > 0,
        "NINTM shape metadata mismatch");

    const std::vector<std::int32_t> ids{
        0, 1, 4, 8,
        7, 5, -1, 9,
        2, 3, 6, 0,
    };
    std::vector<float> shared_input(
        tokens * input_width);
    for (
        std::size_t index = 0;
        index < shared_input.size();
        ++index
    ) {
        shared_input[index] =
            static_cast<float>(
                static_cast<int>(
                    (index * 7 + 3) % 23)
                - 11)
            / 64.0f;
    }
    auto shared = first_weight.routed_matmul(
        mlx::core::array(
            shared_input.begin(),
            mlx::core::Shape{
                tokens,
                input_width,
            }),
        mlx::core::array(
            ids.begin(),
            mlx::core::Shape{
                tokens,
                routes,
            }));
    const auto shared_values =
        evaluated_floats(std::move(shared));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert =
                ids[token * routes + route];
            for (
                int output = 0;
                output < output_width;
                ++output
            ) {
                const float expected =
                    expert < 0
                            || expert >= first.experts
                    ? 0.0f
                    : dot(
                          shared_input,
                          static_cast<std::size_t>(
                              token)
                              * input_width,
                          first,
                          expert,
                          output);
                require_close(
                    shared_values[
                        (
                            token * routes + route
                        ) * output_width
                        + output],
                    expected);
            }
        }
    }

    // Exercise the separate float16 Metal specialization as used by normal
    // model inference.  The reference rounds both the source and output to
    // FP16 at the same boundaries as the kernel.
    const auto half_values = evaluated_floats(
        first_weight.routed_matmul(
            mlx::core::astype(
                mlx::core::array(
                    shared_input.begin(),
                    mlx::core::Shape{
                        tokens,
                        input_width,
                    }),
                mlx::core::float16),
            mlx::core::array(
                ids.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                })));
    for (int token = 0; token < tokens; ++token) {
        std::vector<float> rounded_input(
            input_width);
        for (
            int column = 0;
            column < input_width;
            ++column
        ) {
            rounded_input[column] =
                static_cast<float>(
                    static_cast<
                        mlx::core::float16_t>(
                        shared_input[
                            token * input_width
                            + column]));
        }
        for (int route = 0; route < routes; ++route) {
            const int expert =
                ids[token * routes + route];
            for (
                int output = 0;
                output < output_width;
                ++output
            ) {
                const float reference =
                    expert < 0
                            || expert >= first.experts
                    ? 0.0f
                    : dot(
                          rounded_input,
                          0,
                          first,
                          expert,
                          output);
                const float expected =
                    static_cast<float>(
                        static_cast<
                            mlx::core::float16_t>(
                            reference));
                require_close(
                    half_values[
                        (
                            token * routes + route
                        ) * output_width
                        + output],
                    expected,
                    4e-3f);
            }
        }
    }

    std::vector<float> routed_input(
        tokens * routes * input_width);
    for (
        std::size_t index = 0;
        index < routed_input.size();
        ++index
    ) {
        routed_input[index] =
            static_cast<float>(
                static_cast<int>(
                    (index * 5 + 9) % 29)
                - 14)
            / 96.0f;
    }
    const auto grouped =
        mfq::metal::MlxMoeWeight::
            concatenate_projections(
                {first_weight, second_weight});
    require(
        grouped.projections() == 2,
        "NINTM grouped projection count mismatch");
    const auto grouped_values = evaluated_floats(
        grouped.routed_matmul(
            mlx::core::array(
                routed_input.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                    input_width,
                }),
            mlx::core::array(
                ids.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                })));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert =
                ids[token * routes + route];
            const auto source_offset =
                static_cast<std::size_t>(
                    token * routes + route)
                * input_width;
            for (
                int projection = 0;
                projection < 2;
                ++projection
            ) {
                const auto& fixture =
                    projection == 0 ? first : second;
                for (
                    int output = 0;
                    output < output_width;
                    ++output
                ) {
                    const float expected =
                        expert < 0
                                || expert
                                    >= fixture.experts
                        ? 0.0f
                        : dot(
                              routed_input,
                              source_offset,
                              fixture,
                              expert,
                              output);
                    require_close(
                        grouped_values[
                            (
                                (
                                    token * routes
                                    + route
                                ) * 2
                                + projection
                            ) * output_width
                            + output],
                        expected);
                }
            }
        }
    }
}

void test_swiglu_ffn() {
    constexpr int tokens = 2;
    constexpr int routes = 2;
    constexpr int hidden = 32;
    constexpr int intermediate = 32;
    const std::vector<std::string> profiles{
        "NINT2",
        "NINT7",
        "NINT8-0",
    };
    const auto gate = make_moe_fixture(
        profiles,
        intermediate,
        hidden,
        2);
    const auto up = make_moe_fixture(
        profiles,
        intermediate,
        hidden,
        9);
    const auto down = make_moe_fixture(
        profiles,
        hidden,
        intermediate,
        15);
    const auto ffn =
        mfq::metal::MlxRoutedSwiGluFfn::
            from_blobs(
                gate.blob,
                up.blob,
                down.blob);
    require(
        ffn.gate_up_weight().projections() == 2,
        "SwiGLU gate/up did not group projections");

    const std::vector<std::int32_t> ids{
        0, 2,
        1, 0,
    };
    const std::vector<float> route_weights{
        0.65f, 0.35f,
        0.25f, 0.75f,
    };
    std::vector<float> input(tokens * hidden);
    for (
        std::size_t index = 0;
        index < input.size();
        ++index
    ) {
        input[index] =
            static_cast<float>(
                static_cast<int>(
                    (index * 3 + 1) % 17)
                - 8)
            / 256.0f;
    }
    const auto actual = evaluated_floats(
        ffn.forward(
            mlx::core::array(
                input.begin(),
                mlx::core::Shape{
                    tokens,
                    hidden,
                }),
            mlx::core::array(
                ids.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                }),
            mlx::core::array(
                route_weights.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                })));

    std::vector<float> expected(
        tokens * hidden,
        0.0f);
    std::vector<float> activated(intermediate);
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert =
                ids[token * routes + route];
            for (
                int column = 0;
                column < intermediate;
                ++column
            ) {
                const float gate_value = dot(
                    input,
                    static_cast<std::size_t>(token)
                        * hidden,
                    gate,
                    expert,
                    column);
                const float up_value = dot(
                    input,
                    static_cast<std::size_t>(token)
                        * hidden,
                    up,
                    expert,
                    column);
                activated[column] =
                    gate_value
                    / (
                        1.0f
                        + std::exp(-gate_value)
                    )
                    * up_value;
            }
            const auto weight_offset =
                static_cast<std::size_t>(expert)
                * hidden * intermediate;
            for (int output = 0; output < hidden; ++output) {
                float value = 0.0f;
                for (
                    int column = 0;
                    column < intermediate;
                    ++column
                ) {
                    value +=
                        activated[column]
                        * down.dense[
                            weight_offset
                            + static_cast<std::size_t>(
                                output)
                                * intermediate
                            + column];
                }
                expected[
                    token * hidden + output
                ] +=
                    route_weights[
                        token * routes + route]
                    * value;
            }
        }
    }
    for (
        std::size_t index = 0;
        index < expected.size();
        ++index
    ) {
        require_close(
            actual[index],
            expected[index],
            3e-3f);
    }
}

void test_vq_cohorts_and_ffn() {
    constexpr int tokens = 3;
    constexpr int routes = 3;
    constexpr int width = 24;
    auto fixture = make_vq_moe_fixture({
        make_plain_nvq(width, width),
        make_plain_nvq(width, width, true),
        make_jsc_nvq(width, width),
        make_jsc_nvq(
            width,
            width,
            "NVQ3J",
            2,
            4,
            8),
        make_jsc_nvq(
            width,
            width,
            "NVQ3J-512",
            3,
            4,
            9),
        make_jsc_nvq(
            width,
            width,
            "NVQ2J-XL",
            5,
            8,
            12),
        make_nvq1_s(width, width),
        make_npq(width, width),
        make_rotated_nepq1_s(width, width),
    });
    const char* previous_exec = std::getenv(
        "MFQ_METAL_NINTM_JSC_EXEC");
    const bool had_previous_exec =
        previous_exec != nullptr;
    const std::string previous_exec_value =
        had_previous_exec ? previous_exec : "";
    setenv("MFQ_METAL_NINTM_JSC_EXEC", "1", 1);
    const auto weight =
        mfq::metal::MlxMoeWeight::from_blob(
            fixture.blob);
    setenv("MFQ_METAL_NINTM_JSC_EXEC", "0", 1);
    const auto legacy_weight =
        mfq::metal::MlxMoeWeight::from_blob(
            fixture.blob);
    if (had_previous_exec) {
        setenv(
            "MFQ_METAL_NINTM_JSC_EXEC",
            previous_exec_value.c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_NINTM_JSC_EXEC");
    }
    require(
        weight.experts() == fixture.experts
            && weight.out_per_expert() == width
            && weight.neuron_len() == width,
        "VQ NINTM metadata mismatch");

    const std::vector<std::int32_t> ids{
        0, 8, 2,
        3, 1, 7,
        5, 6, 4,
    };
    std::vector<float> input(tokens * width);
    for (
        std::size_t index = 0;
        index < input.size();
        ++index
    ) {
        input[index] =
            static_cast<float>(
                static_cast<int>(
                    (index * 5 + 3) % 19)
                - 9)
            / 256.0f;
    }
    const auto actual = evaluated_floats(
        weight.routed_matmul(
            mlx::core::array(
                input.begin(),
                mlx::core::Shape{
                    tokens,
                    width,
                }),
            mlx::core::array(
                ids.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                })));
    const auto legacy_actual = evaluated_floats(
        legacy_weight.routed_matmul(
            mlx::core::array(
                input.begin(),
                mlx::core::Shape{
                    tokens,
                    width,
                }),
            mlx::core::array(
                ids.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                })));
    require(
        legacy_actual.size() == actual.size(),
        "JSC execution layout result size mismatch");
    for (std::size_t index = 0;
         index < actual.size();
         ++index) {
        require_close(
            actual[index],
            legacy_actual[index],
            1e-6f);
    }
    for (int token = 0; token < tokens; ++token) {
        std::vector<float> source(
            input.begin() + token * width,
            input.begin() + (token + 1) * width);
        for (int route = 0;
             route < routes;
             ++route) {
            const int expert =
                ids[token * routes + route];
            for (int output = 0;
                 output < width;
                 ++output) {
                require_close(
                    actual[
                        (
                            token * routes + route
                        ) * width + output],
                    routed_vq_dot(
                        source,
                        fixture,
                        expert,
                        output),
                    2e-3f);
            }
        }
    }

    // NEPQ payloads are cross-expert tensors.  One cohort owns two global
    // experts here, with deliberately reversed global IDs, so descriptor
    // local-expert row mapping is tested independently of pool ordering.
    auto cohort = make_rotated_nepq1_s(
        width,
        width,
        0x8899aabbccddeeffull,
        0,
        2);
    std::vector<std::uint8_t> multi_blob{
        'N', 'I', 'M', '2',
    };
    append<std::uint32_t>(multi_blob, 2);
    append<std::uint32_t>(multi_blob, width);
    append<std::uint32_t>(multi_blob, width);
    append<std::uint32_t>(multi_blob, 1);
    append<std::uint32_t>(multi_blob, 2);
    append<std::uint32_t>(
        multi_blob,
        cohort.dtype.size());
    append<std::uint64_t>(
        multi_blob,
        cohort.blob.size());
    append<std::uint64_t>(
        multi_blob,
        cohort.runtime.size());
    append<std::int32_t>(multi_blob, 1);
    append<std::int32_t>(multi_blob, 0);
    multi_blob.insert(
        multi_blob.end(),
        cohort.dtype.begin(),
        cohort.dtype.end());
    multi_blob.insert(
        multi_blob.end(),
        cohort.runtime.begin(),
        cohort.runtime.end());
    multi_blob.insert(
        multi_blob.end(),
        cohort.blob.begin(),
        cohort.blob.end());

    const std::vector<std::int32_t> multi_ids{
        0, 1,
    };
    const auto multi_actual = evaluated_floats(
        mfq::metal::MlxMoeWeight::from_blob(
            multi_blob).routed_matmul(
                mlx::core::array(
                    input.begin(),
                    mlx::core::Shape{1, width}),
                mlx::core::array(
                    multi_ids.begin(),
                    mlx::core::Shape{1, 2})));
    std::vector<float> multi_source(
        input.begin(),
        input.begin() + width);
    const auto rotated_source =
        rotate_vq_input(multi_source, cohort);
    for (int route = 0; route < 2; ++route) {
        const int global_expert = multi_ids[route];
        const int local_expert =
            global_expert == 0 ? 1 : 0;
        for (int output = 0;
             output < width;
             ++output) {
            float expected_value = 0.0f;
            const auto weight_offset = (
                static_cast<std::size_t>(
                    local_expert)
                    * width
                + output
            ) * width;
            for (int column = 0;
                 column < width;
                 ++column) {
                expected_value +=
                    rotated_source[column]
                    * cohort.dense[
                        weight_offset + column];
            }
            require_close(
                multi_actual[
                    route * width + output],
                expected_value,
                2e-3f);
        }
    }

    // Using the same projection fixture three times intentionally exercises
    // projection-buffer offsets and HSG1 variant de-duplication in gate/up.
    const auto ffn =
        mfq::metal::MlxRoutedSwiGluFfn::
            from_blobs(
                fixture.blob,
                fixture.blob,
                fixture.blob);
    const std::vector<float> route_weights{
        0.50f, 0.30f, 0.20f,
        0.15f, 0.55f, 0.30f,
        0.25f, 0.35f, 0.40f,
    };
    const auto ffn_actual = evaluated_floats(
        ffn.forward(
            mlx::core::array(
                input.begin(),
                mlx::core::Shape{
                    tokens,
                    width,
                }),
            mlx::core::array(
                ids.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                }),
            mlx::core::array(
                route_weights.begin(),
                mlx::core::Shape{
                    tokens,
                    routes,
                })));

    std::vector<float> expected(
        tokens * width,
        0.0f);
    for (int token = 0; token < tokens; ++token) {
        std::vector<float> source(
            input.begin() + token * width,
            input.begin() + (token + 1) * width);
        for (int route = 0;
             route < routes;
             ++route) {
            const int expert =
                ids[token * routes + route];
            std::vector<float> hidden(width);
            for (int column = 0;
                 column < width;
                 ++column) {
                const float projected =
                    routed_vq_dot(
                        source,
                        fixture,
                        expert,
                        column);
                hidden[column] =
                    projected
                    / (
                        1.0f
                        + std::exp(-projected)
                    )
                    * projected;
            }
            for (int output = 0;
                 output < width;
                 ++output) {
                expected[token * width + output] +=
                    route_weights[
                        token * routes + route]
                    * routed_vq_dot(
                        hidden,
                        fixture,
                        expert,
                        output);
            }
        }
    }
    for (
        std::size_t index = 0;
        index < expected.size();
        ++index
    ) {
        require_close(
            ffn_actual[index],
            expected[index],
            8e-3f);
    }
}

void test_grouped_vq_mmq_prefill() {
    constexpr int tokens = 49;
    constexpr int routes = 2;
    constexpr int width = 24;
    auto fixture = make_vq_moe_fixture({
        make_jsc_nvq(
            width,
            width,
            "NVQ3J",
            2,
            4,
            8),
        make_jsc_nvq(width, width),
    });
    const auto weight =
        mfq::metal::MlxMoeWeight::from_blob(
            fixture.blob);
    std::vector<float> input(tokens * width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 7 + 5) % 23) - 11)
            / 128.0f;
    }
    std::vector<std::int32_t> ids(tokens * routes);
    for (int row = 0; row < tokens * routes; ++row) {
        ids[row] = row < 31 ? 0 : 1;
    }
    auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, width}),
        mlx::core::float16);
    const auto actual = evaluated_floats(
        weight.routed_matmul(
            input_array,
            mlx::core::array(
                ids.begin(),
                mlx::core::Shape{tokens, routes})));
    auto ids_array = mlx::core::array(
        ids.begin(),
        mlx::core::Shape{tokens, routes});
    auto route_order = mlx::core::contiguous(
        mlx::core::astype(
            mlx::core::argsort(
                mlx::core::reshape(
                    ids_array,
                    mlx::core::Shape{tokens * routes})),
            mlx::core::int32));
    auto block_plan = weight.build_grouped_vq_mmq_plan(
        ids_array,
        route_order);
    require(block_plan.block_rows == 32, "unexpected block row size");
    require(
        block_plan.route_count == tokens * routes,
        "unexpected block plan route count");
    auto plain_sorted = weight.routed_matmul_sorted(
        input_array,
        ids_array,
        route_order,
        false,
        false,
        0.0f,
        &block_plan);
    const auto plain_actual = evaluated_floats(
        mlx::core::take(
            std::move(plain_sorted),
            mlx::core::argsort(route_order),
            0));
    auto fused_sorted = weight.routed_matmul_sorted(
        input_array,
        ids_array,
        route_order,
        false,
        true,
        0.0f,
        &block_plan);
    const auto fused_actual = evaluated_floats(
        mlx::core::reshape(
            mlx::core::take(
                std::move(fused_sorted),
                mlx::core::argsort(route_order),
                0),
            mlx::core::Shape{tokens, routes, width / 2}));
    for (int token = 0; token < tokens; ++token) {
        std::vector<float> source(
            input.begin() + token * width,
            input.begin() + (token + 1) * width);
        for (int route = 0; route < routes; ++route) {
            int expert = ids[token * routes + route];
            for (int output = 0; output < width; ++output) {
                require_close(
                    plain_actual[
                        (token * routes + route) * width + output],
                    actual[(token * routes + route) * width + output],
                    4e-3f);
                require_close(
                    actual[(token * routes + route) * width + output],
                    routed_vq_dot(
                        source,
                        fixture,
                        expert,
                        output),
                    4e-3f);
            }
            for (int output = 0; output < width / 2; ++output) {
                const float gate = routed_vq_dot(
                    source,
                    fixture,
                    expert,
                    output);
                const float up = routed_vq_dot(
                    source,
                    fixture,
                    expert,
                    output + width / 2);
                require_close(
                    fused_actual[
                        (token * routes + route) * (width / 2)
                        + output],
                    gate / (1.0f + std::exp(-gate)) * up,
                    6e-3f);
            }
        }
    }
}

void expect_invalid(
    const std::vector<std::uint8_t>& blob,
    const std::string& label) {
    bool rejected = false;
    try {
        (void)mfq::metal::MlxMoeWeight::
            from_blob(blob);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(
        rejected,
        "malformed NINTM was accepted: " + label);
}

void test_container_validation() {
    const auto one =
        make_nint_tensor(4, 1, 32, 1);
    const auto two =
        make_nint_tensor(4, 2, 32, 2);

    // Legacy NIM1 remains a valid all-NINT container.
    const auto legacy = make_nim1(
        2,
        1,
        32,
        two);
    const auto legacy_weight =
        mfq::metal::MlxMoeWeight::from_blob(
            legacy);
    require(
        legacy_weight.experts() == 2
            && legacy_weight.out_per_expert() == 1,
        "valid NIM1 container was decoded incorrectly");

    expect_invalid(
        std::vector<std::uint8_t>(10, 0),
        "truncated header");

    std::vector<std::uint8_t> impossible_size{
        'N', 'I', 'M', '2',
    };
    append<std::uint32_t>(
        impossible_size,
        std::numeric_limits<std::int32_t>::max());
    append<std::uint32_t>(impossible_size, 1);
    append<std::uint32_t>(impossible_size, 32);
    append<std::uint32_t>(impossible_size, 1);
    expect_invalid(
        impossible_size,
        "expert count larger than payload");

    auto bad_magic = legacy;
    bad_magic[0] = 'X';
    expect_invalid(bad_magic, "bad magic");

    auto zero_pools = legacy;
    std::fill(
        zero_pools.begin() + 16,
        zero_pools.begin() + 20,
        0);
    expect_invalid(zero_pools, "zero pools");

    expect_invalid(
        make_raw_nim2(
            2,
            1,
            32,
            {
                {{0}, "NINT4", one, {}},
                {{0}, "NINT4", one, {}},
            }),
        "duplicate expert ownership");
    expect_invalid(
        make_raw_nim2(
            3,
            1,
            32,
            {
                {{0}, "NINT4", one, {}},
                {{1}, "NINT4", one, {}},
            }),
        "missing expert");
    expect_invalid(
        make_raw_nim2(
            2,
            1,
            32,
            {
                {{0}, "NINT4", one, {}},
                {{2}, "NINT4", one, {}},
            }),
        "out-of-range expert");
    expect_invalid(
        make_raw_nim2(
            1,
            1,
            32,
            {
                {{0}, "NVQ2", one, {}},
            }),
        "unsupported cohort dtype");
    expect_invalid(
        make_raw_nim2(
            1,
            1,
            32,
            {
                {
                    {0},
                    "NINT4",
                    one,
                    {1},
                },
            }),
        "unexpected runtime payload");
    expect_invalid(
        make_raw_nim2(
            1,
            2,
            32,
            {
                {{0}, "NINT4", one, {}},
            }),
        "nested row mismatch");

    auto trailing = make_raw_nim2(
        1,
        1,
        32,
        {
            {{0}, "NINT4", one, {}},
        });
    trailing.push_back(0);
    expect_invalid(trailing, "trailing byte");

    auto truncated = make_raw_nim2(
        1,
        1,
        32,
        {
            {{0}, "NINT4", one, {}},
        });
    truncated.pop_back();
    expect_invalid(truncated, "truncated payload");

    auto non_ascii = make_raw_nim2(
        1,
        1,
        32,
        {
            {{0}, std::string(1, '\xff'), one, {}},
        });
    expect_invalid(non_ascii, "non-ASCII dtype");

    auto wrong_nested_type = make_raw_nim2(
        1,
        1,
        32,
        {
            {{0}, "NINT8-0", one, {}},
        });
    expect_invalid(
        wrong_nested_type,
        "dtype/payload mismatch");

    auto unexpected_runtime =
        make_plain_nvq(24, 24);
    unexpected_runtime.runtime = {1};
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(unexpected_runtime)}).blob,
        "matrix VQ runtime metadata");

    auto missing_hsg1 =
        make_rotated_nepq1_s(24, 24);
    missing_hsg1.runtime.clear();
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(missing_hsg1)}).blob,
        "missing HSG1 metadata");

    auto bad_hsg1_magic =
        make_rotated_nepq1_s(24, 24);
    bad_hsg1_magic.runtime[0] = 'X';
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(bad_hsg1_magic)}).blob,
        "bad HSG1 magic");

    auto mismatched_hsg1 =
        make_rotated_nepq1_s(24, 24);
    const std::uint32_t wrong_block = 16;
    std::memcpy(
        mismatched_hsg1.runtime.data() + 8,
        &wrong_block,
        sizeof(wrong_block));
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(mismatched_hsg1)}).blob,
        "mismatched HSG1 block");

    auto invalid_hsg1_sign =
        make_rotated_nepq1_s(24, 24);
    invalid_hsg1_sign.runtime[20] = 0;
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(invalid_hsg1_sign)}).blob,
        "invalid HSG1 sign");

    auto truncated_vq =
        make_npq(24, 24);
    truncated_vq.blob.pop_back();
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(truncated_vq)}).blob,
        "truncated VQ cohort");

    auto trailing_vq =
        make_nvq1_s(24, 24);
    trailing_vq.blob.push_back(0);
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(trailing_vq)}).blob,
        "trailing VQ cohort byte");

    auto wrong_vq_dtype =
        make_plain_nvq(24, 24);
    wrong_vq_dtype.dtype = "NPQ0-L";
    expect_invalid(
        make_vq_moe_fixture(
            {std::move(wrong_vq_dtype)}).blob,
        "VQ dtype/blob mismatch");

    auto wrong_vq_shape =
        make_vq_moe_fixture({
            make_plain_nvq(24, 24),
        }).blob;
    const std::uint32_t wrong_output = 23;
    std::memcpy(
        wrong_vq_shape.data() + 8,
        &wrong_output,
        sizeof(wrong_output));
    expect_invalid(
        wrong_vq_shape,
        "VQ outer/nested shape mismatch");

    expect_invalid(
        make_vq_moe_fixture({
            make_rotated_nepq1_s(24, 24),
            make_rotated_nepq1_s(
                24,
                24,
                0x123456789abcdef0ull,
                1),
        }).blob,
        "conflicting HSG1 signs");
}

} // namespace

int main() {
    try {
        test_all_families_and_projections();
        test_swiglu_ffn();
        test_vq_cohorts_and_ffn();
        test_grouped_vq_mmq_prefill();
        test_container_validation();
        test_streamed_cccp_residency();
        std::cout
            << "MFQ native heterogeneous NINTM/streamed "
               "CCCP routed Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
