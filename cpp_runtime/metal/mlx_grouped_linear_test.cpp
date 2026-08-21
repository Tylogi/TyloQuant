#include "mlx_grouped_linear.h"

#include "../nvq_codebooks.generated.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mlx/memory.h>
#include <mlx/mlx.h>

namespace {

constexpr int kInputSize = 64;
constexpr int kGroupSize = 16;
constexpr int kGroups = 4;

template <typename T>
void append(std::vector<std::uint8_t>& blob, T value) {
    const auto* bytes =
        reinterpret_cast<const std::uint8_t*>(&value);
    blob.insert(blob.end(), bytes, bytes + sizeof(value));
}

void append_magic(
    std::vector<std::uint8_t>& blob,
    std::string_view magic);

template <typename T>
std::vector<std::uint8_t> pack_values(
    const std::vector<T>& values,
    int bits) {
    std::vector<std::uint8_t> packed(
        (values.size() * static_cast<std::size_t>(bits) + 7) / 8,
        0);
    for (std::size_t index = 0; index < values.size(); ++index) {
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

struct Fixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    int output_size = 0;
};

Fixture make_nint_fixture(int bits, int output_size) {
    const auto maximum = (1u << bits) - 1u;
    const auto metadata_count =
        static_cast<std::size_t>(output_size) * kGroups;
    const auto value_count = metadata_count * kGroupSize;

    std::vector<std::uint8_t> sub_scale(metadata_count, 1);
    std::vector<std::uint8_t> sub_min(metadata_count);
    for (std::size_t index = 0; index < metadata_count; ++index) {
        sub_min[index] = static_cast<std::uint8_t>(
            (index + static_cast<std::size_t>(bits)) & 1u);
    }
    std::vector<std::uint8_t> quantized(value_count);
    for (std::size_t index = 0; index < value_count; ++index) {
        quantized[index] = static_cast<std::uint8_t>(
            (index * static_cast<std::size_t>(bits + 3) + 1)
            % (maximum + 1));
    }

    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(
        blob,
        static_cast<std::uint8_t>(bits));
    append<std::uint8_t>(blob, 1);
    append<std::int32_t>(blob, kGroupSize);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, kInputSize);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, kInputSize);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(output_size));
    append<std::uint32_t>(blob, kGroups);

    // Exactly representable FP16 anchors: scale=1/8, minimum=1/16.
    for (int output = 0; output < output_size; ++output) {
        append<std::uint16_t>(blob, 0x3000);
    }
    for (int output = 0; output < output_size; ++output) {
        append<std::uint16_t>(blob, 0x2c00);
    }
    for (const auto* values : {&sub_scale, &sub_min}) {
        const auto packed = pack_values(*values, 1);
        blob.insert(blob.end(), packed.begin(), packed.end());
    }
    const auto packed_q = pack_values(quantized, bits);
    blob.insert(blob.end(), packed_q.begin(), packed_q.end());

    std::vector<float> dense(
        static_cast<std::size_t>(output_size) * kInputSize);
    for (int output = 0; output < output_size; ++output) {
        for (int column = 0; column < kInputSize; ++column) {
            const int group = column / kGroupSize;
            const auto metadata =
                static_cast<std::size_t>(output) * kGroups + group;
            const auto value =
                metadata * kGroupSize + column % kGroupSize;
            dense[
                static_cast<std::size_t>(output) * kInputSize
                + column
            ] =
                0.125f * static_cast<float>(quantized[value])
                - 0.0625f
                    * static_cast<float>(sub_min[metadata]);
        }
    }
    return {
        std::move(blob),
        std::move(dense),
        output_size,
    };
}

std::vector<std::uint8_t> make_nint4_gs24_blob(
    int input_size,
    int output_size,
    int phase) {
    constexpr int bits = 4;
    constexpr int sub_bits = 8;
    constexpr int group_size = 24;
    const int groups = (input_size + group_size - 1) / group_size;
    const auto metadata_count =
        static_cast<std::size_t>(output_size) * groups;
    const auto value_count = metadata_count * group_size;
    std::vector<std::uint8_t> sub_scale(metadata_count);
    std::vector<std::uint8_t> sub_min(metadata_count);
    std::vector<std::uint8_t> quantized(value_count);
    for (std::size_t index = 0; index < metadata_count; ++index) {
        sub_scale[index] = static_cast<std::uint8_t>(
            1 + (index + static_cast<std::size_t>(phase)) % 3);
        sub_min[index] = static_cast<std::uint8_t>(
            (index + static_cast<std::size_t>(phase)) % 2);
    }
    for (std::size_t index = 0; index < value_count; ++index) {
        quantized[index] = static_cast<std::uint8_t>(
            (index * static_cast<std::size_t>(phase + 5) + 3) & 15u);
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
        append<std::uint16_t>(blob, 0x3000);
    }
    for (int output = 0; output < output_size; ++output) {
        append<std::uint16_t>(blob, 0x2c00);
    }
    blob.insert(blob.end(), sub_scale.begin(), sub_scale.end());
    blob.insert(blob.end(), sub_min.begin(), sub_min.end());
    const auto packed = pack_values(quantized, bits);
    blob.insert(blob.end(), packed.begin(), packed.end());
    return blob;
}

Fixture make_q8_fixture(int output_size) {
    constexpr std::uint16_t scale_bits[] = {
        0x2c00,
        0x3000,
        0x3400,
        0x3800,
    };
    constexpr float scale_values[] = {
        0.0625f,
        0.125f,
        0.25f,
        0.5f,
    };
    constexpr int q8_groups = kInputSize / 32;
    std::vector<std::int8_t> quantized(
        static_cast<std::size_t>(output_size) * kInputSize);
    for (std::size_t index = 0; index < quantized.size(); ++index) {
        quantized[index] = static_cast<std::int8_t>(
            static_cast<int>((index * 11 + 5) % 41) - 20);
    }

    std::vector<std::uint8_t> blob{'N', 'I', '8', '0'};
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, kInputSize);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, kInputSize);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(output_size));
    append<std::uint32_t>(blob, q8_groups);
    for (int output = 0; output < output_size; ++output) {
        for (int group = 0; group < q8_groups; ++group) {
            const int scale_index = (output + group) & 3;
            append<std::uint16_t>(
                blob,
                scale_bits[scale_index]);
            const auto offset =
                static_cast<std::size_t>(output) * kInputSize
                + group * 32;
            const auto* bytes =
                reinterpret_cast<const std::uint8_t*>(
                    quantized.data() + offset);
            blob.insert(blob.end(), bytes, bytes + 32);
        }
    }

    std::vector<float> dense(
        static_cast<std::size_t>(output_size) * kInputSize);
    for (int output = 0; output < output_size; ++output) {
        for (int column = 0; column < kInputSize; ++column) {
            const int group = column / 32;
            dense[
                static_cast<std::size_t>(output) * kInputSize
                + column
            ] =
                scale_values[(output + group) & 3]
                * static_cast<float>(
                    quantized[
                        static_cast<std::size_t>(output)
                            * kInputSize
                        + column]);
        }
    }
    return {
        std::move(blob),
        std::move(dense),
        output_size,
    };
}

Fixture make_tpq_int4_fixture(int output_size) {
    std::vector<std::int8_t> quantized(
        static_cast<std::size_t>(output_size)
            * kInputSize);
    for (std::size_t index = 0;
         index < quantized.size();
         ++index) {
        quantized[index] =
            static_cast<std::int8_t>(
                static_cast<int>(
                    (index * 9 + 5) % 16)
                - 8);
    }
    std::vector<std::uint8_t> packed(
        static_cast<std::size_t>(output_size)
            * (kInputSize / 2));
    for (int output = 0;
         output < output_size;
         ++output) {
        for (int pair = 0;
             pair < kInputSize / 2;
             ++pair) {
            const auto source =
                static_cast<std::size_t>(output)
                    * kInputSize
                + pair * 2;
            packed[
                static_cast<std::size_t>(output)
                    * (kInputSize / 2)
                + pair
            ] =
                static_cast<std::uint8_t>(
                    quantized[source] + 8)
                | static_cast<std::uint8_t>(
                    (quantized[source + 1] + 8)
                    << 4);
        }
    }

    std::vector<std::uint8_t> blob;
    append_magic(blob, "CI41");
    append<std::uint8_t>(blob, 1);
    blob.insert(blob.end(), 3, 0);
    append<std::uint32_t>(blob, 64);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(
        blob,
        kInputSize);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(
        blob,
        output_size);
    append<std::int64_t>(
        blob,
        kInputSize);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(
            output_size));
    append<std::uint32_t>(blob, 1);
    blob.insert(
        blob.end(),
        packed.begin(),
        packed.end());
    for (int output = 0;
         output < output_size;
         ++output) {
        append<std::uint16_t>(
            blob,
            static_cast<std::uint16_t>(
                0x2c00 + (output % 4) * 0x0400));
    }

    constexpr float scales[] = {
        0.0625f,
        0.125f,
        0.25f,
        0.5f,
    };
    std::vector<float> dense(
        static_cast<std::size_t>(output_size)
            * kInputSize);
    for (int output = 0;
         output < output_size;
         ++output) {
        for (int column = 0;
             column < kInputSize;
             ++column) {
            dense[
                static_cast<std::size_t>(output)
                    * kInputSize
                + column
            ] =
                static_cast<float>(
                    quantized[
                        static_cast<std::size_t>(output)
                            * kInputSize
                        + column
                    ])
                * scales[output % 4];
        }
    }
    return {
        std::move(blob),
        std::move(dense),
        output_size,
    };
}

struct TpqPqFixture {
    std::string dtype;
    Fixture fixture;
};

TpqPqFixture make_tpq_pq_fixture(
    std::string dtype,
    int tier,
    int vector_size,
    int entries,
    int bits,
    int output_size) {
    const int blocks =
        kInputSize / vector_size;
    std::vector<float> codebook(
        static_cast<std::size_t>(entries)
            * vector_size);
    for (int entry = 0;
         entry < entries;
         ++entry) {
        for (int component = 0;
             component < vector_size;
             ++component) {
            codebook[
                static_cast<std::size_t>(entry)
                    * vector_size
                + component
            ] =
                static_cast<float>(
                    (
                        entry * 3
                        + component * 5
                    ) % 17
                    - 8)
                / 64.0f;
        }
    }
    std::vector<std::uint16_t> indices(
        static_cast<std::size_t>(output_size)
            * blocks);
    for (std::size_t index = 0;
         index < indices.size();
         ++index) {
        indices[index] =
            static_cast<std::uint16_t>(
                (
                    index * 73
                    + entries
                    - 1
                ) % entries);
    }
    if (!indices.empty()) {
        indices.front() =
            static_cast<std::uint16_t>(
                entries - 1);
    }
    const auto packed =
        pack_values(indices, bits);

    std::vector<std::uint8_t> blob;
    append_magic(blob, "CPQ1");
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(
        blob,
        static_cast<std::uint8_t>(tier));
    append<std::uint8_t>(
        blob,
        static_cast<std::uint8_t>(
            vector_size));
    append<std::uint8_t>(
        blob,
        static_cast<std::uint8_t>(bits));
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(
        blob,
        kInputSize);
    append<std::uint32_t>(blob, 2);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(
            entries));
    append<std::int64_t>(
        blob,
        output_size);
    append<std::int64_t>(
        blob,
        kInputSize);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(
            output_size));
    for (const float value : codebook) {
        append<float>(blob, value);
    }
    blob.insert(
        blob.end(),
        packed.begin(),
        packed.end());

    std::vector<float> dense(
        static_cast<std::size_t>(output_size)
            * kInputSize);
    for (int output = 0;
         output < output_size;
         ++output) {
        for (int block = 0;
             block < blocks;
             ++block) {
            const auto code =
                indices[
                    static_cast<std::size_t>(output)
                        * blocks
                    + block
                ];
            for (int component = 0;
                 component < vector_size;
                 ++component) {
                dense[
                    static_cast<std::size_t>(output)
                        * kInputSize
                    + block * vector_size
                    + component
                ] =
                    codebook[
                        static_cast<std::size_t>(code)
                            * vector_size
                        + component
                    ];
            }
        }
    }
    return {
        std::move(dtype),
        {
            std::move(blob),
            std::move(dense),
            output_size,
        },
    };
}

std::vector<TpqPqFixture>
make_tpq_pq_fixtures() {
    return {
        make_tpq_pq_fixture(
            "TPQ-X", 1, 8, 256, 8, 3),
        make_tpq_pq_fixture(
            "TPQ-X", 1, 8, 256, 12, 4),
        make_tpq_pq_fixture(
            "TPQ-X", 1, 8, 256, 14, 5),
        make_tpq_pq_fixture(
            "TPQ-W", 2, 8, 4096, 12, 3),
        make_tpq_pq_fixture(
            "TPQ-W", 2, 8, 4096, 14, 4),
        make_tpq_pq_fixture(
            "TPQ-W", 2, 8, 4096, 16, 5),
        make_tpq_pq_fixture(
            "TPQ-V", 3, 4, 256, 8, 3),
        make_tpq_pq_fixture(
            "TPQ-V", 3, 4, 256, 12, 4),
        make_tpq_pq_fixture(
            "TPQ-V", 3, 4, 256, 14, 5),
        make_tpq_pq_fixture(
            "TPQ-VV", 4, 4, 4096, 12, 3),
        make_tpq_pq_fixture(
            "TPQ-VV", 4, 4, 4096, 14, 4),
        make_tpq_pq_fixture(
            "TPQ-VV", 4, 4, 4096, 16, 5),
    };
}

void append_magic(
    std::vector<std::uint8_t>& blob,
    std::string_view magic) {
    if (magic.size() != 4) {
        throw std::runtime_error(
            "VQ fixture magic must have four bytes");
    }
    blob.insert(
        blob.end(),
        magic.begin(),
        magic.end());
}

void append_matrix_header(
    std::vector<std::uint8_t>& blob,
    std::string_view magic,
    std::uint8_t profile,
    std::uint8_t state_bits,
    int output_size) {
    append_magic(blob, magic);
    append<std::uint8_t>(blob, profile);
    append<std::uint8_t>(blob, state_bits);
    append<std::uint16_t>(blob, 24);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, kInputSize);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output_size);
    append<std::int64_t>(blob, kInputSize);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(output_size));
}

void append_ones(
    std::vector<std::uint8_t>& blob,
    int count) {
    for (int index = 0; index < count; ++index) {
        append<std::uint16_t>(blob, 0x3c00);
    }
}

void append_vq_streams(
    std::vector<std::uint8_t>& blob,
    int output_size,
    int vector_size,
    int state_bits,
    int index_bits,
    int auxiliary_bits,
    bool signs,
    std::uint16_t index_value = 0) {
    constexpr int groups =
        (kInputSize + 23) / 24;
    const int vectors =
        (kInputSize + vector_size - 1) / vector_size;
    append_ones(blob, output_size);
    const auto state = pack_values(
        std::vector<std::uint16_t>(
            static_cast<std::size_t>(output_size)
                * groups,
            1),
        state_bits);
    blob.insert(
        blob.end(),
        state.begin(),
        state.end());
    const auto indices = pack_values(
        std::vector<std::uint16_t>(
            static_cast<std::size_t>(output_size)
                * vectors,
            index_value),
        index_bits);
    blob.insert(
        blob.end(),
        indices.begin(),
        indices.end());
    if (signs) {
        constexpr int sign_groups =
            (kInputSize + 7) / 8;
        const auto auxiliary = pack_values(
            std::vector<std::uint16_t>(
                static_cast<std::size_t>(output_size)
                    * sign_groups,
                0),
            7);
        blob.insert(
            blob.end(),
            auxiliary.begin(),
            auxiliary.end());
    } else if (auxiliary_bits != 0) {
        const auto auxiliary = pack_values(
            std::vector<std::uint16_t>(
                static_cast<std::size_t>(output_size)
                    * groups,
                0),
            auxiliary_bits);
        blob.insert(
            blob.end(),
            auxiliary.begin(),
            auxiliary.end());
    }
}

std::vector<float> repeated_vq_dense(
    int output_size,
    const std::vector<float>& vector) {
    std::vector<float> result(
        static_cast<std::size_t>(output_size)
            * kInputSize);
    for (int output = 0;
         output < output_size;
         ++output) {
        for (int column = 0;
             column < kInputSize;
             ++column) {
            result[
                static_cast<std::size_t>(output)
                    * kInputSize
                + column
            ] = vector[
                static_cast<std::size_t>(column)
                % vector.size()
            ];
        }
    }
    return result;
}

struct NamedVqFixture {
    std::string dtype;
    Fixture fixture;
};

NamedVqFixture make_plain_vq(
    std::string dtype,
    int profile,
    int vector_size,
    int output_size,
    bool index_parity = false) {
    std::vector<std::uint8_t> blob;
    append_matrix_header(
        blob,
        "NVQ1",
        static_cast<std::uint8_t>(
            profile | 0x40
            | (index_parity ? 0x80 : 0)),
        4,
        output_size);
    for (int entry = 0; entry < 256; ++entry) {
        append<std::uint16_t>(blob, 0);
    }
    append_vq_streams(
        blob,
        output_size,
        vector_size,
        4,
        8,
        7,
        true,
        index_parity ? 128 : 0);
    std::vector<float> decoded(
        static_cast<std::size_t>(vector_size),
        1.0f);
    if (index_parity) {
        decoded.back() = -1.0f;
    }
    return {
        std::move(dtype),
        {
            std::move(blob),
            repeated_vq_dense(
                output_size,
                decoded),
            output_size,
        },
    };
}

NamedVqFixture make_jsc_vq(
    std::string dtype,
    int profile,
    int vector_size,
    int index_bits,
    int output_size,
    bool group64 = false) {
    const int entries = 1 << index_bits;
    std::vector<std::uint8_t> blob;
    append_matrix_header(
        blob,
        "NVQ1",
        static_cast<std::uint8_t>(profile | 0x20),
        4,
        output_size);
    append<std::uint8_t>(blob, group64 ? 2 : 1);
    append<std::uint8_t>(blob, 2);
    append<std::uint8_t>(blob, 16);
    append<std::uint8_t>(blob, 0);
    append_ones(blob, 16);
    for (int state = 0; state < 16; ++state) {
        append<std::uint8_t>(
            blob,
            static_cast<std::uint8_t>(state & 1));
    }
    append<std::uint8_t>(blob, group64 ? 1 : 0);
    blob.insert(blob.end(), 11, 0);
    for (int bank = 0; bank < 2; ++bank) {
        for (int entry = 0;
             entry < entries;
             ++entry) {
            for (int component = 0;
                 component < vector_size;
                 ++component) {
                append<std::int8_t>(
                    blob,
                    static_cast<std::int8_t>(bank + 1));
            }
        }
    }
    if (group64) {
        constexpr int groups = (kInputSize + 23) / 24;
        append_ones(blob, output_size);
        for (int output = 0; output < output_size; ++output) {
            for (int group = 0; group < groups; ++group) {
                append<std::uint64_t>(blob, std::uint64_t{1} << 60);
            }
        }
    } else {
        append_vq_streams(
            blob,
            output_size,
            vector_size,
            4,
            index_bits,
            7,
            true);
    }
    return {
        std::move(dtype),
        {
            std::move(blob),
            repeated_vq_dense(
                output_size,
                std::vector<float>(
                    static_cast<std::size_t>(vector_size),
                    2.0f)),
            output_size,
        },
    };
}

std::vector<float> ternary_vector(
    std::uint16_t word,
    float delta) {
    std::vector<float> result(8);
    for (int component = 0;
         component < 8;
         ++component) {
        const auto digit =
            (word >> (2 * component)) & 3u;
        result[component] =
            static_cast<float>(
                static_cast<int>(digit) - 1)
            + delta;
    }
    return result;
}

void append_nvq1_s_table(
    std::vector<std::uint8_t>& blob) {
    for (int bank = 0; bank < 2; ++bank) {
        for (int entry = 0; entry < 512; ++entry) {
            append<std::uint16_t>(
                blob,
                mfq::nvq_codebooks::
                    kNvq1LCodebookPacked[entry * 4]);
        }
    }
}

NamedVqFixture make_nvq1_l_fixture(int output_size) {
    std::vector<std::uint8_t> blob;
    append_matrix_header(
        blob,
        "NQ1L",
        1,
        3,
        output_size);
    append_vq_streams(
        blob,
        output_size,
        8,
        3,
        11,
        1,
        false);
    return {
        "NVQ1-L",
        {
            std::move(blob),
            repeated_vq_dense(
                output_size,
                ternary_vector(
                    mfq::nvq_codebooks::
                        kNvq1LCodebookPacked[0],
                    0.125f)),
            output_size,
        },
    };
}

NamedVqFixture make_nvq1_s_fixture(int output_size) {
    std::vector<std::uint8_t> blob;
    append_matrix_header(
        blob,
        "NQ1S",
        1,
        4,
        output_size);
    append_nvq1_s_table(blob);
    append_vq_streams(
        blob,
        output_size,
        8,
        4,
        9,
        1,
        false);
    return {
        "NVQ1-S",
        {
            std::move(blob),
            repeated_vq_dense(
                output_size,
                ternary_vector(
                    mfq::nvq_codebooks::
                        kNvq1LCodebookPacked[0],
                    0.15625f)),
            output_size,
        },
    };
}

std::vector<std::uint8_t> npq_table(bool short_profile) {
    const int states = short_profile ? 4 : 8;
    const int first_entries = 8;
    const int second_entries =
        short_profile ? 8 : 16;
    std::vector<std::uint8_t> result(
        short_profile ? 320 : 832,
        0);
    result[0] = static_cast<std::uint8_t>(
        short_profile ? 2 : 1);
    result[1] = static_cast<std::uint8_t>(states);
    result[2] = 3;
    result[3] = static_cast<std::uint8_t>(
        short_profile ? 3 : 4);
    result[4] = 24;
    result[5] = 8;
    const std::uint16_t one = 0x3c00;
    for (int state = 0; state < states; ++state) {
        std::memcpy(
            result.data() + 8 + state * 2,
            &one,
            sizeof(one));
    }
    const int first_count =
        states * first_entries * 4;
    const int second_count =
        states * second_entries * 4;
    std::fill_n(
        result.begin() + 64,
        first_count,
        std::uint8_t{1});
    std::fill_n(
        result.begin() + 64 + first_count,
        second_count,
        std::uint8_t{2});
    return result;
}

NamedVqFixture make_npq_fixture(
    bool short_profile,
    int output_size) {
    std::vector<std::uint8_t> blob;
    append_matrix_header(
        blob,
        short_profile ? "NPQS" : "NPQL",
        static_cast<std::uint8_t>(
            short_profile ? 2 : 1),
        static_cast<std::uint8_t>(
            short_profile ? 2 : 3),
        output_size);
    const auto table = npq_table(short_profile);
    blob.insert(
        blob.end(),
        table.begin(),
        table.end());
    append_vq_streams(
        blob,
        output_size,
        8,
        short_profile ? 2 : 3,
        short_profile ? 6 : 7,
        0,
        false);
    return {
        short_profile ? "NPQ0-S" : "NPQ0-L",
        {
            std::move(blob),
            repeated_vq_dense(
                output_size,
                {
                    1.0f, 1.0f, 1.0f, 1.0f,
                    2.0f, 2.0f, 2.0f, 2.0f,
                }),
            output_size,
        },
    };
}

std::vector<NamedVqFixture> make_vq_fixtures() {
    std::vector<NamedVqFixture> result;
    result.reserve(13);
    result.push_back(
        make_plain_vq("NVQ2", 1, 8, 3));
    result.push_back(
        make_plain_vq("NVQ2", 1, 8, 4, true));
    result.push_back(
        make_jsc_vq("NVQ2J", 1, 8, 8, 4));
    result.push_back(
        make_jsc_vq("NVQ2J-L", 4, 8, 10, 5));
    result.push_back(
        make_jsc_vq("NVQ2J-XL", 5, 8, 12, 3));
    result.push_back(
        make_jsc_vq("NVQ2J-XL", 5, 8, 12, 3, true));
    result.push_back(
        make_plain_vq("NVQ3", 2, 4, 4));
    result.push_back(
        make_jsc_vq("NVQ3J", 2, 4, 8, 5));
    result.push_back(
        make_jsc_vq("NVQ3J-512", 3, 4, 9, 3));
    result.push_back(
        make_jsc_vq("NVQ3J-L", 6, 4, 10, 4));
    result.push_back(make_nvq1_l_fixture(5));
    result.push_back(make_nvq1_s_fixture(3));
    result.push_back(make_npq_fixture(false, 4));
    result.push_back(make_npq_fixture(true, 5));
    return result;
}

std::vector<std::uint8_t> make_nepq_blob() {
    constexpr int experts = 2;
    constexpr int out_per_expert = 2;
    constexpr int outputs = experts * out_per_expert;
    std::vector<std::uint8_t> blob;
    append_magic(blob, "NEP1");
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, 0);
    append<std::uint8_t>(blob, 4);
    append<std::uint8_t>(blob, 0);
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, out_per_expert);
    append<std::uint32_t>(blob, kInputSize);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, 0);
    append<std::uint64_t>(blob, 0);
    const auto table = npq_table(true);
    blob.insert(
        blob.end(),
        table.begin(),
        table.end());
    append_ones(blob, outputs);
    constexpr int groups =
        (kInputSize + 23) / 24;
    const auto states = pack_values(
        std::vector<std::uint16_t>(
            outputs * groups,
            1),
        2);
    blob.insert(
        blob.end(),
        states.begin(),
        states.end());
    const auto indices = pack_values(
        std::vector<std::uint16_t>(
            outputs * (kInputSize / 8),
            0),
        6);
    blob.insert(
        blob.end(),
        indices.begin(),
        indices.end());
    // ceil(groups / 4) == 1 table selector per row.
    blob.insert(blob.end(), outputs, 0);
    return blob;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    float actual,
    float expected,
    float tolerance) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "grouped linear result mismatch: actual="
            + std::to_string(actual)
            + " expected=" + std::to_string(expected));
    }
}

float reference(
    const std::vector<float>& source,
    std::size_t source_offset,
    const Fixture& fixture,
    int output) {
    float result = 0.0f;
    for (int column = 0; column < kInputSize; ++column) {
        result +=
            source[source_offset + column]
            * fixture.dense[
                static_cast<std::size_t>(output) * kInputSize
                + column];
    }
    return result;
}

void require_unsupported(
    const mfq::metal::MlxGroupedLinear& grouped,
    const mlx::core::array& input) {
    bool rejected = false;
    try {
        (void)grouped.matmul(input);
    } catch (
        const mfq::metal::MlxGroupedLinearUnsupported&
    ) {
        rejected = true;
    }
    require(
        rejected,
        "unsupported grouped linear input was accepted");
}

void require_one_row_matches(
    const mfq::metal::MlxGroupedLinear& grouped,
    const mlx::core::array& input,
    const std::vector<float>& source,
    const std::vector<const Fixture*>& fixtures,
    float tolerance = 8e-4f,
    std::string_view context = {}) {
    auto outputs = grouped(input);
    require(
        outputs.size() == fixtures.size(),
        "zero-copy grouped linear output count mismatch");
    for (
        std::size_t projection = 0;
        projection < outputs.size();
        ++projection
    ) {
        auto values_array = mlx::core::contiguous(
            mlx::core::astype(
                outputs[projection],
                mlx::core::float32));
        values_array.eval();
        require(
            outputs[projection].shape()
                == mlx::core::Shape{
                    1,
                    fixtures[projection]->output_size,
                },
            "zero-copy grouped output shape mismatch");
        const auto* values =
            values_array.data<float>();
        for (
            int output = 0;
            output < fixtures[projection]->output_size;
            ++output
        ) {
            try {
                require_close(
                    values[output],
                    reference(
                        source,
                        0,
                        *fixtures[projection],
                        output),
                    tolerance);
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    std::string(context)
                    + " projection="
                    + std::to_string(projection)
                    + " output="
                    + std::to_string(output)
                    + ": "
                    + error.what());
            }
        }
    }
}

void require_rows_match(
    const mfq::metal::MlxGroupedLinear& grouped,
    const mlx::core::array& input,
    const std::vector<float>& source,
    int rows,
    const std::vector<const Fixture*>& fixtures,
    float tolerance = 8e-4f) {
    auto outputs = grouped(input);
    require(
        outputs.size() == fixtures.size(),
        "heterogeneous VQ grouped output count mismatch");
    for (std::size_t projection = 0;
         projection < outputs.size();
         ++projection) {
        auto values_array = mlx::core::contiguous(
            mlx::core::astype(
                outputs[projection],
                mlx::core::float32));
        values_array.eval();
        require(
            outputs[projection].shape()
                == mlx::core::Shape{
                    rows,
                    fixtures[projection]->output_size,
                },
            "heterogeneous VQ grouped output shape mismatch");
        const auto* values =
            values_array.data<float>();
        for (int row = 0; row < rows; ++row) {
            for (int output = 0;
                 output
                    < fixtures[projection]->output_size;
                 ++output) {
                try {
                    require_close(
                        values[
                            row
                                * fixtures[projection]->output_size
                            + output
                        ],
                        reference(
                            source,
                            static_cast<std::size_t>(row)
                                * kInputSize,
                            *fixtures[projection],
                            output),
                        tolerance);
                } catch (const std::exception& error) {
                    throw std::runtime_error(
                        "row-matrix rows="
                        + std::to_string(rows)
                        + " row="
                        + std::to_string(row)
                        + " projection="
                        + std::to_string(projection)
                        + " output="
                        + std::to_string(output)
                        + ": "
                        + error.what());
                }
            }
        }
    }
}

} // namespace

int main() {
    try {
        using namespace mlx::core;

        std::vector<Fixture> fixtures;
        fixtures.reserve(9);
        std::vector<mfq::metal::MlxNintWeight> nint_weights;
        nint_weights.reserve(8);
        for (int bits = 1; bits <= 8; ++bits) {
            fixtures.push_back(
                make_nint_fixture(bits, 2 + bits % 3));
            nint_weights.push_back(
                mfq::metal::MlxNintWeight::from_blob(
                    fixtures.back().blob));
        }
        fixtures.push_back(make_q8_fixture(5));
        const auto q8_weight =
            mfq::metal::MlxNint8ZeroWeight::from_blob(
                fixtures.back().blob);
        auto vq_fixtures = make_vq_fixtures();
        std::vector<mfq::metal::MlxVqWeight> vq_weights;
        vq_weights.reserve(vq_fixtures.size());
        for (const auto& fixture : vq_fixtures) {
            vq_weights.push_back(
                mfq::metal::MlxVqWeight::from_blob(
                    fixture.dtype,
                    fixture.fixture.blob));
        }
        const auto tpq_int4_fixture =
            make_tpq_int4_fixture(6);
        const auto tpq_int4_weight =
            mfq::metal::MlxTpqInt4Weight::from_blob(
                tpq_int4_fixture.blob);
        auto tpq_pq_fixtures =
            make_tpq_pq_fixtures();
        std::vector<mfq::metal::MlxTpqPqWeight>
            tpq_pq_weights;
        tpq_pq_weights.reserve(
            tpq_pq_fixtures.size());
        for (const auto& fixture :
             tpq_pq_fixtures) {
            tpq_pq_weights.push_back(
                mfq::metal::
                    MlxTpqPqWeight::from_blob(
                        fixture.dtype,
                        fixture.fixture.blob));
        }

        std::vector<mfq::metal::MlxGroupedLinearWeightRef>
            references;
        references.reserve(9);
        for (const auto& weight : nint_weights) {
            references.emplace_back(&weight);
        }
        references.emplace_back(&q8_weight);
        const mfq::metal::MlxGroupedLinear grouped(
            std::move(references));

        require(
            grouped.projection_count() == fixtures.size(),
            "grouped linear projection count mismatch");
        require(
            grouped.input_size() == kInputSize,
            "grouped linear input width mismatch");
        require(
            grouped.packed_nbytes() > 0,
            "grouped linear did not retain packed streams");
        require(
            !grouped.uses_zero_copy_storage(),
            "generalized group unexpectedly bypassed pooled fallback");
        require(
            grouped.copied_packed_nbytes() > 0,
            "pooled fallback did not report copied packed bytes");
        for (std::size_t index = 0; index < fixtures.size(); ++index) {
            require(
                grouped.output_sizes()[index]
                    == fixtures[index].output_size,
                "grouped linear output width mismatch");
        }

        std::vector<float> source(kInputSize);
        for (int column = 0; column < kInputSize; ++column) {
            source[column] =
                static_cast<float>((column * 7) % 17 - 8)
                / 128.0f;
        }
        const array input(
            source.begin(),
            Shape{1, kInputSize});
        require(
            grouped.supports(input),
            "valid float32 grouped input was rejected");

        // Materialize the original packed arrays before taking the allocator
        // baseline. On a cold MLX process, first-touch allocation accounting
        // can otherwise settle between two adjacent get_active_memory calls
        // even though the grouped object only retained array references.
        for (const auto& weight : nint_weights) {
            auto q = weight.packed_values();
            auto sub_scale = weight.sub_scales();
            auto sub_min = weight.sub_mins();
            auto neuron_scale = weight.neuron_scales();
            auto neuron_min = weight.neuron_mins();
            q.eval();
            sub_scale.eval();
            sub_min.eval();
            neuron_scale.eval();
            neuron_min.eval();
        }
        auto q8_values = q8_weight.quantized_values();
        auto q8_scales = q8_weight.scales();
        q8_values.eval();
        q8_scales.eval();
        for (const auto& weight : vq_weights) {
            auto indices = weight.packed_indices();
            auto states = weight.packed_states();
            auto auxiliary = weight.packed_auxiliary();
            auto anchors = weight.anchors();
            auto codebooks = weight.codebooks();
            auto scales = weight.scale_lut();
            auto state_banks = weight.state_to_codebank();
            auto bank_ids = weight.bank_ids();
            auto parameters = weight.parameters();
            indices.eval();
            states.eval();
            auxiliary.eval();
            anchors.eval();
            codebooks.eval();
            scales.eval();
            state_banks.eval();
            bank_ids.eval();
            parameters.eval();
        }
        {
            auto values =
                tpq_int4_weight.packed_values();
            auto scales =
                tpq_int4_weight.scales();
            values.eval();
            scales.eval();
        }
        for (const auto& weight :
             tpq_pq_weights) {
            auto indices =
                weight.packed_indices();
            auto codebook =
                weight.codebook();
            indices.eval();
            codebook.eval();
        }

        // Every NINT width uses the production two-projection, direct-buffer
        // path together with NINT8-0. Kernel source is cached by family/layout
        // while the bit/group constants remain Metal template arguments.
        for (std::size_t bit_index = 0;
             bit_index < nint_weights.size();
             ++bit_index) {
            const auto active_before_group =
                mlx::core::get_active_memory();
            const mfq::metal::MlxGroupedLinear direct_pair({
                &nint_weights[bit_index],
                &q8_weight,
            });
            const auto active_after_group =
                mlx::core::get_active_memory();
            require(
                active_after_group <= active_before_group,
                "zero-copy group construction increased active Metal memory "
                + std::to_string(active_before_group)
                + " -> "
                + std::to_string(active_after_group));
            require(
                direct_pair.uses_zero_copy_storage(),
                "two-projection group did not use zero-copy storage");
            require(
                direct_pair.copied_packed_nbytes() == 0,
                "two-projection group copied packed streams");
            require(
                direct_pair.packed_nbytes()
                    == nint_weights[bit_index].packed_nbytes()
                        + q8_weight.packed_nbytes(),
                "two-projection logical packed byte count mismatch");
            require_one_row_matches(
                direct_pair,
                input,
                source,
                {
                    &fixtures[bit_index],
                    &fixtures.back(),
                });
        }

        const mfq::metal::MlxGroupedLinear direct_gate_up({
            &nint_weights[2],
            &nint_weights[6],
        });
        require(
            direct_gate_up.uses_zero_copy_storage()
                && direct_gate_up.copied_packed_nbytes() == 0,
            "ordinary NINT gate/up group copied packed streams");
        require_one_row_matches(
            direct_gate_up,
            input,
            source,
            {
                &fixtures[2],
                &fixtures[6],
            });

        const mfq::metal::MlxGroupedLinear direct_qkv({
            &nint_weights[3],
            &nint_weights[4],
            &nint_weights[5],
        });
        require(
            direct_qkv.uses_zero_copy_storage()
                && direct_qkv.copied_packed_nbytes() == 0,
            "ordinary NINT QKV group copied packed streams");
        require(
            direct_qkv.has_single_row_nint_fast_path(),
            "NINT4/5/6 QKV did not enable the single-row fast path");
        const auto half_input =
            mlx::core::astype(input, mlx::core::float16);
        require_one_row_matches(
            direct_qkv,
            half_input,
            source,
            {
                &fixtures[3],
                &fixtures[4],
                &fixtures[5],
            },
            8e-4f,
            "NINT4/5/6 M=1 fast");
        // Float32 deliberately retains the established direct kernel. This
        // also guards the fast-path dtype gate.
        require_one_row_matches(
            direct_qkv,
            input,
            source,
            {
                &fixtures[3],
                &fixtures[4],
                &fixtures[5],
            },
            8e-4f,
            "NINT4/5/6 float32 fallback");

        // MiniCPM uses K=4096 with GS24, so the final packed group has eight
        // padded weights. Keep non-zero values immediately after the logical
        // input view and verify that the partitioned QKV kernel never reads
        // them as activations.
        {
            constexpr int tail_input_width = 65;
            constexpr int backing_width = 73;
            const auto q_blob = make_nint4_gs24_blob(
                tail_input_width, 17, 2);
            const auto k_blob = make_nint4_gs24_blob(
                tail_input_width, 9, 4);
            const auto v_blob = make_nint4_gs24_blob(
                tail_input_width, 9, 6);
            const auto q_weight =
                mfq::metal::MlxNintWeight::from_blob(q_blob);
            const auto k_weight =
                mfq::metal::MlxNintWeight::from_blob(k_blob);
            const auto v_weight =
                mfq::metal::MlxNintWeight::from_blob(v_blob);
            const mfq::metal::MlxGroupedLinear tail_qkv({
                &q_weight,
                &k_weight,
                &v_weight,
            });
            require(
                tail_qkv.has_single_row_nint_fast_path(),
                "GS24 tail QKV did not enable the partitioned path");

            std::vector<float> backing_values(backing_width);
            for (int column = 0; column < tail_input_width; ++column) {
                backing_values[static_cast<std::size_t>(column)] =
                    static_cast<float>((column * 13) % 31 - 15) / 96.0f;
            }
            for (int column = tail_input_width;
                 column < backing_width;
                 ++column) {
                backing_values[static_cast<std::size_t>(column)] =
                    6.0f + static_cast<float>(column - tail_input_width);
            }
            const auto backing = astype(
                array(
                    backing_values.begin(),
                    Shape{1, backing_width}),
                float16);
            const auto tail_input = slice(
                backing,
                Shape{0, 0},
                Shape{1, tail_input_width});
            auto grouped_outputs = tail_qkv(tail_input);
            std::vector<array> references{
                q_weight.matmul(tail_input),
                k_weight.matmul(tail_input),
                v_weight.matmul(tail_input),
            };
            require(
                grouped_outputs.size() == references.size(),
                "GS24 tail QKV output count mismatch");
            for (std::size_t projection = 0;
                 projection < references.size();
                 ++projection) {
                auto difference = max(abs(
                    astype(grouped_outputs[projection], float32) -
                    astype(references[projection], float32)));
                difference.eval();
                if (!std::isfinite(difference.item<float>()) ||
                    difference.item<float>() > 0.0f) {
                    throw std::runtime_error(
                        "GS24 interleaved QKV changed FP16 values: " +
                        std::to_string(difference.item<float>()));
                }
            }
        }

        // Match MiniCPM-o 4.5's production QKV dimensions and require every
        // FP16 output element to equal the three independent NINT4 matmuls.
        // Small fixtures can miss dispatch-grid errors in the long Q-only
        // tail, so keep this as a full-shape regression test.
        {
            constexpr int production_input_width = 4096;
            constexpr int production_q_outputs = 4096;
            constexpr int production_kv_outputs = 512;
            const auto q_blob = make_nint4_gs24_blob(
                production_input_width, production_q_outputs, 1);
            const auto k_blob = make_nint4_gs24_blob(
                production_input_width, production_kv_outputs, 3);
            const auto v_blob = make_nint4_gs24_blob(
                production_input_width, production_kv_outputs, 5);
            const auto q_weight =
                mfq::metal::MlxNintWeight::from_blob(q_blob);
            const auto k_weight =
                mfq::metal::MlxNintWeight::from_blob(k_blob);
            const auto v_weight =
                mfq::metal::MlxNintWeight::from_blob(v_blob);
            const mfq::metal::MlxGroupedLinear production_qkv({
                &q_weight,
                &k_weight,
                &v_weight,
            });
            std::vector<float> values(production_input_width);
            for (int column = 0;
                 column < production_input_width;
                 ++column) {
                values[static_cast<std::size_t>(column)] =
                    static_cast<float>((column * 17) % 127 - 63)
                    / 512.0f;
            }
            const auto production_input = astype(
                array(
                    values.begin(),
                    Shape{1, production_input_width}),
                float16);
            auto grouped_outputs = production_qkv(production_input);
            std::vector<array> references{
                q_weight.matmul(production_input),
                k_weight.matmul(production_input),
                v_weight.matmul(production_input),
            };
            require(
                grouped_outputs.size() == references.size(),
                "production QKV output count mismatch");
            for (std::size_t projection = 0;
                 projection < references.size();
                 ++projection) {
                auto difference = max(abs(
                    astype(grouped_outputs[projection], float32) -
                    astype(references[projection], float32)));
                difference.eval();
                if (!std::isfinite(difference.item<float>()) ||
                    difference.item<float>() > 0.0f) {
                    throw std::runtime_error(
                        "production interleaved QKV changed FP16 values: " +
                        std::to_string(difference.item<float>()));
                }
            }
        }

        // The DeepSeek-V4 shared expert uses an independent dense router and
        // an equal-width NINT gate/up pair.  Decode can therefore keep the
        // pair's limited SwiGLU inside the single-row grouped dispatch.
        const auto swiglu_gate_fixture =
            make_nint_fixture(5, 9);
        const auto swiglu_up_fixture =
            make_nint_fixture(6, 9);
        const auto swiglu_gate_weight =
            mfq::metal::MlxNintWeight::from_blob(
                swiglu_gate_fixture.blob);
        const auto swiglu_up_weight =
            mfq::metal::MlxNintWeight::from_blob(
                swiglu_up_fixture.blob);
        const mfq::metal::MlxGroupedLinear swiglu_pair({
            &swiglu_gate_weight,
            &swiglu_up_weight,
        });
        require(
            swiglu_pair.supports_single_row_swiglu(half_input),
            "equal-width NINT5/6 gate/up rejected fused SwiGLU");
        require(
            !swiglu_pair.supports_single_row_swiglu(input),
            "float32 gate/up unexpectedly accepted fused SwiGLU");
        require(
            !swiglu_pair.supports_single_row_swiglu(
                mlx::core::concatenate(
                    {half_input, half_input},
                    0)),
            "multi-row gate/up unexpectedly accepted fused SwiGLU");

        constexpr float swiglu_limit = 0.75f;
        auto unfused_swiglu = swiglu_pair(half_input);
        auto fused_swiglu = mlx::core::contiguous(
            mlx::core::astype(
                swiglu_pair.single_row_swiglu(
                    half_input,
                    swiglu_limit),
                mlx::core::float32));
        auto unfused_gate = mlx::core::contiguous(
            mlx::core::astype(
                unfused_swiglu[0],
                mlx::core::float32));
        auto unfused_up = mlx::core::contiguous(
            mlx::core::astype(
                unfused_swiglu[1],
                mlx::core::float32));
        mlx::core::eval(
            {fused_swiglu, unfused_gate, unfused_up});
        require(
            fused_swiglu.shape() == mlx::core::Shape{1, 9},
            "fused grouped SwiGLU output shape mismatch");
        const auto* fused_values = fused_swiglu.data<float>();
        const auto* gate_values = unfused_gate.data<float>();
        const auto* up_values = unfused_up.data<float>();
        for (int output = 0; output < 9; ++output) {
            const float gate = std::min(
                gate_values[output],
                swiglu_limit);
            const float up = std::clamp(
                up_values[output],
                -swiglu_limit,
                swiglu_limit);
            const float expected =
                gate / (1.0f + std::exp(-gate)) * up;
            require_close(
                fused_values[output],
                expected,
                1.5e-3f);
        }

        // Cover tiles where only a subset of heterogeneous projections is
        // still active (Q only, Q+V, and Q+K+V).
        const auto uneven_q_fixture =
            make_nint_fixture(4, 17);
        const auto uneven_k_fixture =
            make_nint_fixture(5, 3);
        const auto uneven_v_fixture =
            make_nint_fixture(6, 9);
        const auto uneven_q_weight =
            mfq::metal::MlxNintWeight::from_blob(
                uneven_q_fixture.blob);
        const auto uneven_k_weight =
            mfq::metal::MlxNintWeight::from_blob(
                uneven_k_fixture.blob);
        const auto uneven_v_weight =
            mfq::metal::MlxNintWeight::from_blob(
                uneven_v_fixture.blob);
        const mfq::metal::MlxGroupedLinear uneven_qkv({
            &uneven_q_weight,
            &uneven_k_weight,
            &uneven_v_weight,
        });
        require(
            uneven_qkv.has_single_row_nint_fast_path(),
            "uneven NINT4/5/6 QKV did not enable M=1 fast path");
        require_one_row_matches(
            uneven_qkv,
            half_input,
            source,
            {
                &uneven_q_fixture,
                &uneven_k_fixture,
                &uneven_v_fixture,
            },
            8e-4f,
            "uneven NINT4/5/6 M=1 fast");

        // Exercise the QKV-sized route, including the special NINT5 execution
        // layout and a heterogeneous NINT8-0 middle projection.
        const mfq::metal::MlxGroupedLinear direct_triple({
            &nint_weights[4],
            &q8_weight,
            &nint_weights[5],
        });
        require(
            direct_triple.uses_zero_copy_storage(),
            "three-projection group did not use zero-copy storage");
        require(
            !direct_triple.has_single_row_nint_fast_path(),
            "mixed NINT/Q8 group incorrectly enabled NINT fast path");
        require(
            direct_triple.copied_packed_nbytes() == 0,
            "three-projection group copied packed streams");
        require_one_row_matches(
            direct_triple,
            input,
            source,
            {
                &fixtures[4],
                &fixtures.back(),
                &fixtures[5],
            });

        // Every ordinary public NVQ/NPQ dtype participates directly in a
        // heterogeneous gate/up-style pair with NINT. No packed model pool
        // is copied; the grouped object retains the original MLX arrays.
        for (std::size_t index = 0;
             index < vq_weights.size();
             ++index) {
            const auto active_before_group =
                mlx::core::get_active_memory();
            const mfq::metal::MlxGroupedLinear vq_pair({
                &vq_weights[index],
                &nint_weights[index % nint_weights.size()],
            });
            const auto active_after_group =
                mlx::core::get_active_memory();
            require(
                active_after_group <= active_before_group,
                vq_fixtures[index].dtype
                    + " grouped construction increased active Metal memory "
                    + std::to_string(active_before_group)
                    + " -> "
                    + std::to_string(active_after_group));
            require(
                vq_pair.uses_zero_copy_storage(),
                vq_fixtures[index].dtype
                    + " pair did not use direct storage");
            require(
                vq_pair.copied_packed_nbytes() == 0,
                vq_fixtures[index].dtype
                    + " pair copied packed streams");
            require(
                vq_pair.packed_nbytes()
                    == vq_weights[index].packed_nbytes()
                        + nint_weights[
                            index % nint_weights.size()
                        ].packed_nbytes(),
                vq_fixtures[index].dtype
                    + " logical packed byte count mismatch");
            require_one_row_matches(
                vq_pair,
                input,
                source,
                {
                    &vq_fixtures[index].fixture,
                    &fixtures[
                        index % nint_weights.size()
                    ],
                },
                8e-4f,
                vq_fixtures[index].dtype + " pair");

            const mfq::metal::MlxGroupedLinear vq_qkv({
                &vq_weights[index],
                &q8_weight,
                &nint_weights[
                    (index + 3) % nint_weights.size()
                ],
            });
            require(
                vq_qkv.uses_zero_copy_storage()
                    && vq_qkv.copied_packed_nbytes() == 0,
                vq_fixtures[index].dtype
                    + " QKV group copied packed streams");
            require_one_row_matches(
                vq_qkv,
                input,
                source,
                {
                    &vq_fixtures[index].fixture,
                    &fixtures.back(),
                    &fixtures[
                        (index + 3) % nint_weights.size()
                    ],
                },
                8e-4f,
                vq_fixtures[index].dtype + " QKV");
        }

        // Exercise the maximum direct-buffer resource layout. Three VQ
        // projections bind 27 shared weight arrays plus x/y, still within
        // Metal's 31-buffer argument limit.
        constexpr std::size_t all_vq_indices[] = {
            1,
            9,
            12,
        };
        const mfq::metal::MlxGroupedLinear all_vq_pair({
            &vq_weights[all_vq_indices[0]],
            &vq_weights[all_vq_indices[2]],
        });
        require(
            all_vq_pair.uses_zero_copy_storage()
                && all_vq_pair.copied_packed_nbytes() == 0,
            "two-VQ direct group copied packed streams");
        require_one_row_matches(
            all_vq_pair,
            input,
            source,
            {
                &vq_fixtures[
                    all_vq_indices[0]
                ].fixture,
                &vq_fixtures[
                    all_vq_indices[2]
                ].fixture,
            },
            8e-4f,
            "two-VQ pair");

        const auto all_vq_active_before =
            mlx::core::get_active_memory();
        const mfq::metal::MlxGroupedLinear all_vq_qkv({
            &vq_weights[all_vq_indices[0]],
            &vq_weights[all_vq_indices[1]],
            &vq_weights[all_vq_indices[2]],
        });
        const auto all_vq_active_after =
            mlx::core::get_active_memory();
        require(
            all_vq_active_after <= all_vq_active_before,
            "three-VQ group increased active Metal memory");
        require(
            all_vq_qkv.uses_zero_copy_storage()
                && all_vq_qkv.copied_packed_nbytes() == 0,
            "three-VQ direct group copied packed streams");
        require_one_row_matches(
            all_vq_qkv,
            input,
            source,
            {
                &vq_fixtures[
                    all_vq_indices[0]
                ].fixture,
                &vq_fixtures[
                    all_vq_indices[1]
                ].fixture,
                &vq_fixtures[
                    all_vq_indices[2]
                ].fixture,
            },
            8e-4f,
            "three-VQ QKV");

        bool four_vq_rejected = false;
        try {
            (void)mfq::metal::MlxGroupedLinear({
                &vq_weights[0],
                &vq_weights[1],
                &vq_weights[2],
                &vq_weights[3],
            });
        } catch (
            const mfq::metal::MlxGroupedLinearUnsupported&
        ) {
            four_vq_rejected = true;
        }
        require(
            four_vq_rejected,
            "four-VQ group incorrectly entered the copied fallback");

        // Every legal TPQ tier/storage layout shares its resident index
        // stream and codebook with an ordinary NINT projection. A second
        // family matrix exercises VQ + I4G64 + product-VQ in one dispatch.
        for (std::size_t index = 0;
             index < tpq_pq_weights.size();
             ++index) {
            const auto active_before =
                mlx::core::get_active_memory();
            const mfq::metal::MlxGroupedLinear
                tpq_pair({
                    &tpq_pq_weights[index],
                    &nint_weights[
                        index % nint_weights.size()
                    ],
                });
            const auto active_after =
                mlx::core::get_active_memory();
            require(
                active_after <= active_before,
                tpq_pq_fixtures[index].dtype
                    + " grouped construction increased "
                      "active Metal memory");
            require(
                tpq_pair.uses_zero_copy_storage()
                    && tpq_pair.copied_packed_nbytes()
                        == 0,
                tpq_pq_fixtures[index].dtype
                    + " pair copied packed streams");
            require(
                tpq_pair.packed_nbytes()
                    == tpq_pq_weights[index]
                            .packed_nbytes()
                        + nint_weights[
                            index
                                % nint_weights.size()
                        ].packed_nbytes(),
                tpq_pq_fixtures[index].dtype
                    + " pair logical byte mismatch");
            require_one_row_matches(
                tpq_pair,
                input,
                source,
                {
                    &tpq_pq_fixtures[
                        index
                    ].fixture,
                    &fixtures[
                        index
                            % nint_weights.size()
                    ],
                },
                2e-3f,
                tpq_pq_fixtures[index].dtype
                    + " p"
                    + std::to_string(
                        tpq_pq_weights[index]
                            .index_bits()));

            const mfq::metal::MlxGroupedLinear
                tpq_heterogeneous({
                    &vq_weights[
                        index % vq_weights.size()
                    ],
                    &tpq_int4_weight,
                    &tpq_pq_weights[index],
                });
            require(
                tpq_heterogeneous
                        .uses_zero_copy_storage()
                    && tpq_heterogeneous
                            .copied_packed_nbytes()
                        == 0,
                "VQ/I4/PQ group copied packed streams");
            require_one_row_matches(
                tpq_heterogeneous,
                input,
                source,
                {
                    &vq_fixtures[
                        index % vq_weights.size()
                    ].fixture,
                    &tpq_int4_fixture,
                    &tpq_pq_fixtures[
                        index
                    ].fixture,
                },
                2e-3f,
                "VQ/I4/PQ heterogeneous group");
        }

        const mfq::metal::MlxGroupedLinear
            q8_tpq_qkv({
                &q8_weight,
                &tpq_int4_weight,
                &tpq_pq_weights[5],
            });
        require_one_row_matches(
            q8_tpq_qkv,
            input,
            source,
            {
                &fixtures.back(),
                &tpq_int4_fixture,
                &tpq_pq_fixtures[5].fixture,
            },
            2e-3f,
            "Q8/I4/PQ heterogeneous group");

        const mfq::metal::MlxGroupedLinear
            tpq_row_matrix({
                &tpq_int4_weight,
                &tpq_pq_weights[4],
                &nint_weights[6],
            });
        for (int rows = 1;
             rows <= 16;
             ++rows) {
            std::vector<float> tpq_row_source(
                static_cast<std::size_t>(rows)
                    * kInputSize);
            for (std::size_t index = 0;
                 index < tpq_row_source.size();
                 ++index) {
                tpq_row_source[index] =
                    static_cast<float>(
                        static_cast<int>(
                            (
                                index * 17
                                + rows
                            ) % 31)
                        - 15)
                    / 256.0f;
            }
            const array tpq_row_input(
                tpq_row_source.begin(),
                Shape{rows, kInputSize});
            require_rows_match(
                tpq_row_matrix,
                tpq_row_input,
                tpq_row_source,
                rows,
                {
                    &tpq_int4_fixture,
                    &tpq_pq_fixtures[4].fixture,
                    &fixtures[6],
                },
                2e-3f);
        }

        std::unique_ptr<
            mfq::metal::MlxGroupedLinear>
            retained_tpq_group;
        const auto retained_tpq_fixture =
            make_tpq_pq_fixture(
                "TPQ-X",
                1,
                8,
                256,
                12,
                4);
        {
            auto temporary =
                mfq::metal::
                    MlxTpqPqWeight::from_blob(
                        retained_tpq_fixture.dtype,
                        retained_tpq_fixture
                            .fixture.blob);
            retained_tpq_group =
                std::make_unique<
                    mfq::metal::MlxGroupedLinear>(
                    std::vector<
                        mfq::metal::
                            MlxGroupedLinearWeightRef>{
                        &temporary,
                        &tpq_int4_weight,
                    });
        }
        require(
            retained_tpq_group
                    ->copied_packed_nbytes()
                == 0,
            "retained TPQ group copied packed storage");
        require_one_row_matches(
            *retained_tpq_group,
            input,
            source,
            {
                &retained_tpq_fixture.fixture,
                &tpq_int4_fixture,
            },
            2e-3f,
            "retained TPQ ownership");

        bool four_tpq_rejected = false;
        try {
            (void)mfq::metal::MlxGroupedLinear({
                &tpq_int4_weight,
                &tpq_pq_weights[0],
                &tpq_pq_weights[1],
                &tpq_pq_weights[2],
            });
        } catch (
            const mfq::metal::
                MlxGroupedLinearUnsupported&
        ) {
            four_tpq_rejected = true;
        }
        require(
            four_tpq_rejected,
            "four-projection TPQ group incorrectly "
            "entered the copied fallback");

        // Exercise every supported small-M row count on one three-family
        // direct dispatch. This covers the production decode-oriented range,
        // including non-divisors of the four-output SIMD tile.
        const std::size_t representative_vq =
            vq_weights.size() - 1;
        const mfq::metal::MlxGroupedLinear vq_row_matrix({
            &vq_weights[representative_vq],
            &nint_weights[4],
            &q8_weight,
        });
        for (int rows = 1; rows <= 16; ++rows) {
            std::vector<float> row_source(
                static_cast<std::size_t>(rows)
                    * kInputSize);
            for (std::size_t index = 0;
                 index < row_source.size();
                 ++index) {
                row_source[index] =
                    static_cast<float>(
                        static_cast<int>(
                            (index * 13 + rows) % 29)
                        - 14)
                    / 256.0f;
            }
            const array row_input(
                row_source.begin(),
                Shape{rows, kInputSize});
            require(
                vq_row_matrix.supports(row_input),
                "valid 1-16 row VQ group was rejected");
            auto standalone_vq = rows == 1
                ? vq_weights[representative_vq].gemv(
                      reshape(
                          row_input,
                          Shape{kInputSize}))
                : vq_weights[representative_vq].mmq(
                      row_input);
            standalone_vq.eval();
            const auto* standalone_values =
                standalone_vq.data<float>();
            for (int row = 0; row < rows; ++row) {
                for (
                    int output = 0;
                    output
                        < vq_fixtures[
                            representative_vq
                        ].fixture.output_size;
                    ++output
                ) {
                    try {
                        require_close(
                            standalone_values[
                                row
                                    * vq_fixtures[
                                        representative_vq
                                    ].fixture.output_size
                                + output
                            ],
                            reference(
                                row_source,
                                static_cast<std::size_t>(row)
                                    * kInputSize,
                                vq_fixtures[
                                    representative_vq
                                ].fixture,
                                output),
                            8e-4f);
                    } catch (
                        const std::exception& error
                    ) {
                        throw std::runtime_error(
                            "standalone VQ MMQ rows="
                            + std::to_string(rows)
                            + " row="
                            + std::to_string(row)
                            + " output="
                            + std::to_string(output)
                            + ": "
                            + error.what());
                    }
                }
            }
            require_rows_match(
                vq_row_matrix,
                row_input,
                row_source,
                rows,
                {
                    &vq_fixtures[
                        representative_vq
                    ].fixture,
                    &fixtures[4],
                    &fixtures.back(),
                });
        }

        // Array handles, rather than raw MlxVqWeight pointers, own the
        // storage after construction.
        const auto retained_fixture =
            make_npq_fixture(true, 4);
        std::unique_ptr<mfq::metal::MlxGroupedLinear>
            retained_vq_group;
        {
            auto temporary =
                mfq::metal::MlxVqWeight::from_blob(
                    retained_fixture.dtype,
                    retained_fixture.fixture.blob);
            retained_vq_group = std::make_unique<
                mfq::metal::MlxGroupedLinear>(
                std::vector<
                    mfq::metal::MlxGroupedLinearWeightRef>{
                    &temporary,
                    &q8_weight,
                });
        }
        require(
            retained_vq_group->copied_packed_nbytes() == 0,
            "retained VQ group copied packed storage");
        require_one_row_matches(
            *retained_vq_group,
            input,
            source,
            {
                &retained_fixture.fixture,
                &fixtures.back(),
            });

        const auto nepq_blob = make_nepq_blob();
        const auto nepq =
            mfq::metal::MlxVqWeight::from_blob(
                "NEPQ0-S",
                nepq_blob);
        bool nepq_rejected = false;
        try {
            (void)mfq::metal::MlxGroupedLinear({
                &nepq,
                &nint_weights[0],
            });
        } catch (
            const mfq::metal::MlxGroupedLinearUnsupported&
        ) {
            nepq_rejected = true;
        }
        require(
            nepq_rejected,
            "expert-shaped NEPQ was accepted as ordinary linear");

        auto outputs = grouped(input);
        require(
            outputs.size() == fixtures.size(),
            "grouped linear output count mismatch");
        for (std::size_t projection = 0;
             projection < outputs.size();
             ++projection) {
            outputs[projection].eval();
            require(
                outputs[projection].shape()
                    == Shape{
                        1,
                        fixtures[projection].output_size,
                    },
                "grouped linear rank-two output shape mismatch");
            const auto* values =
                outputs[projection].data<float>();
            for (
                int output = 0;
                output < fixtures[projection].output_size;
                ++output
            ) {
                require_close(
                    values[output],
                    reference(
                        source,
                        0,
                        fixtures[projection],
                        output),
                    8e-4f);
            }
        }

        std::vector<float> prefix_source(4 * kInputSize);
        for (std::size_t index = 0;
             index < prefix_source.size();
             ++index) {
            prefix_source[index] =
                static_cast<float>(
                    static_cast<int>((index * 5) % 23) - 11)
                / 128.0f;
        }
        const array prefix_f32(
            prefix_source.begin(),
            Shape{2, 2, kInputSize});
        const auto prefix_input = astype(prefix_f32, float16);
        require(
            grouped.supports(prefix_input),
            "valid float16 prefix input was rejected");

        require(
            direct_qkv.supports(prefix_input),
            "zero-copy QKV rejected valid float16 prefix input");
        auto direct_prefix_outputs = direct_qkv(prefix_input);
        constexpr std::size_t direct_fixture_indices[] = {
            3,
            4,
            5,
        };
        for (std::size_t projection = 0;
             projection < direct_prefix_outputs.size();
             ++projection) {
            const auto fixture_index =
                direct_fixture_indices[projection];
            require(
                direct_prefix_outputs[projection].shape()
                    == Shape{
                        2,
                        2,
                        fixtures[fixture_index].output_size,
                    },
                "zero-copy QKV prefix shape was not preserved");
            auto values = astype(
                direct_prefix_outputs[projection],
                float32);
            values.eval();
            const auto* data = values.data<float>();
            for (int row = 0; row < 4; ++row) {
                for (
                    int output = 0;
                    output
                        < fixtures[fixture_index].output_size;
                    ++output
                ) {
                    require_close(
                        data[
                            row
                                * fixtures[fixture_index].output_size
                            + output],
                        reference(
                            prefix_source,
                            static_cast<std::size_t>(row)
                                * kInputSize,
                            fixtures[fixture_index],
                            output),
                        0.035f);
                }
            }
        }

        auto prefix_outputs = grouped(prefix_input);
        for (std::size_t projection = 0;
             projection < prefix_outputs.size();
             ++projection) {
            require(
                prefix_outputs[projection].shape()
                    == Shape{
                        2,
                        2,
                        fixtures[projection].output_size,
                    },
                "grouped linear prefix shape was not preserved");
            auto values = astype(
                prefix_outputs[projection],
                float32);
            values.eval();
            const auto* data = values.data<float>();
            for (int row = 0; row < 4; ++row) {
                for (
                    int output = 0;
                    output < fixtures[projection].output_size;
                    ++output
                ) {
                    require_close(
                        data[
                            row
                                * fixtures[projection].output_size
                            + output],
                        reference(
                            prefix_source,
                            static_cast<std::size_t>(row)
                                * kInputSize,
                            fixtures[projection],
                            output),
                        0.035f);
                }
            }
        }

        const auto maximum_input = zeros(
            Shape{4, 4, kInputSize},
            float16);
        require(
            grouped.supports(maximum_input),
            "16-row grouped input was rejected");
        const auto too_many_rows = zeros(
            Shape{17, kInputSize},
            float16);
        require(
            !grouped.supports(too_many_rows),
            "17-row grouped input was reported supported");
        require_unsupported(grouped, too_many_rows);

        const auto integer_input = zeros(
            Shape{1, kInputSize},
            int32);
        require(
            !grouped.supports(integer_input),
            "integer grouped input was reported supported");
        require_unsupported(grouped, integer_input);

        std::cout
            << "MFQ C++ heterogeneous NINT1-NINT8/NINT8-0/NVQ/NPQ "
               "single-dispatch grouped Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
