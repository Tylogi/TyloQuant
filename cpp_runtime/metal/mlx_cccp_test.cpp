#include "mlx_cccp.h"
#include "mlx_tensor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace {

constexpr int kInputSize = 128;

template <typename T>
void append(
    std::vector<std::uint8_t>& target,
    T value) {
    const auto* bytes =
        reinterpret_cast<const std::uint8_t*>(
            &value);
    target.insert(
        target.end(),
        bytes,
        bytes + sizeof(T));
}

void append_magic(
    std::vector<std::uint8_t>& target,
    std::string_view value) {
    if (value.size() != 4) {
        throw std::runtime_error(
            "test magic must have four bytes");
    }
    target.insert(
        target.end(),
        value.begin(),
        value.end());
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
    float tolerance,
    const std::string& context) {
    if (
        !std::isfinite(actual) ||
        std::fabs(actual - expected) > tolerance
    ) {
        throw std::runtime_error(
            context + ": actual="
            + std::to_string(actual)
            + " expected="
            + std::to_string(expected));
    }
}

template <typename Callable>
void require_throws(
    Callable&& callable,
    const std::string& message) {
    bool rejected = false;
    try {
        callable();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, message);
}

struct DenseFixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    int output_size = 0;
};

DenseFixture make_int4_fixture(
    int output_size = 12) {
    constexpr std::array<std::uint16_t, 4>
        scale_bits{
            0x2c00,
            0x3000,
            0x3400,
            0x3800,
        };
    constexpr std::array<float, 4>
        scale_values{
            0.0625f,
            0.125f,
            0.25f,
            0.5f,
        };
    constexpr int groups =
        kInputSize / 64;
    std::vector<std::int8_t> quantized(
        static_cast<std::size_t>(output_size)
            * kInputSize);
    for (std::size_t index = 0;
         index < quantized.size();
         ++index) {
        quantized[index] =
            static_cast<std::int8_t>(
                static_cast<int>(
                    (index * 11 + 3) % 16)
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
            const auto base =
                static_cast<std::size_t>(output)
                    * kInputSize
                + pair * 2;
            packed[
                static_cast<std::size_t>(output)
                    * (kInputSize / 2)
                + pair
            ] =
                static_cast<std::uint8_t>(
                    quantized[base] + 8)
                | static_cast<std::uint8_t>(
                    (quantized[base + 1] + 8)
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
    append<std::uint32_t>(blob, groups);
    blob.insert(
        blob.end(),
        packed.begin(),
        packed.end());
    for (int output = 0;
         output < output_size;
         ++output) {
        for (int group = 0;
             group < groups;
             ++group) {
            append<std::uint16_t>(
                blob,
                scale_bits[
                    static_cast<std::size_t>(
                        (output + group) & 3)
                ]);
        }
    }

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
                        static_cast<std::size_t>(
                            output)
                            * kInputSize
                        + column
                    ])
                * scale_values[
                    static_cast<std::size_t>(
                        (
                            output
                            + column / 64
                        ) & 3)
                ];
        }
    }
    return {
        std::move(blob),
        std::move(dense),
        output_size,
    };
}

struct PqProfile {
    std::string dtype;
    int tier = 0;
    int vector_size = 0;
    int entries = 0;
    int bits = 0;
};

std::vector<std::uint8_t> pack_indices(
    const std::vector<std::uint16_t>& values,
    int bits) {
    std::vector<std::uint8_t> packed(
        (
            values.size()
                * static_cast<std::size_t>(bits)
            + 7
        ) / 8,
        0);
    for (std::size_t index = 0;
         index < values.size();
         ++index) {
        for (int bit = 0; bit < bits; ++bit) {
            if (
                (
                    values[index]
                    >> bit
                ) & 1u
            ) {
                const auto target =
                    index
                        * static_cast<std::size_t>(
                            bits)
                    + static_cast<std::size_t>(bit);
                packed[target / 8] |=
                    static_cast<std::uint8_t>(
                        1u << (target & 7));
            }
        }
    }
    return packed;
}

DenseFixture make_pq_fixture(
    const PqProfile& profile,
    int output_size = 3) {
    const int blocks =
        kInputSize / profile.vector_size;
    std::vector<float> codebook(
        static_cast<std::size_t>(
            profile.entries)
            * profile.vector_size);
    for (int entry = 0;
         entry < profile.entries;
         ++entry) {
        for (int component = 0;
             component < profile.vector_size;
             ++component) {
            codebook[
                static_cast<std::size_t>(entry)
                    * profile.vector_size
                + component
            ] =
                static_cast<float>(
                    (
                        entry * 3
                        + component * 5
                    ) % 33
                    - 16)
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
                    index * 97
                    + profile.entries
                    - 1
                ) % profile.entries);
    }
    if (!indices.empty()) {
        indices.front() =
            static_cast<std::uint16_t>(
                profile.entries - 1);
    }
    const auto packed = pack_indices(
        indices,
        profile.bits);

    std::vector<std::uint8_t> blob;
    append_magic(blob, "CPQ1");
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(
        blob,
        static_cast<std::uint8_t>(
            profile.tier));
    append<std::uint8_t>(
        blob,
        static_cast<std::uint8_t>(
            profile.vector_size));
    append<std::uint8_t>(
        blob,
        static_cast<std::uint8_t>(
            profile.bits));
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(
        blob,
        kInputSize);
    append<std::uint32_t>(blob, 2);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(
            profile.entries));
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
                 component < profile.vector_size;
                 ++component) {
                dense[
                    static_cast<std::size_t>(output)
                        * kInputSize
                    + block * profile.vector_size
                    + component
                ] =
                    codebook[
                        static_cast<std::size_t>(code)
                            * profile.vector_size
                        + component
                    ];
            }
        }
    }
    return {
        std::move(blob),
        std::move(dense),
        output_size,
    };
}

std::vector<float> make_input(
    int rows,
    int salt) {
    std::vector<float> result(
        static_cast<std::size_t>(rows)
            * kInputSize);
    for (std::size_t index = 0;
         index < result.size();
         ++index) {
        result[index] =
            static_cast<float>(
                static_cast<int>(
                    (
                        index
                            * static_cast<std::size_t>(
                                salt)
                        + rows
                    ) % 41)
                - 20)
            / 256.0f;
    }
    return result;
}

std::vector<float> reference_matmul(
    const std::vector<float>& input,
    int rows,
    const DenseFixture& fixture) {
    std::vector<float> result(
        static_cast<std::size_t>(rows)
            * fixture.output_size);
    for (int row = 0; row < rows; ++row) {
        for (int output = 0;
             output < fixture.output_size;
             ++output) {
            float value = 0.0f;
            for (int column = 0;
                 column < kInputSize;
                 ++column) {
                value +=
                    input[
                        static_cast<std::size_t>(row)
                            * kInputSize
                        + column
                    ]
                    * fixture.dense[
                        static_cast<std::size_t>(output)
                            * kInputSize
                        + column
                    ];
            }
            result[
                static_cast<std::size_t>(row)
                    * fixture.output_size
                + output
            ] = value;
        }
    }
    return result;
}

void require_array_close(
    const mlx::core::array& value,
    const std::vector<float>& expected,
    const mlx::core::Shape& shape,
    float tolerance,
    const std::string& context) {
    auto contiguous =
        mlx::core::contiguous(
            mlx::core::astype(
                value,
                mlx::core::float32));
    contiguous.eval();
    require(
        contiguous.shape() == shape,
        context + " shape mismatch");
    require(
        contiguous.size() == expected.size(),
        context + " element count mismatch");
    const auto* actual =
        contiguous.data<float>();
    for (std::size_t index = 0;
         index < expected.size();
         ++index) {
        require_close(
            actual[index],
            expected[index],
            tolerance,
            context
                + " index="
                + std::to_string(index));
    }
}

template <typename Weight>
void exercise_projection(
    const Weight& weight,
    const DenseFixture& fixture,
    const std::string& context) {
    const auto one_source = make_input(1, 7);
    const mlx::core::array one_input(
        one_source.begin(),
        mlx::core::Shape{1, kInputSize});
    require_array_close(
        weight.gemv(one_input),
        reference_matmul(
            one_source,
            1,
            fixture),
        {1, fixture.output_size},
        8e-4f,
        context + " GEMV");

    for (int rows = 2; rows <= 16; ++rows) {
        const auto source =
            make_input(rows, 11 + rows);
        const mlx::core::array input(
            source.begin(),
            mlx::core::Shape{
                rows,
                kInputSize,
            });
        require_array_close(
            weight.mmq(input),
            reference_matmul(
                source,
                rows,
                fixture),
            {rows, fixture.output_size},
            1.2e-3f,
            context
                + " MMQ rows="
                + std::to_string(rows));
    }

    constexpr int gemm_rows = 23;
    const auto gemm_source =
        make_input(gemm_rows, 19);
    const mlx::core::array gemm_input(
        gemm_source.begin(),
        mlx::core::Shape{
            gemm_rows,
            kInputSize,
        });
    require_array_close(
        weight.gemm(gemm_input),
        reference_matmul(
            gemm_source,
            gemm_rows,
            fixture),
        {gemm_rows, fixture.output_size},
        1.5e-3f,
        context + " packed GEMM");

    constexpr int large_rows = 64;
    const auto large_source =
        make_input(large_rows, 23);
    const mlx::core::array large_f32(
        large_source.begin(),
        mlx::core::Shape{
            large_rows,
            kInputSize,
        });
    const auto large_f16 =
        mlx::core::astype(
            large_f32,
            mlx::core::float16);
    require_array_close(
        weight.matmul(large_f16),
        reference_matmul(
            large_source,
            large_rows,
            fixture),
        {large_rows, fixture.output_size},
        3.5e-3f,
        context + " large-M dense GEMM");
}

void test_int4() {
    using namespace mlx::core;
    const auto fixture = make_int4_fixture();
    const auto weight =
        mfq::metal::MlxCccpInt4Weight::from_blob(
            fixture.blob);
    require(
        weight.input_size() == kInputSize &&
            weight.output_size()
                == fixture.output_size &&
            weight.group_size() == 64 &&
            weight.groups() == 2,
        "CCCP-I4G64 metadata mismatch");
    require(
        weight.packed_values().dtype() == uint8 &&
            weight.scales().dtype() == float16 &&
            weight.packed_nbytes() > 0,
        "CCCP-I4G64 resident arrays mismatch");

    require_array_close(
        weight.dequantize(float32),
        fixture.dense,
        {
            fixture.output_size,
            kInputSize,
        },
        1e-6f,
        "CCCP-I4G64 dequantize");
    exercise_projection(
        weight,
        fixture,
        "CCCP-I4G64");

    const std::vector<std::int32_t> ids{
        0,
        fixture.output_size - 1,
        -1,
        fixture.output_size,
    };
    const array token_ids(
        ids.begin(),
        Shape{2, 2});
    std::vector<float> expected_embedding(
        4 * kInputSize,
        0.0f);
    std::copy_n(
        fixture.dense.begin(),
        kInputSize,
        expected_embedding.begin());
    std::copy_n(
        fixture.dense.begin()
            + static_cast<std::ptrdiff_t>(
                (
                    fixture.output_size - 1
                ) * kInputSize),
        kInputSize,
        expected_embedding.begin()
            + kInputSize);
    require_array_close(
        weight.embedding(token_ids, float32),
        expected_embedding,
        {2, 2, kInputSize},
        1e-6f,
        "CCCP-I4G64 embedding");

    constexpr int group_count = 4;
    constexpr int grouped_rows = 3;
    const auto grouped_source =
        make_input(
            grouped_rows * group_count,
            29);
    const array grouped_input(
        grouped_source.begin(),
        Shape{
            grouped_rows,
            group_count,
            kInputSize,
        });
    const int out_per_group =
        fixture.output_size / group_count;
    std::vector<float> grouped_expected(
        grouped_rows
            * group_count
            * out_per_group);
    for (int row = 0;
         row < grouped_rows;
         ++row) {
        for (int group = 0;
             group < group_count;
             ++group) {
            for (int local_output = 0;
                 local_output < out_per_group;
                 ++local_output) {
                const int output =
                    group * out_per_group
                    + local_output;
                float value = 0.0f;
                for (int column = 0;
                     column < kInputSize;
                     ++column) {
                    value +=
                        grouped_source[
                            (
                                static_cast<std::size_t>(
                                    row)
                                    * group_count
                                + group
                            ) * kInputSize
                            + column
                        ]
                        * fixture.dense[
                            static_cast<std::size_t>(
                                output)
                                * kInputSize
                            + column
                        ];
                }
                grouped_expected[
                    (
                        static_cast<std::size_t>(row)
                            * group_count
                        + group
                    ) * out_per_group
                    + local_output
                ] = value;
            }
        }
    }
    require_array_close(
        weight.grouped_row_matmul(
            grouped_input,
            group_count),
        grouped_expected,
        {
            grouped_rows,
            group_count,
            out_per_group,
        },
        8e-4f,
        "CCCP-I4G64 grouped-row");

    auto bad_magic = fixture.blob;
    bad_magic[0] = 'X';
    require_throws(
        [&]() {
            (void)mfq::metal::
                MlxCccpInt4Weight::from_blob(
                    bad_magic);
        },
        "CCCP-I4G64 bad magic was accepted");
    auto bad_padding = fixture.blob;
    bad_padding[5] = 1;
    require_throws(
        [&]() {
            (void)mfq::metal::
                MlxCccpInt4Weight::from_blob(
                    bad_padding);
        },
        "CCCP-I4G64 reserved padding was accepted");
    auto truncated = fixture.blob;
    truncated.pop_back();
    require_throws(
        [&]() {
            (void)mfq::metal::
                MlxCccpInt4Weight::from_blob(
                    truncated);
        },
        "truncated CCCP-I4G64 was accepted");
}

std::vector<PqProfile> pq_profiles() {
    return {
        {"TPQ-X", 1, 8, 256, 8},
        {"CCCP-X", 1, 8, 256, 12},
        {"CCCP-X", 1, 8, 256, 14},
        {"TPQ-W", 2, 8, 4096, 12},
        {"CCCP-W", 2, 8, 4096, 14},
        {"CCCP-W", 2, 8, 4096, 16},
        {"TPQ-V", 3, 4, 256, 8},
        {"CCCP-V", 3, 4, 256, 12},
        {"CCCP-V", 3, 4, 256, 14},
        {"TPQ-VV", 4, 4, 4096, 12},
        {"CCCP-VV", 4, 4, 4096, 14},
        {"CCCP-VV", 4, 4, 4096, 16},
    };
}

void test_pq() {
    using namespace mlx::core;
    for (const auto& profile : pq_profiles()) {
        const auto fixture =
            make_pq_fixture(profile);
        const auto weight =
            mfq::metal::MlxCccpPqWeight::from_blob(
                profile.dtype,
                fixture.blob);
        const auto context =
            profile.dtype
            + " p"
            + std::to_string(profile.bits);
        require(
            weight.format_label()
                    == profile.dtype &&
                weight.input_size()
                    == kInputSize &&
                weight.output_size()
                    == fixture.output_size &&
                weight.vector_size()
                    == profile.vector_size &&
                weight.blocks()
                    == kInputSize
                        / profile.vector_size &&
                weight.entries()
                    == profile.entries &&
                weight.index_bits()
                    == profile.bits,
            context + " metadata mismatch");
        require(
            weight.packed_indices().dtype()
                    == uint8 &&
                weight.codebook().dtype()
                    == float16 &&
                weight.packed_nbytes() > 0,
            context + " resident arrays mismatch");
        require_array_close(
            weight.dequantize(float32),
            fixture.dense,
            {
                fixture.output_size,
                kInputSize,
            },
            1e-6f,
            context + " dequantize");
        exercise_projection(
            weight,
            fixture,
            context);
    }

    const auto fixture =
        make_pq_fixture(
            {"CCCP-X", 1, 8, 256, 8});
    require_throws(
        [&]() {
            (void)mfq::metal::
                MlxCccpPqWeight::from_blob(
                    "CCCP-V",
                    fixture.blob);
        },
        "CCCP-PQ dtype mismatch was accepted");

    auto bad_bits = fixture.blob;
    bad_bits[7] = 16;
    require_throws(
        [&]() {
            (void)mfq::metal::
                MlxCccpPqWeight::from_blob(
                    "CCCP-X",
                    bad_bits);
        },
        "CCCP-X p16 storage was accepted");

    auto bad_index = fixture.blob;
    const std::size_t header =
        24 + 16 + 4;
    const std::size_t codebook =
        256 * 8 * sizeof(float);
    bad_index[header + codebook] = 255;
    // The fixture already starts with index 255, so invalidate its entry
    // count instead and ensure strict tier metadata rejects the blob.
    const std::uint32_t wrong_entries = 255;
    std::memcpy(
        bad_index.data() + 20,
        &wrong_entries,
        sizeof(wrong_entries));
    require_throws(
        [&]() {
            (void)mfq::metal::
                MlxCccpPqWeight::from_blob(
                    "CCCP-X",
                    bad_index);
        },
        "CCCP-X wrong codebook size was accepted");

    auto nan_codebook = fixture.blob;
    const float nan =
        std::numeric_limits<float>::quiet_NaN();
    std::memcpy(
        nan_codebook.data() + header,
        &nan,
        sizeof(nan));
    require_throws(
        [&]() {
            (void)mfq::metal::
                MlxCccpPqWeight::from_blob(
                    "CCCP-X",
                    nan_codebook);
        },
        "CCCP-PQ NaN codebook was accepted");

    auto truncated = fixture.blob;
    truncated.pop_back();
    require_throws(
        [&]() {
            (void)mfq::metal::
                MlxCccpPqWeight::from_blob(
                    "CCCP-X",
                    truncated);
        },
        "truncated CCCP-PQ was accepted");
}

void test_generic_tensor_wrappers() {
    using namespace mlx::core;

    const auto int4_fixture =
        make_int4_fixture();
    const auto linear_source =
        make_input(5, 37);
    const array linear_input(
        linear_source.begin(),
        Shape{5, kInputSize});
    mfq::metal::MlxLinear int4_linear(
        mfq::metal::MlxCccpInt4Weight::
            from_blob(int4_fixture.blob));
    require(
        int4_linear.packed() &&
            int4_linear.input_size()
                == kInputSize &&
            int4_linear.output_size()
                == int4_fixture.output_size &&
            int4_linear.grouped_weight_ref()
                .has_value(),
        "MlxLinear CCCP-I4G64 metadata mismatch");
    require_array_close(
        int4_linear(linear_input),
        reference_matmul(
            linear_source,
            5,
            int4_fixture),
        {5, int4_fixture.output_size},
        1.2e-3f,
        "MlxLinear CCCP-I4G64");

    mfq::metal::MlxEmbedding int4_embedding(
        mfq::metal::MlxCccpInt4Weight::
            from_blob(int4_fixture.blob));
    require(
        int4_embedding.vocabulary_size()
                == int4_fixture.output_size &&
            int4_embedding.hidden_size()
                == kInputSize,
        "MlxEmbedding CCCP-I4G64 metadata mismatch");
    const std::vector<std::int32_t> ids{
        2,
        int4_fixture.output_size - 1,
        0,
        5,
    };
    const array token_ids(
        ids.begin(),
        Shape{2, 2});
    std::vector<float> embedding_expected(
        ids.size() * kInputSize);
    for (std::size_t index = 0;
         index < ids.size();
         ++index) {
        std::copy_n(
            int4_fixture.dense.begin()
                + static_cast<std::ptrdiff_t>(
                    ids[index] * kInputSize),
            kInputSize,
            embedding_expected.begin()
                + static_cast<std::ptrdiff_t>(
                    index * kInputSize));
    }
    require_array_close(
        int4_embedding(token_ids, float32),
        embedding_expected,
        {2, 2, kInputSize},
        1e-6f,
        "MlxEmbedding CCCP-I4G64 lookup");
    require_array_close(
        int4_embedding.project(linear_input),
        reference_matmul(
            linear_source,
            5,
            int4_fixture),
        {5, int4_fixture.output_size},
        1.2e-3f,
        "MlxEmbedding CCCP-I4G64 project");

    const auto pq_fixture =
        make_pq_fixture(
            {"CCCP-VV", 4, 4, 4096, 14},
            7);
    mfq::metal::MlxLinear pq_linear(
        mfq::metal::MlxCccpPqWeight::
            from_blob(
                "CCCP-VV",
                pq_fixture.blob));
    require(
        pq_linear.packed() &&
            pq_linear.input_size()
                == kInputSize &&
            pq_linear.output_size()
                == pq_fixture.output_size &&
            pq_linear.grouped_weight_ref()
                .has_value(),
        "MlxLinear CCCP-VV p14 metadata mismatch");
    require_array_close(
        pq_linear(linear_input),
        reference_matmul(
            linear_source,
            5,
            pq_fixture),
        {5, pq_fixture.output_size},
        1.5e-3f,
        "MlxLinear CCCP-VV p14");

    constexpr int group_count = 4;
    constexpr int batch_rows = 3;
    const auto grouped_source =
        make_input(
            batch_rows * group_count,
            41);
    const array grouped_input(
        grouped_source.begin(),
        Shape{
            batch_rows,
            group_count,
            kInputSize,
        });
    const int output_per_group =
        int4_fixture.output_size / group_count;
    std::vector<float> grouped_expected(
        static_cast<std::size_t>(
            batch_rows
            * group_count
            * output_per_group));
    for (int batch = 0;
         batch < batch_rows;
         ++batch) {
        for (int group = 0;
             group < group_count;
             ++group) {
            for (int local = 0;
                 local < output_per_group;
                 ++local) {
                const int output =
                    group * output_per_group
                    + local;
                float total = 0.0f;
                for (int column = 0;
                     column < kInputSize;
                     ++column) {
                    total +=
                        grouped_source[
                            (
                                static_cast<std::size_t>(
                                    batch)
                                    * group_count
                                + group
                            ) * kInputSize
                            + column
                        ]
                        * int4_fixture.dense[
                            static_cast<std::size_t>(
                                output)
                                * kInputSize
                            + column
                        ];
                }
                grouped_expected[
                    (
                        static_cast<std::size_t>(
                            batch)
                            * group_count
                        + group
                    ) * output_per_group
                    + local
                ] = total;
            }
        }
    }
    require_array_close(
        int4_linear.grouped_row_matmul(
            grouped_input,
            group_count),
        grouped_expected,
        {
            batch_rows,
            group_count,
            output_per_group,
        },
        8e-4f,
        "MlxLinear CCCP-I4G64 grouped-row");
}

} // namespace

int main() {
    try {
        require(
            mfq::metal::is_tpq_dtype(
                "TPQ-I4G64") &&
                mfq::metal::is_tpq_dtype(
                    "TPQ-X") &&
                mfq::metal::is_tpq_dtype(
                    "TPQ-W") &&
                mfq::metal::is_tpq_dtype(
                    "TPQ-V") &&
                mfq::metal::is_tpq_dtype(
                    "TPQ-VV") &&
                mfq::metal::is_cccp_dtype(
                "CCCP-I4G64") &&
                mfq::metal::is_cccp_dtype(
                    "CCCP-X") &&
                mfq::metal::is_cccp_dtype(
                    "CCCP-W") &&
                mfq::metal::is_cccp_dtype(
                    "CCCP-V") &&
                mfq::metal::is_cccp_dtype(
                    "CCCP-VV") &&
                !mfq::metal::is_cccp_dtype(
                    "NVQ2"),
            "TPQ/CCCP dtype recognition mismatch");
        test_int4();
        test_pq();
        test_generic_tensor_wrappers();
        std::cout
            << "MFQ native C++/MLX TPQ-I4G64 and "
               "TPQ-X/W/V/VV p8/p12/p14/p16 "
               "kernel/wrapper tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
