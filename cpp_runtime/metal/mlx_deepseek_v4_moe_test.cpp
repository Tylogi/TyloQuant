#include "mlx_deepseek_v4_moe.h"

#include "../nvq_codebooks.generated.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace {

using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::DeepseekV4Config;
using mfq::metal::MlxDeepseekV4Moe;
using mfq::metal::MlxDeepseekV4MoeResult;
using mfq::metal::MlxLinear;
using mfq::metal::MlxMoeWeight;
using mfq::metal::MlxNintWeight;
using mfq::metal::MlxRoutedLinear;

constexpr int kHidden = 32;
constexpr int kIntermediate = 16;
constexpr int kExperts = 4;
constexpr int kTopK = 2;
constexpr int kVocab = 6;
constexpr int kNintGroup = 16;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    float actual,
    float expected,
    float tolerance,
    const char* label) {
    if (!std::isfinite(actual) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            std::string("DeepSeek-V4 MoE ") + label +
            " mismatch: actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

template <typename T>
void append(std::vector<std::uint8_t>& output, T value) {
    const auto* bytes =
        reinterpret_cast<const std::uint8_t*>(&value);
    output.insert(
        output.end(),
        bytes,
        bytes + sizeof(value));
}

void append_magic(
    std::vector<std::uint8_t>& output,
    std::string_view magic) {
    require(magic.size() == 4, "fixture magic must have four bytes");
    output.insert(output.end(), magic.begin(), magic.end());
}

template <typename T>
std::vector<std::uint8_t> pack_bits(
    const std::vector<T>& values,
    int bits) {
    require(bits > 0 && bits <= 16, "invalid fixture bit width");
    std::vector<std::uint8_t> output(
        (values.size() * static_cast<std::size_t>(bits) + 7) / 8,
        0);
    const auto limit = std::uint32_t{1} << bits;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto value = static_cast<std::uint32_t>(values[index]);
        require(value < limit, "fixture value exceeds bit width");
        for (int bit = 0; bit < bits; ++bit) {
            if (((value >> bit) & 1u) == 0u) {
                continue;
            }
            const auto target =
                index * static_cast<std::size_t>(bits) +
                static_cast<std::size_t>(bit);
            output[target / 8] |= static_cast<std::uint8_t>(
                1u << (target & 7u));
        }
    }
    return output;
}

void append_bytes(
    std::vector<std::uint8_t>& output,
    const std::vector<std::uint8_t>& values) {
    output.insert(output.end(), values.begin(), values.end());
}

struct RotationFixture {
    int block = 0;
    std::vector<std::int8_t> signs;
};

struct TensorFixture {
    std::string dtype;
    std::vector<std::uint8_t> blob;
    std::vector<std::uint8_t> runtime;
    std::vector<float> dense;
    RotationFixture rotation;
    int output = 0;
    int input = 0;
};

TensorFixture make_nint(
    int bits,
    int output,
    int input,
    int salt) {
    require(
        input % kNintGroup == 0,
        "NINT fixture width must be a multiple of 16");
    const int groups = input / kNintGroup;
    const auto metadata_count =
        static_cast<std::size_t>(output) * groups;
    std::vector<std::uint8_t> sub_scale(metadata_count);
    std::vector<std::uint8_t> sub_min(metadata_count);
    for (std::size_t index = 0; index < metadata_count; ++index) {
        sub_scale[index] = static_cast<std::uint8_t>(
            1 + (index + static_cast<std::size_t>(salt)) % 3);
        sub_min[index] = static_cast<std::uint8_t>(
            (index + static_cast<std::size_t>(salt)) & 1u);
    }
    const auto maximum = (std::uint32_t{1} << bits) - 1u;
    std::vector<std::uint8_t> quantized(
        metadata_count * kNintGroup);
    for (std::size_t index = 0; index < quantized.size(); ++index) {
        quantized[index] = static_cast<std::uint8_t>(
            (index * static_cast<std::size_t>(bits + 3) +
             static_cast<std::size_t>(salt * 5 + 1)) %
            (maximum + 1u));
    }

    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(blob, static_cast<std::uint8_t>(bits));
    append<std::uint8_t>(blob, 2);
    append<std::int32_t>(blob, kNintGroup);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output);
    append<std::int64_t>(blob, input);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, groups);
    for (int row = 0; row < output; ++row) {
        append<std::uint16_t>(blob, 0x2400);
    }
    for (int row = 0; row < output; ++row) {
        append<std::uint16_t>(blob, 0x2000);
    }
    append_bytes(blob, pack_bits(sub_scale, 2));
    append_bytes(blob, pack_bits(sub_min, 2));
    append_bytes(blob, pack_bits(quantized, bits));

    std::vector<float> dense(
        static_cast<std::size_t>(output) * input);
    for (int row = 0; row < output; ++row) {
        for (int column = 0; column < input; ++column) {
            const int group = column / kNintGroup;
            const auto metadata =
                static_cast<std::size_t>(row) * groups + group;
            const auto value =
                metadata * kNintGroup + column % kNintGroup;
            dense[static_cast<std::size_t>(row) * input + column] =
                static_cast<float>(sub_scale[metadata]) / 64.0f *
                    static_cast<float>(quantized[value]) -
                static_cast<float>(sub_min[metadata]) / 128.0f;
        }
    }
    return {
        "NINT" + std::to_string(bits),
        std::move(blob),
        {},
        std::move(dense),
        {},
        output,
        input,
    };
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
    append<std::uint32_t>(blob, output);
}

void append_one_anchors(
    std::vector<std::uint8_t>& blob,
    int count) {
    for (int index = 0; index < count; ++index) {
        append<std::uint16_t>(blob, 0x3c00);
    }
}

std::vector<std::uint16_t> filled(
    std::size_t count,
    std::uint16_t value) {
    return std::vector<std::uint16_t>(count, value);
}

TensorFixture make_nvq2(int output, int input) {
    constexpr int group_size = 24;
    constexpr int vector_size = 8;
    constexpr int state_bits = 4;
    constexpr int index_bits = 8;
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        "NVQ1",
        static_cast<std::uint8_t>(1 | 0x40),
        state_bits,
        group_size,
        output,
        input);
    for (int entry = 0; entry < 256; ++entry) {
        append<std::uint16_t>(
            blob,
            static_cast<std::uint16_t>(entry));
    }
    append_one_anchors(blob, output);
    const int groups = (input + group_size - 1) / group_size;
    const int vectors = (input + vector_size - 1) / vector_size;
    append_bytes(
        blob,
        pack_bits(
            filled(static_cast<std::size_t>(output) * groups, 1),
            state_bits));
    append_bytes(
        blob,
        pack_bits(
            filled(static_cast<std::size_t>(output) * vectors, 0),
            index_bits));
    append_bytes(
        blob,
        pack_bits(
            filled(
                static_cast<std::size_t>(output) *
                    ((input + 7) / 8),
                0),
            7));
    return {
        "NVQ2",
        std::move(blob),
        {},
        std::vector<float>(
            static_cast<std::size_t>(output) * input,
            1.0f),
        {},
        output,
        input,
    };
}

void append_nvq1_s_table(
    std::vector<std::uint8_t>& blob,
    bool reverse) {
    for (int bank = 0; bank < 2; ++bank) {
        for (int entry = 0; entry < 512; ++entry) {
            const int source =
                (reverse ? 511 - entry : entry) * 4;
            append<std::uint16_t>(
                blob,
                mfq::nvq_codebooks::
                    kNvq1LCodebookPacked[source]);
        }
    }
}

std::array<float, 8> nepq_vector(bool reverse) {
    const auto word =
        mfq::nvq_codebooks::kNvq1LCodebookPacked[
            reverse ? 511 * 4 : 0];
    std::array<float, 8> result{};
    for (int component = 0; component < 8; ++component) {
        const auto digit = (word >> (2 * component)) & 3u;
        require(digit <= 2, "invalid generated ternary codebook");
        result[component] =
            static_cast<float>(static_cast<int>(digit) - 1) +
            0.15625f;
    }
    return result;
}

TensorFixture make_nepq1_s(
    int output,
    int input,
    int sign_shift,
    std::uint64_t seed) {
    constexpr int rotation_block = 8;
    constexpr int table_banks = 2;
    require(
        input % rotation_block == 0,
        "NEPQ fixture width must divide the rotation block");
    std::vector<std::uint8_t> blob;
    append_magic(blob, "NEP1");
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, 2);
    append<std::uint8_t>(blob, 4);
    append<std::uint8_t>(blob, 1);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, table_banks);
    append<std::uint32_t>(blob, rotation_block);
    append<std::uint64_t>(blob, seed);
    for (int bank = 0; bank < table_banks; ++bank) {
        append_nvq1_s_table(blob, bank != 0);
    }
    append_one_anchors(blob, output);
    const int groups = (input + 23) / 24;
    const int supergroups = (groups + 3) / 4;
    append_bytes(
        blob,
        pack_bits(
            filled(
                static_cast<std::size_t>(output) * groups,
                1),
            4));
    append_bytes(
        blob,
        pack_bits(
            filled(
                static_cast<std::size_t>(output) * (input / 8),
                0),
            9));
    append_bytes(
        blob,
        pack_bits(
            filled(
                static_cast<std::size_t>(output) * groups,
                0),
            1));
    for (int row = 0;
         row < output * supergroups;
         ++row) {
        append<std::uint8_t>(
            blob,
            static_cast<std::uint8_t>(row & 1));
    }

    std::vector<float> dense(
        static_cast<std::size_t>(output) * input);
    for (int row = 0; row < output; ++row) {
        const auto values = nepq_vector((row & 1) != 0);
        for (int column = 0; column < input; ++column) {
            dense[static_cast<std::size_t>(row) * input + column] =
                values[static_cast<std::size_t>(column) % values.size()];
        }
    }

    std::vector<std::uint8_t> runtime;
    append_magic(runtime, "HSG1");
    append<std::uint32_t>(runtime, input);
    append<std::uint32_t>(runtime, rotation_block);
    append<std::uint64_t>(runtime, seed);
    std::vector<std::int8_t> signs(input);
    for (int column = 0; column < input; ++column) {
        signs[column] = static_cast<std::int8_t>(
            ((column + sign_shift) % 3) == 0 ? -1 : 1);
        append<std::int8_t>(runtime, signs[column]);
    }
    return {
        "NEPQ1-S",
        std::move(blob),
        std::move(runtime),
        std::move(dense),
        {rotation_block, std::move(signs)},
        output,
        input,
    };
}

struct MoeFixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    std::vector<RotationFixture> rotations;
    int output = 0;
    int input = 0;
};

MoeFixture make_mixed_moe(
    int output,
    int input,
    int salt) {
    std::vector<TensorFixture> tensors;
    tensors.push_back(make_nint(2, output, input, salt));
    tensors.push_back(make_nvq2(output, input));
    tensors.push_back(make_nint(5, output, input, salt + 7));
    tensors.push_back(make_nepq1_s(
        output,
        input,
        salt % 3,
        0x123456789abcdef0ull +
            static_cast<std::uint64_t>(salt)));

    std::vector<std::uint8_t> blob{'N', 'I', 'M', '2'};
    append<std::uint32_t>(blob, kExperts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, kExperts);
    std::vector<float> dense(
        static_cast<std::size_t>(kExperts) * output * input);
    std::vector<RotationFixture> rotations(kExperts);
    for (int expert = kExperts - 1; expert >= 0; --expert) {
        const auto& tensor = tensors[expert];
        append<std::uint32_t>(blob, 1);
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(tensor.dtype.size()));
        append<std::uint64_t>(blob, tensor.blob.size());
        append<std::uint64_t>(blob, tensor.runtime.size());
        append<std::int32_t>(blob, expert);
        blob.insert(blob.end(), tensor.dtype.begin(), tensor.dtype.end());
        append_bytes(blob, tensor.runtime);
        append_bytes(blob, tensor.blob);
        std::copy(
            tensor.dense.begin(),
            tensor.dense.end(),
            dense.begin() +
                static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(expert) *
                    output * input));
        rotations[expert] = tensor.rotation;
    }
    return {
        std::move(blob),
        std::move(dense),
        std::move(rotations),
        output,
        input,
    };
}

MoeFixture make_cccp_moe(
    int output,
    int input,
    int salt,
    std::array<bool, kExperts> included = {
        true,
        true,
        true,
        true,
    }) {
    struct Profile {
        const char* dtype;
        int tier;
        int vector_size;
        int entries;
        int bits;
    };
    const std::array<Profile, kExperts> profiles{{
        {"CCCP-X", 1, 8, 256, 8},
        {"CCCP-W", 2, 8, 4096, 12},
        {"CCCP-V", 3, 4, 256, 14},
        {"CCCP-VV", 4, 4, 4096, 16},
    }};
    std::vector<std::uint8_t> blob{
        'N', 'I', 'M', '2',
    };
    append<std::uint32_t>(blob, kExperts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    const auto pool_count = std::count(
        included.begin(),
        included.end(),
        true);
    require(
        pool_count > 0,
        "CCCP MoE fixture requires an expert");
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(
            pool_count));
    std::vector<float> dense(
        static_cast<std::size_t>(kExperts)
            * output * input);
    for (
        int expert = kExperts - 1;
        expert >= 0;
        --expert
    ) {
        if (!included[
                static_cast<std::size_t>(
                    expert)]) {
            continue;
        }
        const auto& profile =
            profiles[
                static_cast<std::size_t>(
                    expert)];
        require(
            input % profile.vector_size == 0,
            "CCCP MoE fixture vector mismatch");
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
                    static_cast<std::size_t>(
                        entry)
                        * profile.vector_size
                    + component
                ] =
                    static_cast<float>(
                        (
                            entry * 3
                            + component * 5
                            + salt
                            + expert * 7
                        ) % 31
                        - 15)
                    / 64.0f;
            }
        }
        std::vector<std::uint16_t> indices(
            static_cast<std::size_t>(
                output)
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
                            salt * 11
                            + expert * 13)
                        + profile.entries - 1
                    ) % profile.entries);
        }
        if (!indices.empty()) {
            indices.front() =
                static_cast<std::uint16_t>(
                    profile.entries - 1);
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
        append<std::int64_t>(payload, output);
        append<std::int64_t>(payload, input);
        append<std::uint32_t>(payload, output);
        for (const auto value : codebook) {
            append<float>(payload, value);
        }
        append_bytes(
            payload,
            pack_bits(indices, profile.bits));

        const std::string dtype(profile.dtype);
        append<std::uint32_t>(blob, 1);
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(
                dtype.size()));
        append<std::uint64_t>(
            blob,
            payload.size());
        append<std::uint64_t>(blob, 0);
        append<std::int32_t>(blob, expert);
        blob.insert(
            blob.end(),
            dtype.begin(),
            dtype.end());
        append_bytes(blob, payload);

        for (
            int row = 0;
            row < output;
            ++row
        ) {
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
                for (
                    int component = 0;
                    component
                        < profile.vector_size;
                    ++component
                ) {
                    dense[
                        (
                            static_cast<std::size_t>(
                                expert)
                                * output
                            + row
                        ) * input
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
    }
    return {
        std::move(blob),
        std::move(dense),
        std::vector<RotationFixture>(
            kExperts),
        output,
        input,
    };
}

std::vector<float> patterned(
    std::size_t count,
    int multiplier,
    int modulus,
    int center,
    float divisor) {
    std::vector<float> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        result[index] =
            static_cast<float>(
                static_cast<int>(
                    (index * static_cast<std::size_t>(multiplier) + 3) %
                    static_cast<std::size_t>(modulus)) -
                center) /
            divisor;
    }
    return result;
}

struct ModelFixture {
    TensorFixture router;
    TensorFixture shared_gate;
    TensorFixture shared_up;
    std::vector<float> shared_down;
    MoeFixture routed_gate_up;
    MoeFixture routed_down;
};

ModelFixture make_model_fixture() {
    return {
        make_nint(4, kExperts, kHidden, 3),
        make_nint(3, kIntermediate, kHidden, 9),
        make_nint(6, kIntermediate, kHidden, 15),
        patterned(
            static_cast<std::size_t>(kHidden) * kIntermediate,
            13,
            31,
            15,
            512.0f),
        make_mixed_moe(2 * kIntermediate, kHidden, 5),
        make_mixed_moe(kHidden, kIntermediate, 19),
    };
}

struct SplitModelFixture {
    ModelFixture reference;
    MoeFixture routed_gate;
    MoeFixture routed_up;
};

MoeFixture concatenate_gate_up_reference(
    const MoeFixture& gate,
    const MoeFixture& up) {
    require(
        gate.output == up.output &&
            gate.input == up.input &&
            gate.rotations.size() == up.rotations.size(),
        "split Gate/Up fixture dimensions mismatch");
    MoeFixture result;
    result.output = gate.output + up.output;
    result.input = gate.input;
    result.rotations = gate.rotations;
    result.dense.resize(
        static_cast<std::size_t>(kExperts) *
        result.output * result.input);
    for (int expert = 0; expert < kExperts; ++expert) {
        const auto& gate_rotation = gate.rotations[expert];
        const auto& up_rotation = up.rotations[expert];
        require(
            gate_rotation.block == up_rotation.block &&
                gate_rotation.signs == up_rotation.signs,
            "split Gate/Up fixture rotations mismatch");
        const auto copy_projection =
            [&](const MoeFixture& source, int output_offset) {
                const auto source_begin =
                    source.dense.begin() +
                    static_cast<std::ptrdiff_t>(
                        static_cast<std::size_t>(expert) *
                        source.output * source.input);
                const auto target_begin =
                    result.dense.begin() +
                    static_cast<std::ptrdiff_t>(
                        (static_cast<std::size_t>(expert) *
                             result.output +
                         output_offset) *
                        result.input);
                std::copy(
                    source_begin,
                    source_begin +
                        static_cast<std::ptrdiff_t>(
                            static_cast<std::size_t>(source.output) *
                            source.input),
                    target_begin);
            };
        copy_projection(gate, 0);
        copy_projection(up, gate.output);
    }
    return result;
}

SplitModelFixture make_split_model_fixture(bool cccp) {
    auto reference = make_model_fixture();
    auto gate = cccp
        ? make_cccp_moe(kIntermediate, kHidden, 31)
        : make_mixed_moe(kIntermediate, kHidden, 5);
    auto up = cccp
        ? make_cccp_moe(kIntermediate, kHidden, 37)
        : make_mixed_moe(kIntermediate, kHidden, 8);
    reference.routed_gate_up =
        concatenate_gate_up_reference(gate, up);
    if (cccp) {
        reference.routed_down =
            make_cccp_moe(kHidden, kIntermediate, 43);
    }
    return {
        std::move(reference),
        std::move(gate),
        std::move(up),
    };
}

DeepseekV4Config test_config(bool normalize) {
    DeepseekV4Config config;
    config.n_layers = 1;
    config.hidden = kHidden;
    config.n_experts = kExperts;
    config.top_k = kTopK;
    config.moe_inter = kIntermediate;
    config.n_shared = 1;
    config.n_heads = 2;
    config.head_dim = 16;
    config.q_lora_rank = 8;
    config.o_lora_rank = 8;
    config.o_groups = 1;
    config.kv_dim = 16;
    config.qk_rope_head_dim = 8;
    config.n_kv_heads = 1;
    config.vocab = kVocab;
    config.norm_topk_prob = normalize;
    config.routed_scaling = 1.5;
    config.swiglu_limit = 0.2;
    config.n_hash_layers = 1;
    config.sliding_window = 16;
    config.index_n_heads = 2;
    config.index_head_dim = 16;
    config.index_topk = 4;
    config.max_position_embeddings = 64;
    config.compress_ratios = {0};
    config.validate();
    return config;
}

array matrix(
    const std::vector<float>& values,
    int output,
    int input) {
    require(
        values.size() ==
            static_cast<std::size_t>(output) * input,
        "invalid dense fixture matrix");
    return array(values.begin(), Shape{output, input});
}

MlxLinear projection(
    const TensorFixture& weight,
    bool packed) {
    if (packed) {
        return MlxLinear(
            MlxNintWeight::from_blob(weight.blob));
    }
    return MlxLinear(
        matrix(weight.dense, weight.output, weight.input));
}

array availability_array(
    const std::array<bool, kExperts>& available) {
    std::array<std::uint8_t, kExperts> bytes{};
    for (int expert = 0; expert < kExperts; ++expert) {
        bytes[expert] =
            static_cast<std::uint8_t>(available[expert]);
    }
    return mlx::core::astype(
        array(bytes.begin(), Shape{kExperts}),
        mlx::core::bool_);
}

MlxDeepseekV4Moe make_moe(
    const ModelFixture& fixture,
    DeepseekV4Config config,
    bool packed_shared,
    const std::optional<std::vector<float>>& bias,
    const std::optional<std::vector<std::int32_t>>& token_experts,
    const std::array<bool, kExperts>& available) {
    std::optional<array> bias_array;
    if (bias.has_value()) {
        bias_array = array(bias->begin(), Shape{kExperts});
    }
    std::optional<array> table_array;
    if (token_experts.has_value()) {
        require(
            token_experts->size() ==
                static_cast<std::size_t>(kVocab * kTopK),
            "invalid hash table fixture");
        table_array = array(
            token_experts->begin(),
            Shape{kVocab, kTopK});
    }
    return MlxDeepseekV4Moe(
        std::move(config),
        projection(fixture.router, packed_shared),
        projection(fixture.shared_gate, packed_shared),
        projection(fixture.shared_up, packed_shared),
        MlxLinear(matrix(
            fixture.shared_down,
            kHidden,
            kIntermediate)),
        MlxRoutedLinear::from_blob(
            fixture.routed_gate_up.blob),
        MlxRoutedLinear::from_blob(
            fixture.routed_down.blob),
        std::move(bias_array),
        std::move(table_array),
        availability_array(available));
}

struct MappedTensor {
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

template <typename T>
std::vector<std::uint8_t> dense_payload(
    const std::vector<T>& values,
    const std::vector<std::int64_t>& shape) {
    std::size_t elements = 1;
    for (const auto dimension : shape) {
        require(
            dimension > 0,
            "dense MFQ fixture dimension must "
            "be positive");
        elements *=
            static_cast<std::size_t>(
                dimension);
    }
    require(
        elements == values.size(),
        "dense MFQ fixture element mismatch");
    std::vector<std::uint8_t> result;
    append<std::uint32_t>(
        result,
        static_cast<std::uint32_t>(
            shape.size()));
    for (const auto dimension : shape) {
        append<std::int64_t>(
            result,
            dimension);
    }
    const auto* bytes =
        reinterpret_cast<const std::uint8_t*>(
            values.data());
    result.insert(
        result.end(),
        bytes,
        bytes + values.size() * sizeof(T));
    return result;
}

class TemporaryDeepseekMfq {
public:
    explicit TemporaryDeepseekMfq(
        const std::vector<MappedTensor>& records,
        std::string_view architecture =
            "deepseek-v4-streamed-test")
        : path_(
              std::filesystem::
                  temp_directory_path()
              / "mfq-metal-dsv4-streamed-moe.mfq") {
        std::vector<std::uint8_t> file;
        append_magic(file, "MFQ1");
        append<std::uint32_t>(file, 1);
        append_mfq_string(
            file,
            architecture);
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
                "cannot create streamed DeepSeek-V4 "
                "MoE fixture");
        }
        stream.write(
            reinterpret_cast<const char*>(
                file.data()),
            static_cast<std::streamsize>(
                file.size()));
        if (!stream) {
            throw std::runtime_error(
                "cannot write streamed DeepSeek-V4 "
                "MoE fixture");
        }
    }

    ~TemporaryDeepseekMfq() {
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

std::vector<MappedTensor>
streamed_model_records(
    const ModelFixture& fixture,
    const std::vector<std::int32_t>&
        token_experts) {
    const auto layer =
        [](std::string_view suffix) {
            return std::string("layers.0.")
                + std::string(suffix);
        };
    return {
        {
            layer("ffn.gate.weight"),
            fixture.router.dtype,
            fixture.router.blob,
        },
        {
            layer(
                "ffn.shared_experts.w1.weight"),
            fixture.shared_gate.dtype,
            fixture.shared_gate.blob,
        },
        {
            layer(
                "ffn.shared_experts.w3.weight"),
            fixture.shared_up.dtype,
            fixture.shared_up.blob,
        },
        {
            layer(
                "ffn.shared_experts.w2.weight"),
            "F32",
            dense_payload(
                fixture.shared_down,
                {kHidden, kIntermediate}),
        },
        {
            layer(
                "ffn.experts.gate_up.weight"),
            "NINTM",
            fixture.routed_gate_up.blob,
        },
        {
            layer(
                "ffn.experts.down.weight"),
            "NINTM",
            fixture.routed_down.blob,
        },
        {
            layer("ffn.gate.tid2eid"),
            "I32",
            dense_payload(
                token_experts,
                {kVocab, kTopK}),
        },
    };
}

std::vector<MappedTensor>
split_model_records(
    const SplitModelFixture& fixture,
    const std::vector<std::int32_t>& token_experts,
    bool ew_names) {
    auto records = streamed_model_records(
        fixture.reference,
        token_experts);
    const auto combined = std::find_if(
        records.begin(),
        records.end(),
        [](const MappedTensor& record) {
            return record.name ==
                "layers.0.ffn.experts.gate_up.weight";
        });
    require(
        combined != records.end(),
        "combined Gate/Up record is missing from fixture");
    const auto position =
        static_cast<std::size_t>(
            std::distance(records.begin(), combined));
    records.erase(combined);
    records.insert(
        records.begin() +
            static_cast<std::ptrdiff_t>(position),
        {
            ew_names
                ? "blk.0.ffn_gate_exps.weight"
                : "layers.0.ffn.experts.gate.weight",
            "NINTM",
            fixture.routed_gate.blob,
        });
    records.insert(
        records.begin() +
            static_cast<std::ptrdiff_t>(position + 1),
        {
            ew_names
                ? "blk.0.ffn_up_exps.weight"
                : "layers.0.ffn.experts.up.weight",
            "NINTM",
            fixture.routed_up.blob,
        });
    if (ew_names) {
        for (auto& record : records) {
            if (record.name ==
                "layers.0.ffn.experts.down.weight") {
                record.name =
                    "blk.0.ffn_down_exps.weight";
            }
        }
    }
    return records;
}

std::vector<MappedTensor>
streamed_router_model_records(
    const ModelFixture& fixture,
    const std::vector<float>& bias) {
    auto records = streamed_model_records(
        fixture,
        std::vector<std::int32_t>(
            kVocab * kTopK,
            0));
    auto& router = records.back();
    router.name =
        "layers.0.ffn.gate.bias";
    router.dtype = "F32";
    router.payload = dense_payload(
        bias,
        {kExperts});
    return records;
}

float softplus_sqrt(float value) {
    const float softplus =
        value > 20.0f
        ? value
        : std::log1p(std::exp(value));
    return std::sqrt(softplus);
}

std::vector<float> matvec(
    const std::vector<float>& weight,
    int output,
    int input,
    const std::vector<float>& source) {
    require(
        source.size() == static_cast<std::size_t>(input),
        "reference matvec input mismatch");
    std::vector<float> result(output);
    for (int row = 0; row < output; ++row) {
        float value = 0.0f;
        for (int column = 0; column < input; ++column) {
            value +=
                source[column] *
                weight[static_cast<std::size_t>(row) * input + column];
        }
        result[row] = value;
    }
    return result;
}

std::vector<float> rotate(
    std::vector<float> values,
    const RotationFixture& rotation) {
    if (rotation.block == 0) {
        return values;
    }
    require(
        rotation.signs.size() == values.size() &&
            values.size() % static_cast<std::size_t>(rotation.block) == 0,
        "reference rotation shape mismatch");
    for (int start = 0;
         start < static_cast<int>(values.size());
         start += rotation.block) {
        for (int index = 0; index < rotation.block; ++index) {
            values[start + index] *=
                static_cast<float>(rotation.signs[start + index]);
        }
        for (int stride = 1; stride < rotation.block; stride <<= 1) {
            for (int base = 0; base < rotation.block; base += 2 * stride) {
                for (int offset = 0; offset < stride; ++offset) {
                    const int first = start + base + offset;
                    const int second = first + stride;
                    const float first_value = values[first];
                    const float second_value = values[second];
                    values[first] = first_value + second_value;
                    values[second] = first_value - second_value;
                }
            }
        }
        const float inverse =
            1.0f / std::sqrt(static_cast<float>(rotation.block));
        for (int index = 0; index < rotation.block; ++index) {
            values[start + index] *= inverse;
        }
    }
    return values;
}

std::vector<float> expert_matvec(
    const MoeFixture& weight,
    int expert,
    const std::vector<float>& source) {
    auto transformed = rotate(source, weight.rotations[expert]);
    const auto offset =
        static_cast<std::size_t>(expert) *
        weight.output * weight.input;
    std::vector<float> expert_weight(
        weight.dense.begin() +
            static_cast<std::ptrdiff_t>(offset),
        weight.dense.begin() +
            static_cast<std::ptrdiff_t>(
                offset +
                static_cast<std::size_t>(weight.output) *
                    weight.input));
    return matvec(
        expert_weight,
        weight.output,
        weight.input,
        transformed);
}

std::vector<int> ranked_experts(
    const std::vector<float>& scores,
    const std::array<bool, kExperts>& available,
    const std::optional<std::vector<float>>& bias) {
    std::vector<int> result;
    for (int expert = 0; expert < kExperts; ++expert) {
        if (available[expert]) {
            result.push_back(expert);
        }
    }
    std::sort(
        result.begin(),
        result.end(),
        [&](int left, int right) {
            const float left_score =
                scores[left] +
                (bias.has_value() ? (*bias)[left] : 0.0f);
            const float right_score =
                scores[right] +
                (bias.has_value() ? (*bias)[right] : 0.0f);
            if (left_score != right_score) {
                return left_score > right_score;
            }
            return left < right;
        });
    return result;
}

struct ReferenceResult {
    std::vector<float> output;
    std::vector<std::int32_t> ids;
    std::vector<float> weights;
};

ReferenceResult reference(
    const ModelFixture& fixture,
    const DeepseekV4Config& config,
    const std::vector<float>& input,
    int rows,
    const std::vector<std::int32_t>& token_ids,
    const std::optional<std::vector<float>>& bias,
    const std::optional<std::vector<std::int32_t>>& token_experts,
    const std::array<bool, kExperts>& available) {
    ReferenceResult result;
    result.output.resize(static_cast<std::size_t>(rows) * kHidden);
    result.ids.resize(static_cast<std::size_t>(rows) * kTopK);
    result.weights.resize(static_cast<std::size_t>(rows) * kTopK);
    for (int row = 0; row < rows; ++row) {
        const std::vector<float> source(
            input.begin() +
                static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(row) * kHidden),
            input.begin() +
                static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(row + 1) * kHidden));
        const auto logits = matvec(
            fixture.router.dense,
            kExperts,
            kHidden,
            source);
        std::vector<float> scores(kExperts);
        for (int expert = 0; expert < kExperts; ++expert) {
            scores[expert] = softplus_sqrt(logits[expert]);
        }

        std::array<int, kTopK> selected{};
        if (token_experts.has_value()) {
            const int token = token_ids[row];
            require(
                token >= 0 && token < kVocab,
                "reference token ID is outside the vocabulary");
            for (int route = 0; route < kTopK; ++route) {
                selected[route] = (*token_experts)[
                    static_cast<std::size_t>(token) * kTopK + route];
            }
            const auto candidates =
                ranked_experts(scores, available, std::nullopt);
            for (int route = 0; route < kTopK; ++route) {
                const int current = selected[route];
                if (current >= 0 &&
                    current < kExperts &&
                    available[current]) {
                    continue;
                }
                int replacement = -1;
                for (const int candidate : candidates) {
                    if (std::find(
                            selected.begin(),
                            selected.end(),
                            candidate) == selected.end()) {
                        replacement = candidate;
                        break;
                    }
                }
                require(
                    replacement >= 0,
                    "reference hash repair exhausted candidates");
                selected[route] = replacement;
            }
        } else {
            const auto ranked = ranked_experts(scores, available, bias);
            for (int route = 0; route < kTopK; ++route) {
                selected[route] = ranked[route];
            }
        }

        float denominator = 1.0f;
        if (config.norm_topk_prob) {
            denominator = 0.0f;
            for (const int expert : selected) {
                denominator += scores[expert];
            }
            denominator = std::max(denominator, 1e-20f);
        }
        for (int route = 0; route < kTopK; ++route) {
            const auto offset =
                static_cast<std::size_t>(row) * kTopK + route;
            result.ids[offset] = selected[route];
            result.weights[offset] =
                scores[selected[route]] / denominator *
                static_cast<float>(config.routed_scaling);
        }

        auto shared_gate = matvec(
            fixture.shared_gate.dense,
            kIntermediate,
            kHidden,
            source);
        auto shared_up = matvec(
            fixture.shared_up.dense,
            kIntermediate,
            kHidden,
            source);
        std::vector<float> shared_hidden(kIntermediate);
        for (int column = 0; column < kIntermediate; ++column) {
            float gate = shared_gate[column];
            float up = shared_up[column];
            if (config.swiglu_limit > 0.0) {
                gate = std::min(
                    gate,
                    static_cast<float>(config.swiglu_limit));
                up = std::clamp(
                    up,
                    -static_cast<float>(config.swiglu_limit),
                    static_cast<float>(config.swiglu_limit));
            }
            shared_hidden[column] =
                gate / (1.0f + std::exp(-gate)) * up;
        }
        auto combined = matvec(
            fixture.shared_down,
            kHidden,
            kIntermediate,
            shared_hidden);

        for (int route = 0; route < kTopK; ++route) {
            const int expert = selected[route];
            auto gate_up = expert_matvec(
                fixture.routed_gate_up,
                expert,
                source);
            std::vector<float> hidden(kIntermediate);
            for (int column = 0; column < kIntermediate; ++column) {
                float gate = gate_up[column];
                float up = gate_up[kIntermediate + column];
                if (config.swiglu_limit > 0.0) {
                    gate = std::min(
                        gate,
                        static_cast<float>(config.swiglu_limit));
                    up = std::clamp(
                        up,
                        -static_cast<float>(config.swiglu_limit),
                        static_cast<float>(config.swiglu_limit));
                }
                hidden[column] =
                    gate / (1.0f + std::exp(-gate)) * up;
            }
            const auto down = expert_matvec(
                fixture.routed_down,
                expert,
                hidden);
            const float route_weight = result.weights[
                static_cast<std::size_t>(row) * kTopK + route];
            for (int column = 0; column < kHidden; ++column) {
                combined[column] += route_weight * down[column];
            }
        }
        std::copy(
            combined.begin(),
            combined.end(),
            result.output.begin() +
                static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(row) * kHidden));
    }
    return result;
}

std::vector<float> evaluated_floats(array value) {
    value = mlx::core::astype(value, mlx::core::float32);
    value.eval();
    return {
        value.data<float>(),
        value.data<float>() + value.size(),
    };
}

std::vector<std::int32_t> evaluated_ids(array value) {
    value = mlx::core::astype(value, mlx::core::int32);
    value.eval();
    return {
        value.data<std::int32_t>(),
        value.data<std::int32_t>() + value.size(),
    };
}

void compare(
    MlxDeepseekV4MoeResult actual,
    const ReferenceResult& expected,
    float output_tolerance = 2e-2f) {
    const auto output = evaluated_floats(std::move(actual.output));
    const auto ids = evaluated_ids(std::move(actual.expert_ids));
    const auto weights =
        evaluated_floats(std::move(actual.expert_weights));
    require(
        output.size() == expected.output.size() &&
            ids.size() == expected.ids.size() &&
            weights.size() == expected.weights.size(),
        "DeepSeek-V4 MoE result size mismatch");
    for (std::size_t index = 0; index < ids.size(); ++index) {
        require(
            ids[index] == expected.ids[index],
            "DeepSeek-V4 MoE expert ID mismatch");
        // Packed Metal projections use SIMD reduction order and Fast Math;
        // the scalar CPU dense reference uses libm and a different sum order.
        require_close(
            weights[index],
            expected.weights[index],
            2e-4f,
            "route weight");
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
        require_close(
            output[index],
            expected.output[index],
            output_tolerance,
            "output");
    }
}

std::vector<float> input_values(int rows, int salt = 0) {
    std::vector<float> result(
        static_cast<std::size_t>(rows) * kHidden);
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] =
            static_cast<float>(
                static_cast<int>(
                    (index * 17 +
                     static_cast<std::size_t>(salt * 11 + 5)) %
                    37) -
                18) /
            5.0f;
    }
    return result;
}

const std::array<bool, kExperts> kAvailable{
    true,
    true,
    false,
    true,
};

const std::vector<float> kBias{
    0.1f,
    3.0f,
    100.0f,
    -0.2f,
};

const std::vector<std::int32_t> kTokenExperts{
    2, 0,
    1, 2,
    -1, 3,
    7, 1,
    0, 3,
    2, 2,
};

void test_grouped_and_fallbacks(
    const ModelFixture& fixture) {
    auto config = test_config(true);
    auto grouped = make_moe(
        fixture,
        config,
        true,
        kBias,
        std::nullopt,
        kAvailable);
    require(
        grouped.uses_grouped_shared_projection(),
        "packed router/shared projections did not form one group");

    {
        constexpr int rows = 3;
        const auto input = input_values(rows);
        const std::vector<std::int32_t> tokens{0, 1, 2};
        const auto expected = reference(
            fixture,
            config,
            input,
            rows,
            tokens,
            kBias,
            std::nullopt,
            kAvailable);
        auto actual = grouped.forward_with_routing(
            array(input.begin(), Shape{1, rows, kHidden}),
            array(tokens.begin(), Shape{1, rows}));
        require(
            actual.output.shape() == Shape{1, rows, kHidden},
            "DeepSeek-V4 MoE did not restore the input prefix shape");
        compare(std::move(actual), expected);

        auto unlimited = config;
        unlimited.swiglu_limit = 0.0;
        const auto without_limit = reference(
            fixture,
            unlimited,
            input,
            rows,
            tokens,
            kBias,
            std::nullopt,
            kAvailable);
        float maximum_difference = 0.0f;
        for (std::size_t index = 0;
             index < expected.output.size();
             ++index) {
            maximum_difference = std::max(
                maximum_difference,
                std::fabs(
                    expected.output[index] -
                    without_limit.output[index]));
        }
        require(
            maximum_difference > 1e-2f,
            "limited SwiGLU fixture did not exercise clipping");
    }

    // The grouped projection kernel is intentionally decode-sized. Seventeen
    // rows force the same packed weights through the independent fallback.
    {
        constexpr int rows = 17;
        const auto input = input_values(rows, 4);
        std::vector<std::int32_t> tokens(rows);
        for (int row = 0; row < rows; ++row) {
            tokens[row] = row % kVocab;
        }
        const auto expected = reference(
            fixture,
            config,
            input,
            rows,
            tokens,
            kBias,
            std::nullopt,
            kAvailable);
        compare(
            grouped.forward_with_routing(
                array(input.begin(), Shape{rows, kHidden}),
                array(tokens.begin(), Shape{rows})),
            expected);
    }

    // Dense projections cannot enter the packed grouped kernel and exercise
    // the structural fallback even for decode-sized inputs.
    {
        constexpr int rows = 4;
        auto dense = make_moe(
            fixture,
            config,
            false,
            kBias,
            std::nullopt,
            kAvailable);
        require(
            !dense.uses_grouped_shared_projection(),
            "dense shared projections unexpectedly formed a packed group");
        const auto input = input_values(rows, 7);
        const std::vector<std::int32_t> tokens{0, 1, 2, 3};
        const auto expected = reference(
            fixture,
            config,
            input,
            rows,
            tokens,
            kBias,
            std::nullopt,
            kAvailable);
        compare(
            dense.forward_with_routing(
                array(input.begin(), Shape{rows, kHidden}),
                array(tokens.begin(), Shape{rows})),
            expected);
    }
}

void test_unnormalized_bias_routing(
    const ModelFixture& fixture) {
    constexpr int rows = 5;
    auto config = test_config(false);
    auto moe = make_moe(
        fixture,
        config,
        true,
        kBias,
        std::nullopt,
        kAvailable);
    const auto input = input_values(rows, 13);
    const std::vector<std::int32_t> tokens{0, 1, 2, 3, 4};
    const auto expected = reference(
        fixture,
        config,
        input,
        rows,
        tokens,
        kBias,
        std::nullopt,
        kAvailable);
    auto actual = moe.forward_with_routing(
        array(input.begin(), Shape{rows, kHidden}),
        array(tokens.begin(), Shape{rows}));
    const auto weights =
        evaluated_floats(actual.expert_weights);
    bool differs_from_normalized_sum = false;
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int route = 0; route < kTopK; ++route) {
            sum += weights[
                static_cast<std::size_t>(row) * kTopK + route];
        }
        differs_from_normalized_sum =
            differs_from_normalized_sum ||
            std::fabs(sum -
                static_cast<float>(config.routed_scaling)) >
                1e-3f;
    }
    require(
        differs_from_normalized_sum,
        "normalize=false unexpectedly normalized route weights");
    compare(std::move(actual), expected);
}

void test_hash_repair_and_mixed_formats(
    const ModelFixture& fixture) {
    constexpr int rows = 6;
    auto config = test_config(true);
    auto moe = make_moe(
        fixture,
        config,
        true,
        std::nullopt,
        kTokenExperts,
        kAvailable);
    const auto input = input_values(rows, 21);
    const std::vector<std::int32_t> tokens{0, 1, 2, 3, 4, 5};
    const auto expected = reference(
        fixture,
        config,
        input,
        rows,
        tokens,
        std::nullopt,
        kTokenExperts,
        kAvailable);
    auto actual = moe.forward_with_routing(
        array(input.begin(), Shape{2, 3, kHidden}),
        array(tokens.begin(), Shape{2, 3}));
    const auto ids = evaluated_ids(actual.expert_ids);
    bool repaired = false;
    std::array<bool, kExperts> selected{};
    for (int row = 0; row < rows; ++row) {
        for (int route = 0; route < kTopK; ++route) {
            const auto index =
                static_cast<std::size_t>(row) * kTopK + route;
            const int expert = ids[index];
            require(
                expert >= 0 &&
                    expert < kExperts &&
                    kAvailable[expert],
                "hash repair retained an unavailable/invalid expert");
            selected[expert] = true;
            repaired = repaired ||
                expert != kTokenExperts[
                    static_cast<std::size_t>(tokens[row]) *
                        kTopK +
                    route];
        }
        require(
            ids[static_cast<std::size_t>(row) * kTopK] !=
                ids[static_cast<std::size_t>(row) * kTopK + 1],
            "hash repair produced duplicate routes");
    }
    require(repaired, "hash fixture did not repair tid2eid routes");
    require(
        selected[0] && selected[1] && selected[3],
        "mixed NINT/NVQ/NEPQ experts were not all dispatched");
    compare(std::move(actual), expected, 3e-2f);
}

void test_split_gate_up_eager_load_and_forward() {
    const auto fixture = make_split_model_fixture(false);
    const TemporaryDeepseekMfq file(
        split_model_records(
            fixture,
            kTokenExperts,
            true),
        "deepseek_v4-ew-mfq");
    const mfq::metal::MfqContainer model(file.path());
    const auto config = test_config(true);
    auto moe = MlxDeepseekV4Moe::load(
        model,
        config,
        0,
        availability_array(kAvailable));
    require(
        !moe.uses_streamed_experts(),
        "split NINTM Gate/Up unexpectedly selected offload");

    const auto run = [&](int rows, int salt) {
        const auto input = input_values(rows, salt);
        std::vector<std::int32_t> tokens(
            static_cast<std::size_t>(rows));
        for (int row = 0; row < rows; ++row) {
            tokens[static_cast<std::size_t>(row)] = row % kVocab;
        }
        const auto expected = reference(
            fixture.reference,
            config,
            input,
            rows,
            tokens,
            std::nullopt,
            kTokenExperts,
            kAvailable);
        auto actual = moe.forward_with_routing(
            array(input.begin(), Shape{rows, kHidden}),
            array(tokens.begin(), Shape{rows}));
        compare(std::move(actual), expected, 3e-2f);
    };
    run(3, 27);
    run(34, 29);
}

void test_split_gate_up_streamed_load_and_forward() {
    const auto fixture = make_split_model_fixture(true);
    const TemporaryDeepseekMfq file(
        split_model_records(
            fixture,
            kTokenExperts,
            true),
        "deepseek_v4-ew-mfq");
    const mfq::metal::MfqContainer model(file.path());
    auto residency =
        std::make_shared<mfq::metal::MlxCccpExpertResidency>(
            model,
            0,
            kExperts);
    const auto config = test_config(true);
    auto moe = MlxDeepseekV4Moe::load(
        model,
        config,
        0,
        availability_array(kAvailable),
        residency);
    require(
        moe.uses_streamed_experts(),
        "split CCCP Gate/Up did not select streamed residency");

    const auto run = [&](int rows, int salt) {
        const auto input = input_values(rows, salt);
        std::vector<std::int32_t> tokens(
            static_cast<std::size_t>(rows));
        for (int row = 0; row < rows; ++row) {
            tokens[static_cast<std::size_t>(row)] = row % kVocab;
        }
        const auto expected = reference(
            fixture.reference,
            config,
            input,
            rows,
            tokens,
            std::nullopt,
            kTokenExperts,
            kAvailable);
        compare(
            moe.forward_with_routing(
                array(input.begin(), Shape{rows, kHidden}),
                array(tokens.begin(), Shape{rows})),
            expected,
            3e-2f);
    };
    run(1, 31);
    run(35, 33);
}

void test_split_gate_up_requires_pair() {
    const auto fixture = make_split_model_fixture(false);
    auto records = split_model_records(
        fixture,
        kTokenExperts,
        false);
    records.erase(
        std::remove_if(
            records.begin(),
            records.end(),
            [](const MappedTensor& record) {
                return record.name ==
                    "layers.0.ffn.experts.up.weight";
            }),
        records.end());
    const TemporaryDeepseekMfq file(records);
    const mfq::metal::MfqContainer model(file.path());
    bool rejected = false;
    try {
        (void)MlxDeepseekV4Moe::load(
            model,
            test_config(true),
            0,
            availability_array(kAvailable));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(
        rejected,
        "DeepSeek-V4 accepted an incomplete split Gate/Up pair");
}

void test_streamed_cccp_load_and_forward() {
    auto fixture = make_model_fixture();
    fixture.routed_gate_up =
        make_cccp_moe(
            2 * kIntermediate,
            kHidden,
            31);
    fixture.routed_down =
        make_cccp_moe(
            kHidden,
            kIntermediate,
            43);
    const TemporaryDeepseekMfq file(
        streamed_model_records(
            fixture,
            kTokenExperts));
    const mfq::metal::MfqContainer model(
        file.path());
    auto residency =
        std::make_shared<
            mfq::metal::
                MlxCccpExpertResidency>(
            model,
            0,
            kExperts);
    auto config = test_config(true);
    auto moe = MlxDeepseekV4Moe::load(
        model,
        config,
        0,
        availability_array(kAvailable),
        residency);
    require(
        moe.uses_streamed_experts(),
        "DeepSeek-V4 MoE did not select streamed "
        "CCCP residency");
    require(
        moe.uses_grouped_shared_projection(),
        "streamed DeepSeek-V4 shared projections "
        "did not remain grouped");

    // Decode/small-M retains one lazy gate->down graph.  The result
    // materialization below is its only post-routing evaluation boundary.
    {
        constexpr int rows = 1;
        const auto input = input_values(rows, 6);
        const std::vector<std::int32_t> tokens{3};
        const auto expected = reference(
            fixture,
            config,
            input,
            rows,
            tokens,
            std::nullopt,
            kTokenExperts,
            kAvailable);
        compare(
            moe.forward_with_routing(
                array(
                    input.begin(),
                    Shape{rows, kHidden}),
                array(
                    tokens.begin(),
                    Shape{rows})),
            expected,
            3e-2f);
        require(
            residency->cached_expert_count()
                <= kTopK,
            "decode retained inactive streamed "
            "experts with a zero-byte cache");
    }

    // Audit the exact 16-row boundary, a three-chunk 33+ path, and a
    // nontrivial batch prefix. Each weighted chunk must materialize before
    // the next active set changes a zero-byte residency.
    const auto run_rows =
        [&](int rows,
            Shape input_shape,
            Shape token_shape,
            int salt,
            const char* label) {
            const auto input =
                input_values(rows, salt);
            std::vector<std::int32_t> tokens(
                static_cast<std::size_t>(
                    rows));
            for (int row = 0;
                 row < rows;
                 ++row) {
                tokens[
                    static_cast<std::size_t>(
                        row)] =
                    row % kVocab;
            }
            const auto expected = reference(
                fixture,
                config,
                input,
                rows,
                tokens,
                std::nullopt,
                kTokenExperts,
                kAvailable);
            constexpr int stream_rows = 16;
            const int final_chunk_start =
                ((rows - 1) / stream_rows)
                * stream_rows;
            std::array<bool, kExperts>
                final_active{};
            for (
                int row = final_chunk_start;
                row < rows;
                ++row
            ) {
                for (
                    int route = 0;
                    route < kTopK;
                    ++route
                ) {
                    final_active[
                        static_cast<std::size_t>(
                            expected.ids[
                                static_cast<
                                    std::size_t>(
                                    row)
                                    * kTopK
                                + route])] =
                        true;
                }
            }
            const auto final_active_count =
                static_cast<std::size_t>(
                    std::count(
                        final_active.begin(),
                        final_active.end(),
                        true));
            auto actual =
                moe.forward_with_routing(
                    array(
                        input.begin(),
                        input_shape),
                    array(
                        tokens.begin(),
                        token_shape));
            require(
                actual.output.shape() ==
                    input_shape,
                std::string(label) +
                    " did not restore the "
                    "input batch prefix");
            compare(
                std::move(actual),
                expected,
                3e-2f);
            require(
                residency->cache_limit_bytes() == 0
                    && residency
                        ->cached_expert_count()
                        == final_active_count
                    && residency
                        ->resident_packed_bytes()
                        > 0,
                std::string(label) +
                    " did not retain only the "
                    "current active experts");
        };
    run_rows(
        16,
        Shape{16, kHidden},
        Shape{16},
        8,
        "16-row streamed CCCP");
    run_rows(
        35,
        Shape{35, kHidden},
        Shape{35},
        9,
        "35-row streamed CCCP");
    run_rows(
        34,
        Shape{2, 17, kHidden},
        Shape{2, 17},
        10,
        "batched streamed CCCP");
}

void test_streamed_router_availability_intersection() {
    constexpr std::array<bool, kExperts> gate_available{
        true,
        true,
        true,
        false,
    };
    constexpr std::array<bool, kExperts> down_available{
        false,
        true,
        true,
        true,
    };
    constexpr std::array<bool, kExperts> intersection{
        false,
        true,
        true,
        false,
    };
    constexpr std::array<bool, kExperts> requested{
        true,
        true,
        true,
        true,
    };

    auto fixture = make_model_fixture();
    fixture.routed_gate_up =
        make_cccp_moe(
            2 * kIntermediate,
            kHidden,
            53,
            gate_available);
    fixture.routed_down =
        make_cccp_moe(
            kHidden,
            kIntermediate,
            61,
            down_available);
    const TemporaryDeepseekMfq file(
        streamed_router_model_records(
            fixture,
            kBias));
    const mfq::metal::MfqContainer model(
        file.path());
    auto residency =
        std::make_shared<
            mfq::metal::
                MlxCccpExpertResidency>(
            model,
            0,
            kExperts);
    auto config = test_config(true);
    config.n_hash_layers = 0;
    config.validate();
    auto moe = MlxDeepseekV4Moe::load(
        model,
        config,
        0,
        availability_array(requested),
        residency);
    require(
        moe.uses_streamed_experts(),
        "ordinary router did not retain streamed "
        "CCCP experts");

    constexpr int rows = 7;
    const auto input = input_values(rows, 17);
    std::vector<std::int32_t> tokens(rows);
    for (int row = 0; row < rows; ++row) {
        tokens[
            static_cast<std::size_t>(
                row)] =
            row % kVocab;
    }
    const auto expected = reference(
        fixture,
        config,
        input,
        rows,
        tokens,
        kBias,
        std::nullopt,
        intersection);
    auto actual = moe.forward_with_routing(
        array(
            input.begin(),
            Shape{rows, kHidden}),
        array(
            tokens.begin(),
            Shape{rows}));
    const auto ids =
        evaluated_ids(actual.expert_ids);
    bool selected_one = false;
    bool selected_two = false;
    for (const auto expert : ids) {
        require(
            expert == 1 || expert == 2,
            "ordinary router selected an expert "
            "outside the gate/down availability "
            "intersection");
        selected_one = selected_one ||
            expert == 1;
        selected_two = selected_two ||
            expert == 2;
    }
    require(
        selected_one && selected_two,
        "ordinary router did not exercise both "
        "experts in the gate/down intersection");
    compare(
        std::move(actual),
        expected,
        3e-2f);
}

void test_availability_validation(
    const ModelFixture& fixture) {
    bool rejected = false;
    try {
        (void)make_moe(
            fixture,
            test_config(true),
            true,
            kBias,
            std::nullopt,
            {true, false, false, false});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(
        rejected,
        "MoE accepted fewer available experts than top_k");
}

} // namespace

int main() {
    try {
        const auto fixture = make_model_fixture();
        test_grouped_and_fallbacks(fixture);
        test_unnormalized_bias_routing(fixture);
        test_hash_repair_and_mixed_formats(fixture);
        test_split_gate_up_eager_load_and_forward();
        test_split_gate_up_streamed_load_and_forward();
        test_split_gate_up_requires_pair();
        test_streamed_cccp_load_and_forward();
        test_streamed_router_availability_intersection();
        test_availability_validation(fixture);
        std::cout
            << "MFQ native DeepSeek-V4 heterogeneous/"
               "streamed CCCP MoE Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "MFQ native DeepSeek-V4 MoE Metal tests failed: "
            << error.what() << '\n';
        return 1;
    }
}
