#include "mlx_moe.h"
#include "mlx_grouped_linear.h"
#include "mlx_tensor.h"
#include "mlx_moe_ops.h"
#include "mlx_mxfp4_sq.h"
#include "mlx_nint.h"
#include "mlx_reference.h"
#include "mfq_container.h"
#include "mlx_qwen4_causal_lm.h"
#include "mlx_deepseek_v41_moe.h"
#include "mlx_deepseek_v41_causal_lm.h"
#include "qwen4_ops.h"
#include "mlx_kernel_prepare.h"
#include "mlx_inference_warmup.h"
#include "mlx_resource_telemetry.h"
#include "mlx_platform.h"
#include "mlx_resident_budget.h"
#include "mlx_ane_matmul.h"
#include "mlx_paged_session_codec.h"
#include "nlohmann/json.hpp"

#include "nvq_codebooks.generated.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <thread>
#include <utility>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/fast_primitives.h>
#include <mlx/backend/metal/device.h>

extern "C" __attribute__((noinline)) void mfq_kernel_prepare_phase(int phase) {
    std::cout << "kernel_prepare_phase=" << phase << std::endl;
}

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

struct Mxfp4Fixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    int rows = 0;
    int input = 0;
};

struct Mxfp8Fixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    int rows = 0;
    int input = 0;
};

struct Mxfp4SqFixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    int rows = 0;
    int input = 0;
};

float decode_mxfp4_code(std::uint8_t code) {
    constexpr std::array<float, 8> table{
        0.0f, 0.5f, 1.0f, 1.5f,
        2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float value = table[code & 7u];
    return (code & 8u) == 0u ? value : -value;
}

Mxfp4Fixture make_mxfp4(int rows, int input, int seed = 0) {
    if (rows <= 0 || input <= 0 || input % 32 != 0) {
        throw std::runtime_error("invalid MXFP4 fixture geometry");
    }
    std::vector<std::uint8_t> values(
        static_cast<std::size_t>(rows) * input / 2);
    std::vector<std::uint8_t> scales(
        static_cast<std::size_t>(rows) * input / 32);
    std::vector<float> dense(
        static_cast<std::size_t>(rows) * input);
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < input / 32; ++group) {
            scales[
                static_cast<std::size_t>(row) * (input / 32)
                + group
            ] = static_cast<std::uint8_t>(
                125 + (row + group + seed) % 5);
        }
        for (int column = 0; column < input; column += 2) {
            const auto low = static_cast<std::uint8_t>(
                (row * 7 + column * 3 + seed * 5 + 1) & 15);
            const auto high = static_cast<std::uint8_t>(
                (row * 11 + column * 5 + seed * 3 + 6) & 15);
            values[
                static_cast<std::size_t>(row) * (input / 2)
                + column / 2
            ] = static_cast<std::uint8_t>(low | (high << 4));
            const float scale = std::ldexp(
                1.0f,
                static_cast<int>(scales[
                    static_cast<std::size_t>(row) * (input / 32)
                    + column / 32
                ]) - 127);
            dense[static_cast<std::size_t>(row) * input + column] =
                decode_mxfp4_code(low) * scale;
            dense[static_cast<std::size_t>(row) * input + column + 1] =
                decode_mxfp4_code(high) * scale;
        }
    }
    std::vector<std::uint8_t> blob;
    append_magic(blob, "MXT1");
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, 4);
    append<std::uint16_t>(blob, 0);
    append<std::uint64_t>(blob, rows);
    append<std::uint64_t>(blob, input);
    append<std::uint64_t>(blob, rows);
    append<std::uint64_t>(blob, input / 2);
    append<std::uint64_t>(blob, rows);
    append<std::uint64_t>(blob, input / 32);
    append_bytes(blob, values);
    append_bytes(blob, scales);
    return {
        std::move(blob),
        std::move(dense),
        rows,
        input,
    };
}

Mxfp4SqFixture make_mxfp4_sq2(int rows, int input) {
    if (rows <= 0 || input <= 0 || input % 32 != 0) {
        throw std::runtime_error("invalid MXFP4-SQ fixture geometry");
    }
    constexpr std::uint8_t base = 124;
    const auto weights = static_cast<std::size_t>(rows) * input;
    const auto blocks = weights / 32;
    std::vector<std::uint8_t> symbols(weights, 0);
    std::vector<std::uint8_t> selectors(blocks, 0);
    std::vector<std::uint8_t> state_scales(
        static_cast<std::size_t>(rows) * 8);
    std::vector<std::uint8_t> state_palettes(
        static_cast<std::size_t>(rows) * 8);
    std::vector<float> dense(weights);
    for (int row = 0; row < rows; ++row) {
        const auto scale_offset = static_cast<std::uint8_t>(row & 3);
        const auto palette = static_cast<std::uint8_t>((row * 7 + 3) & 31);
        for (int state = 0; state < 8; ++state) {
            const auto index = static_cast<std::size_t>(row) * 8 + state;
            state_scales[index] = scale_offset;
            state_palettes[index] = palette;
        }
        const auto nibble = mfq::metal::kMxfp4Sq2PaletteNibbles[
            static_cast<std::size_t>(palette) * 4];
        const float value = decode_mxfp4_code(nibble) *
            std::ldexp(1.0f, static_cast<int>(base + scale_offset) - 127);
        std::fill_n(
            dense.begin() + static_cast<std::ptrdiff_t>(row) * input,
            input,
            value);
    }
    std::vector<std::uint8_t> blob;
    append_magic(blob, std::string_view("SQ2\0", 4));
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, base);
    append<std::uint16_t>(blob, 0);
    append<std::uint64_t>(blob, rows);
    append<std::uint64_t>(blob, input);
    append_bytes(blob, pack_values(symbols, 2));
    append_bytes(blob, pack_values(selectors, 1));
    append_bytes(blob, pack_values(state_scales, 2));
    append_bytes(blob, pack_values(state_palettes, 5));
    return {std::move(blob), std::move(dense), rows, input};
}

float decode_mxfp8_code(std::uint8_t code) {
    const auto magnitude = static_cast<std::uint8_t>(code & 0x7fu);
    const int exponent = magnitude >> 3;
    const int mantissa = magnitude & 7;
    if (exponent == 15 && mantissa == 7) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float value = exponent == 0
        ? static_cast<float>(mantissa) / 512.0f
        : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
              exponent - 7);
    return (code & 0x80u) == 0u ? value : -value;
}

Mxfp8Fixture make_mxfp8(int rows, int input, int salt = 0) {
    if (rows <= 0 || input <= 0 || input % 128 != 0) {
        throw std::runtime_error("invalid MXFP8 fixture geometry");
    }
    const int groups = input / 128;
    const int scale_rows = (rows + 127) / 128;
    std::vector<std::uint8_t> values(
        static_cast<std::size_t>(rows) * input);
    std::vector<std::uint8_t> scales(
        static_cast<std::size_t>(scale_rows) * groups);
    std::vector<float> dense(
        static_cast<std::size_t>(rows) * input);
    constexpr std::array<std::uint8_t, 12> codes{
        0x00, 0x30, 0x34, 0x38, 0x3c, 0x40,
        0x80, 0xb0, 0xb4, 0xb8, 0xbc, 0xc0,
    };
    for (int scale_row = 0; scale_row < scale_rows; ++scale_row) {
        for (int group = 0; group < groups; ++group) {
            scales[static_cast<std::size_t>(scale_row) * groups + group] =
                static_cast<std::uint8_t>(126 + (scale_row + group + salt) % 3);
        }
    }
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < input; ++column) {
            const auto code = codes[static_cast<std::size_t>(
                row * 7 + column * 5 + salt) % codes.size()];
            values[static_cast<std::size_t>(row) * input + column] = code;
            const float scale = std::ldexp(
                1.0f,
                static_cast<int>(scales[
                    static_cast<std::size_t>(row / 128) * groups
                    + column / 128]) - 127);
            dense[static_cast<std::size_t>(row) * input + column] =
                decode_mxfp8_code(code) * scale;
        }
    }
    std::vector<std::uint8_t> blob;
    append_magic(blob, "MXT1");
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, 8);
    append<std::uint16_t>(blob, 0);
    append<std::uint64_t>(blob, rows);
    append<std::uint64_t>(blob, input);
    append<std::uint64_t>(blob, rows);
    append<std::uint64_t>(blob, input);
    append<std::uint64_t>(blob, scale_rows);
    append<std::uint64_t>(blob, groups);
    append_bytes(blob, values);
    append_bytes(blob, scales);
    return {
        std::move(blob),
        std::move(dense),
        rows,
        input,
    };
}

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

VqFixture make_plain_nvq3(
    int output,
    int input) {
    constexpr std::uint8_t custom_codebook = 0x40;
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        "NVQ1",
        static_cast<std::uint8_t>(2 | custom_codebook),
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
        4,
        4,
        8,
        7,
        true);
    return {
        "NVQ3",
        std::move(blob),
        {},
        repeated_vq_dense(
            output,
            input,
            std::vector<float>(4, 1.0f)),
        {},
        0,
        0,
        output,
        input,
    };
}

VqFixture make_varied_plain_nvq(int output, int input, bool d4 = false, bool index_parity = false,
    int group_size = 24) {
    auto result = d4 ? make_plain_nvq3(output, input)
        : make_plain_nvq(output, input, index_parity);
    const int vector_size = d4 ? 4 : 8;
    const int old_groups = (input + 23) / 24;
    const int groups = (input + group_size - 1) / group_size;
    const int vectors = (input + vector_size - 1) / vector_size;
    const int signs = (input + 7) / 8;
    result.blob.resize(result.blob.size() - output * 2
        - (output * old_groups + 1) / 2 - output * vectors - (output * signs * 7 + 7) / 8);
    const auto group16 = static_cast<std::uint16_t>(group_size);
    std::memcpy(result.blob.data() + 6, &group16, sizeof(group16));
    append_vq_anchors(result.blob, output);
    std::vector<std::uint16_t> states(output * groups);
    std::vector<std::uint16_t> indices(output * vectors);
    std::vector<std::uint16_t> masks(output * signs);
    for (int row = 0; row < output; ++row) {
        for (int group = 0; group < groups; ++group) {
            states[row * groups + group] = (row + group * 3) & 15;
        }
        for (int vector = 0; vector < vectors; ++vector) {
            indices[row * vectors + vector] = (row * 137 + vector * 59 + 257) & 255;
        }
        for (int sign = 0; sign < signs; ++sign) {
            masks[row * signs + sign] = (row * 13 + sign * 7 + 21) & 127;
        }
        for (int column = 0; column < input; ++column) {
            const auto index = indices[row * vectors + column / vector_size];
            const auto mask = masks[row * signs + column / 8];
            const auto component = column % vector_size;
            float value = float(2 * ((index >> (component * (d4 ? 3 : 2)))
                & (d4 ? 7 : 3)) + 1) * states[row * groups + column / group_size];
            const auto negative = column % 8 < 7 ? (mask >> (column % 8)) & 1u
                : (std::popcount(mask) & 1u) ^ (index_parity ? (index >> 7u) & 1u : 0u);
            result.dense[row * input + column] = negative ? -value : value;
        }
    }
    append_bytes(result.blob, pack_vq_values(states, 4));
    append_bytes(result.blob, pack_vq_values(indices, 8));
    append_bytes(result.blob, pack_vq_values(masks, 7));
    return result;
}

VqFixture make_jsc_nvq(
    int output,
    int input,
    std::string dtype = "NVQ2J",
    int codebook_id = 1,
    int vector_size = 8,
    int index_bits = 8,
    bool group64 = false,
    int code_banks = 2,
    int scale_exponent = 0) {
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
    append<std::uint8_t>(blob, group64 ? 2 : 1);
    append<std::uint8_t>(blob, code_banks);
    append<std::uint8_t>(blob, 16);
    append<std::uint8_t>(blob, 0);
    for (int state = 0; state < 16; ++state) {
        append<std::uint16_t>(blob, 0x3c00 + scale_exponent * 1024);
    }
    for (int state = 0; state < 16; ++state) {
        append<std::uint8_t>(
            blob,
            static_cast<std::uint8_t>(
                state % code_banks));
    }
    append<std::uint8_t>(blob, group64 ? 1 : 0);
    blob.insert(blob.end(), 11, 0);
    const auto code_value = [](
        int bank,
        int entry,
        int component) {
        return static_cast<std::int8_t>(
            (entry * 3 + component * 5 + bank * 7) % 15 - 7);
    };
    for (int bank = 0; bank < code_banks; ++bank) {
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
                    code_value(bank, entry, component));
            }
        }
    }
    std::vector<float> dense;
    if (group64) {
        const int groups = (input + 23) / 24;
        append_vq_anchors(blob, output);
        dense.resize(
            static_cast<std::size_t>(output) * input);
        for (int row = 0; row < output; ++row) {
            for (int group = 0; group < groups; ++group) {
                const auto state = static_cast<std::uint64_t>(
                    (row + group) & 15);
                std::uint64_t record = state << 60;
                for (int chunk = 0; chunk < 3; ++chunk) {
                    if (group * 3 + chunk >= (input + 7) / 8) {
                        continue;
                    }
                    const auto index = static_cast<std::uint32_t>(
                        (row * 17 + group * 11 + chunk * 3)
                        & ((1 << index_bits) - 1));
                    const auto mask7 = static_cast<std::uint32_t>(
                        (row * 13 + group * 7 + chunk * 5) & 0x7f);
                    const auto sign_bits = mask7
                        | ((std::popcount(mask7) & 1u) << 7u);
                    const auto segment = static_cast<std::uint64_t>(
                        index | (sign_bits << 12u));
                    record |= segment << (chunk * 20);
                    for (int component = 0; component < 8; ++component) {
                        const int column = group * 24
                            + chunk * 8 + component;
                        if (column >= input) {
                            continue;
                        }
                        float value = static_cast<float>(
                            code_value(
                                static_cast<int>(state % code_banks),
                                static_cast<int>(index),
                                component));
                        if ((sign_bits & (1u << component)) != 0u) {
                            value = -value;
                        }
                        value = std::ldexp(value, scale_exponent);
                        dense[
                            static_cast<std::size_t>(row) * input
                                + column
                        ] = value;
                    }
                }
                append<std::uint64_t>(blob, record);
            }
        }
    } else {
        const int groups = (input + 23) / 24;
        const int vectors = (input + vector_size - 1) / vector_size;
        const int signs = (input + 7) / 8;
        std::vector<std::uint16_t> states(output * groups);
        std::vector<std::uint16_t> indices(output * vectors);
        std::vector<std::uint16_t> masks(output * signs);
        append_vq_anchors(blob, output);
        dense.resize(static_cast<std::size_t>(output) * input);
        for (int row = 0; row < output; ++row) {
            for (int group = 0; group < groups; ++group) {
                states[row * groups + group] = (row + group * 3) & 15;
            }
            for (int vector = 0; vector < vectors; ++vector) {
                indices[row * vectors + vector] =
                    (row * 137 + vector * 59 + 257) & ((1 << index_bits) - 1);
            }
            for (int sign = 0; sign < signs; ++sign) {
                masks[row * signs + sign] = (row * 13 + sign * 7 + 21) & 127;
            }
            for (int column = 0; column < input; ++column) {
                const auto mask7 = static_cast<std::uint32_t>(
                    masks[row * signs + column / 8]);
                const auto sign_bits = mask7
                    | ((std::popcount(mask7) & 1u) << 7u);
                float value = static_cast<float>(code_value(
                    states[row * groups + column / 24] % code_banks,
                    indices[row * vectors + column / vector_size],
                    column % vector_size));
                if ((sign_bits & (1u << (column % 8))) != 0u) {
                    value = -value;
                }
                dense[static_cast<std::size_t>(row) * input + column] =
                    std::ldexp(value, scale_exponent);
            }
        }
        append_bytes(blob, pack_vq_values(states, 4));
        append_bytes(blob, pack_vq_values(indices, index_bits));
        append_bytes(blob, pack_vq_values(masks, 7));
    }
    return {
        std::move(dtype),
        std::move(blob),
        {},
        std::move(dense),
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

VqFixture make_nvq1_l(
    int output,
    int input) {
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        "NQ1L",
        1,
        3,
        24,
        output,
        input);
    append_vq_matrix_streams(
        blob,
        output,
        input,
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
        repeated_vq_dense(output, input, vector),
        {},
        0,
        0,
        output,
        input,
    };
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
    for (int bank = 0; bank < 2; ++bank) {
        for (int entry = 0; entry < 512; ++entry) {
            const int source = bank == 0 ? entry : 511 - entry;
            append<std::uint16_t>(
                blob, mfq::nvq_codebooks::kNvq1LCodebookPacked[source * 4]);
        }
    }
    const int groups = (input + 23) / 24;
    const int vectors = (input + 7) / 8;
    std::vector<std::uint16_t> states(output * groups);
    std::vector<std::uint16_t> indices(output * vectors);
    std::vector<std::uint16_t> deltas(output * groups);
    std::vector<float> dense(static_cast<std::size_t>(output) * input);
    append_vq_anchors(blob, output);
    for (int row = 0; row < output; ++row) {
        for (int group = 0; group < groups; ++group) {
            states[row * groups + group] = (row + group * 3) & 15;
            deltas[row * groups + group] = (row + group) & 1;
        }
        for (int vector = 0; vector < vectors; ++vector) {
            indices[row * vectors + vector] =
                (row * 137 + vector * 59 + 257) & 511;
        }
        for (int column = 0; column < input; ++column) {
            const int group = row * groups + column / 24;
            const auto index = indices[row * vectors + column / 8];
            const auto source = deltas[group] ? 511 - index : index;
            const auto vector = decode_ternary_word(
                mfq::nvq_codebooks::kNvq1LCodebookPacked[source * 4]);
            dense[static_cast<std::size_t>(row) * input + column] =
                (vector[column % 8] + (deltas[group] ? -0.15625f : 0.15625f))
                * states[group];
        }
    }
    append_bytes(blob, pack_vq_values(states, 4));
    append_bytes(blob, pack_vq_values(indices, 9));
    append_bytes(blob, pack_vq_values(deltas, 1));
    return {
        "NVQ1-S",
        std::move(blob),
        {},
        std::move(dense),
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
    int input,
    bool short_profile = false) {
    std::vector<std::uint8_t> blob;
    append_vq_matrix_header(
        blob,
        short_profile ? "NPQS" : "NPQL",
        static_cast<std::uint8_t>(short_profile ? 2 : 1),
        static_cast<std::uint8_t>(short_profile ? 2 : 3),
        24,
        output,
        input);
    append_bytes(blob, npq_table(short_profile, 0));
    append_vq_matrix_streams(
        blob,
        output,
        input,
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

VqFixture make_rotated_nepq0_s(
    int output,
    int input,
    std::uint64_t seed = 0x1020304050607080ull) {
    constexpr int table_banks = 2;
    constexpr int rotation_block = 8;
    if (input % 8 != 0) {
        throw std::runtime_error("invalid NEPQ0-S fixture width");
    }
    std::vector<std::uint8_t> blob;
    append_magic(blob, "NEP1");
    append<std::uint8_t>(blob, 1);
    append<std::uint8_t>(blob, 0);
    append<std::uint8_t>(blob, 4);
    append<std::uint8_t>(blob, 1);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, table_banks);
    append<std::uint32_t>(blob, rotation_block);
    append<std::uint64_t>(blob, seed);
    for (int bank = 0; bank < table_banks; ++bank) {
        append_bytes(blob, npq_table(true, bank));
    }
    append_vq_anchors(blob, output);
    const int groups = (input + 23) / 24;
    const int vectors = input / 8;
    const int supergroups = (groups + 3) / 4;
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(
                static_cast<std::size_t>(output) * groups,
                1),
            2));
    append_bytes(
        blob,
        pack_vq_values(
            filled_vq_values(
                static_cast<std::size_t>(output) * vectors,
                0),
            6));
    for (int row = 0; row < output; ++row) {
        for (int group = 0; group < supergroups; ++group) {
            append<std::uint8_t>(
                blob,
                static_cast<std::uint8_t>(row & 1));
        }
    }
    std::vector<float> dense(
        static_cast<std::size_t>(output) * input);
    for (int row = 0; row < output; ++row) {
        const float first = static_cast<float>(1 + (row & 1) * 2);
        const float second = first + 1.0f;
        for (int column = 0; column < input; ++column) {
            dense[static_cast<std::size_t>(row) * input + column] =
                (column & 7) < 4 ? first : second;
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
            column % 3 == 0 ? -1 : 1);
        append<std::int8_t>(runtime, signs[column]);
    }
    return {
        "NEPQ0-S",
        std::move(blob),
        std::move(runtime),
        std::move(dense),
        std::move(signs),
        rotation_block,
        seed,
        output,
        input,
    };
}

VqFixture add_nepq_a_residual(
    VqFixture fixture,
    bool second_stream) {
    constexpr int dictionary_entries = 1024;
    const int position_bits = second_stream ? 4 : 5;
    const int record_bits = position_bits + 10;
    const int block_vectors = second_stream ? 16 : 24;
    const int vectors = fixture.input / 8;
    const int blocks_per_row =
        (vectors + block_vectors - 1) / block_vectors;
    const int block_count = fixture.output * blocks_per_row;
    fixture.dtype = second_stream ? "NEPQ1-A" : "NEPQ0-A";
    fixture.blob[5] = static_cast<std::uint8_t>(
        second_stream ? 5 : 4);

    append_magic(fixture.blob, "NRA1");
    append<std::uint8_t>(fixture.blob, 1);
    append<std::uint8_t>(fixture.blob, record_bits);
    append<std::uint8_t>(fixture.blob, position_bits);
    append<std::uint8_t>(fixture.blob, second_stream ? 1 : 0);
    append<std::uint32_t>(fixture.blob, dictionary_entries);
    append<std::uint32_t>(fixture.blob, block_count);
    append<std::uint32_t>(
        fixture.blob,
        second_stream ? block_count / 2 : 0);
    append<std::uint32_t>(fixture.blob, 0);
    append<std::uint64_t>(fixture.blob, 0);
    fixture.blob.insert(fixture.blob.end(), 32, 0);
    for (int dictionary = 0;
         dictionary < dictionary_entries;
         ++dictionary) {
        for (int component = 0; component < 8; ++component) {
            append<std::uint16_t>(
                fixture.blob,
                dictionary == 0
                    ? 0x3800u
                    : (dictionary == 1 ? 0xb400u : 0u));
        }
    }
    std::vector<std::uint16_t> first(block_count);
    for (int row = 0; row < fixture.output; ++row) {
        for (int block = 0; block < blocks_per_row; ++block) {
            const int available = std::min(
                block_vectors,
                vectors - block * block_vectors);
            const int position = row % available;
            first[row * blocks_per_row + block] =
                static_cast<std::uint16_t>(position);
            for (int component = 0; component < 8; ++component) {
                fixture.dense[
                    static_cast<std::size_t>(row) * fixture.input
                    + (block * block_vectors + position) * 8
                    + component
                ] += 0.5f;
            }
        }
    }
    append_bytes(fixture.blob, pack_vq_values(first, record_bits));
    if (second_stream) {
        std::vector<std::uint16_t> mask(block_count, 0);
        std::vector<std::uint16_t> second;
        for (int block = 1; block < block_count; block += 2) {
            const int row = block / blocks_per_row;
            const int local_block = block % blocks_per_row;
            const int available = std::min(
                block_vectors,
                vectors - local_block * block_vectors);
            const int position = (row + 1) % available;
            mask[block] = 1;
            second.push_back(static_cast<std::uint16_t>(
                (1 << position_bits) | position));
            for (int component = 0; component < 8; ++component) {
                fixture.dense[
                    static_cast<std::size_t>(row) * fixture.input
                    + (local_block * block_vectors + position) * 8
                    + component
                ] -= 0.25f;
            }
        }
        append_bytes(fixture.blob, pack_vq_values(mask, 1));
        append_bytes(fixture.blob, pack_vq_values(second, record_bits));
    }
    return fixture;
}

struct TensorFixture {
    std::vector<std::uint8_t> blob;
    std::vector<float> dense;
    int rows = 0;
    int columns = 0;
};

TensorFixture make_dense_tensor(
    std::string_view dtype,
    int rows,
    int columns,
    int salt) {
    constexpr std::array<float, 9> values{
        0.0f, 0.25f, -0.25f, 0.5f, -0.5f,
        1.0f, -1.0f, 2.0f, -2.0f,
    };
    constexpr std::array<std::uint16_t, 9> f16_bits{
        0x0000, 0x3400, 0xb400, 0x3800, 0xb800,
        0x3c00, 0xbc00, 0x4000, 0xc000,
    };
    constexpr std::array<std::uint16_t, 9> bf16_bits{
        0x0000, 0x3e80, 0xbe80, 0x3f00, 0xbf00,
        0x3f80, 0xbf80, 0x4000, 0xc000,
    };
    const bool bf16 = dtype == "BF16";
    if ((!bf16 && dtype != "F16") || rows <= 0 || columns <= 0) {
        throw std::runtime_error("invalid dense expert fixture");
    }
    std::vector<std::uint8_t> blob;
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, rows);
    append<std::int64_t>(blob, columns);
    std::vector<float> dense(static_cast<std::size_t>(rows) * columns);
    for (std::size_t index = 0; index < dense.size(); ++index) {
        const auto selected = (index * 5 + static_cast<std::size_t>(salt))
            % values.size();
        dense[index] = values[selected];
        append<std::uint16_t>(
            blob,
            bf16 ? bf16_bits[selected] : f16_bits[selected]);
    }
    return {
        std::move(blob),
        std::move(dense),
        rows,
        columns,
    };
}

TensorFixture make_nint_tensor(
    int bits,
    int rows,
    int columns,
    int salt,
    int group_size = kGroupSize) {
    if (group_size <= 0 || columns <= 0) {
        throw std::runtime_error("test NINT geometry is invalid");
    }
    const int groups = (columns + group_size - 1) / group_size;
    const auto metadata_count =
        static_cast<std::size_t>(rows) * groups;
    const auto value_count =
        metadata_count * group_size;
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
        group_size);
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
                column / group_size;
            const auto metadata =
                static_cast<std::size_t>(row)
                    * groups
                + group;
            const auto value =
                metadata * group_size
                + column % group_size;
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

TensorFixture make_nint_v2_tensor(
    int rows,
    int columns,
    int salt,
    int group_size = kGroupSize,
    bool all_q_widths = false,
    int fixed_q = 0) {
    const int nominal_bits = fixed_q != 0 ? fixed_q : 4;
    constexpr int nominal_sub_bits = 6;
    const int groups = (columns + group_size - 1) / group_size;
    const int values_per_row = groups * group_size;
    std::vector<std::uint8_t> row_q_bits(rows);
    std::vector<std::uint8_t> row_sub_bits(rows);
    std::vector<std::uint8_t> q_selectors(rows);
    std::vector<std::uint8_t> sub_selectors(rows);
    std::vector<std::uint8_t> quantized(
        static_cast<std::size_t>(rows) * values_per_row);
    std::vector<std::uint8_t> sub_scale(
        static_cast<std::size_t>(rows) * groups);
    std::vector<std::uint8_t> sub_min(sub_scale.size());
    for (int row = 0; row < rows; ++row) {
        const int q_bits = fixed_q != 0 ? fixed_q : all_q_widths
            ? 1 + (row + salt) % 8
            : 2 + (row + salt) % 5;
        const int sub_bits = 5 + (row + salt) % 4;
        row_q_bits[row] = static_cast<std::uint8_t>(q_bits);
        row_sub_bits[row] = static_cast<std::uint8_t>(sub_bits);
        q_selectors[row] = static_cast<std::uint8_t>(q_bits - 1);
        sub_selectors[row] = static_cast<std::uint8_t>(sub_bits - 5);
        const int q_maximum = (1 << q_bits) - 1;
        for (int group = 0; group < groups; ++group) {
            const auto metadata = static_cast<std::size_t>(row) * groups + group;
            sub_scale[metadata] = static_cast<std::uint8_t>(
                1 + (row * 3 + group + salt) % 7);
            sub_min[metadata] = static_cast<std::uint8_t>(
                (row + group + salt) & 1);
            for (int element = 0; element < group_size; ++element) {
                const auto value = static_cast<std::size_t>(row) * values_per_row
                    + group * group_size + element;
                quantized[value] = static_cast<std::uint8_t>(
                    (row * 11 + group * 5 + element * 3 + salt) & q_maximum);
            }
        }
    }

    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(blob, 0x80u | nominal_bits);
    append<std::uint8_t>(blob, nominal_sub_bits);
    append<std::int32_t>(blob, group_size);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, columns);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, rows);
    append<std::int64_t>(blob, columns);
    append<std::uint32_t>(blob, static_cast<std::uint32_t>(rows));
    append<std::uint32_t>(blob, static_cast<std::uint32_t>(groups));
    for (int row = 0; row < rows; ++row) {
        append<std::uint16_t>(blob, 0x2400);
    }
    for (int row = 0; row < rows; ++row) {
        append<std::uint16_t>(blob, 0x2000);
    }
    append_bytes(blob, pack_values(sub_selectors, 2));
    for (int selector = 0; selector < 4; ++selector) {
        std::vector<std::uint8_t> scales;
        std::vector<std::uint8_t> minima;
        for (int row = 0; row < rows; ++row) {
            if (sub_selectors[row] != selector) continue;
            const auto begin = static_cast<std::size_t>(row) * groups;
            scales.insert(scales.end(), sub_scale.begin() + begin,
                sub_scale.begin() + begin + groups);
            minima.insert(minima.end(), sub_min.begin() + begin,
                sub_min.begin() + begin + groups);
        }
        append_bytes(blob, pack_values(scales, nominal_sub_bits - 1 + selector));
        append_bytes(blob, pack_values(minima, nominal_sub_bits - 1 + selector));
    }
    append_bytes(blob, pack_values(q_selectors, 3));
    for (int selector = 0; selector < 8; ++selector) {
        std::vector<std::uint8_t> cohort;
        for (int row = 0; row < rows; ++row) {
            if (q_selectors[row] != selector) continue;
            const auto begin = static_cast<std::size_t>(row) * values_per_row;
            cohort.insert(cohort.end(), quantized.begin() + begin,
                quantized.begin() + begin + values_per_row);
        }
        append_bytes(blob, pack_values(cohort, selector + 1));
    }

    std::vector<float> dense(static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const int group = column / group_size;
            const auto metadata = static_cast<std::size_t>(row) * groups + group;
            const auto value = static_cast<std::size_t>(row) * values_per_row
                + group * group_size + column % group_size;
            dense[static_cast<std::size_t>(row) * columns + column] =
                static_cast<float>(sub_scale[metadata]) * (1.0f / 64.0f)
                    * static_cast<float>(quantized[value])
                - static_cast<float>(sub_min[metadata]) * (1.0f / 128.0f);
        }
    }
    return {std::move(blob), std::move(dense), rows, columns};
}

std::vector<std::uint8_t> nint_expert_blob(
    const TensorFixture& tensor, int experts, int output) {
    if (tensor.rows != experts * output)
        throw std::runtime_error("NINT expert fixture row mismatch");
    std::vector<std::uint8_t> blob(tensor.blob.begin(), tensor.blob.begin() + 14);
    append<std::uint32_t>(blob, 3);
    append<std::int64_t>(blob, experts);
    append<std::int64_t>(blob, output);
    append<std::int64_t>(blob, tensor.columns);
    blob.insert(blob.end(), tensor.blob.begin() + 34, tensor.blob.end());
    return blob;
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
    int salt,
    int nint_group_size = kGroupSize) {
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
        TensorFixture tensor = [&] {
            if (profile == "NINT8-0") {
                return make_q8_tensor(output, input, salt + expert);
            }
            if (profile == "F16" || profile == "BF16") {
                return make_dense_tensor(
                    profile, output, input, salt + expert);
            }
            if (profile == "NINTv2") {
                return make_nint_v2_tensor(
                    output,
                    input,
                    salt + expert,
                    nint_group_size);
            }
            return make_nint_tensor(
                std::stoi(profile.substr(4)),
                output,
                input,
                salt + expert,
                nint_group_size);
        }();
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
    float tolerance = 8e-4f,
    std::source_location location = std::source_location::current()) {
    if (
        !std::isfinite(actual)
        || std::fabs(actual - expected) > tolerance
    ) {
        throw std::runtime_error(
            "MFE Metal result mismatch: actual="
            + std::to_string(actual)
            + " expected="
            + std::to_string(expected) + " at " + location.function_name()
            + ":" + std::to_string(location.line()));
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
    value = mlx::core::contiguous(mlx::core::astype(
        value,
        mlx::core::float32));
    value.eval();
    return {
        value.data<float>(),
        value.data<float>() + value.size(),
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
            records, std::string_view architecture = "streamed-mfe-test")
        {
        auto filename = (std::filesystem::temp_directory_path() / "mfq-metal-streamed-mfe-test-XXXXXX").string();
        const auto fd = mkstemp(filename.data());
        if (fd < 0) throw std::runtime_error("cannot create temporary MFE container");
        close(fd);
        path_ = filename;
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
                "cannot create streamed MFE "
                "test container");
        }
        stream.write(
            reinterpret_cast<const char*>(
                file.data()),
            static_cast<std::streamsize>(
                file.size()));
        if (!stream) {
            throw std::runtime_error(
                "cannot write streamed MFE "
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

void test_all_families_and_projections(int tokens = 3) {
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
        "NINTv2",
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
    // The weighted-reduce kernel reuses the generic per-descriptor decoder, so
    // mixed NINT widths take the fused Down path whenever the native primitive
    // is enabled (checked numerically below).
    const char* native_env = std::getenv("MFQ_METAL_MFE_NATIVE_PRIMITIVE");
    const bool native_primitive = native_env == nullptr || std::string_view(native_env) != "0";
    require(first_weight.supports_fused_routed_reduce() == native_primitive,
            "mixed NINT Down fused weighted reduction capability mismatch");
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
        "MFE shape metadata mismatch");

    const std::vector<std::int32_t> id_pattern{
        0, 1, 4, 8,
        7, 5, -1, 10,
        2, 3, 6, 9,
    };
    std::vector<std::int32_t> ids(tokens * routes);
    for (std::size_t index = 0; index < ids.size(); ++index) {
        ids[index] = id_pattern[index % id_pattern.size()];
    }
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
    std::vector<float> route_weights(tokens * routes);
    for (std::size_t index = 0; index < route_weights.size(); ++index)
        route_weights[index] = 0.125f * static_cast<float>(index % 5 + 1);
    const auto routed_source = mlx::core::array(
        routed_input.begin(), mlx::core::Shape{tokens, routes, input_width});
    const auto routed_ids = mlx::core::array(ids.begin(), mlx::core::Shape{tokens, routes});
    const auto weight_values = mlx::core::array(
        route_weights.begin(), mlx::core::Shape{tokens, routes});
    const auto fused_reduce =
        first_weight.routed_matmul_reduce(routed_source, routed_ids, weight_values);
    if (native_primitive) {
        const auto* kernel =
            dynamic_cast<const mfq::metal::MlxPreparableKernel*>(&fused_reduce.primitive());
        require(kernel != nullptr && kernel->preparation_key().find("_reduce_v1") != std::string::npos,
                "mixed NINT Down reduction did not dispatch the fused kernel");
    }
    const auto reduced = evaluated_floats(fused_reduce);
    require(reduced.size() == static_cast<std::size_t>(tokens * output_width),
            "mixed NINT Down reduction shape mismatch");
    for (int token = 0; token < tokens; ++token) {
        for (int output = 0; output < output_width; ++output) {
            float expected = 0.0f;
            for (int route = 0; route < routes; ++route) {
                const int index = token * routes + route;
                if (ids[index] >= 0 && ids[index] < first.experts)
                    expected += route_weights[index] * dot(routed_input,
                        static_cast<std::size_t>(index) * input_width, first, ids[index], output);
            }
            require_close(reduced[token * output_width + output], expected);
        }
    }
    // Model decode runs FP16; the fused result must track the unfused fallback.
    const auto half_source = mlx::core::astype(routed_source, mlx::core::float16);
    const auto fused_half = evaluated_floats(
        first_weight.routed_matmul_reduce(half_source, routed_ids, weight_values));
    const auto unfused_half = evaluated_floats(mfq::metal::moe_weighted_reduce(
        first_weight.routed_matmul(half_source, routed_ids), weight_values));
    require(fused_half.size() == unfused_half.size(),
            "mixed NINT FP16 Down reduction shape mismatch");
    for (std::size_t index = 0; index < fused_half.size(); ++index)
        require_close(fused_half[index], unfused_half[index], 2e-3f);
    const auto grouped =
        mfq::metal::MlxMoeWeight::
            concatenate_projections(
                {first_weight, second_weight});
    require(
        grouped.projections() == 2,
        "MFE grouped projection count mismatch");
    require(
        grouped.packed_nbytes()
            == first_weight.packed_nbytes()
                + second_weight.packed_nbytes(),
        "split shared-kernel projection byte accounting mismatch");
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
    const auto input_array = mlx::core::array(
        input.begin(),
        mlx::core::Shape{tokens, hidden});
    const auto ids_array = mlx::core::array(
        ids.begin(),
        mlx::core::Shape{tokens, routes});
    const auto activated_actual = evaluated_floats(
        ffn.gate_up_weight().routed_swiglu(
            input_array,
            ids_array));
    const std::vector<float> single_input(
        input.begin(),
        input.begin() + hidden);
    const std::vector<std::int32_t> single_ids(
        ids.begin(),
        ids.begin() + routes);
    const auto single_source = mlx::core::array(
        single_input.begin(),
        mlx::core::Shape{1, hidden});
    const auto single_routes = mlx::core::array(
        single_ids.begin(),
        mlx::core::Shape{1, routes});
    const char* prior_split = std::getenv(
        "MFQ_METAL_MFE_SPLIT_SWIGLU");
    const std::optional<std::string> saved_split = prior_split == nullptr
        ? std::nullopt
        : std::optional<std::string>(prior_split);
    setenv("MFQ_METAL_MFE_SPLIT_SWIGLU", "0", 1);
    const auto single_activated_reference = evaluated_floats(
        ffn.gate_up_weight().routed_swiglu(
            single_source,
            single_routes));
    setenv("MFQ_METAL_MFE_SPLIT_SWIGLU", "1", 1);
    const auto single_activated_actual = evaluated_floats(
        ffn.gate_up_weight().routed_swiglu(
            single_source,
            single_routes));
    if (saved_split) {
        setenv(
            "MFQ_METAL_MFE_SPLIT_SWIGLU",
            saved_split->c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_SPLIT_SWIGLU");
    }
    require(
        single_activated_actual.size()
            == single_activated_reference.size(),
        "split Gate/Up SwiGLU output shape mismatch");
    for (std::size_t index = 0;
         index < single_activated_actual.size();
         ++index) {
        require_close(
            single_activated_actual[index],
            single_activated_reference[index],
            2e-3f);
    }
    for (
        std::size_t index = 0;
        index < single_activated_actual.size();
        ++index
    ) {
        require_close(
            single_activated_actual[index],
            activated_actual[index],
            3e-3f);
    }
    const auto actual = evaluated_floats(
        ffn.forward(
            input_array,
            ids_array,
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
                require_close(
                    activated_actual[
                        (token * routes + route)
                            * intermediate
                        + column],
                    activated[column],
                    3e-3f);
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

void test_two_stage_nint_shared_decode(
    int shared_group_size = 24,
    int hidden = 32,
    int intermediate = 24,
    bool nvq3jl = false,
    int tokens = 1,
    bool precomputed_gate = false,
    bool packed_gate_up = false,
    int routed_group_size = kGroupSize,
    bool repeated_routes = false,
    int experts = 32,
    bool check_folded_topk = false,
    int fixed_q = 0,
    int shared_up_group_size = 0,
    bool mixed_expert_q = false,
    bool paged = false) {
    constexpr int routes = 10;
    const std::vector<std::string> profiles(experts, "NINTv2");
    const auto make_fixture = [&](int output, int input, int salt) {
        if (!nvq3jl && routed_group_size != 48 && fixed_q == 0 && !mixed_expert_q)
            return make_moe_fixture(profiles, output, input, salt, routed_group_size);
        std::vector<PoolFixture> pools;
        std::vector<float> dense(experts * output * input);
        for (int expert = experts - 1; expert >= 0; --expert) {
            TensorFixture tensor;
            std::string profile;
            if (nvq3jl && (expert + salt) % 3 != 0) {
                auto vq = make_jsc_nvq(output, input, "NVQ3J-L", 6, 4, 10, false, 4, -1);
                profile = vq.dtype;
                tensor = {std::move(vq.blob), std::move(vq.dense), output, input};
            } else {
                profile = "NINT";
                const int bits = mixed_expert_q ? std::array{4, 5, 6, 8}[expert % 4] : fixed_q;
                tensor = make_nint_v2_tensor(output, input, salt + expert, nvq3jl ? 28 : routed_group_size, true, bits);
            }
            std::copy(tensor.dense.begin(), tensor.dense.end(),
                dense.begin() + expert * output * input);
            pools.push_back({{expert}, profile, std::move(tensor), {}});
        }
        return MoeFixture{make_raw_nim2(experts, output, input, pools),
            std::move(dense), experts, output, input};
    };
    const auto gate_fixture = make_fixture(intermediate, hidden, 7);
    const auto up_fixture = make_fixture(intermediate, hidden, nvq3jl ? 20 : 19);
    const auto down_fixture = make_fixture(hidden, intermediate, nvq3jl ? 33 : 31);
    const std::array<std::span<const std::uint8_t>, 2> gate_up_blobs{
        gate_fixture.blob, up_fixture.blob};
    const auto packed_fixture = packed_gate_up
        ? std::optional<MoeFixture>(make_fixture(intermediate * 2, hidden, 7))
        : std::nullopt;
    const auto pages = [&](mfq::metal::MlxMfeWeight weight) {
        if (!paged) return weight;
        std::vector<mfq::metal::MlxMfeWeight> values;
        for (int expert = 0; expert < experts; ++expert) {
            const std::array id{expert};
            values.push_back(weight.select_experts(id));
        }
        return mfq::metal::MlxMfeWeight::concatenate_experts(values);
    };
    const auto gate_up = packed_fixture
        ? pages(mfq::metal::MlxMfeWeight::from_blob(packed_fixture->blob))
        : paged ? mfq::metal::MlxMfeWeight::concatenate_projections({
            pages(mfq::metal::MlxMfeWeight::from_blob(gate_fixture.blob)),
            pages(mfq::metal::MlxMfeWeight::from_blob(up_fixture.blob))})
        : mfq::metal::MlxMfeWeight::from_projection_blobs(gate_up_blobs);
    const auto down = pages(mfq::metal::MlxMfeWeight::from_blob(down_fixture.blob));

    const auto shared_gate_fixture = make_nint_v2_tensor(
        intermediate, hidden, 43, shared_group_size, true, fixed_q);
    const auto shared_up_fixture = make_nint_v2_tensor(
        intermediate, hidden, 47, shared_up_group_size > 0 ? shared_up_group_size : shared_group_size,
        true, fixed_q);
    const auto shared_down_fixture = make_nint_v2_tensor(
        hidden, intermediate, 53, shared_group_size, true, fixed_q);
    const auto shared_gate = mfq::metal::MlxNintWeight::from_blob(
        shared_gate_fixture.blob);
    const auto shared_up = mfq::metal::MlxNintWeight::from_blob(
        shared_up_fixture.blob);
    const auto shared_down = mfq::metal::MlxNintWeight::from_blob(
        shared_down_fixture.blob);
    const auto shared_gate_up =
        mfq::metal::MlxNintSwiGluPair::from_weights(
            shared_gate, shared_up);
    require(shared_gate_up.has_value(),
            "shared NINT Gate/Up pair was not fuseable");

    std::vector<float> input_values(tokens * hidden);
    std::vector<float> shared_gate_values(hidden);
    for (int token = 0; token < tokens; ++token) {
        for (int column = 0; column < hidden; ++column)
            input_values[token * hidden + column] = static_cast<float>(
                (column * 7 + token * 11) % 23 - 11) / (nvq3jl || routed_group_size != kGroupSize ? 1024.0f : 64.0f);
    }
    for (int column = 0; column < hidden; ++column) {
        shared_gate_values[column] = static_cast<float>(
            (column * 5) % 17 - 8) / 128.0f;
    }
    std::vector<std::int32_t> ids(tokens * routes);
    std::vector<float> route_values(tokens * routes);
    for (int token = 0; token < tokens; ++token) {
        float route_total = 0.0f;
        for (int route = 0; route < routes; ++route) {
            ids[token * routes + route] = ((repeated_routes ? token / 2 : token) * 7 + route) % experts;
            route_values[token * routes + route] = static_cast<float>(route + token + 1);
            route_total += route_values[token * routes + route];
        }
        for (int route = 0; route < routes; ++route)
            route_values[token * routes + route] /= route_total;
    }

    const auto input = mlx::core::astype(
        mlx::core::array(input_values.begin(), mlx::core::Shape{tokens, hidden}),
        mlx::core::float16);
    const auto expert_ids = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto route_weights = mlx::core::array(
        route_values.begin(), mlx::core::Shape{tokens, routes});
    const auto shared_gate_weight = mlx::core::astype(
        mlx::core::array(
            shared_gate_values.begin(), mlx::core::Shape{1, hidden}),
        mlx::core::bfloat16);
    auto gate_logits = mlx::core::matmul(
        mlx::core::astype(input, mlx::core::bfloat16),
        mlx::core::transpose(shared_gate_weight));
    std::optional<mlx::core::array> packed_router;
    if (precomputed_gate) {
        const auto gate_fixture = make_nint_v2_tensor(1, hidden, 67, shared_group_size, true, fixed_q);
        const auto router = mfq::metal::MlxNintWeight::from_blob(gate_fixture.blob);
        gate_logits = router.matmul(input);
        packed_router = mfq::metal::pack_nint_single_row(router);
    }
    const auto gate_input = precomputed_gate ? gate_logits : shared_gate_weight;

    auto fused = gate_up.decode_nint_shared(
        down,
        *shared_gate_up,
        shared_down,
        gate_input,
        input,
        expert_ids,
        route_weights,
        precomputed_gate);
    require(fused.has_value(),
            "two-stage NINT/shared decode path was not selected");
    std::optional<mlx::core::array> mixed_shared_activation;
    if (shared_up_group_size > 0)
        mixed_shared_activation = mlx::core::slice(fused->inputs().at(22),
            {0, routes * intermediate}, {tokens, (routes + 1) * intermediate});
    const auto expected_dispatch = std::string("_rowq_auto")
        + (fixed_q != 0 || mixed_expert_q ? "1" : "0") + (fixed_q != 0 ? "s1" : "s0");
    for (const auto* node : {&*fused, &fused->inputs().at(22)}) {
        const auto* kernel = dynamic_cast<const mfq::metal::MlxPreparableKernel*>(&node->primitive());
        require(kernel != nullptr && kernel->preparation_key().find(expected_dispatch) != std::string::npos,
            "NINT row-q dispatch did not match the actual row precision layout");
    }
    if (packed_router) {
        auto packed = gate_up.decode_nint_shared(down, *shared_gate_up, shared_down,
            *packed_router, input, expert_ids, route_weights);
        require(packed.has_value(), "packed shared router disabled two-stage fusion");
        const auto expected_values = evaluated_floats(*fused);
        const auto packed_values = evaluated_floats(*packed);
        float error = 0.0f, scale = 0.0f;
        for (std::size_t index = 0; index < expected_values.size(); ++index) {
            error = std::max(error, std::abs(expected_values[index] - packed_values[index]));
            scale = std::max(scale, std::abs(expected_values[index]));
        }
        require(error == 0.0f, "packed shared router error=" + std::to_string(error) +
            " scale=" + std::to_string(scale));
    }
    if (check_folded_topk) {
        require(tokens == 1, "folded top-k fixture must have one token");
        for (int fixture = 0; fixture < 6; ++fixture) {
            std::vector<float> logits_values(experts);
            for (int expert = 0; expert < experts; ++expert) {
                float value = std::sin(float(expert * 17 + 3)) * 4.0f;
                if (fixture == 1) value = expert % 2 ? -0.0f : 0.0f;
                if (fixture == 2) value = float(expert % 5) / 1024.0f;
                if (fixture == 3) value = std::numeric_limits<float>::quiet_NaN();
                if (fixture == 4 && expert % 3 == 0)
                    value = std::numeric_limits<float>::quiet_NaN();
                if (fixture == 5) value = expert % 2
                    ? std::numeric_limits<float>::infinity()
                    : -std::numeric_limits<float>::infinity();
                logits_values[expert] = value;
            }
            const auto logits = mlx::core::astype(mlx::core::array(
                logits_values.begin(), mlx::core::Shape{1, experts}), mlx::core::float16);
            const auto selected = mfq::metal::moe_topk(logits, routes, false, false, true);
            auto expected = gate_up.decode_nint_shared(down, *shared_gate_up, shared_down,
                gate_input, input, selected.ids, selected.weights, precomputed_gate);
            auto folded = gate_up.decode_nint_shared(down, *shared_gate_up, shared_down,
                gate_input, input, selected.ids, selected.weights, precomputed_gate, nullptr, &logits);
            require(expected.has_value() && folded.has_value(), "folded routing disabled MFE");
            if (fixture == 0) {
                mfq::metal::MlxKernelPreparation preparation(128, 128);
                preparation.add(*folded);
                mlx::core::synchronize();
                const auto active_memory = mlx::core::get_active_memory();
                preparation.finish();
                require(preparation.unsupported().empty(), "folded routing cannot be prepared");
                require(mlx::core::get_active_memory() == active_memory && !folded->is_available(),
                        "folded routing preparation executed the graph");
            }
            const auto routing = folded->inputs().at(27);
            require(routing.shape() == mlx::core::Shape{2, routes} && routing.dtype() == mlx::core::float32,
                    "folded routing was not selected");
            const auto routing_values = evaluated_floats(routing);
            const auto selected_ids = evaluated_floats(selected.ids);
            const auto selected_weights = evaluated_floats(selected.weights);
            require(std::memcmp(routing_values.data(), selected_ids.data(), routes * sizeof(float)) == 0,
                    "folded routing changed expert IDs");
            require(std::memcmp(routing_values.data() + routes, selected_weights.data(), routes * sizeof(float)) == 0,
                    "folded routing changed normalized weights");
            const auto expected_values = evaluated_floats(*expected);
            const auto folded_values = evaluated_floats(*folded);
            require(std::memcmp(expected_values.data(), folded_values.data(), expected_values.size() * sizeof(float)) == 0,
                    "folded routing changed MoE output");
        }
    }
    if (tokens > 1) {
        std::vector<mlx::core::array> row_outputs;
        for (int row = 0; row < tokens; ++row) {
            auto row_output = gate_up.decode_nint_shared(
                down, *shared_gate_up, shared_down,
                precomputed_gate ? mlx::core::slice(gate_logits, {row, 0}, {row + 1, 1}) : shared_gate_weight,
                mlx::core::slice(input, {row, 0}, {row + 1, hidden}),
                mlx::core::slice(expert_ids, {row, 0}, {row + 1, routes}),
                mlx::core::slice(route_weights, {row, 0}, {row + 1, routes}), precomputed_gate);
            require(row_output.has_value(), "row-wise NINT/shared decode was not selected");
            row_outputs.push_back(*row_output);
        }
        const auto batched_values = evaluated_floats(*fused);
        if (tokens <= 24) {
            const auto route_groups = mfq::metal::moe_route_groups_host(ids, tokens, routes);
            require(route_groups.has_value(), "two-stage route chunks were not constructed");
            auto planned = gate_up.decode_nint_shared(down, *shared_gate_up, shared_down,
                gate_input, input, expert_ids, route_weights, precomputed_gate, &*route_groups);
            require(planned.has_value(), "route groups disabled two-stage fusion");
            const auto planned_values = evaluated_floats(*planned);
            require(batched_values == planned_values,
                    "route groups changed two-stage NINT/shared decode output");
        }
        const auto row_values = evaluated_floats(mlx::core::concatenate(row_outputs, 0));
        require(batched_values == row_values,
                "batched NINT/shared decode differs from identical row-wise routes");
    }

    auto routed_intermediate = gate_up.routed_swiglu(input, expert_ids);
    auto routed_output = down.routed_matmul(
        routed_intermediate, expert_ids);
    // Use independently constructed dense weights: comparing two NINT
    // decoders could hide the same bit-addressing or group-tail error.
    const auto dense_shared_matmul = [](
        const TensorFixture& fixture, const mlx::core::array& activation) {
        return mlx::core::astype(
            mlx::core::matmul(
                mlx::core::astype(activation, mlx::core::float32),
                mlx::core::transpose(mlx::core::array(
                    fixture.dense.begin(),
                    mlx::core::Shape{fixture.rows, fixture.columns}))),
            mlx::core::float16);
    };
    if (nvq3jl || packed_gate_up) {
        const auto dense_routed = [&](const MoeFixture& fixture,
                                      const mlx::core::array& activation) {
            auto weights = mlx::core::take(mlx::core::array(fixture.dense.begin(),
                mlx::core::Shape{experts, fixture.output, fixture.input}), expert_ids, 0);
            auto source = activation.ndim() == 2 ? mlx::core::expand_dims(activation, 1) : activation;
            return mlx::core::astype(mlx::core::sum(weights *
                mlx::core::expand_dims(mlx::core::astype(source, mlx::core::float32), 2), -1),
                mlx::core::float16);
        };
        auto gate = dense_routed(gate_fixture, input);
        auto up = dense_routed(up_fixture, input);
        if (packed_fixture) {
            const auto combined = dense_routed(*packed_fixture, input);
            gate = mlx::core::slice(combined, {0, 0, 0}, {tokens, routes, intermediate});
            up = mlx::core::slice(combined, {0, 0, intermediate}, {tokens, routes, intermediate * 2});
        }
        auto gate_fp32 = mlx::core::astype(gate, mlx::core::float32);
        routed_intermediate = mlx::core::astype(gate_fp32 * mlx::core::sigmoid(gate_fp32)
            * mlx::core::astype(up, mlx::core::float32), mlx::core::float16);
        routed_output = dense_routed(down_fixture, routed_intermediate);
    }
    auto shared_gate_output = dense_shared_matmul(shared_gate_fixture, input);
    auto shared_gate_fp32 = mlx::core::astype(shared_gate_output, mlx::core::float32);
    auto shared_intermediate = mlx::core::astype(
        shared_gate_fp32 * mlx::core::sigmoid(shared_gate_fp32) *
        mlx::core::astype(dense_shared_matmul(shared_up_fixture, input), mlx::core::float32),
        mlx::core::float16);
    auto shared_output = dense_shared_matmul(
        shared_down_fixture, shared_intermediate);
    if (shared_up_group_size > 0) {
        auto gate_fp32 = mlx::core::astype(shared_gate_output, mlx::core::float32);
        auto expected_shared = mlx::core::astype(gate_fp32 * mlx::core::sigmoid(gate_fp32) *
            mlx::core::astype(dense_shared_matmul(shared_up_fixture, input), mlx::core::float32),
            mlx::core::float16);
        const auto actual_shared = evaluated_floats(*mixed_shared_activation);
        const auto expected_values = evaluated_floats(expected_shared);
        float error = 0.0f, scale = 0.0f;
        for (std::size_t index = 0; index < actual_shared.size(); ++index) {
            error = std::max(error, std::abs(actual_shared[index] - expected_values[index]));
            scale = std::max(scale, std::abs(expected_values[index]));
        }
        require(scale > 1e-5f && error <= std::max(5e-7f, 0.005f * scale),
            "mixed-GS shared Gate/Up differs from independent dense weights: error=" +
            std::to_string(error) + " scale=" + std::to_string(scale));
    }
    auto reference = mfq::metal::moe_weighted_reduce_shared_gate(
        routed_output,
        route_weights,
        shared_output,
        gate_logits);
    const auto fused_values = evaluated_floats(*fused);
    const auto reference_values = evaluated_floats(reference);
    require(fused_values.size() == reference_values.size(),
            "two-stage NINT/shared decode output size mismatch");
    for (std::size_t index = 0; index < fused_values.size(); ++index) {
        try {
            require_close(
                fused_values[index],
                reference_values[index],
                3e-2f);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "shared NINT GS=" + std::to_string(shared_group_size)
                + " K=" + std::to_string(hidden)
                + " intermediate=" + std::to_string(intermediate)
                + " output=" + std::to_string(index) + ": " + error.what());
        }
    }
    if (precomputed_gate) {
        require(!gate_up.decode_nint_shared(down, *shared_gate_up, shared_down,
            mlx::core::zeros({tokens + 1}), input, expert_ids, route_weights, true),
            "two-stage decode accepted the wrong precomputed gate count");
        for (const auto dtype : {mlx::core::float32, mlx::core::bfloat16}) {
            auto output = gate_up.decode_nint_shared(down, *shared_gate_up, shared_down,
                mlx::core::astype(gate_logits, dtype), input, expert_ids, route_weights, true);
            require(output.has_value(), "two-stage decode rejected floating-point gate logits");
            const auto values = evaluated_floats(*output);
            for (std::size_t index = 0; index < values.size(); ++index)
                require_close(values[index], reference_values[index], 3e-2f);
        }
    }
}

void test_uniform_nint_multirow_decode() {
    for (const int bits : {4, 5, 6, 8}) {
        for (const int rows : {1, 3, 6}) {
            test_two_stage_nint_shared_decode(24, 128, 64, true, rows,
                false, false, 24, true, 32, false, bits);
            test_two_stage_nint_shared_decode(48, 64, 128, true, rows,
                false, true, 48, true, 32, false, bits);
        }
    }
    for (const int rows : {1, 3, 6})
        test_two_stage_nint_shared_decode(28, 128, 64, true, rows,
            false, false, 28, true, 32, false, 8, 24, true);
}

void test_mxfp4_repeated_routes() {
    using namespace mlx::core;
    for (const auto [input, output] : {std::pair{96, 13}, std::pair{64, 128}}) {
        const auto mx = make_mxfp4(2 * output, input, 3);
        for (const auto& nvq : std::array{
            make_nvq1_s(output, input), make_nvq1_l(output, input), make_jsc_nvq(output, input),
            make_jsc_nvq(output, input, "NVQ2J-L", 4, 8, 10, false, 4),
            make_jsc_nvq(output, input, "NVQ2J-XL", 5, 8, 12, true, 4),
            make_jsc_nvq(output, input, "NVQ3J", 2, 4, 8),
            make_jsc_nvq(output, input, "NVQ3J-512", 3, 4, 9, false, 4),
            make_jsc_nvq(output, input, "NVQ3J-L", 6, 4, 10, false, 4)}) {
        const auto weight = mfq::metal::MlxMfeWeight::from_blob(make_raw_nim2(3, output, input, {
            {{2, 0}, "MXFP4", {mx.blob, mx.dense, 2 * output, input}, {}},
            {{1}, nvq.dtype, {nvq.blob, nvq.dense, output, input}, nvq.runtime},
        }));
        const auto mixed_up = mfq::metal::MlxMfeWeight::from_blob(make_raw_nim2(3, output, input, {
            {{1, 0}, "MXFP4", {mx.blob, mx.dense, 2 * output, input}, {}},
            {{2}, nvq.dtype, {nvq.blob, nvq.dense, output, input}, nvq.runtime},
        }));
        const auto pair = mfq::metal::MlxMfeWeight::concatenate_projections({weight, weight})
            .materialize_packed_projections();
        for (int tokens : {1, 2, 3, 6, 8}) for (bool shared : {true, false}) {
            std::vector<std::int32_t> ids(tokens * 4);
            for (int token = 0; token < tokens; ++token) for (int route = 0; route < 4; ++route)
                ids[token * 4 + route] = route == 3 ? -1 : (route + token) % 3;
            auto routes = array(ids.begin(), Shape{tokens, 4});
            auto groups = mfq::metal::moe_route_groups_host(ids, tokens, 4);
            const auto* grouped = groups ? &*groups : nullptr;
            const Shape shape = shared ? Shape{tokens, input} : Shape{tokens, 4, input};
            int count = tokens * input * (shared ? 1 : 4);
            auto source = contiguous(astype(reshape(sin(arange(count,float32) * 0.17f) * 0.1f,shape),float16));
            if (!shared && tokens <= 6) {
                auto scales = full({tokens, 4}, 0.125f, float32);
                const auto expected = evaluated_floats(mfq::metal::moe_weighted_reduce(
                    weight.routed_matmul(source, routes), scales));
                const auto actual = evaluated_floats(weight.routed_matmul_reduce(source, routes, scales));
                require(expected == actual, "mixed NVQ/MX Down fusion changed rounded values");
                const auto grouped_expected = evaluated_floats(mfq::metal::moe_weighted_reduce(
                    weight.routed_matmul(source, routes, grouped), scales));
                const auto grouped_actual = evaluated_floats(weight.routed_matmul_reduce(source, routes, scales, grouped));
                require(grouped_expected == grouped_actual, "Down fusion discarded repeated-route reuse");
            }
            for (bool activated : {false, true}) {
                std::vector<array> reference;
                for (int token = 0; token < tokens; ++token) {
                    Shape begin(source.ndim(),0),end=source.shape();begin[0]=token;end[0]=token+1;
                    auto x = slice(source,begin,end);
                    auto id = slice(routes,{token,0},{token+1,4});
                    reference.push_back(activated ? pair.routed_swiglu(x,id,0.05f) : weight.routed_matmul(x,id));
                }
                const auto expected = evaluated_floats(concatenate(reference,0));
                if (activated && shared && tokens <= 6) {
                    const auto calls = mfq::metal::MlxMfeWeight::indirect_dispatch_count();
                    const auto actual = evaluated_floats(weight.routed_swiglu_pair(
                        weight, source, routes, 0.05f, grouped));
                    for (std::size_t i = 0; i < actual.size(); ++i) require_close(actual[i], expected[i], 1e-6f);
                    if (__builtin_available(macOS 15, *)) {
                        if (mlx::core::metal::device(mlx::core::Device::gpu).mtl_device()->supportsFamily(MTL::GPUFamilyMetal3))
                            require(mfq::metal::MlxMfeWeight::indirect_dispatch_count() == calls + 1,
                                "mixed MXFP4/NVQ Gate/Up did not use one indirect fused dispatch");
                    }
                    const auto mixed_expected = evaluated_floats(mfq::metal::moe_limited_swiglu_pair(
                        weight.routed_matmul(source, routes, grouped),
                        mixed_up.routed_matmul(source, routes, grouped), 0.05f));
                    for (const auto* group : std::array<const array*, 2>{nullptr, grouped}) {
                        const auto mixed_actual = evaluated_floats(weight.routed_swiglu_pair(
                            mixed_up, source, routes, 0.05f, group));
                        for (std::size_t i = 0; i < mixed_actual.size(); ++i)
                            require_close(mixed_actual[i], mixed_expected[i], 1e-6f);
                    }
                }
                for (const array* group : std::array<const array*,2>{nullptr, grouped}) {
                    auto actual = activated ? pair.routed_swiglu(source,routes,0.05f,group)
                        : weight.routed_matmul(source,routes,group);
                    const auto values = evaluated_floats(actual);
                    require(values.size()==expected.size(),"MXFP4 repeated route output shape");
                    for (std::size_t i=0;i<values.size();++i)
                        require_close(values[i],expected[i],1e-6f);
                }
            }
        }
    }
}
}

void test_mxfp4_mfe_and_projection_offsets() {
    constexpr int experts = 2;
    constexpr int output = 7;
    constexpr int input = 64;
    constexpr int tokens = 2;
    constexpr int routes = 2;
    const auto fixture = make_mxfp4(experts * output, input);
    std::vector<std::uint8_t> blob;
    append_magic(blob, "NIM2");
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, 5);
    append<std::uint64_t>(blob, fixture.blob.size());
    append<std::uint64_t>(blob, 0);
    // Reverse global IDs so descriptor local-row mapping is independently
    // checked rather than accidentally matching global expert order.
    append<std::int32_t>(blob, 1);
    append<std::int32_t>(blob, 0);
    blob.insert(blob.end(), {'M', 'X', 'F', 'P', '4'});
    append_bytes(blob, fixture.blob);

    const auto weight =
        mfq::metal::MlxMoeWeight::from_blob(blob);
    require(
        weight.experts() == experts
            && weight.out_per_expert() == output
            && weight.neuron_len() == input
            && weight.packed_nbytes() >= fixture.blob.size(),
        "MXFP4 MFE metadata mismatch");
    std::vector<float> source(tokens * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(
            static_cast<int>((index * 13 + 5) % 29) - 14) / 128.0f;
    }
    const std::vector<std::int32_t> ids{0, 1, 1, 0};
    const auto input_array = mlx::core::array(
        source.begin(),
        mlx::core::Shape{tokens, input});
    const auto id_array = mlx::core::array(
        ids.begin(),
        mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, id_array));
    require(
        weight.supports_grouped_mmq(),
        "MXFP4 MFE must support grouped matrix prefill");
    const std::vector<std::int32_t> order{0, 3, 1, 2};
    const auto order_array = mlx::core::array(
        order.begin(),
        mlx::core::Shape{tokens * routes});
    const auto plan = weight.build_grouped_mmq_plan(
        id_array,
        order_array);
    const auto matrix_actual = evaluated_floats(
        mlx::core::take(
            weight.routed_matmul_sorted(
                input_array,
                id_array,
                order_array,
                false,
                false,
                0.0f,
                &plan),
            mlx::core::argsort(order_array),
            0));
    const auto concatenated =
        mfq::metal::MlxMoeWeight::concatenate_projections(
            {weight, weight});
    const auto projected = evaluated_floats(
        concatenated.routed_matmul(input_array, id_array));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            const int local_expert = expert == 1 ? 0 : 1;
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                const auto dense_offset = (
                    static_cast<std::size_t>(local_expert) * output + row
                ) * input;
                for (int column = 0; column < input; ++column) {
                    expected += source[token * input + column]
                        * fixture.dense[dense_offset + column];
                }
                const auto route_offset =
                    static_cast<std::size_t>(token * routes + route);
                require_close(
                    actual[route_offset * output + row],
                    expected,
                    2e-4f);
                require_close(
                    matrix_actual[route_offset * output + row],
                    actual[route_offset * output + row],
                    4e-3f);
                require_close(
                    projected[route_offset * (2 * output) + row],
                    expected,
                    2e-4f);
                require_close(
                    projected[
                        route_offset * (2 * output) + output + row
                    ],
                    expected,
                    2e-4f);
            }
        }
    }
}

void test_mxfp4_multi_pool_native_slots() {
    constexpr int experts = 2;
    constexpr int output = 7;
    constexpr int input = 64;
    constexpr int tokens = 2;
    const std::array<Mxfp4Fixture, experts> fixtures{
        make_mxfp4(output, input, 0),
        make_mxfp4(output, input, 7),
    };
    std::vector<std::uint8_t> blob;
    append_magic(blob, "MFE1");
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, experts);
    for (int expert = 0; expert < experts; ++expert) {
        append<std::uint32_t>(blob, 1);
        append<std::uint32_t>(blob, 5);
        append<std::uint64_t>(blob, fixtures[expert].blob.size());
        append<std::uint64_t>(blob, 0);
        append<std::int32_t>(blob, expert);
        blob.insert(blob.end(), {'M', 'X', 'F', 'P', '4'});
        append_bytes(blob, fixtures[expert].blob);
    }

    const auto weight = mfq::metal::MlxMoeWeight::from_blob(blob);
    std::vector<float> source(tokens * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(
            static_cast<int>((index * 13 + 5) % 29) - 14) / 128.0f;
    }
    const auto source_array = mlx::core::astype(
        mlx::core::array(source.begin(), mlx::core::Shape{tokens, input}),
        mlx::core::bfloat16);
    const std::vector<std::int32_t> ids{1, 0};
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, 1});
    const auto route_order = mlx::core::contiguous(
        mlx::core::astype(
            mlx::core::argsort(mlx::core::reshape(
                ids_array,
                mlx::core::Shape{tokens})),
            mlx::core::int32));
    auto sorted = weight.routed_matmul_sorted(
        source_array,
        ids_array,
        route_order,
        false,
        false,
        0.0f,
        nullptr,
        true);
    require(
        sorted.dtype() == mlx::core::bfloat16,
        "multi-pool MXFP4 gather-QMM did not preserve BF16");
    const auto actual = evaluated_floats(mlx::core::take(
        std::move(sorted),
        mlx::core::argsort(route_order),
        0));
    const auto rounded_source = evaluated_floats(source_array);
    for (int token = 0; token < tokens; ++token) {
        const int expert = ids[static_cast<std::size_t>(token)];
        for (int row = 0; row < output; ++row) {
            float expected = 0.0f;
            for (int column = 0; column < input; ++column) {
                expected += rounded_source[
                    static_cast<std::size_t>(token) * input + column]
                    * fixtures[expert].dense[
                        static_cast<std::size_t>(row) * input + column];
            }
            require_close(
                actual[static_cast<std::size_t>(token) * output + row],
                expected,
                2.5e-2f);
        }
    }
}

void test_mxfp4_pair_blocks_matches_native_projections() {
    constexpr int experts = 4;
    constexpr int slots = 4;
    constexpr int output = 32;
    constexpr int input = 64;
    constexpr int tokens = 45;
    constexpr int routes = 1;
    constexpr int route_count = tokens * routes;
    constexpr std::size_t packed_stride =
        static_cast<std::size_t>(output) * input / 2;
    constexpr std::size_t scale_stride =
        static_cast<std::size_t>(output) * input / 32;

    std::vector<std::uint8_t> gate_values(slots * packed_stride);
    std::vector<std::uint8_t> up_values(slots * packed_stride);
    std::vector<std::uint8_t> gate_scales(slots * scale_stride);
    std::vector<std::uint8_t> up_scales(slots * scale_stride);
    for (std::size_t index = 0; index < gate_values.size(); ++index) {
        gate_values[index] = static_cast<std::uint8_t>(
            ((index * 13 + 0x31) & 15u)
            | (((index * 7 + 0x0b) & 15u) << 4));
        up_values[index] = static_cast<std::uint8_t>(
            ((index * 5 + 0x07) & 15u)
            | (((index * 11 + 0x03) & 15u) << 4));
    }
    for (std::size_t index = 0; index < gate_scales.size(); ++index) {
        gate_scales[index] = static_cast<std::uint8_t>(126 + index % 3);
        up_scales[index] = static_cast<std::uint8_t>(125 + index % 4);
    }
    const std::vector<std::int32_t> gate_slots{2, 0, 3, 1};
    const std::vector<std::int32_t> up_slots{1, 3, 0, 2};
    const auto gate = mfq::metal::MlxMoeWeight::from_mxfp4_slots(
        experts,
        output,
        input,
        gate_slots,
        mlx::core::array(
            gate_values.begin(),
            mlx::core::Shape{slots, static_cast<int>(packed_stride)}),
        mlx::core::array(
            gate_scales.begin(),
            mlx::core::Shape{slots, static_cast<int>(scale_stride)}));
    const auto up = mfq::metal::MlxMoeWeight::from_mxfp4_slots(
        experts,
        output,
        input,
        up_slots,
        mlx::core::array(
            up_values.begin(),
            mlx::core::Shape{slots, static_cast<int>(packed_stride)}),
        mlx::core::array(
            up_scales.begin(),
            mlx::core::Shape{slots, static_cast<int>(scale_stride)}));
    require(
        gate.supports_mxfp4_pair_blocks(up),
        "compatible MXFP4 slot views rejected pair blocks");

    std::vector<float> input_values(tokens * input);
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = static_cast<float>(
            static_cast<int>((index * 17 + 9) % 41) - 20) / 256.0f;
    }
    std::vector<std::int32_t> sorted_ids(route_count);
    std::fill(sorted_ids.begin(), sorted_ids.begin() + 35, 0);
    std::fill(sorted_ids.begin() + 35, sorted_ids.begin() + 40, 1);
    std::fill(sorted_ids.begin() + 40, sorted_ids.begin() + 43, 2);
    std::fill(sorted_ids.begin() + 43, sorted_ids.end(), 3);
    std::vector<std::int32_t> ids(route_count);
    for (int index = 0; index < route_count; ++index) {
        ids[static_cast<std::size_t>((index * 17) % route_count)] =
            sorted_ids[static_cast<std::size_t>(index)];
    }
    const auto ids_array = mlx::core::array(
        ids.begin(),
        mlx::core::Shape{tokens, routes});
    const auto route_order = mlx::core::contiguous(
        mlx::core::astype(
            mlx::core::argsort(mlx::core::reshape(
                ids_array,
                mlx::core::Shape{route_count})),
            mlx::core::int32));
    const auto plan = gate.build_grouped_mmq_plan(
        ids_array,
        route_order,
        32);
    require(
        plan.block_rows == 32 && plan.route_count == route_count,
        "MXFP4 pair block plan geometry mismatch");

    const auto check_dtype = [&](mlx::core::Dtype dtype, float tolerance) {
        auto source = mlx::core::astype(
            mlx::core::array(
                input_values.begin(),
                mlx::core::Shape{tokens, input}),
            dtype);
        auto sorted_source = mlx::core::take(source, route_order, 0);
        const auto single_reference = evaluated_floats(
            gate.routed_matmul_sorted(
                sorted_source,
                ids_array,
                route_order,
                true,
                false,
                0.0f,
                &plan,
                true));
        auto single = gate.mxfp4_block_matmul_sorted(
            sorted_source,
            plan);
        require(
            single.dtype() == mlx::core::float16,
            "MXFP4 single block output dtype mismatch");
        const auto single_actual = evaluated_floats(std::move(single));
        require(
            single_actual.size() == single_reference.size(),
            "MXFP4 single block output size mismatch");
        for (std::size_t index = 0; index < single_actual.size(); ++index) {
            require_close(
                single_actual[index],
                single_reference[index],
                tolerance);
        }
        auto gate_output = gate.routed_matmul_sorted(
            sorted_source,
            ids_array,
            route_order,
            true,
            false,
            0.0f,
            &plan,
            true);
        auto up_output = up.routed_matmul_sorted(
            sorted_source,
            ids_array,
            route_order,
            true,
            false,
            0.0f,
            &plan,
            true);
        constexpr float swiglu_limit = 10.0f;
        const auto reference = evaluated_floats(
            mfq::metal::moe_limited_swiglu_pair(
                std::move(gate_output),
                std::move(up_output),
                swiglu_limit));
        auto paired = gate.mxfp4_pair_swiglu_sorted(
            up,
            sorted_source,
            plan,
            swiglu_limit);
        require(
            paired.dtype() == mlx::core::float16,
            "MXFP4 pair block output dtype mismatch");
        const auto actual = evaluated_floats(std::move(paired));
        require(
            actual.size() == reference.size(),
            "MXFP4 pair block output size mismatch");
        for (std::size_t index = 0; index < actual.size(); ++index) {
            require_close(actual[index], reference[index], tolerance);
        }

        for (const int small_tokens : {1, 2, 4, 6}) {
            auto small_source = mlx::core::slice(
                source,
                mlx::core::Shape{0, 0},
                mlx::core::Shape{small_tokens, input});
            const auto small_ids = mlx::core::array(
                ids.begin(),
                mlx::core::Shape{small_tokens, routes});
            const auto small_reference = evaluated_floats(
                mfq::metal::moe_limited_swiglu_pair(
                    gate.routed_matmul(small_source, small_ids),
                    up.routed_matmul(small_source, small_ids),
                    swiglu_limit));
            auto small_paired = gate.routed_swiglu_pair(
                up,
                small_source,
                small_ids,
                swiglu_limit);
            require(
                small_paired.dtype() == mlx::core::float16,
                "MXFP4 pair small-M output dtype mismatch");
            const auto small_actual = evaluated_floats(
                std::move(small_paired));
            require(
                small_actual.size() == small_reference.size(),
                "MXFP4 pair small-M output size mismatch");
            for (std::size_t index = 0;
                 index < small_actual.size();
                 ++index) {
                require_close(
                    small_actual[index],
                    small_reference[index],
                    tolerance);
            }
        }
    };
    check_dtype(mlx::core::float16, 8e-3f);
    check_dtype(mlx::core::bfloat16, 3e-2f);
}

void test_mxfp4_sq_mfe_reuses_linear_kernel() {
    constexpr int experts = 3;
    constexpr int output = 5;
    constexpr int input = 64;
    constexpr int tokens = 2;
    constexpr int routes = 3;
    const auto fixture = make_mxfp4_sq2(experts * output, input);
    std::vector<std::uint8_t> blob;
    append_magic(blob, "MFE1");
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, experts);
    const std::string dtype = "MXFP4-SQ";
    append<std::uint32_t>(blob, dtype.size());
    append<std::uint64_t>(blob, fixture.blob.size());
    append<std::uint64_t>(blob, 0);
    // Cohort-local rows intentionally use a non-global expert order.
    const std::array<std::int32_t, experts> local_to_global{2, 0, 1};
    for (const auto expert : local_to_global) {
        append<std::int32_t>(blob, expert);
    }
    blob.insert(blob.end(), dtype.begin(), dtype.end());
    append_bytes(blob, fixture.blob);

    const auto weight = mfq::metal::MlxMoeWeight::from_blob(blob);
    require(
        weight.experts() == experts &&
            weight.out_per_expert() == output &&
            weight.neuron_len() == input &&
            weight.packed_nbytes() >= fixture.blob.size(),
        "MXFP4-SQ MFE metadata mismatch");
    require(
        !weight.supports_grouped_mmq(),
        "MXFP4-SQ must not create a second heterogeneous grouped kernel");

    std::vector<float> source(static_cast<std::size_t>(tokens) * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] =
            static_cast<float>(static_cast<int>((index * 5 + 1) % 23) - 11) /
            128.0f;
    }
    const std::vector<std::int32_t> ids{0, 1, 2, 2, 0, 1};
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            source.begin(), mlx::core::Shape{tokens, input}),
        mlx::core::float16);
    const auto id_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, id_array));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            const auto found = std::find(
                local_to_global.begin(), local_to_global.end(), expert);
            const int local = static_cast<int>(
                found - local_to_global.begin());
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < input; ++column) {
                    expected +=
                        source[static_cast<std::size_t>(token) * input + column]
                        * fixture.dense[
                            (static_cast<std::size_t>(local) * output + row)
                                * input
                            + column];
                }
                const auto index =
                    (static_cast<std::size_t>(token) * routes + route)
                        * output
                    + row;
                require_close(actual[index], expected, 3e-3f);
            }
        }
    }

    const auto doubled = mfq::metal::MlxMoeWeight::concatenate_projections(
        {weight, weight});
    const auto projected = evaluated_floats(
        doubled.routed_matmul(input_array, id_array));
    for (std::size_t route = 0;
         route < static_cast<std::size_t>(tokens * routes);
         ++route) {
        for (int row = 0; row < output; ++row) {
            require_close(
                projected[route * 2 * output + row],
                actual[route * output + row],
                1e-6f);
            require_close(
                projected[route * 2 * output + output + row],
                actual[route * output + row],
                1e-6f);
        }
    }
}

void test_mxfp4_sq_mixed_mfe_routes_without_sq_heterogeneous_kernel() {
    constexpr int experts = 3;
    constexpr int output = 5;
    constexpr int input = 64;
    constexpr int tokens = 2;
    constexpr int routes = 3;
    auto sq = make_mxfp4_sq2(2 * output, input);
    auto nint = make_nint_tensor(4, output, input, 17, kGroupSize);
    std::vector<PoolFixture> pools;
    pools.push_back({
        {2, 0},
        "MXFP4-SQ2",
        TensorFixture{
            std::move(sq.blob),
            std::move(sq.dense),
            2 * output,
            input,
        },
        {},
    });
    pools.push_back({{1}, "NINT4", std::move(nint), {}});
    const auto blob = make_raw_nim2(experts, output, input, pools);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(blob);

    std::vector<float> source(static_cast<std::size_t>(tokens) * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] =
            static_cast<float>(static_cast<int>((index * 11 + 7) % 37) - 18)
            / 128.0f;
    }
    const std::vector<std::int32_t> ids{0, 1, 2, 2, 1, 0};
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            source.begin(), mlx::core::Shape{tokens, input}),
        mlx::core::float16);
    const auto id_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, id_array));

    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            const PoolFixture* pool = expert == 1 ? &pools[1] : &pools[0];
            const auto found = std::find(
                pool->expert_ids.begin(), pool->expert_ids.end(), expert);
            const int local = static_cast<int>(found - pool->expert_ids.begin());
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < input; ++column) {
                    expected +=
                        source[static_cast<std::size_t>(token) * input + column]
                        * pool->tensor.dense[
                            (static_cast<std::size_t>(local) * output + row)
                                * input
                            + column];
                }
                const auto index =
                    (static_cast<std::size_t>(token) * routes + route)
                        * output
                    + row;
                require_close(actual[index], expected, 8e-3f);
            }
        }
    }
}

void test_mxfp4_smallm_nax_policy() {
    constexpr int experts = 24;
    constexpr int output = 8;
    constexpr int input = 64;
    constexpr int routes = 6;
    std::vector<std::int32_t> slot_for_expert(experts);
    std::iota(slot_for_expert.begin(), slot_for_expert.end(), 0);
    std::vector<std::uint8_t> packed(
        static_cast<std::size_t>(experts) * output * input / 2,
        0x22);
    std::vector<std::uint8_t> scales(
        static_cast<std::size_t>(experts) * output * input / 32,
        127);
    const auto weight = mfq::metal::MlxMoeWeight::from_mxfp4_slots(
        experts,
        output,
        input,
        slot_for_expert,
        mlx::core::array(
            packed.begin(),
            mlx::core::Shape{experts, output * input / 2}),
        mlx::core::array(
            scales.begin(),
            mlx::core::Shape{experts, output * input / 32}));

    std::vector<std::int32_t> repeated(4 * routes);
    for (int row = 0; row < 4; ++row) {
        for (int route = 0; route < routes; ++route) {
            repeated[static_cast<std::size_t>(row) * routes + route] = route;
        }
    }
    std::vector<std::int32_t> unique(4 * routes);
    std::iota(unique.begin(), unique.end(), 0);
    const mlx::core::array repeated_ids(
        repeated.begin(),
        mlx::core::Shape{4, routes});
    const mlx::core::array unique_ids(
        unique.begin(),
        mlx::core::Shape{4, routes});
    const mlx::core::array short_ids(
        repeated.begin(),
        mlx::core::Shape{3, routes});

    const char* prior = std::getenv("MFQ_METAL_MFE_SMALLM_NAX");
    const std::optional<std::string> saved = prior == nullptr
        ? std::nullopt
        : std::optional<std::string>(prior);
    setenv("MFQ_METAL_MFE_SMALLM_NAX", "1", 1);
    require(
        weight.prefers_mxfp4_smallm_nax(repeated_ids),
        "small-M MXFP4 NAX must accept repeated M=4 routes");
    require(
        !weight.prefers_mxfp4_smallm_nax(unique_ids),
        "small-M MXFP4 NAX must preserve the unique-route kernel");
    require(
        !weight.prefers_mxfp4_smallm_nax(short_ids),
        "small-M MXFP4 NAX must preserve the M<4 kernel");
    constexpr int arena_slots = 300;
    std::vector<std::int32_t> arena_slot_map(arena_slots);
    std::iota(arena_slot_map.begin(), arena_slot_map.end(), 0);
    std::vector<std::uint8_t> arena_packed(
        static_cast<std::size_t>(arena_slots) * output * input / 2,
        0x22);
    std::vector<std::uint8_t> arena_scales(
        static_cast<std::size_t>(arena_slots) * output * input / 32,
        127);
    const auto arena_weight = mfq::metal::MlxMoeWeight::from_mxfp4_slots(
        arena_slots,
        output,
        input,
        arena_slot_map,
        mlx::core::array(
            arena_packed.begin(),
            mlx::core::Shape{arena_slots, output * input / 2}),
        mlx::core::array(
            arena_scales.begin(),
            mlx::core::Shape{arena_slots, output * input / 32}),
        256);
    std::vector<std::int32_t> high_physical_ids(4 * routes, 299);
    require(
        arena_weight.prefers_mxfp4_smallm_nax(
            mlx::core::array(
                high_physical_ids.begin(),
                mlx::core::Shape{4, routes})),
        "logical expert geometry rejected a valid high SSD arena slot");
    setenv("MFQ_METAL_MFE_SMALLM_NAX", "0", 1);
    require(
        !weight.prefers_mxfp4_smallm_nax(repeated_ids),
        "small-M MXFP4 NAX disable override was ignored");

    const char* prior_prefill = std::getenv("MFQ_METAL_MFE_PREFILL_NAX");
    const std::optional<std::string> saved_prefill = prior_prefill == nullptr
        ? std::nullopt
        : std::optional<std::string>(prior_prefill);
    setenv("MFQ_METAL_MFE_PREFILL_NAX", "1", 1);
    require(
        weight.prefers_mxfp4_nax_prefill(4 * routes),
        "MXFP4 NAX prefill query ignored an explicit enable");
    require(
        weight.recommended_mxfp4_nax_prefill_tokens(routes) == 5440,
        "MXFP4 NAX top-k=6 prefill recommendation mismatch");
    const auto automatic_nax_disabled =
        weight.with_automatic_mxfp4_nax_prefill(false);
    require(
        automatic_nax_disabled.recommended_mxfp4_nax_prefill_tokens(routes)
            == 5440,
        "explicit MXFP4 NAX force must override the source safety policy");
    require(
        weight.recommended_mxfp4_nax_prefill_tokens(1) == 32768,
        "MXFP4 NAX top-k=1 prefill recommendation mismatch");
    require(
        weight.recommended_mxfp4_nax_prefill_tokens(0) == 0,
        "MXFP4 NAX accepted an invalid route count");
    const auto paired = mfq::metal::MlxMoeWeight::concatenate_projections(
        {weight, weight});
    require(
        paired.recommended_mxfp4_nax_prefill_tokens(routes) == 5440,
        "split MXFP4 projections lost their prefill recommendation");
    setenv("MFQ_METAL_MFE_PREFILL_NAX", "0", 1);
    require(
        !weight.prefers_mxfp4_nax_prefill(4 * routes),
        "MXFP4 NAX prefill query ignored an explicit disable");
    require(
        weight.recommended_mxfp4_nax_prefill_tokens(routes) == 0,
        "MXFP4 NAX prefill disable override was ignored");
    if (saved_prefill.has_value()) {
        setenv(
            "MFQ_METAL_MFE_PREFILL_NAX",
            saved_prefill->c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_PREFILL_NAX");
    }

    std::vector<float> input_values(4 * input);
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = static_cast<float>(
            static_cast<int>((index * 11 + 3) % 31) - 15) / 256.0f;
    }
    const mlx::core::array input_array(
        input_values.begin(),
        mlx::core::Shape{4, input});
    const auto route_order = mlx::core::contiguous(
        mlx::core::astype(
            mlx::core::argsort(
                mlx::core::reshape(
                    repeated_ids,
                    mlx::core::Shape{4 * routes})),
            mlx::core::int32));
    const auto reference = evaluated_floats(
        weight.routed_matmul(input_array, repeated_ids));
    const auto forced_nax = evaluated_floats(
        mlx::core::take(
            weight.routed_matmul_sorted(
                input_array,
                repeated_ids,
                route_order,
                false,
                false,
                0.0f,
                nullptr,
                true),
            mlx::core::argsort(route_order),
            0));
    require(
        reference.size() == forced_nax.size(),
        "small-M MXFP4 NAX output size mismatch");
    for (std::size_t index = 0; index < reference.size(); ++index) {
        require_close(reference[index], forced_nax[index], 4e-3f);
    }

    const auto bf16_input = mlx::core::astype(
        input_array,
        mlx::core::bfloat16);
    auto bf16_sorted = weight.routed_matmul_sorted(
        bf16_input,
        repeated_ids,
        route_order,
        false,
        false,
        0.0f,
        nullptr,
        true);
    require(
        bf16_sorted.dtype() == mlx::core::bfloat16,
        "native MXFP4 gather-QMM did not preserve BF16 activations");
    const auto bf16_nax = evaluated_floats(
        mlx::core::take(
            std::move(bf16_sorted),
            mlx::core::argsort(route_order),
            0));
    require(
        reference.size() == bf16_nax.size(),
        "BF16 MXFP4 NAX output size mismatch");
    for (std::size_t index = 0; index < reference.size(); ++index) {
        require_close(reference[index], bf16_nax[index], 1.5e-2f);
    }

    if (saved.has_value()) {
        setenv(
            "MFQ_METAL_MFE_SMALLM_NAX",
            saved->c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_SMALLM_NAX");
    }
}

void test_mxfp4_decode_down_reduce() {
    constexpr int experts = 8;
    constexpr int output = 13;
    constexpr int input = 64;
    constexpr int routes = 6;
    const auto fixture = make_mxfp4(experts * output, input);
    std::vector<std::uint8_t> blob;
    append_magic(blob, "NIM2");
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, 5);
    append<std::uint64_t>(blob, fixture.blob.size());
    append<std::uint64_t>(blob, 0);
    for (int expert = 0; expert < experts; ++expert) {
        append<std::int32_t>(blob, experts - expert - 1);
    }
    blob.insert(blob.end(), {'M', 'X', 'F', 'P', '4'});
    append_bytes(blob, fixture.blob);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(blob);

    std::vector<float> source_values(routes * input);
    for (std::size_t index = 0; index < source_values.size(); ++index) {
        source_values[index] = static_cast<float>(
            static_cast<int>((index * 17 + 9) % 37) - 18) / 128.0f;
    }
    const std::vector<std::int32_t> ids{7, 1, 6, 0, 3, 2};
    const std::vector<float> route_weights{
        0.07f, 0.13f, 0.19f, 0.23f, 0.17f, 0.21f};
    const auto source = mlx::core::astype(
        mlx::core::array(
            source_values.begin(),
            mlx::core::Shape{1, routes, input}),
        mlx::core::float16);
    const mlx::core::array expert_ids(
        ids.begin(),
        mlx::core::Shape{1, routes});
    const mlx::core::array weights(
        route_weights.begin(),
        mlx::core::Shape{1, routes});

    const char* prior = std::getenv(
        "MFQ_METAL_MFE_DECODE_DOWN_REDUCE");
    const std::optional<std::string> saved = prior == nullptr
        ? std::nullopt
        : std::optional<std::string>(prior);
    const char* prior_rows = std::getenv(
        "MFQ_METAL_MFE_DECODE_DOWN_REDUCE_ROWS");
    const std::optional<std::string> saved_rows = prior_rows == nullptr
        ? std::nullopt
        : std::optional<std::string>(prior_rows);
    const char* prior_pack = std::getenv(
        "MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD");
    const std::optional<std::string> saved_pack = prior_pack == nullptr
        ? std::nullopt
        : std::optional<std::string>(prior_pack);
    setenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD", "1", 1);
    const auto pair_reference = evaluated_floats(
        weight.routed_matmul(source, expert_ids));
    for (const char* rows : {"2", "4"}) {
        setenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD", rows, 1);
        const auto packed_rows = evaluated_floats(
            weight.routed_matmul(source, expert_ids));
        require(pair_reference.size() == packed_rows.size(),
                "MXFP4 decode row packing output shape mismatch");
        for (std::size_t index = 0; index < pair_reference.size(); ++index) {
            require_close(pair_reference[index], packed_rows[index], 1e-4f);
        }
    }
    setenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD", "1", 1);
    setenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE", "0", 1);
    require(!weight.supports_fused_routed_reduce(),
            "disabled down-reduce must not advertise fused support");
    const auto reference = evaluated_floats(
        weight.routed_matmul_reduce(source, expert_ids, weights));
    setenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE", "1", 1);
    require(weight.supports_fused_routed_reduce(),
            "native MXFP4 down-reduce capability is missing");
    for (const char* rows : {"1", "2", "4"}) {
        setenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE_ROWS", rows, 1);
        const auto fused = evaluated_floats(
            weight.routed_matmul_reduce(source, expert_ids, weights));
        require(
            reference.size() == static_cast<std::size_t>(output)
                && fused.size() == reference.size(),
            "MXFP4 decode down-reduce output shape mismatch");
        for (std::size_t index = 0; index < reference.size(); ++index) {
            require_close(reference[index], fused[index], 1e-4f);
        }
    }
    std::vector<std::int32_t> packed_ids;
    packed_ids.reserve(ids.size());
    for (const auto expert : ids) {
        packed_ids.push_back(((expert + 1) << 8) | expert);
    }
    const auto packed_fused = evaluated_floats(
        weight.routed_matmul_reduce_packed(
            source,
            mlx::core::array(
                packed_ids.begin(),
                mlx::core::Shape{1, routes}),
            weights));
    require(
        packed_fused.size() == reference.size(),
        "packed MXFP4 decode down-reduce output shape mismatch");
    for (std::size_t index = 0; index < reference.size(); ++index) {
        require_close(reference[index], packed_fused[index], 1e-4f);
    }

    for (const int tokens : {2, 6}) {
        std::vector<float> multi_source_values(
            static_cast<std::size_t>(tokens) * routes * input);
        for (std::size_t index = 0;
             index < multi_source_values.size();
             ++index) {
            multi_source_values[index] = static_cast<float>(
                static_cast<int>((index * 11 + 5) % 41) - 20) / 128.0f;
        }
        std::vector<std::int32_t> multi_ids(
            static_cast<std::size_t>(tokens) * routes);
        std::vector<float> multi_weights(
            static_cast<std::size_t>(tokens) * routes);
        for (int token = 0; token < tokens; ++token) {
            for (int route = 0; route < routes; ++route) {
                const auto index = static_cast<std::size_t>(
                    token * routes + route);
                multi_ids[index] = ids[
                    static_cast<std::size_t>((route + token) % routes)];
                multi_weights[index] = route_weights[
                    static_cast<std::size_t>(route)];
            }
        }
        const auto multi_source = mlx::core::astype(
            mlx::core::array(
                multi_source_values.begin(),
                mlx::core::Shape{tokens, routes, input}),
            mlx::core::float16);
        const auto multi_expert_ids = mlx::core::array(
            multi_ids.begin(),
            mlx::core::Shape{tokens, routes});
        const auto multi_route_weights = mlx::core::array(
            multi_weights.begin(),
            mlx::core::Shape{tokens, routes});
        setenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE", "0", 1);
        const auto multi_reference = evaluated_floats(
            weight.routed_matmul_reduce(
                multi_source,
                multi_expert_ids,
                multi_route_weights));
        setenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE", "1", 1);
        const auto multi_fused = evaluated_floats(
            weight.routed_matmul_reduce(
                multi_source,
                multi_expert_ids,
                multi_route_weights));
        require(
            multi_reference.size()
                    == static_cast<std::size_t>(tokens * output)
                && multi_fused.size() == multi_reference.size(),
            "MXFP4 small-M down-reduce output shape mismatch");
        for (std::size_t index = 0;
             index < multi_reference.size();
             ++index) {
            require_close(
                multi_reference[index],
                multi_fused[index],
                1e-4f);
        }
    }

    if (saved.has_value()) {
        setenv(
            "MFQ_METAL_MFE_DECODE_DOWN_REDUCE",
            saved->c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE");
    }
    if (saved_rows.has_value()) {
        setenv(
            "MFQ_METAL_MFE_DECODE_DOWN_REDUCE_ROWS",
            saved_rows->c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_DECODE_DOWN_REDUCE_ROWS");
    }
    if (saved_pack.has_value()) {
        setenv(
            "MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD",
            saved_pack->c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD");
    }
}

void test_mxfp4_decode_swiglu_row_packing() {
    constexpr int experts = 8;
    constexpr int matrix_output = 16;
    constexpr int input = 64;
    constexpr int routes = 6;
    std::vector<std::int32_t> slots(experts);
    for (int expert = 0; expert < experts; ++expert) {
        slots[static_cast<std::size_t>(expert)] = experts - expert - 1;
    }
    std::vector<std::uint8_t> packed(
        static_cast<std::size_t>(experts) * matrix_output * input / 2,
        0x32);
    std::vector<std::uint8_t> scales(
        static_cast<std::size_t>(experts) * matrix_output * input / 32,
        127);
    const auto weight = mfq::metal::MlxMoeWeight::from_mxfp4_slots(
        experts,
        matrix_output,
        input,
        slots,
        mlx::core::array(
            packed.begin(),
            mlx::core::Shape{
                experts,
                matrix_output * input / 2,
            }),
        mlx::core::array(
            scales.begin(),
            mlx::core::Shape{
                experts,
                matrix_output * input / 32,
            }));
    std::vector<float> source_values(input);
    for (int column = 0; column < input; ++column) {
        source_values[static_cast<std::size_t>(column)] =
            static_cast<float>((column * 7 + 3) % 23 - 11) / 128.0f;
    }
    const auto source = mlx::core::astype(
        mlx::core::array(
            source_values.begin(),
            mlx::core::Shape{1, input}),
        mlx::core::float16);
    const std::vector<std::int32_t> id_values{7, 1, 6, 0, 3, 2};
    const mlx::core::array expert_ids(
        id_values.begin(),
        mlx::core::Shape{1, routes});
    const char* prior = std::getenv(
        "MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD");
    const std::optional<std::string> saved = prior == nullptr
        ? std::nullopt
        : std::optional<std::string>(prior);
    setenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD", "1", 1);
    const auto reference = evaluated_floats(
        weight.routed_swiglu(source, expert_ids, 0.0f));
    for (const char* rows : {"2", "4"}) {
        setenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD", rows, 1);
        const auto candidate = evaluated_floats(
            weight.routed_swiglu(source, expert_ids, 0.0f));
        require(reference.size() == candidate.size(),
                "MXFP4 decode SwiGLU row packing shape mismatch");
        for (std::size_t index = 0; index < reference.size(); ++index) {
            require_close(reference[index], candidate[index], 1e-4f);
        }
    }
    if (saved.has_value()) {
        setenv(
            "MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD",
            saved->c_str(),
            1);
    } else {
        unsetenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD");
    }
}

void test_large_mxfp4_arena_avoids_single_group_builder() {
    constexpr int experts = 1025;
    constexpr int output = 1;
    constexpr int input = 32;
    std::vector<std::int32_t> slots(experts);
    std::iota(slots.begin(), slots.end(), 0);
    std::vector<std::uint8_t> packed(
        static_cast<std::size_t>(experts) * output * input / 2,
        0x22);
    std::vector<std::uint8_t> scales(
        static_cast<std::size_t>(experts) * output * input / 32,
        127);
    const auto weight = mfq::metal::MlxMoeWeight::from_mxfp4_slots(
        experts,
        output,
        input,
        slots,
        mlx::core::array(
            packed.begin(),
            mlx::core::Shape{experts, output * input / 2}),
        mlx::core::array(
            scales.begin(),
            mlx::core::Shape{experts, output * input / 32}));
    require(
        !weight.supports_grouped_mmq(),
        "large SSD arena must not use a single-threadgroup block builder");
    std::vector<float> input_values(32 * input, 0.0f);
    std::vector<std::int32_t> expert_ids(32, 0);
    const auto output_values = evaluated_floats(weight.routed_matmul(
        mlx::core::astype(
            mlx::core::array(
                input_values.begin(),
                mlx::core::Shape{32, input}),
            mlx::core::float16),
        mlx::core::array(
            expert_ids.begin(),
            mlx::core::Shape{32, 1})));
    require(
        output_values.size() == 32,
        "large SSD arena mapped fallback output shape mismatch");

    std::vector<float> small_input_values(4 * input, 0.0f);
    std::vector<std::int32_t> small_expert_ids(4, 0);
    std::vector<std::int32_t> small_order{0, 1, 2, 3};
    const auto direct_values = evaluated_floats(
        weight.routed_matmul_sorted(
            mlx::core::array(
                small_input_values.begin(),
                mlx::core::Shape{4, input}),
            mlx::core::array(
                small_expert_ids.begin(),
                mlx::core::Shape{4, 1}),
            mlx::core::array(
                small_order.begin(),
                mlx::core::Shape{4}),
            false,
            false,
            0.0f,
            nullptr,
            true));
    require(
        direct_values.size() == 4,
        "large SSD arena direct MXFP4 output shape mismatch");
}

void test_vq_small_m_route_order() {
    const char* setting = std::getenv("MFQ_METAL_MFE_SORT_ROUTES");
    const std::optional<std::string> saved = setting
        ? std::optional<std::string>(setting) : std::nullopt;
    for (const auto [output, input] : {std::pair{64, 24}, std::pair{24, 64}}) {
        const auto fixture = make_vq_moe_fixture({
            make_nvq1_s(output, input),
            make_nvq1_l(output, input),
            make_jsc_nvq(output, input),
            make_jsc_nvq(output, input, "NVQ2J-L", 4, 8, 10, false, 4),
            make_jsc_nvq(output, input, "NVQ2J-XL", 5, 8, 12, true, 4),
            make_jsc_nvq(output, input, "NVQ3J", 2, 4, 8),
            make_jsc_nvq(output, input, "NVQ3J-512", 3, 4, 9, false, 4),
            make_jsc_nvq(output, input, "NVQ3J-L", 6, 4, 10, false, 4),
            make_varied_plain_nvq(output, input),
            make_varied_plain_nvq(output, input, false, true),
            make_varied_plain_nvq(output, input, true),
            make_varied_plain_nvq(output, input, false, false, 32),
            make_varied_plain_nvq(output, input, false, true, 32),
            make_varied_plain_nvq(output, input, true, false, 32),
            make_varied_plain_nvq(output, input, false, false, 64),
            make_varied_plain_nvq(output, input, false, true, 64),
            make_varied_plain_nvq(output, input, true, false, 64),
        });
        const auto weight = mfq::metal::MlxMfeWeight::from_blob(fixture.blob);
        const int routes = static_cast<int>(fixture.expert_weights.size());
        std::vector<mfq::metal::MlxVqWeight> direct;
        direct.reserve(routes);
        for (const auto& expert : fixture.expert_weights) {
            direct.push_back(mfq::metal::MlxVqWeight::from_blob(expert.dtype, expert.blob));
        }
        for (int tokens = 1; tokens <= 6; ++tokens) {
            for (bool shared : {true, false}) {
                std::vector<float> values(tokens * (shared ? 1 : routes) * input);
                for (std::size_t index = 0; index < values.size(); ++index) {
                    values[index] = static_cast<float>(static_cast<int>(
                        (index * 17 + 3) % 31) - 15) / 512.0f;
                }
                std::vector<std::int32_t> ids(tokens * routes);
                for (std::size_t index = 0; index < ids.size(); ++index) {
                    ids[index] = static_cast<std::int32_t>((index * 3 + index / routes) % routes);
                }
                const auto source = mlx::core::array(values.begin(), shared
                    ? mlx::core::Shape{tokens, input}
                    : mlx::core::Shape{tokens, routes, input});
                const auto selected = mlx::core::array(ids.begin(), mlx::core::Shape{tokens, routes});
                unsetenv("MFQ_METAL_MFE_SORT_ROUTES");
                const auto actual = evaluated_floats(weight.routed_matmul(source, selected));
                setenv("MFQ_METAL_MFE_SORT_ROUTES", "1", 1);
                const auto sorted = evaluated_floats(weight.routed_matmul(source, selected));
                require(actual == sorted, "small-M VQ route sorting changed output");
                if (tokens > 1) {
                    std::vector<std::int32_t> groups(ids.size() * 8, -1);
                    for (std::size_t index = 0; index < ids.size(); ++index) {
                        int matched = 0;
                        for (std::size_t route = 0; route < ids.size(); ++route)
                            if (ids[route] == ids[index])
                                groups[index * 8 + 2 + matched++] = static_cast<std::int32_t>(route);
                        require(matched <= 6, "route group fixture exceeds Top-K uniqueness bound");
                        groups[index * 8] = ids[index];
                        groups[index * 8 + 1] = matched;
                    }
                    const auto plan = mlx::core::array(groups.begin(), mlx::core::Shape{tokens, routes, 8});
                    const auto planned = evaluated_floats(weight.routed_matmul(source, selected, &plan));
                    require(actual == planned, "route group consumption changed mixed VQ output");
                }
                if (shared) {
                    const auto check_direct = [&](const std::vector<float>& result, int expert,
                                                  std::string_view mode) {
                        for (int token = 0; token < tokens; ++token) {
                            const std::vector<float> row(values.begin() + token * input,
                                values.begin() + (token + 1) * input);
                            for (int column = 0; column < output; ++column) {
                                try {
                                    require_close(result[token * output + column],
                                        routed_vq_dot(row, fixture, expert, column), 2e-3f);
                                } catch (const std::exception& error) {
                                    throw std::runtime_error(std::string(mode) + " "
                                        + fixture.expert_weights[expert].dtype + " M="
                                        + std::to_string(tokens) + " OUT=" + std::to_string(output)
                                        + " K=" + std::to_string(input) + " row=" + std::to_string(token)
                                        + " output=" + std::to_string(column) + ": " + error.what());
                                }
                            }
                        }
                    };
                    for (int expert = 0; expert < routes; ++expert) {
                        check_direct(evaluated_floats(direct[expert].matmul(source)), expert, "standalone");
                    }
                    const mfq::metal::MlxGroupedLinear xl_pair({&direct[4], &direct[4]});
                    for (const auto& result : xl_pair.matmul(mlx::core::astype(source, mlx::core::float16))) {
                        check_direct(evaluated_floats(result), 4, "XL FP16 output tile");
                    }
                    for (int group = 0; group < (routes + 1) / 2; ++group) {
                        const int first = std::min(group * 2, routes - 2);
                        std::vector<mfq::metal::MlxGroupedLinearWeightRef> refs;
                        for (int expert = first; expert < std::min(first + 2, routes); ++expert) {
                            refs.emplace_back(&direct[expert]);
                        }
                        const mfq::metal::MlxGroupedLinear grouped(std::move(refs));
                        const auto results = grouped.matmul(source);
                        for (std::size_t index = 0; index < results.size(); ++index) {
                            check_direct(evaluated_floats(results[index]), first + static_cast<int>(index), "grouped");
                        }
                    }
                }
                for (int token = 0; token < tokens; ++token) {
                    for (int route = 0; route < routes; ++route) {
                        const auto first = (token * (shared ? 1 : routes)
                            + (shared ? 0 : route)) * input;
                        const std::vector<float> row(values.begin() + first, values.begin() + first + input);
                        for (int column = 0; column < output; ++column) {
                            require_close(actual[(token * routes + route) * output + column],
                                routed_vq_dot(row, fixture, ids[token * routes + route], column), 2e-3f);
                        }
                    }
                }
                if (tokens == 3 || tokens == 6) {
                    auto repeated_ids = ids;
                    std::fill(repeated_ids.begin(), repeated_ids.begin() + 8, 4);
                    const auto repeated_routes = mlx::core::array(repeated_ids.begin(),
                        mlx::core::Shape{tokens, routes});
                    const auto repeated = evaluated_floats(weight.routed_matmul(source, repeated_routes));
                    const auto packed = evaluated_floats(weight.routed_swiglu(source, repeated_routes));
                    const auto pair = mfq::metal::MlxMfeWeight::from_projection_blobs(
                        std::array<std::span<const std::uint8_t>, 2>{fixture.blob, fixture.blob});
                    const auto split = evaluated_floats(pair.routed_swiglu(source, repeated_routes));
                    for (int index = 0; index < tokens * routes; ++index) {
                        const auto first = (shared ? index / routes : index) * input;
                        const std::vector<float> row(values.begin() + first, values.begin() + first + input);
                        for (int column = 0; column < output; ++column) {
                            const auto value = repeated[index * output + column];
                            require_close(value, routed_vq_dot(row, fixture,
                                repeated_ids[index], column), 2e-3f);
                            require_close(split[index * output + column],
                                value * value / (1.0f + std::exp(-value)), 2e-3f);
                            if (column < output / 2) {
                                const auto up = repeated[index * output + column + output / 2];
                                require_close(packed[index * (output / 2) + column],
                                    value * up / (1.0f + std::exp(-value)), 2e-3f);
                            }
                        }
                    }
                }
            }
        }
    }
    if (saved) setenv("MFQ_METAL_MFE_SORT_ROUTES", saved->c_str(), 1);
    else unsetenv("MFQ_METAL_MFE_SORT_ROUTES");
}

void test_generic_vq_wide_small_m() {
    using namespace mlx::core;
    constexpr int output = 9;
    for (const int input : {640, 2560}) {
        const auto fixture = make_vq_moe_fixture({
            make_varied_plain_nvq(output, input, false, false, 32),
            make_varied_plain_nvq(output, input, false, true, 32),
            make_varied_plain_nvq(output, input, true, false, 32),
            make_varied_plain_nvq(output, input, false, false, 64),
            make_varied_plain_nvq(output, input, false, true, 64),
            make_varied_plain_nvq(output, input, true, false, 64),
        });
        const auto weight = mfq::metal::MlxMfeWeight::from_blob(fixture.blob);
        const int routes = static_cast<int>(fixture.expert_weights.size());
        for (const auto dtype : {float16, float32}) {
            for (int tokens = 1; tokens <= 6; ++tokens) {
                for (const bool shared : {true, false}) {
                    std::vector<float> values(tokens * (shared ? 1 : routes) * input);
                    for (std::size_t index = 0; index < values.size(); ++index) {
                        values[index] = float(int((index * 17 + 3) % 31) - 15) / 512.0f;
                    }
                    std::vector<std::int32_t> ids(tokens * routes);
                    for (std::size_t index = 0; index < ids.size(); ++index) {
                        ids[index] = (index * 5 + index / routes) % routes;
                    }
                    const auto source = astype(array(values.begin(), shared
                        ? Shape{tokens, input} : Shape{tokens, routes, input}), dtype);
                    const auto selected = array(ids.begin(), Shape{tokens, routes});
                    const auto actual = evaluated_floats(weight.routed_matmul(source, selected));
                    std::vector<float> expected(actual.size());
                    for (int token = 0; token < tokens; ++token) {
                        for (int route = 0; route < routes; ++route) {
                            const int first = (token * (shared ? 1 : routes)
                                + (shared ? 0 : route)) * input;
                            const std::vector<float> row(values.begin() + first,
                                values.begin() + first + input);
                            for (int column = 0; column < output; ++column) {
                                expected[(token * routes + route) * output + column] =
                                    routed_vq_dot(row, fixture, ids[token * routes + route], column);
                            }
                        }
                    }
                    const auto rounded = evaluated_floats(astype(array(expected.begin(),
                        Shape{tokens, routes, output}), dtype));
                    for (std::size_t index = 0; index < actual.size(); ++index) {
                        require_close(actual[index], rounded[index], 2e-3f);
                    }
                }
            }
        }
    }
}

void test_vq_cohorts_and_ffn() {
    constexpr int tokens = 2;
    constexpr int routes = 5;
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
            "NVQ3J-L",
            6,
            4,
            10,
            false,
            4,
            1),
        make_jsc_nvq(
            width,
            width,
            "NVQ2J-XL",
            5,
            8,
            12,
            true),
        make_nvq1_s(width, width),
        make_npq(width, width),
        make_rotated_nepq1_s(width, width),
    });
    const auto weight =
        mfq::metal::MlxMoeWeight::from_blob(
            fixture.blob);
    require(
        weight.experts() == fixture.experts
            && weight.out_per_expert() == width
            && weight.neuron_len() == width,
        "VQ MFE metadata mismatch");

    const std::vector<std::int32_t> ids{
        0, 9, 2, 3, 1,
        6, 7, 8, 4, 5,
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
                try {
                    require_close(
                        actual[(token * routes + route) * width + output],
                        routed_vq_dot(source, fixture, expert, output),
                        2e-3f);
                } catch (const std::exception& error) {
                    throw std::runtime_error(
                        "dense reference expert=" + std::to_string(expert)
                        + " row=" + std::to_string(output)
                        + ": " + error.what());
                }
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
        0.30f, 0.25f, 0.20f, 0.15f, 0.10f,
        0.10f, 0.15f, 0.20f, 0.25f, 0.30f,
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

void test_shared_rotated_input() {
    constexpr int width = 24;
    constexpr int routes = 3;
    const auto fixture = make_vq_moe_fixture({
        make_rotated_nepq1_s(width, width),
        make_rotated_nepq1_s(width, width),
        make_rotated_nepq1_s(width, width, 0x8899aabbccddeeffull, 1),
    });
    const auto weight = mfq::metal::MlxMfeWeight::from_blob(fixture.blob);
    const auto pair = mfq::metal::MlxMfeWeight::concatenate_projections({weight, weight});
    for (int tokens : {1, 6, 65, 513}) {
        std::vector<float> values(tokens * width);
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<float>(
                static_cast<int>((index * 13 + 7) % 29) - 14) / 1024.0f;
        }
        std::vector<std::int32_t> route_ids(tokens * routes);
        for (std::size_t index = 0; index < route_ids.size(); ++index) {
            route_ids[index] = static_cast<int>((index * 7 + index / routes) % routes);
        }
        const auto x = mlx::core::astype(mlx::core::array(
            values.begin(), mlx::core::Shape{tokens, width}), mlx::core::float16);
        const auto repeated = mlx::core::contiguous(mlx::core::broadcast_to(
            mlx::core::expand_dims(x, 1), mlx::core::Shape{tokens, routes, width}));
        const auto ids = mlx::core::array(
            route_ids.begin(), mlx::core::Shape{tokens, routes});
        const auto order = mlx::core::astype(mlx::core::argsort(
            mlx::core::reshape(ids, mlx::core::Shape{tokens * routes})), mlx::core::int32);
        const auto inverse = mfq::metal::moe_inverse_permutation(order);
        for (const auto& projection : {weight, pair}) {
            auto shared_output = projection.routed_matmul(x, ids);
            std::function<void(const mlx::core::array&)> visit = [&](const auto& node) {
                if (!node.has_primitive()) return;
                const auto* name = node.primitive().name();
                if ((std::strcmp(name, "NativeMfePrimitive") == 0
                     || std::strcmp(name, "GroupedMmqPrimitive") == 0)
                    && node.inputs().size() >= 26) {
                    require(node.inputs()[22].size() ==
                        static_cast<std::size_t>(3 * tokens * width),
                        "shared HSG1 input was expanded by route count");
                }
                for (const auto& input : node.inputs()) visit(input);
            };
            visit(shared_output);
            const auto actual = evaluated_floats(shared_output);
            const auto reference = evaluated_floats(projection.routed_matmul(repeated, ids));
            require(actual == reference, "shared HSG1 routed output changed");
            if (tokens >= 32) {
                const auto plan = projection.build_grouped_mmq_plan(ids, order);
                auto shared_sorted = projection.routed_matmul_sorted(
                    x, ids, order, false, false, 0.0f, &plan);
                visit(shared_sorted);
                const auto sorted = evaluated_floats(shared_sorted);
                const auto sorted_reference = evaluated_floats(projection.routed_matmul_sorted(
                    repeated, ids, order, false, false, 0.0f, &plan));
                require(sorted == sorted_reference, "shared HSG1 sorted output changed");
                const auto restored = evaluated_floats(mlx::core::take(shared_sorted, inverse, 0));
                for (std::size_t index = 0; index < actual.size(); ++index)
                    require_close(restored[index], actual[index], 2e-3f);
            }
        }
        const auto actual = evaluated_floats(weight.routed_matmul(x, ids));
        for (int token = 0; token < tokens; ++token) {
            const std::vector<float> source(values.begin() + token * width,
                values.begin() + (token + 1) * width);
            for (int route = 0; route < routes; ++route) {
                const int expert = route_ids[token * routes + route];
                for (int output = 0; output < width; ++output)
                    require_close(actual[(token * routes + route) * width + output],
                        routed_vq_dot(source, fixture, expert, output), 2e-3f);
            }
        }
    }
}

void test_nepq_a_routed_and_fused_swiglu() {
    constexpr int tokens = 2;
    constexpr int routes = 2;
    constexpr int width = 24;
    auto fixture = make_vq_moe_fixture({
        add_nepq_a_residual(
            make_rotated_nepq0_s(width, width),
            false),
        add_nepq_a_residual(
            make_rotated_nepq1_s(width, width),
            true),
    });
    const auto weight =
        mfq::metal::MlxMoeWeight::from_blob(fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "NEPQ-A must support residual-aware grouped prefill");
    std::vector<float> input(tokens * width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 7 + 3) % 17) - 8)
            / 128.0f;
    }
    const std::vector<std::int32_t> ids{0, 1, 1, 0};
    const mlx::core::array input_array(
        input.begin(),
        mlx::core::Shape{tokens, width});
    const mlx::core::array id_array(
        ids.begin(),
        mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, id_array));
    const auto projected_pairs = evaluated_floats(
        mfq::metal::MlxMoeWeight::concatenate_projections(
            {weight, weight}).routed_matmul(
                input_array,
                id_array));
    auto fused_fixture = make_vq_moe_fixture({
        add_nepq_a_residual(
            make_rotated_nepq0_s(2 * width, width),
            false),
        add_nepq_a_residual(
            make_rotated_nepq1_s(2 * width, width),
            true),
    });
    const auto fused_weight =
        mfq::metal::MlxMoeWeight::from_blob(fused_fixture.blob);
    const auto fused = evaluated_floats(
        fused_weight.routed_swiglu(input_array, id_array));
    for (int token = 0; token < tokens; ++token) {
        std::vector<float> source(
            input.begin() + token * width,
            input.begin() + (token + 1) * width);
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            for (int output = 0; output < width; ++output) {
                const float projected_value = routed_vq_dot(
                    source,
                    fixture,
                    expert,
                    output);
                const auto offset = static_cast<std::size_t>(
                    (token * routes + route) * width + output);
                require_close(actual[offset], projected_value, 3e-3f);
                require_close(
                    projected_pairs[
                        static_cast<std::size_t>(
                            token * routes + route)
                            * (2 * width)
                        + output],
                    projected_value,
                    3e-3f);
                require_close(
                    projected_pairs[
                        static_cast<std::size_t>(
                            token * routes + route)
                            * (2 * width)
                        + width + output],
                    projected_value,
                    3e-3f);
                const float gate = routed_vq_dot(
                    source,
                    fused_fixture,
                    expert,
                    output);
                const float up = routed_vq_dot(
                    source,
                    fused_fixture,
                    expert,
                    width + output);
                const float expected_fused = gate
                    / (1.0f + std::exp(-gate))
                    * up;
                require_close(fused[offset], expected_fused, 8e-3f);
            }
        }
    }

    constexpr int prefill_tokens = 49;
    std::vector<float> prefill_input(
        prefill_tokens * width);
    for (std::size_t index = 0;
         index < prefill_input.size();
         ++index) {
        prefill_input[index] = static_cast<float>(
            static_cast<int>((index * 11 + 1) % 23) - 11)
            / 128.0f;
    }
    std::vector<std::int32_t> prefill_ids(
        prefill_tokens * routes);
    for (int token = 0; token < prefill_tokens; ++token) {
        prefill_ids[token * routes] = token & 1;
        prefill_ids[token * routes + 1] = 1 - (token & 1);
    }
    const auto prefill_actual = evaluated_floats(
        weight.routed_matmul(
            mlx::core::astype(
                mlx::core::array(
                    prefill_input.begin(),
                    mlx::core::Shape{
                        prefill_tokens,
                        width,
                    }),
                mlx::core::float16),
            mlx::core::array(
                prefill_ids.begin(),
                mlx::core::Shape{
                    prefill_tokens,
                    routes,
                })));
    for (int token = 0; token < prefill_tokens; ++token) {
        const std::vector<float> token_input(
            prefill_input.begin() + token * width,
            prefill_input.begin() + (token + 1) * width);
        for (int route = 0; route < routes; ++route) {
            const int expert = prefill_ids[token * routes + route];
            for (int output = 0; output < width; ++output) {
                const auto offset = static_cast<std::size_t>(
                    (token * routes + route) * width + output);
                require_close(
                    prefill_actual[offset],
                    routed_vq_dot(
                        token_input,
                        fixture,
                        expert,
                        output),
                    4e-3f);
            }
        }
    }
}

mfq::metal::MlxMoeWeight make_v4f_nepq_benchmark_weight(
    bool residual) {
    constexpr int experts = 6;
    constexpr int output = 2048;
    constexpr int input = 4096;
    std::vector<VqFixture> weights;
    weights.reserve(experts);
    for (int expert = 0; expert < experts; ++expert) {
        auto weight = make_rotated_nepq0_s(
            output,
            input,
            0x1020304050607080ull);
        if (residual) {
            weight = add_nepq_a_residual(
                std::move(weight),
                false);
        }
        weights.push_back(std::move(weight));
    }
    auto fixture = make_vq_moe_fixture(
        std::move(weights));
    return mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
}

double benchmark_v4f_nepq_weight(
    const mfq::metal::MlxMoeWeight& weight,
    const mlx::core::array& input,
    const mlx::core::array& expert_ids,
    int repetitions) {
    const auto started =
        std::chrono::steady_clock::now();
    for (int repetition = 0;
         repetition < repetitions;
         ++repetition) {
        auto output = weight.routed_matmul(
            input,
            expert_ids);
        output.eval();
    }
    const auto elapsed =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now()
            - started).count();
    return elapsed / repetitions;
}

void benchmark_v4f_nepq_a() {
    constexpr int input_width = 4096;
    constexpr int routes = 6;
    constexpr int repetitions = 40;
    std::vector<float> source(input_width);
    for (int column = 0;
         column < input_width;
         ++column) {
        source[static_cast<std::size_t>(column)] =
            static_cast<float>((column * 7) % 29 - 14)
            / 128.0f;
    }
    const std::vector<std::int32_t> ids{
        0, 1, 2, 3, 4, 5,
    };
    const auto input = mlx::core::astype(
        mlx::core::array(
            source.begin(),
            mlx::core::Shape{1, input_width}),
        mlx::core::float16);
    const mlx::core::array expert_ids(
        ids.begin(),
        mlx::core::Shape{1, routes});

    const auto base =
        make_v4f_nepq_benchmark_weight(false);
    const auto residual =
        make_v4f_nepq_benchmark_weight(true);
    for (int warmup = 0; warmup < 8; ++warmup) {
        auto base_output = base.routed_matmul(input, expert_ids);
        auto residual_output = residual.routed_matmul(input, expert_ids);
        base_output.eval();
        residual_output.eval();
    }
    std::vector<double> base_samples;
    std::vector<double> residual_samples;
    for (int round = 0; round < 9; ++round) {
        auto measure_base = [&] {
            base_samples.push_back(
                benchmark_v4f_nepq_weight(
                    base,
                    input,
                    expert_ids,
                    repetitions));
        };
        auto measure_residual = [&] {
            residual_samples.push_back(
                benchmark_v4f_nepq_weight(
                    residual,
                    input,
                    expert_ids,
                    repetitions));
        };
        if ((round & 1) == 0) {
            measure_base();
            measure_residual();
        } else {
            measure_residual();
            measure_base();
        }
    }
    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };
    const double base_ms = median(base_samples);
    const double residual_ms = median(residual_samples);
    constexpr int prefill_tokens = 128;
    constexpr int prefill_repetitions = 8;
    std::vector<float> prefill_source(
        prefill_tokens * input_width);
    for (std::size_t index = 0;
         index < prefill_source.size();
         ++index) {
        prefill_source[index] = static_cast<float>(
            static_cast<int>((index * 5 + 3) % 31) - 15)
            / 128.0f;
    }
    std::vector<std::int32_t> prefill_id_values(
        prefill_tokens * routes);
    for (int token = 0; token < prefill_tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            prefill_id_values[token * routes + route] =
                (token + route) % routes;
        }
    }
    const auto prefill_input = mlx::core::astype(
        mlx::core::array(
            prefill_source.begin(),
            mlx::core::Shape{
                prefill_tokens,
                input_width,
            }),
        mlx::core::float16);
    const mlx::core::array prefill_ids(
        prefill_id_values.begin(),
        mlx::core::Shape{
            prefill_tokens,
            routes,
        });
    for (int warmup = 0; warmup < 4; ++warmup) {
        auto base_output = base.routed_matmul(
            prefill_input,
            prefill_ids);
        auto residual_output = residual.routed_matmul(
            prefill_input,
            prefill_ids);
        base_output.eval();
        residual_output.eval();
    }
    base_samples.clear();
    residual_samples.clear();
    for (int round = 0; round < 7; ++round) {
        auto measure_base = [&] {
            base_samples.push_back(
                benchmark_v4f_nepq_weight(
                    base,
                    prefill_input,
                    prefill_ids,
                    prefill_repetitions));
        };
        auto measure_residual = [&] {
            residual_samples.push_back(
                benchmark_v4f_nepq_weight(
                    residual,
                    prefill_input,
                    prefill_ids,
                    prefill_repetitions));
        };
        if ((round & 1) == 0) {
            measure_base();
            measure_residual();
        } else {
            measure_residual();
            measure_base();
        }
    }
    const double base_prefill_ms = median(base_samples);
    const double residual_prefill_ms = median(residual_samples);
    const auto bandwidth = [](const auto& weight, double ms) {
        return static_cast<double>(weight.packed_nbytes())
            / (ms * 1.0e6);
    };
    std::cout
        << "V4F routed MoE decode benchmark"
        << " in=4096 out=2048 top_k=6 reps="
        << repetitions << "\n"
        << "NEPQ0-S " << base_ms << " ms/dispatch "
        << bandwidth(base, base_ms) << " effective-GB/s\n"
        << "NEPQ0-A " << residual_ms << " ms/dispatch "
        << bandwidth(residual, residual_ms)
        << " effective-GB/s overhead="
        << (residual_ms / base_ms - 1.0) * 100.0
        << "%\n"
        << "grouped-M128 NEPQ0-S " << base_prefill_ms
        << " ms/dispatch\n"
        << "grouped-M128 NEPQ0-A " << residual_prefill_ms
        << " ms/dispatch overhead="
        << (residual_prefill_ms / base_prefill_ms - 1.0) * 100.0
        << "%\n";
}

void benchmark_shared_nint_mfe() {
    constexpr int width = 4096;
    constexpr int repetitions = 30;
    const auto fixture = make_moe_fixture(
        {"NINTv2"}, width, width, 43, 32);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "NINTv2 benchmark must retain grouped-prefill support");
    for (int rows = 2; rows <= 6; ++rows) {
        std::vector<float> values(
            static_cast<std::size_t>(rows) * width);
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<float>(
                static_cast<int>((index * 13 + 5) % 31) - 15) / 128.0f;
        }
        const auto input = mlx::core::astype(
            mlx::core::array(
                values.begin(),
                mlx::core::Shape{rows, width}),
            mlx::core::float16);
        const std::vector<std::int32_t> ids(
            static_cast<std::size_t>(rows), 0);
        const auto expert_ids = mlx::core::array(
            ids.begin(), mlx::core::Shape{rows, 1});
        for (int warmup = 0; warmup < 4; ++warmup) {
            auto output = weight.routed_matmul(input, expert_ids);
            output.eval();
        }
        const double mean_ms = benchmark_v4f_nepq_weight(
            weight,
            input,
            expert_ids,
            repetitions);
        std::cout << "NINTv2 shared-kernel M=" << rows
                  << " " << mean_ms << " ms/dispatch\n";
    }
}

void test_grouped_mmq_prefill() {
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
    auto block_plan = weight.build_grouped_mmq_plan(
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
    const auto fused_automatic = evaluated_floats(
        weight.routed_swiglu(input_array, ids_array));
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
                require_close(
                    fused_automatic[
                        (token * routes + route) * (width / 2)
                        + output],
                    fused_actual[
                        (token * routes + route) * (width / 2)
                        + output],
                    1e-6f);
            }
        }
    }
}

void test_grouped_mmq_sparse_plan_bounds() {
    for (const int experts : {1, 32, 512}) {
        auto fixture = make_moe_fixture(std::vector<std::string>(experts, "NINTv2"), 4, 24, 0, 24);
        const auto weight = mfq::metal::MlxMfeWeight::from_blob(fixture.blob);
        for (const int count : {1, 10, 31, 32, 33, 257, 1024, 4096}) {
            for (const bool skewed : {false, true}) {
                std::vector<std::int32_t> ids(count);
                std::vector<int> counts(experts, 0);
                for (int row = 0; row < count; ++row) {
                    ids[row] = skewed ? 0 : (row * 37 + row / 3) % experts;
                    ++counts[ids[row]];
                }
                auto sorted = ids;
                std::sort(sorted.begin(), sorted.end());
                const auto routes = mlx::core::array(ids.begin(), mlx::core::Shape{count, 1});
                const auto order = mlx::core::astype(mlx::core::argsort(mlx::core::reshape(
                    routes, mlx::core::Shape{count})), mlx::core::int32);
                for (const int rows : {32, 48, 64, 80, 96, 128}) {
                    const auto plan = weight.build_grouped_mmq_plan(routes, order, rows);
                    require(plan.max_blocks == std::min(count, experts + (count + rows - 1) / rows),
                        "grouped block capacity exceeds the routed-row bound");
                    mlx::core::eval(plan.block_meta, plan.block_count);
                    const int actual = plan.block_count.data<std::int32_t>()[0];
                    int expected = 0;
                    for (const int population : counts) expected += (population + rows - 1) / rows;
                    require(actual == expected && actual <= plan.max_blocks,
                        "grouped block count does not cover the expert populations");
                    require(plan.block_meta.shape() == mlx::core::Shape{plan.max_blocks, 3},
                        "grouped block metadata has the wrong capacity");
                    std::vector<int> covered(count, 0);
                    const auto* metadata = plan.block_meta.data<std::int32_t>();
                    for (int block = 0; block < actual; ++block) {
                        const int first = metadata[block * 3];
                        const int expert = metadata[block * 3 + 1];
                        const int length = metadata[block * 3 + 2];
                        require(first >= 0 && length > 0 && length <= rows && first + length <= count,
                            "grouped block points outside the sorted routes");
                        require(expert >= 0 && expert < experts, "grouped block has an invalid expert");
                        for (int row = first; row < first + length; ++row) {
                            require(sorted[row] == expert, "grouped block crosses expert boundaries");
                            ++covered[row];
                        }
                    }
                    require(std::all_of(covered.begin(), covered.end(), [](int visits) { return visits == 1; }),
                        "grouped block plan drops or duplicates routed rows");
                }
            }
        }
    }
}

void test_portable_mmq_row_buckets() {
    constexpr int input_width = 96;
    constexpr int output_width = 68;
    const std::vector<int> counts{1, 8, 9, 16, 17, 32, 33, 48, 49, 64, 65, 80, 96};
    auto fixture = make_vq_moe_fixture({
        make_jsc_nvq(output_width, input_width, "NVQ3J-L", 6, 4, 10, false, 4, -1),
        make_jsc_nvq(output_width, input_width, "NVQ3J-512", 3, 4, 9),
        make_jsc_nvq(output_width, input_width, "NVQ2J-L", 4, 8, 10),
        make_nvq1_l(output_width, input_width),
        make_jsc_nvq(output_width, input_width, "NVQ3J", 2, 4, 8),
        make_jsc_nvq(output_width, input_width, "NVQ3J-L", 6, 4, 10, false, 4, -1),
        make_jsc_nvq(output_width, input_width, "NVQ3J", 2, 4, 8),
        make_nvq1_l(output_width, input_width),
        make_jsc_nvq(output_width, input_width, "NVQ2J-L", 4, 8, 10),
        make_jsc_nvq(output_width, input_width, "NVQ3J-L", 6, 4, 10, false, 4, -1),
        make_nvq1_s(output_width, input_width),
        make_jsc_nvq(output_width, input_width, "NVQ3J-512", 3, 4, 9),
        make_jsc_nvq(output_width, input_width, "NVQ3J", 2, 4, 8),
    });
    const auto weight = mfq::metal::MlxMfeWeight::from_blob(fixture.blob);
    std::vector<std::int32_t> ids;
    for (std::size_t expert = 0; expert < counts.size(); ++expert)
        ids.insert(ids.end(), counts[expert], static_cast<std::int32_t>(expert));
    const int tokens = static_cast<int>(ids.size());
    std::vector<float> values(tokens * input_width);
    for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = float(int((index * 13 + 7) % 29) - 14) / 1024.0f;
    const auto x = mlx::core::astype(mlx::core::array(values.begin(),
        mlx::core::Shape{tokens, input_width}), mlx::core::float16);
    const auto routes = mlx::core::array(ids.begin(), mlx::core::Shape{tokens, 1});
    const auto order = mlx::core::astype(mlx::core::argsort(mlx::core::reshape(
        routes, mlx::core::Shape{tokens})), mlx::core::int32);
    const auto inverse = mlx::core::argsort(order);
    for (const int block_rows : {64, 80, 96}) {
        const auto plan = weight.build_grouped_mmq_plan(routes, order, block_rows);
        const auto plain = evaluated_floats(mlx::core::take(weight.routed_matmul_sorted(
            x, routes, order, false, false, 0.0f, &plan), inverse, 0));
        const auto fused = evaluated_floats(mlx::core::take(weight.routed_matmul_sorted(
            x, routes, order, false, true, 0.0f, &plan), inverse, 0));
        for (int token = 0; token < tokens; ++token) {
            const std::vector<float> source(values.begin() + token * input_width,
                values.begin() + (token + 1) * input_width);
            for (int row = 0; row < output_width; ++row)
                require_close(plain[token * output_width + row],
                    routed_vq_dot(source, fixture, ids[token], row), 2e-3f);
            for (int row = 0; row < output_width / 2; ++row) {
                const float gate = routed_vq_dot(source, fixture, ids[token], row);
                const float up = routed_vq_dot(source, fixture, ids[token], row + output_width / 2);
                require_close(fused[token * (output_width / 2) + row],
                    gate / (1.0f + std::exp(-gate)) * up, 2e-3f);
            }
        }
    }
}

void test_grouped_vq_decoder_tail_prefill(
    int input_width = 640,
    int output = 16) {
    auto fixture = make_vq_moe_fixture({
        make_npq(output, input_width, true),
        make_npq(output, input_width, false),
        make_nvq1_s(output, input_width),
        make_nvq1_l(output, input_width),
        make_plain_nvq(output, input_width),
        make_plain_nvq(output, input_width, true),
        make_plain_nvq3(output, input_width),
        make_jsc_nvq(
            output,
            input_width,
            "NVQ3J-512",
            3,
            4,
            9),
        make_jsc_nvq(
            output,
            input_width,
            "NVQ2J-L",
            4,
            8,
            10),
        make_jsc_nvq(
            output,
            input_width,
            "NVQ3J-L",
            6,
            4,
            10),
        make_jsc_nvq(
            output,
            input_width,
            "NVQ2J-XL",
            5,
            8,
            12),
        make_jsc_nvq(
            output,
            input_width,
            "NVQ2J-XL",
            5,
            8,
            12,
            true),
    });
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "mixed VQ decoder tail fixture must support grouped prefill");
    const auto exercise = [&](int tokens, int first_expert = 0, int forced_block_rows = 0) {
        std::vector<float> input(tokens * input_width);
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = static_cast<float>(
                static_cast<int>((index * 13 + 7) % 29) - 14)
                / 1024.0f;
        }
        std::vector<std::int32_t> ids(tokens);
        for (int token = 0; token < tokens; ++token) {
            ids[token] = (first_expert + token) % fixture.experts;
        }
        const auto x = mlx::core::astype(mlx::core::array(
            input.begin(), mlx::core::Shape{tokens, input_width}), mlx::core::float16);
        const auto routes = mlx::core::array(ids.begin(), mlx::core::Shape{tokens, 1});
        auto result = weight.routed_matmul(x, routes);
        if (forced_block_rows > 0 || (tokens >= 1025
            && weight.recommended_grouped_mmq_block_rows(tokens, false) != 32)) {
            auto order = mlx::core::astype(mlx::core::argsort(
                mlx::core::reshape(routes, mlx::core::Shape{tokens})), mlx::core::int32);
            const int block_rows = forced_block_rows > 0 ? forced_block_rows
                : weight.recommended_grouped_mmq_block_rows(tokens, false);
            auto plan = weight.build_grouped_mmq_plan(routes, order, block_rows);
            result = mlx::core::take(weight.routed_matmul_sorted(
                x, routes, order, false, false, 0.0f, &plan),
                mfq::metal::moe_inverse_permutation(order), 0);
        }
        const auto actual = evaluated_floats(result);
        for (int token = 0; token < tokens; ++token) {
            const std::vector<float> source(
                input.begin() + token * input_width,
                input.begin() + (token + 1) * input_width);
            for (int row = 0; row < output; ++row) {
                require_close(
                    actual[token * output + row],
                    routed_vq_dot(source, fixture, ids[token], row),
                    6e-2f);
            }
        }
    };
    // Every geometry must also exercise the single-token decoder, rather
    // than reaching only the first cohort through the prefill cases.
    for (int expert = 0; expert < fixture.experts; ++expert) {
        exercise(1, expert);
    }
    exercise(31);
    exercise(32);
    exercise(49);
    exercise(641, 0, 96);
    exercise(897, 0, 80);
    exercise(1025);
}

void test_nvq3jl_record_dispatch() {
    constexpr int input_width = 40;
    constexpr int output_width = 34;
    auto fixture = make_vq_moe_fixture({
        make_jsc_nvq(output_width, input_width, "NVQ3J", 2, 4, 8),
        make_jsc_nvq(output_width, input_width, "NVQ3J-L", 6, 4, 10, false, 4, -1),
    });
    const auto weight = mfq::metal::MlxMfeWeight::from_blob(fixture.blob);
    std::vector<float> values(input_width);
    for (int i = 0; i < input_width; ++i) values[i] = float((i * 13 + 7) % 29 - 14) / 1024.0f;
    auto shared = mlx::core::astype(mlx::core::array(values.begin(),
        mlx::core::Shape{1, input_width}), mlx::core::float16);
    auto independent = mlx::core::contiguous(mlx::core::broadcast_to(
        mlx::core::expand_dims(shared, 1), mlx::core::Shape{1, 2, input_width}));
    const std::vector<std::int32_t> ids{0, 1};
    auto routes = mlx::core::array(ids.begin(), mlx::core::Shape{1, 2});
    for (const auto& input : {shared, independent}) {
        const auto actual = evaluated_floats(weight.routed_swiglu(input, routes));
        for (int expert = 0; expert < 2; ++expert) {
            for (int row = 0; row < output_width / 2; ++row) {
                const float gate = routed_vq_dot(values, fixture, expert, row);
                const float up = routed_vq_dot(values, fixture, expert, row + output_width / 2);
                try {
                    require_close(actual[expert * output_width / 2 + row],
                        gate / (1.0f + std::exp(-gate)) * up, 1e-4f);
                } catch (const std::exception& error) {
                    throw std::runtime_error("NVQ3J-L record dispatch expert="
                        + std::to_string(expert) + " row=" + std::to_string(row)
                        + " ndim=" + std::to_string(input.ndim()) + ": " + error.what());
                }
            }
        }
    }
}

void test_nvq3jl_half_chunk(int output) {
    constexpr int input_width = 36;
    auto fixture = make_vq_moe_fixture({
        make_jsc_nvq(output, input_width, "NVQ3J-L", 6, 4, 10, false, 4, -1),
    });
    const auto weight = mfq::metal::MlxMfeWeight::from_blob(fixture.blob);
    for (int tokens : {1, 32, 65, 513}) {
        std::vector<float> values(tokens * input_width);
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(static_cast<int>((i * 13 + 7) % 29) - 14) / 1024.0f;
        }
        std::vector<std::int32_t> ids(tokens, 0);
        auto x = mlx::core::astype(mlx::core::array(values.begin(),
            mlx::core::Shape{tokens, input_width}), mlx::core::float16);
        auto routes = mlx::core::array(ids.begin(), mlx::core::Shape{tokens, 1});
        auto order = mlx::core::astype(mlx::core::argsort(
            mlx::core::reshape(routes, mlx::core::Shape{tokens})), mlx::core::int32);
        const int selected_rows = weight.recommended_grouped_mmq_block_rows(tokens, false);
        auto plan = weight.build_grouped_mmq_plan(routes, order, selected_rows);
        const auto actual = evaluated_floats(tokens == 1
            ? weight.routed_matmul(x, routes)
            : weight.routed_matmul_sorted(x, routes, order, false, false, 0.0f, &plan));
        for (int token = 0; token < tokens; ++token) {
            const std::vector<float> source(values.begin() + token * input_width,
                values.begin() + (token + 1) * input_width);
            for (int row = 0; row < output; ++row) {
                require_close(actual[token * output + row],
                    routed_vq_dot(source, fixture, 0, row), 1e-4f);
            }
        }
    }
}

void test_nint_decode_shape_profiles() {
    const std::vector<std::string> profiles{"NINTv2", "NINT4", "NINT5", "NINT6"};
    for (const int group_size : {1, 7, 23, 24, 28, 32, 48}) {
        for (const auto [output, input] : {
                 std::pair{41, 113}, std::pair{97, 65},
                 std::pair{41, 96}, std::pair{17, 11}}) {
            const auto fixture = make_moe_fixture(
                profiles, output, input, 19, group_size);
            const auto weight = mfq::metal::MlxMfeWeight::from_blob(fixture.blob);
            const auto second = make_moe_fixture(
                profiles, output, input, 37, group_size);
            const auto pair = mfq::metal::MlxMfeWeight::from_projection_blobs(
                std::array<std::span<const std::uint8_t>, 2>{fixture.blob, second.blob});
            const std::vector<std::int32_t> ids{3, 0, 2, 1};
            const auto routes = mlx::core::array(ids.begin(), mlx::core::Shape{1, 4});
            for (const bool shared : {false, true}) {
                std::vector<float> values((shared ? 1 : 4) * input);
                for (std::size_t i = 0; i < values.size(); ++i)
                    values[i] = static_cast<float>(static_cast<int>((i * 17 + 3) % 31) - 15) / 1024.0f;
                const auto x = mlx::core::astype(mlx::core::array(
                    values.begin(), shared ? mlx::core::Shape{1, input}
                                          : mlx::core::Shape{1, 4, input}),
                    mlx::core::float16);
                const auto plain = evaluated_floats(weight.routed_matmul(x, routes));
                const auto fused = evaluated_floats(pair.routed_swiglu(x, routes));
                for (int route = 0; route < 4; ++route) {
                    for (int row = 0; row < output; ++row) {
                        float gate = 0.0f, up = 0.0f;
                        for (int column = 0; column < input; ++column) {
                            const auto index = (ids[route] * output + row) * input + column;
                            const auto activation = values[(shared ? 0 : route * input) + column];
                            gate += activation * fixture.dense[index];
                            up += activation * second.dense[index];
                        }
                        require_close(plain[route * output + row], gate, 2e-3f);
                        require_close(fused[route * output + row],
                            gate / (1.0f + std::exp(-gate)) * up, 2e-3f);
                    }
                }
            }
        }
    }
}

void test_grouped_nint_mmq_prefill(
    int tokens = 513,
    int group_size = 24,
    int input_width = 96,
    int output = 48) {
    // Cross the heterogeneous NAX threshold on supported Apple GPUs while
    // retaining the same reference coverage on compatibility-only devices.
    std::vector<std::string> profiles{
        "NINTv2", "NINT1", "NINT2", "NINT3", "NINT4", "NINT5",
        "NINT6", "NINT7", "NINT8"};
    if (input_width % 32 == 0) profiles.push_back("NINT8-0");
    const int routes = tokens == 1 ? static_cast<int>(profiles.size()) : 2;
    auto fixture = make_moe_fixture(
        profiles, output, input_width, 9, group_size);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "metadata-driven NINT must support grouped prefill");
    std::vector<float> input(tokens * input_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 11 + 3) % 29) - 14)
            / 128.0f;
    }
    std::vector<std::int32_t> ids(tokens * routes);
    for (int row = 0; row < tokens * routes; ++row) {
        ids[row] = row % static_cast<int>(profiles.size());
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, input_width}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto plain = evaluated_floats(
        weight.routed_matmul(input_array, ids_array));
    const auto fused = evaluated_floats(
        weight.routed_swiglu(input_array, ids_array));
    auto reference_weights = fixture.dense;
    if (tokens > 1) {
        for (auto& value : reference_weights) {
            value = static_cast<float>(mlx::core::float16_t(value));
        }
    }
    std::vector<float> reference_dots(plain.size());
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < input_width; ++column) {
                    expected += input[token * input_width + column]
                        * reference_weights[
                            (expert * output + row) * input_width + column];
                }
                reference_dots[(token * routes + route) * output + row] = expected;
                require_close(
                    plain[(token * routes + route) * output + row],
                    tokens > 1 ? static_cast<float>(mlx::core::float16_t(expected)) : expected,
                    2e-2f);
            }
            for (int row = 0; row < output / 2; ++row) {
                const float gate = reference_dots[
                    (token * routes + route) * output + row];
                const float up = reference_dots[
                    (token * routes + route) * output + output / 2 + row];
                const float expected = gate / (1.0f + std::exp(-gate)) * up;
                require_close(
                    fused[(token * routes + route) * (output / 2) + row],
                    static_cast<float>(mlx::core::float16_t(expected)),
                    2e-2f);
            }
        }
    }
}

void test_grouped_split_nint_swiglu_prefill(
    int tokens = 49,
    int input_width = 96,
    int output = 24) {
    constexpr int routes = 2;
    const std::vector<std::string> profiles{
        "NINTv2", "NINT2", "NINT5", "NINT8",
    };
    const auto gate = make_moe_fixture(
        profiles, output, input_width, 13, 24);
    const auto up = make_moe_fixture(
        profiles, output, input_width, 37, 24);
    const auto gate_weight = mfq::metal::MlxMoeWeight::from_blob(
        gate.blob);
    const auto up_weight = mfq::metal::MlxMoeWeight::from_blob(
        up.blob);
    const auto gate_up = mfq::metal::MlxMoeWeight::from_projection_blobs(
        std::array<std::span<const std::uint8_t>, 2>{gate.blob, up.blob});
    require(
        gate_up.projections() == 2
            && gate_up.supports_grouped_mmq(),
        "split NINT Gate/Up must retain grouped prefill support");

    std::vector<float> input(tokens * input_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 17 + 9) % 31) - 15)
            / 256.0f;
    }
    std::vector<std::int32_t> ids(tokens * routes);
    for (int row = 0; row < tokens * routes; ++row) {
        ids[row] = row % static_cast<int>(profiles.size());
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, input_width}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        gate_up.routed_swiglu(input_array, ids_array));
    const auto gate_values = evaluated_floats(
        gate_weight.routed_matmul(input_array, ids_array));
    const auto up_values = evaluated_floats(
        up_weight.routed_matmul(input_array, ids_array));

    auto route_order = mlx::core::contiguous(
        mlx::core::astype(
            mlx::core::argsort(
                mlx::core::reshape(
                    ids_array,
                    mlx::core::Shape{tokens * routes})),
            mlx::core::int32));
    const auto plan = gate_up.build_grouped_mmq_plan(
        ids_array, route_order);
    const auto sorted = gate_up.routed_matmul_sorted(
        input_array,
        ids_array,
        route_order,
        false,
        true,
        0.0f,
        &plan);
    const auto planned = evaluated_floats(
        mlx::core::reshape(
            mlx::core::take(
                sorted,
                mlx::core::argsort(route_order),
                0),
            mlx::core::Shape{tokens, routes, output}));
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float gate_value = gate_values[index];
        const float expected = gate_value
            / (1.0f + std::exp(-gate_value))
            * up_values[index];
        require_close(actual[index], expected, 2e-2f);
        require_close(planned[index], actual[index], 2e-3f);
    }
}

void test_shared_nint2_prefill() {
    constexpr int tokens = 1024;
    constexpr int output = 24;
    constexpr int input_width = 96;
    const auto fixture = make_moe_fixture(
        {"NINT2"}, output, input_width, 23);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "NINT2-16 must support grouped prefill");
    std::vector<float> input(tokens * input_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 17 + 7) % 31) - 15)
            / 128.0f;
    }
    const std::vector<std::int32_t> ids(tokens, 0);
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, input_width}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, 1});
    const auto plain = evaluated_floats(
        weight.routed_matmul(input_array, ids_array));
    const auto fused = evaluated_floats(
        weight.routed_swiglu(input_array, ids_array));
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < output; ++row) {
            float expected = 0.0f;
            for (int column = 0; column < input_width; ++column) {
                expected += input[token * input_width + column]
                    * fixture.dense[row * input_width + column];
            }
            require_close(
                plain[token * output + row], expected, 2e-2f);
        }
        for (int row = 0; row < output / 2; ++row) {
            const float gate = plain[token * output + row];
            const float up = plain[
                token * output + output / 2 + row];
            require_close(
                fused[token * (output / 2) + row],
                gate / (1.0f + std::exp(-gate)) * up,
                2e-2f);
        }
    }
}

void test_shared_nint5_group28_prefill() {
    constexpr int output = 16;
    constexpr int input_width = 640;
    const auto fixture = make_moe_fixture(
        {"NINT5"}, output, input_width, 37, 28);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "NINT5-28 must support grouped prefill");
    const auto exercise = [&](int tokens) {
        std::vector<float> input(tokens * input_width);
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = static_cast<float>(
                static_cast<int>((index * 19 + 11) % 37) - 18)
                / 256.0f;
        }
        const std::vector<std::int32_t> ids(tokens, 0);
        const auto actual = evaluated_floats(
            weight.routed_matmul(
                mlx::core::astype(
                    mlx::core::array(
                        input.begin(),
                        mlx::core::Shape{tokens, input_width}),
                    mlx::core::float16),
                mlx::core::array(
                    ids.begin(), mlx::core::Shape{tokens, 1})));
        for (int token = 0; token < tokens; ++token) {
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < input_width; ++column) {
                    expected += input[token * input_width + column]
                        * fixture.dense[row * input_width + column];
                }
                require_close(
                    actual[token * output + row], expected, 8e-2f);
            }
        }
    };
    exercise(49);
    exercise(1025);
}

void test_shared_nint8_mixed_prefill() {
    constexpr int tokens = 49;
    constexpr int routes = 2;
    constexpr int output = 24;
    constexpr int input_width = 640;
    const auto fixture = make_moe_fixture(
        {"NINT8", "F16"}, output, input_width, 31, 48);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "mixed NINT8-48/F16 MFE must support grouped prefill");
    std::vector<float> input(tokens * input_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 13 + 5) % 31) - 15)
            / 256.0f;
    }
    std::vector<std::int32_t> ids(tokens * routes);
    for (int index = 0; index < tokens * routes; ++index) {
        ids[index] = index % 2;
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, input_width}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, ids_array));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < input_width; ++column) {
                    expected += input[token * input_width + column]
                        * fixture.dense[
                            (expert * output + row) * input_width + column];
                }
                require_close(
                    actual[(token * routes + route) * output + row],
                    expected,
                    8e-2f);
            }
        }
    }
}

void test_shared_nint_mapped_and_packed_routes() {
    constexpr int tokens = 3;
    constexpr int routes = 2;
    constexpr int output = 16;
    constexpr int input_width = 96;
    const auto fixture = make_moe_fixture(
        {"NINT3", "NINT6"}, output, input_width, 43, 24);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "mapped NINT routes must retain grouped-prefill support");

    std::vector<float> input(tokens * input_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 17 + 9) % 31) - 15)
            / 128.0f;
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, input_width}),
        mlx::core::float16);

    const std::vector<std::int32_t> logical_ids{
        2, 1,
        1, 2,
        2, 1,
    };
    const std::vector<std::int32_t> page_table{-1, 1, 0};
    const std::vector<std::int32_t> mapped_experts{
        0, 1,
        1, 0,
        0, 1,
    };
    const auto mapped = evaluated_floats(
        weight.routed_matmul_mapped(
            input_array,
            mlx::core::array(
                logical_ids.begin(),
                mlx::core::Shape{tokens, routes}),
            mlx::core::array(
                page_table.begin(),
                mlx::core::Shape{static_cast<int>(page_table.size())})));

    const std::vector<std::int32_t> packed_ids{
        263, 519,
        519, 263,
        263, 519,
    };
    const auto packed = evaluated_floats(
        weight.routed_matmul_packed(
            input_array,
            mlx::core::array(
                packed_ids.begin(),
                mlx::core::Shape{tokens, routes})));

    const auto validate = [&](const std::vector<float>& actual) {
        for (int token = 0; token < tokens; ++token) {
            for (int route = 0; route < routes; ++route) {
                const int expert = mapped_experts[token * routes + route];
                for (int row = 0; row < output; ++row) {
                    float expected = 0.0f;
                    for (int column = 0; column < input_width; ++column) {
                        expected += input[token * input_width + column]
                            * fixture.dense[
                                (expert * output + row) * input_width
                                + column];
                    }
                    require_close(
                        actual[(token * routes + route) * output + row],
                        expected,
                        6e-2f);
                }
            }
        }
    };
    validate(mapped);
    validate(packed);
}

void test_grouped_dense_quad_tail_prefill(int output = 24, int input_width = 642) {
    constexpr int tokens = 1025;
    constexpr int routes = 2;
    constexpr int experts = 2;
    const auto fixture = make_moe_fixture(
        {"F16", "BF16"}, output, input_width, 37, 48);
    const auto weight = mfq::metal::MlxMoeWeight::from_blob(
        fixture.blob);
    require(
        weight.supports_grouped_mmq(),
        "mixed F16/BF16 tail fixture must support grouped prefill");
    std::vector<float> input(tokens * input_width);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(
            static_cast<int>((index * 7 + 3) % 23) - 11)
            / 256.0f;
    }
    std::vector<std::int32_t> ids(tokens * routes);
    for (int index = 0; index < tokens * routes; ++index) {
        ids[index] = index % experts;
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            input.begin(),
            mlx::core::Shape{tokens, input_width}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, ids_array));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                for (int column = 0; column < input_width; ++column) {
                    expected += input[token * input_width + column]
                        * fixture.dense[
                            (expert * output + row) * input_width + column];
                }
                require_close(
                    actual[(token * routes + route) * output + row],
                    expected,
                    1e-1f);
            }
        }
    }
}

void test_mixed_mfe_native_and_grouped_dispatch(int output = 48, bool skewed = false) {
    constexpr int experts = 4;
    constexpr int routes = 2;
    constexpr int input_width = 128;
    const auto nint = make_nint_tensor(4, output, input_width, 3, 32);
    const auto mxfp8 = make_mxfp8(output, input_width, 5);
    const auto bf16 = make_dense_tensor("BF16", output, input_width, 7);
    const auto f16 = make_dense_tensor("F16", output, input_width, 11);

    std::vector<std::uint8_t> blob;
    append_magic(blob, "NIM2");
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input_width);
    append<std::uint32_t>(blob, experts);
    auto append_pool = [&blob](
        int expert,
        std::string_view dtype,
        const std::vector<std::uint8_t>& payload) {
        append<std::uint32_t>(blob, 1);
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(dtype.size()));
        append<std::uint64_t>(blob, payload.size());
        append<std::uint64_t>(blob, 0);
        append<std::int32_t>(blob, expert);
        blob.insert(blob.end(), dtype.begin(), dtype.end());
        append_bytes(blob, payload);
    };
    // Reverse and mix pool order so descriptor-local and global expert rows
    // cannot accidentally coincide.
    append_pool(3, "F16", f16.blob);
    append_pool(1, "MXFP8", mxfp8.blob);
    append_pool(0, "NINT4", nint.blob);
    append_pool(2, "BF16", bf16.blob);

    std::vector<float> dense(
        static_cast<std::size_t>(experts) * output * input_width);
    const auto copy_expert = [&](int expert, const std::vector<float>& source) {
        std::copy(
            source.begin(),
            source.end(),
            dense.begin() + static_cast<std::ptrdiff_t>(
                static_cast<std::size_t>(expert) * output * input_width));
    };
    copy_expert(0, nint.dense);
    copy_expert(1, mxfp8.dense);
    copy_expert(2, bf16.dense);
    copy_expert(3, f16.dense);

    const auto weight = mfq::metal::MlxMoeWeight::from_blob(blob);
    require(
        weight.supports_grouped_mmq(),
        "metadata-driven NINT must support mixed grouped prefill");
    if (mfq::metal::mlx_metal_nax_disabled() && output >= 64) {
        for (const auto& [mean_rows, block_rows] : {
                 std::pair{31, 32}, std::pair{32, 32}, std::pair{33, 64},
                 std::pair{64, 64}, std::pair{65, 96}, std::pair{96, 96},
                 std::pair{97, 96}, std::pair{128, 96}})
            require(weight.recommended_grouped_mmq_block_rows(mean_rows * experts, false) == block_rows,
                "portable grouped MMQ did not select adaptive row tiles");
    }
    std::optional<std::pair<std::string, std::string>> plan_signature;
    const auto exercise = [&](int tokens) {
        std::vector<float> input(
            static_cast<std::size_t>(tokens) * input_width);
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = static_cast<float>(
                static_cast<int>((index * 13 + 5) % 31) - 15)
                / 1024.0f;
        }
        std::vector<std::int32_t> ids(tokens * routes);
        for (int row = 0; row < tokens * routes; ++row) {
            ids[row] = skewed && row < 3 * tokens * routes / 4
                ? row % 2 : (row * 3 + row / 5) % experts;
        }
        const auto input_array = mlx::core::astype(
            mlx::core::array(
                input.begin(),
                mlx::core::Shape{tokens, input_width}),
            mlx::core::float16);
        const auto ids_array = mlx::core::array(
            ids.begin(),
            mlx::core::Shape{tokens, routes});
        if (tokens >= 25) {
            const auto order = mlx::core::argsort(mlx::core::reshape(
                ids_array, mlx::core::Shape{tokens * routes}));
            const auto plan = weight.build_grouped_mmq_plan(ids_array, order);
            const auto& primitive = plan.block_meta.primitive();
            require(std::string(primitive.name()) == "CustomKernel",
                "grouped block plan must use a custom kernel");
            const auto state = static_cast<const mlx::core::fast::CustomKernel&>(
                primitive).state();
            const auto signature = std::pair{std::get<0>(state), std::get<1>(state)};
            require(!plan_signature || *plan_signature == signature,
                "grouped block plan recompiles for a different route count");
            plan_signature = signature;
            mlx::core::eval(plan.block_meta, plan.block_count);
        }
        const auto plain = evaluated_floats(
            weight.routed_matmul(input_array, ids_array));
        const auto fused = evaluated_floats(
            weight.routed_swiglu(input_array, ids_array));
        for (int token = 0; token < tokens; ++token) {
            for (int route = 0; route < routes; ++route) {
                const int expert = ids[token * routes + route];
                for (int row = 0; row < output; ++row) {
                    float expected = 0.0f;
                    for (int column = 0; column < input_width; ++column) {
                        expected += input[token * input_width + column]
                            * dense[
                                (expert * output + row) * input_width
                                + column];
                    }
                    require_close(
                        plain[(token * routes + route) * output + row],
                        expected,
                        8e-2f);
                }
                for (int row = 0; row < output / 2; ++row) {
                    const float gate = plain[
                        (token * routes + route) * output + row];
                    const float up = plain[
                        (token * routes + route) * output + output / 2 + row];
                    require_close(
                        fused[(token * routes + route) * (output / 2) + row],
                        gate / (1.0f + std::exp(-gate)) * up,
                        8e-2f);
                }
            }
        }
    };
    exercise(3);
    exercise(25);
    exercise(30);
    exercise(31);
    exercise(33);
    exercise(49);
    exercise(54);
    exercise(513);
}

void test_multiple_nint_pools_share_cpp_dispatch(int tokens = 3) {
    constexpr int experts = 4;
    constexpr int routes = 2;
    constexpr int output = 9;
    constexpr int input = 48;
    std::vector<PoolFixture> pools;
    pools.push_back({
        {2, 0},
        "NINT",
        make_nint_tensor(2, 2 * output, input, 17, 24),
        {},
    });
    pools.push_back({
        {3, 1},
        "NINT",
        make_nint_tensor(6, 2 * output, input, 29, 28),
        {},
    });
    auto blob = make_raw_nim2(experts, output, input, pools);
    std::copy_n("MFE1", 4, blob.begin());

    const auto weight = mfq::metal::MlxMfeWeight::from_blob(blob);
    std::vector<float> source(static_cast<std::size_t>(tokens) * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(
            static_cast<int>((index * 7 + 3) % 29) - 14) / 256.0f;
    }
    std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens) * routes);
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = (i * 3) % experts;
    const auto input_array = mlx::core::astype(
        mlx::core::array(source.begin(), mlx::core::Shape{tokens, input}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto actual = evaluated_floats(
        weight.routed_matmul(input_array, ids_array));

    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            const int expert = ids[token * routes + route];
            const PoolFixture* owner = nullptr;
            std::size_t local = 0;
            for (const auto& pool : pools) {
                const auto found = std::find(
                    pool.expert_ids.begin(), pool.expert_ids.end(), expert);
                if (found != pool.expert_ids.end()) {
                    owner = &pool;
                    local = static_cast<std::size_t>(
                        found - pool.expert_ids.begin());
                    break;
                }
            }
            require(owner != nullptr, "test expert has no NINT pool");
            for (int row = 0; row < output; ++row) {
                float expected = 0.0f;
                const auto weight_row =
                    (local * output + static_cast<std::size_t>(row)) * input;
                for (int column = 0; column < input; ++column) {
                    expected += source[token * input + column]
                        * owner->tensor.dense[weight_row + column];
                }
                require_close(
                    actual[(token * routes + route) * output + row],
                    expected,
                    3e-3f);
            }
        }
    }
}

void test_grouped_mxfp4_vq_mmq_prefill(int output = 17, int input = 96) {
    constexpr int experts = 2;
    constexpr int tokens = 37;
    constexpr int routes = 2;
    const auto mx_weight = make_mxfp4(output, input);
    const auto vq_weight = make_jsc_nvq(output, input);
    std::vector<std::uint8_t> blob;
    append_magic(blob, "NIM2");
    append<std::uint32_t>(blob, experts);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, experts);
    auto append_pool = [&blob](
        int expert,
        std::string_view dtype,
        const std::vector<std::uint8_t>& payload,
        const std::vector<std::uint8_t>& runtime) {
        append<std::uint32_t>(blob, 1);
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(dtype.size()));
        append<std::uint64_t>(blob, payload.size());
        append<std::uint64_t>(blob, runtime.size());
        append<std::int32_t>(blob, expert);
        blob.insert(blob.end(), dtype.begin(), dtype.end());
        append_bytes(blob, runtime);
        append_bytes(blob, payload);
    };
    append_pool(0, "MXFP4", mx_weight.blob, {});
    append_pool(
        1,
        vq_weight.dtype,
        vq_weight.blob,
        vq_weight.runtime);

    const auto weight =
        mfq::metal::MlxMoeWeight::from_blob(blob);
    require(
        weight.supports_grouped_mmq(),
        "mixed MXFP4/VQ MFE must support grouped matrix prefill");
    std::vector<float> source(tokens * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(
            static_cast<int>((index * 11 + 3) % 31) - 15) / 128.0f;
    }
    std::vector<std::int32_t> ids(tokens * routes);
    for (std::size_t index = 0; index < ids.size(); ++index) {
        ids[index] = static_cast<std::int32_t>((index / 5) & 1u);
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            source.begin(),
            mlx::core::Shape{tokens, input}),
        mlx::core::float16);
    const auto ids_array = mlx::core::array(
        ids.begin(),
        mlx::core::Shape{tokens, routes});
    const auto direct = evaluated_floats(
        weight.routed_matmul(input_array, ids_array));
    const auto order = mlx::core::contiguous(
        mlx::core::astype(
            mlx::core::argsort(
                mlx::core::reshape(
                    ids_array,
                    mlx::core::Shape{tokens * routes})),
            mlx::core::int32));
    const auto plan = weight.build_grouped_mmq_plan(
        ids_array,
        order,
        weight.recommended_grouped_mmq_block_rows(tokens * routes, false));
    const auto matrix = evaluated_floats(
        mlx::core::take(
            weight.routed_matmul_sorted(
                input_array,
                ids_array,
                order,
                false,
                false,
                0.0f,
                &plan),
            mlx::core::argsort(order),
            0));
    require(
        matrix.size() == direct.size(),
        "mixed MXFP4/VQ grouped prefill output size mismatch");
    for (std::size_t index = 0; index < direct.size(); ++index) {
        require_close(matrix[index], direct[index], 1e-2f);
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
        "malformed MFE was accepted: " + label);
}

void test_concatenate_resident_mfe_experts() {
    constexpr int experts = 3;
    constexpr int tokens = 17;
    constexpr int routes = 2;
    constexpr int output = 11;
    constexpr int input = 96;
    const auto nint = make_nint_tensor(5, output, input, 17, 24);
    const auto mx = make_mxfp4(output, input);
    const auto vq = make_jsc_nvq(output, input);
    const TensorFixture mx_tensor{
        mx.blob, mx.dense, output, input};
    const TensorFixture vq_tensor{
        vq.blob, vq.dense, output, input};

    const std::vector<PoolFixture> full_pools{
        {{0}, "NINT5", nint, {}},
        {{1}, "MXFP4", mx_tensor, {}},
        {{2}, vq.dtype, vq_tensor, vq.runtime},
    };
    const auto full = mfq::metal::MlxMoeWeight::from_blob(
        make_raw_nim2(experts, output, input, full_pools));
    const auto combined =
        mfq::metal::MlxMoeWeight::concatenate_experts({
            mfq::metal::MlxMoeWeight::from_blob(
                make_raw_nim2(
                    1, output, input,
                    {{{0}, "NINT5", nint, {}}})),
            mfq::metal::MlxMoeWeight::from_blob(
                make_raw_nim2(
                    1, output, input,
                    {{{0}, "MXFP4", mx_tensor, {}}})),
            mfq::metal::MlxMoeWeight::from_blob(
                make_raw_nim2(
                    1, output, input,
                    {{{0}, vq.dtype, vq_tensor, vq.runtime}})),
        });
    require(
        combined.experts() == experts
            && combined.projections() == 1
            && combined.supports_grouped_mmq(),
        "resident MFE expert concatenation metadata mismatch");

    std::vector<float> source(
        static_cast<std::size_t>(tokens) * input);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(
            static_cast<int>((index * 19 + 7) % 37) - 18) / 256.0f;
    }
    std::vector<std::int32_t> ids(
        static_cast<std::size_t>(tokens) * routes);
    for (std::size_t index = 0; index < ids.size(); ++index) {
        ids[index] = static_cast<std::int32_t>((index * 5 + 1) % experts);
    }
    const auto input_array = mlx::core::astype(
        mlx::core::array(
            source.begin(), mlx::core::Shape{tokens, input}),
        mlx::core::float16);
    const auto id_array = mlx::core::array(
        ids.begin(), mlx::core::Shape{tokens, routes});
    const auto expected = evaluated_floats(
        full.routed_matmul(input_array, id_array));
    const auto actual = evaluated_floats(
        combined.routed_matmul(input_array, id_array));
    require(
        actual.size() == expected.size(),
        "resident MFE expert concatenation output shape mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::fabs(actual[index] - expected[index]) > 1e-6f) {
            const auto route = index / output;
            throw std::runtime_error(
                "resident MFE expert concatenation mismatch: expert="
                + std::to_string(ids[route])
                + " actual=" + std::to_string(actual[index])
                + " expected=" + std::to_string(expected[index]));
        }
    }
    std::vector<mfq::metal::MlxMfeWeight> pages;
    for (int expert = 0; expert < 75; ++expert) {
        const std::array selected{expert % experts};
        pages.push_back(full.select_experts(selected));
    }
    const auto repeated = mfq::metal::MlxMfeWeight::concatenate_experts(pages);
    for (const int rows : {1, 2, 6, 17}) {
        std::vector<std::int32_t> local(rows * routes), global(rows * routes);
        for (std::size_t index = 0; index < local.size(); ++index) {
            local[index] = static_cast<std::int32_t>((index * 37 + 74) % pages.size());
            global[index] = local[index] % experts;
        }
        const auto x = mlx::core::full({rows, input}, 0.125f, mlx::core::float16);
        require(evaluated_floats(repeated.routed_matmul(x,
            mlx::core::array(local.begin(), mlx::core::Shape{rows, routes})))
            == evaluated_floats(full.routed_matmul(x,
                mlx::core::array(global.begin(), mlx::core::Shape{rows, routes}))),
            "many-expert packed assembly changed values or tail bytes");
    }
}

void test_streamed_mfe_read_validation() {
    constexpr int experts = 4, rows = 16, input = 96;
    for (bool nvq : {false, true}) for (bool prefill : {false, true}) {
        const auto mx = make_mxfp4(experts * rows, input);
        const auto vq = make_jsc_nvq(experts * rows, input);
        auto payload = nvq ? vq.blob : mx.blob;
        if (nvq) {
            const std::uint16_t nan = 0x7e00;
            std::memcpy(payload.data() + 40 + 64 + 2 * 256 * 8 + rows * 2, &nan, 2);
        } else payload[56 + experts * rows * input / 2 + rows * input / 32] = 255;
        const auto wrap = [&](const std::vector<std::uint8_t>& bytes) {
            return make_raw_nim2(experts, rows, input, {
                {{0, 1, 2, 3}, nvq ? vq.dtype : "MXFP4", {bytes, {}, experts * rows, input}, {}}});
        };
        const auto good = wrap(nvq ? vq.blob : mx.blob);
        const TemporaryMfq file({{"bad", "MFE", wrap(payload)}, {"good", "MFE", good}});
        const mfq::metal::MfqContainer model(file.path());
        mfq::metal::MlxMfeOffloadCache cache(model, 1u << 24);
        require(cache.can_stream("bad"), "invalid-value fixture did not reach streaming parser");
        bool rejected = false;
        try { (void)cache.grouped_mfe("bad", {1, 2}, {}, prefill); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "streamed SSD path bypassed metadata validation");
        require(cache.cached_expert_count() == 0 && cache.resident_packed_bytes() == 0,
            "failed streamed SSD read published a partial expert");
        auto recovered = cache.grouped_mfe("good", {0, 1, 2, 3}, {}, prefill);
        const auto full = mfq::metal::MlxMfeWeight::from_blob(good);
        auto x = mlx::core::full({2, input}, 0.125f, mlx::core::float16);
        auto ids = mlx::core::array({0, 1, 2, 3}, mlx::core::Shape{2, 2});
        require(evaluated_floats(recovered.routed_matmul(x, ids)) ==
            evaluated_floats(full.routed_matmul(x, ids)), "failed SSD read corrupted subsequent preparation");
    }
    const auto mx = make_mxfp4(experts * rows, input);
    const auto blob = make_raw_nim2(experts, rows, input,
        {{{0, 1, 2, 3}, "MXFP4", {mx.blob, mx.dense, experts * rows, input}, {}}});
    const TemporaryMfq file({{"short", "MFE", blob}});
    const mfq::metal::MfqContainer model(file.path());
    mfq::metal::MlxMfeOffloadCache cache(model, 1u << 24);
    require(cache.can_stream("short"), "short-read fixture failed preparation");
    std::filesystem::resize_file(file.path(), model.record("short").offset + blob.size() / 2);
    bool rejected = false;
    try { (void)cache.grouped_mfe("short", {1, 2}); }
    catch (const std::exception&) { rejected = true; }
    require(rejected && cache.cached_expert_count() == 0 && cache.resident_packed_bytes() == 0,
        "short SSD read was silently accepted or retained a partial expert");
}

void test_streamed_mfe_overlapped_read() {
    constexpr int experts = 4, rows = 16, input = 96;
    const auto mx = make_mxfp4(experts * rows, input);
    const auto vq = make_nvq1_s(experts * rows, input);
    const auto gate = make_raw_nim2(experts, rows, input,
        {{{0, 1, 2, 3}, "MXFP4", {mx.blob, mx.dense, experts * rows, input}, {}}});
    const auto up = make_raw_nim2(experts, rows, input,
        {{{0, 1, 2, 3}, vq.dtype, {vq.blob, vq.dense, experts * rows, input}, vq.runtime}});
    const std::array<std::string, 3> names{"z_gate", "a_up", "m_down"};
    const TemporaryMfq file({{names[0], "MFE", gate}, {names[1], "MFE", up}, {names[2], "MFE", gate}});
    const mfq::metal::MfqContainer model(file.path());
    mfq::metal::MlxMfeOffloadCache cache(model, 0);
    const std::vector<std::int32_t> active{3, 1};
    std::vector<mfq::metal::MlxMfeWeight> prefix;
    const auto initial_temporary = mfq::metal::MlxResidentBudgetScope::temporary_bytes();
    mfq::metal::MlxResidentBudgetScope budget([](std::size_t) {});
    const auto loaded = cache.grouped_mfe(names, active, {}, false,
        [&](const auto& value) { prefix = value; });
    require(prefix.size() == 2 && loaded.size() == 3, "overlapped SSD prefix has the wrong size");
    require(mfq::metal::MlxResidentBudgetScope::temporary_bytes() == initial_temporary,
        "overlapped SSD read leaked its caller's memory reservation");
    const bool parallel = std::thread::hardware_concurrency() >= 6;
    require(cache.reuse_stats().overlapped_read_batches == static_cast<std::size_t>(parallel),
        "overlapped SSD read did not report its dispatch");
    require(cache.reuse_stats().parallel_body_read_batches ==
        (mfq::metal::mlx_reference_enabled() ? 0 : active.size() * names.size()),
        "overlapped SSD reads did not include both MXFP4 values and scales");
    auto x = mlx::core::full({1, input}, 0.125f, mlx::core::float16);
    auto local = mlx::core::array({0, 1}, mlx::core::Shape{1, 2});
    auto global = mlx::core::array(active.begin(), mlx::core::Shape{1, 2});
    for (int index = 0; index < 3; ++index) {
        const auto reference = mfq::metal::MlxMfeWeight::from_blob(index == 1 ? up : gate);
        require(evaluated_floats(loaded[index].routed_matmul(x, local)) ==
            evaluated_floats(reference.routed_matmul(x, global)), "overlapped SSD read changed projection order or values");
    }
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool rejected = false;
        try {
            (void)cache.grouped_mfe(names, active, {}, false,
                [](const auto&) { throw std::runtime_error("prefix failure"); });
        } catch (const std::runtime_error& error) { rejected = std::string(error.what()) == "prefix failure"; }
        require(rejected && mfq::metal::MlxResidentBudgetScope::temporary_bytes() == initial_temporary,
            "overlapped SSD read swallowed a callback failure or leaked its reservation");
        require(cache.grouped_mfe(names, active).size() == 3, "overlapped SSD callback failure left an in-flight load");
    }
    std::atomic<int> callbacks{0};
    const auto concurrent = [&] {
        return cache.grouped_mfe(names, active, {}, false, [&](const auto&) { ++callbacks; });
    };
    auto first = std::async(std::launch::async, concurrent);
    auto second = std::async(std::launch::async, concurrent);
    require(first.get().size() == 3 && second.get().size() == 3 && callbacks == 2,
        "concurrent overlapped SSD reads did not complete");

    for (bool truncate : {false, true}) {
        auto invalid = mx.blob;
        if (!truncate) invalid[56 + experts * rows * input / 2 + rows * input / 32] = 255;
        const auto bad = make_raw_nim2(experts, rows, input,
            {{{0, 1, 2, 3}, "MXFP4", {invalid, {}, experts * rows, input}, {}}});
        const TemporaryMfq broken({{names[0], "MFE", gate}, {names[1], "MFE", up},
            {names[2], "MFE", gate}, {"bad_down", "MFE", bad}});
        const mfq::metal::MfqContainer source(broken.path());
        mfq::metal::MlxMfeOffloadCache failed(source, 0);
        const std::array<std::string, 3> bad_names{names[0], names[1], "bad_down"};
        for (const auto& name : bad_names) require(failed.can_stream(name), "broken overlap fixture failed layout preparation");
        if (truncate) std::filesystem::resize_file(broken.path(), source.record("bad_down").offset + bad.size() / 2);
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool rejected = false;
            try { (void)failed.grouped_mfe(bad_names, active, {}, false, [](const auto&) {}); }
            catch (const std::runtime_error&) { rejected = true; }
            require(rejected && mfq::metal::MlxResidentBudgetScope::temporary_bytes() == initial_temporary,
                "overlapped SSD read accepted corrupt data or retained an in-flight reservation");
            require(failed.grouped_mfe(names, active).size() == 3,
                "overlapped SSD read did not recover after a reader failure");
        }
    }
}

void test_streamed_mfe_parallel_body_reads() {
    constexpr int experts = 4, rows = 16, input = 96;
    const auto vq = make_nvq1_s(experts * rows, input);
    const auto blob = make_raw_nim2(experts, rows, input,
        {{{0, 1, 2, 3}, vq.dtype, {vq.blob, vq.dense, experts * rows, input}, vq.runtime}});
    const TemporaryMfq file({{"weight", "MFE", blob}});
    const mfq::metal::MfqContainer model(file.path());
    const auto full = mfq::metal::MlxMfeWeight::from_blob(blob);
    const auto x = mlx::core::full({1, input}, 0.125f, mlx::core::float16);
    for (bool prefill : {false, true}) for (int count : {1, 2, 3}) {
        const std::vector<std::int32_t> order{3, 1, 0};
        const std::vector<std::int32_t> active(order.begin(), order.begin() + count);
        mfq::metal::MlxMfeOffloadCache cache(model, 0);
        const auto weight = cache.grouped_mfe("weight", active, {}, prefill);
        const auto expected_batches = !mfq::metal::mlx_reference_enabled()
            && count <= 2 && (!prefill || count == 1) ? count : 0;
        require(cache.reuse_stats().parallel_body_read_batches == static_cast<std::size_t>(expected_batches),
            "parallel body readers did not respect the cold expert queue bound");
        std::vector<std::int32_t> local(count);
        std::iota(local.begin(), local.end(), 0);
        require(evaluated_floats(weight.routed_matmul(x,
            mlx::core::array(local.begin(), mlx::core::Shape{1, count}))) ==
            evaluated_floats(full.routed_matmul(x,
                mlx::core::array(active.begin(), mlx::core::Shape{1, count}))),
            "parallel body readers changed expert order or values");
        require(cache.cached_expert_count() == 0 && cache.resident_packed_bytes() == 0,
            "zero-budget parallel body read retained expert pages");
        const auto read = [&] { return cache.grouped_mfe("weight", active); };
        auto first = std::async(std::launch::async, read);
        auto second = std::async(std::launch::async, read);
        (void)first.get();
        (void)second.get();
    }
    const TemporaryMfq broken({{"good", "MFE", blob}, {"short", "MFE", blob}});
    const mfq::metal::MfqContainer source(broken.path());
    mfq::metal::MlxMfeOffloadCache failed(source, 0);
    require(failed.can_stream("short"), "parallel short-read fixture failed layout preparation");
    std::filesystem::resize_file(broken.path(), source.record("short").offset + blob.size() - 1);
    const auto initial_temporary = mfq::metal::MlxResidentBudgetScope::temporary_bytes();
    mfq::metal::MlxResidentBudgetScope budget([](std::size_t) {});
    for (int attempt = 0; attempt < 3; ++attempt) {
        bool rejected = false;
        try { (void)failed.grouped_mfe("short", {3, 1}); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected && failed.cached_expert_count() == 0
            && mfq::metal::MlxResidentBudgetScope::temporary_bytes() == initial_temporary,
            "parallel body read swallowed a short read or retained a reservation");
        (void)failed.grouped_mfe("good", {3, 1});
    }
}

void test_streamed_mfe_wide_readers() {
    constexpr int experts = 32, cohort = experts / 2, rows = 16, input = 96;
    const auto mx = make_mxfp4(cohort * rows, input);
    const auto vq = make_nvq1_s(cohort * rows, input);
    std::vector<std::int32_t> mx_ids(cohort), vq_ids(cohort), active(experts);
    std::iota(mx_ids.begin(), mx_ids.end(), 0);
    std::iota(vq_ids.begin(), vq_ids.end(), cohort);
    for (int index = 0; index < experts; ++index) active[index] = index * 13 % experts;
    const auto payload = make_raw_nim2(experts, rows, input,
        {{mx_ids, "MXFP4", {mx.blob, mx.dense, cohort * rows, input}, {}},
         {vq_ids, vq.dtype, {vq.blob, vq.dense, cohort * rows, input}, vq.runtime}});
    const std::array<std::string, 3> names{"gate", "up", "down"};
    const TemporaryMfq file({{names[0], "MFE", payload}, {names[1], "MFE", payload}, {names[2], "MFE", payload}});
    const mfq::metal::MfqContainer model(file.path());
    mfq::metal::MlxMfeOffloadCache cache(model, 0);
    const auto initial_temporary = mfq::metal::MlxResidentBudgetScope::temporary_bytes();
    mfq::metal::MlxResidentBudgetScope budget([](std::size_t) {});
    auto loaded = cache.grouped_mfe(names, active);
    require(loaded.size() == 3 && cache.cached_expert_count() == 0
        && mfq::metal::MlxResidentBudgetScope::temporary_bytes() == initial_temporary,
        "wide SSD read leaked its cache pages or memory reservation");
    auto x = mlx::core::full({1, input}, 0.125f, mlx::core::float16);
    const std::array<std::int32_t, 4> selected{0, 7, 19, 31};
    std::array<std::int32_t, 4> global_selected;
    for (std::size_t index = 0; index < selected.size(); ++index) global_selected[index] = active[selected[index]];
    const auto local = mlx::core::array(selected.begin(), mlx::core::Shape{1, 4});
    const auto global = mlx::core::array(global_selected.begin(), mlx::core::Shape{1, 4});
    const auto reference = mfq::metal::MlxMfeWeight::from_blob(payload);
    const auto expected = evaluated_floats(reference.routed_matmul(x, global));
    for (const auto& weight : loaded)
        require(evaluated_floats(weight.routed_matmul(x, local)) == expected,
            "wide SSD read changed projection values or expert order");
    auto first = std::async(std::launch::async, [&] { return cache.grouped_mfe(names, active); });
    auto second = std::async(std::launch::async, [&] { return cache.grouped_mfe(names, active); });
    require(first.get().size() == 3 && second.get().size() == 3,
        "concurrent wide SSD readers did not release in-flight loads");
    for (bool truncate : {false, true}) {
        auto invalid = mx.blob;
        if (!truncate) invalid[56 + cohort * rows * input / 2 + (cohort - 1) * rows * input / 32] = 255;
        const auto bad = make_raw_nim2(experts, rows, input,
            {{mx_ids, "MXFP4", {invalid, {}, cohort * rows, input}, {}},
             {vq_ids, vq.dtype, {vq.blob, vq.dense, cohort * rows, input}, vq.runtime}});
        const TemporaryMfq broken({{names[0], "MFE", payload}, {names[1], "MFE", payload},
            {names[2], "MFE", payload}, {"bad_down", "MFE", bad}});
        const mfq::metal::MfqContainer source(broken.path());
        mfq::metal::MlxMfeOffloadCache failed(source, 0);
        const std::array<std::string, 3> bad_names{names[0], names[1], "bad_down"};
        for (const auto& name : bad_names) require(failed.can_stream(name), "wide SSD fixture could not prepare its layout");
        if (truncate) std::filesystem::resize_file(broken.path(), source.record("bad_down").offset + bad.size() / 2);
        for (int attempt = 0; attempt < 2; ++attempt) {
            bool rejected = false;
            try { (void)failed.grouped_mfe(bad_names, active); }
            catch (const std::runtime_error&) { rejected = true; }
            require(rejected && failed.cached_expert_count() == 0
                && mfq::metal::MlxResidentBudgetScope::temporary_bytes() == initial_temporary,
                "wide SSD readers accepted corrupt data or leaked an in-flight reservation");
            require(failed.grouped_mfe(names, active).size() == 3,
                "wide SSD readers did not recover after a failed read");
        }
    }
}

void test_streamed_mfe_partition_preparation() {
    {
        constexpr int input = 96, output = 13;
        const auto mx = make_mxfp4(output, input);
        const auto nvq = make_nvq1_s(output, input);
        const auto gate = make_raw_nim2(2, output, input,
            {{{0}, "MXFP4", {mx.blob, mx.dense, output, input}, {}},
             {{1}, nvq.dtype, {nvq.blob, nvq.dense, output, input}, nvq.runtime}});
        const auto up = make_raw_nim2(2, output, input,
            {{{1}, "MXFP4", {mx.blob, mx.dense, output, input}, {}},
             {{0}, nvq.dtype, {nvq.blob, nvq.dense, output, input}, nvq.runtime}});
        const TemporaryMfq file({{"gate", "MFE", gate}, {"up", "MFE", up}});
        const mfq::metal::MfqContainer container(file.path());
        mfq::metal::MlxMfeOffloadCache cache(container, 0);
        mfq::metal::MlxKernelPreparation preparation(512, 512);
        preparation.set_routes(3);
        cache.prepare_swiglu_pair("gate", "up", 3);
        const auto count = preparation.size();
        require(count > 0, "streamed Gate/Up preparation did not collect kernels");
        cache.prepare_swiglu_pair("gate", "up", 3);
        require(preparation.size() == count, "streamed Gate/Up preparation did not deduplicate kernels");
        preparation.finish();
        require(preparation.unsupported().empty(), "streamed Gate/Up preparation has unsupported primitives");
        require(cache.cached_expert_count() == 0 && cache.resident_packed_bytes() == 0,
            "Gate/Up kernel preparation populated the expert cache");
    }
    const auto* mixed_nax = std::getenv("MFQ_METAL_MIXED_PREFILL_NAX");
    if (mfq::metal::mlx_metal_nax_disabled() || (mixed_nax
        ? std::string_view(mixed_nax) == "0" || std::string_view(mixed_nax) == "false"
            || std::string_view(mixed_nax) == "off"
        : !mfq::metal::mlx_apple_chip_starts_with("Apple M5"))) return;
    std::array<std::size_t, 2> jobs;
    for (int groups = 2; groups <= 3; ++groups) {
        std::vector<PoolFixture> pools;
        for (int group = 0; group < groups; ++group) {
            std::vector<std::int32_t> ids;
            for (int expert = group; expert < 12; expert += groups) ids.push_back(expert);
            pools.push_back({ids, "NINTv2", make_nint_v2_tensor(
                static_cast<int>(ids.size()) * 13, 96, 43, 24 + 4 * group), {}});
        }
        const TemporaryMfq file({{"mixed", "MFE", make_raw_nim2(12, 13, 96, pools)}});
        const mfq::metal::MfqContainer container(file.path());
        mfq::metal::MlxMfeOffloadCache cache(container, 0);
        mfq::metal::MlxKernelPreparation preparation(512, 512);
        preparation.set_routes(3);
        cache.prepare_projection("mixed", true);
        jobs[groups - 2] = preparation.size();
    }
    require(jobs[1] == jobs[0] + 6,
        "three-GS prefill partition preparation was deduplicated with two-GS partitions");
}

void test_deepseek_v41_ced_text_cache(bool reject_drafts = false) {
    using namespace mfq::metal;
    constexpr int hidden = 256, intermediate = 32, experts = 8, vocab = 16;
    DeepseekV41Config config;
    config.vocab = vocab;
    config.hidden = hidden;
    config.n_layers = 4;
    config.causal_encoder_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 1;
    config.head_dim = 512;
    config.rope_head_dim = 64;
    config.q_lora_rank = 16;
    config.o_lora_rank = 16;
    config.o_groups = 4;
    config.moe_inter = intermediate;
    config.n_experts = experts;
    config.n_shared = 1;
    config.top_k = 2;
    config.swiglu_limit = 10;
    config.rms_eps = 1e-6;
    config.sliding_window = 4;
    config.max_position_embeddings = 64;
    config.compress_ratios = {0, 2, 1, 1, 0, 0, 0};
    config.kv_source_layers = {1, 2};
    config.index_source_layers = {1, 2, 3};
    config.index_n_heads = 2;
    config.index_head_dim = 128;
    config.index_topk = 4;
    config.candidate_source_layer = 2;
    config.candidate_topk_blocks = 4;
    config.candidate_block_size = 2;
    config.hc_mult = 4;
    config.hc_sinkhorn_iters = 20;
    config.engram_layer_ids = {1};
    config.engram_num_embeddings = {37};
    config.engram_max_ngram_size = 2;
    config.engram_vocab_size = vocab;
    config.engram_n_heads = 1;
    config.engram_head_dim = 32;
    config.engram_compressed_vocab_size = vocab;
    config.n_mtp_layers = 3;
    config.dspark_block_size = 5;
    config.dspark_noise_token_id = 7;
    config.dspark_target_layer_ids = {1, 2, 3};
    config.dspark_markov_rank = 16;
    config.dspark_n_experts = experts;
    config.dspark_top_k = 3;
    config.validate();
    std::vector<MappedRecordFixture> records;
    const auto dense = [&](const std::string& name, std::vector<int> shape,
        bool fp32 = false, std::optional<float> constant = std::nullopt) {
        std::vector<std::uint8_t> blob;
        append<std::uint32_t>(blob, shape.size());
        std::size_t count = 1;
        for (const int dimension : shape) { append<std::int64_t>(blob, dimension); count *= dimension; }
        for (std::size_t index = 0; index < count; ++index) {
            const float value = constant.value_or(std::sin(index * 0.13f + name.size()) * 0.002f);
            if (fp32) append<float>(blob, value);
            else append<mlx::core::float16_t>(blob, mlx::core::float16_t(value));
        }
        records.push_back({name, fp32 ? "F32" : "F16", std::move(blob)});
    };
    dense("model.token_embedding.weight", {vocab, hidden});
    dense("model.output.weight", {vocab, hidden});
    const auto projection = [&](int output, int input) {
        return make_raw_nim2(experts, output, input, {
            {{0, 1, 2, 3, 4, 5, 6, 7}, "NINTv2",
                make_nint_v2_tensor(experts * output, input, 13, 16), {}},
        });
    };
    for (int layer = 0; layer < 7; ++layer) {
        const auto root = layer < 4 ? "model.block." + std::to_string(layer)
            : "predictor.stage." + std::to_string(layer - 4);
        const auto attention = root + ".attention";
        dense(attention + ".query_a.weight", {16, hidden});
        dense(attention + ".query_b.weight", {4 * 512, 16});
        dense(attention + ".key_value.weight", {512, hidden});
        dense(attention + ".output_a.weight", {4 * 16, 512});
        dense(attention + ".output_b.weight", {hidden, 4 * 16});
        dense(attention + ".query_a_norm.weight", {16}, true, 1);
        dense(attention + ".key_value_norm.weight", {512}, true, 1);
        dense(attention + ".sink", {4}, true, 0);
        if (config.is_kv_source(layer)) {
            dense(attention + ".compressor.key_value.weight", {512, hidden});
            dense(attention + ".compressor.norm.weight", {512}, true, 1);
            if (config.compress_ratios[layer] > 1)
                dense(attention + ".compressor.gate.weight", {512, hidden});
            dense(attention + ".indexer.key.weight", {128, 512});
            dense(attention + ".indexer.key_norm.weight", {128}, true, 1);
        }
        if (config.is_index_source(layer)) {
            dense(attention + ".indexer.query.weight", {2 * 128, 16});
            dense(attention + ".indexer.score.weight", {2, hidden});
        }
        for (const std::string branch : {"attention", "mlp"}) {
            const auto prefix = root + "." + branch;
            dense(prefix + ".norm.weight", {hidden}, true, 1);
            dense(prefix + ".mhc.pre.function", {24, 4 * hidden}, true);
            dense(prefix + ".mhc.pre.base", {24}, true, 0);
            dense(prefix + ".mhc.pre.scale", {3}, true, 1);
        }
        const auto mlp = root + ".mlp";
        dense(mlp + ".router.weight", {experts, hidden});
        dense(mlp + ".router.bias", {experts}, true, 0);
        dense(mlp + ".shared_expert.gate.weight", {intermediate, hidden});
        dense(mlp + ".shared_expert.up.weight", {intermediate, hidden});
        dense(mlp + ".shared_expert.down.weight", {hidden, intermediate});
        records.push_back({mlp + ".experts.gate.weight", "MFE", projection(intermediate, hidden)});
        records.push_back({mlp + ".experts.up.weight", "MFE", projection(intermediate, hidden)});
        records.push_back({mlp + ".experts.down.weight", "MFE", projection(hidden, intermediate)});
    }
    dense("predictor.stage.0.main_projection.weight", {hidden, 3 * hidden});
    dense("predictor.stage.0.main_norm.weight", {hidden}, true, 1);
    dense("predictor.stage.2.output_norm.weight", {hidden}, true, 1);
    dense("predictor.stage.2.markov.embedding.weight", {vocab, 16}, false,
        reject_drafts ? std::optional<float>{1} : std::nullopt);
    dense("predictor.stage.2.markov.output.weight", {vocab, 16}, false,
        reject_drafts ? std::optional<float>{0} : std::nullopt);
    if (reject_drafts) {
        for (int column = 0; column < 16; ++column) {
            records.back().payload.at(20 + column * 2) = 0;
            records.back().payload.at(21 + column * 2) = 0x3c;
        }
    }
    dense("predictor.stage.2.confidence.projection.weight", {1, hidden + 16}, true);
    const std::string engram = "model.block.1.associative_memory";
    dense(engram + ".query.weight", {4, hidden}, true);
    dense(engram + ".key.weight", {4, hidden}, true);
    dense(engram + ".projection.weight", {5 * hidden, 32});
    records.push_back({engram + ".embedding.weight", "NINT",
        make_nint_v2_tensor(37, 32, 19, 16).blob});
    std::vector<std::uint8_t> hash;
    for (const char c : std::string("MFQENGR1")) hash.push_back(c);
    for (const auto value : {vocab, vocab, 1, 2, 1}) append<std::uint32_t>(hash, value);
    append<std::int32_t>(hash, 0);
    append<std::int32_t>(hash, 1);
    for (const auto value : {37, 37, 0, 3, 5}) append<std::int64_t>(hash, value);
    for (int token = 0; token < vocab; ++token) append<std::int32_t>(hash, token);
    records.push_back({"__mfq_asset__/deepseek-v41-engram-v1.bin", "BLOB", std::move(hash)});
    const TemporaryMfq file(records);
    const MfqContainer model(file.path());
    for (const int mode : {-1, 0, 1}) {
        auto cache = mode < 0 ? nullptr : std::make_shared<MlxMfeOffloadCache>(model,
            mode == 0 ? std::size_t{0} : std::size_t{1} << 20);
        auto rope = mlx_yarn_tables(64, 64, 10000.0f);
        std::vector<MlxDeepseekV41Layer> layers;
        for (int layer = 0; layer < 4; ++layer)
            layers.push_back(MlxDeepseekV41Layer::load(model, config, layer, 64, rope, rope, nullptr, cache));
        auto embedding = MlxEmbedding::load(model, "model.token_embedding.weight");
        auto head = MlxLinear::load(model, "model.output.weight");
        auto dspark = MlxDeepseekV41DSpark::load_if_present(model, config, embedding, head, 64,
            nullptr, cache, 4);
        require(dspark.has_value(), "V4.1 fixture has no DSpark stages");
        MlxDeepseekV41CausalLm runtime(config, std::move(embedding),
            std::move(layers), mlx::core::ones({hidden}), std::move(head),
            MlxDeepseekV41EngramHashState::load(model, config), 64, nullptr, cache,
            std::nullopt, std::move(dspark), std::make_shared<MfqContainer>(model));
        const std::vector<std::int64_t> prompt{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2};
        const auto ids = mlx::core::array(prompt.begin(), mlx::core::Shape{1, 17}, mlx::core::int32);
        {
            const std::vector<std::int64_t> chunked_prompt(prompt.begin(), prompt.begin() + 16);
            const auto chunked_ids = mlx::core::array(chunked_prompt.begin(), mlx::core::Shape{1, 16}, mlx::core::int32);
            const auto full = runtime.prefill(chunked_ids, 8, true);
            const auto reference = evaluated_floats(mlx::core::slice(full, {0, 15, 0}, {1, 16, vocab}));
            const auto reference_state = runtime.capture_text_session_state(chunked_prompt);
            const auto next = mlx::core::array({3}, mlx::core::Shape{1, 1}, mlx::core::int32);
            const auto reference_next = evaluated_floats(runtime.decode(next));
            const auto actual = evaluated_floats(runtime.prefill(chunked_ids, 8));
            const auto state = runtime.capture_text_session_state(chunked_prompt);
            for (std::size_t index = 0; index < actual.size(); ++index)
                require_close(actual[index], reference[index], 0.002f + std::abs(reference[index]) * 0.002f);
            for (const auto& layer : state.layers)
                require(layer.attention.position == 16,
                    "V4.1 deferred Decoder left an incomplete layer position");
            const auto prefix = [](const auto& value) {
                auto end = value.shape();
                end[1] = 16;
                return evaluated_floats(mlx::core::slice(value, mlx::core::Shape(value.ndim(), 0), end));
            };
            require(state.engram_hash.position == 16
                    && state.layers[2].attention.compressed_length == 16
                    && prefix(*state.layers[2].attention.compressed_kv)
                        == prefix(*reference_state.layers[2].attention.compressed_kv)
                    && prefix(*state.layers[2].attention.index_k)
                        == prefix(*reference_state.layers[2].attention.index_k),
                "V4.1 deferred Decoder lost Encoder KV, Indexer or Engram history");
            const auto actual_next = evaluated_floats(runtime.decode(next));
            for (std::size_t index = 0; index < actual_next.size(); ++index)
                require_close(actual_next[index], reference_next[index],
                    0.002f + std::abs(reference_next[index]) * 0.002f);
            MlxSamplingParams sampling;
            sampling.temperature = 0;
            sampling.mtp_max_draft_tokens = 5;
            sampling.enable_mtp = false;
            std::vector<std::int64_t> ordinary, speculative;
            require(runtime.generate(chunked_prompt, sampling, 8,
                [&](auto token) { ordinary.push_back(token); return true; }, {}, {}, std::nullopt, 8) == 8,
                "V4.1 deferred Decoder ordinary generation did not complete");
            sampling.enable_mtp = true;
            require(runtime.generate(chunked_prompt, sampling, 8,
                [&](auto token) { speculative.push_back(token); return true; }, {}, {}, std::nullopt, 8) == 8
                    && speculative == ordinary,
                "V4.1 deferred Decoder changed DSpark generation");
            for (int repeat = 0; repeat < 2; ++repeat) {
                speculative.clear();
                require(runtime.generate(chunked_prompt, sampling, 8,
                    [&](auto token) { speculative.push_back(token); return true; }, {}, {}, chunked_prompt.size(), 8) == 8
                        && speculative == ordinary,
                    "V4.1 deferred Decoder changed persistent-prefix DSpark generation");
            }
        }
        const auto logits = evaluated_floats(runtime.prefill(ids, 8));
        require(logits.size() == vocab && std::all_of(logits.begin(), logits.end(),
            [](float value) { return std::isfinite(value); }), "V4.1 CED prefill returned invalid logits");
        auto saved = runtime.capture_text_session_state(prompt);
        require(saved.engram_hash.position == 17, "V4.1 CED truncated Engram history");
        for (const auto& layer : saved.layers)
            require(layer.attention.position == 17, "V4.1 CED left a decoder cache at the tail length");
        const auto next = mlx::core::array({3}, mlx::core::Shape{1, 1}, mlx::core::int32);
        const auto expected = evaluated_floats(runtime.decode(next));
        using PrefixCodec = MlxPagedSessionCodec<MlxDeepseekV41TextSessionState>;
        const auto persist = [&](const MlxDeepseekV41TextSessionState& state) {
            std::vector<MlxPagedPayload> blocks;
            for (std::size_t index = 0; index < (state.tokens.size() + 7) / 8; ++index)
                blocks.push_back(PrefixCodec::encode_block(state, 8, index));
            require(PrefixCodec::has_exact_boundary(blocks.back()) &&
                    PrefixCodec::has_mtp(blocks.back()) == state.dspark.has_value() &&
                    PrefixCodec::token_count(blocks.back()) == state.tokens.size(),
                "V4.1 prefix codec lost its checkpoint metadata");
            require(!PrefixCodec::has_exact_boundary(blocks.front()) &&
                    PrefixCodec::decodable_blocks({blocks.front()}) == 0,
                "V4.1 prefix accepted a non-exact ring/history boundary");
            for (int damage : {0, 1, 2}) {
                auto damaged = blocks;
                auto bytes = std::make_shared<std::vector<std::uint8_t>>(*blocks.back());
                if (damage == 0) bytes->pop_back();
                else if (damage == 1) (*bytes)[0] ^= 1;
                else bytes->push_back(0);
                damaged.back() = std::move(bytes);
                bool rejected = false;
                try { (void)PrefixCodec::decode(damaged, state.tokens, 8); }
                catch (const std::exception&) { rejected = true; }
                require(rejected, "V4.1 prefix accepted truncated or malformed payload");
            }
            return PrefixCodec::decode(blocks, state.tokens, 8);
        };
        runtime.restore_text_session_state(persist(saved));
        require(evaluated_floats(runtime.decode(next)) == expected,
            "V4.1 persisted CSA2/Engram prefix changed subsequent decode");
        runtime.restore_text_session_state(saved);
        require(evaluated_floats(runtime.decode(next)) == expected,
            "V4.1 CED prefix restore changed subsequent decode");
        if (mode == 1) {
            auto history = prompt;
            history.push_back(3);
            for (int repeat = 0; repeat < 24; ++repeat) {
                runtime.restore_text_session_state(saved);
                const auto actual = evaluated_floats(runtime.decode(next));
                for (std::size_t index = 0; index < actual.size(); ++index)
                    require_close(actual[index], expected[index], 0.002f + std::abs(expected[index]) * 0.002f);
                require(runtime.cache_position() == 18,
                    "V4.1 cached-route batching advanced the context twice");
                const auto state = runtime.capture_text_session_state(history);
                for (const auto& layer : state.layers)
                    require(layer.attention.position == 18 && !layer.attention.speculation,
                        "V4.1 cached-route batching left a pending cache transaction");
            }
            require(cache->reuse_stats().cached_route_hits > 0,
                "V4.1 replay did not exercise cached expert routing");
            const std::string missing_name = "model.block.3.mlp.experts.gate.weight";
            require(bool(cache->cached_projection(missing_name)),
                "V4.1 forced-miss fixture has no cached suffix projection");
            if (auto view = cache->cached_projection(missing_name)) {
                std::vector<std::string> preferred;
                for (int layer = 0; layer < 4; ++layer)
                    for (const std::string projection : {"gate", "up", "down"}) {
                        const auto name = "model.block." + std::to_string(layer)
                            + ".mlp.experts." + projection + ".weight";
                        if (name != missing_name) preferred.push_back(name);
                    }
                cache->set_prefill_priority(preferred);
                const auto limit = cache->cache_limit_bytes();
                while (std::count_if(view->host_expert_map.begin(), view->host_expert_map.end(),
                        [](auto expert) { return expert >= 0; }) > 1) {
                    cache->set_cache_limit(cache->resident_packed_bytes() - 1);
                    cache->set_cache_limit(limit);
                    view = cache->cached_projection(missing_name);
                    require(bool(view), "V4.1 forced miss removed the entire cached projection");
                }
                cache->set_prefill_priority({});
                const auto before = cache->reuse_stats();
                runtime.restore_text_session_state(saved);
                const auto actual = evaluated_floats(runtime.decode(next));
                for (std::size_t index = 0; index < actual.size(); ++index)
                    require_close(actual[index], expected[index], 0.002f + std::abs(expected[index]) * 0.002f);
                const auto after = cache->reuse_stats();
                require(after.cached_route_hits > before.cached_route_hits
                        && after.cached_route_misses > before.cached_route_misses,
                    "V4.1 partial cached-route replay did not preserve its valid prefix");
                const auto state = runtime.capture_text_session_state(history);
                for (const auto& layer : state.layers)
                    require(layer.attention.position == 18 && !layer.attention.speculation,
                        "V4.1 partial cached-route replay left an invalid KV transaction");
            }
        }
        if (mode < 0) {
            const auto limit = runtime.resident_full_expert_bytes() / 2;
            require(limit > 0 && runtime.ssd_expert_payload_bytes() == 0,
                "V4.1 full-resident accounting reported SSD weights");
            runtime.set_expert_cache_limit(limit);
            require(runtime.resident_full_expert_bytes() + runtime.dynamic_weight_bytes() <= limit &&
                    runtime.ssd_expert_payload_bytes() > 0 && runtime.cache_position() == 18,
                "V4.1 dynamic offload exceeded its budget or reset the context");
            runtime.restore_text_session_state(saved);
            const auto paged = evaluated_floats(runtime.decode(next));
            for (std::size_t index = 0; index < paged.size(); ++index)
                require_close(paged[index], expected[index], 0.002f + std::abs(expected[index]) * 0.002f);
        }
        const auto ceiling = runtime.resident_full_expert_bytes() + runtime.expert_cache_limit_bytes();
        const auto resident = runtime.reclaimable_expert_bytes();
        if (resident > 0) {
            mlx_trim_expert_residency(runtime, resident / 2);
            require(runtime.reclaimable_expert_bytes() <= resident / 2 &&
                    runtime.resident_full_expert_bytes() + runtime.expert_cache_limit_bytes() == ceiling,
                "V4.1 temporary reclamation changed the configured expert ceiling");
            runtime.restore_text_session_state(saved);
            const auto restored = evaluated_floats(runtime.decode(next));
            for (std::size_t index = 0; index < restored.size(); ++index)
                require_close(restored[index], expected[index], 0.002f + std::abs(expected[index]) * 0.002f);
            require(runtime.reclaimable_expert_bytes() > 0,
                "V4.1 expert cache could not refill after temporary reclamation");
        }
        runtime.set_expert_cache_limit(0);
        require(runtime.cache_position() == 18 && runtime.reclaimable_expert_bytes() == 0,
            "V4.1 budget resize discarded KV or retained zero-budget expert pages");
        auto all = runtime.prefill(ids, 8, true);
        require(all.shape() == mlx::core::Shape{1, 17, vocab}, "V4.1 CED truncated full-logit evaluation");
        all.eval();
        MlxSamplingParams sampling;
        sampling.temperature = 0;
        sampling.mtp_max_draft_tokens = 5;
        sampling.enable_mtp = false;
        std::vector<std::int64_t> ordinary;
        require(runtime.generate(prompt, sampling, 8,
            [&](std::int64_t token) { ordinary.push_back(token); return true; }, {}, {}, prompt.size(), 8) == 8,
            "V4.1 ordinary decode did not finish the fixture");
        sampling.enable_mtp = true;
        std::vector<std::int64_t> generated;
        const auto count = runtime.generate(prompt, sampling, 8,
            [&](std::int64_t token) { generated.push_back(token); return true; }, {}, {}, prompt.size(), 8);
        require(count == 8 && generated.size() == 8 && runtime.last_mtp_stats().used &&
                runtime.last_mtp_stats().drafted_tokens > 0,
            "V4.1 CED failed to initialize or verify DSpark drafts");
        auto prefix = runtime.capture_text_session_state(prompt);
        require(prefix.dspark && prefix.dspark->position() == 17,
            "V4.1 CED did not preserve DSpark prefix position");
        require(generated == ordinary, "V4.1 DSpark changed greedy target tokens");
        require(reject_drafts
                ? runtime.last_mtp_stats().accepted_tokens < runtime.last_mtp_stats().drafted_tokens
                : runtime.last_mtp_stats().accepted_tokens > 0,
            "V4.1 DSpark fixture did not exercise its acceptance/rollback branch");
        for (bool mtp : {false, true}) for (std::size_t block_size : {8, 32}) {
            auto history = prompt;
            std::vector<std::int64_t> output;
            std::vector<MlxDeepseekV41TextSessionState> checkpoints;
            MlxPrefixCacheHooks hooks{block_size, prompt.size(), true, [&](std::size_t position) {
                require(position <= history.size(), "V4.1 checkpoint precedes delivered history");
                checkpoints.push_back(runtime.capture_text_session_state(
                    std::vector<std::int64_t>(history.begin(), history.begin() + position), true));
            }};
            sampling.enable_mtp = mtp;
            require(runtime.generate(prompt, sampling, 8, [&](auto token) {
                history.push_back(token); output.push_back(token); return true;
            }, {}, {}, prompt.size(), 8, hooks) == 8 && output == ordinary,
                "V4.1 block checkpointing changed ordinary/DSpark greedy output");
            require(!checkpoints.empty() && checkpoints.back().cache_position == 24 &&
                    checkpoints.back().dspark.has_value() == mtp,
                "V4.1 did not cache committed generated tokens and predictor history");
            require(std::any_of(checkpoints.begin(), checkpoints.end(), [&](const auto& state) {
                    return state.tokens.size() == prompt.size() - 1;
                }), "V4.1 omitted the same-prompt resumable checkpoint");
            const auto restored = persist(checkpoints.back());
            runtime.restore_text_session_state(restored);
            const auto before = evaluated_floats(runtime.decode(next));
            runtime.restore_text_session_state(persist(restored));
            const auto after = evaluated_floats(runtime.decode(next));
            for (std::size_t index = 0; index < before.size(); ++index)
                require_close(after[index], before[index], 0.002f + std::abs(before[index]) * 0.002f);
            const auto history_ids = mlx::core::array(restored.tokens.begin(),
                mlx::core::Shape{1, static_cast<int>(restored.tokens.size())}, mlx::core::int32);
            (void)evaluated_floats(runtime.prefill(history_ids, 8));
            const auto fresh = evaluated_floats(runtime.decode(next));
            for (std::size_t index = 0; index < fresh.size(); ++index)
                require_close(before[index], fresh[index], 0.002f + std::abs(fresh[index]) * 0.002f);
        }
    }
}

void test_deepseek_v41_streamed_moe() {
    using namespace mfq::metal;
    constexpr int experts = 8, hidden = 96, intermediate = 64;
    const auto projection = [&](int output, int input, int salt) {
        const auto nint = make_nint_v2_tensor(4 * output, input, salt, 16);
        const auto vq = make_jsc_nvq(4 * output, input);
        return make_raw_nim2(experts, output, input, {
            {{0, 2, 4, 6}, "NINTv2", nint, {}},
            {{1, 3, 5, 7}, vq.dtype, {vq.blob, vq.dense, 4 * output, input}, vq.runtime},
        });
    };
    std::vector<std::uint8_t> bias;
    append<std::uint32_t>(bias, 1);
    append<std::int64_t>(bias, experts);
    for (int index = 0; index < experts; ++index) append<float>(bias, index * 0.01f);
    for (const bool split : {false, true}) {
        const std::string prefix = "model.block.0.mlp";
        std::vector<MappedRecordFixture> records{
            {prefix + ".router.weight", "F16", make_dense_tensor("F16", experts, hidden, 3).blob},
            {prefix + ".router.bias", "F32", bias},
            {prefix + ".shared_expert.gate.weight", "F16",
                make_dense_tensor("F16", intermediate, hidden, 5).blob},
            {prefix + ".shared_expert.up.weight", "F16",
                make_dense_tensor("F16", intermediate, hidden, 7).blob},
            {prefix + ".shared_expert.down.weight", "F16",
                make_dense_tensor("F16", hidden, intermediate, 11).blob},
            {prefix + ".experts.down.weight", "MFE", projection(hidden, intermediate, 13)},
        };
        records.push_back({prefix + (split ? ".experts.gate.weight" : ".experts.gate_up.weight"),
            "MFE", projection(split ? intermediate : 2 * intermediate, hidden, 17)});
        if (split) records.push_back({prefix + ".experts.up.weight", "MFE",
            projection(intermediate, hidden, 19)});
        const TemporaryMfq file(records);
        const MfqContainer model(file.path());
        for (const int top_k : {3, 6}) {
            DeepseekV41Config config;
            config.hidden = hidden;
            config.moe_inter = intermediate;
            config.n_experts = experts;
            config.top_k = top_k;
            config.norm_topk_prob = true;
            config.routed_scaling = 1.5;
            config.swiglu_limit = 10.0;
            const auto resident = MlxDeepseekV41Moe::load(model, config, prefix);
            auto converted = MlxDeepseekV41Moe::load(model, config, prefix);
            require(converted.resident_routed_bytes() > 0,
                "V4.1 resident expert accounting is empty");
            auto resized = std::make_shared<MlxMfeOffloadCache>(model, 1u << 20);
            converted.validate_offload(resized);
            converted.use_offload(resized);
            require(converted.resident_routed_bytes() == 0,
                "V4.1 dynamic offload retained full expert weights");
            for (const std::size_t budget : {std::size_t{0}, std::size_t{1} << 20}) {
                auto cache = std::make_shared<MlxMfeOffloadCache>(model, budget);
                const auto streamed = MlxDeepseekV41Moe::load(model, config, prefix, false, nullptr, cache);
                for (const int rows : {1, 2, 3, 4, 5, 6, 7, 32, 64, 128}) {
                    std::vector<float> input(rows * hidden);
                    for (int row = 0; row < rows; ++row)
                        for (int column = 0; column < hidden; ++column)
                            input[row * hidden + column] = std::sin(column * 0.17f) * 0.01f;
                    const auto x = mlx::core::astype(mlx::core::array(input.begin(),
                        mlx::core::Shape{1, rows, hidden}), mlx::core::float16);
                    const auto expected = resident.forward(x);
                    const auto actual = streamed.forward(x);
                    const auto a = evaluated_floats(actual.routed), b = evaluated_floats(expected.routed);
                    for (std::size_t index = 0; index < a.size(); ++index)
                        require_close(a[index], b[index], 0.002f + std::abs(b[index]) * 0.002f);
                    require(evaluated_floats(actual.shared) == evaluated_floats(expected.shared),
                        "V4.1 streaming changed shared-expert output");
                    require(cache->resident_packed_bytes() <= budget,
                        "V4.1 streaming exceeded the expert-cache budget");
                    if (budget == 0) {
                        resized->set_cache_limit(rows % 2 ? 0 : 1u << 20);
                        const auto dynamic = evaluated_floats(converted.forward(x).routed);
                        for (std::size_t index = 0; index < dynamic.size(); ++index)
                            require_close(dynamic[index], b[index], 0.002f + std::abs(b[index]) * 0.002f);
                        require(resized->resident_packed_bytes() <= resized->cache_limit_bytes(),
                            "V4.1 dynamic resize exceeded its new cache limit");
                    }
                }
                if (budget != 0) {
                    const auto x = mlx::core::full({1, 1, hidden}, 0.01f, mlx::core::float16);
                    const auto expected = evaluated_floats(resident.forward(x).routed);
                    for (int step = 0; step < 24; ++step) {
                        const auto actual = evaluated_floats(streamed.forward(x).routed);
                        for (std::size_t index = 0; index < actual.size(); ++index)
                            require_close(actual[index], expected[index], 0.002f + std::abs(expected[index]) * 0.002f);
                    }
                    if (__builtin_available(macOS 15, *)) {
                        if (mlx::core::metal::device(mlx::core::Device::gpu).mtl_device()->supportsFamily(MTL::GPUFamilyMetal3))
                            require(cache->reuse_stats().cached_route_hits > 0,
                                "V4.1 did not use the shared cached-route path");
                    }
                }
            }
        }
    }
}

void test_deepseek_v41_visual_router_precision() {
    using namespace mfq::metal;
    constexpr int experts = 8, hidden = 96, intermediate = 64;
    const std::string prefix = "model.block.0.mlp";
    const auto projection = [&](int output, int input, int salt) {
        const auto nint = make_nint_v2_tensor(4 * output, input, salt, 16);
        const auto vq = make_jsc_nvq(4 * output, input);
        return make_raw_nim2(experts, output, input, {
            {{0, 2, 4, 6}, "NINTv2", nint, {}},
            {{1, 3, 5, 7}, vq.dtype, {vq.blob, vq.dense, 4 * output, input}, vq.runtime},
        });
    };
    const auto bias = [&](bool visual) {
        std::vector<std::uint8_t> result;
        append<std::uint32_t>(result, 1);
        append<std::int64_t>(result, experts);
        for (int expert = 0; expert < experts; ++expert)
            append<float>(result, expert == 0 ? (visual ? 0.0001f : 0.1f) : expert == 1 ? 0.0f : -100.0f);
        return result;
    };
    const auto records = [&](bool bf16) {
        std::vector<std::uint8_t> router;
        append<std::uint32_t>(router, 2);
        append<std::int64_t>(router, experts);
        append<std::int64_t>(router, hidden);
        for (int expert = 0; expert < experts; ++expert) for (int col = 0; col < hidden; ++col) {
            const float value = col == 0 && expert < 2 ? 1.0f : col == 1 && expert == 1 ? 1.0f / 1024 : 0.0f;
            if (bf16) append<std::uint16_t>(router, static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16));
            else append<float>(router, value);
        }
        return std::vector<MappedRecordFixture>{
            {prefix + ".router.weight", bf16 ? "BF16" : "F32", std::move(router)},
            {prefix + ".router.bias", "F32", bias(false)},
            {prefix + ".router.vision_bias", "F32", bias(true)},
            {prefix + ".shared_expert.gate.weight", "F16", make_dense_tensor("F16", intermediate, hidden, 5).blob},
            {prefix + ".shared_expert.up.weight", "F16", make_dense_tensor("F16", intermediate, hidden, 7).blob},
            {prefix + ".shared_expert.down.weight", "F16", make_dense_tensor("F16", hidden, intermediate, 11).blob},
            {prefix + ".experts.gate_up.weight", "MFE", projection(2 * intermediate, hidden, 17)},
            {prefix + ".experts.down.weight", "MFE", projection(hidden, intermediate, 13)},
        };
    };
    const TemporaryMfq packed_file(records(true)), reference_file(records(false));
    const MfqContainer packed(packed_file.path()), reference(reference_file.path());
    DeepseekV41Config config;
    config.hidden = hidden; config.moe_inter = intermediate;
    config.n_experts = experts; config.top_k = 1;
    config.norm_topk_prob = true; config.routed_scaling = 1.0; config.swiglu_limit = 10.0;
    const auto expected = MlxDeepseekV41Moe::load(reference, config, prefix);
    for (const bool streamed : {false, true}) {
        auto cache = streamed ? std::make_shared<MlxMfeOffloadCache>(packed, 0) : nullptr;
        const auto actual = MlxDeepseekV41Moe::load(packed, config, prefix, false, nullptr, cache);
        for (int rows : {1, 2, 7}) {
            std::vector<float> values(rows * hidden, 0.0f);
            std::vector<std::uint8_t> mask(rows);
            for (int row = 0; row < rows; ++row) {
                values[row * hidden] = values[row * hidden + 1] = 1.0f;
                mask[row] = row % 2 == 0;
            }
            const auto x = mlx::core::astype(mlx::core::array(values.begin(), mlx::core::Shape{1, rows, hidden}), mlx::core::float16);
            const auto images = mlx::core::astype(mlx::core::array(mask.begin(), mlx::core::Shape{1, rows}), mlx::core::bool_);
            const auto a = evaluated_floats(actual.forward(x, images).routed);
            const auto b = evaluated_floats(expected.forward(x, images).routed);
            for (std::size_t index = 0; index < a.size(); ++index)
                require_close(a[index], b[index], 0.002f + std::abs(b[index]) * 0.002f);
        }
    }
}

void test_streamed_mixed_mfe_residency() {
    bool indirect_supported = false;
    if (__builtin_available(macOS 15, *)) indirect_supported =
        mlx::core::metal::device(mlx::core::Device::gpu).mtl_device()->supportsFamily(MTL::GPUFamilyMetal3);
    constexpr int experts = 12;
    constexpr int output = 13;
    constexpr int input = 96;
    constexpr int tokens = 5;
    constexpr int routes = 3;
    const auto nint = make_nint_v2_tensor(
        2 * output, input, 43, 24);
    const auto legacy_nint = make_nint_tensor(
        5, 2 * output, input, 47, 28);
    const auto mx = make_mxfp4(2 * output, input);
    const auto vq = make_jsc_nvq(2 * output, input);
    const auto nvq_s = make_nvq1_s(2 * output, input);
    const auto nvq_l = make_nvq1_l(2 * output, input);
    const TensorFixture mx_tensor{
        mx.blob, mx.dense, 2 * output, input};
    const TensorFixture vq_tensor{
        vq.blob, vq.dense, 2 * output, input};
    const std::vector<PoolFixture> pools{
        {{4, 1}, "NINTv2", nint, {}},
        {{6, 2}, "NINT5", legacy_nint, {}},
        {{3, 0}, "MXFP4", mx_tensor, {}},
        {{7, 5}, vq.dtype, vq_tensor, vq.runtime},
        {{10, 8}, nvq_s.dtype, {nvq_s.blob, nvq_s.dense, 2 * output, input}, {}},
        {{9, 11}, nvq_l.dtype, {nvq_l.blob, nvq_l.dense, 2 * output, input}, {}},
    };
    const auto blob = make_raw_nim2(
        experts, output, input, pools);
    const auto alternate_blob = make_raw_nim2(
        4,
        output,
        input,
        {
            {
                {3, 1, 0, 2},
                "NINTv2",
                make_nint_v2_tensor(4 * output, input, 59, 24),
                {},
            },
        });
    const TemporaryMfq file({
        {"mixed", "MFE", blob},
        {"alternate", "MFE", alternate_blob},
        {"twin", "MFE", blob},
        {"pair_up", "MFE", make_raw_nim2(experts, output, input, {
            {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, "NINTv2",
                make_nint_v2_tensor(experts * output, input, 71, 28), {}},
        })},
    });
    const mfq::metal::MfqContainer model(file.path());
    mfq::metal::MlxMfeOffloadCache residency(
        model, 1u << 24);
    const auto info = residency.projection_info("mixed");
    require(
        residency.can_stream("mixed"),
        "mixed NINTv1/NINTv2/NVQ/MX MFE record was not streamable");
    require(
        info.experts == experts
            && info.out_per_expert == output
            && info.neuron_len == input
            && info.available_experts.size() == experts,
        "mixed streamed MFE projection metadata mismatch");
    const auto alternate_info = residency.projection_info("alternate");
    require(mfq::metal::MlxResourceTelemetry::snapshot().expert_payload_bytes == 0,
        "probing streamable MFE records reported resident weights as SSD backing");
    require(
        alternate_info.experts == 4
            && residency.grouped_mfe("alternate", {3, 0}).experts() == 2,
        "dynamic streamed MFE expert geometry mismatch");
    require(mfq::metal::MlxResourceTelemetry::snapshot().expert_payload_bytes == alternate_blob.size(),
        "streaming one MFE record reported other probed records as SSD backing");
    residency.discard_record("alternate");
    require(mfq::metal::MlxResourceTelemetry::snapshot().expert_payload_bytes == 0,
        "discarding an MFE record retained SSD backing telemetry");

    const auto full = mfq::metal::MlxMfeWeight::from_blob(blob);
    {
        mfq::metal::MlxMfeOffloadCache streamed(model, 1u << 24);
        std::vector<std::int32_t> active(experts);
        std::iota(active.begin(), active.end(), 0);
        const auto paged = streamed.grouped_mfe("mixed", active);
        for (const auto& weight : {full, paged}) {
            require(weight.supports_fused_routed_reduce(),
                "mixed-format Down did not expose fused weighted reduction");
            for (const int rows : {1, 2, 3, 6}) {
                std::vector<std::int32_t> selected(rows * 6);
                for (std::size_t i = 0; i < selected.size(); ++i)
                    selected[i] = i % 9 == 0 ? -1 : (i * 5 + i / 6) % experts;
                const auto ids = mlx::core::array(selected.begin(), mlx::core::Shape{rows, 6});
                for (const auto dtype : {mlx::core::float16, mlx::core::float32}) {
                    auto x = mlx::core::astype(mlx::core::reshape(
                        mlx::core::sin(mlx::core::arange(rows * 6 * input, mlx::core::float32) * 0.03f),
                        {rows, 6, input}), dtype);
                    auto scales = mlx::core::full({rows, 6}, 0.125f, mlx::core::float32);
                    auto expected = evaluated_floats(mfq::metal::moe_weighted_reduce(
                        weight.routed_matmul(x, ids), scales));
                    auto actual = evaluated_floats(weight.routed_matmul_reduce(x, ids, scales));
                    require(actual.size() == expected.size(), "mixed Down reduction shape changed");
                    for (std::size_t i = 0; i < actual.size(); ++i)
                        require_close(expected[i], actual[i], dtype == mlx::core::float16 ? 0.001f : 0.00001f);
                }
            }
        }
    }
    {
        mfq::metal::MlxMfeOffloadCache priority(model, 1u << 24);
        const std::vector<std::int32_t> active{4, 1};
        (void)priority.grouped_mfe("mixed", active);
        const auto encoder_bytes = priority.resident_packed_bytes();
        for (int round = 0; round < 4; ++round)
            (void)priority.grouped_mfe("twin", active);
        priority.set_prefill_priority(std::array{std::string("mixed")});
        priority.set_cache_limit(encoder_bytes);
        const auto loads = priority.reuse_stats().expert_loads;
        (void)priority.grouped_mfe("mixed", active, {}, true);
        require(priority.reuse_stats().expert_loads == loads,
            "prefill priority evicted Encoder pages before hotter Decoder pages");
        priority.set_prefill_priority({});
        priority.set_cache_limit(0);
        require(priority.resident_packed_bytes() == 0,
            "prefill priority prevented zero-budget eviction");
        priority.set_prefill_priority(std::array{std::string("mixed")});
        priority.clear();
        priority.set_cache_limit(encoder_bytes);
        (void)priority.grouped_mfe("mixed", active, {}, true);
        priority.discard_record("mixed");
        priority.set_prefill_priority({});
        require(priority.resident_packed_bytes() == 0,
            "discarding prioritized pages retained resident bytes");
    }
    if (indirect_supported) {
        mfq::metal::MlxMfeOffloadCache hot(model, 1u << 24);
        const std::vector<std::int32_t> active{4, 1, 7};
        (void)hot.grouped_mfe("mixed", active);
        const auto reads = hot.reuse_stats().bytes_read;
        std::size_t reserved = 0;
        mfq::metal::MlxResidentBudgetScope scope([&](std::size_t bytes) {
            reserved += bytes;
            hot.clear();
        });
        const auto paged = hot.grouped_mfe("mixed", active);
        require(reserved == std::size_t{3} * getpagesize() + active.size() * (33 * 4 + 21 * 8),
            "indirect cache hits reserved another full packed weight copy");
        require(hot.reuse_stats().bytes_read == reads && hot.resident_packed_bytes() == 0,
            "a budget trim discarded selected hot pages or readmitted invalidated pages");
        const auto x = mlx::core::full({1, input}, 0.125f, mlx::core::float16);
        require(evaluated_floats(paged.routed_matmul(x, mlx::core::array({0, 1, 2}, {1, 3})))
                == evaluated_floats(full.routed_matmul(x, mlx::core::array(active.begin(), {1, 3}))),
            "budget-protected expert pages changed projection values");
    }
    for (const std::vector<std::int32_t> active : {
            std::vector<std::int32_t>{4, 1, 3, 0, 7, 5}, {3, 0}, {0, 3}}) {
        mfq::metal::MlxMfeOffloadCache scan(model, 0);
        const auto canonical = full.select_experts(active).with_automatic_mxfp4_nax_prefill(false);
        const auto paged = scan.grouped_mfe("mixed", active, {}, true);
        if (indirect_supported && active.size() == 2)
            require(!paged.supports_mxfp4_blocks(),
                "indirect MXFP4 subset advertised contiguous expert slots");
        for (const int rows : {33, 385}) {
            const int count = rows * routes;
            std::vector<std::int32_t> values(count);
            for (int i = 0; i < count; ++i) values[i] = (i * 5 + i / routes) % active.size();
            const auto ids = mlx::core::array(values.begin(), mlx::core::Shape{rows, routes});
            const auto order = mlx::core::astype(mlx::core::argsort(mlx::core::reshape(ids,
                mlx::core::Shape{count})), mlx::core::int32);
            const auto x = mlx::core::full({rows, input}, 0.125f, mlx::core::float16);
            const auto plan = canonical.build_grouped_mmq_plan(ids, order, 32);
            const auto expected = evaluated_floats(canonical.routed_matmul_sorted(
                x, ids, order, false, false, 0.0f, &plan));
            const auto copies = mfq::metal::MlxMfeWeight::packed_copy_count();
            const auto actual = evaluated_floats(paged.routed_matmul_sorted(
                x, ids, order, false, false, 0.0f, &plan));
            require(actual == expected, "indirect mixed prefill changed grouped matrix values");
            bool indirect_supported = false;
            if (__builtin_available(macOS 15, *)) indirect_supported =
                mlx::core::metal::device(mlx::core::Device::gpu).mtl_device()->supportsFamily(MTL::GPUFamilyMetal3);
            require(!indirect_supported || mfq::metal::MlxMfeWeight::packed_copy_count() == copies,
                "indirect mixed prefill unnecessarily copied packed expert pages");
        }
    }
    {
        mfq::metal::MlxMfeOffloadCache individual(model, 0), coalesced(model, 0);
        std::vector<std::int32_t> active(experts);
        std::iota(active.begin(), active.end(), 0);
        const auto serial = individual.grouped_mfe("mixed", active);
        const auto batched = coalesced.grouped_mfe("mixed", active, {}, true);
        const auto x = mlx::core::full({1, input}, 0.125f, mlx::core::float16);
        const auto ids = mlx::core::array(active.begin(), mlx::core::Shape{1, experts});
        require(evaluated_floats(serial.routed_matmul(x, ids)) == evaluated_floats(batched.routed_matmul(x, ids)),
            "coalescing adjacent expert reads changed packed values");
        require(coalesced.reuse_stats().read_calls < individual.reuse_stats().read_calls
                && coalesced.reuse_stats().bytes_read <= individual.reuse_stats().bytes_read,
            "coalesced prefill reads did not reduce system calls or read unrequested gaps");
    }
    {
        mfq::metal::MlxMfeOffloadCache paired(model, 0);
        std::vector<std::int32_t> active(experts);
        std::iota(active.begin(), active.end(), 0);
        const auto weights = paired.grouped_mfe(std::array{std::string("mixed"), std::string("pair_up")}, active);
        paired.clear();
        {
            const auto gate = weights[0].select_experts(std::array<std::int32_t, 4>{3, 1, 7, 2});
            const auto up = weights[1].select_experts(std::array<std::int32_t, 3>{0, 4, 11});
            const auto gate_map = mlx::core::array({0, 1, 3, -1, 2, 0}, mlx::core::int32);
            const auto up_map = mlx::core::array({2, -1, 1, 0, 2, 1}, mlx::core::int32);
            const auto ids = mlx::core::reshape(mlx::core::arange(6, mlx::core::int32), {1, 6});
            const auto x = mlx::core::full({1, input}, 0.125f, mlx::core::float16);
            for (const auto limit : {0.0f, 0.5f}) {
                auto gate_value = gate.routed_matmul_mapped(x, ids, gate_map);
                auto up_value = up.routed_matmul_mapped(x, ids, up_map);
                const auto expected = evaluated_floats(limit > 0
                    ? mfq::metal::moe_limited_swiglu_pair(gate_value, up_value, limit)
                    : mfq::metal::moe_swiglu_pair(gate_value, up_value));
                const auto calls = mfq::metal::MlxMfeWeight::indirect_dispatch_count();
                const auto actual = evaluated_floats(gate.routed_swiglu_pair(
                    up, x, ids, limit, nullptr, &gate_map, &up_map));
                require(actual == expected, "independent Gate/Up page tables changed clipped SwiGLU values");
                require(!indirect_supported || mfq::metal::MlxMfeWeight::indirect_dispatch_count() == calls + 1,
                    "independently mapped Gate/Up did not use one fused dispatch");
            }
        }
        for (const int rows : {1, 2, 3, 4, 5, 6}) for (const float limit : {0.0f, 0.5f}) {
            std::vector<float> values(rows * input);
            for (std::size_t index = 0; index < values.size(); ++index) values[index] = float(int(index % 23) - 11) / 128;
            std::vector<std::int32_t> routes(rows * experts);
            for (std::size_t index = 0; index < routes.size(); ++index) routes[index] = (index * 5 + index / experts) % experts;
            const auto x = mlx::core::astype(mlx::core::array(values.begin(), mlx::core::Shape{rows, input}), mlx::core::float16);
            const auto ids = mlx::core::array(routes.begin(), mlx::core::Shape{rows, experts});
            const auto groups = rows > 1 ? mfq::metal::moe_route_groups_host(routes, rows, experts) : std::nullopt;
            const auto* grouped = groups ? &*groups : nullptr;
            for (const bool reverse : {false, true}) {
                const auto& gate = weights[reverse ? 1 : 0];
                const auto& up = weights[reverse ? 0 : 1];
                const auto expected = evaluated_floats(mfq::metal::moe_limited_swiglu_split(
                    mlx::core::concatenate({gate.routed_matmul(x, ids, grouped), up.routed_matmul(x, ids, grouped)}, -1), limit));
                const auto copies = mfq::metal::MlxMfeWeight::packed_copy_count();
                auto result = gate.routed_swiglu_pair(up, x, ids, limit, grouped);
                require(evaluated_floats(result) == expected, "indirect Gate/Up changed projection or activation values");
                require(!indirect_supported || mfq::metal::MlxMfeWeight::packed_copy_count() == copies,
                    "indirect Gate/Up unnecessarily copied expert weights");
            }
        }
    }
    for (const auto budget : {std::size_t{0}, std::size_t{1}, std::size_t{1u << 24}}) {
        mfq::metal::MlxMfeOffloadCache combined(model, budget), serial(model, budget);
        const auto x = mlx::core::full({2, input}, 0.125f, mlx::core::float16);
        const auto check = [&](const std::vector<std::string>& names,
            const std::vector<std::int32_t>& active, bool scan = false) {
            auto weights = combined.grouped_mfe(names, active, {}, scan);
            std::vector<std::int32_t> ids(2 * active.size());
            for (std::size_t index = 0; index < ids.size(); ++index) ids[index] = index % active.size();
            const auto routes = mlx::core::array(ids.begin(),
                mlx::core::Shape{2, static_cast<int>(active.size())});
            std::vector<std::vector<float>> expected;
            for (const auto& name : names)
                expected.push_back(evaluated_floats(serial.grouped_mfe(name, active).routed_matmul(x, routes)));
            require(weights.size() == names.size(), "combined MFE projection count mismatch");
            for (std::size_t index = 0; index < names.size(); ++index)
                require(evaluated_floats(weights[index].routed_matmul(x, routes)) == expected[index],
                    "combined MFE preparation changed projection order or expert bytes");
            combined.clear();
            for (std::size_t index = 0; index < names.size(); ++index)
                require(evaluated_floats(weights[index].routed_matmul(x, routes)) == expected[index],
                    "combined MFE preparation lost retained expert storage");
        };
        check({"mixed", "alternate"}, {3, 1, 0, 2});
        check({"alternate", "mixed"}, {2, 0, 1, 3});
        check({"mixed", "twin"}, {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
        const auto pooled = mfq::metal::MlxMfeWeight::pooled_staging_count();
        check({"mixed", "alternate"}, {3, 1, 0, 2}, true);
        check({"mixed", "twin"}, {11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, true);
        if (budget <= 1) require(mfq::metal::MlxMfeWeight::pooled_staging_count() >= pooled + 4,
            "prefill misses did not use bounded shared staging arenas");
        require(combined.resident_packed_bytes() <= budget, "combined MFE preparation exceeded cache budget");
        for (const std::vector<std::string> names : {std::vector<std::string>{}, {"mixed", "mixed"}, {"mixed", "zzmissing"}}) {
            bool rejected = false;
            try { (void)combined.grouped_mfe(names, {1, 2}); }
            catch (const std::exception&) { rejected = true; }
            require(rejected, "combined MFE preparation accepted invalid projection names");
            check({"mixed", "alternate"}, {1, 2});
        }
        combined.clear();
        int reservations = 0;
        bool denied = false;
        {
            mfq::metal::MlxResidentBudgetScope scope([&](std::size_t) {
                if (++reservations == 2) throw std::runtime_error("denied second projection staging");
            });
            try { (void)combined.grouped_mfe(std::array{std::string("mixed"), std::string("alternate")}, {1, 2}); }
            catch (const std::runtime_error&) { denied = true; }
        }
        require(denied && reservations == 2, "combined MFE preparation ignored the staging budget");
        check({"mixed", "alternate"}, {1, 2});
        combined.clear();
        std::vector<std::pair<std::size_t, std::size_t>> staging;
        {
            mfq::metal::MlxResidentBudgetScope scope([&](std::size_t bytes) {
                staging.emplace_back(bytes, mfq::metal::MlxResidentBudgetScope::temporary_bytes());
            });
            auto weights = combined.grouped_mfe(std::array{std::string("mixed"), std::string("alternate")}, {1, 2});
            require(weights.size() == 2, "packed staging lost a projection");
            require(staging.size() == 2 && staging[0].first > 0 && staging[0].second == 0
                    && staging[1].second == staging[0].first,
                "parallel projections did not share their outstanding staging reservation");
            require(mfq::metal::MlxResidentBudgetScope::temporary_bytes() == 0,
                "completed expert staging retained a budget reservation");
        }
    }
    {
        mfq::metal::MlxMfeOffloadCache combined(model, 1u << 24);
        const auto previous_stream = mlx::core::default_stream(mlx::core::Device::gpu);
        const auto stream = mlx::core::new_thread_unsafe_stream(mlx::core::Device::gpu);
        mlx::core::set_default_stream(stream);
        std::barrier start(2);
        std::array<std::exception_ptr, 2> errors;
        const auto run = [&](int index) {
            mlx::core::set_default_stream(stream);
            start.arrive_and_wait();
            try { (void)combined.grouped_mfe(index ? std::vector<std::string>{"mixed", "alternate"}
                : std::vector<std::string>{"alternate", "mixed"}, {3, 1, 0, 2}); }
            catch (...) { errors[index] = std::current_exception(); }
        };
        std::thread first([&] { run(0); }), second([&] { run(1); });
        first.join(); second.join();
        for (const auto& error : errors) if (error) std::rethrow_exception(error);
        require(combined.reuse_stats().expert_loads == 8,
            "reversed concurrent MFE projection batches duplicated loads");
        mlx::core::set_default_stream(previous_stream);
    }
    for (const auto budget : {std::size_t{0}, std::size_t{1u << 24}}) {
        mfq::metal::MlxMfeOffloadCache changing(model, budget);
        mfq::metal::MlxKernelPreparation preparation(128, 128);
        preparation.set_routes(routes);
        changing.prepare_projection("mixed", true);
        const auto compiled = preparation.size();
        require(compiled > 0, "SSD expert kernels were not prepared at load time");
        std::vector<mlx::core::array> outputs;
        preparation.collect([&] {
            for (const std::vector<std::int32_t> active : {
                    std::vector<std::int32_t>{1}, {2}, {0}, {7}, {8}, {9},
                    {9, 2, 8, 1}, {11, 10, 7, 6, 5, 4, 3, 2, 1, 0}, {1, 2, 8}}) {
                const auto weight = changing.grouped_mfe("mixed", active);
                for (const int rows : {1, 2, 3, 4, 5, 6, 8, 17, 22}) {
                    const auto x = mlx::core::full({rows, input}, 0.125f, mlx::core::float16);
                    const std::vector<std::int32_t> ids(rows * routes, 0);
                    const auto ids_array = mlx::core::array(ids.begin(), mlx::core::Shape{rows, routes});
                    outputs.push_back(weight.routed_matmul(x, ids_array));
                    preparation.add(outputs.back());
                    if (rows > 1 && rows <= 24) {
                        const auto groups = mfq::metal::moe_route_groups_host(ids, rows, routes);
                        outputs.push_back(weight.routed_matmul(x, ids_array, &*groups));
                        preparation.add(outputs.back());
                    }
                }
                require(preparation.size() == compiled,
                    "changing SSD expert count, formats or GS created a new compilation variant");
                const std::array invalid{-1, static_cast<int>(active.size())};
                const auto empty = evaluated_floats(weight.routed_matmul(
                    mlx::core::full({1, input}, 0.125f, mlx::core::float16),
                    mlx::core::array(invalid.begin(), mlx::core::Shape{1, 2})));
                require(std::all_of(empty.begin(), empty.end(), [](float value) { return value == 0; }),
                    "runtime expert bounds accepted an invalid local ID");
            }
        });
        preparation.finish();
    }
    const auto repeat_pair = mfq::metal::MlxMfeWeight::from_projection_blobs(
        std::array<std::span<const std::uint8_t>, 2>{blob, blob});
    for (const int rows : {2, 3, 4, 5, 6, 8, 12, 17, 24}) {
        std::vector<float> source(rows * input);
        for (std::size_t index = 0; index < source.size(); ++index)
            source[index] = float(int((index * 17 + 5) % 31) - 15) / 128.0f;
        const auto input_values = mlx::core::array(source.begin(), mlx::core::Shape{rows, input});
        for (const auto dtype : {mlx::core::float16, mlx::core::bfloat16, mlx::core::float32}) {
            const auto x = mlx::core::astype(input_values, dtype);
            for (const auto expert : {1, 2, 4, 7, 8, 9}) {
                const std::vector<std::int32_t> ids(rows * routes, expert);
                const auto id_array = mlx::core::array(ids.begin(), mlx::core::Shape{rows, routes});
                const auto groups = mfq::metal::moe_route_groups_host(ids, rows, routes);
                require(evaluated_floats(full.routed_matmul(x, id_array))
                    == evaluated_floats(full.routed_matmul(x, id_array, &*groups)),
                    "MTP route chunks changed repeated expert results");
                require(evaluated_floats(repeat_pair.routed_swiglu(x, id_array))
                    == evaluated_floats(repeat_pair.routed_swiglu(x, id_array, 0.0f, &*groups)),
                    "MTP route chunks changed repeated fused expert results");
            }
        }
    }
    {
        const std::array names{std::string("mixed"), std::string("alternate")};
        mfq::metal::MlxMfeOffloadCache bulk(model, 2 * (blob.size() + alternate_blob.size()));
        auto layer = bulk.layer_mfe(names);
        require(layer && layer->size() == names.size(), "bulk MFE layer was not loaded");
        const auto reads = bulk.reuse_stats();
        require(reads.bytes_read == blob.size() + alternate_blob.size()
            && reads.read_calls == names.size() && reads.io_seconds > 0,
            "bulk expert telemetry lost actual file ranges");
        require(bulk.cached_expert_count() == 0 && bulk.resident_packed_bytes() == 0,
            "bulk prefill polluted the decode expert cache");
        std::vector<float> source(input, 0.125f);
        std::vector<std::int32_t> ids(experts);
        std::iota(ids.begin(), ids.end(), 0);
        const auto x = mlx::core::astype(mlx::core::array(source.begin(), mlx::core::Shape{1, input}),
            mlx::core::float16);
        const auto routes = mlx::core::array(ids.begin(), mlx::core::Shape{1, experts});
        const auto expected = evaluated_floats(full.routed_matmul(x, routes));
        bulk.clear();
        const auto actual = evaluated_floats((*layer)[0].routed_matmul(x, routes));
        require(actual == expected, "bulk MFE output or retained storage mismatch");
        const std::array mixed_name{std::string("mixed")};
        const std::array repeated{4, 4, 4, 4};
        require(!bulk.layer_mfe(mixed_name, repeated), "sparse MFE routes triggered full-layer reads");
        const std::array invalid{experts};
        bool invalid_rejected = false;
        try { (void)bulk.layer_mfe(mixed_name, invalid); }
        catch (const std::out_of_range&) { invalid_rejected = true; }
        require(invalid_rejected, "bulk MFE accepted an out-of-range expert");
        bulk.set_cache_limit(1u << 24);
        (void)bulk.grouped_mfe("mixed", ids);
        require(!bulk.layer_mfe(mixed_name, ids), "cached MFE routes triggered full-layer reads");
        bulk.set_cache_limit(1);
        require(!bulk.layer_mfe(names), "bulk MFE ignored insufficient staging capacity");
        bulk.set_cache_limit(2 * (blob.size() + alternate_blob.size()));
        {
            const auto packed_staging = 2 * (blob.size() + alternate_blob.size());
            bool reserved = false;
            mfq::metal::MlxResidentBudgetScope packed_only([&](std::size_t bytes) {
                require(bytes == packed_staging, "native bulk MFE reserved a nonexistent F16 expert bank");
                reserved = true;
            }, [&](std::size_t bytes) { return bytes <= packed_staging; });
            require(bulk.layer_mfe(names).has_value() && reserved,
                "native bulk MFE was refused despite enough packed-weight staging capacity");
        }
        mfq::metal::MlxResidentBudgetScope deny([](std::size_t) {
            throw std::runtime_error("denied bulk staging should not reserve memory");
        }, [](std::size_t) { return false; });
        require(!bulk.layer_mfe(names), "bulk MFE ignored the total resident budget");
    }
    const auto exercise = [&](const std::vector<std::int32_t>& active) {
        const auto paged = residency.grouped_mfe("mixed", active);
        require(
            paged.experts() == static_cast<int>(active.size())
                && paged.out_per_expert() == output
                && paged.neuron_len() == input
                && paged.supports_grouped_mmq(),
            "mixed streamed MFE active weight metadata mismatch");
        std::vector<float> source(
            static_cast<std::size_t>(tokens) * input);
        for (std::size_t index = 0; index < source.size(); ++index) {
            source[index] = static_cast<float>(
                static_cast<int>((index * 23 + 11) % 41) - 20) / 256.0f;
        }
        std::vector<std::int32_t> global_ids(
            static_cast<std::size_t>(tokens) * routes);
        std::vector<std::int32_t> local_ids(global_ids.size());
        for (std::size_t index = 0; index < global_ids.size(); ++index) {
            const auto local = static_cast<std::int32_t>(index % active.size());
            local_ids[index] = local;
            global_ids[index] = active[static_cast<std::size_t>(local)];
        }
        const auto input_array = mlx::core::astype(
            mlx::core::array(
                source.begin(), mlx::core::Shape{tokens, input}),
            mlx::core::float16);
        const auto expected = evaluated_floats(full.routed_matmul(
            input_array,
            mlx::core::array(
                global_ids.begin(), mlx::core::Shape{tokens, routes})));
        const auto actual = evaluated_floats(paged.routed_matmul(
            input_array,
            mlx::core::array(
                local_ids.begin(), mlx::core::Shape{tokens, routes})));
        const auto groups = mfq::metal::moe_route_groups_host(local_ids, tokens, routes);
        require(groups.has_value(), "small-M CPU route grouping was not selected");
        const auto planned = evaluated_floats(paged.routed_matmul(input_array,
            mlx::core::array(local_ids.begin(), mlx::core::Shape{tokens, routes}), &*groups));
        require(actual == planned, "shared CPU route groups changed mixed MFE results");
        const auto pair = evaluated_floats(paged.routed_swiglu_pair(paged, input_array,
            mlx::core::array(local_ids.begin(), mlx::core::Shape{tokens, routes})));
        const auto planned_pair = evaluated_floats(paged.routed_swiglu_pair(paged, input_array,
            mlx::core::array(local_ids.begin(), mlx::core::Shape{tokens, routes}), 0.0f, &*groups));
        require(pair == planned_pair, "shared CPU groups changed separate Gate/Up results");
        require(
            actual.size() == expected.size(),
            "mixed streamed MFE output shape mismatch");
        for (std::size_t index = 0; index < actual.size(); ++index) {
            if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]) ||
                std::fabs(actual[index] - expected[index]) > 1e-6f) {
                throw std::runtime_error("mixed streamed MFE mismatch: expert=" +
                    std::to_string(global_ids[index / output]) + " actual=" +
                    std::to_string(actual[index]) + " expected=" + std::to_string(expected[index]));
            }
        }
    };
    exercise({1, 2, 3, 5, 8, 9});
    exercise({4, 6, 0, 7, 10, 11});
    require(
        residency.cached_expert_count() == experts
            && residency.resident_packed_bytes() > 0,
        "mixed streamed MFE cache accounting mismatch");
    exercise({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    const auto reuse_before = residency.reuse_stats();
    exercise({11, 9, 8, 5, 3, 2, 1});
    exercise({7, 0, 6, 4});
    const auto reuse_after = residency.reuse_stats();
    require(reuse_after.expert_loads == reuse_before.expert_loads
        && reuse_after.packed_hits == reuse_before.packed_hits + (indirect_supported ? 0 : 2),
        "changing active expert subsets rebuilt packed weights or reread SSD pages");
    require(reuse_before.bytes_read > 0 && reuse_before.read_calls > 0
        && reuse_after.bytes_read == reuse_before.bytes_read
        && reuse_after.read_calls == reuse_before.read_calls,
        "packed cache hits reported missing or duplicate file reads");
    {
        mfq::metal::MlxMfeOffloadCache concurrent(model, 1u << 24);
        std::vector<std::int32_t> forward(experts);
        std::iota(forward.begin(), forward.end(), 0);
        auto reverse = forward;
        std::reverse(reverse.begin(), reverse.end());
        std::array<std::optional<mfq::metal::MlxMfeWeight>, 2> results;
        std::array<std::exception_ptr, 2> errors;
        const auto previous_stream = mlx::core::default_stream(mlx::core::Device::gpu);
        const auto stream = mlx::core::new_thread_unsafe_stream(mlx::core::Device::gpu);
        mlx::core::set_default_stream(stream);
        std::barrier start(2);
        const auto run = [&](int index, const auto& active) {
            mlx::core::set_default_stream(stream);
            start.arrive_and_wait();
            try { results[index] = concurrent.grouped_mfe("mixed", active); }
            catch (...) { errors[index] = std::current_exception(); }
        };
        std::thread first([&] { run(0, forward); });
        std::thread second([&] { run(1, reverse); });
        first.join();
        second.join();
        for (const auto& error : errors) if (error) std::rethrow_exception(error);
        const auto stats = concurrent.reuse_stats();
        require(stats.expert_loads == experts
                && stats.packed_hits == std::size_t(!indirect_supported || stats.load_waits != 0),
            "concurrent requests duplicated expert reads or packed assembly");
        require(concurrent.resident_packed_bytes() <= concurrent.cache_limit_bytes(),
            "concurrent prepared weights exceeded the expert budget");
        const std::vector<float> source(input, 0.125f);
        const auto x = mlx::core::astype(mlx::core::array(source.begin(), mlx::core::Shape{1, input}),
            mlx::core::float16);
        const auto ids = mlx::core::array(forward.begin(), mlx::core::Shape{1, experts});
        const auto expected = evaluated_floats(full.routed_matmul(x, ids));
        concurrent.clear();
        const auto normal = evaluated_floats(results[0]->routed_matmul(x, ids));
        const auto reversed = evaluated_floats(results[1]->routed_matmul(x, ids));
        require(normal == expected, "clearing a cache invalidated concurrent packed views");
        for (int expert = 0; expert < experts; ++expert)
            for (int row = 0; row < output; ++row)
                require(reversed[expert * output + row] == expected[(experts - expert - 1) * output + row],
                    "reordered shared-storage expert view changed numerical results");
        const std::array duplicate{2, 2, 7};
        const auto selected = full.select_experts(duplicate);
        const std::array local{0, 1, 2};
        require(evaluated_floats(selected.routed_matmul(x,
            mlx::core::array(local.begin(), mlx::core::Shape{1, 3})))
            == evaluated_floats(full.routed_matmul(x,
                mlx::core::array(duplicate.begin(), mlx::core::Shape{1, 3}))),
            "duplicate expert descriptor views changed results");
        const std::array invalid{-1};
        bool rejected = false;
        try { (void)full.select_experts(invalid); }
        catch (const std::out_of_range&) { rejected = true; }
        require(rejected, "shared MFE view accepted a negative expert ID");
        const std::array<std::uint32_t, 1> zero{0};
        rejected = false;
        try { (void)concurrent.grouped_mfe("mixed", {1}, zero); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "MFE heat accepted zero activations");
        mlx::core::set_default_stream(previous_stream);
    }
    {
        mfq::metal::MlxMfeOffloadCache bounded(model, 1u << 24);
        auto retained = bounded.grouped_mfe("mixed", {1, 2, 3});
        const auto bytes = bounded.resident_packed_bytes();
        const auto count = bounded.cached_expert_count();
        require(bounded.set_cache_limit(bytes - 1) > 0
                && (indirect_supported ? bounded.cached_expert_count() < count
                    : bounded.cached_expert_count() == count),
            "shrinking the expert cache did not release its actual cached storage");
        bounded.set_cache_limit(1u << 24);
        {
            mfq::metal::MlxResidentBudgetScope scope([&](std::size_t) { bounded.clear(); });
            retained = bounded.grouped_mfe("mixed", {1, 2, 3});
        }
        require(retained.experts() == 3 && bounded.resident_packed_bytes() == 0,
            "clearing a cache during preparation admitted stale expert pages");
        (void)bounded.grouped_mfe("mixed", {1, 2, 3});
        require(bounded.cached_expert_count() == 3,
            "cache invalidation stranded the in-flight loading marker");
    }
    {
        const auto previous_stream = mlx::core::default_stream(mlx::core::Device::gpu);
        const auto stream = mlx::core::new_thread_unsafe_stream(mlx::core::Device::gpu);
        mlx::core::set_default_stream(stream);
        bool observed = false;
        for (int attempt = 0; attempt < 8 && !observed; ++attempt) {
            mfq::metal::MlxMfeOffloadCache transient(model, 0);
            std::vector<std::int32_t> ids(experts);
            std::iota(ids.begin(), ids.end(), 0);
            std::barrier start(2);
            std::array<std::exception_ptr, 2> errors;
            const auto run = [&](int index) {
                mlx::core::set_default_stream(stream);
                start.arrive_and_wait();
                try { (void)transient.grouped_mfe("mixed", ids, {}, true); }
                catch (...) { errors[index] = std::current_exception(); }
            };
            std::thread first([&] { run(0); });
            std::thread second([&] { run(1); });
            first.join();
            second.join();
            for (const auto& error : errors) if (error) std::rethrow_exception(error);
            const auto stats = transient.reuse_stats();
            if (stats.load_waits) {
                require(stats.expert_loads == experts && stats.packed_hits == 1,
                    "zero-budget concurrent requests duplicated in-flight expert reads");
                observed = true;
            }
            require(transient.resident_packed_bytes() == 0 && transient.cached_expert_count() == 0,
                "sharing active expert leases retained a zero-budget cache");
        }
        require(observed, "concurrent transient-expert test never overlapped requests");
        mlx::core::set_default_stream(previous_stream);
    }
    {
        const auto packed = make_mxfp4(4 * 128, 1024);
        const auto pin_blob = make_raw_nim2(4, 128, 1024, {
            {{0, 1, 2, 3}, "MXFP4", {packed.blob, packed.dense, 4 * 128, 1024}, {}},
        });
        const TemporaryMfq pin_file({{"pin", "MFE", pin_blob}});
        const mfq::metal::MfqContainer pin_model(pin_file.path());
        const auto before = mfq::metal::MlxMfeWeight::pinned_packed_bytes();
        const auto failures = mfq::metal::MlxMfeWeight::packed_pin_failures();
        mfq::metal::MlxMfeOffloadCache cache(pin_model, 8u << 20);
        auto retained = cache.grouped_mfe("pin", {0}, {}, true);
        const auto wait_for = [](const auto& ready) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            while (!ready() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        };
        wait_for([&] { return mfq::metal::MlxMfeWeight::pinned_packed_bytes() > before
            || mfq::metal::MlxMfeWeight::packed_pin_failures() > failures; });
        require(mfq::metal::MlxMfeWeight::pinned_packed_bytes() > before
                || mfq::metal::MlxMfeWeight::packed_pin_failures() > failures,
            "admitted prefill expert neither pinned its storage nor reported an OS refusal");
        const auto x = mlx::core::full({1, 1024}, 0.125f, mlx::core::float16);
        const auto ids = mlx::core::reshape(mlx::core::array({0}, mlx::core::int32), {1, 1});
        const auto expected = evaluated_floats(retained.routed_matmul(x, ids));
        cache.clear();
        wait_for([&] { return mfq::metal::MlxMfeWeight::pinned_packed_bytes() == before; });
        require(mfq::metal::MlxMfeWeight::pinned_packed_bytes() == before,
            "clearing expert residency retained locked pages");
        require(evaluated_floats(retained.routed_matmul(x, ids)) == expected,
            "unpinning an evicted expert changed retained GPU storage");
        for (int iteration = 0; iteration < 16; ++iteration) {
            for (int access = 0; access < 3; ++access)
                (void)cache.grouped_mfe("pin", {0, 1, 2, 3});
            cache.set_cache_limit(1);
            require(cache.cached_expert_count() == 0 && cache.resident_packed_bytes() == 0,
                "pending page locks prevented expert budget eviction");
            cache.set_cache_limit(8u << 20);
        }
        cache.discard_record("pin");
        wait_for([&] { return mfq::metal::MlxMfeWeight::pinned_packed_bytes() == before; });
        require(mfq::metal::MlxMfeWeight::pinned_packed_bytes() == before,
            "cancelled page locks retained evicted expert storage");
    }
    {
        const auto mx_heat = make_mxfp4(4 * output, input);
        const auto heat_blob = make_raw_nim2(4, output, input, {
            {{0, 1, 2, 3}, "MXFP4", {mx_heat.blob, mx_heat.dense, 4 * output, input}, {}},
        });
        const TemporaryMfq heat_file({{"heat", "MFE", heat_blob}});
        const mfq::metal::MfqContainer heat_model(heat_file.path());
        mfq::metal::MlxMfeOffloadCache heat(heat_model, 0);
        const auto page = heat.grouped_mfe("heat", {0});
        heat.set_cache_limit(2 * page.packed_nbytes());
        const std::array<std::uint32_t, 1> repeated{6};
        (void)heat.grouped_mfe("heat", {0}, repeated);
        (void)heat.grouped_mfe("heat", {1});
        (void)heat.grouped_mfe("heat", {2});
        const auto before = heat.reuse_stats();
        (void)heat.grouped_mfe("heat", {0});
        require(heat.reuse_stats().expert_loads == before.expert_loads,
            "repeated-token expert heat was lost under cache pressure");
        const auto scan = heat.grouped_mfe("heat", {0, 1, 2, 3}, {}, true);
        const auto scanned = heat.reuse_stats();
        (void)heat.grouped_mfe("heat", {0});
        require(heat.reuse_stats().expert_loads == scanned.expert_loads,
            "prefill scan evicted a hot decode expert");
        const auto x = mlx::core::full({1, input}, 0.125f, mlx::core::float16);
        const auto ids = mlx::core::array({0, 1, 2, 3}, mlx::core::Shape{1, 4});
        const auto copied = scan.routed_matmul(x, ids);
        bool packed_copy = false;
        std::function<void(const mlx::core::array&)> inspect_copy = [&](const auto& node) {
            if (!node.has_primitive()) return;
            if (std::string_view(node.primitive().name()) == "PackedCopyPrimitive") { packed_copy = true; return; }
            for (const auto& input : node.inputs()) inspect_copy(input);
        };
        inspect_copy(copied);
        require(packed_copy, "expert stream assembly did not dispatch the packed copy kernel");
        require(evaluated_floats(copied) ==
            evaluated_floats(mfq::metal::MlxMfeWeight::from_blob(heat_blob).routed_matmul(x, ids)),
            "scan cache eviction released active expert storage");
        const auto reads = heat.reuse_stats();
        require(reads.bytes_read == reads.expert_loads * output * (input / 2 + input / 32)
            && reads.read_calls == 2 * reads.expert_loads && reads.evictions > 0
            && reads.io_seconds > 0 && reads.prepare_seconds > 0 && reads.view_seconds > 0,
            "expert load telemetry lost payload bytes, file reads, timings or evictions");
        require(heat.resident_packed_bytes() <= heat.cache_limit_bytes(),
            "weighted expert heat exceeded the cache budget");
    }
    mlx::core::synchronize();
    const auto uncached_before = mfq::metal::MlxMfeWeight::uncached_packed_bytes();
    {
        constexpr int count = 65, rows = 257, columns = 64;
        std::vector<std::int32_t> experts(count);
        std::iota(experts.begin(), experts.end(), 0);
        const auto mx = make_mxfp4(count * rows, columns);
        const auto wire = make_raw_nim2(count, rows, columns, {
            {experts, "MXFP4", {mx.blob, mx.dense, count * rows, columns}, {}},
        });
        const TemporaryMfq source({{"copy", "MFE", wire}});
        const mfq::metal::MfqContainer container(source.path());
        mfq::metal::MlxMfeOffloadCache transient(container, 0);
        const auto paged = transient.grouped_mfe("copy", experts);
        transient.clear();
        const auto x = mlx::core::full({1, columns}, 0.125f, mlx::core::float16);
        const auto ids = mlx::core::array(experts.begin(), mlx::core::Shape{1, count});
        require(evaluated_floats(paged.routed_matmul(x, ids)) ==
            evaluated_floats(mfq::metal::MlxMfeWeight::from_blob(wire).routed_matmul(x, ids)),
            "multi-batch packed copy changed unaligned stream tails or source ownership");
        require(mfq::metal::MlxMfeWeight::uncached_packed_bytes() > uncached_before,
            "large packed output did not account its uncached allocation");
    }
    mlx::core::synchronize();
    require(mfq::metal::MlxMfeWeight::uncached_packed_bytes() == uncached_before,
        "uncached packed output was not released after its GPU owners completed");
    require(mfq::metal::MlxResourceTelemetry::snapshot().expert_payload_bytes == blob.size(),
        "MFE cache hits duplicated SSD backing telemetry");
    const auto before = residency.resident_packed_bytes();
    const auto active_before = mlx::core::get_active_memory();
    const auto released = residency.set_cache_limit(before / 2);
    require(released > 0 && residency.resident_packed_bytes() <= before / 2,
        "shrinking the mixed MFE cache did not evict expert pages");
    require(residency.set_cache_limit(0) > 0 && residency.cached_expert_count() == 0,
        "zero MFE budget retained expert pages");
    require(mlx::core::get_active_memory() < active_before,
        "shrinking MFE cache did not release actual MLX buffers");
    exercise({1, 2, 3, 5, 8, 9});
    require(residency.resident_packed_bytes() == 0,
        "zero MFE budget retained temporary routed weights");
    residency.set_cache_limit(1u << 24);
    exercise({4, 6, 0, 7, 10, 11});
    residency.discard_record("mixed");
    require(
        residency.cached_expert_count() == 0
            && residency.resident_packed_bytes() == 0,
        "mixed streamed MFE record discard mismatch");
}

void test_qwen4_online_expert_offload(bool mtp, bool prepare = false, bool ttt = false, bool nint_predictor = false,
    bool gdn_block = false, bool qsa_group = false) {
    constexpr int hidden = 96, intermediate = 48, experts = 4;
    std::string config = R"JSON({"model_type":"qwen4_exp_text","vocab_size":32,"hidden_size":96,
        "num_hidden_layers":1,"max_position_embeddings":128,"num_attention_heads":6,
        "num_key_value_heads":2,"head_dim":16,"full_attention_interval":1,"hc_count":2,"hc_lowrank":8,
        "layer_types":["full_attention"],"partial_rotary_factor":0.5,
        "linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":16,
        "linear_value_head_dim":16,"linear_conv_kernel_dim":4,"num_experts":4,"num_experts_per_tok":2,
        "moe_intermediate_size":48,"shared_expert_intermediate_size":48,"indexer_n_heads":2,
        "indexer_head_dim":16,"indexer_compress_ratio":2,"indexer_budget":128,"indexer_kv_heads":1,
        "ple_conv_kernel_size":4,"ple_embed_dim":96,"ple_layer_ids":[],"ngram_size":3,
        "ngram_vocab_size_base":100,"heads_per_ngram":4,"split_ngram_parts":2,
        "mtp_num_hidden_layers":0,"eos_token_id":31,"tie_word_embeddings":false})JSON";
    if (gdn_block) {
        auto parsed = nlohmann::json::parse(config);
        parsed["num_hidden_layers"] = 3;
        parsed["full_attention_interval"] = 4;
        parsed["layer_types"] = {"linear_attention", "linear_attention", "linear_attention"};
        if (qsa_group) {
            parsed["full_attention_interval"] = 2;
            parsed["layer_types"][1] = "full_attention";
        }
        parsed["linear_num_key_heads"] = 1;
        parsed["linear_num_value_heads"] = 1;
        parsed["linear_key_head_dim"] = 128;
        parsed["linear_value_head_dim"] = 128;
        config = parsed.dump();
    }
    if (mtp) {
        const std::string field = "\"mtp_num_hidden_layers\":0";
        config.replace(config.find(field), field.size(), "\"mtp_num_hidden_layers\":1");
    }
    if (ttt) config.replace(config.find("\"eos_token_id\":31"), 17, "\"eos_token_id\":0");
    std::vector<MappedRecordFixture> records{{"__mfq_asset__/model_config.json", "BLOB", {config.begin(), config.end()}}};
    const auto dense = [&](const std::string& name, std::vector<int> shape, bool norm = false) {
        std::vector<std::uint8_t> blob;
        append<std::uint32_t>(blob, shape.size());
        std::size_t count = 1;
        for (const auto dimension : shape) { append<std::int64_t>(blob, dimension); count *= dimension; }
        for (std::size_t index = 0; index < count; ++index)
            append<std::uint16_t>(blob, norm ? 0x3c00 :
                (ttt ? ((index % 5) ? 0x2c00 : 0xac00) : ((index % 5) ? 0x1800 : 0x9800)));
        records.push_back({name, "F16", std::move(blob)});
    };
    const auto mhc = [&](const std::string& root, bool injection) {
        dense(root + ".pre.norm.weight", {hidden * 2}, true);
        dense(root + ".pre.down.weight", {8, hidden * 2});
        dense(root + ".pre.up.weight", {hidden * 2, 8});
        if (injection) dense(root + ".post.inject.weight", {2, hidden * 2});
    };
    dense("model.token_embedding.weight", {32, hidden});
    dense("model.output.weight", {32, hidden});
    mhc("model.mhc", false);
    mhc("model.block.0.attention.mhc", true);
    mhc("model.block.0.mlp.mhc", true);
    const std::string attention = "model.block.0.attention";
    dense(attention + ".query.weight", {192, hidden});
    dense(attention + ".key.weight", {32, hidden});
    dense(attention + ".value.weight", {32, hidden});
    dense(attention + ".output.weight", {hidden, hidden});
    dense(attention + ".query_norm.weight", {16}, true);
    dense(attention + ".key_norm.weight", {16}, true);
    dense(attention + ".indexer.query_key.weight", {48, hidden});
    dense(attention + ".indexer.query_norm.weight", {16}, true);
    dense(attention + ".indexer.key_norm.weight", {16}, true);
    const std::string mlp = "model.block.0.mlp";
    dense(mlp + ".router.weight", {experts, hidden});
    dense(mlp + ".shared_expert.gate.weight", {intermediate, hidden});
    dense(mlp + ".shared_expert.up.weight", {intermediate, hidden});
    if (nint_predictor) {
        auto tensor = make_nint_v2_tensor(hidden, intermediate, 23, 24);
        for (int row = 0; row < hidden; ++row) {
            const std::uint16_t scale = 0x1400, minimum = 0x1000;
            std::memcpy(tensor.blob.data() + 42 + row * 2, &scale, 2);
            std::memcpy(tensor.blob.data() + 42 + hidden * 2 + row * 2, &minimum, 2);
        }
        records.push_back({mlp + ".shared_expert.down.weight", "NINT", std::move(tensor.blob)});
    } else dense(mlp + ".shared_expert.down.weight", {hidden, intermediate});
    dense(mlp + ".shared_expert.router.weight", {1, hidden});
    for (const auto& projection : {std::string("gate"), std::string("up"), std::string("down")}) {
        const int rows = projection == "down" ? hidden : intermediate;
        const int columns = projection == "down" ? intermediate : hidden;
        auto tensor = make_nint_v2_tensor(experts * rows, columns, projection == "up" ? 31 : 17, 24);
        for (int row = 0; row < experts * rows; ++row) {
            const std::uint16_t scale = 0x1400, minimum = 0x1000;
            std::memcpy(tensor.blob.data() + 42 + row * 2, &scale, 2);
            std::memcpy(tensor.blob.data() + 42 + experts * rows * 2 + row * 2, &minimum, 2);
        }
        records.push_back({mlp + ".experts." + projection + ".weight", "MFE",
            make_raw_nim2(experts, rows, columns, {{{0, 1, 2, 3}, "NINTv2",
                tensor, {}}})});
    }
    if (gdn_block) {
        const auto attention_records = records;
        std::erase_if(records, [](const auto& record) {
            return record.name.starts_with("model.block.0.attention.") &&
                !record.name.starts_with("model.block.0.attention.mhc.");
        });
        const auto first = records;
        for (int layer = 1; layer < 3; ++layer) for (const auto& record : first)
            if (record.name.starts_with("model.block.0."))
                records.push_back({"model.block." + std::to_string(layer) + "." + record.name.substr(14),
                    record.dtype, record.payload});
        for (int layer = 0; layer < 3; ++layer) {
            if (qsa_group && layer == 1) {
                for (const auto& record : attention_records)
                    if (record.name.starts_with("model.block.0.attention.") &&
                        !record.name.starts_with("model.block.0.attention.mhc."))
                        records.push_back({"model.block.1." + record.name.substr(14),
                            record.dtype, record.payload});
                continue;
            }
            const auto root = "model.block." + std::to_string(layer) + ".linear_attention";
            dense(root + ".qkv.weight", {384, hidden});
            dense(root + ".gate.weight", {128, hidden});
            dense(root + ".alpha.weight", {1, hidden});
            dense(root + ".beta.weight", {1, hidden});
            dense(root + ".conv.weight", {384, 1, 4});
            dense(root + ".dt_bias", {1});
            dense(root + ".a", {1});
            dense(root + ".norm.weight", {128}, true);
            dense(root + ".output.weight", {hidden, 128});
        }
    }
    if (mtp) {
        const auto target = records;
        const std::string root = "model.block.0.";
        for (const auto& record : target) if (record.name.starts_with(root)) {
            const auto name = "predictor.block.0." + record.name.substr(root.size());
            if (nint_predictor && record.dtype == "MFE") {
                if (name.ends_with(".up.weight")) continue;
                const bool down = name.ends_with(".down.weight");
                const int output = down ? hidden : 2 * intermediate;
                auto tensor = make_nint_v2_tensor(experts * output,
                    down ? intermediate : hidden, 17, 48, true, 8);
                for (int row = 0; row < experts * output; ++row) {
                    const std::uint16_t scale = 0x1400, minimum = 0x1000;
                    std::memcpy(tensor.blob.data() + 42 + row * 2, &scale, 2);
                    std::memcpy(tensor.blob.data() + 42 + experts * output * 2 + row * 2, &minimum, 2);
                }
                records.push_back({down ? name : "predictor.block.0.mlp.experts.gate_up.weight",
                    "NINT", nint_expert_blob(tensor, experts, output)});
            } else if (ttt && record.dtype == "MFE") {
                const bool down = name.ends_with("down.weight");
                dense(name, {experts, down ? hidden : intermediate, down ? intermediate : hidden});
            } else records.push_back({name, record.dtype, record.payload});
        }
        dense("predictor.embedding_norm.weight", {hidden}, true);
        dense("predictor.hidden_norm.weight", {hidden * 2}, true);
        dense("predictor.fusion.embedding.weight", {hidden, hidden});
        dense("predictor.fusion.hidden.weight", {hidden, hidden});
        mhc("predictor.mhc", false);
    }
    const TemporaryMfq file(records, "qwen4_exp");
    const mfq::metal::MfqContainer container(file.path());
    if (ttt) {
        const auto original_fp16 = mfq::metal::mlx_predequantize_fp16_enabled();
        {
            mfq::metal::MlxFp16WeightScope fp16(true);
            bool other_thread = !original_fp16;
            std::thread([&] { other_thread = mfq::metal::mlx_predequantize_fp16_enabled(); }).join();
            require(mfq::metal::mlx_predequantize_fp16_enabled() && other_thread == original_fp16,
                "FP16 predictor loading changed another loading thread");
        }
        require(mfq::metal::mlx_predequantize_fp16_enabled() == original_fp16, "FP16 weight scope leaked");
        setenv("MFQ_MTP_TTT", "1", 1);
        struct ResetTtt { ~ResetTtt() { unsetenv("MFQ_MTP_TTT"); unsetenv("MFQ_MTP_SOURCE"); } } reset_ttt;
        auto packed_records = records;
        for (auto& record : packed_records) if (record.name == "predictor.fusion.hidden.weight") record.dtype = "NINTv2";
        const TemporaryMfq packed_file(packed_records, "qwen4_exp");
        const mfq::metal::MfqContainer packed_container(packed_file.path());
        auto bf16_records = records;
        for (auto& record : bf16_records) if (record.name.starts_with("predictor.") && record.dtype == "F16") {
            record.dtype = "BF16";
            std::uint32_t dimensions;
            std::memcpy(&dimensions, record.payload.data(), sizeof(dimensions));
            for (std::size_t offset = 4 + dimensions * 8; offset < record.payload.size(); offset += 2) {
                std::uint16_t value;
                std::memcpy(&value, record.payload.data() + offset, 2);
                value = value == 0x3c00 ? 0x3f80 : (value == 0x2c00 ? 0x3d80 : 0xbd80);
                std::memcpy(record.payload.data() + offset, &value, 2);
            }
        }
        const TemporaryMfq bf16_file(bf16_records, "qwen4_exp");
        const mfq::metal::MfqContainer bf16_container(bf16_file.path());
        {
            mfq::metal::MlxFp16WeightScope fp16(true);
            const auto linear = mfq::metal::MlxLinear::load(bf16_container, "predictor.fusion.hidden.weight");
            require(linear.dense_weight_ref()->dtype() == mlx::core::float16, "BF16 predictor was not kept as FP16");
        }
        bool rejected = false;
        try { (void)mfq::metal::MlxQwen4CausalLm::load(packed_container, 128); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "TTT silently treated a quantized predictor as original FP16");
        setenv("MFQ_MTP_SOURCE", bf16_file.path().c_str(), 1);
        auto runtime = mfq::metal::MlxQwen4CausalLm::load(packed_container, 128);
        unsetenv("MFQ_MTP_SOURCE");
        require(!runtime.mtp_cache_fingerprint().empty(), "TTT predictor cache has no fingerprint");
        char directory[] = "/tmp/mfq-qwen4-ttt-test-XXXXXX";
        require(mkdtemp(directory) != nullptr, "TTT fixture directory creation failed");
        struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{directory};
        const auto hf_directory = std::filesystem::path(directory) / "original-hf";
        std::filesystem::create_directories(hf_directory / ".mfq-assets" / "hf");
        std::ofstream(hf_directory / "config.json") << config;
        auto header = nlohmann::json::object(), aliases = nlohmann::json::object();
        std::vector<std::uint8_t> payload;
        const auto add_tensor = [&](const std::string& name, const std::string& dtype,
            const std::vector<std::int64_t>& shape, std::span<const std::uint8_t> data) {
            const auto offset = payload.size();
            payload.insert(payload.end(), data.begin(), data.end());
            const auto source_name = name.starts_with("predictor.") ? "mtp." + name.substr(10) : name;
            header[source_name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {offset, payload.size()}}};
            aliases[name] = source_name;
        };
        for (const auto& record : bf16_records) {
            if (record.dtype != "F16" && record.dtype != "BF16") continue;
            std::uint32_t dimensions;
            std::memcpy(&dimensions, record.payload.data(), 4);
            std::vector<std::int64_t> shape(dimensions);
            std::memcpy(shape.data(), record.payload.data() + 4, dimensions * 8);
            const auto data = std::span(record.payload).subspan(4 + dimensions * 8);
            if (record.name.starts_with("predictor.") && dimensions == 3) {
                const auto dot = record.name.rfind('.', record.name.size() - 8);
                const auto size = std::size_t(shape[1] * shape[2]) * 2;
                for (int expert = 0; expert < shape[0]; ++expert)
                    add_tensor(record.name.substr(0, dot + 1) + std::to_string(expert) + record.name.substr(dot),
                        record.dtype, {shape[1], shape[2]}, data.subspan(expert * size, size));
            } else add_tensor(record.name, record.dtype, shape, data);
        }
        std::ofstream(hf_directory / ".mfq-assets" / "hf" / "source_tensor_map.json")
            << nlohmann::json({{"schema", "mfq.hf-source-map"}, {"version", 1}, {"canonical_to_source", aliases}}).dump();
        const auto header_text = header.dump();
        std::vector<std::uint8_t> safetensors;
        append<std::uint64_t>(safetensors, header_text.size());
        safetensors.insert(safetensors.end(), header_text.begin(), header_text.end());
        safetensors.insert(safetensors.end(), payload.begin(), payload.end());
        std::ofstream weights(hf_directory / "model.safetensors", std::ios::binary);
        weights.write(reinterpret_cast<const char*>(safetensors.data()), safetensors.size());
        weights.close();
        setenv("MFQ_MTP_SOURCE", hf_directory.c_str(), 1);
        {
            auto hf_runtime = mfq::metal::MlxQwen4CausalLm::load(packed_container, 128);
            mfq::metal::MlxSamplingParams sampling;
            sampling.temperature = 0;
            sampling.mtp_max_draft_tokens = 2;
            require(hf_runtime.generate({1, 2, 3}, sampling, 8) > 0, "native HF floating experts could not draft");
        }
        unsetenv("MFQ_MTP_SOURCE");
        runtime.prepare_mtp_ttt(directory);
        const auto metric = [&](const char* name) {
            for (const auto& [key, value] : runtime.mtp_session_metrics()) if (key == name) return value;
            throw std::runtime_error("missing TTT model metric");
        };
        mfq::metal::MlxSamplingParams sampling;
        sampling.temperature = 0;
        sampling.top_k = 1;
        sampling.mtp_max_draft_tokens = 2;
        const auto original = evaluated_floats(runtime.forward(mlx::core::array({1, 2, 3}, mlx::core::Shape{1, 3}), false));
        for (int request = 0; request < 12 && metric("mtp_ttt_submitted") == 0; ++request) {
            runtime.reset_generation_state();
            runtime.begin_mtp_session("a");
            require(runtime.generate({1, 2, 3}, sampling, 16) > 0, "TTT fixture could not generate");
            runtime.end_mtp_session();
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (metric("mtp_ttt_completed") < metric("mtp_ttt_submitted")) {
            require(std::chrono::steady_clock::now() < deadline, "TTT model worker timeout");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        for (const auto& [key, value] : runtime.mtp_session_metrics()) std::cout << key << '=' << value << '\n';
        require(metric("mtp_ttt_submitted") > 0 && metric("mtp_ttt_updates") > 0 &&
            metric("mtp_ttt_failed") == 0, "real MTP verification did not train the adapter");
        runtime.begin_mtp_session("a");
        runtime.generate({1, 2, 3}, sampling, 8);
        require(metric("mtp_ttt_active_version") > 0, "trained adapter did not enter draft inference");
        runtime.end_mtp_session();
        runtime.clear_cache();
        const auto after = evaluated_floats(runtime.forward(mlx::core::array({1, 2, 3}, mlx::core::Shape{1, 3}), false));
        require(after == original, "TTT changed the main model forward");
        runtime.set_expert_cache_limit(0);
        runtime.begin_mtp_session("a");
        require(runtime.generate({1, 2, 3}, sampling, 8) > 0 && runtime.reclaimable_expert_bytes() == 0,
            "expert paging attempted to offload the frozen FP16 predictor");
        runtime.end_mtp_session();
        runtime.close_mtp_session("a");
        runtime.clear_mtp_sessions();
        std::cout << "Qwen4 MTP -> ANE KL -> next-round LoRA integration passed\n";
        return;
    }
    auto reference = mfq::metal::MlxQwen4CausalLm::load(container, 128);
    std::optional<mfq::metal::MlxKernelPreparation> preparation;
    if (prepare) {
        preparation.emplace(128, 128);
        preparation->set_routes(2);
    }
    auto paged = mfq::metal::MlxQwen4CausalLm::load(container, 128);
    if (preparation) {
        preparation->collect_sampling(32, {});
        require(preparation->unsupported().empty(), "model preparation has unsupported native kernels");
        const auto memory = mlx::core::get_active_memory();
        mfq_kernel_prepare_phase(1);
        preparation->finish();
        require(mlx::core::get_active_memory() == memory, "model compilation allocated tensor buffers");
        mfq_kernel_prepare_phase(2);
        mfq::metal::MlxSamplingParams sampling;
        sampling.temperature = 0.8;
        sampling.top_k = 16;
        mfq::metal::warm_mlx_inference({1, 2, 3}, 128, 64, sampling, mtp,
            [&](const auto& tokens, const auto& parameters, int count) {
                require(paged.generate(tokens, parameters, count) == count,
                    "model warmup stopped before covering its requested steps");
            }, [&] {
                paged.reset_generation_state();
            });
        require(paged.cache_position() == 0 && paged.kv_cache_bytes() == 0,
            "warmup left live KV or recurrent state");
        require(!paged.last_mtp_stats().used && paged.last_mtp_stats().drafted_tokens == 0,
            "warmup left request statistics");
        sampling.temperature = 0.0;
        sampling.top_k = 0;
        for (const int rows : {48, 127}) {
            std::vector<std::int64_t> tokens(rows);
            for (int row = 0; row < rows; ++row) tokens[row] = 1 + row % 3;
            sampling.enable_mtp = false;
            require(paged.generate(tokens, sampling, 2) == 2,
                "prepared model could not handle a new prefill length");
        }
        paged.reset_generation_state();
    }
    const auto prompt = mlx::core::array({1, 2, 3}, mlx::core::Shape{1, 3});
    mlx::core::eval(reference.forward(prompt), paged.forward(prompt));
    if (preparation) {
        const auto next = mlx::core::array({4}, mlx::core::Shape{1, 1});
        const auto expected = evaluated_floats(reference.forward(next));
        const auto actual = evaluated_floats(paged.forward(next));
        for (std::size_t index = 0; index < actual.size(); ++index)
            require_close(actual[index], expected[index], 1e-3f);
        if (mtp) {
            paged.generate({1, 2, 3}, {}, 8);
            require(paged.last_mtp_stats().used, "prepared model did not exercise MTP");
        }
        reference.clear_cache();
        paged.clear_cache();
        mlx::core::eval(reference.forward(prompt), paged.forward(prompt));
        mfq_kernel_prepare_phase(3);
    }
    const auto kv = paged.kv_cache_bytes();
    require(kv > 0 && paged.reclaimable_expert_bytes() > 0, "Qwen4 full-resident fixture has no weights or KV");
    std::size_t layer_payload = 0, predictor_payload = 0;
    for (const auto& record : records) {
        if (record.name.starts_with("model.block.") && record.name.find(".mlp.experts.") != std::string::npos)
            layer_payload += record.payload.size();
        if (record.name.starts_with("predictor.block.0.mlp.experts.")) predictor_payload += record.payload.size();
    }
    require(paged.ssd_expert_payload_bytes() == 0 &&
            mfq::metal::MlxResourceTelemetry::snapshot().expert_payload_bytes == 0,
        "full-resident Qwen4 model reported SSD expert backing");
    if (mtp) {
        const auto full = paged.resident_full_expert_bytes();
        const auto before = mlx::core::get_active_memory();
        const auto budget = full * 3 / 4;
        paged.set_expert_cache_limit(budget);
        const auto remaining = paged.resident_full_expert_bytes();
        require(remaining > 0 && remaining < full
                && remaining + paged.expert_cache_limit_bytes() == budget,
            "Qwen4 partial shrink did not share the budget with retained expert banks");
        require(paged.reclaimable_expert_bytes() <= full * 3 / 4
                && mlx::core::get_active_memory() < before,
            "Qwen4 partial shrink did not release actual expert buffers");
        require(paged.cache_position() == 3 && paged.kv_cache_bytes() == kv,
            "Qwen4 partial expert shrink discarded the live context");
        require(paged.ssd_expert_payload_bytes() == layer_payload &&
                mfq::metal::MlxResourceTelemetry::snapshot().expert_payload_bytes == layer_payload,
            "Qwen4 partial shrink reported unspilled expert banks as SSD backing");
        const auto input = mlx::core::array({4}, mlx::core::Shape{1, 1});
        const auto expected = evaluated_floats(reference.forward(input));
        const auto actual = evaluated_floats(paged.forward(input));
        for (std::size_t index = 0; index < actual.size(); ++index)
            require_close(actual[index], expected[index], 1e-3f);
        reference.clear_cache();
        paged.clear_cache();
        mlx::core::eval(reference.forward(prompt), paged.forward(prompt));
    }
    const auto active = mlx::core::get_active_memory();
    paged.set_expert_cache_limit(0);
    require(mlx::core::get_active_memory() < active, "Qwen4 expert offload did not release actual buffers");
    require(paged.cache_position() == 3 && paged.kv_cache_bytes() == kv,
        "Qwen4 expert offload discarded the live context");
    require(paged.ssd_expert_payload_bytes() == layer_payload + predictor_payload &&
            mfq::metal::MlxResourceTelemetry::snapshot().expert_payload_bytes == layer_payload + predictor_payload,
        "Qwen4 full offload did not report its active SSD backing records");
    for (const int token : {4, 5, 6}) {
        const auto input = mlx::core::array({token}, mlx::core::Shape{1, 1});
        const auto expected = evaluated_floats(reference.forward(input));
        const auto actual = evaluated_floats(paged.forward(input));
        for (std::size_t index = 0; index < actual.size(); ++index) require_close(actual[index], expected[index], 1e-3f);
        require(paged.reclaimable_expert_bytes() == 0, "Qwen4 zero-budget cache retained routed experts");
    }
    paged.set_expert_cache_limit(1 << 20);
    const auto input = mlx::core::array({7}, mlx::core::Shape{1, 1});
    const auto expected = evaluated_floats(reference.forward(input));
    const auto actual = evaluated_floats(paged.forward(input));
    for (std::size_t index = 0; index < actual.size(); ++index) require_close(actual[index], expected[index], 1e-3f);
    require(paged.reclaimable_expert_bytes() > 0 && paged.cache_position() == 7,
        "Qwen4 cache could not regrow without reloading the model");
    if (gdn_block) {
        const auto before = paged.ssd_expert_cache_stats()->device_route_hits;
        for (int step = 0; step < 24; ++step) {
            const auto ids = mlx::core::array({1 + step % 7}, mlx::core::Shape{1, 1});
            const auto oracle = evaluated_floats(reference.forward(ids));
            const auto result = evaluated_floats(paged.forward(ids));
            for (std::size_t index = 0; index < result.size(); ++index)
                require_close(result[index], oracle[index], 1e-3f);
        }
        require(paged.ssd_expert_cache_stats()->device_route_hits > before + 12,
            "GDN block did not use cached device routes");
        paged.set_expert_cache_limit(0);
        const auto ids = mlx::core::array({9}, mlx::core::Shape{1, 1});
        const auto oracle = evaluated_floats(reference.forward(ids));
        const auto result = evaluated_floats(paged.forward(ids));
        for (std::size_t index = 0; index < result.size(); ++index)
            require_close(result[index], oracle[index], 1e-3f);
        require(paged.cache_position() == 32, "GDN route grouping advanced state twice");
        if (qsa_group) {
            const auto checkpoint = paged.capture_text_session_state(std::vector<std::int64_t>(32, 1));
            require(checkpoint.layers[1].position == 32 && checkpoint.layers[1].kv &&
                    checkpoint.layers[1].kv->position == 32 && checkpoint.layers[1].pooled_keys &&
                    checkpoint.layers[1].pooled_keys->shape(1) == 16,
                "QSA route grouping advanced KV or pooled index keys twice");
        }
    }
    if (mtp) {
        mfq::metal::MlxSamplingParams sampling;
        paged.generate({1, 2, 3}, sampling, 8);
        require(paged.last_mtp_stats().used, "Qwen4 paging fixture did not exercise MTP");
        const auto bytes = paged.kv_cache_bytes();
        const auto position = paged.cache_position();
        paged.set_expert_cache_limit(0);
        require(paged.kv_cache_bytes() == bytes && paged.cache_position() == position,
            "Qwen4 expert shrink discarded target or MTP KV");
    }
    for (const int rows : {32, 65, 127}) {
        paged.set_expert_cache_limit(0);
        std::vector<std::int64_t> ids(rows);
        for (int index = 0; index < rows; ++index) ids[index] = 1 + index % 7;
        const auto input = mlx::core::array(ids.begin(), mlx::core::Shape{1, rows});
        const auto expected = evaluated_floats(reference.forward(input, false));
        const auto actual = evaluated_floats(paged.forward(input, false));
        require(actual.size() == expected.size(), "SSD sorted prefill changed the logit shape");
        for (std::size_t index = 0; index < actual.size(); ++index)
            require_close(actual[index], expected[index], 1e-3f);
        require(paged.reclaimable_expert_bytes() == 0, "sorted SSD prefill ignored a zero expert budget");
    }
    if (!prepare && !nint_predictor) {
        std::vector<std::shared_ptr<mfq::metal::MlxQwen4RequestCache>> reference_caches, paged_caches;
        for (int index = 0; index < 3; ++index) {
            reference_caches.push_back(reference.make_request_cache());
            paged_caches.push_back(paged.make_request_cache());
            (void)evaluated_floats(reference.prefill_request(*reference_caches.back(), {1 + index, 2 + index}));
            (void)evaluated_floats(paged.prefill_request(*paged_caches.back(), {1 + index, 2 + index}));
        }
        paged.set_expert_cache_limit(0);
        paged.set_expert_cache_limit(1u << 20);
        const auto before = paged.ssd_expert_cache_stats()->loads;
        const std::vector<std::vector<std::int64_t>> windows{{4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
        const auto expected = evaluated_floats(reference.verify_requests(reference_caches, windows).first);
        const auto actual = evaluated_floats(paged.verify_requests(paged_caches, windows).first);
        require(actual.size() == expected.size(), "batched SSD verification changed logit shape");
        for (std::size_t index = 0; index < actual.size(); ++index)
            require_close(actual[index], expected[index], 1e-3f);
        require(paged.ssd_expert_cache_stats()->loads > before && paged.reclaimable_expert_bytes() > 0,
            "batched SSD verification incorrectly used prefill-only whole-layer reads");
    }
    if (nint_predictor) {
        auto cold = mfq::metal::MlxQwen4CausalLm::load(container, 128, 1);
        reference.clear_cache();
        const auto expected = evaluated_floats(reference.forward(prompt));
        const auto actual = evaluated_floats(cold.forward(prompt));
        for (std::size_t index = 0; index < actual.size(); ++index)
            require_close(actual[index], expected[index], 1e-3f);
        mfq::metal::MlxSamplingParams sampling;
        cold.generate({1, 2, 3}, sampling, 8);
        require(cold.last_mtp_stats().used, "SSD-loaded 3D NINT predictor did not execute MTP");
    }
}

void test_materialized_mfe_projections() {
    constexpr int experts = 4;
    constexpr int output = 1024;
    constexpr int input = 768;
    const auto nint = make_nint_v2_tensor(2 * output, input, 17, 24);
    const auto vq = make_jsc_nvq(2 * output, input);
    const auto blob = make_raw_nim2(experts, output, input, {
        {{0, 2}, "NINTv2", nint, {}},
        {{3, 1}, vq.dtype, {vq.blob, vq.dense, 2 * output, input}, vq.runtime},
    });
    const auto up_vq = make_nvq1_s(2 * output, input);
    const auto up_blob = make_raw_nim2(experts, output, input, {
        {{1, 2}, "NINTv2", make_nint_v2_tensor(2 * output, input, 23, 24), {}},
        {{0, 3}, up_vq.dtype, {up_vq.blob, up_vq.dense, 2 * output, input}, {}},
    });
    const auto build = [&] {
        return mfq::metal::MlxMfeWeight::concatenate_projections({
            mfq::metal::MlxMfeWeight::from_blob(blob),
            mfq::metal::MlxMfeWeight::from_blob(up_blob),
        });
    };
    const auto bound_storage_bytes = [&](const mfq::metal::MlxMfeWeight& weight) {
        std::unordered_set<const void*> allocations;
        std::size_t bytes = 0;
        std::function<void(const mlx::core::array&)> visit = [&](const auto& node) {
            if (const auto data = node.data_shared_ptr(); data) {
                const auto buffer = [&] {
                    if constexpr (requires { data->buffer(); }) return data->buffer().ptr();
                    else return data->buffer.ptr();
                }();
                if (buffer && allocations.insert(buffer).second) bytes += node.buffer_size();
            }
            if (node.has_primitive())
                for (const auto& input : node.inputs()) visit(input);
        };
        visit(weight.routed_swiglu(
            mlx::core::zeros(mlx::core::Shape{1, input}, mlx::core::float16),
            mlx::core::array({0, 1}, mlx::core::Shape{1, 2}, mlx::core::int32)));
        return bytes;
    };
    mlx::core::synchronize();
    const auto initial = mlx::core::get_active_memory();
    const auto reference = build();
    const auto exercise = [&](const mfq::metal::MlxMfeWeight& weight, int tokens, bool sorted) {
        std::vector<float> source(tokens * input);
        for (std::size_t i = 0; i < source.size(); ++i) {
            source[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 4096.0f;
        }
        std::vector<std::int32_t> ids(tokens * 2);
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = (i * 3 + 1) % experts;
        auto x = mlx::core::astype(mlx::core::array(
            source.begin(), mlx::core::Shape{tokens, input}), mlx::core::float16);
        auto routes = mlx::core::array(ids.begin(), mlx::core::Shape{tokens, 2});
        if (!sorted) return evaluated_floats(weight.routed_swiglu(x, routes));
        auto order = mlx::core::astype(mlx::core::argsort(
            mlx::core::reshape(routes, mlx::core::Shape{tokens * 2})), mlx::core::int32);
        auto plan = weight.build_grouped_mmq_plan(routes, order, 32);
        return evaluated_floats(weight.routed_matmul_sorted(
            x, routes, order, false, true, 0.0f, &plan));
    };
    std::vector<std::vector<float>> expected;
    for (int tokens : {1, 5, 32, 65}) expected.push_back(exercise(reference, tokens, tokens >= 32));
    mlx::core::synchronize();
    mlx::core::clear_cache();
    const auto before = mlx::core::get_active_memory();
    const auto candidate = build().materialize_packed_projections();
    require(candidate.supports_grouped_mmq(), "materialized projections lost grouped MMQ");
    int run = 0;
    for (int tokens : {1, 5, 32, 65}) {
        const auto actual = exercise(candidate, tokens, tokens >= 32);
        require(actual.size() == expected[run].size(), "materialized projection shape mismatch");
        for (std::size_t i = 0; i < actual.size(); ++i) {
            require_close(actual[i], expected[run][i], 1e-6f);
        }
        ++run;
    }
    // Allow small descriptor/allocator overhead, but never a second Gate/Up copy.
    mlx::core::synchronize();
    const auto allocated = mlx::core::get_active_memory() - before;
    const auto bound_bytes = bound_storage_bytes(candidate);
    require(allocated < bound_bytes + (64u << 10),
        "materialized projections retained duplicate source weight buffers: allocated=" +
        std::to_string(allocated) + " bound=" + std::to_string(bound_bytes));
    require(before - initial > allocated + candidate.packed_nbytes() / 2,
        "projection fixture did not reproduce lazy duplicate storage");

    mlx::core::clear_cache();
    const auto before_direct = mlx::core::get_active_memory();
    const auto direct = [&] {
        auto gate_bytes = blob;
        auto up_bytes = up_blob;
        const std::array<std::span<const std::uint8_t>, 2> sources{gate_bytes, up_bytes};
        return mfq::metal::MlxMfeWeight::from_projection_blobs(sources);
    }();
    // Wire vectors are gone before any GPU use; packed buffers must own data.
    require(direct.supports_grouped_mmq(), "direct projections lost grouped MMQ");
    run = 0;
    for (int tokens : {1, 5, 32, 65}) {
        const auto actual = exercise(direct, tokens, tokens >= 32);
        require(actual == expected[run++], "direct packed projection arithmetic changed");
    }
    mlx::core::synchronize();
    const auto direct_allocated = mlx::core::get_active_memory() - before_direct;
    const auto direct_bound_bytes = bound_storage_bytes(direct);
    require(direct_allocated < direct_bound_bytes + (64u << 10),
        "direct projection loader retained duplicate buffers: allocated=" +
        std::to_string(direct_allocated) + " bound=" + std::to_string(direct_bound_bytes));
}

void test_direct_mfe_projection_tails() {
    constexpr int experts = 3;
    constexpr int output = 33;
    constexpr int input = 40;
    const auto nint = make_nint_v2_tensor(output, input, 71, 28, true);
    const auto wide = make_jsc_nvq(output, input, "NVQ3J-L", 6, 4, 10, false, 4, -1);
    const auto compact = make_nvq1_s(output, input);
    const auto gate = make_raw_nim2(experts, output, input, {
        {{2}, "NINT", nint, {}},
        {{0}, wide.dtype, {wide.blob, wide.dense, output, input}, {}},
        {{1}, compact.dtype, {compact.blob, compact.dense, output, input}, {}},
    });
    const auto up = make_raw_nim2(experts, output, input, {
        {{0}, "NINT", make_nint_v2_tensor(output, input, 73, 23, true), {}},
        {{1}, wide.dtype, {wide.blob, wide.dense, output, input}, {}},
        {{2}, compact.dtype, {compact.blob, compact.dense, output, input}, {}},
    });
    const std::array<std::span<const std::uint8_t>, 2> sources{gate, up};
    const auto candidate = mfq::metal::MlxMfeWeight::from_projection_blobs(sources);
    const auto reference = mfq::metal::MlxMfeWeight::concatenate_projections({
        mfq::metal::MlxMfeWeight::from_blob(gate),
        mfq::metal::MlxMfeWeight::from_blob(up),
    }).materialize_packed_projections();
    for (int tokens : {1, 2, 5, 32, 65}) {
        std::vector<float> values(tokens * input);
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 4096.0f;
        }
        std::vector<std::int32_t> ids(tokens * experts);
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = (i * 2 + 1) % experts;
        const auto x = mlx::core::astype(mlx::core::array(values.begin(),
            mlx::core::Shape{tokens, input}), mlx::core::float16);
        const auto routes = mlx::core::array(ids.begin(), mlx::core::Shape{tokens, experts});
        require(evaluated_floats(candidate.routed_matmul(x, routes)) ==
                evaluated_floats(reference.routed_matmul(x, routes)),
            "direct projection tail matmul mismatch");
        require(evaluated_floats(candidate.routed_swiglu(x, routes)) ==
                evaluated_floats(reference.routed_swiglu(x, routes)),
            "direct projection tail SwiGLU mismatch");
        const auto order = mlx::core::astype(mlx::core::argsort(mlx::core::reshape(
            routes, mlx::core::Shape{tokens * experts})), mlx::core::int32);
        const auto plan = candidate.build_grouped_mmq_plan(routes, order, 32);
        const auto reference_plan = reference.build_grouped_mmq_plan(routes, order, 32);
        require(evaluated_floats(candidate.routed_matmul_sorted(
                    x, routes, order, false, true, 0.0f, &plan)) ==
                evaluated_floats(reference.routed_matmul_sorted(
                    x, routes, order, false, true, 0.0f, &reference_plan)),
            "direct projection sorted tail mismatch");
    }

    bool rejected = false;
    try {
        mfq::metal::MlxMfeWeight::from_projection_blobs({});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "empty projection source list must be rejected");
    auto mismatched = up;
    const std::uint32_t invalid_output = output + 1;
    std::memcpy(mismatched.data() + 8, &invalid_output, sizeof(invalid_output));
    const std::array<std::span<const std::uint8_t>, 2> invalid_sources{gate, mismatched};
    rejected = false;
    try {
        mfq::metal::MlxMfeWeight::from_projection_blobs(invalid_sources);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "mismatched projection geometry must be rejected");
}

void test_banked_vq_input_views() {
    using namespace mlx::core;
    constexpr int input = 96;
    constexpr int output = 17;
    constexpr int route_count = 3;
    const auto gate = make_vq_moe_fixture({
        make_jsc_nvq(output, input, "NVQ3J-L", 6, 4, 10, false, 4, -1),
        make_jsc_nvq(output, input, "NVQ3J-512", 3, 4, 9),
    });
    const auto up = make_vq_moe_fixture({
        make_jsc_nvq(output, input, "NVQ3J-512", 3, 4, 9),
        make_jsc_nvq(output, input, "NVQ3J-L", 6, 4, 10, false, 4, -1),
    });
    const auto single = mfq::metal::MlxMfeWeight::from_blob(gate.blob);
    const auto pair = mfq::metal::MlxMfeWeight::from_projection_blobs(
        std::array<std::span<const std::uint8_t>, 2>{gate.blob, up.blob});
    for (int tokens = 2; tokens <= 6; ++tokens) {
        std::vector<std::int32_t> ids(tokens * route_count);
        for (std::size_t index = 0; index < ids.size(); ++index)
            ids[index] = index % route_count == 2 ? -1 : index % route_count;
        const auto routes = array(ids.begin(), Shape{tokens, route_count});
        for (const bool shared : {false, true}) {
            std::vector<float> values(tokens * (shared ? 1 : route_count) * input);
            for (std::size_t index = 0; index < values.size(); ++index)
                values[index] = float(int((index * 13 + 7) % 29) - 14) / 1024.0f;
            for (const auto dtype : {float16, float32}) {
                std::vector<std::vector<float>> reference;
                for (const int offset : {4, 1}) {
                    std::vector<float> backing(values.size() + offset, 17.0f);
                    std::copy(values.begin(), values.end(), backing.begin() + offset);
                    const auto x = reshape(slice(astype(array(backing.begin(),
                        Shape{static_cast<int>(backing.size())}), dtype), Shape{offset},
                        Shape{static_cast<int>(backing.size())}), shared
                            ? Shape{tokens, input} : Shape{tokens, route_count, input});
                    const std::vector<std::vector<float>> actual{
                        evaluated_floats(single.routed_matmul(x, routes)),
                        evaluated_floats(pair.routed_matmul(x, routes)),
                        evaluated_floats(pair.routed_swiglu(x, routes)),
                    };
                    if (offset == 4) reference = actual;
                    else require(actual == reference, "banked VQ aligned/unaligned input mismatch");
                    require(x.offset() == static_cast<std::size_t>(offset) * x.itemsize(),
                        "banked VQ input view lost its offset");
                    for (int token = 0; token < tokens; ++token) {
                        for (int route = 0; route < route_count; ++route) {
                            const int row = token * route_count + route;
                            const int source_row = shared ? token : row;
                            const std::vector<float> source(values.begin() + source_row * input,
                                values.begin() + (source_row + 1) * input);
                            for (int column = 0; column < output; ++column) {
                                const float g = ids[row] < 0 ? 0.0f
                                    : routed_vq_dot(source, gate, ids[row], column);
                                const float u = ids[row] < 0 ? 0.0f
                                    : routed_vq_dot(source, up, ids[row], column);
                                require_close(actual[0][row * output + column], g, 2e-3f);
                                require_close(actual[1][row * 2 * output + column], g, 2e-3f);
                                require_close(actual[1][row * 2 * output + output + column], u, 2e-3f);
                                require_close(actual[2][row * output + column],
                                    g / (1.0f + std::exp(-g)) * u, 6e-3f);
                            }
                        }
                    }
                }
            }
        }
    }
}

void test_direct_mfe_projection_families() {
    const auto check = [](const char* label, const std::vector<std::uint8_t>& gate,
                          const std::vector<std::uint8_t>& up) {
        const auto gate_weight = mfq::metal::MlxMfeWeight::from_blob(gate);
        const auto up_weight = mfq::metal::MlxMfeWeight::from_blob(up);
        const auto reference = mfq::metal::MlxMfeWeight::concatenate_projections({
            gate_weight, up_weight,
        }).materialize_packed_projections();
        const auto candidate = [&] {
            auto gate_wire = gate;
            auto up_wire = up;
            const std::array<std::span<const std::uint8_t>, 2> sources{gate_wire, up_wire};
            auto result = mfq::metal::MlxMfeWeight::from_projection_blobs(sources);
            // Dense/MX payload views must have been copied, not retained.
            std::fill(gate_wire.begin(), gate_wire.end(), 0xad);
            std::fill(up_wire.begin(), up_wire.end(), 0xad);
            return result;
        }();
        require(candidate.supports_grouped_mmq() == reference.supports_grouped_mmq(),
            "direct projection family changed grouped dispatch");
        for (int tokens : {1, 5, 32}) {
            std::vector<float> values(tokens * reference.neuron_len());
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 4096.0f;
            }
            std::vector<std::int32_t> ids(tokens * reference.experts());
            for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = i % reference.experts();
            const auto x = mlx::core::astype(mlx::core::array(values.begin(),
                mlx::core::Shape{tokens, reference.neuron_len()}), mlx::core::float16);
            const auto routes = mlx::core::array(ids.begin(),
                mlx::core::Shape{tokens, reference.experts()});
            const auto actual = evaluated_floats(candidate.routed_matmul(x, routes));
            const auto independent = evaluated_floats(mlx::core::concatenate({
                gate_weight.routed_matmul(x, routes), up_weight.routed_matmul(x, routes)}, -1));
            require(actual == independent,
                std::string("direct projection disagrees with independent weights: ") + label);
            const auto expected = evaluated_floats(reference.routed_matmul(x, routes));
            const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
            require(mismatch.first == actual.end(),
                std::string("direct projection family matmul mismatch: ") + label +
                " tokens=" + std::to_string(tokens) + " index=" +
                std::to_string(mismatch.first - actual.begin()) +
                (mismatch.first == actual.end() ? "" : " actual=" +
                    std::to_string(*mismatch.first) + " expected=" + std::to_string(*mismatch.second)));
            require(evaluated_floats(candidate.routed_swiglu(x, routes)) ==
                    evaluated_floats(reference.routed_swiglu(x, routes)),
                std::string("direct projection family SwiGLU mismatch: ") + label);
        }
    };

    const std::vector<std::string> profiles{"NINTv2", "NINT8-0", "BF16", "F16"};
    check("NINT/Q8/dense", make_moe_fixture(profiles, 33, 64, 5, 24).blob,
          make_moe_fixture(profiles, 33, 64, 7, 28).blob);
    const auto mx4 = make_mxfp4(33, 128, 3);
    const auto mx8 = make_mxfp8(33, 128, 5);
    const auto mx_gate = make_raw_nim2(2, 33, 128, {
        {{1}, "MXFP4", {mx4.blob, mx4.dense, 33, 128}, {}},
        {{0}, "MXFP8", {mx8.blob, mx8.dense, 33, 128}, {}},
    });
    const auto mx_up = make_raw_nim2(2, 33, 128, {
        {{0}, "MXFP4", {mx4.blob, mx4.dense, 33, 128}, {}},
        {{1}, "MXFP8", {mx8.blob, mx8.dense, 33, 128}, {}},
    });
    check("MX", mx_gate, mx_up);

    const auto group64 = make_jsc_nvq(33, 40, "NVQ2J-XL", 5, 8, 12, true);
    const auto nvq_l = make_nvq1_l(33, 40);
    const auto grouped_gate = make_raw_nim2(2, 33, 40, {
        {{1}, group64.dtype, {group64.blob, group64.dense, 33, 40}, {}},
        {{0}, nvq_l.dtype, {nvq_l.blob, nvq_l.dense, 33, 40}, {}},
    });
    const auto grouped_up = make_raw_nim2(2, 33, 40, {
        {{0}, nvq_l.dtype, {nvq_l.blob, nvq_l.dense, 33, 40}, {}},
        {{1}, group64.dtype, {group64.blob, group64.dense, 33, 40}, {}},
    });
    check("group64/unaligned tail", grouped_gate, grouped_up);

    const auto sq = make_mxfp4_sq2(5, 64);
    const auto sq_blob = make_raw_nim2(2, 5, 64, {
        {{1}, "MXFP4-SQ2", {sq.blob, sq.dense, 5, 64}, {}},
        {{0}, "NINT4", make_nint_tensor(4, 5, 64, 11), {}},
    });
    check("SQ", sq_blob, sq_blob);
    const auto rotated = make_rotated_nepq0_s(17, 24);
    const auto rotated_blob = make_raw_nim2(2, 17, 24, {
        {{1}, rotated.dtype, {rotated.blob, rotated.dense, 17, 24}, rotated.runtime},
        {{0}, "NINT4", make_nint_tensor(4, 17, 24, 13), {}},
    });
    // The existing embedding-based unpacked oracle cannot handle rotations.
    // Exercise rotated packed dispatch normally; reference mode audits the
    // non-rotated native/dense/SQ families above.
    if (!mfq::metal::mlx_reference_enabled()) {
        check("rotated", rotated_blob, rotated_blob);
    }
}

void test_mixed_decode_row_packing(
    int group_size, int tokens = 1, int output = 17, int input = 96) {
    using namespace mlx::core;
    const auto nint = make_nint_v2_tensor(
        output, input, 17, group_size, true);
    const auto jsc = input == 640 && output == 2560
        ? make_jsc_nvq(output, input, "NVQ3J-L", 6, 4, 10, false, 4, -1)
        : make_jsc_nvq(output, input);
    const auto nvq = make_nvq1_s(output, input);
    const auto gate = mfq::metal::MlxMfeWeight::from_blob(make_raw_nim2(3, output, input, {
        {{0}, "NINTv2", nint, {}},
        {{1}, jsc.dtype, {jsc.blob, jsc.dense, output, input}, jsc.runtime},
        {{2}, nvq.dtype, {nvq.blob, nvq.dense, output, input}, nvq.runtime},
    }));
    const auto up = mfq::metal::MlxMfeWeight::from_blob(make_raw_nim2(3, output, input, {
        {{2}, "NINTv2", nint, {}},
        {{0}, jsc.dtype, {jsc.blob, jsc.dense, output, input}, jsc.runtime},
        {{1}, nvq.dtype, {nvq.blob, nvq.dense, output, input}, nvq.runtime},
    }));
    const auto combined = mfq::metal::MlxMfeWeight::concatenate_projections({gate, up})
        .materialize_packed_projections();
    const char* prior = std::getenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD");
    const std::optional<std::string> saved = prior == nullptr
        ? std::nullopt : std::optional<std::string>(prior);
    const char* prior_split = std::getenv("MFQ_METAL_MFE_SPLIT_SWIGLU");
    const std::optional<std::string> saved_split = prior_split == nullptr
        ? std::nullopt : std::optional<std::string>(prior_split);
    std::vector<std::int32_t> expert_ids(tokens * 5);
    const int experts[]{2, 0, 1, -1, 3};
    for (int token = 0; token < tokens; ++token)
        for (int route = 0; route < 5; ++route)
            expert_ids[token * 5 + route] = experts[(token + route) % 5];
    const array ids(expert_ids.begin(), Shape{tokens, 5}, int32);
    for (const auto dtype : {float16, float32}) {
        for (const bool routed : {false, true}) {
            const Shape shape = routed ? Shape{tokens, 5, input} : Shape{tokens, input};
            std::vector<float> values(tokens * (routed ? 5 : 1) * input);
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i] = float(int((i * 7 + 3) % 29) - 14) / 256.0f;
            }
            auto x = astype(array(values.begin(), shape), dtype);
            for (const bool swiglu : {false, true}) {
                const auto execute = [&] {
                    return evaluated_floats(swiglu
                        ? combined.routed_swiglu(x, ids, 0.7f)
                        : gate.routed_matmul(x, ids));
                };
                setenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD", "1", 1);
                setenv("MFQ_METAL_MFE_SPLIT_SWIGLU", "0", 1);
                const auto expected = execute();
                if (!swiglu) {
                    const std::vector<float>* dense_weights[]{
                        &nint.dense, &jsc.dense, &nvq.dense};
                    for (int slot = 0; slot < tokens * 5; ++slot) {
                        const auto* dense = expert_ids[slot] >= 0
                            && expert_ids[slot] < 3
                            ? dense_weights[expert_ids[slot]] : nullptr;
                        for (int row = 0; row < output; ++row) {
                            float reference = 0.0f;
                            if (dense != nullptr) {
                                for (int column = 0; column < input; ++column) {
                                    reference += values[(routed ? slot : slot / 5) * input + column]
                                        * (*dense)[row * input + column];
                                }
                            }
                            if (dtype == float16)
                                reference = float(static_cast<float16_t>(reference));
                            require_close(expected[slot * output + row], reference, 2e-3f);
                        }
                    }
                }
                setenv("MFQ_METAL_MFE_SPLIT_SWIGLU", "1", 1);
                for (const char* rows : {"2", "4"}) {
                    setenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD", rows, 1);
                    const auto actual = execute();
                    require(actual.size() == expected.size(), "mixed decode packing shape mismatch");
                    for (std::size_t i = 0; i < actual.size(); ++i) {
                        require_close(actual[i], expected[i], 2e-3f);
                    }
                }
            }
        }
    }
    if (saved) setenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD", saved->c_str(), 1);
    else unsetenv("MFQ_METAL_MFE_DECODE_ROWS_PER_SIMD");
    if (saved_split) {
        setenv("MFQ_METAL_MFE_SPLIT_SWIGLU", saved_split->c_str(), 1);
    } else {
        unsetenv("MFQ_METAL_MFE_SPLIT_SWIGLU");
    }
}

void test_vq_anchor_conversion() {
    constexpr int rows = 0x7c00;
    auto fixture = make_nvq1_l(rows, 8);
    for (std::uint16_t bits = 0; bits < rows; ++bits)
        std::memcpy(fixture.blob.data() + 40 + bits * sizeof(bits), &bits, sizeof(bits));
    const auto parsed = mfq::metal::detail::parse_vq_blob(fixture.dtype, fixture.blob, {});
    for (std::uint16_t bits = 0; bits < rows; ++bits) {
        const int exponent = bits >> 10;
        const int mantissa = bits & 0x3ff;
        const float expected = exponent == 0 ? std::ldexp(float(mantissa), -24)
            : std::ldexp(1.0f + float(mantissa) / 1024.0f, exponent - 15);
        require(std::bit_cast<std::uint32_t>(parsed.anchors[bits]) == std::bit_cast<std::uint32_t>(expected),
            "VQ FP16 anchor conversion changed a finite value");
    }
    const std::uint16_t negative_zero = 0x8000;
    std::memcpy(fixture.blob.data() + 40, &negative_zero, sizeof(negative_zero));
    require(std::signbit(mfq::metal::detail::parse_vq_blob(fixture.dtype, fixture.blob, {}).anchors[0]),
        "VQ FP16 anchor conversion lost negative zero");
    for (const std::uint16_t invalid : {0x7c00, 0xfc00, 0x7e00, 0xfe00, 0x8001}) {
        std::memcpy(fixture.blob.data() + 40, &invalid, sizeof(invalid));
        bool rejected = false;
        try { (void)mfq::metal::detail::parse_vq_blob(fixture.dtype, fixture.blob, {}); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "VQ anchor validation accepted an invalid value");
    }
}

void test_streamed_nint_selector_boundaries() {
    constexpr int experts = 19;
    constexpr int output = 5;
    constexpr int input = 37;
    for (const int group_size : {23, 24, 32, 128}) {
        const auto tensor = make_nint_v2_tensor(
            experts * output, input, 53, group_size, true);
        std::vector<std::int32_t> pool_ids(experts);
        for (int local = 0; local < experts; ++local) pool_ids[local] = experts - local - 1;
        const auto blob = make_raw_nim2(experts, output, input,
            {{pool_ids, "NINTv2", tensor, {}}});
        auto nonfinite = tensor;
        const auto scale_offset = 42 + (experts - 1 - 7) * output * sizeof(std::uint16_t);
        nonfinite.blob[scale_offset] = 0;
        nonfinite.blob[scale_offset + 1] = 0x7e;
        const auto malformed = make_raw_nim2(experts, output, input,
            {{pool_ids, "NINTv2", nonfinite, {}}});
        const TemporaryMfq file({{"valid", "MFE", blob}, {"nonfinite", "MFE", malformed}, {"broken", "MFE", blob}});
        const mfq::metal::MfqContainer model(file.path());
        mfq::metal::MlxMfeOffloadCache cache(model, 1);
        (void)cache.projection_info("valid");
        (void)cache.projection_info("nonfinite");
        (void)cache.projection_info("broken");
        std::filesystem::resize_file(file.path(), model.record("broken").offset);
        mfq::metal::MlxMfeOffloadCache bulk(model, 6 * blob.size());
        for (const auto& bad : {std::string("broken"), std::string("nonfinite")}) {
            for (const std::vector<std::string> names : {
                    std::vector<std::string>{bad, "valid"}, {"valid", bad},
                    {"valid", bad, "valid"}}) {
                bool failed = false;
                try { (void)bulk.layer_mfe(names); }
                catch (const std::exception&) { failed = true; }
                require(failed && bulk.cached_expert_count() == 0 && bulk.backing_payload_bytes() == 0,
                    "failed bulk MFE staging committed partial state");
            }
        }
        const std::array valid_names{std::string("valid")};
        require(bulk.layer_mfe(valid_names).has_value(), "bulk MFE workers did not recover after an error");
        for (const auto& bad : {std::string("broken"), std::string("nonfinite")}) {
            for (const std::vector<std::string> names : {std::vector<std::string>{bad, "valid"}, {"valid", bad}}) {
                cache.clear();
                bool failed = false;
                try { (void)cache.grouped_mfe(names, pool_ids); }
                catch (const std::exception&) { failed = true; }
                require(failed && cache.cached_expert_count() == 0,
                    "failed combined MFE preparation committed partial cache pages");
                require(cache.grouped_mfe("valid", pool_ids).experts() == experts,
                    "combined MFE preparation did not recover after worker failure");
            }
        }
        cache.clear();
        bool rejected = false;
        try { (void)cache.grouped_mfe("broken", pool_ids); }
        catch (const std::exception&) { rejected = true; }
        require(rejected && cache.cached_expert_count() == 0,
            "failed parallel expert reads committed partial cache pages");
        rejected = false;
        try { (void)cache.grouped_mfe("nonfinite", pool_ids); }
        catch (const std::exception&) { rejected = true; }
        require(rejected && cache.cached_expert_count() == 0,
            "failed expert parsing committed partial cache pages");
        const auto full = mfq::metal::MlxMfeWeight::from_blob(blob);
        std::vector<float> source(input);
        for (int column = 0; column < input; ++column) source[column] = float(column - 18) / 256;
        const auto x = mlx::core::astype(
            mlx::core::array(source.begin(), mlx::core::Shape{1, input}), mlx::core::float16);
        for (const std::vector<std::int32_t> active : {
                 std::vector<std::int32_t>{0, 18, 7, 11},
                 std::vector<std::int32_t>{17, 2, 9, 1},
                 pool_ids}) {
            auto requested = active;
            requested.push_back(active.front());
            const auto paged = cache.grouped_mfe("valid", requested);
            require(paged.experts() == static_cast<int>(active.size()),
                "streamed NINT duplicate expert was not deduplicated");
            std::vector<std::int32_t> local(active.size());
            std::iota(local.begin(), local.end(), 0);
            const auto expected = evaluated_floats(full.routed_matmul(x,
                mlx::core::array(active.begin(), mlx::core::Shape{1, static_cast<int>(active.size())})));
            const auto actual = evaluated_floats(paged.routed_matmul(x,
                mlx::core::array(local.begin(), mlx::core::Shape{1, static_cast<int>(local.size())})));
            require(actual.size() == expected.size(), "streamed NINT selector output shape mismatch");
            for (std::size_t index = 0; index < actual.size(); ++index) {
                require(std::isfinite(actual[index]) && std::fabs(actual[index] - expected[index]) <= 1e-6f,
                    "streamed NINT expert prefix rank or unaligned slice mismatch");
            }
            require(cache.resident_packed_bytes() <= cache.cache_limit_bytes(),
                "streamed NINT selector test exceeded cache budget");
        }
    }
}

void test_streamed_cached_projection() {
    using namespace mlx::core;
    using namespace mfq::metal;
    constexpr int experts = 5, output = 7, input = 37;
    const auto tensor = make_nint_v2_tensor(experts * output, input, 53, 23, true);
    const auto blob = nint_expert_blob(tensor, experts, output);
    const TemporaryMfq file({{"gate", "NINT", blob}, {"up", "NINT", blob}});
    const MfqContainer model(file.path());
    const auto full = MlxMfeWeight::from_nint_blob(blob);
    const std::array<std::string, 2> names{"gate", "up"};
    MlxMfeOffloadCache cache(model, std::size_t{8} << 20, experts);
    require(!cache.cached_projection("gate"), "cold cache exposed a device projection");
    for (int step = 0; step < 9; ++step) (void)cache.grouped_mfe(names, {0, 2, 4});
    std::array<std::shared_ptr<const MlxMfeCachedProjection>, 2> views{
        cache.cached_projection("gate"), cache.cached_projection("up")};
    require(views[0] && views[1], "hot projections did not expose device pages");
    require(cache.cached_projection("gate") == views[0], "hot device projection was rebuilt");
    require(views[0]->host_expert_map[1] == -1 && views[0]->host_expert_map[3] == -1,
        "device projection marked an absent expert as resident");
    for (const int tokens : {1, 2, 3, 6}) {
        auto x = full_like(ones({tokens, input}, float16), .125f);
        std::vector<std::int32_t> ids(tokens * 2);
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = std::array{0, 2, 4}[i % 3];
        const auto routes = array(ids.begin(), Shape{tokens, 2}, int32);
        const auto expected = evaluated_floats(full.routed_matmul(x, routes));
        const auto before = cache.reuse_stats();
        for (const auto& view : views) {
            require(evaluated_floats(view->weight.routed_matmul(x, take(view->expert_map, routes))) == expected,
                "cached device projection changed output bits");
        }
        require(cache.resolve_cached_projections(views, ids), "resident routes were reported as a miss");
        require(cache.reuse_stats().cached_route_hits == before.cached_route_hits + 1,
            "device route hit accounting is inconsistent");
    }
    const auto before = cache.reuse_stats();
    const std::array<std::int32_t, 2> missing{1, 0};
    require(!cache.resolve_cached_projections(views, missing), "missing device expert was accepted");
    require(cache.reuse_stats().expert_hits == before.expert_hits,
        "failed device trial promoted partial hits before replay");
    require(!cache.cached_projection("gate"), "device miss did not reset confidence");
    const auto x = full_like(ones({1, input}, float16), .125f);
    const auto ids = array({0, 4}, Shape{1, 2}, int32);
    const auto expected = evaluated_floats(full.routed_matmul(x, ids));
    auto deferred = views[0]->weight.routed_matmul(x, take(views[0]->expert_map, ids));
    views = {};
    cache.set_cache_limit(0);
    require(cache.cached_expert_count() == 0 && cache.resident_packed_bytes() == 0,
        "device projection prevented budget eviction");
    require(evaluated_floats(deferred) == expected, "budget eviction invalidated a lazy device projection");
    auto retained = [&] {
        MlxMfeOffloadCache temporary(model, std::size_t{8} << 20, experts);
        for (int step = 0; step < 9; ++step) (void)temporary.grouped_mfe("gate", {0, 2, 4});
        auto view = temporary.cached_projection("gate");
        require(bool(view), "temporary hot projection was unavailable");
        return view->weight.routed_matmul(x, take(view->expert_map, ids));
    }();
    require(evaluated_floats(retained) == expected, "cache destruction invalidated prepared GPU resources");
}

void test_streamed_nint_expert_tensor() {
    constexpr int experts = 19, output = 5, input = 37;
    {
        mfq::metal::MlxKernelPreparation preparation(128, 128);
        std::vector<std::pair<mfq::metal::MlxGroupedMmqPlan, int>> plans;
        for (const int routes : {128, 65536}) {
            std::size_t kernels = 0;
            const std::vector<int> counts = routes == 128
                ? std::vector<int>{1, 19, 21, 27, 127, 128, 129, 512, 1024}
                : std::vector<int>{128, 129, 255, 512};
            for (const int count : counts) {
                const auto tensor = make_nint_v2_tensor(count * output, input, 53, 23, true, 4);
                const auto weight = mfq::metal::MlxMfeWeight::from_nint_blob(nint_expert_blob(tensor, count, output));
                preparation.collect([&] {
                    const auto ids = mlx::core::zeros({routes / 4, 4}, mlx::core::int32);
                    const auto order = mlx::core::arange(routes, mlx::core::int32);
                    auto plan = weight.build_grouped_mmq_plan(ids, order, 32);
                    preparation.add(plan.block_meta);
                    plans.emplace_back(std::move(plan), routes / 32);
                });
                require(kernels == 0 || preparation.size() == kernels,
                    "grouped block planning compiled another kernel for a different active expert count: experts="
                        + std::to_string(count) + " routes=" + std::to_string(routes)
                        + " before=" + std::to_string(kernels) + " after=" + std::to_string(preparation.size()));
                kernels = preparation.size();
            }
        }
        require(preparation.size() == 2, "grouped block preparation did not retain exactly its two scheduling policies");
        preparation.finish();
        for (const auto& [plan, blocks] : plans) {
            mlx::core::eval({plan.block_meta, plan.block_count});
            require(plan.block_count.item<std::int32_t>() == blocks,
                "runtime expert count changed the grouped block count");
            auto metadata = plan.block_meta;
            metadata.eval();
            const auto* rows = metadata.data<std::int32_t>();
            for (int block = 0; block < blocks; ++block)
                require(rows[block * 3] == block * 32 && rows[block * 3 + 1] == 0 && rows[block * 3 + 2] == 32,
                    "runtime expert count changed grouped row coverage or expert IDs");
        }
    }
    bool indirect_supported = false;
    if (__builtin_available(macOS 15, *)) indirect_supported =
        mlx::core::metal::device(mlx::core::Device::gpu).mtl_device()->supportsFamily(MTL::GPUFamilyMetal3);
    for (const int group_size : {23, 48, 64}) for (const int q : {0, 4, 5, 6, 8}) {
        const auto tensor = make_nint_v2_tensor(experts * output, input, 53, group_size, true, q);
        const auto blob = nint_expert_blob(tensor, experts, output);
        auto invalid = blob;
        const std::int64_t wrong_output = output + 1;
        std::memcpy(invalid.data() + 26, &wrong_output, sizeof(wrong_output));
        const TemporaryMfq file({{"valid", "NINT", blob}, {"invalid", "NINT", invalid},
            {"truncated", "NINT", {blob.begin(), blob.begin() + 49}}});
        const mfq::metal::MfqContainer model(file.path());
        mfq::metal::MlxMfeOffloadCache cache(model, 1, experts);
        for (const auto* name : {"invalid", "truncated"}) {
            bool failed = false;
            try { (void)cache.can_group_mfe(name); }
            catch (const std::exception&) { failed = true; }
            require(failed && cache.cached_expert_count() == 0,
                "invalid NINT expert geometry entered the SSD cache");
        }
        require(cache.can_group_mfe("valid"), "3D NINT expert bank was not pageable");
        const auto full = mfq::metal::MlxMfeWeight::from_nint_blob(blob);
        for (const int tokens : {1, 2, 6, 32}) {
            const std::vector<std::int32_t> active{18, 0, 7, 11};
            const auto paged = cache.grouped_mfe("valid", {18, 0, 7, 11, 18});
            std::vector<float> source(tokens * input);
            for (std::size_t i = 0; i < source.size(); ++i) source[i] = float(int(i % 37) - 18) / 256;
            auto x = mlx::core::astype(mlx::core::array(source.begin(), mlx::core::Shape{tokens, input}), mlx::core::float16);
            std::vector<std::int32_t> global(tokens * active.size()), local(global.size());
            for (int token = 0; token < tokens; ++token) for (int route = 0; route < int(active.size()); ++route) {
                global[token * active.size() + route] = active[route];
                local[token * active.size() + route] = route;
            }
            const auto expected = evaluated_floats(full.routed_matmul(x,
                mlx::core::array(global.begin(), mlx::core::Shape{tokens, int(active.size())})));
            const auto indirect_before = mfq::metal::MlxMfeWeight::indirect_dispatch_count();
            const auto copies_before = mfq::metal::MlxMfeWeight::packed_copy_count();
            const auto actual = evaluated_floats(paged.routed_matmul(x,
                mlx::core::array(local.begin(), mlx::core::Shape{tokens, int(active.size())})));
            require(actual == expected, "SSD 3D NINT changed routed projection values");
            require(!indirect_supported || mfq::metal::MlxMfeWeight::packed_copy_count() == copies_before,
                "SSD NINT unnecessarily copied expert pages for a larger token batch");
            require(mfq::metal::MlxMfeWeight::indirect_dispatch_count() - indirect_before
                    == std::size_t(indirect_supported && tokens <= 24),
                "SSD NINT did not use the expected indirect or large-M fallback path");
            require(paged.experts() == int(active.size()) && cache.resident_packed_bytes() <= 1,
                "SSD 3D NINT duplicate routing or cache budget mismatch");
        }
        const std::array active{18, 0, 7, 11};
        auto x = mlx::core::full({2, input}, 0.125f, mlx::core::float16);
        const auto global = mlx::core::array(active.begin(), mlx::core::Shape{2, 2});
        const auto local = mlx::core::array({0, 1, 2, 3}, mlx::core::Shape{2, 2});
        const auto expected = evaluated_floats(full.routed_matmul(x, global));
        auto deferred = cache.grouped_mfe("valid", {18, 0, 7, 11}).routed_matmul(x, local);
        cache.clear();
        require(evaluated_floats(deferred) == expected,
            "indirect GPU inputs lost expert pages after weight destruction and cache clear");
        mfq::metal::MlxMfeOffloadCache bulk(model, 4 * blob.size(), experts);
        const std::array names{std::string("valid")};
        require(bulk.layer_mfe(names).has_value(), "3D NINT full-layer prefetch failed");
    }
    const auto tensor = make_nint_v2_tensor(33 * 257, 1024, 17, 48, true, 8);
    const auto blob = nint_expert_blob(tensor, 33, 257);
    require(blob.size() > (std::size_t{8} << 20), "NINT fixture misses the wire-buffer path");
    const TemporaryMfq file({{"large", "NINT", blob}});
    const mfq::metal::MfqContainer model(file.path());
    mfq::metal::MlxMfeOffloadCache bulk(model, 4 * blob.size(), 33);
    const std::array names{std::string("large")};
    auto layer = bulk.layer_mfe(names);
    require(layer.has_value(), "large NINT full-layer prefetch failed");
    bulk.clear();
    const auto full = mfq::metal::MlxMfeWeight::from_nint_blob(blob);
    auto x = mlx::core::astype(mlx::core::ones({3, 1024}), mlx::core::float16);
    const auto ids = mlx::core::array({32, 0, 16, 7, 32, 0}, mlx::core::Shape{3, 2});
    require(evaluated_floats((*layer)[0].routed_matmul(x, ids)) == evaluated_floats(full.routed_matmul(x, ids)),
        "NINT full-layer wire storage changed values or lost ownership");
}

void test_mfe_wire_buffer_materialization() {
    constexpr int experts = 65;
    constexpr int output = 257;
    constexpr int input = 1024;
    const std::array columns{0, 511, 1023};
    const auto first = make_mxfp4(32 * output, input, 3);
    const auto second = make_mxfp4(33 * output, input, 11);
    std::vector<std::int32_t> first_ids(32), second_ids(33);
    for (int local = 0; local < 32; ++local) first_ids[local] = 31 - local;
    for (int local = 0; local < 33; ++local) second_ids[local] = 64 - local;
    const auto blob = make_raw_nim2(experts, output, input, {
        {first_ids, "MXFP4", {first.blob, first.dense, 32 * output, input}, {}},
        {second_ids, "MXFP4", {second.blob, second.dense, 33 * output, input}, {}},
    });
    require(std::size_t(experts) * output * input / 2 > (std::size_t{8} << 20),
        "MFE fixture does not cover the GPU stream materialization path");
    const TemporaryMfq file({{"large", "MFE", blob}});
    const mfq::metal::MfqContainer model(file.path());
    mfq::metal::MlxMfeOffloadCache cache(model, 2 * blob.size());
    const std::array names{std::string("large")};
    auto layer = cache.layer_mfe(names);
    require(layer && layer->size() == 1, "large MFE layer was not loaded");
    cache.clear();
    std::vector<float> source(columns.size() * input, 0.0f);
    std::vector<std::int32_t> ids(columns.size() * experts);
    for (std::size_t token = 0; token < columns.size(); ++token) {
        source[token * input + columns[token]] = 1.0f;
        std::iota(ids.begin() + token * experts, ids.begin() + (token + 1) * experts, 0);
    }
    const auto actual = evaluated_floats((*layer)[0].routed_matmul(
        mlx::core::array(source.begin(), mlx::core::Shape{3, input}),
        mlx::core::array(ids.begin(), mlx::core::Shape{3, experts})));
    require(actual.size() == columns.size() * experts * output,
        "GPU MFE materialization output shape mismatch");
    for (std::size_t token = 0; token < columns.size(); ++token) {
        for (int expert = 0; expert < experts; ++expert) {
            const auto& fixture = expert < 32 ? first : second;
            const int local = expert < 32 ? 31 - expert : 64 - expert;
            for (int row = 0; row < output; ++row) {
                const auto expected = fixture.dense[
                    std::size_t(local * output + row) * input + columns[token]];
                require(actual[(token * experts + expert) * output + row] == expected,
                    "GPU MFE copy boundary or source ownership mismatch");
            }
        }
    }
}

void test_streamed_vq_wire_views() {
    constexpr int experts = 3;
    for (const auto [output, input] : {std::pair{33, 40}, std::pair{33, 96}, std::pair{640, 2560}})
        for (int variant = 0; variant < 6; ++variant) {
        const auto fixture = variant == 0 ? make_jsc_nvq(experts * output, input)
            : variant == 1 ? make_jsc_nvq(experts * output, input, "NVQ2J-L", 4, 8, 10, false, 4)
            : variant == 2 ? make_jsc_nvq(experts * output, input, "NVQ2J-XL", 5, 8, 12, true, 4)
            : variant == 3 ? make_jsc_nvq(experts * output, input, "NVQ3J", 2, 4, 8)
            : variant == 4 ? make_jsc_nvq(experts * output, input, "NVQ3J-512", 3, 4, 9, false, 4)
            : make_jsc_nvq(experts * output, input, "NVQ3J-L", 6, 4, 10, false, 4);
        const auto blob = make_raw_nim2(experts, output, input, {
            {{2, 0, 1}, fixture.dtype, {fixture.blob, fixture.dense, experts * output, input}, {}},
        });
        const TemporaryMfq file({{"wire", "MFE", blob}});
        const mfq::metal::MfqContainer model(file.path());
        for (const auto active : {std::vector<std::int32_t>{1, 2}, std::vector<std::int32_t>{1, 2, 0}}) {
            mfq::metal::MlxMfeOffloadCache cache(model, 0);
            const auto copied = cache.grouped_mfe("wire", active);
            auto borrowed = [&] {
                mfq::metal::MlxMfeOffloadCache transient(model, 0);
                auto result = transient.grouped_mfe("wire", active, {}, true);
                transient.clear();
                require(transient.cached_expert_count() == 0 && transient.resident_packed_bytes() == 0,
                    "wire-backed prefill retained expert cache storage");
                return result;
            }();
            for (const int tokens : {1, 2, 3, 6, 33}) {
                std::vector<float> values(tokens * input);
                for (std::size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 29) - 14) / 256.0f;
                std::vector<std::int32_t> routes(tokens * 2);
                for (std::size_t i = 0; i < routes.size(); ++i) routes[i] = i % active.size();
                const auto x = mlx::core::astype(mlx::core::array(values.begin(), mlx::core::Shape{tokens, input}),
                    mlx::core::float16);
                const auto ids = mlx::core::array(routes.begin(), mlx::core::Shape{tokens, 2});
                require(evaluated_floats(borrowed.routed_matmul(x, ids)) ==
                    evaluated_floats(copied.routed_matmul(x, ids)),
                    "wire-backed VQ stream changed packed computation or lost ownership");
                if (tokens == 1 || tokens == 33) {
                    auto pending = [&] {
                        mfq::metal::MlxMfeOffloadCache transient(model, 0);
                        auto weight = transient.grouped_mfe("wire", active, {}, true);
                        auto output = weight.routed_matmul(x, ids);
                        mlx::core::async_eval({output});
                        transient.clear();
                        return output;
                    }();
                    mlx::core::clear_cache();
                    require(evaluated_floats(pending) == evaluated_floats(copied.routed_matmul(x, ids)),
                        "async VQ projection lost its released weight owners");
                }
            }
        }
    }
}

void test_streamed_nvq1_residency() {
    constexpr int experts = 19;
    constexpr int output = 3;
    std::vector<std::int32_t> pool_ids(experts);
    std::iota(pool_ids.begin(), pool_ids.end(), 0);
    std::rotate(pool_ids.begin(), pool_ids.end() - 1, pool_ids.end());
    // Odd expert row counts exercise unaligned state/index/delta boundaries.
    // Cover both embedded tables and the implicit NVQ1-L table, including K tails.
    for (int variant = 0; variant < 4; ++variant) {
        const bool short_profile = variant == 0;
        const bool custom_table = variant == 2;
        const int input = short_profile ? 40 : 37;
        const int state_bits = short_profile ? 4 : variant == 3 ? 8 : 3;
        const int index_bits = short_profile ? 9 : 11;
        const int groups = (input + 23) / 24;
        const int vectors = (input + 7) / 8;
        const std::string dtype = short_profile ? "NVQ1-S" : "NVQ1-L";
        std::vector<std::uint8_t> payload;
        append_vq_matrix_header(payload, short_profile ? "NQ1S" : "NQ1L",
            custom_table ? 2 : 1, state_bits, 24, experts * output, input);
        if (short_profile) {
            // Distinct banks make delta slicing observable independently of scale.
            for (int bank = 0; bank < 2; ++bank) {
                for (int entry = 0; entry < 512; ++entry) {
                    append<std::uint16_t>(payload,
                        mfq::nvq_codebooks::kNvq1LCodebookPacked[
                            bank == 0 ? entry * 4 : (511 - entry) * 4]);
                }
            }
        } else if (custom_table) {
            for (int entry = 0; entry < 2048; ++entry) {
                append<std::uint16_t>(payload,
                    mfq::nvq_codebooks::kNvq1LCodebookPacked[2047 - entry]);
            }
        }
        std::vector<std::uint16_t> states, indices, deltas;
        for (int row = 0; row < experts * output; ++row) {
            append<std::uint16_t>(payload, static_cast<std::uint16_t>(0x3400 + (row % 8) * 128));
            for (int group = 0; group < groups; ++group) {
                states.push_back((row * 3 + group + 1) % (1 << state_bits));
                deltas.push_back((row + group) % 2);
            }
            for (int vector = 0; vector < vectors; ++vector) {
                indices.push_back((row * 71 + vector * 37) % (1 << index_bits));
            }
        }
        append_bytes(payload, pack_vq_values(states, state_bits));
        append_bytes(payload, pack_vq_values(indices, index_bits));
        append_bytes(payload, pack_vq_values(deltas, 1));
        const auto wrap = [&](const std::vector<std::uint8_t>& bytes) {
            return make_raw_nim2(experts, output, input,
                {{pool_ids, dtype, {bytes, {}, experts * output, input}, {}}});
        };
        auto truncated = payload;
        truncated.pop_back();
        auto trailing = payload;
        trailing.push_back(0);
        auto wrong_shape = payload;
        wrong_shape[20] = 1;
        auto wrong_profile = payload;
        wrong_profile[4] = 9;
        const auto blob = wrap(payload);
        const TemporaryMfq file({
            {"valid", "MFE", blob},
            {"truncated", "MFE", wrap(truncated)},
            {"trailing", "MFE", wrap(trailing)},
            {"shape", "MFE", wrap(wrong_shape)},
            {"profile", "MFE", wrap(wrong_profile)},
        });
        const mfq::metal::MfqContainer model(file.path());
        mfq::metal::MlxMfeOffloadCache cache(model, 1);
        require(cache.can_group_mfe("valid"), "NVQ1 MFE was not pageable");
        for (const auto* name : {"truncated", "trailing", "shape", "profile"}) {
            bool rejected = false;
            try { (void)cache.can_group_mfe(name); }
            catch (const std::exception&) { rejected = true; }
            require(rejected, std::string("malformed NVQ1 MFE was accepted: ") + name);
        }
        const auto full = mfq::metal::MlxMfeWeight::from_blob(blob);
        const auto check = [&](const mfq::metal::MlxMfeWeight& paged,
                               const std::vector<std::int32_t>& active) {
            for (const int tokens : {1, 5, 17}) {
                std::vector<float> source(static_cast<std::size_t>(tokens) * input);
                for (std::size_t i = 0; i < source.size(); ++i) {
                    source[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 128.0f;
                }
                std::vector<std::int32_t> global_ids(tokens * 2), local_ids(tokens * 2);
                for (std::size_t i = 0; i < local_ids.size(); ++i) {
                    local_ids[i] = i % active.size();
                    global_ids[i] = active[local_ids[i]];
                }
                const auto x = mlx::core::astype(mlx::core::array(
                    source.begin(), mlx::core::Shape{tokens, input}), mlx::core::float16);
                const auto expected = evaluated_floats(full.routed_matmul(x,
                    mlx::core::array(global_ids.begin(), mlx::core::Shape{tokens, 2})));
                const auto actual = evaluated_floats(paged.routed_matmul(x,
                    mlx::core::array(local_ids.begin(), mlx::core::Shape{tokens, 2})));
                require(actual.size() == expected.size(), "paged NVQ1 shape mismatch");
                for (std::size_t i = 0; i < actual.size(); ++i) {
                    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
                        std::fabs(actual[i] - expected[i]) > 1e-6f) {
                        throw std::runtime_error("paged NVQ1 mismatch: variant=" +
                            std::to_string(variant) + " tokens=" + std::to_string(tokens) +
                            " index=" + std::to_string(i) + " actual=" +
                            std::to_string(actual[i]) + " expected=" + std::to_string(expected[i]));
                    }
                    // Independent scalar oracle also catches shared packed-kernel
                    // errors (in particular reads beyond a partial final vector).
                    const int route = i / output;
                    const int local_expert = (global_ids[route] + 1) % experts;
                    const int row = local_expert * output + i % output;
                    float reference = 0.0f;
                    for (int column = 0; column < input; ++column) {
                        const int group = column / 24;
                        const int delta_bit = deltas[row * groups + group];
                        const int index = indices[row * vectors + column / 8];
                        const int table_index = short_profile
                            ? (delta_bit ? (511 - index) * 4 : index * 4)
                            : custom_table ? 2047 - index : index;
                        const auto word = mfq::nvq_codebooks::kNvq1LCodebookPacked[table_index];
                        const float code = static_cast<int>((word >> (2 * (column % 8))) & 3u) - 1;
                        const float delta = (short_profile ? 0.15625f : 0.125f) *
                            (delta_bit ? -1.0f : 1.0f);
                        const float weight = (0.25f + (row % 8) * 0.03125f) *
                            states[row * groups + group] * (code + delta);
                        reference += source[(route / 2) * input + column] * weight;
                    }
                    require_close(actual[i], reference, 2e-2f);
                }
            }
        };
        const auto retained = cache.grouped_mfe("valid", {2, 0});
        check(retained, {2, 0});
        check(cache.grouped_mfe("valid", pool_ids), pool_ids);
        require(cache.cached_expert_count() == 0, "NVQ1 exceeded its one-byte persistent budget");
        check(cache.grouped_mfe("valid", {1}), {1});
        require(cache.cached_expert_count() == 0, "NVQ1 retained pages above its persistent budget");
        cache.clear();
        require(cache.resident_packed_bytes() == 0, "NVQ1 cache clear retained bytes");
        check(retained, {2, 0});
        check(cache.grouped_mfe("valid", {0, 2, 1}), {0, 2, 1});
        check(cache.grouped_mfe("valid", {2, 0}, {}, true), {2, 0});
        check(cache.grouped_mfe("valid", pool_ids, {}, true), pool_ids);
    }
}

void test_compact_nint_mhc() {
    using namespace mlx::core;
    using mfq::metal::MlxLinear;
    constexpr int hidden = 2560, streams = 4, width = hidden * streams;
    const MlxLinear down(mfq::metal::MlxNintWeight::from_blob(
        make_nint_v2_tensor(320, width, 17, 64, true).blob));
    const MlxLinear up(mfq::metal::MlxNintWeight::from_blob(
        make_nint_v2_tensor(width, 320, 31, 32, true).blob));
    const std::optional<MlxLinear> injection{MlxLinear(mfq::metal::MlxNintWeight::from_blob(
        make_nint_v2_tensor(streams, width, 23, 64, true).blob))};
    const auto dense_down = down.nint_weight_ref()->dequantize();
    const auto dense_up = up.nint_weight_ref()->dequantize();
    const std::optional<array> dense_injection = injection->nint_weight_ref()->dequantize();
    const auto norm = ones({width}, float16);
    for (const auto dtype : {float16, bfloat16, float32}) {
        for (const int rows : {1, 2, 3, 4, 5, 6, 7}) {
            std::vector<float> values(rows * width);
            for (std::size_t index = 0; index < values.size(); ++index)
                values[index] = float(int((index * 37) % 257) - 128) / 257.0f;
            const auto x = astype(array(values.begin(), Shape{1, rows, width}), dtype);
            const auto actual = mfq::metal::qwen4_gated_residual_pre(
                x, norm, down, up, injection, hidden, streams);
            const auto expected = mfq::metal::qwen4_gated_residual_pre(
                x, norm, dense_down, dense_up, dense_injection, hidden, streams);
            const auto compare = [](array a, array b) {
                const auto actual_values = evaluated_floats(a);
                const auto expected_values = evaluated_floats(b);
                require(actual_values.size() == expected_values.size(), "packed MHC output shape changed");
                for (std::size_t index = 0; index < actual_values.size(); ++index)
                    require(std::isfinite(actual_values[index]) &&
                        std::fabs(actual_values[index] - expected_values[index])
                            <= 5e-3f * (1.0f + std::fabs(expected_values[index])),
                        "compact NINT mHC differs from dequantized reference");
            };
            if (rows <= 6 && std::getenv("MFQ_METAL_QWEN_GATED_HC_FAST") == nullptr)
                require(std::string(actual.branch.primitive().name()) == "PackedHcPrimitive",
                    "production-shape compact MHC fixture missed packed decode");
            compare(actual.branch, expected.branch);
            compare(*actual.injection, *expected.injection);
            const auto after = mfq::metal::qwen4_gated_residual_pre_after(
                expected.branch, x, *expected.injection, norm, down, up, injection, hidden, streams);
            const auto expected_after = mfq::metal::qwen4_gated_residual_pre_after(
                expected.branch, x, *expected.injection, norm,
                dense_down, dense_up, dense_injection, hidden, streams);
            compare(after.branch, expected_after.branch);
            compare(*after.injection, *expected_after.injection);
            compare(after.residual, expected_after.residual);
            compare(mfq::metal::qwen4_gated_residual_pre(
                x, norm, down, up, std::nullopt, hidden, streams).branch,
                mfq::metal::qwen4_gated_residual_pre(
                    x, norm, dense_down, dense_up, std::nullopt, hidden, streams).branch);
        }
    }
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
    const std::vector<float> legacy_input(32, 0.125f);
    const std::vector<std::int32_t> legacy_ids{0, 1};
    const auto legacy_output = evaluated_floats(
        legacy_weight.routed_matmul(
            mlx::core::array(
                legacy_input.begin(),
                mlx::core::Shape{1, 32}),
            mlx::core::array(
                legacy_ids.begin(),
                mlx::core::Shape{1, 2})));
    for (int expert = 0; expert < 2; ++expert) {
        float expected = 0.0f;
        for (int column = 0; column < 32; ++column) {
            expected += legacy_input[column]
                * two.dense[expert * 32 + column];
        }
        require_close(
            legacy_output[expert], expected, 2e-2f);
    }

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

void test_kernel_preparation() {
#ifdef MFQ_MLX_METALLIB_DEFAULT
    mlx::core::metal::set_metallib_path(MFQ_MLX_METALLIB_DEFAULT);
#endif
    mlx::core::set_default_device(mlx::core::Device::gpu);
    const auto nint = make_moe_fixture({"NINT4", "NINT5", "NINT6", "NINTv2"}, 33, 64, 5, 24);
    const auto vq = make_vq_moe_fixture({
        make_jsc_nvq(33, 64, "NVQ3J-L", 6, 4, 10, false, 4, -1),
        make_jsc_nvq(33, 64, "NVQ3J-L", 6, 4, 10, false, 4, -1),
    });
    mfq::metal::MlxKernelPreparation preparation(2048, 2048);
    preparation.set_routes(2);
    mfq_kernel_prepare_phase(1);
    const auto first = mfq::metal::MlxMfeWeight::from_blob(nint.blob);
    const auto second = mfq::metal::MlxMfeWeight::from_blob(vq.blob);
    const auto pair = mfq::metal::MlxMfeWeight::concatenate_projections({first, first});
    std::vector<mlx::core::array> descriptions;
    for (const auto rows : preparation.row_buckets()) {
        const auto input = mlx::core::zeros({rows, 64}, mlx::core::float16);
        const auto ids = mlx::core::zeros({rows, 2}, mlx::core::int32);
        descriptions.push_back(pair.routed_swiglu(input, ids));
        preparation.add(descriptions.back());
    }
    require(!preparation.unsupported().size(), "MFE native compilation interface missing");
    const auto memory = mlx::core::get_active_memory();
    preparation.finish();
    require(mlx::core::get_active_memory() == memory, "MFE compilation allocated tensor buffers");
    for (const auto& output : descriptions)
        require(!output.is_available(), "MFE preparation executed a projection");
    mfq_kernel_prepare_phase(2);
    for (const int rows : {9, 17, 29, 128}) {
        std::vector<float> values(rows * 64);
        std::vector<std::int32_t> ids(rows * 2);
        for (std::size_t index = 0; index < values.size(); ++index)
            values[index] = float(int(index % 17) - 8) / 4096.0f;
        for (std::size_t index = 0; index < ids.size(); ++index) ids[index] = index % 2;
        const auto input = mlx::core::astype(mlx::core::array(values.begin(),
            mlx::core::Shape{rows, 64}), mlx::core::float16);
        const auto routes = mlx::core::array(ids.begin(), mlx::core::Shape{rows, 2});
        const auto a = evaluated_floats(first.routed_matmul(input, routes));
        const auto b = evaluated_floats(second.routed_matmul(input, routes));
        for (int row = 0; row < rows; ++row) {
            for (int route = 0; route < 2; ++route) {
                const int expert = ids[row * 2 + route];
                for (int neuron = 0; neuron < 33; ++neuron) {
                    float expected_nint = 0.0f, expected_vq = 0.0f;
                    for (int column = 0; column < 64; ++column) {
                        expected_nint += values[row * 64 + column] *
                            nint.dense[(expert * 33 + neuron) * 64 + column];
                        expected_vq += values[row * 64 + column] *
                            vq.expert_weights[expert].dense[neuron * 64 + column];
                    }
                    const auto index = (row * 2 + route) * 33 + neuron;
                    require_close(a[index], expected_nint, 2e-3f);
                    require_close(b[index], expected_vq, 2e-3f);
                }
            }
        }
        const auto fused = evaluated_floats(pair.routed_swiglu(input, routes));
        const auto separate = evaluated_floats(first.routed_matmul(input, routes));
        for (std::size_t index = 0; index < fused.size(); ++index) {
            const auto value = separate[index];
            require_close(fused[index], value * value / (1.0f + std::exp(-value)), 2e-3f);
        }
    }
    mfq_kernel_prepare_phase(3);
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--test-packed-shared-router") {
        try {
            mlx::core::set_default_device(mlx::core::Device::gpu);
            for (int groups : {7, 24, 28, 48})
                for (int tokens : {1, 2, 3, 4, 5, 6, 8, 24})
                    for (int bits : {4, 8}) {
                        try {
                            test_two_stage_nint_shared_decode(groups, 65, 41, false,
                                tokens, true, false, 24, false, 32, false, bits);
                        } catch (const std::exception& error) {
                            throw std::runtime_error("GS=" + std::to_string(groups) +
                                " M=" + std::to_string(tokens) + " q=" + std::to_string(bits) + ": " + error.what());
                        }
                    }
            std::cout << "MFQ packed shared router tests passed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    if (argc == 5 && std::string_view(argv[1]) == "--probe-mtp-cache") {
        try {
            using namespace mlx::core;
            set_default_device(Device::gpu);
            set_cache_limit(128ULL << 20);
            const auto read = [](const std::filesystem::path& path) {
                std::ifstream stream(path, std::ios::binary | std::ios::ate);
                require(bool(stream), "MTP probe cannot open checkpoint");
                const auto size = stream.tellg();
                require(size > 0 && size < (1ULL << 30), "MTP probe checkpoint size is invalid");
                std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
                stream.seekg(0); stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
                require(bool(stream), "MTP probe checkpoint is truncated");
                return bytes;
            };
            const auto bytes = read(argv[4]);
            const auto text = read(std::string(argv[4]) + ".tokens");
            require(bytes.size() >= 156 && text.size() >= 108 &&
                std::memcmp(bytes.data(), "MFQKVB1\0", 8) == 0 &&
                std::memcmp(text.data(), "MFQTXT1\0", 8) == 0,
                "MTP probe requires a native prefix block and token sidecar");
            std::uint32_t block_size = 0, count = 0, text_count = 0;
            std::uint64_t payload_size = 0;
            std::memcpy(&block_size, bytes.data() + 108, 4);
            std::memcpy(&count, bytes.data() + 112, 4);
            std::memcpy(&payload_size, bytes.data() + 116, 8);
            std::memcpy(&text_count, text.data() + 72, 4);
            require(count > 0 && count <= block_size && count == text_count &&
                bytes.size() == 156 + payload_size && text.size() == 108 + count * 8 &&
                std::memcmp(bytes.data() + 12, text.data() + 8, 64) == 0,
                "MTP probe block metadata mismatch");
            const auto payload_hash = mfq::cache::sha256(bytes.data() + 156, payload_size);
            const auto token_hash = mfq::cache::sha256(text.data() + 108, count * 8);
            require(std::memcmp(payload_hash.data(), bytes.data() + 124, 32) == 0 &&
                std::memcmp(token_hash.data(), text.data() + 76, 32) == 0,
                "MTP probe checkpoint checksum mismatch");
            std::vector<std::int64_t> tokens(count);
            std::memcpy(tokens.data(), text.data() + 108, count * 8);
            auto payload = std::make_shared<const std::vector<std::uint8_t>>(bytes.begin() + 156, bytes.end());
            auto state = mfq::metal::MlxPagedSessionCodec<mfq::metal::MlxQwen4TextSessionState>::decode(
                {payload}, tokens, block_size);
            const mfq::metal::MfqContainer model(argv[2]), original(argv[3], "mtp.");
            std::vector<std::pair<std::string, array>> results;
            for (const auto& name : {std::string("BF16"), std::string("FP16"), std::string("quantized")}) {
                const auto [pending, logits] = mfq::metal::qwen4_mtp_logits_from_state(
                    model, name == "quantized" ? model : original, state, name == "FP16");
                std::cout << "mtp_probe source=" << name << " context=" << count << " pending=" << pending
                    << " top1=" << argmax(logits).item<std::int32_t>() << std::endl;
                results.emplace_back(name, logits);
                clear_cache();
            }
            const auto log_softmax = [](const array& x) { return x - logsumexp(x, -1, true); };
            for (const auto [left, right] : {std::pair{0, 1}, std::pair{0, 2}, std::pair{1, 2}}) {
                const auto reference = log_softmax(results[left].second);
                const auto difference = results[right].second - results[left].second;
                auto maximum = max(abs(difference));
                auto rms = sqrt(mean(square(difference)));
                auto kl = sum(exp(reference) * (reference - log_softmax(results[right].second)));
                eval(maximum, rms, kl);
                std::cout << "mtp_probe compare=" << results[left].first << '/' << results[right].first
                    << " max_logit_diff=" << maximum.item<float>() << " rms_logit_diff=" << rms.item<float>()
                    << " kl=" << kl.item<float>() << std::endl;
            }
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
    if (argc == 2 && std::string_view(argv[1]) == "--test-ane-production") {
        try {
            auto engine = std::make_shared<mfq::metal::MlxAneMatmul>();
            for (const auto [inner, columns] : {std::pair{2560, 6144}, std::pair{6144, 2560}, std::pair{10240, 320}}) {
                std::vector<float> x(inner), w(std::size_t(inner) * columns);
                for (int i = 0; i < inner; ++i) x[i] = std::sin(float(i)) * 0.1f;
                for (std::size_t i = 0; i < w.size(); ++i) w[i] = std::cos(float(i)) * 0.01f;
                engine->prepare(1, inner, columns);
                const auto y = engine->evaluate(1, inner, columns, x, w);
                for (int column : {0, 1, 31, columns / 2, columns - 1}) {
                    float expected = 0;
                    for (int k = 0; k < inner; ++k) expected += x[k] * w[k * columns + column];
                    require(std::isfinite(y[column]) && std::abs(y[column] - expected) < 0.005f,
                        "production ANE tiled matrix differs from CPU reference");
                }
                std::cout << "ANE shape=1x" << inner << "x" << columns << " passed\n";
            }
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
    if (argc == 3 && std::string_view(argv[1]) == "--check-mtp-source") {
        try {
            const mfq::metal::MfqContainer source(argv[2], "mtp.");
            std::uint64_t bytes = 0;
            std::size_t tensors = 0;
            for (const auto& [name, record] : source.records()) {
                if (record.dtype == "BLOB") continue;
                require(name.starts_with("predictor."), "MTP source exposed backbone weights");
                require(record.dtype == "BF16" || record.dtype == "F16" || record.dtype == "F32",
                    "MTP source contains quantized weights");
                bytes += record.nbytes;
                ++tensors;
            }
            require(source.contains("predictor.embedding_norm.weight") && tensors > 0, "empty MTP source");
            std::cout << "mtp_tensors=" << tensors << " bytes=" << bytes << '\n';
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
    if (argc == 2 && std::string_view(argv[1]) == "--test-ane-autograd") {
        try {
            auto engine = std::make_shared<mfq::metal::MlxAneMatmul>();
            const auto cpu = mlx::core::default_stream(mlx::core::Device(mlx::core::Device::cpu));
            const auto x = mlx::core::array({0.2f, -0.3f, 0.4f, 0.1f, 0.7f, -0.2f}, mlx::core::Shape{2, 3});
            const auto w = mlx::core::array({0.1f, 0.2f, -0.4f, 0.3f, 0.5f, -0.1f}, mlx::core::Shape{3, 2});
            auto actual = mlx::core::value_and_grad([&](const std::vector<mlx::core::array>& args) {
                auto y = (*engine)(args[0], args[1]);
                return std::vector<mlx::core::array>{mlx::core::sum(mlx::core::square(y, cpu), cpu)};
            }, std::vector<int>{0, 1})({x, w});
            auto reference = mlx::core::value_and_grad([&](const std::vector<mlx::core::array>& args) {
                return std::vector<mlx::core::array>{mlx::core::sum(
                    mlx::core::square(mlx::core::matmul(args[0], args[1], cpu), cpu), cpu)};
            }, std::vector<int>{0, 1})({x, w});
            const auto close = [&](const mlx::core::array& a, const mlx::core::array& b) {
                return mlx::core::max(mlx::core::abs(mlx::core::subtract(a, b, cpu), cpu), cpu).item<float>() < 0.002f;
            };
            require(close(actual.first[0], reference.first[0]), "ANE forward differs");
            for (std::size_t i = 0; i < actual.second.size(); ++i)
                require(close(actual.second[i], reference.second[i]), "ANE VJP differs");
            const auto storage_x = mlx::core::reshape(mlx::core::arange(35, mlx::core::float32, cpu), {5, 7}) * 0.01f;
            const auto storage_w = mlx::core::reshape(mlx::core::arange(42, mlx::core::float32, cpu), {6, 7}) * 0.01f;
            const auto views = [&](const std::vector<mlx::core::array>& args) {
                return std::pair{mlx::core::slice(args[0], {1, 1}, {5, 5}, {2, 1}, cpu),
                    mlx::core::transpose(mlx::core::slice(args[1], {1, 1}, {4, 5}, cpu), cpu)};
            };
            auto strided = mlx::core::value_and_grad([&](const std::vector<mlx::core::array>& args) {
                auto [left, right] = views(args);
                return std::vector<mlx::core::array>{mlx::core::sum(mlx::core::square((*engine)(left, right), cpu), cpu)};
            }, std::vector<int>{0, 1})({storage_x, storage_w});
            auto strided_reference = mlx::core::value_and_grad([&](const std::vector<mlx::core::array>& args) {
                auto [left, right] = views(args);
                return std::vector<mlx::core::array>{mlx::core::sum(
                    mlx::core::square(mlx::core::matmul(left, right, cpu), cpu), cpu)};
            }, std::vector<int>{0, 1})({storage_x, storage_w});
            require(close(strided.first[0], strided_reference.first[0]), "strided ANE forward differs");
            for (std::size_t i = 0; i < strided.second.size(); ++i)
                require(close(strided.second[i], strided_reference.second[i]), "strided ANE VJP differs");
            std::cout << "ANE CPU-stream forward and backward passed\n";
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--test-folded-topk") {
            for (const int tokens : {1, 2, 6}) {
                test_two_stage_nint_shared_decode(28, 68, 256, true, tokens, true, true, 28, true, 32, false, 0, 0, false, true);
                test_two_stage_nint_shared_decode(28, 68, 256, true, tokens, true, false, 28, true, 32, false, 0, 0, false, true);
            }
            for (const int experts : {32, 64, 128, 512, 4096})
                test_two_stage_nint_shared_decode(24, 32, 24, false, 1, false, false, kGroupSize, false, experts, true);
            test_two_stage_nint_shared_decode(28, 256, 68, true, 1, true, false, kGroupSize, false, 32, true);
            test_two_stage_nint_shared_decode(28, 68, 256, true, 1, true, true, kGroupSize, false, 32, true);
            test_two_stage_nint_shared_decode(28, 260, 72, true, 2, true, false, 28, true);
            test_two_stage_nint_shared_decode(28, 264, 76, true, 2, true, false, 28, true);
            for (const int q : {4, 5, 6, 8}) {
                test_two_stage_nint_shared_decode(28, 260, 72, true, 1, true, false, 28, false, 32, true, q);
                test_two_stage_nint_shared_decode(28, 264, 76, true, 2, true, false, 28, true, 32, false, q);
            }
            std::cout << "MFQ folded top-k decode tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-block-plan") {
            test_grouped_mmq_sparse_plan_bounds();
            std::cout << "MFQ grouped block plan bounds passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-portable-prefill") {
            test_portable_mmq_row_buckets();
            test_grouped_split_nint_swiglu_prefill();
            test_grouped_split_nint_swiglu_prefill(513, 193, 65);
            test_grouped_nint_mmq_prefill(65, 24, 256, 256);
            test_grouped_nint_mmq_prefill(65, 28, 200, 256);
            test_grouped_nint_mmq_prefill(385, 24, 256, 256);
            test_grouped_nint_mmq_prefill(340, 28, 200, 256);
            test_grouped_nint_mmq_prefill(513, 24, 193, 256);
            test_grouped_nint_mmq_prefill(513, 28, 193, 256);
            test_grouped_vq_decoder_tail_prefill(200, 257);
            test_grouped_vq_decoder_tail_prefill(256, 384);
            std::cout << "MFQ portable prefill tile tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-vq-prefill") {
            test_grouped_vq_decoder_tail_prefill(640, 256);
            test_grouped_vq_decoder_tail_prefill(200, 257);
            test_grouped_vq_decoder_tail_prefill(256, 384);
            std::cout << "MFQ VQ prefill decoder tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-nax-prefill") {
            test_mixed_mfe_native_and_grouped_dispatch(256, true);
            test_multiple_nint_pools_share_cpp_dispatch(513);
            test_grouped_dense_quad_tail_prefill(66, 128);
            test_grouped_vq_decoder_tail_prefill(640, 256);
            test_grouped_vq_decoder_tail_prefill(256, 384);
            test_grouped_nint_mmq_prefill(513, 24, 256, 66);
            test_grouped_nint_mmq_prefill(513, 28, 256, 66);
            test_grouped_nint_mmq_prefill(513, 24, 192, 256);
            test_grouped_nint_mmq_prefill(513, 28, 192, 256);
            std::cout << "MFQ NAX prefill decoder tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-shared-rotation") {
            test_shared_rotated_input();
            std::cout << "MFQ shared HSG1 rotation tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-mtp-ttt") {
            test_qwen4_online_expert_offload(true, false, true);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-prepare") {
            test_kernel_preparation();
            std::cout << "MFQ pure MFE preparation tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-model-prepare") {
            test_qwen4_online_expert_offload(true, true);
            std::cout << "MFQ pure model preparation tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-compat") {
            setenv("MFQ_METAL_MFE_NATIVE_PRIMITIVE", "0", 1);
            for (int tokens : {1, 3, 7, 13, 29}) {
                test_all_families_and_projections(tokens);
            }
            test_swiglu_ffn();
            test_vq_cohorts_and_ffn();
            test_shared_rotated_input();
            test_nepq_a_routed_and_fused_swiglu();
            std::cout << "MFQ compatibility MFE variable-length tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-direct-mfe") {
            test_banked_vq_input_views();
            test_direct_mfe_projection_families();
            test_mixed_decode_row_packing(28, 1, 2560, 640);
            test_mixed_decode_row_packing(28, 6, 2560, 640);
            std::cout << "MFQ direct MFE projection family tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-route-groups") {
            test_mxfp4_repeated_routes();
            test_uniform_nint_multirow_decode();
            for (const auto [gate_gs, up_gs] : {std::pair{24, 28}, {28, 24}, {7, 24}, {48, 28}}) {
                for (const int tokens : {1, 2, 6}) {
                    test_two_stage_nint_shared_decode(gate_gs, 65, 41, false, tokens,
                        false, false, 24, false, 32, tokens == 1, 0, up_gs);
                    test_two_stage_nint_shared_decode(gate_gs, 256, 68, true, tokens,
                        false, false, 24, false, 32, tokens == 1, 0, up_gs);
                    test_two_stage_nint_shared_decode(gate_gs, 65, 41, false, tokens,
                        false, false, 24, false, 32, tokens == 1, 8, up_gs, true, true);
                    test_two_stage_nint_shared_decode(gate_gs, 256, 68, true, tokens,
                        false, false, 24, false, 32, tokens == 1, 8, up_gs, true, true);
                }
                test_two_stage_nint_shared_decode(gate_gs, 256, 68, true, 1,
                    false, false, 24, false, 32, true, 8, up_gs);
            }
            for (const int group_size : {12, 24, 28, 48})
                for (const int tokens : {2, 6})
                    test_two_stage_nint_shared_decode(24, 65, 41, false,
                        tokens, true, false, group_size, true);
            for (const int tokens : {2, 3, 5, 6, 8, 12, 17, 24}) {
                test_two_stage_nint_shared_decode(23, 65, 41, false, tokens);
                test_two_stage_nint_shared_decode(24, 65, 41, false, tokens, false, false, 23);
                test_two_stage_nint_shared_decode(28, 68, 256, false, tokens, true, false, 28);
                test_two_stage_nint_shared_decode(48, 256, 64, false, tokens, false, false, 48);
                test_two_stage_nint_shared_decode(28, 256, 68, true, tokens, true);
                test_two_stage_nint_shared_decode(28, 68, 256, true, tokens, true, true);
                test_two_stage_nint_shared_decode(48, 256, 64, false, tokens, true, true, 48);
                test_two_stage_nint_shared_decode(24, 65, 41, false, tokens, false, false, 23, true);
                test_two_stage_nint_shared_decode(28, 256, 68, true, tokens, true, false, 28, true);
                test_two_stage_nint_shared_decode(28, 68, 256, true, tokens, true, true, 28, true);
            }
            std::cout << "MFQ batched route-group tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-mxfp4-reuse") {
            test_mxfp4_repeated_routes();
            std::cout << "MFQ MXFP4 repeated-route tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-v41-vision-routing") {
            test_deepseek_v41_visual_router_precision();
            std::cout << "MFQ DeepSeek-V4.1 visual router precision tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-v41-prefix") {
            test_deepseek_v41_ced_text_cache();
            test_deepseek_v41_ced_text_cache(true);
            std::cout << "MFQ DeepSeek-V4.1 persistent prefix tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--test-streamed-mfe") {
            test_streamed_mfe_read_validation();
            test_streamed_mfe_overlapped_read();
            test_streamed_mfe_parallel_body_reads();
            test_streamed_mfe_wide_readers();
            test_deepseek_v41_ced_text_cache();
            test_deepseek_v41_ced_text_cache(true);
            test_deepseek_v41_streamed_moe();
            test_deepseek_v41_visual_router_precision();
            test_streamed_mfe_partition_preparation();
            test_concatenate_resident_mfe_experts();
            test_compact_nint_mhc();
            test_qwen4_online_expert_offload(false);
            test_qwen4_online_expert_offload(true);
            test_qwen4_online_expert_offload(true, false, false, true);
            test_qwen4_online_expert_offload(false, false, false, false, true);
            test_qwen4_online_expert_offload(false, false, false, false, true, true);
            test_streamed_mixed_mfe_residency();
            test_streamed_vq_wire_views();
            test_streamed_nvq1_residency();
            test_streamed_nint_selector_boundaries();
            test_streamed_cached_projection();
            test_streamed_nint_expert_tensor();
            test_mfe_wire_buffer_materialization();
            test_vq_anchor_conversion();
            test_materialized_mfe_projections();
            test_direct_mfe_projection_tails();
            std::cout << "MFQ streamed MFE residency tests passed\n";
            return 0;
        }
        if (
            argc == 2
            && std::string_view(argv[1])
                == "--benchmark-nepq-a"
        ) {
            benchmark_v4f_nepq_a();
            return 0;
        }
        if (
            argc == 2
            && std::string_view(argv[1])
                == "--benchmark-nint-shared"
        ) {
            benchmark_shared_nint_mfe();
            return 0;
        }
        if (argc != 1) {
            throw std::runtime_error(
                "usage: mfq-metal-moe-test "
                "[--benchmark-nepq-a|--benchmark-nint-shared|--test-streamed-mfe|--test-direct-mfe]");
        }
        test_vq_small_m_route_order();
        test_grouped_mmq_sparse_plan_bounds();
        test_generic_vq_wide_small_m();
        test_all_families_and_projections();
        test_swiglu_ffn();
        test_two_stage_nint_shared_decode();
        test_uniform_nint_multirow_decode();
        test_two_stage_nint_shared_decode(28, 65, 41);
        test_two_stage_nint_shared_decode(23, 65, 41);
        test_two_stage_nint_shared_decode(28, 256, 68, true);
        test_two_stage_nint_shared_decode(28, 68, 256, true);
        for (int tokens = 2; tokens <= 6; ++tokens) {
            test_two_stage_nint_shared_decode(23, 65, 41, false, tokens);
            test_two_stage_nint_shared_decode(28, 256, 68, true, tokens);
            test_two_stage_nint_shared_decode(28, 68, 256, true, tokens);
        }
        for (int tokens = 1; tokens <= 6; ++tokens) {
            test_two_stage_nint_shared_decode(23, 65, 41, false, tokens, true);
            test_two_stage_nint_shared_decode(28, 256, 68, true, tokens, true);
            test_two_stage_nint_shared_decode(28, 68, 256, true, tokens, true);
            test_two_stage_nint_shared_decode(23, 65, 41, false, tokens, false, true);
            test_two_stage_nint_shared_decode(23, 65, 41, false, tokens, true, true);
            test_two_stage_nint_shared_decode(28, 256, 68, true, tokens, true, true);
            test_two_stage_nint_shared_decode(28, 68, 256, true, tokens, true, true);
            test_two_stage_nint_shared_decode(48, 256, 64, false, tokens, true, true, 48);
        }
        for (int tokens : {8, 12, 24}) {
            test_two_stage_nint_shared_decode(23, 65, 41, false, tokens);
            test_two_stage_nint_shared_decode(28, 256, 68, true, tokens, true);
            test_two_stage_nint_shared_decode(28, 68, 256, true, tokens, true, true);
            test_two_stage_nint_shared_decode(48, 256, 64, false, tokens, true, true, 48);
        }
        test_mxfp4_mfe_and_projection_offsets();
        test_mxfp4_repeated_routes();
        test_mxfp4_multi_pool_native_slots();
        test_mxfp4_pair_blocks_matches_native_projections();
        test_mxfp4_sq_mfe_reuses_linear_kernel();
        test_mxfp4_sq_mixed_mfe_routes_without_sq_heterogeneous_kernel();
        test_mxfp4_smallm_nax_policy();
        test_mxfp4_decode_down_reduce();
        test_mxfp4_decode_swiglu_row_packing();
        test_large_mxfp4_arena_avoids_single_group_builder();
        try {
            test_vq_cohorts_and_ffn();
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("VQ cohorts/FFN: ") + error.what());
        }
        test_shared_rotated_input();
        test_nepq_a_routed_and_fused_swiglu();
        test_grouped_mmq_prefill();
        test_portable_mmq_row_buckets();
        try {
            test_grouped_vq_decoder_tail_prefill();
            test_grouped_vq_decoder_tail_prefill(256, 128);
            test_grouped_vq_decoder_tail_prefill(256, 384);
            test_grouped_vq_decoder_tail_prefill(320, 384);
            test_grouped_vq_decoder_tail_prefill(256, 384);
            test_grouped_vq_decoder_tail_prefill(224, 257);
            test_grouped_vq_decoder_tail_prefill(200, 257);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("grouped VQ tail: ") + error.what());
        }
        test_nint_decode_shape_profiles();
        test_grouped_nint_mmq_prefill();
        for (int group_size : {1, 5, 23, 28, 32, 48})
            test_grouped_nint_mmq_prefill(513, group_size, 65);
        test_nvq3jl_record_dispatch();
        test_nvq3jl_half_chunk(16);
        test_nvq3jl_half_chunk(17);
        test_nvq3jl_half_chunk(37);
        test_grouped_nint_mmq_prefill(1, 24, 65);
        test_grouped_nint_mmq_prefill(1, 28, 57);
        test_grouped_split_nint_swiglu_prefill();
        test_grouped_split_nint_swiglu_prefill(513, 193, 65);
        test_grouped_nint_mmq_prefill(513, 24, 193, 66);
        test_grouped_nint_mmq_prefill(513, 24, 64, 96);
        test_grouped_nint_mmq_prefill(513, 24, 224, 384);
        test_grouped_nint_mmq_prefill(513, 24, 640, 672);
        test_grouped_nint_mmq_prefill(513, 28, 701, 66);
        test_grouped_nint_mmq_prefill(513, 48, 193, 66);
        for (int group_size : {24, 28, 48}) {
            for (int input_width : {192, 224}) {
                test_grouped_nint_mmq_prefill(513, group_size, input_width, 66);
            }
        }
        test_shared_nint2_prefill();
        test_shared_nint5_group28_prefill();
        test_shared_nint8_mixed_prefill();
        test_shared_nint_mapped_and_packed_routes();
        test_grouped_dense_quad_tail_prefill();
        test_grouped_dense_quad_tail_prefill(130, 194);
        test_mixed_mfe_native_and_grouped_dispatch();
        test_mixed_mfe_native_and_grouped_dispatch(130, true);
        test_multiple_nint_pools_share_cpp_dispatch();
        test_multiple_nint_pools_share_cpp_dispatch(1);
        test_multiple_nint_pools_share_cpp_dispatch(513);
        test_grouped_mxfp4_vq_mmq_prefill();
        test_grouped_mxfp4_vq_mmq_prefill(129, 224);
        test_streamed_mfe_read_validation();
        test_streamed_mfe_overlapped_read();
        test_streamed_mfe_parallel_body_reads();
        test_streamed_mfe_wide_readers();
        test_streamed_mfe_partition_preparation();
        test_concatenate_resident_mfe_experts();
        test_compact_nint_mhc();
        test_streamed_mixed_mfe_residency();
        test_streamed_vq_wire_views();
        test_streamed_nvq1_residency();
        test_streamed_nint_selector_boundaries();
        test_streamed_cached_projection();
        test_streamed_nint_expert_tensor();
        test_mfe_wire_buffer_materialization();
        test_materialized_mfe_projections();
        test_direct_mfe_projection_tails();
        test_banked_vq_input_views();
        test_direct_mfe_projection_families();
        test_mixed_decode_row_packing(24);
        test_mixed_decode_row_packing(28);
        test_mixed_decode_row_packing(28, 1, 2560, 640);
        test_mixed_decode_row_packing(28, 6, 2560, 640);
        test_container_validation();
        std::cout
            << "MFQ native heterogeneous MFE/streamed "
               "routed Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
