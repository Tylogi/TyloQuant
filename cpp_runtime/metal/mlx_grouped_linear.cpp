#include "mlx_grouped_linear.h"

#include <mlx/allocator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kDescriptorSize = 12;
constexpr int kFamily = 0;
constexpr int kOutput = 1;

constexpr int kFamilyNint = 0;
constexpr int kNintBits = 2;
constexpr int kNintGroupSize = 3;
constexpr int kNintGroups = 4;
constexpr int kNintQOffset = 5;
constexpr int kNintSubOffset = 6;
constexpr int kNintAnchorOffset = 7;
constexpr int kNintQ5Execution = 8;

constexpr int kFamilyNint8Zero = 1;
constexpr int kQ8Groups = 2;
constexpr int kQ8QOffset = 3;
constexpr int kQ8ScaleOffset = 4;

constexpr int kFamilyVq = 2;
constexpr int kFamilyCccpInt4 = 3;
constexpr int kFamilyCccpPq = 4;
constexpr int kFamilyMx = 5;

struct DirectProjectionLayout {
    int family = kFamilyNint;
    int output_size = 0;
    int bits = 0;
    int group_size = 0;
    int groups = 0;
    bool q5_execution = false;
    int vector_size = 0;
    int vectors = 0;
    int index_bits = 0;
    int state_bits = 0;
    int states = 0;
    int entries = 0;
    int code_banks = 0;
    int aux_mode = 0;
    int code_bank_mode = 0;
    int table_banks = 0;
    int groups_per_supergroup = 0;
    int supergroups = 0;
    int blocks = 0;
    int tile_begin = 0;
    int tile_end = 0;
    int output_offset = 0;
};

constexpr const char* kGroupedHeader = R"METAL(
template <typename Stream>
inline uint mfq_grouped_nint_read_bits(
    Stream stream,
    uint value_index,
    uint bits
) {
    // Avoid overflowing the intermediate bit offset for large packed
    // projections whose byte offset still fits in uint.
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index =
        (value_index >> 3) * bits + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) {
        packed |= uint(stream[byte_index + 1u]) << 8;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

template <typename Stream>
inline uint mfq_grouped_nint_read_value(
    Stream stream,
    uint value_index,
    uint bits,
    uint group_size,
    uint q5_execution
) {
    if (q5_execution != 0u && bits == 5u) {
        uint metadata_index = value_index / group_size;
        uint element = value_index - metadata_index * group_size;
        uint low_bytes = (group_size + 1u) >> 1;
        uint high_bytes = (group_size + 7u) >> 3;
        uint group_offset =
            metadata_index * (low_bytes + high_bytes);
        uint low_packed =
            uint(stream[group_offset + (element >> 1)]);
        uint low =
            (low_packed >> ((element & 1u) * 4u)) & 15u;
        uint high = (
            uint(stream[
                group_offset + low_bytes + (element >> 3)])
            >> (element & 7u)
        ) & 1u;
        return low | (high << 4u);
    }
    return mfq_grouped_nint_read_bits(
        stream,
        value_index,
        bits);
}

template <typename Stream>
inline uint mfq_grouped_vq_read_bits(
    Stream stream,
    uint value_index,
    uint bits
) {
    uint residual_bits = (value_index & 7u) * bits;
    uint byte_index =
        (value_index >> 3) * bits + (residual_bits >> 3);
    uint shift = residual_bits & 7u;
    uint packed = uint(stream[byte_index])
        | (uint(stream[byte_index + 1u]) << 8)
        | (uint(stream[byte_index + 2u]) << 16);
    return (packed >> shift)
        & ((1u << bits) - 1u);
}

inline float mfq_grouped_mx_e8m0(uchar raw) {
    if (raw == 255u) {
        return NAN;
    }
    uint bits = raw == 0u ? 0x00400000u : uint(raw) << 23u;
    return as_type<float>(bits);
}

inline float mfq_grouped_mx_fp4(uchar raw) {
    uchar magnitude = raw & 7u;
    float value = magnitude == 0u ? 0.0f
        : (magnitude == 1u ? 0.5f
        : (magnitude == 2u ? 1.0f
        : (magnitude == 3u ? 1.5f
        : (magnitude == 4u ? 2.0f
        : (magnitude == 5u ? 3.0f
        : (magnitude == 6u ? 4.0f : 6.0f))))));
    return (raw & 8u) == 0u ? value : -value;
}

constant ushort mfq_grouped_mx_fp8_half_lut[256] = {
    0x0000u, 0x1800u, 0x1c00u, 0x1e00u, 0x2000u, 0x2100u, 0x2200u, 0x2300u,
    0x2400u, 0x2480u, 0x2500u, 0x2580u, 0x2600u, 0x2680u, 0x2700u, 0x2780u,
    0x2800u, 0x2880u, 0x2900u, 0x2980u, 0x2a00u, 0x2a80u, 0x2b00u, 0x2b80u,
    0x2c00u, 0x2c80u, 0x2d00u, 0x2d80u, 0x2e00u, 0x2e80u, 0x2f00u, 0x2f80u,
    0x3000u, 0x3080u, 0x3100u, 0x3180u, 0x3200u, 0x3280u, 0x3300u, 0x3380u,
    0x3400u, 0x3480u, 0x3500u, 0x3580u, 0x3600u, 0x3680u, 0x3700u, 0x3780u,
    0x3800u, 0x3880u, 0x3900u, 0x3980u, 0x3a00u, 0x3a80u, 0x3b00u, 0x3b80u,
    0x3c00u, 0x3c80u, 0x3d00u, 0x3d80u, 0x3e00u, 0x3e80u, 0x3f00u, 0x3f80u,
    0x4000u, 0x4080u, 0x4100u, 0x4180u, 0x4200u, 0x4280u, 0x4300u, 0x4380u,
    0x4400u, 0x4480u, 0x4500u, 0x4580u, 0x4600u, 0x4680u, 0x4700u, 0x4780u,
    0x4800u, 0x4880u, 0x4900u, 0x4980u, 0x4a00u, 0x4a80u, 0x4b00u, 0x4b80u,
    0x4c00u, 0x4c80u, 0x4d00u, 0x4d80u, 0x4e00u, 0x4e80u, 0x4f00u, 0x4f80u,
    0x5000u, 0x5080u, 0x5100u, 0x5180u, 0x5200u, 0x5280u, 0x5300u, 0x5380u,
    0x5400u, 0x5480u, 0x5500u, 0x5580u, 0x5600u, 0x5680u, 0x5700u, 0x5780u,
    0x5800u, 0x5880u, 0x5900u, 0x5980u, 0x5a00u, 0x5a80u, 0x5b00u, 0x5b80u,
    0x5c00u, 0x5c80u, 0x5d00u, 0x5d80u, 0x5e00u, 0x5e80u, 0x5f00u, 0x7e00u,
    0x8000u, 0x9800u, 0x9c00u, 0x9e00u, 0xa000u, 0xa100u, 0xa200u, 0xa300u,
    0xa400u, 0xa480u, 0xa500u, 0xa580u, 0xa600u, 0xa680u, 0xa700u, 0xa780u,
    0xa800u, 0xa880u, 0xa900u, 0xa980u, 0xaa00u, 0xaa80u, 0xab00u, 0xab80u,
    0xac00u, 0xac80u, 0xad00u, 0xad80u, 0xae00u, 0xae80u, 0xaf00u, 0xaf80u,
    0xb000u, 0xb080u, 0xb100u, 0xb180u, 0xb200u, 0xb280u, 0xb300u, 0xb380u,
    0xb400u, 0xb480u, 0xb500u, 0xb580u, 0xb600u, 0xb680u, 0xb700u, 0xb780u,
    0xb800u, 0xb880u, 0xb900u, 0xb980u, 0xba00u, 0xba80u, 0xbb00u, 0xbb80u,
    0xbc00u, 0xbc80u, 0xbd00u, 0xbd80u, 0xbe00u, 0xbe80u, 0xbf00u, 0xbf80u,
    0xc000u, 0xc080u, 0xc100u, 0xc180u, 0xc200u, 0xc280u, 0xc300u, 0xc380u,
    0xc400u, 0xc480u, 0xc500u, 0xc580u, 0xc600u, 0xc680u, 0xc700u, 0xc780u,
    0xc800u, 0xc880u, 0xc900u, 0xc980u, 0xca00u, 0xca80u, 0xcb00u, 0xcb80u,
    0xcc00u, 0xcc80u, 0xcd00u, 0xcd80u, 0xce00u, 0xce80u, 0xcf00u, 0xcf80u,
    0xd000u, 0xd080u, 0xd100u, 0xd180u, 0xd200u, 0xd280u, 0xd300u, 0xd380u,
    0xd400u, 0xd480u, 0xd500u, 0xd580u, 0xd600u, 0xd680u, 0xd700u, 0xd780u,
    0xd800u, 0xd880u, 0xd900u, 0xd980u, 0xda00u, 0xda80u, 0xdb00u, 0xdb80u,
    0xdc00u, 0xdc80u, 0xdd00u, 0xdd80u, 0xde00u, 0xde80u, 0xdf00u, 0xfe00u,
};

inline float mfq_grouped_mx_fp8(uchar raw) {
    return float(as_type<half>(
        mfq_grouped_mx_fp8_half_lut[uint(raw)]));
}

template <typename ValueStream, typename ScaleStream>
inline float mfq_grouped_mx_weight(
    ValueStream values,
    ScaleStream scales,
    uint output,
    uint column,
    uint bits,
    uint width
) {
    if (bits == 4u) {
        uchar packed = values[output * (width / 2u) + (column >> 1u)];
        uchar code = (column & 1u) == 0u
            ? packed & 15u
            : packed >> 4u;
        uchar scale = scales[output * (width / 32u) + column / 32u];
        return mfq_grouped_mx_fp4(code)
            * mfq_grouped_mx_e8m0(scale);
    }
    uchar code = values[output * width + column];
    uchar scale = scales[
        (output / 128u) * (width / 128u) + column / 128u];
    return mfq_grouped_mx_fp8(code)
        * mfq_grouped_mx_e8m0(scale);
}

template <
    typename IndicesPtr,
    typename StatePtr,
    typename AuxPtr,
    typename AnchorPtr,
    typename CodebookPtr,
    typename ScalePtr,
    typename StateBankPtr,
    typename BankPtr,
    typename ParameterPtr
>
inline float mfq_grouped_vq_decode_weight(
    IndicesPtr indices_packed,
    StatePtr state_packed,
    AuxPtr aux_packed,
    AnchorPtr anchors,
    CodebookPtr codebooks,
    ScalePtr scale_lut,
    StateBankPtr state_to_codebank,
    BankPtr bank_ids,
    ParameterPtr parameters,
    uint output,
    uint column,
    uint group_size,
    uint groups,
    uint vector_size,
    uint vectors,
    uint index_bits,
    uint state_bits,
    uint states,
    uint entries,
    uint code_banks,
    uint aux_mode,
    uint code_bank_mode,
    uint has_table_banks,
    uint groups_per_supergroup,
    uint supergroups,
    uint sign_groups
) {
    uint group = column / group_size;
    uint vector = column / vector_size;
    uint component =
        column - vector * vector_size;
    uint state_index =
        output * groups + group;
    uint state = mfq_grouped_vq_read_bits(
        state_packed,
        state_index,
        state_bits);
    uint table_bank = 0u;
    if (has_table_banks != 0u) {
        table_bank = uint(bank_ids[
            output * supergroups
            + group / groups_per_supergroup
        ]);
    }
    uint index = mfq_grouped_vq_read_bits(
        indices_packed,
        output * vectors + vector,
        index_bits);
    uint auxiliary = 0u;
    if (aux_mode == 1u || aux_mode == 2u) {
        auxiliary = mfq_grouped_vq_read_bits(
            aux_packed,
            output * sign_groups + column / 8u,
            7u);
    } else if (aux_mode == 3u) {
        auxiliary = mfq_grouped_vq_read_bits(
            aux_packed,
            state_index,
            1u);
    }
    uint code_bank = 0u;
    if (code_bank_mode == 1u) {
        code_bank =
            uint(state_to_codebank[state]);
    } else if (code_bank_mode == 2u) {
        code_bank = auxiliary;
    }
    uint code_offset = (
        (
            (
                table_bank * code_banks
                + code_bank
            )
            * entries + index
        )
        * vector_size + component
    );
    float code = float(codebooks[code_offset]);
    if (aux_mode == 1u || aux_mode == 2u) {
        uint sign_position = column & 7u;
        uint negative = sign_position < 7u
            ? ((auxiliary >> sign_position) & 1u)
            : (popcount(auxiliary) & 1u);
        if (
            aux_mode == 2u
            && sign_position == 7u
        ) {
            negative ^= (index >> 7u) & 1u;
        }
        code = negative != 0u ? -code : code;
    } else if (aux_mode == 3u) {
        code += auxiliary != 0u
            ? -parameters[0]
            : parameters[0];
    }
    return anchors[output]
        * scale_lut[table_bank * states + state]
        * code;
}
)METAL";

constexpr const char* kGroupedSource = R"METAL(
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint input_row = workgroup / uint(TOTAL_TILES);
    uint global_tile =
        workgroup - input_row * uint(TOTAL_TILES);
    if (input_row >= uint(ROWS)) {
        return;
    }

    uint projection = 0u;
    while (
        projection + 1u < uint(PROJECTIONS)
        && global_tile >=
            uint(projection_tile_offsets[projection + 1u])
    ) {
        ++projection;
    }
    uint local_tile =
        global_tile - uint(projection_tile_offsets[projection]);
    uint descriptor_base =
        projection * uint(DESCRIPTOR_SIZE);
    uint output_width =
        uint(descriptors[descriptor_base + 1u]);
    uint output_base =
        local_tile * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;
    uint output_offset =
        uint(projection_output_offsets[projection]);
    uint family = uint(descriptors[descriptor_base]);

    float accumulators[ROWS_PER_SIMD] = {0.0f};
    for (uint column = lane; column < uint(K); column += 32u) {
        float activation =
            float(x[input_row * uint(K) + column]);
        for (
            uint local_row = 0u;
            local_row < ROWS_PER_SIMD;
            ++local_row
        ) {
            uint output = output_base + local_row;
            if (output >= output_width) {
                continue;
            }

            float weight;
            if (family == 0u) {
                uint bits =
                    uint(descriptors[descriptor_base + 2u]);
                uint group_size =
                    uint(descriptors[descriptor_base + 3u]);
                uint groups =
                    uint(descriptors[descriptor_base + 4u]);
                uint group = column / group_size;
                uint element = column - group * group_size;
                uint metadata_index = output * groups + group;
                uint quantized_index =
                    metadata_index * group_size + element;
                uint quantized = mfq_grouped_nint_read_value(
                    nint_q
                        + uint(descriptors[
                            descriptor_base + 5u]),
                    quantized_index,
                    bits,
                    group_size,
                    uint(descriptors[descriptor_base + 8u]));
                uint sub_index =
                    uint(descriptors[descriptor_base + 6u])
                    + metadata_index;
                uint anchor_index =
                    uint(descriptors[descriptor_base + 7u])
                    + output;
                float scale =
                    nint_anchor_scale[anchor_index]
                    * float(nint_sub_scale[sub_index]);
                float minimum =
                    nint_anchor_min[anchor_index]
                    * float(nint_sub_min[sub_index]);
                weight =
                    scale * float(quantized) - minimum;
            } else {
                uint groups =
                    uint(descriptors[descriptor_base + 2u]);
                uint group = column >> 5;
                weight =
                    float(q8_scales[
                        uint(descriptors[
                            descriptor_base + 4u])
                        + output * groups + group])
                    * float(q8_q[
                        uint(descriptors[
                            descriptor_base + 3u])
                        + output * uint(K) + column]);
            }
            accumulators[local_row] = fma(
                activation,
                weight,
                accumulators[local_row]);
        }
    }

    for (
        uint local_row = 0u;
        local_row < ROWS_PER_SIMD;
        ++local_row
    ) {
        float total = simd_sum(accumulators[local_row]);
        uint output = output_base + local_row;
        if (lane == 0u && output < output_width) {
            y[
                input_row * uint(TOTAL_OUT)
                + output_offset + output
            ] = T(total);
        }
    }
)METAL";

std::int32_t checked_int(
    std::size_t value,
    const char* name) {
    if (value >
        static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max())) {
        throw MlxGroupedLinearUnsupported(
            std::string("grouped linear ") + name
            + " exceeds int32 range");
    }
    return static_cast<std::int32_t>(value);
}

void append_raw(
    std::vector<std::uint8_t>& target,
    const array& source,
    Dtype expected,
    const char* name) {
    if (source.dtype() != expected ||
        !source.flags().row_contiguous) {
        throw std::runtime_error(
            std::string("invalid grouped linear packed ") + name);
    }
    auto evaluated = source;
    evaluated.eval();
    const auto previous = target.size();
    if (evaluated.nbytes() >
        std::numeric_limits<std::size_t>::max() - previous) {
        throw MlxGroupedLinearUnsupported(
            "grouped linear packed stream size overflows");
    }
    target.resize(previous + evaluated.nbytes());
    std::memcpy(
        target.data() + previous,
        evaluated.data<std::uint8_t>(),
        evaluated.nbytes());
}

array make_raw_array(
    std::vector<std::uint8_t> bytes,
    Dtype dtype) {
    if (bytes.empty()) {
        bytes.resize(dtype.size(), 0);
    }
    if (bytes.size() % dtype.size() != 0) {
        throw std::runtime_error(
            "grouped linear packed stream is misaligned");
    }
    const auto elements = checked_int(
        bytes.size() / dtype.size(),
        "packed stream");
    auto result = array(
        mlx::core::allocator::malloc(bytes.size()),
        Shape{elements},
        dtype);
    std::memcpy(
        result.data<std::uint8_t>(),
        bytes.data(),
        bytes.size());
    return result;
}

array make_int32_array(
    const std::vector<std::int32_t>& values,
    Shape shape) {
    return array(values.begin(), std::move(shape));
}

mlx::core::fast::CustomKernelFunction make_grouped_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_heterogeneous_grouped_linear",
        {
            "descriptors",
            "projection_tile_offsets",
            "projection_output_offsets",
            "nint_q",
            "nint_sub_scale",
            "nint_sub_min",
            "nint_anchor_scale",
            "nint_anchor_min",
            "q8_q",
            "q8_scales",
            "x",
        },
        {"y"},
        kGroupedSource,
        kGroupedHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& grouped_kernel() {
    static const auto kernel = make_grouped_kernel();
    return kernel;
}

std::string direct_kernel_key(
    const std::vector<DirectProjectionLayout>& layouts) {
    std::string key;
    for (const auto& layout : layouts) {
        if (!key.empty()) {
            key.push_back('_');
        }
        if (layout.family == kFamilyNint8Zero) {
            key += "q8";
        } else if (layout.family == kFamilyVq) {
            key += "vq";
        } else if (
            layout.family == kFamilyCccpInt4
        ) {
            key += "ci4";
        } else if (
            layout.family == kFamilyCccpPq
        ) {
            key += "cpq";
        } else if (layout.family == kFamilyMx) {
            key += "mx" + std::to_string(layout.bits);
        } else if (layout.q5_execution) {
            key += "n5x";
        } else {
            key += "n";
        }
    }
    return key;
}

bool supports_single_row_nint_fast_path(
    const std::vector<DirectProjectionLayout>& layouts) noexcept {
    if (layouts.size() < 2 || layouts.size() > 3) {
        return false;
    }
    const int group_size = layouts.front().group_size;
    const int groups = layouts.front().groups;
    if (group_size <= 0 || groups <= 0) {
        return false;
    }
    // The fast kernel loads four FP16 activations at a time. Keeping every
    // group boundary half4-aligned avoids undefined vector loads.
    if ((group_size % 4) != 0) {
        return false;
    }
    for (const auto& layout : layouts) {
        if (layout.family != kFamilyNint ||
            layout.group_size != group_size ||
            layout.groups != groups) {
            return false;
        }
        if (layout.bits == 4) {
            if (layout.q5_execution ||
                (group_size % 2) != 0) {
                return false;
            }
        } else if (layout.bits == 5) {
            if (!layout.q5_execution) {
                return false;
            }
        } else if (layout.bits == 6) {
            if (layout.q5_execution ||
                (group_size % 4) != 0) {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

bool supports_single_row_mxfp8_fast_path(
    const std::vector<DirectProjectionLayout>& layouts) noexcept {
    if (layouts.size() < 2 || layouts.size() > 14) {
        return false;
    }
    bool found_mxfp8 = false;
    for (const auto& layout : layouts) {
        if (layout.family == kFamilyMx && layout.bits == 8) {
            found_mxfp8 = true;
            continue;
        }
        if (layout.family != kFamilyNint8Zero) {
            return false;
        }
    }
    return found_mxfp8;
}

std::string single_row_nint_kernel_key(
    const std::vector<DirectProjectionLayout>& layouts) {
    std::string key = "m1";
    for (const auto& layout : layouts) {
        key += "_b" + std::to_string(layout.bits)
            + "g" + std::to_string(layout.group_size)
            + (layout.q5_execution ? "x" : "p");
    }
    return key;
}

std::vector<std::string> direct_input_names(
    const std::vector<DirectProjectionLayout>& layouts) {
    std::vector<std::string> names;
    names.reserve(layouts.size() * 9 + 1);
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        if (layouts[projection].family == kFamilyNint) {
            names.push_back("q_packed_" + suffix);
            names.push_back("sub_scale_" + suffix);
            names.push_back("sub_min_" + suffix);
            names.push_back("neuron_scale_" + suffix);
            names.push_back("neuron_min_" + suffix);
        } else if (
            layouts[projection].family
                == kFamilyNint8Zero
        ) {
            names.push_back("q8_q_" + suffix);
            names.push_back("q8_scales_" + suffix);
        } else if (
            layouts[projection].family == kFamilyVq
        ) {
            names.push_back("vq_indices_" + suffix);
            names.push_back("vq_state_" + suffix);
            names.push_back("vq_aux_" + suffix);
            names.push_back("vq_anchors_" + suffix);
            names.push_back("vq_codebooks_" + suffix);
            names.push_back("vq_scales_" + suffix);
            names.push_back("vq_state_banks_" + suffix);
            names.push_back("vq_bank_ids_" + suffix);
            names.push_back("vq_parameters_" + suffix);
        } else if (
            layouts[projection].family
                == kFamilyCccpInt4
        ) {
            names.push_back(
                "cccp_i4_packed_" + suffix);
            names.push_back(
                "cccp_i4_scales_" + suffix);
        } else if (
            layouts[projection].family == kFamilyMx
        ) {
            names.push_back("mx_values_" + suffix);
            names.push_back("mx_scales_" + suffix);
        } else {
            names.push_back(
                "cccp_pq_indices_" + suffix);
            names.push_back(
                "cccp_pq_codebook_" + suffix);
        }
    }
    names.emplace_back("x");
    return names;
}

std::string make_direct_source(
    const std::vector<DirectProjectionLayout>& layouts) {
    std::string source = R"METAL(
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint input_row = workgroup / uint(TOTAL_TILES);
    uint global_tile =
        workgroup - input_row * uint(TOTAL_TILES);
    if (input_row >= uint(ROWS)) {
        return;
    }

    uint projection = 0u;
    uint local_tile = 0u;
    uint output_width = 0u;
    uint output_offset = 0u;
)METAL";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        if (projection == 0) {
            source += "    if (";
        } else if (projection + 1 < layouts.size()) {
            source += " else if (";
        } else {
            source += " else {\n";
            source += "        projection = "
                + suffix + "u;\n";
            source += "        local_tile = global_tile"
                " - uint(P" + suffix + "_TILE_BEGIN);\n";
            source += "        output_width = uint(P"
                + suffix + "_OUT);\n";
            source += "        output_offset = uint(P"
                + suffix + "_OUT_OFFSET);\n";
            source += "    }\n";
            continue;
        }
        source += "global_tile < uint(P"
            + suffix + "_TILE_END)) {\n";
        source += "        projection = "
            + suffix + "u;\n";
        source += "        local_tile = global_tile"
            " - uint(P" + suffix + "_TILE_BEGIN);\n";
        source += "        output_width = uint(P"
            + suffix + "_OUT);\n";
        source += "        output_offset = uint(P"
            + suffix + "_OUT_OFFSET);\n";
        source += "    }";
    }

    source += R"METAL(
    uint output_base =
        local_tile * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;

    float accumulators[ROWS_PER_SIMD] = {0.0f};
    for (uint column = lane; column < uint(K); column += 32u) {
        float activation =
            float(x[input_row * uint(K) + column]);
        for (
            uint local_row = 0u;
            local_row < ROWS_PER_SIMD;
            ++local_row
        ) {
            uint output = output_base + local_row;
            if (output >= output_width) {
                continue;
            }

            float weight = 0.0f;
)METAL";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        source += projection == 0
            ? "            if ("
            : " else if (";
        source += "projection == " + suffix + "u) {\n";
        if (layouts[projection].family == kFamilyNint) {
            source +=
                "                uint group = column / uint(P"
                + suffix + "_GS);\n"
                "                uint element = column"
                " - group * uint(P" + suffix + "_GS);\n"
                "                uint metadata_index ="
                " output * uint(P" + suffix + "_NG) + group;\n"
                "                uint quantized_index ="
                " metadata_index * uint(P" + suffix
                + "_GS) + element;\n"
                "                uint quantized ="
                " mfq_grouped_nint_read_value(\n"
                "                    q_packed_" + suffix + ",\n"
                "                    quantized_index,\n"
                "                    uint(P" + suffix + "_BITS),\n"
                "                    uint(P" + suffix + "_GS),\n"
                "                    "
                + std::string(
                    layouts[projection].q5_execution
                        ? "1u"
                        : "0u")
                + ");\n"
                "                float scale ="
                " neuron_scale_" + suffix + "[output]"
                " * float(sub_scale_" + suffix
                + "[metadata_index]);\n"
                "                float minimum ="
                " neuron_min_" + suffix + "[output]"
                " * float(sub_min_" + suffix
                + "[metadata_index]);\n"
                "                weight ="
                " scale * float(quantized) - minimum;\n";
        } else if (
            layouts[projection].family
                == kFamilyNint8Zero
        ) {
            source +=
                "                uint group = column >> 5;\n"
                "                weight = float(q8_scales_" + suffix
                + "[output * uint(P" + suffix
                + "_NG) + group])"
                " * float(q8_q_" + suffix
                + "[output * uint(K) + column]);\n";
        } else if (
            layouts[projection].family
                == kFamilyVq
        ) {
            source +=
                "                weight ="
                " mfq_grouped_vq_decode_weight(\n"
                "                    vq_indices_" + suffix + ",\n"
                "                    vq_state_" + suffix + ",\n"
                "                    vq_aux_" + suffix + ",\n"
                "                    vq_anchors_" + suffix + ",\n"
                "                    vq_codebooks_" + suffix + ",\n"
                "                    vq_scales_" + suffix + ",\n"
                "                    vq_state_banks_" + suffix + ",\n"
                "                    vq_bank_ids_" + suffix + ",\n"
                "                    vq_parameters_" + suffix + ",\n"
                "                    output,\n"
                "                    column,\n"
                "                    uint(P" + suffix + "_GS),\n"
                "                    uint(P" + suffix + "_NG),\n"
                "                    uint(P" + suffix + "_VECTOR_SIZE),\n"
                "                    uint(P" + suffix + "_NVEC),\n"
                "                    uint(P" + suffix + "_INDEX_BITS),\n"
                "                    uint(P" + suffix + "_STATE_BITS),\n"
                "                    uint(P" + suffix + "_STATES),\n"
                "                    uint(P" + suffix + "_ENTRIES),\n"
                "                    uint(P" + suffix + "_CODE_BANKS),\n"
                "                    uint(P" + suffix + "_AUX_MODE),\n"
                "                    uint(P" + suffix + "_CODE_BANK_MODE),\n"
                "                    uint(P" + suffix + "_HAS_TABLE_BANKS),\n"
                "                    uint(P" + suffix + "_GROUPS_PER_SUPER),\n"
                "                    uint(P" + suffix + "_NSUPER),\n"
                "                    (uint(K) + 7u) / 8u);\n";
        } else if (
            layouts[projection].family
                == kFamilyCccpInt4
        ) {
            source +=
                "                uint packed_value = uint(\n"
                "                    cccp_i4_packed_" + suffix + "[\n"
                "                        output * uint(K / 2)\n"
                "                        + (column >> 1)]);\n"
                "                uint quantized ="
                " (column & 1u) == 0u\n"
                "                    ? packed_value & 15u\n"
                "                    : packed_value >> 4u;\n"
                "                float scale = float(\n"
                "                    cccp_i4_scales_" + suffix + "[\n"
                "                        output * uint(P" + suffix
                + "_NG)\n"
                "                        + column / uint(P" + suffix
                + "_GS)]);\n"
                "                weight ="
                " float(int(quantized) - 8) * scale;\n";
        } else if (
            layouts[projection].family == kFamilyMx
        ) {
            source +=
                "                weight = mfq_grouped_mx_weight(\n"
                "                    mx_values_" + suffix + ",\n"
                "                    mx_scales_" + suffix + ",\n"
                "                    output, column,\n"
                "                    uint(P" + suffix + "_BITS),\n"
                "                    uint(K));\n";
        } else {
            source +=
                "                uint block ="
                " column / uint(P" + suffix
                + "_VECTOR_SIZE);\n"
                "                uint component ="
                " column - block * uint(P" + suffix
                + "_VECTOR_SIZE);\n"
                "                uint code ="
                " mfq_grouped_vq_read_bits(\n"
                "                    cccp_pq_indices_" + suffix + ",\n"
                "                    output * uint(P" + suffix
                + "_BLOCKS) + block,\n"
                "                    uint(P" + suffix
                + "_INDEX_BITS));\n"
                "                weight = float(\n"
                "                    cccp_pq_codebook_" + suffix + "[\n"
                "                        code * uint(P" + suffix
                + "_VECTOR_SIZE) + component]);\n";
        }
        source += "            }";
    }

    source += R"METAL(
            accumulators[local_row] = fma(
                activation,
                weight,
                accumulators[local_row]);
        }
    }

    for (
        uint local_row = 0u;
        local_row < ROWS_PER_SIMD;
        ++local_row
    ) {
        float total = simd_sum(accumulators[local_row]);
        uint output = output_base + local_row;
        if (lane == 0u && output < output_width) {
            y[
                input_row * uint(TOTAL_OUT)
                + output_offset + output
            ] = T(total);
        }
    }
)METAL";
    return source;
}

mlx::core::fast::CustomKernelFunction make_direct_kernel(
    const std::vector<DirectProjectionLayout>& layouts) {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    const auto key = direct_kernel_key(layouts);
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_zero_copy_grouped_linear_" + key,
        direct_input_names(layouts),
        {"y"},
        make_direct_source(layouts),
        kGroupedHeader,
        true,
        false,
        options);
}

mlx::core::fast::CustomKernelFunction direct_kernel(
    const std::vector<DirectProjectionLayout>& layouts) {
    static std::mutex mutex;
    static std::unordered_map<
        std::string,
        mlx::core::fast::CustomKernelFunction> kernels;

    const auto key = direct_kernel_key(layouts);
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = kernels.find(key);
    if (found != kernels.end()) {
        return found->second;
    }
    auto kernel = make_direct_kernel(layouts);
    kernels.emplace(key, kernel);
    return kernel;
}

std::string make_single_row_nint_source(
    const std::vector<DirectProjectionLayout>& layouts,
    bool fused_swiglu) {
    std::string source = R"METAL(
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;
)METAL";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        source +=
            "    bool active_" + suffix
            + " = output_base < uint(P" + suffix + "_OUT);\n"
            "    uint outputs_" + suffix
            + "[ROWS_PER_SIMD];\n"
            "    uint metadata_bases_" + suffix
            + "[ROWS_PER_SIMD];\n"
            "    float neuron_scales_" + suffix
            + "[ROWS_PER_SIMD];\n"
            "    float neuron_minimums_" + suffix
            + "[ROWS_PER_SIMD];\n"
            "    float accumulators_" + suffix
            + "[ROWS_PER_SIMD] = {0.0f};\n"
            "    if (active_" + suffix + ") {\n"
            "        for (uint row = 0u;"
            " row < ROWS_PER_SIMD; ++row) {\n"
            "            uint output = min(\n"
            "                output_base + row,"
            " uint(P" + suffix + "_OUT) - 1u);\n"
            "            outputs_" + suffix
            + "[row] = output;\n"
            "            metadata_bases_" + suffix
            + "[row] = output * uint(NG);\n"
            "            neuron_scales_" + suffix
            + "[row] = neuron_scale_" + suffix
            + "[output];\n"
            "            neuron_minimums_" + suffix
            + "[row] = neuron_min_" + suffix
            + "[output];\n"
            "        }\n"
            "    }\n";
    }

    source += R"METAL(
    // A lane owns complete quantization groups. Activations and their sum are
    // loaded once and reused by every active projection in this output tile.
    for (uint group = lane; group < uint(NG); group += 32u) {
        float activation_sum = 0.0f;
)METAL";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        source +=
            "        float quantized_dots_" + suffix
            + "[ROWS_PER_SIMD] = {0.0f};\n";
    }

    source += R"METAL(
        for (uint element = 0u; element < uint(GS); element += 4u) {
            uint column = group * uint(GS) + element;
            float4 activation = float4(0.0f);
            if (column + 3u < uint(K)) {
                activation =
                    float4(*(device const half4*)(x + column));
            } else {
                activation.x =
                    column < uint(K) ? float(x[column]) : 0.0f;
                activation.y =
                    column + 1u < uint(K)
                    ? float(x[column + 1u]) : 0.0f;
                activation.z =
                    column + 2u < uint(K)
                    ? float(x[column + 2u]) : 0.0f;
                activation.w =
                    column + 3u < uint(K)
                    ? float(x[column + 3u]) : 0.0f;
            }
            activation_sum +=
                activation.x + activation.y
                + activation.z + activation.w;
)METAL";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        source +=
            "            if (active_" + suffix + ") {\n"
            "                for (uint row = 0u;"
            " row < ROWS_PER_SIMD; ++row) {\n"
            "                    uint metadata_index =\n"
            "                        metadata_bases_" + suffix
            + "[row] + group;\n";
        if (layouts[projection].bits == 4) {
            source +=
                "                    constexpr uint GROUP_BYTES ="
                " uint(GS) / 2u;\n"
                "                    uint byte_index ="
                " metadata_index * GROUP_BYTES"
                " + (element >> 1u);\n"
                "                    uint packed0 ="
                " uint(q_packed_" + suffix
                + "[byte_index]);\n"
                "                    uint packed1 ="
                " uint(q_packed_" + suffix
                + "[byte_index + 1u]);\n"
                "                    float4 quantized = float4(\n"
                "                        float(packed0 & 15u),\n"
                "                        float(packed0 >> 4u),\n"
                "                        float(packed1 & 15u),\n"
                "                        float(packed1 >> 4u));\n";
        } else if (layouts[projection].bits == 5) {
            source +=
                "                    constexpr uint LOW_BYTES ="
                " (uint(GS) + 1u) / 2u;\n"
                "                    constexpr uint EXEC_BYTES ="
                " LOW_BYTES + (uint(GS) + 7u) / 8u;\n"
                "                    uint group_offset ="
                " metadata_index * EXEC_BYTES;\n"
                "                    uint low0 = uint(q_packed_"
                + suffix
                + "[group_offset + (element >> 1u)]);\n"
                "                    uint low1 = uint(q_packed_"
                + suffix
                + "[group_offset + (element >> 1u) + 1u]);\n"
                "                    uint high ="
                " uint(q_packed_" + suffix
                + "[group_offset + LOW_BYTES"
                " + (element >> 3u)])"
                " >> (element & 7u);\n"
                "                    float4 quantized = float4(\n"
                "                        float((low0 & 15u)"
                " | ((high & 1u) << 4u)),\n"
                "                        float((low0 >> 4u)"
                " | (((high >> 1u) & 1u) << 4u)),\n"
                "                        float((low1 & 15u)"
                " | (((high >> 2u) & 1u) << 4u)),\n"
                "                        float((low1 >> 4u)"
                " | (((high >> 3u) & 1u) << 4u)));\n";
        } else {
            source +=
                "                    constexpr uint GROUP_BYTES ="
                " (uint(GS) / 4u) * 3u;\n"
                "                    uint byte_index ="
                " metadata_index * GROUP_BYTES"
                " + (element >> 2u) * 3u;\n"
                "                    uint packed =\n"
                "                        uint(q_packed_" + suffix
                + "[byte_index])\n"
                "                        | (uint(q_packed_" + suffix
                + "[byte_index + 1u]) << 8u)\n"
                "                        | (uint(q_packed_" + suffix
                + "[byte_index + 2u]) << 16u);\n"
                "                    float4 quantized = float4(\n"
                "                        float(packed & 63u),\n"
                "                        float((packed >> 6u) & 63u),\n"
                "                        float((packed >> 12u) & 63u),\n"
                "                        float((packed >> 18u) & 63u));\n";
        }
        source +=
            "                    quantized_dots_" + suffix
            + "[row] += dot(activation, quantized);\n"
            "                }\n"
            "            }\n";
    }

    source += "        }\n";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        source +=
            "        if (active_" + suffix + ") {\n"
            "            for (uint row = 0u;"
            " row < ROWS_PER_SIMD; ++row) {\n"
            "                uint metadata_index ="
            " metadata_bases_" + suffix + "[row] + group;\n"
            "                float scale ="
            " neuron_scales_" + suffix + "[row]"
            " * float(sub_scale_" + suffix
            + "[metadata_index]);\n"
            "                float minimum ="
            " neuron_minimums_" + suffix + "[row]"
            " * float(sub_min_" + suffix
            + "[metadata_index]);\n"
            "                accumulators_" + suffix + "[row] = fma(\n"
            "                    scale,"
            " quantized_dots_" + suffix + "[row],\n"
            "                    fma(-minimum, activation_sum,"
            " accumulators_" + suffix + "[row]));\n"
            "            }\n"
            "        }\n";
    }
    source += "    }\n";

    if (fused_swiglu) {
        source += R"METAL(
    if (active_0 && active_1) {
        for (uint row = 0u; row < ROWS_PER_SIMD; ++row) {
            float gate = simd_sum(accumulators_0[row]);
            float up = simd_sum(accumulators_1[row]);
            uint output = output_base + row;
            if (lane == 0u && output < uint(P0_OUT)) {
                gate = float(T(gate));
                up = float(T(up));
                if (params[0] > 0.0f) {
                    gate = min(gate, params[0]);
                    up = clamp(up, -params[0], params[0]);
                }
                float activated = gate / (1.0f + exp(-gate));
                y[output] = T(activated * up);
            }
        }
    }
)METAL";
    } else {
        for (
            std::size_t projection = 0;
            projection < layouts.size();
            ++projection
        ) {
            const auto suffix = std::to_string(projection);
            source +=
                "    if (active_" + suffix + ") {\n"
                "        for (uint row = 0u;"
                " row < ROWS_PER_SIMD; ++row) {\n"
                "            float total ="
                " simd_sum(accumulators_" + suffix + "[row]);\n"
                "            uint output = output_base + row;\n"
                "            if (lane == 0u"
                " && output < uint(P" + suffix + "_OUT)) {\n"
                "                y[uint(P" + suffix
                + "_OUT_OFFSET) + output] = T(total);\n"
                "            }\n"
                "        }\n"
                "    }\n";
        }
    }
    return source;
}

mlx::core::fast::CustomKernelFunction make_single_row_nint_kernel(
    const std::vector<DirectProjectionLayout>& layouts,
    bool fused_swiglu) {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    const auto key = single_row_nint_kernel_key(layouts);
    auto inputs = direct_input_names(layouts);
    if (fused_swiglu) {
        inputs.emplace_back("params");
    }
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_single_row_grouped_nint_" + key
            + (fused_swiglu ? "_swiglu" : ""),
        std::move(inputs),
        {"y"},
        make_single_row_nint_source(layouts, fused_swiglu),
        "",
        true,
        false,
        options);
}

mlx::core::fast::CustomKernelFunction single_row_nint_kernel(
    const std::vector<DirectProjectionLayout>& layouts,
    bool fused_swiglu = false) {
    static std::mutex mutex;
    static std::unordered_map<
        std::string,
        mlx::core::fast::CustomKernelFunction> kernels;

    const auto key = single_row_nint_kernel_key(layouts)
        + (fused_swiglu ? "_swiglu" : "");
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = kernels.find(key);
    if (found != kernels.end()) {
        return found->second;
    }
    auto kernel = make_single_row_nint_kernel(
        layouts,
        fused_swiglu);
    kernels.emplace(key, kernel);
    return kernel;
}

std::string make_single_row_mxfp8_source(
    const std::vector<DirectProjectionLayout>& layouts) {
    std::string source = R"METAL(
    constexpr uint SIMD_GROUPS = 4u;
    constexpr uint K_LANES = 8u;
    constexpr uint ROWS_PER_SIMD = 32u / K_LANES;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint MX_BLOCK = 128u;
    constexpr uint MX_BLOCKS = uint(K) / MX_BLOCK;
    constexpr uint Q8_GROUP = 32u;
    constexpr uint Q8_GROUPS_PER_MX = MX_BLOCK / Q8_GROUP;

    threadgroup half fp8_lut[256];
    uint local_thread = thread_index_in_threadgroup;
    fp8_lut[local_thread] = as_type<half>(
        mfq_grouped_mx_fp8_half_lut[local_thread]);
    fp8_lut[local_thread + 128u] = as_type<half>(
        mfq_grouped_mx_fp8_half_lut[local_thread + 128u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_row = lane / K_LANES;
    uint output_index =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD + simd_row;
)METAL";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        source +=
            "    bool active_" + suffix
            + " = output_index < uint(P" + suffix + "_OUT);\n"
            "    uint output_" + suffix + " = min(\n"
            "        output_index, uint(P" + suffix + "_OUT) - 1u);\n"
            "    float accumulator_" + suffix + " = 0.0f;\n";
    }

    source += R"METAL(
    for (
        uint block = k_lane;
        block < MX_BLOCKS;
        block += K_LANES
    ) {
        uint column_base = block * MX_BLOCK;
)METAL";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        if (layouts[projection].family == kFamilyMx) {
            source +=
                "        float mx_dot_" + std::to_string(projection)
                + " = 0.0f;\n";
        }
    }

    source += R"METAL(
        for (
            uint q8_local = 0u;
            q8_local < Q8_GROUPS_PER_MX;
            ++q8_local
        ) {
)METAL";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        if (layouts[projection].family == kFamilyNint8Zero) {
            source +=
                "            float q8_dot_" + std::to_string(projection)
                + " = 0.0f;\n";
        }
    }

    source += R"METAL(
            uint q8_column_base =
                column_base + q8_local * Q8_GROUP;
            for (uint element = 0u; element < Q8_GROUP; element += 4u) {
                uint column = q8_column_base + element;
                half4 activation = *(device const half4*)(x + column);
)METAL";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        if (layouts[projection].family == kFamilyMx) {
            source +=
                "                if (active_" + suffix + ") {\n"
                "                    uint value_offset = output_" + suffix
                + " * uint(K) + column;\n"
                "                    uchar4 code = *(device const uchar4*)(\n"
                "                        mx_values_" + suffix
                + " + value_offset);\n"
                "                    mx_dot_" + suffix + " = fma(\n"
                "                        float(activation.x),\n"
                "                        float(fp8_lut[uint(code.x)]),\n"
                "                        mx_dot_" + suffix + ");\n"
                "                    mx_dot_" + suffix + " = fma(\n"
                "                        float(activation.y),\n"
                "                        float(fp8_lut[uint(code.y)]),\n"
                "                        mx_dot_" + suffix + ");\n"
                "                    mx_dot_" + suffix + " = fma(\n"
                "                        float(activation.z),\n"
                "                        float(fp8_lut[uint(code.z)]),\n"
                "                        mx_dot_" + suffix + ");\n"
                "                    mx_dot_" + suffix + " = fma(\n"
                "                        float(activation.w),\n"
                "                        float(fp8_lut[uint(code.w)]),\n"
                "                        mx_dot_" + suffix + ");\n"
                "                }\n";
        } else {
            source +=
                "                if (active_" + suffix + ") {\n"
                "                    uint value_offset = output_" + suffix
                + " * uint(K) + column;\n"
                "                    char4 quantized = *(device const char4*)(\n"
                "                        q8_q_" + suffix
                + " + value_offset);\n"
                "                    q8_dot_" + suffix + " = fma(\n"
                "                        float(activation.x),\n"
                "                        float(quantized.x),\n"
                "                        q8_dot_" + suffix + ");\n"
                "                    q8_dot_" + suffix + " = fma(\n"
                "                        float(activation.y),\n"
                "                        float(quantized.y),\n"
                "                        q8_dot_" + suffix + ");\n"
                "                    q8_dot_" + suffix + " = fma(\n"
                "                        float(activation.z),\n"
                "                        float(quantized.z),\n"
                "                        q8_dot_" + suffix + ");\n"
                "                    q8_dot_" + suffix + " = fma(\n"
                "                        float(activation.w),\n"
                "                        float(quantized.w),\n"
                "                        q8_dot_" + suffix + ");\n"
                "                }\n";
        }
    }

    source += "            }\n";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        if (layouts[projection].family != kFamilyNint8Zero) {
            continue;
        }
        const auto suffix = std::to_string(projection);
        source +=
            "            if (active_" + suffix + ") {\n"
            "                uint group = block * Q8_GROUPS_PER_MX"
            " + q8_local;\n"
            "                float scale = float(q8_scales_" + suffix
            + "[output_" + suffix + " * uint(P" + suffix
            + "_NG) + group]);\n"
            "                accumulator_" + suffix + " = fma(\n"
            "                    scale, q8_dot_" + suffix + ",\n"
            "                    accumulator_" + suffix + ");\n"
            "            }\n";
    }
    source += "        }\n";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        if (layouts[projection].family != kFamilyMx) {
            continue;
        }
        const auto suffix = std::to_string(projection);
        source +=
            "        if (active_" + suffix + ") {\n"
            "            uint scale_offset =\n"
            "                (output_" + suffix
            + " / MX_BLOCK) * MX_BLOCKS + block;\n"
            "            accumulator_" + suffix + " = fma(\n"
            "                mfq_grouped_mx_e8m0(\n"
            "                    mx_scales_" + suffix
            + "[scale_offset]),\n"
            "                mx_dot_" + suffix + ",\n"
            "                accumulator_" + suffix + ");\n"
            "        }\n";
    }
    source += "    }\n";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        source +=
            "    accumulator_" + suffix
            + " += simd_shuffle_down(accumulator_" + suffix + ", 4);\n"
            "    accumulator_" + suffix
            + " += simd_shuffle_down(accumulator_" + suffix + ", 2);\n"
            "    accumulator_" + suffix
            + " += simd_shuffle_down(accumulator_" + suffix + ", 1);\n"
            "    if (k_lane == 0u && active_" + suffix + ") {\n"
            "        y[uint(P" + suffix + "_OUT_OFFSET) + output_index] =\n"
            "            half(accumulator_" + suffix + ");\n"
            "    }\n";
    }
    return source;
}

mlx::core::fast::CustomKernelFunction single_row_mxfp8_kernel(
    const std::vector<DirectProjectionLayout>& layouts) {
    static std::mutex mutex;
    static std::unordered_map<
        std::string,
        mlx::core::fast::CustomKernelFunction> kernels;
    const auto key = "mxfp8_m1_" + direct_kernel_key(layouts);
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = kernels.find(key);
    if (found != kernels.end()) {
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_single_row_grouped_" + key,
        direct_input_names(layouts),
        {"y"},
        make_single_row_mxfp8_source(layouts),
        kGroupedHeader,
        true,
        false,
        options);
    kernels.emplace(key, kernel);
    return kernel;
}

constexpr const char* kSingleRowMxfp8PairSwiglu = R"METAL(
    constexpr uint SIMD_GROUPS = 4u;
    constexpr uint K_LANES = 16u;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS;
    constexpr uint BLOCK = 128u;
    constexpr uint BLOCKS = uint(K) / BLOCK;

    threadgroup half fp8_lut[256];
    uint local_thread = thread_index_in_threadgroup;
    fp8_lut[local_thread] = as_type<half>(
        mfq_grouped_mx_fp8_half_lut[local_thread]);
    fp8_lut[local_thread + 128u] = as_type<half>(
        mfq_grouped_mx_fp8_half_lut[local_thread + 128u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint lane = thread_index_in_simdgroup;
    uint projection = lane >> 4u;
    uint k_lane = lane & (K_LANES - 1u);
    uint output_index =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simdgroup_index_in_threadgroup;
    uint output = min(output_index, uint(OUT) - 1u);
    uint value_base = output * uint(K);
    uint scale_base = (output / BLOCK) * BLOCKS;
    device const uchar* values = projection == 0u
        ? mx_values_0
        : mx_values_1;
    float accumulator = 0.0f;

    // Gate and Up occupy separate SIMD half-groups. Both halves visit the
    // same 128-column block at once: the 16 lanes in each half read one
    // contiguous 128-byte MXFP8 row segment instead of maintaining 16
    // strided weight streams. The activation is tiny and remains cache-hot.
    for (uint block = 0u; block < BLOCKS; ++block) {
        uint column_base = block * BLOCK;
        uint column = column_base + k_lane * 8u;
        half4 activation0 = *(device const half4*)(x + column);
        half4 activation1 = *(device const half4*)(x + column + 4u);
        uchar4 code0 = *(device const uchar4*)(
            values + value_base + column);
        uchar4 code1 = *(device const uchar4*)(
            values + value_base + column + 4u);
        float4 weight0 = float4(
            float(fp8_lut[uint(code0.x)]),
            float(fp8_lut[uint(code0.y)]),
            float(fp8_lut[uint(code0.z)]),
            float(fp8_lut[uint(code0.w)]));
        float4 weight1 = float4(
            float(fp8_lut[uint(code1.x)]),
            float(fp8_lut[uint(code1.y)]),
            float(fp8_lut[uint(code1.z)]),
            float(fp8_lut[uint(code1.w)]));
        float block_dot =
            dot(float4(activation0), weight0)
            + dot(float4(activation1), weight1);
        uchar scale = projection == 0u
            ? mx_scales_0[scale_base + block]
            : mx_scales_1[scale_base + block];
        accumulator = fma(
            mfq_grouped_mx_e8m0(scale),
            block_dot,
            accumulator);
    }

    // Each 16-lane half is an independent reduction tree. Only lanes 0 and
    // 16 are consumed, so shuffle-down traffic from inactive upper nodes
    // cannot cross-contaminate the two projection sums.
    accumulator += simd_shuffle_down(accumulator, 8);
    accumulator += simd_shuffle_down(accumulator, 4);
    accumulator += simd_shuffle_down(accumulator, 2);
    accumulator += simd_shuffle_down(accumulator, 1);
    float gate = simd_shuffle(accumulator, 0u);
    float up = simd_shuffle(accumulator, 16u);
    if (lane == 0u && output_index < uint(OUT)) {
        // Match the unfused graph's MXFP8 GEMV -> FP16 boundary exactly.
        gate = float(half(gate));
        up = float(half(up));
        if (params[0] > 0.0f) {
            gate = min(gate, params[0]);
            up = clamp(up, -params[0], params[0]);
        }
        float activated = gate / (1.0f + exp(-gate));
        y[output_index] = half(activated * up);
    }
)METAL";

const mlx::core::fast::CustomKernelFunction&
single_row_mxfp8_pair_swiglu_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_single_row_mxfp8_pair_swiglu",
            {
                "mx_values_0",
                "mx_scales_0",
                "mx_values_1",
                "mx_scales_1",
                "x",
                "params",
            },
            {"y"},
            kSingleRowMxfp8PairSwiglu,
            kGroupedHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

void validate_direct_array(
    const array& value,
    Dtype dtype,
    const char* name) {
    if (value.dtype() != dtype ||
        !value.flags().row_contiguous) {
        throw std::runtime_error(
            std::string("invalid zero-copy grouped linear ") + name);
    }
}

int weight_input_size(const MlxGroupedLinearWeightRef& value) {
    return std::visit(
        [](const auto* weight) -> int {
            if (weight == nullptr) {
                throw std::invalid_argument(
                    "grouped linear weight cannot be null");
            }
            return weight->input_size();
        },
        value);
}

int weight_output_size(const MlxGroupedLinearWeightRef& value) {
    return std::visit(
        [](const auto* weight) -> int {
            if (weight == nullptr) {
                throw std::invalid_argument(
                    "grouped linear weight cannot be null");
            }
            return weight->output_size();
        },
        value);
}

} // namespace

struct MlxGroupedLinear::Impl {
    std::optional<array> descriptors;
    std::optional<array> projection_tile_offsets;
    std::optional<array> projection_output_offsets;
    std::optional<array> nint_q;
    std::optional<array> nint_sub_scale;
    std::optional<array> nint_sub_min;
    std::optional<array> nint_anchor_scale;
    std::optional<array> nint_anchor_min;
    std::optional<array> q8_q;
    std::optional<array> q8_scales;
    std::vector<DirectProjectionLayout> direct_layouts;
    std::vector<array> direct_weight_inputs;
    std::vector<int> output_sizes;
    int input_size = 0;
    int total_output_size = 0;
    int total_tiles = 0;
    int single_row_tiles = 0;
    int single_row_mxfp8_tiles = 0;
    std::size_t packed_bytes = 0;
    std::size_t copied_packed_bytes = 0;

    Impl(
        array descriptors_value,
        array tile_offsets_value,
        array output_offsets_value,
        array nint_q_value,
        array nint_sub_scale_value,
        array nint_sub_min_value,
        array nint_anchor_scale_value,
        array nint_anchor_min_value,
        array q8_q_value,
        array q8_scales_value,
        std::vector<int> widths,
        int width,
        int total_output,
        int tiles,
        std::size_t bytes)
        : descriptors(std::move(descriptors_value)),
          projection_tile_offsets(
              std::move(tile_offsets_value)),
          projection_output_offsets(
              std::move(output_offsets_value)),
          nint_q(std::move(nint_q_value)),
          nint_sub_scale(std::move(nint_sub_scale_value)),
          nint_sub_min(std::move(nint_sub_min_value)),
          nint_anchor_scale(std::move(nint_anchor_scale_value)),
          nint_anchor_min(std::move(nint_anchor_min_value)),
          q8_q(std::move(q8_q_value)),
          q8_scales(std::move(q8_scales_value)),
          output_sizes(std::move(widths)),
          input_size(width),
          total_output_size(total_output),
          total_tiles(tiles),
          packed_bytes(bytes),
          copied_packed_bytes(bytes) {}

    Impl(
        std::vector<DirectProjectionLayout> layouts,
        std::vector<array> weight_inputs,
        std::vector<int> widths,
        int width,
        int total_output,
        int tiles,
        std::size_t bytes)
        : direct_layouts(std::move(layouts)),
          direct_weight_inputs(std::move(weight_inputs)),
          output_sizes(std::move(widths)),
          input_size(width),
          total_output_size(total_output),
          total_tiles(tiles),
          packed_bytes(bytes) {
        if (supports_single_row_nint_fast_path(
                direct_layouts)) {
            for (const auto& layout : direct_layouts) {
                single_row_tiles = std::max(
                    single_row_tiles,
                    (layout.output_size + 7) / 8);
            }
        }
        if (supports_single_row_mxfp8_fast_path(
                direct_layouts)) {
            for (const auto& layout : direct_layouts) {
                single_row_mxfp8_tiles = std::max(
                    single_row_mxfp8_tiles,
                    (layout.output_size + 15) / 16);
            }
        }
    }

    bool uses_zero_copy_storage() const noexcept {
        return !direct_layouts.empty();
    }

    bool has_single_row_nint_fast_path() const noexcept {
        return single_row_tiles > 0;
    }

    bool has_single_row_mxfp8_fast_path() const noexcept {
        return single_row_mxfp8_tiles > 0;
    }

    bool supports_single_row_swiglu() const noexcept {
        if (direct_layouts.size() != 2
            || output_sizes.size() != 2
            || output_sizes[0] != output_sizes[1]) {
            return false;
        }
        const bool nint_pair = has_single_row_nint_fast_path();
        const bool mxfp8_pair =
            has_single_row_mxfp8_fast_path()
            && direct_layouts[0].family == kFamilyMx
            && direct_layouts[0].bits == 8
            && direct_layouts[1].family == kFamilyMx
            && direct_layouts[1].bits == 8;
        return nint_pair || mxfp8_pair;
    }
};

MlxGroupedLinear::MlxGroupedLinear(
    std::vector<MlxGroupedLinearWeightRef> weights) {
    if (weights.size() < 2) {
        throw std::invalid_argument(
            "grouped linear requires at least two projections");
    }
    const int shared_input = weight_input_size(weights.front());
    if (shared_input <= 0) {
        throw std::invalid_argument(
            "grouped linear input width must be positive");
    }

    // Q/K/V and gate/up are the production hot paths. Binding the source
    // arrays directly avoids materializing a second model-sized packed pool.
    // Three VQ projections are the largest supported binding: 27 weight
    // buffers (29 resources including x/y), below Metal's 31-buffer kernel
    // argument limit.
    const bool direct_mxfp8_group =
        weights.size() <= 14 &&
        std::any_of(
            weights.begin(),
            weights.end(),
            [](const MlxGroupedLinearWeightRef& weight) {
                const auto* mx = std::get_if<
                    const MlxMxWeight*>(&weight);
                return mx != nullptr && *mx != nullptr &&
                    (*mx)->bits() == 8;
            }) &&
        std::all_of(
            weights.begin(),
            weights.end(),
            [](const MlxGroupedLinearWeightRef& weight) {
                if (std::holds_alternative<
                        const MlxNint8ZeroWeight*>(weight)) {
                    return true;
                }
                const auto* mx = std::get_if<
                    const MlxMxWeight*>(&weight);
                return mx != nullptr && *mx != nullptr &&
                    (*mx)->bits() == 8;
            });
    if (weights.size() <= 3 || direct_mxfp8_group) {
        std::vector<DirectProjectionLayout> layouts;
        std::vector<array> direct_inputs;
        std::vector<int> output_sizes;
        layouts.reserve(weights.size());
        direct_inputs.reserve(weights.size() * 9);
        output_sizes.reserve(weights.size());
        int total_output = 0;
        int total_tiles = 0;
        std::size_t packed_bytes = 0;

        for (
            std::size_t projection = 0;
            projection < weights.size();
            ++projection
        ) {
            const auto& source = weights[projection];
            if (weight_input_size(source) != shared_input) {
                throw std::invalid_argument(
                    "grouped linear projections must share one input width");
            }
            const int output = weight_output_size(source);
            if (output <= 0) {
                throw std::invalid_argument(
                    "grouped linear output width must be positive");
            }

            DirectProjectionLayout layout;
            layout.output_size = output;
            layout.tile_begin = total_tiles;
            layout.output_offset = total_output;
            total_output = checked_int(
                static_cast<std::size_t>(total_output)
                    + static_cast<std::size_t>(output),
                "total output width");
            total_tiles = checked_int(
                static_cast<std::size_t>(total_tiles)
                    + static_cast<std::size_t>((output + 7) / 8),
                "tile count");
            layout.tile_end = total_tiles;

            std::size_t weight_bytes = 0;
            std::visit(
                [&](const auto* weight) {
                    using Weight = std::remove_cv_t<
                        std::remove_pointer_t<decltype(weight)>>;
                    if constexpr (
                        std::is_same_v<Weight, MlxNintWeight>
                    ) {
                        layout.family = kFamilyNint;
                        layout.bits = weight->bits();
                        layout.group_size = weight->group_size();
                        layout.groups = weight->groups();
                        layout.q5_execution =
                            weight->q5_execution_layout();
                        validate_direct_array(
                            weight->packed_values(),
                            mlx::core::uint8,
                            "NINT q");
                        validate_direct_array(
                            weight->sub_scales(),
                            mlx::core::uint8,
                            "NINT sub scales");
                        validate_direct_array(
                            weight->sub_mins(),
                            mlx::core::uint8,
                            "NINT sub minima");
                        validate_direct_array(
                            weight->neuron_scales(),
                            mlx::core::float32,
                            "NINT neuron scales");
                        validate_direct_array(
                            weight->neuron_mins(),
                            mlx::core::float32,
                            "NINT neuron minima");
                        direct_inputs.push_back(
                            weight->packed_values());
                        direct_inputs.push_back(
                            weight->sub_scales());
                        direct_inputs.push_back(
                            weight->sub_mins());
                        direct_inputs.push_back(
                            weight->neuron_scales());
                        direct_inputs.push_back(
                            weight->neuron_mins());
                    } else if constexpr (
                        std::is_same_v<
                            Weight,
                            MlxNint8ZeroWeight>
                    ) {
                        layout.family = kFamilyNint8Zero;
                        layout.groups = weight->groups();
                        validate_direct_array(
                            weight->quantized_values(),
                            mlx::core::int8,
                            "NINT8-0 q");
                        validate_direct_array(
                            weight->scales(),
                            mlx::core::float16,
                            "NINT8-0 scales");
                        direct_inputs.push_back(
                            weight->quantized_values());
                        direct_inputs.push_back(
                            weight->scales());
                    } else if constexpr (
                        std::is_same_v<
                            Weight,
                            MlxVqWeight>
                    ) {
                        if (weight->output_shape().size() != 1 ||
                            weight->rotation_block() != 0) {
                            throw MlxGroupedLinearUnsupported(
                                "ordinary grouped linear rejects "
                                "expert-shaped or rotated NEPQ weights");
                        }
                        layout.family = kFamilyVq;
                        layout.group_size =
                            weight->group_size();
                        layout.groups = weight->groups();
                        layout.vector_size =
                            weight->vector_size();
                        layout.vectors = weight->vectors();
                        layout.index_bits =
                            weight->index_bits();
                        layout.state_bits =
                            weight->state_bits();
                        layout.states = weight->states();
                        layout.entries = weight->entries();
                        layout.code_banks =
                            weight->code_banks();
                        layout.aux_mode =
                            weight->aux_mode();
                        layout.code_bank_mode =
                            weight->code_bank_mode();
                        layout.table_banks =
                            weight->table_banks();
                        layout.groups_per_supergroup =
                            weight->groups_per_supergroup();
                        layout.supergroups =
                            weight->supergroups();
                        validate_direct_array(
                            weight->packed_indices(),
                            mlx::core::uint8,
                            "VQ indices");
                        validate_direct_array(
                            weight->packed_states(),
                            mlx::core::uint8,
                            "VQ states");
                        validate_direct_array(
                            weight->packed_auxiliary(),
                            mlx::core::uint8,
                            "VQ auxiliary stream");
                        validate_direct_array(
                            weight->anchors(),
                            mlx::core::float32,
                            "VQ anchors");
                        validate_direct_array(
                            weight->codebooks(),
                            mlx::core::int8,
                            "VQ codebooks");
                        validate_direct_array(
                            weight->scale_lut(),
                            mlx::core::float32,
                            "VQ scale LUT");
                        validate_direct_array(
                            weight->state_to_codebank(),
                            mlx::core::uint8,
                            "VQ state-to-bank map");
                        validate_direct_array(
                            weight->bank_ids(),
                            mlx::core::uint8,
                            "VQ table-bank selectors");
                        validate_direct_array(
                            weight->parameters(),
                            mlx::core::float32,
                            "VQ parameters");
                        direct_inputs.push_back(
                            weight->packed_indices());
                        direct_inputs.push_back(
                            weight->packed_states());
                        direct_inputs.push_back(
                            weight->packed_auxiliary());
                        direct_inputs.push_back(
                            weight->anchors());
                        direct_inputs.push_back(
                            weight->codebooks());
                        direct_inputs.push_back(
                            weight->scale_lut());
                        direct_inputs.push_back(
                            weight->state_to_codebank());
                        direct_inputs.push_back(
                            weight->bank_ids());
                        direct_inputs.push_back(
                            weight->parameters());
                    } else if constexpr (
                        std::is_same_v<
                            Weight,
                            MlxCccpInt4Weight>
                    ) {
                        layout.family =
                            kFamilyCccpInt4;
                        layout.group_size =
                            weight->group_size();
                        layout.groups =
                            weight->groups();
                        validate_direct_array(
                            weight->packed_values(),
                            mlx::core::uint8,
                            "CCCP-I4G64 packed values");
                        validate_direct_array(
                            weight->scales(),
                            mlx::core::float16,
                            "CCCP-I4G64 scales");
                        direct_inputs.push_back(
                            weight->packed_values());
                        direct_inputs.push_back(
                            weight->scales());
                    } else if constexpr (
                        std::is_same_v<Weight, MlxMxWeight>
                    ) {
                        layout.family = kFamilyMx;
                        layout.bits = weight->bits();
                        validate_direct_array(
                            weight->packed_values(),
                            mlx::core::uint8,
                            "MX packed values");
                        validate_direct_array(
                            weight->block_scales(),
                            mlx::core::uint8,
                            "MX block scales");
                        direct_inputs.push_back(
                            weight->packed_values());
                        direct_inputs.push_back(
                            weight->block_scales());
                    } else {
                        layout.family =
                            kFamilyCccpPq;
                        layout.vector_size =
                            weight->vector_size();
                        layout.blocks =
                            weight->blocks();
                        layout.index_bits =
                            weight->index_bits();
                        layout.entries =
                            weight->entries();
                        validate_direct_array(
                            weight->packed_indices(),
                            mlx::core::uint8,
                            "CCCP-PQ packed indices");
                        validate_direct_array(
                            weight->codebook(),
                            mlx::core::float16,
                            "CCCP-PQ codebook");
                        direct_inputs.push_back(
                            weight->packed_indices());
                        direct_inputs.push_back(
                            weight->codebook());
                    }
                    weight_bytes = weight->packed_nbytes();
                },
                source);
            if (weight_bytes >
                std::numeric_limits<std::size_t>::max()
                    - packed_bytes) {
                throw MlxGroupedLinearUnsupported(
                    "grouped linear packed stream size overflows");
            }
            packed_bytes += weight_bytes;
            layouts.push_back(layout);
            output_sizes.push_back(output);
        }

        // Metal exposes at most 31 buffer arguments. Account for x and y in
        // addition to the retained source arrays and reject safely before
        // constructing a custom kernel which the driver cannot compile.
        if (direct_inputs.size() + 2 > 31) {
            throw MlxGroupedLinearUnsupported(
                "grouped linear direct binding exceeds "
                "Metal's 31-buffer argument limit");
        }

        impl_ = std::make_shared<Impl>(
            std::move(layouts),
            std::move(direct_inputs),
            std::move(output_sizes),
            shared_input,
            total_output,
            total_tiles,
            packed_bytes);
        return;
    }

    if (std::any_of(
            weights.begin(),
            weights.end(),
            [](const MlxGroupedLinearWeightRef& weight) {
                return std::holds_alternative<
                           const MlxVqWeight*>(weight) ||
                    std::holds_alternative<
                        const MlxCccpInt4Weight*>(weight) ||
                    std::holds_alternative<
                        const MlxCccpPqWeight*>(weight) ||
                    std::holds_alternative<
                        const MlxMxWeight*>(weight);
            })) {
        throw MlxGroupedLinearUnsupported(
            "VQ/CCCP/MX grouped linear requires the direct-binding "
            "zero-copy path");
    }

    std::vector<std::int32_t> descriptors(
        weights.size() * kDescriptorSize,
        0);
    std::vector<std::int32_t> tile_offsets(
        weights.size() + 1,
        0);
    std::vector<std::int32_t> output_offsets(
        weights.size() + 1,
        0);
    std::vector<int> output_sizes;
    output_sizes.reserve(weights.size());

    std::vector<std::uint8_t> nint_q;
    std::vector<std::uint8_t> nint_sub_scale;
    std::vector<std::uint8_t> nint_sub_min;
    std::vector<std::uint8_t> nint_anchor_scale;
    std::vector<std::uint8_t> nint_anchor_min;
    std::vector<std::uint8_t> q8_q;
    std::vector<std::uint8_t> q8_scales;

    for (
        std::size_t projection = 0;
        projection < weights.size();
        ++projection
    ) {
        const auto& source = weights[projection];
        if (weight_input_size(source) != shared_input) {
            throw std::invalid_argument(
                "grouped linear projections must share one input width");
        }
        const int output = weight_output_size(source);
        if (output <= 0) {
            throw std::invalid_argument(
                "grouped linear output width must be positive");
        }
        output_sizes.push_back(output);

        const auto base = projection * kDescriptorSize;
        descriptors[base + kOutput] = output;
        output_offsets[projection + 1] = checked_int(
            static_cast<std::size_t>(output_offsets[projection])
                + static_cast<std::size_t>(output),
            "total output width");
        tile_offsets[projection + 1] = checked_int(
            static_cast<std::size_t>(tile_offsets[projection])
                + static_cast<std::size_t>((output + 7) / 8),
            "tile count");

        std::visit(
            [&](const auto* weight) {
                using Weight = std::remove_cv_t<
                    std::remove_pointer_t<decltype(weight)>>;
                if constexpr (
                    std::is_same_v<Weight, MlxNintWeight>
                ) {
                    descriptors[base + kFamily] = kFamilyNint;
                    descriptors[base + kNintBits] =
                        weight->bits();
                    descriptors[base + kNintGroupSize] =
                        weight->group_size();
                    descriptors[base + kNintGroups] =
                        weight->groups();
                    descriptors[base + kNintQOffset] =
                        checked_int(nint_q.size(), "NINT q offset");
                    descriptors[base + kNintSubOffset] =
                        checked_int(
                            nint_sub_scale.size(),
                            "NINT sub offset");
                    descriptors[base + kNintAnchorOffset] =
                        checked_int(
                            nint_anchor_scale.size() /
                                sizeof(float),
                            "NINT anchor offset");
                    descriptors[base + kNintQ5Execution] =
                        static_cast<int>(
                            weight->q5_execution_layout());

                    append_raw(
                        nint_q,
                        weight->packed_values(),
                        mlx::core::uint8,
                        "NINT q");
                    // Two bytes allow a safe cross-byte read at the end
                    // of every independently packed projection.
                    nint_q.insert(nint_q.end(), 2, 0);
                    append_raw(
                        nint_sub_scale,
                        weight->sub_scales(),
                        mlx::core::uint8,
                        "NINT sub scales");
                    append_raw(
                        nint_sub_min,
                        weight->sub_mins(),
                        mlx::core::uint8,
                        "NINT sub minima");
                    append_raw(
                        nint_anchor_scale,
                        weight->neuron_scales(),
                        mlx::core::float32,
                        "NINT neuron scales");
                    append_raw(
                        nint_anchor_min,
                        weight->neuron_mins(),
                        mlx::core::float32,
                        "NINT neuron minima");
                } else if constexpr (
                    std::is_same_v<
                        Weight,
                        MlxNint8ZeroWeight>
                ) {
                    descriptors[base + kFamily] =
                        kFamilyNint8Zero;
                    descriptors[base + kQ8Groups] =
                        weight->groups();
                    descriptors[base + kQ8QOffset] =
                        checked_int(q8_q.size(), "NINT8-0 q offset");
                    descriptors[base + kQ8ScaleOffset] =
                        checked_int(
                            q8_scales.size() / sizeof(std::uint16_t),
                            "NINT8-0 scale offset");
                    append_raw(
                        q8_q,
                        weight->quantized_values(),
                        mlx::core::int8,
                        "NINT8-0 q");
                    append_raw(
                        q8_scales,
                        weight->scales(),
                        mlx::core::float16,
                        "NINT8-0 scales");
                } else {
                    throw MlxGroupedLinearUnsupported(
                        "VQ grouped linear cannot use "
                        "the copied pooled fallback");
                }
            },
            source);
    }

    const auto descriptor_shape = Shape{
        checked_int(weights.size(), "projection count"),
        kDescriptorSize,
    };
    const auto offsets_shape = Shape{
        checked_int(weights.size() + 1, "offset count"),
    };
    const std::size_t packed_bytes =
        nint_q.size()
        + nint_sub_scale.size()
        + nint_sub_min.size()
        + nint_anchor_scale.size()
        + nint_anchor_min.size()
        + q8_q.size()
        + q8_scales.size();

    impl_ = std::make_shared<Impl>(
        make_int32_array(descriptors, descriptor_shape),
        make_int32_array(tile_offsets, offsets_shape),
        make_int32_array(output_offsets, offsets_shape),
        make_raw_array(std::move(nint_q), mlx::core::uint8),
        make_raw_array(
            std::move(nint_sub_scale),
            mlx::core::uint8),
        make_raw_array(
            std::move(nint_sub_min),
            mlx::core::uint8),
        make_raw_array(
            std::move(nint_anchor_scale),
            mlx::core::float32),
        make_raw_array(
            std::move(nint_anchor_min),
            mlx::core::float32),
        make_raw_array(std::move(q8_q), mlx::core::int8),
        make_raw_array(
            std::move(q8_scales),
            mlx::core::float16),
        std::move(output_sizes),
        shared_input,
        output_offsets.back(),
        tile_offsets.back(),
        packed_bytes);
}

bool MlxGroupedLinear::supports(
    const array& input) const noexcept {
    if (input.ndim() == 0 ||
        input.shape(-1) != impl_->input_size ||
        (input.dtype() != mlx::core::float16 &&
         input.dtype() != mlx::core::float32)) {
        return false;
    }
    std::size_t rows = 1;
    for (
        std::size_t dimension = 0;
        dimension + 1 < input.ndim();
        ++dimension
    ) {
        const int value =
            input.shape(static_cast<int>(dimension));
        if (value <= 0 ||
            rows > static_cast<std::size_t>(max_rows()) /
                static_cast<std::size_t>(value)) {
            return false;
        }
        rows *= static_cast<std::size_t>(value);
    }
    return rows >= 1 &&
        rows <= static_cast<std::size_t>(max_rows());
}

bool MlxGroupedLinear::supports_single_row_swiglu(
    const array& input) const noexcept {
    return impl_->supports_single_row_swiglu()
        && input.ndim() > 0
        && input.shape(-1) == impl_->input_size
        && input.dtype() == mlx::core::float16
        && input.size() == static_cast<std::size_t>(
            impl_->input_size);
}

array MlxGroupedLinear::single_row_swiglu(
    const array& input,
    float limit) const {
    if (!supports_single_row_swiglu(input)) {
        throw MlxGroupedLinearUnsupported(
            "grouped SwiGLU requires one FP16 row and two "
            "equal-width NINT or MXFP8 projections");
    }
    if (!std::isfinite(limit) || limit < 0.0f) {
        throw std::invalid_argument(
            "grouped SwiGLU limit must be finite and non-negative");
    }

    Shape prefix(
        input.shape().begin(),
        input.shape().end() - 1);
    auto output_shape = prefix;
    output_shape.push_back(impl_->output_sizes[0]);
    auto source = mlx::core::contiguous(
        mlx::core::reshape(
            input,
            Shape{1, impl_->input_size}));
    const array params({limit}, mlx::core::float32);
    const bool mxfp8_pair =
        impl_->has_single_row_mxfp8_fast_path()
        && impl_->direct_layouts[0].family == kFamilyMx
        && impl_->direct_layouts[0].bits == 8
        && impl_->direct_layouts[1].family == kFamilyMx
        && impl_->direct_layouts[1].bits == 8;
    auto inputs = impl_->direct_weight_inputs;
    inputs.push_back(source);
    inputs.push_back(params);

    if (mxfp8_pair) {
        const auto workgroups =
            (static_cast<std::size_t>(impl_->output_sizes[0]) + 3) / 4;
        const auto grid = workgroups * 128;
        auto result = single_row_mxfp8_pair_swiglu_kernel()(
            inputs,
            {Shape{1, impl_->output_sizes[0]}},
            {source.dtype()},
            {
                checked_int(grid, "MXFP8 SwiGLU Metal grid"),
                1,
                1,
            },
            {128, 1, 1},
            {
                {"K", impl_->input_size},
                {"OUT", impl_->output_sizes[0]},
            },
            std::nullopt,
            false,
            {}).front();
        return mlx::core::reshape(
            std::move(result),
            std::move(output_shape));
    }

    std::vector<
        std::pair<
            std::string,
            mlx::core::fast::TemplateArg>>
        templates{
            {"T", source.dtype()},
            {"K", impl_->input_size},
            {
                "GS",
                impl_->direct_layouts.front().group_size,
            },
            {
                "NG",
                impl_->direct_layouts.front().groups,
            },
        };
    for (
        std::size_t projection = 0;
        projection < impl_->direct_layouts.size();
        ++projection
    ) {
        const auto& layout = impl_->direct_layouts[projection];
        const auto name =
            "P" + std::to_string(projection) + "_";
        templates.emplace_back(
            name + "OUT",
            layout.output_size);
        templates.emplace_back(
            name + "OUT_OFFSET",
            layout.output_offset);
    }

    const auto grid = static_cast<std::size_t>(
        impl_->single_row_tiles) * 64;
    auto result = single_row_nint_kernel(
        impl_->direct_layouts,
        true)(
        inputs,
        {Shape{1, impl_->output_sizes[0]}},
        {source.dtype()},
        {
            checked_int(grid, "SwiGLU Metal grid"),
            1,
            1,
        },
        {64, 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        {}).front();
    return mlx::core::reshape(
        std::move(result),
        std::move(output_shape));
}

std::vector<array> MlxGroupedLinear::matmul(
    const array& input) const {
    if (input.ndim() == 0 ||
        input.shape(-1) != impl_->input_size) {
        throw std::invalid_argument(
            "grouped linear input must end in the shared weight width");
    }
    if (input.dtype() != mlx::core::float16 &&
        input.dtype() != mlx::core::float32) {
        throw MlxGroupedLinearUnsupported(
            "grouped linear supports only float16 or float32 input");
    }

    std::size_t rows = 1;
    Shape prefix(
        input.shape().begin(),
        input.shape().end() - 1);
    for (const int value : prefix) {
        if (value <= 0 ||
            rows > static_cast<std::size_t>(max_rows()) /
                static_cast<std::size_t>(value)) {
            throw MlxGroupedLinearUnsupported(
                "grouped linear supports one through 16 input rows");
        }
        rows *= static_cast<std::size_t>(value);
    }
    if (rows == 0 ||
        rows > static_cast<std::size_t>(max_rows())) {
        throw MlxGroupedLinearUnsupported(
            "grouped linear supports one through 16 input rows");
    }

    auto source = mlx::core::contiguous(
        mlx::core::reshape(
            input,
            Shape{
                static_cast<std::int32_t>(rows),
                impl_->input_size,
            }));
    const bool use_single_row_nint_fast_path =
        rows == 1 &&
        source.dtype() == mlx::core::float16 &&
        impl_->has_single_row_nint_fast_path();
    const bool use_single_row_mxfp8_fast_path =
        rows == 1 &&
        source.dtype() == mlx::core::float16 &&
        impl_->has_single_row_mxfp8_fast_path();
    const int work_tiles = use_single_row_mxfp8_fast_path
        ? impl_->single_row_mxfp8_tiles
        : use_single_row_nint_fast_path
        ? impl_->single_row_tiles
        : impl_->total_tiles;
    const auto grid = rows
        * static_cast<std::size_t>(work_tiles)
        * static_cast<std::size_t>(
            use_single_row_mxfp8_fast_path ? 128 : 64);
    if (grid >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw MlxGroupedLinearUnsupported(
            "grouped linear Metal grid exceeds MLX limits");
    }

    const auto output_shapes = std::vector<Shape>{
        Shape{
            static_cast<std::int32_t>(rows),
            impl_->total_output_size,
        },
    };
    const auto output_dtypes =
        std::vector<Dtype>{source.dtype()};
    const auto grid_shape = std::tuple<int, int, int>{
        static_cast<int>(grid),
        1,
        1,
    };
    const auto threadgroup = std::tuple<int, int, int>{
        use_single_row_mxfp8_fast_path ? 128 : 64,
        1,
        1,
    };

    array combined = [&]() {
        if (impl_->uses_zero_copy_storage()) {
            auto inputs = impl_->direct_weight_inputs;
            inputs.push_back(source);
            if (use_single_row_mxfp8_fast_path) {
                std::vector<
                    std::pair<
                        std::string,
                        mlx::core::fast::TemplateArg>>
                    templates{
                        {"K", impl_->input_size},
                    };
                for (
                    std::size_t projection = 0;
                    projection < impl_->direct_layouts.size();
                    ++projection
                ) {
                    const auto& layout =
                        impl_->direct_layouts[projection];
                    const auto prefix =
                        "P" + std::to_string(projection) + "_";
                    templates.emplace_back(
                        prefix + "OUT",
                        layout.output_size);
                    templates.emplace_back(
                        prefix + "OUT_OFFSET",
                        layout.output_offset);
                    if (layout.family == kFamilyNint8Zero) {
                        templates.emplace_back(
                            prefix + "NG",
                            layout.groups);
                    }
                }
                return single_row_mxfp8_kernel(
                    impl_->direct_layouts)(
                    inputs,
                    output_shapes,
                    output_dtypes,
                    grid_shape,
                    threadgroup,
                    std::move(templates),
                    std::nullopt,
                    false,
                    {}).front();
            }
            if (use_single_row_nint_fast_path) {
                std::vector<
                    std::pair<
                        std::string,
                        mlx::core::fast::TemplateArg>>
                    templates{
                        {"T", source.dtype()},
                        {"K", impl_->input_size},
                        {
                            "GS",
                            impl_->direct_layouts.front().group_size,
                        },
                        {
                            "NG",
                            impl_->direct_layouts.front().groups,
                        },
                    };
                for (
                    std::size_t projection = 0;
                    projection < impl_->direct_layouts.size();
                    ++projection
                ) {
                    const auto& layout =
                        impl_->direct_layouts[projection];
                    const auto prefix =
                        "P" + std::to_string(projection) + "_";
                    templates.emplace_back(
                        prefix + "OUT",
                        layout.output_size);
                    templates.emplace_back(
                        prefix + "OUT_OFFSET",
                        layout.output_offset);
                }
                return single_row_nint_kernel(
                    impl_->direct_layouts)(
                    inputs,
                    output_shapes,
                    output_dtypes,
                    grid_shape,
                    threadgroup,
                    std::move(templates),
                    std::nullopt,
                    false,
                    {}).front();
            }
            std::vector<
                std::pair<
                    std::string,
                    mlx::core::fast::TemplateArg>>
                templates{
                    {"T", source.dtype()},
                    {"ROWS", static_cast<int>(rows)},
                    {"K", impl_->input_size},
                    {"TOTAL_OUT", impl_->total_output_size},
                    {"TOTAL_TILES", impl_->total_tiles},
                };
            for (
                std::size_t projection = 0;
                projection < impl_->direct_layouts.size();
                ++projection
            ) {
                const auto& layout =
                    impl_->direct_layouts[projection];
                const auto prefix =
                    "P" + std::to_string(projection) + "_";
                templates.emplace_back(
                    prefix + "OUT",
                    layout.output_size);
                templates.emplace_back(
                    prefix + "TILE_BEGIN",
                    layout.tile_begin);
                templates.emplace_back(
                    prefix + "TILE_END",
                    layout.tile_end);
                templates.emplace_back(
                    prefix + "OUT_OFFSET",
                    layout.output_offset);
                templates.emplace_back(
                    prefix + "NG",
                    layout.groups);
                if (layout.family == kFamilyNint) {
                    templates.emplace_back(
                        prefix + "BITS",
                        layout.bits);
                    templates.emplace_back(
                        prefix + "GS",
                        layout.group_size);
                } else if (layout.family == kFamilyVq) {
                    templates.emplace_back(
                        prefix + "GS",
                        layout.group_size);
                    templates.emplace_back(
                        prefix + "VECTOR_SIZE",
                        layout.vector_size);
                    templates.emplace_back(
                        prefix + "NVEC",
                        layout.vectors);
                    templates.emplace_back(
                        prefix + "INDEX_BITS",
                        layout.index_bits);
                    templates.emplace_back(
                        prefix + "STATE_BITS",
                        layout.state_bits);
                    templates.emplace_back(
                        prefix + "STATES",
                        layout.states);
                    templates.emplace_back(
                        prefix + "ENTRIES",
                        layout.entries);
                    templates.emplace_back(
                        prefix + "CODE_BANKS",
                        layout.code_banks);
                    templates.emplace_back(
                        prefix + "AUX_MODE",
                        layout.aux_mode);
                    templates.emplace_back(
                        prefix + "CODE_BANK_MODE",
                        layout.code_bank_mode);
                    templates.emplace_back(
                        prefix + "HAS_TABLE_BANKS",
                        static_cast<int>(
                            layout.table_banks > 1));
                    templates.emplace_back(
                        prefix + "GROUPS_PER_SUPER",
                        layout.groups_per_supergroup);
                    templates.emplace_back(
                        prefix + "NSUPER",
                        layout.supergroups);
                } else if (
                    layout.family == kFamilyCccpInt4
                ) {
                    templates.emplace_back(
                        prefix + "GS",
                        layout.group_size);
                } else if (
                    layout.family == kFamilyCccpPq
                ) {
                    templates.emplace_back(
                        prefix + "VECTOR_SIZE",
                        layout.vector_size);
                    templates.emplace_back(
                        prefix + "BLOCKS",
                        layout.blocks);
                    templates.emplace_back(
                        prefix + "INDEX_BITS",
                        layout.index_bits);
                } else if (layout.family == kFamilyMx) {
                    templates.emplace_back(
                        prefix + "BITS",
                        layout.bits);
                }
            }
            return direct_kernel(impl_->direct_layouts)(
                inputs,
                output_shapes,
                output_dtypes,
                grid_shape,
                threadgroup,
                std::move(templates),
                std::nullopt,
                false,
                {}).front();
        }

        return grouped_kernel()(
            {
                *impl_->descriptors,
                *impl_->projection_tile_offsets,
                *impl_->projection_output_offsets,
                *impl_->nint_q,
                *impl_->nint_sub_scale,
                *impl_->nint_sub_min,
                *impl_->nint_anchor_scale,
                *impl_->nint_anchor_min,
                *impl_->q8_q,
                *impl_->q8_scales,
                source,
            },
            output_shapes,
            output_dtypes,
            grid_shape,
            threadgroup,
            {
                {"T", source.dtype()},
                {"ROWS", static_cast<int>(rows)},
                {
                    "PROJECTIONS",
                    static_cast<int>(
                        impl_->output_sizes.size()),
                },
                {"K", impl_->input_size},
                {"TOTAL_OUT", impl_->total_output_size},
                {"TOTAL_TILES", impl_->total_tiles},
                {"DESCRIPTOR_SIZE", kDescriptorSize},
            },
            std::nullopt,
            false,
            {}).front();
    }();

    std::vector<array> outputs;
    outputs.reserve(impl_->output_sizes.size());
    int offset = 0;
    for (const int width : impl_->output_sizes) {
        auto shape = prefix;
        shape.push_back(width);
        outputs.push_back(
            mlx::core::reshape(
                mlx::core::slice(
                    combined,
                    Shape{0, offset},
                    Shape{
                        static_cast<std::int32_t>(rows),
                        offset + width,
                    }),
                std::move(shape)));
        offset += width;
    }
    return outputs;
}

int MlxGroupedLinear::input_size() const noexcept {
    return impl_->input_size;
}

int MlxGroupedLinear::total_output_size() const noexcept {
    return impl_->total_output_size;
}

std::size_t MlxGroupedLinear::projection_count() const noexcept {
    return impl_->output_sizes.size();
}

const std::vector<int>&
MlxGroupedLinear::output_sizes() const noexcept {
    return impl_->output_sizes;
}

std::size_t MlxGroupedLinear::packed_nbytes() const noexcept {
    return impl_->packed_bytes;
}

bool MlxGroupedLinear::uses_zero_copy_storage() const noexcept {
    return impl_->uses_zero_copy_storage();
}

std::size_t
MlxGroupedLinear::copied_packed_nbytes() const noexcept {
    return impl_->copied_packed_bytes;
}

bool MlxGroupedLinear::has_single_row_nint_fast_path()
    const noexcept {
    return impl_->has_single_row_nint_fast_path();
}

bool MlxGroupedLinear::has_single_row_mxfp8_fast_path()
    const noexcept {
    return impl_->has_single_row_mxfp8_fast_path();
}

} // namespace mfq::metal
