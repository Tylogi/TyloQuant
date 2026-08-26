#include "mlx_mxfp4_sq2.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::array;
using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;

constexpr std::array<std::uint8_t, 4> kMagic{'S', 'Q', '2', '1'};
constexpr std::uint8_t kVersion = 1;

class Cursor {
public:
  explicit Cursor(std::span<const std::uint8_t> blob) : blob_(blob) {}

  template <typename T> T scalar(const char *name) {
    if (sizeof(T) > remaining()) {
      throw std::runtime_error(std::string("truncated MXFP4-SQ2 ") + name);
    }
    T value{};
    std::memcpy(&value, blob_.data() + offset_, sizeof(T));
    offset_ += sizeof(T);
    return value;
  }

  std::span<const std::uint8_t> bytes(std::size_t count, const char *name) {
    if (count > remaining()) {
      throw std::runtime_error(std::string("truncated MXFP4-SQ2 ") + name);
    }
    auto value = blob_.subspan(offset_, count);
    offset_ += count;
    return value;
  }

  std::size_t remaining() const noexcept { return blob_.size() - offset_; }

private:
  std::span<const std::uint8_t> blob_;
  std::size_t offset_ = 0;
};

std::size_t checked_product(std::uint64_t left, std::uint64_t right,
                            const char *name) {
  if (left == 0 || right == 0 ||
      left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::runtime_error(std::string("invalid MXFP4-SQ2 ") + name);
  }
  return static_cast<std::size_t>(left * right);
}

int checked_dimension(std::uint64_t value, const char *name) {
  if (value == 0 || value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid MXFP4-SQ2 ") + name);
  }
  return static_cast<int>(value);
}

template <typename T> array make_array(std::span<const T> values, Shape shape) {
  return array(values.begin(), std::move(shape));
}

constexpr const char *kSq2Header = R"METAL(
#include <metal_simdgroup_matrix>

inline uint mfq_sq2_read_symbol(
    device const uchar* symbols,
    uint block_index,
    uint lane
) {
    uchar packed = symbols[block_index * 8u + (lane >> 2u)];
    return (uint(packed) >> ((lane & 3u) * 2u)) & 3u;
}

inline float mfq_sq2_e8m0(uchar raw) {
    uint bits = raw == 0u ? 0x00400000u : uint(raw) << 23u;
    return as_type<float>(bits);
}

constant float mfq_sq2_palette_values[64] = {
    -6.0f, -3.0f, 0.0f, 3.0f,
    -6.0f, -3.0f, 0.5f, 4.0f,
    -6.0f, -2.0f, 1.0f, 4.0f,
    -4.0f, -1.5f, 0.0f, 1.5f,
    -4.0f, -1.5f, 0.5f, 3.0f,
    -4.0f, -1.5f, 1.0f, 4.0f,
    -4.0f, -1.0f, 0.5f, 2.0f,
    -4.0f, -1.0f, 0.5f, 3.0f,
    -4.0f, -1.0f, 1.5f, 4.0f,
    -4.0f, -1.0f, 2.0f, 6.0f,
    -4.0f, -0.5f, 3.0f, 6.0f,
    -3.0f, -1.0f, 0.5f, 2.0f,
    -3.0f, -0.5f, 1.5f, 4.0f,
    -3.0f, 0.0f, 3.0f, 6.0f,
    -2.0f, -0.5f, 1.0f, 3.0f,
    -2.0f, -0.5f, 1.0f, 4.0f,
};

template <typename SelectorStream>
inline uint mfq_sq2_scalar_block_tag(
    device const uchar* symbols,
    SelectorStream selectors,
    uint block_index
) {
    device const uint* words =
        (device const uint*)(symbols + block_index * 8u);
    constexpr uint LOW_BITS = 0x55555555u;
    uint low_count =
        popcount(words[0] & LOW_BITS) + popcount(words[1] & LOW_BITS);
    uint explicit_high =
        (uint(selectors[block_index >> 3u])
            >> (block_index & 7u)) & 1u;
    return (low_count & 1u) | (explicit_high << 1u);
}

template <typename PaletteStream>
inline uint mfq_sq2_read_palette(
    PaletteStream state_palettes,
    uint state_index
) {
    return (uint(state_palettes[state_index >> 1u])
        >> ((state_index & 1u) * 4u)) & 15u;
}
)METAL";

// Multi-row packed decode-dot.  Eight lanes own one output row and complete
// native 32-value blocks, so each block tag/state is decoded once and every
// decoded scalar is reused across TILE_M activation rows.  Two SIMD groups
// produce eight output rows per threadgroup.
constexpr const char *kSq2Mmq = R"METAL(
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint K_LANES = 8u;
    constexpr uint OUTPUTS_PER_SIMD = 32u / K_LANES;
    constexpr uint OUTPUTS_PER_TG = SIMD_GROUPS * OUTPUTS_PER_SIMD;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_output = lane / K_LANES;
    uint output_index =
        threadgroup_position_in_grid.y * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD + simd_output;
    uint output = min(output_index, uint(OUT) - 1u);
    uint first_row = threadgroup_position_in_grid.x * uint(TILE_M);

    float accumulators[TILE_M];
    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        accumulators[local_row] = 0.0f;
    }

    for (uint block = k_lane; block < uint(BLOCKS); block += K_LANES) {
        uint block_index = output * uint(BLOCKS) + block;
        uint tag = mfq_sq2_scalar_block_tag(
            symbols, selectors, block_index);
        uint state_index = output * 4u + tag;
        float scale = mfq_sq2_e8m0(state_scales[state_index]);
        uint palette = mfq_sq2_read_palette(
            state_palettes, state_index);
        uint column_base = block * 32u;
        for (uint packed_index = 0u; packed_index < 8u; ++packed_index) {
            uint column = column_base + packed_index * 4u;
            uint packed = uint(symbols[block_index * 8u + packed_index]);
            float4 weight = float4(
                mfq_sq2_palette_values[
                    palette * 4u + (packed & 3u)],
                mfq_sq2_palette_values[
                    palette * 4u + ((packed >> 2u) & 3u)],
                mfq_sq2_palette_values[
                    palette * 4u + ((packed >> 4u) & 3u)],
                mfq_sq2_palette_values[
                    palette * 4u + (packed >> 6u)]) * scale;
            for (uint local_row = 0u;
                 local_row < uint(TILE_M);
                 ++local_row) {
                uint row = min(first_row + local_row, uint(M) - 1u);
                half4 activation = *(device const half4*)(
                    x + row * uint(K) + column);
                accumulators[local_row] += dot(
                    float4(activation), weight);
            }
        }
    }

    for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
        accumulators[local_row] += simd_shuffle_down(
            accumulators[local_row], 4u);
        accumulators[local_row] += simd_shuffle_down(
            accumulators[local_row], 2u);
        accumulators[local_row] += simd_shuffle_down(
            accumulators[local_row], 1u);
        uint row = first_row + local_row;
        if (k_lane == 0u && output_index < uint(OUT) && row < uint(M)) {
            y[row * uint(OUT) + output_index] = T(accumulators[local_row]);
        }
    }
)METAL";

// FP16 tiled QMM for prefill-sized M.  A threadgroup decodes one packed
// BN-by-BK weight tile into threadgroup FP16, stages the matching activation
// tile, and accumulates with the Apple SIMD matrix unit.  No full dense weight
// tensor is materialized.  BM/BN/BK select the larger-M schedules.
constexpr const char *kSq2Mma = R"METAL(
    constexpr uint BLOCKS_PER_K_TILE = uint(BK) / 32u;
    constexpr uint M_SUBTILES = uint(BM) / 8u;
    constexpr uint N_SUBTILES = uint(BN) / 8u;
    constexpr uint PRODUCTS = M_SUBTILES * N_SUBTILES;

    threadgroup half activation_tile[uint(BM) * uint(BK)];
    threadgroup half weight_tile[uint(BN) * uint(BK)];
    threadgroup float store_tiles[uint(SIMD_GROUPS) * 64u];
    threadgroup float cached_scales[uint(BN) * BLOCKS_PER_K_TILE];
    threadgroup uchar cached_palettes[uint(BN) * BLOCKS_PER_K_TILE];

    uint local_thread = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint first_output = threadgroup_position_in_grid.x * uint(BN);
    uint first_row = threadgroup_position_in_grid.y * uint(BM);

    simdgroup_float8x8 accumulators[ACCUMULATORS];
    for (uint index = 0u; index < uint(ACCUMULATORS); ++index) {
        accumulators[index] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint column_base = 0u;
         column_base < uint(K);
         column_base += uint(BK)) {
        uint first_block = column_base / 32u;
        for (uint metadata = local_thread;
             metadata < uint(BN) * BLOCKS_PER_K_TILE;
             metadata += uint(THREADS)) {
            uint local_output = metadata / BLOCKS_PER_K_TILE;
            uint local_block = metadata - local_output * BLOCKS_PER_K_TILE;
            uint output = min(first_output + local_output, uint(OUT) - 1u);
            uint block = min(
                first_block + local_block, uint(BLOCKS) - 1u);
            uint block_index = output * uint(BLOCKS) + block;
            uint tag = mfq_sq2_scalar_block_tag(
                symbols, selectors, block_index);
            uint state_index = output * 4u + tag;
            cached_scales[metadata] =
                mfq_sq2_e8m0(state_scales[state_index]);
            cached_palettes[metadata] = uchar(
                mfq_sq2_read_palette(state_palettes, state_index));
        }
        for (uint vector_index = local_thread;
             vector_index < uint(BM) * (uint(BK) / 4u);
             vector_index += uint(THREADS)) {
            uint local_row = vector_index / (uint(BK) / 4u);
            uint local_vector =
                vector_index - local_row * (uint(BK) / 4u);
            uint local_column = local_vector * 4u;
            uint row = first_row + local_row;
            uint column = column_base + local_column;
            half4 activation = row < uint(M) && column < uint(K)
                ? *(device const half4*)(x + row * uint(K) + column)
                : half4(0.0h);
            *(threadgroup half4*)(
                activation_tile + local_row * uint(BK) + local_column) =
                    activation;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint vector_index = local_thread;
             vector_index < uint(BN) * (uint(BK) / 4u);
             vector_index += uint(THREADS)) {
            uint local_output = vector_index / (uint(BK) / 4u);
            uint local_vector =
                vector_index - local_output * (uint(BK) / 4u);
            uint local_column = local_vector * 4u;
            uint output = min(first_output + local_output, uint(OUT) - 1u);
            uint column = column_base + local_column;
            if (column < uint(K)) {
                uint local_block = local_column / 32u;
                uint block = column / 32u;
                uint block_index = output * uint(BLOCKS) + block;
                uint packed = uint(symbols[
                    block_index * 8u + ((column & 31u) >> 2u)]);
                uint metadata =
                    local_output * BLOCKS_PER_K_TILE + local_block;
                uint palette = uint(cached_palettes[metadata]) * 4u;
                float scale = cached_scales[metadata];
                half4 weight = half4(float4(
                    mfq_sq2_palette_values[palette + (packed & 3u)],
                    mfq_sq2_palette_values[
                        palette + ((packed >> 2u) & 3u)],
                    mfq_sq2_palette_values[
                        palette + ((packed >> 4u) & 3u)],
                    mfq_sq2_palette_values[palette + (packed >> 6u)]) * scale);
                *(threadgroup half4*)(
                    weight_tile + local_output * uint(BK) + local_column) =
                        weight;
            } else {
                *(threadgroup half4*)(
                    weight_tile + local_output * uint(BK) + local_column) =
                        half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint accumulator = 0u;
             accumulator < uint(ACCUMULATORS);
             ++accumulator) {
            uint product = simd_group
                + accumulator * uint(SIMD_GROUPS);
            uint m_subtile = product / N_SUBTILES;
            uint n_subtile = product - m_subtile * N_SUBTILES;
            for (uint k_subtile = 0u;
                 k_subtile < uint(BK);
                 k_subtile += 8u) {
                simdgroup_half8x8 activation_matrix;
                simdgroup_half8x8 weight_matrix;
                simdgroup_load(
                    activation_matrix,
                    activation_tile
                        + m_subtile * 8u * uint(BK) + k_subtile,
                    uint(BK), 0u, false);
                simdgroup_load(
                    weight_matrix,
                    weight_tile
                        + n_subtile * 8u * uint(BK) + k_subtile,
                    uint(BK), 0u, true);
                simdgroup_multiply_accumulate(
                    accumulators[accumulator],
                    activation_matrix,
                    weight_matrix,
                    accumulators[accumulator]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint accumulator = 0u;
         accumulator < uint(ACCUMULATORS);
         ++accumulator) {
        uint product = simd_group + accumulator * uint(SIMD_GROUPS);
        uint m_subtile = product / N_SUBTILES;
        uint n_subtile = product - m_subtile * N_SUBTILES;
        simdgroup_store(
            accumulators[accumulator],
            store_tiles + simd_group * 64u,
            8u, 0u, false);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = lane; index < 64u; index += 32u) {
            uint matrix_row = index >> 3u;
            uint matrix_output = index & 7u;
            uint row = first_row + m_subtile * 8u + matrix_row;
            uint output =
                first_output + n_subtile * 8u + matrix_output;
            if (row < uint(M) && output < uint(OUT)) {
                y[row * uint(OUT) + output] = T(
                    store_tiles[simd_group * 64u + index]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
)METAL";

// One SIMD group owns one native 32-value block. The lane-local two-bit
// symbol supplies the decoded value while lane zero reconstructs the state.
constexpr const char *kSq2Dequantize = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint block_index = thread_position_in_grid.x >> 5u;
    uint output = block_index / uint(BLOCKS);
    uint block = block_index - output * uint(BLOCKS);
    uint column = block * 32u + lane;
    uint symbol_index = output * uint(K) + column;
    uint symbol = mfq_sq2_read_symbol(symbols, block_index, lane);
    uint tag = lane == 0u
        ? mfq_sq2_scalar_block_tag(symbols, selectors, block_index)
        : 0u;
    tag = simd_broadcast_first(tag);

    uint state_index = output * 4u + tag;
    uint scale_raw = lane == 0u ? uint(state_scales[state_index]) : 0u;
    uint palette = lane == 0u
        ? mfq_sq2_read_palette(state_palettes, state_index)
        : 0u;
    scale_raw = simd_broadcast_first(scale_raw);
    palette = simd_broadcast_first(palette);
    y[symbol_index] = T(
        mfq_sq2_palette_values[palette * 4u + symbol]
        * mfq_sq2_e8m0(uchar(scale_raw)));
)METAL";

// Two independent SIMD groups produce eight output rows per threadgroup.
// Each group loads its own four row states and computes its four rows of block
// tags in parallel before entering the dot loop.
constexpr const char *kSq2Gemv = R"METAL(
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint OUTPUTS_PER_SIMD = 4u;
    constexpr uint OUTPUTS_PER_TG = SIMD_GROUPS * OUTPUTS_PER_SIMD;
    constexpr uint STATES_PER_ROW = 4u;

    threadgroup float cached_scales[OUTPUTS_PER_TG * STATES_PER_ROW];
    threadgroup uchar cached_palettes[OUTPUTS_PER_TG * STATES_PER_ROW];
    threadgroup uchar cached_tags[OUTPUTS_PER_TG * uint(BLOCKS)];

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint first_output_slot = simd_group * OUTPUTS_PER_SIMD;

    if (lane < OUTPUTS_PER_SIMD * STATES_PER_ROW) {
        uint local = lane >> 2u;
        uint state = lane & 3u;
        uint slot = first_output_slot + local;
        uint row_index =
            threadgroup_position_in_grid.x * OUTPUTS_PER_TG + slot;
        uint row = min(row_index, uint(OUT) - 1u);
        uint state_index = row * STATES_PER_ROW + state;
        uint cached_state = slot * STATES_PER_ROW + state;
        cached_scales[cached_state] =
            mfq_sq2_e8m0(state_scales[state_index]);
        cached_palettes[cached_state] = uchar(
            mfq_sq2_read_palette(state_palettes, state_index));
    }

    for (uint local = 0u; local < OUTPUTS_PER_SIMD; ++local) {
        uint slot = first_output_slot + local;
        uint row_index =
            threadgroup_position_in_grid.x * OUTPUTS_PER_TG + slot;
        uint row = min(row_index, uint(OUT) - 1u);
        for (uint block = lane; block < uint(BLOCKS); block += 32u) {
            uint block_index = row * uint(BLOCKS) + block;
            cached_tags[slot * uint(BLOCKS) + block] = uchar(
                mfq_sq2_scalar_block_tag(
                    symbols, selectors, block_index));
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float accumulators[OUTPUTS_PER_SIMD] = {0.0f};
    for (uint block = 0u; block < uint(BLOCKS); ++block) {
        uint column = block * 32u + lane;
        float activation = float(x[column]);
        for (uint local = 0u; local < OUTPUTS_PER_SIMD; ++local) {
            uint slot = first_output_slot + local;
            uint row_index =
                threadgroup_position_in_grid.x * OUTPUTS_PER_TG + slot;
            uint row = min(row_index, uint(OUT) - 1u);
            uint block_index = row * uint(BLOCKS) + block;
            uint symbol = mfq_sq2_read_symbol(
                symbols, block_index, lane);
            uint tag = uint(cached_tags[
                slot * uint(BLOCKS) + block]);
            uint cached_state = slot * STATES_PER_ROW + tag;
            float weight = mfq_sq2_palette_values[
                uint(cached_palettes[cached_state]) * 4u + symbol]
                * cached_scales[cached_state];
            accumulators[local] = fma(
                activation, weight, accumulators[local]);
        }
    }

    for (uint local = 0u; local < OUTPUTS_PER_SIMD; ++local) {
        uint slot = first_output_slot + local;
        uint row_index =
            threadgroup_position_in_grid.x * OUTPUTS_PER_TG + slot;
        float total = simd_sum(accumulators[local]);
        if (lane == 0u && row_index < uint(OUT)) {
            y[row_index] = T(total);
        }
    }
)METAL";

const mlx::core::fast::CustomKernelFunction &sq2_dequantize_kernel() {
  static const auto kernel = [] {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_mxfp4_sq2_dequantize",
        {"symbols", "selectors", "state_scales", "state_palettes"}, {"y"},
        kSq2Dequantize, kSq2Header, true, false, options);
  }();
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &sq2_gemv_kernel() {
  static const auto kernel = [] {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_mxfp4_sq2_gemv",
        {"symbols", "selectors", "state_scales", "state_palettes", "x"}, {"y"},
        kSq2Gemv, kSq2Header, true, false, options);
  }();
  return kernel;
}

mlx::core::fast::CustomKernelFunction make_sq2_matmul_kernel(
    std::string name, const char *source) {
  CompileOptions options;
  options.math_mode = MathMode::Fast;
  return mlx::core::fast::metal_kernel(
      std::move(name),
      {"symbols", "selectors", "state_scales", "state_palettes", "x"},
      {"y"}, source, kSq2Header, true, false, options);
}

const mlx::core::fast::CustomKernelFunction &sq2_mmq_2_6_kernel() {
  static const auto kernel = make_sq2_matmul_kernel(
      "mfq_cpp_mxfp4_sq2_mmq_m2_6", kSq2Mmq);
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &sq2_mmq_7_16_kernel() {
  static const auto kernel = make_sq2_matmul_kernel(
      "mfq_cpp_mxfp4_sq2_mmq_m7_16", kSq2Mmq);
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &sq2_mma_17_32_kernel() {
  static const auto kernel = make_sq2_matmul_kernel(
      "mfq_cpp_mxfp4_sq2_mma_m17_32", kSq2Mma);
  return kernel;
}

const mlx::core::fast::CustomKernelFunction &sq2_mma_33_64_kernel() {
  static const auto kernel = make_sq2_matmul_kernel(
      "mfq_cpp_mxfp4_sq2_mma_m33_64", kSq2Mma);
  return kernel;
}

std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
templates(Dtype dtype, int input_size, int output_size) {
  return {
      {"T", dtype},
      {"K", input_size},
      {"OUT", output_size},
      {"BLOCKS", input_size / 32},
  };
}

std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
mmq_templates(Dtype dtype, int input_size, int output_size, int rows,
              int tile_rows) {
  auto values = templates(dtype, input_size, output_size);
  values.emplace_back("M", rows);
  values.emplace_back("TILE_M", tile_rows);
  return values;
}

std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
mma_templates(Dtype dtype, int input_size, int output_size, int rows, int bm,
              int bn, int bk, int threads) {
  auto values = templates(dtype, input_size, output_size);
  values.emplace_back("M", rows);
  values.emplace_back("BM", bm);
  values.emplace_back("BN", bn);
  values.emplace_back("BK", bk);
  values.emplace_back("THREADS", threads);
  values.emplace_back("SIMD_GROUPS", threads / 32);
  values.emplace_back("ACCUMULATORS", (bm / 8) * (bn / 8) / (threads / 32));
  return values;
}

} // namespace

MlxMxfp4Sq2Weight::MlxMxfp4Sq2Weight(array symbols, array block_selectors,
                                     array state_scales, array state_palettes,
                                     int input_size, int output_size)
    : symbols_(std::move(symbols)),
      block_selectors_(std::move(block_selectors)),
      state_scales_(std::move(state_scales)),
      state_palettes_(std::move(state_palettes)), input_size_(input_size),
      output_size_(output_size) {}

MlxMxfp4Sq2Weight
MlxMxfp4Sq2Weight::from_blob(const std::vector<std::uint8_t> &blob) {
  return from_blob(std::span<const std::uint8_t>(blob));
}

MlxMxfp4Sq2Weight
MlxMxfp4Sq2Weight::from_blob(std::span<const std::uint8_t> blob) {
  Cursor cursor(blob);
  for (const auto expected : kMagic) {
    if (cursor.scalar<std::uint8_t>("magic") != expected) {
      throw std::runtime_error("invalid MXFP4-SQ2 magic");
    }
  }
  const auto version = cursor.scalar<std::uint8_t>("version");
  const auto reserved_byte = cursor.scalar<std::uint8_t>("reserved byte");
  const auto reserved = cursor.scalar<std::uint16_t>("reserved");
  const auto rows = cursor.scalar<std::uint64_t>("rows");
  const auto columns = cursor.scalar<std::uint64_t>("columns");
  if (version != kVersion || reserved_byte != 0 || reserved != 0 ||
      columns % 32 != 0) {
    throw std::runtime_error("invalid MXFP4-SQ2 header or geometry");
  }

  const auto weights = checked_product(rows, columns, "weight count");
  const auto blocks = weights / 32;
  const auto symbol_nbytes = weights / 4;
  const auto selector_nbytes = (blocks + 7) / 8;
  const auto state_scale_nbytes = checked_product(rows, 4, "state scales");
  const auto state_palette_nbytes = checked_product(rows, 2, "state palettes");

  auto symbols = cursor.bytes(symbol_nbytes, "symbols");
  auto selectors = cursor.bytes(selector_nbytes, "selectors");
  auto state_scales = cursor.bytes(state_scale_nbytes, "state scales");
  auto state_palettes = cursor.bytes(state_palette_nbytes, "state palettes");
  if (cursor.remaining() != 0) {
    throw std::runtime_error("trailing bytes in MXFP4-SQ2 tensor");
  }

  return MlxMxfp4Sq2Weight(
      make_array(symbols,
                 Shape{checked_dimension(symbol_nbytes, "symbol bytes")}),
      make_array(selectors,
                 Shape{checked_dimension(selector_nbytes, "selector bytes")}),
      make_array(state_scales, Shape{checked_dimension(state_scale_nbytes,
                                                       "state-scale bytes")}),
      make_array(state_palettes,
                 Shape{checked_dimension(state_palette_nbytes,
                                         "state-palette bytes")}),
      checked_dimension(columns, "input size"),
      checked_dimension(rows, "output size"));
}

array MlxMxfp4Sq2Weight::dequantize(Dtype dtype) const {
  if (dtype != mlx::core::float16 && dtype != mlx::core::float32) {
    throw std::runtime_error(
        "MXFP4-SQ2 dequantization requires float16 or float32");
  }
  const auto blocks = checked_product(
      static_cast<std::uint64_t>(output_size_),
      static_cast<std::uint64_t>(input_size_ / 32), "dequantization blocks");
  const auto grid = checked_product(blocks, 32, "dequantization grid");
  if (grid > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "MXFP4-SQ2 dequantization grid exceeds MLX limits");
  }
  auto outputs = sq2_dequantize_kernel()(
      {symbols_, block_selectors_, state_scales_, state_palettes_},
      {Shape{output_size_, input_size_}}, {dtype},
      {static_cast<int>(grid), 1, 1}, {32, 1, 1},
      templates(dtype, input_size_, output_size_), std::nullopt, false, {});
  return std::move(outputs.front());
}

array MlxMxfp4Sq2Weight::matmul(const array &input) const {
  if (input.ndim() == 0 || input.shape(-1) != input_size_) {
    throw std::runtime_error(
        "MXFP4-SQ2 input width does not match packed weight");
  }
  const auto rows = input.size() / static_cast<std::size_t>(input_size_);
  if (rows == 0 ||
      rows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("unsupported MXFP4-SQ2 input row count");
  }
  Shape output_shape = input.shape();
  output_shape.back() = output_size_;
  auto source = input;
  if (source.dtype() != mlx::core::float16 &&
      source.dtype() != mlx::core::float32) {
    source = mlx::core::astype(source, mlx::core::float16);
  }
  source =
      mlx::core::reshape(source, Shape{static_cast<int>(rows), input_size_});
  if (rows == 1) {
    const auto grid = static_cast<std::size_t>((output_size_ + 7) / 8) * 64;
    if (grid > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("MXFP4-SQ2 GEMV grid exceeds MLX limits");
    }
    auto outputs = sq2_gemv_kernel()(
        {symbols_, block_selectors_, state_scales_, state_palettes_, source},
        {Shape{1, output_size_}}, {source.dtype()},
        {static_cast<int>(grid), 1, 1}, {64, 1, 1},
        templates(source.dtype(), input_size_, output_size_), std::nullopt,
        false, {});
    return mlx::core::reshape(std::move(outputs.front()),
                              std::move(output_shape));
  }

  // The vectorized MMQ/MMA loads are FP16-specialized.  Preserve FP32 through
  // MLX, and use the measured-faster dense GEMM route beyond M=64.  The four
  // M=2..64 FP16 buckets stay packed.
  if (rows > 64 || source.dtype() == mlx::core::float32) {
    auto dense = dequantize(source.dtype());
    auto result = mlx::core::matmul(source, mlx::core::transpose(dense));
    return mlx::core::reshape(std::move(result), std::move(output_shape));
  }

  const mlx::core::fast::CustomKernelFunction *kernel = nullptr;
  std::tuple<int, int, int> grid;
  std::tuple<int, int, int> threadgroup;
  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> arguments;
  if (rows <= 16) {
    const bool first_bucket = rows <= 6;
    const int row_tiles = 1;
    const int tile_rows =
        (static_cast<int>(rows) + row_tiles - 1) / row_tiles;
    const auto grid_x = static_cast<std::size_t>(row_tiles) * 64;
    const auto grid_y = static_cast<std::size_t>((output_size_ + 7) / 8);
    if (grid_x > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        grid_y > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("MXFP4-SQ2 MMQ grid exceeds MLX limits");
    }
    kernel = first_bucket ? &sq2_mmq_2_6_kernel()
                          : &sq2_mmq_7_16_kernel();
    grid = {static_cast<int>(grid_x), static_cast<int>(grid_y), 1};
    threadgroup = {64, 1, 1};
    arguments = mmq_templates(source.dtype(), input_size_, output_size_,
                              static_cast<int>(rows), tile_rows);
  } else {
    int bm = 32;
    int bn = 32;
    int bk = 32;
    int threads = 128;
    if (rows <= 32) {
      kernel = &sq2_mma_17_32_kernel();
    } else {
      kernel = &sq2_mma_33_64_kernel();
      bm = 64;
      bn = 32;
    }
    const auto grid_x =
        static_cast<std::size_t>((output_size_ + bn - 1) / bn) * threads;
    const auto grid_y = (rows + static_cast<std::size_t>(bm) - 1) /
        static_cast<std::size_t>(bm);
    if (grid_x > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        grid_y > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("MXFP4-SQ2 MMA grid exceeds MLX limits");
    }
    grid = {static_cast<int>(grid_x), static_cast<int>(grid_y), 1};
    threadgroup = {threads, 1, 1};
    arguments = mma_templates(source.dtype(), input_size_, output_size_,
                              static_cast<int>(rows), bm, bn, bk, threads);
  }

  auto outputs = (*kernel)(
      {symbols_, block_selectors_, state_scales_, state_palettes_, source},
      {Shape{static_cast<int>(rows), output_size_}}, {source.dtype()}, grid,
      threadgroup, std::move(arguments), std::nullopt, false, {});
  return mlx::core::reshape(std::move(outputs.front()),
                            std::move(output_shape));
}

std::size_t MlxMxfp4Sq2Weight::packed_nbytes() const noexcept {
  return symbols_.nbytes() + block_selectors_.nbytes() +
         state_scales_.nbytes() + state_palettes_.nbytes();
}

} // namespace mfq::metal
