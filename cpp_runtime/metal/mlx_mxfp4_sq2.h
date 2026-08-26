#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// Frozen format-level scalar table selected by the expert-0 fixed16 screen.
// Every entry is an ordinary E2M1 nibble; states store only a four-bit code
// into this table, not a vector codebook.
inline constexpr std::array<std::uint8_t, 64> kMxfp4Sq2PaletteNibbles{
    15, 13, 0, 5, 15, 13, 1, 6, 15, 12, 2, 6, 14, 11, 0, 3,
    14, 11, 1, 5, 14, 11, 2, 6, 14, 10, 1, 4, 14, 10, 1, 5,
    14, 10, 3, 6, 14, 10, 4, 7, 14, 9,  5, 7, 13, 10, 1, 4,
    13, 9,  3, 6, 13, 0,  5, 7, 12, 9,  2, 5, 12, 9,  2, 6,
};

// Frozen format-level fixed32 catalog for the eight-state SQ2 layout.  Each
// five-bit state code selects four ordinary E2M1 nibbles; the table is not
// stored per tensor and the decoded values remain native block-32 MXFP4.
inline constexpr std::array<std::uint8_t, 128> kMxfp4Sq2EightPaletteNibbles{
    15, 13, 0,  5,  15, 13, 1,  6,  15, 12, 2,  6,  14, 11, 0,  3,  14, 11, 1,
    5,  14, 11, 2,  6,  14, 10, 1,  4,  14, 10, 1,  5,  14, 10, 3,  6,  14, 10,
    4,  7,  14, 9,  5,  7,  13, 10, 1,  4,  13, 9,  3,  6,  13, 0,  5,  7,  12,
    9,  2,  5,  12, 9,  2,  6,  15, 12, 3,  7,  11, 0,  3,  6,  12, 9,  3,  6,
    13, 9,  2,  6,  14, 10, 3,  7,  15, 11, 4,  7,  15, 13, 9,  4,  15, 11, 1,
    5,  15, 11, 2,  6,  15, 12, 1,  6,  15, 12, 2,  7,  15, 13, 0,  6,  14, 11,
    0,  4,  13, 1,  5,  7,  15, 14, 13, 12, 15, 14, 13, 11,
};

class MlxMxfp4Sq2Weight {
public:
  static MlxMxfp4Sq2Weight from_blob(const std::vector<std::uint8_t> &blob);
  static MlxMxfp4Sq2Weight from_blob(std::span<const std::uint8_t> blob);

  // The dedicated decode kernel expands directly to FP16/FP32. The
  // single-row matmul path is a fused packed GEMV and does not materialize
  // the complete weight matrix.
  mlx::core::array
  dequantize(mlx::core::Dtype dtype = mlx::core::float16) const;
  mlx::core::array matmul(const mlx::core::array &input) const;

  int input_size() const noexcept { return input_size_; }
  int output_size() const noexcept { return output_size_; }
  std::size_t packed_nbytes() const noexcept;

private:
  MlxMxfp4Sq2Weight(mlx::core::array symbols, mlx::core::array block_selectors,
                    mlx::core::array state_scales,
                    mlx::core::array state_palettes, int input_size,
                    int output_size);

  mlx::core::array symbols_;
  mlx::core::array block_selectors_;
  mlx::core::array state_scales_;
  mlx::core::array state_palettes_;
  int input_size_ = 0;
  int output_size_ = 0;
};

// Versioned eight-state SQ2 payload (SQ22).  It deliberately lives beside
// SQ21 instead of changing the old parser's interpretation.
class MlxMxfp4Sq2EightWeight {
public:
  static MlxMxfp4Sq2EightWeight
  from_blob(const std::vector<std::uint8_t> &blob);
  static MlxMxfp4Sq2EightWeight from_blob(std::span<const std::uint8_t> blob);

  mlx::core::array
  dequantize(mlx::core::Dtype dtype = mlx::core::float16) const;
  mlx::core::array matmul(const mlx::core::array &input) const;

  int input_size() const noexcept { return input_size_; }
  int output_size() const noexcept { return output_size_; }
  std::uint8_t matrix_scale_base() const noexcept {
    return matrix_scale_base_value_;
  }
  std::size_t packed_nbytes() const noexcept;

private:
  MlxMxfp4Sq2EightWeight(mlx::core::array symbols,
                         mlx::core::array block_selectors,
                         mlx::core::array state_scale_stream,
                         mlx::core::array state_palettes,
                         std::uint8_t matrix_scale_base_value, int input_size,
                         int output_size);

  mlx::core::array symbols_;
  mlx::core::array block_selectors_;
  // Byte zero is the matrix E8M0 base; packed two-bit state offsets follow.
  mlx::core::array state_scale_stream_;
  mlx::core::array state_palettes_;
  std::uint8_t matrix_scale_base_value_ = 0;
  int input_size_ = 0;
  int output_size_ = 0;
};

} // namespace mfq::metal
