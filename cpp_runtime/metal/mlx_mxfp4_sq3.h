#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// Frozen format-level scalar table selected by the expert-0 SQ3 design
// screen.  Every entry is an ordinary E2M1 nibble; this is not a vector
// codebook and is not stored in each tensor payload.
inline constexpr std::array<std::uint8_t, 256> kMxfp4Sq3PaletteNibbles{
    15, 14, 13, 11, 0,  3,  5,  7,  15, 13, 11, 0,  3,  5,  6,  7,  15, 14, 13,
    11, 1,  4,  6,  7,  13, 11, 9,  0,  1,  2,  3,  6,  15, 14, 12, 9,  3,  5,
    6,  7,  15, 13, 12, 10, 0,  2,  4,  5,  13, 12, 10, 0,  2,  4,  5,  7,  14,
    12, 10, 0,  2,  4,  5,  7,  15, 14, 12, 10, 0,  3,  5,  7,  15, 13, 11, 0,
    2,  4,  5,  6,  15, 13, 11, 0,  2,  4,  6,  7,  15, 14, 12, 10, 0,  2,  4,
    5,  15, 14, 12, 10, 0,  2,  4,  6,  14, 13, 11, 0,  2,  4,  5,  6,  13, 12,
    10, 0,  3,  5,  6,  7,  15, 13, 12, 10, 0,  2,  4,  6,  14, 13, 12, 10, 0,
    3,  5,  6,  15, 13, 12, 10, 0,  3,  5,  6,  13, 12, 10, 0,  2,  4,  6,  7,
    14, 13, 11, 0,  2,  4,  5,  7,  13, 12, 9,  0,  2,  4,  5,  6,  14, 12, 10,
    0,  2,  4,  6,  7,  15, 14, 13, 11, 0,  3,  6,  7,  15, 14, 11, 0,  3,  5,
    6,  7,  14, 13, 12, 10, 0,  2,  4,  5,  14, 13, 12, 10, 0,  3,  5,  7,  15,
    14, 11, 0,  2,  4,  6,  7,  14, 11, 10, 9,  0,  1,  2,  3,  13, 11, 0,  2,
    4,  5,  6,  7,  15, 13, 12, 10, 0,  2,  4,  7,  15, 12, 10, 0,  2,  4,  5,
    7,  15, 13, 12, 10, 1,  4,  6,  7,
};

class MlxMxfp4Sq3Weight {
public:
  static MlxMxfp4Sq3Weight from_blob(const std::vector<std::uint8_t> &blob);
  static MlxMxfp4Sq3Weight from_blob(std::span<const std::uint8_t> blob);

  // The dedicated decode kernel expands directly to FP16/FP32.  The
  // single-row matmul path is a fused packed GEMV and does not materialize
  // the complete weight matrix.
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
  MlxMxfp4Sq3Weight(mlx::core::array symbols, mlx::core::array block_selectors,
                    mlx::core::array matrix_scale_base,
                    mlx::core::array state_scales,
                    mlx::core::array state_palettes,
                    std::uint8_t matrix_scale_base_value, int input_size,
                    int output_size);

  mlx::core::array symbols_;
  mlx::core::array block_selectors_;
  mlx::core::array matrix_scale_base_;
  mlx::core::array state_scales_;
  mlx::core::array state_palettes_;
  std::uint8_t matrix_scale_base_value_ = 0;
  int input_size_ = 0;
  int output_size_ = 0;
};

} // namespace mfq::metal
