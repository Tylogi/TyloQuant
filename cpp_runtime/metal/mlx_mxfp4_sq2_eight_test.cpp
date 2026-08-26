#include "mlx_mxfp4_sq2.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

template <typename T> void append(std::vector<std::uint8_t> &target, T value) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(&value);
  target.insert(target.end(), bytes, bytes + sizeof(T));
}

void append_bytes(std::vector<std::uint8_t> &target,
                  const std::vector<std::uint8_t> &values) {
  target.insert(target.end(), values.begin(), values.end());
}

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<std::uint8_t> pack_bits(const std::vector<std::uint8_t> &values,
                                    unsigned bits) {
  require(bits > 0 && bits <= 8, "invalid test bit width");
  std::vector<std::uint8_t> packed((values.size() * bits + 7) / 8, 0);
  const std::uint16_t mask = static_cast<std::uint16_t>((1u << bits) - 1u);
  for (std::size_t index = 0; index < values.size(); ++index) {
    require(values[index] <= mask, "test value does not fit bit width");
    const std::size_t bit_offset = index * bits;
    const std::size_t byte_index = bit_offset / 8;
    const unsigned shift = static_cast<unsigned>(bit_offset % 8);
    const std::uint16_t value = static_cast<std::uint16_t>(values[index])
                                << shift;
    packed[byte_index] |= static_cast<std::uint8_t>(value);
    if (shift + bits > 8) {
      packed[byte_index + 1] |= static_cast<std::uint8_t>(value >> 8);
    }
  }
  return packed;
}

float fp4(std::uint8_t nibble) {
  constexpr float magnitudes[]{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = magnitudes[nibble & 7u];
  return (nibble & 8u) == 0 ? magnitude : -magnitude;
}

struct Fixture {
  int rows = 0;
  int columns = 0;
  std::uint8_t matrix_scale_base = 0;
  std::size_t payload_nbytes = 0;
  std::vector<std::uint8_t> blob;
  std::vector<float> dense;
};

Fixture make_fixture(int rows, int columns) {
  require(rows > 0 && columns > 0 && columns % 32 == 0,
          "invalid eight-state SQ2 test fixture geometry");
  constexpr std::uint8_t matrix_scale_base = 124;
  const int blocks_per_row = columns / 32;
  const std::size_t weights = static_cast<std::size_t>(rows) * columns;
  const std::size_t blocks = static_cast<std::size_t>(rows) * blocks_per_row;

  std::vector<std::uint8_t> symbols(weights);
  std::vector<std::uint8_t> selectors(blocks);
  std::vector<std::uint8_t> state_scales(static_cast<std::size_t>(rows) * 8);
  std::vector<std::uint8_t> state_palettes(static_cast<std::size_t>(rows) * 8);
  std::vector<float> dense(weights);
  std::array<bool, 8> seen_tags{};

  for (int row = 0; row < rows; ++row) {
    for (int state = 0; state < 8; ++state) {
      const std::size_t index = static_cast<std::size_t>(row) * 8 + state;
      state_scales[index] = static_cast<std::uint8_t>((row + state * 3) & 3);
      state_palettes[index] =
          static_cast<std::uint8_t>((row * 9 + state * 5) & 31);
    }
    for (int block = 0; block < blocks_per_row; ++block) {
      const std::size_t block_index =
          static_cast<std::size_t>(row) * blocks_per_row + block;
      selectors[block_index] =
          static_cast<std::uint8_t>((row + (block >> 2)) & 1);
      std::uint8_t low_tag = 0;
      for (int lane = 0; lane < 31; ++lane) {
        const int column = block * 32 + lane;
        const std::size_t index =
            static_cast<std::size_t>(row) * columns + column;
        const auto symbol = static_cast<std::uint8_t>(
            (row * 11 + block * 5 + lane * 3 + (lane >> 2)) & 3);
        symbols[index] = symbol;
        low_tag ^= symbol;
      }
      const auto required_low_tag =
          static_cast<std::uint8_t>((row + block) & 3);
      const std::size_t last_index =
          static_cast<std::size_t>(row) * columns + block * 32 + 31;
      symbols[last_index] = low_tag ^ required_low_tag;
      low_tag ^= symbols[last_index];
      require(low_tag == required_low_tag,
              "failed to construct eight-state SQ2 implicit test tag");
      const std::uint8_t tag =
          static_cast<std::uint8_t>(low_tag | (selectors[block_index] << 2u));
      seen_tags[tag] = true;
      const std::size_t state_index = static_cast<std::size_t>(row) * 8 + tag;
      const auto scale_offset = state_scales[state_index];
      const auto palette = state_palettes[state_index];
      const float scale = std::ldexp(1.0f, static_cast<int>(matrix_scale_base) +
                                               scale_offset - 127);
      for (int lane = 0; lane < 32; ++lane) {
        const int column = block * 32 + lane;
        const std::size_t index =
            static_cast<std::size_t>(row) * columns + column;
        const auto nibble = mfq::metal::kMxfp4Sq2EightPaletteNibbles
            [static_cast<std::size_t>(palette) * 4 + symbols[index]];
        dense[index] = fp4(nibble) * scale;
      }
    }
  }
  if (blocks >= seen_tags.size()) {
    for (bool seen : seen_tags) {
      require(seen, "eight-state SQ2 fixture missed a block tag");
    }
  }

  const auto packed_symbols = pack_bits(symbols, 2);
  const auto packed_selectors = pack_bits(selectors, 1);
  const auto packed_state_scales = pack_bits(state_scales, 2);
  const auto packed_state_palettes = pack_bits(state_palettes, 5);

  std::vector<std::uint8_t> blob{'S', 'Q', '2', '2'};
  append<std::uint8_t>(blob, 1);
  append<std::uint8_t>(blob, matrix_scale_base);
  append<std::uint16_t>(blob, 0);
  append<std::uint64_t>(blob, rows);
  append<std::uint64_t>(blob, columns);
  const std::size_t header_nbytes = blob.size();
  append_bytes(blob, packed_symbols);
  append_bytes(blob, packed_selectors);
  append_bytes(blob, packed_state_scales);
  append_bytes(blob, packed_state_palettes);
  return {
      rows,
      columns,
      matrix_scale_base,
      blob.size() - header_nbytes + 1,
      std::move(blob),
      std::move(dense),
  };
}

void test_dequantize() {
  using namespace mlx::core;
  const auto fixture = make_fixture(7, 160);
  const auto weight =
      mfq::metal::MlxMxfp4Sq2EightWeight::from_blob(fixture.blob);
  require(weight.input_size() == fixture.columns,
          "eight-state SQ2 input size mismatch");
  require(weight.output_size() == fixture.rows,
          "eight-state SQ2 output size mismatch");
  require(weight.matrix_scale_base() == fixture.matrix_scale_base,
          "eight-state SQ2 scale base mismatch");
  require(weight.packed_nbytes() == fixture.payload_nbytes,
          "eight-state SQ2 payload byte count mismatch");

  auto fp32 = contiguous(weight.dequantize(float32));
  auto fp16 = contiguous(astype(weight.dequantize(float16), float32));
  eval(fp32, fp16);
  for (std::size_t index = 0; index < fixture.dense.size(); ++index) {
    require(fp32.data<float>()[index] == fixture.dense[index],
            "eight-state SQ2 FP32 dequantization mismatch");
    require(fp16.data<float>()[index] == fixture.dense[index],
            "eight-state SQ2 FP16 dequantization mismatch");
  }
}

void test_fused_gemv() {
  using namespace mlx::core;
  const auto fixture = make_fixture(19, 160);
  const auto weight =
      mfq::metal::MlxMxfp4Sq2EightWeight::from_blob(fixture.blob);
  std::vector<float> input_values(static_cast<std::size_t>(fixture.columns));
  for (int column = 0; column < fixture.columns; ++column) {
    input_values[static_cast<std::size_t>(column)] =
        static_cast<float>((column % 17) - 8) / 64.0f;
  }
  std::vector<float> expected_values(static_cast<std::size_t>(fixture.rows));
  for (int row = 0; row < fixture.rows; ++row) {
    for (int column = 0; column < fixture.columns; ++column) {
      expected_values[static_cast<std::size_t>(row)] +=
          input_values[static_cast<std::size_t>(column)] *
          fixture
              .dense[static_cast<std::size_t>(row) * fixture.columns + column];
    }
  }

  auto input_fp32 = array(input_values.begin(), Shape{1, fixture.columns});
  auto input_fp16 = astype(input_fp32, float16);
  auto output_fp16 = contiguous(astype(weight.matmul(input_fp16), float32));
  auto output_fp32 = contiguous(weight.matmul(input_fp32));
  eval(output_fp16, output_fp32);
  require(output_fp16.shape() == Shape{1, fixture.rows},
          "eight-state SQ2 fused GEMV shape mismatch");
  require(output_fp32.dtype() == float32,
          "eight-state SQ2 FP32 GEMV dtype mismatch");
  float fp16_maximum_difference = 0.0f;
  float fp32_maximum_difference = 0.0f;
  for (int row = 0; row < fixture.rows; ++row) {
    const float expected = expected_values[static_cast<std::size_t>(row)];
    fp16_maximum_difference =
        std::max(fp16_maximum_difference,
                 std::fabs(output_fp16.data<float>()[row] - expected));
    fp32_maximum_difference =
        std::max(fp32_maximum_difference,
                 std::fabs(output_fp32.data<float>()[row] - expected));
  }
  require(fp16_maximum_difference < 2e-3f,
          "eight-state SQ2 FP16 fused GEMV mismatch: max_abs=" +
              std::to_string(fp16_maximum_difference));
  require(fp32_maximum_difference < 2e-3f,
          "eight-state SQ2 FP32 fused GEMV mismatch: max_abs=" +
              std::to_string(fp32_maximum_difference));
}

void test_multirow_buckets() {
  using namespace mlx::core;
  const auto fixture = make_fixture(19, 96);
  const auto weight =
      mfq::metal::MlxMxfp4Sq2EightWeight::from_blob(fixture.blob);
  constexpr std::array<int, 9> row_counts{2, 6, 7, 16, 17, 32, 33, 64, 65};
  for (const int rows : row_counts) {
    std::vector<float> input_values(static_cast<std::size_t>(rows) *
                                    fixture.columns);
    for (int input_row = 0; input_row < rows; ++input_row) {
      for (int column = 0; column < fixture.columns; ++column) {
        const auto index =
            static_cast<std::size_t>(input_row) * fixture.columns + column;
        input_values[index] =
            static_cast<float>((input_row * 7 + column * 3 + 5) % 29 - 14) /
            128.0f;
      }
    }
    const Shape input_shape =
        rows == 6 ? Shape{2, 3, fixture.columns} : Shape{rows, fixture.columns};
    const Shape expected_shape =
        rows == 6 ? Shape{2, 3, fixture.rows} : Shape{rows, fixture.rows};
    auto input = astype(array(input_values.begin(), input_shape), float16);
    auto output = contiguous(astype(weight.matmul(input), float32));
    eval(output);
    require(output.shape() == expected_shape,
            "eight-state SQ2 multirow shape mismatch at M=" +
                std::to_string(rows));
    float maximum_difference = 0.0f;
    for (int input_row = 0; input_row < rows; ++input_row) {
      for (int output_row = 0; output_row < fixture.rows; ++output_row) {
        float expected = 0.0f;
        for (int column = 0; column < fixture.columns; ++column) {
          expected += input_values[static_cast<std::size_t>(input_row) *
                                       fixture.columns +
                                   column] *
                      fixture.dense[static_cast<std::size_t>(output_row) *
                                        fixture.columns +
                                    column];
        }
        maximum_difference = std::max(
            maximum_difference,
            std::fabs(
                output.data<float>()[input_row * fixture.rows + output_row] -
                expected));
      }
    }
    require(maximum_difference < 8e-3f,
            "eight-state SQ2 multirow mismatch at M=" + std::to_string(rows) +
                ": max_abs=" + std::to_string(maximum_difference));
  }
}

void test_fp32_multirow_contract() {
  using namespace mlx::core;
  const auto fixture = make_fixture(9, 96);
  const auto weight =
      mfq::metal::MlxMxfp4Sq2EightWeight::from_blob(fixture.blob);
  constexpr int rows = 7;
  std::vector<float> values(static_cast<std::size_t>(rows) * fixture.columns);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] =
        static_cast<float>(static_cast<int>(index % 23) - 11) / 128.0f;
  }
  auto input = array(values.begin(), Shape{rows, fixture.columns});
  auto actual = contiguous(weight.matmul(input));
  auto expected =
      contiguous(matmul(input, transpose(weight.dequantize(float32))));
  eval(actual, expected);
  require(actual.dtype() == float32,
          "eight-state SQ2 FP32 multirow dtype mismatch");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    require(actual.data<float>()[index] == expected.data<float>()[index],
            "eight-state SQ2 FP32 multirow contract mismatch");
  }
}

void test_blob_validation() {
  auto fixture = make_fixture(3, 64);
  const auto require_rejected = [](const std::vector<std::uint8_t> &blob,
                                   const std::string &message) {
    bool rejected = false;
    try {
      (void)mfq::metal::MlxMxfp4Sq2EightWeight::from_blob(blob);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, message);
  };

  auto truncated = fixture.blob;
  truncated.pop_back();
  require_rejected(truncated, "eight-state SQ2 truncated payload was accepted");

  auto invalid_magic = fixture.blob;
  invalid_magic[0] = 'X';
  require_rejected(invalid_magic, "eight-state SQ2 invalid magic was accepted");

  auto old_magic = fixture.blob;
  old_magic[3] = '1';
  require_rejected(old_magic, "SQ21 payload was accepted as SQ22");

  bool old_parser_rejected = false;
  try {
    (void)mfq::metal::MlxMxfp4Sq2Weight::from_blob(fixture.blob);
  } catch (const std::runtime_error &) {
    old_parser_rejected = true;
  }
  require(old_parser_rejected, "SQ21 parser accepted an SQ22 payload");

  auto trailing = fixture.blob;
  trailing.push_back(0);
  require_rejected(trailing, "eight-state SQ2 trailing payload was accepted");

  auto invalid_scale = fixture.blob;
  invalid_scale[5] = 252;
  require_rejected(invalid_scale,
                   "eight-state SQ2 invalid E8M0 base was accepted");
}

} // namespace

int main() {
  try {
    test_dequantize();
    test_fused_gemv();
    test_multirow_buckets();
    test_fp32_multirow_contract();
    test_blob_validation();
    std::cout
        << "MFQ eight-state native-MXFP4 SQ2 Metal dequant/GEMV/MMQ passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
