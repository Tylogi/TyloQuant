#include "mlx_mx.h"
#include "mlx_mxfp4_sq2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/stream.h>

namespace {

using Clock = std::chrono::steady_clock;
using mfq::metal::MlxMxWeight;
using mlx::core::array;
using mlx::core::Shape;

#if defined(MFQ_MXFP4_SQ2_EIGHT_BENCHMARK)
using Sq2Weight = mfq::metal::MlxMxfp4Sq2EightWeight;
constexpr bool kEightState = true;
constexpr const char *kSq2Format = "MXFP4-SQ2-EIGHT";
#else
using Sq2Weight = mfq::metal::MlxMxfp4Sq2Weight;
constexpr bool kEightState = false;
constexpr const char *kSq2Format = "MXFP4-SQ2";
#endif

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
  std::vector<std::uint8_t> packed((values.size() * bits + 7) / 8, 0);
  for (std::size_t index = 0; index < values.size(); ++index) {
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

struct EncodedPair {
  std::vector<std::uint8_t> sq2_blob;
  std::vector<std::uint8_t> mxfp4_blob;
};

EncodedPair make_pair(int rows, int columns) {
  require(rows > 0 && columns > 0 && columns % 32 == 0,
          "invalid benchmark geometry");
  constexpr std::uint8_t matrix_scale_base = 124;
  constexpr int states_per_row = kEightState ? 8 : 4;
  constexpr std::uint8_t implicit_tag_mask = kEightState ? 3 : 1;
  const int blocks_per_row = columns / 32;
  const std::size_t weights = static_cast<std::size_t>(rows) * columns;
  const std::size_t blocks = static_cast<std::size_t>(rows) * blocks_per_row;

  std::vector<std::uint8_t> symbols(weights);
  std::vector<std::uint8_t> selectors(blocks);
  std::vector<std::uint8_t> state_scales(static_cast<std::size_t>(rows) *
                                         states_per_row);
  std::vector<std::uint8_t> state_palettes(static_cast<std::size_t>(rows) *
                                           states_per_row);
  std::vector<std::uint8_t> native_values(weights / 2, 0);
  std::vector<std::uint8_t> native_scales(blocks);

  for (int row = 0; row < rows; ++row) {
    for (int state = 0; state < states_per_row; ++state) {
      const std::size_t index =
          static_cast<std::size_t>(row) * states_per_row + state;
      state_scales[index] =
          kEightState
              ? static_cast<std::uint8_t>((row * 3 + state * 5) & 3)
              : static_cast<std::uint8_t>(121 + ((row * 3 + state * 5) & 3));
      state_palettes[index] = static_cast<std::uint8_t>(
          (row * 13 + state * 7) & (kEightState ? 31 : 15));
    }
    for (int block = 0; block < blocks_per_row; ++block) {
      const std::size_t block_index =
          static_cast<std::size_t>(row) * blocks_per_row + block;
      selectors[block_index] =
          static_cast<std::uint8_t>((row + (block >> 2)) & 1);
      std::uint8_t low_tag = 0;
      for (int lane = 0; lane < 31; ++lane) {
        const std::size_t index =
            static_cast<std::size_t>(row) * columns + block * 32 + lane;
        const auto symbol = static_cast<std::uint8_t>(
            (row * 11 + block * 5 + lane * 3 + (lane >> 2)) & 3);
        symbols[index] = symbol;
        low_tag ^= symbol & implicit_tag_mask;
      }
      const auto required_low_tag =
          static_cast<std::uint8_t>((row + block) & implicit_tag_mask);
      const std::size_t last_index =
          static_cast<std::size_t>(row) * columns + block * 32 + 31;
      symbols[last_index] =
          kEightState ? static_cast<std::uint8_t>(low_tag ^ required_low_tag)
                      : static_cast<std::uint8_t>((((row + block) & 1) << 1) |
                                                  (low_tag ^ required_low_tag));
      const std::uint8_t tag = static_cast<std::uint8_t>(
          required_low_tag |
          (selectors[block_index] << (kEightState ? 2u : 1u)));
      const std::size_t state_index =
          static_cast<std::size_t>(row) * states_per_row + tag;
      native_scales[block_index] =
          kEightState ? static_cast<std::uint8_t>(matrix_scale_base +
                                                  state_scales[state_index])
                      : state_scales[state_index];
      const auto palette = state_palettes[state_index];
      for (int lane = 0; lane < 32; ++lane) {
        const std::size_t value_index =
            static_cast<std::size_t>(row) * columns + block * 32 + lane;
        const auto nibble = kEightState
                                ? mfq::metal::kMxfp4Sq2EightPaletteNibbles
                                      [static_cast<std::size_t>(palette) * 4 +
                                       symbols[value_index]]
                                : mfq::metal::kMxfp4Sq2PaletteNibbles
                                      [static_cast<std::size_t>(palette) * 4 +
                                       symbols[value_index]];
        const std::size_t byte_index = value_index / 2;
        if ((lane & 1) == 0) {
          native_values[byte_index] = nibble;
        } else {
          native_values[byte_index] |= static_cast<std::uint8_t>(nibble << 4u);
        }
      }
    }
  }

  std::vector<std::uint8_t> sq2_blob{
      'S', 'Q', '2', kEightState ? std::uint8_t{'2'} : std::uint8_t{'1'}};
  append<std::uint8_t>(sq2_blob, 1);
  append<std::uint8_t>(sq2_blob,
                       kEightState ? matrix_scale_base : std::uint8_t{0});
  append<std::uint16_t>(sq2_blob, 0);
  append<std::uint64_t>(sq2_blob, rows);
  append<std::uint64_t>(sq2_blob, columns);
  append_bytes(sq2_blob, pack_bits(symbols, 2));
  append_bytes(sq2_blob, pack_bits(selectors, 1));
  if constexpr (kEightState) {
    append_bytes(sq2_blob, pack_bits(state_scales, 2));
    append_bytes(sq2_blob, pack_bits(state_palettes, 5));
  } else {
    append_bytes(sq2_blob, state_scales);
    append_bytes(sq2_blob, pack_bits(state_palettes, 4));
  }

  std::vector<std::uint8_t> mxfp4_blob{'M', 'X', 'T', '1'};
  append<std::uint8_t>(mxfp4_blob, 1);
  append<std::uint8_t>(mxfp4_blob, 4);
  append<std::uint16_t>(mxfp4_blob, 0);
  append<std::uint64_t>(mxfp4_blob, rows);
  append<std::uint64_t>(mxfp4_blob, columns);
  append<std::uint64_t>(mxfp4_blob, rows);
  append<std::uint64_t>(mxfp4_blob, columns / 2);
  append<std::uint64_t>(mxfp4_blob, rows);
  append<std::uint64_t>(mxfp4_blob, columns / 32);
  append_bytes(mxfp4_blob, native_values);
  append_bytes(mxfp4_blob, native_scales);
  return {std::move(sq2_blob), std::move(mxfp4_blob)};
}

array make_input(int width, int rows = 1) {
  std::vector<float> values(static_cast<std::size_t>(rows) * width);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < width; ++column) {
      const auto index = static_cast<std::size_t>(row) * width + column;
      values[index] =
          static_cast<float>((row * 29 + column * 17 + 11) % 127 - 63) / 256.0f;
    }
  }
  return mlx::core::astype(array(values.begin(), Shape{rows, width}),
                           mlx::core::float16);
}

double milliseconds_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

struct Measurement {
  double milliseconds = 0.0;
  double checksum = 0.0;
  float maximum = 0.0f;
};

template <typename Operation>
Measurement measure(Operation &&operation, int warmup, int repetitions) {
  array output = operation();
  mlx::core::eval(output);
  for (int index = 0; index < warmup; ++index) {
    output = operation();
    mlx::core::eval(output);
  }
  mlx::core::synchronize();
  const auto started = Clock::now();
  for (int index = 0; index < repetitions; ++index) {
    output = operation();
    mlx::core::eval(output);
  }
  mlx::core::synchronize();
  const double elapsed = milliseconds_since(started) / repetitions;

  auto checked =
      mlx::core::contiguous(mlx::core::astype(output, mlx::core::float32));
  mlx::core::eval(checked);
  double checksum = 0.0;
  float maximum = 0.0f;
  for (std::size_t index = 0; index < checked.size(); ++index) {
    const float value = checked.data<float>()[index];
    require(std::isfinite(value), "benchmark produced a non-finite value");
    checksum += static_cast<double>(value);
    maximum = std::max(maximum, std::fabs(value));
  }
  return {elapsed, checksum, maximum};
}

void verify_exact_weights(const Sq2Weight &sq2, const MlxMxWeight &native) {
  auto sq2_dense = mlx::core::contiguous(
      mlx::core::astype(sq2.dequantize(), mlx::core::float32));
  auto native_dense = mlx::core::contiguous(
      mlx::core::astype(native.dequantize(), mlx::core::float32));
  mlx::core::eval(sq2_dense, native_dense);
  require(sq2_dense.size() == native_dense.size(), "dense size mismatch");
  for (std::size_t index = 0; index < sq2_dense.size(); ++index) {
    require(sq2_dense.data<float>()[index] == native_dense.data<float>()[index],
            "SQ2/native MXFP4 decoded-weight mismatch");
  }
}

void verify_matmul(const Sq2Weight &sq2, const MlxMxWeight &native,
                   const array &input, int rows) {
  auto sq2_output = mlx::core::contiguous(
      mlx::core::astype(sq2.matmul(input), mlx::core::float32));
  auto native_output = mlx::core::contiguous(
      mlx::core::astype(native.matmul(input), mlx::core::float32));
  mlx::core::eval(sq2_output, native_output);
  float maximum_difference = 0.0f;
  for (std::size_t index = 0; index < sq2_output.size(); ++index) {
    maximum_difference = std::max(
        maximum_difference, std::fabs(sq2_output.data<float>()[index] -
                                      native_output.data<float>()[index]));
  }
  require(maximum_difference <= 0.125f,
          "SQ2/native MXFP4 matmul difference exceeds tolerance");
  std::cout << "verification\tmatmul_max_abs_diff\t" << rows << '\t'
            << maximum_difference << '\n';
}

template <typename Operation>
void print_trial(const char *format, const char *operation_name, int trial,
                 int rows, std::size_t payload_nbytes,
                 std::size_t output_nbytes, int warmup, int repetitions,
                 Operation &&operation) {
  const auto result =
      measure(std::forward<Operation>(operation), warmup, repetitions);
  const double packed_gbps =
      static_cast<double>(payload_nbytes) / (result.milliseconds * 1.0e6);
  const double total_gbps =
      static_cast<double>(payload_nbytes + output_nbytes) /
      (result.milliseconds * 1.0e6);
  std::cout << format << '\t' << operation_name << '\t' << rows << '\t' << trial
            << '\t' << payload_nbytes << '\t' << repetitions << '\t'
            << std::fixed << std::setprecision(6) << result.milliseconds << '\t'
            << std::setprecision(3) << packed_gbps << '\t' << total_gbps << '\t'
            << std::setprecision(6) << result.checksum << '\t' << result.maximum
            << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const int gemv_repetitions = argc >= 2 ? std::stoi(argv[1]) : 50;
    const int dequant_repetitions = argc >= 3 ? std::stoi(argv[2]) : 5;
    const int multirow_repetitions = argc >= 4 ? std::stoi(argv[3]) : 10;
    const int requested_rows = argc >= 5 ? std::stoi(argv[4]) : -1;
    require(gemv_repetitions > 0 && dequant_repetitions > 0 &&
                multirow_repetitions > 0 &&
                (requested_rows == -1 || requested_rows >= 2),
            "benchmark repetitions must be positive");
    constexpr int rows = 4096;
    constexpr int columns = 4096;
    auto encoded = make_pair(rows, columns);
    const auto sq2 = Sq2Weight::from_blob(encoded.sq2_blob);
    const auto native = MlxMxWeight::from_blob("MXFP4", encoded.mxfp4_blob);
    const auto input = make_input(columns);
    require(sq2.packed_nbytes() == (kEightState ? 4'288'513u : 4'284'416u),
            "unexpected SQ2 benchmark payload size");
    require(native.packed_nbytes() == 8'912'896,
            "unexpected MXFP4 benchmark payload size");

    verify_exact_weights(sq2, native);

    const std::size_t dense_nbytes =
        static_cast<std::size_t>(rows) * columns * sizeof(std::uint16_t);
    const std::size_t gemv_output_nbytes =
        static_cast<std::size_t>(rows) * sizeof(std::uint16_t);
    std::cout << "format\toperation\tM\ttrial\tpayload_bytes\trepetitions\t"
                 "mean_ms\tpacked_GB_s\ttotal_GB_s\tchecksum\tmax_abs\n";
    if (requested_rows == -1) {
      verify_matmul(sq2, native, input, 1);
      for (int trial = 0; trial < 3; ++trial) {
        const bool native_first = trial == 1;
        const auto run_dequant = [&](bool run_native) {
          if (run_native) {
            print_trial("MXFP4", "dequant_fp16", trial, 1,
                        native.packed_nbytes(), dense_nbytes, 2,
                        dequant_repetitions,
                        [&] { return native.dequantize(); });
          } else {
            print_trial(kSq2Format, "dequant_fp16", trial, 1,
                        sq2.packed_nbytes(), dense_nbytes, 2,
                        dequant_repetitions, [&] { return sq2.dequantize(); });
          }
        };
        run_dequant(native_first);
        run_dequant(!native_first);

        const auto run_gemv = [&](bool run_native) {
          if (run_native) {
            print_trial("MXFP4", "fused_gemv_fp16", trial, 1,
                        native.packed_nbytes(), gemv_output_nbytes, 3,
                        gemv_repetitions, [&] { return native.matmul(input); });
          } else {
            print_trial(kSq2Format, "fused_gemv_fp16", trial, 1,
                        sq2.packed_nbytes(), gemv_output_nbytes, 3,
                        gemv_repetitions, [&] { return sq2.matmul(input); });
          }
        };
        run_gemv(native_first);
        run_gemv(!native_first);
      }
    }

    const std::vector<int> row_counts =
        requested_rows == -1
            ? std::vector<int>{2, 6, 7, 16, 17, 32, 33, 64, 65, 128}
            : std::vector<int>{requested_rows};
    for (const int input_rows : row_counts) {
      const auto multirow_input = make_input(columns, input_rows);
      verify_matmul(sq2, native, multirow_input, input_rows);
      const auto output_nbytes =
          static_cast<std::size_t>(input_rows) * rows * sizeof(std::uint16_t);
      for (int trial = 0; trial < 3; ++trial) {
        for (int slot = 0; slot < 3; ++slot) {
          const int operation = (slot + trial) % 3;
          if (operation == 0) {
            print_trial(kSq2Format, "bucket_matmul_fp16", trial, input_rows,
                        sq2.packed_nbytes(), output_nbytes, 2,
                        multirow_repetitions,
                        [&] { return sq2.matmul(multirow_input); });
          } else if (operation == 1) {
            print_trial("MXFP4", "packed_or_dense_matmul_fp16", trial,
                        input_rows, native.packed_nbytes(), output_nbytes, 2,
                        multirow_repetitions,
                        [&] { return native.matmul(multirow_input); });
          } else {
            print_trial(
                kEightState ? "MXFP4-SQ2-EIGHT-DENSE" : "MXFP4-SQ2-DENSE",
                "dequant_matmul_fp16", trial, input_rows, sq2.packed_nbytes(),
                output_nbytes, 2, multirow_repetitions, [&] {
                  return mlx::core::matmul(
                      multirow_input, mlx::core::transpose(sq2.dequantize()));
                });
          }
        }
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MFQ " << kSq2Format
              << " Metal benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
