#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

double moe_phase_seconds[4] = {0.0, 0.0, 0.0, 0.0};
int64_t moe_phase_calls = 0;
double packed_moe_phase_seconds[6] = {
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
int64_t packed_moe_phase_calls = 0;

inline double wall_seconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
inline __m512 lookup_16(
    const float* score,
    const uint8_t* indices,
    int64_t block,
    int64_t codes,
    __m512i stride) {
  const __m128i packed = _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(indices + block));
  const __m512i selected = _mm512_cvtepu8_epi32(packed);
  const __m512i base = _mm512_set1_epi32(
      static_cast<int>(block * codes));
  const __m512i offsets = _mm512_add_epi32(
      selected, _mm512_add_epi32(base, stride));
  return _mm512_i32gather_ps(offsets, score, 4);
}

inline __m512 lookup_16_u16(
    const float* score,
    const uint16_t* indices,
    int64_t block,
    int64_t codes,
    __m512i stride) {
  const __m256i packed = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(indices + block));
  const __m512i selected = _mm512_cvtepu16_epi32(packed);
  const __m512i base = _mm512_set1_epi32(
      static_cast<int>(block * codes));
  const __m512i offsets = _mm512_add_epi32(
      selected, _mm512_add_epi32(base, stride));
  return _mm512_i32gather_ps(offsets, score, 4);
}

inline __m512 lookup_rows_16(
    const float* block_score,
    const uint8_t* row_indices) {
  const __m128i packed = _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(row_indices));
  const __m512i selected = _mm512_cvtepu8_epi32(packed);
  return _mm512_i32gather_ps(selected, block_score, 4);
}

#if defined(__AVX512VBMI__)
inline __m512i lookup_i8_rows_64(
    const int8_t* table,
    const uint8_t* row_indices) {
  const __m512i indices = _mm512_loadu_si512(row_indices);
  const __m512i low_indices = _mm512_and_si512(
      indices, _mm512_set1_epi8(0x7f));
  const __m512i table0 = _mm512_loadu_si512(table);
  const __m512i table1 = _mm512_loadu_si512(table + 64);
  const __m512i table2 = _mm512_loadu_si512(table + 128);
  const __m512i table3 = _mm512_loadu_si512(table + 192);
  const __m512i low_values = _mm512_permutex2var_epi8(
      table0, low_indices, table1);
  const __m512i high_values = _mm512_permutex2var_epi8(
      table2, low_indices, table3);
  const __mmask64 high_mask = _mm512_cmp_epu8_mask(
      indices, _mm512_set1_epi8(static_cast<char>(0x80)),
      _MM_CMPINT_GE);
  return _mm512_mask_blend_epi8(
      high_mask, low_values, high_values);
}

inline void add_i8_scores_64(
    int16_t* partial,
    const __m512i scores) {
  const __m512i low = _mm512_cvtepi8_epi16(
      _mm512_castsi512_si256(scores));
  const __m512i high = _mm512_cvtepi8_epi16(
      _mm512_extracti64x4_epi64(scores, 1));
  _mm512_storeu_si512(
      partial,
      _mm512_add_epi16(
          _mm512_loadu_si512(partial), low));
  _mm512_storeu_si512(
      partial + 32,
      _mm512_add_epi16(
          _mm512_loadu_si512(partial + 32), high));
}
#endif
#endif

inline float lookup_sum(
    const float* score,
    const uint8_t* row_idx,
    int64_t blocks,
    int64_t codes) {
  float sum = 0.0f;
  int64_t b = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
  const __m512i lanes = _mm512_setr_epi32(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i stride = _mm512_mullo_epi32(
      lanes, _mm512_set1_epi32(static_cast<int>(codes)));
  __m512 accumulated = _mm512_setzero_ps();
  // Issue four independent gathers before consuming them.  The additions
  // remain in the original block order, preserving the existing FP32 result,
  // while Xeon can overlap the otherwise latency-bound score-table reads.
  for (; b + 64 <= blocks; b += 64) {
    const __m512 first =
        lookup_16(score, row_idx, b, codes, stride);
    const __m512 second =
        lookup_16(score, row_idx, b + 16, codes, stride);
    const __m512 third =
        lookup_16(score, row_idx, b + 32, codes, stride);
    const __m512 fourth =
        lookup_16(score, row_idx, b + 48, codes, stride);
    accumulated = _mm512_add_ps(accumulated, first);
    accumulated = _mm512_add_ps(accumulated, second);
    accumulated = _mm512_add_ps(accumulated, third);
    accumulated = _mm512_add_ps(accumulated, fourth);
  }
  for (; b + 16 <= blocks; b += 16) {
    accumulated = _mm512_add_ps(
        accumulated, lookup_16(score, row_idx, b, codes, stride));
  }
  sum = _mm512_reduce_add_ps(accumulated);
#endif
  for (; b < blocks; ++b) {
    sum += score[b * codes + row_idx[b]];
  }
  return sum;
}

inline float lookup_sum_u16(
    const float* score,
    const uint16_t* row_idx,
    int64_t blocks,
    int64_t codes) {
  float sum = 0.0f;
  int64_t block = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
  const __m512i lanes = _mm512_setr_epi32(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i stride = _mm512_mullo_epi32(
      lanes, _mm512_set1_epi32(static_cast<int>(codes)));
  __m512 accumulated = _mm512_setzero_ps();
  for (; block + 64 <= blocks; block += 64) {
    const __m512 first =
        lookup_16_u16(score, row_idx, block, codes, stride);
    const __m512 second =
        lookup_16_u16(score, row_idx, block + 16, codes, stride);
    const __m512 third =
        lookup_16_u16(score, row_idx, block + 32, codes, stride);
    const __m512 fourth =
        lookup_16_u16(score, row_idx, block + 48, codes, stride);
    accumulated = _mm512_add_ps(accumulated, first);
    accumulated = _mm512_add_ps(accumulated, second);
    accumulated = _mm512_add_ps(accumulated, third);
    accumulated = _mm512_add_ps(accumulated, fourth);
  }
  for (; block + 16 <= blocks; block += 16) {
    accumulated = _mm512_add_ps(
        accumulated,
        lookup_16_u16(score, row_idx, block, codes, stride));
  }
  sum = _mm512_reduce_add_ps(accumulated);
#endif
  for (; block < blocks; ++block) {
    sum += score[block * codes + row_idx[block]];
  }
  return sum;
}

inline void lookup_sum_pair(
    const float* score,
    const uint8_t* first_indices,
    const uint8_t* second_indices,
    int64_t blocks,
    int64_t codes,
    float& first_sum,
    float& second_sum) {
  int64_t block = 0;
  first_sum = 0.0f;
  second_sum = 0.0f;
#if defined(__AVX512F__) && defined(__AVX512BW__)
  const __m512i lanes = _mm512_setr_epi32(
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i stride = _mm512_mullo_epi32(
      lanes, _mm512_set1_epi32(static_cast<int>(codes)));
  __m512 first_accumulated = _mm512_setzero_ps();
  __m512 second_accumulated = _mm512_setzero_ps();
  for (; block + 64 <= blocks; block += 64) {
    const __m512 first0 =
        lookup_16(score, first_indices, block, codes, stride);
    const __m512 second0 =
        lookup_16(score, second_indices, block, codes, stride);
    const __m512 first1 =
        lookup_16(score, first_indices, block + 16, codes, stride);
    const __m512 second1 =
        lookup_16(score, second_indices, block + 16, codes, stride);
    const __m512 first2 =
        lookup_16(score, first_indices, block + 32, codes, stride);
    const __m512 second2 =
        lookup_16(score, second_indices, block + 32, codes, stride);
    const __m512 first3 =
        lookup_16(score, first_indices, block + 48, codes, stride);
    const __m512 second3 =
        lookup_16(score, second_indices, block + 48, codes, stride);
    first_accumulated = _mm512_add_ps(first_accumulated, first0);
    second_accumulated = _mm512_add_ps(second_accumulated, second0);
    first_accumulated = _mm512_add_ps(first_accumulated, first1);
    second_accumulated = _mm512_add_ps(second_accumulated, second1);
    first_accumulated = _mm512_add_ps(first_accumulated, first2);
    second_accumulated = _mm512_add_ps(second_accumulated, second2);
    first_accumulated = _mm512_add_ps(first_accumulated, first3);
    second_accumulated = _mm512_add_ps(second_accumulated, second3);
  }
  for (; block + 32 <= blocks; block += 32) {
    const __m512 first0 =
        lookup_16(score, first_indices, block, codes, stride);
    const __m512 second0 =
        lookup_16(score, second_indices, block, codes, stride);
    const __m512 first1 =
        lookup_16(score, first_indices, block + 16, codes, stride);
    const __m512 second1 =
        lookup_16(score, second_indices, block + 16, codes, stride);
    first_accumulated = _mm512_add_ps(first_accumulated, first0);
    second_accumulated = _mm512_add_ps(second_accumulated, second0);
    first_accumulated = _mm512_add_ps(first_accumulated, first1);
    second_accumulated = _mm512_add_ps(second_accumulated, second1);
  }
  for (; block + 16 <= blocks; block += 16) {
    const __m512 first =
        lookup_16(score, first_indices, block, codes, stride);
    const __m512 second =
        lookup_16(score, second_indices, block, codes, stride);
    first_accumulated = _mm512_add_ps(first_accumulated, first);
    second_accumulated = _mm512_add_ps(second_accumulated, second);
  }
  first_sum = _mm512_reduce_add_ps(first_accumulated);
  second_sum = _mm512_reduce_add_ps(second_accumulated);
#endif
  for (; block < blocks; ++block) {
    first_sum += score[block * codes + first_indices[block]];
    second_sum += score[block * codes + second_indices[block]];
  }
}

inline float lookup_weighted_many(
    const std::vector<int64_t>& score_offsets,
    const std::vector<const uint8_t*>& index_ptrs,
    const std::vector<int64_t>& blocks,
    const std::vector<int64_t>& codes,
    const float* scores,
    const float* weights,
    int64_t experts,
    int64_t row) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  if (experts <= 16) {
    __m512 accumulated[16];
    __m512i strides[16];
    const __m512i lanes = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15);
    for (int64_t expert = 0; expert < experts; ++expert) {
      accumulated[expert] = _mm512_setzero_ps();
      strides[expert] = _mm512_mullo_epi32(
          lanes,
          _mm512_set1_epi32(static_cast<int>(codes[expert])));
    }
    int64_t maximum_blocks = 0;
    for (int64_t expert = 0; expert < experts; ++expert) {
      maximum_blocks = std::max(maximum_blocks, blocks[expert]);
    }
    for (int64_t block = 0;
         block + 16 <= maximum_blocks;
         block += 16) {
      __m512 gathered[16];
      bool active[16] = {};
      for (int64_t expert = 0; expert < experts; ++expert) {
        if (block + 16 <= blocks[expert]) {
          gathered[expert] = lookup_16(
              scores + score_offsets[expert],
              index_ptrs[expert] + row * blocks[expert],
              block,
              codes[expert],
              strides[expert]);
          active[expert] = true;
        }
      }
      for (int64_t expert = 0; expert < experts; ++expert) {
        if (active[expert]) {
          accumulated[expert] =
              _mm512_add_ps(accumulated[expert], gathered[expert]);
        }
      }
    }
    float result = 0.0f;
    for (int64_t expert = 0; expert < experts; ++expert) {
      float sum = _mm512_reduce_add_ps(accumulated[expert]);
      const int64_t tail = (blocks[expert] / 16) * 16;
      const float* score = scores + score_offsets[expert];
      const uint8_t* indices =
          index_ptrs[expert] + row * blocks[expert];
      for (int64_t block = tail; block < blocks[expert]; ++block) {
        sum += score[block * codes[expert] + indices[block]];
      }
      result += weights[expert] * sum;
    }
    return result;
  }
#endif
  float result = 0.0f;
  for (int64_t expert = 0; expert < experts; ++expert) {
    result += weights[expert] *
              lookup_sum(
                  scores + score_offsets[expert],
                  index_ptrs[expert] + row * blocks[expert],
                  blocks[expert],
                  codes[expert]);
  }
  return result;
}

inline float int4_group_dot(
    const float* x,
    const uint8_t* packed,
    int64_t group_size) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  if (group_size == 64) {
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m512i zero_point = _mm512_set1_epi32(8);
    __m512 sum = _mm512_setzero_ps();
    for (int64_t byte_offset = 0; byte_offset < 32; byte_offset += 16) {
      const __m128i values = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(packed + byte_offset));
      const __m128i low = _mm_and_si128(values, nibble_mask);
      const __m128i high = _mm_and_si128(
          _mm_srli_epi16(values, 4), nibble_mask);
      const __m128i first = _mm_unpacklo_epi8(low, high);
      const __m128i second = _mm_unpackhi_epi8(low, high);
      const __m512 first_weight = _mm512_cvtepi32_ps(
          _mm512_sub_epi32(_mm512_cvtepu8_epi32(first), zero_point));
      const __m512 second_weight = _mm512_cvtepi32_ps(
          _mm512_sub_epi32(_mm512_cvtepu8_epi32(second), zero_point));
      const int64_t x_offset = byte_offset * 2;
      sum = _mm512_fmadd_ps(
          _mm512_loadu_ps(x + x_offset), first_weight, sum);
      sum = _mm512_fmadd_ps(
          _mm512_loadu_ps(x + x_offset + 16), second_weight, sum);
    }
    return _mm512_reduce_add_ps(sum);
  }
#endif
  float sum = 0.0f;
  for (int64_t j = 0; j < group_size / 2; ++j) {
    const uint8_t value = packed[j];
    sum += x[2 * j] *
               static_cast<float>(static_cast<int>(value & 15) - 8) +
           x[2 * j + 1] *
               static_cast<float>(static_cast<int>(value >> 4) - 8);
  }
  return sum;
}

inline float int4_row_dot(
    const float* x,
    const uint8_t* packed,
    const at::Half* scales,
    int64_t cols,
    int64_t group_size) {
  const int64_t groups = cols / group_size;
  const int64_t bytes_per_group = group_size / 2;
  float total = 0.0f;
  for (int64_t group = 0; group < groups; ++group) {
    total +=
        int4_group_dot(
            x + group * group_size,
            packed + group * bytes_per_group,
            group_size) *
        static_cast<float>(scales[group]);
  }
  return total;
}

struct Int8Activation {
  std::vector<int16_t> even;
  std::vector<int16_t> odd;
  std::vector<float> scales;
  int64_t cols = 0;
  int64_t group_size = 0;
};

inline bool cpu_w4a8_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("TPQ_CPU_W4A8");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return enabled;
}

Int8Activation quantize_int8_activation(
    const float* input,
    int64_t cols,
    int64_t group_size) {
  TORCH_CHECK(
      group_size == 64 && cols % group_size == 0,
      "W4A8 CPU path currently requires group size 64");
  Int8Activation quantized;
  quantized.cols = cols;
  quantized.group_size = group_size;
  quantized.even.resize(cols / 2);
  quantized.odd.resize(cols / 2);
  quantized.scales.resize(cols / group_size);
  for (int64_t group = 0; group < cols / group_size; ++group) {
    const float* values = input + group * group_size;
    float maximum = 0.0f;
    for (int64_t index = 0; index < group_size; ++index) {
      maximum = std::max(maximum, std::abs(values[index]));
    }
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    const float inverse = 1.0f / scale;
    quantized.scales[group] = scale;
    int16_t* even = quantized.even.data() + group * (group_size / 2);
    int16_t* odd = quantized.odd.data() + group * (group_size / 2);
    for (int64_t index = 0; index < group_size / 2; ++index) {
      const int first = static_cast<int>(
          std::nearbyint(values[2 * index] * inverse));
      const int second = static_cast<int>(
          std::nearbyint(values[2 * index + 1] * inverse));
      even[index] = static_cast<int16_t>(
          std::max(-127, std::min(127, first)));
      odd[index] = static_cast<int16_t>(
          std::max(-127, std::min(127, second)));
    }
  }
  return quantized;
}

inline float int4_row_dot_w4a8(
    const Int8Activation& input,
    const uint8_t* packed,
    const at::Half* scales) {
  const int64_t groups = input.cols / input.group_size;
  const int64_t values_per_parity = input.group_size / 2;
  const int64_t bytes_per_group = input.group_size / 2;
  float total = 0.0f;
  for (int64_t group = 0; group < groups; ++group) {
    int32_t integer_dot = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m256i packed_values = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(
            packed + group * bytes_per_group));
    const __m512i words = _mm512_cvtepu8_epi16(packed_values);
    const __m512i mask = _mm512_set1_epi16(0x0f);
    const __m512i offset = _mm512_set1_epi16(8);
    const __m512i low = _mm512_sub_epi16(
        _mm512_and_si512(words, mask), offset);
    const __m512i high = _mm512_sub_epi16(
        _mm512_and_si512(_mm512_srli_epi16(words, 4), mask),
        offset);
    const __m512i even = _mm512_loadu_si512(
        input.even.data() + group * values_per_parity);
    const __m512i odd = _mm512_loadu_si512(
        input.odd.data() + group * values_per_parity);
    const __m512i products = _mm512_add_epi32(
        _mm512_madd_epi16(low, even),
        _mm512_madd_epi16(high, odd));
    integer_dot = _mm512_reduce_add_epi32(products);
#else
    const int16_t* even =
        input.even.data() + group * values_per_parity;
    const int16_t* odd =
        input.odd.data() + group * values_per_parity;
    const uint8_t* weights = packed + group * bytes_per_group;
    for (int64_t index = 0; index < values_per_parity; ++index) {
      integer_dot +=
          even[index] * (static_cast<int>(weights[index] & 15) - 8) +
          odd[index] * (static_cast<int>(weights[index] >> 4) - 8);
    }
#endif
    total += static_cast<float>(integer_dot) *
             input.scales[group] *
             static_cast<float>(scales[group]);
  }
  return total;
}

struct Bf16Activation {
  std::vector<at::BFloat16> values;
  int64_t cols = 0;
  int64_t group_size = 0;
};

inline bool cpu_w4abf16_enabled() {
  const char* value = std::getenv("TPQ_CPU_W4ABF16");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

Bf16Activation quantize_bf16_activation(
    const float* input,
    int64_t cols,
    int64_t group_size) {
  TORCH_CHECK(
      group_size == 64 && cols % group_size == 0,
      "W4ABF16 CPU path currently requires group size 64");
  Bf16Activation converted;
  converted.cols = cols;
  converted.group_size = group_size;
  converted.values.resize(cols);
  int64_t index = 0;
#if defined(__AVX512BF16__)
  for (; index + 16 <= cols; index += 16) {
    const __m256bh packed =
        _mm512_cvtneps_pbh(_mm512_loadu_ps(input + index));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i*>(converted.values.data() + index),
        (__m256i)packed);
  }
#endif
  for (; index < cols; ++index) {
    converted.values[index] = at::BFloat16(input[index]);
  }
  return converted;
}

inline float int4_row_dot_w4abf16(
    const Bf16Activation& input,
    const uint8_t* packed,
    const at::Half* scales) {
  const int64_t groups = input.cols / input.group_size;
  const int64_t bytes_per_group = input.group_size / 2;
  float total = 0.0f;
  for (int64_t group = 0; group < groups; ++group) {
    float dot = 0.0f;
#if defined(__AVX512BF16__) && defined(__AVX512BW__)
    const __m512i zero_point = _mm512_set1_epi32(8);
    __m512 accumulated = _mm512_setzero_ps();
    for (int64_t byte_offset = 0;
         byte_offset < bytes_per_group;
         byte_offset += 16) {
      const __m128i values = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(
              packed + group * bytes_per_group + byte_offset));
      const __m128i nibble_mask = _mm_set1_epi8(0x0f);
      const __m128i low = _mm_and_si128(values, nibble_mask);
      const __m128i high = _mm_and_si128(
          _mm_srli_epi16(values, 4), nibble_mask);
      const __m128i first = _mm_unpacklo_epi8(low, high);
      const __m128i second = _mm_unpackhi_epi8(low, high);
      const __m512 first_weight = _mm512_cvtepi32_ps(
          _mm512_sub_epi32(
              _mm512_cvtepu8_epi32(first), zero_point));
      const __m512 second_weight = _mm512_cvtepi32_ps(
          _mm512_sub_epi32(
              _mm512_cvtepu8_epi32(second), zero_point));
      const __m512bh weight = _mm512_cvtne2ps_pbh(
          second_weight, first_weight);
      const __m512bh activation = (__m512bh)_mm512_loadu_si512(
          input.values.data() +
          group * input.group_size + byte_offset * 2);
      accumulated =
          _mm512_dpbf16_ps(accumulated, activation, weight);
    }
    dot = _mm512_reduce_add_ps(accumulated);
#else
    const at::BFloat16* values =
        input.values.data() + group * input.group_size;
    const uint8_t* weights =
        packed + group * bytes_per_group;
    for (int64_t index = 0; index < input.group_size / 2; ++index) {
      dot += static_cast<float>(values[2 * index]) *
                 static_cast<float>(
                     static_cast<int>(weights[index] & 15) - 8) +
             static_cast<float>(values[2 * index + 1]) *
                 static_cast<float>(
                     static_cast<int>(weights[index] >> 4) - 8);
    }
#endif
    total += dot * static_cast<float>(scales[group]);
  }
  return total;
}

struct ExpandedBf16Weight {
  torch::Tensor packed_reference;
  torch::Tensor values;
};

std::unordered_map<const void*, ExpandedBf16Weight>
    expanded_bf16_weights;

inline bool cpu_expand_bf16_enabled() {
  const char* value = std::getenv("TPQ_CPU_EXPAND_BF16");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

torch::Tensor expand_int4_bf16(
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    int64_t cols,
    int64_t group_size) {
  const void* key = packed.data_ptr<uint8_t>();
  auto found = expanded_bf16_weights.find(key);
  if (found != expanded_bf16_weights.end()) {
    return found->second.values;
  }
  TORCH_CHECK(
      !packed.is_cuda() && !scales.is_cuda() &&
          packed.scalar_type() == at::kByte &&
          scales.scalar_type() == at::kHalf &&
          packed.dim() == 2 && scales.dim() == 2 &&
          packed.size(1) * 2 == cols &&
          scales.size(0) == packed.size(0) &&
          scales.size(1) * group_size == cols,
      "CPU BF16 expansion shape mismatch");
  auto output = torch::empty(
      {packed.size(0), cols},
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));
  const uint8_t* qp = packed.data_ptr<uint8_t>();
  const at::Half* sp = scales.data_ptr<at::Half>();
  at::BFloat16* op = output.data_ptr<at::BFloat16>();
  const int64_t rows = packed.size(0);
  const int64_t groups = cols / group_size;
  const int64_t bytes_per_row = cols / 2;
#pragma omp parallel for schedule(static)
  for (int64_t row = 0; row < rows; ++row) {
    const uint8_t* weights = qp + row * bytes_per_row;
    const at::Half* row_scales = sp + row * groups;
    at::BFloat16* destination = op + row * cols;
    for (int64_t group = 0; group < groups; ++group) {
      const float scale = static_cast<float>(row_scales[group]);
      const uint8_t* values =
          weights + group * (group_size / 2);
      for (int64_t index = 0; index < group_size / 2; ++index) {
        destination[group * group_size + 2 * index] =
            at::BFloat16(
                static_cast<float>(
                    static_cast<int>(values[index] & 15) - 8) *
                scale);
        destination[group * group_size + 2 * index + 1] =
            at::BFloat16(
                static_cast<float>(
                    static_cast<int>(values[index] >> 4) - 8) *
                scale);
      }
    }
  }
  expanded_bf16_weights.emplace(
      key, ExpandedBf16Weight{packed, output});
  return output;
}

inline float bf16_row_dot(
    const at::BFloat16* input,
    const at::BFloat16* weight,
    int64_t cols) {
  float result = 0.0f;
  int64_t index = 0;
#if defined(__AVX512BF16__)
  __m512 accumulated = _mm512_setzero_ps();
  for (; index + 32 <= cols; index += 32) {
    const __m512bh x = (__m512bh)_mm512_loadu_si512(input + index);
    const __m512bh w = (__m512bh)_mm512_loadu_si512(weight + index);
    accumulated = _mm512_dpbf16_ps(accumulated, x, w);
  }
  result = _mm512_reduce_add_ps(accumulated);
#endif
  for (; index < cols; ++index) {
    result += static_cast<float>(input[index]) *
              static_cast<float>(weight[index]);
  }
  return result;
}

inline float float_dot(const float* left, const float* right, int64_t size) {
  float sum = 0.0f;
  int64_t index = 0;
#if defined(__AVX512F__)
  __m512 accumulated = _mm512_setzero_ps();
  for (; index + 16 <= size; index += 16) {
    accumulated = _mm512_fmadd_ps(
        _mm512_loadu_ps(left + index),
        _mm512_loadu_ps(right + index),
        accumulated);
  }
  sum = _mm512_reduce_add_ps(accumulated);
#endif
  for (; index < size; ++index) {
    sum += left[index] * right[index];
  }
  return sum;
}

inline void codebook_scores(
    const float* input,
    const float* transposed_codebook,
    float* output,
    int64_t codes,
    int64_t dimension) {
  int64_t code = 0;
#if defined(__AVX512F__)
  // Kimi code vectors are only 4/8 floats wide.  Vectorising across codes
  // keeps 16 independent dot products in one register instead of calling a
  // tiny scalar dot routine K times.
  for (; code + 16 <= codes; code += 16) {
    __m512 accumulated = _mm512_setzero_ps();
    for (int64_t index = 0; index < dimension; ++index) {
      accumulated = _mm512_fmadd_ps(
          _mm512_set1_ps(input[index]),
          _mm512_loadu_ps(
              transposed_codebook + index * codes + code),
          accumulated);
    }
    _mm512_storeu_ps(output + code, accumulated);
  }
#endif
  for (; code < codes; ++code) {
    float sum = 0.0f;
    for (int64_t index = 0; index < dimension; ++index) {
      sum += input[index] *
             transposed_codebook[index * codes + code];
    }
    output[code] = sum;
  }
}

struct TransposedCodebook {
  torch::Tensor source;
  torch::Tensor values;
};

std::unordered_map<const void*, TransposedCodebook>
    transposed_codebooks;
std::mutex transposed_codebooks_mutex;

torch::Tensor cached_transposed_codebook(torch::Tensor codebook) {
  const void* key = codebook.data_ptr<float>();
  std::lock_guard<std::mutex> guard(transposed_codebooks_mutex);
  auto found = transposed_codebooks.find(key);
  if (found != transposed_codebooks.end()) {
    return found->second.values;
  }
  const int64_t codes = codebook.size(0);
  const int64_t dimension = codebook.size(1);
  auto output = torch::empty(
      {dimension, codes},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const float* source = codebook.data_ptr<float>();
  float* destination = output.data_ptr<float>();
  // These codebooks are only a few hundred KiB.  A serial transpose avoids
  // launching the large host thread pool during every layer's first access.
  for (int64_t code = 0; code < codes; ++code) {
    for (int64_t index = 0; index < dimension; ++index) {
      destination[index * codes + code] =
          source[code * dimension + index];
    }
  }
  transposed_codebooks.emplace(
      key, TransposedCodebook{codebook, output});
  return output;
}

inline void float_axpy(
    float* output,
    const float* value,
    float weight,
    int64_t size) {
  int64_t index = 0;
#if defined(__AVX512F__)
  const __m512 scale = _mm512_set1_ps(weight);
  for (; index + 16 <= size; index += 16) {
    _mm512_storeu_ps(
        output + index,
        _mm512_fmadd_ps(
            _mm512_loadu_ps(value + index),
            scale,
            _mm512_loadu_ps(output + index)));
  }
#endif
  for (; index < size; ++index) {
    output[index] += value[index] * weight;
  }
}

torch::Tensor vq_gemv_cpu(
    torch::Tensor x_rows,
    torch::Tensor indices,
    torch::Tensor codebooks) {
  TORCH_CHECK(!x_rows.is_cuda(), "x_rows must be on CPU");
  TORCH_CHECK(!indices.is_cuda(), "indices must be on CPU");
  TORCH_CHECK(!codebooks.is_cuda(), "codebooks must be on CPU");
  TORCH_CHECK(x_rows.dim() == 2, "x_rows must have shape [N|1,C]");
  TORCH_CHECK(indices.dim() == 3, "indices must have shape [N|1,R,B]");
  TORCH_CHECK(codebooks.dim() == 3, "codebooks must have shape [N|1,K,D]");
  const bool indices_u8 = indices.scalar_type() == at::kByte;
  const bool indices_u16 = indices.scalar_type() == at::kUInt16;
  TORCH_CHECK(
      indices_u8 || indices_u16,
      "CPU VQ GEMV supports uint8 or uint16 indices");

  // Converting these two small operands once is much cheaper than expanding
  // every uint8 expert index to int64 and materialising [N,B,R] with gather.
  auto x = x_rows.to(torch::kFloat32).contiguous();
  auto cb = codebooks.to(torch::kFloat32).contiguous();
  auto idx = indices.contiguous();

  const int64_t xn = x.size(0);
  const int64_t in = idx.size(0);
  const int64_t cn = cb.size(0);
  const int64_t rows = idx.size(1);
  const int64_t blocks = idx.size(2);
  const int64_t codes = cb.size(1);
  const int64_t dim = cb.size(2);
  const int64_t n = std::max({xn, in, cn});

  TORCH_CHECK(x.size(1) == blocks * dim,
              "x width must equal index blocks * codebook dimension");
  TORCH_CHECK(
      (indices_u8 && codes <= 256) ||
          (indices_u16 && codes <= 65536),
      "index dtype cannot represent every codebook entry");
  TORCH_CHECK(xn == 1 || xn == n, "x batch is not broadcastable");
  TORCH_CHECK(in == 1 || in == n, "index batch is not broadcastable");
  TORCH_CHECK(cn == 1 || cn == n, "codebook batch is not broadcastable");

  // Lookup scores are shared by all output rows of one expert:
  // score[n,b,k] = dot(x[n,b,:], codebook[n,k,:]).
  // Decode normally has x/codebook batch 1, so this is only B*K floats.
  const int64_t score_n = std::max(xn, cn);
  auto scores = torch::empty(
      {score_n, blocks, codes},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const float* xp = x.data_ptr<float>();
  const float* cp = cb.data_ptr<float>();
  float* sp = scores.data_ptr<float>();
  at::parallel_for(0, score_n * blocks, 8, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t sn = item / blocks;
      const int64_t b = item - sn * blocks;
      const int64_t xbatch = xn == 1 ? 0 : sn;
      const int64_t cbatch = cn == 1 ? 0 : sn;
      const float* xv = xp + (xbatch * blocks + b) * dim;
      const float* codebook = cp + cbatch * codes * dim;
      float* score = sp + (sn * blocks + b) * codes;
      for (int64_t k = 0; k < codes; ++k) {
        const float* code = codebook + k * dim;
        float sum = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
          sum += xv[d] * code[d];
        }
        score[k] = sum;
      }
    }
  });

  auto out = torch::empty(
      {n, rows},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const uint8_t* ip8 =
      indices_u8 ? idx.data_ptr<uint8_t>() : nullptr;
  const uint16_t* ip16 =
      indices_u16 ? idx.data_ptr<uint16_t>() : nullptr;
  float* op = out.data_ptr<float>();
  at::parallel_for(0, n * rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t batch = item / rows;
      const int64_t row = item - batch * rows;
      const int64_t ibatch = in == 1 ? 0 : batch;
      const int64_t sbatch = score_n == 1 ? 0 : batch;
      const float* score = sp + sbatch * blocks * codes;
      const int64_t offset = (ibatch * rows + row) * blocks;
      op[item] = indices_u8
          ? lookup_sum(score, ip8 + offset, blocks, codes)
          : lookup_sum_u16(score, ip16 + offset, blocks, codes);
    }
  });
  return out;
}

torch::Tensor vq_gemv_list_cpu(
    torch::Tensor x_rows,
    std::vector<torch::Tensor> index_list,
    torch::Tensor codebook) {
  TORCH_CHECK(!x_rows.is_cuda() && !codebook.is_cuda(),
              "VQ list operands must be on CPU");
  TORCH_CHECK(x_rows.dim() == 2, "x_rows must be [N|1,C]");
  TORCH_CHECK(codebook.dim() == 2, "codebook must be [K,D]");
  TORCH_CHECK(!index_list.empty(), "VQ index list cannot be empty");
  const int64_t n = static_cast<int64_t>(index_list.size());
  const int64_t rows = index_list[0].size(0);
  const int64_t blocks = index_list[0].size(1);
  const auto index_type = index_list[0].scalar_type();
  const bool indices_u8 = index_type == at::kByte;
  const bool indices_u16 = index_type == at::kUInt16;
  TORCH_CHECK(
      indices_u8 || indices_u16,
      "VQ list indices must be CPU uint8 or uint16 tensors");
  for (const auto& index : index_list) {
    TORCH_CHECK(
        !index.is_cuda() && index.scalar_type() == index_type,
        "VQ list indices must be homogeneous CPU tensors");
    TORCH_CHECK(index.dim() == 2 && index.size(0) == rows &&
                    index.size(1) == blocks,
                "VQ list index shapes must match");
  }
  TORCH_CHECK(x_rows.size(0) == 1 || x_rows.size(0) == n,
              "VQ list input batch must be 1 or expert count");
  const int64_t codes = codebook.size(0);
  const int64_t dim = codebook.size(1);
  TORCH_CHECK(
      (indices_u8 && codes <= 256) ||
          (indices_u16 && codes <= 65536),
      "index dtype cannot represent every codebook entry");
  TORCH_CHECK(x_rows.size(1) == blocks * dim,
              "VQ list input width mismatch");

  auto x = x_rows.to(torch::kFloat32).contiguous();
  auto cb = codebook.to(torch::kFloat32).contiguous();
  std::vector<torch::Tensor> indices;
  std::vector<const uint8_t*> index_ptrs_u8;
  std::vector<const uint16_t*> index_ptrs_u16;
  indices.reserve(n);
  index_ptrs_u8.reserve(indices_u8 ? n : 0);
  index_ptrs_u16.reserve(indices_u16 ? n : 0);
  for (auto& index : index_list) {
    indices.push_back(index.contiguous());
    if (indices_u8) {
      index_ptrs_u8.push_back(indices.back().data_ptr<uint8_t>());
    } else {
      index_ptrs_u16.push_back(indices.back().data_ptr<uint16_t>());
    }
  }
  const int64_t score_n = x.size(0);
  auto scores = torch::empty(
      {score_n, blocks, codes},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto cb_transposed = cached_transposed_codebook(cb);
  const float* xp = x.data_ptr<float>();
  const float* cp = cb_transposed.data_ptr<float>();
  float* scorep = scores.data_ptr<float>();
  at::parallel_for(0, score_n * blocks, 8, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t batch = item / blocks;
      const int64_t block = item - batch * blocks;
      const float* xv = xp + (batch * blocks + block) * dim;
      float* score = scorep + (batch * blocks + block) * codes;
      codebook_scores(xv, cp, score, codes, dim);
    }
  });

  auto out = torch::empty(
      {n, rows},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* op = out.data_ptr<float>();
  at::parallel_for(0, n * rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t batch = item / rows;
      const int64_t row = item - batch * rows;
      const int64_t score_batch = score_n == 1 ? 0 : batch;
      const float* score =
          scorep + score_batch * blocks * codes;
      op[item] = indices_u8
          ? lookup_sum(
                score,
                index_ptrs_u8[batch] + row * blocks,
                blocks,
                codes)
          : lookup_sum_u16(
                score,
                index_ptrs_u16[batch] + row * blocks,
                blocks,
                codes);
    }
  });
  return out;
}

inline float lookup_sum_packed(
    const float* score,
    const uint8_t* packed,
    int64_t start_index,
    int64_t blocks,
    int64_t codes,
    int64_t bits) {
  float sum = 0.0f;
  if (bits == 8) {
    for (int64_t block = 0; block < blocks; ++block) {
      sum += score[block * codes + packed[start_index + block]];
    }
    return sum;
  }
  if (bits == 16) {
    for (int64_t block = 0; block < blocks; ++block) {
      const int64_t index_offset = start_index + block;
      const int64_t offset = index_offset * 2;
      const uint16_t index = static_cast<uint16_t>(packed[offset]) |
          (static_cast<uint16_t>(packed[offset + 1]) << 8);
      sum += score[block * codes + index];
    }
    return sum;
  }
  if (bits == 12) {
    if (start_index % 2 == 0 && blocks % 2 == 0) {
      const int64_t start_byte = (start_index / 2) * 3;
      for (int64_t block = 0; block < blocks; block += 2) {
        const int64_t offset = start_byte + (block / 2) * 3;
        const uint16_t first =
            static_cast<uint16_t>(packed[offset]) |
            ((static_cast<uint16_t>(packed[offset + 1]) & 0x0f) << 8);
        const uint16_t second =
            (static_cast<uint16_t>(packed[offset + 1]) >> 4) |
            (static_cast<uint16_t>(packed[offset + 2]) << 4);
        sum += score[block * codes + first];
        sum += score[(block + 1) * codes + second];
      }
      return sum;
    }
    for (int64_t block = 0; block < blocks; ++block) {
      const int64_t index_offset = start_index + block;
      const int64_t offset = (index_offset / 2) * 3;
      const uint16_t index = index_offset % 2 == 0
          ? static_cast<uint16_t>(packed[offset]) |
              ((static_cast<uint16_t>(packed[offset + 1]) & 0x0f) << 8)
          : (static_cast<uint16_t>(packed[offset + 1]) >> 4) |
              (static_cast<uint16_t>(packed[offset + 2]) << 4);
      sum += score[block * codes + index];
    }
    return sum;
  }
  if (start_index % 4 == 0 && blocks % 4 == 0) {
    const int64_t start_byte = (start_index / 4) * 7;
    for (int64_t block = 0; block < blocks; block += 4) {
      const int64_t offset = start_byte + (block / 4) * 7;
      uint64_t word = 0;
      for (int byte = 0; byte < 7; ++byte) {
        word |= static_cast<uint64_t>(packed[offset + byte]) << (8 * byte);
      }
      sum += score[block * codes + (word & 0x3fff)];
      sum += score[(block + 1) * codes + ((word >> 14) & 0x3fff)];
      sum += score[(block + 2) * codes + ((word >> 28) & 0x3fff)];
      sum += score[(block + 3) * codes + ((word >> 42) & 0x3fff)];
    }
    return sum;
  }
  for (int64_t block = 0; block < blocks; ++block) {
    const int64_t index_offset = start_index + block;
    const int64_t offset = (index_offset / 4) * 7;
    uint64_t word = 0;
    for (int byte = 0; byte < 7; ++byte) {
      word |= static_cast<uint64_t>(packed[offset + byte]) << (8 * byte);
    }
    const int64_t shift = (index_offset % 4) * 14;
    sum += score[block * codes + ((word >> shift) & 0x3fff)];
  }
  return sum;
}

inline uint16_t read_packed_index(
    const uint8_t* packed,
    int64_t index_offset,
    int64_t bits) {
  if (bits == 8) {
    return packed[index_offset];
  }
  if (bits == 16) {
    const int64_t offset = index_offset * 2;
    return static_cast<uint16_t>(packed[offset]) |
        (static_cast<uint16_t>(packed[offset + 1]) << 8);
  }
  if (bits == 12) {
    const int64_t offset = (index_offset / 2) * 3;
    return index_offset % 2 == 0
        ? static_cast<uint16_t>(packed[offset]) |
              ((static_cast<uint16_t>(packed[offset + 1]) & 0x0f) << 8)
        : (static_cast<uint16_t>(packed[offset + 1]) >> 4) |
              (static_cast<uint16_t>(packed[offset + 2]) << 4);
  }
  const int64_t offset = (index_offset / 4) * 7;
  uint64_t word = 0;
  for (int byte = 0; byte < 7; ++byte) {
    word |= static_cast<uint64_t>(packed[offset + byte]) << (8 * byte);
  }
  return static_cast<uint16_t>(
      (word >> ((index_offset % 4) * 14)) & 0x3fff);
}

inline float direct_dot_packed(
    const float* input,
    const float* codebook,
    const uint8_t* packed,
    int64_t start_index,
    int64_t blocks,
    int64_t bits,
    int64_t dim) {
  float sum = 0.0f;
  const auto add_code = [&](int64_t block, uint16_t index) {
    const float* code = codebook + static_cast<int64_t>(index) * dim;
    const float* value = input + block * dim;
    for (int64_t lane = 0; lane < dim; ++lane) {
      sum += value[lane] * code[lane];
    }
  };
  if (bits == 8) {
    const uint8_t* row = packed + start_index;
    for (int64_t block = 0; block < blocks; ++block) {
      add_code(block, row[block]);
    }
    return sum;
  }
  if (bits == 12 && start_index % 2 == 0 && blocks % 2 == 0) {
    const uint8_t* row = packed + (start_index / 2) * 3;
    for (int64_t block = 0; block < blocks; block += 2) {
      const int64_t offset = (block / 2) * 3;
      const uint16_t first =
          static_cast<uint16_t>(row[offset]) |
          ((static_cast<uint16_t>(row[offset + 1]) & 0x0f) << 8);
      const uint16_t second =
          (static_cast<uint16_t>(row[offset + 1]) >> 4) |
          (static_cast<uint16_t>(row[offset + 2]) << 4);
      add_code(block, first);
      add_code(block + 1, second);
    }
    return sum;
  }
  if (bits == 14 && start_index % 4 == 0 && blocks % 4 == 0) {
    const uint8_t* row = packed + (start_index / 4) * 7;
    for (int64_t block = 0; block < blocks; block += 4) {
      const int64_t offset = (block / 4) * 7;
      uint64_t word = 0;
      for (int byte = 0; byte < 7; ++byte) {
        word |= static_cast<uint64_t>(row[offset + byte]) << (8 * byte);
      }
      add_code(block, static_cast<uint16_t>(word & 0x3fff));
      add_code(
          block + 1,
          static_cast<uint16_t>((word >> 14) & 0x3fff));
      add_code(
          block + 2,
          static_cast<uint16_t>((word >> 28) & 0x3fff));
      add_code(
          block + 3,
          static_cast<uint16_t>((word >> 42) & 0x3fff));
    }
    return sum;
  }
  for (int64_t block = 0; block < blocks; ++block) {
    add_code(
        block,
        read_packed_index(packed, start_index + block, bits));
  }
  return sum;
}

torch::Tensor vq_gemv_packed_list_cpu(
    torch::Tensor x_rows,
    std::vector<torch::Tensor> packed_list,
    torch::Tensor codebook,
    int64_t rows,
    int64_t blocks,
    int64_t bits,
    bool allow_direct) {
  TORCH_CHECK(!x_rows.is_cuda() && !codebook.is_cuda(),
              "packed VQ list operands must be on CPU");
  TORCH_CHECK(x_rows.dim() == 2, "x_rows must be [N|1,C]");
  TORCH_CHECK(codebook.dim() == 2, "codebook must be [K,D]");
  TORCH_CHECK(!packed_list.empty(), "packed VQ list cannot be empty");
  TORCH_CHECK(bits == 8 || bits == 12 || bits == 14 || bits == 16,
              "packed VQ width must be 8, 12, 14, or 16");
  TORCH_CHECK(rows > 0 && blocks > 0,
              "packed VQ rows and blocks must be positive");
  const int64_t n = static_cast<int64_t>(packed_list.size());
  const int64_t expected_bits = rows * blocks * bits;
  TORCH_CHECK(expected_bits % 8 == 0,
              "packed VQ payload must be byte aligned");
  const int64_t expected_bytes = expected_bits / 8;
  std::vector<torch::Tensor> payloads;
  std::vector<const uint8_t*> payload_ptrs;
  payloads.reserve(n);
  payload_ptrs.reserve(n);
  for (auto& packed : packed_list) {
    TORCH_CHECK(!packed.is_cuda() && packed.scalar_type() == at::kByte,
                "packed VQ payloads must be CPU uint8 tensors");
    TORCH_CHECK(packed.numel() == expected_bytes,
                "packed VQ payload length mismatch");
    payloads.push_back(packed.contiguous().reshape({-1}));
    payload_ptrs.push_back(payloads.back().data_ptr<uint8_t>());
  }
  TORCH_CHECK(x_rows.size(0) == 1 || x_rows.size(0) == n,
              "packed VQ input batch must be 1 or expert count");
  const int64_t codes = codebook.size(0);
  const int64_t dim = codebook.size(1);
  TORCH_CHECK(codes <= (int64_t{1} << bits),
              "packed width cannot represent every codebook entry");
  TORCH_CHECK(x_rows.size(1) == blocks * dim,
              "packed VQ input width mismatch");

  auto x = x_rows.to(torch::kFloat32).contiguous();
  auto cb = codebook.to(torch::kFloat32).contiguous();
  const int64_t score_n = x.size(0);
  const bool use_direct =
      allow_direct &&
      score_n == n &&
      rows * dim < codes * dim + rows;
  if (use_direct) {
    auto out = torch::empty(
        {n, rows},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    const float* xp = x.data_ptr<float>();
    const float* cp = cb.data_ptr<float>();
    float* op = out.data_ptr<float>();
    at::parallel_for(0, n * rows, 1, [&](int64_t begin, int64_t end) {
      for (int64_t item = begin; item < end; ++item) {
        const int64_t batch = item / rows;
        const int64_t row = item - batch * rows;
        const uint8_t* payload = payload_ptrs[batch];
        const float* input = xp + batch * blocks * dim;
        op[item] = direct_dot_packed(
            input,
            cp,
            payload,
            row * blocks,
            blocks,
            bits,
            dim);
      }
    });
    return out;
  }
  auto scores = torch::empty(
      {score_n, blocks, codes},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto cb_transposed = cached_transposed_codebook(cb);
  const float* xp = x.data_ptr<float>();
  const float* cp = cb_transposed.data_ptr<float>();
  float* scorep = scores.data_ptr<float>();
  at::parallel_for(0, score_n * blocks, 8, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t batch = item / blocks;
      const int64_t block = item - batch * blocks;
      const float* xv = xp + (batch * blocks + block) * dim;
      float* score = scorep + (batch * blocks + block) * codes;
      codebook_scores(xv, cp, score, codes, dim);
    }
  });

  auto out = torch::empty(
      {n, rows},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* op = out.data_ptr<float>();
  at::parallel_for(0, n * rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t batch = item / rows;
      const int64_t row = item - batch * rows;
      const int64_t score_batch = score_n == 1 ? 0 : batch;
      op[item] = lookup_sum_packed(
          scorep + score_batch * blocks * codes,
          payload_ptrs[batch],
          row * blocks,
          blocks,
          codes,
          bits);
    }
  });
  return out;
}

torch::Tensor moe_packed_topk_cpu(
    torch::Tensor x_row,
    std::vector<torch::Tensor> gu_payload_list,
    std::vector<torch::Tensor> gu_codebook_list,
    std::vector<int64_t> gu_rows,
    std::vector<int64_t> gu_blocks,
    std::vector<int64_t> gu_bits,
    std::vector<torch::Tensor> dn_payload_list,
    std::vector<torch::Tensor> dn_codebook_list,
    std::vector<int64_t> dn_rows,
    std::vector<int64_t> dn_blocks,
    std::vector<int64_t> dn_bits,
    torch::Tensor route_weights,
    double limit,
    std::string activation,
    double beta,
    double linear_beta,
    torch::Tensor workspace,
    torch::Tensor result) {
  const int64_t experts =
      static_cast<int64_t>(gu_payload_list.size());
  TORCH_CHECK(
      experts > 0 && experts <= 16 &&
          static_cast<int64_t>(gu_codebook_list.size()) == experts &&
          static_cast<int64_t>(gu_rows.size()) == experts &&
          static_cast<int64_t>(gu_blocks.size()) == experts &&
          static_cast<int64_t>(gu_bits.size()) == experts &&
          static_cast<int64_t>(dn_payload_list.size()) == experts &&
          static_cast<int64_t>(dn_codebook_list.size()) == experts &&
          static_cast<int64_t>(dn_rows.size()) == experts &&
          static_cast<int64_t>(dn_blocks.size()) == experts &&
          static_cast<int64_t>(dn_bits.size()) == experts,
      "packed Top-K MoE operand counts must match");
  TORCH_CHECK(
      !x_row.is_cuda() && x_row.scalar_type() == at::kFloat &&
          x_row.dim() == 2 && x_row.size(0) == 1 &&
          x_row.is_contiguous(),
      "packed Top-K MoE requires one contiguous CPU float32 row");
  TORCH_CHECK(
      !route_weights.is_cuda() &&
          route_weights.scalar_type() == at::kFloat &&
          route_weights.numel() == experts &&
          route_weights.is_contiguous(),
      "packed Top-K MoE route weights must be contiguous CPU float32");
  TORCH_CHECK(
      !workspace.is_cuda() && workspace.scalar_type() == at::kFloat &&
          workspace.dim() == 1 && workspace.is_contiguous(),
      "packed Top-K MoE workspace must be contiguous CPU float32");
  TORCH_CHECK(
      !result.is_cuda() && result.scalar_type() == at::kFloat &&
          result.dim() == 1 && result.is_contiguous(),
      "packed Top-K MoE result must be contiguous CPU float32");
  TORCH_CHECK(
      activation == "silu" || activation == "swiglu" ||
          activation == "situ",
      "packed Top-K MoE activation must be silu, swiglu, or situ");

  const int64_t hidden = x_row.size(1);
  int64_t intermediate = -1;
  std::vector<const uint8_t*> gu_payload_ptrs(experts);
  std::vector<const uint8_t*> dn_payload_ptrs(experts);
  std::vector<const float*> gu_codebook_ptrs(experts);
  std::vector<const float*> dn_codebook_ptrs(experts);
  std::vector<int64_t> gu_codes(experts);
  std::vector<int64_t> gu_dims(experts);
  std::vector<int64_t> dn_codes(experts);
  std::vector<int64_t> dn_dims(experts);
  std::vector<torch::Tensor> gu_transposed(experts);
  std::vector<torch::Tensor> dn_transposed(experts);
  std::vector<bool> dn_direct(experts);

  for (int64_t expert = 0; expert < experts; ++expert) {
    const auto& gu_payload = gu_payload_list[expert];
    const auto& gu_codebook = gu_codebook_list[expert];
    const auto& dn_payload = dn_payload_list[expert];
    const auto& dn_codebook = dn_codebook_list[expert];
    TORCH_CHECK(
        !gu_payload.is_cuda() &&
            gu_payload.scalar_type() == at::kByte &&
            gu_payload.is_contiguous() &&
            !dn_payload.is_cuda() &&
            dn_payload.scalar_type() == at::kByte &&
            dn_payload.is_contiguous(),
        "packed Top-K MoE payloads must be contiguous CPU uint8");
    TORCH_CHECK(
        !gu_codebook.is_cuda() &&
            gu_codebook.scalar_type() == at::kFloat &&
            gu_codebook.dim() == 2 && gu_codebook.is_contiguous() &&
            !dn_codebook.is_cuda() &&
            dn_codebook.scalar_type() == at::kFloat &&
            dn_codebook.dim() == 2 && dn_codebook.is_contiguous(),
        "packed Top-K MoE codebooks must be contiguous CPU float32");
    TORCH_CHECK(
        gu_bits[expert] == 8 || gu_bits[expert] == 12 ||
            gu_bits[expert] == 14 || gu_bits[expert] == 16,
        "unsupported packed GU width");
    TORCH_CHECK(
        dn_bits[expert] == 8 || dn_bits[expert] == 12 ||
            dn_bits[expert] == 14 || dn_bits[expert] == 16,
        "unsupported packed Down width");
    const int64_t gu_dim = gu_codebook.size(1);
    const int64_t dn_dim = dn_codebook.size(1);
    const int64_t current_intermediate =
        dn_blocks[expert] * dn_dim;
    if (intermediate < 0) {
      intermediate = current_intermediate;
    }
    TORCH_CHECK(
        current_intermediate == intermediate &&
            gu_blocks[expert] * gu_dim == hidden &&
            gu_rows[expert] == 2 * intermediate &&
            dn_rows[expert] == hidden,
        "packed Top-K MoE logical matrix shapes do not match");
    TORCH_CHECK(
        gu_payload.numel() * 8 ==
                gu_rows[expert] * gu_blocks[expert] *
                    gu_bits[expert] &&
            dn_payload.numel() * 8 ==
                dn_rows[expert] * dn_blocks[expert] *
                    dn_bits[expert],
        "packed Top-K MoE payload length mismatch");
    TORCH_CHECK(
        gu_codebook.size(0) <=
                (int64_t{1} << gu_bits[expert]) &&
            dn_codebook.size(0) <=
                (int64_t{1} << dn_bits[expert]),
        "packed Top-K MoE bit width cannot represent codebook");
    gu_payload_ptrs[expert] = gu_payload.data_ptr<uint8_t>();
    dn_payload_ptrs[expert] = dn_payload.data_ptr<uint8_t>();
    gu_codebook_ptrs[expert] = gu_codebook.data_ptr<float>();
    dn_codebook_ptrs[expert] = dn_codebook.data_ptr<float>();
    gu_codes[expert] = gu_codebook.size(0);
    gu_dims[expert] = gu_dim;
    dn_codes[expert] = dn_codebook.size(0);
    dn_dims[expert] = dn_dim;
    gu_transposed[expert] =
        cached_transposed_codebook(gu_codebook);
    dn_direct[expert] =
        dn_rows[expert] * dn_dim <
        dn_codes[expert] * dn_dim + dn_rows[expert];
    if (!dn_direct[expert]) {
      dn_transposed[expert] =
          cached_transposed_codebook(dn_codebook);
    }
  }
  TORCH_CHECK(
      intermediate > 0 && result.numel() >= hidden,
      "packed Top-K MoE output size mismatch");

  // The input is shared by all selected experts.  Experts of the same tier
  // normally share a codebook, so calculate one GU score table per unique
  // codebook instead of repeating it for every routed expert.
  std::vector<int64_t> gu_unique_representatives;
  std::vector<int64_t> gu_unique_for_expert(experts);
  for (int64_t expert = 0; expert < experts; ++expert) {
    int64_t unique = -1;
    for (int64_t candidate = 0;
         candidate <
         static_cast<int64_t>(gu_unique_representatives.size());
         ++candidate) {
      const int64_t other = gu_unique_representatives[candidate];
      if (gu_codebook_ptrs[expert] == gu_codebook_ptrs[other] &&
          gu_blocks[expert] == gu_blocks[other] &&
          gu_codes[expert] == gu_codes[other] &&
          gu_dims[expert] == gu_dims[other]) {
        unique = candidate;
        break;
      }
    }
    if (unique < 0) {
      unique =
          static_cast<int64_t>(gu_unique_representatives.size());
      gu_unique_representatives.push_back(expert);
    }
    gu_unique_for_expert[expert] = unique;
  }

  std::vector<int64_t> gu_score_offsets(
      gu_unique_representatives.size());
  int64_t gu_score_count = 0;
  for (int64_t unique = 0;
       unique <
       static_cast<int64_t>(gu_unique_representatives.size());
       ++unique) {
    const int64_t expert = gu_unique_representatives[unique];
    gu_score_offsets[unique] = gu_score_count;
    gu_score_count += gu_blocks[expert] * gu_codes[expert];
  }
  std::vector<int64_t> dn_score_offsets(experts, -1);
  std::vector<int64_t> dn_score_experts;
  std::vector<int64_t> dn_block_offsets(1, 0);
  int64_t dn_score_count = 0;
  for (int64_t expert = 0; expert < experts; ++expert) {
    if (dn_direct[expert]) {
      continue;
    }
    dn_score_offsets[expert] = dn_score_count;
    dn_score_count += dn_blocks[expert] * dn_codes[expert];
    dn_score_experts.push_back(expert);
    dn_block_offsets.push_back(
        dn_block_offsets.back() + dn_blocks[expert]);
  }

  const int64_t gate_offset = gu_score_count;
  const int64_t up_offset =
      gate_offset + experts * intermediate;
  const int64_t activation_offset =
      up_offset + experts * intermediate;
  const int64_t activation_temp_offset =
      activation_offset + experts * intermediate;
  const int64_t dn_score_offset =
      activation_temp_offset + experts * intermediate;
  const int64_t dn_partial_offset =
      dn_score_offset + dn_score_count;
  const int64_t required =
      dn_partial_offset + experts * hidden;
  TORCH_CHECK(
      workspace.numel() >= required,
      "packed Top-K MoE workspace is too small: ",
      workspace.numel(), " < ", required);

  const float* xp = x_row.data_ptr<float>();
  const float* routep = route_weights.data_ptr<float>();
  float* workspacep = workspace.data_ptr<float>();
  float* gu_scorep = workspacep;
  float* gatep = workspacep + gate_offset;
  float* upp = workspacep + up_offset;
  float* activationp = workspacep + activation_offset;
  float* dn_scorep = workspacep + dn_score_offset;
  float* dn_partialp = workspacep + dn_partial_offset;
  float* resultp = result.data_ptr<float>();
  const float activation_limit = static_cast<float>(limit);
  const float situ_beta = static_cast<float>(beta);
  const float situ_linear_beta = static_cast<float>(linear_beta);
  double gu_score_elapsed = 0.0;
  double gu_lookup_elapsed = 0.0;
  double phase_times[5];

  // ATen owns one persistent worker pool for the process.  The whole MoE is
  // still one registered native call.  Score and consume one shared codebook
  // at a time so its 1--30 MiB score table remains hot in LLC; materializing
  // every tier before lookup evicts the first tiers on mixed Top-16 routes.
  for (int64_t unique = 0;
       unique <
       static_cast<int64_t>(gu_unique_representatives.size());
       ++unique) {
    const int64_t expert = gu_unique_representatives[unique];
    const double score_started = wall_seconds();
    at::parallel_for(
        0, gu_blocks[expert], 1,
        [&](int64_t begin, int64_t end) {
      for (int64_t block = begin; block < end; ++block) {
      const int64_t blocks = gu_blocks[expert];
      const int64_t codes = gu_codes[expert];
      const int64_t dim = gu_dims[expert];
      const float* codebook =
          gu_transposed[expert].data_ptr<float>();
      const float* xv = xp + block * dim;
      float* score =
          gu_scorep + gu_score_offsets[unique] + block * codes;
      codebook_scores(xv, codebook, score, codes, dim);
      }
    });
    gu_score_elapsed += wall_seconds() - score_started;
    std::vector<int64_t> members;
    for (int64_t candidate = 0; candidate < experts; ++candidate) {
      if (gu_unique_for_expert[candidate] == unique) {
        members.push_back(candidate);
      }
    }
    const double lookup_started = wall_seconds();
    at::parallel_for(
        0,
        static_cast<int64_t>(members.size()) * intermediate,
        1,
        [&](int64_t begin, int64_t end) {
      for (int64_t item = begin; item < end; ++item) {
      const int64_t expert =
          members[item / intermediate];
      const int64_t row = item % intermediate;
      const int64_t blocks = gu_blocks[expert];
      const int64_t codes = gu_codes[expert];
      const uint8_t* payload = gu_payload_ptrs[expert];
      const float* score = gu_scorep + gu_score_offsets[unique];
      float gate = lookup_sum_packed(
          score,
          payload,
          row * blocks,
          blocks,
          codes,
          gu_bits[expert]);
      float up = lookup_sum_packed(
          score,
          payload,
          (intermediate + row) * blocks,
          blocks,
          codes,
          gu_bits[expert]);
      if (activation_limit != 0.0f) {
        gate = std::min(gate, activation_limit);
        up = std::max(
            -activation_limit, std::min(up, activation_limit));
      }
      const int64_t activation_item =
          expert * intermediate + row;
      gatep[activation_item] = gate;
      upp[activation_item] = up;
      }
    });
    gu_lookup_elapsed += wall_seconds() - lookup_started;
  }
  phase_times[0] = wall_seconds();

  const int64_t activation_count = experts * intermediate;
  auto gate_values =
      workspace.narrow(0, gate_offset, activation_count);
  auto up_values =
      workspace.narrow(0, up_offset, activation_count);
  auto activation_values =
      workspace.narrow(0, activation_offset, activation_count);
  auto activation_temp =
      workspace.narrow(0, activation_temp_offset, activation_count);
  // ATen's persistent CPU pool supplies vectorized exp/tanh.  Scalar libm
  // calls here cost more than all scheduling saved by the fusion on Top-16.
  if (activation == "situ") {
    activation_values.copy_(gate_values);
    activation_values.div_(situ_beta);
    activation_values.tanh_();
    activation_values.mul_(situ_beta);
    activation_temp.copy_(gate_values);
    activation_temp.sigmoid_();
    activation_values.mul_(activation_temp);
    if (situ_linear_beta > 0.0f) {
      up_values.div_(situ_linear_beta);
      up_values.tanh_();
      up_values.mul_(situ_linear_beta);
    }
    activation_values.mul_(up_values);
  } else {
    activation_temp.copy_(gate_values);
    activation_temp.sigmoid_();
    activation_values.copy_(gate_values);
    activation_values.mul_(activation_temp);
    activation_values.mul_(up_values);
  }
  phase_times[1] = wall_seconds();

  at::parallel_for(
      0, dn_block_offsets.back(), 1,
      [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t selected = static_cast<int64_t>(
          std::upper_bound(
              dn_block_offsets.begin(),
              dn_block_offsets.end(),
              item) -
          dn_block_offsets.begin() - 1);
      const int64_t expert = dn_score_experts[selected];
      const int64_t block = item - dn_block_offsets[selected];
      const int64_t codes = dn_codes[expert];
      const int64_t dim = dn_dims[expert];
      const float* xv =
          activationp + expert * intermediate + block * dim;
      const float* codebook =
          dn_transposed[expert].data_ptr<float>();
      float* score =
          dn_scorep + dn_score_offsets[expert] + block * codes;
      codebook_scores(xv, codebook, score, codes, dim);
    }
  });
  phase_times[2] = wall_seconds();

  at::parallel_for(
      0, experts * hidden, 1,
      [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t expert = item / hidden;
      const int64_t row = item - expert * hidden;
      const int64_t blocks = dn_blocks[expert];
      float value;
      if (dn_direct[expert]) {
        value = direct_dot_packed(
            activationp + expert * intermediate,
            dn_codebook_ptrs[expert],
            dn_payload_ptrs[expert],
            row * blocks,
            blocks,
            dn_bits[expert],
            dn_dims[expert]);
      } else {
        value = lookup_sum_packed(
            dn_scorep + dn_score_offsets[expert],
            dn_payload_ptrs[expert],
            row * blocks,
            blocks,
            dn_codes[expert],
            dn_bits[expert]);
      }
      dn_partialp[item] = value * routep[expert];
    }
  });
  phase_times[3] = wall_seconds();

  at::parallel_for(0, hidden, 1, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      float total = 0.0f;
      for (int64_t expert = 0; expert < experts; ++expert) {
        total += dn_partialp[expert * hidden + row];
      }
      resultp[row] = total;
    }
  });
  phase_times[4] = wall_seconds();
  packed_moe_phase_seconds[0] += gu_score_elapsed;
  packed_moe_phase_seconds[1] += gu_lookup_elapsed;
  for (int phase = 0; phase < 4; ++phase) {
    packed_moe_phase_seconds[phase + 2] +=
        phase_times[phase + 1] - phase_times[phase];
  }
  ++packed_moe_phase_calls;
  return result.narrow(0, 0, hidden);
}

void reset_packed_moe_phase_profile_cpu() {
  std::fill(
      std::begin(packed_moe_phase_seconds),
      std::end(packed_moe_phase_seconds),
      0.0);
  packed_moe_phase_calls = 0;
}

std::vector<double> packed_moe_phase_profile_cpu() {
  return {
      static_cast<double>(packed_moe_phase_calls),
      packed_moe_phase_seconds[0],
      packed_moe_phase_seconds[1],
      packed_moe_phase_seconds[2],
      packed_moe_phase_seconds[3],
      packed_moe_phase_seconds[4],
      packed_moe_phase_seconds[5],
  };
}

torch::Tensor kda_recurrent_cpu(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor gate,
    torch::Tensor beta,
    torch::Tensor a_log,
    torch::Tensor dt_bias,
    torch::Tensor state,
    torch::Tensor workspace,
    torch::Tensor output,
    double lower_bound) {
  TORCH_CHECK(
      !query.is_cuda() && query.dim() == 2 &&
          query.is_contiguous() && key.sizes() == query.sizes() &&
          gate.sizes() == query.sizes() && key.is_contiguous() &&
          gate.is_contiguous(),
      "CPU KDA query/key/gate must be contiguous [heads, key_dim]");
  TORCH_CHECK(
      query.scalar_type() == at::kFloat ||
          query.scalar_type() == at::kBFloat16,
      "CPU KDA inputs must be float32 or bfloat16");
  TORCH_CHECK(
      key.scalar_type() == query.scalar_type() &&
          gate.scalar_type() == query.scalar_type() &&
          value.scalar_type() == query.scalar_type() &&
          output.scalar_type() == query.scalar_type(),
      "CPU KDA input and output dtypes must match");
  const int64_t heads = query.size(0);
  const int64_t key_dim = query.size(1);
  TORCH_CHECK(
      value.dim() == 2 && value.size(0) == heads &&
          value.is_contiguous(),
      "CPU KDA value must be contiguous [heads, value_dim]");
  const int64_t value_dim = value.size(1);
  TORCH_CHECK(
      !state.is_cuda() && state.scalar_type() == at::kFloat &&
          state.sizes() ==
              torch::IntArrayRef({heads, value_dim, key_dim}) &&
          state.is_contiguous(),
      "CPU KDA state must be contiguous float32 [heads,value,key]");
  TORCH_CHECK(
      !workspace.is_cuda() &&
          workspace.scalar_type() == at::kFloat &&
          workspace.numel() >= 3 * heads * key_dim &&
          workspace.is_contiguous(),
      "CPU KDA workspace must hold normalized Q/K and decay");
  TORCH_CHECK(
      output.sizes() == value.sizes() && output.is_contiguous(),
      "CPU KDA output shape mismatch");
  TORCH_CHECK(
      beta.numel() >= heads && a_log.numel() >= heads &&
          dt_bias.numel() >= heads * key_dim,
      "CPU KDA beta/A_log/dt_bias shape mismatch");
  TORCH_CHECK(
      (beta.scalar_type() == at::kFloat ||
       beta.scalar_type() == at::kBFloat16) &&
          (a_log.scalar_type() == at::kFloat ||
           a_log.scalar_type() == at::kBFloat16) &&
          (dt_bias.scalar_type() == at::kFloat ||
           dt_bias.scalar_type() == at::kBFloat16),
      "CPU KDA scalar parameters must be float32 or bfloat16");

  const bool bf16 = query.scalar_type() == at::kBFloat16;
  const float* query_f =
      bf16 ? nullptr : query.data_ptr<float>();
  const float* key_f =
      bf16 ? nullptr : key.data_ptr<float>();
  const float* value_f =
      bf16 ? nullptr : value.data_ptr<float>();
  const float* gate_f =
      bf16 ? nullptr : gate.data_ptr<float>();
  const at::BFloat16* query_b =
      bf16 ? query.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* key_b =
      bf16 ? key.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* value_b =
      bf16 ? value.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* gate_b =
      bf16 ? gate.data_ptr<at::BFloat16>() : nullptr;
  const bool beta_bf16 = beta.scalar_type() == at::kBFloat16;
  const bool a_bf16 = a_log.scalar_type() == at::kBFloat16;
  const bool dt_bf16 = dt_bias.scalar_type() == at::kBFloat16;
  const float* betap =
      beta_bf16 ? nullptr : beta.data_ptr<float>();
  const float* ap =
      a_bf16 ? nullptr : a_log.data_ptr<float>();
  const float* dtp =
      dt_bf16 ? nullptr : dt_bias.data_ptr<float>();
  const at::BFloat16* betab =
      beta_bf16 ? beta.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* ab =
      a_bf16 ? a_log.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* dtb =
      dt_bf16 ? dt_bias.data_ptr<at::BFloat16>() : nullptr;

  float* workspacep = workspace.data_ptr<float>();
  float* query_norm = workspacep;
  float* key_norm = query_norm + heads * key_dim;
  float* decay = key_norm + heads * key_dim;
  float* statep = state.data_ptr<float>();
  float* output_f =
      bf16 ? nullptr : output.data_ptr<float>();
  at::BFloat16* output_b =
      bf16 ? output.data_ptr<at::BFloat16>() : nullptr;
  const float lower = static_cast<float>(lower_bound);

  at::parallel_for(0, heads, 1, [&](int64_t begin, int64_t end) {
    for (int64_t head = begin; head < end; ++head) {
      const int64_t base = head * key_dim;
      float query_square = 0.0f;
      float key_square = 0.0f;
      for (int64_t lane = 0; lane < key_dim; ++lane) {
        const int64_t index = base + lane;
        const float q =
            bf16 ? static_cast<float>(query_b[index]) : query_f[index];
        const float k =
            bf16 ? static_cast<float>(key_b[index]) : key_f[index];
        query_square += q * q;
        key_square += k * k;
      }
      const float query_inverse =
          1.0f / std::max(std::sqrt(query_square), 1.0e-6f);
      const float key_inverse =
          1.0f / std::max(std::sqrt(key_square), 1.0e-6f);
      const float a = std::exp(
          a_bf16 ? static_cast<float>(ab[head]) : ap[head]);
      for (int64_t lane = 0; lane < key_dim; ++lane) {
        const int64_t index = base + lane;
        const float q =
            bf16 ? static_cast<float>(query_b[index]) : query_f[index];
        const float k =
            bf16 ? static_cast<float>(key_b[index]) : key_f[index];
        const float g =
            (bf16 ? static_cast<float>(gate_b[index]) : gate_f[index]) +
            (dt_bf16 ? static_cast<float>(dtb[index]) : dtp[index]);
        query_norm[index] = q * query_inverse;
        key_norm[index] = k * key_inverse;
        decay[index] = std::exp(
            lower / (1.0f + std::exp(-a * g)));
      }
    }
  });

  const float output_scale =
      1.0f / std::sqrt(static_cast<float>(key_dim));
  at::parallel_for(
      0, heads * value_dim, 1,
      [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t head = item / value_dim;
      const int64_t row = item - head * value_dim;
      float* state_row =
          statep + (head * value_dim + row) * key_dim;
      const float* key_row = key_norm + head * key_dim;
      const float* query_row = query_norm + head * key_dim;
      const float* decay_row = decay + head * key_dim;
      float prediction = 0.0f;
      int64_t lane = 0;
#if defined(__AVX512F__)
      __m512 prediction_vector = _mm512_setzero_ps();
      for (; lane + 16 <= key_dim; lane += 16) {
        const __m512 current = _mm512_loadu_ps(state_row + lane);
        const __m512 decayed = _mm512_mul_ps(
            current, _mm512_loadu_ps(decay_row + lane));
        _mm512_storeu_ps(state_row + lane, decayed);
        prediction_vector = _mm512_fmadd_ps(
            decayed,
            _mm512_loadu_ps(key_row + lane),
            prediction_vector);
      }
      prediction = _mm512_reduce_add_ps(prediction_vector);
#endif
      for (; lane < key_dim; ++lane) {
        state_row[lane] *= decay_row[lane];
        prediction += state_row[lane] * key_row[lane];
      }
      const int64_t value_index = head * value_dim + row;
      const float current_value =
          bf16
          ? static_cast<float>(value_b[value_index])
          : value_f[value_index];
      const float beta_value =
          beta_bf16
          ? static_cast<float>(betab[head])
          : betap[head];
      const float delta =
          (current_value - prediction) /
          (1.0f + std::exp(-beta_value));
      float output_value = 0.0f;
      lane = 0;
#if defined(__AVX512F__)
      __m512 output_vector = _mm512_setzero_ps();
      const __m512 delta_vector = _mm512_set1_ps(delta);
      for (; lane + 16 <= key_dim; lane += 16) {
        const __m512 updated = _mm512_fmadd_ps(
            delta_vector,
            _mm512_loadu_ps(key_row + lane),
            _mm512_loadu_ps(state_row + lane));
        _mm512_storeu_ps(state_row + lane, updated);
        output_vector = _mm512_fmadd_ps(
            updated,
            _mm512_loadu_ps(query_row + lane),
            output_vector);
      }
      output_value = _mm512_reduce_add_ps(output_vector);
#endif
      for (; lane < key_dim; ++lane) {
        state_row[lane] += delta * key_row[lane];
        output_value += state_row[lane] * query_row[lane];
      }
      output_value *= output_scale;
      if (bf16) {
        output_b[value_index] = at::BFloat16(output_value);
      } else {
        output_f[value_index] = output_value;
      }
    }
  });
  return output;
}

bool short_conv3_cpu(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    std::vector<torch::Tensor> states,
    std::vector<torch::Tensor> weights) {
  TORCH_CHECK(
      !query.is_cuda() && query.dim() == 1 &&
          query.is_contiguous() && key.sizes() == query.sizes() &&
          value.sizes() == query.sizes() && key.is_contiguous() &&
          value.is_contiguous(),
      "CPU short-conv inputs must be contiguous flattened tensors");
  TORCH_CHECK(
      query.scalar_type() == at::kFloat ||
          query.scalar_type() == at::kBFloat16,
      "CPU short-conv inputs must be float32 or bfloat16");
  TORCH_CHECK(
      key.scalar_type() == query.scalar_type() &&
          value.scalar_type() == query.scalar_type() &&
          states.size() == 3 && weights.size() == 3,
      "CPU short-conv operand count or dtype mismatch");
  const int64_t channels = query.numel();
  std::vector<torch::Tensor> inputs = {query, key, value};
  int64_t history = -1;
  for (int stream = 0; stream < 3; ++stream) {
    TORCH_CHECK(
        !states[stream].is_cuda() &&
            states[stream].scalar_type() == query.scalar_type() &&
            states[stream].dim() == 2 &&
            states[stream].size(0) == channels &&
            states[stream].is_contiguous(),
        "CPU short-conv state shape mismatch");
    if (history < 0) {
      history = states[stream].size(1);
    }
    TORCH_CHECK(
        states[stream].size(1) == history &&
            !weights[stream].is_cuda() &&
            weights[stream].numel() == channels * (history + 1) &&
            weights[stream].is_contiguous() &&
            (weights[stream].scalar_type() == at::kFloat ||
             weights[stream].scalar_type() == at::kBFloat16),
        "CPU short-conv weight shape mismatch");
  }
  const bool bf16 = query.scalar_type() == at::kBFloat16;
  at::parallel_for(
      0, 3 * channels, 1,
      [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t stream = item / channels;
      const int64_t channel = item - stream * channels;
      auto& input = inputs[stream];
      auto& state = states[stream];
      auto& weight = weights[stream];
      const bool weight_bf16 =
          weight.scalar_type() == at::kBFloat16;
      float* state_f =
          bf16 ? nullptr : state.data_ptr<float>();
      at::BFloat16* state_b =
          bf16 ? state.data_ptr<at::BFloat16>() : nullptr;
      float* input_f =
          bf16 ? nullptr : input.data_ptr<float>();
      at::BFloat16* input_b =
          bf16 ? input.data_ptr<at::BFloat16>() : nullptr;
      const float* weight_f =
          weight_bf16 ? nullptr : weight.data_ptr<float>();
      const at::BFloat16* weight_b =
          weight_bf16
          ? weight.data_ptr<at::BFloat16>()
          : nullptr;
      const int64_t state_base = channel * history;
      const int64_t weight_base = channel * (history + 1);
      const float current =
          bf16
          ? static_cast<float>(input_b[channel])
          : input_f[channel];
      float sum = 0.0f;
      for (int64_t offset = 0; offset < history; ++offset) {
        const float previous =
            bf16
            ? static_cast<float>(state_b[state_base + offset])
            : state_f[state_base + offset];
        const float coefficient =
            weight_bf16
            ? static_cast<float>(weight_b[weight_base + offset])
            : weight_f[weight_base + offset];
        sum += previous * coefficient;
        if (offset + 1 < history) {
          if (bf16) {
            state_b[state_base + offset] =
                state_b[state_base + offset + 1];
          } else {
            state_f[state_base + offset] =
                state_f[state_base + offset + 1];
          }
        }
      }
      const float final_coefficient =
          weight_bf16
          ? static_cast<float>(weight_b[weight_base + history])
          : weight_f[weight_base + history];
      sum += current * final_coefficient;
      if (history > 0) {
        if (bf16) {
          state_b[state_base + history - 1] =
              at::BFloat16(current);
        } else {
          state_f[state_base + history - 1] = current;
        }
      }
      const float activated = sum / (1.0f + std::exp(-sum));
      if (bf16) {
        input_b[channel] = at::BFloat16(activated);
      } else {
        input_f[channel] = activated;
      }
    }
  });
  return true;
}

torch::Tensor gated_rmsnorm_cpu(
    torch::Tensor value,
    torch::Tensor gate,
    torch::Tensor weight,
    torch::Tensor output,
    double eps) {
  TORCH_CHECK(
      !value.is_cuda() && value.dim() == 2 &&
          value.is_contiguous() && gate.sizes() == value.sizes() &&
          gate.is_contiguous() && output.sizes() == value.sizes() &&
          output.is_contiguous(),
      "CPU gated RMSNorm operands must be contiguous [rows,dim]");
  TORCH_CHECK(
      value.scalar_type() == at::kFloat ||
          value.scalar_type() == at::kBFloat16,
      "CPU gated RMSNorm values must be float32 or bfloat16");
  TORCH_CHECK(
      gate.scalar_type() == value.scalar_type() &&
          output.scalar_type() == value.scalar_type() &&
          !weight.is_cuda() && weight.is_contiguous() &&
          weight.numel() == value.size(1) &&
          (weight.scalar_type() == at::kFloat ||
           weight.scalar_type() == at::kBFloat16),
      "CPU gated RMSNorm dtype or weight shape mismatch");
  const int64_t rows = value.size(0);
  const int64_t dim = value.size(1);
  const bool bf16 = value.scalar_type() == at::kBFloat16;
  const bool weight_bf16 =
      weight.scalar_type() == at::kBFloat16;
  const float* value_f =
      bf16 ? nullptr : value.data_ptr<float>();
  const float* gate_f =
      bf16 ? nullptr : gate.data_ptr<float>();
  float* output_f =
      bf16 ? nullptr : output.data_ptr<float>();
  const at::BFloat16* value_b =
      bf16 ? value.data_ptr<at::BFloat16>() : nullptr;
  const at::BFloat16* gate_b =
      bf16 ? gate.data_ptr<at::BFloat16>() : nullptr;
  at::BFloat16* output_b =
      bf16 ? output.data_ptr<at::BFloat16>() : nullptr;
  const float* weight_f =
      weight_bf16 ? nullptr : weight.data_ptr<float>();
  const at::BFloat16* weight_b =
      weight_bf16 ? weight.data_ptr<at::BFloat16>() : nullptr;
  const float epsilon = static_cast<float>(eps);
  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const int64_t base = row * dim;
      float square = 0.0f;
      for (int64_t lane = 0; lane < dim; ++lane) {
        const float current =
            bf16
            ? static_cast<float>(value_b[base + lane])
            : value_f[base + lane];
        square += current * current;
      }
      const float inverse =
          1.0f / std::sqrt(square / static_cast<float>(dim) + epsilon);
      for (int64_t lane = 0; lane < dim; ++lane) {
        const float current =
            bf16
            ? static_cast<float>(value_b[base + lane])
            : value_f[base + lane];
        const float gate_value =
            bf16
            ? static_cast<float>(gate_b[base + lane])
            : gate_f[base + lane];
        const float scale =
            weight_bf16
            ? static_cast<float>(weight_b[lane])
            : weight_f[lane];
        const float normalized =
            current * inverse * scale /
            (1.0f + std::exp(-gate_value));
        if (bf16) {
          output_b[base + lane] = at::BFloat16(normalized);
        } else {
          output_f[base + lane] = normalized;
        }
      }
    }
  });
  return output;
}

torch::Tensor moe_mixed_cpu(
    torch::Tensor x_row,
    std::vector<torch::Tensor> gu_index_list,
    std::vector<torch::Tensor> gu_codebook_list,
    std::vector<torch::Tensor> dn_index_list,
    std::vector<torch::Tensor> dn_codebook_list,
    torch::Tensor route_weights,
    torch::Tensor shared_w1_q,
    torch::Tensor shared_w1_s,
    torch::Tensor shared_w3_q,
    torch::Tensor shared_w3_s,
    torch::Tensor shared_w2_q,
    torch::Tensor shared_w2_s,
    int64_t group_size,
    double limit,
    bool indices_transposed) {
  const int64_t experts = static_cast<int64_t>(gu_index_list.size());
  const char* vq_int8_mode = std::getenv("TPQ_CPU_VQ_INT8");
  const bool use_vq_int8 =
      indices_transposed && vq_int8_mode != nullptr &&
      vq_int8_mode[0] != '\0' && vq_int8_mode[0] != '0';
  const int64_t vq_chunks = use_vq_int8 ? 32 : 0;
  TORCH_CHECK(
      experts > 0 && experts == static_cast<int64_t>(gu_codebook_list.size()) &&
          experts == static_cast<int64_t>(dn_index_list.size()) &&
          experts == static_cast<int64_t>(dn_codebook_list.size()),
      "routed expert operand counts must match");
  TORCH_CHECK(
      !x_row.is_cuda() && x_row.dim() == 2 && x_row.size(0) == 1 &&
          x_row.scalar_type() == at::kFloat && x_row.is_contiguous(),
      "CPU MoE fusion requires one CPU input row");
  TORCH_CHECK(
      group_size > 0 && group_size % 2 == 0,
      "INT4 group size must be a positive even number");

  const auto& x = x_row;
  const auto& weights = route_weights;
  TORCH_CHECK(
      !weights.is_cuda() && weights.scalar_type() == at::kFloat &&
          weights.is_contiguous(),
      "route weights must be contiguous float32 on CPU");
  TORCH_CHECK(weights.numel() == experts, "route weight count mismatch");
  const int64_t hidden = x.size(1);

  std::vector<int64_t> gu_blocks(experts);
  std::vector<int64_t> gu_codes(experts);
  std::vector<int64_t> gu_dims(experts);
  std::vector<int64_t> dn_blocks(experts);
  std::vector<int64_t> dn_codes(experts);
  std::vector<int64_t> dn_dims(experts);
  int64_t intermediate = -1;
  for (int64_t expert = 0; expert < experts; ++expert) {
    const auto& gu_index = gu_index_list[expert];
    const auto& gu_codebook = gu_codebook_list[expert];
    const auto& dn_index = dn_index_list[expert];
    const auto& dn_codebook = dn_codebook_list[expert];
    TORCH_CHECK(
        !gu_index.is_cuda() && !gu_codebook.is_cuda() &&
            !dn_index.is_cuda() && !dn_codebook.is_cuda(),
        "all routed expert operands must be on CPU");
    TORCH_CHECK(
        gu_index.scalar_type() == at::kByte &&
            dn_index.scalar_type() == at::kByte &&
            gu_codebook.scalar_type() == at::kFloat &&
            dn_codebook.scalar_type() == at::kFloat &&
            gu_index.dim() == 2 && dn_index.dim() == 2 &&
            gu_codebook.dim() == 2 && dn_codebook.dim() == 2 &&
            gu_index.is_contiguous() && dn_index.is_contiguous() &&
            gu_codebook.is_contiguous() && dn_codebook.is_contiguous(),
        "invalid routed VQ operand layout");
    const int64_t gu_rows =
        use_vq_int8 ? gu_index.size(1) : gu_index.size(0);
    const int64_t dn_rows =
        indices_transposed ? dn_index.size(1) : dn_index.size(0);
    const int64_t this_intermediate = gu_rows / 2;
    TORCH_CHECK(
        gu_rows == 2 * this_intermediate && dn_rows == hidden,
        "routed expert row count mismatch");
    if (intermediate < 0) {
      intermediate = this_intermediate;
    }
    TORCH_CHECK(
        this_intermediate == intermediate,
        "routed expert intermediate widths must match");
    gu_blocks[expert] =
        use_vq_int8 ? gu_index.size(0) : gu_index.size(1);
    gu_codes[expert] = gu_codebook.size(0);
    gu_dims[expert] = gu_codebook.size(1);
    dn_blocks[expert] =
        indices_transposed ? dn_index.size(0) : dn_index.size(1);
    dn_codes[expert] = dn_codebook.size(0);
    dn_dims[expert] = dn_codebook.size(1);
    TORCH_CHECK(
        gu_blocks[expert] * gu_dims[expert] == hidden &&
            dn_blocks[expert] * dn_dims[expert] == intermediate,
        "routed expert input width mismatch");
  }

  const auto& w1q = shared_w1_q;
  const auto& w1s = shared_w1_s;
  const auto& w3q = shared_w3_q;
  const auto& w3s = shared_w3_s;
  const auto& w2q = shared_w2_q;
  const auto& w2s = shared_w2_s;
  TORCH_CHECK(
      !w1q.is_cuda() && !w3q.is_cuda() && !w2q.is_cuda() &&
          !w1s.is_cuda() && !w3s.is_cuda() && !w2s.is_cuda(),
      "shared expert operands must be on CPU");
  TORCH_CHECK(
      w1q.scalar_type() == at::kByte &&
          w3q.scalar_type() == at::kByte &&
          w2q.scalar_type() == at::kByte &&
          w1s.scalar_type() == at::kHalf &&
          w3s.scalar_type() == at::kHalf &&
          w2s.scalar_type() == at::kHalf &&
          w1q.is_contiguous() && w3q.is_contiguous() &&
          w2q.is_contiguous() && w1s.is_contiguous() &&
          w3s.is_contiguous() && w2s.is_contiguous(),
      "shared expert quantization dtype mismatch");
  TORCH_CHECK(
      w1q.size(0) == intermediate && w3q.size(0) == intermediate &&
          w1q.size(1) * 2 == hidden && w3q.size(1) * 2 == hidden &&
          w2q.size(0) == hidden && w2q.size(1) * 2 == intermediate,
      "shared expert packed weight shape mismatch");
  TORCH_CHECK(
      w1s.size(0) == intermediate && w3s.size(0) == intermediate &&
          w2s.size(0) == hidden &&
          w1s.size(1) * group_size == hidden &&
          w3s.size(1) * group_size == hidden &&
          w2s.size(1) * group_size == intermediate,
      "shared expert scale shape mismatch");

  std::vector<const uint8_t*> gu_index_ptrs(experts);
  std::vector<const float*> gu_codebook_ptrs(experts);
  std::vector<const uint8_t*> dn_index_ptrs(experts);
  std::vector<const float*> dn_codebook_ptrs(experts);
  for (int64_t expert = 0; expert < experts; ++expert) {
    gu_index_ptrs[expert] =
        gu_index_list[expert].data_ptr<uint8_t>();
    gu_codebook_ptrs[expert] =
        gu_codebook_list[expert].data_ptr<float>();
    dn_index_ptrs[expert] =
        dn_index_list[expert].data_ptr<uint8_t>();
    dn_codebook_ptrs[expert] =
        dn_codebook_list[expert].data_ptr<float>();
  }

  // GU consumes the same input for every selected expert.  Reuse one lookup
  // score table whenever experts share the layer/tier codebook.
  std::vector<int64_t> gu_unique_representatives;
  std::vector<int64_t> gu_unique_for_expert(experts);
  for (int64_t expert = 0; expert < experts; ++expert) {
    int64_t unique = -1;
    for (int64_t candidate = 0;
         candidate < static_cast<int64_t>(gu_unique_representatives.size());
         ++candidate) {
      const int64_t other = gu_unique_representatives[candidate];
      if (gu_codebook_ptrs[expert] == gu_codebook_ptrs[other] &&
          gu_blocks[expert] == gu_blocks[other] &&
          gu_codes[expert] == gu_codes[other] &&
          gu_dims[expert] == gu_dims[other]) {
        unique = candidate;
        break;
      }
    }
    if (unique < 0) {
      unique = static_cast<int64_t>(gu_unique_representatives.size());
      gu_unique_representatives.push_back(expert);
    }
    gu_unique_for_expert[expert] = unique;
  }

  std::vector<int64_t> gu_score_offsets(
      gu_unique_representatives.size());
  std::vector<int64_t> gu_block_offsets(
      gu_unique_representatives.size() + 1, 0);
  int64_t gu_score_count = 0;
  for (int64_t unique = 0;
       unique < static_cast<int64_t>(gu_unique_representatives.size());
       ++unique) {
    gu_score_offsets[unique] = gu_score_count;
    const int64_t expert = gu_unique_representatives[unique];
    gu_score_count += gu_blocks[expert] * gu_codes[expert];
    gu_block_offsets[unique + 1] =
        gu_block_offsets[unique] + gu_blocks[expert];
  }
  std::vector<int64_t> dn_score_offsets(experts);
  std::vector<int64_t> dn_block_offsets(experts + 1, 0);
  int64_t dn_score_count = 0;
  for (int64_t expert = 0; expert < experts; ++expert) {
    dn_score_offsets[expert] = dn_score_count;
    dn_score_count += dn_blocks[expert] * dn_codes[expert];
    dn_block_offsets[expert + 1] =
        dn_block_offsets[expert] + dn_blocks[expert];
  }

  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  const int64_t shards =
      indices_transposed
      ? std::max<int64_t>(
            1,
            std::min<int64_t>(
                8, at::get_num_threads() / std::max<int64_t>(1, experts)))
      : 0;
  const int64_t gu_partial_offset = gu_score_count;
  const int64_t gu_partial_count = 0;
  const int64_t activation_offset = gu_partial_offset;
  const int64_t shared_offset =
      activation_offset + experts * intermediate;
  const int64_t dn_score_offset = shared_offset + intermediate;
  const int64_t dn_partial_offset = dn_score_offset + dn_score_count;
  const int64_t dn_partial_count =
      indices_transposed ? experts * shards * hidden : 0;
  const int64_t result_offset =
      dn_partial_offset + dn_partial_count;
  auto workspace = torch::empty({result_offset + hidden}, options);
  auto result = workspace.narrow(0, result_offset, hidden).view({1, hidden});
  const float* xp = x.data_ptr<float>();
  const float* routep = weights.data_ptr<float>();
  float* workspacep = workspace.data_ptr<float>();
  float* gu_scorep = workspacep;
  float* gu_partialp = workspacep + gu_partial_offset;
  float* activationp = workspacep + activation_offset;
  float* sharedp = workspacep + shared_offset;
  float* dn_scorep = workspacep + dn_score_offset;
  float* dn_partialp = workspacep + dn_partial_offset;
  float* resultp = workspacep + result_offset;
  torch::Tensor quantized_scores;
  torch::Tensor quantized_partials;
  int8_t* gu_quantizedp = nullptr;
  int8_t* dn_quantizedp = nullptr;
  int16_t* gu_i16_partialp = nullptr;
  int16_t* dn_i16_partialp = nullptr;
  if (use_vq_int8) {
    const int64_t gu_quantized_count =
        gu_block_offsets.back() * 256;
    const int64_t dn_quantized_count =
        dn_block_offsets.back() * 256;
    quantized_scores = torch::empty(
        {gu_quantized_count + dn_quantized_count},
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));
    gu_quantizedp = quantized_scores.data_ptr<int8_t>();
    dn_quantizedp = gu_quantizedp + gu_quantized_count;
    const int64_t gu_partial_i16_count =
        experts * vq_chunks * 2 * intermediate;
    const int64_t dn_partial_i16_count =
        experts * vq_chunks * hidden;
    quantized_partials = torch::empty(
        {gu_partial_i16_count + dn_partial_i16_count},
        torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));
    gu_i16_partialp = quantized_partials.data_ptr<int16_t>();
    dn_i16_partialp =
        gu_i16_partialp + gu_partial_i16_count;
  }
  std::vector<float> gu_quant_scales(
      use_vq_int8
          ? gu_unique_representatives.size() * vq_chunks
          : 0,
      1.0f);
  std::vector<float> dn_quant_scales(
      use_vq_int8 ? experts * vq_chunks : 0,
      1.0f);
  const float activation_limit = static_cast<float>(limit);

  const uint8_t* w1qp = w1q.data_ptr<uint8_t>();
  const at::Half* w1sp = w1s.data_ptr<at::Half>();
  const uint8_t* w3qp = w3q.data_ptr<uint8_t>();
  const at::Half* w3sp = w3s.data_ptr<at::Half>();
  const uint8_t* w2qp = w2q.data_ptr<uint8_t>();
  const at::Half* w2sp = w2s.data_ptr<at::Half>();
  const int64_t w1_bytes = hidden / 2;
  const int64_t w1_groups = hidden / group_size;
  const int64_t w2_bytes = intermediate / 2;
  const int64_t w2_groups = intermediate / group_size;
  const bool use_w4a8 = cpu_w4a8_enabled() && group_size == 64;
  const bool use_w4abf16 =
      cpu_w4abf16_enabled() && group_size == 64;
  const bool use_expand_bf16 = cpu_expand_bf16_enabled();
  Int8Activation quantized_input;
  Int8Activation quantized_shared;
  Bf16Activation bf16_input;
  Bf16Activation bf16_shared;
  torch::Tensor expanded_w1;
  torch::Tensor expanded_w3;
  torch::Tensor expanded_w2;
  const at::BFloat16* expanded_w1p = nullptr;
  const at::BFloat16* expanded_w3p = nullptr;
  const at::BFloat16* expanded_w2p = nullptr;
  if (use_w4a8) {
    quantized_input =
        quantize_int8_activation(xp, hidden, group_size);
  }
  if (use_w4abf16 || use_expand_bf16) {
    bf16_input =
        quantize_bf16_activation(xp, hidden, group_size);
  }
  if (use_expand_bf16) {
    expanded_w1 =
        expand_int4_bf16(w1q, w1s, hidden, group_size);
    expanded_w3 =
        expand_int4_bf16(w3q, w3s, hidden, group_size);
    expanded_w2 =
        expand_int4_bf16(w2q, w2s, intermediate, group_size);
    expanded_w1p = expanded_w1.data_ptr<at::BFloat16>();
    expanded_w3p = expanded_w3.data_ptr<at::BFloat16>();
    expanded_w2p = expanded_w2.data_ptr<at::BFloat16>();
  }
  static const bool phase_profile = [] {
    const char* value = std::getenv("TPQ_CPU_MOE_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  double phase_times[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  if (phase_profile) {
    phase_times[0] = wall_seconds();
  }

#pragma omp parallel
  {
 #pragma omp for schedule(static) nowait
    for (int64_t item = 0; item < gu_block_offsets.back(); ++item) {
      const int64_t unique = static_cast<int64_t>(
          std::upper_bound(
              gu_block_offsets.begin(),
              gu_block_offsets.end(),
              item) -
          gu_block_offsets.begin() - 1);
      const int64_t expert = gu_unique_representatives[unique];
      const int64_t blocks = gu_blocks[expert];
      const int64_t codes = gu_codes[expert];
      const int64_t dim = gu_dims[expert];
      float* score = gu_scorep + gu_score_offsets[unique];
      const float* codebook = gu_codebook_ptrs[expert];
      const int64_t block = item - gu_block_offsets[unique];
      const float* xv = xp + block * dim;
      float* block_score = score + block * codes;
      for (int64_t code = 0; code < codes; ++code) {
        block_score[code] =
            float_dot(xv, codebook + code * dim, dim);
      }
    }

#pragma omp for schedule(static) nowait
    for (int64_t row = 0; row < intermediate; ++row) {
      float gate =
          use_expand_bf16
          ? bf16_row_dot(
                bf16_input.values.data(),
                expanded_w1p + row * hidden,
                hidden)
          : use_w4abf16
          ? int4_row_dot_w4abf16(
                bf16_input,
                w1qp + row * w1_bytes,
                w1sp + row * w1_groups)
          : use_w4a8
          ? int4_row_dot_w4a8(
                quantized_input,
                w1qp + row * w1_bytes,
                w1sp + row * w1_groups)
          : int4_row_dot(
                xp,
                w1qp + row * w1_bytes,
                w1sp + row * w1_groups,
                hidden,
                group_size);
      float up =
          use_expand_bf16
          ? bf16_row_dot(
                bf16_input.values.data(),
                expanded_w3p + row * hidden,
                hidden)
          : use_w4abf16
          ? int4_row_dot_w4abf16(
                bf16_input,
                w3qp + row * w1_bytes,
                w3sp + row * w1_groups)
          : use_w4a8
          ? int4_row_dot_w4a8(
                quantized_input,
                w3qp + row * w1_bytes,
                w3sp + row * w1_groups)
          : int4_row_dot(
                xp,
                w3qp + row * w1_bytes,
                w3sp + row * w1_groups,
                hidden,
                group_size);
      if (activation_limit != 0.0f) {
        gate = std::min(gate, activation_limit);
        up = std::max(-activation_limit, std::min(up, activation_limit));
      }
      sharedp[row] = gate / (1.0f + std::exp(-gate)) * up;
    }

#pragma omp barrier
    if (use_vq_int8) {
#pragma omp for schedule(static)
      for (int64_t task = 0;
           task <
               static_cast<int64_t>(gu_unique_representatives.size()) *
                   vq_chunks;
           ++task) {
        const int64_t unique = task / vq_chunks;
        const int64_t chunk = task - unique * vq_chunks;
        const int64_t expert = gu_unique_representatives[unique];
        const int64_t codes = gu_codes[expert];
        const int64_t blocks = gu_blocks[expert];
        const int64_t block_begin = blocks * chunk / vq_chunks;
        const int64_t block_end =
            blocks * (chunk + 1) / vq_chunks;
        float maximum = 0.0f;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const float* source =
              gu_scorep + gu_score_offsets[unique] + block * codes;
          for (int64_t code = 0; code < codes; ++code) {
            maximum = std::max(maximum, std::abs(source[code]));
          }
        }
        const float scale =
            maximum > 0.0f ? maximum / 127.0f : 1.0f;
        gu_quant_scales[task] = scale;
        const float inverse = 1.0f / scale;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const float* source =
              gu_scorep + gu_score_offsets[unique] + block * codes;
          int8_t* destination =
              gu_quantizedp +
              (gu_block_offsets[unique] + block) * 256;
          for (int64_t code = 0; code < codes; ++code) {
            const int value = static_cast<int>(
                std::nearbyint(source[code] * inverse));
            destination[code] = static_cast<int8_t>(
                std::max(-127, std::min(127, value)));
          }
          std::fill(
              destination + codes, destination + 256, int8_t{0});
        }
      }
    }
    if (phase_profile) {
#pragma omp single nowait
      { phase_times[1] = wall_seconds(); }
    }
    if (use_vq_int8) {
#pragma omp for schedule(static)
      for (int64_t task = 0; task < experts * vq_chunks; ++task) {
        const int64_t expert = task / vq_chunks;
        const int64_t chunk = task - expert * vq_chunks;
        const int64_t blocks = gu_blocks[expert];
        const uint8_t* indices = gu_index_ptrs[expert];
        int16_t* partial =
            gu_i16_partialp + task * (2 * intermediate);
        std::fill(
            partial, partial + 2 * intermediate, int16_t{0});
        const int64_t unique = gu_unique_for_expert[expert];
        const int64_t block_begin = blocks * chunk / vq_chunks;
        const int64_t block_end =
            blocks * (chunk + 1) / vq_chunks;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const int8_t* block_score =
              gu_quantizedp +
              (gu_block_offsets[unique] + block) * 256;
          const uint8_t* block_indices =
              indices + block * (2 * intermediate);
          int64_t row = 0;
#if defined(__AVX512VBMI__)
          for (; row + 64 <= intermediate; row += 64) {
            add_i8_scores_64(
                partial + row,
                lookup_i8_rows_64(
                    block_score, block_indices + row));
            add_i8_scores_64(
                partial + intermediate + row,
                lookup_i8_rows_64(
                    block_score,
                    block_indices + intermediate + row));
          }
#endif
          for (; row < intermediate; ++row) {
            partial[row] += block_score[block_indices[row]];
            partial[intermediate + row] +=
                block_score[block_indices[intermediate + row]];
          }
        }
      }
#pragma omp for schedule(static)
      for (int64_t item = 0; item < experts * intermediate; ++item) {
        const int64_t expert = item / intermediate;
        const int64_t row = item - expert * intermediate;
        float gate = 0.0f;
        float up = 0.0f;
        const int64_t unique = gu_unique_for_expert[expert];
        for (int64_t chunk = 0; chunk < vq_chunks; ++chunk) {
          const int16_t* partial =
              gu_i16_partialp +
              (expert * vq_chunks + chunk) * (2 * intermediate);
          const float scale =
              gu_quant_scales[unique * vq_chunks + chunk];
          gate += static_cast<float>(partial[row]) * scale;
          up += static_cast<float>(partial[intermediate + row]) * scale;
        }
        if (activation_limit != 0.0f) {
          gate = std::min(gate, activation_limit);
          up = std::max(
              -activation_limit, std::min(up, activation_limit));
          }
        activationp[item] =
            gate / (1.0f + std::exp(-gate)) * up;
      }
    } else {
#pragma omp for schedule(static)
      for (int64_t item = 0; item < experts * intermediate; ++item) {
        const int64_t expert = item / intermediate;
        const int64_t row = item - expert * intermediate;
        const int64_t blocks = gu_blocks[expert];
        const int64_t codes = gu_codes[expert];
        const float* score =
            gu_scorep +
            gu_score_offsets[gu_unique_for_expert[expert]];
        const uint8_t* indices = gu_index_ptrs[expert];
        float gate;
        float up;
        lookup_sum_pair(
            score,
            indices + row * blocks,
            indices + (intermediate + row) * blocks,
            blocks,
            codes,
            gate,
            up);
        if (activation_limit != 0.0f) {
          gate = std::min(gate, activation_limit);
          up = std::max(
              -activation_limit, std::min(up, activation_limit));
        }
        activationp[item] =
            gate / (1.0f + std::exp(-gate)) * up;
      }
    }
    if (phase_profile) {
#pragma omp single nowait
      { phase_times[2] = wall_seconds(); }
    }
    if (use_w4a8) {
#pragma omp single
      {
        quantized_shared =
            quantize_int8_activation(
                sharedp, intermediate, group_size);
      }
    }
    if (use_w4abf16 || use_expand_bf16) {
#pragma omp single
      {
        bf16_shared =
            quantize_bf16_activation(
                sharedp, intermediate, group_size);
      }
    }

 #pragma omp for schedule(static) nowait
    for (int64_t item = 0; item < dn_block_offsets.back(); ++item) {
      const int64_t expert = static_cast<int64_t>(
          std::upper_bound(
              dn_block_offsets.begin(),
              dn_block_offsets.end(),
              item) -
          dn_block_offsets.begin() - 1);
      const int64_t blocks = dn_blocks[expert];
      const int64_t codes = dn_codes[expert];
      const int64_t dim = dn_dims[expert];
      const float* activated = activationp + expert * intermediate;
      const float* codebook = dn_codebook_ptrs[expert];
      float* score = dn_scorep + dn_score_offsets[expert];
      const int64_t block = item - dn_block_offsets[expert];
      const float* xv = activated + block * dim;
      float* block_score = score + block * codes;
      for (int64_t code = 0; code < codes; ++code) {
        block_score[code] =
            float_dot(xv, codebook + code * dim, dim);
      }
    }

#pragma omp for schedule(static) nowait
    for (int64_t row = 0; row < hidden; ++row) {
      resultp[row] =
          use_expand_bf16
          ? bf16_row_dot(
                bf16_shared.values.data(),
                expanded_w2p + row * intermediate,
                intermediate)
          : use_w4abf16
          ? int4_row_dot_w4abf16(
                bf16_shared,
                w2qp + row * w2_bytes,
                w2sp + row * w2_groups)
          : use_w4a8
          ? int4_row_dot_w4a8(
                quantized_shared,
                w2qp + row * w2_bytes,
                w2sp + row * w2_groups)
          : int4_row_dot(
                sharedp,
                w2qp + row * w2_bytes,
                w2sp + row * w2_groups,
                intermediate,
                group_size);
    }

#pragma omp barrier
    if (use_vq_int8) {
#pragma omp for schedule(static)
      for (int64_t task = 0; task < experts * vq_chunks; ++task) {
        const int64_t expert = task / vq_chunks;
        const int64_t chunk = task - expert * vq_chunks;
        const int64_t codes = dn_codes[expert];
        const int64_t blocks = dn_blocks[expert];
        const int64_t block_begin = blocks * chunk / vq_chunks;
        const int64_t block_end =
            blocks * (chunk + 1) / vq_chunks;
        float maximum = 0.0f;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const float* source =
              dn_scorep + dn_score_offsets[expert] + block * codes;
          for (int64_t code = 0; code < codes; ++code) {
            maximum = std::max(maximum, std::abs(source[code]));
          }
        }
        const float scale =
            maximum > 0.0f ? maximum / 127.0f : 1.0f;
        dn_quant_scales[task] = scale;
        const float inverse = 1.0f / scale;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const float* source =
              dn_scorep + dn_score_offsets[expert] + block * codes;
          int8_t* destination =
              dn_quantizedp +
              (dn_block_offsets[expert] + block) * 256;
          for (int64_t code = 0; code < codes; ++code) {
            const int value = static_cast<int>(
                std::nearbyint(source[code] * inverse));
            destination[code] = static_cast<int8_t>(
                std::max(-127, std::min(127, value)));
          }
          std::fill(
              destination + codes, destination + 256, int8_t{0});
        }
      }
    }
    if (phase_profile) {
#pragma omp single nowait
      { phase_times[3] = wall_seconds(); }
    }
    if (use_vq_int8) {
#pragma omp for schedule(static)
      for (int64_t task = 0; task < experts * vq_chunks; ++task) {
        const int64_t expert = task / vq_chunks;
        const int64_t chunk = task - expert * vq_chunks;
        const int64_t blocks = dn_blocks[expert];
        const uint8_t* indices = dn_index_ptrs[expert];
        int16_t* partial = dn_i16_partialp + task * hidden;
        std::fill(partial, partial + hidden, int16_t{0});
        const int64_t block_begin = blocks * chunk / vq_chunks;
        const int64_t block_end =
            blocks * (chunk + 1) / vq_chunks;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const int8_t* block_score =
              dn_quantizedp +
              (dn_block_offsets[expert] + block) * 256;
          const uint8_t* block_indices = indices + block * hidden;
          int64_t row = 0;
#if defined(__AVX512VBMI__)
          for (; row + 64 <= hidden; row += 64) {
            add_i8_scores_64(
                partial + row,
                lookup_i8_rows_64(
                    block_score, block_indices + row));
          }
#endif
          for (; row < hidden; ++row) {
            partial[row] += block_score[block_indices[row]];
          }
        }
      }
#pragma omp for schedule(static)
      for (int64_t row = 0; row < hidden; ++row) {
        float value = resultp[row];
        for (int64_t expert = 0; expert < experts; ++expert) {
          float expert_value = 0.0f;
          for (int64_t chunk = 0; chunk < vq_chunks; ++chunk) {
            const int16_t partial =
                dn_i16_partialp[
                    (expert * vq_chunks + chunk) * hidden + row];
            expert_value +=
                static_cast<float>(partial) *
                dn_quant_scales[expert * vq_chunks + chunk];
          }
          value += routep[expert] * expert_value;
        }
        resultp[row] = value;
      }
    } else if (indices_transposed) {
#pragma omp for schedule(static)
      for (int64_t task = 0; task < experts * shards; ++task) {
        const int64_t expert = task / shards;
        const int64_t shard = task - expert * shards;
        const int64_t blocks = dn_blocks[expert];
        const int64_t codes = dn_codes[expert];
        const float* score =
            dn_scorep + dn_score_offsets[expert];
        const uint8_t* indices = dn_index_ptrs[expert];
        float* partial = dn_partialp + task * hidden;
        std::fill(partial, partial + hidden, 0.0f);
        const int64_t block_begin = blocks * shard / shards;
        const int64_t block_end = blocks * (shard + 1) / shards;
        for (int64_t block = block_begin; block < block_end; ++block) {
          const float* block_score = score + block * codes;
          const uint8_t* block_indices = indices + block * hidden;
          int64_t row = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
          for (; row + 16 <= hidden; row += 16) {
            _mm512_storeu_ps(
                partial + row,
                _mm512_add_ps(
                    _mm512_loadu_ps(partial + row),
                    lookup_rows_16(
                        block_score, block_indices + row)));
          }
#endif
          for (; row < hidden; ++row) {
            partial[row] += block_score[block_indices[row]];
          }
        }
      }
#pragma omp for schedule(static)
      for (int64_t row = 0; row < hidden; ++row) {
        float value = resultp[row];
        for (int64_t expert = 0; expert < experts; ++expert) {
          float expert_value = 0.0f;
          for (int64_t shard = 0; shard < shards; ++shard) {
            expert_value +=
                dn_partialp[
                    (expert * shards + shard) * hidden + row];
          }
          value += routep[expert] * expert_value;
        }
        resultp[row] = value;
      }
    } else {
#pragma omp for schedule(static)
      for (int64_t row = 0; row < hidden; ++row) {
        resultp[row] += lookup_weighted_many(
            dn_score_offsets,
            dn_index_ptrs,
            dn_blocks,
            dn_codes,
            dn_scorep,
            routep,
            experts,
            row);
      }
    }
  }
  if (phase_profile) {
    phase_times[4] = wall_seconds();
    for (int64_t phase = 0; phase < 4; ++phase) {
      moe_phase_seconds[phase] += phase_times[phase + 1] - phase_times[phase];
    }
    ++moe_phase_calls;
  }
  return result;
}

void reset_moe_phase_profile_cpu() {
  for (double& phase : moe_phase_seconds) {
    phase = 0.0;
  }
  moe_phase_calls = 0;
}

torch::Tensor moe_phase_profile_cpu() {
  auto result = torch::empty(
      {5},
      torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
  double* values = result.data_ptr<double>();
  for (int64_t phase = 0; phase < 4; ++phase) {
    values[phase] = moe_phase_seconds[phase];
  }
  values[4] = static_cast<double>(moe_phase_calls);
  return result;
}

void transpose_vq_indices_into(
    const torch::Tensor& indices,
    torch::Tensor& output) {
  TORCH_CHECK(
      !indices.is_cuda() && indices.scalar_type() == at::kByte &&
          indices.dim() == 2 && indices.is_contiguous(),
      "CPU VQ transpose requires contiguous uint8 indices");
  const int64_t rows = indices.size(0);
  const int64_t blocks = indices.size(1);
  TORCH_CHECK(
      output.scalar_type() == at::kByte &&
          output.dim() == 2 && output.size(0) == blocks &&
          output.size(1) == rows && output.is_contiguous(),
      "CPU VQ transpose output shape mismatch");
  const uint8_t* source = indices.data_ptr<uint8_t>();
  uint8_t* destination = output.data_ptr<uint8_t>();
  constexpr int64_t tile = 32;
  for (int64_t row0 = 0; row0 < rows; row0 += tile) {
    for (int64_t block0 = 0; block0 < blocks; block0 += tile) {
      const int64_t row_end = std::min(rows, row0 + tile);
      const int64_t block_end = std::min(blocks, block0 + tile);
      for (int64_t row = row0; row < row_end; ++row) {
        for (int64_t block = block0; block < block_end; ++block) {
          destination[block * rows + row] =
              source[row * blocks + block];
        }
      }
    }
  }
}

class CpuMoeLayer {
 public:
  CpuMoeLayer(
      std::vector<torch::Tensor> gu_indices,
      std::vector<torch::Tensor> gu_codebooks,
      std::vector<torch::Tensor> dn_indices,
      std::vector<torch::Tensor> dn_codebooks,
      torch::Tensor valid_experts,
      torch::Tensor shared_w1_q,
      torch::Tensor shared_w1_s,
      torch::Tensor shared_w3_q,
      torch::Tensor shared_w3_s,
      torch::Tensor shared_w2_q,
      torch::Tensor shared_w2_s,
      torch::Tensor gate_q,
      torch::Tensor gate_s,
      torch::Tensor gate_bias,
      torch::Tensor gate_mask,
      int64_t group_size,
      double limit,
      int64_t top_k,
      bool normalize_route,
      double routed_scaling)
      : gu_indices_(std::move(gu_indices)),
        gu_codebooks_(std::move(gu_codebooks)),
        dn_indices_(std::move(dn_indices)),
        dn_codebooks_(std::move(dn_codebooks)),
        valid_experts_(valid_experts.to(torch::kBool).contiguous()),
        shared_w1_q_(std::move(shared_w1_q)),
        shared_w1_s_(std::move(shared_w1_s)),
        shared_w3_q_(std::move(shared_w3_q)),
        shared_w3_s_(std::move(shared_w3_s)),
        shared_w2_q_(std::move(shared_w2_q)),
        shared_w2_s_(std::move(shared_w2_s)),
        gate_q_(std::move(gate_q)),
        gate_s_(std::move(gate_s)),
        gate_bias_(gate_bias.to(torch::kFloat32).contiguous()),
        gate_mask_(gate_mask.to(torch::kBool).contiguous()),
        group_size_(group_size),
        limit_(limit),
        top_k_(top_k),
        normalize_route_(normalize_route),
        routed_scaling_(routed_scaling) {
    const int64_t count = static_cast<int64_t>(gu_indices_.size());
    TORCH_CHECK(
        count > 0 &&
            static_cast<int64_t>(gu_codebooks_.size()) == count &&
            static_cast<int64_t>(dn_indices_.size()) == count &&
            static_cast<int64_t>(dn_codebooks_.size()) == count &&
            valid_experts_.numel() == count,
        "cached CPU MoE layer expert counts must match");
    const char* transpose_mode = std::getenv("TPQ_CPU_DN_BLOCK");
    const char* int8_mode = std::getenv("TPQ_CPU_VQ_INT8");
    vq_int8_ =
        int8_mode != nullptr &&
        int8_mode[0] != '\0' && int8_mode[0] != '0';
    indices_transposed_ =
        vq_int8_ ||
        (transpose_mode != nullptr &&
         transpose_mode[0] != '\0' && transpose_mode[0] != '0');
    if (indices_transposed_) {
      if (vq_int8_) {
        gu_transposed_.resize(count);
      }
      dn_transposed_.resize(count);
      std::vector<int64_t> valid_ids;
      const bool* validp = valid_experts_.data_ptr<bool>();
      valid_ids.reserve(count);
      for (int64_t expert = 0; expert < count; ++expert) {
        if (validp[expert]) {
          if (vq_int8_) {
            gu_transposed_[expert] = torch::empty(
                {gu_indices_[expert].size(1), gu_indices_[expert].size(0)},
                torch::TensorOptions()
                    .dtype(torch::kUInt8)
                    .device(torch::kCPU));
          }
          dn_transposed_[expert] = torch::empty(
              {dn_indices_[expert].size(1), dn_indices_[expert].size(0)},
              torch::TensorOptions()
                  .dtype(torch::kUInt8)
                  .device(torch::kCPU));
          valid_ids.push_back(expert);
        }
      }
#pragma omp parallel for schedule(dynamic)
      for (int64_t item = 0;
           item < static_cast<int64_t>(valid_ids.size());
           ++item) {
        const int64_t expert = valid_ids[item];
        if (vq_int8_) {
          transpose_vq_indices_into(
              gu_indices_[expert], gu_transposed_[expert]);
        }
        transpose_vq_indices_into(
            dn_indices_[expert], dn_transposed_[expert]);
      }
    }
    TORCH_CHECK(
        !gate_q_.is_cuda() && !gate_s_.is_cuda() &&
            gate_q_.scalar_type() == at::kByte &&
            gate_s_.scalar_type() == at::kHalf &&
            gate_q_.dim() == 2 && gate_s_.dim() == 2 &&
            gate_q_.size(0) == count && gate_s_.size(0) == count &&
            gate_q_.size(1) * 2 == shared_w1_q_.size(1) * 2 &&
            gate_s_.size(1) * group_size_ == gate_q_.size(1) * 2 &&
            gate_bias_.numel() == count && gate_mask_.numel() == count &&
            top_k_ > 0 && top_k_ <= count,
        "cached CPU MoE router shape mismatch");
    route_scores_ = torch::empty(
        {count},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  }

  torch::Tensor forward(
      torch::Tensor x_row,
      torch::Tensor route_weights,
      torch::Tensor expert_ids) {
    auto ids = expert_ids.to(torch::kLong).contiguous();
    TORCH_CHECK(
        ids.dim() == 1 && route_weights.numel() == ids.numel(),
        "cached CPU MoE route shape mismatch");
    const int64_t count = ids.numel();
    const int64_t* idp = ids.data_ptr<int64_t>();
    return forward_selected(x_row, route_weights, idp, count);
  }

  torch::Tensor forward_learned(torch::Tensor x_row) {
    TORCH_CHECK(
        !x_row.is_cuda() && x_row.scalar_type() == at::kFloat &&
            x_row.dim() == 2 && x_row.size(0) == 1 &&
            x_row.size(1) == gate_q_.size(1) * 2 &&
            x_row.is_contiguous(),
        "cached CPU MoE learned route input mismatch");
    const float* xp = x_row.data_ptr<float>();
    const uint8_t* qp = gate_q_.data_ptr<uint8_t>();
    const at::Half* sp = gate_s_.data_ptr<at::Half>();
    float* scorep = route_scores_.data_ptr<float>();
    const int64_t experts = gate_q_.size(0);
    const int64_t cols = gate_q_.size(1) * 2;
    const int64_t groups = cols / group_size_;
    const int64_t bytes_per_row = cols / 2;
    at::parallel_for(0, experts, 1, [&](int64_t begin, int64_t end) {
      for (int64_t expert = begin; expert < end; ++expert) {
        const float raw = int4_row_dot(
            xp,
            qp + expert * bytes_per_row,
            sp + expert * groups,
            cols,
            group_size_);
        const float softplus =
            raw > 20.0f ? raw : std::log1p(std::exp(raw));
        scorep[expert] = std::sqrt(softplus);
      }
    });

    const float* biasp = gate_bias_.data_ptr<float>();
    const bool* maskp = gate_mask_.data_ptr<bool>();
    std::vector<int64_t> selected(top_k_, -1);
    std::vector<float> choices(
        top_k_, -std::numeric_limits<float>::infinity());
    for (int64_t expert = 0; expert < experts; ++expert) {
      if (!maskp[expert]) {
        continue;
      }
      const float choice = scorep[expert] + biasp[expert];
      for (int64_t rank = 0; rank < top_k_; ++rank) {
        if (choice > choices[rank]) {
          for (int64_t move = top_k_ - 1; move > rank; --move) {
            choices[move] = choices[move - 1];
            selected[move] = selected[move - 1];
          }
          choices[rank] = choice;
          selected[rank] = expert;
          break;
        }
      }
    }
    auto route_weights = torch::empty(
        {top_k_},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* routep = route_weights.data_ptr<float>();
    float denominator = normalize_route_ ? 1.0e-20f : 1.0f;
    for (int64_t rank = 0; rank < top_k_; ++rank) {
      TORCH_CHECK(selected[rank] >= 0, "not enough available routed experts");
      routep[rank] = scorep[selected[rank]];
      if (normalize_route_) {
        denominator += routep[rank];
      }
    }
    const float multiplier =
        static_cast<float>(routed_scaling_) / denominator;
    for (int64_t rank = 0; rank < top_k_; ++rank) {
      routep[rank] *= multiplier;
    }
    return forward_selected(
        x_row,
        route_weights,
        selected.data(),
        top_k_);
  }

 private:
  torch::Tensor forward_selected(
      torch::Tensor x_row,
      torch::Tensor route_weights,
      const int64_t* idp,
      int64_t count) {
    const bool* validp = valid_experts_.data_ptr<bool>();
    std::vector<torch::Tensor> gu_indices;
    std::vector<torch::Tensor> gu_codebooks;
    std::vector<torch::Tensor> dn_indices;
    std::vector<torch::Tensor> dn_codebooks;
    gu_indices.reserve(count);
    gu_codebooks.reserve(count);
    dn_indices.reserve(count);
    dn_codebooks.reserve(count);
    for (int64_t slot = 0; slot < count; ++slot) {
      const int64_t expert = idp[slot];
      TORCH_CHECK(
          expert >= 0 &&
              expert < static_cast<int64_t>(gu_indices_.size()) &&
              validp[expert],
          "route selected an unavailable cached CPU expert");
      if (indices_transposed_) {
        gu_indices.push_back(
            vq_int8_ ? gu_transposed_[expert] : gu_indices_[expert]);
        dn_indices.push_back(dn_transposed_[expert]);
      } else {
        gu_indices.push_back(gu_indices_[expert]);
        dn_indices.push_back(dn_indices_[expert]);
      }
      gu_codebooks.push_back(gu_codebooks_[expert]);
      dn_codebooks.push_back(dn_codebooks_[expert]);
    }
    return moe_mixed_cpu(
        x_row,
        std::move(gu_indices),
        std::move(gu_codebooks),
        std::move(dn_indices),
        std::move(dn_codebooks),
        route_weights,
        shared_w1_q_,
        shared_w1_s_,
        shared_w3_q_,
        shared_w3_s_,
        shared_w2_q_,
        shared_w2_s_,
        group_size_,
        limit_,
        indices_transposed_);
  }

  std::vector<torch::Tensor> gu_indices_;
  std::vector<torch::Tensor> gu_codebooks_;
  std::vector<torch::Tensor> dn_indices_;
  std::vector<torch::Tensor> dn_codebooks_;
  std::vector<torch::Tensor> gu_transposed_;
  std::vector<torch::Tensor> dn_transposed_;
  torch::Tensor valid_experts_;
  torch::Tensor shared_w1_q_;
  torch::Tensor shared_w1_s_;
  torch::Tensor shared_w3_q_;
  torch::Tensor shared_w3_s_;
  torch::Tensor shared_w2_q_;
  torch::Tensor shared_w2_s_;
  torch::Tensor gate_q_;
  torch::Tensor gate_s_;
  torch::Tensor gate_bias_;
  torch::Tensor gate_mask_;
  torch::Tensor route_scores_;
  int64_t group_size_;
  double limit_;
  int64_t top_k_;
  bool normalize_route_;
  bool indices_transposed_;
  bool vq_int8_;
  double routed_scaling_;
};

std::vector<torch::Tensor> int4_gemv_many_cpu(
    torch::Tensor x_row,
    std::vector<torch::Tensor> packed_list,
    std::vector<torch::Tensor> scale_list,
    int64_t group_size) {
  TORCH_CHECK(
      !x_row.is_cuda() && x_row.dim() == 2 && x_row.size(0) == 1,
      "CPU multi-INT4 GEMV requires one CPU input row");
  TORCH_CHECK(
      !packed_list.empty() && packed_list.size() == scale_list.size(),
      "multi-INT4 weight/scale counts must match");
  TORCH_CHECK(
      group_size > 0 && group_size % 2 == 0,
      "INT4 group size must be a positive even number");
  auto x = x_row.to(torch::kFloat32).contiguous();
  const int64_t cols = x.size(1);
  const int64_t groups = cols / group_size;
  const float* xp = x.data_ptr<float>();
  const bool use_w4a8 = cpu_w4a8_enabled() && group_size == 64;
  const bool use_w4abf16 =
      cpu_w4abf16_enabled() && group_size == 64;
  const bool use_expand_bf16 = cpu_expand_bf16_enabled();
  Int8Activation quantized;
  Bf16Activation bf16_input;
  if (use_w4a8) {
    quantized = quantize_int8_activation(xp, cols, group_size);
  }
  if (use_w4abf16 || use_expand_bf16) {
    bf16_input =
        quantize_bf16_activation(xp, cols, group_size);
  }

  std::vector<torch::Tensor> packed;
  std::vector<torch::Tensor> scales;
  std::vector<torch::Tensor> outputs;
  std::vector<const uint8_t*> packed_ptrs;
  std::vector<const at::Half*> scale_ptrs;
  std::vector<float*> output_ptrs;
  std::vector<torch::Tensor> expanded_weights;
  std::vector<const at::BFloat16*> expanded_ptrs;
  std::vector<int64_t> row_offsets(packed_list.size() + 1, 0);
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  for (size_t index = 0; index < packed_list.size(); ++index) {
    auto weight = packed_list[index].contiguous();
    auto scale = scale_list[index].contiguous();
    TORCH_CHECK(
        !weight.is_cuda() && !scale.is_cuda() &&
            weight.scalar_type() == at::kByte &&
            scale.scalar_type() == at::kHalf &&
            weight.dim() == 2 && scale.dim() == 2,
        "invalid multi-INT4 operand");
    TORCH_CHECK(
        weight.size(1) * 2 == cols &&
            scale.size(0) == weight.size(0) &&
            scale.size(1) == groups,
        "multi-INT4 shape mismatch");
    packed.push_back(weight);
    scales.push_back(scale);
    outputs.push_back(torch::empty({1, weight.size(0)}, options));
    packed_ptrs.push_back(packed.back().data_ptr<uint8_t>());
    scale_ptrs.push_back(scales.back().data_ptr<at::Half>());
    output_ptrs.push_back(outputs.back().data_ptr<float>());
    row_offsets[index + 1] = row_offsets[index] + weight.size(0);
  }
  if (use_expand_bf16) {
    expanded_weights.reserve(packed.size());
    expanded_ptrs.reserve(packed.size());
    for (size_t index = 0; index < packed.size(); ++index) {
      expanded_weights.push_back(
          expand_int4_bf16(
              packed[index], scales[index], cols, group_size));
      expanded_ptrs.push_back(
          expanded_weights.back().data_ptr<at::BFloat16>());
    }
  }

  const int64_t bytes_per_row = cols / 2;
#pragma omp parallel for schedule(static)
  for (int64_t item = 0; item < row_offsets.back(); ++item) {
    size_t matrix = 0;
    while (item >= row_offsets[matrix + 1]) {
      ++matrix;
    }
    const int64_t row = item - row_offsets[matrix];
    const uint8_t* weights =
        packed_ptrs[matrix] + row * bytes_per_row;
    const at::Half* row_scales =
        scale_ptrs[matrix] + row * groups;
    output_ptrs[matrix][row] =
        use_expand_bf16
        ? bf16_row_dot(
              bf16_input.values.data(),
              expanded_ptrs[matrix] + row * cols,
              cols)
        : use_w4abf16
        ? int4_row_dot_w4abf16(
              bf16_input, weights, row_scales)
        : use_w4a8
        ? int4_row_dot_w4a8(quantized, weights, row_scales)
        : int4_row_dot(
              xp, weights, row_scales, cols, group_size);
  }
  return outputs;
}

torch::Tensor int4_gemv_cpu(
    torch::Tensor x_row,
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t cols,
    int64_t group_size) {
  TORCH_CHECK(!x_row.is_cuda() && !packed.is_cuda() && !scales.is_cuda(),
              "all INT4 operands must be on CPU");
  TORCH_CHECK(x_row.dim() == 2 && x_row.size(0) == 1,
              "the CPU INT4 decode kernel requires x shape [1,C]");
  TORCH_CHECK(packed.dim() == 2 && packed.scalar_type() == at::kByte,
              "packed weight must be uint8 [R,C/2]");
  TORCH_CHECK(scales.dim() == 2 && scales.scalar_type() == at::kHalf,
              "scales must be float16 [R,C/group]");
  TORCH_CHECK(group_size > 0 && group_size % 2 == 0,
              "group size must be a positive even number");
  TORCH_CHECK(cols == packed.size(1) * 2 && x_row.size(1) == cols,
              "INT4 input width mismatch");
  TORCH_CHECK(scales.size(0) == packed.size(0) &&
                  scales.size(1) * group_size == cols,
              "INT4 scale shape mismatch");

  auto x = x_row.to(torch::kFloat32).contiguous();
  auto q = packed.contiguous();
  auto s = scales.contiguous();
  const int64_t rows = q.size(0);
  const int64_t groups = cols / group_size;
  const int64_t bytes_per_group = group_size / 2;
  const float* xp = x.data_ptr<float>();
  const uint8_t* qp = q.data_ptr<uint8_t>();
  const at::Half* sp = s.data_ptr<at::Half>();
  const bool use_w4a8 = cpu_w4a8_enabled() && group_size == 64;
  const bool use_expand_bf16 = cpu_expand_bf16_enabled();
  Int8Activation quantized;
  Bf16Activation bf16_input;
  torch::Tensor expanded;
  const at::BFloat16* expandedp = nullptr;
  if (use_w4a8) {
    quantized = quantize_int8_activation(xp, cols, group_size);
  }
  if (use_expand_bf16) {
    bf16_input =
        quantize_bf16_activation(xp, cols, group_size);
    expanded =
        expand_int4_bf16(q, s, cols, group_size);
    expandedp = expanded.data_ptr<at::BFloat16>();
  }
  auto out = torch::empty(
      {1, rows},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* op = out.data_ptr<float>();

  // Decode is a GEMV.  Parallelising by output row keeps each packed weight
  // row and its scales sequential while x remains shared in cache.
  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t r = begin; r < end; ++r) {
      const uint8_t* qrow = qp + r * (cols / 2);
      const at::Half* srow = sp + r * groups;
      if (use_expand_bf16) {
        op[r] = bf16_row_dot(
            bf16_input.values.data(),
            expandedp + r * cols,
            cols);
      } else if (use_w4a8) {
        op[r] = int4_row_dot_w4a8(quantized, qrow, srow);
      } else {
        float total = 0.0f;
        for (int64_t g = 0; g < groups; ++g) {
          const uint8_t* qgroup = qrow + g * bytes_per_group;
          const float* xgroup = xp + g * group_size;
          const float dot = int4_group_dot(xgroup, qgroup, group_size);
          total += dot * static_cast<float>(srow[g]);
        }
        op[r] = total;
      }
    }
  });
  return out;
}

torch::Tensor int4_grouped_gemv_cpu(
    torch::Tensor x_groups,
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t cols,
    int64_t group_size,
    int64_t rows_per_input) {
  TORCH_CHECK(
      !x_groups.is_cuda() && !packed.is_cuda() && !scales.is_cuda(),
      "all grouped INT4 operands must be on CPU");
  TORCH_CHECK(x_groups.dim() == 2 && x_groups.size(1) == cols,
              "grouped INT4 input must be [G,C]");
  TORCH_CHECK(packed.dim() == 2 && packed.scalar_type() == at::kByte,
              "packed weight must be uint8 [R,C/2]");
  TORCH_CHECK(scales.dim() == 2 && scales.scalar_type() == at::kHalf,
              "scales must be float16 [R,C/group]");
  TORCH_CHECK(group_size > 0 && group_size % 2 == 0,
              "group size must be a positive even number");
  TORCH_CHECK(cols == packed.size(1) * 2,
              "grouped INT4 input width mismatch");
  TORCH_CHECK(
      rows_per_input > 0 &&
          packed.size(0) == x_groups.size(0) * rows_per_input,
      "grouped INT4 row partition mismatch");

  auto x = x_groups.to(torch::kFloat32).contiguous();
  auto q = packed.contiguous();
  auto s = scales.contiguous();
  const int64_t input_groups = x.size(0);
  const int64_t rows = q.size(0);
  const int64_t weight_groups = cols / group_size;
  TORCH_CHECK(
      scales.size(0) == rows && scales.size(1) == weight_groups,
      "grouped INT4 scale shape mismatch");
  const float* xp = x.data_ptr<float>();
  const uint8_t* qp = q.data_ptr<uint8_t>();
  const at::Half* sp = s.data_ptr<at::Half>();
  auto out = torch::empty(
      {input_groups, rows_per_input},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* op = out.data_ptr<float>();

  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t r = begin; r < end; ++r) {
      const int64_t input_group = r / rows_per_input;
      const float* xrow = xp + input_group * cols;
      const uint8_t* qrow = qp + r * (cols / 2);
      const at::Half* srow = sp + r * weight_groups;
      float total = 0.0f;
      for (int64_t g = 0; g < weight_groups; ++g) {
        total += int4_group_dot(
                     xrow + g * group_size,
                     qrow + g * (group_size / 2),
                     group_size) *
                 static_cast<float>(srow[g]);
      }
      op[r] = total;
    }
  });
  return out;
}

torch::Tensor o_proj_int4_cpu(
    torch::Tensor x_groups,
    torch::Tensor a_packed,
    torch::Tensor a_scales,
    int64_t a_cols,
    int64_t a_group_size,
    int64_t rows_per_input,
    torch::Tensor b_packed,
    torch::Tensor b_scales,
    int64_t b_cols,
    int64_t b_group_size) {
  TORCH_CHECK(
      !x_groups.is_cuda() && !a_packed.is_cuda() &&
          !a_scales.is_cuda() && !b_packed.is_cuda() &&
          !b_scales.is_cuda() && x_groups.scalar_type() == at::kFloat &&
          x_groups.dim() == 2 && x_groups.size(1) == a_cols &&
          a_packed.scalar_type() == at::kByte &&
          a_scales.scalar_type() == at::kHalf &&
          b_packed.scalar_type() == at::kByte &&
          b_scales.scalar_type() == at::kHalf &&
          a_packed.size(0) == x_groups.size(0) * rows_per_input &&
          a_packed.size(1) * 2 == a_cols &&
          a_scales.size(0) == a_packed.size(0) &&
          a_scales.size(1) * a_group_size == a_cols &&
          b_cols == x_groups.size(0) * rows_per_input &&
          b_packed.size(1) * 2 == b_cols &&
          b_scales.size(0) == b_packed.size(0) &&
          b_scales.size(1) * b_group_size == b_cols,
      "CPU fused O projection shape mismatch");
  auto x = x_groups.contiguous();
  auto aq = a_packed.contiguous();
  auto as = a_scales.contiguous();
  auto bq = b_packed.contiguous();
  auto bs = b_scales.contiguous();
  auto middle = torch::empty(
      {x.size(0), rows_per_input},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto output = torch::empty(
      {1, bq.size(0)},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const float* xp = x.data_ptr<float>();
  const uint8_t* aqp = aq.data_ptr<uint8_t>();
  const at::Half* asp = as.data_ptr<at::Half>();
  const uint8_t* bqp = bq.data_ptr<uint8_t>();
  const at::Half* bsp = bs.data_ptr<at::Half>();
  float* mp = middle.data_ptr<float>();
  float* op = output.data_ptr<float>();
  const int64_t a_rows = aq.size(0);
  const int64_t a_bytes = a_cols / 2;
  const int64_t a_groups = a_cols / a_group_size;
  const int64_t b_rows = bq.size(0);
  const int64_t b_bytes = b_cols / 2;
  const int64_t b_groups = b_cols / b_group_size;
  const bool use_w4a8 =
      cpu_w4a8_enabled() &&
      a_group_size == 64 && b_group_size == 64;
  const bool use_w4abf16 =
      cpu_w4abf16_enabled() &&
      a_group_size == 64 && b_group_size == 64;
  const bool use_expand_bf16 = cpu_expand_bf16_enabled();
  std::vector<Int8Activation> quantized_inputs;
  Int8Activation quantized_middle;
  std::vector<Bf16Activation> bf16_inputs;
  Bf16Activation bf16_middle;
  torch::Tensor expanded_a;
  torch::Tensor expanded_b;
  const at::BFloat16* expanded_ap = nullptr;
  const at::BFloat16* expanded_bp = nullptr;
  if (use_w4a8) {
    quantized_inputs.reserve(x.size(0));
    for (int64_t input = 0; input < x.size(0); ++input) {
      quantized_inputs.push_back(
          quantize_int8_activation(
              xp + input * a_cols, a_cols, a_group_size));
    }
  }
  if (use_w4abf16 || use_expand_bf16) {
    bf16_inputs.reserve(x.size(0));
    for (int64_t input = 0; input < x.size(0); ++input) {
      bf16_inputs.push_back(
          quantize_bf16_activation(
              xp + input * a_cols, a_cols, a_group_size));
    }
  }
  if (use_expand_bf16) {
    expanded_a =
        expand_int4_bf16(aq, as, a_cols, a_group_size);
    expanded_b =
        expand_int4_bf16(bq, bs, b_cols, b_group_size);
    expanded_ap = expanded_a.data_ptr<at::BFloat16>();
    expanded_bp = expanded_b.data_ptr<at::BFloat16>();
  }
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (int64_t row = 0; row < a_rows; ++row) {
      const int64_t input = row / rows_per_input;
      const uint8_t* weights = aqp + row * a_bytes;
      const at::Half* row_scales = asp + row * a_groups;
      mp[row] =
          use_expand_bf16
          ? bf16_row_dot(
                bf16_inputs[input].values.data(),
                expanded_ap + row * a_cols,
                a_cols)
          : use_w4abf16
          ? int4_row_dot_w4abf16(
                bf16_inputs[input], weights, row_scales)
          : use_w4a8
          ? int4_row_dot_w4a8(
                quantized_inputs[input], weights, row_scales)
          : int4_row_dot(
                xp + input * a_cols,
                weights,
                row_scales,
                a_cols,
                a_group_size);
    }
    if (use_w4a8) {
#pragma omp single
      {
        quantized_middle =
            quantize_int8_activation(mp, b_cols, b_group_size);
      }
    }
    if (use_w4abf16 || use_expand_bf16) {
#pragma omp single
      {
        bf16_middle =
            quantize_bf16_activation(mp, b_cols, b_group_size);
      }
    }
#pragma omp for schedule(static)
    for (int64_t row = 0; row < b_rows; ++row) {
      const uint8_t* weights = bqp + row * b_bytes;
      const at::Half* row_scales = bsp + row * b_groups;
      op[row] =
          use_expand_bf16
          ? bf16_row_dot(
                bf16_middle.values.data(),
                expanded_bp + row * b_cols,
                b_cols)
          : use_w4abf16
          ? int4_row_dot_w4abf16(
                bf16_middle, weights, row_scales)
          : use_w4a8
          ? int4_row_dot_w4a8(
                quantized_middle, weights, row_scales)
          : int4_row_dot(
                mp, weights, row_scales, b_cols, b_group_size);
    }
  }
  return output;
}

std::vector<torch::Tensor> hc_pre_norm_cpu(
    torch::Tensor x,
    torch::Tensor mixes,
    torch::Tensor scale,
    torch::Tensor base,
    torch::Tensor norm,
    int64_t sinkhorn_iters,
    double rms_eps,
    double hc_eps) {
  TORCH_CHECK(
      !x.is_cuda() && !mixes.is_cuda() && !scale.is_cuda() &&
          !base.is_cuda() && !norm.is_cuda(),
      "all Hyper-Connection operands must be on CPU");
  TORCH_CHECK(
      x.scalar_type() == at::kFloat && mixes.scalar_type() == at::kFloat,
      "CPU Hyper-Connection requires float32 x and mixes");
  TORCH_CHECK(
      x.dim() == 4 && x.size(0) * x.size(1) == 1,
      "CPU Hyper-Connection decode requires one token");
  const int64_t hc = x.size(2);
  const int64_t hidden = x.size(3);
  TORCH_CHECK(hc > 0 && hc <= 8, "unsupported Hyper-Connection width");
  TORCH_CHECK(
      mixes.numel() == (2 + hc) * hc,
      "Hyper-Connection mix width mismatch");
  TORCH_CHECK(
      scale.numel() == 3 && base.numel() == (2 + hc) * hc,
      "Hyper-Connection scale/base shape mismatch");
  TORCH_CHECK(norm.numel() == hidden, "RMSNorm width mismatch");
  TORCH_CHECK(sinkhorn_iters > 0, "Sinkhorn iteration count must be positive");

  auto xc = x.contiguous();
  auto mc = mixes.contiguous();
  auto sc = scale.to(torch::kFloat32).contiguous();
  auto bc = base.to(torch::kFloat32).contiguous();
  auto nc = norm.to(torch::kFloat32).contiguous();
  const float* xp = xc.data_ptr<float>();
  const float* mp = mc.data_ptr<float>();
  const float* sp = sc.data_ptr<float>();
  const float* bp = bc.data_ptr<float>();
  const float* np = nc.data_ptr<float>();

  float square_sum = 0.0f;
#if defined(__AVX512F__)
  __m512 square_acc = _mm512_setzero_ps();
  int64_t flat_index = 0;
  const int64_t flat_size = hc * hidden;
  for (; flat_index + 16 <= flat_size; flat_index += 16) {
    const __m512 value = _mm512_loadu_ps(xp + flat_index);
    square_acc = _mm512_fmadd_ps(value, value, square_acc);
  }
  square_sum = _mm512_reduce_add_ps(square_acc);
  for (; flat_index < flat_size; ++flat_index) {
    square_sum += xp[flat_index] * xp[flat_index];
  }
#else
  for (int64_t i = 0; i < hc * hidden; ++i) {
    square_sum += xp[i] * xp[i];
  }
#endif
  const float input_rms = 1.0f / std::sqrt(
      square_sum / static_cast<float>(hc * hidden) +
      static_cast<float>(rms_eps));

  float pre_values[8];
  float post_values[8];
  float comb_values[64];
  for (int64_t j = 0; j < hc; ++j) {
    const float pre_arg = mp[j] * input_rms * sp[0] + bp[j];
    const float post_arg =
        mp[hc + j] * input_rms * sp[1] + bp[hc + j];
    pre_values[j] =
        1.0f / (1.0f + std::exp(-pre_arg)) + static_cast<float>(hc_eps);
    post_values[j] = 2.0f / (1.0f + std::exp(-post_arg));
  }
  for (int64_t row = 0; row < hc; ++row) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (int64_t col = 0; col < hc; ++col) {
      const int64_t index = row * hc + col;
      const float value =
          mp[2 * hc + index] * input_rms * sp[2] + bp[2 * hc + index];
      comb_values[index] = value;
      maximum = std::max(maximum, value);
    }
    float denominator = 0.0f;
    for (int64_t col = 0; col < hc; ++col) {
      const int64_t index = row * hc + col;
      const float value = std::exp(comb_values[index] - maximum);
      comb_values[index] = value;
      denominator += value;
    }
    for (int64_t col = 0; col < hc; ++col) {
      const int64_t index = row * hc + col;
      comb_values[index] =
          comb_values[index] / denominator + static_cast<float>(hc_eps);
    }
  }
  for (int64_t col = 0; col < hc; ++col) {
    float denominator = static_cast<float>(hc_eps);
    for (int64_t row = 0; row < hc; ++row) {
      denominator += comb_values[row * hc + col];
    }
    for (int64_t row = 0; row < hc; ++row) {
      comb_values[row * hc + col] /= denominator;
    }
  }
  for (int64_t iteration = 1; iteration < sinkhorn_iters; ++iteration) {
    for (int64_t row = 0; row < hc; ++row) {
      float denominator = static_cast<float>(hc_eps);
      for (int64_t col = 0; col < hc; ++col) {
        denominator += comb_values[row * hc + col];
      }
      for (int64_t col = 0; col < hc; ++col) {
        comb_values[row * hc + col] /= denominator;
      }
    }
    for (int64_t col = 0; col < hc; ++col) {
      float denominator = static_cast<float>(hc_eps);
      for (int64_t row = 0; row < hc; ++row) {
        denominator += comb_values[row * hc + col];
      }
      for (int64_t row = 0; row < hc; ++row) {
        comb_values[row * hc + col] /= denominator;
      }
    }
  }

  auto y = torch::empty(
      {1, 1, hidden},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  float* yp = y.data_ptr<float>();
  float y_square_sum = 0.0f;
  int64_t y_index = 0;
#if defined(__AVX512F__)
  __m512 y_square_acc = _mm512_setzero_ps();
  for (; y_index + 16 <= hidden; y_index += 16) {
    __m512 value = _mm512_setzero_ps();
    for (int64_t j = 0; j < hc; ++j) {
      value = _mm512_fmadd_ps(
          _mm512_loadu_ps(xp + j * hidden + y_index),
          _mm512_set1_ps(pre_values[j]),
          value);
    }
    _mm512_storeu_ps(yp + y_index, value);
    y_square_acc = _mm512_fmadd_ps(value, value, y_square_acc);
  }
  y_square_sum = _mm512_reduce_add_ps(y_square_acc);
#endif
  for (; y_index < hidden; ++y_index) {
    float value = 0.0f;
    for (int64_t j = 0; j < hc; ++j) {
      value += pre_values[j] * xp[j * hidden + y_index];
    }
    yp[y_index] = value;
    y_square_sum += value * value;
  }
  const float output_rms = 1.0f / std::sqrt(
      y_square_sum / static_cast<float>(hidden) + static_cast<float>(rms_eps));
  int64_t norm_index = 0;
#if defined(__AVX512F__)
  const __m512 output_scale = _mm512_set1_ps(output_rms);
  for (; norm_index + 16 <= hidden; norm_index += 16) {
    _mm512_storeu_ps(
        yp + norm_index,
        _mm512_mul_ps(
            _mm512_mul_ps(
                _mm512_loadu_ps(yp + norm_index),
                output_scale),
            _mm512_loadu_ps(np + norm_index)));
  }
#endif
  for (; norm_index < hidden; ++norm_index) {
    yp[norm_index] *= output_rms * np[norm_index];
  }

  auto post = torch::empty(
      {1, 1, hc},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto comb = torch::empty(
      {1, 1, hc, hc},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  std::copy(post_values, post_values + hc, post.data_ptr<float>());
  std::copy(
      comb_values, comb_values + hc * hc, comb.data_ptr<float>());
  return {y, post, comb};
}

torch::Tensor hc_post_cpu(
    torch::Tensor out,
    torch::Tensor residual,
    torch::Tensor post,
    torch::Tensor comb) {
  TORCH_CHECK(
      !out.is_cuda() && !residual.is_cuda() && !post.is_cuda() &&
          !comb.is_cuda(),
      "all Hyper-Connection post operands must be on CPU");
  TORCH_CHECK(
      out.scalar_type() == at::kFloat &&
          residual.scalar_type() == at::kFloat,
      "CPU Hyper-Connection post requires float32 operands");
  TORCH_CHECK(
      residual.dim() == 4 && residual.size(0) * residual.size(1) == 1,
      "CPU Hyper-Connection post requires one token");
  const int64_t hc = residual.size(2);
  const int64_t hidden = residual.size(3);
  TORCH_CHECK(
      out.numel() == hidden && post.numel() == hc &&
          comb.numel() == hc * hc,
      "Hyper-Connection post shape mismatch");
  const auto& oc = out;
  const auto& rc = residual;
  const auto& pc = post;
  const auto& cc = comb;
  TORCH_CHECK(
      post.scalar_type() == at::kFloat &&
          comb.scalar_type() == at::kFloat &&
          out.is_contiguous() && residual.is_contiguous() &&
          post.is_contiguous() && comb.is_contiguous(),
      "CPU Hyper-Connection post requires contiguous float32 tensors");
  const float* op = oc.data_ptr<float>();
  const float* rp = rc.data_ptr<float>();
  const float* pp = pc.data_ptr<float>();
  const float* cp = cc.data_ptr<float>();
  auto result = torch::empty_like(rc);
  float* resultp = result.data_ptr<float>();
  for (int64_t channel = 0; channel < hc; ++channel) {
    float* destination = resultp + channel * hidden;
    const float output_weight = pp[channel];
    int64_t d = 0;
#if defined(__AVX512F__)
    const __m512 output_scale = _mm512_set1_ps(output_weight);
    for (; d + 16 <= hidden; d += 16) {
      __m512 value = _mm512_mul_ps(
          _mm512_loadu_ps(op + d), output_scale);
      for (int64_t source = 0; source < hc; ++source) {
        value = _mm512_fmadd_ps(
            _mm512_loadu_ps(rp + source * hidden + d),
            _mm512_set1_ps(cp[source * hc + channel]),
            value);
      }
      _mm512_storeu_ps(destination + d, value);
    }
#endif
    for (; d < hidden; ++d) {
      float value = output_weight * op[d];
      for (int64_t source = 0; source < hc; ++source) {
        value += cp[source * hc + channel] *
                 rp[source * hidden + d];
      }
      destination[d] = value;
    }
  }
  return result;
}

std::vector<torch::Tensor> qkv_pre_cpu(
    torch::Tensor q_rank_raw,
    torch::Tensor kv_raw,
    torch::Tensor q_norm,
    torch::Tensor kv_norm,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    double rms_eps) {
  TORCH_CHECK(
      !q_rank_raw.is_cuda() && !kv_raw.is_cuda(),
      "CPU QKV preprocessing requires CPU tensors");
  TORCH_CHECK(
      q_rank_raw.scalar_type() == at::kFloat &&
          kv_raw.scalar_type() == at::kFloat &&
          q_rank_raw.dim() == 2 && q_rank_raw.size(0) == 1 &&
          kv_raw.dim() == 2 && kv_raw.size(0) == 1,
      "CPU QKV preprocessing requires float32 rows");
  auto qr = q_rank_raw.contiguous();
  auto kv = kv_raw.contiguous();
  auto qn = q_norm.to(torch::kFloat32).contiguous();
  auto kvn = kv_norm.to(torch::kFloat32).contiguous();
  auto cos = rope_cos.to(torch::kFloat32).contiguous();
  auto sin = rope_sin.to(torch::kFloat32).contiguous();
  const int64_t q_width = qr.size(1);
  const int64_t kv_width = kv.size(1);
  const int64_t rope_pairs = cos.numel();
  TORCH_CHECK(
      qn.numel() == q_width && kvn.numel() == kv_width &&
          sin.numel() == rope_pairs && rope_pairs * 2 <= kv_width,
      "CPU QKV preprocessing shape mismatch");
  auto q_out = torch::empty_like(qr);
  auto kv_out = torch::empty_like(kv);
  const float* qrp = qr.data_ptr<float>();
  const float* kvp = kv.data_ptr<float>();
  const float* qnp = qn.data_ptr<float>();
  const float* kvnp = kvn.data_ptr<float>();
  const float* cp = cos.data_ptr<float>();
  const float* sp = sin.data_ptr<float>();
  float* qop = q_out.data_ptr<float>();
  float* kvop = kv_out.data_ptr<float>();

  float q_square = float_dot(qrp, qrp, q_width);
  const float q_scale = 1.0f / std::sqrt(
      q_square / static_cast<float>(q_width) +
      static_cast<float>(rms_eps));
  for (int64_t index = 0; index < q_width; ++index) {
    qop[index] = qrp[index] * q_scale * qnp[index];
  }
  float kv_square = float_dot(kvp, kvp, kv_width);
  const float kv_scale = 1.0f / std::sqrt(
      kv_square / static_cast<float>(kv_width) +
      static_cast<float>(rms_eps));
  for (int64_t index = 0; index < kv_width; ++index) {
    kvop[index] = kvp[index] * kv_scale * kvnp[index];
  }
  const int64_t rope_start = kv_width - rope_pairs * 2;
  for (int64_t pair = 0; pair < rope_pairs; ++pair) {
    const int64_t index = rope_start + 2 * pair;
    const float first = kvop[index];
    const float second = kvop[index + 1];
    kvop[index] = first * cp[pair] - second * sp[pair];
    kvop[index + 1] = first * sp[pair] + second * cp[pair];
  }
  return {q_out, kv_out};
}

torch::Tensor q_post_cpu(
    torch::Tensor query,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    double rms_eps) {
  TORCH_CHECK(
      !query.is_cuda() && query.scalar_type() == at::kFloat &&
          query.dim() == 4 && query.size(0) * query.size(1) == 1,
      "CPU Q postprocessing requires one float32 token");
  auto q = query.contiguous();
  auto cos = rope_cos.to(torch::kFloat32).contiguous();
  auto sin = rope_sin.to(torch::kFloat32).contiguous();
  const int64_t heads = q.size(2);
  const int64_t head_dim = q.size(3);
  const int64_t rope_pairs = cos.numel();
  TORCH_CHECK(
      sin.numel() == rope_pairs && rope_pairs * 2 <= head_dim,
      "CPU Q postprocessing RoPE shape mismatch");
  auto output = torch::empty_like(q);
  const float* qp = q.data_ptr<float>();
  const float* cp = cos.data_ptr<float>();
  const float* sp = sin.data_ptr<float>();
  float* op = output.data_ptr<float>();
  const int64_t rope_start = head_dim - rope_pairs * 2;
#pragma omp parallel for schedule(static)
  for (int64_t head = 0; head < heads; ++head) {
    const float* source = qp + head * head_dim;
    float* destination = op + head * head_dim;
    const float square = float_dot(source, source, head_dim);
    const float scale = 1.0f / std::sqrt(
        square / static_cast<float>(head_dim) +
        static_cast<float>(rms_eps));
    for (int64_t index = 0; index < head_dim; ++index) {
      destination[index] = source[index] * scale;
    }
    for (int64_t pair = 0; pair < rope_pairs; ++pair) {
      const int64_t index = rope_start + 2 * pair;
      const float first = destination[index];
      const float second = destination[index + 1];
      destination[index] = first * cp[pair] - second * sp[pair];
      destination[index + 1] = first * sp[pair] + second * cp[pair];
    }
  }
  return output;
}

torch::Tensor q_int4_post_cpu(
    torch::Tensor q_rank,
    torch::Tensor packed,
    torch::Tensor scales,
    int64_t cols,
    int64_t group_size,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    int64_t heads,
    int64_t head_dim,
    double rms_eps) {
  TORCH_CHECK(
      !q_rank.is_cuda() && !packed.is_cuda() && !scales.is_cuda() &&
          q_rank.scalar_type() == at::kFloat &&
          q_rank.dim() == 2 && q_rank.size(0) == 1 &&
          q_rank.size(1) == cols && packed.scalar_type() == at::kByte &&
          scales.scalar_type() == at::kHalf &&
          packed.size(0) == heads * head_dim &&
          packed.size(1) * 2 == cols &&
          scales.size(0) == packed.size(0) &&
          scales.size(1) * group_size == cols,
      "CPU fused Q INT4 projection shape mismatch");
  auto x = q_rank.contiguous();
  auto q = packed.contiguous();
  auto s = scales.contiguous();
  auto cos = rope_cos.to(torch::kFloat32).contiguous();
  auto sin = rope_sin.to(torch::kFloat32).contiguous();
  const int64_t rope_pairs = cos.numel();
  TORCH_CHECK(
      sin.numel() == rope_pairs && rope_pairs * 2 <= head_dim,
      "CPU fused Q INT4 RoPE shape mismatch");
  auto output = torch::empty(
      {1, 1, heads, head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  const float* xp = x.data_ptr<float>();
  const uint8_t* qp = q.data_ptr<uint8_t>();
  const at::Half* sp = s.data_ptr<at::Half>();
  const float* cp = cos.data_ptr<float>();
  const float* sinp = sin.data_ptr<float>();
  float* op = output.data_ptr<float>();
  const int64_t rows = heads * head_dim;
  const int64_t bytes_per_row = cols / 2;
  const int64_t groups = cols / group_size;
  const int64_t rope_start = head_dim - rope_pairs * 2;
  const bool use_w4a8 = cpu_w4a8_enabled() && group_size == 64;
  const bool use_w4abf16 =
      cpu_w4abf16_enabled() && group_size == 64;
  const bool use_expand_bf16 = cpu_expand_bf16_enabled();
  Int8Activation quantized;
  Bf16Activation bf16_input;
  torch::Tensor expanded;
  const at::BFloat16* expandedp = nullptr;
  if (use_w4a8) {
    quantized = quantize_int8_activation(xp, cols, group_size);
  }
  if (use_w4abf16 || use_expand_bf16) {
    bf16_input =
        quantize_bf16_activation(xp, cols, group_size);
  }
  if (use_expand_bf16) {
    expanded =
        expand_int4_bf16(q, s, cols, group_size);
    expandedp = expanded.data_ptr<at::BFloat16>();
  }
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
      const uint8_t* weights = qp + row * bytes_per_row;
      const at::Half* row_scales = sp + row * groups;
      op[row] =
          use_expand_bf16
          ? bf16_row_dot(
                bf16_input.values.data(),
                expandedp + row * cols,
                cols)
          : use_w4abf16
          ? int4_row_dot_w4abf16(
                bf16_input, weights, row_scales)
          : use_w4a8
          ? int4_row_dot_w4a8(quantized, weights, row_scales)
          : int4_row_dot(
                xp, weights, row_scales, cols, group_size);
    }
#pragma omp for schedule(static)
    for (int64_t head = 0; head < heads; ++head) {
      float* destination = op + head * head_dim;
      const float square = float_dot(destination, destination, head_dim);
      const float scale = 1.0f / std::sqrt(
          square / static_cast<float>(head_dim) +
          static_cast<float>(rms_eps));
      for (int64_t index = 0; index < head_dim; ++index) {
        destination[index] *= scale;
      }
      for (int64_t pair = 0; pair < rope_pairs; ++pair) {
        const int64_t index = rope_start + 2 * pair;
        const float first = destination[index];
        const float second = destination[index + 1];
        destination[index] = first * cp[pair] - second * sinp[pair];
        destination[index + 1] =
            first * sinp[pair] + second * cp[pair];
      }
    }
  }
  return output;
}

torch::Tensor attention_decode_cpu(
    torch::Tensor query,
    torch::Tensor raw_values,
    torch::Tensor raw_positions,
    torch::Tensor selected_values,
    torch::Tensor sink,
    torch::Tensor rope_cos,
    torch::Tensor rope_sin,
    double scale) {
  TORCH_CHECK(
      !query.is_cuda() && !raw_values.is_cuda() &&
          !raw_positions.is_cuda() && !selected_values.is_cuda(),
      "CPU attention received a CUDA tensor");
  TORCH_CHECK(
      query.scalar_type() == at::kFloat &&
          raw_values.scalar_type() == at::kFloat &&
          selected_values.scalar_type() == at::kFloat &&
          sink.scalar_type() == at::kFloat,
      "CPU attention currently requires float32 operands");
  TORCH_CHECK(
      raw_positions.scalar_type() == at::kLong,
      "raw positions must be int64");
  TORCH_CHECK(query.dim() == 3, "query must be [B,H,D]");
  TORCH_CHECK(raw_values.dim() == 3, "raw values must be [B,W,D]");
  TORCH_CHECK(selected_values.dim() == 3,
              "selected values must be [B,K,D]");

  auto q = query.contiguous();
  auto raw = raw_values.contiguous();
  auto positions = raw_positions.contiguous();
  auto selected = selected_values.contiguous();
  auto sinks = sink.to(torch::kFloat32).contiguous();
  auto cos = rope_cos.to(torch::kFloat32).contiguous();
  auto sin = rope_sin.to(torch::kFloat32).contiguous();
  const int64_t batch = q.size(0);
  const int64_t heads = q.size(1);
  const int64_t dim = q.size(2);
  const int64_t raw_count = raw.size(1);
  const int64_t selected_count = selected.size(1);
  const int64_t total_count = raw_count + selected_count;
  const int64_t rope_pairs = cos.numel();
  TORCH_CHECK(raw.size(0) == batch && raw.size(2) == dim,
              "raw value shape mismatch");
  TORCH_CHECK(selected.size(0) == batch && selected.size(2) == dim,
              "selected value shape mismatch");
  TORCH_CHECK(
              positions.dim() == 2 &&
                  positions.size(0) == batch &&
                  positions.size(1) == raw_count,
              "raw position shape mismatch");
  TORCH_CHECK(sinks.numel() == heads, "attention sink shape mismatch");
  TORCH_CHECK(total_count <= 1024, "CPU attention source limit is 1024");
  TORCH_CHECK(rope_pairs * 2 <= dim, "RoPE width exceeds head width");

  const float* qp = q.data_ptr<float>();
  const float* rp = raw.data_ptr<float>();
  const int64_t* pp = positions.data_ptr<int64_t>();
  const float* vp = selected.data_ptr<float>();
  const float* skp = sinks.data_ptr<float>();
  const float* cp = cos.data_ptr<float>();
  const float* snp = sin.data_ptr<float>();
  auto output = torch::zeros_like(q);
  float* op = output.data_ptr<float>();
  const float score_scale = static_cast<float>(scale);

  at::parallel_for(0, batch * heads, 1, [&](int64_t begin, int64_t end) {
    for (int64_t item = begin; item < end; ++item) {
      const int64_t b = item / heads;
      const int64_t h = item - b * heads;
      const float* qrow = qp + item * dim;
      float* out = op + item * dim;
      float scores[1024];
      float maximum = -std::numeric_limits<float>::infinity();
      for (int64_t source = 0; source < raw_count; ++source) {
        if (pp[b * raw_count + source] < 0) {
          scores[source] = -std::numeric_limits<float>::infinity();
          continue;
        }
        const float value = float_dot(
            qrow, rp + (b * raw_count + source) * dim, dim) *
            score_scale;
        scores[source] = value;
        maximum = std::max(maximum, value);
      }
      for (int64_t source = 0; source < selected_count; ++source) {
        const float value = float_dot(
            qrow, vp + (b * selected_count + source) * dim, dim) *
            score_scale;
        scores[raw_count + source] = value;
        maximum = std::max(maximum, value);
      }
      float denominator = std::exp(skp[h] - maximum);
      for (int64_t source = 0; source < total_count; ++source) {
        if (!std::isfinite(scores[source])) {
          continue;
        }
        const float probability = std::exp(scores[source] - maximum);
        denominator += probability;
        const float* value = (
            source < raw_count
            ? rp + (b * raw_count + source) * dim
            : vp + (b * selected_count + source - raw_count) * dim);
        float_axpy(out, value, probability, dim);
      }
      const float inverse_denominator = 1.0f / denominator;
      for (int64_t d = 0; d < dim; ++d) {
        out[d] *= inverse_denominator;
      }
      const int64_t rope_start = dim - rope_pairs * 2;
      for (int64_t pair = 0; pair < rope_pairs; ++pair) {
        const int64_t offset = rope_start + pair * 2;
        const float first = out[offset];
        const float second = out[offset + 1];
        // inverse RoPE: sin is negated relative to the forward rotation.
        out[offset] = first * cp[pair] + second * snp[pair];
        out[offset + 1] = -first * snp[pair] + second * cp[pair];
      }
    }
  });
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<CpuMoeLayer>(module, "CpuMoeLayer")
      .def(
          pybind11::init<
              std::vector<torch::Tensor>,
              std::vector<torch::Tensor>,
              std::vector<torch::Tensor>,
              std::vector<torch::Tensor>,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              torch::Tensor,
              int64_t,
              double,
              int64_t,
              bool,
              double>())
      .def("forward", &CpuMoeLayer::forward)
      .def("forward_learned", &CpuMoeLayer::forward_learned);
  module.def(
      "vq_gemv",
      &vq_gemv_cpu,
      "TPQ uint8/uint16 VQ GEMV for CPU");
  module.def(
      "vq_gemv_list",
      &vq_gemv_list_cpu,
      "TPQ list-backed uint8/uint16 VQ GEMV for CPU");
  module.def(
      "vq_gemv_packed_list",
      &vq_gemv_packed_list_cpu,
      "TPQ list-backed packed 8/12/14/16-bit VQ GEMV for CPU");
  module.def(
      "moe_packed_topk",
      &moe_packed_topk_cpu,
      "TPQ persistent-pool mixed packed Top-K MoE for CPU");
  module.def(
      "reset_packed_moe_phase_profile",
      &reset_packed_moe_phase_profile_cpu,
      "Reset mixed packed Top-K CPU MoE phase timers");
  module.def(
      "packed_moe_phase_profile",
      &packed_moe_phase_profile_cpu,
      "Read mixed packed Top-K CPU MoE phase timers");
  module.def(
      "kda_recurrent",
      &kda_recurrent_cpu,
      "TPQ fused AVX-512 KDA recurrence for CPU");
  module.def(
      "short_conv3",
      &short_conv3_cpu,
      "TPQ fused three-stream short convolution for CPU");
  module.def(
      "gated_rmsnorm",
      &gated_rmsnorm_cpu,
      "TPQ fused gated RMSNorm for CPU");
  module.def(
      "moe_mixed",
      &moe_mixed_cpu,
      "TPQ fused routed VQ and shared INT4 MoE for CPU");
  module.def(
      "reset_moe_phase_profile",
      &reset_moe_phase_profile_cpu,
      "Reset TPQ CPU MoE phase timers");
  module.def(
      "moe_phase_profile",
      &moe_phase_profile_cpu,
      "Read TPQ CPU MoE phase timers");
  module.def("int4_gemv", &int4_gemv_cpu, "TPQ packed INT4 GEMV for CPU");
  module.def(
      "int4_gemv_many",
      &int4_gemv_many_cpu,
      "TPQ shared-input packed INT4 GEMVs for CPU");
  module.def(
      "int4_grouped_gemv",
      &int4_grouped_gemv_cpu,
      "TPQ grouped-input packed INT4 GEMV for CPU");
  module.def(
      "o_proj_int4",
      &o_proj_int4_cpu,
      "TPQ fused grouped and dense packed INT4 O projection for CPU");
  module.def(
      "hc_pre_norm",
      &hc_pre_norm_cpu,
      "TPQ fused Hyper-Connection pre and RMSNorm for CPU");
  module.def(
      "hc_post",
      &hc_post_cpu,
      "TPQ fused Hyper-Connection post for CPU");
  module.def(
      "qkv_pre",
      &qkv_pre_cpu,
      "TPQ fused Q-rank/KV RMSNorm and KV RoPE for CPU");
  module.def(
      "q_post",
      &q_post_cpu,
      "TPQ fused per-head Q RMSNorm and RoPE for CPU");
  module.def(
      "q_int4_post",
      &q_int4_post_cpu,
      "TPQ fused packed INT4 Q projection, RMSNorm and RoPE for CPU");
  module.def(
      "attention_decode",
      &attention_decode_cpu,
      "TPQ fused single-token attention for CPU");
}
