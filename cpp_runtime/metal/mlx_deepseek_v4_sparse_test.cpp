#include "mlx_deepseek_v4_sparse.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

using mlx::core::Shape;
using mlx::core::array;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

array float_array(
    const std::vector<float>& values,
    Shape shape) {
    return array(values.begin(), std::move(shape));
}

array int_array(
    const std::vector<std::int32_t>& values,
    Shape shape) {
    return array(values.begin(), std::move(shape));
}

std::vector<float> evaluated_float(array value) {
    if (value.dtype() != mlx::core::float32) {
        value = mlx::core::astype(
            value,
            mlx::core::float32);
    }
    value.eval();
    return {
        value.data<float>(),
        value.data<float>() + value.size(),
    };
}

std::vector<std::int32_t> evaluated_int(array value) {
    if (value.dtype() != mlx::core::int32) {
        value = mlx::core::astype(
            value,
            mlx::core::int32);
    }
    value.eval();
    return {
        value.data<std::int32_t>(),
        value.data<std::int32_t>() + value.size(),
    };
}

void require_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance,
    const std::string& label) {
    require(
        actual.size() == expected.size(),
        label + " size mismatch");
    for (std::size_t index = 0;
         index < actual.size();
         ++index) {
        const float observed = actual[index];
        const float wanted = expected[index];
        if (std::isinf(observed) || std::isinf(wanted)) {
            if (!(std::isinf(observed) &&
                  std::isinf(wanted) &&
                  std::signbit(observed) ==
                      std::signbit(wanted))) {
                throw std::runtime_error(
                    label + " infinity mismatch at " +
                    std::to_string(index));
            }
            continue;
        }
        if (!std::isfinite(observed) ||
            std::fabs(observed - wanted) > tolerance) {
            throw std::runtime_error(
                label + " mismatch at " +
                std::to_string(index) +
                ": actual=" +
                std::to_string(observed) +
                " expected=" +
                std::to_string(wanted));
        }
    }
}

template <typename Function>
void require_invalid(
    Function&& function,
    const std::string& label) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, label + " was accepted");
}

float pow2_ceil(float value) {
    if (value <= 0.0f) {
        return std::numeric_limits<float>::min();
    }
    return std::exp2(std::ceil(std::log2(value)));
}

float fp4_value(float value, float scale) {
    const float normalized =
        std::clamp(value / scale, -6.0f, 6.0f);
    const float magnitude = std::fabs(normalized);
    float quantized = 0.0f;
    if (magnitude <= 0.25f) {
        quantized = 0.0f;
    } else if (magnitude < 0.75f) {
        quantized = 0.5f;
    } else if (magnitude <= 1.25f) {
        quantized = 1.0f;
    } else if (magnitude < 1.75f) {
        quantized = 1.5f;
    } else if (magnitude <= 2.5f) {
        quantized = 2.0f;
    } else if (magnitude < 3.5f) {
        quantized = 3.0f;
    } else if (magnitude <= 5.0f) {
        quantized = 4.0f;
    } else {
        quantized = 6.0f;
    }
    return std::copysign(
        quantized * scale,
        normalized);
}

void test_fp4_sim() {
    std::vector<float> input(64);
    for (int index = 0; index < 32; ++index) {
        input[index] =
            static_cast<float>(index - 15) * 0.25f;
        input[32 + index] =
            static_cast<float>(index - 16) * 0.5f;
    }
    std::vector<float> expected(input.size());
    for (int group = 0; group < 2; ++group) {
        float maximum = 0.0f;
        for (int lane = 0; lane < 32; ++lane) {
            maximum = std::max(
                maximum,
                std::fabs(input[group * 32 + lane]));
        }
        const float scale = pow2_ceil(
            std::max(
                maximum,
                6.0f *
                    std::numeric_limits<float>::min()) /
            6.0f);
        for (int lane = 0; lane < 32; ++lane) {
            expected[group * 32 + lane] = fp4_value(
                input[group * 32 + lane],
                scale);
        }
    }
    auto actual = mfq::metal::dsv4_fp4_sim(
        float_array(input, Shape{2, 32}));
    require(
        actual.shape() == Shape{2, 32},
        "FP4 output shape mismatch");
    require_close(
        evaluated_float(std::move(actual)),
        expected,
        1e-3f,
        "FP4");
}

struct CompressorFixture {
    int head_dim;
    int ratio;
    bool overlap;
    std::vector<float> kv;
    std::vector<float> gate;
    std::vector<float> ape;
    std::vector<float> norm;
    std::vector<float> cos;
    std::vector<float> sin;

    CompressorFixture(
        int dimension,
        int compression_ratio,
        bool use_overlap = false)
        : head_dim(dimension),
          ratio(compression_ratio),
          overlap(use_overlap),
          kv(
              static_cast<std::size_t>(compression_ratio) *
                  dimension * (use_overlap ? 2 : 1),
              0.5f),
          gate(kv.size(), 0.0f),
          ape(kv.size(), 0.0f),
          norm(dimension, 1.0f),
          cos(32, 1.0f),
          sin(32, 0.0f) {}

    int output_dim() const {
        return head_dim * (overlap ? 2 : 1);
    }
};

array run_compress(
    const CompressorFixture& fixture,
    int quant_mode = 0,
    const std::optional<array>& previous_kv =
        std::nullopt,
    const std::optional<array>& previous_gate =
        std::nullopt,
    bool use_float16 = true) {
    auto kv = float_array(
        fixture.kv,
        Shape{
            1,
            1,
            fixture.ratio,
            fixture.output_dim(),
        });
    auto gate = float_array(
        fixture.gate,
        Shape{
            1,
            1,
            fixture.ratio,
            fixture.output_dim(),
        });
    if (use_float16) {
        kv = mlx::core::astype(kv, mlx::core::float16);
        gate = mlx::core::astype(
            gate,
            mlx::core::float16);
    }
    return mfq::metal::dsv4_compress(
        kv,
        gate,
        float_array(
            fixture.ape,
            Shape{
                fixture.ratio,
                fixture.output_dim(),
            }),
        float_array(
            fixture.norm,
            Shape{fixture.head_dim}),
        previous_kv,
        previous_gate,
        int_array({0}, Shape{1}),
        float_array(fixture.cos, Shape{1, 32}),
        float_array(fixture.sin, Shape{1, 32}),
        fixture.ratio,
        fixture.overlap,
        quant_mode);
}

void test_compressor_quantization() {
    const CompressorFixture base(128, 2);
    require_close(
        evaluated_float(run_compress(base)),
        std::vector<float>(128, 1.0f),
        1e-3f,
        "compressor f16/BF16");
    require_close(
        evaluated_float(
            run_compress(
                base,
                0,
                std::nullopt,
                std::nullopt,
                false)),
        std::vector<float>(128, 1.0f),
        1e-3f,
        "compressor f32/BF16");

    const CompressorFixture fp4(128, 1);
    auto fp4_values =
        evaluated_float(run_compress(fp4, 2));
    std::vector<float> fp4_expected(128, 0.0f);
    fp4_expected[0] = 12.0f;
    require_close(
        fp4_values,
        fp4_expected,
        1e-3f,
        "compressor FP4 Hadamard");

    const CompressorFixture fp8(512, 1);
    require_close(
        evaluated_float(run_compress(fp8, 1)),
        std::vector<float>(512, 1.0f),
        2e-3f,
        "compressor FP8");

    const CompressorFixture overlap(128, 2, true);
    const auto previous_kv = mlx::core::astype(
        float_array(
            std::vector<float>(256, 0.5f),
            Shape{1, 2, 128}),
        mlx::core::float16);
    const auto previous_gate = mlx::core::zeros(
        Shape{1, 2, 128},
        mlx::core::float16);
    require_close(
        evaluated_float(
            run_compress(
                overlap,
                0,
                previous_kv,
                previous_gate)),
        std::vector<float>(128, 1.0f),
        1e-3f,
        "overlap compressor history");
}

mfq::metal::MlxDsv4PoolStep pool_step(
    const CompressorFixture& fixture,
    const array& token,
    const array& state_kv,
    const array& state_gate,
    int length,
    const std::optional<array>& previous_kv =
        std::nullopt,
    const std::optional<array>& previous_gate =
        std::nullopt) {
    return mfq::metal::dsv4_decode_pool_step(
        token,
        token * array(0.0f),
        float_array(
            fixture.ape,
            Shape{
                fixture.ratio,
                fixture.output_dim(),
            }),
        float_array(
            fixture.norm,
            Shape{fixture.head_dim}),
        state_kv,
        state_gate,
        previous_kv,
        previous_gate,
        int_array({length}, Shape{1}),
        float_array(fixture.cos, Shape{1, 32}),
        float_array(fixture.sin, Shape{1, 32}),
        fixture.ratio,
        fixture.overlap);
}

void test_decode_pool_state_and_bounds() {
    const CompressorFixture fixture(128, 2);
    const auto token = mlx::core::astype(
        float_array(
            std::vector<float>(128, 0.5f),
            Shape{1, 1, 128}),
        mlx::core::float16);
    auto state_kv = mlx::core::zeros(
        Shape{1, 2, 128},
        mlx::core::float16);
    auto state_gate = mlx::core::zeros(
        Shape{1, 2, 128},
        mlx::core::float16);

    auto first = pool_step(
        fixture,
        token,
        state_kv,
        state_gate,
        1);
    require(
        evaluated_int(first.emit_rows) ==
            std::vector<std::int32_t>{-1},
        "decode compressor emitted before ratio boundary");
    auto first_state =
        evaluated_float(first.state_kv);
    for (int dimension = 0;
         dimension < 128;
         ++dimension) {
        require(
            std::fabs(first_state[dimension] - 0.5f) <
                1e-3f,
            "decode compressor failed to update slot zero");
        require(
            std::fabs(first_state[128 + dimension]) <
                1e-6f,
            "decode compressor touched the wrong slot");
    }

    auto second = pool_step(
        fixture,
        token,
        first.state_kv,
        first.state_gate,
        2);
    require(
        evaluated_int(second.emit_rows) ==
            std::vector<std::int32_t>{0},
        "decode compressor boundary row mismatch");
    require_close(
        evaluated_float(second.emitted),
        std::vector<float>(128, 1.0f),
        1e-3f,
        "decode compressor emitted row");

    auto valid_update =
        mfq::metal::dsv4_decode_pool_update(
            token,
            token * array(0.0f),
            float_array(
                fixture.ape,
                Shape{2, 128}),
            float_array(
                fixture.norm,
                Shape{128}),
            first.state_kv,
            first.state_gate,
            std::nullopt,
            std::nullopt,
            mlx::core::zeros(
                Shape{1, 1, 128},
                mlx::core::float16),
            int_array({2}, Shape{1}),
            float_array(fixture.cos, Shape{1, 32}),
            float_array(fixture.sin, Shape{1, 32}),
            2,
            false);
    require_close(
        evaluated_float(valid_update.pool),
        std::vector<float>(128, 1.0f),
        1e-3f,
        "decode pool valid update");

    auto third = pool_step(
        fixture,
        token,
        second.state_kv,
        second.state_gate,
        3);
    const auto sentinel = float_array(
        std::vector<float>(128, 0.25f),
        Shape{1, 1, 128});
    auto bounded_update =
        mfq::metal::dsv4_decode_pool_update(
            token,
            token * array(0.0f),
            float_array(
                fixture.ape,
                Shape{2, 128}),
            float_array(
                fixture.norm,
                Shape{128}),
            third.state_kv,
            third.state_gate,
            std::nullopt,
            std::nullopt,
            sentinel,
            int_array({4}, Shape{1}),
            float_array(fixture.cos, Shape{1, 32}),
            float_array(fixture.sin, Shape{1, 32}),
            2,
            false);
    require_close(
        evaluated_float(bounded_update.pool),
        std::vector<float>(128, 0.25f),
        1e-3f,
        "decode pool capacity boundary");
}

void test_fixed_cache_write() {
    auto cache = mlx::core::zeros(
        Shape{2, 5, 3},
        mlx::core::float16);
    cache.eval();
    const void* allocation = cache.buffer().ptr();
    auto updated = mfq::metal::dsv4_cache_write_inplace(
        cache,
        mlx::core::astype(
            float_array(
                {
                    1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f,
                    7.0f, 8.0f, 9.0f,
                    10.0f, 11.0f, 12.0f,
                },
                Shape{2, 2, 3}),
            mlx::core::float16),
        int_array(
            {1, 4, 0, 3},
            Shape{2, 2}));
    updated.eval();
    require(
        updated.buffer().ptr() == allocation,
        "fixed cache update replaced the Metal allocation");
    require_close(
        evaluated_float(updated),
        {
            0.0f, 0.0f, 0.0f,
            1.0f, 2.0f, 3.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            10.0f, 11.0f, 12.0f,
            0.0f, 0.0f, 0.0f,
        },
        1e-3f,
        "fixed cache update");
}

void test_overlap_state() {
    const CompressorFixture fixture(128, 2, true);
    const auto token = mlx::core::astype(
        float_array(
            std::vector<float>(256, 0.5f),
            Shape{1, 1, 256}),
        mlx::core::float16);
    auto state_kv = mlx::core::zeros(
        Shape{1, 2, 256},
        mlx::core::float16);
    auto state_gate = mlx::core::zeros(
        Shape{1, 2, 256},
        mlx::core::float16);
    auto previous_kv = mlx::core::zeros(
        Shape{1, 2, 128},
        mlx::core::float16);
    auto previous_gate = mlx::core::zeros(
        Shape{1, 2, 128},
        mlx::core::float16);
    auto first = pool_step(
        fixture,
        token,
        state_kv,
        state_gate,
        1,
        previous_kv,
        previous_gate);
    auto second = pool_step(
        fixture,
        token,
        first.state_kv,
        first.state_gate,
        2,
        first.prev_kv,
        first.prev_gate);
    require(second.prev_kv.has_value(),
            "overlap state was dropped");
    require_close(
        evaluated_float(*second.prev_kv),
        std::vector<float>(256, 0.5f),
        1e-3f,
        "overlap previous KV rotation");
    require_close(
        evaluated_float(second.emitted),
        std::vector<float>(128, 1.0f),
        1e-3f,
        "overlap emitted row");
}

void test_indexer_paths() {
    constexpr int keys = 70;
    constexpr int queries = 2;
    const std::vector<float> query(
        queries * 64 * 128,
        1.0f);
    const std::vector<float> key(
        keys * 128,
        0.5f);
    const std::vector<float> weights(
        queries * 64,
        1.0f / 64.0f);
    auto scores = mfq::metal::dsv4_indexer_scores(
        float_array(
            query,
            Shape{1, queries, 64, 128}),
        float_array(key, Shape{1, keys, 128}),
        float_array(weights, Shape{1, queries, 64}),
        128,
        2);
    const auto actual =
        evaluated_float(std::move(scores));
    const float expected_score =
        64.0f / std::sqrt(8192.0f);
    for (int query_index = 0;
         query_index < queries;
         ++query_index) {
        const int visible = 64 + query_index;
        for (int key_index = 0;
             key_index < keys;
             ++key_index) {
            const float value =
                actual[query_index * keys + key_index];
            if (key_index < visible) {
                require(
                    std::fabs(value - expected_score) <
                        1.5e-3f,
                    "MMA indexer score mismatch");
            } else {
                require(
                    std::isinf(value) &&
                        std::signbit(value),
                    "MMA indexer visibility mismatch");
            }
        }
    }

    constexpr int decode_keys = 129;
    auto decode = mfq::metal::dsv4_indexer_scores_decode(
        float_array(
            std::vector<float>(64 * 128, 1.0f),
            Shape{1, 1, 64, 128}),
        float_array(
            std::vector<float>(decode_keys * 128, 0.5f),
            Shape{1, decode_keys, 128}),
        float_array(
            std::vector<float>(64, 1.0f / 64.0f),
            Shape{1, 1, 64}),
        130,
        2);
    const auto decode_values =
        evaluated_float(std::move(decode));
    for (int key_index = 0;
         key_index < decode_keys;
         ++key_index) {
        if (key_index < 65) {
            require(
                std::fabs(
                    decode_values[key_index] -
                    expected_score) < 1.5e-3f,
                "decode indexer score mismatch");
        } else {
            require(
                std::isinf(decode_values[key_index]) &&
                    std::signbit(decode_values[key_index]),
                "decode indexer visibility mismatch");
        }
    }

    constexpr int fixed_capacity = 257;
    constexpr int fixed_prefix = 129;
    auto fixed_scores = mfq::metal::dsv4_indexer_scores_decode(
        float_array(
            std::vector<float>(64 * 128, 1.0f),
            Shape{1, 1, 64, 128}),
        float_array(
            std::vector<float>(
                fixed_capacity * 128,
                0.5f),
            Shape{1, fixed_capacity, 128}),
        float_array(
            std::vector<float>(64, 1.0f / 64.0f),
            Shape{1, 1, 64}),
        258,
        2,
        fixed_prefix);
    require(
        fixed_scores.shape() == Shape{1, 1, fixed_capacity},
        "fixed decode indexer scratch shape mismatch");
    const auto fixed_topk = evaluated_int(
        mfq::metal::dsv4_topk512(
            fixed_scores,
            true,
            fixed_prefix));
    for (int index = 0; index < 512; ++index) {
        require(
            fixed_topk[index] ==
                (index < fixed_prefix ? index : 0),
            "fixed decode indexer valid-prefix mismatch");
    }
}

void test_topk() {
    constexpr int keys = 700;
    std::vector<float> values(keys);
    for (int index = 0; index < keys; ++index) {
        values[index] = static_cast<float>(index);
    }
    const auto first = evaluated_int(
        mfq::metal::dsv4_topk512(
            float_array(values, Shape{1, 1, keys})));
    const auto second = evaluated_int(
        mfq::metal::dsv4_topk512(
            float_array(values, Shape{1, 1, keys})));
    require(first == second,
            "deterministic top-k changed across launches");
    auto sorted = first;
    std::sort(sorted.begin(), sorted.end());
    for (int index = 0; index < 512; ++index) {
        require(
            sorted[index] == index + keys - 512,
            "top-k membership mismatch");
    }
    auto atomic = evaluated_int(
        mfq::metal::dsv4_topk512(
            float_array(values, Shape{1, 1, keys}),
            false));
    std::sort(atomic.begin(), atomic.end());
    require(
        atomic == sorted,
        "atomic top-k membership mismatch");
    const auto tied = evaluated_int(
        mfq::metal::dsv4_topk512(
            float_array(
                std::vector<float>(keys, 1.0f),
                Shape{1, 1, keys})));
    for (int index = 0; index < 512; ++index) {
        require(
            tied[index] == index,
            "deterministic top-k tie order mismatch");
    }

    const auto short_result = evaluated_int(
        mfq::metal::dsv4_topk512(
            float_array(
                std::vector<float>{
                    0.0f,
                    1.0f,
                    2.0f,
                    3.0f,
                    4.0f,
                    5.0f,
                    6.0f,
                },
                Shape{1, 1, 7})));
    for (int index = 0; index < 512; ++index) {
        const int expected = index < 7 ? index : 0;
        require(
            short_result[index] == expected,
            "short top-k padding mismatch");
    }
}

void test_sparse_plans() {
    auto prefill = mfq::metal::dsv4_build_prefill_plan(
        int_array(
            {
                0,
                5,
                -1,
                1,
                6,
                100,
            },
            Shape{1, 2, 3}),
        8,
        3,
        6,
        2,
        4);
    const auto prefill_indices =
        evaluated_int(prefill.first);
    const auto prefill_mask =
        evaluated_float(prefill.second);
    const std::vector<int> first_local{0, 1, 2, 3};
    const std::vector<int> second_local{1, 2, 3, 4};
    for (int slot = 0; slot < 4; ++slot) {
        require(
            prefill_indices[slot] ==
                first_local[slot] &&
                prefill_mask[slot] == 0.0f,
            "prefill first local plan mismatch");
        require(
            prefill_indices[32 + slot] ==
                second_local[slot] &&
                prefill_mask[32 + slot] == 0.0f,
            "prefill second local plan mismatch");
    }
    require(
        prefill_indices[4] == 5 &&
            prefill_mask[4] == 0.0f,
        "prefill pooled index mismatch");
    require(
        prefill_indices[32 + 4] == 6 &&
            prefill_mask[32 + 4] == 0.0f,
        "prefill second pooled index mismatch");
    for (const int slot : {5, 6, 7, 31}) {
        require(
            std::isinf(prefill_mask[slot]) &&
                std::signbit(prefill_mask[slot]),
            "prefill invalid mask mismatch");
    }

    auto decode = mfq::metal::dsv4_build_decode_plan(
        int_array(
            {
                0,
                1,
                2,
                3,
                4,
                -1,
            },
            Shape{2, 1, 3}),
        int_array({5, 12}, Shape{2}),
        4,
        2,
        4);
    const auto decode_indices =
        evaluated_int(decode.first);
    const auto decode_mask =
        evaluated_float(decode.second);
    const std::vector<int> local0{1, 2, 3, 0};
    const std::vector<int> local1{0, 1, 2, 3};
    for (int slot = 0; slot < 4; ++slot) {
        require(
            decode_indices[slot] == local0[slot] &&
                decode_mask[slot] == 0.0f,
            "decode first local plan mismatch");
        require(
            decode_indices[32 + slot] ==
                    local1[slot] &&
                decode_mask[32 + slot] == 0.0f,
            "decode second local plan mismatch");
    }
    require(
        decode_indices[4] == 4 &&
            decode_indices[5] == 5 &&
            decode_mask[4] == 0.0f &&
            decode_mask[5] == 0.0f,
        "decode first pooled plan mismatch");
    require(
        decode_indices[32 + 4] == 7 &&
            decode_mask[32 + 4] == 0.0f,
        "decode second pooled plan mismatch");
    require(
        std::isinf(decode_mask[6]) &&
            std::isinf(decode_mask[32 + 5]),
        "decode invalid pooled mask mismatch");
}

std::vector<float> sparse_reference(
    int queries,
    const std::vector<float>& query,
    const std::vector<float>& cache,
    const std::vector<std::int32_t>& indices,
    const std::vector<float>& mask,
    const std::vector<float>& sinks,
    float scale) {
    constexpr int heads = 64;
    constexpr int dimension = 512;
    constexpr int max_seq = 4;
    constexpr int selected = 32;
    std::vector<float> output(
        static_cast<std::size_t>(queries) *
            heads * dimension,
        0.0f);
    for (int query_index = 0;
         query_index < queries;
         ++query_index) {
        for (int head = 0; head < heads; ++head) {
            std::vector<float> scores;
            std::vector<int> rows;
            scores.push_back(sinks[head]);
            rows.push_back(-1);
            for (int item = 0; item < selected; ++item) {
                const int plan_index =
                    query_index * selected + item;
                const int row = indices[plan_index];
                if (!std::isfinite(mask[plan_index]) ||
                    row < 0 || row >= max_seq) {
                    continue;
                }
                float dot = 0.0f;
                const int query_base =
                    (head * queries + query_index) *
                    dimension;
                const int cache_base = row * dimension;
                for (int feature = 0;
                     feature < dimension;
                     ++feature) {
                    dot += query[query_base + feature] *
                        cache[cache_base + feature];
                }
                scores.push_back(
                    dot * scale + mask[plan_index]);
                rows.push_back(row);
            }
            const float maximum = *std::max_element(
                scores.begin(),
                scores.end());
            float denominator = 0.0f;
            std::vector<float> probabilities(
                scores.size());
            for (std::size_t index = 0;
                 index < scores.size();
                 ++index) {
                probabilities[index] =
                    std::exp(scores[index] - maximum);
                denominator += probabilities[index];
            }
            const int output_base =
                (query_index * heads + head) *
                dimension;
            for (std::size_t item = 1;
                 item < rows.size();
                 ++item) {
                const float probability =
                    probabilities[item] / denominator;
                const int cache_base =
                    rows[item] * dimension;
                for (int feature = 0;
                     feature < dimension;
                     ++feature) {
                    output[output_base + feature] +=
                        probability *
                        cache[cache_base + feature];
                }
            }
        }
    }
    return output;
}

void test_sparse_attention_path(int queries) {
    constexpr int heads = 64;
    constexpr int dimension = 512;
    constexpr int max_seq = 4;
    constexpr int selected = 32;
    const float scale =
        1.0f / std::sqrt(512.0f);
    std::vector<float> query(
        static_cast<std::size_t>(heads) *
            queries * dimension,
        0.0f);
    for (int head = 0; head < heads; ++head) {
        for (int query_index = 0;
             query_index < queries;
             ++query_index) {
            query[
                (head * queries + query_index) *
                dimension] =
                0.5f +
                static_cast<float>(head) / 64.0f +
                static_cast<float>(query_index % 4) /
                    8.0f;
        }
    }
    std::vector<float> cache(max_seq * dimension, 0.0f);
    const std::vector<float> key0{
        1.0f,
        -2.0f,
        0.5f,
        3.0f,
    };
    const std::vector<float> value1{
        0.25f,
        0.5f,
        -0.75f,
        1.0f,
    };
    for (int row = 0; row < max_seq; ++row) {
        cache[row * dimension] = key0[row];
        cache[row * dimension + 1] = value1[row];
    }
    std::vector<std::int32_t> indices(
        queries * selected,
        0);
    std::vector<float> mask(
        queries * selected,
        -std::numeric_limits<float>::infinity());
    for (int query_index = 0;
         query_index < queries;
         ++query_index) {
        const int base = query_index * selected;
        indices[base] = 0;
        indices[base + 1] = 1;
        indices[base + 2] = 3;
        indices[base + 3] = -1;
        indices[base + 4] = max_seq;
        mask[base] = 0.0f;
        mask[base + 1] = -0.25f;
        mask[base + 2] = 0.125f;
        mask[base + 3] = 0.0f;
        mask[base + 4] = 0.0f;
    }
    std::vector<float> sinks(heads);
    for (int head = 0; head < heads; ++head) {
        sinks[head] =
            -0.5f + static_cast<float>(head) / 64.0f;
    }
    auto output = mfq::metal::attention_dsv4_sparse(
        float_array(
            query,
            Shape{1, heads, queries, dimension}),
        float_array(
            cache,
            Shape{1, max_seq, dimension}),
        int_array(
            indices,
            Shape{1, queries, selected}),
        float_array(
            mask,
            Shape{1, queries, selected}),
        float_array(sinks, Shape{heads}));
    require(
        output.shape() ==
            Shape{1, queries, heads, dimension},
        "sparse attention output shape mismatch");
    const auto expected = sparse_reference(
        queries,
        query,
        cache,
        indices,
        mask,
        sinks,
        scale);
    require_close(
        evaluated_float(std::move(output)),
        expected,
        queries >= 32 ? 4e-3f : 7e-4f,
        "sparse attention M=" +
            std::to_string(queries));
}

void test_direct_decode_attention_path() {
    constexpr int heads = 64;
    constexpr int dimension = 512;
    constexpr int window = 4;
    constexpr int pool_len = 2;
    constexpr int ratio = 2;
    constexpr int seq_len = 5;
    std::vector<float> query(
        static_cast<std::size_t>(heads) * dimension,
        0.0f);
    for (int head = 0; head < heads; ++head) {
        query[head * dimension] =
            0.25f + static_cast<float>(head) / 128.0f;
        query[head * dimension + 7] = -0.375f;
    }
    std::vector<float> local(window * dimension, 0.0f);
    std::vector<float> pool(pool_len * dimension, 0.0f);
    for (int row = 0; row < window; ++row) {
        local[row * dimension] =
            -0.5f + static_cast<float>(row) * 0.4f;
        local[row * dimension + 7] =
            0.75f - static_cast<float>(row) * 0.2f;
    }
    for (int row = 0; row < pool_len; ++row) {
        pool[row * dimension] =
            1.25f + static_cast<float>(row) * 0.5f;
        pool[row * dimension + 7] =
            -0.25f + static_cast<float>(row) * 0.125f;
    }
    std::vector<float> sinks(heads);
    for (int head = 0; head < heads; ++head) {
        sinks[head] =
            -0.75f + static_cast<float>(head) / 96.0f;
    }
    auto local_array = float_array(
        local,
        Shape{1, window, dimension});
    auto pool_array = float_array(
        pool,
        Shape{1, pool_len, dimension});
    auto topk = int_array(
        {0, 1},
        Shape{1, 1, pool_len});
    auto sink_array = float_array(
        sinks,
        Shape{heads});
    auto plan = mfq::metal::dsv4_build_decode_plan(
        topk,
        int_array({seq_len}, Shape{1}),
        pool_len,
        ratio,
        window);
    auto legacy = mfq::metal::attention_dsv4_sparse(
        float_array(
            query,
            Shape{1, heads, 1, dimension}),
        mlx::core::concatenate(
            {local_array, pool_array},
            1),
        plan.first,
        plan.second,
        sink_array);
    auto direct = mfq::metal::attention_dsv4_sparse_decode(
        float_array(
            query,
            Shape{1, heads, 1, dimension}),
        local_array,
        pool_array,
        pool_len,
        topk,
        sink_array,
        seq_len,
        ratio,
        window);
    require_close(
        evaluated_float(std::move(direct)),
        evaluated_float(std::move(legacy)),
        1e-6f,
        "direct sparse decode attention");
}

void test_invalid_inputs() {
    require_invalid(
        [] {
            (void)mfq::metal::dsv4_fp4_sim(
                mlx::core::zeros(
                    Shape{31},
                    mlx::core::float16));
        },
        "invalid FP4 width");
    require_invalid(
        [] {
            (void)mfq::metal::dsv4_compress(
                mlx::core::zeros(
                    Shape{1, 1, 1, 128},
                    mlx::core::float16),
                mlx::core::zeros(
                    Shape{1, 1, 1, 128},
                    mlx::core::float16),
                mlx::core::zeros(
                    Shape{2, 128},
                    mlx::core::float32),
                mlx::core::ones(
                    Shape{128},
                    mlx::core::float32),
                std::nullopt,
                std::nullopt,
                int_array({0}, Shape{1}),
                mlx::core::ones(
                    Shape{1, 32},
                    mlx::core::float32),
                mlx::core::zeros(
                    Shape{1, 32},
                    mlx::core::float32),
                2,
                false);
        },
        "invalid compressor ratio shape");
    require_invalid(
        [] {
            const CompressorFixture fixture(
                128,
                2,
                true);
            (void)pool_step(
                fixture,
                mlx::core::zeros(
                    Shape{1, 1, 256},
                    mlx::core::float16),
                mlx::core::zeros(
                    Shape{1, 2, 256},
                    mlx::core::float16),
                mlx::core::zeros(
                    Shape{1, 2, 256},
                    mlx::core::float16),
                1);
        },
        "missing overlap state");
    require_invalid(
        [] {
            (void)mfq::metal::dsv4_indexer_scores(
                mlx::core::zeros(
                    Shape{1, 1, 63, 128},
                    mlx::core::float16),
                mlx::core::zeros(
                    Shape{1, 1, 128},
                    mlx::core::float16),
                mlx::core::zeros(
                    Shape{1, 1, 64},
                    mlx::core::float16),
                0,
                1);
        },
        "invalid indexer heads");
    require_invalid(
        [] {
            (void)mfq::metal::dsv4_topk512(
                mlx::core::zeros(
                    Shape{1, 1, 0},
                    mlx::core::float16));
        },
        "empty top-k input");
    require_invalid(
        [] {
            (void)mfq::metal::dsv4_build_prefill_plan(
                mlx::core::zeros(
                    Shape{1, 1, 1},
                    mlx::core::int32),
                0,
                0,
                0,
                1,
                0);
        },
        "invalid prefill window");
    require_invalid(
        [] {
            (void)mfq::metal::attention_dsv4_sparse(
                mlx::core::zeros(
                    Shape{1, 64, 1, 512},
                    mlx::core::float32),
                mlx::core::zeros(
                    Shape{1, 1, 512},
                    mlx::core::float16),
                mlx::core::zeros(
                    Shape{1, 1, 31},
                    mlx::core::int32),
                mlx::core::zeros(
                    Shape{1, 1, 31},
                    mlx::core::float16),
                mlx::core::zeros(
                    Shape{64},
                    mlx::core::float32));
        },
        "unaligned sparse selection");
}

} // namespace

int main() {
    try {
        test_fp4_sim();
        test_compressor_quantization();
        test_fixed_cache_write();
        test_decode_pool_state_and_bounds();
        test_overlap_state();
        test_indexer_paths();
        test_topk();
        test_sparse_plans();
        test_sparse_attention_path(1);
        test_sparse_attention_path(2);
        test_sparse_attention_path(32);
        test_direct_decode_attention_path();
        test_invalid_inputs();
        std::cout
            << "MFQ C++ DeepSeek-V4 sparse Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
