#include "mlx_deepseek_v4_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace {

using mfq::metal::DeepseekV4Config;
using mfq::metal::MlxDeepseekV4Attention;
using mfq::metal::MlxDeepseekV4AttentionComponents;
using mfq::metal::MlxDeepseekV4LayerState;
using mfq::metal::MlxDeepseekV4PoolState;
using mfq::metal::MlxLinear;
using mlx::core::Shape;
using mlx::core::array;

void require(
    bool condition,
    const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_invalid(
    Function&& function,
    const std::string& message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

array float_array(
    const std::vector<float>& values,
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
        if (!std::isfinite(actual[index]) ||
            std::fabs(actual[index] - expected[index]) >
                tolerance) {
            throw std::runtime_error(
                label + " mismatch at " +
                std::to_string(index) +
                ": actual=" +
                std::to_string(actual[index]) +
                " expected=" +
                std::to_string(expected[index]));
        }
    }
}

array token_slice(
    const array& input,
    int begin,
    int end) {
    return mlx::core::slice(
        input,
        Shape{0, begin, 0},
        Shape{
            input.shape(0),
            end,
            input.shape(2),
        });
}

DeepseekV4Config test_config() {
    DeepseekV4Config config;
    config.n_layers = 3;
    config.hidden = 4;
    config.n_experts = 1;
    config.top_k = 1;
    config.moe_inter = 4;
    config.n_shared = 1;
    config.n_heads = 64;
    config.head_dim = 512;
    config.q_lora_rank = 1;
    config.o_lora_rank = 1;
    config.o_groups = 64;
    config.kv_dim = 512;
    config.qk_rope_head_dim = 64;
    config.n_kv_heads = 1;
    config.vocab = 8;
    config.rms_eps = 1e-6;
    config.scoring_func = "sqrtsoftplus";
    config.norm_topk_prob = true;
    config.routed_scaling = 1.0;
    config.swiglu_limit = 0.0;
    config.n_hash_layers = 0;
    config.sliding_window = 4;
    config.rope_theta = 10'000.0;
    config.index_n_heads = 64;
    config.index_head_dim = 128;
    config.index_topk = 512;
    config.max_position_embeddings = 256;
    config.hc_mult = 4;
    config.hc_eps = 1e-6;
    config.hc_sinkhorn_iters = 20;
    config.compress_rope_theta = 160'000.0;
    config.compress_ratios = {0, 4, 128};
    config.validate();
    return config;
}

array dense_weight(
    int rows,
    int columns,
    const std::vector<std::pair<int, float>>& entries = {}) {
    std::vector<float> values(
        static_cast<std::size_t>(rows) * columns,
        0.0f);
    for (const auto& [index, value] : entries) {
        values.at(index) = value;
    }
    return float_array(
        values,
        Shape{rows, columns});
}

MlxDeepseekV4AttentionComponents components(
    const DeepseekV4Config& config,
    int ratio) {
    const int hidden =
        static_cast<int>(config.hidden);
    const int heads =
        static_cast<int>(config.n_heads);
    const int head_dim =
        static_cast<int>(config.head_dim);
    const int attention = heads * head_dim;
    const int groups =
        static_cast<int>(config.o_groups);
    const int rank =
        static_cast<int>(config.o_lora_rank);
    const int group_width = attention / groups;

    auto kv_weight = dense_weight(
        head_dim,
        hidden,
        {{0, 1.0f}});
    std::vector<std::pair<int, float>> wo_a_entries;
    for (int group = 0; group < groups; ++group) {
        wo_a_entries.emplace_back(
            group * group_width,
            1.0f);
    }
    MlxDeepseekV4AttentionComponents result{
        MlxLinear(dense_weight(1, hidden)),
        MlxLinear(std::move(kv_weight)),
        MlxLinear(dense_weight(attention, 1)),
        MlxLinear(
            dense_weight(
                groups * rank,
                group_width,
                wo_a_entries)),
        MlxLinear(
            dense_weight(
                hidden,
                groups * rank,
                {{0, 1.0f}})),
        mlx::core::ones(
            Shape{1},
            mlx::core::float32),
        mlx::core::ones(
            Shape{head_dim},
            mlx::core::float32),
        mlx::core::zeros(
            Shape{heads},
            mlx::core::float32),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
    };
    if (ratio != 0) {
        const int width =
            head_dim * (ratio == 4 ? 2 : 1);
        std::vector<std::pair<int, float>> entries;
        entries.reserve(width);
        for (int row = 0; row < width; ++row) {
            entries.emplace_back(
                row * hidden,
                0.5f);
        }
        result.main_kv.emplace(
            dense_weight(
                width,
                hidden,
                entries));
        result.main_gate.emplace(
            dense_weight(width, hidden));
        result.main_ape = mlx::core::zeros(
            Shape{ratio, width},
            mlx::core::float32);
        result.main_norm = mlx::core::ones(
            Shape{head_dim},
            mlx::core::float32);
    }
    if (ratio == 4) {
        const int index_heads =
            static_cast<int>(config.index_n_heads);
        const int index_dim =
            static_cast<int>(config.index_head_dim);
        const int index_width =
            index_heads * index_dim;
        const int compressor_width = 2 * index_dim;
        std::vector<std::pair<int, float>> entries;
        entries.reserve(compressor_width);
        for (int row = 0;
             row < compressor_width;
             ++row) {
            entries.emplace_back(
                row * hidden,
                0.5f);
        }
        result.index_q_b.emplace(
            dense_weight(index_width, 1));
        result.index_kv.emplace(
            dense_weight(
                compressor_width,
                hidden,
                entries));
        result.index_gate.emplace(
            dense_weight(
                compressor_width,
                hidden));
        result.index_weights.emplace(
            dense_weight(index_heads, hidden));
        result.index_ape = mlx::core::zeros(
            Shape{4, compressor_width},
            mlx::core::float32);
        result.index_norm = mlx::core::ones(
            Shape{index_dim},
            mlx::core::float32);
    }
    return result;
}

MlxDeepseekV4Attention attention(
    const DeepseekV4Config& config,
    int layer,
    int ratio,
    int max_context) {
    auto base =
        mfq::metal::deepseek_v4_yarn_tables(
            static_cast<int>(
                config.qk_rope_head_dim),
            max_context,
            static_cast<float>(config.rope_theta));
    auto compressed =
        mfq::metal::deepseek_v4_yarn_tables(
            static_cast<int>(
                config.qk_rope_head_dim),
            max_context,
            static_cast<float>(
                config.compress_rope_theta));
    return MlxDeepseekV4Attention(
        config,
        layer,
        ratio,
        max_context,
        components(config, ratio),
        std::move(base),
        std::move(compressed));
}

array input_tokens(
    const std::vector<float>& first_feature);

void test_attention_copy_move_lifetime_stress() {
    const auto config = test_config();
    constexpr int iterations = 128;
    for (int iteration = 0;
         iteration < iterations;
         ++iteration) {
        const int schedule = iteration % 3;
        const int layer =
            schedule == 0 ? 0 : schedule == 1 ? 1 : 2;
        const int ratio =
            schedule == 0 ? 0 : schedule == 1 ? 4 : 128;
        const int max_context =
            ratio == 0 ? 4 : ratio == 4 ? 8 : 128;

        auto operation = attention(
            config,
            layer,
            ratio,
            max_context);

        // Reallocation moves the public wrapper while all ProjectionGroup
        // pointers must continue to target the address-stable shared Impl.
        std::vector<MlxDeepseekV4Attention> owners;
        owners.push_back(operation);
        owners.push_back(std::move(operation));
        auto survivor = std::move(owners.back());
        owners.clear();
        require(
            survivor.layer() == layer &&
                survivor.ratio() == ratio &&
                survivor.max_context() == max_context,
            "attention copy/move lifetime metadata mismatch");

        // Periodically execute after every other wrapper has been destroyed.
        // The four sampled iterations cover ratio 0, 4, and 128.
        if (iteration % 32 == 0) {
            auto state =
                MlxDeepseekV4LayerState::allocate(
                    config,
                    ratio,
                    1,
                    max_context);
            auto output = survivor(
                input_tokens({0.25f}),
                state,
                0);
            output.eval();
            require(
                output.shape() == Shape{1, 1, 4},
                "attention copy/move lifetime output mismatch");
        }
    }
}

array input_tokens(
    const std::vector<float>& first_feature) {
    std::vector<float> values(
        first_feature.size() * 4,
        0.0f);
    for (std::size_t token = 0;
         token < first_feature.size();
         ++token) {
        values[token * 4] = first_feature[token];
    }
    return float_array(
        values,
        Shape{
            1,
            static_cast<int>(first_feature.size()),
            4,
        });
}

float normalized_kv(
    float value,
    float eps) {
    const float normalized = value /
        std::sqrt(
            value * value / 512.0f + eps);
    const float raw_scale =
        std::max(std::fabs(normalized), 1.0e-4f) /
        448.0f;
    const float scale = std::exp2(
        std::ceil(std::log2(raw_scale)));
    const float magnitude = std::fabs(normalized / scale);
    float quantized = 0.0f;
    if (magnitude < std::exp2(-6.0f)) {
        quantized = std::rint(magnitude * 512.0f) / 512.0f;
    } else {
        const float exponent = std::floor(std::log2(magnitude));
        const float step = std::exp2(exponent - 3.0f);
        quantized = std::min(
            std::rint(magnitude / step) * step,
            448.0f);
    }
    return std::copysign(quantized * scale, normalized);
}

std::vector<float> local_attention_reference(
    const std::vector<float>& inputs,
    int window,
    float eps) {
    std::vector<float> result(inputs.size() * 4, 0.0f);
    std::vector<float> normalized(inputs.size());
    for (std::size_t token = 0;
         token < inputs.size();
         ++token) {
        normalized[token] = normalized_kv(
            inputs[token],
            eps);
    }
    for (std::size_t token = 0;
         token < inputs.size();
         ++token) {
        const std::size_t begin =
            token + 1 >
                    static_cast<std::size_t>(window)
            ? token + 1 - window
            : 0;
        float sum = 0.0f;
        for (std::size_t item = begin;
             item <= token;
             ++item) {
            sum += normalized[item];
        }
        result[token * 4] =
            sum /
            static_cast<float>(token - begin + 2);
    }
    return result;
}

void test_yarn_rope_and_rms() {
    auto tables =
        mfq::metal::deepseek_v4_yarn_tables(
            4,
            3,
            100.0f);
    const auto cosine = evaluated_float(tables.first);
    const auto sine = evaluated_float(tables.second);
    const std::vector<float> frequencies{1.0f, 0.1f};
    for (int position = 0; position < 3; ++position) {
        for (int pair = 0; pair < 2; ++pair) {
            const float angle =
                position * frequencies[pair];
            require(
                std::fabs(
                    cosine[position * 2 + pair] -
                    std::cos(angle)) < 1e-6f,
                "Yarn cosine mismatch");
            require(
                std::fabs(
                    sine[position * 2 + pair] -
                    std::sin(angle)) < 1e-6f,
                "Yarn sine mismatch");
        }
    }

    const auto value = float_array(
        {1.0f, 2.0f, 3.0f, 4.0f},
        Shape{1, 4});
    const float angle0 = 0.3f;
    const float angle1 = -0.2f;
    const auto cos_pair = float_array(
        {std::cos(angle0), std::cos(angle1)},
        Shape{1, 2});
    const auto sin_pair = float_array(
        {std::sin(angle0), std::sin(angle1)},
        Shape{1, 2});
    const auto rotated =
        mfq::metal::deepseek_v4_rope_adjacent(
            value,
            cos_pair,
            sin_pair);
    const auto restored =
        mfq::metal::deepseek_v4_rope_adjacent(
            rotated,
            cos_pair,
            sin_pair,
            true);
    require_close(
        evaluated_float(restored),
        {1.0f, 2.0f, 3.0f, 4.0f},
        2e-6f,
        "adjacent inverse RoPE");

    const auto rms =
        mfq::metal::deepseek_v4_unweighted_rms(
            float_array(
                {3.0f, 4.0f},
                Shape{1, 2}),
            1e-6f);
    const float inverse =
        1.0f /
        std::sqrt(12.5f + 1e-6f);
    require_close(
        evaluated_float(rms),
        {3.0f * inverse, 4.0f * inverse},
        1e-6f,
        "unweighted RMS");

    auto scaled = test_config().rope_scaling;
    scaled.enabled = true;
    scaled.factor = 2.0;
    scaled.beta_fast = 32.0;
    scaled.beta_slow = 1.0;
    scaled.original_max_position_embeddings = 64;
    auto scaled_tables =
        mfq::metal::deepseek_v4_yarn_tables(
            64,
            4,
            10'000.0f,
            scaled);
    require(
        scaled_tables.first.shape() ==
            Shape{4, 32},
        "scaled Yarn table shape mismatch");
}

void test_pool_and_ratio_schedule() {
    const auto config = test_config();
    auto zero = MlxDeepseekV4LayerState::allocate(
        config,
        0,
        1,
        8);
    auto four = MlxDeepseekV4LayerState::allocate(
        config,
        4,
        1,
        8);
    auto one_twenty_eight =
        MlxDeepseekV4LayerState::allocate(
            config,
            128,
            1,
            128);
    require(
        !zero.main().has_value() &&
            !zero.indexer().has_value(),
        "ratio-zero state allocated compressors");
    require(
        four.main().has_value() &&
            four.indexer().has_value() &&
            four.main()->overlap() &&
            four.indexer()->overlap(),
        "ratio-four state schedule mismatch");
    require(
        one_twenty_eight.main().has_value() &&
            !one_twenty_eight.indexer().has_value() &&
            !one_twenty_eight.main()->overlap(),
        "ratio-128 state schedule mismatch");

    auto pool = MlxDeepseekV4PoolState::allocate(
        4,
        128,
        true,
        1,
        8);
    auto tables =
        mfq::metal::deepseek_v4_yarn_tables(
            64,
            8,
            160'000.0f);
    const auto token = mlx::core::astype(
        float_array(
            std::vector<float>(256, 0.5f),
            Shape{1, 1, 256}),
        mlx::core::float16);
    const auto gate = mlx::core::zeros(
        Shape{1, 1, 256},
        mlx::core::float16);
    const auto ape = mlx::core::zeros(
        Shape{4, 256},
        mlx::core::float32);
    const auto norm = mlx::core::ones(
        Shape{128},
        mlx::core::float32);
    const auto pool_cosine = mlx::core::contiguous(
        mlx::core::slice(
            tables.first,
            Shape{0, 0},
            tables.first.shape(),
            Shape{4, 1}));
    const auto pool_sine = mlx::core::contiguous(
        mlx::core::slice(
            tables.second,
            Shape{0, 0},
            tables.second.shape(),
            Shape{4, 1}));
    for (int length = 1; length <= 4; ++length) {
        pool.update(
            token,
            gate,
            ape,
            norm,
            length,
            pool_cosine,
            pool_sine,
            0,
            1e-6f);
    }
    require(
        pool.pool_len() == 1 &&
            pool.remainder() == 0 &&
            pool.prev_kv().has_value(),
        "pool boundary state mismatch");
    require_invalid(
        [&] {
            pool.update(
                token,
                gate,
                ape,
                norm,
                6,
                pool_cosine,
                pool_sine,
                0,
                1e-6f);
        },
        "non-contiguous pool update was accepted");
}

void test_local_attention_cpu_reference() {
    const auto config = test_config();
    auto operation = attention(
        config,
        0,
        0,
        40);
    auto state = MlxDeepseekV4LayerState::allocate(
        config,
        0,
        1,
        40);
    const std::vector<float> values{
        0.25f,
        -0.5f,
    };
    auto first = operation(
        input_tokens(values),
        state,
        0);
    auto decoded = operation(
        input_tokens({0.75f}),
        state,
        2);
    auto expected =
        local_attention_reference(
            {0.25f, -0.5f, 0.75f},
            4,
            static_cast<float>(config.rms_eps));
    auto actual = evaluated_float(
        mlx::core::concatenate(
            {first, decoded},
            1));
    require_close(
        actual,
        expected,
        3e-2f,
        "prefill/decode CPU reference");
    require(
        state.position() == 3,
        "decode cache position did not advance");
    const auto positions =
        evaluated_int(state.local_positions());
    require(
        positions[0] == 0 &&
            positions[1] == 1 &&
            positions[2] == 2,
        "local cache positions mismatch");

    require_invalid(
        [&] {
            (void)operation(
                input_tokens({1.0f}),
                state,
                2);
        },
        "non-contiguous attention position was accepted");
}

void test_prefill_mma_cpu_reference() {
    const auto config = test_config();
    auto operation = attention(
        config,
        0,
        0,
        32);
    auto state = MlxDeepseekV4LayerState::allocate(
        config,
        0,
        1,
        32);
    std::vector<float> values(32);
    for (int token = 0; token < 32; ++token) {
        values[token] =
            (token & 1) == 0
            ? 0.25f + token * 0.01f
            : -0.25f - token * 0.01f;
    }
    auto actual = operation(
        input_tokens(values),
        state,
        0);
    const auto expected =
        local_attention_reference(
            values,
            4,
            static_cast<float>(config.rms_eps));
    require_close(
        evaluated_float(actual),
        expected,
        4e-2f,
        "prefill-MMA CPU reference");
}

void test_ratio_four_continuity() {
    const auto config = test_config();
    auto chunked = attention(
        config,
        1,
        4,
        8);
    auto one_shot = attention(
        config,
        1,
        4,
        8);
    auto chunked_state =
        MlxDeepseekV4LayerState::allocate(
            config,
            4,
            1,
            8);
    auto one_shot_state =
        MlxDeepseekV4LayerState::allocate(
            config,
            4,
            1,
            8);
    const auto all = input_tokens(
        {0.25f, -0.5f, 0.75f, -1.0f});
    auto first = chunked(
        token_slice(all, 0, 2),
        chunked_state,
        0);
    auto third = chunked(
        token_slice(all, 2, 3),
        chunked_state,
        2);
    auto fourth = chunked(
        token_slice(all, 3, 4),
        chunked_state,
        3);
    auto expected = one_shot(
        all,
        one_shot_state,
        0);
    require_close(
        evaluated_float(
            mlx::core::concatenate(
                {first, third, fourth},
                1)),
        evaluated_float(expected),
        4e-2f,
        "ratio-four prefill/decode continuity");
    require(
        chunked_state.main()->pool_len() == 1 &&
            chunked_state.indexer()->pool_len() == 1 &&
            one_shot_state.main()->pool_len() == 1 &&
            chunked_state.position() == 4,
        "ratio-four compressor continuity mismatch");
    require_close(
        evaluated_float(
            chunked_state.main()->pool()),
        evaluated_float(
            one_shot_state.main()->pool()),
        2e-3f,
        "ratio-four main pool continuity");
    require_close(
        evaluated_float(
            chunked_state.indexer()->pool()),
        evaluated_float(
            one_shot_state.indexer()->pool()),
        2e-3f,
        "ratio-four index pool continuity");
}

void test_ratio_128_forward() {
    const auto config = test_config();
    auto operation = attention(
        config,
        2,
        128,
        128);
    auto state =
        MlxDeepseekV4LayerState::allocate(
            config,
            128,
            1,
            128);
    const std::vector<float> values{
        0.25f,
        -0.5f,
    };
    auto output = operation(
        input_tokens(values),
        state,
        0);
    require_close(
        evaluated_float(output),
        local_attention_reference(
            values,
            4,
            static_cast<float>(config.rms_eps)),
        3e-2f,
        "ratio-128 forward");
    require(
        state.main().has_value() &&
            !state.indexer().has_value() &&
            state.main()->pool_len() == 0 &&
            state.main()->remainder() == 2 &&
            state.position() == 2,
        "ratio-128 forward state mismatch");
}

template <typename T>
void append_scalar(
    std::vector<std::uint8_t>& bytes,
    T value) {
    const auto* source =
        reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(
        bytes.end(),
        source,
        source + sizeof(value));
}

std::vector<std::uint8_t> dense_payload(
    const Shape& shape,
    const std::vector<float>& values) {
    std::size_t elements = 1;
    for (const int extent : shape) {
        elements *= static_cast<std::size_t>(extent);
    }
    require(
        values.size() == elements,
        "synthetic dense payload size mismatch");
    std::vector<std::uint8_t> result;
    append_scalar<std::uint32_t>(
        result,
        static_cast<std::uint32_t>(shape.size()));
    for (const int extent : shape) {
        append_scalar<std::int64_t>(result, extent);
    }
    const auto* first =
        reinterpret_cast<const std::uint8_t*>(
            values.data());
    result.insert(
        result.end(),
        first,
        first + values.size() * sizeof(float));
    return result;
}

void write_string(
    std::ostream& stream,
    std::string_view value) {
    const auto size =
        static_cast<std::uint32_t>(value.size());
    stream.write(
        reinterpret_cast<const char*>(&size),
        sizeof(size));
    stream.write(
        value.data(),
        static_cast<std::streamsize>(value.size()));
}

template <typename T>
void write_scalar(
    std::ostream& stream,
    T value) {
    stream.write(
        reinterpret_cast<const char*>(&value),
        sizeof(value));
}

struct Record {
    std::string name;
    std::vector<std::uint8_t> payload;
};

void write_attention_container(
    const std::filesystem::path& path,
    const DeepseekV4Config& config) {
    const int hidden =
        static_cast<int>(config.hidden);
    const int head_dim =
        static_cast<int>(config.head_dim);
    const int attention =
        static_cast<int>(
            config.n_heads * config.head_dim);
    const int groups =
        static_cast<int>(config.o_groups);
    const int group_width = attention / groups;
    const auto name =
        [](std::string_view suffix) {
            return std::string("layers.0.") +
                std::string(suffix);
        };
    std::vector<Record> records{
        {
            name("attn.wq_a.weight"),
            dense_payload(
                Shape{1, hidden},
                std::vector<float>(hidden, 0.0f)),
        },
        {
            name("attn.wkv.weight"),
            dense_payload(
                Shape{head_dim, hidden},
                std::vector<float>(
                    head_dim * hidden,
                    0.0f)),
        },
        {
            name("attn.wq_b.weight"),
            dense_payload(
                Shape{attention, 1},
                std::vector<float>(
                    attention,
                    0.0f)),
        },
        {
            name("attn.wo_a.weight"),
            dense_payload(
                Shape{groups, group_width},
                std::vector<float>(
                    groups * group_width,
                    0.0f)),
        },
        {
            name("attn.wo_b.weight"),
            dense_payload(
                Shape{hidden, groups},
                std::vector<float>(
                    hidden * groups,
                    0.0f)),
        },
        {
            name("attn.q_norm.weight"),
            dense_payload(
                Shape{1},
                std::vector<float>(1, 1.0f)),
        },
        {
            name("attn.kv_norm.weight"),
            dense_payload(
                Shape{head_dim},
                std::vector<float>(
                    head_dim,
                    1.0f)),
        },
        {
            name("attn.attn_sink"),
            dense_payload(
                Shape{
                    static_cast<int>(
                        config.n_heads),
                },
                std::vector<float>(
                    config.n_heads,
                    0.0f)),
        },
    };
    std::ofstream stream(path, std::ios::binary);
    require(
        static_cast<bool>(stream),
        "cannot create synthetic attention container");
    stream.write("MFQ1", 4);
    write_scalar<std::uint32_t>(stream, 2);
    write_string(stream, "attention-test");
    write_scalar<std::uint32_t>(stream, 0);
    write_scalar<std::uint32_t>(
        stream,
        static_cast<std::uint32_t>(
            records.size()));
    for (const auto& record : records) {
        write_string(stream, record.name);
        write_string(stream, "F32");
        write_scalar<std::uint64_t>(
            stream,
            record.payload.size());
    }
    for (const auto& record : records) {
        stream.write(
            reinterpret_cast<const char*>(
                record.payload.data()),
            static_cast<std::streamsize>(
                record.payload.size()));
    }
}

void test_container_load_and_rejections() {
    const auto config = test_config();
    const auto path =
        std::filesystem::temp_directory_path() /
        "mfq-dsv4-attention-test.mfq";
    write_attention_container(path, config);
    try {
        const mfq::metal::MfqContainer model(path);
        auto loaded = MlxDeepseekV4Attention::load(
            model,
            config,
            0,
            0,
            4);
        auto state =
            MlxDeepseekV4LayerState::allocate(
                config,
                0,
                1,
                4);
        require_close(
            evaluated_float(
                loaded(
                    input_tokens({0.0f}),
                    state,
                    0)),
            std::vector<float>(4, 0.0f),
            1e-6f,
            "container-loaded attention");
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        throw;
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);

    require_invalid(
        [&] {
            auto operation = attention(
                config,
                0,
                0,
                4);
            auto state =
                MlxDeepseekV4LayerState::allocate(
                    config,
                    0,
                    1,
                    4);
            (void)operation(
                mlx::core::zeros(
                    Shape{1, 1, 3},
                    mlx::core::float16),
                state,
                0);
        },
        "wrong attention hidden width was accepted");
    require_invalid(
        [&] {
            (void)attention(
                config,
                0,
                4,
                4);
        },
        "wrong layer compression schedule was accepted");
    require_invalid(
        [&] {
            auto operation = attention(
                config,
                0,
                0,
                4);
            auto wrong_state =
                MlxDeepseekV4LayerState::allocate(
                    config,
                    4,
                    1,
                    4);
            (void)operation(
                input_tokens({0.0f}),
                wrong_state,
                0);
        },
        "wrong attention cache schedule was accepted");
}

} // namespace

int main() {
    try {
        test_yarn_rope_and_rms();
        test_pool_and_ratio_schedule();
        test_local_attention_cpu_reference();
        test_prefill_mma_cpu_reference();
        test_ratio_four_continuity();
        test_ratio_128_forward();
        test_attention_copy_move_lifetime_stress();
        test_container_load_and_rejections();
        std::cout
            << "MFQ C++ DeepSeek-V4 attention/cache "
               "Metal tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
