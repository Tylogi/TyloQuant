#include "mlx_qwen35_causal_lm.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace {

using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MlxEmbedding;
using mfq::metal::MlxLinear;
using mfq::metal::MlxQwen35CausalLm;
using mfq::metal::MlxQwen35DenseSwiGlu;
using mfq::metal::MlxQwen35FullAttentionBlock;
using mfq::metal::MlxQwen35Layer;
using mfq::metal::MlxQwen35LinearAttentionBlock;
using mfq::metal::MlxQwen35MtpModule;
using mfq::metal::MlxRmsNorm;
using mfq::metal::Qwen35Config;
using mfq::metal::Qwen35Embedding;
using mfq::metal::Qwen35Linear;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> patterned(
    int size,
    float scale,
    int period,
    int center) {
    std::vector<float> result;
    result.reserve(static_cast<std::size_t>(size));
    for (int index = 0; index < size; ++index) {
        result.push_back(
            static_cast<float>(index % period - center) * scale);
    }
    return result;
}

array matrix(
    int rows,
    int columns,
    const std::vector<float>& values) {
    require(
        static_cast<int>(values.size()) == rows * columns,
        "invalid synthetic matrix data");
    return array(values.begin(), Shape{rows, columns});
}

array constant_vector(int size, float value) {
    const std::vector<float> values(
        static_cast<std::size_t>(size),
        value);
    return array(values.begin(), Shape{size});
}

Qwen35Config test_config(bool tied = false) {
    Qwen35Config config;
    config.model_type = "qwen3_5";
    config.text_model_type = "qwen3_5_text";
    config.vocab_size = 8;
    config.hidden_size = 4;
    config.intermediate_size = 5;
    config.num_hidden_layers = 2;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;
    config.max_position_embeddings = 32;
    config.head_dim = 4;
    config.rope_base = 10'000.0;
    config.rotary_dim = 4;
    config.rope_sections = {1, 1, 0};
    config.mrope_interleaved = true;
    config.rms_norm_eps = 1e-6;
    config.norm_weight_offset = 0.0;
    config.tie_word_embeddings = tied;
    config.attention_output_gate = false;
    config.layer_types = {
        "linear_attention",
        "full_attention",
    };
    config.linear_conv_kernel_dim = 4;
    config.linear_key_head_dim = 32;
    config.linear_value_head_dim = 32;
    config.linear_num_key_heads = 1;
    config.linear_num_value_heads = 2;
    config.linear_a_is_log = true;
    return config;
}

MlxRmsNorm hidden_norm(const Qwen35Config& config) {
    return MlxRmsNorm(
        constant_vector(
            static_cast<int>(config.hidden_size),
            1.0f),
        static_cast<float>(config.rms_norm_eps),
        static_cast<float>(config.norm_weight_offset));
}

MlxQwen35DenseSwiGlu make_ffn(
    const Qwen35Config& config) {
    const int hidden = static_cast<int>(config.hidden_size);
    const int intermediate =
        static_cast<int>(config.intermediate_size);
    return MlxQwen35DenseSwiGlu(
        MlxLinear(matrix(
            intermediate,
            hidden,
            patterned(
                intermediate * hidden,
                0.021f,
                9,
                4))),
        MlxLinear(matrix(
            intermediate,
            hidden,
            patterned(
                intermediate * hidden,
                0.018f,
                7,
                3))),
        MlxLinear(matrix(
            hidden,
            intermediate,
            patterned(
                hidden * intermediate,
                0.016f,
                11,
                5))));
}

MlxQwen35FullAttentionBlock make_full_block(
    const Qwen35Config& config) {
    const int hidden = static_cast<int>(config.hidden_size);
    const int attention =
        static_cast<int>(config.attention_size());
    const int kv = static_cast<int>(config.kv_size());
    return MlxQwen35FullAttentionBlock(
        config,
        hidden_norm(config),
        MlxLinear(matrix(
            attention,
            hidden,
            patterned(
                attention * hidden,
                0.035f,
                9,
                4))),
        MlxLinear(matrix(
            kv,
            hidden,
            patterned(kv * hidden, 0.055f, 7, 3))),
        MlxLinear(matrix(
            kv,
            hidden,
            patterned(kv * hidden, 0.075f, 5, 2))),
        MlxLinear(matrix(
            hidden,
            attention,
            patterned(
                hidden * attention,
                0.045f,
                7,
                3))),
        std::nullopt,
        std::nullopt,
        hidden_norm(config),
        make_ffn(config));
}

MlxQwen35LinearAttentionBlock make_linear_block(
    const Qwen35Config& config) {
    const int hidden = static_cast<int>(config.hidden_size);
    const int value_size =
        static_cast<int>(config.linear_value_size());
    const int channels =
        static_cast<int>(config.linear_qkv_size());
    const int value_heads =
        static_cast<int>(config.linear_value_heads());
    const int dimension =
        static_cast<int>(config.linear_value_head_dim);
    const int kernel =
        static_cast<int>(config.linear_conv_kernel_dim);

    std::vector<float> convolution(
        static_cast<std::size_t>(channels * kernel),
        0.0f);
    for (int channel = 0; channel < channels; ++channel) {
        convolution[
            static_cast<std::size_t>(channel * kernel)] = 0.04f;
        convolution[
            static_cast<std::size_t>(channel * kernel + 1)] = -0.08f;
        convolution[
            static_cast<std::size_t>(channel * kernel + 2)] = 0.16f;
        convolution[
            static_cast<std::size_t>(channel * kernel + 3)] = 0.88f;
    }

    std::vector<float> a_values(
        static_cast<std::size_t>(value_heads));
    std::vector<float> dt_values(
        static_cast<std::size_t>(value_heads));
    for (int head = 0; head < value_heads; ++head) {
        a_values[static_cast<std::size_t>(head)] =
            std::log(0.65f + 0.1f * static_cast<float>(head));
        dt_values[static_cast<std::size_t>(head)] =
            -0.3f + 0.5f * static_cast<float>(head);
    }

    return MlxQwen35LinearAttentionBlock(
        config,
        true,
        hidden_norm(config),
        MlxLinear(matrix(
            channels,
            hidden,
            patterned(
                channels * hidden,
                0.008f,
                13,
                6))),
        std::nullopt,
        std::nullopt,
        MlxLinear(matrix(
            value_size,
            hidden,
            patterned(
                value_size * hidden,
                0.006f,
                17,
                8))),
        MlxLinear(matrix(
            value_heads,
            hidden,
            patterned(
                value_heads * hidden,
                0.025f,
                7,
                3))),
        MlxLinear(matrix(
            value_heads,
            hidden,
            patterned(
                value_heads * hidden,
                -0.02f,
                7,
                3))),
        matrix(channels, kernel, convolution),
        std::nullopt,
        array(
            dt_values.begin(),
            Shape{value_heads}),
        array(
            a_values.begin(),
            Shape{value_heads}),
        MlxRmsNorm(
            constant_vector(dimension, 1.0f),
            static_cast<float>(config.rms_norm_eps)),
        MlxLinear(matrix(
            hidden,
            value_size,
            patterned(
                hidden * value_size,
                0.004f,
                19,
                9))),
        hidden_norm(config),
        make_ffn(config));
}

Qwen35Embedding make_embedding(
    const Qwen35Config& config) {
    const int vocab = static_cast<int>(config.vocab_size);
    const int hidden = static_cast<int>(config.hidden_size);
    return Qwen35Embedding(
        MlxEmbedding(matrix(
            vocab,
            hidden,
            patterned(
                vocab * hidden,
                0.06f,
                13,
                6))));
}

Qwen35Linear make_output(
    const Qwen35Config& config,
    bool zero) {
    const int vocab = static_cast<int>(config.vocab_size);
    const int hidden = static_cast<int>(config.hidden_size);
    const auto values = zero
        ? std::vector<float>(
              static_cast<std::size_t>(vocab * hidden),
              0.0f)
        : patterned(vocab * hidden, 0.08f, 11, 5);
    return Qwen35Linear(
        MlxLinear(matrix(vocab, hidden, values)));
}

std::vector<MlxQwen35Layer> make_layers(
    const Qwen35Config& config) {
    std::vector<MlxQwen35Layer> layers;
    layers.reserve(2);
    layers.emplace_back(make_linear_block(config));
    layers.emplace_back(make_full_block(config));
    return layers;
}

MlxQwen35CausalLm make_model(
    bool zero_output = false,
    bool tied = false,
    bool with_mtp = false) {
    auto config = test_config(tied);
    config.mtp_num_hidden_layers = with_mtp ? 1 : 0;
    std::optional<Qwen35Linear> output;
    if (!tied) {
        output.emplace(make_output(config, zero_output));
    }
    std::optional<MlxQwen35MtpModule> mtp;
    if (with_mtp) {
        const int hidden = static_cast<int>(config.hidden_size);
        std::vector<MlxQwen35FullAttentionBlock> mtp_layers;
        mtp_layers.emplace_back(make_full_block(config));
        mtp.emplace(
            config,
            hidden_norm(config),
            hidden_norm(config),
            Qwen35Linear(MlxLinear(matrix(
                hidden,
                hidden * 2,
                patterned(hidden * hidden * 2, 0.017f, 13, 6)))),
            std::move(mtp_layers),
            hidden_norm(config));
    }
    return MlxQwen35CausalLm(
        config,
        make_embedding(config),
        make_layers(config),
        hidden_norm(config),
        std::move(output),
        mlx::core::float32,
        std::move(mtp));
}

array token_ids(std::initializer_list<int> values) {
    return array(
        values,
        Shape{1, static_cast<int>(values.size())},
        mlx::core::int32);
}

array three_axis_positions(
    std::initializer_list<int> temporal,
    std::initializer_list<int> height,
    std::initializer_list<int> width) {
    require(
        temporal.size() == height.size() &&
            temporal.size() == width.size() &&
            temporal.size() != 0,
        "invalid synthetic three-axis positions");
    std::vector<std::int32_t> values;
    values.reserve(temporal.size() * 3);
    for (const auto value : temporal) {
        values.push_back(value);
    }
    for (const auto value : height) {
        values.push_back(value);
    }
    for (const auto value : width) {
        values.push_back(value);
    }
    return array(
        values.begin(),
        Shape{3, static_cast<int>(temporal.size())},
        mlx::core::int32);
}

array text_positions(int start, int count) {
    require(start >= 0 && count > 0, "invalid synthetic text positions");
    std::vector<std::int32_t> values;
    values.reserve(static_cast<std::size_t>(count) * 3);
    for (int axis = 0; axis < 3; ++axis) {
        for (int token = 0; token < count; ++token) {
            values.push_back(start + token);
        }
    }
    return array(
        values.begin(),
        Shape{3, count},
        mlx::core::int32);
}

std::vector<float> evaluated_floats(array value) {
    value = mlx::core::astype(value, mlx::core::float32);
    value.eval();
    return std::vector<float>(
        value.data<float>(),
        value.data<float>() + value.size());
}

void require_array_close(
    array actual,
    array expected,
    float tolerance,
    const std::string& name) {
    require(actual.shape() == expected.shape(), name + " shape mismatch");
    const auto actual_values = evaluated_floats(std::move(actual));
    const auto expected_values = evaluated_floats(std::move(expected));
    for (std::size_t index = 0;
         index < actual_values.size();
         ++index) {
        if (!std::isfinite(actual_values[index]) ||
            std::fabs(
                actual_values[index] -
                expected_values[index]) > tolerance) {
            throw std::runtime_error(
                name + " mismatch at " + std::to_string(index) +
                ": actual=" +
                std::to_string(actual_values[index]) +
                " expected=" +
                std::to_string(expected_values[index]));
        }
    }
}

array last_logits(const array& logits) {
    return mlx::core::slice(
        logits,
        Shape{0, logits.shape(1) - 1, 0},
        Shape{
            logits.shape(0),
            logits.shape(1),
            logits.shape(2),
    });
}

void test_explicit_mrope_semantics() {
    const auto ids = token_ids({1, 4, 2});

    auto offset_model = make_model();
    auto explicit_text_model = make_model();
    auto offset_logits = offset_model.forward(ids, false);
    auto explicit_text_logits = explicit_text_model.forward(
        ids,
        text_positions(0, 3),
        false);
    require_array_close(
        explicit_text_logits,
        offset_logits,
        3.0e-4f,
        "three-axis text positions versus contiguous offset");

    auto multimodal_model = make_model();
    auto multimodal_logits = multimodal_model.forward(
        ids,
        three_axis_positions(
            {0, 1, 2},
            {0, 13, 31},
            {0, 7, 29}),
        false);
    const auto text_values =
        evaluated_floats(std::move(explicit_text_logits));
    const auto multimodal_values =
        evaluated_floats(std::move(multimodal_logits));
    require(
        text_values.size() == multimodal_values.size(),
        "multimodal logits size mismatch");
    float maximum_difference = 0.0f;
    for (std::size_t index = 0;
         index < text_values.size();
         ++index) {
        require(
            std::isfinite(multimodal_values[index]),
            "multimodal logits contain a non-finite value");
        maximum_difference = std::max(
            maximum_difference,
            std::fabs(
                multimodal_values[index] -
                text_values[index]));
    }
    require(
        maximum_difference > 1.0e-7f,
        "non-identical MRoPE axes did not change model logits");
}

void test_explicit_mrope_cache_continuity() {
    const auto all_positions = three_axis_positions(
        {0, 1, 2},
        {0, 13, 31},
        {0, 7, 29});

    auto full_model = make_model();
    auto full_logits = full_model.forward(
        token_ids({2, 5, 3}),
        all_positions,
        true);

    auto decode_model = make_model();
    auto prefix = decode_model.forward(
        token_ids({2, 5}),
        three_axis_positions(
            {0, 1},
            {0, 13},
            {0, 7}),
        true);
    auto decoded = decode_model.forward(
        token_ids({3}),
        three_axis_positions(
            {2},
            {31},
            {29}),
        true);
    mlx::core::eval(full_logits, prefix, decoded);
    require(
        full_model.cache_position() == 3 &&
            decode_model.cache_position() == 3,
        "explicit MRoPE cache position did not advance by token count");
    require_array_close(
        std::move(decoded),
        last_logits(full_logits),
        3.0e-3f,
        "explicit MRoPE prefill/decode");

    auto legacy_decode =
        decode_model.forward(token_ids({4}), true);
    legacy_decode.eval();
    require(
        decode_model.cache_position() == 4,
        "legacy contiguous decode did not continue an explicit-position cache");
}

void test_explicit_position_validation() {
    const auto ids = token_ids({1, 4, 2});
    const auto rejected = [&](array positions) {
        auto model = make_model();
        try {
            (void)model.forward(ids, positions, false);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };

    const std::vector<std::int32_t> wrong_axis_values(6, 0);
    require(
        rejected(array(
            wrong_axis_values.begin(),
            Shape{2, 3},
            mlx::core::int32)),
        "two-axis positions were accepted");

    const std::vector<std::int32_t> wrong_length_values(6, 0);
    require(
        rejected(array(
            wrong_length_values.begin(),
            Shape{3, 2},
            mlx::core::int32)),
        "mismatched position length was accepted");

    const std::vector<float> floating_values(9, 0.0f);
    require(
        rejected(array(
            floating_values.begin(),
            Shape{3, 3},
            mlx::core::float32)),
        "floating-point positions were accepted");

    require(
        rejected(three_axis_positions(
            {0, 1, 2},
            {0, -1, 2},
            {0, 1, 2})),
        "negative explicit position was accepted");
    require(
        rejected(three_axis_positions(
            {0, 1, 2},
            {0, 32, 2},
            {0, 1, 2})),
        "out-of-context explicit position was accepted");
}

void test_layer_order_and_forward() {
    const auto config = test_config();
    auto model = make_model();
    require(model.layer_count() == 2, "mixed model layer count mismatch");
    require(
        model.layer_type(0) == "linear_attention" &&
            model.layer_type(1) == "full_attention",
        "mixed model layer order mismatch");

    const auto ids = token_ids({1, 4, 2});
    auto actual = model.forward(ids, false);

    auto embedding = make_embedding(config);
    auto linear = make_linear_block(config);
    auto full = make_full_block(config);
    auto norm = hidden_norm(config);
    auto output = make_output(config, false);
    auto hidden = embedding(ids, mlx::core::float32);
    hidden = linear.forward(hidden, 0, false);
    hidden = full.forward(hidden, 0, false);
    auto expected = output(norm(hidden));

    require_array_close(
        std::move(actual),
        std::move(expected),
        2.5e-3f,
        "mixed layer forward order");
    require(
        model.cache_position() == 0 &&
            model.cache_batch() == 0,
        "uncached forward mutated model cache");
}

void test_prefill_decode_and_reset() {
    auto full_model = make_model();
    auto full = full_model.forward(token_ids({2, 5, 3}), true);

    auto decode_model = make_model();
    auto prefix =
        decode_model.forward(token_ids({2, 5}), true);
    auto decoded =
        decode_model.forward(token_ids({3}), true);
    mlx::core::eval(full, prefix, decoded);

    require(
        full_model.cache_position() == 3 &&
            decode_model.cache_position() == 3,
        "mixed model prefill/decode cache position mismatch");
    require_array_close(
        std::move(decoded),
        last_logits(full),
        3.0e-3f,
        "mixed model prefill/decode");

    decode_model.reset_cache(1);
    require(
        decode_model.cache_position() == 0 &&
            decode_model.cache_batch() == 1,
        "mixed model reset_cache did not reset state");
    auto repeated =
        decode_model.forward(token_ids({2, 5}), true);
    require_array_close(
        std::move(repeated),
        std::move(prefix),
        3.0e-3f,
        "mixed model reset prefill");

    decode_model.clear_cache();
    require(
        decode_model.cache_position() == 0 &&
            decode_model.cache_batch() == 0,
        "mixed model clear_cache did not release state");
}

void test_mtp_greedy_identity() {
    mfq::metal::MlxSamplingParams sampling;
    sampling.temperature = 0.0;
    const std::vector<std::int64_t> prompt{1, 4, 2};

    auto baseline = make_model();
    auto speculative = make_model(false, false, true);
    require(speculative.supports_mtp(), "synthetic MTP head was not attached");

    std::vector<std::int64_t> expected;
    std::vector<std::int64_t> actual;
    const auto expected_count = baseline.generate(
        prompt,
        sampling,
        8,
        [&](std::int64_t token) {
            expected.push_back(token);
            return true;
        });
    const auto actual_count = speculative.generate(
        prompt,
        sampling,
        8,
        [&](std::int64_t token) {
            actual.push_back(token);
            return true;
        });
    require(
        expected_count == 8 && actual_count == 8 && actual == expected,
        "MTP greedy generation changed the target token sequence");

    auto repeated = make_model(false, false, true);
    std::vector<std::int64_t> stopped;
    const auto stopped_count = repeated.generate(
        prompt,
        sampling,
        8,
        [&](std::int64_t token) {
            stopped.push_back(token);
            return stopped.size() < 3;
        });
    require(
        stopped_count == 3 &&
            stopped == std::vector<std::int64_t>(expected.begin(), expected.begin() + 3),
        "MTP callback stop semantics changed");

    mfq::metal::MlxSamplingParams stochastic;
    stochastic.temperature = 0.8;
    stochastic.top_k = 4;
    stochastic.top_p = 0.9;
    stochastic.seed = 1729;
    auto sampled_baseline = make_model();
    auto sampled_fallback = make_model(false, false, true);
    std::vector<std::int64_t> expected_sampled;
    std::vector<std::int64_t> actual_sampled;
    sampled_baseline.generate(
        prompt,
        stochastic,
        8,
        [&](std::int64_t token) {
            expected_sampled.push_back(token);
            return true;
        });
    sampled_fallback.generate(
        prompt,
        stochastic,
        8,
        [&](std::int64_t token) {
            actual_sampled.push_back(token);
            return true;
        });
    require(
        actual_sampled == expected_sampled,
        "MTP stochastic fallback changed the target token sequence");
}

void test_text_session_snapshot_restore() {
    mfq::metal::MlxSamplingParams sampling;
    sampling.temperature = 0.0;

    auto cached = make_model();
    auto prefix_logits = cached.forward(token_ids({1, 2}), true);
    prefix_logits.eval();
    const auto snapshot =
        cached.capture_text_session_state({1, 2});
    require(
        snapshot.tokens == std::vector<std::int64_t>({1, 2}) &&
            snapshot.cache_position == 2 &&
            snapshot.cache_batch == 1 &&
            snapshot.layers.size() == cached.layer_count() &&
            snapshot.bytes > 0,
        "Qwen3.5 text session snapshot metadata mismatch");

    // Advance far enough to mutate both the recurrent linear-attention state
    // and the full-attention KV cache, then restore the saved prefix.
    auto discarded = cached.forward(token_ids({3, 4}), true);
    discarded.eval();
    cached.restore_text_session_state(snapshot);
    require(
        cached.cache_position() == 2 && cached.cache_batch() == 1,
        "Qwen3.5 text session restore position mismatch");

    std::size_t reused_prefill_tokens = 0;
    std::vector<std::int64_t> restored_output;
    (void)cached.generate(
        {1, 2, 5},
        sampling,
        3,
        [&](std::int64_t token) {
            restored_output.push_back(token);
            return true;
        },
        [&](std::size_t tokens, double) {
            reused_prefill_tokens = tokens;
        },
        {},
        2);
    require(
        reused_prefill_tokens == 1 && cached.cache_position() == 2,
        "Qwen3.5 restored session did not evaluate only the suffix");

    auto fresh = make_model();
    std::vector<std::int64_t> fresh_output;
    (void)fresh.generate(
        {1, 2, 5},
        sampling,
        3,
        [&](std::int64_t token) {
            fresh_output.push_back(token);
            return true;
        });
    require(
        restored_output == fresh_output,
        "Qwen3.5 restored session changed generated tokens");

    // Reuse the same saved object after a completed decode. Session snapshots
    // must own their arrays rather than aliasing the runtime cache.
    cached.restore_text_session_state(snapshot);
    std::vector<std::int64_t> repeated_output;
    (void)cached.generate(
        {1, 2, 5},
        sampling,
        3,
        [&](std::int64_t token) {
            repeated_output.push_back(token);
            return true;
        },
        {},
        {},
        2);
    require(
        repeated_output == fresh_output,
        "Qwen3.5 text session snapshot was mutated by resumed decode");
}

void test_tied_embedding_forward_cache_and_generate() {
    const auto config = test_config(true);
    const auto ids = token_ids({1, 4, 2});
    auto model = make_model(false, true);
    require(
        model.config().tie_word_embeddings,
        "tied model lost tie_word_embeddings config");
    auto actual = model.forward(ids, false);

    auto embedding = make_embedding(config);
    auto linear = make_linear_block(config);
    auto full = make_full_block(config);
    auto norm = hidden_norm(config);
    auto hidden = embedding(ids, mlx::core::float32);
    hidden = linear.forward(hidden, 0, false);
    hidden = full.forward(hidden, 0, false);
    auto expected = embedding.project(norm(hidden));
    require_array_close(
        std::move(actual),
        std::move(expected),
        2.5e-3f,
        "tied dense embedding projection");

    auto full_model = make_model(false, true);
    auto full_logits =
        full_model.forward(token_ids({2, 5, 3}), true);
    auto decode_model = make_model(false, true);
    auto prefix =
        decode_model.forward(token_ids({2, 5}), true);
    auto decoded =
        decode_model.forward(token_ids({3}), true);
    mlx::core::eval(full_logits, prefix, decoded);
    require_array_close(
        std::move(decoded),
        last_logits(full_logits),
        3.0e-3f,
        "tied embedding prefill/decode");
    require(
        full_model.cache_position() == 3 &&
            decode_model.cache_position() == 3,
        "tied embedding cache position mismatch");

    mfq::metal::MlxSamplingParams sampling;
    sampling.temperature = 0.0;
    std::vector<std::int64_t> generated;
    auto generate_model = make_model(false, true);
    const auto generated_count = generate_model.generate(
        {1, 2},
        sampling,
        3,
        [&](std::int64_t token) {
            generated.push_back(token);
            return true;
        });
    require(
        generated_count == 3 && generated.size() == 3,
        "tied embedding generation count mismatch");
    require(
        std::all_of(
            generated.begin(),
            generated.end(),
            [](std::int64_t token) {
                return token >= 0 && token < 8;
            }),
        "tied embedding generation produced invalid token");
    require(
        generate_model.cache_position() == 4,
        "tied embedding generation cache progression mismatch");

    std::vector<std::int64_t> repeated;
    auto repeat_model = make_model(false, true);
    const auto repeated_count = repeat_model.generate(
        {1, 2},
        sampling,
        3,
        [&](std::int64_t token) {
            repeated.push_back(token);
            return true;
        });
    require(
        repeated_count == generated_count &&
            repeated == generated,
        "tied embedding greedy generation was not deterministic");
}

int uniform_token(float uniform, int vocab) {
    uniform = std::clamp(uniform, 0.0f, 0.99999994f);
    const float target =
        uniform * static_cast<float>(vocab);
    for (int token = 0; token < vocab; ++token) {
        if (static_cast<float>(token + 1) >= target) {
            return token;
        }
    }
    return vocab - 1;
}

void test_generation_greedy_seed_and_penalties() {
    {
        const auto prompt = token_ids({0, 1, 1});
        mfq::metal::MlxSamplingParams sampling;
        auto counts =
            mfq::metal::detail::
                qwen35_generation_token_counts(
                    sampling,
                    prompt,
                    8);
        require(
            !counts.has_value(),
            "default generation unexpectedly built token counts");

        sampling.presence_penalty = 0.2;
        counts =
            mfq::metal::detail::
                qwen35_generation_token_counts(
                    sampling,
                    prompt,
                    8);
        require(
            counts.has_value(),
            "presence penalty did not build token counts");
        require_array_close(
            mlx::core::astype(
                *counts,
                mlx::core::float32),
            array(
                {
                    1.0f,
                    2.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                }),
            0.0f,
            "prompt token counts");

        sampling = {};
        sampling.frequency_penalty = 0.1;
        require(
            mfq::metal::detail::
                qwen35_generation_token_counts(
                    sampling,
                    prompt,
                    8)
                .has_value(),
            "frequency penalty did not build token counts");
    }

    {
        auto model = make_model(true);
        mfq::metal::MlxSamplingParams sampling;
        sampling.temperature = 0.0;
        std::vector<std::int64_t> generated;
        int prefill_calls = 0;
        std::size_t prefill_tokens = 0;
        double prefill_ms = -1.0;
        const auto count = model.generate(
            {1, 2},
            sampling,
            3,
            [&](std::int64_t token) {
                generated.push_back(token);
                return true;
            },
            [&](std::size_t tokens, double elapsed_ms) {
                ++prefill_calls;
                prefill_tokens = tokens;
                prefill_ms = elapsed_ms;
            });
        require(
            count == 3 &&
                generated ==
                    std::vector<std::int64_t>({0, 0, 0}),
            "greedy generation mismatch");
        require(
            prefill_calls == 1 &&
                prefill_tokens == 2 &&
                prefill_ms >= 0.0,
            "Qwen3.5 prefill callback mismatch");
        require(
            model.cache_position() == 4,
            "generation did not use one prefill plus token decode");

        generated.clear();
        const auto repeated_count = model.generate(
            {1, 2},
            sampling,
            3,
            [&](std::int64_t token) {
                generated.push_back(token);
                return true;
            });
        require(
            repeated_count == 3 &&
                generated ==
                    std::vector<std::int64_t>({0, 0, 0}) &&
                model.cache_position() == 4,
            "reused generation cache did not reset to zero state");
    }

    {
        constexpr std::uint64_t seed =
            0x51f15eeda55a1234ULL;
        auto model = make_model(true);
        mfq::metal::MlxSamplingParams sampling;
        sampling.temperature = 1.0;
        sampling.seed = seed;
        std::vector<std::int64_t> generated;
        const auto count = model.generate(
            {0, 1},
            sampling,
            4,
            [&](std::int64_t token) {
                generated.push_back(token);
                return true;
            });
        require(count == 4, "seeded generation count mismatch");

        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        std::vector<std::int64_t> expected;
        for (int index = 0; index < 4; ++index) {
            expected.push_back(uniform_token(uniform(rng), 8));
        }
        require(
            generated == expected,
            "generation seed did not match persistent CUDA RNG stream");

        std::vector<std::int64_t> repeated;
        const auto repeated_count = model.generate(
            {0, 1},
            sampling,
            4,
            [&](std::int64_t token) {
                repeated.push_back(token);
                return true;
            });
        require(
            repeated_count == 4 && repeated == generated,
            "generation seed was not reset per request");
    }

    {
        auto model = make_model(true);
        mfq::metal::MlxSamplingParams sampling;
        sampling.temperature = 0.0;
        sampling.presence_penalty = 0.2;
        sampling.frequency_penalty = 0.1;
        sampling.repetition_penalty = 2.0;
        std::vector<std::int64_t> generated;
        const auto count = model.generate(
            {0, 1},
            sampling,
            3,
            [&](std::int64_t token) {
                generated.push_back(token);
                return true;
            });
        require(
            count == 3 &&
                generated ==
                    std::vector<std::int64_t>({2, 3, 4}),
            "generation did not include prompt and generated token counts");
    }
}

void test_callback_stop_count() {
    auto model = make_model(true);
    mfq::metal::MlxSamplingParams sampling;
    sampling.temperature = 0.0;
    int callback_calls = 0;
    std::int64_t sampled = -1;
    const auto count = model.generate(
        {3, 4, 5},
        sampling,
        10,
        [&](std::int64_t token) {
            ++callback_calls;
            sampled = token;
            return false;
        });
    require(
        count == 1 &&
            callback_calls == 1 &&
            sampled == 0,
        "callback=false token was not counted before immediate stop");
    require(
        model.cache_position() == 3,
        "callback=false performed an extra decode");
}

void test_generation_context_boundary() {
    auto model = make_model(true);
    std::vector<std::int64_t> prompt(32);
    for (std::size_t index = 0; index < prompt.size(); ++index) {
        prompt[index] = static_cast<std::int64_t>(index % 8);
    }
    mfq::metal::MlxSamplingParams sampling;
    sampling.temperature = 0.0;
    int callback_calls = 0;
    const auto count = model.generate(
        prompt,
        sampling,
        5,
        [&](std::int64_t) {
            ++callback_calls;
            return true;
        });
    require(
        count == 1 && callback_calls == 1,
        "full-context generation sampled beyond the final logits");
    require(
        model.cache_position() == 32,
        "full-context generation forwarded into a full cache");

    prompt.resize(30);
    callback_calls = 0;
    const auto shorter_count = model.generate(
        prompt,
        sampling,
        5,
        [&](std::int64_t) {
            ++callback_calls;
            return true;
        });
    require(
        shorter_count == 3 && callback_calls == 3,
        "generation context limit did not include the final sampled token");
    require(
        model.cache_position() == 32,
        "bounded generation did not stop at full cache");
}

void test_validation() {
    const auto tied_config = test_config(true);
    bool separate_tied_output_rejected = false;
    try {
        (void)MlxQwen35CausalLm(
            tied_config,
            make_embedding(tied_config),
            make_layers(tied_config),
            hidden_norm(tied_config),
            make_output(tied_config, false),
            mlx::core::float32);
    } catch (const std::runtime_error& error) {
        separate_tied_output_rejected =
            std::string(error.what()).find(
                "must not provide a separate") !=
            std::string::npos;
    }
    require(
        separate_tied_output_rejected,
        "tied model accepted a separate output weight");

    const auto untied_config = test_config(false);
    bool missing_untied_output_rejected = false;
    try {
        (void)MlxQwen35CausalLm(
            untied_config,
            make_embedding(untied_config),
            make_layers(untied_config),
            hidden_norm(untied_config),
            std::nullopt,
            mlx::core::float32);
    } catch (const std::runtime_error& error) {
        missing_untied_output_rejected =
            std::string(error.what()).find(
                "require an output weight") !=
            std::string::npos;
    }
    require(
        missing_untied_output_rejected,
        "untied model accepted a missing output weight");

    auto model = make_model();
    bool empty_prompt_rejected = false;
    try {
        (void)model.generate(
            {},
            mfq::metal::MlxSamplingParams{},
            1);
    } catch (const std::invalid_argument&) {
        empty_prompt_rejected = true;
    }
    require(empty_prompt_rejected, "empty prompt was accepted");

    bool invalid_token_rejected = false;
    try {
        (void)model.generate(
            {8},
            mfq::metal::MlxSamplingParams{},
            1);
    } catch (const std::invalid_argument&) {
        invalid_token_rejected = true;
    }
    require(
        invalid_token_rejected,
        "out-of-range prompt token was accepted");
}

} // namespace

int main(int argc, char** argv) {
    try {
#ifdef MFQ_MLX_METALLIB_DEFAULT
        mlx::core::metal::set_metallib_path(
            MFQ_MLX_METALLIB_DEFAULT);
        mlx::core::set_default_device(
            mlx::core::Device::gpu);
#endif
        if (argc == 2) {
            const mfq::metal::MfqContainer container(argv[1]);
            const auto config = Qwen35Config::from_mfq(container);
            auto mtp = MlxQwen35MtpModule::load_if_present(
                container,
                config,
                mfq::metal::Qwen35TensorNames::hugging_face());
            require(mtp.has_value(), "real MFQ MTP head was not loaded");
            require(
                mtp->layer_count() ==
                    static_cast<std::size_t>(config.mtp_num_hidden_layers),
                "real MFQ MTP layer count mismatch");
            std::cout << "MFQ C++ Qwen3.5 MTP container load passed\n";
            return 0;
        }
        require(argc == 1, "usage: qwen35 causal test [MTP_ONLY.mfq]");
        test_layer_order_and_forward();
        test_explicit_mrope_semantics();
        test_explicit_mrope_cache_continuity();
        test_explicit_position_validation();
        test_prefill_decode_and_reset();
        test_mtp_greedy_identity();
        test_text_session_snapshot_restore();
        test_tied_embedding_forward_cache_and_generate();
        test_generation_greedy_seed_and_penalties();
        test_callback_stop_count();
        test_generation_context_boundary();
        test_validation();
        std::cout
            << "MFQ C++ Qwen3.5 mixed causal LM tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "MFQ C++ Qwen3.5 causal LM test failed: "
            << error.what()
            << '\n';
        return 1;
    }
}
