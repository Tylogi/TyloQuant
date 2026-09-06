#include "mlx_qwen35_full_attention.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace {

using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MlxLinear;
using mfq::metal::MlxQwen35DenseSwiGlu;
using mfq::metal::MlxQwen35FullAttentionBlock;
using mfq::metal::MlxRmsNorm;
using mfq::metal::Qwen35Config;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    float actual,
    float expected,
    float tolerance = 2e-4f) {
    if (!std::isfinite(actual) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            "Qwen3.5 full-attention mismatch: actual=" +
            std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

array dense(
    int rows,
    int columns,
    const std::vector<float>& values) {
    require(
        static_cast<int>(values.size()) == rows * columns,
        "invalid test matrix data");
    return array(values.begin(), Shape{rows, columns});
}

std::vector<float> patterned(
    int size,
    float scale,
    int period,
    int center) {
    std::vector<float> values;
    values.reserve(size);
    for (int index = 0; index < size; ++index) {
        values.push_back(
            static_cast<float>(index % period - center) * scale);
    }
    return values;
}

Qwen35Config test_config() {
    Qwen35Config config;
    config.model_type = "qwen3_5";
    config.vocab_size = 16;
    config.hidden_size = 4;
    config.intermediate_size = 5;
    config.num_hidden_layers = 1;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;
    config.max_position_embeddings = 16;
    config.head_dim = 2;
    config.rope_base = 10'000.0;
    config.rotary_dim = 2;
    config.rope_sections.clear();
    config.rms_norm_eps = 1e-6;
    config.norm_weight_offset = 1.0;
    config.attention_output_gate = true;
    config.layer_types = {"full_attention"};
    return config;
}

MlxQwen35FullAttentionBlock make_block() {
    auto config = test_config();
    const array hidden_norm(
        {0.0f, 0.05f, -0.04f, 0.02f},
        Shape{4});
    const array head_norm({0.03f, -0.02f}, Shape{2});

    return MlxQwen35FullAttentionBlock(
        config,
        MlxRmsNorm(
            hidden_norm,
            static_cast<float>(config.rms_norm_eps),
            static_cast<float>(config.norm_weight_offset)),
        MlxLinear(dense(
            8,
            4,
            patterned(32, 0.035f, 9, 4))),
        MlxLinear(dense(
            2,
            4,
            patterned(8, 0.055f, 7, 3))),
        MlxLinear(dense(
            2,
            4,
            patterned(8, 0.075f, 5, 2))),
        MlxLinear(dense(
            4,
            4,
            patterned(16, 0.045f, 7, 3))),
        MlxRmsNorm(
            head_norm,
            static_cast<float>(config.rms_norm_eps),
            static_cast<float>(config.norm_weight_offset)),
        MlxRmsNorm(
            head_norm,
            static_cast<float>(config.rms_norm_eps),
            static_cast<float>(config.norm_weight_offset)),
        MlxRmsNorm(
            hidden_norm,
            static_cast<float>(config.rms_norm_eps),
            static_cast<float>(config.norm_weight_offset)),
        MlxQwen35DenseSwiGlu(
            MlxLinear(dense(
                5,
                4,
                patterned(20, 0.025f, 9, 4))),
            MlxLinear(dense(
                5,
                4,
                patterned(20, 0.03f, 7, 3))),
            MlxLinear(dense(
                4,
                5,
                patterned(20, 0.02f, 11, 5)))));
}

void test_swiglu() {
    MlxQwen35DenseSwiGlu ffn(
        MlxLinear(dense(
            2,
            2,
            {1.0f, 0.0f, 0.0f, 1.0f})),
        MlxLinear(dense(
            2,
            2,
            {1.0f, 0.0f, 0.0f, 1.0f})),
        MlxLinear(dense(
            2,
            2,
            {1.0f, 0.0f, 0.0f, 1.0f})));
    auto result = ffn(array({1.0f, 2.0f}, Shape{1, 1, 2}));
    result.eval();
    const auto* values = result.data<float>();
    require_close(values[0], 1.0f / (1.0f + std::exp(-1.0f)));
    require_close(values[1], 4.0f / (1.0f + std::exp(-2.0f)));
}

void test_important_neuron_swiglu() {
    auto high = std::make_shared<MlxQwen35DenseSwiGlu>(
        MlxLinear(dense(1, 2, {0.0f, 1.0f})),
        MlxLinear(dense(1, 2, {0.0f, 1.0f})),
        MlxLinear(dense(2, 1, {0.0f, 1.0f})));
    MlxQwen35DenseSwiGlu ffn(
        MlxLinear(dense(1, 2, {1.0f, 0.0f})),
        MlxLinear(dense(1, 2, {1.0f, 0.0f})),
        MlxLinear(dense(2, 1, {1.0f, 0.0f})),
        std::move(high));

    auto result = ffn(array({1.0f, 2.0f}, Shape{1, 1, 2}));
    result.eval();
    const auto* values = result.data<float>();
    require(
        ffn.intermediate_size() == 2,
        "important-neuron intermediate width was not combined");
    require_close(values[0], 1.0f / (1.0f + std::exp(-1.0f)));
    require_close(values[1], 4.0f / (1.0f + std::exp(-2.0f)));
}

void test_decode_projection_routing() {
    require(
        !mfq::metal::detail::qwen35_use_grouped_projection_rows(
            64, 64),
        "single-row decode unexpectedly selected grouped projection");
    require(
        mfq::metal::detail::qwen35_use_grouped_projection_rows(
            128, 64),
        "multi-row prefill did not select grouped projection");
}

void test_prefill_decode_cache_equivalence() {
    const array full_input(
        {
            0.25f, -0.5f, 0.75f, 0.1f,
            -0.2f, 0.4f, 0.3f, -0.7f,
            0.6f, 0.15f, -0.35f, 0.8f,
        },
        Shape{1, 3, 4});
    const array prefix_input(
        {
            0.25f, -0.5f, 0.75f, 0.1f,
            -0.2f, 0.4f, 0.3f, -0.7f,
        },
        Shape{1, 2, 4});
    const array decode_input(
        {0.6f, 0.15f, -0.35f, 0.8f},
        Shape{1, 1, 4});

    auto full_block = make_block();
    auto full_output = full_block.forward(full_input, 0, true);

    auto decode_block = make_block();
    auto prefix_output =
        decode_block.forward(prefix_input, 0, true);
    auto decode_output =
        decode_block.forward(decode_input, 2, true);

    auto uncached_block = make_block();
    auto uncached_output =
        uncached_block.forward(full_input, 0, false);

    mlx::core::eval(
        full_output,
        prefix_output,
        decode_output,
        uncached_output);
    require(
        full_block.cache_position() == 3,
        "full prefill cache position mismatch");
    require(
        decode_block.cache_position() == 3,
        "decode cache position mismatch");
    require(
        uncached_block.cache_position() == 0,
        "uncached block unexpectedly created a cache");

    const auto* full = full_output.data<float>();
    const auto* decoded = decode_output.data<float>();
    const auto* uncached = uncached_output.data<float>();
    for (int column = 0; column < 4; ++column) {
        require_close(decoded[column], full[8 + column], 4e-4f);
    }
    for (int index = 0; index < 12; ++index) {
        require_close(uncached[index], full[index], 4e-4f);
    }
    require(
        std::fabs(full[8] - 0.6f) > 1e-5f,
        "attention/FFN path did not update the residual");

    bool rejected_noncontiguous = false;
    try {
        (void)decode_block.forward(decode_input, 1, true);
    } catch (const std::runtime_error&) {
        rejected_noncontiguous = true;
    }
    require(
        rejected_noncontiguous,
        "non-contiguous KV cache append was not rejected");

    decode_block.clear_cache();
    require(
        decode_block.cache_position() == 0 &&
            decode_block.cache_batch() == 0,
        "clear_cache did not clear cache state");
}

} // namespace

int main() {
    try {
#ifdef MFQ_MLX_METALLIB_DEFAULT
        mlx::core::metal::set_metallib_path(
            MFQ_MLX_METALLIB_DEFAULT);
        mlx::core::set_default_device(mlx::core::Device::gpu);
#endif
        test_swiglu();
        test_important_neuron_swiglu();
        test_decode_projection_routing();
        test_prefill_decode_cache_equivalence();
        std::cout
            << "MFQ C++ Qwen3.5 full-attention block tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
