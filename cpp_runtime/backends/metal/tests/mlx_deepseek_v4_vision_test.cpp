#include "mlx_deepseek_v4_vision.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

using mfq::metal::DeepseekV4Config;
using mfq::metal::MlxDeepseekV4Affine;
using mfq::metal::MlxDeepseekV4ImageInput;
using mfq::metal::MlxDeepseekV4Vision;
using mfq::metal::MlxDeepseekV4VisionBlock;
using mfq::metal::MlxEmbedding;
using mfq::metal::MlxLinear;
using mfq::metal::MlxRmsNorm;
using mlx::core::Shape;
using mlx::core::array;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

array zeros(int output, int input) {
    return mlx::core::zeros(
        Shape{output, input}, mlx::core::float32);
}

array constant_vector(int width, float value) {
    return mlx::core::full(
        Shape{width}, value, mlx::core::float32);
}

DeepseekV4Config config() {
    DeepseekV4Config value;
    value.n_layers = 1;
    value.hidden = 16;
    value.n_experts = 2;
    value.top_k = 1;
    value.moe_inter = 16;
    value.n_shared = 1;
    value.n_heads = 2;
    value.head_dim = 8;
    value.q_lora_rank = 8;
    value.o_lora_rank = 4;
    value.o_groups = 2;
    value.kv_dim = 8;
    value.qk_rope_head_dim = 4;
    value.n_kv_heads = 1;
    value.vocab = 8;
    value.n_hash_layers = 0;
    value.sliding_window = 8;
    value.index_n_heads = 2;
    value.index_head_dim = 8;
    value.index_topk = 2;
    value.max_position_embeddings = 32;
    value.compress_ratios = {0};
    value.vision_n_layers = 1;
    value.vision_dim = 4;
    value.vision_n_heads = 1;
    value.vision_inter_dim = 4;
    value.vision_patch_size = 2;
    value.vision_downsample_ratio = 2;
    value.validate();
    return value;
}

MlxDeepseekV4Vision make_vision() {
    const auto cfg = config();
    MlxDeepseekV4VisionBlock block{
        MlxRmsNorm(mlx::core::ones(Shape{4})),
        MlxDeepseekV4Affine(
            MlxLinear(zeros(12, 4)),
            mlx::core::zeros(Shape{12})),
        MlxDeepseekV4Affine(
            MlxLinear(zeros(4, 4)),
            mlx::core::zeros(Shape{4})),
        MlxRmsNorm(mlx::core::ones(Shape{4})),
        MlxLinear(zeros(8, 4)),
        MlxLinear(zeros(4, 4)),
    };
    return MlxDeepseekV4Vision(
        cfg,
        MlxDeepseekV4Affine(
            MlxLinear(zeros(4, 12)),
            mlx::core::zeros(Shape{4})),
        {std::move(block)},
        mlx::core::ones(Shape{4}),
        MlxDeepseekV4Affine(
            MlxLinear(zeros(16, 16)),
            mlx::core::zeros(Shape{16})),
        MlxDeepseekV4Affine(
            MlxLinear(zeros(16, 16)),
            mlx::core::zeros(Shape{16})),
        constant_vector(16, 10.0f),
        constant_vector(16, 20.0f),
        constant_vector(16, 30.0f),
        constant_vector(16, 40.0f));
}

void test_encode_and_common_prompt_injection() {
    auto vision = make_vision();
    auto encoded = vision.encode(
        mlx::core::zeros(Shape{4, 12}), 2, 2);
    encoded.eval();
    require(
        encoded.shape() == Shape{1, 16},
        "vision aligner geometry mismatch");
    for (std::size_t index = 0; index < encoded.size(); ++index) {
        require(
            std::fabs(encoded.data<float>()[index]) < 1e-6f,
            "zero vision tower did not remain zero");
    }

    std::vector<float> embedding_values(8 * 16);
    for (int token = 0; token < 8; ++token) {
        for (int column = 0; column < 16; ++column) {
            embedding_values[static_cast<std::size_t>(token * 16 + column)] =
                static_cast<float>(token);
        }
    }
    MlxEmbedding embedding(array(
        embedding_values.begin(), Shape{8, 16}));
    const std::vector<std::int64_t> tokens{1, -1, -1, -1, -1, -1, 6};
    MlxDeepseekV4ImageInput image{
        mlx::core::zeros(Shape{4, 12}),
        2,
        2,
        1,
        6,
        {0, 2, 3, 1, 4},
        {0},
    };
    auto prompt = vision.embed_prompt(
        tokens, {std::move(image)}, embedding, mlx::core::float32);
    prompt.eval();
    require(
        prompt.shape() == Shape{1, 7, 16},
        "multimodal prompt geometry mismatch");
    const auto* values = prompt.data<float>();
    const std::vector<float> expected{1.0f, 10.0f, 0.0f, 30.0f,
                                      20.0f, 40.0f, 6.0f};
    for (int token = 0; token < 7; ++token) {
        for (int column = 0; column < 16; ++column) {
            require(
                std::fabs(values[token * 16 + column] - expected[token]) <
                    1e-6f,
                "multimodal span injection mismatch");
        }
    }
}

} // namespace

int main() {
    try {
        mlx::core::set_default_device(mlx::core::Device::gpu);
        test_encode_and_common_prompt_injection();
        std::cout << "MFQ DeepSeek-V4 vision tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
