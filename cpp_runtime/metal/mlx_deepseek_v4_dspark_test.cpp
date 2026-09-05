#include "mlx_deepseek_v4_dspark.h"
#include "mlx_deepseek_v4_attention.h"
#include "mlx_moe.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

using mfq::metal::DeepseekV4Config;
using mfq::metal::MlxDeepseekV4DSpark;
using mfq::metal::MlxDeepseekV4DSparkAttentionComponents;
using mfq::metal::MlxDeepseekV4DSparkHeadComponents;
using mfq::metal::MlxDeepseekV4DSparkStageComponents;
using mfq::metal::MlxDeepseekV4Moe;
using mfq::metal::MlxEmbedding;
using mfq::metal::MlxLinear;
using mfq::metal::MlxRoutedLinear;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kHidden = 16;
constexpr int kVocab = 8;
constexpr int kExperts = 2;
constexpr int kIntermediate = 16;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void append(std::vector<std::uint8_t>& output, T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    output.insert(output.end(), bytes, bytes + sizeof(value));
}

std::vector<std::uint8_t> zero_nint_blob(int output, int input) {
    constexpr int bits = 4;
    constexpr int sub_bits = 2;
    constexpr int group = 16;
    const int groups = input / group;
    const std::size_t metadata =
        static_cast<std::size_t>(output) * groups;
    const std::size_t values = metadata * group;
    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(blob, bits);
    append<std::uint8_t>(blob, sub_bits);
    append<std::int32_t>(blob, group);
    append<std::int32_t>(blob, 0);
    append<std::int32_t>(blob, input);
    append<std::uint32_t>(blob, 2);
    append<std::int64_t>(blob, output);
    append<std::int64_t>(blob, input);
    append<std::uint32_t>(blob, output);
    append<std::uint32_t>(blob, groups);
    for (int row = 0; row < 2 * output; ++row) {
        append<std::uint16_t>(blob, 0);
    }
    blob.resize(
        blob.size() + 2 * ((metadata * sub_bits + 7) / 8) +
            (values * bits + 7) / 8,
        0);
    return blob;
}

std::vector<std::uint8_t> zero_nintm_blob(
    int output_per_expert,
    int input) {
    auto payload = zero_nint_blob(
        kExperts * output_per_expert, input);
    const std::string dtype = "NINT4";
    std::vector<std::uint8_t> blob{'N', 'I', 'M', '2'};
    append<std::uint32_t>(blob, kExperts);
    append<std::uint32_t>(blob, output_per_expert);
    append<std::uint32_t>(blob, input);
    append<std::uint32_t>(blob, 1);
    append<std::uint32_t>(blob, kExperts);
    append<std::uint32_t>(blob, static_cast<std::uint32_t>(dtype.size()));
    append<std::uint64_t>(blob, payload.size());
    append<std::uint64_t>(blob, 0);
    for (int expert = 0; expert < kExperts; ++expert) {
        append<std::int32_t>(blob, expert);
    }
    blob.insert(blob.end(), dtype.begin(), dtype.end());
    blob.insert(blob.end(), payload.begin(), payload.end());
    return blob;
}

array zeros(int output, int input) {
    return mlx::core::zeros(
        Shape{output, input}, mlx::core::float32);
}

DeepseekV4Config config() {
    DeepseekV4Config value;
    value.n_layers = 1;
    value.hidden = kHidden;
    value.n_experts = kExperts;
    value.top_k = 1;
    value.moe_inter = kIntermediate;
    value.n_shared = 1;
    value.n_heads = 2;
    value.head_dim = 8;
    value.q_lora_rank = 8;
    value.o_lora_rank = 4;
    value.o_groups = 2;
    value.kv_dim = 8;
    value.qk_rope_head_dim = 4;
    value.n_kv_heads = 1;
    value.vocab = kVocab;
    value.n_hash_layers = 0;
    value.sliding_window = 8;
    value.index_n_heads = 2;
    value.index_head_dim = 8;
    value.index_topk = 2;
    value.max_position_embeddings = 32;
    value.compress_ratios = {0};
    value.n_mtp_layers = 2;
    value.dspark_block_size = 3;
    value.dspark_noise_token_id = 7;
    value.dspark_target_layer_ids = {0};
    value.dspark_markov_rank = 4;
    value.mtp_compress_ratios = {0, 0};
    value.validate();
    return value;
}

MlxDeepseekV4Moe make_moe(const DeepseekV4Config& cfg) {
    return MlxDeepseekV4Moe(
        cfg,
        MlxLinear(zeros(kExperts, kHidden)),
        MlxLinear(zeros(kIntermediate, kHidden)),
        MlxLinear(zeros(kIntermediate, kHidden)),
        MlxLinear(zeros(kHidden, kIntermediate)),
        MlxRoutedLinear::from_blob(
            zero_nintm_blob(2 * kIntermediate, kHidden)),
        MlxRoutedLinear::from_blob(
            zero_nintm_blob(kHidden, kIntermediate)),
        mlx::core::zeros(Shape{kExperts}, mlx::core::float32));
}

MlxDeepseekV4DSparkStageComponents make_stage(
    const DeepseekV4Config& cfg) {
    return {
        MlxDeepseekV4DSparkAttentionComponents{
            MlxLinear(zeros(8, kHidden)),
            MlxLinear(zeros(16, 8)),
            MlxLinear(zeros(8, kHidden)),
            MlxLinear(zeros(8, 8)),
            MlxLinear(zeros(kHidden, 8)),
            mlx::core::ones(Shape{8}, mlx::core::float32),
            mlx::core::ones(Shape{8}, mlx::core::float32),
            mlx::core::zeros(Shape{2}, mlx::core::float32),
        },
        make_moe(cfg),
        mlx::core::ones(Shape{kHidden}, mlx::core::float32),
        mlx::core::ones(Shape{kHidden}, mlx::core::float32),
        MlxLinear(zeros(24, 4 * kHidden)),
        mlx::core::zeros(Shape{24}, mlx::core::float32),
        mlx::core::ones(Shape{3}, mlx::core::float32),
        MlxLinear(zeros(24, 4 * kHidden)),
        mlx::core::zeros(Shape{24}, mlx::core::float32),
        mlx::core::ones(Shape{3}, mlx::core::float32),
    };
}

MlxDeepseekV4DSpark make_dspark() {
    const auto cfg = config();
    std::vector<float> markov_embedding(
        kVocab * cfg.dspark_markov_rank, 0.0f);
    for (int token = 0; token < kVocab; ++token) {
        markov_embedding[
            static_cast<std::size_t>(token) * cfg.dspark_markov_rank] = 1.0f;
    }
    std::vector<float> markov_output(
        kVocab * cfg.dspark_markov_rank, 0.0f);
    markov_output[3 * cfg.dspark_markov_rank] = 2.0f;
    std::vector<MlxDeepseekV4DSparkStageComponents> stages;
    stages.push_back(make_stage(cfg));
    stages.push_back(make_stage(cfg));
    return MlxDeepseekV4DSpark(
        cfg,
        MlxEmbedding(zeros(kVocab, kHidden)),
        MlxLinear(zeros(kVocab, kHidden)),
        MlxLinear(zeros(kHidden, kHidden)),
        mlx::core::ones(Shape{kHidden}, mlx::core::float32),
        std::move(stages),
        MlxDeepseekV4DSparkHeadComponents{
            mlx::core::ones(Shape{kHidden}, mlx::core::float32),
            MlxLinear(zeros(4, 4 * kHidden)),
            mlx::core::zeros(Shape{4}, mlx::core::float32),
            mlx::core::ones(Shape{1}, mlx::core::float32),
            MlxEmbedding(array(
                markov_embedding.begin(),
                Shape{kVocab, static_cast<int>(cfg.dspark_markov_rank)})),
            MlxLinear(array(
                markov_output.begin(),
                Shape{kVocab, static_cast<int>(cfg.dspark_markov_rank)})),
            MlxLinear(zeros(
                1,
                kHidden + static_cast<int>(cfg.dspark_markov_rank))),
        },
        32,
        mfq::metal::deepseek_v4_yarn_tables(4, 32, 10'000.0f));
}

void test_context_and_parallel_draft() {
    auto dspark = make_dspark();
    auto state = dspark.make_state();
    auto prompt_hidden = mlx::core::zeros(
        Shape{1, 5, kHidden}, mlx::core::float16);
    dspark.append_context(prompt_hidden, state, 0);
    require(
        state.position() == 5 && state.stages() == 2,
        "DSpark context position mismatch");
    auto saved = state.snapshot();

    const array anchor({1}, Shape{1, 1}, mlx::core::int32);
    auto draft = dspark.draft_greedy(anchor, state);
    mlx::core::eval({draft.tokens, draft.logits, draft.confidence});
    require(
        draft.tokens.shape() == Shape{1, 3} &&
            draft.logits.shape() == Shape{1, 3, kVocab} &&
            draft.confidence.shape() == Shape{1, 3},
        "DSpark draft output geometry mismatch");
    const auto* tokens = draft.tokens.data<std::int32_t>();
    require(
        tokens[0] == 3 && tokens[1] == 3 && tokens[2] == 3,
        "DSpark sequential Markov decisions mismatch");
    const auto* confidence = draft.confidence.data<float>();
    for (int index = 0; index < 3; ++index) {
        require(
            std::fabs(confidence[index]) < 1e-6f,
            "DSpark confidence projection mismatch");
    }

    dspark.append_context(
        mlx::core::zeros(Shape{1, 2, kHidden}, mlx::core::float16),
        state,
        5);
    require(state.position() == 7, "DSpark context did not advance");
    state.restore_snapshot(std::move(saved));
    require(state.position() == 5, "DSpark snapshot did not roll back");
}

} // namespace

int main() {
    try {
        mlx::core::set_default_device(mlx::core::Device::gpu);
        test_context_and_parallel_draft();
        std::cout << "MFQ DeepSeek-V4 DSpark MoE MTP tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MFQ DeepSeek-V4 DSpark test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
