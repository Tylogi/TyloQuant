#pragma once

#include "deepseek_v4_model.h"
#include "mlx_hf_tensor.h"
#include "mlx_tensor.h"
#include "mlx_transformer.h"

#include <cstdint>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV4ImageInput {
    mlx::core::array patches;
    int n_vit_h = 0;
    int n_vit_w = 0;
    int begin = 0;
    int end = 0;
    std::vector<std::int64_t> token_types;
    std::vector<std::int32_t> permutation;
};

class MlxDeepseekV4Affine {
public:
    MlxDeepseekV4Affine(
        MlxLinear linear,
        mlx::core::array bias);

    mlx::core::array operator()(
        const mlx::core::array& input) const;
    int input_size() const noexcept { return linear_.input_size(); }
    int output_size() const noexcept { return linear_.output_size(); }

private:
    MlxLinear linear_;
    mlx::core::array bias_;
};

struct MlxDeepseekV4VisionBlock {
    MlxRmsNorm norm1;
    MlxDeepseekV4Affine qkv;
    MlxDeepseekV4Affine output;
    MlxRmsNorm norm2;
    MlxLinear gate_up;
    MlxLinear down;
};

// Native implementation of the released DSV4 ViT + 3x3 aligner.  It owns no
// text-model policy: its output is a set of ordinary embedding spans consumed
// through the common multimodal injection API.
class MlxDeepseekV4Vision {
public:
    static MlxDeepseekV4Vision load(
        const MfqContainer& model,
        const DeepseekV4Config& config);
    static MlxDeepseekV4Vision load(
        const MlxHfTensorStore& model,
        const DeepseekV4Config& config);

    MlxDeepseekV4Vision(
        DeepseekV4Config config,
        MlxDeepseekV4Affine patch_embedding,
        std::vector<MlxDeepseekV4VisionBlock> blocks,
        mlx::core::array final_norm,
        MlxDeepseekV4Affine aligner_input,
        MlxDeepseekV4Affine aligner_output,
        mlx::core::array image_start,
        mlx::core::array image_pad,
        mlx::core::array image_newline,
        mlx::core::array image_end);

    mlx::core::array encode(
        const mlx::core::array& patches,
        int n_vit_h,
        int n_vit_w) const;

    mlx::core::array embed_prompt(
        const std::vector<std::int64_t>& token_ids,
        const std::vector<MlxDeepseekV4ImageInput>& images,
        const MlxEmbedding& embedding,
        mlx::core::Dtype dtype = mlx::core::float16) const;

    const DeepseekV4Config& config() const noexcept { return config_; }

private:
    mlx::core::array attention(
        const MlxDeepseekV4VisionBlock& block,
        const mlx::core::array& input,
        int n_vit_h,
        int n_vit_w) const;
    mlx::core::array align(
        const mlx::core::array& input,
        int n_vit_h,
        int n_vit_w) const;

    DeepseekV4Config config_;
    MlxDeepseekV4Affine patch_embedding_;
    std::vector<MlxDeepseekV4VisionBlock> blocks_;
    MlxRmsNorm final_norm_;
    MlxDeepseekV4Affine aligner_input_;
    MlxDeepseekV4Affine aligner_output_;
    mlx::core::array image_parameters_;
};

} // namespace mfq::metal
