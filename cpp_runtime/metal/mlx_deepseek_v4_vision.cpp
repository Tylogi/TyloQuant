#include "mlx_deepseek_v4_vision.h"

#include "mlx_multimodal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

array mfq_dense(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    return load_dense_array(record.dtype, model.read(name));
}

MlxDeepseekV4Affine mfq_affine(
    const MfqContainer& model,
    const std::string& prefix) {
    return MlxDeepseekV4Affine(
        MlxLinear::load(model, prefix + ".weight"),
        mfq_dense(model, prefix + ".bias"));
}

MlxDeepseekV4Affine hf_affine(
    const MlxHfTensorStore& model,
    const std::string& prefix) {
    return MlxDeepseekV4Affine(
        model.load_linear(prefix + ".weight"),
        model.load_dense(prefix + ".bias"));
}

array exact_gelu(const array& input) {
    constexpr float kInvSqrtTwo = 0.7071067811865475f;
    return 0.5f * input *
        (1.0f + mlx::core::erf(input * kInvSqrtTwo));
}

array apply_vision_rope(
    const array& input,
    int height,
    int width,
    float theta) {
    if (input.ndim() != 3 || input.shape(0) != height * width ||
        input.shape(2) <= 0 || input.shape(2) % 4 != 0) {
        throw std::invalid_argument(
            "DeepSeek-V4 vision RoPE input geometry is invalid");
    }
    const int head_dim = input.shape(2);
    const int rope_dim = head_dim / 2;
    const int frequencies = rope_dim / 2;
    std::vector<float> cosine(
        static_cast<size_t>(height * width * rope_dim));
    std::vector<float> sine(cosine.size());
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            const size_t token =
                static_cast<size_t>(row * width + column);
            for (int index = 0; index < frequencies; ++index) {
                const float inverse = std::pow(
                    theta,
                    -static_cast<float>(2 * index) /
                        static_cast<float>(rope_dim));
                for (int axis = 0; axis < 2; ++axis) {
                    const float position = static_cast<float>(
                        axis == 0 ? row : column);
                    const size_t offset =
                        token * static_cast<size_t>(rope_dim) +
                        static_cast<size_t>(axis * frequencies + index);
                    const float angle = position * inverse;
                    cosine[offset] = std::cos(angle);
                    sine[offset] = std::sin(angle);
                }
            }
        }
    }
    auto cos_value = array(
        cosine.begin(),
        Shape{height * width, 1, rope_dim});
    auto sin_value = array(
        sine.begin(),
        Shape{height * width, 1, rope_dim});
    auto pieces = mlx::core::split(input, 2, -1);
    auto rotated = mlx::core::concatenate(
        {
            pieces.at(0) * cos_value - pieces.at(1) * sin_value,
            pieces.at(1) * cos_value + pieces.at(0) * sin_value,
        },
        -1);
    return mlx::core::astype(rotated, input.dtype());
}

void validate_vector(
    const array& value,
    int width,
    const char* name) {
    if (value.ndim() != 1 || value.shape(0) != width) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4 ") + name + " shape mismatch");
    }
}

} // namespace

MlxDeepseekV4Affine::MlxDeepseekV4Affine(
    MlxLinear linear,
    array bias)
    : linear_(std::move(linear)),
      bias_(mlx::core::contiguous(std::move(bias))) {
    validate_vector(bias_, linear_.output_size(), "affine bias");
}

array MlxDeepseekV4Affine::operator()(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size()) {
        throw std::invalid_argument(
            "DeepSeek-V4 affine input width mismatch");
    }
    return linear_(input) + bias_;
}

MlxDeepseekV4Vision MlxDeepseekV4Vision::load(
    const MfqContainer& model,
    const DeepseekV4Config& config) {
    if (!config.has_vision()) {
        throw std::invalid_argument(
            "DeepSeek-V4 config has no vision tower");
    }
    std::vector<MlxDeepseekV4VisionBlock> blocks;
    blocks.reserve(static_cast<size_t>(config.vision_n_layers));
    for (int64_t index = 0; index < config.vision_n_layers; ++index) {
        const auto prefix =
            "vision.blocks." + std::to_string(index);
        blocks.push_back({
            MlxRmsNorm(mfq_dense(model, prefix + ".norm1.weight")),
            mfq_affine(model, prefix + ".attn.wqkv"),
            mfq_affine(model, prefix + ".attn.wo"),
            MlxRmsNorm(mfq_dense(model, prefix + ".norm2.weight")),
            MlxLinear::load(model, prefix + ".mlp.w1.weight"),
            MlxLinear::load(model, prefix + ".mlp.w2.weight"),
        });
    }
    return MlxDeepseekV4Vision(
        config,
        mfq_affine(model, "vision.patch_embed.proj"),
        std::move(blocks),
        mfq_dense(model, "vision.norm.weight"),
        mfq_affine(model, "aligner.w1"),
        mfq_affine(model, "aligner.w2"),
        mfq_dense(model, "image_start"),
        mfq_dense(model, "image_pad"),
        mfq_dense(model, "image_newline"),
        mfq_dense(model, "image_end"));
}

MlxDeepseekV4Vision MlxDeepseekV4Vision::load(
    const MlxHfTensorStore& model,
    const DeepseekV4Config& config) {
    if (!config.has_vision()) {
        throw std::invalid_argument(
            "DeepSeek-V4 config has no vision tower");
    }
    std::vector<MlxDeepseekV4VisionBlock> blocks;
    blocks.reserve(static_cast<size_t>(config.vision_n_layers));
    for (int64_t index = 0; index < config.vision_n_layers; ++index) {
        const auto prefix =
            "vision.blocks." + std::to_string(index);
        blocks.push_back({
            MlxRmsNorm(model.load_dense(prefix + ".norm1.weight")),
            hf_affine(model, prefix + ".attn.wqkv"),
            hf_affine(model, prefix + ".attn.wo"),
            MlxRmsNorm(model.load_dense(prefix + ".norm2.weight")),
            model.load_linear(prefix + ".mlp.w1.weight"),
            model.load_linear(prefix + ".mlp.w2.weight"),
        });
    }
    return MlxDeepseekV4Vision(
        config,
        hf_affine(model, "vision.patch_embed.proj"),
        std::move(blocks),
        model.load_dense("vision.norm.weight"),
        hf_affine(model, "aligner.w1"),
        hf_affine(model, "aligner.w2"),
        model.load_dense("image_start"),
        model.load_dense("image_pad"),
        model.load_dense("image_newline"),
        model.load_dense("image_end"));
}

MlxDeepseekV4Vision::MlxDeepseekV4Vision(
    DeepseekV4Config config,
    MlxDeepseekV4Affine patch_embedding,
    std::vector<MlxDeepseekV4VisionBlock> blocks,
    array final_norm,
    MlxDeepseekV4Affine aligner_input,
    MlxDeepseekV4Affine aligner_output,
    array image_start,
    array image_pad,
    array image_newline,
    array image_end)
    : config_(std::move(config)),
      patch_embedding_(std::move(patch_embedding)),
      blocks_(std::move(blocks)),
      final_norm_(std::move(final_norm)),
      aligner_input_(std::move(aligner_input)),
      aligner_output_(std::move(aligner_output)),
      image_parameters_(mlx::core::stack(
          {
              std::move(image_start),
              image_pad,
              std::move(image_pad),
              std::move(image_newline),
              std::move(image_end),
          },
          0)) {
    config_.validate();
    const int patch_width = static_cast<int>(
        3 * config_.vision_patch_size * config_.vision_patch_size);
    const int align_width = static_cast<int>(
        config_.vision_dim * config_.vision_downsample_ratio *
        config_.vision_downsample_ratio);
    if (!config_.has_vision() ||
        blocks_.size() != static_cast<size_t>(config_.vision_n_layers) ||
        patch_embedding_.input_size() != patch_width ||
        patch_embedding_.output_size() != config_.vision_dim ||
        final_norm_.width() != config_.vision_dim ||
        aligner_input_.input_size() != align_width ||
        aligner_input_.output_size() != config_.hidden ||
        aligner_output_.input_size() != config_.hidden ||
        aligner_output_.output_size() != config_.hidden ||
        image_parameters_.shape() != Shape{5, static_cast<int>(config_.hidden)}) {
        throw std::invalid_argument(
            "DeepSeek-V4 vision component geometry mismatch");
    }
    for (const auto& block : blocks_) {
        if (block.norm1.width() != config_.vision_dim ||
            block.qkv.input_size() != config_.vision_dim ||
            block.qkv.output_size() != 3 * config_.vision_dim ||
            block.output.input_size() != config_.vision_dim ||
            block.output.output_size() != config_.vision_dim ||
            block.norm2.width() != config_.vision_dim ||
            block.gate_up.input_size() != config_.vision_dim ||
            block.gate_up.output_size() != 2 * config_.vision_inter_dim ||
            block.down.input_size() != config_.vision_inter_dim ||
            block.down.output_size() != config_.vision_dim) {
            throw std::invalid_argument(
                "DeepSeek-V4 vision block geometry mismatch");
        }
    }
}

array MlxDeepseekV4Vision::attention(
    const MlxDeepseekV4VisionBlock& block,
    const array& input,
    int n_vit_h,
    int n_vit_w) const {
    const int tokens = n_vit_h * n_vit_w;
    const int heads = static_cast<int>(config_.vision_n_heads);
    const int head_dim = static_cast<int>(config_.vision_dim / heads);
    auto qkv = mlx::core::split(block.qkv(input), 3, -1);
    auto query = apply_vision_rope(
        mlx::core::reshape(qkv.at(0), Shape{tokens, heads, head_dim}),
        n_vit_h, n_vit_w, static_cast<float>(config_.vision_rope_theta));
    auto key = apply_vision_rope(
        mlx::core::reshape(qkv.at(1), Shape{tokens, heads, head_dim}),
        n_vit_h, n_vit_w, static_cast<float>(config_.vision_rope_theta));
    auto value = mlx::core::reshape(
        qkv.at(2), Shape{tokens, heads, head_dim});
    query = mlx::core::transpose(
        mlx::core::expand_dims(query, 0), {0, 2, 1, 3});
    key = mlx::core::transpose(
        mlx::core::expand_dims(key, 0), {0, 2, 1, 3});
    value = mlx::core::transpose(
        mlx::core::expand_dims(value, 0), {0, 2, 1, 3});
    auto output = scaled_dot_product_attention(
        query, key, value, false);
    output = mlx::core::reshape(
        mlx::core::transpose(output, {0, 2, 1, 3}),
        Shape{tokens, static_cast<int>(config_.vision_dim)});
    return block.output(output);
}

array MlxDeepseekV4Vision::align(
    const array& input,
    int n_vit_h,
    int n_vit_w) const {
    const int ratio = static_cast<int>(config_.vision_downsample_ratio);
    const int padded_h = (n_vit_h + ratio - 1) / ratio * ratio;
    const int padded_w = (n_vit_w + ratio - 1) / ratio * ratio;
    const int out_h = padded_h / ratio;
    const int out_w = padded_w / ratio;
    const int padding_index = n_vit_h * n_vit_w;
    std::vector<std::int32_t> indices;
    indices.reserve(static_cast<size_t>(out_h * out_w * ratio * ratio));
    for (int out_row = 0; out_row < out_h; ++out_row) {
        for (int out_column = 0; out_column < out_w; ++out_column) {
            for (int patch_row = 0; patch_row < ratio; ++patch_row) {
                for (int patch_column = 0; patch_column < ratio; ++patch_column) {
                    const int row = out_row * ratio + patch_row;
                    const int column = out_column * ratio + patch_column;
                    indices.push_back(
                        row < n_vit_h && column < n_vit_w
                            ? row * n_vit_w + column
                            : padding_index);
                }
            }
        }
    }
    auto padded = mlx::core::concatenate(
        {
            input,
            mlx::core::zeros(
                Shape{1, static_cast<int>(config_.vision_dim)},
                input.dtype()),
        },
        0);
    auto gathered = mlx::core::take(
        padded,
        array(indices.begin(), Shape{
            out_h * out_w, ratio * ratio}),
        0);
    // torch.unfold flattens channel before the 3x3 kernel axes.
    gathered = mlx::core::reshape(
        mlx::core::transpose(gathered, {0, 2, 1}),
        Shape{
            out_h * out_w,
            static_cast<int>(config_.vision_dim) * ratio * ratio});
    return aligner_output_(exact_gelu(aligner_input_(gathered)));
}

array MlxDeepseekV4Vision::encode(
    const array& patches,
    int n_vit_h,
    int n_vit_w) const {
    const int count = n_vit_h * n_vit_w;
    const int patch_width = static_cast<int>(
        3 * config_.vision_patch_size * config_.vision_patch_size);
    if (patches.ndim() != 2 || patches.shape(0) != count ||
        patches.shape(1) != patch_width) {
        throw std::invalid_argument(
            "DeepSeek-V4 vision patch geometry mismatch");
    }
    auto hidden = patch_embedding_(patches);
    for (const auto& block : blocks_) {
        hidden = hidden + attention(
            block, block.norm1(hidden), n_vit_h, n_vit_w);
        auto gate_up = mlx::core::split(
            block.gate_up(block.norm2(hidden)), 2, -1);
        hidden = hidden + block.down(
            gate_up.at(0) * mlx::core::sigmoid(gate_up.at(0)) *
            gate_up.at(1));
    }
    return align(final_norm_(hidden), n_vit_h, n_vit_w);
}

array MlxDeepseekV4Vision::embed_prompt(
    const std::vector<std::int64_t>& token_ids,
    const std::vector<MlxDeepseekV4ImageInput>& images,
    const MlxEmbedding& embedding,
    mlx::core::Dtype dtype) const {
    if (token_ids.empty() || embedding.vocabulary_size() != config_.vocab ||
        embedding.hidden_size() != config_.hidden) {
        throw std::invalid_argument(
            "DeepSeek-V4 multimodal prompt/embedding geometry mismatch");
    }
    std::vector<std::int32_t> safe_ids;
    safe_ids.reserve(token_ids.size());
    for (const auto token : token_ids) {
        safe_ids.push_back(static_cast<std::int32_t>(
            token >= 0 && token < config_.vocab ? token : 0));
    }
    auto result = embedding(
        array(
            safe_ids.begin(),
            Shape{1, static_cast<int>(safe_ids.size())}),
        dtype);
    for (const auto& image : images) {
        if (image.begin < 0 || image.end <= image.begin ||
            image.end > static_cast<int>(token_ids.size()) ||
            image.token_types.size() !=
                static_cast<size_t>(image.end - image.begin)) {
            throw std::invalid_argument(
                "DeepSeek-V4 image embedding span mismatch");
        }
        auto aligned = encode(
            image.patches, image.n_vit_h, image.n_vit_w);
        if (image.permutation.size() !=
            static_cast<size_t>(aligned.shape(0))) {
            throw std::invalid_argument(
                "DeepSeek-V4 image permutation length mismatch");
        }
        auto ordered = mlx::core::take(
            aligned,
            array(
                image.permutation.begin(),
                Shape{static_cast<int>(image.permutation.size())}),
            0);
        auto visual_rows = mlx::core::concatenate(
            {
                mlx::core::zeros(
                    Shape{1, static_cast<int>(config_.hidden)},
                    ordered.dtype()),
                ordered,
            },
            0);
        std::vector<std::int32_t> visual_indices(
            image.token_types.size(), 0);
        int visual = 0;
        for (size_t index = 0; index < image.token_types.size(); ++index) {
            const auto type = image.token_types[index];
            if (type < 0 || type > 4) {
                throw std::invalid_argument(
                    "DeepSeek-V4 image sentinel type is invalid");
            }
            if (type == 2) {
                visual_indices[index] = ++visual;
            }
        }
        if (visual != ordered.shape(0)) {
            throw std::invalid_argument(
                "DeepSeek-V4 image tokens disagree with aligned patches");
        }
        const auto types = array(
            image.token_types.begin(),
            Shape{static_cast<int>(image.token_types.size())});
        auto block = mlx::core::take(image_parameters_, types, 0);
        const auto selected = mlx::core::take(
            visual_rows,
            array(
                visual_indices.begin(),
                Shape{static_cast<int>(visual_indices.size())}),
            0);
        block = mlx::core::where(
            mlx::core::expand_dims(
                mlx::core::equal(types, array(2, mlx::core::int64)),
                -1),
            selected,
            block);
        result = replace_multimodal_embedding_span(
            result,
            mlx::core::expand_dims(block, 0),
            0,
            image.begin,
            image.end);
    }
    return result;
}

} // namespace mfq::metal
