#include "mlx_deepseek_v41_vision.h"

#include "mlx_multimodal.h"
#include "mlx_tensor.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

array dense_vector(
    const MfqContainer& model,
    const std::string& name,
    int width) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" && record.dtype != "F16" &&
        record.dtype != "F32") {
        throw std::runtime_error(
            "DeepSeek-V4.1 vision sentinel must be dense: " + name);
    }
    const auto mapped = model.map_record(name);
    auto result = mlx::core::contiguous(
        load_dense_array(record.dtype, mapped.view()));
    if (result.shape() != Shape{width}) {
        throw std::runtime_error(
            "DeepSeek-V4.1 vision sentinel shape mismatch: " + name);
    }
    return result;
}

void validate_span(
    const MlxDeepseekV41ImageInput& image,
    int prompt_tokens) {
    if (image.begin < 0 || image.end <= image.begin ||
        image.end > prompt_tokens ||
        image.token_types.size() !=
            static_cast<std::size_t>(image.end - image.begin) ||
        image.n_vit_h <= 0 || image.n_vit_w <= 0) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 image span geometry mismatch");
    }
    if (std::any_of(
            image.token_types.begin(), image.token_types.end(),
            [](std::int64_t type) { return type < 0 || type > 3; })) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 image sentinel type is invalid");
    }
}

} // namespace

MlxDeepseekV41Vision MlxDeepseekV41Vision::load(
    const MfqContainer& model,
    const DeepseekV41Config& config) {
    if (!config.has_vision()) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 config has no vision tower");
    }
    MlxAxialPatchVisionConfig vision{
        config.vision.hidden,
        config.vision.n_layers,
        config.vision.n_heads,
        config.vision.intermediate,
        config.vision.patch_size,
        config.vision.downsample_ratio,
        config.hidden,
        config.vision.rope_theta,
        1e-6,
    };
    const int hidden = static_cast<int>(config.hidden);
    auto start = dense_vector(
        model, "vision.special_token.start", hidden);
    auto newline = dense_vector(
        model, "vision.special_token.newline", hidden);
    auto end = dense_vector(
        model, "vision.special_token.end", hidden);
    return MlxDeepseekV41Vision(
        config,
        MlxAxialPatchVisionTower::load(model, vision),
        mlx::core::stack(
            {
                std::move(start),
                mlx::core::zeros(Shape{hidden}, newline.dtype()),
                std::move(newline),
                std::move(end),
            },
            0));
}

MlxDeepseekV41Vision::MlxDeepseekV41Vision(
    DeepseekV41Config config,
    MlxAxialPatchVisionTower tower,
    array special_embeddings)
    : config_(std::move(config)),
      tower_(std::move(tower)),
      special_embeddings_(mlx::core::contiguous(
          std::move(special_embeddings))) {
    config_.validate();
    if (!config_.has_vision() ||
        special_embeddings_.shape() !=
            Shape{4, static_cast<int>(config_.hidden)} ||
        tower_.config().output_hidden != config_.hidden) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 vision component geometry mismatch");
    }
}

array MlxDeepseekV41Vision::encode(
    const array& patches,
    int n_vit_h,
    int n_vit_w) const {
    return tower_.encode(patches, n_vit_h, n_vit_w);
}

array MlxDeepseekV41Vision::embed_prompt(
    const std::vector<std::int64_t>& token_ids,
    const std::vector<MlxDeepseekV41ImageInput>& images,
    const MlxEmbedding& embedding,
    mlx::core::Dtype dtype) const {
    if (token_ids.empty() || images.empty() ||
        embedding.vocabulary_size() != config_.vocab ||
        embedding.hidden_size() != config_.hidden) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 multimodal prompt geometry mismatch");
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
    int previous_end = 0;
    for (const auto& image : images) {
        validate_span(image, static_cast<int>(token_ids.size()));
        if (image.begin < previous_end) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 image spans overlap or are unordered");
        }
        previous_end = image.end;
        auto aligned = encode(
            image.patches, image.n_vit_h, image.n_vit_w);
        const auto visual_count = static_cast<int>(std::count(
            image.token_types.begin(), image.token_types.end(), 1));
        if (visual_count != aligned.shape(0)) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 image tokens disagree with aligned patches");
        }
        const auto types = array(
            image.token_types.begin(),
            Shape{static_cast<int>(image.token_types.size())});
        auto block = mlx::core::take(special_embeddings_, types, 0);
        std::vector<std::int32_t> visual_indices(
            image.token_types.size(), 0);
        int visual = 0;
        for (std::size_t index = 0;
             index < image.token_types.size();
             ++index) {
            if (image.token_types[index] == 1) {
                visual_indices[index] = visual++;
            }
        }
        auto visual_rows = mlx::core::concatenate(
            {
                aligned,
                mlx::core::zeros(
                    Shape{1, static_cast<int>(config_.hidden)},
                    aligned.dtype()),
            },
            0);
        for (std::size_t index = 0;
             index < image.token_types.size();
             ++index) {
            if (image.token_types[index] != 1) {
                visual_indices[index] = aligned.shape(0);
            }
        }
        auto selected = mlx::core::take(
            visual_rows,
            array(
                visual_indices.begin(),
                Shape{static_cast<int>(visual_indices.size())}),
            0);
        block = mlx::core::where(
            mlx::core::expand_dims(
                mlx::core::equal(types, array(1, mlx::core::int64)),
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

array MlxDeepseekV41Vision::image_mask(
    int token_count,
    const std::vector<MlxDeepseekV41ImageInput>& images) const {
    if (token_count <= 0 || images.empty()) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 image mask requires a nonempty prompt");
    }
    std::vector<bool> mask(static_cast<std::size_t>(token_count), false);
    int previous_end = 0;
    for (const auto& image : images) {
        validate_span(image, token_count);
        if (image.begin < previous_end) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 image spans overlap or are unordered");
        }
        previous_end = image.end;
        std::fill(
            mask.begin() + image.begin,
            mask.begin() + image.end,
            true);
    }
    std::vector<std::uint8_t> bytes;
    bytes.reserve(mask.size());
    for (const auto value : mask) {
        bytes.push_back(static_cast<std::uint8_t>(value));
    }
    return mlx::core::astype(
        array(bytes.begin(), Shape{1, token_count}),
        mlx::core::bool_);
}

} // namespace mfq::metal
