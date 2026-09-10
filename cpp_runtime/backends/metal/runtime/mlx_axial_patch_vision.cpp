#include "mlx_axial_patch_vision.h"

#include "mlx_tensor.h"
#include "mlx_transformer.h"

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

int checked_int(std::int64_t value, const char* name) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            std::string("invalid axial patch vision ") + name);
    }
    return static_cast<int>(value);
}

array dense_array(const MfqContainer& model, const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" && record.dtype != "F16" &&
        record.dtype != "F32") {
        throw std::runtime_error(
            "axial patch vision requires dense tensor " + name);
    }
    const auto mapped = model.map_record(name);
    return mlx::core::contiguous(
        load_dense_array(record.dtype, mapped.view()));
}

struct Affine {
    MlxLinear linear;
    std::optional<array> bias;

    static Affine load(
        const MfqContainer& model,
        const std::string& prefix) {
        const auto bias_name = prefix + ".bias";
        std::optional<array> bias;
        if (model.contains(bias_name)) {
            bias = dense_array(model, bias_name);
        }
        Affine result{
            MlxLinear::load(model, prefix + ".weight"),
            std::move(bias),
        };
        if (result.bias &&
            (result.bias->ndim() != 1 ||
             result.bias->shape(0) != result.linear.output_size())) {
            throw std::runtime_error(
                "axial patch vision affine bias mismatch: " + prefix);
        }
        return result;
    }

    array operator()(const array& input) const {
        auto output = linear(input);
        if (bias) output = output + mlx::core::astype(*bias, output.dtype());
        return output;
    }
};

array exact_gelu(const array& input) {
    constexpr float kInvSqrtTwo = 0.7071067811865475f;
    const auto dtype = input.dtype();
    auto source = mlx::core::astype(input, mlx::core::float32);
    auto result = 0.5f * source *
        (1.0f + mlx::core::erf(source * kInvSqrtTwo));
    return dtype == mlx::core::float32
        ? result
        : mlx::core::astype(std::move(result), dtype);
}

array apply_rope(
    const array& input,
    int height,
    int width,
    float theta) {
    if (input.ndim() != 3 || input.shape(0) != height * width ||
        input.shape(2) <= 0 || input.shape(2) % 4 != 0) {
        throw std::invalid_argument(
            "axial patch vision RoPE geometry mismatch");
    }
    const int head_dim = input.shape(2);
    const int rope_dim = head_dim / 2;
    const int frequencies = rope_dim / 2;
    std::vector<float> cosine(
        static_cast<std::size_t>(height * width * rope_dim));
    std::vector<float> sine(cosine.size());
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            const auto token = static_cast<std::size_t>(row * width + column);
            for (int index = 0; index < frequencies; ++index) {
                const float inverse = std::pow(
                    theta,
                    -static_cast<float>(2 * index) /
                        static_cast<float>(rope_dim));
                for (int axis = 0; axis < 2; ++axis) {
                    const float position = static_cast<float>(
                        axis == 0 ? row : column);
                    const auto offset =
                        token * static_cast<std::size_t>(rope_dim) +
                        static_cast<std::size_t>(axis * frequencies + index);
                    const float angle = position * inverse;
                    cosine[offset] = std::cos(angle);
                    sine[offset] = std::sin(angle);
                }
            }
        }
    }
    const auto cos_value = array(
        cosine.begin(), Shape{height * width, 1, rope_dim});
    const auto sin_value = array(
        sine.begin(), Shape{height * width, 1, rope_dim});
    auto pieces = mlx::core::split(
        mlx::core::astype(input, mlx::core::float32), 2, -1);
    auto rotated = mlx::core::concatenate(
        {
            pieces.at(0) * cos_value - pieces.at(1) * sin_value,
            pieces.at(1) * cos_value + pieces.at(0) * sin_value,
        },
        -1);
    return input.dtype() == mlx::core::float32
        ? rotated
        : mlx::core::astype(std::move(rotated), input.dtype());
}

struct VisionBlock {
    MlxRmsNorm norm1;
    Affine qkv;
    Affine output;
    MlxRmsNorm norm2;
    MlxLinear gate_up;
    MlxLinear down;
};

} // namespace

void MlxAxialPatchVisionConfig::validate() const {
    const int selected_hidden = checked_int(hidden, "hidden size");
    const int selected_heads = checked_int(heads, "head count");
    checked_int(layers, "layer count");
    checked_int(intermediate, "intermediate size");
    checked_int(patch_size, "patch size");
    checked_int(downsample_ratio, "downsample ratio");
    checked_int(output_hidden, "output hidden size");
    if (selected_hidden % selected_heads != 0 ||
        (selected_hidden / selected_heads) % 4 != 0 ||
        !std::isfinite(rope_theta) || rope_theta <= 0.0 ||
        !std::isfinite(rms_eps) || rms_eps <= 0.0) {
        throw std::invalid_argument(
            "invalid axial patch vision geometry or numeric config");
    }
}

struct MlxAxialPatchVisionTower::Impl {
    MlxAxialPatchVisionConfig config;
    Affine patch_embedding;
    std::vector<VisionBlock> blocks;
    MlxRmsNorm output_norm;
    Affine aligner_input;
    Affine aligner_output;

    array attention(
        const VisionBlock& block,
        const array& input,
        int height,
        int width) const {
        const int tokens = height * width;
        const int heads = checked_int(config.heads, "head count");
        const int head_dim = checked_int(config.hidden, "hidden size") / heads;
        auto qkv = mlx::core::split(block.qkv(input), 3, -1);
        auto query = apply_rope(
            mlx::core::reshape(qkv.at(0), Shape{tokens, heads, head_dim}),
            height, width, static_cast<float>(config.rope_theta));
        auto key = apply_rope(
            mlx::core::reshape(qkv.at(1), Shape{tokens, heads, head_dim}),
            height, width, static_cast<float>(config.rope_theta));
        auto value = mlx::core::reshape(
            qkv.at(2), Shape{tokens, heads, head_dim});
        query = mlx::core::transpose(
            mlx::core::expand_dims(query, 0), {0, 2, 1, 3});
        key = mlx::core::transpose(
            mlx::core::expand_dims(key, 0), {0, 2, 1, 3});
        value = mlx::core::transpose(
            mlx::core::expand_dims(value, 0), {0, 2, 1, 3});
        auto result = scaled_dot_product_attention(
            query, key, value, false);
        result = mlx::core::reshape(
            mlx::core::transpose(result, {0, 2, 1, 3}),
            Shape{tokens, checked_int(config.hidden, "hidden size")});
        return block.output(result);
    }

    array align(
        const array& input,
        int height,
        int width) const {
        const int ratio = checked_int(
            config.downsample_ratio, "downsample ratio");
        const int padded_h = (height + ratio - 1) / ratio * ratio;
        const int padded_w = (width + ratio - 1) / ratio * ratio;
        const int out_h = padded_h / ratio;
        const int out_w = padded_w / ratio;
        const int padding_index = height * width;
        std::vector<std::int32_t> indices;
        indices.reserve(static_cast<std::size_t>(
            out_h * out_w * ratio * ratio));
        for (int out_row = 0; out_row < out_h; ++out_row) {
            for (int out_column = 0; out_column < out_w; ++out_column) {
                for (int patch_row = 0; patch_row < ratio; ++patch_row) {
                    for (int patch_column = 0;
                         patch_column < ratio;
                         ++patch_column) {
                        const int row = out_row * ratio + patch_row;
                        const int column = out_column * ratio + patch_column;
                        indices.push_back(
                            row < height && column < width
                                ? row * width + column
                                : padding_index);
                    }
                }
            }
        }
        auto padded = mlx::core::concatenate(
            {
                input,
                mlx::core::zeros(
                    Shape{1, checked_int(config.hidden, "hidden size")},
                    input.dtype()),
            },
            0);
        auto gathered = mlx::core::take(
            padded,
            array(
                indices.begin(),
                Shape{out_h * out_w, ratio * ratio}),
            0);
        gathered = mlx::core::reshape(
            mlx::core::transpose(gathered, {0, 2, 1}),
            Shape{
                out_h * out_w,
                checked_int(config.hidden, "hidden size") * ratio * ratio});
        return aligner_output(exact_gelu(aligner_input(gathered)));
    }

    array encode(
        const array& raw_patches,
        int height,
        int width) const {
        if (height <= 0 || width <= 0 ||
            height > std::numeric_limits<int>::max() / width) {
            throw std::invalid_argument(
                "invalid axial patch vision grid");
        }
        const int count = height * width;
        const int patch_size = checked_int(config.patch_size, "patch size");
        const int patch_width = 3 * patch_size * patch_size;
        auto patches = raw_patches;
        if (patches.ndim() == 4) {
            patches = mlx::core::reshape(
                patches, Shape{patches.shape(0), patch_width});
        }
        if (patches.ndim() != 2 || patches.shape(0) != count ||
            patches.shape(1) != patch_width) {
            throw std::invalid_argument(
                "axial patch vision patch geometry mismatch");
        }
        auto hidden = patch_embedding(patches);
        for (const auto& block : blocks) {
            hidden = hidden + attention(
                block, block.norm1(hidden), height, width);
            auto gate_up = mlx::core::split(
                block.gate_up(block.norm2(hidden)), 2, -1);
            hidden = hidden + block.down(
                gate_up.at(0) * mlx::core::sigmoid(gate_up.at(0)) *
                gate_up.at(1));
        }
        return align(output_norm(hidden), height, width);
    }
};

MlxAxialPatchVisionTower MlxAxialPatchVisionTower::load(
    const MfqContainer& model,
    const MlxAxialPatchVisionConfig& config) {
    config.validate();
    const int hidden = checked_int(config.hidden, "hidden size");
    const int intermediate = checked_int(
        config.intermediate, "intermediate size");
    const int patch_size = checked_int(config.patch_size, "patch size");
    const int ratio = checked_int(
        config.downsample_ratio, "downsample ratio");
    auto patch = Affine::load(model, "vision.patch_embedding");
    auto output_norm = MlxRmsNorm(
        dense_array(model, "vision.output_norm.weight"),
        static_cast<float>(config.rms_eps));
    auto aligner_input = Affine::load(model, "vision.aligner.input");
    auto aligner_output = Affine::load(model, "vision.aligner.output");
    if (patch.linear.input_size() != 3 * patch_size * patch_size ||
        patch.linear.output_size() != hidden ||
        output_norm.width() != hidden ||
        aligner_input.linear.input_size() != hidden * ratio * ratio ||
        aligner_input.linear.output_size() != config.output_hidden ||
        aligner_output.linear.input_size() != config.output_hidden ||
        aligner_output.linear.output_size() != config.output_hidden) {
        throw std::runtime_error(
            "axial patch vision top-level tensor geometry mismatch");
    }
    std::vector<VisionBlock> blocks;
    blocks.reserve(static_cast<std::size_t>(config.layers));
    for (int index = 0; index < config.layers; ++index) {
        const auto prefix = "vision.block." + std::to_string(index);
        VisionBlock block{
            MlxRmsNorm(
                dense_array(model, prefix + ".norm1.weight"),
                static_cast<float>(config.rms_eps)),
            Affine::load(model, prefix + ".attention.qkv"),
            Affine::load(model, prefix + ".attention.output"),
            MlxRmsNorm(
                dense_array(model, prefix + ".norm2.weight"),
                static_cast<float>(config.rms_eps)),
            MlxLinear::load(model, prefix + ".mlp.gate_up.weight"),
            MlxLinear::load(model, prefix + ".mlp.down.weight"),
        };
        if (block.norm1.width() != hidden ||
            block.qkv.linear.input_size() != hidden ||
            block.qkv.linear.output_size() != 3 * hidden ||
            block.output.linear.input_size() != hidden ||
            block.output.linear.output_size() != hidden ||
            block.norm2.width() != hidden ||
            block.gate_up.input_size() != hidden ||
            block.gate_up.output_size() != 2 * intermediate ||
            block.down.input_size() != intermediate ||
            block.down.output_size() != hidden) {
            throw std::runtime_error(
                "axial patch vision block geometry mismatch");
        }
        blocks.push_back(std::move(block));
    }
    return MlxAxialPatchVisionTower(std::make_unique<Impl>(Impl{
        config,
        std::move(patch),
        std::move(blocks),
        std::move(output_norm),
        std::move(aligner_input),
        std::move(aligner_output),
    }));
}

MlxAxialPatchVisionTower::MlxAxialPatchVisionTower(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MlxAxialPatchVisionTower::MlxAxialPatchVisionTower(
    MlxAxialPatchVisionTower&&) noexcept = default;

MlxAxialPatchVisionTower& MlxAxialPatchVisionTower::operator=(
    MlxAxialPatchVisionTower&&) noexcept = default;

MlxAxialPatchVisionTower::~MlxAxialPatchVisionTower() = default;

array MlxAxialPatchVisionTower::encode(
    const array& patches,
    int height,
    int width) const {
    return impl_->encode(patches, height, width);
}

const MlxAxialPatchVisionConfig&
MlxAxialPatchVisionTower::config() const noexcept {
    return impl_->config;
}

} // namespace mfq::metal
