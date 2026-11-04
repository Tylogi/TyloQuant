#pragma once

#include "core/block.h"
#include "core/vision_block.h"
#include "models/nemotron_h/vision.h"
#include "storage/weight_loader.h"

#include "cuda_execution.h"
#include "mfq/runtime.h"
#include "mfq_cuda_activation_ops.h"
#include "mfq_model_graph.h"
#include "mfq_paged_prefix_cache.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda::nemotron_h {

using Tensor = mfq_tensor_backend::Tensor;
using mfq::models::nemotron_h::RadioPositionInterpolation;
using mfq::models::nemotron_h::RadioVisionConfig;

// RADIO patch generator + encoder: normalized row-major patches of one image
// in, the tower's patch features out (cls/register prefix removed, exactly like
// `RadioModel.forward`'s `features`).
//
// The tower is a plain pre-LayerNorm ViT: biased q/k/v stored as one fused
// tensor, full self-attention over the patch grid, no positional rotation, no
// QK norm, and LayerScale folded away because released checkpoints carry no
// layer_scale parameter.
class RadioEncoder {
  public:
    static RadioEncoder load(CudaExecutionContext &execution, const mfq::ModelSource &source,
                             const RadioVisionConfig &config) {
        config.validate();
        RadioEncoder result;
        result.patch_projection = load_quant_linear(
            execution, source, "vision.patch_embedding.weight");
        result.position_embedding =
            load_dense_native_gpu(execution, source, "vision.position_embedding.weight");
        result.cls_register_token =
            load_dense_native_gpu(execution, source, "vision.cls_register_token.weight");
        if (!result.patch_projection.is_dense()) {
            throw std::runtime_error(
                "Nemotron-H RADIO patch projection must stay dense: the tower is exported "
                "in BF16");
        }
        if (result.patch_projection.out() != config.hidden_size ||
            result.patch_projection.neuron_len() != config.patch_width()) {
            throw std::runtime_error(
                "Nemotron-H RADIO patch projection is not [hidden, patch_width]");
        }
        const auto side = config.position_side();
        if (result.position_embedding.sizes().vec() !=
            std::vector<int64_t>{1, side * side, config.hidden_size}) {
            throw std::runtime_error(
                "Nemotron-H RADIO position embedding does not match max_img_size");
        }
        if (result.cls_register_token.sizes().vec() !=
            std::vector<int64_t>{config.summary_tokens(), config.hidden_size}) {
            throw std::runtime_error(
                "Nemotron-H RADIO cls/register table disagrees with its config");
        }
        result.layers.reserve(static_cast<std::size_t>(config.depth));
        for (std::int64_t index = 0; index < config.depth; ++index) {
            result.layers.push_back(load_layer(execution, source, config, index));
        }
        result.config_ = config;
        return result;
    }

    // `patches` is [rows * columns, patch_width] row-major normalized pixels.
    Tensor encode(CudaExecutionContext &execution, Tensor patches, std::int64_t rows,
                  std::int64_t columns) const {
        const auto& config = config_;
        if (rows <= 0 || columns <= 0) {
            throw std::invalid_argument("Nemotron-H RADIO image patch grid must have positive dimensions");
        }
        const auto count = rows * columns;
        if (patches.dim() != 2 || patches.size(0) != count ||
            patches.size(1) != config.patch_width()) {
            throw std::invalid_argument(
                "Nemotron-H RADIO pixel patches disagree with the image grid");
        }
        const auto dtype = layer_dtype();
        auto hidden =
            mfq_linear(patches.to(dtype), patch_projection.dense, std::nullopt)
                .to(dtype)
                .contiguous();
        hidden = hidden + interpolated_positions(rows, columns).to(dtype);
        auto prefix = cls_register_token.to(dtype)
                          .unsqueeze(0)
                          .expand({1, config.summary_tokens(), config.hidden_size})
                          .contiguous();
        hidden = mfq_tensor_backend::cat({prefix, hidden.unsqueeze(0)}, 1).contiguous();
        for (const auto& block : layers) {
            hidden = block.forward(execution, std::move(hidden));
        }
        return hidden.narrow(1, config.summary_tokens(), count).contiguous();
    }

    const RadioVisionConfig &config() const noexcept { return config_; }

  private:
    static VisionBlock load_layer(CudaExecutionContext &execution,
                                  const mfq::ModelSource &source,
                                  const RadioVisionConfig &config, std::int64_t index) {
        const std::string prefix = "vision.block." + std::to_string(index);
        const auto fused = BiasLinear::load(execution, source, prefix + ".attention.qkv");
        const auto split = split_fused_qkv(fused, config.hidden_size);
        VisionBlock block;
        block.query = split[0];
        block.key = split[1];
        block.value = split[2];
        block.output = BiasLinear::load(execution, source, prefix + ".attention.output");
        block.norm1_weight = load_dense_native_gpu(execution, source, prefix + ".norm1.weight");
        block.norm1_bias = load_dense_native_gpu(execution, source, prefix + ".norm1.bias");
        block.norm2_weight = load_dense_native_gpu(execution, source, prefix + ".norm2.weight");
        block.norm2_bias = load_dense_native_gpu(execution, source, prefix + ".norm2.bias");
        block.fc1 = BiasLinear::load(execution, source, prefix + ".mlp.up");
        block.fc2 = BiasLinear::load(execution, source, prefix + ".mlp.down");
        block.heads = config.num_heads;
        block.head_dim = config.head_dim();
        block.layer_norm_eps = config.layer_norm_eps;
        // RADIO's `hidden_act: "gelu"` is the exact erf form, not SigLIP's tanh.
        block.activation = VisionActivation::gelu_erf;
        const bool geometry_ok =
            block.output.weight.out() == config.hidden_size &&
            block.output.weight.neuron_len() == config.hidden_size &&
            block.fc1.weight.out() == config.intermediate_size &&
            block.fc1.weight.neuron_len() == config.hidden_size &&
            block.fc2.weight.out() == config.hidden_size &&
            block.fc2.weight.neuron_len() == config.intermediate_size;
        if (!geometry_ok) {
            throw std::runtime_error("Nemotron-H RADIO block " + std::to_string(index) +
                                     " geometry disagrees with its config");
        }
        return block;
    }

    // Cropped-position embedding: bilinear interpolation of the square table
    // onto this image's patch grid, gathered with the four-tap index/weight
    // table so no host-to-device reshape is needed.
    Tensor interpolated_positions(std::int64_t rows, std::int64_t columns) const {
        const auto interpolation = mfq::models::nemotron_h::make_radio_position_interpolation(
            static_cast<std::int32_t>(rows), static_cast<std::int32_t>(columns),
            static_cast<std::int32_t>(config_.position_side()));
        const auto table = position_embedding.reshape(
            {config_.position_side() * config_.position_side(), config_.hidden_size});
        auto index =
            mfq_tensor_backend::tensor(interpolation.indices,
                                       mfq_tensor_backend::TensorOptions()
                                           .dtype(mfq_tensor_backend::kInt32)
                                           .device(mfq_tensor_backend::kCPU))
                .to(table.device())
                .to(mfq_tensor_backend::kInt64);
        auto weights =
            mfq_tensor_backend::tensor(interpolation.weights,
                                       mfq_tensor_backend::TensorOptions()
                                           .dtype(mfq_tensor_backend::kFloat32)
                                           .device(mfq_tensor_backend::kCPU))
                .reshape({interpolation.patch_count, 4, 1})
                .to(table.device());
        return mfq_tensor_backend::sum(
                   table.index_select(0, index.reshape({-1}))
                           .reshape({interpolation.patch_count, 4, config_.hidden_size}) *
                       weights,
                   1)
            .contiguous();
    }

    mfq_tensor_backend::ScalarType layer_dtype() const {
        return position_embedding.scalar_type();
    }

    QuantLinear patch_projection;
    Tensor position_embedding;
    Tensor cls_register_token;
    std::vector<VisionBlock> layers;
    RadioVisionConfig config_;
};

// Vision-to-language projector: `vision_final_layernorm` -> pixel shuffle
// (downsample 0.5, `ps_version v2`) -> `mlp1` (RMSNorm, up, relu-squared, down).
class RadioProjector {
  public:
    static RadioProjector load(CudaExecutionContext &execution, const mfq::ModelSource &source,
                               const RadioVisionConfig &config) {
        RadioProjector result;
        result.norm_weight =
            load_dense_native_gpu(execution, source, "vision.projector.norm.weight");
        result.norm_bias =
            load_dense_native_gpu(execution, source, "vision.projector.norm.bias");
        result.mlp_norm =
            load_dense_native_gpu(execution, source, "vision.projector.mlp.norm.weight");
        result.up = BiasLinear::load(execution, source, "vision.projector.mlp.up", false);
        result.down = BiasLinear::load(execution, source, "vision.projector.mlp.down", false);
        if (result.norm_weight.numel() != config.hidden_size ||
            result.norm_bias.numel() != config.hidden_size ||
            result.mlp_norm.numel() != config.shuffle_width() ||
            result.up.weight.neuron_len() != config.shuffle_width() ||
            result.up.weight.out() != config.projector_intermediate_size ||
            result.down.weight.neuron_len() != config.projector_intermediate_size ||
            result.down.weight.out() != config.text_hidden_size) {
            throw std::runtime_error(
                "Nemotron-H RADIO projector geometry disagrees with its config");
        }
        result.config_ = config;
        return result;
    }

    // `features` is [1, rows * columns, hidden]; the result is
    // [rows * columns / 4, text_hidden_size] in pixel-shuffle token order.
    Tensor project(CudaExecutionContext &execution, Tensor features, std::int64_t rows,
                   std::int64_t columns) const {
        const auto& config = config_;
        const auto stride = config.downsample_factor;
        if (rows <= 0 || columns <= 0 || rows % stride || columns % stride) {
            throw std::runtime_error(
                "Nemotron-H RADIO patch grid does not divide the pixel shuffle factor");
        }
        const auto patches = rows * columns;
        if (features.dim() != 3 || features.size(0) != 1 || features.size(1) != patches ||
            features.size(2) != config.hidden_size) {
            throw std::runtime_error("Nemotron-H RADIO tower features have an invalid shape");
        }
        const auto dtype = features.scalar_type();
        auto hidden =
            vision_layer_norm(features, norm_weight, norm_bias, config.layer_norm_eps);
        // Pixel shuffle with `ps_version v2`: regroup each 2x2 patch block into
        // one token (4 * hidden features) in row-pair / column-pair order.  The
        // v1 variant skips the final permute and transposes the token grid, so
        // the trailing transpose is not redundant.
        hidden = hidden.reshape({1, rows, columns / stride, stride * config.hidden_size});
        hidden = hidden.transpose(1, 2).contiguous();
        hidden = hidden.reshape(
            {1, columns / stride, rows / stride, stride * stride * config.hidden_size});
        hidden = hidden.transpose(1, 2).contiguous();
        hidden = hidden.reshape({1, patches / (stride * stride), config.shuffle_width()});
        auto normalized = qwen_rms_norm(hidden.to(mfq_tensor_backend::kFloat32).reshape(
                                            {-1, config.shuffle_width()}),
                                        mlp_norm.to(mfq_tensor_backend::kFloat32),
                                        config.projector_norm_eps, 0.0)
                              .reshape(hidden.sizes())
                              .to(dtype);
        auto projected = up.forward(execution, std::move(normalized));
        projected = relu_squared_cuda(projected.contiguous());
        return down.forward(execution, std::move(projected))
            .reshape({patches / (stride * stride), config.text_hidden_size})
            .contiguous();
    }

  private:
    Tensor norm_weight;
    Tensor norm_bias;
    Tensor mlp_norm;
    BiasLinear up;
    BiasLinear down;
    RadioVisionConfig config_;
};

// RADIO tower plus projector, i.e. everything between normalized pixels and
// the language model's placeholder rows.
class RadioVisionTower {
  public:
    static RadioVisionTower load(CudaExecutionContext &execution,
                                 const mfq::ModelSource &source,
                                 const RadioVisionConfig &config) {
        return RadioVisionTower(RadioEncoder::load(execution, source, config),
                                RadioProjector::load(execution, source, config), config);
    }

    Tensor embed(CudaExecutionContext &execution, Tensor patches, std::int64_t rows,
                 std::int64_t columns) const {
        auto features = encoder_.encode(execution, std::move(patches), rows, columns);
        return projector_.project(execution, std::move(features), rows, columns);
    }

    std::int64_t tokens(std::int64_t rows, std::int64_t columns) const {
        const auto stride = config_.downsample_factor;
        return rows * columns / (stride * stride);
    }

    const RadioVisionConfig &config() const noexcept { return config_; }

  private:
    RadioVisionTower(RadioEncoder encoder, RadioProjector projector,
                     RadioVisionConfig config)
        : encoder_(std::move(encoder)), projector_(std::move(projector)),
          config_(std::move(config)) {}

    RadioEncoder encoder_;
    RadioProjector projector_;
    RadioVisionConfig config_;
};

// One image's patch grid from the media payload.  Rows and columns are patch
// counts before the pixel shuffle.
struct RadioImageGrid {
    std::int64_t rows = 0;
    std::int64_t columns = 0;
};

// RADIO prompt component: encodes every image of a media payload and rewrites
// the `<image>` placeholder rows of the prompt embeddings.  Nemotron-H has no
// multimodal position policy (the text backbone is NoPE), so the returned
// positions are plain token offsets and only address the text cache.
class CudaRadioPromptComponent {
  public:
    static CudaRadioPromptComponent load(CudaExecutionContext &execution,
                                         const mfq::ModelSource &source,
                                         const RadioVisionConfig &config,
                                         std::string input_contract,
                                         std::string position_policy) {
        if (input_contract != mfq::kMfqRadioCpeInputContract) {
            throw std::invalid_argument("unsupported RADIO vision input contract: " +
                                        input_contract);
        }
        if (!position_policy.empty()) {
            throw std::invalid_argument("RADIO vision declares no position policy, got: " +
                                        position_policy);
        }
        if (config.image_token_id < 0) {
            throw std::invalid_argument("Nemotron-H RADIO needs the `<image>` placeholder id");
        }
        return CudaRadioPromptComponent(RadioVisionTower::load(execution, source, config), config,
                                        std::move(input_contract));
    }

    template <typename Model>
    mfq::StepSequence<CudaPreparedPrompt> prepare(Model &language,
                                                  const std::vector<std::int64_t> &token_ids,
                                                  const MfqMultimodalInput &media) const {
        const auto grids = validated_grids(media);
        const auto runs = placeholder_runs(token_ids, grids);
        const auto cache_key = media_cache_key(media, grids);
        auto pixels = mfq_tensor_backend::from_blob(
                          const_cast<float *>(media.pixel_values.data()), media.pixel_shape,
                          mfq_tensor_backend::TensorOptions()
                              .dtype(mfq_tensor_backend::kFloat32)
                              .device(mfq_tensor_backend::kCPU))
                          .clone()
                          .to(mfq_tensor_backend::kCUDA);
        co_yield mfq::StepState::advanced;

        auto ids = mfq_tensor_backend::tensor(
                       token_ids, mfq_tensor_backend::TensorOptions()
                                      .dtype(mfq_tensor_backend::kInt64)
                                      .device(mfq_tensor_backend::kCPU))
                       .reshape({1, static_cast<std::int64_t>(token_ids.size())})
                       .to(mfq_tensor_backend::kCUDA);
        auto embeddings = language.embed_forward(ids).contiguous();
        std::int64_t patch_offset = 0;
        std::size_t run_index = 0;
        for (const auto& grid : grids) {
            const auto count = grid.rows * grid.columns;
            const auto expected = tower_.tokens(grid.rows, grid.columns);
            const auto& run = runs[run_index];
            if (static_cast<std::int64_t>(run.size()) != expected) {
                throw std::invalid_argument(
                    "Nemotron-H image placeholders disagree with the projected patch count");
            }
            auto projected =
                tower_.embed(*language.execution, pixels.narrow(0, patch_offset, count),
                             grid.rows, grid.columns);
            auto index = mfq_tensor_backend::tensor(
                             run, mfq_tensor_backend::TensorOptions()
                                      .dtype(mfq_tensor_backend::kInt64)
                                      .device(mfq_tensor_backend::kCPU))
                             .to(mfq_tensor_backend::kCUDA);
            embeddings.index_copy_(1, index,
                                   projected.to(embeddings.scalar_type()).unsqueeze(0));
            patch_offset += count;
            ++run_index;
            co_yield mfq::StepState::advanced;
        }
        auto positions = mfq_tensor_backend::arange(
                             static_cast<std::int64_t>(token_ids.size()),
                             mfq_tensor_backend::TensorOptions()
                                 .dtype(mfq_tensor_backend::kInt64)
                                 .device(mfq_tensor_backend::kCUDA))
                             .reshape({1, static_cast<std::int64_t>(token_ids.size())});
        co_yield CudaPreparedPrompt{token_ids, std::move(embeddings), std::move(positions), 0,
                                    cache_key};
    }

    const std::string &input_contract() const noexcept { return input_contract_; }
    const RadioVisionConfig &config() const noexcept { return config_; }

  private:
    CudaRadioPromptComponent(RadioVisionTower tower, RadioVisionConfig config,
                             std::string input_contract)
        : tower_(std::move(tower)), config_(std::move(config)),
          input_contract_(std::move(input_contract)) {}

    // Geometry and contract checks for one media payload: images only, one
    // `<image>` run per image, and row-major patches that divide the shuffle.
    std::vector<RadioImageGrid> validated_grids(const MfqMultimodalInput &media) const {
        const auto stride = config_.downsample_factor;
        if (media.processor != MfqMultimodalProcessor::grid_vision ||
            media.processor_name != input_contract_ || !media.video_grid.empty() ||
            media.vision_grid_shape.size() != 2 || media.vision_grid_shape[1] != 3 ||
            media.pixel_shape.size() != 2 || media.pixel_shape[0] <= 0 ||
            media.pixel_shape[1] != config_.patch_width() ||
            media.image_grid != media.vision_grid ||
            media.image_grid_shape != media.vision_grid_shape) {
            throw std::invalid_argument("Nemotron-H RADIO media geometry is invalid");
        }
        const auto images = media.vision_grid_shape[0];
        if (images <= 0 || media.vision_types.size() != static_cast<std::size_t>(images)) {
            throw std::invalid_argument("Nemotron-H RADIO media needs one grid per image");
        }
        std::vector<RadioImageGrid> grids;
        grids.reserve(static_cast<std::size_t>(images));
        std::int64_t patches = 0;
        for (std::int64_t image = 0; image < images; ++image) {
            if (media.vision_types[static_cast<std::size_t>(image)] != 1) {
                throw std::invalid_argument(
                    "Nemotron-H RADIO accepts images only; video is not supported");
            }
            const auto offset = static_cast<std::size_t>(3 * image);
            const auto temporal = media.vision_grid[offset];
            const RadioImageGrid grid{media.vision_grid[offset + 1],
                                      media.vision_grid[offset + 2]};
            if (temporal != 1 || grid.rows <= 0 || grid.columns <= 0 ||
                grid.rows % stride || grid.columns % stride) {
                throw std::invalid_argument(
                    "Nemotron-H RADIO patch grid must be a positive multiple of the pixel "
                    "shuffle factor");
            }
            patches += grid.rows * grid.columns;
            grids.push_back(grid);
        }
        if (patches != media.pixel_shape[0]) {
            throw std::invalid_argument(
                "Nemotron-H RADIO patch count disagrees with pixel_values");
        }
        return grids;
    }

    // Token positions of every `<image>` placeholder, grouped per image in media
    // order.  A run must be contiguous: the processor emits `<img>` + N*N_token
    // + `</img>` for each image, so a gap means the prompt and the payload
    // disagree about how many images were sent.
    std::vector<std::vector<std::int64_t>> placeholder_runs(
            const std::vector<std::int64_t> &token_ids,
            const std::vector<RadioImageGrid> &grids) const {
        std::vector<std::int64_t> positions;
        for (std::size_t index = 0; index < token_ids.size(); ++index) {
            if (token_ids[index] == config_.image_token_id) {
                positions.push_back(static_cast<std::int64_t>(index));
            }
        }
        std::vector<std::vector<std::int64_t>> runs;
        runs.reserve(grids.size());
        std::size_t offset = 0;
        for (const auto& grid : grids) {
            const auto count = static_cast<std::size_t>(tower_.tokens(grid.rows, grid.columns));
            if (offset + count > positions.size()) {
                throw std::invalid_argument(
                    "Nemotron-H prompt holds fewer `<image>` placeholders than the images "
                    "require");
            }
            std::vector<std::int64_t> run(positions.begin() + static_cast<std::ptrdiff_t>(offset),
                                          positions.begin() +
                                              static_cast<std::ptrdiff_t>(offset + count));
            for (std::size_t index = 1; index < run.size(); ++index) {
                if (run[index] != run[0] + static_cast<std::int64_t>(index)) {
                    throw std::invalid_argument(
                        "Nemotron-H `<image>` placeholders of one image must be contiguous");
                }
            }
            runs.push_back(std::move(run));
            offset += count;
        }
        if (offset != positions.size()) {
            throw std::invalid_argument(
                "Nemotron-H prompt holds more `<image>` placeholders than the images require");
        }
        return runs;
    }

    std::string media_cache_key(const MfqMultimodalInput &media,
                                const std::vector<RadioImageGrid> &grids) const {
        std::string key = input_contract_ + '|';
        key += mfq::cache::block_hash_hex(mfq::cache::sha256(
            media.pixel_values.data(), media.pixel_values.size() * sizeof(float)));
        for (const auto& grid : grids) {
            key.push_back('|');
            key += std::to_string(grid.rows);
            key.push_back('x');
            key += std::to_string(grid.columns);
        }
        return key;
    }

    RadioVisionTower tower_;
    RadioVisionConfig config_;
    std::string input_contract_;
};

} // namespace mfq::cuda::nemotron_h
