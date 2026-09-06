#include "mlx_grid_vision.h"

#include "mfq_model_graph.h"
#include "mlx_tensor.h"
#include "mlx_transformer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::array;
using mlx::core::Shape;

int checked_int(std::int64_t value, const char *name) {
  if (value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("invalid grid-ViT ") + name);
  }
  return static_cast<int>(value);
}

array dense_tensor(const MfqContainer &model, const std::string &name) {
  const auto &record = model.record(name);
  if (record.dtype != "F16" && record.dtype != "BF16" &&
      record.dtype != "F32") {
    throw std::invalid_argument(
        "grid-ViT dense tensor has unsupported dtype: " + name);
  }
  return mlx::core::contiguous(
      load_dense_array(record.dtype, model.read(name)));
}

array required_vector(const MfqContainer &model, const std::string &name,
                      int width) {
  auto value = dense_tensor(model, name);
  if (value.ndim() != 1 || value.shape(0) != width) {
    throw std::invalid_argument("grid-ViT vector shape mismatch: " + name);
  }
  return value;
}

struct Affine {
  MlxLinear linear;
  std::optional<array> bias;

  static Affine load(const MfqContainer &model,
                     const GridVisionTensorSchema &schema,
                     GridVisionTensorRole weight_role,
                     GridVisionTensorRole bias_role,
                     std::size_t block_index = 0) {
    auto bias_name = schema.resolve(bias_role, block_index);
    std::optional<array> bias;
    if (model.contains(bias_name)) {
      bias = dense_tensor(model, bias_name);
    }
    Affine result{
        MlxLinear::load(model, schema.resolve(weight_role, block_index)),
        std::move(bias),
    };
    if (result.bias && (result.bias->ndim() != 1 ||
                        result.bias->shape(0) != result.linear.output_size())) {
      throw std::invalid_argument("grid-ViT affine bias shape mismatch: " +
                                  bias_name);
    }
    return result;
  }

  array operator()(const array &input) const {
    auto output = linear(input);
    if (bias) {
      output = output + mlx::core::astype(*bias, output.dtype());
    }
    return output;
  }
};

struct LayerNorm {
  array weight;
  array bias;
  float eps = 1e-6f;

  static LayerNorm load(const MfqContainer &model,
                        const GridVisionTensorSchema &schema,
                        GridVisionTensorRole weight_role,
                        GridVisionTensorRole bias_role, int width, float eps,
                        std::size_t block_index = 0) {
    return {
        required_vector(model, schema.resolve(weight_role, block_index), width),
        required_vector(model, schema.resolve(bias_role, block_index), width),
        eps,
    };
  }

  array operator()(const array &input) const {
    if (input.ndim() == 0 || input.shape(-1) != weight.shape(0)) {
      throw std::invalid_argument("grid-ViT LayerNorm input width mismatch");
    }
    const auto dtype = input.dtype();
    auto source = mlx::core::astype(input, mlx::core::float32);
    auto mean = mlx::core::mean(source, -1, true);
    auto centered = source - mean;
    auto variance = mlx::core::mean(centered * centered, -1, true);
    auto normalized = centered * mlx::core::rsqrt(variance + eps);
    auto output = normalized * mlx::core::astype(weight, mlx::core::float32) +
                  mlx::core::astype(bias, mlx::core::float32);
    return mlx::core::astype(output, dtype);
  }
};

array gelu_tanh(const array &input) {
  constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
  const auto dtype = input.dtype();
  auto source = mlx::core::astype(input, mlx::core::float32);
  auto output =
      0.5f * source *
      (1.0f + mlx::core::tanh(kSqrtTwoOverPi *
                              (source + 0.044715f * source * source * source)));
  return mlx::core::astype(output, dtype);
}

array gelu_exact(const array &input) {
  constexpr float kInvSqrtTwo = 0.7071067811865475f;
  const auto dtype = input.dtype();
  auto source = mlx::core::astype(input, mlx::core::float32);
  auto output = 0.5f * source * (1.0f + mlx::core::erf(source * kInvSqrtTwo));
  return mlx::core::astype(output, dtype);
}

array apply_axial_rope(const array &input,
                       const std::vector<std::int32_t> &positions,
                       float theta) {
  if (input.ndim() != 3 || input.shape(0) <= 0 || input.shape(2) <= 0 ||
      input.shape(2) % 4 != 0 ||
      positions.size() != static_cast<std::size_t>(input.shape(0) * 2)) {
    throw std::invalid_argument("grid-ViT axial RoPE geometry mismatch");
  }
  const int tokens = input.shape(0);
  const int head_dim = input.shape(2);
  const int half = head_dim / 2;
  const int frequencies = half / 2;
  std::vector<float> cosine(static_cast<std::size_t>(tokens * head_dim));
  std::vector<float> sine(cosine.size());
  for (int token = 0; token < tokens; ++token) {
    for (int axis = 0; axis < 2; ++axis) {
      const float position = static_cast<float>(
          positions[static_cast<std::size_t>(token * 2 + axis)]);
      for (int index = 0; index < frequencies; ++index) {
        const float inverse = std::pow(theta, -static_cast<float>(2 * index) /
                                                  static_cast<float>(half));
        const float angle = position * inverse;
        const int frequency_column = axis * frequencies + index;
        for (int repeat = 0; repeat < 2; ++repeat) {
          const auto offset = static_cast<std::size_t>(
              token * head_dim + repeat * half + frequency_column);
          cosine[offset] = std::cos(angle);
          sine[offset] = std::sin(angle);
        }
      }
    }
  }
  auto cosine_array = array(cosine.begin(), Shape{tokens, 1, head_dim});
  auto sine_array = array(sine.begin(), Shape{tokens, 1, head_dim});
  auto source = mlx::core::astype(input, mlx::core::float32);
  auto halves = mlx::core::split(source, 2, -1);
  auto rotated = mlx::core::concatenate({-halves.at(1), halves.at(0)}, -1);
  return mlx::core::astype(source * cosine_array + rotated * sine_array,
                           input.dtype());
}

array packed_attention(const array &query, const array &key, const array &value,
                       const std::vector<std::int32_t> &lengths) {
  // Match the shared Python MLX attention contract: BF16 (and non-floating
  // inputs) enter SDPA as FP16, while native FP16/FP32 remain unchanged.
  const auto attention_dtype =
      query.dtype() == mlx::core::float16 || query.dtype() == mlx::core::float32
          ? query.dtype()
          : mlx::core::float16;
  const auto query_source = query.dtype() == attention_dtype
                                ? query
                                : mlx::core::astype(query, attention_dtype);
  const auto key_source = key.dtype() == attention_dtype
                              ? key
                              : mlx::core::astype(key, attention_dtype);
  const auto value_source = value.dtype() == attention_dtype
                                ? value
                                : mlx::core::astype(value, attention_dtype);
  std::vector<array> outputs;
  outputs.reserve(lengths.size());
  int offset = 0;
  for (const auto raw_length : lengths) {
    const int length = checked_int(raw_length, "attention segment length");
    const int end = offset + length;
    const auto slice_segment = [offset, end](const array &source) {
      auto segment =
          mlx::core::slice(source, Shape{offset, 0, 0},
                           Shape{end, source.shape(1), source.shape(2)});
      return mlx::core::expand_dims(mlx::core::transpose(segment, {1, 0, 2}),
                                    0);
    };
    auto attended = scaled_dot_product_attention(
        slice_segment(query_source), slice_segment(key_source),
        slice_segment(value_source), false);
    outputs.push_back(
        mlx::core::transpose(mlx::core::squeeze(attended, 0), {1, 0, 2}));
    offset = end;
  }
  if (offset != query_source.shape(0) || outputs.empty()) {
    throw std::invalid_argument(
        "grid-ViT attention segments do not cover every patch");
  }
  return outputs.size() == 1 ? outputs.front()
                             : mlx::core::concatenate(outputs, 0);
}

struct VisionAttention {
  Affine qkv;
  Affine output;
  int hidden_size = 0;
  int heads = 0;
  int head_dim = 0;
  float rope_theta = 10'000.0f;

  array operator()(const array &input, const GridVisionLayout &layout) const {
    const int tokens = input.shape(0);
    auto projected =
        mlx::core::reshape(qkv(input), Shape{tokens, 3, heads, head_dim});
    auto pieces = mlx::core::split(projected, 3, 1);
    auto query = mlx::core::squeeze(pieces.at(0), 1);
    auto key = mlx::core::squeeze(pieces.at(1), 1);
    auto value = mlx::core::squeeze(pieces.at(2), 1);
    query = apply_axial_rope(query, layout.positions, rope_theta);
    key = apply_axial_rope(key, layout.positions, rope_theta);
    auto attended = packed_attention(query, key, value, layout.segment_lengths);
    return output(mlx::core::reshape(attended, Shape{tokens, hidden_size}));
  }
};

struct VisionBlock {
  LayerNorm norm1;
  VisionAttention attention;
  LayerNorm norm2;
  Affine mlp_up;
  Affine mlp_down;

  array operator()(const array &input, const GridVisionLayout &layout) const {
    auto hidden = input + attention(norm1(input), layout);
    return hidden + mlp_down(gelu_tanh(mlp_up(norm2(hidden))));
  }
};

std::vector<std::pair<std::int32_t, std::int32_t>>
spatial_positions(const GridThw &grid, std::int32_t merge) {
  std::vector<std::pair<std::int32_t, std::int32_t>> result;
  result.reserve(static_cast<std::size_t>(grid.height * grid.width));
  for (std::int32_t block_row = 0; block_row < grid.height / merge;
       ++block_row) {
    for (std::int32_t block_column = 0; block_column < grid.width / merge;
         ++block_column) {
      for (std::int32_t inner_row = 0; inner_row < merge; ++inner_row) {
        for (std::int32_t inner_column = 0; inner_column < merge;
             ++inner_column) {
          result.emplace_back(block_row * merge + inner_row,
                              block_column * merge + inner_column);
        }
      }
    }
  }
  return result;
}

void validate_grid(const GridThw &grid, std::int32_t merge) {
  if (merge <= 0 || grid.temporal <= 0 || grid.height <= 0 || grid.width <= 0 ||
      grid.height % merge != 0 || grid.width % merge != 0) {
    throw std::invalid_argument(
        "grid-ViT THW must be positive and spatially divisible by merge size");
  }
  const auto patches =
      static_cast<std::int64_t>(grid.temporal) * grid.height * grid.width;
  if (patches > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("grid-ViT patch count exceeds native limits");
  }
}

} // namespace

void GridVisionConfig::validate() const {
  const int hidden = checked_int(hidden_size, "hidden size");
  const int heads = checked_int(num_heads, "head count");
  checked_int(intermediate_size, "intermediate size");
  checked_int(depth, "depth");
  checked_int(in_channels, "channel count");
  checked_int(patch_size, "patch size");
  checked_int(temporal_patch_size, "temporal patch size");
  checked_int(spatial_merge_size, "spatial merge size");
  checked_int(out_hidden_size, "output hidden size");
  const int positions =
      checked_int(num_position_embeddings, "position table size");
  const int side = static_cast<int>(std::sqrt(positions));
  if (hidden % heads != 0 || (hidden / heads) % 4 != 0 ||
      side * side != positions || !std::isfinite(rope_theta) ||
      rope_theta <= 0.0 || !std::isfinite(layer_norm_eps) ||
      layer_norm_eps <= 0.0) {
    throw std::invalid_argument("invalid grid-ViT geometry or numeric config");
  }
  checked_int(patch_width(), "flattened patch width");
}

GridVisionLayout make_grid_vision_layout(const std::vector<GridThw> &grids,
                                         std::int32_t spatial_merge_size) {
  if (grids.empty()) {
    throw std::invalid_argument("grid-ViT requires at least one THW item");
  }
  GridVisionLayout result;
  for (const auto &grid : grids) {
    validate_grid(grid, spatial_merge_size);
    const auto spatial = spatial_positions(grid, spatial_merge_size);
    for (std::int32_t frame = 0; frame < grid.temporal; ++frame) {
      result.segment_lengths.push_back(grid.height * grid.width);
      for (const auto [row, column] : spatial) {
        result.positions.push_back(row);
        result.positions.push_back(column);
        ++result.patch_count;
      }
    }
  }
  return result;
}

LearnedPositionInterpolation
make_learned_position_interpolation(const std::vector<GridThw> &grids,
                                    std::int32_t spatial_merge_size,
                                    std::int32_t position_side) {
  if (grids.empty() || position_side <= 0) {
    throw std::invalid_argument(
        "grid-ViT learned-position interpolation geometry is invalid");
  }
  LearnedPositionInterpolation result;
  for (const auto &grid : grids) {
    validate_grid(grid, spatial_merge_size);
    const auto spatial = spatial_positions(grid, spatial_merge_size);
    for (std::int32_t frame = 0; frame < grid.temporal; ++frame) {
      for (const auto [row, column] : spatial) {
        const float row_source =
            static_cast<float>(row) * static_cast<float>(position_side - 1) /
            static_cast<float>(std::max(grid.height - 1, 1));
        const float column_source =
            static_cast<float>(column) * static_cast<float>(position_side - 1) /
            static_cast<float>(std::max(grid.width - 1, 1));
        const auto row_floor =
            static_cast<std::int32_t>(std::floor(row_source));
        const auto column_floor =
            static_cast<std::int32_t>(std::floor(column_source));
        const std::int32_t row_taps[2] = {
            std::clamp(row_floor, 0, position_side - 1),
            std::clamp(row_floor + 1, 0, position_side - 1),
        };
        const std::int32_t column_taps[2] = {
            std::clamp(column_floor, 0, position_side - 1),
            std::clamp(column_floor + 1, 0, position_side - 1),
        };
        const float row_weights[2] = {
            static_cast<float>(row_floor + 1) - row_source,
            row_source - static_cast<float>(row_floor),
        };
        const float column_weights[2] = {
            static_cast<float>(column_floor + 1) - column_source,
            column_source - static_cast<float>(column_floor),
        };
        for (int row_tap = 0; row_tap < 2; ++row_tap) {
          for (int column_tap = 0; column_tap < 2; ++column_tap) {
            result.indices.push_back(row_taps[row_tap] * position_side +
                                     column_taps[column_tap]);
            result.weights.push_back(row_weights[row_tap] *
                                     column_weights[column_tap]);
          }
        }
        ++result.patch_count;
      }
    }
  }
  return result;
}

struct MlxGridVisionEncoder::Impl {
  GridVisionConfig config;
  array patch_weight;
  array patch_bias;
  array position_weight;
  std::vector<VisionBlock> blocks;

  array encode(const array &pixel_values,
               const std::vector<GridThw> &grids) const {
    const auto layout = make_grid_vision_layout(
        grids, checked_int(config.spatial_merge_size, "spatial merge size"));
    const int patches = checked_int(layout.patch_count, "patch count");
    const int patch_width = checked_int(config.patch_width(), "patch width");
    auto input = pixel_values;
    if (input.ndim() == 5) {
      input = mlx::core::reshape(input, Shape{input.shape(0), -1});
    }
    if (input.ndim() != 2 || input.shape(0) != patches ||
        input.shape(1) != patch_width) {
      throw std::invalid_argument(
          "grid-ViT pixel values must be flattened CxTxHxW patches");
    }
    auto flat_weight = mlx::core::reshape(
        patch_weight,
        Shape{checked_int(config.hidden_size, "hidden size"), patch_width});
    auto hidden =
        mlx::core::matmul(mlx::core::astype(input, flat_weight.dtype()),
                          mlx::core::transpose(flat_weight));
    hidden = hidden + mlx::core::astype(patch_bias, hidden.dtype());

    const int side = static_cast<int>(
        std::sqrt(static_cast<double>(config.num_position_embeddings)));
    const auto interpolation = make_learned_position_interpolation(
        grids, checked_int(config.spatial_merge_size, "spatial merge size"),
        side);
    auto indices = array(interpolation.indices.begin(), Shape{patches, 4});
    auto weights = array(interpolation.weights.begin(), Shape{patches, 4, 1});
    auto learned = mlx::core::sum(
        mlx::core::take(position_weight, indices, 0) * weights, 1);
    hidden = hidden + mlx::core::astype(learned, hidden.dtype());
    for (const auto &block : blocks) {
      hidden = block(hidden, layout);
    }
    return hidden;
  }
};

MlxGridVisionEncoder
MlxGridVisionEncoder::load(const MfqContainer &model,
                           const GridVisionConfig &config) {
  return load_with_schema(model, config, canonical_grid_vision_tensor_schema());
}

MlxGridVisionEncoder
MlxGridVisionEncoder::load_with_schema(const MfqContainer &model,
                                       const GridVisionConfig &config,
                                       const GridVisionTensorSchema &schema) {
  config.validate();
  const int hidden = checked_int(config.hidden_size, "hidden size");
  const int intermediate =
      checked_int(config.intermediate_size, "intermediate size");
  auto patch_weight =
      dense_tensor(model, schema.resolve(GridVisionTensorRole::patch_weight));
  const Shape expected_patch = {
      hidden,
      checked_int(config.in_channels, "channel count"),
      checked_int(config.temporal_patch_size, "temporal patch size"),
      checked_int(config.patch_size, "patch size"),
      checked_int(config.patch_size, "patch size"),
  };
  if (patch_weight.shape() != expected_patch) {
    throw std::invalid_argument("grid-ViT patch weight shape mismatch");
  }
  auto patch_bias = required_vector(
      model, schema.resolve(GridVisionTensorRole::patch_bias), hidden);
  auto position_weight = dense_tensor(
      model, schema.resolve(GridVisionTensorRole::position_weight));
  if (position_weight.shape() !=
      Shape{checked_int(config.num_position_embeddings, "position count"),
            hidden}) {
    throw std::invalid_argument(
        "grid-ViT learned position table shape mismatch");
  }

  std::vector<VisionBlock> blocks;
  blocks.reserve(static_cast<std::size_t>(config.depth));
  for (std::size_t index = 0; index < static_cast<std::size_t>(config.depth);
       ++index) {
    VisionBlock block{
        LayerNorm::load(model, schema, GridVisionTensorRole::block_norm1_weight,
                        GridVisionTensorRole::block_norm1_bias, hidden,
                        static_cast<float>(config.layer_norm_eps), index),
        {
            Affine::load(model, schema,
                         GridVisionTensorRole::block_attention_qkv_weight,
                         GridVisionTensorRole::block_attention_qkv_bias, index),
            Affine::load(model, schema,
                         GridVisionTensorRole::block_attention_output_weight,
                         GridVisionTensorRole::block_attention_output_bias,
                         index),
            hidden,
            checked_int(config.num_heads, "head count"),
            checked_int(config.head_dim(), "head dimension"),
            static_cast<float>(config.rope_theta),
        },
        LayerNorm::load(model, schema, GridVisionTensorRole::block_norm2_weight,
                        GridVisionTensorRole::block_norm2_bias, hidden,
                        static_cast<float>(config.layer_norm_eps), index),
        Affine::load(model, schema, GridVisionTensorRole::block_mlp_up_weight,
                     GridVisionTensorRole::block_mlp_up_bias, index),
        Affine::load(model, schema, GridVisionTensorRole::block_mlp_down_weight,
                     GridVisionTensorRole::block_mlp_down_bias, index),
    };
    if (block.attention.qkv.linear.input_size() != hidden ||
        block.attention.qkv.linear.output_size() != 3 * hidden ||
        block.attention.output.linear.input_size() != hidden ||
        block.attention.output.linear.output_size() != hidden ||
        block.mlp_up.linear.input_size() != hidden ||
        block.mlp_up.linear.output_size() != intermediate ||
        block.mlp_down.linear.input_size() != intermediate ||
        block.mlp_down.linear.output_size() != hidden) {
      throw std::invalid_argument(
          "grid-ViT transformer block geometry mismatch");
    }
    blocks.push_back(std::move(block));
  }
  return MlxGridVisionEncoder(std::make_unique<Impl>(Impl{
      config,
      std::move(patch_weight),
      std::move(patch_bias),
      std::move(position_weight),
      std::move(blocks),
  }));
}

MlxGridVisionEncoder::MlxGridVisionEncoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MlxGridVisionEncoder::MlxGridVisionEncoder(MlxGridVisionEncoder &&) noexcept =
    default;

MlxGridVisionEncoder &
MlxGridVisionEncoder::operator=(MlxGridVisionEncoder &&) noexcept = default;

MlxGridVisionEncoder::~MlxGridVisionEncoder() = default;

array MlxGridVisionEncoder::encode(const array &pixel_values,
                                   const std::vector<GridThw> &grid_thw) const {
  return impl_->encode(pixel_values, grid_thw);
}

array MlxGridVisionEncoder::encode(const MlxGridMediaInput &media) const {
  return encode(media.pixel_values, media.media_grids);
}

const GridVisionConfig &MlxGridVisionEncoder::config() const noexcept {
  return impl_->config;
}

struct MlxQwenVisionTower::Impl {
  MlxGridVisionEncoder encoder;
  LayerNorm merger_norm;
  Affine merger_up;
  Affine merger_down;
};

MlxQwenVisionTower MlxQwenVisionTower::load(const MfqContainer &model,
                                            const GridVisionConfig &config) {
  const auto &schema = canonical_grid_vision_tensor_schema();
  config.validate();
  const int hidden = checked_int(config.hidden_size, "hidden size");
  const int unit = checked_int(
      config.spatial_merge_size * config.spatial_merge_size, "merge unit");
  auto merger_norm =
      LayerNorm::load(model, schema, GridVisionTensorRole::merger_norm_weight,
                      GridVisionTensorRole::merger_norm_bias, hidden,
                      static_cast<float>(config.layer_norm_eps));
  auto merger_up =
      Affine::load(model, schema, GridVisionTensorRole::merger_mlp_up_weight,
                   GridVisionTensorRole::merger_mlp_up_bias);
  auto merger_down =
      Affine::load(model, schema, GridVisionTensorRole::merger_mlp_down_weight,
                   GridVisionTensorRole::merger_mlp_down_bias);
  if (merger_up.linear.input_size() != unit * hidden ||
      merger_down.linear.input_size() != merger_up.linear.output_size() ||
      merger_down.linear.output_size() != config.out_hidden_size) {
    throw std::invalid_argument("Qwen vision merger geometry mismatch");
  }
  return MlxQwenVisionTower(std::make_unique<Impl>(Impl{
      MlxGridVisionEncoder::load(model, config),
      std::move(merger_norm),
      std::move(merger_up),
      std::move(merger_down),
  }));
}

MlxQwenVisionTower::MlxQwenVisionTower(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MlxQwenVisionTower::MlxQwenVisionTower(MlxQwenVisionTower &&) noexcept =
    default;

MlxQwenVisionTower &
MlxQwenVisionTower::operator=(MlxQwenVisionTower &&) noexcept = default;

MlxQwenVisionTower::~MlxQwenVisionTower() = default;

QwenVisionOutput
MlxQwenVisionTower::forward(const array &pixel_values,
                            const std::vector<GridThw> &grid_thw) const {
  auto patches = impl_->encoder.encode(pixel_values, grid_thw);
  const int unit = checked_int(
      config().spatial_merge_size * config().spatial_merge_size, "merge unit");
  if (patches.shape(0) % unit != 0) {
    throw std::invalid_argument(
        "Qwen vision patch count is not divisible by merge unit");
  }
  auto normalized = mlx::core::reshape(
      impl_->merger_norm(patches),
      Shape{patches.shape(0) / unit,
            unit * checked_int(config().hidden_size, "hidden size")});
  auto merged = impl_->merger_down(gelu_exact(impl_->merger_up(normalized)));
  return {std::move(patches), std::move(merged)};
}

QwenVisionOutput
MlxQwenVisionTower::forward(const MlxGridMediaInput &media) const {
  return forward(media.pixel_values, media.media_grids);
}

const GridVisionConfig &MlxQwenVisionTower::config() const noexcept {
  return impl_->encoder.config();
}

struct MlxGridVisionPromptComponent::Impl {
  MlxQwenVisionTower tower;
  std::int64_t image_token_id = -1;
  std::int64_t video_token_id = -1;
  std::string input_contract;
  std::string position_policy;
};

MlxGridVisionPromptComponent MlxGridVisionPromptComponent::load(
    const MfqContainer &model, const GridVisionConfig &config,
    std::int64_t image_token_id, std::int64_t video_token_id,
    std::string input_contract, std::string position_policy) {
  if (input_contract != mfq::kMfqGridVisionInputContract) {
    throw std::invalid_argument(
        "unsupported grid-Vision input contract: " + input_contract);
  }
  if (position_policy != mfq::kMfqGridMropePositionPolicy) {
    throw std::invalid_argument(
        "unsupported grid-Vision position policy: " + position_policy);
  }
  if (image_token_id < 0 || video_token_id < 0 ||
      image_token_id == video_token_id) {
    throw std::invalid_argument("invalid grid-Vision placeholder token IDs");
  }
  return MlxGridVisionPromptComponent(std::make_unique<Impl>(Impl{
      MlxQwenVisionTower::load(model, config), image_token_id, video_token_id,
      std::move(input_contract), std::move(position_policy)}));
}

MlxGridVisionPromptComponent::MlxGridVisionPromptComponent(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MlxGridVisionPromptComponent::MlxGridVisionPromptComponent(
    MlxGridVisionPromptComponent &&) noexcept = default;

MlxGridVisionPromptComponent &MlxGridVisionPromptComponent::operator=(
    MlxGridVisionPromptComponent &&) noexcept = default;

MlxGridVisionPromptComponent::~MlxGridVisionPromptComponent() = default;

MlxPreparedPrompt MlxGridVisionPromptComponent::prepare(
    const std::vector<std::int64_t> &token_ids,
    const array &text_embeddings,
    const MlxGridMediaInput &media) const {
  if (media.processor != impl_->input_contract) {
    throw std::invalid_argument(
        "grid-Vision media processor disagrees with model graph");
  }
  if (media.media_grids.empty() ||
      media.media_types.size() != media.media_grids.size()) {
    throw std::invalid_argument("grid-Vision media inventory is invalid");
  }
  std::vector<MlxGridShape> images;
  std::vector<MlxGridShape> videos;
  for (std::size_t index = 0; index < media.media_grids.size(); ++index) {
    if (media.media_types[index] == 1) {
      images.push_back(media.media_grids[index]);
    } else if (media.media_types[index] == 2) {
      videos.push_back(media.media_grids[index]);
    } else {
      throw std::invalid_argument("grid-Vision media type is invalid");
    }
  }
  if ((!media.image_grids.empty() && media.image_grids != images) ||
      (!media.video_grids.empty() && media.video_grids != videos)) {
    throw std::invalid_argument(
        "grid-Vision combined and split media inventories disagree");
  }
  auto visual = impl_->tower.forward(media);
  auto embeddings = replace_multimodal_token_embeddings(
      text_embeddings, visual.merged_features, token_ids,
      {impl_->image_token_id, impl_->video_token_id});
  auto positions = build_grid_mrope_positions(
      token_ids, impl_->image_token_id, impl_->video_token_id,
      static_cast<int>(impl_->tower.config().spatial_merge_size), images,
      videos);
  return {
      token_ids,
      std::move(embeddings),
      std::move(positions.values),
      positions.decode_delta,
  };
}

const std::string &MlxGridVisionPromptComponent::input_contract() const noexcept {
  return impl_->input_contract;
}

const std::string &MlxGridVisionPromptComponent::position_policy() const noexcept {
  return impl_->position_policy;
}

} // namespace mfq::metal
