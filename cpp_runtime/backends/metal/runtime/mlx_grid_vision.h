#pragma once

#include "mfq_container.h"
#include "mlx_multimodal.h"
#include "mlx_tensor_schema.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct GridVisionConfig {
  std::int64_t hidden_size = 0;
  std::int64_t intermediate_size = 0;
  std::int64_t depth = 0;
  std::int64_t num_heads = 0;
  std::int64_t in_channels = 3;
  std::int64_t patch_size = 0;
  std::int64_t temporal_patch_size = 0;
  std::int64_t spatial_merge_size = 0;
  std::int64_t out_hidden_size = 0;
  std::int64_t num_position_embeddings = 0;
  double rope_theta = 10'000.0;
  double layer_norm_eps = 1e-6;

  void validate() const;
  std::int64_t head_dim() const noexcept {
    return num_heads > 0 ? hidden_size / num_heads : 0;
  }
  std::int64_t patch_width() const noexcept {
    return in_channels * temporal_patch_size * patch_size * patch_size;
  }
};

using GridThw = MlxGridShape;

// Host-side layout shared by all block-major grid ViTs. positions is flattened
// [patches,2] row/column data; one segment length is emitted for every frame.
struct GridVisionLayout {
  std::vector<std::int32_t> positions;
  std::vector<std::int32_t> segment_lengths;
  std::int64_t patch_count = 0;
};

struct LearnedPositionInterpolation {
  std::vector<std::int32_t> indices;
  std::vector<float> weights;
  std::int64_t patch_count = 0;
};

GridVisionLayout make_grid_vision_layout(const std::vector<GridThw> &grids,
                                         std::int32_t spatial_merge_size);

LearnedPositionInterpolation
make_learned_position_interpolation(const std::vector<GridThw> &grids,
                                    std::int32_t spatial_merge_size,
                                    std::int32_t position_side);

// Reusable learned-position, axial-RoPE grid ViT encoder. Its default loader
// consumes only the canonical MFQ namespace from mlx_tensor_schema.h.
class MlxGridVisionEncoder {
public:
  static MlxGridVisionEncoder load(const MfqContainer &model,
                                   const GridVisionConfig &config);

  MlxGridVisionEncoder(MlxGridVisionEncoder &&) noexcept;
  MlxGridVisionEncoder &operator=(MlxGridVisionEncoder &&) noexcept;
  ~MlxGridVisionEncoder();

  MlxGridVisionEncoder(const MlxGridVisionEncoder &) = delete;
  MlxGridVisionEncoder &operator=(const MlxGridVisionEncoder &) = delete;

  mlx::core::array encode(const mlx::core::array &pixel_values,
                          const std::vector<GridThw> &grid_thw) const;
  mlx::core::array encode(const MlxGridMediaInput &media) const;

  const GridVisionConfig &config() const noexcept;

private:
  static MlxGridVisionEncoder
  load_with_schema(const MfqContainer &model, const GridVisionConfig &config,
                   const GridVisionTensorSchema &schema);
  struct Impl;
  explicit MlxGridVisionEncoder(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

struct QwenVisionOutput {
  mlx::core::array patch_features;
  mlx::core::array merged_features;
};

// Qwen's shared tower is a generic grid-ViT encoder followed by the learned
// patch merger: LayerNorm -> flatten merge^2 patches -> GELU -> projection.
class MlxQwenVisionTower {
public:
  static MlxQwenVisionTower load(const MfqContainer &model,
                                 const GridVisionConfig &config);

  MlxQwenVisionTower(MlxQwenVisionTower &&) noexcept;
  MlxQwenVisionTower &operator=(MlxQwenVisionTower &&) noexcept;
  ~MlxQwenVisionTower();

  MlxQwenVisionTower(const MlxQwenVisionTower &) = delete;
  MlxQwenVisionTower &operator=(const MlxQwenVisionTower &) = delete;

  QwenVisionOutput forward(const mlx::core::array &pixel_values,
                           const std::vector<GridThw> &grid_thw) const;
  QwenVisionOutput forward(const MlxGridMediaInput &media) const;

  const GridVisionConfig &config() const noexcept;

private:
  struct Impl;
  explicit MlxQwenVisionTower(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// A graph-selected prompt component. It owns the media tower and translates
// the shared grid_vision.v1 contract into the shared PreparedPrompt contract;
// the language model remains unaware of image/video preprocessing details.
class MlxGridVisionPromptComponent {
public:
  static MlxGridVisionPromptComponent load(
      const MfqContainer &model, const GridVisionConfig &config,
      std::int64_t image_token_id, std::int64_t video_token_id,
      std::string input_contract, std::string position_policy);

  MlxGridVisionPromptComponent(MlxGridVisionPromptComponent &&) noexcept;
  MlxGridVisionPromptComponent &
  operator=(MlxGridVisionPromptComponent &&) noexcept;
  ~MlxGridVisionPromptComponent();

  MlxGridVisionPromptComponent(const MlxGridVisionPromptComponent &) = delete;
  MlxGridVisionPromptComponent &
  operator=(const MlxGridVisionPromptComponent &) = delete;

  MlxPreparedPrompt prepare(
      const std::vector<std::int64_t> &token_ids,
      const mlx::core::array &text_embeddings,
      const MlxGridMediaInput &media) const;

  const std::string &input_contract() const noexcept;
  const std::string &position_policy() const noexcept;

private:
  struct Impl;
  explicit MlxGridVisionPromptComponent(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace mfq::metal
