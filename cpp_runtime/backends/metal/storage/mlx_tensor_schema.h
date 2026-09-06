#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace mfq::metal {

// Architecture-neutral semantic roles used by the native grid-ViT runtime.
// Persisted MFQ v1 artifacts use the canonical names returned by
// grid_vision_canonical_name(); source-checkpoint spellings belong only in a
// conversion-time or explicitly legacy resolver.
enum class GridVisionTensorRole {
  patch_weight,
  patch_bias,
  position_weight,
  block_norm1_weight,
  block_norm1_bias,
  block_attention_qkv_weight,
  block_attention_qkv_bias,
  block_attention_output_weight,
  block_attention_output_bias,
  block_norm2_weight,
  block_norm2_bias,
  block_mlp_up_weight,
  block_mlp_up_bias,
  block_mlp_down_weight,
  block_mlp_down_bias,
  merger_norm_weight,
  merger_norm_bias,
  merger_mlp_up_weight,
  merger_mlp_up_bias,
  merger_mlp_down_weight,
  merger_mlp_down_bias,
};

std::string grid_vision_canonical_name(GridVisionTensorRole role,
                                       std::size_t block_index = 0);

class GridVisionTensorSchema {
public:
  virtual ~GridVisionTensorSchema() = default;

  virtual std::string resolve(GridVisionTensorRole role,
                              std::size_t block_index = 0) const = 0;
};

class CanonicalGridVisionTensorSchema final : public GridVisionTensorSchema {
public:
  std::string resolve(GridVisionTensorRole role,
                      std::size_t block_index = 0) const override;
};

const GridVisionTensorSchema &canonical_grid_vision_tensor_schema();

} // namespace mfq::metal
