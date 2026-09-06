#include "mlx_tensor_schema.h"

#include <stdexcept>

namespace mfq::metal {

std::string grid_vision_canonical_name(GridVisionTensorRole role,
                                       std::size_t block_index) {
  const auto block = [&block_index](std::string_view suffix) {
    return "vision.block." + std::to_string(block_index) + std::string(suffix);
  };
  switch (role) {
  case GridVisionTensorRole::patch_weight:
    return "vision.patch_embedding.weight";
  case GridVisionTensorRole::patch_bias:
    return "vision.patch_embedding.bias";
  case GridVisionTensorRole::position_weight:
    return "vision.position_embedding.weight";
  case GridVisionTensorRole::block_norm1_weight:
    return block(".norm1.weight");
  case GridVisionTensorRole::block_norm1_bias:
    return block(".norm1.bias");
  case GridVisionTensorRole::block_attention_qkv_weight:
    return block(".attention.qkv.weight");
  case GridVisionTensorRole::block_attention_qkv_bias:
    return block(".attention.qkv.bias");
  case GridVisionTensorRole::block_attention_output_weight:
    return block(".attention.output.weight");
  case GridVisionTensorRole::block_attention_output_bias:
    return block(".attention.output.bias");
  case GridVisionTensorRole::block_norm2_weight:
    return block(".norm2.weight");
  case GridVisionTensorRole::block_norm2_bias:
    return block(".norm2.bias");
  case GridVisionTensorRole::block_mlp_up_weight:
    return block(".mlp.up.weight");
  case GridVisionTensorRole::block_mlp_up_bias:
    return block(".mlp.up.bias");
  case GridVisionTensorRole::block_mlp_down_weight:
    return block(".mlp.down.weight");
  case GridVisionTensorRole::block_mlp_down_bias:
    return block(".mlp.down.bias");
  case GridVisionTensorRole::merger_norm_weight:
    return "vision.merger.norm.weight";
  case GridVisionTensorRole::merger_norm_bias:
    return "vision.merger.norm.bias";
  case GridVisionTensorRole::merger_mlp_up_weight:
    return "vision.merger.mlp.up.weight";
  case GridVisionTensorRole::merger_mlp_up_bias:
    return "vision.merger.mlp.up.bias";
  case GridVisionTensorRole::merger_mlp_down_weight:
    return "vision.merger.mlp.down.weight";
  case GridVisionTensorRole::merger_mlp_down_bias:
    return "vision.merger.mlp.down.bias";
  }
  throw std::invalid_argument("unknown grid-ViT tensor role");
}

std::string
CanonicalGridVisionTensorSchema::resolve(GridVisionTensorRole role,
                                         std::size_t block_index) const {
  return grid_vision_canonical_name(role, block_index);
}

const GridVisionTensorSchema &canonical_grid_vision_tensor_schema() {
  static const CanonicalGridVisionTensorSchema schema;
  return schema;
}

} // namespace mfq::metal
