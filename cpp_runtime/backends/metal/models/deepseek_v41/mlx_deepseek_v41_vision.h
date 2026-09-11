#pragma once

#include "deepseek_v41_model.h"
#include "mlx_axial_patch_vision.h"
#include "mlx_tensor.h"

#include <cstdint>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxDeepseekV41ImageInput {
    mlx::core::array patches;
    int n_vit_h = 0;
    int n_vit_w = 0;
    int begin = 0;
    int end = 0;
    // Official values: 0=start, 1=image row, 2=newline, 3=end.
    std::vector<std::int64_t> token_types;
};

// V4.1-specific prompt adapter around the reusable axial patch ViT. It owns
// only the model's learned image sentinels and span layout contract.
class MlxDeepseekV41Vision {
public:
    static MlxDeepseekV41Vision load(
        const MfqContainer& model,
        const DeepseekV41Config& config);

    MlxDeepseekV41Vision(
        DeepseekV41Config config,
        MlxAxialPatchVisionTower tower,
        mlx::core::array special_embeddings);

    mlx::core::array encode(
        const mlx::core::array& patches,
        int n_vit_h,
        int n_vit_w) const;

    mlx::core::array embed_prompt(
        const std::vector<std::int64_t>& token_ids,
        const std::vector<MlxDeepseekV41ImageInput>& images,
        const MlxEmbedding& embedding,
        mlx::core::Dtype dtype = mlx::core::float16) const;

    mlx::core::array image_mask(
        int token_count,
        const std::vector<MlxDeepseekV41ImageInput>& images) const;

private:
    DeepseekV41Config config_;
    MlxAxialPatchVisionTower tower_;
    // [4,hidden], with a zero placeholder at row 1 for dynamic ViT rows.
    mlx::core::array special_embeddings_;
};

} // namespace mfq::metal
