#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// Backend-neutral prompt representation after optional input components have
// transformed it. Text-only callers fill token_ids alone. Vision/audio
// components may replace embeddings and positions without creating another
// generation loop or another cache implementation.
struct MlxPreparedPrompt {
    std::vector<std::int64_t> token_ids;
    std::optional<mlx::core::array> embeddings;
    std::optional<mlx::core::array> positions;
    // Difference between the next logical RoPE coordinate and the append-only
    // token-cache position. Grid MRoPE commonly makes this negative.
    int decode_position_delta = 0;

    bool transformed() const noexcept {
        return embeddings.has_value() || positions.has_value() ||
            decode_position_delta != 0;
    }
};

struct MlxGridShape {
    int temporal = 0;
    int height = 0;
    int width = 0;

    bool operator==(const MlxGridShape&) const = default;
};

// Architecture-neutral output of a patch-grid media processor. Pixel
// normalization and patchification happen outside the runtime; model adapters
// consume this one contract regardless of whether the source was an image or
// video.
struct MlxGridMediaInput {
    std::string processor;
    mlx::core::array pixel_values = mlx::core::array(0.0f);
    std::vector<MlxGridShape> media_grids;
    std::vector<std::int32_t> media_types;
    std::vector<MlxGridShape> image_grids;
    std::vector<MlxGridShape> video_grids;
};

struct MlxGridMropePositions {
    mlx::core::array values = mlx::core::array(0, mlx::core::int32);
    int decode_delta = 0;
};

// Shared embedding-injection primitive used by every native multimodal
// architecture.  Keeping span validation here prevents each vision/audio
// frontend from growing a subtly different scatter contract.
mlx::core::array replace_multimodal_embedding_span(
    const mlx::core::array& embeddings,
    const mlx::core::array& replacement,
    int batch,
    int begin,
    int end);

// Replace every image/video placeholder in flattened prompt order. This is
// shared by grid-ViT model families; an adapter supplies only its special IDs.
mlx::core::array replace_multimodal_token_embeddings(
    const mlx::core::array& embeddings,
    const mlx::core::array& replacement,
    const std::vector<std::int64_t>& token_ids,
    const std::vector<std::int64_t>& placeholder_ids);

// Shared Qwen-style three-axis grid-position policy. It is a component
// strategy, not an architecture dispatch: every model declaring this policy
// uses the same implementation.
MlxGridMropePositions build_grid_mrope_positions(
    const std::vector<std::int64_t>& token_ids,
    std::int64_t image_token_id,
    std::int64_t video_token_id,
    int spatial_merge_size,
    const std::vector<MlxGridShape>& image_grids,
    const std::vector<MlxGridShape>& video_grids);

} // namespace mfq::metal
