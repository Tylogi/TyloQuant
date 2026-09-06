#include "mlx_multimodal.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace mfq::metal {

mlx::core::array replace_multimodal_embedding_span(
    const mlx::core::array& embeddings,
    const mlx::core::array& replacement,
    int batch,
    int begin,
    int end) {
    if (embeddings.ndim() != 3 || replacement.ndim() != 3 ||
        batch < 0 || batch >= embeddings.shape(0) || begin < 0 ||
        end <= begin || end > embeddings.shape(1) ||
        replacement.shape(0) != 1 ||
        replacement.shape(1) != end - begin ||
        replacement.shape(2) != embeddings.shape(2)) {
        throw std::invalid_argument(
            "multimodal embedding replacement span is incompatible");
    }
    auto value = replacement;
    if (value.dtype() != embeddings.dtype()) {
        value = mlx::core::astype(value, embeddings.dtype());
    }
    return mlx::core::slice_update(
        embeddings,
        value,
        mlx::core::Shape{batch, begin, 0},
        mlx::core::Shape{
            batch + 1, end, embeddings.shape(2)});
}

mlx::core::array replace_multimodal_token_embeddings(
    const mlx::core::array& embeddings,
    const mlx::core::array& replacement,
    const std::vector<std::int64_t>& token_ids,
    const std::vector<std::int64_t>& placeholder_ids) {
    if (embeddings.ndim() != 3 || embeddings.shape(0) != 1 ||
        embeddings.shape(1) != static_cast<int>(token_ids.size()) ||
        replacement.ndim() != 2 ||
        replacement.shape(1) != embeddings.shape(2) ||
        placeholder_ids.empty()) {
        throw std::invalid_argument(
            "multimodal token-embedding geometry is incompatible");
    }
    const std::unordered_set<std::int64_t> placeholders(
        placeholder_ids.begin(), placeholder_ids.end());
    auto output = embeddings;
    int consumed = 0;
    std::size_t begin = 0;
    while (begin < token_ids.size()) {
        if (!placeholders.contains(token_ids[begin])) {
            ++begin;
            continue;
        }
        std::size_t end = begin + 1;
        while (end < token_ids.size() &&
               placeholders.contains(token_ids[end])) {
            ++end;
        }
        const int count = static_cast<int>(end - begin);
        if (count > replacement.shape(0) - consumed) {
            throw std::invalid_argument(
                "multimodal placeholders exceed replacement embeddings");
        }
        auto values = mlx::core::reshape(
            mlx::core::slice(
                replacement,
                mlx::core::Shape{consumed, 0},
                mlx::core::Shape{consumed + count, replacement.shape(1)}),
            mlx::core::Shape{1, count, replacement.shape(1)});
        output = replace_multimodal_embedding_span(
            output,
            values,
            0,
            static_cast<int>(begin),
            static_cast<int>(end));
        consumed += count;
        begin = end;
    }
    if (consumed != replacement.shape(0)) {
        throw std::invalid_argument(
            "replacement embeddings exceed multimodal placeholders");
    }
    return output;
}

namespace {

void validate_grid(
    const MlxGridShape& grid,
    int merge,
    bool image) {
    if (grid.temporal <= 0 || grid.height <= 0 || grid.width <= 0 ||
        grid.height % merge != 0 || grid.width % merge != 0 ||
        (image && grid.temporal != 1)) {
        throw std::invalid_argument(
            "multimodal grid is incompatible with the position policy");
    }
}

} // namespace

MlxGridMropePositions build_grid_mrope_positions(
    const std::vector<std::int64_t>& token_ids,
    std::int64_t image_token_id,
    std::int64_t video_token_id,
    int spatial_merge_size,
    const std::vector<MlxGridShape>& image_grids,
    const std::vector<MlxGridShape>& video_grids) {
    if (token_ids.empty() || spatial_merge_size <= 0 ||
        image_token_id < 0 || video_token_id < 0 ||
        image_token_id == video_token_id) {
        throw std::invalid_argument(
            "invalid multimodal position-policy configuration");
    }
    for (const auto& grid : image_grids) {
        validate_grid(grid, spatial_merge_size, true);
    }
    std::vector<MlxGridShape> video_frames;
    for (const auto& grid : video_grids) {
        validate_grid(grid, spatial_merge_size, false);
        video_frames.insert(
            video_frames.end(),
            static_cast<std::size_t>(grid.temporal),
            MlxGridShape{1, grid.height, grid.width});
    }

    if (token_ids.size() > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::invalid_argument("multimodal prompt is too long");
    }
    std::vector<std::int32_t> positions(3 * token_ids.size());
    std::size_t image = 0;
    std::size_t video = 0;
    std::size_t begin = 0;
    int current = 0;
    while (begin < token_ids.size()) {
        const int modality = token_ids[begin] == image_token_id
            ? 1
            : (token_ids[begin] == video_token_id ? 2 : 0);
        std::size_t end = begin + 1;
        while (end < token_ids.size()) {
            const int next = token_ids[end] == image_token_id
                ? 1
                : (token_ids[end] == video_token_id ? 2 : 0);
            if (next != modality) break;
            ++end;
        }
        const auto count = end - begin;
        if (modality == 0) {
            for (std::size_t index = 0; index < count; ++index) {
                const auto value = static_cast<std::int32_t>(current + index);
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    positions[axis * token_ids.size() + begin + index] = value;
                }
            }
            current += static_cast<int>(count);
        } else {
            const auto& grids = modality == 1 ? image_grids : video_frames;
            auto& selected = modality == 1 ? image : video;
            if (selected >= grids.size()) {
                throw std::invalid_argument(
                    "prompt contains more multimodal spans than grids");
            }
            const auto& grid = grids[selected++];
            const int rows = grid.height / spatial_merge_size;
            const int columns = grid.width / spatial_merge_size;
            const auto expected = static_cast<std::size_t>(
                grid.temporal * rows * columns);
            if (count != expected) {
                throw std::invalid_argument(
                    "multimodal placeholder span disagrees with its grid");
            }
            std::size_t index = 0;
            for (int temporal = 0; temporal < grid.temporal; ++temporal) {
                for (int row = 0; row < rows; ++row) {
                    for (int column = 0; column < columns; ++column, ++index) {
                        positions[begin + index] = current + temporal;
                        positions[token_ids.size() + begin + index] = current + row;
                        positions[2 * token_ids.size() + begin + index] =
                            current + column;
                    }
                }
            }
            current += std::max(rows, columns);
        }
        begin = end;
    }
    if (image != image_grids.size() || video != video_frames.size()) {
        throw std::invalid_argument(
            "multimodal grids contain unused image/video entries");
    }
    const auto maximum = *std::max_element(positions.begin(), positions.end());
    return {
        mlx::core::array(
            positions.begin(),
            mlx::core::Shape{3, static_cast<int>(token_ids.size())},
            mlx::core::int32),
        static_cast<int>(maximum) + 1 - static_cast<int>(token_ids.size()),
    };
}

} // namespace mfq::metal
