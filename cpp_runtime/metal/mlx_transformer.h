#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

class MlxRmsNorm {
public:
    MlxRmsNorm(
        mlx::core::array weight,
        float eps = 1e-6f,
        float weight_offset = 0.0f);

    mlx::core::array operator()(
        const mlx::core::array& input) const;

    int width() const noexcept {
        return width_;
    }

private:
    mlx::core::array weight_;
    float eps_;
    int width_;
};

mlx::core::array apply_rope(
    const mlx::core::array& input,
    int rotary_dimension,
    float base,
    int offset);

// Apply rotate-half RoPE using explicit token positions. positions accepts
// [tokens], [1,tokens], or [3,tokens]. Three-axis positions select temporal,
// height, and width frequencies according to sections. With interleaved=true,
// the pair layout is THWTHW... (with each axis bounded by its section size);
// otherwise it is the contiguous T...H...W... layout.
mlx::core::array apply_rope(
    const mlx::core::array& input,
    const mlx::core::array& positions,
    int rotary_dimension,
    float base,
    const std::vector<std::int64_t>& sections = {},
    bool interleaved = false);

mlx::core::array scaled_dot_product_attention(
    const mlx::core::array& query,
    const mlx::core::array& key,
    const mlx::core::array& value,
    bool causal = true,
    float scale = 0.0f);

class MlxKvCache {
public:
    MlxKvCache(
        int batch,
        int heads,
        int maximum_sequence,
        int head_dimension,
        int initial_capacity = 16,
        mlx::core::Dtype dtype = mlx::core::float16);

    void reset() noexcept {
        position_ = 0;
    }

    void materialize();

    std::pair<mlx::core::array, mlx::core::array> append(
        const mlx::core::array& key,
        const mlx::core::array& value);

    std::pair<mlx::core::array, mlx::core::array> view() const;

    int position() const noexcept {
        return position_;
    }
    int capacity() const noexcept {
        return key_.shape(2);
    }

private:
    void ensure_capacity(int required);

    int batch_;
    int heads_;
    int maximum_sequence_;
    int head_dimension_;
    mlx::core::Dtype dtype_;
    mlx::core::array key_;
    mlx::core::array value_;
    int position_ = 0;
};

} // namespace mfq::metal
