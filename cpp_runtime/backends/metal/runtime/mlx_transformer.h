#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxKvCacheSnapshot {
    int batch = 0;
    int heads = 0;
    int maximum_sequence = 0;
    int head_dimension = 0;
    int capacity = 0;
    int position = 0;
    mlx::core::Dtype dtype = mlx::core::float16;
    mlx::core::array key = mlx::core::array(0.0f);
    mlx::core::array value = mlx::core::array(0.0f);

    std::size_t nbytes() const noexcept {
        return key.nbytes() + value.nbytes();
    }
};

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
    const mlx::core::array& weight() const noexcept {
        return weight_;
    }
    float eps() const noexcept {
        return eps_;
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
    float scale = 0.0f,
    const std::optional<mlx::core::array>& mask = std::nullopt);

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

    // Reserve cache rows without materializing an indexed update. Specialized
    // decode primitives use this when the projection/post-processing kernel
    // writes the new K/V row directly into the persistent Metal allocation.
    void reserve_append(int tokens);

    // Discard an appended speculative suffix. Storage is intentionally kept;
    // the next append overwrites the now-invisible rows.
    void trim(int tokens);

    std::pair<mlx::core::array, mlx::core::array> view() const;

    // Session snapshots own a compact copy of the visible prefix. Restoring
    // recreates the original allocation capacity without aliasing the saved
    // arrays, so a resumed decode cannot mutate another session snapshot.
    MlxKvCacheSnapshot snapshot() const;
    void restore_snapshot(const MlxKvCacheSnapshot& snapshot);

    int position() const noexcept {
        return position_;
    }
    int capacity() const noexcept {
        return key_.shape(2);
    }
    const mlx::core::array& key_storage() const noexcept {
        return key_;
    }
    const mlx::core::array& value_storage() const noexcept {
        return value_;
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
