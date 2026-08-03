#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

bool is_nint8_zero_dtype(std::string_view dtype) noexcept;

class MlxNint8ZeroWeight {
public:
    static MlxNint8ZeroWeight from_blob(
        std::span<const std::uint8_t> blob);

    mlx::core::array matmul(const mlx::core::array& input) const;
    // DeepSeek-V4 O-LoRA diagonal grouped projection:
    // input [..., group_count, K] ->
    // output [..., group_count, OUT/group_count].
    // Each output block consumes only the matching input group.
    mlx::core::array grouped_row_matmul(
        const mlx::core::array& input,
        int group_count) const;
    // DeepSeek-V4 decode specialization. The attention output is still in
    // RoPE space; apply the inverse adjacent-pair RoPE while consuming the
    // grouped input so no standalone de-rotation tensor is materialized.
    mlx::core::array grouped_row_matmul_inverse_rope(
        const mlx::core::array& input,
        int group_count,
        const mlx::core::array& cosine,
        const mlx::core::array& sine,
        int head_dimension,
        int rotary_dimension) const;
    mlx::core::array embedding(
        const mlx::core::array& token_ids,
        mlx::core::Dtype dtype = mlx::core::float16) const;

    int input_size() const noexcept {
        return input_size_;
    }
    int output_size() const noexcept {
        return output_size_;
    }
    int groups() const noexcept {
        return groups_;
    }
    std::size_t packed_nbytes() const noexcept;

    // Read-only packed storage views used by fused/grouped Metal kernels.
    // The arrays remain owned by this weight.
    const mlx::core::array& quantized_values() const noexcept {
        return q_;
    }
    const mlx::core::array& scales() const noexcept {
        return scales_;
    }

private:
    MlxNint8ZeroWeight(
        mlx::core::array q,
        mlx::core::array scales,
        int input_size,
        int output_size,
        int groups);

    mlx::core::array q_;
    mlx::core::array scales_;
    int input_size_ = 0;
    int output_size_ = 0;
    int groups_ = 0;
};

} // namespace mfq::metal
