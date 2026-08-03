#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

bool is_nint_dtype(std::string_view dtype) noexcept;

class MlxNintWeight {
public:
    static MlxNintWeight from_blob(
        std::span<const std::uint8_t> blob);

    mlx::core::array matmul(const mlx::core::array& input) const;
    bool can_fuse_swiglu(
        const MlxNintWeight& up) const noexcept;
    mlx::core::array swiglu(
        const MlxNintWeight& up,
        const mlx::core::array& input) const;
    mlx::core::array embedding(
        const mlx::core::array& token_ids,
        mlx::core::Dtype dtype = mlx::core::float16) const;

    int bits() const noexcept {
        return bits_;
    }
    int group_size() const noexcept {
        return group_size_;
    }
    int groups() const noexcept {
        return groups_;
    }
    int input_size() const noexcept {
        return input_size_;
    }
    int output_size() const noexcept {
        return output_size_;
    }
    std::size_t packed_nbytes() const noexcept;

    // Read-only packed storage views used by fused/grouped Metal kernels.
    // The arrays remain owned by this weight.
    const mlx::core::array& packed_values() const noexcept {
        return q_packed_;
    }
    const mlx::core::array& sub_scales() const noexcept {
        return sub_scale_;
    }
    const mlx::core::array& sub_mins() const noexcept {
        return sub_min_;
    }
    const mlx::core::array& neuron_scales() const noexcept {
        return neuron_scale_;
    }
    const mlx::core::array& neuron_mins() const noexcept {
        return neuron_min_;
    }
    bool q5_execution_layout() const noexcept {
        return q5_execution_layout_;
    }

private:
    MlxNintWeight(
        mlx::core::array q_packed,
        mlx::core::array sub_scale,
        mlx::core::array sub_min,
        mlx::core::array neuron_scale,
        mlx::core::array neuron_min,
        int bits,
        int group_size,
        int groups,
        int input_size,
        int output_size,
        bool q5_execution_layout);

    mlx::core::array q_packed_;
    mlx::core::array sub_scale_;
    mlx::core::array sub_min_;
    mlx::core::array neuron_scale_;
    mlx::core::array neuron_min_;
    int bits_ = 0;
    int group_size_ = 0;
    int groups_ = 0;
    int input_size_ = 0;
    int output_size_ = 0;
    bool q5_execution_layout_ = false;
};

} // namespace mfq::metal
