#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

bool is_mx_dtype(std::string_view dtype) noexcept;

class MlxMxWeight {
public:
    static MlxMxWeight from_blob(
        std::string_view dtype,
        const std::vector<std::uint8_t>& blob);

    mlx::core::array matmul(const mlx::core::array& input) const;
    mlx::core::array dequantize(
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array embedding(
        const mlx::core::array& token_ids,
        mlx::core::Dtype dtype = mlx::core::float16) const;

    int bits() const noexcept { return bits_; }
    int input_size() const noexcept { return input_size_; }
    int output_size() const noexcept { return output_size_; }
    std::size_t packed_nbytes() const noexcept;

private:
    MlxMxWeight(
        mlx::core::array values,
        mlx::core::array scales,
        int bits,
        int input_size,
        int output_size);

    mlx::core::array values_;
    mlx::core::array scales_;
    int bits_ = 0;
    int input_size_ = 0;
    int output_size_ = 0;
};

} // namespace mfq::metal
