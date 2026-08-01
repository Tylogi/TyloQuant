#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

bool is_tpq_dtype(std::string_view dtype) noexcept;
bool is_cccp_dtype(std::string_view dtype) noexcept;

// Native CCCP symmetric int4-g64 matrix.
//
// The serialized uint8 nibble stream and FP16 group scales stay packed on
// Metal. Small-M calls decode in the projection kernel; matmul() switches to
// a temporary dense MLX GEMM at M >= 64.
class MlxCccpInt4Weight {
public:
    static MlxCccpInt4Weight from_blob(
        const std::vector<std::uint8_t>& blob);

    mlx::core::array dequantize(
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array embedding(
        const mlx::core::array& token_ids,
        mlx::core::Dtype dtype = mlx::core::float16) const;

    mlx::core::array gemv(const mlx::core::array& input) const;
    mlx::core::array mmq(const mlx::core::array& input) const;
    mlx::core::array gemm(const mlx::core::array& input) const;
    mlx::core::array matmul(const mlx::core::array& input) const;
    mlx::core::array operator()(
        const mlx::core::array& input) const {
        return matmul(input);
    }

    // DeepSeek-V4 O-LoRA layout:
    // input [..., group_count, K] -> [..., group_count, OUT/group_count].
    mlx::core::array grouped_row_matmul(
        const mlx::core::array& input,
        int group_count) const;

    int input_size() const noexcept {
        return input_size_;
    }
    int output_size() const noexcept {
        return output_size_;
    }
    int group_size() const noexcept {
        return group_size_;
    }
    int groups() const noexcept {
        return groups_;
    }
    std::size_t packed_nbytes() const noexcept;

    const mlx::core::array& packed_values() const noexcept {
        return packed_;
    }
    const mlx::core::array& scales() const noexcept {
        return scales_;
    }

private:
    MlxCccpInt4Weight(
        mlx::core::array packed,
        mlx::core::array scales,
        int input_size,
        int output_size,
        int group_size,
        int groups);

    mlx::core::array prepare_input(
        const mlx::core::array& input,
        mlx::core::Shape& prefix,
        int& rows) const;
    mlx::core::array packed_matmul(
        const mlx::core::array& source,
        const mlx::core::Shape& prefix,
        int rows,
        int tile_rows) const;
    mlx::core::array reshape_output(
        mlx::core::array value,
        const mlx::core::Shape& prefix) const;

    mlx::core::array packed_;
    mlx::core::array scales_;
    int input_size_ = 0;
    int output_size_ = 0;
    int group_size_ = 0;
    int groups_ = 0;
};

// Native CCCP learned product-vector matrix.
//
// CCCP-X/W/V/VV use fixed 8/4-component code vectors and 256/4096-entry
// codebooks. Index storage may be 8, 12, 14, or 16 bits as permitted by the
// public tier. The index bitstream remains packed; the shared codebook is
// retained as FP16, matching the production MLX runtime.
class MlxCccpPqWeight {
public:
    static MlxCccpPqWeight from_blob(
        std::string_view dtype,
        const std::vector<std::uint8_t>& blob);

    mlx::core::array dequantize(
        mlx::core::Dtype dtype = mlx::core::float16) const;

    mlx::core::array gemv(const mlx::core::array& input) const;
    mlx::core::array mmq(const mlx::core::array& input) const;
    mlx::core::array gemm(const mlx::core::array& input) const;
    mlx::core::array matmul(const mlx::core::array& input) const;
    mlx::core::array operator()(
        const mlx::core::array& input) const {
        return matmul(input);
    }

    const std::string& format_label() const noexcept {
        return format_label_;
    }
    int input_size() const noexcept {
        return input_size_;
    }
    int output_size() const noexcept {
        return output_size_;
    }
    int vector_size() const noexcept {
        return vector_size_;
    }
    int blocks() const noexcept {
        return blocks_;
    }
    int entries() const noexcept {
        return entries_;
    }
    int index_bits() const noexcept {
        return index_bits_;
    }
    std::size_t packed_nbytes() const noexcept;

    const mlx::core::array& packed_indices() const noexcept {
        return indices_;
    }
    const mlx::core::array& codebook() const noexcept {
        return codebook_;
    }

private:
    MlxCccpPqWeight(
        mlx::core::array indices,
        mlx::core::array codebook,
        std::string format_label,
        int input_size,
        int output_size,
        int vector_size,
        int blocks,
        int entries,
        int index_bits);

    mlx::core::array prepare_input(
        const mlx::core::array& input,
        mlx::core::Shape& prefix,
        int& rows) const;
    mlx::core::array packed_matmul(
        const mlx::core::array& source,
        const mlx::core::Shape& prefix,
        int rows,
        int tile_rows) const;
    mlx::core::array reshape_output(
        mlx::core::array value,
        const mlx::core::Shape& prefix) const;

    mlx::core::array indices_;
    mlx::core::array codebook_;
    std::string format_label_;
    int input_size_ = 0;
    int output_size_ = 0;
    int vector_size_ = 0;
    int blocks_ = 0;
    int entries_ = 0;
    int index_bits_ = 0;
};

// Canonical public names. The implementation class names remain aliases so
// existing C++ callers and old CCCP-labelled artifacts keep working.
using MlxTpqInt4Weight = MlxCccpInt4Weight;
using MlxTpqPqWeight = MlxCccpPqWeight;

} // namespace mfq::metal
