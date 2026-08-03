#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

bool is_vq_dtype(std::string_view dtype) noexcept;

struct VqTensorMetadata {
    std::string format_label;
    std::vector<int> output_shape;
    int input_size = 0;
    int output_size = 0;
    int rotation_block = 0;
    std::uint64_t rotation_seed = 0;
};

// Header-only inspection for loader/config validation. This validates the
// public dtype/profile and logical matrix layout without expanding codebooks
// or allocating MLX arrays. from_blob() performs full payload validation.
VqTensorMetadata inspect_vq_blob(
    std::string_view dtype,
    std::span<const std::uint8_t> blob,
    std::span<const std::uint8_t> runtime_payload = {});

// Native packed NVQ/NPQ/NEPQ execution weight.
//
// Indices, group states, signs, and delta selectors remain bit packed.
// Only the small shared codebook/table payloads are expanded to an int8
// Metal lookup layout. Rotated NEPQ cohorts consume the HSG1 runtime payload
// stored alongside their tensor payload in a NINTM v2 pool.
class MlxVqWeight {
public:
    static MlxVqWeight from_blob(
        std::string_view dtype,
        std::span<const std::uint8_t> blob,
        std::span<const std::uint8_t> runtime_payload = {});

    mlx::core::array dequantize(
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array embedding(
        const mlx::core::array& token_ids,
        mlx::core::Dtype dtype = mlx::core::float16) const;

    mlx::core::array gemv(const mlx::core::array& input) const;
    mlx::core::array mmq(const mlx::core::array& input) const;
    mlx::core::array gemm(const mlx::core::array& input) const;
    mlx::core::array matmul(const mlx::core::array& input) const;
    mlx::core::array operator()(const mlx::core::array& input) const {
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
    const std::vector<int>& output_shape() const noexcept {
        return output_shape_;
    }
    int group_size() const noexcept {
        return group_size_;
    }
    int groups() const noexcept {
        return groups_;
    }
    int vector_size() const noexcept {
        return vector_size_;
    }
    int vectors() const noexcept {
        return vectors_;
    }
    int index_bits() const noexcept {
        return index_bits_;
    }
    int state_bits() const noexcept {
        return state_bits_;
    }
    int states() const noexcept {
        return states_;
    }
    int entries() const noexcept {
        return entries_;
    }
    int code_banks() const noexcept {
        return code_banks_;
    }
    int aux_mode() const noexcept {
        return aux_mode_;
    }
    int code_bank_mode() const noexcept {
        return code_bank_mode_;
    }
    int table_banks() const noexcept {
        return table_banks_;
    }
    int groups_per_supergroup() const noexcept {
        return groups_per_supergroup_;
    }
    int supergroups() const noexcept {
        return supergroups_;
    }
    int rotation_block() const noexcept {
        return rotation_block_;
    }
    std::uint64_t rotation_seed() const noexcept {
        return rotation_seed_;
    }
    std::size_t packed_nbytes() const noexcept;

    const mlx::core::array& packed_indices() const noexcept {
        return indices_packed_;
    }
    const mlx::core::array& packed_states() const noexcept {
        return state_packed_;
    }
    const mlx::core::array& packed_auxiliary() const noexcept {
        return aux_packed_;
    }
    const mlx::core::array& anchors() const noexcept {
        return anchors_;
    }
    const mlx::core::array& codebooks() const noexcept {
        return codebooks_;
    }
    const mlx::core::array& scale_lut() const noexcept {
        return scale_lut_;
    }
    const mlx::core::array& state_to_codebank() const noexcept {
        return state_to_codebank_;
    }
    const mlx::core::array& bank_ids() const noexcept {
        return bank_ids_;
    }
    const mlx::core::array& rotation_signs() const noexcept {
        return rotation_signs_;
    }
    const mlx::core::array& parameters() const noexcept {
        return parameters_;
    }

private:
    MlxVqWeight(
        mlx::core::array indices_packed,
        mlx::core::array state_packed,
        mlx::core::array aux_packed,
        mlx::core::array anchors,
        mlx::core::array codebooks,
        mlx::core::array scale_lut,
        mlx::core::array state_to_codebank,
        mlx::core::array bank_ids,
        mlx::core::array rotation_signs,
        mlx::core::array parameters,
        std::string format_label,
        std::vector<int> output_shape,
        int input_size,
        int output_size,
        int group_size,
        int groups,
        int vector_size,
        int vectors,
        int index_bits,
        int state_bits,
        int states,
        int entries,
        int code_banks,
        int aux_mode,
        int code_bank_mode,
        int table_banks,
        int groups_per_supergroup,
        int supergroups,
        int rotation_block,
        std::uint64_t rotation_seed);

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

    mlx::core::array indices_packed_;
    mlx::core::array state_packed_;
    mlx::core::array aux_packed_;
    mlx::core::array anchors_;
    mlx::core::array codebooks_;
    mlx::core::array scale_lut_;
    mlx::core::array state_to_codebank_;
    mlx::core::array bank_ids_;
    mlx::core::array rotation_signs_;
    mlx::core::array parameters_;
    std::string format_label_;
    std::vector<int> output_shape_;
    int input_size_ = 0;
    int output_size_ = 0;
    int group_size_ = 0;
    int groups_ = 0;
    int vector_size_ = 0;
    int vectors_ = 0;
    int index_bits_ = 0;
    int state_bits_ = 0;
    int states_ = 0;
    int entries_ = 0;
    int code_banks_ = 0;
    int aux_mode_ = 0;
    int code_bank_mode_ = 0;
    int table_banks_ = 0;
    int groups_per_supergroup_ = 0;
    int supergroups_ = 0;
    int rotation_block_ = 0;
    std::uint64_t rotation_seed_ = 0;
};

} // namespace mfq::metal
