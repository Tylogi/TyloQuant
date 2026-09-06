#pragma once

#include "mfq_container.h"
#include "mlx_tpq.h"
#include "mlx_grouped_linear.h"
#include "mlx_mx.h"
#include "mlx_nint.h"
#include "mlx_nint8_zero.h"
#include "mlx_vq.h"

#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// Experimental memory-for-bandwidth mode. When enabled before model loading,
// ordinary packed linear and embedding tensors are materialized once as FP16
// and their packed runtime objects are released. NINTM expert containers keep
// their dedicated representation.
void set_mlx_predequantize_fp16(bool enabled) noexcept;
bool mlx_predequantize_fp16_enabled() noexcept;

mlx::core::array load_dense_array(
    const std::string& dtype,
    std::span<const std::uint8_t> blob);

class MlxLinear {
public:
    static MlxLinear load(
        const MfqContainer& model,
        const std::string& name);

    explicit MlxLinear(MlxNintWeight weight);
    explicit MlxLinear(MlxNint8ZeroWeight weight);
    explicit MlxLinear(MlxVqWeight weight);
    explicit MlxLinear(MlxTpqInt4Weight weight);
    explicit MlxLinear(MlxTpqPqWeight weight);
    explicit MlxLinear(MlxMxWeight weight);
    explicit MlxLinear(mlx::core::array weight);

    mlx::core::array operator()(const mlx::core::array& input) const;

    // Returns a token id when the packed layout has a fused single-row
    // LM-head/greedy implementation; otherwise returns std::nullopt.
    std::optional<mlx::core::array> greedy_argmax(
        const mlx::core::array& input) const;

    // DeepSeek-V4 O-LoRA layout:
    // input [...,groups,K] -> [...,groups,OUT/groups].
    // TPQ-I4G64 uses its dedicated Metal kernel; every other supported
    // linear format takes the exact packed/dense fallback without changing
    // model semantics.
    mlx::core::array grouped_row_matmul(
        const mlx::core::array& input,
        int group_count) const;

    int input_size() const noexcept {
        return input_size_;
    }
    int output_size() const noexcept {
        return output_size_;
    }
    bool packed() const noexcept {
        return !std::holds_alternative<mlx::core::array>(weight_);
    }
    std::optional<MlxGroupedLinearWeightRef>
    grouped_weight_ref() const noexcept;
    const MlxNintWeight* nint_weight_ref() const noexcept;
    const MlxNint8ZeroWeight* nint8_zero_weight_ref() const noexcept;
    const MlxMxWeight* mx_weight_ref() const noexcept;
    const mlx::core::array* dense_weight_ref() const noexcept;

    void materialize_fp16();

private:
    std::variant<
        MlxNintWeight,
        MlxNint8ZeroWeight,
        MlxVqWeight,
        MlxTpqInt4Weight,
        MlxTpqPqWeight,
        MlxMxWeight,
        mlx::core::array> weight_;
    int input_size_ = 0;
    int output_size_ = 0;
};

class MlxEmbedding {
public:
    static MlxEmbedding load(
        const MfqContainer& model,
        const std::string& name);

    explicit MlxEmbedding(MlxNintWeight weight);
    explicit MlxEmbedding(MlxNint8ZeroWeight weight);
    explicit MlxEmbedding(MlxVqWeight weight);
    explicit MlxEmbedding(MlxTpqInt4Weight weight);
    explicit MlxEmbedding(MlxMxWeight weight);
    explicit MlxEmbedding(mlx::core::array weight);

    mlx::core::array operator()(
        const mlx::core::array& token_ids,
        mlx::core::Dtype dtype = mlx::core::float16) const;

    // Apply the embedding table as an LM-head projection. This executes
    // directly against this embedding's dense or packed storage, so tied
    // embeddings do not require a second weight object or dequantized copy.
    mlx::core::array project(
        const mlx::core::array& input) const;

    int vocabulary_size() const noexcept {
        return vocabulary_size_;
    }
    int hidden_size() const noexcept {
        return hidden_size_;
    }

    void materialize_fp16();

private:
    std::variant<
        MlxNintWeight,
        MlxNint8ZeroWeight,
        MlxVqWeight,
        MlxTpqInt4Weight,
        MlxMxWeight,
        mlx::core::array> weight_;
    int vocabulary_size_ = 0;
    int hidden_size_ = 0;
};

} // namespace mfq::metal
