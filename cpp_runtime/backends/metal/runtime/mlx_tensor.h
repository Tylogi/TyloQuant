#pragma once

#include "mfq_container.h"
#include "mlx_grouped_linear.h"
#include "mlx_fp8_sq.h"
#include "mlx_mx.h"
#include "mlx_mxfp4_sq.h"
#include "mlx_nint.h"
#include "mlx_nint8_zero.h"
#include "mlx_vq.h"
#include "mlx_tq.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// Opt-in memory-for-bandwidth mode. When enabled before model loading,
// ordinary packed linear and embedding tensors are materialized once as FP16
// and their packed runtime objects are released. MFE expert containers keep
// their dedicated representation.
void set_mlx_predequantize_fp16(bool enabled) noexcept;
bool mlx_predequantize_fp16_enabled() noexcept;

class MlxFp16WeightScope {
public:
    explicit MlxFp16WeightScope(bool enabled) noexcept;
    ~MlxFp16WeightScope();
    MlxFp16WeightScope(const MlxFp16WeightScope&) = delete;
    MlxFp16WeightScope& operator=(const MlxFp16WeightScope&) = delete;
private:
    bool enabled_;
};

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
    explicit MlxLinear(MlxFp8SqWeight weight);
    explicit MlxLinear(MlxMxWeight weight);
    explicit MlxLinear(MlxMxfp4SqWeight weight);
    explicit MlxLinear(MlxTqWeight weight);
    explicit MlxLinear(mlx::core::array weight);

    mlx::core::array operator()(const mlx::core::array& input) const;

    // Project one shared-expert activation and combine it with routed MoE
    // rows. NINT uses a register-resident projection/reduce/gate epilogue;
    // other formats retain the exact composed implementation.
    mlx::core::array moe_shared(
        const mlx::core::array& input,
        const mlx::core::array& routed_pairs,
        const mlx::core::array& route_weights,
        const mlx::core::array& gate_logits) const;

    // Returns a token id when the packed layout has a fused single-row
    // LM-head/greedy implementation; otherwise returns std::nullopt.
    std::optional<mlx::core::array> greedy_argmax(
        const mlx::core::array& input) const;

    // Diagonal grouped-projection layout:
    // input [...,groups,K] -> [...,groups,OUT/groups].
    mlx::core::array grouped_row_matmul(
        const mlx::core::array& input,
        int group_count) const;
    void prepare_grouped_row_matmul(int group_count) const;

    // Diagonal grouped projection whose input is still in adjacent-pair RoPE
    // space: value [...,heads,head_dim] -> output [...,groups,OUT/groups].
    // Packed formats use their fused inverse-RoPE kernel when its geometry is
    // supported; every other format follows the same exact composed fallback.
    mlx::core::array grouped_row_matmul_inverse_rope(
        const mlx::core::array& value,
        int group_count,
        const mlx::core::array& cosine,
        const mlx::core::array& sine,
        int head_dimension,
        int rotary_dimension) const;

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
    const MlxTqWeight* tq_weight_ref() const noexcept;
    const mlx::core::array* dense_weight_ref() const noexcept;

    void materialize_fp16();

private:
    std::variant<
        MlxNintWeight,
        MlxNint8ZeroWeight,
        MlxVqWeight,
        MlxFp8SqWeight,
        MlxMxWeight,
        MlxMxfp4SqWeight,
        MlxTqWeight,
        mlx::core::array> weight_;
    int input_size_ = 0;
    int output_size_ = 0;
};

// Register independent projections over one activation with the backend-wide
// grouped coordinator. Model adapters only identify the participating
// tensors; format eligibility and kernel selection remain runtime-owned.
std::optional<MlxGroupedLinear> mlx_group_linears(
    std::span<const MlxLinear* const> linears);

// Runtime-owned coordinator for an arbitrary number of independent
// projections over the same activation. Eligible contiguous projections are
// partitioned into the largest genuinely fused groups supported by the
// packed kernel; dense or unsupported members remain exact standalone
// calls. Model adapters therefore describe topology without duplicating
// format- or row-count dispatch policy.
class MlxProjectionBatch {
public:
    explicit MlxProjectionBatch(
        std::vector<const MlxLinear*> linears);

    std::vector<mlx::core::array> operator()(
        const mlx::core::array& input) const;

    // Execute a two-projection Gate/Up batch and apply SwiGLU. When the
    // registered formats expose a fused small-M implementation, projection
    // and activation stay in that kernel; otherwise the same coordinator
    // performs the best grouped/standalone projection partition first.
    mlx::core::array swiglu(
        const mlx::core::array& input,
        float limit = 0.0f) const;
    bool supports_fused_swiglu(
        const mlx::core::array& input) const noexcept;

    std::size_t projection_count() const noexcept;
    std::size_t grouped_projection_count() const noexcept;
    bool projections_share_group(
        std::size_t begin,
        std::size_t count) const noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

class MlxEmbedding {
public:
    static MlxEmbedding load(
        const MfqContainer& model,
        const std::string& name);

    explicit MlxEmbedding(MlxNintWeight weight);
    explicit MlxEmbedding(MlxNint8ZeroWeight weight);
    explicit MlxEmbedding(MlxVqWeight weight);
    explicit MlxEmbedding(MlxMxWeight weight);
    explicit MlxEmbedding(MlxTqWeight weight);
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
        MlxMxWeight,
        MlxTqWeight,
        mlx::core::array> weight_;
    int vocabulary_size_ = 0;
    int hidden_size_ = 0;
};

} // namespace mfq::metal
