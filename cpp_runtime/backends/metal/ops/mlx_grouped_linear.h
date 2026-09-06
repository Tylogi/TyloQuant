#pragma once

#include "mlx_tpq.h"
#include "mlx_mx.h"
#include "mlx_nint.h"
#include "mlx_nint8_zero.h"
#include "mlx_vq.h"

#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// A valid packed projection which can participate in the ordinary grouped
// Q/K/V or gate/up Metal kernel. MlxGroupedLinear retains MLX array handles,
// so the underlying packed storage remains shared with the source weight.
using MlxGroupedLinearWeightRef = std::variant<
    const MlxNintWeight*,
    const MlxNint8ZeroWeight*,
    const MlxVqWeight*,
    const MlxTpqInt4Weight*,
    const MlxTpqPqWeight*,
    const MlxMxWeight*>;

class MlxGroupedLinearUnsupported : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Single-dispatch heterogeneous packed linear projections.
//
// All projections share an input width but may use different output widths
// and packed NINT, NINT8-0, NVQ, NPQ, TPQ-I4G64, TPQ product-VQ, MXFP4,
// or MXFP8 layouts. Expert-shaped/rotated NEPQ belongs to the MoE path and is
// rejected here. The decode-oriented kernel intentionally accepts only one
// through max_rows() flattened input rows. Callers should use their ordinary
// per-weight or dense-GEMM path when supports() returns false.
class MlxGroupedLinear {
public:
    static constexpr int max_rows() noexcept {
        return 16;
    }

    explicit MlxGroupedLinear(
        std::vector<MlxGroupedLinearWeightRef> weights);

    bool supports(const mlx::core::array& input) const noexcept;

    std::vector<mlx::core::array> matmul(
        const mlx::core::array& input) const;

    // Decode-only two-projection fast path. Equally sized NINT projections
    // share their activation loads. MXFP8 Gate/Up uses two independent
    // 16-lane halves of each SIMD group so both projections remain parallel.
    // Both paths round to the input dtype before applying limited SwiGLU in
    // the same Metal dispatch.
    bool supports_single_row_swiglu(
        const mlx::core::array& input) const noexcept;
    mlx::core::array single_row_swiglu(
        const mlx::core::array& input,
        float limit) const;
    std::vector<mlx::core::array> operator()(
        const mlx::core::array& input) const {
        return matmul(input);
    }

    int input_size() const noexcept;
    int total_output_size() const noexcept;
    std::size_t projection_count() const noexcept;
    const std::vector<int>& output_sizes() const noexcept;

    // Logical bytes in the referenced packed weights. This does not imply
    // that the grouped object owns another copy of those bytes.
    std::size_t packed_nbytes() const noexcept;

    // Production projection groups bind each source array directly to one
    // Metal dispatch. Larger NINT-only groups use the pooled compatibility
    // kernel; groups which exceed the direct Metal buffer limit and contain
    // VQ, TPQ, or MX layouts are unsupported.
    bool uses_zero_copy_storage() const noexcept;
    std::size_t copied_packed_nbytes() const noexcept;

    // True when a float16, single-row invocation can use a projection-fused
    // decode kernel. The kernels assign the same output tile to every
    // projection, so Q/K/V reuse each activation group in one dispatch.
    // Other dtypes/row counts continue to use the ordinary grouped path.
    bool has_single_row_nint_fast_path() const noexcept;
    bool has_single_row_mxfp8_fast_path() const noexcept;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace mfq::metal
