#pragma once

#include "mfq_container.h"

#include <cstdint>
#include <memory>

#include <mlx/mlx.h>

namespace mfq::metal {

// Shared contract for patch-linear ViTs that use bidirectional attention,
// two-axis RoPE, RMSNorm and a strided spatial aligner. Architecture adapters
// provide only dimensions and prompt-span semantics.
struct MlxAxialPatchVisionConfig {
    std::int64_t hidden = 0;
    std::int64_t layers = 0;
    std::int64_t heads = 0;
    std::int64_t intermediate = 0;
    std::int64_t patch_size = 0;
    std::int64_t downsample_ratio = 0;
    std::int64_t output_hidden = 0;
    double rope_theta = 10'000.0;
    double rms_eps = 1e-6;

    void validate() const;
};

class MlxAxialPatchVisionTower {
public:
    static MlxAxialPatchVisionTower load(
        const MfqContainer& model,
        const MlxAxialPatchVisionConfig& config);

    MlxAxialPatchVisionTower(MlxAxialPatchVisionTower&&) noexcept;
    MlxAxialPatchVisionTower& operator=(
        MlxAxialPatchVisionTower&&) noexcept;
    ~MlxAxialPatchVisionTower();

    MlxAxialPatchVisionTower(const MlxAxialPatchVisionTower&) = delete;
    MlxAxialPatchVisionTower& operator=(
        const MlxAxialPatchVisionTower&) = delete;

    // patches accepts [H*W,3,P,P] or [H*W,3*P*P]. The output is the
    // row-major, ceil(H/R)*ceil(W/R) aligned LLM embedding grid.
    mlx::core::array encode(
        const mlx::core::array& patches,
        int height,
        int width) const;

    const MlxAxialPatchVisionConfig& config() const noexcept;

private:
    struct Impl;
    explicit MlxAxialPatchVisionTower(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::metal
