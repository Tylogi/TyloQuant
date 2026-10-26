#pragma once

#include "core/vision_ops.h"
#include "models/common/axial_patch_vision.h"
#include "step_sequence.h"

#include <cstdint>
#include <memory>

namespace mfq::cuda::axial_patch_vision {

using Config = mfq::models::axial_patch_vision::Config;

// CUDA implementation of the backend-neutral axial patch ViT contract shared
// with the Metal runtime. It owns only the vision tower; prompt-span semantics
// belong to the architecture adapter.
class CudaAxialPatchVisionTower {
  public:
    static CudaAxialPatchVisionTower load(
        CudaExecutionContext& execution,
        const mfq::ModelSource& model,
        const Config& config);

    CudaAxialPatchVisionTower(CudaAxialPatchVisionTower&&) noexcept;
    CudaAxialPatchVisionTower& operator=(
        CudaAxialPatchVisionTower&&) noexcept;
    ~CudaAxialPatchVisionTower();

    CudaAxialPatchVisionTower(const CudaAxialPatchVisionTower&) = delete;
    CudaAxialPatchVisionTower& operator=(
        const CudaAxialPatchVisionTower&) = delete;

    // Patches accept [H*W,3,P,P] or [H*W,3*P*P]. The output is the
    // row-major, ceil(H/R)*ceil(W/R) aligned embedding grid.
    mfq_tensor_backend::Tensor encode(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor patches,
        int height,
        int width) const;

    // Load one block at a time and restore borrowed expert storage before
    // yielding, so image preparation can share the request scheduler.
    mfq::StepSequence<mfq_tensor_backend::Tensor> encode_steps(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor patches,
        int height,
        int width) const;

    const Config& config() const noexcept;

  private:
    struct Impl;
    explicit CudaAxialPatchVisionTower(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::cuda::axial_patch_vision
