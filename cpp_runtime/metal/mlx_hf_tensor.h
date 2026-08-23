#pragma once

#include "hf_safetensors_store.h"
#include "mlx_mx.h"
#include "mlx_tensor.h"

#include <filesystem>
#include <memory>
#include <string>

#include <mlx/mlx.h>

namespace mfq::metal {

// Direct-to-UMA loader for official Hugging Face Safetensors checkpoints.
// Dense BF16/F16/F32/integer tensors preserve their storage dtype. Native
// MXFP4/MXFP8 weight-scale pairs remain packed and use the existing Metal
// kernels without an intermediate MFQ file.
class MlxHfTensorStore {
public:
    explicit MlxHfTensorStore(std::filesystem::path root);
    explicit MlxHfTensorStore(std::shared_ptr<HfSafetensorStore> checkpoint);

    const HfSafetensorStore& checkpoint() const noexcept;
    std::shared_ptr<HfSafetensorStore> shared_checkpoint() const noexcept;

    mlx::core::array load_dense(const std::string& name) const;
    MlxMxWeight load_mx(const std::string& name) const;
    MlxLinear load_linear(const std::string& name) const;
    MlxEmbedding load_embedding(const std::string& name) const;

private:
    std::shared_ptr<HfSafetensorStore> checkpoint_;
};

} // namespace mfq::metal
