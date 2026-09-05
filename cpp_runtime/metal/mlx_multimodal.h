#pragma once

#include <mlx/mlx.h>

namespace mfq::metal {

// Shared embedding-injection primitive used by every native multimodal
// architecture.  Keeping span validation here prevents each vision/audio
// frontend from growing a subtly different scatter contract.
mlx::core::array replace_multimodal_embedding_span(
    const mlx::core::array& embeddings,
    const mlx::core::array& replacement,
    int batch,
    int begin,
    int end);

} // namespace mfq::metal
