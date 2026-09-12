#pragma once

#include "mlx_deepseek_family_causal_lm.h"

#include <cstddef>
#include <filesystem>

namespace mfq::metal {

// Architecture-owned raw-HF factory. It validates the checkpoint as V4.1,
// then returns the established family storage type used by server callbacks.
// The execution object and generated MLX/Metal graph remain unchanged.
MlxDeepseekFamilyCausalLm load_deepseek_v41_hf(
    const std::filesystem::path& model_root,
    int max_context,
    std::size_t expert_cache_bytes,
    std::size_t io_workers = 8,
    bool prefill_overlap = true);

} // namespace mfq::metal
