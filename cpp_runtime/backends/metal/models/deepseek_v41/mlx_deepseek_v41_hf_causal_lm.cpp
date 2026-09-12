#include "mlx_deepseek_v41_hf_causal_lm.h"

#include <stdexcept>

namespace mfq::metal {

MlxDeepseekFamilyCausalLm load_deepseek_v41_hf(
    const std::filesystem::path& model_root,
    int max_context,
    std::size_t expert_cache_bytes,
    std::size_t io_workers,
    bool prefill_overlap) {
    auto runtime = MlxDeepseekFamilyCausalLm::load_hf(
        model_root,
        max_context,
        expert_cache_bytes,
        io_workers,
        prefill_overlap);
    if (!runtime.config().is_v41()) {
        throw std::runtime_error(
            "DeepSeek-V4.1 raw-HF runtime received a V4 checkpoint");
    }
    return runtime;
}

} // namespace mfq::metal
