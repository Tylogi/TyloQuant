#pragma once

#include "core/causal_model.h"
#include "storage/session_state.h"
#include "models/nemotron_h/causal_lm.h"
#include "models/nemotron_h/config.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mfq {
class ModelSource;
}
struct Block;

namespace mfq::cuda::nemotron_h {

using Config = mfq::models::nemotron_h::Config;

std::unique_ptr<::Block> load_block(CudaExecutionContext& execution,
                                    const mfq::ModelSource& source, const Config& config, int layer,
                                    const std::string& type,
                                    std::string_view tensor_root = "model");

bool supports_text_session_state(const std::vector<std::unique_ptr<::Block>>& blocks);

TextSessionState capture_text_session_state(const std::vector<std::unique_ptr<::Block>>& blocks,
                                            const std::vector<std::int64_t>& tokens,
                                            std::int64_t cache_position);

void restore_text_session_state(std::vector<std::unique_ptr<::Block>>& blocks,
                                const TextSessionState& state);

} // namespace mfq::cuda::nemotron_h

namespace mfq::cuda {

struct NemotronHModel : CausalResources {
    template <class Backend> using CausalModel = mfq::models::nemotron_h::CausalLm<Backend>;
    mfq::models::nemotron_h::Config config;

    void adapter_validate_components(const mfq::ModelGraph& graph) const;
    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource& source, int layer, int device,
                                              const std::string& type);
    // Nemotron-H's MTP head consumes the backbone hidden state *after*
    // output_norm (unlike DeepSeek-V3); see the implementation for sources.
    mfq_tensor_backend::Tensor
    adapter_raw_hidden(const mfq_tensor_backend::Tensor& hidden,
                       const mfq_tensor_backend::Tensor& finalized) const;
};

template <> struct CudaSessionCodec<NemotronHModel> {
    using Model = CausalLm<NemotronHModel>;
    static TextSessionStateKind kind(const Model& model);
    static bool supports_paged(const Model& model);
    static TextSessionState capture(const Model& model, const std::vector<int64_t>& tokens);
    static void restore(Model& model, const TextSessionState& state);
};

extern template struct FullAttentionSessionCodec<NemotronHModel>;

} // namespace mfq::cuda

namespace mfq::models {
extern template struct nemotron_h::CausalLm<cuda::CudaCausalOps<cuda::NemotronHModel>>;
} // namespace mfq::models
