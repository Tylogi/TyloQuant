#pragma once

#include "generation_step.h"
#include "cuda_batching.h"
#include "cuda_execution.h"
#include "core/decode_graph.h"
#include "core/mtp.h"
#include "cuda_runtime_config.h"

#include <optional>
#include <utility>

namespace mfq::cuda::internal {
class TextSessionCache;

template <typename Model>
mfq::engine::Generation generate(
    Model& model, DecodeGraphCache& graph, TextSessionCache& cache,
    const CudaRuntimeConfig& config, mfq::engine::InferenceRequest& request,
    mfq::engine::InferenceOutput& output, MtpModule* mtp = nullptr,
    std::optional<CudaPreparedPrompt> prepared = {},
    mfq::engine::ContinuousBatch<CudaBatchOperations<Model>>* batching = nullptr,
    std::string request_id = {});
} // namespace mfq::cuda::internal

template <typename Model>
mfq::engine::Generation run_mtp_generation(
    Model& model, MtpModule& mtp, mfq::engine::InferenceRequest& request,
    mfq::engine::InferenceOutput& output, int64_t prefill_chunk_size = 2048,
    const CudaPreparedPrompt* prepared = nullptr, std::size_t reused_tokens = 0,
    mfq_tensor_backend::Tensor restored_last_hidden = {},
    mfq_tensor_backend::Tensor* session_last_hidden = nullptr,
    std::size_t stable_prefix_tokens = 0, bool pressure_retry = false);
