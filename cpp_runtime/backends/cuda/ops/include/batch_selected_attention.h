#pragma once

#include "mfq_cuda_kv_quantization.h"
#include "../../runtime/batch_decode_inputs.h"

#include <functional>

namespace mfq::cuda {

// Decode geometry is shared by a layer; lengths and allocations belong to rows.
struct BatchSelectedAttentionConfig {
    int64_t pool, budget, capacity;
    KvQuantization quantization;
};

using BatchPoolTransform = std::function<mfq_tensor_backend::Tensor(
    const mfq_tensor_backend::Tensor&, const mfq_tensor_backend::Tensor&)>;

bool batch_selected_attention_enabled();

// Fields are KV keys/values [H,L,D], raw index [L,I] and pooled index [P,I].
// Optional exclusive tier fields follow at field + 4; unused entries are null.
// Index caches are F16; KV is F16 or packed I32. Projections contain one token
// per independent request; rotary positions have [axis,row,1] layout.
mfq_tensor_backend::Tensor batch_selected_attention(const BatchDecodeInputs&, int64_t field,
    const mfq_tensor_backend::Tensor& query, const mfq_tensor_backend::Tensor& key,
    const mfq_tensor_backend::Tensor& value, const mfq_tensor_backend::Tensor& gate,
    const mfq_tensor_backend::Tensor& raw_index, const mfq_tensor_backend::Tensor& index_query,
    const mfq_tensor_backend::Tensor& positions, const BatchSelectedAttentionConfig&,
    const BatchPoolTransform&);

// Shared kernels consume the existing device row table directly.
void batch_attention_store(const BatchDecodeInputs&, int64_t field,
    const mfq_tensor_backend::Tensor& keys, const mfq_tensor_backend::Tensor& values);
mfq_tensor_backend::Tensor batch_index_pool(const BatchDecodeInputs&, int64_t field,
    const mfq_tensor_backend::Tensor& raw, int64_t pool);
void batch_pool_store(const BatchDecodeInputs&, int64_t field,
    const mfq_tensor_backend::Tensor& pooled, int64_t pool);
mfq_tensor_backend::Tensor batch_index_scores(const BatchDecodeInputs&, int64_t field,
    const mfq_tensor_backend::Tensor& query, int64_t pool, int64_t pools);
mfq_tensor_backend::Tensor batch_attention_selection(const BatchDecodeInputs&, int64_t field,
    const mfq_tensor_backend::Tensor& blocks, int64_t pool, int64_t budget, int64_t columns);

} // namespace mfq::cuda
