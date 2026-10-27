#pragma once

#include "mfq_tensor_backend.h"
#include "mfq_hip_kv_hot_cache.h"
#include <cmath>
#include <string>

namespace mfq::cuda {
struct KvQuantization {
    double bits = 0;
    bool enabled() const noexcept { return bits != 0; }
    int key_bits() const noexcept { return int(std::floor(bits)); }
    int value_bits() const noexcept { return int(std::ceil(bits)); }
};
KvQuantization cuda_kv_quantization();
int cuda_kv_packed_width(int dimension, int bits);
std::string cuda_kv_quantization_tag();
mfq_tensor_backend::Tensor cuda_kv_encode(const mfq_tensor_backend::Tensor&, int bits, bool value);
mfq_tensor_backend::Tensor cuda_kv_decode(const mfq_tensor_backend::Tensor&, int dimension, int bits,
    bool value, mfq_tensor_backend::ScalarType dtype = mfq_tensor_backend::kFloat16);

// Packed rows contain the FP32 norm followed by bit-packed centroid indices.
// Both resident and paged attention consume these rows without a floating KV copy.
struct QuantizedKvPages {
    mfq_tensor_backend::Tensor keys, values, table;
    int page_size = 0, pages_per_chunk = 0, heads = 0;
    bool enabled() const noexcept { return page_size > 0; }
};
void cuda_kv_write_paged(const QuantizedKvPages&, const mfq_tensor_backend::Tensor& keys,
    const mfq_tensor_backend::Tensor& values, const mfq_tensor_backend::Tensor& positions);
// Query [B,H,T,D], packed caches [B,KVH,L,P], selected [B,T,N] or empty for dense.
// Positions [T] or [B,T] are device-side causal boundaries, including graph replay.
// Output [B,T,H,D], FP32. Indexer and recurrent state are never quantized.
mfq_tensor_backend::Tensor cuda_kv_attention(const mfq_tensor_backend::Tensor& query,
    const mfq_tensor_backend::Tensor& keys, const mfq_tensor_backend::Tensor& values,
    KvQuantization quantization, const mfq_tensor_backend::Tensor& selected = {},
    const mfq_tensor_backend::Tensor& positions = {}, int64_t query_offset = 0,
    const QuantizedKvPages& pages = {}, int64_t capacity = 0,
    const mfq_tensor_backend::Tensor& lengths = {}, const mfq_tensor_backend::Tensor& mask = {},
    double scale = 0, HotKvView hot = {});
}
