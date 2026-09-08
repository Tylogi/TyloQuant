#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>

void paged_kv_cache_write_cuda(
    mfq_tensor_backend::Tensor k_chunk_ptrs,
    mfq_tensor_backend::Tensor v_chunk_ptrs,
    mfq_tensor_backend::Tensor page_table,
    mfq_tensor_backend::Tensor k,
    mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor positions,
    int64_t page_size,
    int64_t pages_per_chunk);

mfq_tensor_backend::Tensor attention_paged_cache_decode_cuda(
    mfq_tensor_backend::Tensor q,
    mfq_tensor_backend::Tensor k_chunk_ptrs,
    mfq_tensor_backend::Tensor v_chunk_ptrs,
    mfq_tensor_backend::Tensor page_table,
    mfq_tensor_backend::Tensor seq_len,
    double scale,
    int64_t page_size,
    int64_t pages_per_chunk,
    int64_t kv_heads,
    mfq_tensor_backend::Tensor partial_o,
    mfq_tensor_backend::Tensor partial_m,
    mfq_tensor_backend::Tensor partial_l,
    int64_t parts,
    bool dynamic_parts);
