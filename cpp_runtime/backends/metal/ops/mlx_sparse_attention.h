#pragma once

#include <optional>

#include <mlx/mlx.h>

namespace mfq::metal {

// Common selected-block sparse-attention seam for Metal runtimes. Model
// adapters own index construction and cache semantics; this operator owns the
// direct indexed GQA execution. Blocks identify fixed-width, chronological
// cache blocks. The incomplete causal tail is appended inside the kernel.
mlx::core::array mlx_sparse_block_gqa_attention(
    const mlx::core::array& query,
    const mlx::core::array& key,
    const mlx::core::array& value,
    const mlx::core::array& selected_blocks,
    int query_offset,
    int block_size,
    std::optional<float> scale = std::nullopt);

// Common selected-token sparse-MLA seam.  The adapter owns index selection,
// masks, and cache updates; the backend owns the fused indexed softmax/value
// reduction for the 64-head, 512-dimensional latent-attention geometry.
mlx::core::array mlx_sparse_selected_mla_attention(
    const mlx::core::array& query,
    const mlx::core::array& kv_cache,
    const mlx::core::array& selected_indices,
    const mlx::core::array& selected_mask,
    const mlx::core::array& sinks,
    std::optional<float> scale = std::nullopt);

// Single-token sparse-MLA specialization over a circular local cache plus an
// optional compressed pool.  Keeping this in the common backend lets DSA and
// sparse-MLA adapters share execution while retaining their own index policy.
mlx::core::array mlx_sparse_circular_mla_decode_attention(
    const mlx::core::array& query,
    const mlx::core::array& local_kv,
    const std::optional<mlx::core::array>& pooled_kv,
    int pool_len,
    const mlx::core::array& topk,
    const mlx::core::array& sinks,
    int sequence_length,
    int pool_ratio,
    int local_window,
    std::optional<float> scale = std::nullopt);

} // namespace mfq::metal
