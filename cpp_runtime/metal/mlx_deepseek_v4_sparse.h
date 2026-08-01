#pragma once

#include <optional>
#include <utility>

#include <mlx/mlx.h>

namespace mfq::metal {

// A bounded decode-compressor delta.  emitted is [B,1,D] and is valid only
// when the corresponding int32 emit_rows entry is non-negative.
struct MlxDsv4PoolStep {
    mlx::core::array emitted;
    mlx::core::array emit_rows;
    mlx::core::array state_kv;
    mlx::core::array state_gate;
    std::optional<mlx::core::array> prev_kv;
    std::optional<mlx::core::array> prev_gate;
};

// Compatibility result for a capacity-backed compressed cache.
struct MlxDsv4PoolUpdate {
    mlx::core::array pool;
    mlx::core::array state_kv;
    mlx::core::array state_gate;
    std::optional<mlx::core::array> prev_kv;
    std::optional<mlx::core::array> prev_gate;
};

// Simulate DeepSeek-V4's power-of-two-scaled E2M1 cache groups.
mlx::core::array dsv4_fp4_sim(
    const mlx::core::array& input);

// Compress raw windows to 512-D or 128-D cache rows.
mlx::core::array dsv4_compress(
    const mlx::core::array& kv,
    const mlx::core::array& gate,
    const mlx::core::array& ape,
    const mlx::core::array& norm,
    const std::optional<mlx::core::array>& prev_kv,
    const std::optional<mlx::core::array>& prev_gate,
    const mlx::core::array& positions,
    const mlx::core::array& cos,
    const mlx::core::array& sin,
    int ratio,
    bool overlap,
    int quant_mode = 0,
    float eps = 1e-6f);

// Update bounded compressor state without copying the long-context pool.
MlxDsv4PoolStep dsv4_decode_pool_step(
    const mlx::core::array& kv_token,
    const mlx::core::array& gate_token,
    const mlx::core::array& ape,
    const mlx::core::array& norm,
    const mlx::core::array& state_kv,
    const mlx::core::array& state_gate,
    const std::optional<mlx::core::array>& prev_kv,
    const std::optional<mlx::core::array>& prev_gate,
    const mlx::core::array& seq_len,
    const mlx::core::array& cos,
    const mlx::core::array& sin,
    int ratio,
    bool overlap,
    int quant_mode = 0,
    float eps = 1e-6f);

// Apply a bounded compressor delta to a capacity-backed pool with native MLX
// indexed update semantics.
MlxDsv4PoolUpdate dsv4_decode_pool_update(
    const mlx::core::array& kv_token,
    const mlx::core::array& gate_token,
    const mlx::core::array& ape,
    const mlx::core::array& norm,
    const mlx::core::array& state_kv,
    const mlx::core::array& state_gate,
    const std::optional<mlx::core::array>& prev_kv,
    const std::optional<mlx::core::array>& prev_gate,
    const mlx::core::array& pool,
    const mlx::core::array& seq_len,
    const mlx::core::array& cos,
    const mlx::core::array& sin,
    int ratio,
    bool overlap,
    int quant_mode = 0,
    float eps = 1e-6f);

// Compute the 64-head pooled-token indexer score.
mlx::core::array dsv4_indexer_scores(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& weights,
    int query_offset,
    int ratio);

// Decode-specialized streaming indexer score path.
mlx::core::array dsv4_indexer_scores_decode(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& weights,
    int query_offset,
    int ratio);

// Fixed-width half-precision top-512 selection.
mlx::core::array dsv4_topk512(
    const mlx::core::array& scores,
    bool deterministic = true);

// Build circular-local plus pooled sparse-attention plans.  The pair contains
// int32 cache indices followed by a float16 additive mask.
std::pair<mlx::core::array, mlx::core::array>
dsv4_build_prefill_plan(
    const mlx::core::array& topk,
    int query_offset,
    int local_history,
    int pool_len,
    int ratio,
    int window);

std::pair<mlx::core::array, mlx::core::array>
dsv4_build_decode_plan(
    const mlx::core::array& topk,
    const mlx::core::array& seq_len,
    int pool_len,
    int ratio,
    int window);

// Selected-row DSV4 attention.  Dispatches decode, short-query, or prefill-MMA
// Metal kernels from the query count.  meta is intentionally accepted for API
// parity with the CUDA/Python operator and has no numerical role.
mlx::core::array attention_dsv4_sparse(
    const mlx::core::array& q,
    const mlx::core::array& kv,
    const mlx::core::array& indices,
    const mlx::core::array& mask,
    const mlx::core::array& sinks,
    const std::optional<mlx::core::array>& meta = std::nullopt,
    std::optional<float> scale = std::nullopt);

} // namespace mfq::metal
