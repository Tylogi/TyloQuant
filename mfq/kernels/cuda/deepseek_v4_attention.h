#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>
#include <vector>

mfq_tensor_backend::Tensor dsv4_compress_cuda(
    mfq_tensor_backend::Tensor kv,
    mfq_tensor_backend::Tensor gate,
    mfq_tensor_backend::Tensor ape,
    mfq_tensor_backend::Tensor norm,
    mfq_tensor_backend::Tensor prev_kv,
    mfq_tensor_backend::Tensor prev_gate,
    mfq_tensor_backend::Tensor positions,
    mfq_tensor_backend::Tensor cos,
    mfq_tensor_backend::Tensor sin,
    std::int64_t ratio,
    bool overlap,
    std::int64_t quant_mode,
    double eps);

mfq_tensor_backend::Tensor dsv4_fp4_sim_cuda(
    mfq_tensor_backend::Tensor input);

mfq_tensor_backend::Tensor dsv4_decode_pool_update_cuda(
    mfq_tensor_backend::Tensor kv_token,
    mfq_tensor_backend::Tensor gate_token,
    mfq_tensor_backend::Tensor ape,
    mfq_tensor_backend::Tensor norm,
    mfq_tensor_backend::Tensor state_kv,
    mfq_tensor_backend::Tensor state_gate,
    mfq_tensor_backend::Tensor prev_kv,
    mfq_tensor_backend::Tensor prev_gate,
    mfq_tensor_backend::Tensor pool,
    mfq_tensor_backend::Tensor seq_len,
    mfq_tensor_backend::Tensor cos,
    mfq_tensor_backend::Tensor sin,
    std::int64_t ratio,
    bool overlap,
    std::int64_t quant_mode,
    double eps);

mfq_tensor_backend::Tensor dsv4_indexer_scores_cuda(
    mfq_tensor_backend::Tensor q,
    mfq_tensor_backend::Tensor k,
    mfq_tensor_backend::Tensor weights,
    std::int64_t query_offset,
    std::int64_t ratio);

mfq_tensor_backend::Tensor dsv4_topk512_cuda(
    mfq_tensor_backend::Tensor scores);

std::vector<mfq_tensor_backend::Tensor> dsv4_build_prefill_plan_cuda(
    mfq_tensor_backend::Tensor topk,
    std::int64_t query_offset,
    std::int64_t local_history,
    std::int64_t pool_len,
    std::int64_t ratio,
    std::int64_t window);

std::vector<mfq_tensor_backend::Tensor> dsv4_build_decode_plan_cuda(
    mfq_tensor_backend::Tensor topk,
    mfq_tensor_backend::Tensor seq_len,
    std::int64_t pool_len,
    std::int64_t ratio,
    std::int64_t window);

mfq_tensor_backend::Tensor attention_dsv4_sparse_cuda(
    mfq_tensor_backend::Tensor q,
    mfq_tensor_backend::Tensor kv,
    mfq_tensor_backend::Tensor indices,
    mfq_tensor_backend::Tensor mask,
    mfq_tensor_backend::Tensor sinks,
    mfq_tensor_backend::Tensor meta,
    double scale);
