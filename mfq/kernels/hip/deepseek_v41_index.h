#pragma once
#include "mfq_tensor_backend.h"
#include <cstdint>

mfq_tensor_backend::Tensor deepseek_v41_candidate_positions8_cuda(
    mfq_tensor_backend::Tensor blocks, std::int64_t pool_length);

mfq_tensor_backend::Tensor deepseek_v41_finalize_index_topk_cuda(
    mfq_tensor_backend::Tensor scores, mfq_tensor_backend::Tensor relative,
    mfq_tensor_backend::Tensor source_indices, std::int64_t pool_length);

mfq_tensor_backend::Tensor deepseek_v41_block_scores8_cuda(
    mfq_tensor_backend::Tensor scores, std::int64_t ratio, std::int64_t position);

mfq_tensor_backend::Tensor deepseek_v41_candidate_blocks8_cuda(
    mfq_tensor_backend::Tensor scores, std::int64_t ratio,
    std::int64_t position, std::int64_t requested_blocks);

mfq_tensor_backend::Tensor deepseek_v41_candidate_blocks8_f16_cuda(
    mfq_tensor_backend::Tensor scores, std::int64_t ratio,
    std::int64_t position, std::int64_t requested_blocks);

mfq_tensor_backend::Tensor deepseek_v41_candidate_blocks8_ordered16_cuda(
    mfq_tensor_backend::Tensor scores, std::int64_t ratio,
    std::int64_t position, std::int64_t requested_blocks);
