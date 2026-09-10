#pragma once

#include "mfq_tensor_backend.h"

#include <cstdint>
#include <vector>

std::vector<mfq_tensor_backend::Tensor> dsv4_hc_pre_cuda(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor mixes,
    mfq_tensor_backend::Tensor scale,
    mfq_tensor_backend::Tensor base,
    std::int64_t iterations,
    double eps);

mfq_tensor_backend::Tensor dsv4_hc_post_cuda(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor residual,
    mfq_tensor_backend::Tensor post,
    mfq_tensor_backend::Tensor combination);
