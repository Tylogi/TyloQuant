#pragma once

#include "mfq_tensor_backend.h"

mfq_tensor_backend::Tensor deepseek_v41_mxfp8_e4m3_sim_cuda(
    mfq_tensor_backend::Tensor input);

mfq_tensor_backend::Tensor deepseek_v41_mxfp4_e4m3_scale_sim_cuda(
    mfq_tensor_backend::Tensor input);
