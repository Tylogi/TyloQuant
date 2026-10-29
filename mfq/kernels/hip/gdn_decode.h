#pragma once
#include "mfq_tensor_backend.h"
mfq_tensor_backend::Tensor gdn_rms_silu_decode_cuda(mfq_tensor_backend::Tensor attended,
    mfq_tensor_backend::Tensor gate,mfq_tensor_backend::Tensor weight,double eps);
std::vector<mfq_tensor_backend::Tensor> l2_norm_pair_cuda(mfq_tensor_backend::Tensor first,
    mfq_tensor_backend::Tensor second,double eps);
