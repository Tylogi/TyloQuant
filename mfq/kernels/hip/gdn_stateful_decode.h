#pragma once
#include "mfq_tensor_backend.h"
#include <vector>
namespace mfq_hip_gdn_stateful {
std::vector<mfq_tensor_backend::Tensor> decode(
    mfq_tensor_backend::Tensor qkv,mfq_tensor_backend::Tensor output_gate,
    mfq_tensor_backend::Tensor alpha,mfq_tensor_backend::Tensor beta,
    mfq_tensor_backend::Tensor convolution_state,mfq_tensor_backend::Tensor recurrent_state,
    mfq_tensor_backend::Tensor convolution_weight,mfq_tensor_backend::Tensor dt_bias,
    mfq_tensor_backend::Tensor a_log,mfq_tensor_backend::Tensor norm_weight,
    int64_t key_heads,int64_t value_heads,int64_t width,
    double convolution_eps,double norm_eps,bool silu_gate,bool output_half,
    bool transposed_state,mfq_tensor_backend::Tensor workspace,bool inplace_state,
    mfq_tensor_backend::Tensor state_addresses,
    mfq_tensor_backend::Tensor checkpoint_convolution,mfq_tensor_backend::Tensor checkpoint_recurrent);
}
