#pragma once

#include "mfq_tensor_backend.h"
#include <vector>

mfq_tensor_backend::Tensor deepseek_v41_normalize_topk_cuda(
    mfq_tensor_backend::Tensor selected, mfq_tensor_backend::Tensor values, int64_t pool_length);

mfq_tensor_backend::Tensor deepseek_v41_engram_rows_cuda(
    mfq_tensor_backend::Tensor packed, mfq_tensor_backend::Tensor inverse, int64_t width);

mfq_tensor_backend::Tensor deepseek_v41_engram_gate_add_cuda(
    mfq_tensor_backend::Tensor hidden, mfq_tensor_backend::Tensor projected,
    mfq_tensor_backend::Tensor query_key_weight, double epsilon);

std::vector<mfq_tensor_backend::Tensor> deepseek_v41_hc_prepare_cuda(
    mfq_tensor_backend::Tensor residual,
    mfq_tensor_backend::Tensor previous_pre,
    mfq_tensor_backend::Tensor norm,
    double epsilon);

mfq_tensor_backend::Tensor deepseek_v41_hc_input_cuda(
    mfq_tensor_backend::Tensor residual,
    mfq_tensor_backend::Tensor previous_pre,
    mfq_tensor_backend::Tensor norm,
    double epsilon);

mfq_tensor_backend::Tensor deepseek_v41_mxfp8_e4m3_sim_cuda(
    mfq_tensor_backend::Tensor input);

void deepseek_v41_mxfp8_e4m3_sim_into_cuda(
    mfq_tensor_backend::Tensor input,mfq_tensor_backend::Tensor output);

mfq_tensor_backend::Tensor deepseek_v41_join_kv_cuda(
    mfq_tensor_backend::Tensor first,mfq_tensor_backend::Tensor second);

mfq_tensor_backend::Tensor deepseek_v41_mxfp4_e4m3_scale_sim_cuda(
    mfq_tensor_backend::Tensor input);

mfq_tensor_backend::Tensor deepseek_v41_hc_project_cuda(
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor previous,
    mfq_tensor_backend::Tensor function);

std::vector<mfq_tensor_backend::Tensor> deepseek_v41_hc_finalize_cuda(
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor previous,
    mfq_tensor_backend::Tensor norm, mfq_tensor_backend::Tensor partials,
    mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor base,
    int64_t iterations, double rms_epsilon, double hc_epsilon);

void deepseek_v41_kv_norm_rope_store_cuda(
    mfq_tensor_backend::Tensor input,mfq_tensor_backend::Tensor weight,
    mfq_tensor_backend::Tensor positions,mfq_tensor_backend::Tensor cosine,
    mfq_tensor_backend::Tensor sine,mfq_tensor_backend::Tensor output,double epsilon);
