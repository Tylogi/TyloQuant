#pragma once

#include "mfq_tensor_backend.h"

#include <vector>

// Selective state scan (SSD recurrence) shared by linear-attention mixers.
// See mfq/kernels/cuda/selective_state_scan.cu for the recurrence and for the
// dtype contract: the recurrent state, parameters and outputs are float32,
// while x/dt/B/C may be f32, f16 or bf16 (all four sharing one dtype).

// One decode step, state[B,H,P,N] updated in place; returns y[B,H,P] f32.
// dt[B,H] is the raw (pre-softplus) time step, dt_bias[H]/a_log[H]/D[H] are f32
// parameters and B/C[B,G,N] are the grouped projections (head h uses group
// h / (H/G)). dt_limit_lo/dt_limit_hi clamp softplus(dt + dt_bias).
mfq_tensor_backend::Tensor selective_state_scan_decode_cuda(
    mfq_tensor_backend::Tensor state,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor dt,
    mfq_tensor_backend::Tensor dt_bias,
    mfq_tensor_backend::Tensor a_log,
    mfq_tensor_backend::Tensor b,
    mfq_tensor_backend::Tensor c,
    mfq_tensor_backend::Tensor d,
    double dt_limit_lo,
    double dt_limit_hi);

// Any-length prefill of the same recurrence: x[B,T,H,P], dt[B,T,H],
// B/C[B,T,G,N] and an optional initial_state[B,H,P,N]. Returns
// {y[B,T,H,P], final_state[B,H,P,N]}, both f32; the result equals T decode
// calls up to floating point rounding.
std::vector<mfq_tensor_backend::Tensor> selective_state_scan_prefill_cuda(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor dt,
    mfq_tensor_backend::Tensor dt_bias,
    mfq_tensor_backend::Tensor a_log,
    mfq_tensor_backend::Tensor b,
    mfq_tensor_backend::Tensor c,
    mfq_tensor_backend::Tensor d,
    mfq_tensor_backend::Tensor initial_state,
    double dt_limit_lo,
    double dt_limit_hi);
