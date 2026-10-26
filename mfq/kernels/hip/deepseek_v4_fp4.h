#pragma once

#include "mfq_tensor_backend.h"

// Input is the projection's [B,H,T,128] view; output is F16 [B,T,H,128].
mfq_tensor_backend::Tensor dsv4_index_query_rope_fp4_cuda(
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor positions,
    mfq_tensor_backend::Tensor cosine,
    mfq_tensor_backend::Tensor sine);
