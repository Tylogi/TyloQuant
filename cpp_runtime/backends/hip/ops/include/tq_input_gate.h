#pragma once
#include "mfq_tensor_backend.h"
mfq_tensor_backend::Tensor tq_gate_input(mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor gate,int mode);
extern "C" int mfq_tq_input_gate(const void* input,const void* gate,void* output,
    int64_t count,int dtype,int gate_dtype,int mode,int double_math,hipStream_t stream);
