#pragma once
#include <cstdint>
#include <hip/hip_runtime_api.h>

struct ProjectionView {
    const std::uint32_t* words;
    const std::uint64_t* descriptors;
    const void* scales;
    const void* relative;
    const float* bias;
    int rows, stored_columns, input_columns, gslog, columns;
};

struct TqProjectionJob {
    ProjectionView weight;
    void* output;
    int block_end;
};
struct TqProjectionPlan { TqProjectionJob jobs[4]; int count; };
extern "C" int mfq_tq_group(TqProjectionPlan, const void*, int, hipStream_t);
extern "C" int mfq_tq_swiglu(ProjectionView, ProjectionView, const void*, void*, int, hipStream_t);

extern "C" int mfq_tq_forward(ProjectionView, const void*, void*, int, int, hipStream_t);
extern "C" int mfq_tq_backward(ProjectionView, const void*, void*, int, int, hipStream_t);
extern "C" int mfq_tq_materialize(ProjectionView, void*, int, hipStream_t);
extern "C" int mfq_tq_materialize_bf16(ProjectionView, void*, int, hipStream_t);
extern "C" int mfq_tq_prefill_f32_supported(int);
extern "C" int mfq_tq_prefill_f32(ProjectionView,const float*,float*,int,void*,std::uint64_t,hipStream_t);
extern "C" int mfq_tq_group_bf16(TqProjectionPlan,const float*,const std::uint16_t*,const std::uint16_t*,float*,float*,int,int,hipStream_t);
extern "C" int mfq_tq_q8_tokens(ProjectionView, const void*, void*, int, int,
    void*, std::uint64_t, hipStream_t);
