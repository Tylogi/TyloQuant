#pragma once
#include "tq_projection.h"
extern "C" int mfq_tq_q8_group(TqProjectionPlan,const void*,int,void*,std::uint64_t,hipStream_t);
extern "C" int mfq_tq_q8_swiglu(ProjectionView,ProjectionView,const void*,void*,int,
    void*,std::uint64_t,hipStream_t);
