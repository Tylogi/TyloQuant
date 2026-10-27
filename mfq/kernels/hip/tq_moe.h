#pragma once
#include "tq_projection.h"

// Ordinary flattened [experts,rows,columns] TQ banks. The row decoder retains
// each row's precision, column/group scales and original encoded ring.
struct TqMoeRoutePlan {
    const int32_t *ids,*pairs,*bounds,*tiles,*experts;
    int tokens,routes,expert_count;
    int tile_rows=16;
    int expert_offset=0;
};
extern "C" int mfq_tq_moe_projection(ProjectionView,const void*,void*,int rows,
    TqMoeRoutePlan,int dtype,bool routed,hipStream_t);
extern "C" int mfq_tq_moe_swiglu(ProjectionView,ProjectionView,const void*,void*,int rows,
    TqMoeRoutePlan,int dtype,float limit,hipStream_t);
extern "C" int mfq_tq_moe_reduce(const void*,const float*,void*,int tokens,int routes,
    int rows,int dtype,hipStream_t);
extern "C" int mfq_tq_moe_peer_join(const void*,const void*,void*,int64_t,hipStream_t);
extern "C" int mfq_tq_embedding_hip(ProjectionView,const int64_t*,int count,void*,int dtype,hipStream_t);
extern "C" int mfq_tq_embedding_rows_hip(const ProjectionView*,int count,int columns,void*,int dtype,hipStream_t);

// gfx9424096-token native Gate/Up;workspace contains route metadata only.
extern "C" uint64_t mfq_tq_moe_wide128_workspace(ProjectionView,ProjectionView,int,
    TqMoeRoutePlan,int,float);
extern "C" int mfq_tq_moe_swiglu_wide128(ProjectionView,ProjectionView,const void*,void*,int,
    TqMoeRoutePlan,int,float,void*,uint64_t,hipStream_t);
