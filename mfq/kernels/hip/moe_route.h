#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

struct MfqMoeRouteScanView {
    std::int32_t *counts,*cursors,*bounds,*tiles,*secondary,*tertiary;
    int experts,tile_m,secondary_m,tertiary_m;
};

struct MfqMoeRouteMapView {
    MfqMoeRouteScanView scan;
    const std::int32_t* ids;
    std::int32_t *ids_dst,*tile_experts,*secondary_experts,*tertiary_experts;
    int pair_count;
};

// Return false when the caller should use its existing route-map pipeline.
bool mfq_moe_route_map_fast_cuda(const MfqMoeRouteMapView&,int device,hipStream_t);
bool mfq_moe_route_scan_parallel_cuda(const MfqMoeRouteScanView&,int device,hipStream_t);
