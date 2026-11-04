#pragma once

#include <cuda_runtime_api.h>

// Capture status, id and graph of a stream. CUDA 13 folded the versioned query
// into cudaStreamGetCaptureInfo and added an edge-data output.
inline cudaError_t mfq_stream_capture_info(cudaStream_t stream, cudaStreamCaptureStatus* status,
                                           unsigned long long* id, cudaGraph_t* graph) {
#if CUDART_VERSION >= 13000
    return cudaStreamGetCaptureInfo(stream, status, id, graph, nullptr, nullptr, nullptr);
#else
    return cudaStreamGetCaptureInfo_v2(stream, status, id, graph, nullptr, nullptr);
#endif
}

// Direct successors of a graph node; CUDA 13 added an edge-data output here too.
inline cudaError_t mfq_graph_dependent_nodes(cudaGraphNode_t node, cudaGraphNode_t* nodes, size_t* count) {
#if CUDART_VERSION >= 13000
    return cudaGraphNodeGetDependentNodes(node, nodes, nullptr, count);
#else
    return cudaGraphNodeGetDependentNodes(node, nodes, count);
#endif
}
