#pragma once

#include "cuda_execution.h"
#include "fp8_sq.h"
#include "mx.h"
#include "mxfp4_sq.h"
#include "nint.h"
#include "vq.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mfq::cuda {
class MfeMxfp4ExpertStore;
}

class MoeCachedSource;
struct MixedMoeRuntime;
struct MoeRoutePlan;

mfq_tensor_backend::Tensor moe_tensor_to_device(
    ModelParallelCollectiveRuntime& collectives,
    mfq_tensor_backend::Tensor value,
    int device);
const MoeRoutePlan& moe_route_to_device(
    ModelParallelCollectiveRuntime& collectives,
    const MoeRoutePlan& route,
    int device);
mfq_tensor_backend::Tensor reduce_model_parallel_outputs(
    CudaExecutionContext& execution,
    std::vector<mfq_tensor_backend::Tensor> outputs);
int select_nint_prefill_route_tile(
    const MoeRoutePlan& route, int tokens, int routes, int experts);
const mfq_tensor_backend::Tensor& nint_route_tile_bounds(
    const MoeRoutePlan& route, int tile_m);
const mfq_tensor_backend::Tensor& nint_route_tile_experts(
    const MoeRoutePlan& route, int tile_m);

struct MfePoolWeight {
    NintWeight weight;
    mfq_tensor_backend::Tensor expert_local;
    int local_experts = 0;
};

struct MoeActivationKey {
    int input_rows = 0;
    int groups = 0;
    int gs = 0;
    int device = 0;

    bool operator==(const MoeActivationKey & other) const {
        return input_rows == other.input_rows && groups == other.groups &&
               gs == other.gs && device == other.device;
    }
};

struct MoeActivationKeyHash {
    size_t operator()(const MoeActivationKey & key) const {
        size_t value = (size_t)key.input_rows;
        value = value * 1315423911u + (size_t)key.groups;
        value = value * 1315423911u + (size_t)key.gs;
        return value * 1315423911u + (size_t)key.device;
    }
};

struct MoeActivationWorkspace {
    static constexpr int stable_row_limit = 2048;
    static int capacity_rows(int rows, bool stable) {
        if (!stable) return ((rows + 255) / 256) * 256;
        int capacity = 1;
        while (capacity < rows) capacity *= 2;
        return capacity;
    }
    mfq_tensor_backend::Tensor qx;
    mfq_tensor_backend::Tensor xscale;
};

// Eager prefill scratch (key rows 0) is shared only by the expert pools of
// one projection call. Drop it on return; otherwise every MoE layer keeps a
// buffer sized for the longest prompt seen. Later layers reuse the freed
// pages in stream order.
template <class Workspaces>
struct EagerActivationScratch {
    Workspaces & workspaces;
    bool active;
    ~EagerActivationScratch() {
        if (!active) return;
        for (auto it = workspaces.begin(); it != workspaces.end();) {
            if (it->first.input_rows == 0) it = workspaces.erase(it);
            else ++it;
        }
    }
};

struct MoeActivationGeometry {
    int groups = 0;
    int gs = 0;
    int transform_block = 0;
    uint64_t transform_seed = 0;

    bool operator==(const MoeActivationGeometry & other) const {
        return groups == other.groups && gs == other.gs &&
            transform_block == other.transform_block &&
            transform_seed == other.transform_seed;
    }
};

struct MoeRoutePlan {
    mfq_tensor_backend::Tensor ids;
    mfq_tensor_backend::Tensor ids_dst;
    mfq_tensor_backend::Tensor expert_bounds;
    mfq_tensor_backend::Tensor tile_bounds;
    mfq_tensor_backend::Tensor tile_experts;
    mfq_tensor_backend::Tensor mma_tile_bounds;
    mfq_tensor_backend::Tensor mma_tile_experts;
    mfq_tensor_backend::Tensor wide_tile_bounds;
    mfq_tensor_backend::Tensor wide_tile_experts;
    mfq_tensor_backend::Tensor counts;
    mfq_tensor_backend::Tensor cursors;
    int n_experts = 0;
    int mma_tile_m = 8;
    int wide_tile_m = 8;
    bool map_ready = false;
    uint64_t generation = 0;
    mutable std::shared_ptr<std::vector<int32_t>>
        host_unique_experts;
    mutable std::shared_ptr<std::unordered_map<
        int, std::shared_ptr<MoeRoutePlan>>> device_replicas =
            std::make_shared<std::unordered_map<
                int, std::shared_ptr<MoeRoutePlan>>>();
};
