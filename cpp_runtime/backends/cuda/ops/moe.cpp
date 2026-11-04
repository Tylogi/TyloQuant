#include "../runtime/execution_options.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq_cuda_quant_ops.h"
#include "mfq_cuda_nint_route_hint.h"
#include "mfq_cuda_nvq1_decode.h"
#include "storage/weight_loader.h"
#include "moe.h"

#include "quant_linear.h"
#include "format.h"
#include "mfq_format_compat.h"
#include "mfe_expert_store.h"
#include "runtime/moe_pipeline.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <limits>
#include <unordered_set>

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

static MfeCpu unpack_mfe_impl(
        const std::vector<uint8_t> & blob,
        bool allow_partial) {
    if (blob.size() < 20) {
        throw std::runtime_error("invalid MFE header");
    }
    const bool version1 = std::memcmp(blob.data(), "NIM1", 4) == 0;
    const bool delta =
        std::memcmp(blob.data(), "MFD1", 4) == 0 ||
        std::memcmp(blob.data(), "NID2", 4) == 0;
    const bool version2 =
        std::memcmp(blob.data(), "MFE1", 4) == 0 ||
        std::memcmp(blob.data(), "NIM2", 4) == 0 || delta;
    if (!version1 && !version2) throw std::runtime_error("invalid MFE header");
    if (delta && !allow_partial) {
        throw std::runtime_error("MFE delta cannot be loaded as a full tensor");
    }
    size_t off = 4;
    MfeCpu result;
    result.n_experts = (int)read_u32_from(blob, off);
    result.out_per_expert = (int)read_u32_from(blob, off);
    result.neuron_len = (int)read_u32_from(blob, off);
    int pool_count = (int)read_u32_from(blob, off);
    if (result.n_experts <= 0 || result.out_per_expert <= 0 || result.neuron_len <= 0 ||
        pool_count <= 0 || pool_count > result.n_experts) {
        throw std::runtime_error("invalid MFE dimensions");
    }
    std::vector<int> owners((size_t)result.n_experts, -1);
    result.pools.reserve((size_t)pool_count);
    for (int pool_index = 0; pool_index < pool_count; ++pool_index) {
        const size_t pool_header_bytes = version1 ? 12 : 24;
        if (off + pool_header_bytes > blob.size()) {
            throw std::runtime_error("truncated MFE pool header");
        }
        uint32_t expert_count = read_u32_from(blob, off);
        uint32_t dtype_nbytes = version1 ? 0 : read_u32_from(blob, off);
        uint64_t payload_nbytes = read_u64_from(blob, off);
        uint64_t runtime_nbytes = version1 ? 0 : read_u64_from(blob, off);
        if (expert_count == 0 || expert_count > static_cast<uint32_t>(result.n_experts) ||
                expert_count > static_cast<uint32_t>(
                    std::numeric_limits<int>::max() / result.out_per_expert) ||
                dtype_nbytes > 32) {
            throw std::runtime_error("invalid MFE pool dimensions");
        }
        size_t ids_nbytes = (size_t)expert_count * sizeof(int32_t);
        size_t remaining = blob.size() - off;
        if (ids_nbytes > remaining) {
            throw std::runtime_error("truncated MFE pool payload");
        }
        remaining -= ids_nbytes;
        if (dtype_nbytes > remaining) {
            throw std::runtime_error("truncated MFE pool payload");
        }
        remaining -= dtype_nbytes;
        if (runtime_nbytes > remaining) {
            throw std::runtime_error("truncated MFE pool payload");
        }
        remaining -= static_cast<size_t>(runtime_nbytes);
        if (payload_nbytes > remaining) {
            throw std::runtime_error("truncated MFE pool payload");
        }
        MfeCpuPool pool;
        pool.expert_ids.resize(expert_count);
        std::memcpy(pool.expert_ids.data(), blob.data() + off, ids_nbytes);
        off += ids_nbytes;
        for (int32_t expert : pool.expert_ids) {
            if (expert < 0 || expert >= result.n_experts || owners[(size_t)expert] >= 0) {
                throw std::runtime_error("invalid or duplicate MFE expert id");
            }
            owners[(size_t)expert] = pool_index;
        }
        if (version2) {
            if (dtype_nbytes == 0) throw std::runtime_error("empty MFE pool dtype");
            pool.dtype.assign(
                reinterpret_cast<const char *>(blob.data() + off), dtype_nbytes);
            pool.dtype = std::string(
                mfq::canonical_format_dtype(pool.dtype));
            off += dtype_nbytes;
            pool.runtime_payload.assign(
                blob.begin() + (ptrdiff_t)off,
                blob.begin() + (ptrdiff_t)(off + (size_t)runtime_nbytes));
            off += (size_t)runtime_nbytes;
        }
        size_t payload_end = off + (size_t)payload_nbytes;
        pool.payload.assign(
            blob.begin() + (ptrdiff_t)off, blob.begin() + (ptrdiff_t)payload_end);
        off = payload_end;
        if (!version1 && mfq::fp8sq::is_dtype(pool.dtype)) {
            if (!pool.runtime_payload.empty()) {
                throw std::runtime_error(
                    "unexpected MFE FP8-SQ runtime metadata");
            }
            const auto layout = mfq::fp8sq::parse(
                pool.dtype, pool.payload.data(), pool.payload.size());
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (layout.outputs != expected_rows ||
                    layout.width != result.neuron_len) {
                throw std::runtime_error(
                    "MFE FP8-SQ pool weight shape mismatch");
            }
        } else if (!version1 && pool.dtype == "MXFP4-SQ") {
            if (!pool.runtime_payload.empty()) {
                throw std::runtime_error(
                    "unexpected MFE MXFP4-SQ runtime metadata");
            }
            const auto layout = mfq::sq::parse(
                pool.payload.data(), pool.payload.size());
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (layout.outputs != expected_rows ||
                    layout.width != result.neuron_len) {
                throw std::runtime_error(
                    "MFE MXFP4-SQ pool weight shape mismatch");
            }
        } else if (!version1 && pool.dtype == "MXFP4") {
            pool.mxfp4 = unpack_mxfp4(pool.payload);
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (pool.mxfp4.out != expected_rows ||
                    pool.mxfp4.neuron_len != result.neuron_len) {
                throw std::runtime_error(
                    "MFE MXFP4 pool weight shape mismatch");
            }
        } else if (!version1 && pool.dtype == "NINT8-0") {
            pool.q8_zero = unpack_nint8_zero(pool.payload);
            const int expected_rows =
                static_cast<int>(expert_count) * result.out_per_expert;
            if (pool.q8_zero.axis != 0 ||
                pool.q8_zero.out != expected_rows ||
                pool.q8_zero.neuron_len != result.neuron_len ||
                pool.q8_zero.shape.size() != 2 ||
                pool.q8_zero.shape[0] != expected_rows ||
                pool.q8_zero.shape[1] != result.neuron_len) {
                throw std::runtime_error(
                    "MFE NINT8-0 pool weight shape mismatch");
            }
        } else if (version1 || pool.dtype == "NINT") {
            pool.weight = unpack_nint(pool.payload);
            if (version1) pool.dtype = "NINT";
            int expected_rows = (int)expert_count * result.out_per_expert;
            if (pool.weight.axis != 0 || pool.weight.out != expected_rows ||
                pool.weight.neuron_len != result.neuron_len ||
                pool.weight.shape.size() != 2 ||
                pool.weight.shape[0] != expected_rows ||
                pool.weight.shape[1] != result.neuron_len) {
                throw std::runtime_error("MFE pool weight shape mismatch");
            }
        }
        result.pools.push_back(std::move(pool));
    }
    if (off != blob.size() ||
        (!allow_partial &&
         std::find(owners.begin(), owners.end(), -1) != owners.end())) {
        throw std::runtime_error("MFE expert coverage or tail mismatch");
    }
    return result;
}

MfeCpu unpack_mfe(const std::vector<uint8_t> & blob) {
    return unpack_mfe_impl(blob, false);
}

static MfeCpu unpack_mfe_delta(
        const std::vector<uint8_t> & blob) {
    return unpack_mfe_impl(blob, true);
}
static std::atomic<uint64_t> g_moe_route_generation{1};

mfq_tensor_backend::Tensor moe_tensor_to_device(
        ModelParallelCollectiveRuntime& collectives,
        mfq_tensor_backend::Tensor value, int device) {
    return value.defined()
        ? tensor_to_cuda_device(collectives, std::move(value), device)
        : value;
}

static void copy_route_tensor(
        ModelParallelCollectiveRuntime& collectives,
        mfq_tensor_backend::Tensor& destination,
        const mfq_tensor_backend::Tensor& source,
        int device) {
    if (!source.defined()) {
        destination = mfq_tensor_backend::Tensor();
    } else if (source.is_cuda() && source.get_device() == device) {
        destination = source.contiguous();
    } else {
        destination = tensor_to_cuda_device(
            collectives, source, device, std::move(destination));
    }
}

const MoeRoutePlan& moe_route_to_device(
        ModelParallelCollectiveRuntime& collectives,
        const MoeRoutePlan& source,
        int device) {
    auto& replica = (*source.device_replicas)[device];
    if (!replica) replica = std::make_shared<MoeRoutePlan>();
    if (replica->generation == source.generation) return *replica;
    copy_route_tensor(collectives, replica->ids, source.ids, device);
    copy_route_tensor(collectives, replica->ids_dst, source.ids_dst, device);
    copy_route_tensor(collectives, replica->expert_bounds, source.expert_bounds, device);
    copy_route_tensor(collectives, replica->tile_bounds, source.tile_bounds, device);
    copy_route_tensor(collectives, replica->tile_experts, source.tile_experts, device);
    if (source.mma_tile_m == 8) {
        replica->mma_tile_bounds = replica->tile_bounds;
        replica->mma_tile_experts = replica->tile_experts;
    } else {
        copy_route_tensor(collectives, replica->mma_tile_bounds, source.mma_tile_bounds, device);
        copy_route_tensor(collectives, replica->mma_tile_experts, source.mma_tile_experts, device);
    }
    if (source.wide_tile_m == 8) {
        replica->wide_tile_bounds = replica->tile_bounds;
        replica->wide_tile_experts = replica->tile_experts;
    } else if (source.wide_tile_m == source.mma_tile_m) {
        replica->wide_tile_bounds = replica->mma_tile_bounds;
        replica->wide_tile_experts = replica->mma_tile_experts;
    } else {
        copy_route_tensor(collectives, replica->wide_tile_bounds, source.wide_tile_bounds, device);
        copy_route_tensor(collectives, replica->wide_tile_experts, source.wide_tile_experts, device);
    }
    copy_route_tensor(collectives, replica->counts, source.counts, device);
    copy_route_tensor(collectives, replica->cursors, source.cursors, device);
    replica->n_experts = source.n_experts;
    replica->mma_tile_m = source.mma_tile_m;
    replica->wide_tile_m = source.wide_tile_m;
    replica->map_ready = source.map_ready;
    replica->generation = source.generation;
    replica->host_unique_experts = source.host_unique_experts;
    return *replica;
}

MoeRoutePlan build_moe_route_plan(mfq_tensor_backend::Tensor ids, int n_experts) {
    if (!ids.is_cuda() || ids.scalar_type() != mfq_tensor_backend::kInt32 || ids.dim() != 2) {
        throw std::runtime_error("MoE ids must be CUDA int32 [tokens, routes]");
    }
    MoeRoutePlan result;
    result.generation = g_moe_route_generation.fetch_add(
        1, std::memory_order_relaxed);
    result.ids = ids.contiguous();
    result.n_experts = n_experts;
    auto empty = mfq_tensor_backend::empty({0}, result.ids.options());
    result.ids_dst = empty;
    result.expert_bounds = empty;
    result.tile_bounds = empty;
    result.tile_experts = empty;
    result.mma_tile_bounds = empty;
    result.mma_tile_experts = empty;
    result.wide_tile_bounds = empty;
    result.wide_tile_experts = empty;
    result.counts = empty;
    result.cursors = empty;
    if (result.ids.size(0) > 8) {
        const int64_t rows_per_expert = std::max<int64_t>(
            1, (result.ids.numel() + n_experts - 1) / n_experts);
        const bool use_coarse_mma = rows_per_expert >= 4;
        const int mma_tile_m = rows_per_expert <= 16
            ? 16 : (rows_per_expert <= 32 ? 32 : 64);
        auto mapped = !use_coarse_mma
            ? moe_build_expert_map_cuda(result.ids, n_experts, 8)
            : (rows_per_expert > 64
                ? moe_build_expert_maps_cuda(
                    result.ids, n_experts, 8, mma_tile_m, 128)
                : moe_build_expert_maps_cuda(
                    result.ids, n_experts, 8, mma_tile_m, 0));
        result.ids_dst = mapped.at(0);
        result.expert_bounds = mapped.at(1);
        result.tile_bounds = mapped.at(2);
        result.tile_experts = mapped.at(3);
        result.counts = mapped.at(4);
        if (use_coarse_mma) {
            result.mma_tile_bounds = mapped.at(5);
            result.mma_tile_experts = mapped.at(6);
            result.mma_tile_m = mma_tile_m;
            if (rows_per_expert > 64) {
                result.wide_tile_bounds = mapped.at(7);
                result.wide_tile_experts = mapped.at(8);
                result.wide_tile_m = 128;
            } else {
                result.wide_tile_bounds = result.mma_tile_bounds;
                result.wide_tile_experts = result.mma_tile_experts;
                result.wide_tile_m = result.mma_tile_m;
            }
        } else {
            result.mma_tile_bounds = result.tile_bounds;
            result.mma_tile_experts = result.tile_experts;
            result.mma_tile_m = 8;
            result.wide_tile_bounds = result.tile_bounds;
            result.wide_tile_experts = result.tile_experts;
            result.wide_tile_m = 8;
        }
    }
    result.map_ready = result.ids.size(0) <= 8 ||
        result.ids_dst.numel() == result.ids.numel();
    return result;
}

int select_nint_prefill_route_tile(
        const MoeRoutePlan & route,
        int tokens,
        int routes,
        int experts) {
    const int64_t pairs = static_cast<int64_t>(tokens) * routes;
    const int64_t rows_per_expert = std::max<int64_t>(
        1, (pairs + experts - 1) / experts);
    if (rows_per_expert < 4) {
        return 8;
    }
    if (rows_per_expert > 64 && route.wide_tile_m == 128) {
        return 128;
    }
    return route.mma_tile_m == 16 || route.mma_tile_m == 32 ||
            route.mma_tile_m == 64
        ? route.mma_tile_m
        : 8;
}

const mfq_tensor_backend::Tensor & nint_route_tile_bounds(
        const MoeRoutePlan & route, int tile_m) {
    return tile_m == 128
        ? route.wide_tile_bounds
        : (tile_m == 8 ? route.tile_bounds : route.mma_tile_bounds);
}

const mfq_tensor_backend::Tensor & nint_route_tile_experts(
        const MoeRoutePlan & route, int tile_m) {
    return tile_m == 128
        ? route.wide_tile_experts
        : (tile_m == 8 ? route.tile_experts : route.mma_tile_experts);
}
static void initialize_mfe_nint_runtime(MfeWeight & result) {
    if (result.pools.empty()) {
        throw std::runtime_error("MFE NINT runtime requires at least one pool");
    }
    result.activation_geometries.clear();
    result.activation_workspace_domain = 1;
    std::unordered_set<int64_t> quantized_shapes;
    const auto target = result.pools.front().weight.q_packed.device();
    for (const auto & pool : result.pools) {
        MFQ_RUNTIME_CHECK(
            pool.weight.q_packed.is_cuda() &&
            pool.weight.q_packed.device() == target,
            "MFE NINT pools must share one CUDA device");
        const int64_t activation_key =
            (pool.weight.gs << 32) ^ pool.weight.ng;
        if (quantized_shapes.insert(activation_key).second) {
            result.activation_geometries.push_back({
                static_cast<int>(pool.weight.ng),
                static_cast<int>(pool.weight.gs),
                0,
                0,
            });
        }
    }
}

MfeWeight to_gpu_mfe(const MfeCpu & cpu) {
    MfeWeight result;
    result.unified_nint_projection = true;
    result.n_experts = cpu.n_experts;
    result.out_per_expert = cpu.out_per_expert;
    result.neuron_len = cpu.neuron_len;
    result.pools.reserve(cpu.pools.size());
    for (const auto & source_pool : cpu.pools) {
        if (source_pool.dtype != "NINT") {
            throw std::runtime_error("non-NINT cohort reached the NINT-only loader");
        }
        MfePoolWeight pool;
        pool.local_experts = (int)source_pool.expert_ids.size();
        pool.weight = to_device_mfe_nint(
            source_pool.weight, pool.local_experts,
            cpu.out_per_expert, true);
        std::vector<int32_t> local((size_t)cpu.n_experts, -1);
        for (int index = 0; index < pool.local_experts; ++index) {
            const int expert = source_pool.expert_ids.at(static_cast<size_t>(index));
            local[static_cast<size_t>(expert)] = index;
        }
        pool.expert_local = mfq_tensor_backend::from_blob(
            local.data(), {(int64_t)local.size()}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone().to(pool.weight.q_packed.device()).contiguous();
        result.pools.push_back(std::move(pool));
    }
    initialize_mfe_nint_runtime(result);
    return result;
}

void fp8_sq_moe_matmul(
        const Fp8SqWeight & weight,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor expert_ids,
        mfq_tensor_backend::Tensor expert_local,
        int n_experts,
        int local_experts,
        int out_per_expert,
        int neuron_len,
        mfq_tensor_backend::Tensor output) {
    if (weight.dtype == "MXFP8-SQ") {
        mxfp8_sq_moe_matmul_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            std::move(input), std::move(expert_ids),
            std::move(expert_local), n_experts, local_experts,
            out_per_expert, neuron_len, weight.block_rows,
            weight.block_columns, weight.scale_rows, weight.scale_columns,
            weight.palettes_offset, weight.symbols_offset,
            weight.scales_offset, std::move(output));
        return;
    }
    if (weight.dtype == "FP8-128SQ") {
        fp8_128_sq_moe_matmul_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            std::move(input), std::move(expert_ids),
            std::move(expert_local), n_experts, local_experts,
            out_per_expert, neuron_len, weight.scale_kind,
            weight.palettes_offset, weight.symbols_offset,
            weight.scales_offset, std::move(output));
        return;
    }
    throw std::runtime_error("unsupported FP8-SQ MoE dtype");
}

MixedNvqF16FormatGroup mixed_nvq_f16_format_group(int format) {
    switch (format) {
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 10:
        case 11:
            return MixedNvqF16FormatGroup::Standard;
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
            return MixedNvqF16FormatGroup::Extended;
        case 1:
        case 7:
        case 8:
        case 9:
            return MixedNvqF16FormatGroup::Legacy;
        default:
            return MixedNvqF16FormatGroup::All;
    }
}

static void upload_mixed_dispatch_metadata(
        std::initializer_list<std::pair<mfq_tensor_backend::Tensor*,mfq_tensor_backend::Tensor>> tables,
        mfq_tensor_backend::Device device,bool asynchronous,
        mfq_tensor_backend::Tensor& device_owner,mfq_tensor_backend::Tensor& host_owner) {
    namespace tb=mfq_tensor_backend;
    std::size_t bytes=0;
    for(const auto& [target,source]:tables) {
        bytes=(bytes+15)&~std::size_t(15);
        bytes+=source.nbytes();
    }
    // Small metadata packets are rebuilt while previous layers are running.
    // cudaHostAlloc here interrupts that overlap. Keep pageable storage alive
    // through its upload; cross-stream publication still synchronizes below.
    host_owner=tb::empty({int64_t(bytes)},tb::TensorOptions().dtype(tb::kUInt8));
    device_owner=tb::empty({int64_t(bytes)},host_owner.options().device(device).pinned_memory(false));
    std::memset(host_owner.data_ptr(),0,bytes);
    std::size_t offset=0;
    for(const auto& [target,source]:tables) {
        offset=(offset+15)&~std::size_t(15);
        std::memcpy(host_owner.data_ptr<uint8_t>()+offset,source.data_ptr(),source.nbytes());
        *target=tb::from_blob(device_owner.data_ptr<uint8_t>()+offset,source.sizes(),source.options().device(device));
        offset+=source.nbytes();
    }
    const auto stream=mfq_current_cuda_stream();
    MFQ_CUDA_CHECK(cudaMemcpyAsync(device_owner.data_ptr(),host_owner.data_ptr(),bytes,cudaMemcpyHostToDevice,stream));
    // Prefill submits its consumers on this stream and retains both packets.
    // Other callers keep the existing cross-stream initialization guarantee.
    if(!asynchronous)MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void initialize_mixed_nint_dispatch(MixedMoeRuntime& runtime) {
    namespace tb=mfq_tensor_backend;
    runtime.nint_dispatch.reset();runtime.nint_input_plans.clear();
    if(runtime.n_experts<=0 || runtime.n_experts>4096 || runtime.neuron_len<=0 ||
        runtime.out_per_expert<=0 || runtime.out_per_expert>INT_MAX)return;
    auto dispatch=std::make_shared<MixedNintDispatch>();
    std::vector<int64_t> maps;std::vector<int32_t> params;
    tb::Device device(tb::kCUDA,mfq_current_cuda_device());
    for(std::size_t i=0;i<runtime.pools.size();++i) {
        const auto& pool=runtime.pools[i];if(pool.family!=MixedMoeFamily::Nint)continue;
        const auto& w=pool.nint;
        if(!w.q_packed.is_cuda() || pool.local_experts<=0 || w.q8_zero ||
            w.out!=int64_t(pool.local_experts)*runtime.out_per_expert || w.neuron_len!=runtime.neuron_len ||
            w.ng<=0 || w.ng>INT_MAX || w.gs<=0 || w.gs>128 || w.ng*w.gs>INT_MAX ||
            w.ng!=(runtime.neuron_len+w.gs-1)/w.gs ||
            w.q_packed.numel()%pool.local_experts || w.q_packed.numel()/pool.local_experts>INT_MAX)return;
        if(dispatch->pools.empty())device=w.q_packed.device();
        const auto field_ok=[&](const tb::Tensor& field,auto dtype,int64_t count) {
            return field.defined() && field.is_cuda() && field.device()==device && field.is_contiguous() &&
                field.scalar_type()==dtype && field.numel()>=count;
        };
        if(!field_ok(w.q_packed,tb::kUInt8,1) || !field_ok(pool.expert_local,tb::kInt32,runtime.n_experts) ||
            !field_ok(w.row_q_bits,tb::kUInt8,w.out) || !field_ok(w.row_q_bit_offsets,tb::kInt64,w.out) ||
            !field_ok(w.sub_scale,tb::kUInt8,w.out*w.ng) || !field_ok(w.sub_min,tb::kUInt8,w.out*w.ng) ||
            !field_ok(w.neuron_scale,tb::kFloat32,w.out) || !field_ok(w.neuron_min,tb::kFloat32,w.out))return;
        dispatch->pools.push_back(int(i));maps.push_back(reinterpret_cast<int64_t>(pool.expert_local.data_ptr<int32_t>()));
        params.insert(params.end(),{int32_t(w.ng),int32_t(w.gs),int32_t(w.q_packed.numel()/pool.local_experts)});
    }
    if(dispatch->pools.empty())return;
    std::vector<int32_t> missing(runtime.n_experts,-1);
    const auto i32=tb::TensorOptions().dtype(tb::kInt32),i64=tb::TensorOptions().dtype(tb::kInt64);
    upload_mixed_dispatch_metadata({
        {&dispatch->map_ptrs,tb::from_blob(maps.data(),{int64_t(maps.size())},i64)},
        {&dispatch->pool_params,tb::from_blob(params.data(),{int64_t(dispatch->pools.size()),3},i32)},
        {&dispatch->expert_pool,tb::from_blob(missing.data(),{runtime.n_experts},i32)},
        {&dispatch->expert_local,tb::from_blob(missing.data(),{runtime.n_experts},i32)}},
        device,runtime.prefill_only,dispatch->metadata_device,dispatch->metadata_host);
    runtime.nint_dispatch=std::move(dispatch);
}

void initialize_mixed_nvq_dispatch(
        MixedMoeRuntime & runtime,
        const CudaExecutionConfig& config,bool dense_prefill,
        const std::vector<std::vector<int32_t>>* expert_local_host) {
    if(expert_local_host && expert_local_host->size()!=runtime.pools.size())
        throw std::invalid_argument("mixed NVQ host-map pool count mismatch");
    std::unordered_map<const void*,mfq_tensor_backend::Tensor> integer_books;
    for(auto& pool:runtime.pools)if(pool.family==MixedMoeFamily::Nvq) {
        auto& w=pool.nvq;
        prepare_nvq_decode_records(w);
        const auto* key=w.codebook.data_ptr();
        const auto found=integer_books.find(key);
        if(found!=integer_books.end())w.integer_codebook=found->second;
        else {
            prepare_nvq_integer_codebook(w);
            if(w.integer_codebook.defined())integer_books.emplace(key,w.integer_codebook);
        }
    }
    initialize_mixed_nint_dispatch(runtime);
    runtime.nvq_dispatch.reset();
    if (!config.moe_nvq_heterogeneous) return;
    bool has_dense_groups=false;
    for(const auto& pool:runtime.pools)
        if(pool.family==MixedMoeFamily::Nvq && pool.nvq.dense_groups)has_dense_groups=true;
    // Prefill's FP16 weight reader handles compact records directly. Preserve
    // the existing decode dispatch until its separate readers are validated.
    if(has_dense_groups && !dense_prefill)return;

    int nvq_pools = 0;
    for (const auto & pool : runtime.pools) {
        if (pool.family == MixedMoeFamily::Nvq) ++nvq_pools;
    }
    if (nvq_pools == 0) return;

    std::vector<int64_t> weight_ptrs;
    std::vector<int64_t> map_ptrs;
    std::vector<int64_t> weight_sizes;
    std::vector<int32_t> pool_params;
    std::vector<int32_t> expert_pool(
        static_cast<size_t>(runtime.n_experts), -1);
    std::vector<int32_t> expert_local(
        static_cast<size_t>(runtime.n_experts), -1);
    weight_ptrs.reserve(static_cast<size_t>(nvq_pools) * 5);
    weight_sizes.reserve(static_cast<size_t>(nvq_pools) * 3);
    pool_params.reserve(static_cast<size_t>(nvq_pools) * 7);

    mfq_tensor_backend::Device target = mfq_tensor_backend::Device(
        mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
    int dispatch_pool = 0;
    int owned_experts = 0;
    MixedNvqF16FormatGroup f16_format_group =
        MixedNvqF16FormatGroup::All;
    bool first_nvq_format = true;
    for (std::size_t pool_index=0;pool_index<runtime.pools.size();++pool_index) {
        const auto& pool=runtime.pools[pool_index];
        if (pool.family != MixedMoeFamily::Nvq) continue;
        const auto & weight = pool.nvq;
        map_ptrs.push_back(static_cast<int64_t>(reinterpret_cast<uintptr_t>(pool.expert_local.data_ptr<int32_t>())));
        if (weight.gs != 24 || weight.ng <= 0 ||
                weight.ng > std::numeric_limits<int32_t>::max() ||
                weight.sub_bits < 1 || weight.sub_bits > 8 ||
                weight.kernel_format < 1 || weight.kernel_format > 17 ||
                weight.sign_mode < 0 || weight.sign_mode > 1) {
            return;
        }
        if (dispatch_pool == 0) target = weight.indices_packed.device();
        if (weight.indices_packed.device() != target ||
                weight.aux_packed.device() != target ||
                weight.sub_scale_packed.device() != target ||
                weight.neuron_scale.device() != target ||
                weight.codebook.device() != target) {
            return;
        }
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.indices_packed.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.aux_packed.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.sub_scale_packed.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.neuron_scale.data_ptr<float>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.codebook.data_ptr<int8_t>())));
        weight_sizes.push_back(weight.indices_packed.numel());
        weight_sizes.push_back(weight.aux_packed.numel());
        weight_sizes.push_back(weight.sub_scale_packed.numel());
        const int format = static_cast<int>(weight.kernel_format);
        const auto pool_format_group = mixed_nvq_f16_format_group(format);
        if (first_nvq_format) {
            f16_format_group = pool_format_group;
            first_nvq_format = false;
        } else if (f16_format_group != pool_format_group) {
            f16_format_group = MixedNvqF16FormatGroup::All;
        }
        const bool d4 =
            format == 3 || format == 10 || format == 11 ||
            format == 12 || format == 15 || format == 17;
        const int nvec =
            (runtime.neuron_len + (d4 ? 3 : 7)) / (d4 ? 4 : 8);
        const int nsign = (runtime.neuron_len + 7) / 8;
        pool_params.push_back(pool.local_experts);
        pool_params.push_back(static_cast<int32_t>(weight.ng));
        pool_params.push_back(nvec);
        pool_params.push_back(nsign);
        pool_params.push_back(static_cast<int32_t>(weight.sub_bits));
        pool_params.push_back(static_cast<int32_t>(weight.sign_mode));
        pool_params.push_back(format);

        mfq_tensor_backend::Tensor local_host;
        const int32_t* local=nullptr;
        if(expert_local_host) {
            const auto& host=expert_local_host->at(pool_index);
            if(host.size()!=std::size_t(runtime.n_experts))
                throw std::invalid_argument("mixed NVQ host-map expert count mismatch");
            local=host.data();
        } else {
            local_host=pool.expert_local.to(mfq_tensor_backend::kCPU,mfq_tensor_backend::kInt32).contiguous();
            local=local_host.data_ptr<int32_t>();
        }
        for (int expert = 0; expert < runtime.n_experts; ++expert) {
            if (local[expert] < 0) continue;
            if (local[expert] >= pool.local_experts ||
                    expert_pool[static_cast<size_t>(expert)] >= 0) {
                throw std::runtime_error(
                    "mixed NVQ prefill has invalid expert ownership");
            }
            expert_pool[static_cast<size_t>(expert)] = dispatch_pool;
            expert_local[static_cast<size_t>(expert)] = local[expert];
            ++owned_experts;
        }
        ++dispatch_pool;
    }

    auto dispatch = std::make_shared<MixedNvqDispatch>();
    dispatch->pool_count = dispatch_pool;
    dispatch->f16_format_group = f16_format_group;
    dispatch->masked_experts = owned_experts < runtime.n_experts;
    dispatch->dense_groups = has_dense_groups;
    namespace tb=mfq_tensor_backend;
    const auto i32=tb::TensorOptions().dtype(tb::kInt32),i64=tb::TensorOptions().dtype(tb::kInt64);
    upload_mixed_dispatch_metadata({
        {&dispatch->map_ptrs,tb::from_blob(map_ptrs.data(),{int64_t(map_ptrs.size())},i64)},
        {&dispatch->weight_ptrs,tb::from_blob(weight_ptrs.data(),{dispatch_pool,5},i64)},
        {&dispatch->weight_sizes,tb::from_blob(weight_sizes.data(),{dispatch_pool,3},i64)},
        {&dispatch->pool_params,tb::from_blob(pool_params.data(),{dispatch_pool,7},i32)},
        {&dispatch->expert_pool,tb::from_blob(expert_pool.data(),{runtime.n_experts},i32)},
        {&dispatch->expert_local,tb::from_blob(expert_local.data(),{runtime.n_experts},i32)}},
        target,runtime.prefill_only,dispatch->metadata_device,dispatch->metadata_host);
    runtime.nvq_dispatch = std::move(dispatch);
}

static int64_t tensor_storage_bytes(const mfq_tensor_backend::Tensor & value) {
    return value.defined()
        ? value.numel() * (int64_t)value.element_size()
        : 0;
}

int64_t mixed_moe_storage_bytes(const MixedMoeRuntime & runtime) {
    int64_t bytes = 0;
    if(runtime.nint_dispatch) {
        const auto& d=*runtime.nint_dispatch;
        for(const auto& t:{d.map_ptrs,d.pool_params,d.expert_pool,d.expert_local})bytes+=tensor_storage_bytes(t);
    }
    for(const auto& item:runtime.nint_input_plans) {
        bytes+=tensor_storage_bytes(item.second.weight_ptrs)+tensor_storage_bytes(item.second.quantize_descriptors);
    }
    if (runtime.nvq_dispatch) {
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->weight_ptrs);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->weight_sizes);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->pool_params);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->expert_pool);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->expert_local);
    }
    for (const auto & pool : runtime.pools) {
        bytes += tensor_storage_bytes(pool.expert_local);
        if (pool.family == MixedMoeFamily::Nint) {
            bytes += tensor_storage_bytes(pool.nint.q_packed);
            bytes += tensor_storage_bytes(pool.nint.row_q_bits);
            bytes += tensor_storage_bytes(pool.nint.row_q_bit_offsets);
            bytes += tensor_storage_bytes(pool.nint.sub_scale);
            bytes += tensor_storage_bytes(pool.nint.sub_min);
            bytes += tensor_storage_bytes(pool.nint.neuron_scale);
            bytes += tensor_storage_bytes(pool.nint.neuron_min);
        } else if (pool.family == MixedMoeFamily::Nint8Zero) {
            bytes += tensor_storage_bytes(pool.q8_zero.q_packed);
            bytes += tensor_storage_bytes(pool.q8_zero.q8_zero_scale);
        } else if (pool.family == MixedMoeFamily::Mxfp4) {
            bytes += tensor_storage_bytes(pool.mxfp4.values);
            bytes += tensor_storage_bytes(pool.mxfp4.scales);
        } else if (pool.family == MixedMoeFamily::Mxfp4Sq) {
            bytes += tensor_storage_bytes(pool.mxfp4_sq.blob);
            bytes += tensor_storage_bytes(pool.mxfp4_sq.row_q);
            bytes += tensor_storage_bytes(
                pool.mxfp4_sq.row_symbol_byte_offsets);
            bytes += tensor_storage_bytes(pool.mxfp4_sq.row_auxiliary);
        } else if (pool.family == MixedMoeFamily::Fp8Sq) {
            bytes += tensor_storage_bytes(pool.fp8_sq.blob);
            bytes += tensor_storage_bytes(pool.fp8_sq.row_q);
            bytes += tensor_storage_bytes(
                pool.fp8_sq.row_symbol_byte_offsets);
        } else if (pool.family == MixedMoeFamily::Nvq) {
            bytes += tensor_storage_bytes(pool.nvq.indices_packed);
            bytes += tensor_storage_bytes(pool.nvq.aux_packed);
            bytes += tensor_storage_bytes(pool.nvq.sub_scale_packed);
            bytes += tensor_storage_bytes(pool.nvq.neuron_scale);
            bytes += tensor_storage_bytes(pool.nvq.codebook);
            bytes += tensor_storage_bytes(pool.nvq.integer_codebook);
            if(pool.nvq.decode_records.defined() && pool.nvq.decode_records.data_ptr()!=pool.nvq.indices_packed.data_ptr())
                bytes += tensor_storage_bytes(pool.nvq.decode_records);
        } else {
            bytes += tensor_storage_bytes(pool.nepq.indices_packed);
            bytes += tensor_storage_bytes(pool.nepq.aux_packed);
            bytes += tensor_storage_bytes(pool.nepq.state_packed);
            bytes += tensor_storage_bytes(pool.nepq.neuron_scale);
            bytes += tensor_storage_bytes(pool.nepq.table_pool);
            bytes += tensor_storage_bytes(pool.nepq.grouped_table_pool);
            bytes += tensor_storage_bytes(pool.nepq.bank_ids);
            bytes += tensor_storage_bytes(pool.nepq.rotation_signs);
            bytes += tensor_storage_bytes(pool.nepq.residual_codebook);
            bytes += tensor_storage_bytes(pool.nepq.residual_first);
            bytes += tensor_storage_bytes(pool.nepq.residual_second);
        }
    }
    return bytes;
}

std::shared_ptr<MixedMoeRuntime> make_mixed_moe_runtime(
        const MfeCpu & cpu,
        bool cuda,
        const CudaExecutionConfig& config) {
    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu.n_experts;
    runtime->out_per_expert = cpu.out_per_expert;
    runtime->neuron_len = cpu.neuron_len;
    runtime->pools.reserve(cpu.pools.size());
    for (const auto & source : cpu.pools) {
        MixedMoePool pool;
        pool.local_experts = (int)source.expert_ids.size();
        std::vector<int32_t> local((size_t)cpu.n_experts, -1);
        for (int index = 0; index < pool.local_experts; ++index) {
            local[(size_t)source.expert_ids[(size_t)index]] = index;
        }
        pool.expert_local = mfq_tensor_backend::from_blob(
            local.data(), {(int64_t)local.size()},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone();
        if (cuda) {
            const auto target = mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                mfq_current_cuda_device());
            pool.expert_local =
                pool.expert_local.to(target).contiguous();
        }
        const int expected_rows = pool.local_experts * cpu.out_per_expert;
        if (source.dtype == "NINT8-0") {
            pool.family = MixedMoeFamily::Nint8Zero;
            pool.q8_zero = cuda
                ? to_gpu_nint8_zero(source.q8_zero)
                : to_cpu_nint8_zero(source.q8_zero);
            if (pool.q8_zero.out != expected_rows ||
                pool.q8_zero.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed NINT8-0 cohort shape mismatch");
            }
        } else if (source.dtype == "MXFP4") {
            pool.family = MixedMoeFamily::Mxfp4;
            pool.mxfp4 = to_device_mxfp4(source.mxfp4, cuda);
            if (pool.mxfp4.out != expected_rows ||
                    pool.mxfp4.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed MXFP4 cohort shape mismatch");
            }
        } else if (source.dtype == "MXFP4-SQ") {
            pool.family = MixedMoeFamily::Mxfp4Sq;
            pool.mxfp4_sq = to_device_mxfp4_sq(
                source.payload, cuda);
            if (pool.mxfp4_sq.out != expected_rows ||
                    pool.mxfp4_sq.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed MXFP4-SQ cohort shape mismatch");
            }
        } else if (mfq::fp8sq::is_dtype(source.dtype)) {
            pool.family = MixedMoeFamily::Fp8Sq;
            pool.fp8_sq = to_device_fp8_sq(
                source.dtype, source.payload, cuda);
            if (pool.fp8_sq.out != expected_rows ||
                    pool.fp8_sq.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed FP8-SQ cohort shape mismatch");
            }
        } else if (source.dtype == "NINT") {
            pool.family = MixedMoeFamily::Nint;
            pool.nint = to_device_mfe_nint(
                source.weight, pool.local_experts,
                cpu.out_per_expert, cuda);
            if (pool.nint.out != expected_rows ||
                pool.nint.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NINT cohort shape mismatch");
            }
        } else if (source.dtype == "NEPQ") {
            pool.family = MixedMoeFamily::Nepq;
            auto parsed = unpack_nepq(
                source.payload, source.dtype, source.runtime_payload);
            if (parsed.n_experts != pool.local_experts ||
                parsed.out_per_expert != cpu.out_per_expert ||
                parsed.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NEPQ cohort shape mismatch");
            }
            pool.nepq = cuda
                ? to_gpu_nepq(parsed)
                : to_cpu_nepq(parsed);
        } else {
            pool.family = MixedMoeFamily::Nvq;
            auto parsed = unpack_nvq(source.payload, source.dtype);
            if (parsed.out != expected_rows || parsed.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NVQ/NPQ cohort shape mismatch");
            }
            pool.nvq = cuda
                ? to_gpu_nvq(parsed, config)
                : to_cpu_nvq(parsed);
        }
        runtime->pools.push_back(std::move(pool));
    }
    if (cuda) {
        initialize_mixed_nvq_dispatch(*runtime, config);
    }
    return runtime;
}

static MfeWeight wrap_mixed_moe_runtime(
        const std::shared_ptr<MixedMoeRuntime> & runtime) {
    MfeWeight result;
    result.unified_nint_projection = runtime->nint_only();
    result.n_experts = runtime->n_experts;
    result.out_per_expert = runtime->out_per_expert;
    result.neuron_len = runtime->neuron_len;
    result.partial_experts = runtime->partial_experts;
    result.mixed_weight_bytes = mixed_moe_storage_bytes(*runtime);
    result.activation_geometries = runtime->activation_geometry();
    result.activation_workspace_domain = 2;
    result.mixed_forward = [runtime](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x, const MoeRoutePlan & route) {
        return runtime->forward(
            execution.config, execution.kl_mmq,
            execution.force_moe_prefill_mma_off,
            execution.force_moe_pool_path, x, route);
    };
    result.mixed_prequantized_forward = [runtime](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x, const MoeRoutePlan & route) {
        return runtime->forward(
            execution.config, execution.kl_mmq,
            execution.force_moe_prefill_mma_off,
            execution.force_moe_pool_path, x, route, true);
    };
    result.mixed_glu_output_forward = [runtime](
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) {
        return runtime->forward_glu_output(
            execution.config, execution.kl_mmq,
            execution.force_moe_prefill_mma_off,
            execution.force_moe_pool_path, x, route, gelu);
    };
    if (runtime->supports_clamped_swiglu()) {
        result.mixed_clamped_swiglu_forward = [runtime](
                CudaExecutionContext& execution,
                mfq_tensor_backend::Tensor gate_up,
                const MoeRoutePlan & route,
                double limit) {
            return runtime->forward_clamped_swiglu(
                execution.config, execution.kl_mmq,
                execution.force_moe_prefill_mma_off,
                execution.force_moe_pool_path,
                gate_up, route, limit);
        };
    }
    return result;
}

MfeWeight to_gpu_mixed_moe(
        const MfeCpu & cpu,
        const CudaExecutionConfig& config) {
    return wrap_mixed_moe_runtime(
        make_mixed_moe_runtime(cpu, true, config));
}

MfeWeight to_cuda_device_moe_expert_slice(
        const MfeCpu & cpu,
        int64_t expert_begin,
        int64_t expert_end,
        int device,
        const CudaExecutionConfig& config) {
    if (expert_begin < 0 || expert_begin >= expert_end ||
            expert_end > cpu.n_experts) {
        throw std::runtime_error(
            "invalid expert-parallel MoE shard");
    }
    const bool all_nint = std::all_of(
        cpu.pools.begin(), cpu.pools.end(),
        [](const MfeCpuPool & pool) {
            return pool.dtype == "NINT";
        });
    MfqCudaGuard guard(device);
    if (all_nint) {
        MfeCpu sliced = cpu;
        sliced.pools.clear();
        for (const auto & source : cpu.pools) {
            MfeCpuPool destination = source;
            destination.expert_ids.clear();
            std::vector<int64_t> rows;
            for (size_t local = 0;
                 local < source.expert_ids.size(); ++local) {
                const int expert = source.expert_ids[local];
                if (expert < expert_begin || expert >= expert_end) continue;
                destination.expert_ids.push_back(expert);
                for (int row = 0; row < cpu.out_per_expert; ++row) {
                    rows.push_back(
                        static_cast<int64_t>(local) * cpu.out_per_expert + row);
                }
            }
            if (rows.empty()) continue;
            destination.weight = select_nint_cpu_rows(
                source.weight, rows);
            sliced.pools.push_back(std::move(destination));
        }
        auto result = to_gpu_mfe(sliced);
        result.partial_experts = true;
        return result;
    }

    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu.n_experts;
    runtime->out_per_expert = cpu.out_per_expert;
    runtime->neuron_len = cpu.neuron_len;
    runtime->partial_experts = true;
    for (const auto & source : cpu.pools) {
        std::vector<int> expert_ids;
        std::vector<int64_t> rows;
        for (size_t local = 0;
             local < source.expert_ids.size(); ++local) {
            const int expert = source.expert_ids[local];
            if (expert < expert_begin || expert >= expert_end) continue;
            expert_ids.push_back(expert);
            for (int row = 0; row < cpu.out_per_expert; ++row) {
                rows.push_back(
                    static_cast<int64_t>(local) * cpu.out_per_expert + row);
            }
        }
        if (rows.empty()) continue;

        MixedMoePool pool;
        pool.local_experts = static_cast<int>(expert_ids.size());
        std::vector<int32_t> local_map(
            static_cast<size_t>(cpu.n_experts), -1);
        for (size_t local = 0; local < expert_ids.size(); ++local) {
            local_map[static_cast<size_t>(expert_ids[local])] =
                static_cast<int32_t>(local);
        }
        pool.expert_local = mfq_tensor_backend::from_blob(
            local_map.data(),
            {static_cast<int64_t>(local_map.size())},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone().to(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                mfq_current_cuda_device())).contiguous();
        if (source.dtype == "NINT8-0") {
            pool.family = MixedMoeFamily::Nint8Zero;
            pool.q8_zero = to_gpu_nint8_zero(
                select_nint8_zero_cpu_rows(source.q8_zero, rows));
        } else if (source.dtype == "NINT") {
            pool.family = MixedMoeFamily::Nint;
            const auto selected = select_nint_cpu_rows(source.weight, rows);
            pool.nint = to_device_mfe_nint(
                selected, pool.local_experts,
                cpu.out_per_expert, true);
        } else if (source.dtype == "MXFP4") {
            pool.family = MixedMoeFamily::Mxfp4;
            pool.mxfp4 = to_device_mxfp4(
                select_mxfp4_cpu_rows(source.mxfp4, rows), true);
        } else if (source.dtype == "MXFP4-SQ") {
            pool.family = MixedMoeFamily::Mxfp4Sq;
            pool.mxfp4_sq = to_device_mxfp4_sq(
                mfq::sq::select_rows(source.payload, rows), true);
        } else if (mfq::fp8sq::is_dtype(source.dtype)) {
            if (rows.size() != source.expert_ids.size() *
                    static_cast<size_t>(cpu.out_per_expert)) {
                throw std::runtime_error(
                    "FP8-SQ MFE pools must remain matrix-local across devices");
            }
            pool.family = MixedMoeFamily::Fp8Sq;
            pool.fp8_sq = to_device_fp8_sq(
                source.dtype, source.payload, true);
        } else if (source.dtype == "NEPQ") {
            pool.family = MixedMoeFamily::Nepq;
            auto parsed = unpack_nepq(
                source.payload, source.dtype, source.runtime_payload);
            pool.nepq = to_gpu_nepq(select_nepq_cpu_rows(
                parsed, rows, cpu.out_per_expert,
                pool.local_experts));
        } else {
            pool.family = MixedMoeFamily::Nvq;
            auto parsed = unpack_nvq(source.payload, source.dtype);
            pool.nvq = to_gpu_nvq(
                select_nvq_cpu_rows(parsed, rows), config);
        }
        runtime->pools.push_back(std::move(pool));
    }
    if (runtime->pools.empty()) {
        throw std::runtime_error(
            "expert-parallel MoE shard has no owned experts");
    }
    initialize_mixed_nvq_dispatch(*runtime, config);
    return wrap_mixed_moe_runtime(runtime);
}

mfq_tensor_backend::Tensor copy_cpu_weight_to_cuda(
        const mfq_tensor_backend::Tensor & source) {
    return source.defined()
        ? source.to(mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA,
            mfq_current_cuda_device())).contiguous()
        : mfq_tensor_backend::Tensor();
}

static NintWeight copy_cpu_nint_to_cuda(const NintWeight & source) {
    NintWeight result = source;
    result.workspaces.clear();
    result.q_packed = copy_cpu_weight_to_cuda(source.q_packed);
    result.row_q_bits = copy_cpu_weight_to_cuda(source.row_q_bits);
    result.row_q_bit_offsets =
        copy_cpu_weight_to_cuda(source.row_q_bit_offsets);
    result.q8_zero_scale =
        copy_cpu_weight_to_cuda(source.q8_zero_scale);
    result.sub_scale = copy_cpu_weight_to_cuda(source.sub_scale);
    result.sub_min = copy_cpu_weight_to_cuda(source.sub_min);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.neuron_min = copy_cpu_weight_to_cuda(source.neuron_min);
    return result;
}

static NvqWeight copy_cpu_nvq_to_cuda(const NvqWeight & source) {
    NvqWeight result = source;
    result.workspaces.clear();
    result.indices_packed =
        copy_cpu_weight_to_cuda(source.indices_packed);
    result.aux_packed = copy_cpu_weight_to_cuda(source.aux_packed);
    result.sub_scale_packed =
        copy_cpu_weight_to_cuda(source.sub_scale_packed);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.codebook = copy_cpu_weight_to_cuda(source.codebook);
    result.integer_codebook = copy_cpu_weight_to_cuda(source.integer_codebook);
    result.decode_records = source.decode_records.defined() && source.decode_records.data_ptr()==source.indices_packed.data_ptr()
        ? result.indices_packed : copy_cpu_weight_to_cuda(source.decode_records);
    return result;
}

static Mxfp4Weight copy_cpu_mxfp4_to_cuda(const Mxfp4Weight & source) {
    Mxfp4Weight result = source;
    result.values = copy_cpu_weight_to_cuda(source.values);
    result.scales = copy_cpu_weight_to_cuda(source.scales);
    return result;
}

static Mxfp4SqWeight copy_cpu_mxfp4_sq_to_cuda(
        const Mxfp4SqWeight & source) {
    Mxfp4SqWeight result = source;
    result.blob = copy_cpu_weight_to_cuda(source.blob);
    result.row_q = copy_cpu_weight_to_cuda(source.row_q);
    result.row_symbol_byte_offsets =
        copy_cpu_weight_to_cuda(source.row_symbol_byte_offsets);
    result.row_auxiliary =
        copy_cpu_weight_to_cuda(source.row_auxiliary);
    return result;
}

static Fp8SqWeight copy_cpu_fp8_sq_to_cuda(
        const Fp8SqWeight & source) {
    Fp8SqWeight result = source;
    result.blob = copy_cpu_weight_to_cuda(source.blob);
    result.row_q = copy_cpu_weight_to_cuda(source.row_q);
    result.row_symbol_byte_offsets =
        copy_cpu_weight_to_cuda(source.row_symbol_byte_offsets);
    return result;
}

static NepqWeight copy_cpu_nepq_to_cuda(const NepqWeight & source) {
    NepqWeight result = source;
    result.indices_packed =
        copy_cpu_weight_to_cuda(source.indices_packed);
    result.aux_packed = copy_cpu_weight_to_cuda(source.aux_packed);
    result.state_packed =
        copy_cpu_weight_to_cuda(source.state_packed);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.table_pool = copy_cpu_weight_to_cuda(source.table_pool);
    result.grouped_table_pool =
        copy_cpu_weight_to_cuda(source.grouped_table_pool);
    result.bank_ids = copy_cpu_weight_to_cuda(source.bank_ids);
    result.rotation_signs =
        copy_cpu_weight_to_cuda(source.rotation_signs);
    if (source.residual) {
        result.residual_codebook =
            copy_cpu_weight_to_cuda(source.residual_codebook);
        result.residual_first =
            copy_cpu_weight_to_cuda(source.residual_first);
        result.residual_second =
            copy_cpu_weight_to_cuda(source.residual_second);
    }
    return result;
}

MfeWeight stage_cpu_mixed_moe(
        const std::shared_ptr<MixedMoeRuntime> & cpu,
        const CudaExecutionConfig& config) {
    if (!cpu) throw std::runtime_error("missing CPU-offloaded MoE state");
    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu->n_experts;
    runtime->out_per_expert = cpu->out_per_expert;
    runtime->neuron_len = cpu->neuron_len;
    runtime->pools.reserve(cpu->pools.size());
    for (const auto & source : cpu->pools) {
        MixedMoePool pool;
        pool.family = source.family;
        pool.local_experts = source.local_experts;
        pool.expert_local =
            copy_cpu_weight_to_cuda(source.expert_local);
        if (pool.family == MixedMoeFamily::Nint) {
            pool.nint = copy_cpu_nint_to_cuda(source.nint);
        } else if (pool.family == MixedMoeFamily::Nint8Zero) {
            pool.q8_zero = copy_cpu_nint_to_cuda(source.q8_zero);
        } else if (pool.family == MixedMoeFamily::Mxfp4) {
            pool.mxfp4 = copy_cpu_mxfp4_to_cuda(source.mxfp4);
        } else if (pool.family == MixedMoeFamily::Mxfp4Sq) {
            pool.mxfp4_sq = copy_cpu_mxfp4_sq_to_cuda(
                source.mxfp4_sq);
        } else if (pool.family == MixedMoeFamily::Fp8Sq) {
            pool.fp8_sq = copy_cpu_fp8_sq_to_cuda(source.fp8_sq);
        } else if (pool.family == MixedMoeFamily::Nvq) {
            pool.nvq = copy_cpu_nvq_to_cuda(source.nvq);
        } else {
            pool.nepq = copy_cpu_nepq_to_cuda(source.nepq);
        }
        runtime->pools.push_back(std::move(pool));
    }
    initialize_mixed_nvq_dispatch(*runtime, config);
    return wrap_mixed_moe_runtime(runtime);
}

MfeWeight cpu_mixed_moe_metadata(
        const std::shared_ptr<MixedMoeRuntime> & runtime) {
    MfeWeight result;
    result.unified_nint_projection = runtime->nint_only();
    result.n_experts = runtime->n_experts;
    result.out_per_expert = runtime->out_per_expert;
    result.neuron_len = runtime->neuron_len;
    result.mixed_weight_bytes = mixed_moe_storage_bytes(*runtime);
    result.activation_geometries = runtime->activation_geometry();
    result.activation_workspace_domain = 2;
    return result;
}

MfeCpu load_mfe_cpu(
        const mfq::ModelSource & mfq, const std::string & name) {
    const auto& dtype = require_tensor(mfq, name).dtype;
    if (dtype == "NINT") {
        MfeCpuPool pool;
        pool.dtype = "NINT";
        pool.payload = read_tensor(mfq, name);
        pool.weight = unpack_nint(pool.payload);
        const auto& shape = pool.weight.shape;
        MFQ_RUNTIME_CHECK(pool.weight.axis == 0 && shape.size() == 3 &&
            shape[0] > 0 && shape[0] <= std::numeric_limits<int>::max() &&
            shape[1] > 0 && shape[1] <= std::numeric_limits<int>::max() &&
            shape[2] == pool.weight.neuron_len && shape[0] * shape[1] == pool.weight.out,
            "routed NINT tensor must use [experts,out,in]: ", name);
        MfeCpu result;
        result.n_experts = static_cast<int>(shape[0]);
        result.out_per_expert = static_cast<int>(shape[1]);
        result.neuron_len = pool.weight.neuron_len;
        pool.expert_ids.resize(result.n_experts);
        std::iota(pool.expert_ids.begin(), pool.expert_ids.end(), 0);
        pool.weight.shape = {pool.weight.out, pool.weight.neuron_len};
        result.pools.push_back(std::move(pool));
        return result;
    }
    if (dtype != "MFE") {
        throw std::runtime_error("expert tensor must use MFE: " + name);
    }
    return unpack_mfe(read_tensor(mfq, name));
}

std::shared_ptr<MixedMoeRuntime> make_mxfp4_range_runtime(
        const mfq::cuda::MfeMxfp4ExpertStore & store) {
    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = store.num_experts();
    runtime->out_per_expert = store.out_per_expert();
    runtime->neuron_len = store.neuron_len();
    MixedMoePool pool;
    pool.family = MixedMoeFamily::Mxfp4;
    pool.local_experts = store.num_experts();
    pool.mxfp4.out =
        static_cast<int64_t>(store.num_experts()) * store.out_per_expert();
    pool.mxfp4.neuron_len = store.neuron_len();
    std::vector<int32_t> local(static_cast<size_t>(store.num_experts()));
    std::iota(local.begin(), local.end(), int32_t{0});
    pool.expert_local = mfq_tensor_backend::from_blob(
        local.data(),
        {static_cast<int64_t>(local.size())},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
        .clone();
    runtime->pools.push_back(std::move(pool));
    return runtime;
}
mfq_tensor_backend::Tensor mfe_dense_reference(
        const mfq::ModelSource& source,
        const std::string& name,
        mfq_tensor_backend::Tensor input,
        const std::vector<std::int32_t>& expert_ids,
        int tokens,
        int routes,
        bool routed_input) {
    auto cpu = unpack_mfe(read_tensor(source, name));
    auto reference = mfq_tensor_backend::empty(
        {tokens * routes, cpu.out_per_expert},
        input.options().dtype(mfq_tensor_backend::kFloat16));
    for (const auto& pool : cpu.pools) {
        mfq_tensor_backend::Tensor dense_flat;
        int rotation_block = 0;
        mfq_tensor_backend::Tensor rotation_signs;
        if (pool.dtype == "NINT8-0") {
            auto packed = to_gpu_nint8_zero(pool.q8_zero);
            dense_flat = nint8_zero_dequant_cuda(
                packed.q_packed, packed.q8_zero_scale,
                packed.neuron_len);
        } else if (pool.dtype == "NINT") {
            auto packed = to_gpu_nint(pool.weight);
            dense_flat = nint_decode_cuda(
                packed.q_packed, packed.row_q_bits,
                packed.row_q_bit_offsets, packed.sub_scale,
                packed.sub_min, packed.neuron_scale,
                packed.neuron_min, packed.neuron_len, packed.gs);
        } else if (pool.dtype == "MXFP4") {
            dense_flat = dequant_mxfp4_cpu(pool.mxfp4)
                .to(mfq_tensor_backend::kCUDA).contiguous();
        } else if (pool.dtype == "NEPQ") {
            auto packed = to_gpu_nepq(unpack_nepq(
                pool.payload, pool.dtype, pool.runtime_payload));
            dense_flat = nepq_dequant_cuda(
                packed.indices_packed, packed.aux_packed,
                packed.state_packed, packed.neuron_scale,
                packed.table_pool, packed.bank_ids,
                packed.neuron_len, packed.state_bits,
                packed.format);
            if (packed.residual) {
                dense_flat = nepq_sparse_residual_dequant_cuda(
                    packed.residual_codebook,
                    packed.residual_first,
                    packed.residual_second,
                    packed.residual_position_bits,
                    packed.residual_block_vectors,
                    dense_flat.reshape({
                        packed.n_experts * packed.out_per_expert,
                        packed.neuron_len}));
            }
            rotation_block = packed.rotation_block;
            rotation_signs = packed.rotation_signs;
        } else {
            dense_flat = nvq_dequant(to_gpu_nvq(
                unpack_nvq(pool.payload, pool.dtype)));
        }
        auto dense = dense_flat.reshape({
            static_cast<int64_t>(pool.expert_ids.size()),
            cpu.out_per_expert, cpu.neuron_len});
        for (size_t local = 0; local < pool.expert_ids.size(); ++local) {
            const int expert = pool.expert_ids[local];
            std::vector<int64_t> pair_indices;
            std::vector<int64_t> token_indices;
            for (int pair = 0; pair < tokens * routes; ++pair) {
                if (expert_ids[static_cast<size_t>(pair)] == expert) {
                    pair_indices.push_back(pair);
                    token_indices.push_back(pair / routes);
                }
            }
            if (pair_indices.empty()) continue;
            auto pair_index = mfq_tensor_backend::from_blob(
                pair_indices.data(),
                {static_cast<int64_t>(pair_indices.size())},
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64))
                .clone().to(mfq_tensor_backend::kCUDA);
            auto token_index = mfq_tensor_backend::from_blob(
                token_indices.data(),
                {static_cast<int64_t>(token_indices.size())},
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64))
                .clone().to(mfq_tensor_backend::kCUDA);
            auto selected = routed_input
                ? input.reshape({tokens * routes, cpu.neuron_len})
                      .index_select(0, pair_index)
                : input.index_select(0, token_index);
            if (rotation_block != 0) {
                selected = nepq_hadamard_input_cuda(
                    selected.contiguous(), rotation_signs,
                    rotation_block);
            }
            auto expected = mfq_tensor_backend::matmul(
                selected,
                dense.index({static_cast<int64_t>(local)})
                    .transpose(0, 1));
            reference.index_copy_(0, pair_index, expected);
        }
    }
    return reference;
}

mfq_tensor_backend::Tensor materialize_mfe_dense(
        const mfq::ModelSource& source,
        const std::string& name) {
    auto cpu = unpack_mfe(read_tensor(source, name));
    auto dense = mfq_tensor_backend::empty(
        {cpu.n_experts, cpu.out_per_expert, cpu.neuron_len},
        mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCUDA)
            .dtype(mfq_tensor_backend::kFloat16));
    for (const auto& pool : cpu.pools) {
        mfq_tensor_backend::Tensor local_flat;
        if (pool.dtype == "NINT8-0") {
            auto packed = to_gpu_nint8_zero(pool.q8_zero);
            local_flat = nint8_zero_dequant_cuda(
                packed.q_packed, packed.q8_zero_scale,
                packed.neuron_len);
        } else if (pool.dtype == "NINT") {
            auto packed = to_gpu_nint(pool.weight);
            local_flat = nint_decode_cuda(
                packed.q_packed, packed.row_q_bits,
                packed.row_q_bit_offsets, packed.sub_scale,
                packed.sub_min, packed.neuron_scale,
                packed.neuron_min, packed.neuron_len, packed.gs);
        } else if (pool.dtype == "MXFP4") {
            local_flat = dequant_mxfp4_cpu(pool.mxfp4)
                .to(mfq_tensor_backend::kCUDA).contiguous();
        } else {
            throw std::runtime_error(
                "dense MoE reference requires NINT or MXFP4 cohorts");
        }
        auto local = local_flat.reshape({
            static_cast<int64_t>(pool.expert_ids.size()),
            cpu.out_per_expert, cpu.neuron_len});
        auto expert_index = mfq_tensor_backend::from_blob(
            const_cast<int32_t*>(pool.expert_ids.data()),
            {static_cast<int64_t>(pool.expert_ids.size())},
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt32))
            .clone().to(mfq_tensor_backend::kCUDA)
            .to(mfq_tensor_backend::kInt64);
        dense.index_copy_(0, expert_index, local);
    }
    return dense;
}

bool MixedMoeRuntime::nint_only() const {
    return !pools.empty() && std::all_of(
        pools.begin(), pools.end(), [](const MixedMoePool & pool) {
            return pool.family == MixedMoeFamily::Nint;
        });
}

MoeActivationWorkspace MixedMoeRuntime::activation_workspace(
        mfq_tensor_backend::Tensor x, int input_rows, int groups, int gs,
        MixedMoeTransformKey transform, bool stable) const {
    // Shared quantization publishes an exact-row view. Honor that binding
    // before selecting our own scratch, including graph-sized inputs.
    const MixedMoeActivationKey exact{
        input_rows, groups, gs, x.get_device(), transform};
    auto found = activation_workspaces.find(exact);
    if (found != activation_workspaces.end()) return found->second;
    stable = stable && input_rows <= MoeActivationWorkspace::stable_row_limit;
    const int capacity = MoeActivationWorkspace::capacity_rows(input_rows, stable);
    // Permanent capacity buckets bound decode scratch without reserving
    // 2048 rows for a single token. Eager prefill grows independently.
    const MixedMoeActivationKey key{
        stable ? -capacity : 0, groups, gs, x.get_device(), transform};
    auto it = activation_workspaces.find(key);
    if (it == activation_workspaces.end() || it->second.qx.size(0) < input_rows) {
        // Publish both allocations together so an allocation failure cannot
        // leave new Q8 storage paired with an old scale buffer.
        MoeActivationWorkspace workspace;
        workspace.qx = mfq_tensor_backend::empty(
            {capacity, groups * gs}, x.options().dtype(mfq_tensor_backend::kInt8));
        workspace.xscale = mfq_tensor_backend::empty(
            {capacity, groups}, x.options().dtype(mfq_tensor_backend::kFloat32));
        it = activation_workspaces.insert_or_assign(key, std::move(workspace)).first;
    }
    return {it->second.qx.narrow(0, 0, input_rows),
            it->second.xscale.narrow(0, 0, input_rows)};
}

MoeActivationWorkspace MixedMoeRuntime::prefill_activation_workspace(
        mfq_tensor_backend::Tensor x,int input_rows,int groups,int gs,
        MixedMoeTransformKey transform) const {
    return activation_workspace(x, input_rows, groups, gs, transform, false);
}

std::vector<MoeActivationGeometry> MixedMoeRuntime::activation_geometry() const {
    std::vector<MoeActivationGeometry> result;
    for (const auto & pool : pools) {
        int groups = 0;
        int gs = 24;
        int transform_block = 0;
        uint64_t transform_seed = 0;
        if (pool.family == MixedMoeFamily::Nint) {
            groups = static_cast<int>(pool.nint.ng);
            gs = static_cast<int>(pool.nint.gs);
        } else if (pool.family == MixedMoeFamily::Nint8Zero) {
            groups = static_cast<int>(pool.q8_zero.ng);
            gs = 32;
        } else if (pool.family == MixedMoeFamily::Nvq) {
            groups = static_cast<int>(pool.nvq.ng);
            gs = static_cast<int>(pool.nvq.gs);
        } else if (pool.family == MixedMoeFamily::Mxfp4 ||
                pool.family == MixedMoeFamily::Mxfp4Sq ||
                pool.family == MixedMoeFamily::Fp8Sq) {
            continue;
        } else {
            groups = pool.nepq.ng;
            transform_block = pool.nepq.rotation_block;
            transform_seed = pool.nepq.rotation_seed;
        }
        const MoeActivationGeometry geometry{
            groups, gs, transform_block, transform_seed};
        if (std::find(result.begin(), result.end(), geometry) == result.end()) {
            result.push_back(geometry);
        }
    }
    return result;
}

mfq_tensor_backend::Tensor MixedMoeRuntime::forward(
        const CudaExecutionConfig& config,
        KlMmqState& kl_mmq,
        bool force_prefill_mma_off,
        bool force_pool_path,
        mfq_tensor_backend::Tensor x,
        const MoeRoutePlan & route,
        bool input_prequantized,
        int epilogue_mode) const {
    if (epilogue_mode < 0 || epilogue_mode > 2 ||
            (epilogue_mode != 0 &&
             (!nint_only() || (out_per_expert % 2) != 0))) {
        throw std::runtime_error(
            "mixed MFE projection epilogue requires only canonical NINT");
    }
    if (!x.is_cuda() || !x.is_contiguous() ||
        x.scalar_type() != mfq_tensor_backend::kFloat16 ||
        (x.dim() != 2 && x.dim() != 3) || x.size(-1) != neuron_len) {
        throw std::runtime_error(
            "mixed MFE input must be contiguous CUDA f16 with exact K");
    }
    const int tokens = (int)route.ids.size(0);
    const int routes = (int)route.ids.size(1);
    if (route.n_experts != n_experts || x.size(0) != tokens ||
        (x.dim() == 3 && x.size(1) != routes)) {
        throw std::runtime_error("mixed MFE input and route shape mismatch");
    }
    const int input_rows = x.dim() == 3 ? tokens * routes : tokens;
    const bool stable = input_rows <= MoeActivationWorkspace::stable_row_limit;
    const EagerActivationScratch<decltype(activation_workspaces)> eager_scratch{
        activation_workspaces, !stable};
    const int result_width = epilogue_mode == 0
        ? out_per_expert
        : out_per_expert / 2;
    auto output = partial_experts
        ? mfq_tensor_backend::zeros(
            {tokens, routes, result_width},
            x.options().dtype(mfq_tensor_backend::kFloat16))
        : mfq_tensor_backend::empty(
            {tokens, routes, result_width},
            x.options().dtype(mfq_tensor_backend::kFloat16));
    std::unordered_map<
        MixedMoeTransformKey, mfq_tensor_backend::Tensor,
        MixedMoeTransformKeyHash> prepared_inputs;
    const MixedMoeTransformKey identity{};
    prepared_inputs.emplace(identity, x);
    std::unordered_set<
        MixedMoeActivationKey, MixedMoeActivationKeyHash> quantized;
    const bool use_f16_mma =
        config.moe_prefill_mma &&
        !force_prefill_mma_off &&
        (tokens >= config.moe_prefill_mma_min_tokens || prefill_only) &&
        route.map_ready &&
        route.ids_dst.numel() == route.ids.numel();
    const bool use_kl_mmq =
        kl_mmq.mode != KlMmqMode::Default;
    const bool nvq_hetero_prefill_ready = nvq_dispatch &&
        (nvq_dispatch->pool_count > 1 ||
         (out_per_expert >= 128 &&
          static_cast<int64_t>(tokens) * routes >
              static_cast<int64_t>(n_experts) * 16));
    const bool use_nvq_prefill =
        use_f16_mma && tokens>8 && !use_kl_mmq && nvq_hetero_prefill_ready;
    const bool use_nvq_decode =
        !use_f16_mma && !use_kl_mmq && nvq_dispatch && !nvq_dispatch->dense_groups &&
        nvq_dispatch->pool_count > 1 &&
        tokens <= 8 && !prefill_only && !force_pool_path &&
        config.moe_nvq_heterogeneous_decode;
    const bool use_active_nvq=use_nvq_decode && config.moe_nvq_active_decode &&
        route.ids.numel()>0 && route.ids.numel()<=1024;
    mfq_tensor_backend::Tensor active_nvq_plan;
    if(use_active_nvq) {
        if (!route.ids.is_cuda() || !route.ids.is_contiguous() ||
            route.ids.scalar_type() != mfq_tensor_backend::kInt32 ||
            route.ids.get_device() != x.get_device()) {
            throw std::runtime_error("active NVQ routes require contiguous int32 IDs on the input CUDA device");
        }
        const int pairs=int(route.ids.numel());
        auto found=nvq_active_plans.find(pairs);
        if(found==nvq_active_plans.end())found=nvq_active_plans.emplace(pairs,
            mfq_tensor_backend::empty({1+int64_t(pairs)*3},route.ids.options())).first;
        active_nvq_plan=found->second;
        mfq::cuda::moe_plan_nvq_routes(reinterpret_cast<const uint64_t*>(nvq_dispatch->map_ptrs.data_ptr<int64_t>()),
            nvq_dispatch->pool_count,n_experts,route.ids.data_ptr<int32_t>(),pairs,
            active_nvq_plan.data_ptr<int32_t>(),mfq_current_cuda_stream());
        MFQ_CUDA_CHECK(cudaGetLastError());
    } else if(use_nvq_prefill || use_nvq_decode) {
        mfq::cuda::moe_update_nvq_maps(reinterpret_cast<const uint64_t*>(nvq_dispatch->map_ptrs.data_ptr<int64_t>()),
            nvq_dispatch->pool_count,n_experts,nvq_dispatch->expert_pool.data_ptr<int32_t>(),
            nvq_dispatch->expert_local.data_ptr<int32_t>(),mfq_current_cuda_stream());
        MFQ_CUDA_CHECK(cudaGetLastError());
    }
    int nint_pool_phase = 0;
    if (input_prequantized && use_kl_mmq) {
        throw std::runtime_error(
            "mixed prequantized activation reuse is unavailable in KLD MMQ mode");
    }
    if (use_kl_mmq) {
        MFQ_RUNTIME_CHECK(
            route.map_ready && route.ids_dst.numel() == route.ids.numel(),
            "KLD mixed routed FP16 requires the compact route map");
    }

    if (use_nvq_prefill) {
        const int routed_rows_per_expert = std::max(
            1, (tokens * routes + n_experts - 1) / n_experts);
        const bool use_coarse_nvq_tiles =
            routed_rows_per_expert > 64 && route.wide_tile_m > 8;
        const int nvq_tile_m = use_coarse_nvq_tiles
            ? route.wide_tile_m : 8;
        const auto & nvq_tile_bounds = use_coarse_nvq_tiles
            ? route.wide_tile_bounds : route.tile_bounds;
        const auto & nvq_tile_experts = use_coarse_nvq_tiles
            ? route.wide_tile_experts : route.tile_experts;
        nvq_moe_grouped_matmul_hetero_f16_cuda(
            nvq_dispatch->weight_ptrs,
            nvq_dispatch->weight_sizes,
            nvq_dispatch->pool_params,
            nvq_dispatch->expert_pool,
            nvq_dispatch->expert_local,
            x, n_experts, out_per_expert, neuron_len,
            nvq_tile_m, output,
            route.ids_dst, route.expert_bounds,
            nvq_tile_bounds, nvq_tile_experts,
            static_cast<int>(nvq_dispatch->f16_format_group),
            nvq_dispatch->masked_experts);
    }

    if (use_nvq_decode) {
        const int groups = (neuron_len + 23) / 24;
        const MixedMoeActivationKey activation_key{
            input_rows, groups, 24, x.get_device(), identity};
        auto workspace = activation_workspace(
            x, input_rows, groups, 24, identity, stable);
        auto qx = workspace.qx;
        auto xscale = workspace.xscale;
        bool input_quantized = input_prequantized;
        if(use_active_nvq)nvq_moe_grouped_matmul_active_ws_cuda(
            nvq_dispatch->weight_ptrs,nvq_dispatch->weight_sizes,nvq_dispatch->pool_params,
            nvq_dispatch->expert_pool,nvq_dispatch->expert_local,x,route.ids,n_experts,
            out_per_expert,neuron_len,input_quantized,output,qx,xscale,active_nvq_plan);
        else nvq_moe_grouped_matmul_hetero_ws_cuda(
            nvq_dispatch->weight_ptrs,
            nvq_dispatch->weight_sizes,
            nvq_dispatch->pool_params,
            nvq_dispatch->expert_pool,
            nvq_dispatch->expert_local,
            x, route.ids, n_experts, out_per_expert, neuron_len,
            input_quantized, output, qx, xscale);
        quantized.insert(activation_key);
    }

    const bool use_nint_decode=config.moe_nint_heterogeneous_decode && nint_dispatch &&
        !use_f16_mma && !use_kl_mmq && !prefill_only && !force_pool_path && tokens<=8 &&
        int64_t(tokens)*routes<=65535 && epilogue_mode==0;
    if(use_nint_decode) {
        auto found=nint_input_plans.find(input_rows);
        if(found==nint_input_plans.end()) {
            std::vector<int64_t> pointers,quantize;
            std::unordered_set<MixedMoeActivationKey,MixedMoeActivationKeyHash> geometries;
            MixedNintInputPlan plan;
            for(int index:nint_dispatch->pools) {
                const auto& w=pools[index].nint;
                auto workspace=activation_workspace(x,input_rows,int(w.ng),int(w.gs),identity,stable);
                pointers.insert(pointers.end(),{reinterpret_cast<int64_t>(w.q_packed.data_ptr()),
                    reinterpret_cast<int64_t>(w.row_q_bits.data_ptr()),reinterpret_cast<int64_t>(w.row_q_bit_offsets.data_ptr()),
                    reinterpret_cast<int64_t>(w.sub_scale.data_ptr()),reinterpret_cast<int64_t>(w.sub_min.data_ptr()),
                    reinterpret_cast<int64_t>(w.neuron_scale.data_ptr()),reinterpret_cast<int64_t>(w.neuron_min.data_ptr()),
                    reinterpret_cast<int64_t>(workspace.qx.data_ptr()),reinterpret_cast<int64_t>(workspace.xscale.data_ptr())});
                const MixedMoeActivationKey key{input_rows,int(w.ng),int(w.gs),x.get_device(),identity};
                if(geometries.insert(key).second) {
                    quantize.insert(quantize.end(),{reinterpret_cast<int64_t>(workspace.qx.data_ptr()),
                        reinterpret_cast<int64_t>(workspace.xscale.data_ptr()),w.ng,w.gs});
                    plan.total_groups+=int(w.ng);
                }
            }
            plan.weight_ptrs=mfq_tensor_backend::tensor(pointers).reshape({int64_t(nint_dispatch->pools.size()),9}).to(x.device());
            plan.quantize_descriptors=mfq_tensor_backend::tensor(quantize).reshape({int64_t(geometries.size()),4}).to(x.device());
            found=nint_input_plans.emplace(input_rows,std::move(plan)).first;
        }
        auto& plan=found->second;
        const auto row_setting=mfq::cuda::runtime_options::nint_route_row_workspace();
        const auto direct_setting=mfq::cuda::runtime_options::nint_direct_pool();
        const auto group_setting=mfq::cuda::runtime_options::nint_group_dot();
        const auto hint_setting=mfq::cuda::runtime_options::nint_route_hint();
        const auto warp_setting=mfq::cuda::runtime_options::nint_route_warps();
        bool request_rows=row_setting && *row_setting;
        if(!row_setting && tokens==1 && nint_dispatch->pools.size()==1) {
            const auto& w=pools[nint_dispatch->pools.front()].nint;
            if(neuron_len==2560 && out_per_expert==640 && w.gs==28 && w.bits==5 &&
                    nint_late_scale_enabled(neuron_len,int(w.bits),x.get_device())) {
                request_rows=true;
            }
        }
        bool prepare_rows=request_rows && nint_dispatch->pools.size()==1 &&
            (!direct_setting || *direct_setting) && (!group_setting || *group_setting) &&
            (!hint_setting || *hint_setting) && (!warp_setting || *warp_setting!=4);
        if(prepare_rows) {
            const auto& pool=pools[nint_dispatch->pools.front()];const auto& w=pool.nint;
            prepare_rows=(w.gs==24 || w.gs==28) && w.bits>=4 && w.bits<=6;
            if(prepare_rows) {
                if(!plan.row_metadata_workspace.defined())plan.row_metadata_workspace=mfq_tensor_backend::empty(
                    {w.out,4},w.q_packed.options().dtype(mfq_tensor_backend::kInt32));
                moe_quantize_shared_nint_rows_cuda(x.reshape({input_rows,neuron_len}),plan.quantize_descriptors,
                    plan.total_groups,!input_prequantized,w.row_q_bit_offsets,w.neuron_scale,w.neuron_min,
                    plan.row_metadata_workspace,pool.expert_local,route.ids,out_per_expert,pool.local_experts);
            }
        }
        if(!prepare_rows && !input_prequantized)moe_quantize_shared_input_cuda(x.reshape({input_rows,neuron_len}),
            plan.quantize_descriptors,plan.total_groups);
        for(int index:nint_dispatch->pools) {
            const auto& w=pools[index].nint;
            quantized.insert({input_rows,int(w.ng),int(w.gs),x.get_device(),identity});
        }
        bool direct=false;
        if(nint_dispatch->pools.size()==1) {
            const auto& pool=pools[nint_dispatch->pools.front()];const auto& w=pool.nint;
            auto workspace=activation_workspace(x,input_rows,int(w.ng),int(w.gs),identity,stable);
            const auto& row_metadata=prepare_rows?plan.row_metadata_workspace:w.route_metadata;
            const NintSingleRouteWeight view{{reinterpret_cast<int64_t>(w.q_packed.data_ptr()),
                reinterpret_cast<int64_t>(w.row_q_bits.data_ptr()),reinterpret_cast<int64_t>(w.row_q_bit_offsets.data_ptr()),
                reinterpret_cast<int64_t>(w.sub_scale.data_ptr()),reinterpret_cast<int64_t>(w.sub_min.data_ptr()),
                reinterpret_cast<int64_t>(w.neuron_scale.data_ptr()),reinterpret_cast<int64_t>(w.neuron_min.data_ptr()),
                reinterpret_cast<int64_t>(workspace.qx.data_ptr()),reinterpret_cast<int64_t>(workspace.xscale.data_ptr())},
                {int(w.ng),int(w.gs),int(w.q_packed.numel()/pool.local_experts)},pool.local_experts,
                row_metadata.defined()?reinterpret_cast<int64_t>(row_metadata.data_ptr()):0};
            direct=nint_try_single_route_cuda(view,pool.expert_local,route.ids,output,x.dim()==3,int(w.bits),prepare_rows);
        }
        if(!direct) {
        mfq::cuda::moe_update_nvq_maps(reinterpret_cast<const uint64_t*>(nint_dispatch->map_ptrs.data_ptr<int64_t>()),
            int(nint_dispatch->pools.size()),n_experts,nint_dispatch->expert_pool.data_ptr<int32_t>(),
            nint_dispatch->expert_local.data_ptr<int32_t>(),mfq_current_cuda_stream());
        MFQ_CUDA_CHECK(cudaGetLastError());
        const auto& hint=pools[nint_dispatch->pools.front()].nint;
        nint_moe_grouped_matmul_hinted_cuda(plan.weight_ptrs,nint_dispatch->pool_params,
            nint_dispatch->expert_pool,nint_dispatch->expert_local,route.ids,output,neuron_len,x.dim()==3,
            nint_dispatch->pools.size()==1?int(hint.gs):0,int(hint.bits));
        }
    }

    for (const auto & pool : pools) {
        if(pool.family==MixedMoeFamily::Nint && use_nint_decode)continue;
        if (pool.family == MixedMoeFamily::Nvq &&
                (use_nvq_prefill || use_nvq_decode)) {
            continue;
        }
        int gs = 24;
        int groups = 0;
        MixedMoeTransformKey transform{};
        mfq_tensor_backend::Tensor value = x;
        if (pool.family == MixedMoeFamily::Nint) {
            gs = (int)pool.nint.gs;
            groups = (int)pool.nint.ng;
        } else if (pool.family == MixedMoeFamily::Nint8Zero) {
            gs = 32;
            groups = (int)pool.q8_zero.ng;
        } else if (pool.family == MixedMoeFamily::Nvq) {
            gs = (int)pool.nvq.gs;
            groups = (int)pool.nvq.ng;
        } else if (pool.family == MixedMoeFamily::Mxfp4 ||
                pool.family == MixedMoeFamily::Mxfp4Sq ||
                pool.family == MixedMoeFamily::Fp8Sq) {
            groups = 0;
        } else {
            groups = pool.nepq.ng;
            transform = {
                pool.nepq.rotation_block,
                pool.nepq.rotation_seed,
            };
            if (transform.block != 0 && !input_prequantized) {
                auto found = prepared_inputs.find(transform);
                if (found == prepared_inputs.end()) {
                    auto flat = x.reshape({input_rows, neuron_len}).contiguous();
                    auto rotated = nepq_hadamard_input_cuda(
                        flat, pool.nepq.rotation_signs, transform.block);
                    value = x.dim() == 3
                        ? rotated.reshape({tokens, routes, neuron_len})
                        : rotated;
                    prepared_inputs.emplace(transform, value);
                } else {
                    value = found->second;
                }
            }
        }
        if (use_kl_mmq) {
            value = kl_mmq.prepare_activation(value);
            ++kl_mmq.moe_calls;
            if (pool.family == MixedMoeFamily::Mxfp4) {
                mxfp4_moe_grouped_matmul_pool_f16_cuda(
                    pool.mxfp4.values, pool.mxfp4.scales, value,
                    route.ids, pool.expert_local, n_experts,
                    pool.local_experts, out_per_expert, neuron_len,
                    output, route.ids_dst, route.expert_bounds,
                    route.tile_bounds, route.tile_experts);
                continue;
            }
            if (pool.family == MixedMoeFamily::Mxfp4Sq) {
                mxfp4_sq_moe_matmul_cuda(
                    pool.mxfp4_sq.blob,
                    pool.mxfp4_sq.row_q,
                    pool.mxfp4_sq.row_symbol_byte_offsets,
                    pool.mxfp4_sq.row_auxiliary,
                    value, route.ids,
                    pool.expert_local, pool.mxfp4_sq.bits,
                    n_experts, pool.local_experts, out_per_expert,
                    neuron_len, pool.mxfp4_sq.matrix_scale_base,
                    pool.mxfp4_sq.q_sum,
                    pool.mxfp4_sq.sq4_rows,
                    output);
                continue;
            }
            if (pool.family == MixedMoeFamily::Fp8Sq) {
                fp8_sq_moe_matmul(
                    pool.fp8_sq, value, route.ids, pool.expert_local,
                    n_experts, pool.local_experts, out_per_expert,
                    neuron_len, output);
                continue;
            }
            if (pool.family == MixedMoeFamily::Nvq) {
                nvq_moe_grouped_matmul_pool_f16_cuda(
                    pool.nvq.indices_packed, pool.nvq.aux_packed,
                    pool.nvq.sub_scale_packed, pool.nvq.neuron_scale,
                    pool.nvq.codebook, value, pool.expert_local,
                    n_experts, pool.local_experts, out_per_expert,
                    neuron_len, pool.nvq.gs, pool.nvq.sub_bits,
                    pool.nvq.kernel_format, pool.nvq.sign_mode, output,
                    route.ids_dst, route.expert_bounds,
                    route.tile_bounds, route.tile_experts);
                continue;
            }
            if (pool.family == MixedMoeFamily::Nepq) {
                nepq_moe_grouped_matmul_pool_f16_cuda(
                    pool.nepq.indices_packed, pool.nepq.aux_packed,
                    pool.nepq.state_packed, pool.nepq.neuron_scale,
                    pool.nepq.table_pool, pool.nepq.bank_ids, value,
                    pool.expert_local, n_experts, pool.local_experts,
                    out_per_expert, neuron_len, pool.nepq.state_bits,
                    pool.nepq.format, output, route.ids_dst,
                    route.expert_bounds, route.tile_bounds,
                    route.tile_experts);
                if (pool.nepq.residual) {
                    nepq_sparse_residual_grouped_cuda(
                        pool.nepq.residual_codebook,
                        pool.nepq.residual_first,
                        pool.nepq.residual_second,
                        value, route.ids, pool.expert_local,
                        out_per_expert,
                        pool.nepq.residual_position_bits,
                        pool.nepq.residual_block_vectors,
                        output);
                }
                continue;
            }
            ++kl_mmq.fallback_calls;
            throw std::runtime_error(
                "KLD mixed routed FP16 encountered a non-VQ pool");
        }
        if (pool.family == MixedMoeFamily::Mxfp4) {
            mxfp4_moe_grouped_matmul_pool_f16_cuda(
                pool.mxfp4.values, pool.mxfp4.scales, value,
                route.ids, pool.expert_local, n_experts,
                pool.local_experts, out_per_expert, neuron_len,
                output, route.ids_dst, route.expert_bounds,
                route.tile_bounds, route.tile_experts);
            continue;
        }
        if (pool.family == MixedMoeFamily::Mxfp4Sq) {
            mxfp4_sq_moe_matmul_cuda(
                pool.mxfp4_sq.blob,
                pool.mxfp4_sq.row_q,
                pool.mxfp4_sq.row_symbol_byte_offsets,
                pool.mxfp4_sq.row_auxiliary,
                value, route.ids,
                pool.expert_local, pool.mxfp4_sq.bits,
                n_experts, pool.local_experts, out_per_expert,
                neuron_len, pool.mxfp4_sq.matrix_scale_base,
                pool.mxfp4_sq.q_sum,
                pool.mxfp4_sq.sq4_rows,
                output);
            continue;
        }
        if (pool.family == MixedMoeFamily::Fp8Sq) {
            fp8_sq_moe_matmul(
                pool.fp8_sq, value, route.ids, pool.expert_local,
                n_experts, pool.local_experts, out_per_expert,
                neuron_len, output);
            continue;
        }
        if (pool.family == MixedMoeFamily::Nvq && use_f16_mma) {
            nvq_moe_grouped_matmul_pool_f16_cuda(
                pool.nvq.indices_packed, pool.nvq.aux_packed,
                pool.nvq.sub_scale_packed, pool.nvq.neuron_scale,
                pool.nvq.codebook, value, pool.expert_local,
                n_experts, pool.local_experts, out_per_expert,
                neuron_len, pool.nvq.gs, pool.nvq.sub_bits,
                pool.nvq.kernel_format, pool.nvq.sign_mode, output,
                route.ids_dst, route.expert_bounds,
                route.tile_bounds, route.tile_experts);
            continue;
        }
        MixedMoeActivationKey activation_key{
            input_rows, groups, gs, x.get_device(), transform};
        auto workspace = activation_workspace(
            value, input_rows, groups, gs, transform, stable);
        bool input_quantized = input_prequantized ||
            quantized.find(activation_key) != quantized.end();
        bool activation_quantized_after_call = true;
        auto qx = workspace.qx;
        auto xscale = workspace.xscale;
        if (pool.family == MixedMoeFamily::Nint) {
            const int route_tile_m = tokens<=8 ? 8 : select_nint_prefill_route_tile(
                route, tokens, routes, n_experts);
            mfe_nint_matmul_ws_cuda(
                pool.nint.q_packed, pool.nint.row_q_bits,
                pool.nint.row_q_bit_offsets, pool.nint.sub_scale,
                pool.nint.sub_min, pool.nint.neuron_scale,
                pool.nint.neuron_min, value, route.ids,
                pool.expert_local, n_experts, pool.local_experts,
                out_per_expert, gs, epilogue_mode, route.map_ready,
                input_quantized,
                output, qx, xscale, route.ids_dst,
                route.expert_bounds,
                nint_route_tile_bounds(route, route_tile_m),
                nint_route_tile_experts(route, route_tile_m),
                route_tile_m, nint_pool_phase++);
            activation_quantized_after_call =
                input_quantized || route_tile_m == 8;
        } else if (pool.family == MixedMoeFamily::Nint8Zero) {
            const int routed_rows_per_expert = std::max(
                1, (tokens * routes + n_experts - 1) / n_experts);
            const bool use_q8_f16_mma = use_f16_mma &&
                (routed_rows_per_expert >= 5 ||
                 tokens >= n_experts * 4);
            const bool use_coarse_q8_tiles = use_q8_f16_mma &&
                route.mma_tile_m == 64 && routed_rows_per_expert > 32;
            nint8_zero_moe_grouped_matmul_pool_ws_cuda(
                pool.q8_zero.q_packed, pool.q8_zero.q8_zero_scale, value,
                route.ids, pool.expert_local, n_experts,
                pool.local_experts, out_per_expert, route.map_ready,
                input_quantized, use_q8_f16_mma, output,
                qx, xscale,
                route.counts, route.cursors, route.ids_dst,
                route.expert_bounds,
                use_coarse_q8_tiles
                    ? route.mma_tile_bounds : route.tile_bounds,
                use_coarse_q8_tiles
                    ? route.mma_tile_experts : route.tile_experts,
                use_coarse_q8_tiles ? route.mma_tile_m : 8);
        } else if (pool.family == MixedMoeFamily::Nvq) {
            bool decode_done=false;
            if(gs==24 && epilogue_mode==0 &&
               ((pool.nvq.kernel_format==1 && pool.nvq.sub_bits==3) ||
                (pool.nvq.kernel_format==8 && pool.nvq.sub_bits==4))) {
                const auto& w=pool.nvq;Nvq1DecodeView view;
                const std::array<mfq_tensor_backend::Tensor,7> fields={w.indices_packed,w.aux_packed,
                    w.sub_scale_packed,w.neuron_scale,w.codebook,w.decode_records,w.integer_codebook};
                for(int i=0;i<7;++i)if(fields[i].defined())view.pointers[i]=reinterpret_cast<int64_t>(fields[i].data_ptr());
                view.sizes[0]=w.indices_packed.numel();view.sizes[1]=w.aux_packed.numel();
                view.sizes[2]=w.sub_scale_packed.numel();view.sizes[3]=w.decode_records.defined()?w.decode_records.numel():0;
                view.groups=int(w.ng);view.vectors=int((neuron_len+7)/8);
                view.format=int(w.kernel_format);view.local_experts=pool.local_experts;
                decode_done=nvq1_try_decode_cuda(view,value,route.ids,pool.expert_local,qx,xscale,
                    output,n_experts,out_per_expert,input_quantized);
            }
            if(decode_done)activation_quantized_after_call=true;
            else
            nvq_moe_grouped_matmul_pool_ws_cuda(
                pool.nvq.indices_packed, pool.nvq.aux_packed,
                pool.nvq.sub_scale_packed, pool.nvq.neuron_scale,
                pool.nvq.codebook, value, route.ids, pool.expert_local,
                n_experts, pool.local_experts, out_per_expert, neuron_len,
                pool.nvq.gs, pool.nvq.sub_bits, pool.nvq.kernel_format,
                pool.nvq.sign_mode, input_quantized, output,
                qx, xscale, route.ids_dst,
                route.expert_bounds, route.tile_bounds, route.tile_experts);
        } else {
            nepq_moe_grouped_matmul_pool_ws_cuda(
                pool.nepq.indices_packed, pool.nepq.aux_packed,
                pool.nepq.state_packed, pool.nepq.neuron_scale,
                pool.nepq.table_pool, pool.nepq.bank_ids,
                pool.nepq.grouped_table_pool, value, route.ids,
                pool.expert_local, n_experts, pool.local_experts,
                out_per_expert, neuron_len, pool.nepq.state_bits,
                pool.nepq.format, input_quantized, output,
                qx, xscale, route.ids_dst,
                route.expert_bounds, route.tile_bounds, route.tile_experts);
            if (pool.nepq.residual) {
                nepq_sparse_residual_grouped_cuda(
                    pool.nepq.residual_codebook,
                    pool.nepq.residual_first,
                    pool.nepq.residual_second,
                    value, route.ids, pool.expert_local,
                    out_per_expert,
                    pool.nepq.residual_position_bits,
                    pool.nepq.residual_block_vectors,
                    output);
            }
        }
        if (activation_quantized_after_call) {
            quantized.insert(activation_key);
        }
    }
    return output;
}

mfq_tensor_backend::Tensor MixedMoeRuntime::forward_glu_output(
        const CudaExecutionConfig& config,
        KlMmqState& kl_mmq,
        bool force_prefill_mma_off,
        bool force_pool_path,
        mfq_tensor_backend::Tensor x,
        const MoeRoutePlan & route,
        bool gelu) const {
    if (nint_only() && kl_mmq.mode == KlMmqMode::Default) {
        return forward(
            config, kl_mmq, force_prefill_mma_off, force_pool_path,
            x, route, false, gelu ? 2 : 1);
    }
    auto gate_up = forward(
        config, kl_mmq, force_prefill_mma_off, force_pool_path,
        x, route);
    return gelu
        ? moe_geglu_split_cuda(gate_up)
        : moe_swiglu_split_cuda(gate_up);
}

bool MixedMoeRuntime::supports_clamped_swiglu() const {
    return !pools.empty();
}

mfq_tensor_backend::Tensor MixedMoeRuntime::forward_clamped_swiglu(
        const CudaExecutionConfig& config,
        KlMmqState& kl_mmq,
        bool force_prefill_mma_off,
        bool force_pool_path,
        mfq_tensor_backend::Tensor gate_up,
        const MoeRoutePlan & route,
        double limit) const {
    if (!supports_clamped_swiglu() ||
        !gate_up.is_cuda() || !gate_up.is_contiguous() ||
        gate_up.scalar_type() != mfq_tensor_backend::kFloat16 ||
        gate_up.dim() != 3 || gate_up.size(2) != 2 * neuron_len) {
        throw std::runtime_error(
            "mixed MFE clamped SwiGLU input is unsupported");
    }
    const int tokens = static_cast<int>(route.ids.size(0));
    const int routes = static_cast<int>(route.ids.size(1));
    if (route.n_experts != n_experts ||
        gate_up.size(0) != tokens || gate_up.size(1) != routes) {
        throw std::runtime_error(
            "mixed MFE clamped SwiGLU route shape mismatch");
    }
    auto parts = gate_up.split_with_sizes(
        {neuron_len, neuron_len}, -1);
    auto gate = mfq_tensor_backend::clamp_max(parts[0], limit);
    auto up = mfq_tensor_backend::clamp(parts[1], -limit, limit);
    auto activation =
        (mfq_tensor_backend::silu(gate) * up).contiguous();
    return forward(
        config, kl_mmq, force_prefill_mma_off, force_pool_path,
        activation, route);
}
