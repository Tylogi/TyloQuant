#pragma once

#include "mfe_weight.h"

struct MfeCpuPool {
    std::vector<int32_t> expert_ids;
    std::string dtype;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> runtime_payload;
    NintCpu weight;
    Nint8ZeroCpu q8_zero;
    Mxfp4Cpu mxfp4;
    mfq_tensor_backend::Tensor dense;
};

struct MfeCpu {
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<MfeCpuPool> pools;
};

enum class MixedMoeFamily {
    Nint,
    Nint8Zero,
    Mxfp4,
    Mxfp4Sq,
    Fp8Sq,
    Nvq,
    Nepq,
    Dense,
};

struct MixedMoePool {
    MixedMoeFamily family = MixedMoeFamily::Nint;
    NintWeight nint;
    NintWeight q8_zero;
    Mxfp4Weight mxfp4;
    Mxfp4SqWeight mxfp4_sq;
    Fp8SqWeight fp8_sq;
    NvqWeight nvq;
    NepqWeight nepq;
    mfq_tensor_backend::Tensor dense;
    mfq_tensor_backend::Tensor expert_local;
    int local_experts = 0;
};

void fp8_sq_moe_matmul(
    const Fp8SqWeight& weight,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor expert_ids,
    mfq_tensor_backend::Tensor expert_local,
    int n_experts,
    int local_experts,
    int out_per_expert,
    int neuron_len,
    mfq_tensor_backend::Tensor output);

struct MixedMoeTransformKey {
    int block = 0;
    uint64_t seed = 0;

    bool operator==(const MixedMoeTransformKey & other) const {
        return block == other.block && seed == other.seed;
    }
};

struct MixedMoeTransformKeyHash {
    size_t operator()(const MixedMoeTransformKey & key) const {
        return ((size_t)key.block * 1315423911u) ^
            (size_t)(key.seed ^ (key.seed >> 32));
    }
};

struct MixedMoeActivationKey {
    int input_rows = 0;
    int groups = 0;
    int gs = 0;
    int device = 0;
    MixedMoeTransformKey transform;

    bool operator==(const MixedMoeActivationKey & other) const {
        return input_rows == other.input_rows && groups == other.groups &&
            gs == other.gs && device == other.device &&
            transform == other.transform;
    }
};

struct MixedMoeActivationKeyHash {
    size_t operator()(const MixedMoeActivationKey & key) const {
        size_t value = (size_t)key.input_rows;
        value = value * 1315423911u + (size_t)key.groups;
        value = value * 1315423911u + (size_t)key.gs;
        value = value * 1315423911u + (size_t)key.device;
        return value * 1315423911u +
            MixedMoeTransformKeyHash{}(key.transform);
    }
};

enum class MixedNvqF16FormatGroup : int {
    All = 0,
    Standard = 1,
    Extended = 2,
    Legacy = 3,
};

MixedNvqF16FormatGroup mixed_nvq_f16_format_group(int format);

struct MixedNvqDispatch {
    // Retain the packed device allocation and its asynchronous upload source.
    mfq_tensor_backend::Tensor metadata_device,metadata_host;
    mfq_tensor_backend::Tensor map_ptrs;
    mfq_tensor_backend::Tensor weight_ptrs;
    mfq_tensor_backend::Tensor weight_sizes;
    mfq_tensor_backend::Tensor pool_params;
    mfq_tensor_backend::Tensor expert_pool;
    mfq_tensor_backend::Tensor expert_local;
    int pool_count = 0;
    MixedNvqF16FormatGroup f16_format_group =
        MixedNvqF16FormatGroup::All;
    bool masked_experts = false;
    bool dense_groups = false;
};

struct MixedNintDispatch {
    mfq_tensor_backend::Tensor metadata_device,metadata_host;
    mfq_tensor_backend::Tensor map_ptrs,expert_pool,expert_local,pool_params;
    std::vector<int> pools;
};
struct MixedNintInputPlan {
    mfq_tensor_backend::Tensor weight_ptrs,quantize_descriptors;
    mfq_tensor_backend::Tensor row_metadata_workspace;
    int total_groups=0;
};

struct MixedMoeRuntime {
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    bool partial_experts = false;
    // A streamed prompt group is still prefill when a rare expert has <=8
    // rows. Its workspaces are shared by capacity, not cached per row count.
    bool prefill_only = false;
    std::vector<MixedMoePool> pools;
    std::shared_ptr<MixedNvqDispatch> nvq_dispatch;
    std::shared_ptr<MixedNintDispatch> nint_dispatch;
    mutable std::unordered_map<int,MixedNintInputPlan> nint_input_plans;
    mutable std::unordered_map<int,mfq_tensor_backend::Tensor> nvq_active_plans;
    mutable std::unordered_map<
        MixedMoeActivationKey, MoeActivationWorkspace,
        MixedMoeActivationKeyHash> activation_workspaces;

    bool nint_only() const;
    MoeActivationWorkspace activation_workspace(
            mfq_tensor_backend::Tensor x, int input_rows, int groups, int gs,
            MixedMoeTransformKey transform, bool stable = true) const;
    MoeActivationWorkspace prefill_activation_workspace(
            mfq_tensor_backend::Tensor x, int input_rows, int groups, int gs,
            MixedMoeTransformKey transform) const;

    std::vector<MoeActivationGeometry> activation_geometry() const;

    mfq_tensor_backend::Tensor forward(
            const CudaExecutionConfig& config,
            KlMmqState& kl_mmq,
            bool force_prefill_mma_off,
            bool force_pool_path,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool input_prequantized = false,
            int epilogue_mode = 0) const;

    mfq_tensor_backend::Tensor forward_glu_output(
            const CudaExecutionConfig& config,
            KlMmqState& kl_mmq,
            bool force_prefill_mma_off,
            bool force_pool_path,
            mfq_tensor_backend::Tensor x,
            const MoeRoutePlan & route,
            bool gelu) const;

    bool supports_clamped_swiglu() const;

    mfq_tensor_backend::Tensor forward_clamped_swiglu(
            const CudaExecutionConfig& config,
            KlMmqState& kl_mmq,
            bool force_prefill_mma_off,
            bool force_pool_path,
            mfq_tensor_backend::Tensor gate_up,
            const MoeRoutePlan & route,
            double limit) const;
};

MfeCpu unpack_mfe(const std::vector<uint8_t>& blob);
MfeCpu load_mfe_cpu(
    const mfq::ModelSource& source, const std::string& name);
MfeWeight to_gpu_mfe(const MfeCpu& source);
std::shared_ptr<MixedMoeRuntime> make_mixed_moe_runtime(
    const MfeCpu& source,
    bool cuda,
    const CudaExecutionConfig& config = {});
int64_t mixed_moe_storage_bytes(const MixedMoeRuntime& runtime);
void initialize_mixed_nvq_dispatch(MixedMoeRuntime& runtime,const CudaExecutionConfig& config,bool dense_prefill=false,
    const std::vector<std::vector<int32_t>>* expert_local_host=nullptr);
mfq_tensor_backend::Tensor copy_cpu_weight_to_cuda(
    const mfq_tensor_backend::Tensor& source);
MfeWeight to_gpu_mixed_moe(
    const MfeCpu& source,
    const CudaExecutionConfig& config = {});
MfeWeight to_cuda_device_moe_expert_slice(
    const MfeCpu& source,
    int64_t expert_begin,
    int64_t expert_end,
    int device,
    const CudaExecutionConfig& config = {});
std::shared_ptr<MixedMoeRuntime> make_mxfp4_range_runtime(
    const mfq::cuda::MfeMxfp4ExpertStore& store);
std::vector<mfq::TensorParallelSlice> plan_moe_expert_parallel_slices(
    const ParallelConfig& parallel,
    int64_t extent,
    const std::string& name);
