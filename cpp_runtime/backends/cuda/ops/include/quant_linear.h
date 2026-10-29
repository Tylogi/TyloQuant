#pragma once

#include "format.h"
#include "fp8_sq.h"
#include "mx.h"
#include "mxfp4_sq.h"
#include "nint.h"
#include "vq.h"
#include "tq.h"

#include "cuda_execution.h"
#include "mfq_tensor_backend.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct QuantLinear;
struct QuantLinearGroup;
struct DenseLinearGroup;

mfq_tensor_backend::Tensor dense_projection(
    mfq_tensor_backend::Tensor input, const mfq_tensor_backend::Tensor& weight);

bool decode_branch_parallel_enabled(
    const CudaExecutionConfig& config,
    bool serial_branches,
    std::int64_t rows);

mfq_tensor_backend::Tensor run_nint_linear(
    CudaProfiler& profiler, KlMmqState& kl_mmq,
    const NintWeight& weight, mfq_tensor_backend::Tensor input,
    MfqOptional<mfq_tensor_backend::Tensor> gate = mfq_nullopt, int mode = 0);
mfq_tensor_backend::Tensor run_nvq_linear(
    CudaProfiler& profiler, KlMmqState& kl_mmq,
    const NvqWeight& weight, mfq_tensor_backend::Tensor input,
    MfqOptional<mfq_tensor_backend::Tensor> gate = mfq_nullopt, int mode = 0);

mfq_tensor_backend::Tensor tensor_to_cuda_device(
    ModelParallelCollectiveRuntime& collectives,
    mfq_tensor_backend::Tensor value,
    int device,
    mfq_tensor_backend::Tensor reusable = {});
mfq_tensor_backend::Tensor run_quant_linear_shard(
    CudaProfiler& profiler,
    KlMmqState& kl_mmq,
    const struct QuantLinearShard& shard,
    mfq_tensor_backend::Tensor input,
    MfqOptional<mfq_tensor_backend::Tensor> gate = mfq_nullopt,
    int gate_mode = 0);


struct NintLinearGroup {
    NintWeight w;
    std::vector<NintWeight> projection_w;
    std::vector<NintWeight> split_w;
    std::vector<std::vector<int64_t>> split_outs;
    std::vector<int64_t> outs;
    mutable std::shared_ptr<CudaIndependentBranchExecutor>
        branch_executor =
            std::make_shared<CudaIndependentBranchExecutor>();
    std::vector<mfq_tensor_backend::Tensor> forward(
            CudaProfiler& profiler,
            KlMmqState& kl_mmq,
            const CudaExecutionConfig& config,
            bool serial_branches,
            mfq_tensor_backend::Tensor x) const;
    mfq_tensor_backend::Tensor forward_swiglu(
            CudaProfiler& profiler,
            mfq_tensor_backend::Tensor x) const;
    mfq_tensor_backend::Tensor forward_geglu(
            CudaProfiler& profiler,
            mfq_tensor_backend::Tensor x) const;
};

enum class QuantLinearKind {
    Nint,
    Nvq,
    Mxfp4,
    Mxfp4Sq,
    Fp8Sq,
    Mxfp8,
    Dense,
    Tq,
};


struct QuantLinearShard {
    int device = 0;
    int64_t input_begin = 0;
    int64_t input_end = 0;
    int64_t output_begin = 0;
    int64_t output_end = 0;
    QuantLinearKind kind = QuantLinearKind::Nint;
    NintWeight nint;
    NvqWeight nvq;
    Mxfp4Weight mxfp4;
    Mxfp4SqLinear mxfp4_sq;
    Fp8SqLinear fp8_sq;
    Mxfp8Weight mxfp8;
    mfq_tensor_backend::Tensor dense;
};

struct QuantLinear {
    QuantLinearKind kind = QuantLinearKind::Nint;
    NintWeight nint;
    NvqWeight nvq;
    Mxfp4Linear mxfp4;
    Mxfp4SqLinear mxfp4_sq;
    Fp8SqLinear fp8_sq;
    Mxfp8Linear mxfp8;
    mfq_tensor_backend::Tensor dense;
    TqWeight tq;
    TensorParallelAxis tensor_parallel_axis =
        TensorParallelAxis::Mirrored;
    std::vector<QuantLinearShard> tensor_parallel_shards;
    int64_t logical_out = 0;
    int64_t logical_neuron_len = 0;

    bool tensor_parallel() const {
        return !tensor_parallel_shards.empty();
    }

    bool is_nint() const { return kind == QuantLinearKind::Nint; }
    bool is_nvq() const { return kind == QuantLinearKind::Nvq; }
    bool is_mxfp4() const { return kind == QuantLinearKind::Mxfp4; }
    bool is_mxfp4_sq() const { return kind == QuantLinearKind::Mxfp4Sq; }
    bool is_fp8_sq() const { return kind == QuantLinearKind::Fp8Sq; }
    bool is_mxfp8() const { return kind == QuantLinearKind::Mxfp8; }
    bool is_dense() const { return kind == QuantLinearKind::Dense; }
    bool is_tq() const { return kind == QuantLinearKind::Tq; }

    mfq_tensor_backend::Tensor forward_tensor_parallel_flat(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            MfqOptional<mfq_tensor_backend::Tensor> gate,
            int gate_mode) const;

    mfq_tensor_backend::Tensor forward_dense(mfq_tensor_backend::Tensor x) const;

    mfq_tensor_backend::Tensor forward(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const;
    mfq_tensor_backend::Tensor forward_bf16_output(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const;
    mfq_tensor_backend::Tensor forward_mxfp8_groupwise(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor grouped,
            int64_t groups) const;
    mfq_tensor_backend::Tensor forward_input_mul(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor gate,
            int mode) const;
    mfq_tensor_backend::Tensor forward_input_mul_f32_kld(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x,
            mfq_tensor_backend::Tensor gate,
            int mode) const;
    int64_t out() const;
    int64_t neuron_len() const;
};

using QuantLinearProjectionRefs =
    std::vector<const QuantLinear*>;

bool tensor_parallel_output_projections_compatible(
    const QuantLinearProjectionRefs& projections);
std::vector<mfq_tensor_backend::Tensor>
forward_tensor_parallel_output_projections(
    CudaExecutionContext& execution,
    mfq_tensor_backend::Tensor input,
    const QuantLinearProjectionRefs& projections);

struct QuantLinearGroup {
    bool nint_grouped = false;
    bool nvq_prefix2 = false;
    bool decode_branch_parallel = true;
    NintLinearGroup nint;
    TqProjectionGroup tq;
    std::vector<QuantLinear> layers;
    std::vector<int64_t> outs;
    mutable std::shared_ptr<CudaIndependentBranchExecutor>
        branch_executor =
            std::make_shared<CudaIndependentBranchExecutor>();

    QuantLinearProjectionRefs tensor_parallel_output_projections() const;

    bool tensor_parallel_output_compatible() const;

    std::vector<mfq_tensor_backend::Tensor>
    forward_tensor_parallel_output_group(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const;

    std::vector<mfq_tensor_backend::Tensor> forward(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const;
    mfq_tensor_backend::Tensor forward_swiglu(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const;
    mfq_tensor_backend::Tensor forward_geglu(
            CudaExecutionContext& execution,
            mfq_tensor_backend::Tensor x) const;
};

struct DenseLinearGroup {
    mfq_tensor_backend::Tensor w;
    std::vector<int64_t> outs;

    std::vector<mfq_tensor_backend::Tensor> forward(mfq_tensor_backend::Tensor x) const;
};

mfq_tensor_backend::Tensor quant_embedding_lookup(
    const QuantLinear& weight, mfq_tensor_backend::Tensor token_ids);

DenseLinearGroup make_fp32_quant_group(
    CudaExecutionContext& execution,
    QuantLinearGroup group);

mfq_tensor_backend::Tensor nvq_ffn_swiglu_down(
    CudaProfiler& profiler,
    const NvqWeight& gate,
    const NvqWeight& up,
    const NvqWeight& down,
    mfq_tensor_backend::Tensor input,
    MfqOptional<mfq_tensor_backend::Tensor> residual = mfq_nullopt);

bool nvq_fused_residual_format(std::int64_t kernel_format);

mfq_tensor_backend::Tensor nint_matmul_groupwise_u8(
    CudaProfiler& profiler,
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input,
    std::int64_t groups);

bool is_quant_dtype(const std::string& dtype);

QuantLinearGroup make_quant_group(
    CudaExecutionContext& execution,
    std::vector<QuantLinear> layers,
    bool preserve_projection_boundaries = false);

DenseLinearGroup make_dense_group(
    const std::vector<mfq_tensor_backend::Tensor>& weights);

mfq_tensor_backend::Tensor quant_linear_reference_weight(
    const QuantLinear& linear);
