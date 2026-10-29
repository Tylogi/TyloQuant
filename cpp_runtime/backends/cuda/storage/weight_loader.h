#pragma once

#include "quant_linear.h"
#include "mfe_weight.h"
#include "model_source.h"
#include "gated_residual.h"

struct MfqDropFileCacheGuard {
    bool& setting;
    bool previous;

    MfqDropFileCacheGuard(bool enabled, bool& value)
        : setting(value), previous(value) {
        setting = enabled;
    }

    ~MfqDropFileCacheGuard() {
        setting = previous;
    }
};

bool has_tensor(const mfq::ModelSource& source, std::string_view name) noexcept;

std::shared_ptr<mfq::NintRows> load_nint_row_table(
    const mfq::ModelSource& source, const std::string& name, bool prefer_mapping=false);

bool has_tensor_prefix(const mfq::ModelSource& source, std::string_view prefix);

std::vector<std::uint8_t> read_asset(
    const mfq::ModelSource& source, std::string_view name);

std::string read_asset_text(
    const mfq::ModelSource& source, std::string_view name);

MfeWeight load_mfe_gpu(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::string& name,
    bool cacheable = false,
    int layer_id = -1,
    const std::string& projection_role = {});

std::shared_ptr<MixedMoeRuntime> load_mfe_cpu_offloaded(
    const mfq::ModelSource& source,
    const std::string& name);

mfq_tensor_backend::Tensor load_dense_gpu(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::string& name);

// Preserve the stored dtype; load_dense_gpu promotes half-precision floats to F32.
mfq_tensor_backend::Tensor load_dense_cpu(
    const CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::string& name);

mfq_tensor_backend::Tensor load_dense_native_gpu(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::string& name);

QuantLinear load_quant_linear(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::string& name,
    std::optional<TensorParallelAxis> axis_override = std::nullopt,
    const std::vector<mfq::TensorParallelSlice>* slices_override = nullptr);

QuantLinearGroup load_quant_group(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::vector<std::string>& names,
    size_t required_compatible_prefix = 0,
    const std::vector<mfq::TensorParallelSlice>* slices_override = nullptr,
    bool preserve_projection_boundaries = false);

QuantLinearGroup load_paired_gate_up(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    const std::vector<std::string>& names,
    const QuantLinear& down,
    size_t required_compatible_prefix = 2,
    bool preserve_projection_boundaries = false);

mfq_tensor_backend::Tensor mfe_dense_reference(
    const mfq::ModelSource& source,
    const std::string& name,
    mfq_tensor_backend::Tensor input,
    const std::vector<std::int32_t>& expert_ids,
    int tokens,
    int routes,
    bool routed_input);

mfq_tensor_backend::Tensor materialize_mfe_dense(
    const mfq::ModelSource& source,
    const std::string& name);

namespace mfq::cuda::weight_loader {
using Tensor = mfq_tensor_backend::Tensor;
using Linear = std::function<Tensor(CudaExecutionContext&, const Tensor&)>;
struct ResidualLinear {
    Linear forward;
    GatedResidualMixProjection mixed;
    GatedResidualActivationProjection activated;
    std::shared_ptr<const QuantLinear> weight;
    // F32 [column, stream], or same-size vector layout [column/4, stream, 4].
    Tensor prepared_right;
    // Optional immutable GR Up metadata: FP32 coefficients or packed row/group records.
    Tensor up_group_coefficients;
    Tensor operator()(CudaExecutionContext& execution,const Tensor& input)const {
        return forward(execution,input);
    }
    explicit operator bool()const{return bool(forward);}
};
struct Routed {
    using Function=std::function<Tensor(CudaExecutionContext&,const Tensor&,const Tensor&)>;
    Function function;
    std::vector<std::shared_ptr<MfeWeight>> projections;
    std::vector<std::shared_ptr<TqRoutedWeight>> tq_projections;
    Routed()=default;
    template<class Callable, std::enable_if_t<!std::is_same_v<std::decay_t<Callable>,Routed>,int> =0>
    Routed(Callable&& call):function(std::forward<Callable>(call)) {}
    Tensor operator()(CudaExecutionContext& execution,const Tensor& input,const Tensor& ids)const {
        return function(execution,input,ids);
    }
    explicit operator bool()const{return bool(function);}
};

Linear linear(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name);
std::shared_ptr<const QuantLinear> linear_weight(const Linear& function);
using LinearGroup = std::function<std::vector<Tensor>(CudaExecutionContext&, const Tensor&)>;
// Fuse compatible projections while retaining independent TQ storage and
// individual boundaries for GEMM. Unquantized auxiliaries keep their outputs.
LinearGroup grouped_linear(CudaExecutionContext&, std::vector<Linear>& functions,
                           std::size_t joined_prefix=1);

// GR projections retain floating activation math for dense and quantized input.
ResidualLinear residual_linear(CudaExecutionContext&, const mfq::ModelSource&, const std::string& name);

Tensor dense(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name);

Routed routed(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name, int layer,
    int64_t experts, int64_t output, int64_t input, const std::string& role);

Routed routed_gate_up(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& mlp_prefix,
    int layer, int64_t experts, int64_t width, int64_t input, const std::string& role);

void validate_load_options(
        const CudaExecutionContext& execution);

// Bind shared layer definitions to this model's native weight resources.
struct Loader {
    using Tensor = mfq_tensor_backend::Tensor;
    CudaExecutionContext &execution;
    const mfq::ModelSource &source;
    const char *role;

    bool has(const std::string &name) const { return has_tensor(source, name); }
    bool has_prefix(const std::string &prefix) const { return has_tensor_prefix(source, prefix); }
    Tensor dense(const std::string &name) const { return weight_loader::dense(execution, source, name); }
    Linear linear(const std::string &name) const { return weight_loader::linear(execution, source, name); }
    Routed routed(const std::string &name, int layer, int64_t experts, int64_t output, int64_t input) const {
        return weight_loader::routed(execution, source, name, layer, experts, output, input, role);
    }
    Routed routed_gate_up(const std::string &prefix, int layer, int64_t experts, int64_t width,
                         int64_t input) const {
        return weight_loader::routed_gate_up(execution, source, prefix, layer, experts, width, input, role);
    }
    static Tensor fp32(const Tensor &value) { return value.to(mfq_tensor_backend::kFloat32).contiguous(); }
    static auto shape(const Tensor &value) { return value.sizes().vec(); }
    static int64_t elements(const Tensor &value) { return value.numel(); }
};
} // namespace mfq::cuda::weight_loader

namespace mfq::cuda {
template <typename Model>
Model load_causal_lm(CudaExecutionContext &execution, const std::string &model_path,
                     const std::string &config_path, std::int64_t context_size_override = 0,
                     bool load_blocks = true, bool defer_moe_cache_finalize = false,
                     std::shared_ptr<const mfq::ModelSource> source = {}, bool load_input_output = true);

} // namespace mfq::cuda
