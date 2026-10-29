#pragma once

#include "core/block.h"
#include "core/ffn.h"
#include "models/common/weight_loading.h"

#include <memory>
#include <string>
#include <string_view>

namespace mfq { class ModelSource; }

struct FullBlock;
struct TransformerWeightLoader {
    CudaExecutionContext& execution;
    const mfq::ModelSource& source;
    bool preserve_projection_boundaries = false;
    bool cpu_offloaded = false;
    std::size_t shared_gate_up_compatible_prefix = 2;
    mutable std::shared_ptr<const QuantLinear> router_projection;
    bool keep_quantized_router = false;

    bool has(const std::string& name) const;
    mfq_tensor_backend::Tensor dense(const std::string& name) const;
    mfq_tensor_backend::Tensor router_parameter(const std::string& name) const;
    QuantLinear linear(const std::string& name) const;
    QuantLinearGroup projections(const std::vector<std::string>& names) const;
    QuantLinearGroup gate_up(const std::vector<std::string>& names, const QuantLinear& down) const;
    void qkv(FullBlock& block, const std::vector<std::string>& names) const;
    void workspace(FFN& ffn) const;
    void important_neurons(int64_t hidden, int64_t intermediate, FFN& ffn,
        const std::string& down, const std::string& gate, const std::string& up) const;
    void experts(FFN& ffn, mfq::models::ExpertProjection projection,
                 const std::string& name, int layer) const;
    MfeWeight headwise(const std::string& name) const;
    static auto shape(const mfq_tensor_backend::Tensor& value) { return value.sizes().vec(); }
    static int64_t elements(const mfq_tensor_backend::Tensor& value) { return value.numel(); }
};
struct MoeWeightLoadOptions {
    int layer;
    bool cpu_offloaded = false;
    bool router_bias_required = false;
    bool ungated_experts = false;
    bool ungated_shared = false;
    std::size_t shared_gate_up_compatible_prefix = 2;
};

FFN load_moe_weights(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    std::string_view prefix,
    const MoeWeightLoadOptions& options);
void load_important_neuron_branch(
    CudaExecutionContext& execution,
    const mfq::ModelSource& source,
    int64_t hidden_size,
    int64_t intermediate_size,
    FFN& ffn,
    const std::string& down_name,
    const std::string& gate_name,
    const std::string& up_name);
