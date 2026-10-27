#include "transformer_loader.h"
#include "storage/weight_loader.h"
#include "core/full_block.h"
#include "moe.h"

void load_important_neuron_branch(CudaExecutionContext &execution, const mfq::ModelSource &mfq,
                                  int64_t hidden_size, int64_t intermediate_size, FFN &f,
                                  const std::string &down_name, const std::string &gate_name,
                                  const std::string &up_name) {
    const std::string down_high = down_name + ".in_high";
    const std::string gate_high = gate_name + ".in_high";
    const std::string up_high = up_name + ".in_high";
    const bool has_down = has_tensor(mfq, down_high);
    const bool has_gate = has_tensor(mfq, gate_high);
    const bool has_up = has_tensor(mfq, up_high);
    if (!has_down && !has_gate && !has_up) {
        return;
    }
    if (!has_down || !has_gate || !has_up) {
        throw std::runtime_error(
            "important-neuron FFN requires matching gate/up/down .in_high records");
    }
    if (f.is_moe) {
        throw std::runtime_error("important-neuron records are unsupported on routed MoE FFNs");
    }

    auto high = std::make_unique<FFN>();
    high->down = load_quant_linear(execution, mfq, down_high, TensorParallelAxis::Input);
    high->gate_up = load_paired_gate_up(execution, mfq, {gate_high, up_high}, high->down);
    high->geglu = f.geglu;
    high->swiglu_limit = f.swiglu_limit;

    if (f.gate_up.outs.size() != 2 || high->gate_up.outs.size() != 2 ||
        f.gate_up.outs[0] != f.gate_up.outs[1] || high->gate_up.outs[0] != high->gate_up.outs[1] ||
        f.down.out() != hidden_size || high->down.out() != hidden_size ||
        f.down.neuron_len() != f.gate_up.outs[0] ||
        high->down.neuron_len() != high->gate_up.outs[0] ||
        f.down.neuron_len() + high->down.neuron_len() != intermediate_size) {
        throw std::runtime_error("important-neuron FFN tensor shapes disagree with model config");
    }
    f.important_neurons = std::move(high);
}

mfq_tensor_backend::Tensor TransformerWeightLoader::dense(const std::string& name) const {
    return load_dense_gpu(execution, source, name);
}
QuantLinear TransformerWeightLoader::linear(const std::string& name) const {
    return load_quant_linear(execution, source, name);
}
bool TransformerWeightLoader::has(const std::string& name) const { return has_tensor(source, name); }
mfq_tensor_backend::Tensor TransformerWeightLoader::router_parameter(const std::string& name) const {
    return dense(name).to(mfq_tensor_backend::kFloat32).contiguous();
}
QuantLinearGroup TransformerWeightLoader::projections(const std::vector<std::string>& names) const {
    return load_quant_group(execution, source, names);
}
QuantLinearGroup TransformerWeightLoader::gate_up(const std::vector<std::string>& names,
                                                 const QuantLinear& down) const {
    return load_paired_gate_up(execution, source, names, down, shared_gate_up_compatible_prefix,
                              preserve_projection_boundaries);
}
void TransformerWeightLoader::qkv(FullBlock& block, const std::vector<std::string>& names) const {
    const bool mirror_kv = block.attention_output_gate &&
        execution.config.tensor_parallel_mirror_qwen35_attention_kv &&
        is_quant_dtype(require_tensor(source, names[1]).dtype) &&
        is_quant_dtype(require_tensor(source, names[2]).dtype);
    if (mirror_kv) {
        block.split_q_kv_projections = true;
        block.q_projection = linear(names[0]);
        block.k_projection = load_quant_linear(execution, source, names[1], TensorParallelAxis::Mirrored);
        block.v_projection = load_quant_linear(execution, source, names[2], TensorParallelAxis::Mirrored);
    } else {
        block.qkv = load_quant_group(execution, source, names, 2, nullptr,
                                    preserve_projection_boundaries);
    }
}
void TransformerWeightLoader::workspace(FFN& ffn) const { prepare_ffn_workspaces(execution, ffn); }
void TransformerWeightLoader::important_neurons(int64_t hidden, int64_t intermediate, FFN& ffn,
    const std::string& down, const std::string& gate, const std::string& up) const {
    load_important_neuron_branch(execution, source, hidden, intermediate, ffn, down, gate, up);
}
void TransformerWeightLoader::experts(FFN& ffn, mfq::models::ExpertProjection projection,
                                     const std::string& name, int layer) const {
    using Role = mfq::models::ExpertProjection;
    const auto& record = require_tensor(source, name);
    bool homogeneous_dense = record.dtype == "F32" || record.dtype == "F16" || record.dtype == "BF16";
    const auto boundary = name.find(".experts.");
    if (homogeneous_dense && boundary != std::string::npos) {
        const auto prefix = name.substr(0, boundary) + ".experts.";
        for (const auto* role : {"gate", "up", "gate_up", "down"}) {
            const auto* sibling = source.find_tensor(prefix + role + ".weight");
            if (sibling && sibling->dtype != record.dtype) homogeneous_dense = false;
        }
    }
    if (homogeneous_dense) {
        // Source-precision banks stay dense instead of entering an MFE cohort.
        if (cpu_offloaded)
            throw std::runtime_error(
                "CPU-offloaded MoE cannot materialize dense expert tensor " + name + " dtype=" + record.dtype +
                "; quantize experts with NINT/NVQ in MFE");
        auto bank = load_dense_native_gpu(execution, source, name);
        if (bank.dim() != 3)
            throw std::runtime_error(
                "dense routed expert tensor must be [experts,out,in]: " + name);
        auto metadata = MfeWeight{};
        metadata.n_experts = static_cast<int>(bank.size(0));
        metadata.out_per_expert = static_cast<int>(bank.size(1));
        metadata.neuron_len = static_cast<int>(bank.size(2));
        switch (projection) {
            case Role::gate:
                ffn.moe_gate = metadata;
                ffn.moe_dense_gate = std::move(bank);
                break;
            case Role::up:
                ffn.moe_up = metadata;
                ffn.moe_dense_up = std::move(bank);
                break;
            case Role::down:
                ffn.moe_down = metadata;
                ffn.moe_dense_down = std::move(bank);
                break;
            case Role::gate_up:
                ffn.moe_gate_up = metadata;
                ffn.moe_dense_gate_up = std::move(bank);
                break;
        }
        return;
    }
    MfeWeight* weight = nullptr;
    std::shared_ptr<MixedMoeRuntime>* cpu = nullptr;
    const char* role = nullptr;
    switch (projection) {
        case Role::gate: weight = &ffn.moe_gate; cpu = &ffn.cpu_moe_gate; role = "gate"; break;
        case Role::up: weight = &ffn.moe_up; cpu = &ffn.cpu_moe_up; role = "up"; break;
        case Role::gate_up: weight = &ffn.moe_gate_up; cpu = &ffn.cpu_moe_gate_up; role = "gate_up"; break;
        case Role::down: weight = &ffn.moe_down; cpu = &ffn.cpu_moe_down; role = "down"; break;
    }
    if (cpu_offloaded) {
        *cpu = load_mfe_cpu_offloaded(source, name);
        *weight = cpu_mixed_moe_metadata(*cpu);
    } else {
        *weight = load_mfe_gpu(execution, source, name, true, layer, role);
    }
}
MfeWeight TransformerWeightLoader::headwise(const std::string& name) const {
    return load_mfe_gpu(execution, source, name);
}
FFN load_moe_weights(CudaExecutionContext& execution, const mfq::ModelSource& source,
                     std::string_view prefix, const MoeWeightLoadOptions& options) {
    TransformerWeightLoader loader{execution, source};
    loader.cpu_offloaded = options.cpu_offloaded;
    loader.shared_gate_up_compatible_prefix = options.shared_gate_up_compatible_prefix;
    return mfq::models::load_moe_weights<FFN>(
        loader, std::string(prefix), options.layer, options.router_bias_required,
        options.ungated_experts, options.ungated_shared);
}
