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
    if(keep_quantized_router && name.ends_with("router.weight") && require_tensor(source,name).dtype=="TQ") {
        router_projection=std::make_shared<QuantLinear>(linear(name));
        return {};
    }
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
    const auto& record=require_tensor(source,name);
    if(record.dtype=="TQ") {
        MFQ_RUNTIME_CHECK(!cpu_offloaded && projection!=Role::gate_up,
            "TQ routed weights require separate device Gate/Up/Down banks: ",name);
        auto bank=load_tq_routed_weight(source,name,active_weight_load_device(execution),&execution);
        if(!ffn.moe_tq)ffn.moe_tq=std::make_shared<TqMoeBanks>();
        switch(projection) {
            case Role::gate:ffn.moe_tq->gate=std::move(bank);break;
            case Role::up:ffn.moe_tq->up=std::move(bank);break;
            case Role::down:ffn.moe_tq->down=std::move(bank);break;
            case Role::gate_up:break;
        }
        return;
    }
    if(record.dtype=="F32" || record.dtype=="F16" || record.dtype=="BF16") {
        MFQ_RUNTIME_CHECK(!cpu_offloaded && projection!=Role::gate_up,
            "dense routed weights require separate device Gate/Up/Down banks: ",name);
        auto bank=load_dense_native_gpu(execution,source,name);
        MFQ_RUNTIME_CHECK(bank.dim()==3,"dense expert bank must be [experts,out,in]: ",name);
        switch(projection) {
            case Role::gate:ffn.moe_dense_gate=std::move(bank);break;
            case Role::up:ffn.moe_dense_up=std::move(bank);break;
            case Role::down:ffn.moe_dense_down=std::move(bank);break;
            case Role::gate_up:break;
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
    loader.keep_quantized_router=true;
    loader.cpu_offloaded = options.cpu_offloaded;
    loader.shared_gate_up_compatible_prefix = options.shared_gate_up_compatible_prefix;
    auto result=mfq::models::load_moe_weights<FFN>(
        loader, std::string(prefix), options.layer, options.router_bias_required,
        options.ungated_experts, options.ungated_shared);
    if(result.moe_tq) {
        const auto& banks=*result.moe_tq;
        MFQ_RUNTIME_CHECK(banks.up && banks.down && (options.ungated_experts || banks.gate),
            "TQ expert banks cannot be mixed with legacy routed formats");
        MFQ_RUNTIME_CHECK(banks.up->experts==banks.down->experts &&
            banks.up->out==banks.down->input && banks.up->input==banks.down->out &&
            (!banks.gate || (banks.gate->experts==banks.up->experts &&
                banks.gate->out==banks.up->out && banks.gate->input==banks.up->input)),
            "TQ Gate/Up/Down expert dimensions disagree");
    }
    result.moe_router_projection=std::move(loader.router_projection);
    return result;
}
