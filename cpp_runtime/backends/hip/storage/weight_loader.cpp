#include "mfq_weight_memory.h"
#include "hip/hip_runtime.h"
#include "weight_loader.h"
#include "float_projection.h"
#include "selected_attention.h"
#include "qwen4_exp.h"
#include <limits>
#include <cstdlib>
#include <optional>
#include <fstream>
#include <iterator>
#include "moe.h"
#include "mfq_hip_moe_ops.h"
#include "moe_expert_cache.h"
#include "mfe_expert_store.h"
#include "moe_quant_range_source.h"

std::shared_ptr<mfq::NintRows> load_nint_row_table(
        const mfq::ModelSource& source, const std::string& name, bool prefer_mapping) {
    const auto& metadata = require_tensor(source, name);
    MFQ_RUNTIME_CHECK(metadata.dtype == "NINT", "NINT row table requires packed NINT");
    MFQ_RUNTIME_CHECK(metadata.nbytes <= std::numeric_limits<std::size_t>::max(),
        "NINT row table size overflow");
    auto read = source.tensor_reader(name);
    bool parallel=source.supports_parallel_tensor_reads();
    const auto* batched=std::getenv("MFQ_PLE_BATCH_READ");
    auto batch=(!batched || batched[0]!='0') ? source.tensor_batch_reader(name) : mfq::ModelSource::TensorBatchReader{};
    bool mapped_file=false;
    const auto* mapped_setting=std::getenv("MFQ_PLE_FILE_MAP");
    const std::optional<bool> mapped=mapped_setting?std::optional<bool>(mapped_setting[0]=='1'):std::nullopt;
    if(mapped.value_or(prefer_mapping && source.file_read_stats().mode!=mfq::FileReadMode::Direct)) {
        if(auto mapping=source.mapped_tensor_batch_reader(name)) {
            batch=std::move(mapping);
            read=[batch](uint64_t offset,std::byte* out,std::size_t count){batch({{offset,out,count}});};
            parallel=true;mapped_file=true;
        } else if(auto mapping=source.mapped_tensor_reader(name)) {
            read=std::move(mapping);parallel=true;batch={};mapped_file=true;
        }
    }
    if(const auto* trace=std::getenv("MFQ_TRACE_PLE_TIMINGS");trace && trace[0]=='1')
        std::fprintf(stderr,"nint_row_source mapped_file=%d nbytes=%llu name=%s\n",int(mapped_file),
            static_cast<unsigned long long>(metadata.nbytes),name.c_str());
    auto table=std::make_shared<mfq::NintRows>(static_cast<std::size_t>(metadata.nbytes),
        [read = std::move(read)](std::size_t offset, std::uint8_t* out, std::size_t count) {
            read(offset, reinterpret_cast<std::byte*>(out), count);
        }, parallel,std::move(batch));
    mfq::cuda::observe_streamed_ple(table,static_cast<std::size_t>(metadata.nbytes));
    return table;
}

MfeWeight load_mfe_gpu(
        CudaExecutionContext& execution,
        const mfq::ModelSource & mfq, const std::string & name,
        bool cacheable,
        int layer_id,
        const std::string & projection_role) {
    const mfq::cuda::WeightAllocationScope weight_memory;
    auto& cache = execution.moe_expert_cache;
    if (cache && cacheable &&
            !moe_parallel_config(execution).enabled() &&
            execution.config.moe_ssd_ranges) {
        const auto & record = require_tensor(mfq, name);
        mfq::ModelSource::TensorReader reader;
        try {
            reader=mfq.tensor_reader(name);
        } catch (const mfq::TensorReaderUnsupported&) {
            // Sources without retained readers keep the existing CPU loader.
        }
        if (reader) {
            try {
                auto store =
                    std::make_shared<mfq::cuda::MfeMxfp4ExpertStore>(
                        mfq::cuda::MfqRecordRange{
                            name,
                            record.dtype,
                            {},
                            0,
                            record.nbytes,
                            [reader](
                                    std::uint64_t offset,
                                    std::span<std::uint8_t> destination) {
                                reader(offset,
                                    reinterpret_cast<std::byte*>(destination.data()),
                                    destination.size());
                            },
                        });
                auto runtime = make_mxfp4_range_runtime(*store);
                return cache_moe_weight(
                    cache,
                    name,
                    runtime,
                    std::min(
                        execution.moe_cache_registration_min_slots,
                        runtime->n_experts),
                    layer_id,
                    projection_role, std::move(store));
            } catch (const mfq::cuda::MfeMxfp4Unsupported &) {
                try {
                    auto store=std::make_shared<mfq::MfeQuantExpertStore>(record.nbytes,
                        [reader](std::size_t offset,std::uint8_t* output,std::size_t bytes) {
                            reader(offset,reinterpret_cast<std::byte*>(output),bytes);
                        });
                    const int minimum=std::min(execution.moe_cache_registration_min_slots,store->num_experts());
                    return cache_quant_moe_weight(cache,name,std::make_shared<MoeQuantRangeSource>(std::move(store)),
                        minimum,layer_id,projection_role);
                } catch (const mfq::MfeQuantRangeUnsupported&) {
                    // Other canonical pool families retain the existing CPU path.
                }
            }
        }
    }
    auto cpu = load_mfe_cpu(mfq, name);
    const bool has_matrix_local_sq = std::any_of(
        cpu.pools.begin(), cpu.pools.end(), [](const MfeCpuPool & pool) {
            return pool.dtype == "MXFP4-SQ" ||
                mfq::fp8sq::is_dtype(pool.dtype);
        });
    if (moe_parallel_config(execution).enabled()) {
        auto slices = plan_moe_expert_parallel_slices(
            moe_parallel_config(execution), cpu.n_experts, name);
        MfeWeight result;
        result.n_experts = cpu.n_experts;
        result.out_per_expert =
            cpu.out_per_expert;
        result.neuron_len =
            cpu.neuron_len;
        for (const auto & slice : slices) {
            auto shard =
                std::make_shared<MfeWeight>(
                    to_cuda_device_moe_expert_slice(
                        cpu, slice.begin,
                        slice.end,
                        slice.device,
                        execution.config));
            result.expert_parallel_shards.push_back({
                slice.device,
                slice.begin,
                slice.end,
                std::move(shard),
            });
        }
        return result;
    }
    if (cache && cacheable && !has_matrix_local_sq) {
        auto runtime =
            make_mixed_moe_runtime(cpu, false);
        return cache_moe_weight(
            cache,
            name, runtime,
            std::min(
                execution.moe_cache_registration_min_slots,
                runtime->n_experts),
            layer_id,
            projection_role);
    }
    const bool all_nint = std::all_of(
        cpu.pools.begin(), cpu.pools.end(), [](const MfeCpuPool & pool) {
            return pool.dtype == "NINT";
        });
    return all_nint
        ? to_gpu_mfe(cpu)
        : to_gpu_mixed_moe(cpu, execution.config);
}

std::shared_ptr<MixedMoeRuntime> load_mfe_cpu_offloaded(
        const mfq::ModelSource & mfq, const std::string & name) {
    return make_mixed_moe_runtime(load_mfe_cpu(mfq, name), false);
}

namespace mfq::cuda::weight_loader {
namespace tb = mfq_tensor_backend;
namespace {
struct LoadedLinear {
    std::shared_ptr<QuantLinear> weight;
    Tensor operator()(CudaExecutionContext& execution,const Tensor& x)const{return weight->forward(execution,x);}
};
}

Linear linear(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name) {
    auto weight=std::make_shared<QuantLinear>(load_quant_linear(execution, file,name));
    return LoadedLinear{std::move(weight)};
}
std::shared_ptr<const QuantLinear> linear_weight(const Linear& function) {
    const auto* loaded=function.target<LoadedLinear>();return loaded?loaded->weight:nullptr;
}

LinearGroup grouped_linear(CudaExecutionContext& execution, std::vector<Linear>& functions) {
    if(functions.empty() || execution.loading_cpu_layer)return {};
    std::vector<int> tq_indices;std::vector<std::shared_ptr<const QuantLinear>> tq_weights;
    for(int i=0;i<int(functions.size());++i) {
        auto weight=linear_weight(functions[i]);
        if(weight && weight->is_tq() && !weight->tensor_parallel()) {
            if(!tq_weights.empty() && weight->neuron_len()!=tq_weights.front()->neuron_len())return {};
            tq_indices.push_back(i);tq_weights.push_back(std::move(weight));
        }
    }
    if(tq_weights.size()>1 && tq_weights.size()<=4) {
        return [separate=functions,tq_indices=std::move(tq_indices),tq_weights=std::move(tq_weights)](
            CudaExecutionContext& context,const Tensor& input) {
            std::vector<Tensor> outputs(separate.size());
            if(!tq_q8_enabled() && input.is_cuda() && input.numel()==input.size(-1)) {
                std::vector<const TqWeight*> weights;
                for(const auto& weight:tq_weights)weights.push_back(&weight->tq);
                auto fused=tq_forward_group(weights,input);
                for(int i=0;i<int(tq_indices.size());++i)outputs[tq_indices[i]]=std::move(fused[i]);
            }
            for(int i=0;i<int(separate.size());++i)if(!outputs[i].defined())outputs[i]=separate[i](context,input);
            return outputs;
        };
    }
    if(!execution.config.diagnostic_nint_group)return {};
    std::vector<QuantLinear> weights;
    weights.reserve(functions.size());
    for (const auto& function : functions) {
        const auto weight = linear_weight(function);
        if (!weight || !weight->is_nint() || weight->tensor_parallel() || !weight->nint.q_packed.is_cuda()) return {};
        if (!weights.empty() && weight->neuron_len() != weights.front().neuron_len()) return {};
        weights.push_back(*weight);
    }
    auto group = std::make_shared<QuantLinearGroup>(make_quant_group(execution, weights));
    if (!group->nint_grouped) return {};
    std::vector<Linear> separate;
    separate.reserve(weights.size());
    size_t matrix_index = 0;
    int64_t row = 0, packed_bytes = 0;
    for (const auto& original : weights) {
        const auto& matrix = group->nint.split_w.empty() ? group->nint.w : group->nint.split_w.at(matrix_index);
        MFQ_RUNTIME_CHECK(row + original.out() <= matrix.out && original.nint.ng == matrix.ng &&
            original.nint.gs == matrix.gs && original.nint.q8_zero == matrix.q8_zero,
            "grouped projection storage geometry mismatch");
        auto weight = std::make_shared<QuantLinear>(original);
        auto& view = weight->nint;
        view.workspaces.clear();
        view.q_packed = matrix.q_packed;
        if (view.q8_zero) {
            view.q_packed = matrix.q_packed.narrow(0, row, view.out);
            view.q8_zero_scale = matrix.q8_zero_scale.narrow(0, row, view.out);
        }
        else {
            view.row_q_bits = matrix.row_q_bits.narrow(0, row, view.out);
            view.row_q_bit_offsets = matrix.row_q_bit_offsets.narrow(0, row, view.out);
            view.sub_scale = matrix.sub_scale.narrow(0, row, view.out);
            view.sub_min = matrix.sub_min.narrow(0, row, view.out);
            view.neuron_scale = matrix.neuron_scale.narrow(0, row, view.out);
            view.neuron_min = matrix.neuron_min.narrow(0, row, view.out);
            view.aligned_q8 = original.nint.aligned_q8 && (packed_bytes % 4 == 0);
        }
        separate.emplace_back(LoadedLinear{std::move(weight)});
        row += original.out();
        packed_bytes += original.nint.q_packed.numel();
        if (row == matrix.out) {++matrix_index;row = 0;packed_bytes = 0;}
    }
    MFQ_RUNTIME_CHECK(row == 0 && matrix_index == (group->nint.split_w.empty() ? 1 : group->nint.split_w.size()),
        "grouped projection storage is not fully covered");
    functions = separate;
    return [group=std::move(group), separate=std::move(separate)](CudaExecutionContext& context, const Tensor& input) {
        if (input.is_cuda() && context.kl_mmq.mode == KlMmqMode::Default &&
            input.numel() / input.size(-1) <= 8) return group->forward(context, input);
        std::vector<Tensor> outputs;
        outputs.reserve(separate.size());
        for (const auto& projection : separate) outputs.push_back(projection(context, input));
        return outputs;
    };
}

ResidualLinear residual_linear(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name) {
    const mfq::cuda::WeightAllocationScope weight_memory;
    auto weight = std::make_shared<QuantLinear>(load_quant_linear(execution, file, name));
    MFQ_RUNTIME_CHECK(weight->is_dense() || weight->is_nint() || weight->is_tq(),
        "floating residual projection requires dense, NINT or TQ weights: ", name);
    if(weight->is_tq()) {
        Linear forward=[weight](CudaExecutionContext& execution,const Tensor& input) {
            return weight->forward(execution,input.scalar_type()==tb::kFloat16?input:input.to(tb::kFloat32));
        };
        GatedResidualMixProjection mixed=[weight](CudaExecutionContext&,const Tensor& input,const Tensor& normalized,int64_t streams) {
            return tq_gr_mix(weight->tq,input,normalized,streams);
        };
        GatedResidualActivationProjection activated=[weight](CudaExecutionContext&,const Tensor& input,int64_t streams,bool injection) {
            return tq_gr_activation(weight->tq,input,streams,injection);
        };
        return {std::move(forward),std::move(mixed),std::move(activated),std::move(weight)};
    }
    if(weight->is_dense()) {
        auto right=weight->dense.transpose(-1,-2);
        auto promoted=execution.config.gr_prepared_dense_projection?right.to(tb::kFloat32):Tensor{};
        const bool prepare_injection=execution.config.gr_two_stage_dense_injection &&
            weight->dense.scalar_type()==tb::kBFloat16 && weight->dense.is_contiguous() &&
            weight->dense.dim()==2 && weight->dense.size(0)>0 && weight->dense.size(0)<=4;
        auto prepared_right=prepare_injection?(promoted.defined()?promoted:right.to(tb::kFloat32)):Tensor{};
        Linear forward=[right=std::move(right),promoted=std::move(promoted)](
                CudaExecutionContext& execution,const Tensor& input) {
            const auto dtype=input.scalar_type()==right.scalar_type()?input.scalar_type():tb::kFloat32;
            if(execution.config.gr_prepared_dense_projection && promoted.defined() && dtype==tb::kFloat32)
                return tb::matmul(input.to(tb::kFloat32),promoted);
            return mfq_selected_attention::promoted_matmul(input,right);
        };
        GatedResidualActivationProjection activated;
        if(prepare_injection)activated=[weight,forward](CudaExecutionContext& execution,const Tensor& input,
                int64_t streams,bool injection) {
            if(injection && input.is_cuda() && input.is_contiguous() && input.scalar_type()==tb::kFloat32 &&
               input.size(-1)==weight->dense.size(1) && input.numel()/input.size(-1)<=8)
                return dense_gr_injection_cuda(weight->dense,input,streams);
            auto value=forward(execution,input);
            return injection?mfq_qwen4_exp::gated_residual_injection(value,streams):
                mfq_qwen4_exp::gated_residual_bottleneck(value,streams);
        };
        return {std::move(forward),{},std::move(activated),std::move(weight),std::move(prepared_right)};
    }
    Linear forward=[weight](CudaExecutionContext& execution, const Tensor& input) {
        auto output = execution.profiler.measure("nint.float_projection", [&] {
            return nint_float_projection_cuda(weight->nint, input,true,execution.config.gr_native_projection_input,
                execution.config.gr_fixed_group_projection);
        });
        // Match Metal's promotion of NINT's logical F16 weight dtype.
        return input.scalar_type() == tb::kFloat16 ? output.to(tb::kFloat16) : output;
    };
    GatedResidualMixProjection mixed;
    if(weight->is_nint())mixed=[weight](CudaExecutionContext& execution,const Tensor& input,
        const Tensor& normalized,int64_t streams) {
        return execution.profiler.measure("nint.float_projection_mix",[&] {
            return nint_float_projection_mix_cuda(weight->nint,input,normalized,streams,execution.config.gr_native_projection_input,
                execution.config.gr_fixed_group_projection,execution.config.gr_compact_mix);
        });
    };
    GatedResidualActivationProjection activated;
    if(weight->is_nint())activated=[weight](CudaExecutionContext& execution,const Tensor& input,
        int64_t streams,bool injection) {
        return execution.profiler.measure("nint.float_projection_activation",[&] {
            return nint_float_projection_activation_cuda(weight->nint,input,streams,injection,true,
                execution.config.gr_native_projection_input,execution.config.gr_fixed_group_projection);
        });
    };
    return {std::move(forward),std::move(mixed),std::move(activated),std::move(weight)};
}

Tensor dense(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name) {
    const mfq::cuda::WeightAllocationScope weight_memory;
    const auto& dtype=require_tensor(file, name).dtype;
    const bool dense_storage = dtype=="F32" || dtype=="F16" || dtype=="BF16";
    const bool quantized_storage = dtype=="NINT" || dtype=="NINT8-0" || dtype=="MFE";
    MFQ_RUNTIME_CHECK(dense_storage || quantized_storage,
        "expected a dense parameter: ",name);
    auto value=load_dense_gpu(execution, file,name);
    if (!dense_storage) return value;
    return value.to(dtype=="F16" ? tb::kFloat16 : dtype=="BF16" ? tb::kBFloat16 : tb::kFloat32);
}

Routed routed(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& name, int layer,
    int64_t experts, int64_t output, int64_t input, const std::string& role) {
    const auto& dtype=require_tensor(file, name).dtype;
    if(dtype=="TQ") {
        auto weight=load_tq_routed_weight(file,name,active_weight_load_device(execution),&execution);
        MFQ_RUNTIME_CHECK(weight->experts==experts && weight->out==output && weight->input==input,
            "TQ routed dimensions differ: ",name);
        Routed result=[weight](CudaExecutionContext&,const Tensor& x,const Tensor& ids){return weight->forward(x,ids);};
        result.tq_projections.push_back(std::move(weight));return result;
    }
    if (dtype=="F32" || dtype=="F16" || dtype=="BF16") {
        MFQ_RUNTIME_CHECK(!moe_parallel_config(execution).enabled(),
            "expert parallelism requires packed routed tensors: ",name);
        auto w=dense(execution,file,name);
        MFQ_RUNTIME_CHECK(w.sizes().vec()==std::vector<int64_t>({experts,output,input}),
            "dense expert tensor shape mismatch: ",name);
        return [w,output,input](CudaExecutionContext&, const Tensor& x,const Tensor& ids) {
            return moe_dense_matmul_hip(w,x,ids);
        };
    }
    auto w=std::make_shared<MfeWeight>(load_mfe_gpu(execution, file,name,true,layer,role));
    MFQ_RUNTIME_CHECK(w->n_experts==experts && w->out_per_expert==output && w->neuron_len==input,
        "routed tensor shape mismatch: ",name);
    Routed result=[w,experts](CudaExecutionContext& execution, const Tensor& x,const Tensor& ids) {
        auto route=build_moe_route_plan(ids.to(tb::kInt32).contiguous(),int(experts));
        return w->forward(execution,x.contiguous(),route);
    };
    result.projections.push_back(w);
    return result;
}

Routed routed_gate_up(CudaExecutionContext& execution, const mfq::ModelSource& file, const std::string& mlp_prefix,
    int layer, int64_t experts, int64_t width, int64_t input, const std::string& role) {
    const auto base=mlp_prefix+".experts";
    const auto gate_name=base+".gate.weight",up_name=base+".up.weight";
    const bool has_gate=has_tensor(file, gate_name),has_up=has_tensor(file, up_name);
    MFQ_RUNTIME_CHECK(has_gate==has_up,"incomplete routed Gate/Up pair under ",base);
    if (!has_gate) return routed(execution,file,base+".gate_up.weight",layer,experts,2*width,input,role);
    auto gate=routed(execution,file,gate_name,layer,experts,width,input,role);
    auto up=routed(execution,file,up_name,layer,experts,width,input,role);
    auto projections=gate.projections;
    projections.insert(projections.end(),up.projections.begin(),up.projections.end());
    auto tq_projections=gate.tq_projections;
    tq_projections.insert(tq_projections.end(),up.tq_projections.begin(),up.tq_projections.end());
    Routed result=[gate=std::move(gate),up=std::move(up)](CudaExecutionContext& execution, const Tensor& x,const Tensor& ids) {
        return tb::cat({gate(execution,x,ids),up(execution,x,ids)},-1);
    };
    result.projections=std::move(projections);
    result.tq_projections=std::move(tq_projections);
    return result;
}

void validate_load_options(
        const CudaExecutionContext& execution) {
    if (execution.tensor_parallel.enabled() ||
            execution.layer_placement.enabled() ||
            execution.n_gpu_layers >= 0) {
        throw std::runtime_error(
            "native attention adapter supports expert parallelism, but "
            "tensor/layer parallelism and offload require a different placement path");
    }
}

} // namespace mfq::cuda::weight_loader

mfq_tensor_backend::Tensor load_dense_cpu(const CudaExecutionContext &execution,
                                          const mfq::ModelSource &source,
                                          const std::string &name) {
    namespace tb = mfq_tensor_backend;
    const auto &record = require_tensor(source, name);
    tb::ScalarType dtype;
    size_t item_size;
    if (record.dtype == "BF16") { dtype = tb::kBFloat16; item_size = 2; }
    else if (record.dtype == "F16") { dtype = tb::kFloat16; item_size = 2; }
    else if (record.dtype == "F32") { dtype = tb::kFloat32; item_size = 4; }
    else if (record.dtype == "I64") { dtype = tb::kInt64; item_size = 8; }
    else if (record.dtype == "I32") { dtype = tb::kInt32; item_size = 4; }
    else throw std::runtime_error("unsupported dense dtype: " + record.dtype + " tensor " + name);

    auto blob = read_tensor(source, name);
    if (execution.drop_file_cache) source.drop_file_cache();
    MFQ_RUNTIME_CHECK(blob.size() >= sizeof(uint32_t), "truncated dense tensor header: ", name);
    size_t offset = 0;
    const auto rank = read_u32_from(blob, offset);
    MFQ_RUNTIME_CHECK(rank <= (blob.size() - offset) / sizeof(int64_t),
                      "truncated dense tensor shape: ", name);
    std::vector<int64_t> shape(rank);
    int64_t numel = 1;
    for (auto &extent : shape) {
        extent = read_i64_from(blob, offset);
        MFQ_RUNTIME_CHECK(extent >= 0 &&
                              (extent == 0 || numel <= std::numeric_limits<int64_t>::max() / extent),
                          "invalid dense tensor shape: ", name);
        numel *= extent;
    }
    MFQ_RUNTIME_CHECK(!record.shape || *record.shape == shape,
                      "dense tensor metadata shape mismatch: ", name);
    const auto payload = blob.size() - offset;
    MFQ_RUNTIME_CHECK(payload % item_size == 0 && uint64_t(numel) == payload / item_size,
                      "dense tensor payload size mismatch: ", name);
    return tb::from_blob(blob.data() + offset, shape, tb::TensorOptions().dtype(dtype)).clone();
}

mfq_tensor_backend::Tensor load_dense_native_gpu(CudaExecutionContext &execution,
                                                 const mfq::ModelSource &source,
                                                 const std::string &name) {
    const mfq::cuda::WeightAllocationScope weight_memory;
    MfqCudaGuard guard(active_weight_load_device(execution));
    return load_dense_cpu(execution, source, name).to(mfq_tensor_backend::kCUDA).contiguous();
}

namespace mfq::cuda {

std::string load_model_config_json(
        const mfq::ModelSource& source,
        const std::string& external_path) {
    if (external_path.empty()) return source.model_config_json();
    std::ifstream input(external_path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open config: " + external_path);
    return {std::istreambuf_iterator<char>(input), {}};
}

void validate_model_source(const mfq::ModelSource& source) {
    if (source.find_tensor("model.token_embedding.weight") == nullptr) {
        throw std::runtime_error(
            "canonical text tensor inventory has no token embedding");
    }
    for (const auto& component : source.resolved_model_graph().components) {
        if (component.tensor_root != "runtime" &&
                !has_tensor_prefix(source, component.tensor_root)) {
            throw std::runtime_error(
                "model graph declares " + component.kind +
                " but its canonical tensor inventory is missing");
        }
    }
}

} // namespace mfq::cuda
