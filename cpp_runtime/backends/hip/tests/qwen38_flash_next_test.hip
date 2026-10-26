// Native ABI numerical-test bridge: tests/test_cuda_flash_next.py supplies the
// same NumPy oracle cases to this executable and the production Torch module.
#include "mfq_hip_attention_ops.h"
#include "quant_kv_check.h"
#include "hot_kv_check.h"
#include "mfq/kernels/hip/selected_attention.h"
#include "mfq/kernels/hip/glm5_next.h"
#include "mfq/kernels/hip/qwen4_exp.h"
#include "glm5_next/model.h"
#include "qwen4_exp/model.h"
#include "hip_execution.h"
#include "runtime/decode_window.h"
#include "nint.h"
#include "mfq_hip_context.h"
#include "mfq_hip_sampling_ops.h"
#include "mfq_hip_moe_ops.h"
#include "mfq_hip_selective_state_scan_ops.h"
#include "mfq_hip_norm_ops.h"
#include "mfq_hip_activation_ops.h"
#include "mfq_hip_quant_ops.h"
#include "mfq/kernels/hip/deepseek_v41.h"
#include "mfq/kernels/hip/tq_moe.h"
#include "mfq/kernels/hip/deepseek_v4_attention.h"
#include "mfq/kernels/hip/deepseek_v4_fp4.h"
#include "mfq/kernels/hip/deepseek_v4_hc.h"
#include "tq.h"
#include "mfq/model_source.h"
#include "storage/transformer_loader.h"
#include "core/axial_patch_vision.h"
#include "core/grid_vision_component.h"
#include "deepseek_v41/ops.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <limits>
#include <memory>
#include <map>
#include <tuple>
#include <string>
#include "deepseek_v41_runtime_check.h"
#include "gdn_stateful_check.h"
#include "runtime/mtp_compact_sampling.h"
#include "qwen4_graph_reuse_check.h"
#include "qwen4_session_check.h"
#include "qwen4_predictor_graph_check.h"

mfq_tensor_backend::Tensor attention_glm_mla_sparse_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv,
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor meta,
    double scale);
mfq_tensor_backend::Tensor attention_dsv4_sparse_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor kv,
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor mask,
    mfq_tensor_backend::Tensor sinks, mfq_tensor_backend::Tensor meta,
    double scale);
mfq_tensor_backend::Tensor glm_dsa_indexer_layer_norm_cuda(
    mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor,
    mfq_tensor_backend::Tensor, double);
mfq_tensor_backend::Tensor glm_dsa_cache_write_cuda(
    mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor);

namespace {
using namespace mfq::cuda;
using Json = nlohmann::json;


void require_constant(
    const Tensor& value, float expected, float tolerance, const char * name) {
    const auto host = value.to(kFloat32).contiguous().cpu();
    float maximum_error = 0.0f;
    for (int64_t index = 0; index < host.numel(); ++index) {
        maximum_error = std::max(
            maximum_error,
            std::abs(host.data_ptr<float>()[index] - expected));
    }
    if (maximum_error > tolerance) {
        throw std::runtime_error(
            std::string(name) + " mismatch, max_abs=" +
            std::to_string(maximum_error));
    }
}

void require_qsa_gqa_means(
    const Tensor& value, float first, float second, float tolerance) {
    const auto host = value.to(kFloat32).contiguous().cpu();
    float maximum_error = 0.0f;
    for (int head = 0; head < 24; ++head) {
        const float expected = head < 12 ? first : second;
        for (int width = 0; width < 256; ++width) {
            maximum_error = std::max(
                maximum_error,
                std::abs(host.data_ptr<float>()[head * 256 + width] - expected));
        }
    }
    if (maximum_error > tolerance) {
        throw std::runtime_error(
            "QSA sparse GQA mismatch, max_abs=" +
            std::to_string(maximum_error));
    }
}

void check_qsa_sparse_attention() {
    const auto float_cuda = TensorOptions{}.dtype(kFloat32).device(kCUDA);
    const auto half_cuda = TensorOptions{}.dtype(kFloat16).device(kCUDA);
    const auto int_cuda = TensorOptions{}.dtype(kInt32).device(kCUDA);
    constexpr int selected_count = 2048;
    std::vector<int32_t> selected(selected_count);
    float glm_expected = 0.0f;
    for (int index = 0; index < selected_count; ++index) {
        selected[index] = (index * 37 + 11) % 256;
        glm_expected += static_cast<float>(selected[index]);
    }
    glm_expected /= selected.size();
    const auto indices = tensor(selected)
        .reshape({1, 1, selected_count}).to(int_cuda);

    const auto qsa_q = zeros({1, 24, 1, 256}, float_cuda);
    const auto qsa_k = zeros({1, 2, 256, 256}, half_cuda);
    const auto qsa_v = arange(512, float_cuda)
        .reshape({1, 2, 256, 1}).expand({1, 2, 256, 256})
        .contiguous().to(kFloat16);
    require_qsa_gqa_means(
        mfq_qwen4_exp::sparse_gqa_attention(
            qsa_q, qsa_k, qsa_v, indices),
        glm_expected, glm_expected + 256.0f, 6.0e-2f);

}

Tensor input(const Json& j) {
    if (j.is_null()) return {};
    const auto shape = j.at("shape").get<std::vector<int64_t>>();
    const auto dtype = j.value("dtype", "float32");
    const auto host_type=dtype=="int64"?kInt64:dtype=="int32"?kInt32:
        dtype=="uint8"?kUInt8:dtype=="int8"?kInt8:kFloat32;
    auto host = empty(shape, TensorOptions{}.dtype(host_type));
    if(j.contains("file")) {
        std::ifstream file(j.at("file").get<std::string>(),std::ios::binary);
        if(!file.read(static_cast<char*>(host.data_ptr()),host.numel()*host.element_size()))
            throw std::runtime_error("cannot read raw tensor fixture");
    } else if (host_type!=kFloat32) {
        const auto data = j.at("data").get<std::vector<int64_t>>();
        if (data.size() != static_cast<size_t>(host.numel())) throw std::runtime_error("fixture size mismatch");
        auto original=tensor(data).reshape(shape);
        host.copy_(original.to(host_type));
    } else {
        const auto data = j.at("data").get<std::vector<float>>();
        if (data.size() != static_cast<size_t>(host.numel())) throw std::runtime_error("fixture size mismatch");
        std::copy(data.begin(), data.end(), host.data_ptr<float>());
    }
    auto value = host.to(Device{DeviceType::cuda, 0});
    if (dtype == "float16") value = value.to(kFloat16);
    else if (dtype == "bfloat16") value = value.to(kBFloat16);
    else if (dtype != "float32" && host_type==kFloat32) throw std::runtime_error("unknown fixture dtype");
    if (j.value("noncontiguous", false) && value.dim() >= 2)
        value = value.transpose(-1, -2).contiguous().transpose(-1, -2);
    return value;
}

std::vector<Tensor> run(const std::string& op, const std::vector<Tensor>& a, const Json& p) {
    CudaExecutionContext execution;
    if(op=="runtime_dspark_context_check") {
        mfq::cuda::deepseek_v41_runtime::run_dspark_self_check();
        return {mfq_tensor_backend::ones({1},TensorOptions().device(kCUDA).dtype(kFloat32))};
    }
    if(op=="runtime_dsv41_model")return mfq::cuda::tests::dsv41_model(p);
    if(op=="runtime_dsv41_prompt")return mfq::cuda::tests::dsv41_prompt(a,p);
    if(op=="grid_vision") {
        using namespace mfq::cuda::grid_vision_runtime;
        namespace grid_detail=mfq::cuda::grid_vision_runtime::detail;
        auto source=mfq::open_model_source(p.at("model").get<std::string>());
        mfq::GridVisionConfig config{p.at("hidden"),p.at("intermediate"),p.at("layers"),p.at("heads"),
            3,2,1,2,p.at("output_hidden"),16,10000.,1e-6};
        struct LoadingScope {
            bool& flag;bool previous;
            LoadingScope(bool& value,bool host):flag(value),previous(value){flag=host;}
            ~LoadingScope(){flag=previous;}
        };
        const auto loaded=[&] {
            LoadingScope loading(execution.loading_cpu_layer,p.value("streamed",false));
            auto encoder=CudaGridVisionEncoder::load(execution,*source,config);
            auto norm=grid_detail::LayerNorm::load(execution,*source,mfq::GridVisionTensorRole::merger_norm_weight,
                mfq::GridVisionTensorRole::merger_norm_bias,config.hidden_size,config.layer_norm_eps);
            auto up=grid_detail::Affine::load(execution,*source,"vision.merger.mlp.up.weight","vision.merger.mlp.up.bias");
            auto down=grid_detail::Affine::load(execution,*source,"vision.merger.mlp.down.weight","vision.merger.mlp.down.bias");
            return std::tuple{std::move(encoder),std::move(norm),std::move(up),std::move(down)};
        };
        auto [encoder,norm,up,down]=loaded();
        auto encoding=encoder.encode(execution,a.at(0),{{p.value("temporal",1),p.at("height"),p.at("width")}});
        Tensor patches;
        while(auto next=encoding.next())if(next.value)patches=std::move(*next.value);
        norm=grid_detail::to_cuda(norm);up=grid_detail::to_cuda(up);down=grid_detail::to_cuda(down);
        auto merged=mfq::models::grid_vision::merge(patches,patches.size(0),config,
            [&](Tensor x){return norm(x);},[](Tensor x,int64_t rows,int64_t width){return x.reshape({rows,width});},
            [&](Tensor x){return up(execution,x);},[](Tensor x){return mfq_tensor_backend::gelu(x,"none");},
            [&](Tensor x){return down(execution,x);});
        return {patches,merged};
    }
    if(op=="axial_vision") {
        auto source=mfq::open_model_source(p.at("model").get<std::string>());
        mfq::models::axial_patch_vision::Config config{
            p.at("hidden"),p.at("layers"),p.at("heads"),p.at("intermediate"),p.at("patch_size"),
            p.at("downsample_ratio"),p.at("output_hidden"),p.value("theta",10000.),p.value("eps",1e-6)};
        auto tower=mfq::cuda::axial_patch_vision::CudaAxialPatchVisionTower::load(execution,*source,config);
        return {tower.encode(execution,a.at(0),p.at("height"),p.at("width"))};
    }
    if(op=="causal_plain")return {mfq_selected_attention::causal(a.at(0),a.at(1),a.at(2),a.at(3),p.at("columns"),p.at("scale"))};
    if(op=="nint_decode_f32")return {nint_decode_f32_cuda(a.at(0).to(kUInt8),a.at(1).to(kUInt8),a.at(2),a.at(3).to(kUInt8),a.at(4).to(kUInt8),a.at(5),a.at(6),p.at("width"),p.at("group_size"))};
    if(op=="nint8_dequant_f32")return {nint8_zero_dequant_f32_cuda(a.at(0).to(kUInt8),a.at(1),p.at("width"))};
    if(op=="tq_moe_invalid_route") {
        const bool pair=p.at("pair");const int rows=p.at("rows"),count=p.at("expert_count");
        ProjectionView projection{};projection.rows=rows*count;
        projection.input_columns=a.at(1).size(-1);projection.stored_columns=projection.input_columns;
        TqMoeRoutePlan plan{};plan.ids=a.at(0).data_ptr<int32_t>();plan.tokens=1;
        plan.routes=a.at(0).numel();plan.expert_count=count;plan.expert_offset=p.at("expert_offset");
        auto output=empty({plan.routes,rows},a.at(1).options());
        const int error=pair?mfq_tq_moe_swiglu(projection,projection,a.at(1).data_ptr(),output.data_ptr(),
            rows,plan,0,p.value("limit",7.f),mfq_current_cuda_stream()):
            mfq_tq_moe_projection(projection,a.at(1).data_ptr(),output.data_ptr(),rows,plan,0,true,mfq_current_cuda_stream());
        MFQ_RUNTIME_CHECK(error==0,"invalid-route zero output launch failed");return {output};
    }
    if(op=="moe_router")return {p.value("optimized",false)?moe_router_gemv_cuda(a.at(0),a.at(1)):
        mfq_tensor_backend::matmul(a.at(0).to(kFloat32),a.at(1).transpose(0,1))};
    if(op=="moe_topk")return moe_topk_cuda(a.at(0),p.at("top_k"),
        p.value("sigmoid",false),p.value("sqrt_softplus",false),p.value("normalize",true),
        false,a.size()>1?MfqOptional<Tensor>(a.at(1)):mfq_nullopt,1e-20,p.value("scale",1.));
    if(op=="mtp_compact_host") {
        MfqSamplingParams sampling;sampling.top_k=p.at("top_k");sampling.temperature=p.value("temperature",.8);
        sampling.top_p=p.value("top_p",.95);
        auto logits=a.at(0);
        if(a.size()>1)logits=sample_apply_penalties_cuda(logits.clone(),a.at(1),.2,.1,1.05);
        const auto rows=mfq::cuda::mtp_sampling::rows(logits,sampling,p.value("packets",false));
        if(p.value("benchmark_only",false)) {
            uint64_t checksum=0;
            for(const auto& row:rows)if(!row.tokens.empty()) {
                const auto probability=row.probabilities.front();uint64_t raw=0;
                std::memcpy(&raw,&probability,sizeof(probability));checksum^=raw+uint64_t(row.tokens.front());
            }
            static volatile uint64_t consumed=0;consumed=checksum;
            return {};
        }
        auto options=TensorOptions().device(kCPU);
        auto ids=full({logits.size(0),sampling.top_k},-1,options.dtype(kInt64));
        auto probabilities=zeros(ids.sizes(),options.dtype(kFloat32)),bits=zeros(ids.sizes(),options.dtype(kInt64));
        auto lengths=zeros({logits.size(0)},options.dtype(kInt64));
        for(size_t row=0;row<rows.size();++row) {
            lengths.data_ptr<int64_t>()[row]=rows[row].tokens.size();
            for(size_t i=0;i<rows[row].tokens.size();++i) {
                const auto offset=row*sampling.top_k+i;
                ids.data_ptr<int64_t>()[offset]=rows[row].tokens[i];
                const auto probability=rows[row].probabilities[i];
                probabilities.data_ptr<float>()[offset]=float(probability);
                uint64_t raw=0;static_assert(sizeof(probability)<=sizeof(raw));std::memcpy(&raw,&probability,sizeof(probability));
                bits.data_ptr<int64_t>()[offset]=int64_t(raw);
            }
        }
        return {ids,probabilities,bits,lengths};
    }
    if(op=="dsv41_candidate_blocks") {
        namespace idx=mfq::cuda::deepseek_v41_runtime;
        const auto &scores=a.at(0);
        const int64_t ratio=p.at("ratio"),position=p.at("position"),size=p.value("block_size",8),
            requested=p.value("requested_blocks",2048);
        if(p.value("stage",std::string("select"))=="scores") {
            if(p.value("optimized",false)||p.value("native_f16",false))return {deepseek_v41_block_scores8_cuda(scores,ratio,position)};
            const auto batch=scores.size(0),tokens=scores.size(1),width=scores.size(2);
            const auto blocks=(width+size-1)/size,padded=blocks*size;
            auto padded_scores=padded==width?scores:cat({scores,full({batch,tokens,padded-width},
                -std::numeric_limits<float>::infinity(),scores.options())},-1);
            auto block_scores=padded_scores.reshape({batch,tokens,blocks,size}).amax(-1);
            auto options=scores.options().dtype(kInt64);
            auto visible=arange(position+1,position+tokens+1,options).reshape({1,tokens,1})/double(ratio);
            auto last=where(visible>0.,(visible-1.)/double(size),-1.);
            auto ids=arange(blocks,options).reshape({1,1,blocks});
            auto result=where(ids==last,std::numeric_limits<float>::infinity(),block_scores);
            MFQ_RUNTIME_CHECK(result.scalar_type()==kFloat16,"blockscore reference dtype changed");
            return {result};
        }
        if(p.value("native_ordered16",false))return {deepseek_v41_candidate_blocks8_ordered16_cuda(scores,ratio,position,requested)};
        if(p.value("native_f16",false))return {deepseek_v41_candidate_blocks8_f16_cuda(scores,ratio,position,requested)};
        return {p.value("optimized",false)?idx::candidate_blocks_from_scores(scores,ratio,position,requested,size)
            :idx::candidate_blocks_from_scores_generic(scores,ratio,position,requested,size)};
    }
    if(op=="dsv41_index_topk") {
        namespace idx=mfq::cuda::deepseek_v41_runtime;
        const int64_t pool=p.at("pool_length");
        const auto source=a.size()>1?std::optional<Tensor>(a.at(1)):std::nullopt;
        if(p.value("stage",std::string("select"))=="threshold")return {dsv4_topk512_cuda(a.at(0))};
        if(p.value("check_shared",false)) {
            auto relative=dsv4_topk512_cuda(a.at(0));
            auto relative64=relative.to(kInt64).contiguous();
            auto values=a.at(0).gather(-1,relative64);
            auto selected=source?source->gather(-1,relative64):relative64;
            auto old=idx::normalize_topk(selected,values,pool);
            auto fused=deepseek_v41_finalize_index_topk_cuda(
                a.at(0),relative,source?*source:Tensor{},pool);
            return {old.to(kInt64),fused.to(kInt64),relative64};
        }
        return {p.value("optimized",false)?idx::select_topk(a.at(0),512,pool,source)
            :idx::select_topk_generic(a.at(0),512,pool,source)};
    }
    if(op=="dsv41_candidate_positions") {
        namespace idx=mfq::cuda::deepseek_v41_runtime;
        const int64_t size=p.value("block_size",8),pool=p.at("pool_length");
        return {p.value("optimized",false)?idx::candidate_positions(a.at(0),size,pool)
            :idx::candidate_positions_generic(a.at(0),size,pool)};
    }
    if(op=="dsv41_normalize_topk")return {deepseek_v41_normalize_topk_cuda(a.at(0),a.at(1),p.at("pool_length"))};
    if(op=="sample_compact")return {sample_top_k_compact_cuda(a.at(0),p.at("top_k"))};
    if(op=="sample_compact_if")return {sample_top_k_compact_if_cuda(a.at(0),p.at("top_k"),a.at(1).data_ptr<int32_t>())};
    if(op=="dense_gr_injection") {
        const auto weight=a.at(0).to(kBFloat16);
        if(p.value("legacy",false))return {mfq_qwen4_exp::gated_residual_injection(
            mfq_selected_attention::promoted_matmul(a.at(1),weight.transpose(-1,-2)),p.at("streams"))};
        return {dense_gr_injection_cuda(weight,a.at(1),p.at("streams"))};
    }
    if(op=="mxfp8_grouped") {
        const int64_t groups=p.at("groups"),per_group=a.at(0).size(0)/groups;
        if(p.value("legacy",false)) {
            MFQ_RUNTIME_CHECK(per_group%128==0,"legacy MXFP8 group boundary must preserve scale blocks");
            std::vector<Tensor> parts;
            for(int64_t group=0;group<groups;++group) {
                auto values=a.at(0).narrow(0,group*per_group,per_group).contiguous();
                auto scales=a.at(1).narrow(0,group*per_group/128,per_group/128).contiguous();
                auto x=a.at(2).select(1,group).contiguous();
                parts.push_back(p.value("float_output",false)?mxfp8_gemm_f32_cuda(values,scales,x):
                    mxfp8_matmul_f16_cuda(values,scales,x));
            }
            return {cat(parts,-1).contiguous()};
        }
        if(p.value("runtime",false)) {
            Mxfp8Weight weight;weight.values=a.at(0);weight.scales=a.at(1);
            weight.out=a.at(0).size(0);weight.neuron_len=a.at(0).size(1);
            return {p.value("float_output",false)?mxfp8_groupwise_matmul_f32(execution.profiler,weight,a.at(2),groups):
                mxfp8_groupwise_matmul(execution.profiler,weight,a.at(2),groups)};
        }
        return {p.value("float_output",false)?mxfp8_groupwise_gemm_f32_cuda(a.at(0),a.at(1),a.at(2),groups):
            mxfp8_groupwise_gemm_f16_cuda(a.at(0),a.at(1),a.at(2),groups)};
    }
    if(op=="mxfp8_gate_up_clipped") {
        if(p.value("optimized",false))return {mxfp8_gate_up_clipped_swiglu_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),p.at("limit"))};
        Mxfp8Weight g,u;g.values=a.at(0);g.scales=a.at(1);g.out=a.at(0).size(0);g.neuron_len=a.at(0).size(1);
        u.values=a.at(2);u.scales=a.at(3);u.out=g.out;u.neuron_len=g.neuron_len;
        auto gate=mxfp8_matmul(execution.profiler,g,a.at(4));
        auto up=mxfp8_matmul(execution.profiler,u,a.at(4));
        return {clipped_silu_mul_cuda(gate,up,p.at("limit"))};
    }
    if(op=="clipped_silu")return {clipped_silu_mul_cuda(a.at(0),a.at(1),p.at("limit"))};
    if(op=="dsv_tail_rope") {
        auto x=p.value("transposed",false)?a.at(0).transpose(1,2):a.at(0);
        if(p.value("optimized",false))
            return {dsv4_tail_rope_cuda(x,a.at(1),a.at(2),a.at(3),64,
                p.value("float_output",false),p.value("token_major",false))};
        if(p.value("float_output",false))x=x.to(kFloat32);
        if(p.value("fused_reference",false)) {
            auto output=dsv4_tail_rope_cuda(x,a.at(1),a.at(2),a.at(3),64);
            return {p.value("token_major",false)?output.transpose(1,2).contiguous():output};
        }
        auto head=x.slice(-1,0,x.size(-1)-64).contiguous();
        auto tail=x.slice(-1,x.size(-1)-64,x.size(-1)).contiguous();
        auto rotated=glm_interleaved_rope_cuda(tail,a.at(1),a.at(2),a.at(3),64);
        return {cat({head,rotated},-1).contiguous()};
    }
    if(op=="dsv_weighted_rms") {
        if(p.value("keep_output_cast",false))return {rms_norm_offset_cuda(a.at(0),a.at(1),p.at("eps"),0.).to(kFloat16)};
        if(p.value("optimized",false))return {rms_norm_f16_cuda(a.at(0),a.at(1),p.at("eps"),0.)};
        return {mfq::cuda::deepseek_v41_runtime::weighted_rms(a.at(0),a.at(1),p.at("eps"))};
    }
    if(op=="generic_clipped_silu") {
        const double limit=p.at("limit");
        auto gate=clamp_max(a.at(0).to(kFloat32),limit);
        auto up=clamp(a.at(1).to(kFloat32),-limit,limit);
        return {(silu(gate)*up).to(kFloat16).contiguous()};
    }
    if(op=="gdn_short")return linear_conv_qkv_gate_short_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),a.at(5),a.at(6),p.at("key_heads"),p.at("value_heads"),p.at("width"),p.value("eps",1e-6));
    if(op=="ssm_scan")return selective_state_scan_prefill_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),a.at(5),a.at(6),a.at(7),p.value("low",0.0),p.value("high",1e30));
    if(op=="ssm_decode")return {selective_state_scan_decode_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),a.at(5),a.at(6),a.at(7),p.value("low",0.0),p.value("high",1e30)),a.at(0)};
    if(op=="ssm_norm")return {grouped_gated_rms_norm_cuda(a.at(0),a.at(1),a.at(2),p.at("group"),p.value("eps",1e-5))};
    if(op=="relu2")return {relu_squared_cuda(a.at(0))};
    if(op=="runtime_engram_rows") {
        auto source=mfq::open_model_source(p.at("model").get<std::string>());
        auto table=mfq::cuda::deepseek_v41_runtime::EngramTable::load(*source,"embedding.weight",
            p.at("rows"),p.at("width"),p.at("cache_rows"));
        std::vector<Tensor> result;
        for(const auto& sequence:p.at("sequences")) {
            auto ids=sequence.at("ids").get<std::vector<int64_t>>();
            table->prefetch(ids.data(),ids.size());
            auto value=table->gather(ids.data(),sequence.at("batch"),sequence.at("tokens"),
                sequence.at("columns"),mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA,0));
            result.push_back(value.clone());
            MFQ_CUDA_CHECK(hipStreamSynchronize(mfq_current_cuda_stream()));
        }
        return result;
    }
    if(op=="runtime_engram_forward") {
        auto source=mfq::open_model_source(p.at("model").get<std::string>());
        mfq::models::deepseek_v41::Config config;
        config.hc_mult=a.at(0).size(2);config.hidden=a.at(0).size(3);
        config.rms_eps=p.value("eps",1e-6);config.engram_layer_ids={0};
        config.engram_num_embeddings={p.at("rows").get<int64_t>()};
        config.engram_head_dim=p.at("width");config.engram_max_ngram_size=2;
        config.engram_n_heads=p.at("columns");
        execution.config.deepseek_v41_engram_cache_rows=p.value("cache_rows",2);
        auto engram=mfq::cuda::deepseek_v41_runtime::Engram::load(execution,*source,config,0);
        mfq::cuda::deepseek_v41_runtime::EngramHashBatch hashes;
        hashes.batch=a.at(0).size(0);hashes.tokens=a.at(0).size(1);hashes.layers=1;
        hashes.columns=p.at("columns");hashes.values=p.at("ids").get<std::vector<int64_t>>();
        Tensor text_rows;std::vector<int64_t> host_rows;
        if(p.contains("text_rows")) {
            host_rows=p.at("text_rows").get<std::vector<int64_t>>();
            text_rows=mfq_tensor_backend::tensor(host_rows).to(a.at(0).device());
        }
        engram->prefetch(hashes,text_rows.defined()?&host_rows:nullptr);
        return {engram->forward(execution,a.at(0),hashes,text_rows,text_rows.defined()?&host_rows:nullptr)};
    }
    if(op=="runtime_dsv41_collapse") {
        mfq::cuda::deepseek_v41_runtime::Block block;
        block.config.hc_mult=4;block.config.hidden=a.at(0).size(3);
        block.config.hc_sinkhorn_iters=p.value("iterations",20);
        block.config.rms_eps=p.value("rms_eps",1e-6);block.config.hc_eps=p.value("eps",1e-6);
        if(p.value("legacy",false)) {
            using namespace mfq::cuda::deepseek_v41_runtime;
            auto flat=a.at(0).flatten(2).to(kFloat32);
            auto inverse=mfq_tensor_backend::rsqrt(flat.square().mean(-1,true)+block.config.rms_eps);
            auto mixes=mfq_tensor_backend::matmul(flat*inverse,a.at(3).transpose(0,1));
            auto expansion=dsv4_hc_pre_cuda(a.at(0),mixes.contiguous(),a.at(4),a.at(5),
                block.config.hc_sinkhorn_iters,block.config.hc_eps);
            auto pre=mfq_tensor_backend::sigmoid(mixes.slice(-1,0,4)*a.at(4).index({0})+
                a.at(5).slice(0,0,4))+block.config.hc_eps;
            auto branch=(a.at(1).to(kFloat32).unsqueeze(-1)*a.at(0).to(kFloat32)).sum(2);
            branch=weighted_rms(branch,a.at(2),block.config.rms_eps);
            return {branch,pre.contiguous(),expansion.at(1),expansion.at(2)};
        }
        const auto mix=block.collapse(execution.profiler,a.at(0),a.at(1),a.at(3),a.at(4),a.at(5),a.at(2),"test.collapse");
        return {mix.branch,mix.next_pre,mix.post,mix.combination};
    }
    if(op=="dsv41_hc_prepare")return deepseek_v41_hc_prepare_cuda(a.at(0),a.at(1),a.at(2),p.value("eps",1e-6));
    if(op=="dsv41_hc_input")return {deepseek_v41_hc_input_cuda(a.at(0),a.at(1),a.at(2),p.value("eps",1e-6))};
    if(op=="dsv41_hc_project")return {deepseek_v41_hc_project_cuda(a.at(0),a.at(1),a.at(2))};
    if(op=="dsv41_hc_finalize")return deepseek_v41_hc_finalize_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),a.at(5),p.value("iterations",20),p.value("rms_eps",1e-6),p.value("eps",1e-6));
    if(op=="dsv41_engram_rows")return {deepseek_v41_engram_rows_cuda(a.at(0).to(kUInt8),a.at(1),p.at("width"))};
    if(op=="dsv41_engram_gate")return {deepseek_v41_engram_gate_add_cuda(a.at(0),a.at(1),a.at(2),p.value("eps",1e-6))};
    if(op=="dsv4_hc_coefficients")return dsv4_hc_coefficients_cuda(a.at(0),a.at(1),a.at(2),p.value("iterations",20),p.value("eps",1e-6));
    if(op=="dsv4_candidate_scores")return {dsv4_indexer_candidate_scores_cuda(a.at(0),a.at(1),a.at(2),a.at(3),p.at("offset"),p.at("ratio"))};
    if(op=="dsv41_kv_join") {
        auto second=p.contains("pool")?a.at(1).narrow(1,0,int64_t(p.at("pool"))):a.at(1);
        return {p.value("optimized",false)?deepseek_v41_join_kv_cuda(a.at(0),second):
            mfq_tensor_backend::cat({a.at(0),second},1)};
    }
    if(op=="dsv41_kv_norm_rope_store") {
        auto cache=a.at(5);auto target=cache.narrow(1,int64_t(p.at("slot")),1);
        const double eps=p.value("eps",1e-6);
        if(p.value("optimized",false))deepseek_v41_kv_norm_rope_store_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),target,eps);
        else {
            auto normalized=rms_norm_f16_cuda(a.at(0).reshape({1,512}),a.at(1),eps,0.);
            auto rotated=dsv4_tail_rope_cuda(normalized.reshape({1,1,1,512}),a.at(2),a.at(3),a.at(4),64,false,true);
            deepseek_v41_mxfp8_e4m3_sim_into_cuda(rotated.reshape({1,1,512}),target);
        }
        return {cache};
    }
    if(op=="dsv41_fp8_store") {
        auto cache=a.at(1);const int64_t slot=p.at("slot");
        if(p.value("optimized",false))
            deepseek_v41_mxfp8_e4m3_sim_into_cuda(a.at(0),cache.narrow(1,slot,1));
        else {
            auto local=deepseek_v41_mxfp8_e4m3_sim_cuda(a.at(0));
            auto index=mfq_tensor_backend::full({1},slot,cache.options().dtype(kInt64));
            cache.index_copy_(1,index,local);
        }
        return {cache};
    }
    if(op=="dsv41_fp8")return {deepseek_v41_mxfp8_e4m3_sim_cuda(a.at(0))};
    if(op=="dsv41_fp4")return {deepseek_v41_mxfp4_e4m3_scale_sim_cuda(a.at(0))};
    if(op=="dsv4_index_query_fp4") {
        auto x=p.value("transposed",true)?a.at(0).transpose(1,2):a.at(0);
        if(p.value("optimized",false)) {
            Dsv4RopeTable table; table.cos=a.at(2); table.sin=a.at(3);
            return {mfq::cuda::deepseek_v41_runtime::index_query_transform(x,a.at(1),table)};
        }
        auto rotated=dsv4_tail_rope_cuda(x,a.at(1),a.at(2),a.at(3),64,false,true);
        return {dsv4_fp4_sim_cuda(rotated.to(kFloat16).contiguous())};
    }
    if(op=="dsv4_fp4")return {dsv4_fp4_sim_cuda(a.at(0))};
    if(op=="dsv4_hc_pre")return dsv4_hc_pre_cuda(a.at(0),a.at(1),a.at(2),a.at(3),p.value("iterations",20),p.value("eps",1e-6));
    if(op=="dsv4_hc_post")return {dsv4_hc_post_cuda(a.at(0),a.at(1),a.at(2),a.at(3))};
    if(op=="dsv4_scores")return {dsv4_indexer_scores_cuda(a.at(0),a.at(1),a.at(2),p.at("offset"),p.at("ratio"))};
    if(op=="dsv4_topk")return {dsv4_topk512_cuda(a.at(0))};
    if(op=="dsv4_prefill_plan")return dsv4_build_prefill_plan_cuda(a.at(0).to(kInt32),p.at("offset"),p.at("history"),p.at("pools"),p.at("ratio"),p.at("window"));
    if(op=="dsv4_decode_plan")return dsv4_build_decode_plan_cuda(a.at(0).to(kInt32),a.at(1),p.at("pools"),p.at("ratio"),p.at("window"));
    if(op=="dsv4_compress")return {dsv4_compress_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),a.at(5),a.at(6),a.at(7),a.at(8),p.at("ratio"),p.at("overlap"),p.value("quant",0),p.value("eps",1e-6))};
    if(op=="dsv4_pool_update")return {dsv4_decode_pool_update_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),a.at(5),a.at(6),a.at(7),a.at(8),a.at(9),a.at(10),a.at(11),p.at("ratio"),p.at("overlap"),p.value("quant",0),p.value("eps",1e-6)),a.at(4),a.at(5),a.at(6),a.at(7)};
    if(op=="dsv4_sparse_split") {
        auto meta=a.at(6).defined()?a.at(6):empty({16*1024*1024},a.at(0).options().dtype(kFloat32));
        if(p.value("optimized",false))return {attention_dsv4_sparse_split_cuda(a.at(0),a.at(1),a.at(2),
            a.at(3).to(kInt32),a.at(4),a.at(5),meta,p.at("scale"))};
        return {attention_dsv4_sparse_cuda(a.at(0),mfq_tensor_backend::cat({a.at(1),a.at(2)},1),
            a.at(3).to(kInt32),a.at(4),a.at(5),meta,p.at("scale"))};
    }
    if(op=="dsv4_sparse")return {attention_dsv4_sparse_cuda(a.at(0),a.at(1),a.at(2).to(kInt32),a.at(3),a.at(4),a.at(5).defined()?a.at(5):empty({16*1024*1024},a.at(0).options().dtype(kFloat32)),p.at("scale"))};
    if(op=="glm_rope")return {glm_interleaved_rope_cuda(a.at(0),a.at(1),a.at(2),a.at(3),p.at("rotary"))};
    if(op=="glm_index_norm")return {glm_dsa_indexer_layer_norm_cuda(a.at(0),a.at(1),a.at(2),p.value("eps",1e-5))};
    if(op=="glm_cache")return {glm_dsa_cache_write_cuda(a.at(0),a.at(1),a.at(2))};
    if(op=="glm_scores")return {glm_dsa_indexer_scores_cuda(a.at(0),a.at(1),a.at(2),p.at("offset"),p.at("keys"))};
    if(op=="glm_decode_scores")return {glm_dsa_indexer_scores_decode_cuda(a.at(0),a.at(1),a.at(2),a.at(3),p.at("keys"))};
    if(op=="glm_sparse")return {attention_glm_mla_sparse_cuda(a.at(0),a.at(1),a.at(2).to(kInt32),a.at(3).defined()?a.at(3):empty({16*1024*1024},a.at(0).options().dtype(kFloat32)),p.at("scale"))};
    if(op=="runtime_greedy_head")
        return {a.at(0).argmax(-1),sample_greedy_cuda(a.at(0).contiguous())};
    if(op=="runtime_moe_map")
        return moe_build_expert_map_cuda(a.at(0).to(kInt32).contiguous(),p.at("experts"),p.at("tile_rows"));
    const auto optional = [&](size_t i) -> std::optional<Tensor> {
        return a.at(i).defined() ? std::optional<Tensor>(a.at(i)) : std::nullopt;
    };
    std::optional<double> scale;
    if (p.contains("scale") && !p.at("scale").is_null()) scale = p.at("scale").get<double>();
    if(op=="qsa_graph_select") {
        const auto& scores=a.at(0);const auto& positions=a.at(1);
        const int64_t t=scores.size(1),pools=scores.size(2),pool=p.at("pool"),budget=p.at("budget");
        if(p.value("variant","")=="fast")return {mfq_qwen4_exp::qsa_graph_select(scores,positions,pool,budget)};
        auto absolute=positions.reshape({1,t,1});
        auto ends=(arange(pools,positions.options())*pool+pool-1).reshape({1,1,pools});
        auto visible=ends<=absolute;
        auto ranked=where(visible,scores,full_like(scores,-1e30));
        const auto count=std::min(budget/pool,pools),columns=budget+pool-1;
        auto indices=std::get<1>(topk(ranked,count,-1,true,false));
        auto valid=visible.expand({1,t,pools}).gather(-1,indices).unsqueeze(-1).expand({1,t,count,pool});
        auto expanded=indices.unsqueeze(-1)*pool+arange(pool,indices.options());
        expanded=where(valid,expanded,full_like(expanded,-1));
        auto selected=full({1,t,columns},-1,positions.options());
        selected.narrow(-1,0,count*pool).copy_(expanded.reshape({1,t,count*pool}));
        if(pool>1) {
            auto tail_count=(absolute+1).remainder(pool);
            auto offsets=arange(pool-1,positions.options()).reshape({1,1,pool-1});
            auto tail=absolute+1-tail_count+offsets;
            selected.narrow(-1,budget,pool-1).copy_(where(offsets<tail_count,tail,full_like(tail,-1)));
        }
        auto dense=arange(columns,positions.options()).reshape({1,1,columns}).expand({1,t,columns});
        dense=where(dense<=absolute,dense,full_like(dense,-1));
        selected=where(absolute+1<=budget,dense,selected).to(kInt32).contiguous();
        if(p.value("variant","")=="legacy")return {selected};
        return {mfq_qwen4_exp::qsa_graph_select(scores,positions,pool,budget),selected};
    }
    if(op=="qsa_decode_scores") {
        const auto& q=a.at(0);const auto& k=a.at(1);
        if(p.value("variant","")=="fast")return {mfq_qwen4_exp::block_scores(q,k)};
        auto original=matmul(q.to(kFloat32).contiguous(),k.to(kFloat32).transpose(-1,-2).unsqueeze(1)).clamp_min(0.0).sum(-2)/std::sqrt(double(q.size(-1)));
        if(p.value("variant","")=="legacy")return {original};
        auto fast=mfq_qwen4_exp::block_scores(q,k);
        return {fast,original,std::get<1>(topk(fast,512,-1,true,false)),std::get<1>(topk(original,512,-1,true,false))};
    }
    if(op=="fill_decode_positions") {
        auto cache=a.at(0),positions=a.at(1);
        if(p.value("strided",false))positions=positions.reshape({-1,2}).select(1,0);
        if(p.value("alias",false))positions=cache;
        const int64_t start=p.value("start",int64_t(0)),delta=p.value("delta",int64_t(0));
        if(p.value("legacy",false)) {
            cache.copy_(mfq_tensor_backend::arange(start,start+cache.numel(),cache.options()));
            auto semantic=(cache+delta).reshape({1,cache.numel()}).expand({positions.numel()/cache.numel(),cache.numel()});
            positions.copy_(semantic.reshape(positions.sizes()));
        } else mfq_qwen4_exp::fill_decode_positions(cache,positions,start,delta);
        return {cache,p.value("benchmark_only",false)?positions:positions.to(kInt64)};
    }
    if(op=="qwen4_predictor_graph")return mfq::cuda::tests::qwen4_predictor_graph_check(a,p);
    if(op=="qwen4_session")return mfq::cuda::tests::qwen4_session_check(p);
    if(op=="qwen4_graph_reuse")return mfq::cuda::tests::qwen4_graph_reuse(p);
    if(op=="gdn_window")return mfq::cuda::tests::gdn_window(a,p);
    if(op=="gdn_checkpoint")return mfq::cuda::tests::gdn_stateful(a,p);
    if(op=="gdn_decode_core")
        return gdn_decode_core_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),a.at(5),
            a.at(6),a.at(7),a.at(8),a.at(9),p.at("key_heads"),p.at("value_heads"),
            p.at("width"),1e-6,p.value("eps",1e-6),p.value("silu_gate",false),
            p.value("output_half",false),true,Tensor{});
    if (op == "runtime_gdn") {
        const auto linear = [&](int index) -> mfq::cuda::qwen4_exp::Linear {
            auto w=a.at(index);return [w](CudaExecutionContext&, const Tensor& x) {return matmul(x.to(w.scalar_type()),w.transpose(-1,-2));};
        };
        mfq::cuda::qwen4_exp::GdnWeights w{linear(1),linear(2),linear(3),linear(4),linear(5),a.at(6),a.at(7),a.at(8),a.at(9)};
        mfq::cuda::qwen4_exp::Gdn block(std::move(w),p.at("key_heads"),p.at("value_heads"),p.at("width"),
            p.at("kernel"),p.value("eps",1e-6),p.value("silu_gate",false),
            true,true,true,true,true,p.value("fused_core",true));
        std::vector<Tensor> out;
        for (const auto& step:p.at("steps")) {
            if (step.value("reset",false)) block.reset();
            else if (step.value("commit",false)) block.commit();
            else if (step.value("rollback",false)) block.rollback();
            else {
                out.push_back(block.forward(execution,a.at(0).narrow(1,step.at("begin"),step.at("count")),step.value("cache",true),step.value("confirmed",0)));
                out.push_back(block.conv_state().clone());out.push_back(block.recurrent_state().clone());
            }
        }
        return out;
    }
    if (op == "runtime_ngram" || op == "runtime_ple") {
        const auto weights=a.at(op=="runtime_ngram"?1:2);
        std::vector<mfq::cuda::qwen4_exp::Embedding> shards;
        for (int64_t i=0;i<weights.size(0);++i) {
            auto w=weights.select(0,i);
            shards.push_back([w](const Tensor& ids) {
                auto shape=ids.sizes().vec();shape.push_back(w.size(1));
                return w.index_select(0,ids.reshape({-1}).to(kInt64)).reshape(shape);
            });
        }
        mfq::cuda::qwen4_exp::NgramEmbedding embedding(std::move(shards),weights.size(1),weights.size(2),p.at("ngram"),
            p.at("heads_per_ngram"),p.at("eos"),p.at("multipliers").get<std::vector<int64_t>>(),
            p.at("offsets").get<std::vector<int64_t>>(),p.at("vocab").get<std::vector<int64_t>>());
        std::vector<Tensor> out;
        if (op=="runtime_ngram") {
            for (const auto& step:p.at("steps")) {
                if (step.value("reset",false)) embedding.reset();
                else {
                    Tensor ids;
                    out.push_back(embedding.forward(a.at(0).narrow(1,step.at("begin"),step.at("count")),step.value("cache",true),&ids));
                    out.push_back(ids);
                }
            }
        } else {
            const auto linear=[&](int i)->mfq::cuda::qwen4_exp::Linear {
                auto w=a.at(i);return [w](CudaExecutionContext&, const Tensor& x) {return matmul(x.to(w.scalar_type()),w.transpose(-1,-2));};
            };
            mfq::cuda::qwen4_exp::PleWeights w{linear(3),linear(4),a.at(5),a.at(6),a.at(7),a.at(8)};
            mfq::cuda::qwen4_exp::Ple block(std::move(embedding),std::move(w),p.at("hidden"),p.at("streams"),p.at("ngram"),p.value("eps",1e-6));
            for (const auto& step:p.at("steps")) {
                if (step.value("reset",false)) block.reset();
                else if (step.value("commit",false)) block.commit();
                else if (step.value("rollback",false)) block.rollback();
                else {
                    const int64_t begin=step.at("begin"),count=step.at("count");
                    out.push_back(block.forward(execution,a.at(0).narrow(1,begin,count),a.at(1).narrow(1,begin,count),step.value("cache",true),step.value("confirmed",0)));
                    out.push_back(block.conv_state().clone());
                }
            }
        }
        return out;
    }
    if (op == "runtime_qsa") {
        const auto linear = [&](int index) -> mfq::cuda::qwen4_exp::Linear {
            auto weight=a.at(index);
            return [weight](CudaExecutionContext&, const Tensor& x) {return matmul(x.to(weight.scalar_type()),weight.transpose(-1,-2));};
        };
        auto rotary=std::make_shared<mfq::cuda::RotaryEmbedding>(p.at("rotary"),p.at("maximum"),p.value("base",1e7),
            p.value("sections",std::vector<int64_t>{}),p.value("interleaved",false));
        mfq::cuda::qwen4_exp::QsaWeights weights{linear(1),linear(2),linear(3),linear(4),linear(5),a.at(6),a.at(7),a.at(8),a.at(9)};
        mfq::cuda::qwen4_exp::QsaConfig config{p.at("heads"),p.at("kv_heads"),p.at("width"),p.at("index_heads"),
            p.at("index_width"),p.at("pool"),p.at("budget"),p.at("maximum"),p.value("eps",1e-6)};
        mfq::cuda::qwen4_exp::Qsa block(std::move(weights),config,rotary);
        Tensor history;
        std::vector<Tensor> out;
        for (const auto& step:p.at("steps")) {
            if (step.value("reset",false)) {block.reset();history={};}
            else if (step.contains("truncate")) {
                block.truncate(step.at("truncate"));
                history=history.narrow(-1,0,step.at("truncate"));
            } else {
                auto pos=a.at(10).narrow(-1,step.at("begin"),step.at("count"));
                const bool cache=step.value("cache",true);
                auto full=cache && history.defined()?cat({history,pos},-1):pos;
                std::vector<Tensor> trace;
                out.push_back(block.forward(execution,a.at(0).narrow(1,step.at("begin"),step.at("count")),pos,full,cache,
                    p.value("trace",false)?&trace:nullptr));
                out.insert(out.end(),trace.begin(),trace.end());
                if (cache) history=full;
            }
        }
        return out;
    }
    if (op == "runtime_select_pooled_blocks") return {mfq::cuda::attention_ops::select_pooled_blocks(
        a.at(0), p.at("query_offset"), p.at("logical_length"), p.at("pool"), p.at("budget"), p.at("tail"))};
    if (op == "runtime_sequence_cache") {
        mfq::cuda::attention_ops::SequenceCache cache(p.at("maximum"), a.at(0).size(-1));
        std::vector<Tensor> out;
        for (const auto& step : p.at("steps")) {
            if (step.contains("truncate")) cache.truncate(step.at("truncate"));
            else if (step.value("reset", false)) cache.reset();
            else out.push_back(cache.append(a.at(0).narrow(1, step.at("begin"), step.at("count"))).clone());
        }
        return out;
    }
    if (op == "runtime_kda") {
        const auto linear = [&](int index) -> mfq::cuda::glm5_next::Linear {
            auto weight = a.at(index);
            return [weight](CudaExecutionContext&, const Tensor& x) { return matmul(x.to(weight.scalar_type()), weight.transpose(-1,-2)); };
        };
        mfq::cuda::glm5_next::KdaWeights weights{linear(1),linear(2),linear(3),linear(4),linear(5),linear(6),linear(7),
            a.at(8),a.at(9),a.at(10),a.at(11),a.at(12),a.at(13)};
        mfq::cuda::glm5_next::Kda block(std::move(weights), p.at("heads"), p.at("width"), p.at("kernel"),
            p.value("lower_bound", -5.0), p.value("eps", 1e-5));
        std::vector<Tensor> out;
        for (const auto& step : p.at("steps")) {
            if (step.value("rollback", false)) block.rollback();
            else if (step.value("commit", false)) block.commit();
            else if (step.value("reset", false)) block.reset();
            else {
                out.push_back(block.forward(execution,a.at(0).narrow(1, step.at("begin"), step.at("count")),
                    step.value("cache", true), step.value("confirmed", 0)));
                out.push_back(block.conv_state().clone());
                out.push_back(block.recurrent_state().clone());
            }
        }
        return out;
    }
    if(op.rfind("hot_kv_",0)==0)return mfq::cuda::tests::hot_kv_check(op,a,p);
    if(op.rfind("turbo_kv_",0)==0)return mfq::cuda::tests::quant_kv_check(op,a,p);
    if(op=="nint_dispatch") {
        CudaExecutionContext execution;NintWeight weight;
        weight.q_packed=a.at(0);weight.row_q_bits=a.at(1);weight.row_q_bit_offsets=a.at(2);
        weight.sub_scale=a.at(3);weight.sub_min=a.at(4);weight.neuron_scale=a.at(5);weight.neuron_min=a.at(6);
        weight.out=a.at(3).size(0);weight.ng=a.at(3).size(1);weight.gs=p.at("group_size");
        weight.neuron_len=a.at(7).size(1);weight.uniform_qbits=p.value("hint",0);
        Tensor output;
        const auto project=[&] {
            const int mode=p.value("activation_mode",0);
            output=mode?nint_matmul_input_mul(execution.profiler,weight,a.at(7),a.at(8),mode):
                nint_matmul(execution.profiler,weight,a.at(7));
        };
        if(!p.value("capture",true)) {project();return {output};}
        struct Pool {
            std::shared_ptr<mfq::cuda::Context> context;
            hipStream_t stream;
            Pool(int device):context(mfq::cuda::default_context(device)),stream(mfq_current_cuda_stream()) {
                context->begin_graph_pool(stream);
            }
            ~Pool() {context->end_graph_pool(stream);}
        } pool(a.at(7).get_device());
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream(),false);
        window.capture(project,[]{},[&]{output={};});
        window.run();return {output.clone()};
    }
    if(op=="nint_online") {
        if(p.value("legacy",false)) {
            auto weight=nint_decode_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),a.at(5),a.at(6),
                a.at(7).size(1),p.at("group_size"));
            return {matmul(a.at(7),weight.transpose(0,1))};
        }
        return {nint_matmul_online_f16_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),a.at(5),a.at(6),a.at(7),p.at("group_size"))};
    }
    if(op=="nint_fixed") {
        const auto x=a.at(7);const int64_t gs=p.at("group_size"),groups=a.at(3).size(1);
        auto codes=empty({x.size(0),groups*gs},x.options().dtype(kInt8));
        auto scales=empty({x.size(0),groups},x.options().dtype(kFloat32));
        Tensor output;const int mode=p.value("activation_mode",0),hint=p.value("hint",0);
        if(mode)output=hint?nint_matmul_input_mul_fixed_bits_ws_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),a.at(5),a.at(6),
            x,a.at(8),mode,gs,codes,scales,hint):nint_matmul_input_mul_ws_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),a.at(5),a.at(6),
            x,a.at(8),mode,gs,codes,scales);
        else output=hint?nint_matmul_fixed_bits_ws_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),a.at(5),a.at(6),
            x,gs,codes,scales,hint):nint_matmul_ws_cuda(a.at(0),a.at(1),a.at(2),a.at(3),a.at(4),a.at(5),a.at(6),x,gs,codes,scales);
        if(p.value("benchmark_only",false))return {output};
        return {output,codes.to(kInt64),scales};
    }
    if(op=="native_matmul") {
        auto left=a.at(0),right=a.at(1);
        if(p.value("transpose_left",false))left=left.transpose(-1,-2);
        if(p.value("transpose_right",false))right=right.transpose(-1,-2);
        return {matmul(left,right)};
    }
    if (op == "qwen4_grouped_rms_norm") return {mfq_qwen4_exp::grouped_rms_norm(a.at(0), a.at(1), p.at("group_size"), p.value("eps", 1e-6))};
    if (op == "qwen4_gated_residual_pre") return mfq_qwen4_exp::gated_residual_pre(a.at(0), a.at(1), a.at(2), a.at(3), optional(4), p.at("hidden_size"), p.value("hc_count", 4), p.value("eps", 1e-6));
    if (op == "qwen4_gated_residual_post") return {mfq_qwen4_exp::gated_residual_post(a.at(0), a.at(1), a.at(2), p.value("hc_count", 4))};
    if (op == "glm5_mhc_pre") return mfq_glm5_next::mhc_pre(a.at(0), a.at(1), a.at(2), a.at(3), p.value("sinkhorn_iterations", 20), p.value("hc_eps", 1e-6), p.value("rms_eps", 1e-5));
    if (op == "glm5_mhc_post") return {mfq_glm5_next::mhc_post(a.at(0), a.at(1), a.at(2), a.at(3))};
    if (op == "glm5_kda_forget_gate") return {mfq_glm5_next::kda_forget_gate(a.at(0), a.at(1), a.at(2), a.at(3), a.at(4), p.at("num_heads"), p.at("head_dim"), p.value("lower_bound", -5.0))};
    if (op == "qsa_block_scores") return {mfq_qwen4_exp::block_scores(a.at(0), a.at(1))};
    if (op == "glm5_kpool_scores") return {mfq_glm5_next::kpool_scores(a.at(0), a.at(1), a.at(2))};
    if (op == "glm5_kpool_states") return {mfq_glm5_next::kpool_states(a.at(0), a.at(1), a.at(2), p.value("pool_size", 4))};
    if (op == "qwen4_ple_dilated_conv_silu") return mfq_qwen4_exp::ple_dilated_conv_silu(a.at(0), a.at(1), optional(2), p.at("dilation"));
    if (op == "qwen4_dense_gqa_attention") return {mfq_qwen4_exp::dense_gqa_attention(a.at(0), a.at(1), a.at(2), p.at("query_offset"))};
    if (op == "qwen4_sparse_gqa_attention") return {mfq_qwen4_exp::sparse_gqa_attention(a.at(0), a.at(1), a.at(2), a.at(3))};
    if (op == "glm5_dense_mla_attention") return {mfq_glm5_next::dense_mla_attention(a.at(0), a.at(1), p.at("query_offset"), scale)};
    if (op == "glm5_sparse_mla_attention") return {mfq_glm5_next::sparse_mla_attention(a.at(0), a.at(1), a.at(2), scale)};
    throw std::runtime_error("unknown Flash-Next operation");
}

Json output(const std::vector<Tensor>& tensors) {
    Json result = Json::array();
    for (const auto& value : tensors) {
        if (!value.defined()) { result.push_back(nullptr); continue; }
        auto host = value.to(kFloat32).contiguous().cpu();
        std::vector<float> data(host.numel());
        if (host.numel()) std::copy_n(host.data_ptr<float>(), host.numel(), data.data());
        int64_t nan_count=0,positive_infinity_count=0,negative_infinity_count=0;
        for(const float x:data) {
            nan_count+=std::isnan(x);
            positive_infinity_count+=std::isinf(x) && x>0;
            negative_infinity_count+=std::isinf(x) && x<0;
        }
        result.push_back({{"shape", value.sizes().vec()}, {"data", data},
            {"nan_count",nan_count},{"positive_infinity_count",positive_infinity_count},
            {"negative_infinity_count",negative_infinity_count},
            {"dtype", value.scalar_type() == kFloat16 ? "float16" :
                value.scalar_type() == kBFloat16 ? "bfloat16" : "float32"}});
    }
    return result;
}
} // namespace

void check_rotary_fusion() {
    using namespace mfq::cuda;
    int ordinary_cases=0,graph_cases=0,switch_cases=0;
    const auto exact=[](const Tensor& actual,const Tensor& expected) {
        const auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
        if(a.sizes()!=b.sizes() || a.scalar_type()!=b.scalar_type() ||
            std::memcmp(a.data_ptr(),b.data_ptr(),size_t(a.numel()*a.element_size())))
            throw std::runtime_error("fused rotary original output bytes differ, case "+std::to_string(a.numel()));
    };
    const std::array<std::array<int64_t,5>,3> shapes{{{1,24,1,256,96},{2,3,3,128,64},{2,2,35,17,12}}};
    for(auto dtype:{kFloat16,kFloat32,kBFloat16})for(const auto& shape:shapes)
        for(int layout=1;layout<=4;++layout)for(int mode=0;mode<3;++mode) {
            const auto b=shape[0],h=shape[1],t=shape[2],d=shape[3],rotary=shape[4],pairs=rotary/2;
            std::vector<float> values(size_t(b*h*t*d));
            for(size_t i=0;i<values.size();++i) {
                values[i]=float(std::sin(double(i)*.031)*.71);
                if(i%17==0)values[i]=0.f;
                if(i%17==1)values[i]=-0.f;
                if(i%17==2)values[i]=std::numeric_limits<float>::denorm_min();
                if(i%17==3)values[i]=-std::numeric_limits<float>::denorm_min();
            }
            auto x=tensor(values).reshape({b,h,t,d}).to(kCUDA,dtype);
            if(layout==2)x=x.transpose(-1,-2).contiguous().transpose(-1,-2);
            const int64_t axes=layout==1?1:(layout==2?2:(layout==3?3:5));
            const int64_t batches=layout==3?b:1;
            const auto position_shape=layout==1?std::vector<int64_t>{t}:
                (layout==3 || layout==4?std::vector<int64_t>{axes,batches,t}:std::vector<int64_t>{axes,t});
            std::vector<int64_t> positions(size_t(axes*batches*t));
            const std::array<int64_t,9> choices{-5,0,1,96,511,512,2147483647LL,2147483648LL,-2147483649LL};
            for(size_t i=0;i<positions.size();++i)positions[i]=choices[i%choices.size()];
            auto p=tensor(positions).reshape(position_shape).to(kCUDA);
            std::vector<int64_t> sections=mode==0?std::vector<int64_t>{}:std::vector<int64_t>{pairs/3,pairs/3,pairs-2*(pairs/3)};
            RotaryEmbedding original(rotary,512,1e7,sections,mode==2,false);
            RotaryEmbedding fused(rotary,512,1e7,sections,mode==2,true);
            exact(fused.forward(x,p),original.forward(x,p));++ordinary_cases;
            if(mode==2 && layout==3) {
                std::array<StreamHandle,2> streams{stream_from_pool(false),stream_from_pool(false)};
                std::array<Graph,2> graphs;std::array<Tensor,2> outputs;
                for(int variant=0;variant<2;++variant) {
                    StreamGuard guard(streams[variant]);
                    auto& operation=variant?fused:original;
                    graphs[variant].prepare_memory();outputs[variant]=operation.forward(x,p);
                    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(current_stream(0).stream()));
                    outputs[variant]=Tensor{};
                    graphs[variant].capture_begin();outputs[variant]=operation.forward(x,p);graphs[variant].capture_end();
                }
                for(int step=0;step<3;++step) {
                    for(size_t i=0;i<positions.size();++i)positions[i]=choices[(i+step+4)%choices.size()];
                    p.copy_(tensor(positions).reshape(position_shape).to(kCUDA));
                    x.copy_(tensor(values).reshape({b,h,t,d}).to(kCUDA,dtype)*(float(step)-1.f));
                    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(current_stream(0).stream()));
                    for(int variant=0;variant<2;++variant) {
                        StreamGuard guard(streams[variant]);graphs[variant].replay();
                        MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(streams[variant].stream()));
                    }
                    exact(outputs[1],outputs[0]);++graph_cases;
                }
            }
            fused.set_fused(false);
            exact(fused.forward(x,p),original.forward(x,p));++switch_cases;
            fused.set_fused(true);
            exact(fused.forward(x,p),original.forward(x,p));++switch_cases;
        }
    std::cout<<"rotary_fused_ordinary_exact_cases="<<ordinary_cases<<" changing_graph_exact_cases="<<graph_cases
             <<" mode_switch_exact_cases="<<switch_cases<<" PASS\n";
}

void check_rotary_normalized_fusion() {
    using namespace mfq::cuda;
    int ordinary_cases=0,cache_cases=0,graph_cases=0;
    const auto exact=[](const Tensor& actual,const Tensor& expected,const char* label) {
        const auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
        if(a.sizes()!=b.sizes() || a.scalar_type()!=b.scalar_type() ||
            std::memcmp(a.data_ptr(),b.data_ptr(),size_t(a.numel()*a.element_size())))
            throw std::runtime_error(std::string("normalized rotary bytes differ: ")+label);
    };
    const std::array<std::array<int64_t,5>,7> shapes{{
        {1,20,1,128,96},{1,24,1,256,96},{2,3,3,128,64},{2,2,35,17,12},
        {1,3,6,32,16},{1,2,3,64,48},{1,2,3,513,128}}};
    for(auto dtype:{kFloat16,kFloat32})for(const auto& shape:shapes)
        for(int layout=1;layout<=4;++layout)for(int mode=0;mode<3;++mode) {
            const auto b=shape[0],h=shape[1],t=shape[2],d=shape[3],rotary=shape[4],pairs=rotary/2;
            std::vector<float> values(size_t(b*t*h*d*2)),weights(static_cast<size_t>(d));
            for(size_t i=0;i<values.size();++i) {
                values[i]=float(std::sin(double(i)*.031)*.71);
                if(i%17==0)values[i]=0.f;
                if(i%17==1)values[i]=-0.f;
                if(i%17==2)values[i]=std::numeric_limits<float>::denorm_min();
                if(i%17==3)values[i]=-std::numeric_limits<float>::denorm_min();
            }
            for(size_t i=0;i<weights.size();++i)weights[i]=i%11==0?-1.f:float(std::cos(double(i)*.07)*.13);
            auto joined=tensor(values).reshape({b,t,h,2*d}).to(kCUDA,dtype);
            auto x=joined.narrow(-1,0,d),v=joined.narrow(-1,d,d);
            if(layout==2)x=x.transpose(-1,-2).contiguous().transpose(-1,-2);
            if(layout==4)v=v.transpose(-1,-2).contiguous().transpose(-1,-2);
            const auto source=x.clone(),source_value=v.clone();
            auto weight=tensor(weights).to(kCUDA);
            const int64_t axes=layout==1?1:(layout==2?2:(layout==3?3:5));
            const int64_t batches=layout==3?b:1;
            const auto position_shape=layout==1?std::vector<int64_t>{t}:
                (layout==3 || layout==4?std::vector<int64_t>{axes,batches,t}:std::vector<int64_t>{axes,t});
            std::vector<int64_t> positions(size_t(axes*batches*t)),slots(static_cast<size_t>(t));
            const std::array<int64_t,9> choices{-5,0,1,96,511,512,2147483647LL,2147483648LL,-2147483649LL};
            for(size_t i=0;i<positions.size();++i)positions[i]=choices[i%choices.size()];
            for(size_t i=0;i<slots.size();++i)slots[i]=5+3*i;
            const auto index_dtype=layout%2?kInt64:kInt32;
            auto p=tensor(positions).reshape(position_shape).to(kCUDA,index_dtype);
            auto cache_positions=tensor(slots).to(kCUDA,index_dtype);
            std::vector<int64_t> sections=mode==0?std::vector<int64_t>{}:std::vector<int64_t>{pairs/3,pairs/3,pairs-2*(pairs/3)};
            RotaryEmbedding original(rotary,512,1e7,sections,mode==2,false);
            RotaryEmbedding fused(rotary,512,1e7,sections,mode==2,true);
            const auto options=x.options().dtype(kFloat16);
            std::array<Tensor,2> keys{full({b,h,3*t+8,d},-.17,options),full({b,h,3*t+8,d},-.17,options)};
            std::array<Tensor,2> cached_values{full({b,h,3*t+8,d},.23,options),full({b,h,3*t+8,d},.23,options)};
            std::array<Tensor,2> outputs;
            const auto forward=[&](int variant) {
                if(variant) {
                    outputs[1]=fused.forward_normalized(x,weight,p,1e-6);
                    auto unused=fused.forward_normalized(x,weight,p,1e-6,keys[1],cache_positions,v,cached_values[1]);
                    if(unused.defined())throw std::runtime_error("direct cached rotary allocated an output");
                } else {
                    auto norm=grouped_rms_norm_cuda(x.contiguous(),weight,d,1e-6,1.0);
                    outputs[0]=original.forward(norm.permute({0,2,1,3}),p);
                    keys[0].index_copy_(2,cache_positions,outputs[0].to(kFloat16));
                    cached_values[0].index_copy_(2,cache_positions,v.permute({0,2,1,3}).to(kFloat16));
                }
            };
            forward(0);forward(1);
            exact(outputs[1],outputs[0],"output");++ordinary_cases;
            exact(keys[1],keys[0],"keys");exact(cached_values[1],cached_values[0],"values");++cache_cases;
            if(mode==2 && layout==3) {
                std::array<StreamHandle,2> streams{stream_from_pool(false),stream_from_pool(false)};
                std::array<Graph,2> graphs;
                for(int variant=0;variant<2;++variant) {
                    StreamGuard guard(streams[variant]);graphs[variant].prepare_memory();forward(variant);
                    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(streams[variant].stream()));outputs[variant]=Tensor{};
                    graphs[variant].capture_begin();forward(variant);graphs[variant].capture_end();
                }
                for(int step=0;step<3;++step) {
                    for(size_t i=0;i<positions.size();++i)positions[i]=choices[(i+step+4)%choices.size()];
                    for(size_t i=0;i<slots.size();++i)slots[i]=5+3*i+step;
                    p.copy_(tensor(positions).reshape(position_shape).to(kCUDA,index_dtype));
                    cache_positions.copy_(tensor(slots).to(kCUDA,index_dtype));
                    x.copy_(source*(float(step)-1.f));v.copy_(source_value*(float(step)*.5f-1.f));
                    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(current_stream(0).stream()));
                    for(int variant=0;variant<2;++variant) {
                        StreamGuard guard(streams[variant]);graphs[variant].replay();
                        MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(streams[variant].stream()));
                    }
                    try { exact(outputs[1],outputs[0],"graph output"); }
                    catch(const std::exception&) {
                        std::cerr<<"normalized graph mismatch dtype="<<int(dtype)<<" B="<<b<<" H="<<h<<" T="<<t<<" D="<<d<<" step="<<step<<'\n';
                        throw;
                    }
                    exact(keys[1],keys[0],"graph keys");
                    exact(cached_values[1],cached_values[0],"graph values");++graph_cases;
                }
            }
        }
    std::cout<<"normalized_rotary_ordinary_exact_cases="<<ordinary_cases<<" cache_exact_cases="<<cache_cases
             <<" changing_graph_exact_cases="<<graph_cases<<" PASS\n";
}

void check_attention_fusion() {
    using namespace mfq::cuda;
    int ordinary_cases=0,gate_cases=0,graph_cases=0;
    const auto exact=[](const Tensor& actual,const Tensor& expected,const char* label) {
        const auto a=actual.cpu().contiguous(),b=expected.cpu().contiguous();
        if(a.sizes()!=b.sizes() || a.scalar_type()!=b.scalar_type() ||
            std::memcmp(a.data_ptr(),b.data_ptr(),size_t(a.numel()*a.element_size())))
            throw std::runtime_error(std::string("attention fusion bytes differ: ")+label);
    };
    const std::array<std::array<int64_t,7>,6> shapes{{
        {1,20,2,1,128,128,163},{1,24,2,1,256,33,41},
        {2,4,2,3,17,17,23},{1,4,2,2,513,17,23},
        {1,2,1,2,1025,9,13},{1,4,2,3,64,0,13}}};
    for(auto dtype:{kFloat16,kFloat32})for(auto gate_dtype:{kFloat16,kFloat32})
        for(bool half_output:{false,true})for(const auto& shape:shapes)for(int layout=0;layout<2;++layout) {
            const auto b=shape[0],h=shape[1],kh=shape[2],t=shape[3],d=shape[4],capacity=shape[5],columns=shape[6];
            const auto make=[&](int64_t count,float scale,float shift) {
                std::vector<float> numbers(static_cast<size_t>(count));
                for(size_t i=0;i<numbers.size();++i) {
                    numbers[i]=float(std::sin(double(i)*.043+shift)*scale);
                    if(i%31==0)numbers[i]=0.f;
                    if(i%31==1)numbers[i]=-0.f;
                    if(i%31==2)numbers[i]=std::numeric_limits<float>::denorm_min();
                    if(i%31==3)numbers[i]=-std::numeric_limits<float>::denorm_min();
                }
                return tensor(numbers).to(kCUDA);
            };
            auto q=make(b*h*t*d,.71f,.13f).reshape({b,h,t,d}).to(dtype);
            auto k=make(b*kh*capacity*d,.37f,.27f).reshape({b,kh,capacity,d}).to(kFloat16);
            auto v=make(b*kh*capacity*d,1.31f,.31f).reshape({b,kh,capacity,d}).to(kFloat16);
            auto joined=make(b*t*h*d*2,17.f,.43f).reshape({b,t,h,2*d}).to(gate_dtype);
            auto gate=joined.narrow(-1,d,d);
            if(!layout)gate=gate.contiguous();
            else {q=q.transpose(-1,-2).contiguous().transpose(-1,-2);v=v.transpose(-1,-2).contiguous().transpose(-1,-2);}
            auto source_q=q.clone(),source_gate=gate.clone();
            const auto index_dtype=layout?kInt64:kInt32;
            std::vector<int64_t> choices{-5,0,capacity-1,capacity,2147483648LL,-2147483649LL};
            std::vector<int64_t> ends(static_cast<size_t>(t));
            for(size_t i=0;i<ends.size();++i)ends[i]=choices[(i+2)%choices.size()];
            auto positions=tensor(ends).to(kCUDA,index_dtype);
            const auto reference=[&] {
                auto selected=arange(columns,positions.options()).reshape({1,1,columns}).expand({b,t,columns});
                selected=where(selected<=positions.reshape({1,t,1}),selected,full_like(selected,-1));
                auto attended=mfq_qwen4_exp::sparse_gqa_attention(q,k,v,selected);
                auto out=(attended.to(kFloat32)*sigmoid(gate.to(kFloat32))).to(half_output?kFloat16:kFloat32);
                exact(mfq_qwen4_exp::attention_gate(attended,gate,half_output),out,"gate");++gate_cases;
                return out;
            };
            const auto candidate=[&] {return mfq_qwen4_exp::causal_gqa_attention_gate(q,k,v,positions,gate,columns,half_output);};
            try {exact(candidate(),reference(),"causal output");}
            catch(const std::exception&) {std::cerr<<"attention fusion mismatch D="<<d<<" dtype="<<int(dtype)
                <<" gate_dtype="<<int(gate_dtype)<<" half_output="<<half_output<<" layout="<<layout<<'\n';throw;}
            ++ordinary_cases;
            if(layout && gate_dtype==kFloat16) {
                auto stream=stream_from_pool(false);Graph graph;Tensor output;
                {StreamGuard guard(stream);graph.prepare_memory();output=candidate();
                    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(stream.stream()));output={};
                    graph.capture_begin();output=candidate();graph.capture_end();}
                for(int step=0;step<3;++step) {
                    for(size_t i=0;i<ends.size();++i)ends[i]=choices[(i+step+3)%choices.size()];
                    positions.copy_(tensor(ends).to(kCUDA,index_dtype));
                    q.copy_(source_q*(float(step)-1.f));gate.copy_(source_gate*(float(step)*.5f-1.f));
                    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(current_stream(0).stream()));
                    {StreamGuard guard(stream);graph.replay();MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(stream.stream()));}
                    exact(output,reference(),"changing graph");++graph_cases;
                }
            }
        }
    std::cout<<"attention_fusion_ordinary_exact_cases="<<ordinary_cases<<" gate_exact_cases="<<gate_cases
             <<" changing_graph_exact_cases="<<graph_cases<<" PASS\n";
}

std::vector<mfq::cuda::Tensor> check_tq_rows(const nlohmann::json& request) {
    namespace tb=mfq_tensor_backend;
    auto source=mfq::open_model_source(request.at("model").get<std::string>());
    std::vector<std::shared_ptr<TqRowTable>> tables;
    for(const auto& name:request.at("tables"))
        tables.push_back(std::make_shared<TqRowTable>(*source,name.get<std::string>()));
    const tb::Device device{tb::kCUDA,0};
    const auto rows=request.at("rows_per_shard").get<int64_t>();
    const auto ids=request.at("ids").get<std::vector<int64_t>>();
    const auto changed=request.at("changed_ids").get<std::vector<int64_t>>();
    const std::vector<int64_t> shape{int64_t(ids.size())};
    auto stage=make_tq_row_stage(tables,rows,shape,device);
    auto direct=lookup_tq_shards(tables,rows,ids,device).cpu();
    stage->upload(ids,shape);stage->decode();
    auto staged=stage->output().cpu();
    mfq::cuda::Graph graph;
    graph.prepare_memory();stage->decode();
    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(mfq::cuda::current_stream(0).stream()));
    graph.capture_begin();stage->decode();graph.capture_end();
    graph.replay();auto captured=stage->output().cpu();
    stage->upload(changed,shape);graph.replay();auto updated=stage->output().cpu();
    auto expected_changed=lookup_tq_shards(tables,rows,changed,device).cpu();
    return {direct,staged,captured,updated,expected_changed};
}

int main(int argc, char** argv) {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) return 77;
    try {
        const auto context = mfq::cuda::default_context(0);
        auto stream = mfq::cuda::stream_from_pool(false, 0);
        mfq::cuda::StreamGuard stream_guard(stream);
        if(argc==2 && std::string(argv[1])=="--rotary-fused-check") {check_rotary_fusion();return 0;}
        if(argc==2 && std::string(argv[1])=="--rotary-normalized-check") {check_rotary_normalized_fusion();return 0;}
        if(argc==2 && std::string(argv[1])=="--attention-fused-check") {check_attention_fusion();return 0;}
        if (argc == 1) {
            auto q = mfq::cuda::ones({1,1,2,4}, mfq::cuda::TensorOptions{}.device(mfq::cuda::kCUDA));
            auto k = mfq::cuda::ones({1,3,4}, q.options());
            auto got = mfq_qwen4_exp::block_scores(q,k).cpu();
            for (int i = 0; i < 3; ++i)
                if (got.data_ptr<float>()[i] != 4.f) throw std::runtime_error("QSA smoke mismatch");
            check_qsa_sparse_attention();
            std::cout << "Qwen Flash-Next sparse-attention native smoke passed\n";
            return 0;
        }
        if (std::string(argv[1]) != "--json") throw std::runtime_error("expected --json");
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                const auto request = Json::parse(line);
                if(request.at("op")=="source_info") {
                    const auto& params=request.at("params");
                    auto source=mfq::open_model_source(params.at("model").get<std::string>());
                    Json response={{"architecture",std::string(source->architecture())},
                        {"tensors",source->tensors().size()},{"assets",source->assets().size()}};
                    if(params.value("load_dspark",false)) {
                        CudaExecutionContext execution;
                        execution.expert_parallel.devices={0,1};
                        execution.model_parallel_collectives.configure({0,1},false);
                        auto config=mfq::models::deepseek_v41::Config::from_source(*source);
                        config.max_position_embeddings=4608;
                        auto predictor=mfq::cuda::deepseek_v41_runtime::load_dspark_if_present(execution,*source,config);
                        if(!predictor)throw std::runtime_error("native model predictor is absent");
                        response["dspark_loaded"]=true;
                    }
                    for(const auto& name:params.at("mxfp8")) {
                        const auto key=name.get<std::string>();
                        const auto* tensor=source->find_tensor(key);
                        if(!tensor)throw std::runtime_error("missing tensor: "+key);
                        std::vector<uint8_t> blob(tensor->nbytes);
                        source->tensor_reader(key)(0,reinterpret_cast<std::byte*>(blob.data()),blob.size());
                        const auto weight=unpack_mxfp8(blob);
                        response["mxfp8"].push_back({{"name",key},{"rows",weight.out},
                            {"columns",weight.neuron_len},{"block_rows",weight.block_rows},
                            {"block_columns",weight.block_columns}});
                    }
                    std::cout<<response.dump()<<std::endl;
                    continue;
                }
                if(request.at("op")=="tq_rows") {
                    std::cout<<Json({{"outputs",output(check_tq_rows(request.at("params")))}}).dump()<<std::endl;
                    continue;
                }
                std::vector<Tensor> tensors;
                for (const auto& item : request.at("inputs")) tensors.push_back(input(item));
                const auto op = request.at("op").get<std::string>();
                const auto params = request.value("params", Json::object());
                const int capture_repetitions=params.value("capture_repetitions",1);
                MFQ_RUNTIME_CHECK(capture_repetitions>=1 && capture_repetitions<=128,"invalid capture repetition count");
                // Keep parameter caches alive across warmup, capture and replay,
                // as the production model does.
                std::unique_ptr<RotaryEmbedding> rotary;
                CudaExecutionContext generic_execution;
                std::unique_ptr<FFN> generic_ffn;
                std::unique_ptr<TqWeight> gr_down;
                std::unique_ptr<TqWeight> shared_gate,shared_up;
                std::unique_ptr<TqWeight> router_tq;
                std::shared_ptr<TqRoutedWeight> expert_bank;
                std::shared_ptr<TqMoeBanks> expert_moe;
                std::vector<TqWeight> expert_reference;
                if(op=="runtime_tq_expert_pool") {
                    auto source=mfq::open_model_source(params.at("model").get<std::string>());
                    const auto name=params.at("weight").get<std::string>();
                    expert_bank=load_tq_routed_weight(*source,name,0);
                    if(params.value("expert_parallel",false)) {
                        if(params.value("peer_access",false))for(int device=0;device<2;++device) {
                            MfqCudaGuard guard(device);auto status=hipDeviceEnablePeerAccess(1-device,0);
                            if(status==hipErrorPeerAccessAlreadyEnabled)(void)hipGetLastError();
                            else MFQ_CUDA_CHECK(status);
                            hipMemPool_t pool=nullptr;MFQ_CUDA_CHECK(hipDeviceGetDefaultMemPool(&pool,device));
                            hipMemAccessDesc access{};access.location.type=hipMemLocationTypeDevice;
                            access.location.id=1-device;access.flags=hipMemAccessFlagsProtReadWrite;
                            MFQ_CUDA_CHECK(hipMemPoolSetAccess(pool,&access,1));
                        }
                        generic_execution.expert_parallel.devices={0,1};
                        generic_execution.model_parallel_collectives.configure({0,1},false);
                        auto root=std::make_shared<TqRoutedWeight>();root->experts=expert_bank->experts;
                        root->out=expert_bank->out;root->input=expert_bank->input;root->execution=&generic_execution;
                        for(int device=0;device<2;++device) {
                            const int begin=int(root->experts*device/2),end=int(root->experts*(device+1)/2);
                            root->shards.push_back({device,begin,end,std::make_shared<TqRoutedWeight>(*source,name,device,
                                root->experts,root->out,root->input,begin,end)});
                        }
                        expert_bank=std::move(root);
                    }
                    auto reader=source->tensor_reader(name);
                    std::vector<uint8_t> table(64+size_t(expert_bank->experts)*16);
                    reader(0,reinterpret_cast<std::byte*>(table.data()),table.size());
                    for(const auto& expert:params.at("experts")) {
                        const auto index=expert.get<size_t>();
                        if(index>=size_t(expert_bank->experts))throw std::runtime_error("expert index is out of bounds");
                        const auto offset=mfq::TqLayout::integer(table.data()+64+index*16,8);
                        const auto bytes=mfq::TqLayout::integer(table.data()+72+index*16,8);
                        std::vector<uint8_t> blob(bytes);reader(offset,reinterpret_cast<std::byte*>(blob.data()),blob.size());
                        expert_reference.push_back(to_device_tq(blob,expert_bank->input,0));
                    }
                    if(params.value("full_moe",false)) {
                        ::setenv("MFQ_HIP_TQ_EP_PEER",params.value("peer_decode",false)?"1":"0",1);
                        auto* execution=params.value("expert_parallel",false)?&generic_execution:nullptr;
                        expert_moe=std::make_shared<TqMoeBanks>();expert_moe->gate=expert_bank;
                        const auto prefix=name.substr(0,name.size()-std::string("gate.weight").size());
                        expert_moe->up=load_tq_routed_weight(*source,prefix+"up.weight",0,execution);
                        expert_moe->down=load_tq_routed_weight(*source,prefix+"down.weight",0,execution);
                    }
                }
                if(op=="runtime_router_gate") {
                    auto source=mfq::open_model_source(params.at("model").get<std::string>());
                    router_tq=std::make_unique<TqWeight>(load_tq_weight(*source,params.at("router").get<std::string>(),0));
                }
                if(op=="runtime_shared_swiglu") {
                    auto source=mfq::open_model_source(params.at("model").get<std::string>());
                    const auto prefix=params.at("prefix").get<std::string>();
                    shared_gate=std::make_unique<TqWeight>(load_tq_weight(*source,prefix+"gate.weight",0));
                    shared_up=std::make_unique<TqWeight>(load_tq_weight(*source,prefix+"up.weight",0));
                }
                if(op=="runtime_gr_down_injection") {
                    auto source=mfq::open_model_source(params.at("model").get<std::string>());
                    gr_down=std::make_unique<TqWeight>(load_tq_weight(*source,params.at("down").get<std::string>(),0));
                }
                std::unique_ptr<mfq::cuda::qwen4_exp::Gdn> window_gdn;
                if(op=="runtime_gdn_window")window_gdn=mfq::cuda::tests::gdn_window_model(tensors,params);
                Tensor independent_gdn_states;
                if(op=="gdn_independent")independent_gdn_states=mfq::cuda::tests::gdn_state_addresses(tensors,params);

                if(op=="runtime_generic_moe") {
                    generic_execution.decode_graph_serial_branches=request.value("graph",false);
                    auto source=mfq::open_model_source(params.at("model").get<std::string>());
                    generic_ffn=std::make_unique<FFN>(load_moe_weights(generic_execution,*source,"model.mlp.",
                        {.layer=0,.router_bias_required=true}));
                    generic_ffn->moe_top_k=params.at("top_k");
                    generic_ffn->moe_use_sigmoid=true;
                    generic_ffn->moe_delayed_softmax=false;
                    generic_ffn->moe_normalize=true;
                    generic_ffn->moe_shared_ungated=true;
                    generic_ffn->swiglu_limit=params.value("limit",0.);
                }
                if (op == "runtime_rotary") {
                    rotary = std::make_unique<RotaryEmbedding>(
                        params.at("rotary"), params.at("maximum"), params.value("base", 1e7),
                        params.value("sections", std::vector<int64_t>{}), params.value("interleaved", false));
                }
                const auto execute = [&]() -> std::vector<Tensor> {
                    if(expert_moe)return {expert_moe->forward(tensors.at(0),tensors.at(1),tensors.at(2),params.value("limit",10.f))};
                    if(expert_bank) {
                        if(!params.value("legacy",false))return {expert_bank->forward(tensors.at(0),tensors.at(1))};
                        std::vector<Tensor> parts;
                        for(size_t i=0;i<expert_reference.size();++i)parts.push_back(expert_reference[i].forward(
                            tensors.at(0).dim()==3?tensors.at(0).select(1,i).contiguous():tensors.at(0)));
                        return {stack(parts,1)};
                    }
                    if(router_tq) {
                        std::vector<Tensor> result;
                        if(params.value("legacy",false))result={router_tq->forward(tensors.at(1)),
                            mfq_tensor_backend::sigmoid(mfq_selected_attention::promoted_matmul(tensors.at(1),tensors.at(0).transpose(-1,-2)))};
                        else result=tq_router_gate_cuda(router_tq->view,tensors.at(0),tensors.at(1));
                        if(params.value("legacy",false))result.push_back(tensors.at(2)+tensors.at(3)*result[1]);
                        else result.push_back(tq_shared_add_cuda(tensors.at(2),tensors.at(3),result[1]));
                        return result;
                    }
                    if(shared_gate) {
                        if(params.value("legacy",false)) {
                            auto gate=shared_gate->forward(tensors.at(0)),up=shared_up->forward(tensors.at(0));
                            return {(gate*mfq_tensor_backend::sigmoid(gate))*up};
                        }
                        return {tq_shared_swiglu_cuda(shared_gate->view,shared_up->view,tensors.at(0))};
                    }
                    if(gr_down) {
                        const auto streams=params.at("streams").get<int64_t>();
                        if(params.value("legacy",false))return {tq_gr_activation(*gr_down,tensors.at(1),streams,false),
                            dense_gr_injection_cuda(tensors.at(0),tensors.at(1),streams)};
                        return tq_gr_down_injection_cuda(gr_down->view,tensors.at(0),tensors.at(1),streams);
                    }
                    if(window_gdn)return mfq::cuda::tests::gdn_window_model_run(*window_gdn,generic_execution,tensors,params);
                    if(independent_gdn_states.defined())return mfq::cuda::tests::gdn_stateful(tensors,params,independent_gdn_states);

                    if(generic_ffn) {
                        if(params.value("router_bias",false))return {generic_ffn->forward_with_router_bias(
                            generic_execution,tensors.at(0),Tensor{},tensors.at(1))};
                        return {generic_ffn->forward(generic_execution,tensors.at(0))};
                    }
                    if (rotary) return {rotary->forward(tensors.at(0), tensors.at(1))};
                    return run(op, tensors, params);
                };
                std::vector<Tensor> result;
                mfq::cuda::Graph graph;
                if (request.value("graph", false)) {
                    result = execute();
                    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(mfq::cuda::current_stream(0).stream()));
                    graph.prepare_memory();
                    for(int repeat=0;repeat<capture_repetitions;++repeat)result=execute();
                    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(mfq::cuda::current_stream(0).stream()));
                    result.clear();
                    graph.capture_begin();
                    for(int repeat=0;repeat<capture_repetitions;++repeat)result=execute();
                    graph.capture_end();
                    graph.replay();
                    graph.replay();
                    if (request.contains("replay_inputs")) {
                        const auto& updates = request.at("replay_inputs");
                        if (updates.size() != tensors.size()) throw std::runtime_error("replay fixture count mismatch");
                        for (size_t i = 0; i < tensors.size(); ++i) {
                            if (tensors[i].defined()) tensors[i].copy_(input(updates.at(i)));
                        }
                        if(independent_gdn_states.defined() && params.contains("replay_permutation")) {
                            auto changed=params;changed["permutation"]=params.at("replay_permutation");
                            independent_gdn_states.copy_(mfq::cuda::tests::gdn_state_addresses(tensors,changed));
                        }
                        graph.replay();
                    }
                } else result = execute();
                Json response;
                const int iterations=request.value("benchmark_iterations",0);
                if(iterations>0) {
                    if(!request.value("graph",false))throw std::runtime_error("timing requires a captured graph");
                    hipEvent_t begin,end;
                    MFQ_NATIVE_CUDA_CHECK(hipEventCreate(&begin));MFQ_NATIVE_CUDA_CHECK(hipEventCreate(&end));
                    MFQ_NATIVE_CUDA_CHECK(hipEventRecord(begin,stream.stream()));
                    for(int i=0;i<iterations;++i)graph.replay();
                    MFQ_NATIVE_CUDA_CHECK(hipEventRecord(end,stream.stream()));
                    MFQ_NATIVE_CUDA_CHECK(hipEventSynchronize(end));float ms=0;
                    MFQ_NATIVE_CUDA_CHECK(hipEventElapsedTime(&ms,begin,end));
                    MFQ_NATIVE_CUDA_CHECK(hipEventDestroy(begin));MFQ_NATIVE_CUDA_CHECK(hipEventDestroy(end));
                    response["microseconds"]=ms*1000/iterations/capture_repetitions;
                }
                const int host_iterations=request.value("host_benchmark_iterations",0);
                if(host_iterations>0) {
                    MFQ_RUNTIME_CHECK(!request.value("graph",false) && iterations==0 && host_iterations<=10000,
                        "host timing requires eager execution and a bounded iteration count");
                    for(int i=0;i<2;++i)result=execute();
                    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(stream.stream()));
                    const auto begin=std::chrono::steady_clock::now();
                    for(int i=0;i<host_iterations;++i)result=execute();
                    MFQ_NATIVE_CUDA_CHECK(hipStreamSynchronize(stream.stream()));
                    response["microseconds"]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/host_iterations;
                    response["timing"]="host";
                }
                if(request.contains("output_prefix")) {
                    response["outputs"]=Json::array();
                    for(size_t i=0;i<result.size();++i) {
                        MfqCudaGuard result_guard(result[i].get_device());
                        const auto type=result[i].scalar_type()==kInt64?kInt64:kFloat32;
                        auto host=result[i].to(type).contiguous().cpu();
                        const auto path=request.at("output_prefix").get<std::string>()+std::to_string(i)+".bin";
                        std::ofstream file(path,std::ios::binary);
                        if(!file.write(static_cast<const char*>(host.data_ptr()),host.numel()*host.element_size()))
                            throw std::runtime_error("cannot write raw tensor result");
                        response["outputs"].push_back({{"shape",host.sizes().vec()},{"file",path},{"dtype",type==kInt64?"int64":"float32"}});
                    }
                } else if(request.value("return_outputs",true))response["outputs"]=output(result);
                std::cout << response.dump() << std::endl;
            } catch (const std::exception& e) {
                std::cout << Json({{"error", e.what()}}).dump() << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Flash-Next native test failed: " << e.what() << '\n';
        return 1;
    }
}
