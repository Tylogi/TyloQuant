#include "../../runtime/execution_options.h"
#pragma once
#include "cuda_execution.h"
#include "mfq_cuda_resident_allocations.h"
#include "mfq_cuda_linear_attention_ops.h"
#include "mfq_cuda_norm_ops.h"
#include "models/qwen4_exp/causal_lm.h"
#include "models/qwen4_exp/ngram.h"
#include "core/attention.h"
#include "core/rope.h"
#include "core/pooled_cache.h"
#include "mfq/kernels/cuda/qwen4_exp.h"
#include "mfq/kernels/cuda/ple_fused.h"
#include "runtime/nint_row_pipeline.h"
#include "runtime/row_stage.h"
#include "runtime/kv_offload.h"
#include "runtime/kv_memory.h"
#include "runtime/decode_window.h"
#include "runtime/batch_decode_inputs.h"
#include "mfq_cuda_kv_quantization.h"
#include "storage/session_state.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_map>

namespace mfq::cuda::qwen4_exp {
namespace tb = mfq_tensor_backend;
using attention_ops::Tensor;
using attention_ops::Linear;
using attention_ops::Embedding;
using attention_ops::rms_norm;
using attention_ops::SequenceCache;
using attention_ops::select_pooled_blocks;
using LinearGroup = std::function<std::vector<Tensor>(CudaExecutionContext&, const Tensor&)>;


struct GdnWeights {
    Linear qkv, gate, alpha, beta, output;
    Tensor conv, dt_bias, a_log, norm;
    LinearGroup input_projection;
    std::vector<Linear> split_qkv;
};

class Gdn {
  public:
    Gdn(GdnWeights weights, int64_t key_heads, int64_t value_heads, int64_t width, int64_t kernel,
        double eps, bool silu_gate, bool fused_decode = true, bool transposed_state = true,
        bool fused_output = true, bool grouped_projection = true, bool fused_preparation = true,
        bool fused_core = true)
        : w_(std::move(weights)), nk_(key_heads), nv_(value_heads), d_(width), kernel_(kernel),
          eps_(eps), silu_gate_(silu_gate), fused_decode_(fused_decode),
          transposed_state_(fused_decode && transposed_state && (width==32 || width==64 || width==128)),
          fused_output_(fused_output), grouped_projection_(grouped_projection), fused_preparation_(fused_preparation),
          fused_core_(fused_core) {
        MFQ_RUNTIME_CHECK(nk_ > 0 && nv_ > 0 && nv_ % nk_ == 0 && d_ > 0 && kernel_ > 1 && eps_ > 0,
                          "invalid Qwen4 GDN head/convolution geometry");
    }
    void reset(bool retain_storage=false) {
        if(retain_storage) {
            if(conv_.defined())conv_.zero_();
            if(state_.defined())state_.zero_();
            if(core_workspace_.defined())core_workspace_.zero_();
        } else {
            conv_ = Tensor();
            state_ = Tensor();
            core_workspace_ = Tensor();
        }
        commit();
    }
    void commit() {
        saved_conv_ = Tensor();
        saved_state_ = Tensor();
        saved_prefix_conv_.clear();saved_prefix_state_.clear();saved_prefix_position_=-1;
    }
    void rollback(int64_t keep=-1) {
        MFQ_RUNTIME_CHECK(saved_conv_.defined(), "Qwen4 GDN has no speculative checkpoint");
        if(keep>=0 && saved_prefix_position_>=0) {
            const auto index=keep-saved_prefix_position_;
            MFQ_RUNTIME_CHECK(index>=0 && index<int64_t(saved_prefix_conv_.size()),
                "Qwen4 GDN prefix checkpoint is out of range");
            saved_conv_=saved_prefix_conv_[index];saved_state_=saved_prefix_state_[index];
        }
        if(conv_.defined())conv_.copy_(saved_conv_);else conv_=saved_conv_;
        if(state_.defined())state_.copy_(saved_state_);else state_=saved_state_;
        commit();
    }
    const Tensor &conv_state() const { return conv_; }
    void initialize_request_cache(bool allocate_counters=true) {
        // Recurrence is allocated on first use, outside temporary pool pages.
        // Keep the reusable decode counters out of graph scratch as well.
        request_state_storage_=true;
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        ResidentAllocationScope resident;
#endif
        auto options=w_.norm.options().dtype(tb::kFloat32);
        if(allocate_counters && !core_workspace_.defined() && fused_core_ && fused_decode_ &&
            fused_preparation_ && fused_output_ && transposed_state_)
            core_workspace_=tb::zeros({1,nv_,d_+1},options);
    }
    std::size_t growth_bytes() const {
        return (conv_.defined() ? 0 : std::size_t(kernel_-1)*(2*nk_+nv_)*d_*sizeof(float)) +
            (state_.defined() ? 0 : std::size_t(nv_)*d_*d_*sizeof(float));
    }
    std::size_t request_cache_growth_bytes() const {
        return growth_bytes() + (fused_core_ && fused_decode_ && fused_preparation_ &&
            fused_output_ && transposed_state_ && !core_workspace_.defined()
            ? std::size_t(nv_)*(d_+1)*sizeof(float) : 0);
    }
    void restore_session(const Tensor& convolution,const Tensor& recurrent,bool retain_storage=false) {
        MFQ_RUNTIME_CHECK(convolution.defined() && recurrent.defined(),"missing Qwen4 GDN session state");
        if(retain_storage) {
            MFQ_RUNTIME_CHECK(conv_.defined() && state_.defined() && conv_.sizes()==convolution.sizes() &&
                state_.sizes()==recurrent.sizes(),"Qwen4 GDN graph session geometry changed");
            commit();conv_.copy_(convolution);state_.copy_(recurrent);
        } else {reset();conv_=convolution.clone();state_=recurrent.clone();}
    }
    Tensor recurrent_state() const {
        return state_.defined() && transposed_state_ ? state_.transpose(-1,-2) : state_;
    }
    std::vector<Tensor*> graph_state() { return {&conv_,&state_}; }
    std::vector<Tensor> indirect_state_tensors() const { return {conv_, state_, core_workspace_}; }
    std::vector<Tensor> project_inputs(CudaExecutionContext& execution,const Tensor& x) const {
        if(grouped_projection_ && w_.input_projection)return w_.input_projection(execution,x);
        return {w_.qkv(execution,x),w_.gate(execution,x),w_.alpha(execution,x),w_.beta(execution,x)};
    }
    Tensor project_output(CudaExecutionContext& execution,const Tensor& x) const {
        return w_.output(execution,x);
    }
    static Tensor prepare_independent_batch(const std::vector<Gdn*>& rows) {
        if(rows.size()<2 || rows.size()>32)return {};
        const auto inplace=mfq::cuda::runtime_options::gdn_inplace_state();
        if(inplace && !*inplace)return {};
        const auto& first=*rows.front();
        const auto eligible=[](const Gdn& row) {
            return row.fused_core_ && row.fused_decode_ && row.fused_preparation_ &&
                row.fused_output_ && row.transposed_state_ &&
                (row.d_==32 || row.d_==64 || row.d_==128) &&
                row.conv_.defined() && row.state_.defined() && row.core_workspace_.defined() &&
                row.conv_.is_cuda() && row.state_.is_cuda() && row.core_workspace_.is_cuda() &&
                row.conv_.is_contiguous() && row.state_.is_contiguous() && row.core_workspace_.is_contiguous() &&
                row.conv_.scalar_type()==tb::kFloat32 && row.state_.scalar_type()==tb::kFloat32 &&
                row.core_workspace_.scalar_type()==tb::kFloat32 &&
                row.conv_.sizes().vec()==std::vector<int64_t>({1,row.kernel_-1,(2*row.nk_+row.nv_)*row.d_}) &&
                row.state_.sizes().vec()==std::vector<int64_t>({1,row.nv_,row.d_,row.d_}) &&
                row.core_workspace_.sizes().vec()==std::vector<int64_t>({1,row.nv_,row.d_+1});
        };
        if(mfq::cuda::runtime_options::gdn_independent_batch_trace())
            std::fprintf(stderr,"gdn_batch_state eligible=%d flags=%d%d%d%d%d conv=%d state=%d counters=%d geometry=%lld/%lld/%lld/%lld\n",
                int(eligible(first)),int(first.fused_core_),int(first.fused_decode_),int(first.fused_preparation_),
                int(first.fused_output_),int(first.transposed_state_),int(first.conv_.defined()),int(first.state_.defined()),
                int(first.core_workspace_.defined()),(long long)first.nk_,(long long)first.nv_,(long long)first.d_,(long long)first.kernel_);
        if(!eligible(first))return {};
        for(const auto* row:rows)if(!eligible(*row) || row->nk_!=first.nk_ || row->nv_!=first.nv_ ||
            row->d_!=first.d_ || row->kernel_!=first.kernel_ || row->eps_!=first.eps_ ||
            row->silu_gate_!=first.silu_gate_ || row->state_.device()!=first.state_.device() ||
            row->conv_.device()!=first.state_.device() || row->core_workspace_.device()!=first.state_.device() ||
            row->w_.conv.data_ptr()!=first.w_.conv.data_ptr() || row->w_.norm.data_ptr()!=first.w_.norm.data_ptr() ||
            row->w_.dt_bias.data_ptr()!=first.w_.dt_bias.data_ptr() || row->w_.a_log.data_ptr()!=first.w_.a_log.data_ptr())return {};
        auto addresses=tb::empty({int64_t(rows.size()),3},first.state_.options().device(tb::kCPU).dtype(tb::kInt64));
        auto* values=addresses.data_ptr<int64_t>();
        for(std::size_t i=0;i<rows.size();++i) {
            values[3*i]=reinterpret_cast<int64_t>(rows[i]->conv_.data_ptr());
            values[3*i+1]=reinterpret_cast<int64_t>(rows[i]->state_.data_ptr());
            values[3*i+2]=reinterpret_cast<int64_t>(rows[i]->core_workspace_.data_ptr());
        }
        return addresses.to(first.state_.device());
    }
    Tensor forward_independent_batch(const Tensor& x,const std::vector<Tensor>& projections,
                                     const Tensor& addresses) {
        if(mfq::cuda::runtime_options::gdn_independent_batch_trace() && projections.size()==4)
            std::fprintf(stderr,"gdn_batch_projection input=%d qkv=%d gate=%d alpha=%d beta=%d rows=%lld\n",
                int(x.scalar_type()),int(projections[0].scalar_type()),int(projections[1].scalar_type()),
                int(projections[2].scalar_type()),int(projections[3].scalar_type()),(long long)x.size(0));
        if(!addresses.defined() || projections.size()!=4 || x.size(1)!=1 ||
            (x.scalar_type()!=tb::kFloat16 && x.scalar_type()!=tb::kFloat32) ||
            projections[0].scalar_type()!=tb::kFloat16 ||
            projections[1].scalar_type()!=projections[2].scalar_type() ||
            projections[2].scalar_type()!=projections[3].scalar_type() ||
            (projections[1].scalar_type()!=tb::kFloat16 && projections[1].scalar_type()!=tb::kFloat32))return {};
        if(mfq::cuda::runtime_options::gdn_independent_batch_trace())
            std::fprintf(stderr,"gdn_batch_core rows=%lld\n",(long long)x.size(0));
        auto weight=w_.conv.to(tb::kFloat32).contiguous();
        if(weight.dim()==2 && weight.size(0)==(2*nk_+nv_)*d_ && weight.size(1)==kernel_)
            weight=weight.reshape({(2*nk_+nv_)*d_,1,kernel_});
        const auto stride_setting=mfq::cuda::runtime_options::gdn_strided_projections();
        const auto direct=[&](const Tensor& value){return !stride_setting || *stride_setting?value:value.contiguous();};
        auto result=gdn_decode_core_cuda(direct(projections[0]),direct(projections[1]),
            projections[2].reshape({x.size(0),1,nv_}),projections[3].reshape({x.size(0),1,nv_}),
            conv_,state_,weight,w_.dt_bias.to(tb::kFloat32).contiguous(),w_.a_log.to(tb::kFloat32).contiguous(),
            w_.norm.to(tb::kFloat32).contiguous(),nk_,nv_,d_,1e-6,eps_,silu_gate_,
            x.scalar_type()==tb::kFloat16,true,core_workspace_,true,addresses);
        return result[0];
    }
    Tensor forward_graph(CudaExecutionContext& execution,const Tensor& x,int64_t confirmed,
                         std::vector<Tensor>& checkpoint_conv,std::vector<Tensor>& checkpoint_state) {
        if(!confirmed)return forward_chunk(execution,x,true);
        const auto batch=mfq::cuda::runtime_options::gdn_batch_projection();
        const bool grouped=grouped_projection_ && w_.input_projection && x.size(1)<=8 && (!batch || *batch);
        // Only recurrence is split at the confirmation boundary. Both parts
        // share the input projections and one output projection.
        std::vector<Tensor> projected;if(grouped)projected=w_.input_projection(execution,x);
        const auto* window_option=std::getenv("MFQ_MTP_GDN_REGISTER_WINDOW");
        const auto inplace_option=mfq::cuda::runtime_options::gdn_inplace_state();
        const auto* checkpoint_option=std::getenv("MFQ_MTP_GDN_FUSED_CHECKPOINT");
        if(grouped && confirmed==1 && x.size(0)==1 && x.size(1)>=2 && x.size(1)<=8 &&
            (!window_option || window_option[0]!='0') && nv_<=65535 &&
            (!inplace_option || *inplace_option) && (!checkpoint_option || checkpoint_option[0]!='0') &&
            d_==128 && kernel_==4 && fused_core_ && fused_decode_ && fused_preparation_ &&
            fused_output_ && transposed_state_ && conv_.defined() && state_.defined() &&
            projected.size()==4 && projected[0].scalar_type()==tb::kFloat16 &&
            projected[1].scalar_type()==projected[2].scalar_type() &&
            projected[2].scalar_type()==projected[3].scalar_type() &&
            (projected[1].scalar_type()==tb::kFloat16 || projected[1].scalar_type()==tb::kFloat32) &&
            (x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32)) {
            auto weight=w_.conv.to(tb::kFloat32).contiguous();
            const auto channels=(2*nk_+nv_)*d_;
            if(weight.dim()==2 && weight.size(0)==channels && weight.size(1)==kernel_)
                weight=weight.reshape({channels,1,kernel_});
            auto decoded=gdn_decode_window_cuda(projected[0],projected[1],projected[2].reshape({1,x.size(1),nv_}),
                projected[3].reshape({1,x.size(1),nv_}),conv_,state_,weight,w_.dt_bias.to(tb::kFloat32).contiguous(),
                w_.a_log.to(tb::kFloat32).contiguous(),w_.norm.to(tb::kFloat32).contiguous(),int(nk_),int(nv_),int(confirmed),
                1e-6,eps_,silu_gate_,x.scalar_type()==tb::kFloat16,true,checkpoint_conv,checkpoint_state);
            conv_.copy_(decoded[1]);
            return w_.output(execution,decoded[0]);
        }
        const auto slice=[&](int64_t begin,int64_t count) {
            std::vector<Tensor> result;for(const auto& value:projected)result.push_back(value.narrow(1,begin,count));
            return result;
        };
        // Project the whole verification window together, but preserve the
        // recurrent state after each prefix. Partial rejection then selects a
        // checkpoint instead of replaying accepted tokens through every layer.
        std::vector<Tensor> outputs;
        for(int64_t begin=0;begin<x.size(1);) {
            const auto count=begin==0?confirmed:1;
            const auto end=begin+count,index=end-confirmed;
            bool snapshot_written=false;
            outputs.push_back(forward_chunk(execution,x.narrow(1,begin,count),true,
                slice(begin,count),!grouped,
                end<x.size(1)?checkpoint_conv[index]:Tensor{},
                end<x.size(1)?checkpoint_state[index]:Tensor{},&snapshot_written));
            begin=end;
            if(begin<x.size(1) && !snapshot_written) {
                checkpoint_conv[index].copy_(conv_);checkpoint_state[index].copy_(state_);
            }
        }
        auto result=tb::cat(outputs,1);
        return grouped?w_.output(execution,result):result;
    }
    void bind_graph_checkpoint(const std::vector<Tensor>& conv,const std::vector<Tensor>& state,
                               int64_t position) {
        MFQ_RUNTIME_CHECK(!saved_conv_.defined(),"resolve previous GDN checkpoint");
        MFQ_RUNTIME_CHECK(!conv.empty() && conv.size()==state.size(),"invalid GDN prefix checkpoints");
        saved_conv_=conv.front();saved_state_=state.front();
        saved_prefix_conv_=conv;saved_prefix_state_=state;saved_prefix_position_=position;
    }
    void set_grouped_projection(bool enabled) {
        MFQ_RUNTIME_CHECK(!conv_.defined() && !state_.defined(), "reset GDN before switching projection execution");
        grouped_projection_ = enabled;
    }
    Tensor forward(CudaExecutionContext &execution, const Tensor &x, bool cache,
                   int64_t confirmed = 0, int64_t cache_position = -1) {
        MFQ_RUNTIME_CHECK(x.is_cuda() && x.dim() == 3 && x.size(0) > 0,
                          "recurrent attention requires nonempty [B,T,H] input");
        if(cache && cache_position>=0 && confirmed>0 && confirmed<x.size(1) && x.size(1)<=8) {
            MFQ_RUNTIME_CHECK(!saved_conv_.defined() && conv_.defined() && state_.defined(),
                              "GDN verification requires committed recurrent state");
            std::vector<Tensor> conv,state;
            for(int64_t i=confirmed;i<x.size(1);++i) {
                conv.push_back(tb::empty_like(conv_));state.push_back(tb::empty_like(state_));
            }
            auto result=forward_graph(execution,x,confirmed,conv,state);
            bind_graph_checkpoint(conv,state,cache_position+confirmed);
            return result;
        }
        const auto batch=mfq::cuda::runtime_options::gdn_batch_projection();
        const bool grouped=cache && confirmed>0 && confirmed<x.size(1) && x.size(1)<=8 &&
            grouped_projection_ && w_.input_projection && (!batch || *batch);
        std::vector<Tensor> projected;
        if(grouped)projected=w_.input_projection(execution,x);
        auto result=mfq::models::recurrent_window(
            x.size(1), cache, confirmed, saved_conv_.defined(),
            [&](int64_t start, int64_t count) {
                if(!grouped)return forward_chunk(execution,x.narrow(1,start,count),cache);
                // Share both projections across the verification window and
                // preserve the same per-prefix recurrent order as the graph.
                std::vector<Tensor> outputs;
                for(int64_t begin=start;begin<start+count;) {
                    const auto rows=begin==0?count:int64_t(1);
                    std::vector<Tensor> inputs;
                    for(const auto& value:projected)inputs.push_back(value.narrow(1,begin,rows));
                    outputs.push_back(forward_chunk(execution,x.narrow(1,begin,rows),cache,std::move(inputs),false));
                    begin+=rows;
                }
                return outputs.size()==1?std::move(outputs.front()):tb::cat(outputs,1);
            },
            [&] {
                saved_conv_ = conv_.clone();
                saved_state_ = state_.clone();
            },
            [&] { rollback(); },
            [](Tensor prefix, Tensor suffix) { return tb::cat({prefix, suffix}, 1); });
        return grouped?w_.output(execution,result):result;
    }
    Tensor forward_chunk(CudaExecutionContext &execution, const Tensor &x, bool cache,
                         std::vector<Tensor> projections = {},bool project_output=true,
                         Tensor checkpoint_conv={},Tensor checkpoint_state={},bool* snapshot_written=nullptr) {
        if(snapshot_written)*snapshot_written=false;
        const auto b = x.size(0), t = x.size(1), kw = nk_ * d_, vw = nv_ * d_,
                   channels = 2 * kw + vw;
        auto options = x.options().dtype(tb::kFloat32);
        MFQ_RUNTIME_CHECK(!cache || !conv_.defined() ||
                              (conv_.size(0) == b && conv_.device() == x.device()),
                          "reset Qwen4 GDN before changing batch/device");
        auto previous =
            cache && conv_.defined() ? conv_ : tb::zeros({b, kernel_ - 1, channels}, options);
        std::array<Tensor,2> prepared_gates;
        const auto prefill_option=mfq::cuda::runtime_options::gdn_prefill_fused();
        const bool prefill_fused=t>1 && fused_preparation_ && d_<=256 && kernel_<=8 &&
            (!prefill_option || *prefill_option!=0);
        if(cache && conv_.defined() && state_.defined() && fused_core_ && fused_decode_ &&
            fused_preparation_ && fused_output_ && transposed_state_ && t==1 &&
            (d_==32 || d_==64 || d_==128) &&
            (x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32)) {
            if(projections.empty()) {
                if(grouped_projection_ && w_.input_projection)projections=w_.input_projection(execution,x);
                else projections={w_.qkv(execution,x),w_.gate(execution,x),w_.alpha(execution,x),w_.beta(execution,x)};
            }
            MFQ_RUNTIME_CHECK(projections.size()==4,"GDN input projection count mismatch");
            if(projections[0].scalar_type()==tb::kFloat16 &&
                projections[1].scalar_type()==projections[2].scalar_type() &&
                projections[2].scalar_type()==projections[3].scalar_type() &&
                (projections[1].scalar_type()==tb::kFloat16 || projections[1].scalar_type()==tb::kFloat32)) {
                auto weight=w_.conv.to(tb::kFloat32).contiguous();
                if(weight.dim()==2 && weight.size(0)==channels && weight.size(1)==kernel_)
                    weight=weight.reshape({channels,1,kernel_});
                if(!core_workspace_.defined())core_workspace_=tb::zeros({b,nv_,d_+1},options);
                const auto stride_setting=mfq::cuda::runtime_options::gdn_strided_projections();
                const auto direct=[&](const Tensor& value){return !stride_setting || *stride_setting?value:value.contiguous();};
                const auto decode=[&](bool checkpoint) {
                  auto qkv=direct(projections[0]),gate=direct(projections[1]);
                  auto alpha=projections[2].reshape({b,t,nv_}),beta=projections[3].reshape({b,t,nv_});
                  if(checkpoint)return gdn_decode_core_checkpoint_cuda(qkv,gate,alpha,beta,previous,state_,
                    weight,w_.dt_bias.to(tb::kFloat32).contiguous(),w_.a_log.to(tb::kFloat32).contiguous(),
                    w_.norm.to(tb::kFloat32).contiguous(),nk_,nv_,d_,1e-6,eps_,silu_gate_,
                    x.scalar_type()==tb::kFloat16,true,core_workspace_,true,checkpoint_conv,checkpoint_state);
                  return gdn_decode_core_cuda(qkv,gate,
                    projections[2].reshape({b,t,nv_}),projections[3].reshape({b,t,nv_}),previous,state_,
                    weight,w_.dt_bias.to(tb::kFloat32).contiguous(),w_.a_log.to(tb::kFloat32).contiguous(),
                    w_.norm.to(tb::kFloat32).contiguous(),nk_,nv_,d_,1e-6,eps_,silu_gate_,
                    x.scalar_type()==tb::kFloat16,true,core_workspace_,
                    [] {const auto value=mfq::cuda::runtime_options::gdn_inplace_state();return !value || *value;}());
                };
                const auto* checkpoint_option=std::getenv("MFQ_MTP_GDN_FUSED_CHECKPOINT");
                const bool fused_snapshot=checkpoint_conv.defined() && checkpoint_state.defined() &&
                    (!checkpoint_option || checkpoint_option[0]!='0') &&
                    [] {const auto value=mfq::cuda::runtime_options::gdn_inplace_state();return !value || *value;}();
                auto decoded=decode(fused_snapshot);
                if(snapshot_written)*snapshot_written=fused_snapshot;
                if(fused_snapshot) {
                    static thread_local bool reported=false;
                    if(!reported) {
                        reported=true;
                        std::fprintf(stderr,"runtime_mtp_gdn_checkpoint fused=1 width=%lld heads=%lld\n",
                            static_cast<long long>(d_),static_cast<long long>(nv_));
                    }
                }
                auto output=project_output?w_.output(execution,decoded[0]):decoded[0];
                if(conv_.data_ptr()!=decoded[1].data_ptr())conv_.copy_(decoded[1]);
                if(state_.data_ptr()!=decoded[2].data_ptr())state_.copy_(decoded[2]);
                return output;
            }
        }
        return mfq::models::gated_delta_attention(
            cache,
            [&] {
                Tensor projected;
                if(!projections.empty())projected=projections[0];
                else if(grouped_projection_ && w_.input_projection) {
                    projections=w_.input_projection(execution,x);
                    MFQ_RUNTIME_CHECK(projections.size()==4,"GDN input projection count mismatch");
                    projected=projections[0];
                } else projected=w_.qkv(execution,x);
                MFQ_RUNTIME_CHECK(projected.sizes().vec() == std::vector<int64_t>({b, t, channels}),
                                  "Qwen4 GDN projection width mismatch");
                return projected;
            },
            [&](Tensor projected) {
                if(prefill_fused && projected.scalar_type()==tb::kFloat16) {
                    return execution.profiler.measure("prefill.gdn_conv_qkv",[&] {
                        auto weight=w_.conv.to(tb::kFloat32).contiguous();
                        if(weight.dim()==2 && weight.size(0)==channels && weight.size(1)==kernel_)
                            weight=weight.reshape({channels,1,kernel_});
                        const auto short_option=mfq::cuda::runtime_options::gdn_short_fusion();
                        if(t<=8 && (!short_option || *short_option) && projections.size()==4 &&
                            projections[2].scalar_type()==projections[3].scalar_type() &&
                            (projections[2].scalar_type()==tb::kFloat16 || projections[2].scalar_type()==tb::kFloat32)) {
                            const auto stride_setting=mfq::cuda::runtime_options::gdn_strided_projections();
                            const auto direct=[&](const Tensor& value){return !stride_setting || *stride_setting?value:value.contiguous();};
                            auto qkv=linear_conv_qkv_gate_short_cuda(previous,direct(projected),weight,
                                direct(projections[2].reshape({b,t,nv_})),direct(projections[3].reshape({b,t,nv_})),
                                w_.dt_bias.to(tb::kFloat32).contiguous(),w_.a_log.to(tb::kFloat32).contiguous(),nk_,nv_,d_,1e-6);
                            prepared_gates={qkv[4],qkv[5]};
                            return std::array<Tensor,4>{qkv[0],qkv[1],qkv[2],qkv[3]};
                        }
                        auto qkv=linear_conv_qkv_prefill_cuda(previous,
                            projected.narrow(-1,0,2*kw).contiguous(),
                            projected.narrow(-1,2*kw,vw).contiguous(),weight,
                            tb::empty({0},options),nk_,nv_,d_,d_,1e-6);
                        return std::array<Tensor,4>{qkv[0],qkv[1],qkv[2],qkv[3]};
                    });
                }
                if (fused_decode_ && t == 1 && d_ <= 256 && projected.scalar_type() == tb::kFloat16) {
                    auto weight = w_.conv.to(tb::kFloat32).contiguous();
                    if (weight.dim() == 2 && weight.size(0) == channels && weight.size(1) == kernel_)
                        weight = weight.reshape({channels, 1, kernel_});
                    auto next_conv = previous.clone();
                    auto qk=projected.narrow(-1,0,2*kw).contiguous(),v=projected.narrow(-1,2*kw,vw).contiguous();
                    std::vector<Tensor> qkv;
                    if(fused_preparation_) {
                        auto alpha=projections.empty()?w_.alpha(execution,x):projections[2];
                        auto beta=projections.empty()?w_.beta(execution,x):projections[3];
                        if(alpha.scalar_type()!=beta.scalar_type() ||
                            (alpha.scalar_type()!=tb::kFloat16 && alpha.scalar_type()!=tb::kFloat32)) {
                            alpha=alpha.to(tb::kFloat32);beta=beta.to(tb::kFloat32);
                        }
                        qkv=linear_conv_qkv_gate_decode_cuda(next_conv,qk,v,weight,
                            alpha.reshape({b,t,nv_}),beta.reshape({b,t,nv_}),
                            w_.dt_bias.to(tb::kFloat32).contiguous(),w_.a_log.to(tb::kFloat32).contiguous(),
                            nk_,nv_,d_,d_,1e-6);
                        prepared_gates={qkv[3],qkv[4]};
                    } else qkv=linear_conv_qkv_decode_cuda(next_conv,qk,v,weight,
                        tb::empty({0}, options),nk_,nv_,d_,d_,1e-6);
                    return std::array<Tensor, 4>{qkv[0], qkv[1], qkv[2], next_conv};
                }
                auto joined = tb::cat({previous, projected.to(tb::kFloat32)}, 1).contiguous();
                auto convolved = ssm_conv_silu_cuda(joined, w_.conv.to(tb::kFloat32).contiguous(),
                                                    tb::empty({0}, options), t);
                auto next_conv = joined.narrow(1, t, kernel_ - 1).contiguous();
                const auto normalize = [](const Tensor &v) {
                    return (v / tb::clamp_min((v * v).sum(-1, true).sqrt(), 1e-6)).contiguous();
                };
                auto q = normalize(
                    convolved.narrow(-1, 0, kw).reshape({b, t, nk_, d_}).permute({0, 2, 1, 3}));
                auto k = normalize(
                    convolved.narrow(-1, kw, kw).reshape({b, t, nk_, d_}).permute({0, 2, 1, 3}));
                auto v = convolved.narrow(-1, 2 * kw, vw)
                             .reshape({b, t, nv_, d_})
                             .permute({0, 2, 1, 3})
                             .contiguous();
                return std::array<Tensor, 4>{q, k, v, next_conv};
            },
            [&] {
                if(prepared_gates[0].defined())return prepared_gates;
                auto alpha=projections.empty()?w_.alpha(execution,x):projections[2];
                auto raw_beta=projections.empty()?w_.beta(execution,x):projections[3];
                if ((fused_decode_ && t == 1) || prefill_fused) {
                    auto gates = linear_gate_beta_cuda(
                        alpha.to(tb::kFloat32).reshape({b, t, nv_}),
                        raw_beta.to(tb::kFloat32).reshape({b, t, nv_}),
                        w_.dt_bias.to(tb::kFloat32).contiguous(),
                        w_.a_log.to(tb::kFloat32).contiguous(), true);
                    return std::array<Tensor, 2>{gates[0], gates[1]};
                }
                auto gate_input = alpha.to(tb::kFloat32).reshape({b, t, nv_}) +
                                  w_.dt_bias.to(tb::kFloat32).reshape({1, 1, nv_});
                auto softplus =
                    tb::clamp_min(gate_input, 0) + tb::log1p(tb::exp(-gate_input.abs()));
                auto decay = (-w_.a_log.to(tb::kFloat32).exp().reshape({1, 1, nv_}) * softplus)
                                 .transpose(1, 2)
                                 .contiguous();
                auto beta = tb::sigmoid(raw_beta.to(tb::kFloat32).reshape({b, t, nv_}))
                                .transpose(1, 2)
                                .contiguous();
                return std::array<Tensor, 2>{decay, beta};
            },
            [&](const auto &convolution, const auto &gates) {
                auto q = convolution[0], k = convolution[1], v = convolution[2];
                auto decay = gates[0], beta = gates[1];
                auto initial = cache && state_.defined() ? state_ : Tensor{};
                Tensor attended, next_state;
                if (d_ == 32 || d_ == 64 || d_ == 128) {
                    auto result = execution.profiler.measure("prefill.gdn_recurrence",[&] {
                        const auto short_option=mfq::cuda::runtime_options::gdn_short_inplace();
                        const auto inplace=mfq::cuda::runtime_options::gdn_inplace_state();
                        if(cache && initial.defined() && t<=8 && (!short_option || *short_option) && (!inplace || *inplace))
                            return (transposed_state_?gdn_inplace_transposed_cuda:gdn_inplace_cuda)(q,k,v,decay,beta,initial);
                        return (transposed_state_ ? gdn_transposed_cuda : gdn_cuda)(
                            q, k, v, decay, beta,initial.defined() ? MfqOptional<Tensor>(initial) : mfq_nullopt);
                    });
                    attended = result[0];
                    next_state = result[1];
                } else {
                    auto ids = tb::arange(nv_, options.dtype(tb::kInt64)) / (nv_ / nk_);
                    q = q.index_select(1, ids.to(tb::kInt64));
                    k = k.index_select(1, ids.to(tb::kInt64));
                    next_state = initial.defined() ? initial : tb::zeros({b, nv_, d_, d_}, options);
                    std::vector<Tensor> out;
                    for (int64_t i = 0; i < t; ++i) {
                        auto ki = k.select(2, i), rate = decay.select(2, i).exp().unsqueeze(-1);
                        auto projected_key = (next_state * ki.unsqueeze(-1)).sum(-2);
                        auto delta = (v.select(2, i) - rate * projected_key) *
                                     beta.select(2, i).unsqueeze(-1);
                        next_state = rate.unsqueeze(-1) * next_state +
                                     ki.unsqueeze(-1) * delta.unsqueeze(-2);
                        out.push_back(((next_state * q.select(2, i).unsqueeze(-1)).sum(-2) /
                                       std::sqrt(double(d_)))
                                          .unsqueeze(2));
                    }
                    attended = tb::cat(out, 2);
                }
                return std::array<Tensor, 2>{attended, next_state};
            },
            [&](Tensor attended) {
                const auto stride_setting=mfq::cuda::runtime_options::gdn_strided_projections();
                const bool strided=!stride_setting || *stride_setting;
                auto projected_gate=projections.empty()?w_.gate(execution,x):
                    strided?projections[1]:projections[1].contiguous();
                if(fused_decode_ && fused_output_ && attended.is_contiguous() &&
                    attended.scalar_type()==tb::kFloat32 && (projected_gate.is_contiguous() ||
                        (strided && projected_gate.dim()==3 && projected_gate.stride(2)==1)) &&
                    (projected_gate.scalar_type()==tb::kFloat16 || projected_gate.scalar_type()==tb::kFloat32) &&
                    (x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32))
                    return gdn_rms_norm_gate_cuda(attended,projected_gate,
                        w_.norm.to(tb::kFloat32).contiguous(),eps_,silu_gate_,x.scalar_type()==tb::kFloat16);
                projected_gate=projected_gate.contiguous();
                auto z = projected_gate.reshape({b, t, nv_, d_})
                             .permute({0, 2, 1, 3})
                             .to(tb::kFloat32);
                auto gate = tb::sigmoid(z);
                if (silu_gate_)
                    gate = z * gate;
                return rms_norm(attended, w_.norm, eps_) * gate;
            },
            [&](Tensor normalized) {
                if(normalized.dim()==3)return project_output?w_.output(execution,normalized):normalized;
                normalized=normalized.permute({0,2,1,3}).reshape({b,t,vw}).to(x.scalar_type());
                if(!project_output)return normalized;
                auto result = w_.output(
                    execution,normalized);
                return result;
            },
            [&](const auto &convolution, const Tensor &next) {
                if(conv_.defined())conv_.copy_(convolution[3]);
                else {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
                    ResidentAllocationScope resident(request_state_storage_);
#endif
                    conv_=request_state_storage_?convolution[3].clone():convolution[3];
                }
                if(state_.defined()) {if(state_.data_ptr()!=next.data_ptr())state_.copy_(next);}
                else {
#ifdef MFQ_NATIVE_CUDA_RUNTIME
                    ResidentAllocationScope resident(request_state_storage_);
#endif
                    state_=request_state_storage_?next.clone():next;
                }
            });
    }

  private:
    GdnWeights w_;
    int64_t nk_, nv_, d_, kernel_;
    double eps_;
    bool silu_gate_, fused_decode_,transposed_state_,fused_output_,grouped_projection_,fused_preparation_,fused_core_;
    bool request_state_storage_=false;
    Tensor conv_, state_, saved_conv_, saved_state_,core_workspace_;
    std::vector<Tensor> saved_prefix_conv_,saved_prefix_state_;
    int64_t saved_prefix_position_=-1;
};

// The upstream PLE hash is a CPU uint64 algorithm, including signed remainder
// after bitwise mixing and EOS segment boundaries. Keep shards independent.
class NgramEmbedding {
  public:
    using Context = mfq::models::qwen4_exp::NgramContext;
    using RowLookup = std::function<Tensor(const std::vector<int64_t>&,const tb::Device&)>;
    using StageFactory = std::function<std::shared_ptr<RowStage>(const std::vector<int64_t>&,const tb::Device&)>;
    NgramEmbedding(std::vector<Embedding> shards, int64_t rows, int64_t dimension, int64_t ngram,
                   int64_t heads_per_ngram, int64_t eos, std::vector<int64_t> multipliers,
                   std::vector<int64_t> offsets, std::vector<int64_t> vocab,std::shared_ptr<NintRowPipeline> pipeline = {},
                   RowLookup row_lookup = {},StageFactory stage_factory = {})
        : shards_(std::move(shards)), rows_(rows), dimension_(dimension), ngram_(ngram),
          heads_(heads_per_ngram), eos_(eos), multipliers_(std::move(multipliers)),
          offsets_(std::move(offsets)), vocab_(std::move(vocab)),pipeline_(std::move(pipeline)),row_lookup_(std::move(row_lookup)),stage_factory_(std::move(stage_factory)) {
        MFQ_RUNTIME_CHECK(!shards_.empty() && rows_ > 0 && dimension_ > 0 && ngram_ > 1 &&
                              heads_ > 0 && multipliers_.size() == size_t(ngram_) &&
                              offsets_.size() == size_t((ngram_ - 1) * heads_) &&
                              vocab_.size() == offsets_.size(),
                          "invalid Qwen4 ngram metadata geometry");
        for (size_t h = 0; h < vocab_.size(); ++h)
            MFQ_RUNTIME_CHECK(vocab_[h] > 0 && offsets_[h] >= 0 &&
                                  offsets_[h] <= int64_t(shards_.size()) * rows_ - vocab_[h],
                              "Qwen4 ngram vocabulary lies outside embedding shards");
    }
    void reset() { context_ = {};hash_ready_=false; }
    bool graph_supported() const { return bool(pipeline_) || bool(stage_factory_); }
    const Context &context() const { return context_; }
    void restore(Context context) { context_ = std::move(context);hash_ready_=false; }
    Context prefix_context(const Tensor& host_ids,int64_t count) const {
        auto host=host_ids.narrow(1,0,count).to(tb::kInt64).contiguous();
        return mfq::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),host.size(0),count,
            ngram_,heads_,eos_,multipliers_,offsets_,vocab_,context_,true).next;
    }
    void prefetch(const Tensor& host_ids,bool early_prefix=false) {
        if(!pipeline_)return;
        MFQ_RUNTIME_CHECK(host_ids.is_cpu() && host_ids.dim()==2,"PLE prefetch requires host token IDs");
        auto host=host_ids.to(tb::kInt64).contiguous();const auto b=host.size(0),t=host.size(1);
        if(cacheable_hashes(b,t)) {
            const auto& hashes=graph_hashes(host);
            pipeline_->issue(hashes.ids,{b,t,(ngram_-1)*heads_},early_prefix);
            return;
        }
        auto hashes=mfq::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),b,t,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,context_,true);
        pipeline_->issue(hashes.ids,{b,t,(ngram_-1)*heads_},early_prefix);
    }
    std::shared_ptr<RowStage> graph_stage(int64_t b,int64_t t,const tb::Device& device) {
        const std::vector<int64_t> shape{b,t,(ngram_-1)*heads_};
        if(stage_factory_)return stage_factory_(shape,device);
        MFQ_RUNTIME_CHECK(pipeline_,"PLE graph requires prepared range rows");
        return std::make_shared<NintStage>(pipeline_,shape,device);
    }
    void upload_graph(const Tensor& host_ids,RowStage& stage) {
        MFQ_RUNTIME_CHECK(graph_supported() && host_ids.is_cpu() && host_ids.dim()==2,"PLE graph requires host token IDs");
        auto host=host_ids.to(tb::kInt64).contiguous();const auto b=host.size(0),t=host.size(1);
        if(cacheable_hashes(b,t)) {
            auto& hashes=graph_hashes(host);
            stage.upload(hashes.ids,{b,t,(ngram_-1)*heads_});
            restore(std::move(hashes.next));
            return;
        }
        auto hashes=mfq::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),b,t,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,context_,true);
        stage.upload(hashes.ids,{b,t,(ngram_-1)*heads_});
        restore(std::move(hashes.next));
    }
    Tensor forward(const Tensor &ids, bool cache, Tensor *hashed_ids = nullptr,const Tensor& host_ids = {}) {
        MFQ_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0,
                          "Qwen4 ngram IDs must be [B,T]");
        const auto b = ids.size(0), t = ids.size(1), nh = (ngram_ - 1) * heads_;
        auto host = host_ids.defined() ? host_ids.to(tb::kInt64).contiguous() : ids.to(tb::kInt64).contiguous().cpu();
        MFQ_RUNTIME_CHECK(host.is_cpu() && host.sizes()==ids.sizes(),"PLE host token mirror differs");
        const auto *source = host.data_ptr<int64_t>();
        auto hashes = mfq::models::qwen4_exp::ngram_hashes(
            source, b, t, ngram_, heads_, eos_, multipliers_, offsets_, vocab_, context_, cache);
        auto &global = hashes.ids;
        if(row_lookup_) {
            auto result=row_lookup_(global,ids.device());
            if(hashed_ids)*hashed_ids=tb::tensor(global).reshape({b,t,nh}).to(ids.device());
            if(cache)restore(std::move(hashes.next));
            return result.reshape({b,t,nh*dimension_});
        }
        if(pipeline_) {
            auto result=pipeline_->collect(global,{b,t,nh},ids.device());
            if(hashed_ids)*hashed_ids=tb::tensor(global).reshape({b,t,nh}).to(ids.device());
            if(cache)restore(std::move(hashes.next));
            return result.reshape({b,t,nh*dimension_});
        }
        auto global_tensor =
            tb::from_blob(global.data(), {b, t, nh}, tb::TensorOptions().dtype(tb::kInt64))
                .clone()
                .to(ids.device());
        if (hashed_ids)
            *hashed_ids = global_tensor;
        std::set<int64_t> used;
        for (auto value : global)
            used.insert(value / rows_);
        Tensor result;
        for (auto shard : used) {
            auto mask = (global_tensor >= shard * rows_) & (global_tensor < (shard + 1) * rows_);
            auto local =
                tb::where(mask, global_tensor - shard * rows_, tb::zeros_like(global_tensor));
            auto values = shards_.at(shard)(local);
            MFQ_RUNTIME_CHECK(values.sizes().vec() == std::vector<int64_t>({b, t, nh, dimension_}),
                              "Qwen4 ngram embedding shape mismatch");
            values = values * mask.unsqueeze(-1).to(values.scalar_type());
            result = result.defined() ? result + values : values;
        }
        if (cache)
            restore(std::move(hashes.next));
        return result.reshape({b, t, nh * dimension_});
    }

  private:
    struct NintStage final:RowStage {
        std::shared_ptr<NintRowPipeline> pipeline;
        std::shared_ptr<NintRowStage> stage;
        NintStage(std::shared_ptr<NintRowPipeline> p,const std::vector<int64_t>& shape,const tb::Device& device)
            :pipeline(std::move(p)),stage(pipeline->make_stage(shape,device)) {}
        Tensor output()const override{return stage->output();}
        bool mapped_rows()const override{return stage->mapped_rows();}
        void decode()override{stage->decode();}
        void upload(const std::vector<int64_t>& ids,const std::vector<int64_t>& shape)override {
            pipeline->upload(*stage,ids,shape);
        }
    };
    std::vector<Embedding> shards_;
    int64_t rows_, dimension_, ngram_, heads_, eos_;
    std::vector<int64_t> multipliers_, offsets_, vocab_;
    Context context_;
    std::shared_ptr<NintRowPipeline> pipeline_;
    RowLookup row_lookup_;
    StageFactory stage_factory_;
    bool reuse_hashes_=[] {
        const auto* option=std::getenv("MFQ_PLE_HASH_REUSE");return !option || option[0]!='0';
    }();
    bool hash_ready_=false,hash_reported_=false;
    int cached_tokens_=0;
    std::array<int64_t,8> cached_ids_{};
    mfq::models::qwen4_exp::NgramHashes cached_hashes_;
    bool cacheable_hashes(int64_t b,int64_t t) const {return reuse_hashes_ && b==1 && t>0 && t<=8;}
    mfq::models::qwen4_exp::NgramHashes& graph_hashes(const Tensor& host) {
        const auto tokens=host.size(1);const auto* ids=host.data_ptr<int64_t>();
        if(hash_ready_ && cached_tokens_==tokens && std::equal(ids,ids+tokens,cached_ids_.begin())) {
            if(!hash_reported_) {
                hash_reported_=true;
                std::fprintf(stderr,"ple_ngram_hash_reuse enabled=1 tokens=%lld\n",static_cast<long long>(tokens));
            }
            return cached_hashes_;
        }
        cached_hashes_=mfq::models::qwen4_exp::ngram_hashes(ids,1,tokens,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,context_,true);
        std::copy_n(ids,tokens,cached_ids_.begin());cached_tokens_=int(tokens);hash_ready_=true;
        return cached_hashes_;
    }
};

struct PleWeights {
    Linear key, value;
    Tensor key_norm, query_norm, conv_norm, conv;
};
class Ple {
  public:
    Ple(NgramEmbedding embedding, PleWeights weights, int64_t hidden, int64_t streams,
        int64_t dilation, double eps)
        : embedding_(std::move(embedding)), w_(std::move(weights)), hidden_(hidden),
          streams_(streams), dilation_(dilation), eps_(eps) {}
    void reset(bool retain_storage=false) {
        embedding_.reset();
        if(retain_storage && conv_.defined())conv_.zero_();else conv_ = Tensor();
        commit();
    }
    void commit() {
        saved_ = false;
        saved_conv_ = Tensor();
        saved_context_ = {};
        saved_prefix_conv_.clear();saved_prefix_context_.clear();saved_prefix_position_=-1;
    }
    void rollback(int64_t keep=-1) {
        MFQ_RUNTIME_CHECK(saved_, "Qwen4 PLE has no speculative checkpoint");
        if(keep>=0 && saved_prefix_position_>=0) {
            const auto index=keep-saved_prefix_position_;
            MFQ_RUNTIME_CHECK(index>=0 && index<int64_t(saved_prefix_conv_.size()),
                "Qwen4 PLE prefix checkpoint is out of range");
            saved_conv_=saved_prefix_conv_[index];saved_context_=saved_prefix_context_[index];
        }
        if(conv_.defined())conv_.copy_(saved_conv_);else conv_=saved_conv_;
        embedding_.restore(saved_context_);
        commit();
    }
    const Tensor &conv_state() const { return conv_; }
    std::vector<Tensor*> graph_state() { return {&conv_}; }
    bool graph_supported() const { return embedding_.graph_supported(); }
    const NgramEmbedding::Context& session_context() const { return embedding_.context(); }
    void restore_graph_context(NgramEmbedding::Context context) { embedding_.restore(std::move(context)); }
    void restore_session(const Tensor& convolution,NgramEmbedding::Context context,bool retain_storage=false) {
        MFQ_RUNTIME_CHECK(convolution.defined(),"missing Qwen4 PLE session state");
        if(retain_storage) {
            MFQ_RUNTIME_CHECK(conv_.defined() && conv_.sizes()==convolution.sizes(),
                "Qwen4 PLE graph session geometry changed");
            commit();conv_.copy_(convolution);
        } else {reset();conv_=convolution.clone();}
        embedding_.restore(std::move(context));
    }
    std::shared_ptr<RowStage> graph_stage(int64_t tokens,const tb::Device& device) {
        return embedding_.graph_stage(1,tokens,device);
    }
    void upload_graph(const Tensor& host_ids,RowStage& stage) {embedding_.upload_graph(host_ids,stage);}
    NgramEmbedding::Context graph_prefix_context(const Tensor& host_ids,int64_t count) const {
        return embedding_.prefix_context(host_ids,count);
    }
    void bind_graph_checkpoint(const std::vector<Tensor>& conv,
                               std::vector<NgramEmbedding::Context> context,int64_t position) {
        MFQ_RUNTIME_CHECK(!saved_,"resolve previous PLE checkpoint");
        MFQ_RUNTIME_CHECK(!conv.empty() && conv.size()==context.size(),"invalid PLE prefix checkpoints");
        saved_=true;saved_conv_=conv.front();saved_context_=context.front();
        saved_prefix_conv_=conv;saved_prefix_context_=std::move(context);saved_prefix_position_=position;
    }
    Tensor forward_graph(CudaExecutionContext& execution,const Tensor& x,const Tensor& ids,
                         const Tensor& rows,int64_t confirmed,std::vector<Tensor>& checkpoint_conv) {
        if(!confirmed)return forward_chunk(execution,x,ids,true,{},rows);
        auto prefix=forward_chunk(execution,x.narrow(1,0,confirmed),ids.narrow(1,0,confirmed),
            true,{},rows.narrow(1,0,confirmed));
        checkpoint_conv.front().copy_(conv_);
        auto suffix=forward_chunk(execution,x.narrow(1,confirmed,x.size(1)-confirmed),
            ids.narrow(1,confirmed,x.size(1)-confirmed),true,{},
            rows.narrow(1,confirmed,x.size(1)-confirmed),&checkpoint_conv,1);
        return tb::cat({prefix,suffix},1);
    }
    void prefetch_tokens(const Tensor& ids,bool early_prefix=false) { embedding_.prefetch(ids,early_prefix); }
    Tensor forward(CudaExecutionContext &execution, const Tensor &x, const Tensor &ids, bool cache,
                   int64_t confirmed = 0,const Tensor& host_ids = {},int64_t cache_position = -1) {
        MFQ_RUNTIME_CHECK(x.dim() == 3 && ids.dim() == 2 && x.size(0) == ids.size(0) &&
                              x.size(1) == ids.size(1) && x.size(2) == hidden_ * streams_ &&
                              confirmed >= 0 && confirmed <= x.size(1) && (!confirmed || cache),
                          "invalid Qwen4 PLE input");
        if(cache && cache_position>=0 && confirmed>0 && confirmed<x.size(1) && x.size(1)<=8) {
            MFQ_RUNTIME_CHECK(!saved_ && conv_.defined(),"PLE verification requires committed convolution state");
            auto host=host_ids.defined()?host_ids:ids.to(tb::kCPU,tb::kInt64);
            std::vector<Tensor> conv;
            std::vector<NgramEmbedding::Context> context;
            for(int64_t i=confirmed;i<x.size(1);++i) {
                conv.push_back(tb::empty_like(conv_));context.push_back(embedding_.prefix_context(host,i));
            }
            auto rows=embedding_.forward(ids,true,nullptr,host);
            auto result=forward_graph(execution,x,ids,rows,confirmed,conv);
            bind_graph_checkpoint(conv,std::move(context),cache_position+confirmed);
            return result;
        }
        return mfq::models::recurrent_window(
            x.size(1), cache, confirmed, saved_,
            [&](int64_t start, int64_t count) {
                return forward_chunk(execution, x.narrow(1, start, count),
                                     ids.narrow(1, start, count), cache,
                                     host_ids.defined() ? host_ids.narrow(1,start,count) : Tensor{});
            },
            [&] {
                saved_ = true;
                saved_conv_ = conv_.clone();
                saved_context_ = embedding_.context();
            },
            [&] { rollback(); },
            [](Tensor first, Tensor second) { return tb::cat({first, second}, 1); });
    }
    Tensor forward_chunk(CudaExecutionContext &execution, const Tensor &x, const Tensor &ids,
                         bool cache,const Tensor& host_ids = {},const Tensor& graph_embeddings = {},
                         std::vector<Tensor>* checkpoints=nullptr,std::size_t checkpoint_offset=0) {
        auto context = embedding_.context();
        const auto b = x.size(0), t = x.size(1);
        const auto save_prefixes=[&](const Tensor& normalized) {
            if(!checkpoints)return;
            MFQ_RUNTIME_CHECK(cache && conv_.defined(),"PLE prefix history is unavailable");
            const auto lookback=conv_.size(1);
            // The FIR state is a tail of normalized inputs. Recover every
            // intermediate tail while keeping key/value projections batched.
            for(int64_t count=1;count<t;++count) {
                auto& saved=checkpoints->at(checkpoint_offset+std::size_t(count-1));
                if(count>=lookback)saved.copy_(normalized.narrow(1,count-lookback,lookback));
                else {
                    saved.narrow(1,0,lookback-count).copy_(conv_.narrow(1,count,lookback-count));
                    saved.narrow(1,lookback-count,count).copy_(normalized.narrow(1,0,count));
                }
            }
        };
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        const auto native=mfq::cuda::runtime_options::ple_native_fusion();
        if((!native || *native) && x.is_contiguous() && t<=32 && hidden_>128 && hidden_<=8192 &&
            (x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32)) {
            try {
                auto embeddings=graph_embeddings.defined()?graph_embeddings:
                    embedding_.forward(ids,cache,nullptr,host_ids);
                auto key=mfq_qwen4_exp::grouped_rms_norm(w_.key(execution,embeddings),w_.key_norm,hidden_,eps_);
                auto query=mfq_qwen4_exp::grouped_rms_norm(x,w_.query_norm,hidden_,eps_);
                auto value=w_.value(execution,embeddings);
                auto prepared=mfq_qwen4_exp::ple_gate_norm_cuda(key,query,value,
                    w_.conv_norm.to(tb::kFloat32).contiguous(),hidden_,streams_,eps_);
                save_prefixes(prepared[1]);
                auto output=mfq_qwen4_exp::ple_conv_add_cuda(prepared[1],w_.conv,
                    cache && conv_.defined()?std::optional<Tensor>(conv_):std::nullopt,dilation_,prepared[0],
                    x.scalar_type()==tb::kFloat16);
                if(cache) {if(conv_.defined())conv_.copy_(output[1]);else conv_=output[1];}
                return output[0];
            }catch(...) {embedding_.restore(std::move(context));throw;}
        }
#endif
        return mfq::models::qwen4_exp::position_embedding(
            cache, [&] { return graph_embeddings.defined() ? graph_embeddings : embedding_.forward(ids, cache,nullptr,host_ids); },
            [&](const Tensor &embeddings) {
                auto key = mfq_qwen4_exp::grouped_rms_norm(w_.key(execution, embeddings),
                                                           w_.key_norm, hidden_, eps_)
                               .reshape({b, t, streams_, hidden_});
                return key;
            },
            [&] {
                auto query = mfq_qwen4_exp::grouped_rms_norm(x, w_.query_norm, hidden_, eps_)
                                 .reshape({b, t, streams_, hidden_});
                return query;
            },
            [&](const Tensor &key, const Tensor &query, const Tensor &embeddings) {
                auto score = (key.to(tb::kFloat32) * query.to(tb::kFloat32)).sum(-1) /
                             std::sqrt(double(hidden_));
                auto sign = (score > 0).to(tb::kFloat32) - (score < 0).to(tb::kFloat32);
                auto root = sign * tb::clamp_min(score.abs(), 1e-6).sqrt();
                auto gated = (tb::sigmoid(root).unsqueeze(-1) *
                              w_.value(execution, embeddings).to(tb::kFloat32).unsqueeze(-2))
                                 .reshape({b, t, streams_ * hidden_});
                return gated;
            },
            [&](const Tensor &gated) {
                return mfq_qwen4_exp::grouped_rms_norm(gated, w_.conv_norm, hidden_, eps_);
            },
            [&](Tensor normalized) {
                save_prefixes(normalized);
                return mfq_qwen4_exp::ple_dilated_conv_silu(
                    normalized, w_.conv,
                    cache && conv_.defined() ? std::optional<Tensor>(conv_) : std::nullopt,
                    dilation_);
            },
            [&](Tensor gated, Tensor convolved) { return (gated + convolved).to(x.scalar_type()); },
            [&](Tensor next) { if(conv_.defined())conv_.copy_(next);else conv_=std::move(next); },
            [&] { embedding_.restore(std::move(context)); });
    }

  private:
    NgramEmbedding embedding_;
    PleWeights w_;
    int64_t hidden_, streams_, dilation_;
    double eps_;
    Tensor conv_, saved_conv_;
    NgramEmbedding::Context saved_context_;
    std::vector<Tensor> saved_prefix_conv_;
    std::vector<NgramEmbedding::Context> saved_prefix_context_;
    int64_t saved_prefix_position_=-1;
    bool saved_ = false;
};

struct QsaWeights {
    Linear query, key, value, output, index_query_key;
    Tensor query_norm, key_norm, index_query_norm, index_key_norm;
    LinearGroup input_projection;
    Linear gate;
};
struct QsaConfig {
    int64_t heads, kv_heads, width, index_heads, index_width, pool, budget, maximum;
    double eps;
};

class Qsa {
  public:
    Qsa(QsaWeights weights, QsaConfig config, std::shared_ptr<RotaryEmbedding> rotary,
        bool fused_projection = true, bool grouped_projection = true,
        std::shared_ptr<KvOffloadStore> offload = {}, std::shared_ptr<KvMemoryManager> memory = {})
        : w_(std::move(weights)), c_(config), rotary_(std::move(rotary)), fused_projection_(fused_projection), grouped_projection_(grouped_projection),
          keys_(config.maximum, config.kv_heads * key_width(), cache_dtype()),
          values_(config.maximum, config.kv_heads * value_width(), cache_dtype()),
          index_(config.maximum, config.index_width) {
        MFQ_RUNTIME_CHECK(c_.heads > 0 && c_.kv_heads > 0 && c_.heads % c_.kv_heads == 0 &&
                              c_.width > 0 && c_.index_heads > 0 && c_.pool > 0 && c_.budget > 0 &&
                              c_.budget % c_.pool == 0 && rotary_,
                          "invalid Qwen4 QSA configuration");
        // Share immutable FP32 norm coefficients across request states.
        // BF16 sources otherwise record repeated casts in each row's graph.
        for(auto* weight:{&w_.query_norm,&w_.key_norm,&w_.index_query_norm,&w_.index_key_norm})
            if(weight->defined())*weight=weight->to(tb::kFloat32).contiguous();
        memory_manager_=memory;
        offload_ = std::move(offload);
        if(memory && !offload_)memory_=memory->add(c_.maximum,c_.kv_heads,key_width(),value_width(),
            c_.index_width,c_.pool,w_.query_norm.options().dtype(cache_dtype()),c_.maximum<=c_.budget?c_.maximum:c_.budget+c_.pool-1);
        if (offload_) {
            offloaded_kv_ = std::make_unique<KvOffloadSequence>(offload_, c_.kv_heads * (key_width()+value_width()), c_.pool, cache_dtype());
            offloaded_index_ = std::make_unique<KvOffloadSequence>(offload_, c_.index_width, c_.pool);
            offloaded_pooled_ = std::make_unique<KvOffloadSequence>(offload_, c_.index_width, 16);
        }
    }
    int64_t position() const { return offload_ && offloaded_kv_ ? offloaded_kv_->position() : keys_.position(); }
    bool managed_memory() const {return memory_ && memory_->ready;}
    bool has_kv_offload() const {return bool(offload_) || (memory_ && memory_->offload);}
    int64_t graph_capacity() const {return managed_memory()?c_.maximum:0;}
    void activate_memory() {
        if(!managed_memory())return;
        if(!index_.storage().defined())index_.replace_storage(memory_->index);
        pooled_.bind(memory_->pooled);
        graph_pooled_=pooled_.storage();
        if(memory_->offload) {
            offload_=memory_->offload;
            if(!offloaded_kv_)offloaded_kv_=std::make_unique<KvOffloadSequence>(offload_,
                c_.kv_heads*(key_width()+value_width()),c_.pool,cache_dtype());
        } else {graph_keys_=memory_->keys;graph_values_=memory_->values;}
    }
    std::unique_ptr<Qsa> fork_empty() const {
        return std::make_unique<Qsa>(w_, c_, rotary_, fused_projection_, grouped_projection_, offload_,memory_manager_);
    }
    std::vector<Tensor> project_inputs(CudaExecutionContext& execution,const Tensor& x) const {
        if(grouped_projection_ && w_.input_projection)return w_.input_projection(execution,x);
        return {w_.query(execution,x),w_.key(execution,x),w_.value(execution,x),w_.index_query_key(execution,x)};
    }
    Tensor project_output(CudaExecutionContext& execution,const Tensor& x) const {
        return w_.output(execution,x);
    }
    std::size_t growth_bytes(int64_t tokens) const {
        if(managed_memory())return memory_->growth_bytes(position()+tokens);
        if (offload_) return 0;
        const auto required = position() + tokens;
        MFQ_RUNTIME_CHECK(tokens > 0 && required <= c_.maximum, "QSA growth exceeds max_context");
        std::size_t bytes = 0;
        for (const auto* cache : {&keys_, &values_, &index_}) {
            const auto& storage = cache->storage();
            const int64_t current = graph_keys_.defined() && cache != &index_
                ? graph_keys_.size(2) : (storage.defined() ? storage.size(1) : 0);
            if (required <= current) continue;
            int64_t capacity = current ? current : std::min<int64_t>(16, c_.maximum);
            while (capacity < required)
                capacity = std::min(c_.maximum, capacity > c_.maximum / 2 ? c_.maximum : capacity * 2);
            const auto width = cache == &index_ ? c_.index_width : c_.kv_heads * (cache == &keys_ ? key_width() : value_width());
            bytes += std::size_t(capacity) * width * (cache != &index_ && quantization_.enabled() ? 4 : 2);
        }
        if(required>c_.budget)bytes+=pooled_.growth_bytes(required,c_.pool,c_.index_width,2);
        return bytes;
    }
    std::size_t graph_growth_bytes(int64_t capacity) const {
        if(managed_memory())return memory_->growth_bytes(capacity)+
            (graph_prepared_?0:pooled_.position_bytes(c_.maximum));
        if (graph_keys_.defined() && graph_keys_.size(2) >= capacity) return 0;
        auto bytes = std::size_t(capacity) * c_.kv_heads * (key_width()+value_width()) * (quantization_.enabled()?4:2);
        if (!index_.storage().defined() || index_.storage().size(1) != capacity)
            bytes += std::size_t(capacity) * c_.index_width * 2;
        bytes += std::size_t((capacity+c_.pool-1)/c_.pool) * c_.index_width * 2;
        bytes += pooled_.position_bytes(capacity);
        return bytes;
    }
    bool supports_session() const { return !offload_ && !(memory_ && memory_->offload); }
    std::size_t session_bytes(int64_t count) const {
        if(count<=0)return 0;
        return std::size_t(count)*(c_.kv_heads*(key_width()+value_width())*(quantization_.enabled()?4:2)+
            c_.index_width*index_.storage().element_size())+
            std::size_t(std::min(pooled_.completed(),count/c_.pool))*c_.index_width*index_.storage().element_size();
    }
    void capture_session(Qwen4BlockSessionState& state,int64_t count,std::size_t& bytes) const {
        MFQ_RUNTIME_CHECK(!offload_ && count>0 && count<=position(),"invalid Qwen4 QSA session boundary");
        const auto save=[&](const Tensor& tensor) {
            auto saved=tensor.contiguous().clone();bytes+=session_tensor_bytes(saved);return saved;
        };
        if(managed_memory() && memory_->hot.exclusive()) {
            auto pair=memory_->hot.gather(count);
            state.keys=save(pair.first.permute({0,2,1,3}).reshape({1,count,c_.kv_heads*key_width()}));
            state.values=save(pair.second.permute({0,2,1,3}).reshape({1,count,c_.kv_heads*value_width()}));
        } else if(graph_keys_.defined()) {
            state.keys=save(graph_keys_.narrow(2,0,count).permute({0,2,1,3}).reshape({1,count,c_.kv_heads*key_width()}));
            state.values=save(graph_values_.narrow(2,0,count).permute({0,2,1,3}).reshape({1,count,c_.kv_heads*value_width()}));
        } else {
            state.keys=save(keys_.storage().narrow(1,0,count));
            state.values=save(values_.storage().narrow(1,0,count));
        }
        state.index=save(index_.storage().narrow(1,0,count));
        state.pooled=pooled_.capture(count,c_.pool);
        bytes+=session_tensor_bytes(state.pooled.values);
    }
    void restore_session(const Qwen4BlockSessionState& state,int64_t count) {
        MFQ_RUNTIME_CHECK(!offload_ && state.keys.defined() && state.values.defined() && state.index.defined() &&
            state.keys.dim()==3 && state.keys.size(0)==1 && state.keys.size(1)==count &&
            state.keys.size(2)==c_.kv_heads*key_width() && state.keys.scalar_type()==cache_dtype() &&
            state.values.dim()==3 && state.values.size(0)==1 && state.values.size(1)==count &&
            state.values.size(2)==c_.kv_heads*value_width() && state.values.scalar_type()==cache_dtype() &&
            state.index.dim()==3 && state.index.size(0)==1 && state.index.size(1)==count && state.index.size(2)==c_.index_width,
            "incompatible Qwen4 QSA session state");
        reset();activate_memory();
        if(managed_memory()) {
            memory_->ensure(count);
            auto k=state.keys.reshape({1,count,c_.kv_heads,key_width()}).permute({0,2,1,3});
            auto v=state.values.reshape({1,count,c_.kv_heads,value_width()}).permute({0,2,1,3});
            auto positions=tb::arange(count,graph_keys_.options().dtype(tb::kInt64));
            if(memory_->hot.exclusive())memory_->hot.append(k,v,positions);
            else {
                graph_keys_.narrow(2,0,count).copy_(k);graph_values_.narrow(2,0,count).copy_(v);
                memory_->hot.publish(graph_keys_,graph_values_,positions);
            }
            keys_.advance_fixed(count);values_.advance_fixed(count);index_.append(state.index);graph_prepared_=false;
        } else {keys_.append(state.keys);values_.append(state.values);index_.append(state.index);}
        pooled_.restore(state.pooled,count,c_.pool);
    }
    std::vector<Tensor> resident_cache_tensors() const {
        if(managed_memory())return {pooled_.positions()}; // KV tiers are counted through KvMemoryManager.
        return {keys_.storage(),values_.storage(),index_.storage(),graph_keys_,graph_values_,
            pooled_.storage(),pooled_.positions()};
    }
    void set_grouped_projection(bool enabled) {
        MFQ_RUNTIME_CHECK(!graph_keys_.defined() && position()==0,"reset QSA before switching projection execution");
        grouped_projection_=enabled;
    }
    void set_rotary_fused(bool enabled) {
        MFQ_RUNTIME_CHECK(!graph_keys_.defined() && position() == 0,
                          "reset QSA before switching rotary execution");
        rotary_->set_fused(enabled);
    }
    void leave_graph() {
        if(managed_memory()){graph_prepared_=false;return;}
        if(!graph_keys_.defined())return;
        const auto n=position();
        keys_.replace_storage(graph_keys_.narrow(2,0,n).permute({0,2,1,3}).contiguous().reshape({1,n,c_.kv_heads*key_width()}));
        values_.replace_storage(graph_values_.narrow(2,0,n).permute({0,2,1,3}).contiguous().reshape({1,n,c_.kv_heads*value_width()}));
        graph_keys_={};graph_values_={};graph_pooled_={};
    }
    int64_t pooled_groups() const { return pooled_.completed(); }
    int64_t pooled_processed_groups() const { return pooled_.processed_groups(); }
    const Tensor& pooled_cache() const { return pooled_.storage(); }
    void set_incremental_pooling(bool enabled) { pooled_.set_incremental(enabled); }
    void reset() {
        if (offload_) {
            if(offloaded_kv_)offloaded_kv_->reset();
            if(offloaded_index_)offloaded_index_->reset();
            if(offloaded_pooled_)offloaded_pooled_->reset();
        }
        keys_.reset();
        values_.reset();
        index_.reset();
        pooled_.reset();
        graph_prepared_=false;
        if(managed_memory()){memory_->hot.reset();activate_memory();return;}
        graph_keys_={};graph_values_={};graph_pooled_={};
    }
    void prepare_graph(const Tensor& full_positions, int64_t capacity) {
        activate_memory();
        MFQ_RUNTIME_CHECK(!offload_ || managed_memory(), "streamed QSA graphs require a resident indexer");
        if(managed_memory()) {
            if(graph_prepared_)return;
            const auto n=position();
            memory_->ensure(n);
            // A retained window can start a new prompt before any positions exist.
            const auto positions=full_positions.defined()?full_positions:
                tb::empty({0},index_.storage().options().dtype(tb::kInt64));
            pooled_.update(*rotary_,index_.storage(),w_.index_key_norm,positions,n,c_.pool,c_.eps);
            pooled_.prepare_graph(positions,c_.maximum);
            graph_prepared_=true;return;
        }
        if(graph_keys_.defined())return;
        MFQ_RUNTIME_CHECK(position()>0 && keys_.storage().size(0)==1,
            "QSA graph requires a prefilled single sequence");
        const auto n=position();
        MFQ_RUNTIME_CHECK(capacity>=n && capacity<=c_.maximum,"invalid QSA graph capacity");
        auto graph_keys=tb::zeros({1,c_.kv_heads,capacity,key_width()},keys_.storage().options());
        auto graph_values=tb::zeros({1,c_.kv_heads,capacity,value_width()},values_.storage().options());
        graph_keys.narrow(2,0,n).copy_(keys_.storage().narrow(1,0,n).reshape({1,n,c_.kv_heads,key_width()}).permute({0,2,1,3}));
        graph_values.narrow(2,0,n).copy_(values_.storage().narrow(1,0,n).reshape({1,n,c_.kv_heads,value_width()}).permute({0,2,1,3}));
        index_.prepare_fixed(capacity);
        auto graph_pooled=tb::zeros({1,(capacity+c_.pool-1)/c_.pool,c_.index_width},index_.storage().options());
        pooled_.bind(graph_pooled);
        pooled_.update(*rotary_,index_.storage(),w_.index_key_norm,full_positions,n,c_.pool,c_.eps);
        pooled_.prepare_graph(full_positions,capacity);
        // Commit only after all allocation/copy submissions succeed. Replay
        // reads graph KV; retaining the old layout duplicates the full prefix
        // at every growth boundary. leave_graph recreates it when needed.
        graph_keys_=std::move(graph_keys);graph_values_=std::move(graph_values);graph_pooled_=std::move(graph_pooled);
        keys_.release_storage();values_.release_storage();
    }
    void bind_decode_state(BatchDecodeInputs&, int64_t field, int64_t row) const;
    int64_t decode_graph_capacity() const;
    // The independent core requires physically contiguous request KV.
    bool independent_graph_supported() const {
        return !managed_memory() || !memory_->hot.exclusive();
    }
    Tensor forward_independent_graph(CudaExecutionContext&, const Tensor& hidden, const Tensor& positions,
        const BatchDecodeInputs&, int64_t field, int64_t capacity, std::vector<Tensor> projections);
    Tensor forward_indirect_graph(CudaExecutionContext&, const Tensor& hidden, const Tensor& positions,
        const Tensor& cache_positions, const BatchDecodeInputs&, int64_t field, int64_t row,
        int64_t capacity, std::vector<Tensor> projections = {}, bool output_projection = true);
    Tensor forward_graph(CudaExecutionContext& execution,const Tensor& hidden,
                         const Tensor& positions,const Tensor& cache_positions,
                         std::vector<Tensor> projections={},bool project_output=true,bool history_only=false) {
        const auto mark=[](const char* phase) {
            if(auto* window=DecodeWindow::recording())window->mark_current(phase);
        };
        const bool streamed=bool(offload_);
        MFQ_RUNTIME_CHECK((graph_keys_.defined() || (streamed && managed_memory())) && hidden.size(0)==1 && cache_positions.dim()==1,
            "QSA graph cache is not prepared");
        const auto t=hidden.size(1),pools=graph_pooled_.size(1);
        const bool dense_capacity=(streamed?c_.maximum:graph_keys_.size(2))<=c_.budget;
        const bool exclusive=managed_memory() && memory_->hot.exclusive();
        auto p=history_only?project_history(execution,hidden,positions):
            project(execution,hidden,positions,!dense_capacity,cache_positions,
                       quantization_.enabled() || exclusive || streamed?Tensor{}:graph_keys_,quantization_.enabled() || exclusive || streamed?Tensor{}:graph_values_,
                       std::move(projections));
        mark("qsa_input_end");
        const auto finish=[&](Tensor output) {
            mark("qsa_attention_end");
            output=output.reshape({1,t,c_.heads*c_.width});
            return project_output?w_.output(execution,output):output;
        };
        bool index_written=false;
        const auto* prepare_option=std::getenv("MFQ_KV_PREPARE_FUSED");
        const bool begin_prepare=!prepare_option || prepare_option[0]!='0';
        if(exclusive) {
            const auto keys=(quantization_.enabled()?encode_key(p.key):p.key).permute({0,2,1,3});
            const auto values=(quantization_.enabled()?encode_value(p.value):p.value).permute({0,2,1,3});
            index_written=memory_->hot.append_decode(keys,values,cache_positions,p.raw,index_.storage(),begin_prepare);
            if(!index_written)memory_->hot.append(keys.to(cache_dtype()),values.to(cache_dtype()),cache_positions);
        }
        else if (!p.cache_written && !streamed) {
            graph_keys_.index_copy_(2,cache_positions,encode_key(p.key).permute({0,2,1,3}));
            graph_values_.index_copy_(2,cache_positions,encode_value(p.value).permute({0,2,1,3}));
        }
        if(managed_memory() && !exclusive && !streamed)memory_->hot.publish(graph_keys_,graph_values_,cache_positions);
        if(!index_written)index_.append_fixed(p.raw,cache_positions);
        mark("qsa_cache_end");
        if(history_only && streamed) {
            auto* window=DecodeWindow::recording();
            MFQ_RUNTIME_CHECK(window,"streamed QSA history requires a decode window");
            struct HistoryStage {Tensor raw;};
            const auto stage=window->workspace<HistoryStage>(this,[&] {
                ResidentAllocationScope resident;
                auto value=std::make_shared<HistoryStage>();
                value->raw=tb::empty({t,c_.kv_heads*(key_width()+value_width())},hidden.options().dtype(cache_dtype()));
                return value;
            });
            stage->raw.copy_(tb::cat({encode_key(p.key).reshape({t,c_.kv_heads*key_width()}),
                encode_value(p.value).reshape({t,c_.kv_heads*value_width()})},1));
            window->host_barrier([this,stage] {offloaded_kv_->append(stage->raw);});
        }
        if(dense_capacity) {
            if(history_only)return {};
            if(streamed) {
                auto ids=tb::arange(c_.maximum,cache_positions.options()).reshape({1,1,c_.maximum}).expand({1,t,c_.maximum});
                ids=tb::where(ids<=cache_positions.reshape({1,t,1}),ids,tb::full_like(ids,-1)).to(tb::kInt32).contiguous();
                return finish(streamed_graph_attention(p,ids,hidden.scalar_type()));
            }
            const auto columns=std::min(graph_keys_.size(2),c_.budget+c_.pool-1);
            const auto hot=prepare_hot({},cache_positions,columns,index_written && begin_prepare);
            if(quantization_.enabled()) {
                auto attended=cuda_kv_attention(p.query,graph_keys_,graph_values_,quantization_,{},cache_positions,0,{},0,{},{},0,hot);
                return finish((attended*tb::sigmoid(p.gate.to(tb::kFloat32))).to(hidden.scalar_type()));
            }
            if(fused_projection_ && rotary_->fused() &&
                (hidden.scalar_type()==tb::kFloat16 || hidden.scalar_type()==tb::kFloat32)) {
                auto gated=mfq_qwen4_exp::causal_gqa_attention_gate(p.query,graph_keys_,graph_values_,
                    cache_positions,p.gate,columns,hidden.scalar_type()==tb::kFloat16,hot);
                return finish(std::move(gated));
            }
            auto absolute=cache_positions.reshape({1,t,1});
            auto indices=tb::arange(columns,cache_positions.options()).reshape({1,1,columns}).expand({1,t,columns});
            indices=tb::where(indices<=absolute,indices,tb::full_like(indices,-1)).to(tb::kInt32).contiguous();
            auto attended=mfq_qwen4_exp::sparse_gqa_attention(p.query,graph_keys_,graph_values_,indices,hot);
            auto gated=attended.to(tb::kFloat32)*tb::sigmoid(p.gate.to(tb::kFloat32));
            return finish(gated.to(hidden.scalar_type()));
        }
        pooled_.update_graph(*rotary_,index_.storage(),w_.index_key_norm,positions,cache_positions,c_.pool,c_.eps);
        mark("qsa_pool_end");
        if(history_only)return {};
        auto ranked=mfq_qwen4_exp::live_block_scores(p.iq,graph_pooled_,cache_positions,c_.pool);
        mark("qsa_scores_end");
        auto absolute=cache_positions.reshape({1,t,1});
        const auto count=std::min(c_.budget/c_.pool,pools),columns=c_.budget+c_.pool-1;
        auto indices=mfq_qwen4_exp::live_topk_blocks(ranked,cache_positions,c_.pool,count);
        mark("qsa_topk_end");
        Tensor selected;
        const auto fused_selection=mfq::cuda::runtime_options::qsa_select_fused();
        if(!fused_selection || *fused_selection) {
            selected=mfq_qwen4_exp::qsa_selected_tokens(indices,cache_positions,c_.pool,c_.budget);
        } else {
        // Live top-k pads inactive slots with an out-of-range block ID.
        // Check positions directly rather than gathering a visibility table
        // with those sentinels when the unfused selector is requested.
        auto valid=((indices>=0)&(indices*c_.pool+c_.pool-1<=absolute)).unsqueeze(-1).expand({1,t,count,c_.pool});
        auto expanded=indices.unsqueeze(-1)*c_.pool+tb::arange(c_.pool,indices.options());
        expanded=tb::where(valid,expanded,tb::full_like(expanded,-1));
        selected=tb::full({1,t,columns},-1,cache_positions.options());
        selected.narrow(-1,0,count*c_.pool).copy_(expanded.reshape({1,t,count*c_.pool}));
        if(c_.pool>1) {
            auto tail_count=(absolute+1).remainder(c_.pool);
            auto offsets=tb::arange(c_.pool-1,cache_positions.options()).reshape({1,1,c_.pool-1});
            auto tail=absolute+1-tail_count+offsets;
            selected.narrow(-1,c_.budget,c_.pool-1).copy_(tb::where(offsets<tail_count,tail,tb::full_like(tail,-1)));
        }
        // Dense-budget queries select their complete causal prefix. The same
        // native selected-attention kernel serves both paths inside one graph.
        auto dense=tb::arange(columns,cache_positions.options()).reshape({1,1,columns}).expand({1,t,columns});
        dense=tb::where(dense<=absolute,dense,tb::full_like(dense,-1));
        selected=tb::where(absolute+1<=c_.budget,dense,selected).to(tb::kInt32).contiguous();
        }
        mark("qsa_select_end");
        if(streamed)return finish(streamed_graph_attention(p,selected,hidden.scalar_type()));
        Tensor gated;
        const auto hot=prepare_hot(selected,cache_positions,0,index_written && begin_prepare);
        mark("qsa_hot_end");
        if(quantization_.enabled()) {
            auto attended=cuda_kv_attention(p.query,graph_keys_,graph_values_,quantization_,selected,cache_positions,0,{},0,{},{},0,hot);
            gated=(attended*tb::sigmoid(p.gate.to(tb::kFloat32))).to(hidden.scalar_type());
        } else if(fused_projection_ && rotary_->fused() &&
            (hidden.scalar_type()==tb::kFloat16 || hidden.scalar_type()==tb::kFloat32)) {
            const auto sparse_gate=mfq::cuda::runtime_options::qsa_sparse_gate_fused();
            if(!sparse_gate || *sparse_gate)
                gated=mfq_qwen4_exp::sparse_gqa_attention_gate(p.query,graph_keys_,graph_values_,selected,
                    p.gate,hidden.scalar_type()==tb::kFloat16,hot);
            else gated=mfq_qwen4_exp::attention_gate(
                mfq_qwen4_exp::sparse_gqa_attention(p.query,graph_keys_,graph_values_,selected,hot),
                p.gate,hidden.scalar_type()==tb::kFloat16);
        } else {
            auto attended=mfq_qwen4_exp::sparse_gqa_attention(p.query,graph_keys_,graph_values_,selected,hot);
            gated=(attended.to(tb::kFloat32)*tb::sigmoid(p.gate.to(tb::kFloat32))).to(hidden.scalar_type());
        }
        return finish(std::move(gated));
    }
    void advance_graph(int64_t tokens) {
        if(!offload_){keys_.advance_fixed(tokens);values_.advance_fixed(tokens);}
        index_.advance_fixed(tokens);
        if((offload_?c_.maximum:graph_keys_.size(2))>c_.budget)
            pooled_.advance_graph(index_.position(),c_.pool);
    }
    void truncate(int64_t keep) {
        if (offload_) {
            MFQ_RUNTIME_CHECK(keep >= 0 && keep <= position(), "invalid offloaded QSA truncation");
            pooled_.truncate(keep,c_.pool);
            offloaded_kv_->truncate(keep);
            if(managed_memory())index_.truncate(keep);
            else {offloaded_index_->truncate(keep);offloaded_pooled_->truncate(keep/c_.pool);}
            return;
        }
        MFQ_RUNTIME_CHECK(keep >= 0 && keep <= keys_.position() && keep <= values_.position() &&
                              keep <= index_.position(),
                          "invalid Qwen4 QSA cache truncation");
        pooled_.truncate(keep,c_.pool);
        keys_.truncate(keep);
        values_.truncate(keep);
        index_.truncate(keep);
    }
    Tensor forward(CudaExecutionContext &execution, const Tensor &hidden,
                   const Tensor &current_positions, const Tensor &full_positions, bool use_cache,
                   std::vector<Tensor> *selection_trace = nullptr,
                   std::vector<Tensor> projections = {}, bool project_output = true,
                   bool history_only = false) {
        MFQ_RUNTIME_CHECK(hidden.is_cuda() && hidden.dim() == 3 && hidden.size(0) > 0 &&
                              hidden.size(1) > 0,
                          "Qwen4 QSA requires nonempty [B,T,H] input");
        activate_memory();
        MFQ_RUNTIME_CHECK(!history_only || use_cache,"QSA history priming requires a cache");
        if(use_cache && hidden.size(1)>8 && !selection_trace && !history_only) {
            // Index scores grow with both query rows and the accumulated
            // context. Bound that workspace independently of the FFN tile.
            // Keep per-head products, reduction and top-k scratch below the
            // same budget even at the final chunk of a long prompt.
            const auto offset=position(),length=offset+hidden.size(1);
            const auto pools=std::max<int64_t>(1,length/c_.pool);
            const auto row_bytes=std::size_t(hidden.size(0))*pools*(c_.index_heads+4)*sizeof(float);
            const auto rows=std::max<int64_t>(32,int64_t((128ULL<<20)/row_bytes)/32*32);
            if(hidden.size(1)>rows) {
                std::vector<Tensor> outputs;
                for(int64_t begin=0;begin<hidden.size(1);begin+=rows) {
                    const auto count=std::min(rows,hidden.size(1)-begin);
                    std::vector<Tensor> projected;
                    for(const auto& value:projections)projected.push_back(value.narrow(1,begin,count));
                    outputs.push_back(forward(execution,hidden.narrow(1,begin,count),
                        current_positions.narrow(-1,begin,count),full_positions.narrow(-1,0,offset+begin+count),
                        true,nullptr,std::move(projected),project_output));
                }
                return tb::cat(outputs,1);
            }
        }
        if (offload_ && use_cache) {
            if(managed_memory())memory_->ensure(position()+hidden.size(1));
            return forward_offloaded(execution, hidden, current_positions, full_positions, selection_trace,
                                     std::move(projections),project_output,history_only);
        }
        const auto b = hidden.size(0), t = hidden.size(1), offset = use_cache ? position() : 0;
        const auto prefill_option=mfq::cuda::runtime_options::qsa_prefill_fused();
        // Scalar decode and verification use the same grouped projections and
        // normalization/RoPE preparation as captured execution. Keeping short
        // eager calls on separate norms introduces extra FP16 rounding steps.
        const bool prepared_inputs=!projections.empty() || (!prefill_option || *prefill_option!=0);
        MFQ_RUNTIME_CHECK(
            t <= c_.maximum - offset && full_positions.size(-1) == offset + t &&
                (!use_cache || (offset == values_.position() && offset == index_.position())),
            "Qwen4 QSA cache/position geometry mismatch");
        auto project_inputs = [&] {
                if(prepared_inputs)
                    return project(execution,hidden,current_positions,!history_only && offset+t>c_.budget,{},{},{},std::move(projections));
                auto pair = w_.query(execution, hidden).reshape({b, t, c_.heads, 2 * c_.width});
                auto gate = pair.narrow(-1, c_.width, c_.width);
                auto query = rms_norm(pair.narrow(-1, 0, c_.width),
                                      w_.query_norm.to(tb::kFloat32) + 1, c_.eps)
                                 .permute({0, 2, 1, 3});
                auto key =
                    rms_norm(w_.key(execution, hidden).reshape({b, t, c_.kv_heads, c_.width}),
                             w_.key_norm.to(tb::kFloat32) + 1, c_.eps)
                        .permute({0, 2, 1, 3});
                auto value = w_.value(execution, hidden).reshape({b, t, c_.kv_heads, c_.width});
                query = rotary_->forward(query, current_positions);
                key = rotary_->forward(key, current_positions).permute({0, 2, 1, 3});
                auto iqk = w_.index_query_key(execution, hidden);
                auto iq = rms_norm(iqk.narrow(-1, 0, c_.index_heads * c_.index_width)
                                       .reshape({b, t, c_.index_heads, c_.index_width}),
                                   w_.index_query_norm.to(tb::kFloat32) + 1, c_.eps)
                              .permute({0, 2, 1, 3});
                iq = rotary_->forward(iq, current_positions).permute({0, 2, 1, 3});
                auto raw = iqk.narrow(-1, c_.index_heads * c_.index_width, c_.index_width);
                return Projection{gate, query, key, value, iq, raw};
            };
        auto append_inputs = [&](Projection &p) {
                if(managed_memory()) {
                    MFQ_RUNTIME_CHECK(b==1,"preallocated QSA expects one sequence per cache");
                    memory_->ensure(offset+t);
                    if(memory_->hot.exclusive())memory_->hot.append(encode_key(p.key).permute({0,2,1,3}),
                        encode_value(p.value).permute({0,2,1,3}),tb::arange(offset,offset+t,hidden.options().dtype(tb::kInt64)));
                    else {
                        graph_keys_.narrow(2,offset,t).copy_(encode_key(p.key).permute({0,2,1,3}));
                        graph_values_.narrow(2,offset,t).copy_(encode_value(p.value).permute({0,2,1,3}));
                    }
                    if(memory_->hot.defined()) {
                        if(!memory_->hot.exclusive())memory_->hot.publish(graph_keys_,graph_values_,tb::arange(offset,offset+t,hidden.options().dtype(tb::kInt64)));
                        p.key=graph_keys_.permute({0,2,1,3});p.value=graph_values_.permute({0,2,1,3});
                    }else {
                        p.key=graph_keys_.narrow(2,0,offset+t).permute({0,2,1,3});
                        p.value=graph_values_.narrow(2,0,offset+t).permute({0,2,1,3});
                    }
                    keys_.advance_fixed(t);values_.advance_fixed(t);graph_prepared_=false;
                } else {
                    p.key = keys_.append(encode_key(p.key).reshape({b, t, -1}))
                                .reshape({b, offset + t, c_.kv_heads, key_width()});
                    p.value = values_.append(encode_value(p.value).reshape({b, t, -1}))
                                  .reshape({b, offset + t, c_.kv_heads, value_width()});
                }
                p.raw = index_.append(p.raw);
            };
        if(history_only) {
            auto p=project_history(execution,hidden,current_positions);
            try {append_inputs(p);}catch(...) {truncate(offset);throw;}
            return {};
        }
        const bool short_gate=prepared_inputs && fused_projection_ && rotary_->fused() &&
            t<=8 && !quantization_.enabled() &&
            (hidden.scalar_type()==tb::kFloat16 || hidden.scalar_type()==tb::kFloat32);
        const auto attend_projected=[&](const Projection& p,const Tensor& selected) {
            auto key=p.key.permute({0,2,1,3}),value=p.value.permute({0,2,1,3});
            if(!short_gate)return attend(p.query,key,value,selected,offset);
            const auto positions=tb::arange(offset,offset+t,hidden.options().dtype(tb::kInt64));
            const auto hot=prepare_hot(selected,positions,offset+t);
            // Match the decode graph's fused gate and FP32 attention
            // accumulation for short eager/correction calls as well.
            if(selected.defined())return mfq_qwen4_exp::sparse_gqa_attention_gate(
                p.query,key,value,selected,p.gate,hidden.scalar_type()==tb::kFloat16,hot);
            return mfq_qwen4_exp::causal_gqa_attention_gate(p.query,key,value,positions,
                p.gate,offset+t,hidden.scalar_type()==tb::kFloat16,hot);
        };
        return mfq::models::qwen4_exp::sparse_attention(
            offset + t, c_.budget, use_cache, project_inputs, append_inputs,
            [&](const Projection &p) {
                return attend_projected(p,{});
            },
            [&](const Projection &p) {
                return pooled_.update(*rotary_,p.raw,w_.index_key_norm,full_positions,
                    offset+t,c_.pool,c_.eps,use_cache);
            },
            [](const Projection &p, const Tensor &pooled) {
                return mfq_qwen4_exp::block_scores(p.iq, pooled);
            },
            [&](const Tensor &scores, const Projection &p, const Tensor &pooled) {
                auto ids =
                    select_pooled_blocks(scores, offset, offset + t, c_.pool, c_.budget, true);
                if (selection_trace)
                    *selection_trace = {scores, ids, p.iq, pooled};
                return ids;
            },
            [&](const Projection &p, Tensor ids) {
                return attend_projected(p,ids);
            },
            [&](Tensor attended, const Projection &p) {
                auto gated = short_gate?std::move(attended):
                    attended.to(tb::kFloat32) * tb::sigmoid(p.gate.to(tb::kFloat32));
                auto result = gated.reshape({b, t, c_.heads * c_.width}).to(hidden.scalar_type());
                return project_output ? w_.output(execution,result) : result;
            },
            [&] { truncate(offset); });
    }

  private:
    HotKvView prepare_hot(const Tensor& selected,const Tensor& positions={},int64_t columns=0,bool epoch_ready=false) const {
        if(!managed_memory() || !memory_->hot.defined())return {};
        memory_->hot.prepare(graph_keys_,graph_values_,selected,positions,columns,true,epoch_ready);
        return memory_->hot.view();
    }
    Tensor attend(const Tensor& query, Tensor keys, Tensor values, const Tensor& selected, int64_t offset) const {
        if(managed_memory() && memory_->hot.defined()) {
            const auto positions=tb::arange(offset,offset+query.size(2),query.options().dtype(tb::kInt64));
            const auto hot=offset+query.size(2)<=memory_->hot.keys.size(2)?memory_->hot.view():prepare_hot(selected,positions,offset+query.size(2));
            if(quantization_.enabled())return cuda_kv_attention(query,keys,values,quantization_,selected,positions,offset,{},0,{},{},0,hot);
            if(selected.defined())return mfq_qwen4_exp::sparse_gqa_attention(query,keys,values,selected,hot);
            return mfq_qwen4_exp::dense_gqa_attention(query,keys,values,offset,hot);
        }
        if (keys.scalar_type() == tb::kInt32) {
            // Large query tiles amortize one decode per KV row and use the
            // existing matrix attention kernels. Persistent caches stay packed;
            // floating workspace is bounded and lives only for this layer.
            const auto floating_bytes = std::size_t(keys.size(0)) * keys.size(1) * keys.size(2) * c_.width * 4;
            if (query.size(2) <= 8 || floating_bytes > (128ULL << 20))
                return cuda_kv_attention(query, keys, values, quantization_, selected, {}, offset);
            keys = cuda_kv_decode(keys, int(c_.width), quantization_.key_bits(), false);
            values = cuda_kv_decode(values, int(c_.width), quantization_.value_bits(), true);
        }
        if (selected.defined())
            return mfq_qwen4_exp::sparse_gqa_attention(query, keys, values, selected);
        return mfq_qwen4_exp::dense_gqa_attention(query, keys, values, offset);
    }
    int64_t key_width() const { return quantization_.enabled()?cuda_kv_packed_width(int(c_.width),quantization_.key_bits()):c_.width; }
    int64_t value_width() const { return quantization_.enabled()?cuda_kv_packed_width(int(c_.width),quantization_.value_bits()):c_.width; }
    tb::ScalarType cache_dtype() const { return quantization_.enabled()?tb::kInt32:tb::kFloat16; }
    Tensor encode_key(const Tensor& x) const { return quantization_.enabled()?cuda_kv_encode(x,quantization_.key_bits(),false):x.to(tb::kFloat16); }
    Tensor encode_value(const Tensor& x) const { return quantization_.enabled()?cuda_kv_encode(x,quantization_.value_bits(),true):x.to(tb::kFloat16); }
    struct Projection {Tensor gate,query,key,value,iq,raw;bool cache_written=false;};
    Tensor streamed_graph_attention(const Projection& p,const Tensor& selected,tb::ScalarType output_type) {
        auto* window=DecodeWindow::recording();
        MFQ_RUNTIME_CHECK(window && selected.size(1)<=8,"streamed QSA requires a decode window of at most eight rows");
        struct Stage {
            Tensor raw,host_ids,ids;
            std::shared_ptr<KvOffloadStore::GatherWorkspace> gather;
        };
        const auto tokens=selected.size(1),columns=selected.size(2);
        const auto key_row=c_.kv_heads*key_width(),value_row=c_.kv_heads*value_width();
        const auto stage=window->workspace<Stage>(this,[&] {
            ResidentAllocationScope resident;
            auto result=std::make_shared<Stage>();
            result->raw=tb::empty({tokens,key_row+value_row},p.key.options().dtype(cache_dtype()));
            result->host_ids=tb::empty(selected.sizes(),selected.options().device(tb::kCPU).pinned_memory(true));
            result->ids=tb::zeros(selected.sizes(),selected.options());
            // Every layer and MTP width shares the same bounded I/O workspace.
            // A later segment may reuse it only after the prior consumer ends.
            result->gather=offloaded_kv_->gather_workspace(8*columns,c_.kv_heads,key_width());
            return result;
        });
        stage->raw.copy_(tb::cat({encode_key(p.key).reshape({tokens,key_row}),encode_value(p.value).reshape({tokens,value_row})},1));
        MFQ_CUDA_CHECK(cudaMemcpyAsync(stage->host_ids.data_ptr(),selected.data_ptr(),selected.nbytes(),
            cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        window->host_barrier([this,stage] {
            const auto start=position();
            try {
                offloaded_kv_->append(stage->raw);
                auto* ids=stage->host_ids.data_ptr<int32_t>();
                std::unordered_map<int32_t,int32_t> compact;
                std::vector<int64_t> requested;
                requested.reserve(stage->host_ids.numel());compact.reserve(stage->host_ids.numel());
                for(int64_t i=0;i<stage->host_ids.numel();++i)if(ids[i]>=0) {
                    auto [item,added]=compact.emplace(ids[i],int32_t(requested.size()));
                    if(added)requested.push_back(ids[i]);ids[i]=item->second;
                }
                offloaded_kv_->gather_into(requested,*stage->gather);
                MFQ_CUDA_CHECK(cudaMemcpyAsync(stage->ids.data_ptr(),ids,stage->ids.nbytes(),
                    cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            } catch(...) {offloaded_kv_->truncate(start);throw;}
        });
        const auto rows=stage->gather->rows.size(0);
        auto flat=stage->gather->rows.reshape({-1});
        auto keys=flat.narrow(0,0,rows*key_row).reshape({1,c_.kv_heads,rows,key_width()});
        auto values=flat.narrow(0,rows*key_row,rows*value_row).reshape({1,c_.kv_heads,rows,value_width()});
        if(!quantization_.enabled() && fused_projection_ && rotary_->fused() && (output_type==tb::kFloat16 || output_type==tb::kFloat32))
            return mfq_qwen4_exp::sparse_gqa_attention_gate(p.query,keys,values,stage->ids,p.gate,output_type==tb::kFloat16);
        auto attended=attend(p.query,keys,values,stage->ids,rows);
        return (attended.to(tb::kFloat32)*tb::sigmoid(p.gate.to(tb::kFloat32))).to(output_type);
    }
    Tensor forward_offloaded(CudaExecutionContext& execution, const Tensor& hidden,
        const Tensor& positions, const Tensor& full_positions, std::vector<Tensor>* trace,
        std::vector<Tensor> projections={},bool project_output=true,bool history_only=false) {
        const auto offset = position(), tokens = hidden.size(1), length = offset + tokens;
        const auto pool_position=[&]{return pooled_.completed();};
        const auto pool_range=[&](int64_t start,int64_t count){return managed_memory()
            ? graph_pooled_.narrow(1,start,count).reshape({count,c_.index_width}) : offloaded_pooled_->range(start,count);};
        MFQ_RUNTIME_CHECK(hidden.size(0) == 1 && length <= c_.maximum && full_positions.size(-1) == length,
            "streamed QSA requires one sequence within its configured context");
        const auto key_row = c_.kv_heads * key_width(), value_row = c_.kv_heads * value_width();
        const auto max_keys = std::min(length, c_.budget + c_.pool - 1);
        const auto row_bytes = std::size_t(key_row+value_row) * (quantization_.enabled()?4:2);
        if (std::size_t(max_keys) * row_bytes > offload_->config().buffer_bytes / 4)
            throw std::invalid_argument("QSA I/O buffer cannot hold one selected context; increase the resident KV budget");
        auto p = history_only?project_history(execution,hidden,positions):
            project(execution, hidden, positions, length > c_.budget,{},{},{},std::move(projections));
        try {
            offloaded_kv_->append(tb::cat({encode_key(p.key).reshape({tokens,key_row}), encode_value(p.value).reshape({tokens,value_row})}, 1));
            const auto first_pool = pool_position();
            if(managed_memory())index_.append(p.raw);
            else offloaded_index_->append(p.raw.reshape({tokens,c_.index_width}));
            const auto new_pools = length / c_.pool - first_pool;
            if (new_pools) {
                if(managed_memory()) {
                    pooled_.update(*rotary_,index_.storage(),w_.index_key_norm,full_positions,
                        length,c_.pool,c_.eps);
                } else {
                    auto raw=offloaded_index_->range(first_pool*c_.pool,new_pools*c_.pool)
                        .reshape({1,new_pools*c_.pool,c_.index_width});
                    auto pooled=pooled_.update_range(*rotary_,raw,w_.index_key_norm,full_positions,
                        first_pool,new_pools,c_.pool,c_.eps);
                    offloaded_pooled_->append(pooled.reshape({new_pools,c_.index_width}));
                    pooled_.advance_graph(length,c_.pool);
                }
            }
            if(history_only)return {};
            // When one layer fits in staging, reuse that layer across the
            // entire prefill tile instead of gathering the same rows per token.
            // Other layers remain under the GPU/RAM/SSD eviction policy.
            if (tokens > 8 && std::size_t(length) * row_bytes <= offload_->config().buffer_bytes / 2 &&
                    std::size_t(length / c_.pool) * c_.index_width * 2 <= offload_->config().buffer_bytes / 4) {
                auto packed = offloaded_kv_->range(0, length);
                auto key = packed.narrow(1,0,key_row).reshape({1,length,c_.kv_heads,key_width()}).permute({0,2,1,3});
                auto value = packed.narrow(1,key_row,value_row).reshape({1,length,c_.kv_heads,value_width()}).permute({0,2,1,3});
                Tensor ids;
                if (length > c_.budget) {
                    auto pooled = pool_range(0,length / c_.pool).reshape({1,length / c_.pool,c_.index_width});
                    auto scores = mfq_qwen4_exp::block_scores(p.iq, pooled);
                    ids = select_pooled_blocks(scores, offset, length, c_.pool, c_.budget, true);
                    if (trace) *trace = {scores,ids,p.iq,pooled};
                }
                auto attended = attend(p.query, key, value, ids, offset);
                auto gated = attended.to(tb::kFloat32) * tb::sigmoid(p.gate.to(tb::kFloat32));
                auto result = gated.reshape({1,tokens,c_.heads*c_.width}).to(hidden.scalar_type());
                return project_output ? w_.output(execution,result) : result;
            }
            const auto queries = std::max<int64_t>(1, std::min<int64_t>(32,
                offload_->config().buffer_bytes / 4 / (std::size_t(max_keys) * row_bytes)));
            std::vector<Tensor> outputs;
            for (int64_t begin = 0; begin < tokens; begin += queries) {
                const auto count = std::min(queries, tokens - begin);
                auto cache_positions = tb::arange(offset + begin, offset + begin + count, positions.options().dtype(tb::kInt64));
                Tensor selected;
                if (length <= c_.budget) {
                    auto ids = tb::arange(max_keys, cache_positions.options()).reshape({1, 1, max_keys}).expand({1, count, max_keys});
                    selected = tb::where(ids <= cache_positions.reshape({1, count, 1}), ids, tb::full_like(ids, -1)).to(tb::kInt32).contiguous();
                } else {
                    const auto pools = pool_position(), keep = std::min(c_.budget / c_.pool, pools);
                    Tensor best_scores, best_ids;
                    std::vector<Tensor> traced_scores, traced_pools;
                    const auto pool_chunk = std::max<int64_t>(16, std::min<int64_t>(1024,
                        offload_->config().buffer_bytes / 8 / (c_.index_width * 2 + count * 4)));
                    for (int64_t start = 0; start < pools; start += pool_chunk) {
                        const auto n = std::min(pool_chunk, pools - start);
                        auto pooled = pool_range(start, n).reshape({1, n, c_.index_width});
                        auto scores = mfq_qwen4_exp::block_scores(p.iq.narrow(1, begin, count), pooled);
                        auto ids = tb::arange(start, start + n, cache_positions.options());
                        auto visible = (ids * c_.pool + c_.pool - 1).reshape({1, 1, n}) <= cache_positions.reshape({1, count, 1});
                        auto ranked = tb::where(visible, scores.to(tb::kFloat32), tb::full_like(scores, -1e30).to(tb::kFloat32));
                        auto top = tb::topk(ranked, std::min(keep, n), -1, true, false);
                        auto values = std::get<0>(top), indices = std::get<1>(top) + start;
                        if (best_scores.defined()) {
                            values = tb::cat({best_scores, values}, -1);
                            indices = tb::cat({best_ids, indices}, -1);
                            top = tb::topk(values, std::min(keep, values.size(-1)), -1, true, false);
                            best_scores = std::get<0>(top); best_ids = indices.gather(-1, std::get<1>(top));
                        } else { best_scores = values; best_ids = indices; }
                        if (trace) { traced_scores.push_back(scores); traced_pools.push_back(pooled); }
                    }
                    selected = mfq_qwen4_exp::qsa_selected_tokens(best_ids, cache_positions, c_.pool, c_.budget);
                    if (trace && begin == 0) *trace = {tb::cat(traced_scores, -1), selected, p.iq.narrow(1, begin, count), tb::cat(traced_pools, 1)};
                }
                auto cpu_ids = selected.to(tb::kCPU).contiguous();
                auto* mapping = static_cast<int32_t*>(cpu_ids.data_ptr());
                std::unordered_map<int32_t, int32_t> compact;
                std::vector<int64_t> requested;
                for (int64_t index = 0; index < cpu_ids.numel(); ++index) {
                    if (mapping[index] < 0) continue;
                    auto [found, added] = compact.emplace(mapping[index], static_cast<int32_t>(requested.size()));
                    if (added) requested.push_back(mapping[index]);
                    mapping[index] = found->second;
                }
                auto packed = offloaded_kv_->gather(requested);
                const auto n = static_cast<int64_t>(requested.size());
                auto key = packed.narrow(1,0,key_row).reshape({1,n,c_.kv_heads,key_width()}).permute({0,2,1,3}).contiguous();
                auto value = packed.narrow(1,key_row,value_row).reshape({1,n,c_.kv_heads,value_width()}).permute({0,2,1,3}).contiguous();
                // IDs now address a compact union; selection already contains the causal mask.
                outputs.push_back(attend(p.query.narrow(2,begin,count),key,value,cpu_ids.to(selected.device()),n));
            }
            auto attended = tb::cat(outputs, 1);
            auto gated = attended.to(tb::kFloat32) * tb::sigmoid(p.gate.to(tb::kFloat32));
            auto result=gated.reshape({1, tokens, c_.heads * c_.width}).to(hidden.scalar_type());
            return project_output?w_.output(execution,result):result;
        } catch (...) { truncate(offset); throw; }
    }
    Tensor project_norm(const Tensor& value,const Tensor& weight) {
        if(value.scalar_type()==tb::kFloat16 || value.scalar_type()==tb::kFloat32)
            return grouped_rms_norm_cuda(value.contiguous(),weight.to(tb::kFloat32).contiguous(),value.size(-1),c_.eps,1.0);
        return rms_norm(value,weight.to(tb::kFloat32)+1,c_.eps);
    }
    Projection project_history(CudaExecutionContext& execution,const Tensor& hidden,const Tensor& positions) {
        const auto b=hidden.size(0),t=hidden.size(1);
        auto source=w_.key(execution,hidden).reshape({b,t,c_.kv_heads,c_.width});
        Tensor key;
        if(fused_projection_ && rotary_->fused() &&
                (source.scalar_type()==tb::kFloat16 || source.scalar_type()==tb::kFloat32))
            key=rotary_->forward_normalized_grouped({source},{w_.key_norm},positions,c_.eps,1)[0].permute({0,2,1,3});
        else key=rotary_->forward(project_norm(source,w_.key_norm).permute({0,2,1,3}),positions).permute({0,2,1,3});
        auto value=w_.value(execution,hidden).reshape({b,t,c_.kv_heads,c_.width});
        auto raw=w_.index_query_key(execution,hidden).narrow(-1,c_.index_heads*c_.index_width,c_.index_width);
        return {{},{},std::move(key),std::move(value),{},std::move(raw)};
    }
    Projection project(CudaExecutionContext& execution,const Tensor& hidden,const Tensor& positions,
                       bool index_query_needed=true,const Tensor& cache_positions={},
                       const Tensor& key_cache={},const Tensor& value_cache={},
                       std::vector<Tensor> projections={}) {
        const auto b=hidden.size(0),t=hidden.size(1);
        if(projections.empty() && grouped_projection_ && w_.input_projection) {
            projections=w_.input_projection(execution,hidden);
        }
        MFQ_RUNTIME_CHECK(projections.empty() || projections.size()==4,"QSA input projection count mismatch");
        auto pair=(projections.empty()?w_.query(execution,hidden):projections[0]).reshape({b,t,c_.heads,2*c_.width});
        auto gate=pair.narrow(-1,c_.width,c_.width);
        auto key_source=(projections.empty()?w_.key(execution,hidden):projections[1]).reshape({b,t,c_.kv_heads,c_.width});
        auto value=(projections.empty()?w_.value(execution,hidden):projections[2]).reshape({b,t,c_.kv_heads,c_.width});
        const auto supported=[](const Tensor& x) { return x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32; };
        const bool fused=fused_projection_ && rotary_->fused() && supported(pair) && supported(key_source) && supported(value);
        const bool cache_written=fused && key_cache.defined();
        auto iqk=projections.empty()?w_.index_query_key(execution,hidden):projections[3];
        Tensor query,key,iq;
        if(fused) {
            std::vector<Tensor> inputs{pair.narrow(-1,0,c_.width),key_source};
            std::vector<Tensor> weights{w_.query_norm,w_.key_norm};
            if(index_query_needed) {
                auto source=iqk.narrow(-1,0,c_.index_heads*c_.index_width).reshape({b,t,c_.index_heads,c_.index_width});
                if(supported(source)){inputs.push_back(source);weights.push_back(w_.index_query_norm);}
            }
            auto normalized=rotary_->forward_normalized_grouped(inputs,weights,positions,c_.eps,1,
                cache_written?key_cache:Tensor{},cache_written?cache_positions:Tensor{},
                cache_written?value:Tensor{},cache_written?value_cache:Tensor{});
            query=normalized[0];key=normalized[1];
            if(key.defined())key=key.permute({0,2,1,3});
            if(normalized.size()==3)iq=normalized[2].permute({0,2,1,3});
        } else {
            query=project_norm(pair.narrow(-1,0,c_.width),w_.query_norm).permute({0,2,1,3});
            key=project_norm(key_source,w_.key_norm).permute({0,2,1,3});
            query=rotary_->forward(query,positions);key=rotary_->forward(key,positions).permute({0,2,1,3});
        }
        if(index_query_needed && !iq.defined()) {
            auto source=iqk.narrow(-1,0,c_.index_heads*c_.index_width).reshape({b,t,c_.index_heads,c_.index_width});
            if(fused && supported(source))iq=rotary_->forward_normalized(source,w_.index_query_norm,positions,c_.eps).permute({0,2,1,3});
            else {
                iq=project_norm(source,w_.index_query_norm).permute({0,2,1,3});
                iq=rotary_->forward(iq,positions).permute({0,2,1,3});
            }
        }
        return {gate,query,key,value,iq,iqk.narrow(-1,c_.index_heads*c_.index_width,c_.index_width),cache_written};
    }
    QsaWeights w_;
    QsaConfig c_;
    KvQuantization quantization_=cuda_kv_quantization();
    std::shared_ptr<RotaryEmbedding> rotary_;
    bool fused_projection_;
    bool grouped_projection_;
    SequenceCache keys_, values_, index_;
    std::shared_ptr<KvOffloadStore> offload_;
    std::unique_ptr<KvOffloadSequence> offloaded_kv_, offloaded_index_, offloaded_pooled_;
    PooledCache pooled_;
    Tensor graph_keys_,graph_values_,graph_pooled_;
    std::shared_ptr<KvMemoryAllocation> memory_;
    std::shared_ptr<KvMemoryManager> memory_manager_;
    bool graph_prepared_=false;
};
} // namespace mfq::cuda::qwen4_exp
