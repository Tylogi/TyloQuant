#include "runtime/routed_cache_resources.h"
#include "generation.h"
#include "token_row_prefetch.h"
#include "speculative_sequence.h"

#include "cuda_execution.h"
#include "cuda_sampling.h"
#include "generation_policy.h"
#include "inference.h"
#include "mfq_cuda_sampling_ops.h"
#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/qwen35/ops.h"
#include "models/nemotron_h/ops.h"
#include "models/qwen4_exp/ops.h"
#include "models/qwen4_exp/mtp.h"
#include "moe_expert_cache.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

using mfq::cuda::internal::PrefillCudaTimer;

mfq_tensor_backend::Tensor MtpTarget::embed(mfq_tensor_backend::Tensor ids) const {
    return std::visit([&](auto* target) { return target->embed_forward(std::move(ids)); }, model);
}

mfq_tensor_backend::Tensor MtpTarget::logits(mfq_tensor_backend::Tensor hidden) const {
    return std::visit([&](auto* target) { return target->logits_from_hidden(std::move(hidden)); }, model);
}

using mfq::engine::Generation;
using namespace mfq::engine;
template <class Model> struct CudaMtpOps {
  using Tensor = mfq_tensor_backend::Tensor;
  Model &model;
  MtpModule &mtp;
  const CudaPreparedPrompt *prepared;
  MtpTarget target;
  bool transformed_prompt;
  int position_axes;
  int64_t prefill_compute_chunk;
  InferenceOutput* output;
  // Destroy after local/result tensors, including cancellation of priming.
  std::unique_ptr<MoePrefillWorkspaceLoan> predictor_loan;
  const mfq_tensor_backend::TensorOptions options =
      mfq_tensor_backend::TensorOptions()
          .device(mfq_tensor_backend::kCUDA)
          .dtype(mfq_tensor_backend::kInt64);
  Tensor input_ids, counts;
  Tensor known_ids_host, known_ids_device;
  Tensor early_ple_ids_host;
  Tensor projected_target_hidden, projected_target_logits;
  Tensor projected_predictor_hidden, projected_predictor_logits;
  Tensor projected_target_compact,projected_predictor_compact;
  int graph_sampling_reuses=0;
  int32_t projected_predictor_greedy=-1;
  std::vector<int32_t> projected_target_greedy;
  std::optional<mfq::cuda::Sampler> sampler_;
  bool penalties = false;
  bool target_maintenance = false;
  std::uint64_t target_graph_captures = 0;
  std::uint64_t target_transfer_bytes=0;
  std::array<double,6> target_profile{};
  std::shared_ptr<MoeExpertCache> target_route_cache;
  static bool profile_target() {
    static const bool enabled=[] {
      const auto* option=std::getenv("MFQ_PROFILE_MTP");
      return option && option[0]=='1';
    }();
    return enabled;
  }
  std::array<double,6> target_costs() const {
    std::array<double,6> result{};
    for(const auto& [name,value]:mfq::cuda::routed_expert_memory_metrics(*model.execution)) {
      if(name=="expert_route_wait_ms")result[0]=value;
      else if(name=="expert_host_fetch_ms")result[1]=value;
      else if(name=="expert_host_copy_ms")result[2]=value;
      else if(name=="ssd_expert_read_wait_ms")result[3]=value;
    }
    if(model.source) {
      const auto io=model.source->file_read_stats();
      result[4]=double(io.read_nanoseconds)/1e6;
      result[5]=double(io.physical_bytes)/1048576.0;
    }
    return result;
  }
  CudaMtpOps(CudaMtpOps&&)=default;
  ~CudaMtpOps() {
    if(target_route_cache)discard_moe_speculative_routes(target_route_cache);
  }
  void begin_target_routes(int rows) {
    begin_moe_speculative_routes(model.execution->moe_expert_cache,rows);
    target_route_cache=model.execution->moe_expert_cache;
  }
  void resolve_target_routes(int rows,bool cancelled) {
    if(!target_route_cache)return;
    if(cancelled)discard_moe_speculative_routes(target_route_cache);
    else commit_moe_speculative_routes(target_route_cache,rows);
    target_route_cache.reset();
  }
  bool take_target_maintenance() {
    const auto captures=moe_expert_graph_captures(model.execution->moe_expert_cache);
    const bool initialized=captures!=target_graph_captures;
    target_graph_captures=captures;
    return std::exchange(target_maintenance,false) || initialized;
  }
  bool gradual_mtp_warmup() const {return std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>;}
  bool metal_mtp_policy() const {
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>) {
      const auto* option=std::getenv("MFQ_MTP_METAL_POLICY");
      return !option || option[0]!='0';
    }
    return false;
  }
  bool prefetch_target_ids(int32_t token,const std::vector<int32_t>& drafts) {
    if constexpr(requires {model.adapter_prefetch_token_rows(model,early_ple_ids_host);}) {
      const auto* option=std::getenv("MFQ_MTP_EARLY_PLE");
      if(option && option[0]=='0')return false;
      const int tokens=mtp.split_target_verification()?1:1+int(drafts.size());
      if(!early_ple_ids_host.defined() || early_ple_ids_host.size(1)<tokens)
        early_ple_ids_host=mfq_tensor_backend::empty({1,tokens},options.device(mfq_tensor_backend::kCPU));
      auto* ids=early_ple_ids_host.template data_ptr<int64_t>();ids[0]=token;
      for(int i=1;i<tokens;++i)ids[i]=drafts[std::size_t(i-1)];
      if(mfq::cuda::internal::prefetch_token_rows(model,early_ple_ids_host.narrow(1,0,tokens))) {
        static bool reported=false;
        if(!reported) {
          reported=true;std::cerr<<"runtime_mtp_early_token_rows enabled=1 prefix_tokens=1\n";
        }
        if(tokens>1 && tokens<64) {
          static uint64_t reported_prefixes=0;const auto bit=uint64_t(1)<<tokens;
          if(!(reported_prefixes&bit)) {
            reported_prefixes|=bit;
            std::cerr<<"runtime_mtp_token_row_draft_prefix enabled=1 prefix_tokens="<<tokens<<'\n';
          }
        }
        return true;
      }
    }
    return false;
  }
  void prefetch_target_first(int32_t token) {prefetch_target_ids(token,{});}
  void prefetch_target_prefix(int32_t token,const std::vector<int32_t>& drafts) {
    const auto* option=std::getenv("MFQ_MTP_PLE_DRAFT_PREFIX");
    if(!option || option[0]!='0')prefetch_target_ids(token,drafts);
  }
  void prefetch_committed_token(int32_t token) {
    const auto* option=std::getenv("MFQ_MTP_PLE_PUBLICATION");
    if(option && option[0]=='0')return;
    if(prefetch_target_ids(token,{})) {
      static bool reported=false;
      if(!reported) {
        reported=true;std::cerr<<"runtime_mtp_token_row_publication enabled=1\n";
      }
    }
  }
  bool stream_prompt_history(int64_t tokens) const {
    const auto* option=std::getenv("MFQ_MTP_STREAM_PROMPT_HISTORY");
    return std::is_same_v<Model,mfq::cuda::Qwen4CausalLm> && tokens>16384 && (!option || option[0]!='0');
  }
  bool host_prompt_history(int64_t tokens) const {
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>) {
      const auto* option=std::getenv("MFQ_MTP_HOST_PROMPT_HISTORY");
      if(option)return option[0]!='0';
      const auto bytes=std::size_t(tokens)*model.hidden_size()*model.hc_mult()*sizeof(uint16_t);
      return bytes+(std::size_t(32)<<20)>
          mfq::cuda::default_context(mfq_current_cuda_device())->available_memory();
    }
    return false;
  }
  void trace_prompt_prefill(double target_ms,double predictor_ms,double workspace_ms) const {
    // Block predictors seed only their bounded target window and skip the
    // teacher-forced priming hook that restores workspace for chain predictors.
    if(!mtp.teacher_forced_prompt_prime() && model.restore_decode_workspace)
      model.restore_decode_workspace();
    std::cerr<<"runtime_mtp_prefill target_ms="<<target_ms<<" predictor_ms="<<predictor_ms
        <<" predictor_workspace_ms="<<workspace_ms<<" streamed="<<stream_prompt_history(input_ids.size(1))<<std::endl;
  }
  bool bootstrap_before_publication() const {
    const auto* option=std::getenv("MFQ_MTP_EARLY_BOOTSTRAP");
    return std::is_same_v<Model,mfq::cuda::Qwen4CausalLm> && (!option || option[0]!='0');
  }
  bool retains_partial_target_prefix(int accepted) const {
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>)
      if(model.speculative_prefix_retention && accepted>=0)return true;
    return mtp.retains_partial_target_prefix();
  }
  std::size_t predictor_history_merge_limit() const {
    return std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>?128:0;
  }
  std::size_t predictor_history_flush_threshold() const {
    // During plain-decode calibration, consume a full warmed graph tile
    // while that short cycle has room. Re-entry otherwise pays for several
    // history replays immediately before drafting and target verification.
    const auto* option=std::getenv("MFQ_MTP_HISTORY_EARLY_FLUSH");
    return std::is_same_v<Model,mfq::cuda::Qwen4CausalLm> && (!option || option[0]!='0')?4:512;
  }
  std::size_t predictor_parked_history_flush_threshold() const {
    // Amortize the target/predictor expert-exchange boundary while bounding
    // final and re-entry maintenance by four warmed, headless graph tiles.
    const auto* option=std::getenv("MFQ_MTP_HISTORY_EARLY_FLUSH");
    return std::is_same_v<Model,mfq::cuda::Qwen4CausalLm> && (!option || option[0]!='0')?16:512;
  }
  void trace_cycle(uint64_t cycle,int depth,int accepted,double draft_ms,double verify_ms,double commit_ms) {
      const auto* option=std::getenv("MFQ_PROFILE_MTP");
      const bool profile=option && option[0]=='1';
      if(profile || (cycle<=32 && draft_ms+verify_ms+commit_ms>=55.0)) {
        std::cerr<<(profile?"runtime_mtp_cycle_profile cycle=":"runtime_mtp_slow_cycle cycle=")<<cycle<<" depth="<<depth<<" accepted="<<accepted
            <<" draft_ms="<<draft_ms<<" verify_ms="<<verify_ms<<" commit_ms="<<commit_ms
            <<" target_transfer_mb="<<target_transfer_bytes/1048576.0;
        if(profile)
          std::cerr<<" target_route_ms="<<target_profile[0]<<" target_fetch_ms="<<target_profile[1]
              <<" target_host_copy_ms="<<target_profile[2]<<" target_ssd_wait_ms="<<target_profile[3]
              <<" target_parallel_read_ms="<<target_profile[4]<<" target_physical_read_mb="<<target_profile[5];
        std::cerr<<'\n';
      }
  }

  CudaMtpOps(Model &model, MtpModule &mtp, const CudaPreparedPrompt *prepared,
             int64_t compute_chunk, InferenceOutput* output)
      : model(model), mtp(mtp), prepared(prepared), target(model),
        transformed_prompt(prepared && prepared->transformed()),
        position_axes(transformed_prompt && prepared->positions.defined() && prepared->positions.dim() == 2 &&
                              prepared->positions.size(0) == 3
                          ? 3
                          : 1), prefill_compute_chunk(compute_chunk), output(output) {
    target_graph_captures=moe_expert_graph_captures(model.execution->moe_expert_cache);
  }
  void initialize(const std::vector<int64_t> &prompt,
                  const MfqSamplingParams &sampling) {
    input_ids = ids_for(prompt);
    sampler_.emplace(sampling);
    penalties = sampler().has_penalties();
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>)
      if(auto* predictor=dynamic_cast<mfq::cuda::qwen4_exp::Qwen4ExpMtp*>(&mtp))
        predictor->compact_graph_sampling=captured_compact_requested();
    if (penalties) {
      counts = mfq::cuda::SamplingOps::token_counts(input_ids, model.vocab_size());
    }
  }
  auto &sampler() { return *sampler_; }
  bool captured_compact_requested() const {
    const auto* option=std::getenv("MFQ_MTP_CONDITIONAL_TOPK");
    if(option && option[0]=='0')return true;
    const auto& parameters=sampler_->params();
    return !sampler_->greedy() && !penalties && parameters.top_k>0 && parameters.top_k<=64;
  }
  static void add_counts(Tensor counts, const Tensor &ids) {
    sample_token_counts_add_cuda(counts, ids);
  }
  static bool defined(const Tensor &value) { return value.defined(); }
  static int64_t size(const Tensor &value, int axis) {
    return value.size(axis);
  }
  static int rank(const Tensor &value) { return value.dim(); }
  static Tensor clone(const Tensor &value) {
    return value.is_cuda()?value.clone():value.to(mfq_tensor_backend::kCUDA);
  }
  static Tensor slice(const Tensor &value, int axis, int64_t offset,
                      int64_t length) {
    return value.narrow(axis, offset, length).contiguous();
  }
  static Tensor reshape(Tensor value, std::initializer_list<int64_t> shape) {
    return value.reshape(shape);
  }
  static Tensor concatenate(const std::vector<Tensor> &values, int axis) {
    if(std::any_of(values.begin(),values.end(),[](const auto& value){return !value.is_cuda();})) {
      auto host=values;
      for(auto& value:host)value=value.to(mfq_tensor_backend::kCPU);
      return mfq_tensor_backend::cat(host,axis).contiguous();
    }
    return mfq_tensor_backend::cat(values, axis).contiguous();
  }
  Tensor target_forward(Tensor ids, Tensor *raw = nullptr,
                        int64_t confirmed = 0) {
    projected_target_hidden = {}; projected_target_logits = {};
    projected_target_compact={};
    projected_target_greedy.clear();
    target_transfer_bytes = 0;
    const auto transfer_before=moe_expert_streamed_bytes(model.execution->moe_expert_cache);
    const auto costs_before=profile_target()?target_costs():std::array<double,6>{};
    const auto finish_profile=[&] {
      if(!profile_target())return;
      const auto costs_after=target_costs();
      for(std::size_t i=0;i<target_profile.size();++i)target_profile[i]=costs_after[i]-costs_before[i];
    };
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>) {
      if(!transformed_prompt) {
        bool initialized=false;
        const auto host_ids = known_ids_device.defined() &&
            ids.data_ptr() == known_ids_device.data_ptr() && ids.sizes() == known_ids_device.sizes()
            ? known_ids_host : Tensor{};
        Tensor logits;
        if(auto hidden=mfq::cuda::qwen4_decode_graph_hidden(model,ids,raw,confirmed,&initialized,host_ids,&logits,&projected_target_greedy,captured_compact_requested())) {
          target_transfer_bytes=moe_expert_streamed_bytes(model.execution->moe_expert_cache)-transfer_before;
          target_maintenance=target_maintenance || initialized;
          projected_target_hidden = *hidden; projected_target_logits = std::move(logits);
          projected_target_compact=mfq::cuda::qwen4_decode_graph_compact_topk(model);
          finish_profile();
          return std::move(*hidden);
        }
      }
    }
    auto hidden=model.hidden_forward(ids, mfq_nullopt, mfq_nullopt, nullptr,
                                     mfq_nullopt, raw, confirmed);
    target_transfer_bytes=moe_expert_streamed_bytes(model.execution->moe_expert_cache)-transfer_before;
    finish_profile();
    return hidden;
  }
  Tensor target_suffix(Tensor ids, Tensor *raw) {
    projected_target_hidden = {}; projected_target_logits = {};
    projected_target_compact={};
    projected_target_greedy.clear();
    const auto transfer_before=moe_expert_streamed_bytes(model.execution->moe_expert_cache);
    const auto costs_before=profile_target()?target_costs():std::array<double,6>{};
    auto hidden=model.hidden_forward_speculative_suffix(ids, raw);
    target_transfer_bytes+=moe_expert_streamed_bytes(model.execution->moe_expert_cache)-transfer_before;
    if(profile_target()) {
      const auto costs_after=target_costs();
      for(std::size_t i=0;i<target_profile.size();++i)target_profile[i]+=costs_after[i]-costs_before[i];
    }
    return hidden;
  }
  struct PrefillResult {
    Tensor hidden, raw;
    double elapsed;
  };
  double blockwise_prompt_context(Tensor hidden,int64_t position,bool seed) {
    PrefillCudaTimer timer;
    if(seed)mtp.seed_target_context(std::move(hidden),position);
    else {
      const auto start=position-hidden.size(1);
      mtp.append_target_context(std::move(hidden),start);
    }
    MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
    return timer.elapsed_ms();
  }
  PrefillResult prefill(mfq::engine::PrefillChunk chunk) {
    PrefillCudaTimer timer;
    bool prefill_finished=false;
    const auto finish_prefill=[&] {
      if(prefill_finished)return;
      MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
      prefill_finished=true;
    };
    Tensor raw;
    auto ids = input_ids.narrow(1, chunk.offset, chunk.count).contiguous();
    CudaImageMaskScope image_scope(model,prepared,chunk.offset,chunk.count);
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>) {
      if(!transformed_prompt && model.reuse_decode_graphs && model.graph_registry && chunk.count<=128) {
        // Reuse bounded routed windows for short prompts as well as continuations.
        // A whole-layer prefill can exceed the remaining workspace after MTP
        // graphs have been retained by an earlier request. Initialize a fresh
        // causal state with a retained scalar window when available; initial
        // loading without warmed state still uses an eager bootstrap row.
        std::vector<Tensor> hidden_rows,raw_rows;
        for(int64_t offset=0;offset<chunk.count;) {
          const auto rows=model.cache_pos==0?int64_t(1):std::min<int64_t>(4,chunk.count-offset);
          Tensor part;
          hidden_rows.push_back(target_forward(ids.narrow(1,offset,rows),&part,rows>1?1:0).clone());
          raw_rows.push_back(std::move(part));
          if(rows>1)model.commit_speculative();
          offset+=rows;
        }
        auto hidden=hidden_rows.size()==1?std::move(hidden_rows.front()):concatenate(hidden_rows,1);
        raw=raw_rows.size()==1?std::move(raw_rows.front()):concatenate(raw_rows,1);
        MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
        return {std::move(hidden),std::move(raw),timer.elapsed_ms()};
      }
      if(!transformed_prompt && mfq::cuda::qwen4_layer_major_prefill_eligible(model,chunk.count)) {
        // Sampling consumes only the final normalized row. Teacher-forced
        // predictor priming uses the separate, full raw hidden sequence.
        // Decide where the cross-phase history will live before target prefill.
        // It outlives the borrowed expert workspace; predictor priming consumes
        // bounded tiles, so a host copy avoids a sequence-sized VRAM reservation.
        const bool host_history=host_prompt_history(chunk.count);
        auto hidden=mfq::cuda::qwen4_layer_major_prefill_hidden(model,ids,prefill_compute_chunk,true,
            [&] { return output && output->stopped(); }, &raw,host_history,finish_prefill);
        finish_prefill();
        return {std::move(hidden),std::move(raw),timer.elapsed_ms()};
      }
    }
    if constexpr(std::is_same_v<Model,mfq::cuda::DeepseekV41CausalLm>) {
      if(!transformed_prompt) {
        auto hidden=mfq::cuda::deepseek_v41_prefill_hidden(model,ids,prefill_compute_chunk,
            [&] {return output && output->stopped();},&raw);
        finish_prefill();
        return {std::move(hidden),std::move(raw),timer.elapsed_ms()};
      }
    }
    MoePrefillWorkspaceLoan workspace(model.execution->moe_expert_cache);
    const bool host_history=host_prompt_history(chunk.count);
    bool borrowed=false;
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>)
      if(transformed_prompt)borrowed=workspace.begin((std::size_t(4)<<30)+
          std::size_t(chunk.count)*model.hidden_size()*model.hc_mult()*sizeof(float)*4)>0;
    auto hidden =
        transformed_prompt
            ? model.hidden_forward_inputs(
                  ids,
                  prepared->embeddings.narrow(1, chunk.offset, chunk.count)
                      .contiguous(),
                  prepared->positions.defined()?MfqOptional<Tensor>(
                      prepared->positions.narrow(-1,chunk.offset,chunk.count).contiguous()):mfq_nullopt,
                  mfq_nullopt, nullptr, mfq_nullopt, true, mfq_nullopt, &raw)
            : target_forward(ids, &raw);
    if(host_history)raw=raw.to(mfq_tensor_backend::kCPU);
    finish_prefill();
    if(borrowed) {
      workspace.suspend();
      hidden=hidden.narrow(1,hidden.size(1)-1,1).clone();
      if(!host_history)raw=raw.clone();
      workspace.finish();
    }
    return {std::move(hidden), std::move(raw), timer.elapsed_ms()};
  }
  Tensor ids_for(std::vector<int64_t> tokens) {
    known_ids_host = mfq_tensor_backend::tensor(tokens, options.device(mfq_tensor_backend::kCPU))
        .reshape({1, -1}).contiguous();
    known_ids_device = known_ids_host.to(mfq_tensor_backend::kCUDA);
    return known_ids_device;
  }
  Tensor decode_positions(int64_t cache_start, int64_t tokens) {
    MFQ_RUNTIME_CHECK(tokens > 0, "CUDA MTP position span must be positive");
    const int64_t logical_start =
        cache_start +
        (transformed_prompt ? prepared->decode_position_delta : 0);
    MFQ_RUNTIME_CHECK(logical_start >= 0 && logical_start + tokens <=
                                                model.max_position_embeddings(),
                      "CUDA MTP position span exceeds context capacity");
    auto result = mfq_tensor_backend::arange(logical_start,
                                             logical_start + tokens, options);
    return position_axes == 3
               ? result.reshape({1, tokens}).expand({3, tokens}).contiguous()
               : result;
  }
  MtpStep predictor_step(Tensor hidden, Tensor ids, Tensor positions) {
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>) {
      if(auto* predictor=dynamic_cast<mfq::cuda::qwen4_exp::Qwen4ExpMtp*>(&mtp);
          predictor && predictor->layers.size()==1 && model.reuse_decode_graphs &&
          model.graph_registry && ids.size(1)>1 && ids.size(1)<=128) {
        const auto prefix=ids.size(1)-1;
#ifdef MFQ_MTP_VERIFY_HISTORY_TAIL
        static thread_local std::array<bool,129> checked{};
        Tensor expected_hidden,expected_chain;
        const auto check_start=mtp.cache_position();
        if(!checked[ids.size(1)]) {
          const auto start=mtp.cache_position();
          auto expected=mtp.step_positioned(target,hidden,ids,positions);
          expected_hidden=expected.sample_hidden.narrow(1,prefix,1).clone();
          expected_chain=expected.chain_hidden.narrow(1,prefix,1).clone();
          mtp.trim_cache_to(start);
        }
#endif
        // Earlier teacher-forced rows contribute only predictor attention KV.
        // Their FFN/readout has no consumer; draft from the final row alone.
        const auto* chain_option=std::getenv("MFQ_MTP_HISTORY_CHAIN");
        if(prefix>1 && (!chain_option || chain_option[0]!='0')) {
          predictor_history_step(hidden.narrow(1,0,prefix),ids.narrow(1,0,prefix),
              positions.defined()?positions.narrow(-1,0,prefix):Tensor{},true);
          static bool reported=false;
          if(!reported) {
            reported=true;std::cerr<<"runtime_mtp_history_chain enabled=1 rows="<<prefix
                <<" scalar_projections=1\n";
          }
        } else for(int64_t row=0;row<prefix;++row)
            predictor_history_step(hidden.narrow(1,row,1),ids.narrow(1,row,1),
                positions.defined()?positions.narrow(-1,row,1):Tensor{});
        auto result=predictor_step(hidden.narrow(1,prefix,1),ids.narrow(1,prefix,1),
            positions.defined()?positions.narrow(-1,prefix,1):Tensor{});
#ifdef MFQ_MTP_VERIFY_HISTORY_TAIL
        if(expected_hidden.defined()) {
          const auto difference=[&](const MtpStep& reference) {
            double maximum=0;bool close=true;
            for(const auto& pair:std::array<std::pair<Tensor,Tensor>,2>{{
                    {result.sample_hidden,reference.sample_hidden},{result.chain_hidden,reference.chain_hidden}}}) {
              const auto actual=pair.first.to(mfq_tensor_backend::kCPU,mfq_tensor_backend::kFloat32).contiguous();
              const auto expected=pair.second.to(mfq_tensor_backend::kCPU,mfq_tensor_backend::kFloat32).contiguous();
              MFQ_RUNTIME_CHECK(actual.sizes()==expected.sizes(),"MTP history-tail shape differs from full predictor");
              for(int64_t i=0;i<actual.numel();++i) {
                const auto error=std::abs(double(actual.template data_ptr<float>()[i])-expected.template data_ptr<float>()[i]);
                maximum=std::max(maximum,error);
                close=close && std::isfinite(error) && error<=.003+.003*std::abs(expected.template data_ptr<float>()[i]);
              }
            }
            return std::make_pair(maximum,close);
          };
          const auto [batch_error,batch_close]=difference({expected_hidden,expected_chain});
          double scalar_error=0;
          if(!batch_close) {
            mtp.trim_cache_to(check_start);
            MtpStep scalar;
            for(int64_t row=0;row<ids.size(1);++row)
              scalar=mtp.step_positioned(target,hidden.narrow(1,row,1),ids.narrow(1,row,1),
                  positions.defined()?positions.narrow(-1,row,1):Tensor{});
            const auto [error,close]=difference(scalar);scalar_error=error;
            std::fprintf(stderr,"runtime_mtp_history_tail_scalar_check rows=%lld batch_max_error=%.8f scalar_max_error=%.8f close=%d\n",
                static_cast<long long>(ids.size(1)),batch_error,scalar_error,int(close));
            MFQ_RUNTIME_CHECK(close,"MTP history-tail differs from scalar predictor");
          }
          checked[ids.size(1)]=true;target_maintenance=true;
          std::fprintf(stderr,"runtime_mtp_history_tail_check rows=%lld batch_max_error=%.8f scalar_max_error=%.8f PASS\n",
              static_cast<long long>(ids.size(1)),batch_error,scalar_error);
        }
#endif
        return result;
      }
      if(model.reuse_decode_graphs && model.graph_registry && ids.size(1)>4 && ids.size(1)<=128) {
        MtpStep last;
        for(int64_t offset=0;offset<ids.size(1);offset+=4) {
          const auto rows=std::min<int64_t>(4,ids.size(1)-offset);
          auto part_positions=positions.defined()?positions.narrow(-1,offset,rows):Tensor{};
          if(offset+rows<ids.size(1))
            predictor_history_step(hidden.narrow(1,offset,rows),ids.narrow(1,offset,rows),part_positions);
          else last=predictor_step(hidden.narrow(1,offset,rows),ids.narrow(1,offset,rows),part_positions);
        }
        return last;
      }
    }
    auto result=positions.defined()
               ? mtp.step_positioned(target, std::move(hidden), std::move(ids),
                                     std::move(positions))
               : mtp.step(target, std::move(hidden), std::move(ids));
    const bool initialized=mtp.take_graph_initialization();
    target_maintenance=target_maintenance || initialized;
    projected_predictor_logits = result.sample_logits;
    projected_predictor_compact={};
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>)
      if(auto* predictor=dynamic_cast<mfq::cuda::qwen4_exp::Qwen4ExpMtp*>(&mtp);predictor && result.sample_logits.defined())
        projected_predictor_compact=predictor->graph_compact_topk();
    projected_predictor_greedy = result.greedy_token;
    projected_predictor_hidden = result.sample_logits.defined()
        ? result.sample_hidden.narrow(1, result.sample_hidden.size(1)-1, 1).contiguous() : Tensor{};
    return result;
  }
  void predictor_history_step(Tensor hidden, Tensor ids, Tensor positions,bool scalar_chain=false) {
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>) {
      if(auto* predictor=dynamic_cast<mfq::cuda::qwen4_exp::Qwen4ExpMtp*>(&mtp);
          predictor && model.reuse_decode_graphs && model.graph_registry && ids.size(1)<=128) {
        for(int64_t offset=0;offset<ids.size(1);offset+=4) {
          const auto rows=std::min<int64_t>(4,ids.size(1)-offset);
          predictor->history_step(target,hidden.narrow(1,offset,rows),ids.narrow(1,offset,rows),
              positions.defined()?positions.narrow(-1,offset,rows):Tensor{},scalar_chain);
        }
        target_maintenance=mtp.take_graph_initialization() || target_maintenance;
        return;
      }
    }
    (void)predictor_step(std::move(hidden),std::move(ids),std::move(positions));
  }
  int64_t predictor_prefill_chunk_size(int64_t requested) const {
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>)
      // The predictor retains its expert image across these compute tiles.
      return std::min<int64_t>(1024,requested);
    return std::min<int64_t>(512,requested);
  }
  bool cache_only_prompt_history() const {
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>) {
      const auto* option=std::getenv("MFQ_MTP_CACHE_ONLY_PRIME");
      const auto* predictor=dynamic_cast<const mfq::cuda::qwen4_exp::Qwen4ExpMtp*>(&mtp);
      return (!option || option[0]!='0') && predictor && predictor->layers.size()==1;
    }
    return false;
  }
  double begin_predictor_prefill(int64_t pairs) {
    const auto started=std::chrono::steady_clock::now();
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>) {
      if(cache_only_prompt_history())return 0;
      if(pairs<=128 || !model.reuse_decode_graphs || !model.graph_registry)return 0;
      prepare_moe_decode_residency(model.execution->moe_expert_cache);
      const auto rows=std::min<int64_t>(1024,pairs);
      const auto bytes=(std::size_t(1)<<30)+std::size_t(rows)*model.config.hidden*
          model.config.streams*sizeof(float)*8+moe_prefill_transfer_bytes(model.execution->moe_expert_cache,true);
      auto loan=std::make_unique<MoePrefillWorkspaceLoan>(model.execution->moe_expert_cache);
      if(loan->begin(bytes))predictor_loan=std::move(loan);
    }
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
  }
  double end_predictor_prefill() {
    if(predictor_loan){predictor_loan->finish();predictor_loan.reset();}
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>)
      if(model.restore_decode_workspace)model.restore_decode_workspace();
    // Every predictor tile has already synchronized its compute-end event.
    // Returning the expert loan is TTFT maintenance, not prompt computation.
    return 0;
  }
  double predictor_prefill(Tensor hidden, Tensor ids, Tensor positions) {
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>) {
      if(!cache_only_prompt_history() && model.reuse_decode_graphs && model.graph_registry && ids.size(1)<=128)
        end_predictor_prefill();
    }
    PrefillCudaTimer timer;
    bool prefill_finished=false;
    if(!cache_only_prompt_history())hidden=hidden.to(mfq_tensor_backend::kCUDA);
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>) {
      if(cache_only_prompt_history()) {
        auto& predictor=static_cast<mfq::cuda::qwen4_exp::Qwen4ExpMtp&>(mtp);
        MoePrefillWorkspaceLoan workspace(model.execution->moe_expert_cache);
        if(ids.size(1)>128)workspace.begin((std::size_t(64)<<20)+
            std::size_t(ids.size(1))*model.hidden_size()*model.hc_mult()*sizeof(float)*12);
        {
          auto resident_hidden=hidden.to(mfq_tensor_backend::kCUDA);
          (void)predictor.evaluate(target,resident_hidden,ids,0,true,positions,{},true);
        }
        MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
        prefill_finished=true;
        workspace.finish();
      } else if(model.reuse_decode_graphs && model.graph_registry && ids.size(1)<=128) {
        // The short tail uses decode graphs, whose expert addresses must again
        // contain weights. The final long compute tile has already retired.
        for(int64_t offset=0;offset<ids.size(1);offset+=4) {
          const auto rows=std::min<int64_t>(4,ids.size(1)-offset);
          predictor_history_step(hidden.narrow(1,offset,rows),ids.narrow(1,offset,rows),
              positions.defined()?positions.narrow(-1,offset,rows):Tensor{});
        }
      } else (void)predictor_step(std::move(hidden),std::move(ids),std::move(positions));
    } else (void)predictor_step(std::move(hidden),std::move(ids),std::move(positions));
    if(!prefill_finished)
      MFQ_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
    return timer.elapsed_ms();
  }
  int32_t sample_normal(Tensor logits, Tensor token_counts) {
    if(sampler().greedy() && !penalties) {
      if(projected_predictor_greedy>=0 && projected_predictor_logits.defined() &&
          logits.data_ptr()==projected_predictor_logits.data_ptr() && logits.numel()==projected_predictor_logits.numel())
        return projected_predictor_greedy;
      if(projected_target_greedy.size()==1 && projected_target_logits.defined() &&
          logits.data_ptr()==projected_target_logits.data_ptr() && logits.numel()==projected_target_logits.numel())
        return projected_target_greedy.front();
    }
    if (penalties)
      logits = logits.clone();
    return static_cast<int32_t>(
        mfq::cuda::sample_logits(sampler(), std::move(logits), token_counts)
            .template item<int64_t>());
  }
  std::vector<int32_t> greedy_tokens(Tensor logits) {
    MFQ_RUNTIME_CHECK(!penalties && sampler().greedy() && logits.dim() == 2,
                      "batched greedy MTP sampling geometry is invalid");
    if(!projected_target_greedy.empty() && projected_target_logits.defined() &&
        logits.data_ptr()==projected_target_logits.data_ptr() && logits.numel()==projected_target_logits.numel())
      return projected_target_greedy;
    auto host = sampler().ops().sample_greedy(logits.contiguous()).cpu().contiguous();
    const auto* tokens = host.template data_ptr<int64_t>();
    return std::vector<int32_t>(tokens, tokens + host.numel());
  }
  int32_t sample_constrained(Tensor logits, Tensor token_counts,
                             const MfqTokenConstraintPtr &constraint) {
    if(!constraint && sampler().greedy() && !penalties) return sample_normal(logits,token_counts);
    if (penalties)
      logits = logits.clone();
    auto sampler_constraint =
        constraint ? constraint->clone() : MfqTokenConstraintPtr{};
    return static_cast<int32_t>(
        mfq::cuda::sample_logits(sampler(), std::move(logits), token_counts,
                                 sampler_constraint)
            .template item<int64_t>());
  }
  std::vector<float> probabilities(Tensor logits, Tensor token_counts,
                                   const MfqSamplingParams &parameters) {
    logits = logits.contiguous().reshape({1, -1});
    if (penalties) {
      logits = logits.clone();
      sample_apply_penalties_cuda(
          logits, token_counts, parameters.presence_penalty,
          parameters.frequency_penalty, parameters.repetition_penalty);
    }
    auto host = logits.to(mfq_tensor_backend::kFloat32).cpu().contiguous();
    return mfq::engine::mtp::distribution(
        std::span<const float>(host.template data_ptr<float>(), host.numel()),
        parameters.temperature, parameters.top_k, parameters.top_p);
  }
  bool compact_radix_enabled(const Tensor& logits) const {
    const auto* option=std::getenv("MFQ_MTP_RADIX_TOPK");
    return option && option[0]=='1' && logits.size(0)<=32;
  }
  std::vector<mfq::engine::mtp::CompactDistribution>
  compact_radix_rows(Tensor logits,const MfqSamplingParams& parameters) {
    const auto host=sample_top_k_compact_cuda(logits,parameters.top_k).cpu().contiguous();
    return compact_rows_from_host(host,parameters);
  }
  std::vector<mfq::engine::mtp::CompactDistribution>
  compact_rows_from_host(const Tensor& host,const MfqSamplingParams& parameters) {
    const auto* packets=host.template data_ptr<int64_t>();
    std::array<float,64> values;
    std::array<int64_t,64> indices;
    std::vector<mfq::engine::mtp::CompactDistribution> result;
    result.reserve(std::size_t(host.size(0)));
    for(int64_t row=0;row<host.size(0);++row) {
      for(int i=0;i<parameters.top_k;++i) {
        const auto packet=uint64_t(packets[row*host.size(1)+i]);
        const uint32_t raw=uint32_t(packet>>32);
        std::memcpy(&values[i],&raw,sizeof(raw));indices[i]=uint32_t(packet);
      }
      result.push_back(mfq::engine::mtp::compact_distribution_from_topk(values.data(),indices.data(),
          int(parameters.top_k),parameters.temperature,parameters.top_p));
    }
    return result;
  }
  Tensor captured_compact(const Tensor& logits,const MfqSamplingParams& parameters) {
    for(const auto& pair:std::array<std::pair<Tensor,Tensor>,2>{{
        {projected_predictor_logits,projected_predictor_compact},{projected_target_logits,projected_target_compact}}})
      if(pair.first.defined() && pair.second.defined() && pair.first.data_ptr()==logits.data_ptr() &&
          pair.first.numel()==logits.numel() && pair.second.size(0)==logits.size(0)) {
        if(++graph_sampling_reuses==1)std::cerr<<"runtime_mtp_graph_sampling captured_topk=64 host_rows="<<logits.size(0)<<'\n';
#ifdef MFQ_MTP_VERIFY_GRAPH_TOPK
        static thread_local std::array<bool,33> checked{};
        if(!checked[logits.size(0)]) {
            auto expected=compact_radix_rows(logits,parameters);
            auto actual=compact_rows_from_host(pair.second,parameters);
            MFQ_RUNTIME_CHECK(expected.size()==actual.size(),"captured MTP top-k row count differs");
            for(std::size_t row=0;row<actual.size();++row) {
                MFQ_RUNTIME_CHECK(expected[row].tokens==actual[row].tokens,"captured MTP top-k token IDs differ");
                for(std::size_t i=0;i<actual[row].probabilities.size();++i)
                    MFQ_RUNTIME_CHECK(std::abs(actual[row].probabilities[i]-expected[row].probabilities[i])<=1e-6,
                        "captured MTP top-k probabilities differ");
            }
            checked[logits.size(0)]=true;target_maintenance=true;
            std::cerr<<"runtime_mtp_graph_sampling_check rows="<<logits.size(0)<<" PASS\n";
        }
#endif
        return pair.second;
      }
    return {};
  }
  mfq::engine::mtp::CompactDistribution
  compact_probabilities(Tensor logits, Tensor token_counts,
                        const MfqSamplingParams &parameters) {
    MFQ_RUNTIME_CHECK(parameters.top_k > 0 && parameters.top_k <= 64 &&
                          parameters.temperature > 0.0,
                      "compact CUDA MTP sampling requires top-k 1-64");
    logits = logits.contiguous().reshape({1, -1});
    if(!penalties)if(auto host=captured_compact(logits,parameters);host.defined())
        return std::move(compact_rows_from_host(host,parameters).front());
    if (penalties) {
      logits = logits.clone();
      sample_apply_penalties_cuda(
          logits, token_counts, parameters.presence_penalty,
          parameters.frequency_penalty, parameters.repetition_penalty);
    }
    if(compact_radix_enabled(logits))return std::move(compact_radix_rows(logits,parameters).front());
    auto selected =
        mfq_tensor_backend::topk(logits, parameters.top_k, -1, true, true);
    auto values =
        std::get<0>(selected)
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
            .contiguous()
            .reshape({-1});
    auto indices = std::get<1>(selected)
                       .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                       .contiguous()
                       .reshape({-1});
    return mfq::engine::mtp::compact_distribution_from_topk(
        values.template data_ptr<float>(), indices.template data_ptr<int64_t>(),
        static_cast<int>(values.numel()), parameters.temperature,
        parameters.top_p);
  }
  std::vector<mfq::engine::mtp::CompactDistribution>
  compact_probability_rows(Tensor logits, const MfqSamplingParams &parameters) {
    MFQ_RUNTIME_CHECK(!penalties && logits.dim() == 2 && parameters.top_k > 0 &&
                          parameters.top_k <= 64 &&
                          parameters.temperature > 0.0,
                      "batched compact CUDA MTP sampling geometry is invalid");
    logits = logits.contiguous();
    if(auto host=captured_compact(logits,parameters);host.defined())return compact_rows_from_host(host,parameters);
    if(compact_radix_enabled(logits))return compact_radix_rows(logits,parameters);
    auto selected =
        mfq_tensor_backend::topk(logits, parameters.top_k, -1, true, true);
    auto values =
        std::get<0>(selected)
            .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kFloat32)
            .contiguous();
    auto indices = std::get<1>(selected)
                       .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                       .contiguous();
    const int64_t rows = values.size(0);
    const int64_t columns = values.size(1);
    const float *value_data = values.template data_ptr<float>();
    const int64_t *index_data = indices.template data_ptr<int64_t>();
    std::vector<mfq::engine::mtp::CompactDistribution> result;
    result.reserve(static_cast<size_t>(rows));
    for (int64_t row = 0; row < rows; ++row) {
      result.push_back(mfq::engine::mtp::compact_distribution_from_topk(
          value_data + row * columns, index_data + row * columns,
          static_cast<int>(columns), parameters.temperature, parameters.top_p));
    }
    return result;
  }
  Tensor logits_for(Tensor normalized) {
    const auto matches = [&](const Tensor& hidden) {
        return hidden.defined() && hidden.data_ptr()==normalized.data_ptr() && hidden.sizes()==normalized.sizes();
    };
    if(matches(projected_predictor_hidden)) return projected_predictor_logits;
    if(matches(projected_target_hidden)) return projected_target_logits;
    return model.logits_from_hidden(
        (mtp.preserve_output_dtype()
             ? normalized
             : normalized.to(mfq_tensor_backend::kFloat16))
            .contiguous());
  }
};
template <typename Model>
Generation
run_mtp_generation(Model &model, MtpModule &mtp, InferenceRequest &request,
                   InferenceOutput &output, int64_t prefill_chunk_size,
                   const CudaPreparedPrompt *prepared,
                   std::size_t reused_tokens,
                   mfq_tensor_backend::Tensor restored_last_hidden,
                   mfq_tensor_backend::Tensor *session_last_hidden,
                   std::size_t stable_prefix_tokens) {
  auto logical_chunk=prefill_chunk_size;
  if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>) {
    logical_chunk=std::min<int64_t>(logical_chunk,8192);
    if(prefill_chunk_size>8 && reused_tokens==0 && (!prepared || !prepared->transformed()) &&
        mfq::cuda::qwen4_layer_major_prefill_eligible(model,request.prompt.size()))
      logical_chunk=request.prompt.size();
  }
  if constexpr(std::is_same_v<Model,mfq::cuda::DeepseekV41CausalLm>) {
    // Retained small-M continuations must bound decoder replay as well as
    // encoder tiles; merging the suffix would re-enter the large PP pipeline.
    const bool retained_chunks=reused_tokens>0 && prefill_chunk_size<=8;
    if(!retained_chunks && (!prepared || !prepared->transformed()))logical_chunk=request.prompt.size();
  }
  return mfq::engine::speculative_sequence(
      CudaMtpOps<Model>{model, mtp, prepared, prefill_chunk_size, &output}, request, output,
      logical_chunk, reused_tokens, std::move(restored_last_hidden),
      session_last_hidden,stable_prefix_tokens);
}
#define MFQ_INSTANTIATE_MTP(MODEL)                                             \
  template Generation run_mtp_generation<MODEL>(                               \
      MODEL &, MtpModule &, InferenceRequest &, InferenceOutput &, int64_t,    \
      const CudaPreparedPrompt *, std::size_t, mfq_tensor_backend::Tensor,     \
      mfq_tensor_backend::Tensor *, std::size_t)

MFQ_INSTANTIATE_MTP(mfq::cuda::Qwen35CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::MiniCPMO45CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::MiniCPMOTtsCausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::Gemma4CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::GlmDsaCausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::Glm5CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::Qwen4CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::DeepseekV4CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::DeepseekV41CausalLm);
MFQ_INSTANTIATE_MTP(mfq::cuda::NemotronHCausalLm);

#undef MFQ_INSTANTIATE_MTP
