#include "runtime/mtp_compact_sampling.h"
#include "models/nemotron_h/ops.h"
#include "generation.h"
#include "speculative_sequence.h"

#include "hip_execution.h"
#include "hip_sampling.h"
#include "generation_policy.h"
#include "inference.h"
#include "mfq_hip_sampling_ops.h"
#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/qwen35/ops.h"
#include "models/qwen4_exp/ops.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <tuple>
#include <type_traits>
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
  int64_t prefill_compute_chunk=1;
  mfq::engine::InferenceOutput* output=nullptr;
  const mfq_tensor_backend::TensorOptions options =
      mfq_tensor_backend::TensorOptions()
          .device(mfq_tensor_backend::kCUDA)
          .dtype(mfq_tensor_backend::kInt64);
  Tensor input_ids, counts;
  std::optional<mfq::cuda::Sampler> sampler_;
  bool penalties = false;
  bool target_maintenance = false;
  Tensor projected_predictor_hidden,projected_predictor_logits;
  int32_t projected_predictor_greedy=-1;
  bool take_target_maintenance() {return std::exchange(target_maintenance,false);}

  CudaMtpOps(Model &model, MtpModule &mtp, const CudaPreparedPrompt *prepared)
      : model(model), mtp(mtp), prepared(prepared), target(model),
        transformed_prompt(prepared && prepared->transformed()),
        position_axes(transformed_prompt && prepared->positions.defined() && prepared->positions.dim() == 2 &&
                              prepared->positions.size(0) == 3
                          ? 3
                          : 1) {}
  void initialize(const std::vector<int64_t> &prompt,
                  const MfqSamplingParams &sampling) {
    input_ids = ids_for(prompt);
    sampler_.emplace(sampling);
    penalties = sampler().has_penalties();
    if (penalties) {
      counts = mfq::cuda::SamplingOps::token_counts(input_ids, model.vocab_size());
    }
  }
  auto &sampler() { return *sampler_; }
  static void add_counts(Tensor counts, const Tensor &ids) {
    sample_token_counts_add_cuda(counts, ids);
  }
  static bool defined(const Tensor &value) { return value.defined(); }
  static int64_t size(const Tensor &value, int axis) {
    return value.size(axis);
  }
  static int rank(const Tensor &value) { return value.dim(); }
  static Tensor clone(const Tensor &value) { return value.clone(); }
  static Tensor slice(const Tensor &value, int axis, int64_t offset,
                      int64_t length) {
    return value.narrow(axis, offset, length).contiguous();
  }
  static Tensor reshape(Tensor value, std::initializer_list<int64_t> shape) {
    return value.reshape(shape);
  }
  static Tensor concatenate(const std::vector<Tensor> &values, int axis) {
    return mfq_tensor_backend::cat(values, axis).contiguous();
  }
  Tensor target_forward(Tensor ids, Tensor *raw = nullptr,
                        int64_t confirmed = 0) {
    if constexpr(std::is_same_v<Model,mfq::cuda::Qwen4CausalLm>) {
      if(!transformed_prompt) {
        bool initialized=false;
        if(auto hidden=mfq::cuda::qwen4_decode_graph_hidden(model,ids,raw,confirmed,&initialized)) {
          target_maintenance=target_maintenance || initialized;
          return std::move(*hidden);
        }
      }
    }
    return model.hidden_forward(ids, mfq_nullopt, mfq_nullopt, nullptr,
                                mfq_nullopt, raw, confirmed);
  }
  Tensor target_suffix(Tensor ids, Tensor *raw) {
    return model.hidden_forward_speculative_suffix(ids, raw);
  }
  struct PrefillResult {
    Tensor hidden, raw;
    double elapsed;
  };
  double blockwise_prompt_context(Tensor hidden,int64_t position,bool seed) {
    PrefillCudaTimer timer;
    if(seed)mtp.seed_target_context(std::move(hidden),position);
    else {const auto start=position-hidden.size(1);mtp.append_target_context(std::move(hidden),start);}
    MFQ_CUDA_CHECK(hipEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
    return timer.elapsed_ms();
  }
  PrefillResult prefill(mfq::engine::PrefillChunk chunk) {
    PrefillCudaTimer timer;
    Tensor raw;
    auto ids = input_ids.narrow(1, chunk.offset, chunk.count).contiguous();
    if constexpr(std::is_same_v<Model,mfq::cuda::DeepseekV41CausalLm>) {
      if(!transformed_prompt) {
        auto hidden=mfq::cuda::deepseek_v41_prefill_hidden(model,ids,prefill_compute_chunk,
            [&] {return output && output->stopped();},&raw);
        MFQ_CUDA_CHECK(hipEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
        return {std::move(hidden),std::move(raw),timer.elapsed_ms()};
      }
    }
    struct ImageMaskScope {
      Model& model;
      ImageMaskScope(Model& model,const Tensor& mask):model(model) {
        if constexpr(std::is_same_v<Model,mfq::cuda::DeepseekV41CausalLm>)model.adapter_set_image_mask(mask);
      }
      ~ImageMaskScope() {
        if constexpr(std::is_same_v<Model,mfq::cuda::DeepseekV41CausalLm>)model.adapter_set_image_mask({});
      }
    } participation(model,transformed_prompt && prepared->image_mask.defined()?
        prepared->image_mask.narrow(1,chunk.offset,chunk.count).contiguous():Tensor{});
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
    MFQ_CUDA_CHECK(
        hipEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
    return {std::move(hidden), std::move(raw), timer.elapsed_ms()};
  }
  Tensor ids_for(std::vector<int64_t> tokens) {
    return mfq_tensor_backend::tensor(tokens, options)
        .reshape({1, -1})
        .contiguous();
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
    auto result=positions.defined()?mtp.step_positioned(target,std::move(hidden),std::move(ids),std::move(positions)):
        mtp.step(target,std::move(hidden),std::move(ids));
    target_maintenance=mtp.take_graph_initialization() || target_maintenance;
    projected_predictor_logits=result.sample_logits;projected_predictor_greedy=result.greedy_token;
    projected_predictor_hidden=result.sample_logits.defined()?
        result.sample_hidden.narrow(1,result.sample_hidden.size(1)-1,1).contiguous():Tensor{};
    return result;
  }
  int32_t sample_normal(Tensor logits, Tensor token_counts) {
    if(sampler().greedy() && !penalties && projected_predictor_greedy>=0 && projected_predictor_logits.defined() &&
        logits.data_ptr()==projected_predictor_logits.data_ptr() && logits.numel()==projected_predictor_logits.numel())
        return projected_predictor_greedy;
    if (penalties)
      logits = logits.clone();
    return static_cast<int32_t>(
        mfq::cuda::sample_logits(sampler(), std::move(logits), token_counts)
            .template item<int64_t>());
  }
  std::vector<int32_t> greedy_tokens(Tensor logits) {
    auto host=sample_greedy_cuda(logits.contiguous()).to(mfq_tensor_backend::kCPU,mfq_tensor_backend::kInt64).contiguous();
    const auto* data=host.template data_ptr<int64_t>();
    return std::vector<int32_t>(data,data+host.numel());
  }
  int32_t sample_constrained(Tensor logits, Tensor token_counts,
                             const MfqTokenConstraintPtr &constraint) {
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
  mfq::engine::mtp::CompactDistribution
  compact_probabilities(Tensor logits, Tensor token_counts,
                        const MfqSamplingParams &parameters) {
    MFQ_RUNTIME_CHECK(parameters.top_k > 0 && parameters.top_k <= 64 &&
                          parameters.temperature > 0.0,
                      "compact CUDA MTP sampling requires top-k 1-64");
    logits = logits.contiguous().reshape({1, -1});
    if (penalties) {
      logits = logits.clone();
      sample_apply_penalties_cuda(
          logits, token_counts, parameters.presence_penalty,
          parameters.frequency_penalty, parameters.repetition_penalty);
    }
    return std::move(mfq::cuda::mtp_sampling::rows(logits,parameters,
        mfq::cuda::mtp_sampling::use_packets(logits)).front());
  }
  std::vector<mfq::engine::mtp::CompactDistribution>
  compact_probability_rows(Tensor logits, const MfqSamplingParams &parameters) {
    MFQ_RUNTIME_CHECK(!penalties && logits.dim() == 2 && parameters.top_k > 0 &&
                          parameters.top_k <= 64 &&
                          parameters.temperature > 0.0,
                      "batched compact CUDA MTP sampling geometry is invalid");
    logits = logits.contiguous();
    return mfq::cuda::mtp_sampling::rows(logits,parameters,mfq::cuda::mtp_sampling::use_packets(logits));
  }
  Tensor logits_for(Tensor normalized) {
    if(projected_predictor_hidden.defined() && normalized.data_ptr()==projected_predictor_hidden.data_ptr() &&
        normalized.sizes()==projected_predictor_hidden.sizes())return projected_predictor_logits;
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
                   mfq_tensor_backend::Tensor *session_last_hidden) {
  auto ops=CudaMtpOps<Model>{model,mtp,prepared};
  ops.prefill_compute_chunk=prefill_chunk_size;ops.output=&output;
  auto logical_chunk=prefill_chunk_size;
  if constexpr(std::is_same_v<Model,mfq::cuda::DeepseekV41CausalLm>)
    if(!prepared || !prepared->transformed())logical_chunk=request.prompt.size();
  return mfq::engine::speculative_sequence(std::move(ops),request,output,
      logical_chunk,reused_tokens,std::move(restored_last_hidden),session_last_hidden);
}
#define MFQ_INSTANTIATE_MTP(MODEL)                                             \
  template Generation run_mtp_generation<MODEL>(                               \
      MODEL &, MtpModule &, InferenceRequest &, InferenceOutput &, int64_t,    \
      const CudaPreparedPrompt *, std::size_t, mfq_tensor_backend::Tensor,     \
      mfq_tensor_backend::Tensor *)

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
