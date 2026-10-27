#pragma once
#include "mfq_hip_norm_ops.h"
#include "models/common/transformer_layer.h"
#include "models/deepseek_v41/causal_lm.h"

#include "core/causal_model.h"
#include "../deepseek_v4/ops.h"
#include "dspark.h"
#include "engram.h"
#include "mfq/kernels/hip/deepseek_v41.h"
#include "mfq/kernels/hip/deepseek_v4_attention.h"
#include "mfq/kernels/hip/deepseek_v4_fp4.h"
#include "mfq/kernels/hip/deepseek_v41_index.h"
#include "mfq/kernels/hip/deepseek_v4_hc.h"
#include "models/deepseek_v41/config.h"
#include "core/ffn.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda::deepseek_v41_runtime {

using Tensor = mfq_tensor_backend::Tensor;
using CommonConfig = mfq::models::deepseek_v41::Config;

struct SharedState {
    CommonConfig config;
    Tensor compressed_kv;
    Tensor index_k;
    Tensor topk;
    Tensor candidate_blocks;
    Tensor previous_pre;
    Tensor attention_meta;
    std::unique_ptr<EngramHashState> engram_hash;
    EngramHashBatch engram_hashes;
    std::vector<Engram*> engram_prefetch;
    std::vector<Tensor> dspark_target_hiddens;
    bool capture_dspark_targets = false;
    Tensor image_mask, text_rows, routing_mask;
    std::vector<int64_t> text_row_ids;
    std::int64_t compressed_length = 0;
    std::int64_t ratio = 0;
    std::int64_t cache_position = -1;
    std::int64_t batch = 0;
    std::int64_t tokens = 0;
    std::int64_t next_layer = 0;

    void set_image_mask(Tensor mask) {
        if(!mask.defined()) {image_mask={};text_rows={};routing_mask={};text_row_ids.clear();return;}
        MFQ_RUNTIME_CHECK(mask.is_cpu() && mask.dim()==2 && mask.is_contiguous() &&
                          mask.scalar_type()==mfq_tensor_backend::kUInt8,
                          "DeepSeek-V4.1 image mask must be contiguous CPU uint8 [B,T]");
        std::vector<int64_t> images;text_row_ids.clear();
        const auto* flags=mask.data_ptr<uint8_t>();
        for(int64_t i=0;i<mask.numel();++i)(flags[i]?images:text_row_ids).push_back(i);
        if(images.empty()){set_image_mask({});return;}
        auto options=mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
            .dtype(mfq_tensor_backend::kInt64);
        auto text=mfq_tensor_backend::tensor(text_row_ids,options);
        auto routing=mask.reshape({-1,1}).to(mfq_tensor_backend::kCUDA).to(mfq_tensor_backend::kBool);
        image_mask=std::move(mask);text_rows=std::move(text);routing_mask=std::move(routing);
    }

    void begin_forward(bool capture, std::size_t target_count) {
        capture_dspark_targets = capture;
        dspark_target_hiddens.clear();
        if (capture)
            dspark_target_hiddens.resize(target_count);
    }

    void capture_dspark_target(std::int64_t layer, const CommonConfig &config,
                               const Tensor &hidden) {
        if (!capture_dspark_targets)
            return;
        const auto found = std::find(config.dspark_target_layer_ids.begin(),
                                     config.dspark_target_layer_ids.end(), layer);
        if (found == config.dspark_target_layer_ids.end())
            return;
        const auto slot =
            static_cast<std::size_t>(std::distance(config.dspark_target_layer_ids.begin(), found));
        MFQ_RUNTIME_CHECK(slot < dspark_target_hiddens.size() && hidden.dim() == 4 &&
                              hidden.size(2) == config.hc_mult && hidden.size(3) == config.hidden,
                          "DeepSeek-V4.1 DSpark target geometry mismatch");
        dspark_target_hiddens[slot] = hidden.mean(2).contiguous();
    }

    Tensor dspark_target_hidden() const {
        MFQ_RUNTIME_CHECK(capture_dspark_targets && !dspark_target_hiddens.empty() &&
                              std::all_of(dspark_target_hiddens.begin(),
                                          dspark_target_hiddens.end(),
                                          [](const Tensor &value) { return value.defined(); }),
                          "DeepSeek-V4.1 DSpark target layers were not fully captured");
        return dspark_target_hiddens.size() == 1
                   ? dspark_target_hiddens.front()
                   : mfq_tensor_backend::cat(dspark_target_hiddens, -1).contiguous();
    }

    void begin_speculative() {
        MFQ_RUNTIME_CHECK(engram_hash, "DeepSeek-V4.1 Engram state is unavailable");
        engram_hash->begin_speculative();
    }

    void commit_speculative() noexcept {
        if (engram_hash)
            engram_hash->commit_speculative();
    }

    void rollback_speculative(int64_t position) {
        MFQ_RUNTIME_CHECK(engram_hash, "DeepSeek-V4.1 Engram state is unavailable");
        engram_hash->rollback_speculative(position);
    }

    void reset_session(std::int64_t current_batch) {
        compressed_kv = Tensor();
        index_k = Tensor();
        topk = Tensor();
        candidate_blocks = Tensor();
        previous_pre = Tensor();
        engram_hashes = {};
        dspark_target_hiddens.clear();
        capture_dspark_targets = false;
        compressed_length = 0;
        ratio = 0;
        cache_position = -1;
        batch = 0;
        tokens = 0;
        next_layer = 0;
        MFQ_RUNTIME_CHECK(engram_hash && current_batch > 0 &&
                              current_batch <= std::numeric_limits<int>::max(),
                          "DeepSeek-V4.1 Engram state is unavailable");
        engram_hash->reset(static_cast<int>(current_batch));
    }

    void enter_layer(std::int64_t layer, std::int64_t layer_count, std::int64_t current_batch,
                     std::int64_t current_tokens, std::int64_t current_position,
                     const mfq_tensor_backend::Device &device, const Tensor &token_ids) {
        if (layer == 0) {
            compressed_kv = Tensor();
            index_k = Tensor();
            topk = Tensor();
            candidate_blocks = Tensor();
            compressed_length = 0;
            ratio = 0;
            cache_position = current_position;
            batch = current_batch;
            tokens = current_tokens;
            next_layer = 0;
            previous_pre =
                mfq_tensor_backend::zeros({current_batch, current_tokens, 4},
                                          mfq_tensor_backend::TensorOptions().device(device).dtype(
                                              mfq_tensor_backend::kFloat32));
            previous_pre.narrow(2, 0, 1).fill_(1.0);
            MFQ_RUNTIME_CHECK(engram_hash && token_ids.dim() == 2 &&
                                  token_ids.size(0) == current_batch &&
                                  token_ids.size(1) == current_tokens,
                              "DeepSeek-V4.1 Engram token state is unavailable");
            MFQ_RUNTIME_CHECK(!image_mask.defined() || image_mask.sizes()==token_ids.sizes(),
                              "DeepSeek-V4.1 image participation shape mismatch");
            engram_hashes = engram_hash->forward(token_ids, current_position,
                image_mask.defined()?image_mask.data_ptr<uint8_t>():nullptr);
        }
        MFQ_RUNTIME_CHECK(layer == next_layer && layer >= 0 && layer < layer_count &&
                              current_batch == batch && current_tokens == tokens &&
                              current_position == cache_position && previous_pre.defined(),
                              "DeepSeek-V4.1 layer execution order changed");
        // Keep current/next Engram reads ahead of attention without launching
        // every layer's SSD scan together. During CED encoder prefill, decoder
        // tables will later consume only the bounded replay's hash rows.
        const auto prefetch_end=current_tokens>config.sliding_window && layer<config.causal_encoder_layers
            ?config.causal_encoder_layers:layer_count;
        int issued=0;
        for(const auto* memory:engram_prefetch) {
            if(memory->layer_id()<layer || memory->layer_id()>=prefetch_end)continue;
            memory->prefetch(engram_hashes,image_mask.defined()?&text_row_ids:nullptr);
            if(++issued==2)break;
        }
        ++next_layer;
        if (!attention_meta.defined() || attention_meta.device() != device) {
            attention_meta = mfq_tensor_backend::empty(
                {8 * 1024 * 1024}, mfq_tensor_backend::TensorOptions().device(device).dtype(
                                       mfq_tensor_backend::kFloat32));
        }
    }

    Tensor final_collapse(const Tensor &hidden, std::int64_t layer_count) const {
        MFQ_RUNTIME_CHECK(hidden.dim() == 4 && hidden.size(2) == 4 && previous_pre.defined() &&
                              previous_pre.sizes() == hidden.sizes().slice(0, 3) &&
                              next_layer == layer_count,
                          "DeepSeek-V4.1 final Mega-mHC state is incomplete");
        return (previous_pre.to(mfq_tensor_backend::kFloat32).unsqueeze(-1) *
                hidden.to(mfq_tensor_backend::kFloat32))
            .sum(2)
            .to(mfq_tensor_backend::kFloat16)
            .contiguous();
    }
};

struct AttentionState {
    Tensor local_kv;
    Tensor compressed_kv;
    Tensor index_k;
    Tensor partial_kv;
    Tensor partial_score;
    std::int64_t position = 0;
    std::int64_t compressed_length = 0;
    std::int64_t partial_length = 0;
};

inline Tensor weighted_rms(const Tensor &input, const Tensor &weight, double epsilon) {
    const auto shape = input.sizes().vec();
    MFQ_RUNTIME_CHECK(!shape.empty() && weight.numel() == shape.back(),
                      "DeepSeek-V4.1 weighted RMS geometry mismatch");
    if(input.scalar_type()==mfq_tensor_backend::kFloat16 && shape.back()<4096)
        return rms_norm_f16_cuda(input.reshape({-1,shape.back()}).contiguous(),weight,epsilon,0.)
            .reshape(shape);
    return rms_norm_cuda(input.reshape({-1, shape.back()}).to(mfq_tensor_backend::kFloat32), weight,
                         epsilon)
        .reshape(shape)
        .to(mfq_tensor_backend::kFloat16)
        .contiguous();
}

inline Tensor rotate_token_major_tail(Tensor input, const Tensor &positions,
                                      const Dsv4RopeTable &rope, bool inverse = false) {
    MFQ_RUNTIME_CHECK(input.dim() == 3, "DeepSeek-V4.1 RoPE input must be [B,T,D]");
    return dsv4_rotate_rope_tail(input.unsqueeze(1), positions, rope, inverse)
        .squeeze(1)
        .contiguous();
}

// Projection view -> token-major F16 index query, including FP4 rounding.
inline Tensor index_query_transform(Tensor query, const Tensor &positions,
                                    const Dsv4RopeTable &rope) {
    const auto dtype = query.scalar_type();
    if (query.is_cuda() && query.dim() == 4 && query.size(1) == 32 &&
        query.size(3) == 128 && query.stride(3) == 1 && query.numel() > 0 &&
        (dtype == mfq_tensor_backend::kFloat16 || dtype == mfq_tensor_backend::kFloat32) &&
        query.stride(0) % 2 == 0 && query.stride(1) % 2 == 0 && query.stride(2) % 2 == 0 &&
        reinterpret_cast<std::uintptr_t>(query.data_ptr()) %
            (dtype == mfq_tensor_backend::kFloat16 ? 4 : 8) == 0) {
        return dsv4_index_query_rope_fp4_cuda(
            query, positions.contiguous().to(mfq_tensor_backend::kCUDA,
                                             mfq_tensor_backend::kInt64),
            rope.cos, rope.sin);
    }
    auto rotated = dsv4_rotate_rope_tail(query, positions, rope, false, false, true);
    return dsv4_fp4_sim_cuda(rotated.to(mfq_tensor_backend::kFloat16).contiguous());
}

inline Tensor empty_topk(std::int64_t batch, std::int64_t tokens,
                         const mfq_tensor_backend::Device &device) {
    return mfq_tensor_backend::empty(
        {batch, tokens, 0},
        mfq_tensor_backend::TensorOptions().device(device).dtype(mfq_tensor_backend::kInt32));
}

inline Tensor normalize_topk(Tensor selected, const Tensor &values, std::int64_t pool_length) {
    selected = selected.to(mfq_tensor_backend::kInt64).contiguous();
    if(selected.size(-1)>0 && selected.size(-1)<=512 && pool_length>=0 && pool_length<=INT_MAX &&
       (values.scalar_type()==mfq_tensor_backend::kFloat16 || values.scalar_type()==mfq_tensor_backend::kFloat32))
        return deepseek_v41_normalize_topk_cuda(selected,values.contiguous(),pool_length);
    auto valid = (selected >= 0.0) & (selected < static_cast<double>(pool_length));
    valid = valid & mfq_tensor_backend::isneginf(values).logical_not();
    auto masked = mfq_tensor_backend::where(valid, selected, static_cast<double>(pool_length));
    auto ordered = std::get<0>(mfq_tensor_backend::sort(masked, -1, false));
    return mfq_tensor_backend::where(ordered < static_cast<double>(pool_length), ordered, -1.0)
        .to(mfq_tensor_backend::kInt32)
        .contiguous();
}

inline Tensor select_topk_generic(const Tensor &scores, std::int64_t requested, std::int64_t pool_length,
                          const std::optional<Tensor> &source_indices = std::nullopt) {
    const auto width = scores.size(-1);
    const auto count = std::min(requested, width);
    if (count <= 0) {
        return empty_topk(scores.size(0), scores.size(1), scores.device());
    }
    Tensor relative;
    Tensor values;
    if (count == width) {
        relative =
            mfq_tensor_backend::arange(width, scores.options().dtype(mfq_tensor_backend::kInt64))
                .reshape({1, 1, width})
                .expand({scores.size(0), scores.size(1), width})
                .contiguous();
        values = scores;
    } else if (count == 512 && scores.scalar_type() == mfq_tensor_backend::kFloat16) {
        relative =
            dsv4_topk512_cuda(scores.contiguous()).to(mfq_tensor_backend::kInt64).contiguous();
        values = scores.gather(-1, relative);
    } else {
        auto result = mfq_tensor_backend::topk(scores, count, -1, true, false);
        values = std::get<0>(result);
        relative = std::get<1>(result);
    }
    auto selected = source_indices
                        ? source_indices->to(mfq_tensor_backend::kInt64).gather(-1, relative)
                        : relative;
    return normalize_topk(selected, values, pool_length);
}

inline Tensor select_topk(const Tensor &scores, std::int64_t requested, std::int64_t pool_length,
                          const std::optional<Tensor> &source_indices = std::nullopt) {
    const bool source_ok = !source_indices ||
        (source_indices->device() == scores.device() && source_indices->is_contiguous() &&
         source_indices->scalar_type() == mfq_tensor_backend::kInt64 &&
         source_indices->sizes() == scores.sizes());
    if (requested == 512 && scores.is_cuda() && scores.is_contiguous() && scores.dim() == 3 &&
        scores.scalar_type() == mfq_tensor_backend::kFloat16 && scores.size(2) > 512 &&
        pool_length >= 0 && pool_length <= INT_MAX && source_ok) {
        auto relative = dsv4_topk512_cuda(scores);
        return deepseek_v41_finalize_index_topk_cuda(
            scores, relative, source_indices ? *source_indices : Tensor{}, pool_length);
    }
    return select_topk_generic(scores, requested, pool_length, source_indices);
}

inline Tensor candidate_blocks_from_scores_generic(const Tensor &scores, std::int64_t ratio,
                                           std::int64_t position, std::int64_t requested_blocks,
                                           std::int64_t block_size) {
    const auto batch = scores.size(0);
    const auto tokens = scores.size(1);
    const auto width = scores.size(2);
    const auto blocks = (width + block_size - 1) / block_size;
    const auto padded = blocks * block_size;
    Tensor padded_scores = scores;
    if (padded != width) {
        padded_scores = mfq_tensor_backend::cat(
            {scores,
             mfq_tensor_backend::full({batch, tokens, padded - width},
                                      -std::numeric_limits<float>::infinity(), scores.options())},
            -1);
    }
    auto block_scores = padded_scores.reshape({batch, tokens, blocks, block_size}).amax(-1);
    auto int_options = scores.options().dtype(mfq_tensor_backend::kInt64);
    auto visible = (mfq_tensor_backend::arange(position + 1, position + tokens + 1, int_options)
                        .reshape({1, tokens, 1}) /
                    static_cast<double>(ratio));
    auto last = mfq_tensor_backend::where(visible > 0.0,
                                          (visible - 1.0) / static_cast<double>(block_size), -1.0);
    auto block_ids = mfq_tensor_backend::arange(blocks, int_options).reshape({1, 1, blocks});
    block_scores = mfq_tensor_backend::where(block_ids == last,
                                             std::numeric_limits<float>::infinity(), block_scores);
    const auto count = std::min(requested_blocks, blocks);
    auto result = mfq_tensor_backend::topk(block_scores, count, -1, true, false);
    auto values = std::get<0>(result);
    auto selected = std::get<1>(result);
    return mfq_tensor_backend::where(mfq_tensor_backend::isneginf(values).logical_not(), selected,
                                     -1.0)
        .to(mfq_tensor_backend::kInt32)
        .contiguous();
}

inline Tensor candidate_blocks_from_scores(const Tensor &scores, std::int64_t ratio,
                                           std::int64_t position, std::int64_t requested_blocks,
                                           std::int64_t block_size) {
    if (block_size == 8 && scores.is_cuda() && scores.is_contiguous() && scores.dim() == 3 &&
        scores.scalar_type() == mfq_tensor_backend::kFloat16 && scores.size(2) > 0 &&
        scores.size(2) <= INT_MAX && scores.size(0) > 0 && scores.size(0) <= 65535 &&
        scores.size(1) > 0 && scores.size(1) <= 65535 && ratio > 0 && ratio <= INT_MAX &&
        position >= 0 && position <= INT_MAX - scores.size(1) && requested_blocks >= 0) {
        return deepseek_v41_candidate_blocks8_cuda(scores, ratio, position, requested_blocks);
    }
    return candidate_blocks_from_scores_generic(scores, ratio, position, requested_blocks, block_size);
}

inline Tensor candidate_positions_generic(const Tensor &blocks, std::int64_t block_size,
                                  std::int64_t pool_length) {
    const auto count = blocks.size(-1);
    auto block64 = blocks.to(mfq_tensor_backend::kInt64);
    auto within =
        mfq_tensor_backend::arange(block_size, blocks.options().dtype(mfq_tensor_backend::kInt64))
            .reshape({1, 1, 1, block_size});
    auto expanded = (block64.unsqueeze(-1) * static_cast<double>(block_size) + within)
                        .reshape({blocks.size(0), blocks.size(1), count * block_size});
    auto valid = (expanded >= 0.0) & (expanded < static_cast<double>(pool_length));
    return mfq_tensor_backend::where(valid, expanded, -1.0)
        .to(mfq_tensor_backend::kInt64)
        .contiguous();
}

inline Tensor candidate_positions(const Tensor &blocks, std::int64_t block_size,
                                  std::int64_t pool_length) {
    if (block_size == 8 && blocks.is_cuda() && blocks.is_contiguous() &&
        blocks.dim() == 3 && blocks.scalar_type() == mfq_tensor_backend::kInt32 &&
        pool_length >= 0 && pool_length <= INT_MAX) {
        return deepseek_v41_candidate_positions8_cuda(blocks, pool_length);
    }
    return candidate_positions_generic(blocks, block_size, pool_length);
}

struct MhcResult {
    Tensor branch;
    Tensor next_pre;
    Tensor post;
    Tensor combination;
};

struct Block final : ::Block {
    CommonConfig config;
    std::int64_t layer = -1;
    std::int64_t ratio = 0;
    std::int64_t max_context = 0;
    std::shared_ptr<SharedState> shared;
    AttentionState state;
    Tensor current_ids;

    Tensor attention_norm;
    Tensor mlp_norm;
    Tensor query_a_norm;
    Tensor key_value_norm;
    Tensor sinks;
    Tensor attention_mhc_function;
    Tensor attention_mhc_scale;
    Tensor attention_mhc_base;
    Tensor mlp_mhc_function;
    Tensor mlp_mhc_scale;
    Tensor mlp_mhc_base;
    Tensor compressor_norm;
    Tensor index_key_norm;

    QuantLinear query_a;
    QuantLinear query_b;
    QuantLinear key_value;
    QuantLinear output_a;
    QuantLinear output_b;
    QuantLinear compressor_key_value;
    QuantLinear compressor_gate;
    QuantLinear index_query;
    QuantLinear index_key;
    QuantLinear index_score;
    FFN mlp;
    Tensor vision_router_bias;
    Dsv4RopeTable rope;
    std::unique_ptr<Engram> engram;
    Tensor speculative_local_slots;
    Tensor speculative_slot_ids;
    Tensor speculative_partial_kv;
    Tensor speculative_partial_score;
    Tensor speculative_projected_kv;
    Tensor speculative_projected_score;
    std::int64_t speculative_position = -1;
    std::int64_t speculative_compressed_length = 0;
    std::int64_t speculative_partial_length = 0;
    bool speculative_pending = false;
    std::int64_t prefilled_global_end = -1;
    bool replay_local_window = false;

    bool kv_source() const noexcept { return config.is_kv_source(layer); }

    bool index_source() const noexcept { return config.is_index_source(layer); }

    void reset(std::int64_t batch) override {
        const auto half = mfq_tensor_backend::TensorOptions()
                              .device(mfq_tensor_backend::kCUDA)
                              .dtype(mfq_tensor_backend::kFloat16);
        state.local_kv =
            mfq_tensor_backend::zeros({batch, config.sliding_window, config.head_dim}, half);
        state.compressed_kv = Tensor();
        state.index_k = Tensor();
        state.partial_kv = Tensor();
        state.partial_score = Tensor();
        if (kv_source()) {
            const auto capacity = std::max<std::int64_t>(1, max_context / ratio);
            state.compressed_kv =
                mfq_tensor_backend::zeros({batch, capacity, config.head_dim}, half);
            state.index_k =
                mfq_tensor_backend::zeros({batch, capacity, config.index_head_dim}, half);
            if (ratio > 1) {
                const auto fp32 = half.dtype(mfq_tensor_backend::kFloat32);
                state.partial_kv = mfq_tensor_backend::zeros({batch, ratio, config.head_dim}, fp32);
                state.partial_score = mfq_tensor_backend::full(
                    {batch, ratio, config.head_dim}, -std::numeric_limits<float>::infinity(), fp32);
            }
        }
        state.position = 0;
        state.compressed_length = 0;
        state.partial_length = 0;
        speculative_local_slots = Tensor();
        speculative_slot_ids = Tensor();
        speculative_partial_kv = Tensor();
        speculative_partial_score = Tensor();
        speculative_projected_kv = Tensor();
        speculative_projected_score = Tensor();
        speculative_position = -1;
        speculative_pending = false;
        prefilled_global_end = -1;
        replay_local_window = false;
        if (layer == 0)
            shared->reset_session(batch);
    }

    bool supports_speculation() const noexcept override { return true; }

    void begin_speculative(std::int64_t draft_tokens) override {
        MFQ_RUNTIME_CHECK(!speculative_pending && draft_tokens > 0 &&
                              draft_tokens <= config.sliding_window && state.local_kv.defined(),
                          "invalid DeepSeek-V4.1 speculative checkpoint");
        speculative_position = state.position;
        speculative_compressed_length = state.compressed_length;
        speculative_partial_length = state.partial_length;
        speculative_slot_ids =
            mfq_tensor_backend::arange(state.position, state.position + draft_tokens,
                                       state.local_kv.options().dtype(mfq_tensor_backend::kInt64))
                .remainder(static_cast<double>(config.sliding_window))
                .to(mfq_tensor_backend::kInt64)
                .contiguous();
        speculative_local_slots = state.local_kv.index_select(1, speculative_slot_ids).clone();
        if (state.partial_kv.defined()) {
            speculative_partial_kv = state.partial_kv.clone();
            speculative_partial_score = state.partial_score.clone();
        }
        speculative_pending = true;
    }

    void commit_speculative() override {
        speculative_pending = false;
        speculative_local_slots = Tensor();
        speculative_slot_ids = Tensor();
        speculative_partial_kv = Tensor();
        speculative_partial_score = Tensor();
        speculative_projected_kv = Tensor();
        speculative_projected_score = Tensor();
        speculative_position = -1;
    }

    void rollback_speculative(std::int64_t position) override {
        MFQ_RUNTIME_CHECK(speculative_pending && position >= speculative_position &&
                              position <= state.position &&
                              speculative_local_slots.defined() && speculative_slot_ids.defined(),
                          "DeepSeek-V4.1 speculative checkpoint is unavailable");
        const auto kept=position-speculative_position;
        MFQ_RUNTIME_CHECK(kept<=speculative_slot_ids.numel(),"DeepSeek-V4.1 retained prefix is out of range");
        const auto rejected=speculative_slot_ids.numel()-kept;
        if(rejected)state.local_kv.index_copy_(1,speculative_slot_ids.narrow(0,kept,rejected),
            speculative_local_slots.narrow(1,kept,rejected));
        if (speculative_partial_kv.defined()) {
            state.partial_kv.copy_(speculative_partial_kv);
            state.partial_score.copy_(speculative_partial_score);
        }
        state.position = position;
        state.compressed_length = speculative_compressed_length;
        state.partial_length = speculative_partial_length;
        if(kept && kv_source()) {
            const auto total=speculative_partial_length+kept;
            state.compressed_length+=total/ratio;
            state.partial_length=total%ratio;
            // Complete compressed rows already contain the accepted prefix.
            // Rebuild only its unfinished group from the saved projections;
            // this needs no attention, projection or expert recomputation.
            for(int64_t slot=0;slot<state.partial_length;++slot) {
                const auto source=total-state.partial_length+slot-speculative_partial_length;
                if(source<0)continue; // The original partial group was restored above.
                MFQ_RUNTIME_CHECK(speculative_projected_kv.defined() && speculative_projected_score.defined(),
                                  "DeepSeek-V4.1 retained compressor projections are unavailable");
                state.partial_kv.narrow(1,slot,1).copy_(speculative_projected_kv.narrow(1,source,1));
                state.partial_score.narrow(1,slot,1).copy_(speculative_projected_score.narrow(1,source,1));
            }
        }
        commit_speculative();
    }

    void set_token_ids(const Tensor &ids) override { current_ids = ids; }

    MhcResult collapse(CudaProfiler &profiler, const Tensor &residual, const Tensor &previous_pre,
                       const Tensor &function, const Tensor &scale, const Tensor &base,
                       const Tensor &norm, const char *profile) const {
        return profiler.measure(profile, [&]() {
            if(residual.size(0)*residual.size(1)<=8) {
                auto partials=deepseek_v41_hc_project_cuda(residual,previous_pre,function);
                auto result=deepseek_v41_hc_finalize_cuda(residual,previous_pre,norm,partials,scale,base,
                    config.hc_sinkhorn_iters,config.rms_eps,config.hc_eps);
                return MhcResult{std::move(result[0]),std::move(result[1]),std::move(result[2]),std::move(result[3])};
            }
            auto prepared = deepseek_v41_hc_prepare_cuda(residual, previous_pre, norm, config.rms_eps);
            auto mixes = mfq_tensor_backend::matmul(prepared.at(0), function.transpose(0, 1));
            auto coefficients = dsv4_hc_coefficients_cuda(mixes.contiguous(), scale, base,
                                                          config.hc_sinkhorn_iters, config.hc_eps);
            return MhcResult{std::move(prepared.at(1)), std::move(coefficients.at(0)),
                             std::move(coefficients.at(1)), std::move(coefficients.at(2))};
        });
    }

    Tensor expand(CudaProfiler &profiler, const Tensor &branch, const Tensor &residual,
                  const MhcResult &mix, const char *profile) const {
        return profiler.measure(profile, [&]() {
            return dsv4_hc_post_cuda(branch.to(mfq_tensor_backend::kFloat16).contiguous(), residual.contiguous(),
                                     mix.post.contiguous(), mix.combination.contiguous());
        });
    }

    Tensor output_projection(CudaExecutionContext &execution, Tensor attention) const {
        return mfq::cuda::deepseek_v4::output_projection(execution, attention, output_a, output_b,
                                                         config.o_groups, true, false);
    }

    std::optional<Tensor> update_compressed_source(CudaExecutionContext &execution,
                                                   const Tensor &input, std::int64_t position) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        Tensor projected_kv, projected_score;
        return mfq::models::deepseek_v41::compress(
            state, tokens, position, ratio,
            [&] {
                return weighted_rms(compressor_key_value.forward(execution, input), compressor_norm,
                                    config.rms_eps);
            },
            [&] {
                projected_kv = compressor_key_value.forward(execution, input)
                                   .to(mfq_tensor_backend::kFloat32)
                                   .contiguous();
                projected_score = compressor_gate.forward(execution, input)
                                      .to(mfq_tensor_backend::kFloat32)
                                      .contiguous();
                if(speculative_pending) {
                    speculative_projected_kv=projected_kv;
                    speculative_projected_score=projected_score;
                }
            },
            [&](int64_t complete, int64_t cutoff) {
                auto kv_groups = projected_kv.narrow(1, 0, cutoff)
                                     .reshape({batch, complete, ratio, config.head_dim});
                auto score_groups = projected_score.narrow(1, 0, cutoff)
                                        .reshape({batch, complete, ratio, config.head_dim});
                auto pooled = (kv_groups * mfq_tensor_backend::softmax(score_groups, 2)).sum(2);
                return weighted_rms(pooled.to(mfq_tensor_backend::kFloat16), compressor_norm,
                                    config.rms_eps);
            },
            [&](int64_t cutoff, int64_t remainder) {
                state.partial_kv.narrow(1, 0, remainder)
                    .copy_(projected_kv.narrow(1, cutoff, remainder));
                state.partial_score.narrow(1, 0, remainder)
                    .copy_(projected_score.narrow(1, cutoff, remainder));
            },
            [&](int64_t token, int64_t slot) {
                state.partial_kv.narrow(1, slot, 1).copy_(projected_kv.narrow(1, token, 1));
                state.partial_score.narrow(1, slot, 1).copy_(projected_score.narrow(1, token, 1));
            },
            [&] {
                auto pooled =
                    (state.partial_kv * mfq_tensor_backend::softmax(state.partial_score, 1))
                        .sum(1, true);
                return weighted_rms(pooled.to(mfq_tensor_backend::kFloat16), compressor_norm,
                                    config.rms_eps);
            },
            [](const auto &emitted) { return mfq_tensor_backend::cat(emitted, 1).contiguous(); });
    }

    void publish_compressed_source(CudaExecutionContext &execution, const Tensor &input,
                                   std::int64_t position) {
        std::optional<Tensor> latent;
        if (prefilled_global_end == position + input.size(1)) prefilled_global_end = -1;
        else latent = update_compressed_source(execution, input, position);
        if (latent) {
            const auto count = latent->size(1);
            const auto begin = state.compressed_length;
            MFQ_RUNTIME_CHECK(begin + count <= state.compressed_kv.size(1),
                              "DeepSeek-V4.1 compressed cache capacity exceeded");
            auto group_positions =
                mfq_tensor_backend::arange(begin * ratio, (begin + count) * ratio, ratio,
                                           latent->options().dtype(mfq_tensor_backend::kInt64));
            auto index_values =
                weighted_rms(index_key.forward(execution, *latent), index_key_norm, config.rms_eps);
            index_values = rotate_token_major_tail(index_values, group_positions, rope);
            index_values = dsv4_fp4_sim_cuda(index_values.contiguous());
            auto compressed = rotate_token_major_tail(*latent, group_positions, rope);
            compressed = deepseek_v41_mxfp4_e4m3_scale_sim_cuda(compressed.contiguous());
            auto rows = mfq_tensor_backend::arange(
                begin, begin + count, latent->options().dtype(mfq_tensor_backend::kInt64));
            state.index_k.index_copy_(1, rows, index_values);
            state.compressed_kv.index_copy_(1, rows, compressed);
            state.compressed_length += count;
        }
        shared->compressed_kv = state.compressed_kv;
        shared->index_k = state.index_k;
        shared->compressed_length = state.compressed_length;
        shared->ratio = ratio;
    }

    Tensor compute_topk(CudaExecutionContext &execution, const Tensor &input, const Tensor &q_rank,
                        const Tensor &positions, std::int64_t position) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        const auto pool_length = shared->compressed_length;
        Tensor candidates;
        // Restrict scoring only when the candidate budget excludes some KV.
        // An all-inclusive list adds scattered loads without reducing work.
        const bool restricted = pool_length > config.candidate_topk_blocks * config.candidate_block_size;
        if(ratio>0 && pool_length>0 && pool_length<=config.index_topk && !restricted) {
            // Every visible pooled key is selected. The attention plan already
            // masks future groups, so index-query/score projections cannot
            // change the selected set. Reuse the ordered range across layers.
            if(!shared->topk.defined() || shared->topk.dim()!=3 || shared->topk.size(0)!=batch ||
                shared->topk.size(1)!=tokens || shared->topk.size(2)!=pool_length)
                shared->topk=mfq_tensor_backend::arange(pool_length,input.options().dtype(mfq_tensor_backend::kInt32))
                    .reshape({1,1,pool_length}).expand({batch,tokens,pool_length}).contiguous();
            return shared->topk;
        }
        return mfq::models::deepseek_v41::hierarchical_index(
            ratio, index_source(), layer, config.candidate_source_layer, pool_length,
            [&] { return empty_topk(batch, tokens, input.device()); },
            [&] {
                MFQ_RUNTIME_CHECK(shared->topk.defined(),
                                  "DeepSeek-V4.1 CSA2 consumer has no published index plan");
                return shared->topk;
            },
            [&] {
                auto query =
                    index_query.forward(execution, q_rank)
                        .reshape({batch, tokens, config.index_n_heads, config.index_head_dim})
                        .transpose(1, 2);
                query = index_query_transform(query, positions, rope);
                auto weights = index_score.forward(execution, input)
                                   .reshape({batch, tokens, config.index_n_heads})
                                   .to(mfq_tensor_backend::kFloat16)
                                   .contiguous();
                auto keys = shared->index_k.narrow(1, 0, pool_length).contiguous();
                if (layer > config.candidate_source_layer && restricted) {
                    MFQ_RUNTIME_CHECK(shared->candidate_blocks.defined(),
                                      "DeepSeek-V4.1 hierarchical indexer has no candidates");
                    candidates = candidate_positions(shared->candidate_blocks,
                                                     config.candidate_block_size, pool_length);
                    return execution.profiler.measure("deepseek_v41.indexer.candidate_scores", [&] {
                        return dsv4_indexer_candidate_scores_cuda(query, keys, weights, candidates,
                                                                  position, ratio);
                    });
                }
                return execution.profiler.measure("deepseek_v41.indexer.full_scores", [&] {
                    return dsv4_indexer_scores_cuda(query, keys, weights, position, ratio);
                });
            },
            [&](Tensor scores) {
                if (restricted) shared->candidate_blocks = candidate_blocks_from_scores(
                    scores, ratio, position, config.candidate_topk_blocks,
                    config.candidate_block_size);
            },
            [&](Tensor scores) {
                if (!restricted) return select_topk(scores, config.index_topk, pool_length);
                MFQ_RUNTIME_CHECK(candidates.defined(), "DeepSeek-V4.1 candidate scores are unavailable");
                return select_topk(scores, config.index_topk, pool_length, candidates);
            },
            [&](Tensor scores) { return select_topk(scores, config.index_topk, pool_length); },
            [&](Tensor topk) { shared->topk = std::move(topk); });
    }

    Tensor attention_forward(CudaExecutionContext &execution, const Tensor &input,
                             const Tensor &positions, std::int64_t position,
                             const MfqOptional<Tensor> &sequence_lengths,
                             bool truncated_local = false) {
        const auto batch = input.size(0);
        const auto tokens = input.size(1);
        const auto local_history = truncated_local ? 0 : std::min(position, config.sliding_window);
        return mfq::models::deepseek_v41::attention(
            kv_source(), ratio,
            [&] {
                return weighted_rms(query_a.forward(execution, input), query_a_norm,
                                    config.rms_eps);
            },
            [&](Tensor q_rank) {
                auto query = query_b.forward(execution, q_rank)
                                 .reshape({batch, tokens, config.n_heads, config.head_dim})
                                 .transpose(1, 2);
                query = dsv4_rotate_rope_tail(query, positions, rope, false, true);

                return query;
            },
            [&] {
                static const bool fused_kv=[] {
                    const char* value=std::getenv("MFQ_HIP_DSV41_KV_FUSED");
                    return !value || std::atoi(value)!=0;
                }();
                if(fused_kv && batch==1 && tokens==1 && config.head_dim==512) {
                    auto projection=key_value.forward(execution,input);
                    auto target=state.local_kv.narrow(1,position%config.sliding_window,1);
                    deepseek_v41_kv_norm_rope_store_cuda(projection,key_value_norm,positions,
                        rope.cos,rope.sin,target,config.rms_eps);
                    return target;
                }
                auto local = weighted_rms(key_value.forward(execution, input), key_value_norm,
                                          config.rms_eps)
                                 .reshape({batch, tokens, config.head_dim});
                local = rotate_token_major_tail(local, positions, rope);
                if(batch==1 && tokens==1) {
                    auto target=state.local_kv.narrow(1,position%config.sliding_window,1);
                    deepseek_v41_mxfp8_e4m3_sim_into_cuda(local.contiguous(),target);
                    return target;
                }
                local = deepseek_v41_mxfp8_e4m3_sim_cuda(local.contiguous());

                return local;
            },
            [&] { publish_compressed_source(execution, input, position); },
            [&] {
                MFQ_RUNTIME_CHECK(shared->compressed_kv.defined() && shared->index_k.defined() &&
                                      shared->ratio == ratio,
                                  "DeepSeek-V4.1 CSA2 consumer ran before its source layer");
            },
            [&](Tensor q_rank) {
                return compute_topk(execution, input, q_rank, positions, position);
            },
            [&](Tensor local) {
                Tensor local_for_attention;
                if (tokens == 1) {
                  if(batch!=1) {
                    auto slot = mfq_tensor_backend::full(
                        {1}, position % config.sliding_window,
                        positions.options().dtype(mfq_tensor_backend::kInt64));
                    state.local_kv.index_copy_(1, slot, local);
                  }
                    local_for_attention = state.local_kv;
                } else {
                    const auto history = local_history;
                    std::vector<Tensor> local_parts;
                    if (history > 0) {
                        auto slots = mfq_tensor_backend::arange(
                                         position - history, position,
                                         positions.options().dtype(mfq_tensor_backend::kInt64))
                                         .remainder(static_cast<double>(config.sliding_window))
                                         .to(mfq_tensor_backend::kInt64)
                                         .contiguous();
                        local_parts.push_back(state.local_kv.index_select(1, slots));
                    }
                    local_parts.push_back(local);
                    local_for_attention =
                        local_parts.size() == 1
                            ? local_parts.front()
                            : mfq_tensor_backend::cat(local_parts, 1).contiguous();
                    const auto recent = std::min(tokens, config.sliding_window);
                    auto recent_slots = mfq_tensor_backend::arange(
                                            position + tokens - recent, position + tokens,
                                            positions.options().dtype(mfq_tensor_backend::kInt64))
                                            .remainder(static_cast<double>(config.sliding_window))
                                            .to(mfq_tensor_backend::kInt64)
                                            .contiguous();
                    state.local_kv.index_copy_(1, recent_slots,
                                               local.narrow(1, tokens - recent, recent));
                }

                const auto pool_length = ratio > 0 ? shared->compressed_length : 0;
                if(pool_length>0) {
                    if(batch==1 && tokens==1)return local_for_attention;
                    return deepseek_v41_join_kv_cuda(local_for_attention,
                        shared->compressed_kv.narrow(1,0,pool_length));
                }
                return local_for_attention.contiguous();
            },
            [&](Tensor topk) {
                const auto pool_length = ratio > 0 ? shared->compressed_length : 0;
                std::vector<Tensor> plan;
                if (tokens == 1) {
                    auto lengths = sequence_lengths.has_value()
                                       ? sequence_lengths.value()
                                             .to(input.device(), mfq_tensor_backend::kInt64)
                                             .contiguous()
                                       : mfq_tensor_backend::full(
                                             {batch}, position + 1,
                                             positions.options().dtype(mfq_tensor_backend::kInt64));
                    plan = dsv4_build_decode_plan_cuda(
                        topk, lengths, pool_length, ratio > 0 ? ratio : 1, config.sliding_window);
                } else {
                    MFQ_RUNTIME_CHECK(
                        !sequence_lengths.has_value(),
                        "DeepSeek-V4.1 chunked prefill does not accept per-row lengths");
                    const auto history = local_history;
                    plan =
                        dsv4_build_prefill_plan_cuda(topk, position, history, pool_length,
                                                     ratio > 0 ? ratio : 1, config.sliding_window);
                }
                return plan;
            },
            [&](Tensor query, Tensor cache, const auto &plan) {
                const auto pool_length=ratio>0?shared->compressed_length:0;
                if(batch==1 && tokens==1 && pool_length>0)
                    return attention_dsv4_sparse_split_cuda(query,cache,
                        shared->compressed_kv.narrow(1,0,pool_length),plan.at(0),plan.at(1),sinks,
                        shared->attention_meta,1.0/std::sqrt(static_cast<double>(config.head_dim)));
                return attention_dsv4_sparse_cuda(
                    query, cache, plan.at(0), plan.at(1), sinks, shared->attention_meta,
                    1.0 / std::sqrt(static_cast<double>(config.head_dim)));
            },
            [&](Tensor attended) {
                return dsv4_rotate_rope_tail(attended.transpose(1, 2), positions, rope,
                                             true, false, true);
            },
            [&] { state.position += tokens; },
            [&](Tensor attended) { return output_projection(execution, attended); });
    }

    Tensor forward_context(CudaExecutionContext& execution,Tensor hidden,
            const ::Block::Context& context,const RopeCache& rope) override {
        if(context.confirmed_prefix>0) {
            // A joint pass writes pending and draft rows. Protect their whole
            // physical window; rollback retains the confirmed prefix as well.
            begin_speculative(hidden.size(1));
            if(layer==0)shared->begin_speculative();
        }
        return ::Block::forward_context(execution,std::move(hidden),context,rope);
    }

    Tensor forward(CudaExecutionContext &execution, Tensor hidden, Tensor positions,
                   std::int64_t cache_position, const MfqOptional<Tensor> &sequence_lengths,
                   const RopeCache &, const MfqOptional<Tensor> &cache_positions = mfq_nullopt,
                   const MfqOptional<Tensor> &attention_mask = mfq_nullopt) override {
        auto &profiler = execution.profiler;
        MFQ_RUNTIME_CHECK(current_ids.defined() && hidden.dim() == 4 &&
                              hidden.size(2) == config.hc_mult && hidden.size(3) == config.hidden &&
                              positions.dim() == 1 && positions.numel() == hidden.size(1) &&
                              !cache_positions.has_value() && !attention_mask.has_value() &&
                              state.position == cache_position,
                          "DeepSeek-V4.1 block input/cache mismatch");
        shared->enter_layer(layer, config.n_layers, hidden.size(0), hidden.size(1), cache_position,
                            hidden.device(), current_ids);

        const auto batch = hidden.size(0), tokens = hidden.size(1);
        const bool truncated_local = std::exchange(replay_local_window, false);
        return mfq::models::deepseek_v41::decoder_layer(
            std::move(hidden), bool(engram), shared->previous_pre,
            [&](Tensor value) {
                return profiler.measure("deepseek_v41.engram", [&] {
                    return engram->forward(execution, value, shared->engram_hashes,
                                          shared->text_rows,&shared->text_row_ids);
                });
            },
            [&](const Tensor &value) { shared->capture_dspark_target(layer, config, value); },
            [&](Tensor value, const Tensor &previous_pre, int branch) {
                return collapse(profiler, value, previous_pre,
                                branch == 0 ? attention_mhc_function : mlp_mhc_function,
                                branch == 0 ? attention_mhc_scale : mlp_mhc_scale,
                                branch == 0 ? attention_mhc_base : mlp_mhc_base,
                                branch == 0 ? attention_norm : mlp_norm,
                                branch == 0 ? "deepseek_v41.mhc.attention.collapse"
                                            : "deepseek_v41.mhc.mlp.collapse");
            },
            [&](Tensor value) {
                return profiler.measure("deepseek_v41.attention", [&] {
                    return attention_forward(execution, value, positions, cache_position,
                                             sequence_lengths, truncated_local);
                });
            },
            [&](Tensor value, Tensor residual, const auto &mix, int branch) {
                return expand(profiler, value, residual, mix,
                              branch == 0 ? "deepseek_v41.mhc.attention.expand"
                                          : "deepseek_v41.mhc.mlp.expand");
            },
            [&](Tensor value) {
                return profiler.measure("deepseek_v41.mlp", [&] {
                    if(shared->image_mask.defined()) {
                        MFQ_RUNTIME_CHECK(vision_router_bias.defined(),"DeepSeek-V4.1 visual router bias is missing");
                        auto bias=mfq_tensor_backend::where(shared->routing_mask,vision_router_bias,mlp.moe_router_bias).contiguous();
                        return mlp.forward_with_router_bias(execution,value.reshape({-1,config.hidden}),
                            current_ids,bias).reshape({batch,tokens,config.hidden});
                    }
                    return mlp.forward(execution, value.reshape({-1, config.hidden}), current_ids)
                        .reshape({batch, tokens, config.hidden});
                });
            });
    }
};

FFN load_moe_at(CudaExecutionContext &execution, const mfq::ModelSource &model,
                const CommonConfig &config, const std::string &prefix, std::int64_t layer,
                std::int64_t top_k);

std::unique_ptr<::Block> load_block(CudaExecutionContext &execution, const mfq::ModelSource &model,
                                    std::int64_t layer, const std::shared_ptr<SharedState> &shared);
void validate_load_options(const CudaExecutionContext &execution);
std::shared_ptr<SharedState> load_shared_state(const mfq::ModelSource &source,
                                               const CommonConfig &config);
Tensor finalize_hidden(const std::shared_ptr<SharedState> &state, const Tensor &hidden,
                       CudaProfiler &profiler);

inline int run_self_check() {
    EngramHashState::self_check();
    run_dspark_self_check();
    auto options = mfq_tensor_backend::TensorOptions()
                       .device(mfq_tensor_backend::kCUDA)
                       .dtype(mfq_tensor_backend::kInt32);
    auto blocks =
        mfq_tensor_backend::tensor(std::vector<std::int32_t>{0, 2, -1}, options).reshape({1, 1, 3});
    auto positions = candidate_positions(blocks, 2, 5)
                         .to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                         .contiguous();
    const std::vector<std::int64_t> expected{0, 1, 4, -1, -1, -1};
    const auto *values = positions.data_ptr<std::int64_t>();
    MFQ_RUNTIME_CHECK(positions.numel() == static_cast<std::int64_t>(expected.size()) &&
                          std::equal(expected.begin(), expected.end(), values),
                      "DeepSeek-V4.1 candidate hierarchy contract failed");
    std::cout << "deepseek_v41_runtime_check=ok\n";
    return 0;
}

} // namespace mfq::cuda::deepseek_v41_runtime

namespace mfq::cuda {

struct DeepseekV41Model : CausalResources {
    static constexpr bool joint_speculation_supported=true;
    bool adapter_prefetch_token_rows(DeepseekV41CausalLm&,const mfq_tensor_backend::Tensor&);
    template <class Backend> using CausalModel = mfq::models::deepseek_v41::CausalLm<Backend>;
    mfq::models::deepseek_v41::Config config;
    std::shared_ptr<deepseek_v41_runtime::SharedState> shared;

    void adapter_validate_load_options() const;
    void adapter_load_final_state(const mfq::ModelSource &source,
                                  mfq_tensor_backend::Tensor &output_norm);
    void adapter_prepare_blocks(const mfq::ModelSource &source);
    std::unique_ptr<Block> adapter_load_block(const mfq::ModelSource &source, int layer, int device,
                                              const std::string &type);

    void adapter_begin_forward(bool capture_raw_hidden);
    bool adapter_supports_prepared_prompt() const noexcept {return true;}
    void adapter_set_image_mask(mfq_tensor_backend::Tensor mask) {shared->set_image_mask(std::move(mask));}
    mfq_tensor_backend::Tensor collapse_hidden(mfq_tensor_backend::Tensor hidden, int64_t batch,
                                               int64_t tokens) const;
    mfq_tensor_backend::Tensor normalize_hidden(mfq_tensor_backend::Tensor hidden,
                                                const mfq_tensor_backend::Tensor &output_norm,
                                                int64_t batch, int64_t tokens) const;
    mfq_tensor_backend::Tensor
    adapter_raw_hidden(const mfq_tensor_backend::Tensor &hidden,
                       const mfq_tensor_backend::Tensor &finalized) const;
    void adapter_begin_speculative();
    void adapter_commit_speculative();
    void adapter_rollback_speculative(int64_t keep);
};

template <> struct CudaSessionCodec<DeepseekV41Model> {
    using Model = DeepseekV41CausalLm;
    static TextSessionStateKind kind(const Model& model);
    static bool supports_paged(const Model& model);
    static TextSessionState capture(const Model& model,const std::vector<int64_t>& tokens);
    static void restore(Model& model,const TextSessionState& state);
};

mfq_tensor_backend::Tensor deepseek_v41_prefill_hidden(
    DeepseekV41CausalLm& model, mfq_tensor_backend::Tensor ids, std::int64_t compute_chunk,
    const std::function<bool()>& stopped = {}, mfq_tensor_backend::Tensor* raw_hidden = nullptr);

} // namespace mfq::cuda

namespace mfq::models {
extern template struct deepseek_v41::CausalLm<cuda::CudaCausalOps<cuda::DeepseekV41Model>>;
} // namespace mfq::models
