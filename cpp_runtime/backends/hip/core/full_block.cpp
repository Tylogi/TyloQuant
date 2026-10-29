#include "hip/hip_runtime.h"
#include "full_block.h"
#include "models/common/attention.h"
#include "models/common/gated_mlp.h"
#include "models/common/transformer_layer.h"
#include <array>

#include "mfq_hip_activation_ops.h"
#include "mfq_hip_attention_ops.h"
#include "selected_attention.h"
#include "mfq_hip_norm_ops.h"
#include "mfq_hip_paged_kv.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

mfq_tensor_backend::Tensor Block::forward_context(CudaExecutionContext &execution,
                                                  mfq_tensor_backend::Tensor x,
                                                  const Context &context, const RopeCache &rope) {
    MFQ_RUNTIME_CHECK(context.confirmed_prefix == 0 || supports_speculation(),
                      "block does not support speculative verification");
    return forward(execution, std::move(x), context.positions, context.cache_position,
                   context.sequence_lengths, rope, context.cache_positions, context.attention_mask);
}

void FullBlock::reset(int64_t B) {
    if (cache.defined() && cache.batch_size() == B)
        return;
    cache = KVCache();
    decode_partial_o = mfq_tensor_backend::Tensor();
    decode_partial_m = mfq_tensor_backend::Tensor();
    decode_partial_l = mfq_tensor_backend::Tensor();
    decode_mma_mask = mfq_tensor_backend::Tensor();
    decode_mma_kv_max = mfq_tensor_backend::Tensor();
    decode_mma_meta = mfq_tensor_backend::Tensor();
}

mfq_tensor_backend::Tensor
FullBlock::forward(CudaExecutionContext &execution, mfq_tensor_backend::Tensor x,
                   mfq_tensor_backend::Tensor pos, int64_t cache_pos,
                   const MfqOptional<mfq_tensor_backend::Tensor> &seq_len, const RopeCache &rope,
                   const MfqOptional<mfq_tensor_backend::Tensor> &cache_positions,
                   const MfqOptional<mfq_tensor_backend::Tensor> &attention_mask) {
    return forward_impl(execution, std::move(x), std::move(pos), cache_pos, seq_len, rope,
                        cache_positions, attention_mask, 0, 0);
}

mfq_tensor_backend::Tensor FullBlock::forward_context(CudaExecutionContext &execution,
                                                      mfq_tensor_backend::Tensor x,
                                                      const Context &context,
                                                      const RopeCache &rope) {
    MFQ_RUNTIME_CHECK(context.confirmed_prefix == 0 || supports_speculation(),
                      "block does not support speculative verification");
    return forward_impl(execution, std::move(x), context.positions, context.cache_position,
                        context.sequence_lengths, rope, context.cache_positions,
                        context.attention_mask, context.planned_kv_length,
                        context.decode_attention_parts);
}

mfq_tensor_backend::Tensor FullBlock::forward_impl(
    CudaExecutionContext &execution, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos,
    int64_t cache_pos, const MfqOptional<mfq_tensor_backend::Tensor> &seq_len,
    const RopeCache &rope, const MfqOptional<mfq_tensor_backend::Tensor> &cache_positions,
    const MfqOptional<mfq_tensor_backend::Tensor> &attention_mask, int64_t planned_kv_length,
    int64_t decode_attention_parts) {
    auto &profiler = execution.profiler;
    int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    auto trace_qwen_stage = [&](const char *name, const mfq_tensor_backend::Tensor &value,
                                int token_axis = 1) {
        if (!attention_output_gate || execution.gemma_stage_trace == nullptr ||
            layer != execution.gemma_trace_layer)
            return;
        auto ordered = token_axis == 1 ? value : value.transpose(1, token_axis).contiguous();
        trace_gemma_stage(execution, layer, name, ordered.reshape({B, T, -1}));
    };
    if (official_bf16 && x.scalar_type() != mfq_tensor_backend::kBFloat16) {
        x = x.to(mfq_tensor_backend::kBFloat16).contiguous();
    }
    const int64_t nh = attention_heads;
    const int64_t nkh = kv_heads;
    const int64_t hd = attention_head_dim;
    const int64_t attn_width = nh * hd;
    const auto kl_capacity = execution.kl_mmq.kv_cache_capacity;
    const int64_t cache_capacity =
        sliding ? attention_window
                : (kl_capacity > 0 ? std::max<int64_t>(cache_pos + T, kl_capacity)
                                   : max_position_embeddings);
    const RopeCache &active_rope = attention_rope.cos.defined() ? attention_rope : rope;
    const bool apply_rotary = !no_rope;
    if (!cache.defined()) {
        cache =
            KVCache(B, nkh, cache_capacity, hd, sliding, x.device(),
                    official_bf16 ? mfq_tensor_backend::kBFloat16 : mfq_tensor_backend::kFloat16, kv_quantization);
    }
    cache.quantization=kv_quantization;
    if (x.is_cuda()) {
        const int64_t total = B * nh;
        if (!decode_partial_o.defined() || decode_partial_o.size(0) < total ||
            decode_partial_o.size(1) != kDecodeAttentionMaxParts ||
            decode_partial_o.size(2) != hd) {
            auto opts = mfq_tensor_backend::TensorOptions()
                            .device(x.device())
                            .dtype(mfq_tensor_backend::kFloat32);
            decode_partial_o =
                mfq_tensor_backend::empty({total, kDecodeAttentionMaxParts, hd}, opts);
            decode_partial_m = mfq_tensor_backend::empty({total, kDecodeAttentionMaxParts}, opts);
            decode_partial_l = mfq_tensor_backend::empty({total, kDecodeAttentionMaxParts}, opts);
        }
    }
    return mfq::models::attention_layer(
        x,
        [&](mfq_tensor_backend::Tensor x) {
            return profiler.measure("full.attn_norm", [&]() {
                return official_bf16
                           ? qwen_rms_norm_bf16(execution.config, x.reshape({B * T, H}), attn_norm,
                                                rms_norm_eps, norm_weight_offset)
                                 .reshape({B, T, H})
                           : qwen_rms_norm(x.reshape({B * T, H}).contiguous(),
                                           attn_norm, rms_norm_eps, norm_weight_offset)
                                 .reshape({B, T, H});
            });
        },
        [&](mfq_tensor_backend::Tensor xn) {
            auto write_positions =
                cache_positions.has_value()
                    ? cache_positions.value()
                          .to(x.device(), mfq_tensor_backend::kInt64)
                          .contiguous()
                    : pos.to(x.device(), mfq_tensor_backend::kInt64).contiguous();
            if (!((write_positions.dim() == 1 && write_positions.numel() == T) ||
                  (write_positions.dim() == 2 && write_positions.size(0) == B &&
                   write_positions.size(1) == T))) {
                throw std::runtime_error(
                    "KV cache positions must have shape [tokens] or [batch,tokens]");
            }
            // A leading extent of three is semantic axes only for the prepared
            // single-image path. B=3 ordinary batching must remain batch-major.
            const bool grid_mrope_positions =
                B == 1 && cache_positions.has_value() && pos.dim() == 2 && pos.size(0) == 3;
            const bool fused_qk_rope_kv =
                apply_rotary && official_bf16 && x.is_cuda() &&
                write_positions.dim() == 1 && pos.dim() == 1 &&
                T == 1 && !cache.is_paged() && !cache.ring && !cache.quantization.enabled() && !v_norm.defined() &&
                q_norm.defined() && k_norm.defined() && active_rope.rotary_dim == 128 &&
                active_rope.sections.numel() == 0 && nh == 32 && nkh == 8 && hd == 128 &&
                cache.scalar_type() == mfq_tensor_backend::kBFloat16 &&
                execution.config.minicpm_fused_qk_norm_rope_kv;
            std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> kv;
            bool attention_token_major = false;
            return mfq::models::full_attention(
                std::move(xn),
                [&](mfq_tensor_backend::Tensor xn) {
                    trace_qwen_stage("qwen.attn_norm", xn);
                    auto parts = profiler.measure("full.qkv", [&]() {
                        if (!split_q_kv_projections) {
                            return qkv.forward(execution, xn);
                        }
                        if(T==1 && !tq_q8_enabled() && q_projection.is_tq() &&
                           k_projection.is_tq() && v_projection.is_tq() &&
                           !q_projection.tensor_parallel() && !k_projection.tensor_parallel() &&
                           !v_projection.tensor_parallel())
                            return tq_forward_group({&q_projection.tq,&k_projection.tq,&v_projection.tq},xn);
                        std::vector<mfq_tensor_backend::Tensor> result;
                        result.reserve(3);
                        result.push_back(q_projection.forward(execution, xn));
                        result.push_back(k_projection.forward(execution, xn));
                        result.push_back(v_projection.forward(execution, xn));
                        return result;
                    });
                    if (parts.size() != (value_equals_key ? 2u : 3u)) {
                        throw std::runtime_error(
                            "attention projection group has the wrong output count");
                    }
                    auto q_full = parts[0], k_full = parts[1];
                    auto v_full = value_equals_key ? k_full : parts[2];
                    trace_qwen_stage("qwen.q_projection", q_full);
                    trace_qwen_stage("qwen.k_projection", k_full);
                    trace_qwen_stage("qwen.v_projection", v_full);
                    if (official_bf16) {
                        q_full = q_full.to(mfq_tensor_backend::kBFloat16).contiguous();
                        k_full = k_full.to(mfq_tensor_backend::kBFloat16).contiguous();
                        v_full = v_full.to(mfq_tensor_backend::kBFloat16).contiguous();
                    }
                    mfq_tensor_backend::Tensor q_raw, q_gate;
                    profiler.measure("full.qkv_view", [&]() {
                        if (attention_output_gate) {
                            auto qp = q_full.reshape({B, T, nh, hd * 2});
                            auto chunks = qp.chunk(2, -1);
                            q_raw = chunks[0];
                            q_gate = chunks[1];
                        } else {
                            q_raw = q_full.reshape({B, T, nh, hd});
                        }
                        return q_raw;
                    });
                    auto q = profiler.measure("full.q_view",
                                              [&]() { return q_raw.transpose(1, 2).contiguous(); });
                    auto k = profiler.measure("full.k_view", [&]() {
                        return k_full.reshape({B, T, nkh, hd}).transpose(1, 2).contiguous();
                    });
                    auto v = profiler.measure("full.v_view", [&]() {
                        return v_full.reshape({B, T, nkh, hd}).transpose(1, 2).contiguous();
                    });
                    return mfq::models::AttentionHeads<mfq_tensor_backend::Tensor>{
                        q, k, v,
                        q_gate.defined() ? std::optional<mfq_tensor_backend::Tensor>(q_gate)
                                         : std::nullopt};
                },
                [&](auto &heads) {
                    auto &q = heads.query;
                    auto &k = heads.key;
                    auto &v = heads.value;
                    if (!fused_qk_rope_kv)
                        return false;

                    q = profiler.measure("full.qk_norm_rope_kv_write", [&]() {
                        return minicpm_qk_norm_rope_cache_write_bf16_cuda(
                            q.contiguous(), k.contiguous(), v.contiguous(), q_norm, k_norm,
                            pos.contiguous().to(x.device(), mfq_tensor_backend::kInt64),
                            write_positions, active_rope.cos, active_rope.sin, cache.k, cache.v,
                            rms_norm_eps, norm_weight_offset);
                    });
                    kv = {cache.k.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()}),
                          cache.v.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()})};
                    return true;
                },
                [&](auto &heads) {
                    auto &q = heads.query;
                    auto &k = heads.key;
                    auto &v = heads.value;

                    const bool fused_bf16_norm =
                        official_bf16 && execution.config.minicpm_fused_bf16_rmsnorm;
                    if (fused_bf16_norm && q_norm.defined() && k_norm.defined() &&
                        q.scalar_type() == mfq_tensor_backend::kBFloat16 &&
                        k.scalar_type() == mfq_tensor_backend::kBFloat16) {
                        auto normalized = profiler.measure("full.qk_norm", [&]() {
                            return qwen_rms_norm_pair_bf16_cuda(q, k, q_norm, k_norm, rms_norm_eps,
                                                                norm_weight_offset);
                        });
                        q = normalized[0].reshape_as(q);
                        k = normalized[1].reshape_as(k);
                    } else if (!official_bf16 && q_norm.defined() && k_norm.defined() &&
                               q.scalar_type() == mfq_tensor_backend::kFloat16 &&
                               k.scalar_type() == mfq_tensor_backend::kFloat16) {
                        auto normalized = profiler.measure("full.qk_norm", [&]() {
                            return rms_norm_pair_f16_f32_offset_cuda(
                                q, k, q_norm, k_norm, rms_norm_eps, norm_weight_offset);
                        });
                        q = normalized[0].reshape_as(q);
                        k = normalized[1].reshape_as(k);
                    } else {
                        if (q_norm.defined())
                            q = profiler.measure("full.q_norm", [&]() {
                                return official_bf16
                                           ? qwen_rms_norm_bf16(execution.config,
                                                                q.reshape({-1, hd}), q_norm,
                                                                rms_norm_eps, norm_weight_offset)
                                                 .reshape_as(q)
                                           : qwen_rms_norm(q.reshape({-1, hd}).to(
                                                               mfq_tensor_backend::kFloat32),
                                                           q_norm, rms_norm_eps, norm_weight_offset)
                                                 .reshape_as(q);
                            });
                        if (k_norm.defined())
                            k = profiler.measure("full.k_norm", [&]() {
                                return official_bf16
                                           ? qwen_rms_norm_bf16(execution.config,
                                                                k.reshape({-1, hd}), k_norm,
                                                                rms_norm_eps, norm_weight_offset)
                                                 .reshape_as(k)
                                           : qwen_rms_norm(k.reshape({-1, hd}).to(
                                                               mfq_tensor_backend::kFloat32),
                                                           k_norm, rms_norm_eps, norm_weight_offset)
                                                 .reshape_as(k);
                            });
                    }
                    if (v_norm.defined())
                        v = profiler.measure("full.v_norm", [&]() {
                            return official_bf16
                                       ? qwen_rms_norm_bf16(execution.config, v.reshape({-1, hd}),
                                                            v_norm, rms_norm_eps,
                                                            norm_weight_offset)
                                             .reshape_as(v)
                                       : qwen_rms_norm(
                                             v.reshape({-1, hd}).to(mfq_tensor_backend::kFloat32),
                                             v_norm, rms_norm_eps, norm_weight_offset)
                                             .reshape_as(v);
                        });
                },
                [&](auto &heads) {
                    auto &q = heads.query;
                    auto &k = heads.key;
                    auto &v = heads.value;
                    const bool fused_rope_kv =
                        apply_rotary && official_bf16 && x.is_cuda() &&
                        write_positions.dim() == 1 &&
                        pos.dim() == 1 && T == 1 && !cache.is_paged() && !cache.ring &&
                        active_rope.rotary_dim == 128 && active_rope.sections.numel() == 0 &&
                        nh == 32 && nkh == 8 && hd == 128 &&
                        cache.scalar_type() == mfq_tensor_backend::kBFloat16 &&
                        execution.config.minicpm_fused_rope_kv;
                    if (!fused_rope_kv)
                        return false;

                    q = profiler.measure("full.rope_kv_write", [&]() {
                        return minicpm_bf16_rope_cache_write_cuda(
                            q.contiguous(), k.contiguous(), v.contiguous(),
                            pos.contiguous().to(x.device(), mfq_tensor_backend::kInt64),
                            write_positions, active_rope.cos, active_rope.sin, cache.k, cache.v,
                            active_rope.rotary_dim);
                    });
                    kv = {cache.k.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()}),
                          cache.v.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()})};
                    return true;
                },
                [&](auto &heads) {
                    auto &q = heads.query;
                    auto &k = heads.key;

                    q = profiler.measure("full.q_rope", [&]() {
                        return official_bf16 ? active_rope.apply_bf16(execution.config, q, pos)
                                             : active_rope.apply(q, pos, grid_mrope_positions);
                    });
                    k = profiler.measure("full.k_rope", [&]() {
                        return official_bf16 ? active_rope.apply_bf16(execution.config, k, pos)
                                             : active_rope.apply(k, pos, grid_mrope_positions);
                    });
                },
                [&](auto &heads) {
                    auto &k = heads.key;
                    auto &v = heads.value;
                    kv = profiler.measure("full.kv_write", [&]() {
                        return cache.append(execution.config, k, v, write_positions, cache_pos,
                                            cache_pos + T, !seq_len.has_value());
                    });
                },
                [&](auto &heads) {
                    auto &q = heads.query;
                    auto &k = heads.key;
                    auto &v = heads.value;

                    trace_qwen_stage("qwen.q_rope", q, 2);
                    trace_qwen_stage("qwen.k_rope", k, 2);
                    auto minicpmo45_attention_mask = [&](int64_t visible_len,
                                                         bool explicit_causal) {
                        MfqOptional<mfq_tensor_backend::Tensor> result = mfq_nullopt;
                        if (!official_bf16 || (!explicit_causal && !seq_len.has_value() &&
                                               !attention_mask.has_value())) {
                            return result;
                        }
                        auto options = mfq_tensor_backend::TensorOptions()
                                           .device(x.device())
                                           .dtype(mfq_tensor_backend::kInt64);
                        auto key_positions = mfq_tensor_backend::arange(visible_len, options);
                        auto query_positions =
                            write_positions.to(x.device(), mfq_tensor_backend::kInt64);
                        if (query_positions.dim() == 1) {
                            query_positions = query_positions.reshape({1, T}).expand({B, T});
                        }
                        auto allowed = key_positions.reshape({1, 1, visible_len}) <=
                                       query_positions.unsqueeze(-1);
                        if (seq_len.has_value()) {
                            auto lengths = seq_len.value()
                                               .to(x.device(), mfq_tensor_backend::kInt64)
                                               .reshape({B, 1, 1});
                            allowed =
                                allowed & (key_positions.reshape({1, 1, visible_len}) < lengths);
                        }
                        if (attention_mask.has_value()) {
                            auto valid = attention_mask.value().to(x.device());
                            if (valid.dim() != 2 || valid.size(0) != B ||
                                valid.size(1) < visible_len) {
                                throw std::runtime_error(
                                    "MiniCPM-o attention_mask must cover [batch,visible_tokens]");
                            }
                            valid = valid.narrow(1, 0, visible_len).ne(0);
                            allowed = allowed & valid.unsqueeze(1);
                            auto attended = allowed.any(-1, true);
                            allowed = mfq_tensor_backend::where(
                                attended, allowed, mfq_tensor_backend::ones_like(allowed));
                        }
                        const auto mask_dtype =
                            official_bf16 ? mfq_tensor_backend::kBFloat16 : x.scalar_type();
                        auto additive = mfq_tensor_backend::zeros(
                            {B, 1, T, visible_len}, mfq_tensor_backend::TensorOptions()
                                                        .device(x.device())
                                                        .dtype(mask_dtype));
                        const double mask_min =
                            static_cast<double>(std::numeric_limits<mfq_bfloat16>::lowest());
                        additive.masked_fill_(allowed.logical_not().unsqueeze(1), mask_min);
                        result = additive.contiguous();
                        return result;
                    };
                    double attn_scale =
                        attention_scale > 0.0 ? attention_scale : 1.0 / std::sqrt((double)hd);
                    mfq_tensor_backend::Tensor a;
                    a = profiler.measure("full.attention", [&]() {
                        if(cache.quantization.enabled()) {
                            attention_token_major=true;
                            const auto capacity=planned_kv_length>0?planned_kv_length:cache_pos+T;
                            return mfq::cuda::cuda_kv_attention(q,cache.k,cache.v,cache.quantization,{},write_positions,
                                cache_pos,cache.quantized_pages(),capacity,seq_len.value_or(mfq_tensor_backend::Tensor{}),
                                attention_mask.value_or(mfq_tensor_backend::Tensor{}),attn_scale);
                        }
                        if (!x.is_cuda()) {
                            MFQ_RUNTIME_CHECK(
                                !sliding, "CPU dense offload does not support sliding attention");
                            const auto cpu_attention_dtype = official_bf16
                                                                 ? mfq_tensor_backend::kBFloat16
                                                                 : mfq_tensor_backend::kFloat32;
                            auto qh = q.to(cpu_attention_dtype).contiguous();
                            auto kh = kv.first.to(cpu_attention_dtype).contiguous();
                            auto vh = kv.second.to(cpu_attention_dtype).contiguous();
                            if (nkh != nh) {
                                MFQ_RUNTIME_CHECK(nh % nkh == 0,
                                                  "CPU attention head ratio is invalid");
                                const int64_t repeat = nh / nkh;
                                kh = kh.repeat_interleave(repeat, 1);
                                vh = vh.repeat_interleave(repeat, 1);
                            }
                            MfqOptional<mfq_tensor_backend::Tensor> mask = mfq_nullopt;
                            if (official_bf16) {
                                mask = minicpmo45_attention_mask(kh.size(2), T > 1);
                            } else if (T > 1 || seq_len.has_value()) {
                                const int64_t visible = kh.size(2);
                                auto key_positions = mfq_tensor_backend::arange(
                                    visible, mfq_tensor_backend::TensorOptions()
                                                 .device(mfq_tensor_backend::kCPU)
                                                 .dtype(mfq_tensor_backend::kInt64));
                                mfq_tensor_backend::Tensor allowed;
                                if (T > 1) {
                                    auto query_positions =
                                        pos.dim() == 1
                                            ? pos.to(mfq_tensor_backend::kCPU,
                                                     mfq_tensor_backend::kInt64)
                                            : pos.select(0, 0).to(mfq_tensor_backend::kCPU,
                                                                  mfq_tensor_backend::kInt64);
                                    allowed =
                                        key_positions.unsqueeze(0) <= query_positions.unsqueeze(1);
                                    allowed = allowed.unsqueeze(0).expand({B, T, visible});
                                } else {
                                    allowed = mfq_tensor_backend::ones(
                                        {B, T, visible}, mfq_tensor_backend::TensorOptions()
                                                             .device(mfq_tensor_backend::kCPU)
                                                             .dtype(mfq_tensor_backend::kBool));
                                }
                                if (seq_len.has_value()) {
                                    auto lengths = seq_len.value()
                                                       .to(mfq_tensor_backend::kCPU,
                                                           mfq_tensor_backend::kInt64)
                                                       .reshape({B, 1, 1});
                                    allowed = allowed &
                                              (key_positions.reshape({1, 1, visible}) < lengths);
                                }
                                mask = allowed.unsqueeze(1);
                            }
                            return mfq_scaled_dot_product_attention(qh, kh, vh, mask, 0.0, false,
                                                                    attn_scale, false);
                        }
                        const auto attention_dtype = official_bf16 ? mfq_tensor_backend::kBFloat16
                                                                   : mfq_tensor_backend::kFloat16;
                        auto qh = q.to(attention_dtype).contiguous();
                        auto kh = k.to(attention_dtype).contiguous();
                        auto vh = v.to(attention_dtype).contiguous();
                        static const bool disable_mma_attention = [] {
                            const char* v = std::getenv("MFQ_DISABLE_MMA_ATTENTION");
                            return v != nullptr && std::atoi(v) != 0;
                        }();
                        const bool mma_attention_enabled =
                            execution.config.mma_attention && !disable_mma_attention;
                        static const bool disable_nope_mfma=[] {
                            const auto* value=std::getenv("MFQ_HIP_NOPE_MFMA");
                            return value && std::atoi(value)==0;
                        }();
                        if(no_rope && !sliding && !official_bf16 && hd==128 && T>1 &&
                            mma_attention_enabled && !disable_nope_mfma && !cache.is_paged() &&
                            !seq_len.has_value() && !attention_mask.has_value() && pos.dim()==1) {
                            attention_token_major=true;
                            return mfq_selected_attention::causal(qh,cache.k,cache.v,pos,
                                cache_pos+T,attn_scale).to(mfq_tensor_backend::kFloat16);
                        }
                        if (cache_pos == 0 && T > 1) {
                            if (official_bf16) {
                                const bool bf16_flash128 = !sliding && hd == 128 && nh == 4 * nkh &&
                                                           !seq_len.has_value() &&
                                                           !attention_mask.has_value() &&
                                                           execution.config.minicpm_bf16_flash128;
                                if (bf16_flash128) {
                                    const bool specialized_casts =
                                        execution.config.minicpm_flash128_specialized_casts;
                                    auto flash_q = profiler.measure("full.flash128_q_cast", [&]() {
                                        return specialized_casts
                                                   ? minicpm_flash128_q_cast_cuda(qh)
                                                   : qh.to(mfq_tensor_backend::kFloat32)
                                                         .contiguous();
                                    });
                                    auto flash_kv =
                                        profiler.measure("full.flash128_kv_cast", [&]() {
                                            if (specialized_casts) {
                                                return minicpm_flash128_kv_cast_cuda(kh, vh);
                                            }
                                            return std::vector<mfq_tensor_backend::Tensor>{
                                                kh.to(mfq_tensor_backend::kFloat16).contiguous(),
                                                vh.to(mfq_tensor_backend::kFloat16).contiguous()};
                                        });
                                    a = profiler.measure("full.flash128_kernel", [&]() {
                                        return mfq_attention_mma128_cuda(flash_q, flash_kv[0],
                                                                         flash_kv[1], attn_scale);
                                    });
                                    a = profiler.measure("full.flash128_output_cast", [&]() {
                                        return specialized_casts
                                                   ? minicpm_flash128_output_cast_cuda(a)
                                                   : a.to(mfq_tensor_backend::kBFloat16)
                                                         .contiguous();
                                    });
                                    attention_token_major = true;
                                } else {
                                    auto repeated_k = kh;
                                    auto repeated_v = vh;
                                    if (nkh != nh) {
                                        MFQ_RUNTIME_CHECK(
                                            nh % nkh == 0,
                                            "MiniCPM-o Qwen3 attention head ratio is invalid");
                                        const int64_t repeat = nh / nkh;
                                        repeated_k = kh.repeat_interleave(repeat, 1).contiguous();
                                        repeated_v = vh.repeat_interleave(repeat, 1).contiguous();
                                    }
                                    auto mask = minicpmo45_attention_mask(T, false);
                                    a = mfq_scaled_dot_product_attention(
                                        qh, repeated_k, repeated_v, mask, 0.0, !mask.has_value(),
                                        attn_scale, false);
                                }
                            } else {
                                if (!sliding && T % 256 == 0 && hd == 512 && nh == 8 * nkh &&
                                    mma_attention_enabled) {
                                    a = mfq_attention_mma512_cuda(
                                        q.to(mfq_tensor_backend::kFloat32).contiguous(), kh, vh,
                                        attn_scale);
                                    attention_token_major = true;
                                } else if (sliding && T >= 32 && hd == 256 && nh == 2 * nkh &&
                                           mma_attention_enabled) {
                                    a = mfq_attention_mma256_swa_cuda(
                                        q.to(mfq_tensor_backend::kFloat32).contiguous(), kh, vh,
                                        attn_scale, attention_window);
                                    attention_token_major = true;
                                } else if (!sliding && T >= 32 && hd == 256 && nh == 4 * nkh &&
                                           mma_attention_enabled && !seq_len.has_value() &&
                                           !attention_mask.has_value()) {
                                    a = attention_cached_prefill_cuda(
                                        q.to(mfq_tensor_backend::kFloat32).contiguous(), kh, vh,
                                        attn_scale, 0);
                                } else if (!sliding && T >= 32 && hd == 256 &&
                                           (nh == 4 * nkh || nh == 8 * nkh) &&
                                           mma_attention_enabled) {
                                    a = mfq_attention_mma256_cuda(
                                        q.to(mfq_tensor_backend::kFloat32).contiguous(), kh, vh,
                                        attn_scale);
                                    attention_token_major = true;
                                } else if (sliding) {
                                    a = attention_swa_cuda(qh, kh, vh, attn_scale,
                                                           attention_window);
                                } else {
                                    a = mfq_scaled_dot_product_attention(
                                        qh, kh, vh, std::nullopt, 0.0, true, attn_scale, true);
                                }
                            }
                        } else if (!official_bf16 && !sliding && T >= 32 && hd == 256 &&
                                   nh == 4 * nkh && mma_attention_enabled && !cache.is_paged() &&
                                   !seq_len.has_value() && !attention_mask.has_value()) {
                            a = attention_cached_prefill_cuda(qh, cache.k, cache.v,
                                                              attn_scale, cache_pos);
                        } else if (seq_len.has_value()) {
                            const int64_t planned_len =
                                planned_kv_length > 0 ? planned_kv_length : cache_pos + T;
                            const bool bf16_gqa_decode =
                                official_bf16 && T == 1 && execution.config.minicpm_bf16_gqa_decode;
                            const bool aten_decode_enabled = (official_bf16 && !bf16_gqa_decode) ||
                                                             execution.config.attention_decode_aten;
                            const bool mma_decode_enabled = execution.config.mma_attention_decode;
                            auto prepare_mma_decode_workspace = [&](int64_t visible_len,
                                                                    int64_t kv_tile) {
                                const int64_t mask_stride =
                                    (visible_len + kv_tile - 1) / kv_tile * kv_tile;
                                const int64_t ntiles_kv = (visible_len + kv_tile - 1) / kv_tile;
                                const int64_t max_blocks = B * nkh * ntiles_kv;
                                const int64_t meta_float2 = max_blocks * 8 * (2 + hd / 2);
                                auto cuda = mfq_tensor_backend::TensorOptions().device(
                                    mfq_tensor_backend::kCUDA);
                                if (!decode_mma_mask.defined() || decode_mma_mask.size(0) != B ||
                                    decode_mma_mask.size(1) < mask_stride) {
                                    decode_mma_mask = mfq_tensor_backend::empty(
                                        {B, mask_stride}, cuda.dtype(mfq_tensor_backend::kFloat16));
                                }
                                if (!decode_mma_kv_max.defined() || decode_mma_kv_max.numel() < B) {
                                    decode_mma_kv_max = mfq_tensor_backend::empty(
                                        {B}, cuda.dtype(mfq_tensor_backend::kInt32));
                                }
                                if (!decode_mma_meta.defined() ||
                                    decode_mma_meta.numel() < 2 * meta_float2) {
                                    decode_mma_meta = mfq_tensor_backend::empty(
                                        {2 * meta_float2},
                                        cuda.dtype(mfq_tensor_backend::kFloat32));
                                }
                            };
                            if (cache.is_paged()) {
                                const bool split_enabled =
                                    execution.config.attention_decode_split_k;
                                int64_t parts =
                                    split_enabled && cache_pos >= 192 ? (cache_pos + 127) / 128 : 1;
                                const bool dynamic_parts =
                                    split_enabled && decode_attention_parts > 1;
                                if (dynamic_parts)
                                    parts = decode_attention_parts;
                                parts = std::min<int64_t>(parts, kDecodeAttentionMaxParts);
                                a = attention_paged_cache_decode_cuda(
                                    qh, cache.k_chunk_ptrs, cache.v_chunk_ptrs, cache.page_table,
                                    seq_len.value(), attn_scale, cache.page_size,
                                    cache.pages_per_chunk, cache.paged_heads, decode_partial_o,
                                    decode_partial_m, decode_partial_l, parts, dynamic_parts);
                            } else if (aten_decode_enabled) {
                                const int64_t visible_len =
                                    sliding ? std::min<int64_t>(attention_window, cache_pos + T)
                                            : cache_pos + T;
                                auto cached_k =
                                    cache.k
                                        .index({Slice(), Slice(), Slice(0, visible_len), Slice()})
                                        .contiguous();
                                auto cached_v =
                                    cache.v
                                        .index({Slice(), Slice(), Slice(0, visible_len), Slice()})
                                        .contiguous();
                                auto mask =
                                    minicpmo45_attention_mask(visible_len, cache_pos > 0 && T > 1);
                                if (official_bf16 && nkh != nh) {
                                    MFQ_RUNTIME_CHECK(
                                        nh % nkh == 0,
                                        "MiniCPM-o Qwen3 attention head ratio is invalid");
                                    const int64_t repeat = nh / nkh;
                                    cached_k = cached_k.repeat_interleave(repeat, 1).contiguous();
                                    cached_v = cached_v.repeat_interleave(repeat, 1).contiguous();
                                }
                                a = mfq_scaled_dot_product_attention(qh, cached_k, cached_v, mask,
                                                                     0.0, false, attn_scale,
                                                                     !official_bf16);
                            } else if (sliding && T == 1 && mma_decode_enabled && hd == 256 &&
                                       nh == 2 * nkh) {
                                const int64_t visible_len =
                                    std::min<int64_t>(attention_window, planned_len);
                                prepare_mma_decode_workspace(visible_len, 64);
                                a = mfq_attention_mma256_swa_decode_cuda(
                                    q.to(mfq_tensor_backend::kFloat32).contiguous(), cache.k,
                                    cache.v, seq_len.value(), attn_scale, visible_len,
                                    decode_mma_mask, decode_mma_kv_max, decode_mma_meta);
                                attention_token_major = true;
                            } else if (!sliding && T == 1 && mma_decode_enabled && nh == 8 * nkh &&
                                       (hd == 256 || hd == 512)) {
                                const int64_t kv_tile = hd == 512 ? 32 : 64;
                                prepare_mma_decode_workspace(planned_len, kv_tile);
                                a = hd == 512 ? mfq_attention_mma512_decode_cuda(
                                                    q.to(mfq_tensor_backend::kFloat32).contiguous(),
                                                    cache.k, cache.v, seq_len.value(), attn_scale,
                                                    planned_len, decode_mma_mask, decode_mma_kv_max,
                                                    decode_mma_meta)
                                              : mfq_attention_mma256_decode_cuda(
                                                    q.to(mfq_tensor_backend::kFloat32).contiguous(),
                                                    cache.k, cache.v, seq_len.value(), attn_scale,
                                                    planned_len, decode_mma_mask, decode_mma_kv_max,
                                                    decode_mma_meta);
                                attention_token_major = true;
                            } else if (sliding) {
                                a = attention_cache_swa_planned_cuda(qh, cache.k, cache.v,
                                                                     seq_len.value(), attn_scale,
                                                                     attention_window, planned_len);
                            } else {
                                const bool split_enabled =
                                    execution.config.attention_decode_split_k;
                                int64_t parts =
                                    split_enabled && cache_pos >= 192 ? (cache_pos + 127) / 128 : 1;
                                if (split_enabled && decode_attention_parts > 0) {
                                    parts = decode_attention_parts;
                                }
                                parts = std::min<int64_t>(parts, kDecodeAttentionMaxParts);
                                const bool grouped_decode = split_enabled && !official_bf16 &&
                                    hd == 256 && nh == 4 * nkh && planned_len >= 8192 &&
                                    B == 1;
                                a = grouped_decode
                                        ? attention_cache_decode_grouped_cuda(
                                              qh, cache.k, cache.v, seq_len.value(), attn_scale,
                                              decode_partial_o, decode_partial_m, decode_partial_l,
                                              planned_len)
                                    : decode_attention_parts > 1
                                        ? attention_cache_decode_dynamic_cuda(
                                              qh, cache.k, cache.v, seq_len.value(), attn_scale,
                                              decode_partial_o, decode_partial_m, decode_partial_l,
                                              parts)
                                    : parts > 1
                                        ? attention_cache_decode_split_cuda(
                                              qh, cache.k, cache.v, seq_len.value(), attn_scale,
                                              decode_partial_o, decode_partial_m, decode_partial_l,
                                              parts)
                                        : attention_cache_decode_cuda(qh, cache.k, cache.v,
                                                                      seq_len.value(), attn_scale);
                            }
                        } else {
                            if (official_bf16) {
                                auto cached_k = kv.first.contiguous();
                                auto cached_v = kv.second.contiguous();
                                if (nkh != nh) {
                                    MFQ_RUNTIME_CHECK(
                                        nh % nkh == 0,
                                        "MiniCPM-o Qwen3 attention head ratio is invalid");
                                    const int64_t repeat = nh / nkh;
                                    cached_k = cached_k.repeat_interleave(repeat, 1).contiguous();
                                    cached_v = cached_v.repeat_interleave(repeat, 1).contiguous();
                                }
                                auto mask = minicpmo45_attention_mask(cached_k.size(2),
                                                                      cache_pos > 0 && T > 1);
                                a = mfq_scaled_dot_product_attention(
                                    qh, cached_k, cached_v, mask, 0.0,
                                    !mask.has_value() && cache_pos == 0 && T > 1, attn_scale,
                                    false);
                            } else {
                                a = sliding
                                        ? attention_swa_cuda(qh, kv.first.contiguous(),
                                                             kv.second.contiguous(), attn_scale,
                                                             attention_window)
                                        : attention_cuda(qh, kv.first.contiguous(),
                                                         kv.second.contiguous(), attn_scale, true);
                            }
                        }
                        return a;
                    });
                    return a;
                },
                [&](mfq_tensor_backend::Tensor a, std::optional<mfq_tensor_backend::Tensor> gate) {
                    auto q_gate = gate.value_or(mfq_tensor_backend::Tensor{});
                    mfq_tensor_backend::Tensor oo;
                    if (q_gate.defined()) {
                        trace_qwen_stage("qwen.attention", a, attention_token_major ? 1 : 2);
                        auto af = profiler.measure("full.attn_out_view", [&]() {
                            return attention_token_major
                                       ? a.reshape({B, T, attn_width})
                                       : a.transpose(1, 2).contiguous().reshape({B, T, attn_width});
                        });
                        auto gf = profiler.measure("full.q_gate_view", [&]() {
                            return q_gate.contiguous().reshape({B, T, attn_width});
                        });
                        oo = profiler.measure("full.o_proj_gate", [&]() {
                            return o.forward_input_mul(execution, af, gf, 1);
                        });
                    } else {
                        auto af = profiler.measure("full.attn_out_view", [&]() {
                            return attention_token_major
                                       ? a.reshape({B, T, attn_width})
                                       : a.transpose(1, 2).contiguous().reshape({B, T, attn_width});
                        });
                        oo = profiler.measure("full.o_proj", [&]() {
                            return official_bf16 ? o.forward_bf16_output(execution, af)
                                                 : o.forward(execution, af);
                        });
                    }
                    trace_qwen_stage("qwen.o_projection", oo);
                    if (official_bf16) {
                        oo = oo.to(mfq_tensor_backend::kBFloat16).contiguous();
                    }
                    return oo;
                },
                apply_rotary);
        },
        [&](mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor oo) {
            return forward_ffn(execution, std::move(residual), std::move(oo), B, T, H);
        });
}

mfq_tensor_backend::Tensor FullBlock::forward_ffn(CudaExecutionContext &execution,
                                                  mfq_tensor_backend::Tensor residual,
                                                  mfq_tensor_backend::Tensor attention_output,
                                                  int64_t B, int64_t T, int64_t H) {
    auto &profiler = execution.profiler;
    auto oo = std::move(attention_output);
    auto trace_qwen_stage = [&](const char *name, const mfq_tensor_backend::Tensor &value) {
        if (!attention_output_gate || execution.gemma_stage_trace == nullptr ||
            layer != execution.gemma_trace_layer)
            return;
        trace_gemma_stage(execution, layer, name, value.reshape({B, T, -1}));
    };
    return mfq::models::residual_ffn(
        std::move(residual), std::move(oo),
        [&](mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor oo) {
            auto attn_pair = profiler.measure("full.attn_residual_ffn_norm", [&]() {
                auto rr = residual.reshape({-1, H});
                auto oo2 = oo.reshape({-1, H});
                if (!rr.is_cuda()) {
                    auto summed =
                        (rr.to(mfq_tensor_backend::kFloat32) + oo2.to(mfq_tensor_backend::kFloat32))
                            .to(rr.scalar_type())
                            .contiguous();
                    auto normalized =
                        official_bf16 ? qwen_rms_norm_bf16(execution.config, summed, ffn_norm,
                                                           rms_norm_eps, norm_weight_offset)
                                      : qwen_rms_norm(summed.to(mfq_tensor_backend::kFloat32),
                                                      ffn_norm, rms_norm_eps, norm_weight_offset);
                    return std::vector<mfq_tensor_backend::Tensor>{summed, normalized};
                }
                if (oo2.scalar_type() != rr.scalar_type()) {
                    oo2 = oo2.to(rr.scalar_type()).contiguous();
                }
                if (execution.config.diagnostic_fp32_residual) {
                    return acc_rms_norm_cuda(rr.to(mfq_tensor_backend::kFloat32),
                                             oo2.to(mfq_tensor_backend::kFloat32), ffn_norm,
                                             rms_norm_eps, norm_weight_offset);
                }
                if (rr.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    oo2.scalar_type() == mfq_tensor_backend::kFloat16) {
                    return acc_rms_norm_f16_cuda(rr, oo2, ffn_norm, rms_norm_eps,
                                                 norm_weight_offset);
                }
                if (rr.scalar_type() == mfq_tensor_backend::kBFloat16 &&
                    oo2.scalar_type() == mfq_tensor_backend::kBFloat16) {
                    if (official_bf16) {
                        return acc_rms_norm_bf16_cuda(rr, oo2, ffn_norm, rms_norm_eps,
                                                      norm_weight_offset);
                    }
                    auto sum = (rr + oo2).contiguous();
                    auto norm = qwen_rms_norm(sum.to(mfq_tensor_backend::kFloat32), ffn_norm,
                                              rms_norm_eps, norm_weight_offset)
                                    .to(mfq_tensor_backend::kBFloat16)
                                    .contiguous();
                    return std::vector<mfq_tensor_backend::Tensor>{sum, norm};
                }
                return acc_rms_norm_cuda(rr, oo2, ffn_norm, rms_norm_eps, norm_weight_offset);
            });
            return std::array<mfq_tensor_backend::Tensor, 2>{attn_pair[0].reshape({B, T, H}),
                                                             attn_pair[1].reshape({B, T, H})};
        },
        [&](mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor xn) {
            trace_qwen_stage("qwen.ffn_norm", xn);
            return mfq::models::feed_forward(
                std::move(residual), std::move(xn),
                [&](mfq_tensor_backend::Tensor xn, mfq_tensor_backend::Tensor residual)
                    -> std::optional<mfq_tensor_backend::Tensor> {
                    if (!official_bf16) {
                        auto ffn_input = xn.reshape({B * T, H});
                        auto residual_flat = residual.reshape({B * T, H});
                        if (ffn.can_forward_fused_residual(execution.config, ffn_input,
                                                           residual_flat)) {
                            return profiler.measure("full.ffn_down_residual", [&]() {
                                return ffn
                                    .forward_fused_residual(profiler, execution.config, ffn_input,
                                                            residual_flat)
                                    .reshape({B, T, H});
                            });
                        }
                    }
                    return {};
                },
                [&](mfq_tensor_backend::Tensor xn) {
                    mfq_tensor_backend::Tensor ff;
                    if (official_bf16) {
                        auto ffn_input = xn.reshape({B * T, H});
                        using Tensor = mfq_tensor_backend::Tensor;
                        using Activation = mfq::models::GatedActivation;
                        ff = mfq::models::gated_mlp(
                                 ffn_input, false, 0.0,
                                 [](const Tensor &, Activation, double) {
                                     return std::optional<Tensor>{};
                                 },
                                 [](const Tensor &, Activation, double) {
                                     return std::optional<Tensor>{};
                                 },
                                 [&](Tensor ffn_input) {
                                     return profiler.measure("full.minicpmo45_ffn_gate_up", [&]() {
                                         return ffn.gate_up.forward(execution, ffn_input);
                                     });
                                 },
                                 [&](Tensor gate, Tensor up, Activation, double) {
                                     gate = gate.to(mfq_tensor_backend::kBFloat16).contiguous();
                                     up = up.to(mfq_tensor_backend::kBFloat16).contiguous();
                                     return profiler.measure("full.minicpmo45_ffn_swiglu", [&]() {
                                         if (execution.config.minicpm_bf16_swiglu_fusion) {
                                             return silu_mul_cuda(gate, up);
                                         }
                                         return (mfq_tensor_backend::silu(gate) * up).contiguous();
                                     });
                                 },
                                 [&](Tensor activation) {
                                     return profiler.measure("full.minicpmo45_ffn_down", [&] {
                                         return ffn.down.forward_bf16_output(execution, activation);
                                     });
                                 },
                                 [](const Tensor &, const Tensor &, Activation, double) {
                                     return std::optional<Tensor>{};
                                 })
                                 .reshape({B, T, H})
                                 .to(mfq_tensor_backend::kBFloat16)
                                 .contiguous();
                    } else {
                        ff = ffn.forward(execution, xn.reshape({B * T, H})).reshape({B, T, H});
                    }

                    return ff;
                },
                [&](mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor ff) {
                    trace_qwen_stage("qwen.ffn_output", ff);
                    auto output = profiler.measure("full.ffn_residual", [&]() {
                        auto rr = residual.reshape({-1, H});
                        auto ff2 = ff.reshape({-1, H});
                        if (!rr.is_cuda()) {
                            return (rr.to(mfq_tensor_backend::kFloat32) +
                                    ff2.to(mfq_tensor_backend::kFloat32))
                                .to(rr.scalar_type())
                                .reshape({B, T, H})
                                .contiguous();
                        }
                        if (execution.config.diagnostic_fp32_residual) {
                            rr = rr.to(mfq_tensor_backend::kFloat32);
                            ff2 = ff2.to(mfq_tensor_backend::kFloat32);
                        } else if (ff2.scalar_type() != rr.scalar_type()) {
                            ff2 = ff2.to(rr.scalar_type()).contiguous();
                        }
                        if (rr.scalar_type() == mfq_tensor_backend::kBFloat16 &&
                            ff2.scalar_type() == mfq_tensor_backend::kBFloat16) {
                            if (execution.config.minicpm_bf16_residual_acc) {
                                return acc_cuda(rr, ff2).reshape({B, T, H});
                            }
                            return (rr + ff2).contiguous().reshape({B, T, H});
                        }
                        return acc_cuda(rr, ff2).reshape({B, T, H});
                    });
                    return output;
                });
        });
}

mfq_tensor_backend::Tensor qwen_rms_norm(mfq_tensor_backend::Tensor x,
                                         mfq_tensor_backend::Tensor weight, double eps,
                                         double weight_offset) {
    if (!x.is_cuda()) {
        auto xf = x.contiguous().to(mfq_tensor_backend::kFloat32);
        auto wf = weight.contiguous().to(mfq_tensor_backend::kFloat32);
        if (weight_offset != 0.0)
            wf = wf + weight_offset;
        auto inverse = mfq_tensor_backend::rsqrt(xf.square().mean(-1, true) + eps);
        return (xf * inverse * wf).contiguous();
    }
    return rms_norm_offset_cuda(x, weight, eps, weight_offset);
}

mfq_tensor_backend::Tensor qwen_rms_norm_bf16(const CudaExecutionConfig &config,
                                              mfq_tensor_backend::Tensor x,
                                              mfq_tensor_backend::Tensor weight, double eps,
                                              double weight_offset) {
    auto input = x.contiguous().to(mfq_tensor_backend::kBFloat16);
    if (input.is_cuda() && config.minicpm_fused_bf16_rmsnorm) {
        return qwen_rms_norm_bf16_cuda(input, weight.contiguous(), eps, weight_offset);
    }
    auto xf = input.to(mfq_tensor_backend::kFloat32);
    auto inverse = mfq_tensor_backend::rsqrt(xf.square().mean(-1, true) + eps);
    auto normalized = (xf * inverse).to(mfq_tensor_backend::kBFloat16);
    auto scale = weight.contiguous().to(mfq_tensor_backend::kBFloat16);
    if (weight_offset != 0.0)
        scale = scale + weight_offset;
    return (scale * normalized).contiguous();
}
