#pragma once

// Shared stateful Flash-Next building blocks. Keep checkpoint ownership here,
// outside quantized projections and the stateless CUDA compute primitives.
#include "mfq/kernels/cuda/flash_next.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <vector>

std::vector<mfq_tensor_backend::Tensor> gdn_cuda(
    mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor,
    mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor,
    mfq_tensor_backend::Tensor, MfqOptional<mfq_tensor_backend::Tensor>);
mfq_tensor_backend::Tensor ssm_conv_silu_cuda(
    mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor,
    mfq_tensor_backend::Tensor, int64_t);

namespace mfq::flash_next {
namespace tb = mfq_tensor_backend;
using Tensor = tb::Tensor;
using Linear = std::function<Tensor(const Tensor&)>;

inline Tensor rms_norm(const Tensor& value, const Tensor& weight, double eps) {
    auto f = value.to(tb::kFloat32);
    return (f * tb::rsqrt((f * f).mean(-1, true) + eps) * weight.to(tb::kFloat32))
        .to(value.scalar_type());
}

class SequenceCache {
public:
    SequenceCache(int64_t maximum, int64_t width)
        : maximum_(maximum), width_(width) {
        MFQ_RUNTIME_CHECK(maximum > 0 && width > 0, "invalid Flash-Next cache dimensions");
    }
    void reset() { values_ = {}; position_ = 0; }
    int64_t position() const { return position_; }
    const Tensor& storage() const { return values_; }
    void truncate(int64_t keep) {
        MFQ_RUNTIME_CHECK(keep >= 0 && keep <= position_, "invalid Flash-Next cache truncation");
        position_ = keep;
    }
    Tensor append(const Tensor& value) {
        MFQ_RUNTIME_CHECK(value.is_cuda() && value.dim() == 3 && value.size(0) > 0 &&
            value.size(1) > 0 && value.size(2) == width_, "invalid Flash-Next cache append");
        MFQ_RUNTIME_CHECK(!values_.defined() || (value.size(0) == values_.size(0) &&
            value.device() == values_.device()), "reset Flash-Next cache before changing batch/device");
        MFQ_RUNTIME_CHECK(value.size(1) <= maximum_ - position_, "Flash-Next cache exceeds max_context");
        const auto required = position_ + value.size(1);
        if (!values_.defined() || required > values_.size(1)) {
            int64_t capacity = values_.defined() ? values_.size(1) : std::min<int64_t>(16, maximum_);
            while (capacity < required) capacity = std::min(maximum_, capacity > maximum_ / 2 ? maximum_ : capacity * 2);
            auto expanded = tb::zeros({value.size(0), capacity, width_}, value.options().dtype(tb::kFloat16));
            if (position_) expanded.narrow(1, 0, position_).copy_(values_.narrow(1, 0, position_));
            values_ = std::move(expanded);
        }
        values_.narrow(1, position_, value.size(1)).copy_(value.to(tb::kFloat16));
        position_ = required;
        return values_.narrow(1, 0, position_);
    }
private:
    int64_t maximum_, width_, position_ = 0;
    Tensor values_;
};

// Both public graphs select complete visible pools, expand their token IDs,
// pad the fixed budget with -1, and optionally append the incomplete tail.
// argpartition's order/tie order is unspecified in MLX; CUDA top-k selects the
// same set for distinct scores. Attention receives this explicit selected order.
inline Tensor select_pooled_blocks(const Tensor& scores, int64_t query_offset,
    int64_t logical_length, int64_t pool, int64_t budget, bool include_tail) {
    MFQ_RUNTIME_CHECK(scores.is_cuda() && scores.dim() == 3 && pool > 0 && budget > 0 &&
        budget % pool == 0 && query_offset >= 0 && logical_length >= query_offset &&
        scores.size(1) <= logical_length - query_offset && scores.size(2) == logical_length / pool,
        "Flash-Next pool selection geometry mismatch");
    const auto b = scores.size(0), t = scores.size(1), pools = scores.size(2);
    const auto options = scores.options().dtype(tb::kInt64);
    auto absolute = tb::arange(t, options) + query_offset;
    auto selected = tb::full({b,t,budget}, -1, options);
    const auto count = std::min(budget / pool, pools);
    if (count) {
        auto ends = tb::arange(pools, options) * pool + (pool - 1);
        auto visible = (ends.reshape({1,1,pools}) <= absolute.reshape({1,t,1})).expand({b,t,pools});
        auto ranked = tb::where(visible, scores.to(tb::kFloat32), tb::full_like(scores, -1e30).to(tb::kFloat32));
        auto indices = std::get<1>(tb::topk(ranked, count, -1, true, false));
        auto valid = visible.gather(-1, indices).unsqueeze(-1).expand({b,t,count,pool});
        auto expanded = indices.unsqueeze(-1) * pool + tb::arange(pool, options);
        expanded = tb::where(valid, expanded, tb::full_like(expanded, -1));
        selected.narrow(-1, 0, count * pool).copy_(expanded.reshape({b,t,count * pool}));
    }
    if (include_tail && pool > 1) {
        auto count = (absolute + 1).remainder(pool);
        auto offsets = tb::arange(pool - 1, options).unsqueeze(0);
        auto tail = (absolute + 1 - count).unsqueeze(1) + offsets;
        tail = tb::where(offsets < count.unsqueeze(1), tail, tb::full_like(tail, -1));
        selected = tb::cat({selected, tail.unsqueeze(0).expand({b,t,pool - 1})}, -1);
    }
    return selected.to(tb::kInt32).contiguous();
}

struct KdaWeights {
    Linear query, key, value, beta, gate_a, gate_b, output;
    Tensor conv, forget_a, forget_b, dt_bias, a_log, output_norm;
};

class Kda {
public:
    Kda(KdaWeights weights, int64_t heads, int64_t width, int64_t kernel,
        double lower_bound, double eps)
        : w_(std::move(weights)), heads_(heads), width_(width), kernel_(kernel),
          lower_bound_(lower_bound), eps_(eps) {
        MFQ_RUNTIME_CHECK(heads > 0 && width > 0 && kernel > 1 &&
            std::isfinite(lower_bound) && std::isfinite(eps) && eps > 0,
            "invalid GLM KDA configuration");
    }
    void reset() {
        conv_ = {}; recurrent_ = {}; commit();
    }
    const Tensor& conv_state() const { return conv_; }
    const Tensor& recurrent_state() const { return recurrent_; }
    bool pending() const { return rollback_conv_.defined(); }
    void commit() { rollback_conv_ = {}; rollback_recurrent_ = {}; }
    void rollback() {
        MFQ_RUNTIME_CHECK(pending(), "GLM KDA has no speculative checkpoint");
        conv_ = rollback_conv_; recurrent_ = rollback_recurrent_; commit();
    }
    Tensor forward(const Tensor& hidden, bool use_cache, int64_t confirmed = 0) {
        MFQ_RUNTIME_CHECK(hidden.is_cuda() && hidden.dim() == 3 && hidden.size(0) > 0 &&
            hidden.size(1) > 0 && confirmed >= 0 && confirmed <= hidden.size(1) &&
            (!confirmed || use_cache), "invalid GLM KDA input/verification geometry");
        if (confirmed > 0 && confirmed < hidden.size(1)) {
            MFQ_RUNTIME_CHECK(!pending(), "resolve previous GLM KDA speculative checkpoint");
            auto prefix = forward(hidden.narrow(1, 0, confirmed), true);
            // Kernels produce fresh state: retaining these references is enough.
            rollback_conv_ = conv_; rollback_recurrent_ = recurrent_;
            try {
                auto suffix = forward(hidden.narrow(1, confirmed, hidden.size(1) - confirmed), true);
                return tb::cat({prefix, suffix}, 1);
            } catch (...) { rollback(); throw; }
        }
        const auto b = hidden.size(0), t = hidden.size(1), channels = heads_ * width_;
        const auto options = hidden.options().dtype(tb::kFloat32);
        MFQ_RUNTIME_CHECK(!use_cache || !conv_.defined() ||
            (conv_.size(0) == b && conv_.device() == hidden.device()),
            "reset GLM KDA before changing batch/device");
        auto previous = use_cache && conv_.defined() ? conv_ :
            tb::zeros({b,kernel_ - 1,3 * channels}, options);
        auto raw = tb::cat({w_.query(hidden), w_.key(hidden), w_.value(hidden)}, -1).to(tb::kFloat32);
        MFQ_RUNTIME_CHECK(raw.sizes().vec() == std::vector<int64_t>({b,t,3 * channels}),
            "GLM KDA projection shape mismatch");
        auto joined = tb::cat({previous, raw}, 1).contiguous();
        auto convolved = ssm_conv_silu_cuda(joined, w_.conv.to(tb::kFloat32).contiguous(),
            tb::empty({0}, options), t);
        auto next_conv = joined.narrow(1, t, kernel_ - 1).contiguous();
        const auto heads = [&](int64_t begin) {
            return convolved.narrow(-1, begin, channels).reshape({b,t,heads_,width_}).permute({0,2,1,3});
        };
        const auto normalize = [&](const Tensor& x) {
            return (x / tb::clamp_min((x * x).sum(-1, true).sqrt(), 1e-6)).contiguous();
        };
        auto q = normalize(heads(0)), k = normalize(heads(channels)), v = heads(2 * channels).contiguous();
        auto forget = mfq_flash_next::glm5_kda_forget_gate(hidden, w_.forget_a, w_.forget_b,
            w_.dt_bias, w_.a_log, heads_, width_, lower_bound_).permute({0,2,1,3}).contiguous();
        auto beta = tb::sigmoid(w_.beta(hidden).to(tb::kFloat32)).transpose(1,2).contiguous();
        Tensor attended, next_recurrent;
        auto initial = use_cache && recurrent_.defined() ? recurrent_ : Tensor{};
        if (width_ == 32 || width_ == 64 || width_ == 128) {
            auto result = gdn_cuda(q,k,v,forget,beta, initial.defined() ?
                MfqOptional<Tensor>(initial) : mfq_nullopt);
            attended = result[0]; next_recurrent = result[1];
        } else {
            // Public model dimensions use the fused kernel. Keep the exact
            // recurrence available for small independent architecture fixtures.
            next_recurrent = initial.defined() ? initial : tb::zeros({b,heads_,width_,width_}, options);
            std::vector<Tensor> steps;
            for (int64_t i = 0; i < t; ++i) {
                auto ki = k.select(2,i);
                next_recurrent = next_recurrent * forget.select(2,i).exp().unsqueeze(-1);
                auto predicted = (next_recurrent * ki.unsqueeze(-1)).sum(-2);
                auto delta = (v.select(2,i) - predicted) * beta.select(2,i).unsqueeze(-1);
                next_recurrent = next_recurrent + ki.unsqueeze(-1) * delta.unsqueeze(-2);
                steps.push_back((next_recurrent * (q.select(2,i) / std::sqrt(double(width_))).unsqueeze(-1)).sum(-2).unsqueeze(2));
            }
            attended = tb::cat(steps, 2);
        }
        auto gate = w_.gate_b(w_.gate_a(hidden)).reshape({b,t,heads_,width_}).permute({0,2,1,3});
        auto normalized = rms_norm(attended, w_.output_norm, eps_) * tb::sigmoid(gate.to(tb::kFloat32));
        auto result = w_.output(normalized.permute({0,2,1,3}).reshape({b,t,channels}).to(hidden.scalar_type()));
        // Publish state only after the whole branch succeeds.
        if (use_cache) { conv_ = std::move(next_conv); recurrent_ = std::move(next_recurrent); }
        return result;
    }
private:
    KdaWeights w_;
    int64_t heads_, width_, kernel_;
    double lower_bound_, eps_;
    Tensor conv_, recurrent_, rollback_conv_, rollback_recurrent_;
};

struct MlaWeights {
    Linear query_a, key_value_a, query_b, output, index_query, index_key, index_score;
    // Head-wise absorbed projections accept/return [B,T,heads,width].
    Linear embed_query, unembed_output;
    Tensor query_norm, latent_norm, index_norm, index_bias, index_gate, index_position;
};

struct MlaConfig {
    int64_t heads, nope, latent, value_width, index_heads, index_width, pool, budget, maximum;
    bool tail;
    double eps;
};

class SparseMla {
public:
    SparseMla(MlaWeights weights, MlaConfig config)
        : w_(std::move(weights)), c_(config), latent_(config.maximum, config.latent),
          index_(config.maximum, 2 * config.index_width) {
        MFQ_RUNTIME_CHECK(c_.heads > 0 && c_.nope > 0 && c_.value_width > 0 &&
            c_.index_heads > 0 && c_.pool > 0 && c_.budget > 0 && c_.budget % c_.pool == 0,
            "invalid Flash-Next MLA configuration");
    }
    void reset() { latent_.reset(); index_.reset(); }
    int64_t position() const { return latent_.position(); }
    void truncate(int64_t keep) {
        MFQ_RUNTIME_CHECK(keep >= 0 && keep <= latent_.position() && keep <= index_.position(),
            "invalid Flash-Next MLA cache truncation");
        latent_.truncate(keep); index_.truncate(keep);
    }
    Tensor forward(const Tensor& hidden, bool use_cache) {
        MFQ_RUNTIME_CHECK(hidden.is_cuda() && hidden.dim() == 3 && hidden.size(0) > 0 && hidden.size(1) > 0,
            "GLM MLA requires nonempty [B,T,H] input");
        const auto b = hidden.size(0), t = hidden.size(1), offset = use_cache ? position() : 0;
        MFQ_RUNTIME_CHECK(offset == (use_cache ? index_.position() : 0) && t <= c_.maximum - offset,
            "GLM MLA cache position/capacity mismatch");
        auto qr = rms_norm(w_.query_a(hidden), w_.query_norm, c_.eps);
        auto query = w_.query_b(qr).reshape({b,t,c_.heads,c_.nope});
        auto latent = rms_norm(w_.key_value_a(hidden).narrow(-1, 0, c_.latent), w_.latent_norm, c_.eps);
        // Metal's indexer LayerNorm explicitly casts the source/output to F16.
        auto ik = w_.index_key(hidden).to(tb::kFloat16).to(tb::kFloat32);
        auto centered = ik - ik.mean(-1, true);
        ik = (centered * tb::rsqrt((centered * centered).mean(-1, true) + 1e-6) *
            w_.index_norm.to(tb::kFloat32) + w_.index_bias.to(tb::kFloat32)).to(tb::kFloat16);
        auto gate_dtype = hidden.scalar_type() == w_.index_gate.scalar_type() ? hidden.scalar_type() : tb::kFloat32;
        auto gates = tb::matmul(hidden.to(gate_dtype), w_.index_gate.to(gate_dtype).transpose(-1,-2));
        auto packed = tb::cat({ik.to(gates.scalar_type()), gates}, -1);
        try {
            if (use_cache) { latent = latent_.append(latent); packed = index_.append(packed); }
            auto absorbed = w_.embed_query(query).permute({0,2,1,3}).to(tb::kFloat32);
            Tensor attended;
            const double scale = 1.0 / std::sqrt(double(c_.nope));
            if (latent.size(1) <= c_.budget) {
                attended = mfq_flash_next::glm5_dense_mla_attention(absorbed, latent, offset, scale);
            } else {
                auto pooled = mfq_flash_next::glm5_kpool_states(packed.narrow(-1,0,c_.index_width),
                    packed.narrow(-1,c_.index_width,c_.index_width), w_.index_position, c_.pool);
                auto iq = w_.index_query(qr).reshape({b,t,c_.index_heads,c_.index_width});
                auto scores = mfq_flash_next::glm5_kpool_scores(iq, pooled, w_.index_score(hidden));
                auto selected = select_pooled_blocks(scores, offset, latent.size(1), c_.pool, c_.budget, c_.tail);
                attended = mfq_flash_next::glm5_sparse_mla_attention(absorbed, latent, selected, scale);
            }
            auto value = w_.unembed_output(attended.to(tb::kFloat16));
            return w_.output(value.reshape({b,t,c_.heads*c_.value_width}));
        } catch (...) {
            if (use_cache) { latent_.truncate(offset); index_.truncate(offset); }
            throw;
        }
    }
private:
    MlaWeights w_;
    MlaConfig c_;
    SequenceCache latent_, index_;
};
} // namespace mfq::flash_next
