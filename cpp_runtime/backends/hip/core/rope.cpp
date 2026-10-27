#include "hip/hip_runtime.h"
#include "../runtime/execution_options.h"
#include "rope.h"

#include "hip_execution.h"
#include "mfq_hip_attention_ops.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

RopeCache::RopeCache(
        int64_t max_positions,
        int64_t dim,
        double base,
        int64_t frequency_dim,
        int64_t active_pairs,
        mfq_tensor_backend::Device device,
        bool official_reciprocal_frequencies)
    : rotary_dim(dim) {
    int64_t half = rotary_dim / 2;
    const int64_t denominator = frequency_dim > 0 ? frequency_dim : rotary_dim;
    if (active_pairs < 0) active_pairs = half;
    if (active_pairs > half) {
        throw std::runtime_error("RoPE active pair count exceeds rotary dimension");
    }
    auto opts = mfq_tensor_backend::TensorOptions().device(device).dtype(mfq_tensor_backend::kFloat32);
    auto pos = mfq_tensor_backend::arange(max_positions, opts);
    auto ar = mfq_tensor_backend::arange(0, rotary_dim, 2, opts);
    auto freq = mfq_tensor_backend::pow(
        mfq_tensor_backend::full({half}, base, opts),
        -ar / (double)denominator);
    if (official_reciprocal_frequencies) {
        auto cpu_opts = mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32);
#ifdef MFQ_NATIVE_CUDA_RUNTIME
        std::vector<float> official_values(static_cast<size_t>(half));
        const float official_base = static_cast<float>(base);
        const float official_denominator = static_cast<float>(denominator);
        for (int64_t index = 0; index < half; ++index) {
            const float exponent =
                static_cast<float>(index * 2) / official_denominator;
            official_values[static_cast<size_t>(index)] =
                1.0f / std::pow(official_base, exponent);
        }
        auto official_freq = mfq_tensor_backend::tensor(
            official_values, cpu_opts);
#else
        auto exponent = mfq_tensor_backend::arange(0, rotary_dim, 2, cpu_opts) /
            (double)denominator;
        auto official_freq = mfq_tensor_backend::reciprocal(
            mfq_tensor_backend::pow(mfq_tensor_backend::full({half}, base, cpu_opts), exponent));
#endif
        freq.copy_(official_freq);
    }
    if (active_pairs < half) {
        auto pair = mfq_tensor_backend::arange(half, opts);
        freq = mfq_tensor_backend::where(pair < active_pairs, freq, mfq_tensor_backend::zeros_like(freq));
    }
    auto ang = pos.unsqueeze(1) * freq.unsqueeze(0);
    cos = mfq_tensor_backend::cos(ang).contiguous();
    sin = mfq_tensor_backend::sin(ang).contiguous();
    sections = mfq_tensor_backend::empty({0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt64).device(mfq_tensor_backend::kCPU));
    empty_sections = sections;
}

void RopeCache::configure_mrope(
        const std::vector<int64_t>& configured_sections,
        bool interleaved,
        int64_t configured_rotary_dim,
        mfq_tensor_backend::Device device) {
        if (configured_sections.empty()) return;
        auto execution_sections = configured_sections;
        if (interleaved) {
            std::array<std::vector<int64_t>, 3> pairs;
            for (int64_t pair = 0; pair < configured_rotary_dim / 2; ++pair) {
                const int64_t residue = pair % 3;
                const int axis = residue == 1 &&
                        pair < configured_sections[1] * 3
                    ? 1
                    : (residue == 2 && pair < configured_sections[2] * 3
                        ? 2 : 0);
                pairs[static_cast<size_t>(axis)].push_back(pair);
            }
            execution_sections = {
                static_cast<int64_t>(pairs[0].size()),
                static_cast<int64_t>(pairs[1].size()),
                static_cast<int64_t>(pairs[2].size())};
            std::vector<int64_t> order;
            for (const auto& axis : pairs) {
                order.insert(order.end(), axis.begin(), axis.end());
            }
            std::vector<int64_t> inverse(order.size());
            for (size_t index = 0; index < order.size(); ++index) {
                inverse[static_cast<size_t>(order[index])] =
                    static_cast<int64_t>(index);
            }
            interleaved_order = mfq_tensor_backend::tensor(
                order, mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCPU)).to(device);
            interleaved_inverse = mfq_tensor_backend::tensor(
                inverse, mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64)
                    .device(mfq_tensor_backend::kCPU)).to(device);
        }
        sections = mfq_tensor_backend::tensor(
            execution_sections,
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCPU));
    }

mfq_tensor_backend::Tensor RopeCache::apply(
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor pos,
        bool grid_mrope_positions) const {
        if (!x.is_cuda()) {
            MFQ_RUNTIME_CHECK(
                !cos.is_cuda() && !sin.is_cuda(),
                "CPU RoPE requires a CPU-resident table");
            auto positions = pos.contiguous().to(mfq_tensor_backend::kCPU, mfq_tensor_backend::kInt64)
                .clamp(0, cos.size(0) - 1);
            mfq_tensor_backend::Tensor selected_cos;
            mfq_tensor_backend::Tensor selected_sin;
            if (positions.dim() == 1) {
                selected_cos = cos.index_select(0, positions);
                selected_sin = sin.index_select(0, positions);
            } else if (!grid_mrope_positions && positions.dim() == 2) {
                const int64_t batches = positions.size(0);
                const int64_t tokens = positions.size(1);
                selected_cos = cos.index_select(0, positions.reshape({-1}))
                    .reshape({batches, tokens, rotary_dim / 2})
                    .unsqueeze(1);
                selected_sin = sin.index_select(0, positions.reshape({-1}))
                    .reshape({batches, tokens, rotary_dim / 2})
                    .unsqueeze(1);
            } else if (grid_mrope_positions && sections.numel() == 3) {
                const int64_t * section = sections.data_ptr<int64_t>();
                std::vector<mfq_tensor_backend::Tensor> cos_parts;
                std::vector<mfq_tensor_backend::Tensor> sin_parts;
                int64_t offset = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    const int64_t count = section[axis];
                    auto active = positions.select(0, axis);
                    cos_parts.push_back(
                        cos.index_select(0, active).narrow(1, offset, count));
                    sin_parts.push_back(
                        sin.index_select(0, active).narrow(1, offset, count));
                    offset += count;
                }
                selected_cos = mfq_tensor_backend::cat(cos_parts, 1);
                selected_sin = mfq_tensor_backend::cat(sin_parts, 1);
            } else {
                throw std::runtime_error(
                    "multi-axis CPU RoPE requires three position sections");
            }
            auto xf = x.contiguous().to(mfq_tensor_backend::kFloat32);
            const int64_t half = rotary_dim / 2;
            auto first = xf.narrow(-1, 0, half);
            auto second = xf.narrow(-1, half, half);
            auto output = xf.clone();
            output.narrow(-1, 0, half).copy_(
                first * selected_cos - second * selected_sin);
            output.narrow(-1, half, half).copy_(
                second * selected_cos + first * selected_sin);
            return output;
        }
        auto source = x.contiguous().to(mfq_tensor_backend::kFloat32);
        auto active_cos = cos;
        auto active_sin = sin;
        const bool interleaved = grid_mrope_positions &&
            interleaved_order.defined();
        if (interleaved) {
            const int64_t half = rotary_dim / 2;
            std::vector<mfq_tensor_backend::Tensor> parts{
                source.narrow(-1, 0, half).index_select(-1, interleaved_order),
                source.narrow(-1, half, half).index_select(-1, interleaved_order)};
            if (source.size(-1) > rotary_dim) {
                parts.push_back(source.narrow(
                    -1, rotary_dim, source.size(-1) - rotary_dim));
            }
            source = mfq_tensor_backend::cat(parts, -1).contiguous();
            active_cos = cos.index_select(1, interleaved_order).contiguous();
            active_sin = sin.index_select(1, interleaved_order).contiguous();
        }
        auto output = rope_table_cuda(
            source,
            pos.contiguous().to(
                mfq_tensor_backend::kCUDA, mfq_tensor_backend::kInt64),
            active_cos, active_sin, rotary_dim,
            grid_mrope_positions ? sections : empty_sections);
        if (interleaved) {
            const int64_t half = rotary_dim / 2;
            std::vector<mfq_tensor_backend::Tensor> parts{
                output.narrow(-1, 0, half).index_select(
                    -1, interleaved_inverse),
                output.narrow(-1, half, half).index_select(
                    -1, interleaved_inverse)};
            if (output.size(-1) > rotary_dim) {
                parts.push_back(output.narrow(
                    -1, rotary_dim, output.size(-1) - rotary_dim));
            }
            output = mfq_tensor_backend::cat(parts, -1).contiguous();
        }
        return output;
    }

mfq_tensor_backend::Tensor RopeCache::apply_bf16(
        const CudaExecutionConfig& config,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor pos) const {
        MFQ_RUNTIME_CHECK(
            sections.numel() == 0,
            "Qwen3 BF16 RoPE does not support multi-axis sections");
        if (x.is_cuda() && pos.dim() == 1 &&
                config.minicpm_fused_bf16_rope) {
            return rope_table_bf16_cuda(
                x.contiguous().to(mfq_tensor_backend::kBFloat16),
                pos.contiguous().to(cos.device(), mfq_tensor_backend::kInt64),
                cos, sin, rotary_dim);
        }
        auto positions = pos.contiguous().to(cos.device(), mfq_tensor_backend::kInt64)
            .clamp(0, cos.size(0) - 1);
        mfq_tensor_backend::Tensor selected_cos;
        mfq_tensor_backend::Tensor selected_sin;
        if (positions.dim() == 1) {
            selected_cos = cos.index_select(0, positions)
                .to(mfq_tensor_backend::kBFloat16).unsqueeze(0).unsqueeze(0);
            selected_sin = sin.index_select(0, positions)
                .to(mfq_tensor_backend::kBFloat16).unsqueeze(0).unsqueeze(0);
        } else if (positions.dim() == 2) {
            const int64_t batches = positions.size(0);
            const int64_t tokens = positions.size(1);
            selected_cos = cos.index_select(0, positions.reshape({-1}))
                .reshape({batches, tokens, rotary_dim / 2})
                .to(mfq_tensor_backend::kBFloat16).unsqueeze(1);
            selected_sin = sin.index_select(0, positions.reshape({-1}))
                .reshape({batches, tokens, rotary_dim / 2})
                .to(mfq_tensor_backend::kBFloat16).unsqueeze(1);
        } else {
            throw std::runtime_error(
                "Qwen3 BF16 RoPE positions must be rank one or two");
        }
        auto xb = x.contiguous().to(mfq_tensor_backend::kBFloat16);
        const int64_t half = rotary_dim / 2;
        auto first = xb.narrow(-1, 0, half);
        auto second = xb.narrow(-1, half, half);
        auto output = xb.clone();
        output.narrow(-1, 0, half).copy_(
            first * selected_cos - second * selected_sin);
        output.narrow(-1, half, half).copy_(
            second * selected_cos + first * selected_sin);
        return output.contiguous();
    }

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;
using Tensor = tb::Tensor;

RotaryEmbedding::RotaryEmbedding(int64_t dimension, int64_t maximum, double base,
                               std::vector<int64_t> sections, bool interleaved,
                               std::optional<bool> fused)
    : dimension_(dimension), maximum_(maximum), base_(base), sections_(std::move(sections)),
      interleaved_(interleaved) {
    const auto configured = runtime_options::rotary_fusion();
#ifdef MFQ_NATIVE_CUDA_RUNTIME
    constexpr bool default_fused = true;
#else
    constexpr bool default_fused = false;
#endif
    fused_ = fused.value_or(configured.value_or(default_fused));
    MFQ_RUNTIME_CHECK(dimension > 0 && dimension % 2 == 0 && maximum > 0 &&
                          std::isfinite(base) && base > 0,
                      "invalid rotary configuration");
    MFQ_RUNTIME_CHECK(
        (sections_.empty() && !interleaved_) ||
            (sections_.size() == 3 &&
             std::all_of(sections_.begin(), sections_.end(), [](auto n) { return n >= 0; }) &&
             std::accumulate(sections_.begin(), sections_.end(), int64_t(0)) == dimension / 2),
        "MRoPE sections disagree with rotary width");
}

RotaryEmbedding::Prepared RotaryEmbedding::prepare_parameters(const Tensor& value) const {
    const auto pairs = dimension_ / 2;
    std::lock_guard lock(prepared_mutex_);
    auto& cached = prepared_[value.get_device()];
    if (!cached.frequencies.defined()) {
        Prepared next;
        auto pair = tb::arange(pairs, value.options().dtype(tb::kInt64));
        next.frequencies = tb::pow(tb::full({pairs}, base_, value.options().dtype(tb::kFloat32)),
            -(pair.to(tb::kFloat32) * 2) / double(dimension_));
        std::vector<int32_t> selected(static_cast<size_t>(pairs), 0);
        for (int64_t j = 0; j < pairs; ++j) if (!sections_.empty()) {
            if (interleaved_) {
                if (j % 3 == 1 && j < sections_[1] * 3) selected[j] = 1;
                else if (j % 3 == 2 && j < sections_[2] * 3) selected[j] = 2;
            } else selected[j] = j < sections_[0] ? 0 : (j < sections_[0] + sections_[1] ? 1 : 2);
        }
        next.axes = tb::tensor(selected).to(value.device());
        MFQ_RUNTIME_CHECK(hipStreamSynchronize(mfq_current_cuda_stream()) == hipSuccess,
            "warm cached rotary parameters before CUDA graph capture");
        cached = std::move(next);
    }
    return cached;
}

Tensor RotaryEmbedding::forward_pooled(const Tensor& raw,const Tensor& weight,
        const Tensor& positions,double eps,int64_t pool) const {
    if(!fused_ || pool!=4 || dimension_>128 || maximum_>INT_MAX ||
       !raw.defined() || !raw.is_cuda() || !raw.is_contiguous() || raw.scalar_type()!=tb::kFloat16 ||
       raw.dim()!=3 || raw.size(0)!=1 || raw.size(1)<4 || raw.size(1)>INT_MAX || raw.size(2)!=128 ||
       !weight.defined() || weight.device()!=raw.device() || weight.numel()!=128 ||
       !positions.defined() || positions.device()!=raw.device() || !positions.is_contiguous() ||
       (positions.scalar_type()!=tb::kInt32 && positions.scalar_type()!=tb::kInt64) ||
       positions.dim()<1 || positions.dim()>3 || positions.size(-1)!=raw.size(1) ||
       (positions.dim()==3 && positions.size(1)!=1) || positions.numel()<1 ||
       !std::isfinite(eps) || eps<0)return {};
    const auto prepared=prepare_parameters(raw);
    return rotary_pooled_prefill_cuda(raw,weight.to(tb::kFloat32).contiguous(),positions,
        prepared.frequencies,prepared.axes,maximum_,eps);
}

bool RotaryEmbedding::update_pooled_cache(const Tensor& raw,const Tensor& weight,
        const Tensor& positions,const Tensor& cache_positions,const Tensor& pooled,double eps,int64_t pool) const {
    const char* serial_mean=std::getenv("MFQ_DISABLE_NATIVE_PARALLEL_F32_MEAN");
    if(!fused_ || (serial_mean && serial_mean[0]=='1') || pool!=4 || dimension_>128 || maximum_>INT_MAX ||
       !raw.defined() || !raw.is_cuda() || !raw.is_contiguous() || raw.scalar_type()!=tb::kFloat16 ||
       raw.dim()!=3 || raw.size(0)!=1 || raw.size(1)<1 || raw.size(1)>INT_MAX || raw.size(2)!=128 ||
       !pooled.defined() || pooled.device()!=raw.device() || !pooled.is_contiguous() ||
       pooled.scalar_type()!=tb::kFloat16 || pooled.dim()!=3 || pooled.size(0)!=1 ||
       pooled.size(1)!=(raw.size(1)+3)/4 || pooled.size(2)!=128 ||
       !weight.defined() || weight.device()!=raw.device() || weight.numel()!=128 ||
       !cache_positions.defined() || cache_positions.device()!=raw.device() || !cache_positions.is_contiguous() ||
       cache_positions.scalar_type()!=tb::kInt64 || cache_positions.dim()!=1 || cache_positions.numel()!=1 ||
       !positions.defined() || positions.device()!=raw.device() || !positions.is_contiguous() ||
       (positions.scalar_type()!=tb::kInt32 && positions.scalar_type()!=tb::kInt64) ||
       positions.dim()<1 || positions.dim()>3 || positions.size(-1)!=1 ||
       (positions.dim()==3 && positions.size(1)!=1) || positions.numel()<1 ||
       !std::isfinite(eps) || eps<0)return false;
    const auto prepared=prepare_parameters(raw);
    rotary_pooled_cache_update_cuda(raw,weight.to(tb::kFloat32).contiguous(),positions,cache_positions,
        pooled,prepared.frequencies,prepared.axes,maximum_,eps);
    return true;
}

Tensor RotaryEmbedding::forward_normalized(const Tensor& value, const Tensor& weight,
        const Tensor& positions, double eps, const Tensor& key_cache,
        const Tensor& cache_positions, const Tensor& projected_value, const Tensor& value_cache) const {
    MFQ_RUNTIME_CHECK(fused_ && value.is_cuda() && value.dim() == 4 &&
        value.size(-1) >= dimension_, "normalized rotary requires fused [B,T,H,D] input");
    const auto prepared = prepare_parameters(value);
    auto active_positions = positions.scalar_type() == tb::kInt32 || positions.scalar_type() == tb::kInt64
        ? positions.contiguous() : positions.to(tb::kInt32).contiguous();
    return rotary_normalized_cached_cuda(value, weight.to(tb::kFloat32).contiguous(), active_positions,
        prepared.frequencies, prepared.axes, maximum_, eps, key_cache, cache_positions,
        projected_value, value_cache);
}

std::vector<Tensor> RotaryEmbedding::forward_normalized_grouped(
        const std::vector<Tensor>& values,const std::vector<Tensor>& weights,
        const Tensor& positions,double eps,int cache_entry,const Tensor& key_cache,
        const Tensor& cache_positions,const Tensor& projected_value,const Tensor& value_cache) const {
    MFQ_RUNTIME_CHECK(fused_ && !values.empty() && values.size()<=3 && weights.size()==values.size(),
        "normalized rotary requires a fused input/weight group");
    std::vector<Tensor> active_weights;active_weights.reserve(weights.size());
    for(size_t i=0;i<values.size();++i) {
        MFQ_RUNTIME_CHECK(values[i].is_cuda() && values[i].dim()==4 && values[i].size(-1)>=dimension_,
            "normalized rotary group requires [B,T,H,D] inputs");
        active_weights.push_back(weights[i].to(tb::kFloat32).contiguous());
    }
    const auto prepared=prepare_parameters(values[0]);
    auto active_positions=positions.scalar_type()==tb::kInt32 || positions.scalar_type()==tb::kInt64
        ? positions.contiguous():positions.to(tb::kInt32).contiguous();
    return rotary_normalized_grouped_cuda(values,active_weights,active_positions,prepared.frequencies,
        prepared.axes,maximum_,eps,cache_entry,key_cache,cache_positions,projected_value,value_cache);
}

mfq_tensor_backend::Tensor RotaryEmbedding::forward(const Tensor &value, const Tensor &positions) const {
    MFQ_RUNTIME_CHECK(value.is_cuda() && value.dim() == 4 && value.size(-1) >= dimension_ &&
                          positions.is_cuda() && positions.device() == value.device() &&
                          positions.dim() >= 1 && positions.dim() <= 3 &&
                          positions.size(-1) == value.size(2),
                      "RoPE requires [B,H,T,D] with compatible positions");
    const auto b = value.size(0), t = value.size(2), pairs = dimension_ / 2;
    const auto axes = positions.dim() == 1 ? 1 : positions.size(0);
    const auto batches = positions.dim() < 3 ? 1 : positions.size(1);
    MFQ_RUNTIME_CHECK(axes > 0 && (batches == 1 || batches == b),
                      "RoPE position batch mismatch");
    auto input = value.scalar_type() == tb::kFloat32 ? value : value.to(tb::kFloat16);
    if(fused_) {
        const auto prepared = prepare_parameters(value);
        return rotary_embedding_cached_cuda(input.contiguous(),positions.to(tb::kInt32).contiguous(),
            prepared.frequencies,prepared.axes,maximum_);
    }
    auto f = input.to(tb::kFloat32);
    auto pair = tb::arange(pairs, value.options().dtype(tb::kInt64));
    auto frequencies = tb::pow(tb::full({pairs}, base_, f.options()),
                               -(pair.to(tb::kFloat32) * 2) / double(dimension_));
    auto axis_ids = tb::zeros({pairs}, pair.options());
    if (!sections_.empty()) {
        if (interleaved_) {
            auto remainder = pair.remainder(3);
            auto a1 = (remainder == 1) & (pair < sections_[1] * 3);
            auto a2 = (remainder == 2) & (pair < sections_[2] * 3);
            axis_ids = tb::where(a1, tb::full_like(pair, 1),
                                 tb::where(a2, tb::full_like(pair, 2), axis_ids));
        } else {
            axis_ids = tb::where(pair < sections_[0], axis_ids,
                                 tb::where(pair < sections_[0] + sections_[1],
                                           tb::full_like(pair, 1), tb::full_like(pair, 2)));
        }
        axis_ids = tb::where(axis_ids < axes, axis_ids, tb::zeros_like(axis_ids));
    }
    Tensor cosine, sine;
    for (int64_t axis = 0; axis < std::min<int64_t>(axes, 3); ++axis) {
        auto pos = (positions.dim() == 1 ? positions : positions.select(0, axis))
                       .reshape({batches, t})
                       .to(tb::kInt32)
                       .clamp(0, maximum_ - 1)
                       .to(tb::kFloat32);
        auto angles = pos.unsqueeze(-1) * frequencies;
        auto ca = tb::cos(angles), sa = tb::sin(angles);
        if (axis == 0) {
            cosine = ca;
            sine = sa;
        } else {
            cosine = tb::where(axis_ids == axis, ca, cosine);
            sine = tb::where(axis_ids == axis, sa, sine);
        }
    }
    cosine = cosine.unsqueeze(1);
    sine = sine.unsqueeze(1);
    auto first = f.narrow(-1, 0, pairs), second = f.narrow(-1, pairs, pairs);
    return tb::cat({first * cosine - second * sine, second * cosine + first * sine,
                    f.narrow(-1, dimension_, f.size(-1) - dimension_)},
                   -1)
        .to(input.scalar_type());
}

} // namespace mfq::cuda
