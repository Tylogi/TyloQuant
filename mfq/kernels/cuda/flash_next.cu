// Qwen4-Exp / GLM-5-Next primitives ported from the Metal implementation.
// Keep its explicit FP32/FP16 boundaries, pool truncation, causal offset,
// selected-index order, GR averaging and Sinkhorn iteration order.
#include "flash_next.h"

#include <cmath>
#include <limits>

namespace mfq_flash_next {
namespace tb = mfq_tensor_backend;
namespace {

void values(std::initializer_list<const Tensor*> tensors) {
    const auto* first = *tensors.begin();
    for (const auto* value : tensors) {
        MFQ_RUNTIME_CHECK(value->defined() && value->is_cuda(),
            "Flash-Next requires CUDA tensors");
        MFQ_RUNTIME_CHECK(value->device() == first->device(),
            "Flash-Next tensors must share one CUDA device");
        const auto dtype = value->scalar_type();
        MFQ_RUNTIME_CHECK(dtype == tb::kFloat32 || dtype == tb::kFloat16 ||
            dtype == tb::kBFloat16, "Flash-Next requires F32/F16/BF16 values");
    }
}

Tensor mm(const Tensor& left, const Tensor& right) {
    // MLX matmul promotes mixed floating operands, unlike ATen matmul.
    const auto dtype = left.scalar_type() == right.scalar_type()
        ? left.scalar_type() : tb::kFloat32;
    return tb::matmul(left.to(dtype), right.to(dtype));
}

Tensor dots(const Tensor& query, const Tensor& pooled) {
    values({&query, &pooled});
    MFQ_RUNTIME_CHECK(query.dim() == 4 && pooled.dim() == 3,
        "Flash-Next scores require [B,T,H,D] query and [B,P,D] keys");
    MFQ_RUNTIME_CHECK(query.size(0) == pooled.size(0) &&
        query.size(3) == pooled.size(2) && query.size(2) > 0 && query.size(3) > 0,
        "Flash-Next score dimensions disagree");
    return tb::matmul(query.to(tb::kFloat32),
        pooled.to(tb::kFloat32).transpose(-1, -2).unsqueeze(1)).clamp_min(0.0);
}

void attention_shapes(const Tensor& query, const Tensor& key, const Tensor& value) {
    values({&query, &key, &value});
    MFQ_RUNTIME_CHECK(query.dim() == 4 && key.dim() == 4 && value.sizes() == key.sizes(),
        "Flash-Next GQA requires [B,H,T,D] query/key/value");
    MFQ_RUNTIME_CHECK(query.size(0) == key.size(0) && key.size(1) > 0 &&
        query.size(1) > 0 && query.size(1) % key.size(1) == 0 &&
        query.size(3) == key.size(3) && query.size(3) > 0,
        "Flash-Next GQA dimensions disagree");
}

// Each CTA retains the Metal kernel's eight warp partials and online-softmax
// recurrence. The generic wide-D path stores accumulators in its output only;
// it imposes no extra head-width limit. Published D=256/512 stay in registers.
template<int Slices>
__global__ void sparse_attention_kernel(
    const float* query, const half* key, const half* value, const int32_t* indices,
    float* output, int64_t heads, int64_t kv_heads, int64_t tokens,
    int64_t width, int64_t cache_length, int64_t topk, float scale) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t head = blockIdx.x % heads;
    const int64_t row = blockIdx.x / heads;
    const int64_t token = row % tokens;
    const int64_t batch = row / tokens;
    const int64_t kv_head = head / (heads / kv_heads);
    const int64_t qb = ((batch * heads + head) * tokens + token) * width;
    const int64_t ob = ((batch * tokens + token) * heads + head) * width;
    __shared__ float partials[8];
    __shared__ float online[4];
    float accum[Slices > 0 ? Slices : 1] = {};
    if constexpr (Slices == 0) {
        for (int64_t d = tid; d < width; d += 256) output[ob + d] = 0.f;
    }
    if (tid == 0) {
        online[0] = -INFINITY;
        online[1] = online[2] = online[3] = 0.f;
    }
    __syncthreads();
    for (int64_t selected = 0; selected < topk; ++selected) {
        const int64_t index = indices[row * topk + selected];
        const bool valid = index >= 0 && index < cache_length;
        const int64_t cb = ((batch * kv_heads + kv_head) * cache_length +
            (valid ? index : 0)) * width;
        float dot = 0.f;
        if (valid) {
            for (int64_t d = tid; d < width; d += 256) {
                dot += query[qb + d] * __half2float(key[cb + d]);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        if (lane == 0) partials[warp] = dot;
        __syncthreads();
        if (tid == 0) {
            float score = 0.f;
            for (int group = 0; group < 8; ++group) score += partials[group];
            score = valid ? score * scale : -INFINITY;
            const float old_max = online[0];
            const float next_max = fmaxf(old_max, score);
            const float old_scale = isfinite(old_max) ? expf(old_max - next_max) : 0.f;
            const float next_scale = valid ? expf(score - next_max) : 0.f;
            online[0] = next_max;
            online[1] = online[1] * old_scale + next_scale;
            online[2] = old_scale;
            online[3] = next_scale;
        }
        __syncthreads();
        if constexpr (Slices > 0) {
            #pragma unroll
            for (int slice = 0; slice < Slices; ++slice) {
                const int64_t d = tid + slice * 256;
                const float v = valid && d < width ? __half2float(value[cb + d]) : 0.f;
                accum[slice] = accum[slice] * online[2] + v * online[3];
            }
        } else {
            for (int64_t d = tid; d < width; d += 256) {
                const float v = valid ? __half2float(value[cb + d]) : 0.f;
                output[ob + d] = output[ob + d] * online[2] + v * online[3];
            }
        }
        __syncthreads();
    }
    const float inverse = online[1] > 0.f ? 1.f / online[1] : 0.f;
    if constexpr (Slices > 0) {
        #pragma unroll
        for (int slice = 0; slice < Slices; ++slice) {
            const int64_t d = tid + slice * 256;
            if (d < width) output[ob + d] = accum[slice] * inverse;
        }
    } else {
        for (int64_t d = tid; d < width; d += 256) output[ob + d] *= inverse;
    }
}

Tensor sparse_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                        const Tensor& selected, double scale) {
    attention_shapes(q, k, v);
    MFQ_RUNTIME_CHECK(selected.defined() && selected.is_cuda() &&
        selected.device() == q.device() && selected.dim() == 3 &&
        selected.size(0) == q.size(0) && selected.size(1) == q.size(2) &&
        selected.size(2) > 0, "Flash-Next sparse indices must have [B,T,topk] shape");
    MFQ_RUNTIME_CHECK(selected.scalar_type() == tb::kInt32 ||
        selected.scalar_type() == tb::kInt64, "Flash-Next sparse indices must be integer");
    MFQ_RUNTIME_CHECK(std::isfinite(scale), "Flash-Next attention scale must be finite");
    MfqCudaGuard guard(q.device());
    auto query = q.to(tb::kFloat32).contiguous();
    auto key = k.to(tb::kFloat16).contiguous();
    auto value = v.to(tb::kFloat16).contiguous();
    auto indices = selected.to(tb::kInt32).contiguous();
    auto output = tb::empty({q.size(0), q.size(2), q.size(1), q.size(3)},
        q.options().dtype(tb::kFloat32));
    if (output.numel() == 0) return output;
    const int64_t blocks = output.numel() / q.size(3);
    MFQ_RUNTIME_CHECK(blocks <= std::numeric_limits<int>::max(),
        "Flash-Next sparse attention exceeds CUDA grid size");
    const auto stream = mfq_current_cuda_stream();
    #define MFQ_LAUNCH_SPARSE(S) sparse_attention_kernel<S><<<unsigned(blocks), 256, 0, stream>>>( \
        query.data_ptr<float>(), reinterpret_cast<const half*>(key.data_ptr()), \
        reinterpret_cast<const half*>(value.data_ptr()), indices.data_ptr<int32_t>(), \
        output.data_ptr<float>(), q.size(1), k.size(1), q.size(2), q.size(3), \
        k.size(2), selected.size(2), static_cast<float>(scale))
    if (q.size(3) <= 256) { MFQ_LAUNCH_SPARSE(1); }
    else if (q.size(3) <= 512) { MFQ_LAUNCH_SPARSE(2); }
    else if (q.size(3) <= 1024) { MFQ_LAUNCH_SPARSE(4); }
    else { MFQ_LAUNCH_SPARSE(0); }
    #undef MFQ_LAUNCH_SPARSE
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

Tensor dense_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                       int64_t offset, double scale) {
    attention_shapes(query, key, value);
    const auto count = key.size(2);
    MFQ_RUNTIME_CHECK(count > 0 && offset >= 0 && offset <= count &&
        query.size(2) <= count - offset, "Flash-Next query range is outside the cache");
    MFQ_RUNTIME_CHECK(std::isfinite(scale), "Flash-Next attention scale must be finite");
    MfqCudaGuard guard(query.device());
    const auto options = query.options().dtype(tb::kInt64);
    const auto kp = tb::arange(count, options).unsqueeze(0);
    const auto qp = tb::arange(query.size(2), options).unsqueeze(1) + offset;
    auto mask = (kp <= qp).unsqueeze(0).unsqueeze(0);
    return tb::scaled_dot_product_attention(query, key.to(query.scalar_type()),
        value.to(query.scalar_type()), mask, 0.0, false, scale, true).permute({0, 2, 1, 3});
}
} // namespace

Tensor qwen4_grouped_rms_norm(const Tensor& value, const Tensor& weight,
                              int64_t group, double eps) {
    values({&value, &weight});
    MFQ_RUNTIME_CHECK(value.dim() >= 1 && group > 0 && value.size(-1) > 0 &&
        value.size(-1) % group == 0 && weight.dim() == 1 && weight.size(0) == value.size(-1),
        "Qwen4 grouped RMSNorm dimensions disagree");
    MfqCudaGuard guard(value.device());
    auto shape = value.sizes().vec();
    shape.back() /= group;
    shape.push_back(group);
    auto source = value.to(tb::kFloat32).reshape(shape);
    auto normalized = source * tb::rsqrt((source * source).mean(-1, true) + eps);
    return (normalized.reshape(value.sizes()) * (1.0 + weight.to(tb::kFloat32)))
        .to(value.scalar_type());
}

std::vector<Tensor> qwen4_gated_residual_pre(
    const Tensor& input, const Tensor& norm, const Tensor& down, const Tensor& up,
    const std::optional<Tensor>& inject, int64_t hidden, int64_t streams, double eps) {
    values({&input, &norm, &down, &up});
    MFQ_RUNTIME_CHECK(input.dim() >= 1 && hidden > 0 && streams > 0 &&
        input.size(-1) / streams == hidden && input.size(-1) % streams == 0 &&
        down.dim() == 2 && up.dim() == 2 && down.size(1) == input.size(-1) &&
        up.size(0) == input.size(-1) && up.size(1) == down.size(0),
        "Qwen4 gated-residual projection dimensions disagree");
    if (inject) {
        values({&input, &*inject});
        MFQ_RUNTIME_CHECK(inject->dim() == 2 && inject->size(0) == streams &&
            inject->size(1) == input.size(-1), "Qwen4 injection projection dimensions disagree");
    }
    MfqCudaGuard guard(input.device());
    auto normalized = qwen4_grouped_rms_norm(input, norm, hidden, eps);
    auto low = mm(normalized, down.transpose(-1, -2)) / streams;
    low = low * tb::sigmoid(low);
    auto mixing = tb::sigmoid(mm(low, up.transpose(-1, -2)));
    auto shape = input.sizes().vec();
    shape.back() = streams;
    shape.push_back(hidden);
    auto mixed = (mixing.reshape(shape) * normalized.reshape(shape)).mean(-2);
    auto injection = inject
        ? 2.0 * tb::sigmoid(mm(normalized, inject->transpose(-1, -2)) / streams)
        : Tensor{};
    return {mixed, input, injection};
}

Tensor qwen4_gated_residual_post(const Tensor& branch, const Tensor& residual,
                                 const Tensor& injection, int64_t streams) {
    values({&branch, &residual, &injection});
    MFQ_RUNTIME_CHECK(branch.dim() >= 1 && streams > 0 && branch.dim() == residual.dim() &&
        branch.dim() == injection.dim(), "Qwen4 residual ranks disagree");
    auto shape = branch.sizes().vec();
    shape.back() *= streams;
    MFQ_RUNTIME_CHECK(residual.sizes().vec() == shape, "Qwen4 residual width disagrees");
    shape.back() = streams;
    MFQ_RUNTIME_CHECK(injection.sizes().vec() == shape, "Qwen4 injection shape disagrees");
    MfqCudaGuard guard(branch.device());
    return residual + (branch.unsqueeze(-2) * injection.unsqueeze(-1)).reshape(residual.sizes());
}

std::vector<Tensor> glm5_mhc_pre(const Tensor& input, const Tensor& function,
    const Tensor& base, const Tensor& scale, int64_t iterations, double hc_eps, double rms_eps) {
    values({&input, &function, &base, &scale});
    MFQ_RUNTIME_CHECK(input.dim() == 4 && input.size(2) > 0 && input.size(3) > 0,
        "GLM mHC requires [B,T,C,H] input");
    const auto streams = input.size(2), hidden = input.size(3);
    const auto mix = (2 + streams) * streams;
    MFQ_RUNTIME_CHECK(function.dim() == 2 && function.size(0) == mix &&
        function.size(1) == streams * hidden && base.dim() == 1 && base.size(0) == mix &&
        scale.dim() == 1 && scale.size(0) == 3 && iterations > 0,
        "GLM mHC parameters disagree");
    MfqCudaGuard guard(input.device());
    auto flat = input.to(tb::kFloat32).reshape({input.size(0), input.size(1), streams * hidden});
    flat = flat * tb::rsqrt((flat * flat).mean(-1, true) + rms_eps);
    auto logits = tb::matmul(flat, function.to(tb::kFloat32).transpose(-1, -2));
    auto pre = tb::sigmoid(logits.narrow(-1, 0, streams) * scale.select(0, 0) +
        base.narrow(0, 0, streams)) + hc_eps;
    auto post = 2.0 * tb::sigmoid(logits.narrow(-1, streams, streams) * scale.select(0, 1) +
        base.narrow(0, streams, streams));
    auto combination_logits = logits.narrow(-1, 2 * streams, streams * streams)
        .reshape({input.size(0), input.size(1), streams, streams}) * scale.select(0, 2) +
        base.narrow(0, 2 * streams, streams * streams).reshape({streams, streams});
    auto combination = tb::softmax(combination_logits, -1) + hc_eps;
    combination = combination / (combination.sum(-2, true) + hc_eps);
    for (int64_t i = 1; i < iterations; ++i) {
        combination = combination / (combination.sum(-1, true) + hc_eps);
        combination = combination / (combination.sum(-2, true) + hc_eps);
    }
    auto collapsed = (pre.unsqueeze(-1) * input).sum(-2).to(input.scalar_type());
    return {post, combination, collapsed};
}

Tensor glm5_mhc_post(const Tensor& branch, const Tensor& residual,
                      const Tensor& post, const Tensor& combination) {
    values({&branch, &residual, &post, &combination});
    MFQ_RUNTIME_CHECK(residual.dim() == 4 && branch.dim() == 3 &&
        branch.size(0) == residual.size(0) && branch.size(1) == residual.size(1) &&
        branch.size(2) == residual.size(3), "GLM mHC branch/residual dimensions disagree");
    const auto b = branch.size(0), t = branch.size(1), c = residual.size(2);
    MFQ_RUNTIME_CHECK(post.sizes().vec() == std::vector<int64_t>({b,t,c}) &&
        combination.sizes().vec() == std::vector<int64_t>({b,t,c,c}),
        "GLM mHC post metadata dimensions disagree");
    MfqCudaGuard guard(branch.device());
    auto mixed = mm(combination.transpose(-1, -2), residual);
    return post.to(residual.scalar_type()).unsqueeze(-1) * branch.unsqueeze(-2) + mixed;
}

Tensor glm5_kda_forget_gate(const Tensor& input, const Tensor& fa, const Tensor& fb,
    const Tensor& bias, const Tensor& a_log, int64_t heads, int64_t width, double lower_bound) {
    values({&input, &fa, &fb, &bias, &a_log});
    MFQ_RUNTIME_CHECK(input.dim() >= 1 && heads > 0 && width > 0 && fa.dim() == 2 &&
        fa.size(0) == width && fa.size(1) == input.size(-1) && fb.dim() == 2 &&
        fb.size(0) == heads * width && fb.size(1) == width && bias.dim() == 1 &&
        bias.size(0) == heads * width && a_log.dim() == 1 && a_log.size(0) == heads &&
        std::isfinite(lower_bound), "GLM KDA forget-gate dimensions disagree");
    MfqCudaGuard guard(input.device());
    auto reduced = mm(input, fa.transpose(-1, -2));
    auto gate = mm(reduced, fb.transpose(-1, -2)).to(tb::kFloat32) + bias.to(tb::kFloat32);
    auto shape = input.sizes().vec();
    shape.back() = heads;
    shape.push_back(width);
    gate = gate.reshape(shape);
    std::vector<int64_t> rate_shape(shape.size(), 1);
    rate_shape[shape.size() - 2] = heads;
    auto rate = a_log.to(tb::kFloat32).exp().reshape(rate_shape);
    return lower_bound * tb::sigmoid(rate * gate);
}

Tensor qsa_block_scores(const Tensor& query, const Tensor& pooled) {
    values({&query, &pooled});
    MfqCudaGuard guard(query.device());
    return dots(query, pooled).sum(-2) / std::sqrt(double(query.size(-1)));
}

Tensor glm5_kpool_scores(const Tensor& query, const Tensor& pooled, const Tensor& weights) {
    values({&query, &pooled, &weights});
    MFQ_RUNTIME_CHECK(query.dim() == 4 && weights.dim() == 3 &&
        weights.size(0) == query.size(0) && weights.size(1) == query.size(1) &&
        weights.size(2) == query.size(2), "GLM k-pool head weights disagree");
    MfqCudaGuard guard(query.device());
    auto scores = dots(query, pooled) / std::sqrt(double(query.size(3)));
    auto head_weights = weights.to(tb::kFloat32) / std::sqrt(double(query.size(2)));
    return (scores * head_weights.unsqueeze(-1)).sum(-2);
}

Tensor glm5_kpool_states(const Tensor& keys, const Tensor& gates,
                          const Tensor& ape, int64_t pool) {
    values({&keys, &gates, &ape});
    MFQ_RUNTIME_CHECK(keys.dim() == 3 && gates.sizes() == keys.sizes() && pool > 0 &&
        ape.dim() == 2 && ape.size(0) == pool && ape.size(1) == keys.size(2),
        "GLM k-pool parameter dimensions disagree");
    MfqCudaGuard guard(keys.device());
    const auto complete = keys.size(1) / pool;
    if (complete == 0) return tb::zeros({keys.size(0), 0, keys.size(2)}, keys.options());
    const std::vector<int64_t> shape{keys.size(0), complete, pool, keys.size(2)};
    auto grouped_keys = keys.narrow(1, 0, complete * pool).reshape(shape);
    auto grouped_gates = gates.narrow(1, 0, complete * pool).reshape(shape);
    auto probabilities = tb::softmax(grouped_gates.to(tb::kFloat32) +
        ape.unsqueeze(0).unsqueeze(0), -2);
    return (probabilities.to(keys.scalar_type()) * grouped_keys).sum(-2);
}

std::vector<Tensor> qwen4_ple_dilated_conv_silu(const Tensor& input,
    const Tensor& weight, const std::optional<Tensor>& state, int64_t dilation) {
    values({&input, &weight});
    MFQ_RUNTIME_CHECK(input.dim() == 3, "Qwen4 PLE input must have [B,T,C] shape");
    auto w = weight;
    if (w.dim() == 3) {
        MFQ_RUNTIME_CHECK(w.size(0) == input.size(2) && w.size(1) == 1,
            "Qwen4 PLE packed weight dimensions disagree");
        w = w.select(1, 0);
    }
    MFQ_RUNTIME_CHECK(w.dim() == 2 && w.size(0) == input.size(2) &&
        w.size(1) > 0 && dilation > 0 && w.size(1) - 1 <=
            std::numeric_limits<int64_t>::max() / dilation,
        "Qwen4 PLE kernel/dilation dimensions disagree");
    const auto length = (w.size(1) - 1) * dilation;
    const std::vector<int64_t> state_shape{input.size(0), length, input.size(2)};
    if (state) {
        values({&input, &*state});
        MFQ_RUNTIME_CHECK(state->sizes().vec() == state_shape, "Qwen4 PLE state shape disagrees");
    }
    MfqCudaGuard guard(input.device());
    auto previous = state ? state->to(input.scalar_type()) : tb::zeros(state_shape, input.options());
    auto combined = tb::cat({previous, input}, 1);
    auto output = tb::zeros(input.sizes(), input.options().dtype(tb::kFloat32));
    for (int64_t tap = 0; tap < w.size(1); ++tap) {
        output = output + combined.narrow(1, tap * dilation, input.size(1)).to(tb::kFloat32) *
            w.select(1, tap).to(tb::kFloat32).unsqueeze(0).unsqueeze(0);
    }
    output = output * tb::sigmoid(output);
    auto next = combined.narrow(1, length ? combined.size(1) - length : 0, length).contiguous();
    return {output.to(input.scalar_type()), next};
}

Tensor qwen4_dense_gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, int64_t offset) {
    attention_shapes(q, k, v);
    return dense_attention(q, k, v, offset, 1.0 / std::sqrt(double(q.size(3))));
}

Tensor qwen4_sparse_gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& indices) {
    attention_shapes(q, k, v);
    return sparse_attention(q, k, v, indices, 1.0 / std::sqrt(double(q.size(3))));
}

Tensor glm5_dense_mla_attention(const Tensor& query, const Tensor& cache,
                                int64_t offset, std::optional<double> scale) {
    values({&query, &cache});
    MFQ_RUNTIME_CHECK(query.dim() == 4 && cache.dim() == 3,
        "GLM MLA requires [B,H,T,D] query and [B,K,D] cache");
    return dense_attention(query, cache.unsqueeze(1), cache.unsqueeze(1), offset,
        scale.value_or(1.0 / std::sqrt(double(query.size(3)))));
}

Tensor glm5_sparse_mla_attention(const Tensor& query, const Tensor& cache,
    const Tensor& indices, std::optional<double> scale) {
    values({&query, &cache});
    MFQ_RUNTIME_CHECK(query.dim() == 4 && cache.dim() == 3,
        "GLM MLA requires [B,H,T,D] query and [B,K,D] cache");
    return sparse_attention(query, cache.unsqueeze(1), cache.unsqueeze(1), indices,
        scale.value_or(1.0 / std::sqrt(double(query.size(3)))));
}
} // namespace mfq_flash_next
