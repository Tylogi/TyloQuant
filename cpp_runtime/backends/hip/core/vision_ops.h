#pragma once

#include "hip_execution.h"
#include "mfq_hip_attention_ops.h"
#include "mfq_hip_norm_ops.h"
#include "models/common/grid_vision_model.h"
#include "runtime/execution_options.h"
#include "storage/moe_expert_cache.h"
#include "storage/weight_loader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda::vision_ops {

using Tensor = mfq_tensor_backend::Tensor;

// Each streamed block owns its ordinary projection storage until its forward
// completes. TQ images keep their device pointers throughout that lifetime.
template <class Visitor>
void visit_linear(QuantLinear& weight, Visitor visit) {
    MFQ_RUNTIME_CHECK(
        !weight.tensor_parallel(),
        "streamed vision weights require one CUDA device");
    if(weight.is_tq()) {
        MFQ_RUNTIME_CHECK(weight.tq.image.is_cuda(),"vision TQ image must stay on its GPU");
        visit(weight.tq.image);
    }
    auto& n = weight.nint;
    auto& v = weight.nvq;
    for (auto* tensor : {
             &weight.dense,
             &n.q_packed,
             &n.row_q_bits,
             &n.row_q_bit_offsets,
             &n.q8_zero_scale,
             &n.sub_scale,
             &n.sub_min,
             &n.neuron_scale,
             &n.neuron_min,
             &v.indices_packed,
             &v.aux_packed,
             &v.sub_scale_packed,
             &v.neuron_scale,
             &v.codebook,
             &weight.mxfp4.weight.values,
             &weight.mxfp4.weight.scales,
             &weight.mxfp8.weight.values,
             &weight.mxfp8.weight.scales,
             &weight.mxfp4_sq.weight.blob,
             &weight.mxfp4_sq.weight.row_q,
             &weight.mxfp4_sq.weight.row_symbol_byte_offsets,
             &weight.mxfp4_sq.weight.row_auxiliary,
             &weight.fp8_sq.weight.blob,
             &weight.fp8_sq.weight.row_q,
             &weight.fp8_sq.weight.row_symbol_byte_offsets,
         }) {
        if (tensor->defined()) visit(*tensor);
    }
    n.workspaces.clear();
    v.workspaces.clear();
}

template <class Weights>
Weights to_cuda(Weights weights) {
    weights.visit_tensors(
        [](Tensor& tensor) { tensor = tensor.to(mfq_tensor_backend::kCUDA); });
    return weights;
}

template <class Weights>
std::size_t weight_bytes(Weights weights) {
    std::size_t bytes = 0;
    weights.visit_tensors(
        [&](Tensor& tensor) { bytes += tensor.nbytes(); });
    return bytes;
}

// HIP keeps its expert arena private. Vision weights are streamed per block
// through ordinary tensor allocations and released before the next yield.
template <class Forward>
Tensor vision_workspace(CudaExecutionContext&, std::size_t, Forward forward) {
    return forward();
}

struct Affine {
    QuantLinear linear;
    Tensor bias;

    template <class Visitor>
    void visit_tensors(Visitor visit) {
        visit_linear(linear, visit);
        if (bias.defined()) visit(bias);
    }

    static Affine load(
        CudaExecutionContext& execution,
        const mfq::ModelSource& model,
        const std::string& weight_name,
        const std::string& bias_name) {
        struct LoadingScope {
            bool& flag;bool previous;
            LoadingScope(CudaExecutionContext& execution,bool tq):flag(execution.loading_cpu_layer),previous(flag) {
                if(tq)flag=false;
            }
            ~LoadingScope(){flag=previous;}
        } loading(execution,require_tensor(model,weight_name).dtype=="TQ");
        Affine result{
            load_quant_linear(execution, model, weight_name),
            {},
        };
        if (has_tensor(model, bias_name)) {
            result.bias = load_dense_gpu(execution, model, bias_name);
            if (result.bias.dim() != 1 ||
                result.bias.size(0) != result.linear.out()) {
                throw std::runtime_error(
                    "vision affine bias shape mismatch: " + bias_name);
            }
        }
        return result;
    }

    Tensor operator()(
        CudaExecutionContext& execution,
        Tensor input) const {
        auto output = linear.forward(execution, std::move(input));
        return bias.defined()
            ? output + bias.to(output.scalar_type())
            : output;
    }
};

struct RmsNorm {
    Tensor weight;
    double epsilon = 1e-6;

    template <class Visitor>
    void visit_tensors(Visitor visit) {
        visit(weight);
    }

    static RmsNorm load(
        CudaExecutionContext& execution,
        const mfq::ModelSource& model,
        const std::string& name,
        std::int64_t width,
        double epsilon) {
        auto weight = load_dense_native_gpu(execution, model, name)
                          .to(mfq_tensor_backend::kFloat32)
                          .contiguous();
        if (weight.dim() != 1 || weight.size(0) != width) {
            throw std::runtime_error(
                "vision RMSNorm width mismatch: " + name);
        }
        return {std::move(weight), epsilon};
    }

    Tensor operator()(const Tensor& input) const {
        if (input.dim() == 0 || input.size(-1) != weight.size(0)) {
            throw std::runtime_error(
                "vision RMSNorm input width mismatch");
        }
        const auto shape = input.sizes().vec();
        auto normalized = rms_norm_cuda(
            input.reshape({-1, shape.back()}).to(mfq_tensor_backend::kFloat32),
            weight,
            epsilon);
        return normalized.reshape(shape)
            .to(input.scalar_type())
            .contiguous();
    }
};

inline Tensor apply_axial_rope(
    const Tensor& input,
    const std::vector<std::int32_t>& positions,
    double theta) {
    if (input.dim() != 3 || input.size(0) <= 0 ||
        input.size(2) <= 0 || input.size(2) % 4 != 0 ||
        positions.size() != static_cast<std::size_t>(input.size(0) * 2)) {
        throw std::runtime_error(
            "vision axial RoPE geometry mismatch");
    }
    const std::int64_t tokens = input.size(0);
    const std::int64_t head_dim = input.size(2);
    const std::int64_t half = head_dim / 2;
    const std::int64_t frequencies = half / 2;
    std::vector<float> cosine(static_cast<std::size_t>(tokens * head_dim));
    std::vector<float> sine(cosine.size());
    for (std::int64_t token = 0; token < tokens; ++token) {
        for (int axis = 0; axis < 2; ++axis) {
            const float position = static_cast<float>(
                positions[static_cast<std::size_t>(token * 2 + axis)]);
            for (std::int64_t index = 0; index < frequencies; ++index) {
                const float inverse = std::pow(
                    static_cast<float>(theta),
                    -static_cast<float>(2 * index) /
                        static_cast<float>(half));
                const float angle = position * inverse;
                const std::int64_t column = axis * frequencies + index;
                for (int repeat = 0; repeat < 2; ++repeat) {
                    const auto offset = static_cast<std::size_t>(
                        token * head_dim + repeat * half + column);
                    cosine[offset] = std::cos(angle);
                    sine[offset] = std::sin(angle);
                }
            }
        }
    }
    const auto cpu = mfq_tensor_backend::TensorOptions()
                         .dtype(mfq_tensor_backend::kFloat32)
                         .device(mfq_tensor_backend::kCPU);
    auto cos = mfq_tensor_backend::tensor(cosine, cpu)
                   .reshape({tokens, 1, head_dim})
                   .to(input.device());
    auto sin = mfq_tensor_backend::tensor(sine, cpu)
                   .reshape({tokens, 1, head_dim})
                   .to(input.device());
    auto source = input.to(mfq_tensor_backend::kFloat32);
    auto first = source.narrow(-1, 0, half);
    auto second = source.narrow(-1, half, half);
    auto rotated = mfq_tensor_backend::cat({-second, first}, -1);
    return (source * cos + rotated * sin)
        .to(input.scalar_type())
        .contiguous();
}

inline Tensor segmented_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::vector<std::int32_t>& lengths) {
    std::vector<Tensor> outputs;
    outputs.reserve(lengths.size());
    std::int64_t offset = 0;
    for (const std::int32_t length : lengths) {
        if (length <= 0 || offset + length > query.size(0)) {
            throw std::runtime_error(
                "vision attention segment is invalid");
        }
        const auto segment = [offset, length](const Tensor& input) {
            return input.narrow(0, offset, length)
                .transpose(0, 1)
                .unsqueeze(0)
                .contiguous();
        };
        auto attended = attention_cuda(
            segment(query).to(mfq_tensor_backend::kFloat16),
            segment(key).to(mfq_tensor_backend::kFloat16),
            segment(value).to(mfq_tensor_backend::kFloat16),
            1.0 / std::sqrt(static_cast<double>(query.size(2))),
            false);
        outputs.push_back(
            attended.index({0}).transpose(0, 1).contiguous());
        offset += length;
    }
    if (outputs.empty() || offset != query.size(0)) {
        throw std::runtime_error(
            "vision attention segments do not cover every patch");
    }
    return outputs.size() == 1
        ? outputs.front()
        : mfq_tensor_backend::cat(outputs, 0);
}

struct VisionAttention {
    Affine qkv;
    Affine output;
    std::int64_t hidden = 0;
    std::int64_t heads = 0;
    std::int64_t head_dim = 0;
    double theta = 10'000.0;

    Tensor operator()(
        CudaExecutionContext& execution,
        const Tensor& input,
        const std::vector<std::int32_t>& positions,
        const std::vector<std::int32_t>& segment_lengths) const {
        const std::int64_t tokens = input.size(0);
        return mfq::models::grid_vision::attention(
            input,
            [&](Tensor value) { return qkv(execution, std::move(value)); },
            [&](Tensor value) {
                auto projected = value.reshape({tokens, 3, heads, head_dim});
                return std::array<Tensor, 3>{
                    projected.select(1, 0),
                    projected.select(1, 1),
                    projected.select(1, 2),
                };
            },
            [&](Tensor value) {
                return apply_axial_rope(value, positions, theta);
            },
            [&](Tensor q, Tensor k, Tensor v) {
                return segmented_attention(
                    q, k, v, segment_lengths);
            },
            [&](Tensor value) {
                return output(
                    execution,
                    value.reshape({tokens, hidden}));
            });
    }
};

inline Tensor gelu_tanh(const Tensor& input) {
    return mfq_tensor_backend::gelu(input, "tanh");
}

} // namespace mfq::cuda::vision_ops
