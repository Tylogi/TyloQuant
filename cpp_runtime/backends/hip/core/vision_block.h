#pragma once

#include "quant_linear.h"
#include "storage/weight_loader.h"

#include "models/common/vision_block.h"

#include "mfq_tensor_backend.h"

#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::cuda {

// Linear projection with an optional bias vector.  Vision transformers apply
// it to attention projections and to both MLP projections, and every released
// checkpoint of that kind keeps those biases.
struct BiasLinear {
    QuantLinear weight;
    mfq_tensor_backend::Tensor bias;

    static BiasLinear load(CudaExecutionContext &execution, const mfq::ModelSource &mfq,
                           const std::string &prefix, bool with_bias = true) {
        BiasLinear result;
        result.weight = load_quant_linear(execution, mfq, prefix + ".weight");
        const std::string bias_name = prefix + ".bias";
        if (with_bias && has_tensor(mfq, bias_name)) {
            result.bias = load_dense_gpu(execution, mfq, bias_name);
        }
        return result;
    }

    // Row slice of a fused projection: RADIO stores q, k and v in one
    // [3 * width, hidden] tensor, so each head group is a contiguous slab.
    BiasLinear rows(int64_t begin, int64_t count) const {
        if (!weight.is_dense()) {
            throw std::runtime_error("fused vision projection must stay dense to be split");
        }
        if (begin < 0 || count <= 0 || begin + count > weight.dense.size(0)) {
            throw std::runtime_error("fused vision projection row range is invalid");
        }
        BiasLinear result;
        result.weight = weight;
        result.weight.dense = weight.dense.narrow(0, begin, count).contiguous();
        if (bias.defined()) {
            result.bias = bias.narrow(0, begin, count).contiguous();
        }
        return result;
    }

    mfq_tensor_backend::Tensor forward(CudaExecutionContext &execution,
                                       mfq_tensor_backend::Tensor input) const {
        const auto output_dtype = input.scalar_type();
        if (weight.is_dense()) {
            const auto dense_dtype = weight.dense.scalar_type();
            std::optional<mfq_tensor_backend::Tensor> dense_bias = std::nullopt;
            if (bias.defined()) {
                dense_bias = bias.to(dense_dtype);
            }
            return mfq_linear(input.to(dense_dtype), weight.dense, dense_bias)
                .to(output_dtype)
                .contiguous();
        }
        auto output = weight.forward(execution, input);
        if (bias.defined()) {
            output = output + bias.to(output.scalar_type());
        }
        return output.to(output_dtype).contiguous();
    }
};

// LayerNorm with a bias vector, the normalization before every vision block
// branch (unlike the RMSNorm used inside decoder mixers).
inline mfq_tensor_backend::Tensor vision_layer_norm(
        mfq_tensor_backend::Tensor input, const mfq_tensor_backend::Tensor &weight,
        const mfq_tensor_backend::Tensor &bias, double epsilon) {
    const auto dtype = input.scalar_type();
    auto output = mfq_tensor_backend::layer_norm(input, {input.size(-1)}, weight.to(dtype),
                                                 bias.to(dtype), epsilon);
    return output.to(dtype).contiguous();
}

// MLP activation of an image encoder.  SigLIP checkpoints configure the tanh
// approximation of GELU, RADIO's `hidden_act: "gelu"` is the exact erf form;
// the rest of the block is identical, so the variant is a block parameter.
enum class VisionActivation { gelu_tanh, gelu_erf };

inline mfq_tensor_backend::Tensor vision_activation(
        mfq_tensor_backend::Tensor input, VisionActivation activation) {
    return mfq_tensor_backend::gelu(
        std::move(input), activation == VisionActivation::gelu_tanh ? "tanh" : "none");
}

// Full (non-causal) scaled dot-product attention over projected head tensors.
// Vision encoders attend across the whole patch grid, so the mask is optional
// and only used to hide padding patches.
inline mfq_tensor_backend::Tensor vision_non_causal_attention(
        mfq_tensor_backend::Tensor query, mfq_tensor_backend::Tensor key,
        mfq_tensor_backend::Tensor value,
        MfqOptional<mfq_tensor_backend::Tensor> additive_mask, double scale) {
    auto scores = mfq_tensor_backend::matmul(query, key.transpose(-2, -1)) * scale;
    if (additive_mask.has_value()) {
        scores = scores + additive_mask.value().to(scores.scalar_type());
    }
    auto probabilities = mfq_tensor_backend::softmax(scores, -1, mfq_tensor_backend::kFloat32)
                             .to(value.scalar_type());
    return mfq_tensor_backend::matmul(probabilities, value).contiguous();
}

// One pre-LayerNorm vision transformer block, shared by the SigLIP and RADIO
// towers because both run the same ViT graph:
//   h = h + attention(norm1(h));  h = h + mlp(norm2(h))
// with non-causal attention over the patch sequence.  LayerScale is folded into
// the branches by the caller; the released RADIO checkpoints carry no
// layer_scale parameter, so its value is one and the block is unchanged.
struct VisionBlock {
    BiasLinear query;
    BiasLinear key;
    BiasLinear value;
    BiasLinear output;
    mfq_tensor_backend::Tensor norm1_weight;
    mfq_tensor_backend::Tensor norm1_bias;
    mfq_tensor_backend::Tensor norm2_weight;
    mfq_tensor_backend::Tensor norm2_bias;
    BiasLinear fc1;
    BiasLinear fc2;
    int64_t heads = 0;
    int64_t head_dim = 0;
    double layer_norm_eps = 1e-6;
    VisionActivation activation = VisionActivation::gelu_tanh;

    mfq_tensor_backend::Tensor forward(
            CudaExecutionContext &execution, mfq_tensor_backend::Tensor input,
            MfqOptional<mfq_tensor_backend::Tensor> mask = mfq_nullopt) const {
        using Tensor = mfq_tensor_backend::Tensor;
        if (heads <= 0 || head_dim <= 0) {
            throw std::runtime_error("vision block head geometry is invalid");
        }
        const int64_t batch = input.size(0);
        const int64_t tokens = input.size(1);
        auto split_heads = [&](Tensor value) {
            return value.reshape({batch, tokens, heads, head_dim}).transpose(1, 2).contiguous();
        };
        const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
        return mfq::models::vision_block(
            std::move(input),
            [&](const Tensor &x, int stage) {
                return vision_layer_norm(x, stage == 0 ? norm1_weight : norm2_weight,
                                         stage == 0 ? norm1_bias : norm2_bias,
                                         layer_norm_eps);
            },
            [&](Tensor x) {
                return mfq::models::vision_attention(
                    std::move(x),
                    [&](Tensor x) {
                        return split_heads(query.forward(execution, std::move(x)));
                    },
                    [&](Tensor x) {
                        return split_heads(key.forward(execution, std::move(x)));
                    },
                    [&](Tensor x) {
                        return split_heads(this->value.forward(execution, std::move(x)));
                    },
                    [&](Tensor q, Tensor k, Tensor v) {
                        return vision_non_causal_attention(
                            std::move(q), std::move(k), std::move(v), mask, scale);
                    },
                    [&](Tensor attended) {
                        return output.forward(
                            execution, attended.transpose(1, 2).reshape(
                                           {batch, tokens, heads * head_dim}));
                    });
            },
            [&](Tensor x) {
                return mfq::models::vision_mlp(
                    std::move(x),
                    [&](Tensor value) { return fc1.forward(execution, std::move(value)); },
                    [&](Tensor value) {
                        return vision_activation(std::move(value), activation);
                    },
                    [&](Tensor value) { return fc2.forward(execution, std::move(value)); });
            },
            [](Tensor residual, Tensor branch) {
                return (residual + branch.to(residual.scalar_type())).contiguous();
            });
    }
};

// The three per-head projections of a fused qkv checkpoint tensor.
inline std::array<BiasLinear, 3> split_fused_qkv(const BiasLinear &fused, int64_t width) {
    if (width <= 0 || fused.weight.out() != 3 * width) {
        throw std::runtime_error("fused vision qkv projection is not [3*w, hidden]");
    }
    return {fused.rows(0, width), fused.rows(width, width), fused.rows(2 * width, width)};
}

} // namespace mfq::cuda
