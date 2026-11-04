// Native checks for two shared capabilities:
// NoPE GQA through FullBlock and a latent MoE with non-gated relu-squared
// experts. CPU references use double precision; dense grouped projections
// additionally compare with the bounded legacy gather path.

#include "cuda_execution.h"
#include "core/ffn.h"
#include "core/full_block.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq_cuda_activation_ops.h"
#include <cuda_runtime.h>
#include <iomanip>
#include <set>
#include "mfq_tensor_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace tb = mfq_tensor_backend;
using tb::Tensor;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Tensor upload(const std::vector<float>& values, std::initializer_list<std::int64_t> shape,
              tb::ScalarType dtype) {
    return tb::tensor(values).reshape(shape).to(dtype).to(tb::kCUDA).contiguous();
}

std::vector<float> host_values(const Tensor& value) {
    auto host = value.contiguous().to(tb::kCPU).to(tb::kFloat32);
    return std::vector<float>(host.data_ptr<float>(), host.data_ptr<float>() + host.numel());
}

// Create a tensor on the device and return the exact values the kernel reads.
Tensor make_tensor(const std::vector<float>& values, std::initializer_list<std::int64_t> shape,
                   tb::ScalarType dtype, std::vector<double>* stored = nullptr) {
    auto cpu = tb::tensor(values).reshape(shape).to(dtype).contiguous();
    if (stored) {
        const auto exact = host_values(cpu);
        stored->assign(exact.begin(), exact.end());
    }
    return cpu.to(tb::kCUDA).contiguous();
}

QuantLinear dense_linear(Tensor weight) {
    QuantLinear linear;
    linear.kind = QuantLinearKind::Dense;
    linear.dense = std::move(weight);
    return linear;
}

std::vector<float> pattern(std::size_t count, float amplitude, float salt) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i)
        values[i] = amplitude * std::sin(float(i) * 0.137f + salt) +
                    0.2f * amplitude * std::cos(float(i) * 0.051f - salt);
    return values;
}

double rms_norm_value(const std::vector<double>& row, std::size_t column, double eps) {
    double square = 0.0;
    for (double value : row) square += value * value;
    return row[column] / std::sqrt(square / double(row.size()) + eps);
}

struct AttentionGeometry {
    int hidden = 16;
    int heads = 4;
    int kv_heads = 2;
    int head_dim = 4;
    double eps = 1e-6;
};

struct AttentionWeights {
    std::vector<double> q, k, v, o;
};

// No-RoPE (or rotate-half RoPE) causal GQA reference over one cached prefix.
std::vector<double> reference_attention(const std::vector<double>& input,
                                        const AttentionGeometry& g,
                                        const AttentionWeights& w,
                                        const std::vector<int64_t>& positions,
                                        bool rope, double base) {
    const int tokens = static_cast<int>(positions.size());
    const int width = g.heads * g.head_dim;
    const int kv_width = g.kv_heads * g.head_dim;
    const int half = g.head_dim / 2;
    std::vector<double> normalized(static_cast<std::size_t>(tokens) * g.hidden);
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> row(input.begin() + t * g.hidden, input.begin() + (t + 1) * g.hidden);
        for (int i = 0; i < g.hidden; ++i)
            normalized[t * g.hidden + i] = rms_norm_value(row, i, g.eps);
    }
    auto project = [&](const std::vector<double>& weight, int output) {
        std::vector<double> result(static_cast<std::size_t>(tokens) * output);
        for (int t = 0; t < tokens; ++t)
            for (int o = 0; o < output; ++o) {
                double value = 0.0;
                for (int i = 0; i < g.hidden; ++i)
                    value += weight[o * g.hidden + i] * normalized[t * g.hidden + i];
                result[t * output + o] = value;
            }
        return result;
    };
    auto q = project(w.q, width);
    auto k = project(w.k, kv_width);
    auto v = project(w.v, kv_width);
    auto rotate = [&](std::vector<double>& value, int heads, int token, int head) {
        if (!rope) return;
        const double position = double(positions[token]);
        for (int pair = 0; pair < half; ++pair) {
            const double frequency = std::pow(base, -2.0 * pair / double(g.head_dim));
            const double angle = position * frequency;
            const int first = (token * heads + head) * g.head_dim + pair;
            const int second = first + half;
            const double x0 = value[first], x1 = value[second];
            value[first] = x0 * std::cos(angle) - x1 * std::sin(angle);
            value[second] = x1 * std::cos(angle) + x0 * std::sin(angle);
        }
    };
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < g.heads; ++h) rotate(q, g.heads, t, h);
        for (int h = 0; h < g.kv_heads; ++h) rotate(k, g.kv_heads, t, h);
    }
    const double scale = 1.0 / std::sqrt(double(g.head_dim));
    std::vector<double> attended(static_cast<std::size_t>(tokens) * width);
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < g.heads; ++h) {
            const int kv_head = h / (g.heads / g.kv_heads);
            std::vector<double> scores(static_cast<std::size_t>(t + 1));
            double maximum = -std::numeric_limits<double>::infinity();
            for (int s = 0; s <= t; ++s) {
                double dot = 0.0;
                for (int d = 0; d < g.head_dim; ++d)
                    dot += q[(t * g.heads + h) * g.head_dim + d] *
                           k[(s * g.kv_heads + kv_head) * g.head_dim + d];
                scores[s] = dot * scale;
                maximum = std::max(maximum, scores[s]);
            }
            double sum = 0.0;
            for (double& score : scores) {
                score = std::exp(score - maximum);
                sum += score;
            }
            for (int d = 0; d < g.head_dim; ++d) {
                double value = 0.0;
                for (int s = 0; s <= t; ++s)
                    value += scores[s] / sum *
                             v[(s * g.kv_heads + kv_head) * g.head_dim + d];
                attended[(t * g.heads + h) * g.head_dim + d] = value;
            }
        }
    }
    std::vector<double> output(static_cast<std::size_t>(tokens) * g.hidden);
    for (int t = 0; t < tokens; ++t)
        for (int o = 0; o < g.hidden; ++o) {
            double value = input[t * g.hidden + o];
            for (int i = 0; i < width; ++i)
                value += w.o[o * width + i] * attended[t * width + i];
            output[t * g.hidden + o] = value;
        }
    return output;
}

struct AttentionFixture {
    FullBlock block;
    std::vector<double> q, k, v, o;
    std::vector<double> norm;
};

std::unique_ptr<AttentionFixture> make_attention_fixture(const AttentionGeometry& g, bool no_rope,
                                                         tb::ScalarType dtype) {
    auto holder = std::make_unique<AttentionFixture>();
    AttentionFixture& f = *holder;
    const int width = g.heads * g.head_dim;
    const int kv_width = g.kv_heads * g.head_dim;
    auto q_values = pattern(static_cast<std::size_t>(width) * g.hidden, 0.061f, 1.0f);
    auto k_values = pattern(static_cast<std::size_t>(kv_width) * g.hidden, 0.071f, 2.0f);
    auto v_values = pattern(static_cast<std::size_t>(kv_width) * g.hidden, 0.053f, 3.0f);
    auto o_values = pattern(static_cast<std::size_t>(g.hidden) * width, 0.067f, 4.0f);
    f.block.qkv.layers = {
        dense_linear(make_tensor(q_values, {width, g.hidden}, dtype, &f.q)),
        dense_linear(make_tensor(k_values, {kv_width, g.hidden}, dtype, &f.k)),
        dense_linear(make_tensor(v_values, {kv_width, g.hidden}, dtype, &f.v)),
    };
    f.block.qkv.outs = {width, kv_width, kv_width};
    f.block.o = dense_linear(make_tensor(o_values, {g.hidden, width}, dtype, &f.o));
    auto norm_values = std::vector<float>(g.hidden, 1.0f);
    f.block.attn_norm = make_tensor(norm_values, {g.hidden}, tb::kFloat32, &f.norm);
    f.block.ffn_norm = f.block.attn_norm;
    f.block.layer = 0;
    f.block.attention_heads = g.heads;
    f.block.kv_heads = g.kv_heads;
    f.block.attention_head_dim = g.head_dim;
    f.block.max_position_embeddings = 32;
    f.block.rms_norm_eps = g.eps;
    f.block.norm_weight_offset = 0.0;
    f.block.attention_scale = 0.0;
    f.block.no_rope = no_rope;

    const int intermediate = g.hidden;
    f.block.ffn.gate_up.layers = {
        dense_linear(make_tensor(std::vector<float>(g.hidden * intermediate, 0.0f),
                                 {intermediate, g.hidden}, dtype)),
        dense_linear(make_tensor(std::vector<float>(g.hidden * intermediate, 0.0f),
                                 {intermediate, g.hidden}, dtype)),
    };
    f.block.ffn.gate_up.outs = {intermediate, intermediate};
    f.block.ffn.down = dense_linear(make_tensor(
        std::vector<float>(std::size_t(g.hidden) * intermediate, 0.0f),
        {g.hidden, intermediate}, dtype));
    return holder;
}

void check_attention_case(bool no_rope, tb::ScalarType dtype, float tolerance) {
    CudaExecutionContext execution;
    const AttentionGeometry g;
    auto fixture = make_attention_fixture(g, no_rope, dtype);
    const int tokens = 5;
    auto input_values = pattern(static_cast<std::size_t>(tokens) * g.hidden, 0.4f, 7.0f);
    std::vector<double> input;
    auto input_tensor =
        make_tensor(input_values, {1, tokens, g.hidden}, tb::kFloat16, &input);
    auto positions = tb::arange(tokens, tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64));
    RopeCache rope(fixture->block.max_position_embeddings, g.head_dim, 10000.0);
    auto actual = fixture->block.forward(execution, input_tensor, positions, 0, mfq_nullopt, rope);
    auto actual_values = host_values(actual);

    AttentionWeights weights{fixture->q, fixture->k, fixture->v, fixture->o};
    std::vector<int64_t> position_values(tokens);
    std::iota(position_values.begin(), position_values.end(), 0);
    auto expected = reference_attention(input, g, weights, position_values, !no_rope, 10000.0);
    require(actual_values.size() == expected.size(), "NoPE/RoPE prefill output size");
    for (std::size_t i = 0; i < actual_values.size(); ++i)
        require(std::isfinite(actual_values[i]) &&
                    std::abs(actual_values[i] - expected[i]) <=
                        tolerance + 5e-3 * std::abs(expected[i]),
                "NoPE/RoPE prefill output differs at " + std::to_string(i));

    // Decode one token through the populated KV cache. If rotary were still
    // applied in the none case, the query/key would rotate at position 5.
    auto next_values = pattern(static_cast<std::size_t>(g.hidden), 0.3f, 11.0f);
    std::vector<double> next;
    auto next_tensor = make_tensor(next_values, {1, 1, g.hidden}, tb::kFloat16, &next);
    input.insert(input.end(), next.begin(), next.end());
    auto next_positions = tb::tensor(std::vector<int64_t>{tokens})
                              .to(tb::kCUDA).to(tb::kInt64);
    auto decode = fixture->block.forward(execution, next_tensor, next_positions, tokens,
                                        mfq_nullopt, rope);
    auto decode_values = host_values(decode);
    position_values.push_back(tokens);
    auto expected_decode =
        reference_attention(input, g, weights, position_values, !no_rope, 10000.0);
    const std::size_t offset = static_cast<std::size_t>(tokens) * g.hidden;
    for (int i = 0; i < g.hidden; ++i)
        require(std::isfinite(decode_values[static_cast<std::size_t>(i)]) &&
                    std::abs(decode_values[static_cast<std::size_t>(i)] -
                             expected_decode[offset + i]) <=
                        tolerance + 5e-3 * std::abs(expected_decode[offset + i]),
                "NoPE/RoPE decode output differs at " + std::to_string(i));
}

struct MoeGeometry {
    int hidden = 64;
    int latent = 16;
    int experts = 8;
    int top_k = 3;
    int expert_width = 32;
    int shared_width = 48;
};

struct MoeWeights {
    std::vector<double> router, router_bias, latent_down, latent_up;
    std::vector<double> up, down, shared_up, shared_down;
};

struct MoeTensors {
    std::vector<float> router, router_bias, latent_down, latent_up;
    std::vector<float> up, down, shared_up, shared_down;
};

MoeTensors make_moe_tensors(const MoeGeometry& g, tb::ScalarType dtype, MoeWeights& w) {
    MoeTensors t;
    t.router = pattern(static_cast<std::size_t>(g.experts) * g.hidden, 0.09f, 21.0f);
    t.router_bias = pattern(static_cast<std::size_t>(g.experts), 0.05f, 22.0f);
    t.latent_down = pattern(static_cast<std::size_t>(g.latent) * g.hidden, 0.08f, 23.0f);
    t.latent_up = pattern(static_cast<std::size_t>(g.hidden) * g.latent, 0.08f, 24.0f);
    t.up = pattern(static_cast<std::size_t>(g.experts) * g.expert_width * g.latent, 0.2f, 25.0f);
    t.down = pattern(static_cast<std::size_t>(g.experts) * g.latent * g.expert_width, 0.2f, 26.0f);
    t.shared_up = pattern(static_cast<std::size_t>(g.shared_width) * g.hidden, 0.15f, 27.0f);
    t.shared_down = pattern(static_cast<std::size_t>(g.hidden) * g.shared_width, 0.15f, 28.0f);
    auto store = [&](const std::vector<float>& values, std::initializer_list<std::int64_t> shape,
                     std::vector<double>& destination) {
        auto cpu = tb::tensor(values).reshape(shape).to(dtype).contiguous();
        const auto exact = host_values(cpu);
        destination.assign(exact.begin(), exact.end());
    };
    store(t.router, {g.experts, g.hidden}, w.router);
    store(t.router_bias, {g.experts}, w.router_bias);
    store(t.latent_down, {g.latent, g.hidden}, w.latent_down);
    store(t.latent_up, {g.hidden, g.latent}, w.latent_up);
    store(t.up, {g.experts, g.expert_width, g.latent}, w.up);
    store(t.down, {g.experts, g.latent, g.expert_width}, w.down);
    store(t.shared_up, {g.shared_width, g.hidden}, w.shared_up);
    store(t.shared_down, {g.hidden, g.shared_width}, w.shared_down);
    return t;
}

// Build the canonical router plus shared branch from already rounded values.
std::unique_ptr<FFN> make_moe_ffn(const MoeGeometry& g, tb::ScalarType dtype,
                                  const MoeWeights& w, bool with_latent) {
    auto ffn = std::make_unique<FFN>();
    ffn->is_moe = true;
    ffn->moe_ungated_experts = true;
    ffn->activation = mfq::models::MlpActivation::relu_squared;
    ffn->moe_top_k = g.top_k;
    ffn->moe_use_sigmoid = true;
    ffn->moe_normalize = true;
    ffn->moe_delayed_softmax = false;
    ffn->moe_router_scale = 5.0;
    ffn->moe_shared_ungated = true;
    auto tensor = [&](const std::vector<double>& values,
                      std::initializer_list<std::int64_t> shape) {
        std::vector<float> narrow(values.begin(), values.end());
        return upload(narrow, shape, dtype);
    };
    ffn->moe_router = tensor(w.router, {g.experts, g.hidden}).to(tb::kFloat32).contiguous();
    ffn->moe_router_bias = tensor(w.router_bias, {g.experts}).to(tb::kFloat32).contiguous();
    ffn->moe_dense_up = tensor(w.up, {g.experts, g.expert_width, g.latent});
    ffn->moe_dense_down = tensor(w.down, {g.experts, g.latent, g.expert_width});
    if (with_latent) {
        ffn->moe_latent_in = dense_linear(tensor(w.latent_down, {g.latent, g.hidden}));
        ffn->moe_latent_out = dense_linear(tensor(w.latent_up, {g.hidden, g.latent}));
    }
    ffn->shared = std::make_unique<FFN>();
    ffn->shared->activation = mfq::models::MlpActivation::relu_squared;
    ffn->shared->gate_up.layers = {
        dense_linear(tensor(w.shared_up, {g.shared_width, g.hidden}))};
    ffn->shared->gate_up.outs = {g.shared_width};
    ffn->shared->down = dense_linear(tensor(w.shared_down, {g.hidden, g.shared_width}));
    return ffn;
}

std::vector<double> reference_moe(const std::vector<double>& input, const MoeGeometry& g,
                                  const MoeWeights& w, bool with_latent, bool relu_squared) {
    const int tokens = static_cast<int>(input.size()) / g.hidden;
    auto project = [&](const std::vector<double>& weight, const std::vector<double>& source,
                       int rows, int in, int out) {
        std::vector<double> result(static_cast<std::size_t>(rows) * out);
        for (int r = 0; r < rows; ++r)
            for (int o = 0; o < out; ++o) {
                double value = 0.0;
                for (int i = 0; i < in; ++i) value += weight[o * in + i] * source[r * in + i];
                result[r * out + o] = value;
            }
        return result;
    };
    std::vector<double> scores(static_cast<std::size_t>(tokens) * g.experts);
    for (int t = 0; t < tokens; ++t)
        for (int e = 0; e < g.experts; ++e) {
            double logit = 0.0;
            for (int i = 0; i < g.hidden; ++i)
                logit += w.router[e * g.hidden + i] * input[t * g.hidden + i];
            scores[t * g.experts + e] = 1.0 / (1.0 + std::exp(-logit));
        }
    std::vector<double> latent = input;
    if (with_latent) latent = project(w.latent_down, input, tokens, g.hidden, g.latent);
    const int routed_width = with_latent ? g.latent : g.hidden;
    std::vector<double> routed(static_cast<std::size_t>(tokens) * routed_width, 0.0);
    for (int t = 0; t < tokens; ++t) {
        std::vector<int> order(g.experts);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            const double left = scores[t * g.experts + a] + w.router_bias[a];
            const double right = scores[t * g.experts + b] + w.router_bias[b];
            return left > right || (left == right && a < b);
        });
        double denominator = 0.0;
        for (int rank = 0; rank < g.top_k; ++rank)
            denominator += scores[t * g.experts + order[rank]];
        for (int rank = 0; rank < g.top_k; ++rank) {
            const int expert = order[rank];
            const double weight = scores[t * g.experts + expert] / denominator * 5.0;
            std::vector<double> up(g.expert_width);
            for (int o = 0; o < g.expert_width; ++o) {
                double value = 0.0;
                for (int i = 0; i < g.latent; ++i)
                    value += w.up[(static_cast<std::size_t>(expert) * g.expert_width + o) *
                                      g.latent + i] *
                             latent[t * g.latent + i];
                double activated = 0.0;
                if (relu_squared) {
                    activated = std::max(0.0, value);
                    activated *= activated;
                } else {
                    activated = value / (1.0 + std::exp(-value)) * value;
                }
                up[o] = activated;
            }
            for (int o = 0; o < routed_width; ++o) {
                double value = 0.0;
                for (int i = 0; i < g.expert_width; ++i)
                    value += w.down[(static_cast<std::size_t>(expert) * g.latent + o) *
                                        g.expert_width + i] *
                             up[i];
                routed[t * routed_width + o] += weight * value;
            }
        }
    }
    if (with_latent) routed = project(w.latent_up, routed, tokens, g.latent, g.hidden);
    auto shared_up = project(w.shared_up, input, tokens, g.hidden, g.shared_width);
    for (double& value : shared_up) {
        const double positive = std::max(0.0, value);
        value = positive * positive;
    }
    auto shared = project(w.shared_down, shared_up, tokens, g.shared_width, g.hidden);
    for (std::size_t i = 0; i < routed.size(); ++i) routed[i] += shared[i];
    return routed;
}

double max_difference(const std::vector<double>& actual, const std::vector<double>& expected) {
    require(actual.size() == expected.size(), "difference size mismatch");
    double peak = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i)
        peak = std::max(peak, std::abs(actual[i] - expected[i]));
    return peak;
}

double max_difference(const std::vector<float>& actual, const std::vector<double>& expected) {
    return max_difference(std::vector<double>(actual.begin(), actual.end()), expected);
}

double max_difference(const std::vector<float>& actual, const std::vector<float>& expected) {
    return max_difference(std::vector<double>(actual.begin(), actual.end()),
                          std::vector<double>(expected.begin(), expected.end()));
}

void check_latent_removal_detected() {
    CudaExecutionContext execution;
    // A square latent keeps both variants runnable with the same expert bank.
    MoeGeometry g;
    g.latent = g.hidden;
    MoeWeights weights;
    make_moe_tensors(g, tb::kFloat16, weights);
    const int tokens = 3;
    std::vector<double> input;
    auto input_tensor = make_tensor(
        pattern(static_cast<std::size_t>(tokens) * g.hidden, 0.35f, 41.0f),
        {1, tokens, g.hidden}, tb::kFloat16, &input);
    auto with_latent = make_moe_ffn(g, tb::kFloat16, weights, true);
    auto without_latent = make_moe_ffn(g, tb::kFloat16, weights, false);
    auto expected = reference_moe(input, g, weights, true, true);
    auto expected_without = reference_moe(input, g, weights, false, true);
    auto actual = host_values(with_latent->forward(execution, input_tensor));
    auto removed = host_values(without_latent->forward(execution, input_tensor));
    const auto difference = max_difference(actual, expected);
    const auto removed_difference = max_difference(removed, expected_without);
    require(difference <= 3e-2f,
            "latent MoE does not match the double reference: " + std::to_string(difference));
    require(removed_difference <= 3e-2f,
            "latent removal reference is inconsistent: " + std::to_string(removed_difference));
    const auto removal_difference = max_difference(actual, removed);
    require(removal_difference > 1e-3,
            "latent projection removal was not detected: " + std::to_string(removal_difference));
}

void check_latent_moe(tb::ScalarType dtype, float tolerance) {
    CudaExecutionContext execution;
    const MoeGeometry g;
    const int tokens = 4;
    MoeWeights weights;
    make_moe_tensors(g, dtype, weights);
    auto ffn = make_moe_ffn(g, dtype, weights, true);
    std::vector<double> input;
    auto input_tensor = make_tensor(pattern(static_cast<std::size_t>(tokens) * g.hidden, 0.35f, 31.0f),
                                    {1, tokens, g.hidden}, tb::kFloat16, &input);
    auto actual = host_values(ffn->forward(execution, input_tensor));
    auto expected = reference_moe(input, g, weights, true, true);
    const auto difference = max_difference(actual, expected);
    require(difference <= tolerance,
            "latent relu-squared MoE differs from the double reference: " +
                std::to_string(difference));

    // Exact identity projections isolate activation from the shared branch and cancellation.
    auto activation_weights = weights;
    for (auto* values : {&activation_weights.latent_down, &activation_weights.latent_up,
                        &activation_weights.up, &activation_weights.down,
                        &activation_weights.shared_up, &activation_weights.shared_down}) {
        std::fill(values->begin(), values->end(), 0.0);
    }
    for (int column = 0; column < g.latent; ++column) {
        activation_weights.latent_down[column * g.hidden + column] = 1.0;
        activation_weights.latent_up[column * g.latent + column] = 1.0;
        for (int expert = 0; expert < g.experts; ++expert) {
            activation_weights.up[(expert * g.expert_width + column) * g.latent + column] = 1.0;
            activation_weights.down[(expert * g.latent + column) * g.expert_width + column] = 1.0;
        }
    }
    std::vector<float> activation_values(static_cast<std::size_t>(tokens) * g.hidden);
    for (std::size_t index = 0; index < activation_values.size(); ++index)
        activation_values[index] = index % 2 ? -0.5f : 0.5f;
    std::vector<double> activation_input;
    auto activation_tensor = make_tensor(activation_values, {1, tokens, g.hidden}, tb::kFloat16,
                                         &activation_input);
    auto activation_ffn = make_moe_ffn(g, dtype, activation_weights, true);
    auto activation_actual = host_values(activation_ffn->forward(execution, activation_tensor));
    auto activation_expected = reference_moe(activation_input, g, activation_weights, true, true);
    require(max_difference(activation_actual, activation_expected) <= tolerance,
            "isolated latent activation differs from the double reference");

    // Replacing relu^2 with a SwiGLU-style activation must change the result.
    auto swiglu = reference_moe(activation_input, g, activation_weights, true, false);
    require(max_difference(activation_expected, swiglu) > 4 * tolerance,
            "latent MoE fixture does not distinguish the activation within its error bound");
    const auto activation_difference = max_difference(activation_actual, swiglu);
    require(activation_difference > 2 * tolerance,
            "relu-squared activation swap was not detected: " +
                std::to_string(activation_difference));

    // An ungated bank never carries a gate projection.
    auto mixed = make_moe_ffn(g, dtype, weights, true);
    mixed->moe_gate.n_experts = g.experts;
    bool rejected = false;
    try {
        (void)mixed->forward(execution, input_tensor);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("cannot mix a gate") != std::string::npos;
    }
    require(rejected, "ungated routed MoE silently accepted a gate projection");
}

// A long prompt must not gather every route's whole expert matrix at once:
// row chunks bounded by the gather budget give the same projection.
void check_dense_routed_chunking() {
    const int64_t experts = 9, output = 6, width = 5, rows = 11, routes = 3;
    auto bank = upload(pattern(static_cast<std::size_t>(experts * output * width), 0.4f, 3.0f),
                       {experts, output, width}, tb::kFloat32);
    std::vector<float> ids_values(static_cast<std::size_t>(rows * routes));
    for (std::size_t i = 0; i < ids_values.size(); ++i)
        ids_values[i] = static_cast<float>((i * 5 + i / 3) % static_cast<std::size_t>(experts));
    auto ids = upload(ids_values, {rows, routes}, tb::kFloat32).to(tb::kInt64);
    auto flat = upload(pattern(static_cast<std::size_t>(rows * width), 0.6f, 7.0f), {rows, width},
                       tb::kFloat32);
    auto routed = upload(pattern(static_cast<std::size_t>(rows * routes * width), 0.5f, 11.0f),
                         {rows, routes, width}, tb::kFloat32);
    const std::size_t row_bytes = static_cast<std::size_t>(routes * output * width) * sizeof(float);
    const std::size_t unbounded = std::numeric_limits<std::size_t>::max();
    for (const auto& input : {flat, routed}) {
        const auto grouped = host_values(dense_routed_projection(bank, input, ids));
        const auto whole = host_values(dense_routed_projection_gather(bank, input, ids, unbounded));
        // Four rows per chunk leaves a three-row remainder.
        const auto chunked = host_values(dense_routed_projection_gather(bank, input, ids, 4 * row_bytes));
        const auto single = host_values(dense_routed_projection_gather(bank, input, ids, 1));
        require(whole.size() == static_cast<std::size_t>(rows * routes * output),
                "dense routed projection returned the wrong shape");
        const std::vector<double> expected(whole.begin(), whole.end());
        const double chunk_difference = max_difference(chunked, expected);
        const double single_difference = max_difference(single, expected);
        require(max_difference(grouped, expected) <= 1e-6, "grouped F32 tail projection differs");
        require(chunk_difference <= 1e-6 && single_difference <= 1e-6,
                "chunked dense routed projection differs: " + std::to_string(chunk_difference) +
                    " / " + std::to_string(single_difference));
    }
}

#include "dense_moe_checks.h"

std::vector<int64_t> reference_topk(const std::vector<float>& logits, const std::vector<float>& bias,
                                    int top_k) {
    const int experts = static_cast<int>(bias.size());
    std::vector<int64_t> result(top_k);
    std::vector<bool> used(static_cast<std::size_t>(experts), false);
    for (int rank = 0; rank < top_k; ++rank) {
        int best = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int e = 0; e < experts; ++e) {
            if (used[static_cast<std::size_t>(e)]) continue;
            const double score = 1.0 / (1.0 + std::exp(-double(logits[e]))) + bias[e];
            if (score > best_score || (score == best_score && (best < 0 || e < best))) {
                best = e;
                best_score = score;
            }
        }
        used[static_cast<std::size_t>(best)] = true;
        result[rank] = best;
    }
    return result;
}

void check_topk_512() {
    CudaExecutionContext execution;
    constexpr int top_k = 22;
    for (const int experts : {512, 1024}) {
        std::vector<float> logits(static_cast<std::size_t>(experts));
        for (int i = 0; i < experts; ++i)
            logits[static_cast<std::size_t>(i)] =
                0.6f * std::sin(float(i) * 0.171f) + 0.03f * float(i % 7);
        std::vector<float> bias(static_cast<std::size_t>(experts));
        for (int i = 0; i < experts; ++i)
            bias[static_cast<std::size_t>(i)] = 0.02f * std::cos(float(i) * 0.093f);
        auto selected = moe_topk_cuda(
            upload(logits, {1, experts}, tb::kFloat32), top_k, true, false, true, false,
            upload(bias, {experts}, tb::kFloat32), 1e-20, 5.0);
        auto ids = host_values(selected.at(0));
        auto expected = reference_topk(logits, bias, top_k);
        for (int rank = 0; rank < top_k; ++rank)
            require(int64_t(ids[static_cast<std::size_t>(rank)]) ==
                        expected[static_cast<std::size_t>(rank)],
                    "top-22 selected a different expert");
        for (std::size_t i = 0; i < ids.size(); ++i)
            for (std::size_t j = i + 1; j < ids.size(); ++j)
                require(ids[i] != ids[j], "top-22 selected a duplicate expert");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "--dense-moe-memory") {
            const std::string mode = argv[5];
            require(mode == "grouped" || mode == "gather", "unknown projection mode");
            measure_dense_workspace(argv[2], std::stoi(argv[3]), std::stoi(argv[4]), mode == "grouped");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--dense-moe-bench") {
            benchmark_dense_routed();
            return 0;
        }
        check_dense_grouped();
        check_dense_gated_layouts();
        check_attention_case(true, tb::kFloat16, 4e-3f);
        check_attention_case(false, tb::kFloat16, 4e-3f);
        check_latent_moe(tb::kFloat16, 3e-3f);
        check_latent_moe(tb::kBFloat16, 3e-2f);
        check_latent_removal_detected();
        check_dense_routed_chunking();
        check_topk_512();
        std::cout << "NoPE attention and latent MoE checks PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "NoPE/latent MoE test failed: " << error.what() << '\n';
        return 1;
    }
}
