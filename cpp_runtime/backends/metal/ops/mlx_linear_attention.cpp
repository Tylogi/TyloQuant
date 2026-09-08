#include "mlx_linear_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr const char* kCachedDepthwiseConvHeader = R"METAL(
template <
    typename StateT,
    typename FirstT,
    typename SecondT,
    typename WeightT,
    typename BiasT>
inline float mfq_cached_depthwise_conv_value(
    const device StateT* state,
    const device FirstT* first,
    const device SecondT* second,
    const device WeightT* weight,
    const device BiasT* bias,
    uint batch,
    uint token,
    uint channel,
    uint tokens,
    uint channels,
    uint first_channels,
    uint kernel_size,
    uint dilation,
    uint state_length,
    bool has_bias) {
    float value = has_bias ? float(bias[channel]) : 0.0f;
    for (uint tap = 0u; tap < kernel_size; ++tap) {
        int source_token = int(token) + int(tap * dilation) -
            int(state_length);
        float source;
        if (source_token < 0) {
            uint state_row = uint(source_token + int(state_length));
            source = float(state[
                (batch * state_length + state_row) * channels + channel]);
        } else if (channel < first_channels) {
            source = float(first[
                (batch * tokens + uint(source_token)) * first_channels +
                channel]);
        } else {
            uint second_channels = channels - first_channels;
            source = float(second[
                (batch * tokens + uint(source_token)) * second_channels +
                channel - first_channels]);
        }
        value += source * float(weight[channel * kernel_size + tap]);
    }
    return value;
}

template <typename StateT, typename FirstT, typename SecondT>
inline float mfq_cached_depthwise_state_value(
    const device StateT* state,
    const device FirstT* first,
    const device SecondT* second,
    uint batch,
    uint state_row,
    uint channel,
    uint tokens,
    uint channels,
    uint first_channels,
    uint state_length) {
    uint combined = tokens + state_row;
    if (combined < state_length) {
        return float(state[
            (batch * state_length + combined) * channels + channel]);
    }
    uint source_token = combined - state_length;
    if (channel < first_channels) {
        return float(first[
            (batch * tokens + source_token) * first_channels + channel]);
    }
    uint second_channels = channels - first_channels;
    return float(second[
        (batch * tokens + source_token) * second_channels +
        channel - first_channels]);
}
)METAL";

constexpr const char* kGatedDeltaNetSource = R"METAL(
    constexpr uint SIMD_WIDTH = 32u;
    constexpr uint SIMD_GROUPS = 4u;
    constexpr uint ROWS = (uint(D) + SIMD_WIDTH - 1u) / SIMD_WIDTH;
    constexpr uint COLUMN_TILES = (uint(D) + SIMD_GROUPS - 1u) / SIMD_GROUPS;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint bh = workgroup / COLUMN_TILES;
    uint column = (workgroup - bh * COLUMN_TILES) * SIMD_GROUPS + simd_group;
    if (bh >= uint(B * HV) || column >= uint(D)) {
        return;
    }

    uint batch = bh / uint(HV);
    uint value_head = bh - batch * uint(HV);
    uint query_head = TILED_HEADS != 0
        ? value_head % uint(HQ)
        : value_head / uint(HV / HQ);
    uint state_offset = bh * uint(D * D);
    uint query_sequence = (batch * uint(HQ) + query_head) * uint(TOKENS);
    uint value_sequence = bh * uint(TOKENS);

    float state_values[ROWS];
    for (uint row = 0u; row < ROWS; ++row) {
        uint state_row = row * SIMD_WIDTH + lane;
        uint state_index = TRANSPOSED_STATE != 0
            ? column * uint(D) + state_row
            : state_row * uint(D) + column;
        state_values[row] = state_row < uint(D)
            ? state_in[state_offset + state_index]
            : 0.0f;
    }

    const float scale = 1.0f / sqrt(float(D));
    for (uint token = 0u; token < uint(TOKENS); ++token) {
        uint query_offset = (query_sequence + token) * uint(D);
        uint value_offset = (value_sequence + token) * uint(D);
        float key_values[ROWS];
        float query_values[ROWS];
        for (uint row = 0u; row < ROWS; ++row) {
            uint state_row = row * SIMD_WIDTH + lane;
            key_values[row] = state_row < uint(D)
                ? k[query_offset + state_row]
                : 0.0f;
            query_values[row] = state_row < uint(D)
                ? q[query_offset + state_row]
                : 0.0f;
        }

        float projected_key = 0.0f;
        if (KDA != 0) {
            for (uint row = 0u; row < ROWS; ++row) {
                uint state_row = row * SIMD_WIDTH + lane;
                float decay = state_row < uint(D)
                    ? exp(g[value_offset + state_row])
                    : 0.0f;
                projected_key +=
                    decay * state_values[row] * key_values[row];
            }
        } else {
            for (uint row = 0u; row < ROWS; ++row) {
                projected_key += state_values[row] * key_values[row];
            }
        }
        projected_key = simd_sum(projected_key);

        float beta_value = beta[value_sequence + token];
        float delta;
        if (KDA != 0) {
            delta =
                (v[value_offset + column] - projected_key) * beta_value;
        } else {
            float decay = exp(g[value_sequence + token]);
            delta =
                (v[value_offset + column] - decay * projected_key) *
                beta_value;
        }

        float result = 0.0f;
        if (KDA != 0) {
            for (uint row = 0u; row < ROWS; ++row) {
                uint state_row = row * SIMD_WIDTH + lane;
                float decay = state_row < uint(D)
                    ? exp(g[value_offset + state_row])
                    : 0.0f;
                state_values[row] =
                    decay * state_values[row] + key_values[row] * delta;
                result += state_values[row] * query_values[row];
            }
        } else {
            float decay = exp(g[value_sequence + token]);
            for (uint row = 0u; row < ROWS; ++row) {
                state_values[row] =
                    decay * state_values[row] + key_values[row] * delta;
                result += state_values[row] * query_values[row];
            }
        }
        result = simd_sum(result);
        if (lane == 0u) {
            out[value_offset + column] = result * scale;
        }
    }

    for (uint row = 0u; row < ROWS; ++row) {
        uint state_row = row * SIMD_WIDTH + lane;
        if (state_row < uint(D)) {
            uint state_index = TRANSPOSED_STATE != 0
                ? column * uint(D) + state_row
                : state_row * uint(D) + column;
            state_out[state_offset + state_index] = state_values[row];
        }
    }
)METAL";

constexpr const char* kSsmConvSiluSource = R"METAL(
    uint index = thread_position_in_grid.x;
    if (index >= uint(B * TOKENS * C)) {
        return;
    }
    uint channel = index % uint(C);
    uint row = index / uint(C);
    uint token = row % uint(TOKENS);
    uint batch = row / uint(TOKENS);
    uint input_offset =
        (batch * uint(TOKENS + K - 1) + token) * uint(C) + channel;
    float value = HAS_BIAS != 0 ? bias[channel] : 0.0f;
    for (uint tap = 0u; tap < uint(K); ++tap) {
        value += float(x[input_offset + tap * uint(C)])
            * weight[channel * uint(K) + tap];
    }
    out[index] = value / (1.0f + exp(-value));
)METAL";

constexpr const char* kCachedDepthwiseConvDecodeSource = R"METAL(
    uint index = thread_position_in_grid.x;
    uint total = uint(B) * uint(C);
    if (index >= total) return;
    uint batch = index / uint(C);
    uint channel = index % uint(C);
    uint state_base =
        (batch * uint(STATE_LENGTH)) * uint(C) + channel;
    uint input_index = batch * uint(C) + channel;

    float convolution = mfq_cached_depthwise_conv_value(
        state_in,
        input,
        input,
        weight,
        bias,
        batch,
        0u,
        channel,
        1u,
        uint(C),
        uint(C),
        uint(K),
        uint(DILATION),
        uint(STATE_LENGTH),
        HAS_BIAS != 0);
    float tail = 1.0f /
        (1.0f + metal::exp(metal::abs(convolution)));
    float gate = convolution < 0.0f ? tail : 1.0f - tail;
    output[input_index] = T(convolution * gate);

    for (uint position = 0u;
         position < uint(STATE_LENGTH);
         ++position) {
        state_out[state_base + position * uint(C)] = T(
            mfq_cached_depthwise_state_value(
                state_in,
                input,
                input,
                batch,
                position,
                channel,
                1u,
                uint(C),
                uint(C),
                uint(STATE_LENGTH)));
    }
)METAL";

constexpr const char* kLinearConvQkvSource = R"METAL(
    constexpr uint QK_TASKS = uint(B * TOKENS * 2 * NK);
    constexpr uint V_GROUPS = (uint(NV * DV) + 31u) / 32u;
    constexpr uint V_TASKS = uint(B * TOKENS) * V_GROUPS;
    constexpr uint STATE_SIZE = uint(B * (K - 1) * C);
    constexpr uint STATE_TASKS = (STATE_SIZE + 31u) / 32u;

    uint lane = thread_index_in_simdgroup;
    uint task = threadgroup_position_in_grid.x;

    if (task < QK_TASKS) {
        uint head = task % uint(NK);
        uint row = task / uint(NK);
        uint which = row & 1u;
        uint batch_token = row >> 1u;
        uint token = batch_token % uint(TOKENS);
        uint batch = batch_token / uint(TOKENS);
        uint channel_base = (which * uint(NK) + head) * uint(DK);
        float square_sum = 0.0f;
        float values[(uint(DK) + 31u) / 32u];

        uint local = 0u;
        for (uint dimension = lane;
             dimension < uint(DK);
             dimension += 32u) {
            uint channel = channel_base + dimension;
            float value = mfq_cached_depthwise_conv_value(
                state_in,
                qk,
                v_in,
                weight,
                bias,
                batch,
                token,
                channel,
                uint(TOKENS),
                uint(C),
                uint(QKC),
                uint(K),
                1u,
                uint(K - 1),
                HAS_BIAS != 0);
            value = value / (1.0f + exp(-value));
            values[local++] = value;
            square_sum += value * value;
        }
        square_sum = simd_sum(square_sum);
        float inverse = 1.0f / max(sqrt(square_sum), params[0]);
        local = 0u;
        for (uint dimension = lane;
             dimension < uint(DK);
             dimension += 32u) {
            uint output_index =
                (((batch * uint(NK) + head) * uint(TOKENS) + token) *
                     uint(DK) +
                 dimension);
            if (which == 0u) {
                q_out[output_index] = values[local++] * inverse;
            } else {
                k_out[output_index] = values[local++] * inverse;
            }
        }
        return;
    }

    task -= QK_TASKS;
    if (task < V_TASKS) {
        uint value_group = task % V_GROUPS;
        uint batch_token = task / V_GROUPS;
        uint token = batch_token % uint(TOKENS);
        uint batch = batch_token / uint(TOKENS);
        uint value_index = value_group * 32u + lane;
        if (value_index < uint(NV * DV)) {
            uint channel = uint(QKC) + value_index;
            float value = mfq_cached_depthwise_conv_value(
                state_in,
                qk,
                v_in,
                weight,
                bias,
                batch,
                token,
                channel,
                uint(TOKENS),
                uint(C),
                uint(QKC),
                uint(K),
                1u,
                uint(K - 1),
                HAS_BIAS != 0);
            value = value / (1.0f + exp(-value));
            uint head = value_index / uint(DV);
            uint dimension = value_index - head * uint(DV);
            v_out[
                (((batch * uint(NV) + head) * uint(TOKENS) + token) *
                     uint(DV) +
                 dimension)
            ] = value;
        }
        return;
    }

    task -= V_TASKS;
    if (task < STATE_TASKS) {
        uint index = task * 32u + lane;
        if (index < STATE_SIZE) {
            uint channel = index % uint(C);
            uint row = index / uint(C);
            uint state_row = row % uint(K - 1);
            uint batch = row / uint(K - 1);
            state_out[index] = mfq_cached_depthwise_state_value(
                state_in,
                qk,
                v_in,
                batch,
                state_row,
                channel,
                uint(TOKENS),
                uint(C),
                uint(QKC),
                uint(K - 1));
        }
    }
)METAL";

mlx::core::fast::CustomKernelFunction make_gdn_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_gated_delta_net",
        {"q", "k", "v", "g", "beta", "state_in"},
        {"out", "state_out"},
        kGatedDeltaNetSource,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& gdn_kernel() {
    static const auto kernel = make_gdn_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction make_ssm_conv_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_ssm_conv_silu",
        {"x", "weight", "bias"},
        {"out"},
        kSsmConvSiluSource,
        "",
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& ssm_conv_kernel() {
    static const auto kernel = make_ssm_conv_kernel();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
cached_depthwise_conv_decode_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Safe;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_cached_depthwise_conv_decode",
            {"input", "weight", "bias", "state_in"},
            {"output", "state_out"},
            kCachedDepthwiseConvDecodeSource,
            kCachedDepthwiseConvHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

mlx::core::fast::CustomKernelFunction make_linear_conv_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_linear_conv_qkv",
        {"state_in", "qk", "v_in", "weight", "bias", "params"},
        {"q_out", "k_out", "v_out", "state_out"},
        kLinearConvQkvSource,
        kCachedDepthwiseConvHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& linear_conv_kernel() {
    static const auto kernel = make_linear_conv_kernel();
    return kernel;
}

array float32_contiguous(const array& value) {
    auto result = value;
    if (result.dtype() != mlx::core::float32) {
        result = mlx::core::astype(result, mlx::core::float32);
    }
    return mlx::core::contiguous(result);
}

array floating_contiguous(const array& value) {
    auto result = value;
    if (result.dtype() != mlx::core::float16 &&
        result.dtype() != mlx::core::float32) {
        result = mlx::core::astype(result, mlx::core::float16);
    }
    return mlx::core::contiguous(result);
}

array real_contiguous(const array& value) {
    auto result = value;
    if (result.dtype() != mlx::core::float16 &&
        result.dtype() != mlx::core::bfloat16 &&
        result.dtype() != mlx::core::float32) {
        result = mlx::core::astype(result, mlx::core::float16);
    }
    return mlx::core::contiguous(result);
}

int checked_int(std::int64_t value, const char* name) {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            std::string(name) + " exceeds MLX Metal grid limits");
    }
    return static_cast<int>(value);
}

struct ConvWeight {
    array values;
    int kernel;
};

ConvWeight normalize_conv_weight(
    const array& weight,
    int channels) {
    auto source = float32_contiguous(weight);
    int kernel = 0;
    if (source.ndim() == 3 &&
        source.shape(0) == channels &&
        source.shape(1) == 1) {
        kernel = source.shape(2);
        source = mlx::core::reshape(
            source,
            Shape{channels, kernel});
    } else if (source.ndim() == 2 &&
               source.shape(0) == channels) {
        kernel = source.shape(1);
    } else if (source.ndim() == 2 &&
               source.shape(1) == channels) {
        kernel = source.shape(0);
        source = mlx::core::transpose(source);
    } else {
        throw std::runtime_error(
            "SSM convolution weight must have [C,1,K], [C,K], "
            "or [K,C] shape");
    }
    if (kernel <= 0) {
        throw std::runtime_error(
            "SSM convolution kernel width must be positive");
    }
    return {
        mlx::core::contiguous(source),
        kernel,
    };
}

ConvWeight normalize_cached_conv_weight(
    const array& weight,
    int channels) {
    auto source = real_contiguous(weight);
    int kernel = 0;
    if (source.ndim() == 3 &&
        source.shape(0) == channels &&
        source.shape(1) == 1) {
        kernel = source.shape(2);
        source = mlx::core::reshape(source, Shape{channels, kernel});
    } else if (source.ndim() == 2 &&
               source.shape(0) == channels) {
        kernel = source.shape(1);
    } else if (source.ndim() == 2 &&
               source.shape(1) == channels) {
        kernel = source.shape(0);
        source = mlx::core::transpose(source);
    } else {
        throw std::runtime_error(
            "cached depthwise convolution weight must have [C,1,K], "
            "[C,K], or [K,C] shape");
    }
    if (kernel <= 0) {
        throw std::runtime_error(
            "cached depthwise convolution kernel width must be positive");
    }
    return {mlx::core::contiguous(source), kernel};
}

struct ConvBias {
    array values;
    bool present;
};

ConvBias normalize_conv_bias(
    const std::optional<array>& bias,
    int channels) {
    if (!bias.has_value()) {
        return {
            mlx::core::zeros(Shape{channels}, mlx::core::float32),
            false,
        };
    }
    auto result = float32_contiguous(*bias);
    if (result.ndim() != 1 || result.shape(0) != channels) {
        throw std::runtime_error(
            "SSM convolution bias width mismatch");
    }
    return {std::move(result), true};
}

bool cached_depthwise_conv_fast_path_enabled() noexcept {
    static const bool enabled = [] {
        const char* setting =
            std::getenv("MFQ_METAL_CACHED_DEPTHWISE_CONV_FAST");
        return setting == nullptr || std::atoi(setting) != 0;
    }();
    return enabled;
}

std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
gdn_templates(
    int batch,
    int query_heads,
    int value_heads,
    int tokens,
    int dimension,
    bool key_decay_attention,
    bool transposed_state,
    bool tiled_heads) {
    return {
        {"B", batch},
        {"HQ", query_heads},
        {"HV", value_heads},
        {"TOKENS", tokens},
        {"D", dimension},
        {"KDA", static_cast<int>(key_decay_attention)},
        {"TRANSPOSED_STATE", static_cast<int>(transposed_state)},
        {"TILED_HEADS", static_cast<int>(tiled_heads)},
    };
}

} // namespace

MlxGatedDeltaNetResult gated_delta_net(
    const array& query,
    const array& key,
    const array& value,
    const array& gate,
    const array& beta,
    const std::optional<array>& state,
    bool transposed_state,
    bool tiled_heads) {
    auto q = float32_contiguous(query);
    auto k = float32_contiguous(key);
    auto v = float32_contiguous(value);
    auto g = float32_contiguous(gate);
    auto beta_values = float32_contiguous(beta);
    if (q.ndim() != 4) {
        throw std::runtime_error(
            "GDN query must have [B,Hq,T,D] shape");
    }
    const int batch = q.shape(0);
    const int query_heads = q.shape(1);
    const int tokens = q.shape(2);
    const int dimension = q.shape(3);
    if (batch <= 0 || query_heads <= 0 || tokens < 0 ||
        (dimension != 32 && dimension != 64 && dimension != 128)) {
        throw std::runtime_error(
            "GDN dimensions are invalid or unsupported");
    }
    if (k.shape() != q.shape() || v.ndim() != 4) {
        throw std::runtime_error("GDN key/value shape mismatch");
    }
    const int value_heads = v.shape(1);
    if (v.shape(0) != batch ||
        v.shape(2) != tokens ||
        v.shape(3) != dimension ||
        value_heads <= 0 ||
        value_heads % query_heads != 0) {
        throw std::runtime_error(
            "GDN value heads must be a multiple of query heads");
    }

    const bool key_decay_attention = g.ndim() == 4;
    const Shape scalar_gate_shape{
        batch,
        value_heads,
        tokens,
    };
    const Shape vector_gate_shape{
        batch,
        value_heads,
        tokens,
        dimension,
    };
    if (g.shape() !=
        (key_decay_attention ? vector_gate_shape : scalar_gate_shape)) {
        throw std::runtime_error("GDN gate shape mismatch");
    }
    if (beta_values.shape() != scalar_gate_shape) {
        throw std::runtime_error("GDN beta shape mismatch");
    }

    const Shape state_shape{
        batch,
        value_heads,
        dimension,
        dimension,
    };
    auto state_values = state.has_value()
        ? float32_contiguous(*state)
        : mlx::core::zeros(state_shape, mlx::core::float32);
    if (state_values.shape() != state_shape) {
        throw std::runtime_error("GDN state shape mismatch");
    }
    if (tokens == 0) {
        return {
            mlx::core::zeros(v.shape(), mlx::core::float32),
            std::move(state_values),
        };
    }

    const int column_tiles = (dimension + 3) / 4;
    const std::int64_t workgroups =
        static_cast<std::int64_t>(batch) *
        value_heads *
        column_tiles;
    auto outputs = gdn_kernel()(
        {q, k, v, g, beta_values, state_values},
        {v.shape(), state_shape},
        {mlx::core::float32, mlx::core::float32},
        {checked_int(workgroups * 128, "GDN grid"), 1, 1},
        {128, 1, 1},
        gdn_templates(
            batch,
            query_heads,
            value_heads,
            tokens,
            dimension,
            key_decay_attention,
            transposed_state,
            tiled_heads),
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs.at(0)),
        std::move(outputs.at(1)),
    };
}

array ssm_conv_silu(
    const array& input,
    const array& weight,
    int tokens,
    const std::optional<array>& bias) {
    auto source = floating_contiguous(input);
    if (source.ndim() != 3) {
        throw std::runtime_error(
            "SSM convolution input must have [B,K-1+T,C] shape");
    }
    const int batch = source.shape(0);
    const int length = source.shape(1);
    const int channels = source.shape(2);
    const auto packed_weight =
        normalize_conv_weight(weight, channels);
    const auto bias_values =
        normalize_conv_bias(bias, channels);
    if (batch <= 0 || tokens <= 0 || channels <= 0 ||
        length != tokens + packed_weight.kernel - 1) {
        throw std::runtime_error(
            "SSM convolution input length must equal T+K-1");
    }
    const std::int64_t size =
        static_cast<std::int64_t>(batch) *
        tokens *
        channels;
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
        templates{
            {"T", source.dtype()},
            {"B", batch},
            {"TOKENS", tokens},
            {"C", channels},
            {"K", packed_weight.kernel},
            {"HAS_BIAS", static_cast<int>(bias_values.present)},
        };
    auto outputs = ssm_conv_kernel()(
        {source, packed_weight.values, bias_values.values},
        {Shape{batch, tokens, channels}},
        {mlx::core::float32},
        {checked_int(size, "SSM convolution grid"), 1, 1},
        {checked_int(std::min<std::int64_t>(256, size),
                     "SSM convolution threadgroup"),
         1,
         1},
        std::move(templates),
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

MlxCachedDepthwiseConvResult cached_depthwise_conv_silu(
    const array& input,
    const array& weight,
    const std::optional<array>& state,
    int dilation,
    const std::optional<array>& bias) {
    auto source = real_contiguous(input);
    if (source.ndim() != 3) {
        throw std::runtime_error(
            "cached depthwise convolution input must have [B,T,C] shape");
    }
    const int batch = source.shape(0);
    const int tokens = source.shape(1);
    const int channels = source.shape(2);
    if (batch <= 0 || tokens <= 0 || channels <= 0 || dilation <= 0) {
        throw std::runtime_error(
            "cached depthwise convolution dimensions are invalid");
    }
    const auto packed_weight =
        normalize_cached_conv_weight(weight, channels);
    const auto bias_values = normalize_conv_bias(bias, channels);
    const std::int64_t state_length_wide =
        static_cast<std::int64_t>(packed_weight.kernel - 1) * dilation;
    if (state_length_wide < 0 ||
        state_length_wide > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "cached depthwise convolution state length is invalid");
    }
    const int state_length = static_cast<int>(state_length_wide);
    const Shape state_shape{batch, state_length, channels};
    auto state_values = state
        ? (state->dtype() == source.dtype()
               ? mlx::core::contiguous(*state)
               : mlx::core::contiguous(
                     mlx::core::astype(*state, source.dtype())))
        : mlx::core::zeros(state_shape, source.dtype());
    if (state_values.shape() != state_shape) {
        throw std::runtime_error(
            "cached depthwise convolution state shape mismatch");
    }

    const bool fast_decode =
        cached_depthwise_conv_fast_path_enabled() &&
        tokens == 1 && packed_weight.kernel >= 2 &&
        packed_weight.kernel <= 16 && state_length <= 64;
    if (fast_decode) {
        const std::int64_t size =
            static_cast<std::int64_t>(batch) * channels;
        auto outputs = cached_depthwise_conv_decode_kernel()(
            {
                source,
                packed_weight.values,
                bias_values.values,
                state_values,
            },
            {source.shape(), state_shape},
            {source.dtype(), source.dtype()},
            {checked_int(size, "cached depthwise convolution grid"), 1, 1},
            {
                checked_int(
                    std::min<std::int64_t>(256, size),
                    "cached depthwise convolution threadgroup"),
                1,
                1,
            },
            {
                {"T", source.dtype()},
                {"B", batch},
                {"C", channels},
                {"K", packed_weight.kernel},
                {"DILATION", dilation},
                {"STATE_LENGTH", state_length},
                {"HAS_BIAS", static_cast<int>(bias_values.present)},
            },
            std::nullopt,
            false,
            {});
        return {
            std::move(outputs.at(0)),
            std::move(outputs.at(1)),
        };
    }

    auto combined = mlx::core::concatenate({state_values, source}, 1);
    array output = bias_values.present
        ? mlx::core::broadcast_to(
              mlx::core::reshape(
                  bias_values.values, Shape{1, 1, channels}),
              Shape{batch, tokens, channels})
        : mlx::core::zeros(
              Shape{batch, tokens, channels}, mlx::core::float32);
    auto weight_float = mlx::core::astype(
        packed_weight.values, mlx::core::float32);
    for (int tap = 0; tap < packed_weight.kernel; ++tap) {
        auto tap_source = mlx::core::slice(
            combined,
            Shape{0, tap * dilation, 0},
            Shape{batch, tap * dilation + tokens, channels});
        auto coefficient = mlx::core::reshape(
            mlx::core::slice(
                weight_float,
                Shape{0, tap},
                Shape{channels, tap + 1}),
            Shape{1, 1, channels});
        output = output +
            mlx::core::astype(tap_source, mlx::core::float32) * coefficient;
    }
    output = output * mlx::core::sigmoid(output);
    auto next_state = state_length == 0
        ? mlx::core::slice(
              combined, Shape{0, 0, 0}, Shape{batch, 0, channels})
        : mlx::core::slice(
              combined,
              Shape{0, combined.shape(1) - state_length, 0},
              Shape{batch, combined.shape(1), channels});
    return {
        mlx::core::astype(output, source.dtype()),
        mlx::core::contiguous(next_state),
    };
}

MlxLinearConvQkvResult linear_conv_qkv(
    const array& state,
    const array& qk,
    const array& value,
    const array& weight,
    int key_heads,
    int value_heads,
    int key_head_dimension,
    int value_head_dimension,
    const std::optional<array>& bias,
    float eps) {
    auto qk_values = floating_contiguous(qk);
    auto value_values = floating_contiguous(value);
    auto state_values = float32_contiguous(state);
    if (qk_values.ndim() != 3 ||
        value_values.ndim() != 3 ||
        state_values.ndim() != 3) {
        throw std::runtime_error(
            "linear_conv_qkv requires rank-three state, qk, and value");
    }
    const int batch = qk_values.shape(0);
    const int tokens = qk_values.shape(1);
    if (batch <= 0 || tokens <= 0 ||
        key_heads <= 0 || value_heads <= 0 ||
        key_head_dimension <= 0 ||
        value_head_dimension <= 0 ||
        value_heads % key_heads != 0) {
        throw std::runtime_error(
            "linear_conv_qkv dimensions are invalid");
    }
    const int expected_qk =
        2 * key_heads * key_head_dimension;
    const int expected_value =
        value_heads * value_head_dimension;
    if (qk_values.shape(2) != expected_qk ||
        value_values.shape() !=
            Shape{batch, tokens, expected_value}) {
        throw std::runtime_error(
            "linear_conv_qkv projection width mismatch");
    }
    if (value_values.dtype() != qk_values.dtype()) {
        const auto dtype =
            value_values.dtype() == mlx::core::float32 ||
                    qk_values.dtype() == mlx::core::float32
                ? mlx::core::float32
                : mlx::core::float16;
        qk_values = mlx::core::contiguous(
            mlx::core::astype(qk_values, dtype));
        value_values = mlx::core::contiguous(
            mlx::core::astype(value_values, dtype));
    }

    const int channels = expected_qk + expected_value;
    if (state_values.shape(0) != batch ||
        state_values.shape(2) != channels) {
        throw std::runtime_error(
            "linear_conv_qkv state width mismatch");
    }
    const int kernel = state_values.shape(1) + 1;
    if (kernel <= 1) {
        throw std::runtime_error(
            "linear_conv_qkv requires convolution width at least two");
    }
    const auto packed_weight =
        normalize_conv_weight(weight, channels);
    if (packed_weight.kernel != kernel) {
        throw std::runtime_error(
            "linear_conv_qkv state and weight kernel widths differ");
    }
    const auto bias_values =
        normalize_conv_bias(bias, channels);
    if (!std::isfinite(eps) || eps <= 0.0f) {
        throw std::runtime_error(
            "linear_conv_qkv epsilon must be finite and positive");
    }

    const std::int64_t qk_tasks =
        static_cast<std::int64_t>(batch) *
        tokens *
        2 *
        key_heads;
    const int value_groups = (expected_value + 31) / 32;
    const std::int64_t value_tasks =
        static_cast<std::int64_t>(batch) *
        tokens *
        value_groups;
    const std::int64_t state_tasks =
        (static_cast<std::int64_t>(state_values.size()) + 31) / 32;
    const std::int64_t workgroups =
        qk_tasks + value_tasks + state_tasks;
    const array parameters({eps}, Shape{1});
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
        templates{
            {"T", qk_values.dtype()},
            {"B", batch},
            {"TOKENS", tokens},
            {"NK", key_heads},
            {"NV", value_heads},
            {"DK", key_head_dimension},
            {"DV", value_head_dimension},
            {"QKC", expected_qk},
            {"C", channels},
            {"K", kernel},
            {"HAS_BIAS", static_cast<int>(bias_values.present)},
        };
    auto outputs = linear_conv_kernel()(
        {
            state_values,
            qk_values,
            value_values,
            packed_weight.values,
            bias_values.values,
            parameters,
        },
        {
            Shape{batch, key_heads, tokens, key_head_dimension},
            Shape{batch, key_heads, tokens, key_head_dimension},
            Shape{batch, value_heads, tokens, value_head_dimension},
            state_values.shape(),
        },
        {
            mlx::core::float32,
            mlx::core::float32,
            mlx::core::float32,
            mlx::core::float32,
        },
        {checked_int(workgroups * 32, "linear_conv_qkv grid"), 1, 1},
        {32, 1, 1},
        std::move(templates),
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs.at(0)),
        std::move(outputs.at(1)),
        std::move(outputs.at(2)),
        std::move(outputs.at(3)),
    };
}

} // namespace mfq::metal
