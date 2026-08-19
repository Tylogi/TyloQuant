#include "mlx_minicpmo45.h"

#include "mlx_sampling.h"
#include "mlx_tensor.h"
#include "mlx_transformer.h"
#include "mlx_eval_timing.h"

#include "../../third_party/nlohmann/json.hpp"

#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/primitives.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Shape;
using mlx::core::array;

constexpr const char* kResamplerPositionAsset =
    "__mfq_asset__/minicpmo45-resampler-pos-embed-v1.bf16";

void configure_minicpmo_sdpa() {
    // MLX otherwise selects 256 partial-reduction blocks for an 8k decode
    // cache on large Apple GPUs. llama.cpp's vector attention uses far fewer
    // workgroups; 64 is the measured optimum for MiniCPM-o's 32:8 GQA and
    // 128-wide heads. Respect an explicit setting for tuning other devices.
    if (std::getenv("MLX_SDPA_BLOCKS") == nullptr) {
        ::setenv("MLX_SDPA_BLOCKS", "64", 0);
    }
}

int minicpmo_vision_batch_size() noexcept {
    const auto* value = std::getenv("MFQ_METAL_VISION_BATCH_SIZE");
    return value == nullptr
        ? 2
        : std::max(0, std::atoi(value));
}

bool minicpmo_vision_batchable_length(int tokens) noexcept {
    // The official processor normalizes every video frame and image slice to
    // roughly 448x448, which produces about 1k ViT patches.  Smaller or much
    // larger raw tensor inputs are valid at the native boundary, but MLX can
    // select a different GEMM reduction for those uncommon geometries when a
    // batch is split.  Keep them on the original whole-batch path so batching
    // remains bitwise transparent for every accepted input geometry.
    return tokens >= 900 && tokens <= 1100;
}

bool minicpmo_vision_profile_enabled() noexcept {
    const auto* value = std::getenv("MFQ_METAL_VISION_PROFILE");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

array dense(const MfqContainer& model, const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" && record.dtype != "F16" &&
        record.dtype != "F32" && record.dtype != "I32" &&
        record.dtype != "I64") {
        throw std::runtime_error(
            "MiniCPM-o dense tensor has unsupported dtype: " + name);
    }
    const auto mapped = model.map_record(name);
    return load_dense_array(record.dtype, mapped.view());
}

array as_dtype(const array& value, mlx::core::Dtype dtype) {
    return value.dtype() == dtype
        ? value
        : mlx::core::astype(value, dtype);
}

void log_bitwise_hash(
    const array& value,
    const char* environment,
    const char* label) {
    const auto* requested = std::getenv(environment);
    if (requested == nullptr || std::strcmp(requested, "0") == 0) {
        return;
    }
    auto materialized = mlx::core::contiguous(value);
    materialized.eval();
    const auto* bytes = materialized.data<std::uint8_t>();
    std::uint64_t first = 1469598103934665603ULL;
    std::uint64_t second = 0x9e3779b97f4a7c15ULL;
    for (std::size_t index = 0; index < materialized.nbytes(); ++index) {
        first ^= bytes[index];
        first *= 1099511628211ULL;
        second ^= static_cast<std::uint64_t>(bytes[index]) +
            0x9e3779b97f4a7c15ULL + (second << 6U) + (second >> 2U);
    }
    std::cerr
        << label << " bytes="
        << materialized.nbytes()
        << " fnv64=0x" << std::hex << first
        << " mix64=0x" << second << std::dec << '\n';
}

void log_vision_bitwise_hash(const array& value) {
    log_bitwise_hash(
        value,
        "MFQ_METAL_VISION_BITWISE_HASH",
        "minicpmo_vision_bitwise_hash");
}

void log_audio_bitwise_hash(const array& value) {
    log_bitwise_hash(
        value,
        "MFQ_METAL_AUDIO_BITWISE_HASH",
        "minicpmo_audio_bitwise_hash");
}

const mlx::core::fast::CustomKernelFunction&
mini_qk_norm_rope_kernel() {
    static const auto kernel = [] {
        mlx::core::CompileOptions options;
        options.math_mode = mlx::core::MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_minicpmo_qk_norm_rope_4head_32x8x128_f16",
            {
                "query",
                "key",
                "query_norm",
                "key_norm",
                "params",
            },
            {"query_out", "key_out"},
            R"METAL(
    constexpr uint HEADS_PER_TG = 4u;
    constexpr uint SIMD_GROUPS = HEADS_PER_TG * 2u;
    uint tid = thread_index_in_threadgroup;
    uint head_slot = tid >> 6u;
    uint dimension = tid & 63u;
    uint head =
        threadgroup_position_in_grid.x * HEADS_PER_TG + head_slot;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    bool is_query = head < uint(Q_HEADS);
    uint local_head = is_query ? head : head - uint(Q_HEADS);
    device const T* source = is_query ? query : key;
    device const float* norm = is_query ? query_norm : key_norm;
    uint offset = local_head * uint(HEAD_DIM);

    uint half_dim = uint(HEAD_DIM) / 2u;
    float raw_first = float(source[offset + dimension]);
    float raw_second = float(
        source[offset + dimension + half_dim]);
    float square_sum = simd_sum(
        raw_first * raw_first + raw_second * raw_second);
    threadgroup float partials[SIMD_GROUPS];
    threadgroup float inverse_rms[HEADS_PER_TG];
    threadgroup float angle_cosines[64];
    threadgroup float angle_sines[64];
    if (lane == 0u) partials[simd_group] = square_sum;
    if (tid < 64u) {
        float frequency = pow(
            params[1],
            -2.0f * float(tid) / float(HEAD_DIM));
        float angle = params[2] * frequency;
        angle_cosines[tid] = cos(angle);
        angle_sines[tid] = sin(angle);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dimension == 0u) {
        float total =
            partials[head_slot * 2u] +
            partials[head_slot * 2u + 1u];
        inverse_rms[head_slot] = rsqrt(
            total / float(HEAD_DIM) + params[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = inverse_rms[head_slot];

    float first = float(T(
        raw_first * norm[dimension] * inv_rms));
    float second = float(T(
        raw_second * norm[dimension + half_dim] * inv_rms));
    float angle_cos = angle_cosines[dimension];
    float angle_sin = angle_sines[dimension];
    float rotated_first = first * angle_cos - second * angle_sin;
    float rotated_second = second * angle_cos + first * angle_sin;
    if (is_query) {
        query_out[offset + dimension] = T(rotated_first);
        query_out[offset + dimension + half_dim] = T(rotated_second);
    } else {
        key_out[offset + dimension] = T(rotated_first);
        key_out[offset + dimension + half_dim] = T(rotated_second);
    }
)METAL",
            "#define T half\n"
            "#define Q_HEADS 32\n"
            "#define HEAD_DIM 128\n",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
mini_rope_table_kernel() {
    static const auto kernel = [] {
        mlx::core::CompileOptions options;
        options.math_mode = mlx::core::MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_minicpmo_rope_table_128_f32_v1",
            {"params"},
            {"table"},
            R"METAL(
    uint dimension = thread_position_in_grid.x;
    if (dimension >= 64u) return;
    float frequency = pow(
        params[0],
        -2.0f * float(dimension) / 128.0f);
    float angle = params[1] * frequency;
    table[dimension * 2u] = cos(angle);
    table[dimension * 2u + 1u] = sin(angle);
)METAL",
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

array mini_rope_table(float base, int position) {
    const array params(
        {base, static_cast<float>(position)},
        mlx::core::float32);
    auto outputs = mini_rope_table_kernel()(
        {params},
        {Shape{64, 2}},
        {mlx::core::float32},
        {64, 1, 1},
        {64, 1, 1},
        {},
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

std::pair<array, array> mini_qk_norm_rope(
    const array& query,
    const array& key,
    const MlxRmsNorm& query_norm,
    const MlxRmsNorm& key_norm,
    int query_heads,
    int key_heads,
    int head_dim,
    float base,
    int position) {
    if (query.dtype() != mlx::core::float16 ||
        key.dtype() != query.dtype() ||
        query.size() != static_cast<std::size_t>(
            query_heads * head_dim) ||
        key.size() != static_cast<std::size_t>(
            key_heads * head_dim) ||
        query_norm.width() != head_dim ||
        key_norm.width() != head_dim ||
        query_norm.eps() != key_norm.eps() ||
        head_dim != 128 ||
        position < 0) {
        throw std::runtime_error(
            "unsupported MiniCPM-o fused Q/K normalization shape");
    }
    const array params(
        {query_norm.eps(), base, static_cast<float>(position)},
        mlx::core::float32);
    auto outputs = mini_qk_norm_rope_kernel()(
        {
            mlx::core::contiguous(mlx::core::reshape(
                query,
                Shape{query_heads * head_dim})),
            mlx::core::contiguous(mlx::core::reshape(
                key,
                Shape{key_heads * head_dim})),
            query_norm.weight(),
            key_norm.weight(),
            params,
        },
        {
            Shape{1, query_heads, 1, head_dim},
            Shape{1, key_heads, 1, head_dim},
        },
        {query.dtype(), key.dtype()},
        {(query_heads + key_heads) * (head_dim / 2), 1, 1},
        {4 * (head_dim / 2), 1, 1},
        {},
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs.at(0)),
        std::move(outputs.at(1)),
    };
}

struct MiniQkKvCacheParams {
    float eps = 0.0f;
    float base = 0.0f;
    float position = 0.0f;
    float capacity = 0.0f;
};

class MiniQkKvCachePrimitive final
    : public mlx::core::UnaryPrimitive {
public:
    MiniQkKvCachePrimitive(
        mlx::core::Stream stream,
        MiniQkKvCacheParams params)
        : UnaryPrimitive(stream), params_(params) {}

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "MiniCPM-o inline Q/K cache post-processing requires Metal");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 8) {
            throw std::logic_error(
                "MiniCPM-o inline Q/K cache input mismatch");
        }
        output.set_data(
            mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(
            selected_stream.device);
        mlx::core::CompileOptions options;
        options.math_mode = mlx::core::MathMode::Fast;
        auto* library = device.get_library(
            "mfq_minicpmo_qk_norm_rope_cache_4head_v1",
            options,
            [] {
                return std::string(R"METAL(
#include <metal_stdlib>
using namespace metal;

struct MiniQkKvCacheParams {
    float eps;
    float base;
    float position;
    float capacity;
};

kernel void mfq_minicpmo_qk_norm_rope_cache_4head_v1(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const float* query_norm [[buffer(2)]],
    device const float* key_norm [[buffer(3)]],
    device const half* value [[buffer(4)]],
    device half* key_cache [[buffer(5)]],
    device half* value_cache [[buffer(6)]],
    device const float* rope_table [[buffer(7)]],
    device half* query_out [[buffer(8)]],
    constant MiniQkKvCacheParams& params [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid
        [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint Q_HEADS = 32u;
    constexpr uint HEAD_DIM = 128u;
    constexpr uint HEADS_PER_TG = 4u;
    constexpr uint SIMD_GROUPS = HEADS_PER_TG * 2u;
    uint head_slot = tid >> 6u;
    uint dimension = tid & 63u;
    uint head =
        threadgroup_position_in_grid.x * HEADS_PER_TG + head_slot;
    bool is_query = head < Q_HEADS;
    uint local_head = is_query ? head : head - Q_HEADS;
    device const half* source = is_query ? query : key;
    device const float* norm = is_query ? query_norm : key_norm;
    uint offset = local_head * HEAD_DIM;

    uint half_dim = HEAD_DIM / 2u;
    float raw_first = float(source[offset + dimension]);
    float raw_second = float(source[offset + dimension + half_dim]);
    float square_sum = simd_sum(
        raw_first * raw_first + raw_second * raw_second);
    threadgroup float partials[SIMD_GROUPS];
    threadgroup float inverse_rms[HEADS_PER_TG];
    if (lane == 0u) partials[simd_group] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dimension == 0u) {
        float total =
            partials[head_slot * 2u] +
            partials[head_slot * 2u + 1u];
        inverse_rms[head_slot] = rsqrt(
            total / float(HEAD_DIM) + params.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = inverse_rms[head_slot];

    float first = float(half(
        raw_first * norm[dimension] * inv_rms));
    float second = float(half(
        raw_second * norm[dimension + half_dim] * inv_rms));
    float angle_cos = rope_table[dimension * 2u];
    float angle_sin = rope_table[dimension * 2u + 1u];
    float rotated_first = first * angle_cos - second * angle_sin;
    float rotated_second = second * angle_cos + first * angle_sin;
    if (is_query) {
        query_out[offset + dimension] = half(rotated_first);
        query_out[offset + dimension + half_dim] = half(rotated_second);
    } else {
        uint capacity = uint(params.capacity);
        uint cache_base =
            (local_head * capacity + uint(params.position)) * HEAD_DIM;
        key_cache[cache_base + dimension] = half(rotated_first);
        key_cache[cache_base + dimension + half_dim] = half(rotated_second);
        value_cache[cache_base + dimension] = value[offset + dimension];
        value_cache[cache_base + dimension + half_dim] =
            value[offset + dimension + half_dim];
    }
}
)METAL");
            });
        auto* kernel = device.get_kernel(
            "mfq_minicpmo_qk_norm_rope_cache_4head_v1",
            library);
        auto& encoder = mlx::core::metal::get_command_encoder(
            selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 8; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)], index);
        }
        // The kernel mutates both persistent cache buffers in place.  MLX
        // cannot infer that from set_input_array(), so register the buffers
        // as outputs as well.  Without this, the following attention kernel
        // may read the cache before these writes are visible.
        encoder.register_output_array(inputs[5]);
        encoder.register_output_array(inputs[6]);
        encoder.set_output_array(output, 8);
        encoder.set_bytes(params_, 9);
        encoder.dispatch_threadgroups(
            MTL::Size(10, 1, 1),
            MTL::Size(256, 1, 1));
    }

    const char* name() const override {
        return "MiniQkKvCachePrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const MiniQkKvCachePrimitive*>(&other);
        return primitive != nullptr &&
            primitive->params_.eps == params_.eps &&
            primitive->params_.base == params_.base &&
            primitive->params_.position == params_.position &&
            primitive->params_.capacity == params_.capacity;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {Shape{1, 32, 1, 128}};
    }

private:
    MiniQkKvCacheParams params_;
};

array mini_qk_norm_rope_cache(
    const array& query,
    const array& key,
    const array& value,
    const array& rope_table,
    const MlxRmsNorm& query_norm,
    const MlxRmsNorm& key_norm,
    float base,
    int position,
    const array& key_cache,
    const array& value_cache,
    int capacity) {
    if (query.dtype() != mlx::core::float16 ||
        key.dtype() != query.dtype() ||
        value.dtype() != query.dtype() ||
        query.size() != 32u * 128u ||
        key.size() != 8u * 128u ||
        value.size() != 8u * 128u ||
        query_norm.width() != 128 ||
        key_norm.width() != 128 ||
        query_norm.eps() != key_norm.eps() ||
        rope_table.shape() != Shape{64, 2} ||
        rope_table.dtype() != mlx::core::float32 ||
        key_cache.shape() != Shape{1, 8, capacity, 128} ||
        value_cache.shape() != key_cache.shape() ||
        key_cache.dtype() != query.dtype() ||
        value_cache.dtype() != query.dtype() ||
        position < 0 || position >= capacity) {
        throw std::runtime_error(
            "unsupported MiniCPM-o inline Q/K cache shape");
    }
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::runtime_error(
            "MiniCPM-o inline Q/K cache post-processing requires Metal");
    }
    return array(
        Shape{1, 32, 1, 128},
        mlx::core::float16,
        std::make_shared<MiniQkKvCachePrimitive>(
            stream,
            MiniQkKvCacheParams{
                query_norm.eps(),
                base,
                static_cast<float>(position),
                static_cast<float>(capacity),
            }),
        std::vector<array>{
            mlx::core::contiguous(mlx::core::reshape(
                query, Shape{32 * 128})),
            mlx::core::contiguous(mlx::core::reshape(
                key, Shape{8 * 128})),
            query_norm.weight(),
            key_norm.weight(),
            mlx::core::contiguous(mlx::core::reshape(
                value, Shape{8 * 128})),
            key_cache,
            value_cache,
            rope_table,
        });
}

struct MiniKvCacheWriteParams {
    std::uint32_t position = 0;
    std::uint32_t capacity = 0;
};

class MiniKvCacheWritePrimitive final
    : public mlx::core::UnaryPrimitive {
public:
    MiniKvCacheWritePrimitive(
        mlx::core::Stream stream,
        MiniKvCacheWriteParams params)
        : UnaryPrimitive(stream), params_(params) {}

    void eval_cpu(
        const std::vector<array>&,
        array&) override {
        throw std::runtime_error(
            "MiniCPM-o combined KV cache write requires Metal");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 5) {
            throw std::logic_error(
                "MiniCPM-o combined KV cache write input mismatch");
        }
        // The output aliases the already-computed query. This preserves the
        // exact Q/K normalization and RoPE graph while making attention depend
        // on the single command that writes both persistent cache rows.
        output.copy_shared_buffer(inputs[0]);
        auto& selected_stream = stream();
        auto& device = mlx::core::metal::device(
            selected_stream.device);
        mlx::core::CompileOptions options;
        options.math_mode = mlx::core::MathMode::Fast;
        auto* library = device.get_library(
            "mfq_minicpmo_kv_cache_write_v1",
            options,
            [] {
                return std::string(R"METAL(
#include <metal_stdlib>
using namespace metal;

struct MiniKvCacheWriteParams {
    uint position;
    uint capacity;
};

kernel void mfq_minicpmo_kv_cache_write_v1(
    device half* query_passthrough [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device half* key_cache [[buffer(3)]],
    device half* value_cache [[buffer(4)]],
    constant MiniKvCacheWriteParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint HEAD_DIM = 128u;
    constexpr uint ELEMENTS = 8u * HEAD_DIM;
    if (tid >= ELEMENTS) return;
    uint head = tid / HEAD_DIM;
    uint dimension = tid - head * HEAD_DIM;
    uint cache_offset =
        (head * params.capacity + params.position) * HEAD_DIM + dimension;
    key_cache[cache_offset] = key[tid];
    value_cache[cache_offset] = value[tid];
}
)METAL");
            });
        auto* kernel = device.get_kernel(
            "mfq_minicpmo_kv_cache_write_v1",
            library);
        auto& encoder = mlx::core::metal::get_command_encoder(
            selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_output_array(output, 0);
        for (int index = 1; index < 5; ++index) {
            encoder.set_input_array(inputs[index], index);
        }
        // key_cache and value_cache are in-place outputs even though the
        // primitive returns the query passthrough.  Register them so MLX
        // inserts the buffer barrier required by the subsequent attention.
        encoder.register_output_array(inputs[3]);
        encoder.register_output_array(inputs[4]);
        encoder.set_bytes(params_, 5);
        encoder.dispatch_threads(
            MTL::Size(8 * 128, 1, 1),
            MTL::Size(256, 1, 1));
    }

    const char* name() const override {
        return "MiniKvCacheWritePrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const MiniKvCacheWritePrimitive*>(&other);
        return primitive != nullptr &&
            primitive->params_.position == params_.position &&
            primitive->params_.capacity == params_.capacity;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {Shape{1, 32, 1, 128}};
    }

private:
    MiniKvCacheWriteParams params_;
};

array mini_kv_cache_write(
    const array& query,
    const array& key,
    const array& value,
    const array& key_cache,
    const array& value_cache,
    int position,
    int capacity) {
    if (query.dtype() != mlx::core::float16 ||
        key.dtype() != query.dtype() ||
        value.dtype() != query.dtype() ||
        query.size() != 32u * 128u ||
        key.size() != 8u * 128u ||
        value.size() != 8u * 128u ||
        key_cache.shape() != Shape{1, 8, capacity, 128} ||
        value_cache.shape() != key_cache.shape() ||
        key_cache.dtype() != query.dtype() ||
        value_cache.dtype() != query.dtype() ||
        position < 0 || position >= capacity) {
        throw std::runtime_error(
            "unsupported MiniCPM-o combined KV cache write shape");
    }
    auto stream = mlx::core::default_stream(
        mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::runtime_error(
            "MiniCPM-o combined KV cache write requires Metal");
    }
    return array(
        Shape{1, 32, 1, 128},
        mlx::core::float16,
        std::make_shared<MiniKvCacheWritePrimitive>(
            stream,
            MiniKvCacheWriteParams{
                static_cast<std::uint32_t>(position),
                static_cast<std::uint32_t>(capacity),
            }),
        std::vector<array>{
            query,
            mlx::core::contiguous(mlx::core::reshape(
                key, Shape{8 * 128})),
            mlx::core::contiguous(mlx::core::reshape(
                value, Shape{8 * 128})),
            key_cache,
            value_cache,
        });
}

const mlx::core::fast::CustomKernelFunction&
mini_gqa_partial_kernel() {
    static const auto kernel = [] {
        mlx::core::CompileOptions options;
        options.math_mode = mlx::core::MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_minicpmo_gqa_partial_32x8x128_f16",
            {"query", "key", "value", "params"},
            {"stats", "partials"},
            R"METAL(
    constexpr uint QUERIES_PER_KV = uint(Q_HEADS) / uint(KV_HEADS);
    uint lane = thread_index_in_simdgroup;
    uint dimension = lane * 4u;
    uint workgroup = threadgroup_position_in_grid.x;
    uint blocks = uint(params[2]);
    uint kv_head = workgroup / blocks;
    uint block = workgroup - kv_head * blocks;
    uint sequence = uint(params[0]);
    uint capacity = uint(params[1]);
    uint tile = uint(params[3]);
    uint begin = block * tile;
    uint end = min(begin + tile, sequence);
    uint query_head_base = kv_head * QUERIES_PER_KV;

    float4 queries[QUERIES_PER_KV];
    float4 accumulators[QUERIES_PER_KV];
    float current_max[QUERIES_PER_KV];
    float current_sum[QUERIES_PER_KV];
    for (uint query_index = 0u;
         query_index < QUERIES_PER_KV;
         ++query_index) {
        uint query_offset =
            (query_head_base + query_index) * uint(HEAD_DIM) + dimension;
        queries[query_index] = float4(
            *reinterpret_cast<device const half4*>(query + query_offset));
        accumulators[query_index] = float4(0.0f);
        current_max[query_index] = -INFINITY;
        current_sum[query_index] = 0.0f;
    }

    uint token = begin;
    for (; token + 3u < end; token += 4u) {
        uint cache_offset0 =
            (kv_head * capacity + token) * uint(HEAD_DIM) + dimension;
        uint cache_offset1 = cache_offset0 + uint(HEAD_DIM);
        uint cache_offset2 = cache_offset1 + uint(HEAD_DIM);
        uint cache_offset3 = cache_offset2 + uint(HEAD_DIM);
        float4 key_value0 = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset0));
        float4 key_value1 = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset1));
        float4 key_value2 = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset2));
        float4 key_value3 = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset3));
        float4 value_element0 = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset0));
        float4 value_element1 = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset1));
        float4 value_element2 = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset2));
        float4 value_element3 = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset3));
        for (uint query_index = 0u;
             query_index < QUERIES_PER_KV;
             ++query_index) {
            float score0 = simd_sum(dot(
                queries[query_index], key_value0
            )) * 0.08838834764831845f;
            float score1 = simd_sum(dot(
                queries[query_index], key_value1
            )) * 0.08838834764831845f;
            float next_max01 = max(
                current_max[query_index], max(score0, score1));
            float previous01 = exp(
                current_max[query_index] - next_max01);
            float incoming0 = exp(score0 - next_max01);
            float incoming1 = exp(score1 - next_max01);
            current_sum[query_index] =
                current_sum[query_index] * previous01 +
                incoming0 + incoming1;
            current_max[query_index] = next_max01;
            accumulators[query_index] =
                accumulators[query_index] * previous01 +
                value_element0 * incoming0 +
                value_element1 * incoming1;

            float score2 = simd_sum(dot(
                queries[query_index], key_value2
            )) * 0.08838834764831845f;
            float score3 = simd_sum(dot(
                queries[query_index], key_value3
            )) * 0.08838834764831845f;
            float next_max23 = max(
                current_max[query_index], max(score2, score3));
            float previous23 = exp(
                current_max[query_index] - next_max23);
            float incoming2 = exp(score2 - next_max23);
            float incoming3 = exp(score3 - next_max23);
            current_sum[query_index] =
                current_sum[query_index] * previous23 +
                incoming2 + incoming3;
            current_max[query_index] = next_max23;
            accumulators[query_index] =
                accumulators[query_index] * previous23 +
                value_element2 * incoming2 +
                value_element3 * incoming3;
        }
    }
    for (; token + 1u < end; token += 2u) {
        uint cache_offset0 =
            (kv_head * capacity + token) * uint(HEAD_DIM) + dimension;
        uint cache_offset1 = cache_offset0 + uint(HEAD_DIM);
        float4 key_value0 = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset0));
        float4 key_value1 = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset1));
        float4 value_element0 = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset0));
        float4 value_element1 = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset1));
        for (uint query_index = 0u;
             query_index < QUERIES_PER_KV;
             ++query_index) {
            float score0 = simd_sum(dot(
                queries[query_index], key_value0
            )) * 0.08838834764831845f;
            float score1 = simd_sum(dot(
                queries[query_index], key_value1
            )) * 0.08838834764831845f;
            float next_max = max(
                current_max[query_index], max(score0, score1));
            float previous = exp(
                current_max[query_index] - next_max);
            float incoming0 = exp(score0 - next_max);
            float incoming1 = exp(score1 - next_max);
            current_sum[query_index] =
                current_sum[query_index] * previous +
                incoming0 + incoming1;
            current_max[query_index] = next_max;
            accumulators[query_index] =
                accumulators[query_index] * previous +
                value_element0 * incoming0 +
                value_element1 * incoming1;
        }
    }
    if (token < end) {
        uint cache_offset =
            (kv_head * capacity + token) * uint(HEAD_DIM) + dimension;
        float4 key_value = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset));
        float4 value_element = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset));
        for (uint query_index = 0u;
             query_index < QUERIES_PER_KV;
             ++query_index) {
            float score = simd_sum(dot(
                queries[query_index], key_value
            )) * 0.08838834764831845f;
            float next_max = max(current_max[query_index], score);
            float previous = exp(current_max[query_index] - next_max);
            float incoming = exp(score - next_max);
            current_sum[query_index] =
                current_sum[query_index] * previous + incoming;
            current_max[query_index] = next_max;
            accumulators[query_index] =
                accumulators[query_index] * previous +
                value_element * incoming;
        }
    }

    for (uint query_index = 0u;
         query_index < QUERIES_PER_KV;
         ++query_index) {
        uint query_head = query_head_base + query_index;
        uint block_index = query_head * blocks + block;
        if (lane == 0u) {
            stats[block_index * 2u] = current_max[query_index];
            stats[block_index * 2u + 1u] = current_sum[query_index];
        }
        uint partial_offset =
            block_index * uint(HEAD_DIM) + dimension;
        *reinterpret_cast<device float4*>(partials + partial_offset) =
            accumulators[query_index];
    }
)METAL",
            "#define T half\n"
            "#define Q_HEADS 32\n"
            "#define KV_HEADS 8\n"
            "#define HEAD_DIM 128\n",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
mini_gqa_hierarchical_partial_kernel() {
    static const auto kernel = [] {
        mlx::core::CompileOptions options;
        options.math_mode = mlx::core::MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_minicpmo_gqa_hierarchical_partial_32x8x128_f16",
            {"query", "key", "value", "params"},
            {"stats", "partials"},
            R"METAL(
    constexpr uint QUERIES_PER_KV = uint(Q_HEADS) / uint(KV_HEADS);
    constexpr uint SIMD_GROUPS = 8u;
    uint lane = thread_index_in_simdgroup;
    uint dimension = lane * 4u;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint blocks = uint(params[2]);
    uint kv_head = workgroup / blocks;
    uint block = workgroup - kv_head * blocks;
    uint sequence = uint(params[0]);
    uint capacity = uint(params[1]);
    uint tile = uint(params[3]);
    uint segment_begin = block * tile;
    uint segment_end = min(segment_begin + tile, sequence);
    uint segment_size = segment_end - segment_begin;
    uint simd_tile = (segment_size + SIMD_GROUPS - 1u) / SIMD_GROUPS;
    uint begin = min(
        segment_begin + simd_group * simd_tile,
        segment_end);
    uint end = min(begin + simd_tile, segment_end);
    uint query_head_base = kv_head * QUERIES_PER_KV;

    float4 queries[QUERIES_PER_KV];
    float4 accumulators[QUERIES_PER_KV];
    float current_max[QUERIES_PER_KV];
    float current_sum[QUERIES_PER_KV];
    for (uint query_index = 0u;
         query_index < QUERIES_PER_KV;
         ++query_index) {
        uint query_offset =
            (query_head_base + query_index) * uint(HEAD_DIM) + dimension;
        queries[query_index] = float4(
            *reinterpret_cast<device const half4*>(query + query_offset));
        accumulators[query_index] = float4(0.0f);
        current_max[query_index] = -INFINITY;
        current_sum[query_index] = 0.0f;
    }

    uint token = begin;
    for (; token + 3u < end; token += 4u) {
        uint cache_offset0 =
            (kv_head * capacity + token) * uint(HEAD_DIM) + dimension;
        uint cache_offset1 = cache_offset0 + uint(HEAD_DIM);
        uint cache_offset2 = cache_offset1 + uint(HEAD_DIM);
        uint cache_offset3 = cache_offset2 + uint(HEAD_DIM);
        float4 key_value0 = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset0));
        float4 key_value1 = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset1));
        float4 key_value2 = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset2));
        float4 key_value3 = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset3));
        float4 value_element0 = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset0));
        float4 value_element1 = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset1));
        float4 value_element2 = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset2));
        float4 value_element3 = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset3));
        for (uint query_index = 0u;
             query_index < QUERIES_PER_KV;
             ++query_index) {
            float score0 = simd_sum(dot(
                queries[query_index], key_value0
            )) * 0.08838834764831845f;
            float score1 = simd_sum(dot(
                queries[query_index], key_value1
            )) * 0.08838834764831845f;
            float next_max01 = max(
                current_max[query_index], max(score0, score1));
            float previous01 = exp(
                current_max[query_index] - next_max01);
            float incoming0 = exp(score0 - next_max01);
            float incoming1 = exp(score1 - next_max01);
            current_sum[query_index] =
                current_sum[query_index] * previous01 +
                incoming0 + incoming1;
            current_max[query_index] = next_max01;
            accumulators[query_index] =
                accumulators[query_index] * previous01 +
                value_element0 * incoming0 +
                value_element1 * incoming1;

            float score2 = simd_sum(dot(
                queries[query_index], key_value2
            )) * 0.08838834764831845f;
            float score3 = simd_sum(dot(
                queries[query_index], key_value3
            )) * 0.08838834764831845f;
            float next_max23 = max(
                current_max[query_index], max(score2, score3));
            float previous23 = exp(
                current_max[query_index] - next_max23);
            float incoming2 = exp(score2 - next_max23);
            float incoming3 = exp(score3 - next_max23);
            current_sum[query_index] =
                current_sum[query_index] * previous23 +
                incoming2 + incoming3;
            current_max[query_index] = next_max23;
            accumulators[query_index] =
                accumulators[query_index] * previous23 +
                value_element2 * incoming2 +
                value_element3 * incoming3;
        }
    }
    for (; token + 1u < end; token += 2u) {
        uint cache_offset0 =
            (kv_head * capacity + token) * uint(HEAD_DIM) + dimension;
        uint cache_offset1 = cache_offset0 + uint(HEAD_DIM);
        float4 key_value0 = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset0));
        float4 key_value1 = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset1));
        float4 value_element0 = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset0));
        float4 value_element1 = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset1));
        for (uint query_index = 0u;
             query_index < QUERIES_PER_KV;
             ++query_index) {
            float score0 = simd_sum(dot(
                queries[query_index], key_value0
            )) * 0.08838834764831845f;
            float score1 = simd_sum(dot(
                queries[query_index], key_value1
            )) * 0.08838834764831845f;
            float next_max = max(
                current_max[query_index], max(score0, score1));
            float previous = exp(
                current_max[query_index] - next_max);
            float incoming0 = exp(score0 - next_max);
            float incoming1 = exp(score1 - next_max);
            current_sum[query_index] =
                current_sum[query_index] * previous +
                incoming0 + incoming1;
            current_max[query_index] = next_max;
            accumulators[query_index] =
                accumulators[query_index] * previous +
                value_element0 * incoming0 +
                value_element1 * incoming1;
        }
    }
    if (token < end) {
        uint cache_offset =
            (kv_head * capacity + token) * uint(HEAD_DIM) + dimension;
        float4 key_value = float4(
            *reinterpret_cast<device const half4*>(key + cache_offset));
        float4 value_element = float4(
            *reinterpret_cast<device const half4*>(value + cache_offset));
        for (uint query_index = 0u;
             query_index < QUERIES_PER_KV;
             ++query_index) {
            float score = simd_sum(dot(
                queries[query_index], key_value
            )) * 0.08838834764831845f;
            float next_max = max(current_max[query_index], score);
            float previous = exp(current_max[query_index] - next_max);
            float incoming = exp(score - next_max);
            current_sum[query_index] =
                current_sum[query_index] * previous + incoming;
            current_max[query_index] = next_max;
            accumulators[query_index] =
                accumulators[query_index] * previous +
                value_element * incoming;
        }
    }

    threadgroup float group_maxima[QUERIES_PER_KV][SIMD_GROUPS];
    threadgroup float group_sums[QUERIES_PER_KV][SIMD_GROUPS];
    threadgroup float group_partials
        [QUERIES_PER_KV][SIMD_GROUPS][HEAD_DIM];
    threadgroup float combined_maxima[QUERIES_PER_KV];
    threadgroup float combined_sums[QUERIES_PER_KV];
    threadgroup float rescale[QUERIES_PER_KV][SIMD_GROUPS];
    for (uint query_index = 0u;
         query_index < QUERIES_PER_KV;
         ++query_index) {
        if (lane == 0u) {
            group_maxima[query_index][simd_group] =
                current_max[query_index];
            group_sums[query_index][simd_group] =
                current_sum[query_index];
        }
        *reinterpret_cast<threadgroup float4*>(
            group_partials[query_index][simd_group] + dimension) =
            accumulators[query_index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0u) {
        for (uint query_index = 0u;
             query_index < QUERIES_PER_KV;
             ++query_index) {
            float candidate = lane < SIMD_GROUPS
                ? group_maxima[query_index][lane]
                : -INFINITY;
            float maximum = simd_max(candidate);
            float factor = lane < SIMD_GROUPS
                ? exp(candidate - maximum)
                : 0.0f;
            if (lane < SIMD_GROUPS) {
                rescale[query_index][lane] = factor;
            }
            float sum = lane < SIMD_GROUPS
                ? group_sums[query_index][lane] * factor
                : 0.0f;
            sum = simd_sum(sum);
            if (lane == 0u) {
                combined_maxima[query_index] = maximum;
                combined_sums[query_index] = sum;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0u) {
        for (uint query_index = 0u;
             query_index < QUERIES_PER_KV;
             ++query_index) {
            float4 combined = float4(0.0f);
            for (uint group = 0u; group < SIMD_GROUPS; ++group) {
                float4 partial =
                    *reinterpret_cast<threadgroup float4*>(
                        group_partials[query_index][group] + dimension);
                combined += partial * rescale[query_index][group];
            }
            uint query_head = query_head_base + query_index;
            uint block_index = query_head * blocks + block;
            if (lane == 0u) {
                stats[block_index * 2u] = combined_maxima[query_index];
                stats[block_index * 2u + 1u] = combined_sums[query_index];
            }
            uint partial_offset =
                block_index * uint(HEAD_DIM) + dimension;
            *reinterpret_cast<device float4*>(partials + partial_offset) =
                combined;
        }
    }
)METAL",
            "#define T half\n"
            "#define Q_HEADS 32\n"
            "#define KV_HEADS 8\n"
            "#define HEAD_DIM 128\n",
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction&
mini_gqa_reduce_kernel() {
    static const auto kernel = [] {
        mlx::core::CompileOptions options;
        options.math_mode = mlx::core::MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_minicpmo_gqa_reduce_32x128_f16",
            {"stats", "partials", "params"},
            {"output"},
            R"METAL(
    constexpr uint SIMD_GROUPS = 4u;
    uint query_head = threadgroup_position_in_grid.x;
    uint dimension = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint blocks = uint(params[0]);
    threadgroup float reductions[SIMD_GROUPS];

    float local_max = -INFINITY;
    for (uint block = dimension; block < blocks; block += uint(HEAD_DIM)) {
        uint index = query_head * blocks + block;
        local_max = max(local_max, stats[index * 2u]);
    }
    local_max = simd_max(local_max);
    if (lane == 0u) reductions[simd_group] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        float partial = lane < SIMD_GROUPS ? reductions[lane] : -INFINITY;
        partial = simd_max(partial);
        if (lane == 0u) reductions[0] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = reductions[0];

    float local_sum = 0.0f;
    for (uint block = dimension; block < blocks; block += uint(HEAD_DIM)) {
        uint index = query_head * blocks + block;
        local_sum += stats[index * 2u + 1u] *
            exp(stats[index * 2u] - global_max);
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0u) reductions[simd_group] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        float partial = lane < SIMD_GROUPS ? reductions[lane] : 0.0f;
        partial = simd_sum(partial);
        if (lane == 0u) reductions[0] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_sum = reductions[0];

    float accumulator = 0.0f;
    for (uint block = 0u; block < blocks; ++block) {
        uint index = query_head * blocks + block;
        accumulator += partials[
            index * uint(HEAD_DIM) + dimension
        ] * exp(stats[index * 2u] - global_max);
    }
    output[query_head * uint(HEAD_DIM) + dimension] =
        T(accumulator / global_sum);
)METAL",
            "#define T half\n"
            "#define HEAD_DIM 128\n",
            true,
            false,
            options);
    }();
    return kernel;
}

array mini_gqa_attention(
    const array& query,
    const array& key_storage,
    const array& value_storage,
    int sequence,
    int capacity) {
    constexpr int query_heads = 32;
    constexpr int key_heads = 8;
    constexpr int head_dim = 128;
    if (query.shape() != Shape{1, query_heads, 1, head_dim} ||
        query.dtype() != mlx::core::float16 ||
        key_storage.shape() !=
            Shape{1, key_heads, capacity, head_dim} ||
        value_storage.shape() != key_storage.shape() ||
        key_storage.dtype() != query.dtype() ||
        value_storage.dtype() != query.dtype() ||
        sequence <= 0 || sequence > capacity) {
        throw std::runtime_error(
            "unsupported MiniCPM-o fused GQA attention shape");
    }
    const auto* hierarchical_setting = std::getenv(
        "MFQ_MINICPM_HIERARCHICAL_GQA");
    const bool hierarchical = hierarchical_setting != nullptr
        ? std::strcmp(hierarchical_setting, "0") != 0
        : sequence >= 1'024;
    // The hierarchical kernel has eight SIMD groups per block.  Keep enough
    // blocks to occupy large Apple GPUs at the 1k crossover, then scale to 16
    // by 4k.  More than 16 increases the final-reduction cost on M3 Ultra.
    int blocks = hierarchical
        ? std::clamp((sequence + 255) / 256, 8, 16)
        : std::min(48, (sequence + 127) / 128);
    if (const auto* setting = std::getenv("MFQ_MINICPM_GQA_BLOCKS")) {
        const int configured = std::atoi(setting);
        if (configured > 0) {
            blocks = std::min({configured, sequence, 64});
        }
    }
    const int tile = (sequence + blocks - 1) / blocks;
    const array partial_params(
        {sequence, capacity, blocks, tile},
        mlx::core::int32);
    const auto& partial_kernel = hierarchical
        ? mini_gqa_hierarchical_partial_kernel()
        : mini_gqa_partial_kernel();
    const int partial_threadgroup = hierarchical ? 256 : 32;
    auto partial = partial_kernel(
        {query, key_storage, value_storage, partial_params},
        {
            Shape{query_heads, blocks, 2},
            Shape{query_heads, blocks, head_dim},
        },
        {
            mlx::core::float32,
            mlx::core::float32,
        },
        {key_heads * blocks * partial_threadgroup, 1, 1},
        {partial_threadgroup, 1, 1},
        {},
        std::nullopt,
        false,
        {});
    const array reduce_params({blocks}, mlx::core::int32);
    auto output = mini_gqa_reduce_kernel()(
        {partial.at(0), partial.at(1), reduce_params},
        {Shape{1, query_heads, 1, head_dim}},
        {query.dtype()},
        {query_heads * head_dim, 1, 1},
        {head_dim, 1, 1},
        {},
        std::nullopt,
        false,
        {}).front();
    return output;
}

bool use_mini_gqa_attention(int sequence) {
    // The paired-token GQA kernel crosses MLX SDPA at roughly 1k cached
    // tokens on large Apple GPUs and pulls progressively farther ahead as the
    // cache grows. Keep SDPA for the short-cache region where its launch path
    // is still cheaper.
    constexpr int automatic_threshold = 1'024;
    const auto* setting = std::getenv(
        "MFQ_MINICPM_FUSED_GQA_ATTENTION");
    if (setting != nullptr) {
        if (std::strcmp(setting, "0") == 0) return false;
        if (std::strcmp(setting, "1") == 0) return true;
    }
    return sequence >= automatic_threshold;
}

void report_decode_components(
    bool enabled,
    int step,
    int cache_position,
    const std::chrono::steady_clock::time_point& started,
    const detail::ComponentProfile& profile) {
    if (!enabled) return;
    const double wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started)
            .count();
    const double evaluated_ms = profile.evaluated_ms();
    std::cout
        << "component_profile model=minicpmo"
        << " step=" << step
        << " cache_position=" << cache_position
        << " wall_ms=" << std::fixed << std::setprecision(3)
        << wall_ms
        << " evaluated_ms=" << evaluated_ms
        << " unscoped_ms="
        << std::max(0.0, wall_ms - evaluated_ms)
        << std::endl;
    for (const auto& [name, timing] : profile.timings()) {
        std::cout
            << "component_cost model=minicpmo"
            << " step=" << step
            << " name=" << name
            << " ms=" << timing.elapsed_ms
            << " calls=" << timing.evaluations
            << " pct_evaluated="
            << (evaluated_ms > 0.0
                    ? 100.0 * timing.elapsed_ms / evaluated_ms
                    : 0.0)
            << std::endl;
    }
}

array relu(const array& value) {
    return mlx::core::maximum(value, array(0.0f, value.dtype()));
}

array gelu(const array& value, bool tanh_approximation = false) {
    if (tanh_approximation) {
        constexpr float c = 0.7978845608028654f;
        return 0.5f * value *
            (1.0f + mlx::core::tanh(
                c * (value + 0.044715f * value * value * value)));
    }
    constexpr float inv_sqrt_two = 0.7071067811865475f;
    return 0.5f * value *
        (1.0f + mlx::core::erf(value * inv_sqrt_two));
}

array dense_linear(
    const array& input,
    const array& weight,
    const std::optional<array>& bias = std::nullopt) {
    if (weight.ndim() != 2 || input.shape(-1) != weight.shape(1)) {
        throw std::runtime_error(
            "MiniCPM-o dense linear dimensions are incompatible");
    }
    const auto output_dtype = input.dtype();
    auto output = mlx::core::matmul(
        as_dtype(input, weight.dtype()),
        mlx::core::transpose(weight));
    if (bias) {
        output = output + as_dtype(*bias, output.dtype());
    }
    return as_dtype(output, output_dtype);
}

class MiniLinear {
public:
    static MiniLinear load(
        const MfqContainer& model,
        const std::string& prefix,
        bool with_bias = true) {
        std::optional<array> bias;
        if (with_bias && model.contains(prefix + ".bias")) {
            bias = dense(model, prefix + ".bias");
        }
        return MiniLinear(
            MlxLinear::load(model, prefix + ".weight"),
            std::move(bias));
    }

    MiniLinear(MlxLinear weight, std::optional<array> bias)
        : weight_(std::move(weight)), bias_(std::move(bias)) {}

    array operator()(const array& input) const {
        const auto dtype = input.dtype();
        auto output = weight_(input);
        if (bias_) {
            output = output + as_dtype(*bias_, output.dtype());
        }
        return as_dtype(output, dtype);
    }

    array add_to(
        const array& input,
        const array& residual) const {
        const auto* fuse = std::getenv(
            "MFQ_METAL_FUSE_O_RESIDUAL");
        if (!bias_ &&
            (fuse == nullptr || std::strcmp(fuse, "0") != 0)) {
            if (const auto* weight = weight_.nint_weight_ref()) {
                return as_dtype(
                    weight->matmul_add(input, residual),
                    residual.dtype());
            }
        }
        return residual + (*this)(input);
    }

    const MlxNintWeight* unbiased_nint_weight() const noexcept {
        return bias_ ? nullptr : weight_.nint_weight_ref();
    }

    std::optional<MlxGroupedLinearWeightRef>
    unbiased_grouped_weight() const noexcept {
        return bias_ ? std::nullopt : weight_.grouped_weight_ref();
    }

private:
    MlxLinear weight_;
    std::optional<array> bias_;
};

std::optional<MlxGroupedLinear> make_mini_grouped_linear(
    std::initializer_list<const MiniLinear*> projections) {
    std::vector<MlxGroupedLinearWeightRef> weights;
    weights.reserve(projections.size());
    for (const auto* projection : projections) {
        const auto weight = projection->unbiased_grouped_weight();
        if (!weight) return std::nullopt;
        weights.push_back(*weight);
    }
    return MlxGroupedLinear(std::move(weights));
}

bool minicpmo_grouped_qkv_enabled() noexcept {
    const auto* value = std::getenv("MFQ_MINICPM_GROUPED_QKV");
    return value == nullptr || std::strcmp(value, "0") != 0;
}

class MiniLayerNorm {
public:
    static MiniLayerNorm load(
        const MfqContainer& model,
        const std::string& prefix,
        float eps) {
        return MiniLayerNorm(
            dense(model, prefix + ".weight"),
            dense(model, prefix + ".bias"),
            eps);
    }

    MiniLayerNorm(array weight, array bias, float eps)
        : weight_(std::move(weight)),
          bias_(std::move(bias)),
          eps_(eps) {
        if (weight_.ndim() != 1 || bias_.shape() != weight_.shape()) {
            throw std::runtime_error(
                "MiniCPM-o LayerNorm parameter shape mismatch");
        }
    }

    array operator()(const array& input) const {
        if (input.shape(-1) != weight_.shape(0)) {
            throw std::runtime_error(
                "MiniCPM-o LayerNorm input width mismatch");
        }
        const auto dtype = input.dtype();
        auto output = mlx::core::fast::layer_norm(
            input,
            std::optional<array>(as_dtype(weight_, dtype)),
            std::optional<array>(as_dtype(bias_, dtype)),
            eps_);
        return as_dtype(output, dtype);
    }

private:
    array weight_;
    array bias_;
    float eps_;
};

array additive_mask(
    const array& visible,
    mlx::core::Dtype dtype);

struct MiniQwen3Config {
    std::string model_type;
    int vocab = 0;
    int hidden = 0;
    int intermediate = 0;
    int layers = 0;
    int query_heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int maximum_context = 0;
    float rope_base = 1'000'000.0f;
    float norm_eps = 1e-6f;
    bool tie_embeddings = false;
    bool query_key_norm = true;
};

MiniQwen3Config language_config(
    const MfqContainer& model,
    std::int64_t context_override) {
    constexpr const char* asset = "__mfq_asset__/model_config.json";
    if (!model.contains(asset)) {
        throw std::runtime_error(
            "MiniCPM-o MFQ has no embedded model_config.json");
    }
    nlohmann::json config;
    try {
        config = nlohmann::json::parse(model.read_text(asset));
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(
            std::string("invalid MiniCPM-o config JSON: ") + error.what());
    }
    if (config.value("model_type", std::string{}) != "minicpmo" ||
        config.value("version", std::string{}) != "4.5") {
        throw std::runtime_error(
            "native MLX graph requires MiniCPM-o version 4.5");
    }
    MiniQwen3Config result;
    result.model_type = "minicpmo";
    result.vocab = config.at("vocab_size").get<int>();
    result.hidden = config.at("hidden_size").get<int>();
    result.intermediate = config.at("intermediate_size").get<int>();
    result.layers = config.at("num_hidden_layers").get<int>();
    result.query_heads = config.at("num_attention_heads").get<int>();
    result.kv_heads = config.at("num_key_value_heads").get<int>();
    result.head_dim = config.value(
        "head_dim", result.hidden / result.query_heads);
    result.maximum_context =
        config.at("max_position_embeddings").get<int>();
    result.rope_base = config.value("rope_theta", 1'000'000.0f);
    result.norm_eps = config.value("rms_norm_eps", 1e-6f);
    result.tie_embeddings = config.value("tie_word_embeddings", false);
    if (result.hidden != 4096 || result.intermediate != 12288 ||
        result.layers != 36 || result.query_heads != 32 ||
        result.kv_heads != 8 || result.head_dim != 128 ||
        config.value("hidden_act", std::string{}) != "silu" ||
        config.value("attention_bias", true) ||
        config.value("use_sliding_window", true)) {
        throw std::runtime_error(
            "unsupported MiniCPM-o 4.5 Qwen3-8B configuration");
    }
    if (context_override > 0) {
        if (context_override > result.maximum_context ||
            context_override > std::numeric_limits<int>::max()) {
            throw std::runtime_error(
                "MiniCPM-o context override exceeds model capacity");
        }
        result.maximum_context = static_cast<int>(context_override);
    }
    return result;
}

MiniQwen3Config tts_config() {
    MiniQwen3Config result;
    result.model_type = "minicpmtts";
    result.vocab = 152064;
    result.hidden = 768;
    result.intermediate = 3072;
    result.layers = 20;
    result.query_heads = 12;
    result.kv_heads = 12;
    result.head_dim = 64;
    result.maximum_context = 4096;
    result.rope_base = 10000.0f;
    result.norm_eps = 1e-6f;
    result.tie_embeddings = true;
    result.query_key_norm = false;
    return result;
}

struct MiniQwen3Names {
    std::string embedding;
    std::string layer_prefix;
    std::string norm;
    std::string output;
};

MiniQwen3Names language_names() {
    return {
        "llm.model.embed_tokens.weight",
        "llm.model.layers.",
        "llm.model.norm.weight",
        "llm.lm_head.weight",
    };
}

MiniQwen3Names tts_names() {
    return {
        "tts.emb_text.weight",
        "tts.model.layers.",
        "tts.model.norm.weight",
        {},
    };
}

MlxRmsNorm mini_rms_norm(
    const MfqContainer& model,
    const std::string& name,
    float eps) {
    auto weight = dense(model, name);
    if (weight.ndim() != 1) {
        throw std::runtime_error(
            "MiniCPM-o RMSNorm weight must be one-dimensional: " + name);
    }
    return MlxRmsNorm(std::move(weight), eps, 0.0f);
}

class MiniQwen3Ffn {
public:
    static MiniQwen3Ffn load(
        const MfqContainer& model,
        const std::string& prefix) {
        return MiniQwen3Ffn(
            MiniLinear::load(model, prefix + ".gate_proj", false),
            MiniLinear::load(model, prefix + ".up_proj", false),
            MiniLinear::load(model, prefix + ".down_proj", false));
    }

    MiniQwen3Ffn(MiniLinear gate, MiniLinear up, MiniLinear down)
        : gate_(std::move(gate)),
          up_(std::move(up)),
          down_(std::move(down)) {}

    array operator()(const array& input) const {
        return forward(input, nullptr);
    }

    array add_to(
        const array& input,
        const array& residual) const {
        return forward(input, &residual);
    }

private:
    array forward(
        const array& input,
        const array* residual) const {
        const auto* gate_weight = gate_.unbiased_nint_weight();
        const auto* up_weight = up_.unbiased_nint_weight();
        if (input.size() == static_cast<std::size_t>(input.shape(-1)) &&
            gate_weight != nullptr && up_weight != nullptr &&
            gate_weight->can_fuse_swiglu(*up_weight)) {
            auto activated = as_dtype(
                gate_weight->swiglu(*up_weight, input), input.dtype());
            detail::profile_eval("minicpmo.ffn_gate_up", activated);
            auto output = residual != nullptr
                ? down_.add_to(activated, *residual)
                : down_(activated);
            detail::profile_eval("minicpmo.ffn_down", output);
            return output;
        }
        const auto gate = gate_(input);
        const auto up = up_(input);
        auto activated = gate * mlx::core::sigmoid(gate) * up;
        detail::profile_eval("minicpmo.ffn_gate_up", activated);
        auto output = residual != nullptr
            ? down_.add_to(activated, *residual)
            : down_(activated);
        detail::profile_eval("minicpmo.ffn_down", output);
        return output;
    }
    MiniLinear gate_;
    MiniLinear up_;
    MiniLinear down_;
};

class MiniQwen3Block {
public:
    static MiniQwen3Block load(
        const MfqContainer& model,
        const MiniQwen3Config& config,
        const MiniQwen3Names& names,
        int index) {
        const auto layer =
            names.layer_prefix + std::to_string(index);
        std::optional<MlxRmsNorm> query_norm;
        std::optional<MlxRmsNorm> key_norm;
        if (config.query_key_norm) {
            query_norm = mini_rms_norm(
                model,
                layer + ".self_attn.q_norm.weight",
                config.norm_eps);
            key_norm = mini_rms_norm(
                model,
                layer + ".self_attn.k_norm.weight",
                config.norm_eps);
        }
        return MiniQwen3Block(
            config,
            mini_rms_norm(
                model, layer + ".input_layernorm.weight", config.norm_eps),
            MiniLinear::load(model, layer + ".self_attn.q_proj", false),
            MiniLinear::load(model, layer + ".self_attn.k_proj", false),
            MiniLinear::load(model, layer + ".self_attn.v_proj", false),
            MiniLinear::load(model, layer + ".self_attn.o_proj", false),
            std::move(query_norm),
            std::move(key_norm),
            mini_rms_norm(
                model,
                layer + ".post_attention_layernorm.weight",
                config.norm_eps),
            MiniQwen3Ffn::load(model, layer + ".mlp"));
    }

    MiniQwen3Block(
        MiniQwen3Config config,
        MlxRmsNorm attention_norm,
        MiniLinear query,
        MiniLinear key,
        MiniLinear value,
        MiniLinear output,
        std::optional<MlxRmsNorm> query_norm,
        std::optional<MlxRmsNorm> key_norm,
        MlxRmsNorm ffn_norm,
        MiniQwen3Ffn ffn)
        : config_(std::move(config)),
          attention_norm_(std::move(attention_norm)),
          query_(std::move(query)),
          key_(std::move(key)),
          value_(std::move(value)),
          output_(std::move(output)),
          query_norm_(std::move(query_norm)),
          key_norm_(std::move(key_norm)),
          ffn_norm_(std::move(ffn_norm)),
          ffn_(std::move(ffn)) {
        qkv_ = make_mini_grouped_linear({&query_, &key_, &value_});
    }

    void reset_cache(int batch, int initial_capacity = 16) {
        cache_ = std::make_unique<MlxKvCache>(
            batch,
            config_.kv_heads,
            config_.maximum_context,
            config_.head_dim,
            std::min(initial_capacity, config_.maximum_context),
            config_.model_type == "minicpmo"
                ? mlx::core::float16
                : mlx::core::bfloat16);
    }

    void clear_cache() noexcept {
        cache_.reset();
    }

    void materialize_cache() {
        if (cache_) cache_->materialize();
    }

    MlxKvCacheSnapshot snapshot_cache() const {
        if (!cache_) {
            throw std::runtime_error(
                "MiniCPM-o Qwen3 KV cache is unavailable");
        }
        return cache_->snapshot();
    }

    void restore_cache(const MlxKvCacheSnapshot& snapshot) {
        reset_cache(snapshot.batch, snapshot.capacity);
        cache_->restore_snapshot(snapshot);
    }

    array forward(
        const array& input,
        const array* positions,
        int position,
        bool use_cache,
        const std::optional<array>& mask,
        const array* shared_rope_table = nullptr) {
        if (input.ndim() != 3 || input.shape(2) != config_.hidden) {
            throw std::runtime_error(
                "MiniCPM-o Qwen3 block input shape mismatch");
        }
        const int batch = input.shape(0);
        const int tokens = input.shape(1);
        auto normalized = attention_norm_(input);
        detail::profile_eval("minicpmo.attention_norm", normalized);
        std::vector<array> projections;
        if (tokens == 1 && minicpmo_grouped_qkv_enabled() &&
            qkv_ && qkv_->supports(normalized)) {
            projections = (*qkv_)(normalized);
        } else {
            projections = {
                query_(normalized),
                key_(normalized),
                value_(normalized),
            };
        }
        auto query_projection = std::move(projections.at(0));
        auto key_projection = std::move(projections.at(1));
        auto value_projection = std::move(projections.at(2));
        detail::profile_eval("minicpmo.q_proj", query_projection);
        detail::profile_eval("minicpmo.k_proj", key_projection);
        detail::profile_eval("minicpmo.v_proj", value_projection);
        array query = query_projection;
        array key = key_projection;
        if (use_cache && !cache_) reset_cache(batch);
        const auto* fused_qk_post = std::getenv(
            "MFQ_MINICPM_FUSED_QK_POST");
        const bool use_fused_qk_post =
            (fused_qk_post == nullptr ||
             std::strcmp(fused_qk_post, "0") != 0) &&
            batch == 1 &&
            tokens == 1 &&
            positions == nullptr &&
            query_norm_ && key_norm_ &&
            query_projection.dtype() == mlx::core::float16 &&
            key_projection.dtype() == mlx::core::float16 &&
            value_projection.dtype() == mlx::core::float16;
        const auto* fused_kv_post = std::getenv(
            "MFQ_MINICPM_FUSED_KV_POST");
        const bool use_combined_kv_write =
            use_fused_qk_post && use_cache && cache_ &&
            (fused_kv_post == nullptr ||
             std::strcmp(fused_kv_post, "0") != 0);
        const auto* inline_kv_post = std::getenv(
            "MFQ_MINICPM_INLINE_KV_POST");
        const bool use_inline_kv_post =
            use_combined_kv_write && shared_rope_table != nullptr &&
            (inline_kv_post == nullptr ||
             std::strcmp(inline_kv_post, "0") != 0);
        if (use_fused_qk_post) {
            if (use_inline_kv_post) {
                const int append_position = cache_->position();
                cache_->reserve_append(1);
                query = mini_qk_norm_rope_cache(
                    query_projection,
                    key_projection,
                    value_projection,
                    *shared_rope_table,
                    *query_norm_,
                    *key_norm_,
                    config_.rope_base,
                    append_position,
                    cache_->key_storage(),
                    cache_->value_storage(),
                    cache_->capacity());
            } else {
                auto fused = mini_qk_norm_rope(
                    query_projection,
                    key_projection,
                    *query_norm_,
                    *key_norm_,
                    config_.query_heads,
                    config_.kv_heads,
                    config_.head_dim,
                    config_.rope_base,
                    position);
                query = std::move(fused.first);
                key = std::move(fused.second);
            }
        } else {
            query = mlx::core::transpose(
                mlx::core::reshape(
                    query_projection,
                    Shape{
                        batch,
                        tokens,
                        config_.query_heads,
                        config_.head_dim}),
                {0, 2, 1, 3});
            key = mlx::core::transpose(
                mlx::core::reshape(
                    key_projection,
                    Shape{
                        batch,
                        tokens,
                        config_.kv_heads,
                        config_.head_dim}),
                {0, 2, 1, 3});
            if (query_norm_) query = (*query_norm_)(query);
            if (key_norm_) key = (*key_norm_)(key);
            if (positions) {
                query = apply_rope(
                    query, *positions, config_.head_dim, config_.rope_base);
                key = apply_rope(
                    key, *positions, config_.head_dim, config_.rope_base);
            } else {
                query = apply_rope(
                    query, config_.head_dim, config_.rope_base, position);
                key = apply_rope(
                    key, config_.head_dim, config_.rope_base, position);
            }
        }
        auto value = use_combined_kv_write
            ? value_projection
            : mlx::core::transpose(
                  mlx::core::reshape(
                      value_projection,
                      Shape{
                          batch,
                          tokens,
                          config_.kv_heads,
                          config_.head_dim,
                      }),
                  {0, 2, 1, 3});
        if (detail::component_profile_active()) {
            if (use_inline_kv_post) {
                detail::profile_eval(
                    "minicpmo.qkv_norm_rope", query);
            } else {
                detail::profile_eval(
                    "minicpmo.qkv_norm_rope",
                    std::vector<array>{query, key, value});
            }
        }
        const int attention_length = cache_
            ? cache_->position() + (use_inline_kv_post ? 0 : 1)
            : 0;
        const bool use_fused_gqa =
            use_cache && tokens == 1 && cache_ &&
            batch == 1 && !mask &&
            config_.query_heads == 32 &&
            config_.kv_heads == 8 &&
            config_.head_dim == 128 &&
            query.dtype() == mlx::core::float16 &&
            use_mini_gqa_attention(attention_length);
        array keys = key;
        array values = value;
        if (use_cache) {
            if (use_combined_kv_write) {
                if (!use_inline_kv_post) {
                    const int append_position = cache_->position();
                    cache_->reserve_append(1);
                    query = mini_kv_cache_write(
                        query,
                        key,
                        value,
                        cache_->key_storage(),
                        cache_->value_storage(),
                        append_position,
                        cache_->capacity());
                }
                // The fused GQA path reads persistent storage directly. Do
                // not create two dead slice nodes for every decoder layer.
                if (!use_fused_gqa) {
                    auto cached = cache_->view();
                    keys = std::move(cached.first);
                    values = std::move(cached.second);
                }
            } else {
                auto cached = cache_->append(key, value);
                keys = std::move(cached.first);
                values = std::move(cached.second);
            }
            if (detail::component_profile_active() &&
                !use_combined_kv_write) {
                detail::profile_eval(
                    "minicpmo.kv_append",
                    std::vector<array>{
                        cache_->key_storage(),
                        cache_->value_storage(),
                    });
            }
        }
        array attended = use_fused_gqa
            ? mini_gqa_attention(
                  query,
                  cache_->key_storage(),
                  cache_->value_storage(),
                  cache_->position(),
                  cache_->capacity())
            : scaled_dot_product_attention(
                  query,
                  keys,
                  values,
                  !(use_cache && tokens == 1),
                  0.0f,
                  mask);
        attended = mlx::core::reshape(
            mlx::core::transpose(attended, {0, 2, 1, 3}),
            Shape{batch, tokens, config_.query_heads * config_.head_dim});
        detail::profile_eval("minicpmo.attention", attended);
        auto residual = output_.add_to(attended, input);
        detail::profile_eval(
            "minicpmo.attention_output", residual);
        auto output = ffn_.add_to(ffn_norm_(residual), residual);
        detail::profile_eval("minicpmo.ffn", output);
        return output;
    }

private:
    MiniQwen3Config config_;
    MlxRmsNorm attention_norm_;
    MiniLinear query_;
    MiniLinear key_;
    MiniLinear value_;
    MiniLinear output_;
    std::optional<MlxGroupedLinear> qkv_;
    std::optional<MlxRmsNorm> query_norm_;
    std::optional<MlxRmsNorm> key_norm_;
    MlxRmsNorm ffn_norm_;
    MiniQwen3Ffn ffn_;
    std::unique_ptr<MlxKvCache> cache_;
};

class MiniQwen3Language {
public:
    static MiniQwen3Language load(
        const MfqContainer& model,
        MiniQwen3Config config,
        MiniQwen3Names names) {
        std::vector<MiniQwen3Block> blocks;
        blocks.reserve(config.layers);
        for (int index = 0; index < config.layers; ++index) {
            blocks.push_back(
                MiniQwen3Block::load(model, config, names, index));
        }
        std::optional<MlxLinear> output;
        if (!config.tie_embeddings) {
            output = MlxLinear::load(model, names.output);
        }
        const float norm_eps = config.norm_eps;
        return MiniQwen3Language(
            std::move(config),
            MlxEmbedding::load(model, names.embedding),
            std::move(blocks),
            mini_rms_norm(model, names.norm, norm_eps),
            std::move(output));
    }

    MiniQwen3Language(
        MiniQwen3Config config,
        MlxEmbedding embedding,
        std::vector<MiniQwen3Block> blocks,
        MlxRmsNorm output_norm,
        std::optional<MlxLinear> output)
        : config_(std::move(config)),
          embedding_(std::move(embedding)),
          blocks_(std::move(blocks)),
          output_norm_(std::move(output_norm)),
          output_(std::move(output)) {}

    array embed(const array& ids) const {
        if (ids.ndim() != 2 || ids.shape(0) <= 0 || ids.shape(1) <= 0) {
            throw std::runtime_error(
                "MiniCPM-o Qwen3 token ids must be [batch,tokens]");
        }
        return embedding_(
            ids,
            config_.model_type == "minicpmo"
                ? mlx::core::float16
                : mlx::core::bfloat16);
    }

    array hidden_forward(
        const array& ids,
        const array* positions = nullptr,
        const array* attention_mask = nullptr,
        bool use_cache = true) {
        return hidden_forward_inputs(
            embed(ids), positions, attention_mask, use_cache);
    }

    array hidden_forward_inputs(
        const array& input_embeddings,
        const array* positions = nullptr,
        const array* attention_mask = nullptr,
        bool use_cache = true) {
        if (input_embeddings.ndim() != 3 ||
            input_embeddings.shape(2) != config_.hidden) {
            throw std::runtime_error(
                "MiniCPM-o Qwen3 inputs_embeds shape mismatch");
        }
        const int batch = input_embeddings.shape(0);
        const int tokens = input_embeddings.shape(1);
        const int position = use_cache ? cache_position_ : 0;
        if (position > config_.maximum_context - tokens) {
            throw std::runtime_error(
                "MiniCPM-o Qwen3 context capacity exceeded");
        }
        if (positions) {
            const bool shape_ok =
                positions->ndim() == 1 ||
                (positions->ndim() == 2 &&
                 (positions->shape(0) == 1 ||
                  positions->shape(0) == batch));
            if (!shape_ok || positions->shape(-1) != tokens) {
                throw std::runtime_error(
                    "MiniCPM-o Qwen3 position_ids shape mismatch");
            }
        }
        if (use_cache && cache_batch_ == 0) {
            reset_cache(batch, tokens);
        } else if (use_cache && cache_batch_ != batch) {
            throw std::runtime_error(
                "MiniCPM-o Qwen3 cache batch mismatch");
        }
        std::optional<array> mask;
        if (attention_mask) {
            if (attention_mask->ndim() != 2 ||
                attention_mask->shape(0) != batch ||
                attention_mask->shape(1) < position + tokens) {
                throw std::runtime_error(
                    "MiniCPM-o Qwen3 attention mask shape mismatch");
            }
            auto visible = mlx::core::slice(
                *attention_mask,
                Shape{0, 0},
                Shape{batch, position + tokens});
            visible = mlx::core::reshape(
                mlx::core::astype(visible, mlx::core::bool_),
                Shape{batch, 1, 1, position + tokens});
            mask = additive_mask(
                visible,
                config_.model_type == "minicpmo"
                    ? mlx::core::float16
                    : mlx::core::bfloat16);
        }
        const auto activation_dtype = config_.model_type == "minicpmo"
            ? mlx::core::float16
            : mlx::core::bfloat16;
        auto hidden = as_dtype(input_embeddings, activation_dtype);
        std::optional<array> shared_rope_table;
        if (tokens == 1 && positions == nullptr && use_cache &&
            config_.model_type == "minicpmo") {
            shared_rope_table = mini_rope_table(
                static_cast<float>(config_.rope_base), position);
        }
        for (auto& block : blocks_) {
            hidden = block.forward(
                hidden,
                positions,
                position,
                use_cache,
                mask,
                shared_rope_table ? &*shared_rope_table : nullptr);
        }
        if (use_cache) cache_position_ += tokens;
        return output_norm_(hidden);
    }

    array logits(const array& hidden) const {
        auto result = config_.tie_embeddings
            ? embedding_.project(hidden)
            : (*output_)(hidden);
        return as_dtype(
            result,
            config_.model_type == "minicpmo"
                ? mlx::core::float16
                : mlx::core::bfloat16);
    }

    bool supports_fused_greedy() const noexcept {
        if (config_.tie_embeddings || !output_) return false;
        const auto* weight = output_->nint_weight_ref();
        return weight != nullptr && weight->bits() == 6 &&
            weight->group_size() == 24 &&
            !weight->q5_execution_layout();
    }

    array forward_greedy(const array& ids, bool use_cache = true) {
        auto hidden = hidden_forward(ids, nullptr, nullptr, use_cache);
        if (hidden.ndim() != 3 || hidden.shape(0) != 1 ||
            hidden.shape(1) <= 0 || hidden.shape(2) != config_.hidden) {
            throw std::runtime_error(
                "MiniCPM-o fused greedy hidden shape is invalid");
        }
        hidden = mlx::core::reshape(
            mlx::core::slice(
                hidden,
                Shape{0, hidden.shape(1) - 1, 0},
                Shape{1, hidden.shape(1), config_.hidden}),
            Shape{1, config_.hidden});
        auto token = output_->greedy_argmax(hidden);
        if (!token) {
            throw std::runtime_error(
                "MiniCPM-o fused greedy layout became unavailable");
        }
        return std::move(*token);
    }

    array forward(const array& ids, bool use_cache = true) {
        return logits(hidden_forward(ids, nullptr, nullptr, use_cache));
    }

    void reset_cache(int batch, int prompt_tokens = 16) {
        int capacity = 16;
        while (capacity < prompt_tokens &&
               capacity < config_.maximum_context) {
            capacity = std::min(capacity * 2, config_.maximum_context);
        }
        for (auto& block : blocks_) {
            block.reset_cache(batch, capacity);
        }
        cache_position_ = 0;
        cache_batch_ = batch;
        stable_cache_tokens_.clear();
    }

    void materialize_cache() {
        for (auto& block : blocks_) block.materialize_cache();
    }

    void clear_cache() noexcept {
        for (auto& block : blocks_) block.clear_cache();
        cache_position_ = 0;
        cache_batch_ = 0;
        stable_cache_tokens_.clear();
    }

    MlxMiniCPMO45TextSessionState capture_text_session_state(
        const std::vector<std::int64_t>& tokens) const {
        if (cache_batch_ != 1 || cache_position_ <= 0 ||
            static_cast<std::size_t>(cache_position_) != tokens.size() ||
            blocks_.empty()) {
            throw std::runtime_error(
                "MiniCPM-o text session token count does not match cache");
        }
        MlxMiniCPMO45TextSessionState state;
        state.tokens = tokens;
        state.cache_position = cache_position_;
        state.cache_batch = cache_batch_;
        state.layers.reserve(blocks_.size());
        for (const auto& block : blocks_) {
            auto snapshot = block.snapshot_cache();
            state.bytes += snapshot.nbytes();
            state.layers.push_back(std::move(snapshot));
        }
        return state;
    }

    void restore_text_session_state(
        const MlxMiniCPMO45TextSessionState& state) {
        if (state.cache_batch != 1 || state.cache_position <= 0 ||
            static_cast<std::size_t>(state.cache_position) !=
                state.tokens.size() ||
            state.layers.size() != blocks_.size()) {
            throw std::runtime_error(
                "MiniCPM-o text session state is incompatible");
        }
        try {
            for (std::size_t index = 0; index < blocks_.size(); ++index) {
                blocks_[index].restore_cache(state.layers[index]);
            }
            cache_position_ = state.cache_position;
            cache_batch_ = state.cache_batch;
            stable_cache_tokens_ = state.tokens;
        } catch (...) {
            clear_cache();
            throw;
        }
    }

    const MiniQwen3Config& config() const noexcept { return config_; }
    std::size_t layer_count() const noexcept { return blocks_.size(); }
    int cache_position() const noexcept { return cache_position_; }
    int cache_batch() const noexcept { return cache_batch_; }
    const std::vector<std::int64_t>& stable_cache_tokens() const noexcept {
        return stable_cache_tokens_;
    }

private:
    MiniQwen3Config config_;
    MlxEmbedding embedding_;
    std::vector<MiniQwen3Block> blocks_;
    MlxRmsNorm output_norm_;
    std::optional<MlxLinear> output_;
    int cache_position_ = 0;
    int cache_batch_ = 0;
    std::vector<std::int64_t> stable_cache_tokens_;
};

std::vector<std::int64_t> host_i64(
    const array& value,
    const char* label) {
    auto source = mlx::core::contiguous(
        mlx::core::astype(value, mlx::core::int64));
    source.eval();
    const auto* data = source.data<std::int64_t>();
    if (!data && source.size() != 0) {
        throw std::runtime_error(
            std::string("MiniCPM-o cannot materialize ") + label);
    }
    return {data, data + source.size()};
}

std::vector<std::uint8_t> host_bool(
    const array& value,
    const char* label) {
    auto source = mlx::core::contiguous(
        mlx::core::astype(value, mlx::core::bool_));
    source.eval();
    const auto* data = source.data<bool>();
    if (!data && source.size() != 0) {
        throw std::runtime_error(
            std::string("MiniCPM-o cannot materialize ") + label);
    }
    std::vector<std::uint8_t> result(source.size());
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = data[index] ? 1 : 0;
    }
    return result;
}

array additive_mask(
    const array& visible,
    mlx::core::Dtype dtype) {
    return mlx::core::where(
        visible,
        mlx::core::zeros(visible.shape(), dtype),
        mlx::core::full(
            visible.shape(),
            -std::numeric_limits<float>::infinity(),
            dtype));
}

class VisionAttention {
public:
    static VisionAttention load(
        const MfqContainer& model,
        const std::string& prefix) {
        const auto* fuse_value = std::getenv(
            "MFQ_METAL_VISION_FUSED_QKV");
        const bool fuse_qkv =
            fuse_value == nullptr || std::strcmp(fuse_value, "0") != 0;
        if (fuse_qkv) {
            auto weight = mlx::core::concatenate(
                {
                    dense(model, prefix + ".q_proj.weight"),
                    dense(model, prefix + ".k_proj.weight"),
                    dense(model, prefix + ".v_proj.weight"),
                },
                0);
            auto bias = mlx::core::concatenate(
                {
                    dense(model, prefix + ".q_proj.bias"),
                    dense(model, prefix + ".k_proj.bias"),
                    dense(model, prefix + ".v_proj.bias"),
                },
                0);
            return VisionAttention(
                std::move(weight),
                std::move(bias),
                MiniLinear::load(model, prefix + ".out_proj"));
        }
        return VisionAttention(
            MiniLinear::load(model, prefix + ".q_proj"),
            MiniLinear::load(model, prefix + ".k_proj"),
            MiniLinear::load(model, prefix + ".v_proj"),
            MiniLinear::load(model, prefix + ".out_proj"));
    }

    VisionAttention(
        MiniLinear query,
        MiniLinear key,
        MiniLinear value,
        MiniLinear output)
        : query_(std::move(query)),
          key_(std::move(key)),
          value_(std::move(value)),
          output_(std::move(output)) {}

    VisionAttention(
        array qkv_weight,
        array qkv_bias,
        MiniLinear output)
        : qkv_weight_(std::move(qkv_weight)),
          qkv_bias_(std::move(qkv_bias)),
          output_(std::move(output)) {
        qkv_weight_->eval();
        qkv_bias_->eval();
    }

    array operator()(
        const array& input,
        const std::optional<array>& mask) const {
        const int batch = input.shape(0);
        const int tokens = input.shape(1);
        const auto project = [batch, tokens](array value) {
            return mlx::core::transpose(
                mlx::core::reshape(
                    value,
                    Shape{batch, tokens, 16, 72}),
                {0, 2, 1, 3});
        };
        array query = input;
        array key = input;
        array value = input;
        if (qkv_weight_) {
            auto pieces = mlx::core::split(
                dense_linear(
                    input,
                    *qkv_weight_,
                    qkv_bias_),
                3,
                -1);
            query = std::move(pieces.at(0));
            key = std::move(pieces.at(1));
            value = std::move(pieces.at(2));
        } else {
            query = (*query_)(input);
            key = (*key_)(input);
            value = (*value_)(input);
        }
        auto attended = scaled_dot_product_attention(
            project(std::move(query)),
            project(std::move(key)),
            project(std::move(value)),
            false,
            1.0f / std::sqrt(72.0f),
            mask);
        attended = mlx::core::reshape(
            mlx::core::transpose(attended, {0, 2, 1, 3}),
            Shape{batch, tokens, 1152});
        return output_(attended);
    }

private:
    std::optional<MiniLinear> query_;
    std::optional<MiniLinear> key_;
    std::optional<MiniLinear> value_;
    std::optional<array> qkv_weight_;
    std::optional<array> qkv_bias_;
    MiniLinear output_;
};

class VisionLayer {
public:
    static VisionLayer load(
        const MfqContainer& model,
        int index) {
        const auto prefix =
            "vpm.encoder.layers." + std::to_string(index);
        return VisionLayer(
            VisionAttention::load(model, prefix + ".self_attn"),
            MiniLayerNorm::load(model, prefix + ".layer_norm1", 1e-6f),
            MiniLayerNorm::load(model, prefix + ".layer_norm2", 1e-6f),
            MiniLinear::load(model, prefix + ".mlp.fc1"),
            MiniLinear::load(model, prefix + ".mlp.fc2"));
    }

    VisionLayer(
        VisionAttention attention,
        MiniLayerNorm norm1,
        MiniLayerNorm norm2,
        MiniLinear fc1,
        MiniLinear fc2)
        : attention_(std::move(attention)),
          norm1_(std::move(norm1)),
          norm2_(std::move(norm2)),
          fc1_(std::move(fc1)),
          fc2_(std::move(fc2)) {}

    array operator()(
        const array& input,
        const std::optional<array>& mask) const {
        auto hidden = input + attention_(norm1_(input), mask);
        return hidden + fc2_(gelu(fc1_(norm2_(hidden)), true));
    }

private:
    VisionAttention attention_;
    MiniLayerNorm norm1_;
    MiniLayerNorm norm2_;
    MiniLinear fc1_;
    MiniLinear fc2_;
};

class VisionEncoder {
public:
    static VisionEncoder load(const MfqContainer& model) {
        std::vector<VisionLayer> layers;
        layers.reserve(27);
        for (int index = 0; index < 27; ++index) {
            layers.push_back(VisionLayer::load(model, index));
        }
        return VisionEncoder(
            dense(model, "vpm.embeddings.patch_embedding.weight"),
            dense(model, "vpm.embeddings.patch_embedding.bias"),
            dense(model, "vpm.embeddings.position_embedding.weight"),
            std::move(layers),
            MiniLayerNorm::load(model, "vpm.post_layernorm", 1e-6f));
    }

    VisionEncoder(
        array patch_weight,
        array patch_bias,
        array position_embedding,
        std::vector<VisionLayer> layers,
        MiniLayerNorm post_norm)
        : patch_weight_(std::move(patch_weight)),
          patch_bias_(std::move(patch_bias)),
          position_embedding_(std::move(position_embedding)),
          layers_(std::move(layers)),
          post_norm_(std::move(post_norm)) {
        if (patch_weight_.shape() != Shape{1152, 3, 14, 14} ||
            patch_bias_.shape() != Shape{1152} ||
            position_embedding_.shape() != Shape{4900, 1152}) {
            throw std::runtime_error(
                "MiniCPM-o SigLIP tensor shapes disagree with version 4.5");
        }
    }

    array operator()(
        const array& pixels,
        const array& patch_mask_value,
        const array& target_sizes) const {
        if (pixels.ndim() != 4 || pixels.shape(1) != 3 ||
            target_sizes.ndim() != 2 || target_sizes.shape(1) != 2 ||
            target_sizes.shape(0) != pixels.shape(0)) {
            throw std::runtime_error(
                "MiniCPM-o vision input geometry is invalid");
        }
        auto mask = patch_mask_value.ndim() == 3
            ? mlx::core::reshape(
                  patch_mask_value,
                  Shape{patch_mask_value.shape(0),
                        patch_mask_value.shape(1) * patch_mask_value.shape(2)})
            : patch_mask_value;
        if (mask.ndim() != 2 || mask.shape(0) != pixels.shape(0)) {
            throw std::runtime_error(
                "MiniCPM-o patch mask geometry is invalid");
        }

        auto image = mlx::core::transpose(pixels, {0, 2, 3, 1});
        auto weight = mlx::core::transpose(
            patch_weight_, {0, 2, 3, 1});
        auto embedded = mlx::core::conv2d(
            as_dtype(image, weight.dtype()),
            weight,
            {14, 14});
        embedded = embedded + patch_bias_;
        embedded = mlx::core::reshape(
            embedded,
            Shape{embedded.shape(0),
                  embedded.shape(1) * embedded.shape(2),
                  embedded.shape(3)});
        if (embedded.shape(1) != mask.shape(1)) {
            throw std::runtime_error(
                "MiniCPM-o patch mask length does not match patch convolution");
        }

        const auto sizes = host_i64(target_sizes, "target sizes");
        const auto active_mask = host_bool(mask, "patch mask");
        std::vector<std::int32_t> position_ids(embedded.shape(0) * embedded.shape(1));
        bool all_active = true;
        bool uniform_geometry = embedded.shape(0) > 0;
        const auto first_height = embedded.shape(0) > 0 ? sizes.at(0) : 0;
        const auto first_width = embedded.shape(0) > 0 ? sizes.at(1) : 0;
        for (int batch = 0; batch < embedded.shape(0); ++batch) {
            const auto height = sizes.at(2 * batch);
            const auto width = sizes.at(2 * batch + 1);
            if (height <= 0 || width <= 0 ||
                height * width > embedded.shape(1)) {
                throw std::runtime_error(
                    "MiniCPM-o target patch size is invalid");
            }
            uniform_geometry = uniform_geometry &&
                height == first_height && width == first_width;
            std::int64_t active = 0;
            for (int patch = 0; patch < embedded.shape(1); ++patch) {
                if (!active_mask.at(batch * embedded.shape(1) + patch)) {
                    all_active = false;
                    continue;
                }
                if (active >= height * width) {
                    throw std::runtime_error(
                        "MiniCPM-o patch mask has too many active entries");
                }
                const auto row = active / width;
                const auto column = active % width;
                const auto bucket_row = std::min<std::int64_t>(69, row * 70 / height);
                const auto bucket_column = std::min<std::int64_t>(69, column * 70 / width);
                position_ids[batch * embedded.shape(1) + patch] =
                    static_cast<std::int32_t>(bucket_row * 70 + bucket_column);
                ++active;
            }
            if (active != height * width) {
                throw std::runtime_error(
                    "MiniCPM-o patch mask active count disagrees with target size");
            }
        }
        const auto* shared_position_value = std::getenv(
            "MFQ_METAL_VISION_SHARED_POSITION");
        const bool shared_position =
            shared_position_value == nullptr ||
            std::strcmp(shared_position_value, "0") != 0;
        array position = [&] {
            if (shared_position && uniform_geometry && all_active) {
                const array ids(
                    position_ids.begin(),
                    Shape{embedded.shape(1)},
                    mlx::core::int32);
                return mlx::core::expand_dims(
                    mlx::core::take(position_embedding_, ids, 0),
                    0);
            }
            const array ids(
                position_ids.begin(),
                Shape{embedded.shape(0), embedded.shape(1)},
                mlx::core::int32);
            return mlx::core::take(position_embedding_, ids, 0);
        }();
        embedded = embedded + as_dtype(position, embedded.dtype());

        std::optional<array> attention_mask;
        if (!all_active) {
            auto visible = mlx::core::reshape(
                mlx::core::astype(mask, mlx::core::bool_),
                Shape{embedded.shape(0), 1, 1, embedded.shape(1)});
            attention_mask = additive_mask(visible, embedded.dtype());
        }
        const auto encode = [&](array hidden,
                                const std::optional<array>& local_mask) {
            for (const auto& layer : layers_) {
                hidden = layer(hidden, local_mask);
            }
            return post_norm_(hidden);
        };
        const int chunk_size = minicpmo_vision_batch_size();
        if (chunk_size <= 0 || embedded.shape(0) <= chunk_size ||
            !minicpmo_vision_batchable_length(embedded.shape(1)) ||
            !uniform_geometry || !all_active) {
            return encode(std::move(embedded), attention_mask);
        }

        std::vector<array> chunks;
        chunks.reserve(
            (embedded.shape(0) + chunk_size - 1) / chunk_size);
        for (int begin = 0; begin < embedded.shape(0);) {
            const int remaining = embedded.shape(0) - begin;
            const int current_size = remaining == chunk_size + 1
                ? remaining
                : std::min(chunk_size, remaining);
            const int end = begin + current_size;
            auto chunk = mlx::core::slice(
                embedded,
                Shape{begin, 0, 0},
                Shape{end, embedded.shape(1), embedded.shape(2)});
            std::optional<array> chunk_mask;
            if (attention_mask) {
                chunk_mask = mlx::core::slice(
                    *attention_mask,
                    Shape{begin, 0, 0, 0},
                    Shape{end, 1, 1, attention_mask->shape(3)});
            }
            chunk = encode(std::move(chunk), chunk_mask);
            chunk.eval();
            chunks.push_back(std::move(chunk));
            begin = end;
        }
        return mlx::core::concatenate(chunks, 0);
    }

private:
    array patch_weight_;
    array patch_bias_;
    array position_embedding_;
    std::vector<VisionLayer> layers_;
    MiniLayerNorm post_norm_;
};

array load_resampler_position(const MfqContainer& model) {
    if (!model.contains(kResamplerPositionAsset)) {
        throw std::runtime_error(
            "MiniCPM-o MFQ is missing the exact Resampler position asset");
    }
    const auto blob = model.read(kResamplerPositionAsset);
    constexpr std::size_t header = 20;
    constexpr std::size_t values =
        std::size_t{70} * 70 * 4096 * sizeof(std::uint16_t);
    if (blob.size() != header + values ||
        std::memcmp(blob.data(), "MFQRSPB1", 8) != 0) {
        throw std::runtime_error(
            "MiniCPM-o Resampler position asset has an invalid format");
    }
    std::uint32_t dimensions[3]{};
    std::memcpy(dimensions, blob.data() + 8, sizeof(dimensions));
    if (dimensions[0] != 70 || dimensions[1] != 70 ||
        dimensions[2] != 4096) {
        throw std::runtime_error(
            "MiniCPM-o Resampler position asset shape is not 70x70x4096");
    }
    auto result = array(
        mlx::core::allocator::malloc(values),
        Shape{70, 70, 4096},
        mlx::core::bfloat16);
    std::memcpy(result.data<std::uint8_t>(), blob.data() + header, values);
    return result;
}

class Resampler {
public:
    static Resampler load(const MfqContainer& model) {
        return Resampler(
            dense(model, "resampler.query"),
            load_resampler_position(model),
            MiniLinear::load(model, "resampler.kv_proj", false),
            MiniLayerNorm::load(model, "resampler.ln_q", 1e-6f),
            MiniLayerNorm::load(model, "resampler.ln_kv", 1e-6f),
            MiniLayerNorm::load(model, "resampler.ln_post", 1e-6f),
            dense(model, "resampler.attn.in_proj_weight"),
            dense(model, "resampler.attn.in_proj_bias"),
            dense(model, "resampler.attn.out_proj.weight"),
            dense(model, "resampler.attn.out_proj.bias"),
            dense(model, "resampler.proj"));
    }

    Resampler(
        array query,
        array position_embedding,
        MiniLinear kv_projection,
        MiniLayerNorm query_norm,
        MiniLayerNorm kv_norm,
        MiniLayerNorm post_norm,
        array in_weight,
        array in_bias,
        array out_weight,
        array out_bias,
        array final_projection)
        : query_(std::move(query)),
          position_embedding_(std::move(position_embedding)),
          kv_projection_(std::move(kv_projection)),
          query_norm_(std::move(query_norm)),
          kv_norm_(std::move(kv_norm)),
          post_norm_(std::move(post_norm)),
          in_weight_(std::move(in_weight)),
          in_bias_(std::move(in_bias)),
          out_weight_(std::move(out_weight)),
          out_bias_(std::move(out_bias)),
          final_projection_(std::move(final_projection)) {
        if (query_.shape() != Shape{64, 4096} ||
            position_embedding_.shape() != Shape{70, 70, 4096} ||
            in_weight_.shape() != Shape{12288, 4096} ||
            in_bias_.shape() != Shape{12288} ||
            out_weight_.shape() != Shape{4096, 4096} ||
            final_projection_.shape() != Shape{4096, 4096}) {
            throw std::runtime_error(
                "MiniCPM-o Resampler tensor shapes disagree with version 4.5");
        }

        // The learned queries and their Q projection are model constants.
        // Projecting a [batch, 64, 4096] broadcast repeats the same large
        // GEMM once per image.  Materialize the single [64, 4096] result at
        // load time and only broadcast the projected queries per request.
        // This is algebraically identical and matters especially for video
        // batches containing tens or hundreds of frames.
        projected_query_ = dense_linear(
            query_norm_(query_),
            mlx::core::slice(
                in_weight_,
                Shape{0, 0},
                Shape{4096, 4096}),
            mlx::core::slice(
                in_bias_,
                Shape{0},
                Shape{4096}));
        projected_query_.eval();
    }

    array operator()(
        const array& input,
        const array& target_sizes) const {
        if (input.ndim() != 3 || input.shape(2) != 1152 ||
            target_sizes.ndim() != 2 || target_sizes.shape(1) != 2 ||
            target_sizes.shape(0) != input.shape(0)) {
            throw std::runtime_error(
                "MiniCPM-o Resampler input geometry is invalid");
        }
        const int chunk_size = minicpmo_vision_batch_size();
        bool uniform_full_geometry = input.shape(0) > 0;
        if (uniform_full_geometry) {
            const auto sizes = host_i64(
                target_sizes, "Resampler target sizes");
            const auto first_height = sizes.at(0);
            const auto first_width = sizes.at(1);
            for (int index = 0; index < input.shape(0); ++index) {
                const auto height = sizes.at(index * 2);
                const auto width = sizes.at(index * 2 + 1);
                uniform_full_geometry = uniform_full_geometry &&
                    height == first_height && width == first_width &&
                    height * width == input.shape(1);
            }
        }
        if (chunk_size <= 0 || input.shape(0) <= chunk_size ||
            !minicpmo_vision_batchable_length(input.shape(1)) ||
            !uniform_full_geometry) {
            return forward_batch(input, target_sizes);
        }

        std::vector<array> chunks;
        chunks.reserve((input.shape(0) + chunk_size - 1) / chunk_size);
        for (int begin = 0; begin < input.shape(0);) {
            const int remaining = input.shape(0) - begin;
            const int current_size = remaining == chunk_size + 1
                ? remaining
                : std::min(chunk_size, remaining);
            const int end = begin + current_size;
            auto chunk = forward_batch(
                mlx::core::slice(
                    input,
                    Shape{begin, 0, 0},
                    Shape{end, input.shape(1), input.shape(2)}),
                mlx::core::slice(
                    target_sizes,
                    Shape{begin, 0},
                    Shape{end, target_sizes.shape(1)}));
            chunk.eval();
            chunks.push_back(std::move(chunk));
            begin = end;
        }
        return mlx::core::concatenate(chunks, 0);
    }

    array forward_batch(
        const array& input,
        const array& target_sizes) const {
        if (input.ndim() != 3 || input.shape(2) != 1152 ||
            target_sizes.ndim() != 2 || target_sizes.shape(1) != 2 ||
            target_sizes.shape(0) != input.shape(0)) {
            throw std::runtime_error(
                "MiniCPM-o Resampler input geometry is invalid");
        }
        const int batch = input.shape(0);
        const int length = input.shape(1);
        const auto sizes = host_i64(target_sizes, "Resampler target sizes");
        bool uniform_geometry = batch > 0;
        bool all_visible = true;
        const auto first_height = batch > 0 ? sizes.at(0) : 0;
        const auto first_width = batch > 0 ? sizes.at(1) : 0;
        std::vector<std::uint8_t> visible_values(batch * length, 0);
        for (int index = 0; index < batch; ++index) {
            const auto height = sizes.at(index * 2);
            const auto width = sizes.at(index * 2 + 1);
            const auto patches = height * width;
            if (height <= 0 || width <= 0 || height > 70 || width > 70 ||
                patches > length) {
                throw std::runtime_error(
                    "MiniCPM-o Resampler target size is invalid");
            }
            uniform_geometry = uniform_geometry &&
                height == first_height && width == first_width;
            all_visible = all_visible && patches == length;
            for (int patch = 0; patch < patches; ++patch) {
                visible_values[index * length + patch] = 1;
            }
        }
        const auto make_position = [&](std::int64_t height,
                                       std::int64_t width) {
            const auto patches = height * width;
            auto value = mlx::core::slice(
                position_embedding_,
                Shape{0, 0, 0},
                Shape{
                    static_cast<int>(height),
                    static_cast<int>(width),
                    4096});
            value = mlx::core::reshape(
                value,
                Shape{static_cast<int>(patches), 4096});
            if (patches < length) {
                value = mlx::core::concatenate(
                    {value,
                     mlx::core::zeros(
                         Shape{length - static_cast<int>(patches), 4096},
                         value.dtype())},
                    0);
            }
            return value;
        };
        const auto* shared_position_value = std::getenv(
            "MFQ_METAL_VISION_SHARED_POSITION");
        const bool shared_position =
            shared_position_value == nullptr ||
            std::strcmp(shared_position_value, "0") != 0;
        array position = [&] {
            if (shared_position && uniform_geometry) {
                return mlx::core::expand_dims(
                    make_position(first_height, first_width),
                    0);
            }
            std::vector<array> positions;
            positions.reserve(batch);
            for (int index = 0; index < batch; ++index) {
                positions.push_back(make_position(
                    sizes.at(index * 2),
                    sizes.at(index * 2 + 1)));
            }
            return mlx::core::stack(positions, 0);
        }();
        std::optional<array> mask;
        if (!shared_position || !all_visible) {
            const array visible_raw(
                visible_values.begin(),
                Shape{batch, length},
                mlx::core::uint8);
            mask = additive_mask(
                mlx::core::reshape(
                    mlx::core::astype(visible_raw, mlx::core::bool_),
                    Shape{batch, 1, 1, length}),
                input.dtype());
        }

        const auto projection = [&](
            const array& value,
            int offset) {
            return dense_linear(
                value,
                mlx::core::slice(
                    in_weight_,
                    Shape{offset, 0},
                    Shape{offset + 4096, 4096}),
                mlx::core::slice(
                    in_bias_,
                    Shape{offset},
                    Shape{offset + 4096}));
        };
        auto kv = kv_norm_(kv_projection_(input));
        const auto* projected_query_value = std::getenv(
            "MFQ_METAL_RESAMPLER_PROJECTED_QUERY");
        const bool use_projected_query =
            projected_query_value == nullptr ||
            std::strcmp(projected_query_value, "0") != 0;
        array repeated_query = [&] {
            if (use_projected_query) {
                return mlx::core::contiguous(
                    mlx::core::broadcast_to(
                        mlx::core::expand_dims(projected_query_, 0),
                        Shape{batch, 64, 4096}));
            }
            auto normalized_query = query_norm_(query_);
            return projection(
                mlx::core::broadcast_to(
                    mlx::core::expand_dims(normalized_query, 0),
                    Shape{batch, 64, 4096}),
                0);
        }();
        auto query = mlx::core::transpose(
            mlx::core::reshape(
                repeated_query,
                Shape{batch, 64, 32, 128}),
            {0, 2, 1, 3});
        auto key = mlx::core::transpose(
            mlx::core::reshape(
                projection(kv + as_dtype(position, kv.dtype()), 4096),
                Shape{batch, length, 32, 128}),
            {0, 2, 1, 3});
        auto value = mlx::core::transpose(
            mlx::core::reshape(
                projection(kv, 8192),
                Shape{batch, length, 32, 128}),
            {0, 2, 1, 3});
        auto attended = scaled_dot_product_attention(
            query,
            key,
            value,
            false,
            1.0f / std::sqrt(128.0f),
            mask
                ? std::optional<array>(as_dtype(*mask, query.dtype()))
                : std::nullopt);
        attended = mlx::core::reshape(
            mlx::core::transpose(attended, {0, 2, 1, 3}),
            Shape{batch, 64, 4096});
        attended = dense_linear(
            attended,
            out_weight_,
            out_bias_);
        attended = post_norm_(attended);
        return as_dtype(
            mlx::core::matmul(
                as_dtype(attended, final_projection_.dtype()),
                final_projection_),
            input.dtype());
    }

private:
    array query_;
    array position_embedding_;
    MiniLinear kv_projection_;
    MiniLayerNorm query_norm_;
    MiniLayerNorm kv_norm_;
    MiniLayerNorm post_norm_;
    array in_weight_;
    array in_bias_;
    array out_weight_;
    array out_bias_;
    array final_projection_;
    array projected_query_{0.0f};
};

class WhisperAttention {
public:
    static WhisperAttention load(
        const MfqContainer& model,
        const std::string& prefix) {
        const auto* fuse_value = std::getenv(
            "MFQ_METAL_AUDIO_FUSED_QKV");
        const bool fuse_qkv =
            fuse_value == nullptr || std::strcmp(fuse_value, "0") != 0;
        if (fuse_qkv) {
            auto query_bias = dense(model, prefix + ".q_proj.bias");
            auto value_bias = dense(model, prefix + ".v_proj.bias");
            auto weight = mlx::core::concatenate(
                {
                    dense(model, prefix + ".q_proj.weight"),
                    dense(model, prefix + ".k_proj.weight"),
                    dense(model, prefix + ".v_proj.weight"),
                },
                0);
            auto bias = mlx::core::concatenate(
                {
                    query_bias,
                    mlx::core::zeros(
                        Shape{query_bias.shape(0)}, query_bias.dtype()),
                    value_bias,
                },
                0);
            return WhisperAttention(
                std::move(weight),
                std::move(bias),
                MiniLinear::load(model, prefix + ".out_proj"));
        }
        return WhisperAttention(
            MiniLinear::load(model, prefix + ".q_proj"),
            MiniLinear::load(model, prefix + ".k_proj", false),
            MiniLinear::load(model, prefix + ".v_proj"),
            MiniLinear::load(model, prefix + ".out_proj"));
    }

    WhisperAttention(
        MiniLinear query,
        MiniLinear key,
        MiniLinear value,
        MiniLinear output)
        : query_(std::move(query)),
          key_(std::move(key)),
          value_(std::move(value)),
          output_(std::move(output)) {}

    WhisperAttention(
        array qkv_weight,
        array qkv_bias,
        MiniLinear output)
        : qkv_weight_(std::move(qkv_weight)),
          qkv_bias_(std::move(qkv_bias)),
          output_(std::move(output)) {
        qkv_weight_->eval();
        qkv_bias_->eval();
    }

    array forward(
        const array& input,
        const std::optional<array>& mask,
        std::optional<array>* key_cache,
        std::optional<array>* value_cache) const {
        const int batch = input.shape(0);
        const int tokens = input.shape(1);
        const auto project = [batch, tokens](array value) {
            return mlx::core::transpose(
                mlx::core::reshape(
                    value,
                    Shape{batch, tokens, 16, 64}),
                {0, 2, 1, 3});
        };
        array query = input;
        array key = input;
        array value = input;
        if (qkv_weight_) {
            auto pieces = mlx::core::split(
                dense_linear(input, *qkv_weight_, qkv_bias_),
                3,
                -1);
            query = std::move(pieces.at(0));
            key = std::move(pieces.at(1));
            value = std::move(pieces.at(2));
        } else {
            query = (*query_)(input);
            key = (*key_)(input);
            value = (*value_)(input);
        }
        query = project(std::move(query));
        key = project(std::move(key));
        value = project(std::move(value));
        if (key_cache && value_cache) {
            if (*key_cache) {
                key = mlx::core::concatenate({**key_cache, key}, 2);
                value = mlx::core::concatenate({**value_cache, value}, 2);
            }
            *key_cache = key;
            *value_cache = value;
        }
        auto attended = scaled_dot_product_attention(
            query,
            key,
            value,
            false,
            1.0f / std::sqrt(64.0f),
            mask);
        attended = mlx::core::reshape(
            mlx::core::transpose(attended, {0, 2, 1, 3}),
            Shape{batch, tokens, 1024});
        return output_(attended);
    }

private:
    std::optional<MiniLinear> query_;
    std::optional<MiniLinear> key_;
    std::optional<MiniLinear> value_;
    std::optional<array> qkv_weight_;
    std::optional<array> qkv_bias_;
    MiniLinear output_;
};

class WhisperLayer {
public:
    static WhisperLayer load(
        const MfqContainer& model,
        int index) {
        const auto prefix = "apm.layers." + std::to_string(index);
        return WhisperLayer(
            WhisperAttention::load(model, prefix + ".self_attn"),
            MiniLayerNorm::load(
                model, prefix + ".self_attn_layer_norm", 1e-5f),
            MiniLayerNorm::load(
                model, prefix + ".final_layer_norm", 1e-5f),
            MiniLinear::load(model, prefix + ".fc1"),
            MiniLinear::load(model, prefix + ".fc2"));
    }

    WhisperLayer(
        WhisperAttention attention,
        MiniLayerNorm attention_norm,
        MiniLayerNorm final_norm,
        MiniLinear fc1,
        MiniLinear fc2)
        : attention_(std::move(attention)),
          attention_norm_(std::move(attention_norm)),
          final_norm_(std::move(final_norm)),
          fc1_(std::move(fc1)),
          fc2_(std::move(fc2)) {}

    array forward(
        const array& input,
        const std::optional<array>& mask,
        bool use_cache) {
        auto attended = use_cache
            ? attention_.forward(
                  attention_norm_(input),
                  mask,
                  &key_cache_,
                  &value_cache_)
            : attention_.forward(
                  attention_norm_(input),
                  mask,
                  nullptr,
                  nullptr);
        auto hidden = input + attended;
        return hidden + fc2_(gelu(fc1_(final_norm_(hidden))));
    }

    void reset() {
        key_cache_.reset();
        value_cache_.reset();
    }

    int cache_length() const noexcept {
        return key_cache_ ? key_cache_->shape(2) : 0;
    }

private:
    WhisperAttention attention_;
    MiniLayerNorm attention_norm_;
    MiniLayerNorm final_norm_;
    MiniLinear fc1_;
    MiniLinear fc2_;
    std::optional<array> key_cache_;
    std::optional<array> value_cache_;
};

array average_pool_five(const array& input) {
    if (input.ndim() != 3 || input.shape(1) < 5) {
        throw std::runtime_error(
            "MiniCPM-o audio sequence is too short for stride-5 pooling");
    }
    const int pooled = input.shape(1) / 5;
    auto source = mlx::core::slice(
        input,
        Shape{0, 0, 0},
        Shape{input.shape(0), pooled * 5, input.shape(2)});
    return mlx::core::mean(
        mlx::core::reshape(
            source,
            Shape{input.shape(0), pooled, 5, input.shape(2)}),
        2);
}

class AudioEncoder {
public:
    static AudioEncoder load(const MfqContainer& model) {
        std::vector<WhisperLayer> layers;
        layers.reserve(24);
        for (int index = 0; index < 24; ++index) {
            layers.push_back(WhisperLayer::load(model, index));
        }
        return AudioEncoder(
            dense(model, "apm.conv1.weight"),
            dense(model, "apm.conv1.bias"),
            dense(model, "apm.conv2.weight"),
            dense(model, "apm.conv2.bias"),
            dense(model, "apm.embed_positions.weight"),
            std::move(layers),
            MiniLayerNorm::load(model, "apm.layer_norm", 1e-5f),
            MiniLinear::load(model, "audio_projection_layer.linear1"),
            MiniLinear::load(model, "audio_projection_layer.linear2"));
    }

    AudioEncoder(
        array conv1_weight,
        array conv1_bias,
        array conv2_weight,
        array conv2_bias,
        array position_embedding,
        std::vector<WhisperLayer> layers,
        MiniLayerNorm final_norm,
        MiniLinear projector1,
        MiniLinear projector2)
        : conv1_weight_(std::move(conv1_weight)),
          conv1_bias_(std::move(conv1_bias)),
          conv2_weight_(std::move(conv2_weight)),
          conv2_bias_(std::move(conv2_bias)),
          position_embedding_(std::move(position_embedding)),
          layers_(std::move(layers)),
          final_norm_(std::move(final_norm)),
          projector1_(std::move(projector1)),
          projector2_(std::move(projector2)) {
        if (conv1_weight_.shape() != Shape{1024, 80, 3} ||
            conv1_bias_.shape() != Shape{1024} ||
            conv2_weight_.shape() != Shape{1024, 1024, 3} ||
            conv2_bias_.shape() != Shape{1024} ||
            position_embedding_.shape() != Shape{1500, 1024}) {
            throw std::runtime_error(
                "MiniCPM-o Whisper tensor shapes disagree with version 4.5");
        }
    }

    void reset() {
        for (auto& layer : layers_) {
            layer.reset();
        }
    }

    int cache_length() const noexcept {
        return layers_.empty() ? 0 : layers_.front().cache_length();
    }

    static std::vector<std::int64_t> pooled_lengths(
        const array& raw_lengths) {
        const auto lengths = host_i64(raw_lengths, "audio lengths");
        std::vector<std::int64_t> result(lengths.size());
        for (std::size_t index = 0; index < lengths.size(); ++index) {
            const auto after_conv = (lengths[index] - 1) / 2 + 1;
            result[index] = (after_conv - 5) / 5 + 1;
            if (lengths[index] <= 0 || result[index] <= 0) {
                throw std::runtime_error(
                    "MiniCPM-o audio length is too short");
            }
        }
        return result;
    }

    array forward(
        const array& features,
        const array& raw_lengths,
        bool use_cache = false) {
        if (features.ndim() != 3 || features.shape(1) != 80 ||
            raw_lengths.ndim() != 1 ||
            raw_lengths.shape(0) != features.shape(0)) {
            throw std::runtime_error(
                "MiniCPM-o audio input geometry is invalid");
        }
        auto hidden = convolutions(features);
        const int tokens = hidden.shape(1);
        const int past = use_cache ? cache_length() : 0;
        if (past + tokens > 1500) {
            throw std::runtime_error(
                "MiniCPM-o Whisper position cache exceeds 1500 frames");
        }
        hidden = hidden + as_dtype(
            mlx::core::slice(
                position_embedding_,
                Shape{past, 0},
                Shape{past + tokens, 1024}),
            hidden.dtype());

        const auto lengths = host_i64(raw_lengths, "audio lengths");
        std::vector<std::uint8_t> visible_values(
            features.shape(0) * tokens * (past + tokens), 0);
        for (int batch = 0; batch < features.shape(0); ++batch) {
            for (int query = 0; query < tokens; ++query) {
                const int query_position = past + query;
                const int chunk_limit = (query_position / 50 + 1) * 50;
                for (int key = 0; key < past + tokens; ++key) {
                    const bool visible =
                        key < lengths.at(batch) + past && key < chunk_limit;
                    visible_values[
                        (batch * tokens + query) * (past + tokens) + key] =
                        visible ? 1 : 0;
                }
            }
        }
        const array visible_raw(
            visible_values.begin(),
            Shape{features.shape(0), 1, tokens, past + tokens},
            mlx::core::uint8);
        const auto mask = additive_mask(
            mlx::core::astype(visible_raw, mlx::core::bool_),
            mlx::core::float32);
        for (auto& layer : layers_) {
            hidden = layer.forward(hidden, mask, use_cache);
        }
        hidden = final_norm_(hidden);
        hidden = projector2_(relu(projector1_(hidden)));
        return average_pool_five(hidden);
    }

    array forward_streaming(
        const array& features,
        std::int64_t prefix_extra_frames,
        std::int64_t suffix_extra_frames) {
        if (features.ndim() != 3 || features.shape(0) != 1 ||
            features.shape(1) != 80 || prefix_extra_frames < 0 ||
            suffix_extra_frames < 0) {
            throw std::runtime_error(
                "MiniCPM-o streaming audio expects [1,80,frames] and "
                "non-negative extra-frame counts");
        }
        const int conv_tokens_before_crop =
            (features.shape(2) - 1) / 2 + 1;
        if (cache_length() + conv_tokens_before_crop >= 1500) {
            reset();
        }
        auto hidden = convolutions(features);
        const int prefix_remove = prefix_extra_frames > 0
            ? static_cast<int>((prefix_extra_frames + 1) / 2)
            : 0;
        const int suffix_remove = suffix_extra_frames > 0
            ? static_cast<int>((suffix_extra_frames + 1) / 2)
            : 0;
        if (prefix_remove + suffix_remove >= hidden.shape(1)) {
            throw std::runtime_error(
                "MiniCPM-o streaming audio extra context removes every frame");
        }
        hidden = mlx::core::slice(
            hidden,
            Shape{0, prefix_remove, 0},
            Shape{1, hidden.shape(1) - suffix_remove, 1024});
        const int past = cache_length();
        const int tokens = hidden.shape(1);
        if (past + tokens > 1500) {
            throw std::runtime_error(
                "MiniCPM-o streaming Whisper position cache exceeds 1500 frames");
        }
        hidden = hidden + as_dtype(
            mlx::core::slice(
                position_embedding_,
                Shape{past, 0},
                Shape{past + tokens, 1024}),
            hidden.dtype());
        const auto* omit_mask_value = std::getenv(
            "MFQ_METAL_AUDIO_STREAMING_NO_MASK");
        const bool omit_mask =
            omit_mask_value == nullptr ||
            std::strcmp(omit_mask_value, "0") != 0;
        const std::optional<array> mask = omit_mask
            ? std::nullopt
            : std::optional<array>(mlx::core::zeros(
                  Shape{1, 1, tokens, past + tokens},
                  hidden.dtype()));
        for (auto& layer : layers_) {
            hidden = layer.forward(hidden, mask, true);
        }
        hidden = final_norm_(hidden);
        hidden = projector2_(relu(projector1_(hidden)));
        return average_pool_five(hidden);
    }

private:
    array convolutions(const array& features) const {
        auto input = mlx::core::transpose(features, {0, 2, 1});
        auto weight1 = mlx::core::transpose(conv1_weight_, {0, 2, 1});
        auto hidden = mlx::core::conv1d(
            as_dtype(input, weight1.dtype()),
            weight1,
            1,
            1);
        hidden = gelu(hidden + conv1_bias_);
        auto weight2 = mlx::core::transpose(conv2_weight_, {0, 2, 1});
        hidden = mlx::core::conv1d(hidden, weight2, 2, 1);
        return gelu(hidden + conv2_bias_);
    }

    array conv1_weight_;
    array conv1_bias_;
    array conv2_weight_;
    array conv2_bias_;
    array position_embedding_;
    std::vector<WhisperLayer> layers_;
    MiniLayerNorm final_norm_;
    MiniLinear projector1_;
    MiniLinear projector2_;
};

class TtsDecoder {
public:
    static TtsDecoder load(const MfqContainer& model) {
        auto head_g = dense(
            model,
            "tts.head_code.0.parametrizations.weight.original0");
        auto head_v = dense(
            model,
            "tts.head_code.0.parametrizations.weight.original1");
        if (head_g.shape() != Shape{6562, 1} ||
            head_v.shape() != Shape{6562, 768}) {
            throw std::runtime_error(
                "MiniCPM-o TTS code-head shape mismatch");
        }
        auto value = mlx::core::astype(head_v, mlx::core::float32);
        auto norm = mlx::core::sqrt(
            mlx::core::sum(value * value, -1, true));
        norm = mlx::core::maximum(norm, array(1e-12f));
        auto code_head =
            mlx::core::astype(head_g, mlx::core::float32) * value / norm;
        code_head = as_dtype(code_head, head_v.dtype());
        return TtsDecoder(
            MiniQwen3Language::load(model, tts_config(), tts_names()),
            MlxEmbedding::load(model, "tts.emb_text.weight"),
            MlxEmbedding::load(model, "tts.emb_code.0.weight"),
            MiniLinear::load(model, "tts.projector_semantic.linear1"),
            MiniLinear::load(model, "tts.projector_semantic.linear2"),
            MiniLinear::load(model, "tts.projector_spk.linear1"),
            MiniLinear::load(model, "tts.projector_spk.linear2"),
            std::move(code_head));
    }

    TtsDecoder(
        MiniQwen3Language model,
        MlxEmbedding text_embedding,
        MlxEmbedding code_embedding,
        MiniLinear semantic1,
        MiniLinear semantic2,
        MiniLinear speaker1,
        MiniLinear speaker2,
        array code_head)
        : model_(std::move(model)),
          text_embedding_(std::move(text_embedding)),
          code_embedding_(std::move(code_embedding)),
          semantic1_(std::move(semantic1)),
          semantic2_(std::move(semantic2)),
          speaker1_(std::move(speaker1)),
          speaker2_(std::move(speaker2)),
          code_head_(std::move(code_head)) {
        if (text_embedding_.vocabulary_size() != 152064 ||
            text_embedding_.hidden_size() != 768 ||
            code_embedding_.vocabulary_size() != 6562 ||
            code_embedding_.hidden_size() != 768) {
            throw std::runtime_error(
                "MiniCPM-o TTS embedding shapes disagree with version 4.5");
        }
    }

    void reset(int batch = 1) {
        model_.reset_cache(batch);
    }

    array semantic_projection(const array& hidden) const {
        auto projected = semantic2_(relu(semantic1_(hidden)));
        auto f32 = mlx::core::astype(projected, mlx::core::float32);
        auto norm = mlx::core::sqrt(
            mlx::core::sum(f32 * f32, -1, true));
        norm = mlx::core::maximum(norm, array(1e-12f));
        return projected / as_dtype(norm, projected.dtype());
    }

    array speaker_projection(const array& hidden) const {
        return speaker2_(relu(speaker1_(hidden)));
    }

    array condition(
        const array& text_ids,
        const array& language_hidden) const {
        auto ids = text_ids.ndim() == 1
            ? mlx::core::expand_dims(text_ids, 0)
            : text_ids;
        auto hidden = language_hidden.ndim() == 2
            ? mlx::core::expand_dims(language_hidden, 0)
            : language_hidden;
        if (ids.ndim() != 2 || hidden.ndim() != 3 ||
            ids.shape(0) != 1 || hidden.shape(0) != 1 ||
            ids.shape(1) != hidden.shape(1) || hidden.shape(2) != 4096) {
            throw std::runtime_error(
                "MiniCPM-o TTS condition expects one aligned text span");
        }
        auto text = text_embedding_(ids, mlx::core::bfloat16);
        auto merged = text + as_dtype(semantic_projection(hidden), text.dtype());
        const array suffix_ids(
            {151692, 151687}, Shape{1, 2}, mlx::core::int32);
        auto suffix = text_embedding_(suffix_ids, mlx::core::bfloat16);
        return mlx::core::concatenate({merged, suffix}, 1);
    }

    array duplex_condition(
        const array& text_ids,
        const array& language_hidden,
        std::int64_t audio_bos_token) const {
        if (audio_bos_token < 0 || audio_bos_token >= 152064) {
            throw std::runtime_error(
                "MiniCPM-o TTS audio BOS token is out of range");
        }
        auto ids = text_ids.ndim() == 1
            ? mlx::core::expand_dims(text_ids, 0)
            : text_ids;
        auto hidden = language_hidden.ndim() == 2
            ? mlx::core::expand_dims(language_hidden, 0)
            : language_hidden;
        if (ids.ndim() != 2 || hidden.ndim() != 3 ||
            ids.shape(0) != 1 || hidden.shape(0) != 1 ||
            ids.shape(1) != hidden.shape(1) || hidden.shape(2) != 4096) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS condition geometry is invalid");
        }
        std::optional<array> merged;
        if (ids.shape(1) > 0) {
            auto text = text_embedding_(ids, mlx::core::bfloat16);
            merged = text + as_dtype(
                semantic_projection(hidden), text.dtype());
        }
        const array bos_ids(
            {static_cast<std::int32_t>(audio_bos_token)},
            Shape{1, 1},
            mlx::core::int32);
        auto bos = text_embedding_(bos_ids, mlx::core::bfloat16);
        return merged
            ? mlx::core::concatenate({*merged, bos}, 1)
            : bos;
    }

    array hidden_forward(const array& embeddings) {
        return model_.hidden_forward_inputs(
            embeddings, nullptr, nullptr, true);
    }

    array logits(const array& hidden) const {
        return mlx::core::matmul(
            as_dtype(hidden, code_head_.dtype()),
            mlx::core::transpose(code_head_));
    }

    MlxMiniCPMO45TtsResult generate(
        const array& condition_embeddings,
        std::int32_t steps,
        std::uint64_t seed) {
        if (condition_embeddings.ndim() != 3 ||
            condition_embeddings.shape(0) != 1 ||
            condition_embeddings.shape(2) != 768 || steps <= 0) {
            throw std::runtime_error(
                "MiniCPM-o TTS generation input is invalid");
        }
        reset(1);
        MlxSamplingParams sampling;
        sampling.temperature = 0.8;
        sampling.top_p = 0.85;
        sampling.top_k = 25;
        sampling.seed = seed;
        MlxSampler sampler(sampling);
        std::vector<std::int32_t> generated;
        generated.reserve(static_cast<std::size_t>(steps));
        MlxMiniCPMO45TtsResult result{
            mlx::core::zeros(Shape{1, 0, 1}, mlx::core::int32),
            {},
            false,
        };
        auto current = as_dtype(condition_embeddings, mlx::core::bfloat16);
        constexpr int eos = 6561;
        for (int step = 0; step < steps; ++step) {
            auto hidden = hidden_forward(current);
            auto raw = logits(
                mlx::core::reshape(
                    mlx::core::slice(
                        hidden,
                        Shape{0, hidden.shape(1) - 1, 0},
                        Shape{1, hidden.shape(1), 768}),
                    Shape{1, 768}));
            raw = mlx::core::astype(raw, mlx::core::float32);
            result.logits.push_back(raw);
            auto filtered = raw;
            if (!generated.empty()) {
                std::vector<std::int32_t> counts(6562, 0);
                const auto begin = generated.size() > 16
                    ? generated.size() - 16
                    : 0;
                for (std::size_t index = begin; index < generated.size(); ++index) {
                    ++counts.at(generated[index]);
                }
                const array count_array(
                    counts.begin(), Shape{6562}, mlx::core::int32);
                filtered = sample_apply_penalties(
                    filtered, count_array, 0.0, 0.0, 1.05);
            }
            if (step < 50) {
                filtered = mlx::core::slice_update(
                    filtered,
                    mlx::core::full(
                        Shape{1, 1},
                        -std::numeric_limits<float>::infinity(),
                        mlx::core::float32),
                    Shape{0, eos},
                    Shape{1, eos + 1});
            }
            auto token_array = sampler.sample(filtered);
            token_array.eval();
            const auto token = token_array.data<std::int32_t>()[0];
            generated.push_back(token);
            if (token == eos) {
                result.finished = true;
                break;
            }
            const array token_ids(
                {token}, Shape{1, 1}, mlx::core::int32);
            current = code_embedding_(token_ids, mlx::core::bfloat16);
        }
        const auto returned = generated.empty() ? 0 : generated.size() - 1;
        if (returned > 0) {
            std::vector<std::int32_t> values(
                generated.begin(), generated.begin() + returned);
            result.codes = mlx::core::reshape(
                array(
                    values.begin(),
                    Shape{1, static_cast<int>(returned)},
                    mlx::core::int32),
                Shape{1, static_cast<int>(returned), 1});
        }
        return result;
    }

    MlxMiniCPMO45TtsResult generate_duplex_chunk(
        const array& condition_embeddings,
        std::int32_t max_new_tokens,
        std::int32_t minimum_new_tokens,
        std::int32_t eos_token,
        double temperature,
        double repetition_penalty,
        std::uint64_t seed) {
        if (condition_embeddings.ndim() != 3 ||
            condition_embeddings.shape(0) != 1 ||
            condition_embeddings.shape(2) != 768 ||
            max_new_tokens <= 0 || minimum_new_tokens < 0 ||
            eos_token < 0 || eos_token >= 6562 ||
            !std::isfinite(temperature) || temperature <= 0.0 ||
            !std::isfinite(repetition_penalty) ||
            repetition_penalty <= 0.0) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS generation limits are invalid");
        }
        MlxSamplingParams sampling;
        sampling.temperature = temperature;
        sampling.seed = seed;
        MlxSampler sampler(sampling);
        std::vector<std::int32_t> generated;
        generated.reserve(static_cast<std::size_t>(max_new_tokens));
        MlxMiniCPMO45TtsResult result{
            mlx::core::zeros(Shape{1, 0, 1}, mlx::core::int32),
            {},
            false,
        };
        auto current = as_dtype(condition_embeddings, mlx::core::bfloat16);
        for (int step = 0; step < max_new_tokens; ++step) {
            auto hidden = hidden_forward(current);
            auto raw = logits(mlx::core::reshape(
                mlx::core::slice(
                    hidden,
                    Shape{0, hidden.shape(1) - 1, 0},
                    Shape{1, hidden.shape(1), 768}),
                Shape{1, 768}));
            raw = mlx::core::astype(raw, mlx::core::float32);
            result.logits.push_back(raw);
            auto filtered = raw;
            if (!generated.empty() && repetition_penalty != 1.0) {
                std::vector<std::int32_t> counts(6562, 0);
                const auto begin = generated.size() > 16
                    ? generated.size() - 16
                    : 0;
                for (std::size_t index = begin;
                     index < generated.size(); ++index) {
                    ++counts.at(generated[index]);
                }
                const array count_array(
                    counts.begin(), Shape{6562}, mlx::core::int32);
                filtered = sample_apply_penalties(
                    filtered,
                    count_array,
                    0.0,
                    0.0,
                    repetition_penalty);
            }
            if (step < minimum_new_tokens) {
                filtered = mlx::core::slice_update(
                    filtered,
                    mlx::core::full(
                        Shape{1, 1},
                        -std::numeric_limits<float>::infinity(),
                        mlx::core::float32),
                    Shape{0, eos_token},
                    Shape{1, eos_token + 1});
            }
            auto token_array = sampler.sample(filtered);
            token_array.eval();
            const auto token = token_array.data<std::int32_t>()[0];
            generated.push_back(token);
            if (token == eos_token) {
                result.finished = true;
                break;
            }
            const array token_ids(
                {token}, Shape{1, 1}, mlx::core::int32);
            current = code_embedding_(token_ids, mlx::core::bfloat16);
        }

        // The final sampled token has not been fed through the decoder yet.
        // Returning only tokens represented in the KV cache is what makes the
        // next duplex chunk's expected cache position exact.
        const auto returned = generated.empty() ? 0 : generated.size() - 1;
        if (returned > 0) {
            std::vector<std::int32_t> values(
                generated.begin(), generated.begin() + returned);
            result.codes = mlx::core::reshape(
                array(
                    values.begin(),
                    Shape{1, static_cast<int>(returned)},
                    mlx::core::int32),
                Shape{1, static_cast<int>(returned), 1});
        }
        return result;
    }

    int cache_position() const noexcept {
        return model_.cache_position();
    }

private:
    MiniQwen3Language model_;
    MlxEmbedding text_embedding_;
    MlxEmbedding code_embedding_;
    MiniLinear semantic1_;
    MiniLinear semantic2_;
    MiniLinear speaker1_;
    MiniLinear speaker2_;
    array code_head_;
};

struct Bound {
    int batch = 0;
    int source = 0;
    int begin = 0;
    int end = 0;
};

std::vector<Bound> parse_bounds(
    const std::optional<array>& value,
    const char* label) {
    if (!value || value->size() == 0) return {};
    if (value->ndim() != 2 || value->shape(1) != 4) {
        throw std::runtime_error(
            std::string("MiniCPM-o ") + label +
            " bounds must have [count,4] shape");
    }
    const auto values = host_i64(*value, label);
    std::vector<Bound> result;
    result.reserve(value->shape(0));
    for (int row = 0; row < value->shape(0); ++row) {
        Bound bound{
            static_cast<int>(values.at(row * 4)),
            static_cast<int>(values.at(row * 4 + 1)),
            static_cast<int>(values.at(row * 4 + 2)),
            static_cast<int>(values.at(row * 4 + 3)),
        };
        if (bound.batch < 0 || bound.source < 0 ||
            bound.begin < 0 || bound.end <= bound.begin) {
            throw std::runtime_error(
                std::string("MiniCPM-o ") + label +
                " bounds contain an invalid row");
        }
        result.push_back(bound);
    }
    return result;
}

array last_logits(const array& logits, int vocab) {
    return mlx::core::reshape(
        mlx::core::slice(
            logits,
            Shape{0, logits.shape(1) - 1, 0},
            Shape{1, logits.shape(1), vocab}),
        Shape{1, vocab});
}

} // namespace

namespace detail {

void test_minicpmo45_qk_norm_rope() {
    constexpr int query_heads = 32;
    constexpr int key_heads = 8;
    constexpr int head_dim = 128;
    constexpr float eps = 1e-6f;
    constexpr float base = 1'000'000.0f;
    std::vector<float> query_values(query_heads * head_dim);
    std::vector<float> key_values(key_heads * head_dim);
    std::vector<float> value_values(key_heads * head_dim);
    std::vector<float> norm_values(head_dim);
    for (std::size_t index = 0;
         index < query_values.size();
         ++index) {
        query_values[index] = 0.4f * std::sin(
            static_cast<float>(index + 7) * 0.031f);
    }
    for (std::size_t index = 0;
         index < key_values.size();
         ++index) {
        key_values[index] = 0.35f * std::cos(
            static_cast<float>(index + 11) * 0.043f);
        value_values[index] = 0.3f * std::sin(
            static_cast<float>(index + 17) * 0.037f);
    }
    for (std::size_t index = 0;
         index < norm_values.size();
         ++index) {
        norm_values[index] = 0.8f +
            0.003f * static_cast<float>(index);
    }
    const auto query_projection = mlx::core::astype(
        array(
            query_values.begin(),
            Shape{1, 1, query_heads * head_dim}),
        mlx::core::float16);
    const auto key_projection = mlx::core::astype(
        array(
            key_values.begin(),
            Shape{1, 1, key_heads * head_dim}),
        mlx::core::float16);
    const auto value_projection = mlx::core::astype(
        array(
            value_values.begin(),
            Shape{1, 1, key_heads * head_dim}),
        mlx::core::float16);
    const MlxRmsNorm query_norm(
        array(norm_values.begin(), Shape{head_dim}),
        eps);
    const MlxRmsNorm key_norm(
        array(norm_values.begin(), Shape{head_dim}),
        eps);

    for (const int position : {0, 513, 8'556}) {
        auto fused = mini_qk_norm_rope(
            query_projection,
            key_projection,
            query_norm,
            key_norm,
            query_heads,
            key_heads,
            head_dim,
            base,
            position);
        auto reference_query = apply_rope(
            query_norm(mlx::core::transpose(
                mlx::core::reshape(
                    query_projection,
                    Shape{1, 1, query_heads, head_dim}),
                {0, 2, 1, 3})),
            head_dim,
            base,
            position);
        auto reference_key = apply_rope(
            key_norm(mlx::core::transpose(
                mlx::core::reshape(
                    key_projection,
                    Shape{1, 1, key_heads, head_dim}),
                {0, 2, 1, 3})),
            head_dim,
            base,
            position);
        auto query_difference = mlx::core::max(mlx::core::abs(
            mlx::core::astype(fused.first, mlx::core::float32) -
            mlx::core::astype(reference_query, mlx::core::float32)));
        auto key_difference = mlx::core::max(mlx::core::abs(
            mlx::core::astype(fused.second, mlx::core::float32) -
            mlx::core::astype(reference_key, mlx::core::float32)));
        mlx::core::eval(query_difference, key_difference);
        const float maximum = std::max(
            query_difference.item<float>(),
            key_difference.item<float>());
        if (!std::isfinite(maximum) || maximum > 0.002f) {
            throw std::runtime_error(
                "MiniCPM-o fused Q/K post-processing mismatch at " +
                std::to_string(position) + ": " +
                std::to_string(maximum));
        }
    }

    constexpr int capacity = 9'216;
    constexpr int position = 8'556;
    auto key_cache = mlx::core::zeros(
        Shape{1, key_heads, capacity, head_dim},
        mlx::core::float16);
    auto value_cache = mlx::core::zeros(
        key_cache.shape(), mlx::core::float16);
    mlx::core::eval(key_cache, value_cache);
    auto normalized = mini_qk_norm_rope(
        query_projection,
        key_projection,
        query_norm,
        key_norm,
        query_heads,
        key_heads,
        head_dim,
        base,
        position);
    auto reference_value = mlx::core::transpose(
        mlx::core::reshape(
            value_projection,
            Shape{1, 1, key_heads, head_dim}),
        {0, 2, 1, 3});
    auto written_query = mini_kv_cache_write(
        normalized.first,
        normalized.second,
        reference_value,
        key_cache,
        value_cache,
        position,
        capacity);
    written_query.eval();
    const auto cached_key = mlx::core::slice(
        key_cache,
        Shape{0, 0, position, 0},
        Shape{1, key_heads, position + 1, head_dim});
    const auto cached_value = mlx::core::slice(
        value_cache,
        Shape{0, 0, position, 0},
        Shape{1, key_heads, position + 1, head_dim});
    auto query_difference = mlx::core::max(mlx::core::abs(
        mlx::core::astype(written_query, mlx::core::float32) -
        mlx::core::astype(normalized.first, mlx::core::float32)));
    auto key_difference = mlx::core::max(mlx::core::abs(
        mlx::core::astype(cached_key, mlx::core::float32) -
        mlx::core::astype(normalized.second, mlx::core::float32)));
    auto value_difference = mlx::core::max(mlx::core::abs(
        mlx::core::astype(cached_value, mlx::core::float32) -
        mlx::core::astype(reference_value, mlx::core::float32)));
    mlx::core::eval(
        query_difference, key_difference, value_difference);
    const float maximum = std::max({
        query_difference.item<float>(),
        key_difference.item<float>(),
        value_difference.item<float>(),
    });
    if (!std::isfinite(maximum) || maximum > 0.0f) {
        throw std::runtime_error(
            "MiniCPM-o combined KV cache write mismatch: " +
            std::to_string(maximum));
    }

    auto inline_key_cache = mlx::core::zeros(
        Shape{1, key_heads, capacity, head_dim},
        mlx::core::float16);
    auto inline_value_cache = mlx::core::zeros(
        inline_key_cache.shape(), mlx::core::float16);
    mlx::core::eval(inline_key_cache, inline_value_cache);
    auto rope_table = mini_rope_table(base, position);
    auto inline_query = mini_qk_norm_rope_cache(
        query_projection,
        key_projection,
        value_projection,
        rope_table,
        query_norm,
        key_norm,
        base,
        position,
        inline_key_cache,
        inline_value_cache,
        capacity);
    inline_query.eval();
    const auto inline_key = mlx::core::slice(
        inline_key_cache,
        Shape{0, 0, position, 0},
        Shape{1, key_heads, position + 1, head_dim});
    const auto inline_value = mlx::core::slice(
        inline_value_cache,
        Shape{0, 0, position, 0},
        Shape{1, key_heads, position + 1, head_dim});
    auto inline_query_difference = mlx::core::max(mlx::core::abs(
        mlx::core::astype(inline_query, mlx::core::float32) -
        mlx::core::astype(written_query, mlx::core::float32)));
    auto inline_key_difference = mlx::core::max(mlx::core::abs(
        mlx::core::astype(inline_key, mlx::core::float32) -
        mlx::core::astype(cached_key, mlx::core::float32)));
    auto inline_value_difference = mlx::core::max(mlx::core::abs(
        mlx::core::astype(inline_value, mlx::core::float32) -
        mlx::core::astype(cached_value, mlx::core::float32)));
    mlx::core::eval(
        inline_query_difference,
        inline_key_difference,
        inline_value_difference);
    const float inline_maximum = std::max({
        inline_query_difference.item<float>(),
        inline_key_difference.item<float>(),
        inline_value_difference.item<float>(),
    });
    if (!std::isfinite(inline_maximum) || inline_maximum > 0.0f) {
        throw std::runtime_error(
            "MiniCPM-o inline Q/K cache write changed FP16 values: " +
            std::to_string(inline_maximum));
    }

    // Keep the cache write and its first consumer in one lazy graph.  A
    // value-only check after query.eval() cannot detect a missing Metal
    // buffer hazard because the synchronization has already made the cache
    // writes visible.  At sequence length one, attention must return the
    // value vector repeated for every GQA query head.
    constexpr int hazard_capacity = 16;
    auto combined_key_cache = mlx::core::zeros(
        Shape{1, key_heads, hazard_capacity, head_dim},
        mlx::core::float16);
    auto combined_value_cache = mlx::core::zeros(
        combined_key_cache.shape(), mlx::core::float16);
    mlx::core::eval(combined_key_cache, combined_value_cache);
    auto normalized_zero = mini_qk_norm_rope(
        query_projection,
        key_projection,
        query_norm,
        key_norm,
        query_heads,
        key_heads,
        head_dim,
        base,
        0);
    auto combined_query = mini_kv_cache_write(
        normalized_zero.first,
        normalized_zero.second,
        reference_value,
        combined_key_cache,
        combined_value_cache,
        0,
        hazard_capacity);
    auto combined_attention = mini_gqa_attention(
        combined_query,
        combined_key_cache,
        combined_value_cache,
        1,
        hazard_capacity);
    const auto expected_attention = mlx::core::repeat(
        reference_value, 4, 1);
    auto combined_hazard_difference = mlx::core::max(mlx::core::abs(
        mlx::core::astype(combined_attention, mlx::core::float32) -
        mlx::core::astype(expected_attention, mlx::core::float32)));
    combined_hazard_difference.eval();
    const float combined_hazard_maximum =
        combined_hazard_difference.item<float>();
    if (!std::isfinite(combined_hazard_maximum) ||
        combined_hazard_maximum > 0.0f) {
        throw std::runtime_error(
            "MiniCPM-o combined cache write was not visible to attention: " +
            std::to_string(combined_hazard_maximum));
    }

    auto hazard_key_cache = mlx::core::zeros(
        Shape{1, key_heads, hazard_capacity, head_dim},
        mlx::core::float16);
    auto hazard_value_cache = mlx::core::zeros(
        hazard_key_cache.shape(), mlx::core::float16);
    mlx::core::eval(hazard_key_cache, hazard_value_cache);
    auto hazard_query = mini_qk_norm_rope_cache(
        query_projection,
        key_projection,
        value_projection,
        mini_rope_table(base, 0),
        query_norm,
        key_norm,
        base,
        0,
        hazard_key_cache,
        hazard_value_cache,
        hazard_capacity);
    auto hazard_attention = mini_gqa_attention(
        hazard_query,
        hazard_key_cache,
        hazard_value_cache,
        1,
        hazard_capacity);
    auto hazard_difference = mlx::core::max(mlx::core::abs(
        mlx::core::astype(hazard_attention, mlx::core::float32) -
        mlx::core::astype(expected_attention, mlx::core::float32)));
    hazard_difference.eval();
    const float hazard_maximum = hazard_difference.item<float>();
    if (!std::isfinite(hazard_maximum) || hazard_maximum > 0.0f) {
        throw std::runtime_error(
            "MiniCPM-o inline cache write was not visible to attention: " +
            std::to_string(hazard_maximum));
    }
}

void test_minicpmo45_gqa_attention() {
    constexpr int query_heads = 32;
    constexpr int key_heads = 8;
    constexpr int head_dim = 128;
    constexpr int capacity = 9'216;
    std::vector<float> query_values(query_heads * head_dim);
    std::vector<float> key_values(
        static_cast<std::size_t>(key_heads * capacity * head_dim));
    std::vector<float> value_values(key_values.size());
    for (std::size_t index = 0; index < query_values.size(); ++index) {
        query_values[index] = 0.2f * std::sin(
            static_cast<float>(index + 13) * 0.017f);
    }
    for (std::size_t index = 0; index < key_values.size(); ++index) {
        key_values[index] = 0.18f * std::cos(
            static_cast<float>(index + 19) * 0.0031f);
        value_values[index] = 0.22f * std::sin(
            static_cast<float>(index + 29) * 0.0027f);
    }
    const auto query = mlx::core::astype(
        array(
            query_values.begin(),
            Shape{1, query_heads, 1, head_dim}),
        mlx::core::float16);
    const auto keys = mlx::core::astype(
        array(
            key_values.begin(),
            Shape{1, key_heads, capacity, head_dim}),
        mlx::core::float16);
    const auto values = mlx::core::astype(
        array(
            value_values.begin(),
            Shape{1, key_heads, capacity, head_dim}),
        mlx::core::float16);

    for (const int sequence : {257, 1'025, 9'001}) {
        auto fused = mini_gqa_attention(
            query, keys, values, sequence, capacity);
        const auto key_slice = mlx::core::slice(
            keys,
            Shape{0, 0, 0, 0},
            Shape{1, key_heads, sequence, head_dim});
        const auto value_slice = mlx::core::slice(
            values,
            Shape{0, 0, 0, 0},
            Shape{1, key_heads, sequence, head_dim});
        auto reference = scaled_dot_product_attention(
            query,
            key_slice,
            value_slice,
            false);
        auto difference = mlx::core::max(mlx::core::abs(
            mlx::core::astype(fused, mlx::core::float32) -
            mlx::core::astype(reference, mlx::core::float32)));
        difference.eval();
        const float maximum = difference.item<float>();
        if (!std::isfinite(maximum) || maximum > 0.002f) {
            throw std::runtime_error(
                "MiniCPM-o fused GQA attention mismatch at " +
                std::to_string(sequence) + ": " +
                std::to_string(maximum));
        }
    }
}

void test_minicpmo45_qwen3_cache_equivalence() {
    MiniQwen3Config config;
    config.model_type = "minicpmo-test";
    config.vocab = 32;
    config.hidden = 8;
    config.intermediate = 16;
    config.layers = 1;
    config.query_heads = 2;
    config.kv_heads = 1;
    config.head_dim = 4;
    config.maximum_context = 32;
    config.rope_base = 10000.0f;
    config.norm_eps = 1e-5f;

    const auto linear = [](int output, int input, int phase) {
        std::vector<float> values(
            static_cast<std::size_t>(output * input));
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = 0.075f * std::sin(
                static_cast<float>(index + 1 + phase * 17) * 0.37f);
        }
        auto weight = mlx::core::astype(
            array(values.begin(), Shape{output, input}),
            mlx::core::bfloat16);
        return MiniLinear(MlxLinear(std::move(weight)), std::nullopt);
    };
    const auto norm = [](int width) {
        return MlxRmsNorm(
            mlx::core::ones(Shape{width}, mlx::core::float32),
            1e-5f,
            0.0f);
    };
    const auto make_block = [&]() {
        return MiniQwen3Block(
            config,
            norm(8),
            linear(8, 8, 1),
            linear(4, 8, 2),
            linear(4, 8, 3),
            linear(8, 8, 4),
            norm(4),
            norm(4),
            norm(8),
            MiniQwen3Ffn(
                linear(16, 8, 5),
                linear(16, 8, 6),
                linear(8, 16, 7)));
    };

    constexpr int tokens = 5;
    std::vector<float> input_values(tokens * config.hidden);
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = 0.25f * std::cos(
            static_cast<float>(index + 3) * 0.23f);
    }
    const auto input = mlx::core::astype(
        array(input_values.begin(), Shape{1, tokens, config.hidden}),
        mlx::core::bfloat16);

    auto prefill_block = make_block();
    auto prefill = prefill_block.forward(
        input, nullptr, 0, false, std::nullopt);
    std::vector<std::int32_t> position_values(tokens);
    for (int index = 0; index < tokens; ++index) {
        position_values[index] = index;
    }
    const array positions(
        position_values.begin(), Shape{1, tokens}, mlx::core::int32);
    auto explicit_block = make_block();
    auto explicit_positions = explicit_block.forward(
        input, &positions, 0, false, std::nullopt);

    auto cached_block = make_block();
    cached_block.reset_cache(1, 2);
    std::vector<array> pieces;
    pieces.reserve(tokens);
    for (int token = 0; token < tokens; ++token) {
        auto current = mlx::core::slice(
            input,
            Shape{0, token, 0},
            Shape{1, token + 1, config.hidden});
        pieces.push_back(cached_block.forward(
            current, nullptr, token, true, std::nullopt));
    }
    auto cached = mlx::core::concatenate(pieces, 1);
    auto prefill_f32 = mlx::core::astype(prefill, mlx::core::float32);
    auto explicit_f32 = mlx::core::astype(
        explicit_positions, mlx::core::float32);
    auto cached_f32 = mlx::core::astype(cached, mlx::core::float32);
    mlx::core::eval(prefill_f32, explicit_f32, cached_f32);
    if (prefill.dtype() != mlx::core::bfloat16 ||
        explicit_positions.dtype() != mlx::core::bfloat16 ||
        cached.dtype() != mlx::core::bfloat16 ||
        prefill.shape() != Shape{1, tokens, config.hidden} ||
        cached.shape() != prefill.shape()) {
        throw std::runtime_error(
            "MiniCPM-o Qwen3 BF16 numerical test shape/dtype mismatch");
    }
    const auto* expected = prefill_f32.data<float>();
    const auto* positioned = explicit_f32.data<float>();
    const auto* actual = cached_f32.data<float>();
    float maximum_position_delta = 0.0f;
    float maximum_cache_delta = 0.0f;
    for (std::size_t index = 0; index < prefill.size(); ++index) {
        if (!std::isfinite(expected[index]) ||
            !std::isfinite(positioned[index]) ||
            !std::isfinite(actual[index])) {
            throw std::runtime_error(
                "MiniCPM-o Qwen3 numerical test produced non-finite output");
        }
        maximum_position_delta = std::max(
            maximum_position_delta,
            std::fabs(expected[index] - positioned[index]));
        maximum_cache_delta = std::max(
            maximum_cache_delta,
            std::fabs(expected[index] - actual[index]));
    }
    if (maximum_position_delta > 0.02f || maximum_cache_delta > 0.04f) {
        throw std::runtime_error(
            "MiniCPM-o Qwen3 prefill/cache numerical mismatch: positions=" +
            std::to_string(maximum_position_delta) +
            " cache=" + std::to_string(maximum_cache_delta));
    }

    config.tie_embeddings = true;
    const auto make_language = [&]() {
        std::vector<float> embedding_values(
            static_cast<std::size_t>(config.vocab * config.hidden));
        for (std::size_t index = 0;
             index < embedding_values.size(); ++index) {
            embedding_values[index] = 0.11f * std::sin(
                static_cast<float>(index + 5) * 0.19f);
        }
        std::vector<MiniQwen3Block> blocks;
        blocks.emplace_back(make_block());
        return MiniQwen3Language(
            config,
            MlxEmbedding(mlx::core::astype(
                array(
                    embedding_values.begin(),
                    Shape{config.vocab, config.hidden}),
                mlx::core::bfloat16)),
            std::move(blocks),
            norm(config.hidden),
            std::nullopt);
    };
    const auto ids = [](std::initializer_list<std::int32_t> values) {
        return array(
            values,
            Shape{1, static_cast<int>(values.size())},
            mlx::core::int32);
    };
    const auto evaluated = [](array value) {
        value = mlx::core::astype(value, mlx::core::float32);
        value.eval();
        return std::vector<float>(
            value.data<float>(), value.data<float>() + value.size());
    };

    auto resumed_language = make_language();
    resumed_language.reset_cache(1, 2);
    auto prefix_logits = resumed_language.forward(ids({1, 2}), true);
    prefix_logits.eval();
    const auto snapshot = resumed_language.capture_text_session_state({1, 2});
    if (snapshot.cache_position != 2 || snapshot.cache_batch != 1 ||
        snapshot.layers.size() != 1 || snapshot.bytes == 0) {
        throw std::runtime_error(
            "MiniCPM-o Qwen3 text session snapshot metadata mismatch");
    }
    auto discarded = resumed_language.forward(ids({3, 4}), true);
    discarded.eval();
    resumed_language.restore_text_session_state(snapshot);
    const auto resumed = evaluated(
        resumed_language.forward(ids({5}), true));

    auto fresh_language = make_language();
    fresh_language.reset_cache(1, 3);
    auto fresh_prefix = fresh_language.forward(ids({1, 2}), true);
    fresh_prefix.eval();
    const auto fresh = evaluated(fresh_language.forward(ids({5}), true));
    if (resumed.size() != fresh.size()) {
        throw std::runtime_error(
            "MiniCPM-o Qwen3 restored session shape mismatch");
    }
    for (std::size_t index = 0; index < fresh.size(); ++index) {
        if (!std::isfinite(resumed[index]) ||
            std::fabs(resumed[index] - fresh[index]) > 0.04f) {
            throw std::runtime_error(
                "MiniCPM-o Qwen3 restored session numerical mismatch");
        }
    }

    resumed_language.restore_text_session_state(snapshot);
    const auto repeated = evaluated(
        resumed_language.forward(ids({5}), true));
    for (std::size_t index = 0; index < fresh.size(); ++index) {
        if (!std::isfinite(repeated[index]) ||
            std::fabs(repeated[index] - fresh[index]) > 0.04f) {
            throw std::runtime_error(
                "MiniCPM-o Qwen3 session snapshot was mutated by decode");
        }
    }
}

} // namespace detail

class MlxMiniCPMO45Runtime::Impl {
public:
    struct DuplexState {
        static MlxSamplingParams raw_sampling(
            const MlxMiniCPMO45DuplexConfig& config) {
            MlxSamplingParams result;
            result.temperature = config.greedy ? 0.0 : 1.0;
            result.seed = config.seed;
            return result;
        }

        static MlxSamplingParams filtered_sampling(
            const MlxMiniCPMO45DuplexConfig& config) {
            MlxSamplingParams result;
            result.temperature = config.greedy
                ? 0.0
                : config.temperature;
            result.top_k = config.greedy ? 1 : config.top_k;
            result.top_p = config.top_p;
            result.seed = config.seed ^ 0x9e3779b97f4a7c15ULL;
            return result;
        }

        explicit DuplexState(MlxMiniCPMO45DuplexConfig value)
            : config(std::move(value)),
              raw_sampler(raw_sampling(config)),
              filtered_sampler(filtered_sampling(config)) {
            const auto append_if_missing = [this](std::int64_t token) {
                if (std::find(
                        config.forbidden_ids.begin(),
                        config.forbidden_ids.end(),
                        token) == config.forbidden_ids.end()) {
                    config.forbidden_ids.push_back(token);
                }
            };
            append_if_missing(config.special_ids.chunk_eos);
            append_if_missing(config.special_ids.tts_pad);
        }

        MlxMiniCPMO45DuplexConfig config;
        std::vector<std::int64_t> generated_text_ids;
        std::int64_t audio_chunk_index = 0;
        std::int64_t tts_text_start_position = 0;
        bool current_turn_ended = true;
        std::uint64_t tts_seed_counter = 0;
        MlxSampler raw_sampler;
        MlxSampler filtered_sampler;
    };

    Impl(
        MiniQwen3Language language,
        std::optional<VisionEncoder> vision,
        std::optional<Resampler> resampler,
        std::optional<AudioEncoder> audio,
        std::optional<TtsDecoder> tts)
        : language(std::move(language)),
          vision(std::move(vision)),
          resampler(std::move(resampler)),
          audio(std::move(audio)),
          tts(std::move(tts)) {}

    MiniQwen3Language language;
    std::optional<VisionEncoder> vision;
    std::optional<Resampler> resampler;
    std::optional<AudioEncoder> audio;
    std::optional<TtsDecoder> tts;
    std::optional<DuplexState> duplex;

    std::pair<array, array> feed_duplex_embeddings(array embeddings) {
        if (embeddings.ndim() == 2) {
            embeddings = mlx::core::expand_dims(embeddings, 0);
        }
        if (embeddings.ndim() != 3 || embeddings.shape(0) != 1 ||
            embeddings.shape(2) != language.config().hidden) {
            throw std::runtime_error(
                "MiniCPM-o duplex embeddings must be [1,tokens,4096]");
        }
        auto hidden = language.hidden_forward_inputs(
            as_dtype(embeddings, mlx::core::bfloat16),
            nullptr,
            nullptr,
            true);
        auto logits = last_logits(
            language.logits(hidden), language.config().vocab);
        return {std::move(logits), std::move(hidden)};
    }

    std::pair<array, array> feed_duplex_ids(array token_ids) {
        if (token_ids.ndim() == 0) {
            token_ids = mlx::core::reshape(token_ids, Shape{1, 1});
        } else if (token_ids.ndim() == 1) {
            token_ids = mlx::core::expand_dims(token_ids, 0);
        }
        if (token_ids.ndim() != 2 || token_ids.shape(0) != 1 ||
            token_ids.shape(1) <= 0) {
            throw std::runtime_error(
                "MiniCPM-o duplex token IDs must be [1,tokens]");
        }
        return feed_duplex_embeddings(language.embed(token_ids));
    }

    std::pair<array, array> feed_duplex_id(std::int64_t token) {
        const array token_ids(
            {static_cast<std::int32_t>(token)},
            Shape{1, 1},
            mlx::core::int32);
        return feed_duplex_ids(token_ids);
    }

    std::int64_t sample_duplex_text_token(
        const array& logits,
        bool force_listen) {
        if (!duplex) {
            throw std::runtime_error(
                "MiniCPM-o duplex session is not prepared");
        }
        auto& state = *duplex;
        const auto& config = state.config;
        const auto& ids = config.special_ids;
        if (force_listen) return ids.listen;

        auto first = state.raw_sampler.sample(logits);
        first.eval();
        const auto first_id = first.data<std::int32_t>()[0];
        if (first_id == ids.chunk_eos) return first_id;

        const int vocab = language.config().vocab;
        std::vector<float> factors(static_cast<std::size_t>(vocab), 1.0f);
        if (!state.generated_text_ids.empty() &&
            config.repetition_penalty != 1.0) {
            const auto begin = state.generated_text_ids.size() >
                    static_cast<std::size_t>(config.repetition_window)
                ? state.generated_text_ids.size() -
                    static_cast<std::size_t>(config.repetition_window)
                : 0;
            for (std::size_t index = begin;
                 index < state.generated_text_ids.size(); ++index) {
                const auto token = state.generated_text_ids[index];
                if (token >= 0 && token < vocab) {
                    factors[static_cast<std::size_t>(token)] =
                        static_cast<float>(1.0 / config.repetition_penalty);
                }
            }
        }
        factors.at(static_cast<std::size_t>(ids.listen)) *=
            static_cast<float>(config.listen_probability_scale);
        std::vector<std::uint8_t> allowed(
            static_cast<std::size_t>(vocab), 1);
        for (const auto token : config.forbidden_ids) {
            allowed.at(static_cast<std::size_t>(token)) = 0;
        }
        const array factor_array(
            factors.begin(), Shape{vocab}, mlx::core::float32);
        const array allowed_array(
            allowed.begin(), Shape{vocab}, mlx::core::uint8);
        auto filtered = mlx::core::astype(logits, mlx::core::float32) *
            factor_array;
        if (config.length_penalty != 1.0) {
            const auto adjusted = mlx::core::where(
                mlx::core::greater_equal(
                    filtered,
                    array(0.0f, mlx::core::float32)),
                filtered / static_cast<float>(config.length_penalty),
                filtered * static_cast<float>(config.length_penalty));
            const auto token_ids = mlx::core::arange(
                0, vocab, 1, mlx::core::int32);
            filtered = mlx::core::where(
                mlx::core::equal(
                    token_ids,
                    array(
                        static_cast<std::int32_t>(ids.turn_eos),
                        mlx::core::int32)),
                adjusted,
                filtered);
        }
        filtered = mlx::core::where(
            mlx::core::astype(allowed_array, mlx::core::bool_),
            filtered,
            mlx::core::full_like(
                filtered,
                -std::numeric_limits<float>::infinity()));
        auto sampled = state.filtered_sampler.sample(filtered);
        sampled.eval();
        return sampled.data<std::int32_t>()[0];
    }
};

MlxMiniCPMO45Runtime MlxMiniCPMO45Runtime::load(
    const MfqContainer& model,
    std::int64_t context_size,
    bool load_modalities) {
    configure_minicpmo_sdpa();
    auto config = language_config(model, context_size);
    auto language = MiniQwen3Language::load(
        model, config, language_names());
    std::optional<VisionEncoder> vision;
    std::optional<Resampler> resampler;
    std::optional<AudioEncoder> audio;
    std::optional<TtsDecoder> tts;
    if (load_modalities) {
        vision = VisionEncoder::load(model);
        resampler = Resampler::load(model);
        audio = AudioEncoder::load(model);
        tts = TtsDecoder::load(model);
    }
    return MlxMiniCPMO45Runtime(
        std::make_unique<Impl>(
            std::move(language),
            std::move(vision),
            std::move(resampler),
            std::move(audio),
            std::move(tts)));
}

MlxMiniCPMO45Runtime::MlxMiniCPMO45Runtime(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

MlxMiniCPMO45Runtime::MlxMiniCPMO45Runtime(
    MlxMiniCPMO45Runtime&&) noexcept = default;

MlxMiniCPMO45Runtime& MlxMiniCPMO45Runtime::operator=(
    MlxMiniCPMO45Runtime&&) noexcept = default;

MlxMiniCPMO45Runtime::~MlxMiniCPMO45Runtime() = default;

MlxMiniCPMO45ForwardResult MlxMiniCPMO45Runtime::prepare_inputs(
    const MlxMiniCPMO45Inputs& inputs) {
    auto ids = inputs.input_ids.ndim() == 1
        ? mlx::core::expand_dims(inputs.input_ids, 0)
        : inputs.input_ids;
    if (ids.ndim() != 2) {
        throw std::runtime_error(
            "MiniCPM-o input_ids must have [batch,tokens] shape");
    }
    MlxMiniCPMO45ForwardResult result{
        std::nullopt,
        std::nullopt,
        std::nullopt,
        implementation_->language.embed(ids),
        array(0.0f),
        array(0.0f),
    };
    const auto image_bounds = parse_bounds(inputs.image_bounds, "image");
    if (!image_bounds.empty()) {
        if (!implementation_->vision || !implementation_->resampler ||
            !inputs.pixel_values || !inputs.patch_mask ||
            !inputs.target_sizes) {
            throw std::runtime_error(
                "MiniCPM-o image bounds require loaded image components");
        }
        const bool profile_vision = minicpmo_vision_profile_enabled();
        const auto vision_started = std::chrono::steady_clock::now();
        result.vision_states = (*implementation_->vision)(
            *inputs.pixel_values,
            *inputs.patch_mask,
            *inputs.target_sizes);
        if (profile_vision) {
            result.vision_states->eval();
            std::cerr << "minicpmo_vision_profile stage=vpm ms="
                      << std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - vision_started)
                             .count()
                      << '\n';
        }
        const auto resampler_started = std::chrono::steady_clock::now();
        result.image_embeddings = (*implementation_->resampler)(
            *result.vision_states,
            *inputs.target_sizes);
        if (profile_vision) {
            result.image_embeddings->eval();
            std::cerr << "minicpmo_vision_profile stage=resampler ms="
                      << std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - resampler_started)
                             .count()
                      << '\n';
        }
        for (const auto& bound : image_bounds) {
            if (bound.batch >= ids.shape(0) ||
                bound.source >= result.image_embeddings->shape(0) ||
                bound.end > ids.shape(1) ||
                bound.end - bound.begin != result.image_embeddings->shape(1)) {
                throw std::runtime_error(
                    "MiniCPM-o image bound does not match 64 queries");
            }
            auto update = mlx::core::slice(
                *result.image_embeddings,
                Shape{bound.source, 0, 0},
                Shape{bound.source + 1,
                      result.image_embeddings->shape(1),
                      result.image_embeddings->shape(2)});
            result.input_embeddings = mlx::core::slice_update(
                result.input_embeddings,
                as_dtype(update, result.input_embeddings.dtype()),
                Shape{bound.batch, bound.begin, 0},
                Shape{bound.batch + 1, bound.end, 4096});
        }
    }

    const auto audio_bounds = parse_bounds(inputs.audio_bounds, "audio");
    if (!audio_bounds.empty()) {
        if (!implementation_->audio || !inputs.audio_features ||
            !inputs.audio_lengths) {
            throw std::runtime_error(
                "MiniCPM-o audio bounds require loaded audio components");
        }
        implementation_->audio->reset();
        result.audio_embeddings = implementation_->audio->forward(
            *inputs.audio_features, *inputs.audio_lengths, false);
        log_audio_bitwise_hash(*result.audio_embeddings);
        const auto lengths =
            AudioEncoder::pooled_lengths(*inputs.audio_lengths);
        for (const auto& bound : audio_bounds) {
            const int span = bound.end - bound.begin;
            if (bound.batch >= ids.shape(0) ||
                bound.source >= result.audio_embeddings->shape(0) ||
                bound.source >= static_cast<int>(lengths.size()) ||
                bound.end > ids.shape(1) || span != lengths[bound.source]) {
                throw std::runtime_error(
                    "MiniCPM-o audio bound does not match pooled length");
            }
            auto update = mlx::core::slice(
                *result.audio_embeddings,
                Shape{bound.source, 0, 0},
                Shape{bound.source + 1, span, 4096});
            result.input_embeddings = mlx::core::slice_update(
                result.input_embeddings,
                as_dtype(update, result.input_embeddings.dtype()),
                Shape{bound.batch, bound.begin, 0},
                Shape{bound.batch + 1, bound.end, 4096});
        }
    }

    return result;
}

MlxMiniCPMO45ForwardResult MlxMiniCPMO45Runtime::forward(
    const MlxMiniCPMO45Inputs& inputs) {
    auto result = prepare_inputs(inputs);
    const int batch = result.input_embeddings.shape(0);
    const int tokens = result.input_embeddings.shape(1);
    implementation_->language.reset_cache(batch, tokens);
    result.hidden_states = implementation_->language.hidden_forward_inputs(
        result.input_embeddings,
        inputs.position_ids ? &*inputs.position_ids : nullptr,
        inputs.attention_mask ? &*inputs.attention_mask : nullptr,
        true);
    result.logits = implementation_->language.logits(result.hidden_states);
    return result;
}

array MlxMiniCPMO45Runtime::tts_condition(
    const array& text_ids,
    const array& language_hidden) const {
    if (!implementation_->tts) {
        throw std::runtime_error("MiniCPM-o TTS component is not loaded");
    }
    return implementation_->tts->condition(text_ids, language_hidden);
}

array MlxMiniCPMO45Runtime::tts_duplex_condition(
    const array& text_ids,
    const array& language_hidden,
    std::int64_t audio_bos_token) const {
    if (!implementation_->tts) {
        throw std::runtime_error("MiniCPM-o TTS component is not loaded");
    }
    return implementation_->tts->duplex_condition(
        text_ids, language_hidden, audio_bos_token);
}

MlxMiniCPMO45TtsResult MlxMiniCPMO45Runtime::generate_tts(
    const array& condition_embeddings,
    std::int32_t steps,
    std::uint64_t seed) {
    if (!implementation_->tts) {
        throw std::runtime_error("MiniCPM-o TTS component is not loaded");
    }
    return implementation_->tts->generate(
        condition_embeddings, steps, seed);
}

array MlxMiniCPMO45Runtime::encode_audio_streaming(
    const array& features,
    std::int64_t prefix_extra_frames,
    std::int64_t suffix_extra_frames) {
    if (!implementation_->audio) {
        throw std::runtime_error("MiniCPM-o audio component is not loaded");
    }
    auto embeddings = implementation_->audio->forward_streaming(
        features, prefix_extra_frames, suffix_extra_frames);
    log_audio_bitwise_hash(embeddings);
    return embeddings;
}

void MlxMiniCPMO45Runtime::prepare_duplex(
    const MlxMiniCPMO45DuplexConfig& config,
    const std::optional<array>& system_prefix_ids,
    const std::optional<array>& reference_audio_features,
    const std::optional<array>& system_suffix_ids) {
    if (!implementation_->audio || !implementation_->tts) {
        throw std::runtime_error(
            "MiniCPM-o duplex requires the audio and TTS components");
    }
    const auto& ids = config.special_ids;
    const std::int64_t language_ids[] = {
        ids.unit_start,
        ids.unit_end,
        ids.image_start,
        ids.image_end,
        ids.slice_start,
        ids.slice_end,
        ids.listen,
        ids.speak,
        ids.tts_bos,
        ids.tts_eos,
        ids.chunk_eos,
        ids.chunk_tts_eos,
        ids.turn_eos,
        ids.tts_pad,
    };
    const auto vocab = implementation_->language.config().vocab;
    if (std::any_of(
            std::begin(language_ids),
            std::end(language_ids),
            [vocab](std::int64_t token) {
                return token < 0 || token >= vocab;
            }) ||
        ids.audio_bos < 0 || ids.audio_bos >= 152064) {
        throw std::runtime_error(
            "MiniCPM-o duplex special token IDs are out of range");
    }
    if ((!config.greedy &&
         (!std::isfinite(config.temperature) ||
          config.temperature <= 0.0)) ||
        config.top_k < 0 || config.top_k > std::min(vocab, 1024) ||
        !std::isfinite(config.top_p) || config.top_p <= 0.0 ||
        config.top_p > 1.0 ||
        !std::isfinite(config.listen_probability_scale) ||
        config.listen_probability_scale < 0.0 ||
        !std::isfinite(config.repetition_penalty) ||
        config.repetition_penalty <= 0.0 ||
        config.repetition_window <= 0 ||
        !std::isfinite(config.length_penalty) ||
        config.length_penalty <= 0.0 ||
        !std::isfinite(config.tts_temperature) ||
        config.tts_temperature <= 0.0 ||
        !std::isfinite(config.tts_repetition_penalty) ||
        config.tts_repetition_penalty <= 0.0) {
        throw std::runtime_error(
            "MiniCPM-o duplex sampling configuration is invalid");
    }
    for (const auto token : config.forbidden_ids) {
        if (token < 0 || token >= vocab) {
            throw std::runtime_error(
                "MiniCPM-o duplex forbidden token is out of range");
        }
    }

    int system_tokens = 0;
    const auto validate_system_ids = [&](const std::optional<array>& value,
                                         const char* label) {
        if (!value || value->size() == 0) return;
        system_tokens += static_cast<int>(value->size());
        const auto values = host_i64(*value, label);
        if (std::any_of(
                values.begin(),
                values.end(),
                [vocab](std::int64_t token) {
                    return token < 0 || token >= vocab;
                })) {
            throw std::runtime_error(
                "MiniCPM-o duplex system token is out of range");
        }
    };
    validate_system_ids(system_prefix_ids, "duplex system prefix IDs");
    validate_system_ids(system_suffix_ids, "duplex system suffix IDs");
    if (reference_audio_features) {
        if (reference_audio_features->ndim() != 3 ||
            reference_audio_features->shape(0) != 1 ||
            reference_audio_features->shape(1) != 80 ||
            reference_audio_features->shape(2) < 3) {
            throw std::runtime_error(
                "MiniCPM-o duplex reference audio expects [1,80,frames]");
        }
        const auto raw_frames = reference_audio_features->shape(2);
        const auto conv_frames = (raw_frames - 1) / 2 + 1;
        system_tokens += (conv_frames - 5) / 5 + 1;
    }
    implementation_->language.reset_cache(
        1, std::max(16, system_tokens));
    implementation_->audio->reset();
    implementation_->tts->reset(1);
    implementation_->duplex.emplace(config);
    if (system_prefix_ids && system_prefix_ids->size() > 0) {
        implementation_->feed_duplex_ids(*system_prefix_ids);
    }
    if (reference_audio_features) {
        const array lengths(
            {reference_audio_features->shape(2)},
            Shape{1},
            mlx::core::int64);
        auto embeddings = implementation_->audio->forward(
            *reference_audio_features, lengths, false);
        log_audio_bitwise_hash(embeddings);
        implementation_->feed_duplex_embeddings(embeddings);
        implementation_->audio->reset();
    }
    if (system_suffix_ids && system_suffix_ids->size() > 0) {
        implementation_->feed_duplex_ids(*system_suffix_ids);
    }
}

MlxMiniCPMO45DuplexResult MlxMiniCPMO45Runtime::duplex_step(
    const MlxMiniCPMO45DuplexInputs& inputs) {
    if (!implementation_->duplex || !implementation_->audio ||
        !implementation_->tts) {
        throw std::runtime_error(
            "MiniCPM-o duplex session is not prepared");
    }
    if (inputs.max_new_speak_tokens < 2) {
        throw std::runtime_error(
            "MiniCPM-o duplex generation requires at least two token slots");
    }
    auto& state = *implementation_->duplex;
    const auto& ids = state.config.special_ids;
    auto pending = implementation_->feed_duplex_id(ids.unit_start);
    std::optional<array> generation_logits;
    std::optional<array> audio_embeddings;
    bool has_content = false;

    if (inputs.pixel_values) {
        if (!implementation_->vision || !implementation_->resampler ||
            !inputs.patch_mask || !inputs.target_sizes) {
            throw std::runtime_error(
                "MiniCPM-o duplex image pixels require vision, patch mask, "
                "target sizes and resampler components");
        }
        auto vision_states = (*implementation_->vision)(
            *inputs.pixel_values,
            *inputs.patch_mask,
            *inputs.target_sizes);
        auto image_embeddings = (*implementation_->resampler)(
            vision_states, *inputs.target_sizes);
        std::vector<std::int64_t> counts;
        if (inputs.image_slice_counts) {
            counts = host_i64(
                *inputs.image_slice_counts,
                "duplex image slice counts");
        } else {
            counts.assign(
                static_cast<std::size_t>(image_embeddings.shape(0)), 1);
        }
        int offset = 0;
        for (const auto count : counts) {
            if (count <= 0 || offset + count > image_embeddings.shape(0)) {
                throw std::runtime_error(
                    "MiniCPM-o duplex image slice counts are invalid");
            }
            pending = implementation_->feed_duplex_id(ids.image_start);
            pending = implementation_->feed_duplex_embeddings(
                mlx::core::slice(
                    image_embeddings,
                    Shape{offset, 0, 0},
                    Shape{
                        offset + 1,
                        image_embeddings.shape(1),
                        image_embeddings.shape(2)}));
            pending = implementation_->feed_duplex_id(ids.image_end);
            ++offset;
            for (int slice = 1; slice < count; ++slice) {
                pending = implementation_->feed_duplex_id(ids.slice_start);
                pending = implementation_->feed_duplex_embeddings(
                    mlx::core::slice(
                        image_embeddings,
                        Shape{offset, 0, 0},
                        Shape{
                            offset + 1,
                            image_embeddings.shape(1),
                            image_embeddings.shape(2)}));
                pending = implementation_->feed_duplex_id(ids.slice_end);
                ++offset;
            }
        }
        if (offset != image_embeddings.shape(0)) {
            throw std::runtime_error(
                "MiniCPM-o duplex image slice counts do not cover embeddings");
        }
        generation_logits = pending.first;
        has_content = true;
    }

    if (inputs.audio_features) {
        audio_embeddings = implementation_->audio->forward_streaming(
            *inputs.audio_features,
            inputs.audio_prefix_extra_frames,
            inputs.audio_suffix_extra_frames);
        log_audio_bitwise_hash(*audio_embeddings);
        pending = implementation_->feed_duplex_embeddings(*audio_embeddings);
        generation_logits = pending.first;
        ++state.audio_chunk_index;
        has_content = true;
    }

    if (inputs.text_ids && inputs.text_ids->size() > 0) {
        pending = implementation_->feed_duplex_ids(*inputs.text_ids);
        if (!generation_logits) generation_logits = pending.first;
        has_content = true;
    }
    if (!has_content || !generation_logits) {
        throw std::runtime_error(
            "MiniCPM-o duplex step contains no image, audio, or text input");
    }
    if (inputs.pixel_values && !inputs.audio_features) {
        ++state.audio_chunk_index;
    }

    std::vector<std::int64_t> generated;
    std::vector<std::int64_t> spoken_ids;
    std::vector<array> spoken_hidden;
    auto logits = *generation_logits;
    bool force_current = inputs.force_listen;
    bool is_listen = false;
    bool end_of_turn = false;
    for (int index = 0; index < inputs.max_new_speak_tokens; ++index) {
        if (index == inputs.max_new_speak_tokens - 1) {
            implementation_->feed_duplex_id(ids.chunk_eos);
            generated.push_back(ids.chunk_eos);
            break;
        }
        const bool forced_decision = force_current;
        auto token = implementation_->sample_duplex_text_token(
            logits, force_current);
        force_current = false;
        if (!forced_decision && token != ids.chunk_eos) {
            state.generated_text_ids.push_back(token);
        }
        if (!forced_decision && token == ids.listen &&
            (!state.current_turn_ended || inputs.force_speak)) {
            token = ids.tts_bos;
        }
        generated.push_back(token);
        is_listen = token == ids.listen;
        if (token == ids.listen || token == ids.chunk_eos ||
            token == ids.chunk_tts_eos) {
            pending = implementation_->feed_duplex_id(token);
            break;
        }

        state.current_turn_ended = false;
        pending = implementation_->feed_duplex_id(token);
        logits = pending.first;
        end_of_turn = token == ids.turn_eos;
        if (end_of_turn) state.current_turn_ended = true;
        if (index != 0) {
            spoken_ids.push_back(token);
            spoken_hidden.push_back(mlx::core::slice(
                pending.second,
                Shape{0, pending.second.shape(1) - 1, 0},
                Shape{1, pending.second.shape(1), 4096}));
        }
    }
    implementation_->feed_duplex_id(ids.unit_end);

    const array generated_ids(
        generated.begin(),
        Shape{static_cast<int>(generated.size())},
        mlx::core::int64);
    auto tts_codes = mlx::core::zeros(
        Shape{1, 0, 1}, mlx::core::int32);
    bool tts_force_flush = false;
    if (!is_listen) {
        array text_id_tensor = spoken_ids.empty()
            ? mlx::core::zeros(Shape{1, 0}, mlx::core::int64)
            : array(
                  spoken_ids.begin(),
                  Shape{1, static_cast<int>(spoken_ids.size())},
                  mlx::core::int64);
        array hidden_tensor = spoken_hidden.empty()
            ? mlx::core::zeros(
                  Shape{1, 0, 4096}, mlx::core::bfloat16)
            : mlx::core::concatenate(spoken_hidden, 1);
        auto condition = implementation_->tts->duplex_condition(
            text_id_tensor, hidden_tensor, ids.audio_bos);
        const bool first_tts_chunk = state.tts_text_start_position == 0;
        if (first_tts_chunk) {
            implementation_->tts->reset(1);
            tts_force_flush = true;
        }
        if (implementation_->tts->cache_position() !=
            state.tts_text_start_position) {
            throw std::runtime_error(
                "MiniCPM-o duplex TTS cache position is inconsistent");
        }
        const int minimum_codes = end_of_turn || first_tts_chunk ? 0 : 26;
        auto tts_result = implementation_->tts->generate_duplex_chunk(
            condition,
            26,
            minimum_codes,
            6561,
            state.config.tts_temperature,
            state.config.tts_repetition_penalty,
            state.config.seed + state.tts_seed_counter++);
        tts_codes = std::move(tts_result.codes);
        if (end_of_turn) {
            implementation_->tts->reset(1);
            state.tts_text_start_position = 0;
        } else {
            state.tts_text_start_position +=
                condition.shape(1) + tts_codes.shape(1);
            if (implementation_->tts->cache_position() !=
                state.tts_text_start_position) {
                throw std::runtime_error(
                    "MiniCPM-o duplex TTS cache did not advance exactly");
            }
        }
    }

    return MlxMiniCPMO45DuplexResult{
        *generation_logits,
        std::move(audio_embeddings),
        generated_ids,
        std::move(tts_codes),
        is_listen,
        end_of_turn,
        tts_force_flush,
        state.audio_chunk_index,
        implementation_->language.cache_position(),
        implementation_->audio->cache_length(),
        implementation_->tts->cache_position(),
    };
}

bool MlxMiniCPMO45Runtime::duplex_prepared() const noexcept {
    return implementation_->duplex.has_value();
}

void MlxMiniCPMO45Runtime::reset() {
    implementation_->duplex.reset();
    implementation_->language.clear_cache();
    if (implementation_->audio) implementation_->audio->reset();
    if (implementation_->tts) implementation_->tts->reset();
}

std::int32_t MlxMiniCPMO45Runtime::generate(
    const std::vector<std::int64_t>& prompt,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    const MfqTokenConstraintPtr& token_constraint,
    std::optional<std::size_t> stable_prefix_tokens) {
    const auto& config = implementation_->language.config();
    if (prompt.empty() || max_tokens < 0 ||
        prompt.size() > static_cast<std::size_t>(config.maximum_context)) {
        throw std::invalid_argument(
            "MiniCPM-o generation prompt or token limit is invalid");
    }
    std::vector<std::int32_t> values;
    values.reserve(prompt.size());
    for (const auto token : prompt) {
        if (token < 0 || token >= config.vocab) {
            throw std::invalid_argument(
                "MiniCPM-o prompt token is out of range");
        }
        values.push_back(static_cast<std::int32_t>(token));
    }
    if (max_tokens == 0) {
        implementation_->language.reset_cache(1);
        return 0;
    }
    const std::size_t stable_count = stable_prefix_tokens
        ? std::min(*stable_prefix_tokens, prompt.size())
        : 0;
    std::size_t reused_tokens = 0;
    const auto& stable_tokens =
        implementation_->language.stable_cache_tokens();
    if (stable_count > 0 &&
        implementation_->language.cache_batch() == 1 &&
        implementation_->language.cache_position() ==
            static_cast<int>(stable_tokens.size()) &&
        !stable_tokens.empty() && stable_tokens.size() <= stable_count &&
        stable_tokens.size() < prompt.size() &&
        std::equal(
            stable_tokens.begin(), stable_tokens.end(), prompt.begin())) {
        reused_tokens = stable_tokens.size();
    } else {
        implementation_->language.reset_cache(
            1, static_cast<int>(values.size()));
        implementation_->language.materialize_cache();
    }
    const array prompt_ids(
        values.begin(),
        Shape{1, static_cast<int>(values.size())},
        mlx::core::int32);
    std::optional<array> counts;
    if (sampling.has_penalties()) {
        counts = sample_token_counts_add(
            mlx::core::zeros(Shape{config.vocab}, mlx::core::int32),
            prompt_ids);
    }
    const bool fused_greedy =
        sampling.greedy() && !sampling.has_penalties() &&
        !token_constraint &&
        implementation_->language.supports_fused_greedy();
    double prefill_ms = 0.0;
    std::optional<MlxMiniCPMO45TextSessionState> stable_snapshot;
    struct StableStateRestore {
        MiniQwen3Language& language;
        std::optional<MlxMiniCPMO45TextSessionState>& snapshot;
        ~StableStateRestore() noexcept {
            if (!snapshot) return;
            try {
                language.restore_text_session_state(*snapshot);
            } catch (...) {
                language.clear_cache();
            }
        }
    } stable_restore{implementation_->language, stable_snapshot};

    auto logits = [&]() {
        detail::ScopedMlxEvaluationTiming timing(
            prefill_callback ? &prefill_ms : nullptr);
        const auto forward_range = [&](std::size_t begin, std::size_t end) {
            if (begin >= end || end > prompt.size()) {
                throw std::runtime_error(
                    "MiniCPM-o session prefill range is invalid");
            }
            auto ids = mlx::core::slice(
                prompt_ids,
                Shape{0, static_cast<int>(begin)},
                Shape{1, static_cast<int>(end)});
            if (fused_greedy) {
                return implementation_->language.forward_greedy(ids, true);
            }
            return last_logits(
                implementation_->language.forward(ids, true),
                config.vocab);
        };
        std::optional<array> stable_logits;
        array result = [&]() {
            if (stable_count == 0) {
                return forward_range(0, prompt.size());
            }
            if (reused_tokens < stable_count) {
                stable_logits = forward_range(
                    reused_tokens, stable_count);
            }
            if (implementation_->language.cache_position() !=
                static_cast<int>(stable_count)) {
                throw std::runtime_error(
                    "MiniCPM-o stable session cache position mismatch");
            }
            std::vector<std::int64_t> prefix(
                prompt.begin(),
                prompt.begin() +
                    static_cast<std::ptrdiff_t>(stable_count));
            stable_snapshot = implementation_->language
                .capture_text_session_state(prefix);
            if (stable_count < prompt.size()) {
                return forward_range(stable_count, prompt.size());
            }
            if (!stable_logits) {
                throw std::runtime_error(
                    "MiniCPM-o stable prefix has no sampling logits");
            }
            return std::move(*stable_logits);
        }();
        if (prefill_callback) detail::eval_with_timing(result);
        return result;
    }();
    if (prefill_callback) {
        prefill_callback(prompt.size() - reused_tokens, prefill_ms);
    }

    MlxSampler sampler(sampling);
    const auto generation_limit = std::min<std::int32_t>(
        max_tokens,
        config.maximum_context - static_cast<int>(prompt.size()) + 1);
    std::int32_t generated = 0;
    while (generated < generation_limit) {
        const int decode_step = generated;
        const int profile_skip =
            detail::component_profile_skip_steps();
        const bool profile_this_step =
            detail::component_profile_requested() &&
            decode_step >= profile_skip &&
            decode_step - profile_skip <
                detail::component_profile_steps();
        detail::ComponentProfile component_profile;
        detail::ScopedComponentProfile profile_scope(
            profile_this_step ? &component_profile : nullptr);
        const auto component_started =
            std::chrono::steady_clock::now();
        auto token_array = [&]() {
            if (fused_greedy) return logits;
            return counts
                ? sampler.sample(logits, *counts)
                : sampler.sample(logits);
        }();
        if (profile_this_step) {
            detail::profile_eval("minicpmo.sampling", token_array);
        } else {
            token_array.eval();
        }
        auto token = token_array.data<std::int32_t>()[0];
        if (token_constraint && token_constraint->allows &&
            !token_constraint->allows(token)) {
            auto adjusted = counts
                ? sampler.apply_penalties(logits, *counts)
                : logits;
            adjusted = mlx::core::contiguous(
                mlx::core::astype(adjusted, mlx::core::float32));
            adjusted.eval();
            std::vector<float> masked(
                adjusted.data<float>(),
                adjusted.data<float>() + config.vocab);
            token_constraint->apply(masked.data(), masked.size());
            const array constrained(
                masked.begin(), Shape{1, config.vocab}, mlx::core::float32);
            token_array = sampler.sample(constrained);
            token_array.eval();
            token = token_array.data<std::int32_t>()[0];
            if (!token_constraint->allows(token)) {
                throw std::runtime_error(
                    "MiniCPM-o constrained sampler returned invalid token");
            }
        }
        if (token_constraint && token_constraint->accept) {
            token_constraint->accept(token);
        }
        const array token_ids(
            {token}, Shape{1, 1}, mlx::core::int32);
        if (counts) {
            *counts = sample_token_counts_add(*counts, token_ids);
        }
        ++generated;
        if ((callback && !callback(token)) || generated == generation_limit) {
            report_decode_components(
                profile_this_step,
                decode_step,
                implementation_->language.cache_position(),
                component_started,
                component_profile);
            break;
        }
        logits = fused_greedy
            ? implementation_->language.forward_greedy(token_ids, true)
            : last_logits(
                  implementation_->language.forward(token_ids, true),
                  config.vocab);
        detail::profile_eval("minicpmo.logits", logits);
        report_decode_components(
            profile_this_step,
            decode_step,
            implementation_->language.cache_position(),
            component_started,
            component_profile);
    }
    return generated;
}

std::int32_t MlxMiniCPMO45Runtime::generate_multimodal(
    const MlxMiniCPMO45Inputs& inputs,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(
        std::size_t, double, double, double)>& prefill_callback,
    const MfqTokenConstraintPtr& token_constraint) {
    const auto& config = implementation_->language.config();
    auto ids = inputs.input_ids.ndim() == 1
        ? mlx::core::expand_dims(inputs.input_ids, 0)
        : inputs.input_ids;
    if (ids.ndim() != 2 || ids.shape(0) != 1 || ids.shape(1) <= 0 ||
        ids.shape(1) > config.maximum_context || max_tokens < 0) {
        throw std::invalid_argument(
            "MiniCPM-o multimodal generation input is invalid");
    }
    if (!inputs.pixel_values || !inputs.patch_mask ||
        !inputs.target_sizes || !inputs.image_bounds) {
        throw std::invalid_argument(
            "MiniCPM-o multimodal generation requires image tensors");
    }
    if (max_tokens == 0) {
        implementation_->language.reset_cache(1);
        return 0;
    }

    std::optional<array> counts;
    if (sampling.has_penalties()) {
        counts = sample_token_counts_add(
            mlx::core::zeros(Shape{config.vocab}, mlx::core::int32),
            ids);
    }
    double llm_prefill_ms = 0.0;
    double multimodal_ms = 0.0;
    double model_prefill_ms = 0.0;
    array logits = [&]() {
        const auto model_started = std::chrono::steady_clock::now();
        const auto multimodal_started = model_started;
        auto result = prepare_inputs(inputs);
        if (prefill_callback) {
            result.input_embeddings.eval();
            log_vision_bitwise_hash(result.input_embeddings);
            multimodal_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - multimodal_started)
                .count();
        }

        const auto llm_started = std::chrono::steady_clock::now();
        implementation_->language.reset_cache(
            ids.shape(0), ids.shape(1));
        result.hidden_states =
            implementation_->language.hidden_forward_inputs(
                result.input_embeddings,
                inputs.position_ids ? &*inputs.position_ids : nullptr,
                inputs.attention_mask ? &*inputs.attention_mask : nullptr,
                true);
        result.logits =
            implementation_->language.logits(result.hidden_states);
        auto value = last_logits(result.logits, config.vocab);
        if (prefill_callback) detail::eval_with_timing(value);
        if (prefill_callback) {
            llm_prefill_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - llm_started).count();
            model_prefill_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - model_started).count();
        }
        return value;
    }();
    if (prefill_callback) {
        prefill_callback(
            static_cast<std::size_t>(ids.shape(1)),
            llm_prefill_ms,
            multimodal_ms,
            model_prefill_ms);
    }

    MlxSampler sampler(sampling);
    const bool fused_greedy =
        sampling.greedy() && !sampling.has_penalties() &&
        !token_constraint &&
        implementation_->language.supports_fused_greedy();
    const auto generation_limit = std::min<std::int32_t>(
        max_tokens,
        config.maximum_context - ids.shape(1) + 1);
    std::int32_t generated = 0;
    while (generated < generation_limit) {
        const int decode_step = generated;
        const int profile_skip =
            detail::component_profile_skip_steps();
        const bool profile_this_step =
            detail::component_profile_requested() &&
            decode_step >= profile_skip &&
            decode_step - profile_skip <
                detail::component_profile_steps();
        detail::ComponentProfile component_profile;
        detail::ScopedComponentProfile profile_scope(
            profile_this_step ? &component_profile : nullptr);
        const auto component_started =
            std::chrono::steady_clock::now();
        auto token_array = [&]() {
            if (fused_greedy && generated > 0) return logits;
            return counts
                ? sampler.sample(logits, *counts)
                : sampler.sample(logits);
        }();
        if (profile_this_step) {
            detail::profile_eval("minicpmo.sampling", token_array);
        } else {
            token_array.eval();
        }
        auto token = token_array.data<std::int32_t>()[0];
        if (token_constraint && token_constraint->allows &&
            !token_constraint->allows(token)) {
            auto adjusted = counts
                ? sampler.apply_penalties(logits, *counts)
                : logits;
            adjusted = mlx::core::contiguous(
                mlx::core::astype(adjusted, mlx::core::float32));
            adjusted.eval();
            std::vector<float> masked(
                adjusted.data<float>(),
                adjusted.data<float>() + config.vocab);
            token_constraint->apply(masked.data(), masked.size());
            const array constrained(
                masked.begin(), Shape{1, config.vocab}, mlx::core::float32);
            token_array = sampler.sample(constrained);
            token_array.eval();
            token = token_array.data<std::int32_t>()[0];
            if (!token_constraint->allows(token)) {
                throw std::runtime_error(
                    "MiniCPM-o constrained sampler returned invalid token");
            }
        }
        if (token_constraint && token_constraint->accept) {
            token_constraint->accept(token);
        }
        const auto token_ids = mlx::core::reshape(
            token_array, Shape{1, 1});
        if (counts) {
            *counts = sample_token_counts_add(*counts, token_ids);
        }
        ++generated;
        if ((callback && !callback(token)) || generated == generation_limit) {
            report_decode_components(
                profile_this_step,
                decode_step,
                implementation_->language.cache_position(),
                component_started,
                component_profile);
            break;
        }
        logits = fused_greedy
            ? implementation_->language.forward_greedy(token_ids, true)
            : last_logits(
                  implementation_->language.forward(token_ids, true),
                  config.vocab);
        detail::profile_eval("minicpmo.logits", logits);
        report_decode_components(
            profile_this_step,
            decode_step,
            implementation_->language.cache_position(),
            component_started,
            component_profile);
    }
    return generated;
}

std::size_t MlxMiniCPMO45Runtime::layer_count() const noexcept {
    return implementation_->language.layer_count();
}

std::int64_t MlxMiniCPMO45Runtime::maximum_context() const noexcept {
    return implementation_->language.config().maximum_context;
}

std::int64_t MlxMiniCPMO45Runtime::vocabulary_size() const noexcept {
    return implementation_->language.config().vocab;
}

int MlxMiniCPMO45Runtime::cache_position() const noexcept {
    return implementation_->language.cache_position();
}

MlxMiniCPMO45TextSessionState
MlxMiniCPMO45Runtime::capture_text_session_state(
    const std::vector<std::int64_t>& tokens) const {
    return implementation_->language.capture_text_session_state(tokens);
}

void MlxMiniCPMO45Runtime::restore_text_session_state(
    const MlxMiniCPMO45TextSessionState& state) {
    implementation_->duplex.reset();
    implementation_->language.restore_text_session_state(state);
}

} // namespace mfq::metal
