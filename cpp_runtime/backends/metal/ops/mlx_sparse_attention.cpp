#include "mlx_sparse_attention.h"
#include "mlx_kernel_prepare.h"
#include "mlx_builtin_kernels.h"

#include "mfq_mfe_prefill_embedded.h"
#include "mlx_platform.h"

#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/utils.h>
#include <mlx/backend/metal/kernels/steel/gemm/params.h>
#include <mlx/primitives.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;
using Kernel = mlx::core::fast::CustomKernelFunction;
using TemplateArgs = std::vector<
    std::pair<std::string, mlx::core::fast::TemplateArg>>;

#include "../kernels/mfq_sparse_attention_kernels.inc"
#include "../kernels/mfq_sparse_block_nax.inc"

class QsaPrefillScoresPrimitive final : public mlx::core::Primitive, public MlxPreparableKernel {
public:
    QsaPrefillScoresPrimitive(mlx::core::Stream stream, int offset)
        : Primitive(stream), offset_(offset) {}

    std::string preparation_key() const override { return "mfq_qsa_prefill_scores"; }
    void prepare_gpu() override {
        auto& device = kernel_device(stream().device);
        auto* library = prepared_library();
        (void)device.get_kernel("mfq_qsa_prefill_scores_f16", library);
        (void)device.get_kernel("mfq_qsa_prefill_scores_bf16", library);
    }

    MTL::Library* prepared_library() {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return kernel_device(stream().device).get_library(
            "mfq_qsa_prefill_indexer_v1", options, qsa_prefill_indexer_source);
    }

    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("QSA prefill scores require Metal");
    }

    void eval_gpu(const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        const auto& query = inputs.at(0);
        const auto& key = inputs.at(1);
        auto& output = outputs.front();
        output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto& device = kernel_device(stream().device);
        auto* library = prepared_library();
        const char* name = query.dtype() == mlx::core::float16
            ? "mfq_qsa_prefill_scores_f16" : "mfq_qsa_prefill_scores_bf16";
        const int rows = query.shape(2);
        const int keys = key.shape(1);
        const int tiles_m = (rows + 63) / 64;
        const int tiles_n = (keys + 63) / 64;
        const mlx::steel::GEMMParams params{
            rows, keys, 128, 128, 128, keys, tiles_n, tiles_m,
            std::int64_t(4) * rows * 128, std::int64_t(keys) * 128,
            std::int64_t(rows) * keys, 0, 8, 1};
        const int ratio = 4;
        const float divisor = std::sqrt(128.0f);
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(device.get_kernel(name, library));
        encoder.set_input_array(query, 0);
        encoder.set_input_array(key, 1);
        encoder.set_output_array(output, 2);
        encoder.set_bytes(params, 3);
        encoder.set_bytes(ratio, 4);
        encoder.set_bytes(offset_, 5);
        encoder.set_bytes(divisor, 6);
        encoder.dispatch_threadgroups(MTL::Size(tiles_n, tiles_m, 1),
            MTL::Size(128, 1, 1));
    }

    const char* name() const override { return "QsaPrefillScores"; }

    bool is_equivalent(const mlx::core::Primitive& other) const override {
        const auto* rhs = dynamic_cast<const QsaPrefillScoresPrimitive*>(&other);
        return rhs && rhs->offset_ == offset_;
    }

    std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
        return {Shape{1, inputs.at(0).shape(2), inputs.at(1).shape(1)}};
    }

private:
    int offset_;
};

struct QsaPrefillTopKParams {
    int rows;
    int L;
    int K;
    int topk;
    bool causal_valid_prefix;
};

class QsaPrefillTopKPrimitive final : public mlx::core::Primitive, public MlxPreparableKernel {
public:
    explicit QsaPrefillTopKPrimitive(mlx::core::Stream stream)
        : Primitive(stream) {}

    std::string preparation_key() const override { return "mfq_qsa_prefill_topk512"; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& device = kernel_device(stream().device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library("mfq_qsa_prefill_indexer_v1",
            options, qsa_prefill_indexer_source);
        return device.get_kernel("mfq_qsa_prefill_topk512", library);
    }

    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("QSA prefill top-k requires Metal");
    }

    void eval_gpu(const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        const auto& scores = inputs.front();
        auto& output = outputs.front();
        output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto* kernel = prepared_kernel();
        const int rows = scores.shape(1);
        const QsaPrefillTopKParams params{rows, rows, scores.shape(2), 512, false};
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(kernel);
        encoder.set_input_array(scores, 0);
        encoder.set_output_array(output, 1);
        encoder.set_bytes(params, 2);
        encoder.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(256, 1, 1));
    }

    const char* name() const override { return "QsaPrefillTopK512"; }

    bool is_equivalent(const mlx::core::Primitive& other) const override {
        return dynamic_cast<const QsaPrefillTopKPrimitive*>(&other) != nullptr;
    }

    std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
        return {Shape{1, inputs.front().shape(1), 512}};
    }
};

array qsa_prefill_scores(const array& query, const array& keys, int offset) {
    const int rows = query.shape(1);
    const int count = keys.shape(1);
    if (rows >= 32) {
        auto q = mlx::core::contiguous(mlx::core::transpose(query, {0, 2, 1, 3}));
        return array(Shape{1, rows, count}, mlx::core::float32,
            std::make_shared<QsaPrefillScoresPrimitive>(
                mlx::core::default_stream(mlx::core::default_device()), offset),
            {std::move(q), keys});
    }
    auto products = mlx::core::matmul(
        mlx::core::reshape(mlx::core::astype(query, mlx::core::float32),
            Shape{1, rows * 4, 128}),
        mlx::core::transpose(mlx::core::astype(keys, mlx::core::float32), {0, 2, 1}));
    auto scores = mlx::core::sum(mlx::core::maximum(
        mlx::core::reshape(products, Shape{1, rows, 4, count}), array(0.0f)), -2)
        / std::sqrt(128.0f);
    auto complete = mlx::core::floor_divide(
        mlx::core::arange(offset + 1, offset + rows + 1, 1, mlx::core::int32),
        array(4, mlx::core::int32));
    return mlx::core::where(mlx::core::less(
        mlx::core::reshape(mlx::core::arange(count, mlx::core::int32), Shape{1, 1, count}),
        mlx::core::reshape(complete, Shape{1, rows, 1})), scores,
        array(std::numeric_limits<float>::lowest()));
}

array qsa_prefill_blocks(const array& scores, int offset) {
    const int rows = scores.shape(1);
    const int keys = scores.shape(2);
    auto ranked = [&]() {
        if (rows >= 8) {
            return mlx::core::astype(array(Shape{1, rows, 512}, mlx::core::uint32,
                std::make_shared<QsaPrefillTopKPrimitive>(
                    mlx::core::default_stream(mlx::core::default_device())),
                {mlx::core::contiguous(scores)}), mlx::core::int32);
        }
        return mlx::core::sort(mlx::core::astype(mlx::core::slice(
            mlx::core::argpartition(scores, keys - 512, -1),
            Shape{0, 0, keys - 512}, Shape{1, rows, keys}), mlx::core::int32), -1);
    }();
    auto complete = mlx::core::floor_divide(
        mlx::core::arange(offset + 1, offset + rows + 1, 1, mlx::core::int32),
        array(4, mlx::core::int32));
    auto canonical = mlx::core::broadcast_to(mlx::core::reshape(
        mlx::core::arange(512, mlx::core::int32), Shape{1, 1, 512}), Shape{1, rows, 512});
    return mlx::core::where(mlx::core::reshape(
        mlx::core::less_equal(complete, array(512, mlx::core::int32)), Shape{1, rows, 1}),
        canonical, ranked);
}

Kernel make_sparse_kernel(
    const char* name,
    std::vector<std::string> inputs,
    std::vector<std::string> outputs,
    const char* source,
    const char* header = "") {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        name,
        std::move(inputs),
        std::move(outputs),
        source,
        header,
        true,
        false,
        options);
}

class SparseIndexerTopkPrimitive final : public mlx::core::Primitive, public MlxPreparableKernel {
public:
    SparseIndexerTopkPrimitive(
        mlx::core::Stream stream, int heads, int items,
        std::array<int, 4> params)
        : Primitive(stream), heads_(heads), items_(items), params_(params),
          kernel_name_("mfq_sparse_indexer_topk512_" +
              std::to_string(heads) + "_" + std::to_string(items)) {}

    std::string preparation_key() const override { return kernel_name_; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& device = kernel_device(stream().device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_, options, [this] { return source(); });
        return device.get_kernel(kernel_name_, library);
    }

    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("sparse indexer selection requires Metal");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        auto& output = outputs.front();
        output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto* kernel = prepared_kernel();
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(
            kernel);
        encoder.set_input_array(inputs.front(), 0);
        encoder.set_bytes(params_, 1);
        encoder.set_output_array(output, 2);
        encoder.dispatch_threadgroups(
            MTL::Size(output.shape(0) * output.shape(1), 1, 1),
            MTL::Size(256, 1, 1));
    }

    const char* name() const override { return "SparseIndexerTopk512"; }

    bool is_equivalent(const mlx::core::Primitive& other) const override {
        const auto* primitive = dynamic_cast<const SparseIndexerTopkPrimitive*>(&other);
        return primitive && primitive->heads_ == heads_ &&
            primitive->items_ == items_ && primitive->params_ == params_;
    }

private:
    std::string source() const {
        std::string code = "#include <metal_stdlib>\nusing namespace metal;\n";
        code += "#define HEADS " + std::to_string(heads_) +
            "\n#define ITEMS " + std::to_string(items_) + "\n";
        code += "kernel void " + kernel_name_ + "("
            "device const float* scores [[buffer(0)]], "
            "constant int* params [[buffer(1)]], "
            "device int* out [[buffer(2)]], "
            "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], "
            "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]], "
            "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], "
            "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {\n";
        code += kSparseIndexerTopkSource;
        code += "}\n";
        return code;
    }

    int heads_;
    int items_;
    std::array<int, 4> params_;
    std::string kernel_name_;
};

const Kernel& deepselect_topk512_kernel() {
    static const auto kernel = make_sparse_kernel(
        "mfq_cpp_deepselect_topk512",
        {"x", "valid_keys", "topk_params"},
        {"out"},
        kDeepSelectTopkSource,
        kDeepSelectTopkHeader);
    return kernel;
}

const Kernel& sparse_block_nax_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    static const auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_sparse_block_gqa_nax",
        {"q", "k", "v", "sel", "params", "scale"}, {"out"},
        kSparseBlockNaxSource,
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "#define UNROLL _Pragma(\"clang loop unroll(full)\")\n",
        false, false, options);
    return kernel;
}

const Kernel& sparse_block_nax_prefill_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    static const auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_sparse_block_gqa_prefill_nax",
        {"q", "k", "v", "sel", "params", "scale"}, {"out"},
        kSparseBlockNaxSource,
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "#define UNROLL _Pragma(\"clang loop unroll(full)\")\n"
        "#define MFQ_QSA_PREFILL 1\n",
        false, false, options);
    return kernel;
}

bool sparse_block_nax_available() {
    if (mlx_metal_nax_disabled()) return false;
    static const bool available = [] {
        if (!__builtin_available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
            return false;
        }
        const auto& device = kernel_device(mlx::core::Device::gpu);
        const auto& architecture = device.get_architecture();
        return !architecture.empty() && device.get_architecture_gen() >=
            (architecture.back() == 'p' ? 18 : 17);
    }();
    return available;
}

const Kernel& sparse_selected_mla_short_kernel() {
    static const auto kernel = make_sparse_kernel(
        "mfq_cpp_sparse_selected_mla_short",
        {"q", "kv", "indices", "mask", "sinks", "params"},
        {"out"},
        kSparseAttentionSource);
    return kernel;
}

const Kernel& sparse_selected_mla_decode_kernel() {
    static const auto kernel = make_sparse_kernel(
        "mfq_cpp_sparse_selected_mla_decode",
        {"q", "kv", "indices", "mask", "sinks", "params"},
        {"out"},
        kSparseAttentionDecodeSource);
    return kernel;
}

constexpr const char* kSparseBlockGatherSource = R"METAL(
            uint index = thread_position_in_grid.x;
            uint key_count = uint(params[0]);
            uint selected = uint(params[1]);
            uint valid_blocks = uint(params[2]);
            uint queries = uint(params[3]);
            uint row = thread_position_in_grid.y;
            uint batch = row / queries;
            uint visible = uint(params[4]) + row % queries + 1;
            uint complete = visible / uint(BLOCK_SIZE);
            uint row_blocks = min(uint(params[5]), complete);
            if (index >= uint(KV_HEADS) * selected * uint(DIM)) return;
            uint dim = index % uint(DIM);
            uint token = (index / uint(DIM)) % selected;
            uint head = index / (selected * uint(DIM));
            long source = token < valid_blocks * uint(BLOCK_SIZE)
                ? (token < row_blocks * uint(BLOCK_SIZE)
                    ? long(blocks[ulong(row) * uint(params[5]) + token / uint(BLOCK_SIZE)]) * long(BLOCK_SIZE) +
                        long(token % uint(BLOCK_SIZE))
                    : -1)
                : long(complete * uint(BLOCK_SIZE)) +
                    long(token - valid_blocks * uint(BLOCK_SIZE));
            bool present = source >= 0 && source < long(key_count) && source < long(visible);
            ulong key_offset = ulong(batch) * ulong(kv_strides[0]) +
                ulong(head) * ulong(kv_strides[1]) +
                ulong(present ? source : 0) * ulong(kv_strides[2]) +
                ulong(dim) * ulong(kv_strides[3]);
            ulong value_offset = ulong(batch) * ulong(kv_strides[4]) +
                ulong(head) * ulong(kv_strides[5]) +
                ulong(present ? source : 0) * ulong(kv_strides[6]) +
                ulong(dim) * ulong(kv_strides[7]);
            ulong output = ulong(row) * uint(KV_HEADS) * selected * uint(DIM) + index;
            selected_keys[output] = present ? keys[key_offset] : 0;
            selected_values[output] = present ? values[value_offset] : 0;
            if (head == 0 && dim == 0) valid[row * selected + token] = present;
        )METAL";

class SparseBlockGatherPrimitive final : public mlx::core::Primitive, public MlxPreparableKernel {
public:
    SparseBlockGatherPrimitive(
        mlx::core::Stream stream, Dtype dtype,
        int kv_heads, int block_size, std::array<int, 6> params)
        : Primitive(stream), dtype_(dtype), kv_heads_(kv_heads),
          block_size_(block_size), params_(params),
          kernel_name_("mfq_sparse_block_gqa_gather") {
        kernel_name_ += dtype_ == mlx::core::bfloat16 ? "_bf16" : "_f16";
        kernel_name_ += "_" + std::to_string(kv_heads_) +
            "_" + std::to_string(block_size_);
    }

    std::string preparation_key() const override { return kernel_name_; }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& device = kernel_device(stream().device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            kernel_name_, options, [this] { return source(); });
        return device.get_kernel(kernel_name_, library);
    }

    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("sparse KV gather requires Metal");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        std::vector<array>& outputs) override {
        for (auto& output : outputs) {
            output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        }
        auto* kernel = prepared_kernel();
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        encoder.set_compute_pipeline_state(
            kernel);
        for (int index = 0; index < 3; ++index) {
            encoder.set_input_array(inputs[index], index);
        }
        encoder.set_bytes(params_, 3);
        std::array<std::int64_t, 8> kv_strides{};
        for (int input = 0; input < 2; ++input) {
            for (int axis = 0; axis < 4; ++axis) {
                kv_strides[input * 4 + axis] = inputs[input].strides(axis);
            }
        }
        encoder.set_bytes(kv_strides, 7);
        for (int index = 0; index < 3; ++index) {
            encoder.set_output_array(outputs[index], index + 4);
        }
        encoder.dispatch_threads(
            MTL::Size(kv_heads_ * params_[1] * 256, inputs[2].shape(0) * params_[3], 1),
            MTL::Size(256, 1, 1));
    }

    const char* name() const override { return "SparseBlockKvGather"; }

private:
    std::string source() const {
        std::string code =
            "#include <metal_stdlib>\nusing namespace metal;\n";
        code += dtype_ == mlx::core::bfloat16
            ? "using T = bfloat;\n" : "using T = half;\n";
        code += "#define KV_HEADS " + std::to_string(kv_heads_) +
            "\n#define DIM 256\n#define BLOCK_SIZE " +
            std::to_string(block_size_) + "\n";
        code += "kernel void " + kernel_name_ + "("
            "device const T* keys [[buffer(0)]], "
            "device const T* values [[buffer(1)]], "
            "device const int* blocks [[buffer(2)]], "
            "constant int* params [[buffer(3)]], "
            "device T* selected_keys [[buffer(4)]], "
            "device T* selected_values [[buffer(5)]], "
            "device bool* valid [[buffer(6)]], "
            "constant long* kv_strides [[buffer(7)]], "
            "uint3 thread_position_in_grid [[thread_position_in_grid]]) {\n";
        code += kSparseBlockGatherSource;
        code += "}\n";
        return code;
    }

    Dtype dtype_;
    int kv_heads_;
    int block_size_;
    std::array<int, 6> params_;
    std::string kernel_name_;
};

const Kernel& sparse_circular_mla_decode_kernel() {
    static const auto kernel = make_sparse_kernel(
        "mfq_cpp_sparse_circular_mla_decode",
        {
            "q",
            "local_kv",
            "pooled_kv",
            "topk",
            "sinks",
            "params",
            "decode_params",
        },
        {"out"},
        kSparseAttentionDirectDecodeSource);
    return kernel;
}

array typed_contiguous(const array& input, Dtype dtype) {
    auto result = input;
    if (result.dtype() != dtype) {
        result = mlx::core::astype(result, dtype);
    }
    return mlx::core::contiguous(result);
}

array generic_selected_mla_attention(
    const array& query,
    const array& cache,
    const array& indices,
    const array& mask,
    const array& sinks,
    float scale) {
    const int batch = query.shape(0);
    const int heads = query.shape(1);
    const int tokens = query.shape(2);
    const int selected = indices.shape(2);
    const int dimension = query.shape(3);
    auto safe_indices = mlx::core::maximum(
        indices, array(0, mlx::core::int32));
    auto expanded_cache = mlx::core::broadcast_to(
        mlx::core::expand_dims(cache, 1),
        Shape{batch, tokens, cache.shape(1), dimension});
    auto expanded_indices = mlx::core::broadcast_to(
        mlx::core::expand_dims(safe_indices, -1),
        Shape{batch, tokens, selected, dimension});
    auto gathered = mlx::core::astype(
        mlx::core::take_along_axis(
            expanded_cache, expanded_indices, 2),
        mlx::core::float32);
    auto query_values = mlx::core::astype(
        mlx::core::transpose(query, {0, 2, 1, 3}),
        mlx::core::float32);
    auto scores = mlx::core::sum(
        mlx::core::expand_dims(query_values, 3) *
            mlx::core::expand_dims(gathered, 2),
        -1) * scale;
    scores = scores + mlx::core::expand_dims(
        mlx::core::astype(mask, mlx::core::float32), 2);
    auto sink_values = mlx::core::reshape(
        mlx::core::astype(sinks, mlx::core::float32),
        Shape{1, 1, heads});
    auto maximum = mlx::core::maximum(
        mlx::core::max(scores, -1), sink_values);
    auto exponentials = mlx::core::exp(
        scores - mlx::core::expand_dims(maximum, -1));
    auto denominator = mlx::core::sum(exponentials, -1) +
        mlx::core::exp(sink_values - maximum);
    auto probabilities = exponentials /
        mlx::core::expand_dims(denominator, -1);
    return mlx::core::sum(
        mlx::core::expand_dims(probabilities, -1) *
            mlx::core::expand_dims(gathered, 2),
        3);
}

int checked_grid_product(
    std::initializer_list<int> factors,
    const char* label) {
    std::int64_t product = 1;
    for (const int factor : factors) {
        if (factor < 0 ||
            (factor != 0 &&
             product > std::numeric_limits<int>::max() / factor)) {
            throw std::invalid_argument(
                std::string(label) + " exceeds MLX grid limits");
        }
        product *= factor;
    }
    return static_cast<int>(product);
}

struct SparseBlockGqaParams {
    std::int32_t batch = 0;
    std::int32_t query_heads = 0;
    std::int32_t kv_heads = 0;
    std::int32_t queries = 0;
    std::int32_t keys = 0;
    std::int32_t selected_blocks = 0;
    std::int32_t gqa_factor = 0;
    std::int32_t query_offset = 0;
    std::int32_t block_size = 0;
    float scale = 0.0f;
    std::int64_t query_strides[3]{};
    std::int64_t key_strides[3]{};
    std::int64_t value_strides[3]{};
    std::int64_t block_strides[3]{};
};

struct SparseSelectedMlaParams {
    std::int32_t batch = 0;
    std::int32_t queries = 0;
    std::int32_t keys = 0;
    std::int32_t selected = 0;
    float scale = 0.0f;
};

struct SparseCircularMlaParams {
    std::int32_t batch = 0;
    std::int32_t queries = 0;
    std::int32_t local_length = 0;
    std::int32_t pool_capacity = 0;
    std::int32_t pool_length = 0;
    std::int32_t topk = 0;
    std::int32_t local_window = 0;
    std::int32_t pool_ratio = 0;
    std::int32_t query_offset = 0;
    float scale = 0.0f;
};

int sdpa_decode_partitions(mlx::core::Stream stream, int keys, int gqa) {
    const auto& architecture = kernel_device(stream.device).get_architecture();
    if (architecture.empty()) return 0;
    const char family = architecture.back();
    const bool two_pass = family == 'd' ? keys >= 1024
        : (family == 's' && keys >= 1024) || (gqa > 1 && keys >= 4096);
    if (!two_pass) return 0;
    if (family == 's') {
        if (keys > 1024 && gqa > 4)
            return keys <= 8192 ? 128 : keys <= 32768 ? 256 : keys <= 65536 ? 512 : 1024;
        return 64;
    }
    if (family == 'd') {
        const int scaled = ((keys + 255) / 256 + 31) / 32 * 32;
        return std::min(256, std::max(keys >= 4096 ? 64 : 32, scaled));
    }
    return gqa >= 4 ? 64 : 32;
}

class SparseBlockGqaPrimitive final
    : public mlx::core::UnaryPrimitive, public MlxPreparableKernel {
public:
    SparseBlockGqaPrimitive(
        mlx::core::Stream stream,
        SparseBlockGqaParams params,
        mlx::core::Dtype dtype,
        bool vector_decode = false)
        : UnaryPrimitive(stream),
          params_(params),
          dtype_(dtype), vector_decode_(vector_decode) {
        if (vector_decode_) {
            nax_decode_ = params.block_size == 4 && sparse_block_nax_available();
            decode_keys_ = params.selected_blocks * params.block_size +
                (params.queries == 1 ? (params.query_offset + 1) % params.block_size : params.block_size - 1);
            partitions_ = nax_decode_ ? 64 : sdpa_decode_partitions(stream, decode_keys_, params.gqa_factor);
        }
    }

    std::string preparation_key() const override { return "mfq_sparse_block_gqa_" + mlx::core::type_to_name(dtype_) + "_" + std::to_string(vector_decode_) + "_" + std::to_string(partitions_) + "_bs" + std::to_string(params_.block_size) + "_nax" + std::to_string(nax_decode_); }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& device = kernel_device(stream().device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        if (vector_decode_) {
            auto* library = device.get_library("mfq_sdpa_decode_v1", options, sdpa_decode_source);
            if (nax_decode_) {
                (void)device.get_kernel(dtype_ == mlx::core::float16
                    ? "mfq_sdpa_decode_2pass_2_f16_256" : "mfq_sdpa_decode_2pass_2_bf16_256", library);
                return prepared_nax_kernel();
            }
            const bool yes = true, no = false;
            mlx::core::metal::MTLFCList constants{
                {&no, MTL::DataType::DataTypeBool, 20}, {&no, MTL::DataType::DataTypeBool, 21},
                {&no, MTL::DataType::DataTypeBool, 22}, {&no, MTL::DataType::DataTypeBool, 23},
                {&no, MTL::DataType::DataTypeBool, 24}, {&no, MTL::DataType::DataTypeBool, 25},
                {&partitions_, MTL::DataType::DataTypeInt, 26}, {&yes, MTL::DataType::DataTypeBool, 27},
                {&params_.block_size, MTL::DataType::DataTypeInt, 28},
            };
            const std::string type = dtype_ == mlx::core::float16 ? "f16" : "bf16";
            (void)device.get_kernel("mfq_sdpa_decode_2pass_2_" + type + "_256", library);
            const std::string name = "mfq_sdpa_decode_sparse_" + type + "_256";
            return device.get_kernel(
                name, library, name + "_sparse_" + std::to_string(partitions_) + "_bs" + std::to_string(params_.block_size), constants);
        }
        auto* library = device.get_library(
            "mfq_sparse_attention_v1",
            options,
            sparse_attention_source);
        const char* kernel_name = dtype_ == mlx::core::float16
            ? "mfq_sparse_block_gqa_f16_bk64_dc64_gqa12_d256_wm2"
            : "mfq_sparse_block_gqa_bf16_bk64_dc64_gqa12_d256_wm2";
        return device.get_kernel(kernel_name, library);
    }

    void eval_cpu(const std::vector<array>&, array&) override {
        throw std::runtime_error(
            "selected-block sparse GQA has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 4) {
            throw std::logic_error(
                "selected-block sparse GQA input count mismatch");
        }
        output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = kernel_device(selected_stream.device);
        auto* kernel = prepared_kernel();
        auto& encoder = mlx::core::metal::get_command_encoder(selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 4; ++index) {
            encoder.set_input_array(inputs[static_cast<std::size_t>(index)], index);
        }
        auto params = params_;
        for (int axis = 0; axis < 3; ++axis) {
            params.query_strides[axis] = inputs[0].strides(axis);
            params.key_strides[axis] = inputs[1].strides(axis);
            params.value_strides[axis] = inputs[2].strides(axis);
            params.block_strides[axis] = inputs[3].strides(axis);
        }
        encoder.set_bytes(params, 5);
        if (vector_decode_) {
            const int partitions = partitions_;
            const int rows = params.batch * params.queries * params.query_heads;
            array partial(Shape{rows, partitions, 256}, mlx::core::float32, nullptr, {});
            array maximum(Shape{rows, partitions}, mlx::core::float32, nullptr, {});
            array denominator(Shape{rows, partitions}, mlx::core::float32, nullptr, {});
            for (auto* temporary : {&partial, &maximum, &denominator}) {
                temporary->set_data(mlx::core::allocator::malloc(temporary->nbytes()));
                encoder.add_temporary(*temporary);
            }
            if (nax_decode_) {
                encoder.set_output_array(partial, 4);
                encoder.set_output_array(maximum, 6);
                encoder.set_output_array(denominator, 7);
                encoder.dispatch_threadgroups(
                    MTL::Size(params.queries * partitions, params.kv_heads, params.batch),
                    MTL::Size(32, 2, 1));
            } else {
                encoder.set_output_array(partial, 3);
                encoder.set_output_array(denominator, 4);
                encoder.set_output_array(maximum, 5);
                encoder.set_input_array(inputs[3], 6);
                encoder.set_bytes(decode_keys_, 7);
                const std::size_t strides[]{std::size_t(params.key_strides[1]), std::size_t(params.key_strides[2]),
                    std::size_t(params.value_strides[1]), std::size_t(params.value_strides[2])};
                for (int index = 0; index < 4; ++index) encoder.set_bytes(strides[index], 8 + index);
                encoder.set_bytes(params.scale, 12);
                encoder.set_bytes(params, 19);
                encoder.dispatch_threadgroups(
                    MTL::Size(params.kv_heads, params.batch * params.queries, partitions),
                    MTL::Size(32, 12, 1));
            }
            const std::string reduce_name = dtype_ == mlx::core::float16
                ? "mfq_sdpa_decode_2pass_2_f16_256" : "mfq_sdpa_decode_2pass_2_bf16_256";
            CompileOptions options;
            options.math_mode = MathMode::Fast;
            auto* library = device.get_library("mfq_sdpa_decode_v1", options, sdpa_decode_source);
            encoder.set_compute_pipeline_state(device.get_kernel(reduce_name, library));
            encoder.set_input_array(partial, 0);
            encoder.set_input_array(denominator, 1);
            encoder.set_input_array(maximum, 2);
            encoder.set_output_array(output, 3);
            encoder.set_bytes(partitions, 4);
            encoder.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(1024, 1, 1));
            return;
        }
        encoder.set_output_array(output, 4);
        encoder.dispatch_threadgroups(
            MTL::Size(params_.queries, params_.kv_heads, params_.batch),
            MTL::Size(32, 2, 1));
    }

    const char* name() const override {
        return "SparseBlockGqaPrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive =
            dynamic_cast<const SparseBlockGqaPrimitive*>(&other);
        return primitive != nullptr
            && primitive->dtype_ == dtype_
            && primitive->vector_decode_ == vector_decode_
            && primitive->nax_decode_ == nax_decode_
            && primitive->params_.batch == params_.batch
            && primitive->params_.query_heads == params_.query_heads
            && primitive->params_.kv_heads == params_.kv_heads
            && primitive->params_.queries == params_.queries
            && primitive->params_.keys == params_.keys
            && primitive->params_.selected_blocks == params_.selected_blocks
            && primitive->params_.query_offset == params_.query_offset
            && primitive->params_.block_size == params_.block_size
            && primitive->params_.scale == params_.scale;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {Shape{
            params_.batch,
            params_.queries,
            params_.query_heads,
            256,
        }};
    }

private:
    MTL::ComputePipelineState* prepared_nax_kernel() {
        auto& device = kernel_device(stream().device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        const std::string name = "mfq_sparse_block_gqa_split_nax_f32_" + mlx::core::type_to_name(dtype_);
        auto* library = device.get_library(name, options, [this, name] {
            std::string source = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
            source += sparse_attention_source();
            source += "\n#define UNROLL _Pragma(\"clang loop unroll(full)\")\n"
                "#define MFQ_QSA_SPLIT_PARTITIONS 64\n#define GQA 12\n"
                "#define TOPK configuration.selected_blocks\n";
            source += "template<typename T> [[kernel, max_total_threads_per_threadgroup(64)]] void "
                + name + "(const device T* q_data [[buffer(0)]], const device T* k_data [[buffer(1)]], "
                "const device T* v_data [[buffer(2)]], const device int* sel [[buffer(3)]], "
                "device float* out [[buffer(4)]], constant MfqSparseBlockGqaParams& configuration [[buffer(5)]], "
                "device float* partial_maximum [[buffer(6)]], device float* partial_denominator [[buffer(7)]], "
                "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], "
                "ushort simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]], "
                "ushort thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {\n";
            source += R"MSL(
                const device T* q = q_data;
                const device T* k = k_data;
                const device T* v = v_data;
                const long q_strides[] = {configuration.query_strides[0], configuration.query_strides[1], configuration.query_strides[2]};
                const long k_strides[] = {configuration.key_strides[0], configuration.key_strides[1], configuration.key_strides[2]};
                const long v_strides[] = {configuration.value_strides[0], configuration.value_strides[1], configuration.value_strides[2]};
                const int params[] = {configuration.query_offset, configuration.keys, configuration.queries};
                const float scale[] = {configuration.scale};
            )MSL";
            source += kSparseBlockNaxSource;
            source += "\n}\ntemplate [[host_name(\"" + name + "\")]] [[kernel]] decltype("
                + name + "<" + (dtype_ == mlx::core::float16 ? "half" : "bfloat") + ">) "
                + name + "<" + (dtype_ == mlx::core::float16 ? "half" : "bfloat") + ">;\n";
            return source;
        });
        return device.get_kernel(name, library);
    }

    SparseBlockGqaParams params_;
    mlx::core::Dtype dtype_;
    bool vector_decode_;
    bool nax_decode_ = false;
    int partitions_ = 0;
    int decode_keys_ = 0;
};

class SparseSelectedMlaPrimitive final
    : public mlx::core::UnaryPrimitive, public MlxPreparableKernel {
public:
    SparseSelectedMlaPrimitive(
        mlx::core::Stream stream,
        SparseSelectedMlaParams params,
        Dtype dtype)
        : UnaryPrimitive(stream), params_(params), dtype_(dtype) {}

    std::string preparation_key() const override { return "mfq_sparse_selected_mla_" + mlx::core::type_to_name(dtype_); }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& device = kernel_device(stream().device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            "mfq_sparse_attention_v1",
            options,
            sparse_attention_source);
        const char* kernel_name = dtype_ == mlx::core::bfloat16
            ? "mfq_dsa_sparse_prefill_bf16_bk256_dc32"
            : "mfq_dsa_sparse_prefill_f16_bk256_dc32";

        return device.get_kernel(kernel_name, library);
    }

    void eval_cpu(const std::vector<array>&, array&) override {
        throw std::runtime_error(
            "selected-token sparse MLA has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 5) {
            throw std::logic_error(
                "selected-token sparse MLA input count mismatch");
        }
        output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = kernel_device(selected_stream.device);
        auto* kernel = prepared_kernel();
        auto& encoder =
            mlx::core::metal::get_command_encoder(selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 5; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)], index);
        }
        encoder.set_output_array(output, 5);
        encoder.set_bytes(params_, 6);
        encoder.dispatch_threadgroups(
            MTL::Size(params_.queries, params_.batch, 1),
            MTL::Size(32, 8, 1));
    }

    const char* name() const override {
        return "SparseSelectedMlaPrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive = dynamic_cast<
            const SparseSelectedMlaPrimitive*>(&other);
        return primitive != nullptr
            && primitive->dtype_ == dtype_
            && primitive->params_.batch == params_.batch
            && primitive->params_.queries == params_.queries
            && primitive->params_.keys == params_.keys
            && primitive->params_.selected == params_.selected
            && primitive->params_.scale == params_.scale;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {Shape{
            params_.batch,
            params_.queries,
            64,
            512,
        }};
    }

private:
    SparseSelectedMlaParams params_;
    Dtype dtype_;
};

class SparseCircularMlaPrimitive final
    : public mlx::core::UnaryPrimitive, public MlxPreparableKernel {
public:
    SparseCircularMlaPrimitive(
        mlx::core::Stream stream,
        SparseCircularMlaParams params,
        Dtype dtype)
        : UnaryPrimitive(stream), params_(params), dtype_(dtype) {}

    std::string preparation_key() const override { return "mfq_sparse_circular_mla_" + mlx::core::type_to_name(dtype_); }
    void prepare_gpu() override { (void)prepared_kernel(); }

    MTL::ComputePipelineState* prepared_kernel() {
        auto& device = kernel_device(stream().device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(
            "mfq_sparse_attention_v1",
            options,
            sparse_attention_source);
        const char* kernel_name = dtype_ == mlx::core::bfloat16
            ? "mfq_dsa_sparse_circular_bf16_bk256_dc32"
            : "mfq_dsa_sparse_circular_f16_bk256_dc32";

        return device.get_kernel(kernel_name, library);
    }

    void eval_cpu(const std::vector<array>&, array&) override {
        throw std::runtime_error(
            "circular sparse MLA has no CPU path");
    }

    void eval_gpu(
        const std::vector<array>& inputs,
        array& output) override {
        if (inputs.size() != 5) {
            throw std::logic_error(
                "circular sparse MLA input count mismatch");
        }
        output.set_data(mlx::core::allocator::malloc(output.nbytes()));
        auto& selected_stream = stream();
        auto& device = kernel_device(selected_stream.device);
        auto* kernel = prepared_kernel();
        auto& encoder =
            mlx::core::metal::get_command_encoder(selected_stream);
        encoder.set_compute_pipeline_state(kernel);
        for (int index = 0; index < 5; ++index) {
            encoder.set_input_array(
                inputs[static_cast<std::size_t>(index)], index);
        }
        encoder.set_output_array(output, 5);
        encoder.set_bytes(params_, 6);
        encoder.dispatch_threadgroups(
            MTL::Size(params_.queries, params_.batch, 1),
            MTL::Size(32, 8, 1));
    }

    const char* name() const override {
        return "SparseCircularMlaPrimitive";
    }

    bool is_equivalent(
        const mlx::core::Primitive& other) const override {
        const auto* primitive = dynamic_cast<
            const SparseCircularMlaPrimitive*>(&other);
        return primitive != nullptr
            && primitive->dtype_ == dtype_
            && primitive->params_.batch == params_.batch
            && primitive->params_.queries == params_.queries
            && primitive->params_.local_length == params_.local_length
            && primitive->params_.pool_capacity == params_.pool_capacity
            && primitive->params_.pool_length == params_.pool_length
            && primitive->params_.topk == params_.topk
            && primitive->params_.local_window == params_.local_window
            && primitive->params_.pool_ratio == params_.pool_ratio
            && primitive->params_.query_offset == params_.query_offset
            && primitive->params_.scale == params_.scale;
    }

    std::vector<Shape> output_shapes(
        const std::vector<array>&) override {
        return {Shape{
            params_.batch,
            params_.queries,
            64,
            512,
        }};
    }

private:
    SparseCircularMlaParams params_;
    Dtype dtype_;
};

} // namespace

bool mlx_deepselect_topk512_preferred(int width, int rows) noexcept {
    const bool favorable_shape =
        (width >= 16384 && rows >= 64) ||
        (width >= 8192 && rows >= 128);
    if (!favorable_shape) {
        return false;
    }
    if (const auto* requested = std::getenv(
            "MFQ_METAL_DEEPSELECT")) {
        return std::strcmp(requested, "0") != 0 &&
            std::strcmp(requested, "false") != 0 &&
            std::strcmp(requested, "off") != 0;
    }
    return mlx_apple_chip_is("Apple M3 Ultra");
}

array mlx_sparse_indexer_topk512(
    const array& head_scores,
    int query_offset,
    int block_size) {
    if (head_scores.ndim() != 4 ||
        head_scores.dtype() != mlx::core::float32 ||
        head_scores.shape(0) <= 0 || head_scores.shape(1) <= 0 ||
        head_scores.shape(2) <= 0 ||
        head_scores.shape(3) < 512 || head_scores.shape(3) > 32768 ||
        query_offset < 0 || block_size <= 0 ||
        query_offset > std::numeric_limits<int>::max() - head_scores.shape(1)) {
        throw std::invalid_argument("sparse indexer expects f32 [B,M,H,512<=K<=32768]");
    }
    const int width = head_scores.shape(3);
    int items = 4;
    while (items * 256 < width) items *= 2;
    const std::array<int, 4> params{
        query_offset, block_size, width, head_scores.shape(1)};
    checked_grid_product({head_scores.shape(0), head_scores.shape(1)},
        "sparse indexer row count");
    return array(
        Shape{head_scores.shape(0), head_scores.shape(1), 512},
        mlx::core::int32,
        std::make_shared<SparseIndexerTopkPrimitive>(
            mlx::core::default_stream(mlx::core::default_device()),
            head_scores.shape(2), items, params),
        {mlx::core::contiguous(head_scores)});
}

array mlx_deepselect_topk512(
    const array& scores,
    const std::optional<array>& valid_keys) {
    auto source = typed_contiguous(scores, mlx::core::float32);
    if (source.ndim() != 3 || source.shape(0) <= 0 ||
        source.shape(1) <= 0 || source.shape(2) < 512) {
        throw std::invalid_argument(
            "DeepSelect top-k expects f32 [B,M,K>=512]");
    }
    const int batch = source.shape(0);
    const int queries = source.shape(1);
    const int keys = source.shape(2);
    array counts = valid_keys.has_value()
        ? typed_contiguous(*valid_keys, mlx::core::int32)
        : mlx::core::full(
              Shape{batch, queries}, keys, mlx::core::int32);
    if (counts.shape() != Shape{batch, queries}) {
        throw std::invalid_argument(
            "DeepSelect valid-key counts must be [B,M]");
    }
    const int rows = checked_grid_product(
        {batch, queries}, "DeepSelect top-k row count");
    const int grid = checked_grid_product(
        {rows, 1024}, "DeepSelect top-k grid");
    const array topk_params({keys}, mlx::core::int32);
    auto outputs = deepselect_topk512_kernel()(
        {source, counts, topk_params},
        {Shape{batch, queries, 512}},
        {mlx::core::int32},
        {grid, 1, 1},
        {1024, 1, 1},
        {},
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

static array sparse_block_gqa_attention_impl(
    const array& query,
    const array& key,
    const array& value,
    const array& selected_blocks,
    int query_offset,
    int block_size,
    std::optional<float> scale,
    bool prefill) {
    const auto aligned = [](const array& value) {
        if (value.ndim() != 4 || value.strides(3) != 1 || value.offset() % 16 != 0) return false;
        for (int axis = 0; axis < 3; ++axis) {
            if (value.strides(axis) % 8 != 0) return false;
        }
        return true;
    };
    const Dtype attention_dtype =
        query.dtype() == mlx::core::bfloat16 &&
        key.dtype() == mlx::core::bfloat16 &&
        value.dtype() == mlx::core::bfloat16
            ? mlx::core::bfloat16
            : mlx::core::float16;
    auto selected_query = typed_contiguous(query, attention_dtype);
    auto selected_key = key.dtype() == attention_dtype
        ? key : mlx::core::astype(key, attention_dtype);
    auto selected_value = value.dtype() == attention_dtype
        ? value : mlx::core::astype(value, attention_dtype);
    auto blocks = typed_contiguous(selected_blocks, mlx::core::int32);
    if (selected_query.ndim() != 4 || selected_key.ndim() != 4 ||
        selected_value.shape() != selected_key.shape() || blocks.ndim() != 3 ||
        selected_query.shape(0) != selected_key.shape(0) ||
        selected_query.shape(0) != blocks.shape(0) ||
        selected_query.shape(2) != blocks.shape(1) ||
        selected_query.shape(1) != 24 || selected_key.shape(1) != 2 ||
        selected_query.shape(3) != 256 || selected_key.shape(3) != 256 ||
        blocks.shape(2) <= 0 || query_offset < 0 ||
        query_offset + selected_query.shape(2) > selected_key.shape(2) ||
        block_size <= 0) {
        throw std::invalid_argument(
            "unsupported selected-block sparse GQA geometry");
    }
    const float selected_scale = scale.value_or(1.0f / std::sqrt(256.0f));
    if (!std::isfinite(selected_scale)) {
        throw std::invalid_argument(
            "selected-block sparse GQA scale must be finite");
    }
    SparseBlockGqaParams params{
        .batch = selected_query.shape(0),
        .query_heads = selected_query.shape(1),
        .kv_heads = selected_key.shape(1),
        .queries = selected_query.shape(2),
        .keys = selected_key.shape(2),
        .selected_blocks = blocks.shape(2),
        .gqa_factor = selected_query.shape(1) / selected_key.shape(1),
        .query_offset = query_offset,
        .block_size = block_size,
        .scale = selected_scale,
    };
    auto stream = mlx::core::default_stream(mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "selected-block sparse GQA requires Metal");
    }

    const char* gather_setting = std::getenv("MFQ_METAL_SPARSE_GQA_DECODE_GATHER");
    if (selected_query.shape(0) > 0 && selected_query.shape(2) > 0 &&
        selected_query.shape(2) < (prefill ? 24 : 7) &&
        (prefill || gather_setting == nullptr || std::string(gather_setting) != "0")) {
        const int queries = selected_query.shape(2);
        const int rows = selected_query.shape(0) * queries;
        const int visible = query_offset + queries;
        const int complete = visible / block_size;
        const int valid_blocks = std::min(blocks.shape(2), complete);
        const int selected = valid_blocks * block_size +
            (queries == 1 ? visible % block_size : block_size - 1);
        const int decode_keys = params.selected_blocks * block_size +
            (queries == 1 ? visible % block_size : block_size - 1);
        if (!prefill && selected >= 1024 && sdpa_decode_partitions(stream, decode_keys, params.gqa_factor) > 0 &&
            aligned(selected_query) && aligned(selected_key) && aligned(selected_value)) {
            return array(Shape{params.batch, queries, params.query_heads, 256}, attention_dtype,
                std::make_shared<SparseBlockGqaPrimitive>(stream, params, attention_dtype, true),
                {selected_query, selected_key, selected_value, blocks});
        }
        if (selected > 0) {
            const Shape gathered_shape{rows, selected_key.shape(1), selected, 256};
            auto gathered = array::make_arrays(
                {gathered_shape, gathered_shape, Shape{rows, 1, 1, selected}},
                {attention_dtype, attention_dtype, mlx::core::bool_},
                std::make_shared<SparseBlockGatherPrimitive>(
                    mlx::core::default_stream(mlx::core::default_device()),
                    attention_dtype, selected_key.shape(1), block_size,
                    std::array<int, 6>{
                        selected_key.shape(2), selected, valid_blocks,
                        queries, query_offset, blocks.shape(2)}),
                {selected_key, selected_value, blocks});
            auto query_rows = queries == 1 ? selected_query : mlx::core::reshape(
                mlx::core::transpose(selected_query, {0, 2, 1, 3}),
                Shape{rows, selected_query.shape(1), 1, 256});
            auto output = mlx::core::fast::scaled_dot_product_attention(
                query_rows, gathered[0], gathered[1], selected_scale, "", gathered[2]);
            return mlx::core::reshape(output,
                Shape{selected_query.shape(0), queries, selected_query.shape(1), 256});
        }
    }

    if (params.batch > 0 && params.queries >= (prefill ? 24 : 32) && block_size == 4 &&
        sparse_block_nax_available() &&
        aligned(selected_query) && aligned(selected_key) && aligned(selected_value)) {
        const int grid = checked_grid_product(
            {params.queries, 64}, "sparse NAX query grid");
        const bool native_prefill = prefill && params.selected_blocks == 512 && query_offset >= 2047;
        return (native_prefill ? sparse_block_nax_prefill_kernel() : sparse_block_nax_kernel())(
            {selected_query, selected_key, selected_value, blocks,
                array({query_offset, params.keys, params.queries}, mlx::core::int32),
                array({selected_scale}, mlx::core::float32)},
            {Shape{params.batch, params.queries, params.query_heads, 256}},
            {attention_dtype}, {grid, params.kv_heads, params.batch}, {64, 1, 1},
            {{"T", attention_dtype}, {"GQA", params.gqa_factor},
                {"TOPK", params.selected_blocks}},
            std::nullopt, false, {}).front();
    }

    selected_key = mlx::core::contiguous(selected_key);
    selected_value = mlx::core::contiguous(selected_value);
    return array(
        Shape{
            params.batch,
            params.queries,
            params.query_heads,
            256,
        },
        attention_dtype,
        std::make_shared<SparseBlockGqaPrimitive>(
            stream,
            params,
            attention_dtype),
        std::vector<array>{
            std::move(selected_query),
            std::move(selected_key),
            std::move(selected_value),
            std::move(blocks),
        });
}

array mlx_sparse_block_gqa_attention(
    const array& query,
    const array& key,
    const array& value,
    const array& selected_blocks,
    int query_offset,
    int block_size,
    std::optional<float> scale) {
    return sparse_block_gqa_attention_impl(
        query, key, value, selected_blocks, query_offset, block_size, scale, false);
}

array mlx_qsa_prefill_attention(
    const array& query,
    const array& key,
    const array& value,
    const array& index_query,
    const array& pooled_index_keys,
    int query_offset) {
    if (query.ndim() != 4 || key.ndim() != 4 ||
        value.shape() != key.shape() || query.shape(0) != 1 ||
        query.shape(1) != 24 || query.shape(2) < 32 || query.shape(3) != 256 ||
        key.shape(0) != 1 || key.shape(1) != 2 || key.shape(3) != 256 ||
        (query.dtype() != mlx::core::float16 && query.dtype() != mlx::core::bfloat16) ||
        key.dtype() != query.dtype() || value.dtype() != query.dtype() ||
        index_query.shape() != Shape{1, query.shape(2), 4, 128} ||
        index_query.dtype() != query.dtype() ||
        pooled_index_keys.shape() != Shape{1, key.shape(2) / 4, 128} ||
        (pooled_index_keys.dtype() != query.dtype() &&
         pooled_index_keys.dtype() != mlx::core::float32) ||
        query_offset < 0 || query.shape(2) > key.shape(2) ||
        query_offset != key.shape(2) - query.shape(2)) {
        throw std::invalid_argument("unsupported QSA prefill geometry or cache position");
    }
    const int tokens = query.shape(2);
    const int keys = key.shape(2);
    int dense_rows = std::min(tokens, std::max(0, 2051 - query_offset));
    if (tokens - dense_rows == 1) --dense_rows;
    std::vector<array> outputs;
    if (dense_rows > 0) {
        const int end = query_offset + dense_rows;
        auto dense = mlx::core::fast::scaled_dot_product_attention(
            mlx::core::slice(query, Shape{0, 0, 0, 0}, Shape{1, 24, dense_rows, 256}),
            mlx::core::slice(key, Shape{0, 0, 0, 0}, Shape{1, 2, end, 256}),
            mlx::core::slice(value, Shape{0, 0, 0, 0}, Shape{1, 2, end, 256}),
            1.0f / std::sqrt(256.0f), "causal");
        outputs.push_back(mlx::core::transpose(dense, {0, 2, 1, 3}));
        if (dense_rows == tokens) return outputs.front();
    }
    auto pooled = typed_contiguous(pooled_index_keys, query.dtype());
    const int chunk = keys <= 32768 ? 4096 : (keys <= 65536 ? 2048 : 1024);
    for (int start = dense_rows; start < tokens; start += chunk) {
        const int stop = std::min(tokens, start + chunk);
        const int offset = query_offset + start;
        auto index = mlx::core::slice(index_query,
            Shape{0, start, 0, 0}, Shape{1, stop, 4, 128});
        auto blocks = qsa_prefill_blocks(qsa_prefill_scores(index, pooled, offset), offset);
        auto queries = mlx::core::slice(query,
            Shape{0, 0, start, 0}, Shape{1, 24, stop, 256});
        outputs.push_back(sparse_block_gqa_attention_impl(
            queries, key, value, blocks, offset, 4, std::nullopt, true));
    }
    return outputs.size() == 1 ? outputs.front() : mlx::core::concatenate(outputs, 1);
}

array mlx_sparse_selected_mla_attention(
    const array& query,
    const array& kv_cache,
    const array& selected_indices,
    const array& selected_mask,
    const array& sinks,
    std::optional<float> scale) {
    const Dtype cache_dtype = kv_cache.dtype() == mlx::core::bfloat16
        ? mlx::core::bfloat16
        : mlx::core::float16;
    auto selected_query = typed_contiguous(query, mlx::core::float32);
    auto selected_cache = typed_contiguous(kv_cache, cache_dtype);
    auto indices = typed_contiguous(selected_indices, mlx::core::int32);
    auto mask = typed_contiguous(selected_mask, mlx::core::float16);
    auto sink_logits = typed_contiguous(sinks, mlx::core::float32);
    if (selected_query.ndim() != 4 || selected_query.shape(0) <= 0 ||
        selected_query.shape(1) <= 0 || selected_query.shape(2) <= 0 ||
        selected_query.shape(3) <= 0 || selected_cache.ndim() != 3 ||
        selected_cache.shape(0) != selected_query.shape(0) ||
        selected_cache.shape(1) <= 0 ||
        selected_cache.shape(2) != selected_query.shape(3) || indices.ndim() != 3 ||
        indices.shape(0) != selected_query.shape(0) ||
        indices.shape(1) != selected_query.shape(2) ||
        indices.shape(2) <= 0 || indices.shape(2) % 32 != 0 ||
        mask.shape() != indices.shape() ||
        sink_logits.size() != static_cast<std::size_t>(selected_query.shape(1))) {
        throw std::invalid_argument(
            "unsupported selected-token sparse MLA geometry");
    }
    const int heads = selected_query.shape(1);
    const int dimension = selected_query.shape(3);
    const float selected_scale = scale.value_or(
        1.0f / std::sqrt(static_cast<float>(dimension)));
    if (!std::isfinite(selected_scale) || selected_scale <= 0.0f) {
        throw std::invalid_argument(
            "selected-token sparse MLA scale must be finite and positive");
    }
    if (heads != 64 || dimension != 512) {
        return generic_selected_mla_attention(
            selected_query,
            selected_cache,
            indices,
            mask,
            sink_logits,
            selected_scale);
    }
    constexpr int kHeads = 64;
    constexpr int kDimension = 512;
    SparseSelectedMlaParams params{
        .batch = selected_query.shape(0),
        .queries = selected_query.shape(2),
        .keys = selected_cache.shape(1),
        .selected = indices.shape(2),
        .scale = selected_scale,
    };
    if (params.queries < 32) {
        const array scale_parameter({selected_scale}, mlx::core::float32);
        const Shape output_shape{
            params.batch,
            params.queries,
            kHeads,
            kDimension,
        };
        const TemplateArgs templates{};
        // The decode kernel already maps query rows independently and keeps
        // the same 32-lane dot/softmax/value reduction as M=1. Use it for
        // DSpark's M=2..6 verifier blocks as well; the former short-prefill
        // kernel used eight times as many threads and changed reduction order.
        const bool decode_consistent = params.queries <= 6;
        const int grid = decode_consistent
            ? checked_grid_product(
                {params.batch, params.queries, 16, 128},
                "selected-token sparse MLA decode grid")
            : checked_grid_product(
                {params.batch, params.queries, kHeads, 256},
                "selected-token sparse MLA short-query grid");
        auto outputs = (decode_consistent
            ? sparse_selected_mla_decode_kernel()
            : sparse_selected_mla_short_kernel())(
                {
                    selected_query,
                    selected_cache,
                    indices,
                    mask,
                    sink_logits,
                    scale_parameter,
                },
                {output_shape},
                {mlx::core::float32},
                {grid, 1, 1},
                {decode_consistent ? 128 : 256, 1, 1},
                templates,
                std::nullopt,
                false,
                {});
        return std::move(outputs.front());
    }
    auto attention_query = typed_contiguous(selected_query, cache_dtype);
    auto attention_sinks = typed_contiguous(sink_logits, cache_dtype);
    auto stream = mlx::core::default_stream(mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "selected-token sparse MLA requires Metal");
    }
    return array(
        Shape{params.batch, params.queries, kHeads, kDimension},
        cache_dtype,
        std::make_shared<SparseSelectedMlaPrimitive>(
            stream,
            params,
            cache_dtype),
        std::vector<array>{
            std::move(attention_query),
            std::move(selected_cache),
            std::move(indices),
            std::move(mask),
            std::move(attention_sinks),
        });
}

array mlx_sparse_circular_mla_attention(
    const array& query,
    const array& local_kv,
    const std::optional<array>& pooled_kv,
    int pool_len,
    const array& topk,
    const array& sinks,
    int query_offset,
    int pool_ratio,
    int local_window,
    std::optional<float> scale) {
    constexpr int kHeads = 64;
    constexpr int kDimension = 512;
    const Dtype cache_dtype =
        local_kv.dtype() == mlx::core::bfloat16 &&
        (!pooled_kv || pooled_kv->dtype() == mlx::core::bfloat16)
            ? mlx::core::bfloat16
            : mlx::core::float16;
    auto selected_query = typed_contiguous(query, cache_dtype);
    auto local = typed_contiguous(local_kv, cache_dtype);
    auto pool = pooled_kv
        ? typed_contiguous(*pooled_kv, cache_dtype)
        : local;
    auto selected_topk = typed_contiguous(topk, mlx::core::int32);
    auto sink_logits = typed_contiguous(sinks, cache_dtype);
    if (selected_query.ndim() != 4 || selected_query.shape(0) <= 0 ||
        selected_query.shape(1) != kHeads ||
        selected_query.shape(2) < 2 ||
        selected_query.shape(3) != kDimension || local.ndim() != 3 ||
        local.shape(0) != selected_query.shape(0) ||
        local.shape(1) < selected_query.shape(2) ||
        local.shape(2) != kDimension || pool.ndim() != 3 ||
        pool.shape(0) != selected_query.shape(0) ||
        pool.shape(2) != kDimension || pool_len < 0 ||
        pool_len > pool.shape(1) ||
        selected_topk.ndim() != 3 ||
        selected_topk.shape(0) != selected_query.shape(0) ||
        selected_topk.shape(1) != selected_query.shape(2) ||
        (pool_len > 0 && selected_topk.shape(2) <= 0) ||
        sink_logits.size() != kHeads ||
        query_offset < 0 || pool_ratio <= 0 || local_window <= 0) {
        throw std::invalid_argument(
            "unsupported circular multi-query sparse MLA geometry");
    }
    const float selected_scale = scale.value_or(
        1.0f / std::sqrt(static_cast<float>(kDimension)));
    if (!std::isfinite(selected_scale) || selected_scale <= 0.0f) {
        throw std::invalid_argument(
            "circular multi-query sparse MLA scale must be finite and positive");
    }
    SparseCircularMlaParams params{
        .batch = selected_query.shape(0),
        .queries = selected_query.shape(2),
        .local_length = local.shape(1),
        .pool_capacity = pool.shape(1),
        .pool_length = pool_len,
        .topk = selected_topk.shape(2),
        .local_window = local_window,
        .pool_ratio = pool_ratio,
        .query_offset = query_offset,
        .scale = selected_scale,
    };
    auto stream = mlx::core::default_stream(mlx::core::default_device());
    if (stream.device != mlx::core::Device::gpu) {
        throw std::invalid_argument(
            "circular multi-query sparse MLA requires Metal");
    }
    return array(
        Shape{params.batch, params.queries, kHeads, kDimension},
        cache_dtype,
        std::make_shared<SparseCircularMlaPrimitive>(
            stream,
            params,
            cache_dtype),
        std::vector<array>{
            std::move(selected_query),
            std::move(local),
            std::move(pool),
            std::move(selected_topk),
            std::move(sink_logits),
        });
}

array mlx_sparse_circular_mla_decode_attention(
    const array& query,
    const array& local_kv,
    const std::optional<array>& pooled_kv,
    int pool_len,
    const array& topk,
    const array& sinks,
    int sequence_length,
    int pool_ratio,
    int local_window,
    std::optional<float> scale) {
    constexpr int kHeads = 64;
    constexpr int kDimension = 512;
    const Dtype cache_dtype =
        local_kv.dtype() == mlx::core::bfloat16 &&
        (!pooled_kv || pooled_kv->dtype() == mlx::core::bfloat16)
            ? mlx::core::bfloat16
            : mlx::core::float16;
    auto selected_query = typed_contiguous(query, mlx::core::float32);
    auto local = typed_contiguous(local_kv, cache_dtype);
    auto selected_topk = typed_contiguous(topk, mlx::core::int32);
    auto sink_logits = typed_contiguous(sinks, mlx::core::float32);
    auto pool = pooled_kv
        ? typed_contiguous(*pooled_kv, cache_dtype)
        : local;
    if (selected_query.ndim() != 4 || selected_query.shape(0) <= 0 ||
        selected_query.shape(1) != kHeads || selected_query.shape(2) != 1 ||
        selected_query.shape(3) != kDimension ||
        local.shape() != Shape{
            selected_query.shape(0), local_window, kDimension} ||
        pool.ndim() != 3 || pool.shape(0) != selected_query.shape(0) ||
        pool.shape(2) != kDimension || pool_len < 0 ||
        pool_len > pool.shape(1) || selected_topk.ndim() != 3 ||
        selected_topk.shape(0) != selected_query.shape(0) ||
        selected_topk.shape(1) != 1 || sink_logits.size() != kHeads ||
        sequence_length <= 0 || pool_ratio <= 0 || local_window <= 0) {
        throw std::invalid_argument(
            "unsupported circular sparse MLA decode geometry");
    }
    const float selected_scale = scale.value_or(
        1.0f / std::sqrt(static_cast<float>(kDimension)));
    if (!std::isfinite(selected_scale) || selected_scale <= 0.0f) {
        throw std::invalid_argument(
            "circular sparse MLA decode scale must be finite and positive");
    }
    const int batch = selected_query.shape(0);
    const int topk_count = selected_topk.shape(2);
    const int grid = checked_grid_product(
        {batch, 16, 128},
        "circular sparse MLA decode grid");
    const array scale_parameter({selected_scale}, mlx::core::float32);
    const array decode_parameters(
        {sequence_length, pool_len, topk_count},
        mlx::core::int32);
    auto outputs = sparse_circular_mla_decode_kernel()(
        {
            selected_query,
            local,
            pool,
            selected_topk,
            sink_logits,
            scale_parameter,
            decode_parameters,
        },
        {Shape{batch, 1, kHeads, kDimension}},
        {mlx::core::float32},
        {grid, 1, 1},
        {128, 1, 1},
        {
            {"RATIO", pool_ratio},
            {"WINDOW", local_window},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

} // namespace mfq::metal
