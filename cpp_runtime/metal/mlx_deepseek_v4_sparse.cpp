#include "mlx_deepseek_v4_sparse.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <optional>
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

constexpr int kIndexerHeads = 64;
constexpr int kIndexerDimension = 128;
constexpr int kAttentionHeads = 64;
constexpr int kAttentionDimension = 512;

#include "mlx_deepseek_v4_sparse_kernels.inc"

Kernel make_kernel(
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

const Kernel& fp4_sim_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_fp4_sim",
        {"x"},
        {"out"},
        kFp4SimSource,
        kFp4Header);
    return kernel;
}

const Kernel& compress_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_compress",
        {
            "kv",
            "gate",
            "ape",
            "norm",
            "prev_kv",
            "prev_gate",
            "positions",
            "cos_table",
            "sin_table",
            "params",
        },
        {"out"},
        kCompressSource,
        kCompressHeader);
    return kernel;
}

const Kernel& decode_pool_step_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_decode_pool_step",
        {
            "kv_token",
            "gate_token",
            "ape",
            "norm",
            "state_kv",
            "state_gate",
            "prev_kv",
            "prev_gate",
            "seq_len",
            "cos_table",
            "sin_table",
            "params",
        },
        {
            "state_kv_out",
            "state_gate_out",
            "prev_kv_out",
            "prev_gate_out",
            "emitted",
            "emit_rows",
        },
        kDecodePoolStepSource,
        kCompressHeader);
    return kernel;
}

const Kernel& indexer_scores_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_indexer_scores",
        {"q", "k", "weights", "params"},
        {"out"},
        kIndexerScoresSource);
    return kernel;
}

const Kernel& indexer_decode_scores_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_indexer_decode_scores",
        {"q", "k", "weights", "params"},
        {"out"},
        kIndexerDecodeScoresSource);
    return kernel;
}

const Kernel& topk_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_topk512",
        {"x"},
        {"out"},
        kTopkSource,
        kTopkHeader);
    return kernel;
}

const Kernel& prefill_plan_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_prefill_plan",
        {"topk"},
        {"indices", "mask"},
        kPrefillPlanSource);
    return kernel;
}

const Kernel& decode_plan_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_decode_plan",
        {"topk", "seq_len"},
        {"indices", "mask"},
        kDecodePlanSource);
    return kernel;
}

const Kernel& sparse_attention_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_sparse_attention",
        {"q", "kv", "indices", "mask", "sinks", "params"},
        {"out"},
        kSparseAttentionSource);
    return kernel;
}

const Kernel& sparse_attention_prefill_mma_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_sparse_attention_prefill_mma",
        {"q", "kv", "indices", "mask", "sinks", "params"},
        {"out"},
        kSparseAttentionPrefillMmaSource);
    return kernel;
}

const Kernel& sparse_attention_decode_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_sparse_attention_decode",
        {"q", "kv", "indices", "mask", "sinks", "params"},
        {"out"},
        kSparseAttentionDecodeSource);
    return kernel;
}

const Kernel& sparse_attention_direct_decode_kernel() {
    static const auto kernel = make_kernel(
        "mfq_cpp_dsv4_sparse_attention_direct_decode",
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

array floating_contiguous(const array& input) {
    auto result = input;
    if (result.dtype() != mlx::core::float16 &&
        result.dtype() != mlx::core::float32) {
        result = mlx::core::astype(
            result,
            mlx::core::float16);
    }
    return mlx::core::contiguous(result);
}

int checked_product(
    std::initializer_list<int> factors,
    const char* label) {
    std::int64_t product = 1;
    for (const int factor : factors) {
        if (factor < 0 ||
            (factor != 0 &&
             product >
                 std::numeric_limits<int>::max() / factor)) {
            throw std::invalid_argument(
                std::string("DeepSeek-V4 ") + label +
                " exceeds MLX grid limits");
        }
        product *= factor;
    }
    return static_cast<int>(product);
}

void validate_eps(float eps) {
    if (!std::isfinite(eps) || eps <= 0.0f) {
        throw std::invalid_argument(
            "DeepSeek-V4 compressor eps must be finite and positive");
    }
}

void validate_table_pair(
    const array& cosine,
    const array& sine) {
    if (cosine.ndim() != 2 ||
        cosine.shape(0) <= 0 ||
        cosine.shape(1) < 32 ||
        sine.shape() != cosine.shape()) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 rotary table shape");
    }
}

void validate_compressor_mode(
    int head_dim,
    int ratio,
    int quant_mode) {
    if ((head_dim != 128 && head_dim != 512) ||
        ratio <= 0 || ratio > 128 ||
        quant_mode < 0 || quant_mode > 2 ||
        (quant_mode == 1 && head_dim != 512) ||
        (quant_mode == 2 && head_dim != 128)) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 compressor mode");
    }
}

TemplateArgs compressor_templates(
    Dtype dtype,
    int batch,
    int windows,
    int head_dim,
    int ratio,
    bool overlap,
    bool has_prev,
    int table_len,
    int table_stride,
    int quant_mode) {
    return {
        {"T", dtype},
        {"B", batch},
        {"W", windows},
        {"HEAD_DIM", head_dim},
        {"RATIO", ratio},
        {"OVERLAP", static_cast<int>(overlap)},
        {"HAS_PREV", static_cast<int>(has_prev)},
        {"TABLE_LEN", table_len},
        {"TABLE_STRIDE", table_stride},
        {"QUANT_MODE", quant_mode},
    };
}

TemplateArgs decode_compressor_templates(
    Dtype dtype,
    int batch,
    int head_dim,
    int ratio,
    bool overlap,
    int table_len,
    int table_stride,
    int quant_mode) {
    return {
        {"T", dtype},
        {"B", batch},
        {"HEAD_DIM", head_dim},
        {"RATIO", ratio},
        {"OVERLAP", static_cast<int>(overlap)},
        {"TABLE_LEN", table_len},
        {"TABLE_STRIDE", table_stride},
        {"QUANT_MODE", quant_mode},
    };
}

} // namespace

array dsv4_fp4_sim(const array& input) {
    auto source = typed_contiguous(
        input,
        mlx::core::float16);
    if (source.ndim() < 1 ||
        source.size() == 0 ||
        source.shape(-1) % 32 != 0) {
        throw std::invalid_argument(
            "DSV4 FP4 simulation expects nonempty f16 [...,32*n]");
    }
    if (source.size() >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "DSV4 FP4 simulation input exceeds MLX grid limits");
    }
    const int size = static_cast<int>(source.size());
    auto outputs = fp4_sim_kernel()(
        {source},
        {source.shape()},
        {mlx::core::float16},
        {size, 1, 1},
        {32, 1, 1},
        {},
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array dsv4_compress(
    const array& kv,
    const array& gate,
    const array& ape,
    const array& norm,
    const std::optional<array>& prev_kv,
    const std::optional<array>& prev_gate,
    const array& positions,
    const array& cos,
    const array& sin,
    int ratio,
    bool overlap,
    int quant_mode,
    float eps) {
    auto source = floating_contiguous(kv);
    auto gate_values = typed_contiguous(
        gate,
        source.dtype());
    auto ape_values = typed_contiguous(
        ape,
        mlx::core::float32);
    auto norm_values = typed_contiguous(
        norm,
        mlx::core::float32);
    auto position_ids = typed_contiguous(
        positions,
        mlx::core::int32);
    auto cosine = typed_contiguous(
        cos,
        mlx::core::float32);
    auto sine = typed_contiguous(
        sin,
        mlx::core::float32);
    validate_eps(eps);
    const int head_dim =
        static_cast<int>(norm_values.size());
    validate_compressor_mode(
        head_dim,
        ratio,
        quant_mode);
    const int output_dim =
        head_dim * (overlap ? 2 : 1);
    if (source.ndim() != 4) {
        throw std::invalid_argument(
            "invalid DSV4 compressor input");
    }
    const int batch = source.shape(0);
    const int windows = source.shape(1);
    if (batch <= 0 || windows <= 0 ||
        source.shape(2) != ratio ||
        source.shape(3) != output_dim ||
        gate_values.shape() != source.shape() ||
        ape_values.shape() !=
            Shape{ratio, output_dim} ||
        position_ids.size() !=
            static_cast<std::size_t>(batch) * windows) {
        throw std::invalid_argument(
            "invalid DSV4 compressor input");
    }
    validate_table_pair(cosine, sine);

    const bool has_kv =
        prev_kv.has_value() && prev_kv->size() != 0;
    const bool has_gate =
        prev_gate.has_value() && prev_gate->size() != 0;
    if (has_kv != has_gate) {
        throw std::invalid_argument(
            "previous KV and gate state must be provided together");
    }
    const bool has_prev = has_kv;
    array previous_kv = mlx::core::zeros(
        Shape{1},
        source.dtype());
    array previous_gate = mlx::core::zeros(
        Shape{1},
        source.dtype());
    if (has_prev) {
        previous_kv = typed_contiguous(
            *prev_kv,
            source.dtype());
        previous_gate = typed_contiguous(
            *prev_gate,
            source.dtype());
        const Shape expected{
            batch,
            ratio,
            head_dim,
        };
        if (!overlap ||
            previous_kv.shape() != expected ||
            previous_gate.shape() != expected) {
            throw std::invalid_argument(
                "invalid DSV4 overlap history");
        }
    }

    const int rows = checked_product(
        {batch, windows},
        "compressor row count");
    const int grid = checked_product(
        {rows, head_dim},
        "compressor grid");
    const array params({eps}, mlx::core::float32);
    auto outputs = compress_kernel()(
        {
            source,
            gate_values,
            ape_values,
            norm_values,
            previous_kv,
            previous_gate,
            position_ids,
            cosine,
            sine,
            params,
        },
        {Shape{batch, windows, head_dim}},
        {mlx::core::float16},
        {grid, 1, 1},
        {head_dim, 1, 1},
        compressor_templates(
            source.dtype(),
            batch,
            windows,
            head_dim,
            ratio,
            overlap,
            has_prev,
            cosine.shape(0),
            cosine.shape(1),
            quant_mode),
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

MlxDsv4PoolStep dsv4_decode_pool_step(
    const array& kv_token,
    const array& gate_token,
    const array& ape,
    const array& norm,
    const array& state_kv,
    const array& state_gate,
    const std::optional<array>& prev_kv,
    const std::optional<array>& prev_gate,
    const array& seq_len,
    const array& cos,
    const array& sin,
    int ratio,
    bool overlap,
    int quant_mode,
    float eps) {
    auto token = floating_contiguous(kv_token);
    auto gate_values = typed_contiguous(
        gate_token,
        token.dtype());
    auto ape_values = typed_contiguous(
        ape,
        mlx::core::float32);
    auto norm_values = typed_contiguous(
        norm,
        mlx::core::float32);
    auto state_values = typed_contiguous(
        state_kv,
        token.dtype());
    auto state_gate_values = typed_contiguous(
        state_gate,
        token.dtype());
    auto lengths = typed_contiguous(
        seq_len,
        mlx::core::int32);
    auto cosine = typed_contiguous(
        cos,
        mlx::core::float32);
    auto sine = typed_contiguous(
        sin,
        mlx::core::float32);
    validate_eps(eps);
    const int head_dim =
        static_cast<int>(norm_values.size());
    validate_compressor_mode(
        head_dim,
        ratio,
        quant_mode);
    const int output_dim =
        head_dim * (overlap ? 2 : 1);
    const int batch =
        token.ndim() > 0 ? token.shape(0) : 0;
    if (batch <= 0 ||
        token.ndim() != 3 ||
        token.shape(1) != 1 ||
        token.shape(2) != output_dim ||
        gate_values.shape() != token.shape() ||
        ape_values.shape() != Shape{ratio, output_dim} ||
        state_values.shape() !=
            Shape{batch, ratio, output_dim} ||
        state_gate_values.shape() !=
            state_values.shape() ||
        lengths.size() !=
            static_cast<std::size_t>(batch)) {
        throw std::invalid_argument(
            "invalid DSV4 decode pool step input");
    }
    validate_table_pair(cosine, sine);

    const Shape previous_shape{
        batch,
        ratio,
        head_dim,
    };
    array previous_kv = mlx::core::zeros(
        previous_shape,
        token.dtype());
    array previous_gate = mlx::core::zeros(
        previous_shape,
        token.dtype());
    if (overlap) {
        if (!prev_kv.has_value() ||
            !prev_gate.has_value()) {
            throw std::invalid_argument(
                "overlap decode requires previous window state");
        }
        previous_kv = typed_contiguous(
            *prev_kv,
            token.dtype());
        previous_gate = typed_contiguous(
            *prev_gate,
            token.dtype());
        if (previous_kv.shape() != previous_shape ||
            previous_gate.shape() != previous_shape) {
            throw std::invalid_argument(
                "invalid DSV4 previous window state");
        }
    } else if (
        prev_kv.has_value() != prev_gate.has_value()) {
        throw std::invalid_argument(
            "previous KV and gate state must be paired");
    }

    const int grid = checked_product(
        {batch, head_dim},
        "decode compressor grid");
    const array params({eps}, mlx::core::float32);
    auto outputs = decode_pool_step_kernel()(
        {
            token,
            gate_values,
            ape_values,
            norm_values,
            state_values,
            state_gate_values,
            previous_kv,
            previous_gate,
            lengths,
            cosine,
            sine,
            params,
        },
        {
            state_values.shape(),
            state_gate_values.shape(),
            previous_shape,
            previous_shape,
            Shape{batch, 1, head_dim},
            Shape{batch},
        },
        {
            token.dtype(),
            token.dtype(),
            token.dtype(),
            token.dtype(),
            mlx::core::float16,
            mlx::core::int32,
        },
        {grid, 1, 1},
        {head_dim, 1, 1},
        decode_compressor_templates(
            token.dtype(),
            batch,
            head_dim,
            ratio,
            overlap,
            cosine.shape(0),
            cosine.shape(1),
            quant_mode),
        std::nullopt,
        false,
        {});

    std::optional<array> next_previous_kv;
    std::optional<array> next_previous_gate;
    if (overlap) {
        next_previous_kv =
            std::move(outputs.at(2));
        next_previous_gate =
            std::move(outputs.at(3));
    }
    return {
        std::move(outputs.at(4)),
        std::move(outputs.at(5)),
        std::move(outputs.at(0)),
        std::move(outputs.at(1)),
        std::move(next_previous_kv),
        std::move(next_previous_gate),
    };
}

MlxDsv4PoolUpdate dsv4_decode_pool_update(
    const array& kv_token,
    const array& gate_token,
    const array& ape,
    const array& norm,
    const array& state_kv,
    const array& state_gate,
    const std::optional<array>& prev_kv,
    const std::optional<array>& prev_gate,
    const array& pool,
    const array& seq_len,
    const array& cos,
    const array& sin,
    int ratio,
    bool overlap,
    int quant_mode,
    float eps) {
    auto pool_values = typed_contiguous(
        pool,
        mlx::core::float16);
    auto step = dsv4_decode_pool_step(
        kv_token,
        gate_token,
        ape,
        norm,
        state_kv,
        state_gate,
        prev_kv,
        prev_gate,
        seq_len,
        cos,
        sin,
        ratio,
        overlap,
        quant_mode,
        eps);
    const int batch = step.emitted.shape(0);
    const int head_dim = step.emitted.shape(2);
    if (pool_values.ndim() != 3 ||
        pool_values.shape(0) != batch ||
        pool_values.shape(1) <= 0 ||
        pool_values.shape(2) != head_dim) {
        throw std::invalid_argument(
            "invalid DSV4 decode pool update input");
    }
    const int capacity = pool_values.shape(1);
    const auto zero = array(0, mlx::core::int32);
    const auto last = array(
        capacity - 1,
        mlx::core::int32);
    auto safe_rows = mlx::core::minimum(
        mlx::core::maximum(
            step.emit_rows,
            zero),
        last);
    auto row_indices = mlx::core::broadcast_to(
        mlx::core::reshape(
            safe_rows,
            Shape{batch, 1, 1}),
        Shape{batch, 1, head_dim});
    auto existing = mlx::core::take_along_axis(
        pool_values,
        row_indices,
        1);
    auto valid = mlx::core::logical_and(
        mlx::core::greater_equal(
            step.emit_rows,
            zero),
        mlx::core::less(
            step.emit_rows,
            array(capacity, mlx::core::int32)));
    auto replacement = mlx::core::where(
        mlx::core::reshape(
            valid,
            Shape{batch, 1, 1}),
        step.emitted,
        existing);
    auto next_pool = mlx::core::put_along_axis(
        pool_values,
        row_indices,
        replacement,
        1);
    return {
        std::move(next_pool),
        std::move(step.state_kv),
        std::move(step.state_gate),
        std::move(step.prev_kv),
        std::move(step.prev_gate),
    };
}

array dsv4_indexer_scores_decode(
    const array& q,
    const array& k,
    const array& weights,
    int query_offset,
    int ratio) {
    auto query = typed_contiguous(
        q,
        mlx::core::float16);
    auto key = typed_contiguous(
        k,
        mlx::core::float16);
    auto head_weights = typed_contiguous(
        weights,
        mlx::core::float16);
    if (query.ndim() != 4 ||
        key.ndim() != 3 ||
        head_weights.ndim() != 3 ||
        query.shape(1) != 1 ||
        query.shape(2) != kIndexerHeads ||
        query.shape(3) != kIndexerDimension ||
        query.shape(0) <= 0 ||
        key.shape(0) != query.shape(0) ||
        key.shape(1) <= 0 ||
        key.shape(2) != kIndexerDimension ||
        head_weights.shape() !=
            Shape{query.shape(0), 1, kIndexerHeads} ||
        query_offset < 0 ||
        ratio <= 0) {
        throw std::invalid_argument(
            "DSV4 decode indexer score shape mismatch");
    }
    const int batch = query.shape(0);
    const int keys = key.shape(1);
    constexpr int threads = 128;
    const int key_tiles =
        (keys + threads - 1) / threads;
    const int groups = checked_product(
        {batch, key_tiles},
        "decode indexer group count");
    const int grid = checked_product(
        {groups, threads},
        "decode indexer grid");
    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(
                kIndexerDimension * kIndexerHeads));
    const array params({scale}, mlx::core::float32);
    auto outputs = indexer_decode_scores_kernel()(
        {
            query,
            key,
            head_weights,
            params,
        },
        {Shape{batch, 1, keys}},
        {mlx::core::float16},
        {grid, 1, 1},
        {threads, 1, 1},
        {
            {"B", batch},
            {"K", keys},
            {"KEY_TILES", key_tiles},
            {"THREADS", threads},
            {"QUERY_OFFSET", query_offset},
            {"RATIO", ratio},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array dsv4_indexer_scores(
    const array& q,
    const array& k,
    const array& weights,
    int query_offset,
    int ratio) {
    auto query = typed_contiguous(
        q,
        mlx::core::float16);
    auto key = typed_contiguous(
        k,
        mlx::core::float16);
    auto head_weights = typed_contiguous(
        weights,
        mlx::core::float16);
    if (query.ndim() != 4 ||
        key.ndim() != 3 ||
        head_weights.ndim() != 3 ||
        query.shape(0) <= 0 ||
        query.shape(1) <= 0 ||
        query.shape(2) != kIndexerHeads ||
        query.shape(3) != kIndexerDimension ||
        key.shape(0) != query.shape(0) ||
        key.shape(1) <= 0 ||
        key.shape(2) != kIndexerDimension ||
        head_weights.shape() != Shape{
            query.shape(0),
            query.shape(1),
            kIndexerHeads,
        } ||
        query_offset < 0 ||
        ratio <= 0) {
        throw std::invalid_argument(
            "DSV4 indexer score shape mismatch");
    }
    const int batch = query.shape(0);
    const int queries = query.shape(1);
    const int keys = key.shape(1);
    if (queries == 1 && keys <= 1024) {
        return dsv4_indexer_scores_decode(
            query,
            key,
            head_weights,
            query_offset,
            ratio);
    }
    const int key_tiles = (keys + 63) / 64;
    const int groups = checked_product(
        {batch, queries, key_tiles},
        "indexer group count");
    const int grid = checked_product(
        {groups, 256},
        "indexer grid");
    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(
                kIndexerDimension * kIndexerHeads));
    const array params({scale}, mlx::core::float32);
    auto outputs = indexer_scores_kernel()(
        {
            query,
            key,
            head_weights,
            params,
        },
        {Shape{batch, queries, keys}},
        {mlx::core::float16},
        {grid, 1, 1},
        {256, 1, 1},
        {
            {"B", batch},
            {"M", queries},
            {"K", keys},
            {"KEY_TILES", key_tiles},
            {"QUERY_OFFSET", query_offset},
            {"RATIO", ratio},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array dsv4_topk512(
    const array& scores,
    bool deterministic) {
    auto source = typed_contiguous(
        scores,
        mlx::core::float16);
    if (source.ndim() != 3 ||
        source.shape(0) <= 0 ||
        source.shape(1) <= 0 ||
        source.shape(2) <= 0) {
        throw std::invalid_argument(
            "DSV4 top-k expects f16 [B,M,K]");
    }
    const int batch = source.shape(0);
    const int queries = source.shape(1);
    const int keys = source.shape(2);
    const int rows = checked_product(
        {batch, queries},
        "top-k row count");
    const int grid = checked_product(
        {rows, 1024},
        "top-k grid");
    auto outputs = topk_kernel()(
        {source},
        {Shape{batch, queries, 512}},
        {mlx::core::int32},
        {grid, 1, 1},
        {1024, 1, 1},
        {
            {"ROWS", rows},
            {"K", keys},
            {
                "DETERMINISTIC",
                static_cast<int>(deterministic),
            },
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

std::pair<array, array> dsv4_build_prefill_plan(
    const array& topk,
    int query_offset,
    int local_history,
    int pool_len,
    int ratio,
    int window) {
    auto selected_topk = typed_contiguous(
        topk,
        mlx::core::int32);
    if (selected_topk.ndim() != 3 ||
        selected_topk.shape(0) <= 0 ||
        selected_topk.shape(1) <= 0 ||
        query_offset < 0 ||
        local_history < 0 ||
        pool_len < 0 ||
        ratio <= 0 ||
        window <= 0) {
        throw std::invalid_argument(
            "invalid DSV4 prefill plan input");
    }
    const int batch = selected_topk.shape(0);
    const int queries = selected_topk.shape(1);
    const int topk_count = selected_topk.shape(2);
    if (topk_count >
        std::numeric_limits<int>::max() - window - 31) {
        throw std::invalid_argument(
            "DSV4 prefill plan width exceeds MLX limits");
    }
    const int selected =
        ((window + topk_count + 31) / 32) * 32;
    const int total = checked_product(
        {batch, queries, selected},
        "prefill plan size");
    auto outputs = prefill_plan_kernel()(
        {selected_topk},
        {
            Shape{batch, queries, selected},
            Shape{batch, queries, selected},
        },
        {
            mlx::core::int32,
            mlx::core::float16,
        },
        {total, 1, 1},
        {std::min(256, total), 1, 1},
        {
            {"TOTAL", total},
            {"M", queries},
            {"TOPK_COUNT", topk_count},
            {"SELECTED", selected},
            {"QUERY_OFFSET", query_offset},
            {"LOCAL_HISTORY", local_history},
            {"POOL_LEN", pool_len},
            {"RATIO", ratio},
            {"WINDOW", window},
        },
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs.at(0)),
        std::move(outputs.at(1)),
    };
}

std::pair<array, array> dsv4_build_decode_plan(
    const array& topk,
    const array& seq_len,
    int pool_len,
    int ratio,
    int window) {
    auto selected_topk = typed_contiguous(
        topk,
        mlx::core::int32);
    auto lengths = typed_contiguous(
        seq_len,
        mlx::core::int32);
    if (selected_topk.ndim() != 3 ||
        selected_topk.shape(0) <= 0 ||
        selected_topk.shape(1) != 1 ||
        lengths.size() !=
            static_cast<std::size_t>(
                selected_topk.shape(0)) ||
        pool_len < 0 ||
        ratio <= 0 ||
        window <= 0) {
        throw std::invalid_argument(
            "invalid DSV4 decode plan input");
    }
    const int batch = selected_topk.shape(0);
    const int topk_count = selected_topk.shape(2);
    if (topk_count >
        std::numeric_limits<int>::max() - window - 31) {
        throw std::invalid_argument(
            "DSV4 decode plan width exceeds MLX limits");
    }
    const int selected =
        ((window + topk_count + 31) / 32) * 32;
    const int total = checked_product(
        {batch, selected},
        "decode plan size");
    auto outputs = decode_plan_kernel()(
        {
            selected_topk,
            lengths,
        },
        {
            Shape{batch, 1, selected},
            Shape{batch, 1, selected},
        },
        {
            mlx::core::int32,
            mlx::core::float16,
        },
        {total, 1, 1},
        {std::min(256, total), 1, 1},
        {
            {"TOTAL", total},
            {"TOPK_COUNT", topk_count},
            {"SELECTED", selected},
            {"POOL_LEN", pool_len},
            {"RATIO", ratio},
            {"WINDOW", window},
        },
        std::nullopt,
        false,
        {});
    return {
        std::move(outputs.at(0)),
        std::move(outputs.at(1)),
    };
}

array attention_dsv4_sparse(
    const array& q,
    const array& kv,
    const array& indices,
    const array& mask,
    const array& sinks,
    const std::optional<array>& meta,
    std::optional<float> scale) {
    (void)meta;
    auto query = typed_contiguous(
        q,
        mlx::core::float32);
    auto cache = typed_contiguous(
        kv,
        mlx::core::float16);
    auto selected_indices = typed_contiguous(
        indices,
        mlx::core::int32);
    auto selected_mask = typed_contiguous(
        mask,
        mlx::core::float16);
    auto sink_logits = typed_contiguous(
        sinks,
        mlx::core::float32);
    if (query.ndim() != 4 ||
        query.shape(0) <= 0 ||
        query.shape(1) != kAttentionHeads ||
        query.shape(2) <= 0 ||
        query.shape(3) != kAttentionDimension ||
        cache.ndim() != 3 ||
        cache.shape(0) != query.shape(0) ||
        cache.shape(1) <= 0 ||
        cache.shape(2) != kAttentionDimension ||
        selected_indices.ndim() != 3 ||
        selected_mask.shape() !=
            selected_indices.shape() ||
        selected_indices.shape(0) != query.shape(0) ||
        selected_indices.shape(1) != query.shape(2) ||
        selected_indices.shape(2) <= 0 ||
        selected_indices.shape(2) % 32 != 0 ||
        sink_logits.size() != kAttentionHeads) {
        throw std::invalid_argument(
            "DSV4 sparse attention shape mismatch");
    }
    const float selected_scale = scale.value_or(
        1.0f /
        std::sqrt(
            static_cast<float>(
                kAttentionDimension)));
    if (!std::isfinite(selected_scale) ||
        selected_scale <= 0.0f) {
        throw std::invalid_argument(
            "DSV4 sparse attention scale must be finite and positive");
    }
    const int batch = query.shape(0);
    const int queries = query.shape(2);
    const int max_seq = cache.shape(1);
    const int selected = selected_indices.shape(2);
    const array params(
        {selected_scale},
        mlx::core::float32);
    const Shape output_shape{
        batch,
        queries,
        kAttentionHeads,
        kAttentionDimension,
    };
    const TemplateArgs templates{
        {"B", batch},
        {"M", queries},
        {"MAX_SEQ", max_seq},
        {"SELECTED", selected},
    };
    if (queries >= 32) {
        auto half_query = typed_contiguous(
            q,
            mlx::core::float16);
        const int grid = checked_product(
            {batch, queries, 256},
            "sparse prefill attention grid");
        auto outputs =
            sparse_attention_prefill_mma_kernel()(
                {
                    half_query,
                    cache,
                    selected_indices,
                    selected_mask,
                    sink_logits,
                    params,
                },
                {output_shape},
                {mlx::core::float32},
                {grid, 1, 1},
                {256, 1, 1},
                templates,
                std::nullopt,
                false,
                {});
        return std::move(outputs.front());
    }
    if (queries == 1) {
        const int grid = checked_product(
            {batch, queries, 16, 128},
            "sparse decode attention grid");
        auto outputs =
            sparse_attention_decode_kernel()(
                {
                    query,
                    cache,
                    selected_indices,
                    selected_mask,
                    sink_logits,
                    params,
                },
                {output_shape},
                {mlx::core::float32},
                {grid, 1, 1},
                {128, 1, 1},
                templates,
                std::nullopt,
                false,
                {});
        return std::move(outputs.front());
    }
    const int grid = checked_product(
        {batch, queries, kAttentionHeads, 256},
        "sparse attention grid");
    auto outputs = sparse_attention_kernel()(
        {
            query,
            cache,
            selected_indices,
            selected_mask,
            sink_logits,
            params,
        },
        {output_shape},
        {mlx::core::float32},
        {grid, 1, 1},
        {256, 1, 1},
        templates,
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array attention_dsv4_sparse_decode(
    const array& q,
    const array& local_kv,
    const std::optional<array>& pooled_kv,
    int pool_len,
    const array& topk,
    const array& sinks,
    int seq_len,
    int ratio,
    int window,
    std::optional<float> scale) {
    auto query = typed_contiguous(
        q,
        mlx::core::float32);
    auto local = typed_contiguous(
        local_kv,
        mlx::core::float16);
    auto selected_topk = typed_contiguous(
        topk,
        mlx::core::int32);
    auto sink_logits = typed_contiguous(
        sinks,
        mlx::core::float32);
    auto pool = pooled_kv
        ? typed_contiguous(
              *pooled_kv,
              mlx::core::float16)
        : local;
    if (query.ndim() != 4 ||
        query.shape(0) <= 0 ||
        query.shape(1) != kAttentionHeads ||
        query.shape(2) != 1 ||
        query.shape(3) != kAttentionDimension ||
        local.shape() != Shape{
            query.shape(0),
            window,
            kAttentionDimension,
        } ||
        pool.ndim() != 3 ||
        pool.shape(0) != query.shape(0) ||
        pool.shape(2) != kAttentionDimension ||
        pool_len < 0 ||
        pool_len > pool.shape(1) ||
        selected_topk.ndim() != 3 ||
        selected_topk.shape(0) != query.shape(0) ||
        selected_topk.shape(1) != 1 ||
        sink_logits.size() != kAttentionHeads ||
        seq_len <= 0 ||
        ratio <= 0 ||
        window <= 0) {
        throw std::invalid_argument(
            "DSV4 direct decode attention shape mismatch");
    }
    const float selected_scale = scale.value_or(
        1.0f /
        std::sqrt(
            static_cast<float>(
                kAttentionDimension)));
    if (!std::isfinite(selected_scale) ||
        selected_scale <= 0.0f) {
        throw std::invalid_argument(
            "DSV4 direct decode attention scale must be finite and positive");
    }
    const int batch = query.shape(0);
    const int topk_count = selected_topk.shape(2);
    const int grid = checked_product(
        {batch, 16, 128},
        "direct sparse decode attention grid");
    const array params(
        {selected_scale},
        mlx::core::float32);
    const array decode_params(
        {seq_len, pool_len, topk_count},
        mlx::core::int32);
    auto outputs = sparse_attention_direct_decode_kernel()(
        {
            query,
            local,
            pool,
            selected_topk,
            sink_logits,
            params,
            decode_params,
        },
        {Shape{
            batch,
            1,
            kAttentionHeads,
            kAttentionDimension,
        }},
        {mlx::core::float32},
        {grid, 1, 1},
        {128, 1, 1},
        {
            {"B", batch},
            {"POOL_CAPACITY", pool.shape(1)},
            {"RATIO", ratio},
            {"WINDOW", window},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

} // namespace mfq::metal
