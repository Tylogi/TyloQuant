#include "mlx_deepseek_v4_dspark.h"

#include "mlx_deepseek_v4_attention.h"
#include "mlx_deepseek_v4_hc.h"
#include "mlx_deepseek_v4_sparse.h"
#include "mlx_sampling.h"
#include "mlx_transformer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::Dtype;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kConnections = 4;
constexpr int kHcProjectionWidth = 24;

int checked_int(std::int64_t value, const char* label) {
    if (value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            std::string("invalid DeepSeek-V4 DSpark ") + label);
    }
    return static_cast<int>(value);
}

int checked_product(int left, int right, const char* label) {
    if (left <= 0 || right <= 0 ||
        left > std::numeric_limits<int>::max() / right) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4 DSpark ") + label +
            " exceeds MLX limits");
    }
    return left * right;
}

array floating_contiguous(
    const array& input,
    Dtype preferred = mlx::core::float16) {
    auto result = input;
    if (result.dtype() != mlx::core::float16 &&
        result.dtype() != mlx::core::bfloat16 &&
        result.dtype() != mlx::core::float32) {
        result = mlx::core::astype(result, preferred);
    }
    return mlx::core::contiguous(result);
}

array float32_contiguous(const array& input) {
    return mlx::core::contiguous(
        input.dtype() == mlx::core::float32
            ? input
            : mlx::core::astype(input, mlx::core::float32));
}

array load_float(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" && record.dtype != "F16" &&
        record.dtype != "F32") {
        throw std::runtime_error(
            "DeepSeek-V4 DSpark tensor must be BF16/F16/F32: " +
            name);
    }
    const auto mapped = model.map_record(name);
    return float32_contiguous(
        load_dense_array(record.dtype, mapped.view()));
}

array slice_axis(
    const array& input,
    int axis,
    int begin,
    int end) {
    if (axis < 0) axis += static_cast<int>(input.ndim());
    if (axis < 0 || axis >= static_cast<int>(input.ndim()) ||
        begin < 0 || end < begin || end > input.shape(axis)) {
        throw std::invalid_argument("invalid DSpark array slice");
    }
    Shape start(input.ndim(), 0);
    Shape stop = input.shape();
    start[axis] = begin;
    stop[axis] = end;
    return mlx::core::slice(input, std::move(start), std::move(stop));
}

array detached_copy(const array& value) {
    auto result = mlx::core::contiguous(
        value + mlx::core::zeros_like(value));
    result.eval();
    return result;
}

array apply_tail_rope(
    const array& value,
    int rotary,
    const array& cosine,
    const array& sine,
    bool inverse = false) {
    const int width = value.shape(-1);
    if (rotary <= 0 || rotary > width) {
        throw std::invalid_argument("invalid DSpark rotary width");
    }
    auto tail = slice_axis(value, -1, width - rotary, width);
    tail = deepseek_v4_rope_adjacent(
        tail, cosine, sine, inverse);
    if (rotary == width) return tail;
    return mlx::core::concatenate(
        {slice_axis(value, -1, 0, width - rotary), std::move(tail)},
        -1);
}

array full_attention(
    const array& query,
    const array& keys,
    const array& sinks) {
    const int heads = query.shape(2);
    const int dimension = query.shape(3);
    auto q = mlx::core::astype(query, mlx::core::float32);
    auto k = mlx::core::astype(keys, mlx::core::float32);
    auto scores = mlx::core::sum(
        mlx::core::expand_dims(q, 3) *
            mlx::core::expand_dims(
                mlx::core::expand_dims(k, 1), 2),
        -1) /
        std::sqrt(static_cast<double>(dimension));
    auto sink = mlx::core::reshape(
        float32_contiguous(sinks),
        Shape{1, 1, heads});
    auto maximum = mlx::core::maximum(
        mlx::core::max(scores, -1), sink);
    auto exponentials = mlx::core::exp(
        scores - mlx::core::expand_dims(maximum, -1));
    auto denominator =
        mlx::core::sum(exponentials, -1) +
        mlx::core::exp(sink - maximum);
    auto probabilities = exponentials /
        mlx::core::expand_dims(denominator, -1);
    return mlx::core::sum(
        mlx::core::expand_dims(probabilities, -1) *
            mlx::core::expand_dims(
                mlx::core::expand_dims(k, 1), 2),
        3);
}

void require_linear(
    const MlxLinear& value,
    int input,
    int output,
    const char* label) {
    if (value.input_size() != input || value.output_size() != output) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4 DSpark ") + label +
            " dimensions mismatch");
    }
}

void require_vector(
    const array& value,
    int size,
    const char* label) {
    if (value.ndim() != 1 || value.size() != static_cast<std::size_t>(size)) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4 DSpark ") + label +
            " dimensions mismatch");
    }
}

MlxDeepseekV4DSparkAttentionComponents load_attention(
    const MfqContainer& model,
    const std::string& prefix) {
    const auto name = [&prefix](std::string_view suffix) {
        return prefix + ".attn." + std::string(suffix);
    };
    return {
        MlxLinear::load(model, name("wq_a.weight")),
        MlxLinear::load(model, name("wq_b.weight")),
        MlxLinear::load(model, name("wkv.weight")),
        MlxLinear::load(model, name("wo_a.weight")),
        MlxLinear::load(model, name("wo_b.weight")),
        load_float(model, name("q_norm.weight")),
        load_float(model, name("kv_norm.weight")),
        load_float(model, name("attn_sink")),
    };
}

MlxDeepseekV4DSparkAttentionComponents load_attention(
    const MlxHfTensorStore& model,
    const std::string& prefix) {
    const auto name = [&prefix](std::string_view suffix) {
        return prefix + ".attn." + std::string(suffix);
    };
    return {
        model.load_linear(name("wq_a.weight")),
        model.load_linear(name("wq_b.weight")),
        model.load_linear(name("wkv.weight")),
        model.load_linear(name("wo_a.weight")),
        model.load_linear(name("wo_b.weight")),
        float32_contiguous(model.load_dense(name("q_norm.weight"))),
        float32_contiguous(model.load_dense(name("kv_norm.weight"))),
        float32_contiguous(model.load_dense(name("attn_sink"))),
    };
}

MlxDeepseekV4DSparkStageComponents load_stage(
    const MfqContainer& model,
    const DeepseekV4Config& config,
    std::size_t stage,
    std::shared_ptr<MlxNintMoeOffloadCache> expert_offload,
    std::size_t cache_layer) {
    const auto prefix = "mtp." + std::to_string(stage);
    const auto name = [&prefix](std::string_view suffix) {
        return prefix + "." + std::string(suffix);
    };
    return {
        load_attention(model, prefix),
        MlxDeepseekV4Moe::load_named(
            model,
            config,
            prefix,
            std::nullopt,
            std::move(expert_offload),
            cache_layer),
        load_float(model, name("attn_norm.weight")),
        load_float(model, name("ffn_norm.weight")),
        MlxLinear::load(model, name("hc_attn_fn")),
        load_float(model, name("hc_attn_base")),
        load_float(model, name("hc_attn_scale")),
        MlxLinear::load(model, name("hc_ffn_fn")),
        load_float(model, name("hc_ffn_base")),
        load_float(model, name("hc_ffn_scale")),
    };
}

MlxDeepseekV4DSparkStageComponents load_stage(
    const MlxHfTensorStore& model,
    const DeepseekV4Config& config,
    std::size_t stage,
    std::shared_ptr<MlxDeepseekV4SsdExpertCache> expert_cache,
    std::size_t cache_layer) {
    const auto prefix = "mtp." + std::to_string(stage);
    const auto name = [&prefix](std::string_view suffix) {
        return prefix + "." + std::string(suffix);
    };
    return {
        load_attention(model, prefix),
        MlxDeepseekV4Moe::load_named(
            model,
            config,
            prefix,
            std::move(expert_cache),
            cache_layer),
        float32_contiguous(model.load_dense(name("attn_norm.weight"))),
        float32_contiguous(model.load_dense(name("ffn_norm.weight"))),
        model.load_linear(name("hc_attn_fn")),
        float32_contiguous(model.load_dense(name("hc_attn_base"))),
        float32_contiguous(model.load_dense(name("hc_attn_scale"))),
        model.load_linear(name("hc_ffn_fn")),
        float32_contiguous(model.load_dense(name("hc_ffn_base"))),
        float32_contiguous(model.load_dense(name("hc_ffn_scale"))),
    };
}

struct RuntimeStage {
    MlxDeepseekV4DSparkStageComponents components;
    MlxRmsNorm attention_norm;
    MlxRmsNorm ffn_norm;
    MlxRmsNorm q_norm;
    MlxRmsNorm kv_norm;

    RuntimeStage(
        MlxDeepseekV4DSparkStageComponents value,
        float eps)
        : components(std::move(value)),
          attention_norm(components.attention_norm, eps),
          ffn_norm(components.ffn_norm, eps),
          q_norm(components.attention.q_norm, eps),
          kv_norm(components.attention.kv_norm, eps) {}
};

} // namespace

MlxDeepseekV4DSparkState::MlxDeepseekV4DSparkState(
    std::vector<array> rings,
    int position)
    : rings_(std::move(rings)), position_(position) {}

MlxDeepseekV4DSparkState MlxDeepseekV4DSparkState::allocate(
    int stages,
    int batch,
    int window,
    int head_dim,
    Dtype dtype) {
    if (stages <= 0 || batch <= 0 || window <= 0 || head_dim <= 0 ||
        (dtype != mlx::core::float16 && dtype != mlx::core::bfloat16 &&
         dtype != mlx::core::float32)) {
        throw std::invalid_argument("invalid DSpark context allocation");
    }
    std::vector<array> rings;
    rings.reserve(static_cast<std::size_t>(stages));
    for (int stage = 0; stage < stages; ++stage) {
        rings.push_back(mlx::core::zeros(
            Shape{batch, window, head_dim}, dtype));
    }
    return MlxDeepseekV4DSparkState(std::move(rings), 0);
}

int MlxDeepseekV4DSparkState::batch() const noexcept {
    return rings_.empty() ? 0 : rings_.front().shape(0);
}

int MlxDeepseekV4DSparkState::window() const noexcept {
    return rings_.empty() ? 0 : rings_.front().shape(1);
}

const array& MlxDeepseekV4DSparkState::ring(std::size_t stage) const {
    return rings_.at(stage);
}

MlxDeepseekV4DSparkState MlxDeepseekV4DSparkState::snapshot() const {
    std::vector<array> rings;
    rings.reserve(rings_.size());
    for (const auto& ring : rings_) rings.push_back(detached_copy(ring));
    return MlxDeepseekV4DSparkState(std::move(rings), position_);
}

void MlxDeepseekV4DSparkState::restore_snapshot(
    MlxDeepseekV4DSparkState snapshot) {
    if (rings_.size() != snapshot.rings_.size()) {
        throw std::invalid_argument("DSpark snapshot stage mismatch");
    }
    for (std::size_t index = 0; index < rings_.size(); ++index) {
        if (rings_[index].shape() != snapshot.rings_[index].shape() ||
            rings_[index].dtype() != snapshot.rings_[index].dtype()) {
            throw std::invalid_argument("DSpark snapshot geometry mismatch");
        }
    }
    rings_ = std::move(snapshot.rings_);
    position_ = snapshot.position_;
}

struct MlxDeepseekV4DSpark::Impl {
    DeepseekV4Config config;
    MlxEmbedding embedding;
    MlxLinear output;
    MlxLinear main_projection;
    MlxRmsNorm main_norm;
    std::vector<RuntimeStage> stages;
    MlxDeepseekV4DSparkHeadComponents head_components;
    MlxRmsNorm output_norm;
    int maximum_context;
    std::pair<array, array> rope;

    Impl(
        DeepseekV4Config selected_config,
        MlxEmbedding selected_embedding,
        MlxLinear selected_output,
        MlxLinear selected_main_projection,
        array selected_main_norm,
        std::vector<MlxDeepseekV4DSparkStageComponents> selected_stages,
        MlxDeepseekV4DSparkHeadComponents selected_head,
        int max_context,
        std::pair<array, array> selected_rope)
        : config(std::move(selected_config)),
          embedding(std::move(selected_embedding)),
          output(std::move(selected_output)),
          main_projection(std::move(selected_main_projection)),
          main_norm(
              std::move(selected_main_norm),
              static_cast<float>(config.rms_eps)),
          head_components(std::move(selected_head)),
          output_norm(
              head_components.norm,
              static_cast<float>(config.rms_eps)),
          maximum_context(max_context),
          rope{
              float32_contiguous(selected_rope.first),
              float32_contiguous(selected_rope.second),
          } {
        stages.reserve(selected_stages.size());
        for (auto& stage : selected_stages) {
            stages.emplace_back(
                std::move(stage), static_cast<float>(config.rms_eps));
        }
        validate();
    }

    void validate() const {
        config.validate();
        const int hidden = checked_int(config.hidden, "hidden size");
        const int vocab = checked_int(config.vocab, "vocabulary size");
        const int heads = checked_int(config.n_heads, "attention heads");
        const int head_dim = checked_int(config.head_dim, "head dimension");
        const int q_rank = checked_int(config.q_lora_rank, "query rank");
        const int groups = checked_int(config.o_groups, "output groups");
        const int o_rank = checked_int(config.o_lora_rank, "output rank");
        const int target_width = checked_product(
            hidden,
            checked_int(
                static_cast<std::int64_t>(config.dspark_target_layer_ids.size()),
                "target layer count"),
            "target hidden width");
        if (!config.has_dspark() || maximum_context <= 0 ||
            maximum_context > config.max_position_embeddings ||
            stages.size() != static_cast<std::size_t>(config.n_mtp_layers) ||
            config.mtp_compress_ratios.size() != stages.size() ||
            std::any_of(
                config.mtp_compress_ratios.begin(),
                config.mtp_compress_ratios.end(),
                [](std::int64_t value) { return value != 0; }) ||
            embedding.vocabulary_size() != vocab ||
            embedding.hidden_size() != hidden ||
            output.input_size() != hidden || output.output_size() != vocab ||
            main_projection.input_size() != target_width ||
            main_projection.output_size() != hidden ||
            main_norm.width() != hidden ||
            rope.first.shape() != rope.second.shape() ||
            rope.first.ndim() != 2 ||
            rope.first.shape(0) < maximum_context ||
            rope.first.shape(1) != config.qk_rope_head_dim / 2) {
            throw std::invalid_argument(
                "DeepSeek-V4 DSpark top-level dimensions mismatch");
        }
        const int attention = checked_product(heads, head_dim, "attention width");
        const int hc_width = checked_product(kConnections, hidden, "HC width");
        for (const auto& stage : stages) {
            const auto& c = stage.components;
            require_linear(c.attention.q_a, hidden, q_rank, "q_a");
            require_linear(c.attention.q_b, q_rank, attention, "q_b");
            require_linear(c.attention.kv, hidden, head_dim, "wkv");
            require_linear(
                c.attention.wo_a,
                attention / groups,
                groups * o_rank,
                "wo_a");
            require_linear(
                c.attention.wo_b,
                groups * o_rank,
                hidden,
                "wo_b");
            require_vector(c.attention.q_norm, q_rank, "q_norm");
            require_vector(c.attention.kv_norm, head_dim, "kv_norm");
            require_vector(c.attention.sinks, heads, "attention sinks");
            require_vector(c.attention_norm, hidden, "attention norm");
            require_vector(c.ffn_norm, hidden, "FFN norm");
            require_linear(c.hc_attention_fn, hc_width, kHcProjectionWidth, "HC attention");
            require_linear(c.hc_ffn_fn, hc_width, kHcProjectionWidth, "HC FFN");
            require_vector(c.hc_attention_base, kHcProjectionWidth, "HC attention base");
            require_vector(c.hc_attention_scale, 3, "HC attention scale");
            require_vector(c.hc_ffn_base, kHcProjectionWidth, "HC FFN base");
            require_vector(c.hc_ffn_scale, 3, "HC FFN scale");
        }
        require_vector(head_components.norm, hidden, "head norm");
        require_linear(head_components.hc_head_fn, hc_width, kConnections, "head HC");
        require_vector(head_components.hc_head_base, kConnections, "head HC base");
        require_vector(head_components.hc_head_scale, 1, "head HC scale");
        if (head_components.markov_embedding.vocabulary_size() != vocab ||
            head_components.markov_embedding.hidden_size() != config.dspark_markov_rank ||
            head_components.markov_output.input_size() != config.dspark_markov_rank ||
            head_components.markov_output.output_size() != vocab ||
            head_components.confidence.input_size() !=
                hidden + config.dspark_markov_rank ||
            head_components.confidence.output_size() != 1) {
            throw std::invalid_argument(
                "DeepSeek-V4 DSpark head dimensions mismatch");
        }
    }

    MlxDeepseekV4HcPreResult hc_pre_norm(
        const array& residual,
        const MlxLinear& function,
        const array& scale,
        const array& base,
        const MlxRmsNorm& normalizer) const {
        const int batch = residual.shape(0);
        const int tokens = residual.shape(1);
        const int hidden = checked_int(config.hidden, "hidden size");
        auto flat = mlx::core::reshape(
            residual,
            Shape{batch, tokens, kConnections * hidden});
        auto normalized = mlx::core::fast::rms_norm(
            mlx::core::astype(flat, mlx::core::float32),
            std::nullopt,
            static_cast<float>(config.rms_eps));
        auto mixes = mlx::core::astype(
            function(normalized), mlx::core::float32);
        auto result = deepseek_v4_hc_pre(
            residual,
            mixes,
            scale,
            base,
            checked_int(config.hc_sinkhorn_iters, "Sinkhorn iterations"),
            static_cast<float>(config.hc_eps));
        result.reduced = normalizer(result.reduced);
        return result;
    }

    array append_stage_context(
        const array& main_x,
        const RuntimeStage& stage,
        const array& ring,
        int start_position) const {
        const int batch = main_x.shape(0);
        const int tokens = main_x.shape(1);
        const int window = checked_int(config.sliding_window, "window");
        const int rotary = checked_int(config.qk_rope_head_dim, "rotary width");
        auto kv = stage.kv_norm(stage.components.attention.kv(main_x));
        auto positions = mlx::core::arange(
            start_position,
            start_position + tokens,
            1,
            mlx::core::int32);
        auto cosine = mlx::core::take(rope.first, positions, 0);
        auto sine = mlx::core::take(rope.second, positions, 0);
        kv = apply_tail_rope(kv, rotary, cosine, sine);
        kv = deepseek_v4_kv_fp8_sim_prefix(kv, rotary);
        int retained = std::min(tokens, window);
        if (retained != tokens) {
            kv = slice_axis(kv, 1, tokens - retained, tokens);
            positions = slice_axis(positions, 0, tokens - retained, tokens);
        }
        auto rows = mlx::core::broadcast_to(
            mlx::core::reshape(
                mlx::core::remainder(
                    positions,
                    array(window, mlx::core::int32)),
                Shape{1, retained}),
            Shape{batch, retained});
        return dsv4_cache_write_inplace(
            ring,
            mlx::core::astype(kv, ring.dtype()),
            rows);
    }

    array attention(
        const array& input,
        const RuntimeStage& stage,
        const array& ring,
        int position) const {
        const int batch = input.shape(0);
        const int tokens = input.shape(1);
        const int heads = checked_int(config.n_heads, "attention heads");
        const int head_dim = checked_int(config.head_dim, "head dimension");
        const int rotary = checked_int(config.qk_rope_head_dim, "rotary width");
        const int groups = checked_int(config.o_groups, "output groups");
        const int rank = checked_int(config.o_lora_rank, "output rank");
        auto q = mlx::core::reshape(
            stage.components.attention.q_b(
                stage.q_norm(stage.components.attention.q_a(input))),
            Shape{batch, tokens, heads, head_dim});
        q = deepseek_v4_unweighted_rms(
            q, static_cast<float>(config.rms_eps));
        auto kv = stage.kv_norm(stage.components.attention.kv(input));
        auto positions = mlx::core::arange(
            position,
            position + tokens,
            1,
            mlx::core::int32);
        auto cosine = mlx::core::take(rope.first, positions, 0);
        auto sine = mlx::core::take(rope.second, positions, 0);
        q = apply_tail_rope(q, rotary, cosine, sine);
        kv = apply_tail_rope(kv, rotary, cosine, sine);
        kv = deepseek_v4_kv_fp8_sim_prefix(kv, rotary);
        const int active = std::min(position, ring.shape(1));
        if (active <= 0) {
            throw std::runtime_error(
                "DSpark draft requires committed target context");
        }
        auto keys = mlx::core::concatenate(
            {slice_axis(ring, 1, 0, active),
             mlx::core::astype(kv, ring.dtype())},
            1);
        auto attended = full_attention(
            q, keys, stage.components.attention.sinks);
        attended = apply_tail_rope(
            attended, rotary, cosine, sine, true);
        const int group_input = heads * head_dim / groups;
        auto grouped = mlx::core::reshape(
            attended,
            Shape{batch, tokens, groups, group_input});
        auto low_rank = stage.components.attention.wo_a.grouped_row_matmul(
            grouped, groups);
        low_rank = mlx::core::reshape(
            low_rank,
            Shape{batch, tokens, groups * rank});
        return stage.components.attention.wo_b(low_rank);
    }

    array block(
        const array& hidden,
        const array& token_ids,
        const RuntimeStage& stage,
        const array& ring,
        int position) const {
        auto residual = hidden;
        auto attn_hc = hc_pre_norm(
            hidden,
            stage.components.hc_attention_fn,
            stage.components.hc_attention_scale,
            stage.components.hc_attention_base,
            stage.attention_norm);
        auto attended = attention(
            attn_hc.reduced, stage, ring, position);
        auto result = deepseek_v4_hc_post(
            attended,
            residual,
            attn_hc.post,
            attn_hc.combination);

        residual = result;
        auto ffn_hc = hc_pre_norm(
            result,
            stage.components.hc_ffn_fn,
            stage.components.hc_ffn_scale,
            stage.components.hc_ffn_base,
            stage.ffn_norm);
        auto branches = stage.components.moe.forward_branches(
            ffn_hc.reduced, token_ids);
        return deepseek_v4_hc_post_sum(
            branches.routed,
            branches.shared,
            residual,
            ffn_hc.post,
            ffn_hc.combination);
    }

    array head_hidden(const array& hidden) const {
        const int batch = hidden.shape(0);
        const int tokens = hidden.shape(1);
        const int hidden_size = checked_int(config.hidden, "hidden size");
        auto flat = mlx::core::reshape(
            mlx::core::astype(hidden, mlx::core::float32),
            Shape{batch, tokens, kConnections * hidden_size});
        auto normalized = mlx::core::fast::rms_norm(
            flat,
            std::nullopt,
            static_cast<float>(config.rms_eps));
        auto mixes = mlx::core::astype(
            head_components.hc_head_fn(normalized),
            mlx::core::float32);
        auto pre = mlx::core::sigmoid(
            mixes * mlx::core::reshape(
                        head_components.hc_head_scale, Shape{1}) +
            mlx::core::reshape(
                head_components.hc_head_base, Shape{kConnections})) +
            static_cast<float>(config.hc_eps);
        auto reduced = mlx::core::sum(
            mlx::core::expand_dims(pre, -1) *
                mlx::core::astype(hidden, mlx::core::float32),
            2);
        if (hidden.dtype() == mlx::core::float16 ||
            hidden.dtype() == mlx::core::bfloat16) {
            reduced = mlx::core::astype(reduced, hidden.dtype());
        }
        return reduced;
    }
};

std::optional<MlxDeepseekV4DSpark>
MlxDeepseekV4DSpark::load_if_present(
    const MfqContainer& model,
    const DeepseekV4Config& config,
    const MlxEmbedding& embedding,
    const MlxLinear& output,
    int max_context,
    std::shared_ptr<MlxNintMoeOffloadCache> expert_offload,
    std::size_t expert_layer_base) {
    const bool root = model.contains("mtp.0.main_proj.weight");
    bool any = false;
    for (const auto& [name, _record] : model.records()) {
        if (name.starts_with("mtp.")) {
            any = true;
            break;
        }
    }
    if (!root) {
        if (any) {
            throw std::runtime_error(
                "DeepSeek-V4 MFQ contains an incomplete DSpark head");
        }
        return std::nullopt;
    }
    if (!config.has_dspark()) {
        throw std::runtime_error(
            "DeepSeek-V4 MFQ contains DSpark tensors without DSpark config");
    }
    std::vector<MlxDeepseekV4DSparkStageComponents> stages;
    stages.reserve(static_cast<std::size_t>(config.n_mtp_layers));
    for (std::size_t stage = 0;
         stage < static_cast<std::size_t>(config.n_mtp_layers);
         ++stage) {
        stages.push_back(load_stage(
            model,
            config,
            stage,
            expert_offload,
            expert_layer_base + stage));
    }
    const auto first = std::string("mtp.0");
    const auto last = "mtp." + std::to_string(config.n_mtp_layers - 1);
    return MlxDeepseekV4DSpark(
        config,
        embedding,
        output,
        MlxLinear::load(model, first + ".main_proj.weight"),
        load_float(model, first + ".main_norm.weight"),
        std::move(stages),
        {
            load_float(model, last + ".norm.weight"),
            MlxLinear::load(model, last + ".hc_head_fn"),
            load_float(model, last + ".hc_head_base"),
            load_float(model, last + ".hc_head_scale"),
            MlxEmbedding::load(
                model,
                last + ".markov_head.markov_w1.weight"),
            MlxLinear::load(
                model,
                last + ".markov_head.markov_w2.weight"),
            MlxLinear::load(
                model,
                last + ".confidence_head.proj.weight"),
        },
        max_context,
        deepseek_v4_yarn_tables(
            checked_int(config.qk_rope_head_dim, "rotary width"),
            max_context,
            static_cast<float>(config.rope_theta)));
}

MlxDeepseekV4DSpark MlxDeepseekV4DSpark::load_hf(
    const MlxHfTensorStore& model,
    const DeepseekV4Config& config,
    const MlxEmbedding& embedding,
    const MlxLinear& output,
    int max_context,
    std::shared_ptr<MlxDeepseekV4SsdExpertCache> expert_cache,
    std::size_t expert_layer_base) {
    if (!config.has_dspark() || !expert_cache) {
        throw std::invalid_argument(
            "DeepSeek-V4 HF DSpark requires config and SSD expert cache");
    }
    std::vector<MlxDeepseekV4DSparkStageComponents> stages;
    stages.reserve(static_cast<std::size_t>(config.n_mtp_layers));
    for (std::size_t stage = 0;
         stage < static_cast<std::size_t>(config.n_mtp_layers);
         ++stage) {
        stages.push_back(load_stage(
            model,
            config,
            stage,
            expert_cache,
            expert_layer_base + stage));
    }
    const auto first = std::string("mtp.0");
    const auto last = "mtp." + std::to_string(config.n_mtp_layers - 1);
    return MlxDeepseekV4DSpark(
        config,
        embedding,
        output,
        model.load_linear(first + ".main_proj.weight"),
        float32_contiguous(model.load_dense(first + ".main_norm.weight")),
        std::move(stages),
        {
            float32_contiguous(model.load_dense(last + ".norm.weight")),
            model.load_linear(last + ".hc_head_fn"),
            float32_contiguous(model.load_dense(last + ".hc_head_base")),
            float32_contiguous(model.load_dense(last + ".hc_head_scale")),
            model.load_embedding(
                last + ".markov_head.markov_w1.weight"),
            model.load_linear(
                last + ".markov_head.markov_w2.weight"),
            model.load_linear(
                last + ".confidence_head.proj.weight"),
        },
        max_context,
        deepseek_v4_yarn_tables(
            checked_int(config.qk_rope_head_dim, "rotary width"),
            max_context,
            static_cast<float>(config.rope_theta)));
}

MlxDeepseekV4DSpark::MlxDeepseekV4DSpark(
    DeepseekV4Config config,
    MlxEmbedding embedding,
    MlxLinear output,
    MlxLinear main_projection,
    array main_norm,
    std::vector<MlxDeepseekV4DSparkStageComponents> stages,
    MlxDeepseekV4DSparkHeadComponents head,
    int max_context,
    std::pair<array, array> rope)
    : impl_(std::make_shared<Impl>(
          std::move(config),
          std::move(embedding),
          std::move(output),
          std::move(main_projection),
          std::move(main_norm),
          std::move(stages),
          std::move(head),
          max_context,
          std::move(rope))) {}

MlxDeepseekV4DSparkState MlxDeepseekV4DSpark::make_state(
    int batch,
    Dtype dtype) const {
    return MlxDeepseekV4DSparkState::allocate(
        static_cast<int>(impl_->stages.size()),
        batch,
        checked_int(impl_->config.sliding_window, "window"),
        checked_int(impl_->config.head_dim, "head dimension"),
        dtype);
}

void MlxDeepseekV4DSpark::append_context(
    const array& main_hidden,
    MlxDeepseekV4DSparkState& state,
    int start_position) const {
    auto source = floating_contiguous(main_hidden);
    const int target_width = checked_product(
        checked_int(impl_->config.hidden, "hidden size"),
        checked_int(
            static_cast<std::int64_t>(
                impl_->config.dspark_target_layer_ids.size()),
            "target count"),
        "target hidden width");
    if (source.ndim() != 3 || source.shape(0) != state.batch() ||
        source.shape(1) <= 0 || source.shape(2) != target_width ||
        state.stages() != impl_->stages.size() ||
        start_position != state.position_ || start_position < 0 ||
        source.shape(1) > impl_->maximum_context - start_position) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4 DSpark context append");
    }
    auto main_x = impl_->main_norm(impl_->main_projection(source));
    for (std::size_t stage = 0; stage < impl_->stages.size(); ++stage) {
        state.rings_[stage] = impl_->append_stage_context(
            main_x,
            impl_->stages[stage],
            state.rings_[stage],
            start_position);
    }
    state.position_ += source.shape(1);
}

MlxDeepseekV4DSparkDraft MlxDeepseekV4DSpark::draft_greedy(
    const array& anchor_ids,
    MlxDeepseekV4DSparkState& state,
    int width) const {
    auto anchors = anchor_ids;
    if (anchors.dtype() != mlx::core::int32) {
        anchors = mlx::core::astype(anchors, mlx::core::int32);
    }
    anchors = mlx::core::contiguous(anchors);
    const int requested = width == 0 ? block_size() : width;
    const int vocab = checked_int(impl_->config.vocab, "vocabulary size");
    const int hidden_size = checked_int(impl_->config.hidden, "hidden size");
    if (anchors.ndim() != 2 || anchors.shape(0) != state.batch() ||
        anchors.shape(1) != 1 || requested <= 0 ||
        requested > block_size() || state.position_ <= 0 ||
        requested > impl_->maximum_context - state.position_ ||
        state.stages() != impl_->stages.size()) {
        throw std::invalid_argument("invalid DeepSeek-V4 DSpark draft input");
    }
    auto draft_ids = requested == 1
        ? anchors
        : mlx::core::concatenate(
              {anchors,
               mlx::core::full(
                   Shape{state.batch(), requested - 1},
                   static_cast<std::int32_t>(
                       impl_->config.dspark_noise_token_id),
                   mlx::core::int32)},
              1);
    auto embedded = impl_->embedding(draft_ids, state.rings_.front().dtype());
    auto hidden = mlx::core::contiguous(
        mlx::core::broadcast_to(
            mlx::core::expand_dims(embedded, 2),
            Shape{
                state.batch(), requested, kConnections, hidden_size}));
    for (std::size_t stage = 0; stage < impl_->stages.size(); ++stage) {
        hidden = impl_->block(
            hidden,
            draft_ids,
            impl_->stages[stage],
            state.rings_[stage],
            state.position_);
    }
    auto head_hidden = impl_->head_hidden(hidden);
    auto base_logits = mlx::core::astype(
        impl_->output(impl_->output_norm(head_hidden)),
        mlx::core::float32);
    if (base_logits.shape() !=
        Shape{state.batch(), requested, vocab}) {
        throw std::runtime_error("DSpark LM head shape mismatch");
    }

    std::vector<array> tokens;
    std::vector<array> logits;
    std::vector<array> markov_embeddings;
    tokens.reserve(static_cast<std::size_t>(requested));
    logits.reserve(static_cast<std::size_t>(requested));
    markov_embeddings.reserve(static_cast<std::size_t>(requested));
    auto previous = anchors;
    for (int position = 0; position < requested; ++position) {
        auto markov = impl_->head_components.markov_embedding(
            previous, head_hidden.dtype());
        auto bias = mlx::core::astype(
            impl_->head_components.markov_output(markov),
            mlx::core::float32);
        auto row = slice_axis(base_logits, 1, position, position + 1) + bias;
        auto next = sample_greedy(row);
        if (next.ndim() == 1) {
            next = mlx::core::reshape(next, Shape{state.batch(), 1});
        }
        tokens.push_back(next);
        logits.push_back(std::move(row));
        markov_embeddings.push_back(std::move(markov));
        previous = tokens.back();
    }
    auto token_values = mlx::core::concatenate(tokens, 1);
    auto logit_values = mlx::core::concatenate(logits, 1);
    auto markov_values = mlx::core::concatenate(markov_embeddings, 1);
    auto confidence = impl_->head_components.confidence(
        mlx::core::concatenate({head_hidden, markov_values}, -1));
    confidence = mlx::core::reshape(
        confidence,
        Shape{state.batch(), requested});
    return {
        std::move(token_values),
        std::move(logit_values),
        std::move(confidence),
    };
}

int MlxDeepseekV4DSpark::block_size() const noexcept {
    return static_cast<int>(impl_->config.dspark_block_size);
}

std::size_t MlxDeepseekV4DSpark::stage_count() const noexcept {
    return impl_->stages.size();
}

} // namespace mfq::metal
