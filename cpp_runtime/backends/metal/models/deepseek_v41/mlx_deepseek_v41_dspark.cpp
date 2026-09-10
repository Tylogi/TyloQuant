#include "mlx_deepseek_v41_dspark.h"

#include "mlx_deepseek_v41_attention.h"
#include "mlx_deepseek_v4_attention.h"
#include "mlx_deepseek_v4_sparse.h"
#include "mlx_sampling.h"
#include "mlx_transformer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace mfq::metal {
namespace {

using mlx::core::Dtype;
using mlx::core::Shape;
using mlx::core::array;

int checked_int(std::int64_t value, const char* label) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            std::string("invalid DeepSeek-V4.1 DSpark ") + label);
    }
    return static_cast<int>(value);
}

int checked_product(int left, int right, const char* label) {
    if (left <= 0 || right <= 0 ||
        left > std::numeric_limits<int>::max() / right) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4.1 DSpark ") + label +
            " exceeds native limits");
    }
    return left * right;
}

array dense_float(const MfqContainer& model, const std::string& name) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" && record.dtype != "F16" &&
        record.dtype != "F32") {
        throw std::runtime_error(
            "DeepSeek-V4.1 DSpark control tensor must be dense: " + name);
    }
    const auto mapped = model.map_record(name);
    return mlx::core::contiguous(mlx::core::astype(
        load_dense_array(record.dtype, mapped.view()),
        mlx::core::float32));
}

array slice_axis(
    const array& input,
    int axis,
    int begin,
    int end) {
    if (axis < 0) axis += static_cast<int>(input.ndim());
    if (axis < 0 || axis >= static_cast<int>(input.ndim()) ||
        begin < 0 || end < begin || end > input.shape(axis)) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 DSpark tensor slice");
    }
    Shape start(input.ndim(), 0);
    Shape stop = input.shape();
    start[axis] = begin;
    stop[axis] = end;
    return mlx::core::slice(input, std::move(start), std::move(stop));
}

array tail_rope(
    const array& input,
    int rotary,
    const array& cosine,
    const array& sine,
    bool inverse = false) {
    const int width = input.shape(-1);
    if (rotary <= 0 || rotary > width) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 DSpark rotary width");
    }
    auto tail = slice_axis(input, -1, width - rotary, width);
    tail = deepseek_v4_rope_adjacent(
        tail, cosine, sine, inverse);
    if (rotary == width) return tail;
    return mlx::core::concatenate(
        {slice_axis(input, -1, 0, width - rotary), std::move(tail)},
        -1);
}

array full_attention(
    const array& query,
    const array& keys,
    const array& sinks) {
    const int heads = query.shape(2);
    const int dimension = query.shape(3);
    if (heads == 64 && dimension == 512) {
        const int batch = query.shape(0);
        const int queries = query.shape(1);
        const int key_count = keys.shape(1);
        const int selected = ((key_count + 31) / 32) * 32;
        auto indices = mlx::core::arange(
            0, key_count, 1, mlx::core::int32);
        if (selected != key_count) {
            indices = mlx::core::concatenate(
                {
                    indices,
                    mlx::core::full(
                        Shape{selected - key_count},
                        -1,
                        mlx::core::int32),
                },
                0);
        }
        indices = mlx::core::broadcast_to(
            mlx::core::reshape(indices, Shape{1, 1, selected}),
            Shape{batch, queries, selected});
        return attention_dsv4_sparse(
            mlx::core::transpose(query, {0, 2, 1, 3}),
            keys,
            indices,
            mlx::core::zeros(
                Shape{batch, queries, selected}, mlx::core::float16),
            sinks);
    }
    auto q = mlx::core::astype(query, mlx::core::float32);
    auto k = mlx::core::astype(keys, mlx::core::float32);
    auto scores = mlx::core::sum(
        mlx::core::expand_dims(q, 3) *
            mlx::core::expand_dims(mlx::core::expand_dims(k, 1), 2),
        -1) /
        std::sqrt(static_cast<float>(dimension));
    auto sink = mlx::core::reshape(
        mlx::core::astype(sinks, mlx::core::float32),
        Shape{1, 1, heads});
    auto maximum = mlx::core::maximum(mlx::core::max(scores, -1), sink);
    auto exponentials = mlx::core::exp(
        scores - mlx::core::expand_dims(maximum, -1));
    auto denominator = mlx::core::sum(exponentials, -1) +
        mlx::core::exp(sink - maximum);
    return mlx::core::sum(
        mlx::core::expand_dims(
            exponentials / mlx::core::expand_dims(denominator, -1), -1) *
            mlx::core::expand_dims(mlx::core::expand_dims(k, 1), 2),
        3);
}

void require_linear(
    const MlxLinear& linear,
    int input,
    int output,
    const char* label) {
    if (linear.input_size() != input || linear.output_size() != output) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4.1 DSpark ") + label +
            " dimensions mismatch");
    }
}

void require_vector(
    const array& value,
    int width,
    const char* label) {
    if (value.ndim() != 1 || value.shape(0) != width) {
        throw std::invalid_argument(
            std::string("DeepSeek-V4.1 DSpark ") + label +
            " dimensions mismatch");
    }
}

struct AttentionComponents {
    MlxLinear query_a;
    MlxLinear query_b;
    MlxLinear key_value;
    MlxLinear output_a;
    MlxLinear output_b;
    array query_norm;
    array key_value_norm;
    array sinks;
};

AttentionComponents load_attention(
    const MfqContainer& model,
    const std::string& prefix) {
    const auto name = [&prefix](std::string_view suffix) {
        return prefix + ".attention." + std::string(suffix);
    };
    return {
        MlxLinear::load(model, name("query_a.weight")),
        MlxLinear::load(model, name("query_b.weight")),
        MlxLinear::load(model, name("key_value.weight")),
        MlxLinear::load(model, name("output_a.weight")),
        MlxLinear::load(model, name("output_b.weight")),
        dense_float(model, name("query_a_norm.weight")),
        dense_float(model, name("key_value_norm.weight")),
        dense_float(model, name("sink")),
    };
}

struct RuntimeStage {
    AttentionComponents attention;
    MlxRmsNorm query_norm;
    MlxRmsNorm key_value_norm;
    MlxDeepseekV41Mhc attention_mhc;
    MlxDeepseekV41Mhc ffn_mhc;
    MlxDeepseekV41Moe moe;

    RuntimeStage(
        AttentionComponents selected_attention,
        MlxDeepseekV41Mhc selected_attention_mhc,
        MlxDeepseekV41Mhc selected_ffn_mhc,
        MlxDeepseekV41Moe selected_moe,
        float eps)
        : attention(std::move(selected_attention)),
          query_norm(attention.query_norm, eps),
          key_value_norm(attention.key_value_norm, eps),
          attention_mhc(std::move(selected_attention_mhc)),
          ffn_mhc(std::move(selected_ffn_mhc)),
          moe(std::move(selected_moe)) {}
};

} // namespace

MlxDeepseekV41DSparkState::MlxDeepseekV41DSparkState(
    std::vector<array> rings,
    int position)
    : rings_(std::move(rings)), position_(position) {}

MlxDeepseekV41DSparkState MlxDeepseekV41DSparkState::allocate(
    int stages,
    int batch,
    int window,
    int head_dim,
    Dtype dtype) {
    if (stages <= 0 || batch <= 0 || window <= 0 || head_dim <= 0 ||
        (dtype != mlx::core::float16 && dtype != mlx::core::bfloat16 &&
         dtype != mlx::core::float32)) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 DSpark state allocation");
    }
    std::vector<array> rings;
    rings.reserve(static_cast<std::size_t>(stages));
    for (int stage = 0; stage < stages; ++stage) {
        rings.push_back(mlx::core::zeros(
            Shape{batch, window, head_dim}, dtype));
    }
    return MlxDeepseekV41DSparkState(std::move(rings), 0);
}

int MlxDeepseekV41DSparkState::batch() const noexcept {
    return rings_.empty() ? 0 : rings_.front().shape(0);
}

int MlxDeepseekV41DSparkState::window() const noexcept {
    return rings_.empty() ? 0 : rings_.front().shape(1);
}

const array& MlxDeepseekV41DSparkState::ring(std::size_t stage) const {
    return rings_.at(stage);
}

struct MlxDeepseekV41DSpark::Impl {
    DeepseekV41Config config;
    MlxEmbedding embedding;
    MlxLinear output;
    MlxLinear main_projection;
    MlxRmsNorm main_norm;
    std::vector<RuntimeStage> stages;
    MlxRmsNorm output_norm;
    MlxEmbedding markov_embedding;
    MlxLinear markov_output;
    MlxLinear confidence;
    int maximum_context;
    std::pair<array, array> rope;

    Impl(
        DeepseekV41Config selected_config,
        MlxEmbedding selected_embedding,
        MlxLinear selected_output,
        MlxLinear selected_main_projection,
        array selected_main_norm,
        std::vector<RuntimeStage> selected_stages,
        array selected_output_norm,
        MlxEmbedding selected_markov_embedding,
        MlxLinear selected_markov_output,
        MlxLinear selected_confidence,
        int selected_maximum_context,
        std::pair<array, array> selected_rope)
        : config(std::move(selected_config)),
          embedding(std::move(selected_embedding)),
          output(std::move(selected_output)),
          main_projection(std::move(selected_main_projection)),
          main_norm(
              std::move(selected_main_norm),
              static_cast<float>(config.rms_eps)),
          stages(std::move(selected_stages)),
          output_norm(
              std::move(selected_output_norm),
              static_cast<float>(config.rms_eps)),
          markov_embedding(std::move(selected_markov_embedding)),
          markov_output(std::move(selected_markov_output)),
          confidence(std::move(selected_confidence)),
          maximum_context(selected_maximum_context),
          rope{
              mlx::core::contiguous(mlx::core::astype(
                  selected_rope.first, mlx::core::float32)),
              mlx::core::contiguous(mlx::core::astype(
                  selected_rope.second, mlx::core::float32)),
          } {
        validate();
    }

    void validate() const {
        config.validate();
        const int hidden = checked_int(config.hidden, "hidden size");
        const int vocab = checked_int(config.vocab, "vocabulary size");
        const int heads = checked_int(config.n_heads, "head count");
        const int head_dim = checked_int(config.head_dim, "head dimension");
        const int q_rank = checked_int(config.q_lora_rank, "query rank");
        const int groups = checked_int(config.o_groups, "output groups");
        const int o_rank = checked_int(config.o_lora_rank, "output rank");
        const int target_width = checked_product(
            hidden,
            checked_int(
                static_cast<std::int64_t>(
                    config.dspark_target_layer_ids.size()),
                "target layer count"),
            "target width");
        if (!config.has_dspark() || maximum_context <= 0 ||
            maximum_context > config.max_position_embeddings ||
            stages.size() != static_cast<std::size_t>(config.n_mtp_layers) ||
            embedding.vocabulary_size() != vocab ||
            embedding.hidden_size() != hidden ||
            output.input_size() != hidden || output.output_size() != vocab ||
            main_projection.input_size() != target_width ||
            main_projection.output_size() != hidden ||
            main_norm.width() != hidden || output_norm.width() != hidden ||
            markov_embedding.vocabulary_size() != vocab ||
            markov_embedding.hidden_size() != config.dspark_markov_rank ||
            markov_output.input_size() != config.dspark_markov_rank ||
            markov_output.output_size() != vocab ||
            confidence.input_size() != hidden + config.dspark_markov_rank ||
            confidence.output_size() != 1 ||
            rope.first.shape() != rope.second.shape() ||
            rope.first.ndim() != 2 || rope.first.shape(0) < maximum_context ||
            rope.first.shape(1) != config.rope_head_dim / 2) {
            throw std::invalid_argument(
                "DeepSeek-V4.1 DSpark top-level geometry mismatch");
        }
        const int attention_width = checked_product(
            heads, head_dim, "attention width");
        for (const auto& stage : stages) {
            require_linear(
                stage.attention.query_a, hidden, q_rank, "query A");
            require_linear(
                stage.attention.query_b,
                q_rank,
                attention_width,
                "query B");
            require_linear(
                stage.attention.key_value,
                hidden,
                head_dim,
                "key/value");
            require_linear(
                stage.attention.output_a,
                attention_width / groups,
                groups * o_rank,
                "output A");
            require_linear(
                stage.attention.output_b,
                groups * o_rank,
                hidden,
                "output B");
            require_vector(stage.attention.query_norm, q_rank, "query norm");
            require_vector(
                stage.attention.key_value_norm,
                head_dim,
                "key/value norm");
            require_vector(stage.attention.sinks, heads, "attention sinks");
        }
    }

    array append_stage_context(
        const array& main_x,
        const RuntimeStage& stage,
        const array& ring,
        int start_position) const {
        const int batch = main_x.shape(0);
        const int tokens = main_x.shape(1);
        const int window = checked_int(config.sliding_window, "window");
        const int rotary = checked_int(config.rope_head_dim, "rotary width");
        auto key_value = stage.key_value_norm(
            stage.attention.key_value(main_x));
        auto positions = mlx::core::arange(
            start_position,
            start_position + tokens,
            1,
            mlx::core::int32);
        auto cosine = mlx::core::take(rope.first, positions, 0);
        auto sine = mlx::core::take(rope.second, positions, 0);
        key_value = tail_rope(key_value, rotary, cosine, sine);
        key_value = deepseek_v41_mxfp8_e4m3_sim(key_value);
        const int retained = std::min(tokens, window);
        if (retained != tokens) {
            key_value = slice_axis(
                key_value, 1, tokens - retained, tokens);
            positions = slice_axis(
                positions, 0, tokens - retained, tokens);
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
            mlx::core::astype(key_value, ring.dtype()),
            rows);
    }

    array attention(
        const array& input,
        const RuntimeStage& stage,
        const array& ring,
        int position) const {
        const int batch = input.shape(0);
        const int tokens = input.shape(1);
        const int heads = checked_int(config.n_heads, "head count");
        const int head_dim = checked_int(config.head_dim, "head dimension");
        const int rotary = checked_int(config.rope_head_dim, "rotary width");
        const int groups = checked_int(config.o_groups, "output groups");
        const int rank = checked_int(config.o_lora_rank, "output rank");
        auto query = mlx::core::reshape(
            stage.attention.query_b(
                stage.query_norm(stage.attention.query_a(input))),
            Shape{batch, tokens, heads, head_dim});
        auto key_value = stage.key_value_norm(
            stage.attention.key_value(input));
        auto positions = mlx::core::arange(
            position,
            position + tokens,
            1,
            mlx::core::int32);
        auto cosine = mlx::core::take(rope.first, positions, 0);
        auto sine = mlx::core::take(rope.second, positions, 0);
        query = tail_rope(query, rotary, cosine, sine);
        key_value = tail_rope(key_value, rotary, cosine, sine);
        key_value = deepseek_v41_mxfp8_e4m3_sim(key_value);
        const int active = std::min(position, ring.shape(1));
        if (active <= 0) {
            throw std::runtime_error(
                "DeepSeek-V4.1 DSpark draft requires committed context");
        }
        auto keys = mlx::core::concatenate(
            {
                slice_axis(ring, 1, 0, active),
                mlx::core::astype(key_value, ring.dtype()),
            },
            1);
        auto attended = full_attention(
            query, keys, stage.attention.sinks);
        attended = tail_rope(
            attended, rotary, cosine, sine, true);
        const int group_input = heads * head_dim / groups;
        auto grouped = mlx::core::reshape(
            attended,
            Shape{batch, tokens, groups, group_input});
        auto low_rank = stage.attention.output_a.grouped_row_matmul(
            grouped, groups);
        return stage.attention.output_b(mlx::core::reshape(
            low_rank,
            Shape{batch, tokens, groups * rank}));
    }

    std::pair<array, array> block(
        const array& hidden,
        const array& previous_pre,
        const RuntimeStage& stage,
        const array& ring,
        int position) const {
        auto residual = hidden;
        auto attention_mix = stage.attention_mhc.collapse(
            residual, previous_pre);
        auto result = stage.attention_mhc.expand(
            attention(attention_mix.branch, stage, ring, position),
            residual,
            attention_mix.expansion);
        residual = result;
        auto ffn_mix = stage.ffn_mhc.collapse(
            residual, attention_mix.next_pre);
        result = stage.ffn_mhc.expand(
            stage.moe.forward(ffn_mix.branch),
            residual,
            ffn_mix.expansion);
        return {std::move(result), std::move(ffn_mix.next_pre)};
    }
};

std::optional<MlxDeepseekV41DSpark>
MlxDeepseekV41DSpark::load_if_present(
    const MfqContainer& model,
    const DeepseekV41Config& config,
    const MlxEmbedding& embedding,
    const MlxLinear& output,
    int max_context) {
    const bool root = model.contains(
        "predictor.stage.0.main_projection.weight");
    const bool any = root ||
        model.contains("predictor.stage.0.attention.query_a.weight") ||
        model.contains("predictor.stage.0.mlp.router.weight") ||
        model.contains("predictor.stage.0.output_norm.weight");
    if (!root) {
        if (any) {
            throw std::runtime_error(
                "DeepSeek-V4.1 MFQ contains an incomplete DSpark head");
        }
        return std::nullopt;
    }
    if (!config.has_dspark()) {
        throw std::runtime_error(
            "DeepSeek-V4.1 MFQ has DSpark tensors without configuration");
    }
    std::vector<RuntimeStage> stages;
    stages.reserve(static_cast<std::size_t>(config.n_mtp_layers));
    for (int stage = 0; stage < config.n_mtp_layers; ++stage) {
        const auto prefix = DeepseekV41TensorNames::predictor(
            static_cast<std::size_t>(stage), "");
        const auto root_prefix = prefix.substr(0, prefix.size() - 1);
        stages.emplace_back(
            load_attention(model, root_prefix),
            MlxDeepseekV41Mhc::load(
                model,
                config,
                root_prefix + ".attention.mhc.pre",
                root_prefix + ".attention.norm.weight"),
            MlxDeepseekV41Mhc::load(
                model,
                config,
                root_prefix + ".mlp.mhc.pre",
                root_prefix + ".mlp.norm.weight"),
            MlxDeepseekV41Moe::load(
                model, config, root_prefix + ".mlp", true),
            static_cast<float>(config.rms_eps));
    }
    const auto first = std::string("predictor.stage.0");
    const auto last = "predictor.stage." +
        std::to_string(config.n_mtp_layers - 1);
    DeepseekV4RopeScaling no_scaling;
    return MlxDeepseekV41DSpark(std::make_shared<Impl>(
        config,
        embedding,
        output,
        MlxLinear::load(model, first + ".main_projection.weight"),
        dense_float(model, first + ".main_norm.weight"),
        std::move(stages),
        dense_float(model, last + ".output_norm.weight"),
        MlxEmbedding::load(model, last + ".markov.embedding.weight"),
        MlxLinear::load(model, last + ".markov.output.weight"),
        MlxLinear(dense_float(
            model, last + ".confidence.projection.weight")),
        max_context,
        deepseek_v4_yarn_tables(
            checked_int(config.rope_head_dim, "rotary width"),
            max_context,
            static_cast<float>(config.rope_theta),
            no_scaling)));
}

MlxDeepseekV41DSpark::MlxDeepseekV41DSpark(
    std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MlxDeepseekV41DSparkState MlxDeepseekV41DSpark::make_state(
    int batch,
    Dtype dtype) const {
    return MlxDeepseekV41DSparkState::allocate(
        static_cast<int>(impl_->stages.size()),
        batch,
        checked_int(impl_->config.sliding_window, "window"),
        checked_int(impl_->config.head_dim, "head dimension"),
        dtype);
}

void MlxDeepseekV41DSpark::append_context(
    const array& target_hidden,
    MlxDeepseekV41DSparkState& state,
    int start_position) const {
    const int hidden = checked_int(impl_->config.hidden, "hidden size");
    const int target_width = checked_product(
        hidden,
        checked_int(
            static_cast<std::int64_t>(
                impl_->config.dspark_target_layer_ids.size()),
            "target count"),
        "target width");
    auto source = target_hidden;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::bfloat16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    if (source.ndim() != 3 || source.shape(0) != state.batch() ||
        source.shape(1) <= 0 || source.shape(2) != target_width ||
        state.stages() != impl_->stages.size() ||
        start_position != state.position_ || start_position < 0 ||
        source.shape(1) > impl_->maximum_context - start_position) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 DSpark context append");
    }
    auto floating = mlx::core::astype(source, mlx::core::float32);
    auto scale = mlx::core::maximum(
        mlx::core::max(mlx::core::abs(floating), -1, true) /
            array(64.0f),
        array(1.0f));
    auto main_x = impl_->main_norm(impl_->main_projection(
        mlx::core::astype(floating / scale, mlx::core::float16)));
    for (std::size_t stage = 0; stage < impl_->stages.size(); ++stage) {
        state.rings_[stage] = impl_->append_stage_context(
            main_x,
            impl_->stages[stage],
            state.rings_[stage],
            start_position);
    }
    state.position_ += source.shape(1);
}

MlxDeepseekV41DSparkDraft MlxDeepseekV41DSpark::draft(
    const array& anchor_ids,
    MlxDeepseekV41DSparkState& state,
    const MlxMtpTokenSelector& select_token,
    int width) const {
    auto anchors = anchor_ids.dtype() == mlx::core::int32
        ? anchor_ids
        : mlx::core::astype(anchor_ids, mlx::core::int32);
    anchors = mlx::core::contiguous(anchors);
    const int requested = width == 0 ? block_size() : width;
    const int physical_width = std::min(
        block_size(), impl_->maximum_context - state.position_);
    const int hidden_size = checked_int(
        impl_->config.hidden, "hidden size");
    if (anchors.ndim() != 2 || anchors.shape(0) != state.batch() ||
        anchors.shape(1) != 1 || requested <= 0 ||
        requested > block_size() || requested > physical_width ||
        state.position_ <= 0 || state.stages() != impl_->stages.size()) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 DSpark draft input");
    }
    auto draft_ids = physical_width == 1
        ? anchors
        : mlx::core::concatenate(
              {
                  anchors,
                  mlx::core::full(
                      Shape{state.batch(), physical_width - 1},
                      static_cast<std::int32_t>(
                          impl_->config.dspark_noise_token_id),
                      mlx::core::int32),
              },
              1);
    auto embedded = impl_->embedding(
        draft_ids, state.rings_.front().dtype());
    auto hidden = mlx::core::contiguous(mlx::core::broadcast_to(
        mlx::core::expand_dims(embedded, 2),
        Shape{state.batch(), physical_width, 4, hidden_size}));
    auto previous_pre = MlxDeepseekV41Mhc::identity_pre(
        state.batch(), physical_width);
    for (std::size_t stage = 0; stage < impl_->stages.size(); ++stage) {
        auto result = impl_->block(
            hidden,
            previous_pre,
            impl_->stages[stage],
            state.rings_[stage],
            state.position_);
        hidden = std::move(result.first);
        previous_pre = std::move(result.second);
    }
    auto head_hidden = mlx::core::sum(
        mlx::core::expand_dims(
            mlx::core::astype(previous_pre, mlx::core::float32), -1) *
            mlx::core::astype(hidden, mlx::core::float32),
        2);
    if (hidden.dtype() != mlx::core::float32) {
        head_hidden = mlx::core::astype(head_hidden, hidden.dtype());
    }
    auto base_logits = mlx::core::astype(
        impl_->output(impl_->output_norm(head_hidden)),
        mlx::core::float32);
    std::vector<array> tokens;
    std::vector<array> logits;
    std::vector<array> markov_embeddings;
    tokens.reserve(static_cast<std::size_t>(requested));
    logits.reserve(static_cast<std::size_t>(requested));
    markov_embeddings.reserve(static_cast<std::size_t>(requested));
    auto previous = anchors;
    for (int position = 0; position < requested; ++position) {
        auto markov = impl_->markov_embedding(
            previous, head_hidden.dtype());
        auto bias = mlx::core::astype(
            impl_->markov_output(markov), mlx::core::float32);
        auto row = slice_axis(
            base_logits, 1, position, position + 1) + bias;
        auto next = select_token(row);
        if (next.ndim() == 1) {
            next = mlx::core::reshape(
                next, Shape{state.batch(), 1});
        }
        tokens.push_back(next);
        logits.push_back(std::move(row));
        markov_embeddings.push_back(std::move(markov));
        previous = tokens.back();
    }
    auto returned_hidden = requested == physical_width
        ? head_hidden
        : slice_axis(head_hidden, 1, 0, requested);
    auto confidence = impl_->confidence(mlx::core::concatenate(
        {
            returned_hidden,
            mlx::core::concatenate(markov_embeddings, 1),
        },
        -1));
    return {
        mlx::core::concatenate(tokens, 1),
        mlx::core::concatenate(logits, 1),
        mlx::core::reshape(confidence, Shape{state.batch(), requested}),
    };
}

MlxDeepseekV41DSparkDraft MlxDeepseekV41DSpark::draft_greedy(
    const array& anchor_ids,
    MlxDeepseekV41DSparkState& state,
    int width) const {
    return draft(
        anchor_ids,
        state,
        [](const array& logits) { return sample_greedy(logits); },
        width);
}

int MlxDeepseekV41DSpark::block_size() const noexcept {
    return static_cast<int>(impl_->config.dspark_block_size);
}

std::size_t MlxDeepseekV41DSpark::stage_count() const noexcept {
    return impl_->stages.size();
}

} // namespace mfq::metal
