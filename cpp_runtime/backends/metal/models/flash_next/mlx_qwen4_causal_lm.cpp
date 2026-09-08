#include "mlx_qwen4_causal_lm.h"

#include "mlx_eval_timing.h"
#include "qwen4_ops.h"
#include "mlx_legacy_tensor_compat.h"
#include "mlx_linear_attention.h"
#include "mlx_moe.h"
#include "mlx_moe_ops.h"
#include "mlx_sparse_attention.h"
#include "mlx_tensor.h"
#include "mlx_transformer.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace mfq::metal {
namespace {

using json = nlohmann::json;
using mlx::core::Shape;
using mlx::core::array;

constexpr std::string_view kModelConfigAsset =
    "__mfq_asset__/model_config.json";

const json& text_config(const json& outer) {
    const auto found = outer.find("text_config");
    return found == outer.end() ? outer : *found;
}

std::int64_t positive(const json& value, const char* key) {
    const auto found = value.find(key);
    if (found == value.end() || !found->is_number_integer() ||
        found->get<std::int64_t>() <= 0) {
        throw std::runtime_error(
            std::string("invalid Qwen4 positive integer: ") + key);
    }
    return found->get<std::int64_t>();
}

int checked_int(std::int64_t value, const char* name) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid Qwen4 ") + name);
    }
    return static_cast<int>(value);
}

array dense(
    const MfqContainer& model,
    const std::string& name,
    std::optional<mlx::core::Dtype> dtype = std::nullopt) {
    const auto& record = model.record(name);
    if (record.dtype != "BF16" && record.dtype != "F16" &&
        record.dtype != "F32" && record.dtype != "I32" &&
        record.dtype != "I64") {
        throw std::runtime_error(
            "Qwen4 requires a dense tensor: " + name +
            " (got " + record.dtype + ")");
    }
    const auto mapped = model.map_record(name);
    auto result = load_dense_array(record.dtype, mapped.view());
    if (dtype && result.dtype() != *dtype) {
        result = mlx::core::astype(result, *dtype);
    }
    return mlx::core::contiguous(result);
}

array dense_vector(
    const MfqContainer& model,
    const std::string& name,
    mlx::core::Dtype dtype = mlx::core::float32) {
    auto value = dense(model, name, dtype);
    if (value.ndim() != 1) {
        throw std::runtime_error("Qwen4 dense vector has invalid rank: " + name);
    }
    return value;
}

std::vector<std::int64_t> integer_vector(
    const MfqContainer& model,
    const std::string& name) {
    auto value = dense(model, name);
    if (value.ndim() != 1 ||
        (value.dtype() != mlx::core::int32 &&
         value.dtype() != mlx::core::int64)) {
        throw std::runtime_error("Qwen4 integer metadata is invalid: " + name);
    }
    value = mlx::core::contiguous(value);
    value.eval();
    std::vector<std::int64_t> result(value.size());
    if (value.dtype() == mlx::core::int64) {
        std::copy_n(value.data<std::int64_t>(), value.size(), result.begin());
    } else {
        const auto* source = value.data<std::int32_t>();
        std::transform(
            source, source + value.size(), result.begin(),
            [](std::int32_t item) { return static_cast<std::int64_t>(item); });
    }
    return result;
}

float scalar_float(const MfqContainer& model, const std::string& name) {
    auto value = dense_vector(model, name, mlx::core::float32);
    if (value.size() != 1) {
        throw std::runtime_error("Qwen4 scalar tensor is not scalar: " + name);
    }
    value.eval();
    return value.data<float>()[0];
}

MlxNintMoeWeight moe_weight(
    const MfqContainer& model,
    const std::string& name) {
    if (model.record(name).dtype != "NINTM") {
        throw std::runtime_error(
            "Qwen4 routed expert tensor must use NINTM: " + name);
    }
    const auto mapped = model.map_record(name);
    return MlxNintMoeWeight::from_blob(mapped.view());
}

array last_token_logits(const array& logits, int vocab) {
    if (logits.ndim() != 3 || logits.shape(0) != 1 ||
        logits.shape(1) <= 0 || logits.shape(2) != vocab) {
        throw std::runtime_error("Qwen4 logits must have [1,T,V] shape");
    }
    return mlx::core::reshape(
        mlx::core::slice(
            logits,
            Shape{0, logits.shape(1) - 1, 0},
            Shape{1, logits.shape(1), vocab}),
        Shape{1, vocab});
}

class SequenceCache {
public:
    SequenceCache(int maximum, int width)
        : maximum_(maximum), width_(width) {}

    void reset(int batch, int initial_capacity = 16) {
        batch_ = batch;
        position_ = 0;
        const int capacity = std::min(
            maximum_, std::max(1, initial_capacity));
        values_ = mlx::core::zeros(
            Shape{batch_, capacity, width_}, mlx::core::float16);
    }

    std::pair<array, int> append(const array& value) {
        if (value.ndim() != 3 || value.shape(0) <= 0 ||
            value.shape(2) != width_) {
            throw std::runtime_error("Qwen4 sequence-cache append mismatch");
        }
        if (!values_ || batch_ != value.shape(0)) {
            reset(value.shape(0), std::max(16, value.shape(1)));
        }
        const int start = position_;
        const int end = start + value.shape(1);
        ensure(end);
        *values_ = mlx::core::slice_update(
            *values_,
            value.dtype() == mlx::core::float16
                ? value : mlx::core::astype(value, mlx::core::float16),
            Shape{0, start, 0},
            Shape{batch_, end, width_});
        position_ = end;
        return {
            mlx::core::slice(
                *values_, Shape{0, 0, 0}, Shape{batch_, end, width_}),
            start,
        };
    }

    void clear() noexcept {
        values_.reset();
        batch_ = 0;
        position_ = 0;
    }

    int position() const noexcept { return position_; }

private:
    void ensure(int required) {
        if (!values_) throw std::runtime_error("Qwen4 cache is not initialized");
        if (required <= values_->shape(1)) return;
        if (required > maximum_) {
            throw std::runtime_error("Qwen4 cache exceeds context capacity");
        }
        int capacity = values_->shape(1);
        while (capacity < required) {
            capacity = std::min(maximum_, capacity * 2);
        }
        auto expanded = mlx::core::zeros(
            Shape{batch_, capacity, width_}, values_->dtype());
        expanded = mlx::core::slice_update(
            expanded, *values_,
            Shape{0, 0, 0}, Shape{batch_, values_->shape(1), width_});
        values_ = std::move(expanded);
    }

    int maximum_;
    int width_;
    int batch_ = 0;
    int position_ = 0;
    std::optional<array> values_;
};

class GatedResidual {
public:
    static GatedResidual load(
        const MfqContainer& model,
        const Qwen4Config& config,
        const std::string& prefix,
        bool combine = true) {
        std::optional<array> injection;
        if (combine) {
            auto root = prefix;
            if (root.size() < 4 || root.substr(root.size() - 4) != ".pre") {
                throw std::runtime_error("Qwen4 gated residual prefix is invalid");
            }
            root.resize(root.size() - 4);
            injection = dense(model, root + ".post.inject.weight");
        }
        return GatedResidual(
            config,
            dense_vector(model, prefix + ".norm.weight"),
            dense(model, prefix + ".down.weight"),
            dense(model, prefix + ".up.weight"),
            std::move(injection));
    }

    MlxQwen4GatedResidualPre pre(const array& value) const {
        return qwen4_gated_residual_pre(
            value, norm_, down_, up_, injection_,
            static_cast<int>(config_.hidden_size),
            static_cast<int>(config_.hc_count),
            static_cast<float>(config_.rms_norm_eps));
    }

    array post(const array& branch, const MlxQwen4GatedResidualPre& values) const {
        if (!values.injection) {
            throw std::runtime_error("Qwen4 residual injection is absent");
        }
        return qwen4_gated_residual_post(
            branch, values.residual, *values.injection,
            static_cast<int>(config_.hc_count));
    }

    array mix(const array& value) const {
        auto values = pre(value);
        if (values.injection) {
            throw std::runtime_error("Qwen4 final mixer has an injection branch");
        }
        return std::move(values.branch);
    }

private:
    GatedResidual(
        Qwen4Config config,
        array norm,
        array down,
        array up,
        std::optional<array> injection)
        : config_(std::move(config)),
          norm_(std::move(norm)),
          down_(std::move(down)),
          up_(std::move(up)),
          injection_(std::move(injection)) {}

    Qwen4Config config_;
    array norm_;
    array down_;
    array up_;
    std::optional<array> injection_;
};

class DenseFfn {
public:
    static DenseFfn load(
        const MfqContainer& model,
        const std::string& prefix) {
        return DenseFfn(
            MlxLinear::load(model, prefix + ".gate.weight"),
            MlxLinear::load(model, prefix + ".up.weight"),
            MlxLinear::load(model, prefix + ".down.weight"));
    }

    array operator()(const array& value) const {
        auto gate = gate_(value);
        return down_(gate * mlx::core::sigmoid(gate) * up_(value));
    }

private:
    DenseFfn(MlxLinear gate, MlxLinear up, MlxLinear down)
        : gate_(std::move(gate)), up_(std::move(up)), down_(std::move(down)) {}
    MlxLinear gate_;
    MlxLinear up_;
    MlxLinear down_;
};

class Qwen4Moe {
public:
    static Qwen4Moe load(
        const MfqContainer& model,
        const Qwen4Config& config,
        const std::string& prefix) {
        return Qwen4Moe(
            config,
            moe_weight(model, prefix + ".experts.gate_up.weight"),
            moe_weight(model, prefix + ".experts.down.weight"),
            MlxLinear::load(model, prefix + ".router.weight"),
            DenseFfn::load(model, prefix + ".shared_expert"),
            MlxLinear::load(model, prefix + ".shared_expert.router.weight"));
    }

    array operator()(const array& value) const {
        auto source = mlx::core::reshape(
            value.dtype() == mlx::core::float16
                ? value : mlx::core::astype(value, mlx::core::float16),
            Shape{
                static_cast<int>(value.size() /
                    static_cast<std::size_t>(config_.hidden_size)),
                static_cast<int>(config_.hidden_size)});
        auto router_logits = router_(source);
        if (detail::component_profile_active()) {
            detail::profile_eval("qwen4.moe.router", router_logits);
        }
        auto routes = moe_topk(
            router_logits,
            static_cast<int>(config_.num_experts_per_tok),
            false, false, config_.norm_topk_prob);
        if (detail::component_profile_active()) {
            detail::profile_eval(
                "qwen4.moe.topk",
                std::vector<array>{routes.ids, routes.weights});
        }
        std::optional<array> inverse_route_order;
        bool pairs_are_sorted = false;
        array pairs = [&] {
            const int tokens = source.shape(0);
            const int route_count = checked_int(
                static_cast<std::size_t>(tokens)
                    * static_cast<std::size_t>(routes.ids.shape(1)),
                "Qwen4 routed row count");
            if (tokens >= 32
                && gate_up_.supports_grouped_mmq()
                && down_.supports_grouped_mmq()) {
                // Gate/up and down use the same routing. Keep rows in expert
                // order between both packed MMQs so sorting and the large
                // intermediate permutation happen only once; each projection
                // may still use the block plan best suited to its shape.
                auto route_order = mlx::core::contiguous(
                    mlx::core::astype(
                        mlx::core::argsort(
                            mlx::core::reshape(
                                routes.ids,
                                Shape{route_count})),
                        mlx::core::int32));
                const int gate_block_rows =
                    gate_up_.recommended_grouped_mmq_block_rows(
                        route_count,
                        true);
                const int down_block_rows =
                    down_.recommended_grouped_mmq_block_rows(
                        route_count,
                        false);
                auto gate_plan = gate_up_.build_grouped_mmq_plan(
                    routes.ids,
                    route_order,
                    gate_block_rows);
                std::optional<mfq::metal::MlxGroupedMmqPlan> down_plan;
                if (down_block_rows != gate_block_rows) {
                    down_plan.emplace(
                        down_.build_grouped_mmq_plan(
                            routes.ids,
                            route_order,
                            down_block_rows));
                }
                auto intermediate = gate_up_.routed_matmul_sorted(
                    source,
                    routes.ids,
                    route_order,
                    false,
                    true,
                    0.0f,
                    &gate_plan);
                if (detail::component_profile_active()) {
                    detail::profile_eval(
                        "qwen4.moe.routed_gate_up",
                        intermediate);
                }
                auto sorted = down_.routed_matmul_sorted(
                    intermediate,
                    routes.ids,
                    route_order,
                    true,
                    false,
                    0.0f,
                    down_plan.has_value() ? &*down_plan : &gate_plan);
                if (detail::component_profile_active()) {
                    detail::profile_eval(
                        "qwen4.moe.routed_down",
                        sorted);
                }
                inverse_route_order = mlx::core::contiguous(
                    mlx::core::astype(
                        mlx::core::argsort(route_order),
                        mlx::core::int32));
                pairs_are_sorted = true;
                return sorted;
            }
            auto intermediate = gate_up_.routed_swiglu(
                source,
                routes.ids);
            if (detail::component_profile_active()) {
                detail::profile_eval(
                    "qwen4.moe.routed_gate_up",
                    intermediate);
            }
            auto output = down_.routed_matmul(intermediate, routes.ids);
            if (detail::component_profile_active()) {
                detail::profile_eval("qwen4.moe.routed_down", output);
            }
            return output;
        }();
        auto shared = shared_(source);
        if (detail::component_profile_active()) {
            detail::profile_eval("qwen4.moe.shared", shared);
        }
        auto shared_gate = shared_gate_(source);
        if (detail::component_profile_active()) {
            detail::profile_eval("qwen4.moe.shared_gate", shared_gate);
        }
        auto output = pairs_are_sorted
            ? moe_weighted_reduce_shared_gate_sorted(
                  pairs,
                  *inverse_route_order,
                  routes.weights,
                  shared,
                  shared_gate)
            : moe_weighted_reduce_shared_gate(
                  pairs,
                  routes.weights,
                  shared,
                  shared_gate);
        if (detail::component_profile_active()) {
            detail::profile_eval("qwen4.moe.reduce", output);
        }
        return mlx::core::reshape(output, value.shape());
    }

private:
    Qwen4Moe(
        Qwen4Config config,
        MlxNintMoeWeight gate_up,
        MlxNintMoeWeight down,
        MlxLinear router,
        DenseFfn shared,
        MlxLinear shared_gate)
        : config_(std::move(config)),
          gate_up_(std::move(gate_up)),
          down_(std::move(down)),
          router_(std::move(router)),
          shared_(std::move(shared)),
          shared_gate_(std::move(shared_gate)) {
        if (gate_up_.experts() != config_.num_experts ||
            down_.experts() != config_.num_experts ||
            gate_up_.neuron_len() != config_.hidden_size ||
            gate_up_.out_per_expert() != 2 * config_.moe_intermediate_size ||
            down_.neuron_len() != config_.moe_intermediate_size ||
            down_.out_per_expert() != config_.hidden_size) {
            throw std::runtime_error("Qwen4 routed expert geometry disagrees");
        }
    }

    Qwen4Config config_;
    MlxNintMoeWeight gate_up_;
    MlxNintMoeWeight down_;
    MlxLinear router_;
    DenseFfn shared_;
    MlxLinear shared_gate_;
};

class Qwen4NgramEmbedding {
private:
    struct Shard {
        MfqMappedBytes mapping;
        const std::uint8_t* values = nullptr;
        std::int64_t rows = 0;
        std::int64_t width = 0;
    };

public:
    static Qwen4NgramEmbedding load(
        const MfqContainer& model,
        const Qwen4Config& config,
        const std::string& prefix) {
        std::vector<Shard> shards;
        shards.reserve(static_cast<std::size_t>(config.split_ngram_parts));
        std::int64_t rows = 0;
        std::int64_t width = 0;
        for (std::int64_t index = 0;
             index < config.split_ngram_parts; ++index) {
            const auto name = prefix + ".ngram.shard." +
                std::to_string(index) + ".weight";
            const auto& record = model.record(name);
            if (record.dtype != "F8_E4M3") {
                throw std::runtime_error(
                    "Qwen4 PLE row streaming currently requires F8_E4M3: " + name);
            }
            auto mapping = model.map_record(name);
            const auto bytes = mapping.view();
            if (bytes.size() < 20) {
                throw std::runtime_error("truncated Qwen4 PLE shard: " + name);
            }
            std::uint32_t dimensions = 0;
            std::int64_t shard_rows = 0;
            std::int64_t shard_width = 0;
            std::memcpy(&dimensions, bytes.data(), sizeof(dimensions));
            std::memcpy(&shard_rows, bytes.data() + 4, sizeof(shard_rows));
            std::memcpy(&shard_width, bytes.data() + 12, sizeof(shard_width));
            if (dimensions != 2 || shard_rows <= 0 || shard_width <= 0 ||
                static_cast<std::uint64_t>(shard_rows) >
                    (std::numeric_limits<std::uint64_t>::max() - 20) /
                        static_cast<std::uint64_t>(shard_width) ||
                bytes.size() != static_cast<std::size_t>(
                    20 + shard_rows * shard_width)) {
                throw std::runtime_error("invalid Qwen4 PLE shard geometry: " + name);
            }
            if ((!shards.empty() && (shard_rows != rows || shard_width != width))) {
                throw std::runtime_error("Qwen4 PLE shards have different shapes");
            }
            rows = shard_rows;
            width = shard_width;
            const auto* values = mapping.data() + 20;
            shards.push_back({std::move(mapping), values, rows, width});
        }
        const auto metadata = prefix + ".ngram";
        return Qwen4NgramEmbedding(
            config,
            std::move(shards),
            rows,
            width,
            scalar_float(model, metadata + ".weight_scale"),
            integer_vector(model, metadata + ".layer_multipliers"),
            integer_vector(model, metadata + ".head_offsets"),
            integer_vector(model, metadata + ".head_vocab_sizes"));
    }

    void reset(int batch) {
        batch_ = batch;
        context_.assign(
            static_cast<std::size_t>(batch * (config_.ngram_size - 1)),
            config_.eos_token_id);
    }

    array forward(const array& token_ids, bool use_cache) {
        if (token_ids.ndim() != 2 || token_ids.shape(0) <= 0 ||
            token_ids.shape(1) <= 0) {
            throw std::runtime_error("Qwen4 PLE IDs must have [B,T] shape");
        }
        const int batch = token_ids.shape(0);
        const int tokens = token_ids.shape(1);
        const bool int32_ids = token_ids.dtype() == mlx::core::int32;
        auto host = int32_ids || token_ids.dtype() == mlx::core::int64
            ? mlx::core::contiguous(token_ids)
            : mlx::core::contiguous(
                mlx::core::astype(token_ids, mlx::core::int64));
        host.eval();
        const auto* source32 = int32_ids
            ? host.data<std::int32_t>() : nullptr;
        const auto* source64 = int32_ids
            ? nullptr : host.data<std::int64_t>();
        const int prefix = static_cast<int>(config_.ngram_size - 1);
        const int length = prefix + tokens;
        const int heads = static_cast<int>(
            (config_.ngram_size - 1) * config_.heads_per_ngram);
        if (use_cache && (batch_ != batch || context_.empty())) reset(batch);
        std::vector<std::int64_t> history(
            static_cast<std::size_t>(batch * length), config_.eos_token_id);
        std::vector<std::int64_t> global(
            static_cast<std::size_t>(batch * tokens * heads));
        for (int bi = 0; bi < batch; ++bi) {
            if (use_cache) {
                std::copy_n(
                    context_.data() + static_cast<std::size_t>(bi * prefix),
                    prefix,
                    history.data() + static_cast<std::size_t>(bi * length));
            }
            for (int token = 0; token < tokens; ++token) {
                const auto source_index =
                    static_cast<std::size_t>(bi * tokens + token);
                history[static_cast<std::size_t>(
                    bi * length + prefix + token)] = int32_ids
                    ? static_cast<std::int64_t>(source32[source_index])
                    : source64[source_index];
            }
            int segment = 0;
            for (int token = 0; token < length; ++token) {
                std::uint64_t mixed =
                    static_cast<std::uint64_t>(history[bi * length + token]) *
                    static_cast<std::uint64_t>(multipliers_[0]);
                for (int shift = 1; shift < config_.ngram_size; ++shift) {
                    const auto previous = token - shift >= segment
                        ? history[bi * length + token - shift]
                        : config_.eos_token_id;
                    mixed ^= static_cast<std::uint64_t>(previous) *
                        static_cast<std::uint64_t>(multipliers_[shift]);
                    if (token >= prefix) {
                        std::int64_t signed_hash = 0;
                        std::memcpy(&signed_hash, &mixed, sizeof(mixed));
                        for (int head = 0; head < config_.heads_per_ngram; ++head) {
                            const int h = (shift - 1) * config_.heads_per_ngram + head;
                            auto remainder = signed_hash % vocab_[h];
                            if (remainder < 0) remainder += vocab_[h];
                            global[(bi * tokens + token - prefix) * heads + h] =
                                remainder + offsets_[h];
                        }
                    }
                }
                if (history[bi * length + token] == config_.eos_token_id) {
                    segment = token + 1;
                }
            }
        }
        if (use_cache) {
            for (int bi = 0; bi < batch; ++bi) {
                std::copy_n(
                    history.data() + static_cast<std::size_t>(bi * length + tokens),
                    prefix,
                    context_.data() + static_cast<std::size_t>(bi * prefix));
            }
        }
        std::vector<mlx::core::float16_t> result(
            static_cast<std::size_t>(batch * tokens * heads * width_));
        for (std::size_t index = 0; index < global.size(); ++index) {
            const auto row = global[index];
            if (row < 0 || row >= rows_ * static_cast<std::int64_t>(shards_.size())) {
                throw std::runtime_error("Qwen4 PLE hash is outside embedding table");
            }
            const auto shard = static_cast<std::size_t>(row / rows_);
            const auto local = row % rows_;
            const auto* values = shards_[shard].values + local * width_;
            auto* destination = result.data() + index * static_cast<std::size_t>(width_);
            for (std::int64_t column = 0; column < width_; ++column) {
                destination[column] = e4m3_lut_[values[column]];
            }
        }
        auto output = array(
            result.begin(),
            Shape{batch, tokens, heads * static_cast<int>(width_)});
        return output;
    }

private:
    static float decode_e4m3(std::uint8_t raw) {
        const float sign = (raw & 0x80u) ? -1.0f : 1.0f;
        const int exponent = (raw >> 3u) & 0x0fu;
        const int mantissa = raw & 0x07u;
        if ((raw & 0x7fu) == 0x7fu) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return sign * (exponent == 0
            ? std::ldexp(static_cast<float>(mantissa), -9)
            : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                         exponent - 7));
    }

    static std::array<mlx::core::float16_t, 256>
    scaled_e4m3_lut(float scale) {
        std::array<mlx::core::float16_t, 256> result{};
        for (std::size_t raw = 0; raw < result.size(); ++raw) {
            result[raw] = static_cast<mlx::core::float16_t>(
                decode_e4m3(static_cast<std::uint8_t>(raw)) * scale);
        }
        return result;
    }

    Qwen4NgramEmbedding(
        Qwen4Config config,
        std::vector<Shard> shards,
        std::int64_t rows,
        std::int64_t width,
        float weight_scale,
        std::vector<std::int64_t> multipliers,
        std::vector<std::int64_t> offsets,
        std::vector<std::int64_t> vocab)
        : config_(std::move(config)),
          shards_(std::move(shards)),
          rows_(rows),
          width_(width),
          e4m3_lut_(scaled_e4m3_lut(weight_scale)),
          multipliers_(std::move(multipliers)),
          offsets_(std::move(offsets)),
          vocab_(std::move(vocab)) {
        const auto heads = static_cast<std::size_t>(
            (config_.ngram_size - 1) * config_.heads_per_ngram);
        if (shards_.empty() || width_ <= 0 || rows_ <= 0 ||
            multipliers_.size() != static_cast<std::size_t>(config_.ngram_size) ||
            offsets_.size() != heads || vocab_.size() != heads ||
            width_ * static_cast<std::int64_t>(heads) != config_.hidden_size) {
            throw std::runtime_error("Qwen4 PLE embedding metadata disagrees");
        }
    }

    Qwen4Config config_;
    std::vector<Shard> shards_;
    std::int64_t rows_;
    std::int64_t width_;
    std::array<mlx::core::float16_t, 256> e4m3_lut_;
    std::vector<std::int64_t> multipliers_;
    std::vector<std::int64_t> offsets_;
    std::vector<std::int64_t> vocab_;
    int batch_ = 0;
    std::vector<std::int64_t> context_;
};

class Qwen4Ple {
public:
    static Qwen4Ple load(
        const MfqContainer& model,
        const Qwen4Config& config,
        const std::string& prefix) {
        return Qwen4Ple(
            config,
            Qwen4NgramEmbedding::load(model, config, prefix),
            MlxLinear::load(model, prefix + ".key.weight"),
            MlxLinear::load(model, prefix + ".value.weight"),
            dense_vector(model, prefix + ".key_norm.weight"),
            dense_vector(model, prefix + ".query_norm.weight"),
            dense_vector(model, prefix + ".conv_norm.weight"),
            dense(model, prefix + ".conv.weight"));
    }

    void reset(int batch) {
        embedding_.reset(batch);
        convolution_state_.reset();
        batch_ = batch;
    }

    void clear() noexcept {
        convolution_state_.reset();
        batch_ = 0;
    }

    array forward(
        const array& hidden_streams,
        const array& token_ids,
        bool use_cache) {
        const int batch = token_ids.shape(0);
        const int tokens = token_ids.shape(1);
        if (hidden_streams.ndim() != 3 || token_ids.ndim() != 2 ||
            hidden_streams.shape(0) != batch ||
            hidden_streams.shape(1) != tokens ||
            hidden_streams.shape(2) !=
                config_.hidden_size * config_.hc_count) {
            throw std::runtime_error("Qwen4 PLE input geometry disagrees");
        }
        if (use_cache && batch_ != batch) reset(batch);
        auto embeddings = embedding_.forward(token_ids, use_cache);
        const Shape stream_shape{
            batch,
            tokens,
            static_cast<int>(config_.hc_count),
            static_cast<int>(config_.hidden_size),
        };
        auto key = mlx::core::reshape(
            qwen4_grouped_rms_norm(
                key_(embeddings), norm_key_,
                static_cast<int>(config_.hidden_size),
                static_cast<float>(config_.rms_norm_eps)),
            stream_shape);
        auto query = mlx::core::reshape(
            qwen4_grouped_rms_norm(
                hidden_streams, norm_query_,
                static_cast<int>(config_.hidden_size),
                static_cast<float>(config_.rms_norm_eps)),
            stream_shape);
        auto score = mlx::core::sum(
            mlx::core::astype(key, mlx::core::float32) *
                mlx::core::astype(query, mlx::core::float32),
            -1) /
            std::sqrt(static_cast<float>(config_.hidden_size));
        auto sign = mlx::core::astype(
            mlx::core::less(array(0.0f), score), mlx::core::float32) -
            mlx::core::astype(
                mlx::core::less(score, array(0.0f)), mlx::core::float32);
        auto root = sign * mlx::core::sqrt(
            mlx::core::maximum(mlx::core::abs(score), array(1e-6f)));
        auto gated = mlx::core::reshape(
            mlx::core::expand_dims(mlx::core::sigmoid(root), -1) *
                mlx::core::expand_dims(
                    mlx::core::astype(value_(embeddings), mlx::core::float32),
                    -2),
            Shape{
                batch,
                tokens,
                static_cast<int>(config_.hc_count * config_.hidden_size),
            });
        auto normalized = qwen4_grouped_rms_norm(
            gated, norm_conv_, static_cast<int>(config_.hidden_size),
            static_cast<float>(config_.rms_norm_eps));
        auto convolution = cached_depthwise_conv_silu(
            normalized,
            convolution_weight_,
            use_cache ? convolution_state_ : std::nullopt,
            static_cast<int>(config_.ngram_size));
        if (use_cache) convolution_state_ = convolution.state;
        auto output = gated + convolution.output;
        return output.dtype() == hidden_streams.dtype()
            ? output : mlx::core::astype(output, hidden_streams.dtype());
    }

private:
    Qwen4Ple(
        Qwen4Config config,
        Qwen4NgramEmbedding embedding,
        MlxLinear key,
        MlxLinear value,
        array norm_key,
        array norm_query,
        array norm_conv,
        array convolution_weight)
        : config_(std::move(config)),
          embedding_(std::move(embedding)),
          key_(std::move(key)),
          value_(std::move(value)),
          norm_key_(std::move(norm_key)),
          norm_query_(std::move(norm_query)),
          norm_conv_(std::move(norm_conv)),
          convolution_weight_(std::move(convolution_weight)) {}

    Qwen4Config config_;
    Qwen4NgramEmbedding embedding_;
    MlxLinear key_;
    MlxLinear value_;
    array norm_key_;
    array norm_query_;
    array norm_conv_;
    array convolution_weight_;
    std::optional<array> convolution_state_;
    int batch_ = 0;
};

class Qwen4Attention {
public:
    virtual ~Qwen4Attention() = default;
    virtual std::string_view profile_name() const noexcept = 0;
    virtual array forward(
        const array& hidden,
        const array& positions_current,
        const array& positions_full,
        bool use_cache) = 0;
    virtual void reset(int batch) = 0;
    virtual void clear() noexcept = 0;
};

class Qwen4Gdn final : public Qwen4Attention {
public:
    std::string_view profile_name() const noexcept override {
        return "qwen4.linear_attention";
    }

    static std::unique_ptr<Qwen4Gdn> load(
        const MfqContainer& model,
        const Qwen4Config& config,
        const std::string& prefix) {
        return std::unique_ptr<Qwen4Gdn>(new Qwen4Gdn(
            config,
            MlxLinear::load(model, prefix + ".qkv.weight"),
            MlxLinear::load(model, prefix + ".gate.weight"),
            MlxLinear::load(model, prefix + ".alpha.weight"),
            MlxLinear::load(model, prefix + ".beta.weight"),
            dense(model, prefix + ".conv.weight"),
            dense_vector(model, prefix + ".dt_bias"),
            dense_vector(model, prefix + ".a"),
            MlxRmsNorm(
                dense_vector(model, prefix + ".norm.weight"),
                static_cast<float>(config.rms_norm_eps)),
            MlxLinear::load(model, prefix + ".output.weight")));
    }

    void reset(int batch) override {
        const int channels = 2 * key_width() + value_width();
        convolution_state_ = mlx::core::zeros(
            Shape{
                batch,
                static_cast<int>(config_.linear_conv_kernel_dim - 1),
                channels,
            },
            mlx::core::float32);
        recurrent_state_ = mlx::core::zeros(
            Shape{
                batch,
                static_cast<int>(config_.linear_num_value_heads),
                static_cast<int>(config_.linear_value_head_dim),
                static_cast<int>(config_.linear_value_head_dim),
            },
            mlx::core::float32);
        batch_ = batch;
        position_ = 0;
    }

    void clear() noexcept override {
        convolution_state_.reset();
        recurrent_state_.reset();
        batch_ = 0;
        position_ = 0;
    }

    array forward(
        const array& hidden,
        const array&,
        const array&,
        bool use_cache) override {
        if (hidden.ndim() != 3 || hidden.shape(0) <= 0 ||
            hidden.shape(1) <= 0 || hidden.shape(2) != config_.hidden_size) {
            throw std::runtime_error("Qwen4 GDN input geometry disagrees");
        }
        const int batch = hidden.shape(0);
        const int tokens = hidden.shape(1);
        if (use_cache && (batch_ != batch || !convolution_state_)) reset(batch);
        auto projected = qkv_(hidden);
        detail::profile_eval("qwen4.gdn.qkv", projected);
        auto qkv_parts = mlx::core::split(
            projected, Shape{2 * key_width()}, -1);
        auto qk = std::move(qkv_parts.at(0));
        auto value = std::move(qkv_parts.at(1));
        auto z = gate_(hidden);
        detail::profile_eval("qwen4.gdn.gate", z);
        auto beta = mlx::core::reshape(
            mlx::core::sigmoid(
                mlx::core::astype(beta_(hidden), mlx::core::float32)),
            Shape{
                batch,
                tokens,
                static_cast<int>(config_.linear_num_value_heads),
            });
        detail::profile_eval("qwen4.gdn.beta", beta);
        auto alpha = mlx::core::reshape(
            mlx::core::astype(alpha_(hidden), mlx::core::float32),
            beta.shape());
        detail::profile_eval("qwen4.gdn.alpha", alpha);
        auto gate_input = alpha + mlx::core::reshape(
            dt_bias_,
            Shape{1, 1, static_cast<int>(config_.linear_num_value_heads)});
        auto softplus = mlx::core::maximum(gate_input, array(0.0f)) +
            mlx::core::log1p(mlx::core::exp(-mlx::core::abs(gate_input)));
        auto decay = -mlx::core::exp(a_log_) * array(1.0f);
        decay = mlx::core::reshape(
            decay,
            Shape{1, 1, static_cast<int>(config_.linear_num_value_heads)}) *
            softplus;
        auto convolved = linear_conv_qkv(
            use_cache
                ? *convolution_state_
                : mlx::core::zeros(
                      Shape{
                          batch,
                          static_cast<int>(
                              config_.linear_conv_kernel_dim - 1),
                          2 * key_width() + value_width(),
                      },
                      mlx::core::float32),
            qk,
            value,
            convolution_weight_,
            static_cast<int>(config_.linear_num_key_heads),
            static_cast<int>(config_.linear_num_value_heads),
            static_cast<int>(config_.linear_key_head_dim),
            static_cast<int>(config_.linear_value_head_dim),
            std::nullopt,
            1e-6f);
        detail::profile_eval(
            "qwen4.gdn.conv",
            {
                convolved.query,
                convolved.key,
                convolved.value,
                convolved.state,
            });
        auto recurrent = gated_delta_net(
            convolved.query,
            convolved.key,
            convolved.value,
            mlx::core::transpose(decay, {0, 2, 1}),
            mlx::core::transpose(beta, {0, 2, 1}),
            use_cache ? recurrent_state_ : std::nullopt);
        detail::profile_eval(
            "qwen4.gdn.recurrent",
            {recurrent.output, recurrent.state});
        if (use_cache) {
            convolution_state_ = convolved.state;
            recurrent_state_ = recurrent.state;
            position_ += tokens;
        }
        auto normalized = output_norm_(recurrent.output);
        auto output_gate = mlx::core::astype(
            mlx::core::transpose(
                mlx::core::reshape(
                    z,
                    Shape{
                        batch,
                        tokens,
                        static_cast<int>(config_.linear_num_value_heads),
                        static_cast<int>(config_.linear_value_head_dim),
                    }),
                {0, 2, 1, 3}),
            mlx::core::float32);
        output_gate = config_.output_gate_silu
            ? output_gate * mlx::core::sigmoid(output_gate)
            : mlx::core::sigmoid(output_gate);
        normalized = normalized * output_gate;
        normalized = mlx::core::reshape(
            mlx::core::transpose(normalized, {0, 2, 1, 3}),
            Shape{batch, tokens, value_width()});
        detail::profile_eval("qwen4.gdn.output_gate", normalized);
        auto output = output_(mlx::core::astype(normalized, hidden.dtype()));
        detail::profile_eval("qwen4.gdn.output", output);
        return output;
    }

private:
    Qwen4Gdn(
        Qwen4Config config,
        MlxLinear qkv,
        MlxLinear gate,
        MlxLinear alpha,
        MlxLinear beta,
        array convolution_weight,
        array dt_bias,
        array a_log,
        MlxRmsNorm output_norm,
        MlxLinear output)
        : config_(std::move(config)),
          qkv_(std::move(qkv)),
          gate_(std::move(gate)),
          alpha_(std::move(alpha)),
          beta_(std::move(beta)),
          convolution_weight_(std::move(convolution_weight)),
          dt_bias_(std::move(dt_bias)),
          a_log_(std::move(a_log)),
          output_norm_(std::move(output_norm)),
          output_(std::move(output)) {}

    int key_width() const {
        return static_cast<int>(
            config_.linear_num_key_heads * config_.linear_key_head_dim);
    }
    int value_width() const {
        return static_cast<int>(
            config_.linear_num_value_heads * config_.linear_value_head_dim);
    }

    Qwen4Config config_;
    MlxLinear qkv_;
    MlxLinear gate_;
    MlxLinear alpha_;
    MlxLinear beta_;
    array convolution_weight_;
    array dt_bias_;
    array a_log_;
    MlxRmsNorm output_norm_;
    MlxLinear output_;
    std::optional<array> convolution_state_;
    std::optional<array> recurrent_state_;
    int batch_ = 0;
    int position_ = 0;
};

class Qwen4Qsa final : public Qwen4Attention {
public:
    std::string_view profile_name() const noexcept override {
        return "qwen4.full_attention";
    }

    static std::unique_ptr<Qwen4Qsa> load(
        const MfqContainer& model,
        const Qwen4Config& config,
        const std::string& prefix,
        int maximum) {
        return std::unique_ptr<Qwen4Qsa>(new Qwen4Qsa(
            config,
            maximum,
            MlxLinear::load(model, prefix + ".query.weight"),
            MlxLinear::load(model, prefix + ".key.weight"),
            MlxLinear::load(model, prefix + ".value.weight"),
            MlxRmsNorm(
                dense_vector(model, prefix + ".query_norm.weight"),
                static_cast<float>(config.rms_norm_eps), 1.0f),
            MlxRmsNorm(
                dense_vector(model, prefix + ".key_norm.weight"),
                static_cast<float>(config.rms_norm_eps), 1.0f),
            MlxLinear::load(model, prefix + ".output.weight"),
            MlxLinear::load(model, prefix + ".indexer.query_key.weight"),
            MlxRmsNorm(
                dense_vector(model, prefix + ".indexer.query_norm.weight"),
                static_cast<float>(config.rms_norm_eps), 1.0f),
            MlxRmsNorm(
                dense_vector(model, prefix + ".indexer.key_norm.weight"),
                static_cast<float>(config.rms_norm_eps), 1.0f)));
    }

    void reset(int batch) override {
        cache_ = std::make_unique<MlxKvCache>(
            batch,
            static_cast<int>(config_.num_key_value_heads),
            maximum_,
            static_cast<int>(config_.head_dim));
        index_cache_.reset(batch);
        batch_ = batch;
    }

    void clear() noexcept override {
        cache_.reset();
        index_cache_.clear();
        batch_ = 0;
    }

    array forward(
        const array& hidden,
        const array& positions_current,
        const array& positions_full,
        bool use_cache) override {
        const int batch = hidden.shape(0);
        const int tokens = hidden.shape(1);
        auto query_full = query_(hidden);
        auto key_full = key_(hidden);
        auto value_full = value_(hidden);
        auto query_parts = mlx::core::split(
            mlx::core::reshape(
                query_full,
                Shape{
                    batch,
                    tokens,
                    static_cast<int>(config_.num_attention_heads),
                    static_cast<int>(2 * config_.head_dim),
                }),
            2,
            -1);
        auto query = mlx::core::transpose(
            query_norm_(query_parts.at(0)), {0, 2, 1, 3});
        auto output_gate = std::move(query_parts.at(1));
        auto key = mlx::core::transpose(
            key_norm_(mlx::core::reshape(
                key_full,
                Shape{
                    batch,
                    tokens,
                    static_cast<int>(config_.num_key_value_heads),
                    static_cast<int>(config_.head_dim),
                })),
            {0, 2, 1, 3});
        auto value = mlx::core::transpose(
            mlx::core::reshape(
                value_full,
                Shape{
                    batch,
                    tokens,
                    static_cast<int>(config_.num_key_value_heads),
                    static_cast<int>(config_.head_dim),
                }),
            {0, 2, 1, 3});
        query = apply_rope(
            query,
            positions_current,
            static_cast<int>(config_.rotary_dim),
            static_cast<float>(config_.rope_theta),
            config_.rope_sections,
            config_.mrope_interleaved);
        key = apply_rope(
            key,
            positions_current,
            static_cast<int>(config_.rotary_dim),
            static_cast<float>(config_.rope_theta),
            config_.rope_sections,
            config_.mrope_interleaved);
        auto index_parts = mlx::core::split(
            index_query_key_(hidden),
            Shape{static_cast<int>(
                config_.indexer_n_heads * config_.indexer_head_dim)},
            -1);
        auto index_query = index_query_norm_(mlx::core::reshape(
            index_parts.at(0),
            Shape{
                batch,
                tokens,
                static_cast<int>(config_.indexer_n_heads),
                static_cast<int>(config_.indexer_head_dim),
            }));
        index_query = mlx::core::transpose(
            apply_rope(
                mlx::core::transpose(index_query, {0, 2, 1, 3}),
                positions_current,
                static_cast<int>(config_.rotary_dim),
                static_cast<float>(config_.rope_theta),
                config_.rope_sections,
                config_.mrope_interleaved),
            {0, 2, 1, 3});
        auto raw_key = mlx::core::reshape(
            index_parts.at(1),
            Shape{batch, tokens, static_cast<int>(config_.indexer_head_dim)});

        array key_cache = key;
        array value_cache = value;
        array raw_cache = raw_key;
        int query_offset = 0;
        if (use_cache) {
            if (!cache_ || batch_ != batch) reset(batch);
            query_offset = cache_->position();
            auto cache = cache_->append(key, value);
            key_cache = std::move(cache.first);
            value_cache = std::move(cache.second);
            auto index = index_cache_.append(raw_key);
            raw_cache = std::move(index.first);
            if (index.second != query_offset) {
                throw std::runtime_error("Qwen4 QSA caches diverged");
            }
        }
        array attended = key_cache.shape(2) <= config_.indexer_budget
            ? qwen4_dense_gqa_attention(
                  query, key_cache, value_cache, query_offset)
            : mlx_sparse_block_gqa_attention(
                  query,
                  key_cache,
                  value_cache,
                  selected_blocks(
                      index_query, raw_cache, positions_full, query_offset),
                  query_offset,
                  static_cast<int>(config_.indexer_compress_ratio));
        attended = mlx::core::reshape(
            attended,
            Shape{
                batch,
                tokens,
                static_cast<int>(
                    config_.num_attention_heads * config_.head_dim),
            });
        output_gate = mlx::core::reshape(output_gate, attended.shape());
        auto gated = mlx::core::astype(attended, mlx::core::float32) *
            mlx::core::sigmoid(
                mlx::core::astype(output_gate, mlx::core::float32));
        return output_(mlx::core::astype(gated, hidden.dtype()));
    }

private:
    Qwen4Qsa(
        Qwen4Config config,
        int maximum,
        MlxLinear query,
        MlxLinear key,
        MlxLinear value,
        MlxRmsNorm query_norm,
        MlxRmsNorm key_norm,
        MlxLinear output,
        MlxLinear index_query_key,
        MlxRmsNorm index_query_norm,
        MlxRmsNorm index_key_norm)
        : config_(std::move(config)),
          maximum_(maximum),
          query_(std::move(query)),
          key_(std::move(key)),
          value_(std::move(value)),
          query_norm_(std::move(query_norm)),
          key_norm_(std::move(key_norm)),
          output_(std::move(output)),
          index_query_key_(std::move(index_query_key)),
          index_query_norm_(std::move(index_query_norm)),
          index_key_norm_(std::move(index_key_norm)),
          index_cache_(maximum, static_cast<int>(config_.indexer_head_dim)) {}

    array selected_blocks(
        const array& query,
        const array& raw_keys,
        const array& positions_full,
        int query_offset) const {
        const int batch = query.shape(0);
        const int tokens = query.shape(1);
        const int ratio = static_cast<int>(config_.indexer_compress_ratio);
        const int complete = raw_keys.shape(1) / ratio;
        const int block_budget =
            static_cast<int>(config_.indexer_budget) / ratio;
        const int select_count = std::min(
            block_budget,
            complete);
        auto absolute = mlx::core::arange(
            query_offset,
            query_offset + tokens,
            1,
            mlx::core::int32);
        if (select_count <= 0) {
            throw std::runtime_error(
                "Qwen4 sparse attention has no complete blocks");
        }
        {
            auto pooled = mlx::core::mean(
                mlx::core::reshape(
                    mlx::core::slice(
                        raw_keys,
                        Shape{0, 0, 0},
                        Shape{
                            batch,
                            complete * ratio,
                            static_cast<int>(config_.indexer_head_dim),
                        }),
                    Shape{
                        batch,
                        complete,
                        ratio,
                        static_cast<int>(config_.indexer_head_dim),
                    }),
                -2);
            pooled = index_key_norm_(pooled);
            auto starts = mlx::core::arange(
                0, complete * ratio, ratio, mlx::core::int32);
            auto block_positions = mlx::core::take(
                positions_full, starts, -1);
            pooled = mlx::core::reshape(
                apply_rope(
                    mlx::core::expand_dims(pooled, 1),
                    block_positions,
                    static_cast<int>(config_.rotary_dim),
                    static_cast<float>(config_.rope_theta),
                    config_.rope_sections,
                    config_.mrope_interleaved),
                Shape{
                    batch,
                    complete,
                    static_cast<int>(config_.indexer_head_dim),
                });
            auto scores = qwen4_qsa_block_scores(query, pooled);
            auto ends = starts + array(ratio - 1, mlx::core::int32);
            auto visible = mlx::core::less(
                mlx::core::expand_dims(ends, 0),
                mlx::core::expand_dims(absolute + array(1, mlx::core::int32), 1));
            visible = mlx::core::broadcast_to(
                mlx::core::expand_dims(visible, 0),
                Shape{batch, tokens, complete});
            scores = mlx::core::where(
                visible,
                scores,
                array(-1e30f, mlx::core::float32));
            auto canonical = mlx::core::broadcast_to(
                mlx::core::reshape(
                    mlx::core::arange(select_count, mlx::core::int32),
                    Shape{1, 1, select_count}),
                Shape{batch, tokens, select_count});
            auto selected = canonical;
            if (complete > block_budget) {
                auto partition = mlx::core::argpartition(
                    scores, complete - block_budget, -1);
                auto ranked = mlx::core::astype(
                    mlx::core::slice(
                        partition,
                        Shape{0, 0, complete - block_budget},
                        Shape{batch, tokens, complete}),
                    mlx::core::int32);
                auto complete_counts = mlx::core::floor_divide(
                    absolute + array(1, mlx::core::int32),
                    array(ratio, mlx::core::int32));
                auto use_canonical = mlx::core::broadcast_to(
                    mlx::core::reshape(
                        mlx::core::less_equal(
                            complete_counts,
                            array(block_budget, mlx::core::int32)),
                        Shape{1, tokens, 1}),
                    Shape{batch, tokens, select_count});
                selected = mlx::core::where(
                    use_canonical,
                    canonical,
                    ranked);
            }
            // Top-k/argpartition order is unspecified. Chronological block
            // order preserves the checkpoint reduction order and lets the
            // direct-index kernel treat the first min(visible,budget) entries
            // as valid.
            return mlx::core::contiguous(mlx::core::sort(selected, -1));
        }
    }

    Qwen4Config config_;
    int maximum_;
    MlxLinear query_;
    MlxLinear key_;
    MlxLinear value_;
    MlxRmsNorm query_norm_;
    MlxRmsNorm key_norm_;
    MlxLinear output_;
    MlxLinear index_query_key_;
    MlxRmsNorm index_query_norm_;
    MlxRmsNorm index_key_norm_;
    std::unique_ptr<MlxKvCache> cache_;
    SequenceCache index_cache_;
    int batch_ = 0;
};

class Qwen4Layer {
public:
    static Qwen4Layer load(
        const MfqContainer& model,
        const Qwen4Config& config,
        std::size_t index,
        int maximum) {
        const auto prefix =
            "model.block." + std::to_string(index);
        std::unique_ptr<Qwen4Attention> attention;
        if (config.layer_types.at(index) == "linear_attention") {
            attention = Qwen4Gdn::load(
                model, config, prefix + ".linear_attention");
        } else {
            attention = Qwen4Qsa::load(
                model, config, prefix + ".attention", maximum);
        }
        std::unique_ptr<Qwen4Ple> ple;
        if (std::find(
                config.ple_layer_ids.begin(),
                config.ple_layer_ids.end(),
                static_cast<std::int64_t>(index + 1)) !=
            config.ple_layer_ids.end()) {
            ple = std::make_unique<Qwen4Ple>(Qwen4Ple::load(
                model, config, prefix + ".position_embedding"));
        }
        return Qwen4Layer(
            GatedResidual::load(
                model, config, prefix + ".attention.mhc.pre"),
            GatedResidual::load(
                model, config, prefix + ".mlp.mhc.pre"),
            std::move(attention),
            Qwen4Moe::load(model, config, prefix + ".mlp"),
            std::move(ple));
    }

    void reset(int batch) {
        attention_->reset(batch);
        if (ple_) ple_->reset(batch);
    }

    void clear() noexcept {
        attention_->clear();
        if (ple_) ple_->clear();
    }

    array forward(
        array hidden_streams,
        const array& token_ids,
        const array& positions_current,
        const array& positions_full,
        bool use_cache) {
        if (ple_) {
            hidden_streams = hidden_streams +
                ple_->forward(hidden_streams, token_ids, use_cache);
            detail::profile_eval("qwen4.ple", hidden_streams);
        }
        auto attention_values = attention_gr_.pre(hidden_streams);
        detail::profile_eval(
            "qwen4.attention_mhc_pre",
            attention_values.branch);
        auto branch = attention_->forward(
            attention_values.branch,
            positions_current,
            positions_full,
            use_cache);
        detail::profile_eval(attention_->profile_name(), branch);
        hidden_streams = attention_gr_.post(branch, attention_values);
        detail::profile_eval(
            "qwen4.attention_mhc_post",
            hidden_streams);
        auto ffn_values = ffn_gr_.pre(hidden_streams);
        detail::profile_eval("qwen4.ffn_mhc_pre", ffn_values.branch);
        branch = moe_(ffn_values.branch);
        detail::profile_eval("qwen4.moe", branch);
        auto output = ffn_gr_.post(branch, ffn_values);
        detail::profile_eval("qwen4.ffn_mhc_post", output);
        return output;
    }

private:
    Qwen4Layer(
        GatedResidual attention_gr,
        GatedResidual ffn_gr,
        std::unique_ptr<Qwen4Attention> attention,
        Qwen4Moe moe,
        std::unique_ptr<Qwen4Ple> ple)
        : attention_gr_(std::move(attention_gr)),
          ffn_gr_(std::move(ffn_gr)),
          attention_(std::move(attention)),
          moe_(std::move(moe)),
          ple_(std::move(ple)) {}

    GatedResidual attention_gr_;
    GatedResidual ffn_gr_;
    std::unique_ptr<Qwen4Attention> attention_;
    Qwen4Moe moe_;
    std::unique_ptr<Qwen4Ple> ple_;
};

} // namespace

Qwen4Config Qwen4Config::from_json(std::string_view payload) {
    const auto outer = json::parse(payload);
    const auto& text = text_config(outer);
    const auto outer_type = outer.value("model_type", std::string{});
    const auto inner_type = text.value("model_type", outer_type);
    if ((outer_type != "qwen4_exp" && outer_type != "qwen4_exp_text") ||
        inner_type != "qwen4_exp_text") {
        throw std::runtime_error("expected Qwen4-Exp text config");
    }

    Qwen4Config config;
    config.model_type = outer_type;
    config.text_model_type = inner_type;
    config.vocab_size = positive(text, "vocab_size");
    config.hidden_size = positive(text, "hidden_size");
    config.num_hidden_layers = positive(text, "num_hidden_layers");
    config.max_position_embeddings = positive(text, "max_position_embeddings");
    config.num_attention_heads = positive(text, "num_attention_heads");
    config.num_key_value_heads = positive(text, "num_key_value_heads");
    config.head_dim = positive(text, "head_dim");
    config.full_attention_interval = positive(text, "full_attention_interval");
    config.hc_count = positive(text, "hc_count");
    config.hc_lowrank = positive(text, "hc_lowrank");
    config.layer_types = text.at("layer_types").get<std::vector<std::string>>();
    if (config.layer_types.size() !=
        static_cast<std::size_t>(config.num_hidden_layers)) {
        throw std::runtime_error("Qwen4 layer schedule length disagrees");
    }
    for (std::int64_t index = 0;
         index < config.num_hidden_layers; ++index) {
        const auto expected = (index + 1) % config.full_attention_interval == 0
            ? "full_attention" : "linear_attention";
        if (config.layer_types[static_cast<std::size_t>(index)] != expected) {
            throw std::runtime_error("Qwen4 attention schedule disagrees");
        }
    }

    const auto rope = text.value("rope_parameters", json::object());
    const double partial = text.value(
        "partial_rotary_factor",
        rope.value("partial_rotary_factor", 1.0));
    if (!std::isfinite(partial) || partial <= 0.0 || partial > 1.0) {
        throw std::runtime_error("invalid Qwen4 rotary fraction");
    }
    const double rotation = static_cast<double>(config.head_dim) * partial;
    const double lower = std::floor(rotation);
    config.rotary_dim = static_cast<std::int64_t>(lower) +
        ((rotation - lower > 0.5 ||
          (rotation - lower == 0.5 &&
           static_cast<std::int64_t>(lower) % 2 != 0))
             ? 1
             : 0);
    config.rope_sections = rope.value(
        "mrope_section", std::vector<std::int64_t>{});
    config.rope_theta = rope.value("rope_theta", 1e7);
    config.mrope_interleaved = rope.value("mrope_interleaved", false);
    const auto section_sum = std::accumulate(
        config.rope_sections.begin(), config.rope_sections.end(),
        std::int64_t{0});
    if (config.rotary_dim <= 0 || config.rotary_dim % 2 != 0 ||
        (!config.rope_sections.empty() &&
         (config.rope_sections.size() != 3 ||
          section_sum * 2 != config.rotary_dim)) ||
        (config.mrope_interleaved && config.rope_sections.empty()) ||
        !std::isfinite(config.rope_theta) || config.rope_theta <= 0.0) {
        throw std::runtime_error("Qwen4 rotary geometry disagrees");
    }

    config.linear_num_key_heads = positive(text, "linear_num_key_heads");
    config.linear_num_value_heads = positive(text, "linear_num_value_heads");
    config.linear_key_head_dim = positive(text, "linear_key_head_dim");
    config.linear_value_head_dim = positive(text, "linear_value_head_dim");
    config.linear_conv_kernel_dim = positive(text, "linear_conv_kernel_dim");
    config.num_experts = positive(text, "num_experts");
    config.num_experts_per_tok = positive(text, "num_experts_per_tok");
    config.moe_intermediate_size = positive(text, "moe_intermediate_size");
    config.shared_expert_intermediate_size =
        positive(text, "shared_expert_intermediate_size");
    config.indexer_n_heads = positive(text, "indexer_n_heads");
    config.indexer_head_dim = positive(text, "indexer_head_dim");
    config.indexer_compress_ratio = positive(text, "indexer_compress_ratio");
    config.indexer_budget = positive(text, "indexer_budget");
    config.ple_conv_kernel_size = positive(text, "ple_conv_kernel_size");
    config.ngram_size = positive(text, "ngram_size");
    config.heads_per_ngram = positive(text, "heads_per_ngram");
    config.split_ngram_parts = positive(text, "split_ngram_parts");
    (void)positive(text, "ngram_vocab_size_base");
    const auto ple_embed_dim = positive(text, "ple_embed_dim");
    const auto indexer_kv_heads = positive(text, "indexer_kv_heads");
    if (config.num_attention_heads % config.num_key_value_heads != 0 ||
        config.linear_num_value_heads % config.linear_num_key_heads != 0 ||
        config.linear_key_head_dim != config.linear_value_head_dim ||
        config.linear_conv_kernel_dim < 2 || config.hc_count < 2 ||
        ple_embed_dim != config.hidden_size || config.ngram_size < 2 ||
        config.hidden_size %
                ((config.ngram_size - 1) * config.heads_per_ngram) !=
            0 ||
        indexer_kv_heads != 1 ||
        config.indexer_budget % config.indexer_compress_ratio != 0 ||
        config.num_experts_per_tok >
            std::min<std::int64_t>(16, config.num_experts) ||
        config.num_experts > 4096 ||
        config.indexer_head_dim < config.rotary_dim) {
        throw std::runtime_error("Qwen4 attention/routing geometry disagrees");
    }
    config.ple_layer_ids = text.value(
        "ple_layer_ids", std::vector<std::int64_t>{});
    for (const auto id : config.ple_layer_ids) {
        if (id < 1 || id > config.num_hidden_layers ||
            config.layer_types[static_cast<std::size_t>(id - 1)] !=
                "linear_attention") {
            throw std::runtime_error("Qwen4 PLE layer schedule disagrees");
        }
    }
    auto gate = text.value("output_gate_type", std::string{});
    if (gate.empty()) gate = text.value("hidden_act", std::string("silu"));
    if ((gate != "silu" && gate != "sigmoid") ||
        text.value("hidden_act", std::string("silu")) != "silu" ||
        text.value("attention_bias", false)) {
        throw std::runtime_error("unsupported Qwen4 activation semantics");
    }
    config.output_gate_silu = gate == "silu";
    config.rms_norm_eps = text.value("rms_norm_eps", 1e-6);
    if (!std::isfinite(config.rms_norm_eps) || config.rms_norm_eps <= 0.0) {
        throw std::runtime_error("invalid Qwen4 norm epsilon");
    }
    config.norm_topk_prob = text.value("norm_topk_prob", true);
    config.tie_word_embeddings = text.value("tie_word_embeddings", false);
    config.mtp_num_hidden_layers =
        text.value("mtp_num_hidden_layers", std::int64_t{0});
    config.mtp_use_dedicated_embeddings =
        text.value("mtp_use_dedicated_embeddings", false);
    if (config.mtp_num_hidden_layers < 0) {
        throw std::runtime_error("invalid Qwen4 predictor layer count");
    }
    auto eos = text.contains("eos_token_id")
        ? text.at("eos_token_id")
        : outer.value("eos_token_id", json(nullptr));
    if (!config.ple_layer_ids.empty() &&
        (eos.is_null() || (eos.is_array() && eos.empty()))) {
        throw std::runtime_error("Qwen4 PLE requires an EOS token ID");
    }
    config.eos_token_id = eos.is_array()
        ? eos.at(0).get<std::int64_t>()
        : eos.is_null() ? 0 : eos.get<std::int64_t>();
    return config;
}

Qwen4Config Qwen4Config::from_mfq(const MfqContainer& model) {
    if (!model.contains(std::string(kModelConfigAsset))) {
        throw std::runtime_error("Qwen4 MFQ has no embedded model config");
    }
    return from_json(model.read_text(std::string(kModelConfigAsset)));
}

struct MlxQwen4CausalLm::Impl {
    static std::unique_ptr<Impl> load(
        const MfqContainer& model,
        int requested_context) {
        const auto graph = effective_model_graph(model);
        if (graph.graph_kind != "causal_lm" ||
            graph.backbone != "qwen4_exp" ||
            !graph.has_component("text")) {
            throw std::runtime_error(
                "model graph does not describe a Qwen4-Exp causal runtime");
        }
        auto config = Qwen4Config::from_mfq(model);
        const int maximum = std::min(
            checked_int(config.max_position_embeddings, "context"),
            requested_context);
        auto embedding = MlxEmbedding::load(
            model, "model.token_embedding.weight");
        std::optional<MlxLinear> output;
        if (!config.tie_word_embeddings && model.contains("model.output.weight")) {
            output.emplace(MlxLinear::load(model, "model.output.weight"));
        }
        std::vector<Qwen4Layer> layers;
        layers.reserve(static_cast<std::size_t>(config.num_hidden_layers));
        for (std::size_t index = 0;
             index < static_cast<std::size_t>(config.num_hidden_layers);
             ++index) {
            layers.push_back(Qwen4Layer::load(model, config, index, maximum));
        }
        auto mixer = GatedResidual::load(
            model, config, "model.mhc.pre", false);
        return std::unique_ptr<Impl>(new Impl(
            std::move(config),
            maximum,
            std::move(embedding),
            std::move(output),
            std::move(layers),
            std::move(mixer)));
    }

    array forward(
        const array& token_ids,
        bool use_cache,
        bool last_token_only = false) {
        if (token_ids.ndim() != 2 || token_ids.shape(0) <= 0 ||
            token_ids.shape(1) <= 0) {
            throw std::runtime_error("Qwen4 token IDs must have [B,T] shape");
        }
        const int batch = token_ids.shape(0);
        const int tokens = token_ids.shape(1);
        if (use_cache && (cache_batch == 0 || cache_batch != batch)) {
            reset(batch);
        }
        const int start = use_cache ? cache_position : 0;
        if (start + tokens > maximum) {
            throw std::runtime_error("Qwen4 input exceeds context capacity");
        }
        auto ids = token_ids.dtype() == mlx::core::int32
            ? token_ids : mlx::core::astype(token_ids, mlx::core::int32);
        auto hidden = embedding(ids, mlx::core::float16);
        auto streams = mlx::core::reshape(
            mlx::core::broadcast_to(
                mlx::core::expand_dims(hidden, -2),
                Shape{
                    batch,
                    tokens,
                    static_cast<int>(config.hc_count),
                    static_cast<int>(config.hidden_size),
                }),
            Shape{
                batch,
                tokens,
                static_cast<int>(config.hc_count * config.hidden_size),
            });
        auto current_positions = mlx::core::arange(
            start, start + tokens, 1, mlx::core::int32);
        auto full_positions = mlx::core::arange(
            0, start + tokens, 1, mlx::core::int32);
        for (auto& layer : layers) {
            streams = layer.forward(
                std::move(streams),
                ids,
                current_positions,
                full_positions,
                use_cache);
        }
        auto output_hidden = mixer.mix(streams);
        if (last_token_only && tokens > 1) {
            output_hidden = mlx::core::slice(
                output_hidden,
                {0, tokens - 1, 0},
                {batch, tokens, output_hidden.shape(-1)});
        }
        auto logits = output
            ? (*output)(output_hidden)
            : embedding.project(output_hidden);
        if (use_cache) {
            cache_position = start + tokens;
            cache_batch = batch;
        }
        return logits;
    }

    void reset(int batch) {
        if (batch <= 0) {
            throw std::runtime_error("Qwen4 cache batch must be positive");
        }
        for (auto& layer : layers) layer.reset(batch);
        cache_batch = batch;
        cache_position = 0;
    }

    void clear() noexcept {
        for (auto& layer : layers) layer.clear();
        cache_batch = 0;
        cache_position = 0;
    }

private:
    Impl(
        Qwen4Config selected_config,
        int selected_maximum,
        MlxEmbedding selected_embedding,
        std::optional<MlxLinear> selected_output,
        std::vector<Qwen4Layer> selected_layers,
        GatedResidual selected_mixer)
        : config(std::move(selected_config)),
          maximum(selected_maximum),
          embedding(std::move(selected_embedding)),
          output(std::move(selected_output)),
          layers(std::move(selected_layers)),
          mixer(std::move(selected_mixer)) {}

public:
    Qwen4Config config;
    int maximum;
    MlxEmbedding embedding;
    std::optional<MlxLinear> output;
    std::vector<Qwen4Layer> layers;
    GatedResidual mixer;
    int cache_batch = 0;
    int cache_position = 0;
};

MlxQwen4CausalLm::MlxQwen4CausalLm(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MlxQwen4CausalLm::~MlxQwen4CausalLm() = default;
MlxQwen4CausalLm::MlxQwen4CausalLm(MlxQwen4CausalLm&&) noexcept = default;
MlxQwen4CausalLm& MlxQwen4CausalLm::operator=(MlxQwen4CausalLm&&) noexcept = default;

MlxQwen4CausalLm MlxQwen4CausalLm::load(
    const MfqContainer& model,
    int max_context) {
    if (max_context <= 0) {
        throw std::invalid_argument("Qwen4 max_context must be positive");
    }
    return MlxQwen4CausalLm(Impl::load(model, max_context));
}

array MlxQwen4CausalLm::forward(
    const array& token_ids,
    bool use_cache) {
    return impl_->forward(token_ids, use_cache);
}

void MlxQwen4CausalLm::reset_cache(int batch) {
    impl_->reset(batch);
}

void MlxQwen4CausalLm::clear_cache() noexcept {
    impl_->clear();
}

std::int32_t MlxQwen4CausalLm::generate(
    const std::vector<std::int64_t>& prompt,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    const MfqTokenConstraintPtr& token_constraint,
    std::optional<std::size_t>) {
    if (prompt.empty()) {
        throw std::invalid_argument("Qwen4 generation prompt cannot be empty");
    }
    if (max_tokens < 0) {
        throw std::invalid_argument("Qwen4 generation max_tokens is negative");
    }
    const int vocab = checked_int(impl_->config.vocab_size, "vocab size");
    if (prompt.size() > static_cast<std::size_t>(impl_->maximum)) {
        throw std::invalid_argument("Qwen4 prompt exceeds context capacity");
    }
    std::vector<std::int32_t> values;
    values.reserve(prompt.size());
    for (const auto token : prompt) {
        if (token < 0 || token >= vocab) {
            throw std::invalid_argument("Qwen4 prompt token is out of range");
        }
        values.push_back(static_cast<std::int32_t>(token));
    }
    if (max_tokens == 0) {
        reset_cache(1);
        return 0;
    }
    reset_cache(1);
    const array prompt_ids(
        values.begin(), Shape{1, static_cast<int>(values.size())},
        mlx::core::int32);
    std::optional<array> counts;
    if (sampling.has_penalties()) {
        counts = sample_token_counts_add(
            mlx::core::zeros(Shape{vocab}, mlx::core::int32),
            prompt_ids);
    }
    double prefill_ms = 0.0;
    const bool profile_prefill = detail::component_profile_requested();
    detail::ComponentProfile component_profile;
    const auto profile_started = std::chrono::steady_clock::now();
    auto logits = [&] {
        detail::ScopedComponentProfile component_scope(
            profile_prefill ? &component_profile : nullptr);
        detail::ScopedMlxEvaluationTiming timing(
            prefill_callback ? &prefill_ms : nullptr);
        auto value = last_token_logits(
            impl_->forward(prompt_ids, true, true), vocab);
        if (profile_prefill) {
            detail::profile_eval("qwen4.output", value);
        } else if (prefill_callback) {
            detail::eval_with_timing(value);
        }
        return value;
    }();
    if (profile_prefill) {
        const double wall_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - profile_started)
                .count();
        const double evaluated_ms = component_profile.evaluated_ms();
        prefill_ms = wall_ms;
        std::cout
            << "component_profile model=qwen4 phase=prefill"
            << " tokens=" << prompt.size()
            << " wall_ms=" << std::fixed << std::setprecision(3) << wall_ms
            << " evaluated_ms=" << evaluated_ms
            << " unscoped_ms=" << std::max(0.0, wall_ms - evaluated_ms)
            << std::endl;
        for (const auto& [name, timing] : component_profile.timings()) {
            std::cout
                << "component_cost model=qwen4 phase=prefill"
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
    if (prefill_callback) prefill_callback(prompt.size(), prefill_ms);

    MlxSampler sampler(sampling);
    const int limit = std::min<int>(
        max_tokens,
        impl_->maximum - static_cast<int>(prompt.size()) + 1);
    std::int32_t generated = 0;
    while (generated < limit) {
        auto sampled = counts
            ? sampler.sample(logits, *counts)
            : sampler.sample(logits);
        sampled.eval();
        auto token = sampled.data<std::int32_t>()[0];
        if (token < 0 || token >= vocab) {
            throw std::runtime_error("Qwen4 sampler returned an invalid token");
        }
        if (token_constraint && token_constraint->allows &&
            !token_constraint->allows(token)) {
            auto adjusted = counts
                ? sampler.apply_penalties(logits, *counts)
                : logits;
            adjusted = mlx::core::contiguous(
                mlx::core::astype(adjusted, mlx::core::float32));
            adjusted.eval();
            std::vector<float> masked(
                adjusted.data<float>(), adjusted.data<float>() + vocab);
            token_constraint->apply(masked.data(), masked.size());
            const array constrained(
                masked.begin(), Shape{1, vocab}, mlx::core::float32);
            sampled = sampler.sample(constrained);
            sampled.eval();
            token = sampled.data<std::int32_t>()[0];
            if (token < 0 || token >= vocab ||
                !token_constraint->allows(token)) {
                throw std::runtime_error(
                    "Qwen4 constrained sampler returned an invalid token");
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
        if (callback && !callback(token)) break;
        if (generated == limit) break;
        logits = last_token_logits(forward(token_ids, true), vocab);
    }
    return generated;
}

const Qwen4Config& MlxQwen4CausalLm::config() const noexcept {
    return impl_->config;
}

std::size_t MlxQwen4CausalLm::layer_count() const noexcept {
    return impl_->layers.size();
}

int MlxQwen4CausalLm::cache_position() const noexcept {
    return impl_->cache_position;
}

MlxQwen4TextSessionState MlxQwen4CausalLm::capture_text_session_state(
    const std::vector<std::int64_t>&) const {
    throw std::runtime_error(
        "Qwen4 persistent text-session snapshots are not enabled yet");
}

void MlxQwen4CausalLm::restore_text_session_state(
    const MlxQwen4TextSessionState&) {
    throw std::runtime_error(
        "Qwen4 persistent text-session snapshots are not enabled yet");
}

} // namespace mfq::metal
