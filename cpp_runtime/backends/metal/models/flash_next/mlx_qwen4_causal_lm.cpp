#include "mlx_qwen4_causal_lm.h"

#include "mlx_eval_timing.h"
#include "mlx_detached_copy.h"
#include "qwen4_ops.h"
#include "mlx_legacy_tensor_compat.h"
#include "mlx_linear_attention.h"
#include "mlx_moe.h"
#include "mlx_moe_ops.h"
#include "mlx_nint_rows.h"
#include "mlx_tq_rows.h"
#include "mlx_tq_expert_cache.h"
#include "mfe_expert_store.h"
#include "mlx_sparse_attention.h"
#include "mlx_tensor.h"
#include "mlx_kernel_prepare.h"
#include "mlx_mtp_lora.h"
#include "runtime_config.h"
#include "mfq_paged_prefix_cache.h"
#include "mlx_transformer.h"

#include "nlohmann/json.hpp"
#include <mlx/compile.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <tuple>
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

// Qwen4/Flash-Next recursively reuses its single predictor layer to form a
// draft chain. Predictor layer count and maximum draft depth are independent.
constexpr int kQwen4MtpMaximumDraftDepth = 5;

struct Qwen4PromptPositions {
    std::optional<array> prompt;
    int delta = 0;

    array full(int count) const {
        if (!prompt) return mlx::core::arange(0, count, 1, mlx::core::int32);
        const int prefix = std::min(count, prompt->shape(-1));
        auto positions = mlx::core::slice(*prompt, {0, 0}, {3, prefix});
        if (count > prefix) {
            auto tail = mlx::core::broadcast_to(
                mlx::core::arange(prefix + delta, count + delta, 1, mlx::core::int32),
                Shape{3, count - prefix});
            positions = mlx::core::concatenate({positions, tail}, -1);
        }
        return positions;
    }

    array current(const array& full, int start, int count) const {
        return prompt ? mlx::core::slice(full, {0, start}, {3, start + count})
                      : mlx::core::arange(start, start + count, 1, mlx::core::int32);
    }
};

struct Qwen4PleSpeculativeState {
    std::optional<array> convolution_state;
    std::vector<std::int64_t> ngram_context;
    std::optional<array> token_ids, convolution_inputs;
    int confirmed_tokens = 0, total_tokens = 0;
};

struct Qwen4RequestLayerCache {
    int position = 0, batch = 0, index_start = 0, index_position = 0;
    int ple_batch = 0, ngram_batch = 0;
    std::unique_ptr<MlxKvCache> kv;
    std::unique_ptr<MlxQsaKvOffload> offloaded_kv;
    std::unique_ptr<MlxQsaIndexOffload> offloaded_pool;
    std::optional<MlxSequenceCache> pooled;
    std::optional<array> convolution, recurrent, index_tail, ple_convolution;
    std::vector<std::int64_t> ple_context;
    std::optional<Qwen4PleSpeculativeState> ple_rollback;
    std::optional<MlxGatedDeltaSpeculativeState> gdn_rollback;
    int qsa_speculative_trim = 0;
    MlxResourceTelemetry attention_resources, ple_resources;
};

std::string mtp_source_identity(const MfqContainer& model) {
    std::ostringstream identity;
    identity << model.header().architecture << '\n' << model.read_text(std::string(kModelConfigAsset));
    for (const auto& path : model.source_paths()) {
        identity << '\n' << path.string() << ':' << std::filesystem::file_size(path) << ':' <<
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::filesystem::last_write_time(path).time_since_epoch()).count();
    }
    return mfq::cache::block_hash_hex(mfq::cache::sha256(identity.str()));
}

bool mtp_ttt_requested() { return mfq::engine::environment_enabled("MFQ_MTP_TTT", false); }
bool mtp_fp16_requested() {
    const auto* source = std::getenv("MFQ_MTP_SOURCE");
    return mtp_ttt_requested() || (source && *source);
}

bool decode_async_enabled() noexcept {
    static const bool enabled = [] {
        const char* setting = std::getenv("MFQ_METAL_QWEN_DECODE_ASYNC");
        return setting == nullptr || std::atoi(setting) != 0;
    }();
    return enabled;
}

bool gdn_decode_step_enabled() noexcept {
    static const bool enabled = [] {
        const char* setting = std::getenv("MFQ_METAL_GDN_DECODE_STEP");
        return setting == nullptr || std::atoi(setting) != 0;
    }();
    return enabled;
}

bool qsa_decode_prologue_enabled() noexcept {
    static const bool enabled = [] {
        const char* setting = std::getenv(
            "MFQ_METAL_QWEN_QSA_DECODE_PROLOGUE");
        return setting == nullptr || std::atoi(setting) != 0;
    }();
    return enabled;
}

// Adapted from oMLX's Qwen4 eager-dispatch policy (Apache-2.0), revision
// 87460f4d50de79aef9b67e99e215c31f0a89b445, language.py. MFQ adaptation:
// single-sequence forwards up to six tokens, with synchronized profiling excluded.
// Copyright 2025 oMLX contributors. See NOTICE for upstream attribution.
int decode_async_interval() noexcept {
    static const int interval = [] {
        const char* setting = std::getenv("MFQ_METAL_QWEN_DECODE_ASYNC_EVERY");
        if (setting == nullptr || *setting == '\0') return 3;
        char* end = nullptr;
        const long value = std::strtol(setting, &end, 10);
        return end != setting && *end == '\0' && value >= 1 && value <= 1024
            ? static_cast<int>(value) : 3;
    }();
    return interval;
}

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
    if (!dtype && name.starts_with("predictor.") && mtp_fp16_requested() &&
        (result.dtype() == mlx::core::bfloat16 || result.dtype() == mlx::core::float32)) dtype = mlx::core::float16;
    if (dtype && result.dtype() != *dtype) {
        result = mlx::core::astype(result, *dtype);
    }
    return mlx::core::contiguous(result);
}

MlxLinear mhc_projection(const MfqContainer& model, const std::string& name) {
    if (model.record(name).dtype == "TQ") return MlxLinear::load(model, name);
    if (!is_nint_dtype(model.record(name).dtype)) return MlxLinear(dense(model, name));
    const auto mapped = model.map_record(name);
    return MlxLinear(MlxNintWeight::from_blob(mapped.view()));
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

MlxMfeWeight moe_weight(
    const MfqContainer& model,
    const std::string& name) {
    if (model.record(name).dtype != "MFE") {
        throw std::runtime_error(
            "Qwen4 routed expert tensor must use MFE: " + name);
    }
    const auto mapped = model.map_record(name);
    return MlxMfeWeight::from_blob(mapped.view());
}

class Qwen4RoutedWeight {
public:
    const array& training_weight() const {
        if (!dense_) throw std::invalid_argument("MTP training requires floating expert weights");
        return *dense_;
    }
    std::size_t packed_bytes() const noexcept {
        if (tq_) return tq_->packed_nbytes() + (tq_up_ ? tq_up_->packed_nbytes() : 0);
        return packed_ ? packed_->packed_nbytes() : dense_->nbytes();
    }
    static Qwen4RoutedWeight load(
        const MfqContainer& model,
        const std::string& name) {
        if (model.record(name).dtype == "TQ")
            return Qwen4RoutedWeight(MlxTqWeight::load(model, name));
        if (model.record(name).dtype == "MFE") {
            if (model.is_hf_source() && name.starts_with("predictor.") && mtp_fp16_requested()) {
                const auto projection = name.rfind('.', name.size() - 8);
                if (projection == std::string::npos) throw std::invalid_argument("invalid predictor expert name");
                const auto root = name.substr(0, projection + 1);
                const auto suffix = name.substr(projection + 1);
                std::vector<array> experts;
                for (std::size_t expert = 0;; ++expert) {
                    const auto key = root + std::to_string(expert) + "." + suffix;
                    if (!model.contains(key)) break;
                    experts.push_back(dense(model, key, mlx::core::float16));
                }
                if (experts.empty()) throw std::invalid_argument("floating predictor experts are missing");
                auto values = mlx::core::stack(experts, 0);
                values.eval();
                return Qwen4RoutedWeight(std::move(values));
            }
            return Qwen4RoutedWeight(moe_weight(model, name));
        }
        if (model.record(name).dtype == "NINT") {
            const auto mapped = model.map_record(name);
            model.record_prepared(name);
            return Qwen4RoutedWeight(MlxMfeWeight::from_nint_blob(mapped.view()));
        }
        auto values = dense(model, name);
        if (values.ndim() != 3 || values.shape(0) <= 0 ||
            values.shape(1) <= 0 || values.shape(2) <= 0) {
            throw std::runtime_error(
                "Qwen4 dense routed tensor must use [experts,out,in]: " +
                name);
        }
        return Qwen4RoutedWeight(std::move(values));
    }

    static Qwen4RoutedWeight load_gate_up(
        const MfqContainer& model,
        const std::string& prefix) {
        const auto combined = prefix + ".experts.gate_up.weight";
        if (model.contains(combined)) {
            if (model.record(combined).dtype == "TQ")
                throw std::runtime_error("TQ MoE requires separate Gate, Up and Down tensors");
            return load(model, combined);
        }
        const auto gate = prefix + ".experts.gate.weight";
        const auto up = prefix + ".experts.up.weight";
        if (model.record(gate).dtype == "TQ" && model.record(up).dtype == "TQ")
            return Qwen4RoutedWeight(MlxTqWeight::load(model, gate), MlxTqWeight::load(model, up));
        const bool floating_hf = model.is_hf_source() && prefix.starts_with("predictor.") && mtp_fp16_requested();
        if (model.record(gate).dtype == "MFE" &&
            model.record(up).dtype == "MFE" && !floating_hf) {
            return Qwen4RoutedWeight(
                load_routed_gate_up_weight(model, prefix));
        }
        auto gate_values = floating_hf ? *load(model, gate).dense_ : dense(model, gate);
        auto up_values = floating_hf ? *load(model, up).dense_ : dense(model, up);
        if (gate_values.ndim() != 3 ||
            gate_values.shape() != up_values.shape()) {
            throw std::runtime_error(
                "Qwen4 split dense gate/up tensors disagree");
        }
        return Qwen4RoutedWeight(mlx::core::concatenate(
            {std::move(gate_values), std::move(up_values)}, 1));
    }

    array routed_matmul(
        const array& input,
        const array& expert_ids,
        const array* route_groups = nullptr) const {
        if (tq_) return tq_->routed_matmul(input, expert_ids);
        if (packed_) return packed_->routed_matmul(input, expert_ids, route_groups);
        return dense_matmul(input, expert_ids);
    }

    array routed_matmul_reduce(
        const array& input,
        const array& expert_ids,
        const array& route_weights,
        const array* route_groups = nullptr) const {
        if (tq_) return tq_->routed_matmul_reduce(input, expert_ids, route_weights);
        if (packed_) {
            return packed_->routed_matmul_reduce(
                input, expert_ids, route_weights, route_groups);
        }
        return moe_weighted_reduce(
            dense_matmul(input, expert_ids), route_weights);
    }

    array routed_swiglu(
        const array& input,
        const array& expert_ids,
        const array* route_groups = nullptr) const {
        if (tq_) {
            if (!tq_up_) throw std::runtime_error("TQ Gate/Up pair is incomplete");
            return tq_->swiglu(*tq_up_, input, expert_ids);
        }
        if (packed_) return packed_->routed_swiglu(input, expert_ids, 0.0f, route_groups);
        return moe_swiglu_split(dense_matmul(input, expert_ids));
    }

    bool supports_grouped_mmq() const noexcept {
        return packed_ && packed_->supports_grouped_mmq();
    }

    bool supports_fused_routed_reduce(const array* route_groups = nullptr) const noexcept {
        if (tq_) return tq_->input_size() < 1024;
        return packed_ && packed_->supports_fused_routed_reduce(route_groups);
    }

    int recommended_grouped_mmq_block_rows(
        int route_count,
        bool fused_swiglu) const noexcept {
        return packed_
            ? packed_->recommended_grouped_mmq_block_rows(
                  route_count, fused_swiglu)
            : 0;
    }

    MlxGroupedMmqPlan build_grouped_mmq_plan(
        const array& expert_ids,
        const array& route_order,
        int block_rows) const {
        if (!packed_) {
            throw std::runtime_error(
                "dense routed projection has no grouped MMQ plan");
        }
        return packed_->build_grouped_mmq_plan(
            expert_ids, route_order, block_rows);
    }

    array routed_matmul_sorted(
        const array& input,
        const array& expert_ids,
        const array& route_order,
        bool input_is_sorted,
        bool fused_swiglu,
        float swiglu_limit,
        const MlxGroupedMmqPlan* plan) const {
        if (!packed_) {
            throw std::runtime_error(
                "dense routed projection has no grouped MMQ path");
        }
        return packed_->routed_matmul_sorted(
            input,
            expert_ids,
            route_order,
            input_is_sorted,
            fused_swiglu,
            swiglu_limit,
            plan);
    }

    int experts() const noexcept {
        if (tq_) return tq_->experts();
        return packed_ ? packed_->experts() : dense_->shape(0);
    }
    int out_per_expert() const noexcept {
        if (tq_) return tq_->output_size();
        return packed_ ? packed_->out_per_expert() : dense_->shape(1);
    }
    int neuron_len() const noexcept {
        if (tq_) return tq_->input_size();
        return packed_ ? packed_->neuron_len() : dense_->shape(2);
    }
    int projections() const noexcept {
        if (tq_) return tq_up_ ? 2 : 1;
        return packed_ ? packed_->projections() : 1;
    }
    bool uses_tq() const noexcept { return tq_.has_value(); }

    const MlxMfeWeight* packed_weight_ref() const noexcept {
        return packed_ ? &*packed_ : nullptr;
    }

private:
    explicit Qwen4RoutedWeight(MlxTqWeight tq) : tq_(std::move(tq)) {}
    Qwen4RoutedWeight(MlxTqWeight gate, MlxTqWeight up)
        : tq_(std::move(gate)), tq_up_(std::move(up)) {
        if (tq_->experts() != tq_up_->experts() || tq_->input_size() != tq_up_->input_size()
            || tq_->output_size() != tq_up_->output_size())
            throw std::runtime_error("Qwen TQ Gate/Up geometry differs");
    }
    explicit Qwen4RoutedWeight(MlxMfeWeight packed)
        : packed_(std::move(packed)) {}
    explicit Qwen4RoutedWeight(array dense_values)
        : dense_(std::move(dense_values)) {}

    array dense_matmul(
        const array& input,
        const array& expert_ids) const {
        auto source = mlx::core::reshape(
            input.dtype() == dense_->dtype()
                ? input
                : mlx::core::astype(input, dense_->dtype()),
            Shape{
                static_cast<int>(input.size() /
                    static_cast<std::size_t>(neuron_len())),
                neuron_len(),
            });
        if (source.shape(1) != neuron_len() || expert_ids.ndim() != 2 ||
            expert_ids.shape(0) <= 0 || expert_ids.shape(1) <= 0) {
            throw std::runtime_error(
                "Qwen4 dense routed projection geometry disagrees");
        }
        const int source_rows = source.shape(0);
        const int token_rows = expert_ids.shape(0);
        const int routes = static_cast<int>(expert_ids.size());
        const int routes_per_row = expert_ids.shape(1);
        if (routes != token_rows * routes_per_row ||
            (source_rows != token_rows && source_rows != routes)) {
            throw std::runtime_error(
                "Qwen4 dense routed IDs disagree with input rows");
        }
        auto lhs = source_rows == routes
            ? mlx::core::arange(routes, mlx::core::int32)
            : mlx::core::reshape(
                  mlx::core::broadcast_to(
                      mlx::core::expand_dims(
                          mlx::core::arange(
                              token_rows, mlx::core::int32),
                          1),
                      Shape{token_rows, routes_per_row}),
                  Shape{routes});
        auto rhs = mlx::core::contiguous(mlx::core::reshape(
            expert_ids.dtype() == mlx::core::int32
                ? expert_ids
                : mlx::core::astype(expert_ids, mlx::core::int32),
            Shape{routes}));
        auto weights = mlx::core::transpose(*dense_, {0, 2, 1});
        auto output = mlx::core::gather_mm(
            mlx::core::reshape(
                source, Shape{source_rows, 1, neuron_len()}),
            weights,
            lhs,
            rhs);
        return mlx::core::reshape(
            output,
            Shape{token_rows, routes_per_row, out_per_expert()});
    }

    std::optional<MlxMfeWeight> packed_;
    std::optional<MlxTqWeight> tq_, tq_up_;
    std::optional<array> dense_;
};


class GatedResidual {
public:
    MlxQwen4GatedResidualPre training_pre(const array& input, const MlxMtpLoraParameters& p) const {
        const auto cpu = p.cpu();
        const int h = config_.hidden_size, hc = config_.hc_count;
        auto source = mlx::core::reshape(input, {-1, hc, h}, cpu);
        auto normalized = mlx::core::divide(source, mlx::core::sqrt(mlx::core::add(
            mlx::core::mean(mlx::core::square(source, cpu), -1, true, cpu),
            array(float(config_.rms_norm_eps)), cpu), cpu), cpu);
        normalized = mlx::core::multiply(mlx::core::reshape(normalized, input.shape(), cpu),
            mlx::core::add(mlx::core::astype(norm_, mlx::core::float32, cpu), array(1.0f), cpu), cpu);
        auto low = mlx::core::divide(p.linear(normalized, *down_.dense_weight_ref()), array(float(hc)), cpu);
        low = mlx::core::multiply(low, mlx::core::sigmoid(low, cpu), cpu);
        auto mixing = mlx::core::sigmoid(p.linear(low, *up_.dense_weight_ref()), cpu);
        auto branch = mlx::core::mean(mlx::core::multiply(
            mlx::core::reshape(mixing, {-1, hc, h}, cpu),
            mlx::core::reshape(normalized, {-1, hc, h}, cpu), cpu), -2, false, cpu);
        auto shape = input.shape(); shape.back() = h;
        std::optional<array> injection;
        if (injection_) injection = mlx::core::multiply(array(2.0f), mlx::core::sigmoid(
            mlx::core::divide(p.linear(normalized, *injection_->dense_weight_ref()), array(float(hc)), cpu), cpu), cpu);
        return {mlx::core::reshape(branch, shape, cpu), input, std::move(injection)};
    }
    array training_post(const array& branch, const MlxQwen4GatedResidualPre& values,
        const MlxMtpLoraParameters& p) const {
        const auto cpu = p.cpu();
        auto update = mlx::core::multiply(mlx::core::expand_dims(branch, -2, cpu),
            mlx::core::expand_dims(*values.injection, -1, cpu), cpu);
        return mlx::core::add(values.residual, mlx::core::reshape(update, values.residual.shape(), cpu), cpu);
    }
    void prepare_training(MlxMtpLora& lora) const {
        mlx::core::eval(norm_);
        lora.prepare_linear(*down_.dense_weight_ref());
        lora.prepare_linear(*up_.dense_weight_ref());
        if (injection_) lora.prepare_linear(*injection_->dense_weight_ref());
    }
    static GatedResidual load(
        const MfqContainer& model,
        const Qwen4Config& config,
        const std::string& prefix,
        bool combine = true) {
        std::optional<MlxLinear> injection;
        if (combine) {
            auto root = prefix;
            if (root.size() < 4 || root.substr(root.size() - 4) != ".pre") {
                throw std::runtime_error("Qwen4 gated residual prefix is invalid");
            }
            root.resize(root.size() - 4);
            injection = mhc_projection(model, root + ".post.inject.weight");
        }
        return GatedResidual(
            config,
            dense_vector(model, prefix + ".norm.weight"),
            mhc_projection(model, prefix + ".down.weight"),
            mhc_projection(model, prefix + ".up.weight"),
            std::move(injection));
    }

    MlxQwen4GatedResidualPre pre(const array& value) const {
        return qwen4_gated_residual_pre(
            value, norm_, down_, up_, injection_,
            static_cast<int>(config_.hidden_size),
            static_cast<int>(config_.hc_count),
            static_cast<float>(config_.rms_norm_eps));
    }

    MlxQwen4GatedResidualPre pre_after(
        const array& branch,
        const MlxQwen4GatedResidualPre& previous) const {
        if (!previous.injection) {
            throw std::logic_error(
                "Qwen4 chained gated residual has no injection gate");
        }
        return qwen4_gated_residual_pre_after(
            branch,
            previous.residual,
            *previous.injection,
            norm_,
            down_,
            up_,
            injection_,
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
        MlxLinear down,
        MlxLinear up,
        std::optional<MlxLinear> injection)
        : config_(std::move(config)),
          norm_(std::move(norm)),
          down_(std::move(down)),
          up_(std::move(up)),
          injection_(std::move(injection)) {
        if (auto* preparation = MlxKernelPreparation::current()) {
            preparation->collect([&] {
                for (const auto rows : preparation->row_buckets()) {
                    for (const auto dtype : {mlx::core::float16, mlx::core::float32}) {
                        const auto input = mlx::core::zeros(
                            Shape{1, rows, static_cast<int>(config_.hidden_size * config_.hc_count)}, dtype);
                        const auto values = pre(input);
                        preparation->add(values.branch);
                        preparation->add(values.residual);
                        if (values.injection) {
                            preparation->add(*values.injection);
                            const auto branch = mlx::core::zeros(
                                Shape{1, rows, static_cast<int>(config_.hidden_size)}, dtype);
                            preparation->add(post(branch, values));
                            const auto after = pre_after(branch, values);
                            preparation->add(after.branch);
                            if (after.injection) preparation->add(*after.injection);
                        }
                    }
                }
            });
        }
    }

    Qwen4Config config_;
    array norm_;
    MlxLinear down_;
    MlxLinear up_;
    std::optional<MlxLinear> injection_;
};

class DenseFfn {
public:
    array training(const array& value, const MlxMtpLoraParameters& p) const {
        const auto cpu = p.cpu();
        auto gate = mlx::core::add(p.linear(value, *gate_.dense_weight_ref()), p.delta("shared.gate", value), cpu);
        auto up = mlx::core::add(p.linear(value, *up_.dense_weight_ref()), p.delta("shared.up", value), cpu);
        auto intermediate = mlx::core::multiply(mlx::core::multiply(gate, mlx::core::sigmoid(gate, cpu), cpu), up, cpu);
        return mlx::core::add(p.linear(intermediate, *down_.dense_weight_ref()), p.delta("shared.down", intermediate), cpu);
    }
    void prepare_training(MlxMtpLora& lora) const {
        for (const auto* weight : {&gate_, &up_, &down_}) lora.prepare_linear(*weight->dense_weight_ref());
    }
    static DenseFfn load(
        const MfqContainer& model,
        const std::string& prefix) {
        return DenseFfn(
            MlxLinear::load(model, prefix + ".gate.weight"),
            MlxLinear::load(model, prefix + ".up.weight"),
            MlxLinear::load(model, prefix + ".down.weight"));
    }

    array operator()(const array& value) const {
        if (auto* lora = MlxMtpLora::current(); lora && lora->active()) {
            auto gate = lora->apply("shared.gate", value, gate_(value));
            auto up = lora->apply("shared.up", value, up_(value));
            auto intermediate = gate * mlx::core::sigmoid(gate) * up;
            return lora->apply("shared.down", intermediate, down_(intermediate));
        }
        return down_(gate_up_.swiglu(value));
    }

    array moe_shared(
        const array& value,
        const array& routed_pairs,
        const array& route_weights,
        const array& gate_logits) const {
        return down_.moe_shared(
            gate_up_.swiglu(value),
            routed_pairs,
            route_weights,
            gate_logits);
    }

    const MlxNintSwiGluPair* nint_gate_up_pair_ref() const noexcept {
        return nint_gate_up_ ? &*nint_gate_up_ : nullptr;
    }

    const MlxNintWeight* nint_down_ref() const noexcept {
        return down_.nint_weight_ref();
    }

private:
    DenseFfn(MlxLinear gate, MlxLinear up, MlxLinear down)
        : gate_(std::move(gate)),
          up_(std::move(up)),
          down_(std::move(down)),
          gate_up_(std::vector<const MlxLinear*>{&gate_, &up_}) {
        const auto* gate_weight = gate_.nint_weight_ref();
        const auto* up_weight = up_.nint_weight_ref();
        if (gate_weight != nullptr && up_weight != nullptr) {
            nint_gate_up_ = MlxNintSwiGluPair::from_weights(
                *gate_weight, *up_weight);
        }
    }
    MlxLinear gate_;
    MlxLinear up_;
    MlxLinear down_;
    MlxProjectionBatch gate_up_;
    std::optional<MlxNintSwiGluPair> nint_gate_up_;
};

using Qwen4CachedRoute = MlxMfeCachedRoute;

class Qwen4Moe {
public:
    const std::optional<array>& training_routes() const { return training_routes_; }
    array training(const array& value, const array& ids, const MlxMtpLoraParameters& p) const {
        const auto cpu = p.cpu();
        const int h = config_.hidden_size, intermediate = config_.moe_intermediate_size;
        auto source = mlx::core::reshape(value, {1, h}, cpu);
        auto probabilities = mlx::core::softmax(p.linear(source, *router_.dense_weight_ref()), -1, cpu);
        auto route_weights = mlx::core::take(probabilities, mlx::core::astype(mlx::core::reshape(ids, {-1}, cpu),
            mlx::core::int32, cpu), -1, cpu);
        if (config_.norm_topk_prob) route_weights = mlx::core::divide(route_weights,
            mlx::core::sum(route_weights, -1, true, cpu), cpu);
        std::vector<array> outputs;
        const auto& gate_up = gate_up_->training_weight();
        const auto& down = down_->training_weight();
        for (int route = 0; route < ids.size(); ++route) {
            const int expert = static_cast<int>(ids.data<float>()[route]);
            if (expert < 0 || expert >= config_.num_experts) throw std::invalid_argument("invalid MTP training route");
            auto gate_weight = mlx::core::reshape(mlx::core::slice(gate_up,
                {expert, 0, 0}, {expert + 1, intermediate, h}, cpu), {intermediate, h}, cpu);
            auto up_weight = mlx::core::reshape(mlx::core::slice(gate_up,
                {expert, intermediate, 0}, {expert + 1, 2 * intermediate, h}, cpu), {intermediate, h}, cpu);
            const auto prefix = "expert." + std::to_string(expert) + ".";
            auto gate = mlx::core::add(p.linear(source, gate_weight), p.delta(prefix + "gate", source), cpu);
            auto up = mlx::core::add(p.linear(source, up_weight), p.delta(prefix + "up", source), cpu);
            auto act = mlx::core::multiply(mlx::core::multiply(gate, mlx::core::sigmoid(gate, cpu), cpu), up, cpu);
            auto down_weight = mlx::core::reshape(mlx::core::slice(down,
                {expert, 0, 0}, {expert + 1, h, intermediate}, cpu), {h, intermediate}, cpu);
            outputs.push_back(mlx::core::add(p.linear(act, down_weight), p.delta(prefix + "down", act), cpu));
        }
        auto pairs = mlx::core::concatenate(outputs, 0, cpu);
        auto routed = mlx::core::sum(mlx::core::multiply(pairs,
            mlx::core::transpose(route_weights, cpu), cpu), 0, true, cpu);
        auto shared = mlx::core::multiply(shared_.training(source, p),
            mlx::core::sigmoid(p.linear(source, *shared_gate_.dense_weight_ref()), cpu), cpu);
        return mlx::core::reshape(mlx::core::add(routed, shared, cpu), value.shape(), cpu);
    }
    void prepare_training(MlxMtpLora& lora) const {
        if (!gate_up_ || !down_) throw std::invalid_argument("MTP training does not support paged base experts");
        mlx::core::eval(gate_up_->training_weight(), down_->training_weight());
        const auto cpu = mlx::core::default_stream(mlx::core::Device(mlx::core::Device::cpu));
        lora.prepare_linear(mlx::core::reshape(mlx::core::slice(gate_up_->training_weight(),
            {0, 0, 0}, {1, int(config_.moe_intermediate_size), int(config_.hidden_size)}, cpu),
            {int(config_.moe_intermediate_size), int(config_.hidden_size)}, cpu));
        lora.prepare_linear(mlx::core::reshape(mlx::core::slice(down_->training_weight(),
            {0, 0, 0}, {1, int(config_.hidden_size), int(config_.moe_intermediate_size)}, cpu),
            {int(config_.hidden_size), int(config_.moe_intermediate_size)}, cpu));
        lora.prepare_linear(*router_.dense_weight_ref());
        lora.prepare_linear(*shared_gate_.dense_weight_ref());
        shared_.prepare_training(lora);
    }
    std::size_t resident_routed_bytes() const noexcept {
        return gate_up_ ? gate_up_->packed_bytes() + down_->packed_bytes() : 0;
    }
    void validate_offload(const std::shared_ptr<MlxMfeOffloadCache>& cache) const {
        for (const auto& name : {gate_name_, up_name_.value_or(gate_name_), down_name_}) {
            if (!cache->can_group_mfe(name)) throw std::runtime_error("expert projection cannot be offloaded: " + name);
        }
    }
    void use_offload(const std::shared_ptr<MlxMfeOffloadCache>& cache) {
        for (const auto& name : {gate_name_, up_name_.value_or(gate_name_), down_name_})
            cache->activate_record(name);
        gate_up_.reset();
        down_.reset();
        ssd_expert_cache_.reset();
        mfe_offload_cache_ = cache;
    }
    void validate_tq_offload(const std::shared_ptr<MlxTqExpertCache>& cache) const {
        if (!tq_expert_cache_ && (!gate_up_ || !gate_up_->uses_tq()))
            throw std::invalid_argument("TQ cache cannot replace non-TQ experts");
        cache->validate_layer(expert_cache_layer_, int(config_.num_experts),
            int(config_.hidden_size), int(config_.moe_intermediate_size));
    }
    void use_tq_offload(const std::shared_ptr<MlxTqExpertCache>& cache) {
        validate_tq_offload(cache);
        gate_up_.reset(); down_.reset();
        tq_expert_cache_ = cache;
    }
    static Qwen4Moe load(
        const MfqContainer& model,
        const Qwen4Config& config,
        const std::string& prefix,
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache = nullptr,
        std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache = nullptr,
        std::size_t expert_cache_layer = 0,
        std::shared_ptr<MlxTqExpertCache> tq_expert_cache = nullptr) {
        const auto expert_prefix = prefix + ".experts.";
        const bool split_gate_up = model.contains(
            expert_prefix + "gate.weight");
        const auto gate_name = split_gate_up
            ? expert_prefix + "gate.weight"
            : expert_prefix + "gate_up.weight";
        const auto up_name = split_gate_up
            ? std::optional<std::string>(expert_prefix + "up.weight")
            : std::nullopt;
        const auto down_name = expert_prefix + "down.weight";
        if (mfe_offload_cache) {
            for (const auto& name : {gate_name, up_name.value_or(gate_name), down_name}) {
                if (!mfe_offload_cache->can_group_mfe(name)) {
                    throw std::runtime_error(
                        "Qwen4 expert cache cannot page " + name +
                        "; refusing to bypass the configured cache budget");
                }
            }
        }
        std::optional<Qwen4RoutedWeight> gate_up;
        std::optional<Qwen4RoutedWeight> down;
        if (!ssd_expert_cache && !mfe_offload_cache && !tq_expert_cache) {
            gate_up.emplace(Qwen4RoutedWeight::load_gate_up(model, prefix));
            down.emplace(Qwen4RoutedWeight::load(
                model, prefix + ".experts.down.weight"));
        }
        return Qwen4Moe(
            config,
            std::move(gate_up),
            std::move(down),
            MlxLinear::load(model, prefix + ".router.weight"),
            DenseFfn::load(model, prefix + ".shared_expert"),
            MlxLinear::load(model, prefix + ".shared_expert.router.weight"),
            std::move(ssd_expert_cache),
            std::move(mfe_offload_cache),
            gate_name,
            up_name,
            down_name,
            expert_cache_layer,
            std::move(tq_expert_cache));
    }

    std::optional<MlxSsdPrefetchedExpertLayer> prefetch_routed(
        std::size_t rows) const {
        constexpr std::size_t kFullLayerPrefetchRows = 512;
        if (ssd_expert_cache_ && rows >= kFullLayerPrefetchRows &&
            ssd_expert_cache_->prefill_overlap_enabled()) {
            return ssd_expert_cache_->prefetch_layer(
                expert_cache_layer_);
        }
        return std::nullopt;
    }

    std::optional<Qwen4CachedRoute> cached_route() const {
        if (!mfe_offload_cache_) return std::nullopt;
        Qwen4CachedRoute route{mfe_offload_cache_, {}, std::nullopt};
        for (const auto& name : {gate_name_, up_name_.value_or(down_name_), down_name_}) {
            if (route.projections.size() == 2 && !up_name_) break;
            auto projection = mfe_offload_cache_->cached_projection(name);
            if (!projection) return std::nullopt;
            route.projections.push_back(std::move(projection));
        }
        return route;
    }

    array operator()(
        const array& value,
        MlxSsdPrefetchedExpertLayer* prefetched = nullptr,
        bool batched_decode = false,
        Qwen4CachedRoute* deferred_route = nullptr) const {
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
            false, false, config_.norm_topk_prob, false, std::nullopt, std::nullopt, 1e-20f, 1.f,
            !(tq_expert_cache_ || (gate_up_ && gate_up_->uses_tq())));
        const auto* route_groups = routes.groups ? &*routes.groups : nullptr;
        if (auto* lora = MlxMtpLora::current()) {
            if (lora->collecting()) training_routes_ = routes.ids;
            if (lora->active()) {
                auto pair = gate_up_->routed_matmul(source, routes.ids, route_groups);
                auto parts = mlx::core::split(pair, 2, -1);
                auto gate = lora->routed_apply("gate", source, parts[0], routes.ids);
                auto up = lora->routed_apply("up", source, parts[1], routes.ids);
                auto intermediate = gate * mlx::core::sigmoid(gate) * up;
                auto pairs = lora->routed_apply("down", intermediate,
                    down_->routed_matmul(intermediate, routes.ids, route_groups), routes.ids);
                return mlx::core::reshape(moe_weighted_reduce_shared_gate(pairs, routes.weights,
                    shared_(source), shared_gate_(source)), value.shape());
            }
        }
        if (detail::component_profile_active()) {
            detail::profile_eval(
                "qwen4.moe.topk",
                std::vector<array>{routes.ids, routes.weights});
        }
        std::optional<array> shared_gate_logits;
        if (source.shape(0) >= 1 && (source.shape(0) <= 6 || batched_decode) && config_.norm_topk_prob &&
            gate_up_ && down_) {
            const auto* gate_up = gate_up_->packed_weight_ref();
            const auto* down = down_->packed_weight_ref();
            const auto* shared_gate_up = shared_.nint_gate_up_pair_ref();
            const auto* shared_down = shared_.nint_down_ref();
            const auto* shared_gate_weight =
                shared_gate_.dense_weight_ref();
            if (gate_up != nullptr && down != nullptr &&
                shared_gate_up != nullptr && shared_down != nullptr) {
                auto shared_gate_input = packed_shared_gate_ ? *packed_shared_gate_ : shared_gate_weight != nullptr
                    ? *shared_gate_weight
                    : shared_gate_logits.emplace(shared_gate_(source));
                if (auto fused = gate_up->decode_nint_shared(
                        *down,
                        *shared_gate_up,
                        *shared_down,
                        shared_gate_input,
                        source,
                        routes.ids,
                        routes.weights,
                        shared_gate_weight == nullptr, route_groups, &router_logits)) {
                    if (detail::component_profile_active()) {
                        detail::profile_eval(
                            "qwen4.moe.two_stage", *fused);
                    }
                    return mlx::core::reshape(
                        std::move(*fused), value.shape());
                }
            }
        }
        auto shared_gate = shared_gate_logits
            ? std::move(*shared_gate_logits) : shared_gate_(source);
        std::optional<array> inverse_route_order;
        bool pairs_are_sorted = false;
        bool routed_is_reduced = false;
        std::optional<array> parallel_shared;
        array pairs = [&] {
            const int tokens = source.shape(0);
            if (tq_expert_cache_) {
                routed_is_reduced = tokens <= 6;
                if (routed_is_reduced) parallel_shared = shared_(source);
                return tq_expert_cache_->apply(expert_cache_layer_, routes.ids, [&](const auto& weights, const array* reuse) {
                    auto intermediate = weights.gate.swiglu(weights.up, source, routes.ids, 0.f, false, reuse);
                    detail::profile_eval("qwen4.moe.routed_gate_up", intermediate);
                    return intermediate;
                }, [&](const auto& weights, const array& intermediate, const array* reuse) {
                    auto output = routed_is_reduced
                        ? weights.down.routed_matmul_reduce(intermediate, routes.ids, routes.weights, reuse)
                        : weights.down.routed_matmul(intermediate, routes.ids, reuse);
                    detail::profile_eval("qwen4.moe.routed_down", output);
                    return output;
                }, parallel_shared ? &*parallel_shared : nullptr);
            }
            const int route_count = checked_int(
                static_cast<std::size_t>(tokens)
                    * static_cast<std::size_t>(routes.ids.shape(1)),
                "Qwen4 routed row count");
            if (prefetched != nullptr) {
                if (!ssd_expert_cache_ ||
                    prefetched->layer() != expert_cache_layer_) {
                    throw std::invalid_argument(
                        "Qwen4 SSD prefetch layer mismatch");
                }
                const auto& weights = prefetched->wait();
                auto intermediate = weights.gate_up.swiglu(
                    source, routes.ids, 0.0f);
                if (detail::component_profile_active()) {
                    detail::profile_eval(
                        "qwen4.moe.routed_gate_up",
                        intermediate);
                }
                auto output = tokens <= 6
                    ? weights.down.combine(
                          intermediate,
                          routes.ids,
                          routes.weights)
                    : weights.down(intermediate, routes.ids);
                routed_is_reduced = tokens <= 6;
                if (detail::component_profile_active()) {
                    detail::profile_eval(
                        "qwen4.moe.routed_down", output);
                }
                return output;
            }
            if (ssd_expert_cache_) {
                auto prepared = ssd_expert_cache_->prepare_routes(
                    expert_cache_layer_, routes.ids);
                auto intermediate =
                    prepared.weights().gate_up.swiglu(
                        source, prepared.expert_ids(), 0.0f);
                if (detail::component_profile_active()) {
                    detail::profile_eval(
                        "qwen4.moe.routed_gate_up",
                        intermediate);
                }
                auto output = tokens <= 6
                    ? prepared.weights().down.combine(
                          intermediate,
                          prepared.expert_ids(),
                          routes.weights)
                    : prepared.weights().down(
                          intermediate, prepared.expert_ids());
                routed_is_reduced = tokens <= 6;
                if (detail::component_profile_active()) {
                    detail::profile_eval(
                        "qwen4.moe.routed_down", output);
                }
                return output;
            }
            if (mfe_offload_cache_) {
                if (tokens == 1 && !detail::component_profile_active()) {
                    auto local_route = deferred_route ? std::nullopt : cached_route();
                    auto* cached = deferred_route ? deferred_route : local_route ? &*local_route : nullptr;
                    if (cached) {
                        auto output = cached->forward(source, routes.ids);
                        if (deferred_route) return output;
                        mlx::core::async_eval({output});
                        if (cached->resolve()) return output;
                    }
                }
                auto global_ids = mlx::core::contiguous(
                    mlx::core::astype(routes.ids, mlx::core::int32));
                global_ids.eval();
                std::vector<std::int32_t> active;
                active.reserve(static_cast<std::size_t>(global_ids.size()));
                std::vector<std::uint32_t> seen(
                    static_cast<std::size_t>(config_.num_experts), 0);
                const auto* ids = global_ids.data<std::int32_t>();
                for (std::size_t index = 0; index < global_ids.size(); ++index) {
                    const auto expert = ids[index];
                    if (expert < 0 || expert >= config_.num_experts) {
                        throw std::runtime_error(
                            "Qwen4 routed expert ID is out of range");
                    }
                    if (seen[static_cast<std::size_t>(expert)] == 0) {
                        active.push_back(expert);
                    }
                    ++seen[static_cast<std::size_t>(expert)];
                }
                if (tokens > 6 && !batched_decode) {
                    std::vector<std::string> names{gate_name_, down_name_};
                    if (up_name_) names.push_back(*up_name_);
                    if (auto layer = mfe_offload_cache_->layer_mfe(names, active)) {
                        auto intermediate = up_name_
                            ? (*layer)[0].routed_swiglu_pair((*layer)[2], source, global_ids)
                            : (*layer)[0].routed_swiglu(source, global_ids);
                        return (*layer)[1].routed_matmul(intermediate, global_ids);
                    }
                }
                std::vector<std::int32_t> global_to_local(
                    static_cast<std::size_t>(config_.num_experts), -1);
                for (std::size_t local = 0; local < active.size(); ++local) {
                    global_to_local[static_cast<std::size_t>(active[local])] =
                        static_cast<std::int32_t>(local);
                }
                std::vector<std::int32_t> local_routes(global_ids.size());
                for (std::size_t index = 0; index < local_routes.size(); ++index)
                    local_routes[index] = global_to_local[ids[index]];
                auto local_ids = array(local_routes.begin(), global_ids.shape(), mlx::core::int32);
                std::vector<std::uint32_t> activation_counts;
                activation_counts.reserve(active.size());
                for (const auto expert : active) activation_counts.push_back(seen[expert]);
                std::optional<array> groups;
                if (tokens > 1 && (tokens <= 6 || (batched_decode && tokens <= 24))) {
                    groups = moe_route_groups_host(local_routes, tokens, global_ids.shape(1));
                }
                const auto* local_groups = groups ? &*groups : nullptr;
                std::optional<MlxMfeWeight> up;
                const bool prefill_scan = tokens > 6 && !batched_decode;
                auto gate = [&] {
                    if (!up_name_) return mfe_offload_cache_->grouped_mfe(gate_name_, active, activation_counts, prefill_scan);
                    const std::array names{gate_name_, *up_name_};
                    auto weights = mfe_offload_cache_->grouped_mfe(names, active, activation_counts, prefill_scan);
                    up.emplace(std::move(weights[1]));
                    return std::move(weights[0]);
                }();
                std::optional<array> sorted_routes;
                std::optional<MlxGroupedMmqPlan> gate_plan;
                if (prefill_scan && tokens >= 32 && gate.supports_grouped_mmq()
                    && (!up || up->supports_grouped_mmq())) {
                    sorted_routes = mlx::core::contiguous(mlx::core::astype(
                        mlx::core::argsort(mlx::core::reshape(local_ids, Shape{route_count})), mlx::core::int32));
                    gate_plan = gate.build_grouped_mmq_plan(local_ids, *sorted_routes,
                        gate.recommended_grouped_mmq_block_rows(route_count, !up));
                }
                auto intermediate = [&]() {
                    if (sorted_routes) {
                        auto result = gate.routed_matmul_sorted(source, local_ids,
                            *sorted_routes, false, !up, 0.0f, &*gate_plan);
                        if (!up) return result;
                        const int rows = up->recommended_grouped_mmq_block_rows(route_count, false);
                        const auto plan = rows == gate_plan->block_rows ? *gate_plan
                            : up->build_grouped_mmq_plan(local_ids, *sorted_routes, rows);
                        return moe_swiglu_pair(std::move(result), up->routed_matmul_sorted(
                            source, local_ids, *sorted_routes, false, false, 0.0f, &plan));
                    }
                    if (up) {
                        return gate.routed_swiglu_pair(
                            *up, source, local_ids, 0.0f, local_groups);
                    }
                    return gate.routed_swiglu(source, local_ids, 0.0f, local_groups);
                }();
                if (detail::component_profile_active()) {
                    detail::profile_eval(
                        "qwen4.moe.routed_gate_up", intermediate);
                }
                if ((tokens <= 6 || batched_decode || sorted_routes) && !detail::component_profile_active())
                    mlx::core::async_eval({intermediate});
                auto down = mfe_offload_cache_->grouped_mfe(
                    down_name_, active, activation_counts, prefill_scan);
                if (sorted_routes && down.supports_grouped_mmq()) {
                    const int rows = down.recommended_grouped_mmq_block_rows(route_count, false);
                    const auto plan = rows == gate_plan->block_rows ? *gate_plan
                        : down.build_grouped_mmq_plan(local_ids, *sorted_routes, rows);
                    auto output = down.routed_matmul_sorted(intermediate, local_ids,
                        *sorted_routes, true, false, 0.0f, &plan);
                    if (detail::component_profile_active())
                        detail::profile_eval("qwen4.moe.routed_down", output);
                    return mlx::core::reshape(mlx::core::take(output,
                        moe_inverse_permutation(*sorted_routes), 0),
                        Shape{tokens, global_ids.shape(1), down.out_per_expert()});
                }
                if (sorted_routes) intermediate = mlx::core::reshape(mlx::core::take(
                    intermediate, moe_inverse_permutation(*sorted_routes), 0),
                    Shape{tokens, global_ids.shape(1), down.neuron_len()});
                const bool combine_routes = tokens <= 6 &&
                    down.supports_fused_routed_reduce(local_groups);
                auto output = combine_routes
                    ? down.routed_matmul_reduce(
                          intermediate,
                          local_ids,
                          routes.weights, local_groups)
                    : down.routed_matmul(intermediate, local_ids, local_groups);
                routed_is_reduced = combine_routes;
                if (detail::component_profile_active()) {
                    detail::profile_eval(
                        "qwen4.moe.routed_down", output);
                }
                return output;
            }
            if (tokens >= 32
                && gate_up_->supports_grouped_mmq()
                && down_->supports_grouped_mmq()) {
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
                    gate_up_->recommended_grouped_mmq_block_rows(
                        route_count,
                        true);
                const int down_block_rows =
                    down_->recommended_grouped_mmq_block_rows(
                        route_count,
                        false);
                auto gate_plan = gate_up_->build_grouped_mmq_plan(
                    routes.ids,
                    route_order,
                    gate_block_rows);
                std::optional<mfq::metal::MlxGroupedMmqPlan> down_plan;
                if (down_block_rows != gate_block_rows) {
                    down_plan.emplace(
                        down_->build_grouped_mmq_plan(
                            routes.ids,
                            route_order,
                            down_block_rows));
                }
                auto intermediate = gate_up_->routed_matmul_sorted(
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
                auto sorted = down_->routed_matmul_sorted(
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
                inverse_route_order =
                    moe_inverse_permutation(route_order);
                pairs_are_sorted = true;
                return sorted;
            }
            auto intermediate = gate_up_->routed_swiglu(
                source,
                routes.ids, route_groups);
            if (detail::component_profile_active()) {
                detail::profile_eval(
                    "qwen4.moe.routed_gate_up",
                    intermediate);
            }
            // If down has no fused reduction, leave the pairs for the
            // reduce/shared-gate epilogue instead of separate reduction and
            // shared-gate launches.
            const bool combine_routes = tokens <= 6 &&
                down_->supports_fused_routed_reduce(route_groups);
            auto output = combine_routes
                ? down_->routed_matmul_reduce(
                      intermediate,
                      routes.ids,
                      routes.weights, route_groups)
                : down_->routed_matmul(intermediate, routes.ids, route_groups);
            routed_is_reduced = combine_routes;
            if (detail::component_profile_active()) {
                detail::profile_eval("qwen4.moe.routed_down", output);
            }
            return output;
        }();
        if (!detail::component_profile_active() &&
            !routed_is_reduced && !pairs_are_sorted) {
            return mlx::core::reshape(
                shared_.moe_shared(
                    source,
                    pairs,
                    routes.weights,
                    shared_gate),
                value.shape());
        }
        auto shared = parallel_shared ? *parallel_shared : shared_(source);
        if (detail::component_profile_active()) {
            detail::profile_eval("qwen4.moe.shared", shared);
        }
        if (detail::component_profile_active()) {
            detail::profile_eval("qwen4.moe.shared_gate", shared_gate);
        }
        auto output = routed_is_reduced
            ? moe_add_shared_gate(
                  pairs,
                  shared,
                  shared_gate)
            : pairs_are_sorted
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
        std::optional<Qwen4RoutedWeight> gate_up,
        std::optional<Qwen4RoutedWeight> down,
        MlxLinear router,
        DenseFfn shared,
        MlxLinear shared_gate,
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache,
        std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache,
        std::string gate_name,
        std::optional<std::string> up_name,
        std::string down_name,
        std::size_t expert_cache_layer,
        std::shared_ptr<MlxTqExpertCache> tq_expert_cache)
        : config_(std::move(config)),
          gate_up_(std::move(gate_up)),
          down_(std::move(down)),
          router_(std::move(router)),
          shared_(std::move(shared)),
          shared_gate_(std::move(shared_gate)),
          ssd_expert_cache_(std::move(ssd_expert_cache)),
          mfe_offload_cache_(std::move(mfe_offload_cache)),
          tq_expert_cache_(std::move(tq_expert_cache)),
          gate_name_(std::move(gate_name)),
          up_name_(std::move(up_name)),
          down_name_(std::move(down_name)),
          expert_cache_layer_(expert_cache_layer) {
        const bool cached = static_cast<bool>(ssd_expert_cache_)
            || static_cast<bool>(mfe_offload_cache_) || static_cast<bool>(tq_expert_cache_);
        if (shared_.nint_gate_up_pair_ref())
            if (const auto* weight = shared_gate_.nint_weight_ref())
                packed_shared_gate_ = pack_nint_single_row(*weight);
        if (gate_up_.has_value() != down_.has_value() ||
            gate_up_.has_value() == cached
            || (int(bool(ssd_expert_cache_)) + int(bool(mfe_offload_cache_)) + int(bool(tq_expert_cache_)) > 1)) {
            throw std::runtime_error(
                "Qwen4 routed expert backing is inconsistent");
        }
        if (gate_up_ && (
            gate_up_->experts() != config_.num_experts ||
            down_->experts() != config_.num_experts ||
            gate_up_->neuron_len() != config_.hidden_size ||
            (gate_up_->projections() != 1 && gate_up_->projections() != 2) ||
            gate_up_->out_per_expert() * gate_up_->projections() !=
                2 * config_.moe_intermediate_size ||
            down_->neuron_len() != config_.moe_intermediate_size ||
            down_->out_per_expert() != config_.hidden_size)) {
            throw std::runtime_error("Qwen4 routed expert geometry disagrees");
        }
        if (mfe_offload_cache_) {
            for (const auto& name : {gate_name_, up_name_.value_or(gate_name_), down_name_})
                mfe_offload_cache_->activate_record(name);
            mfe_offload_cache_->prepare_projection(gate_name_, true, !up_name_);
            if (up_name_) mfe_offload_cache_->prepare_projection(*up_name_, true);
            mfe_offload_cache_->prepare_projection(down_name_, false);
            if (up_name_) mfe_offload_cache_->prepare_swiglu_pair(
                gate_name_, *up_name_, static_cast<int>(config_.num_experts_per_tok));
        }
        if (tq_expert_cache_)
            tq_expert_cache_->prepare_layer(expert_cache_layer_, static_cast<int>(config_.num_experts_per_tok));
        if (auto* preparation = MlxKernelPreparation::current(); preparation && cached) {
            preparation->collect([&] {
                auto routes = moe_topk(router_(mlx::core::zeros(
                    Shape{8, static_cast<int>(config_.hidden_size)}, mlx::core::float16)),
                    static_cast<int>(config_.num_experts_per_tok), false, false, config_.norm_topk_prob,
                    false, std::nullopt, std::nullopt, 1e-20f, 1.f, !tq_expert_cache_);
                preparation->add(routes.ids);
                preparation->add(routes.weights);
                if (routes.groups) preparation->add(*routes.groups);
            });
        }
        if (auto* preparation = MlxKernelPreparation::current(); preparation && !cached) {
            preparation->collect([&] {
                for (const auto rows : preparation->row_buckets())
                    preparation->add((*this)(mlx::core::zeros(
                        Shape{1, rows, static_cast<int>(config_.hidden_size)}, mlx::core::float16)));
                preparation->add((*this)(mlx::core::zeros(
                    Shape{1, 8, static_cast<int>(config_.hidden_size)}, mlx::core::float16), nullptr, true));
            });
        }
    }

    Qwen4Config config_;
    std::optional<Qwen4RoutedWeight> gate_up_;
    std::optional<Qwen4RoutedWeight> down_;
    mutable std::optional<array> training_routes_;
    MlxLinear router_;
    DenseFfn shared_;
    MlxLinear shared_gate_;
    std::optional<array> packed_shared_gate_;
    std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache_;
    std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache_;
    std::shared_ptr<MlxTqExpertCache> tq_expert_cache_;
    std::string gate_name_;
    std::optional<std::string> up_name_;
    std::string down_name_;
    std::size_t expert_cache_layer_ = 0;
};

class Qwen4NgramEmbedding {
private:
    struct Shard {
        MfqMappedBytes mapping;
        const std::uint8_t* values = nullptr;
        std::int64_t rows = 0;
        std::int64_t width = 0;
        std::optional<MlxMappedNintRows> nint;
        std::optional<MlxMappedTqRows> tq;
    };

public:
    struct Lookup {
        int batch = 0;
        int tokens = 0;
        int heads = 0;
        std::vector<std::int64_t> rows;
        std::vector<std::int64_t> context;
        MlxNintRowBatch selected;
        MlxTqRowBatch tq;
        std::vector<mlx::core::float16_t> values;
    };
    using Prefetched = std::future<Lookup>;

    static Qwen4NgramEmbedding load(
        const MfqContainer& model,
        const Qwen4Config& config,
        const std::string& prefix) {
        std::vector<Shard> shards;
        shards.reserve(static_cast<std::size_t>(config.split_ngram_parts));
        std::int64_t rows = 0;
        std::int64_t width = 0;
        std::optional<int> encoding;
        for (std::int64_t index = 0;
             index < config.split_ngram_parts; ++index) {
            const auto name = prefix + ".ngram.shard." +
                std::to_string(index) + ".weight";
            const auto& record = model.record(name);
            const bool nint = is_nint_dtype(record.dtype);
            const bool tq = record.dtype == "TQ";
            if (!nint && !tq && record.dtype != "F8_E4M3") {
                throw std::runtime_error(
                    "unsupported Qwen4 PLE row format: " + name);
            }
            const int selected_encoding = tq ? 2 : nint ? 1 : 0;
            if (encoding && *encoding != selected_encoding) {
                throw std::runtime_error("Qwen4 PLE shard formats differ");
            }
            encoding = selected_encoding;
            auto mapping = model.map_record(name, true);
            const auto bytes = mapping.view();
            if (tq) {
                const auto heads = (config.ngram_size - 1) * config.heads_per_ngram;
                if (heads <= 0 || config.hidden_size % heads)
                    throw std::runtime_error("Qwen4 PLE head width disagrees");
                MlxMappedTqRows table(bytes, config.hidden_size / heads);
                if (!shards.empty() && (table.rows() != rows || table.width() != width))
                    throw std::runtime_error("Qwen4 PLE shards have different shapes");
                rows = table.rows(); width = table.width();
                shards.push_back({std::move(mapping), nullptr, rows, width, std::nullopt, std::move(table)});
                continue;
            }
            if (nint) {
                MlxMappedNintRows table(bytes);
                if (!shards.empty() && (table.rows() != rows || table.width() != width)) {
                    throw std::runtime_error("Qwen4 PLE shards have different shapes");
                }
                rows = table.rows();
                width = table.width();
                shards.push_back({std::move(mapping), nullptr, rows, width, std::move(table)});
                continue;
            }
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
            shards.push_back({std::move(mapping), values, rows, width, std::nullopt});
        }
        const auto metadata = prefix + ".ngram";
        return Qwen4NgramEmbedding(
            config,
            std::move(shards),
            rows,
            width,
            encoding.value_or(0) ? 1.0f : scalar_float(model, metadata + ".weight_scale"),
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

    const std::vector<std::int64_t>& context() const noexcept {
        return context_;
    }

    void exchange_context(Qwen4RequestLayerCache& cache) {
        std::swap(batch_, cache.ngram_batch);
        context_.swap(cache.ple_context);
    }

    void restore_context(std::vector<std::int64_t> context) {
        if (context.size() != context_.size()) {
            throw std::runtime_error(
                "Qwen4 PLE ngram rollback context disagrees");
        }
        context_ = std::move(context);
    }

    void advance_context(const array& token_ids, int retained_tokens) {
        if (token_ids.ndim() != 2 || token_ids.shape(0) != batch_ ||
            retained_tokens <= 0 || retained_tokens > token_ids.shape(1) ||
            context_.empty()) {
            throw std::runtime_error(
                "Qwen4 PLE ngram retained prefix disagrees");
        }
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
        const int source_tokens = token_ids.shape(1);
        const int prefix = static_cast<int>(config_.ngram_size - 1);
        std::vector<std::int64_t> history(
            static_cast<std::size_t>(prefix + retained_tokens));
        for (int bi = 0; bi < batch_; ++bi) {
            std::copy_n(
                context_.data() + static_cast<std::size_t>(bi * prefix),
                prefix,
                history.data());
            for (int token = 0; token < retained_tokens; ++token) {
                const auto source_index = static_cast<std::size_t>(
                    bi * source_tokens + token);
                history[static_cast<std::size_t>(prefix + token)] = int32_ids
                    ? static_cast<std::int64_t>(source32[source_index])
                    : source64[source_index];
            }
            std::copy_n(
                history.data() + retained_tokens,
                prefix,
                context_.data() + static_cast<std::size_t>(bi * prefix));
        }
    }

    Prefetched prefetch(const array& token_ids, bool use_cache) {
        return std::async(std::launch::async,
            [this, lookup = select_rows(token_ids, use_cache)]() mutable {
                return read_rows(std::move(lookup), true);
            });
    }

    Prefetched prefetch_requests(const array& ids, const std::vector<int>& offsets,
        const std::vector<Qwen4RequestLayerCache*>& caches) {
        std::vector<std::int64_t> contexts;
        contexts.reserve(caches.size() * static_cast<std::size_t>(config_.ngram_size - 1));
        for (const auto* cache : caches) {
            if (cache->ple_context.empty() && cache->ngram_batch == 0)
                contexts.insert(contexts.end(), static_cast<std::size_t>(config_.ngram_size - 1), config_.eos_token_id);
            else contexts.insert(contexts.end(), cache->ple_context.begin(), cache->ple_context.end());
        }
        return std::async(std::launch::async,
            [this, lookup = select_rows(ids, true, &contexts, &offsets)]() mutable {
                return read_rows(std::move(lookup), true);
            });
    }

    array forward_requests(const array& ids, const std::vector<int>& offsets,
        const std::vector<Qwen4RequestLayerCache*>& caches, Prefetched* prefetched) {
        auto pending = prefetched && prefetched->valid()
            ? std::move(*prefetched) : prefetch_requests(ids, offsets, caches);
        auto lookup = pending.get();
        if (lookup.batch != ids.shape(0) || lookup.tokens != ids.shape(1))
            throw std::runtime_error("Qwen4 PLE prefetched request shape disagrees");
        const int prefix = static_cast<int>(config_.ngram_size - 1);
        if (lookup.context.size() != caches.size() * static_cast<std::size_t>(prefix))
            throw std::runtime_error("Qwen4 PLE prefetched request contexts disagree");
        for (int row = 0; row < static_cast<int>(caches.size()); ++row) {
            caches[row]->ngram_batch = 1;
            caches[row]->ple_context.assign(lookup.context.begin() + row * prefix,
                lookup.context.begin() + (row + 1) * prefix);
        }
        return decode_lookup(std::move(lookup));
    }

    array forward(
        const array& token_ids,
        bool use_cache,
        Prefetched* prefetched = nullptr) {
        auto pending = prefetched && prefetched->valid()
            ? std::move(*prefetched) : prefetch(token_ids, use_cache);
        auto lookup = pending.get();
        if (lookup.batch != token_ids.shape(0) || lookup.tokens != token_ids.shape(1)) {
            throw std::runtime_error("Qwen4 PLE prefetched shape disagrees");
        }
        if (use_cache) {
            batch_ = lookup.batch;
            context_ = std::move(lookup.context);
        }
        return decode_lookup(std::move(lookup));
    }

private:
    array decode_lookup(Lookup lookup) const {
        const Shape shape{lookup.batch, lookup.tokens,
            lookup.heads * static_cast<int>(width_)};
        if (shards_.front().tq) return mlx::core::reshape(lookup.tq.decode(), shape);
        if (shards_.front().nint) {
            return mlx::core::reshape(lookup.selected.decode(), shape);
        }
        return array(lookup.values.begin(), shape);
    }

    Lookup select_rows(const array& token_ids, bool use_cache,
        const std::vector<std::int64_t>* contexts = nullptr,
        const std::vector<int>* request_offsets = nullptr) {
        if (token_ids.ndim() != 2 || token_ids.shape(0) <= 0 ||
            token_ids.shape(1) <= 0) {
            throw std::runtime_error("Qwen4 PLE IDs must have [B,T] shape");
        }
        const int batch = request_offsets ? static_cast<int>(request_offsets->size()) - 1 : token_ids.shape(0);
        const int tokens = token_ids.shape(1);
        if (request_offsets && (batch <= 0 || token_ids.shape(0) != 1 ||
            request_offsets->front() != 0 || request_offsets->back() != tokens || !contexts))
            throw std::runtime_error("Qwen4 PLE request offsets disagree");
        if (request_offsets) for (int row = 0; row < batch; ++row)
            if ((*request_offsets)[row] >= (*request_offsets)[row + 1])
                throw std::runtime_error("Qwen4 PLE request windows must be nonempty");
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
        const int heads = static_cast<int>(
            (config_.ngram_size - 1) * config_.heads_per_ngram);
        if (contexts && contexts->size() != static_cast<std::size_t>(batch * prefix))
            throw std::runtime_error("Qwen4 PLE request context shape disagrees");
        if (use_cache && !contexts && (batch_ != batch || context_.empty())) reset(batch);
        const auto& cached = contexts ? *contexts : context_;
        Lookup lookup;
        lookup.batch = token_ids.shape(0);
        lookup.tokens = tokens;
        lookup.heads = heads;
        lookup.rows.resize(token_ids.size() * static_cast<std::size_t>(heads));
        if (use_cache) lookup.context.resize(static_cast<std::size_t>(batch * prefix));
        auto& global = lookup.rows;
        for (int bi = 0; bi < batch; ++bi) {
            const int start = request_offsets ? (*request_offsets)[bi] : bi * tokens;
            const int count = request_offsets ? (*request_offsets)[bi + 1] - start : tokens;
            const int length = prefix + count;
            std::vector<std::int64_t> history(static_cast<std::size_t>(length), config_.eos_token_id);
            if (use_cache) {
                std::copy_n(
                    cached.data() + static_cast<std::size_t>(bi * prefix),
                    prefix,
                    history.data());
            }
            for (int token = 0; token < count; ++token) {
                const auto source_index =
                    static_cast<std::size_t>(start + token);
                history[static_cast<std::size_t>(
                    prefix + token)] = int32_ids
                    ? static_cast<std::int64_t>(source32[source_index])
                    : source64[source_index];
            }
            int segment = 0;
            for (int token = 0; token < length; ++token) {
                std::uint64_t mixed =
                    static_cast<std::uint64_t>(history[token]) *
                    static_cast<std::uint64_t>(multipliers_[0]);
                for (int shift = 1; shift < config_.ngram_size; ++shift) {
                    const auto previous = token - shift >= segment
                        ? history[token - shift]
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
                            global[(start + token - prefix) * heads + h] =
                                remainder + offsets_[h];
                        }
                    }
                }
                if (history[token] == config_.eos_token_id) {
                    segment = token + 1;
                }
            }
            if (use_cache) {
                std::copy_n(
                    history.data() + static_cast<std::size_t>(count),
                    prefix,
                    lookup.context.data() + static_cast<std::size_t>(bi * prefix));
            }
        }
        for (const auto row : global) {
            if (row < 0 || row >= rows_ * static_cast<std::int64_t>(shards_.size())) {
                throw std::runtime_error("Qwen4 PLE hash is outside embedding table");
            }
        }
        return lookup;
    }

    Lookup read_rows(Lookup lookup, bool parallel = false) const {
        const auto& global = lookup.rows;
        if (shards_.front().nint || shards_.front().tq) {
            if (parallel) {
                struct Row {
                    MlxNintRowBatch batch;
                    MlxTqRowBatch tq;
                    std::exception_ptr error;
                };
                const auto task_rows = shards_.front().tq
                    ? std::clamp(global.size() / 256, std::size_t{1}, std::size_t{64}) : std::size_t{1};
                std::vector<Row> rows((global.size() + task_rows - 1) / task_rows);
                struct Work {
                    const Qwen4NgramEmbedding* embedding;
                    const std::int64_t* ids;
                    Row* rows;
                    std::size_t count, task_rows;
                } work{this, global.data(), rows.data(), global.size(), task_rows};
                dispatch_apply_f(rows.size(),
                    dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), &work,
                    [](void* context, std::size_t index) {
                        const auto& work = *static_cast<const Work*>(context);
                        const auto& embedding = *work.embedding;
                        try {
                            const auto end = std::min(work.count, (index + 1) * work.task_rows);
                            for (auto at = index * work.task_rows; at < end; ++at) {
                                const auto row = work.ids[at];
                                const auto shard = static_cast<std::size_t>(row / embedding.rows_);
                                const auto& source = embedding.shards_[shard];
                                if (source.tq) source.tq->append_row(row % embedding.rows_, work.rows[index].tq);
                                else source.nint->append_row(row % embedding.rows_, work.rows[index].batch);
                            }
                        } catch (...) {
                            work.rows[index].error = std::current_exception();
                        }
                    });
                for (const auto& row : rows) {
                    if (row.error) std::rethrow_exception(row.error);
                    if (row.batch.rows()) lookup.selected.append_batch(row.batch);
                    lookup.tq.append_batch(row.tq);
                }
            } else {
                for (const auto row : global) {
                    const auto shard = static_cast<std::size_t>(row / rows_);
                    if (shards_[shard].tq) shards_[shard].tq->append_row(row % rows_, lookup.tq);
                    else shards_[shard].nint->append_row(row % rows_, lookup.selected);
                }
            }
            MlxResourceTelemetry::ple_read_counter().fetch_add(
                lookup.selected.source_bytes_read() + lookup.tq.source_bytes_read(), std::memory_order_relaxed);
        } else {
            lookup.values.resize(global.size() * static_cast<std::size_t>(width_));
            for (std::size_t index = 0; index < global.size(); ++index) {
                const auto row = global[index];
                const auto shard = static_cast<std::size_t>(row / rows_);
                const auto local = row % rows_;
                const auto* values = shards_[shard].values + local * width_;
                auto* destination = lookup.values.data() + index * static_cast<std::size_t>(width_);
                for (std::int64_t column = 0; column < width_; ++column) {
                    destination[column] = e4m3_lut_[values[column]];
                }
            }
            MlxResourceTelemetry::ple_read_counter().fetch_add(
                global.size() * static_cast<std::size_t>(width_), std::memory_order_relaxed);
        }
        return lookup;
    }

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
        std::size_t payload = 0;
        for (const auto& shard : shards_) payload += shard.mapping.size();
        resources_.set({0, 0, 0, 0, payload});
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
    MlxResourceTelemetry resources_;
};

class Qwen4Ple {
public:
    void exchange_cache(Qwen4RequestLayerCache& cache) {
        rollback_.swap(cache.ple_rollback);
        std::swap(batch_, cache.ple_batch);
        convolution_state_.swap(cache.ple_convolution);
        std::swap(resources_, cache.ple_resources);
        embedding_.exchange_context(cache);
    }
    void snapshot(MlxQwen4LayerCacheSnapshot& state) const {
        if (!convolution_state_ || rollback_) throw std::runtime_error("PLE checkpoint is not committed");
        state.ple_convolution = detached_copy(*convolution_state_);
        state.ple_convolution->eval();
        state.ple_context = embedding_.context();
    }

    void restore(const MlxQwen4LayerCacheSnapshot& state) {
        if (!state.ple_convolution || state.ple_context.size() !=
            static_cast<std::size_t>(state.batch * (config_.ngram_size - 1)) ||
            state.ple_convolution->shape() != Shape{state.batch,
                static_cast<int>((config_.ple_conv_kernel_size - 1) * config_.ngram_size),
                static_cast<int>(config_.hc_count * config_.hidden_size)} ||
            state.ple_convolution->dtype() != mlx::core::float32)
            throw std::runtime_error("PLE checkpoint topology mismatch");
        reset(state.batch);
        convolution_state_ = detached_copy(*state.ple_convolution);
        convolution_state_->eval();
        embedding_.restore_context(state.ple_context);
        resources_.set({cache_bytes(), static_cast<std::size_t>(batch_)});
    }
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
        rollback_.reset();
        batch_ = batch;
        resources_.set({});
    }

    void clear() noexcept {
        convolution_state_.reset();
        rollback_.reset();
        batch_ = 0;
        resources_.set({});
    }

    std::size_t cache_bytes() const noexcept {
        return convolution_state_ ? convolution_state_->nbytes() : 0;
    }

    Qwen4NgramEmbedding::Prefetched prefetch(const array& token_ids, bool use_cache) {
        return embedding_.prefetch(token_ids, use_cache);
    }

    Qwen4NgramEmbedding::Prefetched prefetch_requests(const array& ids, const std::vector<int>& offsets,
        const std::vector<Qwen4RequestLayerCache*>& caches) {
        return embedding_.prefetch_requests(ids, offsets, caches);
    }

    array forward_requests(const array& hidden, const array& ids, const std::vector<int>& offsets,
        const std::vector<Qwen4RequestLayerCache*>& caches, int confirmed,
        Qwen4NgramEmbedding::Prefetched* prefetched) {
        const int batch = static_cast<int>(caches.size());
        for (int row = 0; row < batch; ++row) {
            const int tokens = offsets[row + 1] - offsets[row];
            const int retained = tokens > 1 ? confirmed : 0;
            if (retained < 0 || retained >= tokens)
                throw std::runtime_error("Qwen4 PLE speculative request boundary disagrees");
            auto& cache = *caches[row];
            if (cache.ple_rollback) throw std::runtime_error("Qwen4 PLE request is not committed");
            if (retained) cache.ple_rollback.emplace(Qwen4PleSpeculativeState{
                cache.ple_convolution, cache.ple_context,
                mlx::core::slice(ids, {0, offsets[row]}, {1, offsets[row + 1]}), std::nullopt, retained, tokens});
        }
        auto embeddings = embedding_.forward_requests(ids, offsets, caches, prefetched);
        auto [gated, normalized] = project(hidden, embeddings);
        std::vector<array> output;
        output.reserve(batch);
        for (int row = 0; row < batch; ++row) {
            auto& cache = *caches[row];
            auto inputs = mlx::core::slice(normalized, {0, offsets[row], 0}, {1, offsets[row + 1], normalized.shape(2)});
            if (cache.ple_rollback) cache.ple_rollback->convolution_inputs = inputs;
            auto convolution = cached_depthwise_conv_silu(inputs, convolution_weight_, cache.ple_convolution,
                static_cast<int>(config_.ngram_size));
            cache.ple_convolution = std::move(convolution.state);
            cache.ple_batch = 1;
            cache.ple_resources.set({cache.ple_convolution->nbytes(), 1});
            output.push_back(mlx::core::slice(gated, {0, offsets[row], 0}, {1, offsets[row + 1], gated.shape(2)}) + convolution.output);
        }
        auto result = mlx::core::concatenate(output, 1);
        return result.dtype() == hidden.dtype() ? result : mlx::core::astype(result, hidden.dtype());
    }

    array forward(
        const array& hidden_streams,
        const array& token_ids,
        bool use_cache,
        int speculative_confirmed = 0,
        Qwen4NgramEmbedding::Prefetched* prefetched = nullptr) {
        const int batch = token_ids.shape(0);
        const int tokens = token_ids.shape(1);
        if (hidden_streams.ndim() != 3 || token_ids.ndim() != 2 ||
            hidden_streams.shape(0) != batch ||
            hidden_streams.shape(1) != tokens ||
            hidden_streams.shape(2) !=
                config_.hidden_size * config_.hc_count) {
            throw std::runtime_error("Qwen4 PLE input geometry disagrees");
        }
        if (speculative_confirmed < 0 ||
            speculative_confirmed > tokens ||
            (speculative_confirmed != 0 &&
             (!use_cache || speculative_confirmed == tokens))) {
            throw std::runtime_error(
                "Qwen4 PLE speculative boundary disagrees");
        }
        if (speculative_confirmed > 0) {
            if (batch_ != batch) reset(batch);
            rollback_.emplace(Qwen4PleSpeculativeState{
                convolution_state_,
                embedding_.context(),
                token_ids,
                std::nullopt,
                speculative_confirmed,
                tokens,
            });
            try {
                // Evaluate every projection for the verify window as one
                // small-M operation.  Only the cache recurrence is replayed
                // if the speculative suffix is rejected.
                return forward(hidden_streams, token_ids, true, 0, prefetched);
            } catch (...) {
                convolution_state_ = rollback_->convolution_state;
                resources_.set({cache_bytes(), static_cast<std::size_t>(batch_)});
                embedding_.restore_context(
                    std::move(rollback_->ngram_context));
                rollback_.reset();
                throw;
            }
        }
        if (use_cache && batch_ != batch) reset(batch);
        auto embeddings = embedding_.forward(token_ids, use_cache, prefetched);
        auto [gated, normalized] = project(hidden_streams, embeddings);
        if (use_cache && rollback_ &&
            rollback_->total_tokens == tokens &&
            rollback_->confirmed_tokens > 0) {
            rollback_->convolution_inputs = normalized;
        }
        auto convolution = cached_depthwise_conv_silu(
            normalized,
            convolution_weight_,
            use_cache ? convolution_state_ : std::nullopt,
            static_cast<int>(config_.ngram_size));
        if (use_cache) {
            convolution_state_ = convolution.state;
            resources_.set({cache_bytes(), static_cast<std::size_t>(batch_)});
        }
        auto output = gated + convolution.output;
        return output.dtype() == hidden_streams.dtype()
            ? output : mlx::core::astype(output, hidden_streams.dtype());
    }

private:
    std::pair<array, array> project(const array& hidden_streams, const array& embeddings) const {
        const int batch = hidden_streams.shape(0), tokens = hidden_streams.shape(1);
        auto key_value = key_value_(embeddings);
        if (key_value.size() != 2) {
            throw std::logic_error(
                "Qwen4 PLE projection group output mismatch");
        }
        const Shape stream_shape{
            batch,
            tokens,
            static_cast<int>(config_.hc_count),
            static_cast<int>(config_.hidden_size),
        };
        auto key = mlx::core::reshape(
            qwen4_grouped_rms_norm(
                std::move(key_value[0]), norm_key_,
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
                    mlx::core::astype(
                        std::move(key_value[1]),
                        mlx::core::float32),
                    -2),
            Shape{
                batch,
                tokens,
                static_cast<int>(config_.hc_count * config_.hidden_size),
            });
        auto normalized = qwen4_grouped_rms_norm(
            gated, norm_conv_, static_cast<int>(config_.hidden_size),
            static_cast<float>(config_.rms_norm_eps));
        return {std::move(gated), std::move(normalized)};
    }

public:
    void commit_speculative() noexcept {
        rollback_.reset();
    }

    void rollback_speculative(int accepted_tokens) {
        if (!rollback_ || !rollback_->token_ids ||
            !rollback_->convolution_inputs) {
            throw std::runtime_error(
                "Qwen4 PLE has no speculative rollback state");
        }
        const int speculative_tokens =
            rollback_->total_tokens - rollback_->confirmed_tokens;
        if (accepted_tokens < 0 || accepted_tokens > speculative_tokens) {
            throw std::runtime_error(
                "Qwen4 PLE accepted prefix is outside the draft window");
        }
        const int keep = rollback_->confirmed_tokens + accepted_tokens;
        const int batch = rollback_->token_ids->shape(0);
        auto inputs = mlx::core::slice(
            *rollback_->convolution_inputs,
            Shape{0, 0, 0},
            Shape{batch, keep, rollback_->convolution_inputs->shape(2)});
        auto convolution = cached_depthwise_conv_silu(
            inputs,
            convolution_weight_,
            rollback_->convolution_state,
            static_cast<int>(config_.ngram_size));
        convolution_state_ = std::move(convolution.state);
        resources_.set({cache_bytes(), static_cast<std::size_t>(batch_)});
        embedding_.restore_context(std::move(rollback_->ngram_context));
        embedding_.advance_context(*rollback_->token_ids, keep);
        rollback_.reset();
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
          key_value_(std::vector<const MlxLinear*>{&key_, &value_}),
          norm_key_(std::move(norm_key)),
          norm_query_(std::move(norm_query)),
          norm_conv_(std::move(norm_conv)),
          convolution_weight_(std::move(convolution_weight)) {
        if (auto* preparation = MlxKernelPreparation::current()) {
            preparation->collect([&] {
                for (const auto rows : preparation->row_buckets()) {
                    const auto input = mlx::core::zeros(
                        Shape{1, rows, static_cast<int>(config_.hidden_size * config_.hc_count)}, mlx::core::float16);
                    const auto embeddings = mlx::core::zeros(
                        Shape{1, rows, static_cast<int>(config_.hidden_size)}, mlx::core::float16);
                    const auto projected = project(input, embeddings);
                    preparation->add(projected.first);
                    preparation->add(projected.second);
                    const auto normalized = qwen4_grouped_rms_norm(input, norm_conv_,
                        static_cast<int>(config_.hidden_size), static_cast<float>(config_.rms_norm_eps));
                    preparation->add(qwen4_grouped_rms_norm(input, norm_query_,
                        static_cast<int>(config_.hidden_size), static_cast<float>(config_.rms_norm_eps)));
                    const auto state = mlx::core::zeros(
                        Shape{1, static_cast<int>(config_.ple_conv_kernel_size - 1) *
                            static_cast<int>(config_.ngram_size), input.shape(2)}, mlx::core::float32);
                    for (const auto& cached_state : {std::optional<array>{}, std::optional<array>{state}}) {
                        const auto convolution = cached_depthwise_conv_silu(normalized,
                            convolution_weight_, cached_state, static_cast<int>(config_.ngram_size));
                        preparation->add(convolution.output);
                        preparation->add(convolution.state);
                    }
                }
            });
        }
    }

    Qwen4Config config_;
    Qwen4NgramEmbedding embedding_;
    MlxLinear key_;
    MlxLinear value_;
    MlxProjectionBatch key_value_;
    array norm_key_;
    array norm_query_;
    array norm_conv_;
    array convolution_weight_;
    std::optional<array> convolution_state_;
    MlxResourceTelemetry resources_;
    std::optional<Qwen4PleSpeculativeState> rollback_;
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
        bool use_cache,
        int speculative_confirmed = 0) = 0;
    virtual array append_history(
        const array& hidden,
        const array& positions_current,
        const array& positions_full) {
        return forward(hidden, positions_current, positions_full, true);
    }
    virtual void reset(int batch) = 0;
    virtual void clear() noexcept = 0;
    virtual std::size_t cache_bytes() const noexcept = 0;
    virtual void commit_speculative() noexcept = 0;
    virtual void rollback_speculative(int accepted_tokens) = 0;
    virtual MlxQwen4LayerCacheSnapshot snapshot(bool detached) const = 0;
    virtual void restore(const MlxQwen4LayerCacheSnapshot& state) = 0;
    virtual void exchange_cache(Qwen4RequestLayerCache& state) = 0;
    virtual std::vector<array> decode_input_projections(const array& hidden) = 0;
    virtual array decode_projected_row(const array& hidden,
        const std::vector<array>& projected, int position, int confirmed) = 0;
    virtual array decode_output_projection(const array& hidden) = 0;
    virtual array forward_requests(const array& hidden,
        const std::vector<Qwen4RequestLayerCache*>& caches,
        const std::vector<int>& positions, int confirmed = 0) {
        const int batch = hidden.shape(0);
        const int tokens = hidden.shape(1);
        auto grouped = mlx::core::reshape(hidden, {1, batch * tokens, hidden.shape(2)});
        std::vector<int> offsets(batch + 1);
        for (int row = 0; row <= batch; ++row) offsets[row] = row * tokens;
        auto output = forward_packed_requests(grouped, caches, positions, offsets, confirmed);
        return mlx::core::reshape(output, {batch, tokens, output.shape(2)});
    }
    array forward_packed_requests(const array& grouped,
        const std::vector<Qwen4RequestLayerCache*>& caches,
        const std::vector<int>& positions, const std::vector<int>& offsets, int confirmed = 0) {
        const int batch = static_cast<int>(caches.size());
        auto projected = decode_input_projections(grouped);
        detail::profile_eval("qwen4.batch.attention.inputs", projected);
        std::vector<array> outputs;
        outputs.reserve(batch);
        for (int index = 0; index < batch; ++index) {
            std::vector<array> inputs;
            inputs.reserve(projected.size());
            for (const auto& value : projected)
                inputs.push_back(mlx::core::slice(value, {0, offsets[index], 0},
                    {1, offsets[index + 1], value.shape(2)}));
            exchange_cache(*caches[index]);
            try {
                outputs.push_back(decode_projected_row(mlx::core::slice(grouped,
                    {0, offsets[index], 0}, {1, offsets[index + 1], grouped.shape(2)}),
                    inputs, positions[index], offsets[index + 1] - offsets[index] > 1 ? confirmed : 0));
            } catch (...) {
                exchange_cache(*caches[index]);
                throw;
            }
            exchange_cache(*caches[index]);
        }
        detail::profile_eval("qwen4.batch.attention.core", outputs);
        auto output = decode_output_projection(mlx::core::concatenate(outputs, 1));
        detail::profile_eval("qwen4.batch.attention.output", output);
        return output;
    }
    virtual void trim_cache_to(int) {
        throw std::runtime_error(
            "Qwen4 attention cache does not support arbitrary trimming");
    }
};

struct Qwen4GdnRouteCheckpoint {
    array convolution, recurrent;
    int position, batch;
};

class Qwen4Gdn final : public Qwen4Attention {
public:
    bool can_defer_routes() const noexcept {
        return batch_ == 1 && position_ > 0 && convolution_state_ && recurrent_state_ && !rollback_;
    }
    Qwen4GdnRouteCheckpoint route_checkpoint() const {
        if (!can_defer_routes()) throw std::logic_error("GDN route checkpoint is unavailable");
        return {*convolution_state_, *recurrent_state_, position_, batch_};
    }
    void restore_route_checkpoint(const Qwen4GdnRouteCheckpoint& state) {
        convolution_state_ = state.convolution;
        recurrent_state_ = state.recurrent;
        position_ = state.position;
        batch_ = state.batch;
        rollback_.reset();
    }
    array forward_requests(const array& hidden,
        const std::vector<Qwen4RequestLayerCache*>& caches,
        const std::vector<int>& positions, int confirmed = 0) override {
        const int tokens = hidden.shape(1);
        if (tokens < 1 || tokens > 2 || !gdn_decode_step_enabled() || config_.linear_key_head_dim != 128 ||
            config_.linear_value_head_dim != 128 || config_.linear_conv_kernel_dim != 4)
            return Qwen4Attention::forward_requests(hidden, caches, positions, confirmed);
        if (confirmed < 0 || confirmed >= tokens)
            throw std::logic_error("invalid batched GDN speculative boundary");
        const int batch = hidden.shape(0);
        auto grouped = mlx::core::reshape(hidden, {1, batch * tokens, hidden.shape(2)});
        auto projected = project_inputs(grouped);
        detail::profile_eval("qwen4.batch.attention.inputs", projected);
        std::vector<MlxGatedDeltaDecodeInput> inputs;
        inputs.reserve(batch);
        for (int row = 0; row < batch; ++row) {
            const auto& cache = *caches[row];
            if (cache.batch != 1 || !cache.convolution || !cache.recurrent || cache.gdn_rollback)
                throw std::logic_error("cannot batch uninitialized GDN state");
            std::vector<array> values;
            for (const auto& value : projected)
                values.push_back(mlx::core::slice(value, {0, row * tokens, 0},
                    {1, (row + 1) * tokens, value.shape(2)}));
            inputs.push_back({std::move(values[0]), std::move(values[1]), std::move(values[2]),
                std::move(values[3]), std::move(values[4]), *cache.convolution, *cache.recurrent});
        }
        auto decoded = gated_delta_decode_steps(inputs, convolution_weight_, dt_bias_, decay_scale_,
            output_norm_.weight(), static_cast<int>(config_.linear_num_key_heads),
            static_cast<int>(config_.linear_num_value_heads), 128,
            1e-6f, output_norm_.eps(), config_.output_gate_silu);
        std::vector<array> outputs;
        outputs.reserve(batch);
        for (int row = 0; row < batch; ++row) {
            auto& cache = *caches[row];
            if (confirmed) cache.gdn_rollback.emplace(MlxGatedDeltaSpeculativeState{
                *cache.convolution, *cache.recurrent, inputs[row].qk, inputs[row].value,
                decoded[row].gate, decoded[row].beta, cache.position, cache.batch, confirmed, tokens});
            cache.convolution = std::move(decoded[row].convolution_state);
            cache.recurrent = std::move(decoded[row].recurrent_state);
            cache.position += tokens;
            outputs.push_back(std::move(decoded[row].output));
        }
        detail::profile_eval("qwen4.batch.attention.core", outputs);
        auto output = output_(mlx::core::astype(mlx::core::concatenate(outputs, 1), hidden.dtype()));
        detail::profile_eval("qwen4.batch.attention.output", output);
        return mlx::core::reshape(output, {batch, tokens, output.shape(2)});
    }
    std::vector<array> decode_input_projections(const array& hidden) override {
        return project_inputs(hidden);
    }
    array decode_projected_row(const array& hidden,
        const std::vector<array>& projected, int, int confirmed) override {
        if (rollback_) throw std::logic_error("cannot batch speculative GDN state");
        if (batch_ != 1 || !convolution_state_) reset(1);
        const int tokens = hidden.shape(1);
        if (confirmed) rollback_.emplace(MlxGatedDeltaSpeculativeState{
            *convolution_state_, *recurrent_state_, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, position_, batch_, confirmed, tokens});
        std::vector<array> inputs{hidden, *convolution_state_, *recurrent_state_};
        inputs.insert(inputs.end(), projected.begin(), projected.end());
        auto decoded = compiled_projected_step_(inputs);
        if (rollback_) {
            rollback_->qk = decoded[3];
            rollback_->value = decoded[4];
            rollback_->gate = decoded[5];
            rollback_->beta = decoded[6];
        }
        convolution_state_ = std::move(decoded[1]);
        recurrent_state_ = std::move(decoded[2]);
        position_ += tokens;
        return std::move(decoded[0]);
    }
    array decode_output_projection(const array& hidden) override { return output_(hidden); }
    void exchange_cache(Qwen4RequestLayerCache& state) override {
        rollback_.swap(state.gdn_rollback);
        std::swap(position_, state.position);
        std::swap(batch_, state.batch);
        convolution_state_.swap(state.convolution);
        recurrent_state_.swap(state.recurrent);
        std::swap(resources_, state.attention_resources);
    }
    MlxQwen4LayerCacheSnapshot snapshot(bool) const override {
        if (!convolution_state_ || !recurrent_state_ || rollback_ || position_ <= 0)
            throw std::runtime_error("GDN checkpoint is not committed");
        MlxQwen4LayerCacheSnapshot state;
        state.position = position_;
        state.batch = batch_;
        state.convolution = detached_copy(*convolution_state_);
        state.recurrent = detached_copy(*recurrent_state_);
        mlx::core::eval(*state.convolution, *state.recurrent);
        return state;
    }

    void restore(const MlxQwen4LayerCacheSnapshot& state) override {
        if (state.kv || !state.convolution || !state.recurrent || state.batch != 1 ||
            state.position <= 0 || state.position > config_.max_position_embeddings ||
            state.convolution->shape() != Shape{state.batch,
                static_cast<int>(config_.linear_conv_kernel_dim - 1), 2 * key_width() + value_width()} ||
            state.recurrent->shape() != Shape{state.batch,
                static_cast<int>(config_.linear_num_value_heads),
                static_cast<int>(config_.linear_value_head_dim),
                static_cast<int>(config_.linear_value_head_dim)} ||
            state.convolution->dtype() != mlx::core::float32 || state.recurrent->dtype() != mlx::core::float32)
            throw std::runtime_error("GDN checkpoint topology mismatch");
        convolution_state_ = detached_copy(*state.convolution);
        recurrent_state_ = detached_copy(*state.recurrent);
        mlx::core::eval(*convolution_state_, *recurrent_state_);
        batch_ = state.batch;
        position_ = state.position;
        rollback_.reset();
        resources_.set({cache_bytes(), static_cast<std::size_t>(batch_)});
    }
    std::size_t cache_bytes() const noexcept override {
        return (convolution_state_ ? convolution_state_->nbytes() : 0) +
            (recurrent_state_ ? recurrent_state_->nbytes() : 0);
    }
    std::string_view profile_name() const noexcept override {
        return "qwen4.linear_attention";
    }

    static std::unique_ptr<Qwen4Gdn> load(
        const MfqContainer& model,
        const Qwen4Config& config,
        const std::string& prefix) {
        std::vector<MlxLinear> qkv;
        const bool independent = model.contains(prefix + ".query.weight") ||
            model.contains(prefix + ".key.weight") || model.contains(prefix + ".value.weight");
        if (independent) {
            if (model.contains(prefix + ".qkv.weight"))
                throw std::runtime_error("ambiguous Qwen4 GDN input projections");
            for (const auto* name : {"query", "key", "value"})
                qkv.push_back(MlxLinear::load(model, prefix + "." + name + ".weight"));
        } else qkv.push_back(MlxLinear::load(model, prefix + ".qkv.weight"));
        return std::unique_ptr<Qwen4Gdn>(new Qwen4Gdn(
            config,
            std::move(qkv),
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
        rollback_.reset();
        batch_ = batch;
        position_ = 0;
        resources_.set({cache_bytes(), static_cast<std::size_t>(batch_)});
    }

    void clear() noexcept override {
        convolution_state_.reset();
        recurrent_state_.reset();
        rollback_.reset();
        batch_ = 0;
        position_ = 0;
        resources_.set({});
    }

    array forward(
        const array& hidden,
        const array&,
        const array&,
        bool use_cache,
        int speculative_confirmed = 0) override {
        if (hidden.ndim() != 3 || hidden.shape(0) <= 0 ||
            hidden.shape(1) <= 0 || hidden.shape(2) != config_.hidden_size) {
            throw std::runtime_error("Qwen4 GDN input geometry disagrees");
        }
        const int batch = hidden.shape(0);
        const int tokens = hidden.shape(1);
        if (speculative_confirmed < 0 ||
            speculative_confirmed > tokens ||
            (speculative_confirmed != 0 &&
             (!use_cache || speculative_confirmed == tokens))) {
            throw std::runtime_error(
                "Qwen4 GDN speculative boundary disagrees");
        }
        if (speculative_confirmed > 0) {
            if (batch_ != batch || !convolution_state_) reset(batch);
            if (!convolution_state_ || !recurrent_state_) {
                throw std::runtime_error(
                    "Qwen4 GDN recurrent state is unavailable");
            }
            rollback_.emplace(MlxGatedDeltaSpeculativeState{
                *convolution_state_,
                *recurrent_state_,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                position_,
                batch_,
                speculative_confirmed,
                tokens,
            });
            try {
                // Run all expensive projections and the transformer layer as
                // one small-M graph.  The common recurrent transaction keeps
                // projected inputs for a state-only prefix replay on reject.
                return forward(hidden, array(0), array(0), true);
            } catch (...) {
                convolution_state_ = rollback_->convolution_state;
                recurrent_state_ = rollback_->recurrent_state;
                position_ = rollback_->position;
                batch_ = rollback_->batch;
                rollback_.reset();
                throw;
            }
        }
        if (use_cache && (batch_ != batch || !convolution_state_)) reset(batch);
        const bool fused_decode =
            gdn_decode_step_enabled() && use_cache && !rollback_ &&
            batch == 1 && tokens == 1 &&
            config_.linear_key_head_dim == 128 &&
            config_.linear_value_head_dim == 128 &&
            config_.linear_conv_kernel_dim == 4;
        if (fused_decode) {
            const std::vector<array> inputs{
                hidden, *convolution_state_, *recurrent_state_};
            auto decoded = detail::component_profile_active()
                ? decode_step(inputs) : compiled_decode_(inputs);
            convolution_state_ = std::move(decoded[1]);
            recurrent_state_ = std::move(decoded[2]);
            ++position_;
            return std::move(decoded[0]);
        }
        const std::vector<array> inputs{
            hidden,
            use_cache ? *convolution_state_ : mlx::core::zeros(
                Shape{batch, static_cast<int>(config_.linear_conv_kernel_dim - 1),
                    2 * key_width() + value_width()}, mlx::core::float32),
            use_cache ? *recurrent_state_ : mlx::core::zeros(
                Shape{batch, static_cast<int>(config_.linear_num_value_heads),
                    static_cast<int>(config_.linear_value_head_dim),
                    static_cast<int>(config_.linear_value_head_dim)}, mlx::core::float32),
        };
        auto decoded = use_cache && batch == 1 && tokens >= 2 && tokens <= 6 &&
                !detail::component_profile_active()
            ? compiled_batch_step_(inputs) : batch_step(inputs);
        if (use_cache && rollback_ && rollback_->position == position_ &&
            rollback_->total_tokens == tokens) {
            rollback_->qk = decoded[3];
            rollback_->value = decoded[4];
            rollback_->gate = decoded[5];
            rollback_->beta = decoded[6];
        }
        if (use_cache) {
            convolution_state_ = std::move(decoded[1]);
            recurrent_state_ = std::move(decoded[2]);
            position_ += tokens;
        }
        return std::move(decoded[0]);
    }

    void commit_speculative() noexcept override {
        rollback_.reset();
    }

    void rollback_speculative(int accepted_tokens) override {
        if (!rollback_) {
            throw std::runtime_error(
                "Qwen4 GDN has no speculative rollback state");
        }
        auto restored = !detail::component_profile_active()
            ? compiled_replay_(*rollback_, accepted_tokens)
            : replay_gated_delta_speculative_prefix(
            *rollback_,
            accepted_tokens,
            convolution_weight_,
            static_cast<int>(config_.linear_num_key_heads),
            static_cast<int>(config_.linear_num_value_heads),
            static_cast<int>(config_.linear_key_head_dim),
            static_cast<int>(config_.linear_value_head_dim),
            std::nullopt,
            1e-6f,
            true);
        convolution_state_ = std::move(restored.convolution_state);
        recurrent_state_ = std::move(restored.recurrent_state);
        position_ = restored.position;
        batch_ = rollback_->batch;
        rollback_.reset();
    }

private:
    Qwen4Gdn(
        Qwen4Config config,
        std::vector<MlxLinear> qkv,
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
          input_projections_([&] {
              std::vector<const MlxLinear*> inputs;
              for (const auto& projection : qkv_) inputs.push_back(&projection);
              inputs.insert(inputs.end(), {&gate_, &alpha_, &beta_});
              return inputs;
          }()),
          convolution_weight_(std::move(convolution_weight)),
          dt_bias_(std::move(dt_bias)),
          decay_scale_(-mlx::core::exp(a_log) * array(1.0f)),
          output_norm_(std::move(output_norm)),
          output_(std::move(output)) {
        const int hidden = static_cast<int>(config_.hidden_size);
        const std::vector<int> widths = qkv_.size() == 3
            ? std::vector<int>{key_width(), key_width(), value_width()}
            : std::vector<int>{2 * key_width() + value_width()};
        if (qkv_.size() != widths.size())
            throw std::runtime_error("invalid Qwen4 GDN projection count");
        for (std::size_t i = 0; i < widths.size(); ++i)
            if (qkv_[i].input_size() != hidden || qkv_[i].output_size() != widths[i])
                throw std::runtime_error("Qwen4 GDN input projection geometry disagrees");
        compiled_decode_ = mlx::core::compile(
            [this](const std::vector<array>& inputs) {
                return decode_step(inputs);
            });
        compiled_batch_step_ = mlx::core::compile(
            [this](const std::vector<array>& inputs) {
                return batch_step(inputs);
            });
        compiled_projected_step_ = mlx::core::compile(
            [this](const std::vector<array>& inputs) {
                const std::vector<array> states(inputs.begin(), inputs.begin() + 3);
                const std::vector<array> projected(inputs.begin() + 3, inputs.end());
                const bool fused = states[0].shape(1) == 1 && gdn_decode_step_enabled() &&
                    config_.linear_key_head_dim == 128 && config_.linear_value_head_dim == 128 &&
                    config_.linear_conv_kernel_dim == 4;
                return fused ? decode_step(states, &projected, false)
                    : batch_step(states, &projected, false);
            });
        compiled_replay_ = compile_gated_delta_speculative_replay(
            convolution_weight_,
            static_cast<int>(config_.linear_num_key_heads),
            static_cast<int>(config_.linear_num_value_heads),
            static_cast<int>(config_.linear_key_head_dim),
            static_cast<int>(config_.linear_value_head_dim),
            std::nullopt, 1e-6f, true);
        if (auto* preparation = MlxKernelPreparation::current()) {
            preparation->collect([&] {
                const auto conv = mlx::core::zeros(
                    Shape{1, static_cast<int>(config_.linear_conv_kernel_dim - 1),
                        2 * key_width() + value_width()}, mlx::core::float32);
                const auto state = mlx::core::zeros(
                    Shape{1, static_cast<int>(config_.linear_num_value_heads),
                        static_cast<int>(config_.linear_value_head_dim),
                        static_cast<int>(config_.linear_value_head_dim)}, mlx::core::float32);
                for (const auto rows : preparation->row_buckets()) {
                    const auto hidden = mlx::core::zeros(
                        Shape{1, rows, static_cast<int>(config_.hidden_size)}, mlx::core::float16);
                    const std::vector<array> inputs{hidden, conv, state};
                    const bool fused = rows == 1 && gdn_decode_step_enabled() &&
                        config_.linear_key_head_dim == 128 && config_.linear_value_head_dim == 128 &&
                        config_.linear_conv_kernel_dim == 4;
                    const auto outputs = fused ? compiled_decode_(inputs)
                        : rows <= 6 ? compiled_batch_step_(inputs) : batch_step(inputs);
                    for (const auto& output : outputs) preparation->add(output);
                    if (rows == 1) {
                        auto projected_inputs = inputs;
                        const auto projected = project_inputs(hidden);
                        projected_inputs.insert(projected_inputs.end(), projected.begin(), projected.end());
                        for (const auto& output : compiled_projected_step_(projected_inputs)) preparation->add(output);
                    }
                }
                if (gdn_decode_step_enabled() && config_.linear_key_head_dim == 128 &&
                    config_.linear_value_head_dim == 128 && config_.linear_conv_kernel_dim == 4) {
                    const auto hidden = mlx::core::zeros(
                        Shape{1, 2, static_cast<int>(config_.hidden_size)}, mlx::core::float16);
                    const auto projected = project_inputs(hidden);
                    const auto decoded = gated_delta_decode_steps({{projected[0], projected[1], projected[2],
                        projected[3], projected[4], conv, state}}, convolution_weight_, dt_bias_, decay_scale_,
                        output_norm_.weight(), static_cast<int>(config_.linear_num_key_heads),
                        static_cast<int>(config_.linear_num_value_heads), 128, 1e-6f,
                        output_norm_.eps(), config_.output_gate_silu).front();
                    for (const auto& output : {decoded.output, decoded.convolution_state,
                            decoded.recurrent_state, *decoded.gate, *decoded.beta}) preparation->add(output);
                }
            });
        }
    }

    std::vector<array> project_inputs(const array& hidden) const {
        auto projected = input_projections_(hidden);
        if (qkv_.size() == 3)
            return {mlx::core::concatenate({projected[0], projected[1]}, -1),
                std::move(projected[2]), std::move(projected[3]),
                std::move(projected[4]), std::move(projected[5])};
        auto qkv = mlx::core::split(projected[0], Shape{2 * key_width()}, -1);
        return {std::move(qkv[0]), std::move(qkv[1]), std::move(projected[1]),
            std::move(projected[2]), std::move(projected[3])};
    }

    std::vector<array> decode_step(const std::vector<array>& inputs,
        const std::vector<array>* projected = nullptr, bool project_output = true) {
        auto projections = projected ? *projected : project_inputs(inputs[0]);
        detail::profile_eval("qwen4.gdn.qkv", {projections[0], projections[1]});
        detail::profile_eval("qwen4.gdn.gate", projections[2]);
        const Shape gates{1, 1, static_cast<int>(config_.linear_num_value_heads)};
        auto decoded = gated_delta_decode_step(
            projections[0], projections[1], projections[2],
            mlx::core::reshape(projections[3], gates),
            mlx::core::reshape(projections[4], gates),
            inputs[1], inputs[2], convolution_weight_, dt_bias_, decay_scale_,
            output_norm_.weight(),
            static_cast<int>(config_.linear_num_key_heads),
            static_cast<int>(config_.linear_num_value_heads),
            static_cast<int>(config_.linear_value_head_dim),
            1e-6f, output_norm_.eps(), config_.output_gate_silu);
        detail::profile_eval("qwen4.gdn.decode_step",
            {decoded.output, decoded.convolution_state, decoded.recurrent_state});
        auto normalized = mlx::core::astype(decoded.output, inputs[0].dtype());
        auto output = project_output ? output_(normalized) : std::move(normalized);
        detail::profile_eval("qwen4.gdn.output", output);
        return {std::move(output), std::move(decoded.convolution_state),
            std::move(decoded.recurrent_state)};
    }

    std::vector<array> batch_step(const std::vector<array>& inputs,
        const std::vector<array>* projected = nullptr, bool project_output = true) {
        const auto& hidden = inputs[0];
        const int batch = hidden.shape(0);
        const int tokens = hidden.shape(1);
        auto input_projections = projected ? *projected : project_inputs(hidden);
        auto qk = std::move(input_projections[0]);
        auto value = std::move(input_projections[1]);
        detail::profile_eval("qwen4.gdn.qkv", {qk, value});
        auto z = std::move(input_projections[2]);
        detail::profile_eval("qwen4.gdn.gate", z);
        const Shape gate_shape{
            batch, tokens, static_cast<int>(config_.linear_num_value_heads)};
        auto alpha = mlx::core::reshape(input_projections[3], gate_shape);
        auto beta = mlx::core::reshape(input_projections[4], gate_shape);
        auto gates = gated_delta_gates(
            alpha,
            beta,
            dt_bias_,
            decay_scale_);
        auto recurrent_gate = std::move(gates.gate);
        auto recurrent_beta = std::move(gates.beta);
        detail::profile_eval(
            "qwen4.gdn.gates", {recurrent_gate, recurrent_beta});
        auto convolved = linear_conv_qkv(
            inputs[1],
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
            recurrent_gate,
            recurrent_beta,
            inputs[2],
            true);
        detail::profile_eval(
            "qwen4.gdn.recurrent",
            {recurrent.output, recurrent.state});
        auto normalized = [&]() {
            if (tokens >= 64) {
                return gated_delta_output_norm(recurrent.output, z,
                    output_norm_.weight(), output_norm_.eps(),
                    config_.output_gate_silu, hidden.dtype());
            }
            auto value = output_norm_(recurrent.output);
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
            value = value * output_gate;
            value = mlx::core::reshape(
                mlx::core::transpose(value, {0, 2, 1, 3}),
                Shape{batch, tokens, value_width()});
            return mlx::core::astype(value, hidden.dtype());
        }();
        detail::profile_eval("qwen4.gdn.output_gate", normalized);
        auto output = project_output ? output_(normalized) : std::move(normalized);
        detail::profile_eval("qwen4.gdn.output", output);
        return {std::move(output), std::move(convolved.state), std::move(recurrent.state),
            std::move(qk), std::move(value), std::move(recurrent_gate), std::move(recurrent_beta)};
    }

    int key_width() const {
        return static_cast<int>(
            config_.linear_num_key_heads * config_.linear_key_head_dim);
    }
    int value_width() const {
        return static_cast<int>(
            config_.linear_num_value_heads * config_.linear_value_head_dim);
    }

    Qwen4Config config_;
    std::vector<MlxLinear> qkv_;
    MlxLinear gate_;
    MlxLinear alpha_;
    MlxLinear beta_;
    MlxProjectionBatch input_projections_;
    array convolution_weight_;
    array dt_bias_;
    array decay_scale_;
    MlxRmsNorm output_norm_;
    MlxLinear output_;
    std::function<std::vector<array>(const std::vector<array>&)> compiled_decode_;
    std::function<std::vector<array>(const std::vector<array>&)> compiled_batch_step_;
    std::function<std::vector<array>(const std::vector<array>&)> compiled_projected_step_;
    MlxGatedDeltaReplay compiled_replay_;
    std::optional<array> convolution_state_;
    std::optional<array> recurrent_state_;
    MlxResourceTelemetry resources_;
    std::optional<MlxGatedDeltaSpeculativeState> rollback_;
    int batch_ = 0;
    int position_ = 0;
};

class Qwen4Qsa final : public Qwen4Attention {
public:
    void prepare_history_projections() const {
        if (auto* preparation = MlxKernelPreparation::current()) {
            preparation->collect([&] {
                for (const int rows : preparation->row_buckets()) {
                    const auto hidden = mlx::core::zeros(
                        Shape{1, rows, static_cast<int>(config_.hidden_size)}, mlx::core::float16);
                    for (const auto& output : cache_projections_(hidden)) preparation->add(output);
                }
            });
        }
    }
    bool can_defer_routes() const noexcept {
        return batch_ == 1 && cache_ && cache_->position() > 0 &&
            index_position_ == cache_->position() && !speculative_trim_;
    }
    int route_checkpoint() const {
        if (!can_defer_routes()) throw std::logic_error("QSA route checkpoint is unavailable");
        return cache_->position();
    }
    std::vector<array> decode_input_projections(const array& hidden) override {
        return input_projections_(hidden);
    }
    array decode_projected_row(const array& hidden,
        const std::vector<array>& projected, int position, int confirmed) override {
        if (speculative_trim_) throw std::logic_error("cannot batch speculative QSA state");
        auto current = mlx::core::arange(position, position + hidden.shape(1), 1, mlx::core::int32);
        auto full = mlx::core::arange(0, position + hidden.shape(1), 1, mlx::core::int32);
        return forward_projected(hidden, current, full, true, confirmed, projected, false);
    }
    array decode_output_projection(const array& hidden) override { return output_(hidden); }
    void exchange_cache(Qwen4RequestLayerCache& state) override {
        std::swap(speculative_trim_, state.qsa_speculative_trim);
        if (!state.pooled) state.pooled.emplace(maximum_,
            static_cast<int>(config_.indexer_head_dim), mlx::core::float32);
        std::swap(batch_, state.batch);
        cache_.swap(state.kv);
        offloaded_kv_.swap(state.offloaded_kv);
        offloaded_pool_.swap(state.offloaded_pool);
        index_tail_.swap(state.index_tail);
        std::swap(index_tail_start_, state.index_start);
        std::swap(index_position_, state.index_position);
        std::swap(pooled_index_cache_, *state.pooled);
        std::swap(index_resources_, state.attention_resources);
    }
    const std::vector<array>& training_context() const { return training_context_; }
    void prepare_training(MlxMtpLora& lora) const {
        lora.prepare_linear(*query_.dense_weight_ref());
        if (gate_) lora.prepare_linear(*gate_->dense_weight_ref());
        lora.prepare_linear(*output_.dense_weight_ref());
        lora.prepare_attention(config_.head_dim, ((config_.indexer_budget + config_.indexer_compress_ratio + 30) / 32) * 32);
        mlx::core::eval(query_norm_.weight());
    }
    array training(const array& hidden, const array& keys, const array& values, const array& position,
        const MlxMtpLoraParameters& p) const {
        const auto cpu = p.cpu();
        const int heads = config_.num_attention_heads, dimension = config_.head_dim;
        const int width = heads * dimension, rotary = config_.rotary_dim, half = rotary / 2;
        const int attention_width = ((config_.indexer_budget + config_.indexer_compress_ratio + 30) / 32) * 32;
        if (keys.shape(2) > attention_width) throw std::invalid_argument("MTP sparse training cache exceeds its budget");
        auto padded_keys = mlx::core::pad(keys, {{0, 0}, {0, 0}, {0, attention_width - keys.shape(2)}, {0, 0}}, array(0.0f), "constant", cpu);
        auto padded_values = mlx::core::pad(values, {{0, 0}, {0, 0}, {0, attention_width - values.shape(2)}, {0, 0}}, array(0.0f), "constant", cpu);
        auto query_full = p.linear(hidden, *query_.dense_weight_ref());
        auto parts = gate_
            ? std::vector<array>{mlx::core::reshape(query_full, {heads, dimension}, cpu),
                mlx::core::reshape(p.linear(hidden, *gate_->dense_weight_ref()), {heads, dimension}, cpu)}
            : mlx::core::split(mlx::core::reshape(query_full, {heads, 2 * dimension}, cpu), 2, -1, cpu);
        auto raw_query = mlx::core::add(mlx::core::reshape(parts[0], {1, width}, cpu), p.delta("attention.q", hidden), cpu);
        auto output_gate = mlx::core::add(mlx::core::reshape(parts[1], {1, width}, cpu), p.delta("attention.gate", hidden), cpu);
        auto query = mlx::core::reshape(raw_query, {heads, dimension}, cpu);
        query = mlx::core::multiply(mlx::core::divide(query, mlx::core::sqrt(mlx::core::add(
            mlx::core::mean(mlx::core::square(query, cpu), -1, true, cpu), array(query_norm_.eps()), cpu), cpu), cpu),
            mlx::core::astype(query_norm_.weight(), mlx::core::float32, cpu), cpu);
        std::vector<float> cosine(half), sine(half);
        const auto absolute = position.data<float>()[0];
        for (int i = 0; i < half; ++i) {
            const auto angle = absolute * float(yarn_frequency(i, rotary, config_.rope_theta, config_.yarn));
            cosine[i] = std::cos(angle) * config_.yarn.attention_factor;
            sine[i] = std::sin(angle) * config_.yarn.attention_factor;
        }
        const auto c = array(cosine.begin(), Shape{1, half}, mlx::core::float32);
        const auto s = array(sine.begin(), Shape{1, half}, mlx::core::float32);
        auto first = mlx::core::slice(query, {0, 0}, {heads, half}, cpu);
        auto second = mlx::core::slice(query, {0, half}, {heads, rotary}, cpu);
        query = mlx::core::concatenate({
            mlx::core::subtract(mlx::core::multiply(first, c, cpu), mlx::core::multiply(second, s, cpu), cpu),
            mlx::core::add(mlx::core::multiply(first, s, cpu), mlx::core::multiply(second, c, cpu), cpu),
            mlx::core::slice(query, {0, rotary}, {heads, dimension}, cpu)}, -1, cpu);
        std::vector<array> attended;
        for (int head = 0; head < heads; ++head) {
            const int kv_head = head / (heads / config_.num_key_value_heads);
            auto key = mlx::core::reshape(mlx::core::slice(padded_keys, {0, kv_head, 0, 0},
                {1, kv_head + 1, attention_width, dimension}, cpu), {attention_width, dimension}, cpu);
            auto value = mlx::core::reshape(mlx::core::slice(padded_values, {0, kv_head, 0, 0},
                {1, kv_head + 1, attention_width, dimension}, cpu), {attention_width, dimension}, cpu);
            auto scores = mlx::core::divide(p.linear(mlx::core::slice(query, {head, 0}, {head + 1, dimension}, cpu), key),
                array(std::sqrt(float(dimension))), cpu);
            scores = mlx::core::where(mlx::core::less(mlx::core::arange(attention_width, mlx::core::int32, cpu),
                array(keys.shape(2), mlx::core::int32), cpu), scores, array(-1e30f), cpu);
            attended.push_back(p.linear(mlx::core::softmax(scores, -1, cpu), mlx::core::transpose(value, cpu)));
        }
        auto gathered = mlx::core::reshape(mlx::core::concatenate(attended, -1, cpu), {1, width}, cpu);
        auto gated = mlx::core::multiply(gathered, mlx::core::sigmoid(output_gate, cpu), cpu);
        return mlx::core::reshape(mlx::core::add(p.linear(gated, *output_.dense_weight_ref()),
            p.delta("attention.o", gated), cpu), hidden.shape(), cpu);
    }
    array append_history(
        const array& hidden,
        const array& positions_current,
        const array& positions_full) override {
        const int batch = hidden.shape(0);
        const int tokens = hidden.shape(1);
        if (speculative_trim_ != 0)
            throw std::runtime_error("QSA history append requires committed cache");
        if ((!cache_ && !offloaded_kv_) || batch_ != batch) reset(batch);
        auto projected = cache_projections_(hidden);
        auto key = mlx::core::transpose(
            key_norm_(mlx::core::reshape(projected.at(0),
                Shape{batch, tokens, static_cast<int>(config_.num_key_value_heads),
                    static_cast<int>(config_.head_dim)})), {0, 2, 1, 3});
        key = apply_rope(key, positions_current,
            static_cast<int>(config_.rotary_dim),
            static_cast<float>(config_.rope_theta),
            config_.rope_sections, config_.mrope_interleaved, config_.yarn);
        auto value = mlx::core::transpose(
            mlx::core::reshape(projected.at(1),
                Shape{batch, tokens, static_cast<int>(config_.num_key_value_heads),
                    static_cast<int>(config_.head_dim)}), {0, 2, 1, 3});
        const int index_begin = static_cast<int>(config_.indexer_n_heads * config_.indexer_head_dim);
        auto raw_key = mlx::core::slice(projected.at(2),
            Shape{0, 0, index_begin},
            Shape{batch, tokens, index_begin + static_cast<int>(config_.indexer_head_dim)});
        if (offloaded_kv_) {
            offloaded_kv_->append(mlx::core::astype(key, mlx::core::float16),
                mlx::core::astype(value, mlx::core::float16));
            append_index_keys(raw_key, positions_full);
            return *index_tail_;
        }
        const int position = cache_->position();
        cache_->append_only(key, value);
        if (index_position() != position)
            throw std::runtime_error("QSA history caches diverged");
        append_index_keys(raw_key, positions_full);
        return mlx::core::depends({cache_->key_storage()}, {cache_->value_storage(), *index_tail_, pooled_index_cache_.view()}).at(0);
    }

    MlxQwen4LayerCacheSnapshot snapshot(bool detached) const override {
        if ((!cache_ && !offloaded_kv_) || speculative_trim_ || index_position() != cache_position())
            throw std::runtime_error("QSA checkpoint is not committed");
        MlxQwen4LayerCacheSnapshot state;
        state.position = cache_position();
        state.batch = batch_;
        state.index_start = index_tail_start_;
        state.index_ratio = static_cast<int>(config_.indexer_compress_ratio);
        state.index_keys = detached ? detached_copy(*index_tail_) : *index_tail_;
        if (offloaded_kv_) {
            state.offloaded_kv = offloaded_kv_->snapshot();
            if (offloaded_pool_->position()) state.offloaded_pooled_keys = offloaded_pool_->snapshot();
        } else {
            state.kv = cache_->snapshot(detached);
            if (pooled_index_cache_.position() > 0)
                state.pooled_keys = detached ? detached_copy(pooled_index_cache_.view()) : pooled_index_cache_.view();
        }
        if (state.index_keys) state.index_keys->eval();
        if (state.pooled_keys) state.pooled_keys->eval();
        return state;
    }

    void restore(const MlxQwen4LayerCacheSnapshot& state) override {
        const int width = static_cast<int>(config_.indexer_head_dim);
        const int pooled = state.position / static_cast<int>(config_.indexer_compress_ratio);
        if ((!state.kv && !state.offloaded_kv) || !state.index_keys || state.batch != 1 || state.position <= 0 ||
            state.position > maximum_ || (state.kv && state.kv->position != state.position) ||
            (state.offloaded_kv && state.offloaded_kv->rows.position != static_cast<std::size_t>(state.position)) ||
            state.index_ratio != config_.indexer_compress_ratio || state.index_start < 0 ||
            state.index_start % state.index_ratio || state.index_start > state.position ||
            state.index_keys->shape() != Shape{state.batch, state.position - state.index_start, width} ||
            state.position - state.index_start > state.index_ratio + kMlxMtpEngineMaximumDraftDepth + 1 ||
            (state.offloaded_pooled_keys && (state.offloaded_pooled_keys->width != width ||
                state.offloaded_pooled_keys->dtype != mlx::core::float32 ||
                state.offloaded_pooled_keys->rows.position != static_cast<std::size_t>(pooled))) ||
            (state.pooled_keys && (state.pooled_keys->ndim() != 3 ||
                state.pooled_keys->shape(0) != state.batch || state.pooled_keys->shape(1) != pooled ||
                state.pooled_keys->dtype() != mlx::core::float32 || state.pooled_keys->shape(2) != width)) ||
            (pooled && !state.pooled_keys && !state.offloaded_pooled_keys))
            throw std::runtime_error("QSA checkpoint topology mismatch");
        reset(state.batch);
        index_tail_ = detached_copy(*state.index_keys);
        index_tail_start_ = state.index_start;
        index_position_ = state.position;
        index_resources_.set({index_tail_->nbytes() + index_tail_->dtype().size(), static_cast<std::size_t>(state.batch)});
        if (offloaded_kv_) {
            if (state.offloaded_kv) offloaded_kv_->restore(*state.offloaded_kv);
            else offloaded_kv_->append_snapshot(*state.kv);
            if (state.offloaded_pooled_keys) offloaded_pool_->restore(*state.offloaded_pooled_keys);
            else if (state.pooled_keys) offloaded_pool_->append(*state.pooled_keys);
        } else {
            if (!state.kv) throw std::runtime_error("offloaded QSA checkpoint requires KV offload enabled");
            cache_->restore_snapshot(*state.kv);
            if (state.pooled_keys) pooled_index_cache_.append(*state.pooled_keys);
        }
    }
    std::size_t cache_bytes() const noexcept override {
        return (cache_ ? cache_->key_storage().nbytes() +
            cache_->value_storage().nbytes() : 0) +
            (index_tail_ ? index_tail_->nbytes() + index_tail_->dtype().size() : 0) + pooled_index_cache_.storage_bytes() +
            (offloaded_kv_ ? offloaded_kv_->resident_bytes() : 0);
    }
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
                static_cast<float>(config.rms_norm_eps), 1.0f),
            model.contains(prefix + ".gate.weight")
                ? std::optional<MlxLinear>{MlxLinear::load(model, prefix + ".gate.weight")}
                : std::nullopt));
    }

    void reset(int batch) override {
        if (kv_store_ && batch != 1) throw std::runtime_error("QSA KV offload requires batch one");
        if (kv_store_) {
            offloaded_kv_ = std::make_unique<MlxQsaKvOffload>(kv_store_,
                static_cast<int>(config_.num_key_value_heads), static_cast<int>(config_.head_dim),
                maximum_, static_cast<int>(config_.indexer_compress_ratio));
            offloaded_pool_ = std::make_unique<MlxQsaIndexOffload>(kv_store_,
                static_cast<int>(config_.indexer_head_dim), mlx::core::float32, 2);
        } else cache_ = std::make_unique<MlxKvCache>(
            batch,
            static_cast<int>(config_.num_key_value_heads),
            maximum_,
            static_cast<int>(config_.head_dim), 16, mlx::core::float16, true);
        if (!kv_store_) {
            pooled_index_cache_.reset(batch);
        }
        index_tail_.reset();
        index_tail_start_ = 0;
        index_position_ = 0;
        index_resources_.set({});
        speculative_trim_ = 0;
        batch_ = batch;
    }

    void clear() noexcept override {
        cache_.reset();
        offloaded_kv_.reset();
        offloaded_pool_.reset();
        index_tail_.reset();
        index_tail_start_ = 0;
        index_position_ = 0;
        index_resources_.set({});
        pooled_index_cache_.clear();
        speculative_trim_ = 0;
        batch_ = 0;
    }

    array forward(
        const array& hidden,
        const array& positions_current,
        const array& positions_full,
        bool use_cache,
        int speculative_confirmed = 0) override {
        return forward_projected(hidden, positions_current, positions_full, use_cache,
            speculative_confirmed, input_projections_(hidden));
    }
    array forward_projected(
        const array& hidden,
        const array& positions_current,
        const array& positions_full,
        bool use_cache,
        int speculative_confirmed,
        std::vector<array> input_projections,
        bool project_output = true) {
        const int batch = hidden.shape(0);
        const int tokens = hidden.shape(1);
        if (speculative_confirmed < 0 ||
            speculative_confirmed > tokens ||
            (speculative_confirmed != 0 &&
             (!use_cache || speculative_confirmed == tokens))) {
            throw std::runtime_error(
                "Qwen4 QSA speculative boundary disagrees");
        }
        if (input_projections.size() != (gate_ ? 5 : 4)) {
            throw std::logic_error(
                "Qwen4 QSA projection group output mismatch");
        }
        auto query_full = std::move(input_projections[0]);
        if (gate_) {
            const Shape heads{batch, tokens, static_cast<int>(config_.num_attention_heads),
                static_cast<int>(config_.head_dim)};
            query_full = mlx::core::reshape(mlx::core::concatenate({
                mlx::core::reshape(query_full, heads),
                mlx::core::reshape(input_projections[4], heads)}, -1),
                Shape{batch, tokens, static_cast<int>(2 * config_.num_attention_heads * config_.head_dim)});
        }
        if (auto* lora = MlxMtpLora::current(); lora && lora->active()) {
            auto parts = mlx::core::split(mlx::core::reshape(query_full,
                {batch, tokens, int(config_.num_attention_heads), 2 * int(config_.head_dim)}), 2, -1);
            auto q = lora->apply("attention.q", hidden, mlx::core::reshape(parts[0],
                {batch, tokens, int(config_.num_attention_heads * config_.head_dim)}));
            auto gate = lora->apply("attention.gate", hidden, mlx::core::reshape(parts[1], q.shape()));
            query_full = mlx::core::reshape(mlx::core::concatenate({
                mlx::core::reshape(q, parts[0].shape()), mlx::core::reshape(gate, parts[1].shape())}, -1), query_full.shape());
        }
        auto key_full = std::move(input_projections[1]);
        auto value_full = std::move(input_projections[2]);
        auto index_query_key = std::move(input_projections[3]);
        auto index_parts = mlx::core::split(
            index_query_key,
            Shape{static_cast<int>(
                config_.indexer_n_heads * config_.indexer_head_dim)},
            -1);
        auto prologue = [&]() -> MlxQwen4QsaDecodePrologue {
            if (!config_.yarn.enabled && qsa_decode_prologue_enabled() &&
                batch == 1 && tokens >= 1 && tokens <= 6 &&
                positions_current.dtype() == mlx::core::int32 &&
                positions_current.shape() == Shape{tokens} &&
                query_norm_.eps() == key_norm_.eps() &&
                query_norm_.eps() == index_query_norm_.eps()) {
                return qwen4_qsa_decode_prologue(
                    query_full,
                    key_full,
                    index_query_key,
                    query_norm_.weight(),
                    key_norm_.weight(),
                    index_query_norm_.weight(),
                    positions_current,
                    static_cast<int>(config_.num_attention_heads),
                    static_cast<int>(config_.num_key_value_heads),
                    static_cast<int>(config_.indexer_n_heads),
                    static_cast<int>(config_.head_dim),
                    static_cast<int>(config_.indexer_head_dim),
                    static_cast<int>(config_.rotary_dim),
                    static_cast<float>(config_.rope_theta),
                    query_norm_.eps());
            }
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
            query = apply_rope(
                query,
                positions_current,
                static_cast<int>(config_.rotary_dim),
                static_cast<float>(config_.rope_theta),
                config_.rope_sections,
                config_.mrope_interleaved, config_.yarn);
            key = apply_rope(
                key,
                positions_current,
                static_cast<int>(config_.rotary_dim),
                static_cast<float>(config_.rope_theta),
                config_.rope_sections,
                config_.mrope_interleaved, config_.yarn);
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
                    config_.mrope_interleaved, config_.yarn),
                {0, 2, 1, 3});
            return {
                std::move(query),
                std::move(query_parts.at(1)),
                std::move(key),
                std::move(index_query),
            };
        }();
        auto query = std::move(prologue.query);
        auto output_gate = std::move(prologue.output_gate);
        auto key = std::move(prologue.key);
        auto index_query = std::move(prologue.index_query);
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
        auto raw_key = mlx::core::reshape(
            index_parts.at(1),
            Shape{batch, tokens, static_cast<int>(config_.indexer_head_dim)});

        array key_cache = key;
        array value_cache = value;
        int query_offset = 0;
        if (use_cache) {
            if ((!cache_ && !offloaded_kv_) || batch_ != batch) reset(batch);
            query_offset = cache_position();
            if (offloaded_kv_) {
                offloaded_kv_->append(mlx::core::astype(key, mlx::core::float16),
                    mlx::core::astype(value, mlx::core::float16));
            } else {
                cache_->append_only(key, value);
                if (!cache_->quantized()) {
                    auto cache = cache_->view();
                    key_cache = std::move(cache.first);
                    value_cache = std::move(cache.second);
                }
            }
            if (index_position() != query_offset)
                throw std::runtime_error("Qwen4 QSA caches diverged");
            append_index_keys(raw_key, positions_full);
            speculative_trim_ = speculative_confirmed > 0
                ? tokens - speculative_confirmed
                : 0;
        }
        const int ratio = static_cast<int>(config_.indexer_compress_ratio);
        const auto pooled_keys = [&]() -> array {
            return use_cache ? pooled_index_cache_.view() : pool_index_keys(
                raw_key, positions_full, 0, tokens / ratio);
        };
        if (auto* lora = MlxMtpLora::current(); lora && lora->collecting()) {
            if (offloaded_kv_) throw std::invalid_argument("MTP LoRA training does not support QSA KV offload");
            if (use_cache && cache_->quantized()) throw std::invalid_argument("MTP LoRA training does not support quantized QSA KV");
            if (batch != 1 || tokens != 1) throw std::invalid_argument("MTP training requires one-token draft steps");
            array selected = mlx::core::arange(query_offset + 1, mlx::core::int32);
            if (key_cache.shape(2) > config_.indexer_budget) {
                const int complete = key_cache.shape(2) / ratio;
                auto blocks = selected_blocks(index_query, pooled_keys(), query_offset);
                selected = mlx::core::reshape(mlx::core::reshape(blocks, {-1, 1}) * array(ratio, mlx::core::int32) +
                    mlx::core::arange(ratio, mlx::core::int32), {-1});
                if ((query_offset + 1) % ratio) selected = mlx::core::concatenate({selected,
                    mlx::core::arange(complete * ratio, query_offset + 1, 1, mlx::core::int32)}, 0);
            }
            training_context_ = {mlx::core::take(key_cache, selected, 2),
                mlx::core::take(value_cache, selected, 2), positions_current};
        }
        array attended = [&]() {
            if (use_cache && cache_ && cache_->quantized()) {
                std::optional<array> blocks;
                if (cache_->position() > config_.indexer_budget) blocks = selected_blocks(index_query, pooled_keys(), query_offset);
                return cache_->sparse_attention(query, blocks, query_offset, ratio, static_cast<int>(config_.indexer_budget));
            }
            if (use_cache && offloaded_kv_) {
                std::optional<array> blocks;
                if (offloaded_kv_->position() > config_.indexer_budget) {
                    blocks = offloaded_pool_->select(index_query, query_offset,
                        static_cast<int>(config_.indexer_compress_ratio), static_cast<int>(config_.indexer_budget));
                }
                return offloaded_kv_->attention(query, blocks, query_offset, static_cast<int>(config_.indexer_budget));
            }
            if (key_cache.shape(2) <= config_.indexer_budget) {
                return qwen4_dense_gqa_attention(
                    query, key_cache, value_cache, query_offset);
            }
            if (tokens >= 32 && batch == 1 && query.shape(1) == 24 &&
                key_cache.shape(1) == 2 && query.shape(3) == 256 &&
                config_.indexer_n_heads == 4 && config_.indexer_head_dim == 128 &&
                config_.indexer_budget == 2048 && ratio == 4 &&
                (query.dtype() == mlx::core::float16 || query.dtype() == mlx::core::bfloat16) &&
                key_cache.dtype() == query.dtype() && value_cache.dtype() == query.dtype() &&
                index_query.dtype() == query.dtype()) {
                return mlx_qsa_prefill_attention(
                    query, key_cache, value_cache, index_query, pooled_keys(), query_offset);
            }
            const int dense_tokens = tokens >= 32
                ? std::min(tokens, std::max(0,
                    static_cast<int>(config_.indexer_budget) - query_offset))
                : 0;
            std::optional<array> dense_prefix;
            auto sparse_query = query;
            auto sparse_index_query = index_query;
            if (dense_tokens > 0) {
                const int end = query_offset + dense_tokens;
                dense_prefix = qwen4_dense_gqa_attention(
                    mlx::core::slice(query, Shape{0, 0, 0, 0},
                        Shape{batch, query.shape(1), dense_tokens, query.shape(3)}),
                    mlx::core::slice(key_cache, Shape{0, 0, 0, 0},
                        Shape{batch, key_cache.shape(1), end, key_cache.shape(3)}),
                    mlx::core::slice(value_cache, Shape{0, 0, 0, 0},
                        Shape{batch, value_cache.shape(1), end, value_cache.shape(3)}),
                    query_offset);
                if (dense_tokens == tokens) return *dense_prefix;
                sparse_query = mlx::core::slice(query, Shape{0, 0, dense_tokens, 0},
                    Shape{batch, query.shape(1), tokens, query.shape(3)});
                sparse_index_query = mlx::core::slice(index_query,
                    Shape{0, dense_tokens, 0, 0}, index_query.shape());
            }
            array pooled = pooled_keys();
            auto sparse = mlx_sparse_block_gqa_attention(
                sparse_query,
                key_cache,
                value_cache,
                selected_blocks(sparse_index_query, pooled, query_offset + dense_tokens),
                query_offset + dense_tokens,
                ratio);
            return dense_prefix
                ? mlx::core::concatenate({*dense_prefix, sparse}, 1)
                : sparse;
        }();
        if (use_cache) attended = offloaded_kv_ ? mlx::core::depends({attended}, {*index_tail_}).at(0)
            : mlx::core::depends({attended}, {*index_tail_, pooled_index_cache_.view()}).at(0);
        if (tokens >= 32 && attended.ndim() == 4 &&
            attended.shape() == output_gate.shape() &&
            attended.dtype() == hidden.dtype() && output_gate.dtype() == hidden.dtype()) {
            auto gated = qwen4_qsa_prefill_output_gate(attended, output_gate);
            return project_output ? output_(gated) : std::move(gated);
        }
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
        auto output_input = mlx::core::astype(gated, hidden.dtype());
        if (!project_output) return output_input;
        auto output = output_(output_input);
        if (auto* lora = MlxMtpLora::current()) output = lora->apply("attention.o", output_input, output);
        return output;
    }

    void commit_speculative() noexcept override {
        speculative_trim_ = 0;
    }

    void rollback_speculative(int accepted_tokens) override {
        if ((!cache_ && !offloaded_kv_) || speculative_trim_ <= 0) {
            throw std::runtime_error(
                "Qwen4 QSA has no speculative rollback state");
        }
        if (accepted_tokens < 0 || accepted_tokens >= speculative_trim_) {
            throw std::runtime_error(
                "Qwen4 QSA accepted prefix is outside the draft window");
        }
        const int rejected = speculative_trim_ - accepted_tokens;
        if (index_position() - rejected < index_tail_start_)
            throw std::runtime_error("QSA rollback exceeds the uncommitted index tail");
        if (offloaded_kv_) offloaded_kv_->trim(cache_position() - rejected);
        else cache_->trim(rejected);
        trim_index_keys(index_position() - rejected);
        trim_pooled_index_cache();
        speculative_trim_ = 0;
    }

    void trim_cache_to(int position) override {
        if ((!cache_ && !offloaded_kv_) || position < 0 || position > cache_position() ||
            index_position() != cache_position() || position < index_tail_start_) {
            throw std::runtime_error(
                "Qwen4 QSA cache trim position is invalid");
        }
        const int count = cache_position() - position;
        if (count > 0) {
            if (offloaded_kv_) offloaded_kv_->trim(position);
            else cache_->trim(count);
            trim_index_keys(position);
            trim_pooled_index_cache();
        }
        speculative_trim_ = 0;
    }

private:
    int cache_position() const noexcept {
        return offloaded_kv_ ? offloaded_kv_->position() : cache_ ? cache_->position() : 0;
    }
    int index_position() const noexcept {
        return index_position_;
    }
    void append_index_keys(const array& keys, const array& positions_full) {
        const int ratio = static_cast<int>(config_.indexer_compress_ratio);
        const int end = index_position_ + keys.shape(1);
        auto raw = mlx::core::astype(keys, mlx::core::float16);
        if (index_tail_) raw = mlx::core::concatenate({*index_tail_, raw}, 1);
        const int cached = offloaded_pool_ ? offloaded_pool_->position() : pooled_index_cache_.position();
        const int begin = cached - index_tail_start_ / ratio;
        const int complete = end / ratio - index_tail_start_ / ratio;
        if (begin < complete) {
            auto pooled = pool_index_keys(raw, positions_full, begin, complete, index_tail_start_);
            if (offloaded_pool_) offloaded_pool_->append(pooled);
            else pooled_index_cache_.append(pooled);
        }
        const int keep = std::max(index_tail_start_, std::max(0, end - kMlxMtpEngineMaximumDraftDepth - 1) / ratio * ratio);
        index_tail_ = detached_copy(mlx::core::slice(raw, Shape{0, keep - index_tail_start_, 0},
            Shape{keys.shape(0), raw.shape(1), keys.shape(2)}));
        index_tail_start_ = keep;
        index_position_ = end;
        index_resources_.set({index_tail_->nbytes() + index_tail_->dtype().size(), static_cast<std::size_t>(keys.shape(0))});
    }
    void trim_index_keys(int position) {
        if (position < index_tail_start_ || position > index_position_)
            throw std::runtime_error("QSA rollback exceeds the uncommitted index tail");
        if (position == 0) index_tail_.reset();
        else index_tail_ = detached_copy(mlx::core::slice(*index_tail_, Shape{0, 0, 0},
            Shape{batch_, position - index_tail_start_, static_cast<int>(config_.indexer_head_dim)}));
        index_position_ = position;
        index_resources_.set({index_tail_ ? index_tail_->nbytes() + index_tail_->dtype().size() : 0,
            index_tail_ ? static_cast<std::size_t>(batch_) : 0});
    }
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
        MlxRmsNorm index_key_norm,
        std::optional<MlxLinear> gate)
        : config_(std::move(config)),
          maximum_(maximum),
          query_(std::move(query)),
          key_(std::move(key)),
          value_(std::move(value)),
          query_norm_(std::move(query_norm)),
          key_norm_(std::move(key_norm)),
          output_(std::move(output)),
          index_query_key_(std::move(index_query_key)),
          gate_(std::move(gate)),
          input_projections_([&] {
              std::vector<const MlxLinear*> inputs{&query_, &key_, &value_, &index_query_key_};
              if (gate_) inputs.push_back(&*gate_);
              return inputs;
          }()),
          cache_projections_(std::vector<const MlxLinear*>{
              &key_, &value_, &index_query_key_,
          }),
          index_query_norm_(std::move(index_query_norm)),
          index_key_norm_(std::move(index_key_norm)),
          pooled_index_cache_(
              (maximum +
               static_cast<int>(config_.indexer_compress_ratio) - 1) /
                  static_cast<int>(config_.indexer_compress_ratio),
              static_cast<int>(config_.indexer_head_dim), mlx::core::float32) {
        const int width = static_cast<int>(config_.num_attention_heads * config_.head_dim);
        if (query_.output_size() != (gate_ ? width : 2 * width) ||
            (gate_ && (gate_->input_size() != config_.hidden_size || gate_->output_size() != width)))
            throw std::runtime_error("Qwen4 QSA query/gate projection geometry disagrees");
        if (auto* preparation = MlxKernelPreparation::current()) {
            preparation->collect([&] {
                for (const auto requested : preparation->row_buckets()) {
                    const int rows = std::min(maximum_, requested);
                    const auto hidden = mlx::core::zeros(
                        Shape{1, rows, static_cast<int>(config_.hidden_size)}, mlx::core::float16);
                    const std::vector<std::int32_t> position_values(rows, 0);
                    for (const auto& positions : {mlx::core::zeros(Shape{rows}, mlx::core::int32),
                            array(position_values.begin(), Shape{rows}),
                            array(position_values.begin(), Shape{1, rows})})
                        preparation->add(forward(hidden, positions, positions, false));
                    if (config_.num_attention_heads != 24 || config_.num_key_value_heads != 2 ||
                        config_.head_dim != 256 || config_.indexer_n_heads != 4 ||
                        config_.indexer_head_dim != 128 || config_.indexer_budget != 2048 ||
                        config_.indexer_compress_ratio != 4 || maximum_ <= 2048) continue;
                    const int length = std::max(rows, std::min(maximum_, 4096));
                    const int offset = length - rows;
                    const auto query = mlx::core::zeros(Shape{1, 24, rows, 256}, mlx::core::float16);
                    const auto keys = mlx::core::zeros(Shape{1, 2, length, 256}, mlx::core::float16);
                    const auto index = mlx::core::zeros(Shape{1, rows, 4, 128}, mlx::core::float16);
                    const auto pooled = mlx::core::zeros(Shape{1, length / 4, 128}, mlx::core::float32);
                    const auto blocks = selected_blocks(index, pooled, offset);
                    preparation->add(mlx_sparse_block_gqa_attention(query, keys, keys, blocks, offset, 4));
                    if (rows >= 32) {
                        preparation->add(mlx_qsa_prefill_attention(query, keys, keys, index, pooled, offset));
                        const auto output = mlx::core::zeros(Shape{1, rows, 24, 256}, mlx::core::float16);
                        preparation->add(qwen4_qsa_prefill_output_gate(output, output));
                    }
                }
                if (config_.indexer_budget / config_.indexer_compress_ratio == 512) {
                    const int max_keys = std::min(maximum_ / static_cast<int>(config_.indexer_compress_ratio), 32768);
                    for (int width = 1024; width / 2 < max_keys; width *= 2) {
                        preparation->add(qwen4_qsa_select_blocks(
                            mlx::core::zeros(Shape{1, 1, static_cast<int>(config_.indexer_n_heads),
                                static_cast<int>(config_.indexer_head_dim)}, mlx::core::float16),
                            mlx::core::zeros(Shape{1, std::min(width, max_keys),
                                static_cast<int>(config_.indexer_head_dim)}, mlx::core::float32),
                            0, static_cast<int>(config_.indexer_compress_ratio)));
                    }
                }
            });
        }
    }

    array pool_index_keys(
        const array& raw_keys,
        const array& positions_full,
        int begin,
        int end,
        int position_offset = 0) const {
        const int ratio = static_cast<int>(config_.indexer_compress_ratio);
        const int width = static_cast<int>(config_.indexer_head_dim);
        if (raw_keys.ndim() != 3 || raw_keys.shape(0) <= 0 ||
            begin < 0 || end <= begin ||
            end * ratio > raw_keys.shape(1) ||
            raw_keys.shape(2) != width ||
            positions_full.shape(-1) < position_offset + end * ratio) {
            throw std::runtime_error(
                "Qwen4 index pool range disagrees with raw keys");
        }
        const int batch = raw_keys.shape(0);
        auto pooled = mlx::core::mean(
            mlx::core::reshape(
                mlx::core::slice(
                    raw_keys,
                    Shape{0, begin * ratio, 0},
                    Shape{batch, end * ratio, width}),
                Shape{batch, end - begin, ratio, width}),
            -2);
        pooled = index_key_norm_(pooled);
        auto starts = mlx::core::arange(
            position_offset + begin * ratio, position_offset + end * ratio, ratio, mlx::core::int32);
        auto block_positions = mlx::core::take(
            positions_full, starts, -1);
        return mlx::core::reshape(
            apply_rope(
                mlx::core::expand_dims(pooled, 1),
                block_positions,
                static_cast<int>(config_.rotary_dim),
                static_cast<float>(config_.rope_theta),
                config_.rope_sections,
                config_.mrope_interleaved, config_.yarn),
            Shape{batch, end - begin, width});
    }

    array selected_blocks(
        const array& query,
        const array& pooled_keys,
        int query_offset) const {
        const int batch = query.shape(0);
        const int tokens = query.shape(1);
        const int ratio = static_cast<int>(config_.indexer_compress_ratio);
        if (pooled_keys.ndim() != 3 ||
            pooled_keys.shape(0) != batch ||
            pooled_keys.shape(2) != config_.indexer_head_dim) {
            throw std::runtime_error(
                "Qwen4 pooled index keys disagree with the query");
        }
        const int complete = pooled_keys.shape(1);
        const int block_budget =
            static_cast<int>(config_.indexer_budget) / ratio;
        if (block_budget == 512 && complete > 512 &&
            complete <= 32768) {
            return qwen4_qsa_select_blocks(query, pooled_keys, query_offset, ratio);
        }
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
        auto scores = qwen4_qsa_block_scores(query, pooled_keys);
        auto starts = mlx::core::arange(
            0, complete * ratio, ratio, mlx::core::int32);
        auto ends = starts + array(ratio - 1, mlx::core::int32);
        auto visible = mlx::core::less(
            mlx::core::expand_dims(ends, 0),
            mlx::core::expand_dims(
                absolute + array(1, mlx::core::int32), 1));
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
        // Top-k/argpartition order is unspecified. Chronological block order
        // preserves the checkpoint reduction order and lets the direct-index
        // kernel treat the first min(visible,budget) entries as valid.
        return mlx::core::contiguous(mlx::core::sort(selected, -1));
    }

    void trim_pooled_index_cache() {
        const int ratio = static_cast<int>(config_.indexer_compress_ratio);
        const int target = index_position() / ratio;
        if (offloaded_pool_) {
            if (offloaded_pool_->position() > target) offloaded_pool_->trim(target);
            return;
        }
        if (pooled_index_cache_.position() > target) {
            pooled_index_cache_.trim(
                pooled_index_cache_.position() - target);
        }
    }

    Qwen4Config config_;
    int maximum_;
    std::vector<array> training_context_;
    MlxLinear query_;
    MlxLinear key_;
    MlxLinear value_;
    MlxRmsNorm query_norm_;
    MlxRmsNorm key_norm_;
    MlxLinear output_;
    MlxLinear index_query_key_;
    std::optional<MlxLinear> gate_;
    MlxProjectionBatch input_projections_;
    MlxProjectionBatch cache_projections_;
    MlxRmsNorm index_query_norm_;
    MlxRmsNorm index_key_norm_;
    std::shared_ptr<QsaKvStore> kv_store_ = mlx_qsa_kv_offload_store();
    std::unique_ptr<MlxQsaKvOffload> offloaded_kv_;
    std::unique_ptr<MlxQsaIndexOffload> offloaded_pool_;
    std::unique_ptr<MlxKvCache> cache_;
    std::optional<array> index_tail_;
    int index_tail_start_ = 0;
    int index_position_ = 0;
    MlxResourceTelemetry index_resources_;
    MlxSequenceCache pooled_index_cache_;
    int speculative_trim_ = 0;
    int batch_ = 0;
};

class Qwen4Layer {
public:
    using RouteCheckpoint = std::variant<Qwen4GdnRouteCheckpoint, int>;

    std::optional<Qwen4CachedRoute> cached_route() const {
        if (ple_) return std::nullopt;
        const auto* gdn = dynamic_cast<const Qwen4Gdn*>(attention_.get());
        const auto* qsa = dynamic_cast<const Qwen4Qsa*>(attention_.get());
        return (gdn && gdn->can_defer_routes()) || (qsa && qsa->can_defer_routes())
            ? moe_.cached_route() : std::nullopt;
    }
    RouteCheckpoint route_checkpoint() const {
        if (const auto* gdn = dynamic_cast<const Qwen4Gdn*>(attention_.get()))
            return gdn->route_checkpoint();
        return dynamic_cast<const Qwen4Qsa&>(*attention_).route_checkpoint();
    }
    void restore_route_checkpoint(const RouteCheckpoint& state) {
        if (const auto* gdn = std::get_if<Qwen4GdnRouteCheckpoint>(&state))
            dynamic_cast<Qwen4Gdn&>(*attention_).restore_route_checkpoint(*gdn);
        else dynamic_cast<Qwen4Qsa&>(*attention_).trim_cache_to(std::get<int>(state));
    }
    void exchange_cache(Qwen4RequestLayerCache& cache) {
        attention_->exchange_cache(cache);
        if (ple_) ple_->exchange_cache(cache);
    }

    array decode_requests(array streams, const array& ids,
        const std::vector<Qwen4RequestLayerCache*>& caches,
        const std::vector<int>& positions,
        Qwen4NgramEmbedding::Prefetched* prefetched_ple = nullptr,
        int confirmed = 0) {
        const int batch = streams.shape(0);
        const int tokens = streams.shape(1);
        std::vector<int> offsets(batch + 1);
        for (int row = 0; row <= batch; ++row) offsets[row] = row * tokens;
        auto result = forward_packed_requests(
            mlx::core::reshape(streams, {1, batch * tokens, streams.shape(2)}),
            mlx::core::reshape(ids, {1, batch * tokens}), caches, positions, offsets,
            true, confirmed, nullptr, prefetched_ple);
        return mlx::core::reshape(result, {batch, tokens, result.shape(2)});
    }
    array forward_packed_requests(array streams, const array& ids,
        const std::vector<Qwen4RequestLayerCache*>& caches,
        const std::vector<int>& positions, const std::vector<int>& offsets,
        bool decode, int confirmed = 0, MlxSsdPrefetchedExpertLayer* prefetched_routed = nullptr,
        Qwen4NgramEmbedding::Prefetched* prefetched_ple = nullptr) {
        const int batch = static_cast<int>(caches.size());
        bool uniform = true;
        for (int row = 0; row < batch; ++row)
            uniform = uniform && offsets[row + 1] - offsets[row] == offsets[1];
        if (ple_) {
            streams = streams + ple_->forward_requests(streams, ids, offsets, caches, confirmed, prefetched_ple);
            detail::profile_eval("qwen4.ple", streams);
        }
        auto av = attention_gr_.pre(streams);
        detail::profile_eval("qwen4.attention_mhc_pre", av.branch);
        auto attention = [&] {
            if (!decode || !uniform)
                return attention_->forward_packed_requests(av.branch, caches, positions, offsets, confirmed);
            const int tokens = offsets[1];
            auto branches = mlx::core::reshape(av.branch, {batch, tokens, av.branch.shape(2)});
            auto result = attention_->forward_requests(branches, caches, positions, confirmed);
            return mlx::core::reshape(result, {1, batch * tokens, result.shape(2)});
        }();
        auto fv = ffn_gr_.pre_after(attention, av);
        detail::profile_eval("qwen4.attention_post_ffn_pre", fv.branch);
        auto result = ffn_gr_.post(moe_(fv.branch, prefetched_routed, decode), fv);
        detail::profile_eval("qwen4.ffn_mhc_post", result);
        return result;
    }
    void prepare_training(MlxMtpLora& lora) const {
        const auto* qsa = dynamic_cast<const Qwen4Qsa*>(attention_.get());
        if (!qsa || ple_) throw std::invalid_argument("MTP LoRA requires a QSA predictor without PLE");
        attention_gr_.prepare_training(lora);
        ffn_gr_.prepare_training(lora);
        qsa->prepare_training(lora);
        moe_.prepare_training(lora);
    }
    std::size_t resident_routed_bytes() const noexcept { return moe_.resident_routed_bytes(); }
    void validate_offload(const std::shared_ptr<MlxMfeOffloadCache>& cache) const { moe_.validate_offload(cache); }
    void use_offload(const std::shared_ptr<MlxMfeOffloadCache>& cache) { moe_.use_offload(cache); }
    void validate_tq_offload(const std::shared_ptr<MlxTqExpertCache>& cache) const { moe_.validate_tq_offload(cache); }
    void use_tq_offload(const std::shared_ptr<MlxTqExpertCache>& cache) { moe_.use_tq_offload(cache); }
    MlxQwen4LayerCacheSnapshot snapshot(bool detached = true) const {
        auto state = attention_->snapshot(detached);
        if (ple_) ple_->snapshot(state);
        return state;
    }

    void restore(const MlxQwen4LayerCacheSnapshot& state) {
        if (bool(ple_) != bool(state.ple_convolution))
            throw std::runtime_error("PLE checkpoint layer changed");
        attention_->restore(state);
        if (ple_) ple_->restore(state);
    }
    std::size_t cache_bytes() const noexcept {
        return attention_->cache_bytes() + (ple_ ? ple_->cache_bytes() : 0);
    }
    static Qwen4Layer load(
        const MfqContainer& model,
        const Qwen4Config& config,
        std::size_t index,
        int maximum,
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache = nullptr,
        std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache = nullptr,
        std::shared_ptr<MlxTqExpertCache> tq_expert_cache = nullptr) {
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
            Qwen4Moe::load(
                model,
                config,
                prefix + ".mlp",
                std::move(ssd_expert_cache),
                std::move(mfe_offload_cache),
                index, std::move(tq_expert_cache)),
            std::move(ple));
    }

    static Qwen4Layer load_predictor(
        const MfqContainer& model,
        const Qwen4Config& config,
        std::size_t index,
        int maximum,
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache = nullptr,
        std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache = nullptr,
        std::size_t expert_cache_layer_base = 0,
        std::shared_ptr<MlxTqExpertCache> tq_expert_cache = nullptr) {
        const auto prefix =
            "predictor.block." + std::to_string(index);
        auto attention = Qwen4Qsa::load(model, config, prefix + ".attention", maximum);
        attention->prepare_history_projections();
        return Qwen4Layer(
            GatedResidual::load(
                model, config, prefix + ".attention.mhc.pre"),
            GatedResidual::load(
                model, config, prefix + ".mlp.mhc.pre"),
            std::move(attention),
            Qwen4Moe::load(
                model,
                config,
                prefix + ".mlp",
                std::move(ssd_expert_cache),
                std::move(mfe_offload_cache),
                expert_cache_layer_base + index, std::move(tq_expert_cache)),
            nullptr);
    }

    void reset(int batch) {
        attention_->reset(batch);
        if (ple_) ple_->reset(batch);
    }

    void clear() noexcept {
        attention_->clear();
        if (ple_) ple_->clear();
    }

    void commit_speculative() noexcept {
        attention_->commit_speculative();
        if (ple_) ple_->commit_speculative();
    }

    void rollback_speculative(int accepted_tokens) {
        attention_->rollback_speculative(accepted_tokens);
        if (ple_) ple_->rollback_speculative(accepted_tokens);
    }

    void trim_cache_to(int position) {
        if (ple_) {
            throw std::runtime_error(
                "Qwen4 predictor layers cannot contain PLE state");
        }
        attention_->trim_cache_to(position);
    }

    std::optional<MlxSsdPrefetchedExpertLayer> prefetch_routed(
        std::size_t rows) const {
        return moe_.prefetch_routed(rows);
    }

    Qwen4NgramEmbedding::Prefetched prefetch_ple(const array& token_ids, bool use_cache) {
        return ple_ ? ple_->prefetch(token_ids, use_cache)
            : Qwen4NgramEmbedding::Prefetched{};
    }

    Qwen4NgramEmbedding::Prefetched prefetch_ple_requests(const array& ids, const std::vector<int>& offsets,
        const std::vector<Qwen4RequestLayerCache*>& caches) {
        return ple_ ? ple_->prefetch_requests(ids, offsets, caches) : Qwen4NgramEmbedding::Prefetched{};
    }


    array append_history(
        array hidden_streams,
        const array& token_ids,
        const array& positions_current,
        const array& positions_full) {
        if (ple_)
            hidden_streams = hidden_streams + ple_->forward(hidden_streams, token_ids, true);
        auto attention_values = attention_gr_.pre(hidden_streams);
        return attention_->append_history(attention_values.branch, positions_current, positions_full);
    }

    array forward(
        array hidden_streams,
        const array& token_ids,
        const array& positions_current,
        const array& positions_full,
        bool use_cache,
        int speculative_confirmed = 0,
        MlxSsdPrefetchedExpertLayer* prefetched = nullptr,
        Qwen4NgramEmbedding::Prefetched* prefetched_ple = nullptr,
        Qwen4CachedRoute* deferred_route = nullptr) {
        auto* lora = MlxMtpLora::current();
        std::optional<array> training_root;
        if (lora && lora->collecting()) training_root = hidden_streams;
        if (ple_) {
            hidden_streams = hidden_streams +
                ple_->forward(
                    hidden_streams,
                    token_ids,
                    use_cache,
                    speculative_confirmed,
                    prefetched_ple);
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
            use_cache,
            speculative_confirmed);
        detail::profile_eval(attention_->profile_name(), branch);
        auto ffn_values = ffn_gr_.pre_after(branch, attention_values);
        detail::profile_eval(
            "qwen4.attention_mhc_post",
            ffn_values.residual);
        detail::profile_eval("qwen4.ffn_mhc_pre", ffn_values.branch);
        branch = moe_(ffn_values.branch, prefetched, false, deferred_route);
        detail::profile_eval("qwen4.moe", branch);
        auto output = ffn_gr_.post(branch, ffn_values);
        if (lora) {
            if (lora->collecting()) {
                const auto* qsa = dynamic_cast<const Qwen4Qsa*>(attention_.get());
                if (!qsa || qsa->training_context().size() != 3 || !moe_.training_routes())
                    throw std::logic_error("missing internal MTP training context");
                std::vector<array> constants{*training_root};
                constants.insert(constants.end(), qsa->training_context().begin(), qsa->training_context().end());
                constants.push_back(*moe_.training_routes());
                lora->capture(std::move(constants), [this, qsa](const MlxMtpLoraParameters& p, const std::vector<array>& c) {
                    auto av = attention_gr_.training_pre(c.at(0), p);
                    auto attention = qsa->training(av.branch, c.at(1), c.at(2), c.at(3), p);
                    auto fv = ffn_gr_.training_pre(attention_gr_.training_post(attention, av, p), p);
                    return ffn_gr_.training_post(moe_.training(fv.branch, c.at(4), p), fv, p);
                });
            } else lora->capture({}, {});
        }
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

struct Qwen4MtpForward {
    array sample_hidden;
    array hidden_streams;
};

class Qwen4Mtp {
public:
    void prepare_inputs(const MlxEmbedding& embedding) {
        if (auto* preparation = MlxKernelPreparation::current()) {
            preparation->collect([&] {
                for (const auto rows : preparation->row_buckets()) {
                    const auto hidden = mlx::core::zeros(
                        Shape{1, rows, static_cast<int>(config_.hc_count * config_.hidden_size)}, mlx::core::float16);
                    const auto ids = mlx::core::zeros(Shape{1, rows}, mlx::core::int32);
                    preparation->add(fuse_inputs(hidden, ids, embedding, false).streams);
                }
            });
        }
    }
    std::size_t layer_count() const noexcept { return layers_.size(); }
    void exchange_cache(std::vector<Qwen4RequestLayerCache>& layers, int& position, int& batch) {
        if (layers.size() != layers_.size()) throw std::invalid_argument("predictor request topology mismatch");
        for (std::size_t index = 0; index < layers.size(); ++index) layers_[index].exchange_cache(layers[index]);
        std::swap(position_, position);
        std::swap(batch_, batch);
    }
    void prepare_training(MlxMtpLora& lora) const {
        if (layers_.size() != 1) throw std::invalid_argument("MTP LoRA currently requires one recurrent predictor layer");
        layers_.front().prepare_training(lora);
        final_mixer_.prepare_training(lora);
    }
    std::vector<MlxQwen4LayerCacheSnapshot> snapshot(int position) const {
        std::vector<MlxQwen4LayerCacheSnapshot> result;
        if (position == 0) return result;
        if (position < 0 || position > position_) throw std::runtime_error("MTP checkpoint position mismatch");
        for (const auto& layer : layers_) {
            auto state = layer.snapshot();
            state.position = position;
            if (state.offloaded_kv) state.offloaded_kv = state.offloaded_kv->prefix(position);
            else {
                state.kv->position = position;
                const Shape begin{0, 0, 0, 0};
                const Shape end{state.batch, state.kv->heads, position, state.kv->key.shape(3)};
                state.kv->key = mlx::core::slice(state.kv->key, begin, end);
                state.kv->value = mlx::core::slice(state.kv->value, begin,
                    Shape{state.batch, state.kv->heads, position, state.kv->value.shape(3)});
            }
            if (position < state.index_start)
                throw std::runtime_error("MTP checkpoint exceeds the uncommitted index tail");
            state.index_keys = detached_copy(mlx::core::slice(*state.index_keys, Shape{0, 0, 0},
                Shape{state.batch, position - state.index_start, state.index_keys->shape(2)}));
            const int pooled = position / static_cast<int>(config_.indexer_compress_ratio);
            if (state.offloaded_pooled_keys) {
                if (!pooled) state.offloaded_pooled_keys.reset();
                else state.offloaded_pooled_keys->rows = state.offloaded_pooled_keys->rows.prefix(
                    std::min<std::size_t>(pooled, state.offloaded_pooled_keys->rows.position));
            }
            if (pooled == 0 || !state.pooled_keys) state.pooled_keys.reset();
            else state.pooled_keys = mlx::core::slice(*state.pooled_keys, Shape{0, 0, 0},
                Shape{state.batch, std::min(pooled, state.pooled_keys->shape(1)), state.pooled_keys->shape(2)});
            result.push_back(std::move(state));
        }
        return result;
    }

    void restore(const std::vector<MlxQwen4LayerCacheSnapshot>& state) {
        reset(1);
        if (state.empty()) return;
        if (state.size() != layers_.size()) throw std::runtime_error("MTP checkpoint layer count mismatch");
        for (std::size_t i = 0; i < state.size(); ++i) layers_[i].restore(state[i]);
        position_ = state.front().position;
    }
    MlxMtpPredictorDescriptor mtp_descriptor() const noexcept {
        return MlxMtpPredictorDescriptor::recurrent(
            kQwen4MtpMaximumDraftDepth);
    }

    static std::optional<Qwen4Mtp> load_if_present(
        const MfqContainer& model,
        const Qwen4Config& config,
        int maximum,
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache = nullptr,
        std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache = nullptr,
        std::size_t expert_cache_layer_base = 0,
        std::shared_ptr<MlxTqExpertCache> tq_expert_cache = nullptr) {
        bool any = false;
        for (const auto& [name, _] : model.records()) {
            if (name.rfind("predictor.", 0) == 0) {
                any = true;
                break;
            }
        }
        if (config.mtp_num_hidden_layers <= 0 ||
            !model.contains("predictor.embedding_norm.weight")) {
            if (any) {
                throw std::runtime_error(
                    "Qwen4 MFQ contains an incomplete or undeclared predictor");
            }
            return std::nullopt;
        }
        if (config.mtp_use_dedicated_embeddings) {
            throw std::runtime_error(
                "Qwen4 dedicated predictor embeddings are not supported");
        }
        std::vector<Qwen4Layer> layers;
        layers.reserve(static_cast<std::size_t>(
            config.mtp_num_hidden_layers));
        for (std::int64_t index = 0;
             index < config.mtp_num_hidden_layers;
             ++index) {
            layers.push_back(Qwen4Layer::load_predictor(
                model,
                config,
                static_cast<std::size_t>(index),
                maximum,
                ssd_expert_cache,
                mfe_offload_cache,
                expert_cache_layer_base, tq_expert_cache));
        }
        return Qwen4Mtp(
            config,
            MlxRmsNorm(
                dense_vector(model, "predictor.embedding_norm.weight"),
                static_cast<float>(config.rms_norm_eps),
                1.0f),
            MlxRmsNorm(
                dense_vector(model, "predictor.hidden_norm.weight"),
                static_cast<float>(config.rms_norm_eps),
                1.0f),
            MlxLinear::load(model, "predictor.fusion.embedding.weight"),
            MlxLinear::load(model, "predictor.fusion.hidden.weight"),
            std::move(layers),
            GatedResidual::load(
                model, config, "predictor.mhc.pre", false));
    }

    void validate_offload(const std::shared_ptr<MlxMfeOffloadCache>& cache) const {
        for (const auto& layer : layers_) layer.validate_offload(cache);
    }
    std::size_t resident_routed_bytes() const noexcept {
        std::size_t bytes = 0;
        for (const auto& layer : layers_) bytes += layer.resident_routed_bytes();
        return bytes;
    }
    void use_offload(const std::shared_ptr<MlxMfeOffloadCache>& cache) {
        for (auto& layer : layers_) layer.use_offload(cache);
    }
    void validate_tq_offload(const std::shared_ptr<MlxTqExpertCache>& cache) const {
        for (const auto& layer : layers_) layer.validate_tq_offload(cache);
    }
    void use_tq_offload(const std::shared_ptr<MlxTqExpertCache>& cache) {
        for (auto& layer : layers_) layer.use_tq_offload(cache);
    }
    void offload_until(const std::shared_ptr<MlxMfeOffloadCache>& cache, std::size_t& resident, std::size_t limit) {
        for (auto& layer : layers_) {
            if (resident <= limit) break;
            const auto bytes = layer.resident_routed_bytes();
            if (bytes) { layer.use_offload(cache); resident -= bytes; }
        }
    }
    void reset(int batch = 1) {
        if (batch <= 0) {
            throw std::runtime_error(
                "Qwen4 predictor cache batch must be positive");
        }
        for (auto& layer : layers_) layer.reset(batch);
        batch_ = batch;
        position_ = 0;
    }

    void clear() noexcept {
        for (auto& layer : layers_) layer.clear();
        batch_ = 0;
        position_ = 0;
    }

    Qwen4MtpForward forward(
        const array& previous_hidden,
        const array& token_ids,
        const MlxEmbedding& embedding,
        bool use_cache,
        MlxMtpLora* lora = nullptr) {
        MlxMtpLoraScope scope(lora);
        auto input = fuse_inputs(previous_hidden, token_ids, embedding, use_cache);
        const int batch = token_ids.shape(0);
        const int tokens = token_ids.shape(1);
        const int start = use_cache ? position_ : 0;
        auto full_positions = prompt_positions.full(start + tokens);
        auto current_positions = prompt_positions.current(full_positions, start, tokens);
        auto streams = layers_.at(0).forward(
            std::move(input.streams), input.ids,
            current_positions, full_positions, use_cache);
        auto sample_hidden = final_mixer_.mix(streams);
        if (lora) lora->set_final_graph([this](const MlxMtpLoraParameters& p, const std::vector<array>& c) {
            return final_mixer_.training_pre(c.at(0), p).branch;
        });
        if (use_cache) {
            batch_ = batch;
            position_ = start + tokens;
        }
        return {std::move(sample_hidden), std::move(streams)};
    }

    Qwen4MtpForward forward_requests(const array& previous_hidden, const array& token_ids,
        const MlxEmbedding& embedding, const std::vector<std::vector<Qwen4RequestLayerCache>*>& caches,
        const std::vector<int>& positions) {
        auto input = fuse_inputs(previous_hidden, token_ids, embedding, false);
        auto streams = std::move(input.streams);
        std::vector<Qwen4RequestLayerCache*> layers;
        layers.reserve(caches.size());
        for (std::size_t index = 0; index < layers_.size(); ++index) {
            layers.clear();
            for (auto* cache : caches) layers.push_back(&cache->at(index));
            streams = layers_[index].decode_requests(std::move(streams), input.ids, layers, positions);
        }
        auto sampled = final_mixer_.mix(mlx::core::reshape(streams, {1, streams.shape(0), streams.shape(2)}));
        return {std::move(sampled), std::move(streams)};
    }

    array append_history(
        const array& previous_hidden,
        const array& token_ids,
        const MlxEmbedding& embedding) {
        auto input = fuse_inputs(previous_hidden, token_ids, embedding, true);
        const int tokens = token_ids.shape(1);
        auto full_positions = prompt_positions.full(position_ + tokens);
        auto current_positions = prompt_positions.current(full_positions, position_, tokens);
        auto anchor = layers_.at(0).append_history(
            std::move(input.streams), input.ids, current_positions, full_positions);
        position_ += tokens;
        return anchor;
    }

    int cache_position() const noexcept {
        return position_;
    }

    Qwen4PromptPositions prompt_positions;

    std::size_t cache_bytes() const noexcept {
        std::size_t bytes = 0;
        for (const auto& layer : layers_) bytes += layer.cache_bytes();
        return bytes;
    }

    void trim_cache_to(int position) {
        if (position < 0 || position > position_) {
            throw std::runtime_error(
                "Qwen4 predictor cache trim position is invalid");
        }
        for (auto& layer : layers_) layer.trim_cache_to(position);
        position_ = position;
    }

private:
    struct FusedInput {
        array streams;
        array ids;
    };

    FusedInput fuse_inputs(
        const array& previous_hidden,
        const array& token_ids,
        const MlxEmbedding& embedding,
        bool use_cache) {
        if (token_ids.ndim() != 2 || token_ids.shape(0) <= 0 ||
            token_ids.shape(1) <= 0) {
            throw std::runtime_error(
                "Qwen4 predictor token IDs must have non-empty [B,T] shape");
        }
        if (previous_hidden.ndim() != 3) {
            throw std::runtime_error(
                "Qwen4 predictor hidden input must have [B,T,H] shape");
        }
        if (previous_hidden.shape(0) != token_ids.shape(0)) {
            throw std::runtime_error(
                "Qwen4 predictor hidden/token batch sizes disagree");
        }
        if (previous_hidden.shape(1) != token_ids.shape(1)) {
            throw std::runtime_error(
                "Qwen4 predictor hidden/token lengths disagree: hidden=" +
                std::to_string(previous_hidden.shape(1)) + ", tokens=" +
                std::to_string(token_ids.shape(1)));
        }
        if (previous_hidden.shape(2) !=
            config_.hc_count * config_.hidden_size) {
            throw std::runtime_error(
                "Qwen4 predictor hidden width disagrees: got=" +
                std::to_string(previous_hidden.shape(2)) + ", expected=" +
                std::to_string(config_.hc_count * config_.hidden_size));
        }
        const int batch = token_ids.shape(0);
        const int tokens = token_ids.shape(1);
        if (use_cache && (batch_ == 0 || batch_ != batch)) reset(batch);
        auto ids = token_ids.dtype() == mlx::core::int32
            ? token_ids
            : mlx::core::astype(token_ids, mlx::core::int32);
        auto embeds = embedding(ids, mlx::core::float16);
        auto embedding_branch = embedding_fusion_(embedding_norm_(embeds));
        auto hidden = mlx::core::reshape(
            hidden_norm_(previous_hidden),
            Shape{
                batch,
                tokens,
                static_cast<int>(config_.hc_count),
                static_cast<int>(config_.hidden_size),
            });
        auto streams = mlx::core::reshape(
            hidden_fusion_(hidden) +
                mlx::core::expand_dims(embedding_branch, -2),
            Shape{
                batch,
                tokens,
                static_cast<int>(
                    config_.hc_count * config_.hidden_size),
            });
        return {std::move(streams), std::move(ids)};
    }

    Qwen4Mtp(
        Qwen4Config config,
        MlxRmsNorm embedding_norm,
        MlxRmsNorm hidden_norm,
        MlxLinear embedding_fusion,
        MlxLinear hidden_fusion,
        std::vector<Qwen4Layer> layers,
        GatedResidual final_mixer)
        : config_(std::move(config)),
          embedding_norm_(std::move(embedding_norm)),
          hidden_norm_(std::move(hidden_norm)),
          embedding_fusion_(std::move(embedding_fusion)),
          hidden_fusion_(std::move(hidden_fusion)),
          layers_(std::move(layers)),
          final_mixer_(std::move(final_mixer)) {}

    Qwen4Config config_;
    MlxRmsNorm embedding_norm_;
    MlxRmsNorm hidden_norm_;
    MlxLinear embedding_fusion_;
    MlxLinear hidden_fusion_;
    std::vector<Qwen4Layer> layers_;
    GatedResidual final_mixer_;
    int batch_ = 0;
    int position_ = 0;
};

} // namespace

struct MlxQwen4RequestCache {
    const void* owner = nullptr;
    int position = 0, batch = 0;
    bool speculative_pending = false;
    int draft_count = 0;
    std::vector<Qwen4RequestLayerCache> layers;
    std::vector<Qwen4RequestLayerCache> predictor_layers;
    int predictor_position = 0, predictor_batch = 0, predictor_history_position = 0, folded_history = 0;
    std::optional<array> last_hidden;
    std::optional<array> verified_hidden;
    std::optional<array> draft_hidden, draft_ids, draft_result;
    std::vector<std::int32_t> verified_ids;
    MlxMtpHistoryBuffer pending_history;
    mfq::engine::mtp::PolicyState policy;
    std::size_t bytes() const noexcept {
        std::size_t result = 0;
        const auto add_layers = [&](const auto& list) { for (const auto& layer : list) {
            if (layer.kv) result += layer.kv->key_storage().nbytes() + layer.kv->value_storage().nbytes();
            if (layer.offloaded_kv) result += layer.offloaded_kv->resident_bytes();
            if (layer.pooled) result += layer.pooled->storage_bytes();
            for (const auto* value : {&layer.convolution, &layer.recurrent, &layer.index_tail, &layer.ple_convolution})
                if (*value) result += (*value)->nbytes();
        }};
        add_layers(layers);
        add_layers(predictor_layers);
        result += pending_history.bytes();
        if (last_hidden) result += last_hidden->nbytes();
        if (verified_hidden) result += verified_hidden->nbytes();
        for (const auto* value : {&draft_hidden, &draft_ids, &draft_result}) if (*value) result += (*value)->nbytes();
        return result;
    }
};

std::pair<std::int32_t, array> qwen4_mtp_logits_from_state(
    const MfqContainer& model, const MfqContainer& predictor_source,
    const MlxQwen4TextSessionState& state, bool fp16) {
    if (!state.last_hidden || state.tokens.empty() || state.mtp_layers.empty())
        throw std::invalid_argument("MTP probe requires a committed predictor checkpoint");
    const auto config = Qwen4Config::from_mfq(model);
    auto embedding = MlxEmbedding::load(model, "model.token_embedding.weight");
    auto head = MlxLinear::load(model, "model.output.weight");
    auto mixer = GatedResidual::load(model, config, "model.mhc.pre", false);
    auto pending_logits = head(mixer.mix(*state.last_hidden));
    const auto pending = mlx::core::argmax(mlx::core::reshape(pending_logits, {-1})).item<std::int32_t>();
    auto predictor = [&] {
        MlxFp16WeightScope precision(fp16);
        return Qwen4Mtp::load_if_present(predictor_source, config, state.mtp_layers.front().kv->maximum_sequence);
    }();
    if (!predictor) throw std::invalid_argument("MTP probe source has no predictor");
    predictor->restore(state.mtp_layers);
    const array token({pending}, Shape{1, 1}, mlx::core::int32);
    auto result = predictor->forward(*state.last_hidden, token, embedding, true);
    auto logits = mlx::core::astype(mlx::core::reshape(head(result.sample_hidden), {-1}), mlx::core::float32);
    logits.eval();
    return {pending, std::move(logits)};
}

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
    const auto rope_type = rope.value("rope_type", rope.value("type", std::string("default")));
    config.yarn_max_factor = rope_type == "default" || rope_type == "yarn" ? rope.value("factor", 4.0) : 1.0;
    config.yarn.beta_fast = rope.value("beta_fast", 32.0);
    config.yarn.beta_slow = rope.value("beta_slow", 1.0);
    if (rope_type == "yarn") config.max_position_embeddings = rope.value("original_max_position_embeddings", config.max_position_embeddings);
    (void)qwen_yarn_capacity(config.max_position_embeddings, config.yarn_max_factor);
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
    if (outer.contains("vision_config") && !outer.at("vision_config").is_null()) {
        const auto& vision = outer.at("vision_config");
        if (vision.value("hidden_act", std::string("gelu_pytorch_tanh")) != "gelu_pytorch_tanh")
            throw std::runtime_error("Qwen grid-ViT requires gelu_pytorch_tanh hidden_act");
        GridVisionConfig v;
        v.hidden_size = positive(vision, "hidden_size");
        v.intermediate_size = positive(vision, "intermediate_size");
        v.depth = positive(vision, "depth");
        v.num_heads = positive(vision, "num_heads");
        v.in_channels = vision.value("in_channels", 3);
        v.patch_size = positive(vision, "patch_size");
        v.temporal_patch_size = positive(vision, "temporal_patch_size");
        v.spatial_merge_size = positive(vision, "spatial_merge_size");
        v.out_hidden_size = positive(vision, "out_hidden_size");
        v.num_position_embeddings = positive(vision, "num_position_embeddings");
        v.rope_theta = vision.value("rope_parameters", json::object()).value(
            "rope_theta", vision.value("rope_theta", 10000.0));
        v.layer_norm_eps = vision.value("layer_norm_eps", 1e-6);
        v.validate();
        if (v.out_hidden_size != config.hidden_size)
            throw std::runtime_error("Qwen4 vision merger width disagrees with text hidden size");
        config.vision = v;
        for (const auto* name : {"image_token_id", "video_token_id"}) {
            if (!outer.contains(name) || outer.at(name).is_null()) continue;
            const auto token = outer.at(name).get<std::int64_t>();
            if (token < 0 || token >= config.vocab_size)
                throw std::runtime_error("Qwen4 media token is out of range");
            (std::string_view(name) == "image_token_id" ? config.image_token_id : config.video_token_id) = token;
        }
    }
    return config;
}

Qwen4Config Qwen4Config::from_mfq(const MfqContainer& model) {
    if (!model.contains(std::string(kModelConfigAsset))) {
        throw std::runtime_error("Qwen4 MFQ has no embedded model config");
    }
    return from_json(model.read_text(std::string(kModelConfigAsset)));
}

struct MlxQwen4CausalLm::Impl {
    void exchange_request(MlxQwen4RequestCache& cache) {
        if (cache.owner != this || cache.layers.size() != layers.size())
            throw std::invalid_argument("Qwen4 request cache belongs to a different runtime");
        for (std::size_t index = 0; index < layers.size(); ++index)
            layers[index].exchange_cache(cache.layers[index]);
        if (mtp) mtp->exchange_cache(cache.predictor_layers, cache.predictor_position, cache.predictor_batch);
        last_cache_hidden.swap(cache.last_hidden);
        std::swap(cache_position, cache.position);
        std::swap(cache_batch, cache.batch);
        std::swap(speculative_pending, cache.speculative_pending);
    }
    struct RequestScope {
        Impl& runtime;
        MlxQwen4RequestCache& cache;
        RequestScope(Impl& runtime, MlxQwen4RequestCache& cache) : runtime(runtime), cache(cache) {
            runtime.exchange_request(cache);
        }
        ~RequestScope() { runtime.exchange_request(cache); }
    };
    static std::unique_ptr<Impl> load(
        const MfqContainer& model,
        int requested_context,
        std::optional<std::size_t> expert_cache_bytes) {
        const auto graph = effective_model_graph(model);
        if (graph.graph_kind != "causal_lm" ||
            graph.backbone != "qwen4_exp" ||
            !graph.has_component("text")) {
            throw std::runtime_error(
                "model graph does not describe a Qwen4-Exp causal runtime");
        }
        auto config = Qwen4Config::from_mfq(model);
        config.yarn = qwen_yarn_scaling(config.yarn, config.max_position_embeddings, requested_context, config.yarn_max_factor);
        config.max_position_embeddings = requested_context;
        const auto* mtp_source_path = std::getenv("MFQ_MTP_SOURCE");
        const bool full_precision_mtp = mtp_fp16_requested();
        std::unique_ptr<MfqContainer> mtp_source;
        std::string mtp_identity;
        if (full_precision_mtp) {
            if (mtp_source_path && *mtp_source_path)
                mtp_source = std::make_unique<MfqContainer>(std::filesystem::path(mtp_source_path), "mtp.");
            const auto& source = mtp_source ? *mtp_source : model;
            const auto other = Qwen4Config::from_mfq(source);
            const auto shape = [](const Qwen4Config& c) { return std::tuple{
                c.hidden_size, c.vocab_size, c.num_hidden_layers, c.hc_count, c.hc_lowrank,
                c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rotary_dim,
                c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size, c.shared_expert_intermediate_size,
                c.indexer_n_heads, c.indexer_head_dim, c.indexer_compress_ratio, c.indexer_budget,
                c.mtp_num_hidden_layers, c.rms_norm_eps, c.rope_theta, c.rope_sections,
                c.mrope_interleaved, c.norm_topk_prob, c.mtp_use_dedicated_embeddings}; };
            if (shape(config) != shape(other) || config.mtp_num_hidden_layers <= 0)
                throw std::invalid_argument("TTT FP16 predictor source does not match model topology");
            for (const auto& [name, record] : source.records()) if (name.starts_with("predictor.")) {
                if (record.dtype == "MFE" && source.is_hf_source()) continue;
                if (record.dtype != "F16" && record.dtype != "BF16" && record.dtype != "F32")
                    throw std::invalid_argument("FP16 MTP requires original floating predictor weights; set MFQ_MTP_SOURCE to the original checkpoint");
            }
            mtp_identity = "mtp-fp16-output-lora-v1:" + mtp_source_identity(model) + ':' + mtp_source_identity(source);
        }
        const auto* predictor = graph.component("predictor");
        if (predictor != nullptr &&
            (predictor->implementation != "next_token_prediction" ||
             predictor->tensor_root != "predictor")) {
            throw std::runtime_error(
                "model graph declares an unsupported Qwen4 predictor");
        }
        const int maximum = std::min(
            checked_int(config.max_position_embeddings, "context"),
            requested_context);
        auto kv_offload = mlx_configure_qsa_kv_offload(maximum,
            static_cast<int>(std::count(config.layer_types.begin(), config.layer_types.end(), "full_attention")) +
                (predictor ? static_cast<int>(config.mtp_num_hidden_layers) : 0),
            static_cast<int>(config.num_key_value_heads), static_cast<int>(config.head_dim),
            static_cast<int>(config.indexer_head_dim), static_cast<int>(config.indexer_compress_ratio));
        auto embedding = MlxEmbedding::load(
            model, "model.token_embedding.weight");
        std::optional<MlxLinear> output;
        if (!config.tie_word_embeddings && model.contains("model.output.weight")) {
            output.emplace(MlxLinear::load(model, "model.output.weight"));
        }
        std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache;
        std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache;
        std::shared_ptr<MlxTqExpertCache> tq_expert_cache;
        if (expert_cache_bytes.has_value() && *expert_cache_bytes > 0) {
            std::vector<std::string> prefixes;
            prefixes.reserve(static_cast<std::size_t>(
                config.num_hidden_layers + config.mtp_num_hidden_layers));
            for (std::int64_t index = 0;
                 index < config.num_hidden_layers; ++index) {
                prefixes.push_back(
                    "model.block." + std::to_string(index));
            }
            if (predictor != nullptr && !full_precision_mtp) {
                for (std::int64_t index = 0;
                     index < config.mtp_num_hidden_layers; ++index) {
                    prefixes.push_back(
                        "predictor.block." + std::to_string(index));
                }
            }
            constexpr std::size_t prefill_buffers_minimum =
                std::size_t{7} << 30;
            if (model.contains("model.block.0.mlp.experts.gate.weight")
                && model.record("model.block.0.mlp.experts.gate.weight").dtype == "TQ") {
                tq_expert_cache = std::make_shared<MlxTqExpertCache>(model, prefixes, *expert_cache_bytes, 8);
            } else try {
                ssd_expert_cache = std::make_shared<MlxMoeSsdExpertCache>(
                    model,
                    prefixes,
                    static_cast<std::size_t>(config.hidden_size),
                    static_cast<std::size_t>(config.moe_intermediate_size),
                    static_cast<std::size_t>(config.num_experts),
                    *expert_cache_bytes,
                    8,
                    *expert_cache_bytes >= prefill_buffers_minimum);
            } catch (const MlxMfeMxfp4Unsupported&) {
                mfe_offload_cache = std::make_shared<MlxMfeOffloadCache>(
                    model,
                    *expert_cache_bytes,
                    static_cast<int>(config.num_experts));
            }
        }
        std::vector<Qwen4Layer> layers;
        layers.reserve(static_cast<std::size_t>(config.num_hidden_layers));
        for (std::size_t index = 0;
             index < static_cast<std::size_t>(config.num_hidden_layers);
             ++index) {
            layers.push_back(Qwen4Layer::load(
                model,
                config,
                index,
                maximum,
                ssd_expert_cache,
                mfe_offload_cache, tq_expert_cache));
        }
        auto mixer = GatedResidual::load(
            model, config, "model.mhc.pre", false);
        MlxFp16WeightScope fp16_mtp(full_precision_mtp);
        auto mtp = Qwen4Mtp::load_if_present(
            mtp_source ? *mtp_source : model,
            config,
            maximum,
            full_precision_mtp ? nullptr : ssd_expert_cache,
            full_precision_mtp ? nullptr : mfe_offload_cache,
            static_cast<std::size_t>(config.num_hidden_layers),
            full_precision_mtp ? nullptr : tq_expert_cache);
        if ((predictor != nullptr) != mtp.has_value()) {
            throw std::runtime_error(
                predictor != nullptr
                    ? "model graph declares a missing Qwen4 predictor"
                    : "Qwen4 predictor tensors are absent from the model graph");
        }
        if (mtp && graph.topology.predictor_layers !=
                config.mtp_num_hidden_layers) {
            throw std::runtime_error(
                "Qwen4 predictor topology disagrees with model config");
        }
        auto result = std::unique_ptr<Impl>(new Impl(
            std::move(config),
            maximum,
            std::move(embedding),
            std::move(output),
            std::move(layers),
            std::move(mixer),
            std::move(mtp),
            std::move(ssd_expert_cache),
            std::move(mfe_offload_cache), std::move(tq_expert_cache)));
        result->source = std::make_unique<MfqContainer>(model);
        if (const auto* vision = graph.component("vision")) {
            if (vision->implementation != "grid_vit" || !result->config.vision ||
                !result->config.image_token_id || !result->config.video_token_id ||
                result->config.image_token_id == result->config.video_token_id ||
                result->config.rope_sections.empty())
                throw std::runtime_error("model graph Vision component is incompatible with Qwen4");
            result->vision.emplace(MlxGridVisionPromptComponent::load(
                model, *result->config.vision, *result->config.image_token_id,
                *result->config.video_token_id, vision->input_contract, vision->position_policy));
        }
        result->mtp_base_key = std::move(mtp_identity);
        result->mtp_full_precision = full_precision_mtp;
        // Count backing payloads once at load time, not on every UI refresh.
        for (const auto layer : result->config.ple_layer_ids) {
            const auto prefix = "model.block." + std::to_string(layer - 1) +
                ".position_embedding.ngram.shard.";
            for (std::int64_t shard = 0; shard < result->config.split_ngram_parts; ++shard) {
                result->ple_payload_bytes += model.record(prefix +
                    std::to_string(shard) + ".weight").nbytes;
            }
        }
        for (const auto& root : {std::string("model"), std::string("predictor")}) {
            const auto count = root == "model" ? result->config.num_hidden_layers :
                (result->mtp && !full_precision_mtp ? result->config.mtp_num_hidden_layers : 0);
            for (std::int64_t layer = 0; layer < count; ++layer) {
                const auto prefix = root + ".block." + std::to_string(layer) + ".mlp.experts.";
                for (const auto* projection : {"gate", "up", "gate_up", "down"}) {
                    for (const auto* suffix : {".weight", ".weight_scale"}) {
                        const auto name = prefix + projection + suffix;
                        if (model.contains(name)) result->expert_payload_bytes += model.record(name).nbytes;
                    }
                }
            }
        }
        if (auto* preparation = MlxKernelPreparation::current()) {
            preparation->collect([&] {
                for (const auto rows : preparation->row_buckets()) {
                    const auto streams = mlx::core::zeros(
                        Shape{1, rows, static_cast<int>(result->config.hc_count * result->config.hidden_size)}, mlx::core::float16);
                    const auto hidden = result->mixer.mix(streams);
                    preparation->add(result->output ? (*result->output)(hidden) : result->embedding.project(hidden));
                }
            });
            if (result->mtp) result->mtp->prepare_inputs(result->embedding);
        }
        return result;
    }

    std::pair<array, array> forward_with_hidden(
        const array& token_ids,
        bool use_cache,
        bool last_token_only = false,
        int speculative_confirmed = 0,
        const std::optional<array>& embeddings = std::nullopt) {
        if (token_ids.ndim() != 2 || token_ids.shape(0) <= 0 ||
            token_ids.shape(1) <= 0) {
            throw std::runtime_error("Qwen4 token IDs must have [B,T] shape");
        }
        const int batch = token_ids.shape(0);
        const int tokens = token_ids.shape(1);
        if (speculative_confirmed < 0 ||
            speculative_confirmed >= tokens ||
            (speculative_confirmed > 0 &&
             (!use_cache || speculative_pending))) {
            throw std::runtime_error(
                "Qwen4 speculative forward configuration is invalid");
        }
        if (use_cache && (cache_batch == 0 || cache_batch != batch)) {
            reset(batch);
        }
        const int start = use_cache ? cache_position : 0;
        if (start + tokens > maximum) {
            throw std::runtime_error("Qwen4 input exceeds context capacity");
        }
        auto ids = token_ids.dtype() == mlx::core::int32
            ? token_ids : mlx::core::astype(token_ids, mlx::core::int32);
        std::vector<Qwen4NgramEmbedding::Prefetched> ple_pipeline;
        if (!config.ple_layer_ids.empty()) {
            ple_pipeline.resize(layers.size());
            for (const auto layer : config.ple_layer_ids) {
                const auto index = static_cast<std::size_t>(layer - 1);
                ple_pipeline[index] = layers[index].prefetch_ple(ids, use_cache);
            }
        }
        auto hidden = embeddings ? mlx::core::astype(*embeddings, mlx::core::float16)
                                 : embedding(ids, mlx::core::float16);
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
        auto full_positions = prompt_positions.full(start + tokens);
        auto current_positions = prompt_positions.current(full_positions, start, tokens);
        const bool bounded_prefill = tokens > 1;
        const bool dispatch_decode = batch == 1 && tokens <= 6 && decode_async_enabled() &&
            !detail::component_profile_active();
        const auto dispatch_interval = static_cast<std::size_t>(decode_async_interval());
        const auto rows = static_cast<std::size_t>(batch) *
            static_cast<std::size_t>(tokens);
        std::array<
            std::optional<MlxSsdPrefetchedExpertLayer>,
            2> routed_pipeline;
        if (bounded_prefill && !layers.empty()) {
            routed_pipeline[0] = layers[0].prefetch_routed(rows);
            if (routed_pipeline[0].has_value() && layers.size() > 1) {
                routed_pipeline[1] = layers[1].prefetch_routed(rows);
            }
        }
        for (std::size_t index = 0; index < layers.size(); ++index) {
            if (use_cache && batch == 1 && tokens == 1 && !speculative_confirmed &&
                !MlxMtpLora::current() && !detail::component_profile_active()) {
                std::vector<Qwen4CachedRoute> routes;
                std::vector<Qwen4Layer::RouteCheckpoint> states;
                for (auto next = index; next < layers.size() && next < index + 4; ++next) {
                    auto route = layers[next].cached_route();
                    if (!route) break;
                    routes.push_back(std::move(*route));
                    states.push_back(layers[next].route_checkpoint());
                }
                if (routes.size() > 1) {
                    auto trial = streams;
                    const auto restore = [&] {
                        for (std::size_t offset = 0; offset < states.size(); ++offset)
                            layers[index + offset].restore_route_checkpoint(states[offset]);
                    };
                    try {
                        for (std::size_t offset = 0; offset < routes.size(); ++offset)
                            trial = layers[index + offset].forward(std::move(trial), ids,
                                current_positions, full_positions, true, 0, nullptr, nullptr, &routes[offset]);
                        mlx::core::async_eval({*routes.back().ids});
                        mlx::core::async_eval({trial});
                        std::vector<bool> hits;
                        for (auto& route : routes) hits.push_back(route.all_resident());
                        const bool all_hit = std::all_of(hits.begin(), hits.end(), [](bool hit) { return hit; });
                        for (std::size_t offset = 0; offset < routes.size(); ++offset)
                            if (all_hit || !hits[offset]) routes[offset].resolve();
                        if (all_hit) {
                            streams = std::move(trial);
                            index += routes.size() - 1;
                            continue;
                        }
                        trial.eval();
                    } catch (...) {
                        restore();
                        throw;
                    }
                    restore();
                }
            }
            auto* prefetched = routed_pipeline[index % 2].has_value()
                ? &*routed_pipeline[index % 2]
                : nullptr;
            streams = layers[index].forward(
                std::move(streams),
                ids,
                current_positions,
                full_positions,
                use_cache,
                speculative_confirmed,
                prefetched,
                ple_pipeline.empty() ? nullptr : &ple_pipeline[index]);
            if (prefetched != nullptr) {
                detail::eval_with_timing(streams);
                routed_pipeline[index % 2].reset();
                if (index + 2 < layers.size()) {
                    routed_pipeline[index % 2] =
                        layers[index + 2].prefetch_routed(rows);
                }
            } else if (dispatch_decode &&
                       (index < 6 || index % dispatch_interval == 0)) {
                // Fill the GPU queue promptly, then amortize commit overhead
                // over layer groups while keeping host/GPU work overlapped.
                mlx::core::async_eval({streams});
            }
        }
        auto output_hidden = mixer.mix(streams);
        detail::profile_eval("qwen4.final_mhc", output_hidden);
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
        if (speculative_confirmed > 0) {
            speculative_pending = true;
        }
        return {std::move(logits), std::move(streams)};
    }

    array forward(
        const array& token_ids,
        bool use_cache,
        bool last_token_only = false) {
        return forward_with_hidden(
            token_ids, use_cache, last_token_only).first;
    }

    void commit_speculative() {
        if (!speculative_pending) {
            throw std::runtime_error(
                "Qwen4 has no speculative target transaction");
        }
        for (auto& layer : layers) layer.commit_speculative();
        speculative_pending = false;
    }

    void rollback_speculative(
        int accepted_drafts,
        int draft_count) {
        if (!speculative_pending || draft_count <= 0 ||
            accepted_drafts < 0 || accepted_drafts >= draft_count ||
            cache_position < draft_count - accepted_drafts) {
            throw std::runtime_error(
                "Qwen4 speculative rollback range is invalid");
        }
        for (auto& layer : layers) {
            layer.rollback_speculative(accepted_drafts);
        }
        cache_position -= draft_count - accepted_drafts;
        speculative_pending = false;
    }

    void reset(int batch, bool reset_predictor = true) {
        if (batch <= 0) {
            throw std::runtime_error("Qwen4 cache batch must be positive");
        }
        for (auto& layer : layers) layer.reset(batch);
        if (mtp && reset_predictor) mtp->reset(batch);
        cache_batch = batch;
        cache_position = 0;
        speculative_pending = false;
        stable_tokens.clear();
        last_cache_hidden.reset();
        stable_mtp_ready = false;
    }

    void clear() noexcept {
        for (auto& layer : layers) layer.clear();
        if (mtp) mtp->clear();
        cache_batch = 0;
        cache_position = 0;
        speculative_pending = false;
        stable_tokens.clear();
        last_cache_hidden.reset();
        stable_mtp_ready = false;
    }

private:
    Impl(
        Qwen4Config selected_config,
        int selected_maximum,
        MlxEmbedding selected_embedding,
        std::optional<MlxLinear> selected_output,
        std::vector<Qwen4Layer> selected_layers,
        GatedResidual selected_mixer,
        std::optional<Qwen4Mtp> selected_mtp,
        std::shared_ptr<MlxMoeSsdExpertCache> selected_ssd_expert_cache,
        std::shared_ptr<MlxMfeOffloadCache> selected_mfe_offload_cache,
        std::shared_ptr<MlxTqExpertCache> selected_tq_expert_cache)
        : config(std::move(selected_config)),
          maximum(selected_maximum),
          embedding(std::move(selected_embedding)),
          output(std::move(selected_output)),
          layers(std::move(selected_layers)),
          mixer(std::move(selected_mixer)),
          mtp(std::move(selected_mtp)),
          ssd_expert_cache(std::move(selected_ssd_expert_cache)),
          mfe_offload_cache(std::move(selected_mfe_offload_cache)),
          tq_expert_cache(std::move(selected_tq_expert_cache)) {}

public:
    Qwen4Config config;
    std::unique_ptr<MfqContainer> source;
    std::string mtp_base_key;
    bool mtp_full_precision = false;
    std::unique_ptr<MlxMtpLora> mtp_lora;
    int maximum;
    MlxEmbedding embedding;
    std::optional<MlxLinear> output;
    std::optional<MlxGridVisionPromptComponent> vision;
    Qwen4PromptPositions prompt_positions;
    std::vector<Qwen4Layer> layers;
    GatedResidual mixer;
    std::optional<Qwen4Mtp> mtp;
    std::shared_ptr<MlxMoeSsdExpertCache> ssd_expert_cache;
    std::shared_ptr<MlxMfeOffloadCache> mfe_offload_cache;
    std::shared_ptr<MlxTqExpertCache> tq_expert_cache;
    MlxMtpGenerationStats last_mtp_stats;
    mfq::engine::mtp::PolicyState mtp_policy_state;
    int cache_batch = 0;
    int cache_position = 0;
    bool speculative_pending = false;
    std::vector<std::int64_t> stable_tokens;
    std::optional<array> last_cache_hidden;
    bool stable_mtp_ready = false;
    std::size_t ple_payload_bytes = 0;
    std::size_t expert_payload_bytes = 0;
    std::vector<std::weak_ptr<MlxQwen4RequestCache>> request_caches;
};

MlxQwen4CausalLm::MlxQwen4CausalLm(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)), kv_offload_store_(mlx_qsa_kv_offload_store()) {}

MlxQwen4CausalLm::~MlxQwen4CausalLm() = default;
MlxQwen4CausalLm::MlxQwen4CausalLm(MlxQwen4CausalLm&&) noexcept = default;
MlxQwen4CausalLm& MlxQwen4CausalLm::operator=(MlxQwen4CausalLm&&) noexcept = default;

MlxQwen4CausalLm MlxQwen4CausalLm::load(
    const MfqContainer& model,
    int max_context,
    std::optional<std::size_t> expert_cache_bytes) {
    if (max_context <= 0) {
        throw std::invalid_argument("Qwen4 max_context must be positive");
    }
    if (auto* preparation = MlxKernelPreparation::current(); preparation && !expert_cache_bytes)
        preparation->use_model_paths();
    return MlxQwen4CausalLm(Impl::load(
        model, max_context, expert_cache_bytes));
}

array MlxQwen4CausalLm::forward(
    const array& token_ids,
    bool use_cache) {
    return impl_->forward(token_ids, use_cache);
}

std::shared_ptr<MlxQwen4RequestCache> MlxQwen4CausalLm::make_request_cache() {
    auto cache = std::make_shared<MlxQwen4RequestCache>();
    cache->owner = impl_.get();
    cache->layers.resize(impl_->layers.size());
    if (impl_->mtp) cache->predictor_layers.resize(impl_->mtp->layer_count());
    for (auto& layer : cache->layers)
        layer.pooled.emplace(impl_->maximum, static_cast<int>(impl_->config.indexer_head_dim), mlx::core::float32);
    std::erase_if(impl_->request_caches, [](const auto& entry) { return entry.expired(); });
    impl_->request_caches.push_back(cache);
    return cache;
}

void MlxQwen4CausalLm::adopt_request_cache(MlxQwen4RequestCache& cache) {
    if (cache.position || cache.batch) throw std::invalid_argument("request cache must be empty before adoption");
    impl_->exchange_request(cache);
    cache.predictor_history_position = cache.predictor_position;
    if (impl_->mtp) impl_->mtp->clear();
    impl_->last_cache_hidden.reset();
    impl_->stable_tokens.clear();
    impl_->stable_mtp_ready = false;
}

array MlxQwen4CausalLm::prefill_request(MlxQwen4RequestCache& cache,
    const std::vector<std::int64_t>& tokens, bool mtp) {
    mlx_validate_token_set(tokens, static_cast<int>(impl_->config.vocab_size));
    if (cache.speculative_pending || tokens.empty() ||
        tokens.size() > static_cast<std::size_t>(impl_->maximum - cache.position))
        throw std::invalid_argument("Qwen4 request exceeds its context capacity");
    Impl::RequestScope scope(*impl_, cache);
    if (impl_->cache_batch != 1) impl_->reset(1, false);
    const int start = impl_->cache_position;
    const array ids(tokens.begin(), Shape{1, static_cast<int>(tokens.size())}, mlx::core::int32);
    if (!mtp) {
        auto logits = impl_->forward(ids, true, true);
        logits.eval();
        return logits;
    }
    if (!impl_->mtp) throw std::invalid_argument("request model has no predictor");
    auto result = impl_->forward_with_hidden(ids, true, true);
    mlx_prime_mtp_chunk(result.second, ids, impl_->last_cache_hidden,
        [&](const array& hidden, const array& shifted, int) {
            return impl_->mtp->append_history(hidden, shifted, impl_->embedding);
        }, start);
    cache.predictor_history_position = impl_->mtp->cache_position();
    result.first.eval();
    return result.first;
}

std::vector<array> MlxQwen4CausalLm::prefill_requests(
    const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
    const std::vector<std::vector<std::int64_t>>& tokens,
    const std::vector<bool>& mtp) {
    if (caches.empty() || caches.size() != tokens.size() || caches.size() != mtp.size())
        throw std::invalid_argument("Qwen4 prefill batch requires one token window per cache");
    std::unordered_set<const MlxQwen4RequestCache*> unique;
    std::vector<int> positions, offsets{0};
    std::vector<std::int64_t> packed;
    for (std::size_t row = 0; row < caches.size(); ++row) {
        const auto& cache = caches[row];
        if (!cache || cache->owner != impl_.get() || cache->batch < 0 || cache->batch > 1 ||
            cache->position < 0 || cache->position > impl_->maximum || cache->speculative_pending || !unique.insert(cache.get()).second ||
            tokens[row].empty() || tokens[row].size() > static_cast<std::size_t>(impl_->maximum - cache->position) ||
            (mtp[row] && !impl_->mtp))
            throw std::invalid_argument("invalid or duplicate Qwen4 prefill request cache");
        mlx_validate_token_set(tokens[row], static_cast<int>(impl_->config.vocab_size));
        positions.push_back(cache->position);
        packed.insert(packed.end(), tokens[row].begin(), tokens[row].end());
        if (packed.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Qwen4 prefill batch is too large");
        offsets.push_back(static_cast<int>(packed.size()));
    }
    if (caches.size() == 1) return {prefill_request(*caches[0], tokens[0], mtp[0])};
    const int count = static_cast<int>(packed.size());
    const array ids(packed.begin(), Shape{1, count}, mlx::core::int32);
    std::vector<Qwen4NgramEmbedding::Prefetched> ple_pipeline(impl_->layers.size());
    for (const auto layer : impl_->config.ple_layer_ids) {
        const auto index = static_cast<std::size_t>(layer - 1);
        std::vector<Qwen4RequestLayerCache*> state;
        for (const auto& cache : caches) state.push_back(&cache->layers[index]);
        ple_pipeline[index] = impl_->layers[index].prefetch_ple_requests(ids, offsets, state);
    }
    auto hidden = impl_->embedding(ids, mlx::core::float16);
    auto streams = mlx::core::reshape(mlx::core::broadcast_to(mlx::core::expand_dims(hidden, -2),
        Shape{1, count, static_cast<int>(impl_->config.hc_count), static_cast<int>(impl_->config.hidden_size)}),
        Shape{1, count, static_cast<int>(impl_->config.hc_count * impl_->config.hidden_size)});
    std::array<std::optional<MlxSsdPrefetchedExpertLayer>, 2> routed;
    if (!impl_->layers.empty()) {
        routed[0] = impl_->layers[0].prefetch_routed(packed.size());
        if (routed[0] && impl_->layers.size() > 1) routed[1] = impl_->layers[1].prefetch_routed(packed.size());
    }
    std::vector<Qwen4RequestLayerCache*> layer_caches;
    layer_caches.reserve(caches.size());
    for (std::size_t index = 0; index < impl_->layers.size(); ++index) {
        layer_caches.clear();
        for (const auto& cache : caches) layer_caches.push_back(&cache->layers[index]);
        auto* prefetched = routed[index % 2] ? &*routed[index % 2] : nullptr;
        streams = impl_->layers[index].forward_packed_requests(std::move(streams), ids,
            layer_caches, positions, offsets, false, 0, prefetched, &ple_pipeline[index]);
        if (prefetched) {
            detail::eval_with_timing(streams);
            routed[index % 2].reset();
            if (index + 2 < impl_->layers.size())
                routed[index % 2] = impl_->layers[index + 2].prefetch_routed(packed.size());
        }
    }
    std::vector<array> last_hidden;
    for (std::size_t row = 0; row < caches.size(); ++row)
        last_hidden.push_back(mlx::core::slice(streams, {0, offsets[row + 1] - 1, 0},
            {1, offsets[row + 1], streams.shape(2)}));
    auto output_hidden = impl_->mixer.mix(mlx::core::concatenate(last_hidden, 1));
    auto logits = impl_->output ? (*impl_->output)(output_hidden) : impl_->embedding.project(output_hidden);
    for (std::size_t row = 0; row < caches.size(); ++row) {
        auto& cache = *caches[row];
        cache.position = positions[row] + static_cast<int>(tokens[row].size());
        cache.batch = 1;
        if (mtp[row]) {
            Impl::RequestScope scope(*impl_, cache);
            auto rows = mlx::core::slice(streams, {0, offsets[row], 0},
                {1, offsets[row + 1], streams.shape(2)});
            auto row_ids = mlx::core::slice(ids, {0, offsets[row]}, {1, offsets[row + 1]});
            mlx_prime_mtp_chunk(rows, row_ids, impl_->last_cache_hidden,
                [&](const array& value, const array& shifted, int) {
                    return impl_->mtp->append_history(value, shifted, impl_->embedding);
                }, positions[row]);
            cache.predictor_history_position = impl_->mtp->cache_position();
        }
    }
    logits.eval();
    std::vector<array> result;
    for (int row = 0; row < static_cast<int>(caches.size()); ++row)
        result.push_back(mlx::core::slice(logits, {0, row, 0}, {1, row + 1, logits.shape(2)}));
    return result;
}

int MlxQwen4CausalLm::request_cache_position(const MlxQwen4RequestCache& cache) const {
    if (cache.owner != impl_.get()) throw std::invalid_argument("request cache belongs to another runtime");
    return cache.position;
}

MlxMtpEngineCallbacks MlxQwen4CausalLm::request_mtp_callbacks(
    const std::shared_ptr<MlxQwen4RequestCache>& cache, bool batched_drafts) {
    if (!impl_->mtp || !cache || cache->owner != impl_.get() || !cache->last_hidden)
        throw std::invalid_argument("MTP request prefill is incomplete");
    MlxMtpEngineCallbacks callbacks;
    callbacks.predictor = impl_->mtp->mtp_descriptor();
    callbacks.policy_state = &cache->policy;
    callbacks.target_cache_position = [cache] { return cache->position; };
    callbacks.resolve_target = [this, cache](int accepted, int drafts) {
        if (drafts) resolve_request(*cache, accepted);
    };
    callbacks.committed_target = [cache](const MlxMtpDraftContext& context) {
        const int accepted = context.accepted_drafts;
        cache->last_hidden = mlx::core::slice(*context.verified_hidden, {0, accepted, 0},
            {1, accepted + 1, context.verified_hidden->shape(2)});
        cache->verified_hidden = *context.verified_hidden;
        cache->verified_ids.assign(context.next_token_ids.begin(), context.next_token_ids.end());
    };
    const auto history = [this, cache](const MlxMtpDraftContext& context) {
        Impl::RequestScope scope(*impl_, *cache);
        const int width = static_cast<int>(impl_->config.hc_count * impl_->config.hidden_size);
        auto hidden = context.initial ? *impl_->last_cache_hidden : mlx_mtp_committed_hidden(context, 1, width);
        std::vector<std::int32_t> ids = context.initial ? std::vector<std::int32_t>{context.pending_token}
            : std::vector<std::int32_t>(context.next_token_ids.begin(), context.next_token_ids.end());
        if (cache->folded_history) {
            hidden = mlx::core::slice(hidden, {0, cache->folded_history, 0}, {1, hidden.shape(1), width});
            ids.erase(ids.begin(), ids.begin() + cache->folded_history);
            cache->folded_history = 0;
        }
        impl_->mtp->trim_cache_to(cache->predictor_history_position);
        const auto fold = [&] {
            if (cache->pending_history.empty()) return;
            auto history = cache->pending_history.drain();
            (void)impl_->mtp->append_history(history.hidden, history.token_ids, impl_->embedding);
            cache->predictor_history_position += history.token_ids.shape(1);
        };
        if (!context.requested_depth) {
            cache->pending_history.append(hidden, ids);
            if (cache->pending_history.size() >= kMlxMtpHistoryChunkSize) {
                if (context.time_sample) *context.time_sample = false;
                fold();
            }
            return;
        }
        if (!cache->pending_history.empty() && context.time_sample) *context.time_sample = false;
        fold();
        const int committed = static_cast<int>(ids.size());
        const array shifted(ids.begin(), Shape{1, committed}, mlx::core::int32);
        if (committed > 1) (void)impl_->mtp->append_history(
            mlx::core::slice(hidden, {0, 0, 0}, {1, committed - 1, width}),
            mlx::core::slice(shifted, {0, 0}, {1, committed - 1}), impl_->embedding);
        cache->draft_hidden = mlx::core::slice(hidden, {0, committed - 1, 0}, {1, committed, width});
        cache->draft_ids = mlx::core::slice(shifted, {0, committed - 1}, {1, committed});
        cache->predictor_history_position += committed;
    };
    callbacks.prepare_draft = [this, cache, history](const MlxMtpDraftContext& context,
        const MlxMtpTokenSelector& select) {
        history(context);
        if (!context.requested_depth) return;
        Impl::RequestScope scope(*impl_, *cache);
        auto predictor = impl_->mtp->forward(*cache->draft_hidden, *cache->draft_ids, impl_->embedding, true);
        for (int row = 0; row < context.requested_depth; ++row) {
            auto logits = impl_->output ? (*impl_->output)(predictor.sample_hidden)
                : impl_->embedding.project(predictor.sample_hidden);
            auto token = select(mlx_last_token_logits(logits, static_cast<int>(impl_->config.vocab_size)));
            if (row + 1 < context.requested_depth) predictor = impl_->mtp->forward(predictor.hidden_streams,
                mlx::core::reshape(token, {1, 1}), impl_->embedding, true);
        }
    };
    if (batched_drafts) {
        callbacks.prepare_draft_history = history;
        callbacks.draft_logits = [cache] {
            if (!cache->draft_result) throw std::logic_error("MTP draft result is not ready");
            auto result = std::move(*cache->draft_result);
            cache->draft_result.reset();
            return result;
        };
    }
    return callbacks;
}

void MlxQwen4CausalLm::draft_request_heads(
    const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
    const std::vector<std::optional<array>>& previous_tokens) {
    if (!impl_->mtp || caches.empty() || caches.size() != previous_tokens.size())
        throw std::invalid_argument("invalid MTP draft batch");
    std::vector<array> hidden, ids;
    std::vector<int> positions;
    std::vector<std::vector<Qwen4RequestLayerCache>*> layers;
    std::unordered_set<MlxQwen4RequestCache*> unique;
    for (std::size_t row = 0; row < caches.size(); ++row) {
        if (!caches[row]) throw std::invalid_argument("null MTP draft cache");
        auto& cache = *caches[row];
        if (cache.owner != impl_.get() || cache.speculative_pending || !cache.draft_hidden ||
            !cache.draft_ids || cache.draft_result || !unique.insert(&cache).second)
            throw std::invalid_argument("invalid or duplicate MTP draft cache");
        hidden.push_back(*cache.draft_hidden);
        ids.push_back(previous_tokens[row] ? mlx::core::reshape(*previous_tokens[row], {1, 1}) : *cache.draft_ids);
        positions.push_back(cache.predictor_position);
        layers.push_back(&cache.predictor_layers);
    }
    auto predictor = [&] {
        if (caches.size() == 1) {
            Impl::RequestScope scope(*impl_, *caches.front());
            return impl_->mtp->forward(hidden.front(), ids.front(), impl_->embedding, true);
        }
        return impl_->mtp->forward_requests(mlx::core::concatenate(hidden, 0),
            mlx::core::concatenate(ids, 0), impl_->embedding, layers, positions);
    }();
    auto logits = impl_->output ? (*impl_->output)(predictor.sample_hidden)
        : impl_->embedding.project(predictor.sample_hidden);
    const int batch = static_cast<int>(caches.size()), width = predictor.hidden_streams.shape(2);
    logits = mlx::core::reshape(logits, {batch, static_cast<int>(impl_->config.vocab_size)});
    for (int row = 0; row < batch; ++row) {
        auto& cache = *caches[row];
        cache.draft_hidden = mlx::core::slice(predictor.hidden_streams, {row, 0, 0}, {row + 1, 1, width});
        cache.draft_result = mlx::core::slice(logits, {row, 0}, {row + 1, logits.shape(1)});
        if (batch > 1) {
            cache.predictor_batch = 1;
            ++cache.predictor_position;
        }
    }
}

array MlxQwen4CausalLm::decode_requests(
    const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
    const std::vector<std::int64_t>& tokens) {
    return forward_requests(caches, tokens, 1, false).first;
}

std::pair<array, array> MlxQwen4CausalLm::forward_request_tokens(
    const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
    const std::vector<std::int64_t>& tokens) {
    return forward_requests(caches, tokens, 1, false);
}

std::pair<array, array> MlxQwen4CausalLm::verify_requests(
    const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
    const std::vector<std::vector<std::int64_t>>& tokens) {
    if (tokens.empty() || tokens.size() != caches.size() || tokens.front().size() < 2 ||
        tokens.front().size() > kMlxMtpEngineMaximumDraftDepth + 1)
        throw std::invalid_argument("Qwen4 verification requires equal draft windows");
    const int rows = static_cast<int>(tokens.front().size());
    std::vector<std::int64_t> grouped;
    grouped.reserve(caches.size() * rows);
    for (const auto& request : tokens) {
        if (request.size() != static_cast<std::size_t>(rows))
            throw std::invalid_argument("Qwen4 verification requires equal draft windows");
        grouped.insert(grouped.end(), request.begin(), request.end());
    }
    return forward_requests(caches, grouped, rows, true);
}

void MlxQwen4CausalLm::resolve_request(MlxQwen4RequestCache& cache, int accepted_drafts) {
    if (!cache.speculative_pending || cache.draft_count < 1 || accepted_drafts < 0 ||
        accepted_drafts > cache.draft_count)
        throw std::invalid_argument("Qwen4 request has no matching speculative transaction");
    Impl::RequestScope scope(*impl_, cache);
    if (accepted_drafts == cache.draft_count) impl_->commit_speculative();
    else impl_->rollback_speculative(accepted_drafts, cache.draft_count);
    cache.draft_count = 0;
}

std::pair<array, array> MlxQwen4CausalLm::forward_requests(
    const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
    const std::vector<std::int64_t>& tokens, int rows, bool speculative) {
    if (caches.empty() || rows < 1 || tokens.size() != caches.size() * rows)
        throw std::invalid_argument("Qwen4 batch requires equal token windows");
    std::vector<int> offsets(caches.size() + 1);
    for (std::size_t row = 0; row <= caches.size(); ++row) offsets[row] = static_cast<int>(row * rows);
    auto result = forward_packed_requests(caches, tokens, offsets, speculative);
    result.first = mlx::core::reshape(result.first, {static_cast<int>(caches.size()), rows, result.first.shape(2)});
    result.second = mlx::core::reshape(result.second, {static_cast<int>(caches.size()), rows, result.second.shape(2)});
    return result;
}

std::vector<std::pair<array, array>> MlxQwen4CausalLm::forward_request_windows(
    const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
    const std::vector<std::vector<std::int64_t>>& windows,
    std::optional<array>* packed_logits) {
    if (caches.empty() || windows.size() != caches.size())
        throw std::invalid_argument("Qwen4 batch requires one decode window per cache");
    std::vector<int> offsets{0};
    std::vector<std::int64_t> tokens;
    for (const auto& window : windows) {
        if (window.empty() || window.size() > kMlxMtpEngineMaximumDraftDepth + 1 ||
            tokens.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) - window.size())
            throw std::invalid_argument("invalid Qwen4 decode window length");
        tokens.insert(tokens.end(), window.begin(), window.end());
        offsets.push_back(static_cast<int>(tokens.size()));
    }
    auto output = forward_packed_requests(caches, tokens, offsets, true);
    if (packed_logits) *packed_logits = output.first;
    std::vector<std::pair<array, array>> result;
    result.reserve(caches.size());
    for (std::size_t row = 0; row < caches.size(); ++row)
        result.emplace_back(mlx::core::slice(output.first, {0, offsets[row], 0},
            {1, offsets[row + 1], output.first.shape(2)}),
            mlx::core::slice(output.second, {0, offsets[row], 0},
            {1, offsets[row + 1], output.second.shape(2)}));
    return result;
}

std::pair<array, array> MlxQwen4CausalLm::forward_packed_requests(
    const std::vector<std::shared_ptr<MlxQwen4RequestCache>>& caches,
    const std::vector<std::int64_t>& tokens, const std::vector<int>& offsets, bool speculative) {
    mlx_validate_token_set(tokens, static_cast<int>(impl_->config.vocab_size));
    std::unordered_set<const MlxQwen4RequestCache*> unique;
    std::vector<int> positions;
    const int rows = offsets[1];
    bool uniform = true;
    for (std::size_t row = 0; row < caches.size(); ++row) {
        const int count = offsets[row + 1] - offsets[row];
        uniform = uniform && count == rows;
        const auto& cache = caches[row];
        if (!cache || cache->owner != impl_.get() || cache->batch != 1 ||
            cache->position < 1 || cache->position > impl_->maximum - count || cache->speculative_pending ||
            !unique.insert(cache.get()).second)
            throw std::invalid_argument("invalid or duplicate Qwen4 batch request cache");
        positions.push_back(cache->position);
    }
    if (caches.size() == 1) {
        Impl::RequestScope scope(*impl_, *caches.front());
        auto result = impl_->forward_with_hidden(array(tokens.begin(), Shape{1, rows}, mlx::core::int32),
            true, false, speculative && rows > 1 ? 1 : 0);
        if (speculative && rows > 1) caches.front()->draft_count = rows - 1;
        return result;
    }
    const int batch = static_cast<int>(caches.size());
    const int count = static_cast<int>(tokens.size());
    auto ids = array(tokens.begin(), Shape{1, count}, mlx::core::int32);
    std::vector<Qwen4NgramEmbedding::Prefetched> ple_pipeline(impl_->layers.size());
    for (const auto layer : impl_->config.ple_layer_ids) {
        const auto index = static_cast<std::size_t>(layer - 1);
        std::vector<Qwen4RequestLayerCache*> state;
        for (const auto& cache : caches) state.push_back(&cache->layers[index]);
        ple_pipeline[index] = impl_->layers[index].prefetch_ple_requests(ids, offsets, state);
    }
    auto hidden = impl_->embedding(ids, mlx::core::float16);
    auto streams = mlx::core::reshape(mlx::core::broadcast_to(mlx::core::expand_dims(hidden, -2),
        Shape{1, count, static_cast<int>(impl_->config.hc_count), static_cast<int>(impl_->config.hidden_size)}),
        Shape{1, count, static_cast<int>(impl_->config.hc_count * impl_->config.hidden_size)});
    std::vector<Qwen4RequestLayerCache*> layer_caches;
    layer_caches.reserve(batch);
    for (std::size_t index = 0; index < impl_->layers.size(); ++index) {
        layer_caches.clear();
        for (const auto& cache : caches) layer_caches.push_back(&cache->layers[index]);
        streams = impl_->layers[index].forward_packed_requests(std::move(streams), ids, layer_caches, positions,
            offsets, true, speculative && (!uniform || rows > 1) ? 1 : 0, nullptr, &ple_pipeline[index]);
        if (index < 6 || index % static_cast<std::size_t>(decode_async_interval()) == 0)
            mlx::core::async_eval({streams});
    }
    auto output_hidden = impl_->mixer.mix(streams);
    detail::profile_eval("qwen4.final_mhc", output_hidden);
    auto logits = impl_->output ? (*impl_->output)(output_hidden) : impl_->embedding.project(output_hidden);
    detail::profile_eval("qwen4.output", logits);
    for (std::size_t row = 0; row < caches.size(); ++row) {
        const auto& cache = caches[row];
        const int count = offsets[row + 1] - offsets[row];
        cache->position += count;
        cache->speculative_pending = speculative && count > 1;
        cache->draft_count = cache->speculative_pending ? count - 1 : 0;
    }
    return {std::move(logits), std::move(streams)};
}

MlxQwen4TextSessionState MlxQwen4CausalLm::capture_request_cache(MlxQwen4RequestCache& cache,
    const std::vector<std::int64_t>& tokens) {
    Impl::RequestScope scope(*impl_, cache);
    if (impl_->mtp && cache.verified_hidden && !impl_->speculative_pending &&
        cache.predictor_history_position < impl_->cache_position - 1) {
        impl_->mtp->trim_cache_to(cache.predictor_history_position);
        if (!cache.pending_history.empty()) {
            auto history = cache.pending_history.drain();
            (void)impl_->mtp->append_history(history.hidden, history.token_ids, impl_->embedding);
            cache.predictor_history_position += history.token_ids.shape(1);
        }
        const int count = impl_->cache_position - 1 - cache.predictor_history_position;
        if (count > 0) {
            if (count > static_cast<int>(cache.verified_ids.size()))
                throw std::runtime_error("MTP request checkpoint history is incomplete");
            const array shifted(cache.verified_ids.begin(), Shape{1, count}, mlx::core::int32);
            auto anchor = impl_->mtp->append_history(mlx::core::slice(*cache.verified_hidden,
                {0, 0, 0}, {1, count, cache.verified_hidden->shape(2)}), shifted, impl_->embedding);
            anchor.eval();
            cache.predictor_history_position += count;
            cache.folded_history = count;
        }
    }
    return capture_text_session_state(tokens);
}

void MlxQwen4CausalLm::restore_request_cache(MlxQwen4RequestCache& cache,
    const MlxQwen4TextSessionState& state) {
    Impl::RequestScope scope(*impl_, cache);
    restore_text_session_state(state);
    if (impl_->mtp) cache.predictor_history_position = impl_->mtp->cache_position();
    cache.pending_history = {};
    cache.verified_hidden.reset();
    cache.draft_hidden.reset();
    cache.draft_ids.reset();
    cache.draft_result.reset();
    cache.verified_ids.clear();
    cache.folded_history = 0;
    cache.draft_count = 0;
}

void MlxQwen4CausalLm::reset_cache(int batch) {
    impl_->reset(batch);
}

array MlxQwen4CausalLm::score_forward(const array& token_ids, bool last_token_only) {
    return impl_->forward(token_ids, false, last_token_only);
}

void MlxQwen4CausalLm::clear_cache() noexcept {
    impl_->clear();
}

void MlxQwen4CausalLm::reset_generation_state() noexcept {
    clear_cache();
    impl_->last_mtp_stats = {impl_->mtp.has_value(), false, 0, 0, 0};
    impl_->mtp_policy_state = {};
}

std::size_t MlxQwen4CausalLm::kv_cache_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const auto& layer : impl_->layers) bytes += layer.cache_bytes();
    if (impl_->mtp) bytes += impl_->mtp->cache_bytes();
    for (const auto& weak : impl_->request_caches) if (auto cache = weak.lock()) bytes += cache->bytes();
    if (kv_offload_store_) {
        const auto stats = kv_offload_store_->stats();
        bytes += stats.hot_bytes + stats.pending_bytes;
    }
    return bytes;
}

std::optional<QsaKvStoreStats> MlxQwen4CausalLm::qsa_kv_offload_stats() const {
    if (kv_offload_store_) return kv_offload_store_->stats();
    return std::nullopt;
}

std::size_t MlxQwen4CausalLm::kv_cache_contexts() const noexcept {
    std::size_t contexts = impl_->cache_position ? static_cast<std::size_t>(impl_->cache_batch) : 0;
    for (const auto& weak : impl_->request_caches) if (auto cache = weak.lock())
        if (cache->position) ++contexts;
    return contexts;
}

std::size_t MlxQwen4CausalLm::dynamic_weight_bytes() const noexcept {
    if (impl_->tq_expert_cache) return impl_->tq_expert_cache->resident_bytes();
    // MXFP4 arenas are preallocated and already part of the load baseline.
    return impl_->mfe_offload_cache ? impl_->mfe_offload_cache->resident_packed_bytes() : 0;
}

std::size_t MlxQwen4CausalLm::set_expert_cache_limit(std::size_t bytes) {
    if (impl_->tq_expert_cache) {
        return impl_->tq_expert_cache->set_limit(bytes);
    }
    const auto before = mlx::core::get_active_memory();
    if (!impl_->ssd_expert_cache && !impl_->mfe_offload_cache && impl_->source
        && impl_->source->contains("model.block.0.mlp.experts.gate.weight")
        && impl_->source->record("model.block.0.mlp.experts.gate.weight").dtype == "TQ") {
        if (resident_full_expert_bytes() <= bytes) return 0;
        std::vector<std::string> prefixes;
        for (std::size_t layer = 0; layer < impl_->layers.size(); ++layer)
            prefixes.push_back("model.block." + std::to_string(layer));
        if (impl_->mtp && !impl_->mtp_full_precision)
            for (int layer = 0; layer < impl_->config.mtp_num_hidden_layers; ++layer)
                prefixes.push_back("predictor.block." + std::to_string(layer));
        auto cache = std::make_shared<MlxTqExpertCache>(*impl_->source, prefixes, bytes);
        for (const auto& layer : impl_->layers) layer.validate_tq_offload(cache);
        if (impl_->mtp && !impl_->mtp_full_precision) impl_->mtp->validate_tq_offload(cache);
        for (auto& layer : impl_->layers) layer.use_tq_offload(cache);
        if (impl_->mtp && !impl_->mtp_full_precision) impl_->mtp->use_tq_offload(cache);
        impl_->tq_expert_cache = std::move(cache);
        const auto after = mlx::core::get_active_memory();
        return before > after ? before - after : 0;
    }
    if (!impl_->mfe_offload_cache) {
        if (!impl_->ssd_expert_cache && resident_full_expert_bytes() <= bytes) return 0;
        auto cache = std::make_shared<MlxMfeOffloadCache>(*impl_->source, bytes,
            static_cast<int>(impl_->config.num_experts));
        for (const auto& layer : impl_->layers) layer.validate_offload(cache);
        if (impl_->mtp && !impl_->mtp_full_precision) impl_->mtp->validate_offload(cache);
        if (impl_->ssd_expert_cache) impl_->ssd_expert_cache->release_deferred();
        if (impl_->ssd_expert_cache) {
            for (auto& layer : impl_->layers) layer.use_offload(cache);
            if (impl_->mtp && !impl_->mtp_full_precision) impl_->mtp->use_offload(cache);
            impl_->ssd_expert_cache.reset();
        }
        impl_->mfe_offload_cache = std::move(cache);
    }
    auto resident = resident_full_expert_bytes();
    if (resident > bytes) {
        for (auto& layer : impl_->layers) {
            if (resident <= bytes) break;
            const auto released = layer.resident_routed_bytes();
            if (released) { layer.use_offload(impl_->mfe_offload_cache); resident -= released; }
        }
        if (resident > bytes && impl_->mtp && !impl_->mtp_full_precision)
            impl_->mtp->offload_until(impl_->mfe_offload_cache, resident, bytes);
    }
    impl_->mfe_offload_cache->set_cache_limit(bytes > resident ? bytes - resident : 0);
    const auto after = mlx::core::get_active_memory();
    return before > after ? before - after : 0;
}

std::optional<std::size_t> MlxQwen4CausalLm::trim_preallocated_expert_cache(std::size_t bytes) {
    if (impl_->tq_expert_cache) return impl_->tq_expert_cache->trim(bytes);
    if (impl_->source && impl_->source->contains("model.block.0.mlp.experts.gate.weight")
        && impl_->source->record("model.block.0.mlp.experts.gate.weight").dtype == "TQ")
        return set_expert_cache_limit(bytes);
    return std::nullopt;
}

std::size_t MlxQwen4CausalLm::reclaimable_expert_bytes() const noexcept {
    if (impl_->tq_expert_cache) return impl_->tq_expert_cache->resident_bytes();
    if (impl_->ssd_expert_cache) return impl_->ssd_expert_cache->cache_limit_bytes();
    return resident_full_expert_bytes() + (impl_->mfe_offload_cache ? impl_->mfe_offload_cache->resident_packed_bytes() : 0);
}

std::size_t MlxQwen4CausalLm::resident_full_expert_bytes() const noexcept {
    std::size_t bytes = impl_->mtp && !impl_->mtp_full_precision ? impl_->mtp->resident_routed_bytes() : 0;
    for (const auto& layer : impl_->layers) bytes += layer.resident_routed_bytes();
    return bytes;
}

std::size_t MlxQwen4CausalLm::ssd_ple_payload_bytes() const noexcept {
    return impl_->ple_payload_bytes;
}

std::size_t MlxQwen4CausalLm::ssd_expert_payload_bytes() const noexcept {
    if (impl_->tq_expert_cache) return impl_->tq_expert_cache->backing_bytes();
    if (impl_->mfe_offload_cache) return impl_->mfe_offload_cache->backing_payload_bytes();
    return impl_->ssd_expert_cache ? impl_->expert_payload_bytes : 0;
}

std::int32_t MlxQwen4CausalLm::generate(
    const std::vector<std::int64_t>& prompt,
    const MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    const MfqTokenConstraintPtr& token_constraint,
    std::optional<std::size_t> stable_prefix_tokens,
    int prefill_chunk_size,
    const MlxPrefixCacheHooks& prefix_cache) {
    MlxPreparedPrompt prepared;
    prepared.token_ids = prompt;
    return generate_prepared_impl(prepared, sampling, max_tokens, callback,
        prefill_callback, token_constraint, stable_prefix_tokens, prefill_chunk_size, prefix_cache);
}

bool MlxQwen4CausalLm::supports_multimodal() const noexcept {
    return impl_->vision.has_value();
}

std::string_view MlxQwen4CausalLm::multimodal_input_contract() const noexcept {
    return impl_->vision ? impl_->vision->input_contract() : std::string_view{};
}

MlxPreparedPrompt MlxQwen4CausalLm::prepare_multimodal_prompt(
    const std::vector<std::int64_t>& prompt, const MlxGridMediaInput& media) const {
    if (!impl_->vision)
        throw std::invalid_argument("model graph does not provide an enabled Vision component");
    if (prompt.empty() || prompt.size() > static_cast<std::size_t>(impl_->maximum))
        throw std::invalid_argument("multimodal prompt length is invalid");
    std::vector<std::int32_t> values;
    values.reserve(prompt.size());
    for (const auto token : prompt) {
        if (token < 0 || token >= impl_->config.vocab_size)
            throw std::invalid_argument("multimodal prompt token is out of range");
        values.push_back(static_cast<std::int32_t>(token));
    }
    const array ids(values.begin(), Shape{1, static_cast<int>(values.size())}, mlx::core::int32);
    return impl_->vision->prepare(prompt, impl_->embedding(ids, mlx::core::float16), media);
}

std::int32_t MlxQwen4CausalLm::generate_prepared(
    const MlxPreparedPrompt& prompt, const MlxSamplingParams& sampling,
    std::int32_t max_tokens, const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    const MfqTokenConstraintPtr& token_constraint) {
    return generate_prepared_impl(prompt, sampling, max_tokens, callback,
        prefill_callback, token_constraint, std::nullopt, 2048, {});
}

std::int32_t MlxQwen4CausalLm::generate_prepared_impl(
    const MlxPreparedPrompt& prepared, const MlxSamplingParams& sampling,
    std::int32_t max_tokens, const std::function<bool(std::int64_t)>& callback,
    const std::function<void(std::size_t, double)>& prefill_callback,
    const MfqTokenConstraintPtr& token_constraint,
    std::optional<std::size_t> stable_prefix_tokens, int prefill_chunk_size,
    const MlxPrefixCacheHooks& prefix_cache) {
    const auto& prompt = prepared.token_ids;
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
    if (prepared.embeddings && prepared.embeddings->shape() !=
            Shape{1, static_cast<int>(prompt.size()), static_cast<int>(impl_->config.hidden_size)})
        throw std::invalid_argument("prepared prompt embeddings disagree with token geometry");
    if (prepared.positions && (prepared.positions->shape() != Shape{3, static_cast<int>(prompt.size())} ||
            prepared.positions->dtype() != mlx::core::int32))
        throw std::invalid_argument("prepared prompt positions require int32 [3,T]");
    if (static_cast<std::int64_t>(prompt.size()) + prepared.decode_position_delta < 0 ||
        static_cast<std::int64_t>(prompt.size()) + prepared.decode_position_delta > impl_->maximum ||
        (prepared.decode_position_delta != 0 && !prepared.positions))
        throw std::invalid_argument("prepared prompt decode position is invalid");
    std::vector<std::int32_t> values;
    values.reserve(prompt.size());
    for (const auto token : prompt) {
        if (token < 0 || token >= vocab) {
            throw std::invalid_argument("Qwen4 prompt token is out of range");
        }
        values.push_back(static_cast<std::int32_t>(token));
    }
    if (max_tokens == 0) {
        impl_->last_mtp_stats = {
            impl_->mtp.has_value(), false, 0, 0, 0};
        reset_cache(1);
        return 0;
    }
    const int limit = std::min<int>(
        max_tokens,
        impl_->maximum - static_cast<int>(prompt.size()) + 1);
    const bool mtp_active =
        impl_->mtp.has_value() && sampling.enable_mtp &&
        limit > 1;
    impl_->last_mtp_stats = {
        impl_->mtp.has_value(), mtp_active, 0, 0, 0};
    std::size_t reused = 0;
    if (!prepared.transformed() && stable_prefix_tokens && (!mtp_active || impl_->stable_mtp_ready) &&
        impl_->cache_batch == 1 && !impl_->stable_tokens.empty() &&
        impl_->cache_position == static_cast<int>(impl_->stable_tokens.size()) &&
        impl_->stable_tokens.size() <= *stable_prefix_tokens && impl_->stable_tokens.size() < prompt.size() &&
        std::equal(impl_->stable_tokens.begin(), impl_->stable_tokens.end(), prompt.begin())) {
        reused = impl_->stable_tokens.size();
    } else {
        reset_cache(1);
    }
    impl_->prompt_positions = {prepared.positions, prepared.decode_position_delta};
    if (impl_->mtp) impl_->mtp->prompt_positions = impl_->prompt_positions;
    struct PositionReset {
        Impl& runtime;
        ~PositionReset() {
            runtime.prompt_positions = {};
            if (runtime.mtp) runtime.mtp->prompt_positions = {};
            if (transformed) runtime.stable_tokens.clear();
        }
        bool transformed;
    } position_reset{*impl_, prepared.transformed()};
    if (prefill_chunk_size <= 0) throw std::invalid_argument("Qwen4 prefill chunk size must be positive");
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
    std::optional<array> prefill_hidden;
    const bool profile_prefill = detail::component_profile_requested();
    detail::ComponentProfile component_profile;
    const auto profile_started = std::chrono::steady_clock::now();
    auto logits = [&] {
        mfq::engine::PrefillActivity prefill_activity;
        detail::ScopedComponentProfile component_scope(
            profile_prefill ? &component_profile : nullptr);
        detail::ScopedMlxEvaluationTiming timing(
            prefill_callback ? &prefill_ms : nullptr);
        std::optional<array> last_logits;
        for (std::size_t offset = reused; offset < prompt.size();) {
            const auto end = prefix_cache.prefill_end(offset, std::min(prompt.size(),
                offset + static_cast<std::size_t>(prefill_chunk_size)));
            auto ids = mlx::core::slice(prompt_ids, Shape{0, static_cast<int>(offset)},
                Shape{1, static_cast<int>(end)});
            std::optional<array> embeddings;
            if (prepared.embeddings)
                embeddings = mlx::core::slice(*prepared.embeddings,
                    {0, static_cast<int>(offset), 0},
                    {1, static_cast<int>(end), static_cast<int>(impl_->config.hidden_size)});
            auto prefill = impl_->forward_with_hidden(ids, true, true, 0, embeddings);
            last_logits = std::move(prefill.first);
            if (prefix_cache) {
                if (mtp_active) {
                    mlx_prime_mtp_chunk(prefill.second, ids, impl_->last_cache_hidden,
                        [&](const array& rows, const array& shifted, int) {
                            return impl_->mtp->append_history(rows, shifted, impl_->embedding);
                        }, static_cast<int>(offset));
                    impl_->stable_mtp_ready = true;
                } else {
                    impl_->last_cache_hidden = mlx::core::slice(prefill.second,
                        Shape{0, prefill.second.shape(1) - 1, 0},
                        Shape{1, prefill.second.shape(1), prefill.second.shape(2)});
                }
                prefix_cache.checkpoint(end);
            }
            if (mtp_active) {
                if (prefix_cache) prefill_hidden = impl_->last_cache_hidden;
                else if (prefill_hidden) prefill_hidden = mlx::core::concatenate({*prefill_hidden, prefill.second}, 1);
                else prefill_hidden = std::move(prefill.second);
            }
            if (end < prompt.size()) last_logits->eval();
            offset = end;
        }
        auto value = mlx_last_token_logits(
            *last_logits, vocab);
        if (profile_prefill) {
            detail::profile_eval("qwen4.output", value);
        } else if (prefill_callback) {
            detail::eval_with_timing(value);
        }
        return value;
    }();
    // Include route synchronization, host paging and graph construction, not
    // only evaluations wrapped by eval_with_timing().
    const double wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - profile_started).count();
    prefill_ms = wall_ms;
    if (profile_prefill) {
        const double evaluated_ms = component_profile.evaluated_ms();
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
    if (prefill_callback) prefill_callback(prompt.size() - reused, prefill_ms);

    MlxSampler sampler(sampling);
    if (mtp_active) {
        try {
            if (!prefill_hidden) {
                throw std::runtime_error(
                    "Qwen4 MTP prefill did not retain hidden streams");
            }
            if (prompt_ids.shape(1) > 1 && !prefix_cache) {
                mlx_prime_mtp_history(
                    *prefill_hidden,
                    prompt_ids,
                    [&](const array& hidden_rows,
                        const array& shifted_ids,
                        int) {
                        return impl_->mtp->append_history(
                            hidden_rows,
                            shifted_ids,
                            impl_->embedding);
                    });
            }
            // The common engine owns the first sampled token as well as every
            // later speculative cycle.  The adapter supplies only the final
            // prompt hidden row; pairing it with the engine's pending token
            // completes the teacher-forced predictor seam.
            const int hidden_count = prefill_hidden->shape(1);
            auto initial_hidden = mlx::core::slice(
                *prefill_hidden,
                Shape{0, hidden_count - 1, 0},
                Shape{
                    1,
                    hidden_count,
                    static_cast<int>(
                        impl_->config.hc_count * impl_->config.hidden_size),
                });
            int predictor_history_position =
                impl_->mtp->cache_position();
            int folded_history = 0;
            std::optional<MlxMtpDraftContext> terminal_context;
            std::optional<array> terminal_hidden;
            std::vector<std::int32_t> terminal_ids;
            const int hidden_width = static_cast<int>(impl_->config.hc_count * impl_->config.hidden_size);
            MlxMtpHistoryBuffer pending_history;
            const auto fold_pending_history = [&] {
                if (pending_history.empty()) return;
                auto batch = pending_history.drain();
                (void)impl_->mtp->append_history(batch.hidden, batch.token_ids, impl_->embedding);
                predictor_history_position += batch.token_ids.shape(1);
            };
            const auto commit_checkpoint = [&](const MlxMtpDraftContext& context, bool terminal) {
                const int accepted = context.accepted_drafts;
                impl_->last_cache_hidden = mlx::core::slice(*context.verified_hidden,
                    Shape{0, accepted, 0}, Shape{1, accepted + 1, hidden_width});
                if (!prefix_cache.wants(impl_->cache_position, terminal)) return;
                impl_->mtp->trim_cache_to(predictor_history_position);
                fold_pending_history();
                if (accepted > 0) {
                    auto rows = mlx::core::slice(*context.verified_hidden, Shape{0, 0, 0},
                        Shape{1, accepted, hidden_width});
                    array shifted(context.next_token_ids.begin(), Shape{1, accepted}, mlx::core::int32);
                    auto anchor = impl_->mtp->append_history(rows, shifted, impl_->embedding);
                    anchor.eval();
                    predictor_history_position += accepted;
                    folded_history = accepted;
                }
                prefix_cache.checkpoint(impl_->cache_position, terminal);
            };

            MlxMtpEngineCallbacks mtp_callbacks;
            mtp_callbacks.predictor = impl_->mtp->mtp_descriptor();
            mtp_callbacks.policy_state = &impl_->mtp_policy_state;
            if (impl_->mtp_lora) mtp_callbacks.verified_round = [&](const array& teacher, int accepted, double ms) {
                impl_->mtp_lora->verified(teacher, accepted, ms);
            };
            mtp_callbacks.decode_target = [&](std::int32_t token) {
                const array ids({token}, Shape{1, 1}, mlx::core::int32);
                auto step = impl_->forward_with_hidden(ids, true, false);
                return MlxMtpTargetBatch{
                    mlx::core::reshape(step.first, Shape{1, vocab}), std::move(step.second)};
            };
            mtp_callbacks.target_cache_position = [&] {
                return impl_->cache_position;
            };
            mtp_callbacks.draft_limit = [&](int position, int depth) {
                return prefix_cache.draft_limit(position, depth);
            };
            if (prefix_cache) mtp_callbacks.committed_target = [&](const MlxMtpDraftContext& context) {
                terminal_hidden = *context.verified_hidden;
                terminal_ids.assign(context.next_token_ids.begin(), context.next_token_ids.end());
                terminal_context = context;
                terminal_context->verified_hidden = &*terminal_hidden;
                terminal_context->next_token_ids = terminal_ids;
                commit_checkpoint(*terminal_context, false);
            };
            mtp_callbacks.prepare_draft =
                [&, initial_hidden](
                    const MlxMtpDraftContext& context,
                    const MlxMtpTokenSelector& select_token) {
                    if (impl_->mtp_lora && context.requested_depth > 0) impl_->mtp_lora->begin_round();
                    array hidden_rows = initial_hidden;
                    std::vector<std::int32_t> next_ids{
                        context.pending_token};
                    if (!context.initial) {
                        if (context.next_token_ids.empty()) {
                            throw std::runtime_error(
                                "Qwen4 MTP verified history is unavailable");
                        }
                        const int committed = context.accepted_drafts + 1;
                        if (context.next_token_ids.size() !=
                            static_cast<std::size_t>(committed)) {
                            throw std::runtime_error(
                                "Qwen4 MTP shifted history is inconsistent");
                        }
                        hidden_rows = mlx_mtp_committed_hidden(
                            context,
                            1,
                            static_cast<int>(
                                impl_->config.hc_count *
                                impl_->config.hidden_size));
                        next_ids.assign(
                            context.next_token_ids.begin(),
                            context.next_token_ids.end());
                    }
                    // Draft-chain forwards append speculative QSA entries.
                    // Keep the persistent predictor cache committed-only;
                    // the verified history below rebuilds the next boundary.
                    impl_->mtp->trim_cache_to(
                        predictor_history_position);
                    if (folded_history > 0) {
                        hidden_rows = mlx::core::slice(hidden_rows, Shape{0, folded_history, 0},
                            Shape{1, hidden_rows.shape(1), hidden_width});
                        next_ids.erase(next_ids.begin(), next_ids.begin() + folded_history);
                        folded_history = 0;
                    }
                    const int committed = static_cast<int>(next_ids.size());
                    if (context.requested_depth == 0) {
                        pending_history.append(hidden_rows, next_ids);
                        if (pending_history.size() >= kMlxMtpHistoryChunkSize) {
                            if (context.time_sample) *context.time_sample = false;
                            fold_pending_history();
                        }
                        return;
                    }
                    if (!pending_history.empty() && context.time_sample) *context.time_sample = false;
                    fold_pending_history();
                    const array committed_ids(
                        next_ids.begin(),
                        Shape{1, committed},
                        mlx::core::int32);
                    if (committed > 1) {
                        (void)impl_->mtp->append_history(
                            mlx::core::slice(hidden_rows, Shape{0, 0, 0},
                                Shape{1, committed - 1, hidden_width}),
                            mlx::core::slice(committed_ids, Shape{0, 0},
                                Shape{1, committed - 1}),
                            impl_->embedding);
                    }
                    auto predictor = impl_->mtp->forward(
                        mlx::core::slice(hidden_rows, Shape{0, committed - 1, 0},
                            Shape{1, committed, hidden_width}),
                        mlx::core::slice(committed_ids, Shape{0, committed - 1},
                            Shape{1, committed}),
                        impl_->embedding,
                        true, impl_->mtp_lora.get());
                    predictor_history_position += committed;
                    auto head_hidden = std::move(predictor.sample_hidden);
                    auto chain_streams = std::move(predictor.hidden_streams);
                    for (int position = 0;
                         position < context.requested_depth;
                         ++position) {
                        auto draft_logits = impl_->output
                            ? (*impl_->output)(head_hidden)
                            : impl_->embedding.project(head_hidden);
                        auto token = select_token(mlx_last_token_logits(
                            draft_logits, vocab));
                        if (position + 1 == context.requested_depth) {
                            break;
                        }
                        predictor = impl_->mtp->forward(
                            chain_streams,
                            mlx::core::reshape(token, Shape{1, 1}),
                            impl_->embedding,
                            true, impl_->mtp_lora.get());
                        head_hidden = std::move(predictor.sample_hidden);
                        chain_streams = std::move(predictor.hidden_streams);
                    }
                };
            mtp_callbacks.verify_target =
                [&](std::int32_t pending_token,
                    const array& draft_tokens,
                    int draft_count) {
                    auto verify_ids = mlx_mtp_verification_ids(
                        pending_token, draft_tokens, draft_count);
                    auto verified = impl_->forward_with_hidden(
                        verify_ids,
                        true,
                        false,
                        draft_count > 0 ? 1 : 0);
                    return MlxMtpTargetBatch{
                        mlx::core::reshape(
                            verified.first,
                            Shape{draft_count + 1, vocab}),
                        std::move(verified.second),
                    };
                };
            mtp_callbacks.resolve_target =
                [&](int accepted_drafts, int draft_count) {
                    if (draft_count == 0) return;
                    if (accepted_drafts == draft_count) {
                        impl_->commit_speculative();
                    } else {
                        impl_->rollback_speculative(
                            accepted_drafts, draft_count);
                    }
                };
            const std::array<std::int64_t, 1> eos{
                impl_->config.eos_token_id};
            const auto result = run_mlx_mtp_generation(
                MlxMtpEngineRequest{
                    vocab,
                    limit,
                    impl_->maximum,
                    std::move(logits),
                    sampling,
                    std::move(counts),
                    impl_->config.eos_token_id > 0
                        ? std::span<const std::int64_t>(eos)
                        : std::span<const std::int64_t>{},
                    callback,
                    0u,
                    token_constraint,
                },
                mtp_callbacks,
                impl_->last_mtp_stats);
            if (terminal_context && folded_history == 0) commit_checkpoint(*terminal_context, true);
            else prefix_cache.checkpoint(impl_->cache_position, true);
            return result;
        } catch (...) {
            try {
                impl_->reset(1);
            } catch (...) {
            }
            throw;
        }
    }

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
        if (token_constraint &&
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
        if (token_constraint) {
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
        const int decode_step = generated - 1;
        const int profile_skip = detail::component_profile_skip_steps();
        const bool profile_step = detail::component_profile_requested()
            && decode_step >= profile_skip
            && decode_step - profile_skip < detail::component_profile_steps();
        detail::ComponentProfile decode_profile;
        detail::ScopedComponentProfile decode_profile_scope(
            profile_step ? &decode_profile : nullptr);
        const auto decode_started = std::chrono::steady_clock::now();
        if (prefix_cache) {
            auto result = impl_->forward_with_hidden(token_ids, true, false);
            impl_->last_cache_hidden = result.second;
            logits = mlx_last_token_logits(result.first, vocab);
            prefix_cache.checkpoint(impl_->cache_position);
        } else {
            logits = mlx_last_token_logits(forward(token_ids, true), vocab);
        }
        if (profile_step) {
            detail::profile_eval("qwen4.output", logits);
            const double wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decode_started).count();
            const double evaluated_ms = decode_profile.evaluated_ms();
            std::cout << "component_profile model=qwen4 phase=decode step="
                      << decode_step << " wall_ms=" << wall_ms
                      << " evaluated_ms=" << evaluated_ms
                      << " unscoped_ms=" << std::max(0.0, wall_ms - evaluated_ms)
                      << std::endl;
            for (const auto& [name, timing] : decode_profile.timings()) {
                std::cout << "component_cost model=qwen4 phase=decode step="
                          << decode_step << " name=" << name
                          << " ms=" << timing.elapsed_ms
                          << " calls=" << timing.evaluations << std::endl;
            }
        }
    }
    prefix_cache.checkpoint(impl_->cache_position, true);
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

bool MlxQwen4CausalLm::supports_mtp() const noexcept {
    return impl_->mtp.has_value();
}

bool MlxQwen4CausalLm::supports_batch_mtp() const noexcept {
    return impl_->mtp.has_value() && impl_->mtp->layer_count() == 1 && !impl_->mtp_lora;
}

const MlxMtpGenerationStats&
MlxQwen4CausalLm::last_mtp_stats() const noexcept {
    return impl_->last_mtp_stats;
}

std::size_t MlxQwen4CausalLm::expert_cache_limit_bytes() const noexcept {
    if (impl_->tq_expert_cache) return impl_->tq_expert_cache->limit_bytes();
    if (impl_->ssd_expert_cache) {
        return impl_->ssd_expert_cache->cache_limit_bytes();
    }
    return impl_->mfe_offload_cache
        ? impl_->mfe_offload_cache->cache_limit_bytes()
        : 0;
}

std::optional<MlxSsdExpertCacheStats>
MlxQwen4CausalLm::ssd_expert_cache_stats() const {
    if (impl_->tq_expert_cache) {
        const auto cache = impl_->tq_expert_cache->stats();
        MlxSsdExpertCacheStats result;
        result.requests = cache.requests; result.hits = cache.hits;
        result.misses = cache.misses; result.loads = cache.misses;
        result.evictions = cache.evictions; result.bytes_read = cache.bytes_read;
        result.read_calls = cache.read_calls; result.io_seconds = cache.io_seconds;
        result.route_sync_seconds = cache.route_sync_seconds;
        result.prepare_seconds = cache.prepare_seconds;
        result.routed_graph_seconds = cache.graph_seconds;
        result.overlapped_read_batches = cache.overlapped_read_batches;
        result.resident_experts = cache.cached_experts;
        result.resident_bytes = cache.allocated_bytes;
        result.cached_payload_bytes = cache.cached_bytes;
        result.cache_metadata_bytes = cache.metadata_bytes;
        return result;
    }
    if (impl_->ssd_expert_cache) {
        return impl_->ssd_expert_cache->stats();
    }
    if (!impl_->mfe_offload_cache) {
        return std::nullopt;
    }
    MlxSsdExpertCacheStats stats;
    const auto reuse = impl_->mfe_offload_cache->reuse_stats();
    stats.loads = reuse.expert_loads;
    stats.misses = reuse.expert_loads;
    stats.hits = reuse.expert_hits;
    stats.requests = stats.hits + stats.misses;
    stats.coalesced = reuse.load_waits;
    stats.packed_reuses = reuse.packed_hits;
    stats.bytes_read = reuse.bytes_read;
    stats.read_calls = reuse.read_calls;
    stats.evictions = reuse.evictions;
    stats.io_seconds = reuse.io_seconds;
    stats.wait_seconds = reuse.wait_seconds;
    stats.prepare_seconds = reuse.prepare_seconds;
    stats.view_seconds = reuse.view_seconds;
    stats.device_route_hits = reuse.cached_route_hits;
    stats.device_route_misses = reuse.cached_route_misses;
    stats.device_route_layers = reuse.cached_route_hits + reuse.cached_route_misses;
    stats.resident_experts =
        impl_->mfe_offload_cache->cached_expert_count();
    stats.resident_bytes =
        impl_->mfe_offload_cache->resident_packed_bytes();
    return stats;
}

void MlxQwen4CausalLm::prewarm_ssd_expert_arena() {
    if (impl_->ssd_expert_cache) {
        impl_->ssd_expert_cache->prewarm_metal();
    }
}

void MlxQwen4CausalLm::clear_expert_cache() {
    if (impl_->tq_expert_cache) impl_->tq_expert_cache->clear();
    if (impl_->ssd_expert_cache) {
        impl_->ssd_expert_cache->clear();
    }
    if (impl_->mfe_offload_cache) {
        impl_->mfe_offload_cache->clear();
    }
}

const std::string& MlxQwen4CausalLm::mtp_cache_fingerprint() const noexcept { return impl_->mtp_base_key; }

void MlxQwen4CausalLm::prepare_mtp_ttt(const std::filesystem::path& directory) {
    if (!mtp_ttt_requested() || !impl_->mtp_full_precision || impl_->mtp_lora) return;
    const auto hot = mfq::engine::environment_uint64("MFQ_MTP_TTT_RAM_BYTES", 12ULL << 30);
    const auto disk = mfq::engine::environment_uint64("MFQ_MTP_TTT_DISK_BYTES", 16ULL << 30);
    const auto key = mfq::cache::block_hash_hex(mfq::cache::sha256(impl_->mtp_base_key));
    const auto& config = impl_->config;
    const int h = config.hidden_size, q = config.num_attention_heads * config.head_dim;
    std::vector<MlxMtpLoraProjection> layout{{"attention.q", h, q}, {"attention.gate", h, q}, {"attention.o", q, h}};
    for (const auto& role : {std::string("gate"), std::string("up"), std::string("down")}) {
        layout.push_back({"shared." + role, role == "down" ? int(config.shared_expert_intermediate_size) : h,
            role == "down" ? h : int(config.shared_expert_intermediate_size)});
        for (int expert = 0; expert < config.num_experts; ++expert)
            layout.push_back({"expert." + std::to_string(expert) + "." + role,
                role == "down" ? int(config.moe_intermediate_size) : h,
                role == "down" ? h : int(config.moe_intermediate_size)});
    }
    if (!impl_->output) throw std::invalid_argument("MTP LoRA currently requires an untied frozen LM Head");
    auto head = *impl_->output;
    { MlxFp16WeightScope fp16(true); head.materialize_fp16(); }
    impl_->mtp_lora = std::make_unique<MlxMtpLora>(std::move(layout), *head.dense_weight_ref(),
        impl_->mtp_base_key, directory / "mtp-lora" / key, hot, disk, 16, config.num_experts_per_tok);
    impl_->mtp->prepare_training(*impl_->mtp_lora);
}

void MlxQwen4CausalLm::begin_mtp_session(const std::string& session) {
    if (impl_->mtp_lora) impl_->mtp_lora->begin(session);
}
void MlxQwen4CausalLm::end_mtp_session() { if (impl_->mtp_lora) impl_->mtp_lora->end(); }
bool MlxQwen4CausalLm::close_mtp_session(const std::string& session) {
    return impl_->mtp_lora && impl_->mtp_lora->close(session);
}
bool MlxQwen4CausalLm::fork_mtp_session(const std::string& source, const std::string& target) {
    return impl_->mtp_lora && impl_->mtp_lora->fork(source, target);
}
void MlxQwen4CausalLm::clear_mtp_sessions() { if (impl_->mtp_lora) impl_->mtp_lora->clear(); }
std::size_t MlxQwen4CausalLm::trim_mtp_sessions(std::size_t bytes) {
    return impl_->mtp_lora ? impl_->mtp_lora->trim(bytes) : 0;
}
std::size_t MlxQwen4CausalLm::mtp_session_bytes() const { return impl_->mtp_lora ? impl_->mtp_lora->bytes() : 0; }
std::vector<std::pair<std::string, double>> MlxQwen4CausalLm::mtp_session_metrics() const {
    return impl_->mtp_lora ? impl_->mtp_lora->metrics() : std::vector<std::pair<std::string, double>>{{"mtp_ttt_enabled", 0}};
}

MlxQwen4TextSessionState MlxQwen4CausalLm::capture_text_session_state(
    const std::vector<std::int64_t>& tokens, bool detached) const {
    if (impl_->cache_batch != 1 || impl_->cache_position <= 0 || impl_->speculative_pending ||
        tokens.size() != static_cast<std::size_t>(impl_->cache_position))
        throw std::runtime_error("Qwen4 checkpoint token count mismatch");
    MlxQwen4TextSessionState state;
    state.tokens = tokens;
    state.cache_position = impl_->cache_position;
    state.cache_batch = impl_->cache_batch;
    for (const auto& layer : impl_->layers) state.layers.push_back(layer.snapshot(detached));
    if (impl_->mtp && impl_->last_cache_hidden &&
        impl_->mtp->cache_position() >= impl_->cache_position - 1) {
        state.mtp_layers = impl_->mtp->snapshot(impl_->cache_position - 1);
        state.last_hidden = impl_->last_cache_hidden;
    }
    const auto bytes = [&](const MlxQwen4LayerCacheSnapshot& layer) {
        if (layer.kv) state.bytes += layer.kv->nbytes();
        if (layer.offloaded_kv) state.bytes += layer.offloaded_kv->rows.tail.size() +
            layer.offloaded_kv->rows.blocks.size() * sizeof(QsaKvStore::BlockPtr);
        if (layer.offloaded_pooled_keys) state.bytes += layer.offloaded_pooled_keys->rows.tail.size() +
            layer.offloaded_pooled_keys->rows.blocks.size() * sizeof(QsaKvStore::BlockPtr);
        for (const auto* value : {&layer.index_keys, &layer.pooled_keys, &layer.convolution,
            &layer.recurrent, &layer.ple_convolution}) if (*value) state.bytes += (*value)->nbytes();
        state.bytes += layer.ple_context.size() * sizeof(std::int64_t);
    };
    for (const auto& layer : state.layers) bytes(layer);
    for (const auto& layer : state.mtp_layers) bytes(layer);
    if (state.last_hidden) state.bytes += state.last_hidden->nbytes();
    return state;
}

void MlxQwen4CausalLm::restore_text_session_state(
    const MlxQwen4TextSessionState& state) {
    if (state.cache_batch != 1 || state.cache_position <= 0 || state.cache_position > impl_->maximum ||
        state.tokens.size() != static_cast<std::size_t>(state.cache_position) ||
        state.layers.size() != impl_->layers.size())
        throw std::runtime_error("Qwen4 checkpoint topology mismatch");
    try {
        for (std::size_t i = 0; i < state.layers.size(); ++i) {
            if (state.layers[i].position != state.cache_position || state.layers[i].batch != state.cache_batch)
                throw std::runtime_error("Qwen4 checkpoint layer position mismatch");
            impl_->layers[i].restore(state.layers[i]);
        }
        if (impl_->mtp) impl_->mtp->restore(state.mtp_layers);
        impl_->cache_position = state.cache_position;
        impl_->cache_batch = state.cache_batch;
        impl_->stable_tokens = state.tokens;
        impl_->last_cache_hidden = state.last_hidden;
        impl_->stable_mtp_ready = state.last_hidden.has_value() &&
            (state.cache_position == 1 || !state.mtp_layers.empty());
        impl_->speculative_pending = false;
    } catch (...) {
        clear_cache();
        throw;
    }
}

} // namespace mfq::metal
