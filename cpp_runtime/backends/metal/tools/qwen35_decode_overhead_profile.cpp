#include "mfq_container.h"
#include "mlx_linear_attention.h"
#include "mlx_qwen35_causal_lm.h"
#include "mlx_sampling.h"
#include "mlx_transformer.h"
#include "qwen35_model.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <mlx/graph_utils.h>
#include <mlx/memory.h>
#include <mlx/mlx.h>
#include <mlx/primitives.h>

namespace {

using Clock = std::chrono::steady_clock;
using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MfqContainer;
using mfq::metal::MlxKvCache;
using mfq::metal::MlxQwen35CausalLm;
using mfq::metal::MlxRmsNorm;
using mfq::metal::MlxSampler;
using mfq::metal::MlxSamplingParams;

struct GraphStats {
    std::size_t arrays = 0;
    std::size_t leaves = 0;
    std::size_t primitives = 0;
    std::size_t logical_output_bytes = 0;
    std::map<std::string, std::size_t> primitive_counts;
};

double milliseconds_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
}

int positive_argument(
    int argc,
    char** argv,
    int index,
    int fallback,
    const char* name) {
    if (argc <= index) {
        return fallback;
    }
    const int value = std::stoi(argv[index]);
    if (value <= 0) {
        throw std::invalid_argument(
            std::string(name) + " must be positive");
    }
    return value;
}

const char* environment_or_default(
    const char* name,
    const char* fallback) {
    const auto* value = std::getenv(name);
    return value == nullptr || *value == '\0'
        ? fallback
        : value;
}

GraphStats inspect_graph(const array& root) {
    GraphStats result;
    std::vector<array> pending{root};
    std::unordered_set<std::uintptr_t> visited_arrays;
    std::unordered_set<std::uintptr_t> visited_primitives;
    while (!pending.empty()) {
        auto current = std::move(pending.back());
        pending.pop_back();
        if (!visited_arrays.insert(current.id()).second) {
            continue;
        }
        ++result.arrays;
        result.logical_output_bytes += current.nbytes();
        if (!current.has_primitive()) {
            ++result.leaves;
            continue;
        }
        if (visited_primitives.insert(current.primitive_id()).second) {
            ++result.primitives;
            ++result.primitive_counts[current.primitive().name()];
        }
        for (const auto& input : current.inputs()) {
            pending.push_back(input);
        }
    }
    return result;
}

void print_graph_stats(const GraphStats& stats) {
    std::vector<std::pair<std::string, std::size_t>> ordered(
        stats.primitive_counts.begin(),
        stats.primitive_counts.end());
    std::sort(
        ordered.begin(),
        ordered.end(),
        [](const auto& left, const auto& right) {
            if (left.second != right.second) {
                return left.second > right.second;
            }
            return left.first < right.first;
        });
    std::cout
        << "decode_graph arrays=" << stats.arrays
        << " leaves=" << stats.leaves
        << " primitives=" << stats.primitives
        << " logical_output_mib="
        << static_cast<double>(stats.logical_output_bytes) /
               (1024.0 * 1024.0)
        << "\n";
    for (const auto& [name, count] : ordered) {
        std::cout
            << "decode_graph_primitive name=" << name
            << " count=" << count << "\n";
    }
}

array last_token_logits(const array& logits, int vocab) {
    if (logits.ndim() != 3 ||
        logits.shape(0) != 1 ||
        logits.shape(1) <= 0 ||
        logits.shape(2) != vocab) {
        throw std::runtime_error(
            "profile logits must have [1,tokens,vocab] shape");
    }
    return mlx::core::reshape(
        mlx::core::slice(
            logits,
            Shape{0, logits.shape(1) - 1, 0},
            Shape{1, logits.shape(1), vocab}),
        Shape{1, vocab});
}

std::int32_t evaluated_token(array sampled) {
    sampled.eval();
    return sampled.data<std::int32_t>()[0];
}

std::int32_t prefill(
    MlxQwen35CausalLm& model,
    MlxSampler& sampler,
    int prompt_tokens,
    int vocab) {
    std::vector<std::int32_t> values(
        static_cast<std::size_t>(prompt_tokens));
    for (int index = 0; index < prompt_tokens; ++index) {
        values[static_cast<std::size_t>(index)] =
            1 + index % std::max(1, vocab - 1);
    }
    const array ids(
        values.begin(),
        Shape{1, prompt_tokens},
        mlx::core::int32);
    model.reset_cache(1);
    return evaluated_token(
        sampler.sample(
            last_token_logits(
                model.forward(ids, true),
                vocab)));
}

struct DecodeTimings {
    double build_ms = 0.0;
    double eval_ms = 0.0;
    double read_ms = 0.0;
    double extra_sync_ms = 0.0;
};

void run_decode_steps(
    MlxQwen35CausalLm& model,
    MlxSampler& sampler,
    int vocab,
    int steps,
    std::int32_t& token,
    DecodeTimings* timings,
    bool inspect_first) {
    for (int step = 0; step < steps; ++step) {
        const auto build_start = Clock::now();
        const array ids(
            {token},
            Shape{1, 1},
            mlx::core::int32);
        auto logits = model.forward(ids, true);
        auto sampled = sampler.sample(
            last_token_logits(logits, vocab));
        if (timings != nullptr) {
            timings->build_ms +=
                milliseconds_since(build_start);
        }

        if (inspect_first && step == 0) {
            print_graph_stats(inspect_graph(sampled));
        }

        const auto eval_start = Clock::now();
        sampled.eval();
        if (timings != nullptr) {
            timings->eval_ms +=
                milliseconds_since(eval_start);
        }

        const auto read_start = Clock::now();
        token = sampled.data<std::int32_t>()[0];
        if (timings != nullptr) {
            timings->read_ms +=
                milliseconds_since(read_start);
        }

        const auto sync_start = Clock::now();
        mlx::core::synchronize();
        if (timings != nullptr) {
            timings->extra_sync_ms +=
                milliseconds_since(sync_start);
        }
    }
}

double benchmark_sampler(
    const array& evaluated_logits,
    MlxSamplingParams params,
    int warmup,
    int repetitions) {
    MlxSampler sampler(std::move(params));
    for (int index = 0; index < warmup; ++index) {
        sampler.sample(evaluated_logits).eval();
    }
    mlx::core::synchronize();
    const auto start = Clock::now();
    for (int index = 0; index < repetitions; ++index) {
        sampler.sample(evaluated_logits).eval();
    }
    mlx::core::synchronize();
    return milliseconds_since(start) /
        static_cast<double>(repetitions);
}

template <typename Build>
double benchmark_graph(
    Build&& build,
    int warmup,
    int repetitions) {
    for (int index = 0; index < warmup; ++index) {
        auto outputs = build();
        mlx::core::eval(outputs);
    }
    mlx::core::synchronize();
    const auto start = Clock::now();
    for (int index = 0; index < repetitions; ++index) {
        auto outputs = build();
        mlx::core::eval(outputs);
    }
    mlx::core::synchronize();
    return milliseconds_since(start) /
        static_cast<double>(repetitions);
}

array evaluated_zeros(
    Shape shape,
    mlx::core::Dtype dtype) {
    auto result = mlx::core::zeros(
        std::move(shape),
        dtype);
    result.eval();
    return result;
}

double benchmark_linear_attention_state(
    const mfq::metal::Qwen35Config& config,
    int repetitions) {
    const int layer_count = static_cast<int>(
        std::count(
            config.layer_types.begin(),
            config.layer_types.end(),
            "linear_attention"));
    const int key_heads =
        static_cast<int>(config.linear_key_heads());
    const int value_heads =
        static_cast<int>(config.linear_value_heads());
    const int key_dimension =
        static_cast<int>(config.linear_key_head_dim);
    const int value_dimension =
        static_cast<int>(config.linear_value_head_dim);
    const int key_size =
        static_cast<int>(config.linear_key_size());
    const int value_size =
        static_cast<int>(config.linear_value_size());
    const int channels =
        static_cast<int>(config.linear_qkv_size());
    const int kernel =
        static_cast<int>(config.linear_conv_kernel_dim);

    auto qk = evaluated_zeros(
        Shape{1, 1, 2 * key_size},
        mlx::core::float16);
    auto value = evaluated_zeros(
        Shape{1, 1, value_size},
        mlx::core::float16);
    auto convolution_weight = evaluated_zeros(
        Shape{channels, kernel},
        mlx::core::float32);
    auto gate_source = evaluated_zeros(
        Shape{1, 1, value_heads},
        mlx::core::float32);
    auto beta_source = evaluated_zeros(
        Shape{1, 1, value_heads},
        mlx::core::float32);
    auto z = evaluated_zeros(
        Shape{1, 1, value_size},
        mlx::core::float16);
    auto norm_weight = evaluated_zeros(
        Shape{value_dimension},
        mlx::core::float32);
    MlxRmsNorm value_norm(
        std::move(norm_weight),
        static_cast<float>(config.rms_norm_eps));

    std::vector<array> convolution_states;
    std::vector<array> recurrent_states;
    convolution_states.reserve(
        static_cast<std::size_t>(layer_count));
    recurrent_states.reserve(
        static_cast<std::size_t>(layer_count));
    for (int index = 0; index < layer_count; ++index) {
        convolution_states.push_back(
            evaluated_zeros(
                Shape{1, kernel - 1, channels},
                mlx::core::float32));
        recurrent_states.push_back(
            evaluated_zeros(
                Shape{
                    1,
                    value_heads,
                    value_dimension,
                    value_dimension,
                },
                mlx::core::float32));
    }

    return benchmark_graph(
        [&]() {
            std::vector<array> outputs;
            outputs.reserve(
                static_cast<std::size_t>(layer_count) * 3);
            for (int index = 0; index < layer_count; ++index) {
                const auto gate_input =
                    gate_source + array(0.5f);
                const auto gate =
                    (
                        mlx::core::maximum(
                            gate_input,
                            array(0.0f)) +
                        mlx::core::log1p(
                            mlx::core::exp(
                                -mlx::core::abs(
                                    gate_input)))
                    ) * array(-0.25f);
                const auto beta =
                    mlx::core::sigmoid(beta_source);
                auto convolved =
                    mfq::metal::linear_conv_qkv(
                        convolution_states[
                            static_cast<std::size_t>(index)],
                        qk,
                        value,
                        convolution_weight,
                        key_heads,
                        value_heads,
                        key_dimension,
                        value_dimension,
                        std::nullopt,
                        static_cast<float>(
                            config.rms_norm_eps));
                auto recurrent =
                    mfq::metal::gated_delta_net(
                        convolved.query,
                        convolved.key,
                        convolved.value,
                        mlx::core::transpose(
                            gate,
                            {0, 2, 1}),
                        mlx::core::transpose(
                            beta,
                            {0, 2, 1}),
                        recurrent_states[
                            static_cast<std::size_t>(index)],
                        false,
                        true);
                convolution_states[
                    static_cast<std::size_t>(index)] =
                    convolved.state;
                recurrent_states[
                    static_cast<std::size_t>(index)] =
                    recurrent.state;
                auto normalized =
                    value_norm(recurrent.output);
                normalized = mlx::core::reshape(
                    mlx::core::transpose(
                        normalized,
                        {0, 2, 1, 3}),
                    Shape{1, 1, value_size});
                const auto gated =
                    normalized *
                    (z * mlx::core::sigmoid(z));
                outputs.push_back(std::move(gated));
                outputs.push_back(
                    convolution_states[
                        static_cast<std::size_t>(index)]);
                outputs.push_back(
                    recurrent_states[
                        static_cast<std::size_t>(index)]);
            }
            return outputs;
        },
        2,
        repetitions);
}

double benchmark_full_attention_state(
    const mfq::metal::Qwen35Config& config,
    int repetitions) {
    const int layer_count = static_cast<int>(
        std::count(
            config.layer_types.begin(),
            config.layer_types.end(),
            "full_attention"));
    const int query_heads =
        static_cast<int>(config.num_attention_heads);
    const int kv_heads =
        static_cast<int>(config.num_key_value_heads);
    const int dimension =
        static_cast<int>(config.head_dim);
    const int attention_size =
        static_cast<int>(config.attention_size());
    constexpr int initial_context = 16;
    const int maximum_context =
        initial_context + repetitions + 8;

    auto query_source = evaluated_zeros(
        Shape{1, query_heads, 1, dimension},
        mlx::core::float16);
    auto key_source = evaluated_zeros(
        Shape{1, kv_heads, 1, dimension},
        mlx::core::float16);
    auto value = evaluated_zeros(
        Shape{1, kv_heads, 1, dimension},
        mlx::core::float16);
    auto query_gate = evaluated_zeros(
        Shape{1, 1, attention_size},
        mlx::core::float16);
    auto query_norm_weight = evaluated_zeros(
        Shape{dimension},
        mlx::core::float32);
    auto key_norm_weight = evaluated_zeros(
        Shape{dimension},
        mlx::core::float32);
    MlxRmsNorm query_norm(
        std::move(query_norm_weight),
        static_cast<float>(config.rms_norm_eps));
    MlxRmsNorm key_norm(
        std::move(key_norm_weight),
        static_cast<float>(config.rms_norm_eps));

    std::vector<std::unique_ptr<MlxKvCache>> caches;
    caches.reserve(static_cast<std::size_t>(layer_count));
    auto initial_key = evaluated_zeros(
        Shape{
            1,
            kv_heads,
            initial_context,
            dimension,
        },
        mlx::core::float16);
    auto initial_value = evaluated_zeros(
        initial_key.shape(),
        mlx::core::float16);
    std::vector<array> initial_outputs;
    for (int index = 0; index < layer_count; ++index) {
        caches.push_back(
            std::make_unique<MlxKvCache>(
                1,
                kv_heads,
                maximum_context,
                dimension,
                maximum_context,
                mlx::core::float16));
        auto views = caches.back()->append(
            initial_key,
            initial_value);
        initial_outputs.push_back(
            std::move(views.first));
        initial_outputs.push_back(
            std::move(views.second));
    }
    mlx::core::eval(initial_outputs);

    return benchmark_graph(
        [&]() {
            std::vector<array> outputs;
            outputs.reserve(
                static_cast<std::size_t>(layer_count));
            for (int index = 0; index < layer_count; ++index) {
                auto query = query_norm(query_source);
                auto key = key_norm(key_source);
                const int offset =
                    caches[
                        static_cast<std::size_t>(index)]
                        ->position();
                query = mfq::metal::apply_rope(
                    query,
                    static_cast<int>(config.rotary_dim),
                    static_cast<float>(config.rope_base),
                    offset);
                key = mfq::metal::apply_rope(
                    key,
                    static_cast<int>(config.rotary_dim),
                    static_cast<float>(config.rope_base),
                    offset);
                auto views =
                    caches[
                        static_cast<std::size_t>(index)]
                        ->append(key, value);
                auto attended =
                    mfq::metal::scaled_dot_product_attention(
                        query,
                        views.first,
                        views.second,
                        true);
                attended = mlx::core::reshape(
                    mlx::core::transpose(
                        attended,
                        {0, 2, 1, 3}),
                    Shape{1, 1, attention_size});
                attended =
                    attended *
                    mlx::core::sigmoid(query_gate);
                outputs.push_back(std::move(attended));
            }
            return outputs;
        },
        2,
        repetitions);
}

double benchmark_norm_residual_graph(
    const mfq::metal::Qwen35Config& config,
    int repetitions) {
    const int hidden =
        static_cast<int>(config.hidden_size);
    auto hidden_weight = evaluated_zeros(
        Shape{hidden},
        mlx::core::float32);
    MlxRmsNorm norm(
        std::move(hidden_weight),
        static_cast<float>(config.rms_norm_eps));
    auto hidden_value = evaluated_zeros(
        Shape{1, 1, hidden},
        mlx::core::float16);
    return benchmark_graph(
        [&]() {
            auto value = hidden_value;
            for (std::size_t index = 0;
                 index < config.layer_types.size();
                 ++index) {
                value =
                    value +
                    norm(value) * array(0.001f);
                value =
                    value +
                    norm(value) * array(0.001f);
            }
            value = norm(value);
            return std::vector<array>{
                std::move(value),
            };
        },
        2,
        repetitions);
}

void profile_unused_count_graph(
    int vocab,
    int tokens) {
    const array token_ids(
        {1},
        Shape{1, 1},
        mlx::core::int32);
    std::optional<array> counts(
        mlx::core::zeros(
            Shape{vocab},
            mlx::core::int32));
    const auto build_start = Clock::now();
    for (int index = 0; index < tokens; ++index) {
        *counts =
            mfq::metal::sample_token_counts_add(
                *counts,
                token_ids);
    }
    const double build_ms =
        milliseconds_since(build_start);
    const auto stats = inspect_graph(*counts);
    const auto destroy_start = Clock::now();
    counts.reset();
    const double destroy_ms =
        milliseconds_since(destroy_start);
    std::cout
        << "unused_count_graph tokens=" << tokens
        << " build_ms=" << build_ms
        << " build_us_per_token="
        << 1000.0 * build_ms /
               static_cast<double>(tokens)
        << " destroy_ms=" << destroy_ms
        << " retained_primitives="
        << stats.primitives
        << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 5) {
            throw std::invalid_argument(
                "usage: qwen35_decode_overhead_profile MODEL.mfq "
                "[warmup=4] [steps=16] [prompt_tokens=16]");
        }
        const int warmup =
            positive_argument(argc, argv, 2, 4, "warmup");
        const int steps =
            positive_argument(argc, argv, 3, 16, "steps");
        const int prompt_tokens =
            positive_argument(
                argc,
                argv,
                4,
                16,
                "prompt_tokens");

#ifdef MFQ_MLX_METALLIB_DEFAULT
        mlx::core::metal::set_metallib_path(
            MFQ_MLX_METALLIB_DEFAULT);
#endif
        mlx::core::set_default_device(
            mlx::core::Device::gpu);
        std::cout
            << "profile_environment MLX_MAX_OPS_PER_BUFFER="
            << environment_or_default(
                   "MLX_MAX_OPS_PER_BUFFER",
                   "<device-default>")
            << " MLX_MAX_MB_PER_BUFFER="
            << environment_or_default(
                   "MLX_MAX_MB_PER_BUFFER",
                   "<device-default>")
            << "\n";

        const auto load_start = Clock::now();
        const MfqContainer container(argv[1]);
        const auto config =
            mfq::metal::Qwen35Config::from_mfq(container);
        auto model = MlxQwen35CausalLm::load(container);
        std::cout
            << "load_ms=" << milliseconds_since(load_start)
            << " active_mib="
            << static_cast<double>(mlx::core::get_active_memory()) /
                   (1024.0 * 1024.0)
            << "\n";

        MlxSamplingParams greedy;
        MlxSampler sampler(greedy);
        std::int32_t token = prefill(
            model,
            sampler,
            prompt_tokens,
            static_cast<int>(config.vocab_size));
        run_decode_steps(
            model,
            sampler,
            static_cast<int>(config.vocab_size),
            warmup,
            token,
            nullptr,
            false);

        DecodeTimings timings;
        const auto benchmark_start = Clock::now();
        run_decode_steps(
            model,
            sampler,
            static_cast<int>(config.vocab_size),
            steps,
            token,
            &timings,
            true);
        const double total_ms =
            milliseconds_since(benchmark_start);
        std::cout
            << std::fixed << std::setprecision(4)
            << "decode_profile steps=" << steps
            << " total_ms_per_token="
            << total_ms / static_cast<double>(steps)
            << " build_ms_per_token="
            << timings.build_ms / static_cast<double>(steps)
            << " eval_ms_per_token="
            << timings.eval_ms / static_cast<double>(steps)
            << " read_ms_per_token="
            << timings.read_ms / static_cast<double>(steps)
            << " extra_sync_ms_per_token="
            << timings.extra_sync_ms /
                   static_cast<double>(steps)
            << " tok_per_s="
            << 1000.0 * static_cast<double>(steps) / total_ms
            << "\n";

        array sampling_logits(
            mlx::core::zeros(
                Shape{
                    1,
                    static_cast<int>(config.vocab_size),
                },
                mlx::core::float16));
        sampling_logits.eval();
        constexpr int sample_repetitions = 64;
        std::cout
            << "sampling_profile greedy_ms="
            << benchmark_sampler(
                   sampling_logits,
                   MlxSamplingParams{},
                   4,
                   sample_repetitions);
        MlxSamplingParams stochastic;
        stochastic.temperature = 0.7;
        stochastic.top_k = 20;
        stochastic.top_p = 0.8;
        std::cout
            << " topk20_ms="
            << benchmark_sampler(
                   sampling_logits,
                   stochastic,
                   4,
                   sample_repetitions)
            << "\n";
        const int synthetic_repetitions =
            std::min(steps, 16);
        std::cout
            << "nonweight_profile linear_state_ms="
            << benchmark_linear_attention_state(
                   config,
                   synthetic_repetitions)
            << " full_attention_state_ms="
            << benchmark_full_attention_state(
                   config,
                   synthetic_repetitions)
            << " norm_residual_ms="
            << benchmark_norm_residual_graph(
                   config,
                   synthetic_repetitions)
            << "\n";
        profile_unused_count_graph(
            static_cast<int>(config.vocab_size),
            16);
        profile_unused_count_graph(
            static_cast<int>(config.vocab_size),
            4096);
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "Qwen3.5 decode overhead profile failed: "
            << error.what() << "\n";
        return 1;
    }
}
