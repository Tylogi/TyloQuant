#include "mfq_container.h"
#include "mlx_qwen4_causal_lm.h"
#include "mlx_deepseek_v41_causal_lm.h"
#include "mlx_memory_residency.h"
#include "mlx_weight_residency.h"
#include "mlx_eval_timing.h"
#include "mlx_tq_rows.h"

#include <dispatch/dispatch.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace mlx::core;
using namespace mfq::metal;
using Clock = std::chrono::steady_clock;

static void benchmark_rows(const MfqContainer& source, int tokens) {
    std::vector<std::string> names;
    for (const auto& [name, record] : source.records())
        if (record.dtype == "TQ" && name.find(".ngram.shard.") != std::string::npos) names.push_back(name);
    std::sort(names.begin(), names.end());
    if (names.empty()) throw std::invalid_argument("model has no TQ PLE row tables");
    const auto shapes = nlohmann::json::parse(source.header().extra_json.at("tq_shapes"));
    const int width = shapes.at(names.front()).back().get<int>();
    std::vector<MfqMappedBytes> mappings;
    std::vector<MlxMappedTqRows> tables;
    for (const auto& name : names) {
        mappings.push_back(source.map_record(name, true));
        tables.emplace_back(mappings.back().view(), width);
    }
    std::mt19937 rng(20261022);
    std::vector<std::pair<std::size_t, std::int64_t>> ids;
    for (int row = 0; row < tokens * 16; ++row) {
        const auto table = rng() % tables.size();
        ids.emplace_back(table, rng() % tables[table].rows());
    }
    auto read = [&](int grain) {
        struct Part { MlxTqRowBatch batch; std::exception_ptr error; };
        std::vector<Part> parts((ids.size() + grain - 1) / grain);
        struct Work {
            const std::vector<MlxMappedTqRows>& tables;
            const std::vector<std::pair<std::size_t, std::int64_t>>& ids;
            std::vector<Part>& parts;
            std::size_t grain;
        } work{tables, ids, parts, std::size_t(grain)};
        dispatch_apply_f(parts.size(), dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), &work,
            [](void* context, std::size_t part) {
                auto& work = *static_cast<Work*>(context);
                try {
                    const auto end = std::min(work.ids.size(), (part + 1) * work.grain);
                    for (auto index = part * work.grain; index < end; ++index) {
                        const auto [table, row] = work.ids[index];
                        work.tables[table].append_row(row, work.parts[part].batch);
                    }
                } catch (...) { work.parts[part].error = std::current_exception(); }
            });
        MlxTqRowBatch batch;
        for (const auto& part : parts) {
            if (part.error) std::rethrow_exception(part.error);
            batch.append_batch(part.batch);
        }
        return batch;
    };
    auto expected = read(1).decode();
    expected.eval();
    for (int grain : {1, 32, 64}) {
        auto actual = read(grain).decode();
        if (max(abs(astype(actual, float32) - astype(expected, float32))).item<float>() != 0.f)
            throw std::runtime_error("TQ row-read task grouping changes decoded weights");
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
        const std::vector<int> grains = repeat % 2 ? std::vector<int>{64, 32, 1} : std::vector<int>{1, 32, 64};
        for (const int grain : grains) {
            const auto started = Clock::now();
            auto batch = read(grain);
            const auto read_done = Clock::now();
            batch.decode().eval();
            const auto finished = Clock::now();
            std::cout << nlohmann::json{{"kind", "tq_row_read"}, {"repeat", repeat}, {"task_rows", grain},
                {"selected_rows", ids.size()}, {"read_ms", std::chrono::duration<double, std::milli>(read_done-started).count()},
                {"decode_ms", std::chrono::duration<double, std::milli>(finished-read_done).count()},
                {"source_bytes", batch.source_bytes_read()}, {"prepared_bytes", batch.packed_nbytes()},
                {"boundary", "warm mapped row gather and merge; decode includes host-to-MLX copies and completion"}}
                << std::endl;
        }
    }
}

static void compare_resident(const MfqContainer& source, MlxQwen4CausalLm& streamed,
        int vocabulary, int prompt_tokens) {
    std::vector<std::int64_t> prompt(prompt_tokens);
    for (int i = 0; i < prompt_tokens; ++i) prompt[i] = (std::int64_t(i) * 7919 + 1234) % vocabulary;
    auto cache = streamed.make_request_cache();
    std::vector<array> expected;
    expected.push_back(streamed.prefill_request(*cache, prompt, false));
    expected.back().eval();
    for (int step = 0; step < 4; ++step) {
        expected.push_back(streamed.decode_requests({cache}, {1234 + step}));
        expected.back().eval();
    }
    auto resident = MlxQwen4CausalLm::load(source, prompt_tokens + 32);
    auto residency = MlxWeightResidency::finish_load(MlxMemoryResidency::configured_limit());
    auto resident_cache = resident.make_request_cache();
    auto actual = resident.prefill_request(*resident_cache, prompt, false);
    for (int step = 0; step < 5; ++step) {
        if (step) actual = resident.decode_requests({resident_cache}, {1233 + step});
        const auto error = max(abs(astype(actual, float32) - astype(expected[step], float32))).item<float>();
        std::cout << nlohmann::json{{"kind", "streamed_resident_comparison"},
            {"phase", step ? "decode" : "prefill"}, {"step", step}, {"max_absolute_logit_error", error},
            {"prompt_tokens", prompt_tokens}, {"mtp", false}, {"teacher_forced_decode", true}} << std::endl;
        if (error != 0.f) throw std::runtime_error("TQ streamed model differs from resident model");
    }
}

static void benchmark_concurrent(MlxQwen4CausalLm& model, int vocabulary,
        int prompt_tokens, int generated_tokens, int requested_batch = 0) {
    const auto batches = requested_batch ? std::vector<int>{requested_batch} : std::vector<int>{2, 4};
    for (const int requests : batches) {
        std::vector<std::shared_ptr<MlxQwen4RequestCache>> caches;
        std::vector<MlxQwen4TextSessionState> states;
        for (int row = 0; row < requests; ++row) {
            std::vector<std::int64_t> prompt(prompt_tokens - row * 7);
            for (int i = 0; i < int(prompt.size()); ++i)
                prompt[i] = (std::int64_t(i) * 7919 + 1234 + row * 101) % vocabulary;
            auto cache = model.make_request_cache();
            model.prefill_request(*cache, prompt, false).eval();
            states.push_back(model.capture_request_cache(*cache, prompt));
            caches.push_back(std::move(cache));
        }
        auto restore = [&] {
            for (int row = 0; row < requests; ++row)
                model.restore_request_cache(*caches[row], states[row]);
        };
        std::vector<std::vector<array>> reference(requests);
        for (int row = 0; row < requests; ++row) {
            for (int step = 0; step < 4; ++step) {
                auto logits = model.decode_requests({caches[row]}, {1234 + row * 101 + step});
                logits.eval();
                reference[row].push_back(std::move(logits));
            }
        }
        restore();
        for (int row = 0; row < requests; ++row) {
            auto restored = model.decode_requests({caches[row]}, {1234 + row * 101});
            auto error = max(abs(astype(restored, float32) - astype(reference[row][0], float32))).item<float>();
            std::cout << nlohmann::json{{"kind", "request_restore_check"}, {"row", row},
                {"max_absolute_logit_error", error}} << std::endl;
            if (error != 0.f) throw std::runtime_error("restored TQ request logits differ");
        }
        restore();
        double max_error = 0;
        int greedy_matches = 0;
        bool numerical_ok = true;
        std::optional<array> canonical_logits;
        std::vector<array> canonical_sequence;
        for (int step = 0; step < 4; ++step) {
            std::vector<std::int64_t> tokens;
            for (int row = 0; row < requests; ++row) tokens.push_back(1234 + row * 101 + step);
            auto batched = reshape(model.decode_requests(caches, tokens), {requests, vocabulary});
            batched.eval();
            if (step == 0) canonical_logits = batched;
            canonical_sequence.push_back(batched);
            for (int row = 0; row < requests; ++row) {
                auto actual = slice(batched, {row, 0}, {row + 1, vocabulary});
                auto expected = reshape(reference[row][step], {1, vocabulary});
                const auto error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
                max_error = std::max(max_error, double(error));
                const bool close = allclose(actual, expected, .005, .02).item<bool>();
                numerical_ok &= close;
                if (!close) std::cout << nlohmann::json{{"kind", "concurrent_logit_difference"},
                    {"requests", requests}, {"row", row}, {"step", step}, {"max_absolute_error", error},
                    {"max_reference_logit", max(abs(astype(expected, float32))).item<float>()},
                    {"rms_error", sqrt(mean(square(astype(actual, float32) - astype(expected, float32)))).item<float>()}}
                    << std::endl;
                greedy_matches += argmax(actual, -1).item<std::uint32_t>()
                    == argmax(expected, -1).item<std::uint32_t>();
                if (model.request_cache_position(*caches[row]) != int(states[row].tokens.size()) + step + 1)
                    throw std::runtime_error("concurrent TQ request position differs");
            }
        }
        std::cout << nlohmann::json{{"kind", "concurrent_independent_comparison"}, {"requests", requests},
            {"max_absolute_logit_error", max_error}, {"greedy_matches", greedy_matches},
            {"within_tolerance", numerical_ok}, {"comparisons", requests * 4},
            {"unequal_context_lengths", true}} << std::endl;
        restore();
        model.restore_request_cache(*caches.back(), states.front());
        float companion_error = 0.f;
        for (int step = 0; step < 4; ++step) {
            std::vector<std::int64_t> tokens;
            for (int row = 0; row < requests; ++row)
                tokens.push_back(1234 + (row == requests - 1 ? 0 : row * 101) + step);
            const auto changed = reshape(model.decode_requests(caches, tokens), {requests, vocabulary});
            const auto actual = slice(changed, {0, 0}, {1, vocabulary});
            const auto expected = slice(canonical_sequence[step], {0, 0}, {1, vocabulary});
            const auto error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
            companion_error = std::max(companion_error, error);
            if (!std::isfinite(error) || error != 0.f)
                throw std::runtime_error("changing another TQ request changes the first request logits");
        }
        std::cout << nlohmann::json{{"kind", "concurrent_companion_check"}, {"requests", requests},
            {"decode_steps", 4}, {"max_absolute_logit_error", companion_error},
            {"same_batch_geometry", true}, {"changed_context_and_tokens", true}} << std::endl;
        restore();
        auto reversed_caches = caches;
        std::reverse(reversed_caches.begin(), reversed_caches.end());
        std::vector<std::int64_t> reversed_tokens;
        for (int row = requests - 1; row >= 0; --row) reversed_tokens.push_back(1234 + row * 101);
        auto reversed = reshape(model.decode_requests(reversed_caches, reversed_tokens), {requests, vocabulary});
        std::vector<int> row_order;
        for (int row = requests - 1; row >= 0; --row) row_order.push_back(row);
        auto reordered = take(reversed, array(row_order.begin(), Shape{requests}, int32), 0);
        const auto permutation_error = max(abs(astype(reordered, float32) - astype(*canonical_logits, float32))).item<float>();
        std::cout << nlohmann::json{{"kind", "concurrent_permutation_check"}, {"requests", requests},
            {"max_absolute_logit_error", permutation_error}} << std::endl;
        if (permutation_error != 0.f) throw std::runtime_error("TQ request order changes batched logits");
        for (int repeat = 0; repeat < 3; ++repeat) {
            restore();
            std::vector<std::int64_t> tokens;
            for (int row = 0; row < requests; ++row) tokens.push_back(1234 + row * 101);
            const auto started = Clock::now();
            for (int step = 0; step < generated_tokens; ++step) {
                auto logits = model.decode_requests(caches, tokens);
                auto next = argmax(reshape(logits, {requests, vocabulary}), -1);
                next.eval();
                for (int row = 0; row < requests; ++row)
                    tokens[row] = next.data<std::uint32_t>()[row];
            }
            const auto seconds = std::chrono::duration<double>(Clock::now() - started).count();
            std::cout << nlohmann::json{{"kind", "concurrent_benchmark"}, {"requests", requests},
                {"repeat", repeat}, {"decode_steps_per_request", generated_tokens},
                {"seconds", seconds}, {"aggregate_tokens_per_second", requests * generated_tokens / seconds},
                {"per_request_tokens_per_second", generated_tokens / seconds}, {"mtp", false},
                {"boundary", "native request decode, argmax, token readback and one completion wait per batch"}}
                << std::endl;
        }
    }
}

static void check_cache_resize(MlxQwen4CausalLm& model, int vocabulary, int prompt_tokens) {
    std::vector<std::int64_t> prompt(prompt_tokens);
    for (int i = 0; i < prompt_tokens; ++i) prompt[i] = (std::int64_t(i) * 7919 + 1234) % vocabulary;
    auto cache = model.make_request_cache();
    model.prefill_request(*cache, prompt, false).eval();
    auto state = model.capture_request_cache(*cache, prompt);
    auto expected = model.decode_requests({cache}, {1234});
    expected.eval();
    auto expected_next = model.decode_requests({cache}, {1235});
    expected_next.eval();
    const auto original = model.expert_cache_limit_bytes() ? model.expert_cache_limit_bytes() : std::size_t{30} << 30;
    for (const auto budget : {original * 3 / 4, original / 2, original}) {
        model.restore_request_cache(*cache, state);
        auto pending = model.decode_requests({cache}, {1234});
        const auto before_stats = model.ssd_expert_cache_stats();
        const auto before = before_stats ? before_stats->resident_bytes : model.resident_full_expert_bytes();
        const auto started = Clock::now();
        const auto released = model.set_expert_cache_limit(budget);
        const auto after = model.ssd_expert_cache_stats()->resident_bytes;
        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        const auto pending_error = max(abs(astype(pending, float32) - astype(expected, float32))).item<float>();
        auto actual = model.decode_requests({cache}, {1235});
        const auto error = max(abs(astype(actual, float32) - astype(expected_next, float32))).item<float>();
        std::cout << nlohmann::json{{"kind", "expert_cache_resize_check"}, {"budget_bytes", budget},
            {"allocated_before", before}, {"allocated_after", after}, {"released_bytes", released},
            {"resize_seconds", seconds}, {"max_absolute_logit_error", error},
            {"pending_logit_error", pending_error}, {"kv_preserved", true},
            {"resident_to_streamed", !before_stats.has_value()}} << std::endl;
        if (error != 0.f || pending_error != 0.f || after > budget
            || (before_stats && released != (before > after ? before - after : 0))
            || model.resident_full_expert_bytes() != 0)
            throw std::runtime_error("TQ live cache resize changes model output or accounting");
    }
}

template <typename Model>
static auto benchmark_cache(Model& model) { return model.make_request_cache(); }

static std::nullptr_t benchmark_cache(MlxDeepseekV41CausalLm&) { return nullptr; }

template <typename Model, typename Cache>
static array benchmark_prefill(Model& model, Cache& cache, const std::vector<std::int64_t>& prompt) {
    return model.prefill_request(*cache, prompt, false);
}

static array benchmark_prefill(MlxDeepseekV41CausalLm& model, std::nullptr_t&,
        const std::vector<std::int64_t>& prompt) {
    return model.prefill(array(prompt.begin(), Shape{1, int(prompt.size())}, int64), 4096, false);
}

template <typename Model, typename Cache>
static array benchmark_decode(Model& model, Cache& cache, std::int64_t token) {
    return model.decode_requests({cache}, {token});
}

static array benchmark_decode(MlxDeepseekV41CausalLm& model, std::nullptr_t&, std::int64_t token) {
    return model.decode(array({token}, Shape{1, 1}, int64));
}

template <typename Model>
static void benchmark(Model& model, int vocabulary, int prompt_tokens, int generated_tokens, bool profile) {
    std::vector<std::int64_t> prompt(prompt_tokens);
    for (int i = 0; i < prompt_tokens; ++i) prompt[i] = (std::int64_t(i) * 7919 + 1234) % vocabulary;
    for (int repeat = 0; repeat < 3; ++repeat) {
        model.reset_cache();
        auto cache = benchmark_cache(model);
        const auto before = model.ssd_expert_cache_stats();
        const auto started = Clock::now();
        auto logits = benchmark_prefill(model, cache, prompt);
        logits.eval();
        const auto prefilled = Clock::now();
        const auto after_prefill = model.ssd_expert_cache_stats();
        if (!all(isfinite(logits)).template item<bool>())
            throw std::runtime_error("nonfinite full-model prefill logits");
        auto token = argmax(reshape(slice(reshape(logits, {-1}),
            {int(logits.size()) - vocabulary}, {int(logits.size())}), {1, vocabulary}), -1);
        token.eval();
        const auto decoding = Clock::now();
        for (int step = 0; step < generated_tokens; ++step) {
            logits = benchmark_decode(model, cache, token.template item<std::uint32_t>());
            token = argmax(reshape(logits, {1, vocabulary}), -1);
            token.eval();
        }
        const auto finished = Clock::now();
        if (!all(isfinite(logits)).template item<bool>())
            throw std::runtime_error("nonfinite full-model decode logits");
        const double pp = std::chrono::duration<double>(prefilled - started).count();
        const double decode = std::chrono::duration<double>(finished - decoding).count();
        nlohmann::json result{{"repeat", repeat}, {"first_execution", repeat == 0},
            {"prompt_tokens", prompt_tokens}, {"decode_steps", generated_tokens},
            {"prefill_seconds", pp}, {"prefill_tokens_per_second", prompt_tokens / pp},
            {"decode_seconds", decode}, {"decode_tokens_per_second", generated_tokens / decode},
            {"sampling", "greedy"}, {"mtp", false}, {"prefix_reuse", false},
            {"prefill_logits", "last_position"},
            {"boundary", std::is_same_v<Model, MlxDeepseekV41CausalLm>
                ? "native backbone prefill/decode, argmax, token readback and one completion wait per step"
                : "native serving request prefill/decode, argmax, token readback and one completion wait per step"}};
        if (const auto after = model.ssd_expert_cache_stats(); after && before && after_prefill) {
            const auto requests = after->requests - after_prefill->requests;
            result["expert_cache"] = {
                {"allocated_bytes", after->resident_bytes}, {"cached_experts", after->resident_experts},
                {"prefill_bytes_read", after_prefill->bytes_read - before->bytes_read},
                {"prefill_io_seconds", after_prefill->io_seconds - before->io_seconds},
                {"decode_bytes_read", after->bytes_read - after_prefill->bytes_read},
                {"decode_io_seconds", after->io_seconds - after_prefill->io_seconds},
                {"decode_route_sync_seconds", after->route_sync_seconds - after_prefill->route_sync_seconds},
                {"decode_prepare_seconds", after->prepare_seconds - after_prefill->prepare_seconds},
                {"decode_routed_graph_seconds", after->routed_graph_seconds - after_prefill->routed_graph_seconds},
                {"decode_hit_rate", requests ? double(after->hits - after_prefill->hits) / requests : 0.0},
                {"decode_read_batches", after->read_calls - after_prefill->read_calls}};
        }
        std::cout << result << std::endl;
    }
    if (profile) {
        model.reset_cache();
        auto cache = benchmark_cache(model);
        detail::ComponentProfile components;
        {
            detail::ScopedComponentProfile scope(&components);
            auto logits = benchmark_prefill(model, cache, prompt);
            logits.eval();
        }
        for (const auto& [name, timing] : components.timings())
            std::cout << nlohmann::json{{"kind", "component_profile"}, {"stage", "prefill"},
                {"component", name}, {"milliseconds", timing.elapsed_ms}, {"calls", timing.evaluations}} << std::endl;
        detail::ComponentProfile decode_components;
        {
            detail::ScopedComponentProfile scope(&decode_components);
            auto logits = benchmark_decode(model, cache, 1234);
            logits.eval();
        }
        for (const auto& [name, timing] : decode_components.timings())
            std::cout << nlohmann::json{{"kind", "component_profile"}, {"stage", "decode"},
                {"component", name}, {"milliseconds", timing.elapsed_ms}, {"calls", timing.evaluations}} << std::endl;
    }
}

int main(int argc, char** argv) {
    try {
        bool profile = false, concurrent = false, rows = false, resident_check = false, resize_check = false;
        int requested_batch = 0;
        std::optional<std::size_t> expert_cache_bytes;
        while (argc >= 3 && std::string(argv[argc - 1]).starts_with("--")) {
            const std::string flag(argv[--argc]);
            if (flag == "--profile") profile = true;
            else if (flag == "--concurrent") concurrent = true;
            else if (flag.starts_with("--concurrent=")) {
                requested_batch = std::stoi(flag.substr(std::string("--concurrent=").size()));
                if (requested_batch < 2 || requested_batch > 16)
                    throw std::invalid_argument("invalid concurrent benchmark batch size");
                concurrent = true;
            }
            else if (flag == "--rows") rows = true;
            else if (flag == "--resident-check") resident_check = true;
            else if (flag == "--resize-check") resize_check = true;
            else if (flag.starts_with("--expert-cache-gib=")) {
                const auto gib = std::stoull(flag.substr(std::string("--expert-cache-gib=").size()));
                if (gib < 1 || gib > 96) throw std::invalid_argument("invalid expert cache benchmark budget");
                expert_cache_bytes = gib << 30;
            }
            else throw std::invalid_argument("unknown model benchmark flag");
        }
        if (argc < 2 || argc > 4)
            throw std::invalid_argument("usage: mfq-metal-tq-model-benchmark model.mfq [prompt_tokens [decode_steps]]");
        const int prompt_tokens = argc >= 3 ? std::stoi(argv[2]) : 4096;
        const int generated_tokens = argc >= 4 ? std::stoi(argv[3]) : 64;
        if (prompt_tokens < 1 || prompt_tokens > 65536 || generated_tokens < 1 || generated_tokens > 1024)
            throw std::invalid_argument("invalid model benchmark token counts");
        MfqContainer source(argv[1]);
        if (rows) { benchmark_rows(source, prompt_tokens); return 0; }
        MlxMemoryResidency::configure(std::size_t{96} << 30);
        const auto started = Clock::now();
        if (source.header().architecture.starts_with("qwen4")) {
            auto model = MlxQwen4CausalLm::load(source, prompt_tokens + generated_tokens + 16, expert_cache_bytes);
            auto residency = MlxWeightResidency::finish_load(MlxMemoryResidency::configured_limit());
            std::cout << nlohmann::json{{"loaded", true}, {"architecture", source.header().architecture},
                {"layers", model.layer_count()}, {"expert_cache_bytes", model.expert_cache_limit_bytes()},
                {"load_seconds", std::chrono::duration<double>(Clock::now() - started).count()}} << std::endl;
            if (resize_check) {
                check_cache_resize(model, model.config().vocab_size, prompt_tokens);
            } else if (resident_check) {
                if (!expert_cache_bytes) throw std::invalid_argument("resident comparison needs an expert cache budget");
                compare_resident(source, model, model.config().vocab_size, prompt_tokens);
            } else if (concurrent) {
                if (prompt_tokens < 22) throw std::invalid_argument("concurrent benchmark needs at least 22 prompt tokens");
                benchmark_concurrent(model, model.config().vocab_size, prompt_tokens, generated_tokens, requested_batch);
            } else benchmark(model, model.config().vocab_size, prompt_tokens, generated_tokens, profile);
        } else if (source.header().architecture == "deepseek_v41") {
            if (concurrent || resident_check || resize_check)
                throw std::invalid_argument("DeepSeek TQ benchmark requires the single-backbone mode");
            auto model = MlxDeepseekV41CausalLm::load(source, prompt_tokens + generated_tokens + 16, expert_cache_bytes);
            auto residency = MlxWeightResidency::finish_load(MlxMemoryResidency::configured_limit());
            std::cout << nlohmann::json{{"loaded", true}, {"architecture", source.header().architecture},
                {"layers", model.layer_count()}, {"expert_cache_bytes", model.expert_cache_limit_bytes()},
                {"load_seconds", std::chrono::duration<double>(Clock::now() - started).count()}} << std::endl;
            benchmark(model, model.config().vocab, prompt_tokens, generated_tokens, profile);
        } else {
            throw std::invalid_argument("unsupported model benchmark architecture");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
