#include "request_executor.h"
#include "generation_flow.h"
#include "runtime_config.h"

#include <cassert>
#include <cstdlib>
#include <map>

using namespace mfq::engine;

struct TestOps {
    bool batch = false, reset_fails = false;
    int failure = 0, live = 0, resets = 0, advances = 0, accepted = 0;
    std::vector<PrefillChunk> chunks;
    std::map<RequestId, ExecutionRequest *> batched;
    bool can_batch(const EngineRequest &) const { return batch; }
    bool exclusive() const { return false; }
    bool mtp_available() const { return false; }
    void execute(const std::vector<RequestId>&) {}
    void reset() {
        ++resets;
        if (reset_fails)
            throw std::runtime_error("cleanup failed");
    }
    double prefill(PrefillChunk chunk) {
        chunks.push_back(chunk);
        return 1.0;
    }
    std::int64_t first_token() { return 10; }
    std::int64_t advance() { return 10 + ++advances; }
    void accept(std::int64_t) { ++accepted; }
    Generation generate(const RequestId& id, ExecutionRequest& request) {
        auto& input = request.input;
        auto& output = request.output;
        struct Storage {
            TestOps& ops;
            RequestId id;
            Storage(TestOps& ops, const RequestId& id, ExecutionRequest& request) : ops(ops), id(id) {
                ++ops.live;
                if (ops.batch) ops.batched.emplace(id, &request);
            }
            ~Storage() { --ops.live; ops.batched.erase(id); }
        } storage(*this, id, request);
        ExecutionCleanup cleanup{output.cleanup_failure, [&] { reset(); }};
        if (failure == 1)
            throw InferenceInputError(InferenceInputErrorCode::Invalid, "invalid media");
        if (failure == 2)
            throw MfqBackendFailure("device failed");
        if (failure == 4)
            throw 42;
        auto sequence = generate_sequence(
            std::ref(*this), output, input.prompt.size(), 0, input.cache_plan.stable_prefix_tokens, 2);
        while (auto event = sequence.next())
            co_yield std::move(event);
        cleanup.finish();
    }
};

static EngineRequest request(std::string id = "one") {
    EngineRequest result;
    result.id = std::move(id);
    result.token_ids = {1, 2, 3, 4, 5};
    result.input.sampling.max_tokens = 2;
    return result;
}

static EngineInfo info() {
    EngineInfo result;
    result.max_requests = 2;
    result.max_context = 32;
    result.vocab_size = 32;
    return result;
}

static const EventData &terminal_event(const EngineStepResult &result) {
    assert(std::count_if(result.events.begin(), result.events.end(), [](const auto &event) {
        return terminal(event.data);
    }) == 1);
    assert(terminal(result.events.back().data));
    return result.events.back().data;
}

struct TestBackend {
    struct Options { std::int64_t context_size = 32; } options;
    struct Operations : TestOps {
        struct Cache {
            std::uint64_t count = 0;
            std::uint64_t fork_session(const std::string &, const std::string &) { return 0; }
            std::uint64_t close_session(const std::string &) { return 0; }
            std::uint64_t clear() { return std::exchange(count, 0); }
            std::uint64_t trim_hot(std::uint64_t) { return clear(); }
            Metrics metrics() const { return {{"sessions", double(count)}}; }
        } cache;
        template <class T> ControlResult control(T) {
            if constexpr (std::is_same_v<T, RuntimeMetrics>)
                return Metrics{{"advances", double(advances)}};
            else throw std::invalid_argument("no duplex in test backend");
        }
    } ops;
    ~TestBackend() { assert(ops.live == 0); }
    bool loaded = false, fail_load = false;
    int loads = 0;
    auto load() {
        assert(!loaded && ops.live == 0);
        loaded = true;
        ++loads;
        if (fail_load) throw std::runtime_error("load failed after allocation");
        ops = Operations{};
        auto metadata = info();
        metadata.max_context = options.context_size;
        return std::pair{metadata, std::unique_ptr<TextProcessor>{}};
    }
    template <class F> decltype(auto) visit(F&& run) {
        if (!loaded) throw std::runtime_error("unloaded");
        return run(ops);
    }
    bool exclusive() const { return false; }
    void unload() {
        assert(ops.live == 0); // Suspended requests must die before native state.
        ops.batched.clear();
        loaded = false;
    }
};

template <class F> static void rejects(F&& run) {
    bool failed = false;
    try { run(); } catch (const std::exception&) { failed = true; }
    assert(failed);
}

static void check_engine_lifecycle() {
    EngineInstance<TestBackend> first_instance({32}), second_instance({64});
    Engine &first = first_instance, &second = second_instance;
    first_instance.backend.ops.cache.count = 3;
    assert(first.session({SessionCommand::Kind::clear}).count == 3);
    assert(first.session({SessionCommand::Kind::metrics}).metrics == Metrics({{"sessions", 0}}));
    assert(std::get<Metrics>(first.control(RuntimeMetrics{})) == Metrics({{"advances", 0}}));
    rejects([&] { first.control(DecodeTokens{{1, 2}, {}}); });
    rejects([&] { first.control(CountTokens{"prompt"}); });
    assert(first.info().max_context == 32 && second.info().max_context == 64);
    assert(first.admit(request()) == Admission::accepted);
    first.step({"one"});
    assert(first_instance.backend.ops.live == 1);
    rejects([&] { first.reload(48); });
    rejects([&] { second.reload(0); });
    assert(first_instance.backend.loads == 1 && second_instance.backend.loads == 1);
    first.cancel("one");
    assert(std::holds_alternative<Cancelled>(terminal_event(first.step({}))));
    assert(first.reload(48) == 48 && first.status().healthy);
    assert(second.info().max_context == 64 && second_instance.backend.loads == 1);
    first_instance.backend.ops.failure = 2;
    first.admit(request());
    assert(std::holds_alternative<Failed>(terminal_event(first.step({"one"}))));
    assert(!first.status().healthy);
    assert(first.reload(32) == 32 && first.status().healthy);
    first_instance.backend.fail_load = true;
    rejects([&] { first.reload(40); });
    assert(!first.status().healthy && !first_instance.backend.loaded);
    first_instance.backend.fail_load = false;
    assert(first.reload(40) == 40 && first.status().available == 2);
    first.admit(request());
    first.step({"one"});
    first.shutdown();
    first.shutdown();
    assert(!first.status().healthy && first_instance.backend.ops.live == 0);
    assert(second.status().healthy);
    std::unique_ptr<Engine> owned = std::make_unique<EngineInstance<TestBackend>>(TestBackend::Options{});
    owned->admit(request());
    owned->step({"one"});
    // Virtual destruction must release a suspended coroutine before its backend.
    owned.reset();
}


static void check_waiting_and_cleanup() {
    struct WaitingOps : TestOps {
        Generation generate(const RequestId&, ExecutionRequest& request) {
            ExecutionCleanup cleanup{request.output.cleanup_failure, [&] { reset(); }};
            co_yield StepState::advanced; // Work was submitted.
            while (!request.output.stopped()) co_yield StepState::waiting;
            cleanup.finish();
        }
    };
    WaitingOps ops;
    RequestExecutor executor;
    executor.admit(request(), nullptr, info(), ops);
    auto submitted = executor.step({"one"}, ops);
    assert(submitted.advanced == std::vector<RequestId>{"one"} && submitted.events.empty());
    for (int i = 0; i < 3; ++i) {
        auto waiting = executor.step({"one"}, ops);
        assert(waiting.advanced.empty() && waiting.events.empty() && waiting.has_work);
    }
    executor.cancel("one");
    auto cancelled = executor.step({}, ops);
    assert(std::holds_alternative<Cancelled>(terminal_event(cancelled)) && ops.resets == 1);

    // A cleanup error overrides cancellation, is reported once, and closes admission.
    WaitingOps broken;
    RequestExecutor failed;
    failed.admit(request(), nullptr, info(), broken);
    failed.step({"one"}, broken);
    broken.reset_fails = true;
    failed.cancel("one");
    auto result = failed.step({}, broken);
    assert(std::get<Failed>(terminal_event(result)).message == "cleanup failed");
    assert(!result.status.healthy && failed.empty() && broken.resets == 1);
    assert(failed.step({}, broken).events.empty() && broken.resets == 1);

    // Forced destruction observes the same cleanup error before unloading resources.
    EngineInstance<TestBackend> engine({32});
    engine.admit(request());
    engine.step({"one"});
    engine.backend.ops.reset_fails = true;
    rejects([&] { engine.shutdown(); });
    assert(!engine.backend.loaded && engine.backend.ops.live == 0 && engine.backend.ops.resets == 1);
    engine.shutdown();
}

static void check_preparation_steps() {
    struct MediaOps : TestOps {
        using Prepared = int;
        int encoded = 0, media_live = 0, text_runs = 0, fail_at = 0;
        mfq::StepSequence<Prepared> prepare(InferenceRequest&) {
            ++media_live;
            struct Storage { int& live; ~Storage() { --live; } } storage{media_live};
            for (int i = 0; i < 3; ++i) {
                if (++encoded == fail_at) throw std::runtime_error("encoder failed");
                co_yield StepState::advanced;
            }
            co_yield 42;
        }
        auto advance_preparation(mfq::StepSequence<Prepared>& preparation) {
            return std::pair{preparation.next(), 1.0};
        }
        Generation generate_text(InferenceRequest& input, InferenceOutput& output,
            std::optional<Prepared> prepared, const RequestId&, bool) {
            assert(prepared == 42);
            ++text_runs;
            ExecutionCleanup cleanup{output.cleanup_failure, [&] { reset(); }};
            auto sequence = generate_sequence(std::ref(*this), output, input.prompt.size(), 0, 0, 2);
            while (auto step = sequence.next()) co_yield std::move(step);
            cleanup.finish();
        }
        Generation generate(const RequestId& id, ExecutionRequest& request) {
            request.input.vision.emplace();
            return generate_prepared(*this, request.input, request.output, id, false);
        }
    };
    for (int stop_after = 0; stop_after <= 4; ++stop_after) {
        MediaOps ops;
        RequestExecutor executor;
        executor.admit(request(), nullptr, info(), ops);
        for (int tick = 0; tick < stop_after; ++tick) {
            const auto step = executor.step({"one"}, ops);
            assert(step.advanced == std::vector<RequestId>{"one"});
        }
        const auto encoded = ops.encoded;
        executor.cancel("one");
        const auto result = executor.step({}, ops);
        assert(std::holds_alternative<Cancelled>(terminal_event(result)));
        assert(executor.empty() && ops.media_live == 0 && ops.encoded == encoded && ops.text_runs == 0);
    }
    for (bool fail : {false, true}) {
        MediaOps ops;
        ops.fail_at = fail ? 2 : 0;
        RequestExecutor executor;
        executor.admit(request(), nullptr, info(), ops);
        EngineStepResult result;
        for (int ticks = 0; ticks < 20 && !executor.empty(); ++ticks) result = executor.step({"one"}, ops);
        assert(executor.empty() && ops.media_live == 0);
        assert(std::holds_alternative<Failed>(terminal_event(result)) == fail);
        assert(ops.text_runs == (fail ? 0 : 1));
        assert(result.status.healthy);
        if (fail) {
            ops.fail_at = 0;
            executor.admit(request(), nullptr, info(), ops);
            for (int ticks = 0; ticks < 20 && !executor.empty(); ++ticks)
                result = executor.step({"one"}, ops);
            assert(std::holds_alternative<Completed>(terminal_event(result)));
        }
        if (!fail) assert(std::get<Completed>(terminal_event(result)).metrics.multimodal_ms == 4);
    }
}

static void check_execution_error_isolation() {
    struct Operations : TestOps {
        void execute(const std::vector<RequestId>& eligible) {
            if (std::find(eligible.begin(), eligible.end(), "bad") != eligible.end())
                throw std::runtime_error("request execution failed");
        }
    } ops;
    ops.batch = true;
    RequestExecutor executor(2);
    executor.admit(request("bad"), nullptr, info(), ops);
    executor.admit(request("good"), nullptr, info(), ops);
    auto failed = executor.step({"bad"}, ops);
    assert(failed.status.healthy && failed.has_work);
    assert(failed.events.back().id == "bad");
    assert(std::holds_alternative<Failed>(terminal_event(failed)));
    EngineStepResult result;
    for (int ticks = 0; ticks < 20 && !executor.empty(); ++ticks)
        result = executor.step({"good"}, ops);
    assert(executor.empty() && result.status.healthy);
    assert(result.events.back().id == "good");
    assert(std::holds_alternative<Completed>(terminal_event(result)));
}

struct Environment {
    const char* name;
    std::optional<std::string> previous;
    Environment(const char* name, const char* value) : name(name) {
        if (const char* old = std::getenv(name)) previous = old;
        set(value);
    }
    void set(const char* value) {
#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        if (value) setenv(name, value, 1); else unsetenv(name);
#endif
    }
    ~Environment() { set(previous ? previous->c_str() : nullptr); }
};

static void check_runtime_config() {
    const ContinuousBatchConfig positional{2, 16};
    assert(positional.max_sequences == 2 && positional.prefill_token_budget == 16);
    assert(positional.sequence_limit == 32);
    Environment sessions("MFQ_RUNTIME_MAX_KV_SESSIONS", "3");
    Environment block("MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS", "512");
    Environment directory("MFQ_RUNTIME_PREFIX_CACHE_DIR", "cache-example");
    Environment budget("MFQ_CONTINUOUS_BATCH_PREFILL_TOKEN_BUDGET", "7");
    auto config = resolve_runtime_config(16);
    assert(config.generation.prefill_chunk_size == 16);
    assert(config.session_cache.snapshots.max_sessions == 3);
    assert(config.prefix_cache.block_tokens == 512 && config.prefix_cache.directory == "cache-example");
    auto batch = resolve_batch_config(2, 16);
    assert(batch.max_sequences == 2 && batch.prefill_token_budget == 7);
    rejects([] { resolve_runtime_config(0); });
    rejects([] { resolve_batch_config(-1, 16); });
    for (const char* bad : {"-1", "12x", "18446744073709551616"}) {
        sessions.set(bad);
        rejects([] { resolve_runtime_config(16); });
    }
    sessions.set("3");
    for (const char* bad : {"0", "65537"}) {
        block.set(bad);
        rejects([] { resolve_runtime_config(16); });
    }
    budget.set("0");
    rejects([] { resolve_batch_config(2, 16); });
    budget.set(nullptr);
    assert(resolve_batch_config(2, 16).prefill_token_budget == 16);
    Environment limit("MFQ_CONTINUOUS_BATCH_MAX_SEQUENCES", "12");
    assert(resolve_batch_config(12, 16).sequence_limit == 12);
    rejects([] { resolve_batch_config(13, 16); });
    for (const char* bad : {"0", "33", "64", "65", "-1", "4x"}) {
        limit.set(bad);
        rejects([] { resolve_batch_config(2, 16); });
    }
    limit.set(nullptr);
    assert(resolve_batch_config(32, 16).max_sequences == 32);
    Environment initial("MFQ_INIT_NEW_TOKEN_RATIO", "0.5");
    Environment minimum("MFQ_MIN_NEW_TOKEN_RATIO_FACTOR", "0.2");
    Environment conservativeness("MFQ_SCHEDULE_CONSERVATIVENESS", "0.8");
    Environment decay("MFQ_NEW_TOKEN_RATIO_DECAY_STEPS", "2");
    NewTokenRatioTracker ratio(resolve_batch_config(2, 16).admission);
    assert(std::abs(ratio.current() - .4) < 1e-12);
    ratio.decay_step();
    assert(std::abs(ratio.current() - .24) < 1e-12);
    for (const char* valid : {"0.5", ".5", "5e-1"}) {
        initial.set(valid);
        assert(resolve_batch_config(2, 16).admission.initial_ratio == .5);
    }
    for (const char* bad : {"0", "-1", "1.1", "nan", "inf", "0.5x",
                            " 0.5", "0.5 ", "+0.5", "0,5", "0x1p-1", "1e9999"}) {
        initial.set(bad);
        rejects([] { resolve_batch_config(2, 16); });
    }
    initial.set("0.5");
    decay.set("0");
    rejects([] { resolve_batch_config(2, 16); });
    Environment enabled("MFQ_RUNTIME_TRACE_SESSION_CACHE", "false");
    rejects([] { environment_enabled("MFQ_RUNTIME_TRACE_SESSION_CACHE"); });
}

int main() {
    // Text admission needs a tokenizer; a raw token request does not.
    {
        TestOps ops;
        RequestExecutor executor;
        auto text = request();
        text.token_ids.clear();
        bool rejected = false;
        try {
            executor.admit(std::move(text), nullptr, info(), ops);
        } catch (const std::invalid_argument &error) {
            rejected = true;
            assert(std::string(error.what()) == "text input requires a tokenizer");
        }
        assert(rejected && executor.empty() && executor.status().healthy);
        assert(executor.admit(request(), nullptr, info(), ops) == Admission::accepted);
    }
    for (const bool batched : {false, true}) {
        TestOps ops;
        ops.batch = batched;
        auto metadata = info();
        metadata.stop_token_ids = model_stop_token_ids(R"({"eos_token_id":[11,12]})", metadata.vocab_size);
        auto input = request();
        input.input.sampling.max_tokens = 8;
        RequestExecutor executor;
        assert(executor.admit(std::move(input), nullptr, metadata, ops) == Admission::accepted);
        std::vector<int64_t> tokens;
        bool complete = false;
        for (int step = 0; step < 32 && !complete; ++step) {
            const auto events = executor.step({"one"}, ops);
            for (const auto& event : events.events) {
                if (const auto* delta = std::get_if<OutputDelta>(&event.data))
                    tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
                if (const auto* done = std::get_if<Completed>(&event.data)) {
                    assert(done->finish_reason == "stop" && done->usage.completion_tokens == 2);
                    complete = true;
                }
            }
        }
        assert(complete && executor.empty() && ops.advances == 1);
        assert((tokens == std::vector<int64_t>{10, 11}));
    }
    check_waiting_and_cleanup();
    check_preparation_steps();
    check_execution_error_isolation();
    check_engine_lifecycle();
    check_runtime_config();
    // A restored prefix and snapshot boundary must not enlarge a prefill tick.
    {
        TestOps ops;
        InferenceRequest input;
        input.prompt.resize(7, 1);
        input.sampling.max_tokens = 3;
        InferenceOutput output(input, nullptr, "chunks");
        auto sequence = generate_sequence(std::ref(ops), output, 7, 1, 4, 2);
        std::vector<std::int64_t> tokens;
        std::size_t prefilled = 0;
        while (auto step = sequence.next()) {
            auto& event = step.value;
            if (!event) continue;
            if (auto *progress = std::get_if<PrefillProgress>(&*event))
                prefilled = progress->timing.prompt_tokens;
            if (auto *delta = std::get_if<OutputDelta>(&*event))
                tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
        }
        assert(prefilled == 6 && ops.chunks.size() == 4);
        assert(ops.chunks[0].offset == 1 && ops.chunks[0].count == 2);
        assert(ops.chunks[1].offset == 3 && ops.chunks[1].count == 1);
        assert(ops.chunks[2].offset == 4 && ops.chunks[2].count == 2);
        assert(ops.chunks[3].offset == 6 && ops.chunks[3].count == 1);
        assert((tokens == std::vector<std::int64_t>{10, 11, 12}));
        assert(ops.accepted == 3 && ops.advances == 2);
    }
    // Serial and batch use identical usage/terminal rules, including a zero
    // generation budget and cancellation while output-blocked.
    for (bool batch : {false, true}) {
        TestOps ops;
        ops.batch = batch;
        RequestExecutor executor(2);
        assert(executor.admit(request(), nullptr, info(), ops) == Admission::accepted);
        assert(executor.step({}, ops).events.empty());
        std::size_t tokens = 0, terminals = 0;
        for (int tick = 0; !executor.empty() && tick < 12; ++tick) {
            auto result = executor.step({"one"}, ops);
            for (const auto &event : result.events) {
                if (auto *delta = std::get_if<OutputDelta>(&event.data))
                    tokens += delta->token_ids.size();
                if (auto *done = std::get_if<Completed>(&event.data)) {
                    ++terminals;
                    assert(ops.live == 0 && ops.batched.empty());
                    assert(done->usage.prompt_tokens == 5 && done->usage.completion_tokens == 2);
                    assert(std::holds_alternative<UsageUpdate>(
                        result.events[result.events.size() - 2].data));
                    terminal_event(result);
                }
            }
        }
        assert(executor.empty() && tokens == 2 && terminals == 1);
        assert(executor.step({"one"}, ops).events.empty());
        auto zero = request();
        zero.input.sampling.max_tokens = 0;
        assert(executor.admit(std::move(zero), nullptr, info(), ops) == Admission::accepted);
        assert(std::get<Completed>(terminal_event(executor.step({"one"}, ops)))
                   .usage.completion_tokens == 0);
        for (bool started : {false, true}) {
            assert(executor.admit(request(), nullptr, info(), ops) == Admission::accepted);
            if (started)
                executor.step({"one"}, ops);
            const auto chunks = ops.chunks.size();
            executor.cancel("one");
            executor.cancel("one");
            assert(std::holds_alternative<Cancelled>(terminal_event(executor.step({}, ops))));
            assert(executor.empty() && ops.live == 0 && ops.batched.empty());
            assert(ops.chunks.size() == chunks);
        }
    }
    // An ineligible peer must not execute when another batched request cancels.
    {
        TestOps ops;
        ops.batch = true;
        RequestExecutor executor(2);
        assert(executor.admit(request("a"), nullptr, info(), ops) == Admission::accepted);
        assert(executor.admit(request("b"), nullptr, info(), ops) == Admission::accepted);
        assert(executor.admit(request("c"), nullptr, info(), ops) == Admission::deferred);
        executor.cancel("a");
        auto cancelled = executor.step({}, ops);
        assert(std::holds_alternative<Cancelled>(terminal_event(cancelled)));
        assert(cancelled.events.back().id == "a" && executor.status().available == 1);
        assert(ops.advances == 0 && ops.live == 0);
        executor.cancel("b");
        executor.step({}, ops);
    }
    // Request errors are recoverable; device or cleanup failures stop admission.
    for (int failure : {1, 2, 3, 4}) {
        TestOps ops;
        ops.failure = failure == 3 ? 1 : failure;
        ops.reset_fails = failure == 3;
        RequestExecutor executor;
        executor.admit(request(), nullptr, info(), ops);
        auto result = executor.step({"one"}, ops);
        const auto &error = std::get<Failed>(terminal_event(result));
        assert(error.code == (failure == 1 ? "invalid_request" : "backend_failure"));
        assert(ops.live == 0 && ops.resets == 1 && executor.empty());
        const bool recoverable = failure == 1 || failure == 4;
        assert(result.status.healthy == recoverable);
        ops.failure = 0;
        assert(executor.admit(request(), nullptr, info(), ops) ==
               (recoverable ? Admission::accepted : Admission::deferred));
    }
}
