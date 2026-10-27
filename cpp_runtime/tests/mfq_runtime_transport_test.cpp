#include "scheduler.h"
#include "common.h"
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <map>
#include <thread>
using namespace mfq::engine;
using namespace std::chrono_literals;

struct FakeEngine final : Engine {
    struct Work { EngineRequest request; int steps = 0; bool cancelled = false; };
    std::map<std::string, Work> active;
    std::map<std::string, EngineRequest> admitted;
    std::array<std::atomic<int>, 16> advances{};
    std::atomic<int> released{0}, reloads{0};
    std::atomic<bool> blocked{false}, resume{false};
    std::thread::id owner;
    bool fail = false;
    EngineInfo info() const override { return {2, 128, 64, false, false, true, {}}; }
    void check_thread() {
        if (owner == std::thread::id{}) owner = std::this_thread::get_id();
        assert(owner == std::this_thread::get_id());
    }
    Admission admit(EngineRequest request) override {
        check_thread();
        if (active.size() == 2) return Admission::deferred;
        auto id = request.id;
        admitted.insert_or_assign(id, request);
        active.emplace(id, Work{std::move(request)});
        return Admission::accepted;
    }
    void cancel(const RequestId& id) override {
        check_thread(); if (active.contains(id)) active.at(id).cancelled = true;
    }
    EngineStatus status() const override { return {2 - active.size(), true}; }
    EngineStepResult step(const std::vector<RequestId>& eligible) override {
        check_thread();
        EngineStepResult result;
        for (auto it = active.begin(); it != active.end();) {
            auto& work = it->second;
            const auto id = it->first;
            if (work.cancelled) {
                it = active.erase(it); ++released;
                result.events.push_back({id, Cancelled{}}); continue;
            }
            if (std::find(eligible.begin(), eligible.end(), id) == eligible.end()) { ++it; continue; }
            if (work.request.token_ids == std::vector<int64_t>{59} && !resume) {
                blocked = true;
                while (!resume) std::this_thread::sleep_for(1ms);
            }
            if (work.request.token_ids == std::vector<int64_t>{63}) throw MfqBackendFailure("device failure");
            if (work.request.token_ids == std::vector<int64_t>{58}) throw std::runtime_error("request failure");
            ++advances.at(std::stoi(id));
            ++work.steps;
            result.advanced.push_back(id);
            if (work.steps <= int(work.request.token_ids.size()))
                result.events.push_back({id, PrefillProgress{{std::size_t(work.steps), 1, 0, 1}}});
            else if (work.steps <= int(work.request.token_ids.size()) + work.request.input.sampling.max_tokens) {
                OutputDelta delta{{1, 2, 3}, {}};
                if (work.request.token_ids == std::vector<int64_t>{60} ||
                        work.request.token_ids == std::vector<int64_t>{61} ||
                        work.request.token_ids == std::vector<int64_t>{62}) {
                    common_chat_msg_diff diff;
                    const auto marker = work.request.token_ids.front();
                    diff.content_delta.assign(marker == 60 ? (work.steps == 2 ? 2000 : 6000)
                        : (marker == 61 ? 1024 : 8192), 'x');
                    delta.diffs.push_back(std::move(diff));
                }
                result.events.push_back({id, std::move(delta)});
            }
            if (work.steps > int(work.request.token_ids.size()) + work.request.input.sampling.max_tokens ||
                    (work.request.token_ids == std::vector<int64_t>{60} &&
                     work.steps == int(work.request.token_ids.size()) + work.request.input.sampling.max_tokens)) {
                it = active.erase(it); ++released;
                InferenceMetrics metrics;
                metrics.mtp.available = true;
                metrics.mtp.cycles = std::stoi(id);
                result.events.push_back({id, Completed{{}, {}, metrics}}); continue;
            }
            ++it;
        }
        result.status = status(); result.has_work = !active.empty(); return result;
    }
    SessionResult session(const SessionCommand&) override { check_thread(); return {7, {}}; }
    int64_t reload(int64_t context) override {
        check_thread(); assert(active.empty());
        if (context == 13) throw std::runtime_error("load failed");
        ++reloads; return context;
    }
    void shutdown() override { check_thread(); assert(active.empty()); }
    ControlResult control(ControlRequest) override { check_thread(); return Metrics{{"released", released.load()}}; }
};
EngineRequest request(int id, int tokens = 12) {
    EngineRequest result;
    result.id = std::to_string(id); result.token_ids = {1, 2};
    result.input.sampling.max_tokens = tokens; return result;
}
int drain(const std::shared_ptr<MfqScheduledRequest>& handle, bool cancelled = false, bool failed = false,
        const std::string& code = {}) {
    int terminals = 0;
    while (!handle->done()) for (const auto& event : handle->wait()) {
        if (const auto* delta = std::get_if<OutputDelta>(&event.data))
            assert(delta->token_ids == std::vector<int64_t>({1, 2, 3}));
        if (terminal(event.data)) {
            ++terminals;
            assert(std::holds_alternative<Cancelled>(event.data) == cancelled);
            assert(std::holds_alternative<Failed>(event.data) == failed);
            if (const auto* failure = std::get_if<Failed>(&event.data))
                assert(code.empty() || failure->code == code);
            if (const auto* completed = std::get_if<Completed>(&event.data)) {
                assert(completed->metrics.mtp.available);
                assert(completed->metrics.mtp.cycles == std::stoul(event.id));
            }
        }
    }
    assert(terminals == 1);
    assert(handle->wait().empty());
    return terminals;
}
void test_decode_timing() {
    const auto now = InferenceMetrics::Clock::now();
    InferenceMetrics metrics;
    metrics.started = now - 80ms;
    metrics.first_token = now - 70ms;
    metrics.last_token = now - 20ms;
    metrics.saw_token = true;
    InferenceResult result;
    result.completion_tokens = 6;
    const auto values = mfq::transport_detail::request_metric_values(result, metrics);
    assert(std::abs(values.ttft_ms - 10.0) < 1e-12);
    assert(std::abs(values.decode_ms - 50.0) < 1e-12);
    assert(std::abs(values.decode_tps - 100.0) < 1e-12);
    assert(values.generation_ms >= 80.0);
    assert(values.generation_ms - values.ttft_ms - values.decode_ms >= 20.0);
    result.cancelled = true;
    assert(mfq::transport_detail::request_metric_values(result, metrics).decode_ms == values.decode_ms);
    InferenceMetrics single;
    single.mark_token();
    result.completion_tokens = 1;
    const auto one = mfq::transport_detail::request_metric_values(result, single);
    assert(one.decode_ms == 0.0 && one.decode_tps == 0.0);
    result.completion_tokens = 0;
    const auto empty = mfq::transport_detail::request_metric_values(result, InferenceMetrics{});
    assert(empty.decode_ms == 0.0 && empty.decode_tps == 0.0);
}
void test_token_count_contract() {
    using namespace mfq::transport_detail;
    assert(parse_token_count_request({{"prompt", "raw <s> text"}}).prompt == "raw <s> text");
    for (const auto& invalid : std::vector<json>{
            json::array(), json::object(), {{"prompt", ""}}, {{"prompt", 1}},
            {{"prompt", "a"}, {"parse_special", true}}, {{"prompt", std::string(262145, 'a')}}}) {
        bool rejected = false;
        try { (void)parse_token_count_request(invalid); }
        catch (const ApiError&) { rejected = true; }
        assert(rejected);
    }
    const auto counted = token_count_json(TokenCount{2100, 2048});
    assert(counted["prompt_tokens"] == 2100 && counted["context_size"] == 2048);
    assert(counted["tokenization"] == "raw-no-special-tokens");
}

void test_benchmark_input_contract() {
    using namespace mfq::transport_detail;
    json params = {{"input", {{"messages", json::array({{{"role", "user"}, {"content", "test input"}}})}, {"preformatted_prompt", "test input"},
        {"benchmark_prompt_tokens", 32}}}, {"cache", {{"enabled", false}}}};
    auto body = runtime_generate_body(params);
    const auto input = parse_input(body, true, {});
    assert(!input.prefix_cache_enabled && input.benchmark_prompt_tokens == 32);
    body.erase("mfq_benchmark_prompt_tokens");
    body.erase("mfq_prefix_cache_enabled");
    const auto normal = parse_input(body, true, {});
    assert(normal.prefix_cache_enabled && !normal.benchmark_prompt_tokens);
    for (const auto& invalid : std::vector<json>{
            {{"mfq_benchmark_prompt_tokens", 0}, {"mfq_prefix_cache_enabled", false}, {"mfq_preformatted_prompt", "test"}},
            {{"mfq_benchmark_prompt_tokens", 32}, {"mfq_prefix_cache_enabled", true}, {"mfq_preformatted_prompt", "test"}},
            {{"mfq_benchmark_prompt_tokens", 32}, {"mfq_prefix_cache_enabled", false}, {"messages", json::array()}}}) {
        bool rejected = false;
        try { parse_input(invalid, true, {}); } catch (const ApiError&) { rejected = true; }
        assert(rejected);
    }
}
void test_probability_contract() {
    using namespace mfq::transport_detail;
    const auto request = parse_score_request({{"prompt", "Q:\nA:"}, {"continuations", {"A", "B"}}, {"mode", "next_token"}});
    assert(request.next_token && request.prompt == "Q:\nA:" && request.continuations.size() == 2);
    for (const auto& invalid : std::vector<json>{
            {{"prompt", ""}, {"continuations", {"A"}}},
            {{"prompt", "x"}, {"continuations", json::array()}},
            {{"prompt", "x"}, {"continuations", {1}}},
            {{"prompt", "x"}, {"continuations", {"A"}}, {"mode", 1}},
            {{"prompt", "x"}, {"continuations", {"A"}}, {"mode", "generate"}},
            {{"prompt", "x"}, {"continuations", {"A"}}, {"temperature", 1}}}) {
        bool rejected = false;
        try { (void)parse_score_request(invalid); } catch (const ApiError& error) { rejected = error.status == 400; }
        assert(rejected);
    }
    const auto result = likelihood_result_json({3, {{{1, 2}, {-1., -2.}, -3.}}});
    assert(result["log_base"] == "e" && result["scores"][0]["log_likelihood"] == -3);
}

void test_live_request_metrics() {
    using namespace mfq::transport_detail;
    RuntimeRequestMetrics store;
    {
        ActiveRequest first(store), second(store);
        first.progress("one", {{"generated_tokens", 6}});
        second.progress("two", {{"processed_tokens", 2048}});
        auto value = store.snapshot({}, 512, false);
        assert(value["live_requests"].size() == 2);
        assert(value["live_requests"]["one"]["generated_tokens"] == 6);
        first.complete("one", true, true, 10, {}, {});
        value = store.snapshot({}, 512, false);
        assert(value["live_requests"].size() == 1);
        assert(value["live_requests"].contains("two"));
    }
    assert(store.snapshot({}, 512, false)["live_requests"].empty());
    FakeEngine engine;
    MfqScheduler scheduler(engine);
    RequestWork work;
    work.sampling.max_tokens = 2;
    CompletionStream completion(scheduler, work, "0");
    while (completion.next()) {}
    assert(completion.progress()["generated_tokens"] == 6);
}

void test_request_schedule_contract() {
    using namespace mfq::transport_detail;
    const auto defaults = parse_schedule(json::object());
    assert(defaults.priority == 0 && !defaults.deadline);
    const auto explicit_nulls = parse_schedule({{"priority", nullptr}, {"timeout_seconds", nullptr}});
    assert(explicit_nulls.priority == 0 && !explicit_nulls.deadline);
    const auto scheduled = parse_schedule({{"priority", -3}, {"timeout_seconds", 1.5}});
    assert(scheduled.priority == -3 && scheduled.deadline);
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        *scheduled.deadline - Clock::now()).count();
    assert(remaining > 1000 && remaining <= 1600);
    for (const auto& invalid : std::vector<json>{
            {{"priority", "high"}}, {{"priority", 1.5}}, {{"priority", json(2147483648LL)}},
            {{"priority", json(-2147483649LL)}}, {{"timeout_seconds", 0}}, {{"timeout_seconds", -1}},
            {{"timeout_seconds", "1"}}, {{"timeout_seconds", 86401}}}) {
        bool rejected = false;
        try { (void)parse_schedule(invalid); }
        catch (const ApiError& error) { rejected = error.status == 400; }
        assert(rejected);
    }
    assert(parse_schedule({{"priority", json(2147483647LL)}}).priority == 2147483647);
}

void test_completion_stream_passes_schedule() {
    using namespace mfq::transport_detail;
    FakeEngine engine;
    MfqScheduler scheduler(engine);
    RequestWork work;
    work.sampling.max_tokens = 1;
    const auto schedule = parse_schedule({{"priority", 4}, {"timeout_seconds", 2.5}});
    CompletionStream completion(scheduler, work, "0", schedule);
    while (completion.next()) {}
    const auto& submitted = engine.admitted.at("0");
    assert(submitted.priority == 4);
    assert(submitted.deadline.has_value());
}

int main() {
    test_live_request_metrics();
    test_request_schedule_contract();
    test_completion_stream_passes_schedule();
    test_probability_contract();
    test_benchmark_input_contract();
    test_token_count_contract();
    test_decode_timing();
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine);
        auto input = request(0, 1); input.token_ids = {59};
        auto handle = scheduler.submit(input);
        while (!engine.blocked) std::this_thread::yield();
        const auto started = Clock::now();
        for (int index = 0; index < 1000; ++index) {
            const auto snapshot = scheduler.snapshot();
            assert(snapshot.status.healthy && snapshot.runtime == Metrics({{"released", 0}}));
        }
        assert(Clock::now() - started < 100ms);
        assert(engine.advances[0] == 0);
        engine.resume = true;
        drain(handle);
        // A terminal response must already expose post-cleanup resources.
        // Reading status must not require a later blocking control command.
        assert(scheduler.snapshot().runtime == Metrics({{"released", 1}}));
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine, {8, 4096});
        auto slow = scheduler.submit(request(0));
        auto fast = scheduler.submit(request(1));
        drain(fast);
        const auto paused = engine.advances[0].load();
        assert(paused > 0 && paused < 12);
        auto another = scheduler.submit(request(2, 3));
        drain(another);
        assert(engine.advances[0] == paused);
        drain(slow);
        auto blocked = scheduler.submit(request(3, 100));
        while (engine.advances[3] == 0) std::this_thread::yield();
        assert(scheduler.cancel_request("3"));
        assert(!scheduler.cancel_request("unknown"));
        drain(blocked, true);
        auto session = request(4, 100); session.input.cache_plan.session_id = "session";
        auto handle = scheduler.submit(session);
        assert(scheduler.session({SessionCommand::Kind::trim}).count == 7);
        assert(scheduler.session({SessionCommand::Kind::budget}).count == 7);
        for (const auto kind : {SessionCommand::Kind::clear, SessionCommand::Kind::memory_budget,
                SessionCommand::Kind::refresh}) {
            bool rejected = false;
            try { scheduler.session({kind}); } catch (const std::runtime_error&) { rejected = true; }
            assert(rejected);
        }
        session.id = "5";
        bool conflict = false;
        try { scheduler.submit(session); } catch (const std::invalid_argument&) { conflict = true; }
        assert(conflict);
        conflict = false;
        try { scheduler.session({SessionCommand::Kind::close, "session"}); } catch (const std::runtime_error&) { conflict = true; }
        assert(conflict);
        assert(scheduler.cancel_session("session"));
        drain(handle, true);
        assert(scheduler.session({SessionCommand::Kind::close, "session"}).count == 7);
        auto deadline = request(6, 100); deadline.deadline = Clock::now() + 15ms;
        auto expiring = scheduler.submit(deadline);
        std::this_thread::sleep_for(30ms);
        drain(expiring, false, true, "deadline_exceeded");
        auto active = scheduler.submit(request(7, 100));
        auto queued = scheduler.submit(request(8, 100));
        auto waiting = scheduler.submit(request(9, 100));
        assert(scheduler.cancel_request("9"));
        drain(waiting, true);
        assert(scheduler.reload(64) == 64 && engine.reloads == 1);
        drain(active, true); drain(queued, true);
        auto shutdown = scheduler.submit(request(10, 100));
        scheduler.shutdown();
        drain(shutdown, true);
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine);
        auto first = scheduler.submit(request(0, 100));
        auto bad = request(1); bad.token_ids = {63};
        auto second = scheduler.submit(bad);
        drain(first, false, true); drain(second, false, true);
        assert(!scheduler.status().healthy);
        assert(engine.active.empty());
    }
    {
        // An ordinary engine error fails the pending requests but keeps serving.
        FakeEngine engine;
        MfqScheduler scheduler(engine);
        auto bad = request(1); bad.token_ids = {58};
        drain(scheduler.submit(bad), false, true);
        assert(scheduler.status().healthy);
        drain(scheduler.submit(request(2, 100)));
        assert(engine.active.empty());
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine, {64, 4096}); // bytes, not event count, limits progress
        auto large = request(0, 100); large.token_ids = {61};
        auto slow = scheduler.submit(large);
        drain(scheduler.submit(request(1, 4)));
        const auto paused = engine.advances[0].load();
        assert(paused >= 2 && paused <= 3);
        drain(scheduler.submit(request(2, 4)));
        assert(engine.advances[0] == paused);
        scheduler.cancel_request("0");
        drain(slow, true); // reserved terminal space remains available
        auto oversized = request(3); oversized.token_ids = {62};
        auto bad = scheduler.submit(oversized);
        drain(scheduler.submit(request(4, 4)));
        drain(bad, false, true);
        assert(engine.released == 5);
        bool failed = false;
        try { scheduler.reload(13); } catch (const std::runtime_error&) { failed = true; }
        assert(failed);
        failed = false;
        try { scheduler.submit(request(5)); } catch (const std::runtime_error&) { failed = true; }
        assert(failed);
        assert(scheduler.reload(64) == 64);
        assert(scheduler.status().healthy);
        drain(scheduler.submit(request(6, 4)));
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine, {64, 8192});
        auto variable = request(0, 4); variable.token_ids = {60};
        auto slow = scheduler.submit(variable);
        drain(scheduler.submit(request(1, 4)));
        // Both deltas fit individually. Their sum pauses the producer instead
        // of reporting output_limit or losing the second delta.
        int outputs = 0, terminals = 0;
        while (!slow->done()) for (const auto& event : slow->wait()) {
            outputs += std::holds_alternative<OutputDelta>(event.data);
            if (terminal(event.data)) {
                assert(std::holds_alternative<Completed>(event.data));
                ++terminals;
            }
        }
        assert(outputs == 4 && terminals == 1);
    }
    {
        FakeEngine engine;
        MfqScheduler scheduler(engine, {64, 8192});
        auto variable = request(0, 2); variable.token_ids = {60};
        auto slow = scheduler.submit(variable);
        // The final delta and terminal arrive together, with the delta blocked.
        while (engine.released == 0) std::this_thread::yield();
        scheduler.shutdown();
        drain(slow); // shutdown must not wait for the consumer to drain
    }
}
