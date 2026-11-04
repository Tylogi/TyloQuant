#include "continuous_batch.h"
#include "request_executor.h"
#include "generation_policy.h"
#include "models/common/attention.h"
#include "models/common/causal_forward.h"
#include "models/common/causal_model.h"
#include "models/common/gated_mlp.h"
#include "models/common/grid_vision_model.h"
#include "models/common/moe.h"
#include "models/common/transformer_layer.h"
#include "models/deepseek_v4/causal_lm.h"
#include "models/deepseek_v41/causal_lm.h"
#include "models/deepseek_v41/engram.h"
#include "models/gemma4/causal_lm.h"
#include "models/glm5_next/causal_lm.h"
#include "models/glm_dsa/causal_lm.h"
#include "models/minicpmo45/causal_lm.h"
#include "models/nemotron_h/causal_lm.h"
#include "models/qwen35/causal_lm.h"
#include "models/qwen4_exp/causal_lm.h"
#include "sampling.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <map>

using namespace mfq::engine;

struct BatchOps {
    using Request = BatchRequest;
    using State = BatchState<Request>;
    struct Model {
        int vocab_size() const { return 100; }
        int max_position_embeddings() const { return 100; }
    } model_;
    int64_t vocab_size() const { return model_.vocab_size(); }
    int64_t max_context() const { return model_.max_position_embeddings(); }
    struct Sample {
        int64_t token;
        MfqPrefillTiming timing;
    };
    void suspend_decode() {}
    Sample prefill(const std::shared_ptr<Request> &request, PrefillChunk chunk) {
        assert(chunk.offset == request->prefill_offset && chunk.count <= 2);
        return {10, {request->prompt.size(), 1, 0, 1}};
    }
    void activate(const std::shared_ptr<Request> &, const Sample &) {}
    void resume_decode(int64_t) {}
    void discard_prefill(const std::shared_ptr<Request> &) {}
    void retire(const std::vector<std::shared_ptr<Request>> &, int64_t) {}
    int decode(State &) { return 0; }
    Sample sample(const std::shared_ptr<Request> &request, int) {
        return {request->pending_token + 1, {}};
    }
    void accept(const std::shared_ptr<Request> &, const Sample &) {}
    void recover(State &) {}
    Metrics metrics() const { return {}; }
};

template <class Physical> struct BatchExecution {
    ContinuousBatch<Physical> batch;
    template <class... Args> explicit BatchExecution(Args&&... args)
        : batch(8, 2, std::forward<Args>(args)...) {}
    bool can_batch(const EngineRequest&) const { return true; }
    bool exclusive() const { return false; }
    bool mtp_available() const { return false; }
    void execute(const std::vector<RequestId>& eligible) { batch.step(eligible); }
    Generation generate(const RequestId& id, ExecutionRequest& request) {
        return batch.generate(id, request.input, request.output);
    }
};

static EngineRequest batch_request(std::string id, int prompt = 5, int tokens = 3) {
    EngineRequest result;
    result.id = std::move(id);
    result.token_ids.assign(prompt, 1);
    result.input.sampling.max_tokens = tokens;
    return result;
}
static EngineInfo batch_info() { return {2, 100, 100}; }

void batch_test() {
    BatchExecution<BatchOps> ops;
    RequestExecutor executor(2);
    executor.admit(batch_request("a"), nullptr, batch_info(), ops);
    executor.admit(batch_request("b"), nullptr, batch_info(), ops);
    std::map<std::string, std::vector<int64_t>> tokens;
    std::map<std::string, int> terminals;
    const auto tick = [&](std::vector<RequestId> eligible) {
        auto result = executor.step(eligible, ops);
        for (const auto& event : result.events) {
            if (const auto* delta = std::get_if<OutputDelta>(&event.data))
                tokens[event.id].insert(tokens[event.id].end(), delta->token_ids.begin(), delta->token_ids.end());
            if (terminal(event.data)) ++terminals[event.id];
        }
    };
    for (int i = 0; i < 20 && tokens["a"].empty(); ++i) tick({"a"});
    assert(tokens["a"].size() == 1 && tokens["b"].empty());
    for (int i = 0; i < 20 && tokens["b"].empty(); ++i) tick({"b"});
    assert(tokens["a"].size() == 1 && tokens["b"].size() == 1);
    // Cancellation is serviced even when a request is excluded by backpressure.
    executor.cancel("a");
    for (int i = 0; i < 40 && !executor.empty(); ++i) tick({"b"});
    assert(executor.empty() && terminals["a"] == 1 && terminals["b"] == 1);
    assert(tokens["b"] == std::vector<int64_t>({10, 11, 12}));
    executor.admit(batch_request("c"), nullptr, batch_info(), ops);
    executor.cancel("c");
    tick({});
    assert(executor.empty() && terminals["c"] == 1 && tokens["c"].empty());
    // B=1 uses the same sequence, with no detached batch completion path.
    executor.admit(batch_request("single", 1, 1), nullptr, batch_info(), ops);
    for (int i = 0; i < 10 && !executor.empty(); ++i) tick({"single"});
    assert(executor.empty() && tokens["single"] == std::vector<int64_t>{10} && terminals["single"] == 1);
    for (const auto& [key, value] : ops.batch.metrics())
        if (key == "continuous_batching_active" || key == "continuous_batching_prefilling") assert(value == 0);
}

void packed_prefill_plain_sample_test() {
    struct PackedOps : BatchOps {
        std::vector<std::vector<std::string>>& groups;
        explicit PackedOps(std::vector<std::vector<std::string>>& value) : groups(value) {}
        std::vector<Sample> prefill_batch(
            const std::vector<std::pair<std::shared_ptr<Request>, PrefillChunk>>& chunks) {
            int64_t total = 0;
            std::vector<std::string> group;
            std::vector<Sample> result;
            for (const auto& [request, chunk] : chunks) {
                assert(chunk.offset == request->prefill_offset);
                total += chunk.count;
                group.push_back(request->id);
                result.push_back({10, {}});
            }
            assert(total <= 8);
            groups.push_back(std::move(group));
            return result;
        }
    };
    std::vector<std::vector<std::string>> groups;
    ContinuousBatch<PackedOps> batch(8, 8, groups);
    std::array<InferenceRequest, 3> inputs;
    std::vector<std::unique_ptr<InferenceOutput>> outputs;
    std::vector<Generation> generations;
    const std::array<int, 3> lengths{3, 4, 9};
    for (int row = 0; row < 3; ++row) {
        inputs[row].prompt.assign(lengths[row], 1);
        inputs[row].sampling.max_tokens = 2;
        outputs.push_back(std::make_unique<InferenceOutput>(inputs[row], nullptr, std::to_string(row)));
        generations.push_back(batch.generate(std::to_string(row), inputs[row], *outputs.back()));
    }
    std::array<bool, 3> done{};
    std::array<std::vector<int64_t>, 3> tokens;
    for (int tick = 0; tick < 40 && !std::all_of(done.begin(), done.end(), [](bool x) { return x; }); ++tick) {
        std::vector<std::string> eligible;
        for (int row = 0; row < 3; ++row) if (!done[row]) eligible.push_back(std::to_string(row));
        batch.step(eligible);
        for (int row = 0; row < 3; ++row) if (!done[row]) {
            auto step = generations[row].next();
            if (!step) { done[row] = true; continue; }
            if (step.value) if (auto* delta = std::get_if<OutputDelta>(&*step.value))
                tokens[row].insert(tokens[row].end(), delta->token_ids.begin(), delta->token_ids.end());
        }
    }
    assert(batch.empty() && groups.size() == 3);
    assert(groups[0] == std::vector<std::string>({"0", "1"}));
    assert(groups[1] == std::vector<std::string>{"2"} && groups[2] == std::vector<std::string>{"2"});
    for (const auto& row : tokens) assert(row == std::vector<int64_t>({10, 11}));
}

void staggered_decode_batch_test() {
    struct CoalescingOps : BatchOps {
        std::vector<int>& rows;
        explicit CoalescingOps(std::vector<int>& rows) : rows(rows) {}
        int decode(State& state) {
            rows.push_back(static_cast<int>(std::count_if(state.active.begin(),state.active.end(),
                [](const auto& request) { return request->eligible; })));
            return 0;
        }
    };
    std::vector<int> rows;
    BatchExecution<CoalescingOps> ops(rows);
    RequestExecutor executor(2);
    executor.admit(batch_request("short",1,12),nullptr,batch_info(),ops);
    executor.admit(batch_request("long",7,12),nullptr,batch_info(),ops);
    std::map<std::string,std::vector<int64_t>> tokens;
    const auto tick = [&](std::vector<RequestId> eligible) {
        auto result = executor.step(eligible,ops);
        assert(result.status.healthy);
        for (const auto& event : result.events)
            if (const auto* delta = std::get_if<OutputDelta>(&event.data))
                tokens[event.id].insert(tokens[event.id].end(),delta->token_ids.begin(),delta->token_ids.end());
    };
    for (int i = 0; i < 80 && tokens["long"].empty(); ++i) tick({"short","long"});
    assert(!tokens["short"].empty() && !tokens["long"].empty());
    // Backpressure on a peer must not hold an otherwise eligible request.
    const auto before = tokens["short"].size();
    for (int i = 0; i < 12 && tokens["short"].size() == before; ++i) tick({"short"});
    assert(tokens["short"].size() > before && tokens["long"].size() == 1);
    rows.clear();
    for (int i = 0; i < 160 && !executor.empty(); ++i) tick({"short","long"});
    assert(executor.empty());
    assert(std::count(rows.begin(),rows.end(),2) >= 6);
    for (const auto& id : {"short","long"}) {
        std::vector<int64_t> expected(12);
        std::iota(expected.begin(),expected.end(),10);
        assert(tokens[id] == expected);
    }
}

struct IsolatedPrefillOps : BatchOps {
    static constexpr bool isolate_prefill_failures = true;
    Sample prefill(const std::shared_ptr<Request>& request, PrefillChunk chunk) {
        if (request->id == "pressure" && chunk.offset > 0)
            throw InferenceInputError(InferenceInputErrorCode::ContextLength,
                                      "context growth exceeds available memory");
        return BatchOps::prefill(request,chunk);
    }
};
void isolated_prefill_failure_test() {
    BatchExecution<IsolatedPrefillOps> ops;
    RequestExecutor executor(2);
    std::map<std::string,std::vector<int64_t>> tokens;
    std::map<std::string,int> failures, terminals;
    const auto tick = [&](std::vector<RequestId> eligible) {
        auto result = executor.step(eligible,ops);
        assert(result.status.healthy);
        for (const auto& event : result.events) {
            if (const auto* delta = std::get_if<OutputDelta>(&event.data))
                tokens[event.id].insert(tokens[event.id].end(),delta->token_ids.begin(),delta->token_ids.end());
            failures[event.id] += std::holds_alternative<Failed>(event.data);
            terminals[event.id] += terminal(event.data);
        }
    };
    executor.admit(batch_request("survivor",2,5),nullptr,batch_info(),ops);
    for (int i = 0; i < 10 && tokens["survivor"].empty(); ++i) tick({"survivor"});
    assert(tokens["survivor"] == std::vector<int64_t>{10});
    executor.admit(batch_request("pressure",5,5),nullptr,batch_info(),ops);
    for (int i = 0; i < 40 && !executor.empty(); ++i) tick({"pressure","survivor"});
    assert(executor.empty() && failures["pressure"] == 1 && terminals["pressure"] == 1);
    assert(failures["survivor"] == 0 && terminals["survivor"] == 1 &&
        tokens["survivor"] == std::vector<int64_t>({10,11,12,13,14}));
    assert(executor.admit(batch_request("reuse",2,1),nullptr,batch_info(),ops) == Admission::accepted);
    for (int i = 0; i < 10 && !executor.empty(); ++i) tick({"reuse"});
    assert(executor.empty() && tokens["reuse"] == std::vector<int64_t>{10});
}

struct PackedPrefillLog {
    int fault = 0;
    std::vector<std::vector<std::pair<std::string,PrefillChunk>>> groups;
};
struct PackedPrefillOps : BatchOps {
    static constexpr bool isolate_prefill_failures = true;
    PackedPrefillLog& log;
    explicit PackedPrefillOps(PackedPrefillLog& value):log(value) {}
    std::size_t prefill_batch_limit() const {return 3;}
    std::vector<PrefillBatchResult<Sample>> prefill_batch(
            const std::vector<std::pair<std::shared_ptr<Request>,PrefillChunk>>& items) {
        std::vector<std::pair<std::string,PrefillChunk>> group;
        int64_t tokens=0;
        for (const auto& [request,chunk]:items) {
            tokens+=chunk.count;group.emplace_back(request->id,chunk);
        }
        assert(tokens<=4 && items.size()<=3);log.groups.push_back(std::move(group));
        std::vector<PrefillBatchResult<Sample>> result;
        for (const auto& [request,chunk]:items) {
            if (log.fault && request->id=="pressure" && chunk.offset>0) {
                auto error=std::make_exception_ptr(InferenceInputError(
                    InferenceInputErrorCode::ContextLength,"prefill capacity exhausted"));
                if (log.fault==2)std::rethrow_exception(error);
                result.push_back({{},error});
            } else result.push_back({BatchOps::prefill(request,chunk),{}});
        }
        return result;
    }
};
void packed_prefill_test() {
    struct Execution {
        ContinuousBatch<PackedPrefillOps> batch;
        explicit Execution(PackedPrefillLog& log):batch(2,4,log) {}
        bool can_batch(const EngineRequest&) const {return true;}
        bool exclusive() const {return false;}
        bool mtp_available() const {return false;}
        void execute(const std::vector<RequestId>& ids) {batch.step(ids);}
        Generation generate(const RequestId& id,ExecutionRequest& request) {
            return batch.generate(id,request.input,request.output);
        }
    };
    for (int fault:{0,1,2}) {
        PackedPrefillLog log{fault};Execution ops(log);RequestExecutor executor(3);
        std::map<std::string,std::vector<int64_t>> tokens;
        std::map<std::string,int> failures,terminals;
        const auto tick=[&](std::vector<RequestId> ids) {
            auto result=executor.step(ids,ops);assert(result.status.healthy);
            for (const auto& event:result.events) {
                if (const auto* delta=std::get_if<OutputDelta>(&event.data))
                    tokens[event.id].insert(tokens[event.id].end(),delta->token_ids.begin(),delta->token_ids.end());
                failures[event.id]+=std::holds_alternative<Failed>(event.data);
                terminals[event.id]+=terminal(event.data);
            }
        };
        EngineInfo info{3,100,100};
        executor.admit(batch_request("survivor",2,8),nullptr,info,ops);
        for(int i=0;i<12 && tokens["survivor"].empty();++i)tick({"survivor"});
        assert(tokens["survivor"]==std::vector<int64_t>{10});
        executor.admit(batch_request("pressure",5,3),nullptr,info,ops);
        executor.admit(batch_request("peer",3,3),nullptr,info,ops);
        // Register both coroutines while the established row is paused.
        for(int i=0;i<80 && !executor.empty();++i)tick({"pressure","peer","survivor"});
        assert(executor.empty() && !log.groups.empty());
        assert(tokens["survivor"]==std::vector<int64_t>({10,11,12,13,14,15,16,17}));
        assert(failures["survivor"]==0 && failures["pressure"]==(fault>0) && failures["peer"]==(fault==2));
        if(!fault)assert(tokens["pressure"]==std::vector<int64_t>({10,11,12}));
        if(fault!=2)assert(tokens["peer"]==std::vector<int64_t>({10,11,12}));
        for(const auto& id:{"survivor","pressure","peer"})assert(terminals[id]==1);
    }
}

struct IsolatedGrowthOps : BatchOps {
    static constexpr bool isolate_context_growth_failures = true;
    int decode(State& state) {
        for (const auto& request : state.active)
            if (request->eligible && request->id == "pressure")
                throw InferenceInputError(InferenceInputErrorCode::ContextLength,"KV growth exceeds available memory");
        return 0;
    }
};
void isolated_context_growth_test() {
    BatchExecution<IsolatedGrowthOps> ops;
    RequestExecutor executor(2);
    std::map<std::string,std::vector<int64_t>> tokens;
    std::map<std::string,int> failures;
    const auto tick = [&](std::vector<RequestId> eligible) {
        auto result = executor.step(eligible,ops);
        assert(result.status.healthy);
        for (const auto& event : result.events) {
            if (const auto* delta = std::get_if<OutputDelta>(&event.data))
                tokens[event.id].insert(tokens[event.id].end(),delta->token_ids.begin(),delta->token_ids.end());
            failures[event.id] += std::holds_alternative<Failed>(event.data);
        }
    };
    executor.admit(batch_request("survivor",2,5),nullptr,batch_info(),ops);
    executor.admit(batch_request("pressure",2,5),nullptr,batch_info(),ops);
    for (const auto& id : {"survivor","pressure"})
        for (int i = 0; i < 10 && tokens[id].empty(); ++i) tick({id});
    for (int i = 0; i < 40 && !executor.empty(); ++i) tick({"survivor","pressure"});
    assert(executor.empty() && failures["pressure"] == 1 && failures["survivor"] == 0);
    assert(tokens["survivor"] == std::vector<int64_t>({10,11,12,13,14}) &&
           tokens["pressure"] == std::vector<int64_t>{10});
}

void batch_failure_test() {
    enum Stage { suspend, prefill, activate, resume };
    struct Fault { Stage stage; bool backend = false; bool armed = false; size_t recovered = 0; };
    struct FailingOps : BatchOps {
        Fault& fault;
        explicit FailingOps(Fault& state) : fault(state) {}
        void check(Stage stage) {
            if (!fault.armed || fault.stage != stage) return;
            if (fault.backend) throw MfqBackendFailure("device failure");
            throw std::runtime_error("batch step failure");
        }
        void suspend_decode() { check(suspend); }
        Sample prefill(const std::shared_ptr<Request>& request, PrefillChunk chunk) {
            check(Stage::prefill); return BatchOps::prefill(request, chunk);
        }
        void activate(const std::shared_ptr<Request>&, const Sample&) { check(Stage::activate); }
        void resume_decode(int64_t) { check(resume); }
        void recover(State& state) { fault.recovered = state.prefilling.size(); }
    };
    for (auto stage : {suspend, prefill, activate, resume}) for (bool backend : {false, true}) {
        Fault fault{stage, backend};
        BatchExecution<FailingOps> ops(fault);
        RequestExecutor executor(2);
        executor.admit(batch_request("a", 2, 20), nullptr, batch_info(), ops);
        executor.admit(batch_request("b", 2, 20), nullptr, batch_info(), ops);
        // Register both physical rows before the device fault.
        executor.step({"a", "b"}, ops);
        fault.armed = true;
        const auto result = executor.step({"a"}, ops);
        // The batch recovered both rows. A lost backend fails them now and
        // closes admission; otherwise the paused row fails on its next step.
        assert(result.status.healthy == !backend && fault.recovered == 2);
        auto events = result.events;
        fault.armed = false;
        if (!backend) {
            assert(events.size() == 1 && events.front().id == "a");
            const auto paused = executor.step({"b"}, ops);
            events.insert(events.end(), paused.events.begin(), paused.events.end());
        }
        assert(executor.empty() && events.size() == 2);
        for (const auto& event : events) assert(std::holds_alternative<Failed>(event.data));
        for (const auto& [key, value] : ops.batch.metrics())
            if (key == "continuous_batching_active" || key == "continuous_batching_prefilling") assert(value == 0);
    }
    // Cleanup must fail the request even if a backend reports invalid_argument;
    // neither normal teardown nor forced destruction may retry the release.
    struct CleanupFault { int releases = 0, recoveries = 0; bool recovery_fails; };
    struct ReleaseOps : BatchOps {
        CleanupFault& fault;
        explicit ReleaseOps(CleanupFault& value) : fault(value) {}
        void fail() { ++fault.releases; throw std::invalid_argument("slot release failed"); }
        void retire(const std::vector<std::shared_ptr<Request>>&, int64_t) { fail(); }
        void discard_prefill(const std::shared_ptr<Request>&) { fail(); }
        void recover(State& state) {
            ++fault.recoveries;
            assert(state.prefilling.size() == 1);
            if (fault.recovery_fails) throw std::runtime_error("recovery failed");
        }
    };
    for (bool active : {false, true}) for (bool recovery_fails : {false, true}) {
        CleanupFault fault{0, 0, recovery_fails};
        BatchExecution<ReleaseOps> ops(fault);
        RequestExecutor executor;
        executor.admit(batch_request("a", active ? 1 : 5, 1), nullptr, batch_info(), ops);
        EngineStepResult result;
        if (!active) {
            executor.step({"a"}, ops); // A pending prefill is owned but has no slot.
            executor.cancel("a");
        }
        for (int tick = 0; tick < 10 && !executor.empty(); ++tick) result = executor.step({"a"}, ops);
        assert(executor.empty() && !result.status.healthy && fault.releases == 1 && fault.recoveries == 1);
        assert(std::count_if(result.events.begin(), result.events.end(), [](const auto& event) {
            return std::holds_alternative<Failed>(event.data);
        }) == 1);
        assert(std::get<Failed>(result.events.back().data).code == "backend_failure");
        for (const auto& [key, value] : ops.batch.metrics())
            if (key == "continuous_batching_active" || key == "continuous_batching_prefilling") assert(value == 0);
    }
    for (const auto limits : {std::pair{0, 2}, std::pair{2, 0}}) {
        try { ContinuousBatch<BatchOps> batch(limits.first, limits.second); assert(false); }
        catch (const std::invalid_argument&) {}
    }
}

void batch_recoverable_failure_test() {
    struct CapacityError : MfqResourceExhausted { using MfqResourceExhausted::MfqResourceExhausted; };
    struct LimitedOps : BatchOps {
        Sample prefill(const std::shared_ptr<Request>& request, PrefillChunk chunk) {
            if (request->id == "too-large") throw CapacityError("request memory limit");
            return BatchOps::prefill(request, chunk);
        }
        bool recoverable_failure(std::exception_ptr error) const {
            try { std::rethrow_exception(error); }
            catch (const CapacityError&) { return true; }
            catch (...) { return false; }
        }
    };
    BatchExecution<LimitedOps> ops;
    RequestExecutor executor(2);
    executor.admit(batch_request("peer"), nullptr, batch_info(), ops);
    executor.admit(batch_request("too-large"), nullptr, batch_info(), ops);
    int failed = 0, completed = 0;
    std::vector<int64_t> tokens;
    for (int tick = 0; tick < 60 && !executor.empty(); ++tick) {
        auto result = executor.step({"too-large", "peer"}, ops);
        assert(result.status.healthy);
        for (const auto& event : result.events) {
            if (std::holds_alternative<Failed>(event.data)) {
                assert(event.id == "too-large");
                assert(std::get<Failed>(event.data).code == "resource_exhausted" && std::get<Failed>(event.data).retryable);
                ++failed;
            }
            if (std::holds_alternative<Completed>(event.data)) {
                assert(event.id == "peer");
                ++completed;
            }
            if (auto* delta = std::get_if<OutputDelta>(&event.data))
                tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
        }
    }
    assert(executor.empty() && ops.batch.empty() && failed == 1 && completed == 1);
    assert(tokens == std::vector<int64_t>({10, 11, 12}));
    assert(executor.admit(batch_request("replacement", 1, 1), nullptr, batch_info(), ops) == Admission::accepted);
}

void batch_alignment_test() {
    struct AlignedOps : BatchOps {
        std::size_t& maximum;
        explicit AlignedOps(std::size_t& value) : maximum(value) {}
        int decode(State& state) {
            maximum = std::max(maximum, static_cast<std::size_t>(std::count_if(
                state.active.begin(), state.active.end(), [](const auto& row) { return row->eligible; })));
            return 0;
        }
    };
    std::size_t maximum = 0;
    BatchExecution<AlignedOps> ops(maximum);
    RequestExecutor executor(2);
    executor.admit(batch_request("a", 1, 12), nullptr, batch_info(), ops);
    executor.admit(batch_request("b", 1, 12), nullptr, batch_info(), ops);
    for (int tick = 0; tick < 80 && !executor.empty(); ++tick) executor.step({"a", "b"}, ops);
    assert(executor.empty() && maximum == 2);
}

struct Block {
    int kept = -1, begun = 0, committed = 0;
    bool fail = false;
    void reset(int) { kept = 0; }
    void begin_speculative(int) {
        if (fail)
            throw std::runtime_error("begin failed");
        ++begun;
    }
    void commit_speculative() { ++committed; }
    void rollback_speculative(int64_t keep) { kept = keep; }
};
struct Model {
    int64_t cache_pos = 10, decode_position_delta = 2, speculative_start = -1,
            speculative_confirmed = 0;
    bool speculative_suffix_forward = false;
    std::vector<std::unique_ptr<Block>> blocks;
    bool supports_suffix_speculation() const { return true; }
    int max_position_embeddings() const { return 100; }
    void adapter_begin_speculative() {}
    void adapter_commit_speculative() {}
    void adapter_rollback_speculative(int64_t) {}
    void adapter_reset(int64_t) {}
};
void model_test() {
    using namespace mfq::models;
    Model model;
    for (int i = 0; i < 2; ++i)
        model.blocks.push_back(std::make_unique<Block>());
    auto guard = [](const auto &) { return 0; };
    begin_speculative_suffix(model, 4, guard);
    model.cache_pos = 14;
    finish_speculative(model, false, 2, guard);
    assert(model.cache_pos == 12 && model.speculative_start == -1);
    for (const auto &block : model.blocks)
        assert(block->kept == 12);
    model.blocks.back()->fail = true;
    try {
        begin_speculative_suffix(model, 3, guard);
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(model.speculative_start == -1 && model.blocks.front()->committed == 1);
    reset_model(model, 1, guard);
    assert(model.cache_pos == 0 && model.decode_position_delta == 0);

    // Noncommutative scalar operators detect residual/norm ordering changes.
    auto fused = [](int normalized, int residual) -> std::optional<int> {
        return residual + normalized * 3;
    };
    auto fallback = [](int, int) { return std::optional<int>{}; };
    auto ffn = [](int x) { return x * 3; };
    auto add = [](int a, int b) { return a + b; };
    assert(feed_forward(7, 2, fused, ffn, add) == feed_forward(7, 2, fallback, ffn, add));
    auto result = attention_layer(
        5, [](int x) { return x - 2; }, [](int x) { return x * 4; },
        [&](int residual, int attention) {
            return residual_ffn(
                residual, attention,
                [](int a, int b) { return std::array<int, 2>{a + b, (a + b) / 2}; },
                [&](int r, int n) { return feed_forward(r, n, fallback, ffn, add); });
        });
    assert(result == 41);
    int commits = 0;
    assert(hyperconnection_layer(
               2, [](int x) { return x + 1; }, [](int x) { return x * 2; },
               [](int x, int residual, int) { return x + residual; },
               [](int x, int previous) { return x + previous; }, [](int x) { return x * 3; },
               [](int x, int residual, int) { return x + residual; },
               [&](int mix) {
                   assert(mix == 11);
                   ++commits;
               }) == 41);
    assert(commits == 1);
    int expert_calls = 0;
    auto experts = [&] {
        ++expert_calls;
        return 10;
    };
    auto dense = [] { return 2; };
    auto post = [](int x, gemma4::FfnNorm) { return x * 2; };
    auto scale = [](int x) { return x * 2; };
    auto fused_gemma = [](int d, int e, int r) { return ((d * 2 + e * 2) * 2 + r) * 2; };
    assert(gemma4::feed_forward(false, true, false, 1, dense, experts, post, add, add, scale,
                                fused_gemma) == 10 &&
           expert_calls == 0);
    for (bool fusion : {false, true})
        assert(gemma4::feed_forward(true, true, fusion, 1, dense, experts, post, add, add, scale,
                                    fused_gemma) == 98);
    assert(expert_calls == 2);
    assert(pre_norm_layer(
               4, [](int x, int stage) { return x - stage; }, [](int x) { return x * 2; }, ffn,
               add) == 45);
}
void inner_model_flow_test() {
    using namespace mfq::models;
    using Heads = AttentionHeads<int>;
    // All three native fusion choices must implement the same attention graph.
    for (int fusion = 0; fusion < 3; ++fusion) {
        std::string order;
        int cached = 0;
        const auto result = full_attention(
            2,
            [&](int x) {
                order += 'P';
                return Heads{x, 3, 5, 7};
            },
            [&](Heads &h) {
                if (fusion != 2)
                    return false;
                order += 'F';
                cached = h.query = h.query * 2 + 3;
                return true;
            },
            [&](Heads &h) {
                order += 'N';
                h.query *= 2;
            },
            [&](Heads &h) {
                if (fusion != 1)
                    return false;
                order += 'F';
                cached = h.query += 3;
                return true;
            },
            [&](Heads &h) {
                order += 'R';
                h.query += 3;
            },
            [&](const Heads &h) {
                order += 'C';
                cached = h.query;
            },
            [&](const Heads &h) {
                order += 'A';
                assert(cached == h.query);
                return h.query * h.key + h.value;
            },
            [&](int h, std::optional<int> gate) {
                order += 'O';
                return h * gate.value();
            });
        assert(result == 182 && cached == 7);
        assert(order == (fusion == 0 ? "PNRCAO" : fusion == 1 ? "PNFAO" : "PFAO"));
    }
    std::string order;
    auto result = qwen35::linear_attention(
        2,
        [&](int x) {
            order += 'P';
            return qwen35::LinearProjections<int>{x, 3, 4, 5, 6, 7};
        },
        [&](const auto &p) {
            order += 'G';
            return std::array<int, 2>{p.alpha + 1, p.beta - 1};
        },
        [&](const auto &p) {
            order += 'C';
            return std::array<int, 3>{p.qkv + 1, p.qk * 2, p.value - 1};
        },
        [&](auto qkv, auto gates) {
            order += 'R';
            return qkv[0] * gates[0] + qkv[1] * gates[1] + qkv[2];
        },
        [&](int h) {
            order += 'N';
            return h - 2;
        },
        [&](int h, int gate) {
            order += 'O';
            return h * gate;
        });
    assert(result == 290 && order == "PGCRNO");

    for (bool gelu : {false, true})
        for (double limit : {0.0, 2.0}) {
            for (int fusion = 0; fusion < 4; ++fusion) {
                const double effective_limit = gelu ? 0.0 : limit;
                auto activate = [](double gate, double up, GatedActivation mode, double limit) {
                    return (gate + (mode == GatedActivation::gelu ? 10 : 1) + limit) * up;
                };
                auto optional = [](bool selected, double value) {
                    return selected ? std::optional<double>(value) : std::nullopt;
                };
                const double hidden = (3 + (gelu ? 10 : 1) + effective_limit) * 4;
                int down_calls = 0;
                auto output = gated_mlp(
                    2.0, gelu, limit,
                    [&](double x, auto a, double l) {
                        return optional(fusion == 1, activate(x + 1, x * 2, a, l) - 3);
                    },
                    [&](double x, auto a, double l) {
                        return optional(fusion == 2, activate(x + 1, x * 2, a, l));
                    },
                    [](double x) { return std::array<double, 2>{x + 1, x * 2}; }, activate,
                    [&](double h) {
                        ++down_calls;
                        return h - 3;
                    },
                    [&](double g, double u, auto a, double l) {
                        return optional(fusion == 3, activate(g, u, a, l) - 3);
                    });
                assert(output == hidden - 3 && down_calls == (fusion == 0 || fusion == 2));
            }
        }
}

struct PredictorOps {
    struct Tensor {
        std::vector<int64_t> shape;
        int value = 0;
    };
    struct Model {
        struct Config {
            int hidden_size = 4, max_position_embeddings = 8;
            std::vector<int> mrope_sections;
        } config;
        int64_t cache_pos = 0;
        std::array<int, 2> blocks{1, 2};
    } model;
    std::vector<int> visited;
    std::optional<Tensor> seen_lengths, seen_cache;
    static int rank(const Tensor &t) { return t.shape.size(); }
    static int64_t size(const Tensor &t, int axis) {
        return t.shape.at(axis < 0 ? t.shape.size() + axis : axis);
    }
    static int64_t elements(const Tensor &t) {
        int64_t count = 1;
        for (auto n : t.shape)
            count *= n;
        return count;
    }
    static bool defined(const Tensor &t) { return !t.shape.empty(); }
    static Tensor embed(Tensor ids, const Tensor &like) { return {like.shape, ids.value + 2}; }
    static Tensor normalize(Tensor t, mfq::models::qwen35::PredictorNorm role) {
        t.value += static_cast<int>(role) + 1;
        return t;
    }
    static void trace(const char *, const Tensor &) {}
    static Tensor fuse(Tensor e, Tensor h) {
        h.value += e.value * 10;
        return h;
    }
    static Tensor positions(Tensor t, int64_t start, int64_t tokens) {
        return defined(t) ? t : Tensor{{tokens}, static_cast<int>(start)};
    }
    static Tensor lengths(int64_t batch, int64_t end, const Tensor &) {
        return {{batch}, static_cast<int>(end)};
    }
    static Tensor cache_positions(const Tensor &, int64_t start, int64_t tokens) {
        return {{tokens}, static_cast<int>(start)};
    }
    Tensor layer(int block, Tensor h, const Tensor &, const std::optional<Tensor> &lengths,
                 const std::optional<Tensor> &cache) {
        visited.push_back(block);
        seen_lengths = lengths;
        seen_cache = cache;
        h.value = h.value * 2 + block;
        return h;
    }
};

void predictor_flow_test() {
    using mfq::models::qwen35::mtp_predictor;
    PredictorOps ops;
    auto run = [&](int64_t tokens, PredictorOps::Tensor pos = {}) {
        return mtp_predictor(ops, {{1, tokens, 4}, 3}, {{1, tokens}, 5}, pos);
    };
    assert(run(2).value == 347 && ops.model.cache_pos == 2);
    assert((ops.visited == std::vector<int>{1, 2}));
    assert(!ops.seen_lengths && !ops.seen_cache);
    assert(run(1, {{1}, 90}).value == 347 && ops.model.cache_pos == 3);
    assert(ops.seen_lengths->value == 3 && ops.seen_cache->value == 2);
    ops.model.config.mrope_sections = {1, 1, 1};
    run(1, {{3, 1}, 20});
    assert(ops.model.cache_pos == 4 && ops.seen_cache->value == 3);
    const auto layer_calls = ops.visited.size();
    for (int invalid = 0; invalid < 3; ++invalid) {
        try {
            if (invalid == 0)
                run(5); // Context overflow.
            if (invalid == 1)
                run(1, {{2, 1}, 0}); // Invalid position shape.
            if (invalid == 2)
                mtp_predictor(ops, {{1, 1, 4}, 3}, {{2, 1}, 5}, {});
            assert(false);
        } catch (const std::runtime_error &) {
        }
        assert(ops.model.cache_pos == 4 && ops.visited.size() == layer_calls);
    }
}

// Nemotron-H runs one MTP stage made of every declared sub-block, and it
// declares its own history capacity instead of the family config field.
struct NemotronPredictorOps {
    struct Tensor {
        std::vector<int64_t> shape;
        int value = 0;
    };
    struct Model {
        struct Config {
            int hidden_size = 4;
            std::int64_t num_nextn_predict_layers = 1;
            std::vector<mfq::models::nemotron_h::MixerKind> mtp_mixers = {
                mfq::models::nemotron_h::MixerKind::full_attention,
                mfq::models::nemotron_h::MixerKind::moe};
        } config;
        // Smaller than predictor_capacity(): the shared fallback must not win.
        int64_t maximum = 2;
        std::vector<int64_t> lengths{0};
        int64_t batch = 0;
        int64_t predictor_capacity() const noexcept { return 6; }
        void reset(int64_t next) {
            batch = next;
            std::fill(lengths.begin(), lengths.end(), 0);
        }
    } model;
    std::vector<int> stages;
    static int rank(const Tensor &t) { return t.shape.size(); }
    static int64_t size(const Tensor &t, int axis) {
        return t.shape.at(axis < 0 ? t.shape.size() + axis : axis);
    }
    static int64_t elements(const Tensor &t) {
        int64_t count = 1;
        for (auto n : t.shape)
            count *= n;
        return count;
    }
    static bool defined(const Tensor &t) { return !t.shape.empty(); }
    static Tensor embed(Tensor ids, const Tensor &like) { return {like.shape, ids.value + 2}; }
    static Tensor normalize(Tensor t, mfq::models::nemotron_h::PredictorNorm role) {
        t.value += static_cast<int>(role) + 1;
        return t;
    }
    static Tensor positions(Tensor t, int64_t start, int64_t tokens) {
        return defined(t) ? t : Tensor{{tokens}, static_cast<int>(start)};
    }
    static Tensor fuse(Tensor e, Tensor h) {
        h.value += e.value * 10;
        return h;
    }
    Tensor mixer_blocks(Tensor h, int64_t stage, const Tensor &) {
        stages.push_back(static_cast<int>(stage));
        for (std::size_t block = 0; block < model.config.mtp_mixers.size(); ++block)
            h.value = h.value * 3 + static_cast<int>(stage) + static_cast<int>(block);
        return h;
    }
};

void nemotron_predictor_flow_test() {
    using mfq::models::nemotron_h::mtp_predictor;
    NemotronPredictorOps ops;
    auto run = [&](int64_t tokens) {
        return mtp_predictor(ops, {{1, tokens, 4}, 3}, {{1, tokens}, 5}, {}, 0, true);
    };
    // e=7, h=5, fusion=85, then both declared sub-blocks, then the output norm.
    assert(run(2).first.value == 769 && ops.model.lengths.front() == 2);
    // Both declared sub-blocks run inside the single predictor stage.
    assert((ops.stages == std::vector<int>{0}));
    // The declared capacity (6) allows more history than config.maximum (2).
    assert(run(4).first.value == 769 && ops.model.lengths.front() == 6);
    const auto stages = ops.stages.size();
    assert((ops.stages == std::vector<int>{0, 0}));
    try {
        run(1); // Context overflow at the declared capacity.
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(ops.model.lengths.front() == 6 && ops.stages.size() == stages);
}

// A non-CUDA tensor exercises the actual shared CausalLm class end to end.
struct CausalTestOps {
    using Tensor = PredictorOps::Tensor;
    struct ForwardPlan {
        int tag = 0;
    };
    struct SessionState {
        int64_t cache_pos = 0, decode_position_delta = 0;
    };
    enum class SessionStateKind { Unsupported, Full };
    struct SessionCodec {
        template <class Model> static auto kind(const Model &) { return SessionStateKind::Full; }
        template <class Model> static bool supports_paged(const Model &) { return true; }
        template <class Model>
        static SessionState capture(const Model &m, const std::vector<int64_t> &) {
            return {m.cache_pos, 0};
        }
        template <class Model> static void restore(Model &m, const SessionState &s) {
            m.cache_pos = s.cache_pos;
        }
    };
    struct Block {
        int id;
        int64_t kept = -1;
        bool supports_speculation() const { return true; }
        void reset(int64_t) {}
        void begin_speculative(int64_t) {}
        void commit_speculative() {}
        void rollback_speculative(int64_t value) { kept = value; }
    };
    mfq::models::CausalLmMetadata metadata;
    std::vector<std::unique_ptr<Block>> blocks;
    Tensor output_norm;
    int embed = 0, lm_head = 0;
    bool fail_layer = false;
    int seen_position = -1, seen_cache = -1, seen_plan = -1;
    std::optional<Tensor> seen_lengths;
    CausalTestOps() {
        metadata.hidden_size = 4;
        metadata.max_position_embeddings = 32;
        blocks.push_back(std::make_unique<Block>(Block{1}));
        blocks.push_back(std::make_unique<Block>(Block{2}));
    }
    static bool mask_all_ones(const Tensor &value) { return value.value == 1; }
    static int execution_scope() { return 0; }
    static int block_scope(const std::unique_ptr<Block> &) { return 0; }
    static int64_t rank(const Tensor &t) { return t.shape.size(); }
    static int64_t size(const Tensor &t, int axis) {
        return t.shape.at(axis < 0 ? t.shape.size() + axis : axis);
    }
    static Tensor batch_ids(Tensor t) {
        t.shape.insert(t.shape.begin(), 1);
        return t;
    }
    static Tensor device_ids(Tensor t) { return t; }
    static Tensor embed_tokens(Tensor t) {
        t.shape.push_back(4);
        t.value += 4;
        return t;
    }
    static Tensor sequence_lengths(int64_t batch, int64_t end) { return {{batch}, int(end)}; }
    static Tensor last_hidden(Tensor t) {
        t.shape.erase(t.shape.begin() + 1);
        return t;
    }
    static Tensor softcap(Tensor t, double) {
        t.value = -t.value;
        return t;
    }
    static Tensor adapter_finalize_hidden(Tensor t, const Tensor &, int64_t, int64_t) {
        t.value += 3;
        return t;
    }
    static Tensor adapter_logits(int, Tensor t) {
        t.value *= 5;
        return t;
    }
    static Tensor adapter_last_logits(int, Tensor t) {
        t.value *= 7;
        return t;
    }
    static Tensor adapter_next_token(int, Tensor t) {
        t.value %= 11;
        return t;
    }
    static void adapter_reset(int64_t) {}
    static bool adapter_requires_batch_reset(int64_t) { return false; }
    static void adapter_begin_forward(bool) {}
    static void adapter_finish_forward(const Tensor &, int64_t, int64_t) {}
    static void adapter_begin_speculative() {}
    static void adapter_commit_speculative() {}
    static void adapter_rollback_speculative(int64_t) {}
    template <class Model> static auto forward_ops(Model &model, typename Model::Inputs input) {
        struct Ops : Model::Inputs {
            Model &model;
            Tensor pos, full_positions, cache_positions;
            std::optional<Tensor> effective_attention_mask;
            Ops(Model &m, typename Model::Inputs i) : Model::Inputs(std::move(i)), model(m) {}
            static int64_t rank(const Tensor &t) { return CausalTestOps::rank(t); }
            static int64_t size(const Tensor &t, int axis) { return CausalTestOps::size(t, axis); }
            static int64_t elements(const Tensor &t) { return PredictorOps::elements(t); }
            static Tensor device_ids(Tensor t) { return t; }
            static Tensor position_range(int64_t start, int64_t n) { return {{n}, int(start)}; }
            static Tensor offset_positions(Tensor t, int64_t delta) {
                t.value += delta;
                return t;
            }
            static bool has_mrope() { return false; }
            void prepare_inputs() {}
            Tensor prepare_hidden(int64_t, int64_t) {
                model.seen_position = pos.value;
                model.seen_cache = cache_positions.value;
                model.seen_plan = this->plan.tag;
                model.seen_lengths = this->seq_len;
                return this->input_embeddings;
            }
            Tensor layer(const std::unique_ptr<Block> &block, Tensor t) {
                if (model.fail_layer)
                    throw std::runtime_error("injected layer failure");
                t.value = t.value * 2 + block->id;
                return t;
            }
            void trace(const Tensor &t) {
                if (this->block_trace)
                    this->block_trace->push_back(t);
            }
            Tensor finish(Tensor t, int64_t b, int64_t n) {
                if (this->raw_hidden)
                    *this->raw_hidden = t;
                return model.finalize_hidden(t, b, n);
            }
        };
        return Ops(model, std::move(input));
    }
};

void causal_lm_test() {
    using Tensor = CausalTestOps::Tensor;
    struct TestModel : mfq::models::CausalModelBase<CausalTestOps, TestModel> {
        bool adapter_supports_speculation() const { return true; }
        bool adapter_supports_suffix_speculation() const { return true; }
    } model;
    Tensor ids{{1, 2}, 2}, raw;
    std::vector<Tensor> trace;
    auto hidden = model.hidden_forward(ids, {}, {}, &trace, {}, &raw);
    assert(hidden.value == 31 && raw.value == 28 && model.cache_pos == 2);
    assert(trace.size() == 3 && trace[0].value == 6 && trace[1].value == 13 &&
           trace[2].value == 28);
    assert(model.forward(Tensor{{1}, 2}).value == 155 && model.cache_pos == 3);
    assert(model.last_logits(Tensor{{1}, 2}).value == 217 && model.cache_pos == 4);
    assert(model.seen_lengths->value == 4);
    assert(model.next_token(Tensor{{1}, 2}).value == 9 && model.cache_pos == 5);
    model.decode_position_delta = 10;
    model.hidden_forward(Tensor{{1}, 2});
    assert(model.seen_position == 15 && model.seen_cache == 5 && model.cache_pos == 6);
    model.next_token_static(Tensor{{1}, 2}, Tensor{{1}, 20}, Tensor{{1}, 21}, {17});
    assert(model.seen_position == 20 && model.seen_cache == 20 && model.seen_plan == 17 &&
           model.cache_pos == 6);
    auto state = model.capture_text_session_state({1, 2, 3, 4, 5, 6});
    model.reset(1);
    assert(model.cache_pos == 0 && model.decode_position_delta == 0);
    model.restore_text_session_state(state);
    assert(model.cache_pos == 6 && model.decode_position_delta == 10);
    model.begin_speculative_suffix(2);
    model.hidden_forward_speculative_suffix(Tensor{{2}, 2});
    model.rollback_speculative(1);
    assert(model.cache_pos == 7 && model.blocks[0]->kept == 7 && !model.speculative_suffix_forward);
    model.begin_speculative_suffix(1);
    model.fail_layer = true;
    try {
        model.hidden_forward_speculative_suffix(Tensor{{1}, 2});
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(!model.speculative_suffix_forward && model.cache_pos == 7);
    model.fail_layer = false;
    model.rollback_speculative();
    for (Tensor invalid : {Tensor{{}, 0}, Tensor{{1, 1, 1}, 0}, Tensor{{0}, 0}}) {
        try {
            model.hidden_forward(invalid);
            assert(false);
        } catch (const std::runtime_error &) {
        }
        assert(model.cache_pos == 7);
    }
    // Semantic positions do not advance the logical cache unless requested.
    model.hidden_forward_inputs(ids, Tensor{{1, 2, 4}, 6}, Tensor{{2}, 100}, {}, nullptr, {}, true);
    assert(model.cache_pos == 9 && model.seen_position == 100 && model.seen_cache == 7);
    try {
        model.hidden_forward_inputs(ids, Tensor{{1, 2, 3}, 0});
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(model.cache_pos == 9);
}

struct FinalCausalTestOps : CausalTestOps {
    Tensor collapse_hidden(Tensor t, int64_t, int64_t) const {
        t.value = t.value * 2 + 1;
        return t;
    }
    Tensor normalize_hidden(Tensor t, const Tensor &, int64_t, int64_t) const {
        t.value *= 3;
        return t;
    }
};

void family_model_test() {
    using namespace mfq::models;
    struct QwenOps : CausalTestOps {
        Tensor positions;
        static bool defined(const Tensor &t) { return !t.shape.empty(); }
        static Tensor position_axes(Tensor t, int64_t start, int64_t count) {
            assert(start == 1 && count == 3);
            t.shape[0] = count;
            return t;
        }
        static Tensor concat_positions(Tensor first, Tensor second) {
            second.shape.back() += first.shape.back();
            return second;
        }
        qwen4_exp::Config config{};
        int batch = 1;
        static bool adapter_supports_speculation() { return false; }
    };
    qwen4_exp::CausalLm<QwenOps> qwen;
    // Dispatch must reach the family's rules, including from inherited entry points.
    assert(qwen.supports_speculation());
    qwen.hidden_forward_inputs({{1, 2}, 2}, {{1, 2, 4}, 6}, CausalTestOps::Tensor{{2}, 10});
    assert(qwen.cache_pos == 2); // Qwen4 advances even with explicit semantic positions.
    qwen.positions = {{3, 2}, 0};
    auto prepared = qwen.adapter_prepare_positions({{4, 1}, 9}, 1, 1);
    assert(prepared.positions.shape == std::vector<int64_t>({3, 1}));
    assert(prepared.full_positions.shape == std::vector<int64_t>({3, 3}));
    glm5_next::CausalLm<FinalCausalTestOps> glm;
    try {
        glm.hidden_forward({{1, 2}, 2}, CausalTestOps::Tensor{{2}, 10});
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(glm.cache_pos == 0); // GLM accepts only contiguous implicit positions.
    minicpmo45::CausalLm<CausalTestOps> mini;
    mini.next_token({{1}, 2});
    mini.next_token({{1}, 2});
    assert(!mini.seen_lengths); // Its BF16 attention uses a different decode contract.
    const CausalTestOps::Tensor ones{{1, 2}, 1}, masked{{1, 2}, 0};
    assert(!mini.adapter_attention_mask(ones, 2, 0));
    assert(!mini.adapter_attention_mask(ones, 1, 3));
    assert(mini.adapter_attention_mask(ones, 2, 3));
    assert(mini.adapter_attention_mask(masked, 2, 0));
}

void remaining_family_flows_test() {
    using namespace mfq::models;
    int state = 0, saved = -1;
    auto run = [&](int64_t start, int64_t count) {
        state += count;
        return int(start + count);
    };
    auto checkpoint = [&] { saved = state; };
    auto rollback = [&] { state = saved; };
    auto concat = [](int a, int b) { return a * 10 + b; };
    assert(recurrent_window(5, true, 2, false, run, checkpoint, rollback, concat) == 25);
    assert(saved == 2 && state == 5);
    state = 0;
    try {
        recurrent_window(
            5, true, 2, false,
            [&](int64_t start, int64_t count) {
                state += count;
                if (start)
                    throw std::runtime_error("suffix failed");
                return int(count);
            },
            checkpoint, rollback, concat);
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(state == 2);

    assert(deepseek_v4::select_compressed(
               4, 513, [] { return 1; }, [] { return 2; }, [] { return 3; }) == 1);
    assert(deepseek_v4::select_compressed(
               128, 513, [] { return 1; }, [] { return 2; }, [] { return 3; }) == 2);
    assert(deepseek_v4::select_compressed(
               4, 0, [] { return 1; }, [] { return 2; }, [] { return 3; }) == 3);
    assert(glm_dsa::indexed_attention(
               true, 5, 2, 3, [] { return 9; }, [] { return 2; }, [] { return 3; }, concat) == 23);
    try {
        glm_dsa::indexed_attention(
            true, 5, 1, 3, [] { return 9; }, [] { return 2; }, [] { return 3; }, concat);
        assert(false);
    } catch (const std::runtime_error &) {
    }

    // Compression must emit the same complete groups across an unaligned chunk boundary.
    struct Compression {
        int64_t partial_length = 0;
        std::array<int, 4> values{};
    };
    auto compress = [](Compression &state, const std::vector<int> &input, int64_t position) {
        return deepseek_v41::compress(
            state, input.size(), position, 4, [&] { return input; }, [] {},
            [&](int64_t complete, int64_t) {
                std::vector<int> result;
                for (int64_t i = 0; i < complete; ++i)
                    result.push_back(
                        std::accumulate(input.begin() + i * 4, input.begin() + (i + 1) * 4, 0));
                return result;
            },
            [&](int64_t start, int64_t count) {
                std::copy_n(input.begin() + start, count, state.values.begin());
            },
            [&](int64_t token, int64_t slot) { state.values[slot] = input[token]; },
            [&] {
                return std::vector<int>{
                    std::accumulate(state.values.begin(), state.values.end(), 0)};
            },
            [](const auto &parts) {
                std::vector<int> result;
                for (const auto &part : parts)
                    result.insert(result.end(), part.begin(), part.end());
                return result;
            });
    };
    Compression full, chunks;
    const auto expected = compress(full, {1, 2, 3, 4, 5, 6, 7, 8}, 0);
    assert(!compress(chunks, {1, 2, 3}, 0));
    assert(compress(chunks, {4, 5, 6, 7, 8}, 3) == expected);
    assert(full.partial_length == 0 && chunks.partial_length == 0);

    const int64_t sizes[]{2, 2};
    const bool mask[]{true, false, true, true, true};
    auto positions = minicpmo45::patch_positions(1, 5, 70, sizes, mask);
    assert(!positions.all_active && (positions.ids == std::vector<int64_t>{0, 0, 35, 2450, 2485}));

    int64_t tts_position = 0;
    int tts_resets = 0, tts_layers = 0;
    std::array<int, 2> decoder_layers{2, 3};
    auto tts = [&](int64_t tokens) {
        return minicpmo45::tts_forward(
            1, 1, tokens, 4, tts_position, decoder_layers, [&] { ++tts_resets; },
            [](int hidden) { return hidden + 1; }, [&] { return tts_position; },
            [&](int block, int hidden, int64_t position) {
                ++tts_layers;
                assert(position == tts_position);
                return hidden * block;
            },
            [](int hidden) { return hidden + 1; });
    };
    assert(tts(2) == 13 && tts_position == 2 && tts_resets == 1);
    assert(tts(1) == 13 && tts_position == 3 && tts_resets == 1);
    try {
        tts(2);
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(tts_position == 3 && tts_layers == 4);

    deepseek_v41::EngramHashState hashes(8, 3, 2, 0, {1}, {36}, {5, 7, 11, 13}, {0, 5, 12, 23},
                                         {3, 5, 7}, {0, 1, 2, 3, 4, 5, 6, 7});
    const int64_t ids[]{1, 2, 3, 4};
    const auto all = hashes.forward(ids, 1, 4, 0).values;
    assert((all == std::vector<int64_t>{3, 8, 15, 26, 3, 8, 15, 26, 3, 8, 16, 27, 3, 8, 14, 23}));
    auto first = hashes.forward(ids, 1, 2, 0).values;
    hashes.begin_speculative();
    auto second = hashes.forward(ids + 2, 1, 2, 2).values;
    hashes.rollback_speculative();
    assert(hashes.forward(ids + 2, 1, 2, 2).values == second);
    first.insert(first.end(), second.begin(), second.end());
    assert(first == all);

    auto reduce = [](int x) { return x * 2; };
    auto gate = [] { return 3; };
    auto add = [](int x, int y) { return x + y; };
    auto gated = [](int x, int y, int g) { return x + y * g; };
    const auto fallback = [](int, int) { return std::optional<int>{}; };
    const auto fused = [](int x, int y) { return std::optional<int>{x * 2 + y * 3}; };
    assert(combine_experts(5, 7, false, false, fallback, reduce, gate, add, gated) == 31);
    assert(combine_experts(5, 7, false, false, fused, reduce, gate, add, gated) == 31);
    assert(combine_experts(10, 7, true, true, fallback, reduce, gate, add, gated) == 17);
}

void boundary_model_test() {
    using namespace mfq::models;
    for (bool ple : {false, true})
        for (bool linear : {false, true}) {
            int selected = 0, embedded = 0;
            auto result = qwen4_exp::decoder_layer(
                2, ple, linear,
                [&](int x) {
                    ++embedded;
                    return x * 3;
                },
                [](int x, int y) { return x + y; },
                [](int x) { return std::array<int, 2>{x + 2, x}; },
                [&](int x) {
                    selected = 1;
                    return x * 3;
                },
                [&](int x) {
                    selected = 2;
                    return x * 5;
                },
                [](int x, auto mix) { return x + mix[1]; },
                [](int x) { return std::array<int, 2>{x - 1, x}; }, [](int x) { return x * 7; },
                [](int x, auto mix) { return x + mix[1]; });
            const int residual = ple ? 8 : 2;
            const int attended = (residual + 2) * (linear ? 3 : 5) + residual;
            assert(result == (attended - 1) * 7 + attended);
            assert(embedded == int(ple) && selected == (linear ? 1 : 2));
        }
    for (bool linear : {false, true}) {
        std::vector<int> norms;
        auto pre = [](int x) { return std::array<int, 3>{1, 2, x}; };
        auto result = glm5_next::decoder_layer(
            3, linear, pre,
            [&](int x, int role) {
                norms.push_back(role);
                return x + role + 1;
            },
            [](int x) { return x * 2; }, [](int x) { return x * 3; },
            [](int branch, int residual, const auto &) { return branch + residual; }, pre,
            [](int x) { return x * 5; });
        const int attended = 3 + 4 * (linear ? 2 : 3);
        assert(result == attended + (attended + 2) * 5);
        assert((norms == std::vector<int>{0, 1}));
    }
    struct Mix {
        int branch, next_pre;
    };
    int previous = 1, captured = 0;
    auto collapse = [](int hidden, int pre, int role) { return Mix{hidden + pre, pre + role + 1}; };
    auto expand = [](int branch, int residual, const Mix &, int) { return residual + branch; };
    auto result = deepseek_v41::decoder_layer(
        2, true, previous, [](int x) { return x * 3; }, [&](int x) { captured = x; }, collapse,
        [](int x) { return x * 2; }, expand, [](int x) { return x * 3; });
    assert(captured == 6 && result == 86 && previous == 4);
    previous = 1;
    try {
        deepseek_v41::mega_layer(
            2, previous, collapse, [](int x) { return x; }, expand,
            [](int) -> int { throw std::runtime_error("ffn failed"); });
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(previous == 1); // Never commit pre-weights from a failed layer.

    glm5_next::CausalLm<FinalCausalTestOps> glm;
    assert(glm.adapter_finalize_hidden({{1, 1, 4}, 2}, {}, 1, 1).value == 15);
    deepseek_v4::CausalLm<FinalCausalTestOps> v4;
    deepseek_v41::CausalLm<FinalCausalTestOps> v41;
    assert(v4.adapter_finalize_hidden({{1, 1, 4}, 2}, {}, 1, 1).value == 15);
    assert(v41.adapter_finalize_hidden({{1, 1, 4}, 2}, {}, 1, 1).value == 15);

    for (bool parallel : {false, true}) {
        std::array<int, 2> calls{};
        auto result = additive_branches(
            [&](int i) {
                ++calls[i];
                return i == 0 ? 3 : 5;
            },
            [&](auto &run, auto &outputs) {
                if (parallel)
                    outputs = {run(0), run(1)};
                return parallel;
            },
            [](int low, int high) { return low * 10 + high; });
        assert(result == 35 && calls[0] == 1 && calls[1] == 1);
    }
    try {
        additive_branches([](int) { return 1; }, [](auto &, auto &) { return true; },
                          [](int a, int b) { return a + b; });
        assert(false);
    } catch (const std::runtime_error &) {
    }
}

struct TtsTraceOps {
    using Tensor = int;
    std::string order;
    int temperature(int value, double) {
        order += 'T';
        return value;
    }
    int penalties(int value, std::span<const int> history, double) {
        order += 'P';
        assert(history.size() == 16 && history.front() == 4);
        return value;
    }
    void mask_eos(int &, int64_t) { order += 'E'; }
    int reference(int value, int64_t, double, int64_t keep) {
        order += 'R';
        assert(keep == 3);
        return value;
    }
    int top_k(int value, int64_t) {
        order += 'K';
        return value;
    }
    int top_p(int value, double) {
        order += 'N';
        return value;
    }
    int min_p(int value, double) {
        order += 'M';
        return value;
    }
    int sample(int value) {
        order += 'S';
        return value;
    }
};
void token_generation_test() {
    for (int limit : {0, 1, 4})
        for (bool eos : {false, true}) {
            std::vector<int> accepted;
            int samples = 0, advances = 0;
            const auto result = generate_tokens(
                limit,
                [&](int64_t step) {
                    ++samples;
                    return int(step);
                },
                [&](int token, int64_t) { accepted.push_back(token); }, [&](int) { return eos; },
                [&](int) { ++advances; });
            const int count = eos && limit ? 1 : limit;
            assert(samples == count && accepted.size() == size_t(count) && result.tokens == count);
            assert(advances == std::max(0, count - 1) && result.hit_eos == (eos && limit > 0));
        }
    int accepted = 0;
    try {
        generate_tokens(
            3, [](int64_t step) { return step; }, [&](int, int64_t) { ++accepted; },
            [](int) { return false; }, [](int) { throw std::runtime_error("decode failed"); });
        assert(false);
    } catch (const std::runtime_error &) {
    }
    assert(accepted == 1);
    std::array<int, 20> history{};
    std::iota(history.begin(), history.end(), 0);
    mfq::models::minicpmo45::TtsSampling options;
    options.minimum_keep = 3;
    options.min_p = 0.1;
    options.validate(3, 6562);
    TtsTraceOps ops;
    mfq::models::minicpmo45::sample_tts(ops, 0, options, history, 0, false);
    assert(ops.order == "PEKNMTS");
    ops.order.clear();
    mfq::models::minicpmo45::sample_tts(ops, 0, options, history, 0, true);
    assert(ops.order == "TPER");
    ops.order.clear();
    options.minimum_steps = 0;
    options.repetition_penalty = 1;
    mfq::models::minicpmo45::sample_tts(ops, 0, options, {}, 0, false);
    assert(ops.order == "KNMTS");
    std::mt19937 rng(42);
    const float scores[]{0, 1, 5, -1};
    for (int i = 0; i < 10; ++i)
        assert(sample_top_k_top_p(scores, 3, 0.01, 0, rng) == 2);
    const float invalid[]{-INFINITY, -INFINITY};
    try {
        sample_top_k_top_p(invalid, 1, 1, 0, rng);
        assert(false);
    } catch (const std::invalid_argument &) {
    }
}

struct LoadTestModel : mfq::models::CausalModelBase<CausalTestOps, LoadTestModel> {
    static bool accepts_backbone(std::string_view value) { return value == "fixture"; }
    void adapter_set_max_position_embeddings(int64_t) {}
    template <class Graph, class Source>
    void adapter_load_config(std::string_view, const Graph &, const Source &) {
        metadata.num_hidden_layers = 2;
        metadata.layer_types = {"first", "second"};
    }
};
void model_loading_test() {
    struct Graph {
        std::string backbone = "fixture";
        struct {
            int64_t text_layers = 2;
        } topology;
    } graph;
    struct Loader {
        bool has_output = true;
        std::vector<std::string> calls;
        int embedding(const std::string &name) {
            calls.push_back(name);
            return 11;
        }
        int output(const std::string &name) {
            calls.push_back(name);
            return 22;
        }
        int tied_output(int value, const std::string &name) {
            calls.push_back("tie:" + name);
            return value;
        }
        bool has_weight(const std::string &) { return has_output; }
        void final_state(CausalTestOps::Tensor &) { calls.push_back("norm"); }
        void prepare_blocks() { calls.push_back("prepare"); }
        auto block(int layer, const std::string &type) {
            calls.push_back(type);
            return std::make_unique<CausalTestOps::Block>(CausalTestOps::Block{layer});
        }
    };
    for (int mode = 0; mode < 3; ++mode) {
        LoadTestModel model;
        model.blocks.clear();
        model.load_definition("", graph, 0, 16);
        assert(model.max_position_embeddings() == 16);
        model.metadata.tie_word_embeddings = mode == 1;
        Loader loader;
        loader.has_output = mode != 2;
        model.load_weights(loader);
        assert(model.embed == 11 && model.lm_head == (mode ? 11 : 22) && model.blocks.size() == 2);
        assert(loader.calls == std::vector<std::string>({"model.token_embedding.weight", "norm",
                                                         mode ? "tie:model.token_embedding.weight"
                                                              : "model.output.weight",
                                                         "prepare", "first", "second"}));
    }
    LoadTestModel model;
    model.blocks.clear();
    model.load_definition("", graph, 0);
    Loader loader;
    model.load_weights(loader, false);
    assert(model.blocks.empty() && loader.calls.size() == 3);
    for (int invalid = 0; invalid < 3; ++invalid) {
        auto broken = graph;
        if (invalid == 0)
            broken.backbone = "other";
        if (invalid == 1)
            broken.topology.text_layers = 3;
        try {
            model.load_definition("", broken, 0, invalid == 2 ? 33 : 0);
            assert(false);
        } catch (const std::runtime_error &) {
        }
    }
}

struct MediaTestOps {
    using Tensor = CausalTestOps::Tensor;
    std::string calls;
    static int64_t rank(const Tensor &t) { return t.shape.size(); }
    static int64_t size(const Tensor &t, int axis) { return t.shape.at(axis); }
    static bool defined(const Tensor &t) { return !t.shape.empty(); }
    static Tensor batch_ids(Tensor t) {
        t.shape.insert(t.shape.begin(), 1);
        return t;
    }
    static Tensor device_ids(Tensor t) { return t; }
    Tensor embed(Tensor ids) {
        calls += 'E';
        return {{ids.shape[0], ids.shape[1], 4}, 4};
    }
    mfq::StepSequence<Tensor> vision(Tensor, Tensor, Tensor) {
        calls += 'V';
        co_yield Tensor{{1, 4, 3}, 7};
    }
    Tensor resample(Tensor t, Tensor) {
        calls += 'R';
        assert(t.value == 7);
        return {{1, 2, 4}, 10};
    }
    void reset_audio() { calls += 'A'; }
    mfq::StepSequence<Tensor> audio(Tensor, Tensor) {
        calls += 'U';
        co_yield Tensor{{1, 3, 4}, 20};
    }
    auto audio_lengths(Tensor) {
        calls += 'L';
        return std::vector<int64_t>{3};
    }
    void scatter(Tensor &target, const Tensor &source,
                 const mfq::models::minicpmo45::MediaBound &bound) {
        calls += 'S';
        assert(bound.end - bound.begin == source.shape[1]);
        target.value += source.value;
    }
};

void bounded_encoder_test() {
    using namespace mfq::models;
    std::array<int, 3> layers{1, 2, 3};
    for (bool audio : {false, true}) {
        for (int stop = 0; stop <= 5; ++stop) {
            int visits = 0;
            const auto layer = [&](int n, int hidden) { ++visits; return hidden + n; };
            auto sequence = audio
                ? minicpmo45::audio_encoder(1, layers, [](int x) { return x * 2; },
                    [](int x) { return x; }, [](int x) { return x + 1; },
                    [](int x) { return std::array{x, 0}; },
                    [&](int n, int x, int) { return layer(n, x); },
                    [](int x) { return x; }, [](int x) { return x * 2; },
                    [](int x) { return x; }, [](int x) { return x + 1; }, [](int x) { return x; })
                : minicpmo45::vision_encoder(1, layers, [](int x) { return x * 2; },
                    [](int x) { return x + 1; }, layer, [](int x) { return x * 2 + 1; });
            for (int tick = 0; tick < stop; ++tick) {
                auto step = sequence.next();
                assert(step.state == StepState::advanced);
                assert(visits == std::min(tick, 3));
                if (tick == 4) assert(step.value == 19);
                else assert(!step.value);
            }
            sequence = {}; // Cancellation cannot execute another encoder layer.
            assert(visits == std::clamp(stop - 1, 0, 3));
        }
    }
}

void composition_boundary_test() {
    using namespace mfq::models;
    using Tensor = CausalTestOps::Tensor;
    MediaTestOps ops;
    minicpmo45::MultimodalInputs<Tensor> input{{{6}, 1}, {{1}, 1}, {{1}, 1},       {{1}, 1},
                                               {{1}, 1}, {{1}, 1}, {{0, 0, 1, 3}}, {{0, 0, 3, 6}}};
    auto result = mfq::finish_steps(minicpmo45::encode(ops, input));
    assert(ops.calls == "EVRSAULS" && result.input_embeddings.value == 34);
    assert(input.ids.shape == std::vector<int64_t>({1, 6}));
    minicpmo45::CausalLm<CausalTestOps> model;
    auto forwarded = minicpmo45::multimodal_forward(
        model, input.ids, result, std::optional<Tensor>{}, std::optional<Tensor>{});
    assert(model.cache_pos == 6 && forwarded.hidden_states.value == 143 &&
           forwarded.logits.value == 715);
    for (int invalid = 0; invalid < 4; ++invalid) {
        auto broken = input;
        if (invalid == 0)
            broken.images[0].end = 4;
        if (invalid == 1)
            broken.images[0].source = 1;
        if (invalid == 2)
            broken.audios[0].source = -1;
        if (invalid == 3)
            broken.pixels = {};
        try {
            mfq::finish_steps(minicpmo45::encode(ops, broken));
            assert(false);
        } catch (const std::runtime_error &) {
        }
    }
    auto text = input;
    text.images.clear();
    text.audios.clear();
    ops.calls.clear();
    assert(mfq::finish_steps(minicpmo45::encode(ops, text)).input_embeddings.value == 4 && ops.calls == "E");
    try {
        minicpmo45::media_bounds(std::array<int64_t, 4>{0, 0, 2, 1});
        assert(false);
    } catch (const std::runtime_error &) {
    }

    mfq::GridVisionConfig config;
    config.spatial_merge_size = 2;
    config.hidden_size = 3;
    config.num_position_embeddings = 4;
    std::array<int, 2> layers{1, 2};
    std::string order;
    const auto encoded = mfq::finish_steps(grid_vision::encode(
        1, config, {{1, 2, 2}}, layers,
        [&](int x, int64_t count) {
            order += 'P';
            assert(count == 4);
            return x + 2;
        },
        [&](const mfq::LearnedPositionInterpolation &p) {
            order += 'I';
            assert(p.patch_count == 4);
            return 5;
        },
        [&](int x, int y) {
            order += 'A';
            return x + y;
        },
        [&](int layer, int x, const mfq::GridVisionLayout &layout) {
            order += 'L';
            assert(layout.patch_count == 4);
            return x * layer;
        }));
    assert(encoded == 16 && order == "PIALL");
    order.clear();
    auto merge = [&](int64_t count) {
        return grid_vision::merge(
            encoded, count, config,
            [&](int x) {
                order += 'N';
                return x + 1;
            },
            [&](int x, int64_t rows, int64_t width) {
                order += 'R';
                assert(rows == 1 && width == 12);
                return x;
            },
            [&](int x) {
                order += 'U';
                return x * 2;
            },
            [&](int x) {
                order += 'G';
                return x + 3;
            },
            [&](int x) {
                order += 'D';
                return x * 5;
            });
    };
    assert(merge(4) == 185 && order == "NRUGD");
    try {
        merge(3);
        assert(false);
    } catch (const std::runtime_error &) {
    }
    int scatters = 0;
    auto prepare = [&](std::vector<int64_t> ids) {
        return grid_vision::prepare(
            ids, 185, 1, 99, 98, config, {{1, 2, 2}}, [](const auto &) { return 7; },
            [&](int &embedding, int vision, const auto &indices) {
                ++scatters;
                assert(indices == std::vector<int64_t>{1});
                embedding += vision;
            },
            [](const mfq::GridMropePositions &positions) {
                assert(positions.token_count == 3);
                return 1;
            });
    };
    assert(prepare({1, 99, 2}).embeddings == 192 && scatters == 1);
    for (const auto &ids : {std::vector<int64_t>{1, 2, 3}, std::vector<int64_t>{1, 98, 3}}) {
        try {
            prepare(ids);
            assert(false);
        } catch (const std::invalid_argument &) {
        }
    }
    assert(scatters == 1);

    gemma4::Config gemma;
    gemma.num_attention_heads = 8;
    gemma.num_key_value_heads = 2;
    gemma.num_global_key_value_heads = 1;
    gemma.head_dim = 4;
    gemma.global_head_dim = 8;
    gemma.sliding_window = 16;
    gemma.full_rotary_factor = 0.5;
    gemma.attention_key_equals_value = true;
    auto local = gemma4::layer_spec(gemma, "sliding_attention");
    auto global = gemma4::layer_spec(gemma, "full_attention");
    assert(local.head_dim == 4 && local.kv_heads == 2 && local.window == 16 &&
           !local.value_equals_key);
    assert(global.head_dim == 8 && global.kv_heads == 1 && global.window == 0 &&
           global.value_equals_key && global.rotary_pairs == 2);
    try {
        gemma4::layer_spec(gemma, "unknown");
        assert(false);
    } catch (const std::runtime_error &) {
    }

    // Model head policy is shared even for a backend whose projection is just an integer operation.
    model.metadata.final_logit_softcapping = 1;
    assert(model.logits_from_hidden({{1, 4}, 3}).value == -15);
    assert(model.last_logits({{1}, 2}).value == -217);
}

template <class Config> struct LayerParameters {
    Config config;
    std::string attention_gr, ffn_gr, attention_hc, ffn_hc, attention_norm, ffn_norm, ffn;
    std::string gdn, qsa, ple, kda, mla, attention;
};

template <class Config> struct PredictorParameters {
    using Layer = LayerParameters<Config>;
    Config config;
    std::string embedding_norm, hidden_norm, output_norm, embedding_fusion, hidden_fusion, fusion, final_mixer;
    std::vector<Layer> layers;
    std::vector<std::string> positions;
    std::vector<int64_t> lengths;
};

// String handles exercise model assembly without a device or tensor implementation.
struct ParameterLoader {
    using Tensor = std::string;
    using Embedding = std::string;
    using GdnWeights = std::array<std::string, 9>;
    using QsaWeights = std::array<std::string, 9>;
    using PleWeights = std::array<std::string, 6>;
    using KdaWeights = std::array<std::string, 13>;
    using MlaWeights = std::array<std::string, 15>;
    bool qwen = false, present = true, partial = false, bad_norm = false, bad_shard = false;
    std::vector<std::string> names;
    Tensor dense(const std::string &name) { names.push_back(name); return name; }
    Tensor linear(const std::string &name) { return dense(name); }
    Tensor residual_linear(const std::string &name) { return linear(name); }
    static Tensor fp32(Tensor value) { return value; }
    bool has(const std::string &) const { return present; }
    bool has_prefix(const std::string &) const { return present || partial; }
    std::vector<int64_t> shape(const Tensor &value) const {
        return {value == "predictor.hidden_norm.weight" && qwen ? 16 : bad_norm ? 7 : 8};
    }
    int64_t elements(const Tensor &value) const { return shape(value)[0]; }
    Tensor routed(const std::string &name, int, int64_t, int64_t, int64_t) { return dense(name); }
    Tensor routed_gate_up(const std::string &p, int, int64_t, int64_t, int64_t) {
        return dense(p + ".experts.gate_up.weight");
    }
    static Tensor residual(Tensor norm, Tensor, Tensor, Tensor, const mfq::models::qwen4_exp::Config &) { return norm; }
    static Tensor final_mixer(Tensor value) { return value; }
    GdnWeights gdn_weights(const std::string &p) {
        return {linear(p + ".qkv.weight"),linear(p + ".gate.weight"),linear(p + ".alpha.weight"),
            linear(p + ".beta.weight"),linear(p + ".output.weight"),dense(p + ".conv.weight"),
            dense(p + ".dt_bias"),dense(p + ".a"),dense(p + ".norm.weight")};
    }
    QsaWeights qsa_weights(const std::string &p) {
        return {linear(p + ".query.weight"),linear(p + ".key.weight"),linear(p + ".value.weight"),
            linear(p + ".output.weight"),linear(p + ".indexer.query_key.weight"),dense(p + ".query_norm.weight"),
            dense(p + ".key_norm.weight"),dense(p + ".indexer.query_norm.weight"),dense(p + ".indexer.key_norm.weight")};
    }
    static Tensor gdn(GdnWeights w, const mfq::models::qwen4_exp::Config &) { return w[0]; }
    static Tensor qsa(QsaWeights w, const mfq::models::qwen4_exp::Config &) { return w[0]; }
    template <class... Args> static Tensor moe(Args &&...) { return "moe"; }
    static Tensor gated_mlp(Tensor gate, Tensor, Tensor, double) { return gate; }
    auto embedding(const std::string &name) {
        return std::pair{dense(name), std::array<int64_t, 2>{32, bad_shard ? 7 : 8}};
    }
    std::vector<int64_t> integers(const std::string &name) { dense(name); return {1}; }
    static Tensor ple(std::vector<Embedding> shards, int64_t, int64_t,
                      const mfq::models::qwen4_exp::Config &, std::vector<int64_t>,
                      std::vector<int64_t>, std::vector<int64_t>, PleWeights) { return shards.front(); }
    auto block(const mfq::models::qwen4_exp::Config &c, int layer, bool predictor) {
        LayerParameters<mfq::models::qwen4_exp::Config> result;
        mfq::models::qwen4_exp::load_block(result, *this, c, layer, predictor);
        return result;
    }
    static Tensor mhc(Tensor function, Tensor, Tensor) { return function; }
    static Tensor concat(const std::vector<Tensor> &parts) { return parts[0]; }
    static Tensor headwise(Tensor weight, int64_t, int64_t) { return weight; }
    static Tensor kda(KdaWeights w, const mfq::models::glm5_next::Config &) { return w[0]; }
    static Tensor mla(MlaWeights w, const mfq::models::glm5_next::Config &) { return w[0]; }
};

static void family_parameter_loading_test() {
    using namespace mfq::models;
    qwen4_exp::Config q{};
    q.hidden = 8; q.streams = 2; q.ngram = 2; q.ngram_heads = 1; q.shards = 2;
    q.layer_types = {"linear_attention", "full_attention"}; q.ple_layers = {1};
    q.predictor_layers = 3; // Predictor depth is independent of the backbone schedule.
    ParameterLoader qops;
    qops.qwen = true;
    auto linear = qops.block(q, 0, false), sparse = qops.block(q, 1, false);
    assert(linear.gdn == "model.block.0.linear_attention.qkv.weight" && linear.qsa.empty());
    assert(linear.ple == "model.block.0.position_embedding.ngram.shard.0.weight");
    assert(sparse.qsa == "model.block.1.attention.query.weight" && sparse.gdn.empty() && sparse.ple.empty());
    PredictorParameters<qwen4_exp::Config> qp;
    assert(qwen4_exp::load_predictor(qp, qops, q) && qp.layers.size() == 3 && qp.lengths.size() == 3);
    for (size_t i = 0; i < qp.layers.size(); ++i) {
        assert(qp.layers[i].gdn.empty() && qp.layers[i].ple.empty());
        assert(qp.layers[i].qsa == "predictor.block." + std::to_string(i) + ".attention.query.weight");
    }
    qops.bad_shard = true;
    try { qops.block(q, 0, false); assert(false); } catch (const std::runtime_error &) {}

    glm5_next::Config g{};
    g.hidden = 8; g.predictor_layers = 3;
    g.layer_types = {"linear_attention", "deepseek_sparse_attention"};
    g.mlp_types = {"dense", "sparse"};
    ParameterLoader gops;
    LayerParameters<glm5_next::Config> kda, mla;
    glm5_next::load_block(kda, gops, g, 0);
    glm5_next::load_block(mla, gops, g, 1);
    assert(kda.kda == "model.block.0.linear_attention.query.weight" && kda.mla.empty());
    assert(kda.ffn == "model.block.0.mlp.gate.weight" && mla.ffn == "moe");
    assert(mla.mla == "model.block.1.attention.query_a.weight" && mla.kda.empty());
    PredictorParameters<glm5_next::Config> gp;
    assert(glm5_next::load_predictor(gp, gops, g) && gp.layers.size() == 3 && gp.lengths.size() == 3);
    for (size_t i = 0; i < gp.layers.size(); ++i) {
        assert(gp.layers[i].ffn == "moe");
        assert(gp.layers[i].attention == "predictor.block." + std::to_string(i) + ".attention.query_a.weight");
    }
    // Absence is allowed; a partial, undeclared, or incorrectly sized head is not.
    auto check_head = [](auto c, auto load) {
        ParameterLoader ops;
        using Config = decltype(c);
        ops.qwen = std::is_same_v<Config, qwen4_exp::Config>;
        ops.present = false;
        PredictorParameters<Config> absent;
        assert(!load(absent, ops, c));
        for (int mode = 0; mode < 3; ++mode) {
            ops.present = mode != 0;
            ops.partial = mode == 0;
            ops.bad_norm = mode == 2;
            auto config = c;
            if (mode == 1) config.predictor_layers = 0;
            PredictorParameters<Config> result;
            try { load(result, ops, config); assert(false); } catch (const std::runtime_error &) {}
        }
    };
    check_head(q, [](auto &p, auto &ops, const auto &c) { return qwen4_exp::load_predictor(p, ops, c); });
    check_head(g, [](auto &p, auto &ops, const auto &c) { return glm5_next::load_predictor(p, ops, c); });
}

// Shape handles verify that model assembly needs no CUDA tensor or loader type.
namespace assembly_test {
struct Tensor { std::string name; std::vector<int64_t> dims; };
struct Linear {
    Tensor weight;
    int64_t out() const { return weight.dims.at(0); }
    int64_t neuron_len() const { return weight.dims.at(1); }
};
struct Group { std::vector<std::string> names; std::vector<int64_t> outs; };
struct Experts { int64_t n_experts = 0, out_per_expert = 0, neuron_len = 0; };
struct Ffn {
    Linear down;
    Group gate_up;
    Experts moe_gate, moe_up, moe_gate_up, moe_down;
    Tensor moe_router, moe_router_bias, moe_shared_gate;
    std::unique_ptr<Ffn> shared;
    bool is_moe = false, moe_split_gate_up = false, moe_use_sigmoid = false,
         moe_use_sqrt_softplus = false, moe_normalize = false, moe_delayed_softmax = true,
         moe_shared_ungated = false, prepared = false;
    int moe_layer = -1, moe_top_k = 0;
    double moe_router_scale = 1;
};
struct Layer {
    Ffn ffn;
    mfq::models::qwen35::Config qwen_config;
    mfq::models::glm_dsa::Config config;
    int layer = -1;
    int64_t attention_heads = 0, kv_heads = 0, attention_head_dim = 0, max_position_embeddings = 0;
    double rms_norm_eps = 0, norm_weight_offset = 0;
    bool attention_output_gate = false, tiled_v_heads = false, linear = false,
         split_in_proj = false, full_indexer = false;
    Tensor attn_norm, ffn_norm, q_norm, k_norm, conv_weight, conv_bias, dt_bias, a_log,
           linear_norm, q_a_norm, kv_a_norm, index_k_norm, index_k_bias;
    Group qkv, input_proj, q_proj;
    Linear o, o_proj, out_proj;
    Experts embed_q, unembed_out;
};
struct Loader {
    using Ffn = assembly_test::Ffn;
    std::unordered_map<std::string, std::vector<int64_t>> weights;
    bool has(const std::string& name) const { return weights.count(name); }
    Tensor dense(const std::string& name) const { return {name, weights.at(name)}; }
    Tensor router_parameter(const std::string& name) const { return dense(name); }
    Linear linear(const std::string& name) const { return {dense(name)}; }
    Group projections(const std::vector<std::string>& names) const {
        Group result{names, {}};
        for (const auto& name : names) result.outs.push_back(weights.at(name).at(0));
        return result;
    }
    Group gate_up(const std::vector<std::string>& names, const Linear&) const { return projections(names); }
    void qkv(Layer& b, const std::vector<std::string>& names) const { b.qkv = projections(names); }
    static void workspace(Ffn& f) { f.prepared = true; }
    static void important_neurons(int64_t, int64_t, Ffn&, const std::string&, const std::string&,
                                  const std::string&) {}
    Experts headwise(const std::string& name) const {
        const auto& s = weights.at(name);
        return {s.at(0), s.at(1), s.at(2)};
    }
    void experts(Ffn& f, mfq::models::ExpertProjection role, const std::string& name, int) const {
        using Role = mfq::models::ExpertProjection;
        auto& value = role == Role::gate ? f.moe_gate : role == Role::up ? f.moe_up
                    : role == Role::down ? f.moe_down : f.moe_gate_up;
        value = headwise(name);
    }
    static auto shape(const Tensor& value) { return value.dims; }
    static int64_t elements(const Tensor& value) {
        return std::accumulate(value.dims.begin(), value.dims.end(), int64_t{1}, std::multiplies<>{});
    }
    static auto full_block() { return std::make_unique<Layer>(); }
    static auto linear_block() { auto b = full_block(); b->linear = true; return b; }
    static auto finish(std::unique_ptr<Layer> value) { return value; }
    static Tensor log_negative(Tensor value) { value.name = "log(-" + value.name + ")"; return value; }
    void linear_projections(Layer& b, const mfq::models::qwen35::LinearWeightNames& n, bool split) const {
        b.split_in_proj = split;
        b.input_proj = projections(split ? std::vector{n.qk, n.value, n.gate, n.alpha, n.beta}
                                          : std::vector{n.qkv, n.gate, n.alpha, n.beta});
    }
    void linear_output(Layer& b, const std::string& name) const { b.out_proj = linear(name); }
    void dense_ffn(const std::string& p) {
        weights[p + "down.weight"] = {8, 16};
        weights[p + "gate.weight"] = weights[p + "up.weight"] = {16, 8};
    }
    void moe(const std::string& p, bool split) {
        if (split) weights[p + "experts.gate.weight"] = weights[p + "experts.up.weight"] = {2, 4, 8};
        else weights[p + "experts.gate_up.weight"] = {2, 8, 8};
        weights[p + "experts.down.weight"] = {2, 8, 4};
        weights[p + "router.weight"] = {2, 8};
        weights[p + "router.bias"] = {2};
        weights[p + "shared_expert.router.weight"] = {1, 8};
        dense_ffn(p + "shared_expert.");
    }
};
void run() {
    using namespace mfq::models;
    qwen35::Config q{};
    q.hidden_size = 8; q.intermediate_size = 16; q.num_attention_heads = 2;
    q.num_key_value_heads = 1; q.head_dim = 4; q.max_position_embeddings = 32;
    q.num_experts = 2; q.num_experts_per_tok = 1; q.moe_intermediate_size = 4;
    q.shared_expert_intermediate_size = 16; q.attention_output_gate = true;
    q.legacy_tensor_layout.norm_weight_offset = 0.25;
    Loader base;
    const std::string p = "model.block.0.", a = p + "attention.", f = p + "mlp.";
    base.weights[a + "norm.weight"] = base.weights[f + "norm.weight"] = {8};
    for (const auto* name : {"query", "key", "value", "output"}) base.weights[a + name + ".weight"] = {8, 8};
    base.weights[a + "query_norm.weight"] = {4};
    base.dense_ffn(f);
    auto full = qwen35::load_block(base, q, 0, "full_attention");
    assert(full->qkv.names == std::vector({a + "query.weight", a + "key.weight", a + "value.weight"}));
    assert(full->attention_output_gate && full->norm_weight_offset == 0.25 && full->k_norm.name.empty());
    assert(full->ffn.prepared && !full->ffn.is_moe && full->ffn.down.weight.name == f + "down.weight");
    for (bool split : {false, true}) {
        auto ops = base;
        ops.moe(f, split);
        auto moe = qwen35::load_block(ops, q, 0, "full_attention");
        assert(moe->ffn.is_moe && moe->ffn.moe_split_gate_up == split && moe->ffn.shared->prepared);
        assert(moe->ffn.moe_top_k == 1 && moe->ffn.moe_delayed_softmax);
        auto bad = ops;
        bad.weights[f + "experts.down.weight"] = {2, 7, 4};
        try { qwen35::load_block(bad, q, 0, "full_attention"); assert(false); } catch (const std::runtime_error&) {}
        bad = ops;
        if (split) bad.weights.erase(f + "experts.up.weight");
        else bad.weights[f + "experts.gate.weight"] = {2, 4, 8};
        try { qwen35::load_block(bad, q, 0, "full_attention"); assert(false); } catch (const std::runtime_error&) {}
    }
    for (bool split : {false, true}) {
        auto ops = base;
        const auto l = p + "linear_attention.";
        for (const auto* name : {"qkv", "gate", "alpha", "beta", "output", "conv", "norm"})
            ops.weights[l + name + ".weight"] = {8, 8};
        if (split) ops.weights[l + "qk.weight"] = ops.weights[l + "value.weight"] = {8, 8};
        ops.weights[l + "dt_bias"] = ops.weights[l + "a"] = {2};
        q.legacy_tensor_layout.linear_attention_a_is_log = split;
        auto linear = qwen35::load_block(ops, q, 0, "linear_attention");
        assert(linear->linear && linear->split_in_proj == split && linear->ffn.prepared);
        assert(linear->input_proj.names.front() == l + (split ? "qk.weight" : "qkv.weight"));
        assert(linear->a_log.name == (split ? l + "a" : "log(-" + l + "a)"));
        assert(linear->conv_bias.name.empty() && linear->out_proj.weight.name == l + "output.weight");
    }
    for (auto component : {minicpmo45::LanguageComponent::text, minicpmo45::LanguageComponent::tts,
                           minicpmo45::LanguageComponent::standalone_tts}) {
        Layer mini;
        minicpmo45::load_language_block(mini, base, q, 0, "full_attention", component);
        assert(mini.ffn.prepared && !mini.attention_output_gate);
        assert(mini.norm_weight_offset == (component == minicpmo45::LanguageComponent::standalone_tts ? 1 : 0));
    }
    glm_dsa::Config g{};
    g.hidden_size = 8; g.intermediate_size = 16; g.num_attention_heads = 2;
    g.q_lora_rank = 3; g.kv_lora_rank = 4; g.qk_nope_head_dim = 2; g.qk_rope_head_dim = 2;
    g.v_head_dim = 4; g.index_head_dim = 3; g.index_n_heads = 2;
    g.num_experts = 2; g.num_experts_per_tok = 1; g.moe_intermediate_size = 4;
    for (bool indexer : {false, true}) {
        auto ops = base;
        g.indexer_types = {indexer ? "full" : "shared"};
        g.mlp_layer_types = {indexer ? "sparse" : "dense"};
        if (indexer) ops.moe(f, false);
        ops.weights[a + "query_a_norm.weight"] = {3}; ops.weights[a + "key_value_a_norm.weight"] = {4};
        ops.weights[a + "query_a.weight"] = {3, 8}; ops.weights[a + "key_value_a.weight"] = {6, 8};
        ops.weights[a + "query_b.weight"] = {8, 3};
        ops.weights[a + "latent.query_embedding.weight"] = {2, 4, 2};
        ops.weights[a + "latent.output_unembedding.weight"] = {2, 4, 4};
        ops.weights[a + "indexer.key.weight"] = {3, 8}; ops.weights[a + "indexer.score.weight"] = {2, 8};
        ops.weights[a + "indexer.query.weight"] = {6, 3};
        ops.weights[a + "indexer.key_norm.weight"] = ops.weights[a + "indexer.key_norm.bias"] = {3};
        Layer b;
        glm_dsa::load_block(b, ops, g, 0, "glm_dsa");
        assert(b.full_indexer == indexer && b.input_proj.names.size() == (indexer ? 4 : 2));
        assert(b.q_proj.names.size() == (indexer ? 2 : 1) && b.ffn.is_moe == indexer);
        if (indexer) assert(b.ffn.moe_use_sigmoid && b.ffn.moe_shared_ungated && !b.ffn.moe_delayed_softmax);
        ops.weights[a + "latent.query_embedding.weight"] = {2, 5, 2};
        try { glm_dsa::load_block(b, ops, g, 0, "glm_dsa"); assert(false); } catch (const std::runtime_error&) {}
    }
}
} // namespace assembly_test

int main() {
    assembly_test::run();
    family_parameter_loading_test();
    model_loading_test();
    bounded_encoder_test();
    composition_boundary_test();
    boundary_model_test();
    token_generation_test();
    family_model_test();
    remaining_family_flows_test();
    causal_lm_test();
    inner_model_flow_test();
    predictor_flow_test();
    nemotron_predictor_flow_test();
    batch_test();
    packed_prefill_plain_sample_test();
    batch_failure_test();
    batch_recoverable_failure_test();
    batch_alignment_test();
    staggered_decode_batch_test();
    isolated_prefill_failure_test();
    packed_prefill_test();
    isolated_context_growth_test();
    model_test();
}
