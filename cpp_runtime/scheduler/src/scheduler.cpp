#include "scheduler.h"
#include "mfq/fatal_diagnostics.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

using namespace mfq::engine;

namespace {
constexpr std::size_t terminal_reserve = sizeof(EngineEvent) + 128 + 64 + 512;
std::size_t event_bytes(const EventData& data) {
    return std::visit([](const auto& value) -> std::size_t {
        using T = std::decay_t<decltype(value)>;
        std::size_t bytes = sizeof(EngineEvent);
        if constexpr (std::is_same_v<T, OutputDelta>) {
            bytes += value.token_ids.size() * sizeof(std::int64_t) + value.diffs.size() * sizeof(common_chat_msg_diff);
            for (const auto& diff : value.diffs) {
                bytes += diff.content_delta.size() + diff.reasoning_content_delta.size();
                bytes += diff.tool_call_delta.name.size() + diff.tool_call_delta.arguments.size() + diff.tool_call_delta.id.size();
            }
        } else if constexpr (std::is_same_v<T, Completed>) {
            bytes += value.finish_reason.size();
        } else if constexpr (std::is_same_v<T, Failed>) {
            bytes += value.code.size() + value.message.size();
        }
        return bytes;
    }, data);
}
}

std::vector<EngineEvent> MfqScheduledRequest::wait() {
    std::unique_lock lock(mutex_);
    ready_.wait(lock, [&] { return terminal_ || !events_.empty(); });
    std::vector<EngineEvent> result;
    while (!events_.empty()) {
        result.push_back(std::move(events_.front()));
        events_.pop_front();
    }
    bytes_ = 0;
    lock.unlock();
    wake_->notify_one();
    return result;
}

bool MfqScheduledRequest::done() const {
    std::lock_guard lock(mutex_);
    return terminal_ && events_.empty();
}

MfqScheduler::MfqScheduler(Engine& engine) : MfqScheduler(engine, Limits{}) {}
MfqScheduler::MfqScheduler(Engine& engine, Limits limits)
    : engine_(engine), limits_(limits), info_(engine.info()) {
    if (limits.events < 4 || limits.bytes < 4096)
        throw std::invalid_argument("outbox requires at least 4 events and 4096 bytes");
    std::promise<void> ready;
    auto initialized = ready.get_future();
    worker_ = std::thread([this, &ready] {
        refresh_snapshot(true, false, true);
        ready.set_value();
        loop();
    });
    initialized.get();
}
MfqScheduler::~MfqScheduler() { shutdown(); }
void MfqScheduler::shutdown() {
    { std::lock_guard lock(mutex_); stopping_ = true; }
    wake_->notify_one();
    if (worker_.joinable()) worker_.join();
}
void MfqScheduler::enqueue(Command command) const {
    std::lock_guard lock(mutex_);
    if (stopping_) throw std::runtime_error("scheduler is stopping");
    mailbox_.push_back(std::move(command));
    wake_->notify_one();
}
EngineInfo MfqScheduler::info() const {
    std::lock_guard lock(mutex_);
    return info_;
}
EngineStatus MfqScheduler::status() const {
    Status command;
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
MfqScheduler::Snapshot MfqScheduler::snapshot() const {
    std::lock_guard lock(mutex_);
    if (!stopping_ && Clock::now() - snapshot_.captured >= std::chrono::milliseconds(250)) {
        snapshot_requested_ = true;
        wake_->notify_one();
    }
    return snapshot_;
}
void MfqScheduler::refresh_snapshot(bool healthy, bool blocked, bool metrics) {
    auto status = engine_.status();
    status.healthy &= healthy;
    if (blocked || !status.healthy) status.available = 0;
    std::optional<Metrics> runtime, session;
    if (metrics) {
        try { runtime = std::get<Metrics>(engine_.control(RuntimeMetrics{})); } catch (...) {}
        try { session = engine_.session({SessionCommand::Kind::metrics}).metrics; } catch (...) {}
    }
    std::lock_guard lock(mutex_);
    snapshot_.status = status;
    const auto merge = [](Metrics& target, const Metrics& updates) {
        for (const auto& value : updates) {
            auto found = std::find_if(target.begin(), target.end(), [&](const auto& existing) {
                return existing.first == value.first;
            });
            if (found == target.end()) target.push_back(value);
            else found->second = value.second;
        }
    };
    if (runtime) merge(snapshot_.runtime, *runtime);
    if (session) merge(snapshot_.session, *session);
    if (runtime && session) snapshot_.captured = Clock::now();
}
std::shared_ptr<MfqScheduledRequest> MfqScheduler::submit(EngineRequest input) const {
    if (input.id.empty() || input.id.size() > 128)
        throw std::invalid_argument("request ID must contain 1-128 bytes");
    auto outbox = std::make_shared<MfqScheduledRequest>();
    outbox->wake_ = wake_;
    Submit command{{std::move(input), outbox}, {}};
    auto result = command.reply.get_future();
    enqueue(std::move(command));
    result.get();
    return outbox;
}
bool MfqScheduler::cancel_request(const std::string& id) const {
    Cancel command{id, false, false, {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
bool MfqScheduler::cancel_session(const std::string& id) const {
    Cancel command{id, true, false, {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
void MfqScheduler::cancel_all() const {
    Cancel command{{}, false, true, {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); result.get();
}
ControlResult MfqScheduler::control(ControlRequest request) const {
    Control command{std::move(request), {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
SessionResult MfqScheduler::session(SessionCommand request) const {
    Session command{std::move(request), {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
std::int64_t MfqScheduler::reload(std::int64_t context) const {
    Reload command{context, {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
void MfqScheduler::prepare_duplex_session(const std::string& prompt, MfqDuplexSessionParams& params) const {
    params = std::get<MfqDuplexSessionParams>(control(PrepareDuplex{prompt, params}));
}
void MfqScheduler::prepare_duplex_step(const std::string& text, MfqDuplexStepInput& input) const {
    input = std::get<MfqDuplexStepInput>(control(PrepareDuplexStep{text, input}));
}
std::string MfqScheduler::decode_tokens(const std::vector<std::int64_t>& tokens,
        const std::unordered_set<std::int64_t>& excluded) const {
    return std::get<std::string>(control(DecodeTokens{tokens, excluded}));
}

void MfqScheduler::publish(Request& request, EngineEvent event) {
    auto& box = *request.outbox;
    std::unique_lock lock(box.mutex_);
    if (box.terminal_) return;
    const bool end = terminal(event.data);
    request.finished |= end;
    if (request.failure) {
        if (!end) return;
        event.data = *request.failure;
    }
    if (auto* failed = std::get_if<Failed>(&event.data)) {
        failed->code.resize(std::min<std::size_t>(failed->code.size(), 64));
        failed->message.resize(std::min<std::size_t>(failed->message.size(), 512));
    }
    if (auto* completed = std::get_if<Completed>(&event.data))
        completed->finish_reason.resize(std::min<std::size_t>(completed->finish_reason.size(), 64));
    const auto bytes = event_bytes(event.data) + event.id.size();
    if (!end && bytes > limits_.bytes - terminal_reserve) {
        request.failure = Failed{"output_limit", "engine output exceeds the request outbox budget"};
        lock.unlock();
        cancel(request);
        return;
    }
    if (!request.pending.empty() || (!end &&
            (box.events_.size() >= limits_.events - 1 || box.bytes_ + bytes > limits_.bytes - terminal_reserve))) {
        request.pending.push_back(std::move(event));
        return;
    }
    box.bytes_ += bytes;
    box.terminal_ = end;
    box.events_.push_back(std::move(event));
    box.ready_.notify_all();
}
void MfqScheduler::flush(Request& request) {
    auto& box = *request.outbox;
    std::lock_guard lock(box.mutex_);
    while (!request.pending.empty()) {
        auto& event = request.pending.front();
        const bool end = terminal(event.data);
        const auto bytes = event_bytes(event.data) + event.id.size();
        if (!end && (box.events_.size() >= limits_.events - 1 ||
                box.bytes_ + bytes > limits_.bytes - terminal_reserve)) break;
        box.bytes_ += bytes;
        box.terminal_ = end;
        box.events_.push_back(std::move(event));
        request.pending.pop_front();
        box.ready_.notify_all();
    }
}
void MfqScheduler::cancel(Request& request) {
    if (request.finished) {
        if (request.pending.empty()) return;
        auto terminal = std::move(request.pending.back());
        request.pending.clear();
        publish(request, std::move(terminal));
        return;
    }
    if (request.cancelling) return;
    request.cancelling = true;
    request.pending.clear();
    if (request.admitted) engine_.cancel(request.input.id);
    else {
        publish(request, {request.input.id, Cancelled{}});
    }
}
void MfqScheduler::expire(Request& request) {
    // Distinguish an elapsed deadline from a client cancel: the transport
    // reports it as a timeout and the caller never retries an absolute bound.
    // A cancel that already owns the request keeps its own terminal event.
    if (request.cancelling || request.finished) return;
    request.failure = Failed{"deadline_exceeded", "request deadline expired before the engine completed the request"};
    cancel(request);
}

void MfqScheduler::loop() noexcept {
    std::optional<Reload> reload;
    bool healthy = true;
    bool duplex_active = false;
    for (;;) {
        std::deque<Command> commands;
        bool stopping;
        { std::lock_guard lock(mutex_); commands.swap(mailbox_); stopping = stopping_; }
        for (auto& command : commands) std::visit([&](auto& value) {
            using T = std::decay_t<decltype(value)>;
            try {
                if constexpr (std::is_same_v<T, Submit>) {
                    auto& request = value.request;
                    const auto id = request.input.id;
                    if (stopping || !healthy || reload || duplex_active)
                        throw std::runtime_error("scheduler is not accepting requests");
                    if (requests_.contains(id)) throw std::invalid_argument("request ID is already active");
                    const auto& session = request.input.input.cache_plan.session_id;
                    if (!session.empty()) for (const auto& [other_id, other] : requests_)
                        if (other.input.input.cache_plan.session_id == session)
                            throw std::invalid_argument("session already has an active request");
                    order_.push_back(id); requests_.emplace(id, std::move(request));
                    value.reply.set_value();
                } else if constexpr (std::is_same_v<T, Cancel>) {
                    bool found = false;
                    for (auto& [id, request] : requests_) {
                        if (value.all || (value.session ? request.input.input.cache_plan.session_id == value.id : id == value.id)) {
                            cancel(request); found = true;
                        }
                    }
                    value.reply.set_value(found);
                } else if constexpr (std::is_same_v<T, Control>) {
                    if ((std::holds_alternative<ScoreText>(value.request) || std::holds_alternative<ScoreTokens>(value.request)) &&
                        (duplex_active || !requests_.empty()))
                        throw std::runtime_error("probability scoring requires an idle runtime");
                    const bool start_duplex = std::holds_alternative<MfqDuplexSessionParams>(value.request);
                    const bool stop_duplex = std::holds_alternative<StopDuplex>(value.request);
                    if (reload) throw std::runtime_error("reload is pending");
                    if ((std::holds_alternative<MfqDuplexSessionParams>(value.request) ||
                         std::holds_alternative<MfqDuplexStepInput>(value.request) ||
                         std::holds_alternative<StopDuplex>(value.request)) && !requests_.empty())
                        throw std::runtime_error("duplex conflicts with active generation");
                    auto result = engine_.control(std::move(value.request));
                    if (start_duplex) duplex_active = true;
                    if (stop_duplex) duplex_active = false;
                    refresh_snapshot(healthy, duplex_active || stopping, true);
                    value.reply.set_value(std::move(result));
                } else if constexpr (std::is_same_v<T, Session>) {
                    if (value.request.kind != SessionCommand::Kind::metrics) {
                        if (duplex_active || reload) throw std::runtime_error("session operation conflicts with runtime control");
                        for (const auto& [id, request] : requests_) {
                            const auto& session = request.input.input.cache_plan.session_id;
                            if (value.request.kind == SessionCommand::Kind::clear ||
                                value.request.kind == SessionCommand::Kind::memory_budget ||
                                value.request.kind == SessionCommand::Kind::refresh ||
                                ((value.request.kind == SessionCommand::Kind::fork ||
                                  value.request.kind == SessionCommand::Kind::close) &&
                                 (session == value.request.source || session == value.request.target)))
                                throw std::runtime_error("session operation conflicts with active generation");
                        }
                    }
                    auto result = engine_.session(value.request);
                    refresh_snapshot(healthy, reload.has_value() || duplex_active || stopping, true);
                    value.reply.set_value(std::move(result));
                } else if constexpr (std::is_same_v<T, Status>) {
                    auto status = engine_.status();
                    status.healthy &= healthy;
                    if (!status.healthy || reload || duplex_active || stopping) status.available = 0;
                    value.reply.set_value(status);
                } else {
                    if (reload) throw std::runtime_error("reload already pending");
                    if (duplex_active) {
                        (void)engine_.control(StopDuplex{});
                        duplex_active = false;
                    }
                    for (auto& [id, request] : requests_) cancel(request);
                    reload.emplace(std::move(value));
                }
            } catch (const MfqBackendFailure& error) {
                mfq::log_runtime_error("scheduler_control", error.what());
                healthy = false;
                value.reply.set_exception(std::current_exception());
            } catch (const std::exception& error) {
                mfq::log_runtime_error("scheduler_control", error.what());
                value.reply.set_exception(std::current_exception());
            } catch (...) { value.reply.set_exception(std::current_exception()); }
        }, command);
        bool executable = false;
        auto wake_at = Clock::time_point::max();
        try {
            std::vector<RequestId> eligible;
            std::vector<RequestId> ordered(order_.begin(), order_.end());
            std::stable_sort(ordered.begin(), ordered.end(), [&](const auto& a, const auto& b) {
                return requests_.at(a).input.priority > requests_.at(b).input.priority;
            });
            bool admission_blocked = false;
            for (const auto& id : ordered) {
                auto& request = requests_.at(id);
                flush(request);
                if (stopping || !healthy) cancel(request);
                else if (request.input.deadline && *request.input.deadline <= Clock::now()) expire(request);
                if (request.finished) continue;
                if (request.input.deadline && !request.cancelling) wake_at = std::min(wake_at, *request.input.deadline);
                if (request.cancelling) { executable |= request.admitted; continue; }
                if (!request.admitted && !reload && !admission_blocked && engine_.status().available > 0) {
                    try {
                        request.admitted = engine_.admit(request.input) == Admission::accepted;
                        admission_blocked = !request.admitted;
                    }
                    catch (const InferenceInputError& error) {
                        mfq::log_runtime_error("admission", error.what());
                        publish(request, {id, Failed{error.code == InferenceInputErrorCode::Unsupported
                            ? "unsupported_input" : "invalid_request", error.what()}}); continue;
                    } catch (const std::invalid_argument& error) {
                        mfq::log_runtime_error("admission", error.what());
                        publish(request, {id, Failed{"invalid_request", error.what()}}); continue;
                    } catch (const MfqBackendFailure&) {
                        throw;
                    } catch (const MfqResourceExhausted& error) {
                        mfq::log_runtime_error("admission", error.what());
                        publish(request, {id, Failed{"resource_exhausted", error.what(), true}}); continue;
                    } catch (const std::exception& error) {
                        mfq::log_runtime_error("admission", error.what());
                        publish(request, {id, Failed{"backend_failure", error.what()}}); continue;
                    }
                }
                if (!request.admitted) continue;
                if (!request.pending.empty()) continue;
                auto& box = *request.outbox;
                std::lock_guard lock(box.mutex_);
                // Reserve one quantum and a separate terminal slot. No wait or
                // consumer acknowledgement occurs on the execution thread.
                if (!box.terminal_ && box.events_.size() + 3 < limits_.events &&
                        box.bytes_ < (limits_.bytes - terminal_reserve) / 2)
                    eligible.push_back(id);
            }
            executable |= !eligible.empty();
            if (executable) {
                auto result = engine_.step(eligible);
                // Teardown has completed when the engine becomes idle. Publish
                // its final resource snapshot before the terminal response;
                // otherwise a short request can leave prefill usage visible.
                if (!result.has_work && std::any_of(result.events.begin(), result.events.end(),
                        [](const auto& event) { return terminal(event.data); }))
                    refresh_snapshot(healthy && result.status.healthy,
                        reload.has_value() || duplex_active || stopping, true);
                for (auto& event : result.events) {
                    auto found = requests_.find(event.id);
                    if (found != requests_.end()) publish(found->second, std::move(event));
                }
                if (!result.status.healthy) throw std::runtime_error("engine is unhealthy");
                executable = !result.advanced.empty() || !result.events.empty();
                if (result.wake_at) wake_at = std::min(wake_at, *result.wake_at);
            }
        } catch (const std::exception& error) {
            mfq::log_runtime_error("scheduler", error.what());
            healthy &= dynamic_cast<const MfqBackendFailure*>(&error) == nullptr && engine_.status().healthy;
            for (auto& [id, request] : requests_) {
                try { if (request.admitted) engine_.cancel(id); } catch (...) {}
            }
            try { healthy &= engine_.step({}).status.healthy; } catch (...) { healthy = false; }
            for (auto& [id, request] : requests_)
                if (!request.finished)
                    try { publish(request, {id, Failed{"backend_failure", error.what()}}); } catch (...) {}
        } catch (...) {
            mfq::log_runtime_error("scheduler", "unknown backend error");
            healthy &= engine_.status().healthy;
            for (auto& [id, request] : requests_)
                try { engine_.cancel(id); } catch (...) {}
            try { healthy &= engine_.step({}).status.healthy; } catch (...) { healthy = false; }
            for (auto& [id, request] : requests_)
                if (!request.finished)
                    try { publish(request, {id, Failed{"backend_failure", "unknown backend error"}}); } catch (...) {}
        }
        for (auto it = order_.begin(); it != order_.end();) {
            auto box = requests_.at(*it).outbox;
            std::lock_guard lock(box->mutex_);
            if (box->terminal_) { requests_.erase(*it); it = order_.erase(it); }
            else ++it;
        }
        if (reload && requests_.empty()) {
            try {
                const auto context = engine_.reload(reload->context);
                { std::lock_guard lock(mutex_); info_ = engine_.info(); }
                healthy = true; reload->reply.set_value(context);
            } catch (...) { healthy = false; reload->reply.set_exception(std::current_exception()); }
            reload.reset();
        }
        if (!order_.empty()) { order_.push_back(order_.front()); order_.pop_front(); }
        if (stopping && requests_.empty()) {
            try { engine_.shutdown(); } catch (...) {}
            return;
        }
        bool refresh;
        { std::lock_guard lock(mutex_); refresh = std::exchange(snapshot_requested_, false); }
        refresh_snapshot(healthy, reload.has_value() || duplex_active || stopping, !commands.empty() || refresh);
        if (!executable) {
            std::unique_lock lock(mutex_);
            if (mailbox_.empty() && !stopping_) {
                // A timed recheck also covers a drain racing the wait setup.
                wake_->wait_until(lock, std::min(wake_at, Clock::now() + std::chrono::milliseconds(10)));
            }
        }
    }
}
