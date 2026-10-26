#pragma once

#include "generation_step.h"
#include "mfq/fatal_diagnostics.h"

#include <memory>
#include <unordered_map>

namespace mfq::engine {

template <class Cache> SessionResult control_session(Cache &cache, const SessionCommand &command) {
    switch (command.kind) {
    case SessionCommand::Kind::fork:
        return {cache.fork_session(command.source, command.target), {}};
    case SessionCommand::Kind::close:
        return {cache.close_session(command.source), {}};
    case SessionCommand::Kind::clear:
        return {cache.clear(), {}};
    case SessionCommand::Kind::trim:
        return {cache.trim_hot(command.bytes), {}};
    case SessionCommand::Kind::budget:
        if constexpr (requires { cache.set_hot_limit(command.bytes); cache.set_disk_limit(command.bytes); }) {
            if (command.disk_bytes) cache.set_disk_limit(*command.disk_bytes);
            auto released = cache.set_hot_limit(command.bytes);
            if constexpr (requires { cache.set_ram_limit(command.bytes); }) {
                if (command.ram_bytes) released += cache.set_ram_limit(*command.ram_bytes);
            }
            return {released, {}};
        }
        else throw std::invalid_argument("prefix cache budgets are unavailable");
    case SessionCommand::Kind::metrics:
        return {0, cache.metrics()};
    case SessionCommand::Kind::refresh:
        if constexpr (requires { cache.refresh_disk_index(); }) return {cache.refresh_disk_index(), {}};
        else throw std::invalid_argument("prefix cache refresh is unavailable");
    case SessionCommand::Kind::memory_budget:
        throw std::invalid_argument("online resident memory budgets are unavailable for this backend");
    }
    throw std::invalid_argument("unknown session command");
}

// Serial and batched execution write into the same request state. Only the
// executor drains events and finishes the request after device cleanup.
struct ExecutionRequest {
    InferenceRequest input;
    InferenceOutput output;
    Generation generation;
    std::exception_ptr failure;
    bool started = false, batched = false, done = false;

    ExecutionRequest(InferenceRequest prepared, const MfqTokenizer *tokenizer, const RequestId &id)
        : input(std::move(prepared)), output(input, tokenizer, id) {}

    void complete(std::exception_ptr error = {}) {
        done = true;
        failure = error;
    }
};

class RequestExecutor {
  public:
    explicit RequestExecutor(std::size_t capacity = 1) : capacity_(capacity) {}
    bool empty() const { return requests_.empty(); }
    EngineStatus status(bool exclusive = false) const {
        if (!healthy_)
            return {0, false};
        if (exclusive)
            return {0, true};
        for (const auto &[id, request] : requests_)
            if (!request->batched)
                return {0, true};
        return {capacity_ - requests_.size(), true};
    }

    template <class Ops>
    Admission admit(
        EngineRequest request, const TextProcessor *text, const EngineInfo &info, Ops &ops) {
        if (requests_.contains(request.id))
            throw std::invalid_argument("duplicate request ID");
        const bool batched = ops.can_batch(request);
        if (!status(ops.exclusive()).available || (!batched && !empty()))
            return Admission::deferred;
        const bool raw = !request.token_ids.empty();
        InferenceRequest input;
        const MfqTokenizer *tokenizer = nullptr;
        if (raw) {
            input.prompt = std::move(request.token_ids);
            input.sampling = request.input.sampling;
            input.cache_plan = request.input.cache_plan;
            input.stop_token_ids = info.stop_token_ids;
        } else {
            if (!text)
                throw std::invalid_argument("text input requires a tokenizer");
            tokenizer = &text->tokenizer();
            input = text->prepare(std::move(request.input), info.max_context);
        }
        const auto plan = plan_generation(input.prompt,
            info.vocab_size,
            info.max_context,
            input.sampling.max_tokens,
            input.cache_plan.stable_prefix_tokens);
        input.sampling.max_tokens = plan.generation_tokens;
        input.cache_plan.stable_prefix_tokens = plan.stable_prefix_tokens;
        auto current = std::make_unique<ExecutionRequest>(std::move(input), tokenizer, request.id);
        current->output.metrics.mtp.available = ops.mtp_available();
        current->batched = batched;
        requests_.emplace(request.id, std::move(current));
        return Admission::accepted;
    }

    void cancel(const RequestId &id) {
        if (auto it = requests_.find(id); it != requests_.end())
            it->second->output.result.cancelled = true;
    }

    std::exception_ptr clear() {
        std::exception_ptr failure;
        for (auto& [id, request] : requests_) {
            request->generation = {};
            if (request->output.cleanup_failure) failure = request->output.cleanup_failure;
        }
        requests_.clear();
        return failure;
    }

    template <class Ops> EngineStepResult step(const std::vector<RequestId> &eligible, Ops &ops) {
        try {
            ops.execute(eligible);
        } catch (const std::exception& error) {
            mfq::log_runtime_error("execution", error.what());
            const bool backend_failure = dynamic_cast<const MfqBackendFailure*>(&error) != nullptr;
            healthy_ &= !backend_failure;
            for (auto& [id, request] : requests_)
                if (backend_failure || std::find(eligible.begin(), eligible.end(), id) != eligible.end())
                    request->complete(std::current_exception());
        } catch (...) {
            mfq::log_runtime_error("execution", "unknown execution failure");
            for (auto& [id, request] : requests_)
                if (std::find(eligible.begin(), eligible.end(), id) != eligible.end())
                    request->complete(std::current_exception());
        }
        EngineStepResult result;
        for (auto it = requests_.begin(); it != requests_.end();) {
            auto &current = *it->second;
            if (!current.done && (current.output.result.cancelled ||
                std::find(eligible.begin(), eligible.end(), it->first) != eligible.end())) {
                try {
                    if (!current.started) {
                        current.started = true;
                        current.generation = ops.generate(it->first, current);
                    }
                    auto step = current.generation.next();
                    if (step.state == StepState::advanced)
                        result.advanced.push_back(it->first);
                    if (step.value) {
                        if (auto* progress = std::get_if<PrefillProgress>(&*step.value))
                            current.output.metrics.mark_prefill(progress->timing);
                        result.events.push_back({it->first, std::move(*step.value)});
                    }
                    if (!step) current.complete();
                } catch (...) {
                    current.complete(std::current_exception());
                }
            }
            if (!current.done) {
                ++it;
                continue;
            }
            current.generation = {};
            if (current.output.cleanup_failure) {
                healthy_ = false;
                current.failure = current.output.cleanup_failure;
            }
            finish(it->first, current, result);
            it = requests_.erase(it);
        }
        result.status = status(ops.exclusive());
        result.has_work = !empty();
        return result;
    }

  private:
    void finish(const RequestId &id, ExecutionRequest &request, EngineStepResult &result);
    std::size_t capacity_;
    bool healthy_ = true;
    std::unordered_map<RequestId, std::unique_ptr<ExecutionRequest>> requests_;
};

// Backend owns native resources and guards; this composition owns the common
// Engine lifecycle. Destroy request coroutines before their text/model resources.
template <class Backend> class EngineInstance final : public Engine {
  public:
    Backend backend;

    explicit EngineInstance(typename Backend::Options options) : backend{std::move(options)} {
        load();
    }
    EngineInfo info() const override { return info_; }
    EngineStatus status() const override {
        return loaded_ ? requests_.status(backend.exclusive()) : EngineStatus{0, false};
    }
    Admission admit(EngineRequest request) override {
        return backend.visit([&](auto& ops) {
            return requests_.admit(std::move(request), text_.get(), info_, ops);
        });
    }
    void cancel(const RequestId& id) override { requests_.cancel(id); }
    EngineStepResult step(const std::vector<RequestId>& eligible) override {
        return backend.visit([&](auto& ops) { return requests_.step(eligible, ops); });
    }
    SessionResult session(const SessionCommand& command) override {
        if(command.kind == SessionCommand::Kind::memory_budget && !requests_.empty())
            throw std::runtime_error("memory budgeting requires a quiescent engine");
        return backend.visit([&](auto& ops) {
            if constexpr(requires { ops.session(command); })return ops.session(command);
            else return control_session(ops.cache, command);
        });
    }
    ControlResult control(ControlRequest request) override {
        if (!loaded_) throw std::runtime_error("engine is unloaded");
        return std::visit([&](auto&& value) -> ControlResult {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, DecodeTokens> || std::is_same_v<T, CountTokens> ||
                          std::is_same_v<T, PrepareDuplex> || std::is_same_v<T, PrepareDuplexStep>) {
                if (!text_) throw std::invalid_argument("text control requires a tokenizer");
                if constexpr (std::is_same_v<T, DecodeTokens>)
                    return text_->decode_tokens(value.tokens, value.excluded);
                else if constexpr (std::is_same_v<T, CountTokens>)
                    return TokenCount{text_->count_tokens(value.prompt), info_.max_context};
                else if constexpr (std::is_same_v<T, PrepareDuplex>) {
                    text_->prepare_duplex_session(value.prompt, value.parameters);
                    return std::move(value.parameters);
                } else {
                    text_->prepare_duplex_step(value.text, value.input);
                    return std::move(value.input);
                }
            } else
                return backend.visit([&](auto& ops) { return ops.control(std::move(value)); });
        }, std::move(request));
    }
    std::int64_t reload(std::int64_t context) override {
        if (context < 1) throw std::invalid_argument("reload context must be positive");
        if (!requests_.empty()) throw std::runtime_error("reload requires a quiescent engine");
        shutdown();
        backend.options.context_size = context;
        load();
        return info_.max_context;
    }
    void shutdown() override {
        loaded_ = false;
        auto failure = requests_.clear();
        text_.reset();
        backend.unload();
        if (failure) std::rethrow_exception(failure);
    }

  private:
    std::unique_ptr<TextProcessor> text_;
    EngineInfo info_;
    RequestExecutor requests_;
    bool loaded_ = false;

    void load() {
        try {
            auto [info, text] = backend.load();
            info_ = std::move(info);
            text_ = std::move(text);
            if (text_) info_.chat = text_->chat_template_capabilities();
            requests_ = RequestExecutor(info_.max_requests);
            loaded_ = true;
        } catch (...) {
            shutdown();
            throw;
        }
    }
};

} // namespace mfq::engine
