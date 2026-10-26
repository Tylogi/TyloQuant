#include "request_executor.h"
#include "mfq/fatal_diagnostics.h"

namespace mfq::engine {

void RequestExecutor::finish(
    const RequestId &id, ExecutionRequest &request, EngineStepResult &result) {
    try {
        if (request.failure)
            std::rethrow_exception(request.failure);
        auto delta = request.output.finish();
        if (!delta.diffs.empty())
            result.events.push_back({id, std::move(delta)});
        const UsageUpdate usage{request.input.prompt.size(),
            static_cast<std::size_t>(request.output.result.completion_tokens)};
        result.events.push_back({id, usage});
        if (request.output.result.cancelled)
            result.events.push_back({id, Cancelled{usage, request.output.metrics}});
        else
            result.events.push_back({id,
                Completed{request.output.result.finish_reason, usage, request.output.metrics}});
    } catch (const MfqResourceExhausted &error) {
        mfq::log_runtime_error("request", error.what());
        result.events.push_back({id, Failed{healthy_ ? "resource_exhausted" : "backend_failure", error.what(), healthy_}});
    } catch (const InferenceInputError &error) {
        mfq::log_runtime_error("request", error.what());
        const auto code = !healthy_                                            ? "backend_failure"
                          : error.code == InferenceInputErrorCode::Unsupported ? "unsupported_input"
                                                                               : "invalid_request";
        result.events.push_back({id, Failed{code, error.what()}});
    } catch (const std::invalid_argument &error) {
        mfq::log_runtime_error("request", error.what());
        result.events.push_back(
            {id, Failed{healthy_ ? "invalid_request" : "backend_failure", error.what()}});
    } catch (const MfqBackendFailure &error) {
        mfq::log_runtime_error("request", error.what());
        healthy_ = false;
        result.events.push_back({id, Failed{"backend_failure", error.what()}});
    } catch (const std::exception &error) {
        mfq::log_runtime_error("request", error.what());
        result.events.push_back({id, Failed{"backend_failure", error.what()}});
    } catch (...) {
        mfq::log_runtime_error("request", "unknown execution failure");
        result.events.push_back({id, Failed{"backend_failure", "unknown execution failure"}});
    }
}

} // namespace mfq::engine
