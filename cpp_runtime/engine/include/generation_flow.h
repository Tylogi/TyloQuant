#pragma once

#include "generation_step.h"

#include <algorithm>
#include <iostream>

namespace mfq::engine {

inline bool batch_compatible(const EngineRequest &request, bool persistent_prefix, bool has_mtp) {
    const auto &input = request.input;
    return !input.media && input.cache_plan.session_id.empty() &&
           !((request.token_ids.empty() || input.cache_plan.stable_prefix_tokens) &&
               persistent_prefix) &&
           !(input.sampling.enable_mtp && has_mtp);
}

template <class Ops>
Generation generate_prepared(Ops &ops, InferenceRequest &input, InferenceOutput &output,
                             const RequestId& id, bool batched) {
    std::optional<typename Ops::Prepared> prepared;
    if (input.vision && !output.stopped()) {
        auto preparation = ops.prepare(input);
        bool reclaimed=false;
        while (!output.stopped()) {
            mfq::StepResult<typename Ops::Prepared> step;
            double elapsed=0;
            std::exception_ptr failure;
            try {
                auto result=ops.advance_preparation(preparation);
                step=std::move(result.first);
                elapsed=result.second;
            } catch(...) { failure=std::current_exception(); }
            if(failure) {
                preparation={};
                if constexpr(requires { ops.recover_preparation(failure,reclaimed); }) {
                    ops.recover_preparation(failure,reclaimed);
                    reclaimed=true;
                    preparation=ops.prepare(input);
                    continue;
                } else std::rethrow_exception(failure);
            }
            output.metrics.multimodal_ms += elapsed;
            if (step.value) {
                prepared = std::move(*step.value);
                break;
            }
            if (!step) throw std::runtime_error("media preparation returned no prompt");
            co_yield step.state;
        }
        if (output.stopped()) co_return;
        const auto elapsed = output.metrics.multimodal_ms;
        co_yield PrefillProgress{{0, 0.0, elapsed, elapsed}};
    }
    if (output.stopped()) co_return;
    auto sequence = ops.generate_text(input, output, std::move(prepared), id, batched);
    while (auto step = sequence.next()) {
        auto& event = step.value;
        if (auto *progress = event ? std::get_if<PrefillProgress>(&*event) : nullptr) {
            progress->timing.multimodal_ms = output.metrics.multimodal_ms;
            progress->timing.model_ms += progress->timing.multimodal_ms;
        }
        co_yield std::move(step);
    }
}

// Session transactions surround both ordinary and speculative execution.
// Ops owns native prompt/hidden storage and constructs device execution objects.
template <class Ops>
Generation generate_request(Ops ops, InferenceRequest &request, InferenceOutput &output) {
    using Hidden = typename Ops::Hidden;
    auto &model = ops.model;
    auto &session_cache = ops.cache;
    auto *mtp = ops.mtp;
    const auto &prepared = ops.prepared;
    const auto &config = ops.config;
    const auto &prompt = request.prompt;
    auto &sampling = request.sampling;
    const auto &cache_plan = request.cache_plan;
    if (config.generation.prefill_chunk_size <= 0)
        throw std::invalid_argument("prefill chunk size must be positive");
    if (prepared && prepared->token_ids != prompt)
        throw std::invalid_argument("prepared prompt token IDs disagree");
    const auto occupied =
        static_cast<int64_t>(prompt.size()) +
        (prepared && prepared->transformed() ? prepared->decode_position_delta : 0);
    const auto plan = plan_generation(prompt,
        model.vocab_size(),
        model.max_position_embeddings(),
        sampling.max_tokens,
        cache_plan.stable_prefix_tokens,
        std::max<int64_t>(prompt.size(), occupied));
    sampling.max_tokens = plan.generation_tokens;
    if (output.stopped())
        co_return;
    const std::string input_key = prepared ? prepared->cache_key : std::string{};
    const bool caching =
        !ops.batched() && model.supports_text_session_state() &&
        (!prepared || !prepared->transformed() || !input_key.empty()) &&
        plan.stable_prefix_tokens > 0 &&
        (!cache_plan.session_id.empty() || session_cache.persistent_prefix_enabled());
    const bool use_mtp = mtp && sampling.enable_mtp && sampling.max_tokens > 1;
    if constexpr(requires { ops.set_prefix_boundary(size_t{}); })
        ops.set_prefix_boundary(caching?std::min(plan.stable_prefix_tokens,prompt.size()-1):0);
    if (mtp) {
        mtp->last_stats = {};
        mtp->last_stats.available = true;
    }
    ExecutionCleanup cleanup{output.cleanup_failure, [&] { ops.release(); }};
    if constexpr(requires { ops.prepare_request(request); })ops.prepare_request(request);
    typename Ops::Restore restored;
    const bool restore_prefix=[&] {
        if constexpr(requires { ops.allow_prefix_restore(); })return ops.allow_prefix_restore();
        return true;
    }();
    if (caching && restore_prefix && (!use_mtp || mtp->supports_session_state()))
        restored = session_cache.restore_best(model,
            use_mtp ? mtp : nullptr,
            cache_plan.session_id,
            prompt,
            plan.stable_prefix_tokens,
            input_key);
    if (restored.tokens || use_mtp)
        ops.invalidate_plan();
    if (!restored.tokens) ops.reset();
    std::vector<int64_t> history = prompt;
    Hidden last_target_hidden;
    std::size_t last_snapshot = 0;
    bool initial_snapshot_attempted = false;
    const auto snapshot = [&] {
        const auto position = model.cache_pos;
        if (!caching || output.result.cancelled || position <= 0 ||
            static_cast<std::size_t>(position) > history.size() ||
            static_cast<std::size_t>(position) == last_snapshot)
            return;
        try {
            if constexpr(requires { ops.prepare_snapshot(); })
                if(!ops.prepare_snapshot())return;
            std::vector<int64_t> tokens(history.begin(), history.begin() + position);
            auto state = model.capture_text_session_state(tokens);
            state.input_key = input_key;
            if (use_mtp) {
                if (!ops.has_hidden(last_target_hidden) || position <= 1)
                    return;
                state.mtp = mtp->capture_session_state(position, last_target_hidden);
                state.bytes += state.mtp->bytes;
            }
            session_cache.store(cache_plan.session_id, std::move(state));
            last_snapshot = static_cast<std::size_t>(position);
        } catch (const std::exception &error) {
            std::cerr << "runtime_session_cache action=skip error=" << error.what() << '\n';
        }
    };
    auto sequence = use_mtp
        ? ops.speculate(request, output, restored.tokens, restored.mtp_last_target_hidden,
                        &last_target_hidden)
        : ops.plain(request, output, restored.tokens, caching ? plan.stable_prefix_tokens : 0);
    bool traced_decode_boundary = false;
    while (auto step = sequence.next()) {
        auto& event = step.value;
        if (auto *progress = event ? std::get_if<PrefillProgress>(&*event) : nullptr;
            progress && !use_mtp && caching &&
            progress->timing.prompt_tokens + restored.tokens == plan.stable_prefix_tokens)
            snapshot();
        if (auto *progress = event ? std::get_if<PrefillProgress>(&*event) : nullptr;
            progress && progress->session_boundary && use_mtp && caching &&
            !initial_snapshot_attempted) {
            initial_snapshot_attempted = true;
            snapshot();
        }
        if (auto *delta = event ? std::get_if<OutputDelta>(&*event) : nullptr) {
            if(!traced_decode_boundary && !delta->token_ids.empty()) {
                if constexpr(requires { ops.trace_expert_cache("decode_begin", false); })
                    ops.trace_expert_cache("decode_begin", false);
                traced_decode_boundary = true;
            }
            history.insert(history.end(), delta->token_ids.begin(), delta->token_ids.end());
            // Compatibility with speculative producers lacking an explicit
            // pre-token prompt boundary event.
            if (use_mtp && caching && !initial_snapshot_attempted) {
                initial_snapshot_attempted = true;
                snapshot();
            }
        }
        co_yield std::move(step);
    }
    // Flush parser-held text before a final session snapshot can copy or evict
    // a large cache. The executor's terminal finish remains idempotent and
    // still reports cleanup/cache completion after this final text event.
    if(caching) {
        auto tail=output.finish();
        if(!tail.diffs.empty())co_yield std::move(tail);
    }
    snapshot();
    cleanup.finish();
}

} // namespace mfq::engine
