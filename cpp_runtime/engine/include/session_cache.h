#pragma once

#include "mfq_paged_prefix_cache.h"
#include "paged_session_bindings.h"
#include "session_snapshot_cache.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace mfq::engine {

// Matching, eviction and restore/store ordering are shared. Backend supplies
// native snapshot/predictor types and the paged payload codec.
template <class Backend> class SessionCache {
    using Snapshot = typename Backend::Snapshot;
    using Restore = typename Backend::Restore;
    using Predictor = typename Backend::Predictor;
    using StateError = typename Backend::StateError;

  public:
    template <class Visitor> void visit_snapshots(Visitor&& visitor) const {
        snapshots_.visit_states(std::forward<Visitor>(visitor));
    }
    template <class SessionConfig, class PrefixConfig>
    explicit SessionCache(const SessionConfig &session_config, const PrefixConfig &prefix_config,
        std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache = {}, bool supported = true,
        int disabled_reason = 0, bool native_hot_snapshots = false)
        : snapshots_([&] {
              auto configuration = session_config.snapshots;
              if (native_hot_snapshots) configuration.max_bytes = prefix_config.enabled ? prefix_config.hot_bytes : 0;
              return configuration;
          }()), paged_cache_(std::move(paged_cache)),
          paged_bindings_(paged_cache_, snapshots_.max_sessions()),
          paged_disk_budget_(paged_cache_ ? prefix_config.disk_bytes : 0),
          paged_hot_budget_(paged_cache_ ? prefix_config.hot_bytes : 0),
          paged_ram_budget_(paged_cache_ && native_hot_snapshots ? prefix_config.ram_bytes : 0),
          trace_(session_config.trace), supported_(supported), disabled_reason_(disabled_reason),
          native_hot_snapshots_(native_hot_snapshots) {}

    bool persistent_prefix_enabled() const noexcept {
        return static_cast<bool>(paged_cache_) || (native_hot_snapshots_ && supported_ && snapshots_.enabled());
    }

    template <typename Model>
    Restore restore_best(Model &model, Predictor *mtp, const std::string &requested_session,
        const std::vector<int64_t> &prompt, size_t maximum_prefix_tokens,
        const std::string &input_key = {}) {
        if (!supported_)
            return {};
        if (paged_cache_ && !native_hot_snapshots_) {
            auto restored=restore_paged(model,requested_session,prompt,maximum_prefix_tokens,mtp,input_key);
            if(restored.tokens)return restored;
        }
        if (!model.supports_text_session_state())
            return {};
        auto match = snapshots_.find_best(
            native_hot_snapshots_ && requested_session.empty() ? "\x01prefix-global" : requested_session,
            prompt, maximum_prefix_tokens, [&](const Snapshot &state) {
            return state.input_key == input_key &&
                   (mtp == nullptr || (mtp->supports_session_state() && state.mtp.has_value()));
        });
        if (!match) {
            if (paged_cache_ && native_hot_snapshots_)
                return restore_paged(model, requested_session, prompt, maximum_prefix_tokens, mtp, input_key);
            return {};
        }
        try {
            model.restore_text_session_state(*match->state);
            Restore restored{match->tokens(), {}};
            if (mtp != nullptr) {
                mtp->restore_session_state(*match->state->mtp);
                restored.mtp_last_target_hidden = match->state->mtp->last_target_hidden;
            }
            snapshots_.record_hit(*match);
            if (paged_cache_ && native_hot_snapshots_) paged_cache_->record_match(match->tokens());
            if (trace_) {
                std::cerr << "runtime_session_cache action=hit session=" << requested_session
                          << " source=" << match->session_id << " reused_tokens=" << match->tokens()
                          << " prefill_tokens=" << prompt.size() - match->tokens() << std::endl;
            }
            return restored;
        } catch (const std::exception &error) {
            const auto selected_session = match->session_id;
            snapshots_.erase(*match);
            model.reset(1);
            if (mtp != nullptr)
                mtp->reset(1);
            std::cerr << "runtime_session_cache action=invalidate session=" << selected_session
                      << " error=" << error.what() << std::endl;
            return {};
        }
    }

    void store(const std::string &session_id, Snapshot state) {
        if (!supported_)
            return;
        bool paged_mtp=false;
        if constexpr(requires { Backend::paged_mtp(state); })paged_mtp=Backend::paged_mtp(state);
        bool paged_media=state.input_key.empty();
        if constexpr(requires { Backend::paged_media(state); })paged_media|=Backend::paged_media(state);
        if (paged_cache_ && paged_media && (!state.mtp.has_value() || paged_mtp)) {
            store_paged(session_id, state);
            if (!native_hot_snapshots_) return;
        }
        if ((!native_hot_snapshots_ && session_id.empty()) || !snapshots_.enabled())
            return;
        if (state.bytes > snapshots_.max_bytes()) {
            if (trace_) {
                std::cerr << "runtime_session_cache action=skip session=" << session_id
                          << " bytes=" << state.bytes << " budget=" << snapshots_.max_bytes()
                          << std::endl;
            }
            return;
        }
        const auto stored = snapshots_.store(
            native_hot_snapshots_ && session_id.empty() ? "\x01prefix-" + std::to_string(++anonymous_snapshots_) : session_id,
            std::move(state), [](const Snapshot &saved, const Snapshot &candidate) {
            return saved.tokens == candidate.tokens && saved.input_key == candidate.input_key;
        });
        if (trace_ && stored) {
            std::cerr << "runtime_session_cache action=store session=" << session_id
                      << " tokens=" << stored.state->tokens.size()
                      << " bytes=" << stored.state->bytes
                      << " snapshots=" << stored.session_snapshots
                      << " total_bytes=" << stored.total_bytes << std::endl;
        }
    }

    size_t fork_session(const std::string &source_session, const std::string &target_session) {
        const auto copied_paged = paged_bindings_.fork(source_session, target_session);
        const auto copied_snapshots = snapshots_.fork(source_session, target_session);
        if (trace_ && copied_snapshots > 0) {
            std::cerr << "runtime_session_cache action=fork source=" << source_session
                      << " target=" << target_session << " snapshots=" << copied_snapshots
                      << " total_bytes=" << snapshots_.metrics().bytes << std::endl;
        }
        return copied_snapshots + copied_paged;
    }

    size_t close_session(const std::string &session_id) {
        const auto released_paged = paged_bindings_.close(session_id);
        const auto released = snapshots_.close(session_id);
        if (trace_ && released.snapshots > 0) {
            std::cerr << "runtime_session_cache action=close session=" << session_id
                      << " snapshots=" << released.snapshots << " bytes=" << released.bytes
                      << " total_bytes=" << snapshots_.metrics().bytes << std::endl;
        }
        return released.snapshots + released_paged;
    }

    std::vector<std::pair<std::string, double>> metrics() const {
        if (paged_cache_) {
            const auto value = paged_cache_->metrics();
            const auto native = snapshots_.metrics();
            std::vector<std::pair<std::string, double>> result{
                {"prefix_cache_supported", supported_ ? 1.0 : 0.0},
                {"prefix_cache_disabled_reason", static_cast<double>(disabled_reason_)},
                {"prefix_cache_queries", static_cast<double>(value.queries)},
                {"prefix_cache_hits", static_cast<double>(value.hits)},
                {"prefix_cache_hit_tokens", static_cast<double>(value.hit_tokens)},
                {"prefix_cache_sessions", static_cast<double>(paged_bindings_.sessions())},
                {"prefix_cache_snapshots", static_cast<double>(value.blocks ? value.blocks : native.snapshots)},
                {"prefix_cache_tokens", static_cast<double>(paged_bindings_.tokens())},
                {"prefix_cache_bytes", static_cast<double>(native_hot_snapshots_ ? native.bytes : value.hot_bytes)},
                {"prefix_cache_max_sessions", static_cast<double>(snapshots_.max_sessions())},
                {"prefix_cache_max_snapshots_per_session", 1.0},
                {"prefix_cache_max_bytes", static_cast<double>(paged_hot_budget_)},
                {"prefix_cache_disk_blocks", static_cast<double>(value.disk_blocks)},
                {"prefix_cache_disk_bytes", static_cast<double>(value.disk_bytes)},
                {"prefix_cache_disk_max_bytes", static_cast<double>(paged_disk_budget_)},
                {"prefix_cache_hot_blocks", static_cast<double>(native_hot_snapshots_ ? native.snapshots : value.hot_blocks)},
                {"prefix_cache_hot_bytes", static_cast<double>(native_hot_snapshots_ ? native.bytes : value.hot_bytes)},
                {"prefix_cache_native_hot", native_hot_snapshots_ ? 1.0 : 0.0},
                {"prefix_cache_ram_bytes", static_cast<double>(native_hot_snapshots_ ? value.resident_bytes : value.hot_bytes)},
                {"prefix_cache_ram_max_bytes", static_cast<double>(paged_ram_budget_)},
                {"prefix_cache_ram_blocks", static_cast<double>(native_hot_snapshots_ ? value.hot_blocks : 0)},
                {"prefix_cache_ram_hits", static_cast<double>(native_hot_snapshots_ ? value.hot_hits : 0)},
                {"prefix_cache_hot_pressure_bytes", static_cast<double>(native_hot_snapshots_ ? 0 : value.hot_pressure_bytes)},
                {"prefix_cache_ram_pressure_bytes", static_cast<double>(native_hot_snapshots_ ? value.hot_pressure_bytes : 0)},
                {"prefix_cache_dynamic_budget", 1.0},
                {"prefix_cache_pending_writes", static_cast<double>(value.pending_writes)},
                {"prefix_cache_pending_bytes", static_cast<double>(value.pending_bytes)},
                {"prefix_cache_pending_max_bytes", static_cast<double>(value.pending_max_bytes)},
                {"prefix_cache_writes", static_cast<double>(value.writes)},
                {"prefix_cache_deduplicated_writes",
                    static_cast<double>(value.deduplicated_writes)},
                {"prefix_cache_disk_hits", static_cast<double>(value.disk_hits)},
                {"prefix_cache_hot_hits", static_cast<double>(native_hot_snapshots_ ? native.hits : value.hot_hits)},
                {"prefix_cache_evictions", static_cast<double>(value.evictions)},
                {"prefix_cache_corrupt_blocks", static_cast<double>(value.corrupt_blocks)},
                {"prefix_cache_low_disk_space_skips", static_cast<double>(value.low_disk_space_skips)},
                {"prefix_cache_failed_writes", static_cast<double>(value.failed_writes)},
            };
            if (native_hot_snapshots_)
                result.emplace_back("prefix_cache_resident_sessions", static_cast<double>(native.sessions));
            return result;
        }
        const auto value = snapshots_.metrics();
        std::vector<std::pair<std::string, double>> result{
            {"prefix_cache_supported", supported_ ? 1.0 : 0.0},
            {"prefix_cache_disabled_reason", static_cast<double>(disabled_reason_)},
            {"prefix_cache_queries", static_cast<double>(value.queries)},
            {"prefix_cache_hits", static_cast<double>(value.hits)},
            {"prefix_cache_hit_tokens", static_cast<double>(value.hit_tokens)},
            {"prefix_cache_sessions", static_cast<double>(value.sessions)},
            {"prefix_cache_snapshots", static_cast<double>(value.snapshots)},
            {"prefix_cache_tokens", static_cast<double>(value.tokens)},
            {"prefix_cache_bytes", static_cast<double>(value.bytes)},
            {"prefix_cache_max_sessions", static_cast<double>(snapshots_.max_sessions())},
            {"prefix_cache_max_snapshots_per_session",
                static_cast<double>(snapshots_.max_snapshots_per_session())},
            {"prefix_cache_max_bytes", static_cast<double>(snapshots_.max_bytes())},
        };
        if (native_hot_snapshots_) {
            result.insert(result.end(), {{"prefix_cache_native_hot", 1.0}, {"prefix_cache_dynamic_budget", 1.0},
                {"prefix_cache_hot_bytes", static_cast<double>(value.bytes)},
                {"prefix_cache_hot_blocks", static_cast<double>(value.snapshots)},
                {"prefix_cache_resident_sessions", static_cast<double>(value.sessions)},
                {"prefix_cache_ram_bytes", 0.0}});
        }
        return result;
    }

    size_t clear_live_sessions() noexcept { return snapshots_.clear() + paged_bindings_.clear(); }

    size_t clear() {
        const auto snapshots = snapshots_.clear();
        paged_bindings_.clear();
        return snapshots + (paged_cache_ ? paged_cache_->clear() : 0);
    }

    uint64_t trim_hot(uint64_t target_bytes) {
        if (native_hot_snapshots_) return snapshots_.trim(target_bytes);
        if (!paged_cache_)
            return 0;
        const auto released = paged_cache_->trim_hot(target_bytes);
        if (released > 0)
            Backend::release_host_cache();
        return released;
    }

    uint64_t set_hot_limit(uint64_t max_bytes) {
        if (native_hot_snapshots_) {
            paged_hot_budget_ = max_bytes;
            return snapshots_.set_limit(max_bytes);
        }
        if (!paged_cache_) return 0;
        paged_hot_budget_ = max_bytes;
        const auto released = paged_cache_->set_hot_limit(max_bytes);
        if (released > 0) Backend::release_host_cache();
        return released;
    }

    uint64_t set_disk_limit(uint64_t max_bytes) {
        if (!paged_cache_) return 0;
        paged_disk_budget_ = max_bytes;
        return paged_cache_->set_disk_limit(max_bytes);
    }

    uint64_t set_ram_limit(uint64_t max_bytes) {
        if (!paged_cache_ || !native_hot_snapshots_) return 0;
        paged_ram_budget_ = max_bytes;
        const auto released = paged_cache_->set_hot_limit(max_bytes);
        if (released) Backend::release_host_cache();
        return released;
    }

    void reserve_ram(uint64_t bytes) {
        if (paged_cache_ && native_hot_snapshots_ && paged_cache_->reserve_ram(bytes))
            Backend::release_host_cache();
    }

    uint64_t refresh_disk_index() {
        if (!paged_cache_) return 0;
        const auto removed = paged_cache_->refresh_disk_index();
        if (removed > 0) Backend::release_host_cache();
        return removed;
    }

  private:
    template <typename Model>
    Restore restore_paged(Model &model, const std::string &requested_session,
        const std::vector<int64_t> &prompt, size_t maximum_prefix_tokens, Predictor* mtp,
        const std::string& input_key = {}) {
        if (snapshots_.max_sessions() == 0 || !model.supports_paged_text_session_state() ||
            prompt.size() < 2) {
            return {};
        }
        const auto limit = std::min(maximum_prefix_tokens, prompt.size() - 1);
        std::vector<int64_t> candidate(
            prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(limit));
        auto match = paged_cache_->match(candidate, input_key, false);
        if (match.matched_tokens == 0) {
            paged_cache_->record_match(0);
            return {};
        }

        auto payloads = paged_cache_->load_prefix(match.blocks);
        if (payloads.empty()) {
            paged_cache_->record_match(0);
            return {};
        }
        const auto fail = [&](const char *action, const std::exception &error, bool invalidate) {
            if (invalidate && !match.blocks.empty()) {
                paged_cache_->invalidate(match.blocks.back());
            }
            paged_cache_->record_match(0);
            model.reset(1);
            std::cerr << "runtime_session_cache action=" << action
                      << " session=" << requested_session << " error=" << error.what() << std::endl;
            if(mtp)mtp->reset(1);
            return Restore{};
        };
        const char *invalid_action = "paged_codec_invalidate";
        const char *failure_action = "paged_codec_failed";
        try {
            if constexpr(requires { Backend::decodable_blocks(payloads); })
                payloads.resize(Backend::decodable_blocks(payloads));
            if(payloads.empty()) { paged_cache_->record_match(0);return {}; }
            if (payloads.size() != match.blocks.size()) {
                match.blocks.resize(payloads.size());
                match.matched_tokens = payloads.size() * paged_cache_->block_size_tokens();
            }
            if constexpr(requires { Backend::token_count(payloads.front()); }) {
                match.matched_tokens=0;
                for(const auto& payload:payloads) {
                    const auto count = Backend::token_count(payload);
                    if (count == 0 || count > paged_cache_->block_size_tokens() ||
                        count > limit - match.matched_tokens)
                        throw StateError("invalid paged prefix token count");
                    match.matched_tokens += count;
                }
            }
            if (match.matched_tokens > limit)
                throw StateError("paged prefix exceeds the requested prefix");
            std::vector<int64_t> matched_tokens(
                prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(match.matched_tokens));
            auto state =
                Backend::decode(payloads, matched_tokens, paged_cache_->block_size_tokens());
            state.input_key=input_key;
            if(mtp && (!mtp->supports_session_state() || !state.mtp)) {
                paged_cache_->record_match(0);return {};
            }
            invalid_action = "paged_restore_invalidate";
            failure_action = "paged_restore_failed";
            model.restore_text_session_state(state);
            Restore restored{match.matched_tokens,{}};
            if(mtp) {
                mtp->restore_session_state(*state.mtp);
                restored.mtp_last_target_hidden=state.mtp->last_target_hidden;
            }
            if (native_hot_snapshots_)
                snapshots_.store(requested_session.empty() ? "\x01prefix-" + std::to_string(++anonymous_snapshots_) : requested_session,
                    std::move(state), [](const Snapshot &saved, const Snapshot &candidate) {
                    return saved.tokens == candidate.tokens && saved.input_key == candidate.input_key;
                });
            if (!requested_session.empty()) {
                paged_bindings_.bind(requested_session, match.blocks, match.matched_tokens);
            }
            paged_cache_->record_match(match.matched_tokens);
            if (trace_) {
                std::cerr << "runtime_session_cache action=paged_hit "
                          << "session=" << requested_session
                          << " reused_tokens=" << match.matched_tokens
                          << " prefill_tokens=" << prompt.size() - match.matched_tokens
                          << std::endl;
            }
            return restored;
        } catch (const StateError &error) {
            return fail(invalid_action, error, true);
        } catch (const std::exception &error) {
            return fail(failure_action, error, false);
        }
    }

    void store_paged(const std::string &session_id, const Snapshot &state) {
        if (snapshots_.max_sessions() == 0 || state.tokens.empty()) {
            return;
        }
        const auto block_size = paged_cache_->block_size_tokens();
        bool partial=false;
        if constexpr(requires { Backend::partial_blocks(state); })partial=Backend::partial_blocks(state);
        const auto full_blocks = state.tokens.size() / block_size + (partial && state.tokens.size()%block_size ? 1 : 0);
        if (full_blocks == 0)
            return;

        auto existing = paged_cache_->match(state.tokens, state.input_key, false);
        // A shorter tail cannot become the parent of a subsequent full block.
        if(existing.matched_tokens!=state.tokens.size())
            existing.blocks.resize(std::min(existing.blocks.size(),existing.matched_tokens/block_size));
        if (existing.blocks.size() > full_blocks) {
            throw std::runtime_error("paged prefix match exceeds the session state");
        }
        auto blocks = std::move(existing.blocks);
        const bool final_already_stored=blocks.size()==full_blocks;
        mfq::cache::BlockHash parent{};
        if (!blocks.empty())
            parent = blocks.back();
        for (size_t index = blocks.size(); index < full_blocks; ++index) {
            const auto token_offset = index * block_size;
            const auto count=std::min(block_size,state.tokens.size()-token_offset);
            auto payload = Backend::encode(state, block_size, index);
            parent = paged_cache_->store(
                parent, state.tokens.data() + token_offset, count, std::move(payload),state.input_key);
            blocks.push_back(parent);
        }
        if constexpr(requires { Backend::checkpoint_needs_update(mfq::cache::PagedPrefixPayload{},state); }) {
            if(final_already_stored) {
                auto final=paged_cache_->load_prefix({blocks.back()});
                if(final.empty() || Backend::checkpoint_needs_update(final.front(),state)) {
                    const auto index=full_blocks-1,offset=index*block_size;
                    const auto previous=index?blocks[index-1]:mfq::cache::BlockHash{};
                    const auto refreshed=paged_cache_->replace(previous,state.tokens.data()+offset,
                        std::min(block_size,state.tokens.size()-offset),
                        Backend::encode(state,block_size,index),state.input_key);
                    if(refreshed!=blocks[index])throw std::runtime_error("prefix checkpoint changed block identity");
                }
            }
        }
        if (!session_id.empty()) {
            paged_bindings_.bind(session_id, std::move(blocks), partial?state.tokens.size():full_blocks * block_size);
        }
        if (trace_) {
            std::cerr << "runtime_session_cache action=paged_store "
                      << "session=" << session_id << " tokens=" << (partial ? state.tokens.size() : full_blocks * block_size)
                      << " blocks=" << full_blocks << std::endl;
        }
    }

    mfq::engine::SessionSnapshotCache<Snapshot> snapshots_;
    std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache_;
    mfq::engine::PagedSessionBindings paged_bindings_;
    uint64_t paged_disk_budget_ = 0;
    uint64_t paged_hot_budget_ = 0;
    uint64_t paged_ram_budget_ = 0;
    bool trace_ = false;
    bool supported_ = true;
    int disabled_reason_ = 0;
    bool native_hot_snapshots_ = false;
    std::uint64_t anonymous_snapshots_ = 0;
};

} // namespace mfq::engine
