#include "mlx_mtp.h"

#include "mlx_eval_timing.h"
#include "mlx_sampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using mlx::core::CompileOptions;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr const char* kCompactStochasticVerifySource = R"METAL(
    if (thread_position_in_grid.x != 0u) {
        return;
    }

    int accepted = 0;
    for (uint position = 0u; position < uint(DRAFTS); ++position) {
        int draft = draft_tokens[position];
        uint offset = position * uint(TOP_K);
        float q = 0.0f;
        float p = 0.0f;
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            if (proposal_indices[offset + rank] == draft) {
                q = proposal_probabilities[offset + rank];
            }
            if (target_indices[offset + rank] == draft) {
                p = target_probabilities[offset + rank];
            }
        }
        float roll = clamp(random[position], 0.0f, 0.99999994f);
        bool keep = q > 0.0f && (p >= q || roll < p / q);
        if (!keep) {
            break;
        }
        ++accepted;
    }

    float sample_uniform = clamp(
        random[uint(DRAFTS)], 0.0f, 0.99999994f);
    int chosen = -1;

    if (accepted == int(DRAFTS)) {
        uint bonus_offset = uint(DRAFTS) * uint(TOP_K);
        float total = 0.0f;
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            total += target_probabilities[bonus_offset + rank];
        }
        float threshold = sample_uniform * total;
        float cumulative = 0.0f;
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            float mass = target_probabilities[bonus_offset + rank];
            if (mass > 0.0f) {
                chosen = target_indices[bonus_offset + rank];
                cumulative += mass;
                if (cumulative >= threshold) {
                    break;
                }
            }
        }
    } else {
        float residual[TOP_K];
        float total = 0.0f;
        uint offset = uint(accepted) * uint(TOP_K);
        for (uint target_rank = 0u;
             target_rank < uint(TOP_K);
             ++target_rank) {
            int token = target_indices[offset + target_rank];
            float q_token = 0.0f;
            for (uint proposal_rank = 0u;
                 proposal_rank < uint(TOP_K);
                 ++proposal_rank) {
                if (proposal_indices[offset + proposal_rank] == token) {
                    q_token = proposal_probabilities[offset + proposal_rank];
                    break;
                }
            }
            float mass = max(
                target_probabilities[offset + target_rank] - q_token,
                0.0f);
            residual[target_rank] = mass;
            total += mass;
        }
        bool fallback = !(total > 0.0f) || !isfinite(total);
        if (fallback) {
            total = 0.0f;
            for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
                total += target_probabilities[offset + rank];
            }
        }
        float threshold = sample_uniform * total;
        float cumulative = 0.0f;
        for (uint rank = 0u; rank < uint(TOP_K); ++rank) {
            float mass = fallback
                ? target_probabilities[offset + rank]
                : residual[rank];
            if (mass > 0.0f) {
                chosen = target_indices[offset + rank];
                cumulative += mass;
                if (cumulative >= threshold) {
                    break;
                }
            }
        }
    }

    output[0] = accepted;
    output[1] = chosen;
    for (uint position = 0u; position < uint(DRAFTS); ++position) {
        output[2u + position] = draft_tokens[position];
    }
)METAL";

const mlx::core::fast::CustomKernelFunction&
compact_stochastic_verify_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_mtp_verify_top_k",
            {
                "proposal_indices",
                "proposal_probabilities",
                "target_indices",
                "target_probabilities",
                "draft_tokens",
                "random",
            },
            {"output"},
            kCompactStochasticVerifySource,
            "",
            true,
            false,
            options);
    }();
    return kernel;
}

array compact_values(
    const array& values,
    mlx::core::Dtype dtype,
    int expected,
    const char* name) {
    if (expected <= 0 ||
        values.size() != static_cast<std::size_t>(expected)) {
        throw std::invalid_argument(
            std::string("MTP compact ") + name + " has invalid size");
    }
    return mlx::core::contiguous(
        mlx::core::reshape(
            mlx::core::astype(values, dtype),
            Shape{expected}));
}

} // namespace

namespace mfq::metal {

namespace {

constexpr double kDepthAcceptanceAlpha = 0.08;
constexpr double kDepthAcceptancePrior = 0.6;
constexpr std::uint64_t kDepthAcceptanceBootstrapTrials = 16;
constexpr double kDepthTimeTauMs = 400.0;
constexpr double kDepthProbePeriodMs = 1000.0;
constexpr double kDepthExplorePeriodMs = 5000.0;
constexpr int kDepthProbeLength = 4;
constexpr double kDepthProbeDuty = 0.15;
constexpr double kDepthProbeMargin = 1.15;
constexpr double kDepthSpikeRatio = 2.0;
constexpr double kDepthSpikeDamp = 0.25;
constexpr double kDepthMarginalMs = 7.0;
constexpr double kDepthHysteresis = 1.03;
constexpr double kDepthExitMargin = 1.15;
// oMLX can afford a long losing streak because its standard decoder later
// schedules MTP re-entry probes. The native C++ engine hands the remainder of
// this request to plain decode with no re-entry path, so lingering for 16
// cycles consumes most short generations.
constexpr int kDepthExitStreak = 3;
constexpr int kDepthRealizedWindow = 3;
constexpr double kDepthRealizedMargin = 1.03;

} // namespace

MlxMtpDepthController::MlxMtpDepthController(
    int maximum_depth)
    : maximum_depth_(std::clamp(maximum_depth, 1, 5)),
      current_depth_(maximum_depth_),
      acceptance_(
          static_cast<std::size_t>(maximum_depth_),
          kDepthAcceptancePrior),
      acceptance_hits_(static_cast<std::size_t>(maximum_depth_)),
      acceptance_trials_(static_cast<std::size_t>(maximum_depth_)),
      cycle_ms_(
          static_cast<std::size_t>(maximum_depth_ + 1)),
      cycle_age_ms_(
          static_cast<std::size_t>(maximum_depth_ + 1)) {
    // A maximum-width verify observes every conditional acceptance position.
    // Repeat it once so update_time() drops one-time graph compilation, then
    // compare against a stable plain-decode baseline. Intermediate widths are
    // interpolated and periodically probed after warmup.
    warmup_.insert(
        warmup_.end(),
        {maximum_depth_, maximum_depth_});
    warmup_.insert(warmup_.end(), {0, 0, 0});
}

void MlxMtpDepthController::observe(
    int used_depth,
    int accepted_drafts,
    double cycle_ms,
    bool time_sample) {
    ++cycles_;
    used_depth = std::clamp(used_depth, 0, maximum_depth_);
    accepted_drafts = std::clamp(accepted_drafts, 0, used_depth);
    for (int position = 0; position < used_depth; ++position) {
        const double hit = position < accepted_drafts ? 1.0 : 0.0;
        const auto index = static_cast<std::size_t>(position);
        auto& estimate = acceptance_[index];
        auto& trials = acceptance_trials_.at(index);
        auto& hits = acceptance_hits_.at(index);
        ++trials;
        hits += static_cast<std::uint64_t>(hit);
        if (trials <= kDepthAcceptanceBootstrapTrials) {
            // A controller is created for each request. Bootstrap from the
            // observed conditional rate with one prior sample, then switch to
            // oMLX's slow EMA once the estimate is established.
            estimate = (kDepthAcceptancePrior +
                        static_cast<double>(hits)) /
                (1.0 + static_cast<double>(trials));
        } else {
            estimate = (1.0 - kDepthAcceptanceAlpha) * estimate +
                kDepthAcceptanceAlpha * hit;
        }
        if (position >= accepted_drafts) {
            break;
        }
    }

    cycle_ms = std::max(0.0, cycle_ms);
    if (time_sample) {
        update_time(used_depth, cycle_ms);
        if (used_depth > 0) {
            ++realized_window_cycles_;
            realized_window_tokens_ +=
                static_cast<std::uint64_t>(accepted_drafts + 1);
            realized_window_ms_ += cycle_ms;
        }
    }
    for (std::size_t depth = 0; depth < cycle_age_ms_.size(); ++depth) {
        if (cycle_age_ms_[depth]) {
            *cycle_age_ms_[depth] += cycle_ms;
        }
    }
    if (time_sample) {
        cycle_age_ms_[static_cast<std::size_t>(used_depth)] = 0.0;
    }
    milliseconds_since_probe_ += cycle_ms;
    milliseconds_since_explore_ += cycle_ms;

    const auto resolve_realized_window = [&] {
        if (realized_window_cycles_ < kDepthRealizedWindow ||
            !cycle_ms_.front() || realized_window_ms_ <= 0.0) {
            return;
        }
        const double measured_tokens_per_ms =
            static_cast<double>(realized_window_tokens_) /
            realized_window_ms_;
        const double baseline_tokens_per_ms =
            1.0 / std::max(1.0e-6, *cycle_ms_.front());
        realized_speculation_losing_ =
            measured_tokens_per_ms <
                baseline_tokens_per_ms * kDepthRealizedMargin;
        realized_window_cycles_ = 0;
        realized_window_tokens_ = 0;
        realized_window_ms_ = 0.0;
    };

    if (speculation_losing()) {
        ++exit_streak_;
    } else {
        exit_streak_ = 0;
    }

    if (!warmup_.empty()) {
        warmup_.erase(warmup_.begin());
        if (!warmup_.empty()) {
            current_depth_ = warmup_.front();
            return;
        }
        resolve_realized_window();
        current_depth_ = best_depth();
        milliseconds_since_probe_ = 0.0;
        return;
    }

    resolve_realized_window();

    if (probe_left_ > 0) {
        --probe_left_;
        if (probe_left_ == 0) {
            current_depth_ = best_depth();
            milliseconds_since_probe_ = 0.0;
        }
        return;
    }

    current_depth_ = best_depth();
    if (maximum_depth_ <= 1) {
        return;
    }
    const double period = std::max(
        kDepthProbePeriodMs,
        static_cast<double>(kDepthProbeLength) * cycle_ms /
            kDepthProbeDuty);
    if (milliseconds_since_probe_ < period) {
        return;
    }
    const bool explore_due = milliseconds_since_explore_ >=
        std::max(kDepthExplorePeriodMs, 2.0 * period);
    const auto target = explore_due ? most_stale() : best_rival();
    if (!target) {
        return;
    }
    current_depth_ = *target;
    probe_left_ = kDepthProbeLength;
    milliseconds_since_probe_ = 0.0;
    if (explore_due) {
        milliseconds_since_explore_ = 0.0;
    }
}

bool MlxMtpDepthController::should_exit() const noexcept {
    return realized_speculation_losing_ ||
        exit_streak_ >= kDepthExitStreak;
}

double MlxMtpDepthController::conditional_acceptance(
    int position) const {
    if (position < 0 || position >= maximum_depth_) {
        throw std::out_of_range("MTP acceptance position is out of range");
    }
    return acceptance_[static_cast<std::size_t>(position)];
}

std::optional<double> MlxMtpDepthController::measured_cycle_ms(
    int depth) const {
    if (depth < 0 || depth > maximum_depth_) {
        throw std::out_of_range("MTP measured depth is out of range");
    }
    return cycle_ms_[static_cast<std::size_t>(depth)];
}

void MlxMtpDepthController::update_time(
    int depth,
    double cycle_ms) {
    auto& estimate = cycle_ms_[static_cast<std::size_t>(depth)];
    if (!estimate) {
        estimate = cycle_ms;
        return;
    }
    if (!warmup_.empty()) {
        *estimate = std::min(*estimate, cycle_ms);
        return;
    }
    double alpha = 1.0 - std::exp(-cycle_ms / kDepthTimeTauMs);
    if (cycle_ms > kDepthSpikeRatio * *estimate) {
        alpha *= kDepthSpikeDamp;
    }
    *estimate = (1.0 - alpha) * *estimate + alpha * cycle_ms;
}

double MlxMtpDepthController::marginal_estimate() const {
    int low = -1;
    int high = -1;
    for (int depth = 0; depth <= maximum_depth_; ++depth) {
        if (!cycle_ms_[static_cast<std::size_t>(depth)]) {
            continue;
        }
        if (low < 0) low = depth;
        high = depth;
    }
    if (low >= 0 && high > low) {
        const double slope =
            (*cycle_ms_[static_cast<std::size_t>(high)] -
             *cycle_ms_[static_cast<std::size_t>(low)]) /
            static_cast<double>(high - low);
        if (slope > 0.0) {
            return slope;
        }
    }
    return kDepthMarginalMs;
}

double MlxMtpDepthController::time_estimate(int depth) const {
    if (cycle_ms_[static_cast<std::size_t>(depth)]) {
        return *cycle_ms_[static_cast<std::size_t>(depth)];
    }
    int reference = -1;
    for (int candidate = 0; candidate <= maximum_depth_; ++candidate) {
        if (!cycle_ms_[static_cast<std::size_t>(candidate)]) {
            continue;
        }
        if (reference < 0 ||
            std::abs(candidate - depth) < std::abs(reference - depth)) {
            reference = candidate;
        }
    }
    if (reference < 0) {
        return 30.0 + kDepthMarginalMs * depth;
    }
    if (depth == 0) {
        double minimum = std::numeric_limits<double>::infinity();
        for (const auto& measured : cycle_ms_) {
            if (measured) minimum = std::min(minimum, *measured);
        }
        return minimum;
    }
    return std::max(
        1.0e-3,
        *cycle_ms_[static_cast<std::size_t>(reference)] +
            marginal_estimate() * (depth - reference));
}

double MlxMtpDepthController::score(int depth) const {
    double expected = 1.0;
    double run = 1.0;
    for (int position = 0; position < depth; ++position) {
        run *= acceptance_[static_cast<std::size_t>(position)];
        expected += run;
    }
    return expected / std::max(1.0e-6, time_estimate(depth));
}

bool MlxMtpDepthController::speculation_losing() const {
    if (!warmup_.empty() || !cycle_ms_.front()) {
        return false;
    }
    const double baseline = score(0);
    double best = 0.0;
    for (int depth = 1; depth <= maximum_depth_; ++depth) {
        best = std::max(best, score(depth));
    }
    return baseline > 0.0 && best < baseline * kDepthExitMargin;
}

int MlxMtpDepthController::best_depth() const {
    int best = current_depth_;
    double best_score = -1.0;
    const int start = cycle_ms_.front() ? 0 : 1;
    for (int depth = start; depth <= maximum_depth_; ++depth) {
        const double candidate = score(depth);
        if (candidate > best_score) {
            best = depth;
            best_score = candidate;
        }
    }
    if (best != current_depth_ &&
        best_score < score(current_depth_) * kDepthHysteresis) {
        return current_depth_;
    }
    return best;
}

std::optional<int> MlxMtpDepthController::best_rival() const {
    const double current_score = score(current_depth_);
    std::optional<int> rival;
    double rival_score = 0.0;
    for (int depth = 1; depth <= maximum_depth_; ++depth) {
        if (depth == current_depth_) continue;
        const double candidate = score(depth);
        if (candidate > rival_score) {
            rival = depth;
            rival_score = candidate;
        }
    }
    if (current_depth_ != 0) {
        const double candidate = score(0);
        if (candidate > rival_score) {
            rival = 0;
            rival_score = candidate;
        }
    }
    if (current_score <= 0.0) {
        return most_stale();
    }
    return rival && rival_score >= current_score / kDepthProbeMargin
        ? rival
        : std::nullopt;
}

std::optional<int> MlxMtpDepthController::most_stale() const {
    std::optional<int> result;
    double oldest = -1.0;
    const auto consider = [&](int depth) {
        if (depth == current_depth_) return;
        const auto& age = cycle_age_ms_[static_cast<std::size_t>(depth)];
        const double value = age
            ? *age
            : std::numeric_limits<double>::infinity();
        if (value > oldest) {
            result = depth;
            oldest = value;
        }
    };
    for (int depth = 1; depth <= maximum_depth_; ++depth) {
        consider(depth);
    }
    consider(0);
    return result;
}

namespace {

struct PreparedMtpDraft {
    int depth;
    array tokens;
    std::optional<array> compact_indices;
    std::optional<array> compact_probabilities;
    std::vector<std::vector<float>> host_probabilities;
};

bool mtp_phase_profile_requested() noexcept {
    static const bool requested = [] {
        const char* value = std::getenv("MFQ_METAL_PROFILE_MTP");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return requested;
}

void materialize_profiled_draft(PreparedMtpDraft& draft) {
    std::vector<array> pending{draft.tokens};
    if (draft.compact_indices) {
        pending.push_back(*draft.compact_indices);
    }
    if (draft.compact_probabilities) {
        pending.push_back(*draft.compact_probabilities);
    }
    mlx::core::eval(std::move(pending));
}

array mtp_sampling_row(
    const array& logits,
    int vocab,
    const char* source) {
    if (logits.ndim() == 2 && logits.shape() == Shape{1, vocab}) {
        return logits;
    }
    if (logits.ndim() == 3 &&
        logits.shape() == Shape{1, 1, vocab}) {
        return mlx::core::reshape(logits, Shape{1, vocab});
    }
    throw std::runtime_error(
        std::string("MTP ") + source +
        " logits must have [1,vocab] or [1,1,vocab] shape");
}

PreparedMtpDraft prepare_mtp_draft(
    const MlxMtpDraftContext& context,
    const MlxMtpEngineRequest& request,
    const MlxMtpEngineCallbacks& callbacks,
    MlxSampler& sampler,
    const std::optional<array>& committed_counts,
    bool compact_stochastic,
    const MlxSamplingParams& draft_sampling) {
    const int requested = context.requested_depth;
    if (requested < 0 || requested > kMlxMtpEngineMaximumDraftDepth) {
        throw std::runtime_error("MTP requested draft depth is invalid");
    }

    std::vector<array> tokens;
    std::vector<array> compact_indices;
    std::vector<array> compact_probabilities;
    std::vector<std::vector<float>> host_probabilities;
    tokens.reserve(static_cast<std::size_t>(requested));
    compact_indices.reserve(static_cast<std::size_t>(requested));
    compact_probabilities.reserve(static_cast<std::size_t>(requested));
    host_probabilities.reserve(static_cast<std::size_t>(requested));
    auto prospective_counts = committed_counts;

    const MlxMtpTokenSelector select_token = [&](const array& raw_logits) {
        if (static_cast<int>(tokens.size()) >= requested) {
            throw std::runtime_error(
                "MTP predictor selected more tokens than requested");
        }
        auto logits = mtp_sampling_row(
            raw_logits, request.vocab, "predictor");
        if (prospective_counts) {
            logits = sampler.apply_penalties(logits, *prospective_counts);
        }

        array token = mlx::core::zeros(Shape{1}, mlx::core::int32);
        if (request.sampling.greedy()) {
            token = sample_greedy(logits);
        } else if (compact_stochastic) {
            const array random(
                {static_cast<float>(sampler.next_uniform())},
                Shape{1},
                mlx::core::float32);
            auto distribution = sample_top_k_distribution(
                logits,
                random,
                draft_sampling.temperature,
                draft_sampling.top_k,
                draft_sampling.top_p);
            token = std::move(distribution.sampled);
            compact_indices.push_back(std::move(distribution.indices));
            compact_probabilities.push_back(
                std::move(distribution.probabilities));
        } else {
            auto probabilities = host_sampling_distribution(
                logits, draft_sampling);
            const auto host_token = sample_host_distribution(
                probabilities, sampler.next_uniform());
            token = array(
                {host_token}, Shape{1}, mlx::core::int32);
            host_probabilities.push_back(std::move(probabilities));
        }
        token = mlx::core::reshape(token, Shape{1});
        tokens.push_back(token);
        if (prospective_counts) {
            prospective_counts = sample_token_counts_add(
                *prospective_counts, token);
        }
        return token;
    };

    callbacks.prepare_draft(context, select_token);
    if (static_cast<int>(tokens.size()) != requested) {
        throw std::runtime_error(
            "MTP predictor did not produce the requested draft depth");
    }

    auto token_values = tokens.empty()
        ? mlx::core::zeros(Shape{0}, mlx::core::int32)
        : mlx::core::concatenate(tokens, 0);
    std::optional<array> indices;
    std::optional<array> probabilities;
    std::vector<array> pending{token_values};
    if (!compact_indices.empty()) {
        indices = mlx::core::concatenate(compact_indices, 0);
        probabilities = mlx::core::concatenate(
            compact_probabilities, 0);
        pending.push_back(*indices);
        pending.push_back(*probabilities);
    }
    mlx::core::async_eval(std::move(pending));
    return {
        requested,
        std::move(token_values),
        std::move(indices),
        std::move(probabilities),
        std::move(host_probabilities),
    };
}

} // namespace

std::size_t mlx_prime_mtp_history(
    const array& target_hidden,
    const array& prompt_ids,
    const MlxMtpHistoryFold& fold,
    int chunk_size) {
    if (!fold || target_hidden.ndim() != 3 || prompt_ids.ndim() != 2 ||
        target_hidden.shape(0) <= 0 || target_hidden.shape(1) <= 0 ||
        target_hidden.shape(2) <= 0 ||
        prompt_ids.shape(0) != target_hidden.shape(0) ||
        prompt_ids.shape(1) != target_hidden.shape(1) ||
        chunk_size <= 0) {
        throw std::invalid_argument("invalid MTP history priming input");
    }
    const int batch = prompt_ids.shape(0);
    const int tokens = prompt_ids.shape(1);
    const int width = target_hidden.shape(2);
    const int pairs = std::max(0, tokens - 1);
    for (int offset = 0; offset < pairs; offset += chunk_size) {
        const int count = std::min(chunk_size, pairs - offset);
        auto hidden_rows = mlx::core::slice(
            target_hidden,
            Shape{0, offset, 0},
            Shape{batch, offset + count, width});
        auto shifted_ids = mlx::core::slice(
            prompt_ids,
            Shape{0, offset + 1},
            Shape{batch, offset + count + 1});
        auto anchor = fold(hidden_rows, shifted_ids, offset);
        mlx::core::async_eval(std::vector<array>{std::move(anchor)});
    }
    return static_cast<std::size_t>(pairs);
}

std::optional<array> mlx_generation_token_counts(
    const MlxSamplingParams& sampling,
    const array& prompt_ids,
    int vocab) {
    if (vocab <= 0) {
        throw std::invalid_argument(
            "generation vocabulary size must be positive");
    }
    if (!sampling.has_penalties()) {
        return std::nullopt;
    }
    return sample_token_counts_add(
        mlx::core::zeros(Shape{vocab}, mlx::core::int32),
        prompt_ids);
}

std::int32_t run_mlx_mtp_generation(
    MlxMtpEngineRequest request,
    const MlxMtpEngineCallbacks& callbacks,
    MlxMtpGenerationStats& stats) {
    if (request.vocab <= 0 || request.generation_limit <= 0 ||
        request.maximum_context <= 0 ||
        request.predictor_maximum_depth <= 0 ||
        request.predictor_maximum_depth > kMlxMtpEngineMaximumDraftDepth ||
        !callbacks.target_cache_position || !callbacks.prepare_draft ||
        !callbacks.verify_target || !callbacks.resolve_target ||
        !callbacks.plain_decode) {
        throw std::invalid_argument("invalid MTP engine configuration");
    }
    mlx_validate_token_set(request.eos_token_ids, request.vocab);

    stats = {};
    stats.available = true;
    stats.used = true;
    MlxSampler sampler(request.sampling);
    sampler.discard_random(request.sampler_draws_consumed);
    auto counts = std::move(request.token_counts);

    const auto sample_token = [&](const array& raw_logits) {
        const auto logits = mtp_sampling_row(
            raw_logits, request.vocab, "target");
        auto sampled = counts
            ? sampler.sample(logits, *counts)
            : sampler.sample(logits);
        sampled.eval();
        const auto token = sampled.data<std::int32_t>()[0];
        if (token < 0 || token >= request.vocab) {
            throw std::runtime_error(
                "MTP target sampler returned an out-of-range token");
        }
        return token;
    };

    std::int32_t generated = 0;
    const auto emit = [&](std::int32_t token) {
        if (counts) {
            const array token_id(
                {token}, Shape{1, 1}, mlx::core::int32);
            counts = sample_token_counts_add(*counts, token_id);
        }
        ++generated;
        const bool delivered = !request.callback || request.callback(token);
        return delivered &&
            !mlx_token_in_set(request.eos_token_ids, token) &&
            generated < request.generation_limit;
    };

    const bool compact_stochastic =
        !request.sampling.greedy() && request.sampling.top_k > 0 &&
        request.sampling.top_k <= 64;
    MlxSamplingParams draft_sampling = request.sampling;
    if (compact_stochastic) {
        // The proposal may be sharper than the target distribution because
        // exact p/q verification preserves the target sampler.
        draft_sampling.temperature = 0.6;
        draft_sampling.top_p = 0.95;
    }
    const int maximum_depth =
        (!request.sampling.greedy() && !compact_stochastic)
        ? 1
        : std::min(
              request.predictor_maximum_depth,
              std::clamp(request.sampling.mtp_max_draft_tokens, 1, 5));
    MlxMtpDepthController depth_controller(maximum_depth);

    auto pending = sample_token(request.initial_logits);
    if (!emit(pending)) {
        return generated;
    }

    const auto bounded_depth = [&](int desired) {
        const int context_depth = std::max(
            0,
            request.maximum_context - callbacks.target_cache_position() - 1);
        const int output_depth = std::max(
            0, request.generation_limit - generated - 1);
        return std::min({desired, context_depth, output_depth});
    };
    MlxMtpDraftContext initial_context{
        true,
        pending,
        bounded_depth(depth_controller.depth()),
        callbacks.target_cache_position(),
        0,
        nullptr,
        {},
    };
    const bool profile_phases = mtp_phase_profile_requested();
    const auto initial_draft_started = std::chrono::steady_clock::now();
    auto draft = prepare_mtp_draft(
        initial_context,
        request,
        callbacks,
        sampler,
        counts,
        compact_stochastic,
        draft_sampling);
    if (profile_phases) {
        materialize_profiled_draft(draft);
        std::cerr
            << "mtp_phase initial=1 depth=" << draft.depth
            << " draft_ms="
            << std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - initial_draft_started)
                   .count()
            << '\n';
    }

    const auto finish_without_mtp = [&](std::int32_t last_token) {
        while (generated < request.generation_limit) {
            const auto next = sample_token(callbacks.plain_decode(last_token));
            if (!emit(next)) {
                break;
            }
            last_token = next;
        }
        return generated;
    };

    while (generated < request.generation_limit) {
        const auto cycle_started = std::chrono::steady_clock::now();
        const int cycle_cache_start = callbacks.target_cache_position();
        const int draft_count = draft.depth;
        const auto target_started = std::chrono::steady_clock::now();
        detail::ComponentProfile target_profile;
        const bool profile_target_components =
            profile_phases && detail::component_profile_requested();
        auto target = [&]() {
            detail::ScopedComponentProfile profile_scope(
                profile_target_components ? &target_profile : nullptr);
            auto result = callbacks.verify_target(
                pending, draft.tokens, draft_count);
            if (profile_phases) {
                mlx::core::eval(result.logits, result.hidden);
            }
            return result;
        }();
        if (target.logits.ndim() != 2 ||
            target.logits.shape() != Shape{draft_count + 1, request.vocab}) {
            throw std::runtime_error(
                "MTP target adapter returned incompatible logits");
        }
        const auto target_finished = std::chrono::steady_clock::now();
        if (profile_target_components) {
            for (const auto& [name, timing] : target_profile.timings()) {
                std::cerr
                    << "mtp_component cycle=" << (stats.cycles + 1)
                    << " depth=" << draft_count
                    << " name=" << name
                    << " ms=" << timing.elapsed_ms
                    << " evals=" << timing.evaluations
                    << '\n';
            }
        }

        ++stats.cycles;
        stats.drafted_tokens += static_cast<std::uint64_t>(draft_count);
        ++stats.depth_cycles.at(static_cast<std::size_t>(draft_count));
        for (int position = 0; position < draft_count; ++position) {
            ++stats.position_drafted.at(static_cast<std::size_t>(position));
        }

        std::vector<array> adjusted_rows;
        adjusted_rows.reserve(static_cast<std::size_t>(draft_count + 1));
        auto row_counts = counts;
        for (int row = 0; row <= draft_count; ++row) {
            auto logits_row = mlx::core::slice(
                target.logits,
                Shape{row, 0},
                Shape{row + 1, request.vocab});
            adjusted_rows.push_back(
                row_counts
                    ? sampler.apply_penalties(logits_row, *row_counts)
                    : logits_row);
            if (row < draft_count && row_counts) {
                auto token = mlx::core::slice(
                    draft.tokens, Shape{row}, Shape{row + 1});
                row_counts = sample_token_counts_add(*row_counts, token);
            }
        }
        auto target_rows = mlx::core::concatenate(adjusted_rows, 0);

        MlxMtpVerification verification;
        std::vector<std::int32_t> draft_ids;
        draft_ids.reserve(static_cast<std::size_t>(draft_count));
        if (draft_count == 0) {
            auto sampled = sampler.sample(mtp_sampling_row(
                target_rows, request.vocab, "adjusted target"));
            sampled.eval();
            const auto next = sampled.data<std::int32_t>()[0];
            if (next < 0 || next >= request.vocab) {
                throw std::runtime_error(
                    "MTP target sampler returned an out-of-range token");
            }
            verification = {0, next, true};
        } else if (request.sampling.greedy()) {
            auto target_tokens = sample_greedy(target_rows);
            auto compact = mlx::core::concatenate(
                {
                    mlx::core::reshape(draft.tokens, Shape{draft_count}),
                    mlx::core::reshape(
                        target_tokens, Shape{draft_count + 1}),
                },
                0);
            compact.eval();
            const auto* resolved = compact.data<std::int32_t>();
            draft_ids.assign(resolved, resolved + draft_count);
            verification = verify_greedy_mtp(
                std::span<const std::int32_t>(resolved, draft_count),
                std::span<const std::int32_t>(
                    resolved + draft_count, draft_count + 1));
        } else if (draft.compact_indices &&
                   draft.compact_probabilities && compact_stochastic) {
            auto target_distribution = sample_top_k_distribution(
                target_rows,
                mlx::core::zeros(
                    Shape{draft_count + 1}, mlx::core::float32),
                request.sampling.temperature,
                request.sampling.top_k,
                request.sampling.top_p);
            std::vector<float> random_values;
            random_values.reserve(static_cast<std::size_t>(draft_count + 1));
            for (int index = 0; index <= draft_count; ++index) {
                random_values.push_back(
                    static_cast<float>(sampler.next_uniform()));
            }
            const array random(
                random_values.begin(),
                Shape{draft_count + 1},
                mlx::core::float32);
            auto compact = verify_stochastic_mtp_top_k_chain_device(
                *draft.compact_indices,
                *draft.compact_probabilities,
                target_distribution.indices,
                target_distribution.probabilities,
                draft.tokens,
                random,
                draft_count,
                request.sampling.top_k);
            compact.eval();
            const auto* resolved = compact.data<std::int32_t>();
            draft_ids.assign(resolved + 2, resolved + 2 + draft_count);
            verification = {
                static_cast<std::size_t>(resolved[0]),
                resolved[1],
                resolved[0] == draft_count,
            };
        } else {
            if (draft_count != 1 ||
                draft.host_probabilities.size() != 1) {
                throw std::runtime_error(
                    "host MTP verification requires exactly one draft");
            }
            draft.tokens.eval();
            const auto draft_id = draft.tokens.data<std::int32_t>()[0];
            draft_ids.push_back(draft_id);
            verification = verify_stochastic_mtp(
                draft_id,
                draft.host_probabilities.front(),
                host_sampling_distribution(
                    mlx::core::slice(
                        target_rows,
                        Shape{0, 0},
                        Shape{1, request.vocab}),
                    request.sampling),
                host_sampling_distribution(
                    mlx::core::slice(
                        target_rows,
                        Shape{1, 0},
                        Shape{2, request.vocab}),
                    request.sampling),
                sampler.next_uniform(),
                sampler.next_uniform());
        }

        const int accepted = static_cast<int>(verification.accepted_drafts);
        if (accepted < 0 || accepted > draft_count ||
            verification.next_token < 0 ||
            verification.next_token >= request.vocab) {
            throw std::runtime_error("MTP verification returned invalid data");
        }
        stats.accepted_tokens += static_cast<std::uint64_t>(accepted);
        for (int position = 0; position < accepted; ++position) {
            ++stats.position_accepted.at(static_cast<std::size_t>(position));
        }

        int emitted_accepted = 0;
        bool continue_generation = true;
        for (int index = 0; index < accepted; ++index) {
            ++emitted_accepted;
            if (!emit(draft_ids.at(static_cast<std::size_t>(index)))) {
                continue_generation = false;
                break;
            }
        }
        const auto resolve_started = std::chrono::steady_clock::now();
        callbacks.resolve_target(emitted_accepted, draft_count);
        const auto resolve_finished = std::chrono::steady_clock::now();
        if (!continue_generation) {
            return generated;
        }
        if (!emit(verification.next_token)) {
            return generated;
        }

        const double cycle_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cycle_started).count();
        if (profile_phases) {
            std::cerr
                << "mtp_phase cycle=" << stats.cycles
                << " depth=" << draft_count
                << " accepted=" << accepted
                << " target_ms="
                << std::chrono::duration<double, std::milli>(
                       target_finished - target_started)
                       .count()
                << " decision_ms="
                << std::chrono::duration<double, std::milli>(
                       resolve_started - target_finished)
                       .count()
                << " resolve_ms="
                << std::chrono::duration<double, std::milli>(
                       resolve_finished - resolve_started)
                       .count()
                << " cycle_ms=" << cycle_ms
                << '\n';
        }
        depth_controller.observe(draft_count, accepted, cycle_ms);
        stats.selected_depth = depth_controller.depth();
        for (int depth = 0; depth <= depth_controller.maximum_depth(); ++depth) {
            const auto measured = depth_controller.measured_cycle_ms(depth);
            if (measured) {
                stats.measured_depth_ms.at(
                    static_cast<std::size_t>(depth)) = *measured;
            }
        }

        pending = verification.next_token;
        if (depth_controller.should_exit()) {
            return finish_without_mtp(pending);
        }

        std::vector<std::int32_t> next_ids;
        next_ids.reserve(static_cast<std::size_t>(accepted + 1));
        next_ids.insert(
            next_ids.end(), draft_ids.begin(), draft_ids.begin() + accepted);
        next_ids.push_back(pending);
        MlxMtpDraftContext next_context{
            false,
            pending,
            bounded_depth(depth_controller.depth()),
            cycle_cache_start,
            accepted,
            &target.hidden,
            std::span<const std::int32_t>(next_ids),
        };
        const auto draft_started = std::chrono::steady_clock::now();
        draft = prepare_mtp_draft(
            next_context,
            request,
            callbacks,
            sampler,
            counts,
            compact_stochastic,
            draft_sampling);
        if (profile_phases) {
            materialize_profiled_draft(draft);
            std::cerr
                << "mtp_phase initial=0 depth=" << draft.depth
                << " draft_ms="
                << std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - draft_started)
                       .count()
                << '\n';
        }
    }
    return generated;
}

MlxMtpVerification verify_greedy_mtp(
    std::span<const std::int32_t> drafts,
    std::span<const std::int32_t> targets) {
    if (drafts.empty() || targets.size() != drafts.size() + 1) {
        throw std::invalid_argument(
            "MTP verification requires N drafts and N+1 targets");
    }
    for (std::size_t index = 0; index < drafts.size(); ++index) {
        if (targets[index] != drafts[index]) {
            return {index, targets[index], false};
        }
    }
    return {drafts.size(), targets.back(), true};
}

MlxMtpVerification verify_stochastic_mtp(
    std::int32_t draft,
    const std::vector<float>& proposal,
    const std::vector<float>& target,
    const std::vector<float>& bonus,
    double acceptance_uniform,
    double sample_uniform) {
    if (proposal.empty() || proposal.size() != target.size() ||
        target.size() != bonus.size() || draft < 0 ||
        static_cast<std::size_t>(draft) >= proposal.size() ||
        !std::isfinite(acceptance_uniform)) {
        throw std::invalid_argument(
            "stochastic MTP verification inputs are incompatible");
    }
    const double q = proposal[static_cast<std::size_t>(draft)];
    const double p = target[static_cast<std::size_t>(draft)];
    if (!(q > 0.0) || p < 0.0 || !std::isfinite(q) || !std::isfinite(p)) {
        throw std::invalid_argument(
            "stochastic MTP draft has an invalid probability");
    }
    const double acceptance = std::min(1.0, p / q);
    if (std::clamp(acceptance_uniform, 0.0, std::nextafter(1.0, 0.0)) <
        acceptance) {
        return {
            1,
            sample_host_distribution(bonus, sample_uniform),
            true,
        };
    }

    std::vector<float> correction(proposal.size());
    double total = 0.0;
    for (std::size_t token = 0; token < correction.size(); ++token) {
        correction[token] = std::max(0.0f, target[token] - proposal[token]);
        total += correction[token];
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        correction = target;
    } else {
        for (auto& probability : correction) {
            probability = static_cast<float>(probability / total);
        }
    }
    return {
        0,
        sample_host_distribution(correction, sample_uniform),
        false,
    };
}

array verify_stochastic_mtp_top_k_device(
    const array& proposal_indices,
    const array& proposal_probabilities,
    const array& target_indices,
    const array& target_probabilities,
    const array& bonus_indices,
    const array& bonus_probabilities,
    const array& draft_token,
    const array& random,
    int top_k) {
    auto target_ids = mlx::core::concatenate(
        {
            compact_values(
                target_indices, mlx::core::int32, top_k, "target indices"),
            compact_values(
                bonus_indices, mlx::core::int32, top_k, "bonus indices"),
        },
        0);
    auto target_probs = mlx::core::concatenate(
        {
            compact_values(
                target_probabilities,
                mlx::core::float32,
                top_k,
                "target probabilities"),
            compact_values(
                bonus_probabilities,
                mlx::core::float32,
                top_k,
                "bonus probabilities"),
        },
        0);
    return verify_stochastic_mtp_top_k_chain_device(
        proposal_indices,
        proposal_probabilities,
        target_ids,
        target_probs,
        draft_token,
        random,
        1,
        top_k);
}

array verify_stochastic_mtp_top_k_chain_device(
    const array& proposal_indices,
    const array& proposal_probabilities,
    const array& target_indices,
    const array& target_probabilities,
    const array& draft_tokens,
    const array& random,
    int drafts,
    int top_k) {
    if (top_k <= 0 || top_k > 64) {
        throw std::invalid_argument(
            "MTP compact stochastic verification requires top_k in [1,64]");
    }
    if (drafts <= 0 || drafts > 5) {
        throw std::invalid_argument(
            "MTP compact stochastic verification requires drafts in [1,5]");
    }
    const int proposal_size = drafts * top_k;
    const int target_size = (drafts + 1) * top_k;
    auto proposal_ids = compact_values(
        proposal_indices,
        mlx::core::int32,
        proposal_size,
        "proposal indices");
    auto proposal_probs = compact_values(
        proposal_probabilities,
        mlx::core::float32,
        proposal_size,
        "proposal probabilities");
    auto target_ids = compact_values(
        target_indices,
        mlx::core::int32,
        target_size,
        "target indices");
    auto target_probs = compact_values(
        target_probabilities,
        mlx::core::float32,
        target_size,
        "target probabilities");
    auto draft = compact_values(
        draft_tokens, mlx::core::int32, drafts, "draft tokens");
    auto uniforms = compact_values(
        random, mlx::core::float32, drafts + 1, "random values");
    auto outputs = compact_stochastic_verify_kernel()(
        {
            proposal_ids,
            proposal_probs,
            target_ids,
            target_probs,
            draft,
            uniforms,
        },
        {Shape{drafts + 2}},
        {mlx::core::int32},
        {1, 1, 1},
        {1, 1, 1},
        {
            {"DRAFTS", drafts},
            {"TOP_K", top_k},
        },
        std::nullopt,
        false,
        {});
    return outputs.front();
}

} // namespace mfq::metal
