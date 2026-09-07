#include "mlx_mtp.h"

#include "mlx_sampling.h"

#include <algorithm>
#include <cmath>
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
constexpr int kDepthExitStreak = 16;

} // namespace

MlxMtpDepthController::MlxMtpDepthController(
    int maximum_depth)
    : maximum_depth_(std::clamp(maximum_depth, 1, 5)),
      current_depth_(maximum_depth_),
      acceptance_(
          static_cast<std::size_t>(maximum_depth_),
          0.6),
      cycle_ms_(
          static_cast<std::size_t>(maximum_depth_ + 1)),
      cycle_age_ms_(
          static_cast<std::size_t>(maximum_depth_ + 1)) {
    for (int depth = maximum_depth_; depth >= 1; --depth) {
        warmup_.push_back(depth);
    }
    if (maximum_depth_ > 1) {
        warmup_.insert(warmup_.end(), {0, 0, 0});
    }
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
        auto& estimate = acceptance_[static_cast<std::size_t>(position)];
        estimate = (1.0 - kDepthAcceptanceAlpha) * estimate +
            kDepthAcceptanceAlpha * hit;
        if (position >= accepted_drafts) {
            break;
        }
    }

    cycle_ms = std::max(0.0, cycle_ms);
    if (time_sample) {
        update_time(used_depth, cycle_ms);
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
        current_depth_ = best_depth();
        milliseconds_since_probe_ = 0.0;
        return;
    }

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
    return exit_streak_ >= kDepthExitStreak;
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
