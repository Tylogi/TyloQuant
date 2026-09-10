#pragma once

// Host reference policy for the native CUDA MTP chain. The acceptance and
// positive-residual correction match Metal runtime/mlx_mtp.cpp at e27c987.
// Inputs to distribution() have already received the ordinary CUDA penalties.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace mfq::cuda::mtp {

inline constexpr int kMaximumDraftDepth = 5;

struct ChainVerification {
    std::size_t accepted_drafts = 0;
    int32_t next_token = -1;
    bool bonus = false;
};

struct GenerationStats {
    bool available = false;
    bool used = false;
    uint64_t cycles = 0;
    uint64_t drafted_tokens = 0;
    uint64_t accepted_tokens = 0;
    std::array<uint64_t, kMaximumDraftDepth + 1> depth_cycles{};
    std::array<uint64_t, kMaximumDraftDepth> position_drafted{};
    std::array<uint64_t, kMaximumDraftDepth> position_accepted{};
    std::array<double, kMaximumDraftDepth + 1> measured_depth_ms{};
    int selected_depth = 0;
};

inline ChainVerification verify_greedy(
        std::span<const int32_t> drafts,
        std::span<const int32_t> targets) {
    if (drafts.empty() || targets.size() != drafts.size() + 1) {
        throw std::invalid_argument("greedy MTP verification requires one bonus target");
    }
    std::size_t accepted = 0;
    while (accepted < drafts.size() && drafts[accepted] == targets[accepted]) {
        ++accepted;
    }
    return {
        accepted,
        targets[accepted],
        accepted == drafts.size(),
    };
}

// Backend-local controller with the same scheduling contract as Metal. It
// measures target-only and speculative widths, learns conditional acceptance,
// and selects the width with the highest expected emitted tokens per ms.
class DepthController {
public:
    explicit DepthController(int maximum_depth = 3)
        : maximum_depth_(std::clamp(maximum_depth, 1, kMaximumDraftDepth)),
          current_depth_(maximum_depth_),
          acceptance_(static_cast<std::size_t>(maximum_depth_), 0.6),
          warmup_accepts_(static_cast<std::size_t>(maximum_depth_), 0),
          warmup_trials_(static_cast<std::size_t>(maximum_depth_), 0),
          cycle_ms_(static_cast<std::size_t>(maximum_depth_ + 1)),
          cycle_age_ms_(static_cast<std::size_t>(maximum_depth_ + 1)) {
        warmup_.insert(warmup_.end(), {maximum_depth_, maximum_depth_});
        warmup_.insert(warmup_.end(), {0, 0, 0});
    }

    int depth() const noexcept { return current_depth_; }
    int maximum_depth() const noexcept { return maximum_depth_; }

    void observe(
            int used_depth,
            int accepted_drafts,
            double cycle_ms,
            bool time_sample = true) {
        ++cycles_;
        used_depth = std::clamp(used_depth, 0, maximum_depth_);
        accepted_drafts = std::clamp(accepted_drafts, 0, used_depth);
        for (int position = 0; position < used_depth; ++position) {
            const double hit = position < accepted_drafts ? 1.0 : 0.0;
            const auto index = static_cast<std::size_t>(position);
            auto& estimate = acceptance_[index];
            if (!warmup_.empty()) {
                warmup_accepts_[index] += hit > 0.0 ? 1 : 0;
                ++warmup_trials_[index];
                estimate = static_cast<double>(warmup_accepts_[index]) /
                    static_cast<double>(warmup_trials_[index]);
            } else {
                estimate = (1.0 - kAcceptanceAlpha) * estimate +
                    kAcceptanceAlpha * hit;
            }
            if (position >= accepted_drafts) break;
        }

        cycle_ms = std::max(0.0, cycle_ms);
        if (time_sample) update_time(used_depth, cycle_ms);
        for (auto& age : cycle_age_ms_) {
            if (age) *age += cycle_ms;
        }
        if (time_sample) {
            cycle_age_ms_[static_cast<std::size_t>(used_depth)] = 0.0;
        }
        milliseconds_since_probe_ += cycle_ms;
        milliseconds_since_explore_ += cycle_ms;
        exit_streak_ = speculation_losing() ? exit_streak_ + 1 : 0;

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
        if (maximum_depth_ <= 1) return;
        const double period = std::max(
            kProbePeriodMs,
            static_cast<double>(kProbeLength) * cycle_ms / kProbeDuty);
        if (milliseconds_since_probe_ < period) return;
        const bool explore_due = milliseconds_since_explore_ >=
            std::max(kExplorePeriodMs, 2.0 * period);
        const auto target = explore_due ? most_stale() : best_rival();
        if (!target) return;
        current_depth_ = *target;
        probe_left_ = kProbeLength;
        milliseconds_since_probe_ = 0.0;
        if (explore_due) milliseconds_since_explore_ = 0.0;
    }

    bool should_exit() const noexcept { return exit_streak_ >= kExitStreak; }

    double conditional_acceptance(int position) const {
        if (position < 0 || position >= maximum_depth_) {
            throw std::out_of_range("MTP acceptance position is out of range");
        }
        return acceptance_[static_cast<std::size_t>(position)];
    }

    std::optional<double> measured_cycle_ms(int depth) const {
        if (depth < 0 || depth > maximum_depth_) {
            throw std::out_of_range("MTP measured depth is out of range");
        }
        return cycle_ms_[static_cast<std::size_t>(depth)];
    }

private:
    static constexpr double kAcceptanceAlpha = 0.08;
    static constexpr double kTimeTauMs = 400.0;
    static constexpr double kProbePeriodMs = 1000.0;
    static constexpr double kExplorePeriodMs = 5000.0;
    static constexpr int kProbeLength = 4;
    static constexpr double kProbeDuty = 0.15;
    static constexpr double kProbeMargin = 1.15;
    static constexpr double kSpikeRatio = 2.0;
    static constexpr double kSpikeDamp = 0.25;
    static constexpr double kMarginalMs = 7.0;
    static constexpr double kHysteresis = 1.03;
    static constexpr double kExitMargin = 1.15;
    static constexpr int kExitStreak = 16;

    void update_time(int depth, double cycle_ms) {
        auto& estimate = cycle_ms_[static_cast<std::size_t>(depth)];
        if (!estimate) {
            estimate = cycle_ms;
            return;
        }
        if (!warmup_.empty()) {
            *estimate = std::min(*estimate, cycle_ms);
            return;
        }
        double alpha = 1.0 - std::exp(-cycle_ms / kTimeTauMs);
        if (cycle_ms > kSpikeRatio * *estimate) alpha *= kSpikeDamp;
        *estimate = (1.0 - alpha) * *estimate + alpha * cycle_ms;
    }

    double marginal_estimate() const {
        int low = -1;
        int high = -1;
        for (int depth = 0; depth <= maximum_depth_; ++depth) {
            if (!cycle_ms_[static_cast<std::size_t>(depth)]) continue;
            if (low < 0) low = depth;
            high = depth;
        }
        if (low >= 0 && high > low) {
            const double slope =
                (*cycle_ms_[static_cast<std::size_t>(high)] -
                 *cycle_ms_[static_cast<std::size_t>(low)]) /
                static_cast<double>(high - low);
            if (slope > 0.0) return slope;
        }
        return kMarginalMs;
    }

    double time_estimate(int depth) const {
        if (cycle_ms_[static_cast<std::size_t>(depth)]) {
            return *cycle_ms_[static_cast<std::size_t>(depth)];
        }
        int reference = -1;
        for (int candidate = 0; candidate <= maximum_depth_; ++candidate) {
            if (!cycle_ms_[static_cast<std::size_t>(candidate)]) continue;
            if (reference < 0 ||
                std::abs(candidate - depth) < std::abs(reference - depth)) {
                reference = candidate;
            }
        }
        if (reference < 0) return 30.0 + kMarginalMs * depth;
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

    double score(int depth) const {
        double expected = 1.0;
        double run = 1.0;
        for (int position = 0; position < depth; ++position) {
            run *= acceptance_[static_cast<std::size_t>(position)];
            expected += run;
        }
        return expected / std::max(1.0e-6, time_estimate(depth));
    }

    bool speculation_losing() const {
        if (!warmup_.empty() || !cycle_ms_.front()) return false;
        const double baseline = score(0);
        double best = 0.0;
        for (int depth = 1; depth <= maximum_depth_; ++depth) {
            best = std::max(best, score(depth));
        }
        return baseline > 0.0 && best < baseline * kExitMargin;
    }

    int best_depth() const {
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
            best_score < score(current_depth_) * kHysteresis) {
            return current_depth_;
        }
        return best;
    }

    std::optional<int> best_rival() const {
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
        if (current_score <= 0.0) return most_stale();
        return rival && rival_score >= current_score / kProbeMargin
            ? rival
            : std::nullopt;
    }

    std::optional<int> most_stale() const {
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
        for (int depth = 1; depth <= maximum_depth_; ++depth) consider(depth);
        consider(0);
        return result;
    }

    int maximum_depth_ = 1;
    int current_depth_ = 1;
    int cycles_ = 0;
    int probe_left_ = 0;
    int exit_streak_ = 0;
    double milliseconds_since_probe_ = 0.0;
    double milliseconds_since_explore_ = 0.0;
    std::vector<double> acceptance_;
    std::vector<int> warmup_accepts_;
    std::vector<int> warmup_trials_;
    std::vector<std::optional<double>> cycle_ms_;
    std::vector<std::optional<double>> cycle_age_ms_;
    std::vector<int> warmup_;
};

inline std::vector<float> distribution(std::span<const float> logits,
                                      double temperature, int top_k, double top_p) {
    if (logits.empty() || !std::isfinite(temperature) || temperature <= 0. ||
        top_k < 0 || top_k > 1024 || top_k > static_cast<int>(logits.size()) ||
        !std::isfinite(top_p) || top_p <= 0. || top_p > 1.) {
        throw std::invalid_argument("invalid CUDA MTP sampling geometry");
    }
    // Preserve the existing CUDA sampler: top_k=0 selects full softmax,
    // whose current implementation does not apply a nucleus cutoff.
    if (top_k == 0) top_p = 1.;
    const int vocab = static_cast<int>(logits.size());
    std::vector<int> order(vocab);
    std::iota(order.begin(), order.end(), 0);
    const auto value = [&](int token) {
        return std::isnan(logits[token]) ? -std::numeric_limits<float>::infinity() : logits[token];
    };
    const auto before = [&](int a, int b) {
        return value(a) > value(b) || (value(a) == value(b) && a < b);
    };
    const int selected = top_k > 0 ? top_k : vocab;
    if (top_k > 0) {
        std::partial_sort(order.begin(), order.begin() + selected, order.end(), before);
        order.resize(selected);
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (int token : order) maximum = std::max(maximum, value(token));
    if (!std::isfinite(maximum)) throw std::runtime_error("MTP logits have no finite maximum");
    std::vector<double> masses(selected);
    double total = 0.;
    for (int i = 0; i < selected; ++i) {
        masses[i] = std::exp((double(value(order[i])) - maximum) / temperature);
        total += masses[i];
    }
    if (!(total > 0.) || !std::isfinite(total)) throw std::runtime_error("invalid MTP softmax mass");
    int keep = selected;
    double kept_mass = total;
    if (top_p < 1.) {
        kept_mass = 0.;
        for (int i = 0; i < selected; ++i) {
            kept_mass += masses[i];
            if (kept_mass >= top_p * total) { keep = i + 1; break; }
        }
    }
    std::vector<float> probabilities(vocab, 0.f);
    for (int i = 0; i < keep; ++i) probabilities[order[i]] = static_cast<float>(masses[i] / kept_mass);
    return probabilities;
}

inline void validate_distribution(std::span<const float> probabilities) {
    double sum = 0.;
    for (float p : probabilities) {
        if (!std::isfinite(p) || p < 0.f) throw std::invalid_argument("invalid MTP probability");
        sum += p;
    }
    if (probabilities.empty() || std::abs(sum - 1.) > 1.e-5)
        throw std::invalid_argument("MTP probability vector is not normalized");
}

inline int32_t sample(std::span<const float> probabilities, double uniform) {
    validate_distribution(probabilities);
    if (!std::isfinite(uniform)) throw std::invalid_argument("invalid MTP sampling uniform");
    uniform = std::clamp(uniform, 0., std::nextafter(1., 0.));
    double cumulative = 0.;
    int32_t fallback = -1;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        if (probabilities[i] <= 0.f) continue;
        fallback = static_cast<int32_t>(i);
        cumulative += probabilities[i];
        if (uniform < cumulative) return fallback;
    }
    return fallback;
}

struct Verification {
    bool accepted;
    int32_t next_token;
};

inline Verification verify(int32_t draft, std::span<const float> proposal,
                           std::span<const float> target, std::span<const float> bonus,
                           double acceptance_uniform, double sample_uniform) {
    if (proposal.size() != target.size() || target.size() != bonus.size() ||
        draft < 0 || static_cast<size_t>(draft) >= proposal.size() ||
        !std::isfinite(acceptance_uniform) || !std::isfinite(sample_uniform)) {
        throw std::invalid_argument("incompatible MTP verification inputs");
    }
    validate_distribution(proposal);
    validate_distribution(target);
    validate_distribution(bonus);
    const double q = proposal[draft], p = target[draft];
    if (!(q > 0.)) throw std::invalid_argument("draft has zero proposal probability");
    const double accept = std::min(1., p / q);
    if (std::clamp(acceptance_uniform, 0., std::nextafter(1., 0.)) < accept)
        return {true, sample(bonus, sample_uniform)};
    std::vector<float> correction(proposal.size());
    double total = 0.;
    for (size_t i = 0; i < proposal.size(); ++i) {
        correction[i] = std::max(0.f, target[i] - proposal[i]);
        total += correction[i];
    }
    if (!(total > 0.) || !std::isfinite(total)) return {false, sample(target, sample_uniform)};
    for (auto& p_correct : correction) p_correct = static_cast<float>(p_correct / total);
    return {false, sample(correction, sample_uniform)};
}

inline ChainVerification verify_stochastic_chain(
        std::span<const int32_t> drafts,
        const std::vector<std::vector<float>>& proposals,
        const std::vector<std::vector<float>>& targets,
        std::span<const double> acceptance_uniforms,
        double sample_uniform) {
    if (drafts.empty() || proposals.size() != drafts.size() ||
        targets.size() != drafts.size() + 1 ||
        acceptance_uniforms.size() != drafts.size()) {
        throw std::invalid_argument("stochastic MTP chain shapes are incompatible");
    }
    for (std::size_t position = 0; position < drafts.size(); ++position) {
        const auto draft = drafts[position];
        const auto& proposal = proposals[position];
        const auto& target = targets[position];
        if (proposal.empty() || proposal.size() != target.size() || draft < 0 ||
            static_cast<std::size_t>(draft) >= proposal.size()) {
            throw std::invalid_argument("stochastic MTP chain distribution is incompatible");
        }
        validate_distribution(proposal);
        validate_distribution(target);
        const double q = proposal[static_cast<std::size_t>(draft)];
        const double p = target[static_cast<std::size_t>(draft)];
        if (!(q > 0.0) || !std::isfinite(acceptance_uniforms[position])) {
            throw std::invalid_argument("stochastic MTP chain probability is invalid");
        }
        const double acceptance = std::min(1.0, p / q);
        if (std::clamp(
                acceptance_uniforms[position], 0.0,
                std::nextafter(1.0, 0.0)) < acceptance) {
            continue;
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
        return {position, sample(correction, sample_uniform), false};
    }
    validate_distribution(targets.back());
    return {drafts.size(), sample(targets.back(), sample_uniform), true};
}

}  // namespace mfq::cuda::mtp
