#pragma once

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <vector>
#include <random>
#include <stdexcept>
#include <utility>

namespace mfq::engine {

// CPU reference sampling used by evaluators that require float arithmetic and
// their supplied RNG. Temperature and penalties have already been applied.
template <class Rng>
std::int64_t sample_top_k_top_p(std::span<const float> logits, std::int64_t top_k, double top_p,
                                std::int64_t minimum_keep, Rng &rng) {
    if (logits.empty() || top_k <= 0 || top_k > static_cast<std::int64_t>(logits.size()) ||
        !(top_p >= 0.0 && top_p <= 1.0) || minimum_keep < 0 || minimum_keep > top_k)
        throw std::invalid_argument("invalid reference sampling geometry");
    const auto maximum = *std::max_element(logits.begin(), logits.end());
    if (!std::isfinite(maximum))
        throw std::invalid_argument("no finite sampling scores");
    // A zero nucleus always retains the highest-scoring token, even with a
    // caller-supplied minimum_keep, and never constructs an empty distribution.
    if (top_p == 0.0)
        return std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
    std::vector<std::pair<float, std::int64_t>> probabilities;
    probabilities.reserve(logits.size());
    float sum = 0.0F;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const auto probability = std::exp(logits[i] - maximum);
        probabilities.emplace_back(probability, i);
        sum += probability;
    }
    if (!std::isfinite(sum) || sum <= 0.0F)
        throw std::invalid_argument("invalid sampling probabilities");
    for (auto &probability : probabilities)
        probability.first /= sum;
    std::sort(probabilities.begin(), probabilities.end(),
              [](const auto &left, const auto &right) { return left.first > right.first; });
    std::vector<float> kept_probabilities;
    std::vector<std::int64_t> kept_indices;
    kept_probabilities.reserve(top_k);
    kept_indices.reserve(top_k);
    float cumulative = 0.0F;
    for (const auto &probability : probabilities) {
        if (static_cast<std::int64_t>(kept_probabilities.size()) < minimum_keep ||
            (cumulative < static_cast<float>(top_p) &&
             static_cast<std::int64_t>(kept_probabilities.size()) < top_k)) {
            cumulative += probability.first;
            kept_probabilities.push_back(probability.first);
            kept_indices.push_back(probability.second);
        } else
            break;
    }
    const float kept_sum =
        std::accumulate(kept_probabilities.begin(), kept_probabilities.end(), 0.0F);
    for (auto &probability : kept_probabilities)
        probability /= kept_sum;
    std::uniform_real_distribution<float> distribution(0.0F, 1.0F);
    const float sample = distribution(rng);
    float selected_sum = 0.0F;
    for (std::size_t i = 0; i < kept_probabilities.size(); ++i) {
        selected_sum += kept_probabilities[i];
        if (sample <= selected_sum)
            return kept_indices[i];
    }
    return kept_indices.back();
}

// Ops supplies:
//   using Tensor;
//   using Params;
//   Tensor sample_greedy(Tensor);
//   Tensor sample_stochastic(Tensor, float random, const Params&);
//   Tensor apply_penalties(Tensor, const Tensor& counts, const Params&);
// Tensor arguments are consumed, so an Ops implementation may update them in
// place. Callers that still need the original tensor must clone it first.
template <typename Ops>
class Sampler {
public:
    using Tensor = typename Ops::Tensor;
    using Params = typename Ops::Params;

    explicit Sampler(Params params = Params{}, Ops ops = Ops{})
        : params_(std::move(params)),
          ops_(std::move(ops)),
          rng_(params_.seed) {
        if (params_.top_k < 0) {
            throw std::invalid_argument("top_k must be non-negative");
        }
    }

    const Params& params() const noexcept {
        return params_;
    }

    Ops& ops() noexcept {
        return ops_;
    }

    const Ops& ops() const noexcept {
        return ops_;
    }

    bool greedy() const noexcept {
        return params_.temperature <= 0.0 || params_.top_k == 1 || params_.top_p == 0.0;
    }

    bool has_penalties() const noexcept {
        return params_.presence_penalty != 0.0 ||
            params_.frequency_penalty != 0.0 ||
            params_.repetition_penalty != 1.0;
    }

    void reset_seed(std::uint64_t seed) {
        params_.seed = seed;
        rng_.seed(seed);
    }

    void discard_random(std::uint64_t draws) {
        rng_.discard(draws);
    }

    float next_uniform_float() {
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        return uniform(rng_);
    }

    Tensor sample(Tensor logits) {
        if (greedy()) {
            return ops_.sample_greedy(std::move(logits));
        }
        return ops_.sample_stochastic(
            std::move(logits), next_uniform_float(), params_);
    }

    Tensor sample(Tensor logits, const Tensor& counts) {
        if (has_penalties()) {
            logits = apply_penalties(std::move(logits), counts);
        }
        return sample(std::move(logits));
    }

    Tensor apply_penalties(Tensor logits, const Tensor& counts) {
        return ops_.apply_penalties(
            std::move(logits), counts, params_);
    }

    double next_uniform() {
        return std::generate_canonical<double, 53>(rng_);
    }

private:
    Params params_;
    Ops ops_;
    std::mt19937_64 rng_;
};

} // namespace mfq::engine
