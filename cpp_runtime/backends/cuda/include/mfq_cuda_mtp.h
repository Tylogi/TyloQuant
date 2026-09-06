#pragma once

// Host reference policy for the native CUDA MTP chain. The acceptance and
// positive-residual correction match Metal runtime/mlx_mtp.cpp at e27c987.
// Inputs to distribution() have already received the ordinary CUDA penalties.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace mfq::cuda::mtp {

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

}  // namespace mfq::cuda::mtp
