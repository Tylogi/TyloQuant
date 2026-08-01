#pragma once

#include <cstdint>
#include <optional>
#include <random>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxSamplingParams {
    double temperature = 0.0;
    int top_k = 0;
    double top_p = 1.0;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    double repetition_penalty = 1.0;
    std::uint64_t seed = 0;

    bool greedy() const noexcept {
        return temperature <= 0.0 || top_k == 1;
    }

    bool has_penalties() const noexcept {
        return presence_penalty != 0.0 ||
            frequency_penalty != 0.0 ||
            repetition_penalty != 1.0;
    }
};

mlx::core::array sample_greedy(const mlx::core::array& logits);

mlx::core::array sample_softmax(
    const mlx::core::array& logits,
    const mlx::core::array& random,
    double temperature = 1.0);

mlx::core::array sample_top_k_top_p(
    const mlx::core::array& logits,
    const mlx::core::array& random,
    double temperature,
    int top_k,
    double top_p = 1.0);

// Stateless sampling. When random is omitted, the first random values produced
// by params.seed are used. Use MlxSampler for a generation-length RNG stream.
mlx::core::array sample(
    const mlx::core::array& logits,
    const MlxSamplingParams& params = {},
    const std::optional<mlx::core::array>& random = std::nullopt);

mlx::core::array sample_token_counts_add(
    const mlx::core::array& counts,
    const mlx::core::array& tokens);

mlx::core::array sample_apply_penalties(
    const mlx::core::array& logits,
    const mlx::core::array& counts,
    double presence_penalty = 0.0,
    double frequency_penalty = 0.0,
    double repetition_penalty = 1.0);

class MlxSampler {
public:
    explicit MlxSampler(MlxSamplingParams params = {});

    const MlxSamplingParams& params() const noexcept {
        return params_;
    }

    void reset_seed(std::uint64_t seed);

    // Sample raw logits using this sampler's persistent seeded RNG stream.
    mlx::core::array sample(const mlx::core::array& logits);

    // Apply configured penalties using counts, then sample the adjusted logits.
    mlx::core::array sample(
        const mlx::core::array& logits,
        const mlx::core::array& counts);

    mlx::core::array apply_penalties(
        const mlx::core::array& logits,
        const mlx::core::array& counts) const;

private:
    mlx::core::array next_random(const mlx::core::array& logits);

    MlxSamplingParams params_;
    std::mt19937_64 rng_;
};

} // namespace mfq::metal
