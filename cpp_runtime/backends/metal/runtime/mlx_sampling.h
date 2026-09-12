#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

struct MlxSamplingParams {
    double temperature = 0.0;
    int top_k = 0;
    double top_p = 1.0;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    double repetition_penalty = 1.0;
    bool enable_mtp = true;
    int mtp_max_draft_tokens = 3;
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

struct MlxTopKDistribution {
    mlx::core::array sampled;
    mlx::core::array indices;
    mlx::core::array probabilities;
};

// Select the final time step from [batch,tokens,vocab] logits and return
// [batch,vocab]. Generation adapters must not carry private copies of this
// shape policy.
mlx::core::array mlx_last_token_logits(
    const mlx::core::array& logits,
    int vocab);

void mlx_validate_token_set(
    std::span<const std::int64_t> token_ids,
    int vocab);

bool mlx_token_in_set(
    std::span<const std::int64_t> token_ids,
    std::int64_t token) noexcept;

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

// Return the sampled token and the compact, filtered top-k distribution in a
// GPU-resident pass. The final dimension of indices/probabilities is top_k;
// entries removed by top-p have zero probability. Values up to 64 use the
// direct kernel and values up to 128 use exact hierarchical selection. This
// is used by speculative verification to avoid materializing a full
// vocabulary on CPU.
MlxTopKDistribution sample_top_k_distribution(
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

// Exact host-visible distribution used by stochastic speculative decoding.
// It applies the same temperature/top-k/top-p policy as ordinary sampling.
// The expensive path is confined to MTP verification, where proposal and
// target probabilities are required by the acceptance rule.
std::vector<float> host_sampling_distribution(
    const mlx::core::array& logits,
    const MlxSamplingParams& params);

std::int32_t sample_host_distribution(
    const std::vector<float>& probabilities,
    double uniform);

class MlxSampler {
public:
    explicit MlxSampler(MlxSamplingParams params = {});

    const MlxSamplingParams& params() const noexcept {
        return params_;
    }

    void reset_seed(std::uint64_t seed);

    // Advance the persistent RNG when an adapter sampled one or more tokens
    // before handing generation to the common MTP engine.
    void discard_random(std::uint64_t draws);

    // Sample raw logits using this sampler's persistent seeded RNG stream.
    mlx::core::array sample(const mlx::core::array& logits);

    // Apply configured penalties using counts, then sample the adjusted logits.
    mlx::core::array sample(
        const mlx::core::array& logits,
        const mlx::core::array& counts);

    mlx::core::array apply_penalties(
        const mlx::core::array& logits,
        const mlx::core::array& counts) const;

    double next_uniform();

private:
    mlx::core::array next_random(const mlx::core::array& logits);

    MlxSamplingParams params_;
    std::mt19937_64 rng_;
};

} // namespace mfq::metal
