#include "sampling.h"

#include <cassert>
#include <cstdint>

namespace {

struct FakeParams {
    double temperature = 1.0;
    int top_k = 0;
    double top_p = 1.0;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    double repetition_penalty = 1.0;
    std::uint64_t seed = 0;
};

struct FakeTensor {
    double value = 0.0;
};

struct FakeOps {
    using Tensor = FakeTensor;
    using Params = FakeParams;

    Tensor sample_greedy(Tensor logits) {
        return logits;
    }

    Tensor sample_stochastic(
            Tensor logits, float random, const Params&) {
        logits.value += random;
        return logits;
    }

    Tensor apply_penalties(
            Tensor logits, const Tensor& counts, const Params&) {
        logits.value += counts.value;
        return logits;
    }
};

} // namespace

int main() {
    FakeParams stochastic;
    stochastic.seed = 42;
    mfq::engine::Sampler<FakeOps> first(stochastic);
    mfq::engine::Sampler<FakeOps> second(stochastic);
    assert(first.sample({}).value == second.sample({}).value);
    assert(first.next_uniform_float() == second.next_uniform_float());

    stochastic.presence_penalty = 1.0;
    mfq::engine::Sampler<FakeOps> penalized(stochastic);
    const auto sampled = penalized.sample(
        FakeTensor{2.0}, FakeTensor{3.0});
    assert(sampled.value >= 5.0 && sampled.value < 6.0);

    FakeParams greedy;
    greedy.top_k = 1;
    greedy.seed = 7;
    mfq::engine::Sampler<FakeOps> greedy_sampler(greedy);
    assert(greedy_sampler.sample(FakeTensor{9.0}).value == 9.0);
    for (int top_k : {0, 1, 20}) {
        greedy.top_k = top_k;
        greedy.top_p = 0.0;
        mfq::engine::Sampler<FakeOps> zero_nucleus(greedy);
        mfq::engine::Sampler<FakeOps> fresh_greedy(greedy);
        assert(zero_nucleus.greedy());
        assert(zero_nucleus.sample(FakeTensor{9.0}).value == 9.0);
        assert(zero_nucleus.next_uniform_float() == fresh_greedy.next_uniform_float());
    }
    std::mt19937 rng(42);
    const float scores[]{0, 1, 5, -1};
    for (int minimum : {0, 1, 3})
        assert(mfq::engine::sample_top_k_top_p(scores, 3, 0.0, minimum, rng) == 2);
    return 0;
}
