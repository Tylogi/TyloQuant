#include "mfq_cuda_mtp.h"
#include <array>
#include <iostream>

namespace {
void require(bool ok) { if (!ok) throw std::runtime_error("MTP policy test failed"); }
template<class F> void rejects(F&& fn) {
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    require(rejected);
}
}

int main() {
    using namespace mfq::cuda::mtp;
    {
        DepthController controller(3);
        require(controller.depth() == 3);
        controller.observe(3, 3, 30.0);
        controller.observe(3, 3, 29.0);
        controller.observe(0, 0, 22.0);
        controller.observe(0, 0, 21.0);
        controller.observe(0, 0, 23.0);
        require(controller.depth() == 3);
        require(controller.measured_cycle_ms(3).has_value());
        require(controller.conditional_acceptance(0) > 0.6);
    }
    {
        DepthController controller(1);
        controller.observe(1, 0, 80.0);
        controller.observe(1, 0, 75.0);
        controller.observe(0, 0, 40.0);
        controller.observe(0, 0, 39.0);
        controller.observe(0, 0, 41.0);
        require(controller.depth() == 0);
        require(controller.measured_cycle_ms(0).has_value());
    }
    {
        DepthController controller(3);
        controller.observe(3, 0, 70.0);
        controller.observe(3, 0, 65.0);
        controller.observe(0, 0, 30.0);
        controller.observe(0, 0, 29.0);
        controller.observe(0, 0, 31.0);
        require(controller.depth() == 0);
        for (int cycle = 0; cycle < 15; ++cycle) {
            controller.observe(0, 0, 30.0);
        }
        require(!controller.should_exit());
        controller.observe(0, 0, 30.0);
        require(controller.should_exit());
    }
    {
        const std::array<int32_t, 4> drafts{11, 12, 13, 14};
        const std::array<int32_t, 5> partial{11, 12, 99, 14, 15};
        const auto result = verify_greedy(drafts, partial);
        require(result.accepted_drafts == 2 && result.next_token == 99 && !result.bonus);
        const std::array<int32_t, 5> complete{11, 12, 13, 14, 15};
        const auto bonus_result = verify_greedy(drafts, complete);
        require(bonus_result.accepted_drafts == 4 && bonus_result.next_token == 15 && bonus_result.bonus);
        rejects([&] { verify_greedy(drafts, std::span<const int32_t>(partial).first(4)); });
        rejects([&] { verify_greedy(std::span<const int32_t>{}, std::span<const int32_t>{}); });
    }
    {
        const std::array<int32_t, 2> drafts{0, 1};
        const std::vector<std::vector<float>> proposal{
            {.5f, .5f}, {.25f, .75f}};
        const std::vector<std::vector<float>> target{
            {.75f, .25f}, {.5f, .5f}, {0.f, 1.f}};
        const std::array<double, 2> accept_all{.5, .5};
        const auto accepted = verify_stochastic_chain(
            drafts, proposal, target, accept_all, .5);
        require(accepted.accepted_drafts == 2 && accepted.next_token == 1 && accepted.bonus);
        const std::array<double, 2> reject_second{.5, .95};
        const auto rejected = verify_stochastic_chain(
            drafts, proposal, target, reject_second, .5);
        require(rejected.accepted_drafts == 1 && rejected.next_token == 0 && !rejected.bonus);
    }
    const std::array<float, 3> q{.5f, .25f, .25f}, p{.25f, .5f, .25f}, bonus{0.f, 0.f, 1.f};
    require(verify(0, q, p, bonus, .49, .2).accepted);
    require(verify(0, q, p, bonus, .49, .2).next_token == 2);
    require(!verify(0, q, p, bonus, .5, .2).accepted);
    require(verify(0, q, p, bonus, .5, .2).next_token == 1);
    require(verify(1, q, p, bonus, .999, .2).accepted);
    require(sample(bonus, 0.) == 2 && sample(bonus, 1.) == 2);
    rejects([&] { verify(-1, q, p, bonus, .2, .3); });
    rejects([&] { verify(0, bonus, p, bonus, .2, .3); });
    rejects([&] { verify(0, q, p, bonus, NAN, .3); });
    rejects([&] { sample(std::array<float, 3>{1.f, 0.f, NAN}, 0.); });
    rejects([&] { sample(std::array<float, 3>{.1f, .1f, .1f}, 0.); });
    const std::array<float, 4> ties{1.f, 1.f, 1.f, 0.f};
    auto top = distribution(ties, 1., 2, 1.);
    require(top[0] == .5f && top[1] == .5f && top[2] == 0.f && top[3] == 0.f);
    auto nucleus = distribution(ties, 1., 2, .4);
    require(nucleus[0] == 1.f && nucleus[1] == 0.f);
    auto all = distribution(ties, 1., 0, .1);
    require(all[0] > 0.f && all[1] > 0.f && all[2] > 0.f && all[3] > 0.f);
    rejects([&] { distribution(ties, 0., 2, 1.); });
    rejects([&] { distribution(ties, 1., 5, 1.); });

    // Enumerate a deterministic uniform grid: accepted drafts plus residual
    // replacements must recover p, despite a deliberately biased proposal q.
    std::array<int, 3> histogram{};
    for (int draw = 0; draw < 400; ++draw) {
        const auto draft = sample(q, (draw + .5) / 400.);
        for (int accept = 0; accept < 100; ++accept) {
            const auto result = verify(draft, q, p, bonus, (accept + .5) / 100., .5);
            ++histogram[result.accepted ? draft : result.next_token];
        }
    }
    require(histogram[0] == 10000 && histogram[1] == 20000 && histogram[2] == 10000);
    std::cout << "MTP sampling/acceptance/correction tests passed; 40000 exact grid checks\n";
}
