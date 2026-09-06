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
