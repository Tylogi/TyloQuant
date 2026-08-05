#include "mfq_container.h"
#include "mlx_deepseek_v4_causal_lm.h"
#include "../mfq_server.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

namespace {

using mlx::core::Shape;
using mlx::core::array;
using mfq::metal::MfqContainer;
using mfq::metal::MlxDeepseekV4CausalLm;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_text(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open audit text");
    return {std::istreambuf_iterator<char>(input), {}};
}

std::vector<float> evaluated(array value) {
    value = mlx::core::astype(value, mlx::core::float32);
    value.eval();
    return {value.data<float>(), value.data<float>() + value.size()};
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3 || argc == 4,
            "usage: mfq-metal-deepseek-v4-real-cache-audit MODEL.mfq TEXT [TOKENS]");
        const MfqContainer container(argv[1]);
        constexpr const char* tokenizer_asset =
            "__mfq_asset__/tokenizer.gguf";
        require(container.contains(tokenizer_asset),
            "model has no embedded tokenizer");
        auto probe = probe_mfq_tokenizer(
            container.read(tokenizer_asset),
            read_text(argv[2]),
            true,
            true);
        int count = argc == 4 ? std::stoi(argv[3]) : 32;
        count = std::min(count, static_cast<int>(probe.tokens.size()));
        require(count >= 2, "audit text tokenized to fewer than two tokens");
        std::vector<std::int32_t> tokens(
            probe.tokens.begin(), probe.tokens.begin() + count);

        auto model = MlxDeepseekV4CausalLm::load(
            container,
            count + 8);
        const int vocab = static_cast<int>(model.config().vocab);
        const array ids(tokens.begin(), Shape{1, count}, mlx::core::int32);
        const auto full = evaluated(model.forward(ids, false));
        require(full.size() == static_cast<std::size_t>(count) * vocab,
            "full-forward logits shape mismatch");

        model.clear_cache();
        std::vector<float> cached;
        cached.reserve(full.size());
        for (int position = 0; position < count; ++position) {
            const array one(
                {tokens[static_cast<std::size_t>(position)]},
                Shape{1, 1},
                mlx::core::int32);
            auto logits = model.forward(one, position != 0);
            auto values = evaluated(std::move(logits));
            require(values.size() == static_cast<std::size_t>(vocab),
                "cached logits shape mismatch");
            cached.insert(cached.end(), values.begin(), values.end());
        }

        double total_abs = 0.0;
        double total_kld = 0.0;
        float maximum_abs = 0.0f;
        double maximum_kld = 0.0;
        int top1_matches = 0;
        int worst_position = -1;
        for (int position = 0; position < count; ++position) {
            const auto offset = static_cast<std::size_t>(position) * vocab;
            const float* reference = full.data() + offset;
            const float* decode = cached.data() + offset;
            const int reference_top = static_cast<int>(
                std::max_element(reference, reference + vocab) - reference);
            const int decode_top = static_cast<int>(
                std::max_element(decode, decode + vocab) - decode);
            top1_matches += reference_top == decode_top;
            float row_max_abs = 0.0f;
            double reference_max = -std::numeric_limits<double>::infinity();
            double decode_max = -std::numeric_limits<double>::infinity();
            for (int item = 0; item < vocab; ++item) {
                row_max_abs = std::max(
                    row_max_abs,
                    std::fabs(reference[item] - decode[item]));
                total_abs += std::fabs(reference[item] - decode[item]);
                reference_max = std::max(reference_max, double(reference[item]));
                decode_max = std::max(decode_max, double(decode[item]));
            }
            double reference_sum = 0.0;
            double decode_sum = 0.0;
            for (int item = 0; item < vocab; ++item) {
                reference_sum += std::exp(double(reference[item]) - reference_max);
                decode_sum += std::exp(double(decode[item]) - decode_max);
            }
            const double reference_logz = reference_max + std::log(reference_sum);
            const double decode_logz = decode_max + std::log(decode_sum);
            double row_kld = 0.0;
            for (int item = 0; item < vocab; ++item) {
                const double logp = double(reference[item]) - reference_logz;
                const double logq = double(decode[item]) - decode_logz;
                row_kld += std::exp(logp) * (logp - logq);
            }
            if (row_kld > maximum_kld) {
                maximum_kld = row_kld;
                worst_position = position;
            }
            maximum_abs = std::max(maximum_abs, row_max_abs);
            total_kld += row_kld;
            std::cout << "position=" << position
                << " kld=" << row_kld
                << " max_abs=" << row_max_abs
                << " top_full=" << reference_top
                << " top_decode=" << decode_top << "\n";
        }
        std::cout << "summary tokens=" << count
            << " mean_abs=" << total_abs / (double(count) * vocab)
            << " max_abs=" << maximum_abs
            << " mean_kld=" << total_kld / count
            << " max_kld=" << maximum_kld
            << " worst_position=" << worst_position
            << " top1_match=" << top1_matches << "/" << count
            << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cache-audit: " << error.what() << "\n";
        return 1;
    }
}
