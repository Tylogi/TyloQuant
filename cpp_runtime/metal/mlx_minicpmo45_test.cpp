#include "mlx_minicpmo45.h"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        mfq::metal::detail::test_minicpmo45_qwen3_cache_equivalence();
        mfq::metal::detail::test_minicpmo45_qk_norm_rope();
        mfq::metal::detail::test_minicpmo45_gqa_attention();
        if (argc == 2) {
            const mfq::metal::MfqContainer model(argv[1]);
            auto runtime = mfq::metal::MlxMiniCPMO45Runtime::load(
                model, 2048, true);
            if (runtime.layer_count() != 36 ||
                runtime.maximum_context() != 2048 ||
                runtime.vocabulary_size() <= 0) {
                throw std::runtime_error(
                    "MiniCPM-o 4.5 full-model integration geometry mismatch");
            }
            std::cout
                << "MFQ MiniCPM-o 4.5 full MLX composite loaded: layers="
                << runtime.layer_count()
                << " context=" << runtime.maximum_context()
                << " vocab=" << runtime.vocabulary_size() << '\n';
        } else if (argc != 1) {
            throw std::runtime_error(
                "usage: mfq-metal-minicpmo45-test [MODEL.mfq]");
        }
        std::cout
            << "MFQ MiniCPM-o 4.5 Qwen3 BF16/cache numerical tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
