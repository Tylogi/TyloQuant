#include "mfq_container.h"
#include "../mfq_server.h"

#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

constexpr const char* kTokenizerAsset =
    "__mfq_asset__/tokenizer.gguf";

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(
            argc == 2 || argc == 3,
            "usage: mfq-metal-tokenizer-test MODEL.mfq [TEXT]");
        const mfq::metal::MfqContainer model(argv[1]);
        require(
            model.contains(kTokenizerAsset),
            "MFQ has no embedded tokenizer GGUF");
        const auto probe = probe_mfq_tokenizer(
            model.read(kTokenizerAsset),
            argc == 3 ? argv[2] : "你好，MFQ");
        require(
            probe.vocab_size > 0,
            "embedded tokenizer has an empty vocabulary");
        require(
            !probe.tokens.empty(),
            "embedded tokenizer returned no tokens");
        require(
            !probe.chat_template.empty(),
            "embedded tokenizer has no chat template");
        require(
            probe.eos_token >= 0 || probe.eot_token >= 0,
            "embedded tokenizer has no end-of-generation token");
        if (std::getenv("MFQ_TOKENIZER_DUMP_TEMPLATE") != nullptr) {
            std::cout << probe.chat_template << "\n";
        }
        if (argc == 3) {
            std::cout << "token_ids=";
            for (std::size_t index = 0; index < probe.tokens.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << probe.tokens[index];
            }
            std::cout << "\n";
        }

        std::cout
            << "MFQ embedded tokenizer C++ test passed"
            << " vocab=" << probe.vocab_size
            << " tokens=" << probe.tokens.size()
            << " bos=" << probe.bos_token
            << " eos=" << probe.eos_token
            << " eot=" << probe.eot_token
            << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
