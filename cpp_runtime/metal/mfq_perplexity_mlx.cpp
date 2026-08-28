#include "mfq_container.h"
#include "mlx_deepseek_v4_causal_lm.h"
#include "mlx_qwen35_causal_lm.h"
#include "qwen35_model.h"

#include "../mfq_server.h"
#include "../json/nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mach-o/dyld.h>
#include <CommonCrypto/CommonDigest.h>

#include <mlx/memory.h>
#include <mlx/mlx.h>

namespace {

using Clock = std::chrono::steady_clock;
using mlx::core::Shape;
using mlx::core::array;
using json = nlohmann::json;

constexpr const char* kTokenizerAsset =
    "__mfq_asset__/tokenizer.gguf";

struct Arguments {
    std::filesystem::path mfq;
    std::filesystem::path input;
    std::filesystem::path kl_base;
    std::filesystem::path kl_manifest;
    std::filesystem::path logits_file;
    std::filesystem::path logits_manifest;
    std::filesystem::path tokenizer_gguf;
    std::string dataset;
    std::string model_label;
    int context = 512;
    int chunks = -1;
    int parallel = 1;
    int batch_size = 0;
    int ubatch_size = 0;
    int score_count = -1;
    std::optional<double> expert_cache_gb;
    bool context_explicit = false;
    bool parallel_explicit = false;
    bool batch_explicit = false;
    bool ubatch_explicit = false;
    bool help = false;
};

struct PerplexityStats {
    std::uint64_t count = 0;
    long double nll = 0.0;
    long double nll_squared = 0.0;

    void add(float value) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "model produced a non-finite negative log-likelihood");
        }
        const auto precise = static_cast<long double>(value);
        ++count;
        nll += precise;
        nll_squared += precise * precise;
    }

    double perplexity() const {
        if (count == 0) {
            throw std::runtime_error("cannot compute perplexity with no tokens");
        }
        return std::exp(
            static_cast<double>(nll / static_cast<long double>(count)));
    }

    double uncertainty() const {
        if (count < 2) {
            return 0.0;
        }
        const auto divisor = static_cast<long double>(count);
        const auto average = nll / divisor;
        const auto variance = nll_squared / divisor - average * average;
        if (variance <= 0.0) {
            return 0.0;
        }
        const auto error = std::sqrt(
            variance / static_cast<long double>(count - 1));
        return static_cast<double>(error) * perplexity();
    }
};

struct KlChunk {
    std::vector<std::int32_t> tokens;
    std::vector<float> target_log_probs;
    int target_start = 0;
    int score_count = 0;
    std::streamoff row_offset = 0;
};

struct KlReference {
    std::string format;
    int vocabulary = 0;
    int row_elements = 0;
    std::vector<KlChunk> chunks;
};

struct ReferenceContract {
    std::filesystem::path path;
    int n_ctx = 0;
    int n_batch = 0;
    int n_ubatch = 0;
    int n_seq = 0;
    int chunks = 0;
    int target_start = 0;
    int score_count = 0;
    std::uint64_t scored_tokens = 0;
    std::string dataset_id;
    std::string dataset_sha256;
    std::string model_label;
    std::string precision;
    std::string attention;
    std::string kv_cache_dtype;
};

struct KlStats {
    std::uint64_t count = 0;
    std::uint64_t same_top = 0;
    long double kld = 0.0;
    long double reverse_kld = 0.0;
    long double reference_ce = 0.0;
    long double mfq_ce = 0.0;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::runtime_error(
        message +
        "\nRun mfq-perplexity --help for supported options.");
}

long long parse_integer(
    const std::string& text,
    const char* option) {
    std::size_t consumed = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &consumed);
    } catch (const std::exception&) {
        usage_error(std::string(option) + " requires an integer");
    }
    if (consumed != text.size()) {
        usage_error(std::string(option) + " requires an integer");
    }
    return value;
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string value = argv[index];
        auto require_value = [&](const char* option) -> std::string {
            if (++index >= argc) {
                usage_error(std::string("missing value for ") + option);
            }
            return argv[index];
        };
        if (value == "--mfq" || value == "-m") {
            result.mfq = require_value(value.c_str());
        } else if (value == "--file" || value == "-f") {
            result.input = require_value(value.c_str());
        } else if (value == "--kl-base") {
            result.kl_base = require_value("--kl-base");
        } else if (value == "--kl-manifest") {
            result.kl_manifest = require_value("--kl-manifest");
        } else if (value == "--logits-file") {
            result.logits_file = require_value("--logits-file");
        } else if (value == "--logits-manifest") {
            result.logits_manifest = require_value("--logits-manifest");
        } else if (value == "--dataset") {
            result.dataset = require_value("--dataset");
            if (result.dataset.empty()) {
                usage_error("--dataset cannot be empty");
            }
        } else if (value == "--model-label") {
            result.model_label = require_value("--model-label");
            if (result.model_label.empty()) {
                usage_error("--model-label cannot be empty");
            }
        } else if (value == "--tokenizer-gguf") {
            result.tokenizer_gguf = require_value("--tokenizer-gguf");
        } else if (value == "--ctx-size" || value == "-c") {
            const auto parsed = parse_integer(
                require_value(value.c_str()), value.c_str());
            if (parsed < 2 ||
                parsed > std::numeric_limits<int>::max()) {
                usage_error("--ctx-size must be in [2, INT_MAX]");
            }
            result.context = static_cast<int>(parsed);
            result.context_explicit = true;
        } else if (value == "--chunks") {
            const auto parsed = parse_integer(
                require_value("--chunks"), "--chunks");
            if (parsed == 0 || parsed < -1 ||
                parsed > std::numeric_limits<int>::max()) {
                usage_error("--chunks must be -1 or a positive integer");
            }
            result.chunks = static_cast<int>(parsed);
        } else if (value == "--parallel") {
            const auto parsed = parse_integer(
                require_value("--parallel"), "--parallel");
            if (parsed < 1 ||
                parsed > std::numeric_limits<int>::max()) {
                usage_error("--parallel must be positive");
            }
            result.parallel = static_cast<int>(parsed);
            result.parallel_explicit = true;
        } else if (value == "--batch-size" || value == "-b") {
            const auto parsed = parse_integer(
                require_value(value.c_str()), value.c_str());
            if (parsed < 1 ||
                parsed > std::numeric_limits<int>::max()) {
                usage_error("--batch-size must be positive");
            }
            result.batch_size = static_cast<int>(parsed);
            result.batch_explicit = true;
        } else if (value == "--ubatch-size" || value == "-ub") {
            const auto parsed = parse_integer(
                require_value(value.c_str()), value.c_str());
            if (parsed < 1 ||
                parsed > std::numeric_limits<int>::max()) {
                usage_error("--ubatch-size must be positive");
            }
            result.ubatch_size = static_cast<int>(parsed);
            result.ubatch_explicit = true;
        } else if (value == "--kl-score-count") {
            const auto parsed = parse_integer(
                require_value("--kl-score-count"),
                "--kl-score-count");
            if (parsed < 1 ||
                parsed > std::numeric_limits<int>::max()) {
                usage_error("--kl-score-count must be positive");
            }
            result.score_count = static_cast<int>(parsed);
        } else if (value == "--moe-gpu-cache-gb") {
            const auto text = require_value(value.c_str());
            std::size_t consumed = 0;
            double parsed = 0.0;
            try {
                parsed = std::stod(text, &consumed);
            } catch (const std::exception&) {
                usage_error(
                    value + " requires a finite non-negative number");
            }
            if (consumed != text.size() ||
                !std::isfinite(parsed) || parsed < 0.0) {
                usage_error(
                    value + " requires a finite non-negative number");
            }
            result.expert_cache_gb = parsed;
        } else if (value == "--help" || value == "-h") {
            result.help = true;
        } else {
            usage_error("unknown option: " + value);
        }
    }
    if (result.batch_size != 0 && result.parallel_explicit) {
        usage_error("--batch-size and --parallel cannot be used together");
    }
    return result;
}

void print_help() {
    std::cout
        << "MFQ native C++/MLX perplexity evaluator\n\n"
        << "Usage:\n"
        << "  mfq-perplexity --mfq MODEL.mfq --file wiki.test.raw "
           "[--ctx-size 512]\n\n"
        << "  mfq-perplexity --mfq MODEL.mfq --kl-base reference.logits\n\n"
        << "Options:\n"
        << "  -m, --mfq PATH          MFQ model or any split-model shard\n"
        << "  -f, --file PATH         raw evaluation text\n"
        << "      --kl-base PATH      llama-perplexity reference logits; "
           "enables integrated KLD\n"
        << "      --kl-manifest PATH  reference contract; default LOGITS.manifest.json\n"
        << "      --logits-file PATH  export trace_v3 log probabilities\n"
        << "      --logits-manifest P output contract; default LOGITS.manifest.json\n"
        << "      --dataset NAME      export identity; validates KLD if supplied\n"
        << "      --model-label NAME  public reference-model label\n"
        << "  -c, --ctx-size N        logical context window (default 512)\n"
        << "      --chunks N          evaluate at most N windows (default all)\n"
        << "      --parallel N        windows per Metal forward pass (default 1)\n"
        << "  -b, --batch-size N      llama.cpp-compatible total token batch; "
           "N/ctx controls parallelism\n"
        << " -ub, --ubatch-size N     physical token batch (default n_batch)\n"
        << "      --kl-score-count N  score only the first N stored rows/chunk\n"
        << "      --tokenizer-gguf P  external tokenizer GGUF; embedded is default\n"
        << "      --moe-gpu-cache-gb N bounded disk-backed NINTM expert cache\n"
        << "                           default: full unified-memory residency\n\n"
        << "The evaluator follows llama.cpp's non-strided WikiText-2 protocol: "
           "each window is independent and only its second half is scored.\n";
}

int resolved_parallel(const Arguments& arguments, int context) {
    int result = arguments.parallel;
    if (arguments.batch_size != 0) {
        if (arguments.batch_size < context ||
            arguments.batch_size % context != 0) {
            usage_error(
                "--batch-size must be a multiple of the logical context size "
                "and at least one context window");
        }
        result = arguments.batch_size / context;
    }
    if (result > std::numeric_limits<int>::max() / context) {
        usage_error("parallel token batch exceeds INT_MAX");
    }
    return result;
}

int resolved_ubatch(const Arguments& arguments, int batch_size) {
    const int result = arguments.ubatch_size == 0
        ? batch_size
        : arguments.ubatch_size;
    if (result < 1 || result > batch_size) {
        usage_error("--ubatch-size must be in [1, n_batch]");
    }
    return result;
}

std::filesystem::path executable_path() {
    std::uint32_t size = 0;
    (void)_NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size + 1, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        throw std::runtime_error("failed to resolve executable path");
    }
    std::error_code error;
    const auto resolved =
        std::filesystem::weakly_canonical(buffer.data(), error);
    return error ? std::filesystem::path(buffer.data()) : resolved;
}

void configure_mlx_metal() {
    std::vector<std::filesystem::path> candidates;
    if (const char* configured = std::getenv("MFQ_MLX_METALLIB")) {
        if (*configured != '\0') {
            candidates.emplace_back(configured);
        }
    }
    candidates.push_back(
        executable_path().parent_path() / "lib" / "mlx.metallib");
#ifdef MFQ_MLX_METALLIB_DEFAULT
    candidates.emplace_back(MFQ_MLX_METALLIB_DEFAULT);
#endif
    for (const auto& candidate : candidates) {
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error) && !error) {
            mlx::core::metal::set_metallib_path(candidate.string());
            mlx::core::set_default_device(mlx::core::Device::gpu);
            return;
        }
    }
    throw std::runtime_error(
        "cannot locate mlx.metallib; set MFQ_MLX_METALLIB or keep "
        "lib/mlx.metallib beside mfq-perplexity");
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "cannot open evaluation text: " + path.string());
    }
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < 0 ||
        static_cast<std::uintmax_t>(end) >
            static_cast<std::uintmax_t>(
                std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            "cannot determine evaluation text size: " + path.string());
    }
    std::string result(static_cast<std::size_t>(end), '\0');
    stream.seekg(0, std::ios::beg);
    if (!result.empty()) {
        stream.read(result.data(), static_cast<std::streamsize>(result.size()));
        if (!stream) {
            throw std::runtime_error(
                "cannot read evaluation text: " + path.string());
        }
    }
    return result;
}

std::string hex_digest(const unsigned char* bytes, std::size_t size) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(size * 2, '0');
    for (std::size_t index = 0; index < size; ++index) {
        result[index * 2] = digits[bytes[index] >> 4];
        result[index * 2 + 1] = digits[bytes[index] & 0x0f];
    }
    return result;
}

std::string sha256_bytes(const void* data, std::size_t size) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    const auto* cursor = static_cast<const unsigned char*>(data);
    while (size != 0) {
        const auto count = static_cast<CC_LONG>(std::min<std::size_t>(
            size,
            std::numeric_limits<CC_LONG>::max()));
        CC_SHA256_Update(&context, cursor, count);
        cursor += count;
        size -= count;
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    return hex_digest(digest, sizeof(digest));
}

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot hash file: " + path.string());
    }
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    std::vector<char> buffer(8 << 20);
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = stream.gcount();
        if (count > 0) {
            CC_SHA256_Update(
                &context,
                buffer.data(),
                static_cast<CC_LONG>(count));
        }
    }
    if (!stream.eof()) {
        throw std::runtime_error("failed while hashing file: " + path.string());
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    return hex_digest(digest, sizeof(digest));
}

std::uint64_t regular_file_size(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size > std::numeric_limits<std::uint64_t>::max()) {
        throw std::runtime_error("cannot read file size: " + path.string());
    }
    return static_cast<std::uint64_t>(size);
}

std::filesystem::path default_manifest_path(
    const std::filesystem::path& logits) {
    return logits.parent_path() /
        (logits.filename().string() + ".manifest.json");
}

json gpu_manifest() {
    json result = json::object();
    const auto& values = mlx::core::gpu::device_info(0);
    for (const auto& [key, value] : values) {
        std::visit(
            [&](const auto& item) { result[key] = item; },
            value);
    }
    return result;
}

std::vector<std::int32_t> evaluated_tokens(
    const std::vector<std::int64_t>& tokens,
    const MfqTokenizerProbe& tokenizer,
    int context,
    int chunks,
    int vocabulary) {
    std::vector<std::int32_t> result;
    result.reserve(
        static_cast<std::size_t>(context) *
        static_cast<std::size_t>(chunks));
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const auto offset = static_cast<std::size_t>(chunk) * context;
        for (int position = 0; position < context; ++position) {
            auto token = tokens[offset + static_cast<std::size_t>(position)];
            if (position == 0 && tokenizer.add_bos) {
                token = tokenizer.bos_token;
            }
            if (token < 0 || token >= vocabulary) {
                throw std::runtime_error(
                    "cannot serialize an out-of-range evaluation token");
            }
            result.push_back(static_cast<std::int32_t>(token));
        }
    }
    return result;
}

template <typename T>
T read_scalar(std::ifstream& stream, const char* description) {
    T value{};
    stream.read(
        reinterpret_cast<char*>(&value),
        static_cast<std::streamsize>(sizeof(value)));
    if (!stream) {
        throw std::runtime_error(
            std::string("truncated KLD reference ") + description);
    }
    return value;
}

template <typename T>
void read_vector(
    std::ifstream& stream,
    std::vector<T>& values,
    const char* description) {
    if (values.size() >
        static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max()) /
            sizeof(T)) {
        throw std::runtime_error(
            std::string("KLD reference ") + description + " is too large");
    }
    stream.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!stream) {
        throw std::runtime_error(
            std::string("truncated KLD reference ") + description);
    }
}

KlReference load_kl_reference(
    const std::filesystem::path& path,
    int max_chunks) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "cannot open KLD reference: " + path.string());
    }
    char magic[8]{};
    stream.read(magic, sizeof(magic));
    if (!stream) {
        throw std::runtime_error("invalid KLD reference header");
    }

    KlReference result;
    result.format.assign(magic, sizeof(magic));
    if (result.format == "_logits_") {
        const auto context = read_scalar<std::uint32_t>(
            stream, "legacy context size");
        const auto vocabulary = read_scalar<std::int32_t>(
            stream, "legacy vocabulary size");
        const auto chunks = read_scalar<std::int32_t>(
            stream, "legacy chunk count");
        if (context < 2 ||
            context > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()) ||
            vocabulary <= 0 || chunks <= 0 || chunks > (1 << 20)) {
            throw std::runtime_error("invalid legacy KLD reference header");
        }
        if (static_cast<std::uint64_t>(context) *
                static_cast<std::uint64_t>(chunks) >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("legacy KLD token table is too large");
        }
        std::vector<std::int32_t> tokens(
            static_cast<std::size_t>(context) *
            static_cast<std::size_t>(chunks));
        read_vector(stream, tokens, "legacy token table");
        const int first = static_cast<int>(context) / 2;
        const int score_count =
            static_cast<int>(context) - first - 1;
        result.vocabulary = vocabulary;
        result.chunks.resize(static_cast<std::size_t>(chunks));
        for (int index = 0; index < chunks; ++index) {
            auto& chunk = result.chunks[static_cast<std::size_t>(index)];
            const auto begin = tokens.begin() +
                static_cast<std::size_t>(index) * context;
            chunk.tokens.assign(begin, begin + context);
            chunk.target_start = first + 1;
            chunk.score_count = score_count;
        }
    } else if (
        result.format == "_logit2_" ||
        result.format == "_logit3_") {
        const bool exact_targets = result.format == "_logit3_";
        const auto vocabulary = read_scalar<std::uint32_t>(
            stream, "trace vocabulary size");
        const auto chunks = read_scalar<std::uint32_t>(
            stream, "trace chunk count");
        if (vocabulary == 0 ||
            vocabulary > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()) ||
            chunks == 0 || chunks > (1u << 20)) {
            throw std::runtime_error("invalid trace KLD reference header");
        }
        result.vocabulary = static_cast<int>(vocabulary);
        result.chunks.resize(chunks);
        std::vector<std::uint32_t> token_counts(chunks);
        for (std::uint32_t index = 0; index < chunks; ++index) {
            const auto token_count = read_scalar<std::uint32_t>(
                stream, "trace token count");
            const auto target_start = read_scalar<std::uint32_t>(
                stream, "trace target start");
            const auto score_count = read_scalar<std::uint32_t>(
                stream, "trace score count");
            if (token_count < 2 ||
                token_count > static_cast<std::uint32_t>(
                    std::numeric_limits<int>::max()) ||
                target_start < 1 || target_start >= token_count ||
                score_count < 1 ||
                score_count > token_count - target_start) {
                throw std::runtime_error(
                    "invalid trace KLD chunk descriptor");
            }
            token_counts[index] = token_count;
            auto& chunk = result.chunks[index];
            chunk.target_start = static_cast<int>(target_start);
            chunk.score_count = static_cast<int>(score_count);
        }
        for (std::uint32_t index = 0; index < chunks; ++index) {
            auto& chunk = result.chunks[index];
            chunk.tokens.resize(token_counts[index]);
            read_vector(stream, chunk.tokens, "trace token table");
        }
        if (exact_targets) {
            for (auto& chunk : result.chunks) {
                chunk.target_log_probs.resize(
                    static_cast<std::size_t>(chunk.score_count));
                read_vector(
                    stream,
                    chunk.target_log_probs,
                    "trace target log probabilities");
                for (const auto value : chunk.target_log_probs) {
                    if (!std::isfinite(value) || value > 0.0f) {
                        throw std::runtime_error(
                            "invalid trace KLD target log probability");
                    }
                }
            }
        }
    } else {
        throw std::runtime_error(
            "unsupported KLD reference magic; expected "
            "_logits_, _logit2_, or _logit3_");
    }

    const auto row_elements =
        2 * ((static_cast<std::int64_t>(result.vocabulary) + 1) / 2) + 4;
    if (row_elements > std::numeric_limits<int>::max()) {
        throw std::runtime_error("KLD vocabulary row is too large");
    }
    result.row_elements = static_cast<int>(row_elements);
    auto row_offset = stream.tellg();
    if (row_offset < 0) {
        throw std::runtime_error("invalid KLD row data offset");
    }
    for (auto& chunk : result.chunks) {
        chunk.row_offset = row_offset;
        const auto bytes = static_cast<std::uint64_t>(chunk.score_count) *
            static_cast<std::uint64_t>(result.row_elements) *
            sizeof(std::uint16_t);
        if (bytes > static_cast<std::uint64_t>(
                        std::numeric_limits<std::streamoff>::max()) ||
            row_offset > std::numeric_limits<std::streamoff>::max() -
                static_cast<std::streamoff>(bytes)) {
            throw std::runtime_error("KLD row table offset overflow");
        }
        row_offset += static_cast<std::streamoff>(bytes);
    }
    stream.seekg(0, std::ios::end);
    if (!stream || stream.tellg() < row_offset) {
        throw std::runtime_error("truncated KLD reference rows");
    }
    if (max_chunks >= 0 &&
        result.chunks.size() > static_cast<std::size_t>(max_chunks)) {
        result.chunks.resize(static_cast<std::size_t>(max_chunks));
    }
    return result;
}

ReferenceContract load_reference_contract(
    const Arguments& arguments,
    const KlReference& reference) {
    ReferenceContract contract;
    contract.path = arguments.kl_manifest.empty()
        ? default_manifest_path(arguments.kl_base)
        : arguments.kl_manifest;
    if (!std::filesystem::is_regular_file(contract.path)) {
        throw std::runtime_error(
            "KLD requires the logits generation contract: " +
            contract.path.string());
    }
    json document;
    try {
        document = json::parse(read_text(contract.path));
        if (document.at("format").get<std::string>() !=
            "mfq.perplexity-logits-manifest.v1") {
            throw std::runtime_error(
                "unsupported logits manifest format");
        }
        const auto& reference_json = document.at("reference");
        if (reference_json.at("magic").get<std::string>() !=
            reference.format) {
            throw std::runtime_error(
                "logits manifest/reference magic mismatch");
        }
        const auto& file = reference_json.at("file");
        const auto declared_bytes = file.at("bytes").get<std::uint64_t>();
        if (declared_bytes != regular_file_size(arguments.kl_base)) {
            throw std::runtime_error(
                "logits manifest/reference byte size mismatch");
        }
        std::cerr
            << "mfq-perplexity: verifying reference SHA-256...\n";
        if (file.at("sha256").get<std::string>() !=
            sha256_file(arguments.kl_base)) {
            throw std::runtime_error(
                "logits manifest/reference SHA-256 mismatch");
        }

        const auto& evaluation = document.at("evaluation");
        if (evaluation.at("protocol").get<std::string>() !=
            "llama.cpp-non-strided-second-half-v1") {
            throw std::runtime_error(
                "unsupported logits evaluation protocol");
        }
        contract.n_ctx = evaluation.at("n_ctx").get<int>();
        contract.n_batch = evaluation.at("n_batch").get<int>();
        contract.n_ubatch = evaluation.at("n_ubatch").get<int>();
        contract.n_seq = evaluation.at("n_seq").get<int>();
        contract.chunks = evaluation.at("chunks").get<int>();
        contract.target_start = evaluation.at("target_start").get<int>();
        contract.score_count =
            evaluation.at("score_count_per_chunk").get<int>();
        contract.scored_tokens =
            evaluation.at("scored_tokens").get<std::uint64_t>();
        contract.attention = evaluation.at("attention").get<std::string>();
        contract.kv_cache_dtype =
            evaluation.at("kv_cache_dtype").get<std::string>();

        const auto& dataset = document.at("dataset");
        contract.dataset_id = dataset.at("id").get<std::string>();
        contract.dataset_sha256 = dataset.at("sha256").get<std::string>();
        if (contract.dataset_id.empty() ||
            contract.dataset_sha256.size() != 64) {
            throw std::runtime_error(
                "logits manifest has incomplete dataset identity");
        }
        const auto& model = document.at("model");
        contract.model_label = model.at("label").get<std::string>();
        contract.precision = model.at("precision")
            .at("classification").get<std::string>();
        if (contract.model_label.empty() || contract.precision.empty()) {
            throw std::runtime_error(
                "logits manifest has incomplete reference-model identity");
        }
    } catch (const json::exception& error) {
        throw std::runtime_error(
            std::string("invalid logits manifest: ") + error.what());
    }

    if (contract.n_ctx < 2 || contract.n_batch < contract.n_ctx ||
        contract.n_batch % contract.n_ctx != 0 ||
        contract.n_seq != contract.n_batch / contract.n_ctx ||
        contract.n_ubatch < contract.n_seq ||
        contract.n_ubatch > contract.n_batch ||
        contract.chunks < 1 || contract.target_start < 1 ||
        contract.score_count < 1 ||
        contract.attention != "native_model_default" ||
        contract.kv_cache_dtype != "native_model_default" ||
        contract.scored_tokens !=
            static_cast<std::uint64_t>(contract.chunks) *
                static_cast<std::uint64_t>(contract.score_count)) {
        throw std::runtime_error(
            "logits manifest has invalid evaluation geometry");
    }
    if (reference.chunks.size() !=
            static_cast<std::size_t>(contract.chunks) ||
        reference.chunks.empty()) {
        throw std::runtime_error(
            "logits manifest/reference chunk count mismatch");
    }
    std::vector<std::int32_t> tokens;
    for (const auto& chunk : reference.chunks) {
        if (chunk.tokens.size() !=
                static_cast<std::size_t>(contract.n_ctx) ||
            chunk.target_start != contract.target_start ||
            chunk.score_count != contract.score_count) {
            throw std::runtime_error(
                "logits manifest/reference chunk geometry mismatch");
        }
        tokens.insert(tokens.end(), chunk.tokens.begin(), chunk.tokens.end());
    }
    const auto token_sha256 = sha256_bytes(
        tokens.data(),
        tokens.size() * sizeof(std::int32_t));
    try {
        if (document.at("evaluation")
                .at("input_token_ids_sha256")
                .get<std::string>() != token_sha256) {
            throw std::runtime_error(
                "logits manifest/reference token SHA-256 mismatch");
        }
        if (document.at("tokenizer").at("vocab_size").get<int>() !=
            reference.vocabulary) {
            throw std::runtime_error(
                "logits manifest/reference vocabulary mismatch");
        }
    } catch (const json::exception& error) {
        throw std::runtime_error(
            std::string("invalid logits manifest contract: ") + error.what());
    }

    if (arguments.context_explicit &&
        arguments.context != contract.n_ctx) {
        throw std::runtime_error(
            "--ctx-size differs from the logits generation contract");
    }
    if (arguments.batch_explicit &&
        arguments.batch_size != contract.n_batch) {
        throw std::runtime_error(
            "--batch-size differs from the logits generation contract");
    }
    if (arguments.parallel_explicit &&
        arguments.parallel != contract.n_seq) {
        throw std::runtime_error(
            "--parallel differs from the logits generation contract");
    }
    if (arguments.ubatch_explicit &&
        arguments.ubatch_size != contract.n_ubatch) {
        throw std::runtime_error(
            "--ubatch-size differs from the logits generation contract");
    }
    if (arguments.chunks >= 0 && arguments.chunks != contract.chunks) {
        throw std::runtime_error(
            "--chunks differs from the logits generation contract");
    }
    if (arguments.score_count > 0 &&
        arguments.score_count != contract.score_count) {
        throw std::runtime_error(
            "--kl-score-count differs from the logits generation contract");
    }
    if (!arguments.dataset.empty() &&
        arguments.dataset != contract.dataset_id) {
        throw std::runtime_error(
            "--dataset differs from the logits generation contract");
    }
    return contract;
}

MfqTokenizerProbe tokenize(
    const Arguments& arguments,
    const mfq::metal::MfqContainer& model,
    const std::string& text) {
    // llama-perplexity calls common_tokenize(..., add_special=true).
    constexpr bool add_special = true;
    constexpr bool parse_special = true;
    if (!arguments.tokenizer_gguf.empty()) {
        return probe_mfq_tokenizer(
            arguments.tokenizer_gguf.string(),
            text,
            add_special,
            parse_special);
    }
    if (!model.contains(kTokenizerAsset)) {
        throw std::runtime_error(
            "MFQ has no embedded tokenizer GGUF; pass --tokenizer-gguf PATH");
    }
    return probe_mfq_tokenizer(
        model.read(kTokenizerAsset),
        text,
        add_special,
        parse_special);
}

MfqTokenizerProbe tokenizer_policy(
    const Arguments& arguments,
    const mfq::metal::MfqContainer& model) {
    constexpr bool add_special = false;
    constexpr bool parse_special = true;
    if (!arguments.tokenizer_gguf.empty()) {
        return probe_mfq_tokenizer(
            arguments.tokenizer_gguf.string(),
            "",
            add_special,
            parse_special);
    }
    if (!model.contains(kTokenizerAsset)) {
        throw std::runtime_error(
            "legacy KLD references require the embedded tokenizer policy; "
            "pass --tokenizer-gguf PATH");
    }
    return probe_mfq_tokenizer(
        model.read(kTokenizerAsset),
        "",
        add_special,
        parse_special);
}

std::optional<std::size_t> expert_cache_bytes(
    const std::optional<double>& gib) {
    if (!gib.has_value()) {
        return std::nullopt;
    }
    constexpr long double bytes_per_gib =
        static_cast<long double>(std::uint64_t{1} << 30);
    const auto bytes =
        static_cast<long double>(*gib) * bytes_per_gib;
    if (bytes > static_cast<long double>(
                    std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            "DeepSeek-V4 expert cache exceeds addressable memory");
    }
    return static_cast<std::size_t>(bytes);
}

class TraceV3Writer {
public:
    static constexpr float logit_range = 16.0f;
    static constexpr float code_max = 65535.0f;

    TraceV3Writer(
        std::filesystem::path output,
        const std::vector<std::int32_t>& tokens,
        int vocabulary,
        int context,
        int chunks,
        int target_start,
        int score_count)
        : output_(std::move(output)),
          partial_(output_.string() + ".partial"),
          vocabulary_(vocabulary),
          context_(context),
          chunks_(chunks),
          target_start_(target_start),
          score_count_(score_count),
          row_elements_(
              2 * ((static_cast<std::int64_t>(vocabulary) + 1) / 2) + 4) {
        if (output_.empty() || vocabulary_ <= 0 || context_ < 2 ||
            chunks_ <= 0 || target_start_ < 1 || score_count_ < 1 ||
            tokens.size() !=
                static_cast<std::size_t>(context_) * chunks_ ||
            row_elements_ > std::numeric_limits<int>::max()) {
            throw std::runtime_error("invalid trace_v3 output geometry");
        }
        std::error_code error;
        const auto parent = output_.parent_path().empty()
            ? std::filesystem::current_path()
            : output_.parent_path();
        if (!std::filesystem::is_directory(parent, error) || error) {
            throw std::runtime_error(
                "logits output directory does not exist: " + parent.string());
        }
        if (std::filesystem::exists(output_) ||
            std::filesystem::exists(partial_)) {
            throw std::runtime_error(
                "refusing to overwrite logits output or partial file: " +
                output_.string());
        }

        const auto header_bytes =
            std::uint64_t{8} + 2 * sizeof(std::uint32_t) +
            static_cast<std::uint64_t>(chunks_) * 3 *
                sizeof(std::uint32_t);
        const auto token_bytes =
            static_cast<std::uint64_t>(tokens.size()) *
            sizeof(std::int32_t);
        auto offset = header_bytes + token_bytes;
        target_offsets_.reserve(chunks_);
        for (int chunk = 0; chunk < chunks_; ++chunk) {
            target_offsets_.push_back(offset);
            offset += static_cast<std::uint64_t>(score_count_) * sizeof(float);
        }
        row_offsets_.reserve(chunks_);
        const auto row_bytes =
            static_cast<std::uint64_t>(row_elements_) *
            sizeof(std::uint16_t);
        for (int chunk = 0; chunk < chunks_; ++chunk) {
            row_offsets_.push_back(offset);
            if (static_cast<std::uint64_t>(score_count_) >
                (std::numeric_limits<std::uint64_t>::max() - offset) /
                    row_bytes) {
                throw std::runtime_error("trace_v3 output size overflow");
            }
            offset += static_cast<std::uint64_t>(score_count_) * row_bytes;
        }
        total_bytes_ = offset;
        if (total_bytes_ > static_cast<std::uint64_t>(
                               std::numeric_limits<std::streamoff>::max())) {
            throw std::runtime_error("trace_v3 output exceeds stream capacity");
        }

        {
            std::ofstream header(partial_, std::ios::binary | std::ios::trunc);
            if (!header) {
                throw std::runtime_error(
                    "cannot create logits partial: " + partial_.string());
            }
            header.write("_logit3_", 8);
            const auto vocab_u32 = static_cast<std::uint32_t>(vocabulary_);
            const auto chunks_u32 = static_cast<std::uint32_t>(chunks_);
            header.write(
                reinterpret_cast<const char*>(&vocab_u32),
                sizeof(vocab_u32));
            header.write(
                reinterpret_cast<const char*>(&chunks_u32),
                sizeof(chunks_u32));
            for (int chunk = 0; chunk < chunks_; ++chunk) {
                const auto context_u32 = static_cast<std::uint32_t>(context_);
                const auto target_u32 =
                    static_cast<std::uint32_t>(target_start_);
                const auto score_u32 =
                    static_cast<std::uint32_t>(score_count_);
                header.write(
                    reinterpret_cast<const char*>(&context_u32),
                    sizeof(context_u32));
                header.write(
                    reinterpret_cast<const char*>(&target_u32),
                    sizeof(target_u32));
                header.write(
                    reinterpret_cast<const char*>(&score_u32),
                    sizeof(score_u32));
            }
            header.write(
                reinterpret_cast<const char*>(tokens.data()),
                static_cast<std::streamsize>(
                    tokens.size() * sizeof(std::int32_t)));
            if (!header) {
                throw std::runtime_error("cannot write trace_v3 header");
            }
        }
        std::filesystem::resize_file(partial_, total_bytes_, error);
        if (error) {
            throw std::runtime_error(
                "cannot allocate trace_v3 output: " + error.message());
        }
        stream_.open(
            partial_,
            std::ios::binary | std::ios::in | std::ios::out);
        if (!stream_) {
            throw std::runtime_error("cannot reopen trace_v3 output");
        }
    }

    ~TraceV3Writer() {
        if (stream_.is_open()) {
            stream_.close();
        }
        if (!committed_) {
            std::error_code ignored;
            std::filesystem::remove(partial_, ignored);
        }
    }

    TraceV3Writer(const TraceV3Writer&) = delete;
    TraceV3Writer& operator=(const TraceV3Writer&) = delete;

    void write_chunk(
        int chunk,
        const array& logits,
        const float* negative_log_likelihood) {
        if (chunk < 0 || chunk >= chunks_ ||
            logits.ndim() != 2 || logits.shape(0) != score_count_ ||
            logits.shape(1) != vocabulary_) {
            throw std::runtime_error("invalid trace_v3 chunk logits");
        }
        std::vector<float> target_log_probs(
            static_cast<std::size_t>(score_count_));
        for (int row = 0; row < score_count_; ++row) {
            const float value = -negative_log_likelihood[row];
            if (!std::isfinite(value) || value > 1e-5f) {
                throw std::runtime_error(
                    "invalid exact target log probability");
            }
            target_log_probs[static_cast<std::size_t>(row)] = value;
        }
        write_at(
            target_offsets_[static_cast<std::size_t>(chunk)],
            target_log_probs.data(),
            target_log_probs.size() * sizeof(float));

        constexpr int rows_per_batch = 8;
        const float scale_value = logit_range / code_max;
        for (int start = 0; start < score_count_; start += rows_per_batch) {
            const int count = std::min(
                rows_per_batch,
                score_count_ - start);
            auto values = mlx::core::astype(
                mlx::core::slice(
                    logits,
                    Shape{start, 0},
                    Shape{start + count, vocabulary_}),
                mlx::core::float32);
            auto logp = values - mlx::core::logsumexp(
                values, -1, true);
            auto minimum = mlx::core::max(logp, -1, true) - logit_range;
            auto codes = mlx::core::round(
                (logp - minimum) * (code_max / logit_range));
            codes = mlx::core::maximum(
                mlx::core::minimum(codes, array(code_max)),
                array(0.0f));
            codes = mlx::core::astype(codes, mlx::core::uint16);
            mlx::core::eval(minimum, codes);
            const auto* minimum_values = minimum.data<float>();
            const auto* code_values = codes.data<std::uint16_t>();
            std::vector<std::uint16_t> rows(
                static_cast<std::size_t>(count) *
                static_cast<std::size_t>(row_elements_),
                0);
            for (int row = 0; row < count; ++row) {
                auto* destination = rows.data() +
                    static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(row_elements_);
                std::memcpy(destination, &scale_value, sizeof(float));
                std::memcpy(
                    destination + 2,
                    &minimum_values[row],
                    sizeof(float));
                std::memcpy(
                    destination + 4,
                    code_values +
                        static_cast<std::size_t>(row) * vocabulary_,
                    static_cast<std::size_t>(vocabulary_) *
                        sizeof(std::uint16_t));
            }
            const auto row_bytes =
                static_cast<std::uint64_t>(row_elements_) *
                sizeof(std::uint16_t);
            write_at(
                row_offsets_[static_cast<std::size_t>(chunk)] +
                    static_cast<std::uint64_t>(start) * row_bytes,
                rows.data(),
                rows.size() * sizeof(std::uint16_t));
        }
    }

    void commit() {
        stream_.flush();
        if (!stream_) {
            throw std::runtime_error("cannot flush trace_v3 output");
        }
        stream_.close();
        std::error_code error;
        std::filesystem::rename(partial_, output_, error);
        if (error) {
            throw std::runtime_error(
                "cannot commit trace_v3 output: " + error.message());
        }
        committed_ = true;
    }

    std::uint64_t total_bytes() const noexcept {
        return total_bytes_;
    }

private:
    void write_at(
        std::uint64_t offset,
        const void* data,
        std::size_t bytes) {
        stream_.seekp(static_cast<std::streamoff>(offset));
        stream_.write(
            static_cast<const char*>(data),
            static_cast<std::streamsize>(bytes));
        if (!stream_) {
            throw std::runtime_error("cannot write trace_v3 output rows");
        }
    }

    std::filesystem::path output_;
    std::filesystem::path partial_;
    int vocabulary_ = 0;
    int context_ = 0;
    int chunks_ = 0;
    int target_start_ = 0;
    int score_count_ = 0;
    std::int64_t row_elements_ = 0;
    std::uint64_t total_bytes_ = 0;
    std::vector<std::uint64_t> target_offsets_;
    std::vector<std::uint64_t> row_offsets_;
    std::fstream stream_;
    bool committed_ = false;
};

std::string inferred_precision(
    const std::map<std::string, std::uint64_t>& dtypes) {
    if (dtypes.size() == 1) {
        return dtypes.begin()->first;
    }
    bool quantized = false;
    for (const auto& [dtype, count] : dtypes) {
        (void)count;
        if (dtype.rfind("NINT", 0) == 0 ||
            dtype.rfind("NVQ", 0) == 0 ||
            dtype.rfind("NPQ", 0) == 0 ||
            dtype.rfind("NEPQ", 0) == 0 ||
            dtype.rfind("TPQ", 0) == 0 ||
            dtype.rfind("TPQ", 0) == 0) {
            quantized = true;
        }
    }
    return quantized ? "mixed-quantized" : "mixed-full-precision";
}

void write_logits_manifest(
    const Arguments& arguments,
    const mfq::metal::MfqContainer& model,
    const MfqTokenizerProbe& tokenizer,
    const std::vector<std::int32_t>& tokens,
    int vocabulary,
    int context,
    int chunks,
    int batch_size,
    int ubatch_size,
    int target_start,
    int score_count,
    const PerplexityStats& stats) {
    const auto manifest_path = arguments.logits_manifest.empty()
        ? default_manifest_path(arguments.logits_file)
        : arguments.logits_manifest;
    const auto partial = std::filesystem::path(
        manifest_path.string() + ".partial");
    if (std::filesystem::exists(manifest_path) ||
        std::filesystem::exists(partial)) {
        throw std::runtime_error(
            "refusing to overwrite logits manifest or partial: " +
            manifest_path.string());
    }
    if (arguments.dataset.empty()) {
        throw std::runtime_error(
            "--dataset NAME is required with --logits-file");
    }
    std::map<std::string, std::uint64_t> dtype_counts;
    for (const auto& [name, record] : model.records()) {
        if (name.rfind("__mfq_asset__/", 0) != 0 &&
            record.dtype != "BLOB") {
            ++dtype_counts[record.dtype];
        }
    }
    json dtype_json = json::object();
    for (const auto& [dtype, count] : dtype_counts) {
        dtype_json[dtype] = count;
    }

    std::cerr << "mfq-perplexity: hashing model shards for manifest...\n";
    json model_files = json::array();
    for (const auto& path : model.source_paths()) {
        model_files.push_back({
            {"name", path.filename().string()},
            {"bytes", regular_file_size(path)},
            {"sha256", sha256_file(path)},
        });
    }
    json tokenizer_source;
    if (arguments.tokenizer_gguf.empty()) {
        const auto blob = model.read(kTokenizerAsset);
        tokenizer_source = {
            {"kind", "embedded_mfq_asset"},
            {"asset", kTokenizerAsset},
            {"bytes", blob.size()},
            {"sha256", sha256_bytes(blob.data(), blob.size())},
        };
    } else {
        tokenizer_source = {
            {"kind", "external_gguf"},
            {"file_name", arguments.tokenizer_gguf.filename().string()},
            {"bytes", regular_file_size(arguments.tokenizer_gguf)},
            {"sha256", sha256_file(arguments.tokenizer_gguf)},
        };
    }

    const auto logits_bytes = regular_file_size(arguments.logits_file);
    std::cerr << "mfq-perplexity: hashing logits output for manifest...\n";
    const auto logits_sha256 = sha256_file(arguments.logits_file);
    const auto dataset_sha256 = sha256_file(arguments.input);
    const auto token_sha256 = sha256_bytes(
        tokens.data(),
        tokens.size() * sizeof(std::int32_t));
    const auto executable = executable_path();
    const int n_seq = batch_size / context;
    json manifest = {
        {"format", "mfq.perplexity-logits-manifest.v1"},
        {"producer", {
            {"tool", "mfq-perplexity"},
            {"runtime", "native-cpp-mlx-metal"},
            {"mlx_version", mlx::core::version()},
            {"executable", {
                {"name", executable.filename().string()},
                {"bytes", regular_file_size(executable)},
                {"sha256", sha256_file(executable)},
            }},
            {"device", gpu_manifest()},
        }},
        {"reference", {
            {"magic", "_logit3_"},
            {"file", {
                {"name", arguments.logits_file.filename().string()},
                {"bytes", logits_bytes},
                {"sha256", logits_sha256},
            }},
            {"distribution_encoding", {
                {"type", "linear_uint16_log_probability"},
                {"logit_range", TraceV3Writer::logit_range},
                {"code_max", 65535},
                {"zero_code_is_clipped", true},
                {"target_log_probability", "exact_float32"},
            }},
        }},
        {"model", {
            {"label", arguments.model_label.empty()
                ? arguments.mfq.stem().string()
                : arguments.model_label},
            {"architecture", model.header().architecture},
            {"mfq_container_version", model.header().version},
            {"precision", {
                {"classification", inferred_precision(dtype_counts)},
                {"record_dtype_counts", dtype_json},
            }},
            {"files", model_files},
        }},
        {"dataset", {
            {"id", arguments.dataset},
            {"file_name", arguments.input.filename().string()},
            {"bytes", regular_file_size(arguments.input)},
            {"sha256", dataset_sha256},
        }},
        {"tokenizer", {
            {"source", tokenizer_source},
            {"vocab_size", tokenizer.vocab_size},
            {"bos_token", tokenizer.bos_token},
            {"eos_token", tokenizer.eos_token},
            {"eot_token", tokenizer.eot_token},
            {"pad_token", tokenizer.pad_token},
            {"add_bos", tokenizer.add_bos},
            {"add_eos", tokenizer.add_eos},
            {"add_special", true},
            {"parse_special", true},
        }},
        {"evaluation", {
            {"protocol", "llama.cpp-non-strided-second-half-v1"},
            {"n_ctx", context},
            {"n_batch", batch_size},
            {"n_ubatch", ubatch_size},
            {"n_seq", n_seq},
            {"chunks", chunks},
            {"input_tokens", tokens.size()},
            {"input_token_ids_sha256", token_sha256},
            {"per_chunk_bos_replacement", tokenizer.add_bos},
            {"target_start", target_start},
            {"score_count_per_chunk", score_count},
            {"scored_tokens", stats.count},
            {"logit_positions", {target_start - 1, context - 2}},
            {"target_positions", {target_start, context - 1}},
            {"attention", "native_model_default"},
            {"kv_cache_dtype", "native_model_default"},
        }},
        {"result", {
            {"cross_entropy", static_cast<double>(
                stats.nll / static_cast<long double>(stats.count))},
            {"perplexity", stats.perplexity()},
        }},
    };

    {
        std::ofstream stream(partial, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error(
                "cannot create logits manifest: " + partial.string());
        }
        stream << std::setw(2) << manifest << "\n";
        if (!stream) {
            throw std::runtime_error("cannot write logits manifest");
        }
    }
    std::error_code error;
    std::filesystem::rename(partial, manifest_path, error);
    if (error) {
        std::filesystem::remove(partial);
        throw std::runtime_error(
            "cannot commit logits manifest: " + error.message());
    }
    std::cerr
        << "mfq-perplexity: wrote logits=" << arguments.logits_file
        << " manifest=" << manifest_path << "\n";
}

template <typename Runtime>
array forward_with_ubatch(
    Runtime& runtime,
    const array& ids,
    int batch,
    int context,
    int n_ubatch) {
    const int n_batch = batch * context;
    if (n_ubatch >= n_batch) {
        return runtime.forward(ids, false);
    }
    if (n_ubatch < batch) {
        throw std::runtime_error(
            "n_ubatch must fit at least one token for every sequence");
    }
    const int tokens_per_ubatch = std::max(1, n_ubatch / batch);
    runtime.reset_cache(batch);
    std::vector<array> outputs;
    outputs.reserve(
        static_cast<std::size_t>(
            (context + tokens_per_ubatch - 1) / tokens_per_ubatch));
    for (int start = 0; start < context; start += tokens_per_ubatch) {
        const int end = std::min(context, start + tokens_per_ubatch);
        auto chunk_ids = mlx::core::slice(
            ids,
            Shape{0, start},
            Shape{batch, end});
        auto output = runtime.forward(chunk_ids, true);
        // Cache writes are lazy in the Qwen runtime. Materialize every
        // physical ubatch before constructing the next cache-dependent graph.
        output.eval();
        outputs.push_back(std::move(output));
    }
    return mlx::core::concatenate(std::move(outputs), 1);
}

template <typename Runtime>
PerplexityStats evaluate(
    Runtime& runtime,
    const std::vector<std::int64_t>& tokens,
    const MfqTokenizerProbe& tokenizer,
    int vocabulary,
    int context,
    int chunks,
    int parallel,
    int ubatch_size,
    TraceV3Writer* writer = nullptr) {
    if (vocabulary <= 0 || tokenizer.vocab_size != vocabulary) {
        throw std::runtime_error(
            "tokenizer/model vocabulary mismatch: tokenizer=" +
            std::to_string(tokenizer.vocab_size) +
            " model=" + std::to_string(vocabulary));
    }
    if (tokenizer.add_eos) {
        throw std::runtime_error(
            "llama.cpp perplexity protocol requires tokenizer.add_eos=false");
    }
    if (tokenizer.add_bos &&
        (tokenizer.bos_token < 0 || tokenizer.bos_token >= vocabulary)) {
        throw std::runtime_error(
            "tokenizer requests BOS insertion but has no valid BOS token");
    }

    const int first = context / 2;
    const int scored_per_chunk = context - first - 1;
    if (scored_per_chunk <= 0) {
        throw std::runtime_error(
            "context window is too small to score next-token predictions");
    }

    PerplexityStats stats;
    const auto evaluation_started = Clock::now();
    bool printed_eta = false;
    for (int chunk0 = 0; chunk0 < chunks; chunk0 += parallel) {
        const int batch = std::min(parallel, chunks - chunk0);
        std::vector<std::int32_t> input;
        std::vector<std::int32_t> targets;
        input.reserve(static_cast<std::size_t>(batch) * context);
        targets.reserve(
            static_cast<std::size_t>(batch) * scored_per_chunk);
        for (int sequence = 0; sequence < batch; ++sequence) {
            const auto offset = static_cast<std::size_t>(
                chunk0 + sequence) * static_cast<std::size_t>(context);
            for (int position = 0; position < context; ++position) {
                auto token = tokens[offset + static_cast<std::size_t>(position)];
                if (position == 0 && tokenizer.add_bos) {
                    token = tokenizer.bos_token;
                }
                if (token < 0 || token >= vocabulary) {
                    throw std::runtime_error(
                        "evaluation text contains an out-of-range token at "
                        "window " + std::to_string(chunk0 + sequence + 1));
                }
                input.push_back(static_cast<std::int32_t>(token));
                if (position > first) {
                    targets.push_back(static_cast<std::int32_t>(token));
                }
            }
        }

        const array ids(
            input.begin(),
            Shape{batch, context},
            mlx::core::int32);
        auto logits = forward_with_ubatch(
            runtime,
            ids,
            batch,
            context,
            std::min(ubatch_size, batch * context));
        if (logits.ndim() != 3 ||
            logits.shape(0) != batch ||
            logits.shape(1) != context ||
            logits.shape(2) != vocabulary) {
            throw std::runtime_error(
                "model returned an invalid [batch,tokens,vocab] logits shape");
        }
        auto scored_logits = mlx::core::astype(
            mlx::core::slice(
                logits,
                Shape{0, first, 0},
                Shape{batch, context - 1, vocabulary}),
            mlx::core::float32);
        const array target_ids(
            targets.begin(),
            Shape{batch, scored_per_chunk},
            mlx::core::int32);
        auto target_logits = mlx::core::squeeze(
            mlx::core::take_along_axis(
                scored_logits,
                mlx::core::expand_dims(target_ids, -1),
                -1),
            -1);
        auto negative_log_likelihood =
            mlx::core::logsumexp(scored_logits, -1) - target_logits;
        negative_log_likelihood.eval();
        const auto* values =
            negative_log_likelihood.template data<float>();
        for (int sequence = 0; sequence < batch; ++sequence) {
            const auto row = static_cast<std::size_t>(sequence) *
                static_cast<std::size_t>(scored_per_chunk);
            if (writer != nullptr) {
                auto sequence_logits = mlx::core::reshape(
                    mlx::core::slice(
                        scored_logits,
                        Shape{sequence, 0, 0},
                        Shape{
                            sequence + 1,
                            scored_per_chunk,
                            vocabulary,
                        }),
                    Shape{scored_per_chunk, vocabulary});
                writer->write_chunk(
                    chunk0 + sequence,
                    sequence_logits,
                    values + row);
            }
            for (int position = 0; position < scored_per_chunk; ++position) {
                stats.add(values[row + static_cast<std::size_t>(position)]);
            }
            std::cout
                << "[" << (chunk0 + sequence + 1) << "]"
                << std::fixed << std::setprecision(4)
                << stats.perplexity() << ","
                << std::flush;
        }
        runtime.clear_cache();

        if (!printed_eta) {
            const auto seconds = std::chrono::duration<double>(
                Clock::now() - evaluation_started).count();
            const auto estimated = seconds *
                static_cast<double>(chunks) / static_cast<double>(batch);
            std::cerr
                << "\nmfq-perplexity: " << std::fixed
                << std::setprecision(2) << seconds
                << " seconds for first pass; ETA "
                << estimated / 60.0 << " minutes\n";
            printed_eta = true;
        }
    }
    std::cout << "\n";
    return stats;
}

void read_kl_rows(
    std::ifstream& stream,
    const KlReference& reference,
    const KlChunk& chunk,
    int row_start,
    int row_count,
    std::vector<float>& scales,
    std::vector<float>& minimums,
    std::vector<std::int32_t>& codes) {
    const auto row_bytes = static_cast<std::streamoff>(
        reference.row_elements * sizeof(std::uint16_t));
    stream.seekg(
        chunk.row_offset +
        static_cast<std::streamoff>(row_start) * row_bytes);
    if (!stream) {
        throw std::runtime_error("cannot seek within KLD reference rows");
    }
    std::vector<std::uint16_t> rows(
        static_cast<std::size_t>(row_count) *
        static_cast<std::size_t>(reference.row_elements));
    read_vector(stream, rows, "log-probability rows");
    scales.resize(static_cast<std::size_t>(row_count));
    minimums.resize(static_cast<std::size_t>(row_count));
    codes.resize(
        static_cast<std::size_t>(row_count) *
        static_cast<std::size_t>(reference.vocabulary));
    for (int row_index = 0; row_index < row_count; ++row_index) {
        const auto* row = rows.data() +
            static_cast<std::size_t>(row_index) *
                static_cast<std::size_t>(reference.row_elements);
        std::memcpy(
            &scales[static_cast<std::size_t>(row_index)],
            row,
            sizeof(float));
        std::memcpy(
            &minimums[static_cast<std::size_t>(row_index)],
            row + 2,
            sizeof(float));
        const auto scale = scales[static_cast<std::size_t>(row_index)];
        const auto minimum = minimums[static_cast<std::size_t>(row_index)];
        if (!std::isfinite(scale) || scale < 0.0f ||
            !std::isfinite(minimum)) {
            throw std::runtime_error(
                "invalid KLD reference row scale/minimum");
        }
        for (int token = 0; token < reference.vocabulary; ++token) {
            codes[
                static_cast<std::size_t>(row_index) *
                    static_cast<std::size_t>(reference.vocabulary) +
                static_cast<std::size_t>(token)] = row[4 + token];
        }
    }
}

void accumulate_kl_rows(
    std::ifstream& reference_stream,
    const KlReference& reference,
    const KlChunk& chunk,
    const array& predicted_logits,
    int score_count,
    KlStats& stats) {
    constexpr int row_batch = 8;
    const int vocabulary = reference.vocabulary;
    for (int start = 0; start < score_count; start += row_batch) {
        const int count = std::min(row_batch, score_count - start);
        std::vector<float> scales;
        std::vector<float> minimums;
        std::vector<std::int32_t> codes;
        read_kl_rows(
            reference_stream,
            reference,
            chunk,
            start,
            count,
            scales,
            minimums,
            codes);

        const array scale_values(
            scales.begin(), Shape{count, 1}, mlx::core::float32);
        const array minimum_values(
            minimums.begin(), Shape{count, 1}, mlx::core::float32);
        const array code_values(
            codes.begin(),
            Shape{count, vocabulary},
            mlx::core::int32);
        auto reference_logp =
            mlx::core::astype(code_values, mlx::core::float32) *
                scale_values +
            minimum_values;
        auto logits = mlx::core::slice(
            predicted_logits,
            Shape{start, 0},
            Shape{start + count, vocabulary});
        logits = mlx::core::astype(logits, mlx::core::float32);
        auto log_normalizer = mlx::core::logsumexp(
            logits, -1, true);
        auto mfq_logp = logits - log_normalizer;
        auto normalized_reference_logp = reference_logp -
            mlx::core::logsumexp(reference_logp, -1, true);
        auto reference_probability =
            mlx::core::exp(reference_logp) *
            mlx::core::astype(code_values != 0, mlx::core::float32);
        auto kld = mlx::core::sum(
            reference_probability *
                (reference_logp - logits + log_normalizer),
            -1);
        auto reverse_kld = mlx::core::sum(
            mlx::core::exp(mfq_logp) *
                (mfq_logp - normalized_reference_logp),
            -1);

        std::vector<std::int32_t> target_values(
            static_cast<std::size_t>(count));
        for (int row = 0; row < count; ++row) {
            const auto target = chunk.tokens[
                static_cast<std::size_t>(chunk.target_start + start + row)];
            if (target < 0 || target >= vocabulary) {
                throw std::runtime_error(
                    "KLD reference contains an out-of-range target token");
            }
            target_values[static_cast<std::size_t>(row)] = target;
        }
        const array target_ids(
            target_values.begin(), Shape{count, 1}, mlx::core::int32);
        auto mfq_ce = mlx::core::squeeze(
            log_normalizer -
                mlx::core::take_along_axis(
                    logits, target_ids, -1),
            -1);
        auto reference_ce = -mlx::core::squeeze(
            mlx::core::take_along_axis(
                reference_logp, target_ids, -1),
            -1);
        auto same_top = mlx::core::astype(
            mlx::core::argmax(logits, -1) ==
                mlx::core::argmax(reference_logp, -1),
            mlx::core::int32);

        mlx::core::eval(
            kld,
            reverse_kld,
            mfq_ce,
            reference_ce,
            same_top);
        const auto* kld_values = kld.data<float>();
        const auto* reverse_values = reverse_kld.data<float>();
        const auto* mfq_ce_values = mfq_ce.data<float>();
        const auto* reference_ce_values = reference_ce.data<float>();
        const auto* same_values = same_top.data<std::int32_t>();
        for (int row = 0; row < count; ++row) {
            const auto offset = static_cast<std::size_t>(row);
            if (!std::isfinite(kld_values[offset]) ||
                !std::isfinite(reverse_values[offset]) ||
                !std::isfinite(mfq_ce_values[offset]) ||
                !std::isfinite(reference_ce_values[offset])) {
                throw std::runtime_error(
                    "KLD evaluation produced a non-finite metric");
            }
            stats.kld += kld_values[offset];
            stats.reverse_kld += reverse_values[offset];
            stats.mfq_ce += mfq_ce_values[offset];
            if (chunk.target_log_probs.empty()) {
                stats.reference_ce += reference_ce_values[offset];
            } else {
                stats.reference_ce -= chunk.target_log_probs[
                    static_cast<std::size_t>(start + row)];
            }
            stats.same_top += same_values[offset] != 0 ? 1u : 0u;
            ++stats.count;
        }
    }
}

template <typename Runtime>
KlStats evaluate_kl(
    Runtime& runtime,
    const Arguments& arguments,
    const KlReference& reference,
    const ReferenceContract& contract,
    int model_vocabulary,
    int parallel,
    int score_count) {
    if (reference.vocabulary != model_vocabulary) {
        throw std::runtime_error(
            "KLD reference/model vocabulary mismatch: reference=" +
            std::to_string(reference.vocabulary) +
            " model=" + std::to_string(model_vocabulary));
    }
    const int context = static_cast<int>(reference.chunks.front().tokens.size());
    const int target_start = reference.chunks.front().target_start;
    for (const auto& chunk : reference.chunks) {
        if (chunk.tokens.size() != static_cast<std::size_t>(context) ||
            chunk.target_start != target_start ||
            chunk.score_count < score_count) {
            throw std::runtime_error(
                "integrated Metal KLD requires uniform chunk geometry");
        }
    }

    std::ifstream reference_stream(arguments.kl_base, std::ios::binary);
    if (!reference_stream) {
        throw std::runtime_error(
            "cannot reopen KLD reference: " + arguments.kl_base.string());
    }
    KlStats stats;
    const auto started = Clock::now();
    const int chunks = static_cast<int>(reference.chunks.size());
    std::cout
        << "cpp_kl_execution evaluator=optimized"
        << " graph=metal_batched_contexts"
        << " available_chunks=" << chunks
        << " selected_chunks=" << chunks
        << " n_ctx=" << context
        << " n_batch=" << parallel * context
        << " n_ubatch=" << contract.n_ubatch
        << " n_seq=" << parallel
        << " score_count=" << score_count
        << " score_count_override=" << arguments.score_count
        << " logits_start=" << target_start - 1
        << " reference_model=" << contract.model_label
        << " reference_precision=" << contract.precision
        << " dataset=" << contract.dataset_id
        << "\n";

    for (int begin = 0; begin < chunks; begin += parallel) {
        const int batch = std::min(parallel, chunks - begin);
        std::vector<std::int32_t> ids;
        ids.reserve(static_cast<std::size_t>(batch) * context);
        for (int sequence = 0; sequence < batch; ++sequence) {
            const auto& chunk = reference.chunks[
                static_cast<std::size_t>(begin + sequence)];
            for (const auto token : chunk.tokens) {
                if (token < 0 || token >= model_vocabulary) {
                    throw std::runtime_error(
                        "KLD reference contains an out-of-range input token");
                }
                ids.push_back(token);
            }
        }
        const array token_ids(
            ids.begin(), Shape{batch, context}, mlx::core::int32);
        auto logits = forward_with_ubatch(
            runtime,
            token_ids,
            batch,
            context,
            std::min(contract.n_ubatch, batch * context));
        if (logits.ndim() != 3 ||
            logits.shape(0) != batch ||
            logits.shape(1) != context ||
            logits.shape(2) != model_vocabulary) {
            throw std::runtime_error(
                "model returned an invalid KLD logits shape");
        }
        for (int sequence = 0; sequence < batch; ++sequence) {
            const auto& chunk = reference.chunks[
                static_cast<std::size_t>(begin + sequence)];
            auto predicted = mlx::core::reshape(
                mlx::core::slice(
                    logits,
                    Shape{sequence, target_start - 1, 0},
                    Shape{
                        sequence + 1,
                        target_start - 1 + score_count,
                        model_vocabulary,
                    }),
                Shape{score_count, model_vocabulary});
            accumulate_kl_rows(
                reference_stream,
                reference,
                chunk,
                predicted,
                score_count,
                stats);
            std::cout
                << "cpp_kl_chunk=" << (begin + sequence + 1)
                << " mean="
                << static_cast<double>(
                       stats.kld / static_cast<long double>(stats.count))
                << " mean_kld_q_ref="
                << static_cast<double>(
                       stats.reverse_kld /
                       static_cast<long double>(stats.count))
                << " same_top="
                << static_cast<double>(stats.same_top) /
                       static_cast<double>(stats.count)
                << "\n";
        }
        runtime.clear_cache();
    }
    const auto seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    const auto denominator = static_cast<long double>(stats.count);
    const auto mean_kld = stats.kld / denominator;
    const auto mean_reverse = stats.reverse_kld / denominator;
    const auto mean_reference_ce = stats.reference_ce / denominator;
    const auto mean_mfq_ce = stats.mfq_ce / denominator;
    std::cout
        << "cpp_kl_result chunks=" << chunks
        << " scored_tokens=" << stats.count
        << " sec=" << seconds
        << " kld=" << static_cast<double>(mean_kld)
        << " mean_kld_q_ref=" << static_cast<double>(mean_reverse)
        << " reference_ce=" << static_cast<double>(mean_reference_ce)
        << " candidate_ce=" << static_cast<double>(mean_mfq_ce)
        << " reference_ppl=" << std::exp(static_cast<double>(mean_reference_ce))
        << " candidate_ppl=" << std::exp(static_cast<double>(mean_mfq_ce))
        << " kld_pct_reference_ce="
        << static_cast<double>(100.0L * stats.kld / stats.reference_ce)
        << " same_top="
        << static_cast<double>(stats.same_top) /
               static_cast<double>(stats.count)
        << " same_top_count=" << stats.same_top
        << " reference_format="
        << (reference.format == "_logit3_"
                ? "trace_v3"
                : (reference.format == "_logit2_"
                       ? "trace_v2" : "legacy"))
        << " execution=optimized"
        << " graph=metal_batched_contexts"
        << " n_ctx=" << context
        << " n_batch=" << parallel * context
        << " n_ubatch=" << contract.n_ubatch
        << " n_seq=" << parallel
        << " score_count=" << score_count
        << " score_count_override=" << arguments.score_count
        << " reference_model=" << contract.model_label
        << " reference_precision=" << contract.precision
        << " dataset=" << contract.dataset_id
        << " manifest=" << contract.path.filename().string()
        << "\n";
    return stats;
}

int run_kl(
    const Arguments& arguments,
    const mfq::metal::MfqContainer& model) {
    std::cout << std::unitbuf;
    auto reference = load_kl_reference(
        arguments.kl_base,
        -1);
    if (reference.chunks.empty()) {
        throw std::runtime_error(
            "KLD evaluation requires at least one reference chunk");
    }
    if (reference.format == "_logits_") {
        const auto policy = tokenizer_policy(arguments, model);
        if (policy.vocab_size != reference.vocabulary) {
            throw std::runtime_error(
                "legacy KLD tokenizer/reference vocabulary mismatch");
        }
        if (policy.add_bos) {
            if (policy.bos_token < 0 ||
                policy.bos_token >= reference.vocabulary) {
                throw std::runtime_error(
                    "legacy KLD tokenizer requests an invalid BOS token");
            }
            for (auto& chunk : reference.chunks) {
                chunk.tokens.front() = policy.bos_token;
            }
        }
    }
    const auto contract = load_reference_contract(
        arguments,
        reference);
    const int context = contract.n_ctx;
    const int score_count = contract.score_count;
    const int parallel = contract.n_seq;
    std::cerr
        << "mfq-perplexity: integrated KLD reference="
        << (reference.format == "_logit3_"
                ? "trace_v3"
                : (reference.format == "_logit2_"
                       ? "trace_v2" : "legacy"))
        << " chunks=" << reference.chunks.size()
        << " n_ctx=" << context
        << " n_batch=" << parallel * context
        << " n_seq=" << parallel
        << " score_count=" << score_count
        << " n_ubatch=" << contract.n_ubatch
        << " dataset=" << contract.dataset_id
        << " reference_model=" << contract.model_label
        << " reference_precision=" << contract.precision
        << "\n";

    configure_mlx_metal();
    const auto runtime_stream = mlx::core::new_thread_unsafe_stream(
        mlx::core::Device::gpu);
    mlx::core::set_default_stream(runtime_stream);
    const auto load_started = Clock::now();
    const auto& architecture = model.header().architecture;
    if (architecture.rfind("deepseek_v4", 0) == 0) {
        const auto config =
            mfq::metal::DeepseekV4Config::from_mfq(model);
        if (context > config.max_position_embeddings) {
            throw std::runtime_error(
                "KLD reference exceeds DeepSeek-V4 context capacity");
        }
        auto runtime = mfq::metal::MlxDeepseekV4CausalLm::load(
            model,
            context,
            expert_cache_bytes(arguments.expert_cache_gb));
        std::cerr
            << "mfq-perplexity: loaded DeepSeek-V4 in "
            << std::chrono::duration<double>(Clock::now() - load_started).count()
            << " s\n";
        (void)evaluate_kl(
            runtime,
            arguments,
            reference,
            contract,
            static_cast<int>(config.vocab),
            parallel,
            score_count);
    } else {
        const auto config = mfq::metal::Qwen35Config::from_mfq(model);
        if (context > config.max_position_embeddings) {
            throw std::runtime_error(
                "KLD reference exceeds Qwen model context capacity");
        }
        auto runtime = mfq::metal::MlxQwen35CausalLm::load(model);
        const auto model_type = config.text_model_type.empty()
            ? config.model_type
            : config.text_model_type;
        std::cerr
            << "mfq-perplexity: loaded "
            << (model_type.empty() ? "Qwen3.x" : model_type)
            << " in "
            << std::chrono::duration<double>(Clock::now() - load_started).count()
            << " s\n";
        (void)evaluate_kl(
            runtime,
            arguments,
            reference,
            contract,
            static_cast<int>(config.vocab_size),
            parallel,
            score_count);
    }
    return EXIT_SUCCESS;
}

template <typename Runtime>
PerplexityStats run_perplexity(
    Runtime& runtime,
    const Arguments& arguments,
    const mfq::metal::MfqContainer& model,
    const MfqTokenizerProbe& tokenizer,
    int vocabulary,
    int chunks,
    int parallel,
    int ubatch_size) {
    std::unique_ptr<TraceV3Writer> writer;
    std::vector<std::int32_t> serialized_tokens;
    const int target_start = arguments.context / 2 + 1;
    const int score_count = arguments.context - target_start;
    if (!arguments.logits_file.empty()) {
        serialized_tokens = evaluated_tokens(
            tokenizer.tokens,
            tokenizer,
            arguments.context,
            chunks,
            vocabulary);
        writer = std::make_unique<TraceV3Writer>(
            arguments.logits_file,
            serialized_tokens,
            vocabulary,
            arguments.context,
            chunks,
            target_start,
            score_count);
    }
    auto stats = evaluate(
        runtime,
        tokenizer.tokens,
        tokenizer,
        vocabulary,
        arguments.context,
        chunks,
        parallel,
        ubatch_size,
        writer.get());
    if (writer) {
        writer->commit();
        try {
            write_logits_manifest(
                arguments,
                model,
                tokenizer,
                serialized_tokens,
                vocabulary,
                arguments.context,
                chunks,
                parallel * arguments.context,
                ubatch_size,
                target_start,
                score_count,
                stats);
        } catch (...) {
            // A reference without its contract is deliberately unusable.
            // Remove only the output that this invocation just committed.
            std::error_code ignored;
            std::filesystem::remove(arguments.logits_file, ignored);
            throw;
        }
    }
    return stats;
}

int run(const Arguments& arguments) {
    if (arguments.mfq.empty()) {
        usage_error("--mfq is required");
    }
    if (arguments.input.empty() == arguments.kl_base.empty()) {
        usage_error("pass exactly one of --file or --kl-base");
    }
    if (!arguments.logits_file.empty() && arguments.input.empty()) {
        usage_error("--logits-file requires --file");
    }
    if (!arguments.logits_manifest.empty() &&
        arguments.logits_file.empty()) {
        usage_error("--logits-manifest requires --logits-file");
    }
    if (!arguments.kl_manifest.empty() && arguments.kl_base.empty()) {
        usage_error("--kl-manifest requires --kl-base");
    }
    if (!arguments.logits_file.empty() && arguments.dataset.empty()) {
        usage_error("--dataset NAME is required with --logits-file");
    }
    if (arguments.score_count > 0 && arguments.kl_base.empty()) {
        usage_error("--kl-score-count requires --kl-base");
    }
    if (!arguments.logits_file.empty()) {
        const auto manifest = arguments.logits_manifest.empty()
            ? default_manifest_path(arguments.logits_file)
            : arguments.logits_manifest;
        const auto manifest_partial = std::filesystem::path(
            manifest.string() + ".partial");
        if (manifest == arguments.logits_file) {
            usage_error("logits and manifest outputs must be different files");
        }
        if (std::filesystem::exists(manifest) ||
            std::filesystem::exists(manifest_partial)) {
            usage_error(
                "refusing to overwrite logits manifest or partial: " +
                manifest.string());
        }
        std::error_code error;
        const auto parent = manifest.parent_path().empty()
            ? std::filesystem::current_path()
            : manifest.parent_path();
        if (!std::filesystem::is_directory(parent, error) || error) {
            usage_error(
                "logits manifest directory does not exist: " +
                parent.string());
        }
    }

    const mfq::metal::MfqContainer model(arguments.mfq);
    if (!arguments.kl_base.empty()) {
        return run_kl(arguments, model);
    }
    const auto text = read_text(arguments.input);
    const auto tokenization_started = Clock::now();
    const auto tokenizer = tokenize(arguments, model, text);
    const auto tokenization_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - tokenization_started).count();
    if (tokenizer.tokens.size() <
        static_cast<std::size_t>(2) *
            static_cast<std::size_t>(arguments.context)) {
        throw std::runtime_error(
            "need at least " + std::to_string(2 * arguments.context) +
            " tokens for ctx-size " + std::to_string(arguments.context) +
            "; input tokenized to " +
            std::to_string(tokenizer.tokens.size()));
    }
    const auto available_chunks = tokenizer.tokens.size() /
        static_cast<std::size_t>(arguments.context);
    const auto requested_chunks = arguments.chunks < 0
        ? available_chunks
        : std::min<std::size_t>(
              available_chunks,
              static_cast<std::size_t>(arguments.chunks));
    if (requested_chunks >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("evaluation has too many context windows");
    }
    const auto chunks = static_cast<int>(requested_chunks);
    const auto score_count = arguments.context - arguments.context / 2 - 1;
    const int parallel = resolved_parallel(
        arguments,
        arguments.context);
    const int batch_size = parallel * arguments.context;
    const int ubatch_size = resolved_ubatch(arguments, batch_size);
    if (ubatch_size < parallel) {
        usage_error(
            "--ubatch-size must fit at least one token per parallel sequence");
    }
    std::cerr
        << "mfq-perplexity: tokenization took "
        << std::fixed << std::setprecision(2) << tokenization_ms << " ms\n"
        << "mfq-perplexity: calculating perplexity over "
        << chunks
        << " chunks, n_ctx=" << arguments.context
        << ", batch_size=" << batch_size
        << ", ubatch_size=" << ubatch_size
        << ", n_seq=" << parallel
        << ", score_tokens_per_chunk=" << score_count
        << "\n";

    configure_mlx_metal();
    const auto runtime_stream = mlx::core::new_thread_unsafe_stream(
        mlx::core::Device::gpu);
    mlx::core::set_default_stream(runtime_stream);
    const auto load_started = Clock::now();
    PerplexityStats stats;
    const auto& architecture = model.header().architecture;
    if (architecture.rfind("deepseek_v4", 0) == 0) {
        const auto config =
            mfq::metal::DeepseekV4Config::from_mfq(model);
        if (arguments.context > config.max_position_embeddings) {
            throw std::runtime_error(
                "--ctx-size exceeds DeepSeek-V4 model context capacity");
        }
        auto runtime = mfq::metal::MlxDeepseekV4CausalLm::load(
            model,
            arguments.context,
            expert_cache_bytes(arguments.expert_cache_gb));
        std::cerr
            << "mfq-perplexity: loaded DeepSeek-V4 in "
            << std::chrono::duration<double>(Clock::now() - load_started).count()
            << " s\n";
        stats = run_perplexity(
            runtime,
            arguments,
            model,
            tokenizer,
            static_cast<int>(config.vocab),
            chunks,
            parallel,
            ubatch_size);
    } else {
        const auto config = mfq::metal::Qwen35Config::from_mfq(model);
        if (arguments.context > config.max_position_embeddings) {
            throw std::runtime_error(
                "--ctx-size exceeds Qwen3.5 model context capacity");
        }
        auto runtime = mfq::metal::MlxQwen35CausalLm::load(model);
        std::cerr
            << "mfq-perplexity: loaded "
            << (config.text_model_type.empty()
                    ? (config.model_type.empty()
                           ? "Qwen3.x"
                           : config.model_type)
                    : config.text_model_type)
            << " in "
            << std::chrono::duration<double>(Clock::now() - load_started).count()
            << " s\n";
        stats = run_perplexity(
            runtime,
            arguments,
            model,
            tokenizer,
            static_cast<int>(config.vocab_size),
            chunks,
            parallel,
            ubatch_size);
    }

    std::cerr
        << "Final estimate: PPL = "
        << std::fixed << std::setprecision(4) << stats.perplexity()
        << " +/- " << std::setprecision(5) << stats.uncertainty()
        << " (" << stats.count << " scored tokens)\n";
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        if (arguments.help) {
            print_help();
            return EXIT_SUCCESS;
        }
        return run(arguments);
    } catch (const std::exception& error) {
        std::cerr << "mfq-perplexity: error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
