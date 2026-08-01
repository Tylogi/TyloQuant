#include "mfq_container.h"
#include "mlx_deepseek_v4_causal_lm.h"
#include "mlx_qwen35_causal_lm.h"
#include "mlx_tensor.h"
#include "qwen35_model.h"

#ifdef MFQ_METAL_SERVER
#include "../mfq_server.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include <mach-o/dyld.h>

namespace {

struct Arguments {
    std::filesystem::path mfq;
    std::string tensor;
    bool check_container = false;
    bool list_tensors = false;
    bool self_test_metal = false;
    bool server = false;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::int64_t context_size = 32768;
    double expert_cache_gb = 4.0;
    std::string model_name;
    std::string api_key;
    std::filesystem::path web_root;
    std::filesystem::path tokenizer_gguf;
    bool help = false;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::runtime_error(
        message +
        "\nRun mfq-decode-metal --help for supported options.");
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
        if (value == "--mfq") {
            result.mfq = require_value("--mfq");
        } else if (value == "--tensor") {
            result.tensor = require_value("--tensor");
        } else if (value == "--check-mfq-container") {
            result.check_container = true;
        } else if (value == "--list-tensors") {
            result.list_tensors = true;
        } else if (value == "--self-test-metal") {
            result.self_test_metal = true;
        } else if (value == "--server") {
            result.server = true;
        } else if (value == "--host") {
            result.host = require_value("--host");
        } else if (value == "--port") {
            const auto parsed = std::stoll(require_value("--port"));
            if (parsed < 1 || parsed > 65535) {
                usage_error("--port must be in [1, 65535]");
            }
            result.port = static_cast<int>(parsed);
        } else if (value == "--ctx-size") {
            const auto parsed = std::stoll(
                require_value("--ctx-size"));
            if (parsed <= 0) {
                usage_error("--ctx-size must be positive");
            }
            result.context_size = parsed;
        } else if (value == "--expert-cache-gb") {
            const auto text =
                require_value("--expert-cache-gb");
            std::size_t consumed = 0;
            const auto parsed =
                std::stod(text, &consumed);
            if (consumed != text.size() ||
                !std::isfinite(parsed) ||
                parsed < 0.0) {
                usage_error(
                    "--expert-cache-gb must be a finite "
                    "non-negative number");
            }
            result.expert_cache_gb = parsed;
        } else if (value == "--model-name") {
            result.model_name = require_value("--model-name");
        } else if (value == "--api-key") {
            result.api_key = require_value("--api-key");
        } else if (value == "--web-root") {
            result.web_root = require_value("--web-root");
        } else if (value == "--tokenizer-gguf") {
            result.tokenizer_gguf =
                require_value("--tokenizer-gguf");
        } else if (value == "--help" || value == "-h") {
            result.help = true;
        } else {
            usage_error("unknown option: " + value);
        }
    }
    return result;
}

void print_help() {
    std::cout
        << "MFQ native MLX/Metal runtime\n\n"
        << "Usage:\n"
        << "  mfq-decode-metal --mfq MODEL.mfq --check-mfq-container\n"
        << "  mfq-decode-metal --mfq MODEL.mfq --list-tensors\n"
        << "  mfq-decode-metal --mfq MODEL.mfq --tensor NAME\n"
        << "  mfq-decode-metal --mfq MODEL.mfq --server "
           "[--host 127.0.0.1 --port 8080]\n"
        << "  mfq-decode-metal --self-test-metal\n\n"
        << "Options:\n"
        << "  --mfq PATH             MFQ model or any shard in a split model\n"
        << "  --check-mfq-container  validate headers, records, and shard set\n"
        << "  --list-tensors         print record dtype, bytes, and name\n"
        << "  --tensor NAME          load and execute one supported linear weight\n"
        << "  --self-test-metal      execute an MLX C++ graph on Metal\n"
        << "  --server               run the native C++ OpenAI-compatible server\n"
        << "  --host ADDRESS         server bind address (default 127.0.0.1)\n"
        << "  --port PORT            server port (default 8080)\n"
        << "  --ctx-size TOKENS      runtime/API context limit (default 32768)\n"
        << "  --expert-cache-gb N    DeepSeek-V4 packed expert LRU (default 4)\n"
        << "  --model-name NAME      API model name (default MFQ filename)\n"
        << "  --api-key KEY          optional bearer token\n"
        << "  --web-root PATH        WebUI assets (default beside executable)\n"
        << "  --tokenizer-gguf PATH  external tokenizer GGUF when not embedded\n";
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
        "lib/mlx.metallib beside mfq-decode-metal");
}

void self_test_metal() {
    using namespace mlx::core;
    const array left({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    const array right({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2});
    auto result = matmul(left, right);
    result.eval();
    const float expected[] = {19.0f, 22.0f, 43.0f, 50.0f};
    const auto* values = result.data<float>();
    for (std::size_t index = 0; index < 4; ++index) {
        if (std::fabs(values[index] - expected[index]) > 1e-4f) {
            throw std::runtime_error("MLX Metal self-test returned wrong data");
        }
    }
    std::cout << "MLX C++ Metal self-test passed\n";
}

#ifdef MFQ_METAL_SERVER
std::int32_t generate_with_prefill_metrics(
    mfq::metal::MlxQwen35CausalLm& runtime,
    const std::vector<std::int64_t>& prompt,
    const mfq::metal::MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const MfqTokenCallback& callback,
    const MfqPrefillCallback& on_prefill) {
    return runtime.generate(
        prompt,
        sampling,
        max_tokens,
        callback,
        on_prefill);
}

std::int32_t generate_with_prefill_metrics(
    mfq::metal::MlxDeepseekV4CausalLm& runtime,
    const std::vector<std::int64_t>& prompt,
    const mfq::metal::MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const MfqTokenCallback& callback,
    const MfqPrefillCallback& on_prefill) {
    return runtime.generate(
        prompt,
        sampling,
        max_tokens,
        callback,
        std::nullopt,
        512,
        on_prefill);
}

template <typename Runtime>
int serve_loaded_runtime(
    const Arguments& arguments,
    const mfq::metal::MfqContainer& container,
    Runtime& runtime,
    std::string model_type,
    std::int64_t maximum_context,
    std::int64_t vocabulary_size,
    mlx::core::Stream runtime_stream) {
    constexpr const char* tokenizer_asset =
        "__mfq_asset__/tokenizer.gguf";
    MfqServerConfig server;
    server.host = arguments.host;
    server.port = arguments.port;
    server.model_name = arguments.model_name.empty()
        ? arguments.mfq.stem().string()
        : arguments.model_name;
    server.model_type = std::move(model_type);
    if (container.contains(tokenizer_asset)) {
        server.tokenizer_gguf =
            container.read(tokenizer_asset);
    } else {
        server.tokenizer_model =
            arguments.tokenizer_gguf.string();
    }
    server.api_key = arguments.api_key;
    const auto default_web_root =
        executable_path().parent_path() / "web";
    server.web_root = (
        arguments.web_root.empty()
            ? default_web_root
            : arguments.web_root).string();
    server.max_context = std::min<std::int64_t>(
        arguments.context_size,
        maximum_context);
    server.vocab_size = vocabulary_size;

    std::mutex runtime_mutex;
    const MfqGenerateFn generate =
        [&, runtime_stream](
            const std::vector<std::int64_t>& prompt,
            const MfqSamplingParams& sampling,
            const MfqTokenCallback& callback,
            const MfqPrefillCallback& on_prefill) {
            std::lock_guard<std::mutex> lock(runtime_mutex);
            mlx::core::set_default_device(
                mlx::core::Device::gpu);
            mlx::core::set_default_stream(runtime_stream);
            mfq::metal::MlxSamplingParams parameters;
            parameters.temperature = sampling.temperature;
            parameters.top_k = sampling.top_k;
            parameters.top_p = sampling.top_p;
            parameters.presence_penalty =
                sampling.presence_penalty;
            parameters.frequency_penalty =
                sampling.frequency_penalty;
            parameters.repetition_penalty =
                sampling.repetition_penalty;
            parameters.seed = sampling.seed;
            return generate_with_prefill_metrics(
                runtime,
                prompt,
                parameters,
                sampling.max_tokens,
                callback,
                on_prefill);
        };
    return run_mfq_server(server, generate);
}

int run_native_server(
    const Arguments& arguments,
    const mfq::metal::MfqContainer& container) {
    constexpr const char* tokenizer_asset =
        "__mfq_asset__/tokenizer.gguf";
    if (!container.contains(tokenizer_asset) &&
        arguments.tokenizer_gguf.empty()) {
        throw std::runtime_error(
            "MFQ model has no embedded tokenizer.gguf asset; "
            "pass --tokenizer-gguf PATH");
    }

    const auto runtime_stream =
        mlx::core::new_thread_unsafe_stream(
            mlx::core::Device::gpu);
    mlx::core::set_default_stream(runtime_stream);
    const auto started =
        std::chrono::steady_clock::now();
    const auto& architecture =
        container.header().architecture;
    if (architecture.rfind(
            "deepseek_v4",
            0) == 0) {
        const auto config =
            mfq::metal::DeepseekV4Config::from_mfq(
                container);
        const int context = static_cast<int>(
            std::min<std::int64_t>(
                arguments.context_size,
                config.max_position_embeddings));
        constexpr long double bytes_per_gib =
            static_cast<long double>(
                std::uint64_t{1} << 30);
        const long double requested_cache =
            static_cast<long double>(
                arguments.expert_cache_gb) *
            bytes_per_gib;
        if (requested_cache >
            static_cast<long double>(
                std::numeric_limits<
                    std::size_t>::max())) {
            throw std::runtime_error(
                "DeepSeek-V4 expert cache exceeds "
                "addressable memory");
        }
        const auto expert_cache_bytes =
            static_cast<std::size_t>(
                requested_cache);
        std::cout
            << "Loading native C++/MLX DeepSeek-V4 model "
               "on Apple GPU..."
            << std::endl;
        auto runtime =
            mfq::metal::MlxDeepseekV4CausalLm::load(
                container,
                context,
                expert_cache_bytes);
        const auto load_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                started).count();
        std::cout
            << "Loaded " << runtime.layer_count()
            << " DeepSeek-V4 layers in "
            << load_seconds << " s"
            << " expert_cache_gb="
            << arguments.expert_cache_gb
            << std::endl;
        return serve_loaded_runtime(
            arguments,
            container,
            runtime,
            config.model_type,
            config.max_position_embeddings,
            config.vocab,
            runtime_stream);
    }

    const auto config =
        mfq::metal::Qwen35Config::from_mfq(container);
    std::cout
        << "Loading native C++/MLX Qwen3.5 model "
           "on Apple GPU..."
        << std::endl;
    auto runtime =
        mfq::metal::MlxQwen35CausalLm::load(container);
    const auto load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::cout
        << "Loaded " << runtime.layer_count()
        << " Qwen3.5 layers in "
        << load_seconds << " s"
        << std::endl;
    return serve_loaded_runtime(
        arguments,
        container,
        runtime,
        config.text_model_type.empty()
            ? config.model_type
            : config.text_model_type,
        config.max_position_embeddings,
        config.vocab_size,
        runtime_stream);
}
#endif

} // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        if (arguments.help) {
            print_help();
            return EXIT_SUCCESS;
        }
        if (arguments.self_test_metal) {
            configure_mlx_metal();
            self_test_metal();
        }
        if (arguments.mfq.empty()) {
            if (!arguments.self_test_metal) {
                usage_error("--mfq is required");
            }
            return EXIT_SUCCESS;
        }

        const mfq::metal::MfqContainer model(arguments.mfq);
        if (arguments.check_container) {
            std::cout
                << "MFQ container OK"
                << " version=" << model.header().version
                << " architecture=" << model.header().architecture
                << " shards=" << model.source_paths().size()
                << " records=" << model.records().size()
                << "\n";
        }
        if (arguments.list_tensors) {
            for (const auto& item : model.records()) {
                const auto& value = item.second;
                std::cout
                    << value.dtype << "\t"
                    << value.nbytes << "\t"
                    << value.name << "\n";
            }
        }
        if (!arguments.tensor.empty()) {
            configure_mlx_metal();
            const auto& record = model.record(arguments.tensor);
            const auto weight = mfq::metal::MlxLinear::load(
                model,
                arguments.tensor);
            std::vector<float> input_values(
                static_cast<std::size_t>(weight.input_size()));
            for (std::size_t index = 0;
                 index < input_values.size();
                 ++index) {
                input_values[index] =
                    static_cast<float>(
                        static_cast<int>(index % 17) - 8) /
                    16.0f;
            }
            auto input = mlx::core::array(
                input_values.begin(),
                mlx::core::Shape{1, weight.input_size()});
            input = mlx::core::astype(input, mlx::core::float16);
            auto output = weight(input);
            output = mlx::core::astype(
                output,
                mlx::core::float32);
            output.eval();
            const auto* values = output.data<float>();
            float maximum = 0.0f;
            for (std::size_t index = 0; index < output.size(); ++index) {
                if (!std::isfinite(values[index])) {
                    throw std::runtime_error(
                        "Metal linear smoke test returned non-finite data");
                }
                maximum = std::max(maximum, std::fabs(values[index]));
            }
            if (maximum <= 1e-12f) {
                throw std::runtime_error(
                    "Metal linear smoke test unexpectedly returned all zero");
            }
            std::cout
                << "Metal linear smoke test passed"
                << " dtype=" << record.dtype
                << " in=" << weight.input_size()
                << " out=" << weight.output_size()
                << " packed=" << (weight.packed() ? "true" : "false")
                << "\n";
        }
        if (arguments.server) {
            configure_mlx_metal();
#ifdef MFQ_METAL_SERVER
            return run_native_server(arguments, model);
#else
            throw std::runtime_error(
                "this build has no C++ server support; configure with "
                "-DMFQ_BUILD_CPP_SERVER=ON and matching llama.cpp paths");
#endif
        }
        if (!arguments.check_container &&
            !arguments.list_tensors &&
            arguments.tensor.empty() &&
            !arguments.self_test_metal &&
            !arguments.server) {
            usage_error(
                "select --check-mfq-container, --list-tensors, --tensor, "
                "--server, or --self-test-metal");
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "mfq-decode-metal: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
