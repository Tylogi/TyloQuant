#include "mfq_container.h"
#include "mfq/fatal_diagnostics.h"
#include "mlx_deepseek_v4_causal_lm.h"
#include "mlx_deepseek_v41_causal_lm.h"
#include "mlx_legacy_tensor_compat.h"
#include "mlx_minicpmo45.h"
#include "mlx_nemotron_h_causal_lm.h"
#include "nemotron_h_model.h"
#include "mlx_moe.h"
#include "mlx_memory_residency.h"
#include "mlx_resident_budget.h"
#include "mlx_qwen4_causal_lm.h"
#include "mlx_qwen35_causal_lm.h"
#include "mlx_stream_sync.h"
#include "mlx_tensor.h"
#include "qwen35_model.h"

#include "nlohmann/json.hpp"

#ifdef MFQ_METAL_RUNTIME_COMMUNICATION
#include "mfq_paged_prefix_cache.h"
#include "transport.h"
#include "mlx_paged_session_codec.h"
#include "mlx_server_components.h"
#include "mlx_engine.h"
#include "mlx_qwen4_batching.h"
#include "mlx_kernel_prepare.h"
#include "mlx_inference_warmup.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

#include <mach/mach.h>
#include <mach-o/dyld.h>
#include <malloc/malloc.h>
#include <sys/sysctl.h>

namespace {

constexpr std::size_t kMinicpmoDuplexCacheLimitBytes =
    std::size_t{8} << 30;
constexpr std::size_t kMinimumServerCacheLimitBytes =
    std::size_t{1} << 30;
constexpr std::size_t kMaximumServerCacheLimitBytes =
    std::size_t{8} << 30;
constexpr std::size_t kGiB = std::size_t{1} << 30;

void release_host_allocator_cache() {
    malloc_zone_pressure_relief(nullptr, 0);
}

void release_allocator_caches() {
    mlx::core::clear_cache();
    release_host_allocator_cache();
}

void release_model_load_staging_memory(
    const mlx::core::Stream& runtime_stream) {
    // Model conversion and MFE repacking leave large, now-unused buffers in
    // both the MLX cache and macOS malloc's large-object depot. Keeping those
    // pages makes a single fully resident model look tens of GiB larger and
    // can force useful weights into swap before the first request.
    mfq::metal::drain_metal_work(runtime_stream);
    release_allocator_caches();
}

struct Arguments {
    std::filesystem::path mfq;
    std::string tensor;
    std::string benchmark_mfq_moe_prefix;
    int benchmark_reps = 1;
    int benchmark_tokens = 1;
    int benchmark_distributed_routes = 0;
    std::vector<std::int32_t> benchmark_experts;
    std::vector<std::string> benchmark_concat_tensors;
    int benchmark_batch_size = 1;
    bool benchmark_routed_input = false;
    bool benchmark_swiglu = false;
    bool check_container = false;
    bool list_tensors = false;
    bool self_test_metal = false;
    bool server = false;
    bool stdio = false;
    bool predequantize_fp16 = false;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::int64_t context_size = 32768;
    bool yarn_context_extension = false;
    int prefill_chunk_size = 2048;
    bool prefill_chunk_size_explicit = false;
    int continuous_batching = 8;
    std::optional<double> expert_cache_gb;
    std::string model_name;
    std::string api_key;
    std::filesystem::path tokenizer_gguf;
    std::filesystem::path sampling_profile;
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
        if (value == "--continuous-batching") {
            const auto parsed = std::stoll(require_value("--continuous-batching"));
            if (parsed < 0 || parsed > 256) usage_error("--continuous-batching must be in [0, 256]");
            result.continuous_batching = static_cast<int>(parsed);
        } else if (value == "--model") {
            result.mfq = require_value("--model");
        } else if (value == "--tensor") {
            result.tensor = require_value("--tensor");
        } else if (value == "--benchmark-mfq-moe") {
            result.benchmark_mfq_moe_prefix =
                require_value("--benchmark-mfq-moe");
        } else if (value == "--benchmark-swiglu") {
            result.benchmark_swiglu = true;
        } else if (value == "--benchmark-concat-tensor") {
            result.benchmark_concat_tensors.push_back(
                require_value("--benchmark-concat-tensor"));
        } else if (value == "--benchmark-routed-input") {
            result.benchmark_routed_input = true;
        } else if (value == "--benchmark-batch-size") {
            const auto parsed = std::stoll(require_value("--benchmark-batch-size"));
            if (parsed <= 0 || parsed > 10000) {
                usage_error("--benchmark-batch-size must be in [1, 10000]");
            }
            result.benchmark_batch_size = static_cast<int>(parsed);
        } else if (value == "--benchmark-reps") {
            const auto parsed = std::stoll(
                require_value("--benchmark-reps"));
            if (parsed <= 0 || parsed > 10000) {
                usage_error(
                    "--benchmark-reps must be in [1, 10000]");
            }
            result.benchmark_reps = static_cast<int>(parsed);
        } else if (value == "--benchmark-tokens") {
            const auto parsed = std::stoll(
                require_value("--benchmark-tokens"));
            if (parsed <= 0 || parsed > 131072) {
                usage_error(
                    "--benchmark-tokens must be in [1, 131072]");
            }
            result.benchmark_tokens = static_cast<int>(parsed);
        } else if (value == "--benchmark-experts") {
            const auto text = require_value("--benchmark-experts");
            std::size_t begin = 0;
            while (begin < text.size()) {
                const auto end = text.find(',', begin);
                const auto item = text.substr(
                    begin,
                    end == std::string::npos
                        ? std::string::npos
                        : end - begin);
                std::size_t consumed = 0;
                const auto parsed = std::stoll(item, &consumed);
                if (
                    item.empty()
                    || consumed != item.size()
                    || parsed < 0
                    || parsed > std::numeric_limits<std::int32_t>::max()
                ) {
                    usage_error(
                        "--benchmark-experts must be a comma-separated "
                        "list of non-negative expert IDs");
                }
                result.benchmark_experts.push_back(
                    static_cast<std::int32_t>(parsed));
                if (end == std::string::npos) {
                    break;
                }
                begin = end + 1;
            }
            if (result.benchmark_experts.empty()) {
                usage_error("--benchmark-experts cannot be empty");
            }
        } else if (value == "--benchmark-distributed-experts") {
            const auto parsed = std::stoll(
                require_value("--benchmark-distributed-experts"));
            if (parsed <= 0 || parsed > 1024) {
                usage_error(
                    "--benchmark-distributed-experts must be in [1, 1024]");
            }
            result.benchmark_distributed_routes = static_cast<int>(parsed);
        } else if (value == "--check-mfq-container") {
            result.check_container = true;
        } else if (value == "--list-tensors") {
            result.list_tensors = true;
        } else if (value == "--self-test-metal") {
            result.self_test_metal = true;
        } else if (value == "--transport") {
            const std::string transport = require_value("--transport");
            if (result.server) {
                usage_error("runtime transport was specified more than once");
            }
            if (transport != "stdio" && transport != "http") {
                usage_error("--transport must be stdio or http");
            }
            result.server = true;
            result.stdio = transport == "stdio";
        } else if (value == "--server") {
            if (result.server) {
                usage_error("runtime transport was specified more than once");
            }
            result.server = true;
        } else if (value == "--stdio") {
            if (result.server) {
                usage_error("runtime transport was specified more than once");
            }
            result.server = true;
            result.stdio = true;
        } else if (value == "--metal-predequantize-f16") {
            result.predequantize_fp16 = true;
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
        } else if (value == "--yarn-context-extension") {
            result.yarn_context_extension = true;
        } else if (value == "--prefill-chunk-size") {
            const auto parsed = std::stoll(
                require_value("--prefill-chunk-size"));
            if (parsed <= 0 ||
                parsed > std::numeric_limits<int>::max()) {
                usage_error(
                    "--prefill-chunk-size must be a positive integer");
            }
            result.prefill_chunk_size =
                static_cast<int>(parsed);
            result.prefill_chunk_size_explicit = true;
        } else if (value == "--moe-gpu-cache-gb") {
            const auto text =
                require_value(value.c_str());
            std::size_t consumed = 0;
            const auto parsed =
                std::stod(text, &consumed);
            if (consumed != text.size() ||
                !std::isfinite(parsed) ||
                parsed < 0.0) {
                usage_error(
                    value + " must be a finite "
                    "non-negative number");
            }
            result.expert_cache_gb = parsed;
        } else if (value == "--model-name") {
            result.model_name = require_value("--model-name");
        } else if (value == "--api-key") {
            result.api_key = require_value("--api-key");
        } else if (value == "--tokenizer") {
            result.tokenizer_gguf =
                require_value("--tokenizer");
        } else if (value == "--sampling-profile") {
            result.sampling_profile =
                require_value("--sampling-profile");
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
        << "  mfq-decode-metal --model MODEL.mfq --check-mfq-container\n"
        << "  mfq-decode-metal --model MODEL.mfq --list-tensors\n"
        << "  mfq-decode-metal --model MODEL.mfq --tensor NAME\n"
        << "  mfq-decode-metal --model MODEL.mfq --server "
           "[--host 127.0.0.1 --port 8080]\n"
        << "  mfq-decode-metal --model MODEL.mfq --transport stdio\n"
        << "  mfq-decode-metal --model HF_MODEL_DIR --server "
           "--tokenizer TOKENIZER.gguf\n"
        << "  mfq-decode-metal --self-test-metal\n\n"
        << "Options:\n"
        << "  --model PATH           MFQ model/shard, or an HF Safetensors directory\n"
        << "  --check-mfq-container  validate headers, records, and shard set\n"
        << "  --list-tensors         print record dtype, bytes, and name\n"
        << "  --tensor NAME          load and execute one supported linear weight\n"
        << "  --benchmark-mfq-moe PREFIX\n"
        << "                          benchmark one native MFQ Qwen MoE block\n"
        << "  --benchmark-reps N     timed benchmark executions (default 1)\n"
        << "  --benchmark-tokens N   routed input rows for --tensor (default 1)\n"
        << "  --benchmark-experts L  comma-separated MFE expert IDs\n"
        << "  --benchmark-distributed-experts N\n"
        << "                          vary N routed experts across benchmark tokens\n"
        << "  --benchmark-swiglu     fuse an even-width MFE gate/up record\n"
        << "  --benchmark-concat-tensor NAME\n"
        << "                          append an MFE projection (repeatable)\n"
        << "  --benchmark-routed-input\n"
        << "                          use distinct [tokens,routes,in] MFE inputs\n"
        << "  --benchmark-batch-size N\n"
        << "                          MFE executions per synchronization (default 1)\n"
        << "  --self-test-metal      execute an MLX C++ graph on Metal\n"
        << "  --transport TYPE       runtime communication: stdio or http\n"
        << "  --server               deprecated alias for --transport http\n"
        << "  --stdio                deprecated alias for --transport stdio\n"
        << "  --metal-predequantize-f16\n"
        << "                          expand regular weights to FP16 at load time\n"
        << "  --host ADDRESS         server bind address (default 127.0.0.1)\n"
        << "  --port PORT            server port (default 8080)\n"
        << "  --ctx-size TOKENS      runtime/API context limit (default 32768)\n"
        << "  --yarn-context-extension  permit Qwen YaRN context extension on load\n"
        << "  --prefill-chunk-size N maximum prompt chunk (portable default 2048;\n"
        << "                          capable runtimes may autotune when omitted)\n"
        << "  --continuous-batching N Flash-Next ordinary decode concurrency (default 8; 0/1 serial)\n"
        << "  --moe-gpu-cache-gb N   unified-memory hot-expert cache\n"
        << "                          MFQ default: full residency; HF default: auto\n"
        << "  --model-name NAME      API model name (default model path stem)\n"
        << "  --api-key KEY          optional bearer token\n"
        << "  --tokenizer PATH       external tokenizer GGUF when not embedded\n"
        << "  --sampling-profile P  explicit runtime sampling profile JSON\n";
}

std::size_t physical_memory_bytes() {
    std::uint64_t bytes = 0;
    std::size_t size = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) != 0 ||
        bytes == 0 ||
        bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("cannot determine physical memory size");
    }
    return static_cast<std::size_t>(bytes);
}

std::size_t server_cache_limit_bytes() {
    // MLX otherwise defaults its reusable-buffer cache to the much larger
    // process memory limit. Distinct Prefill shapes can then retain tens of
    // GiB of dead activation buffers and push a fully resident model into
    // compression or swap. Keep enough scratch for reuse while scaling the
    // allowance to the host instead of assuming a 128-GiB development Mac.
    static const auto limit = std::clamp(
        physical_memory_bytes() / 16,
        kMinimumServerCacheLimitBytes,
        kMaximumServerCacheLimitBytes);
    return limit;
}

std::size_t resident_allocator_reserve_bytes(std::size_t budget) {
    return std::min(server_cache_limit_bytes(), budget / 20);
}

std::size_t automatic_runtime_memory_budget_bytes() {
    const auto total = physical_memory_bytes();
    const auto reserve = total < 24 * kGiB ? 4 * kGiB : 6 * kGiB;
    const auto physical_ceiling = total <= reserve
        ? std::max<std::size_t>(1, total * 3 / 4)
        : total - reserve;

    std::size_t gpu_ceiling = physical_ceiling;
    try {
        const auto& info = mlx::core::device_info(
            mlx::core::Device(mlx::core::Device::gpu, 0));
        const auto found = info.find("max_recommended_working_set_size");
        if (found != info.end()) {
            if (const auto* value = std::get_if<std::size_t>(&found->second);
                value != nullptr && *value > 0) {
                gpu_ceiling = std::min(gpu_ceiling, *value);
            }
        }
    } catch (...) {
        // Older MLX builds may not expose this property. The physical-memory
        // ceiling remains a safe portable fallback.
    }

    std::size_t available_ceiling = physical_ceiling;
    vm_size_t page_size = 0;
    vm_statistics64_data_t statistics{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const auto host = mach_host_self();
    const bool have_snapshot =
        host_page_size(host, &page_size) == KERN_SUCCESS &&
        host_statistics64(
            host,
            HOST_VM_INFO64,
            reinterpret_cast<host_info64_t>(&statistics),
            &count) == KERN_SUCCESS;
    mach_port_deallocate(mach_task_self(), host);
    if (have_snapshot && page_size > 0) {
        const auto pages =
            static_cast<std::uint64_t>(statistics.free_count) +
            static_cast<std::uint64_t>(statistics.inactive_count);
        const auto available = pages >
                std::numeric_limits<std::size_t>::max() / page_size
            ? std::numeric_limits<std::size_t>::max()
            : static_cast<std::size_t>(pages) * page_size;
        available_ceiling = available <= reserve
            ? std::max<std::size_t>(1, available * 3 / 4)
            : available - reserve;
    }

    // This is deliberately a live incremental budget. It excludes active
    // memory owned by other services and leaves OS headroom, while also
    // respecting Metal's per-process recommended working set.
    return std::max<std::size_t>(
        1,
        std::min({physical_ceiling, gpu_ceiling, available_ceiling}));
}

bool prefill_autotune_enabled() noexcept {
    const char* value = std::getenv("MFQ_METAL_PREFILL_AUTOTUNE");
    if (value == nullptr) return true;
    const auto setting = std::string_view(value);
    return setting != "0" && setting != "false" && setting != "off";
}

std::optional<std::size_t> native_hf_source_bytes(
    const mfq::metal::MfqContainer& container) noexcept {
    std::size_t total = 0;
    for (const auto& path : container.source_paths()) {
        std::error_code error;
        const auto bytes = std::filesystem::file_size(path, error);
        if (error ||
            bytes > std::numeric_limits<std::size_t>::max() - total) {
            return std::nullopt;
        }
        total += static_cast<std::size_t>(bytes);
    }
    return total > 0
        ? std::optional<std::size_t>(total)
        : std::nullopt;
}

bool is_independently_streamed_record(std::string_view name) noexcept {
    constexpr std::string_view marker = ".tq_row_shard.";
    const auto shard = name.rfind(marker);
    if (shard != std::string_view::npos && shard + marker.size() < name.size()
        && std::all_of(name.begin() + shard + marker.size(), name.end(), [](char c) { return c >= '0' && c <= '9'; }))
        name = name.substr(0, shard);
    return name.ends_with(".associative_memory.embedding.weight") ||
        name.ends_with(".engram.embed.weight");
}

bool is_routed_expert_record(std::string_view name) noexcept {
    return name.find(".mlp.experts.") != std::string_view::npos ||
        name.find(".ffn_gate_exps.") != std::string_view::npos ||
        name.find(".ffn_up_exps.") != std::string_view::npos ||
        name.find(".ffn_gate_up_exps.") != std::string_view::npos ||
        name.find(".ffn_down_exps.") != std::string_view::npos;
}

std::size_t record_payload_bytes(
    const mfq::metal::MfqContainer& container,
    bool (*selected)(std::string_view)) noexcept {
    std::size_t total = 0;
    for (const auto& [name, record] : container.records()) {
        if (!selected(name)) continue;
        if (record.nbytes >
            std::numeric_limits<std::size_t>::max() - total) {
            return std::numeric_limits<std::size_t>::max();
        }
        total += static_cast<std::size_t>(record.nbytes);
    }
    return total;
}

std::size_t requested_cache_bytes(
    const std::optional<double>& cache_gb,
    bool hf_streaming,
    const mfq::metal::MfqContainer& container) {
    constexpr long double bytes_per_gib =
        static_cast<long double>(std::uint64_t{1} << 30);
    if (cache_gb.has_value()) {
        const long double requested =
            static_cast<long double>(*cache_gb) * bytes_per_gib;
        if (requested >
            static_cast<long double>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(
                "MoE expert cache exceeds addressable memory");
        }
        const auto bytes = static_cast<std::size_t>(requested);
        // An explicit zero keeps native HF expert banks fully resident.
        return bytes;
    }
    if (!hf_streaming) {
        return 0;
    }
    // Raw-HF is not synonymous with SSD streaming. Direct native-server
    // launches must make the same full-residency decision as the model pool;
    // otherwise a checkpoint which already fits can allocate an unnecessary
    // expert arena on top of its ordinary resident tensors.
    const auto source_bytes = native_hf_source_bytes(container);
    const auto available_bytes = automatic_runtime_memory_budget_bytes();
    const auto independently_streamed_bytes = record_payload_bytes(
        container, is_independently_streamed_record);
    const auto resident_source_bytes = source_bytes.has_value()
        ? *source_bytes - std::min(*source_bytes, independently_streamed_bytes)
        : std::numeric_limits<std::size_t>::max();
    if (source_bytes.has_value() && resident_source_bytes <= available_bytes) {
        return 0;
    }
    const auto expert_bytes = record_payload_bytes(
        container, is_routed_expert_record);
    if (source_bytes.has_value() && expert_bytes > 0) {
        const auto resident_expert_bytes = std::min(
            expert_bytes, resident_source_bytes);
        const auto dense_bytes = resident_source_bytes - resident_expert_bytes;
        const auto runtime_headroom = std::clamp(
            available_bytes / 20,
            kGiB,
            4 * kGiB);
        const auto cache_bytes = available_bytes >
                dense_bytes + runtime_headroom
            ? std::min(
                  resident_expert_bytes,
                  available_bytes - dense_bytes - runtime_headroom)
            : 0;
        return std::max(
            std::min(kGiB, resident_expert_bytes),
            cache_bytes);
    }
    return std::max<std::size_t>(
        kGiB,
        std::min(available_bytes, physical_memory_bytes() * 2 / 3));
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
            // Set residency before loading weights: otherwise macOS can
            // compress early layers while later layers are being loaded.
            // This is a ceiling, not an allocation or a system-wide change.
            try {
                const auto& info = mlx::core::device_info();
                const auto limit = std::get<std::size_t>(
                    info.at("max_recommended_working_set_size"));
                if (limit == 0) {
                    throw std::runtime_error("device returned a zero limit");
                }
                mfq::metal::MlxMemoryResidency::configure(limit);
                std::cerr << "Metal wired memory limit: " << limit
                          << " bytes (device recommendation)\n";
            } catch (const std::exception& error) {
                if (__builtin_available(macOS 15, *)) {
                    throw;
                }
                std::cerr << "Warning: Metal memory residency unavailable: "
                          << error.what() << "\n";
            }
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

int benchmark_native_mfq_moe(
    const Arguments& arguments,
    const mfq::metal::MfqContainer& model) {
    if (!arguments.tensor.empty() || arguments.server) {
        usage_error(
            "--benchmark-mfq-moe cannot be combined with --tensor or "
            "--server");
    }
    if (arguments.benchmark_tokens != 1) {
        usage_error("--benchmark-mfq-moe requires --benchmark-tokens 1");
    }
    if (arguments.benchmark_distributed_routes != 0 ||
        arguments.benchmark_swiglu ||
        arguments.benchmark_routed_input ||
        !arguments.benchmark_concat_tensors.empty()) {
        usage_error(
            "single-tensor MFE benchmark options cannot be combined with "
            "--benchmark-mfq-moe");
    }

    const auto name = [&](std::string_view suffix) {
        return arguments.benchmark_mfq_moe_prefix + std::string(suffix);
    };
    const auto gate_name = name(".experts.gate.weight");
    const auto up_name = name(".experts.up.weight");
    const auto down_name = name(".experts.down.weight");
    for (const auto* record_name : {&gate_name, &up_name, &down_name}) {
        if (model.record(*record_name).dtype != "MFE") {
            usage_error(
                "--benchmark-mfq-moe requires native MFE expert records");
        }
    }

    auto gate_up = mfq::metal::MlxMfeWeight::concatenate_projections({
        mfq::metal::MlxMfeWeight::from_blob(model.read(gate_name)),
        mfq::metal::MlxMfeWeight::from_blob(model.read(up_name)),
    }).materialize_packed_projections();
    auto down = mfq::metal::MlxMfeWeight::from_blob(
        model.read(down_name));

    const auto shared_gate = mfq::metal::MlxLinear::load(
        model, name(".shared_expert.gate.weight"));
    const auto shared_up = mfq::metal::MlxLinear::load(
        model, name(".shared_expert.up.weight"));
    const auto shared_down = mfq::metal::MlxLinear::load(
        model, name(".shared_expert.down.weight"));
    const auto shared_router = mfq::metal::MlxLinear::load(
        model, name(".shared_expert.router.weight"));
    const auto* shared_gate_nint = shared_gate.nint_weight_ref();
    const auto* shared_up_nint = shared_up.nint_weight_ref();
    const auto* shared_down_nint = shared_down.nint_weight_ref();
    const auto* shared_router_dense = shared_router.dense_weight_ref();
    if (shared_gate_nint == nullptr || shared_up_nint == nullptr ||
        shared_down_nint == nullptr || shared_router_dense == nullptr) {
        usage_error(
            "--benchmark-mfq-moe requires NINT shared Gate/Up/Down and a "
            "dense shared router");
    }
    const auto shared_gate_up =
        mfq::metal::MlxNintSwiGluPair::from_weights(
            *shared_gate_nint, *shared_up_nint);
    if (!shared_gate_up.has_value()) {
        throw std::runtime_error(
            "MFQ shared-expert NINT Gate/Up pair is not fuseable");
    }

    std::vector<std::int32_t> expert_values =
        arguments.benchmark_experts;
    if (expert_values.empty()) {
        const int routes = std::min(10, gate_up.experts());
        expert_values.resize(static_cast<std::size_t>(routes));
        for (int route = 0; route < routes; ++route) {
            expert_values[static_cast<std::size_t>(route)] = route;
        }
    }
    if (expert_values.size() > 16) {
        usage_error("--benchmark-mfq-moe supports at most 16 routes");
    }
    for (const auto expert : expert_values) {
        if (expert < 0 || expert >= gate_up.experts()) {
            usage_error(
                "--benchmark-experts contains an out-of-range expert ID");
        }
    }
    const int routes = static_cast<int>(expert_values.size());
    std::vector<float> route_values(static_cast<std::size_t>(routes));
    float route_total = 0.0f;
    for (int route = 0; route < routes; ++route) {
        route_values[static_cast<std::size_t>(route)] =
            static_cast<float>(route + 1);
        route_total += static_cast<float>(route + 1);
    }
    for (auto& value : route_values) value /= route_total;

    std::vector<float> input_values(
        static_cast<std::size_t>(gate_up.neuron_len()));
    for (std::size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = static_cast<float>(
            static_cast<int>(index % 23) - 11) / 32.0f;
    }
    const auto input = mlx::core::astype(
        mlx::core::array(
            input_values.begin(),
            mlx::core::Shape{1, gate_up.neuron_len()}),
        mlx::core::float16);
    const auto expert_ids = mlx::core::array(
        expert_values.begin(), mlx::core::Shape{1, routes});
    const auto route_weights = mlx::core::array(
        route_values.begin(), mlx::core::Shape{1, routes});

    auto execute = [&]() -> mlx::core::array {
        auto output = gate_up.decode_nint_shared(
            down,
            *shared_gate_up,
            *shared_down_nint,
            *shared_router_dense,
            input,
            expert_ids,
            route_weights);
        if (!output.has_value()) {
            throw std::runtime_error(
                "native MFQ two-stage MoE decode path is unsupported for "
                "this block");
        }
        return std::move(*output);
    };

    auto output = execute();
    output.eval();
    const auto started = std::chrono::steady_clock::now();
    std::vector<mlx::core::array> pending;
    pending.reserve(static_cast<std::size_t>(std::min(
        arguments.benchmark_batch_size, arguments.benchmark_reps)));
    for (int rep = 0; rep < arguments.benchmark_reps; ++rep) {
        output = execute();
        pending.push_back(output);
        if (pending.size() ==
                static_cast<std::size_t>(arguments.benchmark_batch_size) ||
            rep + 1 == arguments.benchmark_reps) {
            mlx::core::eval(pending);
            pending.clear();
        }
    }
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started)
            .count();

    auto checked = mlx::core::astype(output, mlx::core::float32);
    checked.eval();
    const auto* values = checked.data<float>();
    double checksum = 0.0;
    for (std::size_t index = 0; index < checked.size(); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::runtime_error(
                "native MFQ MoE benchmark returned non-finite data");
        }
        checksum += static_cast<double>(values[index]) *
            static_cast<double>(index % 251u + 1u);
    }
    std::cout
        << "Native MFQ two-stage MoE benchmark passed"
        << " prefix=" << arguments.benchmark_mfq_moe_prefix
        << " experts=" << gate_up.experts()
        << " routes=" << routes
        << " hidden=" << gate_up.neuron_len()
        << " intermediate=" << gate_up.out_per_expert()
        << " packed=" << gate_up.packed_nbytes() + down.packed_nbytes()
        << " reps=" << arguments.benchmark_reps
        << " batch_size=" << arguments.benchmark_batch_size
        << " ms_per_decode=" << elapsed_ms / arguments.benchmark_reps
        << " checksum=" << checksum
        << "\n";
    return EXIT_SUCCESS;
}

#ifdef MFQ_METAL_RUNTIME_COMMUNICATION
std::uint64_t cache_bytes_from_environment(
    const char* name,
    std::uint64_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return parsed;
}

std::filesystem::path default_prefix_cache_directory() {
    if (const char* configured =
            std::getenv("MFQ_SERVER_PREFIX_CACHE_DIR")) {
        if (configured[0] != '\0') return configured;
    }
    if (const char* home = std::getenv("HOME")) {
        if (home[0] != '\0') {
            return std::filesystem::path(home) / "Library" / "Caches" /
                "TyloQuant" / "MFQ" / "prefix-cache";
        }
    }
    return std::filesystem::temp_directory_path() /
        "tyloquant-mfq-prefix-cache";
}

std::string metal_prefix_cache_compatibility_key(
    const mfq::metal::MfqContainer& container,
    std::string_view codec,
    std::int64_t context_size) {
    std::vector<const mfq::metal::MfqRecord*> records;
    records.reserve(container.records().size());
    for (const auto& [name, record] : container.records()) {
        (void)name;
        records.push_back(&record);
    }
    std::sort(
        records.begin(), records.end(),
        [](const auto* left, const auto* right) {
            return left->name < right->name;
        });

    std::ostringstream key;
    key << "mfq-metal-prefix-v1\n"
        << "codec=" << codec << '\n'
        << "context=" << context_size << '\n'
        << "header=" << container.header().version << '\n'
        << "architecture=" << container.header().architecture << '\n';
    for (std::size_t index = 0;
         index < container.source_paths().size(); ++index) {
        const auto& path = container.source_paths()[index];
        std::error_code error;
        const auto size = std::filesystem::file_size(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify MFQ shard for prefix cache: " +
                error.message());
        }
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error) {
            throw std::runtime_error(
                "cannot identify MFQ shard timestamp: " +
                error.message());
        }
        const auto modified_ns = std::chrono::duration_cast<
            std::chrono::nanoseconds>(modified.time_since_epoch()).count();
        key << "shard=" << index << ':' << path.filename().string() << ':'
            << size << ':' << modified_ns << '\n';
    }
    for (const auto* record : records) {
        key << "tensor=" << record->name << ':' << record->dtype << ':'
            << record->offset << ':' << record->nbytes << '\n';
    }
    return key.str();
}

bool metal_prefix_nax_device() {
    try {
        const auto& info = mlx::core::device_info();
        const auto found = info.find("architecture");
        if (found != info.end()) if (const auto* architecture = std::get_if<std::string>(&found->second))
            return mfq::metal::mlx_prefix_nax_architecture(*architecture);
    } catch (const std::exception&) {
    }
    return false;
}

std::string metal_media_cache_key(const MfqMultimodalInput& media) {
    std::ostringstream key;
    key << "vision-prefix-v1:" << media.processor_name << ':';
    const auto append = [&](const auto& values) {
        key << mfq::cache::block_hash_hex(mfq::cache::sha256(values.data(),
            values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type))) << ':';
    };
    append(media.pixel_values);
    append(media.pixel_shape);
    append(media.vision_grid);
    append(media.image_bounds);
    return key.str();
}

template <typename Runtime>
std::shared_ptr<mfq::cache::PagedPrefixCache> make_metal_paged_cache(
    const mfq::metal::MfqContainer& container,
    std::int64_t context_size,
    int prefill_chunk_size,
    const std::string& predictor_key = {}) {
    using SessionState = decltype(
        std::declval<const Runtime&>().capture_text_session_state(
            std::declval<const std::vector<std::int64_t>&>()));
    using Codec = mfq::metal::MlxPagedSessionCodec<SessionState>;
    if constexpr (!Codec::available) {
        (void)container;
        (void)context_size;
        (void)prefill_chunk_size;
        return {};
    } else {
        if (const char* disabled =
                std::getenv("MFQ_SERVER_DISABLE_PREFIX_CACHE")) {
            if (disabled[0] == '1') return {};
        }
        constexpr bool recurrent = std::is_same_v<Runtime, mfq::metal::MlxQwen35CausalLm> ||
            std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm>;
        const auto default_block = mfq::metal::mlx_prefix_block_size(
            recurrent || std::is_same_v<Runtime, mfq::metal::MlxDeepseekV41CausalLm>, prefill_chunk_size,
            physical_memory_bytes(), metal_prefix_nax_device(), std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm>);
        const auto block_size = cache_bytes_from_environment(
            "MFQ_SERVER_PREFIX_CACHE_BLOCK_TOKENS", default_block);
        if (block_size == 0 || block_size > 65536) {
            throw std::runtime_error(
                "MFQ_SERVER_PREFIX_CACHE_BLOCK_TOKENS must be in [1, 65536]");
        }
        mfq::cache::PagedPrefixCacheConfig config;
        config.reserve_memory = [](std::size_t bytes) { mfq::metal::MlxResidentBudgetScope::reserve(bytes); };
        config.cache_dir = default_prefix_cache_directory();
        if (!container.source_paths().empty()) config.model_path = container.source_paths().front();
        config.compatibility_key = metal_prefix_cache_compatibility_key(
            container, Codec::name, context_size);
        if constexpr (recurrent) config.compatibility_key += mfq::metal::mlx_kv_quantization_tag();
        if constexpr (recurrent) {
            const auto native_context = [&] {
                if constexpr (std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm>)
                    return mfq::metal::Qwen4Config::from_mfq(container).max_position_embeddings;
                else return mfq::metal::Qwen35Config::from_mfq(container).max_position_embeddings;
            }();
            if (context_size > native_context) config.compatibility_key += "\nqwen-yarn-v1";
        }
        if (!predictor_key.empty()) config.compatibility_key += "\n" + predictor_key;
        config.block_size_tokens = static_cast<std::size_t>(block_size);
        config.max_disk_bytes = cache_bytes_from_environment(
            "MFQ_SERVER_PREFIX_CACHE_DISK_BYTES",
            100ULL * 1024ULL * 1024ULL * 1024ULL);
        config.max_hot_bytes = cache_bytes_from_environment(
            "MFQ_SERVER_PREFIX_CACHE_HOT_BYTES",
            2ULL * 1024ULL * 1024ULL * 1024ULL);
        config.max_pending_writes = static_cast<std::size_t>(
            cache_bytes_from_environment(
                "MFQ_SERVER_PREFIX_CACHE_PENDING_WRITES", 64));
        config.max_pending_bytes = cache_bytes_from_environment(
            "MFQ_SERVER_PREFIX_CACHE_PENDING_BYTES",
            512ULL * 1024ULL * 1024ULL);
        return std::make_shared<mfq::cache::PagedPrefixCache>(
            std::move(config));
    }
}

template <typename Runtime>
class MlxServerTextSessionCache {
private:
    using SessionState = decltype(
        std::declval<const Runtime&>().capture_text_session_state(
            std::declval<const std::vector<std::int64_t>&>()));
    using Codec = mfq::metal::MlxPagedSessionCodec<SessionState>;

    struct Entry {
        SessionState state;
        std::uint64_t last_used = 0;
        std::string media_key;
    };

    struct PagedBinding {
        std::vector<mfq::cache::BlockHash> blocks;
        std::size_t tokens = 0;
        std::uint64_t last_used = 0;
    };

public:
    explicit MlxServerTextSessionCache(
        std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache = {})
        : paged_cache_(std::move(paged_cache)) {
        if (const char* value =
                std::getenv("MFQ_SERVER_MAX_KV_SESSIONS")) {
            max_sessions_ = static_cast<std::size_t>(
                std::strtoull(value, nullptr, 10));
        }
        if (const char* value = std::getenv(
                "MFQ_SERVER_MAX_KV_SNAPSHOTS_PER_SESSION")) {
            max_snapshots_per_session_ = static_cast<std::size_t>(
                std::strtoull(value, nullptr, 10));
        }
        if (const char* value =
                std::getenv("MFQ_SERVER_KV_SESSION_BYTES")) {
            max_bytes_ = static_cast<std::size_t>(
                std::strtoull(value, nullptr, 10));
        }
        if (const char* value =
                std::getenv("MFQ_SERVER_TRACE_SESSION_CACHE")) {
            trace_ = value[0] == '1';
        }
        if (paged_cache_) {
            paged_disk_budget_ = cache_bytes_from_environment(
                "MFQ_SERVER_PREFIX_CACHE_DISK_BYTES",
                100ULL * 1024ULL * 1024ULL * 1024ULL);
            paged_hot_budget_ = cache_bytes_from_environment(
                "MFQ_SERVER_PREFIX_CACHE_HOT_BYTES",
                2ULL * 1024ULL * 1024ULL * 1024ULL);
        }
    }

    bool persistent_prefix_enabled() const noexcept {
        if constexpr (Codec::available) {
            return static_cast<bool>(paged_cache_);
        }
        return false;
    }

    std::size_t block_size_tokens() const noexcept {
        return paged_cache_ ? paged_cache_->block_size_tokens() : 256;
    }

    std::size_t normalize_stable_prefix_tokens(
        std::size_t requested) const noexcept {
        if constexpr (requires {
                Codec::requires_exact_block_boundary;
            }) {
            if (Codec::requires_exact_block_boundary && paged_cache_) {
                const auto block_size = paged_cache_->block_size_tokens();
                return requested / block_size * block_size;
            }
        }
        return requested;
    }

    std::size_t restore_best(
        Runtime& runtime,
        const std::string& requested_session,
        const std::vector<std::int64_t>& prompt,
        std::size_t maximum_prefix_tokens,
        bool require_mtp = false, std::string_view media_key = {}) {
        if constexpr (Codec::available) {
            if (paged_cache_) {
                return restore_paged(
                    runtime,
                    requested_session,
                    prompt,
                    maximum_prefix_tokens, require_mtp, media_key);
            }
        }
        if (requested_session.empty() || max_sessions_ == 0 ||
            max_snapshots_per_session_ == 0 || max_bytes_ == 0 ||
            !runtime.supports_text_session_state()) {
            return 0;
        }
        ++queries_;
        std::string selected_session;
        std::size_t selected_snapshot = 0;
        std::size_t selected_tokens = 0;
        for (const auto& [session_id, history] : states_) {
            for (std::size_t index = 0; index < history.size(); ++index) {
                const auto& tokens = history[index].state.tokens;
                if (history[index].media_key != media_key || tokens.empty() || tokens.size() >= prompt.size() ||
                    tokens.size() > maximum_prefix_tokens ||
                    tokens.size() < selected_tokens ||
                    !std::equal(
                        tokens.begin(), tokens.end(), prompt.begin())) {
                    continue;
                }
                const bool requested_tie =
                    tokens.size() == selected_tokens &&
                    session_id == requested_session &&
                    selected_session != requested_session;
                if (tokens.size() > selected_tokens || requested_tie) {
                    selected_session = session_id;
                    selected_snapshot = index;
                    selected_tokens = tokens.size();
                }
            }
        }
        if (selected_session.empty()) return 0;
        auto& selected = states_.at(selected_session)[selected_snapshot];
        try {
            runtime.restore_text_session_state(selected.state);
            selected.last_used = ++clock_;
            ++hits_;
            hit_tokens_ += selected_tokens;
            if (trace_) {
                std::cerr
                    << "server_session_cache backend=metal action=hit session="
                    << requested_session << " source=" << selected_session
                    << " reused_tokens=" << selected_tokens
                    << " prefill_tokens="
                    << prompt.size() - selected_tokens << std::endl;
            }
            return selected_tokens;
        } catch (const std::exception& error) {
            erase_snapshot(
                selected_session, selected_snapshot, "invalidate");
            reset_runtime(runtime);
            std::cerr
                << "server_session_cache backend=metal action=invalidate "
                << "session=" << selected_session
                << " error=" << error.what() << std::endl;
            return 0;
        }
    }

    void store(const std::string& session_id, SessionState state, std::string_view media_key = {}) {
        if constexpr (Codec::available) {
            if (paged_cache_) {
                store_paged(session_id, state, media_key);
                return;
            }
        }
        if (session_id.empty() || max_sessions_ == 0 ||
            max_snapshots_per_session_ == 0 || max_bytes_ == 0) {
            return;
        }
        if (state.bytes > max_bytes_) {
            if (trace_) {
                std::cerr
                    << "server_session_cache backend=metal action=skip session="
                    << session_id << " bytes=" << state.bytes
                    << " budget=" << max_bytes_ << std::endl;
            }
            return;
        }
        Entry entry{std::move(state), ++clock_, std::string(media_key)};
        const auto protected_clock = entry.last_used;
        auto& history = states_[session_id];
        auto previous = std::find_if(
            history.begin(), history.end(),
            [&](const Entry& saved) {
                return saved.media_key == entry.media_key && saved.state.tokens == entry.state.tokens;
            });
        if (previous != history.end()) {
            bytes_ -= previous->state.bytes;
            *previous = std::move(entry);
        } else {
            history.push_back(std::move(entry));
        }
        const auto stored = std::find_if(
            history.begin(), history.end(),
            [&](const Entry& saved) {
                return saved.last_used == protected_clock;
            });
        if (stored == history.end()) {
            throw std::runtime_error(
                "stored Metal session snapshot is unavailable");
        }
        bytes_ += stored->state.bytes;
        evict_history_to_limit(session_id, protected_clock);
        evict_to_budget(session_id, protected_clock);
        sync_telemetry();
        if (trace_) {
            const auto& saved_history = states_.at(session_id);
            const auto saved = std::find_if(
                saved_history.begin(), saved_history.end(),
                [&](const Entry& candidate) {
                    return candidate.last_used == protected_clock;
                });
            if (saved == saved_history.end()) {
                throw std::runtime_error(
                    "protected Metal session snapshot was evicted");
            }
            std::cerr
                << "server_session_cache backend=metal action=store session="
                << session_id << " tokens=" << saved->state.tokens.size()
                << " bytes=" << saved->state.bytes
                << " snapshots=" << saved_history.size()
                << " total_bytes=" << bytes_ << std::endl;
        }
    }

    std::size_t fork_session(
        const std::string& source_session,
        const std::string& target_session) {
        if constexpr (Codec::available) {
            if (paged_cache_) {
                const auto source = paged_bindings_.find(source_session);
                if (source == paged_bindings_.end() ||
                    source_session.empty() || target_session.empty() ||
                    source_session == target_session) {
                    return 0;
                }
                bind_paged_session(
                    target_session,
                    source->second.blocks,
                    source->second.tokens);
                return source->second.blocks.size();
            }
        }
        if (source_session.empty() || target_session.empty() ||
            source_session == target_session || max_sessions_ == 0 ||
            max_snapshots_per_session_ == 0 || max_bytes_ == 0) {
            return 0;
        }
        const auto source = states_.find(source_session);
        if (source == states_.end()) return 0;
        auto copied = source->second;
        close_session(target_session);
        auto& target = states_[target_session];
        std::uint64_t protected_clock = 0;
        for (auto& snapshot : copied) {
            snapshot.last_used = ++clock_;
            protected_clock = snapshot.last_used;
            bytes_ += snapshot.state.bytes;
            target.push_back(std::move(snapshot));
        }
        evict_history_to_limit(target_session, protected_clock);
        evict_to_budget(target_session, protected_clock);
        const auto remaining = states_.find(target_session);
        const std::size_t copied_snapshots = remaining == states_.end()
            ? 0 : remaining->second.size();
        sync_telemetry();
        if (trace_) {
            std::cerr
                << "server_session_cache backend=metal action=fork source="
                << source_session << " target=" << target_session
                << " snapshots=" << copied_snapshots
                << " total_bytes=" << bytes_ << std::endl;
        }
        return copied_snapshots;
    }

    std::size_t close_session(const std::string& session_id) {
        if constexpr (Codec::available) {
            if (paged_cache_) return close_paged_session(session_id);
        }
        auto found = states_.find(session_id);
        if (found == states_.end()) return 0;
        const std::size_t released = found->second.size();
        std::size_t released_bytes = 0;
        for (const auto& snapshot : found->second) {
            released_bytes += snapshot.state.bytes;
        }
        bytes_ -= released_bytes;
        states_.erase(found);
        sync_telemetry();
        if (trace_) {
            std::cerr
                << "server_session_cache backend=metal action=close session="
                << session_id << " snapshots=" << released
                << " bytes=" << released_bytes
                << " total_bytes=" << bytes_ << std::endl;
        }
        return released;
    }

    std::vector<std::pair<std::string, double>> metrics() const {
        std::lock_guard control_lock(paged_control_mutex_);
        if constexpr (Codec::available) {
            if (paged_cache_) {
                const auto value = paged_cache_->metrics();
                return {
                    {"prefix_cache_queries", static_cast<double>(value.queries)},
                    {"prefix_cache_hits", static_cast<double>(value.hits)},
                    {"prefix_cache_hit_tokens", static_cast<double>(value.hit_tokens)},
                    {"prefix_cache_block_tokens", static_cast<double>(paged_cache_->block_size_tokens())},
                    {"prefix_cache_sessions", static_cast<double>(metric_sessions_.load())},
                    {"prefix_cache_snapshots", static_cast<double>(value.disk_blocks)},
                    {"prefix_cache_tokens", static_cast<double>(metric_tokens_.load())},
                    {"prefix_cache_bytes", static_cast<double>(value.hot_bytes)},
                    {"prefix_cache_max_sessions", static_cast<double>(max_sessions_)},
                    {"prefix_cache_max_snapshots_per_session", 1.0},
                    {"prefix_cache_max_bytes", static_cast<double>(paged_hot_budget_)},
                    {"prefix_cache_disk_blocks", static_cast<double>(value.disk_blocks)},
                    {"prefix_cache_disk_bytes", static_cast<double>(value.disk_bytes)},
                    {"prefix_cache_disk_max_bytes", static_cast<double>(paged_disk_budget_)},
                    {"prefix_cache_hot_blocks", static_cast<double>(value.hot_blocks)},
                    {"prefix_cache_hot_bytes", static_cast<double>(value.hot_bytes)},
                    {"prefix_cache_resident_bytes", static_cast<double>(value.resident_bytes)},
                    {"prefix_cache_hot_pressure_bytes", static_cast<double>(value.hot_pressure_bytes)},
                    {"prefix_cache_dynamic_budget", 1.0},
                    {"prefix_cache_concurrent_maintenance", 1.0},
                    {"prefix_cache_low_disk_space_skips", static_cast<double>(value.low_disk_space_skips)},
                    {"prefix_cache_failed_writes", static_cast<double>(value.failed_writes)},
                    {"prefix_cache_pending_writes", static_cast<double>(value.pending_writes)},
                    {"prefix_cache_pending_bytes", static_cast<double>(value.pending_bytes)},
                    {"prefix_cache_pending_max_bytes", static_cast<double>(value.pending_max_bytes)},
                    {"prefix_cache_writes", static_cast<double>(value.writes)},
                    {"prefix_cache_deduplicated_writes", static_cast<double>(value.deduplicated_writes)},
                    {"prefix_cache_disk_hits", static_cast<double>(value.disk_hits)},
                    {"prefix_cache_hot_hits", static_cast<double>(value.hot_hits)},
                    {"prefix_cache_evictions", static_cast<double>(value.evictions)},
                    {"prefix_cache_corrupt_blocks", static_cast<double>(value.corrupt_blocks)},
                };
            }
        }
        return {
            {"prefix_cache_queries", static_cast<double>(queries_.load())},
            {"prefix_cache_hits", static_cast<double>(hits_.load())},
            {"prefix_cache_hit_tokens", static_cast<double>(hit_tokens_.load())},
            {"prefix_cache_sessions", static_cast<double>(metric_sessions_.load())},
            {"prefix_cache_snapshots", static_cast<double>(metric_snapshots_.load())},
            {"prefix_cache_tokens", static_cast<double>(metric_tokens_.load())},
            {"prefix_cache_bytes", static_cast<double>(metric_bytes_.load())},
            {"prefix_cache_max_sessions", static_cast<double>(max_sessions_)},
            {"prefix_cache_max_snapshots_per_session",
                static_cast<double>(max_snapshots_per_session_)},
            {"prefix_cache_max_bytes", static_cast<double>(max_bytes_)},
        };
    }

    std::size_t clear_live_sessions() noexcept {
        if constexpr (Codec::available) {
            if (paged_cache_) {
                const auto sessions = paged_bindings_.size();
                paged_bindings_.clear();
                sync_paged_telemetry();
                return sessions;
            }
        }
        std::size_t snapshots = 0;
        for (const auto& [session_id, history] : states_) {
            (void)session_id;
            snapshots += history.size();
        }
        states_.clear();
        bytes_ = 0;
        sync_telemetry();
        return snapshots;
    }

    std::size_t clear() {
        if constexpr (Codec::available) {
            if (paged_cache_) {
                clear_live_sessions();
                return paged_cache_->clear();
            }
        }
        return clear_live_sessions();
    }

    std::uint64_t trim_hot(std::uint64_t target_bytes) {
        std::lock_guard control_lock(paged_control_mutex_);
        if constexpr (Codec::available) {
            if (paged_cache_) {
                const auto released = paged_cache_->trim_hot(target_bytes);
                if (released > 0) release_host_allocator_cache();
                return released;
            }
        }
        return 0;
    }

    std::uint64_t set_hot_limit(std::uint64_t max_bytes) {
        std::lock_guard control_lock(paged_control_mutex_);
        if (!paged_cache_) return 0;
        paged_hot_budget_ = max_bytes;
        const auto released = paged_cache_->set_hot_limit(max_bytes);
        if (released > 0) release_host_allocator_cache();
        return released;
    }

    std::uint64_t set_disk_limit(std::uint64_t max_bytes) {
        std::lock_guard control_lock(paged_control_mutex_);
        if (!paged_cache_) return 0;
        paged_disk_budget_ = max_bytes;
        return paged_cache_->set_disk_limit(max_bytes);
    }

    std::uint64_t refresh_disk_index() {
        if (!paged_cache_) return 0;
        const auto removed = paged_cache_->refresh_disk_index();
        if (removed > 0) release_host_allocator_cache();
        return removed;
    }

    void replace_paged_cache(
        std::shared_ptr<mfq::cache::PagedPrefixCache> cache,
        std::uint64_t disk_budget,
        std::uint64_t hot_budget) {
        clear_live_sessions();
        {
            std::lock_guard control_lock(paged_control_mutex_);
            paged_cache_.swap(cache);
            paged_disk_budget_ = disk_budget;
            paged_hot_budget_ = hot_budget;
        }
        sync_paged_telemetry();
    }

private:
    static void reset_runtime(Runtime& runtime) {
        if constexpr (requires { runtime.reset_cache(1); }) {
            runtime.reset_cache(1);
        } else if constexpr (requires { runtime.reset(); }) {
            runtime.reset();
        }
    }

    std::size_t restore_paged(
        Runtime& runtime,
        const std::string& requested_session,
        const std::vector<std::int64_t>& prompt,
        std::size_t maximum_prefix_tokens,
        bool require_mtp, std::string_view media_key) {
        if (max_sessions_ == 0 || !runtime.supports_text_session_state() ||
            prompt.size() < 2) {
            return 0;
        }
        const auto limit = std::min(
            maximum_prefix_tokens, prompt.size() - 1);
        std::vector<std::int64_t> candidate(
            prompt.begin(),
            prompt.begin() + static_cast<std::ptrdiff_t>(limit));
        auto match = paged_cache_->match(candidate, media_key, false);
        if (match.matched_tokens == 0) {
            paged_cache_->record_match(0);
            return 0;
        }

        auto payloads = paged_cache_->load_prefix(match.blocks);
        if (payloads.empty()) {
            paged_cache_->record_match(0);
            return 0;
        }
        std::size_t payload_bytes = 0;
        for (const auto& payload : payloads) payload_bytes += payload->size();
        mfq::metal::MlxResidentBudgetHold payload_budget(payload_bytes);
        if (payloads.size() != match.blocks.size()) {
            match.blocks.resize(payloads.size());
            if constexpr (requires { Codec::token_count(payloads.back()); })
                match.matched_tokens = Codec::token_count(payloads.back());
            else match.matched_tokens = payloads.size() * paged_cache_->block_size_tokens();
        }
        std::optional<SessionState> decoded;
        try {
            if constexpr (requires {
                    Codec::decodable_blocks(payloads);
                }) {
                auto decodable = Codec::decodable_blocks(payloads);
                if constexpr (requires { Codec::has_mtp(payloads.back()); }) {
                    if (require_mtp) while (decodable > 0 &&
                        (!Codec::has_exact_boundary(payloads[decodable - 1]) ||
                         !Codec::has_mtp(payloads[decodable - 1]))) --decodable;
                }
                if (decodable == 0) {
                    paged_cache_->record_match(0);
                    return 0;
                }
                if (decodable < payloads.size()) {
                    payloads.resize(decodable);
                    match.blocks.resize(decodable);
                    if constexpr (requires { Codec::token_count(payloads.back()); })
                        match.matched_tokens = Codec::token_count(payloads.back());
                    else match.matched_tokens = decodable * paged_cache_->block_size_tokens();
                }
            }
            std::vector<std::int64_t> matched_tokens(
                prompt.begin(),
                prompt.begin() + static_cast<std::ptrdiff_t>(
                    match.matched_tokens));
            decoded.emplace(Codec::decode(
                payloads,
                matched_tokens,
                paged_cache_->block_size_tokens()));
        } catch (const std::exception& error) {
            if (!match.blocks.empty()) {
                paged_cache_->invalidate(match.blocks.back());
            }
            paged_cache_->record_match(0);
            reset_runtime(runtime);
            std::cerr
                << "server_session_cache backend=metal "
                << "action=paged_codec_invalidate "
                << "session=" << requested_session
                << " error=" << error.what() << std::endl;
            return 0;
        }
        try {
            runtime.restore_text_session_state(std::move(*decoded));
            if (!requested_session.empty()) {
                bind_paged_session(
                    requested_session, match.blocks, match.matched_tokens);
            }
            paged_cache_->record_match(match.matched_tokens);
            if (trace_) {
                std::cerr
                    << "server_session_cache backend=metal action=paged_hit "
                    << "session=" << requested_session
                    << " reused_tokens=" << match.matched_tokens
                    << " prefill_tokens="
                    << prompt.size() - match.matched_tokens << std::endl;
            }
            return match.matched_tokens;
        } catch (const std::exception& error) {
            paged_cache_->record_match(0);
            reset_runtime(runtime);
            std::cerr
                << "server_session_cache backend=metal action=paged_restore_failed "
                << "session=" << requested_session
                << " error=" << error.what() << std::endl;
            return 0;
        }
    }

    void store_paged(
        const std::string& session_id,
        const SessionState& state, std::string_view media_key) {
        if (max_sessions_ == 0 || state.tokens.empty()) {
            return;
        }
        const auto block_size = paged_cache_->block_size_tokens();
        if constexpr (requires {
                Codec::requires_exact_block_boundary;
            }) {
            if (Codec::requires_exact_block_boundary &&
                state.tokens.size() % block_size != 0) {
                return;
            }
        }
        constexpr bool tails = [] {
            if constexpr (requires { Codec::supports_tail_blocks; }) return Codec::supports_tail_blocks;
            else return false;
        }();
        const auto full_blocks = tails ? (state.tokens.size() + block_size - 1) / block_size
            : state.tokens.size() / block_size;
        if (full_blocks == 0) return;

        auto existing = paged_cache_->match(state.tokens, media_key, false);
        if (existing.matched_tokens % block_size != 0 &&
            existing.matched_tokens != state.tokens.size()) {
            existing.blocks.pop_back();
            existing.matched_tokens -= existing.matched_tokens % block_size;
        }
        if (existing.blocks.size() > full_blocks) {
            throw std::runtime_error(
                "paged prefix match exceeds the session state");
        }
        const auto existing_blocks = existing.blocks.size();
        auto blocks = std::move(existing.blocks);
        mfq::cache::BlockHash parent{};
        if (!blocks.empty()) parent = blocks.back();
        for (std::size_t index = blocks.size(); index < full_blocks; ++index) {
            const auto token_offset = index * block_size;
            auto payload = Codec::encode_block(
                state, block_size, index);
            parent = paged_cache_->store(
                parent,
                state.tokens.data() + token_offset,
                std::min(block_size, state.tokens.size() - token_offset),
                std::move(payload), media_key);
            blocks.push_back(parent);
        }
        if constexpr (requires {
                Codec::refresh_final_block;
                Codec::has_exact_boundary(
                    std::declval<
                        const mfq::metal::MlxPagedPayload&>());
            }) {
            if (Codec::refresh_final_block &&
                existing_blocks == full_blocks && !blocks.empty()) {
                auto final_payload = paged_cache_->load_prefix(
                    {blocks.back()});
                const bool exact = final_payload.size() == 1 &&
                    Codec::has_exact_boundary(final_payload.front());
                bool upgrade_mtp = false;
                if constexpr (requires { state.last_hidden; state.mtp_layers; Codec::has_mtp(final_payload.front()); }) {
                    upgrade_mtp = state.last_hidden && (state.cache_position == 1 || !state.mtp_layers.empty()) &&
                        (final_payload.empty() || !Codec::has_mtp(final_payload.front()));
                }
                if constexpr (requires { state.dspark; Codec::has_mtp(final_payload.front()); })
                    upgrade_mtp = state.dspark.has_value() &&
                        (final_payload.empty() || !Codec::has_mtp(final_payload.front()));
                if (!exact || upgrade_mtp) {
                    const auto index = full_blocks - 1;
                    const auto token_offset = index * block_size;
                    const mfq::cache::BlockHash checkpoint_parent =
                        index == 0
                        ? mfq::cache::BlockHash{}
                        : blocks[index - 1];
                    const auto refreshed = paged_cache_->replace(
                        checkpoint_parent,
                        state.tokens.data() + token_offset,
                        std::min(block_size, state.tokens.size() - token_offset),
                        Codec::encode_block(state, block_size, index), media_key);
                    if (refreshed != blocks[index]) {
                        throw std::runtime_error(
                            "paged prefix checkpoint changed block identity");
                    }
                }
            }
        }
        if (!session_id.empty()) {
            bind_paged_session(
                session_id,
                std::move(blocks),
                std::min(full_blocks * block_size, state.tokens.size()));
        }
        if (trace_) {
            std::cerr
                << "server_session_cache backend=metal action=paged_store "
                << "session=" << session_id
                << " tokens=" << std::min(full_blocks * block_size, state.tokens.size())
                << " blocks=" << full_blocks << std::endl;
        }
    }

    void bind_paged_session(
        const std::string& session_id,
        std::vector<mfq::cache::BlockHash> blocks,
        std::size_t tokens) {
        close_paged_session(session_id);
        paged_bindings_[session_id] = PagedBinding{
            std::move(blocks), tokens, ++clock_};
        while (paged_bindings_.size() > max_sessions_) {
            auto victim = paged_bindings_.end();
            for (auto iterator = paged_bindings_.begin();
                 iterator != paged_bindings_.end(); ++iterator) {
                if (iterator->first == session_id) continue;
                if (victim == paged_bindings_.end() ||
                    iterator->second.last_used < victim->second.last_used) {
                    victim = iterator;
                }
            }
            if (victim == paged_bindings_.end()) break;
            close_paged_session(victim->first);
        }
        sync_paged_telemetry();
    }

    std::size_t close_paged_session(const std::string& session_id) {
        auto found = paged_bindings_.find(session_id);
        if (found == paged_bindings_.end()) return 0;
        const auto blocks = found->second.blocks.size();
        paged_bindings_.erase(found);
        sync_paged_telemetry();
        return blocks;
    }

    void sync_paged_telemetry() noexcept {
        std::size_t tokens = 0;
        for (const auto& [session, binding] : paged_bindings_) {
            (void)session;
            tokens += binding.tokens;
        }
        metric_sessions_.store(paged_bindings_.size());
        metric_tokens_.store(tokens);
    }

    void sync_telemetry() noexcept {
        std::size_t snapshots = 0;
        std::size_t tokens = 0;
        for (const auto& [session_id, history] : states_) {
            (void)session_id;
            snapshots += history.size();
            for (const auto& snapshot : history) {
                tokens += snapshot.state.tokens.size();
            }
        }
        metric_sessions_.store(states_.size());
        metric_snapshots_.store(snapshots);
        metric_tokens_.store(tokens);
        metric_bytes_.store(bytes_);
    }

    void evict_history_to_limit(
        const std::string& session_id,
        std::uint64_t protected_clock) {
        auto found = states_.find(session_id);
        while (found != states_.end() &&
               found->second.size() > max_snapshots_per_session_) {
            std::size_t victim = found->second.size();
            for (std::size_t index = 0; index < found->second.size(); ++index) {
                const auto& snapshot = found->second[index];
                if (snapshot.last_used == protected_clock) continue;
                if (victim == found->second.size() ||
                    snapshot.last_used < found->second[victim].last_used) {
                    victim = index;
                }
            }
            if (victim == found->second.size()) break;
            erase_snapshot(session_id, victim, "history_evict");
            found = states_.find(session_id);
        }
    }

    void evict_to_budget(
        const std::string& protected_session,
        std::uint64_t protected_clock) {
        while (states_.size() > max_sessions_) {
            auto victim = states_.end();
            std::uint64_t victim_last_used = 0;
            for (auto it = states_.begin(); it != states_.end(); ++it) {
                if (it->first == protected_session) continue;
                std::uint64_t session_last_used = 0;
                for (const auto& snapshot : it->second) {
                    session_last_used = std::max(
                        session_last_used, snapshot.last_used);
                }
                if (victim == states_.end() ||
                    session_last_used < victim_last_used) {
                    victim = it;
                    victim_last_used = session_last_used;
                }
            }
            if (victim == states_.end()) break;
            close_session(victim->first);
        }
        while (bytes_ > max_bytes_) {
            std::string victim_session;
            std::size_t victim_snapshot = 0;
            std::uint64_t victim_last_used = 0;
            bool found_victim = false;
            for (const auto& [session_id, history] : states_) {
                for (std::size_t index = 0; index < history.size(); ++index) {
                    const auto& snapshot = history[index];
                    if (session_id == protected_session &&
                        snapshot.last_used == protected_clock) {
                        continue;
                    }
                    if (!found_victim ||
                        snapshot.last_used < victim_last_used) {
                        victim_session = session_id;
                        victim_snapshot = index;
                        victim_last_used = snapshot.last_used;
                        found_victim = true;
                    }
                }
            }
            if (!found_victim) break;
            erase_snapshot(
                victim_session, victim_snapshot, "budget_evict");
        }
    }

    void erase_snapshot(
        const std::string& session_id,
        std::size_t index,
        const char* action) {
        auto found = states_.find(session_id);
        if (found == states_.end() || index >= found->second.size()) return;
        const auto removed_bytes = found->second[index].state.bytes;
        if (trace_) {
            std::cerr
                << "server_session_cache backend=metal action=" << action
                << " session=" << session_id
                << " tokens=" << found->second[index].state.tokens.size()
                << " bytes=" << removed_bytes << std::endl;
        }
        bytes_ -= removed_bytes;
        found->second.erase(
            found->second.begin() + static_cast<std::ptrdiff_t>(index));
        if (found->second.empty()) states_.erase(found);
        sync_telemetry();
    }

    std::unordered_map<std::string, std::vector<Entry>> states_;
    std::unordered_map<std::string, PagedBinding> paged_bindings_;
    std::shared_ptr<mfq::cache::PagedPrefixCache> paged_cache_;
    mutable std::mutex paged_control_mutex_;
    std::uint64_t paged_disk_budget_ = 0;
    std::uint64_t paged_hot_budget_ = 0;
    std::size_t max_sessions_ = 4;
    std::size_t max_snapshots_per_session_ = 4;
    std::size_t max_bytes_ = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    std::size_t bytes_ = 0;
    std::uint64_t clock_ = 0;
    std::atomic<std::uint64_t> queries_{0};
    std::atomic<std::uint64_t> hits_{0};
    std::atomic<std::uint64_t> hit_tokens_{0};
    std::atomic<std::size_t> metric_sessions_{0};
    std::atomic<std::size_t> metric_snapshots_{0};
    std::atomic<std::size_t> metric_tokens_{0};
    std::atomic<std::size_t> metric_bytes_{0};
    bool trace_ = false;
};

std::int32_t generate_with_prefill_metrics(
    mfq::metal::MlxQwen35CausalLm& runtime,
    const std::vector<std::int64_t>& prompt,
    const mfq::metal::MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(MfqPrefillTiming)>& on_prefill,
    const MfqPromptCachePlan& cache_plan,
    const MfqTokenConstraintPtr& token_constraint,
    int prefill_chunk_size,
    const mfq::metal::MlxPrefixCacheHooks& prefix_cache = {}) {
    std::function<void(std::size_t, double)> report_prefill;
    if (on_prefill) {
        report_prefill = [on_prefill](std::size_t tokens, double llm_ms) {
            on_prefill(MfqPrefillTiming{tokens, llm_ms, 0.0, llm_ms});
        };
    }
    return runtime.generate(
        prompt,
        sampling,
        max_tokens,
        callback,
        report_prefill,
        token_constraint,
        cache_plan.stable_prefix_tokens > 0
            ? std::optional<std::size_t>(
                  cache_plan.stable_prefix_tokens)
            : std::nullopt,
        prefill_chunk_size,
        prefix_cache);
}

std::int32_t generate_with_prefill_metrics(
    mfq::metal::MlxNemotronHCausalLm& runtime,
    const std::vector<std::int64_t>& prompt,
    const mfq::metal::MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(MfqPrefillTiming)>& on_prefill,
    const MfqPromptCachePlan& cache_plan,
    const MfqTokenConstraintPtr& token_constraint,
    int prefill_chunk_size,
    const mfq::metal::MlxPrefixCacheHooks& = {}) {
    std::function<void(std::size_t, double)> report_prefill;
    if (on_prefill) {
        report_prefill = [on_prefill](std::size_t tokens, double llm_ms) {
            on_prefill(MfqPrefillTiming{tokens, llm_ms, 0.0, llm_ms});
        };
    }
    (void)cache_plan;
    return runtime.generate(
        prompt,
        sampling,
        max_tokens,
        callback,
        prefill_chunk_size,
        token_constraint);
}

std::int32_t generate_with_prefill_metrics(
    mfq::metal::MlxMiniCPMO45Runtime& runtime,
    const std::vector<std::int64_t>& prompt,
    const mfq::metal::MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(MfqPrefillTiming)>& on_prefill,
    const MfqPromptCachePlan& cache_plan,
    const MfqTokenConstraintPtr& token_constraint,
    int prefill_chunk_size,
    const mfq::metal::MlxPrefixCacheHooks& = {}) {
    std::function<void(std::size_t, double)> report_prefill;
    if (on_prefill) {
        report_prefill = [on_prefill](std::size_t tokens, double llm_ms) {
            on_prefill(MfqPrefillTiming{tokens, llm_ms, 0.0, llm_ms});
        };
    }
    return runtime.generate(
        prompt,
        sampling,
        max_tokens,
        callback,
        report_prefill,
        token_constraint,
        cache_plan.stable_prefix_tokens > 0
            ? std::optional<std::size_t>(
                  cache_plan.stable_prefix_tokens)
            : std::nullopt,
        prefill_chunk_size);
}

std::int32_t generate_with_prefill_metrics(
    mfq::metal::MlxDeepseekV4CausalLm& runtime,
    const std::vector<std::int64_t>& prompt,
    const mfq::metal::MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(MfqPrefillTiming)>& on_prefill,
    const MfqPromptCachePlan& cache_plan,
    const MfqTokenConstraintPtr& token_constraint,
    int prefill_chunk_size,
    const mfq::metal::MlxPrefixCacheHooks& = {}) {
    std::function<void(std::size_t, double)> report_prefill;
    if (on_prefill) {
        report_prefill = [on_prefill](std::size_t tokens, double llm_ms) {
            on_prefill(MfqPrefillTiming{tokens, llm_ms, 0.0, llm_ms});
        };
    }
    return runtime.generate(
        prompt,
        sampling,
        max_tokens,
        callback,
        std::nullopt,
        prefill_chunk_size,
        report_prefill,
        cache_plan.stable_prefix_tokens > 0
            ? std::optional<std::size_t>(
                  cache_plan.stable_prefix_tokens)
            : std::nullopt,
        token_constraint);
}

std::int32_t generate_with_prefill_metrics(
    mfq::metal::MlxDeepseekV41CausalLm& runtime,
    const std::vector<std::int64_t>& prompt,
    const mfq::metal::MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(MfqPrefillTiming)>& on_prefill,
    const MfqPromptCachePlan& cache_plan,
    const MfqTokenConstraintPtr& token_constraint,
    int prefill_chunk_size,
    const mfq::metal::MlxPrefixCacheHooks& prefix_cache = {}) {
    std::function<void(std::size_t, double)> report_prefill;
    if (on_prefill) {
        report_prefill = [on_prefill](std::size_t tokens, double llm_ms) {
            on_prefill(MfqPrefillTiming{tokens, llm_ms, 0.0, llm_ms});
        };
    }
    return runtime.generate(
        prompt,
        sampling,
        max_tokens,
        callback,
        report_prefill,
        token_constraint,
        cache_plan.stable_prefix_tokens > 0
            ? std::optional<std::size_t>(
                  cache_plan.stable_prefix_tokens)
            : std::nullopt,
        prefill_chunk_size,
        prefix_cache);
}

std::int32_t generate_with_prefill_metrics(
    mfq::metal::MlxQwen4CausalLm& runtime,
    const std::vector<std::int64_t>& prompt,
    const mfq::metal::MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const std::function<bool(std::int64_t)>& callback,
    const std::function<void(MfqPrefillTiming)>& on_prefill,
    const MfqPromptCachePlan& cache_plan,
    const MfqTokenConstraintPtr& token_constraint,
    int prefill_chunk_size,
    const mfq::metal::MlxPrefixCacheHooks& prefix_cache = {}) {
    std::function<void(std::size_t, double)> report_prefill;
    if (on_prefill) {
        report_prefill = [on_prefill](std::size_t tokens, double llm_ms) {
            on_prefill(MfqPrefillTiming{tokens, llm_ms, 0.0, llm_ms});
        };
    }
    return runtime.generate(
        prompt,
        sampling,
        max_tokens,
        callback,
        report_prefill,
        token_constraint,
        cache_plan.stable_prefix_tokens > 0
            ? std::optional<std::size_t>(cache_plan.stable_prefix_tokens)
            : std::nullopt,
        prefill_chunk_size,
        prefix_cache);
}

template <typename Runtime, typename Loader>
class LoadedMetalModel final : public mfq::metal::MlxEngineModel {
public:
    using Cache = MlxServerTextSessionCache<Runtime>;
    using PagedCacheFactory = std::function<std::shared_ptr<mfq::cache::PagedPrefixCache>(std::int64_t)>;
    LoadedMetalModel(std::shared_ptr<std::mutex> mutex,
        std::shared_ptr<std::optional<Runtime>> runtime,
        std::shared_ptr<Cache> cache, Loader loader, PagedCacheFactory factory,
        std::shared_ptr<std::int64_t> context, mlx::core::Stream stream,
        int chunk_size, std::size_t cache_limit,
        mfq::metal::MlxEngineComponents components,
        mfq::metal::MlxSamplingParams prepare_sampling, int prepare_routes, int prepare_vocab,
        std::vector<std::int64_t> prepare_prompt, int batch_capacity)
        : runtime_mutex(std::move(mutex)), runtime_holder(std::move(runtime)),
          session_cache(std::move(cache)), load_runtime(std::move(loader)),
          paged_cache_factory(std::move(factory)), loaded_context(std::move(context)),
          runtime_stream(stream), prefill_chunk_size(chunk_size),
          allocator_cache_limit(cache_limit), runtime_components(std::move(components)),
          preparation_sampling(prepare_sampling), preparation_routes(prepare_routes),
          preparation_vocab(prepare_vocab), warmup_prompt(std::move(prepare_prompt)),
          configured_batch_capacity(batch_capacity) {
        capture_weight_residency();
        resident_budget = cache_bytes_from_environment("MFQ_SERVER_RESIDENT_BUDGET_BYTES", 0);
        if (resident_budget) enforce_memory_budget(0);
        warm_runtime();
        configure_batching();
    }

    void configure_batching() {
        if constexpr (std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm>) {
            if (configured_batch_capacity > 1) {
                mfq::metal::MlxQwen4BatchOperations::Hooks hooks;
                hooks.block_size = session_cache->block_size_tokens();
                hooks.restore = [this](auto& request) {
                    request.prefix = session_cache->normalize_stable_prefix_tokens(std::min(
                        request.cache_plan.stable_prefix_tokens, request.prompt.size()));
                    if (!request.prefix || (request.cache_plan.session_id.empty() &&
                        !session_cache->persistent_prefix_enabled())) {
                        request.prefix = 0;
                        return std::size_t{0};
                    }
                    return session_cache->restore_best(runtime_holder->value(), request.cache_plan.session_id,
                        request.prompt, std::min(request.prefix, request.prompt.size() - 1), request.mtp);
                };
                hooks.store = [this](auto& request, auto state) {
                    session_cache->store(request.cache_plan.session_id, std::move(state));
                };
                batching = std::make_unique<MetalBatch>(prefill_chunk_size, prefill_chunk_size,
                    runtime_holder->value(), std::move(hooks));
                batching_capacity = static_cast<std::size_t>(configured_batch_capacity);
                batch_mtp_enabled = runtime_holder->value().supports_batch_mtp();
            }
        }
    }

    std::size_t request_capacity() const override { return batching_capacity; }
    bool can_batch(const mfq::engine::EngineRequest& request) const override {
        return batching && !request.input.media &&
            !(request.input.sampling.enable_mtp && runtime_components.mtp_available && !batch_mtp_enabled);
    }
    void execute(const std::vector<mfq::engine::RequestId>& eligible) override {
        if (!batching || batching->empty() || eligible.empty()) return;
        std::lock_guard lock(*runtime_mutex);
        mlx::core::set_default_device(mlx::core::Device::gpu);
        mlx::core::set_default_stream(runtime_stream);
        mfq::metal::MlxResidentBudgetScope budget_scope(
            [this](std::size_t bytes) { enforce_memory_budget(bytes); },
            [this](std::size_t bytes) { return optional_memory_fits(bytes); });
        batching->step(eligible);
    }
    mfq::engine::Generation generate_batch(const mfq::engine::RequestId& id,
        mfq::engine::InferenceRequest& input, mfq::engine::InferenceOutput& output) override {
        if (!batching) throw std::logic_error("Metal batch executor is unavailable");
        auto generation = batching->generate(id, input, output);
        mfq::engine::ExecutionCleanup cleanup{output.cleanup_failure, [&] {
            std::lock_guard lock(*runtime_mutex);
            mlx::core::set_default_device(mlx::core::Device::gpu);
            mlx::core::set_default_stream(runtime_stream);
            mfq::metal::MlxResidentBudgetScope budget_scope(
                [this](std::size_t bytes) { enforce_memory_budget(bytes); },
                [this](std::size_t bytes) { return optional_memory_fits(bytes); });
            generation = {};
        }};
        while (true) {
            auto step = [&] {
                std::lock_guard lock(*runtime_mutex);
                mlx::core::set_default_device(mlx::core::Device::gpu);
                mlx::core::set_default_stream(runtime_stream);
                mfq::metal::MlxResidentBudgetScope budget_scope(
                    [this](std::size_t bytes) { enforce_memory_budget(bytes); },
                    [this](std::size_t bytes) { return optional_memory_fits(bytes); });
                auto step = generation.next();
                return step;
            }();
            if (!step) break;
            co_yield std::move(step);
        }
        cleanup.finish();
    }

    void generate(mfq::engine::InferenceRequest& request,
                  mfq::metal::MlxGenerationJob& job) override {
        const auto& prompt = request.prompt;
        const auto& sampling = request.sampling;
        const auto& cache_plan = request.cache_plan;
        const auto& token_constraint = request.token_constraint;
        const std::function<bool(std::int64_t)> callback =
            [&job](std::int64_t token) { return job.token(token); };
        const std::function<void(MfqPrefillTiming)> on_prefill =
            [&job](MfqPrefillTiming timing) { job.prefill(timing); };
        job.mtp.available = runtime_components.mtp_available;
        try {
            if (request.vision && !std::is_same_v<Runtime, mfq::metal::MlxDeepseekV41CausalLm>) {
                if (!runtime_components.multimodal_generate)
                    throw std::invalid_argument("model has no multimodal component");
                runtime_components.multimodal_generate(prompt, *request.vision,
                    sampling, job, token_constraint);
            } else {
                std::lock_guard<std::mutex> lock(*runtime_mutex);
                if (!runtime_holder->has_value()) {
                    throw std::runtime_error(
                        "model runtime is unavailable after a failed reload");
                }
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
                parameters.enable_mtp = sampling.enable_mtp;
                parameters.mtp_max_draft_tokens =
                    sampling.mtp_max_draft_tokens;
                parameters.seed = sampling.seed;
                auto& loaded_runtime = runtime_holder->value();
                std::vector<mfq::metal::MlxDeepseekV41ImageInput> images;
                std::string media_key;
                if constexpr (std::is_same_v<Runtime, mfq::metal::MlxDeepseekV41CausalLm>) {
                    if (request.vision) {
                        images = mfq::metal::prepare_deepseek_v41_images(prompt, *request.vision, loaded_runtime.config());
                        media_key = metal_media_cache_key(*request.vision);
                    }
                }
                mfq::metal::MlxResidentBudgetScope budget_scope([this](std::size_t bytes) {
                    enforce_memory_budget(bytes);
                }, [this](std::size_t bytes) { return optional_memory_fits(bytes); });
                if constexpr (requires { loaded_runtime.begin_mtp_session(std::string{}); })
                    loaded_runtime.begin_mtp_session(parameters.enable_mtp ? cache_plan.session_id : std::string{});
                struct EndMtpSession {
                    Runtime& runtime;
                    ~EndMtpSession() {
                        if constexpr (requires { runtime.end_mtp_session(); }) runtime.end_mtp_session();
                    }
                 } end_mtp_session{loaded_runtime};
                enforce_memory_budget(0);
                const auto stable_prefix_tokens =
                    session_cache->normalize_stable_prefix_tokens(std::min(
                        cache_plan.stable_prefix_tokens, prompt.size()));
                const bool cache_enabled =
                    stable_prefix_tokens > 0 &&
                    (!cache_plan.session_id.empty() ||
                     session_cache->persistent_prefix_enabled()) &&
                    loaded_runtime.supports_text_session_state();
                if (cache_enabled) {
                    const auto restored = session_cache->restore_best(
                        loaded_runtime,
                        cache_plan.session_id,
                        prompt,
                        stable_prefix_tokens, parameters.enable_mtp && runtime_components.mtp_available &&
                            sampling.max_tokens > 1, media_key);
                    if constexpr (std::is_same_v<Runtime, mfq::metal::MlxDeepseekV41CausalLm>)
                        if (request.vision && restored == 0) loaded_runtime.reset_cache(1);
                }
                auto effective_cache_plan = cache_plan;
                effective_cache_plan.stable_prefix_tokens = cache_enabled
                    ? stable_prefix_tokens
                    : 0;
                mfq::metal::MlxPrefixCacheHooks prefix_hooks;
                auto cache_tokens = prompt;
                std::size_t last_checkpoint = 0;
                if constexpr (std::is_same_v<Runtime, mfq::metal::MlxQwen35CausalLm> ||
                    std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm> ||
                    std::is_same_v<Runtime, mfq::metal::MlxDeepseekV41CausalLm>) {
                    if (cache_enabled) {
                        prefix_hooks.block_size = session_cache->block_size_tokens();
                        prefix_hooks.input_tokens = stable_prefix_tokens;
                        prefix_hooks.output_tokens = cache_plan.cache_output_tokens;
                        prefix_hooks.capture = [&](std::size_t position) {
                            if (position == last_checkpoint || position > cache_tokens.size()) return;
                            try {
                                session_cache->store(cache_plan.session_id,
                                    loaded_runtime.capture_text_session_state(
                                        std::vector<std::int64_t>(cache_tokens.begin(),
                                            cache_tokens.begin() + static_cast<std::ptrdiff_t>(position)),
                                        std::is_same_v<Runtime, mfq::metal::MlxDeepseekV41CausalLm> &&
                                            !session_cache->persistent_prefix_enabled()), media_key);
                                last_checkpoint = position;
                            } catch (const std::exception& error) {
                                std::cerr << "server_session_cache backend=metal action=skip error="
                                          << error.what() << std::endl;
                            }
                        };
                    }
                }
                const auto cached_callback = [&](std::int64_t token) {
                    enforce_memory_budget(0);
                    cache_tokens.push_back(token);
                    return !callback || callback(token);
                };
                const auto generated = [&] {
                    if constexpr (std::is_same_v<Runtime, mfq::metal::MlxDeepseekV41CausalLm>) {
                        if (request.vision) return loaded_runtime.generate_multimodal(prompt, images, parameters,
                            sampling.max_tokens, cached_callback, [&](std::size_t tokens, double ms) {
                                on_prefill(MfqPrefillTiming{tokens, ms, 0.0, ms});
                            }, token_constraint, effective_cache_plan.stable_prefix_tokens, prefill_chunk_size, prefix_hooks);
                    }
                    return generate_with_prefill_metrics(
                    loaded_runtime,
                    prompt,
                    parameters,
                    sampling.max_tokens,
                    cached_callback,
                    on_prefill,
                    effective_cache_plan,
                    token_constraint,
                    prefill_chunk_size,
                    prefix_hooks);
                }();
                if (!prefix_hooks && cache_enabled &&
                    loaded_runtime.cache_position() ==
                        static_cast<int>(stable_prefix_tokens)) {
                    try {
                        std::vector<std::int64_t> stable_tokens(
                            prompt.begin(),
                            prompt.begin() + static_cast<std::ptrdiff_t>(
                                stable_prefix_tokens));
                        session_cache->store(
                            cache_plan.session_id,
                            loaded_runtime.capture_text_session_state(
                                stable_tokens));
                    } catch (const std::exception& error) {
                        std::cerr
                            << "server_session_cache backend=metal action=skip "
                            << "session=" << cache_plan.session_id
                            << " error=" << error.what() << std::endl;
                    }
                }
                (void)generated;
            }
            mfq::metal::drain_metal_work(runtime_stream);
            if constexpr (requires(Runtime& value) { value.last_mtp_stats(); }) {
                const auto& stats = runtime_holder->value().last_mtp_stats();
                job.mtp = {stats.available, stats.used, stats.cycles,
                    stats.drafted_tokens, stats.accepted_tokens, stats.depth_cycles,
                    stats.position_drafted, stats.position_accepted,
                    stats.measured_depth_ms, stats.selected_depth,
                    stats.standard_tokens, stats.park_count, stats.reentry_probes};
            }
        } catch (...) {
            mfq::metal::drain_metal_work(runtime_stream);
            throw;
        }
    }

    std::int64_t reload(std::int64_t requested_context) override {
        std::lock_guard<std::mutex> lock(*runtime_mutex);
        batching.reset();
        mlx::core::set_default_device(
            mlx::core::Device::gpu);
        mlx::core::set_default_stream(runtime_stream);
        mfq::metal::drain_metal_work(runtime_stream);

        const auto previous_context = *loaded_context;
        session_cache->clear_live_sessions();
        weight_residency.reset();
        runtime_holder->reset();
        release_model_load_staging_memory(runtime_stream);
        mfq::metal::MlxWeightResidency::begin_load();
        const auto started =
            std::chrono::steady_clock::now();
        std::cout
            << "Reloading native Metal model with context="
            << requested_context << "..." << std::endl;
        try {
            mfq::metal::MlxKernelPreparation preparation(static_cast<int>(requested_context), prefill_chunk_size);
            preparation.set_routes(preparation_routes);
            runtime_holder->emplace(
                load_runtime(requested_context));
            if constexpr (requires(Runtime& value) {
                    value.prewarm_ssd_expert_arena();
                }) {
                runtime_holder->value()
                    .prewarm_ssd_expert_arena();
            }
            preparation.collect_sampling(preparation_vocab, preparation_sampling);
            preparation.finish();
            session_cache->replace_paged_cache(
                paged_cache_factory(requested_context),
                cache_bytes_from_environment(
                    "MFQ_SERVER_PREFIX_CACHE_DISK_BYTES",
                    100ULL * 1024ULL * 1024ULL * 1024ULL),
                cache_bytes_from_environment(
                    "MFQ_SERVER_PREFIX_CACHE_HOT_BYTES",
                    2ULL * 1024ULL * 1024ULL * 1024ULL));
            release_model_load_staging_memory(runtime_stream);
            capture_weight_residency();
            *loaded_context = requested_context;
            warm_runtime();
            configure_batching();
            const auto seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    started).count();
            std::cout
                << "Reloaded native Metal model in "
                << seconds << " s context="
                << requested_context << std::endl;
            return requested_context;
        } catch (const std::exception& reload_error) {
            const std::string message = reload_error.what();
            std::cerr
                << "Model reload failed; restoring context="
                << previous_context << std::endl;
            try {
                weight_residency.reset();
                runtime_holder->reset();
                release_model_load_staging_memory(runtime_stream);
                mfq::metal::MlxWeightResidency::begin_load();
                mfq::metal::MlxKernelPreparation preparation(static_cast<int>(previous_context), prefill_chunk_size);
                preparation.set_routes(preparation_routes);
                runtime_holder->emplace(
                    load_runtime(previous_context));
                if constexpr (requires(Runtime& value) {
                        value.prewarm_ssd_expert_arena();
                    }) {
                    runtime_holder->value()
                        .prewarm_ssd_expert_arena();
                }
                preparation.collect_sampling(preparation_vocab, preparation_sampling);
                preparation.finish();
                session_cache->replace_paged_cache(
                    paged_cache_factory(previous_context),
                    cache_bytes_from_environment(
                        "MFQ_SERVER_PREFIX_CACHE_DISK_BYTES",
                        100ULL * 1024ULL * 1024ULL * 1024ULL),
                    cache_bytes_from_environment(
                        "MFQ_SERVER_PREFIX_CACHE_HOT_BYTES",
                        2ULL * 1024ULL * 1024ULL * 1024ULL));
                release_model_load_staging_memory(runtime_stream);
                capture_weight_residency();
                *loaded_context = previous_context;
                warm_runtime();
                configure_batching();
            } catch (const std::exception& restore_error) {
                throw std::runtime_error(
                    "model reload failed: " + message +
                    "; restoring the previous model also failed: " +
                    restore_error.what());
            }
            throw std::runtime_error(
                "model reload failed and the previous model was "
                "restored: " + message);
        }
    }

    mfq::engine::SessionResult session(const mfq::engine::SessionCommand& command) override {
        using Kind = mfq::engine::SessionCommand::Kind;
        if (command.kind == Kind::metrics) return {0, session_cache->metrics()};
        if (command.kind == Kind::trim) return {session_cache->trim_hot(command.bytes), {}};
        if (command.kind == Kind::budget) {
            const auto released = session_cache->set_hot_limit(command.bytes);
            if (command.disk_bytes) session_cache->set_disk_limit(*command.disk_bytes);
            return {released, {}};
        }
        std::lock_guard lock(*runtime_mutex);
        switch (command.kind) {
        case Kind::fork: {
            std::size_t copied = 0;
            if constexpr (requires(Runtime& value) { value.fork_mtp_session(command.source, command.target); })
                if (runtime_holder->has_value()) copied = runtime_holder->value().fork_mtp_session(command.source, command.target);
            return {copied + session_cache->fork_session(command.source, command.target), {}};
        }
        case Kind::close: {
            std::size_t removed = 0;
            if constexpr (requires(Runtime& value) { value.close_mtp_session(command.source); })
                if (runtime_holder->has_value()) removed = runtime_holder->value().close_mtp_session(command.source);
            return {removed + session_cache->close_session(command.source), {}};
        }
        case Kind::memory_budget: {
            const auto previous = resident_budget.load();
            const auto previous_weights = resident_weight_budget.load();
            const auto before = mlx::core::get_active_memory();
            resident_budget = command.bytes;
            if (command.weight_bytes) resident_weight_budget = *command.weight_bytes;
            try {
                if (resident_weight_budget && resident_weight_budget < resident_dense_weight_baseline)
                    throw mfq::metal::MlxResidentBudgetExceeded("weight budget is smaller than non-offloadable weights");
                enforce_memory_budget(0); apply_expert_budget();
            }
            catch (...) { resident_budget = previous; resident_weight_budget = previous_weights; throw; }
            if (batching && (resident_budget.load() != previous || resident_weight_budget.load() != previous_weights))
                batching->invalidate_runtime_policy();
            const auto after = mlx::core::get_active_memory();
            return {before > after ? before - after : 0, {}};
        }
        case Kind::refresh:
            return {session_cache->refresh_disk_index(), {}};
        case Kind::clear: {
            mlx::core::set_default_device(mlx::core::Device::gpu);
            mlx::core::set_default_stream(runtime_stream);
            mfq::metal::drain_metal_work(runtime_stream);
            const auto released = session_cache->clear();
            if (runtime_holder->has_value()) {
                if constexpr (requires(Runtime& value) { value.clear_mtp_sessions(); })
                    runtime_holder->value().clear_mtp_sessions();
                if constexpr (requires(Runtime& value) { value.reset_cache(1); })
                    runtime_holder->value().reset_cache(1);
                else runtime_holder->value().reset();
            }
            release_allocator_caches();
            return {released, {}};
        }
        case Kind::trim:
        case Kind::budget:
        case Kind::metrics: break;
        }
        throw std::invalid_argument("unknown Metal session command");
    }

    mfq::engine::ControlResult control(mfq::engine::ControlRequest request) override {
        return std::visit([&](auto value) -> mfq::engine::ControlResult {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, mfq::engine::RuntimeMetrics>) return metrics();
            else if constexpr (std::is_same_v<T, mfq::engine::ScoreTokens>) {
                if constexpr (requires(Runtime& runtime, mlx::core::array ids) { runtime.score_forward(ids, false); }) {
                    std::lock_guard<std::mutex> lock(*runtime_mutex);
                    if (!runtime_holder->has_value()) throw std::runtime_error("model runtime is unavailable");
                    mlx::core::set_default_device(mlx::core::Device::gpu);
                    mlx::core::set_default_stream(runtime_stream);
                    mfq::metal::MlxResidentBudgetScope budget_scope(
                        [this](std::size_t bytes) { enforce_memory_budget(bytes); },
                        [this](std::size_t bytes) { return optional_memory_fits(bytes); });
                    auto& runtime = runtime_holder->value();
                    mfq::engine::LikelihoodResult result{value.prompt_tokens, {}};
                    std::optional<mlx::core::array> next_logprobs;
                    for (std::size_t i = 0; i < value.targets.size(); ++i) {
                        const auto& sequence = value.sequences.at(i);
                        const auto& targets = value.targets[i];
                        const int count = static_cast<int>(targets.size());
                        const auto memory = sequence.size() * static_cast<std::size_t>(runtime.config().vocab_size) * sizeof(float) * 2;
                        enforce_memory_budget(value.next_token ? 0 : memory);
                        const auto score = [&] {
                            if (value.next_token && next_logprobs) return *next_logprobs;
                            std::vector<std::int32_t> tokens(sequence.begin(), sequence.end());
                            if (!value.next_token) tokens.pop_back();
                            mlx::core::array ids(tokens.begin(), mlx::core::Shape{1, static_cast<int>(tokens.size())}, mlx::core::int32);
                            auto logits = runtime.score_forward(ids, value.next_token);
                            auto rows = mlx::core::reshape(logits, {-1, logits.shape(-1)});
                            const int begin = value.next_token ? rows.shape(0) - 1 : static_cast<int>(value.prompt_tokens) - 1;
                            rows = mlx::core::astype(mlx::core::slice(rows, {begin, 0}, {begin + (value.next_token ? 1 : count), rows.shape(1)}), mlx::core::float32);
                            auto logprobs = rows - mlx::core::logsumexp(rows, -1, true);
                            if (value.next_token) { logprobs.eval(); next_logprobs = logprobs; }
                            return logprobs;
                        }();
                        std::vector<std::int32_t> ids(targets.begin(), targets.end());
                        auto selected = mlx::core::take_along_axis(score,
                            mlx::core::array(ids.begin(), mlx::core::Shape{count, 1}, mlx::core::int32), -1);
                        selected.eval();
                        const auto* numbers = selected.template data<float>();
                        mfq::engine::TokenLikelihood output{targets, {}, 0};
                        for (int token = 0; token < count; ++token) {
                            const double number = numbers[token];
                            if (!std::isfinite(number) || number > 1e-5)
                                throw std::runtime_error("runtime produced an invalid token log probability");
                            output.token_logprobs.push_back(number);
                            output.log_likelihood += number;
                        }
                        result.scores.push_back(std::move(output));
                    }
                    return result;
                } else throw std::invalid_argument("probability scoring is unavailable for this model");
            }
            else {
                auto& duplex = runtime_components.duplex;
                if (!duplex) throw std::invalid_argument("model has no duplex component");
                if constexpr (std::is_same_v<T, MfqDuplexSessionParams>) duplex.start(value);
                else if constexpr (std::is_same_v<T, MfqDuplexStepInput>) return duplex.step(value);
                else if constexpr (std::is_same_v<T, mfq::engine::StopDuplex>) duplex.stop();
                else throw std::invalid_argument("text control belongs to the engine");
                return std::monostate{};
            }
        }, std::move(request));
    }

private:
    void apply_expert_budget() {
        const auto weights = resident_weight_budget.load();
        if (weights && weights < resident_dense_weight_baseline)
            throw mfq::metal::MlxResidentBudgetExceeded("weight budget is smaller than non-offloadable weights");
        if constexpr (requires(Runtime& value) { value.set_expert_cache_limit(std::size_t{}); }) {
            auto& runtime = runtime_holder->value();
            auto room = weights ? weights - resident_dense_weight_baseline : std::numeric_limits<std::size_t>::max();
            const auto total = resident_budget.load();
            if (total) {
                const auto available = runtime.reclaimable_expert_bytes();
                const auto used = resident_memory_used();
                const auto cached = mlx::core::get_cache_memory();
                const auto other = used > available ? used - available : 0;
                const auto fixed = other - std::min(other, cached);
                const auto reserve = std::max(resident_allocator_reserve_bytes(total), cached);
                room = std::min(room, total > fixed + reserve ? total - fixed - reserve : 0);
            }
            if (room == std::numeric_limits<std::size_t>::max())
                room = runtime.ssd_expert_payload_bytes() ? runtime.ssd_expert_payload_bytes() : runtime.resident_full_expert_bytes();
            if (room != std::numeric_limits<std::size_t>::max()) {
                mfq::metal::drain_metal_work(runtime_stream);
                const auto active = mlx::core::get_active_memory();
                const auto full = runtime.resident_full_expert_bytes();
                runtime.set_expert_cache_limit(room);
                if (runtime.resident_full_expert_bytes() != full) weight_residency.reset();
                resident_weight_baseline = resident_dense_weight_baseline + runtime.resident_full_expert_bytes();
                if (mlx::core::get_active_memory() < active || runtime.resident_full_expert_bytes() != full) {
                    mlx::core::clear_cache();
                    release_host_allocator_cache();
                }
            }
        }
    }
    bool optional_memory_fits(std::size_t allocation) const {
        const auto budget = resident_budget.load();
        if (!budget || !runtime_holder->has_value()) return true;
        const auto used = resident_memory_used() + mfq::metal::MlxResidentBudgetScope::temporary_bytes();
        const auto reserve = resident_allocator_reserve_bytes(budget);
        const auto cached = mlx::core::get_cache_memory();
        const auto headroom = reserve > cached ? reserve - cached : 0;
        return used <= budget && headroom <= budget - used && allocation <= budget - used - headroom;
    }
    void enforce_memory_budget(std::size_t allocation) {
        const auto budget = resident_budget.load();
        const auto reserve = resident_allocator_reserve_bytes(budget);
        struct RestoreAllocatorCache {
            std::size_t& current;
            std::size_t previous;
            std::size_t target;
            int exceptions = std::uncaught_exceptions();
            ~RestoreAllocatorCache() {
                const auto limit = std::uncaught_exceptions() > exceptions ? previous : target;
                if (current == limit) return;
                mlx::core::set_cache_limit(limit);
                current = limit;
            }
        } restore_cache{allocator_cache_limit, allocator_cache_limit,
            budget ? std::min(server_cache_limit_bytes(), reserve) : server_cache_limit_bytes()};
        if (!budget || !runtime_holder->has_value()) return;
        auto used = [&] {
            const auto actual = resident_memory_used() + mfq::metal::MlxResidentBudgetScope::temporary_bytes();
            const auto cached = mlx::core::get_cache_memory();
            return actual + (reserve > cached ? reserve - cached : 0);
        };
        if (allocation > budget || reserve > budget - allocation)
            throw mfq::metal::MlxResidentBudgetExceeded("resident memory budget cannot accommodate this allocation");
        const auto target = budget - allocation;
        if (used() <= target) return;
        mfq::metal::drain_metal_work(runtime_stream);
        const auto clear_allocator_excess = [&] {
            if (mlx::core::get_cache_memory() > reserve) mlx::core::clear_cache();
        };
        clear_allocator_excess();
        std::size_t prefix = 0;
        for (const auto& metric : session_cache->metrics()) if (metric.first == "prefix_cache_hot_bytes") prefix = static_cast<std::size_t>(metric.second);
        const auto shortage = used() > target ? used() - target : 0;
        session_cache->trim_hot(prefix > shortage ? prefix - shortage : 0);
        clear_allocator_excess();
        if constexpr (requires(Runtime& value) { value.trim_mtp_sessions(std::size_t{}); })
            if (used() > target) runtime_holder->value().trim_mtp_sessions(0);
        clear_allocator_excess();
        if (used() <= target) return;
        if constexpr (requires(Runtime& value) { value.set_expert_cache_limit(std::size_t{}); }) {
            auto& runtime = runtime_holder->value();
            for (auto observed = used(); observed > target; observed = used()) {
                const auto available = runtime.reclaimable_expert_bytes();
                if (!available) break;
                const auto excess = observed - target;
                const auto full = runtime.resident_full_expert_bytes();
                mfq::metal::mlx_trim_expert_residency(runtime, available > excess ? available - excess : 0);
                if (runtime.resident_full_expert_bytes() != full) weight_residency.reset();
                resident_weight_baseline = resident_dense_weight_baseline + runtime.resident_full_expert_bytes();
                clear_allocator_excess();
                release_host_allocator_cache();
                if (runtime.reclaimable_expert_bytes() >= available) break;
            }
        }
        if (used() > target) throw mfq::metal::MlxResidentBudgetExceeded(
            "resident memory budget exhausted by non-offloadable weights, live KV or execution buffers"
            "; used_bytes=" + std::to_string(used()) + "; target_bytes=" + std::to_string(target) +
            "; allocation_bytes=" + std::to_string(allocation) + "; budget_bytes=" + std::to_string(budget));
    }
    std::size_t resident_memory_used() const {
        const auto usage = mfq::metal::MlxResourceTelemetry::snapshot();
        std::size_t prefix = 0;
        for (const auto& metric : session_cache->metrics()) if (metric.first == "prefix_cache_resident_bytes") prefix = static_cast<std::size_t>(metric.second);
        std::size_t adapters = 0;
        if constexpr (requires(Runtime& value) { value.mtp_session_bytes(); })
            if (runtime_holder->has_value()) adapters = runtime_holder->value().mtp_session_bytes();
        std::size_t offloaded = 0;
        if constexpr (requires(Runtime& value) { value.qsa_kv_offload_stats(); }) {
            if (runtime_holder->has_value()) {
                const auto stats = runtime_holder->value().qsa_kv_offload_stats();
                if (stats) offloaded = stats->hot_bytes + stats->pending_bytes;
            }
        }
        return std::max(mlx::core::get_active_memory(), resident_weight_baseline +
            usage.dynamic_weight_bytes + usage.cache_bytes) + offloaded + prefix + adapters + mlx::core::get_cache_memory();
    }
    void capture_weight_residency() {
        mfq::metal::MlxMemoryResidency::refresh();
        weight_residency = mfq::metal::MlxWeightResidency::finish_load(
            mfq::metal::MlxMemoryResidency::configured_limit());
        const auto usage = mfq::metal::MlxResourceTelemetry::snapshot();
        const auto other = usage.cache_bytes + usage.dynamic_weight_bytes;
        const auto active = mlx::core::get_active_memory();
        resident_weight_baseline = active > other ? active - other : 0;
        resident_dense_weight_baseline = resident_weight_baseline;
        if constexpr (requires(Runtime& value) { value.reclaimable_expert_bytes(); }) {
            const auto experts = runtime_holder->value().dynamic_weight_bytes() ? 0 : runtime_holder->value().reclaimable_expert_bytes();
            resident_dense_weight_baseline = resident_weight_baseline > experts ? resident_weight_baseline - experts : 0;
        }
        std::cerr << "Metal weight residency: bytes="
                  << (weight_residency ? weight_residency->bytes() : 0) << std::endl;
        if (resident_weight_budget) apply_expert_budget();
        self_test_metal();
    }

    mfq::engine::Metrics metrics() const {
        double qsa_offload_supported = 0;
        if constexpr (std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm>) {
            if (runtime_holder->has_value()) {
                const auto& config = runtime_holder->value().config();
                qsa_offload_supported = config.mtp_num_hidden_layers > 0 ||
                    std::find(config.layer_types.begin(), config.layer_types.end(), "full_attention") != config.layer_types.end();
            }
        }
        std::vector<std::pair<std::string, double>> metrics{
            {"mlx_active_bytes", static_cast<double>(mlx::core::get_active_memory())},
            {"mfe_packed_copy_count", static_cast<double>(mfq::metal::MlxMfeWeight::packed_copy_count())},
            {"ssd_expert_indirect_dispatch_count", static_cast<double>(mfq::metal::MlxMfeWeight::indirect_dispatch_count())},
            {"mfe_uncached_packed_bytes", static_cast<double>(mfq::metal::MlxMfeWeight::uncached_packed_bytes())},
            {"ssd_expert_pinned_bytes", static_cast<double>(mfq::metal::MlxMfeWeight::pinned_packed_bytes())},
            {"ssd_expert_pin_failures", static_cast<double>(mfq::metal::MlxMfeWeight::packed_pin_failures())},
            {"ssd_expert_pooled_staging_count", static_cast<double>(mfq::metal::MlxMfeWeight::pooled_staging_count())},
            {"qsa_kv_offload_supported", qsa_offload_supported},
            {"kv_quantization_supported", (std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm> ||
                std::is_same_v<Runtime, mfq::metal::MlxQwen35CausalLm>) ? 1.0 : 0.0},
            {"kv_quantization_bits", (std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm> ||
                std::is_same_v<Runtime, mfq::metal::MlxQwen35CausalLm>) ? mfq::metal::mlx_kv_quantization().bits : 0.0},
            {"mlx_cache_bytes", static_cast<double>(mlx::core::get_cache_memory())},
            {"mlx_cache_limit_bytes", static_cast<double>(allocator_cache_limit)},
            {"mlx_peak_bytes", static_cast<double>(mlx::core::get_peak_memory())},
            {"resident_memory_budget_bytes", static_cast<double>(resident_budget.load())},
            {"resident_weight_budget_bytes", static_cast<double>(resident_weight_budget.load())},
            {"resident_weight_dynamic_budget", (std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm> ||
                std::is_same_v<Runtime, mfq::metal::MlxQwen35CausalLm> ||
                std::is_same_v<Runtime, mfq::metal::MlxMiniCPMO45Runtime>) ? 1.0 : 0.0},
            {"resident_memory_dynamic_budget", (std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm> ||
                std::is_same_v<Runtime, mfq::metal::MlxQwen35CausalLm> ||
                std::is_same_v<Runtime, mfq::metal::MlxMiniCPMO45Runtime>) ? 1.0 : 0.0},
            {"metal_wired_limit_bytes", static_cast<double>(
                mfq::metal::MlxMemoryResidency::configured_limit())},
            {"metal_wired_bytes", static_cast<double>(
                mfq::metal::MlxMemoryResidency::wired_bytes())},
            {"metal_wired_available",
                mfq::metal::MlxMemoryResidency::configured_limit() > 0 ? 1.0 : 0.0},
            {"ple_source_bytes_read", static_cast<double>(
                mfq::metal::MlxResourceTelemetry::ple_read_counter().load(std::memory_order_relaxed))},
        };
        std::unique_lock lock(*runtime_mutex, std::try_to_lock);
        if (lock.owns_lock() && runtime_holder->has_value()) {
            if (batching) {
                auto batch_metrics = batching->metrics();
                metrics.insert(metrics.end(), batch_metrics.begin(), batch_metrics.end());
                metrics.emplace_back("continuous_batching_capacity", static_cast<double>(batching_capacity));
            }
            metrics.emplace_back("resident_memory_used_bytes", static_cast<double>(resident_memory_used()));
            if constexpr (requires(Runtime& value) { value.mtp_session_metrics(); }) {
                const auto adapters = runtime_holder->value().mtp_session_metrics();
                metrics.insert(metrics.end(), adapters.begin(), adapters.end());
            }
            const auto usage = mfq::metal::MlxResourceTelemetry::snapshot();
            metrics.emplace_back("resident_weight_bytes", static_cast<double>(
                resident_weight_baseline + usage.dynamic_weight_bytes));
            if constexpr (requires(Runtime& value) { value.qsa_kv_offload_stats(); })
                metrics.emplace_back("kv_cache_bytes", static_cast<double>(runtime_holder->value().kv_cache_bytes()));
            else metrics.emplace_back("kv_cache_bytes", static_cast<double>(usage.cache_bytes));
            if constexpr (std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm>)
                metrics.emplace_back("kv_cache_contexts", static_cast<double>(runtime_holder->value().kv_cache_contexts()));
            else metrics.emplace_back("kv_cache_contexts", static_cast<double>(usage.contexts));
            {
                std::optional<mfq::metal::QsaKvStoreStats> offload;
                if constexpr (requires(Runtime& value) { value.qsa_kv_offload_stats(); })
                    offload = runtime_holder->value().qsa_kv_offload_stats();
                metrics.emplace_back("qsa_kv_offload_enabled", offload ? 1.0 : 0.0);
                metrics.emplace_back("qsa_kv_resident_bytes", offload ? static_cast<double>(offload->hot_bytes + offload->pending_bytes) : 0.0);
                metrics.emplace_back("qsa_kv_ssd_bytes", offload ? static_cast<double>(offload->disk_bytes) : 0.0);
                metrics.emplace_back("qsa_kv_pending_bytes", offload ? static_cast<double>(offload->pending_bytes) : 0.0);
                metrics.emplace_back("qsa_kv_hot_limit_bytes", offload ? static_cast<double>(offload->hot_limit) : 0.0);
                metrics.emplace_back("qsa_kv_budget_bytes", offload ? static_cast<double>(offload->budget_bytes) : 0.0);
                metrics.emplace_back("qsa_kv_ssd_read_bytes", offload ? static_cast<double>(offload->read_bytes) : 0.0);
                metrics.emplace_back("qsa_kv_ssd_written_bytes", offload ? static_cast<double>(offload->written_bytes) : 0.0);
                metrics.emplace_back("qsa_kv_ssd_reads", offload ? static_cast<double>(offload->reads) : 0.0);
                metrics.emplace_back("qsa_kv_ram_hits", offload ? static_cast<double>(offload->hits) : 0.0);
            }
            const auto ple = usage.ple_payload_bytes;
            const auto experts = usage.expert_payload_bytes;
            metrics.emplace_back("ssd_ple_enabled", ple > 0 ? 1.0 : 0.0);
            metrics.emplace_back("ssd_ple_payload_bytes", static_cast<double>(ple));
            metrics.emplace_back("ssd_expert_enabled", experts > 0 ? 1.0 : 0.0);
            metrics.emplace_back("ssd_expert_payload_bytes", static_cast<double>(experts));
            if constexpr (requires(Runtime& value) { value.reclaimable_expert_bytes(); }) {
                metrics.emplace_back("resident_reclaimable_weight_bytes", static_cast<double>(
                    runtime_holder->value().reclaimable_expert_bytes()));
            }
        }
        metrics.emplace_back("continuous_batching_mtp_supported", batch_mtp_enabled ? 1.0 : 0.0);
        if constexpr (requires(Runtime& value) {
                value.supports_mtp();
            }) {
            if (lock.owns_lock() && runtime_holder->has_value()) {
                metrics.emplace_back(
                    "mtp_available",
                    runtime_holder->value().supports_mtp() ? 1.0 : 0.0);
            }
        }
        if constexpr (requires(Runtime& value) {
                value.last_mtp_stats();
            }) {
            if (lock.owns_lock() && runtime_holder->has_value()) {
                const auto& stats =
                    runtime_holder->value().last_mtp_stats();
                metrics.emplace_back("mtp_used", stats.used ? 1.0 : 0.0);
                metrics.emplace_back(
                    "mtp_cycles", static_cast<double>(stats.cycles));
                metrics.emplace_back(
                    "mtp_drafted_tokens",
                    static_cast<double>(stats.drafted_tokens));
                metrics.emplace_back(
                    "mtp_accepted_tokens",
                    static_cast<double>(stats.accepted_tokens));
                metrics.emplace_back(
                    "mtp_acceptance_rate",
                    stats.drafted_tokens == 0
                        ? 0.0
                        : static_cast<double>(stats.accepted_tokens) /
                              stats.drafted_tokens);
                metrics.emplace_back("mtp_standard_tokens", static_cast<double>(stats.standard_tokens));
                metrics.emplace_back("mtp_park_count", static_cast<double>(stats.park_count));
                metrics.emplace_back("mtp_reentry_probes", static_cast<double>(stats.reentry_probes));
                metrics.emplace_back(
                    "mtp_selected_depth",
                    static_cast<double>(stats.selected_depth));
                for (std::size_t depth = 0;
                     depth < stats.depth_cycles.size();
                     ++depth) {
                    metrics.emplace_back(
                        "mtp_depth_" + std::to_string(depth) + "_cycles",
                        static_cast<double>(stats.depth_cycles[depth]));
                }
                for (std::size_t position = 0;
                     position < stats.position_drafted.size();
                     ++position) {
                    metrics.emplace_back(
                        "mtp_position_" + std::to_string(position + 1) +
                            "_acceptance_rate",
                        stats.position_drafted[position] == 0
                            ? 0.0
                            : static_cast<double>(
                                  stats.position_accepted[position]) /
                              stats.position_drafted[position]);
                }
                for (std::size_t depth = 0;
                     depth < stats.measured_depth_ms.size();
                     ++depth) {
                    metrics.emplace_back(
                        "mtp_depth_" + std::to_string(depth) +
                            "_cycle_ms",
                        stats.measured_depth_ms[depth]);
                }
            }
        }
        if constexpr (requires(Runtime& value) {
                value.supports_multimodal();
            }) {
            if (lock.owns_lock() && runtime_holder->has_value()) {
                metrics.emplace_back(
                    "vision_available",
                    runtime_holder->value().supports_multimodal()
                        ? 1.0 : 0.0);
            }
        }
        if constexpr (requires(Runtime& value) {
                value.ssd_expert_cache_stats();
            }) {
            if (lock.owns_lock() && runtime_holder->has_value()) {
                const auto stats =
                    runtime_holder->value().ssd_expert_cache_stats();
                if (stats.has_value()) {
                    metrics.emplace_back(
                        "ssd_expert_requests",
                        static_cast<double>(stats->requests));
                    metrics.emplace_back(
                        "ssd_expert_hits",
                        static_cast<double>(stats->hits));
                    metrics.emplace_back(
                        "ssd_expert_misses",
                        static_cast<double>(stats->misses));
                    metrics.emplace_back(
                        "ssd_expert_evictions",
                        static_cast<double>(stats->evictions));
                    metrics.emplace_back(
                        "ssd_expert_loads",
                        static_cast<double>(stats->loads));
                    metrics.emplace_back(
                        "ssd_expert_hit_rate",
                        stats->hit_rate());
                    metrics.emplace_back(
                        "ssd_expert_bytes_read",
                        static_cast<double>(stats->bytes_read));
                    metrics.emplace_back("ssd_expert_read_calls", static_cast<double>(stats->read_calls));
                    metrics.emplace_back("ssd_expert_prepare_seconds", stats->prepare_seconds);
                    metrics.emplace_back("ssd_expert_route_sync_seconds", stats->route_sync_seconds);
                    metrics.emplace_back("ssd_expert_graph_seconds", stats->routed_graph_seconds);
                    metrics.emplace_back("ssd_expert_view_seconds", stats->view_seconds);
                    metrics.emplace_back("ssd_expert_device_route_layers", stats->device_route_layers);
                    metrics.emplace_back("ssd_expert_device_route_hits", stats->device_route_hits);
                    metrics.emplace_back("ssd_expert_device_route_misses", stats->device_route_misses);
                    metrics.emplace_back("ssd_expert_overlapped_read_batches", stats->overlapped_read_batches);
                    metrics.emplace_back("ssd_expert_overlapped_read_submissions", stats->overlapped_read_submissions);
                    metrics.emplace_back("ssd_expert_parallel_body_read_batches", stats->parallel_body_read_batches);
                    metrics.emplace_back(
                        "ssd_expert_resident_bytes",
                        static_cast<double>(stats->resident_bytes));
                    metrics.emplace_back(
                        "ssd_expert_resident_count",
                        static_cast<double>(stats->resident_experts));
                    metrics.emplace_back(
                        "ssd_expert_cache_slots",
                        static_cast<double>(stats->cache_slots));
                    metrics.emplace_back(
                        "ssd_expert_cache_limit_bytes",
                        static_cast<double>(
                            runtime_holder->value()
                                .expert_cache_limit_bytes()));
                    metrics.emplace_back(
                        "ssd_expert_io_seconds",
                        stats->io_seconds);
                    metrics.emplace_back(
                        "ssd_expert_wait_seconds",
                        stats->wait_seconds);
                }
            }
        }
        if constexpr (requires(Runtime& value) {
                value.engram_ssd_stats();
            }) {
            if (lock.owns_lock() && runtime_holder->has_value()) {
                const auto stats =
                    runtime_holder->value().engram_ssd_stats();
                if (stats.has_value()) {
                    metrics.emplace_back(
                        "engram_row_requests",
                        static_cast<double>(stats->row_requests));
                    metrics.emplace_back(
                        "engram_cache_hits",
                        static_cast<double>(stats->cache_hits));
                    metrics.emplace_back(
                        "engram_cache_misses",
                        static_cast<double>(stats->cache_misses));
                    metrics.emplace_back(
                        "engram_hit_rate",
                        stats->hit_rate());
                    metrics.emplace_back(
                        "engram_rows_loaded",
                        static_cast<double>(stats->rows_loaded));
                    metrics.emplace_back(
                        "engram_bytes_read",
                        static_cast<double>(stats->bytes_read));
                    metrics.emplace_back(
                        "engram_read_calls",
                        static_cast<double>(stats->read_calls));
                    metrics.emplace_back(
                        "engram_io_seconds",
                        stats->io_seconds);
                    metrics.emplace_back(
                        "engram_resident_rows",
                        static_cast<double>(stats->resident_rows));
                    metrics.emplace_back(
                        "engram_resident_bytes",
                        static_cast<double>(
                            stats->resident_payload_bytes));
                    metrics.emplace_back(
                        "engram_cache_limit_bytes",
                        static_cast<double>(stats->cache_limit_bytes));
                }
            }
        }
        if constexpr (requires(Runtime& value) {
                value.fused_hyper_connections_active();
            }) {
            if (lock.owns_lock() && runtime_holder->has_value()) {
                metrics.emplace_back(
                    "fused_hyper_connections",
                    runtime_holder->value()
                            .fused_hyper_connections_active()
                        ? 1.0
                        : 0.0);
            }
        }
        return metrics;
    }

    void warm_runtime() {
        mfq::metal::MlxResidentBudgetScope budget_scope([this](std::size_t bytes) {
            enforce_memory_budget(bytes);
        }, [this](std::size_t bytes) { return optional_memory_fits(bytes); });
        enforce_memory_budget(0);
        auto& runtime = runtime_holder->value();
        int warmup_chunk = prefill_chunk_size;
        if constexpr (std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm>) {
            if (const auto stats = runtime.qsa_kv_offload_stats()) {
                const auto& config = runtime.config();
                const auto layers = std::count(config.layer_types.begin(), config.layer_types.end(), "full_attention") +
                    config.mtp_num_hidden_layers;
                const auto row_bytes = static_cast<std::size_t>(layers) *
                    (config.num_key_value_heads * config.head_dim * 4 +
                        config.indexer_head_dim * 2 +
                        (config.indexer_head_dim * 4 + config.indexer_compress_ratio - 1) / config.indexer_compress_ratio);
                warmup_chunk = static_cast<int>(std::max<std::size_t>(1,
                    std::min<std::size_t>(warmup_chunk, stats->hot_limit / std::max<std::size_t>(1, row_bytes))));
            }
        }
        if constexpr (requires { runtime.prepare_mtp_ttt(default_prefix_cache_directory()); })
            runtime.prepare_mtp_ttt(default_prefix_cache_directory());
        const auto generate = [&](const std::vector<std::int64_t>& prompt,
                                  const mfq::metal::MlxSamplingParams& sampling, int count) {
            generate_with_prefill_metrics(runtime, prompt, sampling, count,
                [](std::int64_t) { return true; }, {}, {}, {}, prefill_chunk_size);
        };
        const auto reset = [&] {
            if constexpr (requires { runtime.reset_generation_state(); }) {
                runtime.reset_generation_state();
            } else {
                mfq::metal::MlxSamplingParams sampling;
                sampling.enable_mtp = false;
                generate({warmup_prompt.front()}, sampling, 0);
                if constexpr (requires { runtime.clear_cache(); }) runtime.clear_cache();
                else runtime.reset();
            }
            release_model_load_staging_memory(runtime_stream);
        };
        mfq::metal::warm_mlx_inference(warmup_prompt, static_cast<int>(*loaded_context),
            warmup_chunk, preparation_sampling, runtime_components.mtp_available,
            generate, reset);
    }

    std::shared_ptr<std::mutex> runtime_mutex;
    std::shared_ptr<std::optional<Runtime>> runtime_holder;
    std::shared_ptr<Cache> session_cache;
    Loader load_runtime;
    PagedCacheFactory paged_cache_factory;
    std::shared_ptr<std::int64_t> loaded_context;
    mlx::core::Stream runtime_stream;
    int prefill_chunk_size;
    std::size_t allocator_cache_limit;
    mfq::metal::MlxSamplingParams preparation_sampling;
    int preparation_routes;
    int preparation_vocab;
    std::vector<std::int64_t> warmup_prompt;
    std::size_t resident_weight_baseline = 0;
    std::size_t resident_dense_weight_baseline = 0;
    std::atomic<std::size_t> resident_budget{0};
    std::atomic<std::size_t> resident_weight_budget{0};
    mfq::metal::MlxEngineComponents runtime_components;
    std::unique_ptr<mfq::metal::MlxWeightResidency> weight_residency;
    using MetalBatch = mfq::engine::ContinuousBatch<mfq::metal::MlxQwen4BatchOperations>;
    int configured_batch_capacity;
    std::size_t batching_capacity = 1;
    bool batch_mtp_enabled = false;
    std::unique_ptr<MetalBatch> batching;
};

template <typename Runtime, typename Loader>
int run_loaded_runtime(
    const Arguments& arguments,
    const mfq::metal::MfqContainer& container,
    Runtime model,
    Loader load_runtime,
    std::string model_type,
    std::int64_t maximum_context,
    std::int64_t vocabulary_size,
    mlx::core::Stream runtime_stream) {
    container.stop_load_observation();
    std::cerr << "mfq_load_progress stage=finalizing" << std::endl;
    constexpr const char* tokenizer_asset =
        "__mfq_asset__/tokenizer.gguf";
    if constexpr (requires(Runtime& value) {
            value.prewarm_ssd_expert_arena();
        }) {
        model.prewarm_ssd_expert_arena();
        release_model_load_staging_memory(runtime_stream);
    }
    int prefill_chunk_size = arguments.prefill_chunk_size;
    if (!arguments.prefill_chunk_size_explicit &&
        prefill_autotune_enabled()) {
        if constexpr (std::is_same_v<Runtime, mfq::metal::MlxQwen35CausalLm> ||
            std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm>) {
            prefill_chunk_size = static_cast<int>(mfq::metal::mlx_prefix_block_size(true,
                prefill_chunk_size, physical_memory_bytes(), metal_prefix_nax_device(),
                std::is_same_v<Runtime, mfq::metal::MlxQwen4CausalLm>));
        }
        if constexpr (requires(const Runtime& value) {
                value.preferred_prefill_chunk_size(prefill_chunk_size);
            }) {
            const int preferred = model.preferred_prefill_chunk_size(
                prefill_chunk_size);
            if (preferred > 0 && preferred != prefill_chunk_size) {
                prefill_chunk_size = preferred;
                std::cout
                    << "Metal prefill autotune: chunk_size="
                    << prefill_chunk_size << std::endl;
            }
        }
    }
    const auto allocator_cache_limit = server_cache_limit_bytes();
    mlx::core::set_cache_limit(allocator_cache_limit);
    MfqHttpRuntimeTransportConfig transport_config;
    transport_config.host = arguments.host;
    transport_config.port = arguments.port;
    transport_config.model_name = arguments.model_name.empty()
        ? arguments.mfq.stem().string()
        : arguments.model_name;
    transport_config.model_type = std::move(model_type);
    const auto model_graph = mfq::metal::effective_model_graph(container);
    transport_config.api_key = arguments.api_key;
    transport_config.max_context = std::min<std::int64_t>(
        arguments.context_size,
        maximum_context);
    transport_config.context_capacity = maximum_context;
    transport_config.vocab_size = vocabulary_size;
    constexpr const char* model_config_asset =
        "__mfq_asset__/model_config.json";
    const auto embedded = [&] {
        const auto found = container.header().extra_json.find(
            "runtime.sampling.v1");
        return found == container.header().extra_json.end()
            ? std::string()
            : found->second;
    }();
    const auto model_config = container.contains(model_config_asset)
        ? container.read_text(model_config_asset)
        : std::string();
    transport_config.runtime_profile = resolve_mfq_runtime_profile(
        arguments.mfq.string(),
        model_graph.architecture,
        transport_config.model_type,
        transport_config.model_name,
        embedded,
        model_config,
        arguments.sampling_profile.string());

    mfq::metal::MlxSamplingParams preparation_sampling;
    const auto& chat_profile = transport_config.runtime_profile.chat;
    preparation_sampling.temperature = chat_profile.temperature.value_or(1.0);
    preparation_sampling.top_k = chat_profile.top_k.value_or(0);
    preparation_sampling.top_p = chat_profile.top_p.value_or(1.0);
    int preparation_routes = 0;
    if (auto* preparation = mfq::metal::MlxKernelPreparation::current()) {
        preparation_routes = preparation->routes();
        preparation->collect_sampling(static_cast<int>(vocabulary_size), preparation_sampling);
        preparation->finish();
    }

    auto runtime_mutex = std::make_shared<std::mutex>();
    auto runtime_holder =
        std::make_shared<std::optional<Runtime>>(
            std::move(model));
    const auto paged_cache_factory =
        [&container, prefill_chunk_size, runtime_holder](std::int64_t context_size)
            -> std::shared_ptr<mfq::cache::PagedPrefixCache> {
            std::string predictor_key;
            if constexpr (requires(Runtime& runtime) { runtime.mtp_cache_fingerprint(); })
                if (runtime_holder->has_value()) predictor_key = runtime_holder->value().mtp_cache_fingerprint();
            return make_metal_paged_cache<Runtime>(
                container, context_size, prefill_chunk_size, predictor_key);
        };
    auto session_cache =
        std::make_shared<MlxServerTextSessionCache<Runtime>>(
            paged_cache_factory(transport_config.max_context));
    auto loaded_context =
        std::make_shared<std::int64_t>(transport_config.max_context);
    auto runtime_components = mfq::metal::make_mlx_engine_components(
        &model_graph,
        runtime_mutex,
        runtime_holder,
        runtime_stream);
    {
        MfqModelCapabilities capabilities;
        capabilities.family = model_graph.architecture;
        capabilities.source = "model-graph";
        capabilities.text = model_graph.has_component("text");
        capabilities.mtp = runtime_components.mtp_available;
        if (const auto* vision = model_graph.component("vision");
            vision != nullptr && runtime_components.multimodal_generate) {
            capabilities.image_input = true;
            capabilities.video_input =
                vision->input_contract == mfq::kMfqGridVisionInputContract ||
                vision->input_contract == "minicpmo45.v1";
        }
        capabilities.audio_input =
            model_graph.has_component("audio_input") &&
            static_cast<bool>(runtime_components.multimodal_generate);
        capabilities.audio_output =
            model_graph.has_component("audio_output");
        capabilities.full_duplex =
            static_cast<bool>(runtime_components.duplex);
        transport_config.model_capabilities = std::move(capabilities);
    }
    auto text = container.contains(tokenizer_asset)
        ? std::make_unique<mfq::engine::TextProcessor>(container.read(tokenizer_asset),
              static_cast<std::int32_t>(vocabulary_size), transport_config.model_type)
        : std::make_unique<mfq::engine::TextProcessor>(arguments.tokenizer_gguf.string(),
              static_cast<std::int32_t>(vocabulary_size), transport_config.model_type);
    mfq::engine::EngineInfo info;
    info.max_context = transport_config.max_context;
    info.vocab_size = static_cast<std::int32_t>(vocabulary_size);
    info.stop_token_ids = mfq::engine::model_stop_token_ids(model_config, info.vocab_size);
    info.model_type = transport_config.model_type;
    info.capabilities = *transport_config.model_capabilities;
    info.multimodal = static_cast<bool>(runtime_components.multimodal_generate);
    info.duplex = static_cast<bool>(runtime_components.duplex);
    info.reload = true;
    info.probability = requires(Runtime& runtime, mlx::core::array ids) { runtime.score_forward(ids, false); };
    auto reload_runtime = [&container, loader = std::move(load_runtime)](std::int64_t context) {
        container.observe_load_records([](std::size_t completed, std::size_t total) {
            std::cerr << "mfq_load_progress completed=" << completed
                      << " total=" << total << std::endl;
        });
        try {
            auto loaded = loader(context);
            container.stop_load_observation();
            std::cerr << "mfq_load_progress stage=finalizing" << std::endl;
            return loaded;
        } catch (...) {
            container.stop_load_observation();
            throw;
        }
    };
    auto engine_model = std::make_unique<LoadedMetalModel<Runtime, decltype(reload_runtime)>>(
        runtime_mutex, runtime_holder, session_cache, std::move(reload_runtime),
        paged_cache_factory, loaded_context, runtime_stream, prefill_chunk_size,
        allocator_cache_limit, std::move(runtime_components), preparation_sampling,
        preparation_routes, static_cast<int>(vocabulary_size),
        text->tokenizer().tokenize("Explain how computers use memory to run language models. "
            "Describe the roles of model weights, attention, and cached context in clear terms.", false),
        arguments.continuous_batching);
    auto inference_engine = std::make_unique<mfq::metal::MlxEngine>(
        std::move(engine_model), std::move(text), std::move(info));
    auto transport = arguments.stdio
        ? make_mfq_stdio_transport(transport_config)
        : make_mfq_http_transport(std::move(transport_config));
    MfqRuntime runtime(std::move(inference_engine), std::move(transport));
    return runtime.run();
}

int run_native_runtime(
    const Arguments& arguments,
    const mfq::metal::MfqContainer& container) {
    container.observe_load_records([](std::size_t completed, std::size_t total) {
        std::cerr << "mfq_load_progress completed=" << completed
                  << " total=" << total << std::endl;
    });
    constexpr const char* tokenizer_asset =
        "__mfq_asset__/tokenizer.gguf";
    if (!container.contains(tokenizer_asset) &&
        arguments.tokenizer_gguf.empty()) {
        throw std::runtime_error(
            "model has no embedded tokenizer.gguf asset; "
            "pass --tokenizer PATH");
    }

    const auto runtime_stream =
        mlx::core::new_thread_unsafe_stream(
            mlx::core::Device::gpu);
    mlx::core::set_default_stream(runtime_stream);
    mfq::metal::MlxKernelPreparation preparation(static_cast<int>(arguments.context_size),
        arguments.prefill_chunk_size);
    const auto started =
        std::chrono::steady_clock::now();
    const auto graph = mfq::metal::effective_model_graph(container);
    const auto& backbone = graph.backbone;
    const bool native_hf =
        container.header().extra_json.count("source.format") != 0 &&
        container.header().extra_json.at("source.format") ==
            "hf-safetensors";
    if (backbone == "deepseek_v41") {
        const auto config =
            mfq::metal::DeepseekV41Config::from_mfq(container, graph);
        preparation.set_routes(static_cast<int>(config.top_k));
        const int context = static_cast<int>(
            std::min<std::int64_t>(
                arguments.context_size,
                config.max_position_embeddings));
        std::optional<std::size_t> expert_cache_bytes;
        if (arguments.expert_cache_gb.has_value() || native_hf) {
            const auto bytes = requested_cache_bytes(
                arguments.expert_cache_gb, native_hf, container);
            if (bytes > 0) expert_cache_bytes = bytes;
        }
        std::cout
            << "Loading native C++/MLX DeepSeek-V4.1 model "
               "on Apple GPU..."
            << std::endl;
        auto runtime =
            mfq::metal::MlxDeepseekV41CausalLm::load(
                container, context, expert_cache_bytes);
        release_model_load_staging_memory(runtime_stream);
        const auto load_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
        std::cout
            << "Loaded " << runtime.layer_count()
            << " DeepSeek-V4.1 layers in "
            << load_seconds << " s; runtime=native-cpp"
            << " expert_backing="
            << (expert_cache_bytes.has_value()
                    ? "ssd-streamed"
                    : "full-resident")
            << std::endl;
        const auto load_runtime =
            [&container, expert_cache_bytes](std::int64_t requested_context) {
                if (requested_context < 1 ||
                    requested_context >
                        std::numeric_limits<int>::max()) {
                    throw std::invalid_argument(
                        "Metal runtime context is out of range");
                }
                return mfq::metal::MlxDeepseekV41CausalLm::load(
                    container,
                    static_cast<int>(requested_context),
                    expert_cache_bytes);
            };
        return run_loaded_runtime(
            arguments,
            container,
            std::move(runtime),
            load_runtime,
            config.has_vision()
                ? std::string("deepseek_v41_vision")
                : config.text_model_type,
            config.max_position_embeddings,
            config.vocab,
            runtime_stream);
    }
    if (backbone == "deepseek_v4") {
        const auto config =
            mfq::metal::DeepseekV4Config::from_mfq(
                container);
        preparation.set_routes(static_cast<int>(config.top_k));
        const int context = static_cast<int>(
            std::min<std::int64_t>(
                arguments.context_size,
                config.max_position_embeddings));
        std::optional<std::size_t> expert_cache_bytes;
        if (arguments.expert_cache_gb.has_value() || native_hf) {
            const auto bytes = requested_cache_bytes(
                arguments.expert_cache_gb, native_hf, container);
            if (bytes > 0) expert_cache_bytes = bytes;
        }
        std::cout
            << "Loading native C++/MLX DeepSeek-V4 model "
               "on Apple GPU..."
            << std::endl;
        auto runtime =
            mfq::metal::MlxDeepseekV4CausalLm::load(
                container,
                context,
                expert_cache_bytes);
        release_model_load_staging_memory(runtime_stream);
        const auto load_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                started).count();
        std::cout
            << "Loaded " << runtime.layer_count()
            << " DeepSeek-V4 layers in "
            << load_seconds << " s";
        if (expert_cache_bytes.has_value()) {
            std::cout
                << " mfe_load=disk-cache cache_gb="
                << static_cast<double>(*expert_cache_bytes) /
                    static_cast<double>(std::uint64_t{1} << 30);
        } else {
            std::cout << " mfe_load=full-resident";
        }
        std::cout << std::endl;
        const auto load_runtime =
            [&container, expert_cache_bytes](
                std::int64_t requested_context) {
                if (requested_context < 1 ||
                    requested_context >
                        std::numeric_limits<int>::max()) {
                    throw std::invalid_argument(
                        "Metal runtime context is out of range");
                }
                return mfq::metal::
                    MlxDeepseekV4CausalLm::load(
                        container,
                        static_cast<int>(requested_context),
                        expert_cache_bytes);
            };
        return run_loaded_runtime(
            arguments,
            container,
            std::move(runtime),
            load_runtime,
            config.has_vision()
                ? std::string("deepseek_v4_vision")
                : config.model_type,
            config.max_position_embeddings,
            config.vocab,
            runtime_stream);
    }

    if (backbone == "nemotron_h") {
        const auto config = mfq::metal::load_nemotron_h_config(container);
        const auto components = mfq::metal::resolve_nemotron_h_components(container);
        preparation.set_routes(static_cast<int>(config.num_experts_per_tok));
        const int context = static_cast<int>(std::min<std::int64_t>(
            arguments.context_size, config.max_position_embeddings));
        std::cout
            << "Loading native C++/MLX Nemotron-H model on Apple GPU..."
            << std::endl;
        auto runtime = mfq::metal::MlxNemotronHCausalLm::load(container, context);
        release_model_load_staging_memory(runtime_stream);
        const auto load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout
            << "Loaded " << runtime.layer_count()
            << " Nemotron-H layers in " << load_seconds << " s"
            << " mtp=" << (runtime.supports_mtp() ? "on" : "off")
            << " vision=" << (runtime.supports_multimodal() ? "on" : "off")
            << std::endl;
        const auto load_runtime =
            [&container, native_context = config.max_position_embeddings](
                std::int64_t requested_context) {
                if (requested_context < 1 ||
                    requested_context > std::numeric_limits<int>::max()) {
                    throw std::invalid_argument(
                        "Metal runtime context is out of range");
                }
                return mfq::metal::MlxNemotronHCausalLm::load(
                    container,
                    std::min<std::int64_t>(requested_context, native_context));
            };
        return run_loaded_runtime(
            arguments,
            container,
            std::move(runtime),
            load_runtime,
            components.has_vision ? std::string("nemotron_h_vision")
                                  : config.model_type,
            config.max_position_embeddings,
            config.vocab_size,
            runtime_stream);
    }

    if (backbone == "minicpmo45") {
        std::cout
            << "Loading native C++/MLX MiniCPM-o 4.5 Qwen3-8B "
               "on Apple GPU..."
            << std::endl;
        auto runtime =
            mfq::metal::MlxMiniCPMO45Runtime::load(
                container,
                arguments.context_size,
                arguments.server);
        release_model_load_staging_memory(runtime_stream);
        mlx::core::set_cache_limit(kMinicpmoDuplexCacheLimitBytes);
        const auto load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        const auto maximum_context = runtime.maximum_context();
        const auto vocabulary = runtime.vocabulary_size();
        std::cout
            << "Loaded " << runtime.layer_count()
            << " MiniCPM-o Qwen3-8B layers in "
            << load_seconds << " s"
            << std::endl;
        const auto load_runtime =
            [&container,
             load_modalities = arguments.server](
                std::int64_t requested_context) {
                return mfq::metal::MlxMiniCPMO45Runtime::load(
                    container,
                    requested_context,
                    load_modalities);
            };
        return run_loaded_runtime(
            arguments,
            container,
            std::move(runtime),
            load_runtime,
            "minicpmo",
            maximum_context,
            vocabulary,
            runtime_stream);
    }

    if (backbone == "qwen4_exp") {
        const auto config =
            mfq::metal::Qwen4Config::from_mfq(container);
        const auto extended_context = mfq::metal::qwen_yarn_capacity(config.max_position_embeddings, config.yarn_max_factor);
        preparation.set_routes(static_cast<int>(config.num_experts_per_tok));
        const int context = static_cast<int>(
            std::min<std::int64_t>(
                arguments.context_size,
                arguments.yarn_context_extension ? extended_context : config.max_position_embeddings));
        auto runtime_arguments = arguments;
        runtime_arguments.context_size = context;
        std::optional<std::size_t> expert_cache_bytes;
        if (arguments.expert_cache_gb.has_value() || native_hf) {
            const auto bytes = requested_cache_bytes(
                arguments.expert_cache_gb, native_hf, container);
            if (bytes > 0) expert_cache_bytes = bytes;
        }
        std::cout
            << "Loading native C++/MLX Qwen4-Exp model "
               "on Apple GPU..."
            << std::endl;
        auto runtime =
            mfq::metal::MlxQwen4CausalLm::load(
                container, context, expert_cache_bytes);
        release_model_load_staging_memory(runtime_stream);
        const auto load_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout
            << "Loaded " << runtime.layer_count()
            << " Qwen4-Exp layers in "
            << load_seconds << " s; runtime=native-cpp"
            << " expert_backing="
            << (expert_cache_bytes.has_value()
                    ? "ssd-streamed"
                    : "full-resident")
            << std::endl;
        const auto load_runtime =
            [&container, expert_cache_bytes](
                std::int64_t requested_context) {
                if (requested_context < 1 ||
                    requested_context > std::numeric_limits<int>::max()) {
                    throw std::invalid_argument(
                        "Metal runtime context is out of range");
                }
                return mfq::metal::MlxQwen4CausalLm::load(
                    container,
                    static_cast<int>(requested_context),
                    expert_cache_bytes);
            };
        return run_loaded_runtime(
            runtime_arguments,
            container,
            std::move(runtime),
            load_runtime,
            config.text_model_type.empty()
                ? config.model_type
                : config.text_model_type,
            extended_context,
            config.vocab_size,
            runtime_stream);
    }

    if (backbone != "qwen3_5") {
        throw std::runtime_error(
            "unsupported native Metal backbone implementation: " + backbone);
    }

    const auto config =
        mfq::metal::Qwen35Config::from_mfq(container);
    const auto extended_context = mfq::metal::qwen_yarn_capacity(config.max_position_embeddings, config.yarn_max_factor);
    const auto context = static_cast<int>(std::min(arguments.context_size,
        arguments.yarn_context_extension ? extended_context : config.max_position_embeddings));
    auto runtime_arguments = arguments;
    runtime_arguments.context_size = context;
    std::cout
        << "Loading native C++/MLX Qwen3.5 model "
           "on Apple GPU..."
        << std::endl;
    auto runtime =
        mfq::metal::MlxQwen35CausalLm::load(container, context);
    release_model_load_staging_memory(runtime_stream);
    const auto load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::cout
        << "Loaded " << runtime.layer_count()
        << " Qwen3.5 layers in "
        << load_seconds << " s; native_mtp="
        << (runtime.supports_mtp() ? "sampling" : "disabled")
        << std::endl;
    const auto load_runtime =
        [&container](std::int64_t requested_context) {
            return mfq::metal::
                MlxQwen35CausalLm::load(container, static_cast<int>(requested_context));
        };
    return run_loaded_runtime(
        runtime_arguments,
        container,
        std::move(runtime),
        load_runtime,
        config.text_model_type.empty()
            ? config.model_type
            : config.text_model_type,
        extended_context,
        config.vocab_size,
        runtime_stream);
}
#endif

} // namespace

int main(int argc, char** argv) {
    mfq::install_fatal_diagnostics();
    try {
        const auto arguments = parse_arguments(argc, argv);
#ifdef MFQ_METAL_RUNTIME_COMMUNICATION
        if (arguments.stdio) {
            prepare_mfq_stdio_transport();
        }
#endif
        mfq::metal::set_mlx_predequantize_fp16(
            arguments.predequantize_fp16);
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
                usage_error("--model is required");
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
        if (!arguments.benchmark_mfq_moe_prefix.empty()) {
            configure_mlx_metal();
            return benchmark_native_mfq_moe(arguments, model);
        }
        if (!arguments.tensor.empty()) {
            configure_mlx_metal();
            const auto& record = model.record(arguments.tensor);
            if (record.dtype == "MFE") {
                auto weight =
                    mfq::metal::MlxMfeWeight::from_blob(
                        model.read(arguments.tensor));
                if (!arguments.benchmark_concat_tensors.empty()) {
                    std::vector<mfq::metal::MlxMfeWeight> projections{weight};
                    for (const auto& name : arguments.benchmark_concat_tensors) {
                        if (model.record(name).dtype != "MFE") {
                            usage_error("--benchmark-concat-tensor requires MFE records");
                        }
                        projections.push_back(mfq::metal::MlxMfeWeight::from_blob(
                            model.read(name)));
                    }
                    weight = mfq::metal::MlxMfeWeight::concatenate_projections(projections)
                        .materialize_packed_projections();
                }
                if (arguments.benchmark_distributed_routes > 0
                    && !arguments.benchmark_experts.empty()) {
                    usage_error(
                        "choose either --benchmark-experts or "
                        "--benchmark-distributed-experts");
                }
                const int routes = arguments.benchmark_distributed_routes > 0
                    ? arguments.benchmark_distributed_routes
                    : arguments.benchmark_experts.empty()
                    ? std::min(6, weight.experts())
                    : static_cast<int>(
                          arguments.benchmark_experts.size());
                if (routes > weight.experts()) {
                    usage_error(
                        "MFE benchmark routes exceed expert count");
                }
                std::vector<float> input_values(
                    static_cast<std::size_t>(arguments.benchmark_tokens)
                        * static_cast<std::size_t>(
                              arguments.benchmark_routed_input ? routes : 1)
                        * static_cast<std::size_t>(weight.neuron_len()));
                for (std::size_t index = 0;
                     index < input_values.size();
                     ++index) {
                    input_values[index] = static_cast<float>(
                        static_cast<int>(index % 17) - 8) / 16.0f;
                }
                const mlx::core::Shape input_shape = arguments.benchmark_routed_input
                    ? mlx::core::Shape{arguments.benchmark_tokens, routes, weight.neuron_len()}
                    : mlx::core::Shape{arguments.benchmark_tokens, weight.neuron_len()};
                auto input = mlx::core::astype(
                    mlx::core::array(
                        input_values.begin(),
                        input_shape),
                    mlx::core::float16);
                std::vector<std::int32_t> expert_values =
                    arguments.benchmark_experts;
                if (expert_values.empty()
                    && arguments.benchmark_distributed_routes == 0) {
                    expert_values.resize(static_cast<std::size_t>(routes));
                    for (int route = 0; route < routes; ++route) {
                        expert_values[static_cast<std::size_t>(route)] =
                            route;
                    }
                }
                for (const auto expert : expert_values) {
                    if (expert >= weight.experts()) {
                        usage_error(
                            "--benchmark-experts contains an out-of-range "
                            "expert ID");
                    }
                }
                std::vector<std::int32_t> routed_experts(
                    static_cast<std::size_t>(arguments.benchmark_tokens)
                        * static_cast<std::size_t>(routes));
                for (int token = 0;
                     token < arguments.benchmark_tokens;
                     ++token) {
                    for (int route = 0; route < routes; ++route) {
                        routed_experts[static_cast<std::size_t>(
                            token * routes + route)] =
                            arguments.benchmark_distributed_routes > 0
                            ? (token * 131 + route * 97) % weight.experts()
                            : expert_values[static_cast<std::size_t>(route)];
                    }
                }
                auto expert_ids = mlx::core::array(
                    routed_experts.begin(),
                    mlx::core::Shape{
                        arguments.benchmark_tokens,
                        routes,
                    });

                auto execute = [&] {
                    return arguments.benchmark_swiglu
                        ? weight.routed_swiglu(input, expert_ids)
                        : weight(input, expert_ids);
                };
                auto warm = execute();
                warm.eval();
                const auto started = std::chrono::steady_clock::now();
                mlx::core::array output = warm;
                std::vector<mlx::core::array> pending;
                pending.reserve(std::min(arguments.benchmark_batch_size, arguments.benchmark_reps));
                for (int rep = 0;
                     rep < arguments.benchmark_reps;
                    ++rep) {
                    output = execute();
                    pending.push_back(output);
                    if (pending.size() == static_cast<std::size_t>(arguments.benchmark_batch_size) ||
                        rep + 1 == arguments.benchmark_reps) {
                        mlx::core::eval(pending);
                        pending.clear();
                    }
                }
                const auto elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started)
                        .count();
                auto checked = mlx::core::astype(
                    output, mlx::core::float32);
                checked.eval();
                const auto* values = checked.data<float>();
                float maximum = 0.0f;
                double checksum = 0.0;
                double l1 = 0.0;
                double l2 = 0.0;
                for (std::size_t index = 0;
                     index < checked.size();
                     ++index) {
                    if (!std::isfinite(values[index])) {
                        throw std::runtime_error(
                            "Metal MFE smoke test returned non-finite data");
                    }
                    maximum = std::max(
                        maximum, std::fabs(values[index]));
                    checksum += static_cast<double>(values[index])
                        * static_cast<double>(index % 251u + 1u);
                    l1 += std::fabs(static_cast<double>(values[index]));
                    l2 += static_cast<double>(values[index])
                        * static_cast<double>(values[index]);
                }
                if (maximum <= 1e-12f) {
                    throw std::runtime_error(
                        "Metal MFE smoke test unexpectedly returned all zero");
                }
                std::cout
                    << "Metal MFE smoke test passed"
                    << " experts=" << weight.experts()
                    << " tokens=" << arguments.benchmark_tokens
                    << " routes=" << routes
                    << " distributed="
                    << static_cast<int>(
                           arguments.benchmark_distributed_routes > 0)
                    << " in=" << weight.neuron_len()
                    << " out=" << weight.out_per_expert()
                    << " projections=" << weight.projections()
                    << " swiglu="
                    << static_cast<int>(
                           arguments.benchmark_swiglu)
                    << " packed=" << weight.packed_nbytes()
                    << " reps=" << arguments.benchmark_reps
                    << " batch_size=" << arguments.benchmark_batch_size
                    << " routed_input=" << static_cast<int>(arguments.benchmark_routed_input)
                    << " ms_per_dispatch="
                    << elapsed_ms / arguments.benchmark_reps
                    << " checksum=" << checksum
                    << " max=" << maximum
                    << " l1=" << l1
                    << " l2=" << l2
                    << "\n";
                return EXIT_SUCCESS;
            }
            if (!arguments.benchmark_concat_tensors.empty() ||
                arguments.benchmark_routed_input || arguments.benchmark_batch_size != 1) {
                usage_error("MFE benchmark options require an MFE --tensor record");
            }
            const auto weight = mfq::metal::MlxLinear::load(
                model,
                arguments.tensor);
            std::vector<float> input_values(
                static_cast<std::size_t>(arguments.benchmark_tokens)
                    * static_cast<std::size_t>(weight.input_size()));
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
                mlx::core::Shape{
                    arguments.benchmark_tokens,
                    weight.input_size(),
                });
            input = mlx::core::astype(input, mlx::core::float16);
            auto execute = [&] {
                return weight(input);
            };
            auto warm = execute();
            warm.eval();
            const auto started = std::chrono::steady_clock::now();
            mlx::core::array output = warm;
            for (int rep = 0;
                 rep < arguments.benchmark_reps;
                 ++rep) {
                output = execute();
                output.eval();
            }
            const auto elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started)
                    .count();
            auto checked = mlx::core::astype(
                output,
                mlx::core::float32);
            checked.eval();
            const auto* values = checked.data<float>();
            float maximum = 0.0f;
            for (std::size_t index = 0; index < checked.size(); ++index) {
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
                << " tokens=" << arguments.benchmark_tokens
                << " in=" << weight.input_size()
                << " out=" << weight.output_size()
                << " packed=" << (weight.packed() ? "true" : "false")
                << " reps=" << arguments.benchmark_reps
                << " ms_per_dispatch="
                << elapsed_ms / arguments.benchmark_reps
                << "\n";
        }
        if (arguments.server) {
            configure_mlx_metal();
#ifdef MFQ_METAL_RUNTIME_COMMUNICATION
            return run_native_runtime(arguments, model);
#else
            throw std::runtime_error(
                "this build has no runtime communication support; configure with "
                "-DMFQ_BUILD_RUNTIME_COMMUNICATION=ON");
#endif
        }
        if (!arguments.check_container &&
            !arguments.list_tensors &&
            arguments.benchmark_mfq_moe_prefix.empty() &&
            arguments.tensor.empty() &&
            !arguments.self_test_metal &&
            !arguments.server) {
            usage_error(
                "select --check-mfq-container, --list-tensors, --tensor, "
                "--benchmark-mfq-moe, --transport, or --self-test-metal");
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "mfq-decode-metal: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
