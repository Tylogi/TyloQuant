#include "mfq_container.h"
#include "mlx_deepseek_v4_causal_lm.h"
#include "mlx_minicpmo45.h"
#include "mlx_moe.h"
#include "mlx_qwen35_causal_lm.h"
#include "mlx_tensor.h"
#include "qwen35_model.h"

#include "../json/nlohmann/json.hpp"

#ifdef MFQ_METAL_SERVER
#include "../mfq_server.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

#include <mach-o/dyld.h>
#include <malloc/malloc.h>
#include <sys/sysctl.h>

namespace {

constexpr std::size_t kMinicpmoDuplexCacheLimitBytes =
    std::size_t{8} << 30;

void release_model_load_staging_memory() {
    // Model conversion and NINTM repacking leave large, now-unused buffers in
    // both the MLX cache and macOS malloc's large-object depot. Keeping those
    // pages makes a single fully resident model look tens of GiB larger and
    // can force useful weights into swap before the first request.
    mlx::core::clear_cache();
    malloc_zone_pressure_relief(nullptr, 0);
}

struct Arguments {
    std::filesystem::path mfq;
    std::string tensor;
    int benchmark_reps = 1;
    std::vector<std::int32_t> benchmark_experts;
    bool benchmark_swiglu = false;
    bool check_container = false;
    bool list_tensors = false;
    bool self_test_metal = false;
    bool server = false;
    bool predequantize_fp16 = false;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::int64_t context_size = 32768;
    int prefill_chunk_size = 2048;
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
        if (value == "--mfq") {
            result.mfq = require_value("--mfq");
        } else if (value == "--tensor") {
            result.tensor = require_value("--tensor");
        } else if (value == "--benchmark-swiglu") {
            result.benchmark_swiglu = true;
        } else if (value == "--benchmark-reps") {
            const auto parsed = std::stoll(
                require_value("--benchmark-reps"));
            if (parsed <= 0 || parsed > 10000) {
                usage_error(
                    "--benchmark-reps must be in [1, 10000]");
            }
            result.benchmark_reps = static_cast<int>(parsed);
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
        } else if (value == "--check-mfq-container") {
            result.check_container = true;
        } else if (value == "--list-tensors") {
            result.list_tensors = true;
        } else if (value == "--self-test-metal") {
            result.self_test_metal = true;
        } else if (value == "--server") {
            result.server = true;
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
        } else if (value == "--tokenizer-gguf") {
            result.tokenizer_gguf =
                require_value("--tokenizer-gguf");
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
        << "  mfq-decode-metal --mfq MODEL.mfq --check-mfq-container\n"
        << "  mfq-decode-metal --mfq MODEL.mfq --list-tensors\n"
        << "  mfq-decode-metal --mfq MODEL.mfq --tensor NAME\n"
        << "  mfq-decode-metal --mfq MODEL.mfq --server "
           "[--host 127.0.0.1 --port 8080]\n"
        << "  mfq-decode-metal --mfq HF_MODEL_DIR --server "
           "--tokenizer-gguf TOKENIZER.gguf\n"
        << "  mfq-decode-metal --self-test-metal\n\n"
        << "Options:\n"
        << "  --mfq PATH             MFQ model/shard, or a supported HF directory\n"
        << "  --check-mfq-container  validate headers, records, and shard set\n"
        << "  --list-tensors         print record dtype, bytes, and name\n"
        << "  --tensor NAME          load and execute one supported linear weight\n"
        << "  --benchmark-reps N     timed executions for --tensor (default 1)\n"
        << "  --benchmark-experts L  comma-separated NINTM expert IDs\n"
        << "  --benchmark-swiglu     fuse an even-width NINTM gate/up record\n"
        << "  --self-test-metal      execute an MLX C++ graph on Metal\n"
        << "  --server               run the native C++ OpenAI-compatible server\n"
        << "  --metal-predequantize-f16\n"
        << "                          expand regular weights to FP16 at load time\n"
        << "  --host ADDRESS         server bind address (default 127.0.0.1)\n"
        << "  --port PORT            server port (default 8080)\n"
        << "  --ctx-size TOKENS      runtime/API context limit (default 32768)\n"
        << "  --prefill-chunk-size N maximum prompt chunk (default 2048)\n"
        << "  --moe-gpu-cache-gb N   unified-memory hot-expert cache\n"
        << "                          MFQ default: full residency; HF default: auto\n"
        << "  --model-name NAME      API model name (default MFQ filename)\n"
        << "  --api-key KEY          optional bearer token\n"
        << "  --tokenizer-gguf PATH  external tokenizer GGUF when not embedded\n"
        << "  --sampling-profile P  explicit runtime sampling profile JSON\n";
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open file: " + path.string());
    }
    std::ostringstream output;
    output << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("cannot read file: " + path.string());
    }
    return output.str();
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

std::size_t requested_cache_bytes(
    const std::optional<double>& cache_gb,
    bool hf_streaming) {
    constexpr long double bytes_per_gib =
        static_cast<long double>(std::uint64_t{1} << 30);
    if (cache_gb.has_value()) {
        const long double requested =
            static_cast<long double>(*cache_gb) * bytes_per_gib;
        if (requested >
            static_cast<long double>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(
                "DeepSeek-V4 expert cache exceeds addressable memory");
        }
        const auto bytes = static_cast<std::size_t>(requested);
        if (hf_streaming && bytes == 0) {
            throw std::runtime_error(
                "HF SSD expert streaming requires a non-zero cache");
        }
        return bytes;
    }
    if (!hf_streaming) {
        return 0;
    }
    // Leave one third of UMA for dense weights, KV, the OS, and request
    // staging. On a 128-GiB machine this gives the expert LRU about 85 GiB.
    return std::max<std::size_t>(
        std::uint64_t{1} << 30,
        physical_memory_bytes() * 2 / 3);
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
template <typename Runtime>
class MlxServerTextSessionCache {
private:
    using SessionState = decltype(
        std::declval<const Runtime&>().capture_text_session_state(
            std::declval<const std::vector<std::int64_t>&>()));

    struct Entry {
        SessionState state;
        std::uint64_t last_used = 0;
    };

public:
    MlxServerTextSessionCache() {
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
    }

    std::size_t restore_best(
        Runtime& runtime,
        const std::string& requested_session,
        const std::vector<std::int64_t>& prompt,
        std::size_t maximum_prefix_tokens) {
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
                if (tokens.empty() || tokens.size() >= prompt.size() ||
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
            std::cerr
                << "server_session_cache backend=metal action=invalidate "
                << "session=" << selected_session
                << " error=" << error.what() << std::endl;
            return 0;
        }
    }

    void store(const std::string& session_id, SessionState state) {
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
        Entry entry{std::move(state), ++clock_};
        const auto protected_clock = entry.last_used;
        auto& history = states_[session_id];
        auto previous = std::find_if(
            history.begin(), history.end(),
            [&](const Entry& saved) {
                return saved.state.tokens == entry.state.tokens;
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

    std::size_t clear() noexcept {
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

private:
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
    const MfqTokenCallback& callback,
    const MfqPrefillCallback& on_prefill,
    const MfqPromptCachePlan& cache_plan,
    const MfqTokenConstraintPtr& token_constraint,
    int) {
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
            : std::nullopt);
}

mlx::core::Shape checked_mlx_shape(
    const std::vector<std::int64_t>& dimensions,
    const char* name) {
    mlx::core::Shape shape;
    shape.reserve(dimensions.size());
    for (const auto dimension : dimensions) {
        if (dimension <= 0 ||
            dimension > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(
                std::string(name) + " has an invalid MLX dimension");
        }
        shape.push_back(static_cast<int>(dimension));
    }
    return shape;
}

std::int32_t generate_with_prefill_metrics(
    mfq::metal::MlxMiniCPMO45Runtime& runtime,
    const std::vector<std::int64_t>& prompt,
    const mfq::metal::MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const MfqTokenCallback& callback,
    const MfqPrefillCallback& on_prefill,
    const MfqPromptCachePlan& cache_plan,
    const MfqTokenConstraintPtr& token_constraint,
    int) {
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
            : std::nullopt);
}

std::int32_t generate_with_prefill_metrics(
    mfq::metal::MlxDeepseekV4CausalLm& runtime,
    const std::vector<std::int64_t>& prompt,
    const mfq::metal::MlxSamplingParams& sampling,
    std::int32_t max_tokens,
    const MfqTokenCallback& callback,
    const MfqPrefillCallback& on_prefill,
    const MfqPromptCachePlan& cache_plan,
    const MfqTokenConstraintPtr& token_constraint,
    int prefill_chunk_size) {
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

template <typename Runtime, typename Loader>
int serve_loaded_runtime(
    const Arguments& arguments,
    const mfq::metal::MfqContainer* container,
    Runtime runtime,
    Loader load_runtime,
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
    if (container != nullptr && container->contains(tokenizer_asset)) {
        server.tokenizer_gguf =
            container->read(tokenizer_asset);
    } else {
        server.tokenizer_model =
            arguments.tokenizer_gguf.string();
    }
    server.api_key = arguments.api_key;
    server.max_context = std::min<std::int64_t>(
        arguments.context_size,
        maximum_context);
    server.context_capacity = maximum_context;
    server.vocab_size = vocabulary_size;
    constexpr const char* model_config_asset =
        "__mfq_asset__/model_config.json";
    const auto embedded = container == nullptr
        ? std::string()
        : [&] {
              const auto found = container->header().extra_json.find(
                  "runtime.sampling.v1");
              return found == container->header().extra_json.end()
                  ? std::string()
                  : found->second;
          }();
    const auto architecture = container == nullptr
        ? std::string("deepseek_v4")
        : container->header().architecture;
    const auto model_config = container == nullptr
        ? read_text(arguments.mfq / "config.json")
        : container->contains(model_config_asset)
        ? container->read_text(model_config_asset)
        : std::string();
    server.runtime_profile = resolve_mfq_runtime_profile(
        arguments.mfq.string(),
        architecture,
        server.model_type,
        server.model_name,
        embedded,
        model_config,
        arguments.sampling_profile.string());

    auto runtime_mutex = std::make_shared<std::mutex>();
    auto runtime_holder =
        std::make_shared<std::optional<Runtime>>(
            std::move(runtime));
    auto session_cache =
        std::make_shared<MlxServerTextSessionCache<Runtime>>();
    auto loaded_context =
        std::make_shared<std::int64_t>(server.max_context);
    const MfqGenerateFn generate =
        [runtime_mutex, runtime_holder, session_cache, runtime_stream,
         prefill_chunk_size = arguments.prefill_chunk_size](
            const std::vector<std::int64_t>& prompt,
            const MfqSamplingParams& sampling,
            const MfqTokenCallback& callback,
            const MfqPrefillCallback& on_prefill,
            const MfqPromptCachePlan& cache_plan,
            const MfqTokenConstraintPtr& token_constraint) {
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
            parameters.seed = sampling.seed;
            auto& loaded_runtime = runtime_holder->value();
            const auto stable_prefix_tokens = std::min(
                cache_plan.stable_prefix_tokens, prompt.size());
            const bool session_enabled =
                !cache_plan.session_id.empty() &&
                stable_prefix_tokens > 0 &&
                loaded_runtime.supports_text_session_state();
            if (session_enabled) {
                (void)session_cache->restore_best(
                    loaded_runtime,
                    cache_plan.session_id,
                    prompt,
                    stable_prefix_tokens);
            }
            const auto generated = generate_with_prefill_metrics(
                loaded_runtime,
                prompt,
                parameters,
                sampling.max_tokens,
                callback,
                on_prefill,
                cache_plan,
                token_constraint,
                prefill_chunk_size);
            if (session_enabled &&
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
            return generated;
        };
    MfqMultimodalGenerateFn multimodal_generate;
    if constexpr (std::is_same_v<
            Runtime, mfq::metal::MlxMiniCPMO45Runtime>) {
        multimodal_generate =
            [runtime_mutex, runtime_holder, runtime_stream](
                const std::vector<std::int64_t>& prompt,
                const MfqVisionInput& vision,
                const MfqSamplingParams& sampling,
                const MfqTokenCallback& callback,
                const MfqPrefillCallback& on_prefill,
                const MfqTokenConstraintPtr& token_constraint) {
                std::lock_guard<std::mutex> lock(*runtime_mutex);
                if (!runtime_holder->has_value()) {
                    throw std::runtime_error(
                        "MiniCPM-o runtime is unavailable after a failed reload");
                }
                mlx::core::set_default_device(
                    mlx::core::Device::gpu);
                mlx::core::set_default_stream(runtime_stream);

                std::vector<std::int32_t> prompt_ids;
                prompt_ids.reserve(prompt.size());
                for (const auto token : prompt) {
                    if (token < 0 ||
                        token > std::numeric_limits<std::int32_t>::max()) {
                        throw std::invalid_argument(
                            "MiniCPM-o prompt token is out of range");
                    }
                    prompt_ids.push_back(static_cast<std::int32_t>(token));
                }
                mfq::metal::MlxMiniCPMO45Inputs inputs{
                    mlx::core::array(
                        prompt_ids.begin(),
                        mlx::core::Shape{
                            1, static_cast<int>(prompt_ids.size())},
                        mlx::core::int32),
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                };
                if (!vision.image_bounds.empty()) {
                    inputs.pixel_values = mlx::core::array(
                        vision.pixel_values.begin(),
                        checked_mlx_shape(
                            vision.pixel_shape, "pixel_values"),
                        mlx::core::float32);
                    inputs.patch_mask = mlx::core::astype(
                        mlx::core::array(
                            vision.patch_mask.begin(),
                            checked_mlx_shape(
                                vision.patch_mask_shape, "patch_mask"),
                            mlx::core::uint8),
                        mlx::core::bool_);
                    inputs.target_sizes = mlx::core::array(
                        vision.target_sizes.begin(),
                        checked_mlx_shape(
                            vision.target_sizes_shape, "target_sizes"),
                        mlx::core::int32);
                    inputs.image_bounds = mlx::core::array(
                        vision.image_bounds.begin(),
                        mlx::core::Shape{
                            static_cast<int>(vision.image_bounds.size() / 4), 4},
                        mlx::core::int64);
                }
                if (!vision.audio_bounds.empty()) {
                    inputs.audio_features = mlx::core::array(
                        vision.audio_features.begin(),
                        checked_mlx_shape(
                            vision.audio_features_shape, "audio_features"),
                        mlx::core::float32);
                    inputs.audio_lengths = mlx::core::array(
                        vision.audio_lengths.begin(),
                        mlx::core::Shape{
                            static_cast<int>(vision.audio_lengths.size())},
                        mlx::core::int64);
                    inputs.audio_bounds = mlx::core::array(
                        vision.audio_bounds.begin(),
                        mlx::core::Shape{
                            static_cast<int>(vision.audio_bounds.size() / 4), 4},
                        mlx::core::int64);
                }

                mfq::metal::MlxSamplingParams parameters;
                parameters.temperature = sampling.temperature;
                parameters.top_k = sampling.top_k;
                parameters.top_p = sampling.top_p;
                parameters.presence_penalty = sampling.presence_penalty;
                parameters.frequency_penalty = sampling.frequency_penalty;
                parameters.repetition_penalty = sampling.repetition_penalty;
                parameters.seed = sampling.seed;
                std::function<void(
                    std::size_t, double, double, double)> report_prefill;
                if (on_prefill) {
                    report_prefill = [on_prefill](
                        std::size_t tokens,
                        double llm_ms,
                        double multimodal_ms,
                        double model_ms) {
                        on_prefill(MfqPrefillTiming{
                            tokens, llm_ms, multimodal_ms, model_ms});
                    };
                }
                return runtime_holder->value().generate_multimodal(
                    inputs, parameters, sampling.max_tokens,
                    callback, report_prefill, token_constraint);
            };
    }
    const MfqReloadFn reload =
        [runtime_mutex, runtime_holder, loaded_context, session_cache,
         load_runtime, runtime_stream](
            std::int64_t requested_context) mutable {
            std::lock_guard<std::mutex> lock(*runtime_mutex);
            mlx::core::set_default_device(
                mlx::core::Device::gpu);
            mlx::core::set_default_stream(runtime_stream);
            mlx::core::synchronize();

            const auto previous_context = *loaded_context;
            session_cache->clear();
            runtime_holder->reset();
            release_model_load_staging_memory();
            const auto started =
                std::chrono::steady_clock::now();
            std::cout
                << "Reloading native Metal model with context="
                << requested_context << "..." << std::endl;
            try {
                runtime_holder->emplace(
                    load_runtime(requested_context));
                release_model_load_staging_memory();
                *loaded_context = requested_context;
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
                    runtime_holder->emplace(
                        load_runtime(previous_context));
                    release_model_load_staging_memory();
                    *loaded_context = previous_context;
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
        };
    MfqDuplexBackend duplex;
    duplex.name = "metal";
    if constexpr (std::is_same_v<
            Runtime, mfq::metal::MlxMiniCPMO45Runtime>) {
        duplex.start =
                [runtime_mutex, runtime_holder, runtime_stream](
                    const MfqDuplexSessionParams& parameters) {
                    if (parameters.special_ids.size() != 15) {
                        throw std::invalid_argument(
                            "MiniCPM-o duplex requires 15 special token IDs");
                    }
                    std::lock_guard<std::mutex> lock(*runtime_mutex);
                    if (!runtime_holder->has_value()) {
                        throw std::runtime_error(
                            "MiniCPM-o runtime is unavailable");
                    }
                    mlx::core::set_default_device(
                        mlx::core::Device::gpu);
                    mlx::core::set_default_stream(runtime_stream);
                    auto& runtime = runtime_holder->value();
                    runtime.reset();
                    mlx::core::clear_cache();

                    mfq::metal::MlxMiniCPMO45DuplexConfig config;
                    auto& ids = config.special_ids;
                    ids.unit_start = parameters.special_ids[0];
                    ids.unit_end = parameters.special_ids[1];
                    ids.image_start = parameters.special_ids[2];
                    ids.image_end = parameters.special_ids[3];
                    ids.slice_start = parameters.special_ids[4];
                    ids.slice_end = parameters.special_ids[5];
                    ids.listen = parameters.special_ids[6];
                    ids.speak = parameters.special_ids[7];
                    ids.tts_bos = parameters.special_ids[8];
                    ids.tts_eos = parameters.special_ids[9];
                    ids.chunk_eos = parameters.special_ids[10];
                    ids.chunk_tts_eos = parameters.special_ids[11];
                    ids.turn_eos = parameters.special_ids[12];
                    ids.tts_pad = parameters.special_ids[13];
                    ids.audio_bos = parameters.special_ids[14];
                    config.forbidden_ids = parameters.forbidden_ids;
                    config.greedy = parameters.greedy;
                    config.temperature = parameters.temperature;
                    config.top_k = parameters.top_k;
                    config.top_p = parameters.top_p;
                    config.listen_probability_scale =
                        parameters.listen_probability_scale;
                    config.repetition_penalty =
                        parameters.repetition_penalty;
                    config.repetition_window =
                        parameters.repetition_window;
                    config.length_penalty =
                        parameters.length_penalty;
                    config.tts_temperature =
                        parameters.tts_temperature;
                    config.tts_repetition_penalty =
                        parameters.tts_repetition_penalty;
                    config.seed = parameters.seed;

                    std::optional<mlx::core::array> system_prefix_ids;
                    if (!parameters.system_prefix.empty()) {
                        system_prefix_ids.emplace(
                            parameters.system_prefix.begin(),
                            mlx::core::Shape{
                                1,
                                static_cast<int>(
                                    parameters.system_prefix.size())},
                            mlx::core::int64);
                    }
                    std::optional<mlx::core::array> reference_features;
                    if (!parameters.reference_audio_features.empty()) {
                        if (parameters.reference_audio_frames <= 0 ||
                            parameters.reference_audio_features.size() !=
                                static_cast<std::size_t>(
                                    parameters.reference_audio_frames) * 80) {
                            throw std::invalid_argument(
                                "MiniCPM-o reference Mel geometry is invalid");
                        }
                        reference_features.emplace(
                            parameters.reference_audio_features.begin(),
                            mlx::core::Shape{
                                1, 80, parameters.reference_audio_frames},
                            mlx::core::float32);
                    }
                    std::optional<mlx::core::array> system_suffix_ids;
                    if (!parameters.system_suffix.empty()) {
                        system_suffix_ids.emplace(
                            parameters.system_suffix.begin(),
                            mlx::core::Shape{
                                1,
                                static_cast<int>(
                                    parameters.system_suffix.size())},
                            mlx::core::int64);
                    }
                    runtime.prepare_duplex(
                        config,
                        system_prefix_ids,
                        reference_features,
                        system_suffix_ids);
                    mlx::core::synchronize(runtime_stream);
                };
            duplex.step =
                [runtime_mutex, runtime_holder, runtime_stream](
                    const MfqDuplexStepInput& input) {
                    const bool has_audio = input.audio_frames > 0;
                    const bool has_text = !input.text_tokens.empty();
                    if (has_audio &&
                        input.audio_features.size() !=
                            static_cast<std::size_t>(input.audio_frames) * 80) {
                        throw std::invalid_argument(
                            "MiniCPM-o duplex Mel geometry is invalid");
                    }
                    if (!has_audio && !has_text) {
                        throw std::invalid_argument(
                            "MiniCPM-o duplex step has no input");
                    }
                    std::lock_guard<std::mutex> lock(*runtime_mutex);
                    if (!runtime_holder->has_value()) {
                        throw std::runtime_error(
                            "MiniCPM-o runtime is unavailable");
                    }
                    mlx::core::set_default_device(
                        mlx::core::Device::gpu);
                    mlx::core::set_default_stream(runtime_stream);
                    auto& runtime = runtime_holder->value();
                    if (!runtime.duplex_prepared()) {
                        throw std::runtime_error(
                            "MiniCPM-o duplex session is not prepared");
                    }

                    mfq::metal::MlxMiniCPMO45DuplexInputs inputs;
                    if (has_audio) {
                        inputs.audio_features.emplace(
                            input.audio_features.begin(),
                            mlx::core::Shape{1, 80, input.audio_frames},
                            mlx::core::float32);
                    }
                    if (has_text) {
                        inputs.text_ids.emplace(
                            input.text_tokens.begin(),
                            mlx::core::Shape{
                                1,
                                static_cast<int>(input.text_tokens.size())},
                            mlx::core::int64);
                    }
                    inputs.audio_prefix_extra_frames =
                        input.audio_prefix_extra_frames;
                    inputs.audio_suffix_extra_frames =
                        input.audio_suffix_extra_frames;
                    inputs.max_new_speak_tokens =
                        input.max_new_speak_tokens;
                    inputs.force_listen = input.force_listen;
                    inputs.force_speak = input.force_speak;

                    const auto started =
                        std::chrono::steady_clock::now();
                    auto result = runtime.duplex_step(inputs);
                    result.generated_ids.eval();
                    result.tts_codes.eval();
                    mlx::core::synchronize(runtime_stream);

                    MfqDuplexStepResult response;
                    const auto* generated =
                        result.generated_ids.template data<std::int64_t>();
                    response.generated_tokens.assign(
                        generated,
                        generated + result.generated_ids.size());
                    const auto* codes =
                        result.tts_codes.template data<std::int32_t>();
                    response.audio_tokens.assign(
                        codes, codes + result.tts_codes.size());
                    response.is_listen = result.is_listen;
                    response.end_of_turn = result.end_of_turn;
                    response.tts_force_flush = result.tts_force_flush;
                    response.audio_chunk_index =
                        result.audio_chunk_index;
                    response.language_cache_position =
                        result.language_cache_position;
                    response.audio_cache_position =
                        result.audio_cache_position;
                    response.tts_cache_position =
                        result.tts_cache_position;
                    response.inference_ms =
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started)
                            .count();
                    return response;
                };
            duplex.stop =
                [runtime_mutex, runtime_holder, runtime_stream]() {
                    std::lock_guard<std::mutex> lock(*runtime_mutex);
                    if (!runtime_holder->has_value()) return;
                    mlx::core::set_default_device(
                        mlx::core::Device::gpu);
                    mlx::core::set_default_stream(runtime_stream);
                    runtime_holder->value().reset();
                    mlx::core::synchronize(runtime_stream);
                    mlx::core::clear_cache();
                    malloc_zone_pressure_relief(nullptr, 0);
                };
    }
    MfqSessionControl session_control;
    session_control.fork =
        [runtime_mutex, session_cache](
            const std::string& source_session_id,
            const std::string& target_session_id) {
            std::lock_guard<std::mutex> lock(*runtime_mutex);
            return session_cache->fork_session(
                source_session_id, target_session_id);
        };
    session_control.close =
        [runtime_mutex, session_cache](const std::string& session_id) {
            std::lock_guard<std::mutex> lock(*runtime_mutex);
            return session_cache->close_session(session_id);
        };
    session_control.metrics = [session_cache] {
        return session_cache->metrics();
    };
    session_control.clear = [runtime_mutex, session_cache] {
        std::lock_guard<std::mutex> lock(*runtime_mutex);
        return session_cache->clear();
    };
    return run_mfq_server(
        server, generate, reload, duplex, session_control,
        multimodal_generate,
        [runtime_mutex, runtime_holder] {
            std::vector<std::pair<std::string, double>> metrics{
                {"mlx_active_bytes", static_cast<double>(mlx::core::get_active_memory())},
                {"mlx_cache_bytes", static_cast<double>(mlx::core::get_cache_memory())},
                {"mlx_peak_bytes", static_cast<double>(mlx::core::get_peak_memory())},
            };
            std::unique_lock lock(*runtime_mutex, std::try_to_lock);
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
                            "ssd_expert_hit_rate",
                            stats->hit_rate());
                        metrics.emplace_back(
                            "ssd_expert_bytes_read",
                            static_cast<double>(stats->bytes_read));
                        metrics.emplace_back(
                            "ssd_expert_resident_bytes",
                            static_cast<double>(stats->resident_bytes));
                        metrics.emplace_back(
                            "ssd_expert_resident_count",
                            static_cast<double>(stats->resident_experts));
                        metrics.emplace_back(
                            "ssd_expert_wait_seconds",
                            stats->wait_seconds);
                    }
                }
            }
            return metrics;
        });
}

int run_native_hf_server(const Arguments& arguments) {
    if (arguments.tokenizer_gguf.empty()) {
        throw std::runtime_error(
            "HF model directories currently require --tokenizer-gguf PATH");
    }
    const auto config = mfq::metal::DeepseekV4Config::from_json(
        read_text(arguments.mfq / "config.json"));
    const int context = static_cast<int>(
        std::min<std::int64_t>(
            arguments.context_size,
            config.max_position_embeddings));
    const auto expert_cache_bytes = requested_cache_bytes(
        arguments.expert_cache_gb, true);
    constexpr std::size_t prefill_buffers_minimum =
        std::size_t{7} << 30;
    const bool prefill_overlap =
        expert_cache_bytes >= prefill_buffers_minimum;
    const auto runtime_stream = mlx::core::new_thread_unsafe_stream(
        mlx::core::Device::gpu);
    mlx::core::set_default_stream(runtime_stream);
    const auto started = std::chrono::steady_clock::now();
    std::cout
        << "Loading native-format DeepSeek-V4 HF weights with SSD expert "
           "streaming on Apple UMA..."
        << std::endl;
    auto runtime = mfq::metal::MlxDeepseekV4CausalLm::load_hf(
        arguments.mfq,
        context,
        expert_cache_bytes,
        8,
        prefill_overlap);
    runtime.prewarm_ssd_expert_arena();
    release_model_load_staging_memory();
    const auto load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    constexpr double gib = static_cast<double>(std::uint64_t{1} << 30);
    std::cout
        << "Loaded " << runtime.layer_count()
        << " DeepSeek-V4 layers in " << load_seconds << " s"
        << " expert_backing=hf-safetensors-ssd"
        << " expert_cache_gib="
        << static_cast<double>(runtime.expert_cache_limit_bytes()) / gib
        << " prefill_double_buffer="
        << static_cast<int>(prefill_overlap)
        << std::endl;
    const auto model_root = arguments.mfq;
    const auto load_runtime =
        [model_root, expert_cache_bytes, prefill_overlap](
            std::int64_t requested_context) {
            if (requested_context < 1 ||
                requested_context > std::numeric_limits<int>::max()) {
                throw std::invalid_argument(
                    "Metal runtime context is out of range");
            }
            return mfq::metal::MlxDeepseekV4CausalLm::load_hf(
                model_root,
                static_cast<int>(requested_context),
                expert_cache_bytes,
                8,
                prefill_overlap);
        };
    const mfq::metal::MfqContainer profile_container(model_root);
    return serve_loaded_runtime(
        arguments,
        &profile_container,
        std::move(runtime),
        load_runtime,
        config.model_type,
        config.max_position_embeddings,
        config.vocab,
        runtime_stream);
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
        std::optional<std::size_t> expert_cache_bytes;
        if (arguments.expert_cache_gb.has_value()) {
            constexpr long double bytes_per_gib =
                static_cast<long double>(
                    std::uint64_t{1} << 30);
            const long double requested_cache =
                static_cast<long double>(
                    *arguments.expert_cache_gb) *
                bytes_per_gib;
            if (requested_cache >
                static_cast<long double>(
                    std::numeric_limits<
                        std::size_t>::max())) {
                throw std::runtime_error(
                    "DeepSeek-V4 expert cache exceeds "
                    "addressable memory");
            }
            expert_cache_bytes =
                static_cast<std::size_t>(
                    requested_cache);
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
        release_model_load_staging_memory();
        const auto load_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                started).count();
        std::cout
            << "Loaded " << runtime.layer_count()
            << " DeepSeek-V4 layers in "
            << load_seconds << " s";
        if (arguments.expert_cache_gb.has_value()) {
            std::cout
                << " nintm_load=disk-cache cache_gb="
                << *arguments.expert_cache_gb;
        } else {
            std::cout << " nintm_load=full-resident";
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
        return serve_loaded_runtime(
            arguments,
            &container,
            std::move(runtime),
            load_runtime,
            config.model_type,
            config.max_position_embeddings,
            config.vocab,
            runtime_stream);
    }

    if (architecture.rfind("minicpmo", 0) == 0) {
        std::cout
            << "Loading native C++/MLX MiniCPM-o 4.5 Qwen3-8B "
               "on Apple GPU..."
            << std::endl;
        auto runtime =
            mfq::metal::MlxMiniCPMO45Runtime::load(
                container,
                arguments.context_size,
                arguments.server);
        release_model_load_staging_memory();
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
        return serve_loaded_runtime(
            arguments,
            &container,
            std::move(runtime),
            load_runtime,
            "minicpmo",
            maximum_context,
            vocabulary,
            runtime_stream);
    }

    const bool qwen35_family =
        architecture.rfind("qwen35", 0) == 0 ||
        architecture.rfind("qwen3_5", 0) == 0 ||
        architecture.rfind("qwen3_6", 0) == 0 ||
        architecture.rfind("qwen3_8", 0) == 0;
    if (!qwen35_family) {
        throw std::runtime_error(
            "unsupported native Metal model architecture: " + architecture);
    }

    const auto config =
        mfq::metal::Qwen35Config::from_mfq(container);
    std::cout
        << "Loading native C++/MLX Qwen3.5 model "
           "on Apple GPU..."
        << std::endl;
    auto runtime =
        mfq::metal::MlxQwen35CausalLm::load(container);
    release_model_load_staging_memory();
    const auto load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::cout
        << "Loaded " << runtime.layer_count()
        << " Qwen3.5 layers in "
        << load_seconds << " s; native_mtp="
        << (runtime.supports_mtp() ? "greedy" : "disabled")
        << std::endl;
    const auto load_runtime =
        [&container](std::int64_t) {
            return mfq::metal::
                MlxQwen35CausalLm::load(container);
        };
    return serve_loaded_runtime(
        arguments,
        &container,
        std::move(runtime),
        load_runtime,
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
                usage_error("--mfq is required");
            }
            return EXIT_SUCCESS;
        }

        if (std::filesystem::is_directory(arguments.mfq)) {
            if (arguments.server) {
                configure_mlx_metal();
#ifdef MFQ_METAL_SERVER
                const auto config_path = arguments.mfq / "config.json";
                const auto config = nlohmann::json::parse(read_text(config_path));
                const auto model_type = config.value("model_type", std::string{});
                if (model_type.rfind("deepseek_v4", 0) == 0) {
                    return run_native_hf_server(arguments);
                }
#else
                throw std::runtime_error(
                    "this build has no C++ server support; configure with "
                    "-DMFQ_BUILD_CPP_SERVER=ON");
#endif
            }
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
            if (record.dtype == "NINTM") {
                const auto weight =
                    mfq::metal::MlxNintMoeWeight::from_blob(
                        model.read(arguments.tensor));
                const int routes = arguments.benchmark_experts.empty()
                    ? std::min(6, weight.experts())
                    : static_cast<int>(
                          arguments.benchmark_experts.size());
                std::vector<float> input_values(
                    static_cast<std::size_t>(weight.neuron_len()));
                for (std::size_t index = 0;
                     index < input_values.size();
                     ++index) {
                    input_values[index] = static_cast<float>(
                        static_cast<int>(index % 17) - 8) / 16.0f;
                }
                auto input = mlx::core::astype(
                    mlx::core::array(
                        input_values.begin(),
                        mlx::core::Shape{1, weight.neuron_len()}),
                    mlx::core::float16);
                std::vector<std::int32_t> expert_values =
                    arguments.benchmark_experts;
                if (expert_values.empty()) {
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
                auto expert_ids = mlx::core::array(
                    expert_values.begin(),
                    mlx::core::Shape{1, routes});

                auto execute = [&] {
                    return arguments.benchmark_swiglu
                        ? weight.routed_swiglu(input, expert_ids)
                        : weight(input, expert_ids);
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
                    output, mlx::core::float32);
                checked.eval();
                const auto* values = checked.data<float>();
                float maximum = 0.0f;
                for (std::size_t index = 0;
                     index < checked.size();
                     ++index) {
                    if (!std::isfinite(values[index])) {
                        throw std::runtime_error(
                            "Metal NINTM smoke test returned non-finite data");
                    }
                    maximum = std::max(
                        maximum, std::fabs(values[index]));
                }
                if (maximum <= 1e-12f) {
                    throw std::runtime_error(
                        "Metal NINTM smoke test unexpectedly returned all zero");
                }
                std::cout
                    << "Metal NINTM smoke test passed"
                    << " experts=" << weight.experts()
                    << " routes=" << routes
                    << " in=" << weight.neuron_len()
                    << " out=" << weight.out_per_expert()
                    << " projections=" << weight.projections()
                    << " swiglu="
                    << static_cast<int>(
                           arguments.benchmark_swiglu)
                    << " packed=" << weight.packed_nbytes()
                    << " reps=" << arguments.benchmark_reps
                    << " ms_per_dispatch="
                    << elapsed_ms / arguments.benchmark_reps
                    << "\n";
                return EXIT_SUCCESS;
            }
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
#ifdef MFQ_METAL_SERVER
            return run_native_server(arguments, model);
#else
            throw std::runtime_error(
                "this build has no C++ server support; configure with "
                "-DMFQ_BUILD_CPP_SERVER=ON");
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
