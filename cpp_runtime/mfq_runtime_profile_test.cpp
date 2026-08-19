#include "mfq_server.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

void write_text(const std::filesystem::path & path, const std::string & value) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot create test profile");
    output << value;
}

} // namespace

int main() {
    const auto registry = resolve_mfq_runtime_profile(
        "", "minicpmo-hf-mfq", "minicpmo", "test");
    require(registry.chat.temperature.has_value(), "registry temperature missing");
    require(std::abs(*registry.chat.temperature - 0.7) < 1e-12,
            "registry temperature mismatch");
    require(registry.chat.enable_thinking.has_value() &&
                !*registry.chat.enable_thinking,
            "registry thinking default mismatch");
    require(registry.duplex.force_listen_count.value_or(-1) == 0,
            "registry duplex defaults missing");
    require(registry.duplex.system_prompt.value_or("") ==
                "Streaming Omni Conversation.",
            "registry duplex system prompt missing");

    const auto root = std::filesystem::temp_directory_path() /
        "mfq-runtime-profile-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto model = root / "model-00002-of-00003.mfq";
    const auto family_sidecar = root / "model.runtime.json";
    const auto exact_sidecar = std::filesystem::path(
        model.string() + ".runtime.json");
    const auto explicit_profile = root / "server.json";
    write_text(family_sidecar,
        R"({"version":1,"chat":{"temperature":0.4,"top_p":0.6}})");
    write_text(exact_sidecar,
        R"({"version":1,"chat":{"temperature":0.3}})");
    write_text(explicit_profile,
        R"({"version":1,"chat":{"temperature":0.2}})");

    const auto resolved = resolve_mfq_runtime_profile(
        model.string(),
        "minicpmo-hf-mfq",
        "minicpmo",
        "test",
        R"({"version":1,"chat":{"temperature":0.5,"top_k":77}})",
        "{}",
        explicit_profile.string());
    require(std::abs(resolved.chat.temperature.value_or(-1.0) - 0.2) < 1e-12,
            "server profile did not win");
    require(std::abs(resolved.chat.top_p.value_or(-1.0) - 0.6) < 1e-12,
            "family sidecar field was lost");
    require(resolved.chat.top_k.value_or(-1) == 77,
            "embedded field was lost");
    require(resolved.source.find("server-explicit:") == 0,
            "profile source mismatch");

    std::filesystem::remove_all(root);
    return 0;
}
