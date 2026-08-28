#include "hf_safetensors_store.h"

#include "../json/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

using json = nlohmann::json;

struct Part {
    const char* suffix;
    const char* dtype;
    std::array<std::int64_t, 2> shape;
    std::size_t bytes;
    std::byte fill;
};

constexpr std::array<Part, 6> kParts{{
    {"w1.scale", "F8_E8M0", {2048, 128}, 2048 * 128, std::byte{0x11}},
    {"w2.scale", "F8_E8M0", {4096, 64}, 4096 * 64, std::byte{0x22}},
    {"w3.scale", "F8_E8M0", {2048, 128}, 2048 * 128, std::byte{0x33}},
    {"w1.weight", "I8", {2048, 2048}, 2048 * 2048, std::byte{0x44}},
    {"w2.weight", "I8", {4096, 1024}, 4096 * 1024, std::byte{0x55}},
    {"w3.weight", "I8", {2048, 2048}, 2048 * 2048, std::byte{0x66}},
}};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_u64(std::ofstream& stream, std::uint64_t value) {
    std::array<char, 8> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<char>((value >> (8 * index)) & 0xffu);
    }
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_fixture(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    const std::string shard_name = "model-00001-of-00001.safetensors";
    json header = json::object();
    json weight_map = json::object();
    std::uint64_t offset = 0;
    for (const auto& part : kParts) {
        const auto name = "layers.0.ffn.experts.0." +
            std::string(part.suffix);
        header[name] = {
            {"dtype", part.dtype},
            {"shape", {part.shape[0], part.shape[1]}},
            {"data_offsets", {offset, offset + part.bytes}},
        };
        weight_map[name] = shard_name;
        offset += part.bytes;
    }
    auto header_text = header.dump();
    while ((8 + header_text.size()) % 8 != 0) {
        header_text.push_back(' ');
    }

    std::ofstream shard(root / shard_name, std::ios::binary);
    require(static_cast<bool>(shard), "cannot create Safetensors fixture");
    write_u64(shard, header_text.size());
    shard.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    std::vector<char> chunk(1 << 20);
    for (const auto& part : kParts) {
        std::fill(
            chunk.begin(),
            chunk.end(),
            static_cast<char>(std::to_integer<unsigned char>(part.fill)));
        std::size_t remaining = part.bytes;
        while (remaining != 0) {
            const auto count = std::min(remaining, chunk.size());
            shard.write(chunk.data(), static_cast<std::streamsize>(count));
            remaining -= count;
        }
    }
    require(static_cast<bool>(shard), "cannot write Safetensors fixture");

    const json index = {
        {"metadata", {{"total_size", offset}}},
        {"weight_map", std::move(weight_map)},
    };
    std::ofstream index_stream(
        root / "model.safetensors.index.json",
        std::ios::binary);
    index_stream << index.dump();
    require(static_cast<bool>(index_stream), "cannot write Safetensors index fixture");
}

void check_part(
    std::span<const std::byte> part,
    const Part& expected) {
    require(part.size() == expected.bytes, "expert part size mismatch");
    require(!part.empty(), "expert part is empty");
    require(part.front() == expected.fill, "expert part prefix mismatch");
    require(part.back() == expected.fill, "expert part suffix mismatch");
}

} // namespace

int main() {
    auto root = std::filesystem::temp_directory_path() /
        ("mfq-hf-store-test-" + std::to_string(::getpid()));
    try {
        write_fixture(root);
        mfq::metal::DeepseekV4NativeExpertStore store(root, 1, 1);
        require(store.checkpoint().shard_count() == 1, "shard count mismatch");
        require(store.checkpoint().tensor_count() == 6, "tensor count mismatch");
        const auto expected_bytes = 3 * (2048 * 128) +
            2 * (2048 * 2048) + 4096 * 1024;
        require(store.slot_bytes() == expected_bytes, "slot byte count mismatch");

        std::vector<std::byte> slot(store.slot_bytes());
        const auto stats = store.load(0, 0, slot);
        require(stats.bytes == slot.size(), "load byte count mismatch");
        require(stats.read_calls == 2, "expert load was not coalesced");
        const auto view = store.view(slot);
        check_part(view.w1_scale, kParts[0]);
        check_part(view.w2_scale, kParts[1]);
        check_part(view.w3_scale, kParts[2]);
        check_part(view.w1_weight, kParts[3]);
        check_part(view.w2_weight, kParts[4]);
        check_part(view.w3_weight, kParts[5]);

        std::array<std::vector<std::byte>, 6> scattered;
        for (std::size_t part = 0; part < scattered.size(); ++part) {
            scattered[part].resize(kParts[part].bytes);
        }
        const auto scatter_stats = store.load_scatter(0, 0, {
            .w1_scale = scattered[0],
            .w2_scale = scattered[1],
            .w3_scale = scattered[2],
            .w1_weight = scattered[3],
            .w2_weight = scattered[4],
            .w3_weight = scattered[5],
        });
        require(scatter_stats.bytes == slot.size(), "scatter byte count mismatch");
        require(scatter_stats.read_calls == 2, "scatter load was not coalesced");
        for (std::size_t part = 0; part < scattered.size(); ++part) {
            check_part(scattered[part], kParts[part]);
            std::fill(scattered[part].begin(), scattered[part].end(), std::byte{0});
        }

        const mfq::metal::DeepseekV4NativeExpertDestination phased{
            .w1_scale = scattered[0],
            .w2_scale = scattered[1],
            .w3_scale = scattered[2],
            .w1_weight = scattered[3],
            .w2_weight = scattered[4],
            .w3_weight = scattered[5],
        };
        const auto scale_stats = store.load_scales_scatter(0, 0, phased);
        const auto gate_stats = store.load_gate_scatter(0, 0, phased);
        const auto up_stats = store.load_up_scatter(0, 0, phased);
        const auto down_stats = store.load_down_scatter(0, 0, phased);
        require(
            scale_stats.read_calls + gate_stats.read_calls +
                    up_stats.read_calls + down_stats.read_calls ==
                4,
            "phased expert read count mismatch");
        require(
            scale_stats.bytes + gate_stats.bytes + up_stats.bytes +
                    down_stats.bytes ==
                slot.size(),
            "phased expert byte count mismatch");
        for (std::size_t part = 0; part < scattered.size(); ++part) {
            check_part(scattered[part], kParts[part]);
        }

        std::filesystem::remove_all(root);
        std::cout << "HF Safetensors SSD expert store passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(root);
        std::cerr << "HF Safetensors SSD expert store failed: "
                  << error.what() << '\n';
        return 1;
    }
}
