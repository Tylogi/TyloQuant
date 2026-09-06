#include "mfq_paged_prefix_cache.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::filesystem::path temporary_directory() {
    const auto value = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    auto path = std::filesystem::temp_directory_path() /
        ("mfq-paged-prefix-cache-test-" + std::to_string(value));
    std::filesystem::create_directories(path);
    return path;
}

std::shared_ptr<const std::vector<std::uint8_t>> payload(
    std::initializer_list<std::uint8_t> bytes) {
    return std::make_shared<const std::vector<std::uint8_t>>(bytes);
}

} // namespace

int main() {
    using namespace mfq::cache;
    require(
        block_hash_hex(sha256("abc")) ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 implementation mismatch");

    const auto root = temporary_directory();
    const std::vector<std::int64_t> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9};
    BlockHash first{};
    BlockHash second{};
    {
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            root,
            "model-sha|layout-v1|fp16",
            4,
            4096,
            64,
            2,
        });
        BlockHash parent{};
        first = cache.store(parent, tokens.data(), 4, payload({10, 11, 12}));
        second = cache.store(
            first, tokens.data() + 4, 4, payload({20, 21, 22, 23}));
        cache.flush();

        const auto match = cache.match(tokens);
        require(match.matched_tokens == 8, "longest prefix match failed");
        require(match.blocks.size() == 2, "prefix block chain length mismatch");
        require(match.blocks[0] == first && match.blocks[1] == second,
                "prefix block chain mismatch");
        const auto loaded = cache.load(second);
        require(loaded && *loaded == std::vector<std::uint8_t>({20, 21, 22, 23}),
                "hot block load mismatch");

        const auto duplicate = cache.store(
            first, tokens.data() + 4, 4, payload({99}));
        require(duplicate == second, "content address changed on duplicate store");
        const auto original = cache.load(second);
        require(original && *original == std::vector<std::uint8_t>({20, 21, 22, 23}),
                "duplicate store replaced deterministic block content");
        cache.flush();
        const auto stats = cache.metrics();
        require(stats.writes == 2, "unexpected physical write count");
        require(stats.deduplicated_writes == 1, "deduplication was not counted");
        require(stats.disk_blocks == 2, "disk block count mismatch");
    }

    const auto namespace_dir =
        root / block_hash_hex(sha256("model-sha|layout-v1|fp16"));
    const auto stale_temporary = namespace_dir / "stale.mfqkv.tmp.1";
    {
        std::ofstream output(stale_temporary, std::ios::binary);
        output << "incomplete";
    }
    {
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            root,
            "model-sha|layout-v1|fp16",
            4,
            4096,
            0,
            2,
        });
        const auto match = cache.match(tokens);
        require(match.matched_tokens == 8, "restart prefix recovery failed");
        const auto loaded = cache.load(first);
        require(loaded && *loaded == std::vector<std::uint8_t>({10, 11, 12}),
                "restart disk load mismatch");
        require(!std::filesystem::exists(stale_temporary),
                "stale temporary write was not removed during recovery");
    }

    {
        PagedPrefixCache incompatible(PagedPrefixCacheConfig{
            root,
            "different-model",
            4,
            4096,
            0,
            2,
        });
        require(
            incompatible.match(tokens).matched_tokens == 0,
            "compatibility namespace isolation failed");
    }

    {
        const auto pin_root = root / "pin-test";
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            pin_root,
            "pin-before-write",
            4,
            1,
            0,
            2,
        });
        BlockHash parent{};
        const auto pinned = cache.store(
            parent, tokens.data(), 4, payload({1, 2, 3, 4}));
        cache.pin({pinned});
        cache.flush();
        require(cache.metrics().disk_blocks == 1,
                "pending pinned block was evicted after write");
        cache.unpin({pinned});
        require(cache.metrics().disk_blocks == 0,
                "unpinned over-budget block was not evicted");
    }

    {
        const auto byte_root = root / "pending-byte-test";
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            byte_root,
            "pending-byte-budget",
            4,
            4096,
            64,
            64,
            3,
        });
        BlockHash parent{};
        (void)cache.store(
            parent, tokens.data(), 4, payload({1, 2, 3, 4}));
        const auto stats = cache.metrics();
        require(stats.writes == 1,
                "oversized pending payload did not write inline");
        require(stats.pending_writes == 0 && stats.pending_bytes == 0,
                "inline write retained pending payload memory");
        require(stats.pending_max_bytes == 3,
                "pending payload byte budget was not reported");
    }

    const auto first_text = block_hash_hex(first);
    const auto block_file =
        root /
        block_hash_hex(sha256("model-sha|layout-v1|fp16")) /
        first_text.substr(0, 2) /
        (first_text + ".mfqkv");
    require(std::filesystem::is_regular_file(block_file),
            "cache block file is missing");
    {
        std::fstream file(block_file, std::ios::binary | std::ios::in | std::ios::out);
        require(static_cast<bool>(file), "cannot open block for corruption test");
        file.seekp(-1, std::ios::end);
        char byte = 0;
        file.write(&byte, 1);
    }
    {
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            root,
            "model-sha|layout-v1|fp16",
            4,
            4096,
            0,
            2,
        });
        const auto match = cache.match(tokens);
        require(match.matched_tokens > 0, "corrupt block was not indexed for validation");
        const auto loaded = cache.load(match.blocks.front());
        require(!loaded, "corrupt block payload was accepted");
        require(cache.metrics().corrupt_blocks == 1,
                "corrupt block was not counted");
    }

    std::filesystem::remove_all(root);
    std::cout << "MFQ paged prefix cache tests passed\n";
    return 0;
}
