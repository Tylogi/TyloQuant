#include "mfq_paged_prefix_cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
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

std::size_t benchmark_size(const char* name, std::size_t fallback) {
    const auto* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    require(end != value && *end == '\0' && parsed > 0,
            "invalid prefix cache benchmark size");
    return static_cast<std::size_t>(parsed);
}

void run_benchmark() {
    using namespace mfq::cache;
    const auto block_count = benchmark_size(
        "MFQ_PREFIX_CACHE_BENCHMARK_BLOCKS", 16);
    constexpr std::size_t block_tokens = 256;
    const auto payload_bytes = benchmark_size(
        "MFQ_PREFIX_CACHE_BENCHMARK_PAYLOAD_BYTES",
        8ULL * 1024ULL * 1024ULL);
    const auto root = temporary_directory();
    std::vector<std::int64_t> tokens(block_count * block_tokens);
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        tokens[index] = static_cast<std::int64_t>(index * 17 + 3);
    }
    auto bytes = std::make_shared<std::vector<std::uint8_t>>(payload_bytes);
    for (std::size_t index = 0; index < bytes->size(); ++index) {
        (*bytes)[index] = static_cast<std::uint8_t>(index * 29 + 11);
    }
    const std::shared_ptr<const std::vector<std::uint8_t>> shared = bytes;
    require(payload_bytes <= std::numeric_limits<std::size_t>::max() / block_count,
            "prefix cache benchmark size overflow");
    const auto total_bytes = block_count * payload_bytes;
    {
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            root,
            "benchmark-v1",
            block_tokens,
            total_bytes * 2,
            total_bytes * 2,
            block_count,
            total_bytes * 2,
        });
        BlockHash parent{};
        for (std::size_t block = 0; block < block_count; ++block) {
            parent = cache.store(
                parent,
                tokens.data() + block * block_tokens,
                block_tokens,
                shared);
        }
        cache.flush();
        const auto match = cache.match(tokens, {}, false);
        volatile std::uint64_t checksum = 0;
        const auto started = std::chrono::steady_clock::now();
        for (const auto& hash : match.blocks) {
            const auto loaded = cache.load(hash);
            require(loaded.has_value(), "benchmark hot load failed");
            checksum += loaded->front();
            checksum += loaded->back();
        }
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "prefix_cache_benchmark phase=legacy_hot blocks="
                  << block_count << " bytes=" << total_bytes
                  << " elapsed_ms=" << elapsed
                  << " checksum=" << checksum << '\n';
        checksum = 0;
        const auto shared_started = std::chrono::steady_clock::now();
        const auto loaded = cache.load_prefix(match.blocks);
        require(loaded.size() == block_count, "benchmark shared hot load failed");
        for (const auto& block : loaded) {
            checksum += block->front();
            checksum += block->back();
        }
        const auto shared_elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - shared_started).count();
        std::cout << "prefix_cache_benchmark phase=shared_hot blocks="
                  << block_count << " bytes=" << total_bytes
                  << " elapsed_ms=" << shared_elapsed
                  << " checksum=" << checksum << '\n';
    }
    const auto benchmark_cold = [&](const char* phase, std::size_t readers) {
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            root,
            "benchmark-v1",
            block_tokens,
            total_bytes * 2,
            total_bytes * 2,
            block_count,
            total_bytes * 2,
            readers,
        });
        const auto match = cache.match(tokens, {}, false);
        volatile std::uint64_t checksum = 0;
        const auto started = std::chrono::steady_clock::now();
        const auto loaded = cache.load_prefix(match.blocks);
        require(loaded.size() == block_count, "benchmark cold load failed");
        for (const auto& block : loaded) {
            checksum += block->front();
            checksum += block->back();
        }
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "prefix_cache_benchmark phase=" << phase << " blocks="
                  << block_count << " bytes=" << total_bytes
                  << " elapsed_ms=" << elapsed
                  << " checksum=" << checksum << '\n';
    };
    benchmark_cold("shared_cold_serial", 1);
    benchmark_cold("shared_cold_2", 2);
    benchmark_cold("shared_cold", 4);
    benchmark_cold("shared_cold_8", 8);
    std::filesystem::remove_all(root);
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
        const auto shared_chain = cache.load_prefix(match.blocks);
        require(shared_chain.size() == 2,
                "shared prefix chain load length mismatch");
        require(*shared_chain[0] == std::vector<std::uint8_t>({10, 11, 12}) &&
                    *shared_chain[1] ==
                        std::vector<std::uint8_t>({20, 21, 22, 23}),
                "shared prefix chain payload mismatch");
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

    {
        const auto concurrent_root = root / "concurrent-load-test";
        std::vector<std::int64_t> concurrent_tokens(32);
        for (std::size_t index = 0; index < concurrent_tokens.size(); ++index) {
            concurrent_tokens[index] = static_cast<std::int64_t>(index + 100);
        }
        {
            PagedPrefixCache cache(PagedPrefixCacheConfig{
                concurrent_root,
                "concurrent-load",
                4,
                1024 * 1024,
                0,
                16,
            });
            BlockHash parent{};
            for (std::size_t offset = 0; offset < concurrent_tokens.size();
                 offset += 4) {
                parent = cache.store(
                    parent,
                    concurrent_tokens.data() + offset,
                    4,
                    payload({1, 3, 5, 7, 9}));
            }
            cache.flush();
        }
        PagedPrefixCache cache(PagedPrefixCacheConfig{
            concurrent_root,
            "concurrent-load",
            4,
            1024 * 1024,
            0,
            16,
        });
        std::atomic<int> failures{0};
        std::vector<std::thread> callers;
        for (int caller = 0; caller < 4; ++caller) {
            callers.emplace_back([&] {
                for (int iteration = 0; iteration < 4; ++iteration) {
                    const auto match = cache.match(concurrent_tokens);
                    const auto loaded = cache.load_prefix(match.blocks);
                    if (loaded.size() != 8 || loaded.front()->at(2) != 5) {
                        ++failures;
                    }
                }
            });
        }
        for (auto& caller : callers) caller.join();
        require(failures.load() == 0,
                "concurrent shared prefix loads were inconsistent");
    }

    const auto second_text = block_hash_hex(second);
    const auto block_file =
        root /
        block_hash_hex(sha256("model-sha|layout-v1|fp16")) /
        second_text.substr(0, 2) /
        (second_text + ".mfqkv");
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
        require(match.matched_tokens == 8,
                "corrupt block was not indexed for validation");
        const auto loaded = cache.load_prefix(match.blocks);
        require(loaded.size() == 1 &&
                    *loaded.front() == std::vector<std::uint8_t>({10, 11, 12}),
                "batch load did not stop at the corrupt block");
        require(cache.metrics().corrupt_blocks == 1,
                "corrupt block was not counted");
    }

    std::filesystem::remove_all(root);
    if (const auto* enabled = std::getenv("MFQ_PREFIX_CACHE_BENCHMARK");
        enabled != nullptr && enabled[0] == '1') {
        run_benchmark();
    }
    std::cout << "MFQ paged prefix cache tests passed\n";
    return 0;
}
