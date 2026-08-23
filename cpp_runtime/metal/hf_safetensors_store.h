#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mfq::metal {

struct HfSafetensorRecord {
    std::string name;
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::size_t shard = 0;
    std::uint64_t offset = 0;
    std::uint64_t nbytes = 0;
};

// Read-only index over a sharded Hugging Face Safetensors checkpoint. Tensor
// payloads stay in their original shards; callers can issue exact range reads
// without materializing or rewriting the checkpoint.
class HfSafetensorStore {
public:
    explicit HfSafetensorStore(std::filesystem::path root);
    ~HfSafetensorStore();

    HfSafetensorStore(const HfSafetensorStore&) = delete;
    HfSafetensorStore& operator=(const HfSafetensorStore&) = delete;
    HfSafetensorStore(HfSafetensorStore&&) noexcept;
    HfSafetensorStore& operator=(HfSafetensorStore&&) noexcept;

    const std::filesystem::path& root() const noexcept;
    std::size_t tensor_count() const noexcept;
    std::size_t shard_count() const noexcept;

    const HfSafetensorRecord& tensor(std::string_view name) const;
    const std::filesystem::path& shard_path(std::size_t shard) const;

    void read_tensor(
        const HfSafetensorRecord& record,
        std::span<std::byte> destination) const;
    void read_range(
        std::size_t shard,
        std::uint64_t offset,
        std::span<std::byte> destination) const;
    void readv_range(
        std::size_t shard,
        std::uint64_t offset,
        std::span<const std::span<std::byte>> destinations) const;

    // Best-effort cache eviction before a storage benchmark. Production reads
    // use F_NOCACHE on macOS so expert streaming does not evict useful VM pages.
    void drop_file_cache() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct DeepseekV4NativeExpertView {
    std::span<const std::byte> w1_scale;
    std::span<const std::byte> w2_scale;
    std::span<const std::byte> w3_scale;
    std::span<const std::byte> w1_weight;
    std::span<const std::byte> w2_weight;
    std::span<const std::byte> w3_weight;
};

struct DeepseekV4NativeExpertDestination {
    std::span<std::byte> w1_scale;
    std::span<std::byte> w2_scale;
    std::span<std::byte> w3_scale;
    std::span<std::byte> w1_weight;
    std::span<std::byte> w2_weight;
    std::span<std::byte> w3_weight;
};

struct DeepseekV4NativeExpertLoadStats {
    std::uint64_t bytes = 0;
    std::uint64_t read_calls = 0;
};

// Exact, non-quantizing view of the official DeepSeek-V4 routed experts. Each
// cache slot preserves the checkpoint's native MXFP4 I8 payload and F8_E8M0
// scales. A cold expert is normally fetched with two pread calls: one contiguous
// scale run and one contiguous packed-weight run.
class DeepseekV4NativeExpertStore {
public:
    static constexpr std::size_t kParts = 6;

    DeepseekV4NativeExpertStore(
        std::filesystem::path root,
        std::size_t num_layers,
        std::size_t num_experts);
    ~DeepseekV4NativeExpertStore();

    const HfSafetensorStore& checkpoint() const noexcept;
    std::size_t num_layers() const noexcept;
    std::size_t num_experts() const noexcept;
    std::size_t slot_bytes() const noexcept;

    DeepseekV4NativeExpertLoadStats load(
        std::size_t layer,
        std::size_t expert,
        std::span<std::byte> slot) const;
    DeepseekV4NativeExpertLoadStats load_scatter(
        std::size_t layer,
        std::size_t expert,
        const DeepseekV4NativeExpertDestination& destination) const;

    DeepseekV4NativeExpertView view(
        std::span<const std::byte> slot) const;

private:
    struct ExpertRecord;

    const ExpertRecord& expert_record(
        std::size_t layer,
        std::size_t expert) const;

    HfSafetensorStore checkpoint_;
    std::size_t num_layers_ = 0;
    std::size_t num_experts_ = 0;
    std::size_t slot_bytes_ = 0;
    std::array<std::size_t, kParts + 1> slot_offsets_{};
    std::vector<ExpertRecord> experts_;
};

} // namespace mfq::metal
