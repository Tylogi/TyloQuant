#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq::cuda {

class NintMxfp4Unsupported : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct MfqRecordRange {
    std::string name;
    std::string dtype;
    std::string source_path;
    std::uint64_t offset = 0;
    std::uint64_t nbytes = 0;
};

struct NintMxfp4ExpertPart {
    std::uint64_t offset = 0;
    std::uint64_t nbytes = 0;
};

// Exact-range view over one canonical NINTM routed projection. The store
// retains only metadata; expert payloads remain in the MFQ container until a
// cache miss asks for their MXFP4 values and scales.
class NintMxfp4ExpertStore {
public:
    enum Field : std::size_t {
        values = 0,
        scales = 1,
        field_count = 2,
    };

    explicit NintMxfp4ExpertStore(MfqRecordRange record);

    int num_experts() const noexcept;
    int out_per_expert() const noexcept;
    int neuron_len() const noexcept;
    std::uint64_t values_bytes_per_expert() const noexcept;
    std::uint64_t scales_bytes_per_expert() const noexcept;
    std::uint64_t payload_bytes() const noexcept;
    const MfqRecordRange& record() const noexcept;

    const NintMxfp4ExpertPart& part(
        int expert,
        std::size_t field) const;
    void read_part_into(
        const NintMxfp4ExpertPart& part,
        std::span<std::uint8_t> destination) const;
    std::vector<std::uint8_t> read_blob() const;

private:
    std::vector<std::uint8_t> read_range(
        std::uint64_t offset,
        std::uint64_t nbytes) const;
    void read_range_into(
        std::uint64_t offset,
        std::span<std::uint8_t> destination) const;

    MfqRecordRange record_;
    int num_experts_ = 0;
    int out_per_expert_ = 0;
    int neuron_len_ = 0;
    std::uint64_t values_bytes_per_expert_ = 0;
    std::uint64_t scales_bytes_per_expert_ = 0;
    std::uint64_t payload_bytes_ = 0;
    std::vector<std::array<NintMxfp4ExpertPart, field_count>> experts_;
};

struct NintMxfp4ReadRequest {
    const NintMxfp4ExpertStore* store = nullptr;
    const NintMxfp4ExpertPart* part = nullptr;
    std::span<std::uint8_t> destination;
};

struct NintMxfp4ReadBatchStats {
    std::uint64_t bytes = 0;
    std::uint64_t calls = 0;
    std::uint64_t file_opens = 0;
    std::uint64_t wall_nanoseconds = 0;
};

struct NintMxfp4ReadState;

class NintMxfp4ReadTicket {
public:
    NintMxfp4ReadTicket() = default;
    NintMxfp4ReadTicket(NintMxfp4ReadTicket&&) noexcept;
    NintMxfp4ReadTicket& operator=(NintMxfp4ReadTicket&&) noexcept;
    ~NintMxfp4ReadTicket();

    NintMxfp4ReadTicket(const NintMxfp4ReadTicket&) = delete;
    NintMxfp4ReadTicket& operator=(const NintMxfp4ReadTicket&) = delete;

    bool valid() const noexcept;
    NintMxfp4ReadBatchStats wait();

private:
    explicit NintMxfp4ReadTicket(
        std::shared_ptr<NintMxfp4ReadState> state);

    std::shared_ptr<NintMxfp4ReadState> state_;

    friend class NintMxfp4ReadPool;
};

class NintMxfp4ReadPool {
public:
    explicit NintMxfp4ReadPool(std::size_t workers);
    ~NintMxfp4ReadPool();

    NintMxfp4ReadPool(const NintMxfp4ReadPool&) = delete;
    NintMxfp4ReadPool& operator=(const NintMxfp4ReadPool&) = delete;

    std::size_t workers() const noexcept;
    NintMxfp4ReadTicket submit(
        std::span<const NintMxfp4ReadRequest> requests);
    NintMxfp4ReadBatchStats read(
        std::span<const NintMxfp4ReadRequest> requests);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::cuda
