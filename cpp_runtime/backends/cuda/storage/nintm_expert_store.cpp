#include "nintm_expert_store.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace mfq::cuda {
namespace {

template <typename T>
T little(std::span<const std::uint8_t> bytes, std::size_t offset) {
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
        throw std::runtime_error("truncated NINTM expert metadata");
    }
    using Unsigned = std::make_unsigned_t<T>;
    Unsigned result{};
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        result |= static_cast<Unsigned>(bytes[offset + index]) << (8 * index);
    }
    T value{};
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right,
    const char* label) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::runtime_error(std::string(label) + " overflows");
    }
    return left + right;
}

std::uint64_t checked_product(
    std::uint64_t left,
    std::uint64_t right,
    const char* label) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::runtime_error(std::string(label) + " overflows");
    }
    return left * right;
}

} // namespace

NintMxfp4ExpertStore::NintMxfp4ExpertStore(MfqRecordRange record)
    : record_(std::move(record)) {
    if (record_.dtype != "NINTM" || record_.nbytes < 20) {
        throw NintMxfp4Unsupported(
            "exact-range expert projection is not NINTM: " + record_.name);
    }
    const auto header = read_range(0, 20);
    if (std::memcmp(header.data(), "NIM2", 4) != 0) {
        throw NintMxfp4Unsupported(
            "exact-range expert projection is not NIM2: " + record_.name);
    }
    const auto experts = little<std::uint32_t>(header, 4);
    const auto output = little<std::uint32_t>(header, 8);
    const auto input = little<std::uint32_t>(header, 12);
    const auto pools = little<std::uint32_t>(header, 16);
    if (experts == 0 || output == 0 || input == 0 ||
        pools == 0 || pools > experts ||
        experts > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        output > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        input > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "invalid exact-range NINTM geometry: " + record_.name);
    }
    if (input % 32 != 0) {
        throw NintMxfp4Unsupported(
            "exact-range MXFP4 geometry is not block aligned: " +
            record_.name);
    }
    num_experts_ = static_cast<int>(experts);
    out_per_expert_ = static_cast<int>(output);
    neuron_len_ = static_cast<int>(input);
    experts_.resize(experts);
    std::vector<std::uint8_t> present(experts, 0);

    std::uint64_t cursor = 20;
    for (std::uint32_t pool = 0; pool < pools; ++pool) {
        if (cursor > record_.nbytes || 24 > record_.nbytes - cursor) {
            throw std::runtime_error(
                "truncated exact-range NINTM pool: " + record_.name);
        }
        const auto pool_header = read_range(cursor, 24);
        const auto count = little<std::uint32_t>(pool_header, 0);
        const auto dtype_bytes = little<std::uint32_t>(pool_header, 4);
        const auto payload_bytes = little<std::uint64_t>(pool_header, 8);
        const auto runtime_bytes = little<std::uint64_t>(pool_header, 16);
        if (count == 0 || count > experts || dtype_bytes == 0 ||
            dtype_bytes > 32) {
            throw std::runtime_error(
                "invalid exact-range NINTM pool metadata: " + record_.name);
        }
        if (runtime_bytes != 0) {
            throw NintMxfp4Unsupported(
                "exact-range MXFP4 pools cannot carry runtime metadata: " +
                record_.name);
        }
        cursor = checked_add(cursor, 24, "NINTM pool offset");
        const auto ids_bytes = checked_product(
            count, sizeof(std::int32_t), "NINTM expert IDs");
        const auto metadata_bytes = checked_add(
            ids_bytes, dtype_bytes, "NINTM pool metadata");
        if (cursor > record_.nbytes || metadata_bytes > record_.nbytes - cursor) {
            throw std::runtime_error(
                "truncated exact-range NINTM metadata: " + record_.name);
        }
        const auto metadata = read_range(cursor, metadata_bytes);
        const std::string dtype(
            reinterpret_cast<const char*>(metadata.data() + ids_bytes),
            dtype_bytes);
        if (dtype != "MXFP4") {
            throw NintMxfp4Unsupported(
                "exact-range cache requires MXFP4 NINTM experts: " +
                record_.name);
        }
        const auto payload_offset = checked_add(
            cursor, metadata_bytes, "NINTM payload offset");
        const auto payload_end = checked_add(
            payload_offset, payload_bytes, "NINTM payload end");
        if (payload_end > record_.nbytes || payload_bytes < 56) {
            throw std::runtime_error(
                "truncated exact-range MXFP4 payload: " + record_.name);
        }
        const auto mx = read_range(payload_offset, 56);
        const auto rows = checked_product(count, output, "MXFP4 rows");
        if (std::memcmp(mx.data(), "MXT1", 4) != 0 || mx[4] != 1 ||
            mx[5] != 4 || little<std::uint16_t>(mx, 6) != 0 ||
            little<std::uint64_t>(mx, 8) != rows ||
            little<std::uint64_t>(mx, 16) != input ||
            little<std::uint64_t>(mx, 24) != rows ||
            little<std::uint64_t>(mx, 32) != input / 2 ||
            little<std::uint64_t>(mx, 40) != rows ||
            little<std::uint64_t>(mx, 48) != input / 32) {
            throw std::runtime_error(
                "unsupported exact-range MXFP4 layout: " + record_.name);
        }
        const auto values_per_expert = checked_product(
            output, input / 2, "MXFP4 expert values");
        const auto scales_per_expert = checked_product(
            output, input / 32, "MXFP4 expert scales");
        if (values_bytes_per_expert_ == 0) {
            values_bytes_per_expert_ = values_per_expert;
            scales_bytes_per_expert_ = scales_per_expert;
        } else if (values_bytes_per_expert_ != values_per_expert ||
                   scales_bytes_per_expert_ != scales_per_expert) {
            throw std::runtime_error(
                "inconsistent exact-range MXFP4 expert sizes: " + record_.name);
        }
        const auto values_bytes = checked_product(
            count, values_per_expert, "MXFP4 values");
        const auto scales_bytes = checked_product(
            count, scales_per_expert, "MXFP4 scales");
        const auto expected_payload = checked_add(
            56,
            checked_add(values_bytes, scales_bytes, "MXFP4 payload"),
            "MXFP4 payload");
        if (payload_bytes != expected_payload) {
            throw std::runtime_error(
                "exact-range MXFP4 payload size mismatch: " + record_.name);
        }
        const auto values_offset = checked_add(
            payload_offset, 56, "MXFP4 values offset");
        const auto scales_offset = checked_add(
            values_offset, values_bytes, "MXFP4 scales offset");
        for (std::uint32_t local = 0; local < count; ++local) {
            const auto expert = little<std::int32_t>(
                metadata, static_cast<std::size_t>(local) * 4);
            if (expert < 0 || expert >= num_experts_ ||
                present[static_cast<std::size_t>(expert)] != 0) {
                throw std::runtime_error(
                    "duplicate or invalid exact-range expert ID: " +
                    record_.name);
            }
            present[static_cast<std::size_t>(expert)] = 1;
            auto& destination = experts_[static_cast<std::size_t>(expert)];
            destination[values] = {
                checked_add(
                    values_offset,
                    checked_product(local, values_per_expert,
                        "MXFP4 expert value offset"),
                    "MXFP4 expert value offset"),
                values_per_expert,
            };
            destination[scales] = {
                checked_add(
                    scales_offset,
                    checked_product(local, scales_per_expert,
                        "MXFP4 expert scale offset"),
                    "MXFP4 expert scale offset"),
                scales_per_expert,
            };
        }
        payload_bytes_ = checked_add(
            payload_bytes_,
            checked_add(values_bytes, scales_bytes, "MXFP4 payload bytes"),
            "MXFP4 payload bytes");
        cursor = payload_end;
    }
    if (cursor != record_.nbytes ||
        std::find(present.begin(), present.end(), 0) != present.end()) {
        throw std::runtime_error(
            "exact-range NINTM expert coverage or tail mismatch: " +
            record_.name);
    }
}

int NintMxfp4ExpertStore::num_experts() const noexcept {
    return num_experts_;
}

int NintMxfp4ExpertStore::out_per_expert() const noexcept {
    return out_per_expert_;
}

int NintMxfp4ExpertStore::neuron_len() const noexcept {
    return neuron_len_;
}

std::uint64_t NintMxfp4ExpertStore::values_bytes_per_expert() const noexcept {
    return values_bytes_per_expert_;
}

std::uint64_t NintMxfp4ExpertStore::scales_bytes_per_expert() const noexcept {
    return scales_bytes_per_expert_;
}

std::uint64_t NintMxfp4ExpertStore::payload_bytes() const noexcept {
    return payload_bytes_;
}

const MfqRecordRange& NintMxfp4ExpertStore::record() const noexcept {
    return record_;
}

const NintMxfp4ExpertPart& NintMxfp4ExpertStore::part(
    int expert,
    std::size_t field) const {
    if (expert < 0 || expert >= num_experts_ || field >= field_count) {
        throw std::out_of_range("exact-range NINTM expert part out of range");
    }
    return experts_[static_cast<std::size_t>(expert)][field];
}

void NintMxfp4ExpertStore::read_part_into(
    const NintMxfp4ExpertPart& part,
    std::span<std::uint8_t> destination) const {
    if (destination.size() != part.nbytes) {
        throw std::runtime_error(
            "exact-range NINTM expert destination size mismatch");
    }
    read_range_into(part.offset, destination);
}

std::vector<std::uint8_t> NintMxfp4ExpertStore::read_blob() const {
    return read_range(0, record_.nbytes);
}

std::vector<std::uint8_t> NintMxfp4ExpertStore::read_range(
    std::uint64_t offset,
    std::uint64_t nbytes) const {
    if (nbytes > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("exact-range read is too large for this host");
    }
    std::vector<std::uint8_t> result(static_cast<std::size_t>(nbytes));
    read_range_into(offset, result);
    return result;
}

void NintMxfp4ExpertStore::read_range_into(
    std::uint64_t offset,
    std::span<std::uint8_t> destination) const {
    const auto nbytes = static_cast<std::uint64_t>(destination.size());
    if (offset > record_.nbytes || nbytes > record_.nbytes - offset) {
        throw std::out_of_range("exact-range read exceeds the MFQ record");
    }
    const auto absolute = checked_add(
        record_.offset, offset, "MFQ absolute range offset");
    if (absolute > static_cast<std::uint64_t>(
            std::numeric_limits<std::streamoff>::max()) ||
        destination.size() > static_cast<std::size_t>(
            std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("exact-range read exceeds stream limits");
    }
    std::ifstream stream(record_.source_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "cannot open MFQ expert source: " + record_.source_path);
    }
    stream.seekg(static_cast<std::streamoff>(absolute));
    if (!stream) {
        throw std::runtime_error(
            "failed seeking MFQ expert source: " + record_.name);
    }
    if (!destination.empty()) {
        stream.read(
            reinterpret_cast<char*>(destination.data()),
            static_cast<std::streamsize>(destination.size()));
        if (stream.gcount() != static_cast<std::streamsize>(destination.size())) {
            throw std::runtime_error(
                "failed reading exact MFQ expert range: " + record_.name);
        }
    }
}

struct NintMxfp4ReadPool::Impl {
    struct Batch {
        std::mutex mutex;
        std::condition_variable condition;
        std::size_t remaining = 0;
        std::uint64_t file_opens = 0;
        std::exception_ptr error;
    };

    struct Task {
        NintMxfp4ReadRequest request;
        std::shared_ptr<Batch> batch;
    };

    explicit Impl(std::size_t requested_workers)
        : worker_count(std::max<std::size_t>(1, requested_workers)) {
        threads.reserve(worker_count);
        try {
            for (std::size_t index = 0; index < worker_count; ++index) {
                threads.emplace_back([this] { worker(); });
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> guard(mutex);
                stopping = true;
            }
            condition.notify_all();
            for (auto& thread : threads) {
                if (thread.joinable()) thread.join();
            }
            throw;
        }
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> guard(mutex);
            stopping = true;
        }
        condition.notify_all();
        for (auto& thread : threads) {
            if (thread.joinable()) thread.join();
        }
    }

    using StreamCache =
        std::unordered_map<std::string, std::unique_ptr<std::ifstream>>;

    void execute(
        const NintMxfp4ReadRequest& request,
        StreamCache& streams,
        bool& opened) {
        if (request.store == nullptr || request.part == nullptr ||
            request.destination.size() != request.part->nbytes) {
            throw std::invalid_argument("invalid exact-range read request");
        }
        const auto& record = request.store->record();
        const auto& part = *request.part;
        if (part.offset > record.nbytes ||
            part.nbytes > record.nbytes - part.offset) {
            throw std::out_of_range("exact-range read exceeds the MFQ record");
        }
        const auto absolute = checked_add(
            record.offset, part.offset, "MFQ absolute range offset");
        if (absolute > static_cast<std::uint64_t>(
                std::numeric_limits<std::streamoff>::max()) ||
            request.destination.size() > static_cast<std::size_t>(
                std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error("exact-range read exceeds stream limits");
        }

        auto found = streams.find(record.source_path);
        if (found == streams.end()) {
            auto stream = std::make_unique<std::ifstream>(
                record.source_path, std::ios::binary);
            if (!*stream) {
                throw std::runtime_error(
                    "cannot open MFQ expert source: " + record.source_path);
            }
            found = streams.emplace(
                record.source_path, std::move(stream)).first;
            opened = true;
        }
        auto& stream = *found->second;
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(absolute));
        if (!stream) {
            streams.erase(found);
            throw std::runtime_error(
                "failed seeking MFQ expert source: " + record.name);
        }
        if (request.destination.empty()) return;
        stream.read(
            reinterpret_cast<char*>(request.destination.data()),
            static_cast<std::streamsize>(request.destination.size()));
        if (stream.gcount() !=
                static_cast<std::streamsize>(request.destination.size())) {
            streams.erase(found);
            throw std::runtime_error(
                "failed reading exact MFQ expert range: " + record.name);
        }
    }

    void worker() {
        StreamCache streams;
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, [this] {
                    return stopping || !tasks.empty();
                });
                if (stopping && tasks.empty()) return;
                task = std::move(tasks.front());
                tasks.pop_front();
            }
            bool opened = false;
            try {
                execute(task.request, streams, opened);
            } catch (...) {
                std::lock_guard<std::mutex> guard(task.batch->mutex);
                if (!task.batch->error) {
                    task.batch->error = std::current_exception();
                }
            }
            {
                std::lock_guard<std::mutex> guard(task.batch->mutex);
                if (opened) ++task.batch->file_opens;
                if (task.batch->remaining == 0) {
                    std::terminate();
                }
                --task.batch->remaining;
            }
            task.batch->condition.notify_one();
        }
    }

    const std::size_t worker_count;
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<Task> tasks;
    std::vector<std::thread> threads;
    bool stopping = false;
};

NintMxfp4ReadPool::NintMxfp4ReadPool(std::size_t workers)
    : impl_(std::make_unique<Impl>(workers)) {}

NintMxfp4ReadPool::~NintMxfp4ReadPool() = default;

std::size_t NintMxfp4ReadPool::workers() const noexcept {
    return impl_->worker_count;
}

NintMxfp4ReadBatchStats NintMxfp4ReadPool::read(
    std::span<const NintMxfp4ReadRequest> requests) {
    NintMxfp4ReadBatchStats result;
    if (requests.empty()) return result;
    for (const auto& request : requests) {
        if (request.store == nullptr || request.part == nullptr ||
            request.destination.size() != request.part->nbytes ||
            request.part->nbytes >
                std::numeric_limits<std::uint64_t>::max() - result.bytes) {
            throw std::invalid_argument("invalid exact-range read batch");
        }
        result.bytes += request.part->nbytes;
    }
    result.calls = requests.size();
    const auto started = std::chrono::steady_clock::now();
    auto batch = std::make_shared<Impl::Batch>();
    batch->remaining = requests.size();
    {
        std::lock_guard<std::mutex> guard(impl_->mutex);
        if (impl_->stopping) {
            throw std::runtime_error("exact-range read pool is stopping");
        }
        for (const auto& request : requests) {
            impl_->tasks.push_back({request, batch});
        }
    }
    impl_->condition.notify_all();
    std::unique_lock<std::mutex> lock(batch->mutex);
    batch->condition.wait(lock, [&batch] {
        return batch->remaining == 0;
    });
    result.file_opens = batch->file_opens;
    if (batch->error) std::rethrow_exception(batch->error);
    result.wall_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
    return result;
}

} // namespace mfq::cuda
