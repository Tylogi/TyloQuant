#include "mfq_container.h"

#include "mlx_legacy_tensor_compat.h"

#include "mfq/mfq_model_source.h"
#include "mfq/tq_rows.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <fcntl.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>

namespace mfq::metal {
namespace {

constexpr std::uint32_t kMaxRecordEntries =
    std::uint32_t{1} << 20;
std::uint64_t checked_file_size(
    const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (
        error
        || size > static_cast<std::uintmax_t>(
            std::numeric_limits<std::uint64_t>::max())
        || size > static_cast<std::uintmax_t>(
            std::numeric_limits<std::streamoff>::max())
    ) {
        throw std::runtime_error(
            "cannot determine usable MFQ file size: " + path.string());
    }
    return static_cast<std::uint64_t>(size);
}

} // namespace

std::vector<std::int64_t> MfqContainer::tq_shape(const std::string& name) const {
    const auto& entry = record(name);
    if (entry.dtype != "TQ") throw std::invalid_argument("tensor is not TQ: " + name);
    std::vector<std::int64_t> logical;
    for (const auto* key : {"tq_shapes", "tensor_shapes"}) {
        const auto found = header_.extra_json.find(key);
        if (found == header_.extra_json.end()) continue;
        const auto shapes = nlohmann::json::parse(found->second);
        if (!shapes.is_object()) throw std::invalid_argument("invalid TQ shape map");
        if (shapes.contains(name)) {
            logical = shapes.at(name).get<std::vector<std::int64_t>>();
            break;
        }
    }
    return mfq::tq_tensor_shape(entry.nbytes,
        [&](std::uint64_t offset, std::byte* output, std::size_t size) {
            read_range_into(name, offset, {output, size});
        }, std::move(logical));
}

struct MfqContainer::RandomAccessFiles : std::enable_shared_from_this<RandomAccessFiles> {
    struct File {
        explicit File(std::filesystem::path source)
            : path(std::move(source)), size(checked_file_size(path)) {}

        ~File() {
            if (descriptor >= 0) {
                ::close(descriptor);
            }
        }

        int open() const {
            std::scoped_lock lock(mutex);
            if (descriptor >= 0) {
                return descriptor;
            }
            descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (descriptor < 0) {
                throw std::runtime_error(
                    "cannot open MFQ record source: " + path.string() +
                    ": " + std::strerror(errno));
            }
#if defined(__APPLE__) && defined(F_NOCACHE)
            if (::fcntl(descriptor, F_NOCACHE, 1) != 0) {
                const auto error = errno;
                ::close(descriptor);
                descriptor = -1;
                throw std::runtime_error(
                    "cannot enable direct MFQ record reads: " + path.string() +
                    ": " + std::strerror(error));
            }
#endif
            return descriptor;
        }

        std::filesystem::path path;
        std::uint64_t size = 0;
        mutable std::mutex mutex;
        mutable int descriptor = -1;
    };

    std::shared_ptr<File> source_for(const std::filesystem::path& path) {
        struct LastFile {
            const RandomAccessFiles* owner = nullptr;
            std::weak_ptr<RandomAccessFiles> lifetime;
            std::filesystem::path path;
            std::weak_ptr<File> file;
        };
        static thread_local LastFile last;
        std::shared_ptr<File> source;
        if (last.owner == this && !last.lifetime.expired() && last.path.native() == path.native())
            source = last.file.lock();
        if (!source) {
            std::scoped_lock lock(mutex);
            const auto found = files.find(path);
            if (found != files.end()) {
                source = found->second;
            } else {
                source = std::make_shared<File>(path);
                files.emplace(path, source);
            }
            last = {this, weak_from_this(), path, source};
        }
        return source;
    }

    void read(
        const std::filesystem::path& path,
        std::uint64_t offset,
        std::span<std::byte> destination) {
        const auto source = source_for(path);
        if (offset > source->size ||
            destination.size() > source->size - offset) {
            throw std::runtime_error(
                "MFQ record source was truncated: " + path.string());
        }
        std::size_t done = 0;
        while (done < destination.size()) {
            constexpr std::size_t maximum_read = std::size_t{32} << 20;
            const auto requested = std::min(
                destination.size() - done, maximum_read);
            const auto count = ::pread(
                source->open(),
                destination.data() + done,
                requested,
                static_cast<off_t>(offset + done));
            if (count < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(
                    "failed reading MFQ record source: " + path.string() +
                    ": " + std::strerror(errno));
            }
            if (count == 0) {
                throw std::runtime_error(
                    "MFQ record source was truncated: " + path.string());
            }
            done += static_cast<std::size_t>(count);
        }
    }

    void readv(const std::filesystem::path& path, std::uint64_t offset,
        std::span<const std::span<std::byte>> destinations, std::uint64_t total) {
        const auto source = source_for(path);
        if (offset > source->size || total > source->size - offset)
            throw std::runtime_error("MFQ record source was truncated: " + path.string());
        std::size_t index = 0, consumed = 0;
        std::uint64_t done = 0;
        std::vector<iovec> vectors;
        vectors.reserve(std::min<std::size_t>(destinations.size(), IOV_MAX));
        while (done < total) {
            vectors.clear();
            std::size_t batch_bytes = 0;
            constexpr std::size_t maximum_read = std::size_t{32} << 20;
            for (auto next = index; next < destinations.size() && vectors.size() < IOV_MAX
                    && batch_bytes < maximum_read; ++next) {
                const auto start = next == index ? consumed : 0;
                const auto count = std::min(destinations[next].size() - start, maximum_read - batch_bytes);
                if (!count) continue;
                vectors.push_back({destinations[next].data() + start, count});
                batch_bytes += count;
            }
            const auto count = ::preadv(source->open(), vectors.data(), static_cast<int>(vectors.size()),
                static_cast<off_t>(offset + done));
            if (count < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("failed reading MFQ record source: " + path.string()
                    + ": " + std::strerror(errno));
            }
            if (!count) throw std::runtime_error("MFQ record source was truncated: " + path.string());
            done += static_cast<std::uint64_t>(count);
            auto remaining = static_cast<std::size_t>(count);
            while (index < destinations.size()) {
                const auto available = destinations[index].size() - consumed;
                if (remaining < available) { consumed += remaining; break; }
                remaining -= available;
                ++index;
                consumed = 0;
                if (!remaining && (index == destinations.size() || !destinations[index].empty())) break;
            }
        }
    }

    std::mutex mutex;
    std::map<std::filesystem::path, std::shared_ptr<File>> files;
};

void MfqContainer::load_hf_directory(
    const std::filesystem::path& requested_path, const std::string& source_prefix) {
    hf_source_ = std::make_shared<mfq::HfModelSource>(requested_path, source_prefix);
    header_.version = 2;
    header_.architecture = std::string(hf_source_->architecture());
    header_.extra_json = hf_source_->metadata();
    header_.record_count = static_cast<std::uint32_t>(
        hf_source_->tensors().size() + hf_source_->assets().size());
    source_paths_ = hf_source_->source_paths();

    for (const auto& tensor : hf_source_->tensors()) {
        MfqRecord record;
        record.name = tensor.name;
        record.dtype = tensor.dtype;
        record.stored_dtype = tensor.stored_dtype;
        record.source_path = requested_path;
        record.nbytes = tensor.nbytes;
        if (!records_.emplace(record.name, std::move(record)).second) {
            throw std::runtime_error(
                "duplicate canonical HF tensor: " + tensor.name);
        }
    }
    for (const auto& name : hf_source_->assets()) {
        MfqRecord record;
        record.name = name;
        record.dtype = "BLOB";
        record.source_path = requested_path;
        record.nbytes = hf_source_->read_asset(name).size();
        if (!records_.emplace(record.name, std::move(record)).second) {
            throw std::runtime_error("duplicate native HF asset: " + name);
        }
    }
    if (records_.size() > kMaxRecordEntries) {
        throw std::runtime_error("HF checkpoint has too many tensor records");
    }
}

MfqContainer::MfqContainer(std::filesystem::path path, std::string hf_source_prefix)
    : random_access_files_(std::make_shared<RandomAccessFiles>()) {
    std::error_code directory_error;
    if (std::filesystem::is_directory(path, directory_error) &&
        !directory_error) {
        load_hf_directory(path, hf_source_prefix);
        return;
    }
    mfq::MfqModelSource source(std::move(path));
    header_.version = source.header().version;
    header_.architecture = source.header().architecture;
    header_.extra_json = source.header().metadata;
    header_.record_count = source.header().record_count;
    source_paths_ = source.source_paths();
    for (const auto& stored : source.records()) {
        MfqRecord record;
        record.name = stored.tensor.name;
        record.dtype = stored.tensor.dtype;
        record.stored_dtype = stored.tensor.stored_dtype;
        record.source_path = stored.source_path;
        record.offset = stored.offset;
        record.nbytes = stored.tensor.nbytes;
        records_.emplace(record.name, std::move(record));
    }
    install_legacy_tensor_compatibility(*this);
}

bool MfqContainer::contains(const std::string& name) const {
    if (records_.find(name) != records_.end()) {
        return true;
    }
    const auto alias = legacy_aliases_.find(name);
    return alias != legacy_aliases_.end() &&
        records_.find(alias->second) != records_.end();
}

const MfqRecord& MfqContainer::record(const std::string& name) const {
    auto found = records_.find(name);
    if (found == records_.end()) {
        const auto alias = legacy_aliases_.find(name);
        if (alias != legacy_aliases_.end()) {
            found = records_.find(alias->second);
        }
    }
    if (found == records_.end()) {
        throw std::runtime_error("missing MFQ record: " + name);
    }
    return found->second;
}

struct MfqContainer::LoadProgress {
    std::mutex mutex;
    std::unordered_set<std::string> prepared;
    std::function<void(std::size_t, std::size_t)> callback;
    std::size_t total = 0;
    std::size_t last_percent = 0;
};

void MfqContainer::observe_load_records(
    std::function<void(std::size_t, std::size_t)> callback) const {
    auto progress = std::make_shared<LoadProgress>();
    for (const auto& [name, value] : records_) {
        if (!name.starts_with("__mfq_asset__/")) ++progress->total;
    }
    progress->callback = std::move(callback);
    load_progress_ = std::move(progress);
}

void MfqContainer::stop_load_observation() const {
    if (!load_progress_) return;
    std::lock_guard lock(load_progress_->mutex);
    load_progress_->callback = {};
}

void MfqContainer::record_prepared(const std::string& name) const {
    if (!load_progress_) return;
    const auto& stored = record(name).name;
    if (stored.starts_with("__mfq_asset__/")) return;
    std::lock_guard lock(load_progress_->mutex);
    auto& progress = *load_progress_;
    if (!progress.callback || progress.total == 0 || !progress.prepared.insert(stored).second) return;
    const auto percent = progress.prepared.size() * 100 / progress.total;
    if (percent <= progress.last_percent) return;
    progress.last_percent = percent;
    progress.callback(progress.prepared.size(), progress.total);
}

std::vector<std::uint8_t> MfqContainer::read(
    const std::string& name) const {
    const auto& value = record(name);
    auto result = read_range(name, 0, value.nbytes);
    record_prepared(name);
    return result;
}

MfqMappedBytes MfqContainer::map_record(
    const std::string& name, bool random_access) const {
    const auto& value = record(name);
    if (value.nbytes == 0) {
        record_prepared(name);
        return {};
    }
    if (hf_source_) {
        auto bytes = std::make_shared<std::vector<std::uint8_t>>(read(name));
        const auto* data = bytes->data();
        const auto size = bytes->size();
        std::shared_ptr<void> owner = bytes;
        return MfqMappedBytes(std::move(owner), data, size);
    }
    if (
        value.nbytes > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())
        || value.offset > static_cast<std::uint64_t>(
            std::numeric_limits<off_t>::max())
    ) {
        throw std::runtime_error(
            "MFQ record is too large to map: " + name);
    }
    const auto page_size = static_cast<std::uint64_t>(
        ::getpagesize());
    const auto mapped_offset =
        value.offset - value.offset % page_size;
    const auto delta = value.offset - mapped_offset;
    if (
        delta > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())
            - value.nbytes
        || mapped_offset > static_cast<std::uint64_t>(
            std::numeric_limits<off_t>::max())
    ) {
        throw std::runtime_error(
            "MFQ mapped record range overflows: " + name);
    }
    const auto mapped_size = static_cast<std::size_t>(
        delta + value.nbytes);
    const int descriptor = ::open(
        value.source_path.c_str(),
        O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        throw std::runtime_error(
            "cannot open MFQ record for mmap: " + name
            + ": " + std::strerror(errno));
    }
    void* mapping = ::mmap(
        nullptr,
        mapped_size,
        PROT_READ,
        MAP_PRIVATE,
        descriptor,
        static_cast<off_t>(mapped_offset));
    const int map_error = errno;
    ::close(descriptor);
    if (mapping == MAP_FAILED) {
        throw std::runtime_error(
            "cannot mmap MFQ record: " + name
            + ": " + std::strerror(map_error));
    }
    if (random_access && ::madvise(mapping, mapped_size, MADV_RANDOM) != 0) {
        const int advice_error = errno;
        ::munmap(mapping, mapped_size);
        throw std::runtime_error(
            "cannot configure random MFQ mapping: " + name
            + ": " + std::strerror(advice_error));
    }
    auto owner = std::shared_ptr<void>(
        mapping,
        [mapped_size](void* address) {
            ::munmap(address, mapped_size);
        });
    record_prepared(name);
    return MfqMappedBytes(
        std::move(owner),
        static_cast<const std::uint8_t*>(mapping)
            + static_cast<std::size_t>(delta),
        static_cast<std::size_t>(value.nbytes));
}

std::vector<std::uint8_t> MfqContainer::read_range(
    const std::string& name,
    std::uint64_t relative_offset,
    std::uint64_t nbytes) const {
    if (nbytes > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            "MFQ record byte range is too large: " + name);
    }
    std::vector<std::uint8_t> result(static_cast<std::size_t>(nbytes));
    read_range_into(
        name,
        relative_offset,
        std::as_writable_bytes(std::span<std::uint8_t>(result)));
    return result;
}

void MfqContainer::read_range_into(
    const std::string& name,
    std::uint64_t relative_offset,
    std::span<std::byte> destination) const {
    const auto& value = record(name);
    const auto nbytes = static_cast<std::uint64_t>(destination.size());
    if (
        relative_offset > value.nbytes
        || nbytes > value.nbytes - relative_offset
    ) {
        throw std::out_of_range(
            "MFQ record byte range is out of bounds: " + name);
    }
    if (nbytes >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::streamsize>::max())
        || value.offset >
            std::numeric_limits<std::uint64_t>::max()
                - relative_offset
    ) {
        throw std::runtime_error(
            "MFQ record byte range is too large: " + name);
    }
    if (nbytes == 0) {
        return;
    }
    if (hf_source_) {
        if (hf_source_->has_asset(value.name)) {
            const auto bytes = hf_source_->read_asset(value.name);
            std::copy_n(
                bytes.data() + relative_offset,
                destination.size(),
                destination.data());
        } else {
            hf_source_->read_range_into(
                value.name,
                relative_offset,
                destination.data(),
                destination.size());
        }
        return;
    }
    const auto absolute_offset =
        value.offset + relative_offset;
    if (
        absolute_offset >
        static_cast<std::uint64_t>(
            std::numeric_limits<std::streamoff>::max())
    ) {
        throw std::runtime_error(
            "MFQ record byte offset is too large: " + name);
    }
    random_access_files_->read(
        value.source_path, absolute_offset, destination);
}

void MfqContainer::readv_range(const std::string& name, std::uint64_t relative_offset,
    std::span<const std::span<std::byte>> destinations) const {
    const auto& value = record(name);
    if (relative_offset > value.nbytes)
        throw std::out_of_range("MFQ record byte range is out of bounds: " + name);
    std::uint64_t total = 0;
    for (const auto destination : destinations) {
        if (destination.size() > value.nbytes - relative_offset - total)
            throw std::out_of_range("MFQ record byte range is out of bounds: " + name);
        total += destination.size();
    }
    if (!total) return;
    const auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max());
    if (value.offset > maximum || relative_offset > maximum - value.offset
        || total > maximum - value.offset - relative_offset)
        throw std::runtime_error("MFQ record byte range is too large: " + name);
    if (hf_source_) {
        for (const auto destination : destinations) {
            read_range_into(name, relative_offset, destination);
            relative_offset += destination.size();
        }
    } else random_access_files_->readv(value.source_path, value.offset + relative_offset, destinations, total);
}


std::string MfqContainer::read_text(const std::string& name) const {
    const auto bytes = read(name);
    return std::string(bytes.begin(), bytes.end());
}

void MfqContainer::drop_source_file_cache() const noexcept {
    if (hf_source_) hf_source_->drop_file_cache();
}

std::optional<mfq::MfqModelGraph> MfqContainer::model_graph() const {
    const std::string asset(mfq::kMfqModelGraphAsset);
    if (!contains(asset)) return std::nullopt;
    return mfq::MfqModelGraph::from_json(read_text(asset));
}

void MfqContainer::install_legacy_aliases(
        std::unordered_map<std::string, std::string> canonical_to_stored,
        mfq::MfqLegacyTensorLayout layout) {
    if (model_graph()) {
        throw std::invalid_argument(
            "canonical MFQ model graphs cannot install legacy tensor aliases");
    }
    for (const auto& [canonical, stored] : canonical_to_stored) {
        if (canonical.empty() || stored.empty() || canonical == stored) {
            throw std::invalid_argument("invalid legacy MFQ tensor alias");
        }
        if (records_.find(canonical) != records_.end()) {
            throw std::invalid_argument(
                "legacy MFQ alias shadows a stored canonical tensor: " + canonical);
        }
        if (records_.find(stored) == records_.end()) {
            throw std::invalid_argument(
                "legacy MFQ alias refers to a missing tensor: " + stored);
        }
        if (canonical_to_stored.find(stored) != canonical_to_stored.end()) {
            throw std::invalid_argument(
                "legacy MFQ tensor alias chains are forbidden: " + canonical);
        }
        const auto existing = legacy_aliases_.find(canonical);
        if (existing != legacy_aliases_.end() && existing->second != stored) {
            throw std::invalid_argument(
                "legacy MFQ tensor alias is ambiguous: " + canonical);
        }
    }
    legacy_aliases_.insert(
        std::make_move_iterator(canonical_to_stored.begin()),
        std::make_move_iterator(canonical_to_stored.end()));
    legacy_tensor_layout_ = layout;
}


} // namespace mfq::metal
