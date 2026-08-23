#include "hf_safetensors_store.h"

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <sys/uio.h>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace mfq::metal {
namespace {

using json = nlohmann::json;

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path.string());
    }
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < 0) {
        throw std::runtime_error("cannot size " + path.string());
    }
    std::string result(static_cast<std::size_t>(end), '\0');
    stream.seekg(0, std::ios::beg);
    stream.read(result.data(), static_cast<std::streamsize>(result.size()));
    if (!stream && !result.empty()) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return result;
}

json parse_json(std::string_view payload, const std::filesystem::path& path) {
    try {
        return json::parse(payload.begin(), payload.end());
    } catch (const json::exception& error) {
        throw std::runtime_error(
            "invalid JSON in " + path.string() + ": " + error.what());
    }
}

std::uint64_t little_u64(const std::array<unsigned char, 8>& bytes) {
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        result |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
    }
    return result;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("Safetensors byte offset overflow");
    }
    return left + right;
}

std::vector<std::int64_t> parse_shape(
    const json& value,
    const std::string& name) {
    if (!value.is_array()) {
        throw std::runtime_error("Safetensors shape is not an array: " + name);
    }
    std::vector<std::int64_t> result;
    result.reserve(value.size());
    for (const auto& dimension : value) {
        if (!dimension.is_number_integer()) {
            throw std::runtime_error(
                "Safetensors shape has a non-integer dimension: " + name);
        }
        const auto size = dimension.get<std::int64_t>();
        if (size < 0) {
            throw std::runtime_error(
                "Safetensors shape has a negative dimension: " + name);
        }
        result.push_back(size);
    }
    return result;
}

void pread_exact(
    int fd,
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::span<std::byte> destination) {
    std::size_t done = 0;
    while (done < destination.size()) {
        const auto request = std::min<std::size_t>(
            destination.size() - done,
            static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const auto result = ::pread(
            fd,
            destination.data() + done,
            request,
            static_cast<off_t>(offset + done));
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::system_error(
                errno,
                std::generic_category(),
                "pread " + path.string());
        }
        if (result == 0) {
            throw std::runtime_error(
                "unexpected EOF while reading " + path.string());
        }
        done += static_cast<std::size_t>(result);
    }
}

void preadv_exact(
    int fd,
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::span<const std::span<std::byte>> destinations) {
    std::vector<iovec> vectors;
    vectors.reserve(destinations.size());
    std::uint64_t total = 0;
    for (const auto destination : destinations) {
        if (destination.empty()) {
            continue;
        }
        vectors.push_back({
            .iov_base = destination.data(),
            .iov_len = destination.size(),
        });
        total = checked_add(total, destination.size());
    }
    if (vectors.empty()) {
        return;
    }
    if (vectors.size() > IOV_MAX) {
        throw std::runtime_error("preadv destination count exceeds IOV_MAX");
    }
    std::size_t vector_index = 0;
    std::size_t vector_offset = 0;
    std::uint64_t done = 0;
    while (done < total) {
        std::vector<iovec> remaining;
        remaining.reserve(vectors.size() - vector_index);
        auto first = vectors[vector_index];
        first.iov_base = static_cast<std::byte*>(first.iov_base) + vector_offset;
        first.iov_len -= vector_offset;
        remaining.push_back(first);
        remaining.insert(
            remaining.end(),
            vectors.begin() + static_cast<std::ptrdiff_t>(vector_index + 1),
            vectors.end());
        const auto result = ::preadv(
            fd,
            remaining.data(),
            static_cast<int>(remaining.size()),
            static_cast<off_t>(offset + done));
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::system_error(
                errno,
                std::generic_category(),
                "preadv " + path.string());
        }
        if (result == 0) {
            throw std::runtime_error(
                "unexpected EOF while reading " + path.string());
        }
        done += static_cast<std::uint64_t>(result);
        auto consumed = static_cast<std::size_t>(result);
        while (consumed != 0) {
            const auto available = vectors[vector_index].iov_len - vector_offset;
            if (consumed < available) {
                vector_offset += consumed;
                consumed = 0;
            } else {
                consumed -= available;
                ++vector_index;
                vector_offset = 0;
            }
        }
    }
}

std::string expert_tensor_name(
    std::size_t layer,
    std::size_t expert,
    std::string_view suffix) {
    return "layers." + std::to_string(layer) + ".ffn.experts." +
        std::to_string(expert) + "." + std::string(suffix);
}

void require_tensor(
    const HfSafetensorRecord& record,
    std::string_view dtype,
    std::initializer_list<std::int64_t> shape) {
    if (record.dtype != dtype ||
        record.shape != std::vector<std::int64_t>(shape)) {
        throw std::runtime_error(
            "unexpected native DeepSeek-V4 expert tensor " + record.name);
    }
}

} // namespace

struct HfSafetensorStore::Impl {
    struct Shard {
        std::filesystem::path path;
        std::uint64_t size = 0;
        mutable int fd = -1;
        mutable std::mutex mutex;

        ~Shard() {
            if (fd >= 0) {
                ::close(fd);
            }
        }

        int open_direct() const {
            std::scoped_lock lock(mutex);
            if (fd >= 0) {
                return fd;
            }
            fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) {
                throw std::system_error(
                    errno,
                    std::generic_category(),
                    "open " + path.string());
            }
#if defined(__APPLE__) && defined(F_NOCACHE)
            if (::fcntl(fd, F_NOCACHE, 1) != 0) {
                const auto saved = errno;
                ::close(fd);
                fd = -1;
                throw std::system_error(
                    saved,
                    std::generic_category(),
                    "F_NOCACHE " + path.string());
            }
#endif
            return fd;
        }
    };

    std::filesystem::path root;
    std::vector<std::unique_ptr<Shard>> shards;
    std::unordered_map<std::string, HfSafetensorRecord> tensors;
};

HfSafetensorStore::HfSafetensorStore(std::filesystem::path root)
    : impl_(std::make_unique<Impl>()) {
    impl_->root = std::filesystem::canonical(std::move(root));
    const auto index_path = impl_->root / "model.safetensors.index.json";
    const auto index = parse_json(read_text_file(index_path), index_path);
    const auto weight_map = index.find("weight_map");
    if (weight_map == index.end() || !weight_map->is_object()) {
        throw std::runtime_error(
            "Safetensors index requires an object weight_map: " +
            index_path.string());
    }

    std::unordered_map<std::string, std::vector<std::string>> names_by_shard;
    for (const auto& [name, shard] : weight_map->items()) {
        if (!shard.is_string()) {
            throw std::runtime_error(
                "Safetensors weight_map value is not a string: " + name);
        }
        names_by_shard[shard.get<std::string>()].push_back(name);
    }

    std::vector<std::string> shard_names;
    shard_names.reserve(names_by_shard.size());
    for (const auto& [name, unused] : names_by_shard) {
        static_cast<void>(unused);
        shard_names.push_back(name);
    }
    std::sort(shard_names.begin(), shard_names.end());

    for (const auto& shard_name : shard_names) {
        const auto path = impl_->root / shard_name;
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("cannot open " + path.string());
        }
        std::array<unsigned char, 8> header_size_bytes{};
        stream.read(
            reinterpret_cast<char*>(header_size_bytes.data()),
            static_cast<std::streamsize>(header_size_bytes.size()));
        if (!stream) {
            throw std::runtime_error(
                "cannot read Safetensors header size: " + path.string());
        }
        const auto header_size = little_u64(header_size_bytes);
        const auto file_size = std::filesystem::file_size(path);
        if (header_size > file_size - header_size_bytes.size()) {
            throw std::runtime_error(
                "invalid Safetensors header size: " + path.string());
        }
        std::string header(static_cast<std::size_t>(header_size), '\0');
        stream.read(header.data(), static_cast<std::streamsize>(header.size()));
        if (!stream && !header.empty()) {
            throw std::runtime_error(
                "cannot read Safetensors header: " + path.string());
        }
        const auto metadata = parse_json(header, path);
        const auto shard_index = impl_->shards.size();
        auto shard = std::make_unique<Impl::Shard>();
        shard->path = path;
        shard->size = file_size;
        impl_->shards.push_back(std::move(shard));
        const auto payload_base = checked_add(8, header_size);

        for (const auto& name : names_by_shard.at(shard_name)) {
            const auto found = metadata.find(name);
            if (found == metadata.end() || !found->is_object()) {
                throw std::runtime_error(
                    "Safetensors index references a missing tensor: " + name);
            }
            const auto dtype = found->find("dtype");
            const auto shape = found->find("shape");
            const auto offsets = found->find("data_offsets");
            if (dtype == found->end() || !dtype->is_string() ||
                shape == found->end() ||
                offsets == found->end() || !offsets->is_array() ||
                offsets->size() != 2 ||
                !(*offsets)[0].is_number_unsigned() ||
                !(*offsets)[1].is_number_unsigned()) {
                throw std::runtime_error(
                    "invalid Safetensors tensor metadata: " + name);
            }
            const auto begin = (*offsets)[0].get<std::uint64_t>();
            const auto end = (*offsets)[1].get<std::uint64_t>();
            if (end < begin || checked_add(payload_base, end) > file_size) {
                throw std::runtime_error(
                    "Safetensors tensor range is outside its shard: " + name);
            }
            HfSafetensorRecord record;
            record.name = name;
            record.dtype = dtype->get<std::string>();
            record.shape = parse_shape(*shape, name);
            record.shard = shard_index;
            record.offset = checked_add(payload_base, begin);
            record.nbytes = end - begin;
            impl_->tensors.emplace(name, std::move(record));
        }
    }
}

HfSafetensorStore::~HfSafetensorStore() = default;
HfSafetensorStore::HfSafetensorStore(HfSafetensorStore&&) noexcept = default;
HfSafetensorStore& HfSafetensorStore::operator=(
    HfSafetensorStore&&) noexcept = default;

const std::filesystem::path& HfSafetensorStore::root() const noexcept {
    return impl_->root;
}

std::size_t HfSafetensorStore::tensor_count() const noexcept {
    return impl_->tensors.size();
}

std::size_t HfSafetensorStore::shard_count() const noexcept {
    return impl_->shards.size();
}

const HfSafetensorRecord& HfSafetensorStore::tensor(
    std::string_view name) const {
    const auto found = impl_->tensors.find(std::string(name));
    if (found == impl_->tensors.end()) {
        throw std::runtime_error(
            "tensor not found in Safetensors checkpoint: " +
            std::string(name));
    }
    return found->second;
}

const std::filesystem::path& HfSafetensorStore::shard_path(
    std::size_t shard) const {
    if (shard >= impl_->shards.size()) {
        throw std::out_of_range("Safetensors shard index out of range");
    }
    return impl_->shards[shard]->path;
}

void HfSafetensorStore::read_tensor(
    const HfSafetensorRecord& record,
    std::span<std::byte> destination) const {
    if (destination.size() != record.nbytes) {
        throw std::runtime_error(
            "Safetensors destination size mismatch for " + record.name);
    }
    read_range(record.shard, record.offset, destination);
}

void HfSafetensorStore::read_range(
    std::size_t shard,
    std::uint64_t offset,
    std::span<std::byte> destination) const {
    if (shard >= impl_->shards.size()) {
        throw std::out_of_range("Safetensors shard index out of range");
    }
    const auto& source = *impl_->shards[shard];
    if (offset > source.size || destination.size() > source.size - offset) {
        throw std::runtime_error(
            "Safetensors range is outside shard " + source.path.string());
    }
    pread_exact(source.open_direct(), source.path, offset, destination);
}

void HfSafetensorStore::readv_range(
    std::size_t shard,
    std::uint64_t offset,
    std::span<const std::span<std::byte>> destinations) const {
    if (shard >= impl_->shards.size()) {
        throw std::out_of_range("Safetensors shard index out of range");
    }
    std::uint64_t bytes = 0;
    for (const auto destination : destinations) {
        bytes = checked_add(bytes, destination.size());
    }
    const auto& source = *impl_->shards[shard];
    if (offset > source.size || bytes > source.size - offset) {
        throw std::runtime_error(
            "Safetensors vector range is outside shard " + source.path.string());
    }
    preadv_exact(
        source.open_direct(),
        source.path,
        offset,
        destinations);
}

void HfSafetensorStore::drop_file_cache() const noexcept {
    for (const auto& shard : impl_->shards) {
        try {
            const auto fd = shard->open_direct();
#if defined(POSIX_FADV_DONTNEED)
            static_cast<void>(::posix_fadvise(
                fd,
                0,
                static_cast<off_t>(shard->size),
                POSIX_FADV_DONTNEED));
#else
            static_cast<void>(fd);
#endif
        } catch (...) {
        }
    }
}

struct DeepseekV4NativeExpertStore::ExpertRecord {
    std::array<const HfSafetensorRecord*, kParts> parts{};
};

DeepseekV4NativeExpertStore::DeepseekV4NativeExpertStore(
    std::filesystem::path root,
    std::size_t num_layers,
    std::size_t num_experts)
    : checkpoint_(std::move(root)),
      num_layers_(num_layers),
      num_experts_(num_experts) {
    if (num_layers_ == 0 || num_experts_ == 0) {
        throw std::invalid_argument(
            "DeepSeek-V4 expert store dimensions must be positive");
    }
    experts_.resize(num_layers_ * num_experts_);
    constexpr std::array<std::string_view, kParts> suffixes = {
        "w1.scale", "w2.scale", "w3.scale",
        "w1.weight", "w2.weight", "w3.weight",
    };

    for (std::size_t layer = 0; layer < num_layers_; ++layer) {
        for (std::size_t expert = 0; expert < num_experts_; ++expert) {
            auto& record = experts_[layer * num_experts_ + expert];
            for (std::size_t part = 0; part < suffixes.size(); ++part) {
                record.parts[part] = &checkpoint_.tensor(
                    expert_tensor_name(layer, expert, suffixes[part]));
            }
            require_tensor(*record.parts[0], "F8_E8M0", {2048, 128});
            require_tensor(*record.parts[1], "F8_E8M0", {4096, 64});
            require_tensor(*record.parts[2], "F8_E8M0", {2048, 128});
            require_tensor(*record.parts[3], "I8", {2048, 2048});
            require_tensor(*record.parts[4], "I8", {4096, 1024});
            require_tensor(*record.parts[5], "I8", {2048, 2048});
            for (std::size_t part = 1; part < 3; ++part) {
                if (record.parts[part]->shard != record.parts[0]->shard ||
                    record.parts[part]->offset !=
                        record.parts[part - 1]->offset +
                            record.parts[part - 1]->nbytes) {
                    throw std::runtime_error(
                        "DeepSeek-V4 expert scales are not contiguous: " +
                        record.parts[part]->name);
                }
            }
            for (std::size_t part = 4; part < 6; ++part) {
                if (record.parts[part]->shard != record.parts[3]->shard ||
                    record.parts[part]->offset !=
                        record.parts[part - 1]->offset +
                            record.parts[part - 1]->nbytes) {
                    throw std::runtime_error(
                        "DeepSeek-V4 expert weights are not contiguous: " +
                        record.parts[part]->name);
                }
            }
            if (record.parts[0]->shard != record.parts[3]->shard) {
                throw std::runtime_error(
                    "DeepSeek-V4 expert parts span multiple shards: " +
                    record.parts[0]->name);
            }
        }
    }

    const auto& first = experts_.front();
    for (std::size_t part = 0; part < kParts; ++part) {
        slot_offsets_[part + 1] =
            slot_offsets_[part] + first.parts[part]->nbytes;
    }
    slot_bytes_ = slot_offsets_.back();
    for (const auto& expert : experts_) {
        for (std::size_t part = 0; part < kParts; ++part) {
            if (expert.parts[part]->nbytes != first.parts[part]->nbytes) {
                throw std::runtime_error(
                    "DeepSeek-V4 expert tensors have inconsistent sizes");
            }
        }
    }
}

DeepseekV4NativeExpertStore::~DeepseekV4NativeExpertStore() = default;

const HfSafetensorStore& DeepseekV4NativeExpertStore::checkpoint() const noexcept {
    return checkpoint_;
}

std::size_t DeepseekV4NativeExpertStore::num_layers() const noexcept {
    return num_layers_;
}

std::size_t DeepseekV4NativeExpertStore::num_experts() const noexcept {
    return num_experts_;
}

std::size_t DeepseekV4NativeExpertStore::slot_bytes() const noexcept {
    return slot_bytes_;
}

const DeepseekV4NativeExpertStore::ExpertRecord&
DeepseekV4NativeExpertStore::expert_record(
    std::size_t layer,
    std::size_t expert) const {
    if (layer >= num_layers_ || expert >= num_experts_) {
        throw std::out_of_range("DeepSeek-V4 expert index out of range");
    }
    return experts_[layer * num_experts_ + expert];
}

DeepseekV4NativeExpertLoadStats DeepseekV4NativeExpertStore::load(
    std::size_t layer,
    std::size_t expert,
    std::span<std::byte> slot) const {
    if (slot.size() < slot_bytes_) {
        throw std::runtime_error("DeepSeek-V4 expert slot is too small");
    }
    DeepseekV4NativeExpertDestination destination{
        .w1_scale = slot.subspan(
            slot_offsets_[0], slot_offsets_[1] - slot_offsets_[0]),
        .w2_scale = slot.subspan(
            slot_offsets_[1], slot_offsets_[2] - slot_offsets_[1]),
        .w3_scale = slot.subspan(
            slot_offsets_[2], slot_offsets_[3] - slot_offsets_[2]),
        .w1_weight = slot.subspan(
            slot_offsets_[3], slot_offsets_[4] - slot_offsets_[3]),
        .w2_weight = slot.subspan(
            slot_offsets_[4], slot_offsets_[5] - slot_offsets_[4]),
        .w3_weight = slot.subspan(
            slot_offsets_[5], slot_offsets_[6] - slot_offsets_[5]),
    };
    return load_scatter(layer, expert, destination);
}

DeepseekV4NativeExpertLoadStats DeepseekV4NativeExpertStore::load_scatter(
    std::size_t layer,
    std::size_t expert,
    const DeepseekV4NativeExpertDestination& destination) const {
    const auto& record = expert_record(layer, expert);
    const std::array<std::span<std::byte>, 3> scales{
        destination.w1_scale,
        destination.w2_scale,
        destination.w3_scale,
    };
    const std::array<std::span<std::byte>, 3> weights{
        destination.w1_weight,
        destination.w2_weight,
        destination.w3_weight,
    };
    const std::array<std::span<std::byte>, kParts> all{
        destination.w1_scale,
        destination.w2_scale,
        destination.w3_scale,
        destination.w1_weight,
        destination.w2_weight,
        destination.w3_weight,
    };
    std::uint64_t bytes = 0;
    for (std::size_t part = 0; part < kParts; ++part) {
        if (all[part].size() != record.parts[part]->nbytes) {
            throw std::runtime_error(
                "DeepSeek-V4 expert destination size mismatch for " +
                record.parts[part]->name);
        }
        bytes = checked_add(bytes, all[part].size());
    }
    checkpoint_.readv_range(
        record.parts[0]->shard,
        record.parts[0]->offset,
        scales);
    checkpoint_.readv_range(
        record.parts[3]->shard,
        record.parts[3]->offset,
        weights);
    return {
        .bytes = bytes,
        .read_calls = 2,
    };
}

DeepseekV4NativeExpertView DeepseekV4NativeExpertStore::view(
    std::span<const std::byte> slot) const {
    if (slot.size() < slot_bytes_) {
        throw std::runtime_error("DeepSeek-V4 expert slot is too small");
    }
    return {
        .w1_scale = slot.subspan(
            slot_offsets_[0], slot_offsets_[1] - slot_offsets_[0]),
        .w2_scale = slot.subspan(
            slot_offsets_[1], slot_offsets_[2] - slot_offsets_[1]),
        .w3_scale = slot.subspan(
            slot_offsets_[2], slot_offsets_[3] - slot_offsets_[2]),
        .w1_weight = slot.subspan(
            slot_offsets_[3], slot_offsets_[4] - slot_offsets_[3]),
        .w2_weight = slot.subspan(
            slot_offsets_[4], slot_offsets_[5] - slot_offsets_[4]),
        .w3_weight = slot.subspan(
            slot_offsets_[5], slot_offsets_[6] - slot_offsets_[5]),
    };
}

} // namespace mfq::metal
