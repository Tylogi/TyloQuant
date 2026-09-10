#include "mfq_paged_prefix_cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#endif

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mfq::cache {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'M', 'F', 'Q', 'K', 'V', 'B', '1', 0};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint64_t kHeaderBytes =
    8 + 4 + 32 + 32 + 32 + 4 + 4 + 8 + 32;
constexpr std::string_view kHashDomain = "mfq-paged-prefix-v1";

constexpr std::array<std::uint32_t, 64> kShaConstants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

std::uint32_t rotate_right(std::uint32_t value, int count) {
    return (value >> count) | (value << (32 - count));
}

class Sha256State {
public:
    Sha256State()
        : state_{
              0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

    void update(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        total_bytes_ += size;
        while (size > 0) {
            const auto count = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, bytes, count);
            block_size_ += count;
            bytes += count;
            size -= count;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    BlockHash finish() {
        const std::uint64_t bit_count = total_bytes_ * 8;
        block_[block_size_++] = 0x80;
        if (block_size_ > 56) {
            std::fill(block_.begin() + block_size_, block_.end(), 0);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + block_size_, block_.begin() + 56, 0);
        for (int index = 0; index < 8; ++index) {
            block_[63 - index] = static_cast<std::uint8_t>(
                bit_count >> (index * 8));
        }
        transform(block_.data());
        BlockHash digest{};
        for (std::size_t index = 0; index < state_.size(); ++index) {
            digest[index * 4] = static_cast<std::uint8_t>(state_[index] >> 24);
            digest[index * 4 + 1] =
                static_cast<std::uint8_t>(state_[index] >> 16);
            digest[index * 4 + 2] =
                static_cast<std::uint8_t>(state_[index] >> 8);
            digest[index * 4 + 3] =
                static_cast<std::uint8_t>(state_[index]);
        }
        return digest;
    }

private:
    void transform(const std::uint8_t* block) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            words[index] =
                (static_cast<std::uint32_t>(block[index * 4]) << 24) |
                (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16) |
                (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8) |
                static_cast<std::uint32_t>(block[index * 4 + 3]);
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const auto s0 = rotate_right(words[index - 15], 7) ^
                rotate_right(words[index - 15], 18) ^
                (words[index - 15] >> 3);
            const auto s1 = rotate_right(words[index - 2], 17) ^
                rotate_right(words[index - 2], 19) ^
                (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const auto sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                rotate_right(e, 25);
            const auto choose = (e & f) ^ (~e & g);
            const auto temp1 = h + sum1 + choose + kShaConstants[index] +
                words[index];
            const auto sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                rotate_right(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_ = 0;
    std::uint64_t total_bytes_ = 0;
};

template <typename T>
void hash_little_endian(Sha256State& state, T value) {
    std::array<std::uint8_t, sizeof(T)> bytes{};
    using Unsigned = std::make_unsigned_t<T>;
    const auto converted = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>(converted >> (index * 8));
    }
    state.update(bytes.data(), bytes.size());
}

template <typename T>
void write_little_endian(std::ostream& output, T value) {
    std::array<std::uint8_t, sizeof(T)> bytes{};
    using Unsigned = std::make_unsigned_t<T>;
    const auto converted = static_cast<Unsigned>(value);
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::uint8_t>(converted >> (index * 8));
    }
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
}

template <typename T>
bool read_little_endian(std::istream& input, T& value) {
    std::array<std::uint8_t, sizeof(T)> bytes{};
    input.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!input) return false;
    using Unsigned = std::make_unsigned_t<T>;
    Unsigned converted = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        converted |= static_cast<Unsigned>(bytes[index]) << (index * 8);
    }
    value = static_cast<T>(converted);
    return true;
}

bool zero_hash(const BlockHash& hash) {
    return std::all_of(hash.begin(), hash.end(), [](std::uint8_t value) {
        return value == 0;
    });
}

struct HashHasher {
    std::size_t operator()(const BlockHash& hash) const noexcept {
        std::size_t value = 0;
        std::memcpy(&value, hash.data(), std::min(sizeof(value), hash.size()));
        return value;
    }
};

std::uint64_t monotonic_tick() {
    return static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

bool sync_regular_file(const std::filesystem::path& path) {
#if defined(_WIN32)
    const int descriptor = _wopen(
        path.c_str(), _O_RDWR | _O_BINARY);
    if (descriptor < 0) return false;
    const bool synced = _commit(descriptor) == 0;
    (void)_close(descriptor);
    return synced;
#else
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) return false;
#if defined(__APPLE__) && defined(F_FULLFSYNC)
    int result = ::fcntl(descriptor, F_FULLFSYNC);
    if (result != 0) result = ::fsync(descriptor);
#else
    const int result = ::fsync(descriptor);
#endif
    (void)::close(descriptor);
    return result == 0;
#endif
}

void sync_directory_best_effort(const std::filesystem::path& path) noexcept {
#if !defined(_WIN32)
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) return;
    (void)::fsync(descriptor);
    (void)::close(descriptor);
#else
    (void)path;
#endif
}

} // namespace

BlockHash sha256(const void* data, std::size_t size) {
#if defined(__APPLE__)
    CC_SHA256_CTX state;
    CC_SHA256_Init(&state);
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    while (size > 0) {
        const auto count = std::min<std::size_t>(
            size, std::numeric_limits<CC_LONG>::max());
        CC_SHA256_Update(
            &state, bytes, static_cast<CC_LONG>(count));
        bytes += count;
        size -= count;
    }
    BlockHash digest{};
    CC_SHA256_Final(digest.data(), &state);
    return digest;
#else
    Sha256State state;
    state.update(data, size);
    return state.finish();
#endif
}

BlockHash sha256(std::string_view value) {
    return sha256(value.data(), value.size());
}

std::string block_hash_hex(const BlockHash& hash) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const auto byte : hash) {
        stream << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return stream.str();
}

class PagedPrefixCache::Implementation {
private:
    struct DiskEntry {
        std::filesystem::path path;
        BlockHash parent{};
        std::uint32_t token_count = 0;
        std::uint64_t payload_bytes = 0;
        std::uint64_t file_bytes = 0;
        std::uint64_t last_used = 0;
        std::size_t pins = 0;
    };

    struct HotEntry {
        std::shared_ptr<const std::vector<std::uint8_t>> payload;
        std::uint64_t last_used = 0;
    };

    struct WriteRequest {
        BlockHash hash{};
        BlockHash parent{};
        std::uint32_t token_count = 0;
        std::shared_ptr<const std::vector<std::uint8_t>> payload;
    };

    struct ParsedHeader {
        BlockHash compatibility{};
        BlockHash hash{};
        BlockHash parent{};
        std::uint32_t block_size = 0;
        std::uint32_t token_count = 0;
        std::uint64_t payload_bytes = 0;
        BlockHash payload_hash{};
    };

    enum class LoadSource {
        Hot,
        Pending,
        Disk,
    };

    struct LoadRequest {
        BlockHash hash{};
        LoadSource source = LoadSource::Disk;
        PagedPrefixPayload payload;
        std::filesystem::path path;
    };

public:
    explicit Implementation(PagedPrefixCacheConfig config)
        : config_(std::move(config)),
          compatibility_hash_(sha256(config_.compatibility_key)) {
        if (config_.cache_dir.empty()) {
            throw std::invalid_argument("paged prefix cache directory is empty");
        }
        if (config_.compatibility_key.empty()) {
            throw std::invalid_argument("paged prefix compatibility key is empty");
        }
        if (config_.block_size_tokens == 0 ||
            config_.block_size_tokens >
                static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::invalid_argument("invalid paged prefix block size");
        }
        if (config_.max_pending_writes == 0) {
            config_.max_pending_writes = 1;
        }
        if (config_.max_parallel_reads == 0) {
            config_.max_parallel_reads = 1;
        }
        namespace_dir_ = config_.cache_dir /
            block_hash_hex(compatibility_hash_);
        std::error_code error;
        std::filesystem::create_directories(namespace_dir_, error);
        if (error) {
            throw std::runtime_error(
                "cannot create paged prefix cache directory: " +
                error.message());
        }
        std::filesystem::permissions(
            namespace_dir_,
            std::filesystem::perms::owner_all,
            std::filesystem::perm_options::replace,
            error);
        scan();
        worker_ = std::thread([this] { writer_loop(); });
    }

    ~Implementation() {
        flush();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        work_available_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    std::size_t block_size_tokens() const noexcept {
        return config_.block_size_tokens;
    }

    const BlockHash& compatibility_hash() const noexcept {
        return compatibility_hash_;
    }

    BlockHash block_hash(
        const BlockHash& parent,
        const std::int64_t* token_ids,
        std::size_t token_count,
        std::string_view extra_key) const {
        if (token_ids == nullptr || token_count == 0 ||
            token_count > config_.block_size_tokens) {
            throw std::invalid_argument("invalid paged prefix token block");
        }
        Sha256State state;
        state.update(kHashDomain.data(), kHashDomain.size());
        state.update(compatibility_hash_.data(), compatibility_hash_.size());
        state.update(parent.data(), parent.size());
        hash_little_endian<std::uint64_t>(state, extra_key.size());
        state.update(extra_key.data(), extra_key.size());
        hash_little_endian<std::uint64_t>(state, token_count);
        for (std::size_t index = 0; index < token_count; ++index) {
            hash_little_endian<std::int64_t>(state, token_ids[index]);
        }
        return state.finish();
    }

    PrefixMatch match(
        const std::vector<std::int64_t>& token_ids,
        std::string_view extra_key,
        bool record_query) {
        PrefixMatch result;
        BlockHash parent{};
        const auto full_blocks = token_ids.size() / config_.block_size_tokens;
        // Hashing a long prompt is pure CPU work. Keep it outside the cache
        // mutex so SSD writers, hot-cache hits, and unrelated requests are
        // not serialized behind the full token walk. Check after each block
        // so a cold miss still stops hashing immediately.
        if (record_query) {
            std::lock_guard<std::mutex> lock(mutex_);
            ++metrics_.queries;
        }
        for (std::size_t index = 0; index < full_blocks; ++index) {
            const auto offset = index * config_.block_size_tokens;
            parent = block_hash(
                parent,
                token_ids.data() + offset,
                config_.block_size_tokens,
                extra_key);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (disk_.count(parent) == 0 &&
                    pending_.count(parent) == 0) {
                    break;
                }
            }
            result.blocks.push_back(parent);
            result.matched_tokens += config_.block_size_tokens;
        }
        if (record_query && result.matched_tokens > 0) {
            std::lock_guard<std::mutex> lock(mutex_);
            ++metrics_.hits;
            metrics_.hit_tokens += result.matched_tokens;
        }
        return result;
    }

    BlockHash store(
        const BlockHash& parent,
        const std::int64_t* token_ids,
        std::size_t token_count,
        std::shared_ptr<const std::vector<std::uint8_t>> payload,
        std::string_view extra_key) {
        if (!payload) {
            throw std::invalid_argument("paged prefix payload is null");
        }
        const auto hash = block_hash(
            parent, token_ids, token_count, extra_key);
        bool write_inline = false;
        const auto payload_bytes = static_cast<std::uint64_t>(
            payload->size());
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (disk_.count(hash) != 0 || pending_.count(hash) != 0) {
                ++metrics_.deduplicated_writes;
                return hash;
            }
            put_hot_locked(hash, payload);
            const bool pending_bytes_full =
                config_.max_pending_bytes == 0 ||
                payload_bytes > config_.max_pending_bytes ||
                pending_write_bytes_ >
                    config_.max_pending_bytes - payload_bytes;
            if (writes_.size() >= config_.max_pending_writes ||
                pending_bytes_full) {
                write_inline = true;
                pending_[hash] = payload;
            } else {
                writes_.push_back(WriteRequest{
                    hash,
                    parent,
                    static_cast<std::uint32_t>(token_count),
                    std::move(payload),
                });
                pending_[hash] = writes_.back().payload;
            }
            pending_write_bytes_ += payload_bytes;
            sync_metrics_locked();
        }
        if (write_inline) {
            const WriteRequest request{
                hash,
                parent,
                static_cast<std::uint32_t>(token_count),
                std::move(payload),
            };
            finish_write(request, write_file(request));
        } else {
            work_available_.notify_one();
        }
        return hash;
    }

    std::optional<std::vector<std::uint8_t>> load(const BlockHash& hash) {
        const auto payloads = load_prefix({hash});
        if (payloads.empty()) return std::nullopt;
        return *payloads.front();
    }

    std::vector<PagedPrefixPayload> load_prefix(
        const std::vector<BlockHash>& hashes) {
        std::vector<LoadRequest> requests;
        requests.reserve(hashes.size());
        std::uint64_t load_epoch = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            load_epoch = cache_epoch_;
            for (const auto& hash : hashes) {
                const auto hot = hot_.find(hash);
                if (hot != hot_.end()) {
                    requests.push_back(LoadRequest{
                        hash,
                        LoadSource::Hot,
                        hot->second.payload,
                        {},
                    });
                    continue;
                }
                const auto queued = pending_.find(hash);
                if (queued != pending_.end()) {
                    requests.push_back(LoadRequest{
                        hash,
                        LoadSource::Pending,
                        queued->second,
                        {},
                    });
                    continue;
                }
                const auto found = disk_.find(hash);
                if (found == disk_.end()) break;
                requests.push_back(LoadRequest{
                    hash,
                    LoadSource::Disk,
                    {},
                    found->second.path,
                });
            }
        }

        std::vector<std::optional<std::vector<std::uint8_t>>> cold_payloads(
            requests.size());
        std::vector<std::size_t> cold_indices;
        cold_indices.reserve(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) {
            if (requests[index].source == LoadSource::Disk) {
                cold_indices.push_back(index);
            }
        }

        const auto read_cold = [&](std::size_t cold_index) {
            const auto request_index = cold_indices[cold_index];
            const auto& request = requests[request_index];
            cold_payloads[request_index] = read_payload(
                request.path, request.hash);
        };
        if (cold_indices.size() == 1 || config_.max_parallel_reads == 1) {
            for (std::size_t index = 0; index < cold_indices.size(); ++index) {
                read_cold(index);
            }
        } else if (!cold_indices.empty()) {
            const auto worker_count = std::min(
                cold_indices.size(), config_.max_parallel_reads);
            std::atomic<std::size_t> next{0};
            std::mutex error_mutex;
            std::exception_ptr first_error;
            std::vector<std::thread> readers;
            readers.reserve(worker_count);
            for (std::size_t worker = 0; worker < worker_count; ++worker) {
                readers.emplace_back([&] {
                    while (true) {
                        const auto index = next.fetch_add(1);
                        if (index >= cold_indices.size()) break;
                        try {
                            read_cold(index);
                        } catch (...) {
                            std::lock_guard<std::mutex> lock(error_mutex);
                            if (!first_error) {
                                first_error = std::current_exception();
                            }
                        }
                    }
                });
            }
            for (auto& reader : readers) reader.join();
            if (first_error) std::rethrow_exception(first_error);
        }

        std::vector<PagedPrefixPayload> result;
        result.reserve(requests.size());
        std::lock_guard<std::mutex> lock(mutex_);
        if (load_epoch != cache_epoch_) return result;
        for (std::size_t index = 0; index < requests.size(); ++index) {
            auto& request = requests[index];
            if (request.source == LoadSource::Disk) {
                if (!cold_payloads[index]) {
                    erase_corrupt_locked(request.hash);
                    break;
                }
                request.payload =
                    std::make_shared<const std::vector<std::uint8_t>>(
                        std::move(*cold_payloads[index]));
                put_hot_locked(request.hash, request.payload);
                const auto found = disk_.find(request.hash);
                if (found != disk_.end()) {
                    found->second.last_used = ++clock_;
                }
                ++metrics_.disk_hits;
            } else {
                if (request.source == LoadSource::Hot) {
                    const auto hot = hot_.find(request.hash);
                    if (hot != hot_.end()) {
                        hot->second.last_used = ++clock_;
                    }
                }
                ++metrics_.hot_hits;
            }
            result.push_back(std::move(request.payload));
        }
        return result;
    }

    void pin(const std::vector<BlockHash>& blocks) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& block : blocks) {
            ++pins_[block];
            auto found = disk_.find(block);
            if (found != disk_.end()) ++found->second.pins;
        }
    }

    void unpin(const std::vector<BlockHash>& blocks) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& block : blocks) {
            auto pinned = pins_.find(block);
            if (pinned != pins_.end()) {
                if (--pinned->second == 0) pins_.erase(pinned);
            }
            auto found = disk_.find(block);
            if (found != disk_.end() && found->second.pins > 0) {
                --found->second.pins;
            }
        }
        enforce_disk_budget_locked();
    }

    void flush() {
        std::unique_lock<std::mutex> lock(mutex_);
        writes_finished_.wait(lock, [this] {
            return writes_.empty() && active_writes_ == 0;
        });
    }

    std::size_t clear() {
        flush();
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t removed = 0;
        for (const auto& [hash, entry] : disk_) {
            (void)hash;
            std::error_code error;
            if (std::filesystem::remove(entry.path, error) && !error) ++removed;
        }
        disk_.clear();
        hot_.clear();
        pending_.clear();
        pins_.clear();
        disk_bytes_ = 0;
        hot_bytes_ = 0;
        pending_write_bytes_ = 0;
        ++cache_epoch_;
        sync_metrics_locked();
        return removed;
    }

    PagedPrefixCacheMetrics metrics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto result = metrics_;
        result.disk_blocks = disk_.size();
        result.disk_bytes = disk_bytes_;
        result.hot_blocks = hot_.size();
        result.hot_bytes = hot_bytes_;
        result.pending_writes = pending_.size();
        result.pending_bytes = pending_write_bytes_;
        result.pending_max_bytes = config_.max_pending_bytes;
        return result;
    }

private:
    std::filesystem::path path_for(const BlockHash& hash) const {
        const auto text = block_hash_hex(hash);
        return namespace_dir_ / text.substr(0, 2) / (text + ".mfqkv");
    }

    bool read_header(
        std::istream& input,
        ParsedHeader& header) const {
        std::array<std::uint8_t, 8> magic{};
        input.read(
            reinterpret_cast<char*>(magic.data()),
            static_cast<std::streamsize>(magic.size()));
        std::uint32_t version = 0;
        if (!input || magic != kMagic ||
            !read_little_endian(input, version) ||
            version != kFormatVersion) {
            return false;
        }
        input.read(
            reinterpret_cast<char*>(header.compatibility.data()), 32);
        input.read(reinterpret_cast<char*>(header.hash.data()), 32);
        input.read(reinterpret_cast<char*>(header.parent.data()), 32);
        if (!input || !read_little_endian(input, header.block_size) ||
            !read_little_endian(input, header.token_count) ||
            !read_little_endian(input, header.payload_bytes)) {
            return false;
        }
        input.read(reinterpret_cast<char*>(header.payload_hash.data()), 32);
        return input && header.compatibility == compatibility_hash_ &&
            header.block_size == config_.block_size_tokens &&
            header.token_count > 0 &&
            header.token_count <= header.block_size &&
            header.payload_bytes <=
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::streamsize>::max());
    }

    bool read_header(
        const std::filesystem::path& path,
        ParsedHeader& header) const {
        std::ifstream input(path, std::ios::binary);
        return input && read_header(input, header);
    }

    std::optional<std::vector<std::uint8_t>> read_payload(
        const std::filesystem::path& path,
        const BlockHash& expected_hash) const {
        std::ifstream input(path, std::ios::binary);
        if (!input) return std::nullopt;
        ParsedHeader header;
        if (!read_header(input, header) || header.hash != expected_hash) {
            return std::nullopt;
        }
        const auto payload_start = input.tellg();
        input.seekg(0, std::ios::end);
        const auto file_end = input.tellg();
        if (payload_start != static_cast<std::streamoff>(kHeaderBytes) ||
            file_end < 0 ||
            header.payload_bytes >
                std::numeric_limits<std::uint64_t>::max() - kHeaderBytes ||
            static_cast<std::uint64_t>(file_end) !=
                kHeaderBytes + header.payload_bytes) {
            return std::nullopt;
        }
        input.seekg(payload_start);
        if (!input) return std::nullopt;
        std::vector<std::uint8_t> payload(
            static_cast<std::size_t>(header.payload_bytes));
        input.read(
            reinterpret_cast<char*>(payload.data()),
            static_cast<std::streamsize>(payload.size()));
        if (!input || sha256(payload.data(), payload.size()) !=
                header.payload_hash) {
            return std::nullopt;
        }
        return payload;
    }

    bool write_file(const WriteRequest& request) {
        const auto final_path = path_for(request.hash);
        std::error_code error;
        std::filesystem::create_directories(final_path.parent_path(), error);
        if (error) return false;
        const auto temporary = final_path.parent_path() /
            (".mfqkv.tmp." + std::to_string(monotonic_tick()) + "." +
             std::to_string(temp_counter_.fetch_add(1)));
        std::ofstream output(
            temporary,
            std::ios::binary | std::ios::trunc);
        if (!output) return false;
        const auto payload_hash = sha256(
            request.payload->data(), request.payload->size());
        output.write(
            reinterpret_cast<const char*>(kMagic.data()),
            static_cast<std::streamsize>(kMagic.size()));
        write_little_endian(output, kFormatVersion);
        output.write(
            reinterpret_cast<const char*>(compatibility_hash_.data()), 32);
        output.write(reinterpret_cast<const char*>(request.hash.data()), 32);
        output.write(reinterpret_cast<const char*>(request.parent.data()), 32);
        write_little_endian(
            output, static_cast<std::uint32_t>(config_.block_size_tokens));
        write_little_endian(output, request.token_count);
        write_little_endian(
            output, static_cast<std::uint64_t>(request.payload->size()));
        output.write(reinterpret_cast<const char*>(payload_hash.data()), 32);
        output.write(
            reinterpret_cast<const char*>(request.payload->data()),
            static_cast<std::streamsize>(request.payload->size()));
        output.flush();
        output.close();
        if (!output) {
            std::filesystem::remove(temporary, error);
            return false;
        }
        if (!sync_regular_file(temporary)) {
            std::filesystem::remove(temporary, error);
            return false;
        }
        std::filesystem::permissions(
            temporary,
            std::filesystem::perms::owner_read |
                std::filesystem::perms::owner_write,
            std::filesystem::perm_options::replace,
            error);
        error.clear();
        std::filesystem::rename(temporary, final_path, error);
        if (error) {
            if (std::filesystem::exists(final_path)) {
                std::filesystem::remove(temporary, error);
                return true;
            }
            std::filesystem::remove(temporary, error);
            return false;
        }
        sync_directory_best_effort(final_path.parent_path());
        return true;
    }

    void finish_write(const WriteRequest& request, bool success) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto pending = pending_.find(request.hash);
        if (pending != pending_.end()) {
            const auto bytes = static_cast<std::uint64_t>(
                pending->second->size());
            pending_write_bytes_ = bytes > pending_write_bytes_
                ? 0
                : pending_write_bytes_ - bytes;
            pending_.erase(pending);
        }
        if (success) {
            const auto path = path_for(request.hash);
            std::error_code error;
            const auto file_bytes = std::filesystem::file_size(path, error);
            if (!error) {
                auto previous = disk_.find(request.hash);
                if (previous != disk_.end()) {
                    disk_bytes_ -= previous->second.file_bytes;
                }
                disk_[request.hash] = DiskEntry{
                    path,
                    request.parent,
                    request.token_count,
                    request.payload->size(),
                    file_bytes,
                    ++clock_,
                    pins_.count(request.hash) == 0
                        ? 0
                        : pins_.at(request.hash),
                };
                disk_bytes_ += file_bytes;
                ++metrics_.writes;
                enforce_disk_budget_locked();
            } else {
                ++metrics_.failed_writes;
            }
        } else {
            ++metrics_.failed_writes;
        }
        sync_metrics_locked();
        writes_finished_.notify_all();
    }

    void writer_loop() {
        while (true) {
            WriteRequest request;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_available_.wait(lock, [this] {
                    return stopping_ || !writes_.empty();
                });
                if (stopping_ && writes_.empty()) return;
                request = std::move(writes_.front());
                writes_.pop_front();
                ++active_writes_;
            }
            const bool success = write_file(request);
            finish_write(request, success);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --active_writes_;
                writes_finished_.notify_all();
            }
        }
    }

    void scan() {
        std::error_code error;
        for (std::filesystem::recursive_directory_iterator iterator(
                 namespace_dir_,
                 std::filesystem::directory_options::skip_permission_denied,
                 error), end;
             iterator != end && !error;
             iterator.increment(error)) {
            if (!iterator->is_regular_file(error)) {
                error.clear();
                continue;
            }
            const auto filename = iterator->path().filename().string();
            if (filename.find(".tmp.") != std::string::npos) {
                std::filesystem::remove(iterator->path(), error);
                error.clear();
                continue;
            }
            if (
                iterator->path().extension() != ".mfqkv") {
                continue;
            }
            ParsedHeader header;
            if (!read_header(iterator->path(), header)) continue;
            const auto file_bytes = iterator->file_size(error);
            if (error) {
                error.clear();
                continue;
            }
            const auto modified = iterator->last_write_time(error);
            const auto last_used = error
                ? ++clock_
                : static_cast<std::uint64_t>(modified.time_since_epoch().count());
            error.clear();
            disk_[header.hash] = DiskEntry{
                iterator->path(),
                header.parent,
                header.token_count,
                header.payload_bytes,
                file_bytes,
                last_used,
                0,
            };
            disk_bytes_ += file_bytes;
        }
        enforce_disk_budget_locked();
        sync_metrics_locked();
    }

    void put_hot_locked(
        const BlockHash& hash,
        std::shared_ptr<const std::vector<std::uint8_t>> payload) {
        if (config_.max_hot_bytes == 0 ||
            payload->size() > config_.max_hot_bytes) {
            return;
        }
        auto previous = hot_.find(hash);
        if (previous != hot_.end()) {
            hot_bytes_ -= previous->second.payload->size();
        }
        hot_[hash] = HotEntry{std::move(payload), ++clock_};
        hot_bytes_ += hot_[hash].payload->size();
        while (hot_bytes_ > config_.max_hot_bytes && !hot_.empty()) {
            auto victim = hot_.begin();
            for (auto iterator = std::next(hot_.begin());
                 iterator != hot_.end(); ++iterator) {
                if (iterator->second.last_used < victim->second.last_used) {
                    victim = iterator;
                }
            }
            hot_bytes_ -= victim->second.payload->size();
            hot_.erase(victim);
        }
        sync_metrics_locked();
    }

    void erase_corrupt_locked(const BlockHash& hash) {
        auto found = disk_.find(hash);
        if (found == disk_.end()) return;
        std::error_code error;
        std::filesystem::remove(found->second.path, error);
        disk_bytes_ -= found->second.file_bytes;
        disk_.erase(found);
        auto hot = hot_.find(hash);
        if (hot != hot_.end()) {
            hot_bytes_ -= hot->second.payload->size();
            hot_.erase(hot);
        }
        ++metrics_.corrupt_blocks;
        sync_metrics_locked();
    }

    void enforce_disk_budget_locked() {
        while (disk_bytes_ > config_.max_disk_bytes && !disk_.empty()) {
            auto victim = disk_.end();
            for (auto iterator = disk_.begin(); iterator != disk_.end();
                 ++iterator) {
                if (iterator->second.pins != 0 ||
                    pending_.count(iterator->first) != 0) {
                    continue;
                }
                if (victim == disk_.end() ||
                    iterator->second.last_used < victim->second.last_used) {
                    victim = iterator;
                }
            }
            if (victim == disk_.end()) break;
            std::error_code error;
            std::filesystem::remove(victim->second.path, error);
            if (error) break;
            disk_bytes_ -= victim->second.file_bytes;
            auto hot = hot_.find(victim->first);
            if (hot != hot_.end()) {
                hot_bytes_ -= hot->second.payload->size();
                hot_.erase(hot);
            }
            disk_.erase(victim);
            ++metrics_.evictions;
        }
        sync_metrics_locked();
    }

    void sync_metrics_locked() {
        metrics_.disk_blocks = disk_.size();
        metrics_.disk_bytes = disk_bytes_;
        metrics_.hot_blocks = hot_.size();
        metrics_.hot_bytes = hot_bytes_;
        metrics_.pending_writes = pending_.size();
        metrics_.pending_bytes = pending_write_bytes_;
        metrics_.pending_max_bytes = config_.max_pending_bytes;
    }

    PagedPrefixCacheConfig config_;
    BlockHash compatibility_hash_{};
    std::filesystem::path namespace_dir_;
    mutable std::mutex mutex_;
    std::condition_variable work_available_;
    std::condition_variable writes_finished_;
    std::thread worker_;
    std::deque<WriteRequest> writes_;
    std::unordered_map<
        BlockHash,
        std::shared_ptr<const std::vector<std::uint8_t>>,
        HashHasher> pending_;
    std::unordered_map<BlockHash, DiskEntry, HashHasher> disk_;
    std::unordered_map<BlockHash, HotEntry, HashHasher> hot_;
    std::unordered_map<BlockHash, std::size_t, HashHasher> pins_;
    std::uint64_t disk_bytes_ = 0;
    std::uint64_t hot_bytes_ = 0;
    std::uint64_t pending_write_bytes_ = 0;
    std::uint64_t clock_ = 0;
    std::uint64_t cache_epoch_ = 0;
    std::size_t active_writes_ = 0;
    bool stopping_ = false;
    PagedPrefixCacheMetrics metrics_;
    std::atomic<std::uint64_t> temp_counter_{0};
};

PagedPrefixCache::PagedPrefixCache(PagedPrefixCacheConfig config)
    : implementation_(
          std::make_unique<Implementation>(std::move(config))) {}

PagedPrefixCache::~PagedPrefixCache() = default;

std::size_t PagedPrefixCache::block_size_tokens() const noexcept {
    return implementation_->block_size_tokens();
}

const BlockHash& PagedPrefixCache::compatibility_hash() const noexcept {
    return implementation_->compatibility_hash();
}

BlockHash PagedPrefixCache::block_hash(
    const BlockHash& parent,
    const std::int64_t* token_ids,
    std::size_t token_count,
    std::string_view extra_key) const {
    return implementation_->block_hash(
        parent, token_ids, token_count, extra_key);
}

PrefixMatch PagedPrefixCache::match(
    const std::vector<std::int64_t>& token_ids,
    std::string_view extra_key,
    bool record_query) {
    return implementation_->match(token_ids, extra_key, record_query);
}

BlockHash PagedPrefixCache::store(
    const BlockHash& parent,
    const std::int64_t* token_ids,
    std::size_t token_count,
    std::shared_ptr<const std::vector<std::uint8_t>> payload,
    std::string_view extra_key) {
    return implementation_->store(
        parent,
        token_ids,
        token_count,
        std::move(payload),
        extra_key);
}

std::optional<std::vector<std::uint8_t>> PagedPrefixCache::load(
    const BlockHash& hash) {
    return implementation_->load(hash);
}

std::vector<PagedPrefixPayload> PagedPrefixCache::load_prefix(
    const std::vector<BlockHash>& blocks) {
    return implementation_->load_prefix(blocks);
}

void PagedPrefixCache::pin(const std::vector<BlockHash>& blocks) {
    implementation_->pin(blocks);
}

void PagedPrefixCache::unpin(const std::vector<BlockHash>& blocks) {
    implementation_->unpin(blocks);
}

void PagedPrefixCache::flush() {
    implementation_->flush();
}

std::size_t PagedPrefixCache::clear() {
    return implementation_->clear();
}

PagedPrefixCacheMetrics PagedPrefixCache::metrics() const {
    return implementation_->metrics();
}

} // namespace mfq::cache
