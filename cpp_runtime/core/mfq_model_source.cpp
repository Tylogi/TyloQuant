#include "mfq/mfq_model_source.h"
#include "mfq/async_range_reader.h"
#include "mfq/mapped_read_profile.h"
#include "mfq/mapped_prefetch_scope.h"
#include "mfq/mapped_tensor_pages.h"

#include "mfq_format_compat.h"
#include "mfq_legacy_tensor_names.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <system_error>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cerrno>
#endif

namespace mfq {
namespace {

constexpr std::uint64_t kMaxStringBytes = std::uint64_t{64} << 20;
constexpr std::uint32_t kMaxMetadataEntries = std::uint32_t{1} << 16;
constexpr std::uint32_t kMaxRecordEntries = std::uint32_t{1} << 20;
constexpr std::string_view kAssetPrefix = "__mfq_asset__/";
std::array<std::atomic<std::uint64_t>,5> mapped_read_totals{};
using MappedClock=std::chrono::steady_clock;
thread_local bool mapped_prefetch_context=false;
thread_local bool mapped_release_context=false;

class MappedTensorRange {
    void* view_=nullptr;
    std::size_t mapped_bytes_=0,prefix_=0,page_bytes_=0;
    std::uint64_t bytes_=0;
    bool profile_=false;
    int prefetch_override_=-1;
#ifdef _WIN32
    HANDLE file_=INVALID_HANDLE_VALUE,mapping_=nullptr;
#else
    int file_=-1;
#endif
    void close() noexcept {
#ifdef _WIN32
        if(view_)UnmapViewOfFile(view_);
        if(mapping_)CloseHandle(mapping_);
        if(file_!=INVALID_HANDLE_VALUE)CloseHandle(file_);
#else
        if(view_)::munmap(view_,mapped_bytes_);
        if(file_>=0)::close(file_);
#endif
    }
public:
    explicit MappedTensorRange(const MfqStoredRecord& record):bytes_(record.tensor.nbytes) {
        const auto* profile=std::getenv("MFQ_PROFILE_SSD_EXPERTS");
        profile_=profile && profile[0]=='1';
        const auto* prefetch=std::getenv("MFQ_MAPPED_BATCH_PREFETCH");
        if(prefetch)prefetch_override_=prefetch[0]!='0';
        if(!bytes_)return;
        try {
            std::uint64_t granularity=0,file_bytes=0;
#ifdef _WIN32
            file_=CreateFileW(record.source_path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_RANDOM_ACCESS,nullptr);
            if(file_==INVALID_HANDLE_VALUE)throw std::system_error(GetLastError(),std::system_category(),"open mapped tensor");
            LARGE_INTEGER size{};
            if(!GetFileSizeEx(file_,&size) || size.QuadPart<0)throw std::runtime_error("mapped tensor file size unavailable");
            file_bytes=static_cast<std::uint64_t>(size.QuadPart);
            SYSTEM_INFO info{};GetSystemInfo(&info);granularity=info.dwAllocationGranularity;
            page_bytes_=info.dwPageSize;
#else
            file_=::open(record.source_path.c_str(),O_RDONLY);
            if(file_<0)throw std::system_error(errno,std::generic_category(),"open mapped tensor");
            struct stat size{};
            if(::fstat(file_,&size) || size.st_size<0)throw std::runtime_error("mapped tensor file size unavailable");
            file_bytes=static_cast<std::uint64_t>(size.st_size);
            const auto page=::sysconf(_SC_PAGESIZE);
            if(page<=0)throw std::runtime_error("mapped tensor page size unavailable");
            page_bytes_=static_cast<std::size_t>(page);
            granularity=static_cast<std::uint64_t>(page);
#endif
            if(record.offset>file_bytes || bytes_>file_bytes-record.offset)throw std::out_of_range("mapped tensor exceeds file");
            const auto base=record.offset/granularity*granularity;
            prefix_=static_cast<std::size_t>(record.offset-base);
            if(bytes_>std::numeric_limits<std::size_t>::max()-prefix_)throw std::overflow_error("mapped tensor exceeds address space");
            mapped_bytes_=prefix_+static_cast<std::size_t>(bytes_);
#ifdef _WIN32
            mapping_=CreateFileMappingW(file_,nullptr,PAGE_READONLY,0,0,nullptr);
            if(!mapping_)throw std::system_error(GetLastError(),std::system_category(),"create tensor mapping");
            view_=MapViewOfFile(mapping_,FILE_MAP_READ,static_cast<DWORD>(base>>32),static_cast<DWORD>(base),mapped_bytes_);
            if(!view_)throw std::system_error(GetLastError(),std::system_category(),"map tensor view");
#else
            if(base>static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))throw std::overflow_error("mapped tensor offset exceeds platform range");
            view_=::mmap(nullptr,mapped_bytes_,PROT_READ,MAP_SHARED,file_,static_cast<off_t>(base));
            if(view_==MAP_FAILED){view_=nullptr;throw std::system_error(errno,std::generic_category(),"map tensor view");}
#endif
        }catch(...){close();throw;}
    }
    ~MappedTensorRange(){close();}
    MappedTensorRange(const MappedTensorRange&)=delete;
    MappedTensorRange& operator=(const MappedTensorRange&)=delete;
    bool release_pages() const noexcept {
        if(!view_ || !mapped_bytes_)return false;
#ifdef _WIN32
        // Unlocked file pages become reclaimable even when the API reports
        // ERROR_NOT_LOCKED. The read-only view and its file stay open.
        (void)VirtualUnlock(view_,mapped_bytes_);
#else
        (void)::madvise(view_,mapped_bytes_,MADV_DONTNEED);
#endif
        return true;
    }
    void release_read_pages(const std::vector<ReadSpan>& spans) const {
        if(spans.empty())return;
        // Large prefill batches benefit from one view-wide trim. For sparse
        // decode rows, walking a multi-GiB view costs more than the reads:
        // release just the touched pages, merging duplicate/adjacent ranges.
        if(spans.size()>1024 || !page_bytes_) {release_pages();return;}
        std::vector<std::pair<std::uintptr_t,std::uintptr_t>> pages;
        pages.reserve(spans.size());
        for(const auto& span:spans)if(span.size) {
            const auto address=reinterpret_cast<std::uintptr_t>(view_)+prefix_+std::size_t(span.offset);
            const auto begin=address/page_bytes_*page_bytes_;
            const auto end=(address+span.size+page_bytes_-1)/page_bytes_*page_bytes_;
            pages.emplace_back(begin,end);
        }
        if(pages.empty())return;
        std::sort(pages.begin(),pages.end());
        const auto release=[](std::uintptr_t begin,std::uintptr_t end) {
#ifdef _WIN32
            (void)VirtualUnlock(reinterpret_cast<void*>(begin),end-begin);
#else
            (void)::madvise(reinterpret_cast<void*>(begin),end-begin,MADV_DONTNEED);
#endif
        };
        auto begin=pages.front().first,end=pages.front().second;
        for(std::size_t i=1;i<pages.size();++i) {
            if(pages[i].first<=end)end=std::max(end,pages[i].second);
            else {release(begin,end);begin=pages[i].first;end=pages[i].second;}
        }
        release(begin,end);
    }
    void read(std::uint64_t offset,std::byte* destination,std::size_t size)const {
        if(offset>bytes_ || size>bytes_-offset)throw std::out_of_range("mapped tensor row range is out of bounds");
        if(!size)return;
        if(!destination)throw std::invalid_argument("mapped tensor destination is null");
        const auto started=profile_?MappedClock::now():MappedClock::time_point{};
        std::memcpy(destination,static_cast<const std::byte*>(view_)+prefix_+static_cast<std::size_t>(offset),size);
        if(profile_) {
            mapped_read_totals[1].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(MappedClock::now()-started).count(),std::memory_order_relaxed);
            mapped_read_totals[2].fetch_add(size,std::memory_order_relaxed);
            mapped_read_totals[3].fetch_add(1,std::memory_order_relaxed);
        }
    }
    void read_batch(const std::vector<ReadSpan>& spans)const {
        // Check the complete batch before touching any destination. A failed
        // hint is harmless: the following bounded read faults in that page.
        for(const auto& span:spans) {
            if(span.offset>bytes_ || span.size>bytes_-span.offset)
                throw std::out_of_range("mapped tensor batch is out of bounds");
            if(span.size && !span.destination)
                throw std::invalid_argument("mapped tensor batch destination is null");
        }
        // Like Strata's PLE prefetch, submit the independent pages together.
        // NINT stores q, sub-scale/min and neuron anchors in separate streams;
        // sequential mapped reads otherwise fault those five streams in turn.
        const auto started=profile_?MappedClock::now():MappedClock::time_point{};
        if(prefetch_override_<0?mapped_prefetch_context:prefetch_override_!=0) {
#ifdef _WIN32
        std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
        ranges.reserve(std::min<std::size_t>(spans.size(),4096));
        const auto prefetch=[&] {
            if(!ranges.empty())(void)PrefetchVirtualMemory(GetCurrentProcess(),ranges.size(),ranges.data(),0);
            ranges.clear();
        };
        for(const auto& span:spans)if(span.size) {
            ranges.push_back({static_cast<std::byte*>(view_)+prefix_+static_cast<std::size_t>(span.offset),span.size});
            if(ranges.size()==4096)prefetch();
        }
        prefetch();
#else
        const auto page=std::size_t(::sysconf(_SC_PAGESIZE));
        if(page && page!=std::size_t(-1))for(const auto& span:spans)if(span.size) {
            const auto address=reinterpret_cast<std::uintptr_t>(view_)+prefix_+std::size_t(span.offset);
            const auto begin=address/page*page,end=(address+span.size+page-1)/page*page;
            (void)::madvise(reinterpret_cast<void*>(begin),end-begin,MADV_WILLNEED);
        }
#endif
        }
        if(profile_) {
            mapped_read_totals[0].fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(MappedClock::now()-started).count(),std::memory_order_relaxed);
            mapped_read_totals[4].fetch_add(1,std::memory_order_relaxed);
        }
        for(const auto& span:spans)if(span.size)read(span.offset,span.destination,span.size);
        // Selected rows now belong to the bounded row cache/packet. Keeping
        // a second working-set copy of sparse pages across all Engram tables
        // can exhaust RAM despite a bounded expert cache.
        if(mapped_release_context)release_read_pages(spans);
    }
};

std::mutex mapped_ranges_mutex;
std::vector<std::weak_ptr<MappedTensorRange>> mapped_ranges;
std::shared_ptr<MappedTensorRange> map_tensor_range(const MfqStoredRecord& record) {
    auto range=std::make_shared<MappedTensorRange>(record);
    std::lock_guard lock(mapped_ranges_mutex);mapped_ranges.push_back(range);
    return range;
}

std::filesystem::path stable_path(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error) {
        throw std::runtime_error("cannot open MFQ file: " + path.string());
    }
    auto result = std::filesystem::weakly_canonical(path, error);
    if (error) {
        error.clear();
        result = std::filesystem::absolute(path, error).lexically_normal();
    }
    if (error || !result.is_absolute()) {
        throw std::runtime_error("cannot resolve MFQ file path: " + path.string());
    }
    return result;
}

std::uint64_t usable_file_size(const std::filesystem::path& path) {
    std::error_code error;
    const auto value = std::filesystem::file_size(path, error);
    if (error || value > std::numeric_limits<std::uint64_t>::max() ||
        value > static_cast<std::uintmax_t>(
                    std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("cannot determine usable MFQ file size: " +
                                 path.string());
    }
    return static_cast<std::uint64_t>(value);
}

class Input {
public:
    Input(std::istream& stream, std::uint64_t size, const std::filesystem::path& path)
        : stream_(stream), size_(size), path_(path) {}

    std::uint64_t position() const noexcept { return position_; }
    std::uint64_t remaining() const noexcept { return size_ - position_; }

    template <typename T>
    T scalar(const char* what) {
        T value{};
        read(reinterpret_cast<char*>(&value), sizeof(value), what);
        return value;
    }

    std::string string(const char* what) {
        const auto size = scalar<std::uint32_t>("string length");
        if (size > kMaxStringBytes) {
            throw std::runtime_error(std::string("MFQ ") + what +
                                     " exceeds the supported string length: " +
                                     path_.string());
        }
        std::string result(size, '\0');
        if (size != 0) read(result.data(), size, what);
        return result;
    }

private:
    void read(char* destination, std::uint64_t size, const char* what) {
        if (size > remaining() ||
            size > static_cast<std::uint64_t>(
                       std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error(std::string("unexpected EOF reading ") +
                                     what + ": " + path_.string());
        }
        stream_.read(destination, static_cast<std::streamsize>(size));
        if (!stream_) {
            throw std::runtime_error(std::string("failed reading MFQ ") + what +
                                     ": " + path_.string());
        }
        position_ += size;
    }

    std::istream& stream_;
    std::uint64_t size_;
    const std::filesystem::path& path_;
    std::uint64_t position_ = 0;
};

void validate_count(
    std::uint32_t count,
    std::uint32_t limit,
    std::uint64_t minimum_bytes,
    std::uint64_t remaining,
    const char* what,
    const std::filesystem::path& path) {
    if (count > limit || count > remaining / minimum_bytes) {
        throw std::runtime_error(std::string("invalid MFQ ") + what + ": " +
                                 path.string());
    }
}

std::uint64_t metadata_uint(
    const MfqSourceHeader& header,
    const std::string& key,
    std::uint64_t fallback) {
    const auto found = header.metadata.find(key);
    if (found == header.metadata.end()) return fallback;
    std::size_t parsed = 0;
    std::uint64_t result = 0;
    try {
        result = std::stoull(found->second, &parsed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid MFQ integer metadata " + key + ": " +
                                 found->second);
    }
    while (parsed < found->second.size() &&
           std::isspace(static_cast<unsigned char>(found->second[parsed]))) {
        ++parsed;
    }
    if (parsed != found->second.size()) {
        throw std::runtime_error("invalid MFQ integer metadata " + key + ": " +
                                 found->second);
    }
    return result;
}

std::vector<std::filesystem::path> shard_paths(
    const std::filesystem::path& path,
    std::uint64_t split_no,
    std::uint64_t split_count) {
    static const std::regex pattern(
        R"(^(.*)-([0-9]{5})-of-([0-9]{5})(\.mfq|\.tq)$)", std::regex::icase);
    std::smatch match;
    const auto filename = path.filename().string();
    if (!std::regex_match(filename, match, pattern)) {
        throw std::runtime_error(
            "sharded MFQ path lacks -00001-of-00000 suffix: " + path.string());
    }
    if (std::stoull(match[2].str()) != split_no + 1 ||
        std::stoull(match[3].str()) != split_count) {
        throw std::runtime_error("MFQ shard filename/metadata mismatch: " +
                                 path.string());
    }
    auto suffix = match[4].str();
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) { return std::tolower(c); });
    const auto same_suffix = [&](const std::ssub_match& candidate) {
        auto value = candidate.str();
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
        return value == suffix;
    };
    std::vector<std::filesystem::path> result(static_cast<std::size_t>(split_count));
    for (std::uint64_t index = 1; index <= split_count; ++index) {
        std::ostringstream name;
        name << match[1].str() << '-' << std::setfill('0') << std::setw(5)
             << index << "-of-" << std::setw(5) << split_count << suffix;
        result[index - 1] = path.parent_path() / name.str();
    }
    if (std::all_of(result.begin(), result.end(),
            [](const auto& candidate) { return std::filesystem::is_regular_file(candidate); })) {
        for (auto& candidate : result) candidate = stable_path(candidate);
        return result;
    }
    std::vector<bool> found(result.size(), false);
    for (const auto& entry : std::filesystem::directory_iterator(path.parent_path())) {
        std::smatch candidate;
        const auto candidate_name = entry.path().filename().string();
        if (!entry.is_regular_file() || !std::regex_match(candidate_name, candidate, pattern) ||
            candidate[1].str() != match[1].str() || !same_suffix(candidate[4]) ||
            std::stoull(candidate[3].str()) != split_count) {
            continue;
        }
        const auto index = std::stoull(candidate[2].str());
        if (index < 1 || index > split_count) continue;
        if (found[index - 1]) {
            throw std::runtime_error("duplicate MFQ shard index " + std::to_string(index) + ": " + path.string());
        }
        result[index - 1] = entry.path();
        found[index - 1] = true;
    }
    // Shared Engram shards keep their original indices, but live once beside
    // the precision directories. Never substitute another model or split set.
    static const std::regex engram_pattern(R"(^(.*)-Engram-.+$)", std::regex::icase);
    std::unordered_map<std::uint64_t, std::filesystem::path> shared;
    std::vector<std::filesystem::path> directories{
        path.parent_path(), path.parent_path() / "Engram",
        path.parent_path().parent_path() / "Engram"};
    std::sort(directories.begin(), directories.end());
    directories.erase(std::unique(directories.begin(), directories.end()), directories.end());
    for (const auto& directory : directories) {
        if (!std::filesystem::is_directory(directory)) continue;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            std::smatch candidate, family;
            const auto name = entry.path().filename().string();
            if (!entry.is_regular_file() || !std::regex_match(name, candidate, pattern)) continue;
            const auto stem = candidate[1].str();
            if (!std::regex_match(stem, family, engram_pattern) ||
                match[1].str().compare(0, family[1].length() + 1, family[1].str() + "-") != 0 ||
                std::stoull(candidate[3].str()) != split_count || !same_suffix(candidate[4])) continue;
            const auto index = std::stoull(candidate[2].str());
            if (index < 1 || index > split_count || found[index - 1]) continue;
            if (!shared.emplace(index, entry.path()).second)
                throw std::runtime_error("ambiguous shared Engram shard " + std::to_string(index));
        }
    }
    for (const auto& [index, candidate] : shared) result[index - 1] = candidate;
    for (std::uint64_t index = 1; index <= split_count; ++index) {
        result[index - 1] = stable_path(result[index - 1]);
    }
    return result;
}

struct ParsedFile {
    MfqSourceHeader header;
    std::vector<MfqStoredRecord> records;
};

ParsedFile parse_file(const std::filesystem::path& requested) {
    const auto path = stable_path(requested);
    const auto size = usable_file_size(path);
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open MFQ file: " + path.string());
    Input input(stream, size, path);

    char magic[4]{};
    for (char& value : magic) value = input.scalar<char>("MFQ magic");
    if (std::memcmp(magic, "MFQ1", sizeof(magic)) != 0) {
        throw std::runtime_error("bad MFQ magic: " + path.string());
    }

    ParsedFile result;
    result.header.version = input.scalar<std::uint32_t>("MFQ version");
    if (result.header.version == 0 || result.header.version > 2) {
        throw std::runtime_error("unsupported MFQ version " +
                                 std::to_string(result.header.version) + ": " +
                                 path.string());
    }
    result.header.architecture = input.string("architecture");
    if (result.header.version >= 2) {
        const auto count = input.scalar<std::uint32_t>("metadata count");
        validate_count(count, kMaxMetadataEntries, 2 * sizeof(std::uint32_t),
                       input.remaining(), "metadata count", path);
        for (std::uint32_t index = 0; index < count; ++index) {
            auto key = input.string("metadata key");
            auto value = input.string("metadata value");
            if (!result.header.metadata.emplace(std::move(key), std::move(value)).second) {
                throw std::runtime_error("duplicate MFQ metadata key: " + path.string());
            }
        }
    }

    result.header.record_count = input.scalar<std::uint32_t>("record count");
    validate_count(result.header.record_count, kMaxRecordEntries,
                   2 * sizeof(std::uint32_t) + sizeof(std::uint64_t),
                   input.remaining(), "record count", path);
    result.records.reserve(result.header.record_count);
    for (std::uint32_t index = 0; index < result.header.record_count; ++index) {
        MfqStoredRecord record;
        record.tensor.name = input.string("record name");
        record.tensor.stored_dtype = input.string("record dtype");
        record.tensor.dtype = std::string(
            canonical_format_dtype(record.tensor.stored_dtype));
        record.tensor.nbytes = input.scalar<std::uint64_t>("record length");
        constexpr std::string_view legacy_asset_prefix = "__tinfer_asset__/";
        if (record.tensor.name.compare(0, legacy_asset_prefix.size(), legacy_asset_prefix) == 0)
            record.tensor.name = std::string(kAssetPrefix) + record.tensor.name.substr(legacy_asset_prefix.size());
        record.asset = record.tensor.name.compare(
                           0, kAssetPrefix.size(), kAssetPrefix) == 0;
        record.source_path = path;
        result.records.push_back(std::move(record));
    }
    auto offset = input.position();
    for (auto& record : result.records) {
        record.offset = offset;
        if (record.tensor.nbytes > size - offset) {
            throw std::runtime_error(
                "MFQ file length does not match record table: " + path.string());
        }
        offset += record.tensor.nbytes;
    }
    if (offset != size) {
        throw std::runtime_error(
            "MFQ file length does not match record table: " + path.string());
    }
    return result;
}

// MSVC's filebuf can split a large xsgetn into thousands of small reads.
// fread sends large contiguous records directly to the buffered file handle.
class BufferedRecordInput {
    std::FILE* file_=nullptr;
public:
    explicit BufferedRecordInput(const std::filesystem::path& path) {
#ifdef _WIN32
        file_=_wfopen(path.c_str(),L"rb");
#else
        file_=std::fopen(path.c_str(),"rb");
#endif
        if(!file_)throw std::runtime_error("cannot open MFQ row source: "+path.string());
    }
    ~BufferedRecordInput(){if(file_)std::fclose(file_);}
    BufferedRecordInput(const BufferedRecordInput&)=delete;
    BufferedRecordInput& operator=(const BufferedRecordInput&)=delete;
    void read(std::uint64_t offset,std::byte* destination,std::size_t size) {
        if(offset>std::uint64_t(std::numeric_limits<std::int64_t>::max()) ||
                size>std::uint64_t(std::numeric_limits<std::int64_t>::max()))
            throw std::overflow_error("model tensor row range overflow");
        if(!size)return;
        std::clearerr(file_);
#ifdef _WIN32
        const auto seek=_fseeki64(file_,static_cast<std::int64_t>(offset),SEEK_SET);
#else
        const auto seek=fseeko(file_,static_cast<off_t>(offset),SEEK_SET);
#endif
        if(seek || std::fread(destination,1,size,file_)!=size)
            throw std::runtime_error("MFQ row source was truncated");
    }
};

void read_exact(
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::byte* destination,
    std::size_t size) {
    if (size == 0) return;
    if (offset > static_cast<std::uint64_t>(
                     std::numeric_limits<std::streamoff>::max()) ||
        size > static_cast<std::size_t>(
                   std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("model source byte range is too large");
    }
    BufferedRecordInput(path).read(offset,destination,size);
}

} // namespace

struct MfqModelSource::Impl {
    FileReadMode read_mode = FileReadMode::Buffered;
    std::unordered_map<std::filesystem::path, std::shared_ptr<FileRangeReader>> direct_readers;
    mutable std::mutex pipeline_reader_mutex;
    mutable std::unordered_map<std::filesystem::path,std::shared_ptr<AsyncRangeReader>> pipeline_readers;
    MfqSourceHeader header;
    std::vector<std::filesystem::path> source_paths;
    std::vector<MfqStoredRecord> records;
    std::vector<TensorMetadata> tensors;
    std::vector<std::string> assets;
    MfqLegacyTensorAliases legacy_tensor_compatibility;
    std::unordered_map<std::string, std::size_t> records_by_name;
    std::unordered_map<std::string, std::size_t> tensors_by_name;
};

std::array<std::uint64_t,5> mapped_tensor_read_profile() {
    std::array<std::uint64_t,5> result{};
    for(std::size_t i=0;i<result.size();++i)result[i]=mapped_read_totals[i].load(std::memory_order_relaxed);
    return result;
}

std::size_t release_mapped_tensor_pages() {
    std::lock_guard lock(mapped_ranges_mutex);
    std::size_t views=0;
    for(auto item=mapped_ranges.begin();item!=mapped_ranges.end();) {
        if(const auto range=item->lock()) {views+=range->release_pages();++item;}
        else item=mapped_ranges.erase(item);
    }
    return views;
}

MappedPrefetchScope::MappedPrefetchScope(bool enabled,bool release_after_read)
        :previous_(mapped_prefetch_context),previous_release_(mapped_release_context) {
    mapped_prefetch_context=enabled;
    mapped_release_context=release_after_read;
}
MappedPrefetchScope::~MappedPrefetchScope(){
    mapped_prefetch_context=previous_;mapped_release_context=previous_release_;
}

MfqModelSource::MfqModelSource(std::filesystem::path requested, FileReadMode mode)
    : impl_(std::make_unique<Impl>()) {
    impl_->read_mode = mode;
    const auto initial_path = stable_path(requested);
    auto initial = parse_file(initial_path);
    const auto split_no = metadata_uint(initial.header, "split.no", 0);
    const auto split_count = metadata_uint(initial.header, "split.count", 1);
    if (split_count == 0 || split_count > 99999 || split_no >= split_count) {
        throw std::runtime_error("invalid MFQ split metadata: " +
                                 initial_path.string());
    }

    auto append = [&](ParsedFile parsed) {
        for (auto& record : parsed.records) {
            if (!impl_->records_by_name.emplace(
                    record.tensor.name, impl_->records.size()).second) {
                throw std::runtime_error("duplicate MFQ tensor record: " +
                                         record.source_path.string());
            }
            impl_->records.push_back(std::move(record));
        }
    };

    if (split_count == 1) {
        impl_->header = initial.header;
        impl_->source_paths.push_back(initial_path);
        append(std::move(initial));
    } else {
        // A numbered entry may be a symlink to an unnumbered downloaded file.
        // Discover siblings from the requested name before resolving each file.
        const auto paths = shard_paths(std::filesystem::absolute(requested).lexically_normal(), split_no, split_count);
        std::uint64_t actual_records = 0;
        std::uint64_t actual_tensors = 0;
        auto expected_records = std::numeric_limits<std::uint64_t>::max();
        auto expected_tensors = std::numeric_limits<std::uint64_t>::max();
        for (std::uint64_t index = 0; index < split_count; ++index) {
            auto current = parse_file(paths[static_cast<std::size_t>(index)]);
            static const std::regex shared_name(R"(^.*-Engram-.+-[0-9]{5}-of-[0-9]{5}\.(mfq|tq)$)", std::regex::icase);
            if (std::regex_match(paths[index].filename().string(), shared_name)) {
                const auto table = [](const MfqStoredRecord& record) {
                    std::string_view name = record.tensor.name;
                    constexpr std::string_view marker = ".tq_row_shard.";
                    const auto shard = name.rfind(marker);
                    if (shard != std::string_view::npos && shard + marker.size() < name.size()
                        && std::all_of(name.begin() + shard + marker.size(), name.end(), [](char c) { return c >= '0' && c <= '9'; }))
                        name = name.substr(0, shard);
                    for (const std::string_view suffix : {".associative_memory.embedding.weight", ".engram.embed.weight"})
                        if (name.size() >= suffix.size() &&
                            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) return true;
                    return false;
                };
                if (current.records.empty() || !std::all_of(current.records.begin(), current.records.end(), table))
                    throw std::runtime_error("shared Engram shard contains non-table records: " + paths[index].string());
            }
            if (current.header.version != initial.header.version ||
                current.header.architecture != initial.header.architecture ||
                metadata_uint(current.header, "split.no", split_count) != index ||
                metadata_uint(current.header, "split.count", 0) != split_count) {
                throw std::runtime_error("MFQ shard metadata mismatch: " +
                                         paths[static_cast<std::size_t>(index)].string());
            }
            if (index == 0) impl_->header = current.header;
            const auto records = metadata_uint(
                current.header, "split.records.count",
                std::numeric_limits<std::uint64_t>::max());
            const auto tensors = metadata_uint(
                current.header, "split.tensors.count",
                std::numeric_limits<std::uint64_t>::max());
            if (expected_records == std::numeric_limits<std::uint64_t>::max()) {
                expected_records = records;
            } else if (records != std::numeric_limits<std::uint64_t>::max() &&
                       records != expected_records) {
                throw std::runtime_error("MFQ shard record count mismatch");
            }
            if (expected_tensors == std::numeric_limits<std::uint64_t>::max()) {
                expected_tensors = tensors;
            } else if (tensors != std::numeric_limits<std::uint64_t>::max() &&
                       tensors != expected_tensors) {
                throw std::runtime_error("MFQ shard tensor count mismatch");
            }
            actual_records += current.header.record_count;
            for (const auto& record : current.records) {
                if (!record.asset) ++actual_tensors;
            }
            append(std::move(current));
        }
        if (expected_records != std::numeric_limits<std::uint64_t>::max() &&
            actual_records != expected_records) {
            throw std::runtime_error("MFQ shard record total mismatch");
        }
        if (expected_tensors != std::numeric_limits<std::uint64_t>::max() &&
            actual_tensors != expected_tensors) {
            throw std::runtime_error("MFQ shard tensor total mismatch");
        }
        impl_->source_paths = paths;
    }

    if (mode == FileReadMode::Direct) {
        for (const auto& path : impl_->source_paths)
            impl_->direct_readers.emplace(path, std::make_shared<FileRangeReader>(path, mode));
    }
    std::vector<std::string> stored_names;
    for (const auto& record : impl_->records) {
        if (record.asset) {
            impl_->assets.push_back(record.tensor.name);
        } else {
            stored_names.push_back(record.tensor.name);
        }
    }
    std::sort(impl_->assets.begin(), impl_->assets.end());

    std::unordered_map<std::string, std::string> stored_to_canonical;
    if (!has_asset(kModelGraphAsset) && has_asset(kModelConfigAsset)) {
        const auto config = read_asset(kModelConfigAsset);
        impl_->legacy_tensor_compatibility = make_legacy_tensor_aliases(
            architecture(),
            std::string_view(
                reinterpret_cast<const char*>(config.data()), config.size()),
            stored_names);
        for (const auto& [canonical, stored] :
             impl_->legacy_tensor_compatibility.canonical_to_stored) {
            const auto record = impl_->records_by_name.find(stored);
            if (record == impl_->records_by_name.end() ||
                    impl_->records[record->second].asset ||
                    impl_->records_by_name.find(canonical) !=
                        impl_->records_by_name.end()) {
                throw std::runtime_error(
                    "invalid legacy MFQ tensor alias: " + canonical);
            }
            const auto [found, inserted] =
                stored_to_canonical.emplace(stored, canonical);
            if (!inserted && found->second != canonical) {
                throw std::runtime_error(
                    "legacy MFQ tensor has multiple canonical names: " +
                    stored);
            }
        }
    }

    for (const auto& record : impl_->records) {
        if (record.asset) continue;
        auto tensor = record.tensor;
        if (const auto found = stored_to_canonical.find(tensor.name);
            found != stored_to_canonical.end()) {
            tensor.name = found->second;
        }
        if (!impl_->tensors_by_name.emplace(
                tensor.name, impl_->tensors.size()).second) {
            throw std::runtime_error(
                "legacy MFQ tensor aliases collide at canonical name: " +
                tensor.name);
        }
        impl_->tensors.push_back(std::move(tensor));
    }
}

MfqModelSource::~MfqModelSource() = default;
MfqModelSource::MfqModelSource(MfqModelSource&&) noexcept = default;
MfqModelSource& MfqModelSource::operator=(MfqModelSource&&) noexcept = default;

const MfqSourceHeader& MfqModelSource::header() const noexcept { return impl_->header; }
const std::vector<std::filesystem::path>& MfqModelSource::source_paths() const noexcept {
    return impl_->source_paths;
}
const std::vector<MfqStoredRecord>& MfqModelSource::records() const noexcept {
    return impl_->records;
}
std::string_view MfqModelSource::architecture() const noexcept {
    return impl_->header.architecture;
}
const std::unordered_map<std::string, std::string>&
MfqModelSource::metadata() const noexcept {
    return impl_->header.metadata;
}
const std::vector<TensorMetadata>& MfqModelSource::tensors() const noexcept {
    return impl_->tensors;
}

const TensorMetadata* MfqModelSource::find_tensor(std::string_view name) const noexcept {
    const auto found = impl_->tensors_by_name.find(std::string(name));
    return found == impl_->tensors_by_name.end()
        ? nullptr
        : &impl_->tensors[found->second];
}

const MfqLegacyTensorAliases&
MfqModelSource::legacy_tensor_compatibility() const noexcept {
    return impl_->legacy_tensor_compatibility;
}

void MfqModelSource::read_range_into(
    std::string_view name,
    std::uint64_t relative_offset,
    std::byte* destination,
    std::size_t size) const {
    auto found = impl_->records_by_name.find(std::string(name));
    if (found == impl_->records_by_name.end()) {
        const auto alias = impl_->legacy_tensor_compatibility
            .canonical_to_stored.find(std::string(name));
        if (alias != impl_->legacy_tensor_compatibility
                .canonical_to_stored.end()) {
            found = impl_->records_by_name.find(alias->second);
        }
    }
    if (found == impl_->records_by_name.end() ||
            impl_->records[found->second].asset) {
        throw std::runtime_error("model tensor not found: " + std::string(name));
    }
    const auto& record = impl_->records[found->second];
    if (relative_offset > record.tensor.nbytes ||
        size > record.tensor.nbytes - relative_offset) {
        throw std::out_of_range("model tensor byte range is out of bounds: " +
                                std::string(name));
    }
    if (relative_offset > std::numeric_limits<std::uint64_t>::max() - record.offset) {
        throw std::overflow_error("model tensor byte offset overflow");
    }
    if (impl_->read_mode == FileReadMode::Direct)
        impl_->direct_readers.at(record.source_path)->read(record.offset + relative_offset, destination, size);
    else read_exact(record.source_path, record.offset + relative_offset, destination, size);
}

ModelSource::TensorReader MfqModelSource::tensor_reader(std::string_view name) const {
    auto found = impl_->records_by_name.find(std::string(name));
    if (found == impl_->records_by_name.end()) {
        const auto alias = impl_->legacy_tensor_compatibility.canonical_to_stored.find(std::string(name));
        if (alias != impl_->legacy_tensor_compatibility.canonical_to_stored.end())
            found = impl_->records_by_name.find(alias->second);
    }
    if (found == impl_->records_by_name.end() || impl_->records[found->second].asset)
        throw std::runtime_error("model tensor not found: " + std::string(name));
    if (impl_->read_mode == FileReadMode::Direct) {
        const auto record = impl_->records[found->second];
        auto reader = impl_->direct_readers.at(record.source_path);
        return [record, reader](std::uint64_t offset, std::byte* destination, std::size_t size) {
            if (offset > record.tensor.nbytes || size > record.tensor.nbytes - offset)
                throw std::out_of_range("model tensor row range is out of bounds");
            reader->read(record.offset + offset, destination, size);
        };
    }
    struct Range {
        MfqStoredRecord record;
        BufferedRecordInput input;
        std::mutex mutex;
        explicit Range(MfqStoredRecord value)
            : record(std::move(value)), input(record.source_path) {}
    };
    auto range = std::make_shared<Range>(impl_->records[found->second]);
    return [range](std::uint64_t offset, std::byte* destination, std::size_t size) {
        const auto& record = range->record;
        if (offset > record.tensor.nbytes || size > record.tensor.nbytes - offset)
            throw std::out_of_range("model tensor row range is out of bounds");
        if (offset > std::numeric_limits<std::uint64_t>::max() - record.offset ||
            record.offset + offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
            size > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
            throw std::overflow_error("model tensor row range overflow");
        if (!size) return;
        std::lock_guard<std::mutex> lock(range->mutex);
        range->input.read(record.offset+offset,destination,size);
    };
}

ModelSource::TensorReader MfqModelSource::mapped_tensor_reader(std::string_view name) const {
    auto found=impl_->records_by_name.find(std::string(name));
    if(found==impl_->records_by_name.end()) {
        const auto alias=impl_->legacy_tensor_compatibility.canonical_to_stored.find(std::string(name));
        if(alias!=impl_->legacy_tensor_compatibility.canonical_to_stored.end())found=impl_->records_by_name.find(alias->second);
    }
    if(found==impl_->records_by_name.end() || impl_->records[found->second].asset)
        throw std::runtime_error("mapped model tensor not found: "+std::string(name));
    auto range=map_tensor_range(impl_->records[found->second]);
    return [range](std::uint64_t offset,std::byte* destination,std::size_t size){range->read(offset,destination,size);};
}

bool MfqModelSource::supports_parallel_tensor_reads() const noexcept {
    return impl_->read_mode == FileReadMode::Direct;
}
ModelSource::TensorBatchReader MfqModelSource::mapped_tensor_batch_reader(std::string_view name) const {
    auto found=impl_->records_by_name.find(std::string(name));
    if(found==impl_->records_by_name.end()) {
        const auto alias=impl_->legacy_tensor_compatibility.canonical_to_stored.find(std::string(name));
        if(alias!=impl_->legacy_tensor_compatibility.canonical_to_stored.end())found=impl_->records_by_name.find(alias->second);
    }
    if(found==impl_->records_by_name.end() || impl_->records[found->second].asset)
        throw std::runtime_error("mapped model tensor not found: "+std::string(name));
    auto range=map_tensor_range(impl_->records[found->second]);
    return [range](const std::vector<ReadSpan>& spans){range->read_batch(spans);};
}
ModelSource::TensorBatchReader MfqModelSource::tensor_batch_reader(std::string_view name) const {
    auto found=impl_->records_by_name.find(std::string(name));
    if(found==impl_->records_by_name.end()) {
        const auto alias=impl_->legacy_tensor_compatibility.canonical_to_stored.find(std::string(name));
        if(alias!=impl_->legacy_tensor_compatibility.canonical_to_stored.end())found=impl_->records_by_name.find(alias->second);
    }
    if(found==impl_->records_by_name.end() || impl_->records[found->second].asset)throw std::runtime_error("model tensor not found: "+std::string(name));
    const auto record=impl_->records[found->second];
    std::shared_ptr<AsyncRangeReader> reader;
    {
        std::lock_guard lock(impl_->pipeline_reader_mutex);
        auto& slot=impl_->pipeline_readers[record.source_path];
        if(!slot)slot=std::make_shared<AsyncRangeReader>(record.source_path,impl_->read_mode);
        reader=slot;
    }
    return [record,reader](const std::vector<ReadSpan>& spans) {
        std::vector<ReadSpan> absolute;absolute.reserve(spans.size());
        for(auto span:spans) {
            if(span.offset>record.tensor.nbytes || span.size>record.tensor.nbytes-span.offset)
                throw std::out_of_range("batched tensor read exceeds canonical record");
            span.offset+=record.offset;absolute.push_back(span);
        }
        reader->read(absolute);
    };
}

FileReadStats MfqModelSource::file_read_stats() const noexcept {
    FileReadStats result;
    result.mode = impl_->read_mode;
    for (const auto& [path, reader] : impl_->direct_readers) {
        const auto value = reader->stats();
        result.files += value.files; result.calls += value.calls;
        result.logical_bytes += value.logical_bytes; result.physical_bytes += value.physical_bytes;
        result.errors += value.errors; result.staging_bytes += value.staging_bytes;
        result.staging_peak_bytes += value.staging_peak_bytes; result.read_nanoseconds += value.read_nanoseconds;
        result.staging_cached_bytes+=value.staging_cached_bytes;result.staging_allocations+=value.staging_allocations;
    }
    {
        std::lock_guard lock(impl_->pipeline_reader_mutex);
        for(const auto& [path,reader]:impl_->pipeline_readers) {
            const auto value=reader->stats();result.calls+=value.calls;result.logical_bytes+=value.logical_bytes;
            result.physical_bytes+=value.physical_bytes;result.errors+=value.errors;
            result.staging_bytes+=value.staging_bytes;result.staging_peak_bytes+=value.staging_peak_bytes;
            result.read_nanoseconds+=value.read_nanoseconds;
            result.staging_cached_bytes+=value.staging_cached_bytes;result.staging_allocations+=value.staging_allocations;
        }
    }
    return result;
}

void MfqModelSource::drop_file_cache() const noexcept {
#if !defined(_WIN32) && defined(POSIX_FADV_DONTNEED)
    for (const auto& path : impl_->source_paths) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) continue;
        static_cast<void>(::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED));
        static_cast<void>(::close(fd));
    }
#endif
}

const std::vector<std::string>& MfqModelSource::assets() const noexcept {
    return impl_->assets;
}

bool MfqModelSource::has_asset(std::string_view name) const noexcept {
    const auto found = impl_->records_by_name.find(std::string(name));
    return found != impl_->records_by_name.end() &&
           impl_->records[found->second].asset;
}

std::vector<std::byte> MfqModelSource::read_asset(std::string_view name) const {
    const auto found = impl_->records_by_name.find(std::string(name));
    if (found == impl_->records_by_name.end() ||
        !impl_->records[found->second].asset) {
        throw std::runtime_error("model asset not found: " + std::string(name));
    }
    const auto& record = impl_->records[found->second];
    if (record.tensor.nbytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("model asset is too large to read");
    }
    std::vector<std::byte> result(static_cast<std::size_t>(record.tensor.nbytes));
    if (impl_->read_mode == FileReadMode::Direct)
        impl_->direct_readers.at(record.source_path)->read(record.offset, result.data(), result.size());
    else read_exact(record.source_path, record.offset, result.data(), result.size());
    return result;
}

std::optional<ModelGraph> MfqModelSource::model_graph() const {
    if (!has_asset(kModelGraphAsset)) return std::nullopt;
    const auto bytes = read_asset(kModelGraphAsset);
    return ModelGraph::from_json(std::string_view(
        reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

} // namespace mfq
