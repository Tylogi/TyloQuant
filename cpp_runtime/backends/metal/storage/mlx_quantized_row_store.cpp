#include "mlx_quantized_row_store.h"
#include "mlx_resource_telemetry.h"
#include "mlx_nint_rows.h"
#include "mfq/host_parallel.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

namespace mfq::metal {
namespace {

constexpr std::size_t kMxHeaderBytes = 56;

struct RowReadCounts {
    std::uint64_t bytes = 0, calls = 0;
};

thread_local RowReadCounts* current_row_reads = nullptr;

struct RowReadScope {
    explicit RowReadScope(RowReadCounts& value)
        : previous(std::exchange(current_row_reads, &value)) {}
    ~RowReadScope() { current_row_reads = previous; }
    RowReadCounts* previous;
};

std::uint64_t read_u64_at(
    std::span<const std::uint8_t> bytes,
    std::size_t offset) {
    if (offset > bytes.size() || 8 > bytes.size() - offset) {
        throw std::runtime_error("truncated native MXFP8 header");
    }
    std::uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(bytes[offset + index])
            << (8 * index);
    }
    return value;
}

std::uint64_t checked_product(
    std::uint64_t left,
    std::uint64_t right,
    const char* description) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(
            std::string("native MXFP8 ") + description + " size overflow");
    }
    return left * right;
}

float decode_e4m3(std::uint8_t raw) {
    if ((raw & 0x7fu) == 0x7fu) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float sign = (raw & 0x80u) ? -1.0f : 1.0f;
    const int exponent = (raw >> 3u) & 0x0fu;
    const int mantissa = raw & 0x07u;
    return sign * (exponent == 0
        ? std::ldexp(static_cast<float>(mantissa), -9)
        : std::ldexp(
              1.0f + static_cast<float>(mantissa) / 8.0f,
              exponent - 7));
}

float decode_e8m0(std::uint8_t raw) {
    return raw == 255
        ? std::numeric_limits<float>::quiet_NaN()
        : std::ldexp(1.0f, static_cast<int>(raw) - 127);
}

} // namespace

struct MlxQuantizedRowStore::Impl {
    struct CacheEntry {
        std::vector<mlx::core::float16_t> values;
        std::list<std::int64_t>::iterator recency;
        mfq::NintRowBatch packed;
        MlxTqRowBatch tq;
    };

    struct RequestedRow {
        std::int64_t row = 0;
        std::vector<mlx::core::float16_t> values;
        bool cold = false;
        mfq::NintRowBatch packed;
        MlxTqRowBatch tq;
    };

    Impl(
        const MfqContainer& selected_model,
        std::string selected_record,
        std::int64_t expected_rows,
        int expected_width,
        std::size_t selected_cache_rows,
        std::size_t io_workers)
        : model(selected_model),
          record(std::move(selected_record)),
          cache_capacity(selected_cache_rows) {
        if (io_workers == 0 || io_workers > static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw std::invalid_argument("quantized row store requires a valid I/O worker count");
        }
        worker_count = static_cast<int>(io_workers);
        std::vector<std::string> prepared_names;
        const auto* entry = model.contains(record) ? &model.record(record) : nullptr;
        if (!entry || entry->dtype == "TQ") {
            std::vector<std::string> names;
            const auto manifest = model.header().extra_json.find("tq_row_shards");
            const auto tables = manifest == model.header().extra_json.end() ? nlohmann::json::object()
                : nlohmann::json::parse(manifest->second);
            if (!tables.is_object()) throw std::runtime_error("invalid TQ row-shard manifest");
            if (entry) {
                if (tables.contains(record)) throw std::runtime_error("ambiguous TQ row table: " + record);
                names.push_back(record);
            } else {
                if (!tables.contains(record) || !tables.at(record).is_array() || tables.at(record).empty())
                    throw std::runtime_error("missing TQ row table: " + record);
                for (const auto& name : tables.at(record)) {
                    if (!name.is_string()) throw std::runtime_error("TQ row shard must name a physical record");
                    names.push_back(name.get<std::string>());
                }
            }
            std::unordered_set<std::string> unique;
            std::vector<MlxMappedTqRows> shards;
            for (const auto& name : names) {
                if (!unique.insert(name).second || model.record(name).dtype != "TQ")
                    throw std::runtime_error("invalid TQ row table shard: " + name);
                tq_mappings.push_back(model.map_record(name, true));
                shards.emplace_back(tq_mappings.back().view(), expected_width);
            }
            tq = std::make_unique<MlxMappedTqTable>(std::move(shards));
            if (tq->rows() != expected_rows || tq->width() != expected_width)
                throw std::runtime_error("TQ row-store geometry disagrees: " + record);
            rows = expected_rows; width = expected_width;
            decoded_row_bytes = tq->selected_row_nbytes_bound();
            prepared_names = std::move(names);
        } else if (entry->dtype == "NINT") {
            nint = std::make_unique<mfq::NintRows>(
                static_cast<std::size_t>(entry->nbytes),
                [this](std::size_t offset, std::uint8_t* output, std::size_t count) {
                    model.read_range_into(record, offset, std::as_writable_bytes(
                        std::span(output, count)));
                }, true, [this](const std::vector<mfq::ReadSpan>& spans) {
                    read_batch(spans);
                });
            if (nint->rows() != expected_rows || nint->width() != expected_width) {
                throw std::runtime_error("NINT row-store geometry disagrees: " + record);
            }
            rows = expected_rows;
            width = expected_width;
            decoded_row_bytes = std::max(static_cast<std::size_t>(width) *
                sizeof(mlx::core::float16_t), nint->selected_row_nbytes_bound() + 27);
            index_bytes = nint->index_nbytes();
        } else if (entry->dtype != "MXFP8") {
            throw std::runtime_error(
                "unsupported quantized row-store format: " +
                record);
        } else {
            initialize_mxfp8(*entry, expected_rows, expected_width);
        }
        if (cache_capacity != 0 && decoded_row_bytes >
                std::numeric_limits<std::size_t>::max() / cache_capacity) {
            throw std::overflow_error("quantized row-store cache size overflow");
        }
        resources.bind([this] {
            std::lock_guard lock(mutex);
            return MlxResourceUsage{0, 0, index_bytes + resident_cache_bytes, 0, 0};
        });
        if (tq) for (const auto& name : prepared_names) model.record_prepared(name);
        else model.record_prepared(record);
    }

    void initialize_mxfp8(const MfqRecord& entry, std::int64_t expected_rows,
                          int expected_width) {
        const auto header = model.read_range(record, 0, kMxHeaderBytes);
        if (header.size() != kMxHeaderBytes ||
            std::memcmp(header.data(), "MXT1", 4) != 0 ||
            header[4] != 1 || header[5] != 8 ||
            header[6] != 0 || header[7] != 0) {
            throw std::runtime_error(
                "invalid native MXFP8 row-store header: " + record);
        }
        const auto logical_rows = read_u64_at(header, 8);
        const auto logical_width = read_u64_at(header, 16);
        const auto storage_rows = read_u64_at(header, 24);
        const auto storage_width = read_u64_at(header, 32);
        const auto scale_rows = read_u64_at(header, 40);
        const auto scale_width = read_u64_at(header, 48);
        if (expected_rows <= 0 || expected_width <= 0 ||
            expected_width % 32 != 0 ||
            logical_rows != static_cast<std::uint64_t>(expected_rows) ||
            logical_width != static_cast<std::uint64_t>(expected_width) ||
            storage_rows != logical_rows || storage_width != logical_width ||
            scale_rows != logical_rows || scale_width != logical_width / 32) {
            throw std::runtime_error(
                "native MXFP8 record is not the expected row-scaled table: " +
                record);
        }
        const auto value_bytes = checked_product(
            logical_rows, logical_width, "value");
        const auto scale_bytes = checked_product(
            scale_rows, scale_width, "scale");
        if (value_bytes > std::numeric_limits<std::uint64_t>::max() -
                kMxHeaderBytes ||
            scale_bytes > std::numeric_limits<std::uint64_t>::max() -
                kMxHeaderBytes - value_bytes ||
            kMxHeaderBytes + value_bytes + scale_bytes != entry.nbytes) {
            throw std::runtime_error(
                "native MXFP8 row-store payload size disagrees: " + record);
        }
        rows = expected_rows;
        width = expected_width;
        scale_columns = expected_width / 32;
        values_offset = kMxHeaderBytes;
        scales_offset = kMxHeaderBytes + value_bytes;
        source_row_bytes = static_cast<std::size_t>(width + scale_columns);
        decoded_row_bytes = static_cast<std::size_t>(width) *
            sizeof(mlx::core::float16_t);
        for (std::size_t scale = 0; scale < 256; ++scale) {
            const auto scale_value = decode_e8m0(
                static_cast<std::uint8_t>(scale));
            for (std::size_t value = 0; value < 256; ++value) {
                decode_lut[(scale << 8u) | value] =
                    static_cast<mlx::core::float16_t>(
                        scale_value * decode_e4m3(
                            static_cast<std::uint8_t>(value)));
            }
        }
    }

    void read_batch(const std::vector<mfq::ReadSpan>& spans) const {
        auto sorted = spans;
        const auto source_size = model.record(record).nbytes;
        for (const auto& span : sorted) {
            if (span.offset > source_size || span.size > source_size - span.offset)
                throw std::out_of_range("NINT row read exceeds its record");
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.offset < b.offset;
        });
        struct Group { std::size_t first, last; std::uint64_t begin, end; };
        std::vector<Group> groups;
        const auto page = static_cast<std::uint64_t>(getpagesize());
        for (std::size_t first = 0; first < sorted.size();) {
            const auto begin = sorted[first].offset;
            auto end = begin + sorted[first].size;
            auto last = first + 1;
            const auto limit = std::max(page, end - begin);
            while (last < sorted.size()) {
                const auto next_end = sorted[last].offset + sorted[last].size;
                if ((sorted[last].offset / page != begin / page && sorted[last].offset > end)
                    || std::max(end, next_end) - begin > limit) break;
                end = std::max(end, next_end);
                ++last;
            }
            groups.push_back({first, last, begin, end});
            first = last;
        }
        workers.run(0, groups.size(), 1, worker_count,
            [&](std::int64_t begin, std::int64_t end) {
                std::vector<std::uint8_t> scratch;
                for (auto item = begin; item < end; ++item) {
                    const auto& group = groups[item];
                    if (group.last == group.first + 1) {
                        const auto& span = sorted[group.first];
                        model.read_range_into(record, span.offset, {span.destination, span.size});
                    } else {
                        scratch.resize(group.end - group.begin);
                        model.read_range_into(record, group.begin,
                            std::as_writable_bytes(std::span(scratch)));
                        for (auto part = group.first; part < group.last; ++part)
                            std::memcpy(sorted[part].destination,
                                scratch.data() + sorted[part].offset - group.begin, sorted[part].size);
                    }
                }
            });
        if (current_row_reads) {
            current_row_reads->calls += groups.size();
            for (const auto& group : groups) current_row_reads->bytes += group.end - group.begin;
        }
    }

    void read_mxfp8(
        std::int64_t row,
        RequestedRow& requested) const {
        auto& output = requested.values;
        std::vector<std::uint8_t> raw(source_row_bytes);
        model.read_range_into(
            record,
            values_offset + static_cast<std::uint64_t>(row) *
                static_cast<std::uint64_t>(width),
            std::as_writable_bytes(std::span<std::uint8_t>(
                raw.data(), static_cast<std::size_t>(width))));
        model.read_range_into(
            record,
            scales_offset + static_cast<std::uint64_t>(row) *
                static_cast<std::uint64_t>(scale_columns),
            std::as_writable_bytes(std::span<std::uint8_t>(
                raw.data() + width,
                static_cast<std::size_t>(scale_columns))));
        output.resize(static_cast<std::size_t>(width));
        for (int group = 0; group < scale_columns; ++group) {
            const auto scale = raw[static_cast<std::size_t>(width + group)];
            if (scale == 255) {
                throw std::runtime_error(
                    "native MXFP8 row contains an invalid E8M0 scale");
            }
            const auto lookup_base = static_cast<std::size_t>(scale) << 8u;
            const int begin = group * 32;
            for (int column = begin; column < begin + 32; ++column) {
                const auto value = raw[static_cast<std::size_t>(column)];
                if ((value & 0x7fu) == 0x7fu) {
                    throw std::runtime_error(
                        "native MXFP8 row contains an E4M3 NaN");
                }
                output[static_cast<std::size_t>(column)] =
                    decode_lut[lookup_base | value];
            }
        }
    }

    void touch(
        typename std::unordered_map<std::int64_t, CacheEntry>::iterator found)
        const {
        recency.splice(recency.begin(), recency, found->second.recency);
        found->second.recency = recency.begin();
    }

    void admit(RequestedRow row) const {
        if (cache_capacity == 0) return;
        auto found = cache.find(row.row);
        if (found != cache.end()) {
            touch(found);
            return;
        }
        while (cache.size() >= cache_capacity) {
            const auto victim = recency.back();
            recency.pop_back();
            const auto& entry = cache.at(victim);
            resident_cache_bytes -= entry.values.size() * sizeof(mlx::core::float16_t)
                + entry.packed.packed_nbytes() + entry.packed.descriptors().size() * 4
                + entry.tq.packed_nbytes();
            cache.erase(victim);
        }
        recency.push_front(row.row);
        resident_cache_bytes += row.values.size() * sizeof(mlx::core::float16_t)
            + row.packed.packed_nbytes() + row.packed.descriptors().size() * 4
            + row.tq.packed_nbytes();
        cache.emplace(
            row.row,
            CacheEntry{std::move(row.values), recency.begin(), std::move(row.packed), std::move(row.tq)});
    }

    MlxQuantizedRowBatch prepare_rows(std::span<const std::int64_t> row_ids) const {
        if (row_ids.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            row_ids.size() > std::numeric_limits<std::size_t>::max() / width) {
            throw std::overflow_error("quantized row gather exceeds output limits");
        }
        if (row_ids.empty()) {
            return {width, {}, {}};
        }
        std::vector<std::size_t> request_sources;
        request_sources.reserve(row_ids.size());
        std::unordered_map<std::int64_t, std::size_t> unique_index;
        unique_index.reserve(row_ids.size());
        std::vector<RequestedRow> requested;
        requested.reserve(row_ids.size());
        std::vector<std::size_t> misses;
        misses.reserve(row_ids.size());
        {
            std::scoped_lock lock(mutex);
            for (const auto row : row_ids) {
                if (row < 0 || row >= rows) {
                    throw std::out_of_range(
                        "quantized row index is outside the table");
                }
                const auto known = unique_index.find(row);
                if (known != unique_index.end()) {
                    request_sources.push_back(known->second);
                    continue;
                }
                const auto index = requested.size();
                unique_index.emplace(row, index);
                request_sources.push_back(index);
                auto found = cache.find(row);
                if (found == cache.end()) {
                    requested.push_back({row, {}, true});
                    misses.push_back(index);
                } else {
                    touch(found);
                    requested.push_back({row, found->second.values, false, found->second.packed, found->second.tq});
                }
            }
        }

        const auto io_begin = std::chrono::steady_clock::now();
        RowReadCounts read_counts;
        if (tq) {
            workers.run(0, misses.size(), 1, worker_count,
                [&](std::int64_t begin, std::int64_t end) {
                    for (auto item = begin; item < end; ++item) {
                        const auto index = misses[item];
                        tq->append_row(requested[index].row, requested[index].tq);
                    }
                });
            for (const auto index : misses) read_counts.bytes += requested[index].tq.source_bytes_read();
        } else if (nint) {
            RowReadScope observation(read_counts);
            std::vector<std::int64_t> selected;
            selected.reserve(misses.size());
            for (const auto index : misses) selected.push_back(requested[index].row);
            mfq::NintRowBatch packed;
            nint->append_rows(selected.data(), selected.size(), packed);
            for (std::size_t item = 0; item < misses.size(); ++item)
                packed.copy_row(item, requested[misses[item]].packed);
        } else {
            workers.run(0, misses.size(), 1, worker_count,
                [&](std::int64_t begin, std::int64_t end) {
                    for (auto item = begin; item < end; ++item) {
                        const auto index = misses[item];
                        read_mxfp8(requested[index].row, requested[index]);
                    }
                });
        }
        const auto io_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - io_begin).count();

        MlxNintRowBatch packed_output;
        MlxTqRowBatch tq_output;
        std::vector<mlx::core::float16_t> output(nint || tq ? 0 :
            row_ids.size() * static_cast<std::size_t>(width));
        for (std::size_t request = 0; request < request_sources.size();
             ++request) {
            const auto& selected = requested[request_sources[request]];
            if (tq) {
                tq_output.append_batch(selected.tq);
                continue;
            }
            if (nint) {
                packed_output.append_batch(selected.packed);
                continue;
            }
            const auto& source = selected.values;
            std::copy_n(
                source.data(),
                width,
                output.data() + request * static_cast<std::size_t>(width));
        }
        {
            std::scoped_lock lock(mutex);
            counters.bytes_read += nint || tq ? read_counts.bytes : misses.size() * source_row_bytes;
            counters.read_calls += nint || tq ? read_counts.calls : misses.size() * 2;
            for (const auto index : misses) {
                admit(std::move(requested[index]));
            }
            counters.row_requests += row_ids.size();
            counters.cache_misses += misses.size();
            counters.cache_hits += row_ids.size() - misses.size();
            counters.rows_loaded += misses.size();
            counters.io_seconds += io_seconds;
        }
        return {width, std::move(output), std::move(packed_output), std::move(tq_output)};
    }

    MfqContainer model;
    std::string record;
    std::int64_t rows = 0;
    int width = 0;
    int scale_columns = 0;
    std::uint64_t values_offset = 0;
    std::uint64_t scales_offset = 0;
    std::size_t source_row_bytes = 0;
    std::size_t decoded_row_bytes = 0;
    std::size_t cache_capacity = 0;
    std::size_t index_bytes = 0;
    mutable std::size_t resident_cache_bytes = 0;
    std::unique_ptr<mfq::NintRows> nint;
    std::vector<MfqMappedBytes> tq_mappings;
    std::unique_ptr<MlxMappedTqTable> tq;
    std::array<mlx::core::float16_t, 1u << 16u> decode_lut{};
    mutable mfq::HostParallelPool workers;
    int worker_count;
    mutable std::mutex mutex;
    mutable std::list<std::int64_t> recency;
    mutable std::unordered_map<std::int64_t, CacheEntry> cache;
    mutable MlxQuantizedRowStoreStats counters;
    MlxResourceTelemetry resources;
};

MlxQuantizedRowStore::MlxQuantizedRowStore(
    const MfqContainer& model,
    std::string record,
    std::int64_t expected_rows,
    int expected_width,
    std::size_t cache_rows,
    std::size_t io_workers)
    : impl_(std::make_unique<Impl>(
          model,
          std::move(record),
          expected_rows,
          expected_width,
          cache_rows,
          io_workers)) {}

MlxQuantizedRowStore::~MlxQuantizedRowStore() = default;

mlx::core::array MlxQuantizedRowStore::gather(
    std::span<const std::int64_t> row_ids) const {
    return prepare_rows(row_ids).decode();
}

MlxQuantizedRowBatch MlxQuantizedRowStore::prepare_rows(
    std::span<const std::int64_t> row_ids) const {
    return impl_->prepare_rows(row_ids);
}

mlx::core::array MlxQuantizedRowBatch::decode() const {
    if (tq.rows()) return tq.decode();
    if (packed.rows()) return packed.decode(mlx::core::float16);
    if (width <= 0 || values.size() % static_cast<std::size_t>(width) != 0)
        throw std::invalid_argument("invalid quantized row batch");
    return mlx::core::array(values.begin(),
        mlx::core::Shape{static_cast<int>(values.size() / width), width});
}

std::int64_t MlxQuantizedRowStore::rows() const noexcept {
    return impl_->rows;
}

int MlxQuantizedRowStore::width() const noexcept {
    return impl_->width;
}

std::size_t MlxQuantizedRowStore::cached_rows() const noexcept {
    std::scoped_lock lock(impl_->mutex);
    return impl_->cache.size();
}

MlxQuantizedRowStoreStats MlxQuantizedRowStore::stats() const {
    std::scoped_lock lock(impl_->mutex);
    auto result = impl_->counters;
    result.resident_rows = impl_->cache.size();
    result.resident_payload_bytes =
        impl_->index_bytes + impl_->resident_cache_bytes;
    result.cache_limit_bytes =
        impl_->index_bytes + impl_->cache_capacity * impl_->decoded_row_bytes;
    return result;
}

void MlxQuantizedRowStore::clear() {
    std::scoped_lock lock(impl_->mutex);
    impl_->cache.clear();
    impl_->recency.clear();
    impl_->resident_cache_bytes = 0;
}

} // namespace mfq::metal
