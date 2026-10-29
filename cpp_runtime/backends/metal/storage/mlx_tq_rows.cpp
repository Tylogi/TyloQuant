#include "mlx_tq_rows.h"
#include "mfq_tq_embedded.h"

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mfq::metal {
namespace {

template <typename T> T load(std::span<const std::uint8_t> bytes, std::size_t offset) {
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
        throw std::runtime_error("truncated TQ row table");
    T result;
    std::memcpy(&result, bytes.data() + offset, sizeof(T));
    return result;
}

const std::array<float, 16>& relative_scales() {
    static const auto values = [] {
        std::array<float, 16> result;
        for (int i = 0; i < 16; ++i)
            result[i] = float(mlx::core::float16_t(std::exp2(-2.f + 4.f * i / 15.f)));
        return result;
    }();
    return values;
}

} // namespace

MlxMappedTqRows::MlxMappedTqRows(std::span<const std::uint8_t> blob, int width)
    : blob_(blob) {
    if (blob.size() >= 64 && !std::memcmp(blob.data(), "RTQEXP01", 8)) {
        const auto count = load<std::uint32_t>(blob, 12), out = load<std::uint32_t>(blob, 16),
            logical = load<std::uint32_t>(blob, 20);
        if (load<std::uint32_t>(blob, 8) != 1 || !count || !out || !logical || logical > INT_MAX
            || std::uint64_t(count) * out > INT_MAX || count > (blob.size() - 64) / 16
            || load<std::uint32_t>(blob, 24) != 16 || load<std::uint32_t>(blob, 28)
            || (width && width != int(logical))
            || std::any_of(blob.begin() + 32, blob.begin() + 64, [](auto b) { return b != 0; }))
            throw std::runtime_error("invalid TQ expert container header");
        std::size_t end = 64 + std::size_t(count) * 16;
        experts_.reserve(count);
        expert_offsets_.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            const auto offset = load<std::uint64_t>(blob, 64 + index * 16),
                size = load<std::uint64_t>(blob, 72 + index * 16);
            if (offset != end || offset > blob.size() || size < 64 || size > blob.size() - offset
                || std::memcmp(blob.data() + offset, "RTRELL05", 8))
                throw std::runtime_error("invalid TQ expert container span");
            experts_.emplace_back(blob.subspan(offset, size), logical);
            if (experts_.back().rows() != out || (index && experts_.back().stored_width() != experts_.front().stored_width()))
                throw std::runtime_error("TQ expert container dimensions differ");
            expert_offsets_.push_back(offset);
            end = offset + size;
        }
        if (end != blob.size()) throw std::runtime_error("TQ expert container length differs");
        rows_ = count * out; width_ = logical;
        stored_width_ = experts_.front().stored_width(); group_size_ = experts_.front().group_size();
        columns_ = experts_.front().columns();
        groups_ = scales_offset_ = packed_offset_ = 0;
        return;
    }
    if (blob.size() < 64 || std::memcmp(blob.data(), "RTRELL05", 8)
        || load<std::uint32_t>(blob, 8) != 5)
        throw std::runtime_error("invalid TQ row table header");
    const auto rows = load<std::uint32_t>(blob, 12);
    const auto stored_width = load<std::uint32_t>(blob, 16);
    const auto columns = load<std::uint32_t>(blob, 20);
    const auto gs = load<std::uint32_t>(blob, 24);
    const auto packed = load<std::uint32_t>(blob, 28);
    if (!rows || rows > INT_MAX || !stored_width || stored_width > INT_MAX
        || stored_width % 256 || columns > 1 || width < 0 || width > int(stored_width)
        || (!columns && (gs < 16 || gs > (1u << 20) || (gs & (gs - 1))))
        || std::any_of(blob.begin() + 32, blob.begin() + 64, [](auto b) { return b != 0; }))
        throw std::runtime_error("invalid TQ row table geometry");
    rows_ = rows; stored_width_ = stored_width; width_ = width ? width : stored_width_;
    group_size_ = columns ? 512 : gs; columns_ = columns;
    groups_ = columns ? 0 : (std::size_t(stored_width_) + gs - 1) / gs;
    scales_offset_ = 64 + std::size_t(rows_) * 8;
    const std::size_t scale_bytes = columns ? std::size_t(stored_width_) * 2
        : (std::size_t(rows_) * groups_ + 1) / 2;
    packed_offset_ = scales_offset_ + scale_bytes;
    if (packed % 4 || packed_offset_ > blob.size() || blob.size() - packed_offset_ != packed)
        throw std::runtime_error("TQ row table payload length differs");
    if (columns_) for (int c = 0; c < stored_width_; ++c) {
        const auto scale = load<std::uint16_t>(blob, scales_offset_ + c * 2);
        if (!scale || (scale & 0x8000) || (scale & 0x7c00) == 0x7c00)
            throw std::runtime_error("invalid TQ row table column scale");
    }
}

const MlxMappedTqRows& MlxMappedTqRows::expert(std::size_t index) const {
    if (experts_.empty()) {
        if (index) throw std::out_of_range("TQ expert index is outside the tensor");
        return *this;
    }
    return experts_.at(index);
}

std::size_t MlxMappedTqRows::selected_row_nbytes_bound() const noexcept {
    if (!experts_.empty()) {
        std::size_t bound = 0;
        for (const auto& expert : experts_) bound = std::max(bound, expert.selected_row_nbytes_bound());
        return bound;
    }
    return 16 + stored_width_ + (columns_ ? std::size_t(stored_width_) : groups_) * 4;
}

std::uint64_t MlxMappedTqRows::descriptor(std::int64_t row) const {
    if (row < 0 || row >= rows_) throw std::out_of_range("TQ row index is outside the table");
    const auto descriptor = load<std::uint64_t>(blob_, 64 + std::size_t(row) * 8);
    const auto anchor = std::uint16_t(descriptor >> 40);
    const std::size_t bytes = std::size_t(stored_width_) * (((descriptor >> 32) & 7) + 1) / 8;
    const std::size_t begin = std::uint32_t(descriptor) * 4ull;
    if ((descriptor & ~std::uint64_t(0x00FFFF07FFFFFFFF)) || (anchor & 0x8000)
        || (anchor & 0x7c00) == 0x7c00 || begin > blob_.size() - packed_offset_
        || bytes > blob_.size() - packed_offset_ - begin)
        throw std::runtime_error("invalid TQ selected row descriptor");
    return descriptor;
}

MlxMappedTqRows::PackedRows MlxMappedTqRows::packed_rows(std::int64_t first, std::size_t count) const {
    if (first < 0 || first >= rows_ || count == 0 || count > std::size_t(rows_ - first))
        throw std::out_of_range("TQ row span is outside the table");
    if (!experts_.empty()) {
        const auto rows = experts_.front().rows(), index = first / rows, local = first % rows;
        if (count > std::size_t(rows - local)) throw std::out_of_range("TQ packed span crosses expert records");
        auto result = experts_[index].packed_rows(local, count);
        result.offset += expert_offsets_[index];
        return result;
    }
    const auto begin = std::uint32_t(descriptor(first)) * 4ull;
    auto end = begin;
    for (std::size_t row = 0; row < count; ++row) {
        const auto current = descriptor(first + row);
        if (std::uint32_t(current) * 4ull != end)
            throw std::runtime_error("TQ selected row span is not contiguous");
        end += std::size_t(stored_width_) * (((current >> 32) & 7) + 1) / 8;
    }
    return {packed_offset_ + begin, end - begin};
}

void MlxMappedTqRows::write_descriptors(std::int64_t first, std::uint32_t word_base,
        std::span<std::uint64_t> destination) const {
    if (!experts_.empty()) {
        if (first < 0 || first >= rows_) throw std::out_of_range("TQ row index is outside the expert bank");
        const auto rows = experts_.front().rows(), index = first / rows, local = first % rows;
        if (destination.empty() || destination.size() > std::size_t(rows - local))
            throw std::out_of_range("TQ descriptor span crosses expert records");
        experts_[index].write_descriptors(local, word_base, destination);
        return;
    }
    const auto range = packed_rows(first, destination.size());
    if (range.bytes / 4 > std::uint64_t(std::numeric_limits<std::uint32_t>::max()) + 1 - word_base)
        throw std::overflow_error("TQ destination word offsets are too large");
    const auto source_base = (range.offset - packed_offset_) / 4;
    for (std::size_t row = 0; row < destination.size(); ++row) {
        const auto current = load<std::uint64_t>(blob_, 64 + std::size_t(first + row) * 8);
        destination[row] = (current & 0xffffffff00000000ull)
            | (std::uint32_t(current) - source_base + word_base);
    }
}

void MlxMappedTqRows::append_row(std::int64_t row, MlxTqRowBatch& batch) const {
    if (!experts_.empty()) {
        if (row < 0 || row >= rows_) throw std::out_of_range("TQ row index is outside the expert bank");
        const auto rows = experts_.front().rows();
        experts_[row / rows].append_row(row % rows, batch);
        return;
    }
    const auto descriptor = this->descriptor(row);
    const std::size_t bytes = std::size_t(stored_width_) * (((descriptor >> 32) & 7) + 1) / 8;
    const std::size_t begin = std::uint32_t(descriptor) * 4ull;
    if (batch.rows() && batch.width_ != width_)
        throw std::invalid_argument("TQ selected row widths differ");
    const std::size_t factors = columns_ ? stored_width_ : groups_;
    if (batch.rows() >= std::size_t(INT_MAX) / 2
        || bytes / 4 > std::size_t(INT_MAX) - batch.packed_.size()
        || factors > std::size_t(INT_MAX) - batch.scales_.size())
        throw std::overflow_error("TQ selected row batch is too large");
    batch.width_ = width_;
    batch.descriptors_.push_back((descriptor & 0xffffffff00000000ull) | batch.packed_.size());
    batch.layouts_.push_back(batch.scales_.size());
    batch.layouts_.push_back(columns_ ? 0u : std::countr_zero(unsigned(group_size_)));
    for (std::size_t group = 0; group < factors; ++group) {
        if (columns_) batch.scales_.push_back(float(load<mlx::core::float16_t>(blob_, scales_offset_ + group * 2)));
        else {
            const auto from = std::size_t(row) * groups_ + group;
            const auto code = (blob_[scales_offset_ + from / 2] >> ((from & 1) * 4)) & 15;
            batch.scales_.push_back(relative_scales()[code]);
        }
    }
    const auto word_base = batch.packed_.size();
    batch.packed_.resize(word_base + bytes / 4);
    std::memcpy(batch.packed_.data() + word_base, blob_.data() + packed_offset_ + begin, bytes);
    batch.source_bytes_ += bytes + 8 + (columns_ ? std::size_t(stored_width_) * 2
        : (std::size_t(row) * groups_ % 2 + groups_ + 1) / 2);
}

MlxMappedTqTable::MlxMappedTqTable(std::vector<MlxMappedTqRows> shards) : shards_(std::move(shards)) {
    if (shards_.empty()) throw std::invalid_argument("TQ row table has no shards");
    for (const auto& shard : shards_) {
        if (shard.width() != width() || shard.rows() > std::numeric_limits<std::int64_t>::max() - rows_)
            throw std::invalid_argument("TQ row table shard geometry differs");
        rows_ += shard.rows();
        ends_.push_back(rows_);
        row_bound_ = std::max(row_bound_, shard.selected_row_nbytes_bound());
    }
}

void MlxMappedTqTable::append_row(std::int64_t row, MlxTqRowBatch& batch) const {
    if (row < 0 || row >= rows_) throw std::out_of_range("TQ row index is outside the logical table");
    const auto shard = std::upper_bound(ends_.begin(), ends_.end(), row) - ends_.begin();
    shards_[shard].append_row(row - (shard ? ends_[shard - 1] : 0), batch);
}

std::size_t MlxTqRowBatch::packed_nbytes() const noexcept {
    return descriptors_.size() * 8 + layouts_.size() * 4 + scales_.size() * 4 + packed_.size() * 4;
}

void MlxTqRowBatch::append_batch(const MlxTqRowBatch& other) {
    if (other.rows() == 0) return;
    if (this == &other) throw std::invalid_argument("TQ row batch cannot append itself");
    if (rows() == 0) { *this = other; return; }
    if (width_ != other.width_) throw std::invalid_argument("TQ selected row widths differ");
    if (other.rows() > std::size_t(INT_MAX) / 2 - rows()
        || other.packed_.size() > std::size_t(INT_MAX) - packed_.size()
        || other.scales_.size() > std::size_t(INT_MAX) - scales_.size())
        throw std::overflow_error("TQ selected row batch is too large");
    const auto word_base = packed_.size(), scale_base = scales_.size();
    for (const auto descriptor : other.descriptors_)
        descriptors_.push_back((descriptor & 0xffffffff00000000ull)
            | (std::uint32_t(descriptor) + word_base));
    for (std::size_t i = 0; i < other.rows(); ++i) {
        layouts_.push_back(other.layouts_[i * 2] + scale_base);
        layouts_.push_back(other.layouts_[i * 2 + 1]);
    }
    packed_.insert(packed_.end(), other.packed_.begin(), other.packed_.end());
    scales_.insert(scales_.end(), other.scales_.begin(), other.scales_.end());
    source_bytes_ += other.source_bytes_;
}

mlx::core::array MlxTqRowBatch::decode() const {
    if (!rows() || rows() > std::size_t(INT_MAX))
        throw std::invalid_argument("TQ selected row batch is empty or too large");
    using namespace mlx::core;
    static const auto kernel = fast::metal_kernel("mfq_cpp_tq_selected_rows",
        {"words", "descriptors", "scales", "layouts", "relative"}, {"y"}, R"METAL(
uint col = thread_position_in_grid.x * 4u, row = thread_position_in_grid.y;
if (col >= K) return;
TqSelectedRowScales<decltype(scales)> factors{scales, layouts[row * 2u], layouts[row * 2u + 1u]};
half4 value = tq_decode4<true, 256, 512>(words, descriptors[row], factors, relative, 0u, col);
tq_store4<K>(y, row, col, value);
)METAL", std::string(kTqDecode) + R"METAL(
template <typename P> struct TqSelectedRowScales {
    P values;
    uint offset, shift;
    float operator[](uint col) const { return values[offset + (col >> shift)]; }
};
)METAL", true, false);
    static const array unused_relative = [] {
        const std::array<float16_t, 16> values{};
        return array(values.data(), Shape{16});
    }();
    return kernel({array(packed_.data(), Shape{int(packed_.size())}),
        array(descriptors_.data(), Shape{int(rows())}), array(scales_.data(), Shape{int(scales_.size())}),
        array(layouts_.data(), Shape{int(layouts_.size())}), unused_relative},
        {Shape{int(rows()), width_}}, {float16}, {(width_ + 3) / 4, int(rows()), 1}, {128, 1, 1},
        {{"K", width_}}, std::nullopt, false, {}).front();
}

} // namespace mfq::metal
