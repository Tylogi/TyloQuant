#include "nintm_expert_store.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace mfq::metal {
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

std::array<std::span<std::byte>, 6> destinations(
    const MlxNativeMxfp4ExpertDestination& value) {
    return {
        value.w1_scale,
        value.w2_scale,
        value.w3_scale,
        value.w1_weight,
        value.w2_weight,
        value.w3_weight,
    };
}

} // namespace

MlxNintMxfp4ExpertStore::MlxNintMxfp4ExpertStore(
    const MfqContainer& model,
    std::vector<std::string> layer_prefixes,
    std::vector<std::size_t> experts_per_layer,
    std::size_t hidden_size,
    std::size_t intermediate_size)
    : model_(model),
      num_layers_(layer_prefixes.size()),
      experts_per_layer_(std::move(experts_per_layer)) {
    if (num_layers_ == 0 || experts_per_layer_.size() != num_layers_ ||
        std::find(experts_per_layer_.begin(), experts_per_layer_.end(), 0) !=
            experts_per_layer_.end() || hidden_size == 0 ||
        intermediate_size == 0) {
        throw std::invalid_argument("NINTM MXFP4 expert geometry is invalid");
    }
    if (hidden_size % 32 != 0 || intermediate_size % 32 != 0) {
        throw MlxNintMxfp4Unsupported(
            "SSD MXFP4 experts require block-aligned geometry");
    }
    max_num_experts_ = *std::max_element(
        experts_per_layer_.begin(), experts_per_layer_.end());
    expert_offsets_.reserve(num_layers_ + 1);
    expert_offsets_.push_back(0);
    for (const auto count : experts_per_layer_) {
        if (count > std::numeric_limits<std::size_t>::max() -
                expert_offsets_.back()) {
            throw std::runtime_error("SSD expert table size overflows");
        }
        expert_offsets_.push_back(expert_offsets_.back() + count);
    }
    experts_.resize(expert_offsets_.back());
    for (std::size_t layer = 0; layer < num_layers_; ++layer) {
        const auto layer_experts = experts_per_layer_[layer];
        struct ProjectionSlice {
            std::size_t scale_part;
            std::size_t value_part;
            std::size_t row_offset;
            std::size_t rows;
        };
        const auto parse_projection = [&](
            const std::string& name,
            std::size_t output,
            std::size_t input,
            std::span<const ProjectionSlice> slices) {
            const auto& outer = model_.record(name);
            if (outer.dtype != "NINTM" || outer.nbytes < 20) {
                throw MlxNintMxfp4Unsupported(
                    "SSD expert projection is not NINTM: " + name);
            }
            const auto header = model_.read_range(name, 0, 20);
            if (std::memcmp(header.data(), "NIM2", 4) != 0) {
                throw MlxNintMxfp4Unsupported(
                    "SSD expert projection is not NIM2: " + name);
            }
            if (little<std::uint32_t>(header, 4) != layer_experts ||
                little<std::uint32_t>(header, 8) != output ||
                little<std::uint32_t>(header, 12) != input) {
                throw std::runtime_error(
                    "SSD expert NINTM geometry mismatch: " + name);
            }
            const auto pools = little<std::uint32_t>(header, 16);
            if (pools == 0 || pools > layer_experts) {
                throw std::runtime_error(
                    "SSD expert NINTM pool count is invalid: " + name);
            }
            std::vector<std::uint8_t> present(layer_experts, 0);
            std::uint64_t cursor = 20;
            for (std::uint32_t pool = 0; pool < pools; ++pool) {
                if (cursor > outer.nbytes || 24 > outer.nbytes - cursor) {
                    throw std::runtime_error(
                        "truncated SSD expert NINTM pool: " + name);
                }
                const auto pool_header = model_.read_range(name, cursor, 24);
                const auto count = little<std::uint32_t>(pool_header, 0);
                const auto dtype_bytes = little<std::uint32_t>(pool_header, 4);
                const auto payload_bytes = little<std::uint64_t>(pool_header, 8);
                const auto runtime_bytes = little<std::uint64_t>(pool_header, 16);
                if (count == 0 || count > layer_experts || dtype_bytes == 0 ||
                    dtype_bytes > 32 || runtime_bytes != 0) {
                    throw std::runtime_error(
                        "unsupported SSD expert NINTM pool metadata: " + name);
                }
                cursor = checked_add(cursor, 24, "NINTM pool offset");
                const auto ids_bytes = checked_product(
                    count, sizeof(std::int32_t), "NINTM expert IDs");
                const auto metadata_bytes = checked_add(
                    ids_bytes, dtype_bytes, "NINTM pool metadata");
                if (cursor > outer.nbytes || metadata_bytes > outer.nbytes - cursor) {
                    throw std::runtime_error(
                        "truncated SSD expert NINTM metadata: " + name);
                }
                const auto metadata = model_.read_range(
                    name, cursor, metadata_bytes);
                const std::string dtype(
                    reinterpret_cast<const char*>(metadata.data() + ids_bytes),
                    dtype_bytes);
                if (dtype != "MXFP4") {
                    throw MlxNintMxfp4Unsupported(
                        "SSD arena requires MXFP4 NINTM experts: " + name);
                }
                const auto payload_offset = checked_add(
                    cursor, metadata_bytes, "NINTM payload offset");
                const auto payload_end = checked_add(
                    payload_offset, payload_bytes, "NINTM payload end");
                if (payload_end > outer.nbytes || payload_bytes < 56) {
                    throw std::runtime_error(
                        "truncated SSD expert MXFP4 payload: " + name);
                }
                const auto mx = model_.read_range(name, payload_offset, 56);
                const auto expected_rows = checked_product(
                    count, output, "MXFP4 rows");
                const auto columns = static_cast<std::uint64_t>(input);
                if (std::memcmp(mx.data(), "MXT1", 4) != 0 || mx[4] != 1 ||
                    mx[5] != 4 || little<std::uint16_t>(mx, 6) != 0 ||
                    little<std::uint64_t>(mx, 8) != expected_rows ||
                    little<std::uint64_t>(mx, 16) != columns ||
                    little<std::uint64_t>(mx, 24) != expected_rows ||
                    little<std::uint64_t>(mx, 32) != columns / 2 ||
                    little<std::uint64_t>(mx, 40) != expected_rows ||
                    little<std::uint64_t>(mx, 48) != columns / 32) {
                    throw std::runtime_error(
                        "unsupported SSD expert MXFP4 layout: " + name);
                }
                const auto values_per_expert = checked_product(
                    output, input / 2,
                    "MXFP4 expert values");
                const auto scales_per_expert = checked_product(
                    output, input / 32,
                    "MXFP4 expert scales");
                const auto values_bytes = checked_product(
                    count, values_per_expert, "MXFP4 values");
                const auto scales_bytes = checked_product(
                    count, scales_per_expert, "MXFP4 scales");
                if (payload_bytes != checked_add(
                        56, checked_add(values_bytes, scales_bytes,
                            "MXFP4 payload"), "MXFP4 payload")) {
                    throw std::runtime_error(
                        "SSD expert MXFP4 payload size mismatch: " + name);
                }
                const auto values_offset = checked_add(
                    payload_offset, 56, "MXFP4 values offset");
                const auto scales_offset = checked_add(
                    values_offset, values_bytes, "MXFP4 scales offset");
                for (std::uint32_t local = 0; local < count; ++local) {
                    const auto expert = little<std::int32_t>(
                        metadata, static_cast<std::size_t>(local) * 4);
                    if (expert < 0 || static_cast<std::size_t>(expert) >= layer_experts ||
                        present[static_cast<std::size_t>(expert)] != 0) {
                        throw std::runtime_error(
                            "duplicate or invalid SSD expert ID: " + name);
                    }
                    present[static_cast<std::size_t>(expert)] = 1;
                    auto& destination = experts_[expert_offsets_[layer] +
                        static_cast<std::size_t>(expert)];
                    for (const auto& slice : slices) {
                        if (slice.rows == 0 ||
                            slice.row_offset > output ||
                            slice.rows > output - slice.row_offset ||
                            slice.scale_part >= 3 ||
                            slice.value_part < 3 ||
                            slice.value_part >= kParts) {
                            throw std::logic_error(
                                "invalid MXFP4 expert projection slice");
                        }
                        const auto first_row = checked_add(
                            checked_product(local, output,
                                "MXFP4 expert row offset"),
                            slice.row_offset,
                            "MXFP4 expert row offset");
                        destination.parts[slice.scale_part] = {
                            name,
                            checked_add(
                                scales_offset,
                                checked_product(first_row, input / 32,
                                    "MXFP4 scale slice offset"),
                                "MXFP4 scale slice offset"),
                            checked_product(slice.rows, input / 32,
                                "MXFP4 scale slice bytes"),
                        };
                        destination.parts[slice.value_part] = {
                            name,
                            checked_add(
                                values_offset,
                                checked_product(first_row, input / 2,
                                    "MXFP4 value slice offset"),
                                "MXFP4 value slice offset"),
                            checked_product(slice.rows, input / 2,
                                "MXFP4 value slice bytes"),
                        };
                    }
                }
                cursor = payload_end;
            }
            if (cursor != outer.nbytes ||
                std::find(present.begin(), present.end(), 0) != present.end()) {
                throw std::runtime_error(
                    "SSD expert NINTM does not cover every expert: " + name);
            }
        };

        const auto experts = layer_prefixes[layer] + ".mlp.experts.";
        const std::array<ProjectionSlice, 1> down{{
            {1, 4, 0, hidden_size},
        }};
        parse_projection(
            experts + "down.weight",
            hidden_size,
            intermediate_size,
            down);

        const bool has_gate = model_.contains(experts + "gate.weight");
        const bool has_up = model_.contains(experts + "up.weight");
        if (has_gate != has_up) {
            throw std::runtime_error(
                "incomplete canonical Gate/Up NINTM pair: " + experts);
        }
        if (has_gate) {
            const std::array<ProjectionSlice, 1> gate{{
                {0, 3, 0, intermediate_size},
            }};
            const std::array<ProjectionSlice, 1> up{{
                {2, 5, 0, intermediate_size},
            }};
            parse_projection(
                experts + "gate.weight",
                intermediate_size,
                hidden_size,
                gate);
            parse_projection(
                experts + "up.weight",
                intermediate_size,
                hidden_size,
                up);
        } else if (model_.contains(experts + "gate_up.weight")) {
            const std::array<ProjectionSlice, 2> gate_up{{
                {0, 3, 0, intermediate_size},
                {2, 5, intermediate_size, intermediate_size},
            }};
            parse_projection(
                experts + "gate_up.weight",
                checked_product(2, intermediate_size, "GateUp output width"),
                hidden_size,
                gate_up);
        } else {
            throw MlxNintMxfp4Unsupported(
                "SSD expert layer has no canonical Gate/Up projection: " +
                experts);
        }
    }

    const auto& first = experts_.front();
    for (std::size_t part = 0; part < kParts; ++part) {
        if (first.parts[part].nbytes >
            std::numeric_limits<std::size_t>::max() - slot_offsets_[part]) {
            throw std::runtime_error("SSD expert slot size overflows");
        }
        slot_offsets_[part + 1] = slot_offsets_[part] +
            static_cast<std::size_t>(first.parts[part].nbytes);
    }
    slot_bytes_ = slot_offsets_.back();
    for (const auto& expert : experts_) {
        for (std::size_t part = 0; part < kParts; ++part) {
            if (expert.parts[part].nbytes != first.parts[part].nbytes) {
                throw std::runtime_error(
                    "SSD NINTM expert tensors have inconsistent sizes");
            }
        }
    }
}

std::size_t MlxNintMxfp4ExpertStore::num_layers() const noexcept {
    return num_layers_;
}

std::size_t MlxNintMxfp4ExpertStore::num_experts(
    std::size_t layer) const {
    if (layer >= num_layers_) {
        throw std::out_of_range("SSD NINTM expert layer out of range");
    }
    return experts_per_layer_[layer];
}

std::size_t MlxNintMxfp4ExpertStore::max_num_experts() const noexcept {
    return max_num_experts_;
}

std::size_t MlxNintMxfp4ExpertStore::slot_bytes() const noexcept {
    return slot_bytes_;
}

const MlxNintMxfp4ExpertStore::ExpertRecord&
MlxNintMxfp4ExpertStore::expert_record(
    std::size_t layer,
    std::size_t expert) const {
    if (layer >= num_layers_ || expert >= experts_per_layer_[layer]) {
        throw std::out_of_range("SSD NINTM expert index out of range");
    }
    return experts_[expert_offsets_[layer] + expert];
}

MlxNativeMxfp4ExpertLoadStats MlxNintMxfp4ExpertStore::load_parts(
    const ExpertRecord& record,
    std::span<const std::size_t> parts,
    const MlxNativeMxfp4ExpertDestination& destination) const {
    const auto targets = destinations(destination);
    MlxNativeMxfp4ExpertLoadStats result;
    for (const auto part : parts) {
        if (part >= kParts || targets[part].size() != record.parts[part].nbytes) {
            throw std::runtime_error(
                "SSD NINTM expert destination size mismatch");
        }
        model_.read_range_into(
            record.parts[part].record,
            record.parts[part].offset,
            targets[part]);
        result.bytes += record.parts[part].nbytes;
        ++result.read_calls;
    }
    return result;
}

MlxNativeMxfp4ExpertLoadStats MlxNintMxfp4ExpertStore::load(
    std::size_t layer,
    std::size_t expert,
    std::span<std::byte> slot) const {
    if (slot.size() < slot_bytes_) {
        throw std::runtime_error("SSD NINTM expert slot is too small");
    }
    return load_scatter(layer, expert, {
        .w1_scale = slot.subspan(slot_offsets_[0], slot_offsets_[1] - slot_offsets_[0]),
        .w2_scale = slot.subspan(slot_offsets_[1], slot_offsets_[2] - slot_offsets_[1]),
        .w3_scale = slot.subspan(slot_offsets_[2], slot_offsets_[3] - slot_offsets_[2]),
        .w1_weight = slot.subspan(slot_offsets_[3], slot_offsets_[4] - slot_offsets_[3]),
        .w2_weight = slot.subspan(slot_offsets_[4], slot_offsets_[5] - slot_offsets_[4]),
        .w3_weight = slot.subspan(slot_offsets_[5], slot_offsets_[6] - slot_offsets_[5]),
    });
}

MlxNativeMxfp4ExpertLoadStats MlxNintMxfp4ExpertStore::load_scatter(
    std::size_t layer,
    std::size_t expert,
    const MlxNativeMxfp4ExpertDestination& destination) const {
    constexpr std::array<std::size_t, 6> parts{0, 1, 2, 3, 4, 5};
    return load_parts(expert_record(layer, expert), parts, destination);
}

MlxNativeMxfp4ExpertLoadStats
MlxNintMxfp4ExpertStore::load_gate_up_scatter(
    std::size_t layer,
    std::size_t expert,
    const MlxNativeMxfp4ExpertDestination& destination) const {
    const auto scales = load_scales_scatter(layer, expert, destination);
    const auto gate = load_gate_scatter(layer, expert, destination);
    const auto up = load_up_scatter(layer, expert, destination);
    return {
        .bytes = scales.bytes + gate.bytes + up.bytes,
        .read_calls = scales.read_calls + gate.read_calls + up.read_calls,
    };
}

MlxNativeMxfp4ExpertLoadStats MlxNintMxfp4ExpertStore::load_scales_scatter(
    std::size_t layer,
    std::size_t expert,
    const MlxNativeMxfp4ExpertDestination& destination) const {
    constexpr std::array<std::size_t, 3> parts{0, 1, 2};
    return load_parts(expert_record(layer, expert), parts, destination);
}

MlxNativeMxfp4ExpertLoadStats MlxNintMxfp4ExpertStore::load_gate_scatter(
    std::size_t layer,
    std::size_t expert,
    const MlxNativeMxfp4ExpertDestination& destination) const {
    constexpr std::array<std::size_t, 1> parts{3};
    return load_parts(expert_record(layer, expert), parts, destination);
}

MlxNativeMxfp4ExpertLoadStats MlxNintMxfp4ExpertStore::load_up_scatter(
    std::size_t layer,
    std::size_t expert,
    const MlxNativeMxfp4ExpertDestination& destination) const {
    constexpr std::array<std::size_t, 1> parts{5};
    return load_parts(expert_record(layer, expert), parts, destination);
}

MlxNativeMxfp4ExpertLoadStats MlxNintMxfp4ExpertStore::load_down_scatter(
    std::size_t layer,
    std::size_t expert,
    const MlxNativeMxfp4ExpertDestination& destination) const {
    constexpr std::array<std::size_t, 1> parts{4};
    return load_parts(expert_record(layer, expert), parts, destination);
}

MlxNativeMxfp4ExpertView MlxNintMxfp4ExpertStore::view(
    std::span<const std::byte> slot) const {
    if (slot.size() < slot_bytes_) {
        throw std::runtime_error("SSD NINTM expert slot is too small");
    }
    return {
        .w1_scale = slot.subspan(slot_offsets_[0], slot_offsets_[1] - slot_offsets_[0]),
        .w2_scale = slot.subspan(slot_offsets_[1], slot_offsets_[2] - slot_offsets_[1]),
        .w3_scale = slot.subspan(slot_offsets_[2], slot_offsets_[3] - slot_offsets_[2]),
        .w1_weight = slot.subspan(slot_offsets_[3], slot_offsets_[4] - slot_offsets_[3]),
        .w2_weight = slot.subspan(slot_offsets_[4], slot_offsets_[5] - slot_offsets_[4]),
        .w3_weight = slot.subspan(slot_offsets_[5], slot_offsets_[6] - slot_offsets_[5]),
    };
}

} // namespace mfq::metal
