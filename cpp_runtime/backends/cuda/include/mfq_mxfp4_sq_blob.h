#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mfq::sq {

// Frozen Metal SQ2/SQ3 v1 layout: original header and all four bitstreams.
struct Layout {
    int bits, outputs, width, base;
    std::size_t symbols, selectors, scales, palettes, bytes;
};

inline Layout layout(std::int64_t bits, std::int64_t outputs,
                     std::int64_t width, std::int64_t base) {
    if ((bits != 2 && bits != 3) || outputs <= 0 || width <= 0 ||
        outputs > std::numeric_limits<int>::max() ||
        width > std::numeric_limits<int>::max() || width % 32 ||
        base < 0 || base > 251) {
        throw std::runtime_error("invalid MXFP4-SQ geometry or scale base");
    }
    const auto weights = std::uint64_t(outputs) * std::uint64_t(width);
    const auto selectors = 24 + (weights / 8) * bits;
    const auto scales = selectors + (weights / 32 + 7) / 8;
    const auto palettes = scales + std::uint64_t(outputs) * 2;
    const auto bytes = palettes + std::uint64_t(outputs) * 5;
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("MXFP4-SQ payload size overflow");
    }
    return {int(bits), int(outputs), int(width), int(base), 24,
            std::size_t(selectors), std::size_t(scales),
            std::size_t(palettes), std::size_t(bytes)};
}

inline Layout parse(const std::uint8_t* data, std::size_t bytes) {
    if (bytes < 24 || !data || data[0] != 'S' || data[1] != 'Q' ||
        (data[2] != '2' && data[2] != '3') || data[3] != 0 ||
        data[4] != 1 || data[6] != 0 || data[7] != 0) {
        throw std::runtime_error("invalid MXFP4-SQ v1 header");
    }
    auto read_dimension = [&](int offset) {
        std::uint64_t value = 0;
        for (int byte = 0; byte < 8; ++byte)
            value |= std::uint64_t(data[offset + byte]) << (byte * 8);
        if (value > std::uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("MXFP4-SQ dimension overflow");
        return std::int64_t(value);
    };
    const auto result = layout(data[2] - '0', read_dimension(8), read_dimension(16), data[5]);
    if (bytes != result.bytes)
        throw std::runtime_error("truncated or trailing MXFP4-SQ payload");
    return result;
}

} // namespace mfq::sq
