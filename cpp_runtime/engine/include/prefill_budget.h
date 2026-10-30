#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mfq::engine {

inline constexpr int64_t max_prefill_compute_chunk = 8192;

inline std::size_t prefill_budget_add(std::size_t a, std::size_t b) {
    return a > std::numeric_limits<std::size_t>::max() - b
        ? std::numeric_limits<std::size_t>::max() : a + b;
}

inline std::size_t prefill_budget_multiply(std::size_t a, std::size_t b) {
    return b && a > std::numeric_limits<std::size_t>::max()/b
        ? std::numeric_limits<std::size_t>::max() : a*b;
}

// One layer's routed Q8 activation and per-group float scales.
struct PrefillMoeGeometry {
    std::size_t routes = 0, intermediate = 0, quant_group = 0;
    std::size_t activation_bytes_per_token() const {
        if (!routes || !intermediate || !quant_group) return 0;
        const auto groups=intermediate/quant_group+(intermediate%quant_group!=0);
        return prefill_budget_multiply(routes,
            prefill_budget_multiply(groups,prefill_budget_add(quant_group,sizeof(float))));
    }
};

// Select before allocating or borrowing. Transfer rings and persistent state
// have first claim on memory; only the remaining budget can grow the tile.
// Zero means the preferred tile cannot fit this estimate. Callers may use a
// smaller pressure tile and let the transactional retry establish actual fit.
inline int64_t select_prefill_chunk(int64_t tokens, int64_t limit,
        std::size_t available, std::size_t fixed_bytes, std::size_t row_bytes) {
    if(tokens <= 0 || limit <= 0 || row_bytes == 0)
        throw std::invalid_argument("invalid prefill workspace geometry");
    const auto wanted = std::min({tokens, limit, max_prefill_compute_chunk});
    const auto rows = available > fixed_bytes ? (available-fixed_bytes)/row_bytes : 0;
    auto selected = static_cast<int64_t>(std::min<std::size_t>(wanted, rows));
    if(selected == wanted) return selected;
    const auto minimum = std::min<int64_t>(wanted, 256);
    if(selected < minimum) return 0;
    return selected/256*256;
}

// Headroom estimates may be conservative. Actual fit is checked by a forward
// transaction, down to one token, before reporting capacity exhaustion.
inline int64_t pressure_prefill_chunk(int64_t tokens, int64_t limit,
        std::size_t available, std::size_t fixed_bytes, std::size_t row_bytes) {
    if (const auto preferred=select_prefill_chunk(tokens,limit,available,fixed_bytes,row_bytes))
        return preferred;
    const auto rows=available>fixed_bytes?(available-fixed_bytes)/row_bytes:0;
    return std::max<int64_t>(1,std::min<std::size_t>({std::size_t(tokens),std::size_t(limit),
        std::size_t(max_prefill_compute_chunk),rows}));
}

} // namespace mfq::engine
