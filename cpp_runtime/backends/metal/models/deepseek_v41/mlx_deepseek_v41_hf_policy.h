#pragma once

#include "deepseek_family_model.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace mfq::metal {

// Raw-HF V4.1 policy lives with the architecture even though the execution
// graph is shared with V4 Flash. Keep these helpers inline so moving ownership
// does not add calls to the tuned per-layer path.
inline bool deepseek_v41_hf_fast_indexer_enabled() noexcept {
    // The tiled kernel keeps indexer scores in FP32 while avoiding the
    // [B, T, H, K, D] broadcast used by the reference expression.
    const char* value = std::getenv("MFQ_METAL_DSV41_FAST_INDEXER");
    if (value == nullptr) {
        return true;
    }
    const auto setting = std::string_view(value);
    return setting != "0" && setting != "false" && setting != "off";
}

inline bool deepseek_v41_hf_circular_prefill_enabled() noexcept {
    // The direct long-prefill kernel consumes chronological local rows plus
    // the capacity-backed CSA pool. Keep a parity escape hatch.
    const char* value = std::getenv("MFQ_METAL_DSV41_CIRCULAR_PREFILL");
    if (value == nullptr) {
        return true;
    }
    const auto setting = std::string_view(value);
    return setting != "0" && setting != "false" && setting != "off";
}

inline bool deepseek_v41_hf_apple_m3_ultra() noexcept {
#if defined(__APPLE__)
    static const bool is_m3_ultra = [] {
        char name[64]{};
        std::size_t size = sizeof(name);
        return ::sysctlbyname(
                   "machdep.cpu.brand_string",
                   name,
                   &size,
                   nullptr,
                   0) == 0 &&
            std::string_view(name).rfind("Apple M3 Ultra", 0) == 0;
    }();
    return is_m3_ultra;
#else
    return false;
#endif
}

inline bool deepseek_v41_hf_fused_kv_prepare_enabled() noexcept {
    if (const char* value = std::getenv(
            "MFQ_METAL_DSV41_FUSED_KV_PREP")) {
        const auto setting = std::string_view(value);
        return setting != "0" && setting != "false" && setting != "off";
    }
    return deepseek_v41_hf_apple_m3_ultra();
}

inline bool deepseek_v41_hf_fused_hyper_connections_enabled() noexcept {
    // Opt in explicitly: the fused BF16 reduction is numerically close but
    // can change greedy token tie-breaks after many layers.
    const char* value = std::getenv("MFQ_METAL_DSV41_FUSED_HC");
    return value != nullptr && std::string_view(value) != "0";
}

inline bool deepseek_v41_hf_exact_hc_post_enabled() noexcept {
    // HC post preserves the generic BF16 graph exactly. Keep the wider HC
    // pre fusion opt-in while using exact post-only by default.
    const char* value = std::getenv("MFQ_METAL_DSV41_EXACT_HC_POST");
    return value == nullptr ||
        (std::string_view(value) != "0" &&
         std::string_view(value) != "false" &&
         std::string_view(value) != "off");
}

inline bool deepseek_v41_hf_exact_hc_collapse_norm_enabled() noexcept {
    const char* value = std::getenv(
        "MFQ_METAL_DSV41_EXACT_HC_COLLAPSE_NORM");
    return value == nullptr ||
        (std::string_view(value) != "0" &&
         std::string_view(value) != "false" &&
         std::string_view(value) != "off");
}

inline bool deepseek_v41_hf_exact_hc_metadata_enabled() noexcept {
    const char* value = std::getenv(
        "MFQ_METAL_DSV41_EXACT_HC_METADATA");
    return value == nullptr ||
        (std::string_view(value) != "0" &&
         std::string_view(value) != "false" &&
         std::string_view(value) != "off");
}

inline int deepseek_v41_hf_resident_prefill_layer_group() noexcept {
    const char* value = std::getenv("MFQ_METAL_DSV41_PREFILL_LAYER_GROUP");
    return value == nullptr ? 1 : std::clamp(std::atoi(value), 1, 43);
}

inline std::size_t deepseek_v41_hf_engram_cache_bytes() {
    constexpr std::size_t default_cache_mib = 256;
    const char* value = std::getenv("MFQ_DEEPSEEK_V41_ENGRAM_CACHE_MIB");
    if (value == nullptr || *value == '\0') {
        return default_cache_mib * 1024u * 1024u;
    }
    char* end = nullptr;
    const auto mib = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || mib == 0 ||
        mib > std::numeric_limits<std::size_t>::max() / (1024u * 1024u)) {
        throw std::invalid_argument(
            "MFQ_DEEPSEEK_V41_ENGRAM_CACHE_MIB must be a positive integer");
    }
    return static_cast<std::size_t>(mib) * 1024u * 1024u;
}

struct DeepseekV41HfExpertCacheBudgets {
    std::size_t backbone = 0;
    std::size_t dspark = 0;
};

inline DeepseekV41HfExpertCacheBudgets
deepseek_v41_hf_split_expert_cache(
    const DeepseekFamilyConfig& config,
    std::size_t total,
    bool prefill_overlap) {
    const auto hidden = static_cast<std::size_t>(config.hidden);
    const auto intermediate = static_cast<std::size_t>(config.moe_inter);
    if (hidden == 0 || intermediate == 0 ||
        hidden > std::numeric_limits<std::size_t>::max() / intermediate) {
        throw std::invalid_argument(
            "invalid DeepSeek-V4.1 expert-cache geometry");
    }
    const auto elements = hidden * intermediate;
    // Three MXFP4 matrices: 4-bit values plus one E8M0 scale per 32 values.
    if (elements > std::numeric_limits<std::size_t>::max() / 51u) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 expert-cache slot size overflows");
    }
    const auto slot_bytes = elements * 51u / 32u;
    const auto backbone_slots = std::size_t{6} +
        (prefill_overlap
             ? 2u * static_cast<std::size_t>(config.n_experts)
             : 0u);
    const auto dspark_slots = std::min(
        static_cast<std::size_t>(config.dspark_n_experts),
        std::max<std::size_t>(
            6,
            4u * static_cast<std::size_t>(config.dspark_top_k)));
    if (slot_bytes == 0 ||
        backbone_slots > std::numeric_limits<std::size_t>::max() / slot_bytes ||
        dspark_slots > std::numeric_limits<std::size_t>::max() / slot_bytes) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 expert-cache minimum overflows");
    }
    const auto backbone_minimum = backbone_slots * slot_bytes;
    const auto dspark_minimum = dspark_slots * slot_bytes;
    if (total < backbone_minimum ||
        total - backbone_minimum < dspark_minimum) {
        throw std::invalid_argument(
            "DeepSeek-V4.1 SSD expert cache is too small for separate "
            "backbone and DSpark arenas");
    }
    return {total - dspark_minimum, dspark_minimum};
}

} // namespace mfq::metal
