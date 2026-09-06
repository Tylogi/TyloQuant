#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mfq {

inline constexpr std::string_view kMfqModelGraphAsset =
    "__mfq_asset__/model_graph.json";
inline constexpr std::string_view kMfqCanonicalTensorNamespace =
    "mfq.tensor";
inline constexpr std::string_view kMfqGridVisionInputContract =
    "grid_vision.v1";
inline constexpr std::string_view kMfqGridMropePositionPolicy =
    "grid_mrope";

struct MfqGraphTopology {
    std::int64_t text_layers = 0;
    std::int64_t vision_layers = 0;
    std::int64_t predictor_layers = 0;
};

// A component describes one composable part of the inference graph. The
// implementation and policies are stable semantic IDs resolved by each
// backend's component registry; they are not source-model class names.
struct MfqGraphComponent {
    std::string kind;
    std::string tensor_root;
    std::string implementation;
    std::string policy;
    std::string input_contract;
    std::string position_policy;
    std::unordered_map<std::string, std::string> options;
};

struct MfqModelGraph {
    std::int32_t schema_version = 0;
    std::string architecture;
    std::string graph_kind;
    std::string backbone;
    std::string tensor_namespace;
    std::int32_t tensor_naming_version = 0;
    std::vector<std::string> component_roots;
    MfqGraphTopology topology;
    std::vector<MfqGraphComponent> components;
    std::vector<std::string> capabilities;

    static MfqModelGraph from_json(std::string_view payload);

    const MfqGraphComponent* component(std::string_view kind) const noexcept;
    bool has_component(std::string_view kind) const noexcept {
        return component(kind) != nullptr;
    }
    bool has_capability(std::string_view capability) const noexcept;
};

} // namespace mfq
