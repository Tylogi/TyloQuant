#include "mfq_model_graph.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace mfq {
namespace {

using json = nlohmann::json;

std::int64_t nonnegative_integer(
    const json& object,
    const char* key,
    std::int64_t fallback = 0) {
    const auto found = object.find(key);
    if (found == object.end()) return fallback;
    if (!found->is_number_integer()) {
        throw std::invalid_argument(
            std::string("model graph field must be an integer: ") + key);
    }
    const auto value = found->get<std::int64_t>();
    if (value < 0) {
        throw std::invalid_argument(
            std::string("model graph field must be non-negative: ") + key);
    }
    return value;
}

std::string required_identity(const json& object, const char* key) {
    const auto found = object.find(key);
    if (found == object.end() || !found->is_string()) {
        throw std::invalid_argument(
            std::string("model graph is missing string field: ") + key);
    }
    auto value = found->get<std::string>();
    if (value.empty() || value.size() > 128 ||
        std::any_of(value.begin(), value.end(), [](unsigned char character) {
            return !(character == '_' || character == '-' || character == '.' ||
                     character >= '0' && character <= '9' ||
                     character >= 'a' && character <= 'z');
        })) {
        throw std::invalid_argument(
            std::string("model graph field is not a stable identity: ") + key);
    }
    return value;
}

std::string optional_identity(const json& object, const char* key) {
    const auto found = object.find(key);
    if (found == object.end() || found->is_null()) return {};
    return required_identity(object, key);
}

MfqGraphComponent parse_component(const json& value) {
    if (!value.is_object()) {
        throw std::invalid_argument("model graph component must be an object");
    }
    MfqGraphComponent result;
    result.kind = required_identity(value, "kind");
    result.tensor_root = required_identity(value, "tensor_root");
    result.implementation = required_identity(value, "implementation");
    result.policy = optional_identity(value, "policy");
    result.input_contract = optional_identity(value, "input_contract");
    result.position_policy = optional_identity(value, "position_policy");
    const auto options = value.find("options");
    if (options != value.end()) {
        if (!options->is_object()) {
            throw std::invalid_argument(
                "model graph component options must be an object");
        }
        for (auto item = options->begin(); item != options->end(); ++item) {
            if (!item.value().is_string()) {
                throw std::invalid_argument(
                    "model graph component option must be a string");
            }
            result.options.emplace(item.key(), item.value().get<std::string>());
        }
    }
    return result;
}

} // namespace

MfqModelGraph MfqModelGraph::from_json(std::string_view payload) {
    json root;
    try {
        root = json::parse(payload);
    } catch (const json::exception& error) {
        throw std::invalid_argument(
            std::string("invalid MFQ model graph JSON: ") + error.what());
    }
    if (!root.is_object()) {
        throw std::invalid_argument("MFQ model graph must be an object");
    }

    MfqModelGraph result;
    const auto version = nonnegative_integer(root, "schema_version");
    if (version <= 0 ||
        version > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("unsupported MFQ model graph schema version");
    }
    result.schema_version = static_cast<std::int32_t>(version);
    result.architecture = required_identity(root, "architecture");

    const auto naming = root.find("canonical_naming");
    if (naming == root.end() || !naming->is_object()) {
        throw std::invalid_argument(
            "MFQ model graph has no canonical_naming contract");
    }
    result.tensor_namespace = required_identity(*naming, "namespace");
    if (result.tensor_namespace != kMfqCanonicalTensorNamespace) {
        throw std::invalid_argument(
            "MFQ model graph uses an unsupported tensor namespace");
    }
    const auto naming_version = nonnegative_integer(*naming, "version");
    if (naming_version <= 0 ||
        naming_version > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(
            "MFQ model graph has an invalid tensor naming version");
    }
    result.tensor_naming_version =
        static_cast<std::int32_t>(naming_version);
    const auto roots = naming->find("component_roots");
    if (roots == naming->end() || !roots->is_array() || roots->empty()) {
        throw std::invalid_argument(
            "MFQ model graph must declare canonical component_roots");
    }
    std::unordered_set<std::string> declared_roots;
    for (const auto& root : *roots) {
        if (!root.is_string()) {
            throw std::invalid_argument(
                "MFQ model graph component root must be a string");
        }
        json wrapper = {{"root", root}};
        auto identity = required_identity(wrapper, "root");
        if (!declared_roots.insert(identity).second) {
            throw std::invalid_argument(
                "MFQ model graph component root is duplicated: " + identity);
        }
        result.component_roots.push_back(std::move(identity));
    }

    const auto topology = root.find("topology");
    if (topology == root.end() || !topology->is_object()) {
        throw std::invalid_argument("MFQ model graph has no topology");
    }
    result.topology = {
        nonnegative_integer(*topology, "text_layers"),
        nonnegative_integer(*topology, "vision_layers"),
        nonnegative_integer(*topology, "predictor_layers"),
    };

    const auto graph = root.find("graph");
    const json* components = nullptr;
    if (graph != root.end()) {
        if (!graph->is_object()) {
            throw std::invalid_argument("MFQ model graph.graph must be an object");
        }
        result.graph_kind = required_identity(*graph, "kind");
        result.backbone = required_identity(*graph, "backbone");
        const auto found = graph->find("components");
        if (found != graph->end()) components = &*found;
    }
    if (result.graph_kind.empty() || result.backbone.empty()) {
        throw std::invalid_argument(
            "MFQ model graph must declare graph.kind and graph.backbone");
    }
    if (components == nullptr) {
        const auto found = root.find("components");
        if (found != root.end()) components = &*found;
    }
    if (components == nullptr || !components->is_array() || components->empty()) {
        throw std::invalid_argument(
            "MFQ model graph must declare a non-empty component list");
    }
    std::unordered_set<std::string> component_kinds;
    for (const auto& component : *components) {
        auto parsed = parse_component(component);
        if (!component_kinds.insert(parsed.kind).second) {
            throw std::invalid_argument(
                "MFQ model graph declares a component kind twice: " +
                parsed.kind);
        }
        if (declared_roots.find(parsed.tensor_root) == declared_roots.end()) {
            throw std::invalid_argument(
                "MFQ model graph component uses an undeclared tensor root: " +
                parsed.tensor_root);
        }
        result.components.push_back(std::move(parsed));
    }
    for (const auto& root : declared_roots) {
        if (std::none_of(
                result.components.begin(), result.components.end(),
                [&](const MfqGraphComponent& component) {
                    return component.tensor_root == root;
                })) {
            throw std::invalid_argument(
                "MFQ model graph declares an unused tensor root: " + root);
        }
    }
    if (!result.has_component("text")) {
        throw std::invalid_argument(
            "MFQ model graph must declare exactly one text component");
    }
    if (result.component("text")->implementation != result.backbone) {
        throw std::invalid_argument(
            "MFQ model graph text implementation/backbone disagree");
    }
    if (result.topology.text_layers <= 0) {
        throw std::invalid_argument(
            "MFQ model graph text topology must be positive");
    }
    if (result.has_component("vision") != (result.topology.vision_layers > 0)) {
        throw std::invalid_argument(
            "MFQ model graph Vision component/topology disagree");
    }
    if (result.has_component("predictor") !=
        (result.topology.predictor_layers > 0)) {
        throw std::invalid_argument(
            "MFQ model graph predictor component/topology disagree");
    }

    const auto capabilities = root.find("capabilities");
    if (capabilities != root.end()) {
        if (!capabilities->is_array()) {
            throw std::invalid_argument(
                "MFQ model graph capabilities must be an array");
        }
        std::unordered_set<std::string> seen;
        for (const auto& capability : *capabilities) {
            if (!capability.is_string()) {
                throw std::invalid_argument(
                    "MFQ model graph capability must be a string");
            }
            const auto identity = capability.get<std::string>();
            if (!seen.insert(identity).second) {
                throw std::invalid_argument(
                    "MFQ model graph capability is duplicated: " + identity);
            }
            result.capabilities.push_back(identity);
        }
    }
    return result;
}

const MfqGraphComponent* MfqModelGraph::component(
        std::string_view kind) const noexcept {
    const auto found = std::find_if(
        components.begin(), components.end(),
        [kind](const MfqGraphComponent& value) {
            return value.kind == kind;
        });
    return found == components.end() ? nullptr : &*found;
}

bool MfqModelGraph::has_capability(std::string_view capability) const noexcept {
    return std::find(capabilities.begin(), capabilities.end(), capability) !=
        capabilities.end();
}

} // namespace mfq
