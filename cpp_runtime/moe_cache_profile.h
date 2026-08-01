#pragma once

#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mfq {

struct MoeCacheLayerProfile {
    std::vector<int> ranking;
    std::unordered_map<int, double> frequencies;
};

struct MoeCacheProfile {
    std::string path;
    std::unordered_map<int, MoeCacheLayerProfile> layers;
};

struct MoeProfileCandidate {
    int layer = -1;
    int expert = -1;
    bool has_frequency = false;
    double frequency = 0.0;
    int rank = std::numeric_limits<int>::max();
};

inline int parse_moe_profile_id(
        const std::string & text,
        const char * field) {
    if (text.empty()) {
        throw std::runtime_error(
            std::string("empty ") + field + " ID");
    }
    size_t consumed = 0;
    int64_t value = -1;
    try {
        value = std::stoll(text, &consumed, 10);
    } catch (const std::exception &) {
        throw std::runtime_error(
            std::string("invalid ") + field + " ID: " + text);
    }
    if (consumed != text.size() || value < 0 ||
            value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            std::string("invalid ") + field + " ID: " + text);
    }
    return static_cast<int>(value);
}

inline void validate_moe_profile_fields(
        const nlohmann::json & object,
        const std::unordered_set<std::string> & allowed,
        const char * context) {
    if (!object.is_object()) {
        throw std::runtime_error(
            std::string(context) + " must be a JSON object");
    }
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (!allowed.count(it.key())) {
            throw std::runtime_error(
                std::string("unknown ") + context +
                " field: " + it.key());
        }
    }
}

inline MoeCacheProfile load_moe_cache_profile(
        const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "cannot open MoE cache profile: " + path);
    }
    nlohmann::json root;
    try {
        input >> root;
    } catch (const std::exception & error) {
        throw std::runtime_error(
            "cannot parse MoE cache profile " + path +
            ": " + error.what());
    }
    validate_moe_profile_fields(
        root, {"version", "metadata", "layers"},
        "MoE cache profile");
    if (!root.contains("version") ||
            !root["version"].is_number_integer() ||
            root["version"].get<int>() != 1) {
        throw std::runtime_error(
            "MoE cache profile version must be 1");
    }
    if (!root.contains("layers") || !root["layers"].is_object()) {
        throw std::runtime_error(
            "MoE cache profile layers must be an object");
    }
    if (root.contains("metadata") &&
            !root["metadata"].is_object()) {
        throw std::runtime_error(
            "MoE cache profile metadata must be an object");
    }

    MoeCacheProfile profile;
    profile.path = path;
    for (auto layer_it = root["layers"].begin();
         layer_it != root["layers"].end();
         ++layer_it) {
        const int layer =
            parse_moe_profile_id(layer_it.key(), "layer");
        validate_moe_profile_fields(
            layer_it.value(),
            {"ranking", "frequencies", "metadata"},
            "MoE cache layer");
        if (layer_it.value().contains("metadata") &&
                !layer_it.value()["metadata"].is_object()) {
            throw std::runtime_error(
                "MoE cache layer metadata must be an object");
        }
        MoeCacheLayerProfile parsed;
        if (layer_it.value().contains("ranking")) {
            const auto & ranking = layer_it.value()["ranking"];
            if (!ranking.is_array()) {
                throw std::runtime_error(
                    "MoE cache ranking must be an array");
            }
            std::unordered_set<int> seen;
            for (const auto & value : ranking) {
                if (!value.is_number_integer()) {
                    throw std::runtime_error(
                        "MoE cache ranking entries must be integers");
                }
                const int64_t expert64 = value.get<int64_t>();
                if (expert64 < 0 ||
                        expert64 > std::numeric_limits<int>::max()) {
                    throw std::runtime_error(
                        "MoE cache ranking expert is invalid");
                }
                const int expert = static_cast<int>(expert64);
                if (!seen.insert(expert).second) {
                    throw std::runtime_error(
                        "MoE cache ranking contains a duplicate expert");
                }
                parsed.ranking.push_back(expert);
            }
        }
        if (layer_it.value().contains("frequencies")) {
            const auto & frequencies =
                layer_it.value()["frequencies"];
            if (!frequencies.is_object()) {
                throw std::runtime_error(
                    "MoE cache frequencies must be an object");
            }
            for (auto frequency_it = frequencies.begin();
                 frequency_it != frequencies.end();
                 ++frequency_it) {
                const int expert = parse_moe_profile_id(
                    frequency_it.key(), "expert");
                if (!frequency_it.value().is_number()) {
                    throw std::runtime_error(
                        "MoE cache frequency must be numeric");
                }
                const double frequency =
                    frequency_it.value().get<double>();
                if (!std::isfinite(frequency) || frequency < 0.0) {
                    throw std::runtime_error(
                        "MoE cache frequency must be finite and non-negative");
                }
                parsed.frequencies.emplace(expert, frequency);
            }
        }
        profile.layers.emplace(layer, std::move(parsed));
    }
    return profile;
}

inline void validate_moe_cache_profile(
        const MoeCacheProfile & profile,
        const std::unordered_map<int, int> & layer_experts) {
    for (const auto & item : profile.layers) {
        const auto found = layer_experts.find(item.first);
        if (found == layer_experts.end()) {
            throw std::runtime_error(
                "MoE cache profile references an unknown layer " +
                std::to_string(item.first));
        }
        const int experts = found->second;
        if (experts <= 0) {
            throw std::runtime_error(
                "MoE cache profile layer has no routed experts");
        }
        for (int expert : item.second.ranking) {
            if (expert >= experts) {
                throw std::runtime_error(
                    "MoE cache profile expert is out of range");
            }
        }
        for (const auto & frequency : item.second.frequencies) {
            if (frequency.first >= experts) {
                throw std::runtime_error(
                    "MoE cache profile expert is out of range");
            }
        }
    }
}

inline std::vector<MoeProfileCandidate>
order_moe_profile_candidates(
        const MoeCacheProfile & profile,
        bool cold_to_hot) {
    std::vector<MoeProfileCandidate> result;
    for (const auto & layer_item : profile.layers) {
        const int layer = layer_item.first;
        const auto & layer_profile = layer_item.second;
        std::unordered_map<int, int> ranks;
        for (int rank = 0;
             rank < static_cast<int>(layer_profile.ranking.size());
             ++rank) {
            ranks.emplace(
                layer_profile.ranking[static_cast<size_t>(rank)],
                rank);
        }
        for (const auto & frequency : layer_profile.frequencies) {
            const auto rank = ranks.find(frequency.first);
            result.push_back({
                layer,
                frequency.first,
                true,
                frequency.second,
                rank == ranks.end()
                    ? std::numeric_limits<int>::max()
                    : rank->second,
            });
        }
        for (int rank = 0;
             rank < static_cast<int>(layer_profile.ranking.size());
             ++rank) {
            const int expert =
                layer_profile.ranking[static_cast<size_t>(rank)];
            if (layer_profile.frequencies.count(expert)) continue;
            result.push_back({
                layer,
                expert,
                false,
                0.0,
                rank,
            });
        }
    }
    std::sort(
        result.begin(), result.end(),
        [](const MoeProfileCandidate & left,
           const MoeProfileCandidate & right) {
            if (left.has_frequency != right.has_frequency) {
                return left.has_frequency > right.has_frequency;
            }
            if (left.has_frequency &&
                    left.frequency != right.frequency) {
                return left.frequency > right.frequency;
            }
            if (left.rank != right.rank) {
                return left.rank < right.rank;
            }
            if (left.layer != right.layer) {
                return left.layer < right.layer;
            }
            return left.expert < right.expert;
        });
    if (cold_to_hot) std::reverse(result.begin(), result.end());
    return result;
}

}  // namespace mfq
