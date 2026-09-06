#include "mlx_legacy_tensor_compat.h"

#include "mfq_container.h"
#include "mfq_legacy_model_graph.h"
#include "mfq_legacy_tensor_names.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

inline constexpr std::string_view kModelConfigAsset =
    "__mfq_asset__/model_config.json";

std::vector<std::string> stored_tensor_names(const MfqContainer& model) {
    std::vector<std::string> result;
    result.reserve(model.records().size());
    for (const auto& [name, _record] : model.records()) {
        if (!name.starts_with("__mfq_asset__/")) result.push_back(name);
    }
    return result;
}

std::string legacy_model_config(const MfqContainer& model) {
    const std::string asset(kModelConfigAsset);
    if (!model.contains(asset)) {
        throw std::runtime_error(
            "pre-schema MFQ artifact has no embedded model_config.json");
    }
    return model.read_text(asset);
}

} // namespace

void install_legacy_tensor_compatibility(MfqContainer& model) {
    if (model.model_graph()) return;
    if (!model.contains(std::string(kModelConfigAsset))) return;
    auto compatibility = mfq::make_legacy_tensor_aliases(
        model.header().architecture,
        legacy_model_config(model),
        stored_tensor_names(model));
    model.install_legacy_aliases(
        std::move(compatibility.canonical_to_stored),
        compatibility.layout);
}

mfq::MfqModelGraph effective_model_graph(const MfqContainer& model) {
    if (auto graph = model.model_graph()) return *graph;
    return mfq::synthesize_legacy_model_graph(
        model.header().architecture,
        legacy_model_config(model),
        [&model](std::string_view name) {
            return model.contains(std::string(name));
        });
}

} // namespace mfq::metal
