#include "mfq/model_source.h"

#include "mfq/hf_model_source.h"
#include "mfq/mfq_model_source.h"
#include "mfq/tq_fixture_source.h"
#include "mfq_legacy_model_graph.h"

#include <cstdlib>

namespace mfq {

std::shared_ptr<const ModelSource> open_model_source(
        const std::filesystem::path& path) {
    for (const char* environment :
         {"MFQ_TENSOR_OVERLAY", "MFQ_EXPERT_OVERLAY"}) {
        const char* overlay = std::getenv(environment);
        if (overlay != nullptr && overlay[0] != '\0') {
            throw std::runtime_error(
                std::string(environment) +
                " is unsupported by the canonical ModelSource path");
        }
    }

    const auto filename=path.filename().u8string();
    const std::string fixture_suffix=".tq-fixture.json";
    if(filename.size()>=fixture_suffix.size() &&
       filename.compare(filename.size()-fixture_suffix.size(),fixture_suffix.size(),fixture_suffix)==0)
        return open_tq_fixture_source(path);
    std::error_code error;
    if (std::filesystem::is_directory(path, error) && !error) {
        return std::make_shared<HfModelSource>(path);
    }
    return std::make_shared<MfqModelSource>(path);
}

std::string ModelSource::model_config_json() const {
    if (!has_asset(kModelConfigAsset)) {
        throw std::runtime_error("model source has no model config");
    }
    const auto bytes = read_asset(kModelConfigAsset);
    return {
        reinterpret_cast<const char*>(bytes.data()),
        bytes.size(),
    };
}

ModelGraph ModelSource::resolved_model_graph() const {
    if (auto graph = model_graph()) return *graph;
    const auto config = model_config_json();
    return synthesize_legacy_model_graph(
        architecture(), config,
        [&](std::string_view name) { return find_tensor(name) != nullptr; });
}

} // namespace mfq
