#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mfq {

// Numeric/layout differences that cannot be represented by a no-copy name
// alias. New schema-v1 artifacts always use canonical semantics; these flags
// exist only while pre-schema files remain readable.
struct MfqLegacyTensorLayout {
    double norm_weight_offset = 1.0;
    bool linear_attention_a_is_log = true;
    bool qwen_gdn_gguf_layout = false;
};

struct MfqLegacyTensorAliases {
    std::unordered_map<std::string, std::string> canonical_to_stored;
    MfqLegacyTensorLayout layout;
};

// The sole backend-neutral C++ name compatibility boundary. Runtime loaders
// ask only for canonical names; this view aliases them to records from older
// MFQ files without copying tensor blobs.
MfqLegacyTensorAliases make_legacy_tensor_aliases(
    std::string_view artifact_architecture,
    std::string_view model_config_json,
    const std::vector<std::string>& stored_names);

} // namespace mfq
