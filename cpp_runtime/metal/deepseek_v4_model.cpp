#include "deepseek_v4_model.h"

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace mfq::metal {
namespace {

using json = nlohmann::json;

json parse_json(
    std::string_view payload,
    const char* description) {
    try {
        auto result = json::parse(payload.begin(), payload.end());
        if (!result.is_object()) {
            throw std::runtime_error(
                std::string(description) + " must be a JSON object");
        }
        return result;
    } catch (const json::exception& error) {
        throw std::runtime_error(
            std::string("invalid ") + description + ": " +
            error.what());
    }
}

json config_object(const json& root) {
    json value = root;
    auto embedded = value.find("tpq_manifest");
    if (embedded == value.end()) {
        embedded = value.find("cccp_manifest");
    }
    if (embedded != value.end()) {
        if (embedded->is_string()) {
            value = parse_json(
                embedded->get<std::string>(),
                "DeepSeek-V4 TPQ manifest");
        } else if (embedded->is_object()) {
            value = *embedded;
        } else {
            throw std::runtime_error(
                "DeepSeek-V4 tpq_manifest must be an object");
        }
    }
    const auto manifest_config = value.find("config");
    if (manifest_config != value.end()) {
        if (!manifest_config->is_object()) {
            throw std::runtime_error(
                "DeepSeek-V4 manifest config must be an object");
        }
        value = *manifest_config;
    }
    const auto text_config = value.find("text_config");
    if (text_config != value.end() &&
        !value.contains("n_layers") &&
        !value.contains("num_hidden_layers")) {
        if (!text_config->is_object()) {
            throw std::runtime_error(
                "DeepSeek-V4 text_config must be an object");
        }
        value = *text_config;
    }
    return value;
}

std::string alias_text(
    std::initializer_list<const char*> keys) {
    std::string result;
    for (const auto* key : keys) {
        if (!result.empty()) {
            result += "/";
        }
        result += key;
    }
    return result;
}

std::int64_t required_integer(
    const json& object,
    std::initializer_list<const char*> keys) {
    std::optional<std::int64_t> result;
    for (const auto* key : keys) {
        const auto found = object.find(key);
        if (found == object.end() || found->is_null()) {
            continue;
        }
        if (!found->is_number_integer()) {
            throw std::runtime_error(
                "DeepSeek-V4 config " + alias_text(keys) +
                " must be an integer");
        }
        const auto value = found->get<std::int64_t>();
        if (result && *result != value) {
            throw std::runtime_error(
                "DeepSeek-V4 config has conflicting aliases " +
                alias_text(keys));
        }
        result = value;
    }
    if (!result) {
        throw std::runtime_error(
            "DeepSeek-V4 config requires integer " +
            alias_text(keys));
    }
    return *result;
}

std::int64_t optional_integer(
    const json& object,
    std::initializer_list<const char*> keys,
    std::int64_t default_value) {
    std::optional<std::int64_t> result;
    for (const auto* key : keys) {
        const auto found = object.find(key);
        if (found == object.end() || found->is_null()) {
            continue;
        }
        if (!found->is_number_integer()) {
            throw std::runtime_error(
                "DeepSeek-V4 config " + alias_text(keys) +
                " must be an integer");
        }
        const auto value = found->get<std::int64_t>();
        if (result && *result != value) {
            throw std::runtime_error(
                "DeepSeek-V4 config has conflicting aliases " +
                alias_text(keys));
        }
        result = value;
    }
    return result.value_or(default_value);
}

double optional_number(
    const json& object,
    std::initializer_list<const char*> keys,
    double default_value) {
    std::optional<double> result;
    for (const auto* key : keys) {
        const auto found = object.find(key);
        if (found == object.end() || found->is_null()) {
            continue;
        }
        if (!found->is_number()) {
            throw std::runtime_error(
                "DeepSeek-V4 config " + alias_text(keys) +
                " must be a number");
        }
        const auto value = found->get<double>();
        if (result && *result != value) {
            throw std::runtime_error(
                "DeepSeek-V4 config has conflicting aliases " +
                alias_text(keys));
        }
        result = value;
    }
    return result.value_or(default_value);
}

bool optional_boolean(
    const json& object,
    std::initializer_list<const char*> keys,
    bool default_value) {
    std::optional<bool> result;
    for (const auto* key : keys) {
        const auto found = object.find(key);
        if (found == object.end() || found->is_null()) {
            continue;
        }
        if (!found->is_boolean()) {
            throw std::runtime_error(
                "DeepSeek-V4 config " + alias_text(keys) +
                " must be boolean");
        }
        const auto value = found->get<bool>();
        if (result && *result != value) {
            throw std::runtime_error(
                "DeepSeek-V4 config has conflicting aliases " +
                alias_text(keys));
        }
        result = value;
    }
    return result.value_or(default_value);
}

std::string optional_string(
    const json& object,
    std::initializer_list<const char*> keys,
    std::string default_value = {}) {
    std::optional<std::string> result;
    for (const auto* key : keys) {
        const auto found = object.find(key);
        if (found == object.end() || found->is_null()) {
            continue;
        }
        if (!found->is_string()) {
            throw std::runtime_error(
                "DeepSeek-V4 config " + alias_text(keys) +
                " must be a string");
        }
        auto value = found->get<std::string>();
        if (result && *result != value) {
            throw std::runtime_error(
                "DeepSeek-V4 config has conflicting aliases " +
                alias_text(keys));
        }
        result = std::move(value);
    }
    return result.value_or(std::move(default_value));
}

std::vector<std::int64_t> integer_array(
    const json& object,
    const char* key) {
    const auto found = object.find(key);
    if (found == object.end() || found->is_null()) {
        return {};
    }
    if (!found->is_array()) {
        throw std::runtime_error(
            std::string("DeepSeek-V4 config ") + key +
            " must be an array");
    }
    std::vector<std::int64_t> result;
    result.reserve(found->size());
    for (const auto& item : *found) {
        if (!item.is_number_integer()) {
            throw std::runtime_error(
                std::string("DeepSeek-V4 config ") + key +
                " entries must be integers");
        }
        result.push_back(item.get<std::int64_t>());
    }
    return result;
}

std::vector<std::int64_t> eos_tokens(const json& object) {
    const auto found = object.find("eos_token_id");
    if (found == object.end() || found->is_null()) {
        return {};
    }
    if (found->is_number_integer()) {
        return {found->get<std::int64_t>()};
    }
    if (!found->is_array()) {
        throw std::runtime_error(
            "DeepSeek-V4 eos_token_id must be an integer or array");
    }
    std::vector<std::int64_t> result;
    result.reserve(found->size());
    for (const auto& item : *found) {
        if (!item.is_number_integer()) {
            throw std::runtime_error(
                "DeepSeek-V4 eos_token_id entries must be integers");
        }
        result.push_back(item.get<std::int64_t>());
    }
    return result;
}

DeepseekV4RopeScaling rope_scaling(const json& object) {
    DeepseekV4RopeScaling result;
    const auto found = object.find("rope_scaling");
    if (found == object.end() || found->is_null()) {
        return result;
    }
    if (!found->is_object()) {
        throw std::runtime_error(
            "DeepSeek-V4 rope_scaling must be an object");
    }
    result.enabled = !found->empty();
    result.type = optional_string(
        *found,
        {"rope_type", "type"});
    result.factor =
        optional_number(*found, {"factor"}, 1.0);
    result.beta_fast =
        optional_number(*found, {"beta_fast"}, 32.0);
    result.beta_slow =
        optional_number(*found, {"beta_slow"}, 1.0);
    result.original_max_position_embeddings =
        optional_integer(
            *found,
            {"original_max_position_embeddings"},
            0);
    return result;
}

bool valid_model_type(std::string_view value) {
    return value == "deepseek_v4" ||
        value == "deepseek-v4";
}

std::int64_t checked_product(
    std::int64_t left,
    std::int64_t right,
    const char* description) {
    if (left <= 0 || right <= 0 ||
        left > std::numeric_limits<std::int64_t>::max() / right) {
        throw std::runtime_error(
            std::string("DeepSeek-V4 dimension overflows: ") +
            description);
    }
    const auto result = left * right;
    if (result >
        std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error(
            std::string("DeepSeek-V4 dimension exceeds MLX range: ") +
            description);
    }
    return result;
}

std::string shape_text(
    const std::vector<std::int64_t>& shape) {
    std::string result = "[";
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            result += ",";
        }
        result += std::to_string(shape[index]);
    }
    result += "]";
    return result;
}

class PrefixCursor {
public:
    explicit PrefixCursor(const std::vector<std::uint8_t>& bytes)
        : bytes_(bytes) {}

    template <typename T>
    T scalar(const char* description) {
        require(sizeof(T), description);
        T value{};
        std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    std::string bytes(
        std::size_t count,
        const char* description) {
        require(count, description);
        std::string result(
            reinterpret_cast<const char*>(
                bytes_.data() + offset_),
            count);
        offset_ += count;
        return result;
    }

    void skip(
        std::size_t count,
        const char* description) {
        require(count, description);
        offset_ += count;
    }

    std::size_t offset() const noexcept {
        return offset_;
    }

private:
    void require(
        std::size_t count,
        const char* description) const {
        if (offset_ > bytes_.size() ||
            count > bytes_.size() - offset_) {
            throw std::runtime_error(
                std::string("truncated DeepSeek-V4 tensor ") +
                description);
        }
    }

    const std::vector<std::uint8_t>& bytes_;
    std::size_t offset_ = 0;
};

std::vector<std::uint8_t> read_record_prefix(
    const MfqRecord& record,
    std::size_t maximum = 256) {
    const auto count = static_cast<std::size_t>(
        std::min<std::uint64_t>(record.nbytes, maximum));
    std::vector<std::uint8_t> result(count);
    std::ifstream stream(record.source_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "cannot open DeepSeek-V4 tensor source: " +
            record.source_path.string());
    }
    stream.seekg(static_cast<std::streamoff>(record.offset));
    stream.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(result.size()));
    if (!stream) {
        throw std::runtime_error(
            "cannot read DeepSeek-V4 tensor header: " +
            record.name);
    }
    return result;
}

std::vector<std::int64_t> read_shape(
    PrefixCursor& cursor,
    std::uint32_t dimensions,
    const std::string& name) {
    if (dimensions == 0 || dimensions > 8) {
        throw std::runtime_error(
            "invalid DeepSeek-V4 tensor rank: " + name);
    }
    std::vector<std::int64_t> result;
    result.reserve(dimensions);
    for (std::uint32_t index = 0; index < dimensions; ++index) {
        const auto value =
            cursor.scalar<std::int64_t>("shape");
        if (value <= 0 ||
            value > std::numeric_limits<std::int32_t>::max()) {
            throw std::runtime_error(
                "invalid DeepSeek-V4 tensor shape: " + name);
        }
        result.push_back(value);
    }
    return result;
}

std::uint64_t checked_elements(
    const std::vector<std::int64_t>& shape,
    const std::string& name) {
    std::uint64_t result = 1;
    for (const auto value : shape) {
        const auto current = static_cast<std::uint64_t>(value);
        if (result >
            std::numeric_limits<std::uint64_t>::max() / current) {
            throw std::runtime_error(
                "DeepSeek-V4 tensor shape overflows: " + name);
        }
        result *= current;
    }
    return result;
}

std::int64_t shape_product_except_axis(
    const std::vector<std::int64_t>& shape,
    std::int32_t axis,
    const std::string& name) {
    if (axis < 0 ||
        static_cast<std::size_t>(axis) >= shape.size()) {
        throw std::runtime_error(
            "invalid DeepSeek-V4 tensor axis: " + name);
    }
    std::int64_t result = 1;
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index == static_cast<std::size_t>(axis)) {
            continue;
        }
        if (result >
            std::numeric_limits<std::int64_t>::max() /
                shape[index]) {
            throw std::runtime_error(
                "DeepSeek-V4 tensor shape overflows: " + name);
        }
        result *= shape[index];
    }
    return result;
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right,
    const std::string& name) {
    if (left >
        std::numeric_limits<std::uint64_t>::max() - right) {
        throw std::runtime_error(
            "DeepSeek-V4 tensor length overflows: " + name);
    }
    return left + right;
}

std::uint64_t checked_multiply(
    std::uint64_t left,
    std::uint64_t right,
    const std::string& name) {
    if (left != 0 &&
        right >
            std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::runtime_error(
            "DeepSeek-V4 tensor length overflows: " + name);
    }
    return left * right;
}

bool is_dense_float_dtype(std::string_view dtype) {
    return dtype == "BF16" || dtype == "F16" || dtype == "F32";
}

bool is_dense_integer_dtype(std::string_view dtype) {
    return dtype == "I32" || dtype == "I64";
}

bool is_mx_dtype(std::string_view dtype) {
    return dtype == "MXFP4" || dtype == "MXFP8";
}

bool is_nint_dtype(std::string_view dtype) {
    if (dtype == "NINT") {
        return true;
    }
    if (dtype.size() != 5 ||
        dtype.substr(0, 4) != "NINT") {
        return false;
    }
    return dtype[4] >= '1' && dtype[4] <= '8';
}

bool is_cccp_pq_dtype(std::string_view dtype) {
    return dtype == "TPQ-X" ||
        dtype == "TPQ-W" ||
        dtype == "TPQ-V" ||
        dtype == "TPQ-VV" ||
        dtype == "CCCP-X" ||
        dtype == "CCCP-W" ||
        dtype == "CCCP-V" ||
        dtype == "CCCP-VV";
}

bool is_tpq_int4_dtype(std::string_view dtype) {
    return dtype == "TPQ-I4G64" ||
        dtype == "CCCP-I4G64";
}

bool is_tpq_tier(
    std::string_view dtype,
    std::string_view tier) {
    return dtype == "TPQ-" + std::string(tier) ||
        dtype == "CCCP-" + std::string(tier);
}

bool is_linear_dtype(std::string_view dtype) {
    return is_dense_float_dtype(dtype) ||
        is_mx_dtype(dtype) ||
        is_nint_dtype(dtype) ||
        dtype == "NINT8-0" ||
        is_tpq_int4_dtype(dtype) ||
        is_cccp_pq_dtype(dtype);
}

bool is_embedding_dtype(std::string_view dtype) {
    return is_dense_float_dtype(dtype) ||
        is_mx_dtype(dtype) ||
        is_nint_dtype(dtype) ||
        dtype == "NINT8-0" ||
        is_tpq_int4_dtype(dtype);
}

void add_binding(
    std::vector<DeepseekV4TensorBinding>& result,
    std::string name,
    std::vector<std::int64_t> shape,
    DeepseekV4TensorKind kind) {
    result.push_back(
        {std::move(name), std::move(shape), kind});
}

} // namespace

DeepseekV4Config DeepseekV4Config::from_json(
    std::string_view payload) {
    const auto root =
        parse_json(payload, "DeepSeek-V4 config JSON");
    const auto object = config_object(root);

    DeepseekV4Config config;
    const auto outer_model_type =
        optional_string(root, {"model_type"});
    const auto inner_model_type =
        optional_string(object, {"model_type"});
    if ((!outer_model_type.empty() &&
         !valid_model_type(outer_model_type)) ||
        (!inner_model_type.empty() &&
         !valid_model_type(inner_model_type))) {
        throw std::runtime_error(
            "unsupported DeepSeek-V4 model_type: " +
            (!inner_model_type.empty()
                 ? inner_model_type
                 : outer_model_type));
    }
    config.model_type = "deepseek_v4";
    config.n_layers = required_integer(
        object,
        {"n_layers", "num_hidden_layers"});
    config.hidden = required_integer(
        object,
        {"hidden", "hidden_size"});
    config.n_experts = required_integer(
        object,
        {"n_experts", "n_routed_experts", "num_experts"});
    config.top_k = required_integer(
        object,
        {"top_k", "num_experts_per_tok", "top_k_experts"});
    config.moe_inter = required_integer(
        object,
        {"moe_inter", "moe_intermediate_size"});
    config.n_shared = optional_integer(
        object,
        {"n_shared", "n_shared_experts"},
        1);
    config.n_heads = required_integer(
        object,
        {"n_heads", "num_attention_heads"});
    config.head_dim =
        optional_integer(object, {"head_dim"}, 512);
    config.q_lora_rank =
        required_integer(object, {"q_lora_rank"});
    config.o_lora_rank =
        required_integer(object, {"o_lora_rank"});
    config.o_groups =
        optional_integer(object, {"o_groups"}, 1);
    config.kv_dim =
        optional_integer(object, {"kv_dim"}, config.head_dim);
    config.qk_rope_head_dim =
        required_integer(object, {"qk_rope_head_dim"});
    config.n_kv_heads = optional_integer(
        object,
        {"n_kv_heads", "num_key_value_heads"},
        1);
    config.vocab = required_integer(
        object,
        {"vocab", "vocab_size"});
    config.rms_eps = optional_number(
        object,
        {"rms_eps", "rms_norm_eps"},
        1e-6);
    config.scoring_func = optional_string(
        object,
        {"scoring_func"},
        "sqrtsoftplus");
    config.norm_topk_prob = optional_boolean(
        object,
        {"norm_topk_prob"},
        true);
    config.routed_scaling = optional_number(
        object,
        {"routed_scaling", "routed_scaling_factor"},
        1.0);
    config.swiglu_limit =
        optional_number(object, {"swiglu_limit"}, 0.0);
    config.n_hash_layers = optional_integer(
        object,
        {"n_hash_layers", "num_hash_layers"},
        0);
    config.sliding_window =
        optional_integer(object, {"sliding_window"}, 128);
    config.rope_theta =
        optional_number(object, {"rope_theta"}, 10'000.0);
    config.rope_scaling =
        ::mfq::metal::rope_scaling(object);
    config.eos_token_id = eos_tokens(object);
    config.index_n_heads =
        optional_integer(object, {"index_n_heads"}, 64);
    config.index_head_dim =
        optional_integer(object, {"index_head_dim"}, 128);
    config.index_topk =
        optional_integer(object, {"index_topk"}, 512);
    config.max_position_embeddings = optional_integer(
        object,
        {"max_position_embeddings"},
        1'048'576);
    config.hc_mult =
        optional_integer(object, {"hc_mult"}, 4);
    config.hc_eps =
        optional_number(object, {"hc_eps"}, 1e-6);
    config.hc_sinkhorn_iters =
        optional_integer(object, {"hc_sinkhorn_iters"}, 20);
    config.compress_rope_theta = optional_number(
        object,
        {"compress_rope_theta"},
        160'000.0);
    config.compress_ratios =
        integer_array(object, "compress_ratios");
    if (config.compress_ratios.empty()) {
        config.compress_ratios.assign(
            static_cast<std::size_t>(config.n_layers),
            0);
    } else if (config.compress_ratios.size() >
               static_cast<std::size_t>(config.n_layers)) {
        // Released V4F configs may append DSpark/MTP layer ratios after the
        // base decoder.  The native causal-LM runtime currently owns only
        // num_hidden_layers base layers, so ignore those speculative tails.
        config.compress_ratios.resize(
            static_cast<std::size_t>(config.n_layers));
    }
    config.validate();
    return config;
}

DeepseekV4Config DeepseekV4Config::from_mfq(
    const MfqContainer& model) {
    if (model.header().architecture ==
            "deepseek_v4-ew-mfq") {
        constexpr const char* model_config_asset =
            "__mfq_asset__/model_config.json";
        if (!model.contains(model_config_asset)) {
            throw std::runtime_error(
                "DeepSeek-V4 EW MFQ has no embedded model config asset");
        }
        return from_json(model.read_text(model_config_asset));
    }
    if (model.header().architecture !=
            "deepseek_v4-tpq-mfq" &&
        model.header().architecture !=
            "deepseek_v4-cccp-mfq") {
        throw std::runtime_error(
            "DeepSeek-V4 C++ loading requires architecture "
            "deepseek_v4-tpq-mfq (legacy "
            "deepseek_v4-cccp-mfq is also accepted), received: " +
            model.header().architecture);
    }
    const auto source =
        model.header().extra_json.find("source_format");
    auto manifest =
        model.header().extra_json.find("tpq_manifest");
    if (manifest == model.header().extra_json.end()) {
        manifest =
            model.header().extra_json.find("cccp_manifest");
    }
    if (source == model.header().extra_json.end() ||
        manifest == model.header().extra_json.end()) {
        throw std::runtime_error(
            "DeepSeek-V4 C++ loading requires native TPQ metadata");
    }

    json source_value;
    try {
        source_value = json::parse(source->second);
    } catch (const json::exception& error) {
        throw std::runtime_error(
            std::string("invalid DeepSeek-V4 source_format metadata: ") +
            error.what());
    }
    if (!source_value.is_string() ||
        source_value.get<std::string>() != "cccp-1") {
        throw std::runtime_error(
            "DeepSeek-V4 C++ loading requires source_format=cccp-1");
    }

    const auto manifest_value = parse_json(
        manifest->second,
        "DeepSeek-V4 CCCP manifest");
    const auto format = manifest_value.find("format");
    if (format != manifest_value.end() &&
        (!format->is_string() ||
         format->get<std::string>() != "cccp-1")) {
        throw std::runtime_error(
            "unsupported DeepSeek-V4 CCCP manifest format");
    }
    const auto config = manifest_value.find("config");
    if (config == manifest_value.end() ||
        !config->is_object()) {
        throw std::runtime_error(
            "DeepSeek-V4 CCCP manifest lacks config");
    }
    return from_json(config->dump());
}

void DeepseekV4Config::validate() const {
    const auto positive =
        [](std::int64_t value, const char* name) {
            if (value <= 0 ||
                value >
                    std::numeric_limits<std::int32_t>::max()) {
                throw std::runtime_error(
                    std::string("invalid DeepSeek-V4 config ") +
                    name);
            }
        };
    positive(n_layers, "n_layers");
    positive(hidden, "hidden");
    positive(n_experts, "n_experts");
    positive(top_k, "top_k");
    positive(moe_inter, "moe_inter");
    positive(n_shared, "n_shared");
    positive(n_heads, "n_heads");
    positive(head_dim, "head_dim");
    positive(q_lora_rank, "q_lora_rank");
    positive(o_lora_rank, "o_lora_rank");
    positive(o_groups, "o_groups");
    positive(kv_dim, "kv_dim");
    positive(qk_rope_head_dim, "qk_rope_head_dim");
    positive(n_kv_heads, "n_kv_heads");
    positive(vocab, "vocab");
    positive(sliding_window, "sliding_window");
    positive(index_n_heads, "index_n_heads");
    positive(index_head_dim, "index_head_dim");
    positive(index_topk, "index_topk");
    positive(
        max_position_embeddings,
        "max_position_embeddings");
    positive(hc_mult, "hc_mult");
    positive(hc_sinkhorn_iters, "hc_sinkhorn_iters");

    if (!valid_model_type(model_type)) {
        throw std::runtime_error(
            "unsupported DeepSeek-V4 model_type: " +
            model_type);
    }
    if (scoring_func != "sqrtsoftplus") {
        throw std::runtime_error(
            "DeepSeek-V4 Metal requires sqrtsoftplus routing");
    }
    if (n_kv_heads != 1) {
        throw std::runtime_error(
            "DeepSeek-V4 Metal requires one KV head");
    }
    if (top_k > std::min<std::int64_t>(16, n_experts)) {
        throw std::runtime_error(
            "DeepSeek-V4 top_k exceeds the Metal router limit");
    }
    const auto attention_width =
        checked_product(n_heads, head_dim, "attention width");
    if (attention_width % o_groups != 0) {
        throw std::runtime_error(
            "DeepSeek-V4 attention width must divide o_groups");
    }
    if (qk_rope_head_dim > head_dim ||
        qk_rope_head_dim > index_head_dim ||
        qk_rope_head_dim % 2 != 0) {
        throw std::runtime_error(
            "invalid DeepSeek-V4 rotary/indexer dimensions");
    }
    if (kv_dim != head_dim) {
        throw std::runtime_error(
            "DeepSeek-V4 Metal requires kv_dim=head_dim");
    }
    if (hc_mult != 4 || hc_sinkhorn_iters != 20) {
        throw std::runtime_error(
            "DeepSeek-V4 Metal requires hc_mult=4 and "
            "hc_sinkhorn_iters=20");
    }
    if (n_hash_layers < 0 || n_hash_layers > n_layers) {
        throw std::runtime_error(
            "DeepSeek-V4 n_hash_layers is outside the layer range");
    }
    if (!std::isfinite(rms_eps) || rms_eps <= 0.0 ||
        !std::isfinite(hc_eps) || hc_eps <= 0.0 ||
        !std::isfinite(rope_theta) || rope_theta <= 0.0 ||
        !std::isfinite(compress_rope_theta) ||
        compress_rope_theta <= 0.0 ||
        !std::isfinite(routed_scaling) ||
        routed_scaling <= 0.0 ||
        !std::isfinite(swiglu_limit) ||
        swiglu_limit < 0.0) {
        throw std::runtime_error(
            "invalid DeepSeek-V4 floating-point configuration");
    }
    if (rope_scaling.enabled &&
        (!std::isfinite(rope_scaling.factor) ||
         rope_scaling.factor <= 0.0 ||
         !std::isfinite(rope_scaling.beta_fast) ||
         rope_scaling.beta_fast <= 0.0 ||
         !std::isfinite(rope_scaling.beta_slow) ||
         rope_scaling.beta_slow <= 0.0 ||
         rope_scaling.original_max_position_embeddings < 0)) {
        throw std::runtime_error(
            "invalid DeepSeek-V4 rope_scaling configuration");
    }
    if (compress_ratios.size() !=
        static_cast<std::size_t>(n_layers)) {
        throw std::runtime_error(
            "DeepSeek-V4 compress_ratios must contain one "
            "entry per layer");
    }
    for (const auto ratio : compress_ratios) {
        if (ratio != 0 && ratio != 4 && ratio != 128) {
            throw std::runtime_error(
                "DeepSeek-V4 compression ratios must be 0, 4, or 128");
        }
    }
    for (const auto token : eos_token_id) {
        if (token < 0 || token >= vocab) {
            throw std::runtime_error(
                "DeepSeek-V4 eos_token_id is outside the vocabulary");
        }
    }

    (void)checked_product(
        n_shared,
        moe_inter,
        "shared expert width");
    (void)checked_product(
        hc_mult,
        hidden,
        "hyper-connection width");
    (void)checked_product(
        index_n_heads,
        index_head_dim,
        "indexer query width");
    (void)checked_product(
        o_groups,
        o_lora_rank,
        "grouped output rank");
    (void)checked_product(
        2,
        moe_inter,
        "routed gate/up width");
    (void)checked_product(
        2,
        head_dim,
        "main compressor width");
    (void)checked_product(
        2,
        index_head_dim,
        "Indexer compressor width");
}

std::string DeepseekV4TensorNames::layer(
    std::size_t index,
    std::string_view suffix) {
    return "layers." + std::to_string(index) + "." +
        std::string(suffix);
}

std::vector<std::string> DeepseekV4TensorNames::required(
    const DeepseekV4Config& config) {
    const auto bindings =
        deepseek_v4_required_bindings(config);
    std::vector<std::string> result;
    result.reserve(bindings.size());
    for (const auto& binding : bindings) {
        result.push_back(binding.name);
    }
    return result;
}

std::vector<DeepseekV4TensorBinding>
deepseek_v4_required_bindings(
    const DeepseekV4Config& config,
    const DeepseekV4TensorNames& names) {
    config.validate();
    std::vector<DeepseekV4TensorBinding> result;
    const auto attention =
        checked_product(
            config.n_heads,
            config.head_dim,
            "attention width");
    const auto shared =
        checked_product(
            config.n_shared,
            config.moe_inter,
            "shared expert width");
    const auto hc_width =
        checked_product(
            config.hc_mult,
            config.hidden,
            "hyper-connection width");
    const auto output_rank =
        checked_product(
            config.o_groups,
            config.o_lora_rank,
            "grouped output rank");
    const auto routed_gate_up =
        checked_product(
            2,
            config.moe_inter,
            "routed gate/up width");
    const auto hc_projection =
        config.hyper_connection_projection_size();

    add_binding(
        result,
        names.embedding,
        {config.vocab, config.hidden},
        DeepseekV4TensorKind::embedding);
    add_binding(
        result,
        names.output_norm,
        {config.hidden},
        DeepseekV4TensorKind::dense_float);
    add_binding(
        result,
        names.output,
        {config.vocab, config.hidden},
        DeepseekV4TensorKind::linear);
    add_binding(
        result,
        names.hc_head_fn,
        {config.hc_mult, hc_width},
        DeepseekV4TensorKind::linear);
    add_binding(
        result,
        names.hc_head_base,
        {config.hc_mult},
        DeepseekV4TensorKind::dense_float);
    add_binding(
        result,
        names.hc_head_scale,
        {1},
        DeepseekV4TensorKind::dense_float);

    for (std::size_t layer = 0;
         layer < config.compress_ratios.size();
         ++layer) {
        const auto name =
            [layer](std::string_view suffix) {
                return DeepseekV4TensorNames::layer(
                    layer,
                    suffix);
            };
        add_binding(
            result,
            name("attn.wq_a.weight"),
            {config.q_lora_rank, config.hidden},
            DeepseekV4TensorKind::linear);
        add_binding(
            result,
            name("attn.q_norm.weight"),
            {config.q_lora_rank},
            DeepseekV4TensorKind::dense_float);
        add_binding(
            result,
            name("attn.wq_b.weight"),
            {attention, config.q_lora_rank},
            DeepseekV4TensorKind::linear);
        add_binding(
            result,
            name("attn.wkv.weight"),
            {config.kv_dim, config.hidden},
            DeepseekV4TensorKind::linear);
        add_binding(
            result,
            name("attn.kv_norm.weight"),
            {config.kv_dim},
            DeepseekV4TensorKind::dense_float);
        add_binding(
            result,
            name("attn.attn_sink"),
            {config.n_heads},
            DeepseekV4TensorKind::dense_float);
        add_binding(
            result,
            name("attn.wo_a.weight"),
            {
                output_rank,
                attention / config.o_groups,
            },
            DeepseekV4TensorKind::linear);
        add_binding(
            result,
            name("attn.wo_b.weight"),
            {
                config.hidden,
                output_rank,
            },
            DeepseekV4TensorKind::linear);
        add_binding(
            result,
            name("attn_norm.weight"),
            {config.hidden},
            DeepseekV4TensorKind::dense_float);
        add_binding(
            result,
            name("ffn_norm.weight"),
            {config.hidden},
            DeepseekV4TensorKind::dense_float);
        add_binding(
            result,
            name("ffn.gate.weight"),
            {config.n_experts, config.hidden},
            DeepseekV4TensorKind::linear);
        add_binding(
            result,
            name("ffn.shared_experts.w1.weight"),
            {shared, config.hidden},
            DeepseekV4TensorKind::linear);
        add_binding(
            result,
            name("ffn.shared_experts.w3.weight"),
            {shared, config.hidden},
            DeepseekV4TensorKind::linear);
        add_binding(
            result,
            name("ffn.shared_experts.w2.weight"),
            {config.hidden, shared},
            DeepseekV4TensorKind::linear);
        add_binding(
            result,
            name("hc_attn_fn"),
            {hc_projection, hc_width},
            DeepseekV4TensorKind::linear);
        add_binding(
            result,
            name("hc_attn_base"),
            {hc_projection},
            DeepseekV4TensorKind::dense_float);
        add_binding(
            result,
            name("hc_attn_scale"),
            {3},
            DeepseekV4TensorKind::dense_float);
        add_binding(
            result,
            name("hc_ffn_fn"),
            {hc_projection, hc_width},
            DeepseekV4TensorKind::linear);
        add_binding(
            result,
            name("hc_ffn_base"),
            {hc_projection},
            DeepseekV4TensorKind::dense_float);
        add_binding(
            result,
            name("hc_ffn_scale"),
            {3},
            DeepseekV4TensorKind::dense_float);
        add_binding(
            result,
            name("ffn.experts.gate_up.weight"),
            {
                config.n_experts,
                routed_gate_up,
                config.hidden,
            },
            DeepseekV4TensorKind::routed_experts);
        add_binding(
            result,
            name("ffn.experts.down.weight"),
            {
                config.n_experts,
                config.hidden,
                config.moe_inter,
            },
            DeepseekV4TensorKind::routed_experts);
        if (layer <
            static_cast<std::size_t>(
                config.n_hash_layers)) {
            add_binding(
                result,
                name("ffn.gate.tid2eid"),
                {config.vocab, config.top_k},
                DeepseekV4TensorKind::dense_integer);
        } else {
            add_binding(
                result,
                name("ffn.gate.bias"),
                {config.n_experts},
                DeepseekV4TensorKind::dense_float);
        }

        const auto ratio = config.compress_ratios[layer];
        if (ratio != 0) {
            const auto compressor_width =
                checked_product(
                    config.head_dim,
                    ratio == 4 ? 2 : 1,
                    "main compressor width");
            add_binding(
                result,
                name("attn.compressor.wkv.weight"),
                {compressor_width, config.hidden},
                DeepseekV4TensorKind::linear);
            add_binding(
                result,
                name("attn.compressor.wgate.weight"),
                {compressor_width, config.hidden},
                DeepseekV4TensorKind::linear);
            add_binding(
                result,
                name("attn.compressor.ape"),
                {ratio, compressor_width},
                DeepseekV4TensorKind::dense_float);
            add_binding(
                result,
                name("attn.compressor.norm.weight"),
                {config.head_dim},
                DeepseekV4TensorKind::dense_float);
        }
        if (ratio == 4) {
            const auto index_width =
                checked_product(
                    config.index_n_heads,
                    config.index_head_dim,
                    "indexer query width");
            const auto compressor_width =
                checked_product(
                    2,
                    config.index_head_dim,
                    "Indexer compressor width");
            add_binding(
                result,
                name("attn.indexer.wq_b.weight"),
                {index_width, config.q_lora_rank},
                DeepseekV4TensorKind::linear);
            add_binding(
                result,
                name("attn.indexer.weights_proj.weight"),
                {config.index_n_heads, config.hidden},
                DeepseekV4TensorKind::linear);
            add_binding(
                result,
                name("attn.indexer.compressor.wkv.weight"),
                {compressor_width, config.hidden},
                DeepseekV4TensorKind::linear);
            add_binding(
                result,
                name("attn.indexer.compressor.wgate.weight"),
                {compressor_width, config.hidden},
                DeepseekV4TensorKind::linear);
            add_binding(
                result,
                name("attn.indexer.compressor.ape"),
                {4, compressor_width},
                DeepseekV4TensorKind::dense_float);
            add_binding(
                result,
                name("attn.indexer.compressor.norm.weight"),
                {config.index_head_dim},
                DeepseekV4TensorKind::dense_float);
        }
    }
    return result;
}

DeepseekV4TensorMetadata
inspect_deepseek_v4_tensor_metadata(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    const auto prefix = read_record_prefix(record);
    PrefixCursor cursor(prefix);
    DeepseekV4TensorMetadata result;
    result.dtype = record.dtype;

    if (is_dense_float_dtype(record.dtype) ||
        is_dense_integer_dtype(record.dtype)) {
        result.shape = read_shape(
            cursor,
            cursor.scalar<std::uint32_t>(
                "dense dimension count"),
            name);
        const std::uint64_t item_size =
            record.dtype == "BF16" ? 2u :
            record.dtype == "F16" ? 2u :
            record.dtype == "F32" ? 4u :
            record.dtype == "I32" ? 4u : 8u;
        const auto payload = checked_multiply(
            checked_elements(result.shape, name),
            item_size,
            name);
        const auto expected = checked_add(
            static_cast<std::uint64_t>(cursor.offset()),
            payload,
            name);
        if (expected != record.nbytes) {
            throw std::runtime_error(
                "invalid dense DeepSeek-V4 tensor length: " +
                name);
        }
        return result;
    }

    if (is_mx_dtype(record.dtype)) {
        result.packed = true;
        if (cursor.bytes(4, "MX magic") != "MXT1" ||
            cursor.scalar<std::uint8_t>("MX version") != 1) {
            throw std::runtime_error(
                "invalid DeepSeek-V4 MX header: " + name);
        }
        const auto kind = cursor.scalar<std::uint8_t>("MX kind");
        const auto reserved = cursor.scalar<std::uint16_t>("MX reserved");
        const auto rows = cursor.scalar<std::uint64_t>("MX rows");
        const auto columns = cursor.scalar<std::uint64_t>("MX columns");
        const auto storage_rows =
            cursor.scalar<std::uint64_t>("MX storage rows");
        const auto storage_columns =
            cursor.scalar<std::uint64_t>("MX storage columns");
        const auto scale_rows =
            cursor.scalar<std::uint64_t>("MX scale rows");
        const auto scale_columns =
            cursor.scalar<std::uint64_t>("MX scale columns");
        const auto bits = record.dtype == "MXFP4" ? 4u : 8u;
        const auto expected_storage_columns =
            bits == 4 ? columns / 2 : columns;
        const auto expected_scale_rows =
            bits == 4 ? rows : (rows + 127) / 128;
        const auto expected_scale_columns =
            bits == 4 ? columns / 32 : columns / 128;
        if (kind != bits || reserved != 0 || rows == 0 || columns == 0 ||
            rows > static_cast<std::uint64_t>(
                std::numeric_limits<std::int32_t>::max()) ||
            columns > static_cast<std::uint64_t>(
                std::numeric_limits<std::int32_t>::max()) ||
            (bits == 4 && columns % 32 != 0) ||
            (bits == 8 && columns % 128 != 0) ||
            storage_rows != rows ||
            storage_columns != expected_storage_columns ||
            scale_rows != expected_scale_rows ||
            scale_columns != expected_scale_columns) {
            throw std::runtime_error(
                "inconsistent DeepSeek-V4 MX geometry: " + name);
        }
        result.shape = {
            static_cast<std::int64_t>(rows),
            static_cast<std::int64_t>(columns),
        };
        const auto expected = checked_add(
            checked_add(
                static_cast<std::uint64_t>(cursor.offset()),
                checked_multiply(storage_rows, storage_columns, name),
                name),
            checked_multiply(scale_rows, scale_columns, name),
            name);
        if (expected != record.nbytes) {
            throw std::runtime_error(
                "invalid DeepSeek-V4 MX tensor length: " + name);
        }
        return result;
    }

    if (is_nint_dtype(record.dtype)) {
        result.packed = true;
        const auto bits =
            cursor.scalar<std::uint8_t>("NINT bits");
        const auto sub_bits =
            cursor.scalar<std::uint8_t>("NINT sub bits");
        const auto group_size =
            cursor.scalar<std::int32_t>("NINT group size");
        const auto axis =
            cursor.scalar<std::int32_t>("NINT axis");
        const auto neuron_len =
            cursor.scalar<std::int32_t>("NINT neuron length");
        result.shape = read_shape(
            cursor,
            cursor.scalar<std::uint32_t>(
                "NINT dimension count"),
            name);
        const auto output =
            cursor.scalar<std::uint32_t>("NINT output size");
        const auto groups =
            cursor.scalar<std::uint32_t>("NINT group count");
        if (bits == 0 || bits > 8 ||
            sub_bits == 0 || sub_bits > 8 ||
            group_size <= 0 || neuron_len <= 0 ||
            axis < 0 ||
            static_cast<std::size_t>(axis) >=
                result.shape.size() ||
            shape_product_except_axis(
                result.shape,
                axis,
                name) != neuron_len ||
            result.shape[static_cast<std::size_t>(axis)] !=
                output ||
            groups != static_cast<std::uint32_t>(
                (neuron_len + group_size - 1) / group_size) ||
            (record.dtype.size() == 5 &&
             bits !=
                 static_cast<std::uint8_t>(
                     record.dtype[4] - '0'))) {
            throw std::runtime_error(
                "inconsistent DeepSeek-V4 NINT header: " +
                name);
        }
        return result;
    }

    if (record.dtype == "NINT8-0") {
        result.packed = true;
        if (cursor.bytes(4, "NINT8-0 magic") != "NI80") {
            throw std::runtime_error(
                "invalid DeepSeek-V4 NINT8-0 magic: " +
                name);
        }
        const auto axis =
            cursor.scalar<std::int32_t>("NINT8-0 axis");
        const auto neuron_len =
            cursor.scalar<std::int32_t>(
                "NINT8-0 neuron length");
        result.shape = read_shape(
            cursor,
            cursor.scalar<std::uint32_t>(
                "NINT8-0 dimension count"),
            name);
        const auto output =
            cursor.scalar<std::uint32_t>(
                "NINT8-0 output size");
        const auto groups =
            cursor.scalar<std::uint32_t>(
                "NINT8-0 group count");
        if (axis < 0 ||
            static_cast<std::size_t>(axis) >=
                result.shape.size() ||
            neuron_len <= 0 ||
            neuron_len % 32 != 0 ||
            shape_product_except_axis(
                result.shape,
                axis,
                name) != neuron_len ||
            result.shape[static_cast<std::size_t>(axis)] !=
                output ||
            groups !=
                static_cast<std::uint32_t>(
                    neuron_len / 32)) {
            throw std::runtime_error(
                "inconsistent DeepSeek-V4 NINT8-0 header: " +
                name);
        }
        return result;
    }

    if (is_tpq_int4_dtype(record.dtype)) {
        result.packed = true;
        if (cursor.bytes(4, "CCCP-I4 magic") != "CI41" ||
            cursor.scalar<std::uint8_t>(
                "CCCP-I4 version") != 1) {
            throw std::runtime_error(
                "invalid DeepSeek-V4 CCCP-I4 header: " +
                name);
        }
        cursor.skip(3, "CCCP-I4 padding");
        const auto group_size =
            cursor.scalar<std::uint32_t>(
                "CCCP-I4 group size");
        const auto axis =
            cursor.scalar<std::int32_t>("CCCP-I4 axis");
        const auto neuron_len =
            cursor.scalar<std::int32_t>(
                "CCCP-I4 neuron length");
        result.shape = read_shape(
            cursor,
            cursor.scalar<std::uint32_t>(
                "CCCP-I4 dimension count"),
            name);
        const auto rows =
            cursor.scalar<std::uint32_t>("CCCP-I4 rows");
        const auto groups =
            cursor.scalar<std::uint32_t>("CCCP-I4 groups");
        if (group_size != 64 || axis != 0 ||
            result.shape.size() != 2 ||
            result.shape[0] != rows ||
            result.shape[1] != neuron_len ||
            neuron_len % 64 != 0 ||
            groups !=
                static_cast<std::uint32_t>(
                    neuron_len / 64)) {
            throw std::runtime_error(
                "inconsistent DeepSeek-V4 CCCP-I4 dimensions: " +
                name);
        }
        const auto values = checked_multiply(
            rows,
            static_cast<std::uint64_t>(neuron_len / 2),
            name);
        const auto scales = checked_multiply(
            checked_multiply(rows, groups, name),
            2,
            name);
        const auto expected = checked_add(
            checked_add(
                static_cast<std::uint64_t>(cursor.offset()),
                values,
                name),
            scales,
            name);
        if (expected != record.nbytes) {
            throw std::runtime_error(
                "invalid DeepSeek-V4 CCCP-I4 tensor length: " +
                name);
        }
        return result;
    }

    if (is_cccp_pq_dtype(record.dtype)) {
        result.packed = true;
        if (cursor.bytes(4, "CCCP-PQ magic") != "CPQ1" ||
            cursor.scalar<std::uint8_t>(
                "CCCP-PQ version") != 1) {
            throw std::runtime_error(
                "invalid DeepSeek-V4 CCCP-PQ header: " +
                name);
        }
        const auto tier =
            cursor.scalar<std::uint8_t>("CCCP-PQ tier");
        const auto vector_size =
            cursor.scalar<std::uint8_t>(
                "CCCP-PQ vector size");
        const auto index_bits =
            cursor.scalar<std::uint8_t>(
                "CCCP-PQ index bits");
        const auto axis =
            cursor.scalar<std::int32_t>("CCCP-PQ axis");
        const auto neuron_len =
            cursor.scalar<std::int32_t>(
                "CCCP-PQ neuron length");
        result.shape = read_shape(
            cursor,
            cursor.scalar<std::uint32_t>(
                "CCCP-PQ dimension count"),
            name);
        const auto entries =
            cursor.scalar<std::uint32_t>(
                "CCCP-PQ codebook entries");
        const auto rows =
            cursor.scalar<std::uint32_t>("CCCP-PQ rows");
        const bool storage_matches =
            (entries == 256 &&
             (index_bits == 8 ||
              index_bits == 12 ||
              index_bits == 14)) ||
            (entries == 4096 &&
             (index_bits == 12 ||
              index_bits == 14 ||
              index_bits == 16));
        const bool tier_matches =
            (is_tpq_tier(record.dtype, "X") &&
             tier == 1 && vector_size == 8 &&
             entries == 256) ||
            (is_tpq_tier(record.dtype, "W") &&
             tier == 2 && vector_size == 8 &&
             entries == 4096) ||
            (is_tpq_tier(record.dtype, "V") &&
             tier == 3 && vector_size == 4 &&
             entries == 256) ||
            (is_tpq_tier(record.dtype, "VV") &&
             tier == 4 && vector_size == 4 &&
             entries == 4096);
        if (!tier_matches || !storage_matches || axis != 0 ||
            result.shape.size() != 2 ||
            result.shape[0] != rows ||
            result.shape[1] != neuron_len ||
            neuron_len % vector_size != 0) {
            throw std::runtime_error(
                "inconsistent DeepSeek-V4 CCCP-PQ dimensions: " +
                name);
        }
        const auto codebook = checked_multiply(
            checked_multiply(entries, vector_size, name),
            4,
            name);
        const auto indices =
            checked_multiply(
                rows,
                static_cast<std::uint64_t>(
                    neuron_len / vector_size),
                name);
        const auto index_bytes = checked_add(
            checked_multiply(indices, index_bits, name),
            7,
            name) / 8;
        const auto expected = checked_add(
            checked_add(
                static_cast<std::uint64_t>(cursor.offset()),
                codebook,
                name),
            index_bytes,
            name);
        if (expected != record.nbytes) {
            throw std::runtime_error(
                "invalid DeepSeek-V4 CCCP-PQ tensor length: " +
                name);
        }
        return result;
    }

    if (record.dtype == "NINTM") {
        result.packed = true;
        const auto magic =
            cursor.bytes(4, "NINTM magic");
        const auto experts =
            cursor.scalar<std::uint32_t>("NINTM experts");
        const auto output =
            cursor.scalar<std::uint32_t>(
                "NINTM output per expert");
        const auto input =
            cursor.scalar<std::uint32_t>(
                "NINTM input width");
        const auto pools =
            cursor.scalar<std::uint32_t>("NINTM pool count");
        if ((magic != "NIM1" && magic != "NIM2") ||
            experts == 0 || output == 0 || input == 0 ||
            pools == 0 || pools > experts ||
            record.nbytes <= cursor.offset()) {
            throw std::runtime_error(
                "invalid DeepSeek-V4 NINTM header: " +
                name);
        }
        result.shape = {
            static_cast<std::int64_t>(experts),
            static_cast<std::int64_t>(output),
            static_cast<std::int64_t>(input),
        };
        return result;
    }

    throw std::runtime_error(
        "unsupported DeepSeek-V4 tensor dtype " +
        record.dtype + ": " + name);
}

void validate_deepseek_v4_model_bindings(
    const MfqContainer& model,
    const DeepseekV4Config& config,
    const DeepseekV4TensorNames& names) {
    const auto validate = [&](const DeepseekV4TensorBinding& binding) {
        if (!model.contains(binding.name)) {
            throw std::runtime_error(
                "DeepSeek-V4 model is missing tensor: " +
                binding.name);
        }
        const auto metadata =
            inspect_deepseek_v4_tensor_metadata(
                model,
                binding.name);
        if (metadata.shape != binding.shape) {
            throw std::runtime_error(
                "DeepSeek-V4 tensor " + binding.name +
                " has shape " + shape_text(metadata.shape) +
                ", expected " + shape_text(binding.shape));
        }

        bool supported = false;
        switch (binding.kind) {
        case DeepseekV4TensorKind::embedding:
            supported = is_embedding_dtype(metadata.dtype);
            break;
        case DeepseekV4TensorKind::linear:
            supported = is_linear_dtype(metadata.dtype);
            break;
        case DeepseekV4TensorKind::dense_float:
            supported = is_dense_float_dtype(metadata.dtype);
            break;
        case DeepseekV4TensorKind::dense_integer:
            supported = is_dense_integer_dtype(metadata.dtype);
            break;
        case DeepseekV4TensorKind::routed_experts:
            supported = metadata.dtype == "NINTM";
            break;
        }
        if (!supported) {
            throw std::runtime_error(
                "DeepSeek-V4 tensor " + binding.name +
                " has unsupported dtype " + metadata.dtype);
        }
    };

    constexpr std::string_view combined_suffix =
        "ffn.experts.gate_up.weight";
    for (const auto& binding :
         deepseek_v4_required_bindings(config, names)) {
        if (binding.name.ends_with(combined_suffix)) {
            const auto prefix = binding.name.substr(
                0,
                binding.name.size() - combined_suffix.size());
            const auto gate_name =
                prefix + "ffn.experts.gate.weight";
            const auto up_name =
                prefix + "ffn.experts.up.weight";
            const bool has_gate = model.contains(gate_name);
            const bool has_up = model.contains(up_name);
            if (has_gate != has_up) {
                throw std::runtime_error(
                    "DeepSeek-V4 split routed Gate/Up records "
                    "are incomplete: " + binding.name);
            }
            if (has_gate) {
                const std::vector<std::int64_t> split_shape{
                    config.n_experts,
                    config.moe_inter,
                    config.hidden,
                };
                validate({
                    gate_name,
                    split_shape,
                    DeepseekV4TensorKind::routed_experts,
                });
                validate({
                    up_name,
                    split_shape,
                    DeepseekV4TensorKind::routed_experts,
                });
                continue;
            }
        }
        validate(binding);
    }
}

} // namespace mfq::metal
