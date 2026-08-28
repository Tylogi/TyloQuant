#include "qwen35_model.h"

#include "../json/nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mfq::metal {
namespace {

using json = nlohmann::json;
using mlx::core::array;

const json& object_or_self(const json& root, const char* key) {
    const auto found = root.find(key);
    if (found == root.end()) {
        return root;
    }
    if (!found->is_object()) {
        throw std::runtime_error(
            std::string("embedded model config ") + key +
            " must be an object");
    }
    return *found;
}

const json& object_or_empty(
    const json& parent,
    const char* key,
    const json& empty) {
    const auto found = parent.find(key);
    if (found == parent.end() || found->is_null()) {
        return empty;
    }
    if (!found->is_object()) {
        throw std::runtime_error(
            std::string("embedded model config ") + key +
            " must be an object");
    }
    return *found;
}

std::int64_t required_integer(const json& object, const char* key) {
    const auto found = object.find(key);
    if (found == object.end() || !found->is_number_integer()) {
        throw std::runtime_error(
            std::string("embedded model config requires integer ") + key);
    }
    return found->get<std::int64_t>();
}

std::int64_t optional_integer(
    const json& object,
    const char* key,
    std::int64_t default_value) {
    const auto found = object.find(key);
    if (found == object.end() || found->is_null()) {
        return default_value;
    }
    if (!found->is_number_integer()) {
        throw std::runtime_error(
            std::string("embedded model config ") + key +
            " must be an integer");
    }
    return found->get<std::int64_t>();
}

double optional_number(
    const json& object,
    const char* key,
    double default_value) {
    const auto found = object.find(key);
    if (found == object.end() || found->is_null()) {
        return default_value;
    }
    if (!found->is_number()) {
        throw std::runtime_error(
            std::string("embedded model config ") + key +
            " must be a number");
    }
    return found->get<double>();
}

bool optional_boolean(
    const json& object,
    const char* key,
    bool default_value) {
    const auto found = object.find(key);
    if (found == object.end() || found->is_null()) {
        return default_value;
    }
    if (!found->is_boolean()) {
        throw std::runtime_error(
            std::string("embedded model config ") + key +
            " must be boolean");
    }
    return found->get<bool>();
}

std::string optional_string(
    const json& object,
    const char* key,
    std::string default_value = {}) {
    const auto found = object.find(key);
    if (found == object.end() || found->is_null()) {
        return default_value;
    }
    if (!found->is_string()) {
        throw std::runtime_error(
            std::string("embedded model config ") + key +
            " must be a string");
    }
    return found->get<std::string>();
}

std::vector<std::int64_t> optional_integer_array(
    const json& object,
    const char* key) {
    const auto found = object.find(key);
    if (found == object.end() || found->is_null()) {
        return {};
    }
    if (!found->is_array()) {
        throw std::runtime_error(
            std::string("embedded model config ") + key +
            " must be an array");
    }
    std::vector<std::int64_t> result;
    result.reserve(found->size());
    for (const auto& value : *found) {
        if (!value.is_number_integer()) {
            throw std::runtime_error(
                std::string("embedded model config ") + key +
                " entries must be integers");
        }
        result.push_back(value.get<std::int64_t>());
    }
    return result;
}

std::vector<std::string> layer_types(
    const json& object,
    std::int64_t count) {
    const auto found = object.find("layer_types");
    if (found == object.end() || found->is_null()) {
        return std::vector<std::string>(
            static_cast<std::size_t>(count),
            "full_attention");
    }
    if (!found->is_array()) {
        throw std::runtime_error(
            "embedded model config layer_types must be an array");
    }
    std::vector<std::string> result;
    result.reserve(found->size());
    for (const auto& value : *found) {
        if (!value.is_string()) {
            throw std::runtime_error(
                "embedded model config layer_types entries must be strings");
        }
        result.push_back(value.get<std::string>());
    }
    return result;
}

void validate_config(const Qwen35Config& config) {
    const auto positive = [](std::int64_t value, const char* name) {
        if (value <= 0 ||
            value > std::numeric_limits<std::int32_t>::max()) {
            throw std::runtime_error(
                std::string("invalid Qwen3.5 config ") + name);
        }
    };
    positive(config.vocab_size, "vocab_size");
    positive(config.hidden_size, "hidden_size");
    positive(config.intermediate_size, "intermediate_size");
    positive(config.num_hidden_layers, "num_hidden_layers");
    positive(config.num_attention_heads, "num_attention_heads");
    positive(config.num_key_value_heads, "num_key_value_heads");
    positive(config.max_position_embeddings, "max_position_embeddings");
    positive(config.head_dim, "head_dim");
    positive(config.rotary_dim, "rotary_dim");
    positive(config.linear_conv_kernel_dim, "linear_conv_kernel_dim");
    positive(config.linear_key_head_dim, "linear_key_head_dim");
    positive(config.linear_value_head_dim, "linear_value_head_dim");
    if (config.linear_num_key_heads < 0 ||
        config.linear_num_value_heads < 0 ||
        config.full_attention_interval < 0 ||
        config.mtp_num_hidden_layers < 0) {
        throw std::runtime_error(
            "invalid negative Qwen3.5 count in model config");
    }
    positive(config.linear_key_heads(), "linear key head count");
    positive(config.linear_value_heads(), "linear value head count");
    if (config.rotary_dim > config.head_dim) {
        throw std::runtime_error(
            "invalid Qwen3.5 config rotary_dim exceeds head_dim");
    }
    if (!std::isfinite(config.rope_base) || config.rope_base <= 0.0 ||
        !std::isfinite(config.rms_norm_eps) ||
        config.rms_norm_eps <= 0.0) {
        throw std::runtime_error(
            "invalid Qwen3.5 RoPE or RMSNorm configuration");
    }
    if (config.layer_types.size() !=
        static_cast<std::size_t>(config.num_hidden_layers)) {
        throw std::runtime_error(
            "Qwen3.5 layer_types length does not match num_hidden_layers");
    }
    for (const auto& type : config.layer_types) {
        if (type != "full_attention" &&
            type != "linear_attention") {
            throw std::runtime_error(
                "unsupported Qwen3.5 layer type: " + type);
        }
    }
    std::int64_t rope_section_sum = 0;
    for (const auto section : config.rope_sections) {
        if (section < 0) {
            throw std::runtime_error(
                "invalid Qwen3.5 config rope_sections entry");
        }
        rope_section_sum += section;
    }
    if (!config.rope_sections.empty() &&
        rope_section_sum * 2 != config.rotary_dim) {
        throw std::runtime_error(
            "Qwen3.5 mRoPE sections do not cover rotary_dim");
    }
}

class PrefixCursor {
public:
    explicit PrefixCursor(const std::vector<std::uint8_t>& bytes)
        : bytes_(bytes) {}

    template <typename T>
    T scalar(const char* what) {
        if (sizeof(T) > bytes_.size() - offset_) {
            throw std::runtime_error(
                std::string("truncated MFQ tensor ") + what);
        }
        T value{};
        std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    std::string bytes(std::size_t count, const char* what) {
        if (count > bytes_.size() - offset_) {
            throw std::runtime_error(
                std::string("truncated MFQ tensor ") + what);
        }
        std::string value(
            reinterpret_cast<const char*>(bytes_.data() + offset_),
            count);
        offset_ += count;
        return value;
    }

    std::size_t offset() const noexcept {
        return offset_;
    }

private:
    const std::vector<std::uint8_t>& bytes_;
    std::size_t offset_ = 0;
};

std::vector<std::uint8_t> read_record_prefix(
    const MfqContainer& model,
    const std::string& name,
    std::size_t maximum = 128) {
    const auto& record = model.record(name);
    const auto count = static_cast<std::size_t>(
        std::min<std::uint64_t>(record.nbytes, maximum));
    return model.read_range(name, 0, count);
}

std::vector<std::int64_t> read_shape(
    PrefixCursor& cursor,
    std::uint32_t dimensions,
    const std::string& name) {
    if (dimensions == 0 || dimensions > 8) {
        throw std::runtime_error(
            "invalid MFQ tensor dimension count: " + name);
    }
    std::vector<std::int64_t> shape;
    shape.reserve(dimensions);
    for (std::uint32_t index = 0; index < dimensions; ++index) {
        const auto value = cursor.scalar<std::int64_t>("shape");
        if (value <= 0 ||
            value > std::numeric_limits<std::int32_t>::max()) {
            throw std::runtime_error(
                "invalid MFQ tensor shape: " + name);
        }
        shape.push_back(value);
    }
    return shape;
}

std::int64_t shape_product_except_axis(
    const std::vector<std::int64_t>& shape,
    std::int32_t axis,
    const std::string& name) {
    if (axis < 0 ||
        static_cast<std::size_t>(axis) >= shape.size()) {
        throw std::runtime_error(
            "invalid MFQ tensor axis: " + name);
    }
    std::int64_t product = 1;
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index == static_cast<std::size_t>(axis)) {
            continue;
        }
        if (product >
            std::numeric_limits<std::int64_t>::max() / shape[index]) {
            throw std::runtime_error(
                "MFQ tensor shape overflows: " + name);
        }
        product *= shape[index];
    }
    return product;
}

bool is_dense_float_dtype(const std::string& dtype) {
    return dtype == "BF16" || dtype == "F16" || dtype == "F32";
}

bool is_supported_linear_dtype(const std::string& dtype) {
    return is_dense_float_dtype(dtype) ||
        is_nint_dtype(dtype) ||
        is_nint8_zero_dtype(dtype) ||
        is_vq_dtype(dtype);
}

std::string shape_text(const std::vector<std::int64_t>& shape) {
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

void require_tensor(
    const MfqContainer& model,
    const std::string& name,
    const std::vector<std::int64_t>& expected_shape,
    bool dense_only) {
    if (!model.contains(name)) {
        throw std::runtime_error(
            "Qwen3.5 model is missing tensor: " + name);
    }
    const auto metadata =
        inspect_qwen35_tensor_metadata(model, name);
    if (metadata.shape != expected_shape) {
        throw std::runtime_error(
            "Qwen3.5 tensor " + name + " has shape " +
            shape_text(metadata.shape) + ", expected " +
            shape_text(expected_shape));
    }
    const bool supported = dense_only
        ? is_dense_float_dtype(metadata.dtype)
        : is_supported_linear_dtype(metadata.dtype);
    if (!supported) {
        throw std::runtime_error(
            "Qwen3.5 tensor " + name +
            " has unsupported dtype " + metadata.dtype);
    }
}

void require_optional_tensor(
    const MfqContainer& model,
    const std::string& name,
    const std::vector<std::int64_t>& expected_shape,
    bool dense_only) {
    if (model.contains(name)) {
        require_tensor(model, name, expected_shape, dense_only);
    }
}

std::string format_layer_name(
    const std::string& pattern,
    std::size_t index) {
    auto result = pattern;
    const auto placeholder = result.find("{i}");
    if (placeholder == std::string::npos) {
        throw std::runtime_error(
            "Qwen3.5 tensor pattern lacks {i}: " + pattern);
    }
    result.replace(placeholder, 3, std::to_string(index));
    return result;
}

std::optional<std::string> format_optional_layer_name(
    const std::optional<std::string>& pattern,
    std::size_t index) {
    if (!pattern) {
        return std::nullopt;
    }
    return format_layer_name(*pattern, index);
}

array load_norm_weight(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    if (!is_dense_float_dtype(record.dtype)) {
        throw std::runtime_error(
            "RMSNorm requires a dense BF16/F16/F32 tensor: " + name);
    }
    auto weight = load_dense_array(record.dtype, model.read(name));
    if (weight.ndim() != 1) {
        throw std::runtime_error(
            "RMSNorm weight must be one-dimensional: " + name);
    }
    return weight.dtype() == mlx::core::float32
        ? weight
        : mlx::core::astype(weight, mlx::core::float32);
}

} // namespace

Qwen35Config Qwen35Config::from_json(std::string_view payload) {
    json root;
    try {
        root = json::parse(payload.begin(), payload.end());
    } catch (const json::exception& error) {
        throw std::runtime_error(
            std::string("invalid embedded model config JSON: ") +
            error.what());
    }
    if (!root.is_object()) {
        throw std::runtime_error(
            "embedded model config must be a JSON object");
    }
    const auto& text = object_or_self(root, "text_config");
    static const json empty = json::object();
    const auto& rope_parameters =
        object_or_empty(text, "rope_parameters", empty);
    const auto& full_rope =
        object_or_empty(rope_parameters, "full_attention", empty);

    Qwen35Config config;
    config.model_type = optional_string(
        root,
        "model_type",
        optional_string(text, "model_type"));
    config.text_model_type =
        optional_string(text, "model_type", config.model_type);
    config.vocab_size = required_integer(text, "vocab_size");
    config.hidden_size = required_integer(text, "hidden_size");
    config.intermediate_size =
        required_integer(text, "intermediate_size");
    config.num_hidden_layers =
        required_integer(text, "num_hidden_layers");
    config.num_attention_heads =
        required_integer(text, "num_attention_heads");
    config.num_key_value_heads =
        required_integer(text, "num_key_value_heads");
    config.max_position_embeddings =
        required_integer(text, "max_position_embeddings");
    const auto default_head_dim =
        config.hidden_size % config.num_attention_heads == 0
        ? config.hidden_size / config.num_attention_heads
        : 0;
    config.head_dim =
        optional_integer(text, "head_dim", default_head_dim);

    config.rope_base = optional_number(
        full_rope,
        "rope_theta",
        optional_number(
            rope_parameters,
            "rope_theta",
            optional_number(text, "rope_theta", 1'000'000.0)));
    const auto partial_rotary_factor = optional_number(
        full_rope,
        "partial_rotary_factor",
        optional_number(
            rope_parameters,
            "partial_rotary_factor",
            optional_number(text, "partial_rotary_factor", 1.0)));
    config.rotary_dim = static_cast<std::int64_t>(
        std::llround(
            partial_rotary_factor *
            static_cast<double>(config.head_dim)));
    config.rope_sections =
        optional_integer_array(full_rope, "mrope_section");
    if (config.rope_sections.empty()) {
        config.rope_sections =
            optional_integer_array(rope_parameters, "mrope_section");
    }
    config.rms_norm_eps =
        optional_number(text, "rms_norm_eps", 1e-6);
    config.norm_weight_offset =
        optional_number(text, "norm_weight_offset", 1.0);
    config.tie_word_embeddings =
        optional_boolean(text, "tie_word_embeddings", false);
    config.attention_output_gate =
        optional_boolean(text, "attn_output_gate", false);
    config.output_gate_type =
        optional_string(text, "output_gate_type");
    config.layer_types =
        ::mfq::metal::layer_types(text, config.num_hidden_layers);
    config.full_attention_interval =
        optional_integer(text, "full_attention_interval", 0);
    config.linear_conv_kernel_dim =
        optional_integer(text, "linear_conv_kernel_dim", 4);
    config.linear_key_head_dim =
        optional_integer(text, "linear_key_head_dim", 128);
    config.linear_value_head_dim =
        optional_integer(text, "linear_value_head_dim", 128);
    config.linear_num_key_heads =
        optional_integer(text, "linear_num_key_heads", 0);
    config.linear_num_value_heads =
        optional_integer(text, "linear_num_value_heads", 0);
    config.linear_a_is_log =
        optional_boolean(text, "linear_a_is_log", true);
    config.mrope_interleaved =
        optional_boolean(rope_parameters, "mrope_interleaved", false);
    config.mtp_num_hidden_layers =
        optional_integer(text, "mtp_num_hidden_layers", 0);
    config.mtp_use_dedicated_embeddings =
        optional_boolean(text, "mtp_use_dedicated_embeddings", false);

    validate_config(config);
    return config;
}

Qwen35Config Qwen35Config::from_mfq(const MfqContainer& model) {
    if (!model.contains(std::string(kMfqModelConfigAsset))) {
        throw std::runtime_error(
            "MFQ has no embedded model config asset");
    }
    return from_json(
        model.read_text(std::string(kMfqModelConfigAsset)));
}

Qwen35TensorNames Qwen35TensorNames::gguf() {
    Qwen35TensorNames names;
    names.ffn_norm = "blk.{i}.post_attention_norm.weight";
    names.linear_qkv = "blk.{i}.attn_qkv.weight";
    names.linear_z = "blk.{i}.attn_gate.weight";
    return names;
}

Qwen35TensorNames Qwen35TensorNames::hugging_face() {
    Qwen35TensorNames names;
    names.token_embedding =
        "model.language_model.embed_tokens.weight";
    names.attention_norm =
        "model.language_model.layers.{i}.input_layernorm.weight";
    names.attention_query =
        "model.language_model.layers.{i}.self_attn.q_proj.weight";
    names.attention_key =
        "model.language_model.layers.{i}.self_attn.k_proj.weight";
    names.attention_value =
        "model.language_model.layers.{i}.self_attn.v_proj.weight";
    names.attention_output =
        "model.language_model.layers.{i}.self_attn.o_proj.weight";
    names.attention_query_norm =
        "model.language_model.layers.{i}.self_attn.q_norm.weight";
    names.attention_key_norm =
        "model.language_model.layers.{i}.self_attn.k_norm.weight";
    names.ffn_norm =
        "model.language_model.layers.{i}.post_attention_layernorm.weight";
    names.ffn_gate =
        "model.language_model.layers.{i}.mlp.gate_proj.weight";
    names.ffn_up =
        "model.language_model.layers.{i}.mlp.up_proj.weight";
    names.ffn_down =
        "model.language_model.layers.{i}.mlp.down_proj.weight";
    names.output_norm = "model.language_model.norm.weight";
    names.output = "lm_head.weight";
    names.linear_qkv =
        "model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight";
    names.linear_qk =
        "model.language_model.layers.{i}.linear_attn.in_proj_qk.weight";
    names.linear_value =
        "model.language_model.layers.{i}.linear_attn.in_proj_v.weight";
    names.linear_z =
        "model.language_model.layers.{i}.linear_attn.in_proj_z.weight";
    names.linear_alpha =
        "model.language_model.layers.{i}.linear_attn.in_proj_a.weight";
    names.linear_beta =
        "model.language_model.layers.{i}.linear_attn.in_proj_b.weight";
    names.linear_conv =
        "model.language_model.layers.{i}.linear_attn.conv1d.weight";
    names.linear_conv_bias =
        "model.language_model.layers.{i}.linear_attn.conv1d.bias";
    names.linear_dt_bias =
        "model.language_model.layers.{i}.linear_attn.dt_bias";
    names.linear_a =
        "model.language_model.layers.{i}.linear_attn.A_log";
    names.linear_norm =
        "model.language_model.layers.{i}.linear_attn.norm.weight";
    names.linear_output =
        "model.language_model.layers.{i}.linear_attn.out_proj.weight";
    return names;
}

Qwen35TensorNames Qwen35TensorNames::detect(
    const MfqContainer& model) {
    const auto hf = hugging_face();
    if (model.contains(hf.token_embedding)) {
        return hf;
    }
    const auto native = gguf();
    if (model.contains(native.token_embedding)) {
        return native;
    }
    throw std::runtime_error(
        "cannot detect Qwen3.5 MFQ tensor naming layout");
}

Qwen35ResolvedLayerNames Qwen35TensorNames::layer(
    std::size_t index) const {
    return {
        format_layer_name(attention_norm, index),
        format_layer_name(attention_query, index),
        format_layer_name(attention_key, index),
        format_layer_name(attention_value, index),
        format_layer_name(attention_output, index),
        format_layer_name(attention_query_norm, index),
        format_layer_name(attention_key_norm, index),
        format_layer_name(ffn_norm, index),
        format_layer_name(ffn_gate, index),
        format_layer_name(ffn_up, index),
        format_layer_name(ffn_down, index),
        format_layer_name(linear_qkv, index),
        format_optional_layer_name(linear_qk, index),
        format_optional_layer_name(linear_value, index),
        format_layer_name(linear_z, index),
        format_layer_name(linear_alpha, index),
        format_layer_name(linear_beta, index),
        format_layer_name(linear_conv, index),
        format_optional_layer_name(linear_conv_bias, index),
        format_layer_name(linear_dt_bias, index),
        format_layer_name(linear_a, index),
        format_layer_name(linear_norm, index),
        format_layer_name(linear_output, index),
    };
}

Qwen35Config adapt_qwen35_config_for_tensor_names(
    Qwen35Config config,
    const Qwen35TensorNames& names) {
    if (names.linear_qkv == "blk.{i}.attn_qkv.weight") {
        config.norm_weight_offset = 0.0;
        config.linear_a_is_log = false;
    }
    return config;
}

Qwen35TensorMetadata inspect_qwen35_tensor_metadata(
    const MfqContainer& model,
    const std::string& name) {
    const auto& record = model.record(name);
    const auto prefix = read_record_prefix(model, name);
    PrefixCursor cursor(prefix);
    Qwen35TensorMetadata metadata;
    metadata.dtype = record.dtype;

    if (is_dense_float_dtype(record.dtype) ||
        record.dtype == "I32" ||
        record.dtype == "I64") {
        metadata.shape = read_shape(
            cursor,
            cursor.scalar<std::uint32_t>("dimension count"),
            name);
        return metadata;
    }

    if (is_nint8_zero_dtype(record.dtype)) {
        if (cursor.bytes(4, "NINT8-0 magic") != "NI80") {
            throw std::runtime_error(
                "invalid NINT8-0 tensor magic: " + name);
        }
        metadata.packed = true;
        metadata.bits = 8;
        metadata.axis = cursor.scalar<std::int32_t>("axis");
        metadata.neuron_len =
            cursor.scalar<std::int32_t>("neuron length");
        metadata.shape = read_shape(
            cursor,
            cursor.scalar<std::uint32_t>("dimension count"),
            name);
        metadata.output_size =
            cursor.scalar<std::uint32_t>("output size");
        metadata.groups =
            cursor.scalar<std::uint32_t>("group count");
        if (metadata.neuron_len <= 0 ||
            metadata.neuron_len % 32 != 0 ||
            shape_product_except_axis(
                metadata.shape,
                metadata.axis,
                name) != metadata.neuron_len ||
            metadata.output_size !=
                static_cast<std::uint32_t>(
                    metadata.shape.at(
                        static_cast<std::size_t>(metadata.axis))) ||
            metadata.groups !=
                static_cast<std::uint32_t>(metadata.neuron_len / 32)) {
            throw std::runtime_error(
                "inconsistent NINT8-0 tensor header: " + name);
        }
        const auto expected = cursor.offset() +
            static_cast<std::uint64_t>(metadata.output_size) *
            metadata.groups * 34u;
        if (expected != record.nbytes) {
            throw std::runtime_error(
                "invalid NINT8-0 tensor length: " + name);
        }
        return metadata;
    }

    if (is_nint_dtype(record.dtype)) {
        metadata.packed = true;
        metadata.bits = cursor.scalar<std::uint8_t>("NINT bits");
        metadata.sub_bits =
            cursor.scalar<std::uint8_t>("NINT sub bits");
        metadata.group_size =
            cursor.scalar<std::int32_t>("NINT group size");
        metadata.axis = cursor.scalar<std::int32_t>("NINT axis");
        metadata.neuron_len =
            cursor.scalar<std::int32_t>("NINT neuron length");
        metadata.shape = read_shape(
            cursor,
            cursor.scalar<std::uint32_t>("dimension count"),
            name);
        metadata.output_size =
            cursor.scalar<std::uint32_t>("output size");
        metadata.groups =
            cursor.scalar<std::uint32_t>("group count");
        if (metadata.bits <= 0 || metadata.bits > 8 ||
            metadata.sub_bits <= 0 || metadata.sub_bits > 8 ||
            metadata.group_size <= 0 ||
            metadata.neuron_len <= 0 ||
            shape_product_except_axis(
                metadata.shape,
                metadata.axis,
                name) != metadata.neuron_len ||
            metadata.output_size !=
                static_cast<std::uint32_t>(
                    metadata.shape.at(
                        static_cast<std::size_t>(metadata.axis))) ||
            metadata.groups !=
                static_cast<std::uint32_t>(
                    (metadata.neuron_len +
                     metadata.group_size - 1) /
                    metadata.group_size) ||
            (record.dtype.size() == 5 &&
             metadata.bits != record.dtype[4] - '0')) {
            throw std::runtime_error(
                "inconsistent NINT tensor header: " + name);
        }
        return metadata;
    }

    if (is_vq_dtype(record.dtype)) {
        const auto vq = inspect_vq_blob(record.dtype, prefix);
        metadata.packed = true;
        metadata.neuron_len = vq.input_size;
        metadata.output_size =
            static_cast<std::uint32_t>(vq.output_size);
        metadata.shape.reserve(vq.output_shape.size() + 1);
        for (const int dimension : vq.output_shape) {
            metadata.shape.push_back(dimension);
        }
        metadata.shape.push_back(vq.input_size);
        return metadata;
    }

    throw std::runtime_error(
        "unsupported Qwen3.5 tensor dtype " +
        record.dtype + ": " + name);
}

void validate_qwen35_model_bindings(
    const MfqContainer& model,
    const Qwen35Config& config,
    const Qwen35TensorNames& names) {
    require_tensor(
        model,
        names.token_embedding,
        {config.vocab_size, config.hidden_size},
        false);
    require_tensor(
        model,
        names.output_norm,
        {config.hidden_size},
        true);
    if (model.contains(names.output)) {
        require_tensor(
            model,
            names.output,
            {config.vocab_size, config.hidden_size},
            false);
    } else if (!config.tie_word_embeddings) {
        throw std::runtime_error(
            "Qwen3.5 model is missing untied output tensor: " +
            names.output);
    }

    for (std::size_t index = 0;
         index < config.layer_types.size();
         ++index) {
        const auto layer = names.layer(index);
        require_tensor(
            model,
            layer.attention_norm,
            {config.hidden_size},
            true);
        require_tensor(
            model,
            layer.ffn_norm,
            {config.hidden_size},
            true);
        require_tensor(
            model,
            layer.ffn_gate,
            {config.intermediate_size, config.hidden_size},
            false);
        require_tensor(
            model,
            layer.ffn_up,
            {config.intermediate_size, config.hidden_size},
            false);
        require_tensor(
            model,
            layer.ffn_down,
            {config.hidden_size, config.intermediate_size},
            false);

        if (config.layer_types[index] == "full_attention") {
            require_tensor(
                model,
                layer.attention_query,
                {config.query_projection_size(), config.hidden_size},
                false);
            require_tensor(
                model,
                layer.attention_key,
                {config.kv_size(), config.hidden_size},
                false);
            require_tensor(
                model,
                layer.attention_value,
                {config.kv_size(), config.hidden_size},
                false);
            require_tensor(
                model,
                layer.attention_output,
                {config.hidden_size, config.attention_size()},
                false);
            require_optional_tensor(
                model,
                layer.attention_query_norm,
                {config.head_dim},
                true);
            require_optional_tensor(
                model,
                layer.attention_key_norm,
                {config.head_dim},
                true);
            continue;
        }

        const bool split_input =
            layer.linear_qk.has_value() &&
            layer.linear_value.has_value() &&
            model.contains(*layer.linear_qk) &&
            model.contains(*layer.linear_value);
        if (split_input) {
            require_tensor(
                model,
                *layer.linear_qk,
                {2 * config.linear_key_size(), config.hidden_size},
                false);
            require_tensor(
                model,
                *layer.linear_value,
                {config.linear_value_size(), config.hidden_size},
                false);
        } else {
            require_tensor(
                model,
                layer.linear_qkv,
                {config.linear_qkv_size(), config.hidden_size},
                false);
        }
        require_tensor(
            model,
            layer.linear_z,
            {config.linear_value_size(), config.hidden_size},
            false);
        require_tensor(
            model,
            layer.linear_alpha,
            {config.linear_value_heads(), config.hidden_size},
            false);
        require_tensor(
            model,
            layer.linear_beta,
            {config.linear_value_heads(), config.hidden_size},
            false);

        const auto conv = inspect_qwen35_tensor_metadata(
            model,
            layer.linear_conv);
        const std::vector<std::int64_t> conv_2d = {
            config.linear_qkv_size(),
            config.linear_conv_kernel_dim,
        };
        const std::vector<std::int64_t> conv_3d = {
            config.linear_qkv_size(),
            1,
            config.linear_conv_kernel_dim,
        };
        if (!is_dense_float_dtype(conv.dtype) ||
            (conv.shape != conv_2d && conv.shape != conv_3d)) {
            throw std::runtime_error(
                "Qwen3.5 tensor " + layer.linear_conv +
                " has incompatible convolution shape/dtype");
        }
        if (layer.linear_conv_bias.has_value()) {
            require_optional_tensor(
                model,
                *layer.linear_conv_bias,
                {config.linear_qkv_size()},
                true);
        }
        require_tensor(
            model,
            layer.linear_dt_bias,
            {config.linear_value_heads()},
            true);
        require_tensor(
            model,
            layer.linear_a,
            {config.linear_value_heads()},
            true);
        require_tensor(
            model,
            layer.linear_norm,
            {config.linear_value_head_dim},
            true);
        require_tensor(
            model,
            layer.linear_output,
            {config.hidden_size, config.linear_value_size()},
            false);
    }
}

MlxRmsNorm load_qwen35_rms_norm(
    const MfqContainer& model,
    const std::string& name,
    double eps,
    double weight_offset) {
    return MlxRmsNorm(
        load_norm_weight(model, name),
        static_cast<float>(eps),
        static_cast<float>(weight_offset));
}

Qwen35Linear Qwen35Linear::load(
    const MfqContainer& model,
    const std::string& name) {
    return Qwen35Linear(MlxLinear::load(model, name));
}

Qwen35Linear::Qwen35Linear(MlxLinear linear)
    : linear_(std::move(linear)) {}

array Qwen35Linear::operator()(const array& input) const {
    return linear_(input);
}

Qwen35Embedding Qwen35Embedding::load(
    const MfqContainer& model,
    const std::string& name) {
    return Qwen35Embedding(MlxEmbedding::load(model, name));
}

Qwen35Embedding::Qwen35Embedding(MlxEmbedding embedding)
    : embedding_(std::move(embedding)) {}

array Qwen35Embedding::operator()(
    const array& token_ids,
    mlx::core::Dtype dtype) const {
    return embedding_(token_ids, dtype);
}

array Qwen35Embedding::project(const array& input) const {
    return embedding_.project(input);
}

} // namespace mfq::metal
