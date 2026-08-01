#include "deepseek_v4_model.h"

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;
using mfq::metal::DeepseekV4Config;
using mfq::metal::DeepseekV4TensorBinding;
using mfq::metal::DeepseekV4TensorKind;
using mfq::metal::DeepseekV4TensorNames;
using mfq::metal::MfqContainer;

void require(
    bool condition,
    const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(
    Function&& function,
    std::string_view expected,
    const std::string& message) {
    try {
        function();
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find(expected) !=
            std::string::npos) {
            return;
        }
        throw std::runtime_error(
            message + ": unexpected error: " + error.what());
    }
    throw std::runtime_error(message + ": no exception");
}

json manifest_config(
    std::vector<std::int64_t> ratios = {0, 4, 128}) {
    return {
        {"n_layers", static_cast<std::int64_t>(ratios.size())},
        {"hidden", 64},
        {"n_experts", 4},
        {"top_k", 2},
        {"moe_inter", 64},
        {"n_shared", 1},
        {"n_heads", 4},
        {"head_dim", 16},
        {"q_lora_rank", 64},
        {"o_lora_rank", 4},
        {"o_groups", 4},
        {"kv_dim", 16},
        {"qk_rope_head_dim", 8},
        {"n_kv_heads", 1},
        {"vocab", 32},
        {"rms_eps", 1e-6},
        {"scoring_func", "sqrtsoftplus"},
        {"norm_topk_prob", true},
        {"routed_scaling", 1.5},
        {"swiglu_limit", 10.0},
        {"n_hash_layers", 1},
        {"sliding_window", 8},
        {"rope_theta", 10'000.0},
        {
            "rope_scaling",
            {
                {"type", "yarn"},
                {"factor", 2.0},
                {"beta_fast", 32.0},
                {"beta_slow", 1.0},
                {"original_max_position_embeddings", 64},
            },
        },
        {"eos_token_id", {2, 3}},
        {"index_n_heads", 4},
        {"index_head_dim", 16},
        {"index_topk", 2},
        {"max_position_embeddings", 128},
        {"hc_mult", 4},
        {"hc_eps", 1e-6},
        {"hc_sinkhorn_iters", 20},
        {"compress_rope_theta", 160'000.0},
        {"compress_ratios", std::move(ratios)},
    };
}

json hf_config() {
    auto config = manifest_config({0, 4, 128});
    return {
        {"model_type", "deepseek_v4"},
        {"num_hidden_layers", config["n_layers"]},
        {"hidden_size", config["hidden"]},
        {"n_routed_experts", config["n_experts"]},
        {"num_experts_per_tok", config["top_k"]},
        {"moe_intermediate_size", config["moe_inter"]},
        {"n_shared_experts", config["n_shared"]},
        {"num_attention_heads", config["n_heads"]},
        {"head_dim", config["head_dim"]},
        {"q_lora_rank", config["q_lora_rank"]},
        {"o_lora_rank", config["o_lora_rank"]},
        {"o_groups", config["o_groups"]},
        {"kv_dim", config["kv_dim"]},
        {"qk_rope_head_dim", config["qk_rope_head_dim"]},
        {"num_key_value_heads", config["n_kv_heads"]},
        {"vocab_size", config["vocab"]},
        {"rms_norm_eps", config["rms_eps"]},
        {"scoring_func", config["scoring_func"]},
        {"norm_topk_prob", config["norm_topk_prob"]},
        {"routed_scaling_factor", config["routed_scaling"]},
        {"swiglu_limit", config["swiglu_limit"]},
        {"num_hash_layers", config["n_hash_layers"]},
        {"sliding_window", config["sliding_window"]},
        {"rope_theta", config["rope_theta"]},
        {"rope_scaling", config["rope_scaling"]},
        {"eos_token_id", 2},
        {"index_n_heads", config["index_n_heads"]},
        {"index_head_dim", config["index_head_dim"]},
        {"index_topk", config["index_topk"]},
        {
            "max_position_embeddings",
            config["max_position_embeddings"],
        },
        {"hc_mult", config["hc_mult"]},
        {"hc_eps", config["hc_eps"]},
        {"hc_sinkhorn_iters", config["hc_sinkhorn_iters"]},
        {
            "compress_rope_theta",
            config["compress_rope_theta"],
        },
        {"compress_ratios", config["compress_ratios"]},
    };
}

template <typename T>
void append_scalar(
    std::vector<std::uint8_t>& output,
    T value) {
    const auto* bytes =
        reinterpret_cast<const std::uint8_t*>(&value);
    output.insert(output.end(), bytes, bytes + sizeof(T));
}

void append_bytes(
    std::vector<std::uint8_t>& output,
    std::string_view value) {
    output.insert(
        output.end(),
        reinterpret_cast<const std::uint8_t*>(value.data()),
        reinterpret_cast<const std::uint8_t*>(
            value.data() + value.size()));
}

std::vector<std::uint8_t> dense_payload(
    const std::vector<std::int64_t>& shape,
    std::size_t item_size) {
    std::vector<std::uint8_t> result;
    append_scalar<std::uint32_t>(
        result,
        static_cast<std::uint32_t>(shape.size()));
    std::size_t elements = 1;
    for (const auto value : shape) {
        append_scalar<std::int64_t>(result, value);
        elements *= static_cast<std::size_t>(value);
    }
    result.resize(result.size() + elements * item_size, 0);
    return result;
}

std::vector<std::uint8_t> cccp_int4_payload(
    const std::vector<std::int64_t>& shape) {
    require(
        shape.size() == 2 &&
            shape[0] > 0 &&
            shape[1] > 0 &&
            shape[1] % 64 == 0,
        "invalid synthetic CCCP-I4 shape");
    const auto rows =
        static_cast<std::uint32_t>(shape[0]);
    const auto columns =
        static_cast<std::int32_t>(shape[1]);
    const auto groups =
        static_cast<std::uint32_t>(columns / 64);

    std::vector<std::uint8_t> result;
    append_bytes(result, "CI41");
    append_scalar<std::uint8_t>(result, 1);
    result.insert(result.end(), 3, 0);
    append_scalar<std::uint32_t>(result, 64);
    append_scalar<std::int32_t>(result, 0);
    append_scalar<std::int32_t>(result, columns);
    append_scalar<std::uint32_t>(result, 2);
    append_scalar<std::int64_t>(result, shape[0]);
    append_scalar<std::int64_t>(result, shape[1]);
    append_scalar<std::uint32_t>(result, rows);
    append_scalar<std::uint32_t>(result, groups);
    result.resize(
        result.size() +
            static_cast<std::size_t>(rows) *
                static_cast<std::size_t>(columns / 2) +
            static_cast<std::size_t>(rows) * groups * 2,
        0);
    return result;
}

std::vector<std::uint8_t> nint_moe_payload(
    const std::vector<std::int64_t>& shape) {
    require(
        shape.size() == 3,
        "invalid synthetic NINTM shape");
    std::vector<std::uint8_t> result;
    append_bytes(result, "NIM2");
    append_scalar<std::uint32_t>(
        result,
        static_cast<std::uint32_t>(shape[0]));
    append_scalar<std::uint32_t>(
        result,
        static_cast<std::uint32_t>(shape[1]));
    append_scalar<std::uint32_t>(
        result,
        static_cast<std::uint32_t>(shape[2]));
    append_scalar<std::uint32_t>(result, 1);
    // Binding inspection is deliberately header-only for potentially huge
    // streamed expert records. Keep one byte after the header so the record
    // cannot be mistaken for a truncated header-only placeholder.
    result.push_back(0);
    return result;
}

struct Record {
    std::string name;
    std::string dtype;
    std::vector<std::uint8_t> payload;
};

Record record_for_binding(
    const DeepseekV4TensorBinding& binding) {
    if (binding.kind ==
        DeepseekV4TensorKind::routed_experts) {
        return {
            binding.name,
            "NINTM",
            nint_moe_payload(binding.shape),
        };
    }
    if (binding.kind ==
        DeepseekV4TensorKind::dense_integer) {
        return {
            binding.name,
            "I32",
            dense_payload(binding.shape, 4),
        };
    }
    if ((binding.name == "embed.weight" ||
         binding.name ==
             "layers.0.attn.wq_a.weight") &&
        binding.shape.size() == 2 &&
        binding.shape[1] % 64 == 0) {
        return {
            binding.name,
            "TPQ-I4G64",
            cccp_int4_payload(binding.shape),
        };
    }
    return {
        binding.name,
        "F16",
        dense_payload(binding.shape, 2),
    };
}

void write_string(
    std::ostream& stream,
    std::string_view value) {
    const auto size =
        static_cast<std::uint32_t>(value.size());
    stream.write(
        reinterpret_cast<const char*>(&size),
        sizeof(size));
    stream.write(
        value.data(),
        static_cast<std::streamsize>(value.size()));
}

template <typename T>
void write_scalar(std::ostream& stream, T value) {
    stream.write(
        reinterpret_cast<const char*>(&value),
        sizeof(value));
}

void write_model(
    const std::filesystem::path& path,
    const json& config_json,
    std::string architecture =
        "deepseek_v4-tpq-mfq",
    std::string_view missing = {},
    std::string_view wrong_shape = {},
    std::string_view wrong_dtype = {}) {
    const auto config =
        DeepseekV4Config::from_json(config_json.dump());
    std::vector<Record> records;
    for (const auto& binding :
         mfq::metal::deepseek_v4_required_bindings(config)) {
        if (binding.name == missing) {
            continue;
        }
        auto record = record_for_binding(binding);
        if (binding.name == wrong_shape) {
            auto shape = binding.shape;
            ++shape.back();
            record.dtype = "F16";
            record.payload = dense_payload(shape, 2);
        }
        if (binding.name == wrong_dtype) {
            record.dtype = "I32";
            record.payload = dense_payload(binding.shape, 4);
        }
        records.push_back(std::move(record));
    }

    const json manifest{
        {"format", "cccp-1"},
        {"config", config_json},
        {"quant", json::object()},
    };
    const std::string source_format =
        json("cccp-1").dump();
    const std::string manifest_text = manifest.dump();

    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "cannot create synthetic DeepSeek-V4 MFQ");
    }
    stream.write("MFQ1", 4);
    write_scalar<std::uint32_t>(stream, 2);
    write_string(stream, architecture);
    write_scalar<std::uint32_t>(stream, 2);
    write_string(stream, "source_format");
    write_string(stream, source_format);
    write_string(stream, "tpq_manifest");
    write_string(stream, manifest_text);
    write_scalar<std::uint32_t>(
        stream,
        static_cast<std::uint32_t>(records.size()));
    for (const auto& record : records) {
        write_string(stream, record.name);
        write_string(stream, record.dtype);
        write_scalar<std::uint64_t>(
            stream,
            static_cast<std::uint64_t>(
                record.payload.size()));
    }
    for (const auto& record : records) {
        stream.write(
            reinterpret_cast<const char*>(
                record.payload.data()),
            static_cast<std::streamsize>(
                record.payload.size()));
    }
}

bool contains(
    const std::vector<std::string>& values,
    std::string_view target) {
    return std::find(
        values.begin(),
        values.end(),
        target) != values.end();
}

void test_manifest_and_hf_normalization() {
    const auto manifest = manifest_config();
    const auto manifest_model =
        DeepseekV4Config::from_json(manifest.dump());
    const auto wrapped_model =
        DeepseekV4Config::from_json(
            json{
                {"format", "cccp-1"},
                {"config", manifest},
            }.dump());
    const auto hf_model =
        DeepseekV4Config::from_json(hf_config().dump());

    require(
        manifest_model.n_layers == 3 &&
            manifest_model.hidden == 64 &&
            manifest_model.compress_ratios ==
                std::vector<std::int64_t>({0, 4, 128}),
        "manifest config normalization mismatch");
    require(
        wrapped_model.compress_ratios ==
            manifest_model.compress_ratios,
        "complete manifest config was not unwrapped");
    require(
        hf_model.n_layers == manifest_model.n_layers &&
            hf_model.hidden == manifest_model.hidden &&
            hf_model.n_experts == manifest_model.n_experts &&
            hf_model.routed_scaling ==
                manifest_model.routed_scaling &&
            hf_model.compress_ratios ==
                manifest_model.compress_ratios &&
            hf_model.eos_token_id ==
                std::vector<std::int64_t>({2}),
        "HF config aliases were not normalized");
    require(
        hf_model.rope_scaling.enabled &&
            hf_model.rope_scaling.type == "yarn" &&
            hf_model.rope_scaling.factor == 2.0,
        "HF Yarn config normalization mismatch");

    auto zero_ratio = manifest_config({0});
    zero_ratio.erase("compress_ratios");
    const auto default_ratios =
        DeepseekV4Config::from_json(zero_ratio.dump());
    require(
        default_ratios.compress_ratios ==
            std::vector<std::int64_t>({0}),
        "missing compression schedule did not default to zero");
}

void test_config_validation() {
    auto invalid_architecture = hf_config();
    invalid_architecture["model_type"] = "qwen3_5";
    require_throws(
        [&] {
            (void)DeepseekV4Config::from_json(
                invalid_architecture.dump());
        },
        "unsupported",
        "invalid DeepSeek-V4 model_type was accepted");

    auto invalid_ratio = manifest_config();
    invalid_ratio["compress_ratios"] = {0, 8, 128};
    require_throws(
        [&] {
            (void)DeepseekV4Config::from_json(
                invalid_ratio.dump());
        },
        "0, 4, or 128",
        "invalid compression ratio was accepted");

    auto short_schedule = manifest_config();
    short_schedule["compress_ratios"] = {0, 4};
    require_throws(
        [&] {
            (void)DeepseekV4Config::from_json(
                short_schedule.dump());
        },
        "one entry per layer",
        "short compression schedule was accepted");

    auto invalid_hc = manifest_config();
    invalid_hc["hc_mult"] = 2;
    require_throws(
        [&] {
            (void)DeepseekV4Config::from_json(
                invalid_hc.dump());
        },
        "hc_mult=4",
        "unsupported HC layout was accepted");

    auto conflicting_aliases = manifest_config();
    conflicting_aliases["hidden_size"] = 65;
    require_throws(
        [&] {
            (void)DeepseekV4Config::from_json(
                conflicting_aliases.dump());
        },
        "conflicting aliases",
        "conflicting manifest/HF aliases were accepted");
}

void test_required_ratio_schedule() {
    const auto config =
        DeepseekV4Config::from_json(
            manifest_config().dump());
    const auto required =
        DeepseekV4TensorNames::required(config);

    require(
        !contains(
            required,
            "layers.0.attn.compressor.wkv.weight"),
        "ratio-0 layer incorrectly requires a compressor");
    require(
        contains(
            required,
            "layers.1.attn.compressor.wkv.weight") &&
            contains(
                required,
                "layers.1.attn.indexer.wq_b.weight") &&
            contains(
                required,
                "layers.1.attn.indexer.compressor.ape"),
        "ratio-4 layer lacks its compressor/Indexer schedule");
    require(
        contains(
            required,
            "layers.2.attn.compressor.wkv.weight") &&
            !contains(
                required,
                "layers.2.attn.indexer.wq_b.weight"),
        "ratio-128 layer has the wrong Indexer schedule");
    require(
        contains(
            required,
            "layers.0.ffn.gate.tid2eid") &&
            contains(
                required,
                "layers.1.ffn.gate.bias"),
        "hash/dynamic router schedule mismatch");
}

void test_container_bindings(
    const std::filesystem::path& directory) {
    const auto valid_path = directory / "dsv4-valid.mfq";
    write_model(valid_path, manifest_config());
    const MfqContainer valid(valid_path);
    const auto config = DeepseekV4Config::from_mfq(valid);
    mfq::metal::validate_deepseek_v4_model_bindings(
        valid,
        config);
    const auto embedding =
        mfq::metal::inspect_deepseek_v4_tensor_metadata(
            valid,
            "embed.weight");
    require(
        embedding.dtype == "TPQ-I4G64" &&
            embedding.shape ==
                std::vector<std::int64_t>({32, 64}) &&
            embedding.packed,
        "TPQ-I4 embedding metadata mismatch");

    const auto missing_path = directory / "dsv4-missing.mfq";
    const std::string missing =
        "layers.1.attn.indexer.wq_b.weight";
    write_model(
        missing_path,
        manifest_config(),
        "deepseek_v4-cccp-mfq",
        missing);
    const MfqContainer missing_model(missing_path);
    const auto missing_config =
        DeepseekV4Config::from_mfq(missing_model);
    require_throws(
        [&] {
            mfq::metal::validate_deepseek_v4_model_bindings(
                missing_model,
                missing_config);
        },
        missing,
        "missing ratio-4 Indexer tensor was accepted");

    const auto shape_path = directory / "dsv4-shape.mfq";
    write_model(
        shape_path,
        manifest_config(),
        "deepseek_v4-cccp-mfq",
        {},
        "layers.2.attn.compressor.ape");
    const MfqContainer shape_model(shape_path);
    const auto shape_config =
        DeepseekV4Config::from_mfq(shape_model);
    require_throws(
        [&] {
            mfq::metal::validate_deepseek_v4_model_bindings(
                shape_model,
                shape_config);
        },
        "has shape",
        "wrong compressor shape was accepted");

    const auto dtype_path = directory / "dsv4-dtype.mfq";
    write_model(
        dtype_path,
        manifest_config(),
        "deepseek_v4-cccp-mfq",
        {},
        {},
        "norm.weight");
    const MfqContainer dtype_model(dtype_path);
    const auto dtype_config =
        DeepseekV4Config::from_mfq(dtype_model);
    require_throws(
        [&] {
            mfq::metal::validate_deepseek_v4_model_bindings(
                dtype_model,
                dtype_config);
        },
        "unsupported dtype",
        "integer output norm was accepted");

    const auto architecture_path =
        directory / "dsv4-architecture.mfq";
    write_model(
        architecture_path,
        manifest_config(),
        "qwen3_5-mfq");
    const MfqContainer architecture_model(
        architecture_path);
    require_throws(
        [&] {
            (void)DeepseekV4Config::from_mfq(
                architecture_model);
        },
        "requires architecture",
        "non-DeepSeek MFQ architecture was accepted");
}

} // namespace

int main() {
    const auto directory =
        std::filesystem::temp_directory_path() /
        "mfq-deepseek-v4-model-test";
    try {
        std::filesystem::create_directories(directory);
        test_manifest_and_hf_normalization();
        test_config_validation();
        test_required_ratio_schedule();
        test_container_bindings(directory);
        std::filesystem::remove_all(directory);
        std::cout
            << "MFQ C++ DeepSeek-V4 model foundation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
        std::cerr
            << "MFQ C++ DeepSeek-V4 model test failed: "
            << error.what()
            << '\n';
        return 1;
    }
}
