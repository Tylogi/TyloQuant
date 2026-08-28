#include "mlx_deepseek_v4_causal_lm.h"

#include "../json/nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

namespace {

using json = nlohmann::json;
using mfq::metal::DeepseekV4Config;
using mfq::metal::MlxDeepseekV4Attention;
using mfq::metal::MlxDeepseekV4AttentionComponents;
using mfq::metal::MlxDeepseekV4CausalLm;
using mfq::metal::MlxDeepseekV4Layer;
using mfq::metal::MlxDeepseekV4LayerComponents;
using mfq::metal::MlxDeepseekV4Moe;
using mfq::metal::MlxEmbedding;
using mfq::metal::MlxLinear;
using mfq::metal::MlxRmsNorm;
using mfq::metal::MlxRoutedLinear;
using mlx::core::Shape;
using mlx::core::array;

constexpr int kHidden = 16;
constexpr int kVocab = 8;
constexpr int kExperts = 2;
constexpr int kIntermediate = 16;
constexpr int kLayers = 3;
constexpr int kContext = 32;
constexpr int kNintGroup = 16;

void require(
    bool condition,
    const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_invalid(
    Function&& function,
    const std::string& message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

template <typename Function>
void require_runtime_error(
    Function&& function,
    const std::string& message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, message);
}

template <typename T>
void append(
    std::vector<std::uint8_t>& output,
    T value) {
    const auto* bytes =
        reinterpret_cast<const std::uint8_t*>(
            &value);
    output.insert(
        output.end(),
        bytes,
        bytes + sizeof(value));
}

std::vector<std::uint8_t> zero_nint_blob(
    int output,
    int input) {
    require(
        output > 0 &&
            input > 0 &&
            input % kNintGroup == 0,
        "invalid synthetic NINT shape");
    constexpr int bits = 4;
    constexpr int sub_bits = 2;
    const int groups = input / kNintGroup;
    const std::size_t metadata =
        static_cast<std::size_t>(output) * groups;
    const std::size_t values =
        metadata * kNintGroup;
    std::vector<std::uint8_t> blob;
    append<std::uint8_t>(
        blob,
        static_cast<std::uint8_t>(bits));
    append<std::uint8_t>(
        blob,
        static_cast<std::uint8_t>(sub_bits));
    append<std::int32_t>(
        blob,
        kNintGroup);
    append<std::int32_t>(
        blob,
        0);
    append<std::int32_t>(
        blob,
        input);
    append<std::uint32_t>(
        blob,
        2);
    append<std::int64_t>(
        blob,
        output);
    append<std::int64_t>(
        blob,
        input);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(output));
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(groups));
    for (int row = 0;
         row < 2 * output;
         ++row) {
        append<std::uint16_t>(
            blob,
            0);
    }
    blob.insert(
        blob.end(),
        (metadata * sub_bits + 7) / 8,
        0);
    blob.insert(
        blob.end(),
        (metadata * sub_bits + 7) / 8,
        0);
    blob.insert(
        blob.end(),
        (values * bits + 7) / 8,
        0);
    return blob;
}

std::vector<std::uint8_t> zero_nintm_blob(
    int output_per_expert,
    int input) {
    auto payload = zero_nint_blob(
        kExperts * output_per_expert,
        input);
    const std::string dtype = "NINT4";
    std::vector<std::uint8_t> blob{
        'N', 'I', 'M', '2',
    };
    append<std::uint32_t>(
        blob,
        kExperts);
    append<std::uint32_t>(
        blob,
        output_per_expert);
    append<std::uint32_t>(
        blob,
        input);
    append<std::uint32_t>(
        blob,
        1);
    append<std::uint32_t>(
        blob,
        kExperts);
    append<std::uint32_t>(
        blob,
        static_cast<std::uint32_t>(
            dtype.size()));
    append<std::uint64_t>(
        blob,
        payload.size());
    append<std::uint64_t>(
        blob,
        0);
    for (int expert = 0;
         expert < kExperts;
         ++expert) {
        append<std::int32_t>(
            blob,
            expert);
    }
    blob.insert(
        blob.end(),
        dtype.begin(),
        dtype.end());
    blob.insert(
        blob.end(),
        payload.begin(),
        payload.end());
    return blob;
}

std::vector<std::uint8_t> zero_tpq_nintm_blob(
    int output_per_expert,
    int input) {
    constexpr int vector_size = 8;
    constexpr int entries = 256;
    constexpr int index_bits = 8;
    require(
        output_per_expert > 0 &&
            input > 0 &&
            input % vector_size == 0,
        "invalid synthetic TPQ NINTM shape");
    const std::string dtype = "TPQ-X";
    std::vector<std::uint8_t> blob{
        'N', 'I', 'M', '2',
    };
    append<std::uint32_t>(
        blob,
        kExperts);
    append<std::uint32_t>(
        blob,
        output_per_expert);
    append<std::uint32_t>(
        blob,
        input);
    append<std::uint32_t>(
        blob,
        kExperts);
    for (int expert = 0;
         expert < kExperts;
         ++expert) {
        std::vector<std::uint8_t> payload{
            'C', 'P', 'Q', '1',
        };
        append<std::uint8_t>(payload, 1);
        append<std::uint8_t>(payload, 1);
        append<std::uint8_t>(
            payload,
            vector_size);
        append<std::uint8_t>(
            payload,
            index_bits);
        append<std::int32_t>(
            payload,
            0);
        append<std::int32_t>(
            payload,
            input);
        append<std::uint32_t>(
            payload,
            2);
        append<std::uint32_t>(
            payload,
            entries);
        append<std::int64_t>(
            payload,
            output_per_expert);
        append<std::int64_t>(
            payload,
            input);
        append<std::uint32_t>(
            payload,
            output_per_expert);
        payload.resize(
            payload.size() +
                static_cast<std::size_t>(
                    entries * vector_size) *
                    sizeof(float) +
                static_cast<std::size_t>(
                    output_per_expert) *
                    (input / vector_size),
            0);

        append<std::uint32_t>(
            blob,
            1);
        append<std::uint32_t>(
            blob,
            static_cast<std::uint32_t>(
                dtype.size()));
        append<std::uint64_t>(
            blob,
            payload.size());
        append<std::uint64_t>(
            blob,
            0);
        append<std::int32_t>(
            blob,
            expert);
        blob.insert(
            blob.end(),
            dtype.begin(),
            dtype.end());
        blob.insert(
            blob.end(),
            payload.begin(),
            payload.end());
    }
    return blob;
}

std::vector<std::uint8_t> zero_dense_payload(
    const std::vector<std::int64_t>& shape) {
    std::vector<std::uint8_t> payload;
    append<std::uint32_t>(
        payload,
        static_cast<std::uint32_t>(
            shape.size()));
    std::size_t elements = 1;
    for (const auto extent : shape) {
        require(
            extent > 0,
            "invalid synthetic dense shape");
        append<std::int64_t>(
            payload,
            extent);
        elements *=
            static_cast<std::size_t>(
                extent);
    }
    payload.resize(
        payload.size() +
            elements * sizeof(float),
        0);
    return payload;
}

void write_string(
    std::ostream& stream,
    std::string_view value) {
    const auto size =
        static_cast<std::uint32_t>(
            value.size());
    stream.write(
        reinterpret_cast<const char*>(&size),
        sizeof(size));
    stream.write(
        value.data(),
        static_cast<std::streamsize>(
            value.size()));
}

template <typename T>
void write_scalar(
    std::ostream& stream,
    T value) {
    stream.write(
        reinterpret_cast<const char*>(&value),
        sizeof(value));
}

json load_test_manifest_config() {
    return {
        {"model_type", "deepseek_v4"},
        {"n_layers", 1},
        {"hidden", kHidden},
        {"n_experts", kExperts},
        {"top_k", 1},
        {"moe_inter", kIntermediate},
        {"n_shared", 1},
        {"n_heads", 2},
        {"head_dim", 8},
        {"q_lora_rank", 8},
        {"o_lora_rank", 4},
        {"o_groups", 1},
        {"kv_dim", 8},
        {"qk_rope_head_dim", 4},
        {"n_kv_heads", 1},
        {"vocab", kVocab},
        {"rms_eps", 1e-6},
        {"scoring_func", "sqrtsoftplus"},
        {"norm_topk_prob", true},
        {"routed_scaling", 1.0},
        {"swiglu_limit", 0.0},
        {"n_hash_layers", 0},
        {"sliding_window", 4},
        {"rope_theta", 10'000.0},
        {"eos_token_id", json::array()},
        {"index_n_heads", 2},
        {"index_head_dim", 8},
        {"index_topk", 2},
        {"max_position_embeddings", kContext},
        {"hc_mult", 4},
        {"hc_eps", 1e-6},
        {"hc_sinkhorn_iters", 20},
        {"compress_rope_theta", 160'000.0},
        {"compress_ratios", {0}},
    };
}

struct ContainerRecord {
    std::string name;
    std::string dtype;
    std::vector<std::uint8_t> payload;
};

void write_causal_lm_container(
    const std::filesystem::path& path,
    bool streamed_experts = false) {
    const auto config_json =
        load_test_manifest_config();
    const auto config =
        DeepseekV4Config::from_json(
            config_json.dump());
    std::vector<ContainerRecord> records;
    for (const auto& binding :
         mfq::metal::deepseek_v4_required_bindings(
             config)) {
        if (binding.kind ==
            mfq::metal::DeepseekV4TensorKind::
                routed_experts) {
            records.push_back(
                {
                    binding.name,
                    "NINTM",
                    streamed_experts
                    ? zero_tpq_nintm_blob(
                          static_cast<int>(
                              binding.shape.at(1)),
                          static_cast<int>(
                              binding.shape.at(2)))
                    : zero_nintm_blob(
                          static_cast<int>(
                              binding.shape.at(1)),
                          static_cast<int>(
                              binding.shape.at(2))),
                });
        } else {
            records.push_back(
                {
                    binding.name,
                    "F32",
                    zero_dense_payload(
                        binding.shape),
                });
        }
    }
    const json manifest{
        {"format", "tpq-1"},
        {"config", config_json},
        {
            "tiers_per_layer",
            {{"0", "xw"}},
        },
    };
    std::ofstream stream(
        path,
        std::ios::binary);
    require(
        static_cast<bool>(stream),
        "cannot create synthetic causal-LM container");
    stream.write("MFQ1", 4);
    write_scalar<std::uint32_t>(
        stream,
        2);
    write_string(
        stream,
        "deepseek_v4-tpq-mfq");
    write_scalar<std::uint32_t>(
        stream,
        2);
    write_string(
        stream,
        "source_format");
    write_string(
        stream,
        json("tpq-1").dump());
    write_string(
        stream,
        "tpq_manifest");
    write_string(
        stream,
        manifest.dump());
    write_scalar<std::uint32_t>(
        stream,
        static_cast<std::uint32_t>(
            records.size()));
    for (const auto& record : records) {
        write_string(
            stream,
            record.name);
        write_string(
            stream,
            record.dtype);
        write_scalar<std::uint64_t>(
            stream,
            record.payload.size());
    }
    for (const auto& record : records) {
        stream.write(
            reinterpret_cast<const char*>(
                record.payload.data()),
            static_cast<std::streamsize>(
                record.payload.size()));
    }
}

array zeros_matrix(
    int output,
    int input) {
    return mlx::core::zeros(
        Shape{output, input},
        mlx::core::float32);
}

array ones_vector(int size) {
    return mlx::core::ones(
        Shape{size},
        mlx::core::float32);
}

array patterned_matrix(
    int output,
    int input,
    float scale,
    int salt) {
    std::vector<float> values(
        static_cast<std::size_t>(output) * input);
    for (std::size_t index = 0;
         index < values.size();
         ++index) {
        values[index] =
            static_cast<float>(
                static_cast<int>(
                    (index * 7 +
                     static_cast<std::size_t>(salt)) %
                    17) -
                8) *
            scale;
    }
    return array(
        values.begin(),
        Shape{output, input});
}

DeepseekV4Config test_config(
    bool eos_zero = false,
    std::vector<std::int64_t> ratios = {}) {
    DeepseekV4Config config;
    config.n_layers = kLayers;
    config.hidden = kHidden;
    config.n_experts = kExperts;
    config.top_k = 1;
    config.moe_inter = kIntermediate;
    config.n_shared = 1;
    // The production compressor kernels deliberately accept only the
    // DeepSeek-V4 main/index dimensions (512/128). Keep the surrounding
    // hidden and MoE widths small while exercising those real cache shapes.
    config.n_heads = 1;
    config.head_dim = 512;
    config.q_lora_rank = 8;
    config.o_lora_rank = 4;
    config.o_groups = 1;
    config.kv_dim = 512;
    config.qk_rope_head_dim = 64;
    config.n_kv_heads = 1;
    config.vocab = kVocab;
    config.rms_eps = 1e-6;
    config.scoring_func = "sqrtsoftplus";
    config.norm_topk_prob = true;
    config.routed_scaling = 1.0;
    config.swiglu_limit = 0.0;
    config.n_hash_layers = 0;
    config.sliding_window = 4;
    config.rope_theta = 10'000.0;
    config.eos_token_id =
        eos_zero
        ? std::vector<std::int64_t>{0}
        : std::vector<std::int64_t>{};
    config.index_n_heads = 1;
    config.index_head_dim = 128;
    config.index_topk = 2;
    config.max_position_embeddings = kContext;
    config.hc_mult = 4;
    config.hc_eps = 1e-6;
    config.hc_sinkhorn_iters = 20;
    config.compress_rope_theta = 160'000.0;
    config.compress_ratios =
        ratios.empty()
        ? std::vector<std::int64_t>{0, 4, 128}
        : std::move(ratios);
    config.validate();
    return config;
}

MlxDeepseekV4AttentionComponents
attention_components(
    const DeepseekV4Config& config,
    int ratio) {
    const int hidden =
        static_cast<int>(config.hidden);
    const int heads =
        static_cast<int>(config.n_heads);
    const int head_dim =
        static_cast<int>(config.head_dim);
    const int attention = heads * head_dim;
    const int q_rank =
        static_cast<int>(config.q_lora_rank);
    const int groups =
        static_cast<int>(config.o_groups);
    const int o_rank =
        static_cast<int>(config.o_lora_rank);
    MlxDeepseekV4AttentionComponents result{
        MlxLinear(
            zeros_matrix(q_rank, hidden)),
        MlxLinear(
            zeros_matrix(head_dim, hidden)),
        MlxLinear(
            zeros_matrix(attention, q_rank)),
        MlxLinear(
            zeros_matrix(
                groups * o_rank,
                attention / groups)),
        MlxLinear(
            zeros_matrix(
                hidden,
                groups * o_rank)),
        ones_vector(q_rank),
        ones_vector(head_dim),
        mlx::core::zeros(
            Shape{heads},
            mlx::core::float32),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
    };
    if (ratio != 0) {
        const int width =
            head_dim * (ratio == 4 ? 2 : 1);
        result.main_kv.emplace(
            zeros_matrix(width, hidden));
        result.main_gate.emplace(
            zeros_matrix(width, hidden));
        result.main_ape =
            mlx::core::zeros(
                Shape{ratio, width},
                mlx::core::float32);
        result.main_norm =
            ones_vector(head_dim);
    }
    if (ratio == 4) {
        const int index_heads =
            static_cast<int>(
                config.index_n_heads);
        const int index_dim =
            static_cast<int>(
                config.index_head_dim);
        result.index_q_b.emplace(
            zeros_matrix(
                index_heads * index_dim,
                q_rank));
        result.index_kv.emplace(
            zeros_matrix(
                2 * index_dim,
                hidden));
        result.index_gate.emplace(
            zeros_matrix(
                2 * index_dim,
                hidden));
        result.index_weights.emplace(
            zeros_matrix(
                index_heads,
                hidden));
        result.index_ape =
            mlx::core::zeros(
                Shape{4, 2 * index_dim},
                mlx::core::float32);
        result.index_norm =
            ones_vector(index_dim);
    }
    return result;
}

MlxDeepseekV4Attention make_attention(
    const DeepseekV4Config& config,
    int layer,
    int ratio) {
    auto rope_base =
        mfq::metal::deepseek_v4_yarn_tables(
            static_cast<int>(
                config.qk_rope_head_dim),
            kContext,
            static_cast<float>(
                config.rope_theta));
    auto rope_compressed =
        mfq::metal::deepseek_v4_yarn_tables(
            static_cast<int>(
                config.qk_rope_head_dim),
            kContext,
            static_cast<float>(
                config.compress_rope_theta),
            config.rope_scaling);
    return MlxDeepseekV4Attention(
        config,
        layer,
        ratio,
        kContext,
        attention_components(
            config,
            ratio),
        std::move(rope_base),
        std::move(rope_compressed));
}

MlxDeepseekV4Moe make_moe(
    const DeepseekV4Config& config) {
    const auto available =
        mlx::core::ones(
            Shape{kExperts},
            mlx::core::bool_);
    return MlxDeepseekV4Moe(
        config,
        MlxLinear(
            zeros_matrix(
                kExperts,
                kHidden)),
        MlxLinear(
            zeros_matrix(
                kIntermediate,
                kHidden)),
        MlxLinear(
            zeros_matrix(
                kIntermediate,
                kHidden)),
        MlxLinear(
            zeros_matrix(
                kHidden,
                kIntermediate)),
        MlxRoutedLinear::from_blob(
            zero_nintm_blob(
                2 * kIntermediate,
                kHidden)),
        MlxRoutedLinear::from_blob(
            zero_nintm_blob(
                kHidden,
                kIntermediate)),
        mlx::core::zeros(
            Shape{kExperts},
            mlx::core::float32),
        std::nullopt,
        available);
}

MlxDeepseekV4Layer make_layer(
    const DeepseekV4Config& config,
    int index) {
    const int ratio = static_cast<int>(
        config.compress_ratios[
            static_cast<std::size_t>(index)]);
    return MlxDeepseekV4Layer(
        config,
        static_cast<std::size_t>(index),
        {
            make_attention(
                config,
                index,
                ratio),
            make_moe(config),
            ones_vector(kHidden),
            ones_vector(kHidden),
            MlxLinear(
                zeros_matrix(
                    24,
                    4 * kHidden)),
            mlx::core::zeros(
                Shape{24},
                mlx::core::float32),
            mlx::core::ones(
                Shape{3},
                mlx::core::float32),
            MlxLinear(
                zeros_matrix(
                    24,
                    4 * kHidden)),
            mlx::core::zeros(
                Shape{24},
                mlx::core::float32),
            mlx::core::ones(
                Shape{3},
                mlx::core::float32),
        });
}

MlxEmbedding make_embedding() {
    return MlxEmbedding(
        patterned_matrix(
            kVocab,
            kHidden,
            0.04f,
            3));
}

MlxLinear make_output(bool zero) {
    return MlxLinear(
        zero
        ? zeros_matrix(
              kVocab,
              kHidden)
        : patterned_matrix(
              kVocab,
              kHidden,
              0.025f,
              11));
}

MlxDeepseekV4CausalLm make_model(
    bool zero_output = false,
    bool eos_zero = false,
    std::vector<std::int64_t> ratios = {}) {
    auto config =
        test_config(
            eos_zero,
            std::move(ratios));
    std::vector<MlxDeepseekV4Layer> layers;
    layers.reserve(kLayers);
    for (int layer = 0;
         layer < kLayers;
         ++layer) {
        layers.push_back(
            make_layer(config, layer));
    }
    return MlxDeepseekV4CausalLm(
        config,
        make_embedding(),
        std::move(layers),
        ones_vector(kHidden),
        make_output(zero_output),
        MlxLinear(
            zeros_matrix(
                4,
                4 * kHidden)),
        mlx::core::zeros(
            Shape{4},
            mlx::core::float32),
        mlx::core::ones(
            Shape{1},
            mlx::core::float32),
        kContext,
        mlx::core::float32);
}

array ids(
    std::initializer_list<int> values) {
    return array(
        values,
        Shape{
            1,
            static_cast<int>(
                values.size()),
        },
        mlx::core::int32);
}

std::vector<float> evaluated(array value) {
    value = mlx::core::astype(
        value,
        mlx::core::float32);
    value.eval();
    return {
        value.data<float>(),
        value.data<float>() + value.size(),
    };
}

void require_close(
    array actual,
    array expected,
    float tolerance,
    const std::string& label) {
    require(
        actual.shape() == expected.shape(),
        label + " shape mismatch");
    const auto actual_values =
        evaluated(std::move(actual));
    const auto expected_values =
        evaluated(std::move(expected));
    float maximum_difference = 0.0f;
    std::size_t maximum_index = 0;
    for (std::size_t index = 0;
         index < actual_values.size();
         ++index) {
        require(
            std::isfinite(actual_values[index]),
            label + " contains a non-finite value");
        const float difference = std::fabs(
            actual_values[index] -
            expected_values[index]);
        if (difference > maximum_difference) {
            maximum_difference = difference;
            maximum_index = index;
        }
    }
    if (maximum_difference > tolerance) {
        throw std::runtime_error(
            label + " maximum mismatch at " +
            std::to_string(maximum_index) +
            ": difference=" +
            std::to_string(maximum_difference) +
            " actual=" +
            std::to_string(
                actual_values[maximum_index]) +
            " expected=" +
            std::to_string(
                expected_values[maximum_index]));
    }
}

array last_logits(const array& logits) {
    return mlx::core::slice(
        logits,
        Shape{
            0,
            logits.shape(1) - 1,
            0,
        },
        Shape{
            logits.shape(0),
            logits.shape(1),
            logits.shape(2),
        });
}

array manual_logits(
    const array& token_ids,
    const DeepseekV4Config& config) {
    auto hidden = make_embedding()(
        token_ids,
        mlx::core::float32);
    hidden = hidden *
        (4.0f *
         (0.5f +
          static_cast<float>(
              config.hc_eps)));
    return make_output(false)(
        MlxRmsNorm(
            ones_vector(kHidden),
            static_cast<float>(
                config.rms_eps))(
            hidden));
}

array token_slice(
    const array& input,
    int begin,
    int end) {
    Shape start(input.ndim(), 0);
    Shape stop = input.shape();
    start[1] = begin;
    stop[1] = end;
    return mlx::core::slice(
        input,
        std::move(start),
        std::move(stop));
}

void test_layer_schedule_and_manual_reference() {
    const auto token_ids = ids({1, 4, 2});
    auto model = make_model();
    require(
        model.layer_count() == kLayers &&
            model.config().compress_ratios ==
                std::vector<std::int64_t>(
                    {0, 4, 128}),
        "DeepSeek-V4 layer schedule mismatch");
    require(
        !model.uses_streamed_experts() &&
            model.expert_cache_limit_bytes() == 0 &&
            model.expert_resident_packed_bytes() == 0 &&
            model.cached_expert_count() == 0,
        "injected DeepSeek-V4 model unexpectedly owns "
        "an expert residency");

    auto logits = model.forward(
        token_ids,
        false);
    require(
        logits.shape() ==
            Shape{1, 3, kVocab},
        "DeepSeek-V4 full logits shape mismatch");
    require(
        model.cache_position() == 3 &&
            model.cache_batch() == 1,
        "DeepSeek-V4 fresh forward cache mismatch");

    auto expected = manual_logits(
        token_ids,
        model.config());
    require_close(
        std::move(logits),
        std::move(expected),
        2.0e-4f,
        "DeepSeek-V4 manual identity-layer reference");

    const auto& local =
        model.layer_state(0);
    const auto& compressed =
        model.layer_state(1);
    const auto& very_compressed =
        model.layer_state(2);
    require(
        !local.main().has_value() &&
            !local.indexer().has_value() &&
            compressed.main().has_value() &&
            compressed.indexer().has_value() &&
            compressed.main()->ratio() == 4 &&
            very_compressed.main().has_value() &&
            !very_compressed.indexer().has_value() &&
            very_compressed.main()->ratio() == 128,
        "DeepSeek-V4 ratio 0/4/128 cache topology mismatch");
}

void test_prefill_decode_and_chunking() {
    const auto all_ids = ids({2, 5, 3});
    {
        auto config = test_config(
            false,
            {0, 0, 0});
        auto full_layer =
            make_layer(config, 0);
        auto split_layer =
            make_layer(config, 0);
        auto full_state =
            mfq::metal::MlxDeepseekV4LayerState::allocate(
                config,
                0,
                1,
                kContext,
                mlx::core::float32);
        auto split_state =
            mfq::metal::MlxDeepseekV4LayerState::allocate(
                config,
                0,
                1,
                kContext,
                mlx::core::float32);
        auto embedded = make_embedding()(
            all_ids,
            mlx::core::float32);
        auto streams = mlx::core::contiguous(
            mlx::core::broadcast_to(
                mlx::core::expand_dims(
                    embedded,
                    2),
                Shape{
                    1,
                    3,
                    4,
                    kHidden,
                }));
        auto full_hidden = full_layer(
            streams,
            all_ids,
            full_state,
            0);
        (void)split_layer(
            token_slice(streams, 0, 2),
            token_slice(all_ids, 0, 2),
            split_state,
            0);
        auto split_hidden = split_layer(
            token_slice(streams, 2, 3),
            token_slice(all_ids, 2, 3),
            split_state,
            2);
        require_close(
            split_hidden,
            token_slice(full_hidden, 2, 3),
            2.0e-6f,
            "DeepSeek-V4 layer prefill/decode");
        require_close(
            std::move(split_hidden),
            token_slice(streams, 2, 3),
            2.0e-6f,
            "DeepSeek-V4 zero-branch layer identity");
    }
    {
        auto local_full_model =
            make_model(
                false,
                false,
                {0, 0, 0});
        auto local_full =
            local_full_model.forward(
                all_ids,
                false);
        auto local_decode_model =
            make_model(
                false,
                false,
                {0, 0, 0});
        (void)local_decode_model.prefill(
            ids({2, 5}),
            1,
            true);
        auto local_decoded =
            local_decode_model.decode(
                ids({3}));
        auto local_full_reference =
            manual_logits(
                all_ids,
                local_full_model.config());
        auto local_decode_reference =
            manual_logits(
                ids({3}),
                local_full_model.config());
        require_close(
            local_full,
            local_full_reference,
            2.0e-4f,
            "DeepSeek-V4 local one-shot direct head");
        require_close(
            local_decoded,
            local_decode_reference,
            2.0e-4f,
            "DeepSeek-V4 local decode direct head");
        // MLX selects different fast RMSNorm/matmul kernels for the
        // three-row prefill head and the one-row decode head. Verify both
        // against their exact direct operator path above, then bound only
        // that independently reproduced GEMM/GEMV rounding difference.
        require_close(
            local_decode_reference,
            last_logits(local_full_reference),
            1.0e-3f,
            "DeepSeek-V4 direct head GEMV/GEMM envelope");
        require_close(
            std::move(local_decoded),
            last_logits(local_full),
            1.0e-3f,
            "DeepSeek-V4 local-only prefill/decode");
    }
    auto full_model = make_model();
    auto full = full_model.forward(
        all_ids,
        false);

    auto decode_model = make_model();
    auto prefix = decode_model.prefill(
        ids({2, 5}),
        1,
        true);
    auto decoded = decode_model.decode(
        ids({3}));
    mlx::core::eval(
        full,
        prefix,
        decoded);
    auto reference = manual_logits(
        all_ids,
        full_model.config());
    auto decode_reference = manual_logits(
        ids({3}),
        full_model.config());
    require_close(
        full,
        reference,
        2.0e-4f,
        "DeepSeek-V4 one-shot manual reference");
    require_close(
        decoded,
        decode_reference,
        2.0e-4f,
        "DeepSeek-V4 decode manual reference");
    require_close(
        std::move(decoded),
        last_logits(full),
        1.0e-3f,
        "DeepSeek-V4 prefill/decode");
    require(
        full_model.cache_position() == 3 &&
            decode_model.cache_position() == 3,
        "DeepSeek-V4 decode cache position mismatch");

    auto chunked_model = make_model();
    auto chunked = chunked_model.prefill(
        ids({2, 5, 3}),
        1,
        true);
    auto direct_chunked =
        mlx::core::concatenate(
            {
                manual_logits(
                    ids({2}),
                    chunked_model.config()),
                manual_logits(
                    ids({5}),
                    chunked_model.config()),
                manual_logits(
                    ids({3}),
                    chunked_model.config()),
            },
            1);
    require_close(
        chunked,
        direct_chunked,
        2.0e-4f,
        "DeepSeek-V4 chunked direct head");
    require_close(
        direct_chunked,
        reference,
        1.5e-3f,
        "DeepSeek-V4 direct chunked/full head envelope");
    require_close(
        std::move(chunked),
        std::move(full),
        1.5e-3f,
        "DeepSeek-V4 chunked prefill");

    auto final_only_model = make_model();
    auto final_only =
        final_only_model.prefill(
            ids({2, 5, 3}),
            2,
            false);
    require(
        final_only.shape() ==
            Shape{1, kVocab},
        "DeepSeek-V4 final-only prefill shape mismatch");
    require_close(
        final_only,
        mlx::core::reshape(
            manual_logits(
                ids({3}),
                final_only_model.config()),
            Shape{1, kVocab}),
        2.0e-4f,
        "DeepSeek-V4 final-only direct head");
    require_close(
        mlx::core::reshape(
            std::move(final_only),
            Shape{1, 1, kVocab}),
        last_logits(chunked_model.forward(
            ids({2, 5, 3}),
            false)),
        1.0e-3f,
        "DeepSeek-V4 final-only prefill");

    decode_model.reset_cache(2);
    require(
        decode_model.cache_position() == 0 &&
            decode_model.cache_batch() == 2,
        "DeepSeek-V4 reset_cache mismatch");
    decode_model.clear_cache();
    require(
        !decode_model.cache_ready() &&
            decode_model.cache_position() == 0,
        "DeepSeek-V4 clear_cache mismatch");
}

void test_generation_eos_and_callback() {
    mfq::metal::MlxSamplingParams sampling;
    sampling.temperature = 0.0;
    {
        auto model = make_model(true, true);
        std::vector<std::int64_t> emitted;
        int prefill_calls = 0;
        std::size_t prefill_tokens = 0;
        double prefill_ms = -1.0;
        const auto count = model.generate(
            {1, 2},
            sampling,
            4,
            [&](std::int64_t token) {
                emitted.push_back(token);
                return true;
            },
            std::nullopt,
            512,
            [&](std::size_t tokens, double elapsed_ms) {
                ++prefill_calls;
                prefill_tokens = tokens;
                prefill_ms = elapsed_ms;
            });
        require(
            count == 1 &&
                emitted ==
                    std::vector<std::int64_t>{0},
            "DeepSeek-V4 configured EOS did not stop generation");
        require(
            prefill_calls == 1 &&
                prefill_tokens == 2 &&
                prefill_ms >= 0.0,
            "DeepSeek-V4 prefill callback mismatch");
        require(
            model.cache_position() == 2,
            "DeepSeek-V4 EOS performed an extra decode");
    }
    {
        auto model = make_model(true, true);
        std::vector<std::int64_t> emitted;
        const auto count = model.generate(
            {1, 2},
            sampling,
            3,
            [&](std::int64_t token) {
                emitted.push_back(token);
                return true;
            },
            std::vector<std::int64_t>{});
        require(
            count == 3 &&
                emitted ==
                    std::vector<std::int64_t>(
                        {0, 0, 0}),
            "DeepSeek-V4 EOS override mismatch");
        require(
            model.cache_position() == 4,
            "DeepSeek-V4 generation cache progression mismatch");
    }
    {
        auto model = make_model(true, false);
        const auto count = model.generate(
            {1, 2},
            sampling,
            3,
            [](std::int64_t) {
                return false;
            });
        require(
            count == 1 &&
                model.cache_position() == 2,
            "DeepSeek-V4 callback stop performed an extra decode");
    }
}

void test_generation_stable_prefix_cache() {
    mfq::metal::MlxSamplingParams sampling;
    sampling.temperature = 0.0;

    auto cached = make_model(true, true);
    std::size_t first_prefill_tokens = 0;
    (void)cached.generate(
        {1, 2, 3},
        sampling,
        2,
        {},
        std::nullopt,
        512,
        [&](std::size_t tokens, double) {
            first_prefill_tokens = tokens;
        },
        2);
    require(
        first_prefill_tokens == 3 &&
            cached.cache_position() == 2,
        "DeepSeek-V4 initial stable cache checkpoint mismatch");

    std::size_t reused_prefill_tokens = 0;
    std::vector<std::int64_t> cached_output;
    (void)cached.generate(
        {1, 2, 4, 5},
        sampling,
        1,
        [&](std::int64_t token) {
            cached_output.push_back(token);
            return true;
        },
        std::vector<std::int64_t>{},
        512,
        [&](std::size_t tokens, double) {
            reused_prefill_tokens = tokens;
        },
        3);
    require(
        reused_prefill_tokens == 2 &&
            cached.cache_position() == 3,
        "DeepSeek-V4 stable prefix was not reused");

    auto fresh = make_model(true, true);
    std::vector<std::int64_t> fresh_output;
    (void)fresh.generate(
        {1, 2, 4, 5},
        sampling,
        1,
        [&](std::int64_t token) {
            fresh_output.push_back(token);
            return true;
        },
        std::vector<std::int64_t>{});
    require(
        cached_output == fresh_output,
        "DeepSeek-V4 reused prefix changed sampled output");

    std::size_t fallback_prefill_tokens = 0;
    (void)cached.generate(
        {1, 7, 4, 5},
        sampling,
        1,
        {},
        std::vector<std::int64_t>{},
        512,
        [&](std::size_t tokens, double) {
            fallback_prefill_tokens = tokens;
        },
        3);
    require(
        fallback_prefill_tokens == 4,
        "DeepSeek-V4 mismatched prefix did not fall back to full prefill");

    auto interrupted = make_model();
    (void)interrupted.generate(
        {1, 2, 3},
        sampling,
        3,
        [](std::int64_t) {
            return false;
        },
        std::vector<std::int64_t>{},
        512,
        {},
        2);
    require(
        interrupted.cache_position() == 2,
        "DeepSeek-V4 interrupted generation lost its stable checkpoint");
    std::size_t interrupted_prefill_tokens = 0;
    (void)interrupted.generate(
        {1, 2, 6},
        sampling,
        1,
        {},
        std::vector<std::int64_t>{},
        512,
        [&](std::size_t tokens, double) {
            interrupted_prefill_tokens = tokens;
        },
        2);
    require(
        interrupted_prefill_tokens == 1,
        "DeepSeek-V4 interrupted request checkpoint was not reusable");

    // A client normally disconnects after several decoded tokens, not before
    // the first decode. Exercise enough steps to overwrite the sliding ring
    // and advance both compressor variants, then require the restored prefix
    // to match a completely fresh runtime.
    auto late_interrupted = make_model();
    int observed = 0;
    (void)late_interrupted.generate(
        {1, 2, 3},
        sampling,
        8,
        [&](std::int64_t) {
            return ++observed < 7;
        },
        std::vector<std::int64_t>{},
        512,
        {},
        2);
    require(
        observed == 7 && late_interrupted.cache_position() == 2,
        "DeepSeek-V4 late interruption did not restore its checkpoint");
    std::vector<std::int64_t> restored_output;
    std::size_t late_interrupted_prefill_tokens = 0;
    (void)late_interrupted.generate(
        {1, 2, 6},
        sampling,
        4,
        [&](std::int64_t token) {
            restored_output.push_back(token);
            return true;
        },
        std::vector<std::int64_t>{},
        512,
        [&](std::size_t tokens, double) {
            late_interrupted_prefill_tokens = tokens;
        },
        2);
    auto late_fresh = make_model();
    std::vector<std::int64_t> late_fresh_output;
    (void)late_fresh.generate(
        {1, 2, 6},
        sampling,
        4,
        [&](std::int64_t token) {
            late_fresh_output.push_back(token);
            return true;
        },
        std::vector<std::int64_t>{});
    require(
        late_interrupted_prefill_tokens == 1,
        "DeepSeek-V4 late interruption lost its checkpoint");
    require(
        restored_output == late_fresh_output,
        "DeepSeek-V4 interrupted decode corrupted the stable cache");

    // The compressor pool is an in-place fixed-capacity cache. Preserve a
    // compact copy of its live prefix while a long response appends rows,
    // then verify a cancelled reroll still reuses that prefix exactly.
    const std::vector<std::int64_t> long_prompt{
        1, 2, 3, 4, 5, 6, 7, 1, 2, 3};
    const std::vector<std::int64_t> long_reroll{
        1, 2, 3, 4, 5, 6, 7, 1, 4, 5};
    auto pool_interrupted = make_model();
    int pool_observed = 0;
    (void)pool_interrupted.generate(
        long_prompt,
        sampling,
        16,
        [&](std::int64_t) {
            return ++pool_observed < 15;
        },
        std::vector<std::int64_t>{},
        512,
        {},
        8);
    std::size_t pool_reroll_prefill_tokens = 0;
    std::vector<std::int64_t> pool_reroll_output;
    (void)pool_interrupted.generate(
        long_reroll,
        sampling,
        4,
        [&](std::int64_t token) {
            pool_reroll_output.push_back(token);
            return true;
        },
        std::vector<std::int64_t>{},
        512,
        [&](std::size_t tokens, double) {
            pool_reroll_prefill_tokens = tokens;
        },
        8);
    auto pool_fresh = make_model();
    std::vector<std::int64_t> pool_fresh_output;
    (void)pool_fresh.generate(
        long_reroll,
        sampling,
        4,
        [&](std::int64_t token) {
            pool_fresh_output.push_back(token);
            return true;
        },
        std::vector<std::int64_t>{});
    require(
        pool_reroll_prefill_tokens == 2,
        "DeepSeek-V4 compact pool prefix was not reused");
    require(
        pool_reroll_output == pool_fresh_output,
        "DeepSeek-V4 compact pool prefix changed reroll output");

    // A max_tokens boundary discards only the response suffix as well.
    auto length_truncated = make_model();
    (void)length_truncated.generate(
        {1, 2, 3},
        sampling,
        8,
        {},
        std::vector<std::int64_t>{},
        512,
        {},
        2);
    std::size_t truncated_prefill_tokens = 0;
    std::vector<std::int64_t> truncated_output;
    (void)length_truncated.generate(
        {1, 2, 6},
        sampling,
        4,
        [&](std::int64_t token) {
            truncated_output.push_back(token);
            return true;
        },
        std::vector<std::int64_t>{},
        512,
        [&](std::size_t tokens, double) {
            truncated_prefill_tokens = tokens;
        },
        2);
    auto truncated_fresh = make_model();
    std::vector<std::int64_t> truncated_fresh_output;
    (void)truncated_fresh.generate(
        {1, 2, 6},
        sampling,
        4,
        [&](std::int64_t token) {
            truncated_fresh_output.push_back(token);
            return true;
        },
        std::vector<std::int64_t>{});
    require(
        truncated_prefill_tokens == 1,
        "DeepSeek-V4 length-truncated request lost its prefix checkpoint");
    require(
        truncated_output == truncated_fresh_output,
        "DeepSeek-V4 length-truncated reroll differs from a fresh runtime");
}

void test_text_session_snapshot_restore() {
    mfq::metal::MlxSamplingParams sampling;
    sampling.temperature = 0.0;

    auto cached = make_model(true, true);
    auto prefix_logits = cached.prefill(ids({1, 2}));
    prefix_logits.eval();
    const auto snapshot =
        cached.capture_text_session_state({1, 2});
    require(
        snapshot.tokens == std::vector<std::int64_t>({1, 2}) &&
            snapshot.cache_position == 2 &&
            snapshot.cache_batch == 1 &&
            snapshot.layers.size() == cached.layer_count() &&
            snapshot.bytes > 0,
        "DeepSeek-V4 text session snapshot metadata mismatch");
    for (const auto& layer : snapshot.layers) {
        const auto require_compact_pool = [](const auto& pool) {
            if (pool) {
                require(
                    pool->pool().size() == 1,
                    "DeepSeek-V4 session snapshot retained a full cache pool");
            }
        };
        require_compact_pool(layer.main());
        require_compact_pool(layer.indexer());
    }

    auto discarded = cached.decode(ids({3, 4, 5}));
    discarded.eval();
    cached.restore_text_session_state(snapshot);
    require(
        cached.cache_position() == 2 && cached.cache_batch() == 1,
        "DeepSeek-V4 text session restore position mismatch");

    std::size_t reused_prefill_tokens = 0;
    std::vector<std::int64_t> restored_output;
    (void)cached.generate(
        {1, 2, 6},
        sampling,
        4,
        [&](std::int64_t token) {
            restored_output.push_back(token);
            return true;
        },
        std::vector<std::int64_t>{},
        512,
        [&](std::size_t tokens, double) {
            reused_prefill_tokens = tokens;
        },
        2);
    require(
        reused_prefill_tokens == 1 && cached.cache_position() == 2,
        "DeepSeek-V4 restored session did not evaluate only the suffix");

    auto fresh = make_model(true, true);
    std::vector<std::int64_t> fresh_output;
    (void)fresh.generate(
        {1, 2, 6},
        sampling,
        4,
        [&](std::int64_t token) {
            fresh_output.push_back(token);
            return true;
        },
        std::vector<std::int64_t>{});
    require(
        restored_output == fresh_output,
        "DeepSeek-V4 restored session changed generated tokens");

    cached.restore_text_session_state(snapshot);
    std::vector<std::int64_t> repeated_output;
    (void)cached.generate(
        {1, 2, 6},
        sampling,
        4,
        [&](std::int64_t token) {
            repeated_output.push_back(token);
            return true;
        },
        std::vector<std::int64_t>{},
        512,
        {},
        2);
    require(
        repeated_output == fresh_output,
        "DeepSeek-V4 text session snapshot was mutated by resumed decode");
}

void test_mfq_container_load() {
    const auto path =
        std::filesystem::temp_directory_path() /
        "mfq-dsv4-causal-lm-test.mfq";
    write_causal_lm_container(path);
    try {
        const mfq::metal::MfqContainer model(path);
        auto runtime =
            MlxDeepseekV4CausalLm::load(
                model,
                16,
                12'345);
        require(
            runtime.layer_count() == 1 &&
                runtime.max_context() == 16 &&
                runtime.expert_cache_limit_bytes() ==
                    12'345 &&
                !runtime.uses_streamed_experts(),
            "container-loaded DeepSeek-V4 runtime metadata "
            "mismatch");
        auto logits = runtime.prefill(
            ids({1, 2}),
            1,
            true);
        require(
            logits.shape() ==
                Shape{1, 2, kVocab} &&
                runtime.cache_position() == 2,
            "container-loaded DeepSeek-V4 forward/cache "
            "mismatch");
        require_close(
            std::move(logits),
            mlx::core::zeros(
                Shape{1, 2, kVocab},
                mlx::core::float32),
            1.0e-6f,
            "container-loaded DeepSeek-V4 zero model");
        runtime.clear_expert_cache();
        require(
            runtime.expert_resident_packed_bytes() == 0 &&
                runtime.cached_expert_count() == 0,
            "container-loaded DeepSeek-V4 expert cache "
            "did not clear");
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(
            path,
            ignored);
        throw;
    }
    std::error_code ignored;
    std::filesystem::remove(
        path,
        ignored);
}

void test_streamed_mfq_container_lifetime() {
    const auto path =
        std::filesystem::temp_directory_path() /
        "mfq-dsv4-causal-lm-lifetime-test.mfq";
    write_causal_lm_container(
        path,
        true);
    try {
        std::optional<MlxDeepseekV4CausalLm> runtime;
        {
            const mfq::metal::MfqContainer model(path);
            runtime.emplace(
                MlxDeepseekV4CausalLm::load(
                    model,
                    16,
                    1U << 20));
            require(
                runtime->uses_streamed_experts(),
                "container-loaded DeepSeek-V4 runtime did not "
                "retain streamed experts");
        }

        auto logits = runtime->prefill(
            ids({1, 2}),
            1,
            true);
        require_close(
            std::move(logits),
            mlx::core::zeros(
                Shape{1, 2, kVocab},
                mlx::core::float32),
            1.0e-6f,
            "streamed DeepSeek-V4 after source-container "
            "destruction");
        require(
            runtime->cache_position() == 2 &&
                runtime->expert_resident_packed_bytes() > 0 &&
                runtime->cached_expert_count() > 0,
            "streamed DeepSeek-V4 did not preserve forward/cache "
            "state after source-container destruction");
        runtime->clear_expert_cache();
        require(
            runtime->expert_resident_packed_bytes() == 0 &&
                runtime->cached_expert_count() == 0,
            "streamed DeepSeek-V4 lifetime-test cache did not "
            "clear");
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(
            path,
            ignored);
        throw;
    }
    std::error_code ignored;
    std::filesystem::remove(
        path,
        ignored);
}

void test_validation() {
    {
        auto model = make_model();
        require_runtime_error(
            [&] {
                (void)model.decode(ids({1}));
            },
            "DeepSeek-V4 decode without prefill was accepted");
    }
    {
        auto model = make_model();
        const std::vector<float> values{
            1.0f, 2.0f,
        };
        require_invalid(
            [&] {
                (void)model.forward(
                    array(
                        values.begin(),
                        Shape{1, 2}),
                    false);
            },
            "DeepSeek-V4 floating token IDs were accepted");
    }
    {
        auto model = make_model();
        require_invalid(
            [&] {
                (void)model.prefill(
                    ids({1}),
                    0,
                    true);
            },
            "DeepSeek-V4 zero prefill chunk was accepted");
    }
    {
        auto model = make_model();
        mfq::metal::MlxSamplingParams sampling;
        require_invalid(
            [&] {
                (void)model.generate(
                    {},
                    sampling,
                    1);
            },
            "DeepSeek-V4 empty generation prompt was accepted");
        require_invalid(
            [&] {
                (void)model.generate(
                    {1},
                    sampling,
                    kContext,
                    {},
                    std::vector<std::int64_t>{});
            },
            "DeepSeek-V4 over-context generation was accepted");
    }
}

} // namespace

int main() {
    try {
        test_layer_schedule_and_manual_reference();
        test_prefill_decode_and_chunking();
        test_generation_eos_and_callback();
        test_generation_stable_prefix_cache();
        test_text_session_snapshot_restore();
        test_mfq_container_load();
        test_streamed_mfq_container_lifetime();
        test_validation();
        std::cout
            << "DeepSeek-V4 C++ layer/causal-LM/generation "
               "tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "DeepSeek-V4 C++ causal-LM test failed: "
            << error.what()
            << '\n';
        return 1;
    }
}
