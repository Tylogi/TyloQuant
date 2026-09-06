#include "qwen35_model.h"
#include "mlx_legacy_tensor_compat.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

void test_synthetic_config() {
    const auto config = mfq::metal::Qwen35Config::from_json(R"JSON(
{
  "model_type": "qwen3_5",
  "image_token_id": 248056,
  "video_token_id": 248057,
  "vision_start_token_id": 248053,
  "vision_end_token_id": 248054,
  "vision_config": {
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "depth": 27,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "out_hidden_size": 4096,
    "num_position_embeddings": 2304
  },
  "text_config": {
    "model_type": "qwen3_5_text",
    "vocab_size": 248320,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 2,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 262144,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": false,
    "attn_output_gate": true,
    "output_gate_type": "swish",
    "layer_types": ["linear_attention", "full_attention"],
    "full_attention_interval": 2,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 8,
    "linear_num_value_heads": 32,
    "rope_parameters": {
      "full_attention": {
        "rope_theta": 1000000.0,
        "partial_rotary_factor": 0.5,
        "mrope_section": [16, 8, 8]
      }
    }
  }
}
)JSON");

    require(config.model_type == "qwen3_5", "model type mismatch");
    require(
        config.text_model_type == "qwen3_5_text",
        "text model type mismatch");
    require(config.head_dim == 128, "head dimension mismatch");
    require(config.rotary_dim == 64, "rotary dimension mismatch");
    require(config.attention_size() == 4096, "attention size mismatch");
    require(
        config.query_projection_size() == 8192,
        "gated query projection size mismatch");
    require(config.kv_size() == 1024, "KV size mismatch");
    require(config.linear_key_size() == 1024, "linear key size mismatch");
    require(
        config.linear_value_size() == 4096,
        "linear value size mismatch");
    require(
        config.linear_qkv_size() == 6144,
        "linear QKV size mismatch");
    require(config.rope_sections.size() == 3, "RoPE sections mismatch");
    require(config.attention_output_gate, "attention gate mismatch");
    require(
        config.output_gate_type == "swish",
        "attention gate type mismatch");
    require(
        config.layer_types.at(0) == "linear_attention",
        "layer type mismatch");
    require(config.has_vision(), "vision config missing");
    require(config.vision->depth == 27, "vision depth mismatch");
    require(config.vision->head_dim() == 72, "vision head dimension mismatch");
    require(
        config.image_token_id == std::optional<std::int64_t>(248056),
        "image token ID mismatch");
    require(
        config.video_token_id == std::optional<std::int64_t>(248057),
        "video token ID mismatch");

    const auto names = mfq::metal::Qwen35TensorNames::canonical();
    const auto layer = names.layer(7);
    require(
        layer.attention_query ==
            "model.block.7.attention.query.weight",
        "canonical attention name mismatch");
    require(
        layer.linear_conv_bias ==
            std::optional<std::string>(
                "model.block.7.linear_attention.conv.bias"),
        "canonical conv bias name mismatch");

    const auto adapted =
        mfq::metal::adapt_qwen35_config_for_storage(
            config,
            true);
    require(
        adapted.norm_weight_offset == 0.0,
        "GGUF norm offset adaptation mismatch");
    require(
        !adapted.linear_a_is_log,
        "GGUF SSM decay adaptation mismatch");
    require(
        adapted.linear_qkv_gguf_layout,
        "GGUF combined-QKV layout adaptation mismatch");
    require(
        mfq::metal::adapt_qwen35_config_for_storage(config, false)
                .norm_weight_offset == config.norm_weight_offset,
        "canonical storage semantics changed norm weights");
}

void test_real_model(const std::string& path) {
    mfq::metal::MfqContainer model(path);
    mfq::metal::install_legacy_tensor_compatibility(model);
    const auto config = mfq::metal::Qwen35Config::from_mfq(model);
    const auto names = mfq::metal::Qwen35TensorNames::canonical();

    require(config.model_type == "qwen3_5", "real model type mismatch");
    require(
        config.text_model_type == "qwen3_5_text",
        "real text model type mismatch");
    require(config.vocab_size == 248320, "real vocabulary mismatch");
    require(config.hidden_size == 5120, "real hidden size mismatch");
    require(config.num_hidden_layers == 64, "real layer count mismatch");
    require(config.head_dim == 256, "real attention head dimension mismatch");
    require(config.rotary_dim == 64, "real rotary dimension mismatch");
    require(config.attention_size() == 6144, "real attention size mismatch");
    require(
        config.query_projection_size() == 12288,
        "real gated query projection size mismatch");
    require(config.linear_key_size() == 2048, "real linear key size mismatch");
    require(
        config.linear_value_size() == 6144,
        "real linear value size mismatch");
    require(
        config.linear_qkv_size() == 10240,
        "real linear QKV size mismatch");
    require(config.full_attention_interval == 4, "real attention interval mismatch");
    require(config.mrope_interleaved, "real mRoPE layout mismatch");
    require(config.mtp_num_hidden_layers == 1, "real MTP layer count mismatch");

    const auto linear = names.layer(0);
    require(
        linear.linear_qkv ==
            "model.block.0.linear_attention.qkv.weight",
        "real linear QKV binding mismatch");
    require(
        linear.linear_output ==
            "model.block.0.linear_attention.output.weight",
        "real linear output binding mismatch");
    const auto full = names.layer(3);
    require(
        full.attention_query ==
            "model.block.3.attention.query.weight",
        "real full attention Q binding mismatch");

    mfq::metal::validate_qwen35_model_bindings(model, config, names);

    const auto qkv = mfq::metal::inspect_qwen35_tensor_metadata(
        model,
        linear.linear_qkv);
    require(qkv.dtype == "NINT6", "real linear QKV dtype mismatch");
    require(
        qkv.shape == std::vector<std::int64_t>({10240, 5120}),
        "real linear QKV shape mismatch");
    const auto linear_output =
        mfq::metal::inspect_qwen35_tensor_metadata(
            model,
            linear.linear_output);
    require(
        linear_output.dtype == "NINT4",
        "real linear output dtype mismatch");
    require(
        linear_output.shape ==
            std::vector<std::int64_t>({5120, 6144}),
        "real linear output shape mismatch");
    const auto query = mfq::metal::inspect_qwen35_tensor_metadata(
        model,
        full.attention_query);
    require(query.dtype == "NINT4", "real attention Q dtype mismatch");
    require(
        query.shape == std::vector<std::int64_t>({12288, 5120}),
        "real attention Q shape mismatch");
}

int main(int argc, char** argv) {
    try {
        require(argc <= 2, "usage: qwen35_config_test [model.mfq]");
        test_synthetic_config();
        if (argc == 2) {
            test_real_model(argv[1]);
        }

        std::cout << "Qwen3.5 config/name/metadata mapping test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
