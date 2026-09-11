#include "mfq_model_graph.h"
#include "mfq_legacy_model_graph.h"
#include "mfq_legacy_tensor_names.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void require_invalid(Callable&& callable, const char* message) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        const auto graph = mfq::MfqModelGraph::from_json(R"json({
          "schema_version": 1,
          "architecture": "qwen3_5",
          "canonical_naming": {"namespace": "mfq.tensor", "version": 1, "component_roots": ["model", "vision", "predictor"]},
          "topology": {"text_layers": 64, "vision_layers": 27, "predictor_layers": 1},
          "graph": {"kind": "causal_lm", "backbone": "qwen3_5"},
          "components": [
            {"kind": "text", "tensor_root": "model", "implementation": "qwen3_5", "policy": "decoder"},
            {"kind": "vision", "tensor_root": "vision", "implementation": "grid_vit", "policy": "optional", "input_contract": "grid_vision.v1", "position_policy": "grid_mrope"},
            {"kind": "predictor", "tensor_root": "predictor", "implementation": "next_token_prediction", "policy": "optional"}
          ],
          "capabilities": ["text", "vision", "speculative_prediction"]
        })json");
        require(graph.schema_version == 1, "schema version mismatch");
        require(graph.graph_kind == "causal_lm", "graph kind mismatch");
        require(graph.backbone == "qwen3_5", "backbone mismatch");
        require(graph.tensor_namespace == "mfq.tensor", "namespace mismatch");
        require(graph.topology.text_layers == 64, "text topology mismatch");
        require(graph.has_component("vision"), "Vision component was lost");
        require(graph.component("vision")->implementation == "grid_vit",
                "Vision implementation mismatch");
        require(graph.component("vision")->position_policy == "grid_mrope",
                "position policy mismatch");
        require(graph.has_component("predictor"), "predictor component was lost");
        require(graph.has_capability("speculative_prediction"),
                "predictor capability was lost");

        require_invalid(
            [] {
                (void)mfq::MfqModelGraph::from_json(R"json({
                  "schema_version": 1,
                  "architecture": "qwen3_5",
                  "canonical_naming": {"namespace": "hf.tensor", "version": 1, "component_roots": ["model"]},
                  "graph": {"kind": "causal_lm", "backbone": "qwen3_5"},
                  "topology": {"text_layers": 1},
                  "components": [{"kind": "text", "tensor_root": "model", "implementation": "qwen3_5"}]
                })json");
            },
            "foreign tensor namespace was accepted");
        require_invalid(
            [] {
                (void)mfq::MfqModelGraph::from_json(R"json({
                  "schema_version": 1,
                  "architecture": "qwen3_5",
                  "canonical_naming": {"namespace": "mfq.tensor", "version": 1, "component_roots": ["model"]},
                  "graph": {"kind": "causal_lm", "backbone": "qwen3_5"},
                  "topology": {"text_layers": 1, "vision_layers": 1},
                  "components": [{"kind": "text", "tensor_root": "model", "implementation": "qwen3_5"}]
                })json");
            },
            "missing Vision component was accepted");
        require_invalid(
            [] {
                (void)mfq::MfqModelGraph::from_json(R"json({
                  "schema_version": 1,
                  "architecture": "qwen3_5",
                  "canonical_naming": {"namespace": "mfq.tensor", "version": 1, "component_roots": ["model"]},
                  "graph": {"kind": "causal_lm", "backbone": "qwen3_5"},
                  "topology": {"text_layers": 1},
                  "components": [
                    {"kind": "text", "tensor_root": "model", "implementation": "qwen3_5"},
                    {"kind": "text", "tensor_root": "model", "implementation": "qwen3_5"}
                  ]
                })json");
            },
            "duplicate text component was accepted");

        const std::unordered_set<std::string> legacy_qwen_tensors{
            "model.visual.patch_embed.proj.weight",
            "mtp.fc.weight",
        };
        const auto legacy_qwen = mfq::synthesize_legacy_model_graph(
            "qwen3_8",
            R"json({
              "model_type":"qwen3_5",
              "num_hidden_layers":64,
              "mtp_num_hidden_layers":1,
              "vision_config":{"depth":27}
            })json",
            [&](std::string_view name) {
                return legacy_qwen_tensors.count(std::string(name)) != 0;
            });
        require(
            legacy_qwen.architecture == "qwen3_5" &&
                legacy_qwen.backbone == "qwen3_5",
            "legacy Qwen identity was not normalized");
        require(
            legacy_qwen.has_component("vision") &&
                legacy_qwen.has_component("predictor"),
            "legacy Qwen component inventory was not recovered");
        require(
            legacy_qwen.topology.text_layers == 64 &&
                legacy_qwen.topology.vision_layers == 27 &&
                legacy_qwen.topology.predictor_layers == 1,
            "legacy Qwen topology was not recovered");

        const std::unordered_set<std::string> legacy_v41_tensors{
            "vision.patch_embedding.weight",
            "predictor.stage.0.main_projection.weight",
        };
        const auto legacy_v41 = mfq::synthesize_legacy_model_graph(
            "deepseek_v41",
            R"json({
              "model_type":"deepseek_v41",
              "text_config":{
                "model_type":"deepseek_v41_text",
                "num_hidden_layers":40,
                "num_nextn_predict_layers":3
              },
              "vision_config":{"num_hidden_layers":32}
            })json",
            [&](std::string_view name) {
                return legacy_v41_tensors.count(std::string(name)) != 0;
            });
        require(
            legacy_v41.architecture == "deepseek_v41" &&
                legacy_v41.backbone == "deepseek_v41",
            "legacy DeepSeek-V4.1 was folded into the V4 runtime");
        require(
            legacy_v41.component("vision") != nullptr &&
                legacy_v41.component("vision")->implementation ==
                    "deepseek_v41_vision" &&
                legacy_v41.component("predictor") != nullptr &&
                legacy_v41.component("predictor")->implementation ==
                    "deepseek_v41_dspark",
            "legacy DeepSeek-V4.1 components were not recovered");

        const auto generic = mfq::synthesize_legacy_model_graph(
            "qwen2_legacy",
            R"json({"model_type":"qwen2","num_hidden_layers":32})json",
            [](std::string_view) { return false; });
        require(
            generic.architecture == "qwen2" &&
                generic.backbone == "generic_qwen" &&
                generic.components.size() == 1,
            "legacy generic decoder compatibility escaped normalization");

        const auto aliases = mfq::make_legacy_tensor_aliases(
            "qwen3_8-hf-mfq",
            R"json({"model_type":"qwen3_5","num_hidden_layers":1})json",
            {
                "model.language_model.embed_tokens.weight",
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.language_model.layers.0.mlp.down_proj.weight",
                "model.language_model.layers.0.mlp.down_proj.weight.in_high",
            });
        require(
            aliases.canonical_to_stored.at("model.token_embedding.weight") ==
                "model.language_model.embed_tokens.weight",
            "legacy root tensor alias was not canonicalized");
        require(
            aliases.canonical_to_stored.at(
                "model.block.0.attention.query.weight") ==
                "model.language_model.layers.0.self_attn.q_proj.weight",
            "legacy block tensor alias was not canonicalized");
        require(
            aliases.canonical_to_stored.at(
                "model.block.0.mlp.down.weight.in_high") ==
                "model.language_model.layers.0.mlp.down_proj.weight.in_high",
            "derived legacy tensor alias was not canonicalized");

        const auto minicpmo_aliases = mfq::make_legacy_tensor_aliases(
            "minicpmo45",
            R"json({"model_type":"minicpmo","num_hidden_layers":1})json",
            {
                "llm.model.layers.0.self_attn.q_proj.weight",
                "vpm.encoder.layers.0.mlp.fc2.weight",
                "apm.layers.0.self_attn.out_proj.weight",
                "tts.model.layers.0.mlp.gate_proj.weight",
                "tts.head_code.0.parametrizations.weight.original0",
            });
        require(
            minicpmo_aliases.canonical_to_stored.at(
                "model.block.0.attention.query.weight") ==
                "llm.model.layers.0.self_attn.q_proj.weight" &&
            minicpmo_aliases.canonical_to_stored.at(
                "vision.block.0.mlp.down.weight") ==
                "vpm.encoder.layers.0.mlp.fc2.weight" &&
            minicpmo_aliases.canonical_to_stored.at(
                "audio.block.0.attention.output.weight") ==
                "apm.layers.0.self_attn.out_proj.weight" &&
            minicpmo_aliases.canonical_to_stored.at(
                "tts.block.0.mlp.gate.weight") ==
                "tts.model.layers.0.mlp.gate_proj.weight" &&
            minicpmo_aliases.canonical_to_stored.at(
                "tts.code_output.0.weight_norm.magnitude") ==
                "tts.head_code.0.parametrizations.weight.original0",
            "legacy MiniCPM-o component aliases were not canonicalized");

        const auto glm_aliases = mfq::make_legacy_tensor_aliases(
            "glm_moe_dsa",
            R"json({"model_type":"glm_moe_dsa","num_hidden_layers":1})json",
            {
                "model.layers.0.self_attn.q_a_proj.weight",
                "model.layers.0.self_attn.indexer.weights_proj.weight",
                "model.layers.0.mlp.shared_experts.down_proj.weight",
            });
        require(
            glm_aliases.canonical_to_stored.at(
                "model.block.0.attention.query_a.weight") ==
                "model.layers.0.self_attn.q_a_proj.weight" &&
            glm_aliases.canonical_to_stored.at(
                "model.block.0.attention.indexer.score.weight") ==
                "model.layers.0.self_attn.indexer.weights_proj.weight" &&
            glm_aliases.canonical_to_stored.at(
                "model.block.0.mlp.shared_expert.down.weight") ==
                "model.layers.0.mlp.shared_experts.down_proj.weight",
            "legacy GLM DSA aliases were not canonicalized");

        const auto deepseek_aliases = mfq::make_legacy_tensor_aliases(
            "deepseek_v4",
            R"json({"model_type":"deepseek_v4"})json",
            {
                "layers.0.attn.wq_a.weight",
                "vision.blocks.0.attn.wqkv.weight",
                "mtp.0.main_proj.weight",
                "mtp.0.norm.weight",
                "mtp.0.markov_head.markov_w1.weight",
                "mtp.2.norm.weight",
                "layers.0.ffn.gate.bias_vl",
                "mtp.0.ffn.gate.bias_vl",
            });
        require(
            deepseek_aliases.canonical_to_stored.at(
                "model.block.0.attention.query_a.weight") ==
                "layers.0.attn.wq_a.weight" &&
            deepseek_aliases.canonical_to_stored.at(
                "vision.block.0.attention.qkv.weight") ==
                "vision.blocks.0.attn.wqkv.weight" &&
            deepseek_aliases.canonical_to_stored.at(
                "predictor.stage.0.main_projection.weight") ==
                "mtp.0.main_proj.weight" &&
            deepseek_aliases.canonical_to_stored.at(
                "predictor.stage.0.output_norm.weight") ==
                "mtp.0.norm.weight" &&
            deepseek_aliases.canonical_to_stored.at(
                "predictor.stage.0.markov.input.weight") ==
                "mtp.0.markov_head.markov_w1.weight" &&
            deepseek_aliases.canonical_to_stored.at(
                "predictor.stage.2.output_norm.weight") ==
                "mtp.2.norm.weight" &&
            deepseek_aliases.canonical_to_stored.at(
                "model.block.0.mlp.router.vision_bias") ==
                "layers.0.ffn.gate.bias_vl" &&
            deepseek_aliases.canonical_to_stored.at(
                "predictor.stage.0.mlp.router.vision_bias") ==
                "mtp.0.ffn.gate.bias_vl",
            "legacy DeepSeek-V4 aliases were not canonicalized");

        const auto v41_aliases = mfq::make_legacy_tensor_aliases(
            "deepseek_v41",
            R"json({"model_type":"deepseek_v41"})json",
            {
                "layers.0.attn.wq_a.weight",
                "layers.0.ffn.experts.3.w1.weight",
                "layers.0.ffn.experts.3.w1.scale",
                "mtp.0.markov_head.embed.weight",
            });
        require(
            v41_aliases.canonical_to_stored.at(
                "model.block.0.attention.query_a.weight") ==
                "layers.0.attn.wq_a.weight" &&
            v41_aliases.canonical_to_stored.at(
                "model.block.0.mlp.experts.3.gate.weight") ==
                "layers.0.ffn.experts.3.w1.weight" &&
            v41_aliases.canonical_to_stored.at(
                "model.block.0.mlp.experts.3.gate.weight_scale") ==
                "layers.0.ffn.experts.3.w1.scale" &&
            v41_aliases.canonical_to_stored.at(
                "predictor.stage.0.markov.embedding.weight") ==
                "mtp.0.markov_head.embed.weight",
            "legacy DeepSeek-V4.1 aliases were not canonicalized");

        const auto v41_raw_hf_aliases = mfq::make_legacy_tensor_aliases(
            "deepseek_v4_raw_hf",
            R"json({"model_type":"deepseek_v41"})json",
            {
                "layers.0.attn.wkv.weight",
                "layers.0.attn.kv_norm.weight",
            });
        require(
            v41_raw_hf_aliases.canonical_to_stored.at(
                "model.block.0.attention.key_value_a.weight") ==
                "layers.0.attn.wkv.weight" &&
            v41_raw_hf_aliases.canonical_to_stored.at(
                "model.block.0.attention.key_value_a_norm.weight") ==
                "layers.0.attn.kv_norm.weight",
            "raw-HF DeepSeek-V4.1 aliases did not preserve the tuned "
            "Metal contract");

        const auto gemma_aliases = mfq::make_legacy_tensor_aliases(
            "gemma4",
            R"json({"model_type":"gemma4","num_hidden_layers":1})json",
            {
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.language_model.layers.0.router.per_expert_scale",
            });
        require(
            gemma_aliases.canonical_to_stored.at(
                "model.block.0.attention.query.weight") ==
                "model.language_model.layers.0.self_attn.q_proj.weight" &&
            gemma_aliases.canonical_to_stored.at(
                "model.block.0.mlp.router.expert_scale") ==
                "model.language_model.layers.0.router.per_expert_scale",
            "legacy Gemma4 aliases were not canonicalized");

        const auto canonical_wins = mfq::make_legacy_tensor_aliases(
            "qwen3_8-hf-mfq",
            R"json({"model_type":"qwen3_5","num_hidden_layers":1})json",
            {
                "model.block.0.attention.query.weight",
                "model.language_model.layers.0.self_attn.q_proj.weight",
            });
        require(
            canonical_wins.canonical_to_stored.count(
                "model.block.0.attention.query.weight") == 0,
            "legacy alias shadowed a canonical tensor record");

        std::cout << "MFQ model graph tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MFQ model graph test failed: " << error.what() << '\n';
        return 1;
    }
}
