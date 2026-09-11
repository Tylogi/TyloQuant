from __future__ import annotations

import json

import pytest

from mfq.architectures.tensor_schema import (
    TensorComponent,
    graph_spec_for_plan,
    map_source_tensor_name,
    topology_from_config,
)


@pytest.mark.parametrize(
    "source,canonical,component",
    [
        ("embed.weight", "model.token_embedding.weight", TensorComponent.MODEL),
        (
            "layers.7.attn.wq_a.weight",
            "model.block.7.attention.query_a.weight",
            TensorComponent.MODEL,
        ),
        (
            "layers.7.attn.compressor.wkv.weight",
            "model.block.7.attention.compressor.key_value.weight",
            TensorComponent.MODEL,
        ),
        (
            "layers.7.ffn.experts.23.w3.scale",
            "model.block.7.mlp.experts.23.up.weight_scale",
            TensorComponent.MODEL,
        ),
        (
            "mtp.2.main_proj.weight",
            "predictor.stage.2.main_projection.weight",
            TensorComponent.PREDICTOR,
        ),
        (
            "mtp.2.confidence_head.proj.weight",
            "predictor.stage.2.confidence.projection.weight",
            TensorComponent.PREDICTOR,
        ),
        (
            "vision.blocks.3.attn.wqkv.bias",
            "vision.block.3.attention.qkv.bias",
            TensorComponent.VISION,
        ),
        (
            "aligner.w2.weight",
            "vision.aligner.output.weight",
            TensorComponent.VISION,
        ),
    ],
)
def test_deepseek_v4_source_names_are_import_only(
    source: str,
    canonical: str,
    component: TensorComponent,
) -> None:
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "vision_n_layers": 32,
        "num_nextn_predict_layers": 1,
        "compress_ratios": [0] * 43 + [1, 1, 1],
        "dspark_target_layer_ids": [7, 20, 34],
    }
    mapped = map_source_tensor_name(source, config, require_registered=True)
    assert mapped is not None
    assert (mapped.canonical_name, mapped.component) == (canonical, component)


def test_deepseek_v4_topology_uses_structural_dspark_stage_count() -> None:
    topology = topology_from_config(
        {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 43,
            # The 0731 config reports one here, while its target/compressor
            # schedule and real inventory contain three DSpark stages.
            "num_nextn_predict_layers": 1,
            "compress_ratios": [0] * 46,
            "dspark_target_layer_ids": [8, 20, 32],
        }
    )
    assert topology.text_layers == 43
    assert topology.predictor_layers == 3


def test_deepseek_v4_graph_composes_vision_and_dspark() -> None:
    graph = graph_spec_for_plan(
        {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 43,
            "vision_n_layers": 32,
            "num_nextn_predict_layers": 3,
        },
        [
            "model.token_embedding.weight",
            "vision.patch_embedding.weight",
            "predictor.stage.0.main_projection.weight",
        ],
    )
    assert graph is not None
    payload = graph.as_dict()
    components = {component["kind"]: component for component in payload["components"]}
    assert components["text"]["implementation"] == "deepseek_v4"
    assert components["vision"]["implementation"] == "deepseek_v4_vision"
    assert components["predictor"]["implementation"] == "dspark"
    assert payload["topology"] == {
        "text_layers": 43,
        "vision_layers": 32,
        "predictor_layers": 3,
    }


@pytest.mark.parametrize(
    "source,canonical,component",
    [
        (
            "layers.8.attn.indexer.wk.weight",
            "model.block.8.attention.indexer.key.weight",
            TensorComponent.MODEL,
        ),
        (
            "layers.14.engram.embed.scale",
            "model.block.14.engram.embedding.weight_scale",
            TensorComponent.MODEL,
        ),
        (
            "layers.14.engram.q_weight",
            "model.block.14.engram.query.weight",
            TensorComponent.MODEL,
        ),
        (
            "mtp.2.markov_head.embed.weight",
            "predictor.stage.2.markov.embedding.weight",
            TensorComponent.PREDICTOR,
        ),
    ],
)
def test_deepseek_v41_source_names_and_graph(
    source: str,
    canonical: str,
    component: TensorComponent,
) -> None:
    config = {
        "model_type": "deepseek_v41",
        "text_config": {
            "model_type": "deepseek_v41_text",
            "num_hidden_layers": 40,
            "num_nextn_predict_layers": 3,
        },
        "vision_config": {"num_hidden_layers": 32},
    }
    mapped = map_source_tensor_name(source, config, require_registered=True)
    assert mapped is not None
    assert (mapped.canonical_name, mapped.component) == (canonical, component)
    graph = graph_spec_for_plan(
        config,
        [
            "model.token_embedding.weight",
            "vision.patch_embedding.weight",
            "predictor.stage.2.markov.output.weight",
        ],
    )
    assert graph is not None
    payload = graph.as_dict()
    assert payload["architecture"] == "deepseek_v41"
    assert payload["graph"]["backbone"] == "deepseek_v4"
    components = {item["kind"]: item for item in payload["components"]}
    assert components["vision"]["implementation"] == "deepseek_v41_vision"
    assert payload["topology"] == {
        "text_layers": 40,
        "vision_layers": 32,
        "predictor_layers": 3,
    }


@pytest.mark.parametrize(
    "source,canonical,component",
    [
        (
            "llm.model.layers.4.self_attn.q_proj.weight",
            "model.block.4.attention.query.weight",
            TensorComponent.MODEL,
        ),
        (
            "vpm.encoder.layers.6.mlp.fc2.bias",
            "vision.block.6.mlp.down.bias",
            TensorComponent.VISION,
        ),
        (
            "resampler.attn.in_proj_weight",
            "vision.resampler.attention.qkv.weight",
            TensorComponent.VISION,
        ),
        (
            "apm.layers.9.self_attn.out_proj.weight",
            "audio.block.9.attention.output.weight",
            TensorComponent.AUDIO,
        ),
        (
            "audio_projection_layer.linear2.bias",
            "audio.projector.output.bias",
            TensorComponent.AUDIO,
        ),
        (
            "tts.model.layers.5.mlp.gate_proj.weight",
            "tts.block.5.mlp.gate.weight",
            TensorComponent.TTS,
        ),
        (
            "tts.head_code.0.parametrizations.weight.original1",
            "tts.code_output.0.weight_norm.direction",
            TensorComponent.TTS,
        ),
    ],
)
def test_minicpmo45_composite_graph_uses_semantic_roots(
    source: str,
    canonical: str,
    component: TensorComponent,
) -> None:
    mapped = map_source_tensor_name(
        source,
        {"model_type": "minicpmo", "version": "4.5", "num_hidden_layers": 36},
        require_registered=True,
    )
    assert mapped is not None
    assert (mapped.canonical_name, mapped.component) == (canonical, component)


def test_minicpmo45_graph_declares_virtual_duplex_component() -> None:
    graph = graph_spec_for_plan(
        {
            "model_type": "minicpmo",
            "version": "4.5",
            "num_hidden_layers": 36,
            "vision_config": {"num_hidden_layers": 27},
        },
        [
            "model.token_embedding.weight",
            "vision.patch_embedding.weight",
            "audio.patch_embedding.conv1.weight",
            "tts.text_embedding.weight",
        ],
    )
    assert graph is not None
    payload = graph.as_dict()
    by_kind = {component["kind"]: component for component in payload["components"]}
    assert by_kind["text"]["implementation"] == "minicpmo45"
    assert by_kind["vision"]["implementation"] == "minicpmo45_vision"
    assert by_kind["audio_input"]["tensor_root"] == "audio"
    assert by_kind["audio_output"]["tensor_root"] == "tts"
    assert by_kind["duplex"] == {
        "kind": "duplex",
        "tensor_root": "runtime",
        "implementation": "minicpmo45_duplex",
        "policy": "optional",
    }
    assert payload["capabilities"] == [
        "text",
        "vision",
        "audio_input",
        "audio_output",
        "realtime_duplex",
    ]
    assert payload["canonical_naming"]["component_roots"] == [
        "model",
        "vision",
        "audio",
        "tts",
        "runtime",
    ]
    encoded = json.dumps(payload, sort_keys=True)
    for source_root in ("llm.", "vpm.", "apm.", "resampler."):
        assert source_root not in encoded


def test_minicpmo45_duplex_requires_both_audio_directions() -> None:
    graph = graph_spec_for_plan(
        {"model_type": "minicpmo", "version": "4.5", "num_hidden_layers": 36},
        ["model.token_embedding.weight", "audio.patch_embedding.conv1.weight"],
    )
    assert graph is not None
    kinds = {component["kind"] for component in graph.as_dict()["components"]}
    assert kinds == {"text", "audio_input"}


@pytest.mark.parametrize(
    "config,source,canonical,recipe",
    [
        (
            {"model_type": "glm_moe_dsa", "num_hidden_layers": 2},
            "model.layers.1.self_attn.embed_q",
            "model.block.1.attention.latent.query_embedding.weight",
            None,
        ),
        (
            {"model_type": "glm_moe_dsa", "num_hidden_layers": 2},
            "model.layers.1.mlp.experts.7.down_proj.weight",
            "model.block.1.mlp.experts.7.down.weight",
            None,
        ),
        (
            {
                "model_type": "gemma4",
                "text_config": {"model_type": "gemma4_text", "num_hidden_layers": 2},
            },
            "model.language_model.layers.1.input_layernorm.weight",
            "model.block.1.attention.norm.weight",
            "blk.1.attn_norm.weight",
        ),
        (
            {
                "model_type": "gemma4",
                "text_config": {"model_type": "gemma4_text", "num_hidden_layers": 2},
            },
            "model.language_model.layers.1.experts.gate_up_proj",
            "model.block.1.mlp.experts.gate_up.weight",
            "blk.1.ffn_gate_up_exps.weight",
        ),
    ],
)
def test_glm_dsa_and_gemma4_share_canonical_operations(
    config: dict[str, object],
    source: str,
    canonical: str,
    recipe: str | None,
) -> None:
    mapped = map_source_tensor_name(source, config, require_registered=True)
    assert mapped is not None
    assert mapped.canonical_name == canonical
    assert mapped.component is TensorComponent.MODEL
    assert mapped.recipe_name == recipe
