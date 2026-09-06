from __future__ import annotations

import json

import pytest

from mfq.architectures.tensor_schema import (
    CANONICAL_TENSOR_NAMESPACE,
    TensorComponent,
    graph_spec_for_plan,
    map_source_tensor_name,
)
from mfq.compat.legacy_tensor_names import (
    LEGACY_HF_TENSOR_NAMES_REMOVE_AFTER,
    legacy_tensor_name_map,
    qwen35_legacy_name_map,
)

QWEN35_CONFIG = {
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 64,
        "mtp_num_hidden_layers": 1,
    },
    "vision_config": {"depth": 27},
}


@pytest.mark.parametrize(
    "source,canonical,component,recipe",
    [
        (
            "model.language_model.embed_tokens.weight",
            "model.token_embedding.weight",
            TensorComponent.MODEL,
            "token_embd.weight",
        ),
        (
            "model.language_model.layers.7.self_attn.q_proj.weight",
            "model.block.7.attention.query.weight",
            TensorComponent.MODEL,
            "blk.7.attn_q.weight",
        ),
        (
            "model.language_model.layers.7.mlp.gate_proj.weight",
            "model.block.7.mlp.gate.weight",
            TensorComponent.MODEL,
            "blk.7.ffn_gate.weight",
        ),
        (
            "model.language_model.layers.8.linear_attn.in_proj_z.weight",
            "model.block.8.linear_attention.gate.weight",
            TensorComponent.MODEL,
            "blk.8.attn_gate.weight",
        ),
        (
            "model.visual.blocks.3.attn.proj.weight",
            "vision.block.3.attention.output.weight",
            TensorComponent.VISION,
            None,
        ),
        (
            "model.visual.merger.linear_fc2.weight",
            "vision.merger.mlp.down.weight",
            TensorComponent.VISION,
            None,
        ),
        (
            "mtp.pre_fc_norm_hidden.weight",
            "predictor.hidden_norm.weight",
            TensorComponent.PREDICTOR,
            "blk.64.nextn.hnorm.weight",
        ),
        (
            "mtp.layers.0.mlp.gate_proj.weight",
            "predictor.block.0.mlp.gate.weight",
            TensorComponent.PREDICTOR,
            "blk.64.ffn_gate.weight",
        ),
    ],
)
def test_qwen35_import_mapping_separates_canonical_and_recipe_names(
    source: str,
    canonical: str,
    component: TensorComponent,
    recipe: str | None,
) -> None:
    mapped = map_source_tensor_name(source, QWEN35_CONFIG, require_registered=True)
    assert mapped is not None
    assert mapped.canonical_name == canonical
    assert mapped.component is component
    assert mapped.recipe_name == recipe


def test_qwen35_graph_spec_is_composable_and_source_name_free() -> None:
    graph = graph_spec_for_plan(
        QWEN35_CONFIG,
        [
            "model.token_embedding.weight",
            "vision.patch_embedding.weight",
            "predictor.fusion.weight",
        ],
    )
    assert graph is not None
    payload = graph.as_dict()

    assert payload["schema_version"] == 1
    assert payload["graph"] == {"kind": "causal_lm", "backbone": "qwen3_5"}
    assert payload["canonical_naming"]["namespace"] == CANONICAL_TENSOR_NAMESPACE
    components = {item["kind"]: item for item in payload["components"]}
    assert components["text"]["implementation"] == "qwen3_5"
    assert components["vision"]["implementation"] == "grid_vit"
    assert components["vision"]["input_contract"] == "grid_vision.v1"
    assert components["vision"]["position_policy"] == "grid_mrope"
    assert components["predictor"]["implementation"] == "next_token_prediction"
    encoded = json.dumps(payload, sort_keys=True)
    assert "model.language_model" not in encoded
    assert "model.visual" not in encoded
    assert "mtp." not in encoded


def test_legacy_hf_mfq_aliases_are_explicitly_removable_and_boundary_only() -> None:
    assert LEGACY_HF_TENSOR_NAMES_REMOVE_AFTER == "canonical-schema-v1-migration"
    assert qwen35_legacy_name_map(
        [
            "model.language_model.layers.2.mlp.down_proj.weight",
            "model.block.2.mlp.up.weight",
            "mtp.fc.weight",
        ],
        QWEN35_CONFIG,
    ) == {
        "model.language_model.layers.2.mlp.down_proj.weight": ("model.block.2.mlp.down.weight"),
        "mtp.fc.weight": "predictor.fusion.weight",
    }


def test_legacy_gguf_alias_cannot_overwrite_an_existing_canonical_tensor() -> None:
    with pytest.raises(ValueError, match="collide after canonicalization"):
        legacy_tensor_name_map(
            [
                "blk.2.attn_q.weight",
                "model.block.2.attention.query.weight",
            ],
            QWEN35_CONFIG,
        )


@pytest.mark.parametrize(
    "config,source,canonical,component",
    [
        (
            {
                "model_type": "qwen4_exp",
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "num_hidden_layers": 48,
                    "mtp_num_hidden_layers": 1,
                },
            },
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "predictor.block.0.mlp.experts.gate_up.weight",
            TensorComponent.PREDICTOR,
        ),
        (
            {
                "model_type": "qwen4_exp",
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "num_hidden_layers": 48,
                    "mtp_num_hidden_layers": 1,
                },
            },
            "model.language_model.layers.7.attn_hyper_connection.hc_norm.weight",
            "model.block.7.attention.mhc.pre.norm.weight",
            TensorComponent.MODEL,
        ),
        (
            {
                "model_type": "glm5_next",
                "text_config": {
                    "model_type": "glm5_next_text",
                    "num_hidden_layers": 45,
                    "num_nextn_predict_layers": 1,
                    "layer_types": ["linear_attention"] * 45,
                },
            },
            "model.language_model.layers.45.eh_proj.weight",
            "predictor.fusion.weight",
            TensorComponent.PREDICTOR,
        ),
        (
            {
                "model_type": "glm5_next",
                "text_config": {
                    "model_type": "glm5_next_text",
                    "num_hidden_layers": 45,
                    "num_nextn_predict_layers": 1,
                    "layer_types": ["linear_attention"] * 45,
                },
            },
            "model.language_model.layers.3.hc_attn_fn",
            "model.block.3.attention.mhc.pre.function",
            TensorComponent.MODEL,
        ),
    ],
)
def test_flash_next_imports_share_the_canonical_component_vocabulary(
    config: dict[str, object],
    source: str,
    canonical: str,
    component: TensorComponent,
) -> None:
    mapped = map_source_tensor_name(source, config, require_registered=True)
    assert mapped is not None
    assert mapped.canonical_name == canonical
    assert mapped.component is component


@pytest.mark.parametrize(
    "model_type,text_type,backbone,predictor_key",
    [
        ("qwen4_exp", "qwen4_exp_text", "qwen4_exp", "mtp_num_hidden_layers"),
        ("glm5_next", "glm5_next_text", "glm5_next", "num_nextn_predict_layers"),
    ],
)
def test_flash_next_graphs_use_the_same_optional_component_contracts(
    model_type: str,
    text_type: str,
    backbone: str,
    predictor_key: str,
) -> None:
    graph = graph_spec_for_plan(
        {
            "model_type": model_type,
            "text_config": {
                "model_type": text_type,
                "num_hidden_layers": 8,
                predictor_key: 1,
            },
            "vision_config": {"depth": 2},
        },
        [
            "model.token_embedding.weight",
            "vision.patch_embedding.weight",
            "predictor.fusion.weight",
        ],
    )
    assert graph is not None
    payload = graph.as_dict()
    assert payload["graph"]["backbone"] == backbone
    assert payload["canonical_naming"]["component_roots"] == [
        "model",
        "vision",
        "predictor",
    ]
    assert payload["topology"] == {
        "text_layers": 8,
        "vision_layers": 2,
        "predictor_layers": 1,
    }
