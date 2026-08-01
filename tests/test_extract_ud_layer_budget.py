from __future__ import annotations

import pytest

from mfq.tools.extract_ud_layer_budget import (
    build_ud_layer_budget_document,
    logical_tensor_elements,
)


def test_logical_tensor_elements_uses_shape_not_compressed_payload_length():
    assert logical_tensor_elements((128, 2048, 5376)) == 128 * 2048 * 5376


def test_build_ud_layer_budget_document_separates_recipe_and_exact_bits():
    document = build_ud_layer_budget_document(
        [
            {
                "name": "blk.0.ffn_gate_up_exps.weight",
                "tensor_type": "Q2_K",
                "payload_bytes": 100,
                "elements": 2000,
            },
            {
                "name": "blk.0.ffn_down_exps.weight",
                "tensor_type": "Q3_K",
                "payload_bytes": 60,
                "elements": 1000,
            },
            {
                "name": "blk.0.attn_q.weight",
                "tensor_type": "Q4_K",
                "payload_bytes": 40,
                "elements": 80,
            },
        ],
        expected_layers=1,
        expected_experts=128,
        expected_top_k=4,
        source={"path": "model.gguf", "sha256": "abc"},
    )

    layer = document["layers"]["0"]
    assert layer["target_storage_bits"] == 1280
    assert layer["routed_elements"] == 3000
    assert layer["effective_bpw"] == pytest.approx(1280 / 3000)
    assert layer["source_types"] == {
        "down": "Q3_K",
        "gate_up": "Q2_K",
    }
    assert document["non_routed_recipe"] == {
        "blk.0.attn_q.weight": "Q4_K"
    }


def test_build_ud_layer_budget_document_rejects_incomplete_routed_layers():
    with pytest.raises(ValueError, match="both routed tensors"):
        build_ud_layer_budget_document(
            [
                {
                    "name": "blk.0.ffn_gate_up_exps.weight",
                    "tensor_type": "Q2_K",
                    "payload_bytes": 100,
                    "elements": 2000,
                }
            ],
            expected_layers=1,
            expected_experts=128,
            expected_top_k=4,
            source={},
        )
