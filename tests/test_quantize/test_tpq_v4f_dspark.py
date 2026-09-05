from __future__ import annotations

import json

import pytest

from mfq.tools.quantize_tpq_v4f_to_mfq import (
    _dspark_expert_plan,
    _tpq_config,
)


def _config() -> dict[str, object]:
    return {
        "num_hidden_layers": 43,
        "hidden_size": 4096,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
        "num_attention_heads": 64,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "kv_dim": 512,
        "qk_rope_head_dim": 64,
        "num_key_value_heads": 1,
        "vocab_size": 129280,
        "num_hash_layers": 3,
        "compress_ratios": [0] * 43 + [0, 0, 0],
        # The original text checkpoint reports one here even though its
        # structural fields and tensors contain three DSpark stages.
        "num_nextn_predict_layers": 1,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_markov_rank": 256,
        "vision_n_layers": 32,
        "vision_dim": 1024,
        "vision_n_heads": 16,
        "vision_inter_dim": 2816,
        "vision_patch_size": 14,
        "vision_downsample_ratio": 3,
    }


def test_tpq_config_preserves_vision_and_structural_dspark_count(
    tmp_path,
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_config()), encoding="utf-8")

    converted = _tpq_config(tmp_path)

    assert converted["n_mtp_layers"] == 3
    assert converted["dspark_block_size"] == 5
    assert converted["dspark_target_layer_ids"] == [40, 41, 42]
    assert converted["vision_n_layers"] == 32
    assert converted["vision_downsample_ratio"] == 3
    assert len(converted["compress_ratios"]) == 46


def test_tpq_config_rejects_conflicting_dspark_structure(tmp_path) -> None:
    config = _config()
    config["dspark_target_layer_ids"] = [42]
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent stage counts"):
        _tpq_config(tmp_path)


def test_dspark_experts_reuse_corresponding_backbone_tiers() -> None:
    plan = _dspark_expert_plan(
        {
            "n_mtp_layers": 3,
            "dspark_target_layer_ids": [40, 41, 42],
        }
    )
    assert len(plan) == 6
    assert plan[0] == (
        0,
        "gate_up",
        "mtp.0.ffn.experts.gate_up.weight",
        "blk.40.ffn_gate_up_exps.weight",
    )
    assert plan[-1] == (
        2,
        "down",
        "mtp.2.ffn.experts.down.weight",
        "blk.42.ffn_down_exps.weight",
    )
