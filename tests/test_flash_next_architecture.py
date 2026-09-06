from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mfq.architectures.flash_next import (
    Glm5NextConfig,
    Qwen4ExpConfig,
    parse_flash_next_config,
)


def _vision(out_hidden_size: int, *, glm: bool = False) -> dict[str, object]:
    value: dict[str, object] = {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "depth": 4,
        "num_heads": 16,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": out_hidden_size,
        "hidden_act": "silu" if glm else "gelu_pytorch_tanh",
        "attention_bias": glm,
    }
    if glm:
        value.update(image_size=448, projection_intermediate_size=10240)
    else:
        value["num_position_embeddings"] = 2304
    return value


def _qwen_config() -> dict[str, object]:
    return {
        "model_type": "qwen4_exp",
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "vision_config": _vision(2560),
        "text_config": {
            "model_type": "qwen4_exp_text",
            "vocab_size": 248320,
            "hidden_size": 2560,
            "num_hidden_layers": 4,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "rope_theta": 10_000_000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "full_attention_interval": 4,
            "hc_count": 4,
            "hc_lowrank": 320,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "ple_layer_ids": [2],
            "ple_embed_dim": 2560,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ngram_vocab_size_base": 20_000_000,
            "split_ngram_parts": 128,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "mtp": {"num_hidden_layers": 1},
            "tie_word_embeddings": False,
            "eos_token_id": 248044,
        },
    }


def _glm_config() -> dict[str, object]:
    return {
        "model_type": "glm5_next",
        "image_token_id": 154854,
        "video_token_id": 154855,
        "image_start_token_id": 154830,
        "image_end_token_id": 154831,
        "video_start_token_id": 154832,
        "video_end_token_id": 154833,
        "vision_config": _vision(4096, glm=True),
        "text_config": {
            "model_type": "glm5_next_text",
            "vocab_size": 154880,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 4,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-5,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "deepseek_sparse_attention",
            ],
            "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
            "mhc": True,
            "mla_use_nope": True,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 20,
            "linear_attn_config": {
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "kda_layers": [0, 1, 2],
                "full_attn_layers": [3],
            },
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "num_attention_heads": 64,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 2048,
            "index_kpool": 4,
            "index_kpool_always_select_tail": True,
            "indexer_types": ["full", "full", "full", "full"],
            "n_routed_experts": 288,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "n_shared_experts": 1,
            "routed_scaling_factor": 2.5,
            "swiglu_limit": 10.0,
            "scoring_func": "sigmoid",
            "norm_topk_prob": True,
            "num_nextn_predict_layers": 1,
            "tie_word_embeddings": False,
            "eos_token_id": [154820, 154827, 154829],
        },
    }


def test_qwen38_flash_next_uses_qwen4_exp_contract() -> None:
    config = parse_flash_next_config(_qwen_config())
    assert isinstance(config, Qwen4ExpConfig)
    assert config.family == "qwen4_exp"
    assert config.rotary_dim == 64
    assert config.indexer_budget // config.indexer_compress_ratio == 512
    assert config.mtp_num_hidden_layers == 1
    assert config.vision is not None and config.vision.out_hidden_size == 2560


def test_glm53_flash_uses_glm5_next_contract() -> None:
    config = parse_flash_next_config(_glm_config())
    assert isinstance(config, Glm5NextConfig)
    assert config.family == "glm5_next"
    assert config.kda_num_heads == 64
    assert config.index_topk // config.index_kpool == 512
    assert config.num_nextn_predict_layers == 1
    assert config.eos_token_ids == (154820, 154827, 154829)
    assert config.video_start_token_id == 154832
    assert config.video_end_token_id == 154833


def test_qwen4_exp_schedule_mismatch_is_rejected() -> None:
    raw = _qwen_config()
    raw["text_config"]["layer_types"][0] = "full_attention"
    with pytest.raises(ValueError, match="schedule differs"):
        parse_flash_next_config(raw)


@pytest.mark.parametrize(
    "environment,expected",
    [
        ("MFQ_TEST_QWEN4_EXP_CONFIG", Qwen4ExpConfig),
        ("MFQ_TEST_GLM5_NEXT_CONFIG", Glm5NextConfig),
    ],
)
def test_optional_real_flash_next_config(environment, expected) -> None:
    raw_path = os.environ.get(environment)
    if not raw_path:
        pytest.skip(f"set {environment} for local checkpoint validation")
    config = parse_flash_next_config(
        json.loads(Path(raw_path).read_text(encoding="utf-8"))
    )
    assert isinstance(config, expected)
