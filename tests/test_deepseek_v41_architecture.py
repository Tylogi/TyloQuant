from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mfq.architectures.deepseek_v41 import parse_deepseek_v41_config
from mfq.architectures.tensor_schema import (
    TensorComponent,
    graph_spec_for_plan,
    map_source_tensor_name,
)
from mfq.tools.quantize_hf_to_mfq import SourceTensorMetadata, _plan


def _config() -> dict[str, object]:
    return {
        "model_type": "deepseek_v41",
        "eos_token_id": 1,
        "image_token_id": 129264,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": 129280,
            "hidden_size": 5120,
            "moe_intermediate_size": 2304,
            "num_hidden_layers": 4,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1280,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "hidden_act": "silu",
            "attention_bias": False,
            "rms_norm_eps": 1e-20,
            "max_position_embeddings": 1048576,
            "rope_theta": 10000,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
            },
            "n_routed_experts": 384,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0,
            "sliding_window": 128,
            "compress_ratios": [0, 2, 1, 1, 0, 0, 0],
            "compress_rope_theta": 160000,
            "kv_source_layer_ids": [1],
            "index_source_layer_ids": [1, 2],
            "index_n_heads": 32,
            "index_head_dim": 128,
            "index_topk": 512,
            "candidate_source_layer_id": 2,
            "candidate_topk_blocks": 2048,
            "candidate_block_size": 8,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "engram_layer_ids": [1],
            "engram_num_embeddings": [384006168],
            "engram_max_ngram_size": 4,
            "engram_vocab_size": 16000000,
            "engram_n_heads": 8,
            "engram_head_dim": 256,
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": 99092,
            "num_nextn_predict_layers": 3,
            "dspark_block_size": 5,
            "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [1, 2, 3],
            "dspark_markov_rank": 256,
            "dspark_n_routed_experts": 128,
            "dspark_num_experts_per_tok": 3,
        },
        "vision_config": {
            "model_type": "deepseek_v41_vision",
            "num_hidden_layers": 32,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "intermediate_size": 2816,
            "patch_size": 14,
            "rope_theta": 10000,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
            "min_pixels": 295936,
            "max_wh_ratio": None,
        },
    }


def test_deepseek_v41_is_an_independent_strict_contract() -> None:
    config = parse_deepseek_v41_config(_config())
    assert config.family == "deepseek_v41"
    assert config.causal_encoder_layers == 2
    assert config.compress_ratios == (0, 2, 1, 1, 0, 0, 0)
    assert config.num_nextn_predict_layers == 3
    assert config.dense_weight_block_size == (32, 32)
    assert config.vision.max_image_tokens == 1024

    wrong = _config()
    wrong["model_type"] = "deepseek_v4"
    with pytest.raises(ValueError, match="expected deepseek_v41"):
        parse_deepseek_v41_config(wrong)


@pytest.mark.parametrize(
    "source,canonical,component",
    [
        ("embed.weight", "model.token_embedding.weight", TensorComponent.MODEL),
        (
            "layers.2.attn.indexer.wk.weight",
            "model.block.2.attention.indexer.key.weight",
            TensorComponent.MODEL,
        ),
        (
            "layers.1.engram.embed.scale",
            "model.block.1.associative_memory.embedding.weight_scale",
            TensorComponent.MODEL,
        ),
        (
            "layers.3.ffn.experts.383.w3.weight",
            "model.block.3.mlp.experts.383.up.weight",
            TensorComponent.MODEL,
        ),
        (
            "mtp.2.markov_head.embed.weight",
            "predictor.stage.2.markov.embedding.weight",
            TensorComponent.PREDICTOR,
        ),
        (
            "mtp.2.confidence_head.proj.weight",
            "predictor.stage.2.confidence.projection.weight",
            TensorComponent.PREDICTOR,
        ),
        (
            "vision.blocks.31.attn.wqkv.weight",
            "vision.block.31.attention.qkv.weight",
            TensorComponent.VISION,
        ),
        ("image_newline", "vision.special_token.newline", TensorComponent.VISION),
    ],
)
def test_deepseek_v41_checkpoint_names_map_to_canonical_graph(
    source: str,
    canonical: str,
    component: TensorComponent,
) -> None:
    mapped = map_source_tensor_name(source, _config(), require_registered=True)
    assert mapped is not None
    assert (mapped.canonical_name, mapped.component) == (canonical, component)


def test_deepseek_v41_graph_keeps_vision_and_dspark_architecture_specific() -> None:
    graph = graph_spec_for_plan(
        _config(),
        [
            "model.token_embedding.weight",
            "vision.patch_embedding.weight",
            "predictor.stage.0.main_projection.weight",
        ],
    )
    assert graph is not None
    payload = graph.as_dict()
    components = {component["kind"]: component for component in payload["components"]}
    assert payload["architecture"] == "deepseek_v41"
    assert components["text"]["implementation"] == "deepseek_v41"
    assert components["vision"]["implementation"] == "deepseek_v41_vision"
    assert components["predictor"]["implementation"] == "deepseek_v41_dspark"
    assert payload["topology"] == {
        "text_layers": 4,
        "vision_layers": 32,
        "predictor_layers": 3,
    }


def test_deepseek_v41_engram_table_plan_preserves_native_row_mxfp8() -> None:
    weight = "layers.1.engram.embed.weight"
    scale = "layers.1.engram.embed.scale"
    inventory = {
        weight: SourceTensorMetadata(
            weight,
            "model.safetensors",
            (384006168, 256),
            "F8_E4M3",
        ),
        scale: SourceTensorMetadata(
            scale,
            "model.safetensors",
            (384006168, 8),
            "F8_E8M0",
        ),
    }

    plan = _plan(
        Path("."),
        False,
        None,
        "F16",
        source_inventory=inventory,
        source_config=_config(),
        default_nint_dtype="NINT4",
    )

    assert len(plan) == 1
    assert plan[0].name == "model.block.1.associative_memory.embedding.weight"
    assert plan[0].target_dtype == "MXFP8"
    assert plan[0].source_quantization == "mxfp8_block1x32"
    assert plan[0].source_scale_name == scale


def test_optional_official_deepseek_v41_inventory_maps_without_gaps() -> None:
    root_raw = os.environ.get("MFQ_TEST_DEEPSEEK_V41_ROOT")
    if not root_raw:
        pytest.skip("set MFQ_TEST_DEEPSEEK_V41_ROOT to the official metadata checkout")
    root = Path(root_raw)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    parse_deepseek_v41_config(config)
    weight_map = json.loads(
        (root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    canonical = []
    for source in weight_map:
        mapped = map_source_tensor_name(source, config, require_registered=True)
        assert mapped is not None
        canonical.append(mapped.canonical_name)
    assert len(canonical) == 96085
    assert len(set(canonical)) == len(canonical)
