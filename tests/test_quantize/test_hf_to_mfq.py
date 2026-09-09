from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import mfq.tools.quantize_hf_to_mfq as hf_to_mfq
from mfq.calibration.artifact import ExpertPrecision
from mfq.formats.assets import (
    HF_CHAT_TEMPLATE_ASSET,
    HF_GENERATION_CONFIG_ASSET,
    HF_TOKENIZER_CONFIG_ASSET,
    HF_TOKENIZER_JSON_ASSET,
    MODEL_CONFIG_ASSET,
    MODEL_GRAPH_ASSET,
    is_asset_record,
)
from mfq.formats.header import FileHeader
from mfq.formats.io import is_bfloat16_array, load_mmap, open_mmap, save, unpack_dense
from mfq.formats.moe import NintMoePool, NintMoeTensor
from mfq.formats.nint import NintSpec
from mfq.formats.shards import format_shard_path
from mfq.quantize.imatrix import ImportanceEntry, ImportanceMatrix
from mfq.quantize.nint_quant import quantize as quantize_nint
from mfq.tools.quantize_hf_to_mfq import (
    TensorPlan,
    _bind_hf_imatrix,
    _dtype_for_recipe_type,
    _GlmExpertRowSource,
    _hf_to_gguf_name,
    _minicpmo45_quantizable_matrix,
    _normalize_hf_expert_storage,
    _RawSafeTensorSlice,
    _ScaledFp8TensorSlice,
    _source_quantization,
    _transform_glm_kv_b,
    _validate_runtime_fused_pairs,
    _write_dense_axis0_blob,
    convert,
)
from mfq.tools.quantize_hf_to_mfq import (
    _plan as build_hf_plan,
)


def _plan(name: str, spec: NintSpec) -> TensorPlan:
    return TensorPlan(
        name=name,
        shard="model.safetensors",
        shape=(16, 16),
        source_dtype="BF16",
        target_dtype=f"NINT{spec.bits}",
        target_spec=spec,
    )


def _standard_plan(name: str, shape: tuple[int, ...] = (16, 48)) -> TensorPlan:
    return TensorPlan(
        name=name,
        shard="model.safetensors",
        shape=shape,
        source_dtype="BF16",
        target_dtype="NINT4" if len(shape) == 2 else "F16",
    )


def test_streamed_dense_writer_preserves_three_dimensional_expert_geometry(tmp_path) -> None:
    values = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(6, 4)

    class Rows:
        def read_rows(self, start, end, *, device="cpu"):
            return values[start:end].to(device)

    blob = tmp_path / "dense-experts.blob"
    _write_dense_axis0_blob(Rows(), (2, 3, 4), blob, "F16", row_chunk=2)

    restored = unpack_dense("F16", blob.read_bytes())
    np.testing.assert_array_equal(restored, values.numpy().reshape(2, 3, 4))


def test_standard_preset_aliases_match_llamacpp() -> None:
    assert hf_to_mfq._normalize_standard_preset("q3-k") == "Q3_K_M"
    assert hf_to_mfq._normalize_standard_preset("Q4_K") == "Q4_K_M"
    assert hf_to_mfq._normalize_standard_preset("q5_k") == "Q5_K_M"
    with pytest.raises(ValueError, match="unsupported standard quantization preset"):
        hf_to_mfq._normalize_standard_preset("Q7_K")


def test_q4_k_m_standard_preset_raises_sensitive_text_matrices() -> None:
    plans = [
        _standard_plan("model.language_model.embed_tokens.weight"),
        _standard_plan("lm_head.weight"),
        _standard_plan("model.language_model.layers.0.mlp.down_proj.weight"),
        _standard_plan("model.language_model.layers.8.mlp.down_proj.weight"),
        _standard_plan("model.language_model.layers.10.mlp.down_proj.weight"),
        _standard_plan("model.language_model.layers.10.self_attn.o_proj.weight"),
    ]
    mapped = hf_to_mfq._apply_standard_preset(
        plans,
        "Q4_K_M",
        {
            "text_config": {
                "num_hidden_layers": 64,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
            }
        },
    )
    by_name = {item.name: item for item in mapped}

    assert by_name["model.language_model.embed_tokens.weight"].target_dtype == "NINT6"
    assert by_name["lm_head.weight"].target_dtype == "NINT6"
    assert by_name["model.language_model.layers.0.mlp.down_proj.weight"].target_dtype == "NINT6"
    assert by_name["model.language_model.layers.8.mlp.down_proj.weight"].target_dtype == "NINT4"
    assert by_name["model.language_model.layers.10.mlp.down_proj.weight"].target_dtype == "NINT6"
    assert by_name["model.language_model.layers.10.self_attn.o_proj.weight"].target_dtype == "NINT4"


def test_q3_k_m_standard_preset_matches_attention_v_mixture() -> None:
    plans = [
        _standard_plan(f"model.language_model.layers.{layer}.self_attn.v_proj.weight")
        for layer in range(3)
    ]
    mapped = hf_to_mfq._apply_standard_preset(
        plans,
        "Q3_K_M",
        {
            "num_hidden_layers": 3,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
        },
    )

    assert [item.target_dtype for item in mapped] == ["NINT5", "NINT5", "NINT4"]


def test_q2_k_standard_preset_uses_gqa_sensitive_types() -> None:
    plans = [
        _standard_plan("model.language_model.layers.7.self_attn.v_proj.weight"),
        _standard_plan("model.language_model.layers.7.self_attn.o_proj.weight"),
        _standard_plan("model.language_model.layers.7.mlp.down_proj.weight"),
    ]
    mapped = hf_to_mfq._apply_standard_preset(
        plans,
        "Q2_K",
        {
            "num_hidden_layers": 8,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
        },
    )

    assert [item.target_dtype for item in mapped] == ["NINT4", "NINT3", "NINT3"]


def test_standard_preset_keeps_all_vision_and_predictor_tensors_native_by_default() -> None:
    plans = [
        _standard_plan("model.visual.blocks.0.attn.qkv.weight"),
        _standard_plan("model.visual.pos_embed.weight"),
        _standard_plan("model.visual.merger.linear_fc2.weight"),
        _standard_plan("model.visual.patch_embed.proj.weight", (8, 3, 2, 2, 2)),
        _standard_plan("mtp.fc.weight"),
        _standard_plan("predictor.block.0.attention.query.weight"),
    ]
    mapped = hf_to_mfq._apply_standard_preset(
        plans,
        "Q4_K_M",
        {
            "text_config": {"num_hidden_layers": 64},
            "vision_config": {"depth": 27},
        },
    )
    by_name = {item.name: item for item in mapped}

    assert by_name["model.visual.blocks.0.attn.qkv.weight"].target_dtype == "BF16"
    assert by_name["model.visual.pos_embed.weight"].target_dtype == "BF16"
    assert by_name["model.visual.merger.linear_fc2.weight"].target_dtype == "BF16"
    assert by_name["model.visual.patch_embed.proj.weight"].target_dtype == "BF16"
    assert by_name["mtp.fc.weight"].target_dtype == "BF16"
    assert by_name["predictor.block.0.attention.query.weight"].target_dtype == "BF16"


def test_standard_preset_keeps_small_sensitive_control_paths_native() -> None:
    plans = [
        _standard_plan("model.block.0.attention.mhc.pre.down.weight", (320, 10240)),
        _standard_plan("model.block.0.linear_attention.conv.weight", (10240, 1, 4)),
        _standard_plan("model.block.0.attention.indexer.query_key.weight", (640, 2560)),
        _standard_plan("model.block.0.position_embedding.key.weight", (10240, 2560)),
        _standard_plan("model.block.0.position_embedding.value.weight", (2560, 2560)),
        _standard_plan("model.block.0.position_embedding.conv.weight", (10240, 1, 4)),
    ]
    mapped = hf_to_mfq._apply_standard_preset(
        plans,
        "Q4_K_M",
        {"num_hidden_layers": 1},
    )

    assert {item.target_dtype for item in mapped} == {"BF16"}


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("Q2_K_S", "NINT6"),
        ("Q4_K_M", "NINT6"),
        ("Q8_0", "NINT8"),
    ],
)
def test_standard_preset_matches_embedding_to_output_precision(
    preset: str,
    expected: str,
) -> None:
    mapped = hf_to_mfq._apply_standard_preset(
        [
            _standard_plan("model.token_embedding.weight"),
            _standard_plan("model.output.weight"),
        ],
        preset,
        {"num_hidden_layers": 1},
    )

    assert [item.target_dtype for item in mapped] == [expected, expected]


@pytest.mark.parametrize("preset", hf_to_mfq.STANDARD_PRESET_NAMES)
def test_standard_preset_always_uses_nint8_for_shared_expert_weights(
    preset: str,
) -> None:
    plans = [
        _standard_plan("model.block.0.mlp.shared_expert.gate.weight"),
        _standard_plan("model.block.0.mlp.shared_expert.up.weight"),
        _standard_plan("model.block.0.mlp.shared_expert.down.weight"),
        _standard_plan("model.block.0.mlp.shared_expert.router.weight"),
    ]
    mapped = hf_to_mfq._apply_standard_preset(
        plans,
        preset,
        {"num_hidden_layers": 1},
    )
    by_name = {item.name: item for item in mapped}

    for projection in ("gate", "up", "down"):
        assert (
            by_name[f"model.block.0.mlp.shared_expert.{projection}.weight"].target_dtype
            == "NINT8"
        )
    assert by_name["model.block.0.mlp.shared_expert.router.weight"].target_dtype == "BF16"


def test_standard_preset_only_treats_schema_expert_banks_as_nintm() -> None:
    plans = [
        _standard_plan("model.block.0.linear_attention.conv.weight", (16, 1, 4)),
        _standard_plan("model.block.0.mlp.experts.down.weight", (8, 16, 48)),
    ]
    mapped = hf_to_mfq._apply_standard_preset(
        plans,
        "Q4_K_M",
        {"num_hidden_layers": 1},
    )

    assert mapped[0].target_dtype == "BF16"
    assert mapped[1].target_dtype == "NINTM"


def test_standard_preset_quantizes_vision_and_predictor_only_with_opt_in() -> None:
    plans = [
        _standard_plan("model.visual.blocks.0.attn.qkv.weight"),
        _standard_plan("model.visual.merger.linear_fc2.weight"),
        _standard_plan("mtp.layers.0.self_attn.q_proj.weight"),
        _standard_plan("mtp.fc.weight"),
    ]
    mapped = hf_to_mfq._apply_standard_preset(
        plans,
        "Q4_K_M",
        {
            "text_config": {"num_hidden_layers": 64},
            "vision_config": {"depth": 27},
        },
        quantize_vision=True,
        quantize_mtp=True,
    )
    by_name = {item.name: item for item in mapped}

    assert by_name["model.visual.blocks.0.attn.qkv.weight"].target_dtype == "NINT6"
    assert by_name["mtp.layers.0.self_attn.q_proj.weight"].target_dtype == "NINT4"
    assert by_name["model.visual.merger.linear_fc2.weight"].target_dtype == "BF16"
    assert by_name["mtp.fc.weight"].target_dtype == "BF16"


def test_normalize_hf_expert_storage_preserves_mixed_nintm_plan() -> None:
    precisions = (
        ExpertPrecision("NINT2", nint_spec=NintSpec(2, 16, 5)),
        ExpertPrecision("NINT5", nint_spec=NintSpec(5, 24, 7)),
    )
    item = TensorPlan(
        name="model.language_model.layers.0.mlp.experts.down_proj.weight",
        shard="model.safetensors",
        shape=(2, 3, 48),
        source_dtype="BF16",
        target_dtype="NINTM",
        expert_shape=(2, 3, 48),
        expert_precisions=precisions,
    )

    normalized = _normalize_hf_expert_storage([item])

    assert normalized == [item]
    assert normalized[0].expert_precisions == precisions


def test_balanced_random_expert_mix_is_reproducible_and_projection_independent() -> None:
    plans = [
        TensorPlan(
            name=f"model.language_model.layers.0.mlp.experts.{suffix}.weight",
            shard="model.safetensors",
            shape=(8, rows, 48),
            source_dtype="F8_E4M3",
            target_dtype="NINTM",
            expert_shape=(8, rows, 48),
            expert_precisions=(ExpertPrecision("NINT4", NintSpec(4, 24, 6)),) * 8,
        )
        for suffix, rows in (("gate_proj", 32), ("up_proj", 32), ("down_proj", 16))
    ]
    profiles = hf_to_mfq._parse_expert_mix_profiles("NINT2,NINT4,NVQ2J")

    mixed = hf_to_mfq._apply_balanced_random_expert_mix(plans, profiles, 17)
    repeated = hf_to_mfq._apply_balanced_random_expert_mix(plans, profiles, 17)

    assert mixed == repeated
    assignments = [item.expert_precisions for item in mixed]
    assert len(set(assignments)) == len(assignments)
    for assignment in assignments:
        counts = {
            family: sum(value.family == family for value in assignment or ())
            for family in ("NINT2", "NINT4", "NVQ2J")
        }
        assert max(counts.values()) - min(counts.values()) <= 1
    _validate_runtime_fused_pairs(mixed)


def test_parse_expert_mix_profiles_uses_runtime_nint_specs() -> None:
    profiles = hf_to_mfq._parse_expert_mix_profiles(
        "NINT2,NINT3,NINT4,NINT5,NINT6,NINT8,NVQ2J,NVQ3J,MXFP4"
    )

    assert [value.family for value in profiles] == [
        "NINT2",
        "NINT3",
        "NINT4",
        "NINT5",
        "NINT6",
        "NINT8",
        "NVQ2J",
        "NVQ3J",
        "MXFP4",
    ]
    assert [value.nint_spec for value in profiles[:6]] == [
        NintSpec(2, 16, 5),
        NintSpec(3, 24, 5),
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 7),
        NintSpec(6, 24, 7),
        NintSpec(8, 48, 7),
    ]


def test_raw_safetensor_slice_streams_bfloat16_rows_and_expert_rows(tmp_path):
    path = tmp_path / "model.safetensors"
    expected = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    expected = (expected / 7).to(torch.bfloat16)
    save_file({"experts": expected}, path)

    source = _RawSafeTensorSlice(path, "experts")

    assert source.shape == (2, 3, 4)
    assert torch.equal(
        source.read_rows(1, 5, device="cpu"),
        expected.reshape(6, 4)[1:5],
    )
    assert torch.equal(
        source.read_expert_rows(1, 1, 3, device="cpu"),
        expected[1, 1:3],
    )
    assert torch.equal(source[2:4], expected.reshape(6, 4)[2:4])


def test_scaled_fp8_tensor_slice_applies_modelopt_block_multipliers(tmp_path):
    path = tmp_path / "model.safetensors"
    base = torch.linspace(-4, 4, 260 * 260, dtype=torch.float32).reshape(260, 260)
    weight = base.to(torch.float8_e4m3fn)
    scale = torch.tensor(
        [[0.25, 0.5, 0.75], [1.0, 1.25, 1.5], [1.75, 2.0, 2.25]],
        dtype=torch.float32,
    )
    save_file(
        {"proj.weight": weight, "proj.weight_scale_inv": scale},
        path,
    )
    source = _ScaledFp8TensorSlice(
        _RawSafeTensorSlice(path, "proj.weight"),
        _RawSafeTensorSlice(path, "proj.weight_scale_inv"),
        "fp8_block128_inv",
    )

    rows = np.asarray([0, 127, 128, 259])
    actual = source.read_rows(rows)
    expected_scale = scale[
        torch.as_tensor(rows // 128)[:, None],
        (torch.arange(260) // 128)[None, :],
    ]
    torch.testing.assert_close(actual, weight[rows].float() * expected_scale)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires the Apple MPS backend",
)
def test_scaled_fp8_tensor_slice_stages_float8_cast_on_cpu_for_mps(tmp_path):
    path = tmp_path / "model.safetensors"
    weight = torch.tensor(
        [[-4.0, -1.0, 2.0], [3.0, 0.5, -0.25]],
        dtype=torch.float8_e4m3fn,
    )
    save_file(
        {
            "proj.weight": weight,
            "proj.weight_scale": torch.tensor([0.125]),
        },
        path,
    )
    source = _ScaledFp8TensorSlice(
        _RawSafeTensorSlice(path, "proj.weight"),
        _RawSafeTensorSlice(path, "proj.weight_scale"),
        "fp8_tensor_scale",
    )

    actual = source.read_rows(0, 2, device="mps")

    assert actual.device.type == "mps"
    torch.testing.assert_close(actual.cpu(), weight.float() * 0.125)


def test_scaled_fp8_tensor_slice_applies_shared_ngram_scale(tmp_path):
    path = tmp_path / "model.safetensors"
    weight = torch.tensor(
        [[-4.0, -1.0, 2.0], [3.0, 0.5, -0.25]],
        dtype=torch.float8_e4m3fn,
    )
    save_file(
        {
            "ple.ngram_embedding.shard_0.weight": weight,
            "ple.ngram_embedding.weight_scale": torch.tensor([0.125]),
        },
        path,
    )
    inventory = hf_to_mfq._hf_source_inventory(tmp_path)
    quantization = _source_quantization("ple.ngram_embedding.shard_0.weight", inventory)
    assert quantization is not None
    assert quantization.scheme == "fp8_tensor_scale"
    source = _ScaledFp8TensorSlice(
        _RawSafeTensorSlice(path, "ple.ngram_embedding.shard_0.weight"),
        _RawSafeTensorSlice(path, quantization.scale_name),
        quantization.scheme,
    )
    torch.testing.assert_close(source.tensor(), weight.float() * 0.125)


def test_qwen4_ple_defaults_to_raw_fp8_and_requires_explicit_quantization(tmp_path):
    weight_name = (
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
        "shard_0.weight"
    )
    scale_name = (
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
        "weight_scale"
    )
    inventory = {
        weight_name: hf_to_mfq.SourceTensorMetadata(
            name=weight_name,
            shard="model.safetensors",
            shape=(32, 16),
            dtype="F8_E4M3",
        ),
        scale_name: hf_to_mfq.SourceTensorMetadata(
            name=scale_name,
            shard="model.safetensors",
            shape=(1,),
            dtype="BF16",
        ),
    }
    for projection, shape in (
        ("gate_proj", (16, 16)),
        ("up_proj", (16, 16)),
        ("down_proj", (16, 16)),
    ):
        name = f"model.language_model.layers.0.mlp.experts.0.{projection}.weight"
        inventory[name] = hf_to_mfq.SourceTensorMetadata(
            name=name,
            shard="model.safetensors",
            shape=shape,
            dtype="BF16",
        )
    config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": 2,
            "mtp_num_hidden_layers": 0,
            "num_experts": 1,
            "hidden_size": 16,
            "moe_intermediate_size": 16,
        },
    }

    preserved = build_hf_plan(
        tmp_path,
        text_only=True,
        recipe_types=None,
        dense_dtype="F16",
        source_inventory=inventory,
        source_config=config,
    )
    preserved_by_name = {item.name: item for item in preserved}
    canonical_weight = "model.block.1.position_embedding.ngram.shard.0.weight"
    canonical_scale = "model.block.1.position_embedding.ngram.weight_scale"
    assert preserved_by_name[canonical_weight].target_dtype == "F8_E4M3"
    assert preserved_by_name[canonical_scale].target_dtype == "BF16"

    quantized = build_hf_plan(
        tmp_path,
        text_only=True,
        recipe_types=None,
        dense_dtype="F16",
        quantize_ple=True,
        source_inventory=inventory,
        source_config=config,
    )
    quantized_by_name = {item.name: item for item in quantized}
    assert quantized_by_name[canonical_weight].target_dtype == "NINT4"
    assert canonical_scale not in quantized_by_name


@pytest.mark.parametrize(
    "left_suffix,right_suffix",
    [
        ("self_attn.q_proj.weight", "self_attn.k_proj.weight"),
        ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
        (
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
        ),
    ],
)
def test_runtime_fused_pairs_require_identical_precision_layout(left_suffix, right_suffix):
    prefix = "model.language_model.layers.3."
    with pytest.raises(ValueError, match="must share one precision layout"):
        _validate_runtime_fused_pairs(
            [
                _plan(prefix + left_suffix, NintSpec(4, 24, 6)),
                _plan(prefix + right_suffix, NintSpec(5, 28, 7)),
            ]
        )


def test_runtime_fused_pairs_accept_identical_precision_layout():
    prefix = "model.language_model.layers.3."
    spec = NintSpec(4, 24, 6)
    _validate_runtime_fused_pairs(
        [
            _plan(prefix + "self_attn.q_proj.weight", spec),
            _plan(prefix + "self_attn.k_proj.weight", spec),
            _plan(prefix + "mlp.gate_proj.weight", spec),
            _plan(prefix + "mlp.up_proj.weight", spec),
            _plan(prefix + "mlp.shared_expert.gate_proj.weight", spec),
            _plan(prefix + "mlp.shared_expert.up_proj.weight", spec),
        ]
    )


@pytest.mark.parametrize(
    "suffix,gguf_suffix",
    [
        ("mlp.experts.down_proj", "ffn_down_exps.weight"),
        ("mlp.experts.down_proj.weight", "ffn_down_exps.weight"),
        ("mlp.experts.gate_up_proj", "ffn_gate_up_exps.weight"),
        ("mlp.experts.gate_up_proj.weight", "ffn_gate_up_exps.weight"),
        ("mlp.gate.weight", "ffn_gate_inp.weight"),
        ("mlp.shared_expert.down_proj.weight", "ffn_down_shexp.weight"),
        ("mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"),
        ("mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight"),
        ("mlp.shared_expert_gate.weight", "ffn_gate_inp_shexp.weight"),
    ],
)
def test_qwen35_moe_hf_to_gguf_name_mapping(suffix, gguf_suffix):
    assert _hf_to_gguf_name(f"model.language_model.layers.17.{suffix}") == f"blk.17.{gguf_suffix}"


@pytest.mark.parametrize(
    "name,gguf_name",
    [
        ("mtp.fc.weight", "blk.40.nextn.eh_proj.weight"),
        ("mtp.pre_fc_norm_embedding.weight", "blk.40.nextn.enorm.weight"),
        ("mtp.pre_fc_norm_hidden.weight", "blk.40.nextn.hnorm.weight"),
        ("mtp.norm.weight", "blk.40.nextn.shared_head_norm.weight"),
        ("mtp.layers.0.mlp.experts.down_proj", "blk.40.ffn_down_exps.weight"),
        (
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "blk.40.ffn_gate_up_exps.weight",
        ),
        ("mtp.layers.0.self_attn.q_proj.weight", "blk.40.attn_q.weight"),
    ],
)
def test_qwen35_mtp_hf_to_gguf_name_mapping(name, gguf_name):
    assert _hf_to_gguf_name(name) == gguf_name


def test_qwen35_mtp_mapping_uses_dynamic_backbone_layer_count():
    assert _hf_to_gguf_name("mtp.fc.weight", mtp_layer_index=64) == "blk.64.nextn.eh_proj.weight"
    assert (
        _hf_to_gguf_name(
            "mtp.layers.0.self_attn.o_proj.weight",
            mtp_layer_index=64,
        )
        == "blk.64.attn_output.weight"
    )


def test_qwen35_mtp_inventory_is_all_or_nothing():
    names = set(hf_to_mfq._MTP_PROTECTED_TENSORS)
    prefix = "mtp.layers.0."
    names.update(
        {
            prefix + "input_layernorm.weight",
            prefix + "post_attention_layernorm.weight",
            prefix + "self_attn.q_proj.weight",
            prefix + "self_attn.k_proj.weight",
            prefix + "self_attn.v_proj.weight",
            prefix + "self_attn.o_proj.weight",
            prefix + "self_attn.q_norm.weight",
            prefix + "self_attn.k_norm.weight",
            prefix + "mlp.gate_proj.weight",
            prefix + "mlp.up_proj.weight",
            prefix + "mlp.down_proj.weight",
        }
    )
    inventory = {name: None for name in names}
    assert hf_to_mfq._mtp_inventory_status(inventory, {"mtp_num_hidden_layers": 1}) == (True, 1)

    inventory.pop("mtp.fc.weight")
    with pytest.raises(ValueError, match="incomplete Qwen MTP head"):
        hf_to_mfq._mtp_inventory_status(inventory, {"mtp_num_hidden_layers": 1})


def test_qwen35_mtp_plan_preserves_complete_head_and_protected_weights(tmp_path):
    names = set(hf_to_mfq._MTP_PROTECTED_TENSORS)
    prefix = "mtp.layers.0."
    names.update(
        {
            prefix + "input_layernorm.weight",
            prefix + "post_attention_layernorm.weight",
            prefix + "self_attn.q_proj.weight",
            prefix + "self_attn.k_proj.weight",
            prefix + "self_attn.v_proj.weight",
            prefix + "self_attn.o_proj.weight",
            prefix + "self_attn.q_norm.weight",
            prefix + "self_attn.k_norm.weight",
            prefix + "mlp.gate_proj.weight",
            prefix + "mlp.up_proj.weight",
            prefix + "mlp.down_proj.weight",
        }
    )
    inventory = {}
    for name in names:
        is_norm = "norm" in name
        shape = (4,) if is_norm else ((4, 8) if name == "mtp.fc.weight" else (4, 4))
        inventory[name] = hf_to_mfq.SourceTensorMetadata(
            name=name,
            shard="model.safetensors",
            shape=shape,
            dtype="F32" if is_norm else "BF16",
        )

    plan = build_hf_plan(
        tmp_path,
        text_only=True,
        recipe_types={},
        dense_dtype="F16",
        source_inventory=inventory,
        source_config={
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 64,
                "mtp_num_hidden_layers": 1,
            }
        },
    )

    by_name = {item.name: item for item in plan}
    by_source = {item.source_name: item for item in plan}
    assert set(by_source) == names
    assert all(name.startswith("predictor.") for name in by_name)
    assert by_source["mtp.fc.weight"].target_dtype == "BF16"
    assert by_source["mtp.pre_fc_norm_hidden.weight"].target_dtype == "F32"
    assert by_source[prefix + "self_attn.q_proj.weight"].target_dtype == "BF16"
    assert by_source[prefix + "self_attn.q_proj.weight"].gguf_name == "blk.64.attn_q.weight"


def test_qwen35_mtp_augmentation_copies_base_and_mirrors_backbone_policy(tmp_path):
    root = tmp_path / "hf"
    root.mkdir()
    config = {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 1,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    layer_suffixes = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    source_tensors = {
        "mtp.fc.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8).to(torch.bfloat16),
        "mtp.pre_fc_norm_embedding.weight": torch.arange(4, dtype=torch.float32).to(torch.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": torch.arange(4, dtype=torch.float32).to(torch.bfloat16),
        "mtp.norm.weight": torch.arange(4, dtype=torch.float32).to(torch.bfloat16),
    }
    base_tensors = {
        "sentinel.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        MODEL_CONFIG_ASSET: json.dumps(config).encode(),
        # An incomplete old head must be replaced, not retained.
        "mtp.norm.weight": np.zeros(4, dtype=np.float16),
    }
    for suffix in layer_suffixes:
        is_norm = "norm" in suffix
        shape = (4,) if is_norm else (4, 4)
        source_tensors[f"mtp.layers.0.{suffix}"] = (
            torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape).to(torch.bfloat16)
        )
        base_tensors[f"model.language_model.layers.0.{suffix}"] = (
            np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
            if is_norm
            else np.arange(int(np.prod(shape)), dtype=np.float16).reshape(shape)
        )
    save_file(source_tensors, root / "model.safetensors")

    base = tmp_path / "base.mfq"
    save(
        base,
        FileHeader(
            version=2,
            model_arch="qwen-test",
            extra={"hf_config": config, "custom_base_metadata": "preserved"},
        ),
        base_tensors,
    )
    output = tmp_path / "augmented.mfq"
    convert(
        hf_to_mfq.build_parser().parse_args(
            [
                "--input",
                str(root),
                "--base-mfq",
                str(base),
                "--output",
                str(output),
                "--quant-backend",
                "cpu",
                "--device",
                "cpu",
                "--row-chunk",
                "4",
            ]
        )
    )

    with open_mmap(base) as before, open_mmap(output) as after:
        assert after.header.model_arch == "qwen-test"
        assert after.header.extra["custom_base_metadata"] == "preserved"
        assert after.header.extra["mtp"]["included"] is True
        assert after.header.extra["mtp"]["tensor_count"] == 15
        assert after.read_blob("sentinel.weight") == before.read_blob("sentinel.weight")
        mtp_names = {name for name in after if name.startswith("predictor.")}
        assert len(mtp_names) == 15
        assert after.records["predictor.fusion.weight"].dtype == "BF16"
        assert after.records["predictor.output_norm.weight"].dtype == "BF16"
        assert after.records["predictor.block.0.attention.query.weight"].dtype == "F16"
        assert after.records["predictor.block.0.attention.norm.weight"].dtype == "F32"
        assert "model.language_model.layers.0.self_attn.q_proj.weight" not in after
        assert "model.block.0.attention.query.weight" in after
        graph = json.loads(after.read_blob(MODEL_GRAPH_ASSET))
        assert graph["schema_version"] == 1
        assert graph["architecture"] == "qwen3_5"
        assert graph["optional_components"]["predictor"] is True


def test_qwen4_mtp_plan_mirrors_mixed_nintm_expert_policy(tmp_path):
    hidden = 8
    expert_hidden = 4
    experts = 2
    config = {
        "model_type": "qwen4_exp_text",
        "hidden_size": hidden,
        "moe_intermediate_size": expert_hidden,
        "num_experts": experts,
        "num_hidden_layers": 1,
        "mtp_num_hidden_layers": 1,
    }
    root_names = {
        "mtp.fc_embedding.weight",
        "mtp.fc_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.hyper_connection_mixer.hc_norm.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    }
    layer_shapes = {
        "attn_hyper_connection.hc_norm.weight": (hidden,),
        "attn_hyper_connection.block_inject_weight.weight": (hidden, hidden),
        "mlp_hyper_connection.hc_norm.weight": (hidden,),
        "mlp_hyper_connection.block_inject_weight.weight": (hidden, hidden),
        "self_attn.q_proj.weight": (hidden, hidden),
        "self_attn.k_proj.weight": (hidden, hidden),
        "self_attn.v_proj.weight": (hidden, hidden),
        "self_attn.o_proj.weight": (hidden, hidden),
        "mlp.gate.weight": (experts, hidden),
    }
    inventory: dict[str, hf_to_mfq.SourceTensorMetadata] = {}
    for name in root_names:
        shape = (hidden,) if "norm" in name else (hidden, hidden)
        inventory[name] = hf_to_mfq.SourceTensorMetadata(
            name=name,
            shard="model.safetensors",
            shape=shape,
            dtype="BF16",
        )
    for suffix, shape in layer_shapes.items():
        name = "mtp.layers.0." + suffix
        inventory[name] = hf_to_mfq.SourceTensorMetadata(
            name=name,
            shard="model.safetensors",
            shape=shape,
            dtype="BF16",
        )
    for expert in range(experts):
        for projection, shape in (
            ("gate_proj", (expert_hidden, hidden)),
            ("up_proj", (expert_hidden, hidden)),
            ("down_proj", (hidden, expert_hidden)),
        ):
            name = f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight"
            inventory[name] = hf_to_mfq.SourceTensorMetadata(
                name=name,
                shard="model.safetensors",
                shape=shape,
                dtype="F8_E4M3",
            )

    plan = [
        TensorPlan(
            name=name,
            shard="model.safetensors",
            shape=metadata.shape,
            source_dtype=metadata.dtype,
            target_dtype="F16",
        )
        for name, metadata in inventory.items()
        if ".mlp.experts." not in name
    ]
    gate_up_name = "mtp.layers.0.mlp.experts.gate_up_proj"
    down_name = "mtp.layers.0.mlp.experts.down_proj"
    gate_up_shape = (experts, 2 * expert_hidden, hidden)
    down_shape = (experts, hidden, expert_hidden)
    plan.extend(
        (
            TensorPlan(
                name=gate_up_name,
                shard="model.safetensors",
                shape=gate_up_shape,
                source_dtype="F8_E4M3",
                target_dtype="NINTM",
                expert_shape=gate_up_shape,
                expert_precisions=(ExpertPrecision("NINT4", NintSpec(4, 8, 6)),) * experts,
            ),
            TensorPlan(
                name=down_name,
                shard="model.safetensors",
                shape=down_shape,
                source_dtype="F8_E4M3",
                target_dtype="NINTM",
                expert_shape=down_shape,
                expert_precisions=(ExpertPrecision("NINT4", NintSpec(4, 4, 6)),) * experts,
            ),
        )
    )

    low = NintSpec(2, 8, 5)
    high = NintSpec(4, 8, 6)

    def mixed(shape: tuple[int, int, int]) -> NintMoeTensor:
        rng = np.random.default_rng(sum(shape))
        pools = []
        for expert, spec in enumerate((low, high)):
            values = rng.normal(size=(shape[1], shape[2])).astype(np.float32)
            pools.append(
                NintMoePool(
                    np.asarray([expert], dtype=np.int32),
                    quantize_nint(values, spec),
                )
            )
        return NintMoeTensor(shape, tuple(pools))

    base_tensors: dict[str, object] = {
        MODEL_CONFIG_ASSET: json.dumps(config).encode(),
        "model.language_model.layers.0.mlp.experts.gate_up_proj": mixed(gate_up_shape),
        "model.language_model.layers.0.mlp.experts.down_proj": mixed(down_shape),
    }
    for suffix, shape in layer_shapes.items():
        base_tensors["model.language_model.layers.0." + suffix] = np.zeros(shape, dtype=np.float16)
    base = tmp_path / "qwen4-base.mfq"
    save(base, FileHeader(version=2, model_arch="qwen4-exp"), base_tensors)

    with open_mmap(base) as store:
        selected = hf_to_mfq._mtp_plan_from_base(
            plan,
            inventory,
            {"text_config": config},
            store,
        )

    selected_by_name = {item.name: item for item in selected}
    assert set(selected_by_name) == {item.name for item in plan}
    for name in (gate_up_name, down_name):
        item = selected_by_name[name]
        assert item.target_dtype == "NINTM"
        assert item.expert_precisions is not None
        assert tuple(value.nint_spec for value in item.expert_precisions) == (low, high)


def test_recipe_dense_types_preserve_bf16_separately_from_f16():
    assert _dtype_for_recipe_type("F32", "F32") == "F32"
    assert _dtype_for_recipe_type("F32", "F16") == "F32"
    assert _dtype_for_recipe_type("F16", "F32") == "F16"
    assert _dtype_for_recipe_type("BF16", "F32") == "BF16"


@pytest.mark.parametrize(
    "recipe_type,target",
    [
        ("IQ1_M", "NVQ1-L"),
        ("IQ2_S", "NVQ2J-XL"),
        ("IQ2_XS", "NVQ2J-L"),
        ("IQ2_XXS", "NVQ2J"),
        ("IQ3_S", "NVQ3J-L"),
        ("IQ3_XXS", "NVQ3J"),
        ("Q4_0", "NINT4"),
        ("Q4_1", "NINT4"),
        ("Q8_0", "NINT8"),
    ],
)
def test_hf_recipe_uses_the_same_compact_family_mapping_as_gguf(
    recipe_type,
    target,
):
    assert _dtype_for_recipe_type(recipe_type, "F32") == target


def test_hf_recipe_plan_keeps_iq_tensor_as_vq(tmp_path):
    root = tmp_path / "hf-recipe-vq"
    root.mkdir()
    name = "model.language_model.layers.0.mlp.down_proj.weight"
    save_file(
        {name: torch.zeros((8, 24), dtype=torch.bfloat16)},
        root / "model.safetensors",
    )
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )

    plan = build_hf_plan(
        root,
        False,
        {"blk.0.ffn_down.weight": "IQ2_XXS"},
        "F32",
    )

    assert len(plan) == 1
    assert plan[0].target_dtype == "NVQ2J"
    assert plan[0].gguf_type == "IQ2_XXS"


def test_hf_recipe_family_flags_remap_the_default_iq3_xxs_profile(tmp_path):
    root = tmp_path / "hf-recipe-iq3"
    root.mkdir()
    name = "model.language_model.layers.0.mlp.down_proj.weight"
    save_file(
        {name: torch.zeros((8, 24), dtype=torch.bfloat16)},
        root / "model.safetensors",
    )
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )
    plan = build_hf_plan(
        root,
        False,
        {"blk.0.ffn_down.weight": "IQ3_XXS"},
        "F32",
    )

    mapped = hf_to_mfq._apply_recipe_family_mappings(
        plan,
        npq0_l=False,
        nvq3_jsc=False,
        nvq3_jsc_512=True,
        nvq3_to_nint3=False,
        iq2_s_to_nint2=False,
        q8_to_nint8_zero=False,
    )
    assert mapped[0].target_dtype == "NVQ3J-512"

    mapped = hf_to_mfq._apply_recipe_family_mappings(
        plan,
        npq0_l=False,
        nvq3_jsc=False,
        nvq3_jsc_512=False,
        nvq3_to_nint3=True,
        iq2_s_to_nint2=False,
        q8_to_nint8_zero=False,
    )
    assert mapped[0].target_dtype == "NINT3"


def test_hf_and_gguf_recipe_family_tables_cannot_diverge():
    from mfq.tools import quantize_gguf_to_mfq as gguf_to_mfq

    assert hf_to_mfq._RECIPE_TARGETS == gguf_to_mfq._RECIPE_TARGETS


def test_artifact_provenance_omits_local_directories(tmp_path):
    assert hf_to_mfq._artifact_provenance_name(str(tmp_path / "Q4_0.gguf")) == "Q4_0.gguf"


@pytest.mark.parametrize(
    "suffix,gguf_suffix",
    [
        ("experts.gate_up_proj", "ffn_gate_up_exps.weight"),
        ("experts.down_proj", "ffn_down_exps.weight"),
        ("router.proj.weight", "ffn_gate_inp.weight"),
        ("router.scale", "ffn_gate_inp.scale"),
        ("router.per_expert_scale", "ffn_down_exps.scale"),
        ("layer_scalar", "layer_output_scale.weight"),
        ("pre_feedforward_layernorm.weight", "ffn_norm.weight"),
        ("pre_feedforward_layernorm_2.weight", "pre_ffw_norm_2.weight"),
        ("post_feedforward_layernorm.weight", "post_ffw_norm.weight"),
        ("post_feedforward_layernorm_1.weight", "post_ffw_norm_1.weight"),
        ("post_feedforward_layernorm_2.weight", "post_ffw_norm_2.weight"),
    ],
)
def test_gemma4_hf_to_gguf_name_mapping(suffix, gguf_suffix):
    assert _hf_to_gguf_name(f"model.language_model.layers.29.{suffix}") == f"blk.29.{gguf_suffix}"


def test_minicpmo45_hf_to_gguf_name_mapping():
    assert _hf_to_gguf_name("llm.model.embed_tokens.weight") == "token_embd.weight"
    assert _hf_to_gguf_name("llm.model.norm.weight") == "output_norm.weight"
    assert _hf_to_gguf_name("llm.lm_head.weight") == "output.weight"
    assert (
        _hf_to_gguf_name("llm.model.layers.3.post_attention_layernorm.weight")
        == "blk.3.ffn_norm.weight"
    )
    assert _hf_to_gguf_name("llm.model.layers.3.self_attn.q_proj.weight") == "blk.3.attn_q.weight"
    assert _hf_to_gguf_name("llm.model.layers.3.mlp.down_proj.weight") == "blk.3.ffn_down.weight"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("llm.model.layers.0.self_attn.q_proj.weight", True),
        ("vpm.encoder.layers.0.self_attn.q_proj.weight", True),
        ("apm.layers.0.self_attn.q_proj.weight", True),
        ("tts.emb_text.weight", True),
        ("resampler.attn.in_proj_weight", False),
        ("resampler.attn.out_proj.weight", False),
        ("resampler.proj", False),
        ("resampler.query", False),
        ("apm.embed_positions.weight", False),
        ("vpm.embeddings.position_embedding.weight", False),
        ("tts.head_code.0.parametrizations.weight.original1", False),
    ],
)
def test_minicpmo45_quantization_policy(name, expected):
    assert _minicpmo45_quantizable_matrix(name, (8, 8)) is expected


def test_minicpmo45_plan_preserves_raw_graph_matrices(tmp_path):
    root = tmp_path / "minicpmo45"
    root.mkdir()
    tensors = {
        "llm.model.layers.0.self_attn.q_proj.weight": torch.zeros((8, 8), dtype=torch.bfloat16),
        "vpm.encoder.layers.0.self_attn.q_proj.weight": torch.zeros((8, 8), dtype=torch.bfloat16),
        "resampler.attn.in_proj_weight": torch.zeros((24, 8), dtype=torch.bfloat16),
        "resampler.attn.out_proj.weight": torch.zeros((8, 8), dtype=torch.bfloat16),
        "resampler.proj": torch.zeros((8, 8), dtype=torch.bfloat16),
        "resampler.query": torch.zeros((4, 8), dtype=torch.bfloat16),
        "apm.embed_positions.weight": torch.zeros((16, 8), dtype=torch.bfloat16),
        "vpm.embeddings.position_embedding.weight": torch.zeros((16, 8), dtype=torch.bfloat16),
        "tts.head_code.0.parametrizations.weight.original1": torch.zeros(
            (8, 8), dtype=torch.bfloat16
        ),
        "apm.conv1.weight": torch.zeros((8, 8, 3), dtype=torch.bfloat16),
    }
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(
        json.dumps({"model_type": "minicpmo", "version": "4.5"}),
        encoding="utf-8",
    )

    plan = build_hf_plan(root, False, None, "F16")
    targets = {item.name: item.target_dtype for item in plan}
    names_by_source = {item.source_name: item.name for item in plan}

    text_name = "model.block.0.attention.query.weight"
    vision_name = "vision.block.0.attention.query.weight"
    assert names_by_source["llm.model.layers.0.self_attn.q_proj.weight"] == text_name
    assert names_by_source["vpm.encoder.layers.0.self_attn.q_proj.weight"] == vision_name
    assert targets[text_name] == "NINT4"
    assert targets[vision_name] == "BF16"
    for source_name in tensors:
        if source_name not in {
            "llm.model.layers.0.self_attn.q_proj.weight",
            "vpm.encoder.layers.0.self_attn.q_proj.weight",
        }:
            assert targets[names_by_source[source_name]] == "BF16"

    text_plan = build_hf_plan(root, True, None, "F16")
    assert [item.name for item in text_plan] == [text_name]

    all_quantized = build_hf_plan(
        root,
        False,
        None,
        "F16",
        quantize_vision=True,
    )
    assert {item.name: item.target_dtype for item in all_quantized}[
        vision_name
    ] == "NINT4"


def test_minicpmo45_llm_recipe_keeps_other_components_at_source_precision(tmp_path):
    root = tmp_path / "minicpmo45-recipe"
    root.mkdir()
    tensors = {
        "llm.model.layers.0.self_attn.q_proj.weight": torch.zeros((8, 8), dtype=torch.bfloat16),
        "vpm.encoder.layers.0.self_attn.q_proj.weight": torch.zeros((8, 8), dtype=torch.bfloat16),
        "tts.emb_text.weight": torch.zeros((8, 8), dtype=torch.bfloat16),
    }
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(
        json.dumps({"model_type": "minicpmo", "version": "4.5"}),
        encoding="utf-8",
    )

    plan = build_hf_plan(
        root,
        False,
        {"blk.0.attn_q.weight": "Q5_K"},
        "F16",
    )
    targets = {item.name: item.target_dtype for item in plan}

    assert targets == {
        "model.block.0.attention.query.weight": "NINT5",
        "tts.text_embedding.weight": "BF16",
        "vision.block.0.attention.query.weight": "BF16",
    }


def test_minicpmo45_llm_recipe_rejects_an_unmapped_language_tensor(tmp_path):
    root = tmp_path / "minicpmo45-incomplete-recipe"
    root.mkdir()
    name = "llm.model.layers.0.self_attn.k_proj.weight"
    save_file(
        {name: torch.zeros((8, 8), dtype=torch.bfloat16)},
        root / "model.safetensors",
    )
    (root / "config.json").write_text(
        json.dumps({"model_type": "minicpmo", "version": "4.5"}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="language tensor is absent from the GGUF recipe",
    ):
        build_hf_plan(
            root,
            False,
            {"blk.0.attn_q.weight": "Q5_K"},
            "F16",
        )


def test_q5_1_recipe_maps_to_nint5():
    assert _dtype_for_recipe_type("Q5_1", "F32") == "NINT5"


def test_q3_k_recipe_maps_to_nint3():
    assert _dtype_for_recipe_type("Q3_K", "F32") == "NINT3"


def test_q2_k_recipe_maps_to_nint2():
    assert _dtype_for_recipe_type("Q2_K", "F32") == "NINT2"


def test_hf_imatrix_prefers_the_tensor_canonical_name_over_recipe_anchor(
    tmp_path,
):
    item = TensorPlan(
        name="model.language_model.layers.3.self_attn.k_proj.weight",
        shard="model.safetensors",
        shape=(2, 4),
        source_dtype="BF16",
        target_dtype="NINT4",
        gguf_name="blk.3.attn_q.weight",
    )
    values = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    imatrix = ImportanceMatrix(
        path=tmp_path / "imatrix.gguf",
        entries={
            "blk.3.attn_k.weight": ImportanceEntry(
                values=values,
                counts=np.asarray([8], dtype=np.int64),
            )
        },
        datasets=("test",),
        chunk_count=1,
        chunk_size=4,
        legacy=False,
    )

    binding = _bind_hf_imatrix(imatrix, [item])[item.name]

    assert binding.entry_name == "blk.3.attn_k.weight"
    np.testing.assert_array_equal(binding.rows(0, 2), values[0])


def test_hf_imatrix_binds_expert_wise_entries(tmp_path):
    item = TensorPlan(
        name="model.language_model.layers.4.mlp.experts.down_proj",
        shard="model.safetensors",
        shape=(2, 3, 4),
        source_dtype="BF16",
        target_dtype="NINTM",
        gguf_name="blk.4.ffn_down_exps.weight",
        expert_shape=(2, 3, 4),
        expert_precisions=(
            ExpertPrecision("NINT4", nint_spec=NintSpec(4, 24, 6)),
            ExpertPrecision("NINT8", nint_spec=NintSpec(8, 48, 7)),
        ),
    )
    values = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    imatrix = ImportanceMatrix(
        path=tmp_path / "imatrix.gguf",
        entries={
            "blk.4.ffn_down_exps.weight": ImportanceEntry(
                values=values,
                counts=np.asarray([8, 8], dtype=np.int64),
            )
        },
        datasets=(),
        chunk_count=1,
        chunk_size=4,
        legacy=False,
    )

    binding = _bind_hf_imatrix(imatrix, [item])[item.name]

    np.testing.assert_array_equal(binding.rows(2, 5), values[[0, 1, 1]])
    np.testing.assert_array_equal(binding.selected(np.asarray([0, 3], dtype=np.int64)), values)


def test_hf_imatrix_binds_an_ordinary_vq_tensor(tmp_path):
    item = TensorPlan(
        name="model.language_model.layers.2.mlp.down_proj.weight",
        shard="model.safetensors",
        shape=(4, 24),
        source_dtype="BF16",
        target_dtype="NVQ2J",
        gguf_name="blk.2.ffn_down.weight",
        gguf_type="IQ2_XXS",
    )
    values = np.linspace(0.25, 2.0, 24, dtype=np.float32).reshape(1, 24)
    imatrix = ImportanceMatrix(
        path=tmp_path / "imatrix.gguf",
        entries={
            "blk.2.ffn_down.weight": ImportanceEntry(
                values=values,
                counts=np.asarray([32], dtype=np.int64),
            )
        },
        datasets=("test",),
        chunk_count=1,
        chunk_size=24,
        legacy=False,
    )

    binding = _bind_hf_imatrix(imatrix, [item])[item.name]

    assert binding.entry_name == "blk.2.ffn_down.weight"
    np.testing.assert_array_equal(binding.rows(0, 4), values[0])


def test_hf_convert_passes_imatrix_rows_to_nint_writer(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "hf"
    root.mkdir()
    tensor_name = "model.language_model.layers.0.mlp.down_proj.weight"
    save_file(
        {
            tensor_name: torch.linspace(-2.0, 2.0, steps=4 * 24, dtype=torch.float32)
            .reshape(4, 24)
            .to(torch.bfloat16)
        },
        root / "model.safetensors",
    )
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    imatrix_path = tmp_path / "imatrix.gguf"
    imatrix_path.write_bytes(b"test")
    importance = np.linspace(0.25, 2.0, 24, dtype=np.float32).reshape(1, 24)
    imatrix = ImportanceMatrix(
        path=imatrix_path,
        entries={
            "blk.0.ffn_down.weight": ImportanceEntry(
                values=importance,
                counts=np.asarray([16], dtype=np.int64),
            )
        },
        datasets=("unit-test",),
        chunk_count=1,
        chunk_size=24,
        legacy=False,
    )
    monkeypatch.setattr(hf_to_mfq, "load_importance_matrix", lambda _path: imatrix)
    original_writer = hf_to_mfq._write_nint_axis0_blob
    captured: list[np.ndarray] = []

    def recording_writer(*args, **kwargs):
        importance_rows = kwargs.get("importance_rows")
        assert importance_rows is not None
        captured.append(np.asarray(importance_rows(0, 1)).copy())
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(hf_to_mfq, "_write_nint_axis0_blob", recording_writer)
    output = tmp_path / "model.mfq"
    args = hf_to_mfq.build_parser().parse_args(
        [
            "--input",
            str(root),
            "--output",
            str(output),
            "--imatrix",
            str(imatrix_path),
            "--quant-backend",
            "cpu",
            "--device",
            "cpu",
            "--row-chunk",
            "4",
        ]
    )

    convert(args)

    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], importance[0])
    header, store = load_mmap(output)
    try:
        assert header.extra["imatrix"]["bindings"] == {
            "model.block.0.mlp.down.weight": "blk.0.ffn_down.weight"
        }
    finally:
        store.close()


def test_hf_default_streaming_writer_matches_explicit_staged_blobs(tmp_path, capsys):
    root = tmp_path / "hf-writer"
    root.mkdir()
    save_file(
        {
            "model.language_model.layers.0.mlp.down_proj.weight": torch.linspace(
                -2.0,
                2.0,
                steps=4 * 24,
                dtype=torch.float32,
            )
            .reshape(4, 24)
            .to(torch.bfloat16),
            "model.language_model.norm.weight": torch.arange(24, dtype=torch.float32),
        },
        root / "model.safetensors",
    )
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )

    streaming_output = tmp_path / "streaming.mfq"
    common = [
        "--input",
        str(root),
        "--quant-backend",
        "cpu",
        "--device",
        "cpu",
        "--row-chunk",
        "4",
    ]
    streaming_args = hf_to_mfq.build_parser().parse_args(
        [*common, "--output", str(streaming_output)]
    )
    assert streaming_args.staged_blobs is False
    convert(streaming_args)

    staged_output = tmp_path / "staged.mfq"
    staged_args = hf_to_mfq.build_parser().parse_args(
        [*common, "--output", str(staged_output), "--staged-blobs"]
    )
    assert staged_args.staged_blobs is True
    convert(staged_args)

    assert streaming_output.read_bytes() == staged_output.read_bytes()
    assert not (tmp_path / ".streaming.mfq.streaming").exists()
    assert not (tmp_path / ".streaming.mfq.tmp_blobs").exists()
    assert not (tmp_path / ".staged.mfq.tmp_blobs").exists()

    completed_output = capsys.readouterr().out
    assert '"writer_mode": "streaming"' in completed_output
    assert '"writer_mode": "staged_blobs"' in completed_output
    for option in ("--resume-temp", "--keep-temp"):
        implied_args = hf_to_mfq.build_parser().parse_args(
            [
                *common,
                "--output",
                str(tmp_path / f"implied-{option[2:]}.mfq"),
                "--dry-run",
                option,
            ]
        )
        convert(implied_args)
        assert '"writer_mode": "staged_blobs"' in capsys.readouterr().out


def test_hf_convert_writes_an_ordinary_vq_tensor_via_precision_override(
    tmp_path,
):
    root = tmp_path / "hf-vq"
    root.mkdir()
    tensor_name = "model.language_model.layers.0.mlp.down_proj.weight"
    save_file(
        {
            tensor_name: torch.linspace(
                -2.0,
                2.0,
                steps=8 * 24,
                dtype=torch.float32,
            )
            .reshape(8, 24)
            .to(torch.bfloat16)
        },
        root / "model.safetensors",
    )
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps({"blk.0.ffn_down.weight": "NVQ2"}),
        encoding="utf-8",
    )
    output = tmp_path / "model-vq.mfq"
    args = hf_to_mfq.build_parser().parse_args(
        [
            "--input",
            str(root),
            "--output",
            str(output),
            "--tensor-precision-overrides",
            str(overrides),
            "--nvq-codebook-scope",
            "fixed",
            "--quant-backend",
            "cpu",
            "--device",
            "cpu",
            "--row-chunk",
            "8",
        ]
    )

    convert(args)

    header, store = load_mmap(output)
    try:
        assert store.records["model.block.0.mlp.down.weight"].dtype == "NVQ2"
        assert header.extra["target_counts"] == {"NVQ2": 1}
        assert header.extra["tensor_precision_overrides"] == {"blk.0.ffn_down.weight": "NVQ2"}
    finally:
        store.close()


def test_hf_convert_trains_and_writes_tensorwise_jsc_vq(tmp_path):
    root = tmp_path / "hf-jsc"
    root.mkdir()
    tensor_name = "model.language_model.layers.0.mlp.down_proj.weight"
    generator = torch.Generator().manual_seed(17)
    save_file(
        {
            tensor_name: torch.randn((16, 24), generator=generator, dtype=torch.float32).to(
                torch.bfloat16
            )
        },
        root / "model.safetensors",
    )
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )
    overrides = tmp_path / "jsc-overrides.json"
    overrides.write_text(
        json.dumps({"blk.0.ffn_down.weight": "NVQ2J"}),
        encoding="utf-8",
    )
    output = tmp_path / "model-jsc.mfq"
    args = hf_to_mfq.build_parser().parse_args(
        [
            "--input",
            str(root),
            "--output",
            str(output),
            "--tensor-precision-overrides",
            str(overrides),
            "--quant-backend",
            "cpu",
            "--device",
            "cpu",
            "--row-chunk",
            "8",
            "--nvq-jsc-banks",
            "1",
            "--nvq-jsc-iterations",
            "1",
            "--nvq-codebook-train-rows",
            "8",
            "--nvq-codebook-validation-rows",
            "4",
        ]
    )

    convert(args)

    header, store = load_mmap(output)
    try:
        assert store.records["model.block.0.mlp.down.weight"].dtype == "NVQ2J"
        result = header.extra["nvq_codebooks"]["model.block.0.mlp.down.weight"]
        assert result["loaded"] is False
        assert Path(result["artifact"]).is_file()
    finally:
        store.close()


def test_hf_convert_matches_llamacpp_mostly_bf16_policy(tmp_path):
    root = tmp_path / "hf-bf16"
    root.mkdir()
    f32_matrix = torch.tensor(
        [[1.00390625, 1.01171875], [-2.0078125, 3.1415927]],
        dtype=torch.float32,
    )
    tensors = {
        "model.language_model.embed_tokens.weight": torch.tensor(
            [[1.0, -2.5], [3.25, 0.125]], dtype=torch.bfloat16
        ),
        "model.language_model.norm.weight": torch.tensor([0.75, 1.5], dtype=torch.float32),
        "lm_head.weight": f32_matrix,
        "model.language_model.layers.0.linear_attn.conv1d.weight": torch.tensor(
            [[0.125, -0.25], [0.5, 2.0]], dtype=torch.float32
        ),
        "model.language_model.position_ids": torch.tensor([0, 1], dtype=torch.int64),
    }
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    output = tmp_path / "model-bf16.mfq"
    args = hf_to_mfq.build_parser().parse_args(
        [
            "--input",
            str(root),
            "--output",
            str(output),
            "--bf16",
        ]
    )

    convert(args)

    header, store = load_mmap(output)
    try:
        assert header.model_arch == "qwen3_5-hf-mfq-bf16"
        assert header.extra["policy"] == "mostly-BF16;1d-and-special=F32"
        assert header.extra["mostly_bf16"] is True
        assert "recipe" not in header.extra
        assert "quant_backend" not in header.extra
        assert "device" not in header.extra
        assert header.extra["target_counts"] == {"BF16": 2, "F32": 2, "I64": 1}
        assert store.records["model.token_embedding.weight"].dtype == "BF16"
        assert store.records["model.output.weight"].dtype == "BF16"
        assert store.records["model.output_norm.weight"].dtype == "F32"
        assert store.records["model.block.0.linear_attention.conv.weight"].dtype == "F32"
        assert store.records["model.position_ids"].dtype == "I64"
        restored = store["model.token_embedding.weight"]
        assert is_bfloat16_array(restored)
        np.testing.assert_array_equal(
            restored,
            tensors["model.language_model.embed_tokens.weight"].view(torch.uint16).numpy(),
        )
        # Match ggml_compute_fp32_to_bf16: quiet NaNs and round-to-nearest-even.
        source_bits = f32_matrix.numpy().view(np.uint32)
        source_bits = np.where(
            (source_bits & 0x7FFFFFFF) > 0x7F800000,
            (source_bits & np.uint32(0xFFFF0000)) | np.uint32(64 << 16),
            source_bits,
        )
        expected_bf16 = (
            (source_bits.astype(np.uint64) + np.uint64(0x7FFF) + ((source_bits >> 16) & 1)) >> 16
        ).astype(np.uint16)
        np.testing.assert_array_equal(store["model.output.weight"], expected_bf16)
        np.testing.assert_array_equal(
            store["model.output_norm.weight"],
            tensors["model.language_model.norm.weight"].numpy(),
        )
        np.testing.assert_array_equal(
            store["model.block.0.linear_attention.conv.weight"],
            tensors["model.language_model.layers.0.linear_attn.conv1d.weight"].numpy(),
        )
    finally:
        store.close()


def test_glm_dsa_plan_derives_headwise_mla_and_streamed_experts(tmp_path):
    root = tmp_path / "glm"
    root.mkdir()
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_size": 24,
        "kv_lora_rank": 24,
        "qk_nope_head_dim": 8,
        "v_head_dim": 12,
        "n_routed_experts": 3,
        "moe_intermediate_size": 24,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "mlp_layer_types": ["dense", "sparse"],
    }
    tensors = {
        f"model.layers.{layer}.self_attn.kv_b_proj.weight": torch.arange(
            2 * (8 + 12) * 24, dtype=torch.float32
        ).reshape(2 * (8 + 12), 24)
        for layer in range(3)
    }
    for expert in range(3):
        base = f"model.layers.1.mlp.experts.{expert}."
        tensors[base + "gate_proj.weight"] = torch.full((24, 24), float(10 * expert + 1))
        tensors[base + "up_proj.weight"] = torch.full((24, 24), float(10 * expert + 2))
        tensors[base + "down_proj.weight"] = torch.full((24, 24), float(10 * expert + 3))
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    plan = build_hf_plan(root, True, None, "F16")
    by_name = {item.name: item for item in plan}
    assert len(plan) == 7
    assert not any(name.startswith("model.block.2.") for name in by_name)
    embed_name = "model.block.0.attention.latent.query_embedding.weight"
    unembed_name = "model.block.0.attention.latent.output_unembedding.weight"
    gate_name = "model.block.1.mlp.experts.gate.weight"
    up_name = "model.block.1.mlp.experts.up.weight"
    embed = by_name[embed_name]
    unembed = by_name[unembed_name]
    assert embed.expert_shape == (2, 24, 8)
    assert unembed.expert_shape == (2, 12, 24)

    source = tensors["model.layers.0.self_attn.kv_b_proj.weight"]
    embed_weight = _transform_glm_kv_b(source, embed)
    unembed_weight = _transform_glm_kv_b(source, unembed)
    source_heads = source.reshape(2, 20, 24)
    assert torch.equal(embed_weight, source_heads[:, :8].transpose(1, 2))
    assert torch.equal(unembed_weight, source_heads[:, 8:])

    gate = by_name[gate_name]
    stream = _GlmExpertRowSource(
        root,
        gate.expert_shape,
        gate.expert_source_names,
        gate.expert_source_shards,
    )
    try:
        expert_one = stream[24:48]
    finally:
        stream.close()
    assert expert_one.shape == (24, 24)
    assert torch.equal(expert_one, tensors["model.layers.1.mlp.experts.1.gate_proj.weight"])
    assert by_name[up_name].expert_shape == (3, 24, 24)

    output = tmp_path / "tiny-glm.mfq"
    args = argparse.Namespace(
        input=str(root),
        output=str(output),
        bits=4,
        groupsize=24,
        sub_bits=6,
        row_chunk=4,
        quant_backend="cpu",
        device="cpu",
        text_only=True,
        recipe_gguf="",
        calibration_scheme="",
        dense_dtype="f16",
        limit_tensors=0,
        dry_run=False,
        overwrite=False,
        keep_temp=False,
        resume_temp=False,
        temp_dir="",
    )
    convert(args)
    _header, store = load_mmap(output)
    try:
        assert {name for name in store.records if not is_asset_record(name)} == set(by_name)
        assert all(
            record.dtype == "NINTM"
            for record in store.records.values()
            if not is_asset_record(record.name)
        )
        assert store[embed_name].shape == (2, 24, 8)
        assert store[gate_name].shape == (3, 24, 24)
        assert store[up_name].shape == (3, 24, 24)
    finally:
        store.close()

    split_output = tmp_path / "tiny-glm-split.mfq"
    split_args = argparse.Namespace(**vars(args))
    split_args.output = str(split_output)
    split_args.split_max_size = 0
    split_args.split_max_tensors = 2
    convert(split_args)
    last_shard = format_shard_path(split_output, 4, 4)
    _header, split_store = load_mmap(last_shard)
    try:
        assert len(split_store.paths) == 4
        assert {name for name in split_store.records if not is_asset_record(name)} == set(by_name)
        assert split_store[gate_name].shape == (3, 24, 24)
        assert split_store[up_name].shape == (3, 24, 24)
    finally:
        split_store.close()


@pytest.mark.parametrize(
    "outer_type,text_type,expert_key,layer",
    [
        ("qwen4_exp", "qwen4_exp_text", "num_experts", 0),
        ("glm5_next", "glm5_next_text", "n_routed_experts", 3),
    ],
)
def test_separate_expert_plan_dequantizes_fp8_without_coupling_projections(
    tmp_path,
    outer_type,
    text_type,
    expert_key,
    layer,
):
    root = tmp_path / outer_type
    root.mkdir()
    hidden = 24
    expert_hidden = 24
    experts = 2
    text_config = {
        "model_type": text_type,
        "hidden_size": hidden,
        "moe_intermediate_size": expert_hidden,
        expert_key: experts,
        "num_hidden_layers": layer + 1,
        "mtp_num_hidden_layers": 0,
    }
    tensors = {}
    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    integer_metadata = "model.language_model.runtime_hash_metadata"
    tensors[integer_metadata] = torch.tensor(
        [23703573157769, 20109073645365],
        dtype=torch.int64,
    )
    for expert in range(experts):
        for projection, multiplier, shape in (
            ("gate_proj", 0.01, (expert_hidden, hidden)),
            ("up_proj", 0.02, (expert_hidden, hidden)),
            ("down_proj", 0.03, (hidden, expert_hidden)),
        ):
            name = f"{prefix}.{expert}.{projection}.weight"
            tensors[name] = torch.full(
                shape,
                float(10 * expert + {"gate_proj": 1, "up_proj": 2, "down_proj": 3}[projection]),
            ).to(torch.float8_e4m3fn)
            tensors[name + "_scale_inv"] = torch.tensor([[multiplier]])
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(
        json.dumps({"model_type": outer_type, "text_config": text_config}),
        encoding="utf-8",
    )

    plan = build_hf_plan(root, True, None, "F16")
    canonical_prefix = f"model.block.{layer}.mlp.experts"
    canonical_metadata = "model.runtime.hash_metadata"
    assert {item.name for item in plan} == {
        canonical_prefix + ".gate.weight",
        canonical_prefix + ".up.weight",
        canonical_prefix + ".down.weight",
        canonical_metadata,
    }
    assert all(
        item.target_dtype == "NINTM"
        for item in plan
        if item.name != canonical_metadata
    )
    assert next(
        item for item in plan if item.name == canonical_metadata
    ).target_dtype == "I64"
    assert not any("scale_inv" in item.name for item in plan)

    for projection, multiplier in (("gate", 0.01), ("up", 0.02)):
        item = next(
            value for value in plan if value.name.endswith(f".{projection}.weight")
        )
        stream = _GlmExpertRowSource(
            root,
            item.expert_shape,
            item.expert_source_names,
            item.expert_source_shards,
            item.expert_source_quantizations,
            item.expert_source_scale_names,
            item.expert_source_scale_shards,
        )
        try:
            expert_one = stream[expert_hidden : 2 * expert_hidden]
        finally:
            stream.close()
        assert expert_one.shape == (expert_hidden, hidden)
        torch.testing.assert_close(
            expert_one,
            tensors[f"{prefix}.1.{projection}_proj.weight"].float() * multiplier,
        )


def test_flash_next_conversion_embeds_self_contained_python_runtime_assets(tmp_path):
    root = tmp_path / "qwen4-exp"
    root.mkdir()
    save_file(
        {"lm_head.weight": torch.ones((2, 2), dtype=torch.bfloat16)},
        root / "model.safetensors",
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text"},
            }
        ),
        encoding="utf-8",
    )
    expected = {
        HF_TOKENIZER_JSON_ASSET: b'{"model":{"type":"BPE"}}',
        HF_TOKENIZER_CONFIG_ASSET: b'{"eos_token":"<eos>"}',
        HF_CHAT_TEMPLATE_ASSET: b"{{ messages }}",
        HF_GENERATION_CONFIG_ASSET: b'{"max_new_tokens":32}',
    }
    for name, record in (
        ("tokenizer.json", HF_TOKENIZER_JSON_ASSET),
        ("tokenizer_config.json", HF_TOKENIZER_CONFIG_ASSET),
        ("chat_template.jinja", HF_CHAT_TEMPLATE_ASSET),
        ("generation_config.json", HF_GENERATION_CONFIG_ASSET),
    ):
        (root / name).write_bytes(expected[record])
    output = tmp_path / "qwen4-exp.mfq"
    args = hf_to_mfq.build_parser().parse_args(
        [
            "--input",
            str(root),
            "--output",
            str(output),
            "--bf16",
            "--quant-backend",
            "cpu",
            "--device",
            "cpu",
        ]
    )

    convert(args)

    with open_mmap(output) as store:
        for record, payload in expected.items():
            assert store[record] == payload
            manifest = store.header.extra["runtime_assets"]["assets"]
            assert record.removeprefix("__mfq_asset__/") in manifest


def test_glm5_next_plan_derives_scaled_fp8_mla_for_backbone_and_mtp(tmp_path):
    root = tmp_path / "glm5-next"
    root.mkdir()
    hidden = 24
    heads = 2
    kv_rank = 24
    nope = 8
    value = 12
    expert_hidden = 24
    text_config = {
        "model_type": "glm5_next_text",
        "hidden_size": hidden,
        "num_hidden_layers": 4,
        "num_attention_heads": heads,
        "kv_lora_rank": kv_rank,
        "qk_nope_head_dim": nope,
        "v_head_dim": value,
        "n_routed_experts": 1,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": expert_hidden,
        "num_nextn_predict_layers": 1,
    }
    tensors: dict[str, torch.Tensor] = {}
    source_names = []
    for layer, multiplier in ((3, 0.25), (4, 0.5)):
        attention = f"model.language_model.layers.{layer}.self_attn"
        source_name = attention + ".kv_b_proj.weight"
        source_names.append(source_name)
        source = torch.arange(
            heads * (nope + value) * kv_rank,
            dtype=torch.float32,
        ).reshape(heads * (nope + value), kv_rank)
        source = ((source % 17) - 8).to(torch.float8_e4m3fn)
        tensors[source_name] = source
        tensors[source_name + "_scale_inv"] = torch.tensor([[multiplier]])
        expert = f"model.language_model.layers.{layer}.mlp.experts.0"
        for projection, shape in (
            ("gate_proj", (expert_hidden, hidden)),
            ("up_proj", (expert_hidden, hidden)),
            ("down_proj", (hidden, expert_hidden)),
        ):
            tensors[f"{expert}.{projection}.weight"] = torch.ones(shape)
    predictor = "model.language_model.layers.4."
    for suffix, shape in {
        "enorm.weight": (hidden,),
        "hnorm.weight": (hidden,),
        "eh_proj.weight": (hidden, 2 * hidden),
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
        "self_attn.q_a_proj.weight": (hidden, hidden),
        "self_attn.kv_a_proj_with_mqa.weight": (hidden, hidden),
        "self_attn.o_proj.weight": (hidden, hidden),
        "mlp.gate.weight": (1, hidden),
        "shared_head.norm.weight": (hidden,),
    }.items():
        tensors[predictor + suffix] = torch.ones(shape)
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(
        json.dumps({"model_type": "glm5_next", "text_config": text_config}),
        encoding="utf-8",
    )

    plan = build_hf_plan(root, True, None, "F16")
    by_name = {item.name: item for item in plan}
    for layer, multiplier in ((3, 0.25), (4, 0.5)):
        attention = f"model.language_model.layers.{layer}.self_attn"
        assert attention + ".kv_b_proj.weight" not in by_name
        canonical = (
            "model.block.3.attention"
            if layer == 3
            else "predictor.block.0.attention"
        )
        embed = by_name[canonical + ".latent.query_embedding.weight"]
        unembed = by_name[canonical + ".latent.output_unembedding.weight"]
        assert embed.expert_shape == (heads, kv_rank, nope)
        assert unembed.expert_shape == (heads, value, kv_rank)
        assert embed.source_quantization == "fp8_block128_inv"
        dequantized = hf_to_mfq._raw_source_for_plan(root, embed)[:]
        expected = tensors[source_names[layer == 4]].float() * multiplier
        torch.testing.assert_close(dequantized, expected)
        transformed = _transform_glm_kv_b(dequantized, embed)
        expected_heads = expected.reshape(heads, nope + value, kv_rank)
        torch.testing.assert_close(
            transformed,
            expected_heads[:, :nope].transpose(1, 2),
        )

    assert sum(name.endswith(".latent.query_embedding.weight") for name in by_name) == 2
    assert sum(name.endswith(".latent.output_unembedding.weight") for name in by_name) == 2
    assert sum(name.endswith(".experts.gate.weight") for name in by_name) == 2
    assert sum(name.endswith(".experts.up.weight") for name in by_name) == 2
    assert sum(name.endswith(".experts.down.weight") for name in by_name) == 2


def test_glm_expert_row_source_streams_across_shards(tmp_path):
    root = tmp_path / "glm-shards"
    root.mkdir()
    gate = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    up = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
    save_file({"expert.gate": gate}, root / "gate.safetensors")
    save_file({"expert.up": up}, root / "up.safetensors")
    stream = _GlmExpertRowSource(
        root,
        (1, 4, 3),
        (("expert.gate", "expert.up"),),
        (("gate.safetensors", "up.safetensors"),),
    )
    try:
        rows = stream[1:4]
    finally:
        stream.close()
    assert torch.equal(rows, torch.cat((gate[1:], up), dim=0))
