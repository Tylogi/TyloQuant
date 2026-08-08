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
from mfq.formats.assets import is_asset_record
from mfq.formats.io import is_bfloat16_array, load_mmap
from mfq.formats.nint import NintSpec
from mfq.formats.shards import format_shard_path
from mfq.quantize.imatrix import ImportanceEntry, ImportanceMatrix
from mfq.tools.quantize_hf_to_mfq import (
    TensorPlan,
    _bind_hf_imatrix,
    _dtype_for_recipe_type,
    _GlmExpertRowSource,
    _hf_to_gguf_name,
    _minicpmo45_quantizable_matrix,
    _RawSafeTensorSlice,
    _transform_glm_kv_b,
    _validate_runtime_fused_pairs,
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
        ("mlp.experts.gate_up_proj", "ffn_gate_up_exps.weight"),
        ("mlp.gate.weight", "ffn_gate_inp.weight"),
        ("mlp.shared_expert.down_proj.weight", "ffn_down_shexp.weight"),
        ("mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"),
        ("mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight"),
        ("mlp.shared_expert_gate.weight", "ffn_gate_inp_shexp.weight"),
    ],
)
def test_qwen35_moe_hf_to_gguf_name_mapping(suffix, gguf_suffix):
    assert (
        _hf_to_gguf_name(f"model.language_model.layers.17.{suffix}")
        == f"blk.17.{gguf_suffix}"
    )


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
        ("IQ3_XXS", "NVQ3"),
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


def test_hf_and_gguf_recipe_family_tables_cannot_diverge():
    from mfq.tools import quantize_gguf_to_mfq as gguf_to_mfq

    assert hf_to_mfq._RECIPE_TARGETS == gguf_to_mfq._RECIPE_TARGETS


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
    assert (
        _hf_to_gguf_name(f"model.language_model.layers.29.{suffix}")
        == f"blk.29.{gguf_suffix}"
    )


def test_minicpmo45_hf_to_gguf_name_mapping():
    assert _hf_to_gguf_name("llm.model.embed_tokens.weight") == "token_embd.weight"
    assert _hf_to_gguf_name("llm.model.norm.weight") == "output_norm.weight"
    assert _hf_to_gguf_name("llm.lm_head.weight") == "output.weight"
    assert (
        _hf_to_gguf_name(
            "llm.model.layers.3.post_attention_layernorm.weight"
        )
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

    assert targets["llm.model.layers.0.self_attn.q_proj.weight"] == "NINT4"
    assert targets["vpm.encoder.layers.0.self_attn.q_proj.weight"] == "NINT4"
    for name in tensors:
        if name not in {
            "llm.model.layers.0.self_attn.q_proj.weight",
            "vpm.encoder.layers.0.self_attn.q_proj.weight",
        }:
            assert targets[name] == "BF16"

    text_plan = build_hf_plan(root, True, None, "F16")
    assert [item.name for item in text_plan] == ["llm.model.layers.0.self_attn.q_proj.weight"]


def test_minicpmo45_llm_recipe_keeps_other_components_at_source_precision(tmp_path):
    root = tmp_path / "minicpmo45-recipe"
    root.mkdir()
    tensors = {
        "llm.model.layers.0.self_attn.q_proj.weight": torch.zeros(
            (8, 8), dtype=torch.bfloat16
        ),
        "vpm.encoder.layers.0.self_attn.q_proj.weight": torch.zeros(
            (8, 8), dtype=torch.bfloat16
        ),
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
        "llm.model.layers.0.self_attn.q_proj.weight": "NINT5",
        "tts.emb_text.weight": "BF16",
        "vpm.encoder.layers.0.self_attn.q_proj.weight": "BF16",
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
    np.testing.assert_array_equal(
        binding.selected(np.asarray([0, 3], dtype=np.int64)), values
    )


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
            tensor_name: torch.linspace(
                -2.0, 2.0, steps=4 * 24, dtype=torch.float32
            ).reshape(4, 24).to(torch.bfloat16)
        },
        root / "model.safetensors",
    )
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}), encoding="utf-8"
    )
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
            tensor_name: "blk.0.ffn_down.weight"
        }
    finally:
        store.close()


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
            ).reshape(8, 24).to(torch.bfloat16)
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
        assert store.records[tensor_name].dtype == "NVQ2"
        assert header.extra["target_counts"] == {"NVQ2": 1}
        assert header.extra["tensor_precision_overrides"] == {
            "blk.0.ffn_down.weight": "NVQ2"
        }
    finally:
        store.close()


def test_hf_convert_trains_and_writes_tensorwise_jsc_vq(tmp_path):
    root = tmp_path / "hf-jsc"
    root.mkdir()
    tensor_name = "model.language_model.layers.0.mlp.down_proj.weight"
    generator = torch.Generator().manual_seed(17)
    save_file(
        {
            tensor_name: torch.randn(
                (16, 24), generator=generator, dtype=torch.float32
            ).to(torch.bfloat16)
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
        assert store.records[tensor_name].dtype == "NVQ2J"
        result = header.extra["nvq_codebooks"][tensor_name]
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
        "model.language_model.norm.weight": torch.tensor(
            [0.75, 1.5], dtype=torch.float32
        ),
        "lm_head.weight": f32_matrix,
        "model.language_model.layers.0.linear_attn.conv1d.weight": torch.tensor(
            [[0.125, -0.25], [0.5, 2.0]], dtype=torch.float32
        ),
        "model.language_model.position_ids": torch.tensor(
            [0, 1], dtype=torch.int64
        ),
    }
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}), encoding="utf-8"
    )
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
        assert header.extra["quant_backend"] == "cpu"
        assert header.extra["target_counts"] == {"BF16": 2, "F32": 2, "I64": 1}
        assert store.records["model.language_model.embed_tokens.weight"].dtype == "BF16"
        assert store.records["lm_head.weight"].dtype == "BF16"
        assert store.records["model.language_model.norm.weight"].dtype == "F32"
        assert (
            store.records[
                "model.language_model.layers.0.linear_attn.conv1d.weight"
            ].dtype
            == "F32"
        )
        assert store.records["model.language_model.position_ids"].dtype == "I64"
        restored = store["model.language_model.embed_tokens.weight"]
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
            (
                source_bits.astype(np.uint64)
                + np.uint64(0x7FFF)
                + ((source_bits >> 16) & 1)
            )
            >> 16
        ).astype(np.uint16)
        np.testing.assert_array_equal(store["lm_head.weight"], expected_bf16)
        np.testing.assert_array_equal(
            store["model.language_model.norm.weight"],
            tensors["model.language_model.norm.weight"].numpy(),
        )
        np.testing.assert_array_equal(
            store["model.language_model.layers.0.linear_attn.conv1d.weight"],
            tensors[
                "model.language_model.layers.0.linear_attn.conv1d.weight"
            ].numpy(),
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
        tensors[base + "gate_proj.weight"] = torch.full(
            (24, 24), float(10 * expert + 1)
        )
        tensors[base + "up_proj.weight"] = torch.full(
            (24, 24), float(10 * expert + 2)
        )
        tensors[base + "down_proj.weight"] = torch.full(
            (24, 24), float(10 * expert + 3)
        )
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    plan = build_hf_plan(root, True, None, "F16")
    by_name = {item.name: item for item in plan}
    assert len(plan) == 6
    assert not any(name.startswith("model.layers.2.") for name in by_name)
    embed = by_name["model.layers.0.self_attn.embed_q"]
    unembed = by_name["model.layers.0.self_attn.unembed_out"]
    assert embed.expert_shape == (2, 24, 8)
    assert unembed.expert_shape == (2, 12, 24)

    source = tensors["model.layers.0.self_attn.kv_b_proj.weight"]
    embed_weight = _transform_glm_kv_b(source, embed)
    unembed_weight = _transform_glm_kv_b(source, unembed)
    source_heads = source.reshape(2, 20, 24)
    assert torch.equal(embed_weight, source_heads[:, :8].transpose(1, 2))
    assert torch.equal(unembed_weight, source_heads[:, 8:])

    gate_up = by_name["model.layers.1.mlp.experts.gate_up_proj"]
    stream = _GlmExpertRowSource(
        root,
        gate_up.expert_shape,
        gate_up.expert_source_names,
        gate_up.expert_source_shards,
    )
    try:
        expert_one = stream[48:96]
    finally:
        stream.close()
    assert expert_one.shape == (48, 24)
    assert torch.equal(expert_one[:24], tensors[
        "model.layers.1.mlp.experts.1.gate_proj.weight"
    ])
    assert torch.equal(expert_one[24:], tensors[
        "model.layers.1.mlp.experts.1.up_proj.weight"
    ])

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
        assert {
            name for name in store.records if not is_asset_record(name)
        } == set(by_name)
        assert all(
            record.dtype == "NINTM"
            for record in store.records.values()
            if not is_asset_record(record.name)
        )
        assert store["model.layers.0.self_attn.embed_q"].shape == (2, 24, 8)
        assert store["model.layers.1.mlp.experts.gate_up_proj"].shape == (
            3,
            48,
            24,
        )
    finally:
        store.close()

    split_output = tmp_path / "tiny-glm-split.mfq"
    split_args = argparse.Namespace(**vars(args))
    split_args.output = str(split_output)
    split_args.split_max_size = 0
    split_args.split_max_tensors = 2
    convert(split_args)
    last_shard = format_shard_path(split_output, 3, 3)
    _header, split_store = load_mmap(last_shard)
    try:
        assert len(split_store.paths) == 3
        assert {
            name for name in split_store.records if not is_asset_record(name)
        } == set(by_name)
        assert split_store[
            "model.layers.1.mlp.experts.gate_up_proj"
        ].shape == (3, 48, 24)
    finally:
        split_store.close()


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
