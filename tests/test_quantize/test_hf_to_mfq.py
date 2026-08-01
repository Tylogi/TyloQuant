from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import mfq.tools.quantize_hf_to_mfq as hf_to_mfq
from mfq.calibration.artifact import ExpertPrecision
from mfq.formats.nint import NintSpec
from mfq.formats.io import load_mmap
from mfq.formats.shards import format_shard_path
from mfq.formats.assets import is_asset_record
from mfq.quantize.imatrix import ImportanceEntry, ImportanceMatrix
from mfq.tools.quantize_hf_to_mfq import (
    TensorPlan,
    _bind_hf_imatrix,
    _dtype_for_recipe_type,
    _GlmExpertRowSource,
    _hf_to_gguf_name,
    _plan as build_hf_plan,
    _RawSafeTensorSlice,
    _transform_glm_kv_b,
    _validate_runtime_fused_pairs,
    convert,
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


def test_recipe_dense_types_preserve_f32_and_store_bf16_as_f16():
    assert _dtype_for_recipe_type("F32", "F32") == "F32"
    assert _dtype_for_recipe_type("F32", "F16") == "F16"
    assert _dtype_for_recipe_type("F16", "F32") == "F16"
    assert _dtype_for_recipe_type("BF16", "F32") == "F16"


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
            ).reshape(4, 24)
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
