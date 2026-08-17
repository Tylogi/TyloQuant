from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mfq.tools.quantize_gguf_to_mfq as gguf_to_mfq
from mfq.calibration.artifact import (
    CalibrationScheme,
    ExpertPrecision,
    ExpertSelection,
    ExpertTensorSelection,
)
from mfq.calibration.nint_profiles import nint_storage_bits
from mfq.formats.nint import NintSpec
from mfq.formats.npq0_l import NPQ0_L, unpack_npq0_l
from mfq.formats.nvq import (
    E8_256,
    NVQ2_E8,
    NVQ3_D4,
    NVQ3_D4_512,
    NvqJscTensor,
    unpack_nvq,
)
from mfq.formats.nvq1_l import unpack_nvq1_l
from mfq.quantize.imatrix import ImportanceEntry, ImportanceMatrix
from mfq.quantize.npq0_l import Npq0LConfig, Npq0LTables
from mfq.quantize.nvq1_l_quant import dequantize as dequantize_nvq1_l
from mfq.quantize.nvq_jsc import (
    NvqJscConfig,
    dequantize_nvq_jsc,
    initial_jsc_tables,
    quantize_nvq_jsc_fixed,
)
from mfq.quantize.nvq_quant import (
    dequantize as dequantize_nvq,
)
from mfq.quantize.nvq_quant import (
    quantize as quantize_nvq,
)
from mfq.quantize.nvq_tensor_codebook import TensorCodebookTrainingConfig
from mfq.tools.quantize_gguf_to_mfq import (
    GgufRowSource,
    GgufTensorPlan,
    _apply_expert_scheme,
    _apply_iq2_s_to_nint2_mapping,
    _apply_npq0_l_mapping,
    _apply_q8_to_nint8_zero_mapping,
    _apply_nvq3_jsc_mapping,
    _apply_nvq3_to_nint3_mapping,
    _apply_tensor_precision_overrides,
    _bind_imatrix,
    _build_plan,
    _canonical_artifact_signature,
    _estimate_blob_bytes,
    _existing_codebook_artifact_path,
    _hf_output_name_map,
    _load_tensor_precision_overrides,
    _load_gguf,
    _sample_codebook_rows,
    _target_dtype,
    _trim_windows_working_set,
    _train_or_load_jsc_tables,
    _train_or_load_npq0_l_tables,
    _train_or_load_tensor_codebook,
    _write_nvq_blob,
)
from mfq.tools.quantize_hf_to_mfq import _nint_moe_blob_nbytes


def _tensor(name, values, qtype, quantize):
    data = quantize(np.asarray(values, dtype=np.float32), qtype)
    return SimpleNamespace(
        name=name,
        tensor_type=qtype,
        shape=np.asarray(tuple(reversed(values.shape)), dtype=np.uint32),
        data=data,
    )


def test_windows_working_set_trim_uses_the_current_process(monkeypatch):
    calls = []

    def get_current_process():
        return 123

    def empty_working_set(process):
        calls.append(process)
        return 1

    fake_windll = SimpleNamespace(
        kernel32=SimpleNamespace(GetCurrentProcess=get_current_process),
        psapi=SimpleNamespace(EmptyWorkingSet=empty_working_set),
    )
    monkeypatch.setattr(gguf_to_mfq.sys, "platform", "win32")
    # ``ctypes.windll`` only exists on Windows, while this test deliberately
    # simulates Windows from every supported development host.
    monkeypatch.setattr(
        gguf_to_mfq.ctypes, "windll", fake_windll, raising=False
    )

    assert _trim_windows_working_set() is True
    assert calls == [123]


def test_working_set_trim_is_disabled_off_windows(monkeypatch):
    monkeypatch.setattr(gguf_to_mfq.sys, "platform", "linux")

    assert _trim_windows_working_set() is False


def test_hf_output_name_map_drops_only_rope_freqs():
    mapping = _hf_output_name_map(
        [
            "output_norm.weight",
            "blk.0.attn_q.weight",
            "rope_freqs.weight",
        ],
        [
            "model.language_model.norm.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
        ],
    )

    assert mapping == {
        "output_norm.weight": "model.language_model.norm.weight",
        "blk.0.attn_q.weight": (
            "model.language_model.layers.0.self_attn.q_proj.weight"
        ),
    }


def test_hf_output_name_map_rejects_unmapped_model_tensors():
    with pytest.raises(ValueError, match="unexpected unmapped"):
        _hf_output_name_map(
            ["blk.0.unknown.weight"],
            ["model.language_model.norm.weight"],
        )


def test_hf_output_name_map_requires_exact_template_coverage():
    with pytest.raises(ValueError, match="exactly cover"):
        _hf_output_name_map(
            ["output_norm.weight", "rope_freqs.weight"],
            [
                "model.language_model.norm.weight",
                "model.language_model.embed_tokens.weight",
            ],
        )


def test_dry_run_may_validate_an_existing_output_path():
    source = inspect.getsource(gguf_to_mfq.convert)

    assert "matching_shard_paths(output)" in source
    assert "not args.overwrite and not args.dry_run" in source


def test_iq_recipe_maps_to_supported_nint_profiles():
    assert _target_dtype("IQ1_M") == "NVQ1-L"
    assert _target_dtype("IQ2_S") == "NVQ2J-XL"
    assert _target_dtype("IQ2_XS") == "NVQ2J-L"
    assert _target_dtype("IQ2_XXS") == "NVQ2J"
    assert _target_dtype("IQ3_S") == "NVQ3J-L"
    assert _target_dtype("Q2_K") == "NINT2"
    assert _target_dtype("IQ3_XXS") == "NVQ3"
    assert _target_dtype("Q3_K") == "NINT3"
    for qtype in ("IQ4_NL", "IQ4_XS", "Q4_K"):
        assert _target_dtype(qtype) == "NINT4"
    for qtype in ("Q5_0", "Q5_1", "Q5_K"):
        assert _target_dtype(qtype) == "NINT5"
    assert _target_dtype("Q6_K") == "NINT6"
    assert _target_dtype("Q8_0") == "NINT8"
    assert _target_dtype("F32") == "F32"
    assert _target_dtype("BF16") == "BF16"


def test_imatrix_binds_nint3_to_nint6_but_not_nint8(tmp_path):
    plans = [
        SimpleNamespace(
            name=f"blk.0.test{bits}.weight",
            source_name=f"blk.0.test{bits}.weight",
            target_dtype=f"NINT{bits}",
            storage_shape=(3, 4),
            original_shape=(3, 4),
            expert_precisions=None,
        )
        for bits in (3, 4, 5, 6, 8)
    ]
    imatrix = ImportanceMatrix(
        path=tmp_path / "imatrix.gguf",
        entries={
            item.name: ImportanceEntry(
                values=np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
                counts=np.asarray([8], dtype=np.int64),
            )
            for item in plans
        },
        datasets=(),
        chunk_count=1,
        chunk_size=4,
        legacy=False,
    )

    bindings = _bind_imatrix(imatrix, plans)

    assert set(bindings) == {item.name for item in plans[:4]}
    np.testing.assert_array_equal(
        bindings[plans[0].name].rows(0, 2),
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )


def test_imatrix_binds_mixed_nint_experts(tmp_path):
    item = SimpleNamespace(
        name="blk.0.ffn_up_exps.weight",
        source_name="blk.0.ffn_up_exps.weight",
        target_dtype="NINTM",
        storage_shape=(6, 4),
        original_shape=(2, 3, 4),
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
            item.name: ImportanceEntry(
                values=values,
                counts=np.asarray([8, 8], dtype=np.int64),
            )
        },
        datasets=(),
        chunk_count=1,
        chunk_size=4,
        legacy=False,
    )

    binding = _bind_imatrix(imatrix, [item])[item.name]

    np.testing.assert_array_equal(binding.rows(2, 5), values[[0, 1, 1]])
    np.testing.assert_array_equal(binding.selected(np.asarray([0, 3])), values)


def test_npq0_l_mode_only_replaces_nvq1_l_recipe_tensors():
    plan = [
        GgufTensorPlan(
            name=f"w{index}",
            source_name=f"w{index}",
            source_shape=(8, 24),
            original_shape=(8, 24),
            storage_shape=(8, 24),
            source_type="BF16",
            recipe_type=recipe,
            target_dtype=dtype,
        )
        for index, (recipe, dtype) in enumerate(
            (("IQ1_M", "NVQ1-L"), ("IQ2_XXS", "NVQ2J"), ("Q4_K", "NINT4"))
        )
    ]
    mapped = _apply_npq0_l_mapping(plan, True)
    assert [item.target_dtype for item in mapped] == ["NPQ0-L", "NVQ2J", "NINT4"]
    assert _apply_npq0_l_mapping(plan, False) is plan


def test_nvq3_jsc_mode_only_replaces_nvq3_recipe_tensors():
    plan = [
        GgufTensorPlan(
            name=f"w{index}",
            source_name=f"w{index}",
            source_shape=(8, 24),
            original_shape=(8, 24),
            storage_shape=(8, 24),
            source_type="BF16",
            recipe_type=recipe,
            target_dtype=dtype,
        )
        for index, (recipe, dtype) in enumerate(
            (("IQ3_XXS", "NVQ3"), ("IQ2_XXS", "NVQ2J"), ("Q4_K", "NINT4"))
        )
    ]
    mapped = _apply_nvq3_jsc_mapping(plan, True)
    assert [item.target_dtype for item in mapped] == ["NVQ3J", "NVQ2J", "NINT4"]
    mapped_512 = _apply_nvq3_jsc_mapping(plan, True, target_dtype="NVQ3J-512")
    assert [item.target_dtype for item in mapped_512] == [
        "NVQ3J-512",
        "NVQ2J",
        "NINT4",
    ]
    assert _apply_nvq3_jsc_mapping(plan, False) is plan


def test_nvq3_to_nint3_mode_only_replaces_nvq3_recipe_tensors():
    plan = [
        GgufTensorPlan(
            name=f"w{index}",
            source_name=f"w{index}",
            source_shape=(8, 24),
            original_shape=(8, 24),
            storage_shape=(8, 24),
            source_type="BF16",
            recipe_type=recipe,
            target_dtype=dtype,
        )
        for index, (recipe, dtype) in enumerate(
            (
                ("IQ3_S", "NVQ3"),
                ("Q3_K", "NINT3"),
                ("Q4_K", "NINT4"),
            )
        )
    ]
    mapped = _apply_nvq3_to_nint3_mapping(plan, True)
    assert [item.target_dtype for item in mapped] == [
        "NINT3",
        "NINT3",
        "NINT4",
    ]
    assert _apply_nvq3_to_nint3_mapping(plan, False) is plan


def test_iq2_s_to_nint2_mode_does_not_replace_iq2_xxs():
    plan = [
        GgufTensorPlan(
            name=f"w{index}",
            source_name=f"w{index}",
            source_shape=(8, 24),
            original_shape=(8, 24),
            storage_shape=(8, 24),
            source_type="BF16",
            recipe_type=recipe,
            target_dtype=dtype,
        )
        for index, (recipe, dtype) in enumerate(
            (
                ("IQ2_S", "NVQ2J"),
                ("IQ2_XXS", "NVQ2J"),
                ("Q2_K", "NINT2"),
            )
        )
    ]
    mapped = _apply_iq2_s_to_nint2_mapping(plan, True)
    assert [item.target_dtype for item in mapped] == [
        "NINT2",
        "NVQ2J",
        "NINT2",
    ]
    assert _apply_iq2_s_to_nint2_mapping(plan, False) is plan


def test_q8_mode_directs_only_q8_recipe_tensors_to_nint8_zero():
    plan = [
        GgufTensorPlan(
            name=f"w{index}",
            source_name=f"w{index}",
            source_shape=(8, 32),
            original_shape=(8, 32),
            storage_shape=(8, 32),
            source_type="BF16",
            recipe_type=recipe,
            target_dtype=dtype,
        )
        for index, (recipe, dtype) in enumerate(
            (("Q8_0", "NINT8"), ("Q6_K", "NINT6"), ("F16", "F16"))
        )
    ]
    mapped = _apply_q8_to_nint8_zero_mapping(plan, True)
    assert [item.target_dtype for item in mapped] == [
        "NINT8-0",
        "NINT6",
        "F16",
    ]
    assert _apply_q8_to_nint8_zero_mapping(plan, False) is plan


def test_tensor_precision_overrides_apply_exact_names_after_recipe_mapping():
    plan = [
        GgufTensorPlan(
            name=f"w{index}",
            source_name=f"w{index}",
            source_shape=(2, 3, 24),
            original_shape=(2, 3, 24),
            storage_shape=(6, 24),
            source_type="BF16",
            recipe_type=recipe,
            target_dtype=dtype,
        )
        for index, (recipe, dtype) in enumerate(
            (("IQ2_S", "NVQ2J"), ("IQ3_S", "NVQ3J"))
        )
    ]

    mapped = _apply_tensor_precision_overrides(
        plan,
        {"w0": "NINT3"},
    )

    assert [item.target_dtype for item in mapped] == ["NINT3", "NVQ3J"]
    assert mapped[0].storage_shape == (6, 24)
    assert mapped[1] is plan[1]


def test_tensor_precision_override_loader_validates_names_and_dtypes(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(
        '{"format":"mfq.tensor-precision-overrides.v1",'
        '"overrides":{"blk.0.ffn_up.weight":"nint4"}}',
        encoding="utf-8",
    )
    assert _load_tensor_precision_overrides(path) == {
        "blk.0.ffn_up.weight": "NINT4"
    }

    path.write_text('{"overrides":{"w":"BAD"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported tensor precision override"):
        _load_tensor_precision_overrides(path)


def test_tensor_precision_overrides_reject_missing_tensor():
    item = GgufTensorPlan(
        name="w",
        source_name="w",
        source_shape=(8, 24),
        original_shape=(8, 24),
        storage_shape=(8, 24),
        source_type="BF16",
        recipe_type="IQ2_S",
        target_dtype="NVQ2J",
    )
    with pytest.raises(ValueError, match="absent from the GGUF recipe"):
        _apply_tensor_precision_overrides([item], {"missing": "NINT3"})


def test_nvq3j512_size_estimate_includes_9bit_indices_and_512_entry_tables():
    shape = (4096, 5120)
    common = dict(
        name="w",
        source_name="w",
        source_shape=shape,
        original_shape=shape,
        storage_shape=shape,
        source_type="BF16",
        recipe_type="IQ3_S",
    )
    old = GgufTensorPlan(target_dtype="NVQ3J", **common)
    new = GgufTensorPlan(target_dtype="NVQ3J-512", **common)
    expected_delta = (
        NVQ3_D4_512.payload_nbytes(*shape)
        - NVQ3_D4.payload_nbytes(*shape)
        + 2 * (512 - 256) * NVQ3_D4.vector_size
    )
    assert (
        _estimate_blob_bytes(new, jsc_banks=2)
        - _estimate_blob_bytes(old, jsc_banks=2)
        == expected_delta
    )


def test_legacy_niq_tensor_artifact_path_and_signature_are_reused(tmp_path):
    canonical_path = tmp_path / "0123456789abcdef-nvq2j.json"
    legacy_path = tmp_path / "0123456789abcdef-niq2j.json"
    legacy_path.write_text("{}", encoding="utf-8")
    assert _existing_codebook_artifact_path(canonical_path) == legacy_path

    legacy = {
        "target_dtype": "NIQ2J",
        "config": {
            "niq1_anchor_multipliers": [0.75],
            "niq1_refine_steps": 2,
            "niq_native_assignment": True,
            "niq1_native_assignment": True,
        },
    }
    assert _canonical_artifact_signature(legacy) == {
        "target_dtype": "NVQ2J",
        "config": {
            "nvq1_l_anchor_multipliers": [0.75],
            "nvq1_l_refine_steps": 2,
            "nvq_native_assignment": True,
            "nvq1_l_native_assignment": True,
        },
    }


def test_merged_expert_gate_up_is_split_per_expert():
    _GGUFReader, dequantize = _load_gguf()
    from gguf import GGMLQuantizationType  # type: ignore
    from gguf.quants import quantize  # type: ignore

    experts, out, neuron_len = 3, 2, 8
    values = np.arange(experts * out * 2 * neuron_len, dtype=np.float32).reshape(
        experts, out * 2, neuron_len
    )
    source_tensor = _tensor(
        "blk.0.ffn_gate_up_exps.weight", values, GGMLQuantizationType.BF16, quantize
    )
    gate_recipe = SimpleNamespace(
        name="blk.0.ffn_gate_exps.weight",
        tensor_type=SimpleNamespace(name="IQ1_M"),
        shape=np.asarray((neuron_len, out, experts), dtype=np.uint32),
    )
    up_recipe = SimpleNamespace(
        name="blk.0.ffn_up_exps.weight",
        tensor_type=SimpleNamespace(name="IQ2_XXS"),
        shape=np.asarray((neuron_len, out, experts), dtype=np.uint32),
    )
    source_reader = SimpleNamespace(tensors=[source_tensor])
    recipe_reader = SimpleNamespace(tensors=[gate_recipe, up_recipe])

    gate_plan, up_plan = _build_plan(source_reader, recipe_reader)
    assert gate_plan.storage_shape == (experts * out, neuron_len)
    assert up_plan.storage_shape == (experts * out, neuron_len)
    assert gate_plan.split == "gate"
    assert up_plan.split == "up"
    assert gate_plan.target_dtype == "NVQ1-L"
    assert up_plan.target_dtype == "NVQ2J"

    gate_source = GgufRowSource(source_tensor, gate_plan, dequantize)
    gate = gate_source[: experts * out].numpy()
    up = GgufRowSource(source_tensor, up_plan, dequantize)[: experts * out].numpy()
    expected = dequantize(source_tensor.data, source_tensor.tensor_type)
    np.testing.assert_array_equal(gate, expected[:, :out, :].reshape(-1, neuron_len))
    np.testing.assert_array_equal(up, expected[:, out:, :].reshape(-1, neuron_len))
    np.testing.assert_array_equal(
        gate_source.read_rows(1, 5, device="cpu").numpy(),
        gate[1:5],
    )
    np.testing.assert_array_equal(
        gate_source.read_rows(np.asarray([5, 1], dtype=np.int64)).numpy(),
        gate[[5, 1]],
    )


def test_gguf_plan_excludes_only_recipe_blocks_beyond_source_main_layers():
    qtype = SimpleNamespace(name="F32")
    source_reader = SimpleNamespace(
        tensors=[
            SimpleNamespace(
                name="blk.0.attn_q.weight",
                tensor_type=qtype,
                shape=np.asarray((8, 8), dtype=np.uint32),
            )
        ]
    )
    recipe_reader = SimpleNamespace(
        tensors=[
            SimpleNamespace(
                name="blk.0.attn_q.weight",
                tensor_type=qtype,
                shape=np.asarray((8, 8), dtype=np.uint32),
            ),
            SimpleNamespace(
                name="blk.1.nextn.eh_proj.weight",
                tensor_type=qtype,
                shape=np.asarray((8, 8), dtype=np.uint32),
            ),
        ]
    )
    excluded = []
    plan = _build_plan(
        source_reader,
        recipe_reader,
        exclude_mtp=True,
        excluded_recipe_tensors=excluded,
    )
    assert [item.name for item in plan] == ["blk.0.attn_q.weight"]
    assert excluded == ["blk.1.nextn.eh_proj.weight"]
    with pytest.raises(KeyError, match="recipe tensor has no BF16 source"):
        _build_plan(source_reader, recipe_reader)


def test_gguf_plan_excludes_mtp_present_in_both_source_and_recipe():
    qtype = SimpleNamespace(name="F32")

    def tensor(name):
        return SimpleNamespace(
            name=name,
            tensor_type=qtype,
            shape=np.asarray((8, 8), dtype=np.uint32),
        )

    source_reader = SimpleNamespace(
        fields={
            "general.architecture": SimpleNamespace(
                contents=lambda: "qwen35"
            ),
            "qwen35.block_count": SimpleNamespace(contents=lambda: 2),
            "qwen35.nextn_predict_layers": SimpleNamespace(
                contents=lambda: 1
            ),
        },
        tensors=[
            tensor("blk.0.attn_q.weight"),
            tensor("blk.1.attn_q.weight"),
            tensor("blk.1.nextn.eh_proj.weight"),
        ],
    )
    recipe_reader = SimpleNamespace(
        tensors=[
            tensor("blk.0.attn_q.weight"),
            tensor("blk.1.attn_q.weight"),
            tensor("blk.1.nextn.eh_proj.weight"),
        ]
    )
    excluded = []

    plan = _build_plan(
        source_reader,
        recipe_reader,
        exclude_mtp=True,
        excluded_recipe_tensors=excluded,
    )

    assert [item.name for item in plan] == [
        "blk.0.attn_q.weight",
    ]
    assert excluded == [
        "blk.1.attn_q.weight",
        "blk.1.nextn.eh_proj.weight",
    ]


def test_gguf_plan_accepts_expertwise_precision_scheme():
    shape = (4, 3, 48)
    name = "blk.0.ffn_down_exps.weight"
    specs = (
        NintSpec(4, 24, 6),
        NintSpec(6, 24, 7),
        NintSpec(4, 24, 6),
        NintSpec(8, 48, 7),
    )
    selection = ExpertTensorSelection(
        name=name,
        group="blk.0.expert_down",
        n_experts=shape[0],
        rows_per_expert=shape[1],
        columns=shape[2],
        selections=tuple(
            ExpertSelection(
                expert_id=expert,
                spec=profile,
                storage_bits=nint_storage_bits(shape[1], shape[2], profile),
                train_loss=1.0,
                validation_loss=1.0,
            )
            for expert, profile in enumerate(specs)
        ),
    )
    scheme = CalibrationScheme(
        path=None,
        target_profile="EXPERT_WISE",
        target_storage_bits=selection.storage_bits,
        selections={},
        metadata={},
        candidate_table={},
        expert_selections={name: selection},
    )
    base = GgufTensorPlan(
        name=name,
        source_name=name,
        source_shape=shape,
        original_shape=shape,
        storage_shape=(shape[0] * shape[1], shape[2]),
        source_type="BF16",
        recipe_type="Q4_K",
        target_dtype="NINT4",
    )
    applied = _apply_expert_scheme([base], scheme)[0]
    assert applied.target_dtype == "NINTM"
    assert applied.expert_specs == specs
    assert _estimate_blob_bytes(applied) == _nint_moe_blob_nbytes(shape, specs)


@pytest.mark.parametrize(
    "target_dtype",
    [
        "NVQ1-L",
        "NVQ2",
        "NVQ2J",
        "NVQ2J-L",
        "NVQ2J-XL",
        "NVQ3",
        "NVQ3J",
        "NVQ3J-512",
        "NVQ3J-L",
    ],
)
def test_streaming_nvq_blob_roundtrip(tmp_path, target_dtype):
    rng = np.random.default_rng(20260716)
    weight = torch.from_numpy(rng.normal(0, 0.05, size=(8, 48)).astype(np.float32))
    blob = tmp_path / f"{target_dtype.lower()}.blob"

    result = _write_nvq_blob(
        weight,
        tuple(weight.shape),
        target_dtype,
        blob,
        row_chunk=4,
        quant_backend="cpu",
        device="cpu",
        group_chunk=16,
        nvq1_l_candidates=0,
        nvq1_l_anchor_multipliers=(0.75,),
        nvq1_l_refine_steps=2,
        jsc_tables=(
            initial_jsc_tables(
                NvqJscConfig(
                    banks=2,
                    learned_scale_lut=False,
                    spec=gguf_to_mfq._NVQ_SPECS[target_dtype],
                )
            )
            if target_dtype in gguf_to_mfq._JSC_DTYPES
            else None
        ),
        search_steps=3,
    )

    payload = blob.read_bytes()
    restored = unpack_nvq1_l(payload) if target_dtype == "NVQ1-L" else unpack_nvq(payload)
    assert result.nbytes == len(payload)
    assert result.gain_calibration is None
    assert restored.shape == tuple(weight.shape)


def test_streaming_jsc_none_calibration_keeps_bound_imatrix(
    tmp_path,
    monkeypatch,
):
    rng = np.random.default_rng(20260726)
    weight = torch.from_numpy(rng.normal(0, 0.05, size=(8, 48)).astype(np.float32))
    importance = np.geomspace(0.01, 100.0, 48, dtype=np.float32)
    tables = initial_jsc_tables(
        NvqJscConfig(
            banks=2,
            learned_scale_lut=False,
            spec=NVQ3_D4_512,
        )
    )
    observed: list[np.ndarray | None] = []
    original = gguf_to_mfq._quantize_nvq_chunk

    def capture_importance(*args, **kwargs):
        observed.append(args[8])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        gguf_to_mfq,
        "_quantize_nvq_chunk",
        capture_importance,
    )
    _write_nvq_blob(
        weight,
        tuple(weight.shape),
        "NVQ3J-512",
        tmp_path / "nvq3j512-imatrix.blob",
        row_chunk=4,
        quant_backend="cpu",
        device="cpu",
        group_chunk=16,
        nvq1_l_candidates=0,
        nvq1_l_anchor_multipliers=(0.75,),
        nvq1_l_refine_steps=2,
        importance_rows=lambda _start, _end: importance,
        jsc_tables=tables,
        search_steps=3,
        calibration_mode="none",
    )
    assert len(observed) == 2
    for value in observed:
        np.testing.assert_array_equal(value, importance)


def test_streaming_npq0_l_blob_roundtrip(tmp_path):
    rng = np.random.default_rng(20260722)
    weight = torch.from_numpy(rng.normal(0, 0.05, size=(8, 48)).astype(np.float32))
    tables = Npq0LTables(
        scale_lut=np.linspace(0.125, 1.0, 8, dtype=np.float16).astype(np.float32),
        first_codebooks=rng.integers(
            -48, 49, size=(8, 8, 4), dtype=np.int16
        ).astype(np.int8),
        second_codebooks=rng.integers(
            -48, 49, size=(8, 16, 4), dtype=np.int16
        ).astype(np.int8),
    )
    config = Npq0LConfig(
        iterations=0,
        assignment_refine_steps=0,
        fixed_refine_steps=1,
        kmeans_iterations=0,
        group_chunk=16,
    )
    blob = tmp_path / "npq0_l.blob"
    result = _write_nvq_blob(
        weight,
        tuple(weight.shape),
        "NPQ0-L",
        blob,
        row_chunk=4,
        quant_backend="cpu",
        device="cpu",
        group_chunk=16,
        nvq1_l_candidates=0,
        nvq1_l_anchor_multipliers=(0.75,),
        nvq1_l_refine_steps=2,
        npq0_l_tables=tables,
        npq0_l_config=config,
        search_steps=1,
    )

    restored = unpack_npq0_l(blob.read_bytes())
    assert result.nbytes == len(blob.read_bytes())
    assert restored.shape == tuple(weight.shape)
    assert result.nbytes == (
        20 + 8 * len(weight.shape) + 4 + NPQ0_L.payload_nbytes(*weight.shape)
    )
    np.testing.assert_array_equal(restored.first_codebooks, tables.first_codebooks)
    np.testing.assert_array_equal(restored.second_codebooks, tables.second_codebooks)


@pytest.mark.parametrize(
    "target_dtype",
    [
        "NVQ1-L",
        "NVQ2",
        "NVQ2J",
        "NVQ2J-L",
        "NVQ2J-XL",
        "NVQ3",
        "NVQ3J",
        "NVQ3J-512",
        "NVQ3J-L",
    ],
)
def test_streaming_nvq_gain_calibration_preserves_codes_and_reduces_diagonal_loss(
    tmp_path,
    target_dtype,
):
    rng = np.random.default_rng(20260728)
    weight = torch.from_numpy(rng.normal(0, 0.05, size=(17, 48)).astype(np.float32))
    baseline_blob = tmp_path / f"{target_dtype.lower()}-baseline.blob"
    calibrated_blob = tmp_path / f"{target_dtype.lower()}-calibrated.blob"
    importance = rng.lognormal(0.0, 1.0, size=48).astype(np.float32)
    jsc_tables = (
        initial_jsc_tables(
            NvqJscConfig(
                banks=2,
                learned_scale_lut=False,
                spec=gguf_to_mfq._NVQ_SPECS[target_dtype],
            )
        )
        if target_dtype in gguf_to_mfq._JSC_DTYPES
        else None
    )
    kwargs = dict(
        source=weight,
        shape=tuple(weight.shape),
        target_dtype=target_dtype,
        row_chunk=4,
        quant_backend="cpu",
        device="cpu",
        group_chunk=16,
        nvq1_l_candidates=0,
        nvq1_l_anchor_multipliers=(0.75,),
        nvq1_l_refine_steps=2,
        search_steps=1,
        importance_rows=lambda _start, _end: importance,
        jsc_tables=jsc_tables,
    )

    baseline_result = _write_nvq_blob(blob_path=baseline_blob, **kwargs)
    calibrated_result = _write_nvq_blob(
        blob_path=calibrated_blob,
        calibration_mode="gain",
        **kwargs,
    )
    if target_dtype == "NVQ1-L":
        baseline = unpack_nvq1_l(baseline_blob.read_bytes())
        calibrated = unpack_nvq1_l(calibrated_blob.read_bytes())
        baseline_reconstruction = dequantize_nvq1_l(baseline)
        reconstruction = dequantize_nvq1_l(calibrated)
    elif target_dtype in gguf_to_mfq._JSC_DTYPES:
        baseline = unpack_nvq(baseline_blob.read_bytes())
        calibrated = unpack_nvq(calibrated_blob.read_bytes())
        assert isinstance(baseline, NvqJscTensor)
        assert isinstance(calibrated, NvqJscTensor)
        baseline_reconstruction = dequantize_nvq_jsc(baseline)
        reconstruction = dequantize_nvq_jsc(calibrated)
    else:
        baseline = unpack_nvq(baseline_blob.read_bytes())
        calibrated = unpack_nvq(calibrated_blob.read_bytes())
        baseline_reconstruction = dequantize_nvq(baseline)
        reconstruction = dequantize_nvq(calibrated)

    np.testing.assert_array_equal(calibrated.sub_scale, baseline.sub_scale)
    np.testing.assert_array_equal(calibrated.indices, baseline.indices)
    auxiliary = "delta_sign" if target_dtype == "NVQ1-L" else "signs"
    np.testing.assert_array_equal(
        getattr(calibrated, auxiliary),
        getattr(baseline, auxiliary),
    )
    reference = weight.numpy()
    baseline_loss = np.sum(importance * np.square(reference - baseline_reconstruction))
    calibrated_loss = np.sum(importance * np.square(reference - reconstruction))
    assert calibrated_loss <= baseline_loss * (1.0 + 2e-4)
    assert baseline_result.gain_calibration is None
    assert calibrated_result.gain_calibration is not None
    assert calibrated_result.gain_calibration["mode"] == "per_neuron_diagonal_regression"


def test_streaming_nvq2j_group24_uses_imatrix_for_discrete_assignment(tmp_path):
    rng = np.random.default_rng(20260801)
    weight = torch.from_numpy(rng.normal(0, 0.05, size=(12, 48)).astype(np.float32))
    importance = np.geomspace(0.01, 100.0, 48, dtype=np.float32)
    tables = initial_jsc_tables(NvqJscConfig(banks=2))
    blob = tmp_path / "nvq2j-group24.blob"

    result = _write_nvq_blob(
        source=weight,
        shape=tuple(weight.shape),
        target_dtype="NVQ2J",
        blob_path=blob,
        row_chunk=4,
        quant_backend="cpu",
        device="cpu",
        group_chunk=16,
        nvq1_l_candidates=0,
        nvq1_l_anchor_multipliers=(0.75,),
        nvq1_l_refine_steps=2,
        importance_rows=lambda _start, _end: importance,
        search_steps=3,
        jsc_tables=tables,
        jsc_assignment_refine_steps=2,
        calibration_mode="group24",
    )
    restored = unpack_nvq(blob.read_bytes())
    direct = quantize_nvq_jsc_fixed(
        weight,
        tables,
        importance=importance,
        assignment_refine_steps=2,
        search_steps=3,
        group_chunk=16,
        device="cpu",
    )

    assert isinstance(restored, NvqJscTensor)
    np.testing.assert_array_equal(restored.state, direct.state)
    np.testing.assert_array_equal(restored.indices, direct.indices)
    np.testing.assert_array_equal(restored.signs, direct.signs)
    assert result.gain_calibration is not None
    assert result.gain_calibration["assignment"] == "imatrix_group24"


def test_streaming_nvq_blob_embeds_custom_codebook(tmp_path):
    weight = torch.from_numpy(
        np.random.default_rng(20260718).normal(0, 0.05, size=(8, 48)).astype(np.float32)
    )
    custom = np.roll(E8_256, 1, axis=0).copy()
    blob = tmp_path / "nvq2-custom.blob"
    _write_nvq_blob(
        weight,
        tuple(weight.shape),
        "NVQ2",
        blob,
        row_chunk=4,
        quant_backend="cpu",
        device="cpu",
        group_chunk=16,
        nvq1_l_candidates=0,
        nvq1_l_anchor_multipliers=(0.75,),
        nvq1_l_refine_steps=2,
        codebook=custom,
        search_steps=1,
    )
    restored = unpack_nvq(blob.read_bytes())
    np.testing.assert_array_equal(restored.codebook, custom)
    assert restored.payload_nbytes == NVQ2_E8.payload_nbytes(8, 48) + 512


def test_streaming_nvq_blob_uses_row_importance(tmp_path):
    rng = np.random.default_rng(20260727)
    weight = torch.from_numpy(rng.normal(0, 0.05, size=(8, 48)).astype(np.float32))
    importance = rng.lognormal(0.0, 1.0, size=(8, 48)).astype(np.float32)
    blob = tmp_path / "nvq2-imatrix.blob"

    _write_nvq_blob(
        weight,
        tuple(weight.shape),
        "NVQ2",
        blob,
        row_chunk=4,
        quant_backend="cpu",
        device="cpu",
        group_chunk=16,
        nvq1_l_candidates=0,
        nvq1_l_anchor_multipliers=(0.75,),
        nvq1_l_refine_steps=2,
        importance_rows=lambda start, end: importance[start:end],
        search_steps=1,
    )

    restored = unpack_nvq(blob.read_bytes())
    direct = quantize_nvq(
        weight.numpy(),
        NVQ2_E8,
        importance=importance,
        search_steps=1,
        group_chunk=16,
    )
    np.testing.assert_array_equal(restored.neuron_scale, direct.neuron_scale)
    np.testing.assert_array_equal(restored.sub_scale, direct.sub_scale)
    np.testing.assert_array_equal(restored.indices, direct.indices)
    np.testing.assert_array_equal(restored.signs, direct.signs)


def test_codebook_row_sampling_is_deterministic_disjoint_and_expert_balanced():
    item = SimpleNamespace(
        name="blk.0.ffn_gate_exps.weight",
        storage_shape=(8 * 32, 64),
        original_shape=(8, 32, 64),
    )
    train_a, validation_a = _sample_codebook_rows(item, 40, 24, 1234)
    train_b, validation_b = _sample_codebook_rows(item, 40, 24, 1234)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    assert not np.intersect1d(train_a, validation_a).size
    train_counts = np.bincount(train_a // 32, minlength=8)
    validation_counts = np.bincount(validation_a // 32, minlength=8)
    assert train_counts.max() - train_counts.min() <= 1
    assert validation_counts.max() - validation_counts.min() <= 1


def test_codebook_row_sampling_preserves_holdout_for_small_matrix():
    item = SimpleNamespace(
        name="blk.3.attn_k.weight",
        storage_shape=(512, 4096),
        original_shape=(512, 4096),
    )
    train, validation = _sample_codebook_rows(item, 2048, 512, 20260716)
    assert train.size == 410
    assert validation.size == 102
    assert train.size + validation.size == 512
    assert not np.intersect1d(train, validation).size


def test_tensor_codebook_artifact_is_reused(tmp_path):
    weight = np.random.default_rng(20260719).normal(0, 0.05, size=(8, 24)).astype(np.float32)

    class Source:
        def read_rows(self, indices):
            return torch.from_numpy(weight[np.asarray(indices)])

    item = SimpleNamespace(
        name="blk.0.test.weight",
        source_name="blk.0.test.weight",
        target_dtype="NVQ2",
        storage_shape=(8, 24),
        original_shape=(8, 24),
    )
    source_path = tmp_path / "source.gguf"
    recipe_path = tmp_path / "recipe.gguf"
    source_path.write_bytes(b"source")
    recipe_path.write_bytes(b"recipe")
    config = TensorCodebookTrainingConfig(
        iterations=0,
        projection_candidates=4,
        quant_backend="cpu",
        group_chunk=8,
        row_chunk=4,
        search_steps=1,
        initializations=("builtin",),
    )
    first_codebook, first = _train_or_load_tensor_codebook(
        Source(), item, source_path, recipe_path, tmp_path / "artifacts", config, 4, 2, 7
    )
    second_codebook, second = _train_or_load_tensor_codebook(
        Source(), item, source_path, recipe_path, tmp_path / "artifacts", config, 4, 2, 7
    )
    assert not first["loaded"]
    assert second["loaded"]
    assert (first_codebook is None) == (second_codebook is None)


def test_tensor_jsc_artifact_is_weight_only_and_reused(tmp_path):
    weight = np.random.default_rng(20260731).normal(0, 0.05, size=(8, 24)).astype(
        np.float32
    )

    class Source:
        def read_rows(self, indices):
            return torch.from_numpy(weight[np.asarray(indices)])

    item = SimpleNamespace(
        name="blk.0.test.weight",
        source_name="blk.0.test.weight",
        target_dtype="NVQ2J",
        storage_shape=(8, 24),
        original_shape=(8, 24),
    )
    source_path = tmp_path / "source.gguf"
    recipe_path = tmp_path / "recipe.gguf"
    source_path.write_bytes(b"source")
    recipe_path.write_bytes(b"recipe")
    config = NvqJscConfig(
        banks=1,
        iterations=0,
        assignment_refine_steps=0,
        search_steps=1,
        group_chunk=8,
    )
    first_tables, first = _train_or_load_jsc_tables(
        Source(), item, source_path, recipe_path, tmp_path / "jsc", config, 4, 2, 7, "cpu"
    )
    second_tables, second = _train_or_load_jsc_tables(
        Source(), item, source_path, recipe_path, tmp_path / "jsc", config, 4, 2, 7, "cpu"
    )
    assert not first["loaded"]
    assert second["loaded"]
    assert first["objective"] == "unweighted_weight_sse"
    np.testing.assert_array_equal(first_tables.scale_lut, second_tables.scale_lut)
    np.testing.assert_array_equal(first_tables.codebooks, second_tables.codebooks)


def test_tensor_jsc_artifact_uses_bound_imatrix(tmp_path):
    weight = np.random.default_rng(20260802).normal(0, 0.05, size=(8, 24)).astype(
        np.float32
    )

    class Source:
        def read_rows(self, indices):
            return torch.from_numpy(weight[np.asarray(indices)])

    item = SimpleNamespace(
        name="blk.0.test.weight",
        source_name="blk.0.test.weight",
        target_dtype="NVQ3J",
        storage_shape=(8, 24),
        original_shape=(8, 24),
    )
    source_path = tmp_path / "source.gguf"
    recipe_path = tmp_path / "recipe.gguf"
    imatrix_path = tmp_path / "imatrix.gguf"
    source_path.write_bytes(b"source")
    recipe_path.write_bytes(b"recipe")
    imatrix_path.write_bytes(b"imatrix")
    imatrix = ImportanceMatrix(
        path=imatrix_path,
        entries={
            item.name: ImportanceEntry(
                values=np.geomspace(0.1, 10.0, 24, dtype=np.float32)[None, :],
                counts=np.asarray([1024], dtype=np.int64),
            )
        },
        datasets=("calibration.txt",),
        chunk_count=8,
        chunk_size=128,
        legacy=False,
    )
    binding = _bind_imatrix(imatrix, [item])[item.name]
    config = NvqJscConfig(
        spec=NVQ3_D4,
        banks=1,
        iterations=0,
        assignment_refine_steps=0,
        search_steps=1,
        group_chunk=8,
    )

    first_tables, first = _train_or_load_jsc_tables(
        Source(),
        item,
        source_path,
        recipe_path,
        tmp_path / "artifacts",
        config,
        4,
        2,
        7,
        "cpu",
        imatrix,
        binding,
    )
    second_tables, second = _train_or_load_jsc_tables(
        Source(),
        item,
        source_path,
        recipe_path,
        tmp_path / "artifacts",
        config,
        4,
        2,
        7,
        "cpu",
        imatrix,
        binding,
    )

    assert first["objective"] == "imatrix_weighted_sse"
    assert first["imatrix_entry"] == item.name
    assert not first["loaded"]
    assert second["loaded"]
    np.testing.assert_array_equal(first_tables.scale_lut, second_tables.scale_lut)
    np.testing.assert_array_equal(first_tables.codebooks, second_tables.codebooks)


def test_tensor_npq0_l_artifact_is_reused(tmp_path):
    weight = np.random.default_rng(20260722).normal(0, 0.05, size=(8, 24)).astype(
        np.float32
    )

    class Source:
        def read_rows(self, indices):
            return torch.from_numpy(weight[np.asarray(indices)])

    item = SimpleNamespace(
        name="blk.0.test.weight",
        source_name="blk.0.test.weight",
        target_dtype="NPQ0-L",
        storage_shape=(8, 24),
        original_shape=(8, 24),
    )
    source_path = tmp_path / "source.gguf"
    recipe_path = tmp_path / "recipe.gguf"
    source_path.write_bytes(b"source")
    recipe_path.write_bytes(b"recipe")
    config = Npq0LConfig(
        iterations=0,
        assignment_refine_steps=0,
        fixed_refine_steps=0,
        kmeans_iterations=1,
        kmeans_initialization_points=64,
        group_chunk=8,
        seed=7,
    )
    first_tables, first = _train_or_load_npq0_l_tables(
        Source(),
        item,
        source_path,
        recipe_path,
        tmp_path / "artifacts",
        config,
        4,
        2,
        7,
        "cpu",
    )
    second_tables, second = _train_or_load_npq0_l_tables(
        Source(),
        item,
        source_path,
        recipe_path,
        tmp_path / "artifacts",
        config,
        4,
        2,
        7,
        "cpu",
    )
    assert not first["loaded"]
    assert second["loaded"]
    np.testing.assert_array_equal(first_tables.scale_lut, second_tables.scale_lut)
    np.testing.assert_array_equal(
        first_tables.first_codebooks,
        second_tables.first_codebooks,
    )
    np.testing.assert_array_equal(
        first_tables.second_codebooks,
        second_tables.second_codebooks,
    )


def test_tensor_codebook_artifact_uses_bound_imatrix(tmp_path):
    weight = np.random.default_rng(20260725).normal(0, 0.05, size=(8, 24)).astype(
        np.float32
    )

    class Source:
        def read_rows(self, indices):
            return torch.from_numpy(weight[np.asarray(indices)])

    item = SimpleNamespace(
        name="blk.0.test.weight",
        source_name="blk.0.test.weight",
        target_dtype="NVQ2",
        storage_shape=(8, 24),
        original_shape=(8, 24),
    )
    source_path = tmp_path / "source.gguf"
    recipe_path = tmp_path / "recipe.gguf"
    imatrix_path = tmp_path / "imatrix.gguf"
    source_path.write_bytes(b"source")
    recipe_path.write_bytes(b"recipe")
    imatrix_path.write_bytes(b"imatrix")
    imatrix = ImportanceMatrix(
        path=imatrix_path,
        entries={
            item.name: ImportanceEntry(
                values=np.geomspace(0.1, 10.0, 24, dtype=np.float32)[None, :],
                counts=np.asarray([1024], dtype=np.int64),
            )
        },
        datasets=("calibration.txt",),
        chunk_count=8,
        chunk_size=128,
        legacy=False,
    )
    binding = _bind_imatrix(imatrix, [item])[item.name]
    config = TensorCodebookTrainingConfig(
        iterations=0,
        projection_candidates=4,
        quant_backend="cpu",
        group_chunk=8,
        row_chunk=4,
        search_steps=1,
        initializations=("builtin",),
    )

    _codebook, first = _train_or_load_tensor_codebook(
        Source(),
        item,
        source_path,
        recipe_path,
        tmp_path / "artifacts",
        config,
        4,
        2,
        7,
        imatrix,
        binding,
    )
    _codebook, second = _train_or_load_tensor_codebook(
        Source(),
        item,
        source_path,
        recipe_path,
        tmp_path / "artifacts",
        config,
        4,
        2,
        7,
        imatrix,
        binding,
    )

    assert first["objective"] == "imatrix_weighted_sse"
    assert first["imatrix_entry"] == item.name
    assert not first["loaded"]
    assert second["loaded"]
