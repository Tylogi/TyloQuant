from __future__ import annotations

import json

import numpy as np
import torch

from mfq.calibration.nint_profiles import NINT_CALIBRATION_PROFILES, nint_storage_bits
from mfq.calibration.reap_expertwise import (
    ExpertProfileEvaluation,
    allocate_expert_profiles,
    allocate_independent_expert_profiles,
    load_reap_expert_table,
)
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import dequantize
from mfq.quantize.nint_quant_torch import quantize_axis0
from mfq.tools.build_reap_expert_scheme import _quantization_sse_by_row


def test_load_reap_table_validates_axes_and_normalizes_exposure(tmp_path):
    path = tmp_path / "reap.jsonl"
    rows = []
    for layer in range(2):
        for expert in range(3):
            probability = (expert + 1) / 2.0
            rows.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "totalTokens": 100,
                    "expertFrequency": int(probability * 100),
                    "expertProbability": probability,
                    "reap": float((layer + 1) * (expert + 2)),
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    observations = load_reap_expert_table(
        path, expected_layers=2, expected_experts=3, expected_top_k=3
    )
    assert len(observations) == 6
    for layer in range(2):
        assert np.isclose(
            sum(observations[(layer, expert)].normalized_exposure for expert in range(3)),
            1.0,
        )
    assert observations[(0, 2)].exposure == 1.5 * 4.0


def test_load_reap_tensor_state_uses_frequency_times_reap(tmp_path):
    path = tmp_path / "partial.pkl"
    torch.save(
        {
            0: {
                "total_tokens": torch.tensor(10),
                "expert_frequency": torch.tensor([6, 4]),
                "reap": torch.tensor([2.0, 1.0]),
            }
        },
        path,
    )
    observations = load_reap_expert_table(
        path, expected_layers=1, expected_experts=2, expected_top_k=1
    )
    assert np.isclose(observations[(0, 0)].exposure, 1.2)
    assert np.isclose(observations[(0, 1)].exposure, 0.4)
    assert np.isclose(observations[(0, 0)].normalized_exposure, 0.75)
    assert np.isclose(observations[(0, 1)].normalized_exposure, 0.25)


def test_quantization_sse_matches_production_quantizer_reconstruction():
    generator = torch.Generator().manual_seed(20260719)
    weight = torch.randn((7, 53), generator=generator, dtype=torch.float32) * 0.05
    spec = NintSpec(4, 24, 6)
    row_sse, row_signal = _quantization_sse_by_row(weight, spec)
    encoded = quantize_axis0(weight, spec, device="cpu")
    reconstruction = torch.from_numpy(dequantize(encoded))
    expected_sse = ((reconstruction - weight) ** 2).sum(dim=1)
    expected_signal = (weight * weight).sum(dim=1)
    torch.testing.assert_close(row_sse, expected_sse, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(row_signal, expected_signal, rtol=0, atol=0)


def test_global_expert_allocation_preserves_gate_down_coupling_and_budget():
    evaluations = []
    for expert in range(4):
        for profile, spec in NINT_CALIBRATION_PROFILES.items():
            profile_error = {"NINT4": 9.0, "NINT5": 4.0, "NINT6": 2.0, "NINT8": 0.2}[profile]
            scale = 8.0 if expert == 0 else 1.0
            evaluations.append(
                ExpertProfileEvaluation(
                    layer=0,
                    expert=expert,
                    profile=profile,
                    spec=spec,
                    gate_name="model.language_model.layers.0.mlp.experts.gate_up_proj",
                    down_name="model.language_model.layers.0.mlp.experts.down_proj",
                    gate_rows=2,
                    gate_columns=48,
                    down_rows=3,
                    down_columns=48,
                    exposure=scale,
                    normalized_exposure=scale / 11.0,
                    gate_sse=profile_error,
                    gate_signal=100.0,
                    down_sse=profile_error,
                    down_signal=100.0,
                )
            )
    scheme, report = allocate_expert_profiles(evaluations, target_profile="NINT5")
    gate = scheme.require_expert(
        "model.language_model.layers.0.mlp.experts.gate_up_proj"
    )
    down = scheme.require_expert("model.language_model.layers.0.mlp.experts.down_proj")
    assert gate.specs == down.specs
    assert scheme.storage_bits <= scheme.target_storage_bits
    assert report["groups"] == 4
    assert sum(report["selected_counts"].values()) == 4


def test_global_expert_allocation_accepts_external_budget():
    evaluations = []
    for expert in range(3):
        for profile, spec in NINT_CALIBRATION_PROFILES.items():
            profile_error = {
                "NINT4": 9.0,
                "NINT5": 4.0,
                "NINT6": 2.0,
                "NINT8": 0.2,
            }[profile]
            evaluations.append(
                ExpertProfileEvaluation(
                    layer=0,
                    expert=expert,
                    profile=profile,
                    spec=spec,
                    gate_name="model.language_model.layers.0.mlp.experts.gate_up_proj",
                    down_name="model.language_model.layers.0.mlp.experts.down_proj",
                    gate_rows=2,
                    gate_columns=48,
                    down_rows=3,
                    down_columns=48,
                    exposure=float(expert + 1),
                    normalized_exposure=float(expert + 1) / 6.0,
                    gate_sse=profile_error,
                    gate_signal=100.0,
                    down_sse=profile_error,
                    down_signal=100.0,
                )
            )
    nint4 = NINT_CALIBRATION_PROFILES["NINT4"]
    nint5 = NINT_CALIBRATION_PROFILES["NINT5"]
    minimum = 3 * (
        nint_storage_bits(2, 48, nint4)
        + nint_storage_bits(3, 48, nint4)
    )
    one_upgrade = (
        nint_storage_bits(2, 48, nint5)
        + nint_storage_bits(3, 48, nint5)
        - nint_storage_bits(2, 48, nint4)
        - nint_storage_bits(3, 48, nint4)
    )
    target = minimum + one_upgrade
    scheme, report = allocate_expert_profiles(
        evaluations,
        target_storage_bits=target,
        target_label="MATCHED_SIZE",
    )
    gate = scheme.require_expert(
        "model.language_model.layers.0.mlp.experts.gate_up_proj"
    )
    down = scheme.require_expert(
        "model.language_model.layers.0.mlp.experts.down_proj"
    )
    assert gate.specs == down.specs
    assert scheme.storage_bits <= target
    assert report["target_label"] == "MATCHED_SIZE"
    assert report["target_storage_bits"] == target
    assert not report["baseline_budget_feasible"]
    assert report["relative_surrogate_reduction"] is None


def test_independent_expert_allocation_can_assign_gate_and_down_differently():
    evaluations = []
    for expert in range(3):
        for profile, spec in NINT_CALIBRATION_PROFILES.items():
            rank = {"NINT4": 8.0, "NINT5": 3.0, "NINT6": 1.0, "NINT8": 0.1}[profile]
            evaluations.append(
                ExpertProfileEvaluation(
                    layer=0,
                    expert=expert,
                    profile=profile,
                    spec=spec,
                    gate_name="model.language_model.layers.0.mlp.experts.gate_up_proj",
                    down_name="model.language_model.layers.0.mlp.experts.down_proj",
                    gate_rows=2,
                    gate_columns=48,
                    down_rows=3,
                    down_columns=48,
                    exposure=10.0 if expert == 0 else 1.0,
                    normalized_exposure=(10.0 if expert == 0 else 1.0) / 12.0,
                    gate_sse=rank * (20.0 if expert == 0 else 1.0),
                    gate_signal=100.0,
                    down_sse=rank * 0.01,
                    down_signal=100.0,
                )
            )
    nint4 = NINT_CALIBRATION_PROFILES["NINT4"]
    nint5 = NINT_CALIBRATION_PROFILES["NINT5"]
    minimum = 3 * (
        evaluations[0].gate_storage_bits + evaluations[0].down_storage_bits
    )
    gate_upgrade = (
        nint_storage_bits(2, 48, nint5) - nint_storage_bits(2, 48, nint4)
    )
    scheme, report = allocate_independent_expert_profiles(
        evaluations,
        target_storage_bits=minimum + gate_upgrade,
    )
    gate = scheme.require_expert(
        "model.language_model.layers.0.mlp.experts.gate_up_proj"
    )
    down = scheme.require_expert("model.language_model.layers.0.mlp.experts.down_proj")
    assert gate.specs != down.specs
    assert gate.specs[0] == nint5
    assert all(spec == nint4 for spec in down.specs)
    assert scheme.storage_bits <= scheme.target_storage_bits
    assert report["groups"] == 6
