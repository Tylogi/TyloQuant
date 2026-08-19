from __future__ import annotations

import json

from mfq.calibration.artifact import load_scheme
from mfq.calibration.evaluator import (
    NINT_EXPERT_PROFILES,
    nint_storage_bits,
)
from mfq.calibration.reap_expertwise import ExpertProfileEvaluation
from mfq.tools.allocate_reap_layer_budget import allocate_reap_layer_budget


def _write_fixture(tmp_path):
    reap_path = tmp_path / "reap.jsonl"
    reap_rows = [
        {
            "layer": 0,
            "expert": 0,
            "totalTokens": 100,
            "expertFrequency": 90,
            "expertProbability": 0.9,
            "reap": 1.0,
        },
        {
            "layer": 0,
            "expert": 1,
            "totalTokens": 100,
            "expertFrequency": 10,
            "expertProbability": 0.1,
            "reap": 1.0,
        },
    ]
    reap_path.write_text(
        "\n".join(json.dumps(row) for row in reap_rows) + "\n",
        encoding="utf-8",
    )

    error = {
        "NINT2": 16.0,
        "NINT3": 4.0,
        "NINT4": 2.0,
        "NINT5": 1.0,
        "NINT6": 0.5,
        "NINT8": 0.1,
    }
    candidates = []
    for expert in range(2):
        for profile, spec in NINT_EXPERT_PROFILES.items():
            embedded_exposure = 0.1 if expert == 0 else 0.9
            candidates.append(
                ExpertProfileEvaluation(
                    layer=0,
                    expert=expert,
                    profile=profile,
                    spec=spec,
                    gate_name=(
                        "model.language_model.layers.0.experts.gate_up_proj"
                    ),
                    down_name=(
                        "model.language_model.layers.0.experts.down_proj"
                    ),
                    gate_rows=2,
                    gate_columns=48,
                    down_rows=2,
                    down_columns=48,
                    exposure=embedded_exposure,
                    normalized_exposure=embedded_exposure,
                    gate_sse=error[profile],
                    gate_signal=100.0,
                    down_sse=1.0,
                    down_signal=100.0,
                ).as_document()
            )
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(
        "\n".join(json.dumps(row) for row in candidates) + "\n",
        encoding="utf-8",
    )

    nint2 = NINT_EXPERT_PROFILES["NINT2"]
    nint3 = NINT_EXPERT_PROFILES["NINT3"]
    minimum = 4 * nint_storage_bits(2, 48, nint2)
    one_gate_upgrade = (
        nint_storage_bits(2, 48, nint3)
        - nint_storage_bits(2, 48, nint2)
    )
    budget_path = tmp_path / "budgets.json"
    budget_path.write_text(
        json.dumps(
            {
                "format": "mfq.ud-layer-budgets.v1",
                "expected_layers": 1,
                "expected_experts": 2,
                "expected_top_k": 1,
                "layers": {
                    "0": {
                        "target_storage_bits": minimum + one_gate_upgrade,
                        "source_types": {
                            "gate_up": "Q2_K",
                            "down": "Q2_K",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return reap_path, candidate_path, budget_path


def test_layer_budget_allocator_reweights_candidates_from_raw_reap(tmp_path):
    reap_path, candidate_path, budget_path = _write_fixture(tmp_path)
    scheme_path = tmp_path / "scheme.json"
    report_path = tmp_path / "report.json"

    report = allocate_reap_layer_budget(
        reap_path=reap_path,
        candidate_table_path=candidate_path,
        layer_budgets_path=budget_path,
        output_scheme_path=scheme_path,
        output_report_path=report_path,
        target_label="V2-M",
    )

    scheme = load_scheme(scheme_path)
    gate = scheme.require_expert("blk.0.ffn_gate_up_exps.weight")
    down = scheme.require_expert("blk.0.ffn_down_exps.weight")
    assert gate.specs[0] == NINT_EXPERT_PROFILES["NINT3"]
    assert gate.specs[1] == NINT_EXPERT_PROFILES["NINT2"]
    assert down.specs == (
        NINT_EXPERT_PROFILES["NINT2"],
        NINT_EXPERT_PROFILES["NINT2"],
    )
    assert report["layer_reports"]["0"]["actual_storage_bits"] <= (
        report["layer_reports"]["0"]["target_storage_bits"]
    )
    assert report["raw_reap_sha256"]
    assert report["candidate_table_sha256"]
    assert report["layer_budgets_sha256"]


def test_layer_budget_allocator_refuses_to_overwrite_outputs(tmp_path):
    reap_path, candidate_path, budget_path = _write_fixture(tmp_path)
    scheme_path = tmp_path / "scheme.json"
    report_path = tmp_path / "report.json"
    scheme_path.write_text("owned", encoding="utf-8")

    try:
        allocate_reap_layer_budget(
            reap_path=reap_path,
            candidate_table_path=candidate_path,
            layer_budgets_path=budget_path,
            output_scheme_path=scheme_path,
            output_report_path=report_path,
            target_label="V2-M",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output must not be overwritten")
