from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from mfq import cli
from mfq.calibration.artifact import load_scheme
from mfq.calibration.ew_solver import (
    load_budget_document,
    load_candidate_document,
    load_importance_document,
    solve_ew_budget,
)
from mfq.tools.quantize_hf_to_mfq import _glm_expert_precisions


def _precision(bits: int) -> dict:
    groupsize = {2: 16, 4: 24}[bits]
    sub_bits = {2: 5, 4: 6}[bits]
    return {
        "family": f"NINT{bits}",
        "nint_spec": {
            "bits": bits,
            "groupsize": groupsize,
            "sub_bits": sub_bits,
        },
    }


def _tensor(
    name: str,
    projection: str,
    *,
    layer: int = 0,
    experts: int = 4,
    reference_bpw: list[float] | None = None,
    fixed_storage_bits: int = 0,
    pool_storage_bits: int = 0,
) -> dict:
    candidates = []
    for expert in range(experts):
        candidates.extend(
            [
                {
                    "expert_id": expert,
                    "profile": "low",
                    "precision": _precision(2),
                    "variable_storage_bits": 100,
                    "pool_key": "low",
                    "pool_storage_bits": pool_storage_bits,
                    "distortion": 1.0,
                    "validation_distortion": 1.25,
                    "effective_bpw": 1.0,
                },
                {
                    "expert_id": expert,
                    "profile": "high",
                    "precision": _precision(4),
                    "variable_storage_bits": 300,
                    "pool_key": "high",
                    "pool_storage_bits": pool_storage_bits,
                    "distortion": 0.0,
                    "validation_distortion": 0.0,
                    "effective_bpw": 3.0,
                },
            ]
        )
    value = {
        "name": name,
        "group": f"layer.{layer}.{projection}",
        "layer": layer,
        "projection": projection,
        "n_experts": experts,
        "rows_per_expert": 1,
        "columns": 100,
        "fixed_storage_bits": fixed_storage_bits,
        "candidates": candidates,
    }
    if reference_bpw is not None:
        value["reference_bpw"] = reference_bpw
    return value


def _candidate_document(*tensors: dict) -> dict:
    return {
        "format": "mfq.ew-candidates.v1",
        "metadata": {"source": "synthetic"},
        "tensors": list(tensors),
    }


def _rank_importance(experts: int) -> dict:
    return {
        "format": "mfq.ew-importance.v1",
        "mode": "rank",
        "rank_weighting": "linear_percentile",
        "entries": [
            {"layer": 0, "expert_id": expert, "rank": expert + 1} for expert in range(experts)
        ],
    }


def test_rank_only_joint_solver_preserves_peak_and_exact_storage() -> None:
    candidates = load_candidate_document(
        _candidate_document(
            _tensor(
                "blk.0.ffn_down_exps.weight",
                "down",
                reference_bpw=[3.0, 3.0, 1.0, 1.0],
                fixed_storage_bits=10,
                pool_storage_bits=10,
            )
        )
    )
    importance = load_importance_document(_rank_importance(4))
    budget = load_budget_document(
        {
            "format": "mfq.ew-budget.v1",
            "target_profile": "TPQ-S-SHAPE",
            "model_weight_count": 400,
            "total": {"target_bits": 830},
            "shape_constraints": [
                {
                    "name": "down-bimodal",
                    "projection": "down",
                    "peak_reference_min_bpw": 3.0,
                    "peak_selected_min_bpw": 3.0,
                    "min_peak_retention": 1.0,
                    "min_contrast_ratio": 1.0,
                    "histogram": [
                        {
                            "side": "ge",
                            "threshold_bpw": 3.0,
                            "relative_tolerance": 0.0,
                        },
                        {
                            "side": "le",
                            "threshold_bpw": 1.0,
                            "relative_tolerance": 0.0,
                        },
                    ],
                }
            ],
        }
    )

    result = solve_ew_budget(importance, candidates, budget)

    assert result.scheme.storage_bits == 830
    assert result.scheme.target_storage_bits == 830
    assert [result.selected[item].profile for item in sorted(result.selected)] == [
        "high",
        "high",
        "low",
        "low",
    ]
    assert (
        sum(
            expert.storage_bits
            for expert in result.scheme.expert_selections["blk.0.ffn_down_exps.weight"].selections
        )
        == 830
    )
    shape = result.report["shape_constraints"]["down-bimodal"]
    assert shape["peak_retention"] == 1.0
    assert shape["contrast_ratio"] == 1.0
    assert all(item["within_bounds"] for item in shape["histogram"])
    assert result.report["budgets"]["total"]["actual_storage_bits"] == 830


def test_score_mode_enforces_total_projection_and_optional_layer_budgets() -> None:
    candidates = load_candidate_document(
        _candidate_document(
            _tensor("blk.0.ffn_gate_exps.weight", "gate", experts=2),
            _tensor("blk.0.ffn_down_exps.weight", "down", experts=2),
            _tensor("blk.1.ffn_gate_exps.weight", "gate", layer=1, experts=2),
            _tensor("blk.1.ffn_down_exps.weight", "down", layer=1, experts=2),
        )
    )
    importance = load_importance_document(
        {
            "format": "mfq.ew-importance.v1",
            "mode": "score",
            "score_normalization": "none",
            "entries": [
                {"layer": 0, "expert_id": 0, "score": 10.0},
                {"layer": 0, "expert_id": 1, "score": 1.0},
                {"layer": 1, "expert_id": 0, "score": 8.0},
                {"layer": 1, "expert_id": 1, "score": 2.0},
            ],
        }
    )
    budget = load_budget_document(
        {
            "format": "mfq.ew-budget.v1",
            "model_weight_count": 800,
            "total": {"target_bpw": 2.0},
            "projections": {"down": {"target_bpw": 3.0}},
            "layers": {"0": {"target_bits": 800}},
        }
    )

    result = solve_ew_budget(importance, candidates, budget)

    assert result.report["model_storage_bits"] == 1600
    assert result.report["model_bpw"] == 2.0
    assert result.report["budgets"]["projection:down"]["actual_bpw"] == 3.0
    assert result.report["budgets"]["layer:0"]["actual_storage_bits"] == 800
    assert {
        result.selected[item].profile for item in result.selected if item.projection == "down"
    } == {"high"}
    assert {
        result.selected[item].profile for item in result.selected if item.projection == "gate"
    } == {"low"}


def test_rank_only_objective_prioritizes_the_highest_rank() -> None:
    candidates = load_candidate_document(
        _candidate_document(_tensor("blk.0.ffn_down_exps.weight", "down", experts=2))
    )
    importance = load_importance_document(_rank_importance(2))
    budget = load_budget_document(
        {
            "format": "mfq.ew-budget.v1",
            "model_weight_count": 200,
            "total": {"target_bits": 400},
        }
    )

    result = solve_ew_budget(importance, candidates, budget)

    selected = {item.expert_id: candidate.profile for item, candidate in result.selected.items()}
    assert selected == {0: "high", 1: "low"}


def test_candidate_table_allows_expert_specific_profile_sets() -> None:
    tensor = _tensor("blk.0.ffn_down_exps.weight", "down", experts=2)
    tensor["candidates"] = [
        candidate
        for candidate in tensor["candidates"]
        if not (candidate["expert_id"] == 0 and candidate["profile"] == "low")
    ]

    candidates = load_candidate_document(_candidate_document(tensor))

    profiles = {
        item.expert_id: {
            candidate.profile for candidate in candidates.candidates if candidate.key == item
        }
        for item in candidates.items
    }
    assert profiles == {0: {"high"}, 1: {"low", "high"}}


def test_milp_matches_bruteforce_with_pool_activation_costs() -> None:
    candidates = load_candidate_document(
        _candidate_document(
            _tensor(
                "blk.0.ffn_down_exps.weight",
                "down",
                experts=3,
                fixed_storage_bits=10,
                pool_storage_bits=20,
            )
        )
    )
    importance = load_importance_document(
        {
            "format": "mfq.ew-importance.v1",
            "mode": "score",
            "entries": [
                {"layer": 0, "expert_id": 0, "score": 10.0},
                {"layer": 0, "expert_id": 1, "score": 5.0},
                {"layer": 0, "expert_id": 2, "score": 1.0},
            ],
        }
    )
    budget = load_budget_document(
        {
            "format": "mfq.ew-budget.v1",
            "model_weight_count": 300,
            "total": {"max_bits": 750},
        }
    )

    result = solve_ew_budget(importance, candidates, budget)
    selected_profiles = tuple(result.selected[item].profile for item in sorted(result.selected))

    weights = (10.0, 5.0, 1.0)
    feasible = []
    for profiles in itertools.product(("low", "high"), repeat=3):
        variable_bits = sum(100 if profile == "low" else 300 for profile in profiles)
        active_pools = len(set(profiles))
        storage_bits = 10 + variable_bits + 20 * active_pools
        if storage_bits <= 750:
            loss = sum(
                weight
                for weight, profile in zip(weights, profiles, strict=True)
                if profile == "low"
            )
            feasible.append((loss, storage_bits, profiles))
    optimum = min(feasible)
    assert selected_profiles == optimum[2]
    assert result.report["model_storage_bits"] == optimum[1]


def test_projection_specific_score_overrides_shared_score() -> None:
    importance = load_importance_document(
        {
            "format": "mfq.ew-importance.v1",
            "mode": "score",
            "entries": [
                {"layer": 0, "expert_id": 0, "score": 1.0},
                {"layer": 0, "expert_id": 1, "score": 10.0},
                {
                    "layer": 0,
                    "expert_id": 0,
                    "projection": "down",
                    "score": 20.0,
                },
                {
                    "layer": 0,
                    "expert_id": 1,
                    "projection": "down",
                    "score": 2.0,
                },
            ],
        }
    )
    candidates = load_candidate_document(
        _candidate_document(_tensor("blk.0.ffn_down_exps.weight", "down", experts=2))
    )
    weights = importance.weights_for(candidates.items)
    by_expert = {item.expert_id: weight for item, weight in weights.items()}
    assert by_expert == {0: 20.0, 1: 2.0}


def test_rank_mode_requires_complete_unique_ranks() -> None:
    document = _rank_importance(3)
    document["entries"][2]["rank"] = 2
    with pytest.raises(ValueError, match="each rank"):
        load_importance_document(document)


@pytest.mark.parametrize("layer", [None, "invalid", float("nan"), True])
def test_importance_rejects_invalid_integer_fields_as_validation_errors(layer: object) -> None:
    document = _rank_importance(1)
    document["entries"][0]["layer"] = layer
    with pytest.raises(ValueError, match="layer must be a non-negative integer"):
        load_importance_document(document)


def test_shape_constraint_requires_reference_bpw() -> None:
    candidates = load_candidate_document(
        _candidate_document(_tensor("blk.0.ffn_down_exps.weight", "down", experts=2))
    )
    importance = load_importance_document(_rank_importance(2))
    budget = load_budget_document(
        {
            "format": "mfq.ew-budget.v1",
            "model_weight_count": 200,
            "total": {"target_bits": 400},
            "shape_constraints": [
                {
                    "name": "missing-reference",
                    "projection": "down",
                    "peak_reference_min_bpw": 3.0,
                    "peak_selected_min_bpw": 3.0,
                    "min_peak_retention": 1.0,
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="requires reference_bpw"):
        solve_ew_budget(importance, candidates, budget)


def test_model_weight_count_covers_routed_candidate_weights() -> None:
    candidates = load_candidate_document(
        _candidate_document(_tensor("blk.0.ffn_down_exps.weight", "down", experts=2))
    )
    importance = load_importance_document(_rank_importance(2))
    budget = load_budget_document(
        {
            "format": "mfq.ew-budget.v1",
            "model_weight_count": 199,
            "total": {"max_bits": 400},
        }
    )
    with pytest.raises(ValueError, match="smaller than the routed candidate weight count"):
        solve_ew_budget(importance, candidates, budget)


def test_solve_ew_cli_requires_distinct_output_paths(tmp_path: Path) -> None:
    output = tmp_path / "allocation.json"
    with pytest.raises(ValueError, match="paths must differ"):
        cli.main(
            [
                "solve-ew",
                "--importance",
                str(tmp_path / "importance.json"),
                "--candidates",
                str(tmp_path / "candidates.json"),
                "--budget",
                str(tmp_path / "budget.json"),
                "--output-scheme",
                str(output),
                "--report",
                str(output),
            ]
        )


def test_solve_ew_cli_writes_quantizer_scheme_and_report(tmp_path: Path) -> None:
    importance_path = tmp_path / "importance.json"
    candidates_path = tmp_path / "candidates.json"
    budget_path = tmp_path / "budget.json"
    scheme_path = tmp_path / "scheme.json"
    report_path = tmp_path / "report.json"
    importance_path.write_text(json.dumps(_rank_importance(4)), encoding="utf-8")
    candidates_path.write_text(
        json.dumps(
            _candidate_document(
                _tensor(
                    "blk.0.ffn_down_exps.weight",
                    "down",
                    reference_bpw=[3.0, 3.0, 1.0, 1.0],
                    fixed_storage_bits=10,
                    pool_storage_bits=10,
                )
            )
        ),
        encoding="utf-8",
    )
    budget_path.write_text(
        json.dumps(
            {
                "format": "mfq.ew-budget.v1",
                "target_profile": "EW-CLI",
                "model_weight_count": 400,
                "total": {"target_bits": 830},
                "shape_constraints": [
                    {
                        "name": "down-peaks",
                        "projection": "down",
                        "peak_reference_min_bpw": 3.0,
                        "peak_selected_min_bpw": 3.0,
                        "min_peak_retention": 1.0,
                        "min_contrast_ratio": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "solve-ew",
                "--importance",
                str(importance_path),
                "--candidates",
                str(candidates_path),
                "--budget",
                str(budget_path),
                "--output-scheme",
                str(scheme_path),
                "--report",
                str(report_path),
            ]
        )
        == 0
    )
    scheme = load_scheme(scheme_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert scheme.storage_bits == 830
    assert scheme.target_profile == "EW-CLI"
    assert report["format"] == "mfq.ew-allocation.v1"
    assert report["scheme_sha256"]
    assert report["shape_constraints"]["down-peaks"]["peak_retention"] == 1.0
    precisions = _glm_expert_precisions("blk.0.ffn_down_exps.weight", (4, 1, 100), scheme)
    assert [precision.family for precision in precisions] == [
        "NINT4",
        "NINT4",
        "NINT2",
        "NINT2",
    ]
