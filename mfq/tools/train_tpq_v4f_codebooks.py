"""Train native V4F TPQ codebooks with the original weight-only objective."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

from mfq.calibration.artifact import (
    CalibrationScheme,
    ExpertPrecision,
    load_scheme,
    save_scheme,
)
from mfq.calibration.tpq import (
    CccpTierAllocation,
    build_cccp_expert_selection,
    cccp_expert_precision,
)
from mfq.formats.tpq import TPQ_PQ_SPECS_BY_LABEL, normalize_tpq_dtype
from mfq.quantize.tpq import (
    CccpKmeansConfig,
    cccp_reconstruction_sums,
    save_cccp_codebook_artifact,
    train_cccp_codebook,
)
from mfq.quantize.v4f_source import V4FCheckpoint


_NAME = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<projection>gate_up|down)_exps\.weight$"
)


def _config(
    precision: ExpertPrecision,
    *,
    default_seed: int,
) -> CccpKmeansConfig:
    return CccpKmeansConfig(
        iterations=int(precision.option("iterations", 12)),
        restarts=int(precision.option("restarts", 2)),
        sample_points=int(precision.option("sample_points", 100_000)),
        seed=int(precision.option("seed", default_seed)),
        distance_bytes=int(
            precision.option("distance_bytes", 1 << 30)
        ),
    )


def _artifact_path(
    scheme_path: Path,
    precision: ExpertPrecision,
) -> Path:
    if precision.artifact is None:
        raise ValueError(f"{precision.family} precision has no artifact path")
    path = Path(precision.artifact)
    if not path.is_absolute():
        path = scheme_path.parent / path
    return path.resolve()


def _cohorts(selection) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for expert in selection.selections:
        family = expert.descriptor.family
        if family in TPQ_PQ_SPECS_BY_LABEL:
            result.setdefault(family, []).append(expert.expert_id)
    return result


def _precision_map(selection) -> dict[int, ExpertPrecision]:
    return {
        expert.expert_id: expert.descriptor
        for expert in selection.selections
    }


def _family_precision(
    selection,
    *,
    family: str,
    layer: int,
    projection: str,
) -> ExpertPrecision:
    for item in selection.selections:
        if item.descriptor.family == family:
            return item.descriptor
    spec = TPQ_PQ_SPECS_BY_LABEL[family]
    return cccp_expert_precision(
        spec.tier,
        artifact=(
            f"artifacts/layer{layer:02d}-{projection}-"
            f"cccp-{spec.tier}.npz"
        ),
    )


def _load_codebook(
    scheme_path: Path,
    precision: ExpertPrecision,
) -> np.ndarray:
    artifact = _artifact_path(scheme_path, precision)
    with np.load(artifact, allow_pickle=False) as payload:
        family = normalize_tpq_dtype(
            str(np.asarray(payload["family"]).item())
        )
        objective = str(np.asarray(payload["objective"]).item())
        codebook = np.asarray(payload["codebook"], dtype=np.float32)
    if family != precision.family:
        raise ValueError(
            f"CCCP artifact family mismatch: {artifact} has {family}, "
            f"scheme expects {precision.family}"
        )
    if objective != "euclidean_sse":
        raise ValueError(
            f"CCCP artifact {artifact} uses objective {objective!r}"
        )
    spec = TPQ_PQ_SPECS_BY_LABEL[family]
    if codebook.shape != (spec.codebook_entries, spec.vector_size):
        raise ValueError(
            f"CCCP artifact {artifact} has codebook shape {codebook.shape}"
        )
    return codebook


def _artifact_is_original(
    artifact: Path,
    family: str,
) -> bool:
    if not artifact.exists():
        return False
    try:
        with np.load(artifact, allow_pickle=False) as payload:
            return (
                normalize_tpq_dtype(
                    str(np.asarray(payload["family"]).item())
                )
                == family
                and str(np.asarray(payload["objective"]).item())
                == "euclidean_sse"
            )
    except (KeyError, OSError, ValueError):
        return False


def _sample_original_cohort(
    checkpoint: V4FCheckpoint,
    *,
    layer: int,
    expert_ids: list[int],
    vector_size: int,
    points_per_expert: int,
    max_experts: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Match CCCP's tier-local, weight-only GU/down point sampling."""

    if not expert_ids:
        raise ValueError("CCCP cohort must contain at least one expert")
    rng = np.random.default_rng(layer)
    sampled_experts = list(expert_ids)
    if len(sampled_experts) > max_experts:
        sampled_experts = sorted(
            int(value)
            for value in rng.choice(
                sampled_experts,
                max_experts,
                replace=False,
            )
        )
    gate_source = checkpoint.expert_source(layer, "gate_up")
    down_source = checkpoint.expert_source(layer, "down")
    gate_points: list[torch.Tensor] = []
    down_points: list[torch.Tensor] = []
    for expert in sampled_experts:
        gate = gate_source.read_expert_rows(
            expert,
            0,
            gate_source.rows_per_expert,
            device=device,
        ).reshape(-1, vector_size)
        down = down_source.read_expert_rows(
            expert,
            0,
            down_source.rows_per_expert,
            device=device,
        ).reshape(-1, vector_size)
        gate_ids = rng.choice(
            gate.shape[0],
            min(points_per_expert, gate.shape[0]),
            replace=False,
        )
        down_ids = rng.choice(
            down.shape[0],
            min(points_per_expert, down.shape[0]),
            replace=False,
        )
        gate_points.append(
            gate.index_select(
                0,
                torch.as_tensor(gate_ids, dtype=torch.int64, device=gate.device),
            ).cpu()
        )
        down_points.append(
            down.index_select(
                0,
                torch.as_tensor(down_ids, dtype=torch.int64, device=down.device),
            ).cpu()
        )
        del gate, down
    return (
        torch.cat(gate_points, dim=0),
        torch.cat(down_points, dim=0),
        sampled_experts,
    )


def _audit_error(
    checkpoint: V4FCheckpoint,
    *,
    layer: int,
    expert: int,
    family: str,
    codebooks: dict[str, dict[str, np.ndarray]],
    device: str,
) -> float:
    spec = TPQ_PQ_SPECS_BY_LABEL[family]
    gate_source = checkpoint.expert_source(layer, "gate_up")
    down_source = checkpoint.expert_source(layer, "down")
    gate = gate_source.read_expert_rows(
        expert,
        0,
        gate_source.rows_per_expert,
        device=device,
    )
    down = down_source.read_expert_rows(
        expert,
        0,
        down_source.rows_per_expert,
        device=device,
    )
    gate_sse, gate_signal = cccp_reconstruction_sums(
        gate,
        spec,
        codebooks[family]["gate_up"],
        device=device,
    )
    down_sse, down_signal = cccp_reconstruction_sums(
        down,
        spec,
        codebooks[family]["down"],
        device=device,
    )
    del gate, down
    return float(
        np.sqrt(
            (gate_sse + down_sse)
            / max(gate_signal + down_signal, 1e-12)
        )
    )


def _allocation_from_final_tiers(
    tiers: list[str],
    scores: list[float],
) -> CccpTierAllocation:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(tiers)
    total = max(float(values.sum()), 1e-12)
    mass = tuple(
        (
            tier,
            float(values[labels == tier].sum() / total),
        )
        for tier in ("vv", "v", "w", "x")
    )
    return CccpTierAllocation(
        tiers=tuple(tiers),
        scores=tuple(float(value) for value in values),
        boundaries=(
            sum(tier in {"vv", "v"} for tier in tiers),
            sum(tier in {"vv", "v", "w"} for tier in tiers),
        ),
        score_mass=mass,
    )


def _score_vector(
    scheme: CalibrationScheme,
    selection_name: str,
    n_experts: int,
) -> list[float]:
    rows = scheme.candidate_table.get(selection_name, ())
    by_expert = {
        int(row["expert_id"]): float(row.get("score", 1.0))
        for row in rows
        if "expert_id" in row
    }
    return [by_expert.get(expert, 1.0) for expert in range(n_experts)]


def train(
    *,
    input_root: str | Path,
    scheme_path: str | Path,
    device: str = "cuda",
    points_per_expert: int = 50_000,
    vv_points_per_expert: int = 300_000,
    max_experts: int = 32,
    layers: set[int] | None = None,
    overwrite: bool = False,
) -> list[dict]:
    """Train every CCCP cohort using the original unweighted CCCP flow."""

    if points_per_expert <= 0 or vv_points_per_expert <= 0:
        raise ValueError("CCCP points-per-expert values must be positive")
    if max_experts <= 0:
        raise ValueError("CCCP max-experts must be positive")
    scheme = load_scheme(scheme_path)
    if scheme.path is None:
        raise ValueError("V4F CCCP scheme has no source path")
    checkpoint = V4FCheckpoint(input_root)
    events: list[dict] = []
    updated_expert_selections = dict(scheme.expert_selections)
    updated_candidate_table = {
        name: [dict(row) for row in rows]
        for name, rows in scheme.candidate_table.items()
    }
    audit_reports: list[dict] = []
    selections: dict[int, dict[str, object]] = {}
    for name, selection in sorted(scheme.expert_selections.items()):
        match = _NAME.fullmatch(name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        projection = match.group("projection")
        if layers is not None and layer not in layers:
            continue
        selections.setdefault(layer, {})[projection] = selection
    for layer, layer_selections in sorted(selections.items()):
        if set(layer_selections) != {"gate_up", "down"}:
            raise ValueError(
                f"CCCP layer {layer} must contain gate_up and down selections"
            )
        gate_selection = layer_selections["gate_up"]
        down_selection = layer_selections["down"]
        gate_cohorts = _cohorts(gate_selection)
        down_cohorts = _cohorts(down_selection)
        if gate_cohorts != down_cohorts:
            raise ValueError(
                f"CCCP layer {layer} assigns different tiers to gate_up and down"
            )
        if not gate_cohorts:
            continue
        started = time.perf_counter()
        gate_by_id = _precision_map(gate_selection)
        down_by_id = _precision_map(down_selection)
        for family, expert_ids in sorted(gate_cohorts.items()):
            spec = TPQ_PQ_SPECS_BY_LABEL[family]
            gate_precision = gate_by_id[expert_ids[0]]
            down_precision = down_by_id[expert_ids[0]]
            artifacts = {
                "gate_up": _artifact_path(scheme.path, gate_precision),
                "down": _artifact_path(scheme.path, down_precision),
            }
            pending = {
                projection
                for projection, artifact in artifacts.items()
                if overwrite
                or not _artifact_is_original(artifact, family)
            }
            sampled_experts: list[int] = []
            gate_points = down_points = None
            if pending:
                per_expert = (
                    vv_points_per_expert
                    if spec.tier == "vv"
                    else points_per_expert
                )
                gate_points, down_points, sampled_experts = (
                    _sample_original_cohort(
                        checkpoint,
                        layer=layer,
                        expert_ids=expert_ids,
                        vector_size=spec.vector_size,
                        points_per_expert=per_expert,
                        max_experts=max_experts,
                        device=device,
                    )
                )
            for projection, precision, points in (
                ("gate_up", gate_precision, gate_points),
                ("down", down_precision, down_points),
            ):
                artifact = artifacts[projection]
                if projection not in pending:
                    status = "reused"
                    seconds = 0.0
                    sse = None
                else:
                    assert points is not None
                    item_started = time.perf_counter()
                    result = train_cccp_codebook(
                        points,
                        spec,
                        config=_config(
                            precision,
                            default_seed=layer
                            + (1000 if projection == "down" else 0),
                        ),
                        device=device,
                    )
                    save_cccp_codebook_artifact(artifact, spec, result)
                    status = "trained"
                    seconds = time.perf_counter() - item_started
                    sse = result.sse
                event = {
                    "layer": layer,
                    "projection": projection,
                    "family": family,
                    "cohort_experts": len(expert_ids),
                    "sampled_experts": sampled_experts,
                    "objective": "euclidean_sse",
                    "artifact": str(artifact),
                    "status": status,
                    "sse": sse,
                    "seconds": seconds,
                }
                events.append(event)
                print(json.dumps(event), flush=True)
            del gate_points, down_points

        if scheme.metadata.get("tier_source") == "fixed_tiers_per_layer":
            event = {
                "layer": layer,
                "fixed_tiers": True,
                "audit": "skipped",
            }
            events.append(event)
            print(json.dumps(event), flush=True)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "layer": layer,
                        "total_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
            continue

        family_precisions: dict[str, dict[str, ExpertPrecision]] = {}
        codebooks: dict[str, dict[str, np.ndarray]] = {}
        for family in gate_cohorts:
            family_precisions[family] = {
                "gate_up": _family_precision(
                    gate_selection,
                    family=family,
                    layer=layer,
                    projection="gate_up",
                ),
                "down": _family_precision(
                    down_selection,
                    family=family,
                    layer=layer,
                    projection="down",
                ),
            }
            codebooks[family] = {
                projection: _load_codebook(
                    scheme.path,
                    family_precisions[family][projection],
                )
                for projection in ("gate_up", "down")
            }

        n_experts = gate_selection.n_experts
        initial_families = [
            gate_by_id[expert].family for expert in range(n_experts)
        ]
        final_families = list(initial_families)
        relative_rmse = [
            _audit_error(
                checkpoint,
                layer=layer,
                expert=expert,
                family=initial_families[expert],
                codebooks=codebooks,
                device=device,
            )
            for expert in range(n_experts)
        ]
        medians = {
            family: sorted(
                relative_rmse[expert]
                for expert in range(n_experts)
                if initial_families[expert] == family
            )[
                sum(
                    initial_families[expert] == family
                    for expert in range(n_experts)
                )
                // 2
            ]
            for family in codebooks
        }
        upgrades: list[dict] = []

        def ensure_vv(expert: int) -> None:
            family = "TPQ-VV"
            if family in codebooks:
                return
            spec = TPQ_PQ_SPECS_BY_LABEL[family]
            precisions = {
                "gate_up": _family_precision(
                    gate_selection,
                    family=family,
                    layer=layer,
                    projection="gate_up",
                ),
                "down": _family_precision(
                    down_selection,
                    family=family,
                    layer=layer,
                    projection="down",
                ),
            }
            artifacts = {
                projection: _artifact_path(scheme.path, precision)
                for projection, precision in precisions.items()
            }
            reuse = (
                not overwrite
                and all(
                    _artifact_is_original(path, family)
                    for path in artifacts.values()
                )
            )
            if not reuse:
                gate_points, down_points, sampled = _sample_original_cohort(
                    checkpoint,
                    layer=layer,
                    expert_ids=[expert],
                    vector_size=spec.vector_size,
                    points_per_expert=vv_points_per_expert,
                    max_experts=max_experts,
                    device=device,
                )
                for projection, points in (
                    ("gate_up", gate_points),
                    ("down", down_points),
                ):
                    result = train_cccp_codebook(
                        points,
                        spec,
                        config=_config(
                            precisions[projection],
                            default_seed=layer
                            + (1000 if projection == "down" else 0),
                        ),
                        device=device,
                    )
                    save_cccp_codebook_artifact(
                        artifacts[projection],
                        spec,
                        result,
                    )
                    event = {
                        "layer": layer,
                        "projection": projection,
                        "family": family,
                        "cohort_experts": 1,
                        "sampled_experts": sampled,
                        "objective": "euclidean_sse",
                        "artifact": str(artifacts[projection]),
                        "status": "trained-for-audit",
                        "sse": result.sse,
                    }
                    events.append(event)
                    print(json.dumps(event), flush=True)
                del gate_points, down_points
            family_precisions[family] = precisions
            codebooks[family] = {
                projection: _load_codebook(
                    scheme.path,
                    precisions[projection],
                )
                for projection in ("gate_up", "down")
            }

        for expert in range(n_experts):
            family = final_families[expert]
            median = medians[family]
            if relative_rmse[expert] <= 1.5 * max(median, 1e-9):
                continue
            previous_error = relative_rmse[expert]
            if family != "TPQ-V" and "TPQ-V" in codebooks:
                candidate_error = _audit_error(
                    checkpoint,
                    layer=layer,
                    expert=expert,
                    family="TPQ-V",
                    codebooks=codebooks,
                    device=device,
                )
                if candidate_error <= 1.5 * max(
                    medians["TPQ-V"],
                    1e-9,
                ):
                    final_families[expert] = "TPQ-V"
                    relative_rmse[expert] = candidate_error
                    upgrades.append(
                        {
                            "expert": expert,
                            "from": family,
                            "to": "TPQ-V",
                            "before": previous_error,
                            "after": candidate_error,
                        }
                    )
                    continue
            ensure_vv(expert)
            candidate_error = _audit_error(
                checkpoint,
                layer=layer,
                expert=expert,
                family="TPQ-VV",
                codebooks=codebooks,
                device=device,
            )
            final_families[expert] = "TPQ-VV"
            relative_rmse[expert] = candidate_error
            upgrades.append(
                {
                    "expert": expert,
                    "from": family,
                    "to": "TPQ-VV",
                    "before": previous_error,
                    "after": candidate_error,
                }
            )
            vv_values = sorted(
                relative_rmse[index]
                for index in range(n_experts)
                if final_families[index] == "TPQ-VV"
            )
            medians["TPQ-VV"] = vv_values[len(vv_values) // 2]

        gate_name = gate_selection.name
        down_name = down_selection.name
        scores = _score_vector(scheme, gate_name, n_experts)
        final_tiers = [
            TPQ_PQ_SPECS_BY_LABEL[family].tier for family in final_families
        ]
        allocation = _allocation_from_final_tiers(final_tiers, scores)
        for projection, original in (
            ("gate_up", gate_selection),
            ("down", down_selection),
        ):
            artifacts = {
                tier: _family_precision(
                    original,
                    family=f"TPQ-{tier.upper()}",
                    layer=layer,
                    projection=projection,
                ).artifact
                for tier in set(final_tiers)
            }
            if any(path is None for path in artifacts.values()):
                raise ValueError(f"CCCP layer {layer} has an artifact-less tier")
            updated_expert_selections[original.name] = (
                build_cccp_expert_selection(
                    name=original.name,
                    group=original.group,
                    allocation=allocation,
                    rows_per_expert=original.rows_per_expert,
                    columns=original.columns,
                    artifacts={
                        tier: str(path)
                        for tier, path in artifacts.items()
                    },
                )
            )
            previous_rows = {
                int(row["expert_id"]): row
                for row in updated_candidate_table.get(original.name, ())
                if "expert_id" in row
            }
            updated_candidate_table[original.name] = [
                {
                    **previous_rows.get(expert, {}),
                    "expert_id": expert,
                    "tier": final_tiers[expert],
                    "score": scores[expert],
                    "audit_relative_rmse": relative_rmse[expert],
                }
                for expert in range(n_experts)
            ]
        audit_report = {
            "layer": layer,
            "metric": "joint_gate_up_down_relative_rmse",
            "upgrade_factor": 1.5,
            "medians": medians,
            "upgrades": upgrades,
            "final_counts": allocation.counts,
        }
        audit_reports.append(audit_report)
        event = {"audit": audit_report}
        events.append(event)
        print(json.dumps(event), flush=True)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "layer": layer,
                    "total_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    if not events:
        raise ValueError("scheme contains no V4F CCCP expert cohorts")
    metadata = {
        **scheme.metadata,
        "codebook_objective": "euclidean_sse",
        "codebook_calibration_data": "none",
    }
    if scheme.metadata.get("tier_source") != "fixed_tiers_per_layer":
        metadata.update(
            {
                "audit_metric": "joint_gate_up_down_relative_rmse",
                "audit_upgrade_factor": 1.5,
                "audit": audit_reports,
            }
        )
    updated = CalibrationScheme(
        path=scheme.path,
        target_profile=scheme.target_profile,
        target_storage_bits=0,
        selections=scheme.selections,
        metadata=metadata,
        candidate_table=updated_candidate_table,
        inint_selector=scheme.inint_selector,
        expert_selections=updated_expert_selections,
    )
    updated = CalibrationScheme(
        path=updated.path,
        target_profile=updated.target_profile,
        target_storage_bits=updated.storage_bits,
        selections=updated.selections,
        metadata=updated.metadata,
        candidate_table=updated.candidate_table,
        inint_selector=updated.inint_selector,
        expert_selections=updated.expert_selections,
    )
    temporary = scheme.path.with_suffix(
        scheme.path.suffix + ".updated.partial"
    )
    temporary.unlink(missing_ok=True)
    save_scheme(temporary, updated)
    os.replace(temporary, scheme.path)
    return events


def _layers(value: str) -> set[int] | None:
    if not value:
        return None
    result: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--scheme", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-expert", type=int, default=50_000)
    parser.add_argument("--vv-points-per-expert", type=int, default=300_000)
    parser.add_argument("--max-experts", type=int, default=32)
    parser.add_argument("--layers", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train(
        input_root=args.input,
        scheme_path=args.scheme,
        device=args.device,
        points_per_expert=args.points_per_expert,
        vv_points_per_expert=args.vv_points_per_expert,
        max_experts=args.max_experts,
        layers=_layers(args.layers),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
