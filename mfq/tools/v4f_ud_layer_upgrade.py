"""Plan a monotonic V4F precision upgrade with explicit UD layer priorities."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mfq.quantize.expert_sensitivity import load_expert_sensitivity_map
from mfq.quantize.v4f_plan import routed_family_pool_bytes
from mfq.quantize.v4f_upgrade import _profile_routed_bytes, _read_reap
from mfq.tools.v4f_sensitivity_overlay import (
    PROJECTIONS,
    UD_IQ1_M_LAYER_CONSTRAINTS,
    _changes,
    _family_counts,
    _mapping,
    _pool_affine_bytes,
    _sha256,
)

FAMILIES = ("NEPQ0-S", "NVQ2J", "NINT4", "NINT5", "NINT6", "NINT8")
HIGH_FAMILIES = frozenset({"NINT4", "NINT5", "NINT6", "NINT8"})
UD_IQ1_M_GATE_UP_HIGH_LAYERS = next(
    constraint.high_layers
    for constraint in UD_IQ1_M_LAYER_CONSTRAINTS
    if constraint.projection == "gate_up"
)
UD_IQ1_M_DOWN_SPECIAL_LAYERS = next(
    constraint.high_layers
    for constraint in UD_IQ1_M_LAYER_CONSTRAINTS
    if constraint.projection == "down"
)
DEFAULT_FAMILY_SNR_DB = {
    "gate_up": {
        "NEPQ0-S": 4.900766498678068,
        "NVQ2J": 10.59587019259313,
        "NINT4": 25.21482056618251,
        "NINT5": 29.26,
        "NINT6": 35.06,
        "NINT8": 43.65,
    },
    "down": {
        "NEPQ0-S": 6.053922850730497,
        "NVQ2J": 11.060234763932346,
        "NINT4": 25.223398406907663,
        "NINT5": 29.26,
        "NINT6": 35.06,
        "NINT8": 43.65,
    },
}


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _layer_map(raw: dict, key: str) -> dict[int, set[int]]:
    return {
        int(layer): {int(expert) for expert in experts}
        for layer, experts in raw.get(key, {}).items()
    }


def profiles_from_sensitivity_plan(
    path: str | Path,
) -> tuple[dict[tuple[str, int], tuple[str, ...]], dict]:
    """Reconstruct the exact final profile of a V4F sensitivity plan."""

    plan_path = Path(path).resolve()
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    if raw.get("format") != "mfq.v4f-sensitivity-reallocation.v1":
        raise ValueError("base profile plan must be a V4F sensitivity reallocation")
    profiles: dict[tuple[str, int], tuple[str, ...]] = {}
    for projection in PROJECTIONS:
        base_nepq = _layer_map(raw, f"base_{projection}_nepq0s")
        base_nint4 = _layer_map(raw, f"base_{projection}_nint4")
        final_nint4 = _layer_map(raw, f"{projection}_nint4")
        final_nint8 = _layer_map(raw, f"{projection}_nint8")
        for layer in range(43):
            families = []
            for expert in range(256):
                if expert in final_nint8.get(layer, set()):
                    family = "NINT8"
                elif expert in final_nint4.get(layer, set()):
                    family = "NINT4"
                elif expert in base_nepq.get(layer, set()):
                    family = "NEPQ0-S"
                elif expert in base_nint4.get(layer, set()):
                    raise ValueError(
                        f"base NINT4 expert disappeared: "
                        f"{projection}:{layer}:{expert}"
                    )
                else:
                    family = "NVQ2J"
                families.append(family)
            profiles[(projection, layer)] = tuple(families)
    routed = _profile_routed_bytes(profiles)
    if routed != int(raw["routed_bytes"]):
        raise ValueError(
            f"reconstructed routed bytes differ: {routed} != {raw['routed_bytes']}"
        )
    counts = Counter(
        family
        for families in profiles.values()
        for family in families
    )
    expected_counts = {
        "NEPQ0-S": int(raw["gate_up_nepq0s_count"])
        + int(raw["down_nepq0s_count"]),
        "NVQ2J": int(raw["gate_up_nvq2j_count"])
        + int(raw["down_nvq2j_count"]),
        "NINT4": int(raw["gate_up_nint4_count"])
        + int(raw["down_nint4_count"]),
        "NINT8": int(raw["gate_up_nint8_count"])
        + int(raw["down_nint8_count"]),
    }
    if dict(counts) != expected_counts:
        raise ValueError(
            f"reconstructed family counts differ: {dict(counts)} "
            f"!= {expected_counts}"
        )
    metadata = {
        "base_plan": str(plan_path),
        "base_plan_sha256": _sha256(plan_path),
        "file_bytes": int(raw["estimated_blob_bytes"]) + 61_298,
        "estimated_blob_bytes": int(raw["estimated_blob_bytes"]),
        "routed_bytes": int(raw["routed_bytes"]),
        "nonexpert_bytes": int(raw["nonexpert_bytes"]),
        "allocation_sha256": str(raw["base_allocation_sha256"]),
        "source_index_sha256": str(raw["source_index_sha256"]),
    }
    return profiles, metadata


def _allowed_families(base_family: str, mark: str) -> tuple[str, ...]:
    if mark == "V" or base_family == "NINT8":
        return ("NINT8",)
    if base_family == "NINT4":
        return (
            ("NINT4", "NINT5", "NINT6", "NINT8")
            if mark == "v"
            else ("NINT4",)
        )
    if base_family == "NVQ2J":
        return (
            ("NVQ2J", "NINT4", "NINT5", "NINT6", "NINT8")
            if mark == "v"
            else ("NVQ2J", "NINT4")
        )
    if base_family == "NEPQ0-S":
        return (
            ("NEPQ0-S", "NINT4", "NINT5", "NINT6", "NINT8")
            if mark == "v"
            else ("NEPQ0-S", "NINT4")
        )
    raise ValueError(f"unsupported base family: {base_family}")


def _family_nmse(
    family_snr_db: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        projection: {
            family: 10.0 ** (-float(snr) / 10.0)
            for family, snr in values.items()
        }
        for projection, values in family_snr_db.items()
    }


def solve_ud_layer_upgrade(
    base: dict[tuple[str, int], tuple[str, ...]],
    energy: dict[tuple[int, int], dict[str, float]],
    marks,
    *,
    routed_limit: int,
    family_snr_db: dict[str, dict[str, float]] = DEFAULT_FAMILY_SNR_DB,
    v_multipliers: tuple[float, ...] = (1.0, 4.0, 8.0, 16.0),
) -> tuple[dict[tuple[str, int], list[str]], dict]:
    """Solve a lexicographic UD-layer quota and robust REAP allocation."""

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if not v_multipliers or any(value <= 0 for value in v_multipliers):
        raise ValueError("v sensitivity multipliers must be positive")
    actual = {
        family
        for families in base.values()
        for family in families
    }
    if not actual <= set(FAMILIES):
        raise ValueError(f"unsupported base families: {sorted(actual)}")
    nmse = _family_nmse(family_snr_db)
    for projection in PROJECTIONS:
        if set(nmse.get(projection, {})) != set(FAMILIES):
            raise ValueError(f"incomplete family SNR table for {projection}")

    groups = [
        (projection, layer, expert)
        for projection, layer in sorted(base)
        for expert in range(256)
    ]
    x_meta: list[tuple[str, int, int, str]] = []
    group_x: list[list[int]] = []
    for projection, layer, expert in groups:
        indices = []
        base_family = base[(projection, layer)][expert]
        for family in _allowed_families(
            base_family,
            marks.mark(layer, expert),
        ):
            indices.append(len(x_meta))
            x_meta.append((projection, layer, expert, family))
        group_x.append(indices)
    x_count = len(x_meta)

    pool_keys = [
        (projection, layer, family)
        for projection, layer in sorted(base)
        for family in FAMILIES
    ]
    pool_index = {
        key: x_count + index
        for index, key in enumerate(pool_keys)
    }
    quota_index = x_count + len(pool_keys)
    worst_loss_index = quota_index + 1
    variable_count = worst_loss_index + 1

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_row(
        entries: list[tuple[int, float]],
        low: float,
        high: float,
    ) -> None:
        row = len(lower)
        for column, value in entries:
            rows.append(row)
            columns.append(column)
            values.append(float(value))
        lower.append(float(low))
        upper.append(float(high))

    for indices in group_x:
        add_row([(index, 1.0) for index in indices], 1.0, 1.0)
    for index, (projection, layer, _expert, family) in enumerate(x_meta):
        add_row(
            [
                (index, 1.0),
                (pool_index[(projection, layer, family)], -1.0),
            ],
            -np.inf,
            0.0,
        )

    fixed_headers = len(base) * 20
    variable_budget = int(routed_limit) - fixed_headers
    if variable_budget <= 0:
        raise ValueError("routed byte limit cannot hold routed headers")
    pool_costs = {
        (projection, family): _pool_affine_bytes(projection, family)
        for projection in PROJECTIONS
        for family in FAMILIES
    }
    budget_entries = [
        (
            index,
            pool_costs[(projection, family)][1] / variable_budget,
        )
        for index, (projection, _layer, _expert, family) in enumerate(x_meta)
    ]
    budget_entries.extend(
        (
            index,
            pool_costs[(projection, family)][0] / variable_budget,
        )
        for (projection, _layer, family), index in pool_index.items()
    )
    add_row(budget_entries, -np.inf, 1.0)

    for layer in UD_IQ1_M_GATE_UP_HIGH_LAYERS:
        entries = [
            (index, 1.0)
            for index, (projection, item_layer, _expert, family)
            in enumerate(x_meta)
            if projection == "gate_up"
            and item_layer == layer
            and family in HIGH_FAMILIES
        ]
        entries.append((quota_index, -1.0))
        add_row(entries, 0.0, np.inf)
    for layer in UD_IQ1_M_DOWN_SPECIAL_LAYERS:
        entries = [
            (index, 1.0)
            for index, (projection, item_layer, _expert, family)
            in enumerate(x_meta)
            if projection == "down"
            and item_layer == layer
            and family in HIGH_FAMILIES
        ]
        add_row(entries, 256.0, np.inf)

    scenario_baselines = []
    for multiplier in v_multipliers:
        baseline = 0.0
        entries = [(worst_loss_index, -1.0)]
        for index, (projection, layer, expert, family) in enumerate(x_meta):
            mark = marks.mark(layer, expert)
            sensitivity = (
                32.0 if mark == "V"
                else multiplier if mark == "v"
                else 1.0
            )
            loss = (
                energy[(layer, expert)][projection]
                * sensitivity
                * nmse[projection][family]
            )
            entries.append((index, loss))
        for projection, layer, expert in groups:
            mark = marks.mark(layer, expert)
            sensitivity = (
                32.0 if mark == "V"
                else multiplier if mark == "v"
                else 1.0
            )
            baseline += (
                energy[(layer, expert)][projection]
                * sensitivity
                * nmse[projection][base[(projection, layer)][expert]]
            )
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("invalid baseline objective")
        scenario_baselines.append(baseline)
        normalized = [
            (
                column,
                value / baseline
                if column != worst_loss_index
                else value,
            )
            for column, value in entries
        ]
        add_row(normalized, -np.inf, 0.0)

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(len(lower), variable_count),
    ).tocsr()
    integrality = np.ones(variable_count, dtype=np.uint8)
    integrality[worst_loss_index] = 0
    bounds_lower = np.zeros(variable_count, dtype=np.float64)
    bounds_upper = np.ones(variable_count, dtype=np.float64)
    bounds_upper[quota_index] = 256.0
    bounds_upper[worst_loss_index] = np.inf
    constraints = LinearConstraint(
        matrix,
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )

    quota_objective = np.zeros(variable_count, dtype=np.float64)
    quota_objective[quota_index] = -1.0
    stage_a = milp(
        c=quota_objective,
        integrality=integrality,
        bounds=Bounds(bounds_lower, bounds_upper),
        constraints=constraints,
        options={"presolve": True},
    )
    if not stage_a.success or stage_a.x is None:
        raise RuntimeError(f"UD quota solve failed: {stage_a.message}")
    quota = int(round(float(stage_a.x[quota_index])))
    if abs(float(stage_a.x[quota_index]) - quota) > 1e-5:
        raise RuntimeError("UD quota solve returned a fractional quota")

    quality_objective = np.zeros(variable_count, dtype=np.float64)
    quality_objective[worst_loss_index] = 1.0
    for index, (projection, _layer, _expert, family) in enumerate(x_meta):
        quality_objective[index] = (
            1e-10
            * pool_costs[(projection, family)][1]
            / variable_budget
        )
    for (projection, _layer, family), index in pool_index.items():
        quality_objective[index] = (
            1e-10
            * pool_costs[(projection, family)][0]
            / variable_budget
        )
    stage_b_lower = bounds_lower.copy()
    stage_b_upper = bounds_upper.copy()
    stage_b_lower[quota_index] = quota
    stage_b_upper[quota_index] = quota
    stage_b = milp(
        c=quality_objective,
        integrality=integrality,
        bounds=Bounds(stage_b_lower, stage_b_upper),
        constraints=constraints,
        options={"presolve": True},
    )
    if not stage_b.success or stage_b.x is None:
        raise RuntimeError(f"UD quality solve failed: {stage_b.message}")

    final = {key: [""] * 256 for key in base}
    for group, indices in zip(groups, group_x, strict=True):
        chosen = max(indices, key=lambda index: stage_b.x[index])
        if stage_b.x[chosen] < 0.5:
            raise RuntimeError("UD MILP returned a fractional expert assignment")
        projection, layer, expert = group
        final[(projection, layer)][expert] = x_meta[chosen][3]
    exact_routed = _profile_routed_bytes(final)
    if exact_routed > routed_limit:
        raise RuntimeError(
            f"UD allocation exceeds exact routed budget: "
            f"{exact_routed} > {routed_limit}"
        )
    changed = _changes(base, final)
    exact_overlay_payload = sum(
        20
        + sum(
            routed_family_pool_bytes(projection, family, count)
            for family, count in Counter(entries.values()).items()
        )
        for (projection, _layer), entries in changed.items()
    )
    quota_counts = {
        str(layer): sum(
            family in HIGH_FAMILIES
            for family in final[("gate_up", layer)]
        )
        for layer in UD_IQ1_M_GATE_UP_HIGH_LAYERS
    }
    down_counts = {
        str(layer): sum(
            family in HIGH_FAMILIES
            for family in final[("down", layer)]
        )
        for layer in UD_IQ1_M_DOWN_SPECIAL_LAYERS
    }
    diagnostics = {
        "solver": "scipy.optimize.milp/highs",
        "stage_a_status": int(stage_a.status),
        "stage_a_message": str(stage_a.message),
        "stage_b_status": int(stage_b.status),
        "stage_b_message": str(stage_b.message),
        "uniform_gate_up_high_precision_quota": quota,
        "gate_up_high_precision_counts": quota_counts,
        "down_special_high_precision_counts": down_counts,
        "scenario_baseline_objectives": scenario_baselines,
        "worst_final_to_baseline_ratio": float(
            stage_b.x[worst_loss_index]
        ),
        "routed_limit_bytes": int(routed_limit),
        "exact_routed_bytes": int(exact_routed),
        "exact_overlay_payload_bytes": int(exact_overlay_payload),
        "binary_variables": variable_count - 1,
        "constraints": len(lower),
    }
    return final, diagnostics


def build_ud_plan(
    base_profile_plan: str | Path,
    reap_csv: str | Path,
    sensitivity_map: str | Path,
    ud_analysis: str | Path,
    *,
    target_bytes: int = 60_000_000_000,
    container_reserve_bytes: int = 4_000_000,
    v_multipliers: tuple[float, ...] = (1.0, 4.0, 8.0, 16.0),
) -> dict:
    base, metadata = profiles_from_sensitivity_plan(base_profile_plan)
    marks = load_expert_sensitivity_map(
        sensitivity_map,
        expected_layers=43,
        expected_experts=256,
    )
    energy = _read_reap(reap_csv)
    routed_limit = (
        int(target_bytes)
        - int(container_reserve_bytes)
        - int(metadata["nonexpert_bytes"])
    )
    final, diagnostics = solve_ud_layer_upgrade(
        base,
        energy,
        marks,
        routed_limit=routed_limit,
        v_multipliers=tuple(float(value) for value in v_multipliers),
    )
    final_routed = _profile_routed_bytes(final)
    changed = _changes(base, final)
    change_map = {key: [""] * 256 for key in final}
    for key, entries in changed.items():
        for expert, family in entries.items():
            change_map[key][expert] = family
    change_counts = Counter(
        f"{base[key][expert]}->{family}"
        for key, entries in changed.items()
        for expert, family in entries.items()
    )
    base_profile_hash = hashlib.sha256(
        json.dumps(
            {
                f"{projection}:{layer}": list(families)
                for (projection, layer), families in sorted(base.items())
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "format": "mfq.v4f-ud-layer-upgrade.v1",
        "method": {
            "solver": "two-stage global multiple-choice MILP",
            "stage_a": (
                "maximize the uniform NINT4-or-higher expert quota over "
                "UD IQ1_M gate/up priority layers"
            ),
            "stage_b": (
                "minimize worst REAP-energy-weighted NMSE over v-mark "
                "strength scenarios while fixing the stage-a quota"
            ),
            "monotonic": "every base expert projection is preserved or upgraded",
            "V": "hard-fixed to NINT8 for both projections",
            "v": "may upgrade monotonically through NINT4/NINT5/NINT6/NINT8",
            "w": "may upgrade monotonically to NINT4",
        },
        "ud_constraints": {
            "recipe": "UD DeepSeek-V4 IQ1_M tensor-type map",
            "gate_up_high_layers": list(UD_IQ1_M_GATE_UP_HIGH_LAYERS),
            "gate_up_min_nint4_or_higher_per_layer": diagnostics[
                "uniform_gate_up_high_precision_quota"
            ],
            "down_special_layers": list(UD_IQ1_M_DOWN_SPECIAL_LAYERS),
            "down_special_min_nint4_or_higher_per_layer": 256,
            "analysis": str(Path(ud_analysis).resolve()),
            "analysis_sha256": _sha256(ud_analysis),
        },
        "surrogate": {
            "family_snr_db": DEFAULT_FAMILY_SNR_DB,
            "nepq0s_snr_source": (
                "layer40 held-out imatrix ADMM probe; gate_up and down"
            ),
            "v_multipliers": list(v_multipliers),
            "V_multiplier": 32.0,
            "diagnostics": diagnostics,
        },
        "target_bytes": int(target_bytes),
        "container_reserve_bytes": int(container_reserve_bytes),
        "base_file_bytes": int(metadata["file_bytes"]),
        "nonexpert_bytes": int(metadata["nonexpert_bytes"]),
        "baseline_routed_bytes": int(metadata["routed_bytes"]),
        "routed_bytes": int(final_routed),
        "estimated_blob_bytes": int(metadata["nonexpert_bytes"]) + final_routed,
        "estimated_headroom_bytes": (
            int(target_bytes)
            - int(metadata["nonexpert_bytes"])
            - final_routed
        ),
        "base_family_counts": _family_counts(base),
        "final_family_counts": _family_counts(final),
        "change_counts": dict(sorted(change_counts.items())),
        "changed_projection_experts": sum(map(len, changed.values())),
        "changes": _mapping(change_map),
        "base_profile_plan": str(Path(base_profile_plan).resolve()),
        "base_profile_plan_sha256": str(metadata["base_plan_sha256"]),
        "base_profile_sha256": base_profile_hash,
        "base_allocation_sha256": str(metadata["allocation_sha256"]),
        "source_index_sha256": str(metadata["source_index_sha256"]),
        "reap_csv": str(Path(reap_csv).resolve()),
        "reap_sha256": _sha256(reap_csv),
        "sensitivity_map": str(Path(sensitivity_map).resolve()),
        "sensitivity_map_sha256": _sha256(sensitivity_map),
        "mark_counts": {
            mark: marks.count(mark)
            for mark in ("V", "v", "w")
        },
        "mtp_included": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-profile-plan", required=True)
    parser.add_argument("--reap-csv", required=True)
    parser.add_argument("--sensitivity-map", required=True)
    parser.add_argument("--ud-analysis", required=True)
    parser.add_argument("--target-bytes", type=int, default=60_000_000_000)
    parser.add_argument("--container-reserve-bytes", type=int, default=4_000_000)
    parser.add_argument("--v-multipliers", default="1,4,8,16")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    document = build_ud_plan(
        args.base_profile_plan,
        args.reap_csv,
        args.sensitivity_map,
        args.ud_analysis,
        target_bytes=args.target_bytes,
        container_reserve_bytes=args.container_reserve_bytes,
        v_multipliers=tuple(
            float(value)
            for value in args.v_multipliers.split(",")
            if value.strip()
        ),
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, document)
    print(json.dumps(document, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
