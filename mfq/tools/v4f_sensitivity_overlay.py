"""Plan and materialize compact DeepSeek-V4 expert-precision overlays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mfq.calibration.artifact import ExpertPrecision
from mfq.formats.header import FileHeader
from mfq.formats.io import (
    _NINT_MOE_HDR,
    _NINT_MOE_POOL_V2_HDR,
)
from mfq.formats.nint import NintSpec
from mfq.quantize.expert_sensitivity import load_expert_sensitivity_map
from mfq.quantize.v4f_imatrix import V4FImportanceMatrix
from mfq.quantize.v4f_plan import (
    routed_family_blob_bytes,
    routed_family_pool_bytes,
)
from mfq.quantize.v4f_source import V4FCheckpoint
from mfq.quantize.v4f_upgrade import (
    _profile_routed_bytes,
    _read_reap,
    _read_v4f_mfq_profiles,
)
from mfq.tools.quantize_hf_to_mfq import (
    _ExpertPoolRowSource,
    _mixed_moe_blob_nbytes,
    _write_flat_family_axis0_blob,
    _write_nint_axis0_blob,
)
from mfq.tools.upgrade_v4f_mfq import _write_header


SPECS = {
    "NINT4": NintSpec(4, 24, 6),
    "NINT5": NintSpec(5, 28, 7),
    "NINT6": NintSpec(6, 24, 7),
    "NINT8": NintSpec(8, 48, 7),
}
PROJECTIONS = ("gate_up", "down")
DELTA_MAGIC = b"NID2"
ROBUST_FAMILIES = ("NVQ2J", "NINT4", "NINT5", "NINT6", "NINT8")
V4F_PROJECTION_WEIGHTS_PER_EXPERT = {
    "gate_up": 2 * 2048 * 4096,
    "down": 4096 * 2048,
}


@dataclass(frozen=True)
class LayerBpwConstraint:
    projection: str
    high_layers: tuple[int, ...]
    normal_dtype: str
    high_dtype: str
    minimum_delta_bpw: float


UD_IQ1_M_LAYER_CONSTRAINTS = (
    LayerBpwConstraint(
        projection="gate_up",
        high_layers=(
            4,
            5,
            6,
            8,
            9,
            20,
            21,
            22,
            24,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            40,
            42,
        ),
        normal_dtype="IQ1_M",
        high_dtype="IQ2_XXS",
        minimum_delta_bpw=2.0625 - 1.75,
    ),
    LayerBpwConstraint(
        projection="down",
        high_layers=(26, 42),
        normal_dtype="IQ3_XXS",
        high_dtype="MXFP4",
        minimum_delta_bpw=4.25 - 3.0625,
    ),
)
DEFAULT_FAMILY_SNR_DB = {
    "gate_up": {
        "NVQ2J": 10.59587019259313,
        "NINT4": 25.21482056618251,
        "NINT5": 29.26,
        "NINT6": 35.06,
        "NINT8": 43.65,
    },
    "down": {
        "NVQ2J": 11.060234763932346,
        "NINT4": 25.223398406907663,
        "NINT5": 29.26,
        "NINT6": 35.06,
        "NINT8": 43.65,
    },
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _mapping(
    values: dict[tuple[str, int], list[str] | tuple[str, ...]],
) -> dict[str, dict[str, dict[str, str]]]:
    result = {projection: {} for projection in PROJECTIONS}
    for (projection, layer), families in sorted(values.items()):
        selected = {
            str(expert): family
            for expert, family in enumerate(families)
            if family
        }
        if selected:
            result[projection][str(layer)] = selected
    return result


def _changes(
    base: dict[tuple[str, int], tuple[str, ...]],
    final: dict[tuple[str, int], list[str] | tuple[str, ...]],
) -> dict[tuple[str, int], dict[int, str]]:
    result = {}
    for key, families in final.items():
        changed = {
            expert: family
            for expert, family in enumerate(families)
            if family != base[key][expert]
        }
        if changed:
            result[key] = changed
    return result


def _family_counts(
    values: dict[tuple[str, int], list[str] | tuple[str, ...]],
) -> dict[str, int]:
    counts = Counter(
        f"{projection}:{family}"
        for (projection, _layer), families in values.items()
        for family in families
    )
    return dict(sorted(counts.items()))


def _demote_tail_to_budget(
    current: dict[tuple[str, int], list[str]],
    marks,
    energy: dict[tuple[int, int], dict[str, float]],
    routed_limit: int,
) -> list[dict[str, int | float | str]]:
    routed = _profile_routed_bytes(current)
    demoted = []
    while routed > routed_limit:
        best = None
        for (projection, layer), families in current.items():
            before_counts = Counter(families)
            before = routed_family_blob_bytes(
                projection,
                dict(before_counts),
            )
            for expert, family in enumerate(families):
                if family != "NINT4" or marks.mark(layer, expert) != "w":
                    continue
                after_counts = dict(before_counts)
                after_counts["NINT4"] -= 1
                after_counts["NVQ2J"] = after_counts.get("NVQ2J", 0) + 1
                saving = before - routed_family_blob_bytes(
                    projection,
                    after_counts,
                )
                value = energy[(layer, expert)][projection]
                rank = (
                    value / saving,
                    value,
                    -saving,
                    projection,
                    layer,
                    expert,
                )
                if best is None or rank < best[0]:
                    best = (
                        rank,
                        projection,
                        layer,
                        expert,
                        saving,
                        value,
                    )
        if best is None:
            raise ValueError(
                "88G budget cannot be met without demoting a v/V expert"
            )
        _, projection, layer, expert, saving, value = best
        current[(projection, layer)][expert] = "NVQ2J"
        routed -= saving
        demoted.append(
            {
                "projection": projection,
                "layer": layer,
                "expert": expert,
                "saved_bytes": saving,
                "reap_energy": value,
            }
        )
    if routed != _profile_routed_bytes(current):
        raise RuntimeError("incremental routed accounting mismatch")
    return demoted


def build_simple_plan(
    base_mfq: str | Path,
    reap_csv: str | Path,
    sensitivity_map: str | Path,
    *,
    target_bytes: int | None = None,
    container_reserve_bytes: int = 4_000_000,
) -> dict:
    base, metadata = _read_v4f_mfq_profiles(base_mfq)
    marks = load_expert_sensitivity_map(
        sensitivity_map,
        expected_layers=43,
        expected_experts=256,
    )
    energy = _read_reap(reap_csv)
    resolved_target = (
        int(metadata["target_bytes"])
        if target_bytes is None
        else int(target_bytes)
    )
    current = {key: list(families) for key, families in base.items()}

    intersections = []
    for (projection, layer), families in base.items():
        for expert, family in enumerate(families):
            if family != "NINT4" or marks.mark(layer, expert) != "v":
                continue
            intersections.append(
                (
                    energy[(layer, expert)][projection],
                    projection,
                    layer,
                    expert,
                )
            )
    intersections.sort(
        key=lambda item: (-item[0], item[1], item[2], item[3])
    )
    first = (len(intersections) + 2) // 3
    second = (2 * len(intersections) + 2) // 3
    for index, (_value, projection, layer, expert) in enumerate(intersections):
        if index < first:
            current[(projection, layer)][expert] = "NINT6"
        elif index < second:
            current[(projection, layer)][expert] = "NINT5"

    for (projection, layer), families in current.items():
        for expert in marks.experts(layer, "V"):
            families[expert] = "NINT8"

    protected_routed = _profile_routed_bytes(current)
    routed_limit = (
        resolved_target
        - int(container_reserve_bytes)
        - int(metadata["nonexpert_bytes"])
    )
    demoted = _demote_tail_to_budget(
        current,
        marks,
        energy,
        routed_limit,
    )
    final_routed = _profile_routed_bytes(current)
    changed = _changes(base, current)
    change_map = {
        key: [""] * 256
        for key in current
    }
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
        "format": "mfq.v4f-sensitivity-simple.v1",
        "method": {
            "V": "NINT8 for both projections",
            "v_reap_intersection": (
                "existing REAP-selected NINT4 entries ranked by REAP energy; "
                "top/middle/bottom thirds use NINT6/NINT5/NINT4"
            ),
            "budget_recovery": (
                "demote only w-marked NINT4 entries in ascending "
                "REAP-energy-per-saved-byte order"
            ),
        },
        "target_bytes": resolved_target,
        "container_reserve_bytes": int(container_reserve_bytes),
        "base_file_bytes": int(metadata["file_bytes"]),
        "nonexpert_bytes": int(metadata["nonexpert_bytes"]),
        "baseline_routed_bytes": int(metadata["routed_bytes"]),
        "pre_rebalance_routed_bytes": protected_routed,
        "routed_bytes": final_routed,
        "estimated_blob_bytes": int(metadata["nonexpert_bytes"]) + final_routed,
        "estimated_headroom_bytes": (
            resolved_target
            - int(metadata["nonexpert_bytes"])
            - final_routed
        ),
        "base_family_counts": _family_counts(base),
        "final_family_counts": _family_counts(current),
        "change_counts": dict(sorted(change_counts.items())),
        "changed_projection_experts": sum(map(len, changed.values())),
        "demoted_count": len(demoted),
        "demoted": demoted,
        "changes": _mapping(change_map),
        "base_mfq": str(Path(base_mfq).resolve()),
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


def _pool_affine_bytes(projection: str, family: str) -> tuple[int, int]:
    one = routed_family_pool_bytes(projection, family, 1)
    two = routed_family_pool_bytes(projection, family, 2)
    per_expert = two - one
    fixed = one - per_expert
    if fixed < 0 or per_expert <= 0:
        raise RuntimeError(
            f"invalid routed pool byte model for {projection}:{family}"
        )
    for count in (1, 2, 3, 127, 256):
        exact = routed_family_pool_bytes(projection, family, count)
        if exact != fixed + per_expert * count:
            raise RuntimeError(
                f"non-affine routed pool bytes for "
                f"{projection}:{family}:{count}"
            )
    return fixed, per_expert


def _family_effective_bpw(projection: str, family: str) -> float:
    if projection not in V4F_PROJECTION_WEIGHTS_PER_EXPERT:
        raise ValueError(f"unsupported V4F projection: {projection}")
    _fixed, per_expert = _pool_affine_bytes(projection, family)
    return (
        per_expert
        * 8.0
        / V4F_PROJECTION_WEIGHTS_PER_EXPERT[projection]
    )


def _layer_effective_bpw(
    families: dict[tuple[str, int], list[str] | tuple[str, ...]],
    projection: str,
    layer: int,
) -> float:
    values = families[(projection, layer)]
    return sum(
        _family_effective_bpw(projection, family)
        for family in values
    ) / len(values)


def _layer_constraint_diagnostics(
    families: dict[tuple[str, int], list[str] | tuple[str, ...]],
    constraints: tuple[LayerBpwConstraint, ...],
) -> list[dict]:
    result = []
    for constraint in constraints:
        projection_layers = sorted(
            layer
            for projection, layer in families
            if projection == constraint.projection
        )
        high_layers = set(constraint.high_layers)
        normal_layers = [
            layer
            for layer in projection_layers
            if layer not in high_layers
        ]
        if not normal_layers:
            raise ValueError(
                f"{constraint.projection} layer policy has no normal layers"
            )
        normal_mean = sum(
            _layer_effective_bpw(
                families,
                constraint.projection,
                layer,
            )
            for layer in normal_layers
        ) / len(normal_layers)
        high_values = {
            str(layer): _layer_effective_bpw(
                families,
                constraint.projection,
                layer,
            )
            for layer in constraint.high_layers
        }
        margins = {
            layer: value - normal_mean
            for layer, value in high_values.items()
        }
        result.append(
            {
                "projection": constraint.projection,
                "normal_dtype": constraint.normal_dtype,
                "high_dtype": constraint.high_dtype,
                "minimum_delta_bpw": constraint.minimum_delta_bpw,
                "normal_layer_mean_bpw": normal_mean,
                "high_layer_bpw": high_values,
                "high_layer_delta_bpw": margins,
                "minimum_observed_delta_bpw": min(margins.values()),
            }
        )
    return result


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


def _robust_allowed_families(mark: str) -> tuple[str, ...]:
    if mark == "V":
        return ("NINT8",)
    if mark == "v":
        return ROBUST_FAMILIES
    if mark == "w":
        return ("NVQ2J", "NINT4")
    raise ValueError(f"unsupported expert sensitivity mark: {mark}")


def _solve_robust_profiles(
    base: dict[tuple[str, int], tuple[str, ...]],
    energy: dict[tuple[int, int], dict[str, float]],
    marks,
    *,
    routed_limit: int,
    family_snr_db: dict[str, dict[str, float]],
    v_multipliers: tuple[float, ...],
    overlay_limit_bytes: int | None = None,
    layer_bpw_constraints: tuple[LayerBpwConstraint, ...] = (),
    solver_time_limit_seconds: float | None = None,
    solver_mip_rel_gap: float | None = None,
) -> tuple[dict[tuple[str, int], list[str]], dict]:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if not v_multipliers or any(value <= 0 for value in v_multipliers):
        raise ValueError("v sensitivity multipliers must be positive")
    actual = {
        family
        for families in base.values()
        for family in families
    }
    if not actual <= set(ROBUST_FAMILIES):
        raise ValueError(
            "robust sensitivity allocation only supports "
            f"{ROBUST_FAMILIES}; found {sorted(actual)}"
        )
    nmse = _family_nmse(family_snr_db)
    for projection in PROJECTIONS:
        if set(nmse.get(projection, {})) != set(ROBUST_FAMILIES):
            raise ValueError(
                f"incomplete family SNR table for {projection}"
            )
    for constraint in layer_bpw_constraints:
        if constraint.projection not in PROJECTIONS:
            raise ValueError(
                f"unsupported layer-constraint projection: "
                f"{constraint.projection}"
            )
        if (
            not constraint.high_layers
            or constraint.minimum_delta_bpw <= 0
        ):
            raise ValueError("invalid layer bpw constraint")
    if len({item.projection for item in layer_bpw_constraints}) != len(
        layer_bpw_constraints
    ):
        raise ValueError("only one layer bpw constraint per projection is supported")
    if (
        solver_time_limit_seconds is not None
        and solver_time_limit_seconds <= 0
    ):
        raise ValueError("solver time limit must be positive")
    if (
        solver_mip_rel_gap is not None
        and not 0 <= solver_mip_rel_gap < 1
    ):
        raise ValueError("solver MIP relative gap must be in [0, 1)")

    groups = [
        (projection, layer, expert)
        for projection, layer in sorted(base)
        for expert in range(256)
    ]
    x_meta: list[tuple[str, int, int, str]] = []
    group_x: list[list[int]] = []
    for projection, layer, expert in groups:
        indices = []
        mark = marks.mark(layer, expert)
        for family in _robust_allowed_families(mark):
            indices.append(len(x_meta))
            x_meta.append((projection, layer, expert, family))
        group_x.append(indices)
    x_count = len(x_meta)
    x_by_projection_layer: dict[tuple[str, int], list[int]] = {
        key: []
        for key in base
    }
    for index, (projection, layer, _expert, _family) in enumerate(x_meta):
        x_by_projection_layer[(projection, layer)].append(index)

    pool_keys = [
        (projection, layer, family)
        for projection, layer in sorted(base)
        for family in ROBUST_FAMILIES
    ]
    pool_index = {
        key: x_count + index
        for index, key in enumerate(pool_keys)
    }
    z_index = x_count + len(pool_keys)
    variable_count = z_index + 1
    scenario_names = [
        f"v_multiplier_{value:g}"
        for value in v_multipliers
    ]
    scenario_baselines = []
    for multiplier in v_multipliers:
        total = 0.0
        for projection, layer, expert in groups:
            mark = marks.mark(layer, expert)
            sensitivity = (
                32.0 if mark == "V"
                else multiplier if mark == "v"
                else 1.0
            )
            total += (
                energy[(layer, expert)][projection]
                * sensitivity
                * nmse[projection][base[(projection, layer)][expert]]
            )
        if not np.isfinite(total) or total <= 0:
            raise ValueError("invalid robust baseline objective")
        scenario_baselines.append(total)

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

    fixed_headers = len(base) * _NINT_MOE_HDR.size
    variable_budget = int(routed_limit) - fixed_headers
    if variable_budget <= 0:
        raise ValueError("routed byte limit cannot hold NINTM headers")
    pool_costs = {
        (projection, family): _pool_affine_bytes(projection, family)
        for projection in PROJECTIONS
        for family in ROBUST_FAMILIES
    }
    budget_entries = []
    for index, (projection, _layer, _expert, family) in enumerate(x_meta):
        budget_entries.append(
            (
                index,
                pool_costs[(projection, family)][1] / variable_budget,
            )
        )
    for key, index in pool_index.items():
        projection, _layer, family = key
        budget_entries.append(
            (
                index,
                pool_costs[(projection, family)][0] / variable_budget,
            )
        )
    budget_row = len(lower)
    add_row(budget_entries, -np.inf, 1.0)

    overlay_row = None
    if overlay_limit_bytes is not None:
        overlay_limit_bytes = int(overlay_limit_bytes)
        overlay_guard = 1_000_000
        overlay_variable_limit = overlay_limit_bytes - overlay_guard
        if overlay_variable_limit <= 0:
            raise ValueError("overlay byte limit is too small")
        overlay_entries = []
        for index, (projection, layer, expert, family) in enumerate(x_meta):
            if family == base[(projection, layer)][expert]:
                continue
            conservative = (
                _NINT_MOE_HDR.size
                + routed_family_pool_bytes(projection, family, 1)
            )
            overlay_entries.append(
                (index, conservative / overlay_variable_limit)
            )
        overlay_row = len(lower)
        add_row(overlay_entries, -np.inf, 1.0)

    layer_constraint_rows = []
    for constraint in layer_bpw_constraints:
        projection_layers = sorted(
            layer
            for projection, layer in base
            if projection == constraint.projection
        )
        high_layers = set(constraint.high_layers)
        missing = high_layers - set(projection_layers)
        if missing:
            raise ValueError(
                f"{constraint.projection} layer policy references missing "
                f"layers: {sorted(missing)}"
            )
        normal_layers = [
            layer
            for layer in projection_layers
            if layer not in high_layers
        ]
        if not normal_layers:
            raise ValueError(
                f"{constraint.projection} layer policy has no normal layers"
            )
        high_scale = 1.0 / 256.0
        baseline_normal_mean = sum(
            _layer_effective_bpw(
                base,
                constraint.projection,
                layer,
            )
            for layer in normal_layers
        ) / len(normal_layers)
        minimum_high_bpw = (
            baseline_normal_mean + constraint.minimum_delta_bpw
        )
        for high_layer in constraint.high_layers:
            entries = []
            for index in x_by_projection_layer[
                (constraint.projection, high_layer)
            ]:
                family = x_meta[index][3]
                entries.append(
                    (
                        index,
                        high_scale
                        * _family_effective_bpw(
                            constraint.projection,
                            family,
                        ),
                    )
                )
            row = len(lower)
            add_row(
                entries,
                minimum_high_bpw,
                np.inf,
            )
            layer_constraint_rows.append(
                {
                    "row": row,
                    "projection": constraint.projection,
                    "layer": high_layer,
                    "baseline_normal_mean_bpw": baseline_normal_mean,
                    "minimum_high_layer_bpw": minimum_high_bpw,
                    "minimum_delta_bpw": (
                        constraint.minimum_delta_bpw
                    ),
                }
            )

    for scenario_index, multiplier in enumerate(v_multipliers):
        baseline = scenario_baselines[scenario_index]
        entries = [(z_index, -1.0)]
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
            entries.append((index, loss / baseline))
        add_row(entries, -np.inf, 0.0)

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(len(lower), variable_count),
    ).tocsr()
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[z_index] = 1.0
    for index, (projection, _layer, _expert, family) in enumerate(x_meta):
        objective[index] = (
            1e-10
            * pool_costs[(projection, family)][1]
            / variable_budget
        )
    for key, index in pool_index.items():
        projection, _layer, family = key
        objective[index] = (
            1e-10
            * pool_costs[(projection, family)][0]
            / variable_budget
        )
    integrality = np.ones(variable_count, dtype=np.uint8)
    integrality[z_index] = 0
    bounds_lower = np.zeros(variable_count, dtype=np.float64)
    bounds_upper = np.ones(variable_count, dtype=np.float64)
    bounds_upper[z_index] = np.inf
    constraint_lower = np.asarray(lower, dtype=np.float64)
    constraint_upper = np.asarray(upper, dtype=np.float64)

    result = None
    final: dict[tuple[str, int], list[str]] = {}
    exact_routed = 0
    budget_upper = 1.0
    for _attempt in range(4):
        solve_upper = constraint_upper.copy()
        solve_upper[budget_row] = budget_upper
        solve_options: dict[str, float | bool] = {"presolve": True}
        if solver_time_limit_seconds is not None:
            solve_options["time_limit"] = float(
                solver_time_limit_seconds
            )
        if solver_mip_rel_gap is not None:
            solve_options["mip_rel_gap"] = float(solver_mip_rel_gap)
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(bounds_lower, bounds_upper),
            constraints=LinearConstraint(
                matrix,
                constraint_lower,
                solve_upper,
            ),
            options=solve_options,
        )
        if result.x is None or int(result.status) not in (0, 1):
            raise RuntimeError(
                f"robust V4F precision allocation failed: {result.message}"
            )
        solution = np.asarray(result.x, dtype=np.float64)
        binary_count = x_count + len(pool_keys)
        binary_error = float(
            np.max(
                np.abs(
                    solution[:binary_count]
                    - np.rint(solution[:binary_count])
                )
            )
        )
        activity = np.asarray(matrix @ solution).reshape(-1)
        lower_violation = float(
            np.max(
                np.maximum(
                    constraint_lower - activity,
                    0.0,
                )
            )
        )
        upper_violation = float(
            np.max(
                np.maximum(
                    activity - solve_upper,
                    0.0,
                )
            )
        )
        if (
            binary_error > 1e-5
            or lower_violation > 1e-6
            or upper_violation > 1e-6
        ):
            raise RuntimeError(
                "robust MILP returned an infeasible incumbent: "
                f"binary_error={binary_error:.3g}, "
                f"lower_violation={lower_violation:.3g}, "
                f"upper_violation={upper_violation:.3g}"
            )
        final = {
            key: [""] * 256
            for key in base
        }
        for group, indices in zip(groups, group_x, strict=True):
            chosen = max(indices, key=lambda index: result.x[index])
            if result.x[chosen] < 0.5:
                raise RuntimeError("robust MILP returned a fractional group")
            projection, layer, expert = group
            final[(projection, layer)][expert] = x_meta[chosen][3]
        exact_routed = _profile_routed_bytes(final)
        if exact_routed <= routed_limit:
            break
        overage = exact_routed - routed_limit
        guard = max(1_000_000, int(routed_limit * 1e-7))
        budget_upper -= (overage + guard) / variable_budget
    else:
        raise RuntimeError(
            "robust V4F allocation exceeds the exact byte budget"
        )
    assert result is not None

    final_objectives = []
    for multiplier in v_multipliers:
        total = 0.0
        for projection, layer, expert in groups:
            mark = marks.mark(layer, expert)
            sensitivity = (
                32.0 if mark == "V"
                else multiplier if mark == "v"
                else 1.0
            )
            total += (
                energy[(layer, expert)][projection]
                * sensitivity
                * nmse[projection][final[(projection, layer)][expert]]
            )
        final_objectives.append(total)
    ratios = [
        final_value / baseline
        for final_value, baseline in zip(
            final_objectives,
            scenario_baselines,
            strict=True,
        )
    ]
    changed = _changes(base, final)
    exact_overlay_payload = sum(
        _NINT_MOE_HDR.size
        + sum(
            routed_family_pool_bytes(
                projection,
                family,
                count,
            )
            for family, count in Counter(entries.values()).items()
        )
        for (projection, _layer), entries in changed.items()
    )
    conservative_overlay = sum(
        _NINT_MOE_HDR.size
        + routed_family_pool_bytes(projection, family, 1)
        for (projection, layer), entries in changed.items()
        for expert, family in entries.items()
    )
    layer_diagnostics = _layer_constraint_diagnostics(
        final,
        layer_bpw_constraints,
    )
    floor_by_projection = {
        entry["projection"]: entry
        for entry in layer_constraint_rows
    }
    for entry in layer_diagnostics:
        floor = floor_by_projection[entry["projection"]]
        entry["baseline_normal_mean_bpw"] = floor[
            "baseline_normal_mean_bpw"
        ]
        entry["minimum_high_layer_bpw"] = floor[
            "minimum_high_layer_bpw"
        ]
        entry["minimum_observed_high_layer_bpw"] = min(
            entry["high_layer_bpw"].values()
        )
        entry["minimum_floor_margin_bpw"] = (
            entry["minimum_observed_high_layer_bpw"]
            - entry["minimum_high_layer_bpw"]
        )
        if entry["minimum_floor_margin_bpw"] < -1e-6:
            raise RuntimeError(
                f"{entry['projection']} fixed layer bpw floor was violated"
            )
        if (
            entry["minimum_observed_delta_bpw"] + 1e-6
            < entry["minimum_delta_bpw"]
        ):
            raise RuntimeError(
                f"{entry['projection']} layer bpw constraint was violated"
            )
    diagnostics = {
        "solver": "scipy.optimize.milp/highs",
        "status": int(result.status),
        "message": str(result.message),
        "mip_gap": (
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        "mip_node_count": (
            int(result.mip_node_count)
            if getattr(result, "mip_node_count", None) is not None
            else None
        ),
        "binary_variables": x_count + len(pool_keys),
        "continuous_layer_mean_variables": 0,
        "constraints": len(lower),
        "scenario_names": scenario_names,
        "baseline_objectives": scenario_baselines,
        "final_objectives": final_objectives,
        "final_to_baseline_ratios": ratios,
        "worst_final_to_baseline_ratio": max(ratios),
        "routed_limit_bytes": int(routed_limit),
        "exact_routed_bytes": int(exact_routed),
        "overlay_limit_bytes": overlay_limit_bytes,
        "exact_overlay_payload_bytes": int(exact_overlay_payload),
        "conservative_overlay_bytes": int(conservative_overlay),
        "overlay_constraint_row": overlay_row,
        "layer_constraint_rows": layer_constraint_rows,
        "layer_bpw": layer_diagnostics,
        "solver_time_limit_seconds": solver_time_limit_seconds,
        "solver_mip_rel_gap": solver_mip_rel_gap,
    }
    return final, diagnostics


def build_robust_plan(
    base_mfq: str | Path,
    reap_csv: str | Path,
    sensitivity_map: str | Path,
    *,
    target_bytes: int | None = None,
    container_reserve_bytes: int = 4_000_000,
    v_multipliers: tuple[float, ...] = (1.0, 4.0, 8.0, 16.0),
    family_snr_db: dict[str, dict[str, float]] | None = None,
    overlay_limit_bytes: int | None = None,
    layer_bpw_constraints: tuple[LayerBpwConstraint, ...] = (),
    layer_policy_source: str | Path | None = None,
    solver_time_limit_seconds: float | None = None,
    solver_mip_rel_gap: float | None = None,
) -> dict:
    base, metadata = _read_v4f_mfq_profiles(base_mfq)
    marks = load_expert_sensitivity_map(
        sensitivity_map,
        expected_layers=43,
        expected_experts=256,
    )
    energy = _read_reap(reap_csv)
    resolved_target = (
        int(metadata["target_bytes"])
        if target_bytes is None
        else int(target_bytes)
    )
    routed_limit = (
        resolved_target
        - int(container_reserve_bytes)
        - int(metadata["nonexpert_bytes"])
    )
    snr = (
        DEFAULT_FAMILY_SNR_DB
        if family_snr_db is None
        else family_snr_db
    )
    current, diagnostics = _solve_robust_profiles(
        base,
        energy,
        marks,
        routed_limit=routed_limit,
        family_snr_db=snr,
        v_multipliers=tuple(float(value) for value in v_multipliers),
        overlay_limit_bytes=overlay_limit_bytes,
        layer_bpw_constraints=layer_bpw_constraints,
        solver_time_limit_seconds=solver_time_limit_seconds,
        solver_mip_rel_gap=solver_mip_rel_gap,
    )
    final_routed = _profile_routed_bytes(current)
    changed = _changes(base, current)
    change_map = {
        key: [""] * 256
        for key in current
    }
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
    document = {
        "format": (
            "mfq.v4f-ud-layer-robust.v1"
            if layer_bpw_constraints
            else "mfq.v4f-sensitivity-robust.v1"
        ),
        "method": {
            "solver": "global multiple-choice MILP with exact NINTM bytes",
            "objective": (
                "minimize worst final/base REAP-energy-weighted NMSE "
                "over sensitivity-strength scenarios, subject to exact "
                "UD high-layer bpw floors relative to the fixed base "
                "normal-layer mean"
                if layer_bpw_constraints
                else "minimize worst final/base REAP-energy-weighted NMSE "
                "over sensitivity-strength scenarios"
            ),
            "V": "hard-fixed to NINT8 for both projections",
            "v": "NVQ2J/NINT4/NINT5/NINT6/NINT8",
            "w": "NVQ2J/NINT4",
        },
        "surrogate": {
            "family_snr_db": snr,
            "v_multipliers": list(v_multipliers),
            "V_multiplier": 32.0,
            "diagnostics": diagnostics,
        },
        "target_bytes": resolved_target,
        "container_reserve_bytes": int(container_reserve_bytes),
        "base_file_bytes": int(metadata["file_bytes"]),
        "nonexpert_bytes": int(metadata["nonexpert_bytes"]),
        "baseline_routed_bytes": int(metadata["routed_bytes"]),
        "routed_bytes": final_routed,
        "estimated_blob_bytes": int(metadata["nonexpert_bytes"]) + final_routed,
        "estimated_headroom_bytes": (
            resolved_target
            - int(metadata["nonexpert_bytes"])
            - final_routed
        ),
        "base_family_counts": _family_counts(base),
        "final_family_counts": _family_counts(current),
        "change_counts": dict(sorted(change_counts.items())),
        "changed_projection_experts": sum(map(len, changed.values())),
        "changes": _mapping(change_map),
        "base_mfq": str(Path(base_mfq).resolve()),
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
    if layer_bpw_constraints:
        document["ud_layer_policy"] = {
            "source": (
                str(Path(layer_policy_source).resolve())
                if layer_policy_source is not None
                else None
            ),
            "source_sha256": (
                _sha256(layer_policy_source)
                if layer_policy_source is not None
                else None
            ),
            "constraints": [
                {
                    "projection": constraint.projection,
                    "high_layers": list(constraint.high_layers),
                    "normal_dtype": constraint.normal_dtype,
                    "high_dtype": constraint.high_dtype,
                    "minimum_delta_bpw": (
                        constraint.minimum_delta_bpw
                    ),
                    "reference": "base normal-layer mean bpw",
                }
                for constraint in layer_bpw_constraints
            ],
        }
    return document


def _plan_changes(plan: dict) -> list[tuple[str, int, dict[int, str]]]:
    result = []
    raw = plan["changes"]
    for projection in PROJECTIONS:
        for layer, entries in raw.get(projection, {}).items():
            result.append(
                (
                    projection,
                    int(layer),
                    {
                        int(expert): str(family)
                        for expert, family in entries.items()
                    },
                )
            )
    return sorted(result, key=lambda item: (item[1], item[0]))


def _nvq2j_precision(artifact: Path) -> ExpertPrecision:
    return ExpertPrecision(
        family="NVQ2J",
        artifact=str(artifact),
        options=(
            ("banks", 4),
            ("assignment_refine_steps", 2),
            ("search_steps", 19),
            ("group_chunk", 1024),
        ),
    )


def _precision(family: str, artifact: Path | None) -> ExpertPrecision:
    if family in SPECS:
        return ExpertPrecision(family=family, nint_spec=SPECS[family])
    if family == "NVQ2J" and artifact is not None:
        return _nvq2j_precision(artifact)
    raise ValueError(f"unsupported overlay family: {family}")


def _artifact_name(layer: int, projection: str) -> str:
    return f"layer{layer:02d}-{projection}-nvq2j.npz"


def _delta_expected_bytes(
    projection: str,
    entries: dict[int, str],
    artifact: Path,
) -> int:
    precisions = tuple(
        _precision(entries[expert], artifact)
        for expert in sorted(entries)
    )
    columns = 4096 if projection == "gate_up" else 2048
    return _mixed_moe_blob_nbytes(
        (len(entries), 4096, columns),
        precisions,
        None,
    )


def _write_delta_blob(
    checkpoint: V4FCheckpoint,
    imatrix: V4FImportanceMatrix,
    *,
    projection: str,
    layer: int,
    entries: dict[int, str],
    artifact: Path,
    output: Path,
    row_chunk: int,
    device: str,
) -> int:
    source = checkpoint.expert_source(layer, projection)
    columns = 4096 if projection == "gate_up" else 2048
    importance = imatrix.expert(layer, projection).values
    cohorts: dict[str, list[int]] = {}
    for expert, family in sorted(entries.items()):
        cohorts.setdefault(family, []).append(expert)
    pool_paths = []
    try:
        with output.open("wb") as handle:
            handle.write(
                _NINT_MOE_HDR.pack(
                    DELTA_MAGIC,
                    256,
                    4096,
                    columns,
                    len(cohorts),
                )
            )
            for pool_index, (family, experts) in enumerate(cohorts.items()):
                pool_source = _ExpertPoolRowSource(
                    source,
                    source.shape,
                    source.shape,
                    tuple(experts),
                )
                pool_path = output.with_name(
                    f"{output.name}.pool{pool_index}.tmp"
                )
                pool_paths.append(pool_path)
                if family in SPECS:
                    nbytes = _write_nint_axis0_blob(
                        pool_source,
                        (len(experts) * 4096, columns),
                        SPECS[family],
                        pool_path,
                        row_chunk,
                        "cuda",
                        device,
                    )
                else:
                    precision = _nvq2j_precision(artifact)
                    selected_importance = np.ascontiguousarray(
                        np.asarray(importance)[np.asarray(experts)]
                    )
                    nbytes = _write_flat_family_axis0_blob(
                        pool_source,
                        (len(experts) * 4096, columns),
                        precision,
                        pool_path,
                        row_chunk,
                        "cuda",
                        device,
                        None,
                        importance=selected_importance,
                        importance_rows_per_entry=4096,
                    )
                dtype = family.encode("ascii")
                handle.write(
                    _NINT_MOE_POOL_V2_HDR.pack(
                        len(experts),
                        len(dtype),
                        nbytes,
                        0,
                    )
                )
                handle.write(
                    np.asarray(experts, dtype=np.int32).tobytes()
                )
                handle.write(dtype)
                with pool_path.open("rb") as source_file:
                    shutil.copyfileobj(
                        source_file,
                        handle,
                        length=32 * 1024 * 1024,
                    )
                pool_path.unlink()
        return output.stat().st_size
    finally:
        for pool_path in pool_paths:
            pool_path.unlink(missing_ok=True)


def materialize_overlay(args) -> None:
    import torch

    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("format") not in {
        "mfq.v4f-sensitivity-simple.v1",
        "mfq.v4f-sensitivity-robust.v1",
        "mfq.v4f-ud-layer-robust.v1",
        "mfq.v4f-ud-layer-upgrade.v1",
    }:
        raise ValueError("unsupported sensitivity overlay plan")
    plan_sha = _sha256(plan_path)
    source_root = Path(args.input).resolve()
    source_index = source_root / "model.safetensors.index.json"
    if _sha256(source_index) != plan["source_index_sha256"]:
        raise ValueError("source checkpoint differs from the base MFQ")
    artifact_root = Path(args.artifact_dir).resolve()
    changes = _plan_changes(plan)
    records = []
    for projection, layer, entries in changes:
        artifact = artifact_root / _artifact_name(layer, projection)
        if "NVQ2J" in entries.values() and not artifact.is_file():
            raise FileNotFoundError(f"missing NVQ2J artifact: {artifact}")
        name = f"blk.{layer}.ffn_{projection}_exps.weight"
        records.append(
            (
                name,
                "NINTMD",
                _delta_expected_bytes(projection, entries, artifact),
                projection,
                layer,
                entries,
                artifact,
            )
        )
    output = Path(args.output).resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    state_path = output.with_suffix(output.suffix + ".state.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    header = FileHeader(
        version=2,
        model_arch="deepseek_v4-ew-overlay",
        num_tensors=len(records),
        extra={
            "base_allocation_sha256": plan["base_allocation_sha256"],
            "base_profile_sha256": plan["base_profile_sha256"],
            "plan_sha256": plan_sha,
            "source_index_sha256": plan["source_index_sha256"],
        },
    )
    table = [(name, dtype, nbytes) for name, dtype, nbytes, *_ in records]
    completed = []
    if partial.exists():
        if not state_path.is_file():
            raise ValueError("overlay partial exists without resume state")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("plan_sha256") != plan_sha:
            raise ValueError("overlay partial belongs to another plan")
        completed = list(state.get("completed", []))
        expected_size = int(state["file_bytes"])
        if partial.stat().st_size != expected_size:
            raise ValueError("overlay partial size differs from resume state")
    else:
        with partial.open("wb") as handle:
            _write_header(handle, header, table)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_json(
            state_path,
            {
                "plan_sha256": plan_sha,
                "completed": [],
                "file_bytes": partial.stat().st_size,
            },
        )

    checkpoint = V4FCheckpoint(source_root)
    imatrix = V4FImportanceMatrix.load(args.imatrix)
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        name, _dtype, expected, projection, layer, entries, artifact = record
        if name in completed:
            continue
        temporary = output.parent / f".{output.name}.{layer}.{projection}.tmp"
        if temporary.exists():
            temporary.unlink()
        if str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        item_started = time.perf_counter()
        actual = _write_delta_blob(
            checkpoint,
            imatrix,
            projection=projection,
            layer=layer,
            entries=entries,
            artifact=artifact,
            output=temporary,
            row_chunk=args.row_chunk,
            device=args.device,
        )
        if actual != expected:
            raise RuntimeError(
                f"overlay size mismatch for {name}: {actual} != {expected}"
            )
        with partial.open("ab") as destination, temporary.open("rb") as source:
            shutil.copyfileobj(source, destination, length=32 * 1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.unlink()
        completed.append(name)
        _atomic_json(
            state_path,
            {
                "plan_sha256": plan_sha,
                "completed": completed,
                "file_bytes": partial.stat().st_size,
            },
        )
        elapsed = time.perf_counter() - item_started
        print(
            json.dumps(
                {
                    "completed": index,
                    "total": len(records),
                    "name": name,
                    "changed_experts": len(entries),
                    "mb": actual / 1e6,
                    "seconds": elapsed,
                    "peak_vram_mib": (
                        torch.cuda.max_memory_reserved(args.device)
                        / (1024 * 1024)
                        if str(args.device).startswith("cuda")
                        else 0.0
                    ),
                    "eta_seconds": (
                        (time.perf_counter() - started)
                        / max(1, len(completed))
                        * (len(records) - len(completed))
                    ),
                }
            ),
            flush=True,
        )
    os.replace(partial, output)
    state_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "sha256": _sha256(output),
                "records": len(records),
                "seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


def command_plan(args) -> None:
    document = build_simple_plan(
        args.base_mfq,
        args.reap_csv,
        args.sensitivity_map,
        target_bytes=args.target_bytes,
        container_reserve_bytes=args.container_reserve_bytes,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, document)
    print(json.dumps(document, ensure_ascii=False), flush=True)


def command_plan_robust(args) -> None:
    document = build_robust_plan(
        args.base_mfq,
        args.reap_csv,
        args.sensitivity_map,
        target_bytes=args.target_bytes,
        container_reserve_bytes=args.container_reserve_bytes,
        v_multipliers=tuple(
            float(value)
            for value in args.v_multipliers.split(",")
            if value.strip()
        ),
        overlay_limit_bytes=args.overlay_limit_bytes,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, document)
    print(json.dumps(document, ensure_ascii=False), flush=True)


def command_plan_ud_robust(args) -> None:
    document = build_robust_plan(
        args.base_mfq,
        args.reap_csv,
        args.sensitivity_map,
        target_bytes=args.target_bytes,
        container_reserve_bytes=args.container_reserve_bytes,
        v_multipliers=tuple(
            float(value)
            for value in args.v_multipliers.split(",")
            if value.strip()
        ),
        overlay_limit_bytes=args.overlay_limit_bytes,
        layer_bpw_constraints=UD_IQ1_M_LAYER_CONSTRAINTS,
        layer_policy_source=args.ud_analysis,
        solver_time_limit_seconds=args.solver_time_limit_seconds,
        solver_mip_rel_gap=args.solver_mip_rel_gap,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, document)
    print(json.dumps(document, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan-simple")
    plan.add_argument("--base-mfq", required=True)
    plan.add_argument("--reap-csv", required=True)
    plan.add_argument("--sensitivity-map", required=True)
    plan.add_argument("--target-bytes", type=int)
    plan.add_argument("--container-reserve-bytes", type=int, default=4_000_000)
    plan.add_argument("--output", required=True)
    plan.set_defaults(func=command_plan)
    robust = commands.add_parser("plan-robust")
    robust.add_argument("--base-mfq", required=True)
    robust.add_argument("--reap-csv", required=True)
    robust.add_argument("--sensitivity-map", required=True)
    robust.add_argument("--target-bytes", type=int)
    robust.add_argument(
        "--container-reserve-bytes",
        type=int,
        default=4_000_000,
    )
    robust.add_argument("--v-multipliers", default="1,4,8,16")
    robust.add_argument("--overlay-limit-bytes", type=int)
    robust.add_argument("--output", required=True)
    robust.set_defaults(func=command_plan_robust)
    ud_robust = commands.add_parser("plan-ud-robust")
    ud_robust.add_argument("--base-mfq", required=True)
    ud_robust.add_argument("--reap-csv", required=True)
    ud_robust.add_argument("--sensitivity-map", required=True)
    ud_robust.add_argument("--ud-analysis", required=True)
    ud_robust.add_argument("--target-bytes", type=int)
    ud_robust.add_argument(
        "--container-reserve-bytes",
        type=int,
        default=4_000_000,
    )
    ud_robust.add_argument("--v-multipliers", default="1,4,8,16")
    ud_robust.add_argument("--overlay-limit-bytes", type=int)
    ud_robust.add_argument(
        "--solver-time-limit-seconds",
        type=float,
        default=300.0,
    )
    ud_robust.add_argument(
        "--solver-mip-rel-gap",
        type=float,
        default=0.002,
    )
    ud_robust.add_argument("--output", required=True)
    ud_robust.set_defaults(func=command_plan_ud_robust)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--input", required=True)
    materialize.add_argument("--imatrix", required=True)
    materialize.add_argument("--artifact-dir", required=True)
    materialize.add_argument("--plan", required=True)
    materialize.add_argument("--output", required=True)
    materialize.add_argument("--device", default="cuda")
    materialize.add_argument("--row-chunk", type=int, default=256)
    materialize.set_defaults(func=materialize_overlay)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
