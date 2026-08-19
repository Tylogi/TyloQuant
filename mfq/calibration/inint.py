"""Calibration-driven per-neuron NINT4/NINT8 selection."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mfq.calibration.evaluator import TensorCandidateEvaluation

_FORMAT = "mfq.inint-selector.v1"
_METADATA_KEY = "__metadata_json__"


@dataclass(frozen=True)
class InintSelector:
    path: Path | None
    target_profile: str
    low_profile: str
    high_profile: str
    selectors: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def selected_rows(self) -> int:
        return int(sum(np.count_nonzero(value) for value in self.selectors.values()))

    @property
    def row_count(self) -> int:
        return int(sum(value.size for value in self.selectors.values()))


def save_inint_selector(path: str | Path, selector: InintSelector) -> None:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"ININT selector already exists: {output}")
    if not selector.selectors:
        raise ValueError("cannot save an empty ININT selector")
    output.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    entries: dict[str, Any] = {}
    for index, (name, raw) in enumerate(sorted(selector.selectors.items())):
        values = np.asarray(raw, dtype=np.bool_).reshape(-1)
        key = f"selector_{index:04d}"
        arrays[key] = np.packbits(values, bitorder="little")
        entries[name] = {"key": key, "rows": int(values.size)}
    document = {
        "format": _FORMAT,
        "target_profile": selector.target_profile,
        "low_profile": selector.low_profile,
        "high_profile": selector.high_profile,
        "metadata": selector.metadata,
        "entries": entries,
    }
    arrays[_METADATA_KEY] = np.frombuffer(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    ).copy()
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, output)


def load_inint_selector(path: str | Path) -> InintSelector:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"ININT selector does not exist: {resolved}")
    with np.load(resolved, allow_pickle=False) as archive:
        if _METADATA_KEY not in archive:
            raise ValueError(f"ININT selector has no metadata: {resolved}")
        document = json.loads(archive[_METADATA_KEY].tobytes().decode("utf-8"))
        if document.get("format") != _FORMAT:
            raise ValueError(f"unsupported ININT selector format: {document.get('format')!r}")
        selectors: dict[str, np.ndarray] = {}
        for name, item in document.get("entries", {}).items():
            rows = int(item["rows"])
            if rows <= 0:
                raise ValueError(f"invalid row count for ININT tensor {name}")
            packed = np.asarray(archive[str(item["key"])], dtype=np.uint8)
            selectors[str(name)] = np.unpackbits(packed, count=rows, bitorder="little").astype(
                np.bool_, copy=False
            )
    if not selectors:
        raise ValueError(f"ININT selector contains no tensors: {resolved}")
    return InintSelector(
        path=resolved,
        target_profile=str(document["target_profile"]),
        low_profile=str(document["low_profile"]),
        high_profile=str(document["high_profile"]),
        selectors=selectors,
        metadata=dict(document.get("metadata", {})),
    )


def _solve_milp(benefit: np.ndarray, cost: np.ndarray, budget: int) -> np.ndarray:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("ININT row allocation requires scipy") from exc

    count = int(benefit.size)
    if count == 0 or budget <= 0:
        return np.zeros(count, dtype=np.bool_)
    positive = benefit > 0
    if int(cost[positive].sum(dtype=np.int64)) <= budget:
        return positive
    scale = max(float(np.median(benefit[positive])), np.finfo(np.float64).tiny)
    objective = -(benefit / scale) + (cost / float(budget)) * 1e-12
    normalized_cost = cost / float(budget)
    matrix = csr_matrix(normalized_cost.reshape(1, -1))
    budget_upper = 1.0
    for _ in range(4):
        result = milp(
            c=objective,
            integrality=np.ones(count, dtype=np.uint8),
            bounds=Bounds(np.zeros(count), positive.astype(np.float64)),
            constraints=LinearConstraint(
                matrix,
                np.asarray([-np.inf]),
                np.asarray([budget_upper]),
            ),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"ININT row MILP failed: {result.message}")
        selected = result.x > 0.5
        actual_cost = int(cost[selected].sum(dtype=np.int64))
        if actual_cost <= budget:
            return selected
        guard = max(1, int(np.ceil(budget * 1e-7)))
        budget_upper -= (actual_cost - budget + guard) / float(budget)
    raise RuntimeError("ININT row MILP exceeded its budget after constraint tightening")


def _solve_large(
    benefit: np.ndarray,
    cost: np.ndarray,
    budget: int,
    boundary_rows: int,
) -> np.ndarray:
    """Solve a large row allocation with a fixed prefix and exact boundary."""

    count = int(benefit.size)
    order = np.lexsort((np.arange(count, dtype=np.int64), -benefit, -(benefit / cost)))
    ordered_cost = cost[order]
    cumulative = np.cumsum(ordered_cost, dtype=np.int64)
    crossing = int(np.searchsorted(cumulative, budget, side="right"))
    half = max(1, boundary_rows // 2)
    start = max(0, crossing - half)
    end = min(count, crossing + half)

    selected = np.zeros(count, dtype=np.bool_)
    fixed = order[:start]
    fixed = fixed[benefit[fixed] > 0]
    selected[fixed] = True
    fixed_cost = int(cost[fixed].sum(dtype=np.int64))
    if fixed_cost > budget:
        raise RuntimeError("ININT fixed prefix exceeded its budget")

    boundary = order[start:end]
    boundary_choice = _solve_milp(benefit[boundary], cost[boundary], budget - fixed_cost)
    selected[boundary[boundary_choice]] = True

    remaining = budget - int(cost[selected].sum(dtype=np.int64))
    if remaining > 0:
        for index in order[end:]:
            if benefit[index] <= 0:
                break
            item_cost = int(cost[index])
            if item_cost <= remaining:
                selected[index] = True
                remaining -= item_cost
    return selected


def build_inint_selector(
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
    output: str | Path,
    *,
    target_profile: str,
    low_profile: str = "NINT4",
    high_profile: str = "NINT8",
    exact_row_limit: int = 100_000,
    boundary_rows: int = 32_768,
    metadata: Mapping[str, Any] | None = None,
) -> InintSelector:
    """Select NINT8 rows under the storage budget of a uniform target profile."""

    target_profile = target_profile.upper()
    low_profile = low_profile.upper()
    high_profile = high_profile.upper()
    if exact_row_limit <= 0 or boundary_rows <= 0:
        raise ValueError("ININT solver limits must be positive")

    names = sorted(evaluations)
    offsets: dict[str, tuple[int, int]] = {}
    benefits: list[np.ndarray] = []
    validation_benefits: list[np.ndarray] = []
    costs: list[np.ndarray] = []
    base_storage = 0
    maximum_storage = 0
    target_storage = 0
    cursor = 0
    for name in names:
        choices = evaluations[name]
        missing = {low_profile, high_profile, target_profile} - set(choices)
        if missing:
            raise ValueError(f"tensor {name} lacks ININT candidates: {sorted(missing)}")
        low = choices[low_profile]
        high = choices[high_profile]
        target = choices[target_profile]
        if (low.rows, low.columns) != (high.rows, high.columns):
            raise ValueError(f"ININT candidate shape mismatch for {name}")
        low_row_cost = low.storage_bits // low.rows
        high_row_cost = high.storage_bits // high.rows
        if low_row_cost * low.rows != low.storage_bits:
            raise ValueError(f"NINT4 storage is not row-separable for {name}")
        if high_row_cost * high.rows != high.storage_bits:
            raise ValueError(f"NINT8 storage is not row-separable for {name}")
        extra = high_row_cost - low_row_cost
        if extra <= 0:
            raise ValueError(f"ININT high profile does not cost more for {name}")
        row_benefit = low.train_row_loss.astype(np.float64) - high.train_row_loss
        row_validation = low.validation_row_loss.astype(np.float64) - high.validation_row_loss
        offsets[name] = (cursor, cursor + low.rows)
        cursor += low.rows
        benefits.append(row_benefit)
        validation_benefits.append(row_validation)
        costs.append(np.full(low.rows, extra, dtype=np.int64))
        base_storage += low.storage_bits
        maximum_storage += high.storage_bits
        target_storage += target.storage_bits

    benefit = np.concatenate(benefits)
    validation_benefit = np.concatenate(validation_benefits)
    cost = np.concatenate(costs)
    budget = min(maximum_storage, target_storage) - base_storage
    if budget < 0:
        raise ValueError(f"target profile {target_profile} is smaller than the {low_profile} base")

    if benefit.size <= exact_row_limit:
        selected = _solve_milp(benefit, cost, budget)
        solver = "scipy.optimize.milp/highs-exact"
    else:
        selected = _solve_large(benefit, cost, budget, boundary_rows)
        solver = f"ratio-prefix+highs-boundary-{boundary_rows}"

    actual_storage = base_storage + int(cost[selected].sum(dtype=np.int64))
    if actual_storage > target_storage:
        raise RuntimeError("ININT allocation exceeds its target profile storage")
    selectors = {
        name: np.ascontiguousarray(selected[start:end], dtype=np.bool_)
        for name, (start, end) in offsets.items()
    }
    train_gain = float(benefit[selected].sum(dtype=np.float64))
    validation_gain = float(validation_benefit[selected].sum(dtype=np.float64))
    selector = InintSelector(
        path=None,
        target_profile=target_profile,
        low_profile=low_profile,
        high_profile=high_profile,
        selectors=selectors,
        metadata={
            "solver": solver,
            "base_storage_bits": int(base_storage),
            "target_storage_bits": int(target_storage),
            "actual_storage_bits": int(actual_storage),
            "unused_storage_bits": int(target_storage - actual_storage),
            "selected_rows": int(np.count_nonzero(selected)),
            "row_count": int(selected.size),
            "train_loss_reduction": train_gain,
            "validation_loss_reduction": validation_gain,
            **dict(metadata or {}),
        },
    )
    save_inint_selector(output, selector)
    return load_inint_selector(output)


__all__ = [
    "InintSelector",
    "build_inint_selector",
    "load_inint_selector",
    "save_inint_selector",
]
