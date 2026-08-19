"""Exact global allocation of independently scored precision groups."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupCandidate:
    group: str
    profile: str
    specs: Mapping[str, object]
    storage_bits: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True)
class AllocationResult:
    selected: dict[str, GroupCandidate]
    target_storage_bits: int
    actual_storage_bits: int
    train_loss: float
    validation_loss: float
    solver: str


def allocate(
    candidates: Iterable[GroupCandidate],
    target_storage_bits: int,
    *,
    baseline_profiles: Mapping[str, str] | None = None,
    max_changed_groups: int | None = None,
) -> AllocationResult:
    """Solve the multiple-choice knapsack exactly with SciPy/HiGHS MILP."""

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("global precision allocation requires scipy") from exc

    values = list(candidates)
    if not values:
        raise ValueError("no precision candidates were provided")
    if target_storage_bits <= 0:
        raise ValueError("target_storage_bits must be positive")
    if (baseline_profiles is None) != (max_changed_groups is None):
        raise ValueError("baseline_profiles and max_changed_groups must be provided together")
    if max_changed_groups is not None and max_changed_groups < 0:
        raise ValueError("max_changed_groups must be non-negative")
    by_group: dict[str, list[int]] = {}
    for index, item in enumerate(values):
        if item.storage_bits <= 0:
            raise ValueError(f"candidate {item.group}/{item.profile} has invalid storage cost")
        if not np.isfinite(item.train_loss) or item.train_loss < 0:
            raise ValueError(f"candidate {item.group}/{item.profile} has invalid train loss")
        if not np.isfinite(item.validation_loss) or item.validation_loss < 0:
            raise ValueError(f"candidate {item.group}/{item.profile} has invalid validation loss")
        if not item.specs:
            raise ValueError(f"candidate {item.group}/{item.profile} contains no tensors")
        by_group.setdefault(item.group, []).append(index)

    minimum = sum(
        min(values[index].storage_bits for index in indices) for indices in by_group.values()
    )
    maximum = sum(
        max(values[index].storage_bits for index in indices) for indices in by_group.values()
    )
    if target_storage_bits < minimum:
        raise ValueError(
            f"target budget {target_storage_bits} bits is below minimum feasible {minimum} bits"
        )
    target_storage_bits = min(int(target_storage_bits), int(maximum))

    group_names = sorted(by_group)
    if baseline_profiles is not None:
        if set(baseline_profiles) != set(group_names):
            raise ValueError("baseline profiles do not cover every precision group")
        for group in group_names:
            profiles = {values[index].profile for index in by_group[group]}
            if baseline_profiles[group] not in profiles:
                raise ValueError(f"group {group} has no baseline profile")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row, group in enumerate(group_names):
        for index in by_group[group]:
            rows.append(row)
            cols.append(index)
            data.append(1.0)
    budget_row = len(group_names)
    cost_scale = float(target_storage_bits)
    for index, item in enumerate(values):
        rows.append(budget_row)
        cols.append(index)
        data.append(item.storage_bits / cost_scale)
    row_count = len(group_names) + 1
    if baseline_profiles is not None:
        change_row = row_count
        for index, item in enumerate(values):
            if item.profile != baseline_profiles[item.group]:
                rows.append(change_row)
                cols.append(index)
                data.append(1.0)
        row_count += 1
    matrix = csr_matrix((data, (rows, cols)), shape=(row_count, len(values)))
    lower = np.concatenate([np.ones(len(group_names)), np.asarray([-np.inf])])
    upper = np.concatenate([np.ones(len(group_names)), np.asarray([1.0])])
    if max_changed_groups is not None:
        lower = np.concatenate([lower, np.asarray([-np.inf])])
        upper = np.concatenate([upper, np.asarray([float(max_changed_groups)])])

    losses = np.asarray([item.train_loss for item in values], dtype=np.float64)
    positive = losses[losses > 0]
    objective_scale = float(np.median(positive)) if positive.size else 1.0
    objective = losses / objective_scale
    normalized_cost = np.asarray(
        [item.storage_bits / cost_scale for item in values], dtype=np.float64
    )
    objective += normalized_cost * 1e-12
    result = None
    selected: dict[str, GroupCandidate] = {}
    actual = 0
    budget_upper = 1.0
    for _attempt in range(4):
        solve_upper = upper.copy()
        solve_upper[budget_row] = budget_upper
        result = milp(
            c=objective,
            integrality=np.ones(len(values), dtype=np.uint8),
            bounds=Bounds(np.zeros(len(values)), np.ones(len(values))),
            constraints=LinearConstraint(matrix, lower, solve_upper),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"global precision allocation failed: {result.message}")

        chosen_indices = np.flatnonzero(result.x > 0.5)
        if chosen_indices.size != len(group_names):
            raise RuntimeError("MILP returned an invalid number of group choices")
        selected = {values[index].group: values[index] for index in chosen_indices}
        if set(selected) != set(group_names):
            raise RuntimeError("MILP omitted or duplicated a precision group")
        actual = sum(item.storage_bits for item in selected.values())
        if actual <= target_storage_bits:
            break

        # HiGHS applies feasibility tolerances to the normalized budget row.
        # Tighten by the observed integer overage plus a small guard and solve
        # the original discrete problem again.
        guard_bits = max(1, int(np.ceil(target_storage_bits * 1e-7)))
        budget_upper -= (actual - target_storage_bits + guard_bits) / cost_scale
    else:
        raise RuntimeError("MILP allocation exceeds the storage budget after tightening")
    assert result is not None
    return AllocationResult(
        selected=selected,
        target_storage_bits=target_storage_bits,
        actual_storage_bits=actual,
        train_loss=float(sum(item.train_loss for item in selected.values())),
        validation_loss=float(sum(item.validation_loss for item in selected.values())),
        solver="scipy.optimize.milp/highs",
    )


def allocate_lp_rounded(
    candidates: Iterable[GroupCandidate],
    target_storage_bits: int,
) -> AllocationResult:
    """Solve the LP relaxation and round its at-most-one fractional group down."""

    try:
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("global precision allocation requires scipy") from exc

    values = list(candidates)
    if not values:
        raise ValueError("no precision candidates were provided")
    if target_storage_bits <= 0:
        raise ValueError("target_storage_bits must be positive")
    by_group: dict[str, list[int]] = {}
    for index, item in enumerate(values):
        if item.storage_bits <= 0:
            raise ValueError(f"candidate {item.group}/{item.profile} has invalid storage cost")
        if not np.isfinite(item.train_loss) or item.train_loss < 0:
            raise ValueError(f"candidate {item.group}/{item.profile} has invalid train loss")
        if not np.isfinite(item.validation_loss) or item.validation_loss < 0:
            raise ValueError(f"candidate {item.group}/{item.profile} has invalid validation loss")
        if not item.specs:
            raise ValueError(f"candidate {item.group}/{item.profile} contains no tensors")
        by_group.setdefault(item.group, []).append(index)

    minimum = sum(
        min(values[index].storage_bits for index in indices) for indices in by_group.values()
    )
    maximum = sum(
        max(values[index].storage_bits for index in indices) for indices in by_group.values()
    )
    if target_storage_bits < minimum:
        raise ValueError(
            f"target budget {target_storage_bits} bits is below minimum feasible {minimum} bits"
        )
    target_storage_bits = min(int(target_storage_bits), int(maximum))

    group_names = sorted(by_group)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row, group in enumerate(group_names):
        for index in by_group[group]:
            rows.append(row)
            cols.append(index)
            data.append(1.0)
    equality = csr_matrix(
        (data, (rows, cols)),
        shape=(len(group_names), len(values)),
    )
    cost_scale = float(target_storage_bits)
    budget = csr_matrix(
        (
            [item.storage_bits / cost_scale for item in values],
            ([0] * len(values), list(range(len(values)))),
        ),
        shape=(1, len(values)),
    )
    losses = np.asarray([item.train_loss for item in values], dtype=np.float64)
    positive = losses[losses > 0]
    objective_scale = float(np.median(positive)) if positive.size else 1.0
    objective = losses / objective_scale
    objective += np.asarray(
        [item.storage_bits / cost_scale for item in values], dtype=np.float64
    ) * 1e-12
    result = linprog(
        objective,
        A_ub=budget,
        b_ub=np.asarray([1.0]),
        A_eq=equality,
        b_eq=np.ones(len(group_names)),
        bounds=(0.0, 1.0),
        method="highs",
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"global precision LP allocation failed: {result.message}")

    tolerance = 1e-8
    selected_indices: dict[str, int] = {}
    fractional_groups: list[str] = []
    for group in group_names:
        indices = by_group[group]
        positive_indices = [index for index in indices if result.x[index] > tolerance]
        if not positive_indices:
            positive_indices = [max(indices, key=lambda index: result.x[index])]
        if len(positive_indices) > 1:
            fractional_groups.append(group)
        selected_indices[group] = min(
            positive_indices,
            key=lambda index: (values[index].storage_bits, values[index].train_loss),
        )

    actual = sum(values[index].storage_bits for index in selected_indices.values())
    if actual > target_storage_bits:
        raise RuntimeError("rounded precision LP allocation exceeds the storage budget")

    # A basic LP solution has at most one fractional precision group. Spend any
    # remaining capacity on its best feasible integer option. The loop also
    # handles numerically degenerate solutions with more than one such group.
    for group in fractional_groups:
        current = selected_indices[group]
        available = target_storage_bits - actual + values[current].storage_bits
        feasible = [
            index
            for index in by_group[group]
            if values[index].storage_bits <= available
        ]
        replacement = min(
            feasible,
            key=lambda index: (values[index].train_loss, values[index].storage_bits),
        )
        actual += values[replacement].storage_bits - values[current].storage_bits
        selected_indices[group] = replacement

    selected = {group: values[index] for group, index in selected_indices.items()}
    return AllocationResult(
        selected=selected,
        target_storage_bits=target_storage_bits,
        actual_storage_bits=actual,
        train_loss=float(sum(item.train_loss for item in selected.values())),
        validation_loss=float(sum(item.validation_loss for item in selected.values())),
        solver="scipy.optimize.linprog/highs+integer-rounding",
    )


__all__ = ["AllocationResult", "GroupCandidate", "allocate", "allocate_lp_rounded"]
