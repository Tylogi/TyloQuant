"""Rate-distortion allocation for per-neuron NINTv2 ``(q, k)`` profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mfq.calibration.allocator import GroupCandidate, allocate_lp_rounded
from mfq.formats.nint import NintSpec

if TYPE_CHECKING:
    from mfq.quantize.weight_solver import (
        ImportanceMap,
        WeightSolver,
        WeightSolverResult,
    )


@dataclass(frozen=True)
class NintV2Allocation:
    """One budget-safe assignment selected from measured row distortions."""

    row_q_bits: np.ndarray
    row_sub_bits: np.ndarray
    target_variable_bits: int
    actual_variable_bits: int
    selected_loss: float
    uniform_loss: float
    solver: str


def candidate_profiles(spec: NintSpec) -> tuple[tuple[int, int], ...]:
    """Return every q width and every k width encodable by this v2 header."""

    minimum_k = max(1, int(spec.sub_bits) - 1)
    maximum_k = min(8, int(spec.sub_bits) + 2)
    return tuple(
        (q_bits, sub_bits)
        for q_bits in range(1, 9)
        for sub_bits in range(minimum_k, maximum_k + 1)
    )


def profile_variable_bits(
    q_bits: int,
    sub_bits: int,
    *,
    values_per_row: int,
    groups_per_row: int,
) -> int:
    """Return packed q plus scale/minimum bits, excluding fixed row fields."""

    if not 1 <= int(q_bits) <= 8 or not 1 <= int(sub_bits) <= 8:
        raise ValueError("NINTv2 q and k widths must lie in [1, 8]")
    if values_per_row <= 0 or groups_per_row <= 0:
        raise ValueError("NINTv2 row geometry must be positive")
    return int(q_bits) * int(values_per_row) + 2 * int(sub_bits) * int(groups_per_row)


def allocate_row_profiles(
    row_losses: np.ndarray,
    profiles: tuple[tuple[int, int], ...],
    spec: NintSpec,
    *,
    values_per_row: int,
    groups_per_row: int,
    target_variable_bits: int | None = None,
) -> NintV2Allocation:
    """Choose one measured ``(q, k)`` candidate per row under one bit budget.

    The supplied loss table should already contain the complete NAQ objective:
    input-channel weighting and output-neuron importance.  The uniform NINTv1
    point is always retained as a non-regression fallback.
    """

    losses = np.asarray(row_losses, dtype=np.float64)
    if losses.ndim != 2 or losses.shape[0] == 0:
        raise ValueError("NINTv2 row losses must be a non-empty matrix")
    if losses.shape[1] != len(profiles) or not profiles:
        raise ValueError("NINTv2 loss columns must match candidate profiles")
    if not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError("NINTv2 row losses must be finite and non-negative")
    if len(set(profiles)) != len(profiles):
        raise ValueError("NINTv2 candidate profiles must be unique")

    rows = int(losses.shape[0])
    costs = np.asarray(
        [
            profile_variable_bits(
                q_bits,
                sub_bits,
                values_per_row=values_per_row,
                groups_per_row=groups_per_row,
            )
            for q_bits, sub_bits in profiles
        ],
        dtype=np.int64,
    )
    try:
        uniform_index = profiles.index((int(spec.bits), int(spec.sub_bits)))
    except ValueError as exc:
        raise ValueError("NINTv2 candidates omit the uniform base profile") from exc
    uniform_bits = rows * int(costs[uniform_index])
    target = uniform_bits if target_variable_bits is None else int(target_variable_bits)
    if target <= 0:
        raise ValueError("NINTv2 target bit budget must be positive")

    candidates = (
        GroupCandidate(
            group=f"row:{row}",
            profile=f"q{q_bits}:k{sub_bits}",
            specs={"q_bits": q_bits, "sub_bits": sub_bits},
            storage_bits=int(costs[index]),
            train_loss=float(losses[row, index]),
            validation_loss=float(losses[row, index]),
        )
        for row in range(rows)
        for index, (q_bits, sub_bits) in enumerate(profiles)
    )
    result = allocate_lp_rounded(candidates, target)
    selected_loss = float(result.train_loss)
    uniform_loss = float(losses[:, uniform_index].sum())
    meaningful_gain = (
        selected_loss < uniform_loss * (1.0 - 1.0e-12)
        if uniform_loss > 0.0
        else False
    )
    if not meaningful_gain or result.actual_storage_bits > target:
        row_q_bits = np.full(rows, int(spec.bits), dtype=np.uint8)
        row_sub_bits = np.full(rows, int(spec.sub_bits), dtype=np.uint8)
        actual = uniform_bits
        selected_loss = uniform_loss
        solver = result.solver + "+uniform-fallback"
    else:
        ordered = [result.selected[f"row:{row}"] for row in range(rows)]
        row_q_bits = np.asarray(
            [int(item.specs["q_bits"]) for item in ordered], dtype=np.uint8
        )
        row_sub_bits = np.asarray(
            [int(item.specs["sub_bits"]) for item in ordered], dtype=np.uint8
        )
        actual = int(result.actual_storage_bits)
        solver = result.solver

    return NintV2Allocation(
        row_q_bits=np.ascontiguousarray(row_q_bits),
        row_sub_bits=np.ascontiguousarray(row_sub_bits),
        target_variable_bits=target,
        actual_variable_bits=actual,
        selected_loss=selected_loss,
        uniform_loss=uniform_loss,
        solver=solver,
    )


def measure_row_profile_losses(
    weight: torch.Tensor | np.ndarray,
    spec: NintSpec,
    profiles: tuple[tuple[int, int], ...],
    *,
    importance: torch.Tensor | np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Measure the complete row objective for every legal q+k candidate."""

    target = torch.device(device)
    value = torch.as_tensor(weight, dtype=torch.float32)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError("NINTv2 profile measurement requires a non-empty matrix")
    rows, columns = map(int, value.shape)
    losses = np.empty((rows, len(profiles)), dtype=np.float64)

    if target.type in {"cuda", "mps"}:
        from mfq.quantize.nint_quant_torch import quantize_axis0

        accelerated = value.to(device=target, dtype=torch.float32)
        for index, (q_bits, sub_bits) in enumerate(profiles):
            _encoded, row_loss = quantize_axis0(
                accelerated,
                NintSpec(q_bits, spec.groupsize, sub_bits),
                device=target,
                importance=importance,
                return_row_sse=True,
            )
            losses[:, index] = row_loss.detach().to("cpu", torch.float64).numpy()
        return losses

    from mfq.quantize.nint_quant import dequantize, quantize

    array = np.ascontiguousarray(value.cpu().numpy(), dtype=np.float32)
    if importance is None:
        importance_rows = None
    else:
        importance_rows = np.asarray(
            importance.detach().cpu().numpy()
            if isinstance(importance, torch.Tensor)
            else importance,
            dtype=np.float32,
        )
        if importance_rows.shape == (columns,):
            importance_rows = np.broadcast_to(importance_rows, (rows, columns))
        elif importance_rows.shape != (rows, columns):
            raise ValueError(
                "NINTv2 importance must have shape [input] or [output,input]"
            )
    for index, (q_bits, sub_bits) in enumerate(profiles):
        encoded = quantize(
            array,
            NintSpec(q_bits, spec.groupsize, sub_bits),
            axis=0,
            importance=importance_rows,
        )
        error = (dequantize(encoded) - array).astype(np.float64) ** 2
        if importance_rows is not None:
            error *= importance_rows
        losses[:, index] = error.sum(axis=1)
    return losses


def solve_nint_v2_weights(
    weight: torch.Tensor | np.ndarray,
    spec: NintSpec,
    allocation: NintV2Allocation,
    solver: WeightSolver,
    *,
    calibration_inputs: torch.Tensor | np.ndarray | None = None,
    hessian: torch.Tensor | np.ndarray | None = None,
    imap: ImportanceMap | None = None,
    tensor_key: str = "anonymous.weight",
) -> WeightSolverResult:
    """Solve value codes for an already allocated NINTv2 q+k Template.

    Template allocation and weight fitting remain peer operations: this helper
    merely binds a resolved allocation to the generic WSolver contract.  It
    neither selects q/k nor changes the tensor's fixed group size or budget.
    """

    from mfq.quantize.nint_solver import NintGridCodec
    from mfq.quantize.weight_solver import WeightSolverProblem

    value = torch.as_tensor(weight)
    if value.ndim != 2:
        raise ValueError("NINTv2 weight solving requires a matrix")
    rows = int(value.shape[0])
    if allocation.row_q_bits.shape != (rows,) or allocation.row_sub_bits.shape != (
        rows,
    ):
        raise ValueError("NINTv2 allocation rows do not match the weight matrix")
    problem = WeightSolverProblem(
        tensor_key,
        value,
        calibration_inputs=calibration_inputs,
        hessian=hessian,
    )
    codec = NintGridCodec(
        spec,
        row_q_bits=allocation.row_q_bits,
        row_sub_bits=allocation.row_sub_bits,
    )
    return solver.solve(problem, codec, imap)


__all__ = [
    "NintV2Allocation",
    "allocate_row_profiles",
    "candidate_profiles",
    "measure_row_profile_losses",
    "profile_variable_bits",
    "solve_nint_v2_weights",
]
