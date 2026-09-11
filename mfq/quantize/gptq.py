"""GPTQ weight solver over MFQ's architecture-neutral scalar-grid contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mfq.quantize.weight_solver import (
    ImportanceMap,
    QuadraticReconstructionObjective,
    ReconstructionObjective,
    ScalarGridCodec,
    ScalarGridTensor,
    WeightSolver,
    WeightSolverProblem,
    WeightSolverResult,
)


@dataclass(frozen=True)
class GptqConfig:
    block_size: int = 128
    damp_percent: float = 0.01
    act_order: bool = True
    hessian_dtype: torch.dtype = torch.float32
    acceptance_tolerance: float = 1.0e-7
    max_cholesky_retries: int = 6

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("GPTQ block size must be positive")
        if self.damp_percent < 0 or not torch.isfinite(torch.tensor(self.damp_percent)):
            raise ValueError("GPTQ dampening must be finite and non-negative")
        if self.hessian_dtype not in {torch.float32, torch.float64}:
            raise ValueError("GPTQ Hessian dtype must be float32 or float64")
        if self.acceptance_tolerance < 0 or self.max_cholesky_retries < 1:
            raise ValueError("invalid GPTQ acceptance or Cholesky configuration")


def _inverse_cholesky_factor(
    hessian: torch.Tensor,
    damp_percent: float,
    retries: int,
) -> tuple[torch.Tensor, float]:
    """Return the upper Cholesky factor of ``H^-1`` with stable dampening."""

    diagonal = torch.diagonal(hessian)
    mean_diagonal = float(diagonal.mean().detach().cpu())
    scale = max(abs(mean_diagonal), torch.finfo(hessian.dtype).eps)
    base_damp = max(float(damp_percent) * scale, torch.finfo(hessian.dtype).eps)
    identity = torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)
    last_error: RuntimeError | None = None
    for attempt in range(retries):
        damp = base_damp * (10.0**attempt)
        try:
            chol = torch.linalg.cholesky(hessian + identity * damp)
            inverse = torch.cholesky_inverse(chol)
            return torch.linalg.cholesky(inverse, upper=True), damp
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError("GPTQ Hessian remained non-positive-definite after dampening") from last_error


class GptqSolver(WeightSolver):
    """One-shot Optimal Brain Quantization with lazy block error updates.

    Grid parameters are initialized by the target codec and held fixed during
    the GPTQ pass.  Only integer assignments are changed, so a format-specific
    codec such as NINT can seal the result without changing its metadata layout.
    """

    def __init__(
        self,
        config: GptqConfig | None = None,
        objective: ReconstructionObjective | None = None,
    ) -> None:
        self.config = GptqConfig() if config is None else config
        self.objective = (
            QuadraticReconstructionObjective() if objective is None else objective
        )

    def solve(
        self,
        problem: WeightSolverProblem,
        codec: ScalarGridCodec,
        imap: ImportanceMap | None = None,
        *,
        initial: ScalarGridTensor | None = None,
    ) -> WeightSolverResult:
        initial_grid = codec.initialize(problem, imap) if initial is None else initial
        if tuple(initial_grid.codes.shape[:1]) != (int(problem.weight.shape[0]),):
            raise ValueError("GPTQ initial grid rows do not match the weight matrix")
        if initial_grid.value_count != int(problem.weight.shape[1]):
            raise ValueError("GPTQ initial grid width does not match the weight matrix")
        device = initial_grid.codes.device
        dtype = self.config.hessian_dtype
        reference = problem.weight.to(device=device, dtype=dtype)
        rows, columns = map(int, reference.shape)
        hessian_provider = getattr(self.objective, "gptq_hessian", None)
        hessian = (
            hessian_provider(problem, imap, device=device, dtype=dtype)
            if hessian_provider is not None
            else None
        )
        if hessian is None:
            raise TypeError(
                "GPTQ requires an objective that supplies a calibration Hessian"
            )
        if tuple(hessian.shape) != (columns, columns):
            raise ValueError("GPTQ objective returned an invalid Hessian")

        permutation = (
            torch.argsort(torch.diagonal(hessian), descending=True)
            if self.config.act_order
            else torch.arange(columns, device=device)
        )
        permutation_list = permutation.detach().cpu().tolist()
        ordered_hessian = hessian.index_select(0, permutation).index_select(1, permutation)
        inverse_factor, actual_damp = _inverse_cholesky_factor(
            ordered_hessian,
            self.config.damp_percent,
            self.config.max_cholesky_retries,
        )
        working = reference.index_select(1, permutation).clone()
        codes = initial_grid.codes.clone()
        scales = initial_grid.effective_scales().to(device=device, dtype=dtype)
        offsets = initial_grid.effective_offsets().to(device=device, dtype=dtype)
        qmin, qmax = initial_grid.q_limits()
        qmin = qmin.to(device=device, dtype=dtype)
        qmax = qmax.to(device=device, dtype=dtype)

        for block_start in range(0, columns, self.config.block_size):
            block_stop = min(block_start + self.config.block_size, columns)
            width = block_stop - block_start
            block_weight = working[:, block_start:block_stop].clone()
            block_reconstruction = torch.empty_like(block_weight)
            block_error = torch.empty_like(block_weight)
            for local_column in range(width):
                ordered_column = block_start + local_column
                original_column = permutation_list[ordered_column]
                group = original_column // initial_grid.group_size
                scale = scales[:, group]
                offset = offsets[:, group]
                safe_scale = torch.where(
                    scale.abs() > torch.finfo(dtype).tiny,
                    scale,
                    torch.ones_like(scale),
                )
                value = block_weight[:, local_column]
                quantized_code = torch.clamp(
                    torch.round((value - offset) / safe_scale), qmin, qmax
                )
                quantized_value = quantized_code * scale + offset
                diagonal = inverse_factor[ordered_column, ordered_column]
                error = (value - quantized_value) / diagonal
                block_reconstruction[:, local_column] = quantized_value
                block_error[:, local_column] = error
                codes[:, original_column] = quantized_code.to(torch.int16)
                block_weight[:, local_column:] -= error[:, None] * inverse_factor[
                    ordered_column, ordered_column:block_stop
                ][None, :]
            working[:, block_start:block_stop] = block_reconstruction
            if block_stop < columns:
                working[:, block_stop:] -= block_error @ inverse_factor[
                    block_start:block_stop, block_stop:
                ]

        candidate_grid = codec.canonicalize(initial_grid.with_values(codes=codes))
        baseline_grid = codec.canonicalize(initial_grid)
        candidate_reconstruction = candidate_grid.dequantize()
        baseline_reconstruction = baseline_grid.dequantize()
        candidate_value = self.objective.evaluate(
            problem, candidate_reconstruction, imap
        )
        baseline_value = self.objective.evaluate(
            problem, baseline_reconstruction, imap
        )
        accepted = candidate_value.total <= baseline_value.total * (
            1.0 + self.config.acceptance_tolerance
        )
        selected_grid = candidate_grid if accepted else baseline_grid
        reconstruction = (
            candidate_reconstruction if accepted else baseline_reconstruction
        )
        objective_value = candidate_value if accepted else baseline_value
        row_losses = self.objective.row_losses(
            problem, reconstruction, imap
        ).detach()
        encoded = codec.finalize(selected_grid)
        improvement = (
            0.0
            if baseline_value.total == 0
            else (baseline_value.total - objective_value.total) / baseline_value.total
        )
        return WeightSolverResult(
            encoded=encoded,
            grid=selected_grid,
            reconstruction=reconstruction.detach(),
            objective=objective_value,
            baseline=baseline_value,
            row_losses=row_losses,
            metrics={
                "accepted": float(accepted),
                "relative_improvement": float(improvement),
                "hessian_damp": float(actual_damp),
                "block_size": float(self.config.block_size),
                "act_order": float(self.config.act_order),
                "rows": float(rows),
                "columns": float(columns),
            },
        )


__all__ = ["GptqConfig", "GptqSolver"]
