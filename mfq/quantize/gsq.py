"""Gumbel-Softmax Quantization (GSQ) over a format-preserving scalar grid."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mfq.quantize.gptq import GptqConfig, GptqSolver
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
class GsqConfig:
    steps: int = 200
    row_chunk_size: int = 32
    local_radius: int = 2
    logits_lr: float = 3.0e-2
    scale_lr: float = 3.0e-3
    offset_lr: float = 3.0e-3
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.0
    temperature_start: float = 1.0
    temperature_end: float = 0.05
    logit_scale_start: float = 0.5
    logit_scale_end: float = 8.0
    initial_logit_bias: float = 2.0
    hard_eval_interval: int = 10
    seed: int = 0
    use_gumbel_noise: bool = True
    use_gptq_init: bool = True
    learn_scales: bool = True
    learn_offsets: bool = False
    acceptance_tolerance: float = 1.0e-7

    def __post_init__(self) -> None:
        positive_ints = (
            self.steps,
            self.row_chunk_size,
            self.hard_eval_interval,
        )
        if any(value <= 0 for value in positive_ints) or self.local_radius < 1:
            raise ValueError("GSQ iteration, chunk, evaluation, and radius values are invalid")
        positive_floats = (
            self.logits_lr,
            self.scale_lr,
            self.offset_lr,
            self.temperature_start,
            self.temperature_end,
            self.logit_scale_start,
            self.logit_scale_end,
        )
        if any(value <= 0 or not torch.isfinite(torch.tensor(value)) for value in positive_floats):
            raise ValueError("GSQ learning rates and annealing endpoints must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("GSQ Lion beta values must lie in [0, 1)")
        if self.weight_decay < 0 or self.acceptance_tolerance < 0:
            raise ValueError("GSQ decay and acceptance tolerance cannot be negative")


def _anneal(start: float, end: float, progress: float) -> float:
    return start * ((end / start) ** progress)


def _gumbel_like(value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if value.device.type in {"cpu", "cuda"}:
        uniform = torch.rand(
            value.shape, device=value.device, dtype=value.dtype, generator=generator
        )
    else:
        uniform = torch.rand(
            value.shape, device="cpu", dtype=value.dtype, generator=generator
        ).to(value.device)
    epsilon = torch.finfo(value.dtype).eps
    uniform = uniform.clamp(min=epsilon, max=1.0 - epsilon)
    return -torch.log(-torch.log(uniform))


def _lion_step(
    parameters: tuple[tuple[torch.Tensor, float], ...],
    moments: tuple[torch.Tensor, ...],
    config: GsqConfig,
) -> None:
    with torch.no_grad():
        for (parameter, learning_rate), moment in zip(parameters, moments, strict=True):
            gradient = parameter.grad
            if gradient is None:
                continue
            if config.weight_decay:
                parameter.mul_(1.0 - learning_rate * config.weight_decay)
            update = moment.mul(config.beta1).add(gradient, alpha=1.0 - config.beta1)
            parameter.add_(update.sign(), alpha=-learning_rate)
            moment.mul_(config.beta2).add_(gradient, alpha=1.0 - config.beta2)
            parameter.grad = None


def _candidate_levels(
    base_codes: torch.Tensor,
    qmin: torch.Tensor,
    qmax: torch.Tensor,
    q_bits: torch.Tensor,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_count = max(4, 2 * radius + 1)
    indices = torch.arange(candidate_count, device=base_codes.device, dtype=torch.int64)
    full = qmin[:, None, None] + indices[None, None, :]
    local = base_codes[:, :, None].to(torch.int64) + (indices - radius)[None, None, :]
    use_full = (q_bits <= 2)[:, None, None]
    levels = torch.where(use_full, full, local)
    valid_full = full <= qmax[:, None, None]
    valid_local = (local >= qmin[:, None, None]) & (local <= qmax[:, None, None])
    valid = torch.where(use_full, valid_full, valid_local).expand_as(levels)
    if not bool(valid.any(dim=-1).all()):
        raise RuntimeError("GSQ candidate construction omitted a weight's valid grid")
    return levels, valid


def _decode_parameters(
    scale_raw: torch.Tensor,
    offset_raw: torch.Tensor | None,
    base_offset: torch.Tensor,
    offset_constraint: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.exp(scale_raw.clamp(min=-30.0, max=30.0))
    if offset_raw is None:
        offset = base_offset
    elif offset_constraint == "nonpositive":
        offset = -torch.exp(offset_raw.clamp(min=-30.0, max=30.0))
    else:
        offset = offset_raw
    return scale, offset


def _reconstruct(
    codes: torch.Tensor,
    scale_parameters: torch.Tensor,
    offset_parameters: torch.Tensor,
    grid: ScalarGridTensor,
    row_start: int,
) -> torch.Tensor:
    row_stop = row_start + int(codes.shape[0])
    scale_indices = grid.scale_indices[row_start:row_stop]
    offset_indices = grid.offset_indices[row_start:row_stop]
    scales = (
        torch.gather(scale_parameters, 1, scale_indices)
        * grid.scale_multipliers[row_start:row_stop]
    )
    offsets = (
        torch.gather(offset_parameters, 1, offset_indices)
        * grid.offset_multipliers[row_start:row_stop]
    )
    group_index = torch.arange(
        grid.value_count, device=codes.device, dtype=torch.int64
    ) // grid.group_size
    return (
        codes.to(torch.float32) * scales[:, group_index]
        + offsets[:, group_index]
    )


class GsqSolver(WeightSolver):
    """Jointly learn integer-grid assignments and representable scale parameters.

    Two-bit rows search their full grid.  Wider rows search the five-point
    neighborhood around their GPTQ code, matching GSQ's local-grid strategy.
    The final result is a hard argmax assignment, and an exact codec projection
    plus non-regression check occurs before it is returned.
    """

    def __init__(
        self,
        config: GsqConfig | None = None,
        objective: ReconstructionObjective | None = None,
        gptq_config: GptqConfig | None = None,
    ) -> None:
        self.config = GsqConfig() if config is None else config
        self.objective = (
            QuadraticReconstructionObjective() if objective is None else objective
        )
        self.gptq_config = GptqConfig() if gptq_config is None else gptq_config

    def solve(
        self,
        problem: WeightSolverProblem,
        codec: ScalarGridCodec,
        imap: ImportanceMap | None = None,
        *,
        initial: ScalarGridTensor | None = None,
    ) -> WeightSolverResult:
        if initial is None:
            initial = codec.initialize(problem, imap)
            if self.config.use_gptq_init:
                initial = GptqSolver(
                    self.gptq_config, objective=self.objective
                ).solve(problem, codec, imap, initial=initial).grid
        baseline_grid = codec.canonicalize(initial)
        baseline_reconstruction = baseline_grid.dequantize()
        baseline_value = self.objective.evaluate(
            problem, baseline_reconstruction, imap
        )
        baseline_rows = self.objective.row_losses(
            problem, baseline_reconstruction, imap
        ).detach()

        codes = baseline_grid.codes.clone()
        scale_parameters = baseline_grid.scale_parameters.clone()
        offset_parameters = baseline_grid.offset_parameters.clone()
        qmin_all, qmax_all = baseline_grid.q_limits()
        if baseline_grid.codes.device.type in {"cpu", "cuda"}:
            generator = torch.Generator(device=baseline_grid.codes.device)
        else:
            generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)

        for row_start in range(0, baseline_grid.rows, self.config.row_chunk_size):
            row_stop = min(row_start + self.config.row_chunk_size, baseline_grid.rows)
            chunk_codes = codes[row_start:row_stop, : baseline_grid.value_count]
            chunk_q_bits = baseline_grid.q_bits[row_start:row_stop]
            chunk_qmin = qmin_all[row_start:row_stop]
            chunk_qmax = qmax_all[row_start:row_stop]
            levels, valid = _candidate_levels(
                chunk_codes,
                chunk_qmin,
                chunk_qmax,
                chunk_q_bits,
                self.config.local_radius,
            )
            initial_match = (levels == chunk_codes[:, :, None]) & valid
            if not bool(initial_match.any(dim=-1).all()):
                raise RuntimeError("GSQ local grid does not contain its initializer")
            logits = torch.zeros(
                levels.shape,
                device=chunk_codes.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            with torch.no_grad():
                logits.add_(initial_match.to(logits.dtype) * self.config.initial_logit_bias)

            scale_base = scale_parameters[row_start:row_stop]
            scale_raw = torch.log(
                torch.clamp(scale_base, min=torch.finfo(torch.float32).tiny)
            ).detach()
            scale_raw.requires_grad_(self.config.learn_scales)
            offset_base = offset_parameters[row_start:row_stop]
            offset_raw: torch.Tensor | None = None
            if self.config.learn_offsets and baseline_grid.offset_constraint != "zero":
                if baseline_grid.offset_constraint == "nonpositive":
                    offset_raw = torch.log(
                        torch.clamp(-offset_base, min=torch.finfo(torch.float32).tiny)
                    ).detach()
                else:
                    offset_raw = offset_base.detach().clone()
                offset_raw.requires_grad_(True)

            parameters: list[tuple[torch.Tensor, float]] = [
                (logits, self.config.logits_lr)
            ]
            if self.config.learn_scales:
                parameters.append((scale_raw, self.config.scale_lr))
            if offset_raw is not None:
                parameters.append((offset_raw, self.config.offset_lr))
            parameter_tuple = tuple(parameters)
            moments = tuple(torch.zeros_like(parameter) for parameter, _ in parameter_tuple)
            best_loss = baseline_rows[row_start:row_stop].to(chunk_codes.device).clone()
            best_codes = chunk_codes.clone()
            best_scale = scale_base.clone()
            best_offset = offset_base.clone()

            for step in range(self.config.steps):
                progress = step / max(self.config.steps - 1, 1)
                temperature = _anneal(
                    self.config.temperature_start,
                    self.config.temperature_end,
                    progress,
                )
                logit_scale = _anneal(
                    self.config.logit_scale_start,
                    self.config.logit_scale_end,
                    progress,
                )
                score = logits * logit_scale
                if self.config.use_gumbel_noise:
                    score = score + _gumbel_like(score, generator)
                score = score.masked_fill(~valid, -torch.inf)
                probabilities = torch.softmax(score / temperature, dim=-1)
                relaxed_codes = (probabilities * levels.to(torch.float32)).sum(dim=-1)
                current_scale, current_offset = _decode_parameters(
                    scale_raw,
                    offset_raw,
                    offset_base,
                    baseline_grid.offset_constraint,
                )
                reconstruction = _reconstruct(
                    relaxed_codes,
                    current_scale,
                    current_offset,
                    baseline_grid,
                    row_start,
                )
                loss = self.objective.loss(
                    problem, reconstruction, imap, row_start=row_start
                ) / float(reconstruction.numel())
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("GSQ relaxed objective became non-finite")
                loss.backward()
                _lion_step(parameter_tuple, moments, self.config)

                if (
                    (step + 1) % self.config.hard_eval_interval == 0
                    or step + 1 == self.config.steps
                ):
                    with torch.no_grad():
                        hard_indices = logits.masked_fill(~valid, -torch.inf).argmax(dim=-1)
                        hard_codes = torch.gather(
                            levels, 2, hard_indices.unsqueeze(-1)
                        ).squeeze(-1)
                        current_scale, current_offset = _decode_parameters(
                            scale_raw,
                            offset_raw,
                            offset_base,
                            baseline_grid.offset_constraint,
                        )
                        hard_reconstruction = _reconstruct(
                            hard_codes,
                            current_scale,
                            current_offset,
                            baseline_grid,
                            row_start,
                        )
                        hard_loss = self.objective.row_losses(
                            problem,
                            hard_reconstruction,
                            imap,
                            row_start=row_start,
                        )
                        improve = hard_loss < best_loss
                        best_loss = torch.where(improve, hard_loss, best_loss)
                        best_codes = torch.where(improve[:, None], hard_codes, best_codes)
                        best_scale = torch.where(
                            improve[:, None], current_scale, best_scale
                        )
                        best_offset = torch.where(
                            improve[:, None], current_offset, best_offset
                        )

            codes[row_start:row_stop, : baseline_grid.value_count] = best_codes.to(torch.int16)
            scale_parameters[row_start:row_stop] = best_scale
            offset_parameters[row_start:row_stop] = best_offset

        candidate_grid = codec.canonicalize(
            baseline_grid.with_values(
                codes=codes,
                scale_parameters=scale_parameters,
                offset_parameters=offset_parameters,
            )
        )
        candidate_reconstruction = candidate_grid.dequantize()
        candidate_value = self.objective.evaluate(
            problem, candidate_reconstruction, imap
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
                "steps": float(self.config.steps),
                "row_chunk_size": float(self.config.row_chunk_size),
                "gptq_initialized": float(self.config.use_gptq_init),
                "learned_scales": float(self.config.learn_scales),
                "learned_offsets": float(
                    self.config.learn_offsets
                    and baseline_grid.offset_constraint != "zero"
                ),
            },
        )


__all__ = ["GsqConfig", "GsqSolver"]
