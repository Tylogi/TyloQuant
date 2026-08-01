"""Exact chunked terminal KL objective for streamed language-model calibration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as functional


class RowMatrix(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...

    def rows(self, start: int, end: int) -> torch.Tensor: ...


HeadWeight = torch.Tensor | RowMatrix


@dataclass(frozen=True)
class TerminalKl:
    sum_kl: float
    positions: int
    gradient: torch.Tensor | None

    @property
    def mean_kl(self) -> float:
        return self.sum_kl / self.positions


@dataclass(frozen=True)
class ChunkedTerminalObjective:
    reference_norm: torch.nn.Module
    candidate_norm: torch.nn.Module
    head_weight: HeadWeight

    def evaluate(
        self,
        reference_hidden: torch.Tensor,
        candidate_hidden: torch.Tensor,
        *,
        row_chunk: int,
        with_gradient: bool,
        gradient_scale: float = 1.0,
    ) -> TerminalKl:
        return chunked_terminal_kl(
            reference_hidden,
            candidate_hidden,
            self.reference_norm,
            self.candidate_norm,
            self.head_weight,
            row_chunk=row_chunk,
            with_gradient=with_gradient,
            gradient_scale=gradient_scale,
        )


def _update_logsumexp(
    logits: torch.Tensor,
    maximum: torch.Tensor,
    denominator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_maximum = logits.max(dim=1).values
    new_maximum = torch.maximum(maximum, local_maximum)
    denominator = denominator * torch.exp(maximum - new_maximum)
    denominator += torch.exp(logits - new_maximum[:, None]).sum(dim=1)
    return new_maximum, denominator


def _head_rows(head_weight: HeadWeight, start: int, end: int) -> torch.Tensor:
    if isinstance(head_weight, torch.Tensor):
        return head_weight[start:end]
    return head_weight.rows(start, end)


def chunked_terminal_kl(
    reference_hidden: torch.Tensor,
    candidate_hidden: torch.Tensor,
    reference_norm: torch.nn.Module,
    candidate_norm: torch.nn.Module,
    head_weight: HeadWeight,
    *,
    row_chunk: int,
    with_gradient: bool,
    gradient_scale: float = 1.0,
) -> TerminalKl:
    """Compute exact teacher-to-candidate KL without materializing full logits."""

    if reference_hidden.shape != candidate_hidden.shape or reference_hidden.ndim != 3:
        raise ValueError("terminal KL hidden states must have the same [batch, seq, hidden] shape")
    if reference_hidden.shape[1] < 2:
        raise ValueError("terminal KL requires at least two sequence positions")
    if len(head_weight.shape) != 2:
        raise ValueError("lm_head weight must be a non-empty matrix")
    rows, columns = (int(head_weight.shape[0]), int(head_weight.shape[1]))
    if rows <= 0:
        raise ValueError("lm_head weight must be a non-empty matrix")
    if columns != reference_hidden.shape[-1]:
        raise ValueError("lm_head width does not match terminal hidden width")
    if row_chunk < 0:
        raise ValueError("lm_head row chunk must be non-negative")
    if row_chunk == 0:
        row_chunk = rows
    if gradient_scale <= 0:
        raise ValueError("gradient_scale must be positive")

    positions = int(reference_hidden.shape[0] * (reference_hidden.shape[1] - 1))
    with torch.no_grad():
        reference_matrix = reference_norm(reference_hidden[:, :-1]).reshape(positions, columns)
    candidate_value = candidate_hidden.detach()
    if with_gradient:
        candidate_value = candidate_value.requires_grad_(True)
    candidate_matrix = candidate_norm(candidate_value[:, :-1]).reshape(positions, columns)

    device = candidate_matrix.device
    reference_maximum = torch.full((positions,), -torch.inf, device=device, dtype=torch.float32)
    candidate_maximum = torch.full_like(reference_maximum, -torch.inf)
    reference_denominator = torch.zeros_like(reference_maximum)
    candidate_denominator = torch.zeros_like(reference_maximum)
    with torch.no_grad():
        for start in range(0, rows, row_chunk):
            end = min(start + row_chunk, rows)
            source = _head_rows(head_weight, start, end)
            reference_weight = source.to(
                device=reference_matrix.device,
                dtype=reference_matrix.dtype,
            )
            candidate_weight = source.to(
                device=candidate_matrix.device,
                dtype=candidate_matrix.dtype,
            )
            reference_logits = functional.linear(reference_matrix, reference_weight).float()
            candidate_logits = functional.linear(
                candidate_matrix.detach(), candidate_weight
            ).float()
            reference_maximum, reference_denominator = _update_logsumexp(
                reference_logits,
                reference_maximum,
                reference_denominator,
            )
            candidate_maximum, candidate_denominator = _update_logsumexp(
                candidate_logits,
                candidate_maximum,
                candidate_denominator,
            )
            del reference_weight, candidate_weight, reference_logits, candidate_logits
    if (
        not torch.isfinite(reference_maximum).all()
        or not torch.isfinite(candidate_maximum).all()
        or not torch.isfinite(reference_denominator).all()
        or not torch.isfinite(candidate_denominator).all()
        or torch.any(reference_denominator <= 0)
        or torch.any(candidate_denominator <= 0)
    ):
        raise FloatingPointError("non-finite terminal KL softmax state")

    reference_log_partition = reference_maximum + reference_denominator.log()
    candidate_log_partition = candidate_maximum + candidate_denominator.log()
    per_position_kl = torch.zeros_like(reference_maximum)
    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)
        source = _head_rows(head_weight, start, end)
        reference_weight = source.to(
            device=reference_matrix.device,
            dtype=reference_matrix.dtype,
        )
        candidate_weight = source.to(
            device=candidate_matrix.device,
            dtype=candidate_matrix.dtype,
        )
        with torch.no_grad():
            reference_logits = functional.linear(reference_matrix, reference_weight).float()
            reference_log_probability = reference_logits - reference_log_partition[:, None]
            reference_probability = torch.exp(reference_log_probability)
        if with_gradient:
            candidate_logits = functional.linear(candidate_matrix, candidate_weight).float()
        else:
            with torch.no_grad():
                candidate_logits = functional.linear(candidate_matrix, candidate_weight).float()
        with torch.no_grad():
            candidate_log_probability = candidate_logits.detach() - candidate_log_partition[:, None]
            candidate_probability = torch.exp(candidate_log_probability)
            per_position_kl += (
                reference_probability * (reference_log_probability - candidate_log_probability)
            ).sum(dim=1)
            coefficients = (candidate_probability - reference_probability) * gradient_scale
        if with_gradient:
            torch.autograd.backward(
                candidate_logits,
                coefficients,
                retain_graph=end < rows,
            )
        del (
            reference_weight,
            candidate_weight,
            reference_logits,
            reference_log_probability,
            reference_probability,
            candidate_logits,
            candidate_log_probability,
            candidate_probability,
            coefficients,
        )

    if not torch.isfinite(per_position_kl).all():
        raise FloatingPointError("non-finite terminal KL value")
    gradient = None
    if with_gradient:
        gradient = candidate_value.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise FloatingPointError("non-finite terminal KL hidden gradient")
        gradient = gradient.detach()
    return TerminalKl(float(per_position_kl.sum().item()), positions, gradient)


__all__ = [
    "ChunkedTerminalObjective",
    "HeadWeight",
    "RowMatrix",
    "TerminalKl",
    "chunked_terminal_kl",
]
