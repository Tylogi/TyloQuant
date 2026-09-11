"""Stable, architecture-neutral contract for scalar-grid weight solvers.

The classes in this module are intentionally smaller than the MFQ Core v2
workflow layer.  They are leaf components that can live on ``master`` while a
larger workflow migration proceeds independently.  Core v2 can use a
``WeightSolver`` directly as its WSolver implementation.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import torch


class ImportanceMap(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    def channel_importance(self, rows) -> np.ndarray: ...

    def neuron_importance(self, rows) -> np.ndarray: ...

    def element_importance(self, rows) -> np.ndarray: ...

    def fingerprint(self) -> str: ...


def _as_float_tensor(value: torch.Tensor | np.ndarray, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


@dataclass(frozen=True)
class WeightSolverProblem:
    """One canonical matrix and the evidence available to a weight solver."""

    tensor_key: str
    weight: torch.Tensor | np.ndarray
    calibration_inputs: torch.Tensor | np.ndarray | None = None
    hessian: torch.Tensor | np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tensor_key:
            raise ValueError("weight solver problems require a canonical tensor key")
        weight = _as_float_tensor(self.weight, name="weight")
        if weight.ndim != 2 or not weight.shape[0] or not weight.shape[1]:
            raise ValueError("weight solver problems require a non-empty [out, in] matrix")
        inputs = self.calibration_inputs
        if inputs is not None:
            inputs = _as_float_tensor(inputs, name="calibration inputs")
            if inputs.ndim != 2 or inputs.shape[1] != weight.shape[1] or not inputs.shape[0]:
                raise ValueError(
                    "calibration inputs must have shape [observations, input_width]"
                )
        hessian = self.hessian
        if hessian is not None:
            hessian = _as_float_tensor(hessian, name="Hessian")
            expected = (int(weight.shape[1]), int(weight.shape[1]))
            if tuple(hessian.shape) != expected:
                raise ValueError(f"Hessian shape {tuple(hessian.shape)} != {expected}")
            if not torch.allclose(hessian, hessian.mT, rtol=1e-4, atol=1e-6):
                raise ValueError("weight solver Hessian must be symmetric")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "calibration_inputs", inputs)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ObjectiveValue:
    total: float
    components: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.total):
            raise ValueError("objective values must be finite")
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))


class ReconstructionObjective(ABC):
    """Executable reconstruction criterion; it owns no update algorithm."""

    @abstractmethod
    def row_losses(
        self,
        problem: WeightSolverProblem,
        candidate: torch.Tensor,
        imap: ImportanceMap | None = None,
        *,
        row_start: int = 0,
    ) -> torch.Tensor:
        """Return one differentiable loss per candidate output row."""

    def loss(
        self,
        problem: WeightSolverProblem,
        candidate: torch.Tensor,
        imap: ImportanceMap | None = None,
        *,
        row_start: int = 0,
    ) -> torch.Tensor:
        return self.row_losses(problem, candidate, imap, row_start=row_start).sum()

    def evaluate(
        self,
        problem: WeightSolverProblem,
        candidate: torch.Tensor,
        imap: ImportanceMap | None = None,
    ) -> ObjectiveValue:
        with torch.no_grad():
            rows = self.row_losses(problem, candidate, imap)
            total = float(rows.sum().detach().cpu())
            return ObjectiveValue(total, {"mean_row_loss": float(rows.mean().cpu())})

    @abstractmethod
    def fingerprint(self) -> str:
        """Return stable executable-semantics provenance."""


class QuadraticReconstructionObjective(ReconstructionObjective):
    """Layer-output reconstruction with NAQ factorized-SSE fallback.

    A supplied full Hessian or activation matrix evaluates the usual GPTQ/GSQ
    layer-output objective.  NAQ neuron importance remains an independent row
    multiplier.  If no full second-order evidence is present, the objective
    falls back to the factorized NAQ input-channel and neuron factors.
    """

    def __init__(self, *, normalize: bool = True) -> None:
        self.normalize = bool(normalize)

    def hessian(
        self,
        problem: WeightSolverProblem,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if problem.hessian is not None:
            return problem.hessian.to(device=device, dtype=dtype)
        if problem.calibration_inputs is None:
            return None
        inputs = problem.calibration_inputs.to(device=device, dtype=dtype)
        result = inputs.mT @ inputs
        if self.normalize:
            result = result / float(inputs.shape[0])
        return result

    def gptq_hessian(
        self,
        problem: WeightSolverProblem,
        imap: ImportanceMap | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        result = self.hessian(problem, device=device, dtype=dtype)
        if result is not None:
            return result
        columns = int(problem.weight.shape[1])
        if imap is None:
            return torch.eye(columns, device=device, dtype=dtype)
        diagonal = np.asarray(
            imap.channel_importance(slice(0, int(problem.weight.shape[0]))),
            dtype=np.float64,
        ).mean(axis=0)
        diagonal = np.maximum(diagonal, np.finfo(np.float32).tiny)
        return torch.diag(torch.as_tensor(diagonal, device=device, dtype=dtype))

    def row_losses(
        self,
        problem: WeightSolverProblem,
        candidate: torch.Tensor,
        imap: ImportanceMap | None = None,
        *,
        row_start: int = 0,
    ) -> torch.Tensor:
        candidate = candidate.to(dtype=torch.float32)
        row_stop = row_start + int(candidate.shape[0])
        reference = problem.weight[row_start:row_stop].to(
            device=candidate.device, dtype=candidate.dtype
        )
        if tuple(candidate.shape) != tuple(reference.shape):
            raise ValueError(
                f"candidate shape {tuple(candidate.shape)} != {tuple(reference.shape)}"
            )
        error = candidate - reference
        if problem.hessian is not None:
            hessian = problem.hessian.to(
                device=candidate.device, dtype=candidate.dtype
            )
            row_loss = torch.einsum("ri,ij,rj->r", error, hessian, error)
        elif problem.calibration_inputs is not None:
            inputs = problem.calibration_inputs.to(
                device=candidate.device, dtype=candidate.dtype
            )
            output_error = error @ inputs.mT
            row_loss = output_error.square().sum(dim=1)
            if self.normalize:
                row_loss = row_loss / float(inputs.shape[0])
        else:
            row_loss = None
        if row_loss is not None:
            if imap is not None:
                neuron = torch.as_tensor(
                    imap.neuron_importance(slice(row_start, row_stop)),
                    device=candidate.device,
                    dtype=candidate.dtype,
                )
                row_loss = row_loss * neuron
            return row_loss
        if imap is None:
            return error.square().sum(dim=1)
        importance = torch.as_tensor(
            imap.element_importance(slice(row_start, row_stop)),
            device=candidate.device,
            dtype=candidate.dtype,
        )
        return (importance * error.square()).sum(dim=1)

    def fingerprint(self) -> str:
        payload = f"mfq.quadratic-reconstruction.v1:normalize={int(self.normalize)}"
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


def normalize_q_bits(q_bits: int | np.ndarray | torch.Tensor, rows: int) -> torch.Tensor:
    values = torch.as_tensor(q_bits, dtype=torch.int64).reshape(-1)
    if values.numel() == 1:
        values = values.expand(rows).clone()
    if tuple(values.shape) != (rows,) or bool(((values < 1) | (values > 8)).any()):
        raise ValueError("scalar-grid q bits must contain one value in [1, 8] per row")
    return values


@dataclass(frozen=True)
class ScalarGridTensor:
    """Integer codes plus a possibly tied affine grid parameterization.

    Parameter indices and multipliers express both ordinary independent
    per-group scales and NINT's neuron-anchored two-level scales.  GSQ can learn
    the underlying parameters without leaving the target format's representable
    family.
    """

    codes: torch.Tensor
    q_bits: torch.Tensor
    group_size: int
    value_count: int
    scale_parameters: torch.Tensor
    scale_indices: torch.Tensor
    scale_multipliers: torch.Tensor
    offset_parameters: torch.Tensor
    offset_indices: torch.Tensor
    offset_multipliers: torch.Tensor
    scale_constraint: str = "positive"
    offset_constraint: str = "free"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        codes = torch.as_tensor(self.codes)
        q_bits = torch.as_tensor(self.q_bits, dtype=torch.int64, device=codes.device)
        if codes.ndim != 2 or not codes.shape[0] or not codes.shape[1]:
            raise ValueError("scalar-grid codes must be a non-empty matrix")
        if self.group_size <= 0 or codes.shape[1] % self.group_size:
            raise ValueError("scalar-grid code width must be divisible by group size")
        if self.value_count <= 0 or self.value_count > codes.shape[1]:
            raise ValueError("scalar-grid logical value count is invalid")
        rows = int(codes.shape[0])
        groups = int(codes.shape[1] // self.group_size)
        if tuple(q_bits.shape) != (rows,) or bool(((q_bits < 1) | (q_bits > 8)).any()):
            raise ValueError("scalar-grid q bits must match rows and lie in [1, 8]")
        scale_parameters = torch.as_tensor(
            self.scale_parameters, dtype=torch.float32, device=codes.device
        )
        offset_parameters = torch.as_tensor(
            self.offset_parameters, dtype=torch.float32, device=codes.device
        )
        scale_indices = torch.as_tensor(
            self.scale_indices, dtype=torch.int64, device=codes.device
        )
        offset_indices = torch.as_tensor(
            self.offset_indices, dtype=torch.int64, device=codes.device
        )
        scale_multipliers = torch.as_tensor(
            self.scale_multipliers, dtype=torch.float32, device=codes.device
        )
        offset_multipliers = torch.as_tensor(
            self.offset_multipliers, dtype=torch.float32, device=codes.device
        )
        if (
            scale_parameters.ndim != 2
            or offset_parameters.ndim != 2
            or scale_parameters.shape[0] != rows
            or offset_parameters.shape[0] != rows
        ):
            raise ValueError("scalar-grid parameter rows must match code rows")
        expected = (rows, groups)
        for name, value in (
            ("scale indices", scale_indices),
            ("offset indices", offset_indices),
            ("scale multipliers", scale_multipliers),
            ("offset multipliers", offset_multipliers),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"scalar-grid {name} shape {tuple(value.shape)} != {expected}")
        if (
            bool((scale_indices < 0).any())
            or bool((scale_indices >= scale_parameters.shape[1]).any())
            or bool((offset_indices < 0).any())
            or bool((offset_indices >= offset_parameters.shape[1]).any())
        ):
            raise ValueError("scalar-grid parameter index is out of range")
        if self.scale_constraint not in {"positive", "free"}:
            raise ValueError("unsupported scalar-grid scale constraint")
        if self.offset_constraint not in {"free", "nonpositive", "zero"}:
            raise ValueError("unsupported scalar-grid offset constraint")
        if (
            not bool(torch.isfinite(scale_parameters).all())
            or not bool(torch.isfinite(offset_parameters).all())
            or not bool(torch.isfinite(scale_multipliers).all())
            or not bool(torch.isfinite(offset_multipliers).all())
        ):
            raise ValueError("scalar-grid parameters must be finite")
        if self.scale_constraint == "positive" and bool((scale_parameters < 0).any()):
            raise ValueError("positive scalar-grid scales cannot be negative")
        if self.offset_constraint == "nonpositive" and bool((offset_parameters > 0).any()):
            raise ValueError("nonpositive scalar-grid offsets cannot be positive")
        if self.offset_constraint == "zero" and bool((offset_parameters != 0).any()):
            raise ValueError("zero-constrained scalar-grid offsets must be zero")
        object.__setattr__(self, "codes", codes.to(torch.int16))
        object.__setattr__(self, "q_bits", q_bits)
        object.__setattr__(self, "scale_parameters", scale_parameters)
        object.__setattr__(self, "scale_indices", scale_indices)
        object.__setattr__(self, "scale_multipliers", scale_multipliers)
        object.__setattr__(self, "offset_parameters", offset_parameters)
        object.__setattr__(self, "offset_indices", offset_indices)
        object.__setattr__(self, "offset_multipliers", offset_multipliers)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        qmin, qmax = self.q_limits()
        if bool((self.codes < qmin[:, None]).any()) or bool((self.codes > qmax[:, None]).any()):
            raise ValueError("scalar-grid code is outside its row precision")

    @property
    def rows(self) -> int:
        return int(self.codes.shape[0])

    @property
    def padded_count(self) -> int:
        return int(self.codes.shape[1])

    @property
    def groups(self) -> int:
        return self.padded_count // self.group_size

    def q_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        symmetric = bool(self.metadata.get("symmetric", False))
        if symmetric:
            qmin = -(1 << (self.q_bits - 1))
            qmax = (1 << (self.q_bits - 1)) - 1
        else:
            qmin = torch.zeros_like(self.q_bits)
            qmax = (1 << self.q_bits) - 1
        return qmin, qmax

    def effective_scales(self) -> torch.Tensor:
        return torch.gather(self.scale_parameters, 1, self.scale_indices) * self.scale_multipliers

    def effective_offsets(self) -> torch.Tensor:
        return torch.gather(self.offset_parameters, 1, self.offset_indices) * self.offset_multipliers

    def dequantize(self) -> torch.Tensor:
        grouped = self.codes.to(torch.float32).reshape(self.rows, self.groups, self.group_size)
        values = (
            grouped * self.effective_scales().unsqueeze(-1)
            + self.effective_offsets().unsqueeze(-1)
        )
        return values.reshape(self.rows, self.padded_count)[:, : self.value_count]

    def with_values(self, **changes: Any) -> ScalarGridTensor:
        return replace(self, **changes)

    def clone(self) -> ScalarGridTensor:
        return replace(
            self,
            codes=self.codes.clone(),
            q_bits=self.q_bits.clone(),
            scale_parameters=self.scale_parameters.clone(),
            scale_indices=self.scale_indices.clone(),
            scale_multipliers=self.scale_multipliers.clone(),
            offset_parameters=self.offset_parameters.clone(),
            offset_indices=self.offset_indices.clone(),
            offset_multipliers=self.offset_multipliers.clone(),
        )


class ScalarGridCodec(ABC):
    """Format-specific construction and sealing around a common scalar grid."""

    @abstractmethod
    def initialize(
        self,
        problem: WeightSolverProblem,
        imap: ImportanceMap | None = None,
    ) -> ScalarGridTensor: ...

    def canonicalize(self, grid: ScalarGridTensor) -> ScalarGridTensor:
        """Project parameters to the exact values representable by the format."""

        return grid

    @abstractmethod
    def finalize(self, grid: ScalarGridTensor) -> Any: ...


class UniformAffineCodec(ScalarGridCodec):
    """Independent affine or signed-symmetric scalar groups."""

    def __init__(
        self,
        q_bits: int | np.ndarray | torch.Tensor,
        group_size: int,
        *,
        symmetric: bool = False,
        fit_iterations: int = 4,
    ) -> None:
        if group_size <= 0 or fit_iterations < 0:
            raise ValueError("invalid affine codec configuration")
        self.q_bits = q_bits
        self.group_size = int(group_size)
        self.symmetric = bool(symmetric)
        self.fit_iterations = int(fit_iterations)

    def initialize(
        self,
        problem: WeightSolverProblem,
        imap: ImportanceMap | None = None,
    ) -> ScalarGridTensor:
        weight = problem.weight.to(torch.float32)
        rows, columns = map(int, weight.shape)
        q_bits = normalize_q_bits(self.q_bits, rows).to(weight.device)
        padding = (-columns) % self.group_size
        padded = torch.nn.functional.pad(weight, (0, padding)) if padding else weight
        groups = int(padded.shape[1] // self.group_size)
        grouped = padded.reshape(rows, groups, self.group_size)
        valid_elements = (
            torch.arange(padded.shape[1], device=weight.device)[None, :]
            < columns
        ).reshape(1, groups, self.group_size)
        if imap is None:
            importance = torch.ones_like(grouped)
        else:
            importance = torch.as_tensor(
                imap.element_importance(slice(0, rows)),
                device=weight.device,
                dtype=torch.float32,
            )
            if padding:
                importance = torch.nn.functional.pad(importance, (0, padding))
            importance = importance.reshape_as(grouped)
        if padding:
            importance[:, -1, self.group_size - padding :] = 0

        if self.symmetric:
            qmin = (-(1 << (q_bits - 1))).to(torch.float32)[:, None]
            qmax = ((1 << (q_bits - 1)) - 1).to(torch.float32)[:, None]
            negative = grouped.masked_fill(~valid_elements, torch.inf).amin(dim=-1)
            positive = grouped.masked_fill(~valid_elements, -torch.inf).amax(dim=-1)
            scale = torch.maximum(
                negative.abs() / torch.clamp(qmin.abs(), min=1),
                positive.abs() / torch.clamp(qmax, min=1),
            )
            offset = torch.zeros_like(scale)
        else:
            qmin = torch.zeros((rows, 1), device=weight.device)
            qmax = ((1 << q_bits) - 1).to(torch.float32)[:, None]
            minimum = grouped.masked_fill(~valid_elements, torch.inf).amin(dim=-1)
            maximum = grouped.masked_fill(~valid_elements, -torch.inf).amax(dim=-1)
            scale = (maximum - minimum) / torch.clamp(qmax - qmin, min=1)
            offset = minimum - scale * qmin
        tiny = torch.finfo(torch.float32).tiny
        scale = torch.clamp(scale, min=tiny)

        for _ in range(self.fit_iterations + 1):
            levels = torch.clamp(
                torch.round((grouped - offset.unsqueeze(-1)) / scale.unsqueeze(-1)),
                qmin[:, :, None],
                qmax[:, :, None],
            )
            if _ == self.fit_iterations:
                break
            if self.symmetric:
                numerator = (importance * grouped * levels).sum(dim=-1)
                denominator = (importance * levels.square()).sum(dim=-1)
                candidate_scale = numerator / torch.clamp(denominator, min=tiny)
                scale = torch.where(
                    (denominator > 0) & (candidate_scale > 0), candidate_scale, scale
                )
                continue
            sum_w = importance.sum(dim=-1)
            sum_q = (importance * levels).sum(dim=-1)
            sum_q2 = (importance * levels.square()).sum(dim=-1)
            sum_x = (importance * grouped).sum(dim=-1)
            sum_qx = (importance * levels * grouped).sum(dim=-1)
            determinant = sum_w * sum_q2 - sum_q.square()
            valid = determinant > tiny
            divisor = torch.where(valid, determinant, torch.ones_like(determinant))
            candidate_scale = (sum_w * sum_qx - sum_q * sum_x) / divisor
            candidate_offset = (sum_q2 * sum_x - sum_q * sum_qx) / divisor
            valid = valid & (candidate_scale > tiny)
            scale = torch.where(valid, candidate_scale, scale)
            offset = torch.where(valid, candidate_offset, offset)

        indices = torch.arange(groups, device=weight.device, dtype=torch.int64)[None, :].expand(rows, -1)
        ones = torch.ones((rows, groups), device=weight.device, dtype=torch.float32)
        return ScalarGridTensor(
            codes=levels.reshape(rows, -1).to(torch.int16),
            q_bits=q_bits,
            group_size=self.group_size,
            value_count=columns,
            scale_parameters=scale,
            scale_indices=indices,
            scale_multipliers=ones,
            offset_parameters=offset,
            offset_indices=indices,
            offset_multipliers=ones,
            scale_constraint="positive",
            offset_constraint="zero" if self.symmetric else "free",
            metadata={"codec": "uniform-affine-v1", "symmetric": self.symmetric},
        )

    def finalize(self, grid: ScalarGridTensor) -> ScalarGridTensor:
        return grid


@dataclass(frozen=True)
class WeightSolverResult:
    encoded: Any
    grid: ScalarGridTensor
    reconstruction: torch.Tensor
    objective: ObjectiveValue
    baseline: ObjectiveValue
    row_losses: torch.Tensor
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


class WeightSolver(ABC):
    """Updates encoded weights while q widths and group size stay fixed."""

    objective: ReconstructionObjective

    @abstractmethod
    def solve(
        self,
        problem: WeightSolverProblem,
        codec: ScalarGridCodec,
        imap: ImportanceMap | None = None,
        *,
        initial: ScalarGridTensor | None = None,
    ) -> WeightSolverResult: ...


__all__ = [
    "ImportanceMap",
    "ObjectiveValue",
    "QuadraticReconstructionObjective",
    "ReconstructionObjective",
    "ScalarGridCodec",
    "ScalarGridTensor",
    "UniformAffineCodec",
    "WeightSolver",
    "WeightSolverProblem",
    "WeightSolverResult",
    "normalize_q_bits",
]
