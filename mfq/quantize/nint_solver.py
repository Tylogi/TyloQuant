"""NINT adapter for architecture-neutral GPTQ and GSQ weight solvers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mfq.formats.nint import (
    NintSpec,
    NintTensor,
    _uint_dtype,
    normalize_row_q_bits,
    normalize_row_sub_bits,
)
from mfq.quantize.weight_solver import (
    ImportanceMap,
    ScalarGridCodec,
    ScalarGridTensor,
    WeightSolverProblem,
    WeightSolverResult,
)


class NintGridCodec(ScalarGridCodec):
    """Expose NINT's two-level affine representation as a tied scalar grid.

    The subgroup integer metadata is initialized by the existing NINT solver.
    GPTQ then changes only value codes. GSQ may additionally learn the shared
    neuron anchors; the group multipliers remain integral and fixed, so every
    intermediate and final hard grid preserves NINT's runtime representation.
    """

    def __init__(
        self,
        spec: NintSpec,
        *,
        row_q_bits: np.ndarray | None = None,
        row_sub_bits: np.ndarray | None = None,
        use_priority_group_refinement: bool = True,
    ) -> None:
        self.spec = spec
        self.row_q_bits = row_q_bits
        self.row_sub_bits = row_sub_bits
        self.use_priority_group_refinement = bool(use_priority_group_refinement)

    def initialize(
        self,
        problem: WeightSolverProblem,
        imap: ImportanceMap | None = None,
    ) -> ScalarGridTensor:
        rows, columns = map(int, problem.weight.shape)
        row_q_bits = normalize_row_q_bits(self.spec, self.row_q_bits, rows)
        row_sub_bits = normalize_row_sub_bits(self.spec, self.row_sub_bits, rows)
        importance = (
            None
            if imap is None
            else np.ascontiguousarray(
                imap.channel_importance(slice(0, rows)), dtype=np.float32
            )
        )
        if problem.weight.device.type in {"cuda", "mps"}:
            from mfq.quantize.nint_quant_torch import quantize_axis0

            encoded = quantize_axis0(
                problem.weight,
                self.spec,
                device=problem.weight.device,
                importance=importance,
                use_priority_group_refinement=self.use_priority_group_refinement,
                row_q_bits=row_q_bits,
                row_sub_bits=row_sub_bits,
            )
        else:
            from mfq.quantize.nint_quant import quantize

            encoded = quantize(
                problem.weight.detach().cpu().numpy(),
                self.spec,
                axis=0,
                importance=importance,
                use_priority_group_refinement=self.use_priority_group_refinement,
                row_q_bits=row_q_bits,
                row_sub_bits=row_sub_bits,
            )
        device = problem.weight.device
        groups = int(encoded.q.shape[1])
        zeros = torch.zeros((rows, groups), device=device, dtype=torch.int64)
        return ScalarGridTensor(
            codes=torch.as_tensor(
                encoded.q.reshape(rows, -1), device=device, dtype=torch.int16
            ),
            q_bits=torch.as_tensor(row_q_bits, device=device, dtype=torch.int64),
            group_size=int(self.spec.groupsize),
            value_count=columns,
            scale_parameters=torch.as_tensor(
                encoded.neuron_scale[:, None], device=device, dtype=torch.float32
            ),
            scale_indices=zeros,
            scale_multipliers=torch.as_tensor(
                encoded.sub_scale, device=device, dtype=torch.float32
            ),
            offset_parameters=-torch.as_tensor(
                encoded.neuron_min[:, None], device=device, dtype=torch.float32
            ),
            offset_indices=zeros,
            offset_multipliers=torch.as_tensor(
                encoded.sub_min, device=device, dtype=torch.float32
            ),
            scale_constraint="positive",
            offset_constraint="nonpositive",
            metadata={
                "codec": "nint-grid-v1",
                "symmetric": False,
                "spec": self.spec,
                "shape": tuple(encoded.shape),
                "axis": int(encoded.axis),
                "row_q_bits": np.ascontiguousarray(row_q_bits, dtype=np.uint8),
                "row_sub_bits": np.ascontiguousarray(row_sub_bits, dtype=np.uint8),
                "sub_scale": np.ascontiguousarray(encoded.sub_scale),
                "sub_min": np.ascontiguousarray(encoded.sub_min),
            },
        )

    def _validate_grid(self, grid: ScalarGridTensor) -> None:
        if grid.metadata.get("codec") != "nint-grid-v1":
            raise TypeError("NINT codec received a scalar grid from another format")
        if grid.group_size != self.spec.groupsize:
            raise ValueError("NINT scalar grid changed its fixed group size")
        if grid.scale_parameters.shape[1] != 1 or grid.offset_parameters.shape[1] != 1:
            raise ValueError("NINT scalar grid must retain one pair of anchors per neuron")

    def canonicalize(self, grid: ScalarGridTensor) -> ScalarGridTensor:
        self._validate_grid(grid)
        scale = torch.clamp(grid.scale_parameters, min=0).to(torch.float16).to(torch.float32)
        minimum = (
            torch.clamp(-grid.offset_parameters, min=0)
            .to(torch.float16)
            .to(torch.float32)
        )
        qmin, qmax = grid.q_limits()
        codes = torch.maximum(
            torch.minimum(grid.codes, qmax[:, None]), qmin[:, None]
        )
        return grid.with_values(
            codes=codes,
            scale_parameters=scale,
            offset_parameters=-minimum,
        )

    def finalize(self, grid: ScalarGridTensor) -> NintTensor:
        grid = self.canonicalize(grid)
        metadata = grid.metadata
        row_q_bits = np.ascontiguousarray(metadata["row_q_bits"], dtype=np.uint8)
        row_sub_bits = np.ascontiguousarray(metadata["row_sub_bits"], dtype=np.uint8)
        maximum_q = (1 << int(row_q_bits.max())) - 1
        maximum_k = (1 << int(row_sub_bits.max())) - 1
        return NintTensor(
            spec=self.spec,
            shape=tuple(metadata["shape"]),
            axis=int(metadata["axis"]),
            q=grid.codes.detach().cpu().numpy().reshape(
                grid.rows, grid.groups, grid.group_size
            ).astype(_uint_dtype(maximum_q), copy=False),
            neuron_scale=grid.scale_parameters[:, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False),
            neuron_min=(-grid.offset_parameters[:, 0])
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False),
            sub_scale=np.asarray(metadata["sub_scale"]).astype(
                _uint_dtype(maximum_k), copy=False
            ),
            sub_min=np.asarray(metadata["sub_min"]).astype(
                _uint_dtype(maximum_k), copy=False
            ),
            neuron_len=grid.value_count,
            row_q_bits=row_q_bits,
            row_sub_bits=row_sub_bits,
        )


def quantize_nint_gptq(
    weight: torch.Tensor | np.ndarray,
    spec: NintSpec,
    *,
    calibration_inputs: torch.Tensor | np.ndarray | None = None,
    hessian: torch.Tensor | np.ndarray | None = None,
    imap: ImportanceMap | None = None,
    row_q_bits: np.ndarray | None = None,
    row_sub_bits: np.ndarray | None = None,
    tensor_key: str = "anonymous.weight",
    config: Any = None,
) -> WeightSolverResult:
    from mfq.quantize.gptq import GptqSolver

    problem = WeightSolverProblem(
        tensor_key,
        weight,
        calibration_inputs=calibration_inputs,
        hessian=hessian,
    )
    codec = NintGridCodec(
        spec, row_q_bits=row_q_bits, row_sub_bits=row_sub_bits
    )
    return GptqSolver(config=config).solve(problem, codec, imap)


def quantize_nint_gsq(
    weight: torch.Tensor | np.ndarray,
    spec: NintSpec,
    *,
    calibration_inputs: torch.Tensor | np.ndarray | None = None,
    hessian: torch.Tensor | np.ndarray | None = None,
    imap: ImportanceMap | None = None,
    row_q_bits: np.ndarray | None = None,
    row_sub_bits: np.ndarray | None = None,
    tensor_key: str = "anonymous.weight",
    config: Any = None,
    gptq_config: Any = None,
) -> WeightSolverResult:
    from mfq.quantize.gsq import GsqSolver

    problem = WeightSolverProblem(
        tensor_key,
        weight,
        calibration_inputs=calibration_inputs,
        hessian=hessian,
    )
    codec = NintGridCodec(
        spec, row_q_bits=row_q_bits, row_sub_bits=row_sub_bits
    )
    return GsqSolver(config=config, gptq_config=gptq_config).solve(
        problem, codec, imap
    )


__all__ = ["NintGridCodec", "quantize_nint_gptq", "quantize_nint_gsq"]
