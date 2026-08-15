"""Accelerator-backed NVQ1-S offline quantization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from mfq.formats.nvq1_s import (
    NVQ1_S,
    NVQ1_S_SYNTHETIC_BANKS,
    Nvq1SSpec,
    Nvq1STensor,
    validate_nvq1_s_banked_codebook,
    validate_nvq1_s_codebook,
)
from mfq.quantize.nvq_quant_torch import _fp16_round, _pad_weight, _prepare_weight


def _basis(
    sub_scale: torch.Tensor,
    delta_sign: torch.Tensor,
    indices: torch.Tensor,
    spec: Nvq1SSpec,
    codebooks: torch.Tensor,
) -> torch.Tensor:
    banks = delta_sign.to(torch.int64).unsqueeze(1).expand_as(indices)
    code = codebooks[banks, indices].reshape(-1, spec.groupsize)
    delta = torch.where(
        delta_sign != 0,
        torch.full_like(delta_sign, -spec.delta, dtype=torch.float32),
        torch.full_like(delta_sign, spec.delta, dtype=torch.float32),
    )
    return sub_scale.to(torch.float32).unsqueeze(1) * (
        code + delta.unsqueeze(1)
    )


def _refit_anchor(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    sub_scale: torch.Tensor,
    delta_sign: torch.Tensor,
    indices: torch.Tensor,
    *,
    out: int,
    groups_per_row: int,
    spec: Nvq1SSpec,
    codebooks: torch.Tensor,
) -> torch.Tensor:
    basis = _basis(sub_scale, delta_sign, indices, spec, codebooks)
    width = groups_per_row * spec.groupsize
    numerator = (wgroup * xgroup * basis).reshape(out, width).sum(1)
    denominator = (wgroup * basis.square()).reshape(out, width).sum(1)
    anchor = torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    return _fp16_round(anchor.clamp_min(0))


def _row_error(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    sub_scale: torch.Tensor,
    delta_sign: torch.Tensor,
    indices: torch.Tensor,
    anchor: torch.Tensor,
    *,
    out: int,
    groups_per_row: int,
    spec: Nvq1SSpec,
    codebooks: torch.Tensor,
) -> torch.Tensor:
    basis = _basis(sub_scale, delta_sign, indices, spec, codebooks)
    reconstruction = basis * anchor.repeat_interleave(groups_per_row).unsqueeze(1)
    return (wgroup * (xgroup - reconstruction).square()).reshape(
        out, groups_per_row * spec.groupsize
    ).sum(1)


def _solve(
    xgroup: torch.Tensor,
    wgroup: torch.Tensor,
    initial_anchor: torch.Tensor,
    *,
    out: int,
    groups_per_row: int,
    valid_last: int,
    spec: Nvq1SSpec,
    codebooks: torch.Tensor,
    native_codebooks: torch.Tensor,
    refine_steps: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from mfq.quantize.metal.nvq import nvq1_s_assign

    anchor = _fp16_round(initial_anchor)

    def assign(current_anchor: torch.Tensor):
        return nvq1_s_assign(
            xgroup,
            wgroup,
            current_anchor.repeat_interleave(groups_per_row),
            native_codebooks,
            groups_per_row,
            valid_last,
            spec.delta,
        )

    scale, delta, indices = assign(anchor)
    error = _row_error(
        xgroup,
        wgroup,
        scale,
        delta,
        indices,
        anchor,
        out=out,
        groups_per_row=groups_per_row,
        spec=spec,
        codebooks=codebooks,
    )
    for _ in range(refine_steps):
        candidate_anchor = _refit_anchor(
            xgroup,
            wgroup,
            scale,
            delta,
            indices,
            out=out,
            groups_per_row=groups_per_row,
            spec=spec,
            codebooks=codebooks,
        )
        candidate_scale, candidate_delta, candidate_indices = assign(
            candidate_anchor
        )
        candidate_error = _row_error(
            xgroup,
            wgroup,
            candidate_scale,
            candidate_delta,
            candidate_indices,
            candidate_anchor,
            out=out,
            groups_per_row=groups_per_row,
            spec=spec,
            codebooks=codebooks,
        )
        improve = candidate_error < error
        group_improve = improve.repeat_interleave(groups_per_row)
        scale = torch.where(group_improve, candidate_scale, scale)
        delta = torch.where(group_improve, candidate_delta, delta)
        indices = torch.where(
            group_improve.unsqueeze(1), candidate_indices, indices
        )
        anchor = torch.where(improve, candidate_anchor, anchor)
        error = torch.where(improve, candidate_error, error)
    return scale, delta, indices, anchor, error


@torch.inference_mode()
def quantize_axis0(
    weight: torch.Tensor,
    spec: Nvq1SSpec = NVQ1_S,
    *,
    device: str | torch.device = "mps",
    importance: np.ndarray | torch.Tensor | None = None,
    codebook: np.ndarray = NVQ1_S_SYNTHETIC_BANKS,
    anchor_multipliers: Sequence[float] = (0.75, 1.0, 1.25),
    refine_steps: int = 2,
) -> Nvq1STensor:
    """Quantize a matrix with native Metal NVQ1-S assignment."""

    target = torch.device(device)
    if target.type != "mps":
        raise ValueError("native NVQ1-S accelerator quantization requires MPS")
    if refine_steps < 0:
        raise ValueError("refine_steps must be non-negative")
    multipliers = tuple(float(value) for value in anchor_multipliers)
    if not multipliers or any(
        not np.isfinite(value) or value <= 0 for value in multipliers
    ):
        raise ValueError("anchor_multipliers must be finite and positive")
    raw_table = np.asarray(codebook)
    table = (
        validate_nvq1_s_banked_codebook(raw_table)
        if raw_table.ndim == 3
        else np.stack(
            (validate_nvq1_s_codebook(raw_table),) * 2,
            axis=0,
        )
    )
    value, out, neuron_len = _prepare_weight(weight, target)
    if neuron_len % spec.vector_size:
        raise ValueError("NVQ1-S neuron length must be divisible by 8")
    value, objective_weight, groups_per_row = _pad_weight(
        value, spec.groupsize, importance
    )
    xgroup = value.reshape(out * groups_per_row, spec.groupsize).contiguous()
    wgroup = objective_weight.reshape_as(xgroup).contiguous()
    codebooks = torch.as_tensor(
        table, device=target, dtype=torch.float32
    ).contiguous()
    native_codebooks = codebooks.to(torch.int8).contiguous()
    qmax = (1 << spec.sub_bits) - 1
    row_peak = value.abs().amax(1)
    base_anchor = torch.where(
        row_peak > 0,
        row_peak / float(qmax * (1.0 + spec.delta)),
        torch.zeros_like(row_peak),
    )
    best_error = torch.full((out,), torch.inf, device=target)
    best_scale = torch.zeros(
        (out, groups_per_row), device=target, dtype=torch.uint8
    )
    best_delta = torch.zeros_like(best_scale)
    best_indices = torch.zeros(
        (out, groups_per_row, 3), device=target, dtype=torch.int64
    )
    best_anchor = torch.zeros(out, device=target)
    valid_last = neuron_len - (groups_per_row - 1) * spec.groupsize
    for multiplier in multipliers:
        scale, delta, indices, anchor, error = _solve(
            xgroup,
            wgroup,
            base_anchor * multiplier,
            out=out,
            groups_per_row=groups_per_row,
            valid_last=valid_last,
            spec=spec,
            codebooks=codebooks,
            native_codebooks=native_codebooks,
            refine_steps=refine_steps,
        )
        improve = error < best_error
        best_scale = torch.where(
            improve.unsqueeze(1), scale.reshape_as(best_scale), best_scale
        )
        best_delta = torch.where(
            improve.unsqueeze(1), delta.reshape_as(best_delta), best_delta
        )
        best_indices = torch.where(
            improve.reshape(out, 1, 1),
            indices.reshape_as(best_indices),
            best_indices,
        )
        best_anchor = torch.where(improve, anchor, best_anchor)
        best_error = torch.where(improve, error, best_error)
    nvec = neuron_len // spec.vector_size
    return Nvq1STensor(
        spec=spec,
        shape=(out, neuron_len),
        axis=0,
        neuron_len=neuron_len,
        neuron_scale=best_anchor.cpu().numpy().astype(np.float32, copy=False),
        sub_scale=best_scale.cpu().numpy().astype(np.uint8, copy=False),
        indices=(
            best_indices.reshape(out, groups_per_row * 3)[:, :nvec]
            .to(torch.int32)
            .cpu()
            .numpy()
            .astype(np.uint16, copy=False)
        ),
        delta_sign=best_delta.cpu().numpy().astype(np.uint8, copy=False),
        codebook=table,
    )


__all__ = ["quantize_axis0"]
