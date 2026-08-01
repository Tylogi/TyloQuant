"""Offline weighted-SSE quantizer for neuron-anchored scalar ternary."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from mfq.formats.ternary import (
    NeuronTernarySpec,
    NeuronTernaryTensor,
)


def _importance_matrix(
    importance: np.ndarray | None,
    weight_shape: tuple[int, ...],
    axis: int,
    out: int,
    neuron_len: int,
) -> np.ndarray:
    if importance is None:
        return np.ones((out, neuron_len), dtype=np.float32)
    imp = np.asarray(importance, dtype=np.float32)
    if imp.ndim == 1:
        if imp.size != neuron_len:
            raise ValueError(f"importance has {imp.size} entries, expected {neuron_len}")
        result = np.broadcast_to(imp[None, :], (out, neuron_len)).copy()
    elif imp.shape == weight_shape:
        result = np.moveaxis(imp, axis, 0).reshape(out, neuron_len).copy()
    else:
        raise ValueError(
            f"importance must have shape ({neuron_len},) or {weight_shape}, got {imp.shape}"
        )
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("importance weights must be finite and non-negative")
    return result


def _assign_groups(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    group_anchor: np.ndarray,
    spec: NeuronTernarySpec,
) -> tuple[np.ndarray, np.ndarray]:
    n_groups = xgroup.shape[0]
    qmax = (1 << spec.sub_bits) - 1
    best_error = np.full(n_groups, np.inf, dtype=np.float32)
    best_scale = np.zeros(n_groups, dtype=np.uint8)
    best_code = np.zeros_like(xgroup, dtype=np.int8)

    for q in range(qmax + 1):
        scale = group_anchor * np.float32(q)
        threshold = np.float32(0.5) * scale[:, None]
        code = np.where(
            xgroup < -threshold,
            -1,
            np.where(xgroup > threshold, 1, 0),
        ).astype(np.int8)
        reconstruction = scale[:, None] * code.astype(np.float32)
        error = (wgroup * (xgroup - reconstruction) ** 2).sum(axis=-1)
        improve = error < best_error
        if np.any(improve):
            best_error[improve] = error[improve]
            best_scale[improve] = q
            best_code[improve] = code[improve]
    return best_scale, best_code


def _refit_anchor(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    sub_scale: np.ndarray,
    code: np.ndarray,
    *,
    out: int,
    ng: int,
    spec: NeuronTernarySpec,
) -> np.ndarray:
    basis = sub_scale.astype(np.float32)[:, None] * code.astype(np.float32)
    numerator = (wgroup * xgroup * basis).reshape(out, ng * spec.groupsize).sum(axis=-1)
    denominator = (wgroup * basis * basis).reshape(out, ng * spec.groupsize).sum(axis=-1)
    anchor = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0,
    )
    return np.maximum(anchor, 0.0).astype(np.float16).astype(np.float32)


def _row_error(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    sub_scale: np.ndarray,
    code: np.ndarray,
    anchor: np.ndarray,
    *,
    out: int,
    ng: int,
    spec: NeuronTernarySpec,
) -> np.ndarray:
    basis = sub_scale.astype(np.float32)[:, None] * code.astype(np.float32)
    reconstruction = np.repeat(anchor, ng)[:, None] * basis
    error = wgroup * (xgroup - reconstruction) ** 2
    return error.reshape(out, ng * spec.groupsize).sum(axis=-1)


def _solve_from_anchor(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    initial_anchor: np.ndarray,
    *,
    out: int,
    ng: int,
    spec: NeuronTernarySpec,
    refine_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor = initial_anchor.astype(np.float16).astype(np.float32)
    scale, code = _assign_groups(xgroup, wgroup, np.repeat(anchor, ng), spec)
    error = _row_error(
        xgroup,
        wgroup,
        scale,
        code,
        anchor,
        out=out,
        ng=ng,
        spec=spec,
    )

    for _ in range(refine_steps):
        candidate_anchor = _refit_anchor(
            xgroup,
            wgroup,
            scale,
            code,
            out=out,
            ng=ng,
            spec=spec,
        )
        candidate_scale, candidate_code = _assign_groups(
            xgroup,
            wgroup,
            np.repeat(candidate_anchor, ng),
            spec,
        )
        candidate_error = _row_error(
            xgroup,
            wgroup,
            candidate_scale,
            candidate_code,
            candidate_anchor,
            out=out,
            ng=ng,
            spec=spec,
        )
        improve = candidate_error < error
        if not np.any(improve):
            break
        group_improve = np.repeat(improve, ng)
        scale[group_improve] = candidate_scale[group_improve]
        code[group_improve] = candidate_code[group_improve]
        anchor[improve] = candidate_anchor[improve]
        error[improve] = candidate_error[improve]

    return scale, code, anchor, error


def quantize(
    weight: np.ndarray,
    spec: NeuronTernarySpec,
    *,
    axis: int = 0,
    importance: np.ndarray | None = None,
    anchor_multipliers: Sequence[float] = (0.75, 1.0, 1.25),
    refine_steps: int = 3,
) -> NeuronTernaryTensor:
    """Quantize with exact per-anchor ternary assignment and anchor refits."""

    if refine_steps < 0:
        raise ValueError("refine_steps must be non-negative")
    multipliers = tuple(float(value) for value in anchor_multipliers)
    if not multipliers or any(not np.isfinite(value) or value <= 0 for value in multipliers):
        raise ValueError("anchor_multipliers must be finite and positive")

    W = np.asarray(weight, dtype=np.float32)
    if W.ndim < 2:
        raise ValueError(f"ternary quantization requires ndim >= 2, got {W.shape}")
    if not np.isfinite(W).all():
        raise ValueError("ternary input must contain only finite values")
    axis = axis % W.ndim
    shape = W.shape
    moved = np.moveaxis(W, axis, 0)
    out = moved.shape[0]
    neuron_len = moved.size // out
    W2 = moved.reshape(out, neuron_len).copy()
    objective_weight = _importance_matrix(
        importance,
        shape,
        axis,
        out,
        neuron_len,
    )

    ng = math.ceil(neuron_len / spec.groupsize)
    padded_len = ng * spec.groupsize
    pad = padded_len - neuron_len
    if pad:
        W2 = np.pad(W2, ((0, 0), (0, pad)))
        objective_weight = np.pad(objective_weight, ((0, 0), (0, pad)))
    xgroup = W2.reshape(out * ng, spec.groupsize)
    wgroup = objective_weight.reshape(out * ng, spec.groupsize)

    qmax = (1 << spec.sub_bits) - 1
    row_peak = np.max(np.abs(W2), axis=-1)
    base_anchor = np.divide(
        row_peak,
        np.float32(qmax),
        out=np.zeros_like(row_peak),
        where=row_peak > 0,
    )

    best_error = np.full(out, np.inf, dtype=np.float32)
    best_scale = np.zeros((out, ng), dtype=np.uint8)
    best_code = np.zeros((out, ng, spec.groupsize), dtype=np.int8)
    best_anchor = np.zeros(out, dtype=np.float32)

    for multiplier in multipliers:
        scale, code, anchor, error = _solve_from_anchor(
            xgroup,
            wgroup,
            base_anchor * np.float32(multiplier),
            out=out,
            ng=ng,
            spec=spec,
            refine_steps=refine_steps,
        )
        improve = error < best_error
        if np.any(improve):
            best_scale[improve] = scale.reshape(out, ng)[improve]
            best_code[improve] = code.reshape(out, ng, spec.groupsize)[improve]
            best_anchor[improve] = anchor[improve]
            best_error[improve] = error[improve]

    code = best_code.reshape(out, padded_len)[:, :neuron_len]
    return NeuronTernaryTensor(
        spec=spec,
        shape=shape,
        axis=axis,
        neuron_len=neuron_len,
        neuron_scale=best_anchor,
        sub_scale=best_scale,
        trits=(code + 1).astype(np.uint8),
    )


def dequantize(tensor: NeuronTernaryTensor) -> np.ndarray:
    spec = tensor.spec
    out = tensor.neuron_scale.size
    code = np.asarray(tensor.trits, dtype=np.float32) - np.float32(1.0)
    group_scale = tensor.neuron_scale[:, None] * tensor.sub_scale.astype(np.float32)
    scale = np.repeat(group_scale, spec.groupsize, axis=1)[:, : tensor.neuron_len]
    reconstruction = code * scale

    shape, axis = tensor.shape, tensor.axis
    moved_shape = (shape[axis],) + shape[:axis] + shape[axis + 1 :]
    return np.moveaxis(reconstruction.reshape(moved_shape), 0, axis)

