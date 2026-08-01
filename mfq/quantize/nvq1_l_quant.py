"""Offline weighted-SSE quantizer for the NVQ1-L packed format.

For a fixed neuron anchor, each group is solved exactly over every integer
sub-scale, both delta signs, and all 2048 codewords. The solver alternates this
discrete assignment with a closed-form FP16 neuron-anchor refit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from mfq.formats.nvq1_l import IQ1S_TERNARY_2048, Nvq1LSpec, Nvq1LTensor


def _validate_codebook(codebook: np.ndarray, expected_entries: int) -> np.ndarray:
    value = np.asarray(codebook)
    expected = (expected_entries, 8)
    if value.shape != expected:
        raise ValueError(f"ternary codebook has shape {value.shape}, expected {expected}")
    rounded = np.rint(value)
    if not np.array_equal(value, rounded) or not np.isin(rounded, (-1, 0, 1)).all():
        raise ValueError("ternary codebook entries must be in {-1, 0, 1}")
    if np.unique(rounded, axis=0).shape[0] != expected_entries:
        raise ValueError("ternary codebook entries must be unique")
    return np.ascontiguousarray(rounded, dtype=np.float32)


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
    spec: Nvq1LSpec,
    codebook: np.ndarray,
    *,
    group_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve all discrete variables exactly for fixed group anchors."""

    n_groups, groupsize = xgroup.shape
    vectors_per_group = groupsize // spec.vector_size
    if group_anchor.shape != (n_groups,):
        raise ValueError("group anchor count does not match NVQ1-L groups")

    best_scale = np.zeros(n_groups, dtype=np.uint8)
    best_delta = np.zeros(n_groups, dtype=np.uint8)
    best_indices = np.zeros((n_groups, vectors_per_group), dtype=np.uint16)
    qmax = (1 << spec.sub_bits) - 1

    for start in range(0, n_groups, group_chunk):
        stop = min(start + group_chunk, n_groups)
        xg = xgroup[start:stop]
        wg = wgroup[start:stop]
        anchors = group_anchor[start:stop]
        chunk_groups = stop - start
        xv = xg.reshape(chunk_groups * vectors_per_group, spec.vector_size)
        wv = wg.reshape(chunk_groups * vectors_per_group, spec.vector_size)
        weighted_x = wv * xv
        const = (weighted_x * xv).sum(axis=-1)

        chunk_error = np.full(chunk_groups, np.inf, dtype=np.float32)
        chunk_scale = np.zeros(chunk_groups, dtype=np.uint8)
        chunk_delta = np.zeros(chunk_groups, dtype=np.uint8)
        chunk_indices = np.zeros(
            (chunk_groups, vectors_per_group),
            dtype=np.uint16,
        )

        for delta_bit, delta in ((0, spec.delta), (1, -spec.delta)):
            bank = codebook if codebook.ndim == 2 else codebook[delta_bit]
            shifted = bank + np.float32(delta)
            cross = weighted_x @ shifted.T
            quad = wv @ (shifted * shifted).T

            for q in range(qmax + 1):
                vector_scale = np.repeat(anchors * np.float32(q), vectors_per_group)
                scale_column = vector_scale[:, None]
                variable = (
                    scale_column * scale_column * quad
                    - np.float32(2.0) * scale_column * cross
                )
                indices = np.argmin(variable, axis=-1).astype(np.uint16)
                rows = np.arange(indices.size)
                vector_error = const + variable[rows, indices]
                group_error = vector_error.reshape(
                    chunk_groups,
                    vectors_per_group,
                ).sum(axis=-1)
                improve = group_error < chunk_error
                if np.any(improve):
                    chunk_error[improve] = group_error[improve]
                    chunk_scale[improve] = q
                    chunk_delta[improve] = delta_bit
                    chunk_indices[improve] = indices.reshape(
                        chunk_groups,
                        vectors_per_group,
                    )[improve]

        best_scale[start:stop] = chunk_scale
        best_delta[start:stop] = chunk_delta
        best_indices[start:stop] = chunk_indices

    return best_scale, best_delta, best_indices


def _group_basis(
    sub_scale: np.ndarray,
    delta_sign: np.ndarray,
    indices: np.ndarray,
    spec: Nvq1LSpec,
    codebook: np.ndarray,
) -> np.ndarray:
    n_groups = sub_scale.size
    vectors_per_group = spec.groupsize // spec.vector_size
    if codebook.ndim == 2:
        code = codebook[indices]
    else:
        code = codebook[delta_sign[:, None], indices]
    code = code.reshape(n_groups, spec.groupsize)
    delta = np.where(delta_sign != 0, -spec.delta, spec.delta).astype(np.float32)
    return (
        sub_scale.astype(np.float32)[:, None]
        * (code.astype(np.float32) + delta[:, None])
    )


def _refit_anchor(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    sub_scale: np.ndarray,
    delta_sign: np.ndarray,
    indices: np.ndarray,
    *,
    out: int,
    ng: int,
    spec: Nvq1LSpec,
    codebook: np.ndarray,
) -> np.ndarray:
    basis = _group_basis(sub_scale, delta_sign, indices, spec, codebook)
    numerator = (wgroup * xgroup * basis).reshape(out, ng * spec.groupsize).sum(axis=-1)
    denominator = (wgroup * basis * basis).reshape(out, ng * spec.groupsize).sum(axis=-1)
    anchor = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0,
    )
    anchor = np.maximum(anchor, 0.0)
    return anchor.astype(np.float16).astype(np.float32)


def _row_error(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    sub_scale: np.ndarray,
    delta_sign: np.ndarray,
    indices: np.ndarray,
    anchor: np.ndarray,
    *,
    out: int,
    ng: int,
    spec: Nvq1LSpec,
    codebook: np.ndarray,
) -> np.ndarray:
    basis = _group_basis(sub_scale, delta_sign, indices, spec, codebook)
    reconstruction = basis * np.repeat(anchor, ng)[:, None]
    error = wgroup * (xgroup - reconstruction) ** 2
    return error.reshape(out, ng * spec.groupsize).sum(axis=-1)


def _solve_from_anchor(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    initial_anchor: np.ndarray,
    *,
    out: int,
    ng: int,
    spec: Nvq1LSpec,
    codebook: np.ndarray,
    group_chunk: int,
    refine_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor = initial_anchor.astype(np.float16).astype(np.float32)
    scale, delta, indices = _assign_groups(
        xgroup,
        wgroup,
        np.repeat(anchor, ng),
        spec,
        codebook,
        group_chunk=group_chunk,
    )
    error = _row_error(
        xgroup,
        wgroup,
        scale,
        delta,
        indices,
        anchor,
        out=out,
        ng=ng,
        spec=spec,
        codebook=codebook,
    )

    for _ in range(refine_steps):
        candidate_anchor = _refit_anchor(
            xgroup,
            wgroup,
            scale,
            delta,
            indices,
            out=out,
            ng=ng,
            spec=spec,
            codebook=codebook,
        )
        candidate_scale, candidate_delta, candidate_indices = _assign_groups(
            xgroup,
            wgroup,
            np.repeat(candidate_anchor, ng),
            spec,
            codebook,
            group_chunk=group_chunk,
        )
        candidate_error = _row_error(
            xgroup,
            wgroup,
            candidate_scale,
            candidate_delta,
            candidate_indices,
            candidate_anchor,
            out=out,
            ng=ng,
            spec=spec,
            codebook=codebook,
        )
        improve = candidate_error < error
        if not np.any(improve):
            break
        group_improve = np.repeat(improve, ng)
        scale[group_improve] = candidate_scale[group_improve]
        delta[group_improve] = candidate_delta[group_improve]
        indices[group_improve] = candidate_indices[group_improve]
        anchor[improve] = candidate_anchor[improve]
        error[improve] = candidate_error[improve]

    return scale, delta, indices, anchor, error


def quantize(
    weight: np.ndarray,
    spec: Nvq1LSpec,
    *,
    axis: int = 0,
    importance: np.ndarray | None = None,
    anchor_multipliers: Sequence[float] = (0.75, 1.0, 1.25),
    refine_steps: int = 2,
    group_chunk: int = 64,
    codebook: np.ndarray | None = None,
) -> Nvq1LTensor:
    """Quantize a tensor with multi-start alternating weighted-SSE search."""

    if refine_steps < 0:
        raise ValueError("refine_steps must be non-negative")
    if group_chunk <= 0:
        raise ValueError("group_chunk must be positive")
    multipliers = tuple(float(value) for value in anchor_multipliers)
    if not multipliers or any(not np.isfinite(value) or value <= 0 for value in multipliers):
        raise ValueError("anchor_multipliers must be finite and positive")
    table = _validate_codebook(
        IQ1S_TERNARY_2048 if codebook is None else codebook,
        1 << spec.index_bits,
    )

    W = np.asarray(weight, dtype=np.float32)
    if W.ndim < 2:
        raise ValueError(f"NVQ1-L requires ndim >= 2, got {W.shape}")
    if not np.isfinite(W).all():
        raise ValueError("NVQ1-L input must contain only finite values")
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
        np.float32(qmax * (1.0 + spec.delta)),
        out=np.zeros_like(row_peak),
        where=row_peak > 0,
    )

    vectors_per_group = spec.groupsize // spec.vector_size
    best_error = np.full(out, np.inf, dtype=np.float32)
    best_scale = np.zeros((out, ng), dtype=np.uint8)
    best_delta = np.zeros((out, ng), dtype=np.uint8)
    best_indices = np.zeros((out, ng, vectors_per_group), dtype=np.uint16)
    best_anchor = np.zeros(out, dtype=np.float32)

    for multiplier in multipliers:
        scale, delta, indices, anchor, error = _solve_from_anchor(
            xgroup,
            wgroup,
            base_anchor * np.float32(multiplier),
            out=out,
            ng=ng,
            spec=spec,
            codebook=table,
            group_chunk=group_chunk,
            refine_steps=refine_steps,
        )
        improve = error < best_error
        if np.any(improve):
            best_scale[improve] = scale.reshape(out, ng)[improve]
            best_delta[improve] = delta.reshape(out, ng)[improve]
            best_indices[improve] = indices.reshape(
                out,
                ng,
                vectors_per_group,
            )[improve]
            best_anchor[improve] = anchor[improve]
            best_error[improve] = error[improve]

    nvec = math.ceil(neuron_len / spec.vector_size)
    return Nvq1LTensor(
        spec=spec,
        shape=shape,
        axis=axis,
        neuron_len=neuron_len,
        neuron_scale=best_anchor,
        sub_scale=best_scale,
        indices=best_indices.reshape(out, ng * vectors_per_group)[:, :nvec],
        delta_sign=best_delta,
        codebook=(None if codebook is None else table.astype(np.int8)),
    )


def dequantize(
    tensor: Nvq1LTensor,
    *,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct a float32 tensor from NVQ1-L indices and scales."""

    spec = tensor.spec
    out = tensor.neuron_scale.size
    table = _validate_codebook(
        (
            tensor.codebook
            if codebook is None and tensor.codebook is not None
            else IQ1S_TERNARY_2048 if codebook is None else codebook
        ),
        1 << spec.index_bits,
    )
    code = table[np.asarray(tensor.indices, dtype=np.uint16)]
    code = code.reshape(out, -1)[:, : tensor.neuron_len].astype(np.float32)
    delta = np.where(tensor.delta_sign != 0, -spec.delta, spec.delta).astype(np.float32)
    delta = np.repeat(delta, spec.groupsize, axis=1)[:, : tensor.neuron_len]
    group_scale = tensor.neuron_scale[:, None] * tensor.sub_scale.astype(np.float32)
    scale = np.repeat(group_scale, spec.groupsize, axis=1)[:, : tensor.neuron_len]
    reconstruction = scale * (code + delta)

    shape, axis = tensor.shape, tensor.axis
    moved_shape = (shape[axis],) + shape[:axis] + shape[axis + 1 :]
    return np.moveaxis(reconstruction.reshape(moved_shape), 0, axis)
