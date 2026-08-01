"""Offline weighted-SSE quantizer for NVQ1-S.

The discrete solver is shared with NVQ1-L. For every fixed neuron anchor it
exhaustively searches 16 integer sub-scales, both delta signs, and all 512
ternary codewords for each of the three 8-weight vectors in a gs24 group.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from mfq.formats.nvq1_s import (
    NVQ1_S,
    NVQ1_S_SYNTHETIC_BANKS,
    Nvq1SSpec,
    Nvq1STensor,
    validate_nvq1_s_banked_codebook,
    validate_nvq1_s_codebook,
)
from mfq.quantize.nvq1_l_quant import (
    _importance_matrix,
    _solve_from_anchor,
)


def quantize(
    weight: np.ndarray,
    spec: Nvq1SSpec = NVQ1_S,
    *,
    axis: int = 0,
    importance: np.ndarray | None = None,
    codebook: np.ndarray = NVQ1_S_SYNTHETIC_BANKS,
    anchor_multipliers: Sequence[float] = (0.75, 1.0, 1.25),
    refine_steps: int = 2,
    group_chunk: int = 64,
) -> Nvq1STensor:
    if refine_steps < 0:
        raise ValueError("refine_steps must be non-negative")
    if group_chunk <= 0:
        raise ValueError("group_chunk must be positive")
    multipliers = tuple(float(value) for value in anchor_multipliers)
    if not multipliers or any(not np.isfinite(value) or value <= 0 for value in multipliers):
        raise ValueError("anchor_multipliers must be finite and positive")
    raw_table = np.asarray(codebook)
    table = (
        validate_nvq1_s_banked_codebook(raw_table)
        if raw_table.ndim == 3
        else validate_nvq1_s_codebook(raw_table)
    ).astype(np.float32)

    weight_array = np.asarray(weight, dtype=np.float32)
    if weight_array.ndim < 2:
        raise ValueError(f"NVQ1-S requires ndim >= 2, got {weight_array.shape}")
    if not np.isfinite(weight_array).all():
        raise ValueError("NVQ1-S input must contain only finite values")
    axis = axis % weight_array.ndim
    shape = weight_array.shape
    moved = np.moveaxis(weight_array, axis, 0)
    out = moved.shape[0]
    neuron_len = moved.size // out
    if neuron_len % 8:
        raise ValueError("NVQ1-S neuron length must be divisible by 8")
    weight_2d = moved.reshape(out, neuron_len).copy()
    objective_weight = _importance_matrix(
        importance,
        shape,
        axis,
        out,
        neuron_len,
    )

    ng = math.ceil(neuron_len / spec.groupsize)
    padded_len = ng * spec.groupsize
    internal_tail = padded_len - neuron_len
    if internal_tail:
        weight_2d = np.pad(weight_2d, ((0, 0), (0, internal_tail)))
        objective_weight = np.pad(objective_weight, ((0, 0), (0, internal_tail)))
    xgroup = weight_2d.reshape(out * ng, spec.groupsize)
    wgroup = objective_weight.reshape(out * ng, spec.groupsize)

    qmax = (1 << spec.sub_bits) - 1
    row_peak = np.max(np.abs(weight_2d), axis=-1)
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

    nvec = neuron_len // spec.vector_size
    return Nvq1STensor(
        spec=spec,
        shape=shape,
        axis=axis,
        neuron_len=neuron_len,
        neuron_scale=best_anchor,
        sub_scale=best_scale,
        indices=best_indices.reshape(out, ng * vectors_per_group)[:, :nvec],
        delta_sign=best_delta,
        codebook=(
            np.stack((table, table), axis=0).astype(np.int8)
            if table.ndim == 2
            else table.astype(np.int8)
        ),
    )


def dequantize(
    tensor: Nvq1STensor,
    *,
    codebook: np.ndarray | None = None,
) -> np.ndarray:
    spec = tensor.spec
    out = tensor.neuron_scale.size
    raw_table = np.asarray(
        tensor.codebook
        if codebook is None and tensor.codebook is not None
        else (NVQ1_S_SYNTHETIC_BANKS if codebook is None else codebook)
    )
    table = (
        validate_nvq1_s_banked_codebook(raw_table)
        if raw_table.ndim == 3
        else validate_nvq1_s_codebook(raw_table)
    )
    index = np.asarray(tensor.indices, dtype=np.uint16)
    if table.ndim == 2:
        code = table[index]
    else:
        vector_bank = np.repeat(tensor.delta_sign, 3, axis=1)[:, : index.shape[1]]
        code = table[vector_bank, index]
    code = code.reshape(out, -1)[:, : tensor.neuron_len].astype(np.float32)
    delta = np.where(tensor.delta_sign != 0, -spec.delta, spec.delta).astype(np.float32)
    delta = np.repeat(delta, spec.groupsize, axis=1)[:, : tensor.neuron_len]
    group_scale = tensor.neuron_scale[:, None] * tensor.sub_scale.astype(np.float32)
    scale = np.repeat(group_scale, spec.groupsize, axis=1)[:, : tensor.neuron_len]
    reconstruction = scale * (code + delta)

    shape, axis = tensor.shape, tensor.axis
    moved_shape = (shape[axis],) + shape[:axis] + shape[axis + 1 :]
    return np.moveaxis(reconstruction.reshape(moved_shape), 0, axis)
