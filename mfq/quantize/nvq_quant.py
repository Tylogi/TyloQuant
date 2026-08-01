"""Reference NVQ tensor quantizer.

This implementation optimizes the actual weighted squared reconstruction
error. It is intended for format search and numeric validation; a CUDA
quantizer/kernel can consume the same :class:`~mfq.formats.nvq.NvqTensor`
layout later.
"""

from __future__ import annotations

import math

import numpy as np

from mfq.formats.nvq import NvqSpec, NvqTensor, codebook_for, validate_codebook


def _resolve_codebook(spec: NvqSpec, codebook: np.ndarray | None) -> np.ndarray:
    table = codebook_for(spec) if codebook is None else validate_codebook(spec, codebook)
    return np.ascontiguousarray(table, dtype=np.float32)


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


def _encode_even_parity_signs(
    x: np.ndarray,
    objective_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sign-normalized targets and packed seven-bit sign masks."""

    out, padded_len = x.shape
    if padded_len % 8:
        raise ValueError("internal NVQ sign padding must be divisible by 8")
    x8 = x.reshape(out, padded_len // 8, 8)
    w8 = objective_weight.reshape(out, padded_len // 8, 8)
    encoded_negative = x8 < 0
    odd = (encoded_negative.sum(axis=-1) & 1).astype(bool)
    flip_cost = w8 * x8 * x8
    flip_index = np.argmin(flip_cost, axis=-1)
    rr, gg = np.nonzero(odd)
    if rr.size:
        encoded_negative[rr, gg, flip_index[rr, gg]] ^= True

    bit_weights = (1 << np.arange(7, dtype=np.uint8))[None, None, :]
    masks = (encoded_negative[..., :7].astype(np.uint8) * bit_weights).sum(axis=-1)
    sign = np.where(encoded_negative, -1.0, 1.0).astype(np.float32)
    target = (x8 * sign).reshape(out, padded_len)
    return target, masks.astype(np.uint8)


def _encode_index_parity_signs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode seven literal signs and carry total parity in the E8 index MSB."""

    out, padded_len = x.shape
    if padded_len % 8:
        raise ValueError("internal NVQ sign padding must be divisible by 8")
    x8 = x.reshape(out, padded_len // 8, 8)
    negative = x8 < 0
    bit_weights = (1 << np.arange(7, dtype=np.uint8))[None, None, :]
    masks = (negative[..., :7].astype(np.uint8) * bit_weights).sum(axis=-1)
    banks = (negative.sum(axis=-1) & 1).astype(np.uint8)
    return np.abs(x).astype(np.float32), masks.astype(np.uint8), banks


def _cached_codebook_products(
    xvec: np.ndarray,
    wvec: np.ndarray,
    codebook: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cb = codebook.astype(np.float32, copy=False)
    cross = (wvec * xvec) @ cb.T
    quad = wvec @ (cb * cb).T
    const = (wvec * xvec * xvec).sum(axis=-1)
    return cross, quad, const


def _indices_at_scale(
    cross: np.ndarray,
    quad: np.ndarray,
    scale_per_vector: np.ndarray,
    bank: np.ndarray | None = None,
) -> np.ndarray:
    scale = scale_per_vector[:, None]
    distance = scale * scale * quad - 2.0 * scale * cross
    if bank is None:
        return np.argmin(distance, axis=-1).astype(np.uint8)
    bank = np.asarray(bank, dtype=np.uint8).reshape(-1)
    if bank.size != distance.shape[0]:
        raise ValueError("index-parity bank count does not match vector count")
    result = np.empty(bank.size, dtype=np.uint8)
    for value in (0, 1):
        selected = bank == value
        if np.any(selected):
            start = 128 * value
            result[selected] = np.argmin(distance[selected, start : start + 128], axis=-1) + start
    return result


def _indices_and_variable_error_at_scale(
    cross: np.ndarray,
    quad: np.ndarray,
    scale_per_vector: np.ndarray,
    bank: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    indices = _indices_at_scale(cross, quad, scale_per_vector, bank)
    rows = np.arange(indices.size)
    scale = scale_per_vector
    error = scale * scale * quad[rows, indices] - 2.0 * scale * cross[rows, indices]
    return indices, error


def _refit_group_scale(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    indices: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    code = codebook[indices].reshape(xgroup.shape).astype(np.float32, copy=False)
    numerator = (wgroup * xgroup * code).sum(axis=-1)
    denominator = (wgroup * code * code).sum(axis=-1)
    scale = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0,
    )
    return np.maximum(scale, 0.0).astype(np.float32)


def _group_error(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    indices: np.ndarray,
    scale: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    code = codebook[indices].reshape(xgroup.shape).astype(np.float32, copy=False)
    diff = scale[:, None] * code - xgroup
    return (wgroup * diff * diff).sum(axis=-1)


def _search_groups(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    spec: NvqSpec,
    codebook: np.ndarray,
    vector_bank: np.ndarray | None,
    *,
    search_steps: int,
    group_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Jointly search lattice indices and one floating scale per group."""

    if search_steps < 1:
        raise ValueError("search_steps must be positive")
    n_groups, gs = xgroup.shape
    vectors_per_group = gs // spec.vector_size
    all_scales = np.empty(n_groups, dtype=np.float32)
    all_indices = np.empty((n_groups, vectors_per_group), dtype=np.uint8)
    qmax = float(codebook.max())
    offsets = np.linspace(-0.12 * qmax, 0.12 * qmax, search_steps, dtype=np.float32)

    for start in range(0, n_groups, group_chunk):
        stop = min(start + group_chunk, n_groups)
        xg = xgroup[start:stop]
        wg = wgroup[start:stop]
        g = stop - start
        bank = (
            vector_bank[start:stop].reshape(-1)
            if vector_bank is not None
            else None
        )
        xv = xg.reshape(g * vectors_per_group, spec.vector_size)
        wv = wg.reshape(g * vectors_per_group, spec.vector_size)
        cross, quad, _ = _cached_codebook_products(xv, wv, codebook)
        max_abs = np.max(np.abs(xg), axis=-1)
        best_error = np.full(g, np.inf, dtype=np.float32)
        best_scale = np.zeros(g, dtype=np.float32)
        best_indices = np.zeros((g, vectors_per_group), dtype=np.uint8)

        for offset in offsets:
            initial = np.divide(
                max_abs,
                qmax + float(offset),
                out=np.zeros_like(max_abs),
                where=max_abs > 0,
            )
            repeated = np.repeat(initial, vectors_per_group)
            idx = _indices_at_scale(cross, quad, repeated, bank).reshape(g, vectors_per_group)
            scale = _refit_group_scale(xg, wg, idx, codebook)
            idx = _indices_at_scale(
                cross,
                quad,
                np.repeat(scale, vectors_per_group),
                bank,
            ).reshape(g, vectors_per_group)
            scale = _refit_group_scale(xg, wg, idx, codebook)
            error = _group_error(xg, wg, idx, scale, codebook)
            better = error < best_error
            best_error = np.where(better, error, best_error)
            best_scale = np.where(better, scale, best_scale)
            best_indices[better] = idx[better]

        all_scales[start:stop] = best_scale
        all_indices[start:stop] = best_indices

    return all_scales, all_indices


def _reassign_at_effective_scales(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    scale: np.ndarray,
    spec: NvqSpec,
    codebook: np.ndarray,
    vector_bank: np.ndarray | None,
    *,
    group_chunk: int,
) -> np.ndarray:
    n_groups, gs = xgroup.shape
    vectors_per_group = gs // spec.vector_size
    result = np.empty((n_groups, vectors_per_group), dtype=np.uint8)
    for start in range(0, n_groups, group_chunk):
        stop = min(start + group_chunk, n_groups)
        g = stop - start
        xv = xgroup[start:stop].reshape(g * vectors_per_group, spec.vector_size)
        wv = wgroup[start:stop].reshape(g * vectors_per_group, spec.vector_size)
        cross, quad, _ = _cached_codebook_products(xv, wv, codebook)
        bank = (
            vector_bank[start:stop].reshape(-1)
            if vector_bank is not None
            else None
        )
        result[start:stop] = _indices_at_scale(
            cross,
            quad,
            np.repeat(scale[start:stop], vectors_per_group),
            bank,
        ).reshape(g, vectors_per_group)
    return result


def _refit_neuron_anchor(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    sub_scale: np.ndarray,
    indices: np.ndarray,
    spec: NvqSpec,
    codebook: np.ndarray,
    *,
    out: int,
    ng: int,
) -> np.ndarray:
    code = codebook[indices].reshape(xgroup.shape).astype(np.float32, copy=False)
    basis = sub_scale.reshape(-1, 1).astype(np.float32) * code
    numerator = (wgroup * xgroup * basis).reshape(out, ng, spec.groupsize).sum(axis=(1, 2))
    denominator = (wgroup * basis * basis).reshape(out, ng, spec.groupsize).sum(axis=(1, 2))
    anchor = np.divide(
        numerator,
        denominator,
        out=np.zeros(out, dtype=np.float32),
        where=denominator > 0,
    )
    return np.maximum(anchor, 0.0).astype(np.float16).astype(np.float32)


def _row_error(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    sub_scale: np.ndarray,
    indices: np.ndarray,
    neuron_scale: np.ndarray,
    spec: NvqSpec,
    codebook: np.ndarray,
    *,
    out: int,
    ng: int,
) -> np.ndarray:
    code = codebook[indices].reshape(xgroup.shape).astype(np.float32, copy=False)
    scale = (neuron_scale[:, None] * sub_scale.reshape(out, ng)).reshape(-1, 1)
    diff = scale * code - xgroup
    return (wgroup * diff * diff).reshape(out, ng, spec.groupsize).sum(axis=(1, 2))


def _assign_quantized_group_scales(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    neuron_scale: np.ndarray,
    spec: NvqSpec,
    codebook: np.ndarray,
    vector_bank: np.ndarray | None,
    *,
    ng: int,
    group_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exactly choose each integer sub-scale and its codewords for fixed anchors."""

    n_groups, gs = xgroup.shape
    vectors_per_group = gs // spec.vector_size
    qmax = (1 << spec.sub_bits) - 1
    anchors = np.repeat(neuron_scale, ng)
    result_scale = np.zeros(n_groups, dtype=np.uint8)
    result_indices = np.zeros((n_groups, vectors_per_group), dtype=np.uint8)

    for start in range(0, n_groups, group_chunk):
        stop = min(start + group_chunk, n_groups)
        g = stop - start
        xv = xgroup[start:stop].reshape(g * vectors_per_group, spec.vector_size)
        wv = wgroup[start:stop].reshape(g * vectors_per_group, spec.vector_size)
        cross, quad, const = _cached_codebook_products(xv, wv, codebook)
        bank = (
            vector_bank[start:stop].reshape(-1)
            if vector_bank is not None
            else None
        )
        best_error = np.full(g, np.inf, dtype=np.float32)
        best_q = np.zeros(g, dtype=np.uint8)
        best_indices = np.zeros((g, vectors_per_group), dtype=np.uint8)
        chunk_anchor = anchors[start:stop]

        for q in range(qmax + 1):
            group_scale = chunk_anchor * np.float32(q)
            vector_scale = np.repeat(group_scale, vectors_per_group)
            indices, variable = _indices_and_variable_error_at_scale(
                cross,
                quad,
                vector_scale,
                bank,
            )
            error = (variable + const).reshape(g, vectors_per_group).sum(axis=-1)
            better = error < best_error
            best_error = np.where(better, error, best_error)
            best_q = np.where(better, q, best_q).astype(np.uint8)
            best_indices[better] = indices.reshape(g, vectors_per_group)[better]

        result_scale[start:stop] = best_q
        result_indices[start:stop] = best_indices
    return result_scale, result_indices


def _refine_integer_scales(
    xgroup: np.ndarray,
    wgroup: np.ndarray,
    sub_scale: np.ndarray,
    indices: np.ndarray,
    neuron_scale: np.ndarray,
    spec: NvqSpec,
    codebook: np.ndarray,
    vector_bank: np.ndarray | None,
    *,
    out: int,
    ng: int,
    group_chunk: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternate exact integer-scale assignment with FP16 row-anchor refits."""

    best_scale = sub_scale.reshape(-1).astype(np.uint8, copy=True)
    best_indices = indices.astype(np.uint8, copy=True)
    best_anchor = neuron_scale.astype(np.float32, copy=True)
    best_error = _row_error(
        xgroup,
        wgroup,
        best_scale,
        best_indices,
        best_anchor,
        spec,
        codebook,
        out=out,
        ng=ng,
    )

    for _ in range(steps):
        candidate_scale, candidate_indices = _assign_quantized_group_scales(
            xgroup,
            wgroup,
            best_anchor,
            spec,
            codebook,
            vector_bank,
            ng=ng,
            group_chunk=group_chunk,
        )
        candidate_anchor = _refit_neuron_anchor(
            xgroup,
            wgroup,
            candidate_scale,
            candidate_indices,
            spec,
            codebook,
            out=out,
            ng=ng,
        )
        candidate_error = _row_error(
            xgroup,
            wgroup,
            candidate_scale,
            candidate_indices,
            candidate_anchor,
            spec,
            codebook,
            out=out,
            ng=ng,
        )
        improve = candidate_error < best_error
        if not np.any(improve):
            break
        group_improve = np.repeat(improve, ng)
        best_scale[group_improve] = candidate_scale[group_improve]
        best_indices[group_improve] = candidate_indices[group_improve]
        best_anchor[improve] = candidate_anchor[improve]
        best_error[improve] = candidate_error[improve]

    return best_scale.reshape(out, ng), best_indices, best_anchor


def quantize(
    weight: np.ndarray,
    spec: NvqSpec,
    *,
    axis: int = 0,
    importance: np.ndarray | None = None,
    search_steps: int = 19,
    group_chunk: int = 1024,
    refit_anchor: bool = True,
    codebook: np.ndarray | None = None,
    scale_refine_steps: int = 0,
) -> NvqTensor:
    """Quantize a tensor with weighted lattice search.

    ``importance`` is either one non-negative weight per input coordinate or a
    tensor matching ``weight``. Passing llama.cpp imatrix values therefore
    optimizes their diagonal weighted-MSE objective directly.
    """

    if scale_refine_steps < 0:
        raise ValueError("scale_refine_steps must be non-negative")
    table = _resolve_codebook(spec, codebook)
    W = np.asarray(weight, dtype=np.float32)
    if W.ndim < 2:
        raise ValueError(f"NVQ requires ndim >= 2, got {W.shape}")
    axis = axis % W.ndim
    shape = W.shape
    Wt = np.moveaxis(W, axis, 0)
    out = Wt.shape[0]
    neuron_len = Wt.size // out
    W2 = Wt.reshape(out, neuron_len).copy()
    objective_weight = _importance_matrix(importance, shape, axis, out, neuron_len)

    ng = math.ceil(neuron_len / spec.groupsize)
    padded_len = ng * spec.groupsize
    pad = padded_len - neuron_len
    if pad:
        W2 = np.pad(W2, ((0, 0), (0, pad)))
        objective_weight = np.pad(objective_weight, ((0, 0), (0, pad)))

    if spec.sign_mode == "index_parity":
        target, sign_masks, sign_banks = _encode_index_parity_signs(W2)
        vector_bank = sign_banks.reshape(out * ng, spec.groupsize // spec.vector_size)
    else:
        target, sign_masks = _encode_even_parity_signs(W2, objective_weight)
        vector_bank = None
    xgroup = target.reshape(out * ng, spec.groupsize)
    wgroup = objective_weight.reshape(out * ng, spec.groupsize)
    raw_scale, indices = _search_groups(
        xgroup,
        wgroup,
        spec,
        table,
        vector_bank,
        search_steps=search_steps,
        group_chunk=group_chunk,
    )

    scale_levels = (1 << spec.sub_bits) - 1
    raw_scale_2d = raw_scale.reshape(out, ng)
    row_max = raw_scale_2d.max(axis=-1)
    neuron_scale = np.where(
        row_max > 0,
        (row_max / scale_levels).astype(np.float16).astype(np.float32),
        np.float32(0.0),
    )
    safe_anchor = np.where(neuron_scale > 0, neuron_scale, 1.0)
    sub_scale = np.clip(
        np.rint(raw_scale_2d / safe_anchor[:, None]),
        0,
        scale_levels,
    ).astype(np.uint8)

    effective_scale = (neuron_scale[:, None] * sub_scale).reshape(-1)
    indices = _reassign_at_effective_scales(
        xgroup,
        wgroup,
        effective_scale,
        spec,
        table,
        vector_bank,
        group_chunk=group_chunk,
    )
    if refit_anchor:
        neuron_scale = _refit_neuron_anchor(
            xgroup,
            wgroup,
            sub_scale,
            indices,
            spec,
            table,
            out=out,
            ng=ng,
        )
        effective_scale = (neuron_scale[:, None] * sub_scale).reshape(-1)
        indices = _reassign_at_effective_scales(
            xgroup,
            wgroup,
            effective_scale,
            spec,
            table,
            vector_bank,
            group_chunk=group_chunk,
        )
        neuron_scale = _refit_neuron_anchor(
            xgroup,
            wgroup,
            sub_scale,
            indices,
            spec,
            table,
            out=out,
            ng=ng,
        )

    if scale_refine_steps:
        sub_scale, indices, neuron_scale = _refine_integer_scales(
            xgroup,
            wgroup,
            sub_scale,
            indices,
            neuron_scale,
            spec,
            table,
            vector_bank,
            out=out,
            ng=ng,
            group_chunk=group_chunk,
            steps=scale_refine_steps,
        )

    nvec = math.ceil(neuron_len / spec.vector_size)
    nsign = math.ceil(neuron_len / 8)
    flat_indices = indices.reshape(out, ng * (spec.groupsize // spec.vector_size))[:, :nvec]
    return NvqTensor(
        spec=spec,
        shape=shape,
        axis=axis,
        neuron_len=neuron_len,
        neuron_scale=neuron_scale.astype(np.float32),
        sub_scale=sub_scale,
        indices=flat_indices.astype(np.uint8),
        signs=sign_masks[:, :nsign].astype(np.uint8),
        codebook=(
            None
            if codebook is None
            else validate_codebook(spec, codebook)
        ),
    )


def _decode_signs(
    sign_masks: np.ndarray,
    neuron_len: int,
    spec: NvqSpec,
    indices: np.ndarray,
) -> np.ndarray:
    masks = np.asarray(sign_masks, dtype=np.uint8)
    lower = ((masks[..., None] >> np.arange(7, dtype=np.uint8)) & 1).astype(np.uint8)
    lower_parity = (lower.sum(axis=-1, keepdims=True) & 1).astype(np.uint8)
    if spec.sign_mode == "index_parity":
        bank = ((np.asarray(indices, dtype=np.uint8) >> 7) & 1)[..., None]
        last = lower_parity ^ bank
    else:
        last = lower_parity
    negative = np.concatenate([lower, last], axis=-1)
    sign = np.where(negative != 0, -1.0, 1.0).astype(np.float32)
    return sign.reshape(masks.shape[0], -1)[:, :neuron_len]


def dequantize(tensor: NvqTensor, *, codebook: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct a float32 tensor from NVQ indices and scales."""

    spec = tensor.spec
    out = tensor.neuron_scale.size
    table = _resolve_codebook(
        spec,
        tensor.codebook if codebook is None else codebook,
    )
    magnitude = table[tensor.indices].reshape(out, -1)[:, : tensor.neuron_len]
    sign = _decode_signs(tensor.signs, tensor.neuron_len, spec, tensor.indices)
    group_scale = tensor.neuron_scale[:, None] * tensor.sub_scale.astype(np.float32)
    scale = np.repeat(group_scale, spec.groupsize, axis=1)[:, : tensor.neuron_len]
    recon = magnitude * sign * scale
    shape, axis = tensor.shape, tensor.axis
    moved_shape = (shape[axis],) + shape[:axis] + shape[axis + 1 :]
    return np.moveaxis(recon.reshape(moved_shape), 0, axis)
