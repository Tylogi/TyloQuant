"""NINT 张量级量化与反量化。

把 :mod:`mfq.formats.nint` 的 1D neuron codec 提升到 nD 张量：沿指定 ``axis``
切出每一条 neuron 行（该神经元的全部权重），批量做 neuron-anchored 两级量化。

权重张量 ``W[out, in]`` 取 ``axis=0``：每行 = 一个输出神经元，整行共享一个
f16 neuron scale/min，行内按 ``groupsize`` 切 sub-group。要求 ``weight.ndim >= 2``；
1D 权重请直接用 :mod:`mfq.formats.nint` 的 1D codec。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mfq.formats.nint import NintSpec, _uint_dtype, make_qkx2


_IMATRIX_SUPERBLOCK = 256


@dataclass
class NintTensor:
    """一个 nD 权重张量的 NINT 表示（所有 neuron 行打包到一起）。"""

    spec: NintSpec
    shape: tuple[int, ...]
    axis: int
    q: np.ndarray               # (out, ng, gs) uint，含尾组补零
    neuron_scale: np.ndarray    # (out,) float32（f16-round-tripped neu_s/K）
    neuron_min: np.ndarray      # (out,) float32（f16-round-tripped neu_m/K）
    sub_scale: np.ndarray       # (out, ng) uint
    sub_min: np.ndarray         # (out, ng) uint
    neuron_len: int             # 每 neuron 实际有效长度（未含补零）


def _importance_as_rows(
    importance: np.ndarray,
    shape: tuple[int, ...],
    axis: int,
    out: int,
    neuron_len: int,
) -> np.ndarray:
    values = np.asarray(importance, dtype=np.float32)
    if values.ndim == 1:
        if values.shape[0] != neuron_len:
            raise ValueError(
                f"NINT importance width {values.shape[0]} != neuron length {neuron_len}"
            )
        rows = np.broadcast_to(values, (out, neuron_len))
    elif values.shape == shape:
        rows = np.moveaxis(values, axis, 0).reshape(out, neuron_len)
    elif values.shape == (out, neuron_len):
        rows = values
    else:
        raise ValueError(
            "NINT importance must be one input-channel vector, the weight shape, "
            f"or {(out, neuron_len)} row weights; got {values.shape}"
        )
    if not np.isfinite(rows).all() or np.any(rows < 0):
        raise ValueError("NINT importance must contain finite non-negative values")
    return np.ascontiguousarray(rows, dtype=np.float32)


def _imatrix_element_weights(
    rows: np.ndarray,
    importance_rows: np.ndarray,
    neuron_len: int,
) -> np.ndarray:
    """Build llama.cpp-style element weights using 256-value sigma² blocks."""

    out = rows.shape[0]
    block = _IMATRIX_SUPERBLOCK
    block_pad = (-neuron_len) % block
    real = rows[:, :neuron_len]
    if block_pad:
        real = np.concatenate(
            [real, np.zeros((out, block_pad), dtype=np.float32)], axis=1
        )
    blocks = real.reshape(out, -1, block)
    counts = np.full((blocks.shape[1],), block, dtype=np.float32)
    if block_pad:
        counts[-1] -= block_pad
    sigma2 = 2.0 * (blocks * blocks).sum(axis=-1) / counts[None, :]
    sigma2_elements = np.repeat(sigma2, block, axis=1)[:, :neuron_len]
    weights = importance_rows * np.sqrt(
        sigma2_elements + rows[:, :neuron_len] * rows[:, :neuron_len]
    )
    if rows.shape[1] != neuron_len:
        weights = np.concatenate(
            [
                weights,
                np.zeros(
                    (out, rows.shape[1] - neuron_len), dtype=np.float32
                ),
            ],
            axis=1,
        )
    return np.ascontiguousarray(weights, dtype=np.float32)


def _make_qkx3(
    x: np.ndarray,
    weights: np.ndarray,
    nmax: int,
    rmin: float = -0.9,
    rdelta: float = 0.05,
    nstep: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched weighted affine search matching llama.cpp make_qkx3_quants."""

    zero = np.float32(0.0)
    mn = np.minimum(x.min(axis=-1), zero)
    mx = x.max(axis=-1)
    sum_w = weights.sum(axis=-1)
    sum_x = (weights * x).sum(axis=-1)
    degen = (mx <= mn) | (sum_w <= 0)
    rng = np.where(degen, np.float32(1.0), mx - mn)

    iscale0 = np.float32(nmax) / rng
    scale0 = np.float32(1.0) / iscale0
    levels0 = np.clip(
        np.rint(iscale0[..., None] * (x - mn[..., None])), 0, nmax
    )
    diff0 = scale0[..., None] * levels0 + mn[..., None] - x
    best_error = (weights * diff0 * diff0).sum(axis=-1)
    best_scale = scale0.copy()
    best_min = mn.copy()

    for step in range(nstep + 1):
        iscale = (
            np.float32(rmin + rdelta * step + nmax) / rng
        )
        levels = np.clip(
            np.rint(iscale[..., None] * (x - mn[..., None])), 0, nmax
        )
        sum_l = (weights * levels).sum(axis=-1)
        sum_l2 = (weights * levels * levels).sum(axis=-1)
        sum_xl = (weights * levels * x).sum(axis=-1)
        determinant = sum_w * sum_l2 - sum_l * sum_l
        valid = determinant > 0
        divisor = np.where(valid, determinant, np.float32(1.0))
        candidate_scale = (
            sum_w * sum_xl - sum_x * sum_l
        ) / divisor
        candidate_min = (
            sum_l2 * sum_x - sum_l * sum_xl
        ) / divisor
        positive_min = candidate_min > 0
        safe_sum_l2 = np.where(
            sum_l2 > 0, sum_l2, np.float32(1.0)
        )
        candidate_scale = np.where(
            positive_min, sum_xl / safe_sum_l2, candidate_scale
        )
        candidate_min = np.where(
            positive_min, zero, candidate_min
        )
        candidate_diff = (
            candidate_scale[..., None] * levels
            + candidate_min[..., None]
            - x
        )
        candidate_error = (
            weights * candidate_diff * candidate_diff
        ).sum(axis=-1)
        better = valid & (candidate_error < best_error)
        best_error = np.where(better, candidate_error, best_error)
        best_scale = np.where(better, candidate_scale, best_scale)
        best_min = np.where(better, candidate_min, best_min)

    best_scale = np.where(degen, zero, best_scale)
    best_min = np.where(degen, np.minimum(mn, zero), best_min)
    return (
        best_scale.astype(np.float32),
        best_min.astype(np.float32),
    )


def _make_qp(
    x: np.ndarray,
    weights: np.ndarray,
    nmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted neuron-level scale quantization from llama.cpp make_qp_quants."""

    maximum = x.max(axis=-1)
    active = maximum >= np.float32(1e-15)
    safe_maximum = np.where(active, maximum, np.float32(1.0))
    iscale = np.float32(nmax) / safe_maximum
    levels = np.clip(
        np.rint(iscale[..., None] * x), 0, nmax
    ).astype(np.int32)
    scale = np.float32(1.0) / iscale
    difference = x - scale[..., None] * levels
    best_error = (weights * difference * difference).sum(axis=-1)

    for offset in range(-4, 5):
        if offset == 0:
            continue
        candidate_iscale = (
            np.float32(nmax + 0.1 * offset) / safe_maximum
        )
        candidate_scale = np.float32(1.0) / candidate_iscale
        candidate_levels = np.clip(
            np.rint(candidate_iscale[..., None] * x), 0, nmax
        )
        candidate_difference = (
            x - candidate_scale[..., None] * candidate_levels
        )
        candidate_error = (
            weights * candidate_difference * candidate_difference
        ).sum(axis=-1)
        better = active & (candidate_error < best_error)
        best_error = np.where(better, candidate_error, best_error)
        iscale = np.where(better, candidate_iscale, iscale)

    levels = np.clip(
        np.rint(iscale[..., None] * x), 0, nmax
    ).astype(np.int32)
    levels_f = levels.astype(np.float32)
    sum_lx = (weights * x * levels_f).sum(axis=-1)
    sum_l2 = (weights * levels_f * levels_f).sum(axis=-1)

    for _ in range(5):
        changed = False
        for index in range(x.shape[-1]):
            old_level = levels[:, index].astype(np.float32)
            weight = weights[:, index]
            value = x[:, index]
            candidate_lx = sum_lx - weight * value * old_level
            candidate_l2 = sum_l2 - weight * old_level * old_level
            valid = (candidate_lx > 0) & (candidate_l2 > 0)
            safe_lx = np.where(
                valid, candidate_lx, np.float32(1.0)
            )
            new_level = np.clip(
                np.rint(value * candidate_l2 / safe_lx), 0, nmax
            ).astype(np.int32)
            new_level_f = new_level.astype(np.float32)
            updated_lx = candidate_lx + weight * value * new_level_f
            updated_l2 = (
                candidate_l2 + weight * new_level_f * new_level_f
            )
            accept = (
                valid
                & (new_level != levels[:, index])
                & (
                    updated_lx * updated_lx * sum_l2
                    > sum_lx * sum_lx * updated_l2
                )
            )
            if np.any(accept):
                changed = True
                levels[:, index] = np.where(
                    accept, new_level, levels[:, index]
                )
                sum_lx = np.where(accept, updated_lx, sum_lx)
                sum_l2 = np.where(accept, updated_l2, sum_l2)
        if not changed:
            break

    scale = np.where(
        active & (sum_l2 > 0),
        sum_lx / np.where(sum_l2 > 0, sum_l2, np.float32(1.0)),
        np.float32(0.0),
    )
    levels = np.where(active[:, None], levels, 0)
    return scale.astype(np.float32), levels


def _refine_imatrix_final_encoding(
    groups: np.ndarray,
    weights: np.ndarray,
    q: np.ndarray,
    neuron_scale: np.ndarray,
    neuron_min: np.ndarray,
    sub_scale: np.ndarray,
    sub_min: np.ndarray,
    nmax: int,
    sub_nmax: int,
    passes: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Minimize weighted SSE for the values that are actually stored.

    The first-stage affine fit produces one floating-point scale/minimum per
    group.  NINT then stores those values through a second integer level plus
    one fp16 row scale/minimum.  Optimizing the two stages independently does
    not minimize the reconstruction emitted by the final format.  This
    routine performs monotonic coordinate descent directly on

    ``neuron_scale * sub_scale * q - neuron_min * sub_min``.

    It is intentionally used only for the imatrix path so the historical
    weight-only encoding remains byte-for-byte unchanged.
    """

    values = np.asarray(groups, dtype=np.float32)
    objective_weight = np.asarray(weights, dtype=np.float32)
    levels = np.asarray(q, dtype=np.int32).copy()
    row_scale = np.asarray(neuron_scale, dtype=np.float32).copy()
    row_min = np.asarray(neuron_min, dtype=np.float32).copy()
    scale_levels = np.asarray(sub_scale, dtype=np.int32).copy()
    min_levels = np.asarray(sub_min, dtype=np.int32).copy()

    def statistics(current_levels: np.ndarray):
        levels_f = current_levels.astype(np.float32)
        sum_w = objective_weight.sum(axis=-1, dtype=np.float64)
        sum_x = (objective_weight * values).sum(axis=-1, dtype=np.float64)
        sum_x2 = (
            objective_weight * values * values
        ).sum(axis=-1, dtype=np.float64)
        sum_q = (
            objective_weight * levels_f
        ).sum(axis=-1, dtype=np.float64)
        sum_q2 = (
            objective_weight * levels_f * levels_f
        ).sum(axis=-1, dtype=np.float64)
        sum_qx = (
            objective_weight * levels_f * values
        ).sum(axis=-1, dtype=np.float64)
        return sum_w, sum_x, sum_x2, sum_q, sum_q2, sum_qx

    def group_error(
        stats,
        candidate_scale: np.ndarray,
        candidate_min: np.ndarray,
        candidate_scale_levels: np.ndarray,
        candidate_min_levels: np.ndarray,
    ) -> np.ndarray:
        sum_w, sum_x, sum_x2, sum_q, sum_q2, sum_qx = stats
        effective_scale = (
            candidate_scale[:, None].astype(np.float64)
            * candidate_scale_levels.astype(np.float64)
        )
        effective_min = (
            candidate_min[:, None].astype(np.float64)
            * candidate_min_levels.astype(np.float64)
        )
        return (
            effective_scale * effective_scale * sum_q2
            + effective_min * effective_min * sum_w
            + sum_x2
            - 2.0 * effective_scale * effective_min * sum_q
            - 2.0 * effective_scale * sum_qx
            + 2.0 * effective_min * sum_x
        )

    def fp16_round(values_: np.ndarray) -> np.ndarray:
        finite = np.where(np.isfinite(values_), values_, 0.0)
        return np.maximum(
            finite, 0.0
        ).astype(np.float16).astype(np.float32)

    stats = statistics(levels)
    best_error = group_error(
        stats, row_scale, row_min, scale_levels, min_levels
    ).sum(axis=-1)

    for _ in range(max(0, int(passes))):
        candidate_levels = levels.copy()
        candidate_scale = row_scale.copy()
        candidate_min = row_min.copy()
        candidate_scale_levels = scale_levels.copy()
        candidate_min_levels = min_levels.copy()
        candidate_stats = stats

        # With q fixed, each group's integer scale/minimum pair is a convex
        # two-variable problem.  Alternating its exact one-dimensional
        # minimizers is inexpensive and directly targets the final format.
        for _coordinate_pass in range(2):
            sum_w, sum_x, _, sum_q, sum_q2, sum_qx = candidate_stats
            effective_min = (
                candidate_min[:, None].astype(np.float64)
                * candidate_min_levels.astype(np.float64)
            )
            scale_denominator = (
                candidate_scale[:, None].astype(np.float64) * sum_q2
            )
            valid_scale = scale_denominator > 0.0
            proposed_scale_levels = np.clip(
                np.rint(
                    np.divide(
                        sum_qx + effective_min * sum_q,
                        np.where(valid_scale, scale_denominator, 1.0),
                    )
                ),
                0,
                sub_nmax,
            ).astype(np.int32)
            proposed_scale_levels = np.where(
                valid_scale, proposed_scale_levels, candidate_scale_levels
            )
            old_group_error = group_error(
                candidate_stats,
                candidate_scale,
                candidate_min,
                candidate_scale_levels,
                candidate_min_levels,
            )
            proposed_group_error = group_error(
                candidate_stats,
                candidate_scale,
                candidate_min,
                proposed_scale_levels,
                candidate_min_levels,
            )
            better = proposed_group_error < old_group_error
            candidate_scale_levels = np.where(
                better, proposed_scale_levels, candidate_scale_levels
            )

            effective_scale = (
                candidate_scale[:, None].astype(np.float64)
                * candidate_scale_levels.astype(np.float64)
            )
            min_denominator = (
                candidate_min[:, None].astype(np.float64) * sum_w
            )
            valid_min = min_denominator > 0.0
            proposed_min_levels = np.clip(
                np.rint(
                    np.divide(
                        effective_scale * sum_q - sum_x,
                        np.where(valid_min, min_denominator, 1.0),
                    )
                ),
                0,
                sub_nmax,
            ).astype(np.int32)
            proposed_min_levels = np.where(
                valid_min, proposed_min_levels, candidate_min_levels
            )
            old_group_error = group_error(
                candidate_stats,
                candidate_scale,
                candidate_min,
                candidate_scale_levels,
                candidate_min_levels,
            )
            proposed_group_error = group_error(
                candidate_stats,
                candidate_scale,
                candidate_min,
                candidate_scale_levels,
                proposed_min_levels,
            )
            better = proposed_group_error < old_group_error
            candidate_min_levels = np.where(
                better, proposed_min_levels, candidate_min_levels
            )

        sum_w, sum_x, _, sum_q, sum_q2, sum_qx = candidate_stats
        scale_level_f = candidate_scale_levels.astype(np.float64)
        min_level_f = candidate_min_levels.astype(np.float64)
        sum_aa = (scale_level_f * scale_level_f * sum_q2).sum(axis=-1)
        sum_mm = (min_level_f * min_level_f * sum_w).sum(axis=-1)
        sum_am = (scale_level_f * min_level_f * sum_q).sum(axis=-1)
        sum_ax = (scale_level_f * sum_qx).sum(axis=-1)
        sum_mx = (min_level_f * sum_x).sum(axis=-1)
        determinant = sum_aa * sum_mm - sum_am * sum_am

        row_best_error = group_error(
            candidate_stats,
            candidate_scale,
            candidate_min,
            candidate_scale_levels,
            candidate_min_levels,
        ).sum(axis=-1)
        row_best_scale = candidate_scale.copy()
        row_best_min = candidate_min.copy()

        def consider(
            scale_value: np.ndarray,
            min_value: np.ndarray,
            current_stats=candidate_stats,
            current_scale_levels=candidate_scale_levels,
            current_min_levels=candidate_min_levels,
        ) -> None:
            nonlocal row_best_error, row_best_scale, row_best_min
            scale_value = fp16_round(scale_value)
            min_value = fp16_round(min_value)
            error = group_error(
                current_stats,
                scale_value,
                min_value,
                current_scale_levels,
                current_min_levels,
            ).sum(axis=-1)
            improve = error < row_best_error
            row_best_error = np.where(improve, error, row_best_error)
            row_best_scale = np.where(
                improve, scale_value, row_best_scale
            )
            row_best_min = np.where(improve, min_value, row_best_min)

        valid_joint = determinant > 0.0
        safe_determinant = np.where(valid_joint, determinant, 1.0)
        joint_scale = (
            sum_mm * sum_ax - sum_am * sum_mx
        ) / safe_determinant
        joint_min = (
            sum_am * sum_ax - sum_aa * sum_mx
        ) / safe_determinant
        joint_scale = np.where(valid_joint, joint_scale, row_best_scale)
        joint_min = np.where(valid_joint, joint_min, row_best_min)
        consider(joint_scale, joint_min)
        consider(
            np.divide(
                sum_ax,
                np.where(sum_aa > 0.0, sum_aa, 1.0),
            ),
            np.zeros_like(sum_ax),
        )
        consider(
            np.zeros_like(sum_mx),
            np.divide(
                -sum_mx,
                np.where(sum_mm > 0.0, sum_mm, 1.0),
            ),
        )
        candidate_scale = row_best_scale
        candidate_min = row_best_min

        effective_scale = (
            candidate_scale[:, None]
            * candidate_scale_levels.astype(np.float32)
        )
        effective_min = (
            candidate_min[:, None]
            * candidate_min_levels.astype(np.float32)
        )
        safe_effective_scale = np.where(
            effective_scale > 0.0, effective_scale, 1.0
        )
        candidate_levels = np.clip(
            np.rint(
                (values + effective_min[..., None])
                / safe_effective_scale[..., None]
            ),
            0,
            nmax,
        ).astype(np.int32)
        candidate_levels = np.where(
            (effective_scale > 0.0)[..., None], candidate_levels, 0
        )
        candidate_stats = statistics(candidate_levels)
        candidate_error = group_error(
            candidate_stats,
            candidate_scale,
            candidate_min,
            candidate_scale_levels,
            candidate_min_levels,
        ).sum(axis=-1)
        improve = candidate_error < best_error
        if not np.any(improve):
            break
        best_error = np.where(improve, candidate_error, best_error)
        levels = np.where(improve[:, None, None], candidate_levels, levels)
        row_scale = np.where(improve, candidate_scale, row_scale)
        row_min = np.where(improve, candidate_min, row_min)
        scale_levels = np.where(
            improve[:, None], candidate_scale_levels, scale_levels
        )
        min_levels = np.where(
            improve[:, None], candidate_min_levels, min_levels
        )
        stats = statistics(levels)

    return levels, row_scale, row_min, scale_levels, min_levels


def quantize(
    weight: np.ndarray,
    spec: NintSpec,
    axis: int = 0,
    importance: np.ndarray | None = None,
) -> NintTensor:
    """沿 ``axis`` 把 ``weight`` 切成 neuron 行，批量 neuron-anchored 量化。

    ``importance`` 是 imatrix 记录的输入通道二阶矩。给定时，组内搜索权重
    与 llama.cpp K-Quant 一致地乘以 ``sqrt(2*mean(W²) + W²)``；不给定时
    保持原有 weight-only 路径不变。
    """

    W = np.asarray(weight, dtype=np.float32)
    if W.ndim < 2:
        raise ValueError(f"nint_quant 需要 ndim>=2 的张量，得到 shape {W.shape}；1D 请用 mfq.formats.nint")
    shape = W.shape
    Wt = np.moveaxis(W, axis, 0)
    out = Wt.shape[0]
    neuron_len = Wt.size // out
    W2 = Wt.reshape(out, neuron_len).copy()

    gs = spec.groupsize
    nmax = spec.nmax
    k = spec.sub_bits
    K = (1 << k) - 1

    pad = (-neuron_len) % gs
    if pad:
        W2 = np.concatenate([W2, np.zeros((out, pad), dtype=np.float32)], axis=1)
    ng = W2.shape[1] // gs
    grps = W2.reshape(out, ng, gs)

    sx2 = (grps * grps).sum(-1)
    if importance is None:
        av = np.sqrt(sx2 / gs)
        w = av[..., None] + np.abs(grps)
        if pad:
            w[:, -1, gs - pad:] = 0.0
    else:
        importance_rows = _importance_as_rows(
            importance, shape, axis, out, neuron_len
        )
        w = _imatrix_element_weights(
            W2, importance_rows, neuron_len
        ).reshape(out, ng, gs)

    if importance is None:
        scale, zp = make_qkx2(grps, w, nmax=nmax)
    else:
        scale, zp = _make_qkx3(grps, w, nmax=nmax)
    the_min = -zp                                   # (out, ng)

    if importance is None:
        neu_s = scale.max(-1)
        neu_m = the_min.max(-1)
        neu_d = np.where(neu_s > 0, (neu_s / K).astype(np.float16).astype(np.float32), np.float32(0.0))
        neu_dm = np.where(neu_m > 0, (neu_m / K).astype(np.float16).astype(np.float32), np.float32(0.0))

        nss = np.where(neu_s > 0, neu_s, 1.0)
        nmm = np.where(neu_m > 0, neu_m, 1.0)
        sub_scale = np.clip(np.rint(K * scale / nss[..., None]), 0, K)
        sub_min = np.clip(np.rint(K * the_min / nmm[..., None]), 0, K)
    else:
        group_weights = w.sum(axis=-1)
        neu_d, sub_scale = _make_qp(
            scale, group_weights, nmax=K
        )
        neu_dm, sub_min = _make_qp(
            the_min, group_weights, nmax=K
        )
        neu_d = neu_d.astype(np.float16).astype(np.float32)
        neu_dm = neu_dm.astype(np.float16).astype(np.float32)

    d_eff = neu_d[:, None] * sub_scale.astype(np.float32)     # (out, ng)
    m_eff = neu_dm[:, None] * sub_min.astype(np.float32)
    de = np.where(d_eff > 0, d_eff, 1.0)
    q = np.clip(np.rint((grps + m_eff[..., None]) / de[..., None]), 0, nmax)
    if importance is not None:
        q, neu_d, neu_dm, sub_scale, sub_min = (
            _refine_imatrix_final_encoding(
                grps,
                w,
                q,
                neu_d,
                neu_dm,
                sub_scale,
                sub_min,
                nmax=nmax,
                sub_nmax=K,
            )
        )

    return NintTensor(
        spec=spec,
        shape=shape,
        axis=axis,
        q=q.astype(_uint_dtype(nmax)),
        neuron_scale=neu_d.astype(np.float32),
        neuron_min=neu_dm.astype(np.float32),
        sub_scale=sub_scale.astype(_uint_dtype(K)),
        sub_min=sub_min.astype(_uint_dtype(K)),
        neuron_len=neuron_len,
    )


def dequantize(tensor: NintTensor) -> np.ndarray:
    """反量化回原始 shape 的 float32 张量。"""

    spec = tensor.spec
    out, ng, gs = tensor.q.shape
    q = tensor.q.astype(np.float32)
    d_eff = tensor.neuron_scale[:, None] * tensor.sub_scale.astype(np.float32)   # (out, ng)
    m_eff = tensor.neuron_min[:, None] * tensor.sub_min.astype(np.float32)
    recon = d_eff[..., None] * q - m_eff[..., None]                              # (out, ng, gs)
    recon = recon.reshape(out, ng * gs)[:, :tensor.neuron_len]                  # 去尾组补零
    S, a = tensor.shape, tensor.axis
    wt_shape = (S[a],) + S[:a] + S[a + 1:]
    return np.moveaxis(recon.reshape(wt_shape), 0, a)
