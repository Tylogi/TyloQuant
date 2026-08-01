"""逐张量 NintSpec 搜索（profile-based）。

给定权重与目标 bpw，在 **固定 profile 目录**（`(bits, gs)` 对，见
:data:`mfq.formats.nint.PROFILE_CATALOG`）中为每个 profile 搜自由的 ``k ∈ [2,8]``，
取 SNR 最高的 :class:`~mfq.formats.nint.NintSpec`。这是 MFQ「逐张量混合精度」的核心
（开发文档 v2 §1.7、§2.2）。

为什么固定 gs、自由 k：
- ``gs`` 决定 kernel tiling，每个不同 gs 需要一个专用 kernel → 限制 gs 种类 = 控制 kernel 数。
- ``k`` 烘焙进 ``neuron_scale = f16(neu_s/(2^k−1))``，反量化算术里不出现 → **k 对 kernel
  不可见，可任意变**。故 bpw 细粒度由 k 连续提供，不增加 kernel。

搜索对整个矩阵做 quantize/dequantize 评估 SNR（不采样子集），保证结果严谨可复现。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mfq.formats.nint import PROFILE_CATALOG, NintSpec
from mfq.quantize import nint_quant
from mfq.utils.tensor import snr

_K_MIN, _K_MAX = 2, 8


@dataclass(frozen=True)
class SearchResult:
    """搜索结果：最优 spec、其 SNR、bpw，以及全部候选。

    ``evaluated`` 为 ``(spec, snr_db, bpw)`` 三元组列表，按 SNR 降序排列。
    SNR 由对**整张权重**量化/反量化后与原值比较得到（非子采样）。
    """

    spec: NintSpec
    snr_db: float
    bpw: float
    evaluated: list[tuple[NintSpec, float, float]]


def _eval_spec(W: np.ndarray, spec: NintSpec, axis: int) -> float:
    return snr(W, nint_quant.dequantize(nint_quant.quantize(W, spec, axis=axis)))


def search(
    weight: np.ndarray,
    target_bpw: float,
    axis: int = 0,
    profiles: tuple[tuple[int, int], ...] = PROFILE_CATALOG,
) -> SearchResult:
    """对每个 profile 搜自由 k，返回预算下 SNR 最优的完整 :class:`SearchResult`。

    全量评估：对 ``weight`` 整张做量化/反量化计算 SNR，不做行子采样。
    """

    W = np.asarray(weight, dtype=np.float32)
    if W.ndim < 2:
        raise ValueError(f"search 需要 ndim>=2 的张量，得到 shape {W.shape}")
    Wt = np.moveaxis(W, axis, 0)
    out = Wt.shape[0]
    neuron_len = Wt.size // out

    evaluated: list[tuple[NintSpec, float, float]] = []
    for bits, gs in profiles:
        for k in range(_K_MIN, _K_MAX + 1):
            s = NintSpec(bits=bits, groupsize=gs, sub_bits=k)
            b = s.bpw(neuron_len)
            if b > target_bpw + 1e-9:
                continue
            evaluated.append((s, _eval_spec(W, s, axis), b))

    if not evaluated:
        raise ValueError(
            f"目标 bpw {target_bpw} 在 profiles={profiles} 下无合法 (bits,gs,k)"
        )
    evaluated.sort(key=lambda t: t[1], reverse=True)
    best_spec_, best_snr, best_bpw = evaluated[0]
    return SearchResult(spec=best_spec_, snr_db=best_snr, bpw=best_bpw, evaluated=evaluated)


def best_spec(
    weight: np.ndarray,
    target_bpw: float,
    axis: int = 0,
    profiles: tuple[tuple[int, int], ...] = PROFILE_CATALOG,
) -> NintSpec:
    """在 ``≤ target_bpw`` 候选中返回 SNR 最高的 NintSpec。"""

    return search(weight, target_bpw, axis, profiles).spec
