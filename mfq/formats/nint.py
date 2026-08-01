"""NINT — Neuron-anchored INT quantization (MFQ 核心权重格式).

两级仿射 INT，顶层 scale 锚定在一个 **Neuron**（一条输出神经元的整条权重行），
而非任意 superblock：

    x ≈ (d_neuron · ls) · q − (dmin_neuron · lm)

- ``d_neuron``, ``dmin_neuron``：每个神经元一个 f16 scale/min（整行共享）。
- ``ls``, ``lm``：每组 k-bit sub-scale/sub-min（相对 neuron max 的比值）。
- ``q``：每元素 b-bit INT。

per-group scale/min 由加权最小二乘 (:func:`make_qkx2`) 搜索——与 llama.cpp
K-quant 同款优化器。Neuron 锚定利用 per-neuron 幅度结构，实测在真实 LLM 权重上
同等 bpw 比 superblock 锚定的 Q4_K 高约 0.5 dB（INT4/gs=24/k=6 → 23.4 dB @ 4.5 bpw）。

理论 bpw = bits + 32/neuron_len + 2·sub_bits/groupsize
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NintSpec:
    """Neuron-anchored INT 量化规格（量化器逐张量搜索这三个旋钮）。

    Attributes:
        bits: 每元素比特数（2、3、4、5、…）。
        groupsize: sub-group 大小 gs（INT4 甜点 24；不必整除 neuron_len，
            尾组自动补零、补零位在优化时置零权重忽略）。
        sub_bits: sub-scale 位数 k（INT4 甜点 6；过大会让 d_neuron=neu_s/K
            落入 f16 次正规区而退化，上限约 8）。
    """

    bits: int = 4
    groupsize: int = 24
    sub_bits: int = 6

    @property
    def nmax(self) -> int:
        return (1 << self.bits) - 1

    def bpw(self, neuron_len: int) -> float:
        """理论 bits-per-weight（假设 sub-byte 位打包）。"""
        return self.bits + 32.0 / neuron_len + 2.0 * self.sub_bits / self.groupsize

    @property
    def profile_label(self) -> str:
        """kernel 分派用的 profile 标签，如 ``NINT4-24``（只看 bits/gs，不含 k）。"""
        return profile_label(self.bits, self.groupsize)


# 2026-07-26 三维 Pareto 搜索固定点：
# groupsize=16, scale_bits=min_bits=sub_bits=5, weighted-MSE。
# K=5120 时 2.63125 bpw；256 行 × 10 个 Qwen3.6-27B 矩阵上为 11.1920 dB。
NINT2_SPEC = NintSpec(bits=2, groupsize=16, sub_bits=5)


# 固定 profile 目录：(bits, groupsize)。k 自由（烘焙进 neuron_scale，kernel 不可见）。
# runtime kernel 按 profile 分派，每个 (bits, gs) 一个 kernel 变体。
PROFILE_CATALOG: tuple[tuple[int, int], ...] = (
    (2, 16),
    *((4, gs) for gs in (16, 24, 32, 48, 64)),
    *((5, gs) for gs in (16, 24, 32, 48, 64)),
    *((6, gs) for gs in (20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 48, 64)),
    *((8, gs) for gs in (16, 24, 32, 48, 64)),
    (3, 24),
)

RUNTIME_PROFILE_CATALOG: tuple[tuple[int, int], ...] = (
    (2, 16),
    (4, 16),
    (4, 24),
    (4, 32),
    (4, 48),
    *((5, gs) for gs in (16, 24, 32, 48, 64)),
    *((6, gs) for gs in (20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 48, 64)),
    *((8, gs) for gs in (16, 24, 32, 48, 64)),
    (3, 24),
)


def profile_label(bits: int, groupsize: int) -> str:
    """返回 profile 标签，如 ``NINT4-24``。"""
    return f"NINT{bits}-{groupsize}"


@dataclass
class NintCode:
    """一个 neuron 行经 NINT 量化后的紧凑表示。

    存储约定（convention B，与验证实验一致）：``neuron_scale = f16(neu_s / K)``，
    反量化 ``d_eff = neuron_scale · sub_scale``，其中 sub_scale ∈ [0, K]。
    """

    spec: NintSpec
    n: int                       # 有效元素数（≤ 实际存储 q 的长度；尾组补零部分不算）
    q: np.ndarray                # uint，每元素；长度 = ng·gs（含尾组补零）
    neuron_scale: np.float32     # f16-round-tripped 的 neu_s/K
    neuron_min: np.float32       # f16-round-tripped 的 neu_m/K（the_min/K，≥0）
    sub_scale: np.ndarray        # uint，每组，k-bit
    sub_min: np.ndarray          # uint，每组，k-bit


def _uint_dtype(maxval: int) -> np.dtype:
    return np.dtype(np.uint8) if maxval <= 255 else np.dtype(np.uint16)


def _qkx2_search(nmax: int) -> tuple[float, float, int]:
    """make_qkx2 搜索参数随 nmax 取（对齐 llama.cpp Q4_K/Q5_K 的取值）。"""
    if nmax <= 15:
        return -1.0, 0.1, 20
    return -0.5, 0.1, 15


def make_qkx2(x: np.ndarray, w: np.ndarray, nmax: int = 15,
              rmin: float | None = None, rdelta: float = 0.1,
              nstep: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """加权最小二乘搜索最优仿射 (scale, zp)，每组（最后一维）独立。

    x, w: ``(..., gs)``。返回 ``(scale, zp)``，重构 ``recon = zp + scale·L``，
    ``L ∈ [0, nmax]``，``zp ≤ 0``。权重 ``w`` 典型取 ``av_x + |x|``
    （``av_x`` 为组内 RMS）；``w == 0`` 的元素被忽略（用于尾组补零）。
    """

    if rmin is None:
        rmin = -1.0 if nmax <= 15 else -0.5
    if nstep is None:
        nstep = 20 if nmax <= 15 else 15

    mn = np.minimum(x.min(-1), 0.0)
    mx = x.max(-1)
    sum_w = w.sum(-1)
    sum_x = (w * x).sum(-1)
    degen = (mx == mn) | (sum_w <= 0)
    rng = np.where(degen, 1.0, mx - mn)

    iscale0 = nmax / rng
    scale0 = 1.0 / iscale0
    L0 = np.clip(np.rint(iscale0[..., None] * (x - mn[..., None])), 0, nmax).astype(np.int32)
    diff = scale0[..., None] * L0 + mn[..., None] - x
    best_err = (w * diff * diff).sum(-1)
    best_scale = scale0.copy()
    best_min = mn.copy()

    for is_ in range(nstep + 1):
        iscale = (rmin + rdelta * is_ + nmax) / rng
        Laux = np.clip(np.rint(iscale[..., None] * (x - mn[..., None])), 0, nmax).astype(np.int32)
        Laf = Laux.astype(np.float32)
        sl = (w * Laf).sum(-1)
        sl2 = (w * Laf * Laf).sum(-1)
        sxl = (w * Laf * x).sum(-1)
        D = sum_w * sl2 - sl * sl
        valid = D > 0
        Ds = np.where(valid, D, 1.0)
        ts = (sum_w * sxl - sum_x * sl) / Ds
        tm = (sl2 * sum_x - sl * sxl) / Ds
        pos = tm > 0
        sl2s = np.where(sl2 > 0, sl2, 1.0)
        ts = np.where(pos, sxl / sl2s, ts)
        tm = np.where(pos, 0.0, tm)
        cd = ts[..., None] * Laux + tm[..., None] - x
        ce = (w * cd * cd).sum(-1)
        better = valid & (ce < best_err)
        best_err = np.where(better, ce, best_err)
        best_scale = np.where(better, ts, best_scale)
        best_min = np.where(better, tm, best_min)

    best_scale = np.where(degen, 0.0, best_scale)
    best_min = np.where(degen, np.minimum(mn, 0.0), best_min)
    return best_scale.astype(np.float32), best_min.astype(np.float32)


def quantize(x: np.ndarray, spec: NintSpec) -> NintCode:
    """量化一个 1D neuron 行（该神经元整条权重）为 :class:`NintCode`。"""

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    gs = spec.groupsize
    nmax = spec.nmax
    k = spec.sub_bits
    K = (1 << k) - 1
    n_real = x.size

    pad = (-n_real) % gs
    if pad:
        x = np.concatenate([x, np.zeros(pad, dtype=np.float32)])
    ng = x.size // gs
    grps = x.reshape(ng, gs)

    sx2 = (grps * grps).sum(-1)
    av = np.sqrt(sx2 / gs)
    w = av[..., None] + np.abs(grps)
    if pad:
        w[ -1, gs - pad:] = 0.0

    scale, zp = make_qkx2(grps, w, nmax=nmax)   # recon = zp + scale·L
    the_min = -zp                                  # ≥ 0

    neu_s = float(scale.max())
    neu_m = float(the_min.max())
    neu_d = np.float16(neu_s / K).astype(np.float32) if neu_s > 0 else np.float32(0.0)
    neu_dm = np.float16(neu_m / K).astype(np.float32) if neu_m > 0 else np.float32(0.0)

    if neu_s > 0:
        sub_scale = np.clip(np.rint(K * scale / neu_s), 0, K)
    else:
        sub_scale = np.zeros(ng, dtype=np.float64)
    if neu_m > 0:
        sub_min = np.clip(np.rint(K * the_min / neu_m), 0, K)
    else:
        sub_min = np.zeros(ng, dtype=np.float64)

    d_eff = neu_d * sub_scale.astype(np.float32)    # (ng,) ≈ scale
    m_eff = neu_dm * sub_min.astype(np.float32)     # (ng,) ≈ the_min
    de = np.where(d_eff > 0, d_eff, 1.0)
    q = np.clip(np.rint((grps + m_eff[..., None]) / de[..., None]), 0, nmax)
    q = q.astype(_uint_dtype(nmax)).reshape(-1)

    return NintCode(
        spec=spec,
        n=n_real,
        q=q,
        neuron_scale=neu_d,
        neuron_min=neu_dm,
        sub_scale=sub_scale.astype(_uint_dtype(K)),
        sub_min=sub_min.astype(_uint_dtype(K)),
    )


def dequantize(code: NintCode) -> np.ndarray:
    """将 :class:`NintCode` 反量化回 1D float32（长度 = ``code.n``）。"""

    gs = code.spec.groupsize
    k = code.spec.sub_bits
    K = (1 << k) - 1
    ng = code.q.size // gs
    q = code.q.reshape(ng, gs).astype(np.float32)
    d_eff = code.neuron_scale * code.sub_scale.astype(np.float32)   # (ng,)
    m_eff = code.neuron_min * code.sub_min.astype(np.float32)       # (ng,)
    recon = d_eff[:, None] * q - m_eff[:, None]
    return recon.reshape(-1)[:code.n]
