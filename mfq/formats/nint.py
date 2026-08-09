"""NINT -- Neuron-anchored INT quantization, MFQ's core weight format.

Two-level affine INT whose top-level scale is anchored to a **neuron** (the complete weight row of one output neuron)
rather than an arbitrary superblock:

    x ≈ (d_neuron · ls) · q − (dmin_neuron · lm)

- ``d_neuron``, ``dmin_neuron``: one f16 scale/minimum per neuron, shared across the full row.
- ``ls``, ``lm``: one k-bit sub-scale/sub-minimum per group, relative to the neuron maximum.
- ``q``: one b-bit INT per element.

Per-group scale/minimum values are searched by weighted least squares (:func:`make_qkx2`), using the same optimizer
as llama.cpp K-quant. Neuron anchoring exploits per-neuron magnitude structure and measures about 0.5 dB higher than
superblock-anchored Q4_K at equal bpw on real LLM weights (INT4/gs=24/k=6 -> 23.4 dB at 4.5 bpw).

Theoretical bpw = bits + 32/neuron_len + 2*sub_bits/groupsize
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NintSpec:
    """Neuron-anchored INT quantization specification; the quantizer searches these three controls per tensor.

    Attributes:
        bits: Bits per element (2, 3, 4, 5, ...).
        groupsize: Sub-group size gs (24 is the INT4 sweet spot). It need not divide neuron_len;
            the trailing group is zero-padded, and padded positions receive zero weight during optimization.
        sub_bits: Number of sub-scale bits k (6 is the INT4 sweet spot). Excessive values make
            d_neuron=neu_s/K fall into the f16 subnormal range and degrade; the practical upper limit is about 8.
    """

    bits: int = 4
    groupsize: int = 24
    sub_bits: int = 6

    @property
    def nmax(self) -> int:
        return (1 << self.bits) - 1

    def bpw(self, neuron_len: int) -> float:
        """Theoretical bits per weight, assuming sub-byte bit packing."""
        return self.bits + 32.0 / neuron_len + 2.0 * self.sub_bits / self.groupsize

    @property
    def profile_label(self) -> str:
        """Profile label for kernel dispatch, such as ``NINT4-24``; includes bits/gs but not k."""
        return profile_label(self.bits, self.groupsize)


# Fixed point from the three-dimensional Pareto search on 2026-07-26:
# groupsize=16, scale_bits=min_bits=sub_bits=5, weighted-MSE。
# 2.63125 bpw at K=5120; 11.1920 dB over 256 rows x 10 Qwen3.6-27B matrices.
NINT2_SPEC = NintSpec(bits=2, groupsize=16, sub_bits=5)


# Fixed profile catalog: (bits, groupsize). k is free (baked into neuron_scale and invisible to the kernel).
# Runtime kernels dispatch by profile, with one kernel variant per (bits, gs).
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
    """Return a profile label such as ``NINT4-24``."""
    return f"NINT{bits}-{groupsize}"


@dataclass
class NintCode:
    """Compact representation of one neuron row after NINT quantization.

    Storage convention B, matching validation experiments: ``neuron_scale = f16(neu_s / K)``.
    Dequantization uses ``d_eff = neuron_scale * sub_scale``, where sub_scale is in [0, K].
    """

    spec: NintSpec
    n: int                       # Number of valid elements (<= stored q length; excludes trailing-group padding)
    q: np.ndarray                # uint per element; length = ng*gs (including trailing-group padding)
    neuron_scale: np.float32     # neu_s/K after an f16 round trip
    neuron_min: np.float32       # neu_m/K after an f16 round trip (the_min/K, >=0)
    sub_scale: np.ndarray        # k-bit uint per group
    sub_min: np.ndarray          # k-bit uint per group


def _uint_dtype(maxval: int) -> np.dtype:
    return np.dtype(np.uint8) if maxval <= 255 else np.dtype(np.uint16)


def _qkx2_search(nmax: int) -> tuple[float, float, int]:
    """Choose make_qkx2 search parameters by nmax, matching llama.cpp Q4_K/Q5_K values."""
    if nmax <= 15:
        return -1.0, 0.1, 20
    return -0.5, 0.1, 15


def make_qkx2(x: np.ndarray, w: np.ndarray, nmax: int = 15,
              rmin: float | None = None, rdelta: float = 0.1,
              nstep: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Use weighted least squares to find the best affine (scale, zp) independently for each group on the final dimension.

    x, w: ``(..., gs)``. Returns ``(scale, zp)`` and reconstructs ``recon = zp + scale*L``, where
    ``L`` is in [0, nmax] and ``zp <= 0``. Weights ``w`` typically use ``av_x + |x|``, where ``av_x``
    is the group RMS. Elements with ``w == 0`` are ignored, which supports trailing-group padding.
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
    """Quantize one 1D neuron row, containing all weights for that neuron, into :class:`NintCode`."""

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
    """Dequantize :class:`NintCode` to 1D float32 of length ``code.n``."""

    gs = code.spec.groupsize
    k = code.spec.sub_bits
    K = (1 << k) - 1
    ng = code.q.size // gs
    q = code.q.reshape(ng, gs).astype(np.float32)
    d_eff = code.neuron_scale * code.sub_scale.astype(np.float32)   # (ng,)
    m_eff = code.neuron_min * code.sub_min.astype(np.float32)       # (ng,)
    recon = d_eff[:, None] * q - m_eff[:, None]
    return recon.reshape(-1)[:code.n]
