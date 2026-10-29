"""Shared-column, 16-state-bit TQ fitting with sparse scale repair."""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache
from pathlib import Path

import torch
from mfq.formats.tq import fixed_codebook

_BOOK_BOUND = float(abs(fixed_codebook()).max())

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _path_metric_kernel(X, M, Out, BOOK_BOUND: tl.constexpr):
        ring = tl.program_id(0)
        offsets = ring * 256 + tl.arange(0, 256)
        x, metric = tl.load(X + offsets), tl.load(M + offsets)
        peak = tl.max((tl.abs(x) + BOOK_BOUND) * tl.abs(metric), axis=0)
        exponent = tl.ceil(tl.log2(tl.maximum(peak / 32., 1.)))
        tl.store(Out + offsets, metric * tl.exp2(-exponent))


def _path_metric(x, metric):
    """Rescale each independent ring's objective to avoid FP16 path overflow.

    Powers of two preserve the mathematical minimizer. A branch residual is
    bounded by 32; the trellis forgets its initial state within 16 steps and
    normalizes every 16 steps, keeping reachable costs below FP16 overflow.
    Normal rings retain their exact input metric. No reconstruction range is
    imposed and no stored tensor, codebook or state representation changes.
    """
    bound = _BOOK_BOUND
    if x.is_cuda and triton is not None:
        result = torch.empty_like(metric)
        _path_metric_kernel[(len(x),)](x, metric, result, bound, num_warps=4)
        return result
    peak = ((x.abs() + bound) * metric.abs()).amax(-1, keepdim=True)
    exponent = torch.ceil(torch.log2((peak / 32.).clamp_min(1.)))
    return metric * torch.exp2(-exponent)


@lru_cache(None)
def _library():
    from torch.utils.cpp_extension import load
    path = load(name="mfq_tq_columns_v1",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3", "--fmad=false"], is_python_module=False)
    lib = ct.CDLL(path)
    lib.half_wave.argtypes = [ct.c_int] * 2
    lib.half_wave.restype = ct.c_int
    lib.half_encode.argtypes = [ct.c_void_p] * 6 + [ct.c_int] * 4 + [ct.c_void_p]
    lib.half_encode.restype = ct.c_int
    return lib


def encode(x, metric, bits):
    if (bits not in range(1, 9) or x.ndim != 2 or x.shape[1] != 256 or not len(x)
            or not x.is_cuda or x.dtype != torch.float32 or metric.dtype != torch.float32
            or x.shape != metric.shape or x.device != metric.device
            or not x.is_contiguous() or not metric.is_contiguous()):
        raise ValueError("Expected matching contiguous FP32 [rings,256] tensors and 1..8 bits")
    metric = _path_metric(x, metric)
    if torch.version.hip:
        from .tq_fitter import encode as hip_encode
        with torch.cuda.device(x.device):
            return hip_encode(x, metric, bits, fast_columns=True, column_scaling=True)
    with torch.cuda.device(x.device):
        lib = _library()
        wave = lib.half_wave(len(x), bits)
        if wave <= 0:
            raise RuntimeError(f"TQ column fitter occupancy failed: {wave}")
        wave = min(wave, 4096)
        divisor = 8 if bits == 1 else 2 if bits in (2, 3) else 1
        history = torch.empty((wave, 256 * (1 << (16 - bits)) // divisor),
                              device=x.device, dtype=torch.uint8)
        y = torch.empty_like(x)
        states = torch.empty_like(x, dtype=torch.int16)
        bounds = torch.empty(len(x), device=x.device, dtype=torch.int32)
        code = lib.half_encode(*(v.data_ptr() for v in (x, metric, y, states, history, bounds)),
            len(x), bits, wave, 16, torch.cuda.current_stream(x.device).cuda_stream)
        if code:
            raise RuntimeError(f"TQ column fitter returned CUDA error {code}")
        return y, states, bounds


@lru_cache(None)
def _constants(device):
    table = torch.tensor(fixed_codebook(), device=device).float()
    values = table.square().mean().sqrt(), table.min(), table.max()
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    return values, ready


def _get_constants(device):
    values, ready = _constants(device)
    torch.cuda.current_stream(device).wait_event(ready)
    return values


@lru_cache(None)
def _factors(device, one_bit):
    factors = ((.65, .7, .75, .8, .85, .9, .95, 1., 1.05, 1.1, 1.2, 1.3)
               if one_bit else (1., .9, 1.05, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.))
    values = torch.tensor(factors, device=device)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    return values, ready


def _get_factors(device, one_bit):
    values, ready = _factors(device, one_bit)
    torch.cuda.current_stream(device).wait_event(ready)
    return values


@torch.inference_mode()
def prepare_columns(weight):
    """One precision-independent scale vector shared by every row/candidate."""
    w = weight.float()
    rms, _, _ = _get_constants(w.device)
    columns = (w.square().mean(0).sqrt() / w.square().mean().sqrt().clamp_min(2**-24)).clamp_min(.01).half()
    for _ in range(2):
        anchor = ((w / columns.float()).square().mean(-1).sqrt() / rms).clamp_min(2**-24).half()
        columns = ((w / anchor.float()[:, None]).square().mean(0).sqrt() / rms).clamp_min(.01).half()
    return columns


class ColumnFitter:
    """Reuse bit-independent inputs with the upstream zero-clipping policy."""

    @torch.inference_mode()
    def __init__(self, weight, h, columns, *, state_bits=16, fast=False,
                 repair_scales=True, allow_clipping=False):
        if state_bits != 16:
            raise ValueError("This backend requires complete 16-bit states")
        self.w = weight.float()
        self.h = h
        self.columns = columns.float()
        self.root_h = h.sqrt()
        self.fast, self.repair_scales = fast, repair_scales
        rows, _ = self.w.shape
        rms, low, high = _get_constants(self.w.device)
        self.row_rms = (self.w / self.columns).square().mean(-1).sqrt() / rms
        self.anchor = self.row_rms.clamp_min(2**-24).half()
        self.floor = None
        if not allow_clipping:
            bound = (self.w.abs() / (torch.where(self.w >= 0, high, -low) * self.columns[None, :])).amax(-1)
            floor = bound.half()
            self.floor = torch.where(floor.float() < bound,
                torch.nextafter(floor, torch.full_like(floor, torch.inf)), floor)
        self.ids = torch.linspace(0, rows - 1, min(rows, 16 if fast else 64), device=self.w.device).long()
        self.sample = self.w[self.ids]
        self.sample_targets, self.sample_floors = {}, {}

    @torch.inference_mode()
    def fit(self, k):
        w, h, columns, anchor, floor = self.w, self.h, self.columns, self.anchor, self.floor
        rows, width = w.shape
        factor_table = _get_factors(w.device, k == 1)
        count = len(factor_table)

        def trial(target, a, lower=None):
            if lower is not None:
                a = torch.maximum(a, lower)
            scale = a.float()[:, None] * columns[None, :]
            metric = scale * self.root_h[None, :]
            metric /= metric.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-30)
            x = (target / scale).reshape(-1, 256).contiguous()
            y, state, _ = encode(x, metric.reshape(-1, 256).contiguous(), k)
            basis = y.reshape_as(target) * columns[None, :]
            proposal = ((target * basis * h).sum(-1) /
                        (basis.square() * h).sum(-1).clamp_min(1e-30))
            proposed = proposal.half()
            minimum = a.float() * x.reshape_as(target).abs().amax(-1) / 65504.
            valid = torch.isfinite(proposed) & (proposed.float() > minimum)
            proposed = torch.where(valid, proposed, a)
            if lower is not None:
                proposed = torch.maximum(proposed, lower)
            old = ((target - (basis * a.float()[:, None]).half().float()).square() * h).sum(-1)
            new = ((target - (basis * proposed.float()[:, None]).half().float()).square() * h).sum(-1)
            return state.reshape_as(target), torch.where(new < old, proposed, a), torch.minimum(new, old)

        sample_a = (anchor[self.ids].float()[None, :] * factor_table[:, None]).half().flatten()
        if count not in self.sample_targets:
            self.sample_targets[count] = self.sample.repeat(count, 1).contiguous()
            self.sample_floors[count] = None if floor is None else floor[self.ids].repeat(count)
        _, _, loss = trial(self.sample_targets[count], sample_a, self.sample_floors[count])
        order = loss.reshape(count, len(self.ids)).sum(-1).argsort()
        if self.fast:
            selected = factor_table[order[:1]]
            states, anchors, best = trial(w, (anchor.float() * selected).half().clamp_min(2**-24), floor)
            anchors = torch.where(best < torch.inf, anchors, anchor)
        else:
            best = torch.full((rows,), torch.inf, device=w.device)
            states = torch.empty_like(w, dtype=torch.int16)
            anchors = anchor.clone()
            for index in order[:2]:
                state, a, loss = trial(w, (anchor.float() * factor_table[index]).half().clamp_min(2**-24), floor)
                take = loss < best
                best[take], anchors[take], states[take] = loss[take], a[take], state[take]
            state, a, loss = trial(w, anchors.clone(), floor)
            take = loss < best
            best[take], anchors[take], states[take] = loss[take], a[take], state[take]
        if self.repair_scales:
            from .tq_scale_repair import propose_scales
            proposals = propose_scales(w, h, columns, anchors, best)
            ids = torch.nonzero(proposals > anchors).flatten()
            if len(ids):
                state, a, loss = trial(w[ids].contiguous(), proposals[ids],
                                      None if floor is None else floor[ids])
                take = loss < best[ids]
                selected = ids[take]
                anchors[selected], states[selected] = a[take], state[take]
        anchors.masked_fill_(self.row_rms == 0, 0)
        return states, anchors


@torch.inference_mode()
def fit_columns(weight, k, h, columns, *, state_bits=16, fast=False,
                repair_scales=True, allow_clipping=False):
    """Fit fixed-column rows with full source-range coverage by default."""
    return ColumnFitter(weight, h, columns, state_bits=state_bits, fast=fast,
                        repair_scales=repair_scales, allow_clipping=allow_clipping).fit(k)
