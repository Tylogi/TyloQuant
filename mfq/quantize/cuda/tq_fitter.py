"""Full-state TQ fitting with paired FP16 costs and sampled scale selection."""

from __future__ import annotations
import ctypes as ct
from functools import lru_cache
from pathlib import Path
import torch
from mfq.formats.tq import RELATIVE, fixed_codebook


def _library(state_bits=16, fast_columns=False, *, columns=False):
    production = bool(columns or fast_columns) if torch.version.hip else False
    return _load_library(state_bits, bool(fast_columns), production)


@lru_cache(None)
def _load_library(state_bits, fast_columns, columns):
    if state_bits != 16:
        raise ValueError("This backend requires complete 16-bit states")
    from torch.utils.cpp_extension import load

    hip = bool(torch.version.hip)
    production = columns or fast_columns
    source = (Path(__file__).parents[1] / "hip" /
              ("tq_fitter_columns.hip" if production else "tq_fitter.hip")
              if hip else Path(__file__).with_suffix(".cu"))
    flags = (["-O3", "-ffp-contract=off", "-U__HIP_NO_HALF_OPERATORS__",
              "-U__HIP_NO_HALF_CONVERSIONS__"] if hip else ["-O3", "--fmad=false"])
    flags += [f"-DMFQ_TQ_STATE_BITS={state_bits}",
              f"-DMFQ_TQ_WARMUP={16 if fast_columns else 128}"]
    name = ("mfq_tq_fitter_hip" + ("_columns" if production else "") if hip
            else "mfq_tq_fitter_cuda_production")
    path = load(name=name + ("_fast" if fast_columns else ""),
                sources=[str(source)], extra_cuda_cflags=flags,
                is_python_module=False)
    lib = ct.CDLL(path)
    P = ct.c_void_p
    I = ct.c_int
    lib.half_wave.argtypes = [I] * 2
    lib.half_wave.restype = I
    lib.half_encode.argtypes = [P] * 6 + [I] * 4 + [P] * (2 if hip else 1)
    lib.half_encode.restype = I
    lib.fit_refine.argtypes = [P] * 9 + [I] * 3 + [P]
    lib.fit_refine.restype = I
    lib.pack.argtypes = [P] * 2 + [I] * 2 + [P]
    lib.pack.restype = I
    return lib


def _status(code):
    if code:
        raise RuntimeError(f"TQ fitter returned GPU error {code}")


def encode(x, metric, bits, *, state_bits=16, fast_columns=False, column_scaling=False):
    lib = _library(state_bits, fast_columns and state_bits == 16, columns=column_scaling)
    n = len(x)
    wave = lib.half_wave(n, bits)
    if wave <= 0:
        raise RuntimeError(f"TQ fitter occupancy failed: {wave}")
    # User ceiling applies inside the native fitting path as well.
    if not torch.version.hip:
        wave = min(wave, 4096)
    trace_divisor = 8 if bits == 1 else (
        2 if fast_columns and state_bits == 16 and bits in (2, 3) else 1
    )
    if torch.version.hip and not (column_scaling or fast_columns) and bits in (2, 3, 4):
        trace_divisor = 2
    history = torch.empty(
        (wave, 256 * (1 << (state_bits - bits)) // trace_divisor),
        device=x.device,
        dtype=torch.uint8,
    )
    y = torch.empty_like(x)
    states = torch.empty_like(x, dtype=torch.int16)
    bounds = torch.empty(n, device=x.device, dtype=torch.int32)
    scratch_args = []
    if torch.version.hip:
        scratch = torch.empty((wave, 512 if bits == 1 else 1),
                              device=x.device, dtype=torch.float16)
        scratch_args.append(scratch.data_ptr())
    _status(
        lib.half_encode(
            *(v.data_ptr() for v in (x, metric, y, states, history, bounds)),
            n,
            bits,
            wave,
            16,
            *scratch_args,
            torch.cuda.current_stream().cuda_stream,
        )
    )
    return y, states, bounds


def pack(state, bits):
    state = state.reshape(-1, 256).contiguous()
    out = torch.empty(len(state) * 32 * bits, device=state.device, dtype=torch.uint8)
    _status(
        _library(16, True).pack(
            state.data_ptr(),
            out.data_ptr(),
            len(state),
            bits,
            torch.cuda.current_stream().cuda_stream,
        )
    )
    return out


# Keep the historical entry points on the same search policy as quantize().
from .tq_columns import ColumnFitter, fit_columns, prepare_columns


@torch.inference_mode()
def fit_rows(weight, k, h, gs, *, state_bits=16, fast=False):
    finalists, refine = (1, 0) if fast else (0, 1)
    rows, width = weight.shape
    w = weight.float()
    h = h.float()
    relative = torch.tensor(RELATIVE, device=w.device)
    rel = relative.float()
    lut = torch.tensor(fixed_codebook(), device=w.device)
    rms = lut.float().square().mean().sqrt()
    a = (w.square().mean(-1).sqrt() / rms).clamp_min(2**-24).half()
    desired = w.reshape(rows, -1, gs).square().mean(-1).sqrt() / rms / a.float()[:, None]
    codes = (desired.clamp_min(2**-24).log2()[..., None] - rel.log2()).abs().argmin(-1)
    factors = (
        (0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3)
        if k == 1
        else (1.0, 0.9, 1.05, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0)
    )
    lib = _library(state_bits)
    stream = torch.cuda.current_stream().cuda_stream
    sample_ids = torch.linspace(0, rows - 1, min(rows, 16 if fast else 64), device=w.device).long()
    count = len(sample_ids)
    nf = len(factors)
    sample_w = w[sample_ids].repeat(nf, 1).contiguous()
    sample_a = (
        (a[sample_ids].float()[None, :] * torch.tensor(factors, device=w.device)[:, None])
        .half()
        .flatten()
    )
    sample_c = codes[sample_ids].byte().repeat(nf, 1).contiguous()
    screen_best = torch.full((nf * count,), torch.inf, device=w.device)
    screen_state = torch.empty_like(sample_w, dtype=torch.int16)
    for screen_step in range(1 if fast else 2):
        expanded = sample_a.float()[:, None] * rel[sample_c.long()].repeat_interleave(gs, -1)
        metric = expanded * h.sqrt()[None, :]
        metric = metric / metric.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-30)
        y, st, _ = encode(
            (sample_w / expanded).reshape(-1, 256).contiguous(),
            metric.reshape(-1, 256).contiguous(),
            k, state_bits=state_bits,
        )
        if screen_step == 0:
            original_loss = (
                ((sample_w - (y.reshape_as(sample_w) * expanded).half().float()).square() * h)
                .sum(-1)
                .reshape(nf, count)
                .T
            )
        _status(
            lib.fit_refine(
                *[
                    v.data_ptr()
                    for v in (
                        sample_w,
                        h,
                        y,
                        st,
                        sample_a,
                        sample_c,
                        screen_best,
                        screen_state,
                        relative,
                    )
                ],
                nf * count,
                width,
                gs,
                stream,
            )
        )
    sample_losses = screen_best.reshape(nf, count).T
    global_order = sample_losses.sum(0).argsort()
    if finalists == 0:
        reference_loss = original_loss.min(-1).values
        prefix = sample_losses[:, global_order].cummin(-1).values
        sufficient = prefix.sum(0) <= reference_loss.sum() * (1 + 1e-6)
        chosen = torch.nonzero(sufficient).flatten()
        finalists = int(chosen[0]) + 1 if len(chosen) else nf
    order = global_order[:finalists][None, :].expand(rows, -1)

    factor_table = torch.tensor(factors, device=w.device)
    best = torch.full((rows,), torch.inf, device=w.device)
    ba = a.clone()
    bc = codes.byte().clone()
    bs = torch.empty_like(w, dtype=torch.int16)
    lib = _library(state_bits)
    stream = torch.cuda.current_stream().cuda_stream

    def full_fit(trial, cs):
        expanded = trial.float()[:, None] * rel[cs.long()].repeat_interleave(gs, -1)
        metric = expanded * h.sqrt()[None, :]
        metric = metric / metric.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-30)
        y, st, _ = encode(
            (w / expanded).reshape(-1, 256).contiguous(), metric.reshape(-1, 256).contiguous(), k,
            state_bits=state_bits,
        )
        loss = ((w - (y.reshape_as(w) * expanded).half().float()).square() * h).sum(-1)
        take = loss < best
        best[take] = loss[take]
        ba[take] = trial[take]
        bc[take] = cs[take].byte()
        bs[take] = st.reshape_as(bs)[take]
        # Refit row anchors and subgroup gains for the same exact decoded points.
        # The native routine only replaces a row when actual FP16 SSE improves.
        _status(
            lib.fit_refine(
                *[v.data_ptr() for v in (w, h, y, st, ba, bc, best, bs, relative)],
                rows,
                width,
                gs,
                stream,
            )
        )

    for f in range(finalists):
        full_fit((a.float() * factor_table[order[:, f]]).half().clamp_min(2**-24), codes)
    for _ in range(refine):
        full_fit(ba.clone(), bc.clone())
    return bs, ba, bc
