"""Independent CUDA Viterbi ring encoder; no external encoder dependency."""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(None)
def _library():
    from torch.utils.cpp_extension import load

    hip = bool(torch.version.hip)
    source = (Path(__file__).parents[1] / "hip/tq_encoder.hip" if hip
              else Path(__file__).with_suffix(".cu"))
    path = load(name="mfq_tq_encoder_hip" if hip else "mfq_tq_encoder",
                sources=[str(source)],
                extra_cuda_cflags=(["-O3", "-ffp-contract=off"] if hip
                                  else ["-O3", "--fmad=false"]), is_python_module=False)
    lib = ct.CDLL(path)
    lib.mfq_tq_encoder_wave.argtypes = [ct.c_int, ct.c_int]
    lib.mfq_tq_encoder_wave.restype = ct.c_int
    lib.mfq_tq_encode.argtypes = [ct.c_void_p] * 8 + [ct.c_int] * 3 + [ct.c_void_p]
    lib.mfq_tq_encode.restype = ct.c_int
    return lib


def encode(x: torch.Tensor, metric: torch.Tensor, table: torch.Tensor, bits: int):
    """Return decoded values, uint16 states in int16 storage, and boundaries.

    Inputs: contiguous FP32 [rings,256] x/metric, FP16 [65536] table,
    all on the same CUDA device. Scratch is reused across occupancy-sized waves.
    """
    if bits not in range(1, 9) or x.ndim != 2 or x.shape[1] != 256 or x.shape[0] == 0:
        raise ValueError("Expected [rings,256] input and 1..8 bits")
    if (not x.is_cuda or x.dtype != torch.float32 or metric.dtype != torch.float32
            or metric.shape != x.shape or table.dtype != torch.float16
            or table.shape != (65536,) or metric.device != x.device or table.device != x.device
            or not all(t.is_contiguous() for t in (x, metric, table))):
        raise ValueError("Encoder tensors must be contiguous CUDA tensors with matching devices")
    with torch.cuda.device(x.device):
        lib = _library()
        wave = lib.mfq_tq_encoder_wave(x.shape[0], bits)
        if wave <= 0:
            raise RuntimeError(f"Trellis encoder occupancy failed: {wave}")
        edges = 1 << (16 - bits)
        history = torch.empty((wave, 256, edges), dtype=torch.uint8, device=x.device)
        global_cost = bits <= 2 if torch.version.hip else bits == 1
        cost = torch.empty((wave, edges if global_cost else 1), device=x.device)
        values = torch.empty_like(x)
        states = torch.empty_like(x, dtype=torch.int16)
        boundaries = torch.empty(x.shape[0], dtype=torch.int32, device=x.device)
        code = lib.mfq_tq_encode(
            *(t.data_ptr() for t in (x, metric, table, values, states, history, cost, boundaries)),
            x.shape[0], bits, wave, torch.cuda.current_stream(x.device).cuda_stream)
        if code:
            raise RuntimeError(f"Trellis encoder returned CUDA error {code}")
    return values, states, boundaries
