"""GDN CUDA glue (kernel in gated_delta_net.cu, built into the mfq_cuda extension).

Requires nvcc + (on Windows) MSVC cl on PATH (source vcvars64, or a VS x64 Native prompt).
"""

from __future__ import annotations

import torch

from mfq.kernels.cuda._ext import ext


def gated_delta_net(q, k, v, g, beta, state=None):
    """CUDA GDN forward. Returns ``(out [B,H,T,D], new_state [B,H,D,D])``, on cuda.

    Inputs are moved to cuda f32 contiguous; D in {32,64,128}. Python reference recurrence
    in :mod:`mfq.kernels.gated_delta_net`.
    """

    kw = dict(device="cuda", dtype=torch.float32)
    q = q.contiguous().to(**kw)
    k = k.contiguous().to(**kw)
    v = v.contiguous().to(**kw)
    g = g.contiguous().to(**kw)
    beta = beta.contiguous().to(**kw)
    st = None if state is None else state.contiguous().to(**kw)
    out, new_state = ext().gdn_cuda(q, k, v, g, beta, st)
    return out, new_state
