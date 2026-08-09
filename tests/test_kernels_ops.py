"""Reference tests for kernel operators, comparing the gated_delta_net recurrence with CUDA kernels and NumPy.

Norm, RoPE, attention, and accumulation have moved to native CUDA (``mfq.kernels.cuda.*``); see
``test_kernels_cuda.py`` for correctness tests against built-in torch implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mfq.kernels.gated_delta_net import gated_delta_net  # noqa: E402


def _gdn_ref(q, k, v, g, beta):
    """NumPy reference recurrence matching the algorithm in ops.cpp."""
    B, H, T, D = q.shape
    S = np.zeros((B, H, D, D))
    kda = g.ndim == 4
    out = np.zeros((B, H, T, D))
    for t in range(T):
        qt, qk, qv, bt = q[:, :, t], k[:, :, t], v[:, :, t], beta[:, :, t]
        if kda:
            S = S * np.exp(g[:, :, t, :])[..., None]
        else:
            S = S * np.exp(g[:, :, t])[:, :, None, None]
        stk = (S.transpose(0, 1, 3, 2) @ qk[..., None])[..., 0]
        delta = (qv - stk) * bt[..., None]
        S = S + qk[..., None] * delta[..., None, :]
        out[:, :, t] = (S.transpose(0, 1, 3, 2) @ qt[..., None])[..., 0] * (D ** -0.5)
    return out, S


def test_gdn_matches_reference():
    torch.manual_seed(2)
    B, H, T, D = 2, 3, 5, 4
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    g = torch.randn(B, H, T)
    beta = torch.sigmoid(torch.randn(B, H, T))
    y, S = gated_delta_net(q, k, v, g, beta)
    yn, Sn = _gdn_ref(*[a.double().numpy() for a in (q, k, v, g, beta)])
    torch.testing.assert_close(y, torch.tensor(yn, dtype=torch.float32), atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(S, torch.tensor(Sn, dtype=torch.float32), atol=1e-4, rtol=1e-3)


def test_gdn_kda_perdim_gate():
    torch.manual_seed(3)
    B, H, T, D = 1, 2, 4, 4
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    g = torch.randn(B, H, T, D)
    beta = torch.sigmoid(torch.randn(B, H, T))
    y, S = gated_delta_net(q, k, v, g, beta)
    assert y.shape == (B, H, T, D) and S.shape == (B, H, D, D)
    assert torch.isfinite(y).all()


def test_gdn_state_carryover():
    """Segmented and single-pass processing should produce identical final states."""
    torch.manual_seed(4)
    B, H, T, D = 1, 2, 6, 4
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    g = torch.randn(B, H, T)
    beta = torch.sigmoid(torch.randn(B, H, T))
    _, S_full = gated_delta_net(q, k, v, g, beta)
    _, S1 = gated_delta_net(q[:, :, :3], k[:, :, :3], v[:, :, :3], g[:, :, :3], beta[:, :, :3])
    _, S2 = gated_delta_net(q[:, :, 3:], k[:, :, 3:], v[:, :, 3:], g[:, :, 3:], beta[:, :, 3:], state=S1)
    torch.testing.assert_close(S2, S_full, atol=1e-4, rtol=1e-3)
