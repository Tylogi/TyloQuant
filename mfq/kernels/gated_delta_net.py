"""Gated DeltaNet linear attention corresponding to ggml gated_delta_net.cu / ops.cpp.

GDN occupies most Qwen3.5/3.6 layers and uses a gated delta-rule recurrent state. Per-token recurrence for each head:

    S <- decay*S            decay = exp(g), scalar gating or per-dimension diag(exp(g)) (KDA)
    δ = (v − Sᵀk) · β      delta-rule
    S <- S + k outer delta  outer-product update
    o = scale*(S^T q)        retrieval, scale = 1/sqrt(D)

The reference implementation is a T-step Python loop prioritizing correctness and matching the naive recurrence in ops.cpp.
Production speed requires a chunked-parallel version that partitions T and initializes chunk states in parallel, corresponding
to the chunked kernel in ``gated_delta_net.cu``. This remains future work for torch.compile or custom CUDA.

Tensor conventions (torch layout ``[B, H, T, D]``, corresponding to ggml ``[D, H, T, B]``):

    q, k, v : [B, H, T, D]   (for GQA, repeat q/k to v's head count before calling)
    g       : [B, H, T] (scalar gating) or [B, H, T, D] (per-dimension KDA)
    beta    : [B, H, T]
    state   : [B, H, D, D]   initial recurrent state s0 (None -> zero)

Returns ``(out [B,H,T,D], new_state [B,H,D,D])``.
"""

from __future__ import annotations

import torch


def gated_delta_net(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated DeltaNet forward pass using the reference recurrence implementation."""

    B, H, T, D = q.shape
    scale = D ** -0.5
    S = q.new_zeros(B, H, D, D) if state is None else state.clone()
    kda = g.dim() == 4   # Per-dimension gating

    outs = []
    for t in range(T):
        qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]   # [B,H,D]
        bt = beta[:, :, t]                                  # [B,H]

        if kda:
            S = S * torch.exp(g[:, :, t, :]).unsqueeze(-1)            # S[i,j]*=exp(g[i])
        else:
            S = S * torch.exp(g[:, :, t]).view(B, H, 1, 1)            # S*=exp(g)

        stk = (S.transpose(-1, -2) @ kt.unsqueeze(-1)).squeeze(-1)    # [B,H,D] = Sᵀk
        delta = (vt - stk) * bt.unsqueeze(-1)                         # δ = (v−Sᵀk)·β
        S = S + kt.unsqueeze(-1) * delta.unsqueeze(-2)                # S += k⊗δ

        o = (S.transpose(-1, -2) @ qt.unsqueeze(-1)).squeeze(-1) * scale
        outs.append(o)

    return torch.stack(outs, dim=2), S
