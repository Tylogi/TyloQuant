"""Gated DeltaNet 线性注意力（对应 ggml gated_delta_net.cu / ops.cpp）。

Qwen3.5/3.6 占大多数层的 GDN：带门控的 delta-rule 递归状态。逐 token 递推（per head）：

    S ← decay·S            decay = exp(g)，标量门控或 per-dim diag(exp(g))（KDA）
    δ = (v − Sᵀk) · β      delta-rule
    S ← S + k⊗δ            外积更新
    o = scale·(Sᵀq)         检索，scale = 1/√D

参考实现为 T 步 Python 循环（正确性优先，对应 ops.cpp 的朴素递推）。生产级速度需
chunked-parallel 版（按 T 分块、并行初始化各块状态），对应 ``gated_delta_net.cu`` 的
chunked kernel——留作后续（torch.compile / 自定义 CUDA）。

张量约定（torch 布局 ``[B, H, T, D]``，对应 ggml 的 ``[D, H, T, B]``）：

    q, k, v : [B, H, T, D]   （GQA 须在调用前把 q/k repeat 到与 v 同 head 数）
    g       : [B, H, T]（标量门控）或 [B, H, T, D]（per-dim，KDA）
    beta    : [B, H, T]
    state   : [B, H, D, D]   初始递归状态 s0（None → 零）

返回 ``(out [B,H,T,D], new_state [B,H,D,D])``。
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
    """Gated DeltaNet 前向（参考递推实现）。"""

    B, H, T, D = q.shape
    scale = D ** -0.5
    S = q.new_zeros(B, H, D, D) if state is None else state.clone()
    kda = g.dim() == 4   # per-dim 门控

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
