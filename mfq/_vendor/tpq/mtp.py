"""TPQ MTP speculative decoding: forward pass and draft generation for GLM-5.2's built-in MTP layer 78.

MTP forward specification (DeepSeek-V3 style, with tensor shapes verified in the FP8 checkpoint):
    x = eh_proj(cat([hnorm(h_main), enorm(embed(t_next))], -1))   # [., 6144]
    h78 = decoder_layer_78(x)          # Full MLA attention plus MoE (256 experts, all v tier)
    logits = lm_head(shared_head.norm(h78))
Draft chaining feeds the previous h78 and the draft token embedding back into the same module.
The MTP layer has an independent KV cache (its own layer-78 K/V), cleared together with a main-model reset.

Speculative decoding uses greedy acceptance and produces output **identical to pure greedy decoding token by token**, with zero quality risk:
  1. One main-model forward pass obtains true next token t1 and the main hidden state.
  2. MTP chains k drafts d1..dk.
  3. One main-model forward pass validates [t1, d1..dk] by comparing argmax values position by position, accepting the
     longest consecutive matching prefix and using the argmax at the first mismatch as a bonus token.
  4. Each streaming round costs about one main forward pass and produces 1 plus the accepted-count tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .kernels import Int4Weight, rmsnorm


def _lin(x: torch.Tensor, w) -> torch.Tensor:
    if isinstance(w, Int4Weight):
        return w.matmul_T(x)
    return x.float() @ w.t()


class MTPHead:
    """Inference head for GLM-5.2's MTP layer 78."""

    LAYER = 78

    def __init__(self, model):
        self.m = model
        self.store = model.store
        assert self.store.has_mtp(), "模型目录缺 MTP 附件（mtp.safetensors / experts.L78.safetensors）"
        self._w: dict[str, object] = {}
        self.kv: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset(self) -> None:
        self.kv = None

    def w(self, name: str):
        wt = self._w.get(name)
        if wt is None:
            wt = self.store.get_mtp(name)
            dev = self.m.device
            if dev.type != "cpu":
                if isinstance(wt, Int4Weight):
                    wt = Int4Weight(wt.q.to(dev), wt.s.to(dev), wt.cols, wt.gs)
                else:
                    wt = wt.to(dev)
            self._w[name] = wt
        return wt

    # ---- Layer-78 forward pass (mathematically matches attention/MoE in model.py) ----
    def _attention(self, x: torch.Tensor, pos0: int) -> torch.Tensor:
        c = self.m.cfg
        H = c["n_heads"]
        T = x.shape[0]
        q_resid = rmsnorm(_lin(x, self.w("attn.q_a")), self.w("attn.q_a_norm"), 1e-6)
        q = _lin(q_resid, self.w("attn.q_b")).view(T, H, c["qk_head_dim"]).transpose(0, 1)
        q_nope, q_rot = q.split([c["qk_nope_head_dim"], c["qk_rope_head_dim"]], dim=-1)
        kv = _lin(x, self.w("attn.kv_a"))
        k_pass, k_rot = kv.split([c["kv_lora_rank"], c["qk_rope_head_dim"]], dim=-1)
        k_pass = rmsnorm(k_pass, self.w("attn.kv_a_norm"), 1e-6)
        k_pass = _lin(k_pass, self.w("attn.kv_b"))
        k_pass = k_pass.view(T, H, c["qk_nope_head_dim"] + c["v_head_dim"]).transpose(0, 1)
        k_nope, v = k_pass.split([c["qk_nope_head_dim"], c["v_head_dim"]], dim=-1)
        q_rot, k_rot = self.m.rope.apply(q_rot, k_rot.view(1, T, c["qk_rope_head_dim"]), pos0)
        k_rot = k_rot.expand(H, T, c["qk_rope_head_dim"])
        q_f = torch.cat([q_nope, q_rot], dim=-1)
        k_f = torch.cat([k_nope, k_rot], dim=-1)
        if self.kv is not None:
            k_f = torch.cat([self.kv[0].float(), k_f], dim=1)
            v = torch.cat([self.kv[1].float(), v], dim=1)
        self.kv = (k_f.half(), v.half())
        scores = (q_f.float() @ k_f.float().transpose(1, 2)) / math.sqrt(c["qk_head_dim"])
        S = scores.shape[-1]
        if T > 1:
            kpos = torch.arange(S, device=x.device)
            qpos = torch.arange(pos0, pos0 + T, device=x.device)
            scores = scores.masked_fill((kpos[None, :] > qpos[:, None])[None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v.float()).transpose(0, 1).reshape(T, H * c["v_head_dim"])
        return _lin(out, self.w("attn.o"))

    def _moe(self, x: torch.Tensor) -> torch.Tensor:
        c = self.m.cfg
        logits = _lin(x, self.w("router")).float()
        prob = logits.sigmoid()
        choice = prob + self.w("router_bias").float()
        mask = self.m._mask(self.LAYER)
        choice = choice.masked_fill(~mask, float("-inf"))
        idx = choice.topk(c["top_k"], dim=-1).indices
        w = prob.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * c["routed_scaling"]
        I = c["moe_inter"]
        y = torch.zeros_like(x)
        need = [(self.LAYER, e) for e in idx.unique().tolist()]
        experts = self.m.pool.get_many(need)
        for e in idx.unique().tolist():
            toks, slots = (idx == e).nonzero(as_tuple=True)
            gu, dn = experts[(self.LAYER, e)]
            h = gu.matmul_T(x[toks])
            inter = F.silu(h[:, :I]) * h[:, I:]
            y.index_add_(0, toks, dn.matmul_T(inter) * w[toks, slots].unsqueeze(1))
        shared = _lin(F.silu(_lin(x, self.w("shared_gate")))
                      * _lin(x, self.w("shared_up")), self.w("shared_down"))
        return y + shared

    def _layer78(self, x: torch.Tensor, pos0: int) -> torch.Tensor:
        eps = self.m.cfg["rms_eps"]
        h = self._attention(rmsnorm(x, self.w("input_norm"), eps), pos0)
        x = x + h
        return x + self._moe(rmsnorm(x, self.w("post_norm"), eps))

    # ---- MTP interface ----
    def _combine(self, h_main: torch.Tensor, tok_ids: list[int]) -> torch.Tensor:
        """Compute eh_proj(cat[enorm(embed(tok)), hnorm(h_main)]) with embedding before hidden state.
        Isolated testing showed that [emb,h] matches the main model's prediction distribution, while [h,emb] is invalid."""
        emb = self.m.embed(tok_ids)
        hn = rmsnorm(h_main, self.w("hnorm"), 1e-5)
        en = rmsnorm(emb, self.w("enorm"), 1e-5)
        return _lin(torch.cat([en, hn], dim=-1), self.w("eh_proj"))

    def prefill(self, h_main: torch.Tensor, ids: list[int]) -> torch.Tensor:
        """Map main-model hidden states [T, hidden] and a token sequence to final-position h78 [1, hidden].

        MTP input at position j is (h_main[j], embed(ids[j+1])) and predicts ids[j+2].
        """
        T = len(ids)
        x = self._combine(h_main[: T - 1], ids[1:])
        h78 = self._layer78(x, 1)  # MTP inputs occupy RoPE positions 1..T-1 (aligned with tokens)
        return h78[-1:]

    def step(self, h78_prev: torch.Tensor, tok_id: int, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One step: previous h78 plus a draft token -> (new h78, logits[vocab])."""
        x = self._combine(h78_prev, [tok_id])
        h78 = self._layer78(x, pos)
        logits = _lin(rmsnorm(h78, self.w("shared_head_norm"), 1e-5),
                      self.m.w("lm_head.weight")).squeeze(0)
        return h78, logits
