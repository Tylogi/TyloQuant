"""TPQ DSpark block-parallel speculative decoding head using the three mtp.* layers built into DeepSeek-V4-Flash-DSpark.

Structure, verified against tensor names and shapes in source shards model-00046/47/48-of-00048:
  mtp.0/1/2 are three DSpark layers (compress_ratio=0, no Compressor, otherwise isomorphic to main layers:
  low-rank Q attention, Hyper-Connections, and a 256-expert MoE).
  Stage 0 also has main_norm/main_proj [4096, 3*4096], whose input is concatenated mean-HC hidden states
  [., 3*4096] from main-model layers 40/41/42. Stage 2 also has norm, hc_head_fn/base/scale,
  markov_head.markov_w1/w2 [129280, 256] (a low-rank logits-bias head), and confidence_head.proj
  (unused in v1). Embed/head are shared with the main model; the official converter skips mtp.*emb*
  and mtp.*head.weight.

Draft forward pass, a pure-torch reproduction of forward_spec / DSparkAttention in official inference/model.py:
  main_x = main_norm(main_proj(main_hidden))            # [1,1,D]
  Replicate embeddings for draft input [t1, noise_token x 4] across four HC channels as [1,5,4,D].
  In each DSparkAttention layer, write main_kv = wkv(main_x) at phase start_pos into ring slot start_pos%128.
  Five draft positions at phases start_pos+1..+5 attend to all active ring slots plus their own five positions.
  This is block-parallel: draft positions have no causal mask, noise positions contain no real future information,
  matching the official implementation. The softmax denominator includes attn_sink, and the final 64 output dimensions
  are inverse-rotated as in the main model. MoE uses sqrtsoftplus top-6 plus gate.bias selection, matching main-model
  layers >=3. The final layer applies hc_head -> norm -> shared lm_head to obtain five-position logits, then adds the
  Markov bias in block order (logits[j] += markov_w2 @ markov_w1[prev_token]) and greedily produces five drafts.

KV synchronization, critical to draft quality while correctness is ensured by greedy main-model validation:
  The DSpark ring stores main_kv only for validated accepted positions. After prefill, prefill_kv builds it from
  all main_hidden states, retaining only the final 128 in ring order. After each validation round, update_kv writes
  the accepted prefix; the next draft call idempotently writes the final accepted position internally.
  Rejected drafts never enter the DSpark ring and require no rollback.

Weight source: dspark.safetensors in the TPQ artifact directory, exported from the original checkpoint by
`python -m TPQ dspark-export` while preserving tensor names and dtypes. The artifact is self-contained and does not
depend on the original model directory. FP8 dequantized values are exactly representable in bf16, so large matrices
reside in bf16. Routed experts use packed FP4 e2m1 plus ue8m0 scaling. FP4Weight is structurally analogous to
Int4Weight: a 256-byte LUT retrieves both nibbles and matmul dequantizes online in blocks. Packed weights reside in
an in-memory LRU (1.5 GB by default, configurable with TPQ_DSPARK_GB), with misses read from the artifact file.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F

from .presets import load_manifest

from .dsv4 import SafeFile, dequant_fp8, rmsnorm, rope_apply, hc_pre, hc_post, hc_head, \
    gate_route, expert_mlp

from .kernels import VQWeight

# Complete 16-value e2m1 table (bit 3 is the sign bit; matches TPQ/fp4io.py)
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

_FP4_GROUP = 32  # ue8m0 scaling granularity (one exponent byte per 32 elements in each row)


def _make_fp4_lut() -> torch.Tensor:
    """Map 256 byte values to (low-nibble value, high-nibble value), with the low nibble/even column first."""
    tab = torch.tensor(_E2M1, dtype=torch.float32)
    t = torch.arange(256)
    return torch.stack([tab[t & 15], tab[t >> 4]], 1)


_FP4_LUT = _make_fp4_lut()
_LUTS: dict = {"cpu": _FP4_LUT}


def _lut_on(device) -> torch.Tensor:
    key = str(device)
    lut = _LUTS.get(key)
    if lut is None:
        lut = _FP4_LUT.to(device)
        _LUTS[key] = lut
    return lut


class FP4Weight:
    """Packed FP4 e2m1 weights: I8 [R, C//2] with the low nibble first plus ue8m0 [R, C//32].
    Matmul dequantizes row blocks online to f32 before torch.mm, using a residency scheme analogous to Int4Weight."""

    __slots__ = ("q", "s", "cols")

    def __init__(self, q: torch.Tensor, s: torch.Tensor, cols: int):
        self.q = q          # u8 [R, C//2]
        self.s = s          # u8 [R, C//32] (ue8m0 exponent bytes)
        self.cols = cols

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.q.shape[0], self.cols])

    @property
    def nbytes(self) -> int:
        return self.q.numel() + self.s.numel()

    def dequant_rows(self, r0: int, r1: int, device) -> torch.Tensor:
        q = self.q[r0:r1].to(device)
        s = self.s[r0:r1].to(device)
        w = _lut_on(device)[q.long()].view(r1 - r0, self.cols)
        sp = torch.pow(2.0, s.float() - 127.0)
        w.view(r1 - r0, self.cols // _FP4_GROUP, _FP4_GROUP).mul_(sp.unsqueeze(-1))
        return w

    def matmul_T(self, x: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        """Compute y = x @ W.T for f32 x [T, C] -> f32 [T, R], dequantizing adaptive row blocks of at most 64 MB."""
        R = self.q.shape[0]
        if chunk is None:
            chunk = max(512, min(R, (64 * 2**20) // max(self.cols * 4, 1)))
        out = torch.empty(x.shape[0], R, dtype=torch.float32, device=x.device)
        for r0 in range(0, R, chunk):
            r1 = min(r0 + chunk, R)
            out[:, r0:r1] = x @ self.dequant_rows(r0, r1, x.device).t()
        return out


class DSparkStore:
    """Reader for dspark.safetensors in the artifact directory. The self-contained artifact follows the
    dspark_file reference in the TPQ manifest and does not depend on the original model directory. Tensor names
    match the original checkpoint (mtp.*), and the exporter preserves companion FP8 .scale tensors verbatim.
    The artifact is generated by `python -m TPQ dspark-export`."""

    def __init__(self, model_dir: str):
        _root, man = load_manifest(model_dir)
        fn = man.get("dspark_file")
        if not fn:
            raise FileNotFoundError(
                f"{model_dir} 的 TPQ 清单缺少 dspark_file"
            )
        self.man = man
        self.sf = SafeFile(os.path.join(model_dir, fn))
        self.keys = set(self.sf.keys())

    def hyper(self) -> dict:
        """Return DSpark hyperparameters (block_size/noise_id/targets), using official defaults when absent from the manifest."""
        d = self.man.get("dspark", {})
        return {"block_size": int(d.get("block_size", 5)),
                "noise_id": int(d.get("noise_id", 128799)),
                "targets": tuple(d.get("targets", (40, 41, 42)))}

    def has_scale(self, name: str) -> bool:
        # Companion-scale naming: X.weight <-> X.scale (the official converter only renames it; it does not take the reciprocal)
        sname = name[:-len("weight")] + "scale" if name.endswith("weight") else name + ".scale"
        return sname in self.keys

    def get_raw(self, name: str) -> torch.Tensor:
        return self.sf.get_tensor(name)

    def get_f32(self, name: str) -> torch.Tensor:
        """Use block-level FP8 dequantization when a companion .scale exists; otherwise convert the stored dtype to f32."""
        sname = name[:-len("weight")] + "scale" if name.endswith("weight") else name + ".scale"
        if sname in self.keys:
            return dequant_fp8(self.get_raw(name), self.get_raw(sname))
        return self.get_raw(name).float()

    # ---- VQ artifact (dspark-vq.safetensors) ----
    def is_vq(self) -> bool:
        """Return whether the artifact is VQ-quantized and contains stage codebook keys."""
        return "s0.cb.gu" in self.keys

    def cb(self, stage: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stage-shared codebooks (cb_gu, cb_dn) as f32 [K, dim]."""
        return (self.get_raw(f"s{stage}.cb.gu").float(),
                self.get_raw(f"s{stage}.cb.dn").float())


def _linb(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Linear layer for large bf16 matrices. e4m3*2^k is exact in bf16; activations enter GEMM as bf16
    to match the official implementation, and outputs return to f32 to preserve later normalization/softmax accuracy."""
    return (x.bfloat16() @ w.t()).float()


class DSparkHead:
    """Three-layer DSpark draft head for DeepSeek-V4-Flash-DSpark. The provider reuses DSV4TPQModel:
    embed/head/RoPE/device come from the main model, while mtp.* weights load from the original shards."""

    N_STAGES = 3
    WIN = 128

    def __init__(self, model, src: str | None = None):
        self.m = model
        self.device = model.device
        self.cfg = model._cfg_obj()
        self.store = DSparkStore(model.store.root)   # Load only from the artifact directory (dspark_file)
        hp = self.store.hyper()
        self.block_size = hp["block_size"]
        self.noise_id = hp["noise_id"]
        self.targets = hp["targets"]
        self.rope = model.rope_base          # ratio=0 -> theta=10000, without YaRN
        self._stages: dict[int, dict] = {}
        self._experts: OrderedDict[tuple[int, int], tuple[FP4Weight, FP4Weight]] = OrderedDict()
        self._ebytes = 0
        self._ebudget = int(float(os.environ.get("TPQ_DSPARK_GB", "1.5")) * 2**30)
        self.ehits = 0
        self.emiss = 0
        self.rings: list[torch.Tensor] | None = None

    # ---- Weights ----
    def stage_w(self, s: int) -> dict:
        w = self._stages.get(s)
        if w is not None:
            return w
        p = f"mtp.{s}"
        st = self.store

        def bf16(name: str) -> torch.Tensor:
            # FP8 (e4m3 * 2^k) dequantized values are exactly representable in bf16; preserve BF16 and store F32 separately as f32
            t = st.get_f32(name)
            return t.bfloat16().to(self.device)

        def f32(name: str) -> torch.Tensor:
            return st.get_f32(name).to(self.device)

        w = {
            "wq_a": bf16(f"{p}.attn.wq_a.weight"),
            "q_norm": f32(f"{p}.attn.q_norm.weight"),
            "wq_b": bf16(f"{p}.attn.wq_b.weight"),
            "wkv": bf16(f"{p}.attn.wkv.weight"),
            "kv_norm": f32(f"{p}.attn.kv_norm.weight"),
            "attn_sink": f32(f"{p}.attn.attn_sink"),
            "wo_a": bf16(f"{p}.attn.wo_a.weight"),
            "wo_b": bf16(f"{p}.attn.wo_b.weight"),
            "attn_norm": f32(f"{p}.attn_norm.weight"),
            "ffn_norm": f32(f"{p}.ffn_norm.weight"),
            "gate": f32(f"{p}.ffn.gate.weight"),
            "gate_bias": f32(f"{p}.ffn.gate.bias"),
            "sh_w1": f32(f"{p}.ffn.shared_experts.w1.weight"),
            "sh_w3": f32(f"{p}.ffn.shared_experts.w3.weight"),
            "sh_w2": f32(f"{p}.ffn.shared_experts.w2.weight"),
            "hc_attn_fn": f32(f"{p}.hc_attn_fn"),
            "hc_attn_base": f32(f"{p}.hc_attn_base"),
            "hc_attn_scale": f32(f"{p}.hc_attn_scale"),
            "hc_ffn_fn": f32(f"{p}.hc_ffn_fn"),
            "hc_ffn_base": f32(f"{p}.hc_ffn_base"),
            "hc_ffn_scale": f32(f"{p}.hc_ffn_scale"),
        }
        if s == 0:
            w["main_proj"] = bf16(f"{p}.main_proj.weight")
            w["main_norm"] = f32(f"{p}.main_norm.weight")
        if s == self.N_STAGES - 1:
            w["norm"] = f32(f"{p}.norm.weight")
            w["hc_head_fn"] = f32(f"{p}.hc_head_fn")
            w["hc_head_base"] = f32(f"{p}.hc_head_base")
            w["hc_head_scale"] = f32(f"{p}.hc_head_scale")
            w["markov_w1"] = st.get_raw(f"{p}.markov_head.markov_w1.weight").to(self.device)
            w["markov_w2"] = st.get_raw(f"{p}.markov_head.markov_w2.weight").to(self.device)
        self._stages[s] = w
        return w

    def _cbs(self, stage: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a lazily cached device copy of a stage-shared codebook."""
        cache = getattr(self, "_cb_cache", None)
        if cache is None:
            cache = self._cb_cache = {}
        cb = cache.get(stage)
        if cb is None:
            g, d = self.store.cb(stage)
            cb = (g.to(self.device), d.to(self.device))
            cache[stage] = cb
        return cb

    def _expert(self, stage: int, eid: int) -> tuple:
        """Return a routed expert (gu, dn). VQ artifacts use VQWeight with u8 indices plus stage-shared
        codebooks, LUT matmul without matrix reconstruction, and a GPU-resident LRU. FP4 artifacts use
        FP4Weight with packed weights in an in-memory LRU."""
        key = (stage, eid)
        ent = self._experts.get(key)
        if ent is not None:
            self.ehits += 1
            self._experts.move_to_end(key)
            return ent
        self.emiss += 1
        st = self.store
        if st.is_vq():
            cb_gu, cb_dn = self._cbs(stage)
            gu = VQWeight(st.get_raw(f"s{stage}.e{eid}.gu").to(self.device),
                          cb_gu, self.cfg.hidden)
            dn = VQWeight(st.get_raw(f"s{stage}.e{eid}.dn").to(self.device),
                          cb_dn, self.cfg.moe_inter)
        else:
            p = f"mtp.{stage}.ffn.experts.{eid}"
            q1 = st.get_raw(f"{p}.w1.weight").view(torch.uint8)
            s1 = st.get_raw(f"{p}.w1.scale").view(torch.uint8)
            q3 = st.get_raw(f"{p}.w3.weight").view(torch.uint8)
            s3 = st.get_raw(f"{p}.w3.scale").view(torch.uint8)
            q2 = st.get_raw(f"{p}.w2.weight").view(torch.uint8)
            s2 = st.get_raw(f"{p}.w2.scale").view(torch.uint8)
            mi = self.cfg.moe_inter
            gu = FP4Weight(torch.cat([q1, q3], 0), torch.cat([s1, s3], 0), self.cfg.hidden)
            dn = FP4Weight(q2, s2, mi)
        ent = (gu, dn)
        nb = gu.nbytes + dn.nbytes
        while self._ebytes + nb > self._ebudget and self._experts:
            _, (g, d) = self._experts.popitem(last=False)
            self._ebytes -= g.nbytes + d.nbytes
        self._experts[key] = ent
        self._ebytes += nb
        return ent

    # ---- KV ring (stores main_kv only for accepted positions) ----
    def reset(self) -> None:
        self.rings = None

    def _alloc(self) -> None:
        hd = self.cfg.head_dim
        self.rings = [torch.zeros(1, self.WIN, hd, device=self.device)
                      for _ in range(self.N_STAGES)]

    def _main_x(self, mh: torch.Tensor) -> torch.Tensor:
        """Map main_hidden [., 3D] to main_norm(main_proj(mh)) [., D], shared by all three layers."""
        w0 = self.stage_w(0)
        return rmsnorm(_linb(mh, w0["main_proj"]), w0["main_norm"], self.cfg.rms_eps)

    def _kv_write(self, main_x: torch.Tensor, pos0: int) -> None:
        """Write main_kv for positions pos0..pos0+T-1 into each layer's ring, using slot = pos % 128."""
        cfg = self.cfg
        hd, rd = cfg.head_dim, cfg.qk_rope_head_dim
        T = main_x.shape[0]
        cos = self.rope.cos[pos0:pos0 + T]
        sin = self.rope.sin[pos0:pos0 + T]
        slots = (torch.arange(pos0, pos0 + T, device=self.device) % self.WIN)
        for s in range(self.N_STAGES):
            w = self.stage_w(s)
            kv = rmsnorm(_linb(main_x, w["wkv"]), w["kv_norm"], cfg.rms_eps)  # [T, hd]
            kv[:, hd - rd:] = rope_apply(kv[:, hd - rd:], cos.view(T, -1), sin.view(T, -1))
            self.rings[s][:, slots] = kv

    @torch.no_grad()
    def prefill_kv(self, mh: torch.Tensor) -> None:
        """Build the ring from main_hidden [T, 3D] at all prompt positions, retaining only the final 128 in ring order."""
        if self.rings is None:
            self._alloc()
        T = mh.shape[0]
        n = min(T, self.WIN)
        main_x = self._main_x(mh[T - n:])  # Only the final 128 positions remain in the ring, avoiding unnecessary projections
        self._kv_write(main_x, T - n)

    @torch.no_grad()
    def update_kv(self, mh_rows: torch.Tensor, pos0: int) -> None:
        """Write main_kv for a segment of accepted positions, mapping mh_rows [n, 3D] to positions pos0..pos0+n-1."""
        if mh_rows.shape[0] == 0:
            return
        self._kv_write(self._main_x(mh_rows), pos0)

    # ---- Draft forward pass ----
    def _qkv(self, x: torch.Tensor, w: dict, pos0: int, T: int):
        """Low-rank Q plus MQA KV, mathematically matching TPQ.dsv4._qkv and using bf16 GEMM for large matrices."""
        cfg = self.cfg
        H, hd, rd = cfg.n_heads, cfg.head_dim, cfg.qk_rope_head_dim
        qr = rmsnorm(_linb(x, w["wq_a"]), w["q_norm"], cfg.rms_eps)
        q = _linb(qr, w["wq_b"]).view(1, T, H, hd)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + cfg.rms_eps)  # Per-head weightless RMS normalization
        cos = self.rope.cos[pos0:pos0 + T]
        sin = self.rope.sin[pos0:pos0 + T]
        q[..., hd - rd:] = rope_apply(q[..., hd - rd:], cos.view(1, T, 1, -1),
                                      sin.view(1, T, 1, -1))
        kv = rmsnorm(_linb(x, w["wkv"]), w["kv_norm"], cfg.rms_eps)
        kv[..., hd - rd:] = rope_apply(kv[..., hd - rd:], cos.view(1, T, -1),
                                       sin.view(1, T, -1))
        return q, kv

    def _o_proj(self, o: torch.Tensor, w: dict) -> torch.Tensor:
        """Grouped LoRA O using bf16 GEMM, mapping o [1,T,H*hd] to [1,T,D]."""
        cfg = self.cfg
        G = cfg.o_groups
        o = o.reshape(1, -1, G, cfg.n_heads * cfg.head_dim // G).bfloat16()
        wo_a = w["wo_a"].view(G, cfg.o_lora_rank, -1)
        o = torch.einsum("btgd,grd->btgr", o, wo_a)
        return _linb(o.flatten(2), w["wo_b"])

    def _attn(self, x: torch.Tensor, w: dict, ring: torch.Tensor,
              main_x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """DSparkAttention decode: place main_kv in the start_pos ring slot and let five draft positions attend
        to all active ring slots plus their own five positions, with no inter-draft mask and sink in the denominator."""
        cfg = self.cfg
        H, hd, rd = cfg.n_heads, cfg.head_dim, cfg.qk_rope_head_dim
        T = self.block_size
        # Write main_kv (phase start_pos) into ring slot start_pos % win (idempotent: the same hidden state yields the same value)
        mkv = rmsnorm(_linb(main_x, w["wkv"]), w["kv_norm"], cfg.rms_eps)  # [1, 1, hd]
        cos1 = self.rope.cos[start_pos:start_pos + 1]
        sin1 = self.rope.sin[start_pos:start_pos + 1]
        mkv[..., hd - rd:] = rope_apply(mkv[..., hd - rd:], cos1.view(1, 1, -1),
                                        sin1.view(1, 1, -1))
        ring[:, start_pos % self.WIN] = mkv[0, 0]
        # Draft q/kv (phases start_pos+1..+T)
        q, dkv = self._qkv(x, w, start_pos + 1, T)
        n = min(self.WIN, start_pos + 1)   # Number of active slots (after prefill, slots equal positions; all slots are active once the ring is full)
        keys = torch.cat([ring[:, :n], dkv], dim=1)                       # [1, n+T, hd]
        scores = torch.einsum("bthd,bsd->bhts", q * (hd ** -0.5), keys)
        m = scores.amax(dim=-1)                                   # Maximum excludes the sink
        e = (scores - m.unsqueeze(-1)).exp()
        denom = e.sum(dim=-1) + (w["attn_sink"].view(1, -1, 1) - m).exp()
        o = torch.einsum("bhts,bsd->bthd", e, keys) / denom.transpose(1, 2).unsqueeze(-1)
        cosT = self.rope.cos[start_pos + 1:start_pos + 1 + T].view(1, T, 1, -1)
        sinT = self.rope.sin[start_pos + 1:start_pos + 1 + T].view(1, T, 1, -1)
        o[..., hd - rd:] = rope_apply(o[..., hd - rd:], cosT, sinT, inverse=True)  # Inverse-rotate the output
        return self._o_proj(o.flatten(2), w)

    def _moe(self, x: torch.Tensor, w: dict, stage: int, ids: torch.Tensor) -> torch.Tensor:
        """sqrtsoftplus top-6 routing plus FP4 experts and a shared expert, mathematically matching main-model _moe.
        Dispatch uses argsort plus searchsorted to avoid implicit synchronization from per-expert nonzero calls."""
        cfg = self.cfg
        B, T, D = x.shape
        xf = x.reshape(B * T, D).float()
        gw = {"gate": w["gate"], "gate_bias": w["gate_bias"]}
        weights, indices = gate_route(xf, gw, cfg, ids.reshape(-1))
        y = torch.zeros_like(xf)
        limit = cfg.swiglu_limit
        mi = cfg.moe_inter
        K = indices.shape[1]
        flat = indices.reshape(-1)
        order = torch.argsort(flat)
        bounds = torch.searchsorted(flat[order],
                                    torch.arange(cfg.n_experts + 1, device=flat.device))
        bl = bounds.tolist()
        rows_all = torch.div(order, K, rounding_mode="floor")
        cols_all = order % K
        for e in range(cfg.n_experts):
            if bl[e + 1] == bl[e]:
                continue
            sl = slice(bl[e], bl[e + 1])
            rows, cols = rows_all[sl], cols_all[sl]
            gu, dn = self._expert(stage, e)
            h = gu.matmul_T(xf[rows])
            g, u = h[:, :mi], h[:, mi:]
            if limit:
                u = u.clamp(-limit, limit)
                g = g.clamp(max=limit)
            y[rows] += dn.matmul_T(F.silu(g) * u) \
                * weights[rows, cols, None]
        y += expert_mlp(xf, w["sh_w1"], w["sh_w3"], w["sh_w2"], limit)
        return y.view(B, T, D)

    def _block(self, h: torch.Tensor, stage: int, draft_ids: torch.Tensor,
               main_x: torch.Tensor, start_pos: int) -> torch.Tensor:
        cfg = self.cfg
        w = self.stage_w(stage)
        residual = h
        y, post, comb = hc_pre(h, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"], cfg)
        y = rmsnorm(y, w["attn_norm"], cfg.rms_eps)
        a = self._attn(y, w, self.rings[stage], main_x, start_pos)
        h = hc_post(a, residual, post, comb)
        residual = h
        y, post, comb = hc_pre(h, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"], cfg)
        y = rmsnorm(y, w["ffn_norm"], cfg.rms_eps)
        f = self._moe(y, w, stage, draft_ids)
        return hc_post(f, residual, post, comb)

    @torch.no_grad()
    def draft(self, t1: int, mh_last: torch.Tensor, start_pos: int) -> list[int]:
        """Produce block_size draft tokens in one DSpark forward pass.

        t1: Main model's greedy token at the current position, used as input at position start_pos+1.
        mh_last: Main-model main_hidden [3D] or [1, 3D] at position start_pos.
        start_pos: Final accepted position (>0; prompt_len-1 after prefill).
        """
        cfg = self.cfg
        T = self.block_size
        main_x = self._main_x(mh_last.view(1, -1)).unsqueeze(0)        # [1, 1, D]
        ids = torch.full((1, T), self.noise_id, dtype=torch.long, device=self.device)
        ids[0, 0] = t1
        h = self.m._embed(ids).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        for s in range(self.N_STAGES):
            h = self._block(h, s, ids, main_x, start_pos)
        wl = self.stage_w(self.N_STAGES - 1)
        x = hc_head(h, wl["hc_head_fn"], wl["hc_head_scale"], wl["hc_head_base"], cfg)
        x = rmsnorm(x, wl["norm"], cfg.rms_eps)
        logits = self.m.logits_of(x[0])                # [T, V] f32 (shared lm_head)
        mw1, mw2 = wl["markov_w1"], wl["markov_w2"]    # [V, 256] bf16
        prev = torch.tensor([t1], dtype=torch.long, device=self.device)
        outs = []
        for j in range(T):
            emb = mw1[prev]                            # [1, 256]
            logits[j] += (emb @ mw2.t()).float()[0]    # Low-rank Markov bias
            prev = logits[j].argmax().view(1)
            outs.append(prev)
        return [int(t) for t in outs]
