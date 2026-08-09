"""TPQ model forward pass: complete CPU inference implementation of GLM-5.2 with MLA and MoE.

Numerics match TPQ/modelmath.py line by line and were verified against a naive element-wise implementation with
max_diff<1e-8. Only the weight source differs: dense weights use blocked Int4Weight dequantization and experts use
the ExpertPool VQ LUT. Attention is full causal attention, equivalent to DSA top-2048 for short contexts below 2048,
and KV cache is stored in f16. MTP layers and the DSA indexer are absent from TPQ artifacts and are not implemented here.
"""

from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

from .kernels import (
    BlockFP8Weight,
    Int4Weight,
    RopeCache,
    VQWeight,
    merge_attention_scores,
    rmsnorm,
)
from .precision import compute_dtype
from .store import TPQStore, ExpertPool


def _linear(
    x: torch.Tensor,
    w,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense linear with packed INT4 decode and a general prefill fallback."""
    if isinstance(w, (Int4Weight, BlockFP8Weight)):
        return w.matmul_T_decode_fused(x, output=output)
    if w.dtype != torch.float32:
        result = (x.to(w.dtype) @ w.t()).float()
        if output is not None:
            output.copy_(result)
            return output
        return result
    return x.float() @ w.t()


def _attention_linear(
    x: torch.Tensor,
    w,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention-only decode dispatch for the fused packed INT4 GEMV."""
    if isinstance(w, Int4Weight):
        return w.matmul_T_decode_fused(x, output=output)
    return _linear(x, w)


def _swiglu_linear(
    x: torch.Tensor,
    gate,
    up,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse packed INT4 Gate/Up GEMVs and FP32 SwiGLU for decode."""
    if (
        isinstance(gate, Int4Weight)
        and isinstance(up, Int4Weight)
        and gate.cols == up.cols
        and gate.gs == up.gs
        and gate.q.shape == up.q.shape
        and gate.s.shape == up.s.shape
    ):
        from .fusedext import int4_swiglu_fused

        fused = int4_swiglu_fused(
            x,
            gate.q,
            gate.s,
            up.q,
            up.s,
            gate.cols,
            gate.gs,
            output=output,
        )
        if fused is not None:
            return fused
    return F.silu(_linear(x, gate)) * _linear(x, up)


def _glm_route(
    logits: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
    top_k: int,
    routed_scaling: float,
    output_buffers: tuple[
        torch.Tensor,
        torch.Tensor,
    ] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized GLM route weights and Top-K expert IDs."""
    from .ops import route_topk

    fused = route_topk(
        logits,
        bias,
        mask,
        scoring_func="sigmoid",
        top_k=top_k,
        normalize=True,
        scaling=routed_scaling,
        output_buffers=output_buffers,
    )
    if fused is not None:
        return fused
    probability = logits.sigmoid()
    choice = probability + bias
    choice = choice.masked_fill(~mask, float("-inf"))
    indices = choice.topk(top_k, dim=-1).indices
    weights = probability.gather(1, indices)
    weights = weights / (
        weights.sum(-1, keepdim=True) + 1e-20
    ) * routed_scaling
    return weights, indices


class GLMModel:
    """Inference model for TPQ-format GLM-5.2 with CPU and CUDA paths.

    device="cpu": packed dense int4 resides in memory and experts are cached in memory by default.
    device="cuda": dense weights (~9.2 GB packed int4) and KV cache reside in VRAM; experts are cached in VRAM
    under the vram_cache_gb budget, with misses uploaded from disk or the memory page cache.
    """

    def __init__(
        self,
        root: str,
        cache_gb: float = 16.0,
        max_ctx: int = 2048,
        device: str = "cpu",
        vram_cache_gb: float = 4.0,
        tp_size: int = 1,
    ):
        self.device = torch.device(device)
        self.store = TPQStore(root)
        self.cfg = self.store.cfg
        from .ops import ModelOperatorConfig

        self.operator_config = ModelOperatorConfig.from_manifest(
            {
                "model_family": self.store.man.model_family or "glm",
                "config": self.cfg,
            }
        )
        gpu = self.device.type != "cpu"
        # Static pinning of hot experts has been disproven: routing popularity strongly depends on the input domain
        # (only a 14% measured hit rate for coding prompts; the average profile's 66% top-32 coverage is merely a cross-domain mean).
        # An LRU makes better use of session locality.
        pin_gb = float(os.environ.get("TPQ_PIN_GB", "0")) if gpu else 0.0
        self._cache_gb = cache_gb
        self._vram_cache_gb = vram_cache_gb
        self._pin_gb = pin_gb
        self.requested_tp_size = int(tp_size)
        self.effective_tp_size = 1
        self.expert_parallel = None
        if self.requested_tp_size > 1:
            if not gpu:
                raise ValueError("tp_size > 1 requires CUDA")
            from .expert_parallel import GpuResidentExpertParallel

            self.expert_parallel = GpuResidentExpertParallel(
                self.store, self.requested_tp_size, self.device
            )
            self.pool = None
        else:
            self.pool = ExpertPool(
                self.store,
                vram_cache_gb if gpu else cache_gb,
                device=device,
                ram_gb=cache_gb - pin_gb if gpu else 0.0,
                pin_gb=pin_gb,
            )
        # The logical context can be large, but placing the entire RoPE table in every layer graph's shared working set
        # slows short and medium contexts. Start with a fixed 32K address window, double it at boundaries, and recapture
        # all graphs together. This does not change the logical max_ctx admission limit.
        rope_initial = max(
            2048,
            int(os.environ.get("TPQ_ROPE_INITIAL_CTX", "32768")),
        )
        self.rope = RopeCache(
            self.cfg["qk_rope_head_dim"],
            self.cfg["rope_theta"],
            max_len=min(max_ctx + 8, rope_initial + 8),
        )
        if gpu:
            self.rope.cos = self.rope.cos.to(self.device)
            self.rope.sin = self.rope.sin.to(self.device)
        self.max_ctx = max_ctx
        self._wcache: dict[str, object] = {}
        self._lm_head_int4: Int4Weight | None = None
        self._decode_workspaces: dict[
            tuple[int, str], torch.Tensor
        ] = {}
        self._decode_position = (
            torch.empty(
                1,
                dtype=torch.long,
                device=self.device,
            )
            if gpu
            else None
        )
        self._attention_graphs: dict[
            int,
            tuple[
                torch.cuda.CUDAGraph,
                torch.Tensor,
                int,
            ],
        ] = {}
        self._attention_graph_stream = (
            torch.cuda.Stream(device=self.device)
            if gpu
            else None
        )
        self._attention_graph_failed = False
        self._masks: dict[int, torch.Tensor] = {}
        self._prev_ids: dict[int, list[int]] = {}   # Layer -> routed experts for the previous token (used for prefetch)
        # Latent MLA KV (TPQ_LATENT_KV, enabled by default) stores c_kv [S,512] plus k_rot [S,64] in f16
        # (~0.09 MB/token). Attention uses absorbed form (q_nope@Wuk, ctx@Wuv^T) without per-head expansion.
        # The legacy path (=0) stores full per-head K/V in f16 (5.11 MB/token, limiting context on 22 GB cards).
        self.latent_kv = (os.environ.get("TPQ_LATENT_KV", "1") != "0"
                          and self.device.type != "cpu")
        # Compute dtype (precision-policy layer): use half-precision tensor cores on GPU (Turing -> fp16, Ampere+ -> bf16).
        # Store MLA absorption matrices and latent KV in this dtype so attention einsum needs no promotion via .float().
        self.cdt = compute_dtype(self.device) if gpu else torch.float32
        self._wuk: dict[int, torch.Tensor] = {}   # [H, nope, R] in the compute dtype
        self._wuv: dict[int, torch.Tensor] = {}   # [H, v, R] in the compute dtype
        # Per-layer KV cache: latent mode uses c_kv [S,R] f16 and k_rot [S,rd] f16;
        # legacy mode uses k [H, S, qk_head_dim] f16 and v [H, S, v_head_dim] f16.
        self.kv: list[tuple[torch.Tensor, torch.Tensor] | None] = \
            [None] * self.cfg["n_layers"]
        # CUDA latent KV uses growable reusable storage; self.kv continues to hold views of the used range,
        # preserving the truncation interface. Decode writes one row in place instead of allocating two torch.cat
        # results for every layer and token.
        self._latent_buffers: list[
            tuple[torch.Tensor, torch.Tensor] | None
        ] = [None] * self.cfg["n_layers"]
        # FlashInfer directly reuses the separated KV buffers above. When disabled, unavailable, or failed,
        # the original PyTorch MLA path remains unchanged.
        self._flashinfer_mla_runner = None
        self._flashinfer_mla_unavailable = False
        self._flashinfer_mla_state = None
        self._direct_mla_bmm = (
            os.environ.get("TPQ_GLM_DIRECT_BMM", "1") != "0"
        )
        try:
            from .fusedext import (
                glm_latent_kv_decode_prepare_fused,
                glm_mla_bmm_decode_fused,
                glm_moe_residual_add_fused,
                glm_norm_qkv_int4_fused,
                glm_residual_norm_router_fused,
                int4_glm_qb_split_fused,
            )

            self._latent_kv_decode_prepare = (
                glm_latent_kv_decode_prepare_fused
            )
            self._mla_bmm_decode = glm_mla_bmm_decode_fused
            self._q_b_split_decode = int4_glm_qb_split_fused
            self._norm_qkv_decode = glm_norm_qkv_int4_fused
            self._residual_norm_router_decode = (
                glm_residual_norm_router_fused
            )
            self._moe_residual_add_decode = (
                glm_moe_residual_add_fused
            )
        except ImportError:
            self._latent_kv_decode_prepare = None
            self._mla_bmm_decode = None
            self._q_b_split_decode = None
            self._norm_qkv_decode = None
            self._residual_norm_router_decode = None
            self._moe_residual_add_decode = None
        self.pos = 0

    def preload(self) -> None:
        """GPU path: preload all dense weights into VRAM (about 13 GB, including resident f32 lm_head/router)
        and preread pinned hot experts into RAM to eliminate the cold-start tail."""
        if self.device.type == "cpu":
            return
        t0 = time.time()
        names = self.store.dense_names()
        for i, name in enumerate(names):
            self.w(name)
            if (i + 1) % 200 == 0:
                print(f"[tpq] 预载 dense {i + 1}/{len(names)}", flush=True)
        vram = torch.cuda.memory_allocated(self.device) / 2**30
        print(f"[tpq] dense 预载完成（{time.time() - t0:.1f}s，显存 {vram:.1f}GB）",
              flush=True)
        if self.latent_kv:
            self._build_absorbed()
        if (
            self.expert_parallel is not None
            and self.expert_parallel.preload_if_fits()
        ):
            self.pool = self.expert_parallel
            self.effective_tp_size = self.requested_tp_size
            return
        if self.expert_parallel is not None:
            print(
                "[tpq] 全显存专家容量/P2P不满足，回退单卡 RAM+显存缓存",
                flush=True,
            )
            self.expert_parallel = None
            self.pool = ExpertPool(
                self.store,
                self._vram_cache_gb,
                device=self.device,
                ram_gb=self._cache_gb - self._pin_gb,
                pin_gb=self._pin_gb,
            )
        resident_all = self.pool.preload_all()
        if resident_all:
            self.pool.pin_host_resident()
        else:   # When memory is insufficient (already warned), fall back to hot pinning plus LRU
            self.pool.preload_pinned()
        self.pool.build_gpu_arenas()

    def _build_absorbed_layer(self, layer: int) -> None:
        """Decompose one layer's kv_b_proj into Wuk/Wuv, either during full preload or lazily on first demand."""
        c = self.cfg
        H, R = c["n_heads"], c["kv_lora_rank"]
        nope, vd = c["qk_nope_head_dim"], c["v_head_dim"]
        w = self.w(f"model.layers.{layer}.self_attn.kv_b_proj.weight")
        if isinstance(w, (Int4Weight, BlockFP8Weight)):
            w = w.dequant_rows(0, w.shape[0])
        w = w.float().view(H, nope + vd, R)
        self._wuk[layer] = w[:, :nope].to(self.cdt).to(self.device)
        self._wuv[layer] = w[:, nope:].to(self.cdt).to(self.device)

    def _build_absorbed(self) -> None:
        """Predecompose each layer's kv_b_proj into absorbed-form Wuk/Wuv matrices resident in f16 VRAM (~2.3 GB/78 layers).
        kv_b_proj [H*(nope+v), R] maps the first nope rows per head to Wuk and the final v rows to Wuv."""
        c = self.cfg
        t0 = time.time()
        for layer in range(c["n_layers"]):
            self._build_absorbed_layer(layer)
        vram = torch.cuda.memory_allocated(self.device) / 2**30
        print(f"[tpq] MLA 吸收矩阵预分解完成（{time.time() - t0:.1f}s，"
              f"KV 潜变量模式 ≈0.09MB/token，显存 {vram:.1f}GB）", flush=True)

    # ---- Weight access (with caching) ----
    def w(self, name: str):
        wt = self._wcache.get(name)
        if wt is None:
            wt = self.store.get_dense(name)
            if self.device.type != "cpu":
                # GPU path: place all dense weights in VRAM (packed int4 goes directly to the device; lm_head/router dequantize to f32)
                if isinstance(wt, Int4Weight):
                    if name == "lm_head.weight":
                        packed_lm = Int4Weight(
                            wt.q.to(self.device),
                            wt.s.to(self.device),
                            wt.cols,
                            wt.gs,
                            half=False,
                        )
                        self._lm_head_int4 = packed_lm
                        wt = (
                            wt.dequant_rows(
                                0,
                                wt.shape[0],
                            ).to(self.device)
                            if os.environ.get(
                                "TPQ_LM_HEAD_KEEP_F32",
                                "0",
                            ) != "0"
                            else packed_lm
                        )
                    elif self.f32_resident(name):
                        wt = wt.dequant_rows(0, wt.shape[0]).to(self.device)
                    else:
                        # int4 fp16 computation is disabled by default (enable with TPQ_INT4_HALF=1; see the dsv4model comment)
                        wt = Int4Weight(
                            wt.q.to(self.device),
                            wt.s.to(self.device),
                            wt.cols,
                            wt.gs,
                            half=os.environ.get(
                                "TPQ_INT4_HALF",
                                "0",
                            ) == "1",
                        )
                elif isinstance(wt, BlockFP8Weight):
                    wt = BlockFP8Weight(
                        wt.q.to(self.device),
                        wt.s.to(self.device),
                        wt.cols,
                        wt.block,
                    )
                else:
                    wt = wt.to(self.device)
            # Keep frequently used large matrices resident in f32: lm_head (951M, multiplied in full for every token)
            # and each layer's router (118M gate.weight). Other dense weights remain packed int4 with blocked dequantization to save memory.
            elif isinstance(wt, Int4Weight) and self.f32_resident(name):
                wt = wt.dequant_rows(0, wt.shape[0])
            self._wcache[name] = wt
        return wt

    @staticmethod
    def f32_resident(name: str) -> bool:
        return name == "lm_head.weight" or name.endswith(".mlp.gate.weight")

    def reset_kv(self) -> None:
        self.kv = [None] * self.cfg["n_layers"]
        self._flashinfer_mla_state = None
        self.pos = 0

    def truncate_kv(self, keep: int) -> None:
        """Truncate KV to the first ``keep`` positions for rollback after MTP speculative validation.
        Slice legacy k/v [H, S, d] on dimension 1 and latent ckv/krot [S, d] on dimension 0."""
        if self.latent_kv:
            self.kv = [(k_[:keep], v_[:keep]) if k_ is not None else None
                       for k_, v_ in self.kv]
        else:
            self.kv = [(k_[:, :keep], v_[:, :keep]) if k_ is not None else None
                       for k_, v_ in self.kv]
        self._flashinfer_mla_state = None
        self.pos = keep

    def _latent_buffer(
        self,
        layer: int,
        required: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a reusable latent-KV buffer capable of holding at least ``required`` rows."""
        current = self._latent_buffers[layer]
        if current is not None and current[0].shape[0] >= required:
            return current
        if current is not None and self._attention_graphs:
            # Each layer graph captures only its own KV address. Growing one layer must not invalidate the other 77 layers,
            # or long-context boundaries would degrade into per-token recapture.
            self._attention_graphs.pop(layer, None)
        initial = self._latent_initial_capacity(layer)
        old_capacity = 0 if current is None else current[0].shape[0]
        capacity = min(
            self.max_ctx,
            max(required, initial, old_capacity * 2),
        )
        # FlashInfer MLA uses page=64; the extra tail is storage capacity only and does not change the logical max_ctx limit.
        capacity = ((capacity + 63) // 64) * 64
        ckv = torch.empty(
            capacity,
            self.cfg["kv_lora_rank"],
            dtype=self.cdt,
            device=self.device,
        )
        krot = torch.empty(
            capacity,
            self.cfg["qk_rope_head_dim"],
            dtype=self.cdt,
            device=self.device,
        )
        if current is not None and self.pos:
            used = min(self.pos, current[0].shape[0])
            ckv[:used].copy_(current[0][:used])
            krot[:used].copy_(current[1][:used])
        result = (ckv, krot)
        self._latent_buffers[layer] = result
        return result

    def _ensure_latent_capacity(self, required: int) -> None:
        """Safely grow fixed-address latent KV buffers before decode."""

        if self.device.type != "cuda" or not self.latent_kv:
            return
        growing = [
            layer
            for layer, current in enumerate(self._latent_buffers)
            if current is not None and current[0].shape[0] < required
        ]
        if not growing:
            return
        captured_growth = any(
            layer in self._attention_graphs
            for layer in growing
        )
        if captured_growth:
            # Graph replay is asynchronous.  Finish all users of the old
            # addresses before replacing any captured KV buffer.
            torch.cuda.synchronize(self.device)
            self._attention_graphs.clear()
        for layer in growing:
            ckv, krot = self._latent_buffer(layer, required)
            used = min(self.pos, required)
            self.kv[layer] = (ckv[:used], krot[:used])

    def _latent_initial_capacity(self, layer: int) -> int:
        """Choose a stable KV address window for graph-backed decode."""
        configured = os.environ.get("TPQ_LATENT_KV_INITIAL")
        if configured is not None:
            return max(1, min(self.max_ctx, int(configured)))
        graph_resident = (
            self.device.type == "cuda"
            and layer >= 4
            and os.environ.get("TPQ_ATTENTION_GRAPH", "1") != "0"
            and self.expert_parallel is not None
            and getattr(self.pool, "full_resident", False)
        )
        if not graph_resident:
            return min(self.max_ctx, 2048)
        graph_window = max(
            2048,
            int(
                os.environ.get(
                    "TPQ_LATENT_KV_GRAPH_INITIAL",
                    "32768",
                )
            ),
        )
        return min(self.max_ctx, graph_window)

    def _prepare_flashinfer_mla_decode(self, end: int):
        """Plan FlashInfer MLA once per token and reuse it across all 78 layers."""
        if (
            not self.latent_kv
            or self.cdt != torch.bfloat16
            or os.environ.get("TPQ_FLASHINFER_MLA", "1") == "0"
            or self._flashinfer_mla_unavailable
        ):
            return None
        from .flashinfer_mla import last_error
        from .ops import attention_step

        if self._flashinfer_mla_runner is None:
            try:
                self._flashinfer_mla_runner = attention_step(
                    "paged_latent_create",
                    self.device.type,
                    device=self.device,
                    max_ctx=self.max_ctx,
                    heads=self.cfg["n_heads"],
                    ckv_dim=self.cfg["kv_lora_rank"],
                    kpe_dim=self.cfg["qk_rope_head_dim"],
                    dtype=self.cdt,
                    qk_head_dim=self.cfg["qk_head_dim"],
                )
            except (ImportError, LookupError, RuntimeError):
                self._flashinfer_mla_runner = None
            if self._flashinfer_mla_runner is None:
                self._flashinfer_mla_unavailable = True
                print(
                    "[tpq] FlashInfer MLA 不可用，回退原 PyTorch MLA："
                    f"{last_error()}",
                    flush=True,
                )
                return None
            print(
                "[tpq] FlashInfer MLA decode 已启用（复用分离 latent KV）",
                flush=True,
            )
        runner = self._flashinfer_mla_runner
        if runner is None:
            return None
        try:
            prepared = attention_step(
                "paged_latent_prepare",
                self.device.type,
                runner=runner,
                length=end,
            )
        except (LookupError, RuntimeError):
            prepared = False
        if not prepared:
            self._flashinfer_mla_unavailable = True
            print(
                "[tpq] FlashInfer MLA 运行失败，回退原 PyTorch MLA："
                f"{last_error()}",
                flush=True,
            )
            return None
        return runner

    def _ensure_rope_capacity(self, required: int) -> None:
        if required <= self.rope.cos.shape[0]:
            return
        if self._attention_graphs and self.device.type == "cuda":
            # Captured kernels may still read the old cos/sin addresses.
            torch.cuda.synchronize(self.device)
        if self.rope.ensure_length(min(self.max_ctx + 8, required + 8)):
            # The attention graph captures cos/sin addresses directly. After growth, recapture once at the boundary;
            # the old addresses cannot be replayed.
            self._attention_graphs.clear()

    # ---- Primitives ----
    def _decode_workspace(
        self,
        layer: int,
        name: str,
        rows: int,
    ) -> torch.Tensor | None:
        """Return a stable FP32 decode output buffer when reuse is enabled."""
        return self._decode_tensor_workspace(
            layer,
            name,
            (1, rows),
            torch.float32,
        )

    def _decode_tensor_workspace(
        self,
        layer: int,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Return a stable decode tensor with an arbitrary shape and dtype."""
        if (
            self.device.type == "cpu"
            or os.environ.get(
                "TPQ_DECODE_WORKSPACES",
                "1",
            ) == "0"
        ):
            return None
        key = (layer, name)
        output = self._decode_workspaces.get(key)
        if (
            output is None
            or output.shape != shape
            or output.dtype != dtype
        ):
            output = torch.empty(
                shape,
                dtype=dtype,
                device=self.device,
            )
            self._decode_workspaces[key] = output
        return output

    def _shared_expert_eager(
        self,
        x: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        p = f"model.layers.{layer}.mlp.shared_experts"
        gate = self.w(f"{p}.gate_proj.weight")
        up = self.w(f"{p}.up_proj.weight")
        down = self.w(f"{p}.down_proj.weight")
        reuse = (
            x.shape[0] == 1
            and isinstance(gate, Int4Weight)
            and isinstance(up, Int4Weight)
            and isinstance(down, Int4Weight)
        )
        return _linear(
            _swiglu_linear(
                x,
                gate,
                up,
                output=(
                    self._decode_workspace(
                        layer,
                        "shared_intermediate",
                        int(gate.shape[0]),
                    )
                    if reuse
                    else None
                ),
            ),
            down,
            output=(
                self._decode_workspace(
                    layer,
                    "shared_output",
                    int(down.shape[0]),
                )
                if reuse
                else None
            ),
        )

    def embed(
        self,
        ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        emb = self.w("model.embed_tokens.weight")
        if isinstance(emb, Int4Weight):
            if (
                isinstance(ids, torch.Tensor)
                and ids.is_cuda
                and ids.dtype == torch.long
                and ids.numel() == 1
                and emb.q.is_cuda
            ):
                from .fusedext import int4_embedding_device_fused

                fused = int4_embedding_device_fused(
                    emb.q,
                    emb.s,
                    ids.reshape(1),
                    emb.cols,
                    emb.gs,
                    output=self._decode_workspace(
                        -1,
                        "embedding",
                        emb.cols,
                    ),
                )
                if fused is not None:
                    return fused
                ids = [int(ids.item())]
            if len(ids) == 1 and emb.q.is_cuda:
                from .fusedext import int4_embedding_fused

                fused = int4_embedding_fused(
                    emb.q,
                    emb.s,
                    ids[0],
                    emb.cols,
                    emb.gs,
                    output=self._decode_workspace(
                        -1,
                        "embedding",
                        emb.cols,
                    ),
                )
                if fused is not None:
                    return fused
            return torch.stack([emb.row(i) for i in ids])
        return emb[ids].float()

    def _attention_output(
        self,
        x: torch.Tensor,
        layer: int,
        weight,
    ) -> torch.Tensor:
        output = (
            self._decode_workspace(
                layer,
                "o_proj",
                int(weight.shape[0]),
            )
            if x.shape[0] == 1 and isinstance(weight, Int4Weight)
            else None
        )
        return _attention_linear(x, weight, output=output)

    def _attention(
        self,
        x: torch.Tensor,
        layer: int,
        pos0: int,
        input_norm_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.latent_kv:
            return self._attention_latent(
                x,
                layer,
                pos0,
                input_norm_weight,
            )
        if input_norm_weight is not None:
            x = rmsnorm(x, input_norm_weight, self.cfg["rms_eps"])
        return self._attention_full(x, layer, pos0)

    def _attention_latent(
        self,
        x: torch.Tensor,
        layer: int,
        pos0: int,
        input_norm_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        graph_enabled = (
            os.environ.get("TPQ_ATTENTION_GRAPH", "1") != "0"
            and not self._attention_graph_failed
            and self.expert_parallel is not None
            and getattr(self.pool, "full_resident", False)
            and x.shape[0] == 1
            and layer >= 4
            and self._flashinfer_mla_state is not None
            and os.environ.get("TPQ_GLM_QB_SPLIT", "1") != "0"
            and os.environ.get("TPQ_DECODE_WORKSPACES", "1") != "0"
        )
        if not graph_enabled:
            return self._attention_latent_eager(
                x,
                layer,
                pos0,
                input_norm_weight,
            )
        cached = self._attention_graphs.get(layer)
        if cached is not None and cached[2] == x.data_ptr():
            cached[0].replay()
            # CUDA Graph replays tensor writes but not this Python metadata
            # assignment.  After reset(), rebuild the logical KV views so a
            # short sequential prompt can be truncated or extended safely
            # without requiring one batch-prefill pass first.
            end = pos0 + x.shape[0]
            ckv_buffer, krot_buffer = self._latent_buffer(
                layer,
                end,
            )
            self.kv[layer] = (
                ckv_buffer[:end],
                krot_buffer[:end],
            )
            return cached[1]

        eager_output = self._attention_latent_eager(
            x,
            layer,
            pos0,
            input_norm_weight,
        )
        stream = self._attention_graph_stream
        if stream is None:
            return eager_output
        try:
            current = torch.cuda.current_stream(self.device)
            stream.wait_stream(current)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    graph_output = self._attention_latent_eager(
                        x,
                        layer,
                        pos0,
                        input_norm_weight,
                    )
            current.wait_stream(stream)
            self._attention_graphs[layer] = (
                graph,
                graph_output,
                x.data_ptr(),
            )
            return graph_output
        except Exception as error:
            self._attention_graph_failed = True
            self._attention_graphs.clear()
            print(
                "[tpq] Attention CUDA Graph 捕获失败，"
                f"回退逐算子路径：{error}",
                flush=True,
            )
            return eager_output

    def _attention_latent_eager(
        self,
        x: torch.Tensor,
        layer: int,
        pos0: int,
        input_norm_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Absorbed-form MLA attention stores only latent KV (c_kv [S,R] plus k_rot [S,rd] in compute dtype).
        Scores = (q_nope@Wuk)@c_kv^T + q_rot@k_rot^T, and output = (attn@c_kv)@Wuv^T.
        This is mathematically equivalent to _attention_full, with different expansion points slightly shifting
        half-precision rounding. KV VRAM falls from 5.11 to 0.09 MB/token, so long contexts no longer exhaust 22 GB cards.
        GEMMs use half precision from the precision policy (fp16/bf16 tensor cores), while softmax remains f32."""
        c = self.cfg
        H = c["n_heads"]
        p = f"model.layers.{layer}.self_attn"
        T = x.shape[0]
        dt = self.cdt
        if layer not in self._wuk:
            self._build_absorbed_layer(layer)   # Lazily build for paths that skipped preload (self-check/debug)
        q_a_weight = self.w(f"{p}.q_a_proj.weight")
        kv_a_weight = self.w(f"{p}.kv_a_proj_with_mqa.weight")
        fused_qkv = (
            self._norm_qkv_decode(
                x,
                input_norm_weight,
                q_a_weight.q,
                q_a_weight.s,
                kv_a_weight.q,
                kv_a_weight.s,
                q_a_weight.cols,
                q_a_weight.gs,
                c["rms_eps"],
                (
                    self._decode_workspace(
                        layer,
                        "q_a",
                        int(q_a_weight.shape[0]),
                    ),
                    self._decode_workspace(
                        layer,
                        "kv_a",
                        int(kv_a_weight.shape[0]),
                    ),
                )
                if os.environ.get(
                    "TPQ_DECODE_WORKSPACES",
                    "1",
                ) != "0"
                else None
            )
            if (
                T == 1
                and input_norm_weight is not None
                and self._norm_qkv_decode is not None
                and isinstance(q_a_weight, Int4Weight)
                and isinstance(kv_a_weight, Int4Weight)
                and q_a_weight.cols == kv_a_weight.cols
                and q_a_weight.gs == kv_a_weight.gs
            )
            else None
        )
        if fused_qkv is None:
            if input_norm_weight is not None:
                x = rmsnorm(x, input_norm_weight, c["rms_eps"])
            q_a = _attention_linear(x, q_a_weight)
            kv = _attention_linear(x, kv_a_weight)
        else:
            q_a, kv = fused_qkv
        q_norm_weight = self.w(f"{p}.q_a_layernorm.weight")
        q_b_weight = self.w(f"{p}.q_b_proj.weight")
        q_resid = rmsnorm(
            q_a,
            q_norm_weight,
            1e-6,
            output=(
                self._decode_workspace(
                    layer,
                    "q_a_norm",
                    int(q_a.shape[1]),
                )
                if (
                    T == 1
                    and os.environ.get(
                        "TPQ_RMSNORM_WORKSPACES",
                        "1",
                    )
                    != "0"
                )
                else None
            ),
        )
        fused_q_parts = (
            self._q_b_split_decode(
                q_resid,
                q_b_weight.q,
                q_b_weight.s,
                q_b_weight.cols,
                q_b_weight.gs,
                H,
                c["qk_nope_head_dim"],
                c["qk_rope_head_dim"],
                self._decode_tensor_workspace(
                    layer,
                    "q_nope_bf16",
                    (H, 1, c["qk_nope_head_dim"]),
                    dt,
                ),
                self._decode_tensor_workspace(
                    layer,
                    "q_rope_f32",
                    (H, 1, c["qk_rope_head_dim"]),
                    torch.float32,
                ),
            )
            if (
                T == 1
                and isinstance(q_b_weight, Int4Weight)
                and self._q_b_split_decode is not None
            )
            else None
        )
        if fused_q_parts is None:
            q = _attention_linear(
                q_resid,
                q_b_weight,
                output=(
                    self._decode_workspace(
                        layer,
                        "q_b",
                        int(q_b_weight.shape[0]),
                    )
                    if T == 1
                    and isinstance(q_b_weight, Int4Weight)
                    else None
                ),
            )
            q = q.view(
                T,
                H,
                c["qk_head_dim"],
            ).transpose(0, 1)
            q_nope, q_rot = q.split(
                [
                    c["qk_nope_head_dim"],
                    c["qk_rope_head_dim"],
                ],
                dim=-1,
            )
        else:
            q_nope, q_rot = fused_q_parts

        c_raw, k_rot = kv.split(
            [c["kv_lora_rank"], c["qk_rope_head_dim"]],
            dim=-1,
        )

        end = pos0 + T
        ckv_buffer, krot_buffer = self._latent_buffer(layer, end)
        prepared_q_rot = (
            self._latent_kv_decode_prepare(
                c_raw,
                self.w(f"{p}.kv_a_layernorm.weight"),
                q_rot,
                k_rot.view(1, T, c["qk_rope_head_dim"]),
                self.rope.cos,
                self.rope.sin,
                ckv_buffer,
                krot_buffer,
                self._decode_position,
                1e-6,
                (
                    self._decode_tensor_workspace(
                        layer,
                        "prepared_q_rot",
                        (
                            H,
                            1,
                            c["qk_rope_head_dim"],
                        ),
                        dt,
                    )
                    if os.environ.get(
                        "TPQ_ATTENTION_TENSOR_WORKSPACES",
                        "0",
                    ) != "0"
                    or os.environ.get(
                        "TPQ_ATTENTION_GRAPH",
                        "1",
                    ) != "0"
                    else None
                ),
            )
            if (
                T == 1
                and dt == torch.bfloat16
                and self._latent_kv_decode_prepare is not None
            )
            else None
        )
        if prepared_q_rot is not None:
            q_rot = prepared_q_rot
        else:
            c_new = rmsnorm(
                c_raw,
                self.w(f"{p}.kv_a_layernorm.weight"),
                1e-6,
            )
            q_rot, k_rot = self.rope.apply(
                q_rot,
                k_rot.view(1, T, c["qk_rope_head_dim"]),
                pos0,
            )
            ckv_buffer[pos0:end].copy_(c_new)
            krot_buffer[pos0:end].copy_(k_rot[0])
        ckv = ckv_buffer[:end]
        krot = krot_buffer[:end]
        self.kv[layer] = (ckv, krot)
        S = ckv.shape[0]

        scale = math.sqrt(c["qk_head_dim"])
        attention_workspace_enabled = (
            T == 1
            and (
                os.environ.get(
                    "TPQ_ATTENTION_TENSOR_WORKSPACES",
                    "0",
                )
                != "0"
                or os.environ.get(
                    "TPQ_ATTENTION_GRAPH",
                    "1",
                )
                != "0"
            )
        )
        # The contraction is a plain batched matmul.  Calling bmm directly
        # avoids einsum's equation parsing plus its permute/reshape dispatcher
        # path on every layer and decode token.
        if self._direct_mla_bmm:
            qa_input = (
                q_nope
                if q_nope.dtype == dt
                else q_nope.to(dt)
            )
            qa = (
                self._mla_bmm_decode(
                    qa_input,
                    self._wuk[layer],
                    False,
                    self._decode_tensor_workspace(
                        layer,
                        "mla_qa",
                        (H, 1, c["kv_lora_rank"]),
                        dt,
                    )
                )
                if (
                    T == 1
                    and self._mla_bmm_decode is not None
                    and (
                        os.environ.get(
                            "TPQ_GLM_CUBLAS_Q",
                            "0",
                        )
                        != "0"
                        or os.environ.get(
                            "TPQ_GLM_CUBLAS_DECODE",
                            "0",
                        )
                        != "0"
                    )
                )
                else None
            )
            if qa is None:
                qa = torch.bmm(
                    qa_input,
                    self._wuk[layer],
                    out=(
                        self._decode_tensor_workspace(
                            layer,
                            "mla_qa",
                            (H, 1, c["kv_lora_rank"]),
                            dt,
                        )
                        if attention_workspace_enabled
                        else None
                    ),
                )
        else:
            qa = torch.einsum(
                "htn,hnr->htr",
                q_nope.to(dt),
                self._wuk[layer],
            )
        flash_state = self._flashinfer_mla_state if T == 1 else None
        if flash_state is not None:
            from .flashinfer_mla import last_error
            from .ops import attention_step

            page = flash_state.page_size
            flash_out = attention_step(
                "paged_latent_decode",
                self.device.type,
                runner=flash_state,
                query_nope=qa.transpose(0, 1),
                query_rope=q_rot.to(dt).transpose(0, 1),
                latent_cache=ckv_buffer.view(
                    -1,
                    page,
                    c["kv_lora_rank"],
                ),
                rope_cache=krot_buffer.view(
                    -1,
                    page,
                    c["qk_rope_head_dim"],
                ),
            )
            if flash_out is not None:
                ctx = flash_out.transpose(0, 1)
                if self._direct_mla_bmm:
                    out = (
                        self._mla_bmm_decode(
                            ctx,
                            self._wuv[layer],
                            True,
                            self._decode_tensor_workspace(
                                layer,
                                "mla_value_output",
                                (H, 1, c["v_head_dim"]),
                                dt,
                            )
                        )
                        if (
                            T == 1
                            and self._mla_bmm_decode is not None
                            and (
                                os.environ.get(
                                    "TPQ_GLM_CUBLAS_VALUE",
                                    "1",
                                )
                                != "0"
                                or os.environ.get(
                                    "TPQ_GLM_CUBLAS_DECODE",
                                    "0",
                                )
                                != "0"
                            )
                        )
                        else None
                    )
                    if out is None:
                        out = torch.bmm(
                            ctx,
                            self._wuv[layer].transpose(1, 2),
                            out=(
                                self._decode_tensor_workspace(
                                    layer,
                                    "mla_value_output",
                                    (H, 1, c["v_head_dim"]),
                                    dt,
                                )
                                if attention_workspace_enabled
                                else None
                            ),
                        )
                else:
                    out = torch.einsum(
                        "htr,hnr->htn",
                        ctx,
                        self._wuv[layer],
                    )
                out = out.transpose(0, 1).reshape(
                    T,
                    H * c["v_head_dim"],
                )
                return self._attention_output(
                    out,
                    layer,
                    self.w(f"{p}.o_proj.weight"),
                )
            self._flashinfer_mla_unavailable = True
            self._flashinfer_mla_state = None
            print(
                "[tpq] FlashInfer MLA kernel 失败，回退原 PyTorch MLA："
                f"{last_error()}",
                flush=True,
            )
        score_nope = qa @ ckv.t()
        score_rope = q_rot.to(dt) @ krot.t()
        scores = merge_attention_scores(
            score_nope, score_rope, scale
        )                                                           # [H,T,S] f32
        if T > 1:  # During single-token decode, all history is visible and no mask is needed
            kpos = torch.arange(S, device=x.device)
            qpos = torch.arange(pos0, pos0 + T, device=x.device)
            causal = kpos[None, :] > qpos[:, None]
            scores = scores.masked_fill(causal[None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = attn.to(dt) @ ckv                                     # [H,T,R]
        if self._direct_mla_bmm:
            out = (
                self._mla_bmm_decode(
                    ctx,
                    self._wuv[layer],
                    True,
                    self._decode_tensor_workspace(
                        layer,
                        "mla_value_output",
                        (H, 1, c["v_head_dim"]),
                        dt,
                    )
                )
                if (
                    T == 1
                    and self._mla_bmm_decode is not None
                    and (
                        os.environ.get(
                            "TPQ_GLM_CUBLAS_VALUE",
                            "1",
                        )
                        != "0"
                        or os.environ.get(
                            "TPQ_GLM_CUBLAS_DECODE",
                            "0",
                        )
                        != "0"
                    )
                )
                else None
            )
            if out is None:
                out = torch.bmm(
                    ctx,
                    self._wuv[layer].transpose(1, 2),
                    out=(
                        self._decode_tensor_workspace(
                            layer,
                            "mla_value_output",
                            (H, 1, c["v_head_dim"]),
                            dt,
                        )
                        if attention_workspace_enabled
                        else None
                    ),
                )
        else:
            out = torch.einsum(
                "htr,hnr->htn",
                ctx,
                self._wuv[layer],
            )
        out = out.transpose(0, 1).reshape(T, H * c["v_head_dim"])
        return self._attention_output(
            out,
            layer,
            self.w(f"{p}.o_proj.weight"),
        )

    def _attention_full(self, x: torch.Tensor, layer: int, pos0: int) -> torch.Tensor:
        c = self.cfg
        H = c["n_heads"]
        p = f"model.layers.{layer}.self_attn"
        T = x.shape[0]
        q_resid = rmsnorm(_attention_linear(x, self.w(f"{p}.q_a_proj.weight")),
                          self.w(f"{p}.q_a_layernorm.weight"), 1e-6)
        q = _attention_linear(q_resid, self.w(f"{p}.q_b_proj.weight"))
        q = q.view(T, H, c["qk_head_dim"]).transpose(0, 1)
        q_nope, q_rot = q.split([c["qk_nope_head_dim"], c["qk_rope_head_dim"]], dim=-1)

        kv = _attention_linear(
            x, self.w(f"{p}.kv_a_proj_with_mqa.weight")
        )
        k_pass, k_rot = kv.split([c["kv_lora_rank"], c["qk_rope_head_dim"]], dim=-1)
        k_pass = rmsnorm(k_pass, self.w(f"{p}.kv_a_layernorm.weight"), 1e-6)
        k_pass = _attention_linear(
            k_pass, self.w(f"{p}.kv_b_proj.weight")
        )
        k_pass = k_pass.view(T, H, c["qk_nope_head_dim"] + c["v_head_dim"]).transpose(0, 1)
        k_nope, v = k_pass.split([c["qk_nope_head_dim"], c["v_head_dim"]], dim=-1)

        q_rot, k_rot = self.rope.apply(q_rot, k_rot.view(1, T, c["qk_rope_head_dim"]), pos0)
        k_rot = k_rot.expand(H, T, c["qk_rope_head_dim"])
        q_f = torch.cat([q_nope, q_rot], dim=-1)
        k_f = torch.cat([k_nope, k_rot], dim=-1)

        past = self.kv[layer]
        if past is not None:
            k_f = torch.cat([past[0].float(), k_f], dim=1)
            v = torch.cat([past[1].float(), v], dim=1)
        self.kv[layer] = (k_f.half(), v.half())

        scores = (q_f.float() @ k_f.float().transpose(1, 2)) / math.sqrt(c["qk_head_dim"])
        S = scores.shape[-1]
        if T > 1:  # During single-token decode, all history is visible and no mask is needed
            kpos = torch.arange(S, device=x.device)
            qpos = torch.arange(pos0, pos0 + T, device=x.device)
            causal = kpos[None, :] > qpos[:, None]
            scores = scores.masked_fill(causal[None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v.float()).transpose(0, 1).reshape(T, H * c["v_head_dim"])
        return self._attention_output(
            out,
            layer,
            self.w(f"{p}.o_proj.weight"),
        )

    def _moe(
        self,
        x: torch.Tensor,
        layer: int,
        route_logits: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.cfg
        p = f"model.layers.{layer}.mlp"
        parallel = self.expert_parallel
        route_start = (
            parallel.profile_event()
            if parallel is not None and parallel.profile_enabled
            else None
        )
        logits = (
            route_logits
            if route_logits is not None
            else _linear(x, self.w(f"{p}.gate.weight")).float()
        )
        mask = self._mask(layer)
        w, idx = _glm_route(
            logits,
            self.w(f"{p}.gate.e_score_correction_bias").float(),
            mask,
            c["top_k"],
            c["routed_scaling"],
            (
                parallel.decode_route_outputs()
                if parallel is not None
                else None
            ),
        )

        def merge_outputs(
            routed: torch.Tensor,
            shared: torch.Tensor,
        ) -> torch.Tensor:
            routed = routed.to(shared.dtype)
            fused = (
                self._moe_residual_add_decode(
                    residual,
                    routed,
                    shared,
                )
                if (
                    residual is not None
                    and self._moe_residual_add_decode is not None
                )
                else None
            )
            if fused is not None:
                return fused
            result = routed + shared
            return residual + result if residual is not None else result

        if parallel is not None:
            route_end = parallel.profile_event()
            parallel.profile_cuda("route", route_start, route_end)

            def compute_shared_expert() -> torch.Tensor:
                return self._shared_expert_eager(x, layer)

            overlapped_final = (
                parallel.compute_final_overlap(
                    x,
                    layer,
                    idx,
                    w,
                    compute_shared_expert,
                    residual,
                )
                if residual is not None
                else None
            )
            if overlapped_final is not None:
                return overlapped_final

            shared_start = parallel.profile_event()
            shared = compute_shared_expert()
            shared_end = parallel.profile_event()
            final = (
                parallel.compute_final(
                    x,
                    layer,
                    idx,
                    w,
                    shared,
                    residual,
                )
                if residual is not None
                else None
            )
            routed = (
                parallel.compute(x, layer, idx, w)
                if final is None
                else None
            )
            parallel.profile_cuda(
                "shared_expert",
                shared_start,
                shared_end,
            )
            if final is not None:
                return final
            assert routed is not None
            add_start = parallel.profile_event()
            result = merge_outputs(routed, shared)
            add_end = parallel.profile_event()
            parallel.profile_cuda("final_add", add_start, add_end)
            return result

        # Single-token decode reuses DSV4's validated top-k VQ grouped/SM120-slot path.
        # Queue the shared expert first so the GPU remains busy during the indices DtoH synchronization.
        if x.shape[0] == 1 and os.environ.get("TPQ_GROUPED", "1") != "0":
            from .grouped import moe_mlp_grouped_mixed

            shared = _linear(
                _swiglu_linear(
                    x,
                    self.w(
                        f"{p}.shared_experts.gate_proj.weight"
                    ),
                    self.w(f"{p}.shared_experts.up_proj.weight"),
                ),
                self.w(f"{p}.shared_experts.down_proj.weight"),
            )
            eids = idx[0].tolist()
            self._prev_ids[layer] = eids
            got = self.pool.get_many([(layer, expert) for expert in eids])
            experts = [got[(layer, expert)] for expert in eids]
            if all(
                isinstance(gu, VQWeight) and isinstance(dn, VQWeight)
                for gu, dn in experts
            ):
                routed = moe_mlp_grouped_mixed(
                    x,
                    experts,
                    w[0],
                    limit=0.0,
                )
                return merge_outputs(
                    routed.unsqueeze(0),
                    shared,
                )

        inter_size = c["moe_inter"]
        y = torch.zeros_like(x)
        uniq = idx.unique()
        eids = uniq.tolist()                        # One DtoH synchronization (the original implementation used two)
        self._prev_ids[layer] = eids
        need = [(layer, e) for e in eids]
        experts = self.pool.get_many(need)                  # Parallel loading/upload
        for e in eids:
            toks, slots = (idx == e).nonzero(as_tuple=True)
            gu, dn = experts[(layer, e)]
            h = gu.matmul_T(x[toks])                        # [N, 2I]
            inter = F.silu(h[:, :inter_size]) * h[:, inter_size:]
            y.index_add_(0, toks, dn.matmul_T(inter) * w[toks, slots].unsqueeze(1))
        shared = _linear(
            _swiglu_linear(
                x,
                self.w(f"{p}.shared_experts.gate_proj.weight"),
                self.w(f"{p}.shared_experts.up_proj.weight"),
            ),
            self.w(f"{p}.shared_experts.down_proj.weight"),
        )
        return merge_outputs(y, shared)

    def _dense_mlp(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        p = f"model.layers.{layer}.mlp"
        return _linear(
            _swiglu_linear(
                x,
                self.w(f"{p}.gate_proj.weight"),
                self.w(f"{p}.up_proj.weight"),
            ),
            self.w(f"{p}.down_proj.weight"),
        )

    def _mask(self, layer: int) -> torch.Tensor:
        """Return the cached available-expert mask for this layer, with dropped experts set to False."""
        m = self._masks.get(layer)
        if m is None:
            m = self.store.available_mask(layer).to(self.device)
            self._masks[layer] = m
        return m

    def forward_hidden(
        self,
        ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Forward a token segment and return final-normalized hidden states [T, hidden] for all positions."""
        c = self.cfg
        eps = c["rms_eps"]
        if self.pos + len(ids) > self.max_ctx:
            raise RuntimeError(f"上下文超限（{self.pos + len(ids)} > {self.max_ctx}），"
                               f"请 /clear 或调大 --max-ctx")
        pos0 = self.pos
        self._ensure_rope_capacity(pos0 + len(ids))
        self._ensure_latent_capacity(pos0 + len(ids))
        if len(ids) == 1 and self._decode_position is not None:
            self._decode_position.fill_(pos0)
        self._flashinfer_mla_state = (
            self._prepare_flashinfer_mla_decode(pos0 + len(ids))
            if len(ids) == 1
            else None
        )
        x = self.embed(ids)
        if self._prev_ids and len(ids) == 1 and os.environ.get("TPQ_PREFETCH", "1") != "0":
            # Single decode step: token-level all-layer prefetch (window = the entire token; see dsv4model.decode)
            for layer_id, expert_ids in self._prev_ids.items():
                self.pool.prefetch(
                    [(layer_id, expert_id) for expert_id in expert_ids]
                )
        moe_set = set(c["moe_layers"])
        for layer in range(c["n_layers"]):
            # Cross-layer expert prefetch (B2): overlap experts routed by this layer for the previous token with this layer's attention computation
            prev = self._prev_ids.get(layer)
            if prev and len(ids) == 1 and os.environ.get("TPQ_PREFETCH", "1") != "0":
                self.pool.prefetch([(layer, e) for e in prev])
            h = self._attention(
                x,
                layer,
                pos0,
                self.w(
                    f"model.layers.{layer}.input_layernorm.weight"
                ),
            )
            if layer in moe_set:
                post_norm = self.w(
                    f"model.layers.{layer}."
                    "post_attention_layernorm.weight"
                )
                route_weight = self.w(
                    f"model.layers.{layer}.mlp.gate.weight"
                )
                fused_post = (
                    self._residual_norm_router_decode(
                        x,
                        h,
                        post_norm,
                        route_weight,
                        eps,
                        (
                            self.expert_parallel.decode_norm_output()
                            if self.expert_parallel is not None
                            else None
                        ),
                        (
                            self._decode_workspace(
                                layer,
                                "post_attention_residual",
                                int(x.shape[1]),
                            ),
                            self._decode_workspace(
                                layer,
                                "route_logits",
                                int(route_weight.shape[0]),
                            ),
                        )
                        if (
                            self.expert_parallel is not None
                            and os.environ.get(
                                "TPQ_DECODE_WORKSPACES",
                                "1",
                            ) != "0"
                        )
                        else None,
                    )
                    if self._residual_norm_router_decode is not None
                    else None
                )
                if fused_post is None:
                    x = x + h
                    hn = rmsnorm(x, post_norm, eps)
                    route_logits = None
                else:
                    x, hn, route_logits = fused_post
                x = self._moe(
                    hn,
                    layer,
                    route_logits=route_logits,
                    residual=x,
                )
            else:
                x = x + h
                hn = rmsnorm(
                    x,
                    self.w(
                        f"model.layers.{layer}."
                        "post_attention_layernorm.weight"
                    ),
                    eps,
                )
                x = x + self._dense_mlp(hn, layer)
        self.pos += len(ids)
        return rmsnorm(
            x,
            self.w("model.norm.weight"),
            eps,
            output=(
                self._decode_workspace(
                    -1,
                    "final_norm",
                    int(x.shape[1]),
                )
                if (
                    x.shape[0] == 1
                    and
                    os.environ.get(
                        "TPQ_STATIC_LM_OUTPUT",
                        "0",
                    )
                    != "0"
                    and os.environ.get(
                        "TPQ_RMSNORM_WORKSPACES",
                        "1",
                    )
                    != "0"
                )
                else None
            ),
        )

    def logits_of(self, h: torch.Tensor) -> torch.Tensor:
        """Map hidden [N, hidden] to logits [N, vocab] through lm_head using blocked int4 or direct f32 multiplication."""
        lm = self.w("lm_head.weight")
        if (
            self._lm_head_int4 is not None
            and os.environ.get("TPQ_LM_HEAD_INT4", "1") != "0"
        ):
            lm = self._lm_head_int4
        if isinstance(lm, Int4Weight):
            return lm.matmul_T_decode_fused(
                h,
                output=(
                    self._decode_workspace(
                        -1,
                        "lm_logits",
                        int(lm.shape[0]),
                    )
                    if (
                        h.shape[0] == 1
                        and os.environ.get(
                            "TPQ_STATIC_LM_OUTPUT",
                            "0",
                        )
                        != "0"
                    )
                    else None
                ),
            )
        return h.float() @ lm.t()

    def forward(
        self,
        ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Forward a token segment for prefill or one-step decode and return final-position f32 logits [vocab]."""
        h = self.forward_hidden(ids)
        return self.logits_of(h[-1:]).squeeze(0)
