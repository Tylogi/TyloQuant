"""TPQ 推理：DeepSeek-V4（CCCP 产物）前向模型。

加载 "cccp-1" 格式（cccp.json + dense.safetensors + experts.L*.safetensors），
前向复用 CCCP/dsv4.py 的公共数学件（hc_pre/hc_post/hc_head/hc_split/rmsnorm/
rope_apply/compressor_*/attn_* 的无权重依赖部分）；权重路径：
  - 大 dense 矩阵（wq_a/wq_b/wkv/wo_a/wo_b/shared/head/embed）：Int4Weight 打包驻留
    （显存/内存），经 _linear 走 LUT 反量化矩阵乘；
  - 小权重（compressor/norms/hc/gate/attn_sink/ape/tid2eid）：f32 原样；
  - routed 专家：ExpertPool 两级 LRU 的 VQWeight（LUT 免还原矩阵乘）。
与 CCCP/dsv4.py 的关系：数值公式一致；为接入 int4/VQ 权重，线性层经本文件的
_linear 分派（F.linear ↔ Int4Weight.matmul_T ↔ VQWeight.matmul_T）。
"""

from __future__ import annotations

import os
import copy
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .dsv4cache import ContextCapacityError, PagedKV
from .dsv4indexer import IndexerState
from .kernels import (
    BlockFP8Weight,
    Int4Weight,
    VQWeight,
    rmsnorm,
)
from .precision import compute_dtype
from .store import TPQStore, ExpertPool


_DENSE_BF16_ELIGIBLE = frozenset({
    "attention", "compressor", "embed", "head", "hyper", "indexer",
    "norm", "shared",
})
_DENSE_BF16_ALIASES = {
    "core": "attention",
    "embedding": "embed",
    "output_head": "head",
    "shared_experts": "shared",
}


@dataclass(frozen=True)
class DSV4LayerKVSnapshot:
    """Copied mutable state for one DSV4 layer at a stable prompt boundary."""

    kv: torch.Tensor
    win_pos: torch.Tensor
    compressed_length: int | None
    ckv: torch.Tensor | None
    cscore: torch.Tensor | None
    indexer_length: int | None
    indexer_ckv: torch.Tensor | None
    indexer_cscore: torch.Tensor | None


@dataclass(frozen=True)
class DSV4KVSnapshot:
    """One bounded rollback point; paged payloads remain append-only."""

    pos: int
    layers: tuple[DSV4LayerKVSnapshot, ...]

    @property
    def nbytes(self) -> int:
        tensors: list[torch.Tensor] = []
        for layer in self.layers:
            tensors.extend(
                tensor
                for tensor in (
                    layer.kv,
                    layer.win_pos,
                    layer.ckv,
                    layer.cscore,
                    layer.indexer_ckv,
                    layer.indexer_cscore,
                )
                if tensor is not None
            )
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
        )


def _prefill_ranges(tokens: int, chunk_size: int = 512):
    if tokens < 0 or chunk_size <= 0:
        raise ValueError("tokens must be non-negative and chunk_size positive")
    for start in range(0, tokens, chunk_size):
        yield start, min(tokens, start + chunk_size)


def _dense_bf16_group(name: str) -> str | None:
    """Return the BF16 residency group for a dense Int4 weight, if eligible."""
    if name == "head.weight":
        return "head"
    if name == "embed.weight":
        return "embed"
    if ".ffn.shared_experts." in name:
        return "shared"
    if name.endswith("_fn"):
        return "hyper"
    if name == "norm.weight" or name.endswith(".attn_norm.weight") \
            or name.endswith(".ffn_norm.weight"):
        return "norm"
    if name.endswith(".q_norm.weight") or name.endswith(".kv_norm.weight"):
        return None
    if name.endswith(".norm.weight"):
        return None
    if name.endswith(".attn.attn_sink"):
        return None
    if ".attn.indexer." in name:
        return "indexer"
    if ".attn.compressor." in name:
        return "compressor"
    if ".attn." in name:
        return "attention"
    # Router and hyperconnection weights are explicitly consumed as FP32.
    return None


def _parse_dense_bf16(value: str | None = None) -> frozenset[str]:
    """Parse TPQ_DENSE_BF16 into the eligible dense residency groups."""
    raw = os.environ.get("TPQ_DENSE_BF16", "none") if value is None else value
    raw = raw.strip().lower()
    if raw in ("", "0", "false", "off", "none"):
        return frozenset()
    if raw in ("1", "true", "all"):
        return _DENSE_BF16_ELIGIBLE
    groups = {
        _DENSE_BF16_ALIASES.get(part.strip().replace("-", "_"),
                                part.strip().replace("-", "_"))
        for part in raw.split(",") if part.strip()
    }
    unknown = groups - _DENSE_BF16_ELIGIBLE
    if unknown:
        valid = ", ".join(sorted(_DENSE_BF16_ELIGIBLE))
        raise ValueError(
            f"unknown TPQ_DENSE_BF16 group(s): {', '.join(sorted(unknown))}; "
            f"valid groups: {valid}"
        )
    return frozenset(groups)


def _tpq_lin(x: torch.Tensor, w) -> torch.Tensor:
    """安装进 CCCP.dsv4._lin 的分派：Int4Weight/VQWeight 走 LUT 矩阵乘，其余 F.linear。
    权重类的 matmul_T 只收 2D 输入，dsv4 的 3D [B,T,D] 在此压平再还原；
    dense bf16 常驻后：matmul 与层间 hidden 均保持 bf16；需要稳定归约的算子
    在各自内部局部升到 f32。"""
    if isinstance(w, (Int4Weight, BlockFP8Weight, VQWeight)):
        if x.dim() > 2:
            sh = x.shape
            rows = x.reshape(-1, sh[-1])
            out = (
                w.matmul_T_decode_fused(rows)
                if isinstance(w, BlockFP8Weight)
                else w.matmul_T(rows)
            ).view(*sh[:-1], -1)
        else:
            out = (
                w.matmul_T_decode_fused(x)
                if isinstance(w, BlockFP8Weight)
                else w.matmul_T(x)
            )
        return out.to(compute_dtype(x.device))
    if x.dtype != w.dtype:
        x = x.to(w.dtype)
    return F.linear(x, w)


from . import dsv4 as _dsv4
def _o_proj_tpq(o: torch.Tensor, w: dict, cfg) -> torch.Tensor:
    """分组 LoRA O 的实现：Int4Weight 走逐组 dequant_rows；bf16 常驻张量走原生路径
    （dtype 对齐）。数值与 CCCP.dsv4._o_proj 一致。"""
    wo_a = w["wo_a"]
    if not isinstance(wo_a, (Int4Weight, BlockFP8Weight)):
        if o.dtype != wo_a.dtype:
            o = o.to(wo_a.dtype)
        return _dsv4._o_proj(o, w, cfg)
    B, T = o.shape[0], o.shape[1]
    G = cfg.o_groups
    rank = cfg.o_lora_rank
    o = o.reshape(B, T, G, -1)
    if isinstance(wo_a, BlockFP8Weight):
        # Each group owns an aligned row range in the compact source tensor.
        # The public linear operator reads a zero-copy FP8 view directly;
        # only the token-sized LoRA result is concatenated.
        from .ops import linear

        groups = []
        for group in range(G):
            compact = wo_a.row_view(
                group * rank,
                (group + 1) * rank,
            )
            value = o[:, :, group].reshape(-1, o.shape[-1])
            groups.append(
                linear(value, compact).view(B, T, rank)
            )
        return _tpq_lin(torch.cat(groups, dim=-1), w["wo_b"])
    if not o.is_cuda and B * T == 1:
        wo_b = w["wo_b"]
        if isinstance(wo_b, Int4Weight):
            from .cpuext import o_proj_int4_cpu

            fused = o_proj_int4_cpu(
                o.reshape(G, -1),
                wo_a.q,
                wo_a.s,
                wo_a.cols,
                wo_a.gs,
                rank,
                wo_b.q,
                wo_b.s,
                wo_b.cols,
                wo_b.gs,
            )
            if fused is not None:
                return fused.view(B, T, -1)

        from .cpuext import int4_grouped_gemv_cpu

        grouped = int4_grouped_gemv_cpu(
            o.reshape(G, -1),
            wo_a.q,
            wo_a.s,
            wo_a.cols,
            wo_a.gs,
            rank,
        )
        if grouped is not None:
            return _tpq_lin(
                grouped.reshape(B, T, G * rank), w["wo_b"]
            )
    outs = []
    for g in range(G):
        wa_g = wo_a.dequant_rows(g * rank, (g + 1) * rank)  # [rank, D]（Int4Weight.half 时 fp16）
        og = o[:, :, g]
        outs.append((og.half() @ wa_g.t()).float() if wa_g.dtype != og.dtype else og @ wa_g.t())
    o = torch.stack(outs, dim=2)
    return _tpq_lin(o.flatten(2), w["wo_b"])


_dsv4._lin = _tpq_lin              # 线性层钩子：dsv4.py 全部线性层经此分派
_dsv4._o_proj_hook = _o_proj_tpq  # O 投影钩子安装（dsv4.py 的 attn 经此走 Int4 分组反量化）

# HC sinkhorn 融合钩子：20 轮 4×4 双随机归一化原本每轮 4 次小 kernel（每层 attn/ffn
# 两次调用，逐 token ~6500 次 launch），融合后一次 launch。无扩展/非 CUDA/f64 时
# 返回 None 回退原 torch 循环（dspark.py 的 hc_pre/hc_post 同享本钩子）。
from .fusedext import hc_split_fused as _hc_fused

_hc_split_orig = _dsv4.hc_split


def _hc_split_tpq(mixes, scale, base, hc, iters, eps):
    r = _hc_fused(mixes, scale, base, hc, iters, eps)
    if r is not None:
        return r
    return _hc_split_orig(mixes, scale, base, hc, iters, eps)


_dsv4.hc_split = _hc_split_tpq

# RMSNorm 融合钩子：pow/mean/rsqrt/两次乘 ~6 次 launch → 1 次（dsv4 每层 4+ 处）。
from .fusedext import rmsnorm_fused as _rms_fused

_rmsnorm_orig = _dsv4.rmsnorm


def _rmsnorm_tpq(x, w, eps):
    r = _rms_fused(x, w, eps)
    if r is not None:
        return r
    return _rmsnorm_orig(x, w, eps)


_dsv4.rmsnorm = _rmsnorm_tpq

# RoPE 融合钩子：decode 单相位（全部行同一 cos/sin）时 1 次 launch 替代 ~8 次
from .fusedext import rope1_fused as _rope_fused

_rope_orig = _dsv4.rope_apply


def _rope_tpq(x, cos, sin, inverse=False):
    r = _rope_fused(x, cos, sin, inverse)
    if r is not None:
        return r
    return _rope_orig(x, cos, sin, inverse=inverse)


_dsv4.rope_apply = _rope_tpq

from .fusedext import dsv4_attn_decode_fused as _attn_decode_fused

_dsv4._attn_decode_core_hook = _attn_decode_fused

from .fusedext import dsv4_hc_pre_fused as _hc_pre_fused
from .fusedext import dsv4_route_post_fused as _route_post_fused
from .ops import (
    hyper_connection_post as _hyper_connection_post,
    hyper_connection_post_moe as _hyper_connection_post_moe,
    hyper_connection_pre_norm as _hyper_connection_pre_norm,
)

_hc_pre_orig = _dsv4.hc_pre
_hc_post_orig = _dsv4.hc_post


def _hc_pre_tpq(x, fn, scale, base, cfg):
    r = _hc_pre_fused(
        x, fn, scale, base, cfg.hc_sinkhorn_iters, cfg.hc_eps
    )
    if r is not None:
        return r
    return _hc_pre_orig(x, fn, scale, base, cfg)


_dsv4.hc_pre = _hc_pre_tpq


def _hc_post_tpq(out, residual, post, comb, output=None):
    if not residual.is_cuda:
        from .cpuext import hc_post_cpu

        cpu_result = hc_post_cpu(out, residual, post, comb)
        if cpu_result is not None:
            return cpu_result
    r = _hyper_connection_post(
        out,
        residual,
        post,
        comb,
        output=output,
    )
    if r is not None:
        return r
    return _hc_post_orig(out, residual, post, comb)


_dsv4.hc_post = _hc_post_tpq


def _hc_pre_norm_tpq(
    x,
    fn,
    scale,
    base,
    norm,
    cfg,
    output_buffers=None,
):
    """HC pre 与随后 RMSNorm 的 BF16 热路径；归约仍在核内使用 FP32。"""
    if not x.is_cuda and x.shape[0] * x.shape[1] == 1:
        from .cpuext import hc_pre_norm_cpu

        xf = x.flatten(2).float()
        mixes = _tpq_lin(xf, fn).float()
        cpu_result = hc_pre_norm_cpu(
            x,
            mixes,
            scale,
            base,
            norm,
            cfg.hc_sinkhorn_iters,
            cfg.rms_eps,
            cfg.hc_eps,
        )
        if cpu_result is not None:
            return cpu_result
    r = _hyper_connection_pre_norm(
        x,
        fn,
        scale,
        base,
        norm,
        cfg.hc_sinkhorn_iters,
        cfg.rms_eps,
        output_buffers=output_buffers,
    )
    if r is not None:
        return r
    y, post, comb = _dsv4.hc_pre(x, fn, scale, base, cfg)
    y = _dsv4.rmsnorm(y, norm, cfg.rms_eps)
    dtype = compute_dtype(x.device)
    return y.to(dtype), post.to(dtype), comb.to(dtype)


def _linear(x: torch.Tensor, w) -> torch.Tensor:
    """dense 线性层：Int4Weight 走分块反量化（3D 输入压平再还原），其余按 dtype 对齐 matmul。"""
    if isinstance(w, (Int4Weight, BlockFP8Weight)):
        if x.dim() > 2:
            sh = x.shape
            rows = x.reshape(-1, sh[-1])
            out = (
                w.matmul_T_decode_fused(rows)
                if isinstance(w, BlockFP8Weight)
                else w.matmul_T(rows)
            ).view(*sh[:-1], -1)
        else:
            out = (
                w.matmul_T_decode_fused(x)
                if isinstance(w, BlockFP8Weight)
                else w.matmul_T(x)
            )
        return out.to(compute_dtype(x.device))
    if x.dtype != w.dtype:
        x = x.to(w.dtype)
    return x @ w.t()


def _qkv_tpq(x, w, cfg, cache, pos0):
    """CPU decode 将共享输入的 Q-rank 与 KV INT4 投影合并到一个并行区。"""
    if (
        not x.is_cuda
        and x.shape[0] * x.shape[1] == 1
        and os.environ.get("TPQ_CPU_ATTN_MANY", "1") != "0"
        and isinstance(w["wq_a"], Int4Weight)
        and isinstance(w["wkv"], Int4Weight)
        and w["wq_a"].gs == w["wkv"].gs
    ):
        from .cpuext import int4_gemv_many_cpu

        outputs = int4_gemv_many_cpu(
            x.flatten(0, 1),
            [w["wq_a"].q, w["wkv"].q],
            [w["wq_a"].s, w["wkv"].s],
            w["wq_a"].gs,
        )
        if outputs is not None:
            B, T = x.shape[:2]
            H, hd, rd = (
                cfg.n_heads,
                cfg.head_dim,
                cfg.qk_rope_head_dim,
            )
            cos = cache.cos[pos0:pos0 + T]
            sin = cache.sin[pos0:pos0 + T]
            if os.environ.get("TPQ_CPU_QKV_POST", "1") != "0":
                from .cpuext import q_post_cpu, qkv_pre_cpu

                preprocessed = qkv_pre_cpu(
                    outputs[0],
                    outputs[1],
                    w["q_norm"],
                    w["kv_norm"],
                    cos,
                    sin,
                    cfg.rms_eps,
                )
                if preprocessed is not None:
                    qr = preprocessed[0].view(B, T, -1)
                    kv = preprocessed[1].view(B, T, -1)
                    if isinstance(w["wq_b"], Int4Weight):
                        from .cpuext import q_int4_post_cpu

                        wq_b = w["wq_b"]
                        q = q_int4_post_cpu(
                            preprocessed[0],
                            wq_b.q,
                            wq_b.s,
                            wq_b.cols,
                            wq_b.gs,
                            cos,
                            sin,
                            H,
                            hd,
                            cfg.rms_eps,
                        )
                        if q is not None:
                            return qr, q, kv
                    q = _tpq_lin(qr, w["wq_b"]).view(
                        B, T, H, hd
                    ).float()
                    q = q_post_cpu(q, cos, sin, cfg.rms_eps)
                    if q is not None:
                        return qr, q, kv
            qr = _dsv4.rmsnorm(
                outputs[0].view(B, T, -1),
                w["q_norm"],
                cfg.rms_eps,
            )
            q = _tpq_lin(qr, w["wq_b"]).view(B, T, H, hd).float()
            q *= torch.rsqrt(
                q.square().mean(-1, keepdim=True) + cfg.rms_eps
            )
            q[..., hd - rd:] = _dsv4.rope_apply(
                q[..., hd - rd:],
                cos.view(1, T, 1, -1),
                sin.view(1, T, 1, -1),
            )
            kv = _dsv4.rmsnorm(
                outputs[1].view(B, T, -1),
                w["kv_norm"],
                cfg.rms_eps,
            )
            kv[..., hd - rd:] = _dsv4.rope_apply(
                kv[..., hd - rd:],
                cos.view(1, T, -1),
                sin.view(1, T, -1),
            )
            return qr, q, kv
    return _dsv4._qkv(x, w, cfg, cache, pos0)


def _compressor_decode_tpq(
    x,
    w,
    ratio,
    d,
    rd,
    cos,
    sin,
    eps,
    st,
    pos,
):
    """CPU decode 将 Compressor 的 KV/Gate INT4 投影合并。"""
    if (
        not x.is_cuda
        and x.shape[0] * x.shape[1] == 1
        and os.environ.get("TPQ_CPU_ATTN_MANY", "1") != "0"
        and isinstance(w["wkv"], Int4Weight)
        and isinstance(w["wgate"], Int4Weight)
        and w["wkv"].gs == w["wgate"].gs
    ):
        from .cpuext import int4_gemv_many_cpu

        outputs = int4_gemv_many_cpu(
            x.flatten(0, 1),
            [w["wkv"].q, w["wgate"].q],
            [w["wkv"].s, w["wgate"].s],
            w["wkv"].gs,
        )
        if outputs is not None:
            B, T = x.shape[:2]
            kv = outputs[0].view(B, T, -1)
            score = (
                outputs[1].view(B, T, -1)
                + w["ape"][pos % ratio]
            )
            coff = kv.shape[-1] // d
            overlap = coff == 2
            should_pool = (pos + 1) % ratio == 0
            if overlap:
                st["ckv"][:, ratio + pos % ratio] = kv[:, 0]
                st["cscore"][:, ratio + pos % ratio] = score[:, 0]
                if not should_pool:
                    return None
                kvs = torch.cat(
                    [
                        st["ckv"][:, :ratio, :d],
                        st["ckv"][:, ratio:, d:],
                    ],
                    dim=1,
                )
                scores = torch.cat(
                    [
                        st["cscore"][:, :ratio, :d],
                        st["cscore"][:, ratio:, d:],
                    ],
                    dim=1,
                )
                probs = scores.float().softmax(dim=1)
                pooled = (
                    kvs.float() * probs
                ).sum(dim=1, keepdim=True)
                st["ckv"][:, :ratio] = st["ckv"][:, ratio:].clone()
                st["cscore"][:, :ratio] = (
                    st["cscore"][:, ratio:].clone()
                )
            else:
                st["ckv"][:, pos % ratio] = kv[:, 0]
                st["cscore"][:, pos % ratio] = score[:, 0]
                if not should_pool:
                    return None
                probs = st["cscore"].float().softmax(dim=1)
                pooled = (
                    st["ckv"].float() * probs
                ).sum(dim=1, keepdim=True)
            pooled = _dsv4.rmsnorm(pooled, w["norm"], eps)
            pooled[..., d - rd:] = _dsv4.rope_apply(
                pooled[..., d - rd:], cos, sin
            )
            return pooled
    return _dsv4.compressor_decode(
        x, w, ratio, d, rd, cos, sin, eps, st, pos
    )


class DSV4TPQModel:
    """DeepSeek-V4 CCCP 产物的推理模型（CPU/CUDA，内存显存自动适配由外层 Engine 定）。"""

    def __init__(self, root: str, cache_gb: float = 16.0, max_ctx: int = 2048,
                 device: str = "cpu", vram_cache_gb: float = 4.0,
                 tp_size: int = 1):
        self.tp_size = int(tp_size)
        if self.tp_size <= 0:
            raise ValueError("tp_size must be positive")
        requested = torch.device(device)
        if requested.type == "cuda" and self.tp_size > 1:
            if self.tp_size > torch.cuda.device_count():
                raise ValueError(
                    f"tp={self.tp_size} exceeds visible CUDA devices"
                )
            self.devices = tuple(
                torch.device("cuda", rank)
                for rank in range(self.tp_size)
            )
            self.device = self.devices[0]
        else:
            self.device = requested
            self.devices = (self.device,)
        self._cpu_numa_interleaved = False
        self._cpu_threads = 0
        if self.device.type == "cpu":
            # RAM/GPU 预设可能携带 BF16；CPU 单 Token GEMV 与融合 HC 的
            # 已验证热路径是 FP32，除非显式开启实验开关，否则在模型构造前纠正。
            if os.environ.get("TPQ_CPU_BF16", "0") != "1":
                os.environ["TPQ_COMPUTE_DTYPE"] = "fp32"
                os.environ["TPQ_DENSE_BF16"] = "none"
            from .cpuext import (
                configure_cpu_threads,
                configure_numa_interleave,
            )

            self._cpu_threads = configure_cpu_threads()
            self._cpu_numa_interleaved = configure_numa_interleave()
        self.store = TPQStore(root)
        from .ops import ModelOperatorConfig

        self.operator_config = ModelOperatorConfig.from_manifest(
            {
                "model_family": (
                    self.store.man.model_family or "deepseek"
                ),
                "config": self.store.cfg,
            }
        )
        self.packed_operator_name: str | None = None
        if self.store.man.projection_vq:
            from .ops import packed_moe_operator_name

            capabilities = {
                tuple(
                    sorted(
                        self.store.man.projection_operator_capability(
                            layer
                        ).items()
                    )
                )
                for layer in self.store.man.expert_files
            }
            names = set()
            for capability_items in capabilities:
                capability = dict(capability_items)
                names.add(
                    packed_moe_operator_name(
                        device_type=self.device.type,
                        activation=(
                            self.operator_config.expert_activation
                        ),
                        top_k=self.operator_config.top_k,
                        **capability,
                    )
                )
            self.packed_operator_name = ",".join(sorted(names))
            print(
                "[tpq] 公共 packed MoE="
                f"{self.packed_operator_name}；"
                f"activation={self.operator_config.expert_activation}；"
                "projection_fused="
                f"{os.environ.get('TPQ_PROJECTION_FUSED', '1')}",
                flush=True,
            )
        self.cfg = self.store.cfg  # dict（DSV4Config.to_json）
        gpu = self.device.type != "cpu"
        self._packed_device_pool = False
        self._packed_full_gpu = False
        if self.store.man.projection_vq:
            if not gpu:
                from .store import PackedCpuExpertPool

                self.pool = PackedCpuExpertPool(
                    self.store,
                    budget_gb=cache_gb,
                )
            elif (
                self.tp_size == 1
                and os.environ.get("TPQ_PACKED_FULL_GPU", "0") != "1"
            ):
                from .kimi_hybrid import PackedHybridPool

                self.pool = PackedHybridPool(
                    self.store,
                    vram_cache_gb,
                    device=self.device,
                    ram_gb=cache_gb,
                )
                self._packed_device_pool = True
            else:
                from .kimi_experts import (
                    PackedExpertPool,
                    build_primary_dense_packed_plan,
                )

                plan = build_primary_dense_packed_plan(
                    self.store,
                    self.tp_size,
                )
                self.pool = PackedExpertPool(
                    self.store,
                    self.devices,
                    plan,
                    parallelism=(
                        "pipeline" if self.tp_size == 1 else "tensor"
                    ),
                )
                self._packed_device_pool = True
                self._packed_full_gpu = True
        else:
            if self.tp_size != 1:
                raise ValueError(
                    "legacy DeepSeek-V4 archives support only tp=1"
                )
            self.pool = ExpertPool(
                self.store,
                vram_cache_gb if gpu else cache_gb,
                device=str(self.device),
                ram_gb=cache_gb if gpu else 0.0,
            )
        # Benchmark/API diagnostics must distinguish a real all-rank packed
        # pool from the legacy or RAM paths.  Multi-card construction has no
        # silent fallback: preload failure aborts startup before this value can
        # be reported as a successful run.
        self.effective_tp_size = (
            self.tp_size if self._packed_full_gpu else 1
        )
        self.max_ctx = max_ctx
        self._w: dict[str, object] = {}
        self._layers: dict[int, dict] = {}
        self._dense_bf16 = _parse_dense_bf16()
        self._hc_decode_workspaces: dict[
            tuple[torch.device, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._hc_post_workspaces: dict[
            tuple[torch.device, int],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._prefetch_auto = True
        self._prev_ids: dict[int, list[int]] = {}   # 层 → 上一 token 路由专家（预取用）
        self._profile_enabled = False
        self._profile_records: list[
            tuple[int, tuple[tuple[str, object], ...]]
        ] = []
        self._moe_profile_records: list[
            tuple[int, tuple[tuple[str, object], ...]]
        ] = []
        self.last_layer_profile: dict[str, object] = {}
        self._cpu_resident_experts: dict[
            int, tuple[tuple[VQWeight, VQWeight] | None, ...]
        ] = {}
        self._cpu_moe_layers: dict[int, object] = {}
        self._tp_shared_mlp = None
        self._tp_router = None
        self._tp_attention_contexts: tuple[list[dict], ...] | None = None
        self._tp_route_weights: tuple[list[dict], ...] | None = None
        self._tp_route_buffers: dict[
            int,
            tuple[
                tuple[torch.Tensor, ...],
                tuple[torch.Tensor, ...],
                tuple[torch.Tensor, ...],
            ],
        ] = {}
        self._tp_states_ready = False
        self.tp_dataflow = "single"
        self.states: list[dict] | None = None
        c = self.cfg
        ratios = (list(c.get("compress_ratios") or []) + [0] * c["n_layers"])[: c["n_layers"]]
        self.ratios = ratios
        from .dsv4 import RopeCache  # 复用包内频率预计算（纯 torch 无依赖）
        rd = c["qk_rope_head_dim"]
        self.rope_base = RopeCache(rd, max_ctx + 8, c["rope_theta"], None)
        self.rope_cmp = RopeCache(rd, max_ctx + 8, c.get("compress_rope_theta", 160000.0),
                                  c.get("rope_scaling") or None)
        if gpu:
            for rc in (self.rope_base, self.rope_cmp):
                rc.cos = rc.cos.to(self.device)
                rc.sin = rc.sin.to(self.device)

    # ---- 权重访问 ----
    def w(self, name: str):
        wt = self._w.get(name)
        if wt is None:
            wt = self.store.get_dense(name)
            if self.device.type != "cpu":
                group = _dense_bf16_group(name)
                use_bf16 = group in self._dense_bf16
                if isinstance(wt, Int4Weight):
                    if use_bf16:
                        if compute_dtype(self.device) != torch.bfloat16:
                            raise RuntimeError(
                                "TPQ_DENSE_BF16 requires BF16 compute; "
                                "set TPQ_COMPUTE_DTYPE=bf16 on a supported GPU"
                            )
                        wt = wt.dequant_rows(0, wt.shape[0]).to(
                            self.device, dtype=torch.bfloat16
                        )
                    elif group == "hyper":
                        # HC consumes ``fn`` as a dense Tensor (and the fused
                        # path accepts FP32/BF16), not as an Int4Weight object.
                        # Keep the no-TPQ_DENSE_BF16 default functional by
                        # materializing this small matrix in FP32.
                        wt = wt.dequant_rows(0, wt.shape[0]).to(self.device)
                    elif name == "head.weight":
                        # lm_head 每 token 全量乘，常驻 f32（2.1GB，与 GLM 的 lm_head 策略一致）
                        wt = wt.dequant_rows(0, wt.shape[0]).to(self.device)
                    else:
                        # int4 dense GEMM 的 fp16 计算：默认关闭（TPQ_INT4_HALF=1 开启）。
                        # 实测 43 层残差+HC 放大使逐层 hidden rel 差达 1-3%（超 0.5% 门），
                        # 而内存受限卡上提速可忽略；KL 虽不变，从严回 f32。
                        wt = Int4Weight(wt.q.to(self.device), wt.s.to(self.device),
                                        wt.cols, wt.gs,
                                        half=os.environ.get("TPQ_INT4_HALF", "0") == "1")
                else:
                    wt = wt.to(
                        self.device, dtype=torch.bfloat16
                    ) if use_bf16 else wt.to(self.device)
            self._w[name] = wt
        return wt

    def _prefetch_enabled(self) -> bool:
        raw = os.environ.get("TPQ_PREFETCH", "auto").strip().lower()
        if raw in ("", "auto"):
            return self._prefetch_auto
        return raw not in ("0", "false", "off", "no")

    def layer(self, i: int) -> dict:
        """一层 dense 权重（attn/hc/norm/gate/compressor/shared），按键名惰性组装。"""
        w = self._layers.get(i)
        if w is not None:
            return w
        p = f"layers.{i}"
        w = {
            "wq_a": self.w(f"{p}.attn.wq_a.weight"),
            "q_norm": self.w(f"{p}.attn.q_norm.weight"),
            "wq_b": self.w(f"{p}.attn.wq_b.weight"),
            "wkv": self.w(f"{p}.attn.wkv.weight"),
            "kv_norm": self.w(f"{p}.attn.kv_norm.weight"),
            "attn_sink": self.w(f"{p}.attn.attn_sink"),
            "wo_a": self.w(f"{p}.attn.wo_a.weight"),
            "wo_b": self.w(f"{p}.attn.wo_b.weight"),
            "attn_norm": self.w(f"{p}.attn_norm.weight"),
            "ffn_norm": self.w(f"{p}.ffn_norm.weight"),
            "gate": (
                self.w(f"{p}.ffn.gate.weight")
                if self.device.type == "cpu"
                else _f32(self.w(f"{p}.ffn.gate.weight"))
            ),
            "sh_w1": self.w(f"{p}.ffn.shared_experts.w1.weight"),
            "sh_w3": self.w(f"{p}.ffn.shared_experts.w3.weight"),
            "sh_w2": self.w(f"{p}.ffn.shared_experts.w2.weight"),
            "hc_attn_fn": self.w(f"{p}.hc_attn_fn"),
            "hc_attn_base": self.w(f"{p}.hc_attn_base"),
            "hc_attn_scale": self.w(f"{p}.hc_attn_scale"),
            "hc_ffn_fn": self.w(f"{p}.hc_ffn_fn"),
            "hc_ffn_base": self.w(f"{p}.hc_ffn_base"),
            "hc_ffn_scale": self.w(f"{p}.hc_ffn_scale"),
        }
        if self.store.has(f"{p}.attn.compressor.wkv.weight"):
            w["cmp"] = {
                "wkv": self.w(f"{p}.attn.compressor.wkv.weight"),
                "wgate": self.w(f"{p}.attn.compressor.wgate.weight"),
                "ape": _f32(self.w(f"{p}.attn.compressor.ape")),  # 需下标切片，须 f32
                "norm": self.w(f"{p}.attn.compressor.norm.weight"),
            }
        if self.ratios[i] == 4:
            w["indexer"] = {
                "wq_b": self.w(f"{p}.attn.indexer.wq_b.weight"),
                "weights_proj": self.w(
                    f"{p}.attn.indexer.weights_proj.weight"
                ),
                "wkv": self.w(
                    f"{p}.attn.indexer.compressor.wkv.weight"
                ),
                "wgate": self.w(
                    f"{p}.attn.indexer.compressor.wgate.weight"
                ),
                "ape": _f32(
                    self.w(f"{p}.attn.indexer.compressor.ape")
                ),
                "norm": self.w(
                    f"{p}.attn.indexer.compressor.norm.weight"
                ),
            }
        if self.store.has(f"{p}.ffn.gate.bias"):
            w["gate_bias"] = self.w(f"{p}.ffn.gate.bias")
        if self.store.has(f"{p}.ffn.gate.tid2eid"):
            w["tid2eid"] = self.w(f"{p}.ffn.gate.tid2eid").long()
        self._layers[i] = w
        return w

    # ---- 前向（数值与 CCCP/dsv4.py 一致；线性层经 _linear 分派） ----
    def _rope(self, i: int):
        return self.rope_cmp if self.ratios[i] else self.rope_base

    def _alloc(self, B: int) -> None:
        self.states = self._allocate_states(B, self.device)

    def _allocate_states(
        self,
        B: int,
        device: torch.device,
    ) -> list[dict]:
        """Allocate one complete MQA/compressor state replica on a TP rank."""
        c = self.cfg
        win, hd = c["sliding_window"], c["head_dim"]
        hot_dtype = compute_dtype(device)
        states = []
        for i in range(c["n_layers"]):
            ratio = self.ratios[i]
            st = {
                # 现有 decode attention 核仍以 FP32 做局部 score/value；
                # SM120 BF16 sparse kernel 接入后再把窗口状态切回 hot_dtype。
                "kv": torch.zeros(
                    B, win, hd, device=device, dtype=torch.float32
                ),
                "win_pos": torch.full(
                    (B, win), -1, dtype=torch.long, device=device
                ),
            }
            if ratio:
                st["compressed"] = PagedKV(
                    batch=B,
                    page_items=max(1, 4096 // ratio),
                    dim=hd,
                    device=device,
                    dtype=compute_dtype(device),
                    max_items=(self.max_ctx + ratio - 1) // ratio,
                )
                if ratio == 4:
                    st["indexer"] = IndexerState(
                        batch=B,
                        head_dim=c.get("index_head_dim", 128),
                        rope_dim=c["qk_rope_head_dim"],
                        page_items=max(1, 4096 // ratio),
                        device=device,
                        dtype=compute_dtype(device),
                        max_items=(self.max_ctx + ratio - 1) // ratio,
                    )
                coff = 2 if ratio == 4 else 1
                st["ckv"] = torch.zeros(
                    B,
                    coff * ratio,
                    coff * hd,
                    device=device,
                    dtype=hot_dtype,
                )
                st["cscore"] = torch.full((B, coff * ratio, coff * hd), float("-inf"),
                                          device=device, dtype=torch.float32)
            states.append(st)
        return states

    def ensure_position(self, position: int) -> None:
        """Reserve every compressed-layer page before any token state is mutated."""
        if position < 0 or self.states is None:
            return
        try:
            for layer, ratio in enumerate(self.ratios):
                if ratio:
                    self.states[layer]["compressed"].reserve(position // ratio)
                    indexer = self.states[layer].get("indexer")
                    if indexer is not None:
                        indexer.reserve_position(position)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            raise ContextCapacityError(position, exc) from exc

    @torch.no_grad()
    def snapshot_kv(self) -> DSV4KVSnapshot:
        """Copy every in-place mutable field needed to restore one boundary."""
        if self.states is None:
            raise ValueError("cannot snapshot an empty DSV4 KV state")
        if len(self.states) != len(self.ratios):
            raise ValueError("DSV4 KV state/ratio layer count mismatch")

        layers = []
        for ratio, state in zip(self.ratios, self.states):
            indexer = state.get("indexer")
            layers.append(
                DSV4LayerKVSnapshot(
                    kv=state["kv"].clone(),
                    win_pos=state["win_pos"].clone(),
                    compressed_length=(
                        int(state["compressed"].length)
                        if ratio
                        else None
                    ),
                    ckv=state["ckv"].clone() if ratio else None,
                    cscore=(
                        state["cscore"].clone() if ratio else None
                    ),
                    indexer_length=(
                        int(indexer.keys.length)
                        if indexer is not None
                        else None
                    ),
                    indexer_ckv=(
                        indexer.ckv.clone()
                        if indexer is not None
                        else None
                    ),
                    indexer_cscore=(
                        indexer.cscore.clone()
                        if indexer is not None
                        else None
                    ),
                )
            )
        return DSV4KVSnapshot(
            pos=int(self.pos),
            layers=tuple(layers),
        )

    @torch.no_grad()
    def restore_kv(self, snapshot: DSV4KVSnapshot) -> None:
        """Atomically validate, then restore a stable DSV4 prompt boundary."""
        if self.states is None:
            raise ValueError("cannot restore into an empty DSV4 KV state")
        if (
            len(snapshot.layers) != len(self.states)
            or len(self.ratios) != len(self.states)
        ):
            raise ValueError("DSV4 KV snapshot layer count mismatch")

        def require_tensor(
            live: torch.Tensor,
            saved: torch.Tensor | None,
            label: str,
        ) -> None:
            if saved is None:
                raise ValueError(
                    f"DSV4 KV snapshot missing {label}"
                )
            if (
                live.shape != saved.shape
                or live.dtype != saved.dtype
                or live.device != saved.device
            ):
                raise ValueError(
                    f"DSV4 KV snapshot {label} mismatch"
                )

        # Validate every layer before copying any tensor.
        for ratio, state, saved in zip(
            self.ratios,
            self.states,
            snapshot.layers,
        ):
            require_tensor(state["kv"], saved.kv, "raw ring")
            require_tensor(
                state["win_pos"],
                saved.win_pos,
                "win_pos",
            )

            compressed = state.get("compressed")
            if ratio:
                if (
                    compressed is None
                    or saved.compressed_length is None
                    or saved.compressed_length < 0
                    or saved.compressed_length > compressed.length
                ):
                    raise ValueError(
                        "DSV4 KV snapshot compressed length mismatch"
                    )
                require_tensor(
                    state["ckv"],
                    saved.ckv,
                    "compressor ckv",
                )
                require_tensor(
                    state["cscore"],
                    saved.cscore,
                    "compressor cscore",
                )
            elif saved.compressed_length is not None:
                raise ValueError(
                    "DSV4 KV snapshot unexpected compressed state"
                )

            indexer = state.get("indexer")
            if ratio == 4:
                if (
                    indexer is None
                    or saved.indexer_length is None
                    or saved.indexer_length < 0
                    or saved.indexer_length > indexer.keys.length
                ):
                    raise ValueError(
                        "DSV4 KV snapshot Indexer length mismatch"
                    )
                require_tensor(
                    indexer.ckv,
                    saved.indexer_ckv,
                    "Indexer ckv",
                )
                require_tensor(
                    indexer.cscore,
                    saved.indexer_cscore,
                    "Indexer cscore",
                )
            elif saved.indexer_length is not None:
                raise ValueError(
                    "DSV4 KV snapshot unexpected Indexer state"
                )

        for state, saved in zip(self.states, snapshot.layers):
            state["kv"].copy_(saved.kv)
            state["win_pos"].copy_(saved.win_pos)
            if saved.compressed_length is not None:
                state["compressed"].truncate(
                    saved.compressed_length
                )
                state["ckv"].copy_(saved.ckv)
                state["cscore"].copy_(saved.cscore)
            if saved.indexer_length is not None:
                indexer = state["indexer"]
                indexer.keys.truncate(saved.indexer_length)
                indexer.ckv.copy_(saved.indexer_ckv)
                indexer.cscore.copy_(saved.indexer_cscore)

        self.pos = snapshot.pos
        self._spec = None
        self._prev_ids.clear()

    def reset(self) -> None:
        self.states = None
        if self._tp_attention_contexts is not None:
            for rank_contexts in self._tp_attention_contexts:
                for context in rank_contexts:
                    context["state"] = None
        self._tp_states_ready = False
        self.pos = 0
        self._spec = None
        self._prev_ids.clear()

    # ---- Engine 接口（与 TPQ GLMModel 同名：forward/forward_hidden/logits_of/reset_kv/pos） ----
    pos: int = 0

    DSPARK_TARGETS = (40, 41, 42)   # DSpark main_hidden 的取材层（hc 均值隐态）

    def reset_kv(self) -> None:
        self.reset()

    def preload(self) -> None:
        """GPU 路径：全部 dense 权重上显存（int4 打包态 + head f32 常驻）。"""
        if self.device.type == "cpu":
            print(
                f"[tpq] CPU 推理线程：{self._cpu_threads}",
                flush=True,
            )
            if self._cpu_numa_interleaved:
                print(
                    "[tpq] CPU NUMA：专家与 dense 内存跨节点交错分配",
                    flush=True,
                )
            # CPU 首次启动时提前编译/装载融合内核，避免首个 decode 卡顿。
            from .cpuext import prebuild as prebuild_cpu
            prebuild_cpu()
            # CPU 也需要真正的 RAM 模式；仅使用 LRU 会让每批新路由专家
            # 重复承担 zlib 解压和张量构造，即使机器还有数百 GiB 可用内存。
            resident_all = self.pool.preload_all()
            self._prefetch_auto = not resident_all
            if not resident_all:
                self.pool.preload_pinned()
            else:
                n_experts = self.cfg["n_experts"]
                for layer in range(self.cfg["n_layers"]):
                    experts = tuple(
                        self.pool.pinned.get((layer, expert))
                        for expert in range(n_experts)
                    )
                    if any(expert is not None for expert in experts):
                        self._cpu_resident_experts[layer] = experts
            return
        import time
        t0 = time.time()
        if self._dense_bf16:
            print("[tpq] dense BF16 常驻: "
                  + ",".join(sorted(self._dense_bf16)), flush=True)
        names = self.store.dense_names()
        for name in names:
            self.w(name)
        self._prepare_tp_shared_mlp()
        self._prepare_tp_decode_metadata()
        vram = torch.cuda.memory_allocated(self.device) / 2**30
        print(f"[tpq] dense 预载完成（{time.time() - t0:.1f}s，显存 {vram:.1f}GB）",
              flush=True)
        # 预热分组 GEMM / 融合 kernel（命中编译缓存 ~1s），避免首个 decode 卡顿。
        if os.environ.get("TPQ_GROUPED", "1") != "0":
            from . import grouped as _g
            print(f"[tpq] 分组GEMM: "
                  + ("fused CUDA kernel" if _g._fused is not None else "torch 批量路径"),
                  flush=True)
        if self._packed_full_gpu:
            self.pool.preload()
            self._prefetch_auto = False
            return
        resident_all = self.pool.preload_all()
        self._prefetch_auto = not resident_all
        if not resident_all:   # 内存够则全量常驻；不够（已警告）回退热钉住+LRU
            self.pool.preload_pinned()
        else:
            self.pool.pin_host_resident()
            if os.environ.get("TPQ_PREFETCH", "auto").strip().lower() in ("", "auto"):
                print("[tpq] 全量 RAM 常驻：自动关闭专家预测预取（避免 staging 竞争）",
                      flush=True)
        self.pool.build_gpu_arenas()

    def _prepare_tp_shared_mlp(self) -> None:
        """Shard every shared expert through the public gated-MLP TP op.

        This is intentionally capability driven: DSV4 only supplies separate
        Gate/Up weights and the clamped SwiGLU parameters.  Weight slicing,
        block-FP8 GEMV, activation and Row-TP reduction remain in ``tpq.ops``.
        """
        if (
            self.tp_size <= 1
            or not self.store.man.projection_vq
            or os.environ.get("TPQ_SHARED_MLP_TP", "1") == "0"
        ):
            return
        from .kernels import ProjectionGroup
        from .ops.tensor_parallel import (
            GatedMLPSpec,
            TensorParallelGatedMLP,
        )

        intermediate = int(self.layer(0)["sh_w1"].shape[0])
        executor = TensorParallelGatedMLP(
            self.devices,
            GatedMLPSpec(
                hidden_size=int(self.cfg["hidden"]),
                intermediate_size=intermediate,
                activation=self.operator_config.expert_activation,
                activation_beta=float(self.cfg.get("situ_beta", 4.0)),
                activation_linear_beta=self.cfg.get("situ_linear_beta"),
                activation_limit=float(self.cfg.get("swiglu_limit", 0.0)),
            ),
        )
        for layer in range(int(self.cfg["n_layers"])):
            weights = self.layer(layer)
            executor.add_layer(
                layer,
                0,
                ProjectionGroup(
                    (weights["sh_w1"], weights["sh_w3"])
                ),
                weights["sh_w2"],
            )
        executor.capture()
        self._tp_shared_mlp = executor
        print(
            "[tpq] 公共共享 Dense MLP Column/Row-TP Graph 完成："
            f"{self.cfg['n_layers']} 层×TP{self.tp_size}；"
            "FP8 分片常驻、每层一次 Row 规约",
            flush=True,
        )

    @staticmethod
    def _weight_to_device(weight, device: torch.device):
        if isinstance(weight, (BlockFP8Weight, Int4Weight)):
            return weight.to(device)
        if isinstance(weight, torch.Tensor):
            return weight.to(device)
        raise TypeError(f"unsupported TP weight {type(weight)!r}")

    def _prepare_tp_decode_metadata(self) -> None:
        """Prepare all-rank Head/Dense/MoE decode metadata.

        The model layer only describes the HC and compressed-MQA topology.
        Compact FP8 slicing, fixed TPHidden buffers, shared Dense TP and packed
        expert TP are all supplied by the public operator library.
        """
        if (
            self.tp_size <= 1
            or self._tp_shared_mlp is None
            or os.environ.get("TPQ_DSV4_FULL_TP", "1") == "0"
        ):
            return
        if os.environ.get("TPQ_TP_HIDDEN", "0") == "0":
            raise RuntimeError("DSV4 full TP requires TPQ_TP_HIDDEN=1")
        from .dsv4 import RopeCache
        from .ops import shard_linear_input, shard_linear_output

        base_cfg = self._cfg_obj()
        if (
            int(base_cfg.n_heads) % self.tp_size
            or int(base_cfg.o_groups) % self.tp_size
        ):
            raise ValueError("DSV4 heads/O groups must divide the TP width")
        attention_by_rank: list[list[dict]] = []
        route_by_rank: list[list[dict]] = []
        for rank, device in enumerate(self.devices):
            local_cfg = copy.copy(base_cfg)
            local_cfg.n_heads = int(base_cfg.n_heads) // self.tp_size
            local_cfg.o_groups = int(base_cfg.o_groups) // self.tp_size
            rope_base = RopeCache(
                int(base_cfg.qk_rope_head_dim),
                self.max_ctx + 8,
                float(base_cfg.rope_theta),
                None,
            )
            rope_cmp = RopeCache(
                int(base_cfg.qk_rope_head_dim),
                self.max_ctx + 8,
                float(getattr(base_cfg, "compress_rope_theta", 160000.0)),
                getattr(base_cfg, "rope_scaling", None),
            )
            for rope in (rope_base, rope_cmp):
                rope.cos = rope.cos.to(device)
                rope.sin = rope.sin.to(device)
            rank_attention = []
            rank_routes = []
            for layer in range(int(self.cfg["n_layers"])):
                source = self.layer(layer)
                weights = {
                    key: self._weight_to_device(source[key], device)
                    for key in ("wq_a", "q_norm", "wkv", "kv_norm")
                }
                weights["wq_b"] = shard_linear_output(
                    source["wq_b"], rank, self.tp_size, device
                )
                weights["attn_sink"] = (
                    source["attn_sink"]
                    .chunk(self.tp_size, dim=0)[rank]
                    .to(device)
                    .contiguous()
                )
                weights["wo_a"] = shard_linear_output(
                    source["wo_a"], rank, self.tp_size, device
                )
                weights["wo_b"] = shard_linear_input(
                    source["wo_b"], rank, self.tp_size, device
                )
                for nested in ("cmp", "indexer"):
                    if nested in source:
                        weights[nested] = {
                            key: self._weight_to_device(value, device)
                            for key, value in source[nested].items()
                        }
                rank_attention.append(
                    {
                        "cfg": local_cfg,
                        "weights": weights,
                        "state": None,
                        "rope": rope_cmp if self.ratios[layer] else rope_base,
                        "ratio": int(self.ratios[layer]),
                    }
                )
                route_item = {
                    key: self._weight_to_device(source[key], device)
                    for key in (
                        "hc_attn_fn",
                        "hc_attn_scale",
                        "hc_attn_base",
                        "attn_norm",
                        "hc_ffn_fn",
                        "hc_ffn_scale",
                        "hc_ffn_base",
                        "ffn_norm",
                    )
                }
                route_item["gate_bias"] = self._weight_to_device(
                    source.get(
                        "gate_bias",
                        torch.zeros(
                            int(self.cfg["n_experts"]),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                    ),
                    device,
                )
                route_item["mask"] = self.store.available_mask(layer).to(
                    device
                )
                if "tid2eid" in source:
                    route_item["tid2eid"] = source["tid2eid"].to(device)
                rank_routes.append(route_item)
            attention_by_rank.append(rank_attention)
            route_by_rank.append(rank_routes)
        self._tp_attention_contexts = tuple(attention_by_rank)
        self._tp_route_weights = tuple(route_by_rank)
        self.tp_dataflow = "all-rank-head-dense-packed"

        from .ops.tensor_parallel import (
            RowParallelLinearSpec,
            TensorParallelRowLinear,
        )

        router = TensorParallelRowLinear(
            self.devices,
            RowParallelLinearSpec(
                in_features=int(self.cfg["hidden"]),
                out_features=int(self.cfg["n_experts"]),
                input_dtype=torch.bfloat16,
                weight_dtype=torch.float32,
                output_dtype=torch.float32,
            ),
        )
        for layer in range(int(self.cfg["n_layers"])):
            router.add_layer(layer, 0, self.layer(layer)["gate"])
            router.bind_input_hidden(
                layer,
                self._tp_shared_mlp.input_hidden(layer),
            )
        router.capture()
        self._tp_router = router

        top_k = int(self.cfg["top_k"])
        experts = int(self.cfg["n_experts"])
        for layer in range(int(self.cfg["n_layers"])):
            logits = []
            route_weights = []
            route_ids = []
            for device in self.devices:
                logits.append(
                    torch.empty(1, experts, dtype=torch.float32, device=device)
                )
                route_weights.append(
                    torch.empty(1, top_k, dtype=torch.float32, device=device)
                )
                route_ids.append(
                    torch.empty(1, top_k, dtype=torch.long, device=device)
                )
            self._tp_route_buffers[layer] = (
                tuple(logits),
                tuple(route_weights),
                tuple(route_ids),
            )
            self.pool.bind_hidden_inputs(
                layer,
                self._tp_shared_mlp.input_hidden(layer),
                tuple(route_weights),
                tuple(route_ids),
            )
        print(
            "[tpq] DSV4 公共真 TP decode 元数据完成："
            f"Head-TP + shared Dense Column/Row-TP + packed MoE TP，"
            f"{self.cfg['n_layers']} 层×TP{self.tp_size}",
            flush=True,
        )

    def forward(self, ids: list[int]) -> torch.Tensor:
        """前向一段 token（prefill 或单步 decode），返回最后位置 logits [vocab]。"""
        t = torch.tensor([ids], device=self.device)
        if self.states is None:
            lg = self.prefill(t, full_logits=False)
            self.pos = len(ids)
            return lg.squeeze(0)
        out = []
        for i, tok in enumerate(ids):
            lg = self.decode(torch.tensor([tok], device=self.device), self.pos + i)
            out.append(lg)
        self.pos += len(ids)
        return out[-1].squeeze(0)

    def forward_hidden(self, ids: list[int]) -> torch.Tensor:
        """前向一段 token，返回全部位置的最终 hidden [T, hidden]（已过 final norm）。"""
        t = torch.tensor([ids], device=self.device)
        if self.states is not None and len(ids) == 1:
            raise RuntimeError("forward_hidden 增量模式未实现（投机解码暂未接入 DSV4）")
        from .dsv4 import hc_head
        cfg = self._cfg_obj()
        self._alloc(1)
        self.ensure_position(len(ids) - 1)
        h = self._embed(t).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        for i in range(cfg.n_layers):
            h = self._block(h, i, t, 0)
        y = hc_head(h, *self._hc_head_w(), cfg)
        y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
        self.pos = len(ids)
        return y.squeeze(0)   # [1,T,D] → [T,D]，与 GLMModel/评测脚本口径一致

    def logits_of(self, h: torch.Tensor) -> torch.Tensor:
        """hidden [N, hidden] → logits [N, vocab]。"""
        return _linear(h, self.w("head.weight")).float()

    def _cfg_obj(self):
        """把 dict 配置包装为 dsv4.py 函数期望的属性对象。"""
        if getattr(self, "_co", None) is None:
            from .cconfig import DSV4Config
            self._co = DSV4Config.from_json(self.cfg)
        return self._co

    def _expert_mlp_tpq(self, x, gu, dn, weights):
        """VQ 专家 MLP（数值同 dsv4.expert_mlp：up±10、gate≤10、silu(gate)*up）。"""
        limit = self.cfg.get("swiglu_limit", 0.0)
        mi = self.cfg["moe_inter"]
        h = gu.matmul_T(x)                       # [N, 2*mi]
        g, u = h[:, :mi], h[:, mi:]
        if limit:
            u = u.clamp(-limit, limit)
            g = g.clamp(max=limit)
        out = dn.matmul_T(F.silu(g) * u)
        return out * weights

    def _mask(self, layer: int) -> torch.Tensor:
        """该层可用专家布尔掩码（drop 为 False），缓存。"""
        m = getattr(self, "_masks", {}).get(layer)
        if m is None:
            if not hasattr(self, "_masks"):
                self._masks = {}
            m = self.store.available_mask(layer).to(self.device)
            self._masks[layer] = m
        return m

    def _route_tpq(self, xf: torch.Tensor, w: dict, cfg, ids: torch.Tensor, layer: int):
        """带 drop 掩码的 sqrtsoftplus 路由（数值同 dsv4.gate_route；丢弃专家不可选）。
        learned 层：choice 掩 -inf 后 top-k；hash 层（tid2eid 静态表）：坏槽用
        「未选中且可用」的最高分专家逐个递补。"""
        mask = self._mask(layer)
        gate = w["gate"]
        scores = F.softplus(
            _tpq_lin(
                xf,
                gate.float() if isinstance(gate, torch.Tensor) else gate,
            )
        ).sqrt()
        tid2eid = w.get("tid2eid")
        if tid2eid is not None:
            indices = tid2eid[ids].clone()
            bad = ~mask[indices]
            if bad.any():
                cand = scores.masked_fill(~mask[None, :], -1e30)
                top_cand = cand.topk(cfg.top_k * 2, dim=-1).indices
                for n, k in bad.nonzero().tolist():
                    for c in top_cand[n].tolist():
                        if not (indices[n] == c).any():
                            indices[n, k] = c
                            break
        else:
            fused = _route_post_fused(
                scores,
                w["gate_bias"].float(),
                mask,
                cfg.top_k,
            )
            if fused is not None:
                weights, indices = fused
                if cfg.norm_topk_prob:
                    weights = weights / (
                        weights.sum(dim=-1, keepdim=True) + 1e-20
                    )
                return weights * cfg.routed_scaling, indices
            choice = scores + w["gate_bias"].float()
            choice = choice.masked_fill(~mask[None, :], float("-inf"))
            indices = choice.topk(cfg.top_k, dim=-1).indices
        weights = scores.gather(1, indices)
        if cfg.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * cfg.routed_scaling, indices

    def _moe(
        self,
        x: torch.Tensor,
        layer: int,
        ids: torch.Tensor,
        *,
        return_parts: bool = False,
    ):
        from .dsv4 import expert_mlp
        c = self.cfg
        w = self.layer(layer)
        B, T, D = x.shape
        output_dtype = compute_dtype(x.device)
        x_rows = x.reshape(B * T, D)
        xf = x_rows.float()
        cfg_obj = self._cfg_obj()
        profile_moe = (
            self._profile_enabled and self.store.man.projection_vq
        )
        moe_events: list[tuple[str, object]] = []

        def mark_moe(name: str) -> None:
            if profile_moe:
                if x_rows.is_cuda:
                    event = torch.cuda.Event(enable_timing=True)
                    event.record(torch.cuda.current_stream(self.device))
                    moe_events.append((name, event))
                else:
                    moe_events.append((name, time.perf_counter()))

        mark_moe("start")
        resident = self._cpu_resident_experts.get(layer)
        cached_layer = None
        if B * T == 1 and not x_rows.is_cuda and resident is not None:
            shared_weights = (w["sh_w1"], w["sh_w3"], w["sh_w2"])
            gate = w["gate"]
            if (
                isinstance(gate, Int4Weight)
                and all(
                    isinstance(weight, Int4Weight)
                    for weight in shared_weights
                )
            ):
                cached_layer = self._cpu_moe_layers.get(layer)
                if cached_layer is None:
                    from .cpuext import make_moe_layer_cpu

                    w1, w3, w2 = shared_weights
                    gate_bias = w.get("gate_bias")
                    if gate_bias is None:
                        gate_bias = w.setdefault(
                            "_cpu_gate_bias",
                            torch.zeros(
                                c["n_experts"],
                                dtype=torch.float32,
                                device=x_rows.device,
                            ),
                        )
                    cached_layer = make_moe_layer_cpu(
                        resident,
                        w1.q,
                        w1.s,
                        w3.q,
                        w3.s,
                        w2.q,
                        w2.s,
                        gate.q,
                        gate.s,
                        gate_bias,
                        self._mask(layer),
                        w1.gs,
                        c.get("swiglu_limit", 0.0),
                        cfg_obj.top_k,
                        cfg_obj.norm_topk_prob,
                        cfg_obj.routed_scaling,
                    )
                    if cached_layer is not None:
                        self._cpu_moe_layers[layer] = cached_layer
                if cached_layer is not None and w.get("tid2eid") is None:
                    fused_cpu = cached_layer.forward_learned(x_rows)
                    return fused_cpu.view(B, T, D).to(output_dtype)
        weights, indices = self._route_tpq(
            xf,
            w,
            cfg_obj,
            ids.reshape(-1),
            layer,
        )
        mark_moe("route")
        if self.store.man.projection_vq:
            activation = self.operator_config.expert_activation
            limit = float(c.get("swiglu_limit", 0.0))
            shared = (
                self._tp_shared_mlp.run(layer, x_rows)
                if self._tp_shared_mlp is not None and B * T == 1
                else expert_mlp(
                    x_rows,
                    w["sh_w1"],
                    w["sh_w3"],
                    w["sh_w2"],
                    limit,
                )
            )
            mark_moe("shared")
            routed_rows = []
            for row in range(B * T):
                if self._packed_device_pool:
                    routed = self.pool.run(
                        layer,
                        x_rows[row : row + 1],
                        indices[row],
                        weights[row],
                        activation=activation,
                        activation_beta=float(c.get("situ_beta", 4.0)),
                        activation_linear_beta=c.get(
                            "situ_linear_beta"
                        ),
                        limit=limit,
                    )
                else:
                    selected = self.pool.get_many(
                        [
                            (layer, int(expert_id))
                            for expert_id in indices[row].tolist()
                        ]
                    )
                    experts = [
                        selected[(layer, int(expert_id))]
                        for expert_id in indices[row].tolist()
                    ]
                    from .ops import packed_moe_selected_topk

                    routed = packed_moe_selected_topk(
                        x_rows[row : row + 1].float(),
                        experts,
                        weights[row].float(),
                        activation=activation,
                        activation_beta=float(c.get("situ_beta", 4.0)),
                        activation_linear_beta=c.get(
                            "situ_linear_beta"
                        ),
                        limit=limit,
                    )
                    if routed is None:
                        raise RuntimeError(
                            "no public packed CPU MoE operator accepted "
                            f"DeepSeek-V4 layer {layer}"
                        )
                routed_rows.append(routed.reshape(-1))
            mark_moe("routed")
            # Full-resident packed experts have nothing to prefetch.  Pulling
            # route IDs back through ``tolist`` would otherwise insert one
            # GPU->CPU synchronization per layer and serialize the decode
            # launch queue.  RAM/LRU profiles retain the exact old behaviour.
            if not self._packed_full_gpu and self._prefetch_enabled():
                self._prev_ids[layer] = [
                    int(expert_id) for expert_id in indices[-1].tolist()
                ]
            if return_parts and B * T == 1:
                mark_moe("merge")
                if profile_moe:
                    self._moe_profile_records.append(
                        (layer, tuple(moe_events))
                    )
                return (
                    routed_rows[0].view(1, D),
                    shared.reshape(1, D),
                )
            routed = (
                routed_rows[0].view(1, D)
                if len(routed_rows) == 1
                else torch.stack(routed_rows)
            ).to(shared.dtype)
            output = (routed + shared).view(B, T, D).to(output_dtype)
            mark_moe("merge")
            if profile_moe:
                self._moe_profile_records.append(
                    (layer, tuple(moe_events))
                )
            return output
        if cached_layer is not None:
            fused_cpu = cached_layer.forward(
                x_rows,
                weights[0],
                indices[0],
            )
            return fused_cpu.view(B, T, D).to(output_dtype)
        # T=1 解码热路径：把 top-k 专家堆叠成批做分组 GEMM（每层 launch 数 ÷6，
        # 数值与下方逐专家循环等价，见 tpq/grouped.py 自检）。
        # 全部选中专家均为 VQWeight 且未设 TPQ_GROUPED=0 时启用，否则回退原循环。
        if B * T == 1 and os.environ.get("TPQ_GROUPED", "1") != "0":
            from .grouped import moe_mlp_grouped_mixed
            # 先发射共享专家 GEMM（不依赖路由结果），CPU 等 indices DtoH 期间 GPU 有活干
            shared = None
            if x_rows.is_cuda:
                shared = expert_mlp(
                    x_rows,
                    w["sh_w1"],
                    w["sh_w3"],
                    w["sh_w2"],
                    c.get("swiglu_limit", 0.0),
                )
            eids = indices[0].tolist()
            self._prev_ids[layer] = eids
            if resident is not None:
                elist = [resident[e] for e in eids]
                if any(expert is None for expert in elist):
                    got = self.pool.get_many([(layer, e) for e in eids])
                    elist = [got[(layer, e)] for e in eids]
            else:
                got = self.pool.get_many([(layer, e) for e in eids])
                elist = [got[(layer, e)] for e in eids]
            if all(isinstance(g, VQWeight) and isinstance(d, VQWeight) for g, d in elist):
                shared_weights = (w["sh_w1"], w["sh_w3"], w["sh_w2"])
                if (
                    not x_rows.is_cuda
                    and all(
                        isinstance(weight, Int4Weight)
                        for weight in shared_weights
                    )
                ):
                    from .cpuext import moe_mixed_cpu

                    w1, w3, w2 = shared_weights
                    fused_cpu = moe_mixed_cpu(
                        x_rows,
                        [gu.idx for gu, _ in elist],
                        [gu.cb for gu, _ in elist],
                        [dn.idx for _, dn in elist],
                        [dn.cb for _, dn in elist],
                        weights[0],
                        w1.q,
                        w1.s,
                        w3.q,
                        w3.s,
                        w2.q,
                        w2.s,
                        w1.gs,
                        c.get("swiglu_limit", 0.0),
                    )
                    if fused_cpu is not None:
                        return fused_cpu.view(B, T, D).to(output_dtype)
                if shared is None:
                    shared = expert_mlp(
                        x_rows,
                        w["sh_w1"],
                        w["sh_w3"],
                        w["sh_w2"],
                        c.get("swiglu_limit", 0.0),
                    )
                y = moe_mlp_grouped_mixed(x_rows, elist, weights[0],
                                          c.get("swiglu_limit", 0.0)).unsqueeze(0)
                y += shared
                return y.view(B, T, D).to(output_dtype)
        y = torch.zeros_like(xf)
        # 分派：argsort + searchsorted 按专家分段（每层一次 DtoH 同步），替代逐专家
        # nonzero（每个都隐式同步，投机验证 T=6 时 ~1200 次/轮、WDDM 下约 0.6s）。
        # 专家遍历顺序仍为升序，与 indices.unique() 一致 → 数值不变。
        K = indices.shape[1]
        flat = indices.reshape(-1)
        order = torch.argsort(flat)
        bounds = torch.searchsorted(flat[order],
                                    torch.arange(c["n_experts"] + 1, device=flat.device))
        bl = bounds.tolist()
        present = [e for e in range(c["n_experts"]) if bl[e + 1] > bl[e]]
        self._prev_ids[layer] = present
        rows_all = torch.div(order, K, rounding_mode="floor")
        cols_all = order % K
        experts = self.pool.get_many([(layer, e) for e in present])
        from .grouped import expert_mlp_batched
        limit = c.get("swiglu_limit", 0.0)
        for e in present:
            sl = slice(bl[e], bl[e + 1])
            rows, cols = rows_all[sl], cols_all[sl]
            gu, dn = experts[(layer, e)]
            if isinstance(gu, VQWeight) and isinstance(dn, VQWeight):
                # 投机验证 T>1：idx 广播走融合 kernel（一次调用替代逐 token LUT 循环）
                y[rows] += expert_mlp_batched(xf[rows], gu, dn, limit) \
                    * weights[rows, cols, None]
            else:
                y[rows] += self._expert_mlp_tpq(xf[rows], gu, dn,
                                               weights[rows, cols, None])
        y += expert_mlp(xf, w["sh_w1"], w["sh_w3"], w["sh_w2"],
                        c.get("swiglu_limit", 0.0))
        return y.view(B, T, D).to(output_dtype)

    def _hc_decode_workspace(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return one stream-ordered HC workspace for batch-1 CUDA decode."""
        if (
            not hidden.is_cuda
            or hidden.dtype != torch.bfloat16
            or hidden.shape[0] * hidden.shape[1] != 1
        ):
            return None
        width = int(hidden.shape[-1])
        key = (hidden.device, width)
        buffers = self._hc_decode_workspaces.get(key)
        if buffers is None:
            options = {"dtype": hidden.dtype, "device": hidden.device}
            buffers = (
                torch.empty((1, width), **options),
                torch.empty((1, 4), **options),
                torch.empty((1, 16), **options),
            )
            self._hc_decode_workspaces[key] = buffers
        return buffers

    def _hc_post_workspace(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return distinct Attention/FFN HC result buffers for decode."""
        if (
            not hidden.is_cuda
            or hidden.dtype != torch.bfloat16
            or hidden.shape[0] * hidden.shape[1] != 1
        ):
            return None
        width = int(hidden.shape[-1])
        key = (hidden.device, width)
        buffers = self._hc_post_workspaces.get(key)
        if buffers is None:
            shape = (1, 1, 4, width)
            buffers = (
                torch.empty(shape, dtype=hidden.dtype, device=hidden.device),
                torch.empty(shape, dtype=hidden.dtype, device=hidden.device),
            )
            self._hc_post_workspaces[key] = buffers
        return buffers

    def _block(self, h: torch.Tensor, layer: int, ids: torch.Tensor, pos0: int,
               spec: dict | None = None) -> torch.Tensor:
        from .dsv4 import hc_post
        cfg = self._cfg_obj()
        w = self.layer(layer)
        st = self.states[layer]
        profile = self._profile_enabled
        events: list[tuple[str, object]] = []

        def mark(name: str) -> None:
            if profile:
                if h.is_cuda:
                    event = torch.cuda.Event(enable_timing=True)
                    event.record(torch.cuda.current_stream(self.device))
                    events.append((name, event))
                else:
                    events.append((name, time.perf_counter()))

        mark("start")
        hc_workspace = self._hc_decode_workspace(h)
        hc_post_workspace = self._hc_post_workspace(h)
        # 跨层专家预取（B2）：用上一 token 本层路由结果提前装填（时序局部性），
        # attention 计算与专家 读盘/DMA 重叠；未命中回退正常加载，无正确性风险
        prev = self._prev_ids.get(layer)
        if prev and ids.shape[-1] == 1 and self._prefetch_enabled():
            self.pool.prefetch([(layer, e) for e in prev])
        residual = h
        y, post, comb = _hc_pre_norm_tpq(
            h,
            w["hc_attn_fn"],
            w["hc_attn_scale"],
            w["hc_attn_base"],
            w["attn_norm"],
            cfg,
            output_buffers=hc_workspace,
        )
        mark("attn_hc_norm")
        a = self._attn_batch(y, layer, pos0, spec)
        mark("attention")
        h = hc_post(
            a,
            residual,
            post,
            comb,
            output=(
                None if hc_post_workspace is None else hc_post_workspace[0]
            ),
        )
        residual = h
        y, post, comb = _hc_pre_norm_tpq(
            h,
            w["hc_ffn_fn"],
            w["hc_ffn_scale"],
            w["hc_ffn_base"],
            w["ffn_norm"],
            cfg,
            output_buffers=hc_workspace,
        )
        mark("ffn_hc_norm")
        moe_result = self._moe(
            y,
            layer,
            ids,
            return_parts=(
                y.is_cuda
                and y.shape[0] * y.shape[1] == 1
                and self.store.man.projection_vq
            ),
        )
        mark("moe")
        if isinstance(moe_result, tuple):
            routed, shared = moe_result
            output = _hyper_connection_post_moe(
                routed,
                shared,
                residual,
                post,
                comb,
                output=(
                    None
                    if hc_post_workspace is None
                    else hc_post_workspace[1]
                ),
            )
            if output is None:
                combined = (
                    routed.to(shared.dtype) + shared
                ).view(1, 1, -1)
                output = hc_post(
                    combined,
                    residual,
                    post,
                    comb,
                    output=(
                        None
                        if hc_post_workspace is None
                        else hc_post_workspace[1]
                    ),
                )
        else:
            output = hc_post(
                moe_result,
                residual,
                post,
                comb,
                output=(
                    None
                    if hc_post_workspace is None
                    else hc_post_workspace[1]
                ),
            )
        output = output.to(compute_dtype(h.device))
        mark("ffn_hc_post")
        if profile:
            self._profile_records.append((layer, tuple(events)))
        return output

    def start_profile(self) -> None:
        """Start one-token CPU/CUDA stage profiling for the benchmark CLI."""
        self._profile_records = []
        self._moe_profile_records = []
        self.last_layer_profile = {}
        if self.device.type == "cpu" and self.store.man.projection_vq:
            from .cpuext import (
                reset_block_fp8_gemv_profile,
                reset_three_projection_phase_profile,
            )

            reset_three_projection_phase_profile()
            reset_block_fp8_gemv_profile()
        self._profile_enabled = True

    def finish_profile(self) -> dict[str, object]:
        """Finish a CLI stage probe and aggregate primary-stream events."""
        self._profile_enabled = False
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        def elapsed_ms(events: tuple, index: int) -> float:
            current = events[index][1]
            following = events[index + 1][1]
            if self.device.type == "cuda":
                return float(current.elapsed_time(following))
            return (float(following) - float(current)) * 1000.0
        stage_names = (
            "attn_hc_norm",
            "attention",
            "ffn_hc_norm",
            "moe",
            "ffn_hc_post",
        )
        totals = {name: 0.0 for name in stage_names}
        layers: list[dict[str, float | int]] = []
        for layer, events in self._profile_records:
            values: dict[str, float | int] = {"layer": layer}
            for index, name in enumerate(stage_names):
                milliseconds = elapsed_ms(events, index)
                values[f"{name}_ms"] = milliseconds
                totals[name] += milliseconds
            values["total_ms"] = sum(
                float(values[f"{name}_ms"])
                for name in stage_names
            )
            layers.append(values)
        top_layers = sorted(
            layers,
            key=lambda item: float(item["total_ms"]),
            reverse=True,
        )[:8]
        moe_stage_names = ("route", "shared", "routed", "merge")
        moe_totals = {name: 0.0 for name in moe_stage_names}
        moe_layers: list[dict[str, float | int]] = []
        for layer, events in self._moe_profile_records:
            values: dict[str, float | int] = {"layer": layer}
            for index, name in enumerate(moe_stage_names):
                milliseconds = elapsed_ms(events, index)
                values[f"{name}_ms"] = milliseconds
                moe_totals[name] += milliseconds
            moe_layers.append(values)
        result: dict[str, object] = {
            "layer_count": len(layers),
            "totals_ms": totals,
            "covered_ms": sum(totals.values()),
            "top_layers": top_layers,
            "layers": layers,
            "moe_totals_ms": moe_totals,
            "moe_layers": moe_layers,
        }
        if self.device.type == "cpu" and self.store.man.projection_vq:
            from .cpuext import (
                block_fp8_gemv_profile,
                three_projection_phase_profile,
            )

            result["packed_three_projection"] = (
                three_projection_phase_profile()
            )
            result["block_fp8_gemv"] = block_fp8_gemv_profile()
        self.last_layer_profile = result
        self._profile_records = []
        self._moe_profile_records = []
        return result

    def _attn_batch(
        self,
        y: torch.Tensor,
        layer: int,
        pos0: int,
        spec: dict | None,
        *,
        tp_context: dict | None = None,
    ) -> torch.Tensor:
        """批量增量注意力（投机验证用）：T 个 token（positions pos0..pos0+T-1）一次前向。

        数学与 CCCP.dsv4.attn_decode 逐步等价：环形窗自因果掩码（含窗约束）、
        压缩槽可见性 n < (qpos+1)//ratio、sink 在分母、输出末 64 维反旋转。
        Compressor 为每 token 顺序状态机（逐 token 调用 compressor_decode），
        并把每步的 ckv/cscore 快照记入 spec["steps"]（供 spec_commit 回滚）。
        """
        from . import dsv4 as _d
        from .dsv4 import rope_apply
        from .dsv4indexer import (
            hadamard_rotate,
            indexer_scores,
            select_index_positions,
        )

        cfg = (
            tp_context["cfg"]
            if tp_context is not None
            else self._cfg_obj()
        )
        w = (
            tp_context["weights"]
            if tp_context is not None
            else self.layer(layer)
        )
        st = (
            tp_context["state"]
            if tp_context is not None
            else self.states[layer]
        )
        cache = (
            tp_context["rope"]
            if tp_context is not None
            else self._rope(layer)
        )
        ratio = (
            int(tp_context["ratio"])
            if tp_context is not None
            else self.ratios[layer]
        )
        H, hd, rd = cfg.n_heads, cfg.head_dim, cfg.qk_rope_head_dim
        win = cfg.sliding_window
        B, T, _ = y.shape
        dev = y.device
        single_main_decode = (
            T == 1
            and spec is None
            and os.environ.get("TPQ_SINGLE_TOKEN_ATTN_FAST", "1") != "0"
        )
        qr, q, kv = _qkv_tpq(y, w, cfg, cache, pos0)
        scale = hd ** -0.5
        poss = (
            None
            if single_main_decode
            else torch.arange(pos0, pos0 + T, device=dev)
        )

        if ratio:
            steps = (
                spec.setdefault("steps", {}).setdefault(layer, [])
                if spec is not None
                else None
            )
            for j in range(T):
                p = pos0 + j
                rope_pos = max(0, p + 1 - ratio)
                cos = cache.cos[rope_pos].view(1, 1, -1)
                sin = cache.sin[rope_pos].view(1, 1, -1)
                ck = _compressor_decode_tpq(
                    y[:, j:j + 1],
                    w["cmp"],
                    ratio,
                    hd,
                    rd,
                    cos,
                    sin,
                    cfg.rms_eps,
                    st,
                    p,
                )
                if ck is not None:
                    st["compressed"].write(p // ratio, ck[:, 0])
                if steps is not None:
                    steps.append(
                        (st["ckv"].clone(), st["cscore"].clone())
                    )

        compressed_count = st["compressed"].length if ratio else 0
        direct_compressed_prefix = False
        if ratio == 4:
            indexer = st["indexer"]
            iw = w["indexer"]
            index_steps = (
                spec.setdefault("indexer_steps", {}).setdefault(layer, [])
                if spec is not None
                else None
            )
            for j in range(T):
                p = pos0 + j
                rope_pos = max(0, p + 1 - ratio)
                index_pooled = _compressor_decode_tpq(
                    y[:, j:j + 1],
                    iw,
                    indexer.ratio,
                    indexer.head_dim,
                    indexer.rope_dim,
                    cache.cos[rope_pos].view(1, 1, -1),
                    cache.sin[rope_pos].view(1, 1, -1),
                    cfg.rms_eps,
                    indexer.compressor_state,
                    p,
                )
                if index_pooled is not None:
                    index_pooled = hadamard_rotate(index_pooled)
                    indexer.keys.write(
                        p // indexer.ratio,
                        index_pooled[:, 0],
                    )
                if index_steps is not None:
                    index_steps.append(
                        (indexer.ckv.clone(), indexer.cscore.clone())
                    )
            if indexer.keys.length != compressed_count:
                raise RuntimeError(
                    "Indexer/main compressed KV length mismatch: "
                    f"{indexer.keys.length} != {compressed_count}"
                )
            visible_count = (
                compressed_count
                if single_main_decode
                else (poss + 1) // ratio
            )
            if compressed_count <= cfg.index_topk:
                direct_compressed_prefix = (
                    single_main_decode
                    and compressed_count <= st["compressed"].page_items
                    and os.environ.get("TPQ_DIRECT_KV_PREFIX", "1") != "0"
                )
                if direct_compressed_prefix:
                    selected_positions = None
                    selected_valid = None
                else:
                    selected_positions = torch.arange(
                        compressed_count, device=dev, dtype=torch.long
                    ).view(1, 1, -1).expand(B, T, -1)
                    selected_valid = (
                        selected_positions < visible_count.view(1, T, 1)
                    )
            else:
                iq = _linear(qr, iw["wq_b"]).view(
                    B, T, cfg.index_n_heads, cfg.index_head_dim
                )
                cos = cache.cos[pos0:pos0 + T].view(1, T, 1, -1)
                sin = cache.sin[pos0:pos0 + T].view(1, T, 1, -1)
                iq[..., cfg.index_head_dim - rd:] = rope_apply(
                    iq[..., cfg.index_head_dim - rd:], cos, sin
                )
                iq = hadamard_rotate(iq.to(compute_dtype(dev)))
                if compressed_count <= indexer.keys.page_items:
                    all_index_keys = indexer.keys.contiguous_prefix(
                        compressed_count
                    )
                else:
                    all_index_keys = indexer.keys.gather(
                        torch.arange(
                            compressed_count,
                            device=dev,
                            dtype=torch.long,
                        )
                    )
                index_weights = _linear(y, iw["weights_proj"]) * (
                    cfg.index_head_dim ** -0.5
                    * cfg.index_n_heads ** -0.5
                )
                selection_scores = indexer_scores(
                    iq, all_index_keys, index_weights
                )
                if not single_main_decode:
                    candidate_positions = torch.arange(
                        compressed_count, device=dev
                    )
                    selection_scores = selection_scores.masked_fill(
                        candidate_positions.view(1, 1, -1)
                        >= visible_count.view(1, T, 1),
                        float("-inf"),
                    )
                selected_positions = select_index_positions(
                    selection_scores, cfg.index_topk
                ).long()
                selected_valid = (
                    None
                    if single_main_decode
                    else selected_positions
                    < visible_count.view(1, T, 1)
                )
        elif compressed_count:
            direct_compressed_prefix = (
                single_main_decode
                and compressed_count <= st["compressed"].page_items
                and os.environ.get("TPQ_DIRECT_KV_PREFIX", "1") != "0"
            )
            if direct_compressed_prefix:
                selected_positions = None
                selected_valid = None
            else:
                selected_positions = torch.arange(
                    compressed_count, device=dev, dtype=torch.long
                ).view(1, 1, -1).expand(B, T, -1)
                selected_valid = (
                    None
                    if single_main_decode
                    else selected_positions < (
                        (poss + 1) // ratio
                    ).view(1, T, 1)
                )
        else:
            if single_main_decode:
                selected_positions = None
                selected_valid = None
            else:
                selected_positions = torch.empty(
                    B, T, 0, dtype=torch.long, device=dev
                )
                selected_valid = torch.empty(
                    B, T, 0, dtype=torch.bool, device=dev
                )

        if direct_compressed_prefix:
            selected_values = (
                st["compressed"]
                .contiguous_prefix(compressed_count)
                .unsqueeze(1)
                .to(q.dtype)
            )
        elif selected_positions is not None and selected_positions.numel():
            selected_values = st["compressed"].gather_batched(
                selected_positions.clamp_min(0)
            ).to(q.dtype)
        elif single_main_decode:
            selected_values = st["kv"][:, :0].unsqueeze(1)
        else:
            selected_values = torch.empty(
                B, T, 0, hd, device=dev, dtype=q.dtype
            )

        # Batched verification needs the pre-commit ring plus this chunk exactly
        # once.  Building this view after the commit duplicates current keys.
        if T > 1:
            raw_values = torch.cat([st["kv"], kv], dim=1)
            raw_positions = torch.cat(
                [st["win_pos"], poss.view(1, T).expand(B, T)],
                dim=1,
            )

        # Commit this chunk so fused T=1 decode and subsequent calls see it.
        if single_main_decode:
            slot = pos0 % win
            st["kv"][:, slot] = kv[:, 0]
            st["win_pos"][:, slot] = pos0
        else:
            recent = min(T, win)
            slots = poss[-recent:] % win
            st["kv"][:, slots] = kv[:, -recent:]
            st["win_pos"][:, slots] = poss[-recent:]

        out_cos = cache.cos[pos0:pos0 + T].view(1, T, 1, -1)
        out_sin = cache.sin[pos0:pos0 + T].view(1, T, 1, -1)
        if T == 1 and dev.type == "cpu":
            from .cpuext import attention_decode_cpu

            fused_cpu = attention_decode_cpu(
                q[:, 0],
                st["kv"],
                st["win_pos"],
                selected_values[:, 0],
                w["attn_sink"],
                out_cos,
                out_sin,
                scale,
            )
            if fused_cpu is not None:
                return _d._o_proj_hook(
                    fused_cpu.unsqueeze(1).flatten(2), w, cfg
                )
        if T == 1 and q.is_cuda:
            from .ops import attention_step

            fused = attention_step(
                "sliding_compressed_mqa_decode",
                q.device.type,
                query=q[:, 0],
                window_kv=st["kv"],
                window_positions=st["win_pos"],
                compressed_kv=selected_values[:, 0],
                sink=w["attn_sink"],
                cos=out_cos,
                sin=out_sin,
                scale=scale,
            )
            if fused is not None:
                return _d._o_proj_hook(
                    fused.unsqueeze(1).flatten(2), w, cfg
                )

        if selected_valid is None:
            selected_valid = torch.ones(
                B,
                1,
                selected_values.shape[2],
                device=dev,
                dtype=torch.bool,
            )
        if poss is None:
            poss = torch.arange(pos0, pos0 + T, device=dev)

        # The T=1 fused path above reads the committed ring directly.  If it
        # cannot run, the fallback must use that same ring without appending
        # the just-committed token a second time.
        if T == 1:
            raw_values = st["kv"]
            raw_positions = st["win_pos"]
        raw_scores = torch.einsum(
            "bthd,bsd->bhts", q * scale, raw_values
        )
        raw_allow = (
            (raw_positions.unsqueeze(1) >= 0)
            & (raw_positions.unsqueeze(1) <= poss.view(1, T, 1))
            & (raw_positions.unsqueeze(1) > poss.view(1, T, 1) - win)
        )
        raw_scores = raw_scores.masked_fill(
            ~raw_allow.unsqueeze(1), float("-inf")
        )
        if selected_values.shape[2]:
            compressed_scores = torch.einsum(
                "bthd,btkd->bhtk", q * scale, selected_values
            ).masked_fill(~selected_valid.unsqueeze(1), float("-inf"))
        else:
            compressed_scores = torch.empty(
                B, H, T, 0, device=dev, dtype=q.dtype
            )

        scores = torch.cat([raw_scores, compressed_scores], dim=-1).float()
        m = scores.amax(dim=-1)
        e = (scores - m.unsqueeze(-1)).exp()
        denom = e.sum(dim=-1) + (w["attn_sink"].view(1, -1, 1) - m).exp()
        probs = (
            e / denom.unsqueeze(-1)
        ).to(raw_values.dtype)
        raw_width = raw_values.shape[1]
        o = torch.einsum(
            "bhts,bsd->bthd", probs[..., :raw_width], raw_values
        )
        if selected_values.shape[2]:
            o += torch.einsum(
                "bhtk,btkd->bthd", probs[..., raw_width:], selected_values
            )
        o[..., hd - rd:] = rope_apply(
            o[..., hd - rd:], out_cos, out_sin, inverse=True
        )
        return _d._o_proj_hook(o.flatten(2), w, cfg)

    def _spec_snapshot(self, pos0: int, T: int) -> dict:
        """验证前快照各层将被触碰的状态：环槽（值+win_pos）、压缩槽、
        以及 compressor 每步状态容器（在 _attn_batch 中逐步填充）。"""
        cfg = self._cfg_obj()
        win = cfg.sliding_window
        poss = torch.arange(pos0, pos0 + T, device=self.device)
        spec = {
            "pos0": pos0,
            "T": T,
            "pre": [],
            "steps": {},
            "indexer_steps": {},
        }
        for i in range(cfg.n_layers):
            st = self.states[i]
            slots = poss % win
            pre = {"slots": slots.clone(),
                   "kv": st["kv"][:, slots].clone(),
                   "win_pos": st["win_pos"][:, slots].clone()}
            ratio = self.ratios[i]
            if ratio:
                cn = torch.unique(poss // ratio)
                pre["cslots"] = cn
                pre["compressed_length"] = st["compressed"].length
                pre["compressed_values"] = st["compressed"].gather(cn).clone()
                spec["steps"][i] = []
                if ratio == 4:
                    indexer = st["indexer"]
                    pre["indexer_length"] = indexer.keys.length
                    pre["indexer_values"] = indexer.keys.gather(cn).clone()
                    spec["indexer_steps"][i] = []
            spec["pre"].append(pre)
        return spec

    def spec_commit(self, keep: int) -> None:
        """验证后按接受前缀截断：恢复被拒位置的环槽/压缩槽，compressor 状态
        回滚到「处理完 position keep-1」的快照，model.pos = keep。"""
        spec = getattr(self, "_spec", None)
        assert spec is not None, "spec_commit 前须先 forward_verify"
        pos0, T = spec["pos0"], spec["T"]
        cfg = self._cfg_obj()
        win = cfg.sliding_window
        a = keep - pos0 - 1                     # 最末保留 token 的批内下标
        assert -1 <= a < T
        for i in range(cfg.n_layers):
            st = self.states[i]
            pre = spec["pre"][i]
            for j in range(a + 1, T):           # 被拒位置：恢复旧环槽内容
                st["kv"][:, pre["slots"][j]] = pre["kv"][:, j]
                st["win_pos"][:, pre["slots"][j]] = pre["win_pos"][:, j]
            if self.ratios[i]:
                ratio = self.ratios[i]
                for k, n in enumerate(pre["cslots"].tolist()):
                    if (n + 1) * ratio - 1 >= keep:     # 池化完成于被拒位置 → 恢复
                        st["compressed"].write(
                            n, pre["compressed_values"][:, k]
                        )
                target_length = max(pre["compressed_length"], keep // ratio)
                st["compressed"].truncate(target_length)
                ckv, cscore = spec["steps"][i][a]
                st["ckv"].copy_(ckv)
                st["cscore"].copy_(cscore)
                if ratio == 4:
                    indexer = st["indexer"]
                    for k, n in enumerate(pre["cslots"].tolist()):
                        if (n + 1) * ratio - 1 >= keep:
                            indexer.keys.write(
                                n, pre["indexer_values"][:, k]
                            )
                    indexer_length = max(
                        pre["indexer_length"], keep // ratio
                    )
                    indexer.keys.truncate(indexer_length)
                    index_ckv, index_cscore = spec["indexer_steps"][i][a]
                    indexer.ckv.copy_(index_ckv)
                    indexer.cscore.copy_(index_cscore)
        self._spec = None
        self.pos = keep

    @torch.no_grad()
    def forward_verify(self, ids_list: list[int], pos0: int) -> tuple[torch.Tensor, torch.Tensor]:
        """投机验证：一次批量前向处理 [t1, d1..dk]（positions pos0..pos0+T-1）。

        返回 (logits [T, vocab], main_hidden [T, 3·hidden])；KV 状态前进到
        pos0+T（随后由 spec_commit(keep) 截断到接受前缀）。main_hidden 为
        DSPARK_TARGETS 各层 hc 均值隐态的拼接（供 DSpark 草稿头）。
        """
        from .dsv4 import hc_head
        cfg = self._cfg_obj()
        assert self.states is not None and pos0 > 0, "forward_verify 前须先 prefill"
        T = len(ids_list)
        ids = torch.tensor([ids_list], device=self.device).long()
        self.ensure_position(pos0 + T - 1)
        if self._prev_ids:
            for l, es in self._prev_ids.items():
                self.pool.prefetch([(l, e) for e in es])
        self._spec = self._spec_snapshot(pos0, T)
        h = self._embed(ids).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        mh = []
        for i in range(cfg.n_layers):
            h = self._block(h, i, ids, pos0, self._spec)
            if i in self.DSPARK_TARGETS:
                mh.append(h.mean(dim=2))
        y = hc_head(h, *self._hc_head_w(), cfg)
        y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
        logits = _linear(y, self.w("head.weight")).float()
        return logits[0], torch.cat(mh, dim=-1)[0]

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.w("embed.weight")
        if isinstance(emb, Int4Weight):
            e = torch.stack([emb.row(int(i)) for i in ids.reshape(-1)])
            return e.view(*ids.shape, -1).to(compute_dtype(self.device))
        return emb[ids]

    @torch.no_grad()
    def prefill_chunked(
        self,
        ids: torch.Tensor,
        chunk_size: int = 512,
        capture_mh: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build long-context state without a sequence-squared attention tensor."""
        from .dsv4 import hc_head

        cfg = self._cfg_obj()
        ids = ids.to(self.device).long()
        B, T = ids.shape
        self._alloc(B)
        main_hidden_parts = [] if capture_mh else None
        final_y = None
        for start, end in _prefill_ranges(T, chunk_size):
            self.ensure_position(end - 1)
            chunk_ids = ids[:, start:end]
            h = self._embed(chunk_ids).unsqueeze(2).repeat(
                1, 1, cfg.hc_mult, 1
            )
            chunk_main_hidden = []
            for layer in range(cfg.n_layers):
                h = self._block(h, layer, chunk_ids, start)
                if capture_mh and layer in self.DSPARK_TARGETS:
                    chunk_main_hidden.append(h.mean(dim=2))
            if capture_mh:
                main_hidden_parts.append(
                    torch.cat(chunk_main_hidden, dim=-1)
                )
            y = hc_head(h, *self._hc_head_w(), cfg)
            final_y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
        if final_y is None:
            raise ValueError("prefill requires at least one token")
        self.pos = T
        logits = _linear(final_y[:, -1], self.w("head.weight")).float()
        main_hidden = (
            torch.cat(main_hidden_parts, dim=1)
            if main_hidden_parts is not None
            else None
        )
        return logits, main_hidden

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, full_logits: bool = True) -> torch.Tensor:
        from .dsv4 import hc_head
        cfg = self._cfg_obj()
        ids = ids.to(self.device).long()
        B, T = ids.shape
        if T > 512:
            if full_logits:
                raise RuntimeError(
                    "long prefill full_logits would materialize [T, vocab]; "
                    "use full_logits=False"
                )
            logits, _ = self.prefill_chunked(ids)
            return logits
        self._alloc(B)
        self.ensure_position(T - 1)
        h = self._embed(ids).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        for i in range(cfg.n_layers):
            h = self._block(h, i, ids, 0)
        y = hc_head(h, *self._hc_head_w(), cfg)
        y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
        logits = _linear(y, self.w("head.weight")).float()
        return logits if full_logits else logits[:, -1]

    @torch.no_grad()
    def prefill_mh(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """prefill + 捕获 DSpark main_hidden。返回 (logits 末位 [1, vocab],
        main_hidden [1, T, 3·hidden])；同时建立 KV 并置 model.pos = T。"""
        from .dsv4 import hc_head
        cfg = self._cfg_obj()
        ids = ids.to(self.device).long()
        B, T = ids.shape
        if T > 512:
            logits, main_hidden = self.prefill_chunked(
                ids, capture_mh=True
            )
            assert main_hidden is not None
            return logits, main_hidden
        self._alloc(B)
        self.ensure_position(T - 1)
        h = self._embed(ids).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        mh = []
        for i in range(cfg.n_layers):
            h = self._block(h, i, ids, 0)
            if i in self.DSPARK_TARGETS:
                mh.append(h.mean(dim=2))
        y = hc_head(h, *self._hc_head_w(), cfg)
        y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
        logits = _linear(y, self.w("head.weight")).float()
        self.pos = T
        return logits[:, -1], torch.cat(mh, dim=-1)

    @staticmethod
    def _copy_paged_state(source: PagedKV, target: PagedKV) -> None:
        for page_index, page in enumerate(source.pages):
            target.ensure_page(page_index).copy_(page.to(target.device))
        target.length = int(source.length)

    def _sync_tp_attention_states(self) -> None:
        if self._tp_attention_contexts is None or self.states is None:
            raise RuntimeError("DSV4 TP state cannot be initialized")
        if self._tp_states_ready:
            return
        rank_states = [self.states]
        for rank in range(1, self.tp_size):
            rank_states.append(self._allocate_states(1, self.devices[rank]))
        for rank, states in enumerate(rank_states):
            for layer, state in enumerate(states):
                self._tp_attention_contexts[rank][layer]["state"] = state
                if rank == 0:
                    continue
                source = self.states[layer]
                state["kv"].copy_(source["kv"].to(self.devices[rank]))
                state["win_pos"].copy_(
                    source["win_pos"].to(self.devices[rank])
                )
                if self.ratios[layer]:
                    self._copy_paged_state(
                        source["compressed"], state["compressed"]
                    )
                    state["ckv"].copy_(
                        source["ckv"].to(self.devices[rank])
                    )
                    state["cscore"].copy_(
                        source["cscore"].to(self.devices[rank])
                    )
                    if self.ratios[layer] == 4:
                        source_indexer = source["indexer"]
                        target_indexer = state["indexer"]
                        self._copy_paged_state(
                            source_indexer.keys,
                            target_indexer.keys,
                        )
                        target_indexer.ckv.copy_(
                            source_indexer.ckv.to(self.devices[rank])
                        )
                        target_indexer.cscore.copy_(
                            source_indexer.cscore.to(self.devices[rank])
                        )
        self._tp_states_ready = True

    def _ensure_tp_position(self, position: int) -> None:
        if self._tp_attention_contexts is None:
            return
        for rank in range(self.tp_size):
            for layer, ratio in enumerate(self.ratios):
                if not ratio:
                    continue
                state = self._tp_attention_contexts[rank][layer]["state"]
                state["compressed"].reserve(position // ratio)
                indexer = state.get("indexer")
                if indexer is not None:
                    indexer.reserve_position(position)

    def _tp_route(
        self,
        layer: int,
        rank: int,
        router_logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .ops import route_topk

        assert self._tp_route_weights is not None
        item = self._tp_route_weights[rank][layer]
        logits, weights, indices = self._tp_route_buffers[layer]
        with torch.cuda.device(self.devices[rank]):
            tid2eid = item.get("tid2eid")
            if tid2eid is None:
                routed = route_topk(
                    router_logits,
                    item["gate_bias"],
                    item["mask"],
                    scoring_func="sqrtsoftplus",
                    top_k=int(self.cfg["top_k"]),
                    normalize=bool(self.cfg.get("norm_topk_prob", True)),
                    scaling=float(self.cfg.get("routed_scaling", 1.0)),
                    output_buffers=(weights[rank], indices[rank]),
                )
                if routed is None:
                    raise RuntimeError(
                        "public sqrtsoftplus Router rejected DSV4 TP inputs"
                    )
                return routed
            scores = F.softplus(router_logits).sqrt()
            selected = tid2eid[token_ids.reshape(-1)].reshape(
                1, int(self.cfg["top_k"])
            )
            selected_weights = scores.gather(1, selected)
            if self.cfg.get("norm_topk_prob", True):
                selected_weights = selected_weights / (
                    selected_weights.sum(dim=-1, keepdim=True) + 1e-20
                )
            selected_weights *= float(
                self.cfg.get("routed_scaling", 1.0)
            )
            logits[rank].copy_(scores)
            weights[rank].copy_(selected_weights)
            indices[rank].copy_(selected)
            return weights[rank], indices[rank]

    def _decode_tp(self, ids: torch.Tensor, pos: int) -> torch.Tensor:
        from .dsv4 import hc_head, hc_post
        from .ops import TPHidden

        if (
            self._tp_attention_contexts is None
            or self._tp_route_weights is None
            or self._tp_shared_mlp is None
        ):
            raise RuntimeError("DSV4 full TP metadata is unavailable")
        self._sync_tp_attention_states()
        self._ensure_tp_position(pos)
        cfg = self._cfg_obj()
        embedded = (
            self._embed(ids)
            .unsqueeze(1)
            .unsqueeze(2)
            .repeat(1, 1, cfg.hc_mult, 1)
        )
        hidden = TPHidden.empty(
            self.devices,
            tuple(embedded.shape),
            dtype=embedded.dtype,
        ).copy_from_owner(embedded, 0)

        for layer in range(int(self.cfg["n_layers"])):
            attention_partials = []
            attention_events = []
            attention_aux = []
            for rank, device in enumerate(self.devices):
                route_item = self._tp_route_weights[rank][layer]
                with torch.cuda.device(device):
                    local = hidden.wait_on(device)
                    residual = local
                    y, post, comb = _hc_pre_norm_tpq(
                        local,
                        route_item["hc_attn_fn"],
                        route_item["hc_attn_scale"],
                        route_item["hc_attn_base"],
                        route_item["attn_norm"],
                        cfg,
                    )
                    partial = self._attn_batch(
                        y,
                        layer,
                        pos,
                        None,
                        tp_context=self._tp_attention_contexts[rank][layer],
                    ).float().contiguous()
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    attention_partials.append(partial)
                    attention_events.append(event)
                    attention_aux.append((residual, post, comb))
            for device in self.devices:
                with torch.cuda.device(device):
                    stream = torch.cuda.current_stream(device)
                    for event in attention_events:
                        stream.wait_event(event)
            attention = TPHidden.empty(
                self.devices,
                tuple(attention_partials[0].shape),
                dtype=compute_dtype(self.devices[0]),
            ).reduce_from(attention_partials)

            ffn_input = self._tp_shared_mlp.input_hidden(layer)
            ffn_aux = []
            for rank, device in enumerate(self.devices):
                route_item = self._tp_route_weights[rank][layer]
                with torch.cuda.device(device):
                    value = attention.wait_on(device)
                    residual, post, comb = attention_aux[rank]
                    prefix = hc_post(value, residual, post, comb)
                    ffn_residual = prefix
                    normalized, ffn_post, ffn_comb = _hc_pre_norm_tpq(
                        prefix,
                        route_item["hc_ffn_fn"],
                        route_item["hc_ffn_scale"],
                        route_item["hc_ffn_base"],
                        route_item["ffn_norm"],
                        cfg,
                    )
                    ffn_input.replicas[rank].copy_(
                        normalized.reshape(1, -1)
                    )
                    ffn_input.ready_events[rank].record(
                        torch.cuda.current_stream(device)
                    )
                    ffn_aux.append(
                        (ffn_residual, ffn_post, ffn_comb)
                    )

            shared = self._tp_shared_mlp.run_hidden(layer, ffn_input)
            if self._tp_router is None:
                raise RuntimeError("DSV4 TP Router is unavailable")
            router_logits = self._tp_router.run_sharded(
                layer,
                self._tp_router.bound_input_sharded(layer, ffn_input),
            )
            routes = tuple(
                self._tp_route(
                    layer,
                    rank,
                    router_logits.wait_on(self.devices[rank]),
                    ids.to(self.devices[rank]),
                )
                for rank in range(self.tp_size)
            )
            routed = self.pool.run_hidden(
                layer,
                ffn_input,
                routes,
                activation=self.operator_config.expert_activation,
                activation_beta=float(self.cfg.get("situ_beta", 4.0)),
                activation_linear_beta=self.cfg.get("situ_linear_beta"),
            )
            output = TPHidden.empty(
                self.devices,
                tuple(hidden.shape),
                dtype=hidden.dtype,
            )
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    stream = torch.cuda.current_stream(device)
                    stream.wait_event(shared.ready_events[rank])
                    stream.wait_event(routed.ready_events[rank])
                    ffn_residual, ffn_post, ffn_comb = ffn_aux[rank]
                    combined = (
                        routed.replicas[rank]
                        + shared.replicas[rank]
                    ).view(1, 1, -1)
                    output.replicas[rank].copy_(
                        hc_post(
                            combined,
                            ffn_residual,
                            ffn_post,
                            ffn_comb,
                        )
                    )
                    output.ready_events[rank].record(stream)
            hidden = output

        with torch.cuda.device(self.device):
            final_hidden = hidden.wait_on(self.device)
            y = hc_head(final_hidden, *self._hc_head_w(), cfg)
            y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
            return _linear(y[:, 0], self.w("head.weight")).float()

    @torch.inference_mode()
    def decode(self, ids: torch.Tensor, pos: int) -> torch.Tensor:
        from .dsv4 import hc_head
        cfg = self._cfg_obj()
        ids = ids.to(self.device).long()
        if self._tp_attention_contexts is not None:
            return self._decode_tp(ids, pos)
        self.ensure_position(pos)
        # token 级全层预取：上一 token 各层路由专家在本 token 计算窗口内并行读盘/DMA
        # （时序局部性 70-90%；逐层预取窗口只有 attention 一段，全层预取窗口是整个 token）
        if self._prev_ids and self._prefetch_enabled():
            for l, es in self._prev_ids.items():
                self.pool.prefetch([(l, e) for e in es])
        h = self._embed(ids).unsqueeze(1).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        for i in range(cfg.n_layers):
            h = self._block(h, i, ids.view(-1, 1), pos)
        y = hc_head(h, *self._hc_head_w(), cfg)
        y = rmsnorm(y, self.w("norm.weight"), cfg.rms_eps)
        return _linear(y[:, 0], self.w("head.weight")).float()

    def _hc_head_w(self):
        return (
            self.w("hc_head_fn"),
            self.w("hc_head_scale"),
            self.w("hc_head_base"),
        )


def _f32(w):
    """Int4Weight → 就地反量化 f32（共享专家走 f32 精确路径，体积小）。"""
    if isinstance(w, Int4Weight):
        return w.dequant_rows(0, w.shape[0])
    if isinstance(w, BlockFP8Weight):
        return w.dequant_rows(0, w.shape[0], torch.float32)
    return w.float()
