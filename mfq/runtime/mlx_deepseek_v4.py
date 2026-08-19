"""DeepSeek-V4 TPQ runtime for Apple silicon.

The runtime keeps dense int4 and routed product-VQ weights packed, executes
ordinary multi-projection groups in one Metal dispatch, and owns persistent
local/compressed/indexer caches for prefill and autoregressive generation.
"""

from __future__ import annotations

import math
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's MLX runtime requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.formats import io
from mfq.formats.tpq import TpqPqSpec
from mfq.formats.moe import NintMoeTensor
from mfq.kernels.metal.tpq import (
    MetalTpqMoeWeight,
    MetalTpqPqWeight,
    tpq_grouped_moe_matmul,
    tpq_int4_grouped_row_matmul,
)
from mfq.kernels.metal.deepseek_v4 import (
    attention_dsv4_sparse,
    dsv4_build_decode_plan,
    dsv4_build_prefill_plan,
    dsv4_decode_pool_step,
    dsv4_fp4_sim,
    dsv4_indexer_scores,
    dsv4_topk512,
)
from mfq.kernels.metal.deepseek_v4_hc import dsv4_hc_post, dsv4_hc_pre
from mfq.kernels.metal.moe_ops import (
    moe_topk,
    sqrtsoftplus_weights,
    weighted_reduce,
)
from mfq.kernels.metal.ops import rms_norm
from mfq.kernels.metal.sampling import sample as _sample
from mfq.kernels.metal.vq import signed_hadamard
from mfq.runtime.mlx_tpq import MlxTpqInt4Linear
from mfq.runtime.mlx_linear import (
    MlxDenseLinear,
    MlxLinearGroup,
    MlxNintModel,
    mlx_dense_array,
)
from mfq.runtime.mlx_moe import MlxRoutedLinear


def _int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    return tuple(int(item) for item in value)


@dataclass(frozen=True)
class MlxDeepseekV4Config:
    """Normalized DeepSeek-V4 text-model configuration."""

    n_layers: int
    hidden: int
    n_experts: int
    top_k: int
    moe_inter: int
    n_shared: int
    n_heads: int
    head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    kv_dim: int
    qk_rope_head_dim: int
    n_kv_heads: int
    vocab: int
    rms_eps: float
    scoring_func: str
    norm_topk_prob: bool
    routed_scaling: float
    swiglu_limit: float
    n_hash_layers: int
    sliding_window: int
    rope_theta: float
    rope_scaling: dict[str, Any]
    eos_token_id: tuple[int, ...]
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    max_position_embeddings: int = 1_048_576
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    compress_rope_theta: float = 160_000.0
    compress_ratios: tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def from_manifest(
        cls,
        value: dict[str, Any],
    ) -> MlxDeepseekV4Config:
        text = dict(value)
        layer_count = int(text["n_layers"])
        raw_ratios = tuple(int(item) for item in (text.get("compress_ratios") or ()))
        ratios = raw_ratios or (0,) * layer_count
        if len(ratios) != layer_count:
            raise ValueError("DeepSeek-V4 compress_ratios must contain one entry per layer")
        if any(item not in (0, 4, 128) for item in ratios):
            raise ValueError("DeepSeek-V4 compression ratios must be 0, 4, or 128")
        config = cls(
            n_layers=layer_count,
            hidden=int(text["hidden"]),
            n_experts=int(text["n_experts"]),
            top_k=int(text["top_k"]),
            moe_inter=int(text["moe_inter"]),
            n_shared=int(text.get("n_shared", 1)),
            n_heads=int(text["n_heads"]),
            head_dim=int(text.get("head_dim", 512)),
            q_lora_rank=int(text["q_lora_rank"]),
            o_lora_rank=int(text.get("o_lora_rank", 0)),
            o_groups=int(text.get("o_groups", 1)),
            kv_dim=int(text.get("kv_dim", text.get("head_dim", 512))),
            qk_rope_head_dim=int(text["qk_rope_head_dim"]),
            n_kv_heads=int(text.get("n_kv_heads", 1)),
            vocab=int(text["vocab"]),
            rms_eps=float(text.get("rms_eps", 1e-6)),
            scoring_func=str(text.get("scoring_func", "sqrtsoftplus")),
            norm_topk_prob=bool(text.get("norm_topk_prob", True)),
            routed_scaling=float(text.get("routed_scaling", 1.0)),
            swiglu_limit=float(text.get("swiglu_limit", 0.0)),
            n_hash_layers=int(text.get("n_hash_layers", 0)),
            sliding_window=int(text.get("sliding_window", 128)),
            rope_theta=float(text.get("rope_theta", 10_000.0)),
            rope_scaling=dict(text.get("rope_scaling") or {}),
            eos_token_id=_int_tuple(text.get("eos_token_id")),
            index_n_heads=int(text.get("index_n_heads", 64)),
            index_head_dim=int(text.get("index_head_dim", 128)),
            index_topk=int(text.get("index_topk", 512)),
            max_position_embeddings=int(text.get("max_position_embeddings", 1_048_576)),
            hc_mult=int(text.get("hc_mult", 4)),
            hc_eps=float(text.get("hc_eps", 1e-6)),
            hc_sinkhorn_iters=int(text.get("hc_sinkhorn_iters", 20)),
            compress_rope_theta=float(text.get("compress_rope_theta", 160_000.0)),
            compress_ratios=ratios,
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive = {
            "n_layers": self.n_layers,
            "hidden": self.hidden,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "moe_inter": self.moe_inter,
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "q_lora_rank": self.q_lora_rank,
            "o_lora_rank": self.o_lora_rank,
            "o_groups": self.o_groups,
            "vocab": self.vocab,
            "sliding_window": self.sliding_window,
            "hc_mult": self.hc_mult,
        }
        invalid = [name for name, item in positive.items() if int(item) <= 0]
        if invalid:
            raise ValueError(
                "DeepSeek-V4 configuration fields must be positive: " + ", ".join(invalid)
            )
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError("DeepSeek-V4 Metal requires sqrtsoftplus routing")
        if self.n_kv_heads != 1:
            raise ValueError("DeepSeek-V4 Metal currently requires one KV head")
        if self.top_k > min(16, self.n_experts):
            raise ValueError("DeepSeek-V4 top_k exceeds the Metal router limit")
        if self.n_heads * self.head_dim % self.o_groups:
            raise ValueError("DeepSeek-V4 attention width must divide o_groups")
        if self.qk_rope_head_dim > self.head_dim:
            raise ValueError("DeepSeek-V4 rotary width exceeds head_dim")
        if self.hc_mult != 4 or self.hc_sinkhorn_iters != 20:
            raise ValueError("DeepSeek-V4 Metal requires hc_mult=4 and hc_sinkhorn_iters=20")

    @property
    def fast_attention(self) -> bool:
        return self.n_heads == 64 and self.head_dim == 512

    @property
    def fast_hyper_connections(self) -> bool:
        return self.hidden == 4096

    @property
    def fast_indexer(self) -> bool:
        return self.index_n_heads == 64 and self.index_head_dim == 128


@dataclass(frozen=True)
class MlxDeepseekV4Names:
    """Canonical names emitted by the TPQ-to-MFQ importer."""

    embedding: str = "embed.weight"
    output_norm: str = "norm.weight"
    output: str = "head.weight"
    hc_head_fn: str = "hc_head_fn"
    hc_head_base: str = "hc_head_base"
    hc_head_scale: str = "hc_head_scale"

    @staticmethod
    def layer(index: int, suffix: str) -> str:
        return f"layers.{int(index)}.{suffix}"

    @classmethod
    def required(
        cls,
        config: MlxDeepseekV4Config,
    ) -> tuple[str, ...]:
        names = cls()
        result = [
            names.embedding,
            names.output_norm,
            names.output,
            names.hc_head_fn,
            names.hc_head_base,
            names.hc_head_scale,
        ]
        common = (
            "attn.wq_a.weight",
            "attn.q_norm.weight",
            "attn.wq_b.weight",
            "attn.wkv.weight",
            "attn.kv_norm.weight",
            "attn.attn_sink",
            "attn.wo_a.weight",
            "attn.wo_b.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
            "ffn.gate.weight",
            "ffn.shared_experts.w1.weight",
            "ffn.shared_experts.w3.weight",
            "ffn.shared_experts.w2.weight",
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
            "ffn.experts.gate_up.weight",
            "ffn.experts.down.weight",
        )
        compressor = (
            "attn.compressor.wkv.weight",
            "attn.compressor.wgate.weight",
            "attn.compressor.ape",
            "attn.compressor.norm.weight",
        )
        indexer = (
            "attn.indexer.wq_b.weight",
            "attn.indexer.weights_proj.weight",
            "attn.indexer.compressor.wkv.weight",
            "attn.indexer.compressor.wgate.weight",
            "attn.indexer.compressor.ape",
            "attn.indexer.compressor.norm.weight",
        )
        for layer, ratio in enumerate(config.compress_ratios):
            result.extend(cls.layer(layer, suffix) for suffix in common)
            result.append(
                cls.layer(
                    layer,
                    ("ffn.gate.tid2eid" if layer < config.n_hash_layers else "ffn.gate.bias"),
                )
            )
            if ratio:
                result.extend(cls.layer(layer, suffix) for suffix in compressor)
            if ratio == 4:
                result.extend(cls.layer(layer, suffix) for suffix in indexer)
        return tuple(result)


def _array(
    model: MlxNintModel,
    name: str,
    *,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    value = model.tensors[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"DeepSeek-V4 tensor {name!r} must be stored densely, received {type(value).__name__}"
        )
    return mx.contiguous(mlx_dense_array(value, dtype=dtype))


def _yarn_tables(
    dim: int,
    length: int,
    theta: float,
    yarn: dict[str, Any] | None,
) -> tuple[mx.array, mx.array]:
    frequency = 1.0 / (float(theta) ** (np.arange(0, dim, 2, dtype=np.float32) / float(dim)))
    if yarn:
        original = float(yarn.get("original_max_position_embeddings", 0))
        if original > 0:
            factor = float(yarn.get("factor", 1.0))
            beta_fast = float(yarn.get("beta_fast", 32.0))
            beta_slow = float(yarn.get("beta_slow", 1.0))

            def correction(rotations: float) -> float:
                return (
                    dim
                    * math.log(original / (rotations * 2.0 * math.pi))
                    / (2.0 * math.log(float(theta)))
                )

            low = max(math.floor(correction(beta_fast)), 0)
            high = min(math.ceil(correction(beta_slow)), dim - 1)
            high = high + 0.001 if low == high else high
            ramp = np.clip(
                (np.arange(dim // 2, dtype=np.float32) - low) / (high - low),
                0.0,
                1.0,
            )
            smooth = 1.0 - ramp
            frequency = frequency / factor * (1.0 - smooth) + frequency * smooth
    angles = np.outer(
        np.arange(length, dtype=np.float32),
        frequency.astype(np.float32),
    )
    cosine = mx.array(np.ascontiguousarray(np.cos(angles), dtype=np.float32))
    sine = mx.array(np.ascontiguousarray(np.sin(angles), dtype=np.float32))
    mx.eval(cosine, sine)
    return cosine, sine


def _rope_adjacent(
    value: mx.array,
    cosine: mx.array,
    sine: mx.array,
    *,
    inverse: bool = False,
) -> mx.array:
    source = value.astype(mx.float32)
    cos = cosine.astype(mx.float32)
    sin = -sine.astype(mx.float32) if inverse else sine.astype(mx.float32)
    first = source[..., 0::2]
    second = source[..., 1::2]
    rotated = mx.stack(
        (first * cos - second * sin, first * sin + second * cos),
        axis=-1,
    ).reshape(source.shape)
    return rotated.astype(value.dtype)


def _unweighted_rms(value: mx.array, eps: float) -> mx.array:
    source = value.astype(mx.float32)
    inverse = mx.rsqrt(mx.mean(source * source, axis=-1, keepdims=True) + eps)
    return (source * inverse).astype(value.dtype)


def _mxfp8_fake_quant_prefix(value: mx.array, block_size: int = 64) -> mx.array:
    """Match V4F's in-place UE8M0/E4M3 activation simulation."""
    dtype = value.dtype
    width = value.shape[-1]
    padding = (-width) % block_size
    source = value.astype(mx.float32)
    if padding:
        source = mx.pad(source, [(0, 0)] * (source.ndim - 1) + [(0, padding)])
    grouped = source.reshape((*source.shape[:-1], -1, block_size))
    amax = mx.maximum(mx.max(mx.abs(grouped), axis=-1, keepdims=True), 1.0e-4)
    scale = mx.power(2.0, mx.ceil(mx.log2(amax / 448.0)))
    normalized = mx.clip(grouped / scale, -448.0, 448.0)
    magnitude = mx.abs(normalized)
    subnormal = mx.round(magnitude * 512.0) / 512.0
    exponent = mx.floor(mx.log2(mx.maximum(magnitude, mx.array(1.0e-30))))
    step = mx.power(2.0, exponent - 3.0)
    normal = mx.minimum(mx.round(magnitude / step) * step, 448.0)
    quantized = mx.where(magnitude < 2.0**-6, subnormal, normal)
    restored = mx.sign(normalized) * quantized * scale
    restored = restored.reshape(source.shape)[..., :width]
    return restored.astype(dtype)


def _limited_swiglu(
    gate: mx.array,
    up: mx.array,
    limit: float,
) -> mx.array:
    if limit > 0.0:
        gate = mx.minimum(gate, mx.array(float(limit), dtype=gate.dtype))
        up = mx.minimum(
            mx.maximum(up, mx.array(-float(limit), dtype=up.dtype)),
            mx.array(float(limit), dtype=up.dtype),
        )
    return (gate * mx.sigmoid(gate)) * up


def _hc_pre_generic(
    residual: mx.array,
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    *,
    eps: float,
) -> tuple[mx.array, mx.array, mx.array]:
    pre = mx.sigmoid(mixes[..., :4] * scale[0] + base[:4]) + eps
    post = 2.0 * mx.sigmoid(mixes[..., 4:8] * scale[1] + base[4:8])
    combination = mixes[..., 8:].reshape((*mixes.shape[:-1], 4, 4)) * scale[2] + base[8:].reshape(
        (4, 4)
    )
    combination = mx.softmax(combination, axis=-1) + eps
    combination = combination / (mx.sum(combination, axis=-2, keepdims=True) + eps)
    for _ in range(19):
        combination = combination / (mx.sum(combination, axis=-1, keepdims=True) + eps)
        combination = combination / (mx.sum(combination, axis=-2, keepdims=True) + eps)
    reduced = mx.sum(pre[..., None] * residual, axis=-2)
    return reduced, post, combination


def _hc_post_generic(
    branch: mx.array,
    residual: mx.array,
    post: mx.array,
    combination: mx.array,
) -> mx.array:
    mixed = mx.einsum("btjk,btjd->btkd", combination, residual)
    return (post[..., None] * branch[..., None, :] + mixed).astype(residual.dtype)


@dataclass
class MlxDeepseekV4PoolState:
    """Persistent bounded state for one DSV4 KV compressor."""

    ratio: int
    head_dim: int
    overlap: bool
    batch: int
    capacity: int
    dtype: mx.Dtype
    pool: mx.array
    state_kv: mx.array
    state_gate: mx.array
    prev_kv: mx.array | None
    prev_gate: mx.array | None
    pool_len: int = 0
    remainder: int = 0

    @classmethod
    def allocate(
        cls,
        *,
        ratio: int,
        head_dim: int,
        overlap: bool,
        batch: int,
        max_context: int,
        dtype: mx.Dtype = mx.float16,
    ) -> MlxDeepseekV4PoolState:
        output_dim = head_dim * (2 if overlap else 1)
        capacity = max(1, (max_context + ratio - 1) // ratio)
        previous_shape = (batch, ratio, head_dim)
        return cls(
            ratio=ratio,
            head_dim=head_dim,
            overlap=overlap,
            batch=batch,
            capacity=capacity,
            dtype=dtype,
            pool=mx.zeros((batch, capacity, head_dim), dtype=dtype),
            state_kv=mx.zeros((batch, ratio, output_dim), dtype=dtype),
            state_gate=mx.full(
                (batch, ratio, output_dim),
                -mx.inf,
                dtype=dtype,
            ),
            prev_kv=(mx.zeros(previous_shape, dtype=dtype) if overlap else None),
            prev_gate=(mx.full(previous_shape, -mx.inf, dtype=dtype) if overlap else None),
        )

    def _generic_step(
        self,
        kv_token: mx.array,
        gate_token: mx.array,
        ape: mx.array,
        norm: mx.array,
        length: int,
        cosine: mx.array,
        sine: mx.array,
        eps: float,
    ) -> None:
        slot = (length - 1) % self.ratio
        self.state_kv[:, slot] = kv_token[:, 0]
        self.state_gate[:, slot] = gate_token[:, 0]
        if length % self.ratio:
            self.remainder = length % self.ratio
            return
        scores = self.state_gate.astype(mx.float32) + ape[None]
        if self.overlap:
            assert self.prev_kv is not None and self.prev_gate is not None
            values = mx.concatenate(
                (
                    self.prev_kv,
                    self.state_kv[..., self.head_dim :],
                ),
                axis=1,
            )
            scores = mx.concatenate(
                (
                    self.prev_gate.astype(mx.float32) + ape[None, :, : self.head_dim],
                    scores[..., self.head_dim :],
                ),
                axis=1,
            )
        else:
            values = self.state_kv
        probabilities = mx.softmax(scores, axis=1)
        pooled = mx.sum(values.astype(mx.float32) * probabilities, axis=1)
        pooled = rms_norm(pooled, norm, eps)
        position = (length // self.ratio - 1) * self.ratio
        rotary = min(self.head_dim, int(cosine.shape[1]) * 2)
        rotated = _rope_adjacent(
            pooled[..., -rotary:],
            cosine[position, : rotary // 2],
            sine[position, : rotary // 2],
        )
        pooled = mx.concatenate((pooled[..., :-rotary], rotated), axis=-1)
        row = length // self.ratio - 1
        self.pool[:, row] = pooled.astype(self.dtype)
        if self.overlap:
            self.prev_kv = self.state_kv[..., : self.head_dim]
            self.prev_gate = self.state_gate[..., : self.head_dim]
        self.pool_len = max(self.pool_len, row + 1)
        self.remainder = 0

    def update(
        self,
        kv_token: mx.array,
        gate_token: mx.array,
        ape: mx.array,
        norm: mx.array,
        *,
        length: int,
        cosine: mx.array,
        sine: mx.array,
        quant_mode: int,
        eps: float,
    ) -> None:
        if length <= 0 or length > self.capacity * self.ratio:
            raise ValueError("DeepSeek-V4 compressor length exceeds capacity")
        fast = self.head_dim in (128, 512)
        if not fast:
            self._generic_step(
                kv_token,
                gate_token,
                ape,
                norm,
                length,
                cosine,
                sine,
                eps,
            )
            return
        downsampled_cosine = mx.contiguous(cosine[:: self.ratio])
        downsampled_sine = mx.contiguous(sine[:: self.ratio])
        step = dsv4_decode_pool_step(
            kv_token,
            gate_token,
            ape,
            norm,
            self.state_kv,
            self.state_gate,
            self.prev_kv,
            self.prev_gate,
            mx.full((self.batch,), length, dtype=mx.int32),
            downsampled_cosine,
            downsampled_sine,
            self.ratio,
            self.overlap,
            quant_mode,
            eps,
        )
        self.state_kv = step.state_kv
        self.state_gate = step.state_gate
        self.prev_kv = step.prev_kv
        self.prev_gate = step.prev_gate
        if length % self.ratio == 0:
            row = length // self.ratio - 1
            self.pool[:, row : row + 1] = step.emitted
            self.pool_len = max(self.pool_len, row + 1)
        self.remainder = length % self.ratio

    def arrays(self) -> tuple[mx.array, ...]:
        result = [self.pool, self.state_kv, self.state_gate]
        if self.prev_kv is not None:
            result.append(self.prev_kv)
        if self.prev_gate is not None:
            result.append(self.prev_gate)
        return tuple(result)


@dataclass
class _MlxDsv4LayerState:
    local: mx.array
    local_positions: mx.array
    main: MlxDeepseekV4PoolState | None
    indexer: MlxDeepseekV4PoolState | None

    @classmethod
    def allocate(
        cls,
        config: MlxDeepseekV4Config,
        ratio: int,
        batch: int,
        max_context: int,
    ) -> _MlxDsv4LayerState:
        main = (
            MlxDeepseekV4PoolState.allocate(
                ratio=ratio,
                head_dim=config.head_dim,
                overlap=ratio == 4,
                batch=batch,
                max_context=max_context,
            )
            if ratio
            else None
        )
        indexer = (
            MlxDeepseekV4PoolState.allocate(
                ratio=ratio,
                head_dim=config.index_head_dim,
                overlap=True,
                batch=batch,
                max_context=max_context,
            )
            if ratio == 4
            else None
        )
        return cls(
            local=mx.zeros(
                (batch, config.sliding_window, config.head_dim),
                dtype=mx.float16,
            ),
            local_positions=mx.full(
                (batch, config.sliding_window),
                -1,
                dtype=mx.int32,
            ),
            main=main,
            indexer=indexer,
        )

    def arrays(self) -> tuple[mx.array, ...]:
        result = [self.local, self.local_positions]
        if self.main is not None:
            result.extend(self.main.arrays())
        if self.indexer is not None:
            result.extend(self.indexer.arrays())
        return tuple(result)


_TPQ_PQ_HEADER = struct.Struct("<4sBBBBiiII")


class _UnsupportedStreamedExpertsError(TypeError):
    pass


@dataclass(frozen=True)
class _MlxTpqStreamPool:
    spec: TpqPqSpec
    codebook: mx.array
    indices_offset: int
    rows_per_expert: int
    columns: int
    expert_count: int

    @property
    def blocks(self) -> int:
        return self.columns // self.spec.vector_size

    @property
    def indices_per_expert(self) -> int:
        return self.rows_per_expert * self.blocks


class _MlxTpqExpertResidency:
    """Bounded per-expert Metal residency over mmap-backed TPQ records."""

    def __init__(
        self,
        store: io.MMapTensorStore,
        *,
        cache_gb: float,
        experts: int,
    ) -> None:
        if not math.isfinite(cache_gb) or cache_gb < 0.0:
            raise ValueError("DeepSeek-V4 expert_cache_gb must be non-negative")
        self.store = store
        self.experts = int(experts)
        self.cache_limit = int(float(cache_gb) * (1 << 30))
        self.cache: OrderedDict[
            tuple[str, int],
            MetalTpqPqWeight,
        ] = OrderedDict()
        self.cache_nbytes = 0
        self.projections: dict[
            str,
            dict[int, tuple[_MlxTpqStreamPool, int]],
        ] = {}

    def _parse_projection(
        self,
        name: str,
    ) -> dict[int, tuple[_MlxTpqStreamPool, int]]:
        cached = self.projections.get(name)
        if cached is not None:
            return cached
        record = self.store.records[name]
        if record.dtype != "NINTM":
            raise _UnsupportedStreamedExpertsError(f"expert record {name!r} is not NINTM")
        source = self.store.mmap_for(record)
        start = int(record.offset)
        end = start + int(record.nbytes)
        if start + io._NINT_MOE_HDR.size > end:
            raise ValueError(f"truncated native TPQ expert header: {name}")
        magic, experts, rows_per_expert, columns, pool_count = io._NINT_MOE_HDR.unpack_from(
            source, start
        )
        if magic != b"NIM2" or int(experts) != self.experts:
            raise _UnsupportedStreamedExpertsError(
                f"expert record {name!r} is not a native TPQ NIM2 container"
            )
        offset = start + io._NINT_MOE_HDR.size
        result: dict[int, tuple[_MlxTpqStreamPool, int]] = {}
        for _ in range(int(pool_count)):
            if offset + io._NINT_MOE_POOL_V2_HDR.size > end:
                raise ValueError(f"truncated native TPQ pool header: {name}")
            (
                expert_count,
                dtype_nbytes,
                payload_nbytes,
                runtime_nbytes,
            ) = io._NINT_MOE_POOL_V2_HDR.unpack_from(source, offset)
            offset += io._NINT_MOE_POOL_V2_HDR.size
            ids_nbytes = int(expert_count) * np.dtype("<i4").itemsize
            ids_end = offset + ids_nbytes
            dtype_end = ids_end + int(dtype_nbytes)
            runtime_end = dtype_end + int(runtime_nbytes)
            payload_end = runtime_end + int(payload_nbytes)
            if expert_count <= 0 or dtype_nbytes <= 0 or dtype_nbytes > 32 or payload_end > end:
                raise ValueError(f"invalid native TPQ pool metadata: {name}")
            expert_ids = np.frombuffer(
                source,
                dtype="<i4",
                count=int(expert_count),
                offset=offset,
            )
            dtype = bytes(source[ids_end:dtype_end]).decode("ascii")
            if not dtype.startswith("TPQ-") or runtime_nbytes:
                raise _UnsupportedStreamedExpertsError(
                    f"expert record {name!r} contains non-TPQ cohorts"
                )
            payload_start = runtime_end
            if payload_start + _TPQ_PQ_HEADER.size > payload_end:
                raise ValueError(f"truncated native TPQ payload: {name}")
            (
                pq_magic,
                pq_version,
                _tier_id,
                vector_size,
                index_bits,
                axis,
                neuron_len,
                ndim,
                entries,
            ) = _TPQ_PQ_HEADER.unpack_from(source, payload_start)
            if (
                pq_magic != b"CPQ1"
                or pq_version != 1
                or index_bits not in (8, 16)
                or axis != 0
                or ndim != 2
            ):
                raise _UnsupportedStreamedExpertsError(
                    f"expert record {name!r} is not byte-aligned TPQ"
                )
            spec = TpqPqSpec(
                tier=dtype.removeprefix("TPQ-").lower(),
                vector_size=int(vector_size),
                codebook_entries=int(entries),
            )
            if spec.index_bits != int(index_bits):
                raise ValueError(f"native TPQ tier metadata is inconsistent: {name}")
            payload_offset = payload_start + _TPQ_PQ_HEADER.size
            shape = tuple(int(value) for value in struct.unpack_from("<2q", source, payload_offset))
            payload_offset += 16
            rows = int(struct.unpack_from("<I", source, payload_offset)[0])
            payload_offset += 4
            expected_shape = (
                int(expert_count) * int(rows_per_expert),
                int(columns),
            )
            if (
                shape != expected_shape
                or rows != shape[0]
                or int(neuron_len) != shape[1]
                or shape[1] % spec.vector_size
            ):
                raise ValueError(f"native TPQ pool shape mismatch in {name}")
            codebook_count = spec.codebook_entries * spec.vector_size
            codebook_end = payload_offset + codebook_count * 4
            if codebook_end > payload_end:
                raise ValueError(f"truncated native TPQ codebook: {name}")
            codebook = np.frombuffer(
                source,
                dtype="<f4",
                count=codebook_count,
                offset=payload_offset,
            ).reshape((spec.codebook_entries, spec.vector_size))
            index_count = shape[0] * (shape[1] // spec.vector_size)
            expected_end = codebook_end + index_count * (index_bits // 8)
            if expected_end != payload_end:
                raise ValueError(f"native TPQ index payload size mismatch: {name}")
            pool = _MlxTpqStreamPool(
                spec=spec,
                codebook=mx.array(np.ascontiguousarray(codebook, dtype=np.float16)),
                indices_offset=codebook_end,
                rows_per_expert=int(rows_per_expert),
                columns=int(columns),
                expert_count=int(expert_count),
            )
            for local, expert in enumerate(expert_ids):
                expert_id = int(expert)
                if expert_id in result:
                    raise ValueError(f"duplicate native TPQ expert {expert_id}: {name}")
                result[expert_id] = (pool, local)
            offset = payload_end
        if offset != end:
            raise ValueError(f"invalid native TPQ expert tail: {name}")
        self.projections[name] = result
        return result

    def can_stream(self, name: str) -> bool:
        try:
            self._parse_projection(name)
        except _UnsupportedStreamedExpertsError:
            return False
        return True

    def _load(
        self,
        name: str,
        expert: int,
    ) -> MetalTpqPqWeight:
        projection = self._parse_projection(name)
        try:
            pool, local = projection[int(expert)]
        except KeyError as exc:
            raise KeyError(f"native TPQ expert {expert} is absent from {name}") from exc
        dtype = np.dtype(np.uint8 if pool.spec.index_bits == 8 else "<u2")
        count = pool.indices_per_expert
        offset = pool.indices_offset + int(local) * count * dtype.itemsize
        indices = np.frombuffer(
            self.store.mmap_for(name),
            dtype=dtype,
            count=count,
            offset=offset,
        ).reshape((pool.rows_per_expert, pool.blocks))
        index_dtype = mx.uint8 if pool.spec.index_bits == 8 else mx.uint16
        return MetalTpqPqWeight(
            indices=mx.array(np.ascontiguousarray(indices)).astype(index_dtype),
            codebook=pool.codebook,
            out=pool.rows_per_expert,
            neuron_len=pool.columns,
            vector_size=pool.spec.vector_size,
            entries=pool.spec.codebook_entries,
            index_bits=pool.spec.index_bits,
        )

    def weights(
        self,
        name: str,
        expert_ids: tuple[int, ...],
    ) -> tuple[tuple[int, MetalTpqPqWeight], ...]:
        active: list[tuple[int, MetalTpqPqWeight]] = []
        for expert in expert_ids:
            key = (name, int(expert))
            weight = self.cache.pop(key, None)
            if weight is None:
                weight = self._load(name, expert)
                self.cache_nbytes += int(weight.indices.nbytes)
            self.cache[key] = weight
            active.append((int(expert), weight))
        active_keys = {(name, int(expert)) for expert in expert_ids}
        while self.cache_nbytes > self.cache_limit and self.cache:
            key = next(
                (candidate for candidate in self.cache if candidate not in active_keys),
                None,
            )
            if key is None:
                break
            weight = self.cache.pop(key)
            self.cache_nbytes -= int(weight.indices.nbytes)
        return tuple(active)

    def grouped(
        self,
        name: str,
        expert_ids: tuple[int, ...],
        *,
        out_per_expert: int,
        neuron_len: int,
    ) -> MetalTpqMoeWeight:
        return MetalTpqMoeWeight.from_expert_weights(
            self.weights(name, expert_ids),
            experts=self.experts,
            out_per_expert=out_per_expert,
            neuron_len=neuron_len,
        )

    def clear(self) -> None:
        self.cache.clear()
        self.projections.clear()
        self.cache_nbytes = 0


class MlxDeepseekV4MoE:
    """Packed routed and shared DeepSeek-V4 expert graph."""

    def __init__(
        self,
        model: MlxNintModel,
        config: MlxDeepseekV4Config,
        layer: int,
        available: np.ndarray,
        residency: _MlxTpqExpertResidency | None,
    ) -> None:
        prefix = f"layers.{layer}.ffn"
        self.config = config
        self.layer = layer
        self.projections = MlxLinearGroup(
            (
                model.linear(f"{prefix}.gate.weight"),
                model.linear(f"{prefix}.shared_experts.w1.weight"),
                model.linear(f"{prefix}.shared_experts.w3.weight"),
            )
        )
        self.shared_down = model.linear(f"{prefix}.shared_experts.w2.weight")
        self.gate_up_name = f"{prefix}.experts.gate_up.weight"
        self.down_name = f"{prefix}.experts.down.weight"
        self.residency = (
            residency
            if residency is not None
            and residency.can_stream(self.gate_up_name)
            and residency.can_stream(self.down_name)
            else None
        )
        if self.residency is None:
            gate_up = model.tensors[self.gate_up_name]
            down = model.tensors[self.down_name]
            if not isinstance(gate_up, NintMoeTensor) or not isinstance(
                down,
                NintMoeTensor,
            ):
                raise TypeError(f"DeepSeek-V4 layer {layer} expert records must be NINTM")
            self.gate_up: MlxRoutedLinear | None = MlxRoutedLinear(gate_up)
            self.down: MlxRoutedLinear | None = MlxRoutedLinear(down)
        else:
            self.gate_up = None
            self.down = None
        self.available = mx.array(np.ascontiguousarray(available, dtype=np.bool_))
        self.router_bias = (
            None if layer < config.n_hash_layers else _array(model, f"{prefix}.gate.bias")
        )
        self.tid2eid = (
            _array(
                model,
                f"{prefix}.gate.tid2eid",
                dtype=mx.int32,
            )
            if layer < config.n_hash_layers
            else None
        )

    def _repair_hash_ids(
        self,
        logits: mx.array,
        static_ids: mx.array,
    ) -> mx.array:
        config = self.config
        candidate_count = min(
            16,
            config.n_experts,
            max(config.top_k * 2, config.top_k),
        )
        candidates, _ = moe_topk(
            logits,
            candidate_count,
            use_sqrt_softplus=True,
            available=self.available,
        )
        result = static_ids.astype(mx.int32)
        for route in range(config.top_k):
            current = result[:, route]
            bad = ~self.available[current]
            replacement = candidates[:, 0]
            found = mx.zeros(bad.shape, dtype=mx.bool_)
            for candidate_slot in range(candidate_count):
                candidate = candidates[:, candidate_slot]
                duplicate = mx.any(result == candidate[:, None], axis=1)
                take = bad & ~found & ~duplicate
                replacement = mx.where(take, candidate, replacement)
                found = found | take
            columns = [
                replacement if index == route else result[:, index] for index in range(config.top_k)
            ]
            result = mx.stack(columns, axis=1)
        return result

    def __call__(
        self,
        x: mx.array,
        token_ids: mx.array,
    ) -> mx.array:
        config = self.config
        shape = tuple(int(item) for item in x.shape)
        rows = int(x.size) // config.hidden
        source = x.reshape((rows, config.hidden))
        logits, shared_gate, shared_up = self.projections(source)
        if self.tid2eid is not None:
            ids = self.tid2eid[token_ids.reshape((-1,))]
            ids = self._repair_hash_ids(logits, ids)
            if config.norm_topk_prob:
                weights = sqrtsoftplus_weights(
                    logits,
                    ids,
                    scale=config.routed_scaling,
                )
            else:
                scores = mx.sqrt(
                    mx.logaddexp(
                        logits.astype(mx.float32),
                        mx.array(0.0, dtype=mx.float32),
                    )
                )
                weights = mx.take_along_axis(scores, ids, axis=1) * config.routed_scaling
        else:
            ids, weights = moe_topk(
                logits,
                config.top_k,
                use_sqrt_softplus=True,
                normalize=config.norm_topk_prob,
                bias=self.router_bias,
                available=self.available,
                scale=config.routed_scaling,
            )
        if self.residency is None:
            assert self.gate_up is not None and self.down is not None
            gate_up = self.gate_up(source, ids)
            gate, up = mx.split(gate_up, 2, axis=-1)
            hidden = _limited_swiglu(gate, up, config.swiglu_limit)
            routed = self.down.combine(hidden, ids, weights)
        else:
            routed_parts = []
            for start in range(0, rows, 16):
                end = min(rows, start + 16)
                chunk_ids = mx.contiguous(ids[start:end])
                mx.eval(chunk_ids)
                selected = tuple(
                    sorted(int(item) for item in np.unique(np.asarray(chunk_ids)) if int(item) >= 0)
                )
                gate_up_weight = self.residency.grouped(
                    self.gate_up_name,
                    selected,
                    out_per_expert=2 * config.moe_inter,
                    neuron_len=config.hidden,
                )
                gate_up = tpq_grouped_moe_matmul(
                    gate_up_weight,
                    source[start:end],
                    chunk_ids,
                )
                gate, up = mx.split(gate_up, 2, axis=-1)
                hidden = _limited_swiglu(
                    gate,
                    up,
                    config.swiglu_limit,
                )
                down_weight = self.residency.grouped(
                    self.down_name,
                    selected,
                    out_per_expert=config.hidden,
                    neuron_len=config.moe_inter,
                )
                down = tpq_grouped_moe_matmul(
                    down_weight,
                    hidden,
                    chunk_ids,
                )
                routed_parts.append(weighted_reduce(down, weights[start:end]))
            routed = mx.concatenate(routed_parts, axis=0)
        shared_hidden = _limited_swiglu(
            shared_gate,
            shared_up,
            config.swiglu_limit,
        )
        shared = self.shared_down(shared_hidden)
        return (routed + shared).reshape(shape)


class MlxDeepseekV4Attention:
    """MQA, KV compression, Indexer selection, and grouped O LoRA."""

    def __init__(
        self,
        model: MlxNintModel,
        config: MlxDeepseekV4Config,
        layer: int,
        ratio: int,
        max_context: int,
        rope_base: tuple[mx.array, mx.array],
        rope_compressed: tuple[mx.array, mx.array],
    ) -> None:
        prefix = f"layers.{layer}.attn"
        self.config = config
        self.layer = layer
        self.ratio = ratio
        self.max_context = max_context
        self.rope = rope_compressed if ratio else rope_base
        projection_names = [
            f"{prefix}.wq_a.weight",
            f"{prefix}.wkv.weight",
        ]
        self._projection_keys = ["q_a", "kv"]
        if ratio:
            projection_names.extend(
                (
                    f"{prefix}.compressor.wkv.weight",
                    f"{prefix}.compressor.wgate.weight",
                )
            )
            self._projection_keys.extend(("main_kv", "main_gate"))
        if ratio == 4:
            projection_names.extend(
                (
                    f"{prefix}.indexer.compressor.wkv.weight",
                    f"{prefix}.indexer.compressor.wgate.weight",
                    f"{prefix}.indexer.weights_proj.weight",
                )
            )
            self._projection_keys.extend(("index_kv", "index_gate", "index_weights"))
        self.projections = MlxLinearGroup(tuple(model.linear(name) for name in projection_names))
        self.q_b = model.linear(f"{prefix}.wq_b.weight")
        self.q_norm = _array(model, f"{prefix}.q_norm.weight")
        self.kv_norm = _array(model, f"{prefix}.kv_norm.weight")
        self.sinks = _array(model, f"{prefix}.attn_sink")
        self.wo_a = model.linear(f"{prefix}.wo_a.weight")
        self.wo_b = model.linear(f"{prefix}.wo_b.weight")
        self.main_ape = _array(model, f"{prefix}.compressor.ape") if ratio else None
        self.main_norm = _array(model, f"{prefix}.compressor.norm.weight") if ratio else None
        self.index_q_b = model.linear(f"{prefix}.indexer.wq_b.weight") if ratio == 4 else None
        self.index_ape = _array(model, f"{prefix}.indexer.compressor.ape") if ratio == 4 else None
        self.index_norm = (
            _array(model, f"{prefix}.indexer.compressor.norm.weight") if ratio == 4 else None
        )
        self._hadamard_signs = (
            mx.ones((config.index_head_dim,), dtype=mx.int8) if ratio == 4 else None
        )

    def _project(
        self,
        x: mx.array,
        positions: mx.array,
    ) -> tuple[dict[str, mx.array], mx.array, mx.array, mx.array]:
        config = self.config
        outputs = dict(
            zip(
                self._projection_keys,
                self.projections(x),
                strict=True,
            )
        )
        q_rank = rms_norm(
            outputs["q_a"],
            self.q_norm,
            config.rms_eps,
        )
        q = self.q_b(q_rank).reshape((*x.shape[:-1], config.n_heads, config.head_dim))
        q = _unweighted_rms(q, config.rms_eps)
        kv = rms_norm(outputs["kv"], self.kv_norm, config.rms_eps)
        cosine = self.rope[0][positions]
        sine = self.rope[1][positions]
        rotary = config.qk_rope_head_dim
        q_rotary = _rope_adjacent(
            q[..., -rotary:],
            cosine[None, :, None],
            sine[None, :, None],
        )
        kv_rotary = _rope_adjacent(
            kv[..., -rotary:],
            cosine[None],
            sine[None],
        )
        q = mx.concatenate((q[..., :-rotary], q_rotary), axis=-1)
        kv_prefix = _mxfp8_fake_quant_prefix(kv[..., :-rotary])
        kv = mx.concatenate((kv_prefix, kv_rotary), axis=-1)
        return outputs, q_rank, q, kv

    def _update_compressors(
        self,
        outputs: dict[str, mx.array],
        state: _MlxDsv4LayerState,
        *,
        pos0: int,
    ) -> None:
        if not self.ratio:
            return
        assert state.main is not None
        assert self.main_ape is not None and self.main_norm is not None
        tokens = int(outputs["main_kv"].shape[1])
        for token in range(tokens):
            length = pos0 + token + 1
            state.main.update(
                outputs["main_kv"][:, token : token + 1],
                outputs["main_gate"][:, token : token + 1],
                self.main_ape,
                self.main_norm,
                length=length,
                cosine=self.rope[0],
                sine=self.rope[1],
                quant_mode=0,
                eps=self.config.rms_eps,
            )
            if self.ratio == 4:
                assert state.indexer is not None
                assert self.index_ape is not None
                assert self.index_norm is not None
                state.indexer.update(
                    outputs["index_kv"][:, token : token + 1],
                    outputs["index_gate"][:, token : token + 1],
                    self.index_ape,
                    self.index_norm,
                    length=length,
                    cosine=self.rope[0],
                    sine=self.rope[1],
                    quant_mode=2 if self.config.fast_indexer else 0,
                    eps=self.config.rms_eps,
                )
                if state.indexer.pool_len != state.main.pool_len:
                    raise RuntimeError("DeepSeek-V4 main and Indexer pool lengths diverged")

    def _index_query(
        self,
        q_rank: mx.array,
        outputs: dict[str, mx.array],
        positions: mx.array,
    ) -> tuple[mx.array, mx.array]:
        assert self.index_q_b is not None
        config = self.config
        query = self.index_q_b(q_rank).reshape(
            (
                int(q_rank.shape[0]),
                int(q_rank.shape[1]),
                config.index_n_heads,
                config.index_head_dim,
            )
        )
        rotary = config.qk_rope_head_dim
        cosine = self.rope[0][positions]
        sine = self.rope[1][positions]
        rotated = _rope_adjacent(
            query[..., -rotary:],
            cosine[None, :, None],
            sine[None, :, None],
        )
        query = mx.concatenate((query[..., :-rotary], rotated), axis=-1)
        if self._hadamard_signs is not None:
            query = signed_hadamard(
                query.reshape((-1, config.index_head_dim)),
                self._hadamard_signs,
                config.index_head_dim,
            ).reshape(query.shape)
            query = dsv4_fp4_sim(query.astype(mx.float16))
        return query, outputs["index_weights"]

    def _topk(
        self,
        q_rank: mx.array,
        outputs: dict[str, mx.array],
        state: _MlxDsv4LayerState,
        *,
        positions: mx.array,
        pos0: int,
    ) -> mx.array:
        batch, tokens = map(int, q_rank.shape[:2])
        pool_len = 0 if state.main is None else state.main.pool_len
        if pool_len <= 0:
            return mx.zeros((batch, tokens, 0), dtype=mx.int32)
        if self.ratio != 4 or pool_len <= self.config.index_topk:
            values = mx.arange(pool_len, dtype=mx.int32)
            return mx.broadcast_to(
                values[None, None],
                (batch, tokens, pool_len),
            )
        assert state.indexer is not None
        index_query, weights = self._index_query(
            q_rank,
            outputs,
            positions,
        )
        if not self.config.fast_indexer:
            dots = mx.einsum(
                "bmhd,bkd->bmhk",
                index_query.astype(mx.float32),
                state.indexer.pool[:, :pool_len].astype(mx.float32),
            )
            scores = mx.sum(
                mx.maximum(dots, 0.0) * weights[..., None],
                axis=2,
            ) / math.sqrt(self.config.index_head_dim * self.config.index_n_heads)
            visible = (
                mx.arange(pool_len, dtype=mx.int32)[None, None, :]
                < (positions[None, :, None] + 1) // self.ratio
            )
            scores = mx.where(visible, scores, -mx.inf)
            return mx.argpartition(
                scores,
                kth=pool_len - self.config.index_topk,
                axis=-1,
            )[..., -self.config.index_topk :].astype(mx.int32)
        scores = dsv4_indexer_scores(
            index_query,
            state.indexer.pool[:, :pool_len],
            weights,
            pos0,
            self.ratio,
        )
        return dsv4_topk512(scores)

    def _attention(
        self,
        q: mx.array,
        cache: mx.array,
        indices: mx.array,
        mask: mx.array,
    ) -> mx.array:
        config = self.config
        query = mx.transpose(q, (0, 2, 1, 3))
        if config.fast_attention:
            return attention_dsv4_sparse(
                query,
                cache,
                indices,
                mask,
                self.sinks,
            )
        batch_indices = mx.arange(
            int(cache.shape[0]),
            dtype=mx.int32,
        )[:, None, None]
        selected = cache[batch_indices, indices]
        scores = mx.einsum(
            "bmhd,bmsd->bmhs",
            q.astype(mx.float32),
            selected.astype(mx.float32),
        ) / math.sqrt(config.head_dim)
        scores = scores + mask[:, :, None, :].astype(mx.float32)
        maximum = mx.maximum(
            mx.max(scores, axis=-1),
            self.sinks[None, None],
        )
        exponentials = mx.exp(scores - maximum[..., None])
        denominator = mx.sum(exponentials, axis=-1) + mx.exp(self.sinks[None, None] - maximum)
        return mx.einsum(
            "bmhs,bmsd->bmhd",
            exponentials / denominator[..., None],
            selected.astype(mx.float32),
        )

    def _o_projection(self, value: mx.array) -> mx.array:
        config = self.config
        batch, tokens = map(int, value.shape[:2])
        input_width = config.n_heads * config.head_dim // config.o_groups
        grouped = value.reshape((batch, tokens, config.o_groups, input_width))
        if isinstance(self.wo_a, MlxTpqInt4Linear):
            low_rank = tpq_int4_grouped_row_matmul(
                self.wo_a.packed_weight,
                grouped,
                groups=config.o_groups,
            )
        elif isinstance(self.wo_a, MlxDenseLinear):
            dense = self.wo_a.weight.reshape((config.o_groups, config.o_lora_rank, input_width))
            low_rank = mx.einsum("btgd,grd->btgr", grouped, dense)
        else:
            pieces = []
            for group in range(config.o_groups):
                complete = self.wo_a(grouped[:, :, group])
                start = group * config.o_lora_rank
                pieces.append(complete[..., start : start + config.o_lora_rank])
            low_rank = mx.stack(pieces, axis=2)
        return self.wo_b(low_rank.reshape((batch, tokens, -1)))

    def __call__(
        self,
        x: mx.array,
        state: _MlxDsv4LayerState,
        *,
        pos0: int,
    ) -> mx.array:
        config = self.config
        batch, tokens = map(int, x.shape[:2])
        positions = mx.arange(pos0, pos0 + tokens, dtype=mx.int32)
        outputs, q_rank, q, kv = self._project(x, positions)
        self._update_compressors(outputs, state, pos0=pos0)
        topk = self._topk(
            q_rank,
            outputs,
            state,
            positions=positions,
            pos0=pos0,
        )
        window = config.sliding_window
        pool_len = 0 if state.main is None else state.main.pool_len
        if tokens == 1:
            slot = pos0 % window
            state.local[:, slot] = kv[:, 0].astype(mx.float16)
            state.local_positions[:, slot] = pos0
            unified = (
                state.local
                if state.main is None
                else mx.concatenate(
                    (state.local, state.main.pool[:, :pool_len]),
                    axis=1,
                )
            )
            indices, mask = dsv4_build_decode_plan(
                topk,
                mx.full((batch,), pos0 + 1, dtype=mx.int32),
                pool_len,
                self.ratio or 1,
                window,
            )
        else:
            history = min(pos0, window)
            history_positions = mx.arange(
                pos0 - history,
                pos0,
                dtype=mx.int32,
            )
            history_values = (
                state.local[:, history_positions % window] if history else state.local[:, :0]
            )
            parts = [history_values, kv.astype(mx.float16)]
            if state.main is not None:
                parts.append(state.main.pool[:, :pool_len])
            unified = mx.concatenate(parts, axis=1)
            indices, mask = dsv4_build_prefill_plan(
                topk,
                query_offset=pos0,
                local_history=history,
                pool_len=pool_len,
                ratio=self.ratio or 1,
                window=window,
            )
            recent = min(tokens, window)
            recent_positions = positions[-recent:]
            state.local[:, recent_positions % window] = kv[:, -recent:].astype(mx.float16)
            state.local_positions[:, recent_positions % window] = recent_positions[None]
        attended = self._attention(q, unified, indices, mask)
        rotary = config.qk_rope_head_dim
        cosine = self.rope[0][positions]
        sine = self.rope[1][positions]
        inverse = _rope_adjacent(
            attended[..., -rotary:],
            cosine[None, :, None],
            sine[None, :, None],
            inverse=True,
        )
        attended = mx.concatenate(
            (attended[..., :-rotary], inverse),
            axis=-1,
        )
        return self._o_projection(attended)


class _MlxDeepseekV4Layer:
    def __init__(
        self,
        model: MlxNintModel,
        config: MlxDeepseekV4Config,
        index: int,
        available: np.ndarray,
        max_context: int,
        rope_base: tuple[mx.array, mx.array],
        rope_compressed: tuple[mx.array, mx.array],
        residency: _MlxTpqExpertResidency | None,
    ) -> None:
        prefix = f"layers.{index}"
        self.config = config
        self.index = index
        self.ratio = config.compress_ratios[index]
        self.attention = MlxDeepseekV4Attention(
            model,
            config,
            index,
            self.ratio,
            max_context,
            rope_base,
            rope_compressed,
        )
        self.moe = MlxDeepseekV4MoE(
            model,
            config,
            index,
            available,
            residency,
        )
        self.attention_norm = _array(model, f"{prefix}.attn_norm.weight")
        self.ffn_norm = _array(model, f"{prefix}.ffn_norm.weight")
        self.hc_attn_fn = model.linear(f"{prefix}.hc_attn_fn")
        self.hc_attn_base = _array(model, f"{prefix}.hc_attn_base")
        self.hc_attn_scale = _array(model, f"{prefix}.hc_attn_scale")
        self.hc_ffn_fn = model.linear(f"{prefix}.hc_ffn_fn")
        self.hc_ffn_base = _array(model, f"{prefix}.hc_ffn_base")
        self.hc_ffn_scale = _array(model, f"{prefix}.hc_ffn_scale")

    def _hc_pre(
        self,
        residual: mx.array,
        fn,
        scale: mx.array,
        base: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        config = self.config
        flat = residual.reshape((*residual.shape[:2], config.hc_mult * config.hidden))
        inverse = mx.rsqrt(
            mx.mean(flat.astype(mx.float32) ** 2, axis=-1, keepdims=True) + config.rms_eps
        )
        mixes = fn(flat).astype(mx.float32) * inverse
        if config.fast_hyper_connections:
            return dsv4_hc_pre(
                residual,
                mixes,
                scale,
                base,
                config.hc_sinkhorn_iters,
                config.hc_eps,
            )
        return _hc_pre_generic(
            residual,
            mixes,
            scale,
            base,
            eps=config.hc_eps,
        )

    def _hc_post(
        self,
        branch: mx.array,
        residual: mx.array,
        post: mx.array,
        combination: mx.array,
    ) -> mx.array:
        if self.config.fast_hyper_connections:
            return dsv4_hc_post(branch, residual, post, combination)
        return _hc_post_generic(branch, residual, post, combination)

    def __call__(
        self,
        hidden: mx.array,
        token_ids: mx.array,
        state: _MlxDsv4LayerState,
        *,
        pos0: int,
    ) -> mx.array:
        residual = hidden
        branch, post, combination = self._hc_pre(
            hidden,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        branch = rms_norm(
            branch,
            self.attention_norm,
            self.config.rms_eps,
        )
        branch = self.attention(branch, state, pos0=pos0)
        hidden = self._hc_post(branch, residual, post, combination)
        residual = hidden
        branch, post, combination = self._hc_pre(
            hidden,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        branch = rms_norm(
            branch,
            self.ffn_norm,
            self.config.rms_eps,
        )
        branch = self.moe(branch, token_ids)
        return self._hc_post(branch, residual, post, combination)


class MlxDeepseekV4:
    """Complete native-MFQ DeepSeek-V4 prefill/decode/generation runtime."""

    def __init__(
        self,
        model: MlxNintModel,
        config: MlxDeepseekV4Config,
        *,
        manifest: dict[str, Any],
        max_context: int,
        expert_cache_gb: float = 4.0,
        names: MlxDeepseekV4Names | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.manifest = manifest
        self.names = names or MlxDeepseekV4Names()
        self.max_context = min(
            int(max_context),
            config.max_position_embeddings,
        )
        if self.max_context <= 0:
            raise ValueError("DeepSeek-V4 max_context must be positive")
        self.embedding = model.embedding(self.names.embedding)
        self.output_norm = _array(model, self.names.output_norm)
        self.output = model.linear(self.names.output)
        self.hc_head_fn = model.linear(self.names.hc_head_fn)
        self.hc_head_base = _array(model, self.names.hc_head_base)
        self.hc_head_scale = _array(model, self.names.hc_head_scale)
        self.rope_base = _yarn_tables(
            config.qk_rope_head_dim,
            self.max_context,
            config.rope_theta,
            None,
        )
        self.rope_compressed = _yarn_tables(
            config.qk_rope_head_dim,
            self.max_context,
            config.compress_rope_theta,
            config.rope_scaling or None,
        )
        self._availability = self._expert_availability()
        self.expert_residency = (
            _MlxTpqExpertResidency(
                model.tensors,
                cache_gb=expert_cache_gb,
                experts=config.n_experts,
            )
            if isinstance(model.tensors, io.MMapTensorStore)
            else None
        )
        self.layers: list[_MlxDeepseekV4Layer | None] = [None] * config.n_layers
        self.states: list[_MlxDsv4LayerState] | None = None
        self.batch = 0
        self.position = 0

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        *,
        mmap: bool = True,
        max_context: int = 4096,
        expert_cache_gb: float = 4.0,
    ) -> MlxDeepseekV4:
        header, tensors = io.load_mmap(path) if mmap else io.load(path)
        model = MlxNintModel(tensors)
        try:
            manifest = header.extra.get("tpq_manifest")
            if (
                header.extra.get("source_format") != "tpq-1"
                or not isinstance(manifest, dict)
            ):
                raise ValueError(
                    "DeepSeek-V4 Metal loading requires a native TPQ MFQ file"
                )
            config_text = manifest.get("config")
            if not isinstance(config_text, dict) or "hc_mult" not in config_text:
                raise ValueError("native TPQ manifest is not a DeepSeek-V4 model")
            config = MlxDeepseekV4Config.from_manifest(config_text)
            record_names = (
                set(tensors.records) if isinstance(tensors, io.MMapTensorStore) else set(tensors)
            )
            missing = [
                name for name in MlxDeepseekV4Names.required(config) if name not in record_names
            ]
            if missing:
                preview = ", ".join(missing[:4])
                if len(missing) > 4:
                    preview += f", ... ({len(missing)} missing)"
                raise KeyError("DeepSeek-V4 MFQ is missing required tensors: " + preview)
            return cls(
                model,
                config,
                manifest=manifest,
                max_context=max_context,
                expert_cache_gb=expert_cache_gb,
            )
        except BaseException:
            model.close()
            raise

    def _expert_availability(self) -> tuple[np.ndarray, ...]:
        assignments = self.manifest.get("tiers_per_layer", {})
        result = []
        for layer in range(self.config.n_layers):
            raw = assignments.get(str(layer), assignments.get(layer))
            if raw is None:
                result.append(np.ones((self.config.n_experts,), dtype=np.bool_))
                continue
            tiers = str(raw)
            if len(tiers) != self.config.n_experts:
                raise ValueError(
                    f"DeepSeek-V4 layer {layer} tier assignment length does not match n_experts"
                )
            invalid = set(tiers) - set("xwvVd")
            if invalid:
                raise ValueError(
                    f"DeepSeek-V4 layer {layer} has invalid expert tiers {sorted(invalid)}"
                )
            available = np.fromiter(
                (item != "d" for item in tiers),
                dtype=np.bool_,
                count=self.config.n_experts,
            )
            if int(available.sum()) < self.config.top_k:
                raise ValueError(
                    f"DeepSeek-V4 layer {layer} has fewer than top_k available experts"
                )
            result.append(available)
        return tuple(result)

    def _layer(self, index: int) -> _MlxDeepseekV4Layer:
        layer = self.layers[index]
        if layer is None:
            layer = _MlxDeepseekV4Layer(
                self.model,
                self.config,
                index,
                self._availability[index],
                self.max_context,
                self.rope_base,
                self.rope_compressed,
                self.expert_residency,
            )
            self.layers[index] = layer
        return layer

    def reset_cache(self, batch: int = 1) -> None:
        batch_size = int(batch)
        if batch_size <= 0:
            raise ValueError("DeepSeek-V4 cache batch must be positive")
        self.states = [
            _MlxDsv4LayerState.allocate(
                self.config,
                ratio,
                batch_size,
                self.max_context,
            )
            for ratio in self.config.compress_ratios
        ]
        self.batch = batch_size
        self.position = 0

    def _head(self, hidden: mx.array) -> mx.array:
        config = self.config
        flat = hidden.reshape((*hidden.shape[:2], config.hc_mult * config.hidden))
        inverse = mx.rsqrt(
            mx.mean(flat.astype(mx.float32) ** 2, axis=-1, keepdims=True) + config.rms_eps
        )
        mixes = self.hc_head_fn(flat).astype(mx.float32) * inverse
        pre = (
            mx.sigmoid(mixes * self.hc_head_scale.reshape((-1,))[0] + self.hc_head_base)
            + config.hc_eps
        )
        reduced = mx.sum(pre[..., None] * hidden, axis=2)
        return rms_norm(reduced, self.output_norm, config.rms_eps)

    def _materialize_cache(self) -> None:
        if self.states is None:
            return
        arrays = [array for state in self.states for array in state.arrays()]
        if arrays:
            mx.eval(*arrays)

    def _forward_chunk(
        self,
        ids: mx.array,
        *,
        pos0: int,
    ) -> mx.array:
        assert self.states is not None
        config = self.config
        hidden = self.embedding(ids).astype(mx.float16)
        hidden = mx.broadcast_to(
            hidden[:, :, None, :],
            (
                int(ids.shape[0]),
                int(ids.shape[1]),
                config.hc_mult,
                config.hidden,
            ),
        )
        hidden = mx.contiguous(hidden)
        for index in range(config.n_layers):
            hidden = self._layer(index)(
                hidden,
                ids,
                self.states[index],
                pos0=pos0,
            )
            # Expert slices are mmap-loaded under a global LRU.  Materialize
            # each layer boundary so the lazy graph cannot retain evicted
            # slices from every preceding layer until the final logits.
            mx.eval(hidden, *self.states[index].arrays())
        logits = self.output(self._head(hidden)).astype(mx.float32)
        self._materialize_cache()
        return logits

    def forward(
        self,
        input_ids: mx.array | np.ndarray,
        *,
        use_cache: bool = False,
    ) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        if ids.ndim != 2:
            raise ValueError("DeepSeek-V4 input IDs must have [batch,tokens] shape")
        ids = mx.contiguous(ids.astype(mx.int32))
        batch, tokens = map(int, ids.shape)
        if tokens == 0:
            return mx.zeros((batch, 0, self.config.vocab), dtype=mx.float32)
        if not use_cache or self.states is None or self.batch != batch:
            self.reset_cache(batch)
        if self.position + tokens > self.max_context:
            raise ValueError("DeepSeek-V4 decode exceeds max_context")
        start = self.position
        logits = self._forward_chunk(ids, pos0=start)
        self.position += tokens
        return logits

    def prefill(
        self,
        input_ids: mx.array | np.ndarray,
        *,
        chunk_size: int = 512,
        full_logits: bool = True,
    ) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        if ids.ndim != 2 or int(ids.shape[1]) <= 0:
            raise ValueError("DeepSeek-V4 prefill requires nonempty [batch,tokens] IDs")
        ids = mx.contiguous(ids.astype(mx.int32))
        batch, tokens = map(int, ids.shape)
        size = int(chunk_size)
        if size <= 0:
            raise ValueError("DeepSeek-V4 prefill chunk_size must be positive")
        if tokens > self.max_context:
            raise ValueError("DeepSeek-V4 prefill exceeds max_context")
        self.reset_cache(batch)
        outputs = []
        for start in range(0, tokens, size):
            chunk = self._forward_chunk(
                ids[:, start : start + size],
                pos0=start,
            )
            outputs.append(chunk)
            self.position = min(tokens, start + size)
        logits = mx.concatenate(outputs, axis=1)
        return logits if full_logits else logits[:, -1]

    def decode(self, input_ids: mx.array | np.ndarray) -> mx.array:
        if self.states is None:
            raise RuntimeError("DeepSeek-V4 decode requires prefill first")
        return self.forward(input_ids, use_cache=True)

    def generate(
        self,
        input_ids: mx.array | np.ndarray,
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | tuple[int, ...] | None = None,
        chunk_size: int = 512,
    ) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        if ids.ndim != 2 or int(ids.shape[1]) <= 0:
            raise ValueError("DeepSeek-V4 generation requires a nonempty prompt")
        if int(max_new_tokens) < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not 0.0 < float(top_p) <= 1.0:
            raise ValueError("top_p must be in (0,1]")
        ids = mx.contiguous(ids.astype(mx.int32))
        if int(max_new_tokens) == 0:
            return ids
        if int(ids.shape[1]) + int(max_new_tokens) > self.max_context:
            raise ValueError("DeepSeek-V4 generation exceeds max_context")
        logits = self.prefill(
            ids,
            chunk_size=chunk_size,
            full_logits=False,
        )
        pieces = [ids]
        next_id = _sample(
            logits,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
        )
        eos = (
            self.config.eos_token_id
            if eos_token_id is None
            else (
                (int(eos_token_id),)
                if isinstance(eos_token_id, int)
                else tuple(int(item) for item in eos_token_id)
            )
        )
        for step in range(int(max_new_tokens)):
            pieces.append(next_id[:, None])
            if eos:
                mx.eval(next_id)
                if np.isin(np.asarray(next_id), eos).all():
                    break
            if step + 1 < int(max_new_tokens):
                logits = self.decode(next_id[:, None])[:, -1]
                next_id = _sample(
                    logits,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                )
        return mx.concatenate(pieces, axis=1)

    def close(self) -> None:
        if self.expert_residency is not None:
            self.expert_residency.clear()
        self.model.close()

    def __enter__(self) -> MlxDeepseekV4:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __call__(
        self,
        input_ids: mx.array | np.ndarray,
        *,
        use_cache: bool = False,
    ) -> mx.array:
        return self.forward(input_ids, use_cache=use_cache)


__all__ = [
    "MlxDeepseekV4",
    "MlxDeepseekV4Attention",
    "MlxDeepseekV4Config",
    "MlxDeepseekV4MoE",
    "MlxDeepseekV4Names",
    "MlxDeepseekV4PoolState",
]
