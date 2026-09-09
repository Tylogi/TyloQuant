"""Qwen3.8-Flash-Next (``qwen4_exp``) MFQ runtime for Apple silicon."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's MLX runtime requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.architectures.flash_next import Qwen4ExpConfig
from mfq.formats.assets import MODEL_CONFIG_ASSET
from mfq.formats.io import MfqTensor
from mfq.kernels.metal.flash_next import (
    qsa_block_scores,
    qwen4_dense_gqa_attention,
    qwen4_gated_residual_post,
    qwen4_gated_residual_pre,
    qwen4_grouped_rms_norm,
    qwen4_ple_dilated_conv_silu,
    qwen4_sparse_gqa_attention,
)
from mfq.kernels.metal.linear_attention import gated_delta_net, linear_conv_qkv
from mfq.kernels.metal.moe_ops import moe_topk, weighted_reduce
from mfq.kernels.metal.sampling import sample as _sample
from mfq.runtime.mlx_attention import MlxKVCache
from mfq.runtime.mlx_linear import (
    MlxLinearGroup,
    MlxNintModel,
    MlxShardedEmbedding,
    mlx_dense_array,
)
from mfq.runtime.mlx_moe import load_routed_gate_up
from mfq.runtime.mlx_ops import MlxRMSNorm, MlxRoPE


@dataclass(frozen=True)
class MlxQwen4ExpNames:
    token_embedding: str = "model.token_embedding.weight"
    output: str = "model.output.weight"
    final_mixer: str = "model.mhc.pre"
    layer_prefix: str = "model.block.{i}"

    def layer(self, index: int) -> str:
        return self.layer_prefix.format(i=index)


def _dense_array(
    model: MlxNintModel,
    name: str,
    *,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    if name not in model.tensors:
        raise KeyError(f"tensor {name!r} is not present in the Qwen4-Exp model")
    value = model.tensors[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Qwen4-Exp tensor {name!r} must be dense")
    return mx.contiguous(mlx_dense_array(value, dtype=dtype))


def _dense_vector(model: MlxNintModel, name: str) -> mx.array:
    value = _dense_array(model, name, dtype=mx.float32)
    if value.ndim != 1:
        raise TypeError(f"Qwen4-Exp tensor {name!r} must be a vector")
    return value


def _numpy_integer_array(model: MlxNintModel, name: str) -> np.ndarray:
    if name not in model.tensors:
        raise KeyError(f"tensor {name!r} is not present in the Qwen4-Exp model")
    value = model.tensors[name]
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "iu":
        raise TypeError(f"Qwen4-Exp tensor {name!r} must be a dense integer array")
    return np.ascontiguousarray(value)


def _embedding_shape(embedding: object) -> tuple[int, int]:
    out = getattr(embedding, "out", None)
    width = getattr(embedding, "neuron_len", None)
    if out is not None and width is not None:
        return int(out), int(width)
    packed = getattr(embedding, "packed_weight", None)
    if packed is not None:
        out = getattr(packed, "out", None)
        width = getattr(packed, "neuron_len", None)
        if width is None:
            width = getattr(packed, "in_features", None)
        if out is not None and width is not None:
            return int(out), int(width)
    weight = getattr(embedding, "weight", None)
    if isinstance(weight, mx.array) and weight.ndim == 2:
        return int(weight.shape[0]), int(weight.shape[1])
    raise TypeError("Qwen4-Exp embedding runtime does not expose a matrix shape")


class _MlxQwenSequenceCache:
    def __init__(self, max_sequence: int, width: int) -> None:
        self.max_sequence = int(max_sequence)
        self.width = int(width)
        self.values: mx.array | None = None
        self.batch = 0
        self.position = 0

    def reset(self, batch: int, *, initial_capacity: int = 16) -> None:
        self.batch = int(batch)
        if self.batch <= 0:
            raise ValueError("Qwen4-Exp cache batch must be positive")
        capacity = min(max(1, int(initial_capacity)), self.max_sequence)
        self.values = mx.zeros(
            (self.batch, capacity, self.width),
            dtype=mx.float16,
        )
        self.position = 0

    def _ensure(self, required: int) -> None:
        if self.values is None:
            raise RuntimeError("Qwen4-Exp cache was not initialized")
        if required <= int(self.values.shape[1]):
            return
        if required > self.max_sequence:
            raise ValueError("Qwen4-Exp cache exceeds max_context")
        capacity = int(self.values.shape[1])
        while capacity < required:
            capacity = min(self.max_sequence, capacity * 2)
        expanded = mx.zeros(
            (self.batch, capacity, self.width),
            dtype=self.values.dtype,
        )
        self.values = mx.slice_update(
            expanded,
            self.values,
            start_indices=mx.array([0, 0, 0], dtype=mx.int32),
            axes=(0, 1, 2),
        )

    def append(self, value: mx.array) -> tuple[mx.array, int]:
        if value.ndim != 3 or int(value.shape[-1]) != self.width:
            raise ValueError("Qwen4-Exp cache append shape mismatch")
        batch, tokens = (int(item) for item in value.shape[:2])
        if self.values is None or self.batch != batch:
            self.reset(batch, initial_capacity=max(16, tokens))
        start = self.position
        end = start + tokens
        self._ensure(end)
        assert self.values is not None
        self.values = mx.slice_update(
            self.values,
            value.astype(self.values.dtype),
            start_indices=mx.array([0, start, 0], dtype=mx.int32),
            axes=(0, 1, 2),
        )
        self.position = end
        return self.values[:, :end], start


class MlxQwen4ExpGatedResidual:
    """One Qwen4-Exp four-stream gated-residual site."""

    def __init__(
        self,
        model: MlxNintModel,
        config: Qwen4ExpConfig,
        prefix: str,
        *,
        combine: bool = True,
    ) -> None:
        self.config = config
        self.norm = _dense_vector(model, prefix + ".norm.weight")
        self.down = _dense_array(model, prefix + ".down.weight")
        self.up = _dense_array(model, prefix + ".up.weight")
        injection_name = prefix.removesuffix(".pre") + ".post.inject.weight"
        if combine and injection_name not in model.tensors:
            raise KeyError(f"Qwen4-Exp gated residual lacks {injection_name!r}")
        self.injection = _dense_array(model, injection_name) if combine else None

    def pre(self, value: mx.array) -> tuple[mx.array, mx.array, mx.array | None]:
        return qwen4_gated_residual_pre(
            value,
            self.norm,
            self.down,
            self.up,
            self.injection,
            hidden_size=self.config.hidden_size,
            hc_count=self.config.hc_count,
            eps=self.config.rms_norm_eps,
        )

    def post(
        self,
        branch: mx.array,
        residual: mx.array,
        injection: mx.array | None,
    ) -> mx.array:
        if injection is None:
            raise ValueError("Qwen4-Exp branch injection weights are absent")
        return qwen4_gated_residual_post(
            branch,
            residual,
            injection,
            hc_count=self.config.hc_count,
        )

    def mix(self, value: mx.array) -> mx.array:
        mixed, _residual, injection = self.pre(value)
        if injection is not None:
            raise ValueError("Qwen4-Exp final mixer unexpectedly has an injection branch")
        return mixed


class MlxQwen4ExpDenseFFN:
    def __init__(self, model: MlxNintModel, prefix: str) -> None:
        self.ffn = model.ffn(
            prefix + ".gate.weight",
            prefix + ".up.weight",
            prefix + ".down.weight",
        )

    def __call__(self, value: mx.array) -> mx.array:
        return self.ffn(value)


class MlxQwen4ExpMoE:
    """Qwen softmax top-k experts plus sigmoid-gated shared expert."""

    def __init__(
        self,
        model: MlxNintModel,
        config: Qwen4ExpConfig,
        prefix: str,
    ) -> None:
        gate_up = load_routed_gate_up(model, prefix)
        down = model.routed(prefix + ".experts.down.weight")
        if (
            gate_up.n_experts != config.num_experts
            or down.n_experts != config.num_experts
            or gate_up.neuron_len != config.hidden_size
            or gate_up.out_per_expert != 2 * config.moe_intermediate_size
            or down.neuron_len != config.moe_intermediate_size
            or down.out_per_expert != config.hidden_size
        ):
            raise ValueError("Qwen4-Exp routed expert tensor shapes disagree")
        self.gate_up = gate_up
        self.down = down
        self.router = model.linear(prefix + ".router.weight")
        self.shared = MlxQwen4ExpDenseFFN(model, prefix + ".shared_expert")
        self.shared_gate = model.linear(prefix + ".shared_expert.router.weight")
        self.config = config

    def __call__(self, value: mx.array) -> mx.array:
        config = self.config
        source = value.reshape((-1, config.hidden_size)).astype(mx.float16)
        ids, weights = moe_topk(
            self.router(source),
            config.num_experts_per_tok,
            normalize=config.norm_topk_prob,
        )
        gate, up = mx.split(self.gate_up(source, ids), 2, axis=-1)
        routed = weighted_reduce(
            self.down((gate * mx.sigmoid(gate)) * up, ids),
            weights,
        )
        shared = mx.sigmoid(self.shared_gate(source)) * self.shared(source)
        return (routed + shared).reshape(value.shape)


class MlxQwen4ExpGdn:
    """Qwen4-Exp Gated DeltaNet with its native head counts and output gate."""

    def __init__(
        self,
        model: MlxNintModel,
        config: Qwen4ExpConfig,
        prefix: str,
    ) -> None:
        self.config = config
        self.qkv = model.linear(prefix + ".qkv.weight")
        self.zab = MlxLinearGroup(
            (
                model.linear(prefix + ".gate.weight"),
                model.linear(prefix + ".alpha.weight"),
                model.linear(prefix + ".beta.weight"),
            )
        )
        self.conv_weight = _dense_array(model, prefix + ".conv.weight")
        self.dt_bias = _dense_vector(model, prefix + ".dt_bias")
        self.a_log = _dense_vector(model, prefix + ".a")
        self.output_norm = MlxRMSNorm(
            _dense_vector(model, prefix + ".norm.weight"),
            config.rms_norm_eps,
        )
        self.output = model.linear(prefix + ".output.weight")
        self.conv_state: mx.array | None = None
        self.recurrent_state: mx.array | None = None
        self.rollback_state: tuple[mx.array, mx.array] | None = None
        self.batch = 0

    @property
    def key_width(self) -> int:
        return self.config.linear_num_key_heads * self.config.linear_key_head_dim

    @property
    def value_width(self) -> int:
        return self.config.linear_num_value_heads * self.config.linear_value_head_dim

    def reset_cache(self, batch: int) -> None:
        config = self.config
        selected = int(batch)
        self.conv_state = mx.zeros(
            (
                selected,
                config.linear_conv_kernel_dim - 1,
                2 * self.key_width + self.value_width,
            ),
            dtype=mx.float32,
        )
        self.recurrent_state = mx.zeros(
            (
                selected,
                config.linear_num_value_heads,
                config.linear_value_head_dim,
                config.linear_value_head_dim,
            ),
            dtype=mx.float32,
        )
        self.rollback_state = None
        self.batch = selected

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        use_cache: bool,
        n_confirmed: int = 0,
    ) -> mx.array:
        config = self.config
        batch, tokens = (int(item) for item in hidden_states.shape[:2])
        confirmed = int(n_confirmed)
        if not 0 <= confirmed <= tokens:
            raise ValueError("Qwen4-Exp confirmed-token count is outside the input")
        if confirmed and not use_cache:
            raise ValueError("Qwen4-Exp speculative verification requires a cache")
        if 0 < confirmed < tokens:
            accepted = self(
                hidden_states[:, :confirmed],
                use_cache=True,
            )
            assert self.conv_state is not None and self.recurrent_state is not None
            self.rollback_state = (self.conv_state, self.recurrent_state)
            speculative = self(
                hidden_states[:, confirmed:],
                use_cache=True,
            )
            return mx.concatenate((accepted, speculative), axis=1)
        projected = self.qkv(hidden_states)
        qk, value_input = mx.split(projected, [2 * self.key_width], axis=-1)
        z, alpha, beta = self.zab(hidden_states)
        beta = mx.sigmoid(beta.astype(mx.float32)).reshape(
            batch,
            tokens,
            config.linear_num_value_heads,
        )
        alpha = alpha.astype(mx.float32).reshape(beta.shape)
        gate_input = alpha + self.dt_bias.reshape(1, 1, -1)
        softplus = mx.maximum(gate_input, 0.0) + mx.log1p(mx.exp(-mx.abs(gate_input)))
        decay = -mx.exp(self.a_log).reshape(1, 1, -1) * softplus
        if use_cache:
            if self.conv_state is None or self.batch != batch:
                self.reset_cache(batch)
            assert self.conv_state is not None and self.recurrent_state is not None
            conv_state = self.conv_state
            recurrent_state = self.recurrent_state
        else:
            conv_state = mx.zeros(
                (
                    batch,
                    config.linear_conv_kernel_dim - 1,
                    2 * self.key_width + self.value_width,
                ),
                dtype=mx.float32,
            )
            recurrent_state = None
        query, key, value, next_conv_state = linear_conv_qkv(
            conv_state,
            qk,
            value_input,
            self.conv_weight,
            num_key_heads=config.linear_num_key_heads,
            num_value_heads=config.linear_num_value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
            eps=1e-6,
        )
        attended, next_recurrent_state = gated_delta_net(
            query,
            key,
            value,
            mx.transpose(decay, (0, 2, 1)),
            mx.transpose(beta, (0, 2, 1)),
            recurrent_state,
        )
        if use_cache:
            self.conv_state = next_conv_state
            self.recurrent_state = next_recurrent_state
        normalized = self.output_norm(attended)
        output_gate = mx.transpose(
            z.reshape(
                batch,
                tokens,
                config.linear_num_value_heads,
                config.linear_value_head_dim,
            ),
            (0, 2, 1, 3),
        ).astype(mx.float32)
        if config.output_gate_type == "sigmoid":
            normalized = normalized * mx.sigmoid(output_gate)
        else:
            normalized = normalized * (output_gate * mx.sigmoid(output_gate))
        normalized = mx.transpose(normalized, (0, 2, 1, 3)).reshape(
            batch,
            tokens,
            self.value_width,
        )
        return self.output(normalized.astype(hidden_states.dtype))

    def commit_speculative_cache(self) -> None:
        self.rollback_state = None

    def rollback_speculative_cache(self) -> None:
        if self.rollback_state is None:
            raise RuntimeError("Qwen4-Exp GDN has no speculative rollback state")
        self.conv_state, self.recurrent_state = self.rollback_state
        self.rollback_state = None


class MlxQwen4ExpQsa:
    """Qwen sparse GQA with four-token mean-pooled index keys."""

    def __init__(
        self,
        model: MlxNintModel,
        config: Qwen4ExpConfig,
        prefix: str,
        max_context: int,
    ) -> None:
        self.config = config
        self.qkv = MlxLinearGroup(
            (
                model.linear(prefix + ".query.weight"),
                model.linear(prefix + ".key.weight"),
                model.linear(prefix + ".value.weight"),
            )
        )
        self.q_norm = MlxRMSNorm(
            _dense_vector(model, prefix + ".query_norm.weight"),
            config.rms_norm_eps,
            weight_offset=1.0,
        )
        self.k_norm = MlxRMSNorm(
            _dense_vector(model, prefix + ".key_norm.weight"),
            config.rms_norm_eps,
            weight_offset=1.0,
        )
        self.output = model.linear(prefix + ".output.weight")
        self.index_qk = model.linear(prefix + ".indexer.query_key.weight")
        self.index_q_norm = MlxRMSNorm(
            _dense_vector(model, prefix + ".indexer.query_norm.weight"),
            config.rms_norm_eps,
            weight_offset=1.0,
        )
        self.index_k_norm = MlxRMSNorm(
            _dense_vector(model, prefix + ".indexer.key_norm.weight"),
            config.rms_norm_eps,
            weight_offset=1.0,
        )
        self.rope = MlxRoPE(
            config.rotary_dim,
            max_context,
            base=config.rope_theta,
            sections=config.rope_sections,
            mrope_interleaved=config.mrope_interleaved,
        )
        self.cache: MlxKVCache | None = None
        self.index_cache = _MlxQwenSequenceCache(max_context, config.indexer_head_dim)
        self.batch = 0
        self.max_context = int(max_context)

    def reset_cache(self, batch: int) -> None:
        selected = int(batch)
        self.cache = MlxKVCache(
            selected,
            self.config.num_key_value_heads,
            self.max_context,
            self.config.head_dim,
        )
        self.index_cache.reset(selected)
        self.batch = selected

    def _selected_indices(
        self,
        query: mx.array,
        raw_keys: mx.array,
        positions_full: mx.array,
        query_offset: int,
    ) -> mx.array:
        config = self.config
        batch, tokens = (int(item) for item in query.shape[:2])
        ratio = config.indexer_compress_ratio
        complete = int(raw_keys.shape[1]) // ratio
        select_count = min(config.indexer_budget // ratio, complete)
        absolute = mx.arange(tokens, dtype=mx.int32) + int(query_offset)
        if select_count:
            pooled = mx.mean(
                raw_keys[:, : complete * ratio]
                .reshape(
                    batch,
                    complete,
                    ratio,
                    config.indexer_head_dim,
                )
                .astype(mx.float32),
                axis=-2,
            ).astype(raw_keys.dtype)
            pooled = self.index_k_norm(pooled)
            starts = mx.arange(complete, dtype=mx.int32) * ratio
            block_positions = positions_full[..., starts]
            pooled = self.rope(pooled[:, None], block_positions)[:, 0]
            scores = qsa_block_scores(query, pooled)
            ends = starts + ratio - 1
            visible = ends[None, None, :] <= absolute[None, :, None]
            visible = mx.broadcast_to(visible, (batch, tokens, complete))
            scores = mx.where(visible, scores, mx.array(-1e30, dtype=mx.float32))
            partition = mx.argpartition(scores, complete - select_count, axis=-1)
            blocks = partition[..., -select_count:].astype(mx.int32)
            valid = mx.take_along_axis(visible, blocks, axis=-1)
            offsets = mx.arange(ratio, dtype=mx.int32)
            selected = (blocks[..., None] * ratio + offsets).reshape(batch, tokens, -1)
            valid = mx.broadcast_to(valid[..., None], (*valid.shape, ratio)).reshape(
                batch,
                tokens,
                -1,
            )
            selected = mx.where(valid, selected, mx.array(-1, dtype=mx.int32))
        else:
            selected = mx.full((batch, tokens, 0), -1, dtype=mx.int32)
        width = int(selected.shape[-1])
        if width < config.indexer_budget:
            selected = mx.concatenate(
                (
                    selected,
                    mx.full(
                        (batch, tokens, config.indexer_budget - width),
                        -1,
                        dtype=mx.int32,
                    ),
                ),
                axis=-1,
            )
        else:
            selected = selected[..., : config.indexer_budget]
        tail_width = ratio - 1
        if tail_width:
            tail_count = (absolute + 1) % ratio
            tail_start = absolute + 1 - tail_count
            tail_offsets = mx.arange(tail_width, dtype=mx.int32)
            tail = tail_start[:, None] + tail_offsets[None]
            tail = mx.where(
                tail_offsets[None] < tail_count[:, None],
                tail,
                mx.array(-1, dtype=mx.int32),
            )
            selected = mx.concatenate(
                (selected, mx.broadcast_to(tail[None], (batch, tokens, tail_width))),
                axis=-1,
            )
        return selected

    def __call__(
        self,
        hidden_states: mx.array,
        positions_current: mx.array,
        positions_full: mx.array,
        *,
        use_cache: bool,
    ) -> mx.array:
        config = self.config
        batch, tokens = (int(item) for item in hidden_states.shape[:2])
        query_full, key_full, value_full = self.qkv(hidden_states)
        query_pair = query_full.reshape(
            batch,
            tokens,
            config.num_attention_heads,
            2 * config.head_dim,
        )
        query, output_gate = mx.split(query_pair, 2, axis=-1)
        query = mx.transpose(self.q_norm(query), (0, 2, 1, 3))
        key = mx.transpose(
            self.k_norm(
                key_full.reshape(
                    batch,
                    tokens,
                    config.num_key_value_heads,
                    config.head_dim,
                )
            ),
            (0, 2, 1, 3),
        )
        value = mx.transpose(
            value_full.reshape(
                batch,
                tokens,
                config.num_key_value_heads,
                config.head_dim,
            ),
            (0, 2, 1, 3),
        )
        query = self.rope(query, positions_current)
        key = self.rope(key, positions_current)
        index_qk = self.index_qk(hidden_states)
        index_q, raw_key = mx.split(
            index_qk,
            [config.indexer_n_heads * config.indexer_head_dim],
            axis=-1,
        )
        index_q = self.index_q_norm(
            index_q.reshape(
                batch,
                tokens,
                config.indexer_n_heads,
                config.indexer_head_dim,
            )
        )
        index_q = self.rope.forward(index_q, positions_current, sequence_axis=1)
        raw_key = raw_key.reshape(batch, tokens, config.indexer_head_dim)
        if use_cache:
            if self.cache is None or self.batch != batch:
                self.reset_cache(batch)
            assert self.cache is not None
            query_offset = self.cache.pos
            key_cache, value_cache = self.cache.append(key, value)
            raw_key_cache, index_offset = self.index_cache.append(raw_key)
            if index_offset != query_offset:
                raise RuntimeError("Qwen4-Exp attention/index caches diverged")
        else:
            query_offset = 0
            key_cache, value_cache = key, value
            raw_key_cache = raw_key
        logical_length = int(key_cache.shape[2])
        if logical_length <= config.indexer_budget:
            attended = qwen4_dense_gqa_attention(
                query,
                key_cache,
                value_cache,
                query_offset=query_offset,
            )
        else:
            selected = self._selected_indices(
                index_q,
                raw_key_cache,
                positions_full,
                query_offset,
            )
            attended = qwen4_sparse_gqa_attention(
                query,
                key_cache,
                value_cache,
                selected,
            )
        attended = attended.reshape(
            batch,
            tokens,
            config.num_attention_heads * config.head_dim,
        )
        output_gate = output_gate.reshape(attended.shape)
        gated = attended.astype(mx.float32) * mx.sigmoid(output_gate.astype(mx.float32))
        return self.output(gated.astype(hidden_states.dtype))


class MlxQwen4ExpNgramEmbedding:
    """Lookup Qwen's sharded PLE table with row-selective mmap residency."""

    def __init__(
        self,
        model: MlxNintModel,
        config: Qwen4ExpConfig,
        prefix: str,
    ) -> None:
        metadata_prefix = prefix + ".ngram"
        embedding_prefix = metadata_prefix + ".shard"
        self.embeddings = tuple(
            model.embedding(
                f"{embedding_prefix}.{index}.weight",
                residency="mmap",
            )
            for index in range(config.split_ngram_parts)
        )
        shapes = tuple(_embedding_shape(embedding) for embedding in self.embeddings)
        if len(set(shapes)) != 1 or len(shapes[0]) != 2:
            raise ValueError("Qwen4-Exp PLE embedding shards must have one common shape")
        self.rows_per_shard, self.head_dimension = shapes[0]
        self.embedding_table = MlxShardedEmbedding(self.embeddings)
        storage_dtypes = {
            getattr(embedding, "storage_dtype", None)
            for embedding in self.embeddings
        }
        if "F8_E4M3" in storage_dtypes and storage_dtypes != {"F8_E4M3"}:
            raise ValueError("Qwen4-Exp PLE shards cannot mix raw E4M3 and other dtypes")
        self.raw_e4m3 = storage_dtypes == {"F8_E4M3"}
        self.weight_scale = (
            _dense_vector(model, metadata_prefix + ".weight_scale")
            if self.raw_e4m3
            else None
        )
        if self.weight_scale is not None and int(self.weight_scale.size) != 1:
            raise ValueError("Qwen4-Exp PLE E4M3 scale must contain one value")
        self.layer_multipliers = _numpy_integer_array(
            model,
            metadata_prefix + ".layer_multipliers",
        ).astype(np.int64)
        self.layer_multipliers_u64 = self.layer_multipliers.view(np.uint64)
        self.head_offsets = _numpy_integer_array(
            model,
            metadata_prefix + ".head_offsets",
        ).astype(np.int64)
        self.head_vocab_sizes = _numpy_integer_array(
            model,
            metadata_prefix + ".head_vocab_sizes",
        ).astype(np.int64)
        self.config = config
        self.context: np.ndarray | None = None
        self.last_touched_shards: tuple[int, ...] = ()
        self.rows_read = 0
        self.logical_bytes_read = 0
        expected_heads = (config.ngram_size - 1) * config.heads_per_ngram
        if (
            self.layer_multipliers.shape != (config.ngram_size,)
            or self.head_offsets.shape != (expected_heads,)
            or self.head_vocab_sizes.shape != (expected_heads,)
            or self.head_dimension * expected_heads != config.ple_embed_dim
        ):
            raise ValueError("Qwen4-Exp PLE hash metadata dimensions disagree")

    def reset_cache(self, batch: int) -> None:
        eos = self.config.eos_token_ids[0]
        self.context = np.full(
            (int(batch), self.config.ngram_size - 1),
            eos,
            dtype=np.int64,
        )

    def _shift(self, ids: np.ndarray, shift: int) -> np.ndarray:
        if shift == 0:
            return ids
        eos = self.config.eos_token_ids[0]
        batch, tokens = ids.shape
        result = np.full((batch, tokens), eos, dtype=np.int64)
        for batch_index in range(batch):
            segment_start = 0
            for token in range(tokens):
                if token - shift >= segment_start:
                    result[batch_index, token] = ids[batch_index, token - shift]
                if ids[batch_index, token] == eos:
                    segment_start = token + 1
        return result

    def _ids(self, input_ids: np.ndarray, *, use_cache: bool) -> np.ndarray:
        batch, tokens = input_ids.shape
        if use_cache:
            if self.context is None or self.context.shape[0] != batch:
                self.reset_cache(batch)
            assert self.context is not None
            history = np.concatenate((self.context, input_ids), axis=1)
            self.context = history[:, -(self.config.ngram_size - 1) :].copy()
        else:
            eos = self.config.eos_token_ids[0]
            history = np.concatenate(
                (
                    np.full(
                        (batch, self.config.ngram_size - 1),
                        eos,
                        dtype=np.int64,
                    ),
                    input_ids,
                ),
                axis=1,
            )
        shifted = tuple(self._shift(history, shift) for shift in range(self.config.ngram_size))
        blocks = []
        for ngram in range(2, self.config.ngram_size + 1):
            start = (ngram - 2) * self.config.heads_per_ngram
            end = start + self.config.heads_per_ngram
            mixed = shifted[0].astype(np.uint64) * self.layer_multipliers_u64[0]
            for position in range(1, ngram):
                mixed = np.bitwise_xor(
                    mixed,
                    shifted[position].astype(np.uint64) * self.layer_multipliers_u64[position],
                )
            signed = mixed.view(np.int64)
            values = np.remainder(
                signed[..., None],
                self.head_vocab_sizes[start:end][None, None],
            )
            blocks.append(values + self.head_offsets[start:end][None, None])
        return np.concatenate(blocks, axis=-1)[:, -tokens:]

    def __call__(self, input_ids: mx.array, *, use_cache: bool) -> mx.array:
        mx.eval(input_ids)
        host_ids = np.asarray(input_ids, dtype=np.int64)
        global_ids = self._ids(host_ids, use_cache=use_cache)
        result = self.embedding_table(global_ids)
        if self.weight_scale is not None:
            result = result * self.weight_scale.astype(result.dtype).reshape(())
        self.last_touched_shards = self.embedding_table.last_touched_shards
        self.rows_read = self.embedding_table.rows_read
        self.logical_bytes_read = self.embedding_table.logical_bytes_read
        return result.reshape((*global_ids.shape[:2], -1))


class MlxQwen4ExpPle:
    """Per-layer hashed lexical embedding and dilated-convolution injection."""

    def __init__(
        self,
        model: MlxNintModel,
        config: Qwen4ExpConfig,
        prefix: str,
    ) -> None:
        self.config = config
        self.embedding = MlxQwen4ExpNgramEmbedding(model, config, prefix)
        self.key = model.linear(prefix + ".key.weight")
        self.value = model.linear(prefix + ".value.weight")
        self.norm_key = _dense_vector(model, prefix + ".key_norm.weight")
        self.norm_query = _dense_vector(model, prefix + ".query_norm.weight")
        self.norm_conv = _dense_vector(model, prefix + ".conv_norm.weight")
        self.conv_weight = _dense_array(model, prefix + ".conv.weight")
        self.conv_state: mx.array | None = None
        self.rollback_state: tuple[mx.array | None, np.ndarray | None] | None = None
        self.batch = 0

    def reset_cache(self, batch: int) -> None:
        self.embedding.reset_cache(batch)
        self.conv_state = None
        self.rollback_state = None
        self.batch = int(batch)

    def __call__(
        self,
        hidden_streams: mx.array,
        input_ids: mx.array,
        *,
        use_cache: bool,
        n_confirmed: int = 0,
    ) -> mx.array:
        config = self.config
        batch, tokens = (int(item) for item in input_ids.shape)
        confirmed = int(n_confirmed)
        if not 0 <= confirmed <= tokens:
            raise ValueError("Qwen4-Exp PLE confirmed-token count is outside the input")
        if confirmed and not use_cache:
            raise ValueError("Qwen4-Exp speculative PLE verification requires a cache")
        if 0 < confirmed < tokens:
            accepted = self(
                hidden_streams[:, :confirmed],
                input_ids[:, :confirmed],
                use_cache=True,
            )
            self.rollback_state = (self.conv_state, self.embedding.context)
            speculative = self(
                hidden_streams[:, confirmed:],
                input_ids[:, confirmed:],
                use_cache=True,
            )
            return mx.concatenate((accepted, speculative), axis=1)
        if use_cache and self.batch != batch:
            self.reset_cache(batch)
        embeddings = self.embedding(input_ids, use_cache=use_cache)
        key = qwen4_grouped_rms_norm(
            self.key(embeddings),
            self.norm_key,
            config.hidden_size,
            config.rms_norm_eps,
        ).reshape(batch, tokens, config.hc_count, config.hidden_size)
        query = qwen4_grouped_rms_norm(
            hidden_streams,
            self.norm_query,
            config.hidden_size,
            config.rms_norm_eps,
        ).reshape(batch, tokens, config.hc_count, config.hidden_size)
        value = self.value(embeddings)
        gate = mx.sum(key.astype(mx.float32) * query.astype(mx.float32), axis=-1)
        gate = gate / math.sqrt(config.hidden_size)
        signed_root = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated = mx.sigmoid(signed_root)[..., None] * value[..., None, :]
        gated = gated.reshape(batch, tokens, config.hc_count * config.hidden_size)
        normalized = qwen4_grouped_rms_norm(
            gated,
            self.norm_conv,
            config.hidden_size,
            config.rms_norm_eps,
        )
        conv_state = self.conv_state if use_cache else None
        convolved, next_state = qwen4_ple_dilated_conv_silu(
            normalized,
            self.conv_weight,
            conv_state,
            dilation=config.ngram_size,
        )
        if use_cache:
            self.conv_state = next_state
        return (gated + convolved).astype(hidden_streams.dtype)

    def commit_speculative_cache(self) -> None:
        self.rollback_state = None

    def rollback_speculative_cache(self) -> None:
        if self.rollback_state is None:
            raise RuntimeError("Qwen4-Exp PLE has no speculative rollback state")
        self.conv_state, self.embedding.context = self.rollback_state
        self.rollback_state = None


class MlxQwen4ExpLayer:
    def __init__(
        self,
        model: MlxNintModel,
        config: Qwen4ExpConfig,
        names: MlxQwen4ExpNames,
        index: int,
        max_context: int,
    ) -> None:
        prefix = names.layer(index)
        self.attention_gr = MlxQwen4ExpGatedResidual(
            model,
            config,
            prefix + ".attention.mhc.pre",
        )
        self.ffn_gr = MlxQwen4ExpGatedResidual(
            model,
            config,
            prefix + ".mlp.mhc.pre",
        )
        self.attention = (
            MlxQwen4ExpGdn(model, config, prefix + ".linear_attention")
            if config.layer_types[index] == "linear_attention"
            else MlxQwen4ExpQsa(
                model,
                config,
                prefix + ".attention",
                max_context,
            )
        )
        self.moe = MlxQwen4ExpMoE(model, config, prefix + ".mlp")
        self.ple = (
            MlxQwen4ExpPle(model, config, prefix + ".position_embedding")
            if index + 1 in config.ple_layer_ids
            else None
        )

    def reset_cache(self, batch: int) -> None:
        self.attention.reset_cache(batch)
        if self.ple is not None:
            self.ple.reset_cache(batch)

    def __call__(
        self,
        hidden_streams: mx.array,
        input_ids: mx.array,
        positions_current: mx.array,
        positions_full: mx.array,
        *,
        use_cache: bool,
        n_confirmed: int = 0,
    ) -> mx.array:
        if self.ple is not None:
            hidden_streams = hidden_streams + self.ple(
                hidden_streams,
                input_ids,
                use_cache=use_cache,
                n_confirmed=n_confirmed,
            )
        branch, residual, injection = self.attention_gr.pre(hidden_streams)
        if isinstance(self.attention, MlxQwen4ExpGdn):
            branch = self.attention(
                branch,
                use_cache=use_cache,
                n_confirmed=n_confirmed,
            )
        else:
            branch = self.attention(
                branch,
                positions_current,
                positions_full,
                use_cache=use_cache,
            )
        hidden_streams = self.attention_gr.post(branch, residual, injection)
        branch, residual, injection = self.ffn_gr.pre(hidden_streams)
        branch = self.moe(branch)
        return self.ffn_gr.post(branch, residual, injection)


class MlxQwen4ExpMtpLayer:
    """One text-only Qwen4-Exp MTP layer (full QSA, no PLE)."""

    def __init__(
        self,
        model: MlxNintModel,
        config: Qwen4ExpConfig,
        prefix: str,
        max_context: int,
    ) -> None:
        self.attention_gr = MlxQwen4ExpGatedResidual(
            model,
            config,
            prefix + ".attention.mhc.pre",
        )
        self.ffn_gr = MlxQwen4ExpGatedResidual(
            model,
            config,
            prefix + ".mlp.mhc.pre",
        )
        self.attention = MlxQwen4ExpQsa(
            model,
            config,
            prefix + ".attention",
            max_context,
        )
        self.moe = MlxQwen4ExpMoE(model, config, prefix + ".mlp")

    def reset_cache(self, batch: int) -> None:
        self.attention.reset_cache(batch)

    def __call__(
        self,
        hidden_streams: mx.array,
        positions_current: mx.array,
        positions_full: mx.array,
        *,
        use_cache: bool,
    ) -> mx.array:
        branch, residual, injection = self.attention_gr.pre(hidden_streams)
        branch = self.attention(
            branch,
            positions_current,
            positions_full,
            use_cache=use_cache,
        )
        hidden_streams = self.attention_gr.post(branch, residual, injection)
        branch, residual, injection = self.ffn_gr.pre(hidden_streams)
        branch = self.moe(branch)
        return self.ffn_gr.post(branch, residual, injection)


class MlxQwen4ExpMtp:
    """Native Qwen3.8-Flash-Next multi-stream MTP draft runtime."""

    def __init__(
        self,
        tensors: Mapping[str, MfqTensor] | MlxNintModel,
        config: Qwen4ExpConfig,
        names: MlxQwen4ExpNames | None = None,
        *,
        max_context: int = 4096,
    ) -> None:
        if config.mtp_num_hidden_layers <= 0:
            raise ValueError("Qwen4-Exp checkpoint does not declare MTP layers")
        self.model = tensors if isinstance(tensors, MlxNintModel) else MlxNintModel(tensors)
        self.config = config
        self.names = MlxQwen4ExpNames() if names is None else names
        self.max_context = min(int(max_context), config.max_position_embeddings)
        if self.max_context <= 0:
            raise ValueError("Qwen4-Exp MTP max_context must be positive")
        self.embedding = self.model.embedding(self.names.token_embedding)
        self.output = self.model.linear(
            self.names.token_embedding
            if config.tie_word_embeddings or self.names.output not in self.model.tensors
            else self.names.output
        )
        self.embedding_norm = MlxRMSNorm(
            _dense_vector(self.model, "predictor.embedding_norm.weight"),
            config.rms_norm_eps,
            weight_offset=1.0,
        )
        self.hidden_norm = MlxRMSNorm(
            _dense_vector(self.model, "predictor.hidden_norm.weight"),
            config.rms_norm_eps,
            weight_offset=1.0,
        )
        self.fc_embedding = self.model.linear("predictor.fusion.embedding.weight")
        self.fc_hidden = self.model.linear("predictor.fusion.hidden.weight")
        self.layers = tuple(
            MlxQwen4ExpMtpLayer(
                self.model,
                config,
                f"predictor.block.{index}",
                self.max_context,
            )
            for index in range(config.mtp_num_hidden_layers)
        )
        self.final_mixer = MlxQwen4ExpGatedResidual(
            self.model,
            config,
            "predictor.mhc.pre",
            combine=False,
        )
        self.batch = 0
        self.layer_positions = [0] * len(self.layers)
        self.position_caches: list[mx.array | None] = [None] * len(self.layers)

    @classmethod
    def load_if_present(
        cls,
        causal_lm: MlxQwen4Exp,
        config: Qwen4ExpConfig,
        *,
        max_context: int,
    ) -> MlxQwen4ExpMtp | None:
        """Attach the native MTP head when the loaded artifact contains it."""

        tensors = causal_lm.model.tensors
        has_any_weight = any(name.startswith("predictor.") for name in tensors)
        probe = "predictor.embedding_norm.weight"
        if config.mtp_num_hidden_layers <= 0 or probe not in tensors:
            if has_any_weight:
                raise ValueError("Qwen4-Exp MFQ contains an incomplete or undeclared MTP head")
            return None
        return cls(causal_lm.model, config, max_context=max_context)

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        config: Qwen4ExpConfig | Mapping[str, object] | None = None,
        names: MlxQwen4ExpNames | None = None,
        *,
        mmap: bool = True,
        max_context: int = 4096,
    ) -> MlxQwen4ExpMtp:
        """Load the complete native MTP head from a converted MFQ model."""

        model = MlxNintModel.from_mfq(path, mmap=mmap)
        try:
            selected = config
            if selected is None:
                if MODEL_CONFIG_ASSET not in model.tensors:
                    raise ValueError(
                        "MFQ has no embedded model config; pass Qwen4ExpConfig explicitly"
                    )
                payload = model.tensors[MODEL_CONFIG_ASSET]
                if not isinstance(payload, bytes):
                    raise TypeError("embedded MFQ model config must be a BLOB record")
                selected = json.loads(payload)
            normalized = (
                selected
                if isinstance(selected, Qwen4ExpConfig)
                else Qwen4ExpConfig.from_hf_config(selected)
            )
            return cls(
                model,
                normalized,
                names,
                max_context=max_context,
            )
        except BaseException:
            model.close()
            raise

    def reset_cache(self, batch: int = 1) -> None:
        selected = int(batch)
        if selected <= 0:
            raise ValueError("Qwen4-Exp MTP cache batch must be positive")
        for layer in self.layers:
            layer.reset_cache(selected)
        self.batch = selected
        self.layer_positions = [0] * len(self.layers)
        self.position_caches = [None] * len(self.layers)

    def forward(
        self,
        input_ids: mx.array | np.ndarray,
        previous_hidden_states: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        inputs_embeds: mx.array | np.ndarray | None = None,
        spec_step_index: int = 0,
        use_cache: bool = False,
    ) -> tuple[mx.array, mx.array]:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        if ids.ndim != 2:
            raise ValueError("Qwen4-Exp MTP IDs must have [batch,tokens] shape")
        batch, tokens = (int(item) for item in ids.shape)
        hidden = (
            previous_hidden_states
            if isinstance(previous_hidden_states, mx.array)
            else mx.array(previous_hidden_states)
        )
        if hidden.ndim == 4:
            hidden = hidden.reshape(batch, tokens, -1)
        expected_width = self.config.hc_count * self.config.hidden_size
        if tuple(hidden.shape) != (batch, tokens, expected_width):
            raise ValueError(
                "Qwen4-Exp MTP previous hidden must have "
                "[batch,tokens,hc_count*hidden] shape"
            )
        embeds = self.embedding(ids.astype(mx.int32)) if inputs_embeds is None else inputs_embeds
        embeds = embeds if isinstance(embeds, mx.array) else mx.array(embeds)
        if tuple(embeds.shape) != (batch, tokens, self.config.hidden_size):
            raise ValueError("Qwen4-Exp MTP embeddings have an incompatible shape")
        if use_cache and self.batch not in (0, batch):
            self.reset_cache(batch)
        layer_index = int(spec_step_index) % len(self.layers)
        start = self.layer_positions[layer_index] if use_cache else 0
        current_positions = MlxQwen4Exp._positions(positions, start, tokens)
        if current_positions.ndim == 3 and int(current_positions.shape[1]) != batch:
            raise ValueError("Qwen4-Exp MTP position batch does not match input batch")
        if use_cache:
            previous_positions = self.position_caches[layer_index]
            full_positions = (
                current_positions
                if previous_positions is None
                else mx.concatenate((previous_positions, current_positions), axis=-1)
            )
            self.position_caches[layer_index] = full_positions
        else:
            full_positions = current_positions
        embedding_branch = self.fc_embedding(self.embedding_norm(embeds))
        hidden_streams = self.hidden_norm(hidden).reshape(
            batch,
            tokens,
            self.config.hc_count,
            self.config.hidden_size,
        )
        hidden_streams = self.fc_hidden(hidden_streams)
        hidden_streams = (
            hidden_streams + embedding_branch[..., None, :]
        ).reshape(batch, tokens, expected_width)
        multi_hidden = self.layers[layer_index](
            hidden_streams,
            current_positions,
            full_positions,
            use_cache=use_cache,
        )
        sample_hidden = self.final_mixer.mix(multi_hidden)
        if use_cache:
            self.batch = batch
            self.layer_positions[layer_index] = start + tokens
        return sample_hidden, multi_hidden

    def compute_logits(self, sample_hidden: mx.array | np.ndarray) -> mx.array:
        hidden = sample_hidden if isinstance(sample_hidden, mx.array) else mx.array(sample_hidden)
        if hidden.ndim != 3 or int(hidden.shape[-1]) != self.config.hidden_size:
            raise ValueError("Qwen4-Exp MTP sample hidden has an incompatible shape")
        return self.output(hidden)

    def __call__(
        self,
        input_ids: mx.array | np.ndarray,
        previous_hidden_states: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        inputs_embeds: mx.array | np.ndarray | None = None,
        spec_step_index: int = 0,
        use_cache: bool = False,
    ) -> tuple[mx.array, mx.array]:
        return self.forward(
            input_ids,
            previous_hidden_states,
            positions,
            inputs_embeds=inputs_embeds,
            spec_step_index=spec_step_index,
            use_cache=use_cache,
        )

    def close(self) -> None:
        self.model.close()


class MlxQwen4Exp:
    """Complete Qwen3.8-Flash-Next text decoder with PLE and QSA."""

    def __init__(
        self,
        tensors: Mapping[str, MfqTensor] | MlxNintModel,
        config: Qwen4ExpConfig,
        names: MlxQwen4ExpNames | None = None,
        *,
        max_context: int = 4096,
    ) -> None:
        self.model = tensors if isinstance(tensors, MlxNintModel) else MlxNintModel(tensors)
        self.config = config
        self.names = MlxQwen4ExpNames() if names is None else names
        self.max_context = min(int(max_context), config.max_position_embeddings)
        if self.max_context <= 0:
            raise ValueError("Qwen4-Exp max_context must be positive")
        self.embedding = self.model.embedding(self.names.token_embedding)
        self.output = self.model.linear(
            self.names.token_embedding
            if config.tie_word_embeddings or self.names.output not in self.model.tensors
            else self.names.output
        )
        self.layers = tuple(
            MlxQwen4ExpLayer(
                self.model,
                config,
                self.names,
                index,
                self.max_context,
            )
            for index in range(config.num_hidden_layers)
        )
        self.final_mixer = MlxQwen4ExpGatedResidual(
            self.model,
            config,
            self.names.final_mixer,
            combine=False,
        )
        self.batch = 0
        self.position = 0
        self.position_cache: mx.array | None = None
        self._speculative_checkpoint: tuple[int, int, int] | None = None

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        config: Qwen4ExpConfig | Mapping[str, object] | None = None,
        names: MlxQwen4ExpNames | None = None,
        *,
        mmap: bool = True,
        max_context: int = 4096,
    ) -> MlxQwen4Exp:
        model = MlxNintModel.from_mfq(path, mmap=mmap)
        try:
            selected = config
            if selected is None:
                if MODEL_CONFIG_ASSET not in model.tensors:
                    raise ValueError(
                        "MFQ has no embedded model config; pass Qwen4ExpConfig explicitly"
                    )
                payload = model.tensors[MODEL_CONFIG_ASSET]
                if not isinstance(payload, bytes):
                    raise TypeError("embedded MFQ model config must be a BLOB record")
                selected = json.loads(payload)
            normalized = (
                selected
                if isinstance(selected, Qwen4ExpConfig)
                else Qwen4ExpConfig.from_hf_config(selected)
            )
            return cls(
                model,
                normalized,
                names,
                max_context=max_context,
            )
        except BaseException:
            model.close()
            raise

    def reset_cache(self, batch: int = 1) -> None:
        selected = int(batch)
        if selected <= 0:
            raise ValueError("Qwen4-Exp cache batch must be positive")
        for layer in self.layers:
            layer.reset_cache(selected)
        self.batch = selected
        self.position = 0
        self.position_cache = None
        self._speculative_checkpoint = None

    def commit_speculative_cache(self) -> None:
        """Keep the most recent verify window and discard its rollback refs."""

        for layer in self.layers:
            if isinstance(layer.attention, MlxQwen4ExpGdn):
                layer.attention.commit_speculative_cache()
            if layer.ple is not None:
                layer.ple.commit_speculative_cache()
        self._speculative_checkpoint = None

    def rollback_speculative_cache(self, accepted_drafts: int = 0) -> None:
        """Drop a rejected depth-one draft while retaining its confirmed token."""

        if self._speculative_checkpoint is None:
            raise RuntimeError("Qwen4-Exp has no speculative verify window")
        start, confirmed, tokens = self._speculative_checkpoint
        accepted = int(accepted_drafts)
        if accepted != 0 or confirmed <= 0 or tokens != confirmed + 1:
            raise ValueError("Qwen4-Exp currently rolls back one rejected draft")
        keep = start + confirmed
        for layer in self.layers:
            if isinstance(layer.attention, MlxQwen4ExpGdn):
                layer.attention.rollback_speculative_cache()
            else:
                if layer.attention.cache is None:
                    raise RuntimeError("Qwen4-Exp QSA cache is not initialized")
                layer.attention.cache.pos = keep
                layer.attention.index_cache.position = keep
            if layer.ple is not None:
                layer.ple.rollback_speculative_cache()
        self.position = keep
        if self.position_cache is not None:
            self.position_cache = self.position_cache[..., :keep]
        self._speculative_checkpoint = None

    @staticmethod
    def _positions(
        positions: mx.array | np.ndarray | None,
        start: int,
        tokens: int,
    ) -> mx.array:
        if positions is None:
            return mx.arange(start, start + tokens, dtype=mx.int32)
        value = positions if isinstance(positions, mx.array) else mx.array(positions)
        if value.ndim in (2, 3) and int(value.shape[0]) == 4:
            value = value[1:]
        if value.ndim not in (1, 2, 3) or int(value.shape[-1]) != tokens:
            raise ValueError(
                "Qwen4-Exp positions must have [T], [3,T], [4,T], "
                "[3,B,T], or [4,B,T] shape"
            )
        return value.astype(mx.int32)

    def forward_embeddings_with_hidden(
        self,
        embeddings: mx.array | np.ndarray,
        input_ids: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        use_cache: bool = False,
        n_confirmed: int = 0,
    ) -> tuple[mx.array, mx.array]:
        hidden = embeddings if isinstance(embeddings, mx.array) else mx.array(embeddings)
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        if (
            hidden.ndim != 3
            or ids.ndim != 2
            or tuple(hidden.shape[:2]) != tuple(ids.shape)
            or int(hidden.shape[-1]) != self.config.hidden_size
        ):
            raise ValueError("Qwen4-Exp embeddings/IDs have incompatible shapes")
        batch, tokens = (int(item) for item in ids.shape)
        if use_cache and self.batch not in (0, batch):
            self.reset_cache(batch)
        start = self.position if use_cache else 0
        if start + tokens > self.max_context:
            raise ValueError("Qwen4-Exp input exceeds max_context")
        confirmed = int(n_confirmed)
        if not 0 <= confirmed <= tokens:
            raise ValueError("Qwen4-Exp confirmed-token count is outside the input")
        if confirmed and (not use_cache or confirmed == tokens):
            raise ValueError("Qwen4-Exp speculative verification needs a draft suffix")
        self.commit_speculative_cache()
        if confirmed:
            self._speculative_checkpoint = (start, confirmed, tokens)
        current_positions = self._positions(positions, start, tokens)
        if current_positions.ndim == 3 and int(current_positions.shape[1]) != batch:
            raise ValueError("Qwen4-Exp position batch does not match input batch")
        if use_cache:
            full_positions = (
                current_positions
                if self.position_cache is None
                else mx.concatenate((self.position_cache, current_positions), axis=-1)
            )
            self.position_cache = full_positions
        else:
            full_positions = current_positions
        hidden_streams = mx.tile(
            hidden.astype(mx.float16),
            (1, 1, self.config.hc_count),
        )
        ids = ids.astype(mx.int32)
        for layer in self.layers:
            hidden_streams = layer(
                hidden_streams,
                ids,
                current_positions,
                full_positions,
                use_cache=use_cache,
                n_confirmed=confirmed,
            )
        output = self.final_mixer.mix(hidden_streams)
        logits = self.output(output)
        if use_cache:
            self.batch = batch
            self.position = start + tokens
        return logits, hidden_streams

    def forward_embeddings(
        self,
        embeddings: mx.array | np.ndarray,
        input_ids: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        use_cache: bool = False,
        n_confirmed: int = 0,
    ) -> mx.array:
        logits, _hidden_streams = self.forward_embeddings_with_hidden(
            embeddings,
            input_ids,
            positions,
            use_cache=use_cache,
            n_confirmed=n_confirmed,
        )
        return logits

    def forward_with_hidden(
        self,
        input_ids: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        use_cache: bool = False,
        n_confirmed: int = 0,
    ) -> tuple[mx.array, mx.array]:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        if ids.ndim != 2:
            raise ValueError("Qwen4-Exp IDs must have [batch,tokens] shape")
        ids = ids.astype(mx.int32)
        return self.forward_embeddings_with_hidden(
            self.embedding(ids),
            ids,
            positions,
            use_cache=use_cache,
            n_confirmed=n_confirmed,
        )

    def forward(
        self,
        input_ids: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        use_cache: bool = False,
        n_confirmed: int = 0,
    ) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        if ids.ndim != 2:
            raise ValueError("Qwen4-Exp IDs must have [batch,tokens] shape")
        ids = ids.astype(mx.int32)
        return self.forward_embeddings(
            self.embedding(ids),
            ids,
            positions,
            use_cache=use_cache,
            n_confirmed=n_confirmed,
        )

    def __call__(
        self,
        input_ids: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        use_cache: bool = False,
        n_confirmed: int = 0,
    ) -> mx.array:
        return self.forward(
            input_ids,
            positions,
            use_cache=use_cache,
            n_confirmed=n_confirmed,
        )

    def prefill(self, input_ids: mx.array | np.ndarray) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        self.reset_cache(int(ids.shape[0]))
        return self.forward(ids, use_cache=True)

    def decode(self, input_ids: mx.array | np.ndarray) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[:, None] if self.batch == int(ids.size) else ids[None]
        if ids.ndim != 2 or int(ids.shape[1]) != 1:
            raise ValueError("Qwen4-Exp decode accepts one token per batch")
        return self.forward(ids, use_cache=True)

    def generate(
        self,
        input_ids: mx.array | np.ndarray,
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | tuple[int, ...] | None = None,
    ) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        ids = ids.astype(mx.int32)
        if ids.ndim != 2:
            raise ValueError("Qwen4-Exp generation IDs must have [batch,tokens] shape")
        if int(ids.shape[1]) + int(max_new_tokens) > self.max_context:
            raise ValueError("Qwen4-Exp generation exceeds max_context")
        if int(max_new_tokens) <= 0:
            return ids
        logits = self.prefill(ids)
        pieces = [ids]
        eos = (
            self.config.eos_token_ids
            if eos_token_id is None
            else (
                (int(eos_token_id),)
                if isinstance(eos_token_id, int)
                else tuple(int(value) for value in eos_token_id)
            )
        )
        next_id = _sample(
            logits[:, -1],
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
        )
        for step in range(int(max_new_tokens)):
            pieces.append(next_id[:, None])
            if eos:
                mx.eval(next_id)
                if np.isin(np.asarray(next_id), eos).all():
                    break
            if step + 1 < int(max_new_tokens):
                logits = self.decode(next_id[:, None])
                next_id = _sample(
                    logits[:, -1],
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                )
        return mx.concatenate(pieces, axis=1)

    def close(self) -> None:
        self.model.close()

    def __enter__(self) -> MlxQwen4Exp:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "MlxQwen4Exp",
    "MlxQwen4ExpDenseFFN",
    "MlxQwen4ExpGatedResidual",
    "MlxQwen4ExpGdn",
    "MlxQwen4ExpLayer",
    "MlxQwen4ExpMtp",
    "MlxQwen4ExpMtpLayer",
    "MlxQwen4ExpMoE",
    "MlxQwen4ExpNames",
    "MlxQwen4ExpNgramEmbedding",
    "MlxQwen4ExpPle",
    "MlxQwen4ExpQsa",
]
