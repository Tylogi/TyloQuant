"""GLM DSA native-MFQ inference runtime for Apple silicon."""

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

from mfq.formats.assets import MODEL_CONFIG_ASSET
from mfq.formats.io import MfqTensor
from mfq.formats.moe import NintMoeTensor
from mfq.kernels.metal.glm_dsa import (
    attention_glm_mla_dense,
    attention_glm_mla_sparse,
    glm_dsa_cache_write,
    glm_dsa_indexer_layer_norm,
    glm_dsa_indexer_scores,
    glm_interleaved_rope,
)
from mfq.kernels.metal.moe_ops import (
    moe_topk,
    swiglu_split,
    weighted_reduce,
)
from mfq.kernels.metal.ops import residual_add, residual_rms_norm, rope_tables
from mfq.kernels.metal.sampling import sample as _sample
from mfq.runtime.mlx_linear import MlxLinearGroup, MlxNintModel, mlx_dense_array
from mfq.runtime.mlx_moe import MlxRoutedLinear
from mfq.runtime.mlx_ops import MlxRMSNorm


@dataclass(frozen=True)
class MlxGlmDsaConfig:
    """Normalized fixed-shape GLM-MoE-DSA configuration."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    q_lora_rank: int
    num_attention_heads: int
    num_key_value_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    indexer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]
    rope_base: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = False
    tie_word_embeddings: bool = False
    eos_token_id: tuple[int, ...] = ()

    @classmethod
    def from_hf_config(cls, outer: Mapping) -> MlxGlmDsaConfig:
        """Normalize and validate the published GLM DSA configuration."""

        text = dict(outer.get("text_config") or outer)
        layers = int(text["num_hidden_layers"])
        indexer_types = tuple(str(value) for value in text.get("indexer_types", ()))
        if not indexer_types:
            frequency = max(1, int(text.get("index_topk_freq", 1)))
            offset = int(text.get("index_skip_topk_offset", 0))
            indexer_types = tuple(
                ("full" if max(index - offset + 1, 0) % frequency == 0 else "shared")
                for index in range(layers)
            )
        mlp_layer_types = tuple(str(value) for value in text.get("mlp_layer_types", ()))
        if not mlp_layer_types:
            first_dense = int(text.get("first_k_dense_replace", 0))
            frequency = max(1, int(text.get("moe_layer_freq", 1)))
            mlp_layer_types = tuple(
                ("sparse" if index >= first_dense and index % frequency == 0 else "dense")
                for index in range(layers)
            )
        if len(indexer_types) != layers or len(mlp_layer_types) != layers:
            raise ValueError("GLM DSA schedules must match num_hidden_layers")
        have_full = False
        for index, value in enumerate(indexer_types):
            if value == "full":
                have_full = True
            elif value != "shared" or not have_full:
                raise ValueError(f"invalid GLM DSA indexer schedule at layer {index}")
        unsupported_mlp = set(mlp_layer_types) - {"dense", "sparse"}
        if unsupported_mlp:
            raise ValueError(f"unsupported GLM DSA MLP types: {sorted(unsupported_mlp)}")
        shared_intermediate = int(text.get("shared_expert_intermediate_size", 0))
        if shared_intermediate <= 0:
            shared_intermediate = int(text.get("n_shared_experts", 1)) * int(
                text["moe_intermediate_size"]
            )
        eos = text.get("eos_token_id", outer.get("eos_token_id", ()))
        if eos is None:
            eos_values: tuple[int, ...] = ()
        elif isinstance(eos, int):
            eos_values = (int(eos),)
        else:
            eos_values = tuple(int(value) for value in eos)
        config = cls(
            vocab_size=int(text.get("vocab_size", outer.get("vocab_size", 0))),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text.get("intermediate_size", 0)),
            num_hidden_layers=layers,
            max_position_embeddings=int(text["max_position_embeddings"]),
            q_lora_rank=int(text["q_lora_rank"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            kv_lora_rank=int(text["kv_lora_rank"]),
            qk_nope_head_dim=int(text["qk_nope_head_dim"]),
            qk_rope_head_dim=int(text["qk_rope_head_dim"]),
            v_head_dim=int(text["v_head_dim"]),
            index_head_dim=int(text["index_head_dim"]),
            index_n_heads=int(text["index_n_heads"]),
            index_topk=int(text["index_topk"]),
            num_experts=int(text.get("num_experts", text.get("n_routed_experts", 0))),
            num_experts_per_tok=int(
                text.get(
                    "num_experts_per_tok",
                    text.get("top_k_experts", 0),
                )
            ),
            moe_intermediate_size=int(text["moe_intermediate_size"]),
            shared_expert_intermediate_size=shared_intermediate,
            indexer_types=indexer_types,
            mlp_layer_types=mlp_layer_types,
            rope_base=float(text.get("rope_theta", 1_000_000.0)),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            routed_scaling_factor=float(text.get("routed_scaling_factor", 1.0)),
            norm_topk_prob=bool(text.get("norm_topk_prob", False)),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            eos_token_id=eos_values,
        )
        config._validate_architecture(text)
        return config

    def _validate_architecture(self, raw: Mapping | None = None) -> None:
        required = (
            self.num_attention_heads == 64
            and self.num_key_value_heads == 64
            and self.kv_lora_rank == 512
            and self.qk_nope_head_dim == 192
            and self.qk_rope_head_dim == 64
            and self.v_head_dim == 256
            and self.index_head_dim == 128
            and self.index_n_heads == 32
            and self.index_topk == 2048
            and self.q_lora_rank > 0
            and self.num_experts > 0
            and self.num_experts_per_tok > 0
        )
        if not required:
            raise ValueError("unsupported GLM DSA architecture dimensions")
        if raw is not None:
            expected = (
                int(raw.get("qk_head_dim", 256)) == 256
                and not bool(raw.get("attention_bias", False))
                and bool(raw.get("rope_interleave", True))
                and bool(raw.get("indexer_rope_interleave", True))
                and str(raw.get("hidden_act", "silu")) == "silu"
                and int(raw.get("n_group", 1)) == 1
                and int(raw.get("topk_group", 1)) == 1
                and int(raw.get("n_shared_experts", 1)) == 1
                and str(raw.get("scoring_func", "sigmoid")) == "sigmoid"
                and str(raw.get("topk_method", "noaux_tc")) == "noaux_tc"
            )
            if not expected:
                raise ValueError("unsupported GLM DSA configuration semantics")


@dataclass(frozen=True)
class MlxGlmDsaNames:
    token_embedding: str = "model.embed_tokens.weight"
    output_norm: str = "model.norm.weight"
    output: str = "lm_head.weight"
    layer_prefix: str = "model.layers.{i}"

    def layer(self, index: int) -> str:
        return self.layer_prefix.format(i=index)


def _dense_array(model: MlxNintModel, name: str) -> mx.array:
    if name not in model.tensors:
        raise KeyError(f"tensor {name!r} is not present in the GLM DSA model")
    value = model.tensors[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"GLM DSA tensor {name!r} must be dense")
    return mlx_dense_array(value)


def _dense_vector(model: MlxNintModel, name: str) -> mx.array:
    value = _dense_array(model, name)
    if value.ndim != 1:
        raise TypeError(f"GLM DSA tensor {name!r} must be a vector")
    return mx.contiguous(value.astype(mx.float32))


def _nint_moe(model: MlxNintModel, name: str) -> NintMoeTensor:
    if name not in model.tensors:
        raise KeyError(f"tensor {name!r} is not present in the GLM DSA model")
    value = model.tensors[name]
    if not isinstance(value, NintMoeTensor):
        raise TypeError(f"GLM DSA tensor {name!r} must use NINTM")
    return value


class MlxGlmDsaDenseFFN:
    def __init__(self, model: MlxNintModel, prefix: str) -> None:
        self.ffn = model.ffn(
            f"{prefix}.gate_proj.weight",
            f"{prefix}.up_proj.weight",
            f"{prefix}.down_proj.weight",
        )

    def __call__(self, value: mx.array) -> mx.array:
        return self.ffn(value.astype(mx.float16))


class MlxGlmDsaMoE:
    """GLM sigmoid/noaux routed experts plus the ungated shared expert."""

    def __init__(
        self,
        model: MlxNintModel,
        config: MlxGlmDsaConfig,
        prefix: str,
    ) -> None:
        gate_up = _nint_moe(model, f"{prefix}.experts.gate_up_proj")
        down = _nint_moe(model, f"{prefix}.experts.down_proj")
        if (
            gate_up.n_experts != config.num_experts
            or down.n_experts != config.num_experts
            or gate_up.neuron_len != config.hidden_size
            or gate_up.out_per_expert != 2 * config.moe_intermediate_size
            or down.neuron_len != config.moe_intermediate_size
            or down.out_per_expert != config.hidden_size
        ):
            raise ValueError("GLM DSA MoE tensor shapes disagree with config")
        self.gate_up = MlxRoutedLinear(gate_up)
        self.down = MlxRoutedLinear(down)
        self.router = model.linear(f"{prefix}.gate.weight")
        self.router_bias = _dense_vector(
            model,
            f"{prefix}.gate.e_score_correction_bias",
        )
        self.shared = MlxGlmDsaDenseFFN(
            model,
            f"{prefix}.shared_experts",
        )
        self.config = config

    def forward(self, value: mx.array) -> mx.array:
        source = value.reshape((-1, self.config.hidden_size)).astype(mx.float16)
        logits = self.router(source.astype(mx.float32))
        ids, weights = moe_topk(
            logits,
            self.config.num_experts_per_tok,
            use_sigmoid=True,
            normalize=self.config.norm_topk_prob,
            bias=self.router_bias,
            scale=self.config.routed_scaling_factor,
        )
        gate_up = self.gate_up(source, ids)
        hidden = swiglu_split(gate_up)
        routed = weighted_reduce(self.down(hidden, ids), weights)
        shared = self.shared(source)
        return residual_add(routed, shared.astype(routed.dtype))

    def __call__(self, value: mx.array) -> mx.array:
        return self.forward(value)


@dataclass
class _MlxGlmDsaSharedState:
    topk_indices: mx.array | None = None
    dense_prefix_rows: int = 0

    def reset(self) -> None:
        self.topk_indices = None
        self.dense_prefix_rows = 0


class MlxGlmDsaLayer:
    """One fixed-shape GLM DSA attention and FFN block."""

    def __init__(
        self,
        model: MlxNintModel,
        config: MlxGlmDsaConfig,
        names: MlxGlmDsaNames,
        index: int,
        shared_state: _MlxGlmDsaSharedState,
        max_context: int,
        rope: tuple[mx.array, mx.array],
    ) -> None:
        self.config = config
        self.index = int(index)
        self.full_indexer = config.indexer_types[index] == "full"
        self.shared_state = shared_state
        self.max_context = int(max_context)
        self.rope = rope
        prefix = names.layer(index)
        attention_prefix = f"{prefix}.self_attn"
        self.attn_norm = MlxRMSNorm(
            _dense_vector(model, f"{prefix}.input_layernorm.weight"),
            config.rms_norm_eps,
        )
        self.ffn_norm_weight = _dense_vector(
            model,
            f"{prefix}.post_attention_layernorm.weight",
        )
        self.q_a_norm = MlxRMSNorm(
            _dense_vector(model, f"{attention_prefix}.q_a_layernorm.weight"),
            1e-6,
        )
        self.kv_a_norm = MlxRMSNorm(
            _dense_vector(model, f"{attention_prefix}.kv_a_layernorm.weight"),
            1e-6,
        )
        first_names = [
            f"{attention_prefix}.q_a_proj.weight",
            f"{attention_prefix}.kv_a_proj_with_mqa.weight",
        ]
        second_names = [f"{attention_prefix}.q_b_proj.weight"]
        if self.full_indexer:
            first_names.extend(
                (
                    f"{attention_prefix}.indexer.wk.weight",
                    f"{attention_prefix}.indexer.weights_proj.weight",
                )
            )
            second_names.append(f"{attention_prefix}.indexer.wq_b.weight")
            self.index_k_norm = _dense_vector(
                model,
                f"{attention_prefix}.indexer.k_norm.weight",
            )
            self.index_k_bias = _dense_vector(
                model,
                f"{attention_prefix}.indexer.k_norm.bias",
            )
        else:
            self.index_k_norm = None
            self.index_k_bias = None
        self.input_projection = MlxLinearGroup(tuple(model.linear(name) for name in first_names))
        q_layers = tuple(model.linear(name) for name in second_names)
        self.q_projection = MlxLinearGroup(q_layers) if len(q_layers) > 1 else q_layers[0]
        embed = _nint_moe(model, f"{attention_prefix}.embed_q")
        unembed = _nint_moe(model, f"{attention_prefix}.unembed_out")
        if (
            embed.n_experts != config.num_attention_heads
            or embed.neuron_len != config.qk_nope_head_dim
            or embed.out_per_expert != config.kv_lora_rank
            or unembed.n_experts != config.num_attention_heads
            or unembed.neuron_len != config.kv_lora_rank
            or unembed.out_per_expert != config.v_head_dim
        ):
            raise ValueError(f"GLM DSA head-wise tensor shapes disagree at layer {index}")
        self.embed_q = MlxRoutedLinear(embed)
        self.unembed_out = MlxRoutedLinear(unembed)
        self.output = model.linear(f"{attention_prefix}.o_proj.weight")
        mlp_prefix = f"{prefix}.mlp"
        self.ffn = (
            MlxGlmDsaMoE(model, config, mlp_prefix)
            if config.mlp_layer_types[index] == "sparse"
            else MlxGlmDsaDenseFFN(model, mlp_prefix)
        )
        self.kv_cache: mx.array | None = None
        self.index_cache: mx.array | None = None
        self.batch = 0

    def reset_cache(self, batch: int) -> None:
        self.kv_cache = None
        self.index_cache = None
        self.batch = int(batch)

    def _ensure_cache(self, batch: int) -> None:
        if self.kv_cache is not None and self.batch == int(batch):
            return
        self.batch = int(batch)
        self.kv_cache = mx.zeros(
            (batch, self.max_context, 576),
            dtype=mx.float16,
        )
        self.index_cache = (
            mx.zeros(
                (batch, self.max_context, self.config.index_head_dim),
                dtype=mx.float16,
            )
            if self.full_indexer
            else None
        )

    def _headwise(
        self,
        layer: MlxRoutedLinear,
        value: mx.array,
        output_width: int,
    ) -> mx.array:
        batch, tokens, heads, width = (int(item) for item in value.shape)
        if heads != self.config.num_attention_heads:
            raise ValueError("GLM head-wise projection head count mismatch")
        rows = batch * tokens * heads
        ids = mx.tile(
            mx.arange(heads, dtype=mx.int32),
            batch * tokens,
        ).reshape((rows, 1))
        output = layer(
            value.reshape((rows, width)),
            ids,
        )
        return output.reshape((batch, tokens, heads, output_width))

    def _update_indexer(
        self,
        index_query: mx.array,
        index_weights: mx.array,
        cache_position: int,
        logical_len: int,
    ) -> None:
        config = self.config
        tokens = int(index_query.shape[1])
        if logical_len <= config.index_topk:
            self.shared_state.topk_indices = None
            self.shared_state.dense_prefix_rows = tokens
            return
        prefix = max(
            0,
            min(tokens, config.index_topk - int(cache_position)),
        )
        sparse_rows = tokens - prefix
        if sparse_rows <= 0:
            self.shared_state.topk_indices = None
            self.shared_state.dense_prefix_rows = tokens
            return
        assert self.index_cache is not None
        indices = []
        max_score_elements = 32 * 1024 * 1024
        rows_per_chunk = max(
            1,
            max_score_elements // max(1, int(index_query.shape[0]) * logical_len),
        )
        rows_per_chunk = min(rows_per_chunk, 256)
        if rows_per_chunk >= 64:
            rows_per_chunk = rows_per_chunk // 64 * 64
        for start in range(0, sparse_rows, rows_per_chunk):
            count = min(rows_per_chunk, sparse_rows - start)
            query_start = prefix + start
            scores = glm_dsa_indexer_scores(
                index_query[:, query_start : query_start + count],
                self.index_cache,
                index_weights[:, query_start : query_start + count],
                cache_position + query_start,
                logical_len,
            )
            partition = mx.argpartition(
                scores,
                logical_len - config.index_topk,
                axis=-1,
            )
            indices.append(partition[..., -config.index_topk :].astype(mx.int32))
        self.shared_state.topk_indices = mx.concatenate(indices, axis=1)
        self.shared_state.dense_prefix_rows = prefix

    def forward(
        self,
        value: mx.array,
        positions: mx.array,
        cache_position: int,
    ) -> mx.array:
        config = self.config
        batch, tokens, hidden = (int(item) for item in value.shape)
        logical_len = int(cache_position) + tokens
        self._ensure_cache(batch)
        assert self.kv_cache is not None
        residual = mx.contiguous(value.astype(mx.float16))
        normalized = self.attn_norm(residual)
        first = self.input_projection(normalized)
        query_reduced = self.q_a_norm(
            first[0].reshape((batch, tokens, config.q_lora_rank)).astype(mx.float16)
        )
        projected_query = self.q_projection(query_reduced)
        second = projected_query if isinstance(projected_query, tuple) else (projected_query,)

        query_main = (
            second[0]
            .astype(mx.float16)
            .reshape(
                (
                    batch,
                    tokens,
                    config.num_attention_heads,
                    config.qk_nope_head_dim + config.qk_rope_head_dim,
                )
            )
        )
        query_nope = query_main[..., : config.qk_nope_head_dim]
        query_pe = mx.transpose(
            query_main[..., config.qk_nope_head_dim :],
            (0, 2, 1, 3),
        )
        query_pe = glm_interleaved_rope(
            query_pe,
            positions,
            self.rope[0],
            self.rope[1],
            config.qk_rope_head_dim,
        )

        compressed = (
            first[1]
            .astype(mx.float16)
            .reshape((batch, tokens, config.kv_lora_rank + config.qk_rope_head_dim))
        )
        kv_latent = self.kv_a_norm(compressed[..., : config.kv_lora_rank])
        key_pe = mx.transpose(
            compressed[..., config.kv_lora_rank :].reshape(
                (batch, tokens, 1, config.qk_rope_head_dim)
            ),
            (0, 2, 1, 3),
        )
        key_pe = glm_interleaved_rope(
            key_pe,
            positions,
            self.rope[0],
            self.rope[1],
            config.qk_rope_head_dim,
        )
        kv_rows = mx.concatenate(
            (
                kv_latent,
                mx.transpose(key_pe, (0, 2, 1, 3)).reshape(
                    (batch, tokens, config.qk_rope_head_dim)
                ),
            ),
            axis=-1,
        )
        self.kv_cache = glm_dsa_cache_write(
            self.kv_cache,
            kv_rows,
            positions,
        )

        if self.full_indexer:
            assert self.index_k_norm is not None
            assert self.index_k_bias is not None
            assert self.index_cache is not None
            index_key = glm_dsa_indexer_layer_norm(
                first[2].astype(mx.float16).reshape((batch, tokens, config.index_head_dim)),
                self.index_k_norm,
                self.index_k_bias,
                1e-5,
            )
            index_key = glm_interleaved_rope(
                mx.transpose(
                    index_key.reshape((batch, tokens, 1, config.index_head_dim)),
                    (0, 2, 1, 3),
                ),
                positions,
                self.rope[0],
                self.rope[1],
                config.qk_rope_head_dim,
            )
            index_key = mx.transpose(index_key, (0, 2, 1, 3)).reshape(
                (batch, tokens, config.index_head_dim)
            )
            self.index_cache = glm_dsa_cache_write(
                self.index_cache,
                index_key,
                positions,
            )
            index_query = (
                second[1]
                .astype(mx.float16)
                .reshape(
                    (
                        batch,
                        tokens,
                        config.index_n_heads,
                        config.index_head_dim,
                    )
                )
            )
            index_query = glm_interleaved_rope(
                mx.transpose(index_query, (0, 2, 1, 3)),
                positions,
                self.rope[0],
                self.rope[1],
                config.qk_rope_head_dim,
            )
            index_query = mx.transpose(index_query, (0, 2, 1, 3))
            index_weights = (
                first[3].reshape((batch, tokens, config.index_n_heads)).astype(mx.float32)
            )
            self._update_indexer(
                index_query,
                index_weights,
                cache_position,
                logical_len,
            )

        absorbed = self._headwise(
            self.embed_q,
            query_nope,
            config.kv_lora_rank,
        )
        query_mla = mx.concatenate(
            (mx.transpose(absorbed, (0, 2, 1, 3)), query_pe),
            axis=-1,
        ).astype(mx.float32)
        scale = 1.0 / math.sqrt(config.qk_nope_head_dim + config.qk_rope_head_dim)
        selected = self.shared_state.topk_indices
        if selected is None:
            attended = attention_glm_mla_dense(
                query_mla,
                self.kv_cache,
                logical_len,
                scale,
            )
        else:
            prefix = self.shared_state.dense_prefix_rows
            sparse_rows = int(selected.shape[1])
            if prefix + sparse_rows != tokens:
                raise ValueError("GLM shared index state row count mismatch")
            parts = []
            if prefix:
                parts.append(
                    attention_glm_mla_dense(
                        query_mla[:, :, :prefix],
                        self.kv_cache,
                        cache_position + prefix,
                        scale,
                    )
                )
            parts.append(
                attention_glm_mla_sparse(
                    query_mla[:, :, prefix:],
                    self.kv_cache,
                    selected,
                    scale=scale,
                )
            )
            attended = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        value_heads = self._headwise(
            self.unembed_out,
            attended.astype(mx.float16),
            config.v_head_dim,
        )
        attention_output = self.output(
            value_heads.reshape(
                (
                    batch,
                    tokens,
                    config.num_attention_heads * config.v_head_dim,
                )
            )
        ).astype(mx.float16)
        hidden, ffn_input = residual_rms_norm(
            residual.reshape((batch * tokens, hidden)),
            attention_output.reshape((batch * tokens, hidden)),
            self.ffn_norm_weight,
            config.rms_norm_eps,
        )
        ffn_output = self.ffn(ffn_input).reshape((batch * tokens, hidden.shape[-1]))
        return residual_add(hidden, ffn_output).reshape((batch, tokens, config.hidden_size))

    def __call__(
        self,
        value: mx.array,
        positions: mx.array,
        cache_position: int,
    ) -> mx.array:
        return self.forward(value, positions, cache_position)


class MlxGlmDsa:
    """Complete GLM DSA MFQ model with shared index state and generation."""

    def __init__(
        self,
        tensors: Mapping[str, MfqTensor] | MlxNintModel,
        config: MlxGlmDsaConfig,
        names: MlxGlmDsaNames | None = None,
        *,
        max_context: int = 4096,
    ) -> None:
        config._validate_architecture()
        self.model = tensors if isinstance(tensors, MlxNintModel) else MlxNintModel(tensors)
        self.config = config
        self.names = MlxGlmDsaNames() if names is None else names
        self.max_context = min(
            int(max_context),
            config.max_position_embeddings,
        )
        if self.max_context <= 0:
            raise ValueError("GLM DSA max_context must be positive")
        self.embedding = self.model.embedding(self.names.token_embedding)
        self.output_norm = MlxRMSNorm(
            _dense_vector(self.model, self.names.output_norm),
            config.rms_norm_eps,
        )
        self.output = self.model.linear(
            self.names.token_embedding
            if config.tie_word_embeddings or self.names.output not in self.model.tensors
            else self.names.output
        )
        self.rope = rope_tables(
            config.rope_base,
            config.qk_rope_head_dim,
            self.max_context,
        )
        self.shared_state = _MlxGlmDsaSharedState()
        self.layers = tuple(
            MlxGlmDsaLayer(
                self.model,
                config,
                self.names,
                index,
                self.shared_state,
                self.max_context,
                self.rope,
            )
            for index in range(config.num_hidden_layers)
        )
        self.batch = 0
        self.position = 0

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        config: MlxGlmDsaConfig | Mapping | None = None,
        names: MlxGlmDsaNames | None = None,
        *,
        mmap: bool = True,
        max_context: int = 4096,
    ) -> MlxGlmDsa:
        model = MlxNintModel.from_mfq(path, mmap=mmap)
        try:
            selected = config
            if selected is None:
                if MODEL_CONFIG_ASSET not in model.tensors:
                    raise ValueError(
                        "MFQ has no embedded model config; pass MlxGlmDsaConfig explicitly"
                    )
                payload = model.tensors[MODEL_CONFIG_ASSET]
                if not isinstance(payload, bytes):
                    raise TypeError("embedded MFQ model config must be a BLOB record")
                selected = json.loads(payload)
            normalized = (
                selected
                if isinstance(selected, MlxGlmDsaConfig)
                else MlxGlmDsaConfig.from_hf_config(selected)
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
        batch_size = int(batch)
        if batch_size <= 0:
            raise ValueError("GLM DSA cache batch must be positive")
        for layer in self.layers:
            layer.reset_cache(batch_size)
        self.shared_state.reset()
        self.batch = batch_size
        self.position = 0

    def forward(
        self,
        input_ids: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        use_cache: bool = False,
    ) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2:
            raise ValueError("GLM DSA IDs must have [batch,tokens] shape")
        ids = ids.astype(mx.int32)
        batch, tokens = (int(item) for item in ids.shape)
        if tokens <= 0:
            raise ValueError("GLM DSA input cannot be empty")
        if not use_cache or self.batch != batch:
            self.reset_cache(batch)
        start = self.position if use_cache else 0
        if start + tokens > self.max_context:
            raise ValueError("GLM DSA input exceeds max_context")
        if positions is None:
            position_array = mx.arange(
                start,
                start + tokens,
                dtype=mx.int32,
            )
        else:
            position_array = (
                (positions if isinstance(positions, mx.array) else mx.array(positions))
                .astype(mx.int32)
                .reshape((-1,))
            )
            if int(position_array.size) != tokens:
                raise ValueError("GLM DSA positions must match token count")
        self.shared_state.reset()
        hidden = self.embedding(ids).astype(mx.float16)
        for layer in self.layers:
            hidden = layer(hidden, position_array, start)
        logits = self.output(self.output_norm(hidden.astype(mx.float32)))
        if use_cache:
            self.position = start + tokens
        return logits

    def __call__(
        self,
        input_ids: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        use_cache: bool = False,
    ) -> mx.array:
        return self.forward(
            input_ids,
            positions,
            use_cache=use_cache,
        )

    def prefill(self, input_ids: mx.array | np.ndarray) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2 or int(ids.shape[1]) == 0:
            raise ValueError("GLM DSA prefill IDs must have non-empty [batch,tokens] shape")
        self.reset_cache(int(ids.shape[0]))
        return self.forward(ids, use_cache=True)

    def decode(self, input_ids: mx.array | np.ndarray) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            if int(ids.size) != 1:
                raise ValueError("GLM DSA decode accepts one token per batch")
            ids = ids[None, :]
        if ids.ndim != 2 or int(ids.shape[1]) != 1:
            raise ValueError("GLM DSA decode accepts one token per batch")
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
            ids = ids[None, :]
        if ids.ndim != 2:
            raise ValueError("GLM DSA generation IDs must have [batch,tokens] shape")
        ids = ids.astype(mx.int32)
        if int(ids.shape[1]) + int(max_new_tokens) > self.max_context:
            raise ValueError("GLM DSA generation exceeds max_context")
        if int(max_new_tokens) <= 0:
            return ids
        logits = self.prefill(ids)
        pieces = [ids]
        eos = (
            self.config.eos_token_id
            if eos_token_id is None
            else (
                (int(eos_token_id),)
                if isinstance(eos_token_id, int)
                else tuple(int(value) for value in eos_token_id)
            )
        )
        next_id = _sample(
            logits[:, -1, :],
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
                    logits[:, -1, :],
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                )
        return mx.concatenate(pieces, axis=1)

    def close(self) -> None:
        self.model.close()

    def __enter__(self) -> MlxGlmDsa:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "MlxGlmDsa",
    "MlxGlmDsaConfig",
    "MlxGlmDsaDenseFFN",
    "MlxGlmDsaLayer",
    "MlxGlmDsaMoE",
    "MlxGlmDsaNames",
]
