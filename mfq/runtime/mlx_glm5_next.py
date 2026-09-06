"""GLM-5.3-Flash (``glm5_next``) MFQ reference runtime for Apple silicon."""

from __future__ import annotations

import json
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

from mfq.architectures.flash_next import Glm5NextConfig
from mfq.formats.assets import MODEL_CONFIG_ASSET
from mfq.formats.io import MfqTensor
from mfq.kernels.metal.flash_next import (
    glm5_dense_mla_attention,
    glm5_kda_forget_gate,
    glm5_kpool_scores,
    glm5_kpool_states,
    glm5_mhc_post,
    glm5_mhc_pre,
    glm5_sparse_mla_attention,
)
from mfq.kernels.metal.glm_dsa import glm_dsa_indexer_layer_norm
from mfq.kernels.metal.linear_attention import gated_delta_net, linear_conv_qkv
from mfq.kernels.metal.moe_ops import moe_topk, weighted_reduce
from mfq.kernels.metal.sampling import sample as _sample
from mfq.runtime.mlx_linear import MlxLinearGroup, MlxNintModel, mlx_dense_array
from mfq.runtime.mlx_moe import MlxRoutedLinear
from mfq.runtime.mlx_ops import MlxRMSNorm


@dataclass(frozen=True)
class MlxGlm5NextNames:
    token_embedding: str = "model.language_model.embed_tokens.weight"
    output_norm: str = "model.language_model.norm.weight"
    output: str = "lm_head.weight"
    layer_prefix: str = "model.language_model.layers.{i}"

    def layer(self, index: int) -> str:
        return self.layer_prefix.format(i=index)


def _dense_array(
    model: MlxNintModel,
    name: str,
    *,
    dtype: mx.Dtype | None = None,
) -> mx.array:
    if name not in model.tensors:
        raise KeyError(f"tensor {name!r} is not present in the GLM-5-Next model")
    value = model.tensors[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"GLM-5-Next tensor {name!r} must be dense")
    return mx.contiguous(mlx_dense_array(value, dtype=dtype))


def _dense_vector(model: MlxNintModel, name: str) -> mx.array:
    value = _dense_array(model, name, dtype=mx.float32)
    if value.ndim != 1:
        raise TypeError(f"GLM-5-Next tensor {name!r} must be a vector")
    return value


class _MlxSequenceCache:
    """Dynamically grow one contiguous ``[batch,sequence,width]`` cache."""

    def __init__(self, max_sequence: int, width: int) -> None:
        self.max_sequence = int(max_sequence)
        self.width = int(width)
        self.batch = 0
        self.position = 0
        self.values: mx.array | None = None

    def reset(self, batch: int, *, initial_capacity: int = 16) -> None:
        selected_batch = int(batch)
        if selected_batch <= 0:
            raise ValueError("GLM-5-Next cache batch must be positive")
        capacity = min(max(int(initial_capacity), 1), self.max_sequence)
        self.values = mx.zeros(
            (selected_batch, capacity, self.width),
            dtype=mx.float16,
        )
        self.batch = selected_batch
        self.position = 0

    def _ensure_capacity(self, required: int) -> None:
        if self.values is None:
            raise RuntimeError("GLM-5-Next cache was not initialized")
        if required <= int(self.values.shape[1]):
            return
        if required > self.max_sequence:
            raise ValueError("GLM-5-Next cache exceeds max_context")
        capacity = int(self.values.shape[1])
        while capacity < required:
            capacity = min(self.max_sequence, capacity * 2)
        expanded = mx.zeros(
            (self.batch, capacity, self.width),
            dtype=self.values.dtype,
        )
        expanded = mx.slice_update(
            expanded,
            self.values,
            start_indices=mx.array([0, 0, 0], dtype=mx.int32),
            axes=(0, 1, 2),
        )
        self.values = expanded

    def append(self, value: mx.array) -> tuple[mx.array, int]:
        if value.ndim != 3 or int(value.shape[-1]) != self.width:
            raise ValueError("GLM-5-Next cache append shape mismatch")
        batch, tokens = (int(item) for item in value.shape[:2])
        if self.values is None or self.batch != batch:
            self.reset(batch, initial_capacity=max(16, tokens))
        start = self.position
        end = start + tokens
        self._ensure_capacity(end)
        assert self.values is not None
        self.values = mx.slice_update(
            self.values,
            value.astype(self.values.dtype),
            start_indices=mx.array([0, start, 0], dtype=mx.int32),
            axes=(0, 1, 2),
        )
        self.position = end
        return self.values[:, :end], start


class MlxGlm5NextMhc:
    """One attention/FFN manifold hyper-connection site."""

    def __init__(
        self,
        model: MlxNintModel,
        config: Glm5NextConfig,
        layer_prefix: str,
        site: str,
    ) -> None:
        prefix = f"{layer_prefix}.hc_{site}"
        self.function = _dense_array(model, prefix + "_fn")
        self.base = _dense_vector(model, prefix + "_base")
        self.scale = _dense_vector(model, prefix + "_scale")
        self.config = config

    def pre(self, hidden_streams: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        return glm5_mhc_pre(
            hidden_streams,
            self.function,
            self.base,
            self.scale,
            sinkhorn_iterations=self.config.hc_sinkhorn_iters,
            hc_eps=self.config.hc_eps,
            rms_eps=self.config.rms_norm_eps,
        )

    @staticmethod
    def post(
        branch: mx.array,
        residual: mx.array,
        post: mx.array,
        combination: mx.array,
    ) -> mx.array:
        return glm5_mhc_post(branch, residual, post, combination)


class MlxGlm5NextDenseFFN:
    """GLM's bounded SwiGLU dense/shared expert."""

    def __init__(
        self,
        model: MlxNintModel,
        prefix: str,
        swiglu_limit: float,
    ) -> None:
        self.gate_up = MlxLinearGroup(
            (
                model.linear(prefix + ".gate_proj.weight"),
                model.linear(prefix + ".up_proj.weight"),
            )
        )
        self.down = model.linear(prefix + ".down_proj.weight")
        self.limit = float(swiglu_limit)

    def __call__(self, value: mx.array) -> mx.array:
        gate, up = self.gate_up(value)
        gate = mx.minimum(gate, self.limit)
        up = mx.clip(up, -self.limit, self.limit)
        return self.down((gate * mx.sigmoid(gate)) * up)


class MlxGlm5NextMoE:
    """GLM sigmoid/noaux routing plus its bounded shared expert."""

    def __init__(
        self,
        model: MlxNintModel,
        config: Glm5NextConfig,
        prefix: str,
    ) -> None:
        gate_up = model.routed(prefix + ".experts.gate_up_proj")
        down = model.routed(prefix + ".experts.down_proj")
        if (
            gate_up.n_experts != config.num_experts
            or down.n_experts != config.num_experts
            or gate_up.neuron_len != config.hidden_size
            or gate_up.out_per_expert != 2 * config.moe_intermediate_size
            or down.neuron_len != config.moe_intermediate_size
            or down.out_per_expert != config.hidden_size
        ):
            raise ValueError("GLM-5-Next routed expert tensor shapes disagree")
        self.gate_up = gate_up
        self.down = down
        self.router = model.linear(prefix + ".gate.weight")
        self.router_bias = _dense_vector(
            model,
            prefix + ".gate.e_score_correction_bias",
        )
        self.shared = MlxGlm5NextDenseFFN(
            model,
            prefix + ".shared_experts",
            config.swiglu_limit,
        )
        self.config = config

    def __call__(self, value: mx.array) -> mx.array:
        config = self.config
        source = value.reshape((-1, config.hidden_size)).astype(mx.float16)
        ids, weights = moe_topk(
            self.router(source.astype(mx.float32)),
            config.num_experts_per_tok,
            use_sigmoid=True,
            normalize=config.norm_topk_prob,
            bias=self.router_bias,
            scale=config.routed_scaling_factor,
        )
        gate_up = self.gate_up(source, ids)
        gate, up = mx.split(gate_up, 2, axis=-1)
        gate = mx.minimum(gate, config.swiglu_limit)
        up = mx.clip(up, -config.swiglu_limit, config.swiglu_limit)
        routed = weighted_reduce(
            self.down((gate * mx.sigmoid(gate)) * up, ids),
            weights,
        )
        return (routed + self.shared(source)).reshape(value.shape)


class MlxGlm5NextKda:
    """Kimi Delta Attention block used by three of each four GLM layers."""

    def __init__(
        self,
        model: MlxNintModel,
        config: Glm5NextConfig,
        prefix: str,
    ) -> None:
        self.config = config
        self.qkv = MlxLinearGroup(
            tuple(
                model.linear(f"{prefix}.{projection}_proj.weight") for projection in ("q", "k", "v")
            )
        )
        self.conv_weight = mx.concatenate(
            tuple(
                _dense_array(model, f"{prefix}.{projection}_conv1d.weight")
                for projection in ("q", "k", "v")
            ),
            axis=0,
        )
        self.f_a = _dense_array(model, prefix + ".f_a_proj.weight")
        self.f_b = _dense_array(model, prefix + ".f_b_proj.weight")
        self.dt_bias = _dense_vector(model, prefix + ".dt_bias")
        self.a_log = _dense_vector(model, prefix + ".A_log")
        self.beta = model.linear(prefix + ".b_proj.weight")
        self.gate_a = model.linear(prefix + ".g_a_proj.weight")
        self.gate_b = model.linear(prefix + ".g_b_proj.weight")
        self.output_norm = MlxRMSNorm(
            _dense_vector(model, prefix + ".o_norm.weight"),
            config.rms_norm_eps,
        )
        self.output = model.linear(prefix + ".o_proj.weight")
        self.conv_state: mx.array | None = None
        self.recurrent_state: mx.array | None = None
        self.rollback_state: tuple[mx.array, mx.array] | None = None
        self.batch = 0

    @property
    def qkv_width(self) -> int:
        return self.config.kda_num_heads * self.config.kda_head_dim

    def reset_cache(self, batch: int) -> None:
        selected = int(batch)
        config = self.config
        self.conv_state = mx.zeros(
            (
                selected,
                config.kda_conv_kernel_size - 1,
                3 * self.qkv_width,
            ),
            dtype=mx.float32,
        )
        self.recurrent_state = mx.zeros(
            (
                selected,
                config.kda_num_heads,
                config.kda_head_dim,
                config.kda_head_dim,
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
            raise ValueError("GLM-5-Next confirmed-token count is outside the input")
        if confirmed and not use_cache:
            raise ValueError("GLM-5-Next speculative verification requires a cache")
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
        query_input, key_input, value_input = self.qkv(hidden_states)
        if use_cache:
            if self.conv_state is None or self.batch != batch:
                self.reset_cache(batch)
            assert self.conv_state is not None and self.recurrent_state is not None
            conv_state = self.conv_state
            recurrent_state = self.recurrent_state
        else:
            conv_state = mx.zeros(
                (batch, config.kda_conv_kernel_size - 1, 3 * self.qkv_width),
                dtype=mx.float32,
            )
            recurrent_state = None
        query, key, value, next_conv_state = linear_conv_qkv(
            conv_state,
            mx.concatenate((query_input, key_input), axis=-1),
            value_input,
            self.conv_weight,
            num_key_heads=config.kda_num_heads,
            num_value_heads=config.kda_num_heads,
            key_head_dim=config.kda_head_dim,
            value_head_dim=config.kda_head_dim,
            eps=1e-6,
        )
        forget = glm5_kda_forget_gate(
            hidden_states,
            self.f_a,
            self.f_b,
            self.dt_bias,
            self.a_log,
            num_heads=config.kda_num_heads,
            head_dim=config.kda_head_dim,
            lower_bound=config.kda_gate_lower_bound,
        )
        beta = mx.sigmoid(self.beta(hidden_states).astype(mx.float32))
        attended, next_recurrent_state = gated_delta_net(
            query,
            key,
            value,
            mx.transpose(forget, (0, 2, 1, 3)),
            mx.transpose(beta, (0, 2, 1)),
            recurrent_state,
        )
        if use_cache:
            self.conv_state = next_conv_state
            self.recurrent_state = next_recurrent_state
        gate = self.gate_b(self.gate_a(hidden_states)).reshape(
            batch,
            tokens,
            config.kda_num_heads,
            config.kda_head_dim,
        )
        gate = mx.transpose(gate, (0, 2, 1, 3))
        normalized = self.output_norm(attended) * mx.sigmoid(gate.astype(mx.float32))
        normalized = mx.transpose(normalized, (0, 2, 1, 3)).reshape(
            batch,
            tokens,
            self.qkv_width,
        )
        return self.output(normalized.astype(hidden_states.dtype))

    def commit_speculative_cache(self) -> None:
        self.rollback_state = None

    def rollback_speculative_cache(self) -> None:
        if self.rollback_state is None:
            raise RuntimeError("GLM-5-Next KDA has no speculative rollback state")
        self.conv_state, self.recurrent_state = self.rollback_state
        self.rollback_state = None


class MlxGlm5NextSparseAttention:
    """NoPE MLA and four-token k-pool DSA used by GLM-5.3-Flash."""

    def __init__(
        self,
        model: MlxNintModel,
        config: Glm5NextConfig,
        prefix: str,
        max_context: int,
    ) -> None:
        self.config = config
        self.max_context = int(max_context)
        self.first = MlxLinearGroup(
            (
                model.linear(prefix + ".q_a_proj.weight"),
                model.linear(prefix + ".kv_a_proj_with_mqa.weight"),
            )
        )
        self.q_a_norm = MlxRMSNorm(
            _dense_vector(model, prefix + ".q_a_layernorm.weight"),
            config.rms_norm_eps,
        )
        self.kv_a_norm = MlxRMSNorm(
            _dense_vector(model, prefix + ".kv_a_layernorm.weight"),
            config.rms_norm_eps,
        )
        self.q_b = model.linear(prefix + ".q_b_proj.weight")
        embed = model.routed(prefix + ".embed_q")
        unembed = model.routed(prefix + ".unembed_out")
        if (
            embed.n_experts != config.num_attention_heads
            or embed.neuron_len != config.qk_nope_head_dim
            or embed.out_per_expert != config.kv_lora_rank
            or unembed.n_experts != config.num_attention_heads
            or unembed.neuron_len != config.kv_lora_rank
            or unembed.out_per_expert != config.v_head_dim
        ):
            raise ValueError("GLM-5-Next absorbed MLA tensor shapes disagree")
        self.embed_query = embed
        self.unembed_output = unembed
        self.output = model.linear(prefix + ".o_proj.weight")
        self.index_query = model.linear(prefix + ".indexer.wq_b.weight")
        self.index_key = model.linear(prefix + ".indexer.wk.weight")
        self.index_weights = model.linear(prefix + ".indexer.weights_proj.weight")
        self.index_norm_weight = _dense_vector(
            model,
            prefix + ".indexer.k_norm.weight",
        )
        self.index_norm_bias = _dense_vector(
            model,
            prefix + ".indexer.k_norm.bias",
        )
        self.index_gate = _dense_array(
            model,
            prefix + ".indexer.index_kpool_compress_gate",
        )
        self.index_ape = _dense_array(
            model,
            prefix + ".indexer.index_kpool_compress_ape",
        )
        self.latent_cache = _MlxSequenceCache(max_context, config.kv_lora_rank)
        self.index_cache = _MlxSequenceCache(max_context, 2 * config.index_head_dim)

    def reset_cache(self, batch: int) -> None:
        self.latent_cache.reset(batch)
        self.index_cache.reset(batch)

    @staticmethod
    def _headwise(
        layer: MlxRoutedLinear,
        value: mx.array,
        output_width: int,
    ) -> mx.array:
        batch, tokens, heads, width = (int(item) for item in value.shape)
        if heads != layer.n_experts:
            raise ValueError("GLM-5-Next head-wise projection count mismatch")
        rows = batch * tokens * heads
        ids = mx.tile(mx.arange(heads, dtype=mx.int32), batch * tokens).reshape(
            rows,
            1,
        )
        output = layer(value.reshape(rows, width), ids)
        return output.reshape(batch, tokens, heads, output_width)

    def _selected_indices(
        self,
        query: mx.array,
        head_weights: mx.array,
        packed_cache: mx.array,
        query_offset: int,
    ) -> mx.array:
        config = self.config
        batch, tokens = (int(item) for item in query.shape[:2])
        keys = packed_cache[..., : config.index_head_dim]
        gates = packed_cache[..., config.index_head_dim :]
        pooled = glm5_kpool_states(
            keys,
            gates,
            self.index_ape,
            pool_size=config.index_kpool,
        )
        pools = int(pooled.shape[1])
        select_count = min(config.index_topk // config.index_kpool, pools)
        absolute = mx.arange(tokens, dtype=mx.int32) + int(query_offset)
        if select_count:
            scores = glm5_kpool_scores(query, pooled, head_weights)
            pool_ends = (
                mx.arange(pools, dtype=mx.int32) * config.index_kpool + config.index_kpool - 1
            )
            visible = pool_ends[None, None, :] <= absolute[None, :, None]
            visible = mx.broadcast_to(visible, (batch, tokens, pools))
            scores = mx.where(visible, scores, mx.array(-1e30, dtype=mx.float32))
            partition = mx.argpartition(scores, pools - select_count, axis=-1)
            selected_pools = partition[..., -select_count:].astype(mx.int32)
            selected_valid = mx.take_along_axis(
                visible,
                selected_pools,
                axis=-1,
            )
            offsets = mx.arange(config.index_kpool, dtype=mx.int32)
            selected = (
                selected_pools[..., None] * config.index_kpool + offsets[None, None, None, :]
            ).reshape(batch, tokens, -1)
            selected = mx.where(
                mx.broadcast_to(
                    selected_valid[..., None],
                    (*selected_valid.shape, config.index_kpool),
                ).reshape(batch, tokens, -1),
                selected,
                mx.array(-1, dtype=mx.int32),
            )
        else:
            selected = mx.full((batch, tokens, 0), -1, dtype=mx.int32)

        selected_width = int(selected.shape[-1])
        if selected_width < config.index_topk:
            selected = mx.concatenate(
                (
                    selected,
                    mx.full(
                        (batch, tokens, config.index_topk - selected_width),
                        -1,
                        dtype=mx.int32,
                    ),
                ),
                axis=-1,
            )
        else:
            selected = selected[..., : config.index_topk]

        if config.index_kpool_always_select_tail and config.index_kpool > 1:
            tail_width = config.index_kpool - 1
            tail_count = (absolute + 1) % config.index_kpool
            tail_start = absolute + 1 - tail_count
            tail_offsets = mx.arange(tail_width, dtype=mx.int32)
            tail = tail_start[:, None] + tail_offsets[None, :]
            tail = mx.where(
                tail_offsets[None, :] < tail_count[:, None],
                tail,
                mx.array(-1, dtype=mx.int32),
            )
            selected = mx.concatenate(
                (selected, mx.broadcast_to(tail[None], (batch, tokens, tail_width))),
                axis=-1,
            )
        return selected

    def __call__(self, hidden_states: mx.array, *, use_cache: bool) -> mx.array:
        config = self.config
        batch, tokens = (int(item) for item in hidden_states.shape[:2])
        q_reduced, kv_reduced = self.first(hidden_states)
        q_reduced = self.q_a_norm(q_reduced)
        query = self.q_b(q_reduced).reshape(
            batch,
            tokens,
            config.num_attention_heads,
            config.qk_nope_head_dim,
        )
        latent = self.kv_a_norm(kv_reduced[..., : config.kv_lora_rank])

        index_key = glm_dsa_indexer_layer_norm(
            self.index_key(hidden_states),
            self.index_norm_weight,
            self.index_norm_bias,
            1e-6,
        )
        gate_scores = mx.matmul(
            hidden_states,
            mx.swapaxes(self.index_gate, -1, -2),
        )
        packed = mx.concatenate((index_key, gate_scores), axis=-1)
        if use_cache:
            latent_cache, query_offset = self.latent_cache.append(latent)
            index_cache, index_offset = self.index_cache.append(packed)
            if index_offset != query_offset:
                raise RuntimeError("GLM-5-Next MLA/index caches diverged")
        else:
            latent_cache = latent
            index_cache = packed
            query_offset = 0

        absorbed = self._headwise(
            self.embed_query,
            query,
            config.kv_lora_rank,
        )
        absorbed = mx.transpose(absorbed, (0, 2, 1, 3)).astype(mx.float32)
        logical_length = int(latent_cache.shape[1])
        attention_scale = 1.0 / config.qk_nope_head_dim**0.5
        if logical_length <= config.index_topk:
            attended = glm5_dense_mla_attention(
                absorbed,
                latent_cache,
                query_offset=query_offset,
                scale=attention_scale,
            )
        else:
            index_query = self.index_query(q_reduced).reshape(
                batch,
                tokens,
                config.index_n_heads,
                config.index_head_dim,
            )
            head_weights = self.index_weights(hidden_states)
            selected = self._selected_indices(
                index_query,
                head_weights,
                index_cache,
                query_offset,
            )
            attended = glm5_sparse_mla_attention(
                absorbed,
                latent_cache,
                selected,
                scale=attention_scale,
            )
        value = self._headwise(
            self.unembed_output,
            attended.astype(mx.float16),
            config.v_head_dim,
        )
        return self.output(
            value.reshape(
                batch,
                tokens,
                config.num_attention_heads * config.v_head_dim,
            )
        )


class MlxGlm5NextLayer:
    """One GLM-5-Next decoder layer including both mHC sites."""

    def __init__(
        self,
        model: MlxNintModel,
        config: Glm5NextConfig,
        names: MlxGlm5NextNames,
        index: int,
        max_context: int,
    ) -> None:
        self.config = config
        prefix = names.layer(index)
        self.attention_hc = MlxGlm5NextMhc(model, config, prefix, "attn")
        self.ffn_hc = MlxGlm5NextMhc(model, config, prefix, "ffn")
        self.attention_norm = MlxRMSNorm(
            _dense_vector(model, prefix + ".input_layernorm.weight"),
            config.rms_norm_eps,
        )
        self.ffn_norm = MlxRMSNorm(
            _dense_vector(model, prefix + ".post_attention_layernorm.weight"),
            config.rms_norm_eps,
        )
        attention_prefix = prefix + ".self_attn"
        self.attention = (
            MlxGlm5NextKda(model, config, attention_prefix)
            if config.layer_types[index] == "linear_attention"
            else MlxGlm5NextSparseAttention(
                model,
                config,
                attention_prefix,
                max_context,
            )
        )
        mlp_prefix = prefix + ".mlp"
        self.ffn = (
            MlxGlm5NextMoE(model, config, mlp_prefix)
            if config.mlp_layer_types[index] == "sparse"
            else MlxGlm5NextDenseFFN(model, mlp_prefix, config.swiglu_limit)
        )

    def reset_cache(self, batch: int) -> None:
        self.attention.reset_cache(batch)

    def __call__(
        self,
        hidden_streams: mx.array,
        *,
        use_cache: bool,
        n_confirmed: int = 0,
    ) -> mx.array:
        residual = hidden_streams
        post, combination, branch = self.attention_hc.pre(hidden_streams)
        branch = self.attention(
            self.attention_norm(branch),
            use_cache=use_cache,
            **(
                {"n_confirmed": n_confirmed}
                if isinstance(self.attention, MlxGlm5NextKda)
                else {}
            ),
        )
        hidden_streams = self.attention_hc.post(
            branch,
            residual,
            post,
            combination,
        )

        residual = hidden_streams
        post, combination, branch = self.ffn_hc.pre(hidden_streams)
        branch = self.ffn(self.ffn_norm(branch))
        return self.ffn_hc.post(branch, residual, post, combination)


class MlxGlm5NextMtpLayer:
    """One appended GLM-5-Next MTP layer without target-model mHC."""

    def __init__(
        self,
        model: MlxNintModel,
        config: Glm5NextConfig,
        prefix: str,
        max_context: int,
    ) -> None:
        self.config = config
        self.enorm = MlxRMSNorm(
            _dense_vector(model, prefix + ".enorm.weight"),
            config.rms_norm_eps,
        )
        self.hnorm = MlxRMSNorm(
            _dense_vector(model, prefix + ".hnorm.weight"),
            config.rms_norm_eps,
        )
        self.eh_projection = model.linear(prefix + ".eh_proj.weight")
        self.input_norm = MlxRMSNorm(
            _dense_vector(model, prefix + ".input_layernorm.weight"),
            config.rms_norm_eps,
        )
        self.post_attention_norm = MlxRMSNorm(
            _dense_vector(model, prefix + ".post_attention_layernorm.weight"),
            config.rms_norm_eps,
        )
        self.attention = MlxGlm5NextSparseAttention(
            model,
            config,
            prefix + ".self_attn",
            max_context,
        )
        self.ffn = MlxGlm5NextMoE(model, config, prefix + ".mlp")
        self.shared_head_norm = MlxRMSNorm(
            _dense_vector(model, prefix + ".shared_head.norm.weight"),
            config.rms_norm_eps,
        )

    def reset_cache(self, batch: int) -> None:
        self.attention.reset_cache(batch)

    def __call__(
        self,
        inputs_embeds: mx.array,
        previous_hidden_states: mx.array,
        positions: mx.array,
        *,
        use_cache: bool,
    ) -> mx.array:
        if positions.ndim == 1:
            mask = positions[None, :, None] == 0
        elif positions.ndim == 2:
            mask = positions[..., None] == 0
        else:
            raise ValueError("GLM-5-Next MTP positions must have [T] or [B,T] shape")
        masked_embeds = mx.where(mask, mx.zeros_like(inputs_embeds), inputs_embeds)
        fused_input = mx.concatenate(
            (
                self.enorm(masked_embeds),
                self.hnorm(previous_hidden_states),
            ),
            axis=-1,
        )
        hidden = self.eh_projection(fused_input)
        residual = hidden
        attention = self.attention(self.input_norm(hidden), use_cache=use_cache)
        # Dense source checkpoints retain this projection as BF16, while the
        # fused residual/RMSNorm kernel deliberately executes activations in
        # FP16 or FP32.  Preserve FP32 debug/reference graphs and lower BF16
        # activations to the production FP16 path explicitly.
        activation_dtype = (
            mx.float32 if hidden.dtype == mx.float32 else mx.float16
        )
        residual, hidden = self.post_attention_norm.add_and_forward(
            residual,
            attention,
            normalized_dtype=activation_dtype,
        )
        ffn = self.ffn(hidden)
        _residual, normalized = self.shared_head_norm.add_and_forward(
            residual,
            ffn,
            normalized_dtype=hidden.dtype,
        )
        return normalized


class MlxGlm5NextMtp:
    """Native GLM-5.3-Flash appended-layer MTP draft runtime."""

    def __init__(
        self,
        tensors: Mapping[str, MfqTensor] | MlxNintModel,
        config: Glm5NextConfig,
        names: MlxGlm5NextNames | None = None,
        *,
        max_context: int = 4096,
    ) -> None:
        if config.num_nextn_predict_layers <= 0:
            raise ValueError("GLM-5-Next checkpoint does not declare MTP layers")
        self.model = tensors if isinstance(tensors, MlxNintModel) else MlxNintModel(tensors)
        self.config = config
        self.names = MlxGlm5NextNames() if names is None else names
        self.max_context = min(int(max_context), config.max_position_embeddings)
        if self.max_context <= 0:
            raise ValueError("GLM-5-Next MTP max_context must be positive")
        self.embedding = self.model.embedding(self.names.token_embedding)
        self.output = self.model.linear(
            self.names.token_embedding
            if config.tie_word_embeddings or self.names.output not in self.model.tensors
            else self.names.output
        )
        self.layers = tuple(
            MlxGlm5NextMtpLayer(
                self.model,
                config,
                self.names.layer(config.num_hidden_layers + index),
                self.max_context,
            )
            for index in range(config.num_nextn_predict_layers)
        )
        self.batch = 0
        self.layer_positions = [0] * len(self.layers)

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        config: Glm5NextConfig | Mapping[str, object] | None = None,
        names: MlxGlm5NextNames | None = None,
        *,
        mmap: bool = True,
        max_context: int = 4096,
    ) -> MlxGlm5NextMtp:
        """Load the appended native MTP layer from a converted MFQ model."""

        model = MlxNintModel.from_mfq(path, mmap=mmap)
        try:
            selected = config
            if selected is None:
                if MODEL_CONFIG_ASSET not in model.tensors:
                    raise ValueError(
                        "MFQ has no embedded model config; pass Glm5NextConfig explicitly"
                    )
                payload = model.tensors[MODEL_CONFIG_ASSET]
                if not isinstance(payload, bytes):
                    raise TypeError("embedded MFQ model config must be a BLOB record")
                selected = json.loads(payload)
            normalized = (
                selected
                if isinstance(selected, Glm5NextConfig)
                else Glm5NextConfig.from_hf_config(selected)
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
            raise ValueError("GLM-5-Next MTP cache batch must be positive")
        for layer in self.layers:
            layer.reset_cache(selected)
        self.batch = selected
        self.layer_positions = [0] * len(self.layers)

    def forward(
        self,
        input_ids: mx.array | np.ndarray,
        previous_hidden_states: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        inputs_embeds: mx.array | np.ndarray | None = None,
        spec_step_index: int = 0,
        use_cache: bool = False,
    ) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        if ids.ndim != 2:
            raise ValueError("GLM-5-Next MTP IDs must have [batch,tokens] shape")
        batch, tokens = (int(item) for item in ids.shape)
        hidden = (
            previous_hidden_states
            if isinstance(previous_hidden_states, mx.array)
            else mx.array(previous_hidden_states)
        )
        if tuple(hidden.shape) != (batch, tokens, self.config.hidden_size):
            raise ValueError(
                "GLM-5-Next MTP previous hidden must have [batch,tokens,hidden] shape"
            )
        embeds = self.embedding(ids.astype(mx.int32)) if inputs_embeds is None else inputs_embeds
        embeds = embeds if isinstance(embeds, mx.array) else mx.array(embeds)
        if tuple(embeds.shape) != tuple(hidden.shape):
            raise ValueError("GLM-5-Next MTP embeddings have an incompatible shape")
        if use_cache and self.batch not in (0, batch):
            self.reset_cache(batch)
        layer_index = int(spec_step_index) % len(self.layers)
        start = self.layer_positions[layer_index] if use_cache else 0
        if positions is None:
            selected_positions = mx.arange(start, start + tokens, dtype=mx.int32)
        else:
            selected_positions = positions if isinstance(positions, mx.array) else mx.array(positions)
            if selected_positions.ndim not in (1, 2) or int(selected_positions.shape[-1]) != tokens:
                raise ValueError("GLM-5-Next MTP positions must have [T] or [B,T] shape")
            selected_positions = selected_positions.astype(mx.int32)
        output = self.layers[layer_index](
            embeds,
            hidden,
            selected_positions,
            use_cache=use_cache,
        )
        if use_cache:
            self.batch = batch
            self.layer_positions[layer_index] = start + tokens
        return output

    def compute_logits(self, hidden_states: mx.array | np.ndarray) -> mx.array:
        hidden = hidden_states if isinstance(hidden_states, mx.array) else mx.array(hidden_states)
        if hidden.ndim != 3 or int(hidden.shape[-1]) != self.config.hidden_size:
            raise ValueError("GLM-5-Next MTP hidden states have an incompatible shape")
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
    ) -> mx.array:
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


class MlxGlm5Next:
    """Complete text decoder for GLM-5.3-Flash MFQ checkpoints."""

    def __init__(
        self,
        tensors: Mapping[str, MfqTensor] | MlxNintModel,
        config: Glm5NextConfig,
        names: MlxGlm5NextNames | None = None,
        *,
        max_context: int = 4096,
    ) -> None:
        self.model = tensors if isinstance(tensors, MlxNintModel) else MlxNintModel(tensors)
        self.config = config
        self.names = MlxGlm5NextNames() if names is None else names
        self.max_context = min(int(max_context), config.max_position_embeddings)
        if self.max_context <= 0:
            raise ValueError("GLM-5-Next max_context must be positive")
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
        self.layers = tuple(
            MlxGlm5NextLayer(
                self.model,
                config,
                self.names,
                index,
                self.max_context,
            )
            for index in range(config.num_hidden_layers)
        )
        self.batch = 0
        self.position = 0
        self._speculative_checkpoint: tuple[int, int, int] | None = None

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        config: Glm5NextConfig | Mapping[str, object] | None = None,
        names: MlxGlm5NextNames | None = None,
        *,
        mmap: bool = True,
        max_context: int = 4096,
    ) -> MlxGlm5Next:
        model = MlxNintModel.from_mfq(path, mmap=mmap)
        try:
            selected = config
            if selected is None:
                if MODEL_CONFIG_ASSET not in model.tensors:
                    raise ValueError(
                        "MFQ has no embedded model config; pass Glm5NextConfig explicitly"
                    )
                payload = model.tensors[MODEL_CONFIG_ASSET]
                if not isinstance(payload, bytes):
                    raise TypeError("embedded MFQ model config must be a BLOB record")
                selected = json.loads(payload)
            normalized = (
                selected
                if isinstance(selected, Glm5NextConfig)
                else Glm5NextConfig.from_hf_config(selected)
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
            raise ValueError("GLM-5-Next cache batch must be positive")
        for layer in self.layers:
            layer.reset_cache(selected)
        self.batch = selected
        self.position = 0
        self._speculative_checkpoint = None

    def commit_speculative_cache(self) -> None:
        """Keep the latest verify window and release recurrent rollback refs."""

        for layer in self.layers:
            if isinstance(layer.attention, MlxGlm5NextKda):
                layer.attention.commit_speculative_cache()
        self._speculative_checkpoint = None

    def rollback_speculative_cache(self, accepted_drafts: int = 0) -> None:
        """Drop a rejected depth-one draft while retaining its confirmed token."""

        if self._speculative_checkpoint is None:
            raise RuntimeError("GLM-5-Next has no speculative verify window")
        start, confirmed, tokens = self._speculative_checkpoint
        accepted = int(accepted_drafts)
        if accepted != 0 or confirmed <= 0 or tokens != confirmed + 1:
            raise ValueError("GLM-5-Next currently rolls back one rejected draft")
        keep = start + confirmed
        for layer in self.layers:
            if isinstance(layer.attention, MlxGlm5NextKda):
                layer.attention.rollback_speculative_cache()
            else:
                layer.attention.latent_cache.position = keep
                layer.attention.index_cache.position = keep
        self.position = keep
        self._speculative_checkpoint = None

    def forward_embeddings_with_hidden(
        self,
        embeddings: mx.array | np.ndarray,
        *,
        use_cache: bool = False,
        n_confirmed: int = 0,
    ) -> tuple[mx.array, mx.array]:
        hidden = embeddings if isinstance(embeddings, mx.array) else mx.array(embeddings)
        if hidden.ndim != 3 or int(hidden.shape[-1]) != self.config.hidden_size:
            raise ValueError("GLM-5-Next embeddings must have [batch,tokens,hidden] shape")
        batch, tokens = (int(item) for item in hidden.shape[:2])
        if tokens <= 0:
            raise ValueError("GLM-5-Next input cannot be empty")
        if use_cache and self.batch not in (0, batch):
            self.reset_cache(batch)
        start = self.position if use_cache else 0
        if start + tokens > self.max_context:
            raise ValueError("GLM-5-Next input exceeds max_context")
        confirmed = int(n_confirmed)
        if not 0 <= confirmed <= tokens:
            raise ValueError("GLM-5-Next confirmed-token count is outside the input")
        if confirmed and (not use_cache or confirmed == tokens):
            raise ValueError("GLM-5-Next speculative verification needs a draft suffix")
        self.commit_speculative_cache()
        if confirmed:
            self._speculative_checkpoint = (start, confirmed, tokens)
        streams = mx.broadcast_to(
            hidden.astype(mx.float16)[..., None, :],
            (batch, tokens, self.config.hc_mult, self.config.hidden_size),
        )
        streams = mx.contiguous(streams)
        for layer in self.layers:
            streams = layer(
                streams,
                use_cache=use_cache,
                n_confirmed=confirmed,
            )
        collapsed = mx.mean(streams, axis=-2)
        normalized = self.output_norm(collapsed)
        logits = self.output(normalized)
        if use_cache:
            self.batch = batch
            self.position = start + tokens
        return logits, normalized

    def forward_embeddings(
        self,
        embeddings: mx.array | np.ndarray,
        *,
        use_cache: bool = False,
        n_confirmed: int = 0,
    ) -> mx.array:
        logits, _hidden_states = self.forward_embeddings_with_hidden(
            embeddings,
            use_cache=use_cache,
            n_confirmed=n_confirmed,
        )
        return logits

    def forward_with_hidden(
        self,
        input_ids: mx.array | np.ndarray,
        *,
        use_cache: bool = False,
        n_confirmed: int = 0,
    ) -> tuple[mx.array, mx.array]:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        if ids.ndim != 2:
            raise ValueError("GLM-5-Next IDs must have [batch,tokens] shape")
        return self.forward_embeddings_with_hidden(
            self.embedding(ids.astype(mx.int32)),
            use_cache=use_cache,
            n_confirmed=n_confirmed,
        )

    def forward(
        self,
        input_ids: mx.array | np.ndarray,
        *,
        use_cache: bool = False,
        n_confirmed: int = 0,
    ) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        if ids.ndim != 2:
            raise ValueError("GLM-5-Next IDs must have [batch,tokens] shape")
        return self.forward_embeddings(
            self.embedding(ids.astype(mx.int32)),
            use_cache=use_cache,
            n_confirmed=n_confirmed,
        )

    def __call__(
        self,
        input_ids: mx.array | np.ndarray,
        *,
        use_cache: bool = False,
        n_confirmed: int = 0,
    ) -> mx.array:
        return self.forward(
            input_ids,
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
            raise ValueError("GLM-5-Next decode accepts one token per batch")
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
            raise ValueError("GLM-5-Next generation IDs must have [batch,tokens] shape")
        if int(ids.shape[1]) + int(max_new_tokens) > self.max_context:
            raise ValueError("GLM-5-Next generation exceeds max_context")
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

    def __enter__(self) -> MlxGlm5Next:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "MlxGlm5Next",
    "MlxGlm5NextDenseFFN",
    "MlxGlm5NextKda",
    "MlxGlm5NextLayer",
    "MlxGlm5NextMtp",
    "MlxGlm5NextMtpLayer",
    "MlxGlm5NextMhc",
    "MlxGlm5NextMoE",
    "MlxGlm5NextNames",
    "MlxGlm5NextSparseAttention",
]
