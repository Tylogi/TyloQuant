"""End-to-end full-attention causal LM assembled from MFQ MLX primitives."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
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
from mfq.kernels.metal.linear_attention import (
    gated_delta_net,
    linear_conv_qkv,
)
from mfq.kernels.metal.sampling import sample as _sample
from mfq.runtime.mlx_attention import MlxKVCache, attention
from mfq.runtime.mlx_linear import (
    MlxLinearGroup,
    MlxNintLinear,
    MlxNintModel,
    mlx_dense_array,
)
from mfq.runtime.mlx_ops import MlxRMSNorm, MlxRoPE


@dataclass(frozen=True)
class MlxCausalLMConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    attention_head_dim: int | None = None
    rope_base: float = 1_000_000.0
    rotary_dim: int | None = None
    rope_sections: tuple[int, int, int] | None = None
    rms_norm_eps: float = 1e-6
    norm_weight_offset: float = 0.0
    tie_word_embeddings: bool = False
    attention_output_gate: bool = False
    layer_types: tuple[str, ...] | None = None
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_a_is_log: bool = True

    @property
    def head_dim(self) -> int:
        if self.attention_head_dim is not None:
            return int(self.attention_head_dim)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must divide num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def effective_rotary_dim(self) -> int:
        return self.head_dim if self.rotary_dim is None else int(self.rotary_dim)

    @classmethod
    def from_qwen35_hf_config(cls, config: dict) -> MlxCausalLMConfig:
        """Build an MLX configuration from a Qwen3.5 Hugging Face config."""

        text = config.get("text_config", config)
        rope_parameters = text.get("rope_parameters", {})
        head_dim = int(
            text.get(
                "head_dim",
                int(text["hidden_size"]) // int(text["num_attention_heads"]),
            )
        )
        partial = float(
            text.get(
                "partial_rotary_factor",
                rope_parameters.get("partial_rotary_factor", 1.0),
            )
        )
        return cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            attention_head_dim=head_dim,
            rope_base=float(rope_parameters.get("rope_theta", 1_000_000.0)),
            rotary_dim=int(round(partial * head_dim)),
            rope_sections=tuple(int(value) for value in rope_parameters.get("mrope_section", ()))
            or None,
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            norm_weight_offset=1.0,
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            attention_output_gate=bool(text.get("attn_output_gate", False)),
            layer_types=tuple(
                text.get(
                    "layer_types",
                    ("full_attention",) * int(text["num_hidden_layers"]),
                )
            ),
            linear_conv_kernel_dim=int(text.get("linear_conv_kernel_dim", 4)),
            linear_key_head_dim=int(text.get("linear_key_head_dim", 128)),
            linear_value_head_dim=int(text.get("linear_value_head_dim", 128)),
            linear_num_key_heads=int(text.get("linear_num_key_heads", 0)),
            linear_num_value_heads=int(text.get("linear_num_value_heads", 0)),
            linear_a_is_log=True,
        )


@dataclass(frozen=True)
class MlxCausalLMNames:
    token_embd: str = "token_embd.weight"
    attn_norm: str = "blk.{i}.attn_norm.weight"
    attn_q: str = "blk.{i}.attn_q.weight"
    attn_k: str = "blk.{i}.attn_k.weight"
    attn_v: str = "blk.{i}.attn_v.weight"
    attn_out: str = "blk.{i}.attn_output.weight"
    attn_q_norm: str = "blk.{i}.attn_q_norm.weight"
    attn_k_norm: str = "blk.{i}.attn_k_norm.weight"
    ffn_norm: str = "blk.{i}.ffn_norm.weight"
    ffn_gate: str = "blk.{i}.ffn_gate.weight"
    ffn_up: str = "blk.{i}.ffn_up.weight"
    ffn_down: str = "blk.{i}.ffn_down.weight"
    output_norm: str = "output_norm.weight"
    output: str = "output.weight"
    linear_qkv: str = "blk.{i}.ssm_qkv.weight"
    linear_qk: str | None = "blk.{i}.ssm_qk.weight"
    linear_v: str | None = "blk.{i}.ssm_v.weight"
    linear_z: str = "blk.{i}.ssm_z.weight"
    linear_alpha: str = "blk.{i}.ssm_alpha.weight"
    linear_beta: str = "blk.{i}.ssm_beta.weight"
    linear_conv: str = "blk.{i}.ssm_conv1d.weight"
    linear_conv_bias: str | None = None
    linear_dt_bias: str = "blk.{i}.ssm_dt.bias"
    linear_a: str = "blk.{i}.ssm_a"
    linear_norm: str = "blk.{i}.ssm_norm.weight"
    linear_out: str = "blk.{i}.ssm_out.weight"

    def layer(self, template: str, index: int) -> str:
        return template.format(i=index)

    @classmethod
    def qwen35_gguf(cls) -> MlxCausalLMNames:
        return cls(
            ffn_norm="blk.{i}.post_attention_norm.weight",
            linear_qkv="blk.{i}.attn_qkv.weight",
            linear_z="blk.{i}.attn_gate.weight",
        )

    @classmethod
    def qwen35_hf(cls) -> MlxCausalLMNames:
        return cls(
            token_embd="model.language_model.embed_tokens.weight",
            attn_norm="model.language_model.layers.{i}.input_layernorm.weight",
            attn_q="model.language_model.layers.{i}.self_attn.q_proj.weight",
            attn_k="model.language_model.layers.{i}.self_attn.k_proj.weight",
            attn_v="model.language_model.layers.{i}.self_attn.v_proj.weight",
            attn_out="model.language_model.layers.{i}.self_attn.o_proj.weight",
            attn_q_norm="model.language_model.layers.{i}.self_attn.q_norm.weight",
            attn_k_norm="model.language_model.layers.{i}.self_attn.k_norm.weight",
            ffn_norm="model.language_model.layers.{i}.post_attention_layernorm.weight",
            ffn_gate="model.language_model.layers.{i}.mlp.gate_proj.weight",
            ffn_up="model.language_model.layers.{i}.mlp.up_proj.weight",
            ffn_down="model.language_model.layers.{i}.mlp.down_proj.weight",
            output_norm="model.language_model.norm.weight",
            output="lm_head.weight",
            linear_qkv=("model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight"),
            linear_qk="model.language_model.layers.{i}.linear_attn.in_proj_qk.weight",
            linear_v="model.language_model.layers.{i}.linear_attn.in_proj_v.weight",
            linear_z="model.language_model.layers.{i}.linear_attn.in_proj_z.weight",
            linear_alpha="model.language_model.layers.{i}.linear_attn.in_proj_a.weight",
            linear_beta="model.language_model.layers.{i}.linear_attn.in_proj_b.weight",
            linear_conv="model.language_model.layers.{i}.linear_attn.conv1d.weight",
            linear_conv_bias="model.language_model.layers.{i}.linear_attn.conv1d.bias",
            linear_dt_bias="model.language_model.layers.{i}.linear_attn.dt_bias",
            linear_a="model.language_model.layers.{i}.linear_attn.A_log",
            linear_norm="model.language_model.layers.{i}.linear_attn.norm.weight",
            linear_out="model.language_model.layers.{i}.linear_attn.out_proj.weight",
        )


def _dense_vector(model: MlxNintModel, name: str) -> mx.array:
    if name not in model.tensors:
        raise KeyError(f"tensor {name!r} is not present in the MFQ model")
    value = model.tensors[name]
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise TypeError(f"runtime tensor {name!r} must be a dense vector")
    return mlx_dense_array(value, dtype=mx.float32)


def _dense_array(model: MlxNintModel, name: str) -> mx.array:
    if name not in model.tensors:
        raise KeyError(f"tensor {name!r} is not present in the MFQ model")
    value = model.tensors[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"runtime tensor {name!r} must be a dense array")
    return mlx_dense_array(value, dtype=mx.float32)


def _optional_norm(
    model: MlxNintModel,
    name: str,
    config: MlxCausalLMConfig,
) -> MlxRMSNorm | None:
    if name not in model.tensors:
        return None
    return MlxRMSNorm(
        _dense_vector(model, name),
        config.rms_norm_eps,
        weight_offset=config.norm_weight_offset,
    )


class MlxFullAttentionBlock:
    """One full-attention decoder block with packed projections."""

    def __init__(
        self,
        model: MlxNintModel,
        config: MlxCausalLMConfig,
        names: MlxCausalLMNames,
        layer_index: int,
    ) -> None:
        self.config = config
        self.attn_norm = MlxRMSNorm(
            _dense_vector(model, names.layer(names.attn_norm, layer_index)),
            config.rms_norm_eps,
            weight_offset=config.norm_weight_offset,
        )
        self.ffn_norm = MlxRMSNorm(
            _dense_vector(model, names.layer(names.ffn_norm, layer_index)),
            config.rms_norm_eps,
            weight_offset=config.norm_weight_offset,
        )
        self.q_norm = _optional_norm(
            model,
            names.layer(names.attn_q_norm, layer_index),
            config,
        )
        self.k_norm = _optional_norm(
            model,
            names.layer(names.attn_k_norm, layer_index),
            config,
        )
        self.qkv = MlxLinearGroup(
            (
                model.linear(names.layer(names.attn_q, layer_index)),
                model.linear(names.layer(names.attn_k, layer_index)),
                model.linear(names.layer(names.attn_v, layer_index)),
            )
        )
        self.output = model.linear(names.layer(names.attn_out, layer_index))
        self.ffn = model.ffn(
            names.layer(names.ffn_gate, layer_index),
            names.layer(names.ffn_up, layer_index),
            names.layer(names.ffn_down, layer_index),
        )
        self.rope = MlxRoPE(
            config.effective_rotary_dim,
            config.max_position_embeddings,
            base=config.rope_base,
            sections=config.rope_sections,
        )
        self.cache: MlxKVCache | None = None

    def reset_cache(self, batch: int) -> None:
        config = self.config
        self.cache = MlxKVCache(
            batch,
            config.num_key_value_heads,
            config.max_position_embeddings,
            config.head_dim,
        )

    def forward(
        self,
        x: mx.array,
        positions: mx.array,
        *,
        use_cache: bool,
    ) -> mx.array:
        config = self.config
        batch, tokens, hidden = (int(item) for item in x.shape)
        normalized = self.attn_norm(x)
        query_full, key_full, value_full = self.qkv(normalized)
        if config.attention_output_gate:
            query_pair = query_full.reshape(
                batch,
                tokens,
                config.num_attention_heads,
                config.head_dim * 2,
            )
            query_raw, query_gate = mx.split(query_pair, 2, axis=-1)
        else:
            query_raw = query_full.reshape(
                batch,
                tokens,
                config.num_attention_heads,
                config.head_dim,
            )
            query_gate = None
        query = mx.transpose(query_raw, (0, 2, 1, 3))
        key = mx.transpose(
            key_full.reshape(
                batch,
                tokens,
                config.num_key_value_heads,
                config.head_dim,
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
        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)
        query = self.rope(query, positions)
        key = self.rope(key, positions)

        if use_cache:
            if self.cache is None:
                self.reset_cache(batch)
            assert self.cache is not None
            cache_positions = positions[0] if positions.ndim == 2 else positions
            key_cache, value_cache = self.cache.append(
                key,
                value,
                cache_positions,
            )
        else:
            key_cache, value_cache = key, value
        attended = attention(query, key_cache, value_cache, causal=True)
        attended = mx.transpose(attended, (0, 2, 1, 3)).reshape(
            batch,
            tokens,
            config.attention_size,
        )
        if query_gate is not None:
            gate = query_gate.reshape(batch, tokens, config.attention_size)
            attended = attended * mx.sigmoid(gate)
        x = x + self.output(attended)
        x = x + self.ffn(self.ffn_norm(x))
        if int(x.shape[-1]) != hidden:  # pragma: no cover - defensive
            raise ValueError("decoder block changed hidden width")
        return x

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        *,
        use_cache: bool,
    ) -> mx.array:
        return self.forward(x, positions, use_cache=use_cache)


class MlxQwen35LinearAttentionBlock:
    """Qwen3.5 Gated DeltaNet block backed by fused Metal kernels."""

    def __init__(
        self,
        model: MlxNintModel,
        config: MlxCausalLMConfig,
        names: MlxCausalLMNames,
        layer_index: int,
    ) -> None:
        self.config = config
        self.gguf_layout = names.linear_qkv == "blk.{i}.attn_qkv.weight"
        self.attn_norm = MlxRMSNorm(
            _dense_vector(model, names.layer(names.attn_norm, layer_index)),
            config.rms_norm_eps,
            weight_offset=config.norm_weight_offset,
        )
        self.ffn_norm = MlxRMSNorm(
            _dense_vector(model, names.layer(names.ffn_norm, layer_index)),
            config.rms_norm_eps,
            weight_offset=config.norm_weight_offset,
        )
        qk_name = names.layer(names.linear_qk, layer_index) if names.linear_qk is not None else None
        v_name = names.layer(names.linear_v, layer_index) if names.linear_v is not None else None
        self.split_input = (
            qk_name is not None
            and v_name is not None
            and qk_name in model.tensors
            and v_name in model.tensors
        )
        if self.split_input:
            assert qk_name is not None and v_name is not None
            self.qk_v = MlxLinearGroup((model.linear(qk_name), model.linear(v_name)))
            self.qkv = None
        else:
            self.qk_v = None
            self.qkv = model.linear(names.layer(names.linear_qkv, layer_index))
        self.zab = MlxLinearGroup(
            (
                model.linear(names.layer(names.linear_z, layer_index)),
                model.linear(names.layer(names.linear_alpha, layer_index)),
                model.linear(names.layer(names.linear_beta, layer_index)),
            )
        )
        self.conv_weight = _dense_array(
            model,
            names.layer(names.linear_conv, layer_index),
        )
        self.conv_bias = (
            None
            if names.linear_conv_bias is None
            or names.layer(names.linear_conv_bias, layer_index) not in model.tensors
            else _dense_vector(
                model,
                names.layer(names.linear_conv_bias, layer_index),
            )
        )
        self.dt_bias = _dense_vector(
            model,
            names.layer(names.linear_dt_bias, layer_index),
        )
        self.a = _dense_vector(model, names.layer(names.linear_a, layer_index))
        self.linear_norm = MlxRMSNorm(
            _dense_vector(model, names.layer(names.linear_norm, layer_index)),
            config.rms_norm_eps,
        )
        self.output = model.linear(names.layer(names.linear_out, layer_index))
        self.ffn = model.ffn(
            names.layer(names.ffn_gate, layer_index),
            names.layer(names.ffn_up, layer_index),
            names.layer(names.ffn_down, layer_index),
        )
        self.conv_state: mx.array | None = None
        self.gdn_state: mx.array | None = None
        self._cache_position = 0

    @property
    def key_heads(self) -> int:
        return self.config.linear_num_key_heads or self.config.num_key_value_heads

    @property
    def value_heads(self) -> int:
        return self.config.linear_num_value_heads or self.config.num_attention_heads

    @property
    def key_size(self) -> int:
        return self.key_heads * self.config.linear_key_head_dim

    @property
    def value_size(self) -> int:
        return self.value_heads * self.config.linear_value_head_dim

    @property
    def cache_pos(self) -> int:
        return self._cache_position

    def reset_cache(self, batch: int) -> None:
        channels = 2 * self.key_size + self.value_size
        dimension = self.config.linear_value_head_dim
        self.conv_state = mx.zeros(
            (
                int(batch),
                self.config.linear_conv_kernel_dim - 1,
                channels,
            ),
            dtype=mx.float32,
        )
        self.gdn_state = mx.zeros(
            (int(batch), self.value_heads, dimension, dimension),
            dtype=mx.float32,
        )
        self._cache_position = 0

    def forward(
        self,
        x: mx.array,
        positions: mx.array,
        *,
        use_cache: bool,
    ) -> mx.array:
        del positions
        config = self.config
        batch, tokens, hidden = (int(item) for item in x.shape)
        activation_dtype = x.dtype
        normalized = self.attn_norm(x)
        if self.split_input:
            assert self.qk_v is not None
            qk, value_input = self.qk_v(normalized)
        else:
            assert self.qkv is not None
            projected = self.qkv(normalized)
            qk, value_input = mx.split(
                projected,
                [2 * self.key_size],
                axis=-1,
            )
        z, alpha_raw, beta_raw = self.zab(normalized)
        beta = mx.sigmoid(beta_raw.astype(mx.float32)).reshape(
            batch,
            tokens,
            self.value_heads,
        )
        alpha = alpha_raw.astype(mx.float32).reshape(
            batch,
            tokens,
            self.value_heads,
        )
        gate_input = alpha + self.dt_bias.reshape(1, 1, -1)
        gate = mx.maximum(gate_input, 0.0) + mx.log1p(mx.exp(-mx.abs(gate_input)))
        a = -mx.exp(self.a) if config.linear_a_is_log else self.a
        gate = gate * a.reshape(1, 1, -1)

        if use_cache:
            if self.conv_state is None or self.gdn_state is None:
                self.reset_cache(batch)
            assert self.conv_state is not None and self.gdn_state is not None
            conv_state = self.conv_state
            gdn_state = self.gdn_state
        else:
            conv_state = mx.zeros(
                (
                    batch,
                    config.linear_conv_kernel_dim - 1,
                    2 * self.key_size + self.value_size,
                ),
                dtype=mx.float32,
            )
            gdn_state = None
        query, key, value, new_conv_state = linear_conv_qkv(
            conv_state,
            qk,
            value_input,
            self.conv_weight,
            num_key_heads=self.key_heads,
            num_value_heads=self.value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
            bias=self.conv_bias,
            eps=config.rms_norm_eps,
        )
        if config.linear_key_head_dim != config.linear_value_head_dim:
            raise ValueError("the GDN Metal kernel requires equal key and value head dimensions")
        attended, new_gdn_state = gated_delta_net(
            query,
            key,
            value,
            mx.transpose(gate, (0, 2, 1)),
            mx.transpose(beta, (0, 2, 1)),
            gdn_state,
            tiled_heads=self.gguf_layout,
        )
        if use_cache:
            self.conv_state = new_conv_state
            self.gdn_state = new_gdn_state
            self._cache_position += tokens
        normalized_value = self.linear_norm(attended)
        normalized_value = mx.transpose(
            normalized_value,
            (0, 2, 1, 3),
        ).reshape(batch, tokens, self.value_size)
        z = z.reshape(
            batch,
            tokens,
            self.value_heads,
            config.linear_value_head_dim,
        ).reshape(batch, tokens, self.value_size)
        gated_value = normalized_value * (z * mx.sigmoid(z))
        # The recurrent state and its reduction intentionally accumulate in
        # FP32, but Qwen's gated RMSNorm returns the original activation
        # dtype before the output projection.  Without this boundary the
        # first linear-attention layer promotes the entire residual stream
        # (and every later packed projection) to FP32.
        if gated_value.dtype != activation_dtype:
            gated_value = gated_value.astype(activation_dtype)
        x = x + self.output(gated_value)
        x = x + self.ffn(self.ffn_norm(x))
        if int(x.shape[-1]) != hidden:  # pragma: no cover - defensive
            raise ValueError("linear-attention block changed hidden width")
        return x

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        *,
        use_cache: bool,
    ) -> mx.array:
        return self.forward(x, positions, use_cache=use_cache)


class MlxCausalLM:
    """Full-attention MFQ causal language model returning logits."""

    def __init__(
        self,
        tensors: Mapping[str, MfqTensor] | MlxNintModel,
        config: MlxCausalLMConfig,
        names: MlxCausalLMNames | None = None,
    ) -> None:
        self.model = tensors if isinstance(tensors, MlxNintModel) else MlxNintModel(tensors)
        self.config = config
        self.names = MlxCausalLMNames() if names is None else names
        self.embedding = self.model.embedding(self.names.token_embd)
        layer_types = config.layer_types or (("full_attention",) * config.num_hidden_layers)
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError("layer_types length must match num_hidden_layers")
        unsupported = set(layer_types) - {"full_attention", "linear_attention"}
        if unsupported:
            raise ValueError(f"unsupported MLX layer types: {sorted(unsupported)}")
        self.layers = tuple(
            (
                MlxFullAttentionBlock(
                    self.model,
                    config,
                    self.names,
                    layer,
                )
                if layer_types[layer] == "full_attention"
                else MlxQwen35LinearAttentionBlock(
                    self.model,
                    config,
                    self.names,
                    layer,
                )
            )
            for layer in range(config.num_hidden_layers)
        )
        self.output_norm = MlxRMSNorm(
            _dense_vector(self.model, self.names.output_norm),
            config.rms_norm_eps,
            weight_offset=config.norm_weight_offset,
        )
        if config.tie_word_embeddings:
            self.output = MlxNintLinear.from_packed_weight(self.embedding.packed_weight)
        else:
            self.output = self.model.linear(self.names.output)

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        config: MlxCausalLMConfig | None = None,
        names: MlxCausalLMNames | None = None,
        *,
        mmap: bool = True,
    ) -> MlxCausalLM:
        model = MlxNintModel.from_mfq(path, mmap=mmap)
        try:
            if config is None:
                if MODEL_CONFIG_ASSET not in model.tensors:
                    raise ValueError(
                        "MFQ has no embedded model config; pass MlxCausalLMConfig explicitly"
                    )
                payload = model.tensors[MODEL_CONFIG_ASSET]
                if not isinstance(payload, bytes):
                    raise TypeError("embedded MFQ model config must be a BLOB record")
                parsed = json.loads(payload)
                if not isinstance(parsed, dict):
                    raise ValueError("embedded MFQ model config must be a JSON object")
                config = MlxCausalLMConfig.from_qwen35_hf_config(parsed)
            if names is None:
                if "model.language_model.embed_tokens.weight" in model.tensors:
                    names = MlxCausalLMNames.qwen35_hf()
                elif "blk.0.post_attention_norm.weight" in model.tensors:
                    names = MlxCausalLMNames.qwen35_gguf()
                else:
                    names = MlxCausalLMNames()
            if names.linear_qkv == "blk.{i}.attn_qkv.weight":
                config = replace(
                    config,
                    norm_weight_offset=0.0,
                    linear_a_is_log=False,
                )
            return cls(model, config, names)
        except BaseException:
            model.close()
            raise

    def reset_cache(self, batch: int) -> None:
        for layer in self.layers:
            layer.reset_cache(batch)

    def forward(
        self,
        input_ids: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        use_cache: bool = False,
    ) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim != 2:
            raise ValueError("causal LM input IDs must have [batch,tokens] shape")
        if ids.dtype not in (mx.int32, mx.uint32):
            ids = ids.astype(mx.int32)
        batch, tokens = (int(item) for item in ids.shape)
        if positions is None:
            start = 0
            if use_cache and self.layers:
                first_layer = self.layers[0]
                if isinstance(first_layer, MlxFullAttentionBlock):
                    if first_layer.cache is not None:
                        start = first_layer.cache.pos
                else:
                    start = first_layer.cache_pos
            position_array = mx.arange(start, start + tokens, dtype=mx.int32)
        else:
            position_array = positions if isinstance(positions, mx.array) else mx.array(positions)
            if position_array.dtype not in (mx.int32, mx.uint32):
                position_array = position_array.astype(mx.int32)
        hidden = self.embedding(ids)
        for layer in self.layers:
            hidden = layer(hidden, position_array, use_cache=use_cache)
        logits = self.output(self.output_norm(hidden))
        if tuple(int(item) for item in logits.shape[:2]) != (batch, tokens):
            raise ValueError("causal LM output leading dimensions are invalid")
        return logits

    def __call__(
        self,
        input_ids: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        use_cache: bool = False,
    ) -> mx.array:
        return self.forward(input_ids, positions, use_cache=use_cache)

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
            raise ValueError("generation input IDs must have [batch,tokens] shape")
        ids = ids.astype(mx.int32)
        self.reset_cache(int(ids.shape[0]))
        if int(max_new_tokens) <= 0:
            return ids
        logits = self.forward(ids, use_cache=True)
        pieces = [ids]
        next_id = _sample(
            logits[:, -1, :],
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
        )
        eos = (
            ()
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
                logits = self.forward(next_id[:, None], use_cache=True)
                next_id = _sample(
                    logits[:, -1, :],
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                )
        return mx.concatenate(pieces, axis=1)

    def close(self) -> None:
        self.model.close()

    def __enter__(self) -> MlxCausalLM:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "MlxCausalLM",
    "MlxCausalLMConfig",
    "MlxCausalLMNames",
    "MlxFullAttentionBlock",
    "MlxQwen35LinearAttentionBlock",
]
