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

from mfq.architectures.flash_next import VisionTowerConfig
from mfq.formats.assets import MODEL_CONFIG_ASSET
from mfq.formats.io import MfqTensor
from mfq.kernels.metal.linear_attention import (
    gated_delta_net,
    linear_conv_qkv,
    prepare_linear_conv_qkv,
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
    mrope_interleaved: bool = False
    mtp_num_hidden_layers: int = 0
    mtp_use_dedicated_embeddings: bool = False
    family: str = "generic"
    eos_token_ids: tuple[int, ...] = ()
    image_token_id: int | None = None
    video_token_id: int | None = None
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None
    vision: VisionTowerConfig | None = None

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
        raw_eos = text.get("eos_token_id")
        eos_token_ids = (
            ()
            if raw_eos is None
            else (
                (int(raw_eos),)
                if isinstance(raw_eos, int)
                else tuple(int(value) for value in raw_eos)
            )
        )
        raw_vision = config.get("vision_config")
        vision = (
            VisionTowerConfig.from_mapping(raw_vision)
            if isinstance(raw_vision, Mapping)
            else None
        )
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
        result = cls(
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
            mrope_interleaved=bool(rope_parameters.get("mrope_interleaved", False)),
            mtp_num_hidden_layers=int(text.get("mtp_num_hidden_layers", 0) or 0),
            mtp_use_dedicated_embeddings=bool(
                text.get("mtp_use_dedicated_embeddings", False)
            ),
            family="qwen3_5",
            eos_token_ids=eos_token_ids,
            image_token_id=(
                None if config.get("image_token_id") is None else int(config["image_token_id"])
            ),
            video_token_id=(
                None if config.get("video_token_id") is None else int(config["video_token_id"])
            ),
            vision_start_token_id=(
                None
                if config.get("vision_start_token_id") is None
                else int(config["vision_start_token_id"])
            ),
            vision_end_token_id=(
                None
                if config.get("vision_end_token_id") is None
                else int(config["vision_end_token_id"])
            ),
            vision=vision,
        )
        if vision is not None and vision.out_hidden_size != result.hidden_size:
            raise ValueError("Qwen3.5 vision output width must match hidden_size")
        return result


@dataclass(frozen=True)
class MlxCausalLMNames:
    token_embd: str = "model.token_embedding.weight"
    attn_norm: str = "model.block.{i}.attention.norm.weight"
    attn_q: str = "model.block.{i}.attention.query.weight"
    attn_k: str = "model.block.{i}.attention.key.weight"
    attn_v: str = "model.block.{i}.attention.value.weight"
    attn_out: str = "model.block.{i}.attention.output.weight"
    attn_q_norm: str = "model.block.{i}.attention.query_norm.weight"
    attn_k_norm: str = "model.block.{i}.attention.key_norm.weight"
    ffn_norm: str = "model.block.{i}.mlp.norm.weight"
    ffn_gate: str = "model.block.{i}.mlp.gate.weight"
    ffn_up: str = "model.block.{i}.mlp.up.weight"
    ffn_down: str = "model.block.{i}.mlp.down.weight"
    output_norm: str = "model.output_norm.weight"
    output: str = "model.output.weight"
    linear_qkv: str = "model.block.{i}.linear_attention.qkv.weight"
    linear_qk: str | None = "model.block.{i}.linear_attention.qk.weight"
    linear_v: str | None = "model.block.{i}.linear_attention.value.weight"
    linear_z: str = "model.block.{i}.linear_attention.gate.weight"
    linear_alpha: str = "model.block.{i}.linear_attention.alpha.weight"
    linear_beta: str = "model.block.{i}.linear_attention.beta.weight"
    linear_conv: str = "model.block.{i}.linear_attention.conv.weight"
    linear_conv_bias: str | None = "model.block.{i}.linear_attention.conv.bias"
    linear_dt_bias: str = "model.block.{i}.linear_attention.dt_bias"
    linear_a: str = "model.block.{i}.linear_attention.a"
    linear_norm: str = "model.block.{i}.linear_attention.norm.weight"
    linear_out: str = "model.block.{i}.linear_attention.output.weight"

    def layer(self, template: str, index: int) -> str:
        return template.format(i=index)



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
            ),
            # The heterogeneous grouped projection saves launches during
            # prefill, but its descriptor/branching overhead makes M=1 much
            # slower than the tuned per-weight GEMVs used for decode.
            grouped_min_rows=2,
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
            mrope_interleaved=config.mrope_interleaved,
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
        positions: mx.array | None,
        *,
        use_cache: bool,
        position_offset: int = 0,
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
        if positions is None:
            query = self.rope.forward_contiguous(query, position_offset)
            key = self.rope.forward_contiguous(key, position_offset)
        else:
            query = self.rope(query, positions)
            key = self.rope(key, positions)

        if use_cache:
            if self.cache is None:
                self.reset_cache(batch)
            assert self.cache is not None
            # Multimodal RoPE coordinates are logical attention positions,
            # not physical KV slots.  Cache storage always follows prompt
            # order even when the three MRoPE axes repeat within an image.
            key_cache, value_cache = self.cache.append(key, value)
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
        positions: mx.array | None,
        *,
        use_cache: bool,
        position_offset: int = 0,
    ) -> mx.array:
        return self.forward(
            x,
            positions,
            use_cache=use_cache,
            position_offset=position_offset,
        )


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
        self.gguf_layout = model.legacy_semantics == "qwen35_gguf"
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
            self.qk_v = MlxLinearGroup(
                (model.linear(qk_name), model.linear(v_name)),
                grouped_min_rows=2,
            )
            self.qkv = None
        else:
            self.qk_v = None
            self.qkv = model.linear(names.layer(names.linear_qkv, layer_index))
        self.zab = MlxLinearGroup(
            (
                model.linear(names.layer(names.linear_z, layer_index)),
                model.linear(names.layer(names.linear_alpha, layer_index)),
                model.linear(names.layer(names.linear_beta, layer_index)),
            ),
            grouped_min_rows=2,
        )
        conv_weight = _dense_array(
            model,
            names.layer(names.linear_conv, layer_index),
        )
        conv_bias = (
            None
            if names.linear_conv_bias is None
            or names.layer(names.linear_conv_bias, layer_index) not in model.tensors
            else _dense_vector(
                model,
                names.layer(names.linear_conv_bias, layer_index),
            )
        )
        self.conv_parameters = prepare_linear_conv_qkv(
            conv_weight,
            conv_bias,
            channels=2 * self.key_size + self.value_size,
            eps=config.rms_norm_eps,
        )
        self.dt_bias = mx.contiguous(
            _dense_vector(
                model,
                names.layer(names.linear_dt_bias, layer_index),
            ).reshape(1, 1, -1)
        )
        a = _dense_vector(model, names.layer(names.linear_a, layer_index))
        # A is a model constant. Rebuilding Exp/Negative in every decode
        # graph adds one primitive per linear-attention layer and token.
        self.a = mx.contiguous(
            (-mx.exp(a) if config.linear_a_is_log else a).reshape(1, 1, -1)
        )
        mx.eval(self.dt_bias, self.a)
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
        self._speculative_rollback: tuple[mx.array, mx.array, int] | None = None

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
        self._speculative_rollback = None

    def _forward_impl(
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
        gate_input = alpha + self.dt_bias
        gate = mx.maximum(gate_input, 0.0) + mx.log1p(mx.exp(-mx.abs(gate_input)))
        gate = gate * self.a

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
            self.conv_parameters,
            num_key_heads=self.key_heads,
            num_value_heads=self.value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
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

    def forward(
        self,
        x: mx.array,
        positions: mx.array | None,
        *,
        use_cache: bool,
        n_confirmed: int = 0,
        position_offset: int = 0,
    ) -> mx.array:
        del position_offset
        confirmed = int(n_confirmed)
        tokens = int(x.shape[1])
        if confirmed == 0:
            return self._forward_impl(x, positions, use_cache=use_cache)
        if not use_cache or confirmed <= 0 or confirmed >= tokens:
            raise ValueError("Qwen3.5 speculative linear-attention window is invalid")
        if self._speculative_rollback is not None:
            raise RuntimeError("Qwen3.5 speculative linear-attention window is unresolved")
        accepted = self._forward_impl(
            x[:, :confirmed],
            None if positions is None else positions[..., :confirmed],
            use_cache=True,
        )
        assert self.conv_state is not None and self.gdn_state is not None
        self._speculative_rollback = (
            self.conv_state,
            self.gdn_state,
            self._cache_position,
        )
        speculative = self._forward_impl(
            x[:, confirmed:],
            None if positions is None else positions[..., confirmed:],
            use_cache=True,
        )
        return mx.concatenate((accepted, speculative), axis=1)

    def commit_speculative_cache(self) -> None:
        self._speculative_rollback = None

    def rollback_speculative_cache(self) -> None:
        if self._speculative_rollback is None:
            raise RuntimeError("Qwen3.5 linear attention has no speculative window")
        self.conv_state, self.gdn_state, self._cache_position = self._speculative_rollback
        self._speculative_rollback = None

    def __call__(
        self,
        x: mx.array,
        positions: mx.array | None,
        *,
        use_cache: bool,
        n_confirmed: int = 0,
        position_offset: int = 0,
    ) -> mx.array:
        return self.forward(
            x,
            positions,
            use_cache=use_cache,
            n_confirmed=n_confirmed,
            position_offset=position_offset,
        )


class MlxCausalLM:
    """Full-attention MFQ causal language model returning logits."""

    def __init__(
        self,
        tensors: Mapping[str, MfqTensor] | MlxNintModel,
        config: MlxCausalLMConfig,
        names: MlxCausalLMNames | None = None,
        *,
        max_context: int | None = None,
    ) -> None:
        selected_context = (
            config.max_position_embeddings
            if max_context is None or int(max_context) <= 0
            else min(int(max_context), config.max_position_embeddings)
        )
        if selected_context <= 0:
            raise ValueError("causal LM max_context must be positive")
        if selected_context != config.max_position_embeddings:
            config = replace(config, max_position_embeddings=selected_context)
        self.model = tensors if isinstance(tensors, MlxNintModel) else MlxNintModel(tensors)
        self.config = config
        self.max_context = selected_context
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
        self._speculative_checkpoint: tuple[int, int, int] | None = None

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        config: MlxCausalLMConfig | None = None,
        names: MlxCausalLMNames | None = None,
        *,
        mmap: bool = True,
        max_context: int | None = None,
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
                names = MlxCausalLMNames()
            if model.legacy_semantics == "qwen35_gguf":
                config = replace(
                    config,
                    norm_weight_offset=0.0,
                    linear_a_is_log=False,
                )
            return cls(model, config, names, max_context=max_context)
        except BaseException:
            model.close()
            raise

    @property
    def position(self) -> int:
        if not self.layers:
            return 0
        first = self.layers[0]
        if isinstance(first, MlxFullAttentionBlock):
            return 0 if first.cache is None else int(first.cache.pos)
        return int(first.cache_pos)

    def reset_cache(self, batch: int = 1) -> None:
        for layer in self.layers:
            layer.reset_cache(batch)
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
        if value.ndim not in (1, 2, 3) or int(value.shape[-1]) != tokens:
            raise ValueError(
                "causal LM positions must have [T], [3,T], or [3,B,T] shape"
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
            raise ValueError("causal LM embeddings/IDs have incompatible shapes")
        batch, tokens = (int(item) for item in ids.shape)
        start = self.position if use_cache else 0
        if start + tokens > self.max_context:
            raise ValueError("causal LM input exceeds max_context")
        confirmed = int(n_confirmed)
        if confirmed < 0 or confirmed >= tokens or (confirmed > 0 and not use_cache):
            raise ValueError("causal LM speculative confirmation window is invalid")
        if confirmed > 0 and self._speculative_checkpoint is not None:
            raise RuntimeError("causal LM speculative window is unresolved")
        position_array = (
            None if positions is None else self._positions(positions, start, tokens)
        )
        if (
            position_array is not None
            and position_array.ndim == 3
            and int(position_array.shape[1]) != batch
        ):
            raise ValueError("causal LM position batch does not match input batch")
        for layer in self.layers:
            if confirmed > 0 and isinstance(layer, MlxQwen35LinearAttentionBlock):
                hidden = layer(
                    hidden,
                    position_array,
                    use_cache=True,
                    n_confirmed=confirmed,
                    position_offset=start,
                )
            else:
                hidden = layer(
                    hidden,
                    position_array,
                    use_cache=use_cache,
                    position_offset=start,
                )
        if confirmed > 0:
            self._speculative_checkpoint = (start, confirmed, tokens)
        logits = self.output(self.output_norm(hidden))
        if tuple(int(item) for item in logits.shape[:2]) != (batch, tokens):
            raise ValueError("causal LM output leading dimensions are invalid")
        return logits, hidden

    def forward_embeddings(
        self,
        embeddings: mx.array | np.ndarray,
        input_ids: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        use_cache: bool = False,
    ) -> mx.array:
        return self.forward_embeddings_with_hidden(
            embeddings,
            input_ids,
            positions,
            use_cache=use_cache,
        )[0]

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
        return self.forward_embeddings(
            self.embedding(ids),
            ids,
            positions,
            use_cache=use_cache,
        )

    def forward_with_hidden(
        self,
        input_ids: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        use_cache: bool = False,
        n_confirmed: int = 0,
    ) -> tuple[mx.array, mx.array]:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim != 2:
            raise ValueError("causal LM input IDs must have [batch,tokens] shape")
        if ids.dtype not in (mx.int32, mx.uint32):
            ids = ids.astype(mx.int32)
        return self.forward_embeddings_with_hidden(
            self.embedding(ids),
            ids,
            positions,
            use_cache=use_cache,
            n_confirmed=n_confirmed,
        )

    def commit_speculative_cache(self) -> None:
        if self._speculative_checkpoint is None:
            raise RuntimeError("causal LM has no speculative window")
        for layer in self.layers:
            if isinstance(layer, MlxQwen35LinearAttentionBlock):
                layer.commit_speculative_cache()
        self._speculative_checkpoint = None

    def rollback_speculative_cache(self, accepted_drafts: int = 0) -> None:
        if self._speculative_checkpoint is None:
            raise RuntimeError("causal LM has no speculative window")
        start, confirmed, tokens = self._speculative_checkpoint
        if int(accepted_drafts) != 0 or confirmed <= 0 or tokens != confirmed + 1:
            raise ValueError("causal LM currently rolls back one rejected draft")
        keep = start + confirmed
        for layer in self.layers:
            if isinstance(layer, MlxQwen35LinearAttentionBlock):
                layer.rollback_speculative_cache()
            else:
                if layer.cache is None:
                    raise RuntimeError("full-attention cache is not initialized")
                layer.cache.pos = keep
        self._speculative_checkpoint = None

    def __call__(
        self,
        input_ids: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        use_cache: bool = False,
    ) -> mx.array:
        return self.forward(input_ids, positions, use_cache=use_cache)

    def prefill(self, input_ids: mx.array | np.ndarray) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        self.reset_cache(int(ids.shape[0]))
        return self.forward(ids, use_cache=True)

    def decode(self, input_ids: mx.array | np.ndarray) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        if ids.ndim != 2 or int(ids.shape[1]) != 1:
            raise ValueError("causal LM decode accepts one token per batch")
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


class MlxQwen35Mtp:
    """Optional Qwen3.5 depth-one MTP head sharing a loaded backbone store."""

    def __init__(self, causal_lm: MlxCausalLM) -> None:
        self.causal_lm = causal_lm
        self.model = causal_lm.model
        self.config = causal_lm.config
        self.embedding = causal_lm.embedding
        self.output = causal_lm.output
        config = self.config
        if config.mtp_num_hidden_layers <= 0:
            raise ValueError("Qwen3.5 config does not declare MTP layers")
        if config.mtp_use_dedicated_embeddings:
            raise ValueError("Qwen3.5 dedicated MTP embeddings are not supported")

        names = MlxCausalLMNames(
            attn_norm="predictor.block.{i}.attention.norm.weight",
            attn_q="predictor.block.{i}.attention.query.weight",
            attn_k="predictor.block.{i}.attention.key.weight",
            attn_v="predictor.block.{i}.attention.value.weight",
            attn_out="predictor.block.{i}.attention.output.weight",
            attn_q_norm="predictor.block.{i}.attention.query_norm.weight",
            attn_k_norm="predictor.block.{i}.attention.key_norm.weight",
            ffn_norm="predictor.block.{i}.mlp.norm.weight",
            ffn_gate="predictor.block.{i}.mlp.gate.weight",
            ffn_up="predictor.block.{i}.mlp.up.weight",
            ffn_down="predictor.block.{i}.mlp.down.weight",
        )
        first_layer = 0
        hidden_norm = "predictor.hidden_norm.weight"
        embedding_norm = "predictor.embedding_norm.weight"
        fusion = "predictor.fusion.weight"
        output_norm = "predictor.output_norm.weight"

        mtp_config = replace(
            config,
            num_hidden_layers=config.mtp_num_hidden_layers,
            layer_types=("full_attention",) * config.mtp_num_hidden_layers,
        )
        self.hidden_norm = MlxRMSNorm(
            _dense_vector(self.model, hidden_norm),
            config.rms_norm_eps,
            weight_offset=config.norm_weight_offset,
        )
        self.embedding_norm = MlxRMSNorm(
            _dense_vector(self.model, embedding_norm),
            config.rms_norm_eps,
            weight_offset=config.norm_weight_offset,
        )
        self.fusion = self.model.linear(fusion)
        self.layers = tuple(
            MlxFullAttentionBlock(
                self.model,
                mtp_config,
                names,
                first_layer + index,
            )
            for index in range(config.mtp_num_hidden_layers)
        )
        self.output_norm = MlxRMSNorm(
            _dense_vector(self.model, output_norm),
            config.rms_norm_eps,
            weight_offset=config.norm_weight_offset,
        )

    @classmethod
    def load_if_present(
        cls,
        causal_lm: MlxCausalLM,
        _config: MlxCausalLMConfig | None = None,
        *,
        max_context: int | None = None,
    ) -> MlxQwen35Mtp | None:
        del max_context
        tensors = causal_lm.model.tensors
        fusion = "predictor.fusion.weight"
        has_any = any(
            name.startswith("predictor.")
            for name in tensors
        )
        if fusion not in tensors:
            if has_any:
                raise ValueError("Qwen3.5 MFQ contains an incomplete MTP head")
            return None
        return cls(causal_lm)

    @property
    def position(self) -> int:
        if not self.layers or self.layers[0].cache is None:
            return 0
        return int(self.layers[0].cache.pos)

    def reset_cache(self, batch: int = 1) -> None:
        for layer in self.layers:
            layer.reset_cache(int(batch))

    def forward(
        self,
        input_ids: mx.array | np.ndarray,
        previous_hidden_states: mx.array | np.ndarray,
        positions: mx.array | np.ndarray | None = None,
        *,
        inputs_embeds: mx.array | np.ndarray | None = None,
        use_cache: bool = False,
    ) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None]
        if ids.ndim != 2:
            raise ValueError("Qwen3.5 MTP IDs must have [batch,tokens] shape")
        hidden = (
            previous_hidden_states
            if isinstance(previous_hidden_states, mx.array)
            else mx.array(previous_hidden_states)
        )
        batch, tokens = (int(item) for item in ids.shape)
        if tuple(hidden.shape) != (batch, tokens, self.config.hidden_size):
            raise ValueError("Qwen3.5 MTP previous hidden has an incompatible shape")
        embeds = self.embedding(ids.astype(mx.int32)) if inputs_embeds is None else inputs_embeds
        embeds = embeds if isinstance(embeds, mx.array) else mx.array(embeds)
        if tuple(embeds.shape) != tuple(hidden.shape):
            raise ValueError("Qwen3.5 MTP embeddings have an incompatible shape")
        start = self.position if use_cache else 0
        position_array = (
            None
            if positions is None
            else MlxCausalLM._positions(positions, start, tokens)
        )
        fused = self.fusion(
            mx.concatenate(
                (self.embedding_norm(embeds), self.hidden_norm(hidden)),
                axis=-1,
            )
        )
        for layer in self.layers:
            fused = layer(
                fused,
                position_array,
                use_cache=use_cache,
                position_offset=start,
            )
        return self.output_norm(fused)

    def compute_logits(self, hidden: mx.array | np.ndarray) -> mx.array:
        value = hidden if isinstance(hidden, mx.array) else mx.array(hidden)
        if value.ndim != 3 or int(value.shape[-1]) != self.config.hidden_size:
            raise ValueError("Qwen3.5 MTP hidden has an incompatible shape")
        return self.output(value)


__all__ = [
    "MlxCausalLM",
    "MlxCausalLMConfig",
    "MlxCausalLMNames",
    "MlxFullAttentionBlock",
    "MlxQwen35Mtp",
    "MlxQwen35LinearAttentionBlock",
]
