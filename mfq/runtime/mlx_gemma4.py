"""Gemma4 MFQ inference runtime for Apple silicon."""

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
from mfq.kernels.metal.gemma4 import (
    gemma4_attn_residual_pre_norms,
    gemma4_ffn_merge,
)
from mfq.kernels.metal.moe_ops import (
    apply_expert_scale,
    geglu_split,
    moe_topk,
    weighted_reduce,
)
from mfq.kernels.metal.ops import gelu_mul
from mfq.kernels.metal.sampling import sample as _sample
from mfq.runtime.mlx_attention import (
    MlxKVCache,
    MlxSlidingWindowKVCache,
    attention,
)
from mfq.runtime.mlx_linear import MlxLinearGroup, MlxNintModel
from mfq.runtime.mlx_moe import MlxRoutedLinear
from mfq.runtime.mlx_ops import MlxRMSNorm, MlxRoPE


def _bf16_round_scalar(value: float) -> float:
    """Round one FP32 scalar to BF16 with round-to-nearest-even."""

    bits = np.asarray([value], dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    result = (rounded & np.uint32(0xFFFF0000)).view(np.float32)
    return float(result[0])


@dataclass(frozen=True)
class MlxGemma4Config:
    """Normalized Gemma4 text-model configuration."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_global_key_value_heads: int
    head_dim: int
    global_head_dim: int
    max_position_embeddings: int
    sliding_window: int
    layer_types: tuple[str, ...]
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    full_rope_base: float = 1_000_000.0
    sliding_rope_base: float = 10_000.0
    full_partial_rotary_factor: float = 1.0
    rms_norm_eps: float = 1e-6
    attention_k_eq_v: bool = False
    tie_word_embeddings: bool = False
    final_logit_softcapping: float = 0.0
    eos_token_id: tuple[int, ...] = ()

    @property
    def embed_scale(self) -> float:
        return _bf16_round_scalar(math.sqrt(self.hidden_size))

    @classmethod
    def from_hf_config(cls, outer: Mapping) -> MlxGemma4Config:
        """Normalize a Hugging Face Gemma4 configuration."""

        text = dict(outer.get("text_config") or outer)
        full = dict(text.get("full_attention") or {})
        sliding = dict(text.get("sliding_attention") or {})
        layers = int(text["num_hidden_layers"])
        layer_types = tuple(
            str(value)
            for value in text.get(
                "layer_types",
                ("full_attention",) * layers,
            )
        )
        if len(layer_types) != layers:
            raise ValueError("Gemma4 layer_types length must match num_hidden_layers")
        unsupported = set(layer_types) - {
            "full_attention",
            "sliding_attention",
        }
        if unsupported:
            raise ValueError(f"unsupported Gemma4 layer types: {sorted(unsupported)}")
        heads = int(text["num_attention_heads"])
        hidden = int(text["hidden_size"])
        head_dim = int(text.get("head_dim", hidden // heads))
        eos = text.get("eos_token_id", outer.get("eos_token_id", ()))
        if eos is None:
            eos_values: tuple[int, ...] = ()
        elif isinstance(eos, int):
            eos_values = (int(eos),)
        else:
            eos_values = tuple(int(value) for value in eos)
        return cls(
            vocab_size=int(text.get("vocab_size", outer.get("vocab_size", 0))),
            hidden_size=hidden,
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=int(text["num_key_value_heads"]),
            num_global_key_value_heads=int(
                text.get(
                    "num_global_key_value_heads",
                    text["num_key_value_heads"],
                )
            ),
            head_dim=head_dim,
            global_head_dim=int(text.get("global_head_dim", head_dim)),
            max_position_embeddings=int(text["max_position_embeddings"]),
            sliding_window=int(text.get("sliding_window", 0)),
            layer_types=layer_types,
            num_experts=int(text.get("num_experts", text.get("n_routed_experts", 0))),
            num_experts_per_tok=int(
                text.get(
                    "num_experts_per_tok",
                    text.get("top_k_experts", 0),
                )
            ),
            moe_intermediate_size=int(text.get("moe_intermediate_size", 0)),
            full_rope_base=float(
                full.get(
                    "rope_theta",
                    text.get("rope_theta", 1_000_000.0),
                )
            ),
            sliding_rope_base=float(sliding.get("rope_theta", 10_000.0)),
            full_partial_rotary_factor=float(
                full.get(
                    "partial_rotary_factor",
                    text.get("partial_rotary_factor", 1.0),
                )
            ),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            attention_k_eq_v=bool(text.get("attention_k_eq_v", False)),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            final_logit_softcapping=float(text.get("final_logit_softcapping", 0.0)),
            eos_token_id=eos_values,
        )


@dataclass(frozen=True)
class MlxGemma4Names:
    """HF tensor-name mapping used by native Gemma4 MFQ artifacts."""

    token_embedding: str = "model.language_model.embed_tokens.weight"
    output_norm: str = "model.language_model.norm.weight"
    output: str = "lm_head.weight"
    layer_prefix: str = "model.language_model.layers.{i}"
    expert_gate_up: str = "model.language_model.layers.{i}.experts.gate_up_proj"
    expert_down: str = "model.language_model.layers.{i}.experts.down_proj"

    def layer(self, index: int) -> str:
        return self.layer_prefix.format(i=index)

    def expert(self, template: str, index: int) -> str:
        return template.format(i=index)


def _dense_array(model: MlxNintModel, name: str) -> mx.array:
    if name not in model.tensors:
        raise KeyError(f"tensor {name!r} is not present in the Gemma4 model")
    value = model.tensors[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Gemma4 tensor {name!r} must be dense")
    return mx.array(np.ascontiguousarray(value))


def _dense_vector(model: MlxNintModel, name: str) -> mx.array:
    value = _dense_array(model, name)
    if value.ndim != 1:
        raise TypeError(f"Gemma4 tensor {name!r} must be a vector")
    return mx.contiguous(value.astype(mx.float32))


def _nint_moe(model: MlxNintModel, name: str) -> NintMoeTensor:
    if name not in model.tensors:
        raise KeyError(f"tensor {name!r} is not present in the Gemma4 model")
    value = model.tensors[name]
    if not isinstance(value, NintMoeTensor):
        raise TypeError(f"Gemma4 expert tensor {name!r} must use NINTM")
    return value


class MlxGemma4DenseFFN:
    """Dense GeGLU branch with one heterogeneous gate/up dispatch."""

    def __init__(self, model: MlxNintModel, prefix: str) -> None:
        self.gate_up = MlxLinearGroup(
            (
                model.linear(f"{prefix}.gate_proj.weight"),
                model.linear(f"{prefix}.up_proj.weight"),
            )
        )
        self.down = model.linear(f"{prefix}.down_proj.weight")

    def forward(self, value: mx.array) -> mx.array:
        gate, up = self.gate_up(value)
        return self.down(gelu_mul(gate, up))

    def __call__(self, value: mx.array) -> mx.array:
        return self.forward(value)


class MlxGemma4MoE:
    """Gemma4 delayed-softmax GeGLU routed-expert branch."""

    def __init__(
        self,
        model: MlxNintModel,
        config: MlxGemma4Config,
        names: MlxGemma4Names,
        layer: int,
        prefix: str,
    ) -> None:
        gate_up = _nint_moe(
            model,
            names.expert(names.expert_gate_up, layer),
        )
        down = _nint_moe(
            model,
            names.expert(names.expert_down, layer),
        )
        if (
            gate_up.n_experts != config.num_experts
            or down.n_experts != config.num_experts
            or gate_up.neuron_len != config.hidden_size
            or gate_up.out_per_expert != 2 * config.moe_intermediate_size
            or down.neuron_len != config.moe_intermediate_size
            or down.out_per_expert != config.hidden_size
        ):
            raise ValueError(f"Gemma4 expert shapes disagree with config at layer {layer}")
        self.gate_up = MlxRoutedLinear(gate_up)
        self.down = MlxRoutedLinear(down)
        self.router = model.linear(f"{prefix}.router.proj.weight")
        self.expert_scale = _dense_vector(
            model,
            f"{prefix}.router.per_expert_scale",
        )
        if int(self.expert_scale.size) != config.num_experts:
            raise ValueError("Gemma4 expert-scale length disagrees with config")
        self.top_k = config.num_experts_per_tok

    def forward(
        self,
        value: mx.array,
        router_input: mx.array,
    ) -> mx.array:
        logits = self.router(router_input)
        ids, weights = moe_topk(
            logits,
            self.top_k,
            delayed_softmax=True,
        )
        weights = apply_expert_scale(weights, ids, self.expert_scale)
        gate_up = self.gate_up(value, ids)
        hidden = geglu_split(gate_up)
        down = self.down(hidden, ids)
        return weighted_reduce(down, weights)

    def __call__(
        self,
        value: mx.array,
        router_input: mx.array,
    ) -> mx.array:
        return self.forward(value, router_input)


class MlxGemma4Layer:
    """One full- or sliding-attention Gemma4 decoder layer."""

    def __init__(
        self,
        model: MlxNintModel,
        config: MlxGemma4Config,
        names: MlxGemma4Names,
        index: int,
    ) -> None:
        self.config = config
        self.index = int(index)
        self.sliding = config.layer_types[index] == "sliding_attention"
        if self.sliding and config.sliding_window <= 0:
            raise ValueError("Gemma4 sliding_window must be positive")
        self.head_dim = config.head_dim if self.sliding else config.global_head_dim
        self.kv_heads = (
            config.num_key_value_heads if self.sliding else config.num_global_key_value_heads
        )
        self.value_equals_key = not self.sliding and config.attention_k_eq_v
        prefix = names.layer(index)
        attention_prefix = f"{prefix}.self_attn"
        self.input_norm = MlxRMSNorm(
            _dense_vector(model, f"{prefix}.input_layernorm.weight"),
            config.rms_norm_eps,
        )
        self.attention_post_weight = _dense_vector(
            model,
            f"{prefix}.post_attention_layernorm.weight",
        )
        self.q_norm = MlxRMSNorm(
            _dense_vector(model, f"{attention_prefix}.q_norm.weight"),
            config.rms_norm_eps,
        )
        self.k_norm = MlxRMSNorm(
            _dense_vector(model, f"{attention_prefix}.k_norm.weight"),
            config.rms_norm_eps,
        )
        self.v_norm = MlxRMSNorm(
            mx.ones((self.head_dim,), dtype=mx.float32),
            config.rms_norm_eps,
        )
        projection_names = [
            f"{attention_prefix}.q_proj.weight",
            f"{attention_prefix}.k_proj.weight",
        ]
        if not self.value_equals_key:
            projection_names.append(f"{attention_prefix}.v_proj.weight")
        self.qkv = MlxLinearGroup(tuple(model.linear(name) for name in projection_names))
        self.attention_output = model.linear(f"{attention_prefix}.o_proj.weight")
        active_pairs = (
            None
            if self.sliding
            else int(round(config.full_partial_rotary_factor * self.head_dim / 2.0))
        )
        self.rope = MlxRoPE(
            self.head_dim,
            config.max_position_embeddings,
            base=(config.sliding_rope_base if self.sliding else config.full_rope_base),
            frequency_dim=self.head_dim,
            active_pairs=active_pairs,
        )
        self.dense_pre_weight = _dense_vector(
            model,
            f"{prefix}.pre_feedforward_layernorm.weight",
        )
        self.final_post_weight = _dense_vector(
            model,
            f"{prefix}.post_feedforward_layernorm.weight",
        )
        self.dense_post_weight = _dense_vector(
            model,
            f"{prefix}.post_feedforward_layernorm_1.weight",
        )
        self.moe_pre_weight = _dense_vector(
            model,
            f"{prefix}.pre_feedforward_layernorm_2.weight",
        )
        self.moe_post_weight = _dense_vector(
            model,
            f"{prefix}.post_feedforward_layernorm_2.weight",
        )
        self.layer_scale = mx.contiguous(
            _dense_array(model, f"{prefix}.layer_scalar").astype(mx.float16)
        )
        if int(self.layer_scale.size) != 1:
            raise ValueError("Gemma4 layer_scalar must contain one value")
        router_scale = _dense_vector(model, f"{prefix}.router.scale")
        if int(router_scale.size) != config.hidden_size:
            raise ValueError("Gemma4 router scale length disagrees with config")
        self.router_norm_weight = mx.contiguous(router_scale / math.sqrt(config.hidden_size))
        self.dense_ffn = MlxGemma4DenseFFN(model, f"{prefix}.mlp")
        self.moe = MlxGemma4MoE(model, config, names, index, prefix)
        self.cache: MlxKVCache | MlxSlidingWindowKVCache | None = None

    @property
    def cache_position(self) -> int:
        return 0 if self.cache is None else int(self.cache.pos)

    def reset_cache(self, batch: int) -> None:
        if self.sliding:
            self.cache = MlxSlidingWindowKVCache(
                batch,
                self.kv_heads,
                self.config.sliding_window,
                self.head_dim,
            )
        else:
            self.cache = MlxKVCache(
                batch,
                self.kv_heads,
                self.config.max_position_embeddings,
                self.head_dim,
            )

    def _sliding_mask(self, tokens: int) -> mx.array:
        query = mx.arange(tokens, dtype=mx.int32)[:, None]
        key = mx.arange(tokens, dtype=mx.int32)[None, :]
        visible = (key <= query) & (key >= query - self.config.sliding_window + 1)
        return mx.where(
            visible,
            mx.array(0.0, dtype=mx.float32),
            mx.array(-mx.inf, dtype=mx.float32),
        )

    def forward(
        self,
        value: mx.array,
        positions: mx.array,
        *,
        use_cache: bool,
    ) -> mx.array:
        config = self.config
        batch, tokens, hidden = (int(item) for item in value.shape)
        normalized = self.input_norm(value.astype(mx.float32))
        projections = self.qkv(normalized)
        query_full, key_full = projections[:2]
        value_full = key_full if self.value_equals_key else projections[2]
        query = mx.transpose(
            query_full.reshape(
                batch,
                tokens,
                config.num_attention_heads,
                self.head_dim,
            ),
            (0, 2, 1, 3),
        )
        key = mx.transpose(
            key_full.reshape(
                batch,
                tokens,
                self.kv_heads,
                self.head_dim,
            ),
            (0, 2, 1, 3),
        )
        cached_value = mx.transpose(
            value_full.reshape(
                batch,
                tokens,
                self.kv_heads,
                self.head_dim,
            ),
            (0, 2, 1, 3),
        )
        query = self.rope(
            self.q_norm(query.astype(mx.float32)),
            positions,
        ).astype(mx.float16)
        key = self.rope(
            self.k_norm(key.astype(mx.float32)),
            positions,
        ).astype(mx.float16)
        cached_value = self.v_norm(cached_value.astype(mx.float32)).astype(mx.float16)
        if use_cache:
            if tokens != 1:
                raise ValueError(
                    "cached Gemma4 layers accept one token per call; "
                    "use MlxGemma4.prefill for prompts"
                )
            if self.cache is None:
                self.reset_cache(batch)
            assert self.cache is not None
            if isinstance(self.cache, MlxKVCache):
                key_cache, value_cache = self.cache.append(
                    key,
                    cached_value,
                    positions.reshape((-1,)),
                )
            else:
                key_cache, value_cache = self.cache.append(key, cached_value)
            attended = attention(
                query,
                key_cache,
                value_cache,
                causal=False,
                scale=1.0,
            )
        elif self.sliding:
            attended = attention(
                query,
                key,
                cached_value,
                causal=False,
                scale=1.0,
                mask=self._sliding_mask(tokens),
            )
        else:
            attended = attention(
                query,
                key,
                cached_value,
                causal=True,
                scale=1.0,
            )
        attended = mx.transpose(attended, (0, 2, 1, 3)).reshape(
            batch,
            tokens,
            config.num_attention_heads * self.head_dim,
        )
        attention_output = self.attention_output(attended)
        rows = batch * tokens
        (
            residual,
            dense_input,
            router_input,
            moe_input,
        ) = gemma4_attn_residual_pre_norms(
            value.reshape((rows, hidden)),
            attention_output.reshape((rows, hidden)),
            self.attention_post_weight,
            self.dense_pre_weight,
            self.router_norm_weight,
            self.moe_pre_weight,
            config.rms_norm_eps,
        )
        dense_output = self.dense_ffn(dense_input)
        moe_output = self.moe(moe_input, router_input)
        output = gemma4_ffn_merge(
            dense_output,
            moe_output,
            residual,
            self.dense_post_weight,
            self.moe_post_weight,
            self.final_post_weight,
            self.layer_scale,
            config.rms_norm_eps,
        )
        return output.reshape((batch, tokens, hidden))

    def __call__(
        self,
        value: mx.array,
        positions: mx.array,
        *,
        use_cache: bool,
    ) -> mx.array:
        return self.forward(value, positions, use_cache=use_cache)


class MlxGemma4:
    """Complete native-MFQ Gemma4 model with Metal KV caches and generation."""

    def __init__(
        self,
        tensors: Mapping[str, MfqTensor] | MlxNintModel,
        config: MlxGemma4Config,
        names: MlxGemma4Names | None = None,
    ) -> None:
        self.model = tensors if isinstance(tensors, MlxNintModel) else MlxNintModel(tensors)
        self.config = config
        self.names = MlxGemma4Names() if names is None else names
        self.embedding = self.model.embedding(self.names.token_embedding)
        self.layers = tuple(
            MlxGemma4Layer(
                self.model,
                config,
                self.names,
                index,
            )
            for index in range(config.num_hidden_layers)
        )
        self.output_norm = MlxRMSNorm(
            _dense_vector(self.model, self.names.output_norm),
            config.rms_norm_eps,
        )
        self.output = self.model.linear(
            self.names.token_embedding
            if config.tie_word_embeddings or self.names.output not in self.model.tensors
            else self.names.output
        )

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        config: MlxGemma4Config | Mapping | None = None,
        names: MlxGemma4Names | None = None,
        *,
        mmap: bool = True,
    ) -> MlxGemma4:
        """Load a self-contained, optionally sharded Gemma4 MFQ artifact."""

        model = MlxNintModel.from_mfq(path, mmap=mmap)
        try:
            selected = config
            if selected is None:
                if MODEL_CONFIG_ASSET not in model.tensors:
                    raise ValueError(
                        "MFQ has no embedded model config; pass MlxGemma4Config explicitly"
                    )
                payload = model.tensors[MODEL_CONFIG_ASSET]
                if not isinstance(payload, bytes):
                    raise TypeError("embedded MFQ model config must be a BLOB record")
                selected = json.loads(payload)
            normalized = (
                selected
                if isinstance(selected, MlxGemma4Config)
                else MlxGemma4Config.from_hf_config(selected)
            )
            return cls(model, normalized, names)
        except BaseException:
            model.close()
            raise

    @property
    def cache_position(self) -> int:
        return 0 if not self.layers else self.layers[0].cache_position

    def reset_cache(self, batch: int) -> None:
        for layer in self.layers:
            layer.reset_cache(batch)

    def _forward_chunk(
        self,
        ids: mx.array,
        positions: mx.array,
        *,
        use_cache: bool,
    ) -> mx.array:
        hidden = (self.embedding(ids) * self.config.embed_scale).astype(mx.float16)
        for layer in self.layers:
            hidden = layer(hidden, positions, use_cache=use_cache)
        logits = self.output(self.output_norm(hidden.astype(mx.float32)))
        cap = self.config.final_logit_softcapping
        if cap > 0.0:
            logits = mx.tanh(logits / cap) * cap
        return logits

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
            raise ValueError("Gemma4 input IDs must have [batch,tokens] shape")
        ids = ids.astype(mx.int32)
        tokens = int(ids.shape[1])
        if positions is None:
            start = self.cache_position if use_cache else 0
            position_array = mx.arange(
                start,
                start + tokens,
                dtype=mx.int32,
            )
        else:
            position_array = (
                positions if isinstance(positions, mx.array) else mx.array(positions)
            ).astype(mx.int32)
            position_array = position_array.reshape((-1,))
            if int(position_array.size) != tokens:
                raise ValueError("Gemma4 positions must contain one index per token")
        if use_cache and tokens > 1:
            return mx.concatenate(
                tuple(
                    self._forward_chunk(
                        ids[:, token : token + 1],
                        position_array[token : token + 1],
                        use_cache=True,
                    )
                    for token in range(tokens)
                ),
                axis=1,
            )
        return self._forward_chunk(
            ids,
            position_array,
            use_cache=use_cache,
        )

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

    def prefill(
        self,
        input_ids: mx.array | np.ndarray,
    ) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2 or int(ids.shape[1]) == 0:
            raise ValueError("Gemma4 prefill IDs must have non-empty [batch,tokens] shape")
        self.reset_cache(int(ids.shape[0]))
        return self.forward(ids, use_cache=True)

    def decode(self, input_ids: mx.array | np.ndarray) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if int(ids.size) != (int(ids.shape[0]) if ids.ndim == 2 else 1):
            raise ValueError("Gemma4 decode accepts one token per batch")
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
            raise ValueError("Gemma4 generation IDs must have [batch,tokens] shape")
        ids = ids.astype(mx.int32)
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

    def __enter__(self) -> MlxGemma4:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "MlxGemma4",
    "MlxGemma4Config",
    "MlxGemma4DenseFFN",
    "MlxGemma4Layer",
    "MlxGemma4MoE",
    "MlxGemma4Names",
]
