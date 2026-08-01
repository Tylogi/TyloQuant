"""Kimi-K3 decode graph for MFQ/TPQ2 models on Apple silicon."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's MLX runtime requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.formats import io
from mfq.formats.io import MfqTensor
from mfq.formats.moe import NintMoeTensor
from mfq.kernels.metal.kimi_k3 import (
    kimi_attention_residual,
    kimi_gated_rmsnorm,
    kimi_kda_recurrent,
    kimi_route_experts,
    kimi_short_conv3,
    situ_mul,
)
from mfq.kernels.metal.sampling import sample as _sample
from mfq.runtime.mlx_linear import MlxLinearGroup, MlxNintModel, mlx_dense_array
from mfq.runtime.mlx_moe import MlxRoutedSiTUFFN
from mfq.runtime.mlx_ops import MlxRMSNorm


@dataclass(frozen=True)
class MlxKimiK3Config:
    """Normalized Kimi-K3 text graph configuration."""

    n_layers: int
    hidden: int
    routed_hidden: int
    n_experts: int
    top_k: int
    moe_inter: int
    n_shared: int
    inter_dense: int
    first_dense_layers: int
    vocab: int
    rms_eps: float
    routed_scaling: float
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    activation: str
    situ_beta: float
    situ_linear_beta: float | None
    latent_moe_use_norm: bool
    n_heads: int
    head_dim: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_lora_rank: int
    max_position_embeddings: int
    attn_res_block_size: int
    kda_layers: tuple[int, ...] = field(default_factory=tuple)
    full_attn_layers: tuple[int, ...] = field(default_factory=tuple)
    eos_token_id: tuple[int, ...] = field(default_factory=tuple)
    short_conv_kernel_size: int = 4
    gate_lower_bound: float | None = -5.0

    @classmethod
    def from_json(cls, values: Mapping) -> MlxKimiK3Config:
        """Load a normalized TPQ2 manifest configuration."""

        data = dict(values)
        for name in ("kda_layers", "full_attn_layers", "eos_token_id"):
            data[name] = tuple(int(value) for value in data.get(name, ()))
        data.setdefault("short_conv_kernel_size", 4)
        data.setdefault("gate_lower_bound", -5.0)
        return cls(**data)

    @classmethod
    def from_hf_config(cls, outer: Mapping) -> MlxKimiK3Config:
        """Normalize the published Hugging Face Kimi-K3 configuration."""

        text = dict(outer.get("text_config") or outer)
        linear = dict(text.get("linear_attn_config") or {})
        eos = text.get("eos_token_id", outer.get("eos_token_id", ()))
        if isinstance(eos, int):
            eos = (eos,)
        linear_beta = text.get("activation_situ_linear_beta")
        return cls(
            n_layers=int(text["num_hidden_layers"]),
            hidden=int(text["hidden_size"]),
            routed_hidden=int(text.get("routed_expert_hidden_size", text["hidden_size"])),
            n_experts=int(text["num_experts"]),
            top_k=int(text["num_experts_per_token"]),
            moe_inter=int(text["moe_intermediate_size"]),
            n_shared=int(text.get("num_shared_experts", 0)),
            inter_dense=int(text["intermediate_size"]),
            first_dense_layers=int(text.get("first_k_dense_replace", 0)),
            vocab=int(text.get("vocab_size", outer.get("vocab_size", 0))),
            rms_eps=float(text.get("rms_norm_eps", 1.0e-5)),
            routed_scaling=float(text.get("routed_scaling_factor", 1.0)),
            scoring_func=str(text.get("moe_router_activation_func", "sigmoid")),
            norm_topk_prob=bool(text.get("moe_renormalize", True)),
            n_group=int(text.get("num_expert_group", 1)),
            topk_group=int(text.get("topk_group", 1)),
            activation=str(text.get("hidden_act", "situ")),
            situ_beta=float(text.get("activation_situ_beta", 4.0)),
            situ_linear_beta=(None if linear_beta is None else float(linear_beta)),
            latent_moe_use_norm=bool(text.get("latent_moe_use_norm", False)),
            n_heads=int(linear.get("num_heads", text["num_attention_heads"])),
            head_dim=int(linear.get("head_dim", 128)),
            kv_lora_rank=int(text["kv_lora_rank"]),
            qk_nope_head_dim=int(text["qk_nope_head_dim"]),
            qk_rope_head_dim=int(text["qk_rope_head_dim"]),
            v_head_dim=int(text["v_head_dim"]),
            q_lora_rank=int(text["q_lora_rank"]),
            max_position_embeddings=int(text.get("max_position_embeddings", 1_048_576)),
            attn_res_block_size=int(text.get("attn_res_block_size", 12)),
            kda_layers=tuple(int(value) - 1 for value in linear.get("kda_layers", ())),
            full_attn_layers=tuple(int(value) - 1 for value in linear.get("full_attn_layers", ())),
            eos_token_id=tuple(int(value) for value in eos),
            short_conv_kernel_size=int(text.get("short_conv_kernel_size", 4)),
            gate_lower_bound=(
                None
                if text.get("gate_lower_bound", -5.0) is None
                else float(text.get("gate_lower_bound", -5.0))
            ),
        )


@dataclass(frozen=True)
class MlxKimiK3Names:
    """Tensor-name mapping for the TPQ2 Kimi-K3 artifact."""

    token_embd: str = "language_model.model.embed_tokens.weight"
    output_norm: str = "language_model.model.norm.weight"
    output_res_proj: str = "language_model.model.output_attn_res_proj.weight"
    output_res_norm: str = "language_model.model.output_attn_res_norm.weight"
    output: str = "language_model.lm_head.weight"
    input_norm: str = "language_model.model.layers.{i}.input_layernorm.weight"
    post_attention_norm: str = "language_model.model.layers.{i}.post_attention_layernorm.weight"
    attention_res_proj: str = "language_model.model.layers.{i}.self_attention_res_proj.weight"
    attention_res_norm: str = "language_model.model.layers.{i}.self_attention_res_norm.weight"
    mlp_res_proj: str = "language_model.model.layers.{i}.mlp_res_proj.weight"
    mlp_res_norm: str = "language_model.model.layers.{i}.mlp_res_norm.weight"
    attention_prefix: str = "language_model.model.layers.{i}.self_attn"
    dense_mlp_prefix: str = "language_model.model.layers.{i}.mlp"
    moe_prefix: str = "language_model.model.layers.{i}.block_sparse_moe"
    expert_gate_up: str = "layers.{i}.ffn.experts.gate_up.weight"
    expert_down: str = "layers.{i}.ffn.experts.down.weight"

    @staticmethod
    def layer(template: str, index: int) -> str:
        return template.format(i=index)


def _dense_array(model: MlxNintModel, name: str) -> mx.array:
    if name not in model.tensors:
        raise KeyError(f"tensor {name!r} is not present in the Kimi model")
    value = model.tensors[name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Kimi state tensor {name!r} must be dense")
    return mlx_dense_array(value)


def _dense_vector(model: MlxNintModel, name: str) -> mx.array:
    value = _dense_array(model, name)
    if value.ndim != 1:
        raise TypeError(f"Kimi tensor {name!r} must be a vector")
    return mx.contiguous(value.astype(mx.float32))


def _linear_weight(model: MlxNintModel, name: str) -> mx.array:
    layer = model.linear(name)
    weight = getattr(layer, "weight", None)
    if weight is None:
        raise TypeError(f"Kimi projection {name!r} cannot be materialized")
    return mx.contiguous(weight.astype(mx.float32))


class MlxKimiSiTUFFN:
    """Dense or shared Kimi SiTU MLP."""

    def __init__(
        self,
        model: MlxNintModel,
        prefix: str,
        *,
        beta: float,
        linear_beta: float | None,
    ) -> None:
        self.gate_up = MlxLinearGroup(
            (
                model.linear(f"{prefix}.gate_proj.weight"),
                model.linear(f"{prefix}.up_proj.weight"),
            )
        )
        self.down = model.linear(f"{prefix}.down_proj.weight")
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)

    def forward(self, value: mx.array) -> mx.array:
        gate, up = self.gate_up(value)
        return self.down(
            situ_mul(
                gate,
                up,
                beta=self.beta,
                linear_beta=self.linear_beta,
            )
        )

    def __call__(self, value: mx.array) -> mx.array:
        return self.forward(value)


class MlxKimiKDA:
    """One Kimi KDA attention layer with persistent Metal state."""

    def __init__(
        self,
        model: MlxNintModel,
        config: MlxKimiK3Config,
        prefix: str,
    ) -> None:
        self.config = config
        self.input = MlxLinearGroup(
            tuple(
                model.linear(f"{prefix}.{name}.weight")
                for name in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "g_proj",
                    "f_a_proj",
                    "b_proj",
                )
            )
        )
        self.f_b = model.linear(f"{prefix}.f_b_proj.weight")
        self.conv_weights = tuple(
            _dense_array(model, f"{prefix}.{name}_conv1d.weight") for name in ("q", "k", "v")
        )
        self.a_log = _dense_vector(model, f"{prefix}.A_log")
        self.dt_bias = _dense_vector(model, f"{prefix}.dt_bias")
        self.output_norm = _dense_vector(model, f"{prefix}.o_norm.weight")
        self.output = model.linear(f"{prefix}.o_proj.weight")
        self.conv_state: tuple[mx.array, mx.array, mx.array] | None = None
        self.recurrent_state: mx.array | None = None
        self.batch = 0

    def reset_cache(self, batch: int) -> None:
        config = self.config
        channels = config.n_heads * config.head_dim
        history = config.short_conv_kernel_size - 1
        if history < 1:
            raise ValueError("Kimi short convolution must contain at least 2 taps")
        self.conv_state = tuple(
            mx.zeros((int(batch), channels, history), dtype=mx.float16) for _ in range(3)
        )
        self.recurrent_state = mx.zeros(
            (
                int(batch),
                config.n_heads,
                config.head_dim,
                config.head_dim,
            ),
            dtype=mx.float32,
        )
        self.batch = int(batch)

    def forward(self, value: mx.array) -> mx.array:
        config = self.config
        batch = int(value.shape[0])
        if self.conv_state is None or self.batch != batch:
            self.reset_cache(batch)
        assert self.conv_state is not None
        assert self.recurrent_state is not None
        projected = self.input(value)
        channels = config.n_heads * config.head_dim
        query, key, val, output_gate = (item.reshape((batch, channels)) for item in projected[:4])
        low_rank_gate = projected[4]
        beta = projected[5].reshape((batch, config.n_heads))
        recurrent_gate = self.f_b(low_rank_gate).reshape((batch, config.n_heads, config.head_dim))

        next_conv: list[tuple[mx.array, mx.array, mx.array]] = []
        attended: list[mx.array] = []
        next_recurrent: list[mx.array] = []
        for item in range(batch):
            q, k, v, states = kimi_short_conv3(
                query[item],
                key[item],
                val[item],
                tuple(state[item] for state in self.conv_state),
                self.conv_weights,
            )
            next_conv.append(states)
            output, recurrent = kimi_kda_recurrent(
                q.reshape((config.n_heads, config.head_dim)),
                k.reshape((config.n_heads, config.head_dim)),
                v.reshape((config.n_heads, config.head_dim)),
                recurrent_gate[item],
                beta[item],
                self.a_log,
                self.dt_bias,
                self.recurrent_state[item],
                lower_bound=config.gate_lower_bound,
            )
            attended.append(output)
            next_recurrent.append(recurrent)
        self.conv_state = tuple(
            mx.stack([states[stream] for states in next_conv], axis=0) for stream in range(3)
        )
        self.recurrent_state = mx.stack(next_recurrent, axis=0)
        recurrent_output = mx.stack(attended, axis=0)
        normalized = kimi_gated_rmsnorm(
            recurrent_output,
            output_gate.reshape((batch, config.n_heads, config.head_dim)),
            self.output_norm,
            config.rms_eps,
        )
        return self.output(normalized.reshape((batch, channels)))

    def __call__(self, value: mx.array) -> mx.array:
        return self.forward(value)


class MlxKimiMLA:
    """One latent-cache Kimi MLA decode layer."""

    def __init__(
        self,
        model: MlxNintModel,
        config: MlxKimiK3Config,
        prefix: str,
    ) -> None:
        self.config = config
        self.input = MlxLinearGroup(
            (
                model.linear(f"{prefix}.q_a_proj.weight"),
                model.linear(f"{prefix}.kv_a_proj_with_mqa.weight"),
                model.linear(f"{prefix}.g_proj.weight"),
            )
        )
        self.query_norm = MlxRMSNorm(
            _dense_vector(model, f"{prefix}.q_a_layernorm.weight"),
            1.0e-6,
        )
        self.query_b = model.linear(f"{prefix}.q_b_proj.weight")
        self.latent_norm = MlxRMSNorm(
            _dense_vector(model, f"{prefix}.kv_a_layernorm.weight"),
            1.0e-6,
        )
        kv_b = _linear_weight(model, f"{prefix}.kv_b_proj.weight").reshape(
            (
                config.n_heads,
                config.qk_nope_head_dim + config.v_head_dim,
                config.kv_lora_rank,
            )
        )
        self.key_absorb = mx.contiguous(kv_b[:, : config.qk_nope_head_dim])
        self.value_absorb = mx.contiguous(kv_b[:, config.qk_nope_head_dim :])
        self.output = model.linear(f"{prefix}.o_proj.weight")
        self.latent_cache: mx.array | None = None
        self.rope_cache: mx.array | None = None
        self.batch = 0

    def reset_cache(self, batch: int) -> None:
        self.latent_cache = None
        self.rope_cache = None
        self.batch = int(batch)

    def forward(self, value: mx.array) -> mx.array:
        config = self.config
        batch = int(value.shape[0])
        if self.batch != batch:
            self.reset_cache(batch)
        query_source, compressed, output_gate = self.input(value)
        query = self.query_b(self.query_norm(query_source)).reshape(
            (
                batch,
                config.n_heads,
                config.qk_nope_head_dim + config.qk_rope_head_dim,
            )
        )
        query_nope, query_rope = mx.split(
            query,
            [config.qk_nope_head_dim],
            axis=-1,
        )
        latent, key_rope = mx.split(
            compressed,
            [config.kv_lora_rank],
            axis=-1,
        )
        latent = self.latent_norm(latent)
        if self.latent_cache is None:
            self.latent_cache = latent[:, None, :]
            self.rope_cache = key_rope[:, None, :]
        else:
            assert self.rope_cache is not None
            self.latent_cache = mx.concatenate(
                (self.latent_cache, latent[:, None, :]),
                axis=1,
            )
            self.rope_cache = mx.concatenate(
                (self.rope_cache, key_rope[:, None, :]),
                axis=1,
            )
        absorbed_query = mx.einsum(
            "bhn,hnk->bhk",
            query_nope.astype(mx.float32),
            self.key_absorb,
        )
        scores = mx.einsum(
            "bhk,blk->bhl",
            absorbed_query,
            self.latent_cache.astype(mx.float32),
        )
        scores = scores + mx.einsum(
            "bhr,blr->bhl",
            query_rope.astype(mx.float32),
            self.rope_cache.astype(mx.float32),
        )
        scores = scores / math.sqrt(config.qk_nope_head_dim + config.qk_rope_head_dim)
        probabilities = mx.softmax(scores, axis=-1)
        context = mx.einsum(
            "bhl,blk->bhk",
            probabilities,
            self.latent_cache.astype(mx.float32),
        )
        output = mx.einsum(
            "bhk,hvk->bhv",
            context,
            self.value_absorb,
        ).reshape((batch, config.n_heads * config.v_head_dim))
        output = output.astype(value.dtype) * mx.sigmoid(output_gate)
        return self.output(output)

    def __call__(self, value: mx.array) -> mx.array:
        return self.forward(value)


class MlxKimiMoE:
    """Kimi latent routed experts plus the shared SiTU branch."""

    def __init__(
        self,
        model: MlxNintModel,
        config: MlxKimiK3Config,
        names: MlxKimiK3Names,
        layer: int,
        available: mx.array | np.ndarray | None,
    ) -> None:
        prefix = names.layer(names.moe_prefix, layer)
        self.config = config
        self.router = model.linear(f"{prefix}.gate.weight")
        self.correction = _dense_vector(
            model,
            f"{prefix}.gate.e_score_correction_bias",
        )
        self.routed_down = model.linear(f"{prefix}.routed_expert_down_proj.weight")
        self.routed_up = model.linear(f"{prefix}.routed_expert_up_proj.weight")
        self.routed_norm = (
            MlxRMSNorm(
                _dense_vector(model, f"{prefix}.routed_expert_norm.weight"),
                config.rms_eps,
            )
            if config.latent_moe_use_norm
            else None
        )
        gate_up = model.tensors[names.layer(names.expert_gate_up, layer)]
        down = model.tensors[names.layer(names.expert_down, layer)]
        if not isinstance(gate_up, NintMoeTensor) or not isinstance(down, NintMoeTensor):
            raise TypeError("Kimi TPQ2 expert records must use NINTM")
        self.experts = MlxRoutedSiTUFFN(
            gate_up,
            down,
            beta=config.situ_beta,
            linear_beta=config.situ_linear_beta,
        )
        self.shared = MlxKimiSiTUFFN(
            model,
            f"{prefix}.shared_experts",
            beta=config.situ_beta,
            linear_beta=config.situ_linear_beta,
        )
        self.available = (
            mx.ones((config.n_experts,), dtype=mx.bool_)
            if available is None
            else mx.array(available).astype(mx.bool_)
        )

    def forward(self, value: mx.array) -> mx.array:
        config = self.config
        logits = self.router(value)
        weights, ids = kimi_route_experts(
            logits,
            self.correction,
            self.available,
            top_k=config.top_k,
            normalize=config.norm_topk_prob,
            scaling=config.routed_scaling,
            n_group=config.n_group,
            topk_group=config.topk_group,
        )
        latent = self.routed_down(value)
        routed = self.experts(latent, ids, weights)
        if self.routed_norm is not None:
            routed = self.routed_norm(routed)
        return self.routed_up(routed) + self.shared(value)

    def __call__(self, value: mx.array) -> mx.array:
        return self.forward(value)


class _MlxKimiLayer:
    def __init__(
        self,
        model: MlxNintModel,
        config: MlxKimiK3Config,
        names: MlxKimiK3Names,
        layer: int,
        available: mx.array | np.ndarray | None,
    ) -> None:
        self.index = int(layer)
        prefix = names.layer(names.attention_prefix, layer)
        self.input_norm = MlxRMSNorm(
            _dense_vector(model, names.layer(names.input_norm, layer)),
            config.rms_eps,
        )
        self.post_attention_norm = _dense_vector(
            model,
            names.layer(names.post_attention_norm, layer),
        )
        self.attention_res_proj = _dense_vector(
            model,
            names.layer(names.attention_res_proj, layer),
        )
        self.attention_res_norm = _dense_vector(
            model,
            names.layer(names.attention_res_norm, layer),
        )
        self.mlp_res_proj = _dense_vector(
            model,
            names.layer(names.mlp_res_proj, layer),
        )
        self.mlp_res_norm = _dense_vector(
            model,
            names.layer(names.mlp_res_norm, layer),
        )
        self.attention = (
            MlxKimiKDA(model, config, prefix)
            if layer in set(config.kda_layers)
            else MlxKimiMLA(model, config, prefix)
        )
        if layer < config.first_dense_layers:
            mlp_prefix = names.layer(names.dense_mlp_prefix, layer)
            self.mlp = MlxKimiSiTUFFN(
                model,
                mlp_prefix,
                beta=config.situ_beta,
                linear_beta=config.situ_linear_beta,
            )
        else:
            self.mlp = MlxKimiMoE(
                model,
                config,
                names,
                layer,
                available,
            )

    def reset_cache(self, batch: int) -> None:
        self.attention.reset_cache(batch)


class MlxKimiK3:
    """Decode-correct Kimi-K3 graph using packed MFQ/TPQ2 Metal kernels."""

    def __init__(
        self,
        tensors: Mapping[str, MfqTensor] | MlxNintModel,
        config: MlxKimiK3Config,
        names: MlxKimiK3Names | None = None,
        *,
        expert_masks: Mapping[int, mx.array | np.ndarray] | None = None,
    ) -> None:
        if config.activation.lower() != "situ":
            raise ValueError("the Kimi-K3 Metal graph currently requires SiTU")
        if config.n_layers <= 0 or config.attn_res_block_size <= 0:
            raise ValueError("Kimi layer count and residual block size must be positive")
        self.model = tensors if isinstance(tensors, MlxNintModel) else MlxNintModel(tensors)
        self.config = config
        self.names = MlxKimiK3Names() if names is None else names
        self.embedding = self.model.embedding(self.names.token_embd)
        masks = {} if expert_masks is None else expert_masks
        self.layers = tuple(
            _MlxKimiLayer(
                self.model,
                config,
                self.names,
                layer,
                masks.get(layer),
            )
            for layer in range(config.n_layers)
        )
        self.output_norm = _dense_vector(self.model, self.names.output_norm)
        self.output_res_proj = _dense_vector(
            self.model,
            self.names.output_res_proj,
        )
        self.output_res_norm = _dense_vector(
            self.model,
            self.names.output_res_norm,
        )
        self.output = self.model.linear(self.names.output)
        self.position = 0
        self.batch = 0

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        config: MlxKimiK3Config | None = None,
        names: MlxKimiK3Names | None = None,
        *,
        mmap: bool = True,
        expert_masks: Mapping[int, mx.array | np.ndarray] | None = None,
    ) -> MlxKimiK3:
        if config is None:
            header, tensors = io.load_mmap(path) if mmap else io.load(path)
            manifest = header.extra.get(
                "tpq_manifest",
                header.extra.get("cccp_manifest"),
            )
            if not isinstance(manifest, dict):
                close = getattr(tensors, "close", None)
                if close is not None:
                    close()
                raise ValueError(
                    "automatic Kimi config loading requires a native TPQ MFQ"
                )
            values = manifest.get("config")
            if not isinstance(values, dict):
                close = getattr(tensors, "close", None)
                if close is not None:
                    close()
                raise ValueError("native TPQ MFQ has no Kimi config")
            model_family = str(manifest.get("model_family", "")).lower()
            if model_family != "kimi_k3" and not (
                "kda_layers" in values and "routed_hidden" in values
            ):
                close = getattr(tensors, "close", None)
                if close is not None:
                    close()
                raise ValueError("native TPQ MFQ is not a Kimi-K3 artifact")
            config = MlxKimiK3Config.from_json(values)
            if expert_masks is None:
                tiers = manifest.get("tiers_per_layer", {})
                expert_masks = {
                    int(layer): np.fromiter(
                        (item != "d" for item in str(assignments)),
                        dtype=np.bool_,
                    )
                    for layer, assignments in tiers.items()
                    if len(str(assignments)) == config.n_experts
                }
            try:
                return cls(
                    MlxNintModel(tensors),
                    config,
                    names,
                    expert_masks=expert_masks,
                )
            except BaseException:
                close = getattr(tensors, "close", None)
                if close is not None:
                    close()
                raise
        return cls(
            MlxNintModel.from_mfq(path, mmap=mmap),
            config,
            names,
            expert_masks=expert_masks,
        )

    def reset_cache(self, batch: int) -> None:
        self.batch = int(batch)
        self.position = 0
        for layer in self.layers:
            layer.reset_cache(batch)

    def _decode_hidden_step(self, hidden: mx.array) -> mx.array:
        config = self.config
        batch = int(hidden.shape[0])
        block_residual = mx.zeros(
            (batch, 0, config.hidden),
            dtype=hidden.dtype,
        )
        for layer in self.layers:
            prefix_sum: mx.array | None = hidden
            rows = int(block_residual.shape[1])
            if rows:
                attention_input = kimi_attention_residual(
                    prefix_sum,
                    block_residual,
                    layer.attention_res_proj,
                    layer.attention_res_norm,
                    config.rms_eps,
                    post_norm_weight=layer.input_norm.weight,
                )
            else:
                attention_input = layer.input_norm(hidden)
            if layer.index % config.attn_res_block_size == 0:
                block_residual = mx.concatenate(
                    (block_residual, prefix_sum[:, None, :]),
                    axis=1,
                )
                prefix_sum = None
            attention = layer.attention(attention_input)
            prefix_sum = attention if prefix_sum is None else prefix_sum + attention
            mlp_input = kimi_attention_residual(
                prefix_sum,
                block_residual,
                layer.mlp_res_proj,
                layer.mlp_res_norm,
                config.rms_eps,
                post_norm_weight=layer.post_attention_norm,
            )
            hidden = prefix_sum + layer.mlp(mlp_input)
        hidden = kimi_attention_residual(
            hidden,
            block_residual,
            self.output_res_proj,
            self.output_res_norm,
            config.rms_eps,
            post_norm_weight=self.output_norm,
        )
        self.position += 1
        return hidden

    def forward(
        self,
        input_ids: mx.array | np.ndarray,
        *,
        use_cache: bool = False,
    ) -> mx.array:
        ids = input_ids if isinstance(input_ids, mx.array) else mx.array(input_ids)
        if ids.ndim != 2:
            raise ValueError("Kimi input IDs must have [batch,tokens] shape")
        ids = ids.astype(mx.int32)
        batch, tokens = map(int, ids.shape)
        if not use_cache or self.batch != batch:
            self.reset_cache(batch)
        if self.position + tokens > self.config.max_position_embeddings:
            raise ValueError("Kimi decode exceeds max_position_embeddings")
        logits: list[mx.array] = []
        embedded = self.embedding(ids)
        for token in range(tokens):
            hidden = self._decode_hidden_step(embedded[:, token])
            logits.append(self.output(hidden))
        if not logits:
            return mx.zeros((batch, 0, self.config.vocab), dtype=mx.float16)
        return mx.stack(logits, axis=1)

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
        """Autoregressively generate tokens while retaining KDA/MLA caches."""

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
            logits[:, -1],
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
                logits = self.forward(next_id[:, None], use_cache=True)
                next_id = _sample(
                    logits[:, -1],
                    temperature=float(temperature),
                    top_k=int(top_k),
                    top_p=float(top_p),
                )
        return mx.concatenate(pieces, axis=1)

    def close(self) -> None:
        self.model.close()

    def __enter__(self) -> MlxKimiK3:
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
    "MlxKimiK3",
    "MlxKimiK3Config",
    "MlxKimiK3Names",
    "MlxKimiKDA",
    "MlxKimiMLA",
    "MlxKimiMoE",
    "MlxKimiSiTUFFN",
]
