"""CUDA causal LM runtime assembled from MFQ kernels."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from mfq.formats import io
from mfq.formats.io import MfqTensor
from mfq.kernels.cuda.acc import acc
from mfq.kernels.cuda.attention import attention
from mfq.kernels.cuda.gated_delta_net import gated_delta_net
from mfq.kernels.cuda.kv_cache import KVCache
from mfq.kernels.cuda.norm import l2_norm, rms_norm
from mfq.kernels.cuda.rope import rope
from mfq.kernels.cuda.sampling import sample
from mfq.kernels.cuda.ssm_conv import ssm_conv_silu
from mfq.runtime.torch_linear import (
    QuantizedTensor,
    TorchLinearGroup,
    TorchNintEmbedding,
    TorchNintLinear,
    TorchNintLinearGroup,
    TorchNvqEmbedding,
    TorchNvqLinear,
    TorchSwiGLUFFN,
    is_quantized_tensor,
)
from mfq.quantize.nint_quant import NintTensor

TensorMapping = Mapping[str, MfqTensor]


def _profile_scope(name: str):
    if os.environ.get("MFQ_PROFILE_SCOPES") == "1":
        return torch.profiler.record_function(name)
    return nullcontext()


@dataclass(frozen=True)
class TorchNintCausalLMConfig:
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
    tie_word_embeddings: bool = False
    layer_types: tuple[str, ...] | None = None
    qwen35_attn_q_gate: bool = False
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_a_is_log: bool = True
    norm_weight_offset: float = 0.0

    @property
    def head_dim(self) -> int:
        if self.attention_head_dim is not None:
            return self.attention_head_dim
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must divide num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def attention_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def effective_rotary_dim(self) -> int:
        if self.rotary_dim is not None:
            return self.rotary_dim
        return self.head_dim

    @classmethod
    def from_qwen35_hf_config(cls, cfg: dict) -> "TorchNintCausalLMConfig":
        text = cfg.get("text_config", cfg)
        rope_params = text.get("rope_parameters", {})
        return cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            attention_head_dim=int(text.get("head_dim", 0)) or None,
            rope_base=float(rope_params.get("rope_theta", 1_000_000.0)),
            rotary_dim=int(round(float(text.get("partial_rotary_factor", rope_params.get("partial_rotary_factor", 1.0))) * int(text.get("head_dim", text["hidden_size"] // text["num_attention_heads"])))),
            rope_sections=tuple(int(v) for v in rope_params.get("mrope_section", ())) or None,
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            layer_types=tuple(text.get("layer_types", ("full_attention",) * int(text["num_hidden_layers"]))),
            qwen35_attn_q_gate=bool(text.get("attn_output_gate", False)),
            linear_conv_kernel_dim=int(text.get("linear_conv_kernel_dim", 4)),
            linear_key_head_dim=int(text.get("linear_key_head_dim", 128)),
            linear_value_head_dim=int(text.get("linear_value_head_dim", 128)),
            linear_num_key_heads=int(text.get("linear_num_key_heads", 0)),
            linear_num_value_heads=int(text.get("linear_num_value_heads", 0)),
            linear_a_is_log=True,
            norm_weight_offset=1.0,
        )


@dataclass(frozen=True)
class TorchNintCausalLMNames:
    token_embd: str = "token_embd.weight"
    attn_norm: str = "blk.{i}.attn_norm.weight"
    attn_q: str = "blk.{i}.attn_q.weight"
    attn_k: str = "blk.{i}.attn_k.weight"
    attn_v: str = "blk.{i}.attn_v.weight"
    attn_out: str = "blk.{i}.attn_output.weight"
    ffn_norm: str = "blk.{i}.ffn_norm.weight"
    ffn_gate: str = "blk.{i}.ffn_gate.weight"
    ffn_up: str = "blk.{i}.ffn_up.weight"
    ffn_down: str = "blk.{i}.ffn_down.weight"
    output_norm: str = "output_norm.weight"
    output: str = "output.weight"
    attn_q_norm: str = "blk.{i}.attn_q_norm.weight"
    attn_k_norm: str = "blk.{i}.attn_k_norm.weight"
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

    def layer(self, template: str, i: int) -> str:
        return template.format(i=i)

    @classmethod
    def qwen35_gguf(cls) -> "TorchNintCausalLMNames":
        return cls(
            ffn_norm="blk.{i}.post_attention_norm.weight",
            linear_qkv="blk.{i}.attn_qkv.weight",
            linear_z="blk.{i}.attn_gate.weight",
        )

    @classmethod
    def qwen35_hf(cls) -> "TorchNintCausalLMNames":
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
            linear_qkv="model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight",
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


class TorchFullAttentionBlock:
    def __init__(
        self,
        tensors: TensorMapping,
        config: TorchNintCausalLMConfig,
        names: TorchNintCausalLMNames,
        layer_idx: int,
        device: str | torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.attn_norm = _dense(tensors, names.layer(names.attn_norm, layer_idx), device)
        self.ffn_norm = _dense(tensors, names.layer(names.ffn_norm, layer_idx), device)
        self.q_norm = _dense_optional(tensors, names.layer(names.attn_q_norm, layer_idx), device)
        self.k_norm = _dense_optional(tensors, names.layer(names.attn_k_norm, layer_idx), device)
        self.qkv_proj = _linear_group(
            tensors,
            (
                names.layer(names.attn_q, layer_idx),
                names.layer(names.attn_k, layer_idx),
                names.layer(names.attn_v, layer_idx),
            ),
            device,
        )
        self.o_proj = _linear(tensors, names.layer(names.attn_out, layer_idx), device)
        self.ffn = TorchSwiGLUFFN.from_tensors(
            _require_quantized(tensors, names.layer(names.ffn_gate, layer_idx)),
            _require_quantized(tensors, names.layer(names.ffn_up, layer_idx)),
            _require_quantized(tensors, names.layer(names.ffn_down, layer_idx)),
            device,
        )
        self.cache: KVCache | None = None

    def reset_cache(self, batch: int) -> None:
        c = self.config
        self.cache = KVCache(batch, c.num_key_value_heads, c.max_position_embeddings, c.head_dim, device=self.device)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, use_cache: bool) -> torch.Tensor:
        c = self.config
        B, T, H = x.shape
        residual = x
        with _profile_scope("full_attn/attn_norm"):
            xn = _qwen_rms_norm(x.reshape(B * T, H).to(torch.float32), self.attn_norm, c).reshape(B, T, H)
        with _profile_scope("full_attn/qkv_proj"):
            q_full, k_full, v_full = self.qkv_proj(xn)
        if c.qwen35_attn_q_gate:
            q_pair = q_full.reshape(B, T, c.num_attention_heads, c.head_dim * 2)
            q_raw, q_gate = q_pair.chunk(2, dim=-1)
        else:
            q_raw, q_gate = q_full, None
            q_raw = q_raw.reshape(B, T, c.num_attention_heads, c.head_dim)
        q = q_raw.transpose(1, 2).contiguous()
        k = k_full.reshape(B, T, c.num_key_value_heads, c.head_dim).transpose(1, 2).contiguous()
        v = v_full.reshape(B, T, c.num_key_value_heads, c.head_dim).transpose(1, 2).contiguous()
        with _profile_scope("full_attn/qk_norm_rope"):
            if self.q_norm is not None:
                q = _qwen_rms_norm(q.reshape(-1, c.head_dim).to(torch.float32), self.q_norm, c).reshape_as(q)
            if self.k_norm is not None:
                k = _qwen_rms_norm(k.reshape(-1, c.head_dim).to(torch.float32), self.k_norm, c).reshape_as(k)
            q = rope(q, positions, c.rope_base, c.effective_rotary_dim, c.rope_sections, c.max_position_embeddings)
            k = rope(k, positions, c.rope_base, c.effective_rotary_dim, c.rope_sections, c.max_position_embeddings)
        cache_positions = positions[0] if positions.dim() == 2 else positions
        if use_cache:
            if self.cache is None:
                self.reset_cache(B)
            assert self.cache is not None
            kc, vc = self.cache.append(k, v, cache_positions)
        else:
            kc, vc = k, v
        with _profile_scope("full_attn/attention"):
            a = attention(q, kc, vc, causal=True)
        with _profile_scope("full_attn/o_proj"):
            if q_gate is not None:
                a = a.transpose(1, 2).contiguous().reshape(B, T, c.attention_size)
                gate = q_gate.contiguous().reshape(B, T, c.attention_size)
                o = self.o_proj.forward_input_mul(a, gate, "sigmoid")
            else:
                a = a.transpose(1, 2).contiguous().reshape(B, T, c.attention_size)
                o = self.o_proj(a)
        with _profile_scope("full_attn/residual"):
            x = acc(residual.reshape(-1, H), o.reshape(-1, H))
        x = x.reshape(B, T, H)

        residual = x
        with _profile_scope("full_attn/ffn_norm"):
            xn = _qwen_rms_norm(x.reshape(B * T, H).to(torch.float32), self.ffn_norm, c).reshape(B, T, H)
        with _profile_scope("full_attn/ffn"):
            f = self.ffn(xn.reshape(B * T, H)).reshape(B, T, H)
        with _profile_scope("full_attn/ffn_residual"):
            x = acc(residual.reshape(-1, H), f.reshape(-1, H))
        return x.reshape(B, T, H)


class TorchQwen35LinearAttentionBlock:
    def __init__(
        self,
        tensors: TensorMapping,
        config: TorchNintCausalLMConfig,
        names: TorchNintCausalLMNames,
        layer_idx: int,
        device: str | torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.layer_idx = layer_idx
        self.gguf_layout = names.linear_qkv == "blk.{i}.attn_qkv.weight"
        self.attn_norm = _dense(tensors, names.layer(names.attn_norm, layer_idx), device)
        self.ffn_norm = _dense(tensors, names.layer(names.ffn_norm, layer_idx), device)
        linear_qk_name = names.layer(names.linear_qk, layer_idx) if names.linear_qk is not None else None
        linear_v_name = names.layer(names.linear_v, layer_idx) if names.linear_v is not None else None
        self.split_in_proj = linear_qk_name in tensors and linear_v_name in tensors
        if self.split_in_proj:
            assert linear_qk_name is not None and linear_v_name is not None
            self.linear_qk = _linear(tensors, linear_qk_name, device)
            self.linear_v = _linear(tensors, linear_v_name, device)
            linear_z_name = names.layer(names.linear_z, layer_idx)
            if is_quantized_tensor(tensors[linear_z_name]):
                self.linear_z = _linear(tensors, linear_z_name, device)
                self.ab_proj = _linear_group(
                    tensors,
                    (
                        names.layer(names.linear_alpha, layer_idx),
                        names.layer(names.linear_beta, layer_idx),
                    ),
                    device,
                )
                self.zab_proj = None
            else:
                self.linear_z = None
                self.ab_proj = None
                self.zab_proj = _linear_group(
                    tensors,
                    (
                        linear_z_name,
                        names.layer(names.linear_alpha, layer_idx),
                        names.layer(names.linear_beta, layer_idx),
                    ),
                    device,
                )
            self.in_proj = None
        else:
            self.linear_qk = None
            self.linear_v = None
            self.linear_z = None
            self.ab_proj = None
            self.zab_proj = None
            self.in_proj = _linear_group(
                tensors,
                (
                    names.layer(names.linear_qkv, layer_idx),
                    names.layer(names.linear_z, layer_idx),
                    names.layer(names.linear_alpha, layer_idx),
                    names.layer(names.linear_beta, layer_idx),
                ),
                device,
            )
        self.conv_weight = _dense(tensors, names.layer(names.linear_conv, layer_idx), device)
        self.conv_bias = (
            _dense_optional(tensors, names.layer(names.linear_conv_bias, layer_idx), device)
            if names.linear_conv_bias is not None else None
        )
        self.dt_bias = _dense(tensors, names.layer(names.linear_dt_bias, layer_idx), device)
        self.a = _dense(tensors, names.layer(names.linear_a, layer_idx), device)
        self.linear_norm = _dense(tensors, names.layer(names.linear_norm, layer_idx), device)
        self.out_proj = _linear(tensors, names.layer(names.linear_out, layer_idx), device)
        self.ffn = TorchSwiGLUFFN.from_tensors(
            _require_quantized(tensors, names.layer(names.ffn_gate, layer_idx)),
            _require_quantized(tensors, names.layer(names.ffn_up, layer_idx)),
            _require_quantized(tensors, names.layer(names.ffn_down, layer_idx)),
            device,
        )
        self.conv_state: torch.Tensor | None = None
        self.gdn_state: torch.Tensor | None = None

    def reset_cache(self, batch: int) -> None:
        c = self.config
        conv_dim = 2 * self.linear_k_size + self.linear_v_size
        self.conv_state = torch.zeros(
            batch, c.linear_conv_kernel_dim - 1, conv_dim, device=self.device, dtype=torch.float32
        )
        self.gdn_state = torch.zeros(
            batch, self.linear_num_value_heads, self.linear_value_head_dim, self.linear_value_head_dim,
            device=self.device, dtype=torch.float32,
        )

    @property
    def linear_num_key_heads(self) -> int:
        return self.config.linear_num_key_heads or self.config.num_key_value_heads

    @property
    def linear_num_value_heads(self) -> int:
        return self.config.linear_num_value_heads or self.config.num_attention_heads

    @property
    def linear_key_head_dim(self) -> int:
        return self.config.linear_key_head_dim

    @property
    def linear_value_head_dim(self) -> int:
        return self.config.linear_value_head_dim

    @property
    def linear_k_size(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_v_size(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    def forward(self, x: torch.Tensor, positions: torch.Tensor, use_cache: bool) -> torch.Tensor:
        del positions
        c = self.config
        B, T, H = x.shape
        residual = x
        with _profile_scope("linear_attn/attn_norm"):
            xn = _qwen_rms_norm(x.reshape(B * T, H).to(torch.float32), self.attn_norm, c).reshape(B, T, H)
        with _profile_scope("linear_attn/in_proj"):
            if self.split_in_proj:
                assert self.linear_qk is not None and self.linear_v is not None
                qk = self.linear_qk(xn)
                v_part = self.linear_v(xn)
                qkv = torch.cat((qk.to(torch.float32), v_part.to(torch.float32)), dim=-1)
                if self.linear_z is not None:
                    assert self.ab_proj is not None
                    z = self.linear_z(xn)
                    alpha_raw, beta_raw = self.ab_proj(xn)
                else:
                    assert self.zab_proj is not None
                    z, alpha_raw, beta_raw = self.zab_proj(xn)
            else:
                assert self.in_proj is not None
                qkv, z, alpha_raw, beta_raw = self.in_proj(xn)
        with _profile_scope("linear_attn/gates"):
            qkv = qkv.to(torch.float32)
            beta = torch.sigmoid(beta_raw.to(torch.float32)).reshape(B, T, self.linear_num_value_heads)
            alpha = alpha_raw.to(torch.float32).reshape(B, T, self.linear_num_value_heads)
            gate = F.softplus(alpha + self.dt_bias.view(1, 1, -1))
            a = -torch.exp(self.a) if c.linear_a_is_log else self.a
            gate = gate * a.view(1, 1, -1)

        if use_cache:
            if self.conv_state is None or self.gdn_state is None:
                self.reset_cache(B)
            assert self.conv_state is not None
            conv_in = torch.cat([self.conv_state, qkv], dim=1)
            self.conv_state = conv_in[:, -int(c.linear_conv_kernel_dim) + 1 :, :].contiguous()
            state = self.gdn_state
        else:
            pad = torch.zeros(B, int(c.linear_conv_kernel_dim) - 1, qkv.shape[-1], device=x.device, dtype=torch.float32)
            conv_in = torch.cat([pad, qkv], dim=1)
            state = None

        with _profile_scope("linear_attn/conv_silu"):
            conv = ssm_conv_silu(conv_in, self.conv_weight, T, self.conv_bias)
        q_end = self.linear_k_size
        k_end = q_end + self.linear_k_size
        v_end = k_end + self.linear_v_size
        q = conv[:, :, :q_end].reshape(B, T, self.linear_num_key_heads, self.linear_key_head_dim).transpose(1, 2)
        k = conv[:, :, q_end:k_end].reshape(B, T, self.linear_num_key_heads, self.linear_key_head_dim).transpose(1, 2)
        v = conv[:, :, k_end:v_end].reshape(B, T, self.linear_num_value_heads, self.linear_value_head_dim).transpose(1, 2)
        with _profile_scope("linear_attn/qk_l2_norm"):
            q = l2_norm(q.contiguous().reshape(-1, self.linear_key_head_dim), c.rms_norm_eps).reshape_as(q)
            k = l2_norm(k.contiguous().reshape(-1, self.linear_key_head_dim), c.rms_norm_eps).reshape_as(k)
        if self.linear_num_key_heads != self.linear_num_value_heads:
            rep = self.linear_num_value_heads // self.linear_num_key_heads
            if self.gguf_layout:
                q = q.repeat(1, rep, 1, 1)
                k = k.repeat(1, rep, 1, 1)
            else:
                q = q.repeat_interleave(rep, dim=1)
                k = k.repeat_interleave(rep, dim=1)
        g = gate.transpose(1, 2).contiguous()
        b = beta.transpose(1, 2).contiguous()
        with _profile_scope("linear_attn/gdn"):
            y, new_state = gated_delta_net(q.contiguous(), k.contiguous(), v.contiguous(), g, b, state)
        if use_cache:
            self.gdn_state = new_state
        z = z.reshape(B, T, self.linear_num_value_heads, self.linear_value_head_dim).transpose(1, 2).contiguous()
        with _profile_scope("linear_attn/gated_norm"):
            y_norm = rms_norm(
                y.reshape(-1, self.linear_value_head_dim).to(torch.float32), self.linear_norm, c.rms_norm_eps
            ).reshape_as(y)
            y = y_norm.transpose(1, 2).contiguous().reshape(B, T, self.linear_v_size)
            z = z.transpose(1, 2).contiguous().reshape(B, T, self.linear_v_size)
        with _profile_scope("linear_attn/out_proj"):
            o = self.out_proj.forward_input_mul(y, z, "silu")
        with _profile_scope("linear_attn/residual"):
            x = acc(residual.reshape(-1, H), o.reshape(-1, H))
        x = x.reshape(B, T, H)

        residual = x
        with _profile_scope("linear_attn/ffn_norm"):
            xn = _qwen_rms_norm(x.reshape(B * T, H).to(torch.float32), self.ffn_norm, c).reshape(B, T, H)
        with _profile_scope("linear_attn/ffn"):
            f = self.ffn(xn.reshape(B * T, H)).reshape(B, T, H)
        with _profile_scope("linear_attn/ffn_residual"):
            x = acc(residual.reshape(-1, H), f.reshape(-1, H))
        return x.reshape(B, T, H)


class TorchNintCausalLM:
    def __init__(
        self,
        tensors: TensorMapping,
        config: TorchNintCausalLMConfig,
        names: TorchNintCausalLMNames | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.tensors = tensors
        if names is None:
            names = (
                TorchNintCausalLMNames.qwen35_gguf()
                if "blk.0.post_attention_norm.weight" in tensors
                else TorchNintCausalLMNames()
            )
        if names.linear_qkv == "blk.{i}.attn_qkv.weight":
            config = replace(config, linear_a_is_log=False, norm_weight_offset=0.0)
        self.config = config
        self.names = names
        self.device = device
        embed_tensor = _require_quantized(tensors, self.names.token_embd)
        self.embed = (
            TorchNintEmbedding(embed_tensor, device)
            if isinstance(embed_tensor, NintTensor)
            else TorchNvqEmbedding(embed_tensor, device)
        )
        layer_types = config.layer_types or ("full_attention",) * config.num_hidden_layers
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError("layer_types length must match num_hidden_layers")
        self.blocks = [
            _make_block(tensors, config, self.names, i, layer_types[i], device)
            for i in range(config.num_hidden_layers)
        ]
        self.output_norm = _dense(tensors, self.names.output_norm, device)
        if config.tie_word_embeddings:
            self.lm_head = _linear(tensors, self.names.token_embd, device)
        else:
            self.lm_head = _linear(tensors, self.names.output, device)
        self.cache_pos = 0

    @classmethod
    def from_mfq(
        cls,
        path: str | Path,
        config: TorchNintCausalLMConfig,
        names: TorchNintCausalLMNames | None = None,
        device: str | torch.device = "cuda",
        mmap: bool = False,
    ) -> "TorchNintCausalLM":
        _header, tensors = io.load_mmap(path) if mmap else io.load(path)
        return cls(tensors, config, names, device)

    def reset_cache(self, batch: int) -> None:
        for block in self.blocks:
            block.reset_cache(batch)
        self.cache_pos = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        ids = torch.as_tensor(input_ids, device=self.device, dtype=torch.int64)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, T = ids.shape
        if positions is None:
            cache_pos = self.cache_pos if use_cache else 0
            positions = torch.arange(cache_pos, cache_pos + T, device=self.device, dtype=torch.int64)
        else:
            positions = torch.as_tensor(positions, device=self.device, dtype=torch.int64)
        if use_cache and self.cache_pos == 0:
            self.reset_cache(B)
        with _profile_scope("model/embed"):
            x = self.embed(ids)
        for i, block in enumerate(self.blocks):
            with _profile_scope(f"model/block_{i:02d}"):
                x = block.forward(x, positions, use_cache)
        if use_cache:
            self.cache_pos += T
        H = self.config.hidden_size
        with _profile_scope("model/output_norm"):
            x = _qwen_rms_norm(x.reshape(B * T, H).to(torch.float32), self.output_norm, self.config)
        with _profile_scope("model/lm_head"):
            logits = self.lm_head(x.reshape(B, T, H))
        return logits

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(input_ids, **kwargs)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        ids = torch.as_tensor(input_ids, device=self.device, dtype=torch.int64)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        self.reset_cache(int(ids.shape[0]))
        logits = self.forward(ids, use_cache=True)
        pieces = [ids]
        next_id = sample(logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p)
        eos = None
        if eos_token_id is not None:
            eos = torch.as_tensor(
                (eos_token_id,) if isinstance(eos_token_id, int) else eos_token_id,
                device=self.device,
                dtype=torch.int64,
            )
        for i in range(max_new_tokens):
            pieces.append(next_id[:, None])
            if eos is not None and torch.isin(next_id, eos).all():
                break
            if i + 1 < max_new_tokens:
                logits = self.forward(next_id[:, None], use_cache=True)
                next_id = sample(logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p)
        return torch.cat(pieces, dim=1)


def _require_quantized(tensors: TensorMapping, name: str) -> QuantizedTensor:
    if name not in tensors:
        raise KeyError(f"missing tensor {name!r}")
    tensor = tensors[name]
    if not is_quantized_tensor(tensor):
        raise TypeError(f"tensor {name!r} must be NINT/NVQ")
    return tensor


def _linear(
    tensors: TensorMapping,
    name: str,
    device: str | torch.device,
) -> TorchNintLinear | TorchNvqLinear:
    tensor = _require_quantized(tensors, name)
    return TorchNintLinear(tensor, device) if isinstance(tensor, NintTensor) else TorchNvqLinear(tensor, device)


def _linear_group(
    tensors: TensorMapping,
    names: tuple[str, ...],
    device: str | torch.device,
) -> TorchNintLinearGroup | TorchLinearGroup:
    vals = []
    for name in names:
        if name not in tensors:
            raise KeyError(f"missing tensor {name!r}")
        vals.append(tensors[name])
    if all(isinstance(t, NintTensor) for t in vals):
        try:
            return TorchNintLinearGroup(tuple(vals), device)  # type: ignore[arg-type]
        except ValueError:
            pass
    return TorchLinearGroup(tuple(vals), device)


def _dense(tensors: TensorMapping, name: str, device: str | torch.device) -> torch.Tensor:
    if name not in tensors:
        raise KeyError(f"missing tensor {name!r}")
    tensor = tensors[name]
    if is_quantized_tensor(tensor):
        raise TypeError(f"tensor {name!r} must be dense")
    return torch.as_tensor(tensor, device=device, dtype=torch.float32).contiguous()


def _dense_optional(tensors: TensorMapping, name: str, device: str | torch.device) -> torch.Tensor | None:
    if name not in tensors:
        return None
    return _dense(tensors, name, device)


def _qwen_rms_norm(x: torch.Tensor, weight: torch.Tensor, config: TorchNintCausalLMConfig) -> torch.Tensor:
    if config.norm_weight_offset:
        weight = weight + float(config.norm_weight_offset)
    return rms_norm(x, weight, config.rms_norm_eps)


def _make_block(
    tensors: TensorMapping,
    config: TorchNintCausalLMConfig,
    names: TorchNintCausalLMNames,
    layer_idx: int,
    layer_type: str,
    device: str | torch.device,
):
    if layer_type == "linear_attention":
        return TorchQwen35LinearAttentionBlock(tensors, config, names, layer_idx, device)
    if layer_type == "full_attention":
        return TorchFullAttentionBlock(tensors, config, names, layer_idx, device)
    raise ValueError(f"unsupported layer type: {layer_type!r}")


def _causal_depthwise_conv(conv_input: torch.Tensor, conv_weight: torch.Tensor, n_tokens: int) -> torch.Tensor:
    if conv_weight.dim() == 3:
        conv_weight = conv_weight.squeeze(1).T.contiguous()
    K = int(conv_weight.shape[0])
    windows = conv_input.unfold(1, K, 1)[:, :n_tokens, :, :]
    return (windows * conv_weight.T.view(1, 1, conv_weight.shape[1], K)).sum(-1)
