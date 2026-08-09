"""Model configuration module that reads GLM-5.2 (GlmMoeDsa) architecture parameters from HF config.json.

Retain only fields needed for quantization and inference: layer and expert counts, top-k, hidden dimensions,
MLA attention parameters, RoPE parameters, and so on. Skip the entire MTP layer (num_nextn_predict_layers,
layer 78 in this model) following HF conventions. DSA indexer weights are not quantized because for short
contexts below 2048, TPQ full attention is exactly equivalent to top-2048 sparse attention. See Appendix B
of docs/METHODOLOGY.md, "align numerics with full attention first."
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

from .presets import load_manifest as _load_tpq_manifest


@dataclass
class ModelConfig:
    """Architecture description for GLM-5.2 (GlmMoeDsaForCausalLM)."""

    n_layers: int                 # Number of main-model layers (excluding the MTP layer)
    hidden: int                   # hidden_size
    n_experts: int                # n_routed_experts
    top_k: int                    # num_experts_per_tok
    moe_inter: int                # moe_intermediate_size (intermediate dimension of routed experts)
    n_shared: int                 # n_shared_experts
    inter_dense: int              # intermediate_size (MLP intermediate dimension of dense layers)
    n_heads: int                  # num_attention_heads
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_head_dim: int
    vocab: int
    rms_eps: float
    rope_theta: float
    rope_interleave: bool         # Whether main-attention RoPE is interleaved (GLM-5.2 = True)
    norm_topk_prob: bool
    routed_scaling: float
    n_group: int
    topk_group: int
    scoring_func: str             # sigmoid
    tie_embeddings: bool
    eos_token_id: list[int]
    moe_layers: list[int] = field(default_factory=list)   # Layer indices where mlp_layer_types == "sparse"
    dense_layers: list[int] = field(default_factory=list) # Layer indices where mlp_layer_types == "dense"
    max_position_embeddings: int = 0

    @classmethod
    def from_hf(cls, path: str) -> "ModelConfig":
        """Construct from config.json in an HF model directory; path may be a directory or config.json file."""
        p = path if path.endswith(".json") else os.path.join(path, "config.json")
        with open(p, "r", encoding="utf-8") as f:
            c = json.load(f)
        n_layers = int(c["num_hidden_layers"])  # 78, excluding MTP layer 78
        types = c.get("mlp_layer_types")
        if types is None:  # Legacy configuration: the first first_k_dense_replace layers are dense
            k = int(c.get("first_k_dense_replace", 0))
            types = ["dense" if i < k else "sparse" for i in range(n_layers)]
        types = types[:n_layers]
        moe = [i for i, t in enumerate(types) if t == "sparse"]
        dense = [i for i, t in enumerate(types) if t != "sparse"]
        rope = c.get("rope_parameters") or {}
        eos = c.get("eos_token_id", [])
        if isinstance(eos, int):
            eos = [eos]
        return cls(
            n_layers=n_layers,
            hidden=int(c["hidden_size"]),
            n_experts=int(c["n_routed_experts"]),
            top_k=int(c["num_experts_per_tok"]),
            moe_inter=int(c["moe_intermediate_size"]),
            n_shared=int(c.get("n_shared_experts", 0)),
            inter_dense=int(c["intermediate_size"]),
            n_heads=int(c["num_attention_heads"]),
            q_lora_rank=int(c["q_lora_rank"]),
            kv_lora_rank=int(c["kv_lora_rank"]),
            qk_nope_head_dim=int(c["qk_nope_head_dim"]),
            qk_rope_head_dim=int(c["qk_rope_head_dim"]),
            v_head_dim=int(c["v_head_dim"]),
            qk_head_dim=int(c["qk_head_dim"]),
            vocab=int(c["vocab_size"]),
            rms_eps=float(c.get("rms_norm_eps", 1e-5)),
            rope_theta=float(rope.get("rope_theta", 10000.0)),
            rope_interleave=bool(c.get("rope_interleave", True)),
            norm_topk_prob=bool(c.get("norm_topk_prob", True)),
            routed_scaling=float(c.get("routed_scaling_factor", 1.0)),
            n_group=int(c.get("n_group", 1)),
            topk_group=int(c.get("topk_group", 1)),
            scoring_func=str(c.get("scoring_func", "sigmoid")),
            tie_embeddings=bool(c.get("tie_word_embeddings", False)),
            eos_token_id=eos,
            moe_layers=moe,
            dense_layers=dense,
            max_position_embeddings=int(c.get("max_position_embeddings", 0)),
        )

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "ModelConfig":
        return cls(**d)


@dataclass
class DSV4Config:
    """Architecture description for DeepSeek-V4 (deepseek_v4, such as DeepSeek-V4-Flash-DSpark).

    Differences from GLM: q LoRA (1024 to 64 heads x 512), MQA (kv=1) plus o LoRA (o_groups=8),
    head_dim=512 with RoPE applied only to the qk_rope_head_dim=64 segment, KV Compressor plus Indexer
    (DSA-style sparse attention), sliding-window layers, hash layers (static tid2eid routing plus hc tables),
    sqrtsoftplus routing (top-6), FP4 expert storage (e2m1 with ue8m0 blocks of 32), and YaRN RoPE.
    """

    n_layers: int                 # num_hidden_layers (43, excluding the MTP layer)
    hidden: int                   # hidden_size
    n_experts: int                # n_routed_experts
    top_k: int                    # num_experts_per_tok（6）
    moe_inter: int                # moe_intermediate_size
    n_shared: int                 # n_shared_experts
    n_heads: int                  # num_attention_heads（64）
    head_dim: int                 # head_dim（512）
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    kv_dim: int                   # wkv output dimension (512 = kv_lora plus RoPE segment, inferred from weight shape)
    qk_rope_head_dim: int
    n_kv_heads: int               # num_key_value_heads（1，MQA）
    vocab: int
    rms_eps: float
    scoring_func: str             # sqrtsoftplus
    norm_topk_prob: bool
    routed_scaling: float
    swiglu_limit: float
    n_hash_layers: int            # num_hash_layers (number of static tid2eid routing layers)
    sliding_window: int
    rope_theta: float
    rope_scaling: dict
    eos_token_id: list[int]
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    max_position_embeddings: int = 1_048_576
    n_mtp_layers: int = 1
    hc_mult: int = 4                  # Number of Hyper-Connections channels
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    compress_rope_theta: float = 160000.0   # RoPE theta for ratio>0 (compression) layers
    compress_ratios: list[int] = field(default_factory=list)  # Per-layer compression ratio (0/4/128)

    @classmethod
    def from_hf(cls, path: str) -> "DSV4Config":
        """Construct from config.json in an HF model directory; path may be a directory or config.json file."""
        p = path if path.endswith(".json") else os.path.join(path, "config.json")
        with open(p, "r", encoding="utf-8") as f:
            c = json.load(f)
        eos = c.get("eos_token_id", [])
        if isinstance(eos, int):
            eos = [eos]
        return cls(
            n_layers=int(c["num_hidden_layers"]),
            hidden=int(c["hidden_size"]),
            n_experts=int(c["n_routed_experts"]),
            top_k=int(c["num_experts_per_tok"]),
            moe_inter=int(c["moe_intermediate_size"]),
            n_shared=int(c.get("n_shared_experts", 0)),
            n_heads=int(c["num_attention_heads"]),
            head_dim=int(c.get("head_dim", 512)),
            q_lora_rank=int(c["q_lora_rank"]),
            o_lora_rank=int(c.get("o_lora_rank", 0)),
            o_groups=int(c.get("o_groups", 1)),
            kv_dim=int(c.get("kv_dim", 512)),
            qk_rope_head_dim=int(c["qk_rope_head_dim"]),
            n_kv_heads=int(c.get("num_key_value_heads", 1)),
            vocab=int(c["vocab_size"]),
            rms_eps=float(c.get("rms_norm_eps", 1e-6)),
            scoring_func=str(c.get("scoring_func", "sqrtsoftplus")),
            norm_topk_prob=bool(c.get("norm_topk_prob", True)),
            routed_scaling=float(c.get("routed_scaling_factor", 1.0)),
            swiglu_limit=float(c.get("swiglu_limit", 0.0)),
            n_hash_layers=int(c.get("num_hash_layers", 0)),
            sliding_window=int(c.get("sliding_window", 0)),
            rope_theta=float(c.get("rope_theta", 10000.0)),
            rope_scaling=dict(c.get("rope_scaling") or {}),
            eos_token_id=eos,
            index_n_heads=int(c.get("index_n_heads", 64)),
            index_head_dim=int(c.get("index_head_dim", 128)),
            index_topk=int(c.get("index_topk", 512)),
            max_position_embeddings=int(
                c.get("max_position_embeddings", 1_048_576)
            ),
            n_mtp_layers=int(c.get("num_nextn_predict_layers", 1)),
            hc_mult=int(c.get("hc_mult", 4)),
            hc_eps=float(c.get("hc_eps", 1e-6)),
            hc_sinkhorn_iters=int(c.get("hc_sinkhorn_iters", 20)),
            compress_rope_theta=float(c.get("compress_rope_theta", 160000.0)),
            compress_ratios=[int(v) for v in (c.get("compress_ratios") or [])],
        )

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "DSV4Config":
        values = dict(d)
        values.setdefault("index_n_heads", 64)
        values.setdefault("index_head_dim", 128)
        values.setdefault("index_topk", 512)
        if not values.get("max_position_embeddings"):
            values["max_position_embeddings"] = 1_048_576
        return cls(**values)

    # ---- Derived values ----
    @property
    def expert_params(self) -> int:
        """Parameter count of one routed expert (gate plus up plus down)."""
        return 3 * self.hidden * self.moe_inter

    @property
    def total_routed_params(self) -> int:
        return self.expert_params * self.n_experts * self.n_layers


@dataclass
class KimiK3Config:
    """Normalized text-runtime configuration for Kimi K3."""

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
    kda_layers: list[int] = field(default_factory=list)
    full_attn_layers: list[int] = field(default_factory=list)
    eos_token_id: list[int] = field(default_factory=list)

    @classmethod
    def from_json(cls, values: dict) -> "KimiK3Config":
        return cls(**values)

    @classmethod
    def from_hf(cls, path: str) -> "KimiK3Config":
        config_path = (
            path if path.endswith(".json")
            else os.path.join(path, "config.json")
        )
        with open(config_path, "r", encoding="utf-8") as handle:
            outer = json.load(handle)
        text = outer.get("text_config") or outer
        linear = text.get("linear_attn_config") or {}
        eos = text.get("eos_token_id", outer.get("eos_token_id", []))
        if isinstance(eos, int):
            eos = [eos]
        return cls(
            n_layers=int(text["num_hidden_layers"]),
            hidden=int(text["hidden_size"]),
            routed_hidden=int(
                text.get("routed_expert_hidden_size", text["hidden_size"])
            ),
            n_experts=int(text["num_experts"]),
            top_k=int(text["num_experts_per_token"]),
            moe_inter=int(text["moe_intermediate_size"]),
            n_shared=int(text.get("num_shared_experts", 0)),
            inter_dense=int(text["intermediate_size"]),
            first_dense_layers=int(text.get("first_k_dense_replace", 0)),
            vocab=int(text.get("vocab_size", outer.get("vocab_size", 0))),
            rms_eps=float(text.get("rms_norm_eps", 1e-5)),
            routed_scaling=float(text.get("routed_scaling_factor", 1.0)),
            scoring_func=str(
                text.get("moe_router_activation_func", "sigmoid")
            ),
            norm_topk_prob=bool(text.get("moe_renormalize", True)),
            n_group=int(text.get("num_expert_group", 1)),
            topk_group=int(text.get("topk_group", 1)),
            activation=str(text.get("hidden_act", "situ")),
            situ_beta=float(text.get("activation_situ_beta", 4.0)),
            situ_linear_beta=(
                None
                if text.get("activation_situ_linear_beta") is None
                else float(text["activation_situ_linear_beta"])
            ),
            latent_moe_use_norm=bool(
                text.get("latent_moe_use_norm", False)
            ),
            n_heads=int(linear.get(
                "num_heads", text["num_attention_heads"]
            )),
            head_dim=int(linear.get("head_dim", 128)),
            kv_lora_rank=int(text["kv_lora_rank"]),
            qk_nope_head_dim=int(text["qk_nope_head_dim"]),
            qk_rope_head_dim=int(text["qk_rope_head_dim"]),
            v_head_dim=int(text["v_head_dim"]),
            q_lora_rank=int(text["q_lora_rank"]),
            max_position_embeddings=int(
                text.get("max_position_embeddings", 1_048_576)
            ),
            attn_res_block_size=int(
                text.get("attn_res_block_size", 12)
            ),
            # Published Kimi config uses one-based layer numbers.
            kda_layers=[int(value) - 1 for value in linear.get(
                "kda_layers", []
            )],
            full_attn_layers=[int(value) - 1 for value in linear.get(
                "full_attn_layers", []
            )],
            eos_token_id=[int(value) for value in eos],
        )

    def to_json(self) -> dict:
        return asdict(self)


def detect_arch(path: str) -> str:
    """Read model_type from config.json and return an architecture ID: glm / deepseek_v4 / qwen3_moe."""
    p = path if path.endswith(".json") else os.path.join(path, "config.json")
    with open(p, "r", encoding="utf-8") as f:
        c = json.load(f)
    mt = str(c.get("model_type", "")).lower()
    if "kimi_k3" in mt or "kimik3" in mt:
        return "kimi_k3"
    if "deepseek_v4" in mt or "deepseek-v4" in mt:
        return "deepseek_v4"
    if "glm" in mt:
        return "glm"
    if "qwen" in mt and "moe" in mt:
        return "qwen3_moe"
    if "deepseek" in mt:
        return "deepseek_v3"
    raise ValueError(f"未支持的架构 model_type={mt!r}（{p}）")


def load_config(path: str):
    """Load configuration by architecture, returning ModelConfig for glm or DSV4Config for deepseek_v4/v3."""
    arch = detect_arch(path)
    if arch == "kimi_k3":
        root = path if os.path.isdir(path) else os.path.dirname(path)
        try:
            _root, manifest = _load_tpq_manifest(root)
        except (OSError, ValueError, KeyError, TypeError):
            manifest = None
        if manifest is not None:
            values = manifest["config"]
            return KimiK3Config.from_json(values)
        return KimiK3Config.from_hf(path)
    if arch == "glm":
        return ModelConfig.from_hf(path)
    if arch in ("deepseek_v4", "deepseek_v3"):
        return DSV4Config.from_hf(path)
    raise ValueError(f"架构 {arch} 的配置解析尚未实现")

    # ---- Derived values ----
    @property
    def expert_params(self) -> int:
        """Parameter count of one routed expert (gate plus up plus down)."""
        return 3 * self.hidden * self.moe_inter

    @property
    def total_routed_params(self) -> int:
        return self.expert_params * self.n_experts * len(self.moe_layers)
