"""Architecture-neutral tensor roles and standard mixed-bit presets.

The policy mirrors llama.cpp's common K-quant mixtures, but it does not know
about any concrete model class.  Checkpoint-specific code only supplies tensor
names/shapes and a small topology summary; this module turns them into semantic
roles and target recipe types.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

STANDARD_PRESET_NAMES = (
    "Q2_K_S",
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
)

_ALIASES = {
    "Q3_K": "Q3_K_M",
    "Q4_K": "Q4_K_M",
    "Q5_K": "Q5_K_M",
}

_DEFAULT_TYPE = {
    "Q2_K_S": "Q2_K",
    "Q2_K": "Q2_K",
    "Q3_K_S": "Q3_K",
    "Q3_K_M": "Q3_K",
    "Q3_K_L": "Q3_K",
    "Q4_K_S": "Q4_K",
    "Q4_K_M": "Q4_K",
    "Q5_K_S": "Q5_K",
    "Q5_K_M": "Q5_K",
    "Q6_K": "Q6_K",
    "Q8_0": "Q8_0",
}


class TensorRole(str, Enum):
    TOKEN_EMBEDDING = "token_embedding"
    PLE_EMBEDDING = "ple_embedding"
    OUTPUT = "output"
    ATTENTION_Q = "attention_q"
    ATTENTION_K = "attention_k"
    ATTENTION_V = "attention_v"
    ATTENTION_OUTPUT = "attention_output"
    FFN_GATE = "ffn_gate"
    FFN_UP = "ffn_up"
    FFN_DOWN = "ffn_down"
    OTHER = "other"


class TensorScope(str, Enum):
    TEXT = "text"
    VISION = "vision"
    PREDICTOR = "predictor"
    OTHER = "other"


@dataclass(frozen=True)
class ModelTopology:
    text_layers: int
    text_heads: int
    text_kv_heads: int
    vision_layers: int = 0
    vision_heads: int = 0
    vision_kv_heads: int = 0

    @property
    def text_gqa(self) -> int:
        return max(1, self.text_heads // max(1, self.text_kv_heads))

    @property
    def vision_gqa(self) -> int:
        return max(1, self.vision_heads // max(1, self.vision_kv_heads))

    @classmethod
    def from_hf_config(cls, config: Mapping[str, object]) -> ModelTopology:
        raw_text = config.get("text_config", config)
        text = raw_text if isinstance(raw_text, Mapping) else {}
        raw_vision = config.get("vision_config", {})
        vision = raw_vision if isinstance(raw_vision, Mapping) else {}
        text_heads = int(text.get("num_attention_heads", 0) or 0)
        text_kv_heads = int(text.get("num_key_value_heads", text_heads) or text_heads or 1)
        vision_heads = int(vision.get("num_heads", vision.get("num_attention_heads", 0)) or 0)
        vision_kv_heads = int(vision.get("num_key_value_heads", vision_heads) or vision_heads or 1)
        return cls(
            text_layers=int(text.get("num_hidden_layers", 0) or 0),
            text_heads=text_heads,
            text_kv_heads=text_kv_heads,
            vision_layers=int(vision.get("depth", vision.get("num_hidden_layers", 0)) or 0),
            vision_heads=vision_heads,
            vision_kv_heads=vision_kv_heads,
        )


@dataclass(frozen=True)
class TensorDescriptor:
    role: TensorRole
    scope: TensorScope
    layer_index: int | None
    quantizable: bool


_LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|blocks|block|blk)\.(\d+)(?:\.|$)")
_VISION_COMPONENTS = frozenset(
    {
        "visual",
        "vision",
        "vision_model",
        "vision_tower",
        "vision_encoder",
        "image_encoder",
        "image_tower",
        "vpm",
        "vit",
    }
)
_PREDICTOR_COMPONENTS = frozenset({"mtp", "draft", "draft_model", "speculator", "medusa"})
_DENSE_COMPONENTS = frozenset(
    {
        "merger",
        "projector",
        "connector",
        "patch_embed",
        "patch_embedding",
        "pos_embed",
        "position_embeddings",
        "relative_position",
        "router",
    }
)


def normalize_preset(value: str) -> str:
    preset = value.strip().upper().replace("-", "_")
    preset = _ALIASES.get(preset, preset)
    if preset not in _DEFAULT_TYPE:
        supported = ", ".join(STANDARD_PRESET_NAMES)
        raise ValueError(
            f"unsupported standard quantization preset {value!r}; choose one of: {supported}"
        )
    return preset


def _scope(name: str) -> TensorScope:
    components = set(name.split("."))
    if components & _VISION_COMPONENTS:
        return TensorScope.VISION
    if components & _PREDICTOR_COMPONENTS:
        return TensorScope.PREDICTOR
    if components & {"language_model", "text_model", "llm"} or name.startswith(
        ("model.", "blk.")
    ):
        return TensorScope.TEXT
    return TensorScope.OTHER


def _role(name: str, canonical_name: str | None) -> TensorRole:
    canonical = canonical_name or name
    if (
        ".position_embedding.ngram.shard." in canonical
        or ".ngram_embedding.shard_" in name
    ) and canonical.endswith(".weight"):
        return TensorRole.PLE_EMBEDDING
    if canonical in {"output.weight", "model.output.weight"} or name.endswith("lm_head.weight"):
        return TensorRole.OUTPUT
    if canonical in {
        "token_embd.weight",
        "per_layer_token_embd.weight",
        "model.token_embedding.weight",
    } or name.endswith(("embed_tokens.weight", "token_embedding.weight")):
        return TensorRole.TOKEN_EMBEDDING
    if any(
        marker in canonical
        for marker in (
            "attn_qkv.weight",
            "attn_kv_b.weight",
            "attn_v.weight",
            "attention.qkv.weight",
            "attention.value.weight",
        )
    ) or name.endswith(("qkv.weight", "in_proj_qkv.weight", "v_proj.weight")):
        return TensorRole.ATTENTION_V
    if (
        "attn_k.weight" in canonical
        or ".attention.key.weight" in canonical
        or name.endswith("k_proj.weight")
    ):
        return TensorRole.ATTENTION_K
    if (
        "attn_q.weight" in canonical
        or ".attention.query.weight" in canonical
        or name.endswith("q_proj.weight")
    ):
        return TensorRole.ATTENTION_Q
    if (
        "attn_output.weight" in canonical
        or ".attention.output.weight" in canonical
        or name.endswith(("o_proj.weight", "attn.proj.weight"))
    ):
        return TensorRole.ATTENTION_OUTPUT
    if (
        "ffn_down" in canonical
        or ".mlp.down" in canonical
        or name.endswith(("down_proj.weight", "down_proj", "linear_fc2.weight"))
    ):
        return TensorRole.FFN_DOWN
    if (
        "ffn_up" in canonical
        or ".mlp.up" in canonical
        or name.endswith(("up_proj.weight", "up_proj", "linear_fc1.weight"))
    ):
        return TensorRole.FFN_UP
    if (
        "ffn_gate" in canonical
        or ".mlp.gate" in canonical
        or name.endswith(("gate_proj.weight", "gate_proj"))
    ):
        return TensorRole.FFN_GATE
    return TensorRole.OTHER


def describe_tensor(
    name: str,
    shape: Sequence[int],
    source_dtype: str,
    *,
    canonical_name: str | None = None,
) -> TensorDescriptor:
    """Describe a tensor using common HF/GGUF naming conventions."""

    source_scope = _scope(name)
    canonical_scope = _scope(canonical_name) if canonical_name else TensorScope.OTHER
    scope = (
        canonical_scope
        if canonical_scope in {TensorScope.VISION, TensorScope.PREDICTOR}
        else source_scope
        if source_scope in {TensorScope.VISION, TensorScope.PREDICTOR}
        else canonical_scope
        if canonical_scope is not TensorScope.OTHER
        else source_scope
    )
    role = _role(name, canonical_name)
    source_layer_match = _LAYER_PATTERN.search(name)
    match = source_layer_match or _LAYER_PATTERN.search(canonical_name or "")
    layer_index = int(match.group(1)) if match is not None else None
    components = set(name.split("."))
    quantizable = len(shape) in (2, 3) and source_dtype not in {"I32", "I64"}
    quantizable &= len(shape) == 3 or name.endswith(".weight")
    quantizable &= not any(
        marker in name
        for marker in (
            "_norm.weight",
            "layernorm.weight",
            ".norm.weight",
            "ssm_conv1d.weight",
            "conv1d.weight",
            "shortconv.conv.weight",
            ".mlp.gate.weight",
            "shared_expert_gate.weight",
            ".position_embd",
            ".rel_pos",
        )
    )
    quantizable &= not bool(components & _DENSE_COMPONENTS)
    # Root-level predictor fusion/projection matrices are small, shared, and
    # outside the repeated decoder stack.  Preserve them without naming a
    # particular speculative architecture.
    quantizable &= not (scope is TensorScope.PREDICTOR and source_layer_match is None)
    return TensorDescriptor(role, scope, layer_index, bool(quantizable))


def use_more_bits(index: int, count: int) -> bool:
    if count <= 0:
        return False
    eighth = count // 8
    return index < eighth or index >= 7 * eighth or (index - eighth) % 3 == 2


def select_recipe_type(
    preset: str,
    role: TensorRole,
    *,
    layer_index: int | None,
    layer_count: int,
    attention_index: int | None,
    attention_count: int,
    gqa: int,
) -> str:
    """Select one llama.cpp recipe type from a semantic tensor role."""

    preset = normalize_preset(preset)
    target = _DEFAULT_TYPE[preset]
    if role is TensorRole.OUTPUT:
        return "Q8_0" if preset == "Q8_0" else "Q6_K"
    if role is TensorRole.ATTENTION_V:
        index = attention_index or 0
        if preset == "Q2_K":
            return "Q4_K" if gqa >= 4 else "Q3_K"
        if preset == "Q2_K_S" and gqa >= 4:
            return "Q4_K"
        if preset == "Q3_K_M":
            return "Q5_K" if index < 2 else "Q4_K"
        if preset == "Q3_K_L":
            return "Q5_K"
        if preset in {"Q4_K_M", "Q5_K_M"} and use_more_bits(index, attention_count):
            return "Q6_K"
        if preset == "Q4_K_S" and index < 4:
            return "Q5_K"
        return target
    if role is TensorRole.FFN_DOWN:
        index = layer_index or 0
        layers = max(1, layer_count)
        if preset == "Q2_K":
            return "Q3_K"
        if preset == "Q2_K_S" and index < layers // 8:
            return "Q4_K"
        if preset == "Q3_K_M":
            if index < layers // 16:
                return "Q5_K"
            return "Q4_K" if use_more_bits(index, layers) else "Q3_K"
        if preset == "Q3_K_L":
            return "Q5_K"
        if preset in {"Q4_K_M", "Q5_K_M"} and use_more_bits(index, layers):
            return "Q6_K"
        if preset == "Q4_K_S" and index < layers // 8:
            return "Q5_K"
        return target
    if role is TensorRole.ATTENTION_OUTPUT:
        if preset == "Q2_K":
            return "Q3_K"
        if preset == "Q3_K_M":
            return "Q4_K"
        if preset == "Q3_K_L":
            return "Q5_K"
    return target


__all__ = [
    "ModelTopology",
    "STANDARD_PRESET_NAMES",
    "TensorDescriptor",
    "TensorRole",
    "TensorScope",
    "describe_tensor",
    "normalize_preset",
    "select_recipe_type",
    "use_more_bits",
]
