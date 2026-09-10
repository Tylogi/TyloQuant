"""Architecture-neutral tensor roles and MFQ standard mixed-bit presets."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

STANDARD_PRESET_NAMES = (
    "S2-S",
    "S2-M",
    "S3-S",
    "S3-M",
    "S3-L",
    "S4-S",
    "S4-M",
    "S5-S",
    "S5-M",
    "S6",
    "S8",
)

_DEFAULT_DTYPE = {
    "S2-S": "NINT2",
    "S2-M": "NINT2",
    "S3-S": "NINT3",
    "S3-M": "NINT3",
    "S3-L": "NINT3",
    "S4-S": "NINT4",
    "S4-M": "NINT4",
    "S5-S": "NINT5",
    "S5-M": "NINT5",
    "S6": "NINT6",
    "S8": "NINT8",
}


class TensorRole(str, Enum):
    TOKEN_EMBEDDING = "token_embedding"
    PLE_EMBEDDING = "ple_embedding"
    PLE_PROJECTION = "ple_projection"
    OUTPUT = "output"
    ATTENTION_Q = "attention_q"
    ATTENTION_K = "attention_k"
    ATTENTION_V = "attention_v"
    ATTENTION_OUTPUT = "attention_output"
    ATTENTION_INDEXER = "attention_indexer"
    FFN_GATE = "ffn_gate"
    FFN_UP = "ffn_up"
    FFN_DOWN = "ffn_down"
    SHARED_EXPERT = "shared_expert"
    ROUTED_EXPERT = "routed_expert"
    MHC = "mhc"
    SHORT_CONVOLUTION = "short_convolution"
    OTHER = "other"


class TensorScope(str, Enum):
    TEXT = "text"
    VISION = "vision"
    PREDICTOR = "predictor"
    PLE = "ple"
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
_PREDICTOR_COMPONENTS = frozenset(
    {"mtp", "predictor", "draft", "draft_model", "speculator", "medusa"}
)
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
    preset = value.strip().upper().replace("_", "-")
    if preset not in _DEFAULT_DTYPE:
        supported = ", ".join(STANDARD_PRESET_NAMES)
        raise ValueError(
            f"unsupported standard quantization preset {value!r}; choose one of: {supported}"
        )
    return preset


def _is_ple_path(name: str) -> bool:
    components = set(name.split("."))
    return (
        "ple" in components
        or "associative_memory" in components
        or bool(re.search(r"(?:^|\.)block\.\d+\.position_embedding\.", name))
    )


def _scope(name: str) -> TensorScope:
    components = set(name.split("."))
    if components & _VISION_COMPONENTS:
        return TensorScope.VISION
    if components & _PREDICTOR_COMPONENTS:
        return TensorScope.PREDICTOR
    if _is_ple_path(name):
        return TensorScope.PLE
    if components & {"language_model", "text_model", "llm"} or name.startswith(("model.", "blk.")):
        return TensorScope.TEXT
    return TensorScope.OTHER


def _role(name: str, canonical_name: str | None) -> TensorRole:
    canonical = canonical_name or name
    names = (name, canonical)
    if (
        ".position_embedding.ngram.shard." in canonical or ".ngram_embedding.shard_" in name
        or canonical.endswith(".associative_memory.embedding.weight")
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
        any(
            component in {"mhc", "hyper_connection", "hyper_connection_mixer"}
            or component.endswith("_hyper_connection")
            for component in value.split(".")
        )
        for value in names
    ):
        return TensorRole.MHC
    if any("indexer" in value.split(".") for value in names):
        return TensorRole.ATTENTION_INDEXER
    if any(
        value.endswith(".weight")
        and any(
            component in {"conv", "conv1d", "ssm_conv1d"} or component.endswith("_conv")
            for component in value.split(".")
        )
        for value in names
    ):
        return TensorRole.SHORT_CONVOLUTION
    if any(
        any(component in {"shared_expert", "shared_experts"} for component in value.split("."))
        or "_shexp." in value
        for value in names
    ):
        return TensorRole.SHARED_EXPERT
    if any(
        any(component in {"experts", "expert"} for component in value.split(".")) for value in names
    ):
        return TensorRole.ROUTED_EXPERT
    if any(_is_ple_path(value) for value in names):
        return TensorRole.PLE_PROJECTION
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
        if canonical_scope in {TensorScope.VISION, TensorScope.PREDICTOR, TensorScope.PLE}
        else source_scope
        if source_scope in {TensorScope.VISION, TensorScope.PREDICTOR, TensorScope.PLE}
        else canonical_scope
        if canonical_scope is not TensorScope.OTHER
        else source_scope
    )
    role = _role(name, canonical_name)
    source_layer_match = _LAYER_PATTERN.search(name)
    match = source_layer_match or _LAYER_PATTERN.search(canonical_name or "")
    layer_index = int(match.group(1)) if match is not None else None
    names = (name, canonical_name or name)
    components = {component for value in names for component in value.split(".")}
    is_matrix = len(shape) == 2
    is_expert_bank = len(shape) == 3 and role is TensorRole.ROUTED_EXPERT
    quantizable = (is_matrix or is_expert_bank) and source_dtype not in {"I32", "I64"}
    quantizable &= any(value.endswith(".weight") for value in names)
    quantizable &= not any(
        marker in value
        for value in names
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
    quantizable &= role not in {
        TensorRole.ATTENTION_INDEXER,
        TensorRole.MHC,
        TensorRole.SHORT_CONVOLUTION,
    }
    # Root-level predictor fusion/projection matrices are small, shared, and
    # outside the repeated decoder stack.  Preserve them without naming a
    # particular speculative architecture.
    quantizable &= not (
        scope is TensorScope.PREDICTOR
        and source_layer_match is None
        and role is not TensorRole.ROUTED_EXPERT
    )
    return TensorDescriptor(role, scope, layer_index, bool(quantizable))


def use_more_bits(index: int, count: int) -> bool:
    if count <= 0:
        return False
    eighth = count // 8
    return index < eighth or index >= 7 * eighth or (index - eighth) % 3 == 2


def select_target_dtype(
    preset: str,
    role: TensorRole,
    *,
    layer_index: int | None,
    layer_count: int,
    attention_index: int | None,
    attention_count: int,
    gqa: int,
) -> str:
    """Select one native MFQ target dtype from a semantic tensor role."""

    preset = normalize_preset(preset)
    target = _DEFAULT_DTYPE[preset]
    if role in {TensorRole.OUTPUT, TensorRole.TOKEN_EMBEDDING}:
        return "NINT8" if preset == "S8" else "NINT6"
    if role is TensorRole.SHARED_EXPERT:
        return "NINT8"
    if role is TensorRole.ATTENTION_V:
        index = attention_index or 0
        if preset == "S2-M":
            return "NINT4" if gqa >= 4 else "NINT3"
        if preset == "S2-S" and gqa >= 4:
            return "NINT4"
        if preset == "S3-M":
            return "NINT5" if index < 2 else "NINT4"
        if preset == "S3-L":
            return "NINT5"
        if preset in {"S4-M", "S5-M"} and use_more_bits(index, attention_count):
            return "NINT6"
        if preset == "S4-S" and index < 4:
            return "NINT5"
        return target
    if role is TensorRole.FFN_DOWN:
        index = layer_index or 0
        layers = max(1, layer_count)
        if preset == "S2-M":
            return "NINT3"
        if preset == "S2-S" and index < layers // 8:
            return "NINT4"
        if preset == "S3-M":
            if index < layers // 16:
                return "NINT5"
            return "NINT4" if use_more_bits(index, layers) else "NINT3"
        if preset == "S3-L":
            return "NINT5"
        if preset in {"S4-M", "S5-M"} and use_more_bits(index, layers):
            return "NINT6"
        if preset == "S4-S" and index < layers // 8:
            return "NINT5"
        return target
    if role is TensorRole.ATTENTION_OUTPUT:
        if preset == "S2-M":
            return "NINT3"
        if preset == "S3-M":
            return "NINT4"
        if preset == "S3-L":
            return "NINT5"
    return target


__all__ = [
    "ModelTopology",
    "STANDARD_PRESET_NAMES",
    "TensorDescriptor",
    "TensorRole",
    "TensorScope",
    "describe_tensor",
    "normalize_preset",
    "select_target_dtype",
    "use_more_bits",
]
