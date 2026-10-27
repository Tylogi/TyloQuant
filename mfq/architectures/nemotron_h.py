"""Strict normalized contract for NVIDIA Nemotron-H and its omni wrapper.

A released ``nemotron_h_omni`` checkpoint nests the language model under
``llm_config`` (not ``text_config``), labels its decoder blocks with the legacy
mixer names ``mamba``/``attention``, and derives the layer count from
``layers_block_type``.  Those conventions stop here: every other module
consumes the normalized fields below.  A plain ``nemotron_h`` text checkpoint
is the same contract without the outer wrapper.

Every block is exactly one pre-normed mixer, so ``layers_block_type`` is the
only authority for a block's kind.  All four mixer kinds transformers still
accepts are supported: ``linear_attention`` (Mamba-2, stored as legacy
``mamba``), ``full_attention`` (NoPE GQA, stored as legacy ``attention``),
``moe`` (latent relu-squared experts) and ``mlp`` (the dense
``NemotronHMLP``).

Two fields are training-only and therefore ignored by the inference contract:
``rescale_prenorm_residual`` (an initialization trick) and
``residual_in_fp32`` (always false for this family).  The inference time-step
clamp is ``time_step_limit``; ``time_step_min``/``time_step_max``/
``time_step_floor`` only steer the training-time initialization and are kept
so a round trip through the importer does not lose them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mfq.architectures.hf_config import text_section

TEXT_MODEL_TYPES = frozenset({"nemotron_h", "nemotron_h_text"})
OMNI_MODEL_TYPES = frozenset({"nemotron_h_omni"})

# Transformers renamed the mixer labels and still accepts the legacy spelling
# found in released configs.
LEGACY_MIXER_NAMES = {"mamba": "linear_attention", "attention": "full_attention"}
MIXER_TYPES = ("linear_attention", "full_attention", "moe", "mlp")

# The routed experts of this family are plain relu-squared MLPs (no gate).
EXPERT_ACTIVATIONS = {"relu2"}
MAMBA_ACTIVATIONS = {"silu"}


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _positive(value: object, name: str) -> int:
    result = int(value or 0)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    return tuple(str(item) for item in value)


def _integers(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    return tuple(int(item) for item in value)


def _float(value: object, name: str) -> float:
    """Resolve one float, including the ``{"__float__": "Infinity"}`` form."""

    if isinstance(value, Mapping):
        raw = value.get("__float__")
        if raw is None:
            raise ValueError(f"{name} must be a float")
        value = raw
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a float") from None


def _eos_ids(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    return _integers(value, "eos_token_id")


def _sequence(raw: Mapping[str, object], name: str) -> Sequence[object]:
    value = raw.get(name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    return value


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def normalize_mixer_types(value: object, name: str) -> tuple[str, ...]:
    """Map one legacy block-type list onto the current mixer vocabulary."""

    mixers = tuple(LEGACY_MIXER_NAMES.get(raw, raw) for raw in _strings(value, name))
    unknown = sorted(set(mixers) - set(MIXER_TYPES))
    if unknown:
        raise ValueError(
            f"{name} contains unsupported mixer types: {unknown}; "
            f"Nemotron-H accepts {sorted(MIXER_TYPES)}"
        )
    return mixers


def layer_mixers_from_config(config: Mapping[str, object]) -> tuple[str, ...]:
    """Return the backbone mixer schedule without validating anything else."""

    text = text_section(config)
    raw = text.get("layers_block_type", text.get("layer_types"))
    if raw is None and "hybrid_override_pattern" in text:
        pattern = text["hybrid_override_pattern"]
        labels = {"M": "linear_attention", "*": "full_attention", "-": "mlp", "E": "moe"}
        if not isinstance(pattern, str) or not pattern or set(pattern) - labels.keys():
            raise ValueError("hybrid_override_pattern must contain only M, *, -, E")
        raw = [labels[item] for item in pattern]
    return normalize_mixer_types(raw, "layers_block_type")


def mtp_mixers_from_config(config: Mapping[str, object]) -> tuple[str, ...]:
    """Return the predictor mixer schedule declared by ``mtp_layers_block_type``."""

    text = text_section(config)
    if int(text.get("num_nextn_predict_layers", 0) or 0) <= 0:
        return ()
    raw = text.get("mtp_layers_block_type")
    if raw is None:
        raise ValueError(
            "mtp_layers_block_type is required when num_nextn_predict_layers > 0"
        )
    return normalize_mixer_types(raw, "mtp_layers_block_type")


@dataclass(frozen=True)
class RadioVisionConfig:
    """C-RADIO vision tower geometry (LayerNorm, CPE, GELU, non-causal)."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    num_channels: int
    patch_size: int
    layer_norm_eps: float
    max_img_size: int
    num_cls_tokens: int
    num_registers: int
    video_temporal_patch_size: int | None
    summary_idxs: tuple[int, ...]

    @property
    def position_side(self) -> int:
        return self.max_img_size // self.patch_size

    @property
    def summary_tokens(self) -> int:
        return self.num_cls_tokens + self.num_registers

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> RadioVisionConfig:
        model_type = str(raw.get("model_type", ""))
        if model_type != "radio":
            raise ValueError(f"Nemotron-H vision tower must be radio, got {model_type!r}")
        hidden = _positive(raw.get("hidden_size"), "vision_config.hidden_size")
        heads = _positive(raw.get("num_attention_heads"), "vision_config.num_attention_heads")
        if hidden % heads:
            raise ValueError("Nemotron-H vision hidden size must divide its heads")
        if (
            str(raw.get("hidden_act", "")) != "gelu"
            or not bool(raw.get("qkv_bias", False))
            or bool(raw.get("use_swiglu_ffn", False))
        ):
            raise ValueError("unsupported Nemotron-H vision-tower activation semantics")
        if float(raw.get("layerscale_value", 1.0)) != 1.0:
            raise ValueError(
                "Nemotron-H checkpoints carry no LayerScale weights, so "
                "vision_config.layerscale_value must be 1.0"
            )
        ratio = float(raw.get("mlp_ratio", 4.0))
        intermediate = int(hidden * ratio)
        if intermediate <= 0 or intermediate != hidden * ratio:
            raise ValueError("Nemotron-H vision mlp_ratio must describe a whole width")
        patch_size = _positive(raw.get("patch_size"), "vision_config.patch_size")
        cls_tokens = _positive(raw.get("num_cls_tokens"), "vision_config.num_cls_tokens")
        registers = int(raw.get("num_registers", 0) or 0)
        summary_idxs = _integers(raw.get("summary_idxs", ()), "vision_config.summary_idxs")
        if any(index < 0 or index >= cls_tokens + registers for index in summary_idxs):
            raise ValueError("Nemotron-H vision summary_idxs must address cls/register tokens")
        temporal = raw.get("video_temporal_patch_size")
        return cls(
            model_type=model_type,
            hidden_size=hidden,
            num_hidden_layers=_positive(
                raw.get("num_hidden_layers"), "vision_config.num_hidden_layers"
            ),
            num_attention_heads=heads,
            intermediate_size=intermediate,
            num_channels=_positive(
                raw.get("num_channels", 3), "vision_config.num_channels"
            ),
            patch_size=patch_size,
            layer_norm_eps=float(raw.get("layer_norm_eps", 1e-6)),
            max_img_size=_positive(raw.get("max_img_size"), "vision_config.max_img_size"),
            num_cls_tokens=cls_tokens,
            num_registers=registers,
            video_temporal_patch_size=(
                None if temporal is None else _positive(temporal, "video_temporal_patch_size")
            ),
            summary_idxs=summary_idxs,
        )


@dataclass(frozen=True)
class RadioProjectorConfig:
    """Pixel-shuffle plus two-layer projector between RADIO and the text model."""

    vision_hidden_size: int
    intermediate_size: int
    text_hidden_size: int
    downsample_ratio: float
    pixel_shuffle_version: str

    @property
    def squeeze(self) -> int:
        return int(round(1.0 / self.downsample_ratio))

    @property
    def patch_width(self) -> int:
        return self.vision_hidden_size * self.squeeze * self.squeeze

    @classmethod
    def from_hf_config(
        cls,
        outer: Mapping[str, object],
        text_hidden_size: int,
        vision: RadioVisionConfig,
    ) -> RadioProjectorConfig:
        ratio = float(outer.get("downsample_ratio", 0.0) or 0.0)
        if ratio != 0.5:
            raise ValueError(
                "Nemotron-H projector assumes downsample_ratio 0.5 "
                f"(2x2 pixel shuffle), got {ratio}"
            )
        version = str(outer.get("ps_version", ""))
        if version != "v2":
            raise ValueError(
                f"Nemotron-H projector assumes ps_version 'v2', got {version!r}"
            )
        return cls(
            vision_hidden_size=vision.hidden_size,
            intermediate_size=_positive(
                outer.get("projector_hidden_size"), "projector_hidden_size"
            ),
            text_hidden_size=text_hidden_size,
            downsample_ratio=ratio,
            pixel_shuffle_version=version,
        )


@dataclass(frozen=True)
class NemotronHConfig:
    family: str
    model_type: str
    text_model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    mixer_types: tuple[str, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_bias: bool
    max_position_embeddings: int
    norm_epsilon: float
    mamba_num_heads: int
    mamba_head_dim: int
    ssm_state_size: int
    mamba_groups: int
    conv_kernel: int
    conv_bias: bool
    chunk_size: int
    time_step_limit: tuple[float, float]
    time_step_init_limit: tuple[float, float]
    time_step_floor: float
    intermediate_size: int
    moe_intermediate_size: int
    moe_latent_size: int
    moe_shared_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    expert_group_count: int
    expert_group_topk: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    expert_activation: str
    num_nextn_predict_layers: int
    mtp_mixer_types: tuple[str, ...]
    rescale_prenorm_residual: bool
    residual_in_fp32: bool
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...]
    image_token_id: int | None
    video_token_id: int | None
    vision: RadioVisionConfig | None
    projector: RadioProjectorConfig | None

    @property
    def mamba_inner_size(self) -> int:
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def mamba_conv_width(self) -> int:
        """Width of the depthwise causal convolution input (``xBC``)."""

        return self.mamba_inner_size + 2 * self.mamba_groups * self.ssm_state_size

    @property
    def mamba_in_projection_width(self) -> int:
        """Rows of the fused ``in_proj``: z, xBC and the per-head ``dt``."""

        return (
            2 * self.mamba_inner_size
            + 2 * self.mamba_groups * self.ssm_state_size
            + self.mamba_num_heads
        )

    @property
    def attention_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @classmethod
    def from_hf_config(cls, outer: Mapping[str, object]) -> NemotronHConfig:
        outer_type = str(outer.get("model_type", ""))
        text = text_section(outer)
        text_type = str(text.get("model_type", outer_type))
        if outer_type not in OMNI_MODEL_TYPES | TEXT_MODEL_TYPES:
            raise ValueError(f"expected a nemotron_h checkpoint, got {outer_type!r}")
        if text_type not in TEXT_MODEL_TYPES:
            raise ValueError(
                f"expected nemotron_h/nemotron_h_text text config, got {text_type!r}"
            )

        mixer_types = layer_mixers_from_config(outer)
        if not mixer_types:
            raise ValueError("layers_block_type must describe at least one decoder layer")
        declared_layers = text.get("num_hidden_layers")
        if declared_layers is not None and int(declared_layers) != len(mixer_types):
            raise ValueError(
                f"num_hidden_layers ({int(declared_layers)}) disagrees with "
                f"layers_block_type ({len(mixer_types)}); Nemotron-H derives the "
                "layer count from layers_block_type"
            )
        mtp_mixer_types = mtp_mixers_from_config(outer)

        hidden = _positive(text.get("hidden_size"), "hidden_size")
        heads = _positive(text.get("num_attention_heads"), "num_attention_heads")
        kv_heads = _positive(
            text.get("num_key_value_heads", text.get("num_attention_heads")),
            "num_key_value_heads",
        )
        head_dim = _positive(text.get("head_dim"), "head_dim")
        if heads % kv_heads:
            raise ValueError("Nemotron-H attention heads must divide into KV heads")
        if heads * head_dim != hidden and "hybrid_override_pattern" not in text:
            raise ValueError(
                f"Nemotron-H attention width {heads * head_dim} must equal hidden_size {hidden}"
            )
        if bool(text.get("attention_bias", False)):
            raise ValueError("Nemotron-H attention projections carry no bias")
        if bool(text.get("use_bias", False)) or bool(text.get("mlp_bias", False)):
            raise ValueError("Nemotron-H checkpoints carry no MLP or projection biases")
        if bool(text.get("mamba_proj_bias", False)):
            raise ValueError("Nemotron-H mamba projections carry no bias")
        if str(text.get("mamba_hidden_act", "silu")) not in MAMBA_ACTIVATIONS:
            raise ValueError("unsupported Nemotron-H mamba activation")
        if str(text.get("mamba_ssm_cache_dtype", "float32")) != "float32":
            raise ValueError("Nemotron-H requires a float32 SSM cache")

        mamba_heads = _positive(text.get("mamba_num_heads"), "mamba_num_heads")
        mamba_head_dim = _positive(text.get("mamba_head_dim"), "mamba_head_dim")
        groups = _positive(text.get("n_groups"), "n_groups")
        ssm_state = _positive(text.get("ssm_state_size"), "ssm_state_size")
        inner = mamba_heads * mamba_head_dim
        if inner % groups:
            raise ValueError("Nemotron-H mamba inner width must divide its groups")
        expand = int(text.get("expand", 0) or 0)
        if expand and expand * hidden != inner:
            raise ValueError(
                f"Nemotron-H mamba expand {expand} disagrees with the inner width {inner}"
            )
        time_limit = tuple(
            _float(value, "time_step_limit") for value in (_sequence(text, "time_step_limit") if "time_step_limit" in text else (0.0, float("inf")))
        )
        if len(time_limit) != 2 or not 0.0 <= time_limit[0] <= time_limit[1]:
            raise ValueError("Nemotron-H time_step_limit must be [low, high] with 0 <= low <= high")
        time_init = (
            _float(text.get("time_step_min"), "time_step_min"),
            _float(text.get("time_step_max"), "time_step_max"),
        )
        if not 0.0 < time_init[0] <= time_init[1]:
            raise ValueError("Nemotron-H time_step_min/time_step_max must be positive and ordered")

        needs_moe = "moe" in mixer_types or "moe" in mtp_mixer_types
        latent = text.get("moe_latent_size")
        if needs_moe and latent is None:
            raise ValueError(
                "Nemotron-H latent MoE requires moe_latent_size; a routed-expert "
                "bank cannot be sized without it"
            )
        group_count = int(text.get("n_group", 1) or 1)
        group_topk = int(text.get("topk_group", 1) or 1)
        if needs_moe and (group_count != 1 or group_topk != 1):
            raise ValueError(
                "Nemotron-H group-limited routing is not supported: n_group and "
                f"topk_group must both be 1, got {group_count}/{group_topk}"
            )
        expert_activation = str(text.get("mlp_hidden_act", "relu2"))
        if needs_moe and expert_activation not in EXPERT_ACTIVATIONS:
            raise ValueError(
                f"unsupported Nemotron-H routed-expert activation {expert_activation!r}; "
                f"the imported bank is limited to {sorted(EXPERT_ACTIVATIONS)}"
            )
        routed = _positive(text.get("n_routed_experts"), "n_routed_experts") if needs_moe else 0
        active = (
            _positive(text.get("num_experts_per_tok"), "num_experts_per_tok") if needs_moe else 0
        )
        if needs_moe and active > routed:
            raise ValueError("num_experts_per_tok cannot exceed n_routed_experts")
        shared_experts = int(text.get("n_shared_experts", 0) or 0)
        if needs_moe and shared_experts not in {0, 1}:
            raise ValueError("Nemotron-H supports at most one shared expert")

        predictor_layers = int(text.get("num_nextn_predict_layers", 0) or 0)
        if predictor_layers > 1:
            raise ValueError(
                "Nemotron-H multi-stage prediction heads have no canonical fusion "
                f"roots; got num_nextn_predict_layers={predictor_layers}"
            )

        vision_raw = outer.get("vision_config")
        vision = (
            None
            if vision_raw is None
            else RadioVisionConfig.from_mapping(_object(vision_raw, "vision_config"))
        )
        vit_hidden = outer.get("vit_hidden_size")
        if vision is not None and vit_hidden is not None and int(vit_hidden) != vision.hidden_size:
            raise ValueError("vit_hidden_size must equal vision_config.hidden_size")
        projector = (
            None
            if vision is None
            else RadioProjectorConfig.from_hf_config(outer, hidden, vision)
        )
        return cls(
            family="nemotron_h",
            model_type=outer_type,
            text_model_type=text_type,
            vocab_size=_positive(text.get("vocab_size"), "vocab_size"),
            hidden_size=hidden,
            num_hidden_layers=len(mixer_types),
            mixer_types=mixer_types,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            attention_bias=bool(text.get("attention_bias", False)),
            max_position_embeddings=_positive(
                text.get("max_position_embeddings"), "max_position_embeddings"
            ),
            norm_epsilon=float(text.get("layer_norm_epsilon", text.get("rms_norm_eps", 1e-5))),
            mamba_num_heads=mamba_heads,
            mamba_head_dim=mamba_head_dim,
            ssm_state_size=ssm_state,
            mamba_groups=groups,
            conv_kernel=_positive(text.get("conv_kernel"), "conv_kernel"),
            conv_bias=bool(text.get("use_conv_bias", True)),
            chunk_size=_positive(text.get("chunk_size"), "chunk_size"),
            time_step_limit=(time_limit[0], time_limit[1]),
            time_step_init_limit=time_init,
            time_step_floor=_float(text.get("time_step_floor"), "time_step_floor"),
            intermediate_size=_positive(text.get("intermediate_size"), "intermediate_size"),
            moe_intermediate_size=(
                _positive(text.get("moe_intermediate_size"), "moe_intermediate_size")
                if needs_moe
                else 0
            ),
            moe_latent_size=_positive(latent, "moe_latent_size") if needs_moe else 0,
            moe_shared_intermediate_size=(
                int(text.get("moe_shared_expert_intermediate_size", 0) or 0)
                if needs_moe
                else 0
            ),
            num_experts=routed,
            num_experts_per_tok=active,
            n_shared_experts=shared_experts if needs_moe else 0,
            expert_group_count=group_count,
            expert_group_topk=group_topk,
            norm_topk_prob=bool(text.get("norm_topk_prob", True)),
            routed_scaling_factor=float(text.get("routed_scaling_factor", 1.0)),
            expert_activation=expert_activation,
            num_nextn_predict_layers=predictor_layers,
            mtp_mixer_types=mtp_mixer_types,
            rescale_prenorm_residual=bool(text.get("rescale_prenorm_residual", False)),
            residual_in_fp32=bool(text.get("residual_in_fp32", False)),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            eos_token_ids=_eos_ids(text.get("eos_token_id", outer.get("eos_token_id"))),
            image_token_id=_optional_int(outer.get("img_context_token_id")),
            video_token_id=_optional_int(outer.get("video_context_token_id")),
            vision=vision,
            projector=projector,
        )


def parse_nemotron_h_config(config: Mapping[str, object]) -> NemotronHConfig:
    return NemotronHConfig.from_hf_config(config)


__all__ = [
    "LEGACY_MIXER_NAMES",
    "MIXER_TYPES",
    "NemotronHConfig",
    "OMNI_MODEL_TYPES",
    "RadioProjectorConfig",
    "RadioVisionConfig",
    "TEXT_MODEL_TYPES",
    "layer_mixers_from_config",
    "mtp_mixers_from_config",
    "normalize_mixer_types",
    "parse_nemotron_h_config",
]
