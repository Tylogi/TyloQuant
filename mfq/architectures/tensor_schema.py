"""MFQ canonical tensor names and architecture graph contracts.

Checkpoint names are an import concern.  New MFQ artifacts use the names in
this module regardless of which framework/checkpoint spelling supplied the
weights.  Runtime code should depend on these names (and ``model_graph.json``),
never on Hugging Face paths.

Schema v1 normalizes every supported checkpoint family at the import boundary.
The registry is deliberately architecture keyed, while the emitted vocabulary
is component/operation keyed: source spellings such as ``llm``, ``vpm``,
``layers`` and ``mtp`` never become new MFQ record namespaces.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from mfq.architectures.hf_config import text_section
from mfq.architectures.nemotron_h import (
    NemotronHConfig,
    layer_mixers_from_config,
    mtp_mixers_from_config,
)

if TYPE_CHECKING:
    from mfq.formats.assets import RuntimeAsset

CANONICAL_TENSOR_SCHEMA_VERSION = 1
CANONICAL_TENSOR_NAMESPACE = "mfq.tensor"
GRID_VISION_INPUT_CONTRACT = "grid_vision.v1"
GRID_MROPE_POSITION_POLICY = "grid_mrope"
# Cropped-position-embedding vision towers (RADIO) take normalized row-major
# patches and reserve one template token per pixel-shuffled patch group.
RADIO_CPE_INPUT_CONTRACT = "radio_cpe.v1"


class TensorComponent(str, Enum):
    MODEL = "model"
    VISION = "vision"
    PREDICTOR = "predictor"
    AUDIO = "audio"
    TTS = "tts"
    # Virtual orchestration component. It owns no checkpoint tensor and is
    # inserted declaratively when its required weighted components exist.
    RUNTIME = "runtime"


@dataclass(frozen=True)
class TensorNameMapping:
    """One source tensor resolved at the import boundary.

    ``recipe_name`` is the optional llama.cpp/GGUF alias used only to borrow a
    quantization recipe.  It is not an MFQ on-disk name.
    """

    canonical_name: str
    component: TensorComponent
    recipe_name: str | None = None


@dataclass(frozen=True)
class GraphTopology:
    text_layers: int
    vision_layers: int = 0
    predictor_layers: int = 0


@dataclass(frozen=True)
class ModelGraphSpec:
    architecture: str
    backbone: str
    topology: GraphTopology
    components: tuple[TensorComponent, ...]
    component_specs: tuple[GraphComponentSpec, ...] = ()

    def as_dict(self) -> dict[str, object]:
        present = set(self.components)
        ordered = tuple(item for item in TensorComponent if item in present)
        component_contracts: dict[TensorComponent, dict[str, object]] = {
            TensorComponent.MODEL: {
                "kind": "text",
                "tensor_root": "model",
                "implementation": self.backbone,
                "policy": "decoder",
            },
            TensorComponent.VISION: {
                "kind": "vision",
                "tensor_root": "vision",
                "implementation": "grid_vit",
                "policy": "optional",
                "input_contract": GRID_VISION_INPUT_CONTRACT,
                "position_policy": GRID_MROPE_POSITION_POLICY,
            },
            TensorComponent.PREDICTOR: {
                "kind": "predictor",
                "tensor_root": "predictor",
                "implementation": "next_token_prediction",
                "policy": "optional",
            },
            TensorComponent.AUDIO: {
                "kind": "audio_input",
                "tensor_root": "audio",
                "implementation": "audio_encoder",
                "policy": "optional",
            },
            TensorComponent.TTS: {
                "kind": "audio_output",
                "tensor_root": "tts",
                "implementation": "tts_decoder",
                "policy": "optional",
            },
            TensorComponent.RUNTIME: {
                "kind": "runtime",
                "tensor_root": "runtime",
                "implementation": "runtime",
                "policy": "optional",
            },
        }
        component_capabilities: dict[TensorComponent, tuple[str, ...]] = {
            TensorComponent.MODEL: ("text",),
            TensorComponent.VISION: ("vision",),
            TensorComponent.PREDICTOR: ("speculative_prediction",),
            TensorComponent.AUDIO: ("audio_input",),
            TensorComponent.TTS: ("audio_output",),
            TensorComponent.RUNTIME: (),
        }
        for spec in self.component_specs:
            contract: dict[str, object] = {
                "kind": spec.kind,
                "tensor_root": spec.component.value,
                "implementation": spec.implementation,
                "policy": spec.policy,
            }
            if spec.input_contract:
                contract["input_contract"] = spec.input_contract
            if spec.position_policy:
                contract["position_policy"] = spec.position_policy
            component_contracts[spec.component] = contract
            component_capabilities[spec.component] = spec.capabilities
        return {
            "schema_version": CANONICAL_TENSOR_SCHEMA_VERSION,
            "architecture": self.architecture,
            "graph": {
                "kind": "causal_lm",
                "backbone": self.backbone,
            },
            "topology": {
                "text_layers": self.topology.text_layers,
                "vision_layers": self.topology.vision_layers,
                "predictor_layers": self.topology.predictor_layers,
            },
            "optional_components": {
                "vision": TensorComponent.VISION in ordered,
                "predictor": TensorComponent.PREDICTOR in ordered,
                "audio_input": TensorComponent.AUDIO in ordered,
                "audio_output": TensorComponent.TTS in ordered,
                "duplex": TensorComponent.RUNTIME in ordered,
            },
            "capabilities": [
                capability
                for component in ordered
                for capability in component_capabilities[component]
            ],
            "components": [component_contracts[item] for item in ordered],
            "canonical_naming": {
                "namespace": CANONICAL_TENSOR_NAMESPACE,
                "version": CANONICAL_TENSOR_SCHEMA_VERSION,
                "component_roots": [item.value for item in ordered],
            },
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            self.as_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")


SourceMapper = Callable[
    [str, GraphTopology, Mapping[str, object]],
    TensorNameMapping | None,
]
SourceShapeValidator = Callable[
    [Mapping[str, object], Mapping[str, tuple[int, ...]]],
    None,
]
RecipeNameMapper = Callable[[str], str | None]
SourceAssetBuilder = Callable[
    [Path, Mapping[str, object]],
    Sequence["RuntimeAsset"],
]


@dataclass(frozen=True)
class GraphComponentSpec:
    """Declarative backend component contract for one canonical root."""

    component: TensorComponent
    kind: str
    implementation: str
    policy: str = "optional"
    input_contract: str = ""
    position_policy: str = ""
    capabilities: tuple[str, ...] = ()
    requires: tuple[TensorComponent, ...] = ()


@dataclass(frozen=True)
class TensorSchemaRegistration:
    architecture: str
    aliases: tuple[str, ...]
    source_mapper: SourceMapper
    backbone: str = ""
    component_specs: tuple[GraphComponentSpec, ...] = ()
    source_asset_builder: SourceAssetBuilder | None = None
    # Routed-expert projections this family stores one expert at a time.  The
    # shared importer banks exactly these projections and derives the bank
    # geometry from ``up``, so a non-gated family needs no importer branch.
    expert_projections: tuple[str, ...] = ("gate", "up", "down")
    # Canonical suffixes that must keep float32 storage on import (recurrent
    # state, convolution and router-bias parameters).
    float32_weight_roles: tuple[str, ...] = ()
    # Optional import-boundary check between the declared geometry and the
    # checkpoint payload, keyed by source tensor name.
    source_shape_validator: SourceShapeValidator | None = None
    # Canonical name -> llama.cpp/GGUF recipe alias.  A recipe file is a GGUF
    # artifact, so its tensor spellings are architecture specific; families
    # register their own mapper instead of the recipe consumer testing a model
    # name.
    recipe_name_mapper: RecipeNameMapper | None = None
    # Families whose routed-expert banks follow the recipe's routed alias
    # (``ffn_*_exps``) unless calibration selects another precision.
    recipe_expert_defaults: bool = False

    def __post_init__(self) -> None:
        if not self.backbone:
            object.__setattr__(self, "backbone", self.architecture)


_SCHEMAS: dict[str, TensorSchemaRegistration] = {}


def register_tensor_schema(registration: TensorSchemaRegistration) -> None:
    """Register one import mapping without changing the canonical vocabulary."""

    keys = (registration.architecture, *registration.aliases)
    for key in keys:
        normalized = key.strip().lower()
        previous = _SCHEMAS.get(normalized)
        if previous is not None and previous != registration:
            raise ValueError(f"tensor schema alias is already registered: {key}")
        _SCHEMAS[normalized] = registration


def tensor_schema_for_config(
    config: Mapping[str, object],
) -> TensorSchemaRegistration | None:
    text = text_section(config)
    for value in (text.get("model_type"), config.get("model_type")):
        registration = _SCHEMAS.get(str(value or "").lower())
        if registration is not None:
            return registration
    return None


def topology_from_config(config: Mapping[str, object]) -> GraphTopology:
    text = text_section(config)
    raw_vision = config.get("vision_config", {})
    vision = raw_vision if isinstance(raw_vision, Mapping) else {}
    text_layers = int(text.get("num_hidden_layers", 0) or 0)
    # Hybrid schedules declare one mixer per layer and no layer count; the
    # block list is authoritative there (as in transformers) because the
    # stored ``num_hidden_layers`` is a deprecated, possibly stale echo.
    block_types = text.get("layers_block_type")
    if isinstance(block_types, Sequence) and not isinstance(block_types, (str, bytes)) and block_types:
        text_layers = len(block_types)
    predictor_layers = int(
        text.get(
            "mtp_num_hidden_layers",
            text.get("num_nextn_predict_layers", 0),
        )
        or 0
    )
    # A predictor body may itself be a block list (Nemotron-H attention + moe).
    mtp_types = text.get("mtp_layers_block_type")
    if (
        predictor_layers > 0
        and isinstance(mtp_types, Sequence)
        and not isinstance(mtp_types, (str, bytes))
        and mtp_types
    ):
        predictor_layers = len(mtp_types)
    model_type = str(text.get("model_type", config.get("model_type", ""))).lower()
    if model_type in {"deepseek_v4", "deepseek_v41_text"}:
        target_layers = text.get("dspark_target_layer_ids", ())
        compress_ratios = text.get("compress_ratios", ())
        structural = [int(text.get("n_mtp_layers", 0) or 0)]
        if isinstance(target_layers, Sequence) and not isinstance(
            target_layers, (str, bytes)
        ):
            structural.append(len(target_layers))
        if isinstance(compress_ratios, Sequence) and not isinstance(
            compress_ratios, (str, bytes)
        ):
            structural.append(max(0, len(compress_ratios) - text_layers))
        predictor_layers = max(predictor_layers, *structural)
    return GraphTopology(
        text_layers=text_layers,
        vision_layers=int(
            vision.get(
                "depth",
                vision.get(
                    "num_hidden_layers",
                    text.get("vision_n_layers", config.get("vision_n_layers", 0)),
                ),
            )
            or 0
        ),
        predictor_layers=predictor_layers,
    )


def graph_spec_for_plan(
    config: Mapping[str, object],
    canonical_names: Sequence[str],
) -> ModelGraphSpec | None:
    registration = tensor_schema_for_config(config)
    if registration is None:
        return None
    roots = {name.partition(".")[0] for name in canonical_names}
    selected = [component for component in TensorComponent if component.value in roots]
    for spec in registration.component_specs:
        if (
            spec.component not in selected
            and spec.requires
            and all(required in selected for required in spec.requires)
        ):
            selected.append(spec.component)
    components = tuple(component for component in TensorComponent if component in selected)
    if TensorComponent.MODEL not in components:
        components = (TensorComponent.MODEL, *components)
    topology = topology_from_config(config)
    topology = GraphTopology(
        text_layers=topology.text_layers,
        vision_layers=(
            max(1, topology.vision_layers)
            if TensorComponent.VISION in components
            else 0
        ),
        predictor_layers=(
            max(1, topology.predictor_layers)
            if TensorComponent.PREDICTOR in components
            else 0
        ),
    )
    return ModelGraphSpec(
        architecture=registration.architecture,
        backbone=registration.backbone,
        topology=topology,
        components=components,
        component_specs=registration.component_specs,
    )


def graph_spec_for_source_names(
    config: Mapping[str, object],
    source_names: Sequence[str],
) -> ModelGraphSpec | None:
    """Build a runtime graph directly from one registered source checkpoint.

    Direct loading is a container capability, not a second architecture
    capability.  The same source-name mapping used by import/quantization is
    therefore the sole authority for normalizing an HF directory before the
    runtime graph is selected.
    """

    if tensor_schema_for_config(config) is None:
        return None
    canonical_names = []
    for name in source_names:
        mapped = map_source_tensor_name(name, config)
        canonical_names.append(
            mapped.canonical_name if mapped is not None else name
        )
    return graph_spec_for_plan(config, canonical_names)


def source_runtime_assets(
    root: str | Path,
    config: Mapping[str, object],
) -> tuple["RuntimeAsset", ...]:
    """Build optional source-derived assets through architecture registration.

    The storage/direct-load path only invokes this hook. Architectures that
    need no generated constants gain direct loading from their existing
    canonical schema with no additional loader code.
    """

    registration = tensor_schema_for_config(config)
    if registration is None or registration.source_asset_builder is None:
        return ()
    return tuple(registration.source_asset_builder(Path(root), config))


_SCALE_COMPANION_SUFFIXES = (
    (".weight_scale_inv", ".weight_scale"),
    (".weight_scale_2", ".weight_scale_2"),
    (".weight_scale", ".weight_scale"),
    (".scale", ".weight_scale"),
)


def _canonical_scale_companion(
    source_name: str,
    resolved_sources: Mapping[str, TensorNameMapping | None],
) -> str | None:
    """Bind source scale metadata to an already mapped sibling weight."""

    for source_suffix, canonical_suffix in _SCALE_COMPANION_SUFFIXES:
        if not source_name.endswith(source_suffix):
            continue
        weight_source = source_name.removesuffix(source_suffix) + ".weight"
        weight = resolved_sources.get(weight_source)
        if weight is None or not weight.canonical_name.endswith(".weight"):
            return None
        return weight.canonical_name.removesuffix(".weight") + canonical_suffix
    return None


def canonical_source_tensor_map(
    config: Mapping[str, object],
    source_names: Sequence[str],
) -> dict[str, str]:
    """Return the canonical container view of a source checkpoint.

    This is the shared direct-load/import boundary. Architecture registration
    supplies tensor semantics once; every storage backend receives the same
    canonical-to-source map without adding an architecture-specific loader.
    A source tensor that is already canonical wins over any compatibility
    spelling for the same tensor. Scale companions inherit the canonical name
    of a mapped sibling weight, so explicit maps remain complete and closed.
    """

    ordered_sources = sorted(set(source_names))
    resolved_sources = {
        source_name: map_source_tensor_name(source_name, config)
        for source_name in ordered_sources
    }
    mapped: list[tuple[str, str]] = []
    for source_name in ordered_sources:
        resolved = resolved_sources[source_name]
        if resolved is not None:
            mapped.append((resolved.canonical_name, source_name))
            continue
        companion = _canonical_scale_companion(source_name, resolved_sources)
        if companion is not None:
            mapped.append((companion, source_name))

    identities = {
        canonical: source
        for canonical, source in mapped
        if canonical == source
    }
    result = dict(identities)
    for canonical, source in mapped:
        if canonical in identities or canonical == source:
            continue
        previous = result.setdefault(canonical, source)
        if previous != source:
            raise ValueError(
                "source tensor names collide after canonicalization: "
                f"{previous!r}, {source!r} -> {canonical!r}"
            )
    return result


def map_source_tensor_name(
    source_name: str,
    config: Mapping[str, object],
    *,
    require_registered: bool = False,
) -> TensorNameMapping | None:
    registration = tensor_schema_for_config(config)
    if registration is None:
        if require_registered:
            raise ValueError("model architecture has no canonical tensor schema")
        return None
    mapped = registration.source_mapper(
        source_name,
        topology_from_config(config),
        config,
    )
    if mapped is None and require_registered:
        raise KeyError(
            f"{registration.architecture} source tensor has no canonical mapping: {source_name}"
        )
    if mapped is not None and mapped.recipe_name is None:
        mapped = _with_registered_recipe_name(registration, mapped)
    return mapped


def _with_registered_recipe_name(
    registration: TensorSchemaRegistration,
    mapped: TensorNameMapping,
) -> TensorNameMapping:
    """Fill a missing recipe alias from the registered GGUF spelling table."""

    if registration.recipe_name_mapper is None:
        return mapped
    recipe_name = registration.recipe_name_mapper(mapped.canonical_name)
    if recipe_name is None:
        return mapped
    return TensorNameMapping(mapped.canonical_name, mapped.component, recipe_name)


def recipe_name_mapper_for_config(
    config: Mapping[str, object],
) -> RecipeNameMapper | None:
    """Return the registered GGUF recipe-name mapper for a model config."""

    registration = tensor_schema_for_config(config)
    return None if registration is None else registration.recipe_name_mapper


def canonical_recipe_name(
    canonical_name: str,
    config: Mapping[str, object],
) -> str | None:
    """Resolve one canonical tensor to the registered GGUF recipe alias."""

    mapper = recipe_name_mapper_for_config(config)
    return None if mapper is None else mapper(canonical_name)


def model_block(layer: int, suffix: str) -> str:
    return f"model.block.{int(layer)}.{suffix}"


def vision_block(layer: int, suffix: str) -> str:
    return f"vision.block.{int(layer)}.{suffix}"


def predictor_block(layer: int, suffix: str) -> str:
    return f"predictor.block.{int(layer)}.{suffix}"


_TEXT_SUFFIXES: dict[str, tuple[str, str | None]] = {
    "input_layernorm.weight": ("attention.norm.weight", "attn_norm.weight"),
    "post_attention_layernorm.weight": ("mlp.norm.weight", "post_attention_norm.weight"),
    "self_attn.q_proj.weight": ("attention.query.weight", "attn_q.weight"),
    "self_attn.k_proj.weight": ("attention.key.weight", "attn_k.weight"),
    "self_attn.v_proj.weight": ("attention.value.weight", "attn_v.weight"),
    "self_attn.o_proj.weight": ("attention.output.weight", "attn_output.weight"),
    "self_attn.q_norm.weight": ("attention.query_norm.weight", "attn_q_norm.weight"),
    "self_attn.k_norm.weight": ("attention.key_norm.weight", "attn_k_norm.weight"),
    "mlp.down_proj.weight": ("mlp.down.weight", "ffn_down.weight"),
    "mlp.gate_proj.weight": ("mlp.gate.weight", "ffn_gate.weight"),
    "mlp.up_proj.weight": ("mlp.up.weight", "ffn_up.weight"),
    "mlp.experts.down_proj": ("mlp.experts.down.weight", "ffn_down_exps.weight"),
    "mlp.experts.down_proj.weight": ("mlp.experts.down.weight", "ffn_down_exps.weight"),
    "mlp.experts.gate_up_proj": ("mlp.experts.gate_up.weight", "ffn_gate_up_exps.weight"),
    "mlp.experts.gate_up_proj.weight": ("mlp.experts.gate_up.weight", "ffn_gate_up_exps.weight"),
    "mlp.gate.weight": ("mlp.router.weight", "ffn_gate_inp.weight"),
    "mlp.shared_expert.down_proj.weight": (
        "mlp.shared_expert.down.weight",
        "ffn_down_shexp.weight",
    ),
    "mlp.shared_expert.gate_proj.weight": (
        "mlp.shared_expert.gate.weight",
        "ffn_gate_shexp.weight",
    ),
    "mlp.shared_expert.up_proj.weight": ("mlp.shared_expert.up.weight", "ffn_up_shexp.weight"),
    "mlp.shared_expert_gate.weight": (
        "mlp.shared_expert.router.weight",
        "ffn_gate_inp_shexp.weight",
    ),
    "linear_attn.in_proj_qkv.weight": ("linear_attention.qkv.weight", "attn_qkv.weight"),
    "linear_attn.in_proj_qk.weight": ("linear_attention.qk.weight", None),
    "linear_attn.in_proj_v.weight": ("linear_attention.value.weight", None),
    "linear_attn.in_proj_z.weight": ("linear_attention.gate.weight", "attn_gate.weight"),
    "linear_attn.in_proj_a.weight": ("linear_attention.alpha.weight", "ssm_alpha.weight"),
    "linear_attn.in_proj_b.weight": ("linear_attention.beta.weight", "ssm_beta.weight"),
    "linear_attn.conv1d.weight": ("linear_attention.conv.weight", "ssm_conv1d.weight"),
    "linear_attn.conv1d.bias": ("linear_attention.conv.bias", None),
    "linear_attn.dt_bias": ("linear_attention.dt_bias", "ssm_dt.bias"),
    "linear_attn.A_log": ("linear_attention.a", "ssm_a"),
    "linear_attn.norm.weight": ("linear_attention.norm.weight", "ssm_norm.weight"),
    "linear_attn.out_proj.weight": ("linear_attention.output.weight", "ssm_out.weight"),
}

_VISION_SUFFIXES: dict[str, str] = {
    "patch_embed.proj.weight": "patch_embedding.weight",
    "patch_embed.proj.bias": "patch_embedding.bias",
    "pos_embed.weight": "position_embedding.weight",
    "merger.norm.weight": "merger.norm.weight",
    "merger.norm.bias": "merger.norm.bias",
    "merger.linear_fc1.weight": "merger.mlp.up.weight",
    "merger.linear_fc1.bias": "merger.mlp.up.bias",
    "merger.linear_fc2.weight": "merger.mlp.down.weight",
    "merger.linear_fc2.bias": "merger.mlp.down.bias",
}

_VISION_BLOCK_SUFFIXES: dict[str, str] = {
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "attn.qkv.weight": "attention.qkv.weight",
    "attn.qkv.bias": "attention.qkv.bias",
    "attn.proj.weight": "attention.output.weight",
    "attn.proj.bias": "attention.output.bias",
    "attn.q_norm.weight": "attention.query_norm.weight",
    "attn.k_norm.weight": "attention.key_norm.weight",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
    "mlp.linear_fc1.weight": "mlp.up.weight",
    "mlp.linear_fc1.bias": "mlp.up.bias",
    "mlp.linear_fc2.weight": "mlp.down.weight",
    "mlp.linear_fc2.bias": "mlp.down.bias",
}

_TEXT_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_VISION_BLOCK_RE = re.compile(r"^model\.visual\.blocks\.(\d+)\.(.+)$")
_PREDICTOR_LAYER_RE = re.compile(r"^mtp\.layers\.(\d+)\.(.+)$")


def _qwen35_source_mapper(
    source_name: str,
    topology: GraphTopology,
    _config: Mapping[str, object],
) -> TensorNameMapping | None:
    roots = {
        "model.language_model.embed_tokens.weight": (
            "model.token_embedding.weight",
            "token_embd.weight",
        ),
        "model.language_model.norm.weight": (
            "model.output_norm.weight",
            "output_norm.weight",
        ),
        "lm_head.weight": ("model.output.weight", "output.weight"),
        "model.language_model.position_ids": ("model.position_ids", None),
    }
    root = roots.get(source_name)
    if root is not None:
        return TensorNameMapping(root[0], TensorComponent.MODEL, root[1])

    match = _TEXT_LAYER_RE.match(source_name)
    if match is not None:
        suffix = _TEXT_SUFFIXES.get(match.group(2))
        if suffix is None:
            return None
        return TensorNameMapping(
            model_block(int(match.group(1)), suffix[0]),
            TensorComponent.MODEL,
            f"blk.{match.group(1)}.{suffix[1]}" if suffix[1] else None,
        )

    vision = _VISION_SUFFIXES.get(source_name.removeprefix("model.visual."))
    if vision is not None and source_name.startswith("model.visual."):
        return TensorNameMapping(f"vision.{vision}", TensorComponent.VISION)
    match = _VISION_BLOCK_RE.match(source_name)
    if match is not None:
        suffix = _VISION_BLOCK_SUFFIXES.get(match.group(2))
        if suffix is None:
            return None
        return TensorNameMapping(vision_block(int(match.group(1)), suffix), TensorComponent.VISION)

    predictor_roots = {
        "mtp.fc.weight": "predictor.fusion.weight",
        "mtp.fc_embedding.weight": "predictor.fusion.embedding.weight",
        "mtp.fc_hidden.weight": "predictor.fusion.hidden.weight",
        "mtp.pre_fc_norm_embedding.weight": "predictor.embedding_norm.weight",
        "mtp.pre_fc_norm_hidden.weight": "predictor.hidden_norm.weight",
        "mtp.norm.weight": "predictor.output_norm.weight",
        "mtp.hyper_connection_mixer.hc_norm.weight": "predictor.mhc.pre.norm.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": "predictor.mhc.pre.down.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": "predictor.mhc.pre.up.weight",
    }
    predictor = predictor_roots.get(source_name)
    if predictor is not None:
        gguf_suffix = {
            "mtp.fc.weight": "eh_proj.weight",
            "mtp.pre_fc_norm_embedding.weight": "enorm.weight",
            "mtp.pre_fc_norm_hidden.weight": "hnorm.weight",
            "mtp.norm.weight": "shared_head_norm.weight",
        }.get(source_name)
        return TensorNameMapping(
            predictor,
            TensorComponent.PREDICTOR,
            (
                f"blk.{topology.text_layers}.nextn.{gguf_suffix}"
                if gguf_suffix is not None
                else None
            ),
        )
    match = _PREDICTOR_LAYER_RE.match(source_name)
    if match is not None:
        suffix = _TEXT_SUFFIXES.get(match.group(2))
        if suffix is None:
            return None
        recipe_layer = topology.text_layers + int(match.group(1))
        return TensorNameMapping(
            predictor_block(int(match.group(1)), suffix[0]),
            TensorComponent.PREDICTOR,
            f"blk.{recipe_layer}.{suffix[1]}" if suffix[1] else None,
        )
    if source_name.startswith(("model.block.", "model.runtime.")) or source_name in {
        "model.token_embedding.weight",
        "model.output_norm.weight",
        "model.output.weight",
        "model.position_ids",
    }:
        return TensorNameMapping(source_name, TensorComponent.MODEL)
    for component in (TensorComponent.VISION, TensorComponent.PREDICTOR):
        if source_name.startswith(component.value + "."):
            return TensorNameMapping(source_name, component)
    return None


def _already_canonical(source_name: str) -> TensorNameMapping | None:
    if source_name.startswith(
        (
            "vision.block.",
            "vision.patch_embedding.",
            "vision.position_embedding.",
            "vision.output_norm.",
            "vision.merger.",
            "vision.resampler.",
            "vision.aligner.",
            "vision.special_token.",
            "vision.projector.",
            "vision.cls_register_token.",
            "vision.video_patch_embedding.",
            "vision.input_conditioner.",
            "vision.summary_indices",
        )
    ):
        return TensorNameMapping(source_name, TensorComponent.VISION)
    if source_name.startswith("predictor."):
        return TensorNameMapping(source_name, TensorComponent.PREDICTOR)
    if source_name.startswith(
        (
            "audio.block.",
            "audio.patch_embedding.",
            "audio.position_embedding.",
            "audio.output_norm.",
            "audio.projector.",
        )
    ):
        return TensorNameMapping(source_name, TensorComponent.AUDIO)
    if source_name.startswith(
        (
            "tts.block.",
            "tts.token_embedding.",
            "tts.text_embedding.",
            "tts.code_embedding.",
            "tts.code_output.",
            "tts.output_norm.",
            "tts.semantic_projector.",
            "tts.speaker_projector.",
        )
    ):
        return TensorNameMapping(source_name, TensorComponent.TTS)
    if source_name.startswith("model.block.") or source_name in {
        "model.token_embedding.weight",
        "model.output_norm.weight",
        "model.output.weight",
        "model.position_ids",
        "model.mhc.pre.norm.weight",
        "model.mhc.pre.down.weight",
        "model.mhc.pre.up.weight",
    }:
        return TensorNameMapping(source_name, TensorComponent.MODEL)
    return None


_QWEN4_LAYER_SUFFIXES: dict[str, str] = {
    "attn_hyper_connection.hc_norm.weight": "attention.mhc.pre.norm.weight",
    "attn_hyper_connection.input_mix_weight_down.weight": "attention.mhc.pre.down.weight",
    "attn_hyper_connection.input_mix_weight_up.weight": "attention.mhc.pre.up.weight",
    "attn_hyper_connection.block_inject_weight.weight": "attention.mhc.post.inject.weight",
    "mlp_hyper_connection.hc_norm.weight": "mlp.mhc.pre.norm.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight": "mlp.mhc.pre.down.weight",
    "mlp_hyper_connection.input_mix_weight_up.weight": "mlp.mhc.pre.up.weight",
    "mlp_hyper_connection.block_inject_weight.weight": "mlp.mhc.post.inject.weight",
    "linear_attn.in_proj_qkv.weight": "linear_attention.qkv.weight",
    "linear_attn.in_proj_z.weight": "linear_attention.gate.weight",
    "linear_attn.in_proj_a.weight": "linear_attention.alpha.weight",
    "linear_attn.in_proj_b.weight": "linear_attention.beta.weight",
    "linear_attn.conv1d.weight": "linear_attention.conv.weight",
    "linear_attn.dt_bias": "linear_attention.dt_bias",
    "linear_attn.A_log": "linear_attention.a",
    "linear_attn.norm.weight": "linear_attention.norm.weight",
    "linear_attn.out_proj.weight": "linear_attention.output.weight",
    "self_attn.q_proj.weight": "attention.query.weight",
    "self_attn.k_proj.weight": "attention.key.weight",
    "self_attn.v_proj.weight": "attention.value.weight",
    "self_attn.o_proj.weight": "attention.output.weight",
    "self_attn.q_norm.weight": "attention.query_norm.weight",
    "self_attn.k_norm.weight": "attention.key_norm.weight",
    "self_attn.indexer.index_qk_proj.weight": "attention.indexer.query_key.weight",
    "self_attn.indexer.q_layernorm.weight": "attention.indexer.query_norm.weight",
    "self_attn.indexer.k_layernorm.weight": "attention.indexer.key_norm.weight",
    "mlp.gate.weight": "mlp.router.weight",
    "mlp.experts.gate_up_proj": "mlp.experts.gate_up.weight",
    "mlp.experts.down_proj": "mlp.experts.down.weight",
    "mlp.shared_expert.gate_proj.weight": "mlp.shared_expert.gate.weight",
    "mlp.shared_expert.up_proj.weight": "mlp.shared_expert.up.weight",
    "mlp.shared_expert.down_proj.weight": "mlp.shared_expert.down.weight",
    "mlp.shared_expert_gate.weight": "mlp.shared_expert.router.weight",
    "ple.key_proj.weight": "position_embedding.key.weight",
    "ple.value_proj.weight": "position_embedding.value.weight",
    "ple.norm_key.weight": "position_embedding.key_norm.weight",
    "ple.norm_query.weight": "position_embedding.query_norm.weight",
    "ple.norm_conv.weight": "position_embedding.conv_norm.weight",
    "ple.conv1d.weight": "position_embedding.conv.weight",
    "ple.ple_embedding.layer_multipliers": "position_embedding.ngram.layer_multipliers",
    "ple.ple_embedding.ngram_heads_offsets": "position_embedding.ngram.head_offsets",
    "ple.ple_embedding.ngram_heads_vocab_sizes": "position_embedding.ngram.head_vocab_sizes",
    "ple.ple_embedding.ngram_embedding.weight_scale": "position_embedding.ngram.weight_scale",
}

_QWEN4_EXPERT_RE = re.compile(
    r"^((?:model\.language_model|mtp)\.layers\.(\d+)\.mlp\.experts)\."
    r"(\d+)\.(gate_proj|up_proj|down_proj)\."
    r"(weight|input_scale|weight_scale|weight_scale_2)$"
)
_QWEN4_NGRAM_SHARD_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.ple\.ple_embedding\."
    r"ngram_embedding\.shard_(\d+)\.weight$"
)


def _qwen4_component_block(
    source_prefix: str,
    source_index: int,
    topology: GraphTopology,
) -> tuple[TensorComponent, int]:
    if source_prefix == "mtp" or source_prefix.startswith("mtp."):
        return TensorComponent.PREDICTOR, source_index
    if source_index >= topology.text_layers:
        return TensorComponent.PREDICTOR, source_index - topology.text_layers
    return TensorComponent.MODEL, source_index


def _component_block(component: TensorComponent, layer: int, suffix: str) -> str:
    if component is TensorComponent.PREDICTOR:
        return predictor_block(layer, suffix)
    if component is TensorComponent.VISION:
        return vision_block(layer, suffix)
    return model_block(layer, suffix)


def _map_qwen_vision(source_name: str) -> TensorNameMapping | None:
    vision = _VISION_SUFFIXES.get(source_name.removeprefix("model.visual."))
    if vision is not None and source_name.startswith("model.visual."):
        return TensorNameMapping(f"vision.{vision}", TensorComponent.VISION)
    match = _VISION_BLOCK_RE.match(source_name)
    if match is None:
        return None
    suffix = _VISION_BLOCK_SUFFIXES.get(match.group(2))
    if suffix is None:
        return None
    return TensorNameMapping(
        vision_block(int(match.group(1)), suffix),
        TensorComponent.VISION,
    )


def qwen4_recipe_name(canonical_name: str) -> str | None:
    """Qwen4-Exp GGUF aliases; canonical MFQ storage names stay unchanged."""
    roots = {
        "model.token_embedding.weight": "token_embd.weight",
        "model.output.weight": "output.weight",
        "model.mhc.pre.norm.weight": "output_hc_norm.weight",
        "model.mhc.pre.down.weight": "output_hc_down.weight",
        "model.mhc.pre.up.weight": "output_hc_up.weight",
    }
    if canonical_name in roots:
        return roots[canonical_name]
    match = re.fullmatch(r"model\.block\.(\d+)\.(.+)", canonical_name)
    if match is None:
        return None
    layer, suffix = match.groups()
    if re.fullmatch(r"position_embedding\.ngram\.shard\.\d+\.weight", suffix):
        return "per_layer_token_embd.weight"
    aliases = {
        "linear_attention.qkv.weight": "attn_qkv.weight",
        "linear_attention.gate.weight": "attn_gate.weight",
        "linear_attention.alpha.weight": "ssm_alpha.weight",
        "linear_attention.beta.weight": "ssm_beta.weight",
        "linear_attention.conv.weight": "ssm_conv1d.weight",
        "linear_attention.dt_bias": "ssm_dt.bias",
        "linear_attention.a": "ssm_a",
        "linear_attention.norm.weight": "ssm_norm.weight",
        "linear_attention.output.weight": "ssm_out.weight",
        "attention.query.weight": "attn_q.weight",
        "attention.key.weight": "attn_k.weight",
        "attention.value.weight": "attn_v.weight",
        "attention.output.weight": "attn_output.weight",
        "attention.query_norm.weight": "attn_q_norm.weight",
        "attention.key_norm.weight": "attn_k_norm.weight",
        "attention.indexer.query_key.weight": "indexer.q_proj.weight",
        "attention.indexer.query_norm.weight": "indexer.q_norm.weight",
        "attention.indexer.key_norm.weight": "indexer.k_norm.weight",
        "mlp.router.weight": "ffn_gate_inp.weight",
        "mlp.shared_expert.router.weight": "ffn_gate_inp_shexp.weight",
        "position_embedding.key.weight": "ple_key.weight",
        "position_embedding.value.weight": "ple_value.weight",
        "position_embedding.key_norm.weight": "ple_norm_key.weight",
        "position_embedding.query_norm.weight": "ple_norm_query.weight",
        "position_embedding.conv_norm.weight": "ple_norm_conv.weight",
        "position_embedding.conv.weight": "ple_conv1d.weight",
    }
    for projection in ("gate", "up", "down"):
        aliases[f"mlp.experts.{projection}.weight"] = f"ffn_{projection}_exps.weight"
        aliases[f"mlp.shared_expert.{projection}.weight"] = f"ffn_{projection}_shexp.weight"
    for scope, gguf_scope in (("attention", "attn"), ("mlp", "ffn")):
        for field in ("norm", "down", "up"):
            aliases[f"{scope}.mhc.pre.{field}.weight"] = f"hc_{gguf_scope}_{field}.weight"
        aliases[f"{scope}.mhc.post.inject.weight"] = f"hc_{gguf_scope}_inject.weight"
    alias = aliases.get(suffix)
    return f"blk.{layer}.{alias}" if alias is not None else None


def _qwen4_mapping(name: str, component: TensorComponent) -> TensorNameMapping:
    return TensorNameMapping(name, component, qwen4_recipe_name(name))


def _qwen4_source_mapper(
    source_name: str,
    topology: GraphTopology,
    _config: Mapping[str, object],
) -> TensorNameMapping | None:
    canonical = _already_canonical(source_name)
    if canonical is not None:
        return _qwen4_mapping(canonical.canonical_name, canonical.component)
    roots = {
        "model.language_model.embed_tokens.weight": "model.token_embedding.weight",
        "model.language_model.runtime_hash_metadata": "model.runtime.hash_metadata",
        "lm_head.weight": "model.output.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight": (
            "model.mhc.pre.norm.weight"
        ),
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": (
            "model.mhc.pre.down.weight"
        ),
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": (
            "model.mhc.pre.up.weight"
        ),
        "mtp.pre_fc_norm_embedding.weight": "predictor.embedding_norm.weight",
        "mtp.pre_fc_norm_hidden.weight": "predictor.hidden_norm.weight",
        "mtp.fc_embedding.weight": "predictor.fusion.embedding.weight",
        "mtp.fc_hidden.weight": "predictor.fusion.hidden.weight",
        "mtp.hyper_connection_mixer.hc_norm.weight": "predictor.mhc.pre.norm.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": (
            "predictor.mhc.pre.down.weight"
        ),
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": (
            "predictor.mhc.pre.up.weight"
        ),
    }
    root = roots.get(source_name)
    if root is not None:
        component = (
            TensorComponent.PREDICTOR
            if root.startswith("predictor.")
            else TensorComponent.MODEL
        )
        return _qwen4_mapping(root, component)

    vision = _map_qwen_vision(source_name)
    if vision is not None:
        return vision

    ngram = _QWEN4_NGRAM_SHARD_RE.match(source_name)
    if ngram is not None:
        return _qwen4_mapping(
            model_block(
                int(ngram.group(1)),
                f"position_embedding.ngram.shard.{int(ngram.group(2))}.weight",
            ),
            TensorComponent.MODEL,
        )

    expert = _QWEN4_EXPERT_RE.match(source_name)
    if expert is not None:
        component, layer = _qwen4_component_block(
            expert.group(1), int(expert.group(2)), topology
        )
        projection = {
            "gate_proj": "gate",
            "up_proj": "up",
            "down_proj": "down",
        }[expert.group(4)]
        leaf = expert.group(5)
        return _qwen4_mapping(
            _component_block(
                component,
                layer,
                f"mlp.experts.{int(expert.group(3))}.{projection}.{leaf}",
            ),
            component,
        )

    for pattern, source_prefix in (
        (_TEXT_LAYER_RE, "model.language_model"),
        (_PREDICTOR_LAYER_RE, "mtp"),
    ):
        match = pattern.match(source_name)
        if match is None:
            continue
        suffix = _QWEN4_LAYER_SUFFIXES.get(match.group(2))
        if suffix is None:
            return None
        component, layer = _qwen4_component_block(
            source_prefix, int(match.group(1)), topology
        )
        return _qwen4_mapping(
            _component_block(component, layer, suffix),
            component,
        )
    return None


_GLM5_LAYER_SUFFIXES: dict[str, str] = {
    "input_layernorm.weight": "attention.norm.weight",
    "post_attention_layernorm.weight": "mlp.norm.weight",
    "hc_attn_fn": "attention.mhc.pre.function",
    "hc_attn_base": "attention.mhc.pre.base",
    "hc_attn_scale": "attention.mhc.pre.scale",
    "hc_ffn_fn": "mlp.mhc.pre.function",
    "hc_ffn_base": "mlp.mhc.pre.base",
    "hc_ffn_scale": "mlp.mhc.pre.scale",
    "mlp.gate_proj.weight": "mlp.gate.weight",
    "mlp.up_proj.weight": "mlp.up.weight",
    "mlp.down_proj.weight": "mlp.down.weight",
    "mlp.gate.weight": "mlp.router.weight",
    "mlp.gate.e_score_correction_bias": "mlp.router.bias",
    "mlp.experts.gate_up_proj": "mlp.experts.gate_up.weight",
    "mlp.experts.down_proj": "mlp.experts.down.weight",
    "mlp.shared_experts.gate_proj.weight": "mlp.shared_expert.gate.weight",
    "mlp.shared_experts.up_proj.weight": "mlp.shared_expert.up.weight",
    "mlp.shared_experts.down_proj.weight": "mlp.shared_expert.down.weight",
    "self_attn.q_proj.weight": "linear_attention.query.weight",
    "self_attn.k_proj.weight": "linear_attention.key.weight",
    "self_attn.v_proj.weight": "linear_attention.value.weight",
    "self_attn.q_conv1d.weight": "linear_attention.query_conv.weight",
    "self_attn.k_conv1d.weight": "linear_attention.key_conv.weight",
    "self_attn.v_conv1d.weight": "linear_attention.value_conv.weight",
    "self_attn.f_a_proj.weight": "linear_attention.forget_a.weight",
    "self_attn.f_b_proj.weight": "linear_attention.forget_b.weight",
    "self_attn.g_a_proj.weight": "linear_attention.gate_a.weight",
    "self_attn.g_b_proj.weight": "linear_attention.gate_b.weight",
    "self_attn.b_proj.weight": "linear_attention.beta.weight",
    "self_attn.dt_bias": "linear_attention.dt_bias",
    "self_attn.A_log": "linear_attention.a",
    "self_attn.o_norm.weight": "linear_attention.output_norm.weight",
    "self_attn.o_proj.weight": "attention.output.weight",
    "self_attn.q_a_proj.weight": "attention.query_a.weight",
    "self_attn.q_a_layernorm.weight": "attention.query_a_norm.weight",
    "self_attn.q_b_proj.weight": "attention.query_b.weight",
    "self_attn.kv_a_proj_with_mqa.weight": "attention.key_value_a.weight",
    "self_attn.kv_a_layernorm.weight": "attention.key_value_a_norm.weight",
    "self_attn.kv_b_proj.weight": "attention.key_value_b.source.weight",
    "self_attn.embed_q": "attention.latent.query_embedding.weight",
    "self_attn.unembed_out": "attention.latent.output_unembedding.weight",
    "self_attn.indexer.wq_b.weight": "attention.indexer.query.weight",
    "self_attn.indexer.wk.weight": "attention.indexer.key.weight",
    "self_attn.indexer.weights_proj.weight": "attention.indexer.score.weight",
    "self_attn.indexer.k_norm.weight": "attention.indexer.key_norm.weight",
    "self_attn.indexer.k_norm.bias": "attention.indexer.key_norm.bias",
    "self_attn.indexer.index_kpool_compress_gate": "attention.indexer.pool.gate",
    "self_attn.indexer.index_kpool_compress_ape": "attention.indexer.pool.position",
}

_GLM5_EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\."
    r"(\d+)\.(gate_proj|up_proj|down_proj)\."
    r"(weight|weight_scale|weight_scale_2)$"
)

_GLM5_VISION_ROOTS: dict[str, str] = {
    "patch_embed.proj.weight": "patch_embedding.weight",
    "patch_embed.proj.bias": "patch_embedding.bias",
    "post_layernorm.weight": "output_norm.weight",
    "downsample.weight": "downsample.weight",
    "downsample.bias": "downsample.bias",
    "merger.proj.weight": "merger.projection.weight",
    "merger.post_projection_norm.weight": "merger.norm.weight",
    "merger.post_projection_norm.bias": "merger.norm.bias",
    "merger.gate_proj.weight": "merger.mlp.gate.weight",
    "merger.up_proj.weight": "merger.mlp.up.weight",
    "merger.down_proj.weight": "merger.mlp.down.weight",
}
_GLM5_VISION_BLOCK_SUFFIXES: dict[str, str] = {
    "norm1.weight": "norm1.weight",
    "norm2.weight": "norm2.weight",
    "attn.qkv.weight": "attention.qkv.weight",
    "attn.qkv.bias": "attention.qkv.bias",
    "attn.q_norm.weight": "attention.query_norm.weight",
    "attn.k_norm.weight": "attention.key_norm.weight",
    "attn.proj.weight": "attention.output.weight",
    "attn.proj.bias": "attention.output.bias",
    "mlp.gate_proj.weight": "mlp.gate.weight",
    "mlp.gate_proj.bias": "mlp.gate.bias",
    "mlp.up_proj.weight": "mlp.up.weight",
    "mlp.up_proj.bias": "mlp.up.bias",
    "mlp.down_proj.weight": "mlp.down.weight",
    "mlp.down_proj.bias": "mlp.down.bias",
}


def _glm5_source_mapper(
    source_name: str,
    topology: GraphTopology,
    config: Mapping[str, object],
) -> TensorNameMapping | None:
    canonical = _already_canonical(source_name)
    if canonical is not None:
        return canonical
    roots = {
        "model.language_model.embed_tokens.weight": "model.token_embedding.weight",
        "model.language_model.runtime_hash_metadata": "model.runtime.hash_metadata",
        "model.language_model.norm.weight": "model.output_norm.weight",
        "lm_head.weight": "model.output.weight",
    }
    root = roots.get(source_name)
    if root is not None:
        return TensorNameMapping(root, TensorComponent.MODEL)

    if source_name.startswith("model.visual."):
        relative = source_name.removeprefix("model.visual.")
        vision_root = _GLM5_VISION_ROOTS.get(relative)
        if vision_root is not None:
            return TensorNameMapping(
                f"vision.{vision_root}", TensorComponent.VISION
            )
        match = _VISION_BLOCK_RE.match(source_name)
        if match is not None:
            suffix = _GLM5_VISION_BLOCK_SUFFIXES.get(match.group(2))
            if suffix is not None:
                return TensorNameMapping(
                    vision_block(int(match.group(1)), suffix),
                    TensorComponent.VISION,
                )

    expert = _GLM5_EXPERT_RE.match(source_name)
    if expert is not None:
        source_layer = int(expert.group(1))
        component = (
            TensorComponent.PREDICTOR
            if source_layer >= topology.text_layers
            else TensorComponent.MODEL
        )
        layer = (
            source_layer - topology.text_layers
            if component is TensorComponent.PREDICTOR
            else source_layer
        )
        projection = {
            "gate_proj": "gate",
            "up_proj": "up",
            "down_proj": "down",
        }[expert.group(3)]
        return TensorNameMapping(
            _component_block(
                component,
                layer,
                f"mlp.experts.{int(expert.group(2))}.{projection}.{expert.group(4)}",
            ),
            component,
        )

    match = _TEXT_LAYER_RE.match(source_name)
    if match is None:
        return None
    source_layer = int(match.group(1))
    component = (
        TensorComponent.PREDICTOR
        if source_layer >= topology.text_layers
        else TensorComponent.MODEL
    )
    layer = (
        source_layer - topology.text_layers
        if component is TensorComponent.PREDICTOR
        else source_layer
    )
    source_suffix = match.group(2)
    if component is TensorComponent.PREDICTOR:
        predictor_roots = {
            "enorm.weight": "predictor.embedding_norm.weight",
            "hnorm.weight": "predictor.hidden_norm.weight",
            "eh_proj.weight": "predictor.fusion.weight",
            "shared_head.norm.weight": "predictor.output_norm.weight",
        }
        predictor_root = predictor_roots.get(source_suffix)
        if predictor_root is not None:
            return TensorNameMapping(predictor_root, component)
    suffix = _GLM5_LAYER_SUFFIXES.get(source_suffix)
    if source_suffix == "self_attn.o_proj.weight":
        text = text_section(config)
        layer_types = text.get("layer_types", ())
        is_linear = (
            component is TensorComponent.MODEL
            and isinstance(layer_types, Sequence)
            and not isinstance(layer_types, (str, bytes))
            and layer < len(layer_types)
            and str(layer_types[layer]) == "linear_attention"
        )
        suffix = (
            "linear_attention.output.weight"
            if is_linear
            else "attention.output.weight"
        )
    if suffix is None:
        return None
    return TensorNameMapping(
        _component_block(component, layer, suffix),
        component,
    )


_MINICPM_TEXT_LAYER_RE = re.compile(r"^llm\.model\.layers\.(\d+)\.(.+)$")
_MINICPM_VISION_LAYER_RE = re.compile(r"^vpm\.encoder\.layers\.(\d+)\.(.+)$")
_MINICPM_AUDIO_LAYER_RE = re.compile(r"^apm\.layers\.(\d+)\.(.+)$")
_MINICPM_TTS_LAYER_RE = re.compile(r"^tts\.model\.layers\.(\d+)\.(.+)$")

_MINICPM_VISION_LAYER_SUFFIXES: dict[str, str] = {
    "layer_norm1.weight": "norm1.weight",
    "layer_norm1.bias": "norm1.bias",
    "layer_norm2.weight": "norm2.weight",
    "layer_norm2.bias": "norm2.bias",
    "self_attn.q_proj.weight": "attention.query.weight",
    "self_attn.q_proj.bias": "attention.query.bias",
    "self_attn.k_proj.weight": "attention.key.weight",
    "self_attn.k_proj.bias": "attention.key.bias",
    "self_attn.v_proj.weight": "attention.value.weight",
    "self_attn.v_proj.bias": "attention.value.bias",
    "self_attn.out_proj.weight": "attention.output.weight",
    "self_attn.out_proj.bias": "attention.output.bias",
    "mlp.fc1.weight": "mlp.up.weight",
    "mlp.fc1.bias": "mlp.up.bias",
    "mlp.fc2.weight": "mlp.down.weight",
    "mlp.fc2.bias": "mlp.down.bias",
}

_MINICPM_AUDIO_LAYER_SUFFIXES: dict[str, str] = {
    "self_attn_layer_norm.weight": "attention.norm.weight",
    "self_attn_layer_norm.bias": "attention.norm.bias",
    "self_attn.q_proj.weight": "attention.query.weight",
    "self_attn.q_proj.bias": "attention.query.bias",
    "self_attn.k_proj.weight": "attention.key.weight",
    "self_attn.k_proj.bias": "attention.key.bias",
    "self_attn.v_proj.weight": "attention.value.weight",
    "self_attn.v_proj.bias": "attention.value.bias",
    "self_attn.out_proj.weight": "attention.output.weight",
    "self_attn.out_proj.bias": "attention.output.bias",
    "final_layer_norm.weight": "mlp.norm.weight",
    "final_layer_norm.bias": "mlp.norm.bias",
    "fc1.weight": "mlp.up.weight",
    "fc1.bias": "mlp.up.bias",
    "fc2.weight": "mlp.down.weight",
    "fc2.bias": "mlp.down.bias",
}

_MINICPM_TTS_LAYER_SUFFIXES: dict[str, str] = {
    "input_layernorm.weight": "attention.norm.weight",
    "post_attention_layernorm.weight": "mlp.norm.weight",
    "self_attn.q_proj.weight": "attention.query.weight",
    "self_attn.k_proj.weight": "attention.key.weight",
    "self_attn.v_proj.weight": "attention.value.weight",
    "self_attn.o_proj.weight": "attention.output.weight",
    "mlp.gate_proj.weight": "mlp.gate.weight",
    "mlp.up_proj.weight": "mlp.up.weight",
    "mlp.down_proj.weight": "mlp.down.weight",
}


def _minicpmo_source_mapper(
    source_name: str,
    _topology: GraphTopology,
    _config: Mapping[str, object],
) -> TensorNameMapping | None:
    canonical = _already_canonical(source_name)
    if canonical is not None:
        return canonical

    roots: dict[str, tuple[str, TensorComponent, str | None]] = {
        "llm.model.embed_tokens.weight": (
            "model.token_embedding.weight",
            TensorComponent.MODEL,
            "token_embd.weight",
        ),
        "llm.model.norm.weight": (
            "model.output_norm.weight",
            TensorComponent.MODEL,
            "output_norm.weight",
        ),
        "llm.lm_head.weight": (
            "model.output.weight",
            TensorComponent.MODEL,
            "output.weight",
        ),
        "vpm.embeddings.patch_embedding.weight": (
            "vision.patch_embedding.weight",
            TensorComponent.VISION,
            None,
        ),
        "vpm.embeddings.patch_embedding.bias": (
            "vision.patch_embedding.bias",
            TensorComponent.VISION,
            None,
        ),
        "vpm.embeddings.position_embedding.weight": (
            "vision.position_embedding.weight",
            TensorComponent.VISION,
            None,
        ),
        "vpm.post_layernorm.weight": (
            "vision.output_norm.weight",
            TensorComponent.VISION,
            None,
        ),
        "vpm.post_layernorm.bias": (
            "vision.output_norm.bias",
            TensorComponent.VISION,
            None,
        ),
        "resampler.query": ("vision.resampler.query", TensorComponent.VISION, None),
        "resampler.proj": (
            "vision.resampler.output.weight",
            TensorComponent.VISION,
            None,
        ),
        "resampler.kv_proj.weight": (
            "vision.resampler.key_value.weight",
            TensorComponent.VISION,
            None,
        ),
        "resampler.ln_q.weight": (
            "vision.resampler.query_norm.weight",
            TensorComponent.VISION,
            None,
        ),
        "resampler.ln_q.bias": (
            "vision.resampler.query_norm.bias",
            TensorComponent.VISION,
            None,
        ),
        "resampler.ln_kv.weight": (
            "vision.resampler.key_value_norm.weight",
            TensorComponent.VISION,
            None,
        ),
        "resampler.ln_kv.bias": (
            "vision.resampler.key_value_norm.bias",
            TensorComponent.VISION,
            None,
        ),
        "resampler.ln_post.weight": (
            "vision.resampler.output_norm.weight",
            TensorComponent.VISION,
            None,
        ),
        "resampler.ln_post.bias": (
            "vision.resampler.output_norm.bias",
            TensorComponent.VISION,
            None,
        ),
        "resampler.attn.in_proj_weight": (
            "vision.resampler.attention.qkv.weight",
            TensorComponent.VISION,
            None,
        ),
        "resampler.attn.in_proj_bias": (
            "vision.resampler.attention.qkv.bias",
            TensorComponent.VISION,
            None,
        ),
        "resampler.attn.out_proj.weight": (
            "vision.resampler.attention.output.weight",
            TensorComponent.VISION,
            None,
        ),
        "resampler.attn.out_proj.bias": (
            "vision.resampler.attention.output.bias",
            TensorComponent.VISION,
            None,
        ),
        "apm.conv1.weight": (
            "audio.patch_embedding.conv1.weight",
            TensorComponent.AUDIO,
            None,
        ),
        "apm.conv1.bias": (
            "audio.patch_embedding.conv1.bias",
            TensorComponent.AUDIO,
            None,
        ),
        "apm.conv2.weight": (
            "audio.patch_embedding.conv2.weight",
            TensorComponent.AUDIO,
            None,
        ),
        "apm.conv2.bias": (
            "audio.patch_embedding.conv2.bias",
            TensorComponent.AUDIO,
            None,
        ),
        "apm.embed_positions.weight": (
            "audio.position_embedding.weight",
            TensorComponent.AUDIO,
            None,
        ),
        "apm.layer_norm.weight": (
            "audio.output_norm.weight",
            TensorComponent.AUDIO,
            None,
        ),
        "apm.layer_norm.bias": (
            "audio.output_norm.bias",
            TensorComponent.AUDIO,
            None,
        ),
        "audio_projection_layer.linear1.weight": (
            "audio.projector.input.weight",
            TensorComponent.AUDIO,
            None,
        ),
        "audio_projection_layer.linear1.bias": (
            "audio.projector.input.bias",
            TensorComponent.AUDIO,
            None,
        ),
        "audio_projection_layer.linear2.weight": (
            "audio.projector.output.weight",
            TensorComponent.AUDIO,
            None,
        ),
        "audio_projection_layer.linear2.bias": (
            "audio.projector.output.bias",
            TensorComponent.AUDIO,
            None,
        ),
        "tts.emb_text.weight": ("tts.text_embedding.weight", TensorComponent.TTS, None),
        "tts.model.embed_tokens.weight": (
            "tts.token_embedding.weight",
            TensorComponent.TTS,
            None,
        ),
        "tts.model.norm.weight": ("tts.output_norm.weight", TensorComponent.TTS, None),
    }
    root = roots.get(source_name)
    if root is not None:
        return TensorNameMapping(root[0], root[1], root[2])

    match = _MINICPM_TEXT_LAYER_RE.match(source_name)
    if match is not None:
        suffix = _TEXT_SUFFIXES.get(match.group(2))
        if suffix is None:
            return None
        return TensorNameMapping(
            model_block(int(match.group(1)), suffix[0]),
            TensorComponent.MODEL,
            f"blk.{match.group(1)}.{suffix[1]}" if suffix[1] else None,
        )

    match = _MINICPM_VISION_LAYER_RE.match(source_name)
    if match is not None:
        suffix = _MINICPM_VISION_LAYER_SUFFIXES.get(match.group(2))
        return (
            TensorNameMapping(
                vision_block(int(match.group(1)), suffix),
                TensorComponent.VISION,
            )
            if suffix is not None
            else None
        )

    match = _MINICPM_AUDIO_LAYER_RE.match(source_name)
    if match is not None:
        suffix = _MINICPM_AUDIO_LAYER_SUFFIXES.get(match.group(2))
        return (
            TensorNameMapping(
                f"audio.block.{int(match.group(1))}.{suffix}",
                TensorComponent.AUDIO,
            )
            if suffix is not None
            else None
        )

    match = _MINICPM_TTS_LAYER_RE.match(source_name)
    if match is not None:
        suffix = _MINICPM_TTS_LAYER_SUFFIXES.get(match.group(2))
        return (
            TensorNameMapping(
                f"tts.block.{int(match.group(1))}.{suffix}",
                TensorComponent.TTS,
            )
            if suffix is not None
            else None
        )

    match = re.match(r"^tts\.emb_code\.(\d+)\.weight$", source_name)
    if match is not None:
        return TensorNameMapping(
            f"tts.code_embedding.{int(match.group(1))}.weight",
            TensorComponent.TTS,
        )
    match = re.match(
        r"^tts\.head_code\.(\d+)\.parametrizations\.weight\.original([01])$",
        source_name,
    )
    if match is not None:
        leaf = "magnitude" if match.group(2) == "0" else "direction"
        return TensorNameMapping(
            f"tts.code_output.{int(match.group(1))}.weight_norm.{leaf}",
            TensorComponent.TTS,
        )
    match = re.match(
        r"^tts\.projector_(semantic|spk)\.linear([12])\.(weight|bias)$",
        source_name,
    )
    if match is not None:
        projector = "semantic" if match.group(1) == "semantic" else "speaker"
        projection = "input" if match.group(2) == "1" else "output"
        return TensorNameMapping(
            f"tts.{projector}_projector.{projection}.{match.group(3)}",
            TensorComponent.TTS,
        )
    return None


_DEEPSEEK_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
_DEEPSEEK_PREDICTOR_RE = re.compile(r"^mtp\.(\d+)\.(.+)$")
_DEEPSEEK_EXPERT_RE = re.compile(
    r"^ffn\.experts\.(\d+)\.w([123])\.(weight|scale)$"
)

_DEEPSEEK_BLOCK_SUFFIXES: dict[str, str] = {
    "attn_norm.weight": "attention.norm.weight",
    "ffn_norm.weight": "mlp.norm.weight",
    "hc_attn_fn": "attention.mhc.pre.function",
    "hc_attn_base": "attention.mhc.pre.base",
    "hc_attn_scale": "attention.mhc.pre.scale",
    "hc_ffn_fn": "mlp.mhc.pre.function",
    "hc_ffn_base": "mlp.mhc.pre.base",
    "hc_ffn_scale": "mlp.mhc.pre.scale",
    "attn.wq_a.weight": "attention.query_a.weight",
    "attn.wq_a.scale": "attention.query_a.weight_scale",
    "attn.q_norm.weight": "attention.query_a_norm.weight",
    "attn.wq_b.weight": "attention.query_b.weight",
    "attn.wq_b.scale": "attention.query_b.weight_scale",
    "attn.wkv.weight": "attention.key_value_a.weight",
    "attn.wkv.scale": "attention.key_value_a.weight_scale",
    "attn.kv_norm.weight": "attention.key_value_a_norm.weight",
    "attn.attn_sink": "attention.sink",
    "attn.wo_a.weight": "attention.output_a.weight",
    "attn.wo_a.scale": "attention.output_a.weight_scale",
    "attn.wo_b.weight": "attention.output_b.weight",
    "attn.wo_b.scale": "attention.output_b.weight_scale",
    "attn.compressor.wkv.weight": "attention.compressor.key_value.weight",
    "attn.compressor.wgate.weight": "attention.compressor.gate.weight",
    "attn.compressor.ape": "attention.compressor.position",
    "attn.compressor.norm.weight": "attention.compressor.norm.weight",
    "attn.indexer.wq_b.weight": "attention.indexer.query.weight",
    "attn.indexer.wq_b.scale": "attention.indexer.query.weight_scale",
    "attn.indexer.weights_proj.weight": "attention.indexer.score.weight",
    "attn.indexer.compressor.wkv.weight": (
        "attention.indexer.compressor.key_value.weight"
    ),
    "attn.indexer.compressor.wgate.weight": (
        "attention.indexer.compressor.gate.weight"
    ),
    "attn.indexer.compressor.ape": "attention.indexer.compressor.position",
    "attn.indexer.compressor.norm.weight": (
        "attention.indexer.compressor.norm.weight"
    ),
    "ffn.gate.weight": "mlp.router.weight",
    "ffn.gate.bias": "mlp.router.bias",
    "ffn.gate.bias_vl": "mlp.router.vision_bias",
    "ffn.gate.tid2eid": "mlp.router.token_to_expert",
    "ffn.experts.gate_up.weight": "mlp.experts.gate_up.weight",
    "ffn.experts.down.weight": "mlp.experts.down.weight",
}


def _deepseek_block_suffix(source_suffix: str) -> str | None:
    direct = _DEEPSEEK_BLOCK_SUFFIXES.get(source_suffix)
    if direct is not None:
        return direct
    match = _DEEPSEEK_EXPERT_RE.match(source_suffix)
    if match is not None:
        projection = {"1": "gate", "2": "down", "3": "up"}[match.group(2)]
        leaf = "weight" if match.group(3) == "weight" else "weight_scale"
        return f"mlp.experts.{int(match.group(1))}.{projection}.{leaf}"
    match = re.match(r"^ffn\.shared_experts\.w([123])\.(weight|scale)$", source_suffix)
    if match is not None:
        projection = {"1": "gate", "2": "down", "3": "up"}[match.group(1)]
        leaf = "weight" if match.group(2) == "weight" else "weight_scale"
        return f"mlp.shared_expert.{projection}.{leaf}"
    return None


def _map_deepseek_vision(source_name: str) -> TensorNameMapping | None:
    roots = {
        "vision.patch_embed.proj.weight": "vision.patch_embedding.weight",
        "vision.patch_embed.proj.bias": "vision.patch_embedding.bias",
        "vision.norm.weight": "vision.output_norm.weight",
        "aligner.w1.weight": "vision.aligner.input.weight",
        "aligner.w1.bias": "vision.aligner.input.bias",
        "aligner.w2.weight": "vision.aligner.output.weight",
        "aligner.w2.bias": "vision.aligner.output.bias",
        "image_start": "vision.special_token.start",
        "image_pad": "vision.special_token.pad",
        "image_newline": "vision.special_token.newline",
        "image_end": "vision.special_token.end",
    }
    root = roots.get(source_name)
    if root is not None:
        return TensorNameMapping(root, TensorComponent.VISION)
    match = re.match(r"^vision\.blocks\.(\d+)\.(.+)$", source_name)
    if match is None:
        return None
    suffix = {
        "norm1.weight": "norm1.weight",
        "norm2.weight": "norm2.weight",
        "attn.wqkv.weight": "attention.qkv.weight",
        "attn.wqkv.bias": "attention.qkv.bias",
        "attn.wo.weight": "attention.output.weight",
        "attn.wo.bias": "attention.output.bias",
        "mlp.w1.weight": "mlp.gate_up.weight",
        "mlp.w2.weight": "mlp.down.weight",
    }.get(match.group(2))
    return (
        TensorNameMapping(
            vision_block(int(match.group(1)), suffix),
            TensorComponent.VISION,
        )
        if suffix is not None
        else None
    )


_DEEPSEEK_V41_BLOCK_SUFFIXES: dict[str, str] = {
    "attn_norm.weight": "attention.norm.weight",
    "ffn_norm.weight": "mlp.norm.weight",
    "hc_attn_fn": "attention.mhc.pre.function",
    "hc_attn_base": "attention.mhc.pre.base",
    "hc_attn_scale": "attention.mhc.pre.scale",
    "hc_ffn_fn": "mlp.mhc.pre.function",
    "hc_ffn_base": "mlp.mhc.pre.base",
    "hc_ffn_scale": "mlp.mhc.pre.scale",
    "attn.wq_a.weight": "attention.query_a.weight",
    "attn.wq_a.scale": "attention.query_a.weight_scale",
    "attn.q_norm.weight": "attention.query_a_norm.weight",
    "attn.wq_b.weight": "attention.query_b.weight",
    "attn.wq_b.scale": "attention.query_b.weight_scale",
    "attn.wkv.weight": "attention.key_value.weight",
    "attn.wkv.scale": "attention.key_value.weight_scale",
    "attn.kv_norm.weight": "attention.key_value_norm.weight",
    "attn.attn_sink": "attention.sink",
    "attn.wo_a.weight": "attention.output_a.weight",
    "attn.wo_a.scale": "attention.output_a.weight_scale",
    "attn.wo_b.weight": "attention.output_b.weight",
    "attn.wo_b.scale": "attention.output_b.weight_scale",
    "attn.compressor.wkv.weight": "attention.compressor.key_value.weight",
    "attn.compressor.wgate.weight": "attention.compressor.gate.weight",
    "attn.compressor.norm.weight": "attention.compressor.norm.weight",
    "attn.indexer.wq_b.weight": "attention.indexer.query.weight",
    "attn.indexer.wq_b.scale": "attention.indexer.query.weight_scale",
    "attn.indexer.wk.weight": "attention.indexer.key.weight",
    "attn.indexer.k_norm.weight": "attention.indexer.key_norm.weight",
    "attn.indexer.weights_proj.weight": "attention.indexer.score.weight",
    "engram.embed.weight": "associative_memory.embedding.weight",
    "engram.embed.scale": "associative_memory.embedding.weight_scale",
    "engram.q_weight": "associative_memory.query.weight",
    "engram.k_weight": "associative_memory.key.weight",
    "engram.wkv.weight": "associative_memory.projection.weight",
    "engram.wkv.scale": "associative_memory.projection.weight_scale",
    "ffn.gate.weight": "mlp.router.weight",
    "ffn.gate.bias": "mlp.router.bias",
    "ffn.gate.bias_vl": "mlp.router.vision_bias",
}


def _deepseek_v41_block_suffix(source_suffix: str) -> str | None:
    direct = _DEEPSEEK_V41_BLOCK_SUFFIXES.get(source_suffix)
    if direct is not None:
        return direct
    expert = _DEEPSEEK_EXPERT_RE.match(source_suffix)
    if expert is not None:
        projection = {"1": "gate", "2": "down", "3": "up"}[expert.group(2)]
        leaf = "weight" if expert.group(3) == "weight" else "weight_scale"
        return f"mlp.experts.{int(expert.group(1))}.{projection}.{leaf}"
    shared = re.match(r"^ffn\.shared_experts\.w([123])\.(weight|scale)$", source_suffix)
    if shared is not None:
        projection = {"1": "gate", "2": "down", "3": "up"}[shared.group(1)]
        leaf = "weight" if shared.group(2) == "weight" else "weight_scale"
        return f"mlp.shared_expert.{projection}.{leaf}"
    return None


def _map_deepseek_v41_vision(source_name: str) -> TensorNameMapping | None:
    root = {
        "vision.patch_embed.proj.weight": "vision.patch_embedding.weight",
        "vision.patch_embed.proj.bias": "vision.patch_embedding.bias",
        "vision.norm.weight": "vision.output_norm.weight",
        "aligner.w1.weight": "vision.aligner.input.weight",
        "aligner.w1.bias": "vision.aligner.input.bias",
        "aligner.w2.weight": "vision.aligner.output.weight",
        "aligner.w2.bias": "vision.aligner.output.bias",
        "image_start": "vision.special_token.start",
        "image_newline": "vision.special_token.newline",
        "image_end": "vision.special_token.end",
    }.get(source_name)
    if root is not None:
        return TensorNameMapping(root, TensorComponent.VISION)
    match = re.match(r"^vision\.blocks\.(\d+)\.(.+)$", source_name)
    if match is None:
        return None
    suffix = {
        "norm1.weight": "norm1.weight",
        "norm2.weight": "norm2.weight",
        "attn.wqkv.weight": "attention.qkv.weight",
        "attn.wqkv.bias": "attention.qkv.bias",
        "attn.wo.weight": "attention.output.weight",
        "attn.wo.bias": "attention.output.bias",
        "mlp.w1.weight": "mlp.gate_up.weight",
        "mlp.w2.weight": "mlp.down.weight",
    }.get(match.group(2))
    if suffix is None:
        return None
    return TensorNameMapping(
        vision_block(int(match.group(1)), suffix),
        TensorComponent.VISION,
    )


def _deepseek_v41_source_mapper(
    source_name: str,
    _topology: GraphTopology,
    _config: Mapping[str, object],
) -> TensorNameMapping | None:
    canonical = _already_canonical(source_name)
    if canonical is not None:
        return canonical
    root = {
        "embed.weight": "model.token_embedding.weight",
        "norm.weight": "model.output_norm.weight",
        "head.weight": "model.output.weight",
    }.get(source_name)
    if root is not None:
        return TensorNameMapping(root, TensorComponent.MODEL)
    vision = _map_deepseek_v41_vision(source_name)
    if vision is not None:
        return vision

    layer = _DEEPSEEK_LAYER_RE.match(source_name)
    if layer is not None:
        suffix = _deepseek_v41_block_suffix(layer.group(2))
        if suffix is None:
            return None
        return TensorNameMapping(
            model_block(int(layer.group(1)), suffix),
            TensorComponent.MODEL,
        )

    predictor = _DEEPSEEK_PREDICTOR_RE.match(source_name)
    if predictor is None:
        return None
    stage = int(predictor.group(1))
    source_suffix = predictor.group(2)
    suffix = _deepseek_v41_block_suffix(source_suffix)
    if suffix is None:
        suffix = {
            "main_norm.weight": "main_norm.weight",
            "main_proj.weight": "main_projection.weight",
            "main_proj.scale": "main_projection.weight_scale",
            "norm.weight": "output_norm.weight",
            "confidence_head.proj.weight": "confidence.projection.weight",
            "markov_head.embed.weight": "markov.embedding.weight",
            "markov_head.head.weight": "markov.output.weight",
        }.get(source_suffix)
    if suffix is None:
        return None
    return TensorNameMapping(
        f"predictor.stage.{stage}.{suffix}",
        TensorComponent.PREDICTOR,
    )


def _deepseek_v4_source_mapper(
    source_name: str,
    _topology: GraphTopology,
    _config: Mapping[str, object],
) -> TensorNameMapping | None:
    canonical = _already_canonical(source_name)
    if canonical is not None:
        return canonical
    roots = {
        "embed.weight": "model.token_embedding.weight",
        "norm.weight": "model.output_norm.weight",
        "head.weight": "model.output.weight",
        "hc_head_fn": "model.mhc.output.function",
        "hc_head_base": "model.mhc.output.base",
        "hc_head_scale": "model.mhc.output.scale",
    }
    root = roots.get(source_name)
    if root is not None:
        return TensorNameMapping(root, TensorComponent.MODEL)
    vision = _map_deepseek_vision(source_name)
    if vision is not None:
        return vision

    match = _DEEPSEEK_LAYER_RE.match(source_name)
    if match is not None:
        suffix = _deepseek_block_suffix(match.group(2))
        return (
            TensorNameMapping(
                model_block(int(match.group(1)), suffix),
                TensorComponent.MODEL,
            )
            if suffix is not None
            else None
        )

    match = _DEEPSEEK_PREDICTOR_RE.match(source_name)
    if match is None:
        return None
    stage = int(match.group(1))
    source_suffix = match.group(2)
    suffix = _deepseek_block_suffix(source_suffix)
    if suffix is not None:
        return TensorNameMapping(
            f"predictor.stage.{stage}.{suffix}",
            TensorComponent.PREDICTOR,
        )
    predictor_suffixes = {
        "main_norm.weight": "main_norm.weight",
        "main_proj.weight": "main_projection.weight",
        "main_proj.scale": "main_projection.weight_scale",
        "norm.weight": "output_norm.weight",
        "hc_head_fn": "mhc.output.function",
        "hc_head_base": "mhc.output.base",
        "hc_head_scale": "mhc.output.scale",
        "confidence_head.proj.weight": "confidence.projection.weight",
        "markov_head.markov_w1.weight": "markov.input.weight",
        "markov_head.markov_w2.weight": "markov.output.weight",
    }
    suffix = predictor_suffixes.get(source_suffix)
    return (
        TensorNameMapping(
            f"predictor.stage.{stage}.{suffix}",
            TensorComponent.PREDICTOR,
        )
        if suffix is not None
        else None
    )


_GLM_DSA_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_GLM_DSA_EXPERT_RE = re.compile(
    r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
_GLM_DSA_LAYER_SUFFIXES: dict[str, str] = {
    "input_layernorm.weight": "attention.norm.weight",
    "post_attention_layernorm.weight": "mlp.norm.weight",
    "self_attn.q_a_proj.weight": "attention.query_a.weight",
    "self_attn.q_a_layernorm.weight": "attention.query_a_norm.weight",
    "self_attn.q_b_proj.weight": "attention.query_b.weight",
    "self_attn.kv_a_proj_with_mqa.weight": "attention.key_value_a.weight",
    "self_attn.kv_a_layernorm.weight": "attention.key_value_a_norm.weight",
    "self_attn.kv_b_proj.weight": "attention.key_value_b.source.weight",
    "self_attn.embed_q": "attention.latent.query_embedding.weight",
    "self_attn.unembed_out": "attention.latent.output_unembedding.weight",
    "self_attn.o_proj.weight": "attention.output.weight",
    "self_attn.indexer.wq_b.weight": "attention.indexer.query.weight",
    "self_attn.indexer.wk.weight": "attention.indexer.key.weight",
    "self_attn.indexer.weights_proj.weight": "attention.indexer.score.weight",
    "self_attn.indexer.k_norm.weight": "attention.indexer.key_norm.weight",
    "self_attn.indexer.k_norm.bias": "attention.indexer.key_norm.bias",
    "mlp.gate_proj.weight": "mlp.gate.weight",
    "mlp.up_proj.weight": "mlp.up.weight",
    "mlp.down_proj.weight": "mlp.down.weight",
    "mlp.gate.weight": "mlp.router.weight",
    "mlp.gate.e_score_correction_bias": "mlp.router.bias",
    "mlp.experts.gate_up_proj": "mlp.experts.gate_up.weight",
    "mlp.experts.down_proj": "mlp.experts.down.weight",
    "mlp.shared_experts.gate_proj.weight": "mlp.shared_expert.gate.weight",
    "mlp.shared_experts.up_proj.weight": "mlp.shared_expert.up.weight",
    "mlp.shared_experts.down_proj.weight": "mlp.shared_expert.down.weight",
}


def _glm_dsa_source_mapper(
    source_name: str,
    _topology: GraphTopology,
    _config: Mapping[str, object],
) -> TensorNameMapping | None:
    canonical = _already_canonical(source_name)
    if canonical is not None:
        return canonical
    root = {
        "model.embed_tokens.weight": "model.token_embedding.weight",
        "model.norm.weight": "model.output_norm.weight",
        "lm_head.weight": "model.output.weight",
    }.get(source_name)
    if root is not None:
        return TensorNameMapping(root, TensorComponent.MODEL)
    match = _GLM_DSA_LAYER_RE.match(source_name)
    if match is None:
        return None
    source_suffix = match.group(2)
    suffix = _GLM_DSA_LAYER_SUFFIXES.get(source_suffix)
    if suffix is None:
        expert = _GLM_DSA_EXPERT_RE.match(source_suffix)
        if expert is not None:
            projection = {
                "gate_proj": "gate",
                "up_proj": "up",
                "down_proj": "down",
            }[expert.group(2)]
            suffix = f"mlp.experts.{int(expert.group(1))}.{projection}.weight"
    return (
        TensorNameMapping(
            model_block(int(match.group(1)), suffix),
            TensorComponent.MODEL,
        )
        if suffix is not None
        else None
    )


_GEMMA4_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_GEMMA4_LAYER_SUFFIXES: dict[str, tuple[str, str | None]] = {
    "input_layernorm.weight": ("attention.norm.weight", "attn_norm.weight"),
    "post_attention_layernorm.weight": (
        "attention.output_norm.weight",
        "post_attention_norm.weight",
    ),
    "self_attn.q_proj.weight": ("attention.query.weight", "attn_q.weight"),
    "self_attn.k_proj.weight": ("attention.key.weight", "attn_k.weight"),
    "self_attn.v_proj.weight": ("attention.value.weight", "attn_v.weight"),
    "self_attn.o_proj.weight": ("attention.output.weight", "attn_output.weight"),
    "self_attn.q_norm.weight": ("attention.query_norm.weight", "attn_q_norm.weight"),
    "self_attn.k_norm.weight": ("attention.key_norm.weight", "attn_k_norm.weight"),
    "pre_feedforward_layernorm.weight": (
        "mlp.dense.input_norm.weight",
        "ffn_norm.weight",
    ),
    "post_feedforward_layernorm.weight": (
        "mlp.output_norm.weight",
        "post_ffw_norm.weight",
    ),
    "post_feedforward_layernorm_1.weight": (
        "mlp.dense.output_norm.weight",
        "post_ffw_norm_1.weight",
    ),
    "pre_feedforward_layernorm_2.weight": (
        "mlp.experts.input_norm.weight",
        "pre_ffw_norm_2.weight",
    ),
    "post_feedforward_layernorm_2.weight": (
        "mlp.experts.output_norm.weight",
        "post_ffw_norm_2.weight",
    ),
    "layer_scalar": ("output_scale", "layer_output_scale.weight"),
    "mlp.gate_proj.weight": ("mlp.gate.weight", "ffn_gate.weight"),
    "mlp.up_proj.weight": ("mlp.up.weight", "ffn_up.weight"),
    "mlp.down_proj.weight": ("mlp.down.weight", "ffn_down.weight"),
    "experts.gate_up_proj": ("mlp.experts.gate_up.weight", "ffn_gate_up_exps.weight"),
    "experts.gate_up_proj.weight": (
        "mlp.experts.gate_up.weight",
        "ffn_gate_up_exps.weight",
    ),
    "experts.down_proj": ("mlp.experts.down.weight", "ffn_down_exps.weight"),
    "experts.down_proj.weight": ("mlp.experts.down.weight", "ffn_down_exps.weight"),
    "router.proj.weight": ("mlp.router.weight", "ffn_gate_inp.weight"),
    "router.scale": ("mlp.router.norm.weight", "ffn_gate_inp.scale"),
    "router.per_expert_scale": ("mlp.router.expert_scale", "ffn_down_exps.scale"),
}


def _gemma4_source_mapper(
    source_name: str,
    _topology: GraphTopology,
    _config: Mapping[str, object],
) -> TensorNameMapping | None:
    canonical = _already_canonical(source_name)
    if canonical is not None:
        return canonical
    roots = {
        "model.language_model.embed_tokens.weight": (
            "model.token_embedding.weight",
            "token_embd.weight",
        ),
        "model.language_model.norm.weight": (
            "model.output_norm.weight",
            "output_norm.weight",
        ),
        "lm_head.weight": ("model.output.weight", "output.weight"),
    }
    root = roots.get(source_name)
    if root is not None:
        return TensorNameMapping(root[0], TensorComponent.MODEL, root[1])
    match = _GEMMA4_LAYER_RE.match(source_name)
    if match is None:
        return None
    suffix = _GEMMA4_LAYER_SUFFIXES.get(match.group(2))
    if suffix is None:
        return None
    return TensorNameMapping(
        model_block(int(match.group(1)), suffix[0]),
        TensorComponent.MODEL,
        f"blk.{match.group(1)}.{suffix[1]}" if suffix[1] else None,
    )


_NEMOTRON_BACKBONE_LAYER_RE = re.compile(r"^language_model\.backbone\.layers\.(\d+)\.(.+)$")
_NEMOTRON_MTP_LAYER_RE = re.compile(r"^language_model\.mtp\.layers\.(\d+)\.(.+)$")

_NEMOTRON_ROOTS = {
    "language_model.backbone.embeddings.weight": "model.token_embedding.weight",
    "language_model.backbone.norm_f.weight": "model.output_norm.weight",
    "language_model.lm_head.weight": "model.output.weight",
}

# Every Nemotron-H block is one pre-normed mixer, so the block-scoped norm keeps
# the existing pre-mixer role whatever the mixer kind is.
_NEMOTRON_BLOCK_NORM = "attention.norm.weight"

_NEMOTRON_MIXER_SUFFIXES: dict[str, dict[str, str]] = {
    "linear_attention": {
        "mixer.A_log": "linear_attention.a",
        "mixer.D": "linear_attention.d",
        "mixer.conv1d.weight": "linear_attention.conv.weight",
        "mixer.conv1d.bias": "linear_attention.conv.bias",
        "mixer.dt_bias": "linear_attention.dt_bias",
        "mixer.in_proj.weight": "linear_attention.in_projection.weight",
        "mixer.norm.weight": "linear_attention.norm.weight",
        "mixer.out_proj.weight": "linear_attention.output.weight",
    },
    "full_attention": {
        "mixer.q_proj.weight": "attention.query.weight",
        "mixer.k_proj.weight": "attention.key.weight",
        "mixer.v_proj.weight": "attention.value.weight",
        "mixer.o_proj.weight": "attention.output.weight",
    },
    "moe": {
        "mixer.gate.weight": "mlp.router.weight",
        "mixer.gate.e_score_correction_bias": "mlp.router.bias",
        "mixer.fc1_latent_proj.weight": "mlp.latent.down.weight",
        "mixer.fc2_latent_proj.weight": "mlp.latent.up.weight",
        "mixer.shared_experts.up_proj.weight": "mlp.shared_expert.up.weight",
        "mixer.shared_experts.down_proj.weight": "mlp.shared_expert.down.weight",
    },
    "mlp": {
        "mixer.up_proj.weight": "mlp.up.weight",
        "mixer.down_proj.weight": "mlp.down.weight",
    },
}

_NEMOTRON_EXPERT_RE = re.compile(r"^mixer\.experts\.(\d+)\.(up|down)_proj\.weight$")

# The MTP fusion roots live next to the first predictor block in the payload;
# they are shared by every stage and therefore keep unindexed canonical names.
_NEMOTRON_MTP_ROOTS = {
    "eh_proj.weight": "predictor.fusion.weight",
    "enorm.weight": "predictor.embedding_norm.weight",
    "hnorm.weight": "predictor.hidden_norm.weight",
    "final_layernorm.weight": "predictor.output_norm.weight",
}

_NEMOTRON_VISION_ROOTS = {
    "vision_model.radio_model.input_conditioner.norm_mean": "vision.input_conditioner.mean",
    "vision_model.radio_model.input_conditioner.norm_std": "vision.input_conditioner.std",
    "vision_model.radio_model.model.patch_generator.embedder.weight": (
        "vision.patch_embedding.weight"
    ),
    "vision_model.radio_model.model.patch_generator.video_embedder.weight": (
        "vision.video_patch_embedding.weight"
    ),
    "vision_model.radio_model.model.patch_generator.pos_embed": (
        "vision.position_embedding.weight"
    ),
    "vision_model.radio_model.model.patch_generator.cls_token.token": (
        "vision.cls_register_token.weight"
    ),
    "vision_model.summary_idxs": "vision.summary_indices",
    "vision_projector.vision_final_layernorm.weight": "vision.projector.norm.weight",
    "vision_projector.vision_final_layernorm.bias": "vision.projector.norm.bias",
    "mlp1.0.weight": "vision.projector.mlp.norm.weight",
    "mlp1.1.weight": "vision.projector.mlp.up.weight",
    "mlp1.3.weight": "vision.projector.mlp.down.weight",
}

_NEMOTRON_VISION_BLOCK_RE = re.compile(
    r"^vision_model\.radio_model\.model\.blocks\.(\d+)\.(.+)$"
)
_NEMOTRON_VISION_BLOCK_SUFFIXES = {
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
    "attn.qkv.weight": "attention.qkv.weight",
    "attn.qkv.bias": "attention.qkv.bias",
    "attn.proj.weight": "attention.output.weight",
    "attn.proj.bias": "attention.output.bias",
    "mlp.fc1.weight": "mlp.up.weight",
    "mlp.fc1.bias": "mlp.up.bias",
    "mlp.fc2.weight": "mlp.down.weight",
    "mlp.fc2.bias": "mlp.down.bias",
}

_NEMOTRON_CANONICAL_BLOCK_RE = re.compile(r"^(model|predictor)\.block\.(\d+)\.(.+)$")
_NEMOTRON_CANONICAL_EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(up|down)\.weight$")
_NEMOTRON_CANONICAL_VISION_BLOCK_RE = re.compile(r"^vision\.block\.(\d+)\.(.+)$")


def _nemotron_h_block_suffix(
    layer: int,
    source_suffix: str,
    config: Mapping[str, object],
    *,
    predictor: bool = False,
) -> str | None:
    """Resolve one Nemotron-H block tensor to its canonical leaf.

    The mixer kind is taken from the declared block schedule, never inferred
    from the tensor name: a MoE block and a Mamba block both carry a
    ``...norm.weight`` and an ``...mixer.*`` prefix.
    """

    if source_suffix == "norm.weight":
        return _NEMOTRON_BLOCK_NORM
    mixers = mtp_mixers_from_config(config) if predictor else layer_mixers_from_config(config)
    if layer >= len(mixers):
        return None
    mixer = mixers[layer]
    suffixes = _NEMOTRON_MIXER_SUFFIXES[mixer]
    expert = _NEMOTRON_EXPERT_RE.match(source_suffix)
    if expert is not None:
        if mixer != "moe":
            return None
        return f"mlp.experts.{int(expert.group(1))}.{expert.group(2)}.weight"
    return suffixes.get(source_suffix)


def _nemotron_h_source_mapper(
    source_name: str,
    _topology: GraphTopology,
    config: Mapping[str, object],
) -> TensorNameMapping | None:
    """Map one Nemotron-H text/omni tensor into the canonical vocabulary."""

    if source_name.startswith(("backbone.", "lm_head.")):
        source_name = "language_model." + source_name
    canonical = _already_canonical(source_name)
    if canonical is not None:
        return canonical
    root = _NEMOTRON_ROOTS.get(source_name)
    if root is not None:
        return TensorNameMapping(root, TensorComponent.MODEL)
    vision = _NEMOTRON_VISION_ROOTS.get(source_name)
    if vision is not None:
        return TensorNameMapping(vision, TensorComponent.VISION)
    match = _NEMOTRON_VISION_BLOCK_RE.match(source_name)
    if match is not None:
        suffix = _NEMOTRON_VISION_BLOCK_SUFFIXES.get(match.group(2))
        if suffix is None:
            return None
        return TensorNameMapping(
            vision_block(int(match.group(1)), suffix),
            TensorComponent.VISION,
        )

    match = _NEMOTRON_BACKBONE_LAYER_RE.match(source_name)
    if match is not None:
        suffix = _nemotron_h_block_suffix(int(match.group(1)), match.group(2), config)
        if suffix is None:
            return None
        return TensorNameMapping(
            model_block(int(match.group(1)), suffix),
            TensorComponent.MODEL,
        )

    match = _NEMOTRON_MTP_LAYER_RE.match(source_name)
    if match is None:
        return None
    layer = int(match.group(1))
    source_suffix = match.group(2)
    predictor_root = _NEMOTRON_MTP_ROOTS.get(source_suffix)
    if predictor_root is not None:
        return TensorNameMapping(predictor_root, TensorComponent.PREDICTOR)
    suffix = _nemotron_h_block_suffix(layer, source_suffix, config, predictor=True)
    if suffix is None:
        return None
    return TensorNameMapping(
        predictor_block(layer, suffix),
        TensorComponent.PREDICTOR,
    )


# llama.cpp spells every Nemotron-H block ``blk.N.attn_norm`` plus its mixer, so
# the canonical block leaf alone determines the GGUF recipe name.
_NEMOTRON_RECIPE_ROOTS = {
    "model.token_embedding.weight": "token_embd.weight",
    "model.output_norm.weight": "output_norm.weight",
    "model.output.weight": "output.weight",
}

_NEMOTRON_RECIPE_SUFFIXES = {
    "attention.norm.weight": "attn_norm.weight",
    "linear_attention.a": "ssm_a",
    "linear_attention.d": "ssm_d",
    "linear_attention.conv.weight": "ssm_conv1d.weight",
    "linear_attention.conv.bias": "ssm_conv1d.bias",
    "linear_attention.dt_bias": "ssm_dt.bias",
    "linear_attention.in_projection.weight": "ssm_in.weight",
    "linear_attention.norm.weight": "ssm_norm.weight",
    "linear_attention.output.weight": "ssm_out.weight",
    "attention.query.weight": "attn_q.weight",
    "attention.key.weight": "attn_k.weight",
    "attention.value.weight": "attn_v.weight",
    "attention.output.weight": "attn_output.weight",
    "mlp.router.weight": "ffn_gate_inp.weight",
    "mlp.router.bias": "exp_probs_b.bias",
    "mlp.latent.down.weight": "ffn_latent_down.weight",
    "mlp.latent.up.weight": "ffn_latent_up.weight",
    "mlp.shared_expert.up.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert.down.weight": "ffn_down_shexp.weight",
    "mlp.experts.up.weight": "ffn_up_exps.weight",
    "mlp.experts.down.weight": "ffn_down_exps.weight",
    "mlp.up.weight": "ffn_up.weight",
    "mlp.down.weight": "ffn_down.weight",
}


def nemotron_h_recipe_name(canonical_name: str) -> str | None:
    """Nemotron-H GGUF recipe aliases for the main text blocks only.

    The GGUF conversion folds the whole MTP head into one trailing block with
    ``nextn.*`` roots, so the predictor graph (one pre-normed mixer per stage)
    has no one-to-one recipe name; MTP keeps its own precision policy instead.
    ``mlp.experts.E.{up,down}.weight`` are per-expert views that the importer
    banks into ``ffn_{up,down}_exps``, so only the bank name has an alias.
    """

    root = _NEMOTRON_RECIPE_ROOTS.get(canonical_name)
    if root is not None:
        return root
    match = _NEMOTRON_CANONICAL_BLOCK_RE.match(canonical_name)
    if match is None or match.group(1) != "model":
        return None
    if _NEMOTRON_CANONICAL_EXPERT_RE.match(match.group(3)) is not None:
        return None
    alias = _NEMOTRON_RECIPE_SUFFIXES.get(match.group(3))
    return f"blk.{match.group(2)}.{alias}" if alias is not None else None


def _nemotron_h_block_shape(
    contract: NemotronHConfig,
    mixer: str,
    suffix: str,
) -> tuple[int, ...] | None:
    """Return the payload shape one canonical block tensor must have."""

    hidden = contract.hidden_size
    if suffix == _NEMOTRON_BLOCK_NORM:
        return (hidden,)
    if mixer == "linear_attention":
        heads = contract.mamba_num_heads
        return {
            "linear_attention.in_projection.weight": (
                contract.mamba_in_projection_width,
                hidden,
            ),
            "linear_attention.conv.weight": (
                contract.mamba_conv_width,
                1,
                contract.conv_kernel,
            ),
            "linear_attention.conv.bias": (contract.mamba_conv_width,),
            "linear_attention.a": (heads,),
            "linear_attention.d": (heads,),
            "linear_attention.dt_bias": (heads,),
            "linear_attention.norm.weight": (contract.mamba_inner_size,),
            "linear_attention.output.weight": (hidden, contract.mamba_inner_size),
        }.get(suffix)
    if mixer == "full_attention":
        return {
            "attention.query.weight": (contract.attention_width, hidden),
            "attention.key.weight": (contract.key_value_width, hidden),
            "attention.value.weight": (contract.key_value_width, hidden),
            "attention.output.weight": (hidden, contract.attention_width),
        }.get(suffix)
    if mixer == "mlp":
        return {
            "mlp.up.weight": (contract.intermediate_size, hidden),
            "mlp.down.weight": (hidden, contract.intermediate_size),
        }.get(suffix)
    expert = _NEMOTRON_CANONICAL_EXPERT_RE.match(suffix)
    if expert is not None:
        index = int(expert.group(1))
        if index >= contract.num_experts:
            raise ValueError(
                f"Nemotron-H expert index {index} exceeds n_routed_experts "
                f"{contract.num_experts}"
            )
        expert_hidden = contract.moe_intermediate_size
        latent = contract.moe_latent_size
        return (
            (expert_hidden, latent)
            if expert.group(2) == "up"
            else (latent, expert_hidden)
        )
    latent = contract.moe_latent_size
    return {
        "mlp.router.weight": (contract.num_experts, hidden),
        "mlp.router.bias": (contract.num_experts,),
        "mlp.latent.down.weight": (latent, hidden),
        "mlp.latent.up.weight": (hidden, latent),
        "mlp.shared_expert.up.weight": (contract.moe_shared_intermediate_size, hidden),
        "mlp.shared_expert.down.weight": (hidden, contract.moe_shared_intermediate_size),
    }.get(suffix)


def _nemotron_h_vision_shape(
    contract: NemotronHConfig,
    canonical_name: str,
) -> tuple[int, ...] | None:
    """Return the payload shape one canonical RADIO/projector tensor must have."""

    vision = contract.vision
    if vision is None:
        return None
    patch_width = vision.num_channels * vision.patch_size * vision.patch_size
    shapes = {
        "vision.patch_embedding.weight": (vision.hidden_size, patch_width),
        "vision.position_embedding.weight": (
            1,
            vision.position_side * vision.position_side,
            vision.hidden_size,
        ),
        "vision.cls_register_token.weight": (vision.summary_tokens, vision.hidden_size),
        # RADIO stores the conditioner as [C, 1, 1] so it broadcasts over [C, H, W].
        "vision.input_conditioner.mean": (vision.num_channels, 1, 1),
        "vision.input_conditioner.std": (vision.num_channels, 1, 1),
        "vision.summary_indices": (len(vision.summary_idxs),),
    }
    temporal = vision.video_temporal_patch_size
    if temporal:
        shapes["vision.video_patch_embedding.weight"] = (
            vision.hidden_size,
            temporal * patch_width,
        )
    projector = contract.projector
    if projector is not None:
        shapes.update(
            {
                "vision.projector.norm.weight": (projector.vision_hidden_size,),
                "vision.projector.norm.bias": (projector.vision_hidden_size,),
                "vision.projector.mlp.norm.weight": (projector.patch_width,),
                "vision.projector.mlp.up.weight": (
                    projector.intermediate_size,
                    projector.patch_width,
                ),
                "vision.projector.mlp.down.weight": (
                    projector.text_hidden_size,
                    projector.intermediate_size,
                ),
            }
        )
    shape = shapes.get(canonical_name)
    if shape is not None:
        return shape
    match = _NEMOTRON_CANONICAL_VISION_BLOCK_RE.match(canonical_name)
    if match is None:
        return None
    return {
        "norm1.weight": (vision.hidden_size,),
        "norm1.bias": (vision.hidden_size,),
        "norm2.weight": (vision.hidden_size,),
        "norm2.bias": (vision.hidden_size,),
        "attention.qkv.weight": (3 * vision.hidden_size, vision.hidden_size),
        "attention.qkv.bias": (3 * vision.hidden_size,),
        "attention.output.weight": (vision.hidden_size, vision.hidden_size),
        "attention.output.bias": (vision.hidden_size,),
        "mlp.up.weight": (vision.intermediate_size, vision.hidden_size),
        "mlp.up.bias": (vision.intermediate_size,),
        "mlp.down.weight": (vision.hidden_size, vision.intermediate_size),
        "mlp.down.bias": (vision.hidden_size,),
    }.get(match.group(2))


def _nemotron_h_canonical_shape(
    contract: NemotronHConfig,
    canonical_name: str,
) -> tuple[int, ...] | None:
    """Return the payload shape implied by the config for one canonical name."""

    hidden = contract.hidden_size
    root = {
        "model.token_embedding.weight": (contract.vocab_size, hidden),
        "model.output.weight": (contract.vocab_size, hidden),
        "model.output_norm.weight": (hidden,),
        "predictor.fusion.weight": (hidden, 2 * hidden),
        "predictor.embedding_norm.weight": (hidden,),
        "predictor.hidden_norm.weight": (hidden,),
        "predictor.output_norm.weight": (hidden,),
    }.get(canonical_name)
    if root is not None:
        return root
    match = _NEMOTRON_CANONICAL_BLOCK_RE.match(canonical_name)
    if match is None:
        return _nemotron_h_vision_shape(contract, canonical_name)
    component, raw_layer, suffix = match.groups()
    mixers = contract.mixer_types if component == "model" else contract.mtp_mixer_types
    layer = int(raw_layer)
    if layer >= len(mixers):
        return None
    return _nemotron_h_block_shape(contract, mixers[layer], suffix)


def _nemotron_h_source_shape_validator(
    config: Mapping[str, object],
    shapes: Mapping[str, tuple[int, ...]],
) -> None:
    """Reject a Nemotron-H payload whose geometry contradicts its config.

    Runs at the import boundary before any tensor is read, so a config that
    was copied between variants fails loudly instead of producing a container
    that cannot be loaded.
    """

    contract = NemotronHConfig.from_hf_config(config)
    topology = GraphTopology(text_layers=contract.num_hidden_layers)
    for source_name, actual in shapes.items():
        mapped = _nemotron_h_source_mapper(source_name, topology, config)
        if mapped is None:
            continue
        expected = _nemotron_h_canonical_shape(contract, mapped.canonical_name)
        if expected is None or tuple(int(value) for value in actual) == expected:
            continue
        raise ValueError(
            f"Nemotron-H source tensor shape mismatch for {source_name}: "
            f"checkpoint={tuple(actual)}, expected={expected}"
        )


def _minicpmo_source_assets(
    _root: Path,
    _config: Mapping[str, object],
) -> Sequence["RuntimeAsset"]:
    from mfq.formats.assets import minicpmo45_resampler_pos_embed_asset

    return (minicpmo45_resampler_pos_embed_asset(),)


def _deepseek_v41_source_assets(
    root: Path,
    config: Mapping[str, object],
) -> Sequence["RuntimeAsset"]:
    from mfq.formats.assets import deepseek_v41_engram_asset

    text = text_section(config)
    layer_ids = text.get("engram_layer_ids", ())
    if not isinstance(layer_ids, Sequence) or isinstance(layer_ids, (str, bytes)):
        return ()
    if not layer_ids:
        return ()
    return (deepseek_v41_engram_asset(root / "tokenizer.json", dict(text)),)


register_tensor_schema(
    TensorSchemaRegistration(
        architecture="qwen3_5",
        aliases=("qwen3_5_text", "qwen3_5_moe", "qwen3_6", "qwen3_8"),
        source_mapper=_qwen35_source_mapper,
    )
)

register_tensor_schema(
    TensorSchemaRegistration(
        architecture="qwen4_exp",
        aliases=("qwen4_exp_text",),
        source_mapper=_qwen4_source_mapper,
        recipe_name_mapper=qwen4_recipe_name,
        recipe_expert_defaults=True,
    )
)

register_tensor_schema(
    TensorSchemaRegistration(
        architecture="glm5_next",
        aliases=("glm5_next_text",),
        source_mapper=_glm5_source_mapper,
    )
)

register_tensor_schema(
    TensorSchemaRegistration(
        architecture="deepseek_v41",
        aliases=("deepseek_v41_text", "deepseek_v41_vision"),
        source_mapper=_deepseek_v41_source_mapper,
        source_asset_builder=_deepseek_v41_source_assets,
        component_specs=(
            GraphComponentSpec(
                TensorComponent.VISION,
                "vision",
                "deepseek_v41_vision",
                input_contract="deepseek_v41_vision.v1",
                position_policy="deepseek_v41_positions",
                capabilities=("vision",),
            ),
            GraphComponentSpec(
                TensorComponent.PREDICTOR,
                "predictor",
                "deepseek_v41_dspark",
                capabilities=("speculative_prediction",),
            ),
        ),
    )
)

register_tensor_schema(
    TensorSchemaRegistration(
        architecture="deepseek_v4",
        aliases=("deepseek_v4_text", "deepseek_v4_vision"),
        source_mapper=_deepseek_v4_source_mapper,
        component_specs=(
            GraphComponentSpec(
                TensorComponent.VISION,
                "vision",
                "deepseek_v4_vision",
                input_contract="deepseek_v4_vision.v1",
                position_policy="deepseek_v4_positions",
                capabilities=("vision",),
            ),
            GraphComponentSpec(
                TensorComponent.PREDICTOR,
                "predictor",
                "dspark",
                capabilities=("speculative_prediction",),
            ),
        ),
    )
)

register_tensor_schema(
    TensorSchemaRegistration(
        architecture="minicpmo",
        aliases=("minicpmo45", "minicpmo_4_5"),
        source_mapper=_minicpmo_source_mapper,
        backbone="minicpmo45",
        source_asset_builder=_minicpmo_source_assets,
        component_specs=(
            GraphComponentSpec(
                TensorComponent.VISION,
                "vision",
                "minicpmo45_vision",
                input_contract="minicpmo45.v1",
                position_policy="minicpmo45_positions",
                capabilities=("vision",),
            ),
            GraphComponentSpec(
                TensorComponent.AUDIO,
                "audio_input",
                "minicpmo45_audio",
                capabilities=("audio_input",),
            ),
            GraphComponentSpec(
                TensorComponent.TTS,
                "audio_output",
                "minicpmo45_tts",
                capabilities=("audio_output",),
            ),
            GraphComponentSpec(
                TensorComponent.RUNTIME,
                "duplex",
                "minicpmo45_duplex",
                capabilities=("realtime_duplex",),
                requires=(TensorComponent.AUDIO, TensorComponent.TTS),
            ),
        ),
    )
)

register_tensor_schema(
    TensorSchemaRegistration(
        architecture="glm_dsa",
        aliases=("glm_moe_dsa",),
        source_mapper=_glm_dsa_source_mapper,
    )
)

register_tensor_schema(
    TensorSchemaRegistration(
        architecture="gemma4",
        aliases=("gemma4_text",),
        source_mapper=_gemma4_source_mapper,
    )
)

register_tensor_schema(
    TensorSchemaRegistration(
        architecture="nemotron_h_omni",
        aliases=("nemotron_h", "nemotron_h_text"),
        source_mapper=_nemotron_h_source_mapper,
        backbone="nemotron_h",
        # Routed experts are plain relu-squared MLPs: the bank stores up/down
        # only, so a non-gated family needs no importer-side special case.
        expert_projections=("up", "down"),
        # Recurrent state, depthwise convolution and router parameters must
        # not be narrowed on import.
        float32_weight_roles=(
            ".linear_attention.a",
            ".linear_attention.d",
            ".linear_attention.dt_bias",
            ".linear_attention.conv.weight",
            ".linear_attention.conv.bias",
            # Every normalization and router parameter stays exact.
            ".norm.weight",
            ".norm.bias",
            ".output_norm.weight",
            ".mlp.router.weight",
            ".mlp.router.bias",
        ),
        source_shape_validator=_nemotron_h_source_shape_validator,
        recipe_name_mapper=nemotron_h_recipe_name,
        recipe_expert_defaults=True,
        component_specs=(
            GraphComponentSpec(
                TensorComponent.VISION,
                "vision",
                "radio",
                input_contract=RADIO_CPE_INPUT_CONTRACT,
                capabilities=("vision",),
            ),
            GraphComponentSpec(
                TensorComponent.PREDICTOR,
                "predictor",
                "nemotron_h_mtp",
                capabilities=("speculative_prediction",),
            ),
        ),
    )
)


__all__ = [
    "CANONICAL_TENSOR_NAMESPACE",
    "CANONICAL_TENSOR_SCHEMA_VERSION",
    "GraphTopology",
    "GraphComponentSpec",
    "GRID_MROPE_POSITION_POLICY",
    "RADIO_CPE_INPUT_CONTRACT",
    "GRID_VISION_INPUT_CONTRACT",
    "ModelGraphSpec",
    "TensorComponent",
    "TensorNameMapping",
    "TensorSchemaRegistration",
    "canonical_recipe_name",
    "canonical_source_tensor_map",
    "graph_spec_for_plan",
    "graph_spec_for_source_names",
    "map_source_tensor_name",
    "model_block",
    "nemotron_h_recipe_name",
    "predictor_block",
    "qwen4_recipe_name",
    "recipe_name_mapper_for_config",
    "register_tensor_schema",
    "source_runtime_assets",
    "tensor_schema_for_config",
    "topology_from_config",
    "vision_block",
]
