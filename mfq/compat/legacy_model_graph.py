"""One compatibility boundary for pre-schema MFQ runtime graphs.

Only this module may infer components from old architecture strings or source
checkpoint tensor spellings. New artifacts carry ``model_graph.json`` and all
runtime routing consumes its stable implementation/component identities.
Delete this module with the canonical-schema-v1 compatibility window.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from mfq.architectures.tensor_schema import (
    CANONICAL_TENSOR_NAMESPACE,
    GRID_MROPE_POSITION_POLICY,
    GRID_VISION_INPUT_CONTRACT,
    graph_spec_for_plan,
    map_source_tensor_name,
)


def _identity(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def legacy_model_graph(
    architecture: str,
    record_names: Collection[str],
    config: Mapping[str, object] | None = None,
) -> dict[str, Any] | None:
    """Synthesize stable component IDs for one pre-schema artifact."""

    if config is not None:
        canonical_names = []
        for name in record_names:
            mapped = map_source_tensor_name(name, config)
            canonical_names.append(
                mapped.canonical_name if mapped is not None else name
            )
        spec = graph_spec_for_plan(config, canonical_names)
        if spec is not None:
            return spec.as_dict()

    identity = _identity(architecture)
    roots: list[dict[str, Any]] = []
    capabilities = ["text"]
    if identity.startswith(("qwen35", "qwen3_5", "qwen3_6", "qwen3_8")):
        roots.append(
            {
                "kind": "text",
                "tensor_root": "model",
                "implementation": "qwen3_5",
                "policy": "decoder",
            }
        )
        has_vision = any(
            name in record_names
            for name in (
                "vision.patch_embedding.weight",
                "model.visual.patch_embed.proj.weight",
            )
        )
        has_predictor = any(
            name in record_names
            for name in (
                "predictor.fusion.weight",
                "mtp.fc.weight",
            )
        )
        if has_vision:
            roots.append(
                {
                    "kind": "vision",
                    "tensor_root": "vision",
                    "implementation": "grid_vit",
                    "policy": "optional",
                    "input_contract": GRID_VISION_INPUT_CONTRACT,
                    "position_policy": GRID_MROPE_POSITION_POLICY,
                }
            )
            capabilities.append("vision")
        if has_predictor:
            roots.append(
                {
                    "kind": "predictor",
                    "tensor_root": "predictor",
                    "implementation": "next_token_prediction",
                    "policy": "optional",
                }
            )
            capabilities.append("speculative_prediction")
        family = backbone = "qwen3_5"
    elif identity.startswith("qwen4_exp"):
        family = backbone = "qwen4_exp"
        roots.append(
            {
                "kind": "text",
                "tensor_root": "model",
                "implementation": backbone,
                "policy": "decoder",
            }
        )
        if any(name.startswith(("vision.", "model.visual.")) for name in record_names):
            roots.append(
                {
                    "kind": "vision",
                    "tensor_root": "vision",
                    "implementation": "grid_vit",
                    "policy": "optional",
                    "input_contract": GRID_VISION_INPUT_CONTRACT,
                    "position_policy": GRID_MROPE_POSITION_POLICY,
                }
            )
            capabilities.append("vision")
        if any(name.startswith(("predictor.", "mtp.")) for name in record_names):
            roots.append(
                {
                    "kind": "predictor",
                    "tensor_root": "predictor",
                    "implementation": "next_token_prediction",
                    "policy": "optional",
                }
            )
            capabilities.append("speculative_prediction")
    elif identity.startswith("glm5_next"):
        family = backbone = "glm5_next"
        roots.append(
            {
                "kind": "text",
                "tensor_root": "model",
                "implementation": backbone,
                "policy": "decoder",
            }
        )
    else:
        return None

    component_roots = [str(component["tensor_root"]) for component in roots]
    return {
        "schema_version": 1,
        "architecture": family,
        "graph": {"kind": "causal_lm", "backbone": backbone},
        "topology": {
            "text_layers": 1,
            "vision_layers": 1 if "vision" in component_roots else 0,
            "predictor_layers": 1 if "predictor" in component_roots else 0,
        },
        "components": roots,
        "capabilities": capabilities,
        "canonical_naming": {
            "namespace": CANONICAL_TENSOR_NAMESPACE,
            "version": 1,
            "component_roots": component_roots,
        },
    }


__all__ = ["legacy_model_graph"]
