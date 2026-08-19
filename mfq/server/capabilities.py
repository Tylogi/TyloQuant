"""Architecture-family capability registry for MFQ clients."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mfq.server.models import ModelCapabilities, ModelFeatureSet


@dataclass(frozen=True)
class _CapabilityRegistration:
    family: str
    aliases: tuple[str, ...]
    features: ModelFeatureSet


_REGISTRY = (
    _CapabilityRegistration(
        family="minicpmo",
        aliases=("minicpmo",),
        features=ModelFeatureSet(
            text=True,
            image_input=True,
            video_input=True,
            audio_input=True,
            audio_output=True,
            full_duplex=True,
        ),
    ),
    _CapabilityRegistration(
        family="minicpmo_tts",
        aliases=("minicpmtts",),
        features=ModelFeatureSet(text=False, audio_output=True),
    ),
    _CapabilityRegistration(
        family="deepseek_v4",
        aliases=("deepseek_v4",),
        features=ModelFeatureSet(),
    ),
    _CapabilityRegistration(
        family="glm_dsa",
        aliases=("glm_moe_dsa",),
        features=ModelFeatureSet(),
    ),
    _CapabilityRegistration(
        family="gemma4",
        aliases=("gemma4", "gemma4_text"),
        features=ModelFeatureSet(),
    ),
    _CapabilityRegistration(
        family="qwen3_5",
        aliases=("qwen3_5", "qwen3_5_text"),
        features=ModelFeatureSet(),
    ),
)


def normalize_architecture(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "unknown"


def capabilities_for_architecture(model_type: str) -> ModelCapabilities:
    identity = normalize_architecture(model_type)
    for registration in _REGISTRY:
        if identity in registration.aliases:
            return ModelCapabilities(
                architecture_family=registration.family,
                source=f"architecture-registry:{registration.family}",
                features=registration.features,
            )
    return ModelCapabilities(
        architecture_family=identity,
        source=f"architecture-registry:{identity}",
        features=ModelFeatureSet(),
    )


__all__ = ["capabilities_for_architecture", "normalize_architecture"]
