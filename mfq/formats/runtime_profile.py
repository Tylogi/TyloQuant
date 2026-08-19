"""Versioned runtime sampling profiles for MFQ containers.

Profiles are intentionally partial: an absent field remains absent and is
filled only by a lower-priority source or by the runtime's generic defaults.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RUNTIME_SAMPLING_METADATA_KEY = "runtime.sampling.v1"
RUNTIME_SAMPLING_SCHEMA = "mfq.runtime.sampling"
RUNTIME_SAMPLING_VERSION = 1

_CHAT_FIELDS = {
    "max_tokens": int,
    "temperature": float,
    "top_k": int,
    "top_p": float,
    "presence_penalty": float,
    "frequency_penalty": float,
    "repetition_penalty": float,
    "enable_thinking": bool,
}
_DUPLEX_FIELDS = {
    "system_prompt": str,
    "decode_mode": str,
    "temperature": float,
    "top_k": int,
    "top_p": float,
    "text_repetition_penalty": float,
    "text_repetition_window_size": int,
    "length_penalty": float,
    "listen_prob_scale": float,
    "force_listen_count": int,
    "max_new_speak_tokens_per_chunk": int,
}
_TTS_FIELDS = {
    "temperature": float,
    "repetition_penalty": float,
    "token2wav_steps": int,
}


def _profile(
    *,
    chat: Mapping[str, Any] | None = None,
    duplex: Mapping[str, Any] | None = None,
    tts: Mapping[str, Any] | None = None,
    source: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": RUNTIME_SAMPLING_SCHEMA,
        "version": RUNTIME_SAMPLING_VERSION,
    }
    if chat:
        result["chat"] = dict(chat)
    if duplex:
        result["duplex"] = dict(duplex)
    if tts:
        result["tts"] = dict(tts)
    result["provenance"] = {"source": source}
    return result


# Verified model-family defaults. Keep these partial instead of copying generic
# runtime defaults into every profile.
_ARCHITECTURE_REGISTRY: dict[str, dict[str, Any]] = {
    "deepseek_v4": _profile(
        chat={
            "temperature": 1.0,
            "top_p": 0.8,
            "repetition_penalty": 1.05,
            "presence_penalty": 0.0,
        },
        source="architecture-registry:deepseek_v4",
    ),
    "minicpmo": _profile(
        chat={
            "temperature": 0.7,
            "top_k": 100,
            "top_p": 0.8,
            "repetition_penalty": 1.02,
            "enable_thinking": False,
        },
        duplex={
            "system_prompt": "Streaming Omni Conversation.",
            "decode_mode": "sampling",
            "temperature": 0.7,
            "top_k": 100,
            "top_p": 0.8,
            "text_repetition_penalty": 1.05,
            "text_repetition_window_size": 512,
            "length_penalty": 1.0,
            "listen_prob_scale": 1.0,
            "force_listen_count": 0,
            "max_new_speak_tokens_per_chunk": 20,
        },
        tts={
            "temperature": 0.8,
            "repetition_penalty": 1.05,
            "token2wav_steps": 10,
        },
        source="architecture-registry:minicpmo",
    ),
}

_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "deepseek_v4_flash_0731": _profile(
        chat={
            "temperature": 1.0,
            "top_p": 0.8,
            "repetition_penalty": 1.05,
            "presence_penalty": 0.0,
        },
        source="model-registry:deepseek-v4-flash-0731",
    ),
    "minicpm_o_4_5": _profile(
        chat={
            "temperature": 0.7,
            "top_k": 100,
            "top_p": 0.8,
            "repetition_penalty": 1.02,
            "enable_thinking": False,
        },
        duplex={
            "system_prompt": "Streaming Omni Conversation.",
            "decode_mode": "sampling",
            "temperature": 0.7,
            "top_k": 100,
            "top_p": 0.8,
            "text_repetition_penalty": 1.05,
            "text_repetition_window_size": 512,
            "length_penalty": 1.0,
            "listen_prob_scale": 1.0,
            "force_listen_count": 0,
            "max_new_speak_tokens_per_chunk": 20,
        },
        tts={
            "temperature": 0.8,
            "repetition_penalty": 1.05,
            "token2wav_steps": 10,
        },
        source="model-registry:minicpm-o-4_5",
    ),
}


def _normalise_identity(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def architecture_profile(*identities: object) -> dict[str, Any] | None:
    names = [_normalise_identity(value) for value in identities if value]
    for name in names:
        for key, profile in _ARCHITECTURE_REGISTRY.items():
            if name == key or name.startswith(f"{key}_") or key in name:
                return copy.deepcopy(profile)
    return None


def model_profile(*identities: object) -> dict[str, Any] | None:
    for value in identities:
        key = _normalise_identity(value)
        for model_key, profile in _MODEL_REGISTRY.items():
            if key == model_key or model_key in key:
                return copy.deepcopy(profile)
    return None


def _validate_section(
    name: str,
    value: object,
    fields: Mapping[str, type],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"runtime profile {name} must be a JSON object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key not in fields:
            # Unknown fields survive for forward compatibility.
            result[str(key)] = copy.deepcopy(item)
            continue
        expected = fields[key]
        if expected is bool:
            if not isinstance(item, bool):
                raise ValueError(
                    f"runtime profile {name}.{key} must be a boolean"
                )
            result[key] = item
            continue
        if expected is str:
            if not isinstance(item, str):
                raise ValueError(f"runtime profile {name}.{key} must be a string")
            if key == "decode_mode" and item not in {"sampling", "greedy"}:
                raise ValueError("runtime profile duplex.decode_mode is invalid")
            result[key] = item
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"runtime profile {name}.{key} must be numeric")
        if not math.isfinite(float(item)):
            raise ValueError(f"runtime profile {name}.{key} must be finite")
        if expected is int and not float(item).is_integer():
            raise ValueError(f"runtime profile {name}.{key} must be an integer")
        result[key] = int(item) if expected is int else float(item)
    if "temperature" in result and not 0.0 <= result["temperature"] <= 10.0:
        raise ValueError(f"runtime profile {name}.temperature must be in [0, 10]")
    if "top_p" in result and not 0.0 <= result["top_p"] <= 1.0:
        raise ValueError(f"runtime profile {name}.top_p must be in [0, 1]")
    if "top_k" in result and result["top_k"] < 0:
        raise ValueError(f"runtime profile {name}.top_k must be non-negative")
    for key in ("max_tokens", "text_repetition_window_size", "max_new_speak_tokens_per_chunk", "token2wav_steps"):
        if key in result and result[key] <= 0:
            raise ValueError(f"runtime profile {name}.{key} must be positive")
    for key in ("repetition_penalty", "text_repetition_penalty", "length_penalty"):
        if key in result and result[key] <= 0.0:
            raise ValueError(f"runtime profile {name}.{key} must be positive")
    if "listen_prob_scale" in result and result["listen_prob_scale"] < 0.0:
        raise ValueError(
            f"runtime profile {name}.listen_prob_scale must be non-negative"
        )
    if "force_listen_count" in result and not 0 <= result["force_listen_count"] <= 60:
        raise ValueError(
            f"runtime profile {name}.force_listen_count must be in [0, 60]"
        )
    return result


def validate_runtime_profile(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("runtime sampling profile must be a JSON object")
    if value.get("schema", RUNTIME_SAMPLING_SCHEMA) != RUNTIME_SAMPLING_SCHEMA:
        raise ValueError("unsupported runtime sampling profile schema")
    if value.get("version", RUNTIME_SAMPLING_VERSION) != RUNTIME_SAMPLING_VERSION:
        raise ValueError("unsupported runtime sampling profile version")
    result: dict[str, Any] = {
        "schema": RUNTIME_SAMPLING_SCHEMA,
        "version": RUNTIME_SAMPLING_VERSION,
    }
    for name, fields in (
        ("chat", _CHAT_FIELDS),
        ("duplex", _DUPLEX_FIELDS),
        ("tts", _TTS_FIELDS),
    ):
        if name in value:
            section = _validate_section(name, value[name], fields)
            if section:
                result[name] = section
    provenance = value.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise ValueError("runtime profile provenance must be a JSON object")
        result["provenance"] = copy.deepcopy(dict(provenance))
    return result


def merge_runtime_profiles(*profiles: Mapping[str, Any] | None) -> dict[str, Any] | None:
    result: dict[str, Any] = {
        "schema": RUNTIME_SAMPLING_SCHEMA,
        "version": RUNTIME_SAMPLING_VERSION,
    }
    used = False
    for raw in profiles:
        if raw is None:
            continue
        profile = validate_runtime_profile(raw)
        for section in ("chat", "duplex", "tts"):
            if section in profile:
                result.setdefault(section, {}).update(profile[section])
                used = True
        if "provenance" in profile:
            result["provenance"] = profile["provenance"]
    return result if used else None


def load_runtime_profile(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_runtime_profile(value)


def sidecar_paths(mfq_path: str | Path) -> tuple[Path, ...]:
    path = Path(mfq_path)
    exact = Path(f"{path}.runtime.json")
    match = re.match(r"^(.*)-[0-9]{5}-of-[0-9]{5}\.mfq$", path.name)
    family = (
        path.with_name(f"{match.group(1)}.runtime.json")
        if match
        else path.with_suffix(".runtime.json")
    )
    return (family, exact) if family != exact else (exact,)


def load_mfq_sidecar(mfq_path: str | Path) -> dict[str, Any] | None:
    result = None
    for path in sidecar_paths(mfq_path):
        if path.is_file():
            result = merge_runtime_profiles(result, load_runtime_profile(path))
    return result


def generation_config_profile(value: Mapping[str, Any]) -> dict[str, Any] | None:
    aliases = {
        "max_new_tokens": "max_tokens",
        "temperature": "temperature",
        "top_k": "top_k",
        "top_p": "top_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "repetition_penalty": "repetition_penalty",
    }
    chat = {
        target: value[source]
        for source, target in aliases.items()
        if source in value and value[source] is not None
    }
    if not chat:
        return None
    return validate_runtime_profile(
        _profile(chat=chat, source="hf:generation_config.json")
    )


def profile_for_new_mfq(
    source: str | Path | None,
    config: Mapping[str, Any] | None,
    *,
    explicit_profile: str | Path | None = None,
    inherited_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build metadata for a newly written MFQ without inventing defaults."""

    config = config or {}
    identities = (
        config.get("_name_or_path"),
        config.get("model_type"),
        *(config.get("architectures", ()) or ()),
    )
    profile = merge_runtime_profiles(
        architecture_profile(*identities),
        model_profile(*identities),
        inherited_profile,
    )
    root = Path(source) if source else None
    if root is not None and root.is_dir():
        generation_path = root / "generation_config.json"
        if generation_path.is_file():
            raw = json.loads(generation_path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise ValueError("generation_config.json must be a JSON object")
            profile = merge_runtime_profiles(profile, generation_config_profile(raw))
        source_sidecar = root / "mfq-runtime.json"
        if source_sidecar.is_file():
            profile = merge_runtime_profiles(profile, load_runtime_profile(source_sidecar))
    if explicit_profile:
        profile = merge_runtime_profiles(profile, load_runtime_profile(explicit_profile))
    return profile
