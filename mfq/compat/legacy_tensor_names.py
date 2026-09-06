"""Compatibility for MFQ artifacts that stored Hugging Face tensor paths.

DELETE THIS MODULE after the schema-v1 migration window.  New importers must
write canonical names and runtime graph implementations must not add legacy
name probes.  A container loader may use this adapter once, at its boundary,
to expose an old artifact as a canonical record mapping.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from mfq.architectures.tensor_schema import map_source_tensor_name
from mfq.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET

# Machine-searchable removal marker; intentionally not a feature/version gate.
LEGACY_HF_TENSOR_NAMES_REMOVE_AFTER = "canonical-schema-v1-migration"


_QWEN35_GGUF_ROOTS = {
    "token_embd.weight": "model.token_embedding.weight",
    "output_norm.weight": "model.output_norm.weight",
    "output.weight": "model.output.weight",
}
_QWEN35_GGUF_SUFFIXES = {
    "attn_norm.weight": "attention.norm.weight",
    "attn_q.weight": "attention.query.weight",
    "attn_k.weight": "attention.key.weight",
    "attn_v.weight": "attention.value.weight",
    "attn_output.weight": "attention.output.weight",
    "attn_q_norm.weight": "attention.query_norm.weight",
    "attn_k_norm.weight": "attention.key_norm.weight",
    "post_attention_norm.weight": "mlp.norm.weight",
    "ffn_gate.weight": "mlp.gate.weight",
    "ffn_up.weight": "mlp.up.weight",
    "ffn_down.weight": "mlp.down.weight",
    "attn_qkv.weight": "linear_attention.qkv.weight",
    "ssm_qk.weight": "linear_attention.qk.weight",
    "ssm_v.weight": "linear_attention.value.weight",
    "attn_gate.weight": "linear_attention.gate.weight",
    "ssm_alpha.weight": "linear_attention.alpha.weight",
    "ssm_beta.weight": "linear_attention.beta.weight",
    "ssm_conv1d.weight": "linear_attention.conv.weight",
    "ssm_conv1d.bias": "linear_attention.conv.bias",
    "ssm_dt.bias": "linear_attention.dt_bias",
    "ssm_a": "linear_attention.a",
    "ssm_norm.weight": "linear_attention.norm.weight",
    "ssm_out.weight": "linear_attention.output.weight",
}
_QWEN35_GGUF_BLOCK = re.compile(r"^blk\.(\d+)\.(.+)$")


def qwen35_legacy_name_map(
    record_names: Iterable[str],
    config: Mapping[str, object],
) -> dict[str, str]:
    """Return ``legacy -> canonical`` aliases for one old Qwen3.5 artifact."""

    names = tuple(record_names)
    aliases: dict[str, str] = {}
    canonical_seen: dict[str, str] = {}
    for legacy_name in names:
        mapped = map_source_tensor_name(legacy_name, config)
        if mapped is not None and mapped.canonical_name == legacy_name:
            canonical_seen[legacy_name] = legacy_name
    for legacy_name in names:
        mapped = map_source_tensor_name(legacy_name, config)
        if mapped is None or mapped.canonical_name == legacy_name:
            continue
        previous = canonical_seen.get(mapped.canonical_name)
        if previous is not None and previous != legacy_name:
            raise ValueError(
                "legacy MFQ tensor names collide after canonicalization: "
                f"{previous!r}, {legacy_name!r} -> {mapped.canonical_name!r}"
            )
        aliases[legacy_name] = mapped.canonical_name
        canonical_seen[mapped.canonical_name] = legacy_name
    return aliases


def _qwen35_gguf_name_map(
    record_names: Iterable[str],
    config: Mapping[str, object],
) -> dict[str, str]:
    raw_text = config.get("text_config", config)
    text = raw_text if isinstance(raw_text, Mapping) else {}
    text_layers = int(text.get("num_hidden_layers", 0) or 0)
    predictor_layers = int(text.get("mtp_num_hidden_layers", 0) or 0)
    aliases: dict[str, str] = {}
    for stored in record_names:
        root = _QWEN35_GGUF_ROOTS.get(stored)
        if root is not None:
            aliases[stored] = root
            continue
        match = _QWEN35_GGUF_BLOCK.match(stored)
        if match is None:
            continue
        layer = int(match.group(1))
        suffix = match.group(2)
        if layer == text_layers and predictor_layers > 0 and suffix.startswith("nextn."):
            predictor_root = {
                "nextn.hnorm.weight": "predictor.hidden_norm.weight",
                "nextn.enorm.weight": "predictor.embedding_norm.weight",
                "nextn.eh_proj.weight": "predictor.fusion.weight",
                "nextn.shared_head_norm.weight": "predictor.output_norm.weight",
            }.get(suffix)
            if predictor_root is not None:
                aliases[stored] = predictor_root
            continue
        canonical_suffix = _QWEN35_GGUF_SUFFIXES.get(suffix)
        if canonical_suffix is None:
            continue
        if layer < text_layers:
            aliases[stored] = f"model.block.{layer}.{canonical_suffix}"
        elif layer < text_layers + predictor_layers:
            aliases[stored] = (
                f"predictor.block.{layer - text_layers}.{canonical_suffix}"
            )
    return aliases


def legacy_tensor_name_map(
    record_names: Iterable[str],
    config: Mapping[str, object],
) -> dict[str, str]:
    """Map every pre-schema stored name to the canonical runtime namespace."""

    names = tuple(record_names)
    aliases = qwen35_legacy_name_map(names, config)
    raw_text = config.get("text_config", config)
    text = raw_text if isinstance(raw_text, Mapping) else {}
    identity = str(text.get("model_type", config.get("model_type", ""))).lower()
    if identity.startswith(("qwen3_5", "qwen3_6", "qwen3_8", "qwen35")):
        aliases.update(_qwen35_gguf_name_map(names, config))
    visible_names: dict[str, str] = {}
    for stored in names:
        canonical = aliases.get(stored, stored)
        previous = visible_names.get(canonical)
        if previous is not None and previous != stored:
            raise ValueError(
                "legacy MFQ tensor names collide after canonicalization: "
                f"{previous!r}, {stored!r} -> {canonical!r}"
            )
        visible_names[canonical] = stored
    return aliases


class LegacyCanonicalTensorView(Mapping[str, Any]):
    """Expose one old tensor table through canonical names without copying blobs."""

    def __init__(
        self,
        store: Any,
        aliases: Mapping[str, str],
        *,
        legacy_semantics: str | None = None,
    ) -> None:
        self._store = store
        self.legacy_semantics = legacy_semantics
        self._stored_by_visible: dict[str, str] = {}
        for stored in store.records:
            visible = aliases.get(stored, stored)
            previous = self._stored_by_visible.get(visible)
            if previous is not None and previous != stored:
                raise ValueError(
                    f"legacy tensor alias collision: {previous!r}, {stored!r} -> {visible!r}"
                )
            self._stored_by_visible[visible] = stored
        self.records = {
            visible: store.records[stored]
            for visible, stored in self._stored_by_visible.items()
        }
        for attribute in ("path", "paths", "header"):
            if hasattr(store, attribute):
                setattr(self, attribute, getattr(store, attribute))

    def _stored(self, name: str) -> str:
        return self._stored_by_visible[name]

    def __getitem__(self, name: str) -> Any:
        return self._store[self._stored(name)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._stored_by_visible)

    def __len__(self) -> int:
        return len(self._stored_by_visible)

    def read_blob(self, name: str) -> bytes:
        return self._store.read_blob(self._stored(name))

    def blob_view(self, record: Any) -> memoryview:
        if isinstance(record, str):
            record = self._stored(record)
        return self._store.blob_view(record)

    def mmap_for(self, record: Any) -> Any:
        if isinstance(record, str):
            record = self._stored(record)
        return self._store.mmap_for(record)

    def file_for(self, record: Any) -> Any:
        if isinstance(record, str):
            record = self._stored(record)
        return self._store.file_for(record)

    def close(self) -> None:
        self._store.close()


def canonical_tensor_view(store: Any) -> Any:
    """Return a canonical view; schema-v1 artifacts pass through unchanged."""

    if MODEL_GRAPH_ASSET in store.records:
        return store
    if MODEL_CONFIG_ASSET not in store.records:
        return store
    payload = json.loads(store.read_blob(MODEL_CONFIG_ASSET))
    if not isinstance(payload, Mapping):
        raise ValueError("embedded legacy model config must be a JSON object")
    aliases = legacy_tensor_name_map(store.records, payload)
    raw_text = payload.get("text_config", payload)
    text = raw_text if isinstance(raw_text, Mapping) else {}
    identity = str(text.get("model_type", payload.get("model_type", ""))).lower()
    gguf_semantics = identity.startswith(
        ("qwen3_5", "qwen3_6", "qwen3_8", "qwen35")
    ) and any(name.startswith("blk.") for name in store.records)
    return (
        LegacyCanonicalTensorView(
            store,
            aliases,
            legacy_semantics="qwen35_gguf" if gguf_semantics else None,
        )
        if aliases
        else store
    )


__all__ = [
    "LegacyCanonicalTensorView",
    "LEGACY_HF_TENSOR_NAMES_REMOVE_AFTER",
    "canonical_tensor_view",
    "legacy_tensor_name_map",
    "qwen35_legacy_name_map",
]
