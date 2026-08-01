"""Per-output-row importance artifacts for activation-aware quantization."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_FORMAT = "mfq.row-importance.v1"
_METADATA_KEY = "__metadata_json__"


@dataclass(frozen=True)
class RowImportance:
    path: Path
    entries: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def require(self, name: str, rows: int) -> np.ndarray:
        try:
            value = self.entries[name]
        except KeyError as exc:
            raise KeyError(f"row-importance artifact has no entry for {name}") from exc
        if value.shape != (rows,):
            raise ValueError(
                f"row importance for {name} has shape {value.shape}, expected {(rows,)}"
            )
        return value


def save_row_importance(
    path: str | Path,
    entries: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"row-importance artifact already exists: {output}")
    arrays: dict[str, np.ndarray] = {}
    entry_metadata: dict[str, dict[str, Any]] = {}
    for index, (name, raw) in enumerate(sorted(entries.items())):
        value = np.ascontiguousarray(raw, dtype=np.float32).reshape(-1)
        if not value.size or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"invalid row importance for {name}")
        if not np.any(value > 0):
            raise ValueError(f"row importance for {name} is entirely zero")
        key = f"entry_{index:04d}"
        arrays[key] = value
        entry_metadata[name] = {"key": key, "rows": int(value.size)}
    document = {
        "format": _FORMAT,
        "metadata": dict(metadata),
        "entries": entry_metadata,
    }
    arrays[_METADATA_KEY] = np.frombuffer(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8,
    ).copy()
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, output)


def load_row_importance(path: str | Path) -> RowImportance:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"row-importance artifact does not exist: {resolved}")
    with np.load(resolved, allow_pickle=False) as archive:
        if _METADATA_KEY not in archive.files:
            raise ValueError(f"row-importance artifact has no metadata: {resolved}")
        try:
            document = json.loads(archive[_METADATA_KEY].tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid row-importance metadata: {resolved}") from exc
        if document.get("format") != _FORMAT:
            raise ValueError(
                f"unsupported row-importance format {document.get('format')!r}: {resolved}"
            )
        entries: dict[str, np.ndarray] = {}
        for name, item in document.get("entries", {}).items():
            key = str(item["key"])
            rows = int(item["rows"])
            if key not in archive.files:
                raise ValueError(f"row-importance entry {name} is missing array {key}")
            value = np.ascontiguousarray(archive[key], dtype=np.float32).reshape(-1)
            if value.shape != (rows,):
                raise ValueError(
                    f"row-importance entry {name} has shape {value.shape}, expected {(rows,)}"
                )
            if not np.isfinite(value).all() or np.any(value < 0) or not np.any(value > 0):
                raise ValueError(f"invalid row-importance values for {name}")
            entries[str(name)] = value
    if not entries:
        raise ValueError(f"row-importance artifact contains no entries: {resolved}")
    return RowImportance(resolved, entries, dict(document.get("metadata", {})))


__all__ = ["RowImportance", "load_row_importance", "save_row_importance"]
