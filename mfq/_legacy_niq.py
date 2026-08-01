"""Compatibility helpers for the pre-NVQ Python API."""

from __future__ import annotations

from types import ModuleType
from typing import Any


def export_niq_aliases(namespace: dict[str, Any], implementation: ModuleType) -> None:
    """Expose a canonical NVQ module under its former NIQ spellings."""

    exported: list[str] = []
    for name, value in vars(implementation).items():
        if name.startswith("_"):
            continue
        namespace[name] = value
        exported.append(name)
        legacy_name = name.replace("NVQ", "NIQ").replace("Nvq", "Niq").replace("nvq", "niq")
        if legacy_name != name:
            namespace[legacy_name] = value
            exported.append(legacy_name)
    namespace["__all__"] = sorted(set(exported))
