"""Validated categorical per-expert sensitivity maps."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_ROW = re.compile(
    r'^\s*"(?P<layer>\d+)"\s*:\s*"(?P<marks>[Vvw]+)"\s*,?\s*$'
)
_STRUCTURAL_LINES = {"", "{", "}", "},", "};"}


@dataclass(frozen=True)
class ExpertSensitivityMap:
    """One categorical mark for every ``(layer, expert)`` pair."""

    layers: tuple[str, ...]

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def n_experts(self) -> int:
        return len(self.layers[0])

    def mark(self, layer: int, expert: int) -> str:
        return self.layers[int(layer)][int(expert)]

    def experts(self, layer: int, mark: str) -> tuple[int, ...]:
        if mark not in {"V", "v", "w"}:
            raise ValueError(f"unsupported expert sensitivity mark: {mark!r}")
        return tuple(
            expert
            for expert, value in enumerate(self.layers[int(layer)])
            if value == mark
        )

    def count(self, mark: str) -> int:
        if mark not in {"V", "v", "w"}:
            raise ValueError(f"unsupported expert sensitivity mark: {mark!r}")
        return sum(layer.count(mark) for layer in self.layers)


def load_expert_sensitivity_map(
    path: str | Path,
    *,
    expected_layers: int,
    expected_experts: int,
) -> ExpertSensitivityMap:
    """Parse a JSON-object fragment and reject incomplete or ambiguous maps."""

    source = Path(path)
    rows: dict[int, str] = {}
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if stripped in _STRUCTURAL_LINES:
            continue
        match = _ROW.fullmatch(line)
        if match is None:
            raise ValueError(
                f"invalid expert sensitivity row {line_number} in {source}"
            )
        layer = int(match.group("layer"))
        marks = match.group("marks")
        if layer in rows:
            raise ValueError(f"duplicate expert sensitivity layer: {layer}")
        if len(marks) != expected_experts:
            raise ValueError(
                f"expert sensitivity layer {layer} has {len(marks)} marks; "
                f"expected {expected_experts}"
            )
        rows[layer] = marks

    expected = set(range(expected_layers))
    actual = set(rows)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "expert sensitivity layer coverage mismatch: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    return ExpertSensitivityMap(
        layers=tuple(rows[layer] for layer in range(expected_layers))
    )


__all__ = ["ExpertSensitivityMap", "load_expert_sensitivity_map"]
