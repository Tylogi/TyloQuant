"""Serializable calibration schemes consumed by MFQ conversion tools."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mfq.formats.nint import NintSpec
from mfq.formats.tpq import normalize_tpq_dtype

_FORMAT_V1 = "mfq.calibration-scheme.v1"
_FORMAT_V2 = "mfq.calibration-scheme.v2"
_FORMAT = "mfq.calibration-scheme.v3"

EXPERT_PRECISION_FAMILIES = frozenset(
    {
        "NINT2",
        "NINT3",
        "NINT4",
        "NINT5",
        "NINT6",
        "NINT8",
        "MXFP4",
        "NVQ1-L",
        "NVQ1-S",
        "NPQ0-L",
        "NPQ0-S",
        "NVQ2",
        "NVQ2J",
        "NVQ2J-L",
        "NVQ2J-XL",
        "NVQ3",
        "NVQ3J",
        "NVQ3J-512",
        "NVQ3J-L",
        "NEPQ0-S",
        "NEPQ0-L",
        "NEPQ1-S",
        "NEPQ1-L",
        "TPQ-X",
        "TPQ-W",
        "TPQ-V",
        "TPQ-VV",
    }
)
_OPTION_SCALARS = (str, int, float, bool)


@dataclass(frozen=True)
class ExpertPrecision:
    """One serializable expert precision and its cohort-level quantizer state."""

    family: str
    nint_spec: NintSpec | None = None
    artifact: str | None = None
    options: tuple[tuple[str, str | int | float | bool], ...] = ()

    def __post_init__(self) -> None:
        family = normalize_tpq_dtype(str(self.family))
        if family not in EXPERT_PRECISION_FAMILIES:
            raise ValueError(f"unsupported expert precision family: {family}")
        if family.startswith("NINT"):
            if self.nint_spec is None:
                raise ValueError(f"{family} requires a NINT spec")
            if family != f"NINT{self.nint_spec.bits}":
                raise ValueError(
                    f"expert precision family/spec mismatch: {family}/{self.nint_spec}"
                )
        elif self.nint_spec is not None:
            raise ValueError(f"{family} cannot carry a NINT spec")
        if self.artifact is not None and not str(self.artifact):
            raise ValueError("expert precision artifact path cannot be empty")
        normalized: list[tuple[str, str | int | float | bool]] = []
        seen: set[str] = set()
        for raw_key, value in self.options:
            key = str(raw_key)
            if not key or key in seen:
                raise ValueError(f"invalid or duplicate expert precision option: {key!r}")
            if not isinstance(value, _OPTION_SCALARS):
                raise TypeError(
                    f"expert precision option {key!r} must be a JSON scalar"
                )
            seen.add(key)
            normalized.append((key, value))
        object.__setattr__(self, "family", family)
        object.__setattr__(
            self,
            "artifact",
            None if self.artifact is None else str(self.artifact),
        )
        object.__setattr__(self, "options", tuple(sorted(normalized)))

    def option(self, name: str, default: Any = None) -> Any:
        return dict(self.options).get(name, default)


def nint_expert_precision(spec: NintSpec) -> ExpertPrecision:
    return ExpertPrecision(family=f"NINT{spec.bits}", nint_spec=spec)


@dataclass(frozen=True)
class TensorSelection:
    name: str
    group: str
    spec: NintSpec
    rows: int
    columns: int
    storage_bits: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True)
class ExpertSelection:
    """One expert's independently selected compact precision profile."""

    expert_id: int
    spec: NintSpec | None
    storage_bits: int
    train_loss: float
    validation_loss: float
    precision: ExpertPrecision | None = None

    @property
    def descriptor(self) -> ExpertPrecision:
        if self.precision is None:
            if self.spec is None:
                raise ValueError(f"expert {self.expert_id} has no precision descriptor")
            return nint_expert_precision(self.spec)
        if self.spec is not None and self.precision.nint_spec != self.spec:
            raise ValueError(
                f"expert {self.expert_id} has conflicting NINT and family descriptors"
            )
        return self.precision


@dataclass(frozen=True)
class ExpertTensorSelection:
    """Per-expert selections for one logical ``[E, O, K]`` weight tensor."""

    name: str
    group: str
    n_experts: int
    rows_per_expert: int
    columns: int
    selections: tuple[ExpertSelection, ...]

    def __post_init__(self) -> None:
        if self.n_experts <= 0 or self.rows_per_expert <= 0 or self.columns <= 0:
            raise ValueError("expert tensor dimensions must be positive")
        if len(self.selections) != self.n_experts:
            raise ValueError(
                f"expert tensor {self.name!r} has {len(self.selections)} selections; "
                f"expected {self.n_experts}"
            )
        owners = [False] * self.n_experts
        for item in self.selections:
            if item.expert_id < 0 or item.expert_id >= self.n_experts:
                raise ValueError(f"expert id {item.expert_id} is outside [0, {self.n_experts})")
            if owners[item.expert_id]:
                raise ValueError(f"expert {item.expert_id} has more than one precision selection")
            if item.storage_bits <= 0:
                raise ValueError(f"expert {item.expert_id} has invalid storage_bits")
            item.descriptor
            owners[item.expert_id] = True

    @property
    def storage_bits(self) -> int:
        return int(sum(item.storage_bits for item in self.selections))

    @property
    def weight_count(self) -> int:
        return int(self.n_experts * self.rows_per_expert * self.columns)

    @property
    def specs(self) -> tuple[NintSpec, ...]:
        precisions = self.precisions
        if any(item.nint_spec is None for item in precisions):
            raise TypeError(f"expert tensor {self.name!r} contains non-NINT precisions")
        return tuple(item.nint_spec for item in precisions if item.nint_spec is not None)

    @property
    def precisions(self) -> tuple[ExpertPrecision, ...]:
        ordered: list[ExpertPrecision | None] = [None] * self.n_experts
        for item in self.selections:
            ordered[item.expert_id] = item.descriptor
        return tuple(precision for precision in ordered if precision is not None)


@dataclass(frozen=True)
class CalibrationScheme:
    path: Path | None
    target_profile: str
    target_storage_bits: int
    selections: dict[str, TensorSelection]
    metadata: dict[str, Any]
    candidate_table: dict[str, list[dict[str, Any]]]
    inint_selector: str | None = None
    expert_selections: dict[str, ExpertTensorSelection] = field(default_factory=dict)

    def require(self, name: str) -> TensorSelection:
        try:
            return self.selections[name]
        except KeyError as exc:
            raise KeyError(f"calibration scheme has no tensor {name!r}") from exc

    def require_expert(self, name: str) -> ExpertTensorSelection:
        try:
            return self.expert_selections[name]
        except KeyError as exc:
            raise KeyError(f"calibration scheme has no expert tensor {name!r}") from exc

    @property
    def storage_bits(self) -> int:
        return int(
            sum(item.storage_bits for item in self.selections.values())
            + sum(item.storage_bits for item in self.expert_selections.values())
        )

    @property
    def weight_count(self) -> int:
        return int(
            sum(item.rows * item.columns for item in self.selections.values())
            + sum(item.weight_count for item in self.expert_selections.values())
        )

    @property
    def bpw(self) -> float:
        return self.storage_bits / self.weight_count if self.weight_count else 0.0


def _spec_document(spec: NintSpec) -> dict[str, int]:
    return {
        "bits": int(spec.bits),
        "groupsize": int(spec.groupsize),
        "sub_bits": int(spec.sub_bits),
    }


def _precision_document(precision: ExpertPrecision) -> dict[str, Any]:
    result: dict[str, Any] = {"family": precision.family}
    if precision.nint_spec is not None:
        result["nint_spec"] = _spec_document(precision.nint_spec)
    if precision.artifact is not None:
        result["artifact"] = precision.artifact
    if precision.options:
        result["options"] = dict(precision.options)
    return result


def _precision_from_document(raw: Mapping[str, Any]) -> ExpertPrecision:
    nint_raw = raw.get("nint_spec")
    options_raw = raw.get("options", {})
    if not isinstance(options_raw, Mapping):
        raise ValueError("expert precision options must be an object")
    return ExpertPrecision(
        family=str(raw["family"]),
        nint_spec=None if nint_raw is None else NintSpec(**nint_raw),
        artifact=None if raw.get("artifact") is None else str(raw["artifact"]),
        options=tuple((str(key), value) for key, value in options_raw.items()),
    )


def save_scheme(path: str | Path, scheme: CalibrationScheme) -> None:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"calibration scheme already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "format": _FORMAT,
        "target_profile": scheme.target_profile,
        "target_storage_bits": int(scheme.target_storage_bits),
        "actual_storage_bits": scheme.storage_bits,
        "weight_count": scheme.weight_count,
        "bpw": scheme.bpw,
        "inint_selector": scheme.inint_selector,
        "metadata": scheme.metadata,
        "selections": {
            name: {
                "group": item.group,
                "spec": _spec_document(item.spec),
                "rows": int(item.rows),
                "columns": int(item.columns),
                "storage_bits": int(item.storage_bits),
                "train_loss": float(item.train_loss),
                "validation_loss": float(item.validation_loss),
            }
            for name, item in sorted(scheme.selections.items())
        },
        "expert_selections": {
            name: {
                "group": item.group,
                "n_experts": int(item.n_experts),
                "rows_per_expert": int(item.rows_per_expert),
                "columns": int(item.columns),
                "experts": [
                    {
                        "expert_id": int(expert.expert_id),
                        "precision": _precision_document(expert.descriptor),
                        "storage_bits": int(expert.storage_bits),
                        "train_loss": float(expert.train_loss),
                        "validation_loss": float(expert.validation_loss),
                    }
                    for expert in sorted(item.selections, key=lambda value: value.expert_id)
                ],
            }
            for name, item in sorted(scheme.expert_selections.items())
        },
        "candidate_table": scheme.candidate_table,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output)


def load_scheme(path: str | Path) -> CalibrationScheme:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"calibration scheme does not exist: {resolved}")
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if document.get("format") not in {_FORMAT_V1, _FORMAT_V2, _FORMAT}:
        raise ValueError(f"unsupported calibration scheme format: {document.get('format')!r}")
    selections: dict[str, TensorSelection] = {}
    for name, raw in document.get("selections", {}).items():
        spec = NintSpec(**raw["spec"])
        item = TensorSelection(
            name=str(name),
            group=str(raw["group"]),
            spec=spec,
            rows=int(raw["rows"]),
            columns=int(raw["columns"]),
            storage_bits=int(raw["storage_bits"]),
            train_loss=float(raw["train_loss"]),
            validation_loss=float(raw["validation_loss"]),
        )
        if item.rows <= 0 or item.columns <= 0 or item.storage_bits <= 0:
            raise ValueError(f"invalid tensor selection for {name}")
        selections[str(name)] = item
    expert_selections: dict[str, ExpertTensorSelection] = {}
    for name, raw in document.get("expert_selections", {}).items():
        experts_list: list[ExpertSelection] = []
        for expert in raw.get("experts", []):
            if "precision" in expert:
                precision = _precision_from_document(expert["precision"])
                spec = precision.nint_spec
            else:
                spec = NintSpec(**expert["spec"])
                precision = nint_expert_precision(spec)
            experts_list.append(
                ExpertSelection(
                    expert_id=int(expert["expert_id"]),
                    spec=spec,
                    storage_bits=int(expert["storage_bits"]),
                    train_loss=float(expert["train_loss"]),
                    validation_loss=float(expert["validation_loss"]),
                    precision=precision,
                )
            )
        experts = tuple(experts_list)
        item = ExpertTensorSelection(
            name=str(name),
            group=str(raw["group"]),
            n_experts=int(raw["n_experts"]),
            rows_per_expert=int(raw["rows_per_expert"]),
            columns=int(raw["columns"]),
            selections=experts,
        )
        expert_selections[str(name)] = item
    overlap = sorted(set(selections) & set(expert_selections))
    if overlap:
        raise ValueError(f"tensors have both uniform and expert-wise selections: {overlap[:8]}")
    if not selections and not expert_selections:
        raise ValueError(f"calibration scheme contains no selections: {resolved}")
    scheme = CalibrationScheme(
        path=resolved,
        target_profile=str(document["target_profile"]),
        target_storage_bits=int(document["target_storage_bits"]),
        selections=selections,
        metadata=dict(document.get("metadata", {})),
        candidate_table={
            str(key): list(value) for key, value in document.get("candidate_table", {}).items()
        },
        inint_selector=document.get("inint_selector"),
        expert_selections=expert_selections,
    )
    if scheme.storage_bits != int(document["actual_storage_bits"]):
        raise ValueError("calibration scheme storage total does not match its selections")
    return scheme


def scheme_specs(scheme: CalibrationScheme) -> Mapping[str, NintSpec]:
    return {name: item.spec for name, item in scheme.selections.items()}


def scheme_expert_specs(scheme: CalibrationScheme) -> Mapping[str, tuple[NintSpec, ...]]:
    return {name: item.specs for name, item in scheme.expert_selections.items()}


def scheme_expert_precisions(
    scheme: CalibrationScheme,
) -> Mapping[str, tuple[ExpertPrecision, ...]]:
    return {name: item.precisions for name, item in scheme.expert_selections.items()}


__all__ = [
    "CalibrationScheme",
    "EXPERT_PRECISION_FAMILIES",
    "ExpertPrecision",
    "ExpertSelection",
    "ExpertTensorSelection",
    "TensorSelection",
    "load_scheme",
    "nint_expert_precision",
    "save_scheme",
    "scheme_expert_precisions",
    "scheme_expert_specs",
    "scheme_specs",
]
