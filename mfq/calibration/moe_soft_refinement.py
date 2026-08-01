"""Expert-wise precision groups for end-to-end MoE KL calibration."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mfq.calibration.artifact import (
    CalibrationScheme,
    ExpertSelection,
    ExpertTensorSelection,
    nint_expert_precision,
)
from mfq.calibration.rate_distortion import PrecisionGroup, PrecisionOption
from mfq.formats.nint import NintSpec


@dataclass(frozen=True)
class ExpertCandidate:
    layer: int
    expert: int
    profile: str
    spec: NintSpec
    gate_name: str
    down_name: str
    gate_shape: tuple[int, int]
    down_shape: tuple[int, int]
    gate_storage_bits: int
    down_storage_bits: int
    gate_loss: float
    down_loss: float

    @property
    def storage_bits(self) -> int:
        return self.gate_storage_bits + self.down_storage_bits

    @property
    def loss(self) -> float:
        return self.gate_loss + self.down_loss


def expert_group_name(layer: int, expert: int) -> str:
    return f"layer.{layer}.expert.{expert}.gate_up_down"


def expert_member_name(tensor_name: str, expert: int) -> str:
    return f"{tensor_name}#expert={expert}"


def _spec(raw: Mapping[str, Any]) -> NintSpec:
    return NintSpec(
        bits=int(raw["bits"]),
        groupsize=int(raw["groupsize"]),
        sub_bits=int(raw["sub_bits"]),
    )


def _ordered_experts(selection: ExpertTensorSelection) -> tuple[ExpertSelection, ...]:
    ordered: list[ExpertSelection | None] = [None] * selection.n_experts
    for item in selection.selections:
        ordered[item.expert_id] = item
    if any(item is None for item in ordered):
        raise ValueError(f"expert tensor {selection.name!r} has incomplete expert IDs")
    return tuple(item for item in ordered if item is not None)


@dataclass(frozen=True)
class CoupledExpertPrecisionProblem:
    """All coupled gate/up+down choices used by the generic soft-KL optimizer."""

    candidate_path: Path
    candidate_sha256: str
    groups: tuple[PrecisionGroup, ...]
    records: dict[tuple[int, int, str], ExpertCandidate]
    tensors_by_layer: dict[int, tuple[str, str]]

    def candidate(self, layer: int, expert: int, profile: str) -> ExpertCandidate:
        try:
            return self.records[(layer, expert, profile)]
        except KeyError as exc:
            raise KeyError(
                f"missing MoE candidate layer={layer} expert={expert} profile={profile!r}"
            ) from exc

    def build_scheme(
        self,
        base_scheme: CalibrationScheme,
        groups: Sequence[PrecisionGroup],
        profiles: Mapping[str, str],
        metadata: Mapping[str, Any],
    ) -> CalibrationScheme:
        expected_groups = {group.name for group in self.groups}
        if {group.name for group in groups} != expected_groups:
            raise ValueError("MoE scheme builder received a different precision-group set")
        if set(profiles) != expected_groups:
            raise ValueError("selected MoE profiles do not cover every expert group")

        selected_by_tensor: dict[str, list[ExpertSelection | None]] = {}
        for name, tensor in base_scheme.expert_selections.items():
            selected_by_tensor[name] = [None] * tensor.n_experts

        for group in self.groups:
            profile = profiles[group.name]
            record = self.candidate(group.layer, _group_expert(group), profile)
            gate = selected_by_tensor[record.gate_name]
            down = selected_by_tensor[record.down_name]
            precision = nint_expert_precision(record.spec)
            gate[record.expert] = ExpertSelection(
                expert_id=record.expert,
                spec=record.spec,
                precision=precision,
                storage_bits=record.gate_storage_bits,
                train_loss=record.gate_loss,
                validation_loss=record.gate_loss,
            )
            down[record.expert] = ExpertSelection(
                expert_id=record.expert,
                spec=record.spec,
                precision=precision,
                storage_bits=record.down_storage_bits,
                train_loss=record.down_loss,
                validation_loss=record.down_loss,
            )

        expert_selections: dict[str, ExpertTensorSelection] = {}
        for name, base in base_scheme.expert_selections.items():
            values = selected_by_tensor[name]
            if any(item is None for item in values):
                raise RuntimeError(f"soft-KL output did not assign every expert in {name}")
            expert_selections[name] = ExpertTensorSelection(
                name=base.name,
                group=base.group,
                n_experts=base.n_experts,
                rows_per_expert=base.rows_per_expert,
                columns=base.columns,
                selections=tuple(item for item in values if item is not None),
            )

        result = CalibrationScheme(
            path=None,
            target_profile=f"SOFT_KL_{base_scheme.target_profile}",
            target_storage_bits=base_scheme.target_storage_bits,
            selections=dict(base_scheme.selections),
            metadata={
                **base_scheme.metadata,
                **metadata,
                "moe_soft_kl_candidates": {
                    "path": str(self.candidate_path),
                    "sha256": self.candidate_sha256,
                    "coupling": "gate_up_and_down_share_profile_per_expert",
                },
            },
            candidate_table=base_scheme.candidate_table,
            inint_selector=base_scheme.inint_selector,
            expert_selections=expert_selections,
        )
        if result.storage_bits > result.target_storage_bits:
            raise RuntimeError(
                f"soft-KL MoE scheme exceeds storage budget: "
                f"{result.storage_bits} > {result.target_storage_bits}"
            )
        return result


def _group_expert(group: PrecisionGroup) -> int:
    marker = ".expert."
    try:
        return int(group.name.split(marker, 1)[1].split(".", 1)[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid expert precision group name: {group.name!r}") from exc


def load_coupled_expert_precision_problem(
    candidate_path: str | Path,
    base_scheme: CalibrationScheme,
) -> CoupledExpertPrecisionProblem:
    """Load the full per-expert candidate table and bind it to an EW base scheme."""

    path = Path(candidate_path).resolve()
    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    records: dict[tuple[int, int, str], ExpertCandidate] = {}
    by_expert: dict[tuple[int, int], list[ExpertCandidate]] = defaultdict(list)
    tensors_by_layer: dict[int, tuple[str, str]] = {}

    for line_number, raw_line in enumerate(raw_bytes.splitlines(), start=1):
        if not raw_line.strip():
            continue
        raw = json.loads(raw_line)
        layer = int(raw["layer"])
        expert = int(raw["expert"])
        profile = str(raw["profile"])
        normalized_exposure = float(raw["normalized_exposure"])
        gate_nmse = float(raw["gate_nmse"])
        down_nmse = float(raw["down_nmse"])
        values = (
            normalized_exposure,
            gate_nmse,
            down_nmse,
            float(raw["loss"]),
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"invalid candidate loss at {path}:{line_number}")
        gate_name = str(raw["gate_name"])
        down_name = str(raw["down_name"])
        pair = (gate_name, down_name)
        previous = tensors_by_layer.setdefault(layer, pair)
        if previous != pair:
            raise ValueError(f"layer {layer} has inconsistent expert tensor names")
        record = ExpertCandidate(
            layer=layer,
            expert=expert,
            profile=profile,
            spec=_spec(raw["spec"]),
            gate_name=gate_name,
            down_name=down_name,
            gate_shape=tuple(int(value) for value in raw["gate_shape"]),
            down_shape=tuple(int(value) for value in raw["down_shape"]),
            gate_storage_bits=int(raw["gate_storage_bits"]),
            down_storage_bits=int(raw["down_storage_bits"]),
            # The source table defines combined loss as the mean of the two
            # projection NMSE terms, weighted by routed exposure.
            gate_loss=0.5 * normalized_exposure * gate_nmse,
            down_loss=0.5 * normalized_exposure * down_nmse,
        )
        key = (layer, expert, profile)
        if key in records:
            raise ValueError(f"duplicate candidate {key} in {path}")
        if min(record.gate_storage_bits, record.down_storage_bits) <= 0:
            raise ValueError(f"candidate {key} has non-positive storage")
        if abs(record.loss - float(raw["loss"])) > max(1e-12, 1e-6 * float(raw["loss"])):
            raise ValueError(f"candidate {key} loss decomposition disagrees with source")
        records[key] = record
        by_expert[(layer, expert)].append(record)

    if not records:
        raise ValueError(f"empty expert candidate table: {path}")

    groups: list[PrecisionGroup] = []
    for (layer, expert), candidates in sorted(by_expert.items()):
        gate_name, down_name = tensors_by_layer[layer]
        try:
            gate_base = base_scheme.require_expert(gate_name)
            down_base = base_scheme.require_expert(down_name)
        except KeyError as exc:
            raise ValueError(
                f"base scheme has no routed-expert tensors for layer {layer}"
            ) from exc
        if gate_base.n_experts != down_base.n_experts:
            raise ValueError(f"layer {layer} gate/down expert counts disagree")
        if expert < 0 or expert >= gate_base.n_experts:
            raise ValueError(f"candidate expert {expert} is out of range at layer {layer}")

        gate_selected = _ordered_experts(gate_base)[expert].descriptor
        down_selected = _ordered_experts(down_base)[expert].descriptor
        if gate_selected != down_selected:
            raise ValueError(
                f"coupled base scheme disagrees at layer {layer} expert {expert}: "
                f"{gate_selected.family} vs {down_selected.family}"
            )
        base_spec = gate_selected.nint_spec
        if base_spec is None:
            raise TypeError("end-to-end MoE soft calibration currently requires NINT candidates")

        options = tuple(
            PrecisionOption(
                profile=item.profile,
                specs={
                    expert_member_name(gate_name, expert): item.spec,
                    expert_member_name(down_name, expert): item.spec,
                },
                storage_bits=item.storage_bits,
                surrogate_train_loss=item.loss,
                surrogate_validation_loss=item.loss,
            )
            for item in sorted(
                candidates,
                key=lambda value: (value.storage_bits, value.profile),
            )
        )
        if len({item.profile for item in options}) != len(options):
            raise ValueError(f"duplicate profiles at layer {layer} expert {expert}")
        base_profiles = [
            item.profile
            for item in candidates
            if item.spec == base_spec
        ]
        if len(base_profiles) != 1:
            raise ValueError(
                f"base precision has {len(base_profiles)} candidate matches at "
                f"layer {layer} expert {expert}"
            )
        groups.append(
            PrecisionGroup(
                name=expert_group_name(layer, expert),
                layer=layer,
                tensor_names=(
                    expert_member_name(gate_name, expert),
                    expert_member_name(down_name, expert),
                ),
                options=options,
                base_profile=base_profiles[0],
            )
        )

    layers = set(tensors_by_layer)
    if {group.layer for group in groups} != layers:
        raise RuntimeError("expert candidate groups do not cover every source layer")
    for layer, (gate_name, down_name) in tensors_by_layer.items():
        expected = base_scheme.require_expert(gate_name).n_experts
        actual = sum(group.layer == layer for group in groups)
        if actual != expected:
            raise ValueError(
                f"layer {layer} has {actual} candidate expert groups; expected {expected}"
            )
        if base_scheme.require_expert(down_name).n_experts != expected:
            raise ValueError(f"layer {layer} down tensor has a different expert count")

    return CoupledExpertPrecisionProblem(
        candidate_path=path,
        candidate_sha256=digest,
        groups=tuple(groups),
        records=records,
        tensors_by_layer=tensors_by_layer,
    )


__all__ = [
    "CoupledExpertPrecisionProblem",
    "ExpertCandidate",
    "expert_group_name",
    "expert_member_name",
    "load_coupled_expert_precision_problem",
]
