"""REAP-conditioned per-expert NINT precision allocation."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mfq.calibration.allocator import GroupCandidate, allocate
from mfq.calibration.artifact import (
    CalibrationScheme,
    ExpertSelection,
    ExpertTensorSelection,
)
from mfq.calibration.evaluator import NINT_EXPERT_PROFILES, nint_storage_bits
from mfq.formats.nint import NintSpec


@dataclass(frozen=True)
class ReapExpertObservation:
    layer: int
    expert: int
    total_tokens: int
    expert_frequency: int
    expert_probability: float
    reap: float
    exposure: float
    normalized_exposure: float


@dataclass(frozen=True)
class ExpertProfileEvaluation:
    layer: int
    expert: int
    profile: str
    spec: NintSpec
    gate_name: str
    down_name: str
    gate_rows: int
    gate_columns: int
    down_rows: int
    down_columns: int
    exposure: float
    normalized_exposure: float
    gate_sse: float
    gate_signal: float
    down_sse: float
    down_signal: float

    @property
    def gate_nmse(self) -> float:
        return self.gate_sse / self.gate_signal

    @property
    def down_nmse(self) -> float:
        return self.down_sse / self.down_signal

    @property
    def loss(self) -> float:
        return self.normalized_exposure * 0.5 * (self.gate_nmse + self.down_nmse)

    @property
    def gate_storage_bits(self) -> int:
        return nint_storage_bits(self.gate_rows, self.gate_columns, self.spec)

    @property
    def down_storage_bits(self) -> int:
        return nint_storage_bits(self.down_rows, self.down_columns, self.spec)

    @property
    def storage_bits(self) -> int:
        return self.gate_storage_bits + self.down_storage_bits

    def as_document(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "expert": self.expert,
            "profile": self.profile,
            "spec": {
                "bits": self.spec.bits,
                "groupsize": self.spec.groupsize,
                "sub_bits": self.spec.sub_bits,
            },
            "gate_name": self.gate_name,
            "down_name": self.down_name,
            "gate_shape": [self.gate_rows, self.gate_columns],
            "down_shape": [self.down_rows, self.down_columns],
            "exposure": self.exposure,
            "normalized_exposure": self.normalized_exposure,
            "gate_sse": self.gate_sse,
            "gate_signal": self.gate_signal,
            "gate_nmse": self.gate_nmse,
            "down_sse": self.down_sse,
            "down_signal": self.down_signal,
            "down_nmse": self.down_nmse,
            "loss": self.loss,
            "gate_storage_bits": self.gate_storage_bits,
            "down_storage_bits": self.down_storage_bits,
            "storage_bits": self.storage_bits,
        }


def load_reap_expert_table(
    path: str | Path,
    *,
    expected_layers: int,
    expected_experts: int,
    expected_top_k: int | None = None,
) -> dict[tuple[int, int], ReapExpertObservation]:
    """Load the public REAP table and normalize exposure within each layer."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"REAP expert table does not exist: {source}")
    raw: dict[tuple[int, int], tuple[int, int, float, float, float]] = {}
    per_layer: dict[int, list[tuple[int, float]]] = defaultdict(list)
    rows: Iterable[tuple[int, int, int, int, float, float]]
    if source.suffix.lower() in {".pkl", ".pt", ".pth"}:
        import torch

        try:
            document = torch.load(source, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"cannot safely load REAP tensor state: {source}") from exc
        if not isinstance(document, Mapping):
            raise ValueError(f"REAP tensor state must be a layer mapping: {source}")
        tensor_rows: list[tuple[int, int, int, int, float, float]] = []
        for layer_key, state in document.items():
            if not isinstance(state, Mapping):
                raise ValueError(f"REAP layer {layer_key!r} is not a mapping")
            try:
                layer = int(layer_key)
                total_tokens = int(state["total_tokens"].item())
                frequency_values = state["expert_frequency"].reshape(-1)
                reap_values = state["reap"].reshape(-1)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"invalid REAP tensor state for layer {layer_key!r}") from exc
            if frequency_values.numel() != expected_experts or reap_values.numel() != expected_experts:
                raise ValueError(
                    f"REAP layer {layer} tensor width differs from {expected_experts} experts"
                )
            for expert in range(expected_experts):
                frequency = int(frequency_values[expert].item())
                probability = frequency / total_tokens if total_tokens > 0 else math.nan
                reap = float(reap_values[expert].item())
                tensor_rows.append(
                    (layer, expert, total_tokens, frequency, probability, reap)
                )
        rows = tensor_rows
    else:
        json_rows: list[tuple[int, int, int, int, float, float]] = []
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    json_rows.append(
                        (
                            int(row["layer"]),
                            int(row["expert"]),
                            int(row["totalTokens"]),
                            int(row["expertFrequency"]),
                            float(row["expertProbability"]),
                            float(row["reap"]),
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid REAP row {line_number} in {source}") from exc
        rows = json_rows

    for layer, expert, total_tokens, frequency, probability, reap in rows:
        key = (layer, expert)
        if key in raw:
            raise ValueError(f"duplicate REAP layer/expert pair {key}")
        if not (0 <= layer < expected_layers and 0 <= expert < expected_experts):
            raise ValueError(f"REAP pair {key} is outside the expected model dimensions")
        if total_tokens <= 0 or frequency < 0:
            raise ValueError(f"invalid token or frequency count for REAP pair {key}")
        if not math.isfinite(probability) or probability < 0:
            raise ValueError(f"invalid expert probability for REAP pair {key}")
        if not math.isfinite(reap) or reap < 0:
            raise ValueError(f"invalid REAP saliency for pair {key}")
        exposure = probability * reap
        raw[key] = (total_tokens, frequency, probability, reap, exposure)
        per_layer[layer].append((expert, exposure))

    expected_count = expected_layers * expected_experts
    if len(raw) != expected_count:
        raise ValueError(f"REAP table has {len(raw)} pairs; expected {expected_count}")
    observations: dict[tuple[int, int], ReapExpertObservation] = {}
    for layer in range(expected_layers):
        values = per_layer.get(layer, [])
        if len(values) != expected_experts:
            raise ValueError(
                f"REAP layer {layer} has {len(values)} experts; expected {expected_experts}"
            )
        layer_exposure = sum(value for _expert, value in values)
        if not math.isfinite(layer_exposure) or layer_exposure <= 0:
            raise ValueError(f"REAP layer {layer} has invalid total exposure")
        probability_sum = sum(raw[(layer, expert)][2] for expert in range(expected_experts))
        if expected_top_k is not None and not math.isclose(
            probability_sum, float(expected_top_k), rel_tol=0.0, abs_tol=2e-5
        ):
            raise ValueError(
                f"REAP layer {layer} probability sum is {probability_sum}; "
                f"expected top-k {expected_top_k}"
            )
        for expert in range(expected_experts):
            total_tokens, frequency, probability, reap, exposure = raw[(layer, expert)]
            observations[(layer, expert)] = ReapExpertObservation(
                layer=layer,
                expert=expert,
                total_tokens=total_tokens,
                expert_frequency=frequency,
                expert_probability=probability,
                reap=reap,
                exposure=exposure,
                normalized_exposure=exposure / layer_exposure,
            )
    return observations


def evaluation_from_document(document: Mapping[str, Any]) -> ExpertProfileEvaluation:
    spec = NintSpec(**document["spec"])
    gate_shape = tuple(int(value) for value in document["gate_shape"])
    down_shape = tuple(int(value) for value in document["down_shape"])
    if len(gate_shape) != 2 or len(down_shape) != 2:
        raise ValueError("expert candidate tensor shapes must be two-dimensional")
    return ExpertProfileEvaluation(
        layer=int(document["layer"]),
        expert=int(document["expert"]),
        profile=str(document["profile"]),
        spec=spec,
        gate_name=str(document["gate_name"]),
        down_name=str(document["down_name"]),
        gate_rows=gate_shape[0],
        gate_columns=gate_shape[1],
        down_rows=down_shape[0],
        down_columns=down_shape[1],
        exposure=float(document["exposure"]),
        normalized_exposure=float(document["normalized_exposure"]),
        gate_sse=float(document["gate_sse"]),
        gate_signal=float(document["gate_signal"]),
        down_sse=float(document["down_sse"]),
        down_signal=float(document["down_signal"]),
    )


def allocate_expert_profiles(
    evaluations: Iterable[ExpertProfileEvaluation],
    *,
    target_profile: str = "NINT5",
    target_storage_bits: int | None = None,
    target_label: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[CalibrationScheme, dict[str, Any]]:
    """Solve the global expert allocation and emit a v2 calibration scheme."""

    values = list(evaluations)
    if not values:
        raise ValueError("no expert profile evaluations were provided")
    by_group: dict[tuple[int, int], list[ExpertProfileEvaluation]] = defaultdict(list)
    for item in values:
        if item.profile not in NINT_EXPERT_PROFILES:
            raise ValueError(f"unsupported expert profile {item.profile}")
        if item.spec != NINT_EXPERT_PROFILES[item.profile]:
            raise ValueError(f"expert profile {item.profile} has a non-canonical NINT spec")
        for scalar_name, scalar in (
            ("exposure", item.exposure),
            ("normalized_exposure", item.normalized_exposure),
            ("gate_sse", item.gate_sse),
            ("gate_signal", item.gate_signal),
            ("down_sse", item.down_sse),
            ("down_signal", item.down_signal),
        ):
            if not math.isfinite(scalar) or scalar < 0:
                raise ValueError(
                    f"invalid {scalar_name} for layer {item.layer} expert {item.expert}"
                )
        if item.gate_signal <= 0 or item.down_signal <= 0:
            raise ValueError(
                f"zero signal for layer {item.layer} expert {item.expert}/{item.profile}"
            )
        by_group[(item.layer, item.expert)].append(item)

    profile_set = {item.profile for item in values}
    for key, group in by_group.items():
        profiles = [item.profile for item in group]
        if len(profiles) != len(set(profiles)):
            raise ValueError(f"duplicate profile for expert group {key}")
        if set(profiles) != profile_set:
            raise ValueError(
                f"expert group {key} has profiles {sorted(profiles)}; "
                f"expected {sorted(profile_set)}"
            )
        gate_names = {item.gate_name for item in group}
        down_names = {item.down_name for item in group}
        if len(gate_names) != 1 or len(down_names) != 1:
            raise ValueError(f"expert group {key} crosses tensor names")

    if target_profile not in profile_set:
        raise ValueError(f"unknown target profile {target_profile}")
    uniform_reference_storage_bits = sum(
        next(item.storage_bits for item in group if item.profile == target_profile)
        for group in by_group.values()
    )
    if target_storage_bits is None:
        target_storage_bits = uniform_reference_storage_bits
    elif target_storage_bits <= 0:
        raise ValueError("target expert storage budget must be positive")
    resolved_target_label = target_label or target_profile
    candidates = [
        GroupCandidate(
            group=f"layer.{item.layer}.expert.{item.expert}",
            profile=item.profile,
            specs={item.gate_name: item.spec, item.down_name: item.spec},
            storage_bits=item.storage_bits,
            train_loss=item.loss,
            validation_loss=item.loss,
        )
        for item in values
    ]
    allocation = allocate(candidates, target_storage_bits)
    selected_values: dict[tuple[int, int], ExpertProfileEvaluation] = {}
    for key, group in by_group.items():
        group_name = f"layer.{key[0]}.expert.{key[1]}"
        profile = allocation.selected[group_name].profile
        selected_values[key] = next(item for item in group if item.profile == profile)

    tensor_items: dict[str, list[ExpertProfileEvaluation]] = defaultdict(list)
    for item in selected_values.values():
        tensor_items[item.gate_name].append(item)
        tensor_items[item.down_name].append(item)
    expert_selections: dict[str, ExpertTensorSelection] = {}
    for name, items in sorted(tensor_items.items()):
        items.sort(key=lambda item: item.expert)
        experts = {item.expert for item in items}
        if experts != set(range(len(items))):
            raise ValueError(f"tensor {name} does not cover a contiguous expert axis")
        is_gate = name == items[0].gate_name
        rows = items[0].gate_rows if is_gate else items[0].down_rows
        columns = items[0].gate_columns if is_gate else items[0].down_columns
        selections: list[ExpertSelection] = []
        for item in items:
            local_nmse = item.gate_nmse if is_gate else item.down_nmse
            local_loss = item.normalized_exposure * 0.5 * local_nmse
            local_storage = item.gate_storage_bits if is_gate else item.down_storage_bits
            selections.append(
                ExpertSelection(
                    expert_id=item.expert,
                    spec=item.spec,
                    storage_bits=local_storage,
                    train_loss=local_loss,
                    validation_loss=local_loss,
                )
            )
        expert_selections[name] = ExpertTensorSelection(
            name=name,
            group=f"layer.{items[0].layer}.routed_experts",
            n_experts=len(items),
            rows_per_expert=rows,
            columns=columns,
            selections=tuple(selections),
        )

    scheme = CalibrationScheme(
        path=None,
        target_profile=f"REAP_EW_{resolved_target_label}",
        target_storage_bits=target_storage_bits,
        selections={},
        metadata={
            "method": "reap-exposure-times-exact-nint-weight-nmse",
            "exposure": "expertProbability * reap, normalized within each layer",
            "expert_coupling": "gate_up and down share one profile",
            "target_label": resolved_target_label,
            "uniform_reference_profile": target_profile,
            "validation": "no held-out activation validation; train_loss and validation_loss are identical offline surrogates",
            **dict(metadata or {}),
        },
        candidate_table={},
        expert_selections=expert_selections,
    )
    if scheme.storage_bits != allocation.actual_storage_bits:
        raise RuntimeError("expert scheme storage differs from the MILP allocation")
    if scheme.storage_bits > target_storage_bits:
        raise RuntimeError("expert scheme exceeds the target storage budget")

    selected_counts = Counter(item.profile for item in selected_values.values())
    per_layer: dict[str, dict[str, int]] = {}
    for layer in sorted({key[0] for key in selected_values}):
        per_layer[str(layer)] = dict(
            sorted(Counter(
                item.profile
                for (item_layer, _expert), item in selected_values.items()
                if item_layer == layer
            ).items())
        )
    baseline_loss = sum(
        next(item.loss for item in group if item.profile == target_profile)
        for group in by_group.values()
    )
    report = {
        "format": "mfq.reap-expert-allocation.v1",
        "solver": allocation.solver,
        "groups": len(by_group),
        "profiles": sorted(profile_set),
        "target_label": resolved_target_label,
        "uniform_reference_profile": target_profile,
        "target_storage_bits": target_storage_bits,
        "actual_storage_bits": scheme.storage_bits,
        "storage_utilization": scheme.storage_bits / target_storage_bits,
        "baseline_loss": baseline_loss,
        "baseline_storage_bits": uniform_reference_storage_bits,
        "baseline_budget_feasible": uniform_reference_storage_bits <= target_storage_bits,
        "selected_loss": allocation.train_loss,
        "relative_surrogate_reduction": (
            (baseline_loss - allocation.train_loss) / baseline_loss
            if baseline_loss and uniform_reference_storage_bits <= target_storage_bits
            else None
        ),
        "selected_counts": dict(sorted(selected_counts.items())),
        "per_layer_counts": per_layer,
    }
    return scheme, report


def allocate_independent_expert_profiles(
    evaluations: Iterable[ExpertProfileEvaluation],
    *,
    target_storage_bits: int,
    baseline_profiles: Mapping[str, str] | None = None,
    target_label: str = "UD_BPW",
    metadata: Mapping[str, Any] | None = None,
) -> tuple[CalibrationScheme, dict[str, Any]]:
    """Allocate gate_up and down precision independently under one expert budget."""

    values = list(evaluations)
    if not values:
        raise ValueError("no expert profile evaluations were provided")
    if target_storage_bits <= 0:
        raise ValueError("target expert storage budget must be positive")

    by_expert: dict[tuple[int, int], list[ExpertProfileEvaluation]] = defaultdict(list)
    profile_set = {item.profile for item in values}
    for item in values:
        if item.profile not in NINT_EXPERT_PROFILES:
            raise ValueError(f"unsupported expert profile {item.profile}")
        if item.spec != NINT_EXPERT_PROFILES[item.profile]:
            raise ValueError(f"expert profile {item.profile} has a non-canonical NINT spec")
        for scalar_name, scalar in (
            ("exposure", item.exposure),
            ("normalized_exposure", item.normalized_exposure),
            ("gate_sse", item.gate_sse),
            ("gate_signal", item.gate_signal),
            ("down_sse", item.down_sse),
            ("down_signal", item.down_signal),
        ):
            if not math.isfinite(scalar) or scalar < 0:
                raise ValueError(
                    f"invalid {scalar_name} for layer {item.layer} expert {item.expert}"
                )
        if item.gate_signal <= 0 or item.down_signal <= 0:
            raise ValueError(
                f"zero signal for layer {item.layer} expert {item.expert}/{item.profile}"
            )
        by_expert[(item.layer, item.expert)].append(item)

    for key, group in by_expert.items():
        profiles = [item.profile for item in group]
        if len(profiles) != len(set(profiles)):
            raise ValueError(f"duplicate profile for expert group {key}")
        if set(profiles) != profile_set:
            raise ValueError(
                f"expert group {key} has profiles {sorted(profiles)}; "
                f"expected {sorted(profile_set)}"
            )
        if len({item.gate_name for item in group}) != 1:
            raise ValueError(f"expert group {key} crosses gate tensor names")
        if len({item.down_name for item in group}) != 1:
            raise ValueError(f"expert group {key} crosses down tensor names")

    candidates: list[GroupCandidate] = []
    candidate_items: dict[tuple[str, str], tuple[ExpertProfileEvaluation, str]] = {}
    for item in values:
        gate_group = f"layer.{item.layer}.expert.{item.expert}.gate"
        down_group = f"layer.{item.layer}.expert.{item.expert}.down"
        gate_loss = item.normalized_exposure * 0.5 * item.gate_nmse
        down_loss = item.normalized_exposure * 0.5 * item.down_nmse
        candidates.append(
            GroupCandidate(
                group=gate_group,
                profile=item.profile,
                specs={item.gate_name: item.spec},
                storage_bits=item.gate_storage_bits,
                train_loss=gate_loss,
                validation_loss=gate_loss,
            )
        )
        candidates.append(
            GroupCandidate(
                group=down_group,
                profile=item.profile,
                specs={item.down_name: item.spec},
                storage_bits=item.down_storage_bits,
                train_loss=down_loss,
                validation_loss=down_loss,
            )
        )
        candidate_items[(gate_group, item.profile)] = (item, "gate")
        candidate_items[(down_group, item.profile)] = (item, "down")

    allocation = allocate(candidates, target_storage_bits)
    selected_by_tensor: dict[
        str, list[tuple[int, ExpertProfileEvaluation, str, float, int]]
    ] = defaultdict(list)
    selected_counts: dict[str, Counter[str]] = {
        "gate": Counter(),
        "down": Counter(),
    }
    for group_name, choice in allocation.selected.items():
        item, kind = candidate_items[(group_name, choice.profile)]
        name = item.gate_name if kind == "gate" else item.down_name
        selected_by_tensor[name].append(
            (item.expert, item, kind, choice.train_loss, choice.storage_bits)
        )
        selected_counts[kind][choice.profile] += 1

    expert_selections: dict[str, ExpertTensorSelection] = {}
    for name, items in sorted(selected_by_tensor.items()):
        items.sort(key=lambda value: value[0])
        experts = {expert for expert, _item, _kind, _loss, _storage in items}
        if experts != set(range(len(items))):
            raise ValueError(f"tensor {name} does not cover a contiguous expert axis")
        first = items[0][1]
        kind = items[0][2]
        if any(item_kind != kind for _expert, _item, item_kind, _loss, _storage in items):
            raise RuntimeError(f"tensor {name} mixes gate and down candidates")
        rows = first.gate_rows if kind == "gate" else first.down_rows
        columns = first.gate_columns if kind == "gate" else first.down_columns
        expert_selections[name] = ExpertTensorSelection(
            name=name,
            group=f"layer.{first.layer}.routed_experts.{kind}",
            n_experts=len(items),
            rows_per_expert=rows,
            columns=columns,
            selections=tuple(
                ExpertSelection(
                    expert_id=expert,
                    spec=item.spec,
                    storage_bits=storage,
                    train_loss=loss,
                    validation_loss=loss,
                )
                for expert, item, _kind, loss, storage in items
            ),
        )

    scheme = CalibrationScheme(
        path=None,
        target_profile=f"REAP_EW_{target_label}",
        target_storage_bits=int(target_storage_bits),
        selections={},
        metadata={
            "method": "reap-exposure-times-exact-nint-weight-nmse",
            "exposure": "expertProbability * reap, normalized within each layer",
            "expert_coupling": "gate_up and down are allocated independently",
            "target_label": target_label,
            "validation": "no held-out activation validation; train_loss and validation_loss are identical offline surrogates",
            **dict(metadata or {}),
        },
        candidate_table={},
        expert_selections=expert_selections,
    )
    if scheme.storage_bits != allocation.actual_storage_bits:
        raise RuntimeError("expert scheme storage differs from the MILP allocation")
    if scheme.storage_bits > target_storage_bits:
        raise RuntimeError("expert scheme exceeds the target storage budget")

    group_names = set(allocation.selected)
    baseline_loss: float | None = None
    baseline_storage_bits: int | None = None
    baseline_counts: dict[str, Counter[str]] | None = None
    if baseline_profiles is not None:
        if set(baseline_profiles) != group_names:
            raise ValueError("baseline profiles do not cover every independent expert group")
        baseline_loss = 0.0
        baseline_storage_bits = 0
        baseline_counts = {"gate": Counter(), "down": Counter()}
        candidate_lookup = {(item.group, item.profile): item for item in candidates}
        for group_name, profile in baseline_profiles.items():
            try:
                baseline = candidate_lookup[(group_name, profile)]
            except KeyError as exc:
                raise ValueError(
                    f"baseline group {group_name} has no profile {profile}"
                ) from exc
            kind = "gate" if group_name.endswith(".gate") else "down"
            baseline_loss += baseline.train_loss
            baseline_storage_bits += baseline.storage_bits
            baseline_counts[kind][profile] += 1

    minimum_by_group: dict[str, int] = {}
    for item in candidates:
        current = minimum_by_group.get(item.group)
        if current is None or item.storage_bits < current:
            minimum_by_group[item.group] = item.storage_bits
    minimum_storage_bits = sum(minimum_by_group.values())
    report: dict[str, Any] = {
        "format": "mfq.reap-independent-expert-allocation.v1",
        "solver": allocation.solver,
        "groups": len(group_names),
        "profiles": sorted(profile_set),
        "target_label": target_label,
        "target_storage_bits": int(target_storage_bits),
        "actual_storage_bits": scheme.storage_bits,
        "storage_utilization": scheme.storage_bits / target_storage_bits,
        "minimum_storage_bits": minimum_storage_bits,
        "selected_loss": allocation.train_loss,
        "selected_counts": {
            kind: dict(sorted(counts.items()))
            for kind, counts in selected_counts.items()
        },
    }
    if baseline_loss is not None and baseline_storage_bits is not None:
        report.update(
            {
                "baseline_loss": baseline_loss,
                "baseline_storage_bits": baseline_storage_bits,
                "relative_surrogate_reduction": (
                    (baseline_loss - allocation.train_loss) / baseline_loss
                    if baseline_loss
                    else 0.0
                ),
                "baseline_counts": {
                    kind: dict(sorted(counts.items()))
                    for kind, counts in (baseline_counts or {}).items()
                },
            }
        )
    return scheme, report


__all__ = [
    "ExpertProfileEvaluation",
    "ReapExpertObservation",
    "allocate_expert_profiles",
    "allocate_independent_expert_profiles",
    "evaluation_from_document",
    "load_reap_expert_table",
]
