"""Global rate-distortion allocation for per-layer precision groups."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from mfq.calibration.allocator import GroupCandidate, allocate
from mfq.calibration.artifact import CalibrationScheme, TensorSelection
from mfq.calibration.evaluator import (
    NINT_CALIBRATION_PROFILES,
    TensorCandidateEvaluation,
    nint_storage_bits,
)
from mfq.formats.nint import NintSpec

_LAYER_GROUP = re.compile(r"^layer\.(\d+)\.")


@dataclass(frozen=True)
class PrecisionOption:
    profile: str
    specs: dict[str, NintSpec]
    storage_bits: int
    surrogate_train_loss: float
    surrogate_validation_loss: float


@dataclass(frozen=True)
class PrecisionGroup:
    name: str
    layer: int
    tensor_names: tuple[str, ...]
    options: tuple[PrecisionOption, ...]
    base_profile: str

    def require(self, profile: str) -> PrecisionOption:
        matches = [item for item in self.options if item.profile == profile]
        if len(matches) != 1:
            raise KeyError(f"precision group {self.name} has no unique profile {profile!r}")
        return matches[0]


@dataclass(frozen=True)
class DiscreteProposal:
    profiles: dict[str, str]
    changed_groups: tuple[str, ...]
    storage_bits: int
    learned_utility_delta: float


@dataclass(frozen=True)
class DiscreteSearchStep:
    iteration: int
    metric: float
    storage_bits: int
    changed_groups: tuple[str, ...]


def build_precision_groups(
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
    scheme: CalibrationScheme,
) -> tuple[PrecisionGroup, ...]:
    """Build one independent decision for each runtime-compatible precision group."""

    by_group: dict[str, list[str]] = defaultdict(list)
    for name, candidates in evaluations.items():
        if not candidates:
            raise ValueError(f"tensor {name} has no precision candidates")
        candidate_groups = {item.group for item in candidates.values()}
        if len(candidate_groups) != 1:
            raise ValueError(f"tensor {name} belongs to multiple precision groups")
        group = next(iter(candidate_groups))
        by_group[group].append(name)

    groups: list[PrecisionGroup] = []
    for group_name, unsorted_names in sorted(by_group.items()):
        match = _LAYER_GROUP.match(group_name)
        if match is None:
            raise ValueError(f"precision group has no decoder-layer prefix: {group_name}")
        names = tuple(sorted(unsorted_names))
        common_profiles = set(evaluations[names[0]])
        for name in names[1:]:
            common_profiles.intersection_update(evaluations[name])
        if not common_profiles:
            raise ValueError(f"precision group {group_name} has no common candidate profile")

        options: list[PrecisionOption] = []
        for profile in sorted(common_profiles):
            values = [evaluations[name][profile] for name in names]
            if any(item.group != group_name for item in values):
                raise ValueError(f"profile {profile} crosses precision groups")
            options.append(
                PrecisionOption(
                    profile=profile,
                    specs={item.name: item.spec for item in values},
                    storage_bits=sum(item.storage_bits for item in values),
                    surrogate_train_loss=sum(item.train_loss for item in values),
                    surrogate_validation_loss=sum(item.validation_loss for item in values),
                )
            )
        options.sort(key=lambda item: (item.storage_bits, item.profile))

        base_matches = [
            item
            for item in options
            if all(scheme.require(name).spec == item.specs[name] for name in names)
        ]
        if len(base_matches) != 1:
            raise ValueError(
                f"base scheme has {len(base_matches)} common profiles for {group_name}"
            )
        groups.append(
            PrecisionGroup(
                name=group_name,
                layer=int(match.group(1)),
                tensor_names=names,
                options=tuple(options),
                base_profile=base_matches[0].profile,
            )
        )
    return tuple(groups)


def build_precision_groups_from_scheme(
    scheme: CalibrationScheme,
    *,
    profiles: Mapping[str, NintSpec] = NINT_CALIBRATION_PROFILES,
) -> tuple[PrecisionGroup, ...]:
    """Reconstruct soft-search groups from an audited allocation artifact."""

    if not scheme.candidate_table:
        raise ValueError("calibration scheme has no candidate table")
    by_group: dict[str, list[str]] = defaultdict(list)
    for name, selection in scheme.selections.items():
        if selection.group in scheme.candidate_table:
            by_group[selection.group].append(name)
    missing = sorted(set(scheme.candidate_table) - set(by_group))
    if missing:
        raise ValueError(f"candidate-table groups have no selected tensors: {missing}")

    groups: list[PrecisionGroup] = []
    for group_name, raw_options in sorted(scheme.candidate_table.items()):
        match = _LAYER_GROUP.match(group_name)
        if match is None:
            raise ValueError(f"precision group has no decoder-layer prefix: {group_name}")
        names = tuple(sorted(by_group[group_name]))
        options: list[PrecisionOption] = []
        seen_profiles: set[str] = set()
        for raw in raw_options:
            profile = str(raw.get("profile", ""))
            if profile in seen_profiles:
                raise ValueError(f"precision group {group_name} repeats profile {profile!r}")
            seen_profiles.add(profile)
            try:
                spec = profiles[profile]
            except KeyError as exc:
                raise ValueError(
                    f"precision group {group_name} uses unknown profile {profile!r}"
                ) from exc
            storage_bits = int(raw["storage_bits"])
            expected_storage_bits = sum(
                nint_storage_bits(
                    scheme.require(name).rows,
                    scheme.require(name).columns,
                    spec,
                )
                for name in names
            )
            if storage_bits != expected_storage_bits:
                raise ValueError(
                    f"precision group {group_name} profile {profile} stores {storage_bits} bits; "
                    f"expected {expected_storage_bits}"
                )
            train_loss = float(raw["train_loss"])
            validation_loss = float(raw["validation_loss"])
            if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
                raise ValueError(
                    f"precision group {group_name} profile {profile} has non-finite loss"
                )
            options.append(
                PrecisionOption(
                    profile=profile,
                    specs={name: spec for name in names},
                    storage_bits=storage_bits,
                    surrogate_train_loss=train_loss,
                    surrogate_validation_loss=validation_loss,
                )
            )
        if not options:
            raise ValueError(f"precision group {group_name} has no candidate options")
        options.sort(key=lambda item: (item.storage_bits, item.profile))
        base_matches = [
            option
            for option in options
            if all(scheme.require(name).spec == option.specs[name] for name in names)
        ]
        if len(base_matches) != 1:
            raise ValueError(
                f"base scheme has {len(base_matches)} common profiles for {group_name}"
            )
        groups.append(
            PrecisionGroup(
                name=group_name,
                layer=int(match.group(1)),
                tensor_names=names,
                options=tuple(options),
                base_profile=base_matches[0].profile,
            )
        )
    return tuple(groups)


def fixed_storage_bits(
    scheme: CalibrationScheme,
    groups: Sequence[PrecisionGroup],
) -> int:
    assigned = {name for group in groups for name in group.tensor_names}
    dense_bits = sum(
        selection.storage_bits
        for name, selection in scheme.selections.items()
        if name not in assigned
    )
    expert_bits = sum(
        expert.storage_bits
        for name, selection in scheme.expert_selections.items()
        for expert in selection.selections
        if f"{name}#expert={expert.expert_id}" not in assigned
    )
    return dense_bits + expert_bits


def initialize_gate_logits(
    groups: Sequence[PrecisionGroup],
    *,
    device: str | torch.device,
    base_logit: float = 1.5,
) -> dict[str, torch.Tensor]:
    if base_logit < 0:
        raise ValueError("base_logit must be non-negative")
    result: dict[str, torch.Tensor] = {}
    for group in groups:
        values = torch.zeros(len(group.options), device=device, dtype=torch.float32)
        base_index = next(
            index
            for index, option in enumerate(group.options)
            if option.profile == group.base_profile
        )
        values[base_index] = float(base_logit)
        result[group.name] = values.requires_grad_(True)
    return result


def group_probabilities(
    group: PrecisionGroup,
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if logits.shape != (len(group.options),):
        raise ValueError(f"logits for {group.name} have shape {tuple(logits.shape)}")
    return torch.softmax(logits / float(temperature), dim=0)


def expected_storage_bits(
    groups: Sequence[PrecisionGroup],
    logits: Mapping[str, torch.Tensor],
    *,
    temperature: float,
    fixed_bits: int = 0,
) -> torch.Tensor:
    if fixed_bits < 0:
        raise ValueError("fixed_bits must be non-negative")
    terms: list[torch.Tensor] = []
    for group in groups:
        probabilities = group_probabilities(group, logits[group.name], temperature)
        costs = probabilities.new_tensor([item.storage_bits for item in group.options])
        terms.append(torch.dot(probabilities, costs))
    if not terms:
        return torch.tensor(float(fixed_bits), dtype=torch.float64)
    return torch.stack(terms).sum() + float(fixed_bits)


def add_rate_gradient_(
    groups: Sequence[PrecisionGroup],
    logits: Mapping[str, torch.Tensor],
    *,
    temperature: float,
    target_storage_bits: int,
    dual: float,
) -> None:
    """Add the gradient of ``dual * expected_bits / target_bits`` in place."""

    if target_storage_bits <= 0:
        raise ValueError("target_storage_bits must be positive")
    if dual < 0:
        raise ValueError("dual must be non-negative")
    for group in groups:
        value = logits[group.name]
        if value.grad is None:
            value.grad = torch.zeros_like(value)
        probabilities = group_probabilities(group, value.detach(), temperature)
        costs = probabilities.new_tensor([item.storage_bits for item in group.options])
        expected = torch.dot(probabilities, costs)
        rate_gradient = probabilities * (costs - expected)
        rate_gradient *= float(dual) / (float(temperature) * target_storage_bits)
        value.grad.add_(rate_gradient)


def update_dual(
    dual: float,
    expected_bits: float,
    target_bits: int,
    *,
    learning_rate: float,
) -> float:
    if dual < 0 or learning_rate < 0:
        raise ValueError("dual and dual learning rate must be non-negative")
    if target_bits <= 0:
        raise ValueError("target_bits must be positive")
    violation = expected_bits / target_bits - 1.0
    return max(0.0, float(dual + learning_rate * violation))


def selected_storage_bits(
    groups: Sequence[PrecisionGroup],
    profiles: Mapping[str, str],
    *,
    fixed_bits: int = 0,
) -> int:
    if set(profiles) != {group.name for group in groups}:
        raise ValueError("selected profiles do not cover every precision group")
    return fixed_bits + sum(group.require(profiles[group.name]).storage_bits for group in groups)


def discretize_gate_logits(
    groups: Sequence[PrecisionGroup],
    logits: Mapping[str, torch.Tensor],
    *,
    target_storage_bits: int,
    fixed_bits: int = 0,
) -> dict[str, str]:
    """Select one profile per group with an exact global storage constraint."""

    group_budget = target_storage_bits - fixed_bits
    if group_budget <= 0:
        raise ValueError("fixed tensors consume the entire target budget")
    candidates: list[GroupCandidate] = []
    for group in groups:
        values = logits[group.name].detach().float().cpu()
        maximum = float(values.max().item())
        for index, option in enumerate(group.options):
            loss = max(0.0, maximum - float(values[index].item()))
            candidates.append(
                GroupCandidate(
                    group=group.name,
                    profile=option.profile,
                    specs=option.specs,
                    storage_bits=option.storage_bits,
                    train_loss=loss,
                    validation_loss=loss,
                )
            )
    result = allocate(candidates, group_budget)
    return {group: item.profile for group, item in result.selected.items()}


def _learned_utility(
    group: PrecisionGroup,
    profile: str,
    logits: Mapping[str, torch.Tensor],
) -> float:
    index = next(index for index, option in enumerate(group.options) if option.profile == profile)
    return float(logits[group.name].detach().float().cpu()[index].item())


def discrete_refinement_proposals(
    groups: Sequence[PrecisionGroup],
    profiles: Mapping[str, str],
    logits: Mapping[str, torch.Tensor],
    *,
    target_storage_bits: int,
    fixed_bits: int = 0,
    max_single: int = 16,
    max_pair: int = 32,
) -> tuple[DiscreteProposal, ...]:
    if max_single < 0 or max_pair < 0:
        raise ValueError("proposal limits must be non-negative")
    current_bits = selected_storage_bits(groups, profiles, fixed_bits=fixed_bits)
    changes: list[tuple[str, str, int, float]] = []
    for group in groups:
        current_profile = profiles[group.name]
        current = group.require(current_profile)
        current_utility = _learned_utility(group, current_profile, logits)
        for option in group.options:
            if option.profile == current_profile:
                continue
            changes.append(
                (
                    group.name,
                    option.profile,
                    option.storage_bits - current.storage_bits,
                    _learned_utility(group, option.profile, logits) - current_utility,
                )
            )

    singles: list[DiscreteProposal] = []
    for group, profile, bit_delta, utility_delta in changes:
        storage = current_bits + bit_delta
        if storage <= target_storage_bits:
            proposal = dict(profiles)
            proposal[group] = profile
            singles.append(DiscreteProposal(proposal, (group,), storage, utility_delta))
    singles.sort(key=lambda item: (-item.learned_utility_delta, item.storage_bits))

    pairs: list[DiscreteProposal] = []
    for left_index, left in enumerate(changes):
        for right in changes[left_index + 1 :]:
            if left[0] == right[0]:
                continue
            storage = current_bits + left[2] + right[2]
            if storage > target_storage_bits:
                continue
            proposal = dict(profiles)
            proposal[left[0]] = left[1]
            proposal[right[0]] = right[1]
            pairs.append(
                DiscreteProposal(
                    proposal,
                    tuple(sorted((left[0], right[0]))),
                    storage,
                    left[3] + right[3],
                )
            )
    pairs.sort(key=lambda item: (-item.learned_utility_delta, item.storage_bits))

    deduplicated: dict[tuple[tuple[str, str], ...], DiscreteProposal] = {}
    for item in [*singles[:max_single], *pairs[:max_pair]]:
        key = tuple(sorted(item.profiles.items()))
        deduplicated.setdefault(key, item)
    return tuple(deduplicated.values())


def refine_discrete_profiles(
    groups: Sequence[PrecisionGroup],
    initial_profiles: Mapping[str, str],
    logits: Mapping[str, torch.Tensor],
    evaluate: Callable[[Mapping[str, str]], float],
    *,
    target_storage_bits: int,
    fixed_bits: int = 0,
    max_iterations: int = 3,
    max_single: int = 16,
    max_pair: int = 32,
) -> tuple[dict[str, str], tuple[DiscreteSearchStep, ...]]:
    """Run exact whole-model checks on shortlisted one- and two-group swaps."""

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    current = dict(initial_profiles)
    current_metric = float(evaluate(current))
    if not torch.isfinite(torch.tensor(current_metric)):
        raise FloatingPointError("initial discrete metric is non-finite")
    history = [
        DiscreteSearchStep(
            iteration=0,
            metric=current_metric,
            storage_bits=selected_storage_bits(groups, current, fixed_bits=fixed_bits),
            changed_groups=(),
        )
    ]
    for iteration in range(1, max_iterations + 1):
        proposals = discrete_refinement_proposals(
            groups,
            current,
            logits,
            target_storage_bits=target_storage_bits,
            fixed_bits=fixed_bits,
            max_single=max_single,
            max_pair=max_pair,
        )
        best_metric = current_metric
        best: DiscreteProposal | None = None
        for proposal in proposals:
            metric = float(evaluate(proposal.profiles))
            if not torch.isfinite(torch.tensor(metric)):
                raise FloatingPointError("discrete proposal metric is non-finite")
            if metric < best_metric:
                best_metric = metric
                best = proposal
        if best is None:
            break
        current = best.profiles
        current_metric = best_metric
        history.append(
            DiscreteSearchStep(
                iteration=iteration,
                metric=current_metric,
                storage_bits=best.storage_bits,
                changed_groups=best.changed_groups,
            )
        )
    return current, tuple(history)


def scheme_from_profiles(
    base_scheme: CalibrationScheme,
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
    groups: Sequence[PrecisionGroup],
    profiles: Mapping[str, str],
    *,
    metadata: Mapping[str, Any],
) -> CalibrationScheme:
    selections = dict(base_scheme.selections)
    for group in groups:
        profile = profiles[group.name]
        option = group.require(profile)
        for name in group.tensor_names:
            value = evaluations[name][profile]
            if value.spec != option.specs[name]:
                raise RuntimeError(f"candidate table changed for {name}/{profile}")
            selections[name] = TensorSelection(
                name=name,
                group=value.group,
                spec=value.spec,
                rows=value.rows,
                columns=value.columns,
                storage_bits=value.storage_bits,
                train_loss=value.train_loss,
                validation_loss=value.validation_loss,
            )
    result = CalibrationScheme(
        path=None,
        target_profile=base_scheme.target_profile,
        target_storage_bits=base_scheme.target_storage_bits,
        selections=selections,
        metadata={**base_scheme.metadata, **metadata},
        candidate_table=base_scheme.candidate_table,
        inint_selector=base_scheme.inint_selector,
        expert_selections=base_scheme.expert_selections,
    )
    if result.storage_bits > result.target_storage_bits:
        raise RuntimeError("rate-distortion scheme exceeds its storage budget")
    return result


__all__ = [
    "DiscreteProposal",
    "DiscreteSearchStep",
    "PrecisionGroup",
    "PrecisionOption",
    "add_rate_gradient_",
    "build_precision_groups",
    "build_precision_groups_from_scheme",
    "discrete_refinement_proposals",
    "discretize_gate_logits",
    "expected_storage_bits",
    "fixed_storage_bits",
    "group_probabilities",
    "initialize_gate_logits",
    "refine_discrete_profiles",
    "scheme_from_profiles",
    "selected_storage_bits",
    "update_dual",
]
