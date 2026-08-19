"""Cumulative layerwise strategy refinement on real hidden states."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from mfq.calibration.allocator import AllocationResult, GroupCandidate, allocate
from mfq.calibration.artifact import (
    CalibrationScheme,
    TensorSelection,
    load_scheme,
    precision_document,
    save_scheme,
)
from mfq.calibration.collector import HiddenStateStore, HiddenTrace, LayerwiseBackend
from mfq.calibration.dataset import CalibrationBatch, CalibrationCorpus
from mfq.calibration.dense_precision import DensePrecision, normalize_dense_precision
from mfq.calibration.evaluator import (
    LayerStrategy,
    TensorCandidateEvaluation,
    build_global_layer_strategies,
)


class RefinementBackend(LayerwiseBackend, Protocol):
    def prepare_layer_strategies(
        self,
        layer_index: int,
        strategies: Sequence[Mapping[str, DensePrecision]],
    ) -> None: ...

    def layer_for_strategy(
        self,
        layer_index: int,
        specs: Mapping[str, DensePrecision],
    ): ...

    def clear_quantized_cache(self, layer_index: int | None = None) -> None: ...


@dataclass(frozen=True)
class StrategyTrace:
    strategy: LayerStrategy
    trace: HiddenTrace

    def document(self) -> dict[str, Any]:
        return {
            "name": self.strategy.name,
            "storage_bits": self.strategy.storage_bits,
            "surrogate_train_loss": self.strategy.train_loss,
            "surrogate_validation_loss": self.strategy.validation_loss,
            **self.trace.document(),
        }


@dataclass(frozen=True)
class ParetoKneeSelection:
    selected: StrategyTrace
    unconstrained: StrategyTrace
    frontier: tuple[StrategyTrace, ...]
    selected_score: float


@dataclass(frozen=True)
class LayerRefinement:
    layer: int
    selected: LayerStrategy
    candidates: tuple[StrategyTrace, ...]
    base_storage_bits: int | None = None
    target_storage_bits: int | None = None
    available_storage_bits: int | None = None
    remaining_credit_bits: int | None = None
    pareto_frontier: tuple[str, ...] = ()
    unconstrained_knee: str | None = None
    knee_score: float | None = None


@dataclass(frozen=True)
class GlobalRefinementAttempt:
    max_changed_layers: int
    proposed_storage_bits: int
    current_normalized_squared_error: float
    predicted_normalized_squared_error: float
    replayed_normalized_squared_error: float
    changed_layers: tuple[int, ...]
    accepted: bool


@dataclass(frozen=True)
class GlobalRefinementRound:
    iteration: int
    input_storage_bits: int
    output_storage_bits: int
    normalized_squared_error: float
    changed_layers: tuple[int, ...]
    converged: bool
    layers: tuple[LayerRefinement, ...]
    solver: str
    attempts: tuple[GlobalRefinementAttempt, ...]


def _layout(batches: Sequence[CalibrationBatch]) -> list[tuple[CalibrationBatch, int, int]]:
    result = []
    cursor = 0
    for batch in batches:
        count = int(batch.input_ids.size)
        result.append((batch, cursor, cursor + count))
        cursor += count
    return result


def _accumulate(reference: torch.Tensor, value: torch.Tensor, totals: np.ndarray) -> None:
    reference = reference.float()
    value = value.float()
    difference = value - reference
    totals += np.asarray(
        [
            reference.square().sum(dtype=torch.float64).item(),
            value.square().sum(dtype=torch.float64).item(),
            difference.square().sum(dtype=torch.float64).item(),
            (reference * value).sum(dtype=torch.float64).item(),
            reference.numel(),
        ],
        dtype=np.float64,
    )


def _trace(layer: int, totals: np.ndarray) -> HiddenTrace:
    return HiddenTrace(
        layer=layer,
        reference_energy=float(totals[0]),
        quantized_energy=float(totals[1]),
        squared_error=float(totals[2]),
        dot_product=float(totals[3]),
        value_count=int(totals[4]),
    )


def _select_pareto_knee(
    candidates: Sequence[StrategyTrace],
    available_storage_bits: int,
) -> ParetoKneeSelection:
    if available_storage_bits <= 0:
        raise ValueError("available_storage_bits must be positive")
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.strategy.storage_bits,
            item.trace.squared_error,
            item.strategy.name,
        ),
    )
    frontier: list[StrategyTrace] = []
    best_error = float("inf")
    for item in ordered:
        if item.trace.squared_error < best_error:
            frontier.append(item)
            best_error = item.trace.squared_error
    if not frontier:
        raise ValueError("cannot select a Pareto knee from no candidates")

    minimum_bits = float(frontier[0].strategy.storage_bits)
    maximum_bits = float(frontier[-1].strategy.storage_bits)
    maximum_error = float(frontier[0].trace.squared_error)
    minimum_error = float(frontier[-1].trace.squared_error)
    bit_range = maximum_bits - minimum_bits
    error_range = maximum_error - minimum_error

    scored: list[tuple[StrategyTrace, float]] = []
    for item in frontier:
        normalized_bits = (
            (item.strategy.storage_bits - minimum_bits) / bit_range if bit_range > 0 else 0.0
        )
        normalized_error = (
            (item.trace.squared_error - minimum_error) / error_range if error_range > 0 else 0.0
        )
        scored.append((item, 1.0 - normalized_bits - normalized_error))

    def rank(value: tuple[StrategyTrace, float]) -> tuple[float, int, str]:
        item, score = value
        return (-score, item.strategy.storage_bits, item.strategy.name)

    unconstrained, _unconstrained_score = min(scored, key=rank)
    feasible = [
        value for value in scored if value[0].strategy.storage_bits <= available_storage_bits
    ]
    if not feasible:
        raise RuntimeError(
            f"no real Pareto strategy fits the {available_storage_bits}-bit layer budget"
        )
    selected, selected_score = min(feasible, key=rank)
    return ParetoKneeSelection(
        selected=selected,
        unconstrained=unconstrained,
        frontier=tuple(frontier),
        selected_score=float(selected_score),
    )


def _selection_for_spec(
    name: str,
    spec: DensePrecision,
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
) -> TensorSelection:
    descriptor = normalize_dense_precision(spec)
    matches = [
        value for value in evaluations[name].values()
        if value.descriptor == descriptor
    ]
    if len(matches) != 1:
        raise ValueError(
            f"tensor {name} has {len(matches)} candidates for precision "
            f"{descriptor.family}"
        )
    value = matches[0]
    return TensorSelection(
        name=name,
        group=value.group,
        spec=value.spec,
        rows=value.rows,
        columns=value.columns,
        storage_bits=value.storage_bits,
        train_loss=value.train_loss,
        validation_loss=value.validation_loss,
        precision=value.precision,
    )


def _target_profile_layer_budgets(
    base_scheme: CalibrationScheme,
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
    strategies: Mapping[int, Sequence[LayerStrategy]],
) -> tuple[dict[int, int], int]:
    target_profile = base_scheme.target_profile.upper()
    budgets: dict[int, int] = {}
    covered_names: set[str] = set()
    for layer, layer_strategies in strategies.items():
        if not layer_strategies:
            raise ValueError(f"layer {layer} has no refinement strategies")
        layer_names = set(layer_strategies[0].specs)
        covered_names.update(layer_names)
        budget = 0
        for name in layer_names:
            choices = evaluations.get(name)
            if choices is None or target_profile not in choices:
                raise ValueError(f"tensor {name} has no target profile {target_profile}")
            budget += choices[target_profile].storage_bits
        budgets[layer] = budget

    fixed_storage_bits = sum(
        selection.storage_bits
        for name, selection in base_scheme.selections.items()
        if name not in covered_names
    )
    initial_credit_bits = (
        base_scheme.target_storage_bits - fixed_storage_bits - sum(budgets.values())
    )
    return budgets, initial_credit_bits


def refine_layerwise(
    backend: RefinementBackend,
    corpus: CalibrationCorpus,
    base_scheme: CalibrationScheme,
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
    strategies: Mapping[int, Sequence[LayerStrategy]],
    output_scheme: str | Path,
    report: str | Path,
    *,
    work_dir: str | Path,
    window_length: int = 256,
    batch_size: int = 1,
    max_tokens: int = 8_192,
    seed: int = 20260718,
    cumulative_budget: bool = False,
    keep_hidden: bool = False,
) -> tuple[CalibrationScheme, list[LayerRefinement]]:
    """Choose each layer's strategy on train hidden states and carry its output forward."""

    scheme_path = Path(output_scheme).resolve()
    report_path = Path(report).resolve()
    if scheme_path.exists() or report_path.exists():
        raise FileExistsError("layerwise refinement output already exists")
    if set(strategies) != set(range(backend.num_layers)):
        raise ValueError("layerwise strategies do not cover every decoder layer")
    if cumulative_budget and base_scheme.storage_bits > base_scheme.target_storage_bits:
        raise ValueError("base scheme already exceeds its target storage")

    batches = list(
        corpus.iter_batches(
            "train",
            window_length=window_length,
            batch_size=batch_size,
            max_tokens=max_tokens,
            seed=seed,
        )
    )
    if not batches:
        raise ValueError("calibration corpus produced no training batches")
    layout = _layout(batches)
    token_count = layout[-1][2]
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    teacher_path = work / "refine-teacher-hidden.bin"
    quantized_path = work / "refine-quantized-hidden.bin"
    for path in (teacher_path, quantized_path):
        if path.exists():
            raise FileExistsError(f"layerwise hidden-state file already exists: {path}")
    teacher_store = HiddenStateStore(
        teacher_path,
        token_count,
        backend.hidden_size,
        backend.teacher_dtype,
    )
    quantized_store = HiddenStateStore(
        quantized_path,
        token_count,
        backend.hidden_size,
        backend.quantized_dtype,
    )
    device = getattr(backend, "device", "cuda")
    refinements: list[LayerRefinement] = []
    if cumulative_budget:
        target_layer_budgets, initial_credit_bits = _target_profile_layer_budgets(
            base_scheme,
            evaluations,
            strategies,
        )
    else:
        target_layer_budgets = {}
        initial_credit_bits = 0
    budget_credit_bits = initial_credit_bits
    try:
        for batch, start, _end in layout:
            ids = torch.as_tensor(batch.input_ids, device=device, dtype=torch.int64)
            hidden = backend.initial_hidden(ids)
            teacher_store.write(start, hidden)
            quantized_store.write(start, hidden)
        backend.release_initial_state()

        for layer_index in range(backend.num_layers):
            with backend.layer(layer_index, quantized=False) as layer:
                for batch, start, end in layout:
                    hidden = teacher_store.read(start, end, batch.input_ids.shape, device=device)
                    output = backend.forward_layer(layer, layer_index, hidden)
                    teacher_store.write(start, output)
            teacher_store.flush()

            all_layer_strategies = list(strategies[layer_index])
            if not all_layer_strategies:
                raise ValueError(f"layer {layer_index} has no refinement strategies")
            layer_names = set(all_layer_strategies[0].specs)
            if any(set(item.specs) != layer_names for item in all_layer_strategies):
                raise ValueError(f"layer {layer_index} strategies cover different tensors")
            base_layer_storage_bits = sum(
                base_scheme.require(name).storage_bits for name in layer_names
            )
            target_layer_storage_bits = (
                target_layer_budgets[layer_index] if cumulative_budget else base_layer_storage_bits
            )
            available_storage_bits = (
                target_layer_storage_bits + budget_credit_bits if cumulative_budget else None
            )
            layer_strategies = all_layer_strategies

            candidate_traces: list[StrategyTrace] = []
            backend.prepare_layer_strategies(
                layer_index,
                [strategy.specs for strategy in layer_strategies],
            )
            quantized_inputs = torch.empty(
                (token_count, backend.hidden_size),
                device=device,
                dtype=backend.quantized_dtype,
            )
            references = torch.empty(
                (token_count, backend.hidden_size),
                device=device,
                dtype=backend.teacher_dtype,
            )
            for batch, start, end in layout:
                shape = batch.input_ids.shape
                quantized_inputs[start:end].copy_(
                    quantized_store.read(start, end, shape, device=device).reshape(
                        -1, backend.hidden_size
                    )
                )
                references[start:end].copy_(
                    teacher_store.read(start, end, shape, device=device).reshape(
                        -1, backend.hidden_size
                    )
                )

            best_output = torch.empty_like(quantized_inputs)
            for strategy in layer_strategies:
                totals = np.zeros(5, dtype=np.float64)
                with backend.layer_for_strategy(layer_index, strategy.specs) as layer:
                    for batch, start, end in layout:
                        shape = batch.input_ids.shape
                        hidden = quantized_inputs[start:end].reshape(
                            *shape,
                            backend.hidden_size,
                        )
                        reference = references[start:end].reshape(
                            *shape,
                            backend.hidden_size,
                        )
                        output = backend.forward_layer(layer, layer_index, hidden)
                        _accumulate(reference, output, totals)
                item = StrategyTrace(strategy, _trace(layer_index, totals))
                candidate_traces.append(item)
                print(
                    json.dumps(
                        {
                            "event": "layer_candidate",
                            "layer": layer_index,
                            **item.document(),
                        }
                    ),
                    flush=True,
                )

            if not candidate_traces:
                raise RuntimeError(f"layer {layer_index} has no evaluated strategy")
            knee: ParetoKneeSelection | None = None
            if cumulative_budget:
                if available_storage_bits is None:
                    raise RuntimeError("cumulative refinement has no available layer budget")
                knee = _select_pareto_knee(candidate_traces, available_storage_bits)
                chosen = knee.selected
            else:
                chosen = min(
                    candidate_traces,
                    key=lambda item: (
                        item.trace.squared_error,
                        item.strategy.storage_bits,
                        item.strategy.name,
                    ),
                )

            with backend.layer_for_strategy(layer_index, chosen.strategy.specs) as layer:
                for batch, start, end in layout:
                    shape = batch.input_ids.shape
                    hidden = quantized_inputs[start:end].reshape(
                        *shape,
                        backend.hidden_size,
                    )
                    output = backend.forward_layer(layer, layer_index, hidden)
                    best_output[start:end].copy_(output.reshape(-1, backend.hidden_size))

            if cumulative_budget:
                budget_credit_bits += target_layer_storage_bits - chosen.strategy.storage_bits
                if budget_credit_bits < 0:
                    raise RuntimeError("cumulative greedy refinement overspent its budget")
            for batch, start, end in layout:
                output = best_output[start:end].reshape(
                    *batch.input_ids.shape,
                    backend.hidden_size,
                )
                quantized_store.write(start, output)
            del (
                quantized_inputs,
                references,
                best_output,
            )
            quantized_store.flush()
            backend.clear_quantized_cache(layer_index)
            refinements.append(
                LayerRefinement(
                    layer_index,
                    chosen.strategy,
                    tuple(candidate_traces),
                    base_storage_bits=(base_layer_storage_bits if cumulative_budget else None),
                    target_storage_bits=(target_layer_storage_bits if cumulative_budget else None),
                    available_storage_bits=available_storage_bits,
                    remaining_credit_bits=(budget_credit_bits if cumulative_budget else None),
                    pareto_frontier=(
                        tuple(item.strategy.name for item in knee.frontier) if knee else ()
                    ),
                    unconstrained_knee=(knee.unconstrained.strategy.name if knee else None),
                    knee_score=(knee.selected_score if knee else None),
                )
            )
            print(
                json.dumps(
                    {
                        "event": "layer_selected",
                        "layer": layer_index,
                        "base_storage_bits": base_layer_storage_bits,
                        "target_storage_bits": target_layer_storage_bits,
                        "available_storage_bits": available_storage_bits,
                        "remaining_credit_bits": (
                            budget_credit_bits if cumulative_budget else None
                        ),
                        "pareto_frontier_count": len(knee.frontier) if knee else None,
                        "unconstrained_knee": (knee.unconstrained.strategy.name if knee else None),
                        "knee_score": knee.selected_score if knee else None,
                        **chosen.document(),
                    }
                ),
                flush=True,
            )

        selections = dict(base_scheme.selections)
        for refinement in refinements:
            for name, spec in refinement.selected.specs.items():
                selections[name] = _selection_for_spec(name, spec, evaluations)
        refined = CalibrationScheme(
            path=None,
            target_profile=base_scheme.target_profile,
            target_storage_bits=base_scheme.target_storage_bits,
            selections=selections,
            metadata={
                **base_scheme.metadata,
                "layerwise_refinement": {
                    "objective": "real_next_hidden_sse_storage_pareto_knee",
                    "selection_split": "train",
                    "corpus": str(corpus.root),
                    "window_length": int(window_length),
                    "token_count": int(token_count),
                    "allocation": (
                        "cumulative_budget_greedy" if cumulative_budget else "per_layer_greedy"
                    ),
                    "initial_credit_bits": int(initial_credit_bits),
                    "remaining_credit_bits": int(budget_credit_bits),
                    "candidate_execution": "serial_resident_hidden_selected_replay",
                    "candidate_counts": {
                        str(item.layer): len(item.candidates) for item in refinements
                    },
                },
            },
            candidate_table=base_scheme.candidate_table,
            inint_selector=base_scheme.inint_selector,
            expert_selections=base_scheme.expert_selections,
        )
        storage_limit = (
            base_scheme.target_storage_bits if cumulative_budget else base_scheme.storage_bits
        )
        if refined.storage_bits > storage_limit:
            raise RuntimeError("layerwise refinement exceeded its storage limit")
        save_scheme(scheme_path, refined)

        report_document = {
            "format": "mfq.layerwise-refinement.v4",
            "base_scheme": str(base_scheme.path),
            "output_scheme": str(scheme_path),
            "corpus": str(corpus.root),
            "selection_split": "train",
            "window_length": int(window_length),
            "batch_size": int(batch_size),
            "token_count": int(token_count),
            "allocation": ("cumulative_budget_greedy" if cumulative_budget else "per_layer_greedy"),
            "target_storage_bits": int(base_scheme.target_storage_bits),
            "initial_credit_bits": int(initial_credit_bits),
            "remaining_credit_bits": int(budget_credit_bits),
            "candidate_execution": "serial_resident_hidden_selected_replay",
            "layers": [
                {
                    "layer": item.layer,
                    "selected": item.selected.name,
                    "base_storage_bits": item.base_storage_bits,
                    "target_storage_bits": item.target_storage_bits,
                    "available_storage_bits": item.available_storage_bits,
                    "remaining_credit_bits": item.remaining_credit_bits,
                    "pareto_frontier": list(item.pareto_frontier),
                    "unconstrained_knee": item.unconstrained_knee,
                    "knee_score": item.knee_score,
                    "candidates": [candidate.document() for candidate in item.candidates],
                }
                for item in refinements
            ],
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report_document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, report_path)
    finally:
        backend.release_initial_state()
        backend.clear_quantized_cache()
        teacher_store.close()
        quantized_store.close()
        if not keep_hidden:
            teacher_path.unlink(missing_ok=True)
            quantized_path.unlink(missing_ok=True)
    return load_scheme(scheme_path), refinements


def _strategy_key(strategy: LayerStrategy) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                name,
                json.dumps(
                    precision_document(normalize_dense_precision(spec)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for name, spec in strategy.specs.items()
        )
    )


def _strategy_for_scheme(
    layer_index: int,
    strategies: Sequence[LayerStrategy],
    scheme: CalibrationScheme,
) -> LayerStrategy:
    expected = {
        name: (
            scheme.require(name).spec
            if scheme.require(name).spec is not None
            else scheme.require(name).descriptor
        )
        for name in strategies[0].specs
    }
    matches = [item for item in strategies if item.specs == expected]
    if len(matches) != 1:
        raise RuntimeError(f"scheme has {len(matches)} matching strategies for layer {layer_index}")
    return matches[0]


def _scheme_with_strategies(
    base_scheme: CalibrationScheme,
    selected: Mapping[int, LayerStrategy],
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
) -> CalibrationScheme:
    selections = dict(base_scheme.selections)
    assigned: set[str] = set()
    for layer, strategy in sorted(selected.items()):
        overlap = assigned.intersection(strategy.specs)
        if overlap:
            raise RuntimeError(f"layer {layer} repeats tensors: {sorted(overlap)[:4]}")
        assigned.update(strategy.specs)
        for name, spec in strategy.specs.items():
            selections[name] = _selection_for_spec(name, spec, evaluations)
    return CalibrationScheme(
        path=None,
        target_profile=base_scheme.target_profile,
        target_storage_bits=base_scheme.target_storage_bits,
        selections=selections,
        metadata=dict(base_scheme.metadata),
        candidate_table=base_scheme.candidate_table,
        inint_selector=base_scheme.inint_selector,
        expert_selections=base_scheme.expert_selections,
    )


def _normalized_squared_error(trace: HiddenTrace) -> float:
    return trace.squared_error / max(trace.reference_energy, np.finfo(np.float64).tiny)


def _allocate_refinement_round(
    refinements: Sequence[LayerRefinement],
    *,
    target_storage_bits: int,
    fixed_storage_bits: int,
    max_changed_layers: int,
) -> tuple[dict[int, LayerStrategy], AllocationResult]:
    candidates: list[GroupCandidate] = []
    lookup: dict[tuple[str, str], LayerStrategy] = {}
    baseline_profiles: dict[str, str] = {}
    for refinement in refinements:
        group = f"layer.{refinement.layer}"
        current_key = _strategy_key(refinement.selected)
        for index, item in enumerate(refinement.candidates):
            profile = f"strategy.{index}"
            loss = _normalized_squared_error(item.trace)
            candidates.append(
                GroupCandidate(
                    group=group,
                    profile=profile,
                    specs=item.strategy.specs,
                    storage_bits=item.strategy.storage_bits,
                    train_loss=loss,
                    validation_loss=loss,
                )
            )
            lookup[(group, profile)] = item.strategy
            if _strategy_key(item.strategy) == current_key:
                if group in baseline_profiles:
                    raise RuntimeError(f"layer {refinement.layer} repeats its baseline strategy")
                baseline_profiles[group] = profile
        if group not in baseline_profiles:
            raise RuntimeError(f"layer {refinement.layer} has no baseline strategy")

    layer_budget = target_storage_bits - fixed_storage_bits
    if layer_budget <= 0:
        raise ValueError("fixed selections consume the full global storage budget")
    allocation = allocate(
        candidates,
        layer_budget,
        baseline_profiles=baseline_profiles,
        max_changed_groups=max_changed_layers,
    )
    selected: dict[int, LayerStrategy] = {}
    for group, candidate in allocation.selected.items():
        layer = int(group.split(".", 1)[1])
        selected[layer] = lookup[(group, candidate.profile)]
    return selected, allocation


def _initialize_refinement_hidden(
    backend: RefinementBackend,
    layout: Sequence[tuple[CalibrationBatch, int, int]],
    teacher_store: HiddenStateStore,
    quantized_store: HiddenStateStore,
) -> None:
    device = getattr(backend, "device", "cuda")
    for batch, start, _end in layout:
        ids = torch.as_tensor(batch.input_ids, device=device, dtype=torch.int64)
        hidden = backend.initial_hidden(ids)
        teacher_store.write(start, hidden)
        quantized_store.write(start, hidden)
    teacher_store.flush()
    quantized_store.flush()
    backend.release_initial_state()


def _evaluate_strategy_sweep(
    backend: RefinementBackend,
    layout: Sequence[tuple[CalibrationBatch, int, int]],
    token_count: int,
    teacher_store: HiddenStateStore,
    quantized_store: HiddenStateStore,
    current_scheme: CalibrationScheme,
    strategies: Mapping[int, Sequence[LayerStrategy]],
    *,
    iteration: int,
    phase: str,
) -> list[LayerRefinement]:
    _initialize_refinement_hidden(backend, layout, teacher_store, quantized_store)
    device = getattr(backend, "device", "cuda")
    refinements: list[LayerRefinement] = []

    for layer_index in range(backend.num_layers):
        layer_strategies = list(strategies[layer_index])
        if not layer_strategies:
            raise ValueError(f"layer {layer_index} has no refinement strategies")
        current = _strategy_for_scheme(layer_index, layer_strategies, current_scheme)

        with backend.layer(layer_index, quantized=False) as layer:
            for batch, start, end in layout:
                hidden = teacher_store.read(
                    start,
                    end,
                    batch.input_ids.shape,
                    device=device,
                )
                output = backend.forward_layer(layer, layer_index, hidden)
                teacher_store.write(start, output)
        teacher_store.flush()

        backend.prepare_layer_strategies(
            layer_index,
            [strategy.specs for strategy in layer_strategies],
        )
        quantized_inputs = torch.empty(
            (token_count, backend.hidden_size),
            device=device,
            dtype=backend.quantized_dtype,
        )
        references = torch.empty(
            (token_count, backend.hidden_size),
            device=device,
            dtype=backend.teacher_dtype,
        )
        for batch, start, end in layout:
            shape = batch.input_ids.shape
            quantized_inputs[start:end].copy_(
                quantized_store.read(start, end, shape, device=device).reshape(
                    -1, backend.hidden_size
                )
            )
            references[start:end].copy_(
                teacher_store.read(start, end, shape, device=device).reshape(
                    -1, backend.hidden_size
                )
            )

        current_output = torch.empty_like(quantized_inputs)
        candidate_traces: list[StrategyTrace] = []
        current_key = _strategy_key(current)
        current_seen = False
        for strategy in layer_strategies:
            totals = np.zeros(5, dtype=np.float64)
            is_current = _strategy_key(strategy) == current_key
            with backend.layer_for_strategy(layer_index, strategy.specs) as layer:
                for batch, start, end in layout:
                    shape = batch.input_ids.shape
                    hidden = quantized_inputs[start:end].reshape(
                        *shape,
                        backend.hidden_size,
                    )
                    reference = references[start:end].reshape(
                        *shape,
                        backend.hidden_size,
                    )
                    output = backend.forward_layer(layer, layer_index, hidden)
                    _accumulate(reference, output, totals)
                    if is_current:
                        current_output[start:end].copy_(output.reshape(-1, backend.hidden_size))
            item = StrategyTrace(strategy, _trace(layer_index, totals))
            candidate_traces.append(item)
            current_seen = current_seen or is_current
            print(
                json.dumps(
                    {
                        "event": "global_layer_candidate",
                        "phase": phase,
                        "iteration": iteration,
                        "layer": layer_index,
                        "current": is_current,
                        "normalized_squared_error": _normalized_squared_error(item.trace),
                        **item.document(),
                    }
                ),
                flush=True,
            )

        if not current_seen:
            raise RuntimeError(f"current layer {layer_index} strategy was not evaluated")
        for batch, start, end in layout:
            quantized_store.write(
                start,
                current_output[start:end].reshape(
                    *batch.input_ids.shape,
                    backend.hidden_size,
                ),
            )
        quantized_store.flush()
        backend.clear_quantized_cache(layer_index)
        del quantized_inputs, references, current_output
        refinements.append(LayerRefinement(layer_index, current, tuple(candidate_traces)))
        print(
            json.dumps(
                {
                    "event": "global_path_layer",
                    "phase": phase,
                    "iteration": iteration,
                    "layer": layer_index,
                    "strategy": current.name,
                }
            ),
            flush=True,
        )
    return refinements


def _current_path_traces(
    refinements: Sequence[LayerRefinement],
) -> list[StrategyTrace]:
    result: list[StrategyTrace] = []
    for refinement in refinements:
        key = _strategy_key(refinement.selected)
        matches = [item for item in refinement.candidates if _strategy_key(item.strategy) == key]
        if len(matches) != 1:
            raise RuntimeError(f"layer {refinement.layer} has {len(matches)} current-path traces")
        result.append(matches[0])
    return result


def refine_layerwise_global(
    backend: RefinementBackend,
    corpus: CalibrationCorpus,
    base_scheme: CalibrationScheme,
    evaluations: Mapping[str, Mapping[str, TensorCandidateEvaluation]],
    strategies: Mapping[int, Sequence[LayerStrategy]],
    output_scheme: str | Path,
    report: str | Path,
    *,
    work_dir: str | Path,
    window_length: int = 256,
    batch_size: int = 1,
    max_tokens: int = 8_192,
    seed: int = 20260718,
    max_iterations: int = 3,
    max_changed_layers: int = 0,
    keep_hidden: bool = False,
) -> tuple[CalibrationScheme, list[GlobalRefinementRound]]:
    """Iteratively transfer storage across layers using real hidden-state errors."""

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if max_changed_layers < 0:
        raise ValueError("max_changed_layers must be non-negative")
    scheme_path = Path(output_scheme).resolve()
    report_path = Path(report).resolve()
    if scheme_path.exists() or report_path.exists():
        raise FileExistsError("global layerwise refinement output already exists")
    if set(strategies) != set(range(backend.num_layers)):
        raise ValueError("global layerwise strategies do not cover every decoder layer")

    strategy_names: set[str] = set()
    for layer_index, choices in strategies.items():
        if not choices:
            raise ValueError(f"layer {layer_index} has no global strategies")
        names = set(choices[0].specs)
        if any(set(item.specs) != names for item in choices):
            raise ValueError(f"layer {layer_index} strategies cover different tensors")
        overlap = strategy_names.intersection(names)
        if overlap:
            raise ValueError(f"global strategies repeat tensors: {sorted(overlap)[:4]}")
        strategy_names.update(names)
    fixed_storage_bits = sum(
        item.storage_bits
        for name, item in base_scheme.selections.items()
        if name not in strategy_names
    )

    batches = list(
        corpus.iter_batches(
            "train",
            window_length=window_length,
            batch_size=batch_size,
            max_tokens=max_tokens,
            seed=seed,
        )
    )
    if not batches:
        raise ValueError("calibration corpus produced no training batches")
    layout = _layout(batches)
    token_count = layout[-1][2]
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    teacher_path = work / "global-refine-teacher-hidden.bin"
    quantized_path = work / "global-refine-quantized-hidden.bin"
    for path in (teacher_path, quantized_path):
        if path.exists():
            raise FileExistsError(f"global refinement hidden-state file already exists: {path}")
    teacher_store = HiddenStateStore(
        teacher_path,
        token_count,
        backend.hidden_size,
        backend.teacher_dtype,
    )
    quantized_store = HiddenStateStore(
        quantized_path,
        token_count,
        backend.hidden_size,
        backend.quantized_dtype,
    )

    current_scheme = base_scheme
    rounds: list[GlobalRefinementRound] = []
    final_path: list[StrategyTrace] = []
    converged = False
    active_layers = tuple(sorted(strategies))
    requested_change_limit = min(
        len(active_layers),
        max_changed_layers or max(2, (len(active_layers) + 3) // 4),
    )
    round_strategies = {layer: list(items) for layer, items in strategies.items()}
    candidate_counts: dict[str, int] = {}
    try:
        for iteration in range(1, max_iterations + 1):
            candidate_counts = {str(layer): len(items) for layer, items in round_strategies.items()}
            evaluated = _evaluate_strategy_sweep(
                backend,
                layout,
                token_count,
                teacher_store,
                quantized_store,
                current_scheme,
                round_strategies,
                iteration=iteration,
                phase="candidate_sweep",
            )
            current_path = _current_path_traces(evaluated)
            current_loss = float(
                sum(_normalized_squared_error(item.trace) for item in current_path)
            )
            input_storage_bits = current_scheme.storage_bits
            change_limit = requested_change_limit
            attempts: list[GlobalRefinementAttempt] = []
            accepted_selected = {item.layer: item.selected for item in evaluated}
            accepted_loss = current_loss
            accepted_changed_layers: tuple[int, ...] = ()
            last_solver = ""

            while True:
                selected, allocation = _allocate_refinement_round(
                    evaluated,
                    target_storage_bits=base_scheme.target_storage_bits,
                    fixed_storage_bits=fixed_storage_bits,
                    max_changed_layers=change_limit,
                )
                last_solver = allocation.solver
                proposed_scheme = _scheme_with_strategies(base_scheme, selected, evaluations)
                if proposed_scheme.storage_bits > base_scheme.target_storage_bits:
                    raise RuntimeError("global refinement exceeded its storage upper bound")
                changed_layers = tuple(
                    layer
                    for layer, strategy in sorted(selected.items())
                    if _strategy_key(strategy) != _strategy_key(accepted_selected[layer])
                )
                if not changed_layers:
                    attempts.append(
                        GlobalRefinementAttempt(
                            max_changed_layers=change_limit,
                            proposed_storage_bits=proposed_scheme.storage_bits,
                            current_normalized_squared_error=current_loss,
                            predicted_normalized_squared_error=allocation.train_loss,
                            replayed_normalized_squared_error=current_loss,
                            changed_layers=(),
                            accepted=False,
                        )
                    )
                    converged = True
                    final_path = current_path
                    break

                proposal_strategies = {layer: [selected[layer]] for layer in active_layers}
                replayed = _evaluate_strategy_sweep(
                    backend,
                    layout,
                    token_count,
                    teacher_store,
                    quantized_store,
                    proposed_scheme,
                    proposal_strategies,
                    iteration=iteration,
                    phase=f"proposal_replay_k{change_limit}",
                )
                proposal_path = _current_path_traces(replayed)
                proposal_loss = float(
                    sum(_normalized_squared_error(item.trace) for item in proposal_path)
                )
                tolerance = max(1e-12, current_loss * 1e-6)
                accepted = proposal_loss < current_loss - tolerance
                attempts.append(
                    GlobalRefinementAttempt(
                        max_changed_layers=change_limit,
                        proposed_storage_bits=proposed_scheme.storage_bits,
                        current_normalized_squared_error=current_loss,
                        predicted_normalized_squared_error=allocation.train_loss,
                        replayed_normalized_squared_error=proposal_loss,
                        changed_layers=changed_layers,
                        accepted=accepted,
                    )
                )
                print(
                    json.dumps(
                        {
                            "event": "global_refinement_attempt",
                            "iteration": iteration,
                            "max_changed_layers": change_limit,
                            "current_normalized_squared_error": current_loss,
                            "predicted_normalized_squared_error": allocation.train_loss,
                            "replayed_normalized_squared_error": proposal_loss,
                            "changed_layers": list(changed_layers),
                            "accepted": accepted,
                        }
                    ),
                    flush=True,
                )
                if accepted:
                    current_scheme = proposed_scheme
                    accepted_selected = selected
                    accepted_loss = proposal_loss
                    accepted_changed_layers = changed_layers
                    final_path = proposal_path
                    converged = False
                    break
                if change_limit <= 1:
                    converged = True
                    final_path = current_path
                    break
                change_limit = max(1, (change_limit + 1) // 2)

            round_layers = tuple(
                LayerRefinement(item.layer, accepted_selected[item.layer], item.candidates)
                for item in evaluated
            )
            rounds.append(
                GlobalRefinementRound(
                    iteration=iteration,
                    input_storage_bits=input_storage_bits,
                    output_storage_bits=current_scheme.storage_bits,
                    normalized_squared_error=accepted_loss,
                    changed_layers=accepted_changed_layers,
                    converged=converged,
                    layers=round_layers,
                    solver=last_solver,
                    attempts=tuple(attempts),
                )
            )
            print(
                json.dumps(
                    {
                        "event": "global_refinement_round",
                        "iteration": iteration,
                        "input_storage_bits": input_storage_bits,
                        "output_storage_bits": current_scheme.storage_bits,
                        "target_storage_bits": base_scheme.target_storage_bits,
                        "normalized_squared_error": accepted_loss,
                        "changed_layers": list(accepted_changed_layers),
                        "converged": converged,
                    }
                ),
                flush=True,
            )
            if converged:
                break
            all_strategies = build_global_layer_strategies(evaluations, current_scheme)
            round_strategies = {layer: all_strategies[layer] for layer in active_layers}

        refined = CalibrationScheme(
            path=None,
            target_profile=current_scheme.target_profile,
            target_storage_bits=current_scheme.target_storage_bits,
            selections=current_scheme.selections,
            metadata={
                **base_scheme.metadata,
                "global_layerwise_refinement": {
                    "objective": "sum_layer_normalized_hidden_squared_error",
                    "selection_split": "train",
                    "corpus": str(corpus.root),
                    "window_length": int(window_length),
                    "token_count": int(token_count),
                    "candidate_execution": "serial_resident_hidden",
                    "candidate_neighborhood": "one_precision_group_per_layer",
                    "max_changed_layers": int(requested_change_limit),
                    "acceptance": "strict_replayed_objective_decrease",
                    "max_iterations": int(max_iterations),
                    "completed_iterations": len(rounds),
                    "converged": converged,
                    "final_path_replayed": True,
                    "candidate_counts": candidate_counts,
                },
            },
            candidate_table=base_scheme.candidate_table,
            inint_selector=base_scheme.inint_selector,
            expert_selections=current_scheme.expert_selections,
        )
        if refined.storage_bits > refined.target_storage_bits:
            raise RuntimeError("global refinement output exceeds its target storage")
        save_scheme(scheme_path, refined)

        report_document = {
            "format": "mfq.global-layerwise-refinement.v2",
            "base_scheme": str(base_scheme.path),
            "output_scheme": str(scheme_path),
            "corpus": str(corpus.root),
            "selection_split": "train",
            "window_length": int(window_length),
            "batch_size": int(batch_size),
            "token_count": int(token_count),
            "target_storage_bits": int(base_scheme.target_storage_bits),
            "actual_storage_bits": int(refined.storage_bits),
            "fixed_storage_bits": int(fixed_storage_bits),
            "candidate_execution": "serial_resident_hidden",
            "candidate_neighborhood": "one_precision_group_per_layer",
            "max_changed_layers": int(requested_change_limit),
            "acceptance": "strict_replayed_objective_decrease",
            "converged": converged,
            "rounds": [
                {
                    "iteration": item.iteration,
                    "input_storage_bits": item.input_storage_bits,
                    "output_storage_bits": item.output_storage_bits,
                    "normalized_squared_error": item.normalized_squared_error,
                    "changed_layers": list(item.changed_layers),
                    "converged": item.converged,
                    "solver": item.solver,
                    "attempts": [
                        {
                            "max_changed_layers": attempt.max_changed_layers,
                            "proposed_storage_bits": attempt.proposed_storage_bits,
                            "current_normalized_squared_error": (
                                attempt.current_normalized_squared_error
                            ),
                            "predicted_normalized_squared_error": (
                                attempt.predicted_normalized_squared_error
                            ),
                            "replayed_normalized_squared_error": (
                                attempt.replayed_normalized_squared_error
                            ),
                            "changed_layers": list(attempt.changed_layers),
                            "accepted": attempt.accepted,
                        }
                        for attempt in item.attempts
                    ],
                    "layers": [
                        {
                            "layer": layer.layer,
                            "selected": layer.selected.name,
                            "candidates": [candidate.document() for candidate in layer.candidates],
                        }
                        for layer in item.layers
                    ],
                }
                for item in rounds
            ],
            "final_path": [item.document() for item in final_path],
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report_document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, report_path)
    finally:
        backend.release_initial_state()
        backend.clear_quantized_cache()
        teacher_store.close()
        quantized_store.close()
        if not keep_hidden:
            teacher_path.unlink(missing_ok=True)
            quantized_path.unlink(missing_ok=True)
    return load_scheme(scheme_path), rounds


__all__ = [
    "GlobalRefinementAttempt",
    "GlobalRefinementRound",
    "LayerRefinement",
    "StrategyTrace",
    "refine_layerwise",
    "refine_layerwise_global",
]
