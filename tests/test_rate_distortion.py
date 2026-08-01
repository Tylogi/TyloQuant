from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from mfq.calibration.artifact import CalibrationScheme, TensorSelection
from mfq.calibration.dataset import build_corpus_from_records
from mfq.calibration.evaluator import (
    TensorCandidateEvaluation,
    _cache_path,
    _save_candidate_cache,
    load_scheme_candidate_evaluations,
    nint_storage_bits,
)
from mfq.calibration.rate_distortion import (
    PrecisionGroup,
    PrecisionOption,
    add_rate_gradient_,
    build_precision_groups,
    build_precision_groups_from_scheme,
    discretize_gate_logits,
    expected_storage_bits,
    refine_discrete_profiles,
)
from mfq.calibration.soft_refinement import refine_soft_rate_distortion
from mfq.calibration.terminal_kl import ChunkedTerminalObjective, chunked_terminal_kl
from mfq.formats.nint import NintSpec
from mfq.kernels import torch_backend
from mfq.quantize.nint_quant import quantize as quantize_nint


def _option(profile: str, bits: int, storage_bits: int) -> PrecisionOption:
    name = "weight"
    return PrecisionOption(
        profile=profile,
        specs={name: NintSpec(bits, 24, 6 if bits == 4 else 7)},
        storage_bits=storage_bits,
        surrogate_train_loss=0.0,
        surrogate_validation_loss=0.0,
    )


def _groups() -> tuple[PrecisionGroup, ...]:
    options = (_option("low", 4, 10), _option("high", 6, 20))
    return (
        PrecisionGroup("layer.0.a", 0, ("a",), options, "low"),
        PrecisionGroup("layer.1.b", 1, ("b",), options, "low"),
    )


def _evaluation(
    name: str,
    group: str,
    profile: str,
    spec: NintSpec,
) -> TensorCandidateEvaluation:
    rows, columns = 4, 24
    storage = nint_storage_bits(rows, columns, spec)
    return TensorCandidateEvaluation(
        name=name,
        group=group,
        profile=profile,
        spec=spec,
        rows=rows,
        columns=columns,
        storage_bits=storage,
        train_loss=float(9 - spec.bits),
        validation_loss=float(10 - spec.bits),
        train_nmse_percent=1.0,
        validation_nmse_percent=1.0,
        train_row_loss=np.ones(rows, dtype=np.float32),
        validation_row_loss=np.ones(rows, dtype=np.float32),
    )


def test_build_precision_groups_preserves_runtime_fusion_groups() -> None:
    specs = {
        "NINT4": NintSpec(4, 24, 6),
        "NINT6": NintSpec(6, 24, 7),
    }
    names = (
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
    )
    group = "layer.0.ffn_gate_up"
    evaluations = {
        name: {profile: _evaluation(name, group, profile, spec) for profile, spec in specs.items()}
        for name in names
    }
    selections = {}
    for name in names:
        value = evaluations[name]["NINT4"]
        selections[name] = TensorSelection(
            name,
            group,
            value.spec,
            value.rows,
            value.columns,
            value.storage_bits,
            value.train_loss,
            value.validation_loss,
        )
    base = CalibrationScheme(
        path=None,
        target_profile="NINT4",
        target_storage_bits=sum(item.storage_bits for item in selections.values()),
        selections=selections,
        metadata={},
        candidate_table={},
    )

    groups = build_precision_groups(evaluations, base)

    assert len(groups) == 1
    assert groups[0].name == group
    assert groups[0].tensor_names == tuple(sorted(names))
    assert groups[0].base_profile == "NINT4"
    assert [item.profile for item in groups[0].options] == ["NINT4", "NINT6"]


def test_scheme_groups_and_score_cache_are_reused_read_only(tmp_path: Path) -> None:
    specs = {
        "NINT4": NintSpec(4, 24, 6),
        "NINT6": NintSpec(6, 24, 7),
    }
    names = ("gate.weight", "up.weight")
    group = "layer.0.ffn_gate_up"
    evaluations = {
        name: {profile: _evaluation(name, group, profile, spec) for profile, spec in specs.items()}
        for name in names
    }
    selections = {
        name: TensorSelection(
            name=name,
            group=group,
            spec=evaluations[name]["NINT4"].spec,
            rows=evaluations[name]["NINT4"].rows,
            columns=evaluations[name]["NINT4"].columns,
            storage_bits=evaluations[name]["NINT4"].storage_bits,
            train_loss=evaluations[name]["NINT4"].train_loss,
            validation_loss=evaluations[name]["NINT4"].validation_loss,
        )
        for name in names
    }
    candidate_table = {
        group: [
            {
                "profile": profile,
                "storage_bits": sum(evaluations[name][profile].storage_bits for name in names),
                "train_loss": sum(evaluations[name][profile].train_loss for name in names),
                "validation_loss": sum(
                    evaluations[name][profile].validation_loss for name in names
                ),
            }
            for profile in specs
        ]
    }
    base = CalibrationScheme(
        path=None,
        target_profile="NINT4",
        target_storage_bits=sum(item.storage_bits for item in selections.values()),
        selections=selections,
        metadata={},
        candidate_table=candidate_table,
    )
    cache = tmp_path / "scores"
    cache.mkdir()
    identities = ("old-identity", "new-identity")
    paths = []
    for name_index, name in enumerate(names):
        for profile, value in evaluations[name].items():
            path = _cache_path(cache, name, profile)
            _save_candidate_cache(path, value, identities[name_index])
            paths.append(path)
    original = {path: path.read_bytes() for path in paths}

    loaded, audit = load_scheme_candidate_evaluations(base, cache, profiles=specs)
    groups = build_precision_groups_from_scheme(base, profiles=specs)

    assert set(loaded) == set(names)
    assert audit["write_mode"] == "read_only"
    assert audit["identities"] == {"new-identity": 2, "old-identity": 2}
    assert groups[0].tensor_names == names
    assert groups[0].base_profile == "NINT4"
    assert [option.profile for option in groups[0].options] == ["NINT4", "NINT6"]
    assert {path: path.read_bytes() for path in paths} == original


def test_rate_gradient_matches_autograd() -> None:
    groups = _groups()
    logits = {
        groups[0].name: torch.tensor([0.2, -0.1], requires_grad=True),
        groups[1].name: torch.tensor([-0.3, 0.4], requires_grad=True),
    }
    temperature = 0.7
    target = 31
    dual = 0.25
    objective = (
        dual
        * expected_storage_bits(
            groups,
            logits,
            temperature=temperature,
        )
        / target
    )
    objective.backward()
    reference = {name: value.grad.detach().clone() for name, value in logits.items()}
    for value in logits.values():
        value.grad = None

    add_rate_gradient_(
        groups,
        logits,
        temperature=temperature,
        target_storage_bits=target,
        dual=dual,
    )

    for name, value in logits.items():
        torch.testing.assert_close(value.grad, reference[name])


def test_discretize_gate_logits_uses_global_budget() -> None:
    groups = _groups()
    logits = {
        groups[0].name: torch.tensor([0.0, 4.0]),
        groups[1].name: torch.tensor([3.0, 1.0]),
    }

    selected = discretize_gate_logits(
        groups,
        logits,
        target_storage_bits=30,
    )

    assert selected == {"layer.0.a": "high", "layer.1.b": "low"}


def test_discrete_refinement_can_accept_budget_neutral_pair() -> None:
    groups = _groups()
    logits = {
        groups[0].name: torch.tensor([1.0, 2.0]),
        groups[1].name: torch.tensor([2.0, 1.5]),
    }
    initial = {"layer.0.a": "high", "layer.1.b": "low"}

    def evaluate(profiles: Mapping[str, str]) -> float:
        pair = (profiles["layer.0.a"], profiles["layer.1.b"])
        return {
            ("high", "low"): 10.0,
            ("low", "low"): 12.0,
            ("low", "high"): 5.0,
        }.get(pair, 20.0)

    selected, history = refine_discrete_profiles(
        groups,
        initial,
        logits,
        evaluate,
        target_storage_bits=30,
        max_iterations=2,
        max_single=4,
        max_pair=4,
    )

    assert selected == {"layer.0.a": "low", "layer.1.b": "high"}
    assert [item.metric for item in history] == [10.0, 5.0]
    assert history[-1].changed_groups == ("layer.0.a", "layer.1.b")


def test_chunked_terminal_kl_matches_full_logits_and_gradient() -> None:
    generator = torch.Generator().manual_seed(2026)
    reference = torch.randn((2, 5, 7), generator=generator)
    candidate = reference + 0.1 * torch.randn((2, 5, 7), generator=generator)
    head = torch.randn((13, 7), generator=generator)
    positions = 2 * 4

    direct_candidate = candidate.clone().requires_grad_(True)
    reference_logits = torch.nn.functional.linear(reference[:, :-1], head).float()
    candidate_logits = torch.nn.functional.linear(direct_candidate[:, :-1], head).float()
    reference_log_probability = torch.log_softmax(reference_logits, dim=-1)
    reference_probability = torch.exp(reference_log_probability)
    candidate_log_probability = torch.log_softmax(candidate_logits, dim=-1)
    direct_sum = (
        reference_probability * (reference_log_probability - candidate_log_probability)
    ).sum()
    (direct_sum / positions).backward()

    for row_chunk in (3, 13, 0):
        result = chunked_terminal_kl(
            reference,
            candidate,
            torch.nn.Identity(),
            torch.nn.Identity(),
            head,
            row_chunk=row_chunk,
            with_gradient=True,
            gradient_scale=1.0 / positions,
        )
        assert result.positions == positions
        assert abs(result.sum_kl - float(direct_sum.item())) < 2e-5
        torch.testing.assert_close(result.gradient, direct_candidate.grad, rtol=2e-5, atol=2e-5)


def test_execution_ready_nint_cache_matches_regular_gpu_metadata() -> None:
    generator = np.random.default_rng(73)
    for spec in (
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 7),
        NintSpec(6, 24, 7),
        NintSpec(8, 48, 7),
    ):
        weight = generator.standard_normal((4, spec.groupsize * 2), dtype=np.float32)
        encoded = quantize_nint(weight, spec, axis=0)
        reference = torch_backend.to_gpu(encoded, "cpu")
        arrays = torch_backend.nint_deploy_arrays(encoded)
        cached = torch_backend.nint_deploy_to_gpu(
            arrays,
            bits=spec.bits,
            groupsize=spec.groupsize,
            neuron_len=encoded.neuron_len,
            shape=encoded.shape,
            axis=encoded.axis,
            device="cpu",
        )
        for name in ("q_packed", "sub_scale", "sub_min", "neuron_scale", "neuron_min"):
            torch.testing.assert_close(cached[name], reference[name])
        for name in ("bits", "out", "ng", "gs", "neuron_len", "shape", "axis"):
            assert cached[name] == reference[name]


class _Tokenizer:
    eos_token_id = 2
    vocab_size = 512
    name_or_path = "rate-distortion-test"
    chat_template = ""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [3 + ord(value) % 251 for value in text]


class _SoftBackend:
    num_layers = 2
    hidden_size = 2
    teacher_dtype = torch.float32
    quantized_dtype = torch.float32
    soft_dtype = torch.bfloat16
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.soft_seen_dtypes: list[torch.dtype] = []

    def initial_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        value = input_ids.float() / 512.0
        return torch.stack((value, torch.ones_like(value) * 0.25), dim=-1)

    def release_initial_state(self) -> None:
        return None

    @contextmanager
    def layer(self, layer_index: int, *, quantized: bool):
        yield ("hard" if quantized else "dense", layer_index, 6)

    @contextmanager
    def layer_for_soft_assignment(self, layer_index, groups, logits, temperature):
        group = next(item for item in groups if item.layer == layer_index)
        yield "soft", layer_index, group, logits[group.name], temperature

    @contextmanager
    def layer_for_strategy(self, layer_index, specs):
        bits = next(iter(specs.values())).bits
        yield "hard", layer_index, bits

    @contextmanager
    def terminal_objective(self):
        head = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        yield ChunkedTerminalObjective(torch.nn.Identity(), torch.nn.Identity(), head)

    def clear_quantized_cache(self, layer_index=None) -> None:
        del layer_index

    def _forward(self, layer, hidden_states):
        kind, layer_index = layer[:2]
        output = hidden_states
        if layer_index == 1:
            output = output * output.new_tensor([3.0, 1.0])
        if kind == "dense":
            return output
        if kind == "soft":
            self.soft_seen_dtypes.append(hidden_states.dtype)
            group, logits, temperature = layer[2:]
            probabilities = torch.softmax(logits / temperature, dim=0)
            low_index = next(
                index for index, option in enumerate(group.options) if option.profile == "NINT4"
            )
            error = 0.5 if layer_index == 0 else 0.1
            return output + probabilities[low_index] * output.new_tensor([error, 0.0])
        error = 0.5 if layer_index == 0 else 0.1
        return output if layer[2] == 6 else output + output.new_tensor([error, 0.0])

    def forward_layer(self, layer, layer_index, hidden_states):
        assert layer[1] == layer_index
        return self._forward(layer, hidden_states)

    def forward_layer_with_grad(self, layer, layer_index, hidden_states):
        return self.forward_layer(layer, layer_index, hidden_states)


def test_soft_rate_distortion_learns_layer_specific_precision(tmp_path: Path) -> None:
    specs = {
        "NINT4": NintSpec(4, 24, 6),
        "NINT6": NintSpec(6, 24, 7),
    }
    evaluations = {}
    selections = {}
    low_total = 0
    upgrade = 0
    names = {}
    for layer in range(2):
        name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        names[layer] = name
        group = f"layer.{layer}.ffn_down"
        evaluations[name] = {
            profile: _evaluation(name, group, profile, spec) for profile, spec in specs.items()
        }
        low = evaluations[name]["NINT4"]
        high = evaluations[name]["NINT6"]
        low_total += low.storage_bits
        upgrade = high.storage_bits - low.storage_bits
        selections[name] = TensorSelection(
            name,
            group,
            low.spec,
            low.rows,
            low.columns,
            low.storage_bits,
            low.train_loss,
            low.validation_loss,
        )
    base = CalibrationScheme(
        path=None,
        target_profile="NINT4",
        target_storage_bits=low_total + upgrade,
        selections=selections,
        metadata={},
        candidate_table={},
    )
    corpus = build_corpus_from_records(
        {"language": [f"global allocation sample {index}" for index in range(20)]},
        _Tokenizer(),
        tmp_path / "corpus",
        train_tokens=16,
        validation_tokens=8,
        sequence_length=4,
        seed=17,
        render_mode="plain",
    )

    backend = _SoftBackend()
    result = refine_soft_rate_distortion(
        backend,
        corpus,
        base,
        evaluations,
        tmp_path / "scheme.json",
        tmp_path / "report.json",
        work_dir=tmp_path / "work",
        window_length=4,
        batch_size=2,
        train_tokens=8,
        validation_tokens=8,
        steps=24,
        learning_rate=0.15,
        dual_learning_rate=0.5,
        temperature_start=1.0,
        temperature_end=0.2,
        head_row_chunk=2,
        discrete_iterations=1,
        discrete_single_candidates=4,
        discrete_pair_candidates=4,
    )

    assert result.accepted
    assert result.selected_train_kl < result.base_train_kl
    assert result.selected_validation_kl < result.base_validation_kl
    assert result.scheme.selections[names[0]].spec.bits == 6
    assert result.scheme.selections[names[1]].spec.bits == 4
    assert set(backend.soft_seen_dtypes) == {torch.bfloat16}
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["initial_profiles"]
    assert report["refined_profiles"] == report["selected_profiles"]
    assert report["initial_storage_bits"] <= base.target_storage_bits
    assert report["refined_storage_bits"] == result.scheme.storage_bits


def test_sharded_soft_rate_distortion_covers_all_tokens_and_external_validation(
    tmp_path: Path,
) -> None:
    specs = {
        "NINT4": NintSpec(4, 24, 6),
        "NINT6": NintSpec(6, 24, 7),
    }
    evaluations = {}
    selections = {}
    low_total = 0
    upgrade = 0
    names = {}
    for layer in range(2):
        name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        names[layer] = name
        group = f"layer.{layer}.ffn_down"
        evaluations[name] = {
            profile: _evaluation(name, group, profile, spec) for profile, spec in specs.items()
        }
        low = evaluations[name]["NINT4"]
        high = evaluations[name]["NINT6"]
        low_total += low.storage_bits
        upgrade = high.storage_bits - low.storage_bits
        selections[name] = TensorSelection(
            name,
            group,
            low.spec,
            low.rows,
            low.columns,
            low.storage_bits,
            low.train_loss,
            low.validation_loss,
        )
    base = CalibrationScheme(
        path=None,
        target_profile="NINT4",
        target_storage_bits=low_total + upgrade,
        selections=selections,
        metadata={},
        candidate_table={},
    )
    train_corpus = build_corpus_from_records(
        {"language": [f"sharded train sample {index}" for index in range(40)]},
        _Tokenizer(),
        tmp_path / "train-corpus",
        train_tokens=32,
        validation_tokens=8,
        sequence_length=4,
        seed=23,
        render_mode="plain",
    )
    validation_corpus = build_corpus_from_records(
        {"language": [f"external validation sample {index}" for index in range(40)]},
        _Tokenizer(),
        tmp_path / "validation-corpus",
        train_tokens=8,
        validation_tokens=16,
        sequence_length=4,
        seed=29,
        render_mode="plain",
    )
    backend = _SoftBackend()

    result = refine_soft_rate_distortion(
        backend,
        train_corpus,
        base,
        evaluations,
        tmp_path / "sharded-scheme.json",
        tmp_path / "sharded-report.json",
        work_dir=tmp_path / "sharded-work",
        window_length=4,
        batch_size=2,
        train_tokens=16,
        validation_tokens=8,
        validation_corpus=validation_corpus,
        shard_tokens=8,
        epochs=12,
        hard_train_tokens=8,
        learning_rate=0.15,
        dual_learning_rate=0.5,
        temperature_start=1.0,
        temperature_end=0.2,
        head_row_chunk=2,
        discrete_iterations=1,
        discrete_single_candidates=4,
        discrete_pair_candidates=4,
    )

    report = json.loads((tmp_path / "sharded-report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "sharded"
    assert report["corpus"] == str(train_corpus.root)
    assert report["validation_corpus"] == str(validation_corpus.root)
    assert report["train_tokens"] == 16
    assert report["hard_train_tokens"] == 8
    assert report["validation_tokens"] == 8
    assert report["shard_count"] == 2
    assert report["epochs"] == 12
    assert report["updates"] == 24
    assert len(result.steps) == 24
    for epoch in range(12):
        epoch_steps = [step for step in result.steps if step.epoch == epoch]
        assert {step.shard_index for step in epoch_steps} == {0, 1}
        assert sum(step.shard_tokens or 0 for step in epoch_steps) == 16
    assert result.scheme.storage_bits <= base.target_storage_bits
    assert set(backend.soft_seen_dtypes) == {torch.bfloat16}
    assert not any((tmp_path / "sharded-work").glob("soft-session-*"))
    checkpoint = torch.load(
        tmp_path / "sharded-work" / "soft-rate-distortion-checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["schedule"]["mode"] == "sharded"
    assert checkpoint["schedule"]["train_tokens"] == 16
    assert checkpoint["schedule"]["shard_count"] == 2
    assert checkpoint["schedule"]["total_updates"] == 24
