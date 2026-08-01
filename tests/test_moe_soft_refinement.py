from __future__ import annotations

import json
from pathlib import Path

import torch

from mfq.calibration.artifact import (
    CalibrationScheme,
    ExpertSelection,
    ExpertTensorSelection,
)
from mfq.calibration.layerwise_gemma4_moe import (
    Gemma4MoESoftBackend,
    _SoftGemma4Experts,
)
from mfq.calibration.moe_soft_refinement import (
    load_coupled_expert_precision_problem,
)
from mfq.calibration.rate_distortion import fixed_storage_bits
from mfq.formats.nint import NintSpec


def test_soft_gemma4_experts_mix_complete_expert_functions() -> None:
    torch.manual_seed(7)
    candidates = 2
    experts = 2
    hidden = 3
    intermediate = 2
    gate_weights = tuple(
        torch.randn(experts, 2 * intermediate, hidden)
        for _ in range(candidates)
    )
    down_weights = tuple(
        torch.randn(experts, hidden, intermediate)
        for _ in range(candidates)
    )
    logits = tuple(
        torch.tensor([0.3 - expert, -0.2 + expert], requires_grad=True)
        for expert in range(experts)
    )
    module = _SoftGemma4Experts(
        torch.stack(gate_weights, dim=1),
        torch.stack(down_weights, dim=1),
        logits,
        temperature=0.7,
        hidden_activation="gelu_pytorch_tanh",
    )
    hidden_states = torch.randn(4, hidden)
    top_k_index = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
    top_k_weights = torch.tensor(
        [[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.9, 0.1]]
    )

    actual = module(hidden_states, top_k_index, top_k_weights)

    expected = torch.zeros_like(hidden_states)
    activation = torch.nn.functional.gelu
    for token in range(hidden_states.shape[0]):
        for route in range(top_k_index.shape[1]):
            expert = int(top_k_index[token, route])
            probabilities = torch.softmax(logits[expert] / 0.7, dim=0)
            mixed = torch.zeros(hidden)
            for candidate in range(candidates):
                gate, up = torch.nn.functional.linear(
                    hidden_states[token],
                    gate_weights[candidate][expert],
                ).chunk(2, dim=-1)
                output = torch.nn.functional.linear(
                    activation(gate, approximate="tanh") * up,
                    down_weights[candidate][expert],
                )
                mixed = mixed + probabilities[candidate] * output
            expected[token] += mixed * top_k_weights[token, route]

    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert all(value.grad is not None for value in logits)
    assert all(torch.isfinite(value.grad).all() for value in logits if value.grad is not None)
    assert all(float(value.grad.abs().sum()) > 0 for value in logits if value.grad is not None)


def _expert_tensor(
    name: str,
    specs: tuple[NintSpec, ...],
    storage: tuple[int, ...],
) -> ExpertTensorSelection:
    return ExpertTensorSelection(
        name=name,
        group="layer.0.experts",
        n_experts=len(specs),
        rows_per_expert=4,
        columns=8,
        selections=tuple(
            ExpertSelection(
                expert_id=expert,
                spec=spec,
                storage_bits=storage[expert],
                train_loss=float(expert + 1),
                validation_loss=float(expert + 1),
            )
            for expert, spec in enumerate(specs)
        ),
    )


def test_coupled_expert_problem_rebuilds_base_scheme_exactly(tmp_path: Path) -> None:
    gate_name = "model.language_model.layers.0.experts.gate_up_proj"
    down_name = "model.language_model.layers.0.experts.down_proj"
    low = NintSpec(4, 8, 6)
    high = NintSpec(8, 8, 8)
    selected_specs = (low, high)
    gate_storage = (100, 180)
    down_storage = (80, 140)
    gate = _expert_tensor(gate_name, selected_specs, gate_storage)
    down = _expert_tensor(down_name, selected_specs, down_storage)
    base = CalibrationScheme(
        path=None,
        target_profile="EW",
        target_storage_bits=gate.storage_bits + down.storage_bits,
        selections={},
        metadata={},
        candidate_table={},
        expert_selections={gate_name: gate, down_name: down},
    )
    candidate_path = tmp_path / "expert-candidates.jsonl"
    rows = []
    for expert in range(2):
        for profile, spec, gate_bits, down_bits in (
            ("NINT4", low, 100, 80),
            ("NINT8", high, 180, 140),
        ):
            exposure = 0.5 + 0.1 * expert
            gate_nmse = 0.01 if profile == "NINT4" else 0.001
            down_nmse = 0.02 if profile == "NINT4" else 0.002
            rows.append(
                {
                    "layer": 0,
                    "expert": expert,
                    "profile": profile,
                    "spec": {
                        "bits": spec.bits,
                        "groupsize": spec.groupsize,
                        "sub_bits": spec.sub_bits,
                    },
                    "normalized_exposure": exposure,
                    "gate_nmse": gate_nmse,
                    "down_nmse": down_nmse,
                    "loss": 0.5 * exposure * (gate_nmse + down_nmse),
                    "gate_name": gate_name,
                    "down_name": down_name,
                    "gate_shape": [4, 8],
                    "down_shape": [4, 8],
                    "gate_storage_bits": gate_bits,
                    "down_storage_bits": down_bits,
                }
            )
    candidate_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    problem = load_coupled_expert_precision_problem(candidate_path, base)
    profiles = {group.name: group.base_profile for group in problem.groups}
    rebuilt = problem.build_scheme(base, problem.groups, profiles, {"test": True})

    assert len(problem.groups) == 2
    assert fixed_storage_bits(base, problem.groups) == 0
    assert rebuilt.storage_bits == base.storage_bits
    assert rebuilt.target_storage_bits == base.target_storage_bits
    assert rebuilt.require_expert(gate_name).specs == selected_specs
    assert rebuilt.require_expert(down_name).specs == selected_specs
    assert rebuilt.metadata["test"] is True


def test_soft_weight_cache_roundtrip(tmp_path: Path) -> None:
    backend = object.__new__(Gemma4MoESoftBackend)
    backend.device = torch.device("cpu")
    gate = torch.arange(2 * 3 * 4 * 5, dtype=torch.bfloat16).reshape(2, 3, 4, 5)
    down = torch.arange(2 * 3 * 5 * 2, dtype=torch.bfloat16).reshape(2, 3, 5, 2)
    data_path = tmp_path / "layer.bf16"
    metadata_path = tmp_path / "layer.json"
    metadata = {"format": "test", "layer": 0}

    backend._save_soft_weight_cache(
        data_path,
        metadata_path,
        metadata,
        gate,
        down,
    )
    restored = backend._read_soft_weight_cache(
        data_path,
        metadata_path,
        metadata,
        (2, 4, 5),
        (2, 5, 2),
        3,
    )

    assert restored is not None
    restored_gate, restored_down = restored
    torch.testing.assert_close(restored_gate, gate)
    torch.testing.assert_close(restored_down, down)
