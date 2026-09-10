from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mfq.calibration.imatrix import ActivationImatrixCollector, ImatrixTarget, _targets


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            )
        )
        self.down_proj = nn.Parameter(
            torch.tensor([[[2.0], [4.0]], [[3.0], [5.0]]])
        )
        self.act_fn = lambda value: value

    def forward(self, hidden, selected, weights):
        output = torch.zeros_like(hidden)
        for expert in range(2):
            token_idx, top_k_pos = torch.where(selected == expert)
            current = hidden[token_idx]
            gate, up = torch.nn.functional.linear(
                current, self.gate_up_proj[expert]
            ).chunk(2, dim=-1)
            value = torch.nn.functional.linear(
                self.act_fn(gate) * up, self.down_proj[expert]
            )
            output.index_add_(0, token_idx, value * weights[token_idx, top_k_pos, None])
        return output


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(2, 1, bias=False)
        self.experts = _Experts()


class _SiluExpertLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _Experts()
        self.experts.act_fn = torch.nn.functional.silu


class _SiluMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(2, 2, bias=False)
        self.up_proj = nn.Linear(2, 2, bias=False)
        self.down_proj = nn.Linear(2, 2, bias=False)
        self.act_fn = torch.nn.functional.silu
        with torch.no_grad():
            self.gate_proj.weight.copy_(torch.tensor([[1.0, -0.5], [0.25, 0.75]]))
            self.up_proj.weight.copy_(torch.tensor([[0.5, 1.0], [-1.0, 0.5]]))
            self.down_proj.weight.copy_(torch.tensor([[2.0, -1.0], [0.5, 3.0]]))

    def forward(self, value):
        return self.down_proj(self.act_fn(self.gate_proj(value)) * self.up_proj(value))


class _DenseGateLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _SiluMlp()


class _GatedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 1
        self.q_proj = nn.Linear(2, 4, bias=False)
        self.o_proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0], [0.5, -0.25], [0.0, 1.0], [-0.5, 0.75]]
                )
            )
            self.o_proj.weight.copy_(torch.tensor([[2.0, -1.0], [0.5, 3.0]]))

    def forward(self, value):
        projected = self.q_proj(value).reshape(*value.shape[:-1], 2, 2)
        attended, gate = projected.chunk(2, dim=-1)
        gated = attended.reshape(*value.shape[:-1], 2) * torch.sigmoid(
            gate.reshape(*value.shape[:-1], 2)
        )
        return self.o_proj(gated)


class _AttentionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _GatedAttention()


class _RmsGated(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.25]))
        self.variance_epsilon = 1e-6

    def forward(self, hidden, gate):
        normalized = hidden.float() * torch.rsqrt(
            hidden.float().square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        return (normalized * self.weight * torch.nn.functional.silu(gate.float())).to(
            hidden.dtype
        )


class _LinearAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj_z = nn.Linear(2, 2, bias=False)
        self.norm = _RmsGated()
        self.out_proj = nn.Linear(2, 2, bias=False)
        self.act = torch.nn.functional.silu
        with torch.no_grad():
            self.in_proj_z.weight.copy_(torch.tensor([[1.0, -0.5], [0.5, 1.0]]))
            self.out_proj.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.5]]))

    def forward(self, value):
        gate = self.in_proj_z(value).reshape(-1, 1)
        hidden = value.reshape(-1, 1)
        return self.out_proj(self.norm(hidden, gate).reshape(-1, 2))


class _LinearAttentionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mixer = _LinearAttention()


def test_targets_discover_fused_experts_without_model_type_branch():
    prefix = "model.language_model.layers.0.moe."

    class Index:
        weight_map = {
            prefix + "gate_up_proj": "unused",
            prefix + "down_proj": "unused",
        }

        @staticmethod
        def shape(name):
            return (3, 4, 5) if name.endswith("gate_up_proj") else (3, 5, 2)

    backend = SimpleNamespace(index=Index(), num_layers=1)
    by_layer, all_targets = _targets(backend, "brand_new_architecture")

    assert by_layer[0] == all_targets
    assert [(target.module_name, target.kind) for target in all_targets] == [
        ("moe", "expert_gate_up"),
        ("moe", "expert_down"),
    ]


def test_collector_accumulates_dense_second_moment():
    target = ImatrixTarget("dense.weight", "dense", 2)
    collector = ActivationImatrixCollector(
        (target,), torch.device("cpu"), accumulation_dtype=torch.float64
    )
    layer = _Layer()
    collector.install_layer(layer, 0, (target,))
    layer.dense(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    collector.close()

    entry = collector.entries()[target.name]
    np.testing.assert_allclose(entry.values, [[5.0, 10.0]])
    np.testing.assert_array_equal(entry.counts, [2])


def test_collector_accumulates_routed_gate_up_and_down_without_changing_output():
    gate = ImatrixTarget("experts.gate_up_proj", "experts", 2, 2, "expert_gate_up")
    down = ImatrixTarget("experts.down_proj", "experts", 1, 2, "expert_down")
    collector = ActivationImatrixCollector(
        (gate, down), torch.device("cpu"), accumulation_dtype=torch.float64
    )
    layer = _Layer()
    # Use more tokens than top-k slots so swapping torch.where's two returned
    # dimensions cannot accidentally remain in bounds.
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    selected = torch.tensor([[0, 1], [1, 0], [0, 1]])
    weights = torch.tensor([[0.75, 0.25], [0.4, 0.6], [0.3, 0.7]])
    expected = layer.experts(hidden, selected, weights)

    collector.install_layer(layer, 0, (gate, down))
    actual = layer.experts(hidden, selected, weights)
    collector.close()

    torch.testing.assert_close(actual, expected)
    entries = collector.entries()
    np.testing.assert_allclose(
        entries[gate.name].values,
        [[35.0 / 3.0, 56.0 / 3.0], [35.0 / 3.0, 56.0 / 3.0]],
    )
    np.testing.assert_array_equal(entries[gate.name].counts, [3, 3])
    np.testing.assert_allclose(entries[down.name].values, [[1048.0 / 3.0], [1048.0 / 3.0]])
    np.testing.assert_array_equal(entries[down.name].counts, [3, 3])


def test_collector_excludes_right_padding_from_dense_and_routed_statistics():
    dense = ImatrixTarget("dense.weight", "dense", 2)
    gate = ImatrixTarget("experts.gate_up_proj", "experts", 2, 2, "expert_gate_up")
    down = ImatrixTarget("experts.down_proj", "experts", 1, 2, "expert_down")
    collector = ActivationImatrixCollector(
        (dense, gate, down), torch.device("cpu"), accumulation_dtype=torch.float64
    )
    layer = _Layer()
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]])
    selected = torch.tensor([[0, 1], [1, 0], [0, 1]])
    weights = torch.full((3, 2), 0.5)

    collector.install_layer(layer, 0, (dense, gate, down))
    collector.set_valid_mask(torch.tensor([True, True, False]))
    layer.dense(hidden)
    layer.experts(hidden, selected, weights)
    collector.close()

    entries = collector.entries()
    np.testing.assert_allclose(entries[dense.name].values, [[5.0, 10.0]])
    np.testing.assert_array_equal(entries[dense.name].counts, [2])
    np.testing.assert_allclose(entries[gate.name].values, [[5.0, 10.0], [5.0, 10.0]])
    np.testing.assert_array_equal(entries[gate.name].counts, [2, 2])
    np.testing.assert_allclose(entries[down.name].values, [[74.0], [74.0]])
    np.testing.assert_array_equal(entries[down.name].counts, [2, 2])


def test_aaq_uses_compact_nonlinear_energy_for_fused_routed_gate_up():
    gate = ImatrixTarget("experts.gate_up_proj", "experts", 2, 2, "expert_gate_up")
    down = ImatrixTarget("experts.down_proj", "experts", 1, 2, "expert_down")
    layer = _SiluExpertLayer()
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    selected = torch.tensor([[0, 1], [1, 0], [0, 1]])
    weights = torch.tensor([[0.75, 0.25], [0.4, 0.6], [0.3, 0.7]])
    expected_output = layer.experts(hidden, selected, weights)
    collector = ActivationImatrixCollector(
        (gate, down),
        torch.device("cpu"),
        accumulation_dtype=torch.float64,
        nonlinear=True,
    )
    collector.install_layer(layer, 0, (gate, down))
    actual_output = layer.experts(hidden, selected, weights)
    collector.close()

    torch.testing.assert_close(actual_output, expected_output)
    expected = []
    for expert in range(2):
        token_idx, top_k_pos = torch.where(selected == expert)
        current = hidden[token_idx]
        projected = torch.nn.functional.linear(
            current, layer.experts.gate_up_proj[expert]
        )
        gate_value, up_value = projected.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate_value)
        probability = torch.sigmoid(gate_value)
        derivative = probability * (1.0 + gate_value * (1.0 - probability))
        downstream = layer.experts.down_proj[expert].float().square().sum(0)
        energy = (
            ((up_value * derivative).square() + activated.square()) * downstream
        ).sum(-1)
        energy *= weights[token_idx, top_k_pos].square()
        nonlinear = (current.square() * energy[:, None]).mean(0)
        ordinary = current.square().mean(0)
        nonlinear *= ordinary.mean() / nonlinear.mean()
        expected.append(nonlinear)
    entry = collector.entries()[gate.name]
    np.testing.assert_allclose(
        entry.values, torch.stack(expected).detach().numpy(), rtol=1e-6
    )
    np.testing.assert_array_equal(entry.counts, [3, 3])


def test_aaq_uses_silu_derivative_and_downstream_energy_for_dense_ffn_gate():
    targets = (
        ImatrixTarget("mlp.gate_proj.weight", "mlp.gate_proj", 2),
        ImatrixTarget("mlp.up_proj.weight", "mlp.up_proj", 2),
        ImatrixTarget("mlp.down_proj.weight", "mlp.down_proj", 2),
    )
    layer = _DenseGateLayer()
    value = torch.tensor([[1.0, 2.0], [-0.5, 1.5], [2.0, -1.0]])
    expected_output = layer.mlp(value)
    collector = ActivationImatrixCollector(
        targets,
        torch.device("cpu"),
        accumulation_dtype=torch.float64,
        nonlinear=True,
    )
    collector.install_layer(layer, 0, targets)
    actual_output = layer.mlp(value)
    collector.close()

    torch.testing.assert_close(actual_output, expected_output)
    gate = layer.mlp.gate_proj(value).float()
    up = layer.mlp.up_proj(value).float()
    probability = torch.sigmoid(gate)
    derivative = probability * (1.0 + gate * (1.0 - probability))
    coupled = (up * derivative).square().T @ value.square()
    downstream = layer.mlp.down_proj.weight.float().square().sum(0)
    ordinary = value.square().mean(0)
    expected = coupled * downstream[:, None] / value.shape[0]
    expected *= ordinary.mean() / expected.mean()
    entry = collector.entries()[targets[0].name]
    np.testing.assert_allclose(entry.values, expected.detach().numpy(), rtol=1e-6)
    np.testing.assert_array_equal(entry.counts, [3, 3])


def test_aaq_replaces_only_interleaved_attention_gate_rows():
    targets = (
        ImatrixTarget("attention.q_proj.weight", "attention.q_proj", 2),
        ImatrixTarget("attention.o_proj.weight", "attention.o_proj", 2),
    )
    layer = _AttentionLayer()
    value = torch.tensor([[1.0, 2.0], [-0.5, 1.5], [2.0, -1.0]])
    expected_output = layer.attention(value)
    collector = ActivationImatrixCollector(
        targets,
        torch.device("cpu"),
        accumulation_dtype=torch.float64,
        nonlinear=True,
    )
    collector.install_layer(layer, 0, targets)
    actual_output = layer.attention(value)
    collector.close()

    torch.testing.assert_close(actual_output, expected_output)
    projected = layer.attention.q_proj(value).reshape(-1, 2, 2)
    attended, gate = projected.chunk(2, dim=-1)
    gate = gate.reshape(-1, 2).float()
    gated = attended.reshape(-1, 2).float() * torch.sigmoid(gate)
    sensitivity = gated * (1.0 - torch.sigmoid(gate))
    ordinary = value.square().mean(0)
    coupled = sensitivity.square().T @ value.square()
    downstream = layer.attention.o_proj.weight.float().square().sum(0)
    expected_gate = coupled * downstream[:, None] / value.shape[0]
    expected_gate *= ordinary.mean() / expected_gate.mean()
    expected = ordinary.expand(4, -1).clone()
    expected[[1, 3]] = expected_gate
    entry = collector.entries()[targets[0].name]
    np.testing.assert_allclose(entry.values, expected.detach().numpy(), rtol=1e-6)
    np.testing.assert_array_equal(entry.counts, [3, 3, 3, 3])


def test_aaq_uses_gated_norm_activation_energy_for_linear_attention_gate():
    targets = (
        ImatrixTarget("mixer.in_proj_z.weight", "mixer.in_proj_z", 2),
        ImatrixTarget("mixer.out_proj.weight", "mixer.out_proj", 2),
    )
    layer = _LinearAttentionLayer()
    value = torch.tensor([[1.0, 2.0], [-0.5, 1.5], [2.0, -1.0]])
    expected_output = layer.mixer(value)
    collector = ActivationImatrixCollector(
        targets,
        torch.device("cpu"),
        accumulation_dtype=torch.float64,
        nonlinear=True,
    )
    collector.install_layer(layer, 0, targets)
    actual_output = layer.mixer(value)
    collector.close()

    torch.testing.assert_close(actual_output, expected_output)
    gate = layer.mixer.in_proj_z(value).reshape(-1, 1).float()
    hidden = value.reshape(-1, 1).float()
    normalized = hidden * torch.rsqrt(
        hidden.square().mean(-1, keepdim=True) + layer.mixer.norm.variance_epsilon
    )
    normalized *= layer.mixer.norm.weight.float()
    probability = torch.sigmoid(gate)
    derivative = probability * (1.0 + gate * (1.0 - probability))
    sensitivity = (normalized * derivative).reshape(-1, 2)
    coupled = sensitivity.square().T @ value.square()
    downstream = layer.mixer.out_proj.weight.float().square().sum(0)
    ordinary = value.square().mean(0)
    expected = coupled * downstream[:, None] / value.shape[0]
    expected *= ordinary.mean() / expected.mean()
    entry = collector.entries()[targets[0].name]
    np.testing.assert_allclose(entry.values, expected.detach().numpy(), rtol=1e-6)
    np.testing.assert_array_equal(entry.counts, [3, 3])


@pytest.mark.parametrize(
    ("backend", "device", "accumulation_dtype"),
    (("cuda", "cuda:0", "float64"), ("metal", "mps", "float32")),
)
def test_cli_dispatches_same_imatrix_collector_for_cuda_and_metal(
    monkeypatch, backend, device, accumulation_dtype
):
    import mfq.calibration.dataset as dataset_module
    import mfq.calibration.imatrix as imatrix_module
    from mfq.cli import _calibrate_imatrix

    class Corpus:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    corpus = Corpus()
    received = {}
    monkeypatch.setattr(dataset_module, "load_corpus", lambda path: corpus)

    def fake_collect(model, actual_corpus, output, **kwargs):
        received.update(model=model, corpus=actual_corpus, output=output, **kwargs)

    monkeypatch.setattr(imatrix_module, "collect_imatrix", fake_collect)
    args = SimpleNamespace(
        model="model",
        corpus="corpus",
        output="output.imatrix",
        backend=backend,
        device="",
        attention="sdpa",
        window_length=2048,
        batch_size=1,
        train_tokens=4096,
        seed=7,
        work_dir="",
        keep_hidden=False,
        accumulation_dtype="auto",
        objective="aaq",
    )

    assert _calibrate_imatrix(args) == 0
    assert received["corpus"] is corpus
    assert received["device"] == device
    assert received["accumulation_dtype"] == accumulation_dtype
    assert received["objective"] == "aaq"
