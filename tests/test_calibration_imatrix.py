from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mfq.calibration.imatrix import (
    ActivationImatrixCollector,
    ImatrixTarget,
    _backend,
    _targets,
)
from mfq.calibration.layerwise_qwen35 import _create_causal_mask_compat


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


@pytest.mark.parametrize("model_type", ("qwen3_5_text", "qwen3_5_moe_text"))
def test_backend_normalizes_text_subconfig_family(monkeypatch, tmp_path, model_type):
    import transformers

    import mfq.calibration.layerwise_qwen35 as qwen_backend

    sentinel = object()
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            text_config=SimpleNamespace(model_type=model_type)
        ),
    )
    monkeypatch.setattr(
        qwen_backend,
        "Qwen35LayerwiseBackend",
        lambda *_args, **_kwargs: sentinel,
    )

    backend, resolved_type = _backend(tmp_path, torch.device("cpu"), "sdpa")

    assert backend is sentinel
    assert resolved_type == model_type


def test_causal_mask_compat_supports_current_and_legacy_transformers():
    current_calls = []

    def current(**arguments):
        current_calls.append(arguments)
        return "current"

    assert _create_causal_mask_compat(current, value=1) == "current"
    assert current_calls == [{"value": 1}]

    legacy_calls = []

    def legacy(**arguments):
        legacy_calls.append(arguments)
        if "cache_position" not in arguments:
            raise TypeError("missing required keyword-only argument: 'cache_position'")
        return "legacy"

    assert _create_causal_mask_compat(legacy, value=2) == "legacy"
    assert legacy_calls[-1] == {"value": 2, "cache_position": None}


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


def test_naq_collects_input_channels_and_output_neurons_for_ordinary_linear():
    target = ImatrixTarget("proj.weight", "proj", 2)
    layer = nn.Module()
    layer.proj = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        layer.proj.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
        )
    value = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    collector = ActivationImatrixCollector(
        (target,),
        torch.device("cpu"),
        accumulation_dtype=torch.float64,
        neural=True,
    )
    collector.install_layer(layer, 0, (target,))
    output = layer.proj(value)
    collector.close()

    entry = collector.entries()[target.name]
    expected_neurons = output.square().mean(0)
    expected_neurons /= expected_neurons.mean()
    np.testing.assert_allclose(entry.values, value.square().mean(0).numpy()[None, :])
    np.testing.assert_allclose(entry.row_importance, expected_neurons.detach().numpy())
    assert collector.naq_categories[target.name] == "linear"


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


def test_naq_uses_activation_corrected_energy_for_fused_routed_gate_up():
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
        neural=True,
    )
    collector.install_layer(layer, 0, (gate, down))
    actual_output = layer.experts(hidden, selected, weights)
    collector.close()

    torch.testing.assert_close(actual_output, expected_output)
    expected_rows = []
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
        route_weight2 = weights[token_idx, top_k_pos].square()[:, None]
        gate_rows = (
            (up_value * derivative).square() * downstream * route_weight2
        ).mean(0)
        up_rows = (activated.square() * downstream * route_weight2).mean(0)
        corrected = torch.cat((gate_rows, up_rows))
        corrected /= corrected.mean()
        expected_rows.append(corrected)
    entry = collector.entries()[gate.name]
    np.testing.assert_allclose(
        entry.values,
        torch.stack(
            [hidden[torch.where(selected == expert)[0]].square().mean(0) for expert in range(2)]
        ).numpy(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        entry.row_importance,
        torch.stack(expected_rows).reshape(-1).detach().numpy(),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(entry.counts, [3, 3])


def test_naq_uses_silu_and_downstream_energy_for_dense_ffn_gate_and_up():
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
        neural=True,
    )
    collector.install_layer(layer, 0, targets)
    actual_output = layer.mlp(value)
    collector.close()

    torch.testing.assert_close(actual_output, expected_output)
    gate = layer.mlp.gate_proj(value).float()
    up = layer.mlp.up_proj(value).float()
    probability = torch.sigmoid(gate)
    derivative = probability * (1.0 + gate * (1.0 - probability))
    downstream = layer.mlp.down_proj.weight.float().square().sum(0)
    ordinary = value.square().mean(0)
    expected_gate = (up * derivative).square().mean(0) * downstream
    expected_gate /= expected_gate.mean()
    expected_up = torch.nn.functional.silu(gate).square().mean(0) * downstream
    expected_up /= expected_up.mean()
    entries = collector.entries()
    for target, expected in (
        (targets[0], expected_gate),
        (targets[1], expected_up),
    ):
        entry = entries[target.name]
        np.testing.assert_allclose(
            entry.values, ordinary.detach().numpy()[None, :], rtol=1e-6
        )
        np.testing.assert_allclose(
            entry.row_importance, expected.detach().numpy(), rtol=1e-6
        )
        np.testing.assert_array_equal(entry.counts, [3])


def test_naq_corrects_only_interleaved_attention_gate_neurons():
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
        neural=True,
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
    downstream = layer.attention.o_proj.weight.float().square().sum(0)
    expected_gate = sensitivity.square().mean(0) * downstream
    expected = layer.attention.q_proj(value).float().square().mean(0)
    expected[[1, 3]] = expected_gate
    expected /= expected.mean()
    entry = collector.entries()[targets[0].name]
    np.testing.assert_allclose(entry.values, ordinary.detach().numpy()[None, :], rtol=1e-6)
    np.testing.assert_allclose(entry.row_importance, expected.detach().numpy(), rtol=1e-6)
    np.testing.assert_array_equal(entry.counts, [3])


def test_naq_uses_gated_norm_activation_energy_for_linear_attention_gate():
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
        neural=True,
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
    downstream = layer.mixer.out_proj.weight.float().square().sum(0)
    ordinary = value.square().mean(0)
    expected = sensitivity.square().mean(0) * downstream
    expected /= expected.mean()
    entry = collector.entries()[targets[0].name]
    np.testing.assert_allclose(entry.values, ordinary.detach().numpy()[None, :], rtol=1e-6)
    np.testing.assert_allclose(entry.row_importance, expected.detach().numpy(), rtol=1e-6)
    np.testing.assert_array_equal(entry.counts, [3])


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
        objective="naq",
    )

    assert _calibrate_imatrix(args) == 0
    assert received["corpus"] is corpus
    assert received["device"] == device
    assert received["accumulation_dtype"] == accumulation_dtype
    assert received["objective"] == "naq"


def test_cli_defaults_to_naq_imatrix():
    from mfq.cli import _build_parser

    args = _build_parser().parse_args(
        [
            "calibrate",
            "imatrix",
            "--model",
            "model",
            "--corpus",
            "corpus",
            "--output",
            "output.imatrix",
        ]
    )

    assert args.objective == "naq"
