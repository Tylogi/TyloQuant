from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mfq.calibration.imatrix import ActivationImatrixCollector, ImatrixTarget


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
    )

    assert _calibrate_imatrix(args) == 0
    assert received["corpus"] is corpus
    assert received["device"] == device
    assert received["accumulation_dtype"] == accumulation_dtype
