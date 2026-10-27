"""A zero nucleus must select argmax before RNG or extension compilation."""

from types import SimpleNamespace

import pytest
import torch

from mfq.kernels.cuda import sampling


@pytest.mark.parametrize("top_k", [0, 1, 20])
def test_python_cuda_zero_nucleus_uses_greedy(monkeypatch, top_k):
    logits = torch.tensor([[1.0, 5.0, 2.0], [4.0, 4.0, -1.0]])
    monkeypatch.setattr(sampling, "ext", lambda: SimpleNamespace(
        sample_greedy_cuda=lambda values: values.argmax(dim=-1)))

    def unexpected_rng(*args, **kwargs):
        pytest.fail("zero nucleus consumed RNG")

    monkeypatch.setattr(torch, "rand", unexpected_rng)
    result = sampling.sample(logits, temperature=1.0, top_k=top_k, top_p=0.0)
    assert result.tolist() == [1, 0]
