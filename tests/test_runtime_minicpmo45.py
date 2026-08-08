from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from mfq.formats import io
from mfq.formats.nint import NintSpec
from mfq.quantize import nint_quant
from mfq.runtime.minicpmo45 import (
    _dense_to_torch,
    _MfqEmbeddingModule,
    _MfqLinearModule,
    _record_is_quantized,
    _replace_module,
    _validate_matching_graph_config,
    _validate_minicpmo45_config,
)
from mfq.runtime.torch_linear import TorchNintEmbedding, TorchNintLinear


def _quantized(weight: np.ndarray):
    return nint_quant.quantize(
        np.asarray(weight, dtype=np.float32),
        NintSpec(4, 16, 6),
        axis=0,
    )


def test_minicpmo45_config_validation():
    config = {
        "model_type": "minicpmo",
        "version": "4.5",
        "hidden_size": 4096,
    }
    _validate_minicpmo45_config(config)
    _validate_matching_graph_config(config, dict(config))

    with pytest.raises(ValueError, match="version 4.5"):
        _validate_minicpmo45_config({"model_type": "minicpmo", "version": "4.0"})


def test_minicpmo45_dense_bfloat16_bits_are_preserved():
    bits = np.asarray([[0x3F80, 0xC020], [0x0000, 0x7F80]], dtype=np.uint16)
    tagged = bits.view(io.BFloat16Array)

    result = _dense_to_torch(tagged, "cpu")

    assert result.dtype == torch.bfloat16
    np.testing.assert_array_equal(result.view(torch.uint16).numpy(), bits)


def test_minicpmo45_record_dtype_detection():
    assert _record_is_quantized("NINT4")
    assert _record_is_quantized("NVQ3J512")
    assert _record_is_quantized("NPQ0-L")
    assert not _record_is_quantized("BF16")


def test_minicpmo45_replaces_nested_module_list_entry():
    root = nn.Module()
    root.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
    replacement = nn.Identity()

    _replace_module(root, "layers.0", replacement)

    assert root.layers[0] is replacement


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_minicpmo45_quantized_linear_and_embedding_wrappers():
    rng = np.random.default_rng(11)
    linear_weight = rng.normal(0, 0.1, size=(6, 16)).astype(np.float32)
    embedding_weight = rng.normal(0, 0.1, size=(32, 16)).astype(np.float32)
    bias = rng.normal(0, 0.1, size=(6,)).astype(np.float32)
    linear_tensor = _quantized(linear_weight)
    embedding_tensor = _quantized(embedding_weight)
    source_linear = nn.Linear(16, 6, bias=True, device="meta")
    source_embedding = nn.Embedding(32, 16, device="meta")

    linear = _MfqLinearModule(
        linear_tensor,
        source_linear,
        bias=bias,
        device="cuda",
        weight_dtype=torch.bfloat16,
    )
    embedding = _MfqEmbeddingModule(
        embedding_tensor,
        source_embedding,
        device="cuda",
        weight_dtype=torch.bfloat16,
    )
    direct_linear = TorchNintLinear(linear_tensor, "cuda")
    direct_embedding = TorchNintEmbedding(embedding_tensor, "cuda")
    x = torch.randn((2, 3, 16), device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[1, 3, 5]], device="cuda", dtype=torch.int64)

    expected_linear = direct_linear(x).to(torch.bfloat16) + torch.as_tensor(
        bias,
        device="cuda",
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(linear(x), expected_linear, atol=0, rtol=0)
    torch.testing.assert_close(
        embedding(ids),
        direct_embedding(ids).to(torch.bfloat16),
        atol=0,
        rtol=0,
    )
    assert linear.weight.dtype == torch.bfloat16
    assert linear(x).dtype == torch.bfloat16
    assert embedding(ids).dtype == torch.bfloat16
    assert embedding.weight.device.type == "cuda"
