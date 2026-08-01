from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from mfq.formats import io  # noqa: E402
from mfq.formats.nint import NintSpec  # noqa: E402
from mfq.quantize.nint_quant import dequantize, quantize as quantize_cpu  # noqa: E402
from mfq.quantize.nint_quant_torch import (  # noqa: E402
    _imatrix_element_weights,
    _importance_as_rows,
    make_qkx3_cuda,
    make_qkx3_torch,
    make_qp_cuda,
    make_qp_torch,
    quantize_axis0 as quantize_gpu,
)
from mfq.tools.quantize_hf_to_mfq import convert  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")

_NINT_IMATRIX_SPECS = (
    NintSpec(2, 16, 5),
    NintSpec(3, 24, 5),
    NintSpec(4, 24, 6),
    NintSpec(5, 28, 7),
    NintSpec(6, 24, 7),
    NintSpec(6, 26, 7),
    NintSpec(8, 48, 7),
    NintSpec(8, 64, 7),
)


def test_nint_quant_torch_matches_cpu_quality():
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.05, size=(37, 113)).astype(np.float32)
    spec = NintSpec(4, 24, 6)
    cpu = quantize_cpu(W, spec, axis=0)
    gpu = quantize_gpu(torch.from_numpy(W), spec, device="cuda")

    assert gpu.shape == W.shape
    assert gpu.axis == 0
    assert gpu.q.shape == cpu.q.shape
    assert gpu.sub_scale.shape == cpu.sub_scale.shape
    rel_cpu = np.linalg.norm(dequantize(cpu) - W) / np.linalg.norm(W)
    rel_gpu = np.linalg.norm(dequantize(gpu) - W) / np.linalg.norm(W)
    assert rel_gpu <= rel_cpu * 1.08 + 1e-4


@pytest.mark.parametrize(
    "spec",
    _NINT_IMATRIX_SPECS,
)
def test_nint_imatrix_cuda_matches_cpu_quality(spec: NintSpec):
    rng = np.random.default_rng(23)
    weight = rng.normal(0, 0.05, size=(37, 113)).astype(np.float32)
    importance = np.geomspace(0.01, 100.0, weight.shape[1]).astype(np.float32)
    cpu = quantize_cpu(weight, spec, axis=0, importance=importance)
    gpu = quantize_gpu(
        torch.from_numpy(weight),
        spec,
        device="cuda",
        importance=importance,
    )

    cpu_error = float(
        np.sum(importance * (dequantize(cpu) - weight) ** 2)
    )
    gpu_error = float(
        np.sum(importance * (dequantize(gpu) - weight) ** 2)
    )
    assert gpu_error <= cpu_error * 1.08 + 1e-5


@pytest.mark.parametrize(
    "spec",
    _NINT_IMATRIX_SPECS,
)
def test_nint_imatrix_fused_cuda_matches_torch_objective(spec: NintSpec):
    rng = np.random.default_rng(31)
    weight = rng.normal(0, 0.05, size=(41, 137)).astype(np.float32)
    importance = np.geomspace(0.01, 100.0, weight.shape[1]).astype(np.float32)
    reference = quantize_gpu(
        torch.from_numpy(weight),
        spec,
        device="cuda",
        importance=importance,
        use_cuda_imatrix_kernels=False,
    )
    fused = quantize_gpu(
        torch.from_numpy(weight),
        spec,
        device="cuda",
        importance=importance,
        use_cuda_imatrix_kernels=True,
    )

    weight_cuda = torch.from_numpy(weight).to("cuda")
    importance_rows = _importance_as_rows(
        importance,
        weight.shape[0],
        weight.shape[1],
        "cuda",
    )
    objective_weight = _imatrix_element_weights(
        weight_cuda,
        importance_rows,
        weight.shape[1],
    ).cpu().numpy()
    reference_error = float(
        np.sum(
            objective_weight
            * (dequantize(reference) - weight) ** 2
        )
    )
    fused_error = float(
        np.sum(
            objective_weight
            * (dequantize(fused) - weight) ** 2
        )
    )
    assert fused_error <= reference_error * 1.001 + 1e-7
    assert np.isfinite(fused.neuron_scale).all()
    assert np.isfinite(fused.neuron_min).all()
    assert int(fused.q.max(initial=0)) <= spec.nmax
    assert int(fused.sub_scale.max(initial=0)) <= (1 << spec.sub_bits) - 1
    assert int(fused.sub_min.max(initial=0)) <= (1 << spec.sub_bits) - 1


def test_nint_qkx3_fused_cuda_matches_halfway_rounding_objective():
    torch.manual_seed(7)
    values = torch.randn(
        41, 6, 24, device="cuda", dtype=torch.float32
    ) * 0.05
    objective_weight = torch.rand_like(values) * 3.0
    reference_scale, reference_minimum = make_qkx3_torch(
        values, objective_weight, 15
    )
    fused_scale, fused_minimum = make_qkx3_cuda(
        values, objective_weight, 15
    )

    def weighted_error(
        scale: torch.Tensor,
        minimum: torch.Tensor,
    ) -> torch.Tensor:
        levels = torch.clamp(
            torch.round(
                (values - minimum.unsqueeze(-1))
                / scale.unsqueeze(-1)
            ),
            0,
            15,
        )
        difference = (
            scale.unsqueeze(-1) * levels
            + minimum.unsqueeze(-1)
            - values
        )
        return (objective_weight * difference.square()).sum()

    reference_error = weighted_error(
        reference_scale, reference_minimum
    )
    fused_error = weighted_error(fused_scale, fused_minimum)
    assert float(fused_error) <= float(reference_error) * 1.000001 + 1e-8


@pytest.mark.parametrize("width", (224, 896))
def test_nint_qp_fused_cuda_long_rows_match_reference_objective(width: int):
    torch.manual_seed(20260802 + width)
    values = torch.rand(7, width, device="cuda", dtype=torch.float32) * 0.08
    objective_weight = torch.exp(
        torch.linspace(-6.0, 6.0, width, device="cuda")
    ).unsqueeze(0).expand_as(values).contiguous()
    reference_scale, reference_levels = make_qp_torch(
        values, objective_weight, 127
    )
    fused_scale, fused_levels = make_qp_cuda(
        values, objective_weight, 127
    )

    def weighted_error(scale, levels):
        difference = values - scale.unsqueeze(-1) * levels
        return (objective_weight * difference.square()).sum(dtype=torch.float64)

    reference_error = weighted_error(reference_scale, reference_levels)
    fused_error = weighted_error(fused_scale, fused_levels)
    assert float(fused_error) <= float(reference_error) * 1.0001 + 1e-10


def test_hf_to_mfq_cuda_backend_smoke(tmp_path: Path):
    root = tmp_path / "hf"
    root.mkdir()
    tensors = {
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(17, 29, dtype=torch.bfloat16),
        "model.language_model.layers.0.input_layernorm.weight": torch.ones(29, dtype=torch.bfloat16),
        "model.language_model.layers.0.linear_attn.conv1d.weight": torch.randn(17, 1, 4, dtype=torch.bfloat16),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(root / shard))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard for k in tensors}}),
        encoding="utf-8",
    )
    (root / "config.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "tiny.mfq"

    class Args:
        input = str(root)
        output = str(out)
        bits = 4
        groupsize = 24
        sub_bits = 6
        row_chunk = 6
        quant_backend = "cuda"
        device = "cuda"
        text_only = False
        limit_tensors = 0
        overwrite = False
        keep_temp = False

    convert(Args())
    header, loaded = io.load(out)
    assert header.extra["quant_backend"] == "cuda"
    assert loaded["model.language_model.layers.0.mlp.gate_proj.weight"].shape == (17, 29)
    assert loaded["model.language_model.layers.0.input_layernorm.weight"].dtype == np.float16
    assert loaded["model.language_model.layers.0.linear_attn.conv1d.weight"].dtype == np.float16
    assert not (out.parent / f".{out.name}.tmp_blobs").exists()
