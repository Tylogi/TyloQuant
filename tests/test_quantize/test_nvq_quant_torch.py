from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mfq.formats.nvq import NVQ2_E8, NVQ3_D4  # noqa: E402
from mfq.formats.nvq1_l import NVQ1_L_T8_S3  # noqa: E402
from mfq.quantize.nvq1_l_quant import dequantize as dequantize_nvq1_l  # noqa: E402
from mfq.quantize.nvq1_l_quant import quantize as quantize_nvq1_l_cpu  # noqa: E402
from mfq.quantize.nvq_quant import dequantize as dequantize_nvq  # noqa: E402
from mfq.quantize.nvq_quant import quantize as quantize_nvq_cpu  # noqa: E402
from mfq.quantize.nvq_quant_torch import quantize_axis0  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("spec", [NVQ2_E8, NVQ3_D4])
def test_cuda_nvq_matches_reference_quality(spec):
    rng = np.random.default_rng(20260716 + spec.vector_size)
    weight = rng.normal(0, 0.05, size=(5, 53)).astype(np.float32)
    cpu = quantize_nvq_cpu(weight, spec, axis=0, group_chunk=32)
    gpu = quantize_axis0(torch.from_numpy(weight), spec, group_chunk=32)
    cpu_recon = dequantize_nvq(cpu)
    gpu_recon = dequantize_nvq(gpu)
    cpu_sse = float(np.square(cpu_recon - weight).sum())
    gpu_sse = float(np.square(gpu_recon - weight).sum())
    assert gpu_sse <= cpu_sse * 1.0005 + 1e-8
    assert gpu.shape == weight.shape


@pytest.mark.parametrize("spec", [NVQ2_E8, NVQ3_D4])
def test_native_nvq_assignment_matches_torch_path_exactly(spec):
    weight = torch.from_numpy(
        np.random.default_rng(20260722 + spec.vector_size)
        .normal(0, 0.05, size=(5, 53))
        .astype(np.float32)
    )
    kwargs = {
        "device": "cuda",
        "search_steps": 5,
        "group_chunk": 64,
    }
    native = quantize_axis0(
        weight,
        spec,
        nvq_native_assignment=True,
        **kwargs,
    )
    reference = quantize_axis0(
        weight,
        spec,
        nvq_native_assignment=False,
        **kwargs,
    )
    np.testing.assert_array_equal(native.neuron_scale, reference.neuron_scale)
    np.testing.assert_array_equal(native.sub_scale, reference.sub_scale)
    np.testing.assert_array_equal(native.indices, reference.indices)
    np.testing.assert_array_equal(native.signs, reference.signs)


@pytest.mark.parametrize("spec", [NVQ2_E8, NVQ3_D4])
def test_weighted_native_nvq_assignment_matches_torch_path_exactly(spec):
    rng = np.random.default_rng(20260723 + spec.vector_size)
    weight = torch.from_numpy(rng.normal(0, 0.05, size=(5, 53)).astype(np.float32))
    importance = rng.lognormal(0.0, 1.0, size=(5, 53)).astype(np.float32)
    importance[:, ::11] = 0.0
    kwargs = {
        "device": "cuda",
        "importance": importance,
        "search_steps": 5,
        "group_chunk": 64,
    }
    native = quantize_axis0(weight, spec, nvq_native_assignment=True, **kwargs)
    reference = quantize_axis0(weight, spec, nvq_native_assignment=False, **kwargs)
    np.testing.assert_array_equal(native.neuron_scale, reference.neuron_scale)
    np.testing.assert_array_equal(native.sub_scale, reference.sub_scale)
    np.testing.assert_array_equal(native.indices, reference.indices)
    np.testing.assert_array_equal(native.signs, reference.signs)


def test_cuda_nvq1_l_matches_reference_quality():
    rng = np.random.default_rng(20260717)
    weight = rng.normal(0, 0.05, size=(3, 48)).astype(np.float32)
    cpu = quantize_nvq1_l_cpu(weight, NVQ1_L_T8_S3, axis=0, group_chunk=16)
    gpu = quantize_axis0(
        torch.from_numpy(weight),
        NVQ1_L_T8_S3,
        group_chunk=16,
    )
    cpu_recon = dequantize_nvq1_l(cpu)
    gpu_recon = dequantize_nvq1_l(gpu)
    cpu_sse = float(np.square(cpu_recon - weight).sum())
    gpu_sse = float(np.square(gpu_recon - weight).sum())
    assert gpu_sse <= cpu_sse * 1.0005 + 1e-8
    assert gpu.shape == weight.shape


def test_native_nvq1_l_assignment_matches_torch_path_exactly():
    weight = torch.from_numpy(
        np.random.default_rng(20260721).normal(0, 0.05, size=(5, 53)).astype(np.float32)
    )
    kwargs = {
        "device": "cuda",
        "anchor_multipliers": (0.75,),
        "refine_steps": 2,
        "group_chunk": 64,
        "nvq1_l_candidates": 0,
    }
    native = quantize_axis0(
        weight,
        NVQ1_L_T8_S3,
        nvq1_l_native_assignment=True,
        **kwargs,
    )
    reference = quantize_axis0(
        weight,
        NVQ1_L_T8_S3,
        nvq1_l_native_assignment=False,
        **kwargs,
    )
    np.testing.assert_array_equal(native.neuron_scale, reference.neuron_scale)
    np.testing.assert_array_equal(native.sub_scale, reference.sub_scale)
    np.testing.assert_array_equal(native.delta_sign, reference.delta_sign)
    np.testing.assert_array_equal(native.indices, reference.indices)


def test_weighted_native_nvq1_l_assignment_matches_torch_path_exactly():
    rng = np.random.default_rng(20260724)
    weight = torch.from_numpy(rng.normal(0, 0.05, size=(5, 53)).astype(np.float32))
    importance = rng.lognormal(0.0, 1.0, size=53).astype(np.float32)
    importance[::11] = 0.0
    kwargs = {
        "device": "cuda",
        "importance": importance,
        "anchor_multipliers": (0.75,),
        "refine_steps": 2,
        "group_chunk": 64,
        "nvq1_l_candidates": 0,
    }
    native = quantize_axis0(weight, NVQ1_L_T8_S3, nvq1_l_native_assignment=True, **kwargs)
    reference = quantize_axis0(weight, NVQ1_L_T8_S3, nvq1_l_native_assignment=False, **kwargs)
    np.testing.assert_array_equal(native.neuron_scale, reference.neuron_scale)
    np.testing.assert_array_equal(native.sub_scale, reference.sub_scale)
    np.testing.assert_array_equal(native.delta_sign, reference.delta_sign)
    np.testing.assert_array_equal(native.indices, reference.indices)
