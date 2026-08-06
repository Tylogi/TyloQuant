from __future__ import annotations

import numpy as np
import pytest
import torch

from mfq.formats.npq0_s import pack_npq0_s, unpack_npq0_s
from mfq.quantize.npq0_s import (
    Npq0SConfig,
    Npq0STables,
    dequantize_npq0_s,
    npq0_s_tables_from_tensor,
    quantize_npq0_s_fixed,
    train_npq0_s,
)


def _config() -> Npq0SConfig:
    return Npq0SConfig(
        iterations=1,
        assignment_refine_steps=1,
        fixed_refine_steps=1,
        kmeans_iterations=2,
        kmeans_initialization_points=128,
        group_chunk=16,
        anchor_multipliers=(0.8, 1.0, 1.25),
        seed=23,
    )


def test_npq0_s_cpu_training_and_fixed_table_assignment() -> None:
    generator = torch.Generator().manual_seed(24)
    train = 0.04 * torch.randn((12, 48), generator=generator)
    validation = 0.04 * torch.randn((4, 48), generator=generator)
    trained, history = train_npq0_s(train, config=_config(), device="cpu")
    fixed = quantize_npq0_s_fixed(
        validation,
        npq0_s_tables_from_tensor(trained),
        config=_config(),
        device="cpu",
    )
    reconstruction = dequantize_npq0_s(fixed)
    assert reconstruction.shape == tuple(validation.shape)
    assert np.isfinite(reconstruction).all()
    assert min(item.weighted_sse for item in history) <= history[0].weighted_sse
    signal = float(validation.square().sum())
    error = float(np.square(reconstruction - validation.numpy()).sum())
    assert 100.0 * error / signal < 65.0


def test_npq0_s_serialized_reconstruction_preserves_training_error() -> None:
    weight = torch.linspace(-0.07, 0.08, 8 * 40, dtype=torch.float32).reshape(8, 40)
    tensor, history = train_npq0_s(weight, config=_config(), device="cpu")
    restored = unpack_npq0_s(pack_npq0_s(tensor))
    reconstruction = dequantize_npq0_s(restored)
    error = float(np.square(reconstruction - weight.numpy()).sum())
    assert error == pytest.approx(min(item.weighted_sse for item in history), rel=2e-3, abs=1e-7)


def test_npq0_s_index_selects_a_cartesian_product_codeword() -> None:
    weight = torch.zeros((1, 24), dtype=torch.float32)
    tensor, _ = train_npq0_s(weight, config=_config(), device="cpu")
    tensor.neuron_scale[:] = 1.0
    tensor.scale_lut[:] = 1.0
    tensor.state[:] = 0
    tensor.first_codebooks.fill(0)
    tensor.second_codebooks.fill(0)
    for index in range(8):
        tensor.first_codebooks[0, index] = np.arange(4, dtype=np.int8) + 10 * index
        tensor.second_codebooks[0, index] = -np.arange(4, dtype=np.int8) - 10 * index

    first_indices = np.array([0, 3, 7], dtype=np.uint8)
    second_indices = np.array([2, 5, 7], dtype=np.uint8)
    tensor.indices[0] = first_indices | (second_indices << 3)
    expected = np.concatenate(
        (
            tensor.first_codebooks[0, first_indices],
            tensor.second_codebooks[0, second_indices],
        ),
        axis=1,
    ).reshape(1, 24)
    np.testing.assert_array_equal(dequantize_npq0_s(tensor), expected)


def test_npq0_s_rejects_non_vec8_width() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        train_npq0_s(torch.zeros((2, 34)), config=_config(), device="cpu")


def test_npq0_s_weighted_kmeans_does_not_clip_training_samples() -> None:
    from mfq.quantize.npq0_s import _weighted_kmeans

    samples = torch.tensor(
        [[200.0, 200.0, 200.0, 200.0], [-100.0, -100.0, -100.0, -100.0]]
    )
    objective_weight = torch.ones_like(samples)
    table = _weighted_kmeans(
        samples,
        objective_weight,
        1,
        iterations=1,
        initialization_points=2,
        seed=0,
    )
    assert torch.equal(table, torch.full((1, 4), 50, dtype=torch.int8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_npq0_s_native_fixed_assignment_matches_torch_reference() -> None:
    from mfq.quantize.cuda._ext import ext
    from mfq.quantize.npq0_s import (
        _assign_groups,
        _refit_anchor_and_lut,
        _validate_tables,
    )
    from mfq.quantize.nvq_quant_torch import _fp16_round, _pad_weight

    rng = np.random.default_rng(81)
    tables = Npq0STables(
        scale_lut=np.linspace(0.2, 1.5, 4, dtype=np.float32),
        first_codebooks=rng.integers(-31, 32, (4, 8, 4), dtype=np.int8),
        second_codebooks=rng.integers(-31, 32, (4, 8, 4), dtype=np.int8),
    )
    scale_np, first_np, second_np = _validate_tables(tables)
    generator = torch.Generator(device="cuda").manual_seed(83)
    value = 0.05 * torch.randn((3, 40), generator=generator, device="cuda")
    importance = 0.2 + torch.rand((40,), generator=generator, device="cuda")
    padded, objective_weight, ng = _pad_weight(value, 24, importance)
    xgroup = padded.reshape(-1, 24).contiguous()
    wgroup = objective_weight.reshape_as(xgroup).contiguous()
    scale_lut = torch.as_tensor(scale_np, device="cuda")
    first = torch.as_tensor(first_np, device="cuda")
    second = torch.as_tensor(second_np, device="cuda")
    maximum_code = max(int(np.abs(first_np).max()), int(np.abs(second_np).max()))
    anchor = _fp16_round(
        padded.abs().amax(1) / (maximum_code * float(scale_np.max()))
    ).contiguous()

    state, first_index, second_index, _ = _assign_groups(
        xgroup,
        wgroup,
        anchor,
        scale_lut,
        first,
        second,
        ng=ng,
        group_chunk=16,
    )
    fitted, _ = _refit_anchor_and_lut(
        xgroup,
        wgroup,
        state,
        first_index,
        second_index,
        anchor,
        scale_lut,
        first,
        second,
        out=3,
        ng=ng,
        learn_lut=False,
    )
    ref_state, ref_first, ref_second, ref_error = _assign_groups(
        xgroup,
        wgroup,
        fitted,
        scale_lut,
        first,
        second,
        ng=ng,
        group_chunk=16,
    )
    native_anchor, native_state, native_first, native_second, native_error = (
        ext().npq0_s_assign(
            padded.contiguous(),
            objective_weight.contiguous(),
            anchor,
            scale_lut.contiguous(),
            first.contiguous(),
            second.contiguous(),
            40,
            1,
        )
    )

    torch.testing.assert_close(native_anchor, fitted, rtol=0, atol=0)
    assert torch.equal(native_state.reshape(-1).to(torch.int64), ref_state)
    assert torch.equal(native_first.reshape(-1, 3).to(torch.int64), ref_first)
    assert torch.equal(native_second.reshape(-1, 3).to(torch.int64), ref_second)
    torch.testing.assert_close(
        native_error,
        ref_error.reshape(3, ng).sum(1),
        rtol=2e-6,
        atol=2e-7,
    )
