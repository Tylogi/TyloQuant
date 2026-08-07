from __future__ import annotations

import numpy as np
import pytest
import torch

from mfq.formats.npq0_l import pack_npq0_l, unpack_npq0_l
from mfq.quantize.npq0_l import (
    Npq0LConfig,
    Npq0LTables,
    dequantize_npq0_l,
    npq0_l_tables_from_tensor,
    quantize_npq0_l_fixed,
    train_npq0_l,
)


def _config() -> Npq0LConfig:
    return Npq0LConfig(
        iterations=1,
        assignment_refine_steps=1,
        fixed_refine_steps=1,
        kmeans_iterations=2,
        kmeans_initialization_points=128,
        group_chunk=16,
        anchor_multipliers=(0.8, 1.0, 1.25),
        seed=17,
    )


def test_npq0_l_cpu_training_and_fixed_table_assignment() -> None:
    generator = torch.Generator().manual_seed(18)
    train = 0.04 * torch.randn((8, 48), generator=generator)
    validation = 0.04 * torch.randn((4, 48), generator=generator)
    trained, history = train_npq0_l(train, config=_config(), device="cpu")
    fixed = quantize_npq0_l_fixed(
        validation,
        npq0_l_tables_from_tensor(trained),
        config=_config(),
        device="cpu",
    )
    reconstruction = dequantize_npq0_l(fixed)
    assert reconstruction.shape == tuple(validation.shape)
    assert np.isfinite(reconstruction).all()
    assert min(item.weighted_sse for item in history) <= history[0].weighted_sse
    signal = float(validation.square().sum())
    error = float(np.square(reconstruction - validation.numpy()).sum())
    assert 100.0 * error / signal < 45.0


def test_npq0_l_serialized_reconstruction_preserves_training_error() -> None:
    weight = torch.linspace(-0.07, 0.08, 6 * 40, dtype=torch.float32).reshape(6, 40)
    tensor, history = train_npq0_l(weight, config=_config(), device="cpu")
    restored = unpack_npq0_l(pack_npq0_l(tensor))
    reconstruction = dequantize_npq0_l(restored)
    error = float(np.square(reconstruction - weight.numpy()).sum())
    assert error == pytest.approx(min(item.weighted_sse for item in history), rel=2e-3, abs=1e-7)


def test_npq0_l_rejects_non_vec8_width() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        train_npq0_l(torch.zeros((2, 34)), config=_config(), device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_npq0_l_native_fixed_assignment_matches_torch_reference() -> None:
    from mfq.quantize.cuda._ext import ext
    from mfq.quantize.npq0_l import (
        _assign_groups,
        _refit_anchor_and_lut,
        _validate_tables,
    )
    from mfq.quantize.nvq_quant_torch import _fp16_round, _pad_weight

    rng = np.random.default_rng(91)
    tables = Npq0LTables(
        scale_lut=np.linspace(0.2, 1.5, 8, dtype=np.float32),
        first_codebooks=rng.integers(-31, 32, (8, 8, 4), dtype=np.int8),
        second_codebooks=rng.integers(-31, 32, (8, 16, 4), dtype=np.int8),
    )
    scale_np, first_np, second_np = _validate_tables(tables)
    generator = torch.Generator(device="cuda").manual_seed(93)
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
        ext().npq0_l_assign(
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
