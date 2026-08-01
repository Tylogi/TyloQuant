import numpy as np
import pytest
import torch

from mfq.formats.npq0_s import unpack_npq0_s_tables
from mfq.quantize.nepq_train import (
    NepqBankTrainConfig,
    hadamard_diagonal_importance,
    signed_hadamard_rotate,
    train_nepq0_s_banks,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _samples():
    generator = torch.Generator(device="cuda")
    generator.manual_seed(17)
    value = torch.randn((4, 12, 96), generator=generator, device="cuda")
    row_scale = torch.linspace(0.2, 1.4, 12, device="cuda")
    expert_scale = torch.tensor([0.7, 1.0, 1.3, 1.7], device="cuda")
    return value * row_scale[None, :, None] * expert_scale[:, None, None]


def test_batched_nepq_bank_training_is_deterministic_and_finite():
    config = NepqBankTrainConfig(
        iterations=1,
        assignment_refine_steps=1,
        kmeans_iterations=2,
        expert_batch=2,
    )
    first = train_nepq0_s_banks(_samples(), config=config)
    second = train_nepq0_s_banks(_samples(), config=config)
    np.testing.assert_array_equal(first.table_payloads, second.table_payloads)
    np.testing.assert_allclose(
        first.weighted_nmse_percent,
        second.weighted_nmse_percent,
        rtol=0,
        atol=0,
    )
    assert first.table_payloads.shape == (4, 320)
    assert np.isfinite(first.weighted_nmse_percent).all()
    assert np.all(first.weighted_nmse_percent < 40.0)
    for payload in first.table_payloads:
        scale, first_codebook, second_codebook, consumed = unpack_npq0_s_tables(
            payload.tobytes()
        )
        assert consumed == 320
        assert scale.shape == (4,)
        assert first_codebook.shape == (4, 8, 4)
        assert second_codebook.shape == (4, 8, 4)


def test_signed_hadamard_rotation_preserves_energy():
    source = _samples()
    rotated = signed_hadamard_rotate(source, 32, 1234)
    torch.testing.assert_close(
        source.float().square().sum(),
        rotated.square().sum(),
        rtol=2e-6,
        atol=2e-5,
    )


def test_hadamard_diagonal_importance_is_block_mean():
    importance = torch.arange(1, 97, device="cuda", dtype=torch.float32).reshape(
        1, 96
    )
    diagonal = hadamard_diagonal_importance(importance, 32)
    expected = importance.reshape(1, 3, 32).mean(2, keepdim=True).expand(
        -1, -1, 32
    ).reshape_as(importance)
    torch.testing.assert_close(diagonal, expected, rtol=0, atol=0)


def test_batched_nepq_bank_training_consumes_imatrix():
    samples = _samples()
    importance = torch.ones((4, 96), device="cuda")
    importance[:, :32] = 64.0
    config = NepqBankTrainConfig(
        iterations=1,
        assignment_refine_steps=1,
        kmeans_iterations=2,
        expert_batch=2,
    )
    uniform = train_nepq0_s_banks(
        samples,
        config=config,
        rotation_block=32,
        rotation_seed=1234,
    )
    weighted = train_nepq0_s_banks(
        samples,
        importance=importance,
        config=config,
        rotation_block=32,
        rotation_seed=1234,
    )
    assert not np.array_equal(uniform.table_payloads, weighted.table_payloads)
    assert np.isfinite(weighted.weighted_nmse_percent).all()
