from __future__ import annotations

import numpy as np
import pytest
import torch

from mfq.quantize.nvq_jsc import NvqJscConfig, train_nvq_jsc
from mfq.quantize.nvq_product import (
    NvqProductConfig,
    dequantize_nvq_product,
    product_tables_from_tensor,
    quantize_nvq_product_fixed,
    train_nvq_product,
)


def _configs() -> tuple[NvqJscConfig, NvqProductConfig]:
    return (
        NvqJscConfig(
            banks=1,
            iterations=0,
            assignment_refine_steps=1,
            search_steps=3,
            group_chunk=16,
        ),
        NvqProductConfig(
            banks=1,
            iterations=1,
            assignment_refine_steps=1,
            fixed_refine_steps=1,
            kmeans_iterations=1,
            kmeans_initialization_points=64,
            group_chunk=8,
            anchor_multipliers=(0.75, 1.0, 1.25),
            seed=17,
        ),
    )


def test_nvq_product_roundtrip_and_rate_match_on_cpu() -> None:
    generator = torch.Generator().manual_seed(11)
    weight = 0.04 * torch.randn((8, 48), generator=generator)
    importance = np.linspace(0.5, 1.5, 48, dtype=np.float32)
    jsc_config, product_config = _configs()
    initial, _ = train_nvq_jsc(
        weight,
        importance=importance,
        config=jsc_config,
        device="cpu",
    )
    product, history = train_nvq_product(
        weight,
        initial,
        importance=importance,
        config=product_config,
        device="cpu",
    )
    reconstruction = dequantize_nvq_product(product)

    assert reconstruction.shape == tuple(weight.shape)
    assert np.isfinite(reconstruction).all()
    assert product.first_codebooks.shape == (1, 256, 4)
    assert product.second_codebooks.shape == (1, 128, 4)
    assert product.first_indices.shape == (8, 6)
    assert product.second_indices.shape == (8, 6)
    assert int(product.second_indices.max()) < 128
    assert len(history) == 2
    assert min(item.weighted_sse for item in history) <= history[0].weighted_sse

    # Both streams spend 15 bits per 8-D vector. SPQ has a smaller table.
    baseline_streams = initial.payload_nbytes - 64 - initial.codebooks.size
    product_tables = 16 * 2 + 16 + product.first_codebooks.size + product.second_codebooks.size
    product_streams = product.payload_nbytes - product_tables
    assert product_streams == baseline_streams
    assert product.payload_nbytes < initial.payload_nbytes


def test_nvq_product_fixed_tables_do_not_use_validation_rows_for_training() -> None:
    generator = torch.Generator().manual_seed(23)
    train = 0.03 * torch.randn((8, 48), generator=generator)
    validation = 0.03 * torch.randn((4, 48), generator=generator)
    jsc_config, product_config = _configs()
    initial, _ = train_nvq_jsc(train, config=jsc_config, device="cpu")
    trained, _ = train_nvq_product(train, initial, config=product_config, device="cpu")
    fixed = quantize_nvq_product_fixed(
        validation,
        product_tables_from_tensor(trained),
        config=product_config,
        device="cpu",
    )
    reconstruction = dequantize_nvq_product(fixed)

    assert fixed.shape == tuple(validation.shape)
    assert np.isfinite(reconstruction).all()
    assert fixed.first_codebooks.tobytes() == trained.first_codebooks.tobytes()
    assert fixed.second_codebooks.tobytes() == trained.second_codebooks.tobytes()
    assert np.square(reconstruction - validation.numpy()).sum() > 0


def test_nvq_product_rejects_bank_mismatch() -> None:
    generator = torch.Generator().manual_seed(29)
    weight = 0.02 * torch.randn((4, 24), generator=generator)
    initial, _ = train_nvq_jsc(
        weight,
        config=NvqJscConfig(
            banks=1,
            iterations=0,
            assignment_refine_steps=0,
            search_steps=3,
            group_chunk=8,
        ),
        device="cpu",
    )
    with pytest.raises(ValueError, match="bank count"):
        train_nvq_product(
            weight,
            initial,
            config=NvqProductConfig(
                banks=2,
                iterations=0,
                kmeans_iterations=0,
                kmeans_initialization_points=16,
                group_chunk=8,
            ),
            device="cpu",
        )
