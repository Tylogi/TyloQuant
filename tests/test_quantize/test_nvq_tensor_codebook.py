from __future__ import annotations

import numpy as np
import pytest

from mfq.formats.nvq import NVQ2_E8, NVQ3_D4, validate_codebook
from mfq.formats.nvq1_l import NVQ1_L_T8_S3, validate_ternary_codebook
from mfq.quantize.nvq_quant import dequantize, quantize
from mfq.quantize.nvq_tensor_codebook import (
    TensorCodebookTrainingConfig,
    train_tensor_codebook,
)


@pytest.mark.parametrize("spec", [NVQ2_E8, NVQ3_D4, NVQ1_L_T8_S3])
def test_tensor_codebook_training_keeps_a_legal_unique_table(spec):
    rng = np.random.default_rng(20260716 + spec.vector_size)
    train = rng.normal(0, 0.05, size=(4, 24)).astype(np.float32)
    validation = rng.normal(0, 0.05, size=(3, 24)).astype(np.float32)
    result = train_tensor_codebook(
        "synthetic.weight",
        train,
        validation,
        spec,
        TensorCodebookTrainingConfig(
            iterations=1,
            projection_candidates=4,
            quant_backend="cpu",
            group_chunk=8,
            search_steps=1,
            nvq1_l_anchor_multipliers=(0.75,),
            nvq1_l_refine_steps=0,
        ),
    )

    if spec is NVQ1_L_T8_S3:
        validate_ternary_codebook(result.trained_codebook)
    else:
        validate_codebook(spec, result.trained_codebook)
    assert result.history[-1]["sse"] <= result.history[0]["sse"] + 1e-8
    assert result.selected_custom == (
        result.trained_validation_sse < result.fixed_validation_sse
    )


def test_tensor_codebook_validation_gate_can_reject_overfit_table():
    rng = np.random.default_rng(20260720)
    train = rng.normal(0, 0.05, size=(4, 24)).astype(np.float32)
    validation = rng.normal(0, 0.05, size=(3, 24)).astype(np.float32)
    result = train_tensor_codebook(
        "synthetic.weight",
        train,
        validation,
        NVQ2_E8,
        TensorCodebookTrainingConfig(
            iterations=1,
            projection_candidates=4,
            quant_backend="cpu",
            group_chunk=8,
            search_steps=1,
            min_validation_improvement=0.99,
        ),
    )
    assert not result.selected_custom


def test_tensor_codebook_metrics_use_importance_weights():
    rng = np.random.default_rng(20260726)
    train = rng.normal(0, 0.05, size=(4, 24)).astype(np.float32)
    validation = rng.normal(0, 0.05, size=(3, 24)).astype(np.float32)
    importance = np.geomspace(0.1, 10.0, 24, dtype=np.float32)
    config = TensorCodebookTrainingConfig(
        iterations=0,
        projection_candidates=4,
        quant_backend="cpu",
        group_chunk=8,
        row_chunk=4,
        search_steps=1,
        initializations=("builtin",),
    )

    result = train_tensor_codebook(
        "synthetic.weight",
        train,
        validation,
        NVQ2_E8,
        config,
        train_importance=importance,
        validation_importance=importance,
    )

    encoded_train = quantize(
        train,
        NVQ2_E8,
        importance=importance,
        search_steps=1,
        group_chunk=8,
        scale_refine_steps=2,
    )
    encoded_validation = quantize(
        validation,
        NVQ2_E8,
        importance=importance,
        search_steps=1,
        group_chunk=8,
        scale_refine_steps=2,
    )
    train_sse = float(
        (importance[None, :] * np.square(train - dequantize(encoded_train))).sum(
            dtype=np.float64
        )
    )
    validation_sse = float(
        (
            importance[None, :]
            * np.square(validation - dequantize(encoded_validation))
        ).sum(dtype=np.float64)
    )

    assert result.history[0]["sse"] == pytest.approx(train_sse, rel=1e-6)
    assert result.fixed_validation_sse == pytest.approx(validation_sse, rel=1e-6)
