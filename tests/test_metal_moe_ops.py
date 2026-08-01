"""Apple-silicon tests for fused MoE routing and post-processing."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.kernels.metal.moe_ops import (  # noqa: E402
    add_shared_gate,
    apply_expert_scale,
    geglu_split,
    moe_topk,
    sqrtsoftplus_weights,
    swiglu_split,
    weighted_reduce,
    weighted_reduce_shared_gate,
)


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _topk_indices(scores: np.ndarray, count: int) -> np.ndarray:
    return np.argsort(-scores, axis=-1, kind="stable")[:, :count].astype(np.int32)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_moe_topk_full_softmax(dtype):
    logits = np.random.default_rng(101).normal(size=(7, 64)).astype(dtype)
    ids, weights = moe_topk(logits, 6)
    logits32 = logits.astype(np.float32)
    probabilities = np.exp(logits32 - np.max(logits32, axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected_ids = _topk_indices(probabilities, 6)
    expected_weights = np.take_along_axis(probabilities, expected_ids, axis=1)
    np.testing.assert_array_equal(_array(ids), expected_ids)
    np.testing.assert_allclose(
        _array(weights),
        expected_weights,
        rtol=3e-5,
        atol=3e-6,
    )


def test_moe_topk_delayed_softmax():
    logits = np.random.default_rng(102).normal(size=(3, 256)).astype(np.float32)
    ids, weights = moe_topk(logits, 8, delayed_softmax=True)
    expected_ids = _topk_indices(logits, 8)
    selected = np.take_along_axis(logits, expected_ids, axis=1)
    expected = np.exp(selected - selected.max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    np.testing.assert_array_equal(_array(ids), expected_ids)
    np.testing.assert_allclose(_array(weights), expected, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("mode", ["sigmoid", "sqrtsoftplus"])
def test_moe_topk_transforms_bias_normalize(mode: str):
    logits = np.random.default_rng(103).normal(0, 2, size=(5, 256)).astype(np.float16)
    bias = np.random.default_rng(104).normal(0, 0.05, size=(256,)).astype(np.float32)
    transformed = (
        1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        if mode == "sigmoid"
        else np.sqrt(np.logaddexp(logits.astype(np.float32), 0.0))
    )
    ids, weights = moe_topk(
        logits,
        6,
        use_sigmoid=mode == "sigmoid",
        use_sqrt_softplus=mode == "sqrtsoftplus",
        normalize=True,
        bias=bias,
        scale=1.5,
    )
    expected_ids = _topk_indices(transformed + bias, 6)
    expected = np.take_along_axis(transformed, expected_ids, axis=1)
    expected = expected / expected.sum(axis=-1, keepdims=True) * 1.5
    np.testing.assert_array_equal(_array(ids), expected_ids)
    np.testing.assert_allclose(_array(weights), expected, rtol=5e-5, atol=5e-6)


def test_sqrtsoftplus_hash_weights():
    logits = np.random.default_rng(105).normal(0, 2, size=(9, 256)).astype(np.float16)
    ids = np.random.default_rng(106).integers(
        0,
        256,
        size=(9, 6),
        dtype=np.int32,
    )
    actual = _array(sqrtsoftplus_weights(logits, ids, scale=1.5))
    transformed = np.sqrt(np.logaddexp(logits.astype(np.float32), 0.0))
    expected = np.take_along_axis(transformed, ids, axis=1)
    expected = expected / expected.sum(axis=-1, keepdims=True) * 1.5
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-6)


def test_weighted_reduce_and_fused_shared_gate():
    rng = np.random.default_rng(107)
    pairs = rng.normal(size=(11, 8, 37)).astype(np.float16)
    raw_weights = rng.normal(size=(11, 8)).astype(np.float32)
    weights = np.exp(raw_weights)
    weights /= weights.sum(axis=-1, keepdims=True)
    shared = rng.normal(size=(11, 37)).astype(np.float16)
    gate = rng.normal(size=(11, 1)).astype(np.float32)
    reduced = _array(weighted_reduce(pairs, weights))
    expected = np.sum(pairs.astype(np.float32) * weights[..., None], axis=1).astype(np.float16)
    np.testing.assert_array_equal(reduced, expected)

    actual_shared = _array(add_shared_gate(reduced, shared, gate))
    expected_shared = (
        reduced.astype(np.float32) + (1.0 / (1.0 + np.exp(-gate))) * shared.astype(np.float32)
    ).astype(np.float16)
    np.testing.assert_array_equal(actual_shared, expected_shared)
    np.testing.assert_array_equal(
        _array(weighted_reduce_shared_gate(pairs, weights, shared, gate)),
        actual_shared,
    )


def test_glu_splits_and_expert_scale():
    values = np.random.default_rng(108).normal(size=(3, 4, 66)).astype(np.float16)
    gate, up = np.split(values.astype(np.float32), 2, axis=-1)
    swiglu = (gate / (1.0 + np.exp(-gate)) * up).astype(np.float16)
    np.testing.assert_array_equal(_array(swiglu_split(values)), swiglu)
    inner = np.sqrt(2.0 / np.pi) * (gate + 0.044715 * gate**3)
    geglu = (0.5 * gate * (1.0 + np.tanh(inner)) * up).astype(np.float16)
    np.testing.assert_allclose(
        _array(geglu_split(values)),
        geglu,
        rtol=1e-3,
        atol=1e-3,
    )

    weights = np.random.default_rng(109).normal(size=(3, 4)).astype(np.float32)
    ids = np.asarray([[0, 2, 1, 3], [2, 2, 0, 1], [3, 0, 1, 2]], np.int32)
    scales = np.asarray([0.5, 1.0, 1.5, 2.0], np.float32)
    np.testing.assert_array_equal(
        _array(apply_expert_scale(weights, ids, scales)),
        weights * scales[ids],
    )
